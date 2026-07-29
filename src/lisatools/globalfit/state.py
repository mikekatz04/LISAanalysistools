"""Per-branch sampler-state subclasses used by the global fit."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from eryn.state import Branch as eryn_Branch
from eryn.state import State as eryn_State


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


def branch_nleaves_max(possible_state, name: str) -> int:
    """``nleaves_max`` for branch ``name`` from a coords-like dict OR an eryn ``State``.

    The per-branch sub-state constructors historically assumed a
    ``{branch: coords}`` dict input, but on HDF reload
    (``GFHDFBackend.get_a_sample``) they receive a plain
    :class:`eryn.state.State`, which is not subscriptable. Both carry the
    ``(ntemps, nwalkers, nleaves_max, ndim)`` shape; dispatch on which one
    arrived.
    """
    branches = getattr(possible_state, "branches", None)
    if branches is not None:
        return int(branches[name].shape[-2])
    return int(possible_state[name].shape[-2])


def ensure_leaf_cap_fields(band_info: dict, num_bands: int) -> None:
    """Backfill the per-band progressive leaf-cap arrays on ``band_info``.

    Three ``(num_bands,)`` arrays drive the search-mode leaf cap
    (see ``GBSpecialBase._update_band_leaf_caps``):

    - ``band_leaf_cap``: max alive leaves allowed per band at EVERY
      temperature. ``-1`` = cap disarmed (the fresh-state sentinel; the
      first cap-enabled RJ move arms it to its ``leaf_cap_start``).
    - ``band_cap_iters``: RJ iterations spent at the current cap.
    - ``band_best_ll``: running max of the per-band cold-walker residual
      ll at the current cap (reset to ``-inf`` on each increment).

    Kept OUT of ``band_info_keys`` so band-info dicts loaded from HDF5
    files written before this feature still pass the setter's required-key
    check; this backfill runs from both branches of
    :meth:`GBState.initialize_band_information`.
    """
    band_info.setdefault("band_leaf_cap", np.full(num_bands, -1, dtype=int))
    band_info.setdefault("band_cap_iters", np.zeros(num_bands, dtype=int))
    band_info.setdefault("band_best_ll", np.full(num_bands, -np.inf))


class GBState(eryn_State):
    """Galactic-binary (GB) sampler state with per-band bookkeeping.

    Tracks per-band temperature ladders, swap counters, and binary-count
    arrays that the GB special moves use to drive the band-temperature
    sampler.

    Args:
        possible_state: Existing :class:`GBState` or a state-like object to
            initialize from. When it is already a :class:`GBState`, band info
            is copied over.
        band_info: Optional pre-built band-information dict.
        copy: If ``True``, deep-copy the band info from ``possible_state``.
    """

    # copy this still for each. At general hdf5 function to deal with these setups rather than specific
    @property
    def band_initialized(self):
        """Whether band tracking has been initialized for this state."""
        if hasattr(self, "band_info") and "initialized" in self.band_info:
            return self.band_info["initialized"]
        else:
            return False

    def __init__(self, possible_state, band_info=None, copy=False, **kwargs):

        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            if possible_state.band_initialized and hasattr(possible_state, "band_info"):
                self.band_info = dc(possible_state.band_info)
        elif band_info is not None:
            self.band_info = band_info

    @property
    def band_info_keys(self):
        """List of required keys for the :attr:`band_info` dict."""
        return [
            "initialized",
            "band_edges",
            "band_temps",
            "band_swaps_proposed",
            "band_swaps_accepted",
            "band_num_proposed",
            "band_num_accepted",
            "band_num_proposed_rj",
            "band_num_accepted_rj",
            "band_num_binaries",
        ]

    @property
    def band_info(self):
        """Dict holding per-band counters, temperatures, and edges."""
        return self._band_info

    @band_info.setter
    def band_info(self, band_info):
        assert isinstance(band_info, dict)
        for key in self.band_info_keys:
            if key not in band_info and key != "initialized":
                raise ValueError(f"Missing required key: {key}, for band information.")
        self._band_info = band_info
        self._band_info["initialized"] = True

    def initialize_band_information(self, nwalkers, ntemps, band_edges, band_temps):
        """Allocate the band-info dict with zeroed counters.

        Args:
            nwalkers: Number of MCMC walkers.
            ntemps: Number of temperatures in the ladder.
            band_edges: 1D array of frequency-band edges.
            band_temps: ``(num_bands, ntemps)`` array of inverse temperatures.
        """

        if not self.band_initialized:
            band_info = {}
            band_info["nwalkers"], band_info["ntemps"], band_info["band_edges"] = (
                nwalkers,
                ntemps,
                band_edges,
            )

            band_info["num_bands"] = len(band_info["band_edges"]) - 1

            assert band_temps.shape == (band_info["num_bands"], band_info["ntemps"])
            band_info["band_temps"] = band_temps

            band_info["band_swaps_proposed"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"] - 1), dtype=int
            )
            band_info["band_swaps_accepted"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"] - 1), dtype=int
            )

            band_info["band_num_proposed"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )
            band_info["band_num_accepted"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )

            band_info["band_num_proposed_rj"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )
            band_info["band_num_accepted_rj"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )

            band_info["band_num_binaries"] = np.zeros(
                (band_info["ntemps"], band_info["nwalkers"], band_info["num_bands"]),
                dtype=int,
            )
            ensure_leaf_cap_fields(band_info, band_info["num_bands"])
            band_info["initialized"] = True
            self.band_info = band_info

        else:
            # already initialized: validate the geometry is unchanged.
            # band_info dicts that round-tripped through the HDF backend
            # (GBHDFBackend stores only the ``band_info_keys`` arrays)
            # lack the nwalkers/ntemps/num_bands scalars -- backfill them
            # from the array shapes before validating.
            bi = self.band_info
            # Arrays loaded through GBHDFBackend.get_band_info keep the
            # backend's leading step axis; in-run consumers index the bare
            # per-iteration shapes, so strip it. Rank-based (not
            # shape[0]==1) so genuine single-band/single-temp axes survive.
            _bare_ndim = {
                "band_temps": 2, "band_num_proposed": 2, "band_num_accepted": 2,
                "band_num_proposed_rj": 2, "band_num_accepted_rj": 2,
                "band_swaps_proposed": 2, "band_swaps_accepted": 2,
                "band_num_binaries": 3, "band_leaf_cap": 1,
                "band_cap_iters": 1, "band_best_ll": 1,
            }
            for _key, _nd in _bare_ndim.items():
                _arr = bi.get(_key)
                if isinstance(_arr, np.ndarray) and _arr.ndim == _nd + 1:
                    bi[_key] = _arr[-1]
            bi.setdefault("num_bands", len(bi["band_edges"]) - 1)
            bi.setdefault("ntemps", int(bi["band_temps"].shape[-1]))
            bi.setdefault("nwalkers", int(bi["band_num_binaries"].shape[-2]))
            ensure_leaf_cap_fields(bi, bi["num_bands"])
            assert nwalkers == bi["nwalkers"]
            assert ntemps == bi["ntemps"]
            assert np.all(band_edges == bi["band_edges"])

    def update_band_information(
        self,
        band_temps,
        band_num_proposed,
        band_num_accepted,
        band_swaps_proposed,
        band_swaps_accepted,
        band_num_binaries,
        is_rj,
    ):
        """Accumulate one iteration's worth of band counters.

        Args:
            band_temps: New ``(num_bands, ntemps)`` temperature ladder.
            band_num_proposed: ``(num_bands, ntemps)`` proposal counts.
            band_num_accepted: ``(num_bands, ntemps)`` acceptance counts.
            band_swaps_proposed: ``(num_bands, ntemps - 1)`` swap proposals.
            band_swaps_accepted: ``(num_bands, ntemps - 1)`` swap acceptances.
            band_num_binaries: ``(ntemps, nwalkers, num_bands)`` binary count.
            is_rj: ``True`` to credit reversible-jump counters, otherwise
                in-model counters.
        """
        self.band_info["band_temps"][:] = band_temps
        self.band_info["band_num_binaries"][:] = band_num_binaries

        if not is_rj:
            self.band_info["band_num_proposed"] += band_num_proposed
            self.band_info["band_num_accepted"] += band_num_accepted
        else:
            self.band_info["band_num_proposed_rj"] += band_num_proposed
            self.band_info["band_num_accepted_rj"] += band_num_accepted

        self.band_info["band_swaps_proposed"] += band_swaps_proposed
        self.band_info["band_swaps_accepted"] += band_swaps_accepted

    def accumulate_proposals(self, proposed, accepted, is_rj: bool) -> None:
        """Accumulate ``(num_bands, ntemps)`` proposal/acceptance counts into
        the RJ or in-model counter family."""
        if is_rj:
            self.band_info["band_num_proposed_rj"] += proposed
            self.band_info["band_num_accepted_rj"] += accepted
        else:
            self.band_info["band_num_proposed"] += proposed
            self.band_info["band_num_accepted"] += accepted

    def accumulate_swaps(self, proposed, accepted) -> None:
        """Accumulate ``(num_bands, ntemps - 1)`` tempering swap counts."""
        self.band_info["band_swaps_proposed"] += proposed
        self.band_info["band_swaps_accepted"] += accepted

    def reset_band_counters(self):
        """Zero all per-band proposal/acceptance/swap counters."""
        self.band_info["band_num_proposed"][:] = 0
        self.band_info["band_num_accepted"][:] = 0
        self.band_info["band_num_proposed_rj"][:] = 0
        self.band_info["band_num_accepted_rj"][:] = 0
        self.band_info["band_swaps_proposed"][:] = 0
        self.band_info["band_swaps_accepted"][:] = 0

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        # TODO: this okay for future?
        return dict(
            num_bands=len(self.band_info["band_edges"]) - 1,
            band_edges=self.band_info["band_edges"],
        )


class MBHState(eryn_State):
    """Massive black-hole binary sampler state with per-leaf temperature ladder.

    Args:
        possible_state: Existing :class:`MBHState` or coords-like dict.
        betas_all: Optional ``(num_mbhs, ntemps)`` array of inverse
            temperatures, one row per MBH leaf.
        copy: If ``True``, deep-copy data from ``possible_state``.
    """

    remove_kwargs = ["betas_all"]

    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
            self.num_mbhs = possible_state.num_mbhs
        else:
            self.betas_all = betas_all
            if possible_state is None:
                # HDF warm-start: get_a_sample passes possible_state=None and
                # only betas_all (num_mbhs, ntemps) — its second-to-last axis is
                # num_mbhs. The coords live in the main GFState, so there is no
                # branch to index here.
                self.num_mbhs = betas_all.shape[-2] if betas_all is not None else 20
            else:
                self.num_mbhs = branch_nleaves_max(possible_state, "mbh")

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        # TODO: this okay for future?
        return dict(
            num_mbhs=self.num_mbhs,
        )


class EMRIState(eryn_State):
    """Extreme mass-ratio inspiral sampler state with per-leaf temperature ladder."""

    remove_kwargs = ["betas_all"]

    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
            self.num_emris = possible_state.num_emris
        else:
            self.betas_all = betas_all
            if possible_state is None:
                # HDF warm-start: get_a_sample passes possible_state=None and
                # only betas_all (num_emris, ntemps) — its second-to-last axis is
                # num_emris. The coords live in the main GFState, so there is no
                # branch to index here.
                self.num_emris = betas_all.shape[-2] if betas_all is not None else 20
            else:
                self.num_emris = branch_nleaves_max(possible_state, "emri")

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        # TODO: this okay for future?
        return dict(num_emris=self.num_emris)  # self.betas_all.shape[0]


class SOBBHState(eryn_State):
    """Stellar-origin BBH (SOBBH) sampler state with per-leaf temperature ladder.

    Mirrors :class:`EMRIState` — one row of ``betas_all`` per SOBBH leaf so each
    source carries its own tempering ladder.
    """

    remove_kwargs = ["betas_all"]

    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
            self.num_sobbhs = possible_state.num_sobbhs
        else:
            self.betas_all = betas_all
            if possible_state is None:
                # HDF warm-start: get_a_sample passes possible_state=None and
                # only betas_all (num_sobbhs, ntemps) — its second-to-last axis
                # is num_sobbhs. The coords live in the main GFState, so there is
                # no branch to index here.
                self.num_sobbhs = betas_all.shape[-2] if betas_all is not None else 20
            else:
                self.num_sobbhs = branch_nleaves_max(possible_state, "sobbh")

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        return dict(num_sobbhs=self.num_sobbhs)


class GFState(eryn_State):
    """Composite global-fit state holding per-source-class sub-states.

    Wraps an :class:`eryn.state.State` with a dict mapping each branch name
    (``gb``, ``mbh``, ``emri``, ...) to an associated state subclass
    (e.g. :class:`GBState`, :class:`MBHState`).

    Args:
        possible_state: Either an existing :class:`GFState` to copy from or a
            coords-like input.
        is_eryn_state_input: When ``True``, treat ``possible_state`` as a
            plain :class:`eryn.state.State` rather than a :class:`GFState`.
        sub_state_bases: Mapping ``{branch_name: state_class}`` giving the
            sub-state class to instantiate for each branch.
    """

    # TODO: bandaid fix this
    def __init__(
        self,
        possible_state,
        *args,
        is_eryn_state_input: bool = False,
        sub_state_bases: dict = None,
        **kwargs,
    ):

        eryn_State.__init__(self, possible_state, *args, **kwargs)
        self.sub_states = {}
        if isinstance(possible_state, type(self)) and not is_eryn_state_input:
            self.sub_state_bases = possible_state.sub_state_bases
            for name in self.branches:
                sub_state_base = self.sub_state_bases.get(name, None)
                if sub_state_base is not None:
                    self.sub_states[name] = sub_state_base(
                        possible_state.sub_states[name], *args, **kwargs
                    )
                else:
                    self.sub_states[name] = None

        else:
            self.sub_state_bases = sub_state_bases

            for name in self.branches:
                if sub_state_bases is not None and sub_state_bases[name] is not None:
                    self.sub_states[name] = sub_state_bases[name](
                        possible_state,  # this is just coords in the first input
                        *args,
                        **kwargs,
                    )
                else:
                    self.sub_states[name] = None

        # elif sub_state_bases is None and is_eryn_state_input:
        #     raise ValueError

        # elif is_eryn_state_input:
        #     self.sub_state_bases = sub_state_bases
        #     for name in self.branches:
        #         sub_state_base = sub_state_bases.get(name, None)
        #         if sub_state_base is not None:
        #             self.sub_states[name] = sub_state_base(
        #                 None,
        #                 *args,
        #                 **kwargs
        #             )
        #         else:
        #             self.sub_states[name] = None


class AllGFBranchInfo:
    """Aggregate of two or more :class:`GFBranchInfo` instances.

    Combines per-branch metadata dicts (``ndims``, ``nleaves_max``,
    ``nleaves_min``, ``branch_state``, ``branch_backend``) so that a global
    fit can query branch info uniformly regardless of how many sources are
    in the model. Use the ``+`` operator on :class:`GFBranchInfo` /
    :class:`AllGFBranchInfo` to chain them together.
    """

    def __init__(self, branch_1, branch_2):

        for key in [
            "name",
            "ndims",
            "nleaves_max",
            "nleaves_min",
            "branch_state",
            "branch_backend",
        ]:
            if isinstance(branch_1, AllGFBranchInfo) and isinstance(branch_2, AllGFBranchInfo):
                if key == "name":
                    self.branch_names = branch_1.branch_names + branch_2.name
                    continue
                setattr(self, key, {**getattr(branch_1, key), **getattr(branch_2, key)})

            elif isinstance(branch_1, GFBranchInfo) and isinstance(branch_2, GFBranchInfo):
                if key == "name":
                    self.branch_names = [branch_1.name, branch_2.name]
                    continue
                setattr(
                    self,
                    key,
                    {
                        branch_1.name: getattr(branch_1, key),
                        branch_2.name: getattr(branch_2, key),
                    },
                )
            else:
                if not isinstance(branch_2, GFBranchInfo):
                    # switch so all branch is in position 1
                    tmp = branch_1
                    branch_1 = branch_2
                    branch_2 = tmp
                if key == "name":
                    self.branch_names = branch_1.branch_names + [branch_2.name]
                    continue
                setattr(
                    self,
                    key,
                    {**getattr(branch_1, key), branch_2.name: getattr(branch_2, key)},
                )

    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)

    @property
    def ndims(self):
        return self._ndims

    @ndims.setter
    def ndims(self, ndims):
        assert isinstance(ndims, dict)
        self._ndims = ndims

    @property
    def branch_names(self):
        return self._branch_names

    @branch_names.setter
    def branch_names(self, branch_names):
        assert isinstance(branch_names, list)
        self._branch_names = branch_names

    @property
    def nleaves_max(self):
        return self._nleaves_max

    @nleaves_max.setter
    def nleaves_max(self, nleaves_max):
        assert isinstance(nleaves_max, dict)
        self._nleaves_max = nleaves_max

    @property
    def nleaves_min(self):
        return self._nleaves_min

    @nleaves_min.setter
    def nleaves_min(self, nleaves_min):
        assert isinstance(nleaves_min, dict)
        self._nleaves_min = nleaves_min

    @property
    def branch_state(self):
        return self._branch_state

    @branch_state.setter
    def branch_state(self, branch_state):
        self._branch_state = branch_state

    @property
    def branch_backend(self):
        return self._branch_backend

    @branch_backend.setter
    def branch_backend(self, branch_backend):
        self._branch_backend = branch_backend


from eryn.backends import backend as eryn_Backend


@dataclass
class GFBranchInfo:
    """Metadata describing a single branch in the global fit.

    Args:
        name: Branch name (e.g. ``"gb"``, ``"mbh"``).
        ndims: Number of parameters per leaf.
        nleaves_max: Maximum allowed leaves on this branch.
        nleaves_min: Minimum allowed leaves on this branch.
        branch_state: Optional state class associated with this branch.
        branch_backend: Optional backend object associated with this branch.
    """

    name: str
    ndims: int
    nleaves_max: int
    nleaves_min: int
    branch_state: eryn_State = None
    branch_backend: eryn_Backend = None

    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)
