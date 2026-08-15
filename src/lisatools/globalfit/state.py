"""Per-branch sampler-state subclasses used by the global fit."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from eryn.state import Branch as eryn_Branch
from eryn.state import State as eryn_State


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


def _scalar_or_none(value):
    """``value`` as an int if it is scalar-like, else ``None``.

    The main backend's flat kwargs merge can hand branch-keyed dicts (e.g.
    ``nleaves_max={branch: n}``) to a sub-backend reset; those are not this
    branch's dimensions and are treated as absent.
    """
    if value is None or isinstance(value, dict):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def make_cap_edges(band_edges, divisor: int):
    """Subdivide every sub-band into ``divisor`` equal-width CAP CELLS.

    The leaf-cap grid (user design 2026-08-15). Sub-band widths are set by
    what the likelihood engine can run concurrently; the scale at which
    sources actually get confused is the POSTERIOR width, which is much
    narrower. So the caps move off the band grid onto a finer one, while
    scheduling / units / buffers / tempering / band shutoff all stay on the
    band grid.

    Every cap cell is CONTAINED in exactly one sub-band (band ``b`` owns
    cells ``b*K ... b*K + K - 1``). That containment is what keeps the
    bookkeeping and the HDF5 storage simple and makes the construction work
    unchanged under both band-edge modes (``uniform`` and the free-floating
    ``get_n`` grid).

    ``divisor == 1`` returns a copy of ``band_edges`` itself, so the cap
    grid IS the band grid and every downstream cap computation reduces
    bit-identically to the pre-2026-08-15 per-band behaviour.

    Args:
        band_edges: 1D ascending array of sub-band edges (Hz).
        divisor: Number of cap cells per sub-band (``K >= 1``).

    Returns:
        ``np.ndarray`` of length ``K * num_bands + 1``.
    """
    be = np.asarray(band_edges, dtype=float)
    k = max(1, int(divisor))
    if k == 1:
        return be.copy()
    lo = be[:-1]
    step = (be[1:] - be[:-1]) / k
    inner = lo[:, None] + np.arange(k)[None, :] * step[:, None]
    return np.concatenate([inner.ravel(), be[-1:]])


def cap_divisor_from_edges(band_edges, cap_edges) -> int:
    """Infer the cap divisor ``K`` from a stored (band_edges, cap_edges) pair."""
    nb = len(np.asarray(band_edges)) - 1
    nc = len(np.asarray(cap_edges)) - 1
    if nb <= 0 or nc <= 0 or nc % nb != 0:
        raise ValueError(
            f"cap grid ({nc} cells) is not an integer refinement of the band "
            f"grid ({nb} bands)."
        )
    return nc // nb


def ensure_cap_cell_fields(band_info: dict, num_cells: int) -> None:
    """Backfill the per-CAP-CELL progressive leaf-cap arrays on ``band_info``.

    The cap-cell twins of the ``band_*`` cap arrays (see
    :func:`ensure_leaf_cap_fields`). These are the ARRAYS THE CAP MACHINERY
    ACTUALLY READS as of 2026-08-15; the ``band_leaf_cap`` family stays
    written (as the per-band max over its cells) purely so the monitor and
    the existing diagnostics keep working.

    - ``cap_cell_leaf_cap``: max alive leaves allowed per cap cell at EVERY
      temperature. ``-1`` = disarmed sentinel.
    - ``cap_cell_iters``: RJ iterations spent at the current cell cap.
    - ``cap_cell_best_ll``: running max of the per-cell cold-walker
      likelihood statistic at the current cap (reset on each increment).
    - ``cap_cell_cold_ll``: ``(nwalkers, num_cells)`` per-cold-walker
      statistic, refreshed every step so the decision is auditable.

    DIVISOR-1 SHORT CIRCUIT: when the cap grid is the band grid there is
    nothing to allocate -- the move reads the ``band_*`` arrays directly, so
    divisor 1 is bit-identical to the pre-2026-08-15 code AND keeps
    resuming stores written before the cap grid existed.
    """
    if int(num_cells) == int(band_info.get("num_bands", num_cells)):
        return
    band_info.setdefault("cap_cell_leaf_cap", np.full(num_cells, -1, dtype=int))
    band_info.setdefault("cap_cell_iters", np.zeros(num_cells, dtype=int))
    band_info.setdefault("cap_cell_best_ll", np.full(num_cells, -np.inf))
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if _nw:
        band_info.setdefault(
            "cap_cell_cold_ll", np.full((_nw, num_cells), -np.inf)
        )


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
    # ``band_cold_ll``: the per-band residual ll of EVERY cold-chain walker,
    # refreshed each step. ``band_best_ll`` is the running max over this;
    # keeping the full per-walker array makes the cap decision auditable
    # after the fact (and is what the lnL-improvement criterion is judged
    # from). ``(nwalkers, num_bands)``.
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if _nw:
        band_info.setdefault(
            "band_cold_ll", np.full((_nw, num_bands), -np.inf)
        )


class ModuleSubState(eryn_State):
    """Base class for the per-module (per-branch) global-fit sub-states.

    A sub-state owns the module-specific sampler information for one branch
    (temperature ladders, per-band counters, ...). Its storage contract is
    "the sub-state IS the schema": the matching
    :class:`~lisatools.globalfit.hdfbackend.ModuleSubBackend` derives every
    HDF5 dataset from the arrays the sub-state allocates -- there is no
    separate schema layer. Subclasses implement/extend small setup methods
    and name lists:

    - :meth:`storage_arrays`: ``{on-disk name: array}`` written every saved
      iteration.
    - :meth:`static_arrays`: ``{on-disk name: array}`` written once at
      backend reset (e.g. ``band_edges``).
    - :meth:`storage_attrs`: ``{name: scalar}`` written as HDF5 group attrs
      at reset (e.g. ``num_bands``).
    - ``static_names`` / ``dim_attr_names``: name lists the backend uses to
      read ``reset_kwargs`` back from an existing file.
    - :meth:`make_template`: allocate a zeroed instance from dimension
      kwargs; the backend's ``reset`` shapes every dataset from it.
    - :meth:`from_stored`: rebuild an instance from one stored iteration.
    - :meth:`reset_delta_counters`: zero per-iteration-delta counters after
      each save (default: nothing to zero).
    """

    static_names: tuple = ()
    dim_attr_names: tuple = ("ntemps", "nwalkers", "nleaves_max", "ndim")

    #: arrays carried by the tempered block (allocated by
    #: :meth:`initialize_tempered`, copied by the copy path; missing names
    #: are skipped so subclasses with their own ladder storage compose)
    tempered_array_names: tuple = (
        "coords",
        "inds",
        "log_like",
        "log_prior",
        "betas",
        "d_h",
        "h_h",
        "in_model_proposed",
        "in_model_accepted",
        "rj_proposed",
        "rj_accepted",
        "swaps_proposed",
        "swaps_accepted",
    )
    #: name of the ladder array this class stores. The base allocates and
    #: persists a flat ``betas (ntemps,)``; subclasses with richer ladders
    #: (``band_temps``, ``betas_all``) override this and skip the flat one.
    betas_attr_name: str = "betas"
    #: counters zeroed after every save (per-iteration deltas)
    delta_counter_names: tuple = (
        "in_model_proposed",
        "in_model_accepted",
        "rj_proposed",
        "rj_accepted",
        "swaps_proposed",
        "swaps_accepted",
    )
    #: arrays stored with the backend's float dtype for continuity with the
    #: pre-rework files (everything else keeps its in-memory dtype)
    legacy_dtype_names: tuple = ()

    def __init__(self, possible_state=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            self._copy_tempered_from(possible_state, deepcopy if copy else return_x)

    # ------------------------------------------------------------------
    # Tempered ensemble block: the module's full (ntemps, nwalkers, ...)
    # ensemble, owned by the sub-state
    # ------------------------------------------------------------------

    @property
    def tempered_initialized(self) -> bool:
        """Whether the tempered ensemble block has been allocated."""
        return getattr(self, "_tempered_initialized", False)

    def _ll_shape(self):
        return (self.ntemps, self.nwalkers)

    def _counter_shape(self):
        return (self.ntemps,)

    def _swaps_shape(self):
        return (max(self.ntemps - 1, 0),)

    def initialize_tempered(self, ntemps, nwalkers, nleaves_max, ndim, coords=None, inds=None):
        """Allocate (or validate + refill) the module's tempered ensemble.

        Idempotent: on an already-initialized sub-state the geometry must
        match exactly and any provided ``coords`` / ``inds`` are copied in.
        """
        dims = (int(ntemps), int(nwalkers), int(nleaves_max), int(ndim))
        if self.tempered_initialized:
            current = (self.ntemps, self.nwalkers, self.nleaves_max, self.ndim)
            if current != dims:
                raise ValueError(
                    f"tempered geometry mismatch: sub-state has "
                    f"(ntemps, nwalkers, nleaves_max, ndim)={current}, "
                    f"initialize_tempered got {dims}."
                )
            if coords is not None:
                self.coords[:] = coords
            if inds is not None:
                self.inds[:] = inds
            return

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = dims
        shape = dims[:1] + dims[1:2] + (self.nleaves_max, self.ndim)
        if coords is not None:
            coords = np.array(coords, dtype=float, copy=True)
            if coords.shape != shape:
                raise ValueError(
                    f"coords shape {coords.shape} does not match tempered "
                    f"geometry {shape}."
                )
            self.coords = coords
        else:
            self.coords = np.zeros(shape)
        if inds is not None:
            inds = np.array(inds, dtype=bool, copy=True)
            if inds.shape != shape[:-1]:
                raise ValueError(
                    f"inds shape {inds.shape} does not match tempered "
                    f"geometry {shape[:-1]}."
                )
            self.inds = inds
        else:
            self.inds = np.ones(shape[:-1], dtype=bool)

        self.log_like = np.zeros(self._ll_shape())
        self.log_prior = np.zeros(self._ll_shape())
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is None:
            self.betas = np.ones(self.ntemps)
        # per-leaf cold-chain inner products <d|h> and <h|h>, recorded by the
        # in-model moves at the end of their repeat blocks (NaN = dead leaf
        # or not recorded this iteration)
        self.d_h = np.full((self.nwalkers, self.nleaves_max), np.nan)
        self.h_h = np.full((self.nwalkers, self.nleaves_max), np.nan)
        self.in_model_proposed = np.zeros(self._counter_shape(), dtype=int)
        self.in_model_accepted = np.zeros(self._counter_shape(), dtype=int)
        self.rj_proposed = np.zeros(self._counter_shape(), dtype=int)
        self.rj_accepted = np.zeros(self._counter_shape(), dtype=int)
        self.swaps_proposed = np.zeros(self._swaps_shape(), dtype=int)
        self.swaps_accepted = np.zeros(self._swaps_shape(), dtype=int)
        self._tempered_initialized = True

    def _copy_tempered_from(self, other, dc):
        """Copy the tempered block (if any) from ``other`` using copier ``dc``."""
        if not getattr(other, "tempered_initialized", False):
            return
        for name in ("ntemps", "nwalkers", "nleaves_max", "ndim"):
            setattr(self, name, getattr(other, name))
        for name in self.tempered_array_names:
            if hasattr(other, name):
                setattr(self, name, dc(getattr(other, name)))
        self._tempered_initialized = True

    @property
    def branch(self) -> eryn_Branch:
        """An eryn ``Branch`` VIEW over this sub-state's coords/inds (shared memory)."""
        return eryn_Branch(self.coords, inds=self.inds)

    def sync_cold_row(self, main_state, branch_name: str):
        """Write this sub-state's cold row (temp 0) into the main state."""
        main_branch = main_state.branches[branch_name]
        main_branch.coords[0] = self.coords[0]
        main_branch.inds[0] = self.inds[0]

    def check_cold_row(self, main_state, branch_name: str):
        """Verify the main state's cold row matches this sub-state's row 0.

        Raises:
            ValueError: labeled description of the mismatch (inds or coords).
        """
        main_branch = main_state.branches[branch_name]
        if not np.array_equal(main_branch.inds[0], self.inds[0]):
            n_bad = int(np.sum(main_branch.inds[0] != self.inds[0]))
            raise ValueError(
                f"[{branch_name}] cold-chain inds mismatch between the main "
                f"state and its sub-state ({n_bad} differing leaf slots). "
                "A move updated one representation without the other."
            )
        main_alive = main_branch.coords[0][main_branch.inds[0]]
        sub_alive = self.coords[0][self.inds[0]]
        if not np.array_equal(main_alive, sub_alive):
            n_bad = int(np.sum(np.any(main_alive != sub_alive, axis=-1)))
            raise ValueError(
                f"[{branch_name}] cold-chain coords mismatch between the "
                f"main state and its sub-state ({n_bad} of {len(sub_alive)} "
                "alive leaves differ). A move updated one representation "
                "without the other."
            )

    def pull_from_main(self, main_state, branch_name: str):
        """Mirror the main state's full ensemble for this branch into the sub-state.

        Initializes the tempered block from the main branch on first use;
        afterwards copies coords/inds at every temperature (the Phase-2
        dual-representation sync).
        """
        main_branch = main_state.branches[branch_name]
        if not self.tempered_initialized:
            self.initialize_tempered(
                main_branch.ntemps,
                main_branch.nwalkers,
                main_branch.nleaves_max,
                main_branch.ndim,
                coords=main_branch.coords,
                inds=main_branch.inds,
            )
            return
        self.coords[:] = main_branch.coords
        self.inds[:] = main_branch.inds

    # ------------------------------------------------------------------
    # Storage contract
    # ------------------------------------------------------------------

    def tempered_storage_arrays(self) -> dict:
        """The standard tempered dict (``chain``/``inds``/logL/logP + counters)."""
        if not self.tempered_initialized:
            return {}
        out = {
            "chain": self.coords,
            "inds": self.inds,
            "log_like": self.log_like,
            "log_prior": self.log_prior,
        }
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is not None:
            out["betas"] = self.betas
        for name in ("d_h", "h_h"):
            if getattr(self, name, None) is not None:
                out[name] = getattr(self, name)
        for name in self.delta_counter_names:
            out[name] = getattr(self, name)
        return out

    def _load_tempered_from_stored(self, arrays):
        """Fill the tempered block from one stored iteration (leading axis 1)."""
        if "chain" not in arrays:
            return
        coords = np.asarray(arrays["chain"][0])
        inds = np.asarray(arrays["inds"][0]).astype(bool)
        self.initialize_tempered(*coords.shape, coords=coords, inds=inds)
        # some branches (GB) store only chain/inds -- band_info carries the
        # rest of their tempering record
        for name in (
            "log_like",
            "log_prior",
            "betas",
            "d_h",
            "h_h",
        ) + self.delta_counter_names:
            if name in arrays and getattr(self, name, None) is not None:
                getattr(self, name)[:] = arrays[name][0]

    def storage_arrays(self) -> dict:
        """``{on-disk name: array}`` persisted every saved iteration."""
        return self.tempered_storage_arrays()

    def static_arrays(self) -> dict:
        """``{on-disk name: array}`` persisted once at backend reset."""
        return {}

    def storage_attrs(self) -> dict:
        """``{name: scalar}`` written as HDF5 group attributes at reset."""
        if not self.tempered_initialized:
            return {}
        return {
            "ntemps": self.ntemps,
            "nwalkers": self.nwalkers,
            "nleaves_max": self.nleaves_max,
            "ndim": self.ndim,
        }

    @classmethod
    def make_template(cls, nwalkers, ntemps, nleaves_max=None, ndim=None, **dims):
        """Allocate a zeroed instance from dimension kwargs (extras ignored)."""
        template = cls(None)
        if _scalar_or_none(nleaves_max) is None or _scalar_or_none(ndim) is None:
            raise ValueError("Must provide nleaves_max and ndim kwargs.")
        template.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        """Rebuild an instance from one stored iteration's arrays."""
        instance = cls(None)
        instance._load_tempered_from_stored(arrays)
        return instance

    def reset_delta_counters(self):
        """Zero per-iteration-delta counters (called after each save)."""
        if self.tempered_initialized:
            for name in self.delta_counter_names:
                getattr(self, name)[:] = 0

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        out = {name: value for name, value in self.storage_attrs().items()}
        out.update(self.static_arrays())
        return out


class GBState(ModuleSubState):
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
            self._copy_tempered_from(possible_state, dc)
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

    def initialize_band_information(
        self, nwalkers, ntemps, band_edges, band_temps, cap_edges=None
    ):
        """Allocate the band-info dict with zeroed counters.

        Args:
            nwalkers: Number of MCMC walkers.
            ntemps: Number of temperatures in the ladder.
            band_edges: 1D array of frequency-band edges.
            band_temps: ``(num_bands, ntemps)`` array of inverse temperatures.
            cap_edges: Optional 1D array of LEAF-CAP cell edges, a refinement
                of ``band_edges`` (see :func:`make_cap_edges`). ``None``
                (default) means the cap grid IS the band grid, i.e. divisor 1
                and the pre-2026-08-15 behaviour.
        """
        if cap_edges is None:
            cap_edges = np.asarray(band_edges, dtype=float).copy()

        if not self.band_initialized:
            band_info = {}
            band_info["nwalkers"], band_info["ntemps"], band_info["band_edges"] = (
                nwalkers,
                ntemps,
                band_edges,
            )
            band_info["cap_edges"] = np.asarray(cap_edges, dtype=float)

            band_info["num_bands"] = len(band_info["band_edges"]) - 1
            band_info["num_cap_cells"] = len(band_info["cap_edges"]) - 1

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
            ensure_cap_cell_fields(band_info, band_info["num_cap_cells"])
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
                "band_cold_ll": 2,
                "cap_cell_leaf_cap": 1, "cap_cell_iters": 1,
                "cap_cell_best_ll": 1, "cap_cell_cold_ll": 2,
            }
            for _key, _nd in _bare_ndim.items():
                _arr = bi.get(_key)
                if isinstance(_arr, np.ndarray) and _arr.ndim == _nd + 1:
                    bi[_key] = _arr[-1]
            bi.setdefault("num_bands", len(bi["band_edges"]) - 1)
            # Stores written before the cap grid existed carry no
            # ``cap_edges``: their cap grid IS the band grid (divisor 1),
            # which is exactly what the guard below then compares against.
            bi.setdefault("cap_edges", np.asarray(bi["band_edges"], dtype=float).copy())
            bi["num_cap_cells"] = len(bi["cap_edges"]) - 1
            bi.setdefault("ntemps", int(bi["band_temps"].shape[-1]))
            bi.setdefault("nwalkers", int(bi["band_num_binaries"].shape[-2]))
            ensure_leaf_cap_fields(bi, bi["num_bands"])
            ensure_cap_cell_fields(bi, bi["num_cap_cells"])
            assert nwalkers == bi["nwalkers"]
            assert ntemps == bi["ntemps"]
            # Band-grid geometry check. Explicit (the old bare
            # ``assert np.all(==)`` compared unequal-length arrays to a
            # scalar False with a DeprecationWarning) and tolerant to float
            # round-off (a migrated store's edges may differ by ~1 ulp from
            # the settings-derived ones when layer_df was reconstructed).
            _cfg_edges = np.asarray(band_edges, dtype=float)
            _stored_edges = np.asarray(bi["band_edges"], dtype=float)
            if _cfg_edges.shape != _stored_edges.shape or not np.allclose(
                _cfg_edges, _stored_edges, rtol=1e-9, atol=0.0
            ):
                raise ValueError(
                    f"band grid mismatch: state stores "
                    f"{len(_stored_edges) - 1} sub-bands but the run config "
                    f"builds {len(_cfg_edges) - 1}. The band-edge knobs "
                    f"(GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT / "
                    f"GB_BAND_MIN_LAYERS / GB_SUBBAND_DIVISOR) changed, or "
                    f"the store needs scripts/fstat_proposal/"
                    f"migrate_gb_band_edges.py."
                )
            # Leaf-CAP grid check (user design 2026-08-15). The cap grid is
            # a refinement of the band grid by GB_CAP_DIVISOR; a store whose
            # cap grid disagrees with the configured divisor must refuse
            # loudly rather than silently resume a run whose per-cell cap
            # state means something else.
            _cfg_cap = np.asarray(cap_edges, dtype=float)
            _stored_cap = np.asarray(bi["cap_edges"], dtype=float)
            if _cfg_cap.shape != _stored_cap.shape or not np.allclose(
                _cfg_cap, _stored_cap, rtol=1e-9, atol=0.0
            ):
                _k_cfg = (len(_cfg_cap) - 1) / max(len(_cfg_edges) - 1, 1)
                _k_store = (len(_stored_cap) - 1) / max(len(_stored_edges) - 1, 1)
                raise ValueError(
                    f"leaf-cap grid mismatch: the state stores "
                    f"{len(_stored_cap) - 1} cap cells (divisor ~{_k_store:g}) "
                    f"but the run config builds {len(_cfg_cap) - 1} "
                    f"(GB_CAP_DIVISOR ~{_k_cfg:g}). Either restore the old "
                    f"GB_CAP_DIVISOR / GBSettings.cap_divisor, or migrate the "
                    f"store with scripts/fstat_proposal/migrate_gb_cap_grid.py "
                    f"(which splits each band's stored cap state into its "
                    f"children, inheriting cap + min-iters counters)."
                )

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

    # ------------------------------------------------------------------
    # ModuleSubState storage contract
    # ------------------------------------------------------------------

    static_names = ("band_edges", "cap_edges")
    dim_attr_names = (
        "num_bands", "num_cap_cells", "ntemps", "nwalkers", "nleaves_max", "ndim",
    )
    # GB's ladder is per band (band_info["band_temps"]); no flat betas
    betas_attr_name = "band_temps"
    #: all band arrays keep the backend float dtype (pre-rework layout)
    legacy_dtype_names = (
        "band_edges",
        "band_temps",
        "band_swaps_proposed",
        "band_swaps_accepted",
        "band_num_proposed",
        "band_num_accepted",
        "band_num_proposed_rj",
        "band_num_accepted_rj",
        "band_num_binaries",
        "band_leaf_cap",
        "band_cap_iters",
        "band_best_ll",
        "band_cold_ll",
        "cap_edges",
        "cap_cell_leaf_cap",
        "cap_cell_iters",
        "cap_cell_best_ll",
        "cap_cell_cold_ll",
    )

    #: band_info entries that are STATIC (written once at backend reset),
    #: not per-iteration storage arrays
    _static_band_info_names = ("band_edges", "cap_edges")

    def storage_arrays(self):
        """Every per-band array plus the tempered ``chain``/``inds``.

        The per-branch ``log_like``/``log_prior`` and base counters are
        omitted -- ``band_info`` carries the GB tempering record
        (``band_temps`` + ``band_num_*``) at per-band resolution.
        """
        out = {
            name: dat
            for name, dat in self.band_info.items()
            if isinstance(dat, np.ndarray)
            and name not in self._static_band_info_names
        }
        if self.tempered_initialized:
            out["chain"] = self.coords
            out["inds"] = self.inds
            for name in ("d_h", "h_h"):
                if getattr(self, name, None) is not None:
                    out[name] = getattr(self, name)
        return out

    def static_arrays(self):
        return {
            "band_edges": self.band_info["band_edges"],
            "cap_edges": self.band_info.get(
                "cap_edges", self.band_info["band_edges"]
            ),
        }

    def storage_attrs(self):
        out = dict(super().storage_attrs())
        out["num_bands"] = len(self.band_info["band_edges"]) - 1
        out["num_cap_cells"] = len(self.static_arrays()["cap_edges"]) - 1
        return out

    @classmethod
    def make_template(
        cls,
        nwalkers,
        ntemps,
        num_bands=None,
        band_edges=None,
        cap_edges=None,
        num_cap_cells=None,
        nleaves_max=None,
        ndim=None,
        **kwargs,
    ):
        if num_bands is None or band_edges is None:
            raise ValueError("Must provide num_bands and band_edges kwargs.")
        template = cls(None)
        template.initialize_band_information(
            nwalkers, ntemps, band_edges, np.zeros((num_bands, ntemps)),
            cap_edges=cap_edges,
        )
        if _scalar_or_none(nleaves_max) is not None and _scalar_or_none(ndim) is not None:
            template.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        # The stored band arrays keep their leading step axis; GBState's
        # initialize_band_information strips it rank-based on reload.
        band_info = {
            name: value
            for name, value in arrays.items()
            if name.startswith("band_") or name.startswith("cap_cell_")
        }
        band_info["band_edges"] = statics["band_edges"]
        # Files written before the cap grid existed have no ``cap_edges``
        # static: divisor 1, cap grid == band grid.
        band_info["cap_edges"] = statics.get("cap_edges", statics["band_edges"])
        band_info["initialized"] = True
        instance = cls(None, band_info=band_info)
        instance._load_tempered_from_stored(arrays)
        return instance

    def reset_delta_counters(self):
        self.reset_band_counters()


class PerLeafLadderState(ModuleSubState):
    """Shared base for sub-states carrying one temperature ladder per leaf.

    ``betas_all`` has shape ``(nleaves_max, ntemps)`` -- one independent
    ladder per source. Concrete classes set ``branch_name`` and the legacy
    per-branch leaf-count attribute name (``num_mbhs`` / ``num_emris`` /
    ``num_sobbhs``) via ``leaf_count_name``.

    Args:
        possible_state: Existing instance of the same class or coords-like
            dict.
        betas_all: Optional ``(nleaves_max, ntemps)`` array of inverse
            temperatures, one row per leaf.
        copy: If ``True``, deep-copy data from ``possible_state``.
    """

    branch_name: str = None
    leaf_count_name: str = None
    remove_kwargs = ["betas_all"]

    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
            self._set_leaf_count(getattr(possible_state, self.leaf_count_name))
            self._copy_tempered_from(possible_state, dc)
        else:
            self.betas_all = betas_all
            if possible_state is None:
                # HDF warm-start: from_stored passes possible_state=None and
                # only betas_all (nleaves, ntemps) — its second-to-last axis
                # is the leaf count. The coords live in the main GFState, so
                # there is no branch to index here.
                self._set_leaf_count(
                    betas_all.shape[-2] if betas_all is not None else 20
                )
            else:
                self._set_leaf_count(
                    branch_nleaves_max(possible_state, self.branch_name)
                )

    def _set_leaf_count(self, n):
        setattr(self, self.leaf_count_name, int(n))

    @property
    def num_leaves(self):
        """Leaf count under its generic name (aliases ``num_mbhs`` etc.)."""
        return getattr(self, self.leaf_count_name)

    # ------------------------------------------------------------------
    # ModuleSubState storage contract
    # ------------------------------------------------------------------

    legacy_dtype_names = ("betas_all",)
    # the ladder is per leaf (betas_all); no flat betas
    betas_attr_name = "betas_all"

    # per-leaf resolution: each leaf carries its own ladder, likelihood
    # rows, and counters
    def _ll_shape(self):
        return (self.nleaves_max, self.ntemps, self.nwalkers)

    def _counter_shape(self):
        return (self.nleaves_max, self.ntemps)

    def _swaps_shape(self):
        return (self.nleaves_max, max(self.ntemps - 1, 0))

    def storage_arrays(self):
        out = {"betas_all": self.betas_all}
        out.update(self.tempered_storage_arrays())
        return out

    def storage_attrs(self):
        out = dict(super().storage_attrs())
        out[self.leaf_count_name] = self.num_leaves
        return out

    @classmethod
    def make_template(cls, nwalkers, ntemps, nleaves_max=None, ndim=None, **dims):
        num_leaves = _scalar_or_none(dims.get(cls.leaf_count_name))
        if num_leaves is None:
            num_leaves = _scalar_or_none(nleaves_max)
        if num_leaves is None:
            raise ValueError(f"Must provide {cls.leaf_count_name} kwarg.")
        template = cls(None, betas_all=np.zeros((num_leaves, ntemps)))
        if _scalar_or_none(ndim) is not None:
            template.initialize_tempered(ntemps, nwalkers, num_leaves, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        # [0] squeezes the (single) iteration axis to the live
        # (nleaves, ntemps) shape.
        instance = cls(None, betas_all=arrays["betas_all"][0])
        instance._load_tempered_from_stored(arrays)
        return instance


class MBHState(PerLeafLadderState):
    """Massive black-hole binary sampler state with per-leaf temperature ladder."""

    branch_name = "mbh"
    leaf_count_name = "num_mbhs"
    dim_attr_names = ("num_mbhs", "ntemps", "nwalkers", "nleaves_max", "ndim")


class EMRIState(PerLeafLadderState):
    """Extreme mass-ratio inspiral sampler state with per-leaf temperature ladder."""

    branch_name = "emri"
    leaf_count_name = "num_emris"
    dim_attr_names = ("num_emris", "ntemps", "nwalkers", "nleaves_max", "ndim")


class SOBBHState(PerLeafLadderState):
    """Stellar-origin BBH (SOBBH) sampler state with per-leaf temperature ladder.

    Mirrors :class:`EMRIState` — one row of ``betas_all`` per SOBBH leaf so each
    source carries its own tempering ladder.
    """

    branch_name = "sobbh"
    leaf_count_name = "num_sobbhs"
    dim_attr_names = ("num_sobbhs", "ntemps", "nwalkers", "nleaves_max", "ndim")


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
