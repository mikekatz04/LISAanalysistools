"""Walker-block slicing of a :class:`~lisatools.globalfit.state.GFState`.

A slice is a self-contained ``GFState`` over ``B = w1 - w0`` walkers: every
walker-axis array cut by column (copies), ladders by value, sub-state delta
counters zeroed, and ``supplemental["walker_inds"]`` REMAPPED to ``0..B-1``
because a rank's ACA has exactly ``B`` rows. :func:`merge_state` writes a
slice's walker columns back and SUM-adds the counters; ladders and the
head's global ``walker_inds`` are never written back.

Used by the multi-rank fan-out: the head slices each compute rank's block
out of its full state, ships it with the command, and merges the rank's
returned slice (:mod:`lisatools.globalfit.moves.walkerfanout`); also by
``run.py`` and the GB moves.

Attributes:
    WALKER_INDS_KEY (str): ``"walker_inds"``, the supplemental key that is
        remapped on slicing and skipped on merging.
"""

from __future__ import annotations

import numpy as np
from eryn.state import BranchSupplemental

from ..state import GFState

WALKER_INDS_KEY = "walker_inds"


def _nwalkers_of(state) -> int:
    """Walker count of ``state`` (from its first branch)."""
    first = next(iter(state.branches.values()))
    return int(first.nwalkers)


def _walker_block(w0, w1, nwalkers):
    """Validate ``0 <= w0 < w1 <= nwalkers`` and return ``(w0, w1)`` as ints."""
    w0, w1 = int(w0), int(w1)
    if not (0 <= w0 < w1 <= int(nwalkers)):
        raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {nwalkers}).")
    return w0, w1


def _slice_supp(supp, w0, w1):
    """Column-slice every array a ``BranchSupplemental`` holds (copies)."""
    if supp is None:
        return None
    sliced = {
        name: np.array(values, copy=True)
        for name, values in supp[(slice(None), slice(w0, w1))].items()
    }
    base_shape = (int(supp.base_shape[0]), w1 - w0) + tuple(supp.base_shape[2:])
    return BranchSupplemental(sliced, base_shape=base_shape, copy=False)


def slice_state(full, w0, w1, *, sub_states="all"):
    """A ``GFState`` holding walkers ``[w0, w1)`` of ``full`` (all copies).

    Coordinates, ``inds``, branch and state supplementals, ``log_like``,
    ``log_prior`` and ``blobs`` are cut on the walker axis (axis 1); the
    ``betas`` ladder and ``sub_state_bases`` are copied whole;
    ``supplemental["walker_inds"]`` becomes ``0..B-1`` on every temperature.
    Only initialized tempered sub-states are sliced
    (:meth:`~lisatools.globalfit.state.ModuleSubState.slice_walkers`, counters
    start at zero).

    Args:
        full (GFState): The head's full state.
        w0 (int): First walker of the block.
        w1 (int): One past the last walker of the block.
        sub_states (str or list[str]): ``"all"`` slices every tempered
            sub-state; a list of branch names slices only those (the rest are
            ``None``); ``[]`` slices none. Default ``"all"``.

    Returns:
        GFState: The block's state with ``B = w1 - w0`` walkers.

    Raises:
        ValueError: When the block is outside ``[0, nwalkers)`` or
            ``sub_states`` names an unknown branch.
    """
    w0, w1 = _walker_block(w0, w1, _nwalkers_of(full))
    if sub_states != "all":
        unknown = sorted(set(sub_states) - set(full.branches))
        if unknown:
            raise ValueError(
                f"slice_state: unknown sub_states {unknown}; "
                f"valid branch names are {sorted(full.branches)}"
            )
    B = w1 - w0
    coords = {name: np.array(br.coords[:, w0:w1], copy=True) for name, br in full.branches.items()}
    inds = {name: np.array(br.inds[:, w0:w1], copy=True) for name, br in full.branches.items()}
    branch_supps = {
        name: _slice_supp(br.branch_supplemental, w0, w1) for name, br in full.branches.items()
    }
    supp = _slice_supp(full.supplemental, w0, w1)
    if supp is not None and WALKER_INDS_KEY in supp.holder:
        ntemps = int(supp.holder[WALKER_INDS_KEY].shape[0])
        supp.holder[WALKER_INDS_KEY] = np.tile(np.arange(B), (ntemps, 1))

    def _cols(arr):
        return None if arr is None else np.array(arr[:, w0:w1], copy=True)

    part = GFState(
        coords,
        inds=inds,
        branch_supplemental=branch_supps,
        supplemental=supp,
        log_like=_cols(full.log_like),
        log_prior=_cols(full.log_prior),
        betas=None if full.betas is None else np.array(full.betas, copy=True),
        blobs=_cols(full.blobs),
        sub_state_bases=None,
    )
    part.sub_state_bases = dict(getattr(full, "sub_state_bases", None) or {})
    wanted = set(full.branches) if sub_states == "all" else set(sub_states)
    full_subs = getattr(full, "sub_states", None) or {}
    part.sub_states = {}
    for name in full.branches:
        sub = full_subs.get(name)
        if name in wanted and sub is not None and getattr(sub, "tempered_initialized", False):
            part.sub_states[name] = sub.slice_walkers(w0, w1)
        else:
            part.sub_states[name] = None
    return part


def merge_state(full, part, w0, w1):
    """Write ``part``'s walker columns back into ``full[:, w0:w1]``.

    Sub-state delta counters are SUM-added; ladders and the head's global
    ``supplemental["walker_inds"]`` are untouched.

    Modifies ``full`` in place: coordinates, ``inds``, branch supplementals,
    ``log_like``, ``log_prior``, ``blobs``, the state supplemental (minus
    ``walker_inds``), and every sub-state present on both sides
    (:meth:`~lisatools.globalfit.state.ModuleSubState.merge_walkers`).

    Args:
        full (GFState): The head's full state (updated in place).
        part (GFState): A slice with ``w1 - w0`` walkers, e.g. from
            :func:`slice_state` after a rank ran a command on it.
        w0 (int): First walker of the block.
        w1 (int): One past the last walker of the block.

    Raises:
        ValueError: When the block is invalid or ``part`` has the wrong
            walker count.
    """
    w0, w1 = _walker_block(w0, w1, _nwalkers_of(full))
    if _nwalkers_of(part) != w1 - w0:
        raise ValueError(
            f"slice has {_nwalkers_of(part)} walkers but the block [{w0}, {w1}) has {w1 - w0}."
        )
    for name, br in full.branches.items():
        pbr = part.branches[name]
        br.coords[:, w0:w1] = pbr.coords
        br.inds[:, w0:w1] = pbr.inds
        if br.branch_supplemental is not None and pbr.branch_supplemental is not None:
            br.branch_supplemental[(slice(None), slice(w0, w1))] = pbr.branch_supplemental.holder
    for name in ("log_like", "log_prior", "blobs"):
        dst = getattr(full, name, None)
        src = getattr(part, name, None)
        if dst is not None and src is not None:
            dst[:, w0:w1] = src
    if full.supplemental is not None and part.supplemental is not None:
        payload = {k: v for k, v in part.supplemental.holder.items() if k != WALKER_INDS_KEY}
        full.supplemental[(slice(None), slice(w0, w1))] = payload
    full_subs = getattr(full, "sub_states", None) or {}
    part_subs = getattr(part, "sub_states", None) or {}
    for name, sub in full_subs.items():
        psub = part_subs.get(name)
        if sub is not None and psub is not None:
            sub.merge_walkers(psub, w0, w1)
