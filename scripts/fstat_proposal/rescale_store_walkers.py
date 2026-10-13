"""Clone a run store into a copy with ``factor`` x as many walkers.

PE scale-up from the chain (``--mode lag``, the default since 2026-10-07)
-------------------------------------------------------------------------
``--nwalkers N`` (a multiple of 4 and of the store's walker count ``nw``)
builds the new walkers from SAMPLES IN THE CHAIN: with ``m = N / nw``, new
walker ``k = j * nw + i`` is source walker ``(i + j) % nw`` lagged by ``j``
saved rows, for ``j = 0 .. m-1``. So at the last row (the state a resume
starts from) 4 -> 8 walkers takes the last 2 rows of the 4 chains, 4 -> 32
the last 8, and each block of ``nw`` walkers is rotated by one more slot
(8 walkers: 0 1 2 3 | 1 2 3 0) so the same source chain sits at a different
index in every block -- the layout that mixes best should the permuted
("fancy") swaps be turned on later (Mike 2026-10-07; they stay OFF,
GB_RUN_FANCY_TEMPERING=0). Every row-wise dataset (main chain/inds/
log_like/log_prior, every sub_backend chain, ladder, d_h/h_h and per-walker
band table) is mapped the same way, so each new walker is ONE consistent
past state of the whole model, and over the whole history walker ``k``
reads as its source chain delayed by ``j`` rows (rows before the start
repeat row 0). Walkers ``0 .. nw-1`` are the source chains unchanged.
Walker-free per-row data (betas, band temperatures, counters) is the row's
own; the walker-axis datasets with no step axis (``accepted``) take the
same rotation without a lag. Run the copy with the PE settings and
``NWALKERS=N``.

Why this exists: the walker-block layout gives each rank ``B = nwalkers /
n_compute`` walkers, and several things are only measurable (or only
CORRECT) at ``B > 1`` -- above all the GB/VGB permuted tempering swap,
whose walker permutation is over ``arange(B)`` and is therefore the
identity at one walker per rank. Going from 4 walkers to 8 on the same 4
GPUs turns that back on. Starting the 8-walker run from scratch would
throw away the fitted noise, the VGB state, the F-stat epoch cache and
every GB leaf the search has found; this script carries all of it over by
DUPLICATING the walker axis instead.

What it does
------------
1. Copies the run directory (``gb_fstat_fit/``, ``warmstart/``, logs,
   truth npz, telemetry CSVs -- everything) to a new directory.
2. Rebuilds the ``*_testing.h5`` store with every walker axis tiled
   ``factor`` times, and updates the ``nwalkers`` attributes.
3. Leaves behind the sidecars that encode the OLD walker count.

What it deliberately does NOT regenerate: the F-stat epoch cache and the
warm-start components. Neither has a walker axis -- the epoch grids are
fitted against one reference walker's residual and the warm-start
components are a population model -- so both are carried over untouched,
which is the entire point of cloning rather than restarting.

The duplication is a DEGENERATE START, stated plainly
--------------------------------------------------
Walker ``w`` of the copy is a bit-for-bit clone of walker ``w % nwalkers``
of the source. The ensemble therefore begins with ``factor`` identical
copies of every state. That is harmless for MCMC validity (walkers are
exchangeable, and the chains separate on the first propose because every
(temp, walker) cell draws its own proposal), and it is irrelevant to a
TIMING measurement, which is what this script was written for. It is NOT
a fresh independent ensemble: do not read the first few iterations of the
copy as evidence about mixing.

``--mode tile`` (the default) lays the copies out so that each RANK holds
distinct states: with 8 walkers over 4 ranks the blocks are (0,1), (2,3),
(0,1), (2,3), so every rank has two different walkers to swap between
from the first iteration. ``--mode repeat`` gives (0,0), (1,1), (2,2),
(3,3) instead -- each rank holds one state twice, and its swap moves
nothing until the two diverge. Use ``tile`` unless you have a reason.

Walker axes are NOT inferred from the shape
-------------------------------------------
They cannot be. ``sub_backend/mbh`` has ``nleaves_max == nwalkers == 4``,
so ``mbh/chain`` at ``(nsteps, 2, 4, 4, 11)`` has two axes of length 4 and
only one of them is the walker axis. Every dataset is therefore matched
against a FULL expected shape signature built from its group's own
``ntemps`` / ``nwalkers`` / ``nleaves_max`` / ``ndim`` attributes, and
anything that matches no signature while still carrying an axis of length
``nwalkers`` is a REFUSAL, not a guess -- see ``plan_rescale``.

Usage (dry run first -- nothing is written without ``--apply``)::

    python scripts/fstat_proposal/rescale_store_walkers.py \
        /shared/data/global_fit_output/gf_prod_6mo_v8_4gpu \
        /shared/data/global_fit_output/gf_prod_6mo_v8_4gpu_8w --factor 2
    ... same command with --apply

Stop the job first: an open writer holds the store and h5py blocks rather
than erroring when it cannot take the lock.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compact_gf_store import (  # noqa: E402  (path shim above)
    DEFAULT_BUFFER_BYTES,
    _copy_attrs,
    _row_count,
    iter_blocks,
    pids_holding,
    sidecar_paths,
)

MB = 1024 ** 2

#: Per-(walker, band) GB search tables, ``(nwalkers, num_bands or
#: num_cap_cells)`` per row: the search-stage record and the per-walker RJ
#: shutoff valve with its window. Mirrors lisatools.globalfit.state's
#: SEARCH_STAGE_FIELDS, SEARCH_SHUTOFF_FIELDS[0] and
#: SEARCH_SHUTOFF_WINDOW_FIELDS (a test keeps the two in step; this script
#: stays importable without the package).
PER_WALKER_BAND_TABLES = (
    "band_stage_w", "band_stage_occ_last_w", "band_stage_streak_w",
    "band_rj_shutoff_w",
    "band_shutoff_best_w", "band_shutoff_streak_w", "band_cold_logl_max_w",
    "band_cold_logl_w", "band_shutoff_reset_w", "band_cold_logl_peak_w",
)


# --------------------------------------------------------------------------
# walker-axis classification
# --------------------------------------------------------------------------
def _main_signatures(attrs, branch_shapes):
    """``{dataset path: (walker axis|None, expected trailing shape|None)}``
    for the top-level group.

    Layout is ``(nsteps, nsamplers, ntemps, nwalkers, ...)`` for the chain
    family. ``betas`` stops at ntemps and has no walker axis; ``accepted``
    is ``(nsamplers, ntemps, nwalkers)`` with no step axis at all.
    """
    ns, nt, nw = attrs["nsamplers"], attrs["ntemps"], attrs["nwalkers"]
    sig = {
        "log_like": (3, (ns, nt, nw)),
        "log_prior": (3, (ns, nt, nw)),
        "betas": (None, (ns, nt)),
        "accepted": (2, (ns, nt, nw)),
        "samplers_running": (None, (ns,)),
        "swaps_accepted": (None, None),
        # per-row leg-ender names (hdfbackend.SAVED_AFTER_DS), walker-free
        "saved_after": (None, None),
    }
    for branch, (nleaves, ndim) in branch_shapes.items():
        sig[f"chain/{branch}"] = (3, (ns, nt, nw, nleaves, ndim))
        sig[f"inds/{branch}"] = (3, (ns, nt, nw, nleaves))
        sig[f"blobs/{branch}"] = (3, None)
    return sig


def _sub_signatures(a):
    """``{name: (walker axis|None, expected trailing shape|None)}`` for one
    ``sub_backend/<branch>`` group, from that group's own attributes.

    Three families share this space and they do NOT agree on where the
    walker axis sits, so each is written out rather than pattern-matched:

    * every branch: ``chain``/``inds`` are ``(ntemps, nwalkers, ...)`` and
      ``d_h``/``h_h`` are ``(nwalkers, nleaves)``;
    * per-leaf ladders (mbh/emri/sobbh): ``log_like`` is
      ``(nleaves, ntemps, nwalkers)`` -- walker LAST;
    * flat ladders (psd/galfor): ``log_like`` is ``(ntemps, nwalkers)``;
    * GB family (gb/vgb): no ``log_like`` at all, but band tables --
      ``band_cold_ll``/``cap_cell_cold_ll`` are ``(nwalkers, nbands)`` and
      ``band_num_binaries`` is ``(ntemps, nwalkers, nbands)``. Every other
      band table is walker-free (per-band or per-(band, temp)), which is
      why the cap/shutoff state survives this rescale untouched.
    """
    nt, nw = int(a["ntemps"]), int(a["nwalkers"])
    nl, nd = int(a["nleaves_max"]), int(a["ndim"])
    nb = int(a["num_bands"]) if "num_bands" in a else None

    sig = {
        "chain": (2, (nt, nw, nl, nd)),
        "inds": (2, (nt, nw, nl)),
        "d_h": (1, (nw, nl)),
        "h_h": (1, (nw, nl)),
        # both ladder shapes are accepted; the builder picks by measurement
        "log_like": ((3, (nl, nt, nw)), (2, (nt, nw))),
        "log_prior": ((3, (nl, nt, nw)), (2, (nt, nw))),
    }
    # Counters and ladders, walker-free in BOTH families. These are listed
    # with their full expected shapes rather than left to fall through as
    # "no axis of length nwalkers", because for mbh nleaves_max == nwalkers
    # == 4: `in_model_accepted` at (nsteps, 4, 2) is per-LEAF, and a script
    # that guessed by length would have doubled the leaf axis and produced
    # a store that loads and is wrong.
    for name in ("in_model_accepted", "in_model_proposed",
                 "rj_accepted", "rj_proposed"):
        sig[name] = ((None, (nl, nt)), (None, (nt,)))
    for name in ("swaps_accepted", "swaps_proposed"):
        sig[name] = ((None, (nl, nt - 1)), (None, (nt - 1,)))
    sig["betas_all"] = (None, (nl, nt))
    sig["betas"] = (None, (nt,))

    if nb is not None:
        # The cap-cell grid is a refinement of the band grid by
        # GB_CAP_DIVISOR; a store written before the cap grid existed has
        # no ``num_cap_cells`` attribute and the two coincide.
        nc = int(a["num_cap_cells"]) if "num_cap_cells" in a else nb
        sig["band_cold_ll"] = (1, (nw, nb))
        sig["cap_cell_cold_ll"] = (1, (nw, nb))
        sig["band_num_binaries"] = (2, (nt, nw, nb))
        # PER-WALKER leaf caps (GB_LEAF_CAP_PER_WALKER, 2026-09-22). The
        # rest of the cap family is walker-free -- which is exactly why a
        # walker rescale can carry the whole GB search state forward --
        # and these four are the exception: they are tiled like any other
        # walker-axis table, so walker w of the copy inherits the caps and
        # the patience clocks of walker ``w % nwalkers``. Both the band-
        # and cap-grid widths are accepted, because the store may or may
        # not carry ``num_cap_cells`` and the two agree at divisor 1.
        for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                     "cap_cell_best_ll_w"):
            sig[name] = ((1, (nw, nc)), (1, (nw, nb)))
        sig["band_best_ll_w"] = (1, (nw, nb))
        for name in PER_WALKER_BAND_TABLES:
            sig[name] = ((1, (nw, nb)), (1, (nw, nc)))
        # the valve's recipe-step stamp and the min-over-walkers stage
        # mirror: walker-free despite the names
        sig["band_shutoff_w_step"] = (None, (1,))
        # the valve's window-rule stamp (state.VALVE_WINDOW_RULE_FIELD,
        # 2026-10-13): one int per row, walker-free
        sig["band_window_rule"] = (None, (1,))
        sig["band_stage"] = (None, (nb,))
        # per-band and per-(band, temp) state: the leaf caps, the shutoff
        # ladder and the swap census. All walker-free, which is exactly why
        # a walker rescale can carry the whole GB search state forward.
        for name in ("band_best_ll", "band_cap_iters", "band_leaf_cap",
                     "band_occ_last", "band_occ_streak", "band_rj_shutoff",
                     "cap_cell_best_ll", "cap_cell_iters",
                     "cap_cell_leaf_cap"):
            sig[name] = (None, (nb,))
        for name in ("band_num_accepted", "band_num_accepted_rj",
                     "band_num_proposed", "band_num_proposed_rj",
                     "band_temps"):
            sig[name] = (None, (nb, nt))
        for name in ("band_swaps_accepted", "band_swaps_proposed"):
            sig[name] = (None, (nb, nt - 1))
        for name in ("band_shutoff_epoch", "band_shutoff_since_revive"):
            sig[name] = (None, (1,))
        # static grids: no step axis, so the shape IS the trailing shape
        sig["band_edges"] = (None, (nb + 1,))
        sig["cap_edges"] = (None, (nb + 1,))
    return sig


def _match(name, sig, trailing):
    """Resolve one dataset against its signature entry.

    Returns ``(axis, "ok")``, or ``(None, reason)`` when the entry is
    unknown or the shape disagrees. A disagreement is never downgraded to
    "leave it alone": the caller turns it into a refusal, because a store
    whose layout has moved is exactly the case where a silent pass writes
    a corrupt copy.
    """
    if name not in sig:
        return None, "not in the signature table"
    entry = sig[name]
    options = entry if isinstance(entry[0], tuple) else (entry,)
    for axis, expect in options:
        if expect is None or tuple(trailing) == tuple(int(v) for v in expect):
            return axis, "ok"
    shapes = " or ".join(str(tuple(int(v) for v in e)) for _, e in options
                         if e is not None)
    return None, f"shape {tuple(trailing)} != expected {shapes}"


class Entry:
    __slots__ = ("path", "dset", "axis", "rowwise", "note")

    def __init__(self, path, dset, axis, rowwise, note):
        self.path, self.dset = path, dset
        self.axis, self.rowwise, self.note = axis, rowwise, note


def plan_rescale(f, group, factor):
    """Classify every dataset under ``group``. Returns ``(entries, refusals)``.

    ``rowwise`` marks the datasets whose axis 0 is the step axis (their
    first dimension equals the store's allocated row count); only those are
    copied up to ``iteration``. Everything else -- ``band_edges``,
    ``accepted``, static tables -- is copied whole, which is why the row
    count is measured rather than assumed.
    """
    g = f[group]
    a = dict(g.attrs)
    n_alloc, _row_src = _row_count(f, group)
    nw_main = int(a["nwalkers"])

    branch_shapes = {}
    for branch in g["chain"]:
        branch_shapes[branch] = (int(g["nleaves_max"].attrs[branch]),
                                 int(g["ndims"].attrs[branch]))
    main_sig = _main_signatures(
        {k: int(a[k]) for k in ("nsamplers", "ntemps", "nwalkers")},
        branch_shapes)

    entries, refusals = [], []

    def classify(path, dset, sig, nw, key):
        shape = tuple(int(s) for s in dset.shape)
        rowwise = bool(shape) and shape[0] == n_alloc
        trailing = shape[1:] if rowwise else shape
        axis, why = _match(key, sig, trailing)
        if why != "ok":
            # Only an axis that could BE the walker axis is dangerous.
            if nw in shape:
                refusals.append(f"{path}: {why} (shape {shape} carries an "
                                f"axis of length nwalkers={nw})")
            else:
                entries.append(Entry(path, dset, None, rowwise,
                                     f"no walker axis ({why})"))
            return
        # Signature axes are indices into the FULL dataset shape (the step
        # axis included where there is one); only the SHAPE comparison
        # above drops it. Keep it that way -- an axis that shifts with
        # ``rowwise`` is how the first version of this walked off the end
        # of ``log_like``.
        full_axis = axis
        if full_axis is not None and shape[full_axis] != nw:
            refusals.append(
                f"{path}: axis {full_axis} is {shape[full_axis]}, expected "
                f"nwalkers={nw}")
            return
        entries.append(Entry(path, dset, full_axis, rowwise, "ok"))

    for name in ("log_like", "log_prior", "betas", "accepted",
                 "samplers_running", "swaps_accepted", "saved_after"):
        if name in g:
            classify(name, g[name], main_sig, nw_main, name)
    for fam in ("chain", "inds", "blobs"):
        if fam not in g:
            continue
        for branch in g[fam]:
            key = f"{fam}/{branch}"
            classify(key, g[fam][branch], main_sig, nw_main, key)

    if "sub_backend" in g:
        for branch in g["sub_backend"]:
            sub = g["sub_backend"][branch]
            sa = dict(sub.attrs)
            if "nwalkers" not in sa:
                refusals.append(f"sub_backend/{branch}: no nwalkers attr")
                continue
            sig = _sub_signatures(sa)
            nw_sub = int(sa["nwalkers"])
            for name in sub:
                obj = sub[name]
                if not isinstance(obj, h5py.Dataset):
                    continue
                classify(f"sub_backend/{branch}/{name}", obj, sig, nw_sub,
                         name)

    # COMPLETENESS. The loops above enumerate the layout this script knows;
    # a dataset added anywhere else would simply be dropped from the copy,
    # silently, and the run would fail later with a missing-key error far
    # from here. Count what is actually in the file and refuse on a
    # mismatch instead.
    seen = set()

    def note(name, obj):
        if isinstance(obj, h5py.Dataset):
            seen.add(name)

    g.visititems(note)
    planned = {e.path for e in entries}
    missed = sorted(seen - planned - {r.split(":")[0] for r in refusals})
    for m in missed:
        refusals.append(f"{m}: present in the store but not enumerated by "
                        f"this script")

    return entries, refusals, n_alloc, int(a["iteration"])


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def _create_scaled(src_ds, parent, name, new_shape, new_maxshape):
    """``compact_gf_store._create_like`` with an explicit maxshape.

    The upstream helper clones the source's maxshape, which is right for a
    compaction (shapes only shrink) and wrong here: a store whose walker
    axis is created with a FIXED maxshape equal to nwalkers rejects the
    doubled dataset with "Maxdims is smaller than dims". So the walker
    axis's bound is scaled alongside its extent, and an UNLIMITED bound
    (the step axis) stays unlimited.
    """
    new_shape = tuple(int(s) for s in new_shape)
    try:
        dcpl = src_ds.id.get_create_plist()
        maxdims = tuple(h5py.h5s.UNLIMITED if m is None else int(m)
                        for m in new_maxshape)
        sid = h5py.h5s.create_simple(new_shape, maxdims)
        dsid = h5py.h5d.create(parent.id, name.encode("utf-8"),
                               src_ds.id.get_type(), sid, dcpl)
        return h5py.Dataset(dsid)
    except Exception as exc:                       # pragma: no cover
        print(f"    NOTE: dcpl clone failed for {name} ({exc!r}); falling "
              "back to explicit creation properties")
        kwargs = dict(shape=new_shape, dtype=src_ds.dtype,
                      chunks=src_ds.chunks, compression=src_ds.compression,
                      compression_opts=src_ds.compression_opts,
                      shuffle=src_ds.shuffle, fletcher32=src_ds.fletcher32,
                      scaleoffset=src_ds.scaleoffset,
                      fillvalue=src_ds.fillvalue)
        if src_ds.chunks is not None:
            kwargs["maxshape"] = new_maxshape
        return parent.create_dataset(name, **kwargs)


def _expand(block, axis, factor, mode):
    if mode == "repeat":
        return np.repeat(block, factor, axis=axis)
    reps = [1] * block.ndim
    reps[axis] = factor
    return np.tile(block, reps)


def _lagged_rows(src, sl, lag):
    """Source rows ``r - lag`` for the destination rows of block ``sl``
    (rows before the start repeat row 0)."""
    a, b = int(sl[0].start or 0), int(sl[0].stop)
    rest = tuple(sl[1:])
    lo, hi = a - lag, b - lag
    if hi <= 0:
        return np.repeat(src[(slice(0, 1),) + rest], b - a, axis=0)
    part = src[(slice(max(lo, 0), hi),) + rest]
    if lo < 0:
        part = np.concatenate(
            [np.repeat(src[(slice(0, 1),) + rest], -lo, axis=0), part], axis=0)
    return part


def _copy_dataset(src, dst, axis, factor, mode, n_rows, cap_bytes,
                  rowwise=False):
    """Copy ``src`` into ``dst``, duplicating the walker axis.

    Blocks may split the walker axis, so tiling inside a block would be
    wrong. Instead each source block is written ``factor`` times, once per
    copy ``k``, at the destination walker offset ``k * nwalkers`` -- which
    is correct for any block split because tile maps destination walker
    ``w`` to source walker ``w % nwalkers``. ``repeat`` mode cannot be
    written that way (its destination walkers interleave), so there the
    block is expanded in memory and written to a stretched slice. ``lag``
    writes slot ``i`` of copy ``k`` from source walker ``(i + k) % nw``,
    taken ``k`` rows earlier for a ROW-WISE dataset (:func:`_lagged_rows`)
    and at the same index for a walker-axis dataset with no step axis.
    """
    if not src.shape:
        dst[()] = src[()]
        return
    nw = src.shape[axis] if axis is not None else None
    for sl in iter_blocks(src.shape, src.dtype.itemsize, cap_bytes, n_rows,
                          chunks=src.chunks):
        block = src[sl]
        if axis is None:
            dst[sl] = block
            continue
        s = list(sl)
        w = s[axis]
        w0, w1 = w.start or 0, w.stop if w.stop is not None else nw
        if mode == "repeat":
            s[axis] = slice(w0 * factor, w1 * factor)
            dst[tuple(s)] = _expand(block, axis, factor, mode)
        elif mode == "lag":
            # destination slot i of copy k <- source walker (i + k) % nw,
            # k rows earlier (row-wise datasets only). Copy 0 is the block.
            s[axis] = slice(w0, w1)
            dst[tuple(s)] = block
            for k in range(1, factor):
                if (w0, w1) == (0, nw):
                    # the block spans every walker: one (lagged) read per
                    # copy, rotated in memory -- slot i <- walker (i+k) % nw
                    s[axis] = slice(k * nw, (k + 1) * nw)
                    part = _lagged_rows(src, sl, k) if rowwise else block
                    dst[tuple(s)] = np.roll(part, -k, axis=axis)
                    continue
                for i in range(w0, w1):
                    src_sl = list(sl)
                    src_sl[axis] = slice((i + k) % nw, (i + k) % nw + 1)
                    src_sl = tuple(src_sl)
                    s[axis] = slice(k * nw + i, k * nw + i + 1)
                    dst[tuple(s)] = (_lagged_rows(src, src_sl, k) if rowwise
                                     else src[src_sl])
        else:
            for k in range(factor):
                s[axis] = slice(k * nw + w0, k * nw + w1)
                dst[tuple(s)] = block


def build_rescaled(src_path, dst_path, group, factor, mode,
                   cap_bytes=DEFAULT_BUFFER_BYTES, verbose=False):
    """Write a store at ``dst_path`` with ``factor`` x the walkers."""
    with h5py.File(src_path, "r") as fs:
        entries, refusals, n_alloc, iteration = plan_rescale(fs, group, factor)
        if refusals:
            raise RuntimeError("refusing to rescale:\n  "
                               + "\n  ".join(refusals))
        if mode == "lag" and iteration < factor:
            raise RuntimeError(
                f"lag mode needs at least factor={factor} saved rows; the "
                f"store has iteration={iteration}")
        with h5py.File(dst_path, "w") as fd:
            _copy_attrs(fs, fd)
            # groups + attrs first, so attribute fixups below have a home
            def mk(name, obj):
                if isinstance(obj, h5py.Group):
                    _copy_attrs(obj, fd.require_group(name))
            fs.visititems(mk)

            for e in entries:
                full = f"{group}/{e.path}"
                parent = fd.require_group(os.path.dirname(full))
                shape = list(e.dset.shape)
                maxshape = list(e.dset.maxshape)
                if e.axis is not None:
                    shape[e.axis] *= factor
                    if maxshape[e.axis] is not None:
                        maxshape[e.axis] = int(maxshape[e.axis]) * factor
                new = _create_scaled(e.dset, parent, os.path.basename(full),
                                     tuple(shape), tuple(maxshape))
                _copy_attrs(e.dset, new)
                rows = iteration if e.rowwise else (
                    e.dset.shape[0] if e.dset.shape else 0)
                t0 = time.time()
                _copy_dataset(e.dset, new, e.axis, factor, mode, rows,
                              cap_bytes, rowwise=e.rowwise)
                if verbose:
                    print(f"    {e.path:52s} axis={e.axis} rows={rows} "
                          f"{time.time() - t0:6.1f}s")

            # nwalkers attrs LAST: the plan read them, so they stay truthful
            # for the whole build and only then describe the new store.
            fd[group].attrs.modify("nwalkers",
                                   int(fs[group].attrs["nwalkers"]) * factor)
            if "sub_backend" in fd[group]:
                for branch in fd[group]["sub_backend"]:
                    sub = fd[group]["sub_backend"][branch]
                    if "nwalkers" in sub.attrs:
                        sub.attrs.modify(
                            "nwalkers", int(sub.attrs["nwalkers"]) * factor)
    return iteration


# --------------------------------------------------------------------------
# directory clone
# --------------------------------------------------------------------------
def find_store(run_dir, base_name=None):
    """The LIVE ``*_testing.h5`` in ``run_dir``.

    A run directory routinely holds more than one: a legacy base name, an
    800-byte stub, a ``*_extract.h5`` reduced copy. Picks by the largest
    stored ``iteration`` attribute, breaking ties by mtime, and reports
    what it rejected so a wrong pick is visible rather than silent.
    """
    cands = []
    for entry in sorted(os.listdir(run_dir)):
        if not entry.endswith(".h5"):
            continue
        if entry.endswith("_running_backup_copy.h5"):
            continue
        if base_name and not entry.startswith(base_name):
            continue
        path = os.path.join(run_dir, entry)
        try:
            with h5py.File(path, "r") as f:
                if "global_fit" not in f:
                    continue
                it = int(f["global_fit"].attrs.get("iteration", -1))
        except OSError:
            continue
        cands.append((it, os.path.getmtime(path), path))
    if not cands:
        raise SystemExit(f"no global-fit store found in {run_dir}")
    cands.sort(reverse=True)
    if len(cands) > 1:
        print(f"  {len(cands)} candidate stores; picked the one at the "
              f"highest iteration:")
        for it, mt, p in cands:
            mark = "->" if p == cands[0][2] else "  "
            print(f"    {mark} {os.path.basename(p):48s} iteration={it}")
    return cands[0][2]


def clone_dir(src_dir, dst_dir, store_path, apply_, link_big=False):
    """Copy everything EXCEPT the store and the walker-shaped sidecars."""
    skip = {os.path.basename(store_path)}
    side = sidecar_paths(store_path)
    skip.add(os.path.basename(side["backup"]))
    skip.add(os.path.basename(side["midit"]))

    total = 0
    print("\n  directory clone:")
    for entry in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, entry)
        if entry in skip:
            print(f"    SKIP  {entry:52s} (rebuilt or stale at the new "
                  f"walker count)")
            continue
        size = 0
        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                size += sum(os.path.getsize(os.path.join(root, fn))
                            for fn in files)
        else:
            size = os.path.getsize(src)
        total += size
        how = "link" if (link_big and os.path.isdir(src)) else "copy"
        print(f"    {how.upper():5s} {entry:52s} {size / MB:9.1f} MB")
        if not apply_:
            continue
        dst = os.path.join(dst_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(
                src, dst,
                copy_function=os.link if link_big else shutil.copy2)
        else:
            shutil.copy2(src, dst)
    print(f"    {'':5s} {'TOTAL':52s} {total / MB:9.1f} MB")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir", help="the run directory to clone")
    ap.add_argument("dst_dir", help="the new run directory (must not exist)")
    ap.add_argument("--nwalkers", type=int, default=None,
                    help="target walker count: a multiple of 4 and of the "
                         "store's walker count (4 -> 8 uses the last 2 rows "
                         "of the chain, 4 -> 32 the last 8)")
    ap.add_argument("--factor", type=int, default=None,
                    help="walker multiplier instead of --nwalkers")
    ap.add_argument("--mode", default="lag", choices=("lag", "tile", "repeat"),
                    help="lag (default): new walker j*nw + w is walker w "
                         "lagged by j saved rows, i.e. the new walkers come "
                         "from the last nwalkers/nw samples of each chain. "
                         "tile: new walker w is a copy of w %% nwalkers at "
                         "the same row. repeat: each source walker is "
                         "duplicated in place")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--base-name", default=None,
                    help="restrict the store search to this filename prefix")
    ap.add_argument("--buffer-mb", type=float, default=64.0)
    ap.add_argument("--link-fstat", action="store_true",
                    help="HARD-LINK copied directories (gb_fstat_fit/, "
                         "warmstart/) instead of copying. Saves the space "
                         "and the time, and is safe while both runs only "
                         "ADD epoch directories -- but a tool that rewrites "
                         "an existing cached file in place would corrupt "
                         "both runs. Off by default.")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args(argv)

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    if not os.path.isdir(src_dir):
        print(f"no such directory: {src_dir}", file=sys.stderr)
        return 2
    if os.path.exists(dst_dir) and args.apply:
        print(f"REFUSING: {dst_dir} already exists", file=sys.stderr)
        return 3
    if (args.nwalkers is None) == (args.factor is None):
        print("give exactly one of --nwalkers / --factor", file=sys.stderr)
        return 2

    store = find_store(src_dir, args.base_name)
    print(f"\n  store           : {store}")
    with h5py.File(store, "r") as f:
        nw = int(f[args.group].attrs["nwalkers"])
    if args.nwalkers is not None:
        if args.nwalkers % 4 or args.nwalkers % nw or args.nwalkers <= nw:
            print(f"--nwalkers {args.nwalkers} must be a multiple of 4 and of "
                  f"the store's {nw} walkers, and larger than {nw}",
                  file=sys.stderr)
            return 2
        args.factor = args.nwalkers // nw
    if args.factor < 2 or (nw * args.factor) % 4:
        print(f"--factor {args.factor}: need >= 2 and {nw} x factor a "
              f"multiple of 4", file=sys.stderr)
        return 2

    with h5py.File(store, "r") as f:
        entries, refusals, n_alloc, iteration = plan_rescale(
            f, args.group, args.factor)

    print(f"  allocated rows  : {n_alloc}")
    print(f"  stored iteration: {iteration}  (rows 0..{iteration - 1} copied)")
    print(f"  walkers         : {nw} -> {nw * args.factor}  "
          f"(mode={args.mode})")
    if args.mode == "lag":
        if iteration < args.factor:
            print(f"REFUSING: lag mode needs {args.factor} saved rows, the "
                  f"store has {iteration}", file=sys.stderr)
            return 3
        print(f"  resume state of each new walker (row, source walker): "
              f"{[(iteration - 1 - w // nw, (w % nw + w // nw) % nw) for w in range(nw * args.factor)]}")
    elif args.mode == "tile":
        print(f"  source walker of each new walker: "
              f"{[w % nw for w in range(nw * args.factor)]}")
    else:
        print(f"  source walker of each new walker: "
              f"{[w // args.factor for w in range(nw * args.factor)]}")

    scaled = [e for e in entries if e.axis is not None]
    print(f"\n  datasets: {len(entries)} total, {len(scaled)} carry a "
          f"walker axis")
    for e in scaled:
        shape = tuple(e.dset.shape)
        new = list(shape)
        new[e.axis] *= args.factor
        print(f"    {e.path:52s} axis {e.axis}  {shape} -> {tuple(new)}")
    untouched = [e for e in entries if e.axis is None]
    print(f"\n  walker-free (copied as-is): "
          f"{', '.join(e.path.split('/')[-1] for e in untouched[:12])}"
          f"{' ...' if len(untouched) > 12 else ''}")

    if refusals:
        print("\n  REFUSING -- unclassified datasets:", file=sys.stderr)
        for r in refusals:
            print(f"    {r}", file=sys.stderr)
        return 3

    side = sidecar_paths(store)
    print("\n  sidecars:")
    for key, path in side.items():
        exists = os.path.exists(path)
        if key == "fstat":
            verb = "CARRIED OVER (no walker axis)" if exists else "absent"
        else:
            verb = "NOT copied (holds the old walker count)" if exists \
                else "absent"
        print(f"    {key:7s} {os.path.basename(path):48s} {verb}")
    print("    eigen   *_eigen_tables.pkl                        "
          "carried over; walker_max entries self-reject on the "
          "nwalkers guard and rebuild")

    # Size projection. The walker-scaled datasets grow by `factor`; the rest
    # is carried as-is. Stored (post-compression) bytes are what matters --
    # a GB chain is 212 GB logical and ~23 MB on disk -- so the estimate is
    # built from get_storage_size(), not nbytes.
    with h5py.File(store, "r") as f:
        grow = sum(f[f"{args.group}/{e.path}"].id.get_storage_size()
                   for e in scaled)
        keep = sum(f[f"{args.group}/{e.path}"].id.get_storage_size()
                   for e in untouched)
    floor = grow * args.factor + keep
    dir_bytes = clone_dir(src_dir, dst_dir, store, apply_=False,
                          link_big=args.link_fstat)
    need = floor + (0 if args.link_fstat else dir_bytes)
    free = shutil.disk_usage(os.path.dirname(dst_dir) or ".").free
    print(f"\n  store on disk   : {(grow + keep) / MB:.1f} MB "
          f"({grow / MB:.1f} walker-scaled + {keep / MB:.1f} carried)")
    print(f"  new store, LOWER BOUND: {floor / MB:.1f} MB")
    print("    The real figure is larger, sometimes several times larger, "
          "and cannot be")
    print("    projected from stored sizes: reading a row that the source "
          "never allocated")
    print("    returns fill values, and writing them MATERIALISES the chunk "
          "in the copy.")
    print("    Measured 8x on a sparse extract; a store whose live rows were "
          "all written")
    print("    lands near the bound. Leave real headroom.")
    print(f"  disk            : {free / MB:.0f} MB free, at least "
          f"{need / MB:.0f} MB needed")
    if free < need:
        print("  REFUSING: free space is below even the lower bound.",
              file=sys.stderr)
        return 3
    if free < need * 5:
        print("  WARNING: less than 5x the lower bound is free. If the build "
              "runs out of\n           space it leaves a partial store "
              "behind; the SOURCE is never touched.")

    if not args.apply:
        print("\n  DRY RUN -- nothing written. Re-run with --apply.")
        return 0

    holders = pids_holding(store)
    if holders:
        print(f"\n  REFUSING: {store} is open in PID(s) "
              f"{', '.join(map(str, holders))}. Stop the job first.",
              file=sys.stderr)
        return 3

    os.makedirs(dst_dir, exist_ok=False)
    clone_dir(src_dir, dst_dir, store, apply_=True, link_big=args.link_fstat)

    dst_store = os.path.join(dst_dir, os.path.basename(store))
    print(f"\n  building {os.path.basename(dst_store)} ...", flush=True)
    t0 = time.time()
    try:
        build_rescaled(store, dst_store, args.group, args.factor, args.mode,
                       cap_bytes=max(1, int(args.buffer_mb * MB)),
                       verbose=True)
    except Exception as exc:
        if os.path.exists(dst_store):
            os.remove(dst_store)
        print(f"  BUILD FAILED ({exc!r}); source untouched.", file=sys.stderr)
        return 3
    print(f"  wrote {dst_store} "
          f"({os.path.getsize(dst_store) / MB:.1f} MB) in "
          f"{time.time() - t0:.1f}s")
    print(f"\n  relaunch: your usual launch line with\n"
          f"    STORE_DIR={dst_dir}/ NWALKERS={nw * args.factor}\n"
          f"  prepended (e.g. ./scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
