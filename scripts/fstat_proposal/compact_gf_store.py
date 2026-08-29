"""Rebuild a global-fit store holding ONLY the live rows -- the one way to
actually reclaim the space a rewind discards.

WHY THIS EXISTS. ``reset_recipe_stage.py --rewind-to-empty gb`` moves
``global_fit.attrs["iteration"]`` back to the handover row, and that is
enough to make the RUN correct: the backend and every ``ModuleSubBackend``
resume from row ``iteration - 1`` and the next ``grow()`` truncates the
extent. It is not enough to make the FILE small. Neither ``resize`` nor
``h5repack`` reclaims the space, and the reason is chunk geometry rather
than a missing flag:

``sub_backend/gb/chain`` is chunked ``(32, 2, 2, 625, 1)`` under gzip -- the
STEP axis is chunked 32 rows at a time. So the chunk covering rows 0..31
holds BOTH live rows (0..iteration-1) and discarded ones. It is PARTIALLY
live: shrinking the extent cannot free a chunk that still holds live data,
and ``h5repack`` copies every allocated chunk whole. Measured on the
3-month v7 store (iteration 57, extent 2054 rows) that one dataset is
406.1 MiB of the 507.0 MiB of allocated dataset storage in a 570.6 MiB file.

WHAT THIS DOES INSTEAD (user's ruling): "rebuild the file and just fill in
the right rows through the right iteration." A NEW file is created and only
rows ``[0, iteration)`` of every per-iteration dataset are written into it,
plus everything else copied intact. Nothing dead is ever written, so nothing
dead can be allocated -- the partial-chunk problem does not get worked
around, it never arises.

MEASURED, on a byte-identical copy of that store at ``--iteration 5``:
598,271,577 B -> 20,801,049 B, i.e. 570.6 MiB -> 19.8 MiB, a factor of 28.8.
72 s to build, 208 s end to end including verification and the swap.

THE TRAP THIS TOOL IS BUILT AROUND is deciding WHICH datasets have a step
axis. A dataset is per-iteration iff its first axis equals the backend's
ALLOCATED ROW COUNT -- sized here from a known step-axis dataset, the first
branch under ``<group>/inds``. It is NOT "shape[0] > iteration":
``sub_backend/gb/band_edges`` is ``(num_bands + 1,) = (1233,)``, which is
comfortably larger than any sane rewind target, and truncating it would
destroy the band grid that the resume guard checks and that every cap-cell
index is defined against. ``band_edges`` and ``cap_edges`` are exactly
``GBState.static_names``; they and every other non-step dataset are copied
WHOLE. A second guard backs the row-count test up: a step dataset is
resizable by construction (the backend calls ``grow()`` on it), so
``maxshape[0]`` must also be ``None``. That is what tells a genuine step
dataset apart from a static that merely happens to be as long as the row
count, and it is why ``global_fit/accepted`` -- first axis 1, fixed
maxshape -- is copied whole rather than mistaken for a one-row chain.

WHAT IS PRESERVED. Group hierarchy; every group and dataset attribute
(including ``global_fit.attrs["iteration"]`` and the ``recipe/*``
``status`` / ``order num`` flags that decide which stage resumes); dtypes;
``maxshape`` (the step axis stays UNLIMITED -- the backend calls ``grow()``,
so a frozen step axis would kill the run on its first save); chunk shape;
and compression/filters. Structure is cloned at the HDF5 level: the new
dataset is created from the SOURCE's dataset-creation property list
(``h5py.h5d.create`` with the source ``dcpl``), so filters, chunk shape,
fill value and allocation policy come across byte-for-byte instead of being
guessed from h5py's high-level properties. A 32-row chunk on a 5-row
dataset is kept as-is on purpose: matching what the backend expects on
resume matters more than the few KB a re-chunk would save, and re-chunking
would be a silent, untested deviation from every other store on disk.

MEMORY. One logical row of ``sub_backend/gb/chain`` is 24*24*10000*9*8 B =
414 MB, so even a single row cannot be read whole on an 8 GB box. Copying
and verification both go through :func:`iter_blocks`, which tiles the live
rows into pieces of at most ``--buffer-mb`` (default 64 MB), splitting
INSIDE a row along the trailing axes when a row alone is too big. The tiling
takes all live rows together and lands on chunk boundaries where it can, so
each destination chunk is compressed exactly once; a block that stopped
part-way through a chunk would force HDF5 to decompress and recompress it
again for the next block.

SAFETY. Nothing is ever edited in place. The rebuild goes to a sibling temp
file, is verified against the original (same names, dtypes, non-step
dimensions, step extents, attributes, a still-``None`` ``maxshape[0]``, and
byte-identical first and last live rows), and only then swapped in with
``os.replace``. The original is KEPT as ``<name>.pre-compact-<timestamp>``,
never deleted. If anything at all fails to verify, the temp file is removed
and the original is left untouched. The run must be stopped first: an open
writer holds the file, so the tool refuses to start when ``lsof`` reports
another process on it.

SIDECARS (own flags, all off by default; everything is MOVED, never
deleted). ``reset_recipe_stage.py`` has no notion of these -- it edits one
integer in the h5 and stops there -- but after a rewind all three are stale:

* ``--reset-backup`` moves ``<base>_running_backup_copy.h5`` aside. It sits
  at a LATER iteration than the rewind point, and ``hdfbackend.py``'s
  ``_maybe_promote_backup`` swaps it over the primary whenever the primary
  fails the resume-readability check. A single unlucky read would therefore
  silently promote the pre-rewind chain and UNDO the rewind, with the run
  carrying on from it.
* ``--reset-midit`` removes ``<base>_midit_checkpoint.pkl``
  (:func:`lisatools.globalfit.midit_checkpoint.checkpoint_path`), which
  holds mid-iteration state captured before the rewind.
* ``--reset-fstat`` moves the run directory's ``gb_fstat_fit/`` aside.
  ``reset_recipe_stage.py`` deliberately KEEPS this cache -- "that is the
  expensive part of a run and there is no reason to pay for it twice" --
  and that is right for a stage RE-OPEN, which keeps the sources the grid
  was fitted against. It is WRONG after ``--rewind-to-empty gb``.
  ``GBSpecialRJFStatGridMove`` fits its comb/peak grids against the LIVE
  RESIDUAL, and each epoch's ``fstat_centers.npz`` centers are built
  against that same reference walker's residual snapshot. The rewind
  removes every GB leaf, so at resume the residual is the pre-search one:
  every GB template the epoch was fitted against is gone, and the stored
  peaks and centres describe a residual that no longer exists. It does not
  self-correct either -- at the default ``fstat_refit_every <= 0`` the move
  "fits exactly once, ever", so a complete ``epoch_<k>/`` is simply loaded
  and never refitted; and with a cadence set (production runs
  ``GB_FSTAT_REFIT_EVERY=50``) the refit clock is restart-persistent in
  ``gb_fstat_fit/shared/clock.json`` (``{"clock": 66}`` in this snapshot),
  which seeds the propose census on the first read while the last-fit tick
  rides in the epoch's ``DONE.json``, so the rewound run inherits both and
  waits out the cadence from the old position. Moving the directory aside
  restarts epoch numbering at ``epoch_0000`` with a zero clock. Refitting
  costs about 639 s, once.

Usage (dry run first -- nothing is written without ``--apply``)::

    python scripts/fstat_proposal/compact_gf_store.py STORE/base_testing.h5
    python scripts/fstat_proposal/compact_gf_store.py STORE/base_testing.h5 \
        --apply --reset-backup --reset-midit --reset-fstat

    # compact to an explicit rewind target instead of the store's own
    python scripts/fstat_proposal/compact_gf_store.py STORE/base_testing.h5 \
        --iteration 5 --apply

Run it AFTER ``reset_recipe_stage.py`` has set the iteration, or pass
``--iteration`` to do both at once (the attribute is rewritten to match the
new extent either way -- an extent shorter than the recorded iteration would
make ``get_last_sample`` read a row that is not there).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import os
import posixpath
import shutil
import subprocess
import sys
import time
import typing

import h5py
import numpy as np

DEFAULT_BUFFER_BYTES = 64 * 1024 ** 2
MB = 1024.0 ** 2
#: Above this many allocated chunks, skip the exact per-chunk size scan in
#: the dry-run report (it is a Python-level loop) and fall back to the
#: linear estimate. Reporting is not worth minutes of walking.
CHUNK_SCAN_LIMIT = 400_000


# --------------------------------------------------------------------------
# block iteration -- nothing large ever lands in RAM
# --------------------------------------------------------------------------
def _prod(values, start=1):
    out = start
    for v in values:
        out *= v
    return out


def _chunk_aligned_steps(shape, itemsize, cap_bytes, n_rows, chunks):
    """Per-axis block extents, each a whole number of chunks, under the cap.

    Why bother: a block that stops part-way through a chunk forces HDF5 to
    read, decompress, modify and recompress that chunk again when the next
    block reaches the rest of it. On ``sub_backend/gb/chain`` -- 20736
    allocated chunks in the live step-chunk row -- a misaligned tiling
    doubles the gzip work for no benefit. Taking whole chunks means every
    chunk is written exactly once.

    The live rows are always taken TOGETHER (``steps[0] = n_rows``): they sit
    inside one step-chunk row, or span whole ones, so a single pass over them
    touches each chunk once. Returns ``None`` when even one chunk-column of
    live rows exceeds the cap, in which case the caller falls back to plain
    splitting.
    """
    ndim = len(shape)
    steps = [min(int(chunks[i]), shape[i]) for i in range(ndim)]
    steps[0] = n_rows
    if _prod(steps, itemsize) > cap_bytes:
        return None
    for axis in range(ndim - 1, 0, -1):
        if steps[axis] >= shape[axis]:
            continue
        per_index = _prod(steps, itemsize) // steps[axis]
        ck = min(int(chunks[axis]), shape[axis])
        mult = max(1, (cap_bytes // per_index) // ck)
        steps[axis] = min(shape[axis], ck * mult)
    return steps


def iter_blocks(shape, itemsize, cap_bytes, n_rows, chunks=None):
    """Tile rows ``[0, n_rows)`` of ``shape`` into slice tuples.

    Each yielded block holds at most ``cap_bytes``, so nothing large lands in
    RAM: one logical row of ``sub_backend/gb/chain`` is 414 MB and is
    subdivided along its trailing axes rather than read whole. The step axis
    is taken WHOLE (all live rows at once) and only the trailing axes are
    split -- writing a chunk's live rows in one go is what keeps each chunk
    to a single compress pass. With ``chunks`` supplied the split lands on
    chunk boundaries as well; without it (or when a chunk-column alone
    exceeds the cap) the tiling falls back to plain recursive splitting,
    which is still correct, just less efficient. A zero-sized dataset yields
    nothing.
    """
    shape = tuple(int(s) for s in shape)
    n_rows = int(n_rows)
    if not shape or n_rows <= 0 or any(d == 0 for d in shape[1:]):
        return
    if len(shape) == 1:
        yield (slice(0, n_rows),)
        return

    if chunks is not None:
        steps = _chunk_aligned_steps(shape, itemsize, cap_bytes, n_rows,
                                     chunks)
        if steps is not None:
            axis_ranges = [[(0, n_rows)]]
            for axis in range(1, len(shape)):
                step = steps[axis]
                axis_ranges.append(
                    [(a, min(a + step, shape[axis]))
                     for a in range(0, shape[axis], step)])
            for combo in itertools.product(*axis_ranges):
                yield tuple(slice(a, b) for a, b in combo)
            return

    def rec(axis, prefix):
        # `prefix` fixes the step axis to all live rows and every axis
        # between 1 and `axis` to a single index, so one index along `axis`
        # with everything deeper full costs exactly `unit` bytes.
        unit = _prod(shape[axis + 1:], itemsize * n_rows)
        if unit == 0:
            return
        step = max(1, cap_bytes // unit)
        last = axis + 1 >= len(shape)
        for start in range(0, shape[axis], step):
            stop = min(start + step, shape[axis])
            sl = prefix + (slice(start, stop),)
            if last or (stop - start) * unit <= cap_bytes:
                yield sl + tuple(slice(None) for _ in shape[axis + 1:])
            else:
                yield from rec(axis + 1, sl)

    row0 = (slice(0, n_rows),)
    if _prod(shape[1:], itemsize * n_rows) <= cap_bytes:
        yield row0 + tuple(slice(None) for _ in shape[1:])
        return
    yield from rec(1, row0)


# --------------------------------------------------------------------------
# planning -- which datasets have a step axis, and what it would cost
# --------------------------------------------------------------------------
@dataclasses.dataclass
class DatasetPlan:
    name: str
    is_step: bool
    old_rows: int
    new_rows: int
    shape: tuple
    new_shape: tuple
    dtype: np.dtype
    maxshape: tuple
    chunks: typing.Optional[tuple]
    old_bytes: int
    #: predicted rebuilt size, from per-chunk-row occupancy -- see
    #: :func:`_project_new_bytes`.
    est_new_bytes: int
    #: exact stored bytes of every chunk that touches a kept row: a rigorous
    #: upper bound, since the rebuild can only shrink them. ``None`` when the
    #: chunk index was not scanned.
    bound_new_bytes: typing.Optional[int]


@dataclasses.dataclass
class Plan:
    group: str
    n_rows: int
    iteration: int
    stored_iteration: int
    groups: typing.List[str]
    datasets: typing.List[DatasetPlan]
    warnings: typing.List[str]

    @property
    def old_bytes(self):
        return sum(d.old_bytes for d in self.datasets)

    @property
    def est_new_bytes(self):
        return sum(d.est_new_bytes for d in self.datasets)

    @property
    def bound_new_bytes(self):
        return sum(d.bound_new_bytes if d.bound_new_bytes is not None
                   else d.est_new_bytes for d in self.datasets)


def _row_count(f, group):
    """Allocated step extent, sized from the first branch under ``inds``.

    Deliberately taken from a dataset that is a step dataset BY DEFINITION
    rather than inferred from the shapes at large.
    """
    if group not in f:
        raise ValueError(f"no {group!r} group -- is this a global-fit store?")
    g = f[group]
    if "inds" not in g or not len(g["inds"]):
        raise ValueError(
            f"{group}/inds is missing or empty; cannot size the step axis")
    branch = sorted(g["inds"])[0]
    return int(g["inds"][branch].shape[0]), f"{group}/inds/{branch}"


def _chunk_row_bytes(dset):
    """Stored bytes per STEP-CHUNK ROW: ``{k: bytes}`` for chunks at rows
    ``[k*c0, (k+1)*c0)``. ``None`` when the index cannot be walked cheaply.
    """
    if dset.chunks is None:
        return None
    c0 = dset.chunks[0]
    try:
        nchunks = dset.id.get_num_chunks()
        if nchunks > CHUNK_SCAN_LIMIT:
            return None
        out = {}
        for i in range(nchunks):
            info = dset.id.get_chunk_info(i)
            k = info.chunk_offset[0] // c0
            out[k] = out.get(k, 0) + int(info.size)
        return out
    except Exception:
        return None


def _project_new_bytes(rows_bytes, c0, keep_rows, written_rows):
    """Predict the rebuilt size, and a rigorous upper bound, in bytes.

    The mechanism the whole tool is about, run forwards. A step-chunk row
    that held ``w`` written rows and now holds ``k`` of them keeps roughly
    ``k/w`` of its compressed bytes: the rows that go away are replaced by
    fill, which costs almost nothing under gzip. Chunk rows past the kept
    range disappear entirely. The BOUND is the same sum with no shrinkage
    credit -- every chunk that touches a kept row, whole -- which the
    rebuild cannot exceed.

    Measured against the 3-month v7 store (2054-row extent, 57 rows
    written, 5 kept): 29.3 MiB predicted against 17.6 MiB of allocated
    dataset storage actually written, under a 209.5 MiB bound. The linear
    row-fraction estimate this replaces said 1.2 MiB -- 15x low.
    """
    est = bound = 0
    for k, nbytes in rows_bytes.items():
        lo = k * c0
        kept = max(0, min(keep_rows, lo + c0) - lo)
        if kept <= 0:
            continue
        bound += nbytes
        written = max(0, min(written_rows, lo + c0) - lo)
        est += nbytes if written <= 0 else nbytes * min(kept, written) / written
    return int(round(est)), int(bound)


def plan_compaction(f, group, iteration, scan_chunks=True):
    """Classify every object in ``f`` and price the rebuild.

    ``f`` is an open, read-mode :class:`h5py.File`. Raises ``ValueError``
    for an iteration outside ``1..n_rows``.
    """
    n_rows, sizer = _row_count(f, group)
    iteration = int(iteration)
    if not 1 <= iteration <= n_rows:
        raise ValueError(
            f"--iteration must be in 1..{n_rows} (the allocated extent of "
            f"{sizer}), got {iteration}")

    groups, datasets, warnings = [], [], []
    stored_iteration = int(f[group].attrs.get("iteration", -1))

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
            for key in obj:
                link = obj.get(key, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    warnings.append(
                        f"{name}/{key} is a {type(link).__name__}, not a hard "
                        "link; this tool only rebuilds hard links")
            return
        shape = tuple(int(s) for s in obj.shape)
        maxshape = obj.maxshape
        if not shape:                              # scalar dataset
            is_step, old_rows, new_rows = False, 0, 0
            new_shape = shape
        else:
            old_rows = shape[0]
            resizable = maxshape[0] is None
            is_step = (old_rows == n_rows) and resizable
            if old_rows == n_rows and not resizable:
                warnings.append(
                    f"{name}: first axis is {old_rows} (the row count) but "
                    "maxshape[0] is fixed, so it cannot be a step dataset "
                    "the backend grows -- copying it WHOLE")
            if resizable and old_rows != n_rows:
                warnings.append(
                    f"{name}: resizable step axis but extent {old_rows} != "
                    f"{n_rows}; copying it WHOLE (extents disagree)")
            new_rows = iteration if is_step else old_rows
            new_shape = (new_rows,) + shape[1:]
        old_bytes = int(obj.id.get_storage_size())
        if is_step and old_rows:
            # crude fallback if the chunk index cannot be walked
            est = int(round(old_bytes * new_rows / old_rows))
            bound = None
            if scan_chunks:
                rows_bytes = _chunk_row_bytes(obj)
                if rows_bytes is not None:
                    written = (stored_iteration if stored_iteration > 0
                               else old_rows)
                    est, bound = _project_new_bytes(
                        rows_bytes, obj.chunks[0], new_rows, written)
        else:
            est, bound = old_bytes, old_bytes
        datasets.append(DatasetPlan(
            name=name, is_step=is_step, old_rows=old_rows, new_rows=new_rows,
            shape=shape, new_shape=new_shape, dtype=obj.dtype,
            maxshape=maxshape, chunks=obj.chunks, old_bytes=old_bytes,
            est_new_bytes=est, bound_new_bytes=bound))

    for key in f:
        link = f.get(key, getlink=True)
        if not isinstance(link, h5py.HardLink):
            warnings.append(f"/{key} is a {type(link).__name__}, not a hard "
                            "link; this tool only rebuilds hard links")
    f.visititems(visit)
    return Plan(group=group, n_rows=n_rows, iteration=iteration,
                stored_iteration=stored_iteration, groups=groups,
                datasets=datasets, warnings=warnings)


# --------------------------------------------------------------------------
# rebuilding
# --------------------------------------------------------------------------
def _copy_attrs(src, dst):
    for key in src.attrs:
        dst.attrs.create(key, src.attrs[key])


def _create_like(src_ds, parent, name, new_shape):
    """Create ``name`` under ``parent`` with the SOURCE's creation plist.

    Cloning the dcpl carries filters, chunk shape, fill value and allocation
    policy across exactly, rather than round-tripping them through h5py's
    high-level ``compression``/``chunks``/... properties, which do not
    represent every filter a store may carry.
    """
    new_shape = tuple(int(s) for s in new_shape)
    try:
        dcpl = src_ds.id.get_create_plist()
        if not new_shape:
            sid = h5py.h5s.create(h5py.h5s.SCALAR)
        else:
            maxdims = tuple(h5py.h5s.UNLIMITED if m is None else int(m)
                            for m in src_ds.maxshape)
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
            kwargs["maxshape"] = src_ds.maxshape
        return parent.create_dataset(name, **kwargs)


def _copy_rows(src_ds, dst_ds, n_rows, cap_bytes):
    if not src_ds.shape:
        dst_ds[()] = src_ds[()]
        return
    for sl in iter_blocks(src_ds.shape, src_ds.dtype.itemsize, cap_bytes,
                          n_rows, chunks=src_ds.chunks):
        dst_ds[sl] = src_ds[sl]


def build_compacted(src_path, dst_path, group, iteration,
                    cap_bytes=DEFAULT_BUFFER_BYTES, verbose=False):
    """Write a new store at ``dst_path`` holding rows ``[0, iteration)``."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        plan = plan_compaction(src, group, iteration, scan_chunks=False)
        _copy_attrs(src, dst)
        for name in sorted(plan.groups, key=lambda n: n.count("/")):
            _copy_attrs(src[name], dst.require_group(name))
        for i, p in enumerate(plan.datasets, 1):
            src_ds = src[p.name]
            parent_name = posixpath.dirname(p.name)
            parent = dst.require_group(parent_name) if parent_name else dst
            dst_ds = _create_like(src_ds, parent,
                                  posixpath.basename(p.name), p.new_shape)
            _copy_attrs(src_ds, dst_ds)
            _copy_rows(src_ds, dst_ds, p.new_rows, cap_bytes)
            if verbose:
                print(f"    [{i:>3}/{len(plan.datasets)}] {p.name} "
                      f"({p.old_rows} -> {p.new_rows} rows)", flush=True)
        # The extent is now `iteration`; the recorded counter must agree, or
        # get_last_sample reads a row that is not there.
        dst[group].attrs["iteration"] = np.int64(iteration)
    return plan


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _norm_scalar(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def _values_equal(x, y):
    try:
        ax, ay = np.asarray(x), np.asarray(y)
    except Exception:                              # pragma: no cover
        return bool(x == y)
    if ax.shape != ay.shape:
        return False
    if ax.dtype.kind == "O" or ay.dtype.kind == "O":
        return ([_norm_scalar(v) for v in ax.ravel()]
                == [_norm_scalar(v) for v in ay.ravel()])
    if ax.dtype.kind != ay.dtype.kind:
        return False
    if ax.dtype.kind == "f":
        return bool(np.array_equal(ax, ay, equal_nan=True))
    return bool(np.array_equal(ax, ay))


def _attr_problems(label, a, b):
    problems = []
    ka, kb = set(a.attrs), set(b.attrs)
    for key in sorted(ka - kb):
        problems.append(f"{label}: attr {key!r} missing from the rebuild")
    for key in sorted(kb - ka):
        problems.append(f"{label}: attr {key!r} appeared in the rebuild")
    for key in sorted(ka & kb):
        if not _values_equal(a.attrs[key], b.attrs[key]):
            problems.append(f"{label}: attr {key!r} changed value")
    return problems


def _inventory(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(
        n, "dataset" if isinstance(o, h5py.Dataset) else "group"))
    return out


def _blocks_equal(a_ds, b_ds, row_slice, cap_bytes):
    """Byte-compare one row range. ``tobytes`` so NaN payloads count too."""
    start, stop = row_slice
    shape = a_ds.shape
    for sl in iter_blocks((stop - start,) + shape[1:], a_ds.dtype.itemsize,
                          cap_bytes, stop - start, chunks=a_ds.chunks):
        shifted = (slice(sl[0].start + start, sl[0].stop + start),) + sl[1:]
        x, y = a_ds[shifted], b_ds[shifted]
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        try:
            if x.tobytes() != y.tobytes():
                return False
        except Exception:                          # pragma: no cover
            if not np.array_equal(x, y):
                return False
    return True


def verify(src_path, dst_path, group, iteration,
           cap_bytes=DEFAULT_BUFFER_BYTES, full_rows=False):
    """Compare the rebuild against the original. Returns a list of problems.

    An empty list is the only thing that permits the swap.
    """
    problems = []
    with h5py.File(src_path, "r") as a, h5py.File(dst_path, "r") as b:
        try:
            plan = plan_compaction(a, group, iteration, scan_chunks=False)
        except ValueError as exc:
            return [f"cannot plan the source: {exc}"]
        step = {p.name: p for p in plan.datasets}

        inv_a, inv_b = _inventory(a), _inventory(b)
        for name in sorted(set(inv_a) - set(inv_b)):
            problems.append(f"{name}: missing from the rebuild")
        for name in sorted(set(inv_b) - set(inv_a)):
            problems.append(f"{name}: appeared in the rebuild")

        problems += _attr_problems("/", a, b)
        if group in b:
            got = int(b[group].attrs.get("iteration", -1))
            if got != iteration:
                problems.append(
                    f"{group}: iteration attr is {got}, expected {iteration}")

        for name in sorted(set(inv_a) & set(inv_b)):
            if inv_a[name] != inv_b[name]:
                problems.append(
                    f"{name}: {inv_a[name]} became {inv_b[name]}")
                continue
            oa, ob = a[name], b[name]
            # `iteration` on the top group is legitimately rewritten to the
            # new extent, and is checked against `iteration` above instead.
            problems += [p for p in _attr_problems(name, oa, ob)
                         if not (name == group and "'iteration'" in p)]
            if inv_a[name] != "dataset":
                continue
            p = step.get(name)
            if oa.dtype != ob.dtype:
                problems.append(f"{name}: dtype {oa.dtype} -> {ob.dtype}")
            shape_bad = False
            if oa.shape[1:] != ob.shape[1:]:
                problems.append(
                    f"{name}: non-step dims {oa.shape[1:]} -> {ob.shape[1:]}")
                continue
            if oa.maxshape[1:] != ob.maxshape[1:]:
                problems.append(
                    f"{name}: non-step maxshape {oa.maxshape[1:]} -> "
                    f"{ob.maxshape[1:]}")
            if p is None:                          # pragma: no cover
                continue
            if p.is_step:
                if ob.shape[0] != iteration:
                    problems.append(
                        f"{name}: step extent {ob.shape[0]}, expected "
                        f"{iteration}")
                    shape_bad = True
                if ob.maxshape[0] is not None:
                    problems.append(
                        f"{name}: step axis is no longer resizable "
                        f"(maxshape[0] = {ob.maxshape[0]}); the backend's "
                        "grow() would fail on the first save")
            elif ob.shape[0] != oa.shape[0]:
                problems.append(
                    f"{name}: copied WHOLE but first axis {oa.shape[0]} -> "
                    f"{ob.shape[0]} (a static was truncated)")
                shape_bad = True
            # only a first-axis mismatch makes the row compare meaningless;
            # a changed attr must NOT suppress it
            if shape_bad or not oa.shape or 0 in oa.shape:
                continue
            if p.is_step:
                # Spot-checking rows 0 and iteration-1 SEPARATELY would
                # decompress the same chunks twice whenever the live rows sit
                # inside one or two step-chunk rows -- which is the normal
                # case after a rewind (5 live rows, a 32-row step chunk). Then
                # comparing every kept row is the same work and a strictly
                # stronger check, so take it.
                span = oa.chunks[0] if oa.chunks else iteration
                rows = ([(0, iteration)]
                        if full_rows or iteration <= 2 * span
                        else sorted({(0, 1), (iteration - 1, iteration)}))
            else:
                rows = [(0, oa.shape[0])]
            for lo, hi in rows:
                if not _blocks_equal(oa, ob, (lo, hi), cap_bytes):
                    problems.append(
                        f"{name}: rows [{lo}, {hi}) are not bit-identical")
    return problems


# --------------------------------------------------------------------------
# open-file guard, moves, reporting
# --------------------------------------------------------------------------
def _lsof(path):
    return subprocess.run(["lsof", "-t", "--", path], check=False,
                          capture_output=True, text=True, timeout=30).stdout


def pids_holding(path, _runner=None):
    """PIDs other than ours with ``path`` open, via ``lsof``.

    A missing or failing ``lsof`` returns ``[]`` -- the guard is a
    convenience, not a lock, and must not be the reason a run cannot be
    cleaned up.
    """
    runner = _runner or _lsof
    try:
        out = runner(path)
    except Exception:
        return []
    me = os.getpid()
    pids = []
    for token in (out or "").split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != me and pid not in pids:
            pids.append(pid)
    return pids


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _free_name(base):
    if not os.path.exists(base):
        return base
    n = 1
    while os.path.exists(f"{base}.{n}"):
        n += 1
    return f"{base}.{n}"


def move_aside(path, ts, apply):
    """Move a stale sidecar out of the way. Returns the new path, or None."""
    if not os.path.exists(path):
        return None
    dest = _free_name(f"{path}.stale-{ts}")
    if apply:
        os.rename(path, dest)
    return dest


def sidecar_paths(store):
    """The three sidecars, by the conventions their producers use."""
    base, _ = os.path.splitext(store)
    return {
        # hdfbackend.py::_atomic_backup_copy / _maybe_promote_backup
        "backup": base + "_running_backup_copy.h5",
        # midit_checkpoint.checkpoint_path
        "midit": base + "_midit_checkpoint.pkl",
        # recipe.py: dirname(main_file_path) / "gb_fstat_fit"
        "fstat": os.path.join(os.path.dirname(os.path.abspath(store)),
                              "gb_fstat_fit"),
    }


def _report(plan, store):
    print(f"{store}")
    print(f"  group           : {plan.group}")
    print(f"  allocated rows  : {plan.n_rows}")
    print(f"  stored iteration: {plan.stored_iteration}")
    print(f"  keeping rows    : 0..{plan.iteration - 1}  "
          f"({plan.iteration} rows)")
    print(f"  file on disk    : {os.path.getsize(store) / MB:.1f} MB")
    print()
    print("  per-dataset allocated storage. `old` is exact "
          "(dset.id.get_storage_size());\n  `~new` is predicted from "
          "per-chunk-row occupancy -- a step-chunk row that held\n  w written "
          "rows and now holds k keeps about k/w of its compressed bytes, the "
          "rest\n  becoming fill. It cannot exceed the bound on the TOTAL "
          "line (those same chunks,\n  whole).")
    print(f"    {'dataset':<48} {'rows':>13}  {'old MB':>9} {'~new MB':>9}")
    for p in sorted(plan.datasets, key=lambda d: -d.old_bytes):
        rows = f"{p.old_rows} -> {p.new_rows}" if p.shape else "scalar"
        tag = "" if p.is_step else "  (whole)"
        print(f"    {p.name:<48} {rows:>13}  {p.old_bytes / MB:9.3f} "
              f"{p.est_new_bytes / MB:9.3f}{tag}")
    print(f"    {'TOTAL':<48} {'':>13}  {plan.old_bytes / MB:9.3f} "
          f"{plan.est_new_bytes / MB:9.3f}   "
          f"(bound {plan.bound_new_bytes / MB:.1f} MB)")
    n_step = sum(1 for p in plan.datasets if p.is_step)
    print(f"\n  {n_step} per-iteration datasets truncated to "
          f"{plan.iteration} rows; "
          f"{len(plan.datasets) - n_step} copied whole "
          "(statics, band_edges/cap_edges included).")
    for w in plan.warnings:
        print(f"  WARNING: {w}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="path to the run's *_testing.h5")
    ap.add_argument("--group", default="global_fit",
                    help="top-level group name (default: global_fit)")
    ap.add_argument("--iteration", type=int, default=None,
                    help="rows 0..N-1 are kept; default is the store's own "
                         "global_fit.attrs['iteration']")
    ap.add_argument("--buffer-mb", type=float, default=64.0,
                    help="copy/verify working-buffer cap in MB (default 64)")
    ap.add_argument("--verify-full", action="store_true",
                    help="byte-compare EVERY kept row rather than the first "
                         "and the last (slow on a multi-GB chain)")
    ap.add_argument("--ignore-lock", action="store_true",
                    help="proceed even though lsof reports the store open "
                         "elsewhere (do not use on a live run)")
    ap.add_argument("--reset-backup", action="store_true",
                    help="move <base>_running_backup_copy.h5 aside; it is at "
                         "a LATER iteration and a truncation restore would "
                         "silently undo the rewind")
    ap.add_argument("--reset-midit", action="store_true",
                    help="move <base>_midit_checkpoint.pkl aside; it holds "
                         "pre-rewind mid-iteration state")
    ap.add_argument("--reset-fstat", action="store_true",
                    help="move gb_fstat_fit/ aside; its grids were fitted "
                         "against a residual the rewind deleted")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args(argv)

    store = os.path.abspath(args.store)
    if not os.path.exists(store):
        print(f"no such file: {store}", file=sys.stderr)
        return 2
    cap_bytes = max(1, int(args.buffer_mb * MB))

    try:
        with h5py.File(store, "r") as f:
            iteration = (args.iteration if args.iteration is not None
                         else int(f[args.group].attrs["iteration"]))
            plan = plan_compaction(f, args.group, iteration)
    except (ValueError, KeyError, OSError) as exc:
        print(f"cannot plan {store}: {exc}", file=sys.stderr)
        return 2

    _report(plan, store)
    hard = [w for w in plan.warnings if "hard link" in w]

    side = sidecar_paths(store)
    wanted = {"backup": args.reset_backup, "midit": args.reset_midit,
              "fstat": args.reset_fstat}
    print("\n  sidecars:")
    for key, path in side.items():
        exists = os.path.exists(path)
        state = "present" if exists else "absent"
        verb = ("would move aside" if wanted[key] and exists
                else "MOVE ASIDE" if wanted[key] else "kept (no flag)")
        print(f"    {key:<7} {os.path.basename(path):<48} {state:<8} {verb}")

    if not args.apply:
        print("\n  DRY RUN -- nothing written. Re-run with --apply.")
        if hard:
            print("  (and resolve the non-hard-link warnings first)")
        return 0

    if hard:
        print("\n  REFUSING: the store contains non-hard links this tool "
              "would not reproduce.", file=sys.stderr)
        return 3

    holders = pids_holding(store)
    if holders and not args.ignore_lock:
        print(f"\n  REFUSING: {store} is open in PID(s) "
              f"{', '.join(map(str, holders))}. Stop the job first "
              "(--ignore-lock overrides).", file=sys.stderr)
        return 3

    need = int(plan.bound_new_bytes * 1.2) + 64 * 1024 ** 2
    free = shutil.disk_usage(os.path.dirname(store) or ".").free
    print(f"\n  disk: {free / MB:.0f} MB free, need about "
          f"{need / MB:.0f} MB (the original is kept alongside)")
    if free < need:
        print("  REFUSING: not enough free space to write the rebuild "
              "beside the original.", file=sys.stderr)
        return 3

    ts = _timestamp()
    tmp = _free_name(f"{store}.compacting-{ts}.h5")
    print(f"\n  building {os.path.basename(tmp)} ...", flush=True)
    t0 = time.time()
    try:
        build_compacted(store, tmp, args.group, plan.iteration, cap_bytes,
                        verbose=True)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  BUILD FAILED ({exc!r}); original untouched.",
              file=sys.stderr)
        if "filter returned failure" in str(exc):
            # Seen for real on a bad snapshot copy, 2026-08-29: every
            # gb/chain chunk in the live row range failed while the
            # production store beside it read fine.
            print(
                "\n  That is the TORN GZIP CHUNK signature of a store killed "
                "mid-write (see hdfbackend.promote_backup_if_store_unreadable"
                "):\n  the chunk opens fine and only fails when READ. This "
                "store cannot serve a RESUME\n  either, so compaction is not "
                "the first problem to fix. Check whether\n  "
                "<base>_running_backup_copy.h5 is sound -- "
                "hdfbackend._validate_resume_readable(path)\n  reads exactly "
                "what a resume reads -- and promote it if it is, keeping the "
                "damaged\n  file for forensics. Then compact the promoted "
                "store.", file=sys.stderr)
        return 4
    print(f"  built in {time.time() - t0:.1f} s -- "
          f"{os.path.getsize(tmp) / MB:.1f} MB "
          f"(was {os.path.getsize(store) / MB:.1f} MB)")

    print("  verifying ...", flush=True)
    problems = verify(store, tmp, args.group, plan.iteration, cap_bytes,
                      full_rows=args.verify_full)
    if problems:
        print(f"  VERIFICATION FAILED ({len(problems)} problem(s)); the "
              "original is untouched:", file=sys.stderr)
        for p in problems[:40]:
            print(f"    - {p}", file=sys.stderr)
        if len(problems) > 40:
            print(f"    ... and {len(problems) - 40} more", file=sys.stderr)
        os.remove(tmp)
        return 5
    print("  verified: names, dtypes, non-step dims, step extents, attrs, "
          "resizable step axis,\n            statics copied whole, and the "
          "kept rows bit-identical (every row where the\n            live "
          "rows fit inside two step chunks, else the first and the last).")

    kept = _free_name(f"{store}.pre-compact-{ts}")
    os.replace(store, kept)
    os.replace(tmp, store)
    print(f"\n  WROTE {os.path.basename(store)} "
          f"({os.path.getsize(store) / MB:.1f} MB)")
    print(f"  KEPT  {os.path.basename(kept)} "
          f"({os.path.getsize(kept) / MB:.1f} MB) -- delete it yourself once "
          "you are happy")

    for key, path in side.items():
        if not wanted[key]:
            continue
        dest = move_aside(path, ts, apply=True)
        if dest is None:
            print(f"  sidecar {key}: {os.path.basename(path)} not present")
        else:
            print(f"  MOVED {os.path.basename(path)} -> "
                  f"{os.path.basename(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
