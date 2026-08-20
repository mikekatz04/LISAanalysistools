#!/usr/bin/env python
"""Extract a small, analysis-complete copy of a global-fit store.

The production h5 is multi-GB (full per-iteration chains at 24 internal GB
rungs); the monitor page and every deep-dive analysis of 2026-08 read only:

* full HISTORY of the small arrays (log_like, band/cap bookkeeping, noise
  and VGB sub-chains, swap counters, leaf-count masks), and
* the LAST FEW ITERATIONS of the huge ones (cold + all-rung GB chains).

This writes ``<store>_extract.h5`` with the SAME group/dataset layout and
the SAME logical shapes -- big datasets are created chunked+gzip and only
the last ``--keep`` rows are written, so unwritten regions occupy no disk
and read back as zeros. Consumers that index ``[NIT-1]`` (the monitor, the
census scripts) work unchanged; nothing needs to know it is an extract.

Run in the store directory on the cluster:

    python gf_store_extract.py gf_prod_1yr_v4/gf_prod_1yr_testing.h5

Then ship the extract (plus the run dir's logs / artifacts / gb_fstat_fit /
dissect, which are already small) instead of the main h5.

Typical size: a few hundred MB against a multi-GB store.
"""
import argparse
import os
import sys

import h5py
import numpy as np

# Datasets whose FIRST axis is the iteration axis and which are too big to
# keep in full: only the last --keep rows are written. Matched by path
# suffix. Everything else is copied whole.
LAST_K_ONLY = (
    "chain/gb",            # main cold chain (it, 1, 1, nw, nleaves, 9)
    "inds/gb",             # kept in full if small enough; see SIZE_CAP
    "sub_backend/gb/chain",        # all-rung GB chain (it, nt, nw, nleaves, 9)
    "sub_backend/gb/inds",
    "sub_backend/gb/d_h",
    "sub_backend/gb/h_h",
    "sub_backend/gb/band_num_binaries",
    "chain/vgb",
    "sub_backend/vgb/chain",       # kept whole if under SIZE_CAP
)
# Anything (matched or not) whose raw size is below this is copied whole.
SIZE_CAP = 600 * 1024 * 1024


def _wanted_rows(ds, keep, it):
    """Rows of a first-axis-is-iteration dataset worth writing."""
    n = ds.shape[0]
    hi = min(it if it and 0 < it <= n else n, n)
    lo = max(hi - keep, 0)
    return lo, hi


def extract(src_path, dst_path, keep):
    src = h5py.File(src_path, "r")
    it = None
    try:
        it = int(src["global_fit"].attrs.get("iteration"))
    except Exception:
        pass
    print(f"[extract] {src_path} -> {dst_path}  (iteration attr: {it}, "
          f"keep last {keep} rows of the big iteration-axis datasets)")
    dst = h5py.File(dst_path, "w")
    stats = {"full": 0, "partial": 0, "bytes_in": 0}

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            g = dst.require_group(name)
            for k, v in obj.attrs.items():
                g.attrs[k] = v
            return
        raw = int(np.prod(obj.shape)) * obj.dtype.itemsize if obj.shape else 0
        stats["bytes_in"] += raw
        partial = raw > SIZE_CAP and any(name.endswith(s) or s in name
                                         for s in LAST_K_ONLY)
        if not partial and raw > SIZE_CAP:
            # Unlisted-but-huge: keep last rows anyway rather than blow up
            # the extract; loudly, so a new consumer knows to look here.
            print(f"  [warn] unlisted large dataset {name} "
                  f"({raw/1e9:.2f} GB) -> last-{keep} rows only")
            partial = True
        if obj.shape == ():  # scalar
            d = dst.create_dataset(name, data=obj[()])
        elif not partial:
            d = dst.create_dataset(name, data=obj[...],
                                   compression="gzip", compression_opts=4)
            stats["full"] += 1
        else:
            chunks = (1,) + tuple(min(s, c) for s, c in
                                  zip(obj.shape[1:],
                                      obj.chunks[1:] if obj.chunks
                                      else obj.shape[1:]))
            d = dst.create_dataset(name, shape=obj.shape, dtype=obj.dtype,
                                   chunks=chunks, compression="gzip",
                                   compression_opts=4, fillvalue=0)
            lo, hi = _wanted_rows(obj, keep, it)
            for r in range(lo, hi):
                d[r] = obj[r]
            stats["partial"] += 1
            print(f"  [last-{hi-lo}] {name} {obj.shape}")
        for k, v in obj.attrs.items():
            d.attrs[k] = v

    for k, v in src.attrs.items():
        dst.attrs[k] = v
    src.visititems(visit)
    dst.close()
    out = os.path.getsize(dst_path)
    print(f"[extract] done: {stats['full']} full + {stats['partial']} "
          f"partial datasets; {stats['bytes_in']/1e9:.2f} GB raw -> "
          f"{out/1e6:.0f} MB on disk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store", help="path to the live store h5")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--keep", type=int, default=3,
                    help="iterations of the big chains to keep (default 3)")
    a = ap.parse_args()
    out = a.out or a.store.replace(".h5", "_extract.h5")
    extract(a.store, out, a.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
