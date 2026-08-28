#!/usr/bin/env python
"""Precompute the unequal-arm coarse basis cache in parallel.

``UnequalArmInstrumentNoise._build_coarse_basis_data`` folds the six-link TDI-2
transfer once per WDM time column.  It is a pure function of that column's
light-travel times, so the loop parallelises exactly -- but the run builds it
serially, which is ~75 min on the two-year grid before iteration 1.

This writes the same ``.npz``, under the same content-addressed name, so the
run finds it as a cache hit and starts sampling immediately.  Column results
are accumulated in column order by the parent, so the output is bit-identical
to the serial build, not merely close.

    python scripts/noise/build_unequal_arm_cache.py --two-years --coarse-Q 150

Pass the same data/model flags the run will use; a mismatch just produces a
different key, so the run rebuilds rather than reading something wrong.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_noise_only as rn  # noqa: E402

_W = {}


def _init(settings_args, settings_kwargs, tdi_generation, fill_nans):
    """One component + settings per worker; the fold map builds once here."""
    from lisatools.domains import WDMSettings
    from lisatools.sensitivity import UnequalArmInstrumentNoise

    fine = WDMSettings(*settings_args, **settings_kwargs)
    comp = UnequalArmInstrumentNoise.__new__(UnequalArmInstrumentNoise)
    comp.wdm_psd_method = "fold"
    comp.fill_nans = fill_nans
    comp.tdi_generation = tdi_generation
    _W["fine"], _W["comp"] = fine, comp


def _chunk(payload):
    lo, rows = payload
    comp, fine = _W["comp"], _W["fine"]
    out = np.stack([comp._folded_unit_column(fine, r) for r in rows], axis=-1)
    return lo, np.asarray(out)          # (2, 3, 3, Nf_active, len(rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noise-file", default=rn.NOISE_FILE)
    p.add_argument("--out-dir", default="./noise-galfor-pe2/")
    p.add_argument("--coarse-Q", type=int, required=True)
    p.add_argument("--two-years", action="store_true")
    p.add_argument("--min-freq", type=float, default=3e-4)
    p.add_argument("--max-freq", type=float, default=8e-3)
    p.add_argument(
        "--fill-nans", type=float, default=0.0,
        help="must match CompositeSensitivityBackend.instrument_fill_nans "
        "(default 0.0). _persistent_coarse_key does NOT hash it, so a "
        "mismatch here would be loaded by the run as a silent cache hit.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--chunk", type=int, default=64)
    args = p.parse_args(argv)

    from lisatools.domains import CoarseWDMSettings, WDMSettings
    from lisatools.sensitivity import LinkDelayTable, UnequalArmInstrumentNoise

    if not args.two_years:
        raise SystemExit("only the --two-years grid is wired up here")
    nf, nt, _ = rn._two_year_grid(args.noise_file)

    class _GS:                      # what _domain_t0 needs, no engine build
        dt = rn.RUN_DT
        preprocess_kwargs = dict(trim_kwargs=dict(
            duration=rn.TRIM_DURATION, is_percent=False,
            trimming_type="from_each_end"))

    t0 = _domain = rn._domain_t0(args.noise_file, _GS())
    fine = WDMSettings(nf, nt, rn.RUN_DT, min_freq=args.min_freq,
                       max_freq=args.max_freq, force_backend="cpu")
    coarse = CoarseWDMSettings.from_fine(fine, args.coarse_Q)
    table = LinkDelayTable.from_l1_file(args.noise_file, stride=200, data_t0=t0)

    cache_dir = os.path.join(args.out_dir, "unequal_arm_coarse_cache")
    comp = UnequalArmInstrumentNoise(
        ltts=table, tdi_generation=2, fill_nans=args.fill_nans,
        coarse_cache_dir=cache_dir, wdm_psd_method="fold",
    )
    key = comp._persistent_coarse_key(coarse)
    path = os.path.join(cache_dir, f"unequal_arm_{key}.npz")
    print(f"grid Nf={nf} Nt={nt} Q={args.coarse_Q} -> Ncoarse={coarse.Ncoarse}")
    print(f"domain t0 = file t0 + {t0 - float(rn._read_xyz(args.noise_file)[1][0]):.0f} s")
    print(f"target    = {path}")
    if os.path.exists(path):
        print("already present; nothing to do")
        return 0

    ltts = comp._resolve_ltts(fine)
    idx = list(range(fine.ind_min_t, fine.ind_max_t + 1))
    jobs = [(lo, ltts[idx[lo:lo + args.chunk]])
            for lo in range(0, len(idx), args.chunk)]
    print(f"{len(idx)} columns in {len(jobs)} chunks on {args.workers} workers",
          flush=True)

    started = time.monotonic()
    columns = np.empty((2, 3, 3, fine.Nf_active, len(idx)), dtype=np.float64)
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(fine.args, fine.kwargs, 2, args.fill_nans),
    ) as pool:
        for lo, block in pool.map(_chunk, jobs):
            columns[..., lo:lo + block.shape[-1]] = block
            done += block.shape[-1]
            el = time.monotonic() - started
            print(f"  {done}/{len(idx)} ({100*done/len(idx):.1f}%), "
                  f"{el/60:.1f} min elapsed, ETA {el*(len(idx)-done)/done/60:.1f} min",
                  flush=True)

    # Same accumulation as _build_coarse_basis_data, in column order.
    coarse_sums = np.zeros((2, 3, 3, fine.Nf_active, coarse.Ncoarse))
    fine_diagonals = np.empty((2, 3, fine.Nf_active, fine.Nt_active))
    for local in range(len(idx)):
        column = columns[..., local]
        coarse_sums[..., local // args.coarse_Q] += column
        fine_diagonals[..., local] = np.real(
            np.stack([column[:, a, a] for a in range(3)], axis=1))
    coarse_sums /= np.asarray(coarse.cell_sizes, dtype=float)[None, None, None, None, :]

    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, B_oms=coarse_sums[0], B_acc=coarse_sums[1],
             fine_diagonals=fine_diagonals)
    os.replace(tmp, path)
    print(f"wrote {path}  [{(time.monotonic()-started)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
