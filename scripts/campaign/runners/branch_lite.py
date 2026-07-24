#!/usr/bin/env python
"""Generic campaign runner: build a stock variant and sample it, printing one
``[RESULT]`` line per iteration for the ledger.

Everything flows through the stock structure: ``erebor.<variant>()`` →
``build()`` → ``fit.sample()``.  Knobs arrive via env (GB_MODE, GB_DOMAIN,
{BRANCH}_DEBUG, USE_GPU/GPUS, ...) or kwargs — the stock precedence chain is
untouched.  Defaults to the lite preset; ``--full`` runs production settings.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time


def _count_debug_pngs() -> int:
    dirs = {
        v
        for k, v in os.environ.items()
        if k.endswith("_DEBUG_DIR") and v
    }
    n = 0
    for d in dirs:
        n += len(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--lite", action="store_true", help="(default) lite preset")
    ap.add_argument("--full", action="store_true", help="production settings")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from lisatools.globalfit.stock import erebor

    name = args.variant
    kwargs = {}
    if not args.full and not name.endswith("_lite") and name != "blank":
        kwargs["lite"] = True
    fit = getattr(erebor, name)(**kwargs)
    t_build = time.perf_counter()
    fit.build()
    print(f"[RESULT] variant={name} build=ok build_s={time.perf_counter() - t_build:.1f}",
          flush=True)

    all_finite = True
    t_prev = time.perf_counter()
    for i, (model, state) in enumerate(fit.sample(iterations=args.iterations)):
        ll = float(state.log_like.max())
        wall = time.perf_counter() - t_prev
        t_prev = time.perf_counter()
        finite = math.isfinite(ll) and ll > -1e290
        all_finite = all_finite and finite
        print(f"[RESULT] it={i} ll_max={ll:.6e} it_wall_s={wall:.2f}", flush=True)

    n_png = _count_debug_pngs()
    print(
        f"[RESULT] variant={name} iterations={args.iterations} "
        f"ll_finite={int(all_finite)} debug_pngs={n_png}",
        flush=True,
    )
    sys.exit(0 if all_finite else 1)


if __name__ == "__main__":
    main()
