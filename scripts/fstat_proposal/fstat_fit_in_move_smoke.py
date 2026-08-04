#!/usr/bin/env python
"""CPU smoke for the in-move F-stat grid fit (``GB_FSTAT_FIT_IN_MOVE=1``).

Exercises :class:`GBSpecialRJFStatGridMove` on the stock API: a tiny
3-layer synthetic ``gb_no_fg`` run whose RJ birth move fits its OWN comb /
peak grids inside ``setup()`` on the first iteration -- no offline
``plot_fstat_proposal_mojito.py`` prep.

Three phases, each a fresh process:

* **fit**    -- empty fit dir; asserts the grid is built during iteration 1,
  that ``epoch_0000`` ends up with both npz caches + ``DONE.json``, and that
  iteration 2 takes the cheap path (no second fit).
* **load**   -- same fit dir; asserts the complete epoch is LOADED, with no
  comb/stage-B sweep at all.
* **resume** -- deletes the stage-B npz but keeps the comb npz; asserts the
  comb is reused and only stage B re-runs.

Run:  python scripts/fstat_proposal/fstat_fit_in_move_smoke.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

CHILD_ENV = dict(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    GB_MODE="search", GB_FSTAT_FIT_IN_MOVE="1", DATA_MODE="synthetic",
    GB_NWALKERS="2", GB_NTEMPS="2", GB_CENTER_FREQ="8.5e-3", GB_N_LAYERS="3",
    # tiny grid knobs: the point is the plumbing, not the science
    FSTAT_BATCH="2048", FSTAT_F0_SPACING_MHZ="0.002", FSTAT_COMB_NSKY="4",
    FSTAT_PEAKS_PER_BAND="2", FSTAT_N_MC="2", FSTAT_N_ALPHA="2",
    FSTAT_N_SINDELTA="2", FSTAT_PEAK_HALF_MHZ="0.01", FSTAT_PEAK_MIN_F="0.5",
)

CHILD = r'''
import logging, os
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
from lisatools.globalfit.stock import erebor
fit = erebor.gb_no_fg(lite=True)
assert fit.gb.fstat_fit_in_move, "GB_FSTAT_FIT_IN_MOVE did not resolve"
n = 0
for model, state in fit.sample(2):
    n += 1
    print(f"[child] iteration {n} complete", flush=True)
'''


def _run(tag, fit_dir, store_dir):
    env = {**os.environ, **CHILD_ENV,
           "GB_FSTAT_FIT_DIR": fit_dir, "FILE_STORE_DIR": store_dir}
    os.makedirs(store_dir, exist_ok=True)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env,
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(f"[smoke] phase {tag}: exit={p.returncode}")
    if p.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"[smoke] phase {tag} FAILED")
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="fstat_fit_in_move_")
    fit_dir = os.path.join(tmp, "fitdir")
    print(f"[smoke] scratch: {tmp}")
    fails = []

    # --- phase 1: fit from empty -------------------------------------------
    out = _run("fit", fit_dir, os.path.join(tmp, "run1"))
    if "F-stat grid fit epoch 0 starting" not in out:
        fails.append("phase fit: the in-move grid fit never ran")
    if out.count("F-stat grid fit epoch") < 2:   # starting + done
        fails.append("phase fit: the fit did not complete")
    if out.count("grid fit epoch 0 starting") != 1:
        fails.append("phase fit: refit on iteration 2 (should be once only)")
    if "[child] iteration 2 complete" not in out:
        fails.append("phase fit: did not reach iteration 2")
    # Any OTHER move sharing the fit dir must reuse, never refit.
    if out.count("grid fit epoch 0 starting") > 1:
        fails.append("phase fit: a second move re-fitted the shared grid")

    # Grids are SHARED across moves by default -> <fit_dir>/shared.
    root = None
    for cand in ("shared", "rj_prior_search", "rj_prior"):
        d = os.path.join(fit_dir, cand, "epoch_0000")
        if os.path.isdir(d):
            root = d
            break
    if root is None:
        fails.append("phase fit: no epoch_0000 directory")
    else:
        have = set(os.listdir(root))
        for need in ("DONE.json", "fstat_grid_comb.npz",
                     "fstat_grid_peaks_stacked.npz"):
            if need not in have:
                fails.append(f"phase fit: {need} missing from epoch_0000")
        print(f"[smoke] epoch_0000 contents: {sorted(have)}")

    # --- phase 2: restart loads the complete epoch --------------------------
    out = _run("load", fit_dir, os.path.join(tmp, "run2"))
    if "loading complete F-stat grid epoch 0" not in out:
        fails.append("phase load: the complete epoch was not loaded")
    if "grid fit epoch 0 starting" in out:
        fails.append("phase load: refit despite a complete epoch on disk")
    if "evals in one chunked stream" in out:
        fails.append("phase load: entered the kernel despite a complete epoch")

    # --- phase 3: comb reused, stage B re-runs ------------------------------
    if root is not None:
        os.remove(os.path.join(root, "fstat_grid_peaks_stacked.npz"))
        os.remove(os.path.join(root, "DONE.json"))
        out = _run("resume", fit_dir, os.path.join(tmp, "run3"))
        if "reusing comb cache" not in out:
            fails.append("phase resume: comb cache was not reused")
        if "[stageB]" not in out:
            fails.append("phase resume: stage B did not re-run")
        if "[comb] total" in out:
            fails.append("phase resume: comb scan re-ran (cache ignored)")

    if fails:
        print("\n[smoke] FAILURES:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"\n[smoke] ALL CHECKS PASSED (scratch under {tmp})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
