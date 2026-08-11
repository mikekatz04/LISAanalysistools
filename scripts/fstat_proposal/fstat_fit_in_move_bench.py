#!/usr/bin/env python
"""Timing + parity bench for the in-``setup()`` F-stat grid fit on GPUs.

The GPU half of the in-move F-stat fit work
(``GB_FSTAT_FIT_IN_MOVE=1`` -> ``GBSpecialRJFStatGridMove.setup()`` fits the
comb scan / stage-B grids against the live residual). The CPU plumbing smoke
lives next door (``fstat_fit_in_move_smoke.py``); this script answers the
on-hardware questions it cannot:

* **How long does the fit inside ``setup()`` take** on a real device, split
  into comb scan / stage B / total, versus the surrounding iteration time
  (``GF_MOVE_TIMING=1`` per-move lines are enabled and parsed)?
* **Does the fit behave on a sharded 2-GPU ACA?** The F-stat scores every
  candidate against ONE reference walker, so 100% of the rows land on that
  walker's shard through ``_RoutedBandEngine.route_fstat_ll`` -- the exact
  configuration that reads foreign-device comp buffers when something is
  still shared. Expect *no speedup* from the second GPU (single-shard
  scoring by design); the multi-GPU leg is a correctness gate, not a
  performance target.
* **Do all legs fit identical grids?** The comb npz and the stacked stage-B
  npz must match across legs (same synthetic data, same reference walker,
  deterministic sweep). Bit-identical is the expectation; ``--rtol`` is the
  formal gate.

Each leg runs in a fresh process (``GPUS`` is consumed at build time and
cupy's current device is process-global), entirely through the stock API
(``erebor.gb_no_fg`` with ``GB_MODE=search``) -- no settings files, no
hand-rolled likelihood. Gates per leg: the fit runs EXACTLY once (iteration
2 must take the cheap skip path), ``DONE.json`` lands in the epoch dir,
every iteration's lnL is finite, and the log shows no cross-device error.

Usage::

    # cluster: single-GPU vs 2-GPU, tiny grid knobs (minutes)
    GPUS=0,1 python scripts/fstat_proposal/fstat_fit_in_move_bench.py \
        --preset smoke

    # cluster: production-scale timing -- ambient GB_*/FSTAT_* knobs rule
    # (set the band + grid knobs exactly as the main run will)
    GPUS=0,1 GB_CENTER_FREQ=8.5e-3 GB_N_LAYERS=25 python \
        scripts/fstat_proposal/fstat_fit_in_move_bench.py --preset ambient

    # resume a killed production fit instead of starting clean
    ... --preset ambient --resume

    # laptop plumbing check (two CPU legs, cross-leg parity of the bench
    # machinery itself)
    python scripts/fstat_proposal/fstat_fit_in_move_bench.py \
        --preset smoke --legs cpu cpu

Legs default to ``<first GPU>`` and ``<all GPUS>`` when ``GPUS`` lists two
or more devices; ``--legs`` overrides (each entry a device list, or ``cpu``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

# Tiny-grid knobs shared with the CPU smoke -- ONE definition of "smoke
# scale" for the in-move fit.
from fstat_fit_in_move_smoke import CHILD_ENV as SMOKE_ENV

#: env every leg gets regardless of preset (thread pins per the MPI-only
#: rule; per-move timing is the point of the bench).
BASE_ENV = dict(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    MPLBACKEND="Agg", GF_MOVE_TIMING="1",
)

#: the ambient preset forces only what makes the in-move fit live; every
#: FSTAT_* / GB_* scale knob comes from the caller's environment.
AMBIENT_FORCED = dict(GB_MODE="search", GB_FSTAT_FIT_IN_MOVE="1")
AMBIENT_DEFAULTED = dict(DATA_MODE="synthetic", GB_FSTAT_REFIT_EVERY="0")

RE_FIT_START = re.compile(r"F-stat grid fit epoch (\d+) starting")
RE_FIT_DONE = re.compile(
    r"F-stat grid fit epoch (\d+) done in ([0-9.]+)s \((\d+) peaks\)")
RE_COMB_SWEEP = re.compile(
    r"\[sweep:comb\.nsky\d+\] (\d+) evals in one chunked stream "
    r"\(batch=\d+, (\d+)s")
RE_STAGEB_SWEEP = re.compile(
    r"\[stageB\] (\d+) evals in one chunked stream \(batch=\d+, (\d+)s")
RE_GF_TIMING = re.compile(
    r"\[GF_TIMING\] stage=(\S+) move=(\S+) it=(\d+) wall_s=([0-9.]+)")
RE_BENCH_IT = re.compile(
    r"\[BENCH\] it=(\d+) iter_wall_s=([0-9.]+) ll_max=(\S+) finite=(\d)")
RE_DONE_JSON = re.compile(r"\[BENCH\] done_json=(.+)")

#: substrings that fail a leg outright wherever they appear in the log.
FATAL_MARKS = ("NotImplementedError", "peer access", "PeerAccess",
               "different device", "Traceback (most recent call last)")

#: npz keys compared across legs (comb, stacked) -- the grid IS the
#: deliverable of the fit, so parity is judged on it directly.
COMB_KEYS = ("f0_nodes_mHz", "F_max", "best_alpha", "best_sin_delta",
             "peaks")
STACKED_KEYS = ("logp_grids", "f0_los", "f0_dxs", "peak_f0_mHz", "peak_F",
                "band_idx")


# ---------------------------------------------------------------- child


def run_leg(iterations: int) -> int:
    """One leg under the ambient env: build the search-mode gb_no_fg fit and
    run ``iterations`` sampler steps; the in-move fit fires inside the first
    GB RJ ``setup()``."""
    import logging
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s")

    import numpy as np

    from lisatools.globalfit.stock import erebor
    from lisatools.utils.utility import asnumpy

    lite = os.environ.get("FSTAT_BENCH_LITE", "0") == "1"
    fit = erebor.gb_no_fg(lite=lite)
    if not fit.gb.fstat_fit_in_move:
        print("[BENCH] FATAL: GB_FSTAT_FIT_IN_MOVE did not resolve on "
              "fit.gb.fstat_fit_in_move", flush=True)
        return 2
    gpus = getattr(fit.general, "gpus", None)
    print(f"[BENCH] gpus={gpus} lite={int(lite)} "
          f"fit_dir={fit.gb.fstat_fit_dir!r} "
          f"data_mode={fit.general.data_mode!r}", flush=True)

    ok = True
    t = time.perf_counter()
    for i, (_model, state) in enumerate(fit.sample(int(iterations))):
        dt = time.perf_counter() - t
        ll = float(np.max(asnumpy(state.log_like)))
        finite = bool(np.isfinite(ll) and ll > -1e290)
        ok = ok and finite
        print(f"[BENCH] it={i} iter_wall_s={dt:.2f} ll_max={ll:.9e} "
              f"finite={int(finite)}", flush=True)
        t = time.perf_counter()

    done = sorted(glob.glob(
        os.path.join(fit.gb.fstat_fit_dir, "*", "epoch_*", "DONE.json")))
    for p in done:
        print(f"[BENCH] done_json={p}", flush=True)
    if not done:
        print("[BENCH] FATAL: no DONE.json under the fit dir -- the "
              "in-move fit never completed", flush=True)
        return 3
    return 0 if ok else 1


# --------------------------------------------------------------- driver


def _leg_env(preset: str, leg_gpus: str, leg_dir: str) -> dict:
    env = dict(os.environ)
    env.update(BASE_ENV)
    if preset == "smoke":
        env.update(SMOKE_ENV)          # forced: tiny knobs ARE the preset
        env["FSTAT_BENCH_LITE"] = "1"
    else:
        env.update(AMBIENT_FORCED)
        for k, v in AMBIENT_DEFAULTED.items():
            env.setdefault(k, v)
    env["GB_FSTAT_FIT_DIR"] = os.path.join(leg_dir, "fitdir")
    env["FILE_STORE_DIR"] = os.path.join(leg_dir, "store") + os.sep
    if leg_gpus == "cpu":
        env.pop("GPUS", None)
    else:
        env["GPUS"] = leg_gpus
    return env


def _spawn(label: str, env: dict, iterations: int):
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--iterations", str(iterations)]
    print(f"\n=== leg {label}: GPUS={env.get('GPUS', '(cpu)')} ===",
          flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line.rstrip("\n"))
    proc.wait()
    return proc.returncode, lines


def _parse_leg(rc: int, lines: list) -> dict:
    text = "\n".join(lines)
    r = dict(rc=rc, fatal=[m for m in FATAL_MARKS if m in text])
    r["fit_starts"] = len(RE_FIT_START.findall(text))
    m = RE_FIT_DONE.search(text)
    r["fit_wall_s"] = float(m.group(2)) if m else None
    r["n_peaks"] = int(m.group(3)) if m else None
    r["comb_evals"] = sum(int(a) for a, _ in RE_COMB_SWEEP.findall(text))
    r["comb_s"] = sum(int(b) for _, b in RE_COMB_SWEEP.findall(text))
    r["stageb_evals"] = sum(int(a) for a, _ in RE_STAGEB_SWEEP.findall(text))
    r["stageb_s"] = sum(int(b) for _, b in RE_STAGEB_SWEEP.findall(text))
    r["iters"] = {int(i): float(w)
                  for i, w, _ll, _f in RE_BENCH_IT.findall(text)}
    r["move_walls"] = {}
    for stage, move, it, wall in RE_GF_TIMING.findall(text):
        r["move_walls"].setdefault((stage, move), {})[int(it)] = float(wall)
    r["done_json"] = RE_DONE_JSON.findall(text)
    return r


def _epoch_dir(leg_dir: str):
    hits = sorted(glob.glob(
        os.path.join(leg_dir, "fitdir", "*", "epoch_*")))
    return hits[-1] if hits else None


def _compare_npz(path_a: str, path_b: str, keys, rtol: float):
    """Per-key comparison -> list of (key, status, max_rel) tuples."""
    import numpy as np

    out = []
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    for k in keys:
        if k not in a or k not in b:
            out.append((k, "MISSING", None))
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out.append((k, f"SHAPE {x.shape} vs {y.shape}", None))
            continue
        if np.array_equal(x, y):
            out.append((k, "bit-identical", 0.0))
            continue
        denom = np.maximum(np.abs(x), np.abs(y))
        denom[denom == 0] = 1.0
        rel = float(np.max(np.abs(x - y) / denom))
        out.append((k, "ok" if rel <= rtol else "DIFF", rel))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=("smoke", "ambient"),
                    default="smoke",
                    help="smoke = tiny shared knobs (minutes); ambient = "
                         "the caller's env sets every scale knob")
    ap.add_argument("--legs", nargs="+", default=None,
                    help="device list per leg (e.g. 0 0,1) or 'cpu'; "
                         "default: first GPU, then all of GPUS")
    ap.add_argument("--iterations", type=int, default=2,
                    help="sampler iterations per leg (>=2 proves the "
                         "iteration-2 skip path)")
    ap.add_argument("--out", default="fstat_fit_bench_out",
                    help="scratch root; one subdir per leg")
    ap.add_argument("--rtol", type=float, default=1e-12,
                    help="cross-leg grid parity gate")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing leg dirs (mid-fit checkpoint "
                         "resume) instead of starting clean")
    ap.add_argument("--child", action="store_true",
                    help="internal: run ONE leg under the ambient env")
    args = ap.parse_args()

    if args.child:
        sys.exit(run_leg(args.iterations))

    legs = args.legs
    if legs is None:
        gpus = [d for d in os.environ.get("GPUS", "").split(",")
                if d.strip()]
        if len(gpus) >= 2:
            legs = [gpus[0], ",".join(gpus)]
        elif len(gpus) == 1:
            legs = [gpus[0]]
        else:
            legs = ["cpu"]
    labels = []
    for j, leg in enumerate(legs):
        base = "cpu" if leg == "cpu" else f"gpu{leg}"
        labels.append(base if base not in labels else f"{base}_{j}")

    os.makedirs(args.out, exist_ok=True)
    results, epoch_dirs = {}, {}
    for label, leg in zip(labels, legs):
        leg_dir = os.path.abspath(os.path.join(args.out, label))
        if not args.resume and os.path.isdir(leg_dir):
            shutil.rmtree(leg_dir)
        os.makedirs(leg_dir, exist_ok=True)
        rc, lines = _spawn(
            label, _leg_env(args.preset, leg, leg_dir), args.iterations)
        results[label] = _parse_leg(rc, lines)
        epoch_dirs[label] = _epoch_dir(leg_dir)

    # ---- per-leg gates + timing table ---------------------------------
    fails = []
    print("\n=== in-move F-stat fit bench summary ===")
    hdr = (f"{'leg':12s} {'fit_wall_s':>10s} {'comb_s':>7s} "
           f"{'stageB_s':>8s} {'evals/s':>9s} {'peaks':>5s} "
           f"{'it0_s':>8s} {'it1_s':>8s}")
    print(hdr)
    for label, r in results.items():
        if r["rc"] != 0:
            fails.append(f"{label}: leg exit code {r['rc']}")
        if r["fatal"]:
            fails.append(f"{label}: fatal marks in log: {r['fatal']}")
        if r["fit_starts"] != 1:
            fails.append(
                f"{label}: fit started {r['fit_starts']}x (expected exactly "
                "once -- refit or cross-move duplicate)")
        evals = r["comb_evals"] + r["stageb_evals"]
        secs = r["comb_s"] + r["stageb_s"]
        rate = f"{evals / secs:9.0f}" if secs else "      n/a"
        fmt = lambda v, w: f"{v:{w}.1f}" if v is not None else " " * (w - 3) + "n/a"
        print(f"{label:12s} {fmt(r['fit_wall_s'], 10)} "
              f"{r['comb_s']:7d} {r['stageb_s']:8d} {rate} "
              f"{str(r['n_peaks']):>5s} "
              f"{fmt(r['iters'].get(0), 8)} {fmt(r['iters'].get(1), 8)}")
    # the per-move split, for the fit-vs-iteration budget question
    for label, r in results.items():
        for (stage, move), walls in sorted(r["move_walls"].items()):
            if move == "__total__":
                continue
            w = " ".join(f"it{k}={v:.1f}s" for k, v in sorted(walls.items()))
            print(f"    {label}: [{stage}] {move}: {w}")

    # ---- cross-leg grid parity ----------------------------------------
    if len(labels) > 1:
        ref = labels[0]
        if epoch_dirs[ref] is None:
            fails.append(f"{ref}: no epoch dir; cannot compare grids")
        for other in labels[1:]:
            if epoch_dirs[other] is None:
                fails.append(f"{other}: no epoch dir; cannot compare grids")
                continue
            print(f"\n--- grid parity: {ref} vs {other} "
                  f"(rtol {args.rtol:g}) ---")
            for fname, keys in (("fstat_grid_comb.npz", COMB_KEYS),
                                ("fstat_grid_peaks_stacked.npz",
                                 STACKED_KEYS)):
                pa = os.path.join(epoch_dirs[ref], fname)
                pb = os.path.join(epoch_dirs[other], fname)
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    fails.append(f"{other}: {fname} missing on one leg")
                    continue
                for k, status, rel in _compare_npz(pa, pb, keys, args.rtol):
                    rel_s = f"max_rel={rel:.3e}" if rel else ""
                    print(f"  {fname}:{k:16s} {status} {rel_s}")
                    if status not in ("bit-identical", "ok"):
                        fails.append(f"{other}: {fname}:{k} {status} {rel_s}")

    print()
    if fails:
        print("[BENCH] FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"[BENCH] ALL GATES PASSED ({len(labels)} leg(s); scratch under "
          f"{args.out})")


if __name__ == "__main__":
    main()
