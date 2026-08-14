#!/usr/bin/env python
"""GATE: rj/in-model GPU concurrency (GIL-release wraps + per-call uploads).

Runs the SAME seeded narrow-band rj search twice on this allocation --
leg A with serial shard dispatch (GB_ROUTER_THREADED=0), leg B with
concurrent lanes (=1) -- and hard-compares the stored chains bit for
bit. Threading only overlaps deterministic per-shard kernel work (every
random draw happens on the main thread from the seeded stream), so ANY
difference is a concurrency bug: a race in a guarded wrap, shared state
the audit missed, or the per-call device-struct upload changing
numerics. The run's own alarms (cold [GB_CELL_LL] excesses, ll drift,
CUDA errors) are scanned as a second layer.

Usage on a 2-GPU node in the run env (LAT >= 42d546f, GBGPU wheel
rebuilt from >= a54e8d3 -- CHECK `pip show gbgpu` for the new suffix):

    GPUS=0,1 python scripts/gb_chunked_het/gate_rj_concurrency.py

Optional env: GATE_OUT_DIR (default ./gate_rj_conc), NITER (default 4),
GB_MIN_FREQ/GB_MAX_FREQ (default 7.0e-3/7.8e-3), NWALKERS/NTEMPS
(runner defaults 16/6). Wall: roughly 2x (fit ~2-3 min + NITER
iterations) -- ~15-25 min total.

Exit 0 + "GATE: PASS" = restart production on the new wheels.
"""
import os
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

BASE = os.environ.get("GATE_OUT_DIR", "./gate_rj_conc")
RUNNER = os.path.join(os.path.dirname(__file__), "..", "fstat_proposal",
                      "run_fstat_rj_search.py")

COMPARE_GLOBAL = ("chain/gb", "inds/gb", "log_like", "log_prior")
COMPARE_SUB = ("gb/chain", "gb/inds", "gb/band_cold_ll", "gb/band_temps",
               "gb/band_leaf_cap", "gb/d_h", "gb/h_h")


LEG_STATS = {}


def run_leg(name: str, threaded: str) -> str:
    out = os.path.join(BASE, name)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "GB_ROUTER_THREADED": threaded,
        "FIT_DIR": out + "/",
        "NITER": os.environ.get("NITER", "4"),
        "GB_MIN_FREQ": os.environ.get("GB_MIN_FREQ", "7.0e-3"),
        "GB_MAX_FREQ": os.environ.get("GB_MAX_FREQ", "7.8e-3"),
        # identical fstat path in both legs; the fan-out is separately
        # gated bit-identical, so leave whatever the env pins.
    })
    print(f"[gate] leg {name}: GB_ROUTER_THREADED={threaded} -> {out}")
    # Built-in nvidia-smi sampler (5 s cadence) so utilization evidence is
    # automatic -- no one has to remember to watch a terminal.
    smi = None
    csv_path = os.path.join(out, "gpu_util_gate.csv")
    if shutil.which("nvidia-smi"):
        smi = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=timestamp,index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-l", "5"],
            stdout=open(csv_path, "w"), stderr=subprocess.DEVNULL)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, RUNNER], env=env)
    finally:
        if smi is not None:
            smi.terminate()
    wall = time.perf_counter() - t0
    util = {}
    if os.path.exists(csv_path):
        for line in open(csv_path):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].isdigit():
                util.setdefault(int(parts[1]), []).append(float(parts[2]))
    LEG_STATS[name] = (wall, {d: float(np.mean(u)) for d, u in util.items()})
    us = "  ".join(f"dev{d} mean {m:.0f}%"
                   for d, m in sorted(LEG_STATS[name][1].items()))
    print(f"[gate] leg {name}: wall {wall:.0f} s  {us or '(no smi data)'}")
    if r.returncode != 0:
        print(f"GATE: FAIL -- leg {name} exited {r.returncode}")
        sys.exit(1)
    return out


def h5path(leg_dir: str) -> str:
    for root, _, fns in os.walk(leg_dir):
        for fn in fns:
            if fn.endswith(".h5") and "testing" in fn:
                return os.path.join(root, fn)
    raise FileNotFoundError(f"no backend h5 under {leg_dir}")


def filled_iters(f) -> int:
    ll = f["global_fit"]["log_like"][:, 0, 0, :]
    nz = np.where(np.any(ll != 0.0, axis=1))[0]
    return int(nz.max()) + 1 if nz.size else 0


def compare(pa: str, pb: str) -> bool:
    ok = True
    with h5py.File(pa) as fa, h5py.File(pb) as fb:
        na, nb = filled_iters(fa), filled_iters(fb)
        print(f"[gate] filled iterations: serial={na} threaded={nb}")
        if na != nb or na == 0:
            print("GATE: FAIL -- iteration counts differ (or zero).")
            return False
        for grp, keys in (("global_fit", COMPARE_GLOBAL),
                          ("global_fit/sub_backend", COMPARE_SUB)):
            for k in keys:
                if f"{grp}/{k}" not in fa:
                    continue
                a = fa[f"{grp}/{k}"][:na]
                b = fb[f"{grp}/{k}"][:na]
                if np.array_equal(a, b):
                    print(f"  {k}: IDENTICAL")
                    continue
                ok = False
                d = np.argwhere(a != b)
                fa_ = a[tuple(d[0])] if d.size else None
                fb_ = b[tuple(d[0])] if d.size else None
                mx = (np.nanmax(np.abs(np.asarray(a, float)
                                       - np.asarray(b, float)))
                      if a.dtype != bool else int((a != b).sum()))
                print(f"  {k}: DIFFERS -- {len(d)} elements, first at "
                      f"{tuple(d[0])} ({fa_!r} vs {fb_!r}), max|d|={mx}")
    return ok


def scan_log(leg_dir: str, name: str) -> bool:
    ok = True
    logp = None
    for root, _, fns in os.walk(leg_dir):
        for fn in fns:
            if fn == "globalfit_run.log":
                logp = os.path.join(root, fn)
    if logp is None:
        print(f"[gate] {name}: no globalfit_run.log found (non-fatal)")
        return True
    txt = open(logp, errors="replace").read()
    cuda = len(re.findall(r"cudaError|CUDARuntimeError|GPUassert", txt))
    cold_cell = len(re.findall(
        r"GB_CELL_LL.*exceeds its temperature-scaled allowance", txt))
    drifts = [float(x) for x in re.findall(
        r"incremental ll drift ([\d.eE+-]+)", txt)]
    big_drift = sum(1 for x in drifts if x > 1e-2)
    print(f"[gate] {name}: cuda-errors={cuda} cold-cell-ll-warnings="
          f"{cold_cell} drift-lines={len(drifts)} (>1e-2: {big_drift})")
    if cuda:
        print(f"GATE: FAIL -- CUDA errors in leg {name}.")
        ok = False
    if cold_cell:
        print(f"GATE: FAIL -- allowance-scaled [GB_CELL_LL] excess in "
              f"leg {name} (post-guard these are real).")
        ok = False
    return ok


def main():
    os.makedirs(BASE, exist_ok=True)
    a = run_leg("serial", "0")
    b = run_leg("threaded", "1")
    ok = compare(h5path(a), h5path(b))
    ok = scan_log(a, "serial") and ok
    ok = scan_log(b, "threaded") and ok
    if "serial" in LEG_STATS and "threaded" in LEG_STATS:
        ws, wt = LEG_STATS["serial"][0], LEG_STATS["threaded"][0]
        print(f"[gate] speedup (whole leg incl. fit/build): "
              f"{ws / max(wt, 1e-9):.2f}x  (serial {ws:.0f} s -> "
              f"threaded {wt:.0f} s)")
    print("GATE: PASS" if ok else "GATE: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
