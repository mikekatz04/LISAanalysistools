#!/usr/bin/env python
"""1-GPU vs N-GPU parity gate for the GB sig-het and F-stat paths.

The on-hardware half of the multi-GPU work whose structural half lives in
``tests/test_gb_shard_router.py`` (per-device comp/engine replicas, the
per-slot ``slab_min_f`` shard slice, the sig-het per-shard reference stash).
Those CPU fake-shard tests can prove the ROUTING; only a real multi-GPU node
can prove that a shard's kernels read device-local buffers.

Two arms, both driven entirely through the stock API
(``erebor.<variant>()`` -> ``fit.build()``) -- no settings files, no
hand-rolled likelihood:

* ``sighet`` -- the VGB variant with ``GB_SIGHET_INMODEL=1``. This is the
  path that used to raise ``NotImplementedError`` on any multi-shard buffer
  ("sig-het in-model references hold single-shard state"); with per-device
  comp replicas each shard builds its own heterodyne reference stash, its own
  slot->reference map and its own ``_in_model`` flag.
* ``fstat`` -- the gb_no_fg variant with ``GB_MODE=search``, which makes
  ``rj_fstat_dist_birth`` live so ``_RoutedBandEngine.route_fstat_ll`` runs
  for real. The in-fit F-stat scores every candidate against ONE reference
  walker, so 100% of the rows land on that walker's shard -- which is exactly
  the configuration that reads foreign-device comp buffers when the comp is
  shared. This arm has never been exercised on two GPUs.

**The gate.** Each arm runs twice, in separate processes (``GPUS`` is
consumed at build time and cupy's current device is process-global), and the
per-walker INITIAL likelihood -- ``sampler.analysis_container_arr.likelihood()``
straight after ``prepare_main()``, before any move -- must be **bit-identical**
between the two. That is the bar the all_sources multi-GPU work met; sharding
splits walkers, never the frequency axis, so each walker's likelihood is
computed by exactly one shard and there is no reduction-order excuse for a
difference. The iterations that follow are a liveness check only (finite
``ll``, no ``NotImplementedError``, no cupy peer-access error); their exact
values are not compared, because the proposal RNG stream is not guaranteed
identical across shard counts.

Exits non-zero on the first failing arm.

Env knobs (stock convention: an env knob is the capitalized attribute name,
per-branch blocks prefixed with the branch namespace)::

    GPUS                 device list for the MULTI-GPU leg (default "0,1")
    GB_MULTIGPU_BASE_GPU device for the single-GPU leg  (default the first
                         entry of GPUS)
    GB_SIGHET_INMODEL    forced to 1 by the ``sighet`` arm
    GB_MODE              forced to ``search`` by the ``fstat`` arm
    NUM_ITERATIONS       liveness iterations per leg (default 2)
    DATA_MODE            stock data pipeline selector (default the variant's)
    FSTAT_GRID_DIR       peak-grid directory for the fstat arm's births; when
                         unset the arm still runs (comb/floor births only)

Usage::

    # both arms, GPU 0 vs GPUs 0+1
    GPUS=0,1 python scripts/gb_chunked_het/gb_multigpu_parity.py

    # one arm, explicit devices
    GPUS=2,3 python scripts/gb_chunked_het/gb_multigpu_parity.py --arms sighet

    # the per-leg worker (spawned by the driver; runnable by hand to debug)
    GPUS=0,1 python scripts/gb_chunked_het/gb_multigpu_parity.py \
        --child sighet
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------- arms

#: arm -> (stock variant, env forced ON for that arm)
ARMS = {
    "sighet": ("vgb", {"GB_SIGHET_INMODEL": "1"}),
    "fstat": ("gb_no_fg", {"GB_MODE": "search"}),
}

#: printed by the child, parsed by the driver
LL_PREFIX = "[PARITY] initial_ll_hex="
IT_PREFIX = "[PARITY] it="
DONE_PREFIX = "[PARITY] arm_done="


# ---------------------------------------------------------------- child


def run_leg(arm: str, iterations: int) -> int:
    """One leg: build the arm's stock fit under the ambient ``GPUS`` and print
    the initial per-walker likelihood plus ``iterations`` liveness steps."""
    import logging

    # Without this the leg is SILENT between the initial-lnL print and the
    # first completed iteration -- at full-band search scale that is tens of
    # minutes of nothing, indistinguishable from a hang.
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s")

    import numpy as np

    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor
    from lisatools.utils.utility import asnumpy

    variant, forced = ARMS[arm]
    for k, v in forced.items():
        os.environ[k] = v

    fit = getattr(erebor, variant)()
    fit.general.num_iterations = int(iterations)

    # A stale HDF checkpoint would resume instead of initializing from
    # priors, and the two legs must start from the SAME state for the parity
    # comparison to mean anything. Removed BEFORE the build, which is what
    # opens the backend (same pattern as scripts/campaign/runners/branch_lite).
    g = fit.general
    backend_path = f"{g.file_store_dir}{g.base_file_name}_{g.main_file_key}.h5"
    if os.path.exists(backend_path):
        os.remove(backend_path)
        print(f"[PARITY] removed stale backend {backend_path}", flush=True)
    fit.build()

    # ``prepare_main`` is what builds the sampler + the (possibly sharded)
    # AnalysisContainerArray; its likelihood is the pre-move initial lnL --
    # the same quantity the all_sources 1-vs-2-GPU gate compared.
    gf = GlobalFit(fit, comm=None)
    gf.prepare_main()
    ll0 = np.asarray(
        asnumpy(gf.sampler.analysis_container_arr.likelihood(complex=False))
    ).ravel()
    gpus = getattr(fit.general, "gpus", None)
    print(f"[PARITY] arm={arm} variant={variant} gpus={gpus} "
          f"nwalkers={ll0.size}", flush=True)
    # float.hex() is exact: a 1-ulp difference is visible, a formatted decimal
    # would hide it.
    print(LL_PREFIX + ",".join(float(v).hex() for v in ll0), flush=True)

    ok = True
    for i, (_model, state) in enumerate(fit.sample(iterations=int(iterations))):
        ll = float(np.max(asnumpy(state.log_like)))
        finite = np.isfinite(ll) and ll > -1e290
        ok = ok and bool(finite)
        print(f"{IT_PREFIX}{i} ll_max={ll:.9e} finite={int(finite)}",
              flush=True)
    print(f"{DONE_PREFIX}{arm} ok={int(ok)}", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------- driver


def _spawn(arm: str, gpus: str, iterations: int) -> tuple[int, list[str]]:
    """Run one leg in a fresh process under ``GPUS=gpus``; return (rc, lines)."""
    env = dict(os.environ)
    env["GPUS"] = gpus
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    # Progress visibility: since loginfo's propagate=False rework (7d63b95)
    # lisatools INFO lines reach the console only via the run's verbose
    # knob -- the child's logging.basicConfig alone no longer shows them.
    env.setdefault("VERBOSE", "1")
    env.setdefault("PROGRESS", "0")
    env["NUM_ITERATIONS"] = str(iterations)
    cmd = [sys.executable, os.path.abspath(__file__), "--child", arm,
           "--iterations", str(iterations)]
    print(f"\n=== {arm}: GPUS={gpus} ===", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line.rstrip("\n"))
    proc.wait()
    return proc.returncode, lines


def _extract(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def compare_arm(arm: str, single_gpus: str, multi_gpus: str,
                iterations: int) -> bool:
    """Run both legs of one arm and report PASS/FAIL."""
    rc1, out1 = _spawn(arm, single_gpus, iterations)
    rc2, out2 = _spawn(arm, multi_gpus, iterations)

    if rc1 != 0 or rc2 != 0:
        print(f"[GATE] {arm}: FAIL -- leg exit codes "
              f"single={rc1} multi={rc2}", flush=True)
        return False

    ll1 = _extract(out1, LL_PREFIX)
    ll2 = _extract(out2, LL_PREFIX)
    if ll1 is None or ll2 is None:
        print(f"[GATE] {arm}: FAIL -- no initial likelihood reported "
              f"(single={ll1 is not None} multi={ll2 is not None})", flush=True)
        return False
    if ll1 != ll2:
        a = ll1.split(",")
        b = ll2.split(",")
        print(f"[GATE] {arm}: FAIL -- initial lnL NOT bit-identical", flush=True)
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"          walker {i}: {float.fromhex(x):.17e} != "
                      f"{float.fromhex(y):.17e}", flush=True)
        if len(a) != len(b):
            print(f"          walker count differs: {len(a)} vs {len(b)}",
                  flush=True)
        return False

    n = len(ll1.split(","))
    print(f"[GATE] {arm}: PASS -- initial lnL bit-identical over {n} walkers, "
          f"{iterations} iteration(s) live on both legs", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"comma-separated subset of {sorted(ARMS)}")
    ap.add_argument("--iterations", type=int,
                    default=int(os.environ.get("NUM_ITERATIONS", 2)),
                    help="liveness iterations per leg")
    ap.add_argument("--child", default=None,
                    help="internal: run ONE leg under the ambient GPUS")
    args = ap.parse_args()

    if args.child is not None:
        sys.exit(run_leg(args.child, args.iterations))

    gpus_multi = os.environ.get("GPUS", "0,1").strip()
    devices = [d for d in gpus_multi.split(",") if d.strip()]
    if len(devices) < 2:
        sys.exit(f"GPUS={gpus_multi!r} lists {len(devices)} device(s); the "
                 "parity gate needs at least two (e.g. GPUS=0,1).")
    gpus_single = os.environ.get("GB_MULTIGPU_BASE_GPU", devices[0]).strip()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")

    results = {
        arm: compare_arm(arm, gpus_single, gpus_multi, args.iterations)
        for arm in arms
    }
    print("\n=== multi-GPU parity summary ===", flush=True)
    for arm, ok in results.items():
        print(f"  {arm:8s} {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
