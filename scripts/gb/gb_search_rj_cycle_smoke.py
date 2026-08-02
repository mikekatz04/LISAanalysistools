#!/usr/bin/env python
"""CPU smoke for the GB search RJ cycle (replace + prior-removal + iter-only caps).

Exercises, on the stock API (``erebor.gb_no_fg``, ``data_mode="synthetic"``,
``GB_MODE=search``, tiny 2x2 sampler), the 2026-08-01 additions:

* ``GB_SEARCH_RJ_REPLACE=1``   -> ``rj_replace`` (fixed-dimension replacement,
  scored through ``SubBandBuffer.get_replace_ll``);
* ``GB_SEARCH_PRIOR_REMOVAL=1``-> ``rj_prior_removal`` (removal-only pruning,
  births force-rejected);
* ``GB_LEAF_CAP_ITER_ONLY=1``  -> fixed-schedule leaf-cap advancement;
* the guaranteed per-iteration cycle order fstat-birth -> replace -> removal
  (one Stage -> one GFCombineMove -> sequential eryn CombineMove.propose),
  read back from the ``[GF_TIMING]`` lines.

The prior container stands in for the fstat birth container (acceptable for
the smoke); a single LOUD synthetic GB is injected in the sampled layer so
births stick quickly and the replace/removal steps see alive leaves.

Two subprocess phases (fresh env resolution each):

* Phase A (honest run, ``GB_REPLACE_DEBUG=1``): asserts the cycle order, the
  expose/score/restore round-trip of ``get_replace_ll`` (residual rows
  BIT-IDENTICAL before the call vs after -- exactly the rejected-replacement
  invariant), and >=1 iteration-only leaf-cap increment.
* Phase B (adds ``GB_REPLACE_FORCE_ACCEPT=1``, debug-only knob): forces a
  replacement acceptance so the accepted-swap residual identity
  ``r_after == r_before + h_old - h_new`` (direct fill_template comparison
  on one cell) is asserted deterministically.

Run:  python scripts/gb/gb_search_rj_cycle_smoke.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

CYCLE = ["rj_prior", "rj_replace", "rj_prior_removal"]

ROUND_TRIP_MSG = "expose/score/restore round-trip bit-identical"
SWAP_IDENTITY_MSG = "accepted-swap residual identity OK"
CAP_INC_MSG = "leaf cap incremented for bands"


def _child(phase: str, iterations: int) -> None:
    """Runs inside the subprocess: build the tiny fit and sample it."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    import numpy as np

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(lite=True)
    # Narrow 3-layer band centered on the injection; f0_lims resolve to the
    # central layer, so every birth draw lands in the one interior band.
    fit.gb.center_freq = 8.5e-3
    fit.gb.n_layers = 3
    # One LOUD synthetic source (100x the stock table's amplitude) so
    # amp/phase-maximized births near its f0 clear the SNR clamp fast.
    fit.general.gb_injection_params = np.array(
        [[3.0e-21, 8.5e-3, 5.0e-15, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6]]
    )
    print(
        f"[SMOKE] phase={phase} mode={fit.gb.mode} "
        f"replace={fit.gb.search_rj_replace} "
        f"removal={fit.gb.search_prior_removal} "
        f"cap_iter_only={fit.gb.leaf_cap_iter_only} "
        f"nwalkers={fit.general.nwalkers} ntemps={fit.gb.ntemps}",
        flush=True,
    )
    fit.build()
    for i, (model, state) in enumerate(fit.sample(iterations=iterations)):
        ll = float(state.log_like.max())
        nleaves = int(state.branches["gb"].inds[0].sum())
        assert np.isfinite(ll), f"non-finite ll at iteration {i}"
        print(
            f"[SMOKE] it={i} ll_max={ll:.6e} cold_leaves={nleaves}",
            flush=True,
        )
    print(f"[SMOKE] phase={phase} sampling done", flush=True)


def _run_phase(phase: str, iterations: int, extra_env: dict, log_path: str) -> str:
    env = dict(os.environ)
    scratch = tempfile.mkdtemp(prefix=f"gb_rj_cycle_smoke_{phase}_")
    env.update(
        {
            # laptop budget: single process, every pool pinned
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            # stock-API smoke configuration
            "DATA_MODE": "synthetic",
            "GB_MODE": "search",
            "NWALKERS": "2",
            "GB_NTEMPS": "2",
            # the three new knobs, all ON
            "GB_SEARCH_RJ_REPLACE": "1",
            "GB_SEARCH_PRIOR_REMOVAL": "1",
            "GB_LEAF_CAP_ITER_ONLY": "1",
            "GB_LEAF_CAP_MIN_ITERS": "1",
            # instrumentation
            "GB_REPLACE_DEBUG": "1",
            "GF_MOVE_TIMING": "1",
            "FILE_STORE_DIR": scratch + "/",
        }
    )
    env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__), "--phase", phase,
           "--iterations", str(iterations)]
    print(f"[smoke] phase {phase}: {iterations} iterations "
          f"(store {scratch})", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True
        )
    out = open(log_path).read()
    print(f"[smoke] phase {phase} exit={proc.returncode} "
          f"wall={time.perf_counter() - t0:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        print(out[-8000:])
        raise SystemExit(f"phase {phase} FAILED (exit {proc.returncode})")
    return out


def _check_cycle_order(out: str) -> int:
    """Every iteration's [GF_TIMING] GB-move sequence must equal CYCLE."""
    seq: dict[int, list] = {}
    for m in re.finditer(r"\[GF_TIMING\] stage=(\S+) move=(\S+) it=(\d+)", out):
        _stage, move, it = m.group(1), m.group(2), int(m.group(3))
        if move in CYCLE:
            seq.setdefault(it, []).append(move)
    assert seq, "no [GF_TIMING] GB move lines found (GF_MOVE_TIMING=1 set?)"
    for it, moves in sorted(seq.items()):
        assert moves == CYCLE, (
            f"iteration {it}: GB move order {moves} != expected cycle {CYCLE}"
        )
    return len(seq)


def main() -> None:
    if "--phase" in sys.argv:
        i = sys.argv.index("--phase")
        phase = sys.argv[i + 1]
        iters = int(sys.argv[sys.argv.index("--iterations") + 1])
        _child(phase, iters)
        return

    log_dir = tempfile.mkdtemp(prefix="gb_rj_cycle_smoke_logs_")

    # ---- Phase A: honest 3-iteration run --------------------------------
    out_a = _run_phase("A", 3, {}, os.path.join(log_dir, "phase_a.log"))
    n_it = _check_cycle_order(out_a)
    print(f"[smoke] PASS cycle order fstat-birth -> replace -> removal "
          f"({n_it} iterations checked)", flush=True)

    n_rt = out_a.count(ROUND_TRIP_MSG)
    assert n_rt >= 1, (
        "no get_replace_ll round-trip check fired -- no replace proposal "
        "saw an alive source (births did not stick?)"
    )
    print(f"[smoke] PASS replace wrapper round-trip bit-identical "
          f"({n_rt} proposal rounds)", flush=True)

    n_cap = out_a.count(CAP_INC_MSG)
    assert n_cap >= 1, "iter-only leaf caps never incremented"
    print(f"[smoke] PASS iter-only leaf caps incremented ({n_cap} log lines)",
          flush=True)

    if SWAP_IDENTITY_MSG in out_a:
        print("[smoke] BONUS: a replacement was accepted naturally in "
              "phase A (identity asserted there too)", flush=True)

    # ---- Phase B: forced-accept accepted-swap identity -------------------
    out_b = _run_phase(
        "B", 2, {"GB_REPLACE_FORCE_ACCEPT": "1"},
        os.path.join(log_dir, "phase_b.log"),
    )
    n_id = out_b.count(SWAP_IDENTITY_MSG)
    assert n_id >= 1, (
        "no accepted-swap residual identity check fired under "
        "GB_REPLACE_FORCE_ACCEPT=1"
    )
    print(f"[smoke] PASS accepted-swap residual identity r_after == "
          f"r_before + h_old - h_new ({n_id} accepts checked)", flush=True)

    print(f"[smoke] ALL CHECKS PASSED (logs under {log_dir})", flush=True)


if __name__ == "__main__":
    main()
