#!/usr/bin/env python
"""Measure the GB proposal-Fisher reuse geometry (CPU, stock API).

``GBSpecialBase._compute_proposal_cholesky`` is the top compute lever in a GB
search: an information matrix per source per iteration (~17 waveform evals),
measured at 56-60% of a search iteration and *leaf-independent*, so it does
not amortize as the fit deepens.  The open question is not "does caching
help" but **what the reuse structure actually is**:

* **temporal** -- how far does one ``(temp, walker, leaf)`` slot move between
  consecutive Fishers, in the cached Fisher's own metric, and what does the
  covariance cost if it is reused anyway?
* **cross-slot** -- how many *distinct* Fishers does one propose contain once
  slots sitting on the same physical source are clustered?
* **temperature** -- do hot rungs behave differently enough to warrant a
  cold-chain-only or beta-scaled policy?

This runs the tiny stock GB search (``erebor.gb_no_fg(lite=True)``,
``data_mode="synthetic"``, ``GB_MODE=search``) with ``GB_FISHER_AUDIT=1``,
which returns the DIRECT factors (cache-off physics, bit for bit) and only
measures.  The audit's own summary is logged by
``lisatools.globalfit.moves._fisher_audit`` at exit; this script echoes the
final block and the ``[GB_TIMING]`` cholesky share so the ceiling on any
caching scheme is visible next to the reuse numbers.

Run:  python scripts/gb/gb_fisher_cache_audit.py [--iterations N]
      [--nwalkers N] [--ntemps N] [--sources N]

Laptop policy: single process, every thread pool pinned (MPI-only, no OMP).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time


def _child(args) -> None:
    """Runs inside the subprocess: build the tiny search fit and sample it."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    import numpy as np

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(lite=True)
    # Narrow 3-layer band centred on the injections; f0_lims resolve to the
    # central layer so every birth draw lands in the one interior band.
    fit.gb.center_freq = 8.5e-3
    fit.gb.n_layers = 3
    # A few LOUD synthetic sources spread across the band: cross-slot
    # clustering is only measurable when several sources share the band and
    # many (temp, walker) slots sit on each of them.
    rng = np.random.default_rng(1234)
    n_src = args.sources
    f0s = 8.5e-3 + np.linspace(-1.0, 1.0, n_src) * 2.0e-5 if n_src > 1 else [8.5e-3]
    inj = []
    for k in range(n_src):
        inj.append([
            3.0e-21, float(f0s[k]), 5.0e-15, 0.0,
            float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.0, np.pi)),
            float(rng.uniform(0.0, 2 * np.pi)), 4.0, -0.6,
        ])
    fit.general.gb_injection_params = np.array(inj)
    print(
        f"[AUDIT] mode={fit.gb.mode} nwalkers={fit.general.nwalkers} "
        f"ntemps={fit.gb.ntemps} sources={n_src}",
        flush=True,
    )
    fit.build()
    for i, (model, state) in enumerate(fit.sample(iterations=args.iterations)):
        ll = float(state.log_like.max())
        nleaves = int(state.branches["gb"].inds[0].sum())
        assert np.isfinite(ll), f"non-finite ll at iteration {i}"
        print(f"[AUDIT] it={i} ll_max={ll:.6e} cold_leaves={nleaves}", flush=True)
    print("[AUDIT] sampling done", flush=True)


def _run(args, log_path: str) -> str:
    env = dict(os.environ)
    scratch = tempfile.mkdtemp(prefix="gb_fisher_audit_")
    env.update({
        # laptop budget: single process, every pool pinned
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        # stock-API smoke configuration
        "DATA_MODE": "synthetic",
        # "pe" starts the leaves AT the injections: the converged in-model
        # regime with stable leaf identity, i.e. the BEST case for reuse.
        # "search" adds RJ birth/death churn -- the worst case.  Measuring
        # both brackets the answer.
        "GB_MODE": args.mode,
        "NWALKERS": str(args.nwalkers),
        "GB_NTEMPS": str(args.ntemps),
        # The replace/removal cycle rewrites an alive leaf's parameters with
        # a fresh prior draw, which by construction destroys temporal slot
        # continuity -- OFF by default so the audit measures the in-model
        # regime, ON via --rj-cycle to quantify how much churn costs.
        "GB_SEARCH_RJ_REPLACE": "1" if args.rj_cycle else "0",
        "GB_SEARCH_PRIOR_REMOVAL": "1" if args.rj_cycle else "0",
        "GB_LEAF_CAP_ITER_ONLY": "1",
        # Cap advancement is a cost knob here: every extra leaf per walker
        # multiplies the (temp x walker x leaf) Fisher batch, and the reuse
        # geometry is a per-source property that does not need a deep fit.
        "GB_LEAF_CAP_MIN_ITERS": str(args.cap_every),
        # THE measurement
        "GB_FISHER_AUDIT": "1",
        "GB_FISHER_AUDIT_EVERY": str(args.audit_every),
        "GB_FISHER_AUDIT_OUT": os.path.join(scratch, "fisher_audit.npz"),
        "GB_FISHER_AUDIT_PROBE": str(args.probe),
        "GB_FISHER_AUDIT_PROBE_AFTER": str(args.probe_after),
        "FILE_STORE_DIR": scratch + "/",
    })
    cmd = [
        sys.executable, os.path.abspath(__file__), "--child",
        "--iterations", str(args.iterations),
        "--nwalkers", str(args.nwalkers),
        "--ntemps", str(args.ntemps),
        "--sources", str(args.sources),
    ]
    print(f"[audit] {args.iterations} iterations, {args.nwalkers}x{args.ntemps} "
          f"sampler (store {scratch})", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    out = open(log_path).read()
    print(f"[audit] exit={proc.returncode} wall={time.perf_counter() - t0:.1f}s "
          f"log={log_path}", flush=True)
    if proc.returncode != 0:
        print(out[-8000:])
        raise SystemExit(f"audit run FAILED (exit {proc.returncode})")
    return out


def _timing_share(out: str) -> None:
    """Echo the cholesky share of each propose from the [GB_TIMING] lines."""
    rows = []
    for m in re.finditer(r"\[GB_TIMING [^\]]+\] total=([\d.]+)s .*", out):
        line = m.group(0)
        total = float(m.group(1))
        ch = re.search(r"inmodel_cholesky=([\d.]+)s", line)
        if ch:
            rows.append((total, float(ch.group(1))))
    if not rows:
        print("[audit] no [GB_TIMING] cholesky spans found")
        return
    tot = sum(r[0] for r in rows)
    cho = sum(r[1] for r in rows)
    print(f"[audit] cholesky share of propose wall time: {cho:.2f}s / {tot:.2f}s "
          f"= {100.0 * cho / max(tot, 1e-12):.1f}%  ({len(rows)} proposes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--nwalkers", type=int, default=6)
    p.add_argument("--ntemps", type=int, default=4)
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--mode", choices=("pe", "search"), default="pe")
    p.add_argument("--probe-after", type=int, default=40,
                   help="delay the sensitivity scan past this many audit "
                        "calls, so it lands on locked-on rather than "
                        "prior-drawn sources")
    p.add_argument("--probe", type=int, default=0,
                   help="sources to run the per-column sensitivity scan on "
                        "(costs ndim x 3 extra Fisher evaluations each)")
    p.add_argument("--rj-cycle", action="store_true",
                   help="enable the search replace/removal RJ cycle "
                        "(destroys temporal slot continuity by design)")
    p.add_argument("--audit-every", type=int, default=10,
                   help="summary-log cadence in audit calls (partial results "
                        "are readable while a long run is still going)")
    p.add_argument("--cap-every", type=int, default=6,
                   help="iterations between leaf-cap increments (cost knob)")
    args = p.parse_args()

    if args.child:
        _child(args)
        return

    log_dir = tempfile.mkdtemp(prefix="gb_fisher_audit_logs_")
    out = _run(args, os.path.join(log_dir, "audit.log"))

    _timing_share(out)
    i = out.rfind("FINAL")
    if i < 0:
        print(out[-4000:])
        raise SystemExit("audit produced no FINAL summary "
                         "(did _compute_proposal_cholesky ever run?)")
    print("\n=== FISHER REUSE AUDIT ===")
    print(out[i:].split("[AUDIT]")[0].strip())


if __name__ == "__main__":
    main()
