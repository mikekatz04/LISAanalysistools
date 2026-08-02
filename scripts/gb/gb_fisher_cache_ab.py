#!/usr/bin/env python
"""A/B the GB Fisher/Cholesky proposal cache: GB_FISHER_CACHE=1 vs =0.

The cache in ``GBSpecialBase._compute_proposal_cholesky`` reuses a factored
proposal covariance across quantized-parameter cells.  This is the
black-box check the cache work needs: run the SAME tiny stock GB search
twice, once with the cache on and once off, and report the three numbers
that decide whether it earns its place:

* **hit rate** -- straight from the move's own cache counters (a key was
  either seen before or it was not; no derived metric involved);
* **wall time** -- the ``inmodel_cholesky`` span and the propose total from
  the ``[GB_TIMING]`` line, i.e. what the cache is actually buying;
* **sampling outcome** -- per-iteration cold-chain max lnL and leaf count,
  as a sanity check that cache-on is not visibly worse.

The two runs diverge after the first differing proposal (a reused factor is
not bit-identical to a fresh one), so the lnL columns are a smell test, not
an equivalence proof; the cache is a proposal-shape approximation and M-H
corrects it.

Run:  python scripts/gb/gb_fisher_cache_ab.py [--iterations N]
Laptop policy: runs sequentially, single process, every thread pool pinned.
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

    # Same start for both arms so the comparison begins from one state.
    np.random.seed(args.seed)

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(lite=True)
    fit.gb.center_freq = 8.5e-3
    fit.gb.n_layers = 3
    rng = np.random.default_rng(1234)
    n_src = args.sources
    f0s = (8.5e-3 + np.linspace(-1.0, 1.0, n_src) * 2.0e-5 if n_src > 1
           else [8.5e-3])
    fit.general.gb_injection_params = np.array([
        [3.0e-21, float(f0s[k]), 5.0e-15, 0.0,
         float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.0, np.pi)),
         float(rng.uniform(0.0, 2 * np.pi)), 4.0, -0.6]
        for k in range(n_src)
    ])
    print(f"[AB] cache={os.environ.get('GB_FISHER_CACHE')} "
          f"mode={fit.gb.mode} nwalkers={fit.general.nwalkers} "
          f"ntemps={fit.gb.ntemps} sources={n_src}", flush=True)
    fit.build()
    for i, (model, state) in enumerate(fit.sample(iterations=args.iterations)):
        ll = float(state.log_like.max())
        nleaves = int(state.branches["gb"].inds[0].sum())
        assert np.isfinite(ll), f"non-finite ll at iteration {i}"
        print(f"[AB] it={i} ll_max={ll:.6e} cold_leaves={nleaves}", flush=True)
    print("[AB] sampling done", flush=True)


def _run(args, cache_on: bool, log_path: str) -> str:
    env = dict(os.environ)
    scratch = tempfile.mkdtemp(prefix=f"gb_fisher_ab_{int(cache_on)}_")
    env.update({
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "DATA_MODE": "synthetic",
        "GB_MODE": "search",
        "NWALKERS": str(args.nwalkers),
        "GB_NTEMPS": str(args.ntemps),
        "GB_LEAF_CAP_ITER_ONLY": "1",
        "GB_LEAF_CAP_MIN_ITERS": str(args.cap_every),
        # THE knob under test
        "GB_FISHER_CACHE": "1" if cache_on else "0",
        # log the hit rate often enough for a short run to report one
        "GB_FISHER_CACHE_LOG": "20",
        "GB_FISHER_AUDIT": "0",
        "FILE_STORE_DIR": scratch + "/",
    })
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--iterations", str(args.iterations),
           "--nwalkers", str(args.nwalkers), "--ntemps", str(args.ntemps),
           "--sources", str(args.sources), "--seed", str(args.seed)]
    label = "ON " if cache_on else "OFF"
    print(f"[ab] cache {label}: {args.iterations} iterations "
          f"({args.nwalkers}x{args.ntemps})", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    out = open(log_path).read()
    wall = time.perf_counter() - t0
    print(f"[ab] cache {label}: exit={proc.returncode} wall={wall:.1f}s "
          f"log={log_path}", flush=True)
    if proc.returncode != 0:
        print(out[-6000:])
        raise SystemExit(f"cache={label} run FAILED (exit {proc.returncode})")
    return out


def _spans(out: str):
    """Summed inmodel_cholesky and propose-total seconds from [GB_TIMING]."""
    tot = cho = 0.0
    n = 0
    for m in re.finditer(r"\[GB_TIMING [^\]]+\] total=([\d.]+)s[^\n]*", out):
        line = m.group(0)
        tot += float(m.group(1))
        c = re.search(r"inmodel_cholesky=([\d.]+)s", line)
        if c:
            cho += float(c.group(1))
        n += 1
    return tot, cho, n


def _lls(out: str):
    return [(int(a), float(b), int(c)) for a, b, c in
            re.findall(r"\[AB\] it=(\d+) ll_max=([-\de.+]+) cold_leaves=(\d+)",
                       out)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--nwalkers", type=int, default=6)
    p.add_argument("--ntemps", type=int, default=4)
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--cap-every", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.child:
        _child(args)
        return

    log_dir = tempfile.mkdtemp(prefix="gb_fisher_ab_logs_")
    outs = {}
    for on in (False, True):          # cache OFF first: the reference arm
        outs[on] = _run(args, on, os.path.join(
            log_dir, f"cache_{int(on)}.log"))

    print("\n=== FISHER CACHE A/B ===")
    print(f"{'':<12}{'propose total':>15}{'inmodel_cholesky':>19}"
          f"{'cholesky share':>17}")
    ref = None
    for on in (False, True):
        tot, cho, n = _spans(outs[on])
        if not on:
            ref = cho
        share = 100.0 * cho / max(tot, 1e-12)
        print(f"cache {'ON ' if on else 'OFF':<7}{tot:>13.1f}s"
              f"{cho:>17.1f}s{share:>16.1f}%")
    _, cho_on, _ = _spans(outs[True])
    if ref:
        print(f"\ncholesky wall-time change: {100.0 * (cho_on - ref) / ref:+.1f}%"
              f"  ({ref:.1f}s -> {cho_on:.1f}s)")

    hits = re.findall(r"fisher cache hit rate ([\d.]+)% \((\d+) entries\)",
                      outs[True])
    print(f"\ncache hit rate: "
          + (f"{hits[-1][0]}% ({hits[-1][1]} entries) "
             f"[{len(hits)} log points]" if hits
             else "NO hit-rate line logged (too few lookups)"))

    print("\nper-iteration cold-chain max lnL / leaves:")
    a, b = _lls(outs[False]), _lls(outs[True])
    for i in range(max(len(a), len(b))):
        la = f"{a[i][1]:.4e} ({a[i][2]})" if i < len(a) else "-"
        lb = f"{b[i][1]:.4e} ({b[i][2]})" if i < len(b) else "-"
        print(f"  it={i:<3} OFF {la:<22} ON {lb}")
    print(f"\nlogs: {log_dir}")


if __name__ == "__main__":
    main()
