#!/usr/bin/env python
"""t0-foundation/build-all-lite: every lite stock variant builds through the
stock API, with the deepcopy+pickle contract verified pre-build.

Parent mode loops the variants, spawning one subprocess per variant (memory
isolation on the laptop; one crash cannot hide the others).  Child mode
(``--one NAME``) does the actual work and prints one ``[RESULT]`` line.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import resource
import subprocess
import sys

VARIANTS = (
    "gb_no_fg_lite",
    "vgb",              # built with lite=True (no registered _lite twin)
    "noise_only_lite",
    "noise_sgwb_lite",
    "full_year_combined_lite",
    "all_sources_lite",
    "blank",
)


def _rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3


def build_one(name: str) -> None:
    from lisatools.globalfit.stock import erebor

    kwargs = {"lite": True} if not name.endswith("_lite") and name != "blank" else {}
    fit = getattr(erebor, name)(**kwargs)

    # repo contract: unbuilt fits are cheap, deepcopy- and pickle-safe
    fit2 = copy.deepcopy(fit)
    pickle.loads(pickle.dumps(fit2))

    fit.build()
    n_moves = len(fit.recipe.list_moves()) if hasattr(fit.recipe, "list_moves") else -1
    print(
        f"[RESULT] variant={name} build=ok moves={n_moves} rss_mb={_rss_mb():.0f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    args = ap.parse_args()

    if args.one:
        build_one(args.one)
        return

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("USE_GPU", "0")
    env.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")

    built = failures = 0
    for name in VARIANTS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one", name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode == 0 and f"variant={name} build=ok" in proc.stdout:
            built += 1
        else:
            failures += 1
            print(f"[RESULT] variant={name} build=FAIL exit={proc.returncode}")
    print(f"[RESULT] variants_built={built} build_failures={failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
