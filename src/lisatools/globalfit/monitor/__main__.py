"""Build the monitor page, and optionally the snapshot tar, in one command.

    python -m lisatools.globalfit.monitor [--snapshot] [--build-truth] \\
        RUN_DIR [OUT.html]

Two directions, two entry points:

* **here** -- a LIVE run directory -> page (and with ``--snapshot``, the
  tarball too). This is the cluster-side command.
* :mod:`lisatools.globalfit.monitor.from_tar` -- a downloaded tarball ->
  page. That one CONSUMES a tar; it does not make one.

BOTH ARTIFACTS LAND BESIDE THE RUN FOLDER (user ruling 2026-09-26), not
inside it::

    /data/gf_prod_6mo_v9_4gpu/                    <- the run
    /data/gf_prod_6mo_v9_4gpu_monitor.html        <- the page
    /data/gf_prod_6mo_v9_4gpu_snapshot.tar.gz     <- the tar

⚠ This REVERSES an earlier guarantee: the tar used to contain the page,
because the page was written inside the directory being archived. It no
longer does. Copy both if the report has to travel with the data.

``--short-page`` (2026-10-03) builds the SHORT page INSTEAD of the full
one: the topline, likelihood and leaf count over time, the other sources, the
phase-maximised overlap and the current noise measurement -- the block that
also opens the full page -- in well under a minute, written
to ``RUN_DIR_monitor_short.html`` beside the run folder (a separate file; the
full page's name is untouched). The user's request and the page's contents
are in ``_short.py``. It composes with ``--snapshot`` and ``--build-truth``,
not with ``--snapshot-only``. In-run equivalent: ``GF_MONITOR_PAGE_SHORT=1``.

``--add-info OTHER_RUN_DIR`` (repeatable; user request 2026-10-12) overlays
another run, read at ITS last stored row, on the full page's comparison
figures only: the phase-maximised overlap CDF and survival count, recovered
[%] vs injected SNR and vs frequency (each run against its own truth set), the
Soms_d / Sa_a and galfor posterior histograms, and the PSD + foreground and
sensitivity curves. Everything else stays this run's. In-run equivalent:
``GF_MONITOR_ADD_INFO`` (directories joined by ``:``); the flag wins over it,
and ``--add-info none`` drops an exported list. See ``_add_info.py``.

``--build-truth`` generates the detectability truth set into the run
directory first. Anything after ``--`` goes straight to ``build_truth``
(``--catalogue``, ``--l1-brick``, ``--flo`` ...), though normally none of
it is needed: the mojito tree is resolved from ``MOJITO_INFO_PATH``, the
run's own settings, the legacy ``MOJITO_*`` vars or the standard
cluster/laptop paths, and handed to build_truth for both the catalogue
and the L1 orbit bricks.
"""

import argparse
import sys
import time

from . import (build_monitor, build_short_monitor, build_truth_set,
               check_truth, default_out_path, default_short_out_path,
               describe_run)
from ._add_info import ENV as ADD_ENV
from ._add_info import parse_add_info
from .snapshot import build_snapshot

USAGE = ("usage: python -m lisatools.globalfit.monitor [--snapshot] "
         "[--short-page] [--build-truth] [--keep N] [--add-info OTHER_RUN_DIR ...] "
         "RUN_DIR [OUT.html] [-- BUILD_TRUTH_ARGS...]\n"
         "       --snapshot     also build <RUN_DIR>_snapshot.tar.gz\n"
         "       --add-info D   overlay run D's last row on the comparison figures "
         "(repeatable)\n"
         "       --short-page   the SHORT page <RUN_DIR>_monitor_short.html "
         "instead of the full one\n"
         "       --output-data  ONLY <RUN_DIR>_output_data.tar.gz: store extract + "
         "residual snapshot + WDM lookup tables, no logs, no page\n"
         "       --build-truth  generate the truth set into RUN_DIR first\n"
         "       --keep N       keep N warm-chain iterations (default: 5, or 1 with --short)\n"
         "       --cold-keep N  keep N cold-chain iterations (default: 12, or 1 with --short)\n"
         "       (a downloaded tarball goes the other way: python -m "
         "lisatools.globalfit.monitor.from_tar SNAP.tar.gz)")


def _parser():
    """A REAL parser (2026-09-27).

    This used to be ad-hoc list surgery: strip the flags it knew about,
    then take ``argv[0]`` as the run dir and ``argv[1]`` as the output
    path. So an unrecognised flag did not error -- it silently BECAME THE
    OUTPUT PATH. ``--build-truth --catalogue /path/...`` set
    ``out="--catalogue"``, which is how a real cluster invocation went
    wrong. argparse rejects what it does not know, which is the point.
    """
    ap = argparse.ArgumentParser(
        prog="python -m lisatools.globalfit.monitor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out", nargs="?", default=None,
                    help="output .html (default: <RUN_DIR>_monitor.html, "
                         "or <RUN_DIR>_monitor_short.html with --short-page, "
                         "beside the run folder)")
    # A DIFFERENT FLAG FROM --short, which is the short TAR. This one is the
    # short PAGE, and it REPLACES the full page for this command rather than
    # adding to it: the point is a page that is cheap to make, and building
    # the full one as well would spend the minutes it exists to save.
    ap.add_argument("--short-page", action="store_true",
                    help="build the SHORT page <RUN_DIR>_monitor_short.html "
                         "INSTEAD of the full page: topline, lnL and leaf "
                         "count over time, phase-maximised overlap, current "
                         "noise; under a minute. In-run equivalent: "
                         "GF_MONITOR_PAGE_SHORT=1.")
    ap.add_argument("--snapshot", action="store_true",
                    help="also build <RUN_DIR>_snapshot.tar.gz beside it")
    ap.add_argument("--short", action="store_true",
                    help="build <RUN_DIR>_short.tar.gz instead: log "
                         "information and the most recent state, no large "
                         "files (keep=1, every oversized *.log reduced to "
                         "filtered + tail, nothing over 5 MB, no fstat). "
                         "In-run equivalent: GF_MONITOR_SNAPSHOT_SHORT=1.")
    ap.add_argument("--output-data", action="store_true",
                    help="build ONLY <RUN_DIR>_output_data.tar.gz: the store "
                         "extract (every stored row of every cold chain and "
                         "of the small arrays; the large tempered arrays from "
                         "the residual snapshot's row to the last row), the "
                         "residual snapshot and the run's WDM "
                         "lookup tables (wdm_lookup_*.h5). No logs, no page. "
                         "For postprocessing (LATW tutorial 09).")
    ap.add_argument("--snapshot-only", action="store_true",
                    help="build ONLY the tar, skip the page. The page can "
                         "then be rendered offline from it with "
                         "`python -m lisatools.globalfit.monitor.from_tar "
                         "SNAP.tar.gz`. In-run equivalent: GF_MONITOR_PAGE=0.")
    # The F-stat fit's epoch caches (gb_fstat_fit/**: the comb and the
    # stacked grid peaks, GBs at 1 yr) are OUT of every tar by default and
    # have been since the shell recipe; only their DONE.json markers ride
    # along. User ruling 2026-10-03: "make it default to leaving them out.
    # If you want them, you add --add-fstat". In-run equivalent:
    # GF_MONITOR_SNAPSHOT_FSTAT=1. A --short tar never carries them.
    ap.add_argument("--add-fstat", action="store_true",
                    help="ALSO ship the F-stat fit's epoch caches under "
                         "gb_fstat_fit/ (large; off by default, only their "
                         "DONE.json markers are kept). Ignored by --short. "
                         "In-run equivalent: GF_MONITOR_SNAPSHOT_FSTAT=1.")
    ap.add_argument("--keep", type=int, default=None,
                    help="number of warm-chain iterations to keep in the "
                         "snapshot extract (default: 5 for normal, 1 for "
                         "--short). Explicit value overrides --short default.")
    ap.add_argument("--cold-keep", type=int, default=None,
                    help="number of cold-chain iterations to keep in the "
                         "snapshot extract (default: 12 for normal, 1 for "
                         "--short). Explicit value overrides --short default.")
    ap.add_argument("--build-truth", action="store_true",
                    help="generate the detectability truth set into the run "
                         "directory first when it is missing or was built "
                         "for a different Tobs. Tens of minutes, CPU-only.")
    ap.add_argument("--subprocess", action="store_true",
                    help="render in a child interpreter. Output identical.")
    # REPEATABLE, order kept (user request 2026-10-12: "It should be something
    # like --add-info /path/to/gf_prod_6mo_v9_4gpu" ... "it should be able to
    # add a number of runs").
    ap.add_argument("--add-info", action="append", default=None, metavar="OTHER_RUN_DIR",
                    help="overlay another run (its last stored row) on the full page's "
                         "comparison figures: the phase-maximised overlap figures and the "
                         "noise / foreground posteriors and curves. Repeatable; the page "
                         "stays this run's everywhere else. Wins over the in-run knob "
                         "GF_MONITOR_ADD_INFO (':'-joined dirs); '--add-info none' drops it.")
    # FIRST-CLASS, not just pass-through. These are the two overrides
    # anyone actually reaches for, and requiring `-- --catalogue X` for
    # them is a trap: the natural spelling produced "unrecognized
    # arguments" on a real cluster invocation (2026-09-27).
    ap.add_argument("--catalogue", default=None,
                    help="GB catalogue hdf5 (or the directory holding it) "
                         "for --build-truth. Normally unnecessary.")
    ap.add_argument("--l1-brick", default=None,
                    help="explicit mojito L1 .h5 for the injected orbits "
                         "used by --build-truth. Normally unnecessary.")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # SPLIT ON `--` BEFORE PARSING, do not use argparse.REMAINDER.
    # REMAINDER absorbs everything from the first token it does not
    # recognise onward -- INCLUDING flags this parser defines. With it,
    # `RUN_DIR --build-truth -- --catalogue X` parsed build_truth as
    # FALSE and handed "--build-truth" to build_truth instead. It also
    # meant an unknown flag was swallowed rather than rejected, which is
    # the behaviour this parser exists to provide.
    if "--" in argv:
        _i = argv.index("--")
        argv, extra = argv[:_i], argv[_i + 1:]
    else:
        extra = []
    ap = _parser()
    a = ap.parse_args(argv)
    if a.short_page and a.snapshot_only:
        ap.error("--short-page builds a page and --snapshot-only builds none; "
                 "use --short-page --snapshot for both")
    if a.output_data and (a.short or a.short_page or a.snapshot):
        ap.error("--output-data builds only the data tar; it does not combine "
                 "with --short, --short-page or --snapshot")
    for _flag, _val in (("--catalogue", a.catalogue),
                        ("--l1-brick", a.l1_brick)):
        if _val:
            extra += [_flag, _val]
    run_dir = a.run_dir
    out = a.out or (default_short_out_path(run_dir) if a.short_page
                    else default_out_path(run_dir))

    _truth, _tnote = check_truth(run_dir)
    if _truth:
        print(f"[monitor] truth set: {_tnote}")
    elif a.build_truth:
        print(f"[monitor] truth set: {_tnote} -- BUILDING one now "
              "(tens of minutes, CPU-only)")
        try:
            build_truth_set(run_dir, extra_argv=extra)
            print(f"[monitor] truth set: {check_truth(run_dir)[1]}")
        except Exception as e:                    # noqa: BLE001
            print(f"[monitor] truth build FAILED ({type(e).__name__}: {e})",
                  file=sys.stderr)
    else:
        print(f"[monitor] WARNING: {_tnote} -- no detectability overlays. "
              "Pass --build-truth to generate one.", file=sys.stderr)
        if extra:
            # Otherwise these vanish without a word, which is how you spend
            # an hour wondering why --catalogue had no effect.
            print(f"[monitor] NOTE: {extra} was given but --build-truth was "
                  "not, so it does nothing.", file=sys.stderr)

    _state = describe_run(run_dir)
    if _state:
        print(f"[monitor] run state: {_state}")

    # ADDED RUNS: the flag's list (or None: the environment's, read by the
    # generator itself), resolved here only to say what the page will carry.
    add_info = (parse_add_info(a.add_info, base_run_dir=run_dir)
                if a.add_info is not None else None)
    _shown = add_info if add_info is not None else parse_add_info(base_run_dir=run_dir)
    if _shown and (a.short_page or a.snapshot_only):
        print("[monitor] NOTE: added runs overlay the FULL page only; "
              f"{_shown} not used here.", file=sys.stderr)
    elif _shown:
        print("[monitor] added runs (last stored row each, on the comparison figures): "
              + ", ".join(_shown)
              + ("" if add_info is not None else f"  (from {ADD_ENV})"))

    if not (a.snapshot_only or a.output_data):
        st = time.perf_counter()
        if a.short_page:
            build_short_monitor(run_dir, out)
        else:
            build_monitor(run_dir, out, in_process=not a.subprocess, add_info=add_info)
        print(f"[monitor] wrote {out} in {time.perf_counter() - st:.1f} s")

    if a.snapshot or a.snapshot_only or a.short or a.output_data:
        st = time.perf_counter()
        tar = build_snapshot(run_dir, short=a.short, include_fstat=a.add_fstat,
                             keep=a.keep, cold_keep=a.cold_keep,
                             output_data=a.output_data)
        if tar:
            print(f"[monitor] wrote {tar} in {time.perf_counter() - st:.1f} s")
        else:
            # Under --snapshot-only the tar is the ONLY product, so say
            # that rather than reassuring the operator about a page that
            # was never asked for.
            print("[monitor] snapshot FAILED (see warnings above)"
                  + ("" if (a.snapshot_only or a.output_data) else "; the page was still "
                     "written"), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
