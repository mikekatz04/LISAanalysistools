"""``snapshot.tar.gz`` -> monitor page, in one command.

    python -m lisatools.globalfit.monitor.from_tar SNAPSHOT.tar.gz [OUT.html]

This is the loop that was being done by hand on every snapshot: make a
scratch directory, untar, hunt for the run directory somewhere under
``shared/data/global_fit_output/<run>/``, then call the generator with
that path. Each step is easy and each one is easy to get subtly wrong --
pointing the generator at the tar's ROOT, for instance, renders a page
with every panel missing and no error.

Defaults chosen so the common case is zero flags:

* the run directory is DISCOVERED (the deepest directory holding a
  ``*testing*.h5``), so the tar's internal layout does not matter;
* the output lands beside the tarball, named after the run, unless a
  second argument says otherwise;
* extraction goes to a scratch directory that is REUSED when it already
  holds the same tar (``--fresh`` forces a re-extract), because a 130 MB
  archive is slow to unpack and usually gets rendered more than once;
* the mojito tree is RESOLVED, not demanded -- from ``MOJITO_INFO_PATH``,
  the run's own settings inside the tar, the legacy ``MOJITO_*`` vars, or
  the standard cluster/laptop paths. Both things it feeds degrade
  SILENTLY, so the chosen tree is always printed and a failure to find
  one is warned before the slow work starts.

``--short-page`` (2026-10-03) renders the SHORT page instead of the full
one (``_short.py``), named ``RUN_monitor_short.html`` beside the tar unless
a second argument says otherwise. A short tar (``--short`` /
``GF_MONITOR_SNAPSHOT_SHORT=1``) carries everything it reads.

ONE PYTHON PROCESS. The page renders in THIS interpreter; ``--subprocess``
opts into a child if you would rather keep the ~2.5 GB peak and any
matplotlib fault out of it. Both produce byte-identical output -- see
``_generator.py``'s ``_NOISE_PANEL_FONT`` for the one rcParam that used to
make that untrue.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time


def _default_scratch() -> str:
    base = (os.environ.get("TMPDIR") or "/tmp").rstrip("/")
    return os.path.join(base, "gf_monitor_from_tar")


def _tar_key(tar_path: str) -> str:
    """Identity of a tarball without reading 130 MB of it.

    Path + size + mtime. A re-downloaded snapshot with the same name gets
    a new mtime and therefore a new key, which is the case that matters:
    the browser names them all ``... (3).gz`` and reusing a stale
    extraction silently renders the wrong iteration.
    """
    st = os.stat(tar_path)
    h = hashlib.sha1(
        f"{os.path.abspath(tar_path)}|{st.st_size}|{st.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    return h


def find_run_dir(root: str) -> str | None:
    """The run directory inside an extracted snapshot.

    The deepest directory containing a live-looking store. Backups,
    corrupt copies and extracts are all accepted here -- a snapshot tar
    normally ships ONLY the ``*_extract.h5``, so requiring a full store
    would find nothing.
    """
    best = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if any(fn.endswith(".h5") and "testing" in fn for fn in filenames):
            depth = dirpath.count(os.sep)
            if best is None or depth > best[0]:
                best = (depth, dirpath)
    return best[1] if best else None


def extract(tar_path: str, scratch: str, fresh: bool = False) -> str:
    """Untar into a per-tarball scratch directory. Returns that directory."""
    dest = os.path.join(scratch, _tar_key(tar_path))
    stamp = os.path.join(dest, ".extracted")
    if fresh and os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    if os.path.exists(stamp):
        print(f"[from_tar] reusing extraction at {dest} (--fresh to redo)")
        return dest
    os.makedirs(dest, exist_ok=True)
    st = time.perf_counter()
    with tarfile.open(tar_path, "r:*") as tf:
        # filter="data" refuses absolute paths and ../ escapes. It is the
        # default from Python 3.14 and a warning before that; setting it
        # explicitly keeps behaviour identical across versions.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:                      # pragma: no cover - <3.12
            tf.extractall(dest)
    open(stamp, "w").close()
    print(f"[from_tar] extracted {os.path.basename(tar_path)} in "
          f"{time.perf_counter() - st:.1f} s -> {dest}")
    return dest


def main(argv=None) -> int:
    # RESOLVE argv BEFORE ANYTHING READS IT. The `--` split below indexes
    # argv directly, so leaving it None crashed every real command line
    # (`TypeError: argument of type 'NoneType' is not iterable`) while
    # every test passed -- the tests all call main([...]) with an explicit
    # list, which is exactly the path that was never broken.
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="python -m lisatools.globalfit.monitor.from_tar",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tar", help="the gf_prod_*_snapshot.tar.gz")
    ap.add_argument("out", nargs="?", default=None,
                    help="output .html (default: beside the tar, named "
                         "after the run directory)")
    ap.add_argument("--scratch", default=None,
                    help=f"extraction root (default {_default_scratch()})")
    ap.add_argument("--fresh", action="store_true",
                    help="re-extract even if this tar was unpacked before")
    ap.add_argument("--run-dir", default=None,
                    help="skip discovery and render this directory")
    ap.add_argument("--mojito", default=None,
                    help="the mojito tree (the directory holding "
                         "catalogues/ and data/). Normally unnecessary: it "
                         "is resolved from MOJITO_INFO_PATH, the run's own "
                         "settings inside the tar, the legacy MOJITO_* "
                         "vars, then the standard cluster/laptop paths.")
    ap.add_argument("--build-truth", action="store_true",
                    help="generate the detectability truth set into the run "
                         "directory first if it is missing or was built for "
                         "a different Tobs. Tens of minutes, CPU-only. "
                         "Without it the page still renders, just with no "
                         "detectability overlays.")
    ap.add_argument("--short-page", action="store_true",
                    help="render the SHORT page (topline, lnL and leaf count "
                         "over time, phase-maximised overlap, current noise) "
                         "INSTEAD of the full one; default name "
                         "<run>_monitor_short.html beside the tar")
    ap.add_argument("--subprocess", action="store_true",
                    help="render in a CHILD interpreter instead of this one. "
                         "Output is identical; use it to keep the ~2.5 GB "
                         "peak and any matplotlib fault out of this process.")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds before the page build is killed")
    ap.add_argument("--add-info", action="append", default=None, metavar="OTHER_RUN_DIR",
                    help="overlay another run directory (e.g. a second extracted "
                         "snapshot) at its last stored row on the full page's comparison "
                         "figures. Repeatable; see python -m lisatools.globalfit.monitor -h.")
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
    # SPLIT ON `--` BEFORE PARSING, do not use argparse.REMAINDER.
    # REMAINDER absorbs everything from the first token it does not
    # recognise onward -- INCLUDING flags this parser defines. With it,
    # `RUN_DIR --build-truth -- --catalogue X` parsed build_truth as
    # FALSE and handed "--build-truth" to build_truth instead. It also
    # meant an unknown flag was swallowed rather than rejected, which is
    # the behaviour this parser exists to provide.
    if "--" in argv:
        _i = argv.index("--")
        argv, _passthru = argv[:_i], argv[_i + 1:]
    else:
        _passthru = []
    a = ap.parse_args(argv)
    for _flag, _val in (("--catalogue", a.catalogue),
                        ("--l1-brick", a.l1_brick)):
        if _val:
            _passthru += [_flag, _val]

    if not os.path.isfile(a.tar):
        ap.error(f"{a.tar} is not a file")

    root = a.run_dir or extract(
        a.tar, a.scratch or _default_scratch(), fresh=a.fresh)
    run_dir = a.run_dir or find_run_dir(root)
    if run_dir is None:
        print(f"[from_tar] no run directory (a dir holding *testing*.h5) "
              f"under {root}", file=sys.stderr)
        return 2
    print(f"[from_tar] run dir: {run_dir}")

    # ONE COMMAND, NOT TWO (user 2026-09-26). Resolved here rather than
    # left to the caller's shell: one of the sources is the snapshot's own
    # run_settings.log, so a tarball rendered on the machine that produced
    # it needs nothing configured at all. Reported either way -- both
    # things this feeds degrade SILENTLY, and learning that after a
    # two-minute render is the annoying way.
    from . import describe_run, resolve_mojito_path

    # WHAT ITERATION IS THIS? The first thing anyone wants to know about a
    # snapshot, and until now the only way to find out was to open the
    # finished page.
    _state = describe_run(run_dir)
    if _state:
        print(f"[from_tar] run state: {_state}")

    _moj, _src = resolve_mojito_path(run_dir=run_dir, explicit=a.mojito)
    if _moj:
        print(f"[from_tar] mojito data: {_moj}  (from {_src})")
    else:
        print(f"[from_tar] WARNING: {_src}.\n"
              "[from_tar]          The page will use the ANALYTIC PSD "
              "injection instead of a fit to the brick and will DROP the "
              "residual-spectrum and data/template/residual panels, with "
              "no error on the page itself. Pass --mojito /path/to/tree "
              "to fix.", file=sys.stderr)

    # THE TRUTH SET, in the same command (user 2026-09-26: "make the
    # regeneration part of the full python path"). Reported either way --
    # without it the page silently loses its completeness denominator and
    # every detectable-source target line.
    from . import build_truth_set, check_truth

    _truth, _tnote = check_truth(run_dir)
    if _truth:
        print(f"[from_tar] truth set: {_tnote}")
    elif a.build_truth:
        print(f"[from_tar] truth set: {_tnote} -- BUILDING one now "
              "(tens of minutes, CPU-only)")
        try:
            build_truth_set(run_dir,
extra_argv=_passthru)
            _truth, _tnote = check_truth(run_dir)
            print(f"[from_tar] truth set: {_tnote}")
        except Exception as e:                    # noqa: BLE001
            print(f"[from_tar] truth build FAILED ({type(e).__name__}: {e}); "
                  "rendering without detectability overlays", file=sys.stderr)
    else:
        print(f"[from_tar] WARNING: {_tnote}. The page will have NO "
              "detectability overlays (no completeness denominator, no "
              "detectable-source target line). Pass --build-truth to "
              "generate one into the run directory.", file=sys.stderr)

    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(a.tar)),
        f"{os.path.basename(run_dir.rstrip('/'))}_monitor"
        f"{'_short' if a.short_page else ''}.html")

    from . import build_monitor, build_short_monitor
    from ._add_info import parse_add_info

    add_info = (parse_add_info(a.add_info, base_run_dir=run_dir)
                if a.add_info is not None else None)
    if add_info:
        print(f"[from_tar] added runs: {', '.join(add_info)}"
              + ("  (full page only; not used by --short-page)" if a.short_page else ""))

    st = time.perf_counter()
    try:
        if a.short_page:
            build_short_monitor(run_dir, out, mojito=_moj)
        else:
            # ONE PYTHON: no child interpreter. See build_monitor_in_process.
            build_monitor(run_dir, out, timeout=a.timeout, mojito=_moj,
                          in_process=not a.subprocess, add_info=add_info)
    except Exception as e:                       # noqa: BLE001
        print(f"[from_tar] page build FAILED ({type(e).__name__}: {e})",
              file=sys.stderr)
        return 1
    print(f"[from_tar] wrote {out} "
          f"({os.path.getsize(out) / 1048576.0:.1f} MB) in "
          f"{time.perf_counter() - st:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
