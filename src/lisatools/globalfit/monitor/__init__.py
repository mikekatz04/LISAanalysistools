"""Installed monitor-page and snapshot production.

WHAT THIS IS (user ruling 2026-09-26): "containerize the html production
so it easily adjustable but build into the lisatools/globalfit package.
It does not have to be clean, we will fix it up later, but make sure it
reproduces the html as is done now."

**Reproduction is the hard requirement; tidiness is not.**
``_generator.py`` is ``scripts/diagnostics/gf_monitor_gen.py`` moved here
byte-for-byte -- 5908 lines, 431 top-level statements, 217 module-level
names, ``sys.argv`` read at import, ``raise SystemExit(0)`` if imported.
Turning it into a library is a real refactor (its 23 top-level helpers
close over those 217 globals) and is explicitly deferred.

What moving it buys, which is the part that was blocking: it SHIPS.
``wheel.packages = ["src/lisatools"]`` covers this directory, so a wheel
install has the generator and the saver rank no longer has to walk the
filesystem hunting for a ``scripts/`` directory that exists only in a
source checkout.

WHY EVERY PATH HERE SPAWNS A FRESH INTERPRETER
----------------------------------------------
Not for isolation (though it gives that too) -- for CORRECTNESS. The
generator sets its own dark-theme ``plt.rcParams`` near the top, and
~700 lines later imports ``lisatools...erebor.noise``, whose chain
reaches eryn, which calls ``plt.style.use(["science"])`` as an import
side effect. So in the original invocation the science style lands AFTER
the generator's block and wins.

Import this package and that order inverts: ``import lisatools`` already
pulls matplotlib, pyplot AND eryn (measured), so the style is applied
BEFORE the generator's block and the generator's block wins instead.
That is not hypothetical -- running it in-process changed the
"instrument noise posteriors" panel (25722 vs 27586 base64 chars, 25806
pixels across the whole plot area). Everything else on the page was
identical, which is exactly how a defect like this survives review.

A child interpreter running ``python <_generator.py> RUN_DIR OUT`` has
imported nothing, so the original order is restored and the page is
identical BY CONSTRUCTION rather than by inspection. The subprocess also
returns the build's ~2.5 GB peak RSS to the OS and keeps a matplotlib
segfault away from the caller -- on the saver rank, the run's only
writer, that second property is not optional either.

``build_truth.py`` came along unchanged and NOT renamed: the generator
locates it as a ``__file__`` sibling (``from build_truth import opt_snr``)
for column 2 of the #detect table. Renaming it would drop that column
behind a caught exception.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)

__all__ = [
    "build_monitor",
    "build_short_monitor",
    "default_out_path",
    "default_short_out_path",
    "generator_path",
    "resolve_mojito_path",
]

#: A mojito tree is the level holding BOTH of these. Anything else is
#: someone pointing one directory too high or too low, which is the
#: mistake this whole resolution exists to absorb.
MOJITO_SUBDIRS = ("catalogues", "data")

#: Where mojito data lives when nobody says. The cluster layout is flat;
#: the laptop cache nests two levels deeper under build_truth.py's
#: convention, which is exactly the asymmetry people get wrong.
_MOJITO_CANDIDATES = (
    "/shared/data/mojito_cache",
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0",
    "~/.mojito_cache",
)


def _mojito_ok(p) -> bool:
    return bool(p) and all(
        os.path.isdir(os.path.join(p, d)) for d in MOJITO_SUBDIRS)


def _mojito_try(p):
    """``p``, or its nested brickmarket layout, if either validates."""
    if not p:
        return None
    p = os.path.expanduser(str(p))
    if _mojito_ok(p):
        return p
    nested = os.path.join(p, "brickmarket", "mojito_light_v1_0_0")
    return nested if _mojito_ok(nested) else None


def _mojito_from_run(run_dir):
    """The path the RUN ITSELF recorded, read back out of the snapshot.

    ``run_settings.log`` carries the ``mojito_data_path`` the job actually
    loaded bricks from, so a snapshot is self-describing: on the machine
    that produced it, nothing needs to be configured at all. On another
    machine the recorded path simply will not exist and the caller falls
    through to the local candidates.
    """
    import glob as _glob

    for pat in ("*_artifacts/run_settings.log", "run_settings.log"):
        for fp in _glob.glob(os.path.join(run_dir, pat)):
            try:
                with open(fp, errors="replace") as fh:
                    txt = fh.read()
            except OSError:
                continue
            import re as _re

            for m in _re.finditer(r"(/[\w./-]*mojito[\w./-]*)", txt):
                hit = _mojito_try(m.group(1))
                if hit:
                    return hit
    return None


def resolve_mojito_path(run_dir=None, explicit=None):
    """``(path, where_it_came_from)``, or ``(None, reason)``.

    ONE resolution for the whole monitor, so "set MOJITO_INFO_PATH first"
    stops being a second step people have to remember. Order:

    1. ``explicit`` (a ``--mojito`` flag) -- a human overriding on purpose;
    2. ``MOJITO_INFO_PATH`` -- the documented knob;
    3. the path the SNAPSHOT ITSELF records, so a tarball rendered on the
       machine that produced it needs no configuration at all;
    4. the legacy vars, so existing runbooks keep working;
    5. the standard cluster and laptop locations.

    Every candidate is validated for ``catalogues/`` and ``data/`` and
    gets one rescue attempt down the nested ``brickmarket`` layout, so
    pointing one level too high still resolves.
    """
    hit = _mojito_try(explicit)
    if hit:
        return hit, "--mojito"
    hit = _mojito_try(os.environ.get("MOJITO_INFO_PATH"))
    if hit:
        return hit, "MOJITO_INFO_PATH"
    if run_dir:
        hit = _mojito_from_run(run_dir)
        if hit:
            return hit, "the run's own run_settings.log"
    for name in ("MOJITO_CAT", "MOJITO_DATA_PATH", "MOJITO_CACHE_DIR"):
        v = os.environ.get(name)
        if v:
            # MOJITO_CAT may name the catalogue FILE; its tree is two up.
            v = os.path.expanduser(v)
            if os.path.isfile(v):
                v = os.path.dirname(os.path.dirname(v))
            hit = _mojito_try(v)
            if hit:
                return hit, name
    for cand in _MOJITO_CANDIDATES:
        hit = _mojito_try(cand)
        if hit:
            return hit, "the default search path"
    return None, ("no directory holding catalogues/ and data/ was found "
                  "(tried --mojito, MOJITO_INFO_PATH, the run's own "
                  "settings, MOJITO_CAT/DATA_PATH/CACHE_DIR, and "
                  + ", ".join(_MOJITO_CANDIDATES) + ")")


def generator_path() -> str:
    """Absolute path to ``_generator.py``.

    Resolved from this package's own ``__file__`` rather than by
    importing the generator -- importing it raises the ``SystemExit(0)``
    its guard exists to raise.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_generator.py")


def default_out_path(run_dir: str) -> str:
    """Where the page lands when the caller does not say.

    A SIBLING of the run directory, not a file inside it (user ruling
    2026-09-26: "same path as to the --snapshot folder (not in the folder
    same path as that folder)"), so both artifacts land together:

        /data/gf_prod_6mo_v9_4gpu/              <- the run
        /data/gf_prod_6mo_v9_4gpu_monitor.html  <- the page
        /data/gf_prod_6mo_v9_4gpu_snapshot.tar.gz

    ⚠ CONSEQUENCE, stated because it reverses an earlier guarantee: the
    tar no longer CONTAINS the page. It used to, precisely because the
    page was written inside the directory being archived. Ship both files
    if the report needs to travel with the data.
    """
    return os.path.abspath(str(run_dir).rstrip("/")) + "_monitor.html"


def describe_run(run_dir: str) -> Optional[str]:
    """One line saying HOW FAR ALONG the run is, or ``None``.

    The generator never printed this and neither did the wrappers, so the
    only way to learn what iteration a page describes was to open the
    page. That is the first thing anyone wants to know about a snapshot --
    especially when re-rendering a tarball that may be the same one as
    last time.

    Rewind- and torn-row aware for the same reasons
    :func:`build_truth.latest_iteration` is: ``log_like`` rows are
    preallocated so the dataset length is capacity, not progress; a
    rewound store's ``iteration`` attr is lower than its filled extent and
    the rows past it are a discarded trajectory. Never raises -- this is a
    convenience line, not a gate.
    """
    import glob as _glob

    try:
        import h5py
        import numpy as np

        cands = [p for p in _glob.glob(os.path.join(run_dir, "*.h5"))
                 if "testing" in os.path.basename(p)
                 and "CORRUPT" not in p and "backup" not in p]
        if not cands:
            return None
        store = max(cands, key=os.path.getmtime)
        with h5py.File(store, "r") as f:
            g = f["global_fit"]
            ll = g["log_like"][:, 0, 0, :]
            filled = np.where(np.any(ll != 0.0, axis=1))[0]
            if not filled.size:
                return f"{os.path.basename(store)}: no filled iterations yet"
            nit = int(filled.max()) + 1
            attr = g.attrs.get("iteration")
            rewound = attr is not None and 0 < int(attr) < nit
            if rewound:
                nit = int(attr)
            last = nit - 1
            L = ll[last]
            bits = [f"iteration {last} ({nit} stored)"]
            if rewound:
                bits.append("REWOUND")
            if "inds/gb" in g:
                inds = g["inds/gb"]
                per = [int(np.count_nonzero(inds[last, 0, 0, w]))
                       for w in range(inds.shape[3])]
                bits.append(f"{sum(per):,} GB leaves {per}")
            bits.append(f"max lnL {L.max():.1f} (walker spread "
                        f"{L.max() - L.min():.1f})")
            return "; ".join(bits)
    except Exception as e:                        # noqa: BLE001
        logger.debug("describe_run failed: %r", e)
        return None


#: The generator looks for this EXACT name, in the run directory then the
#: CWD. The "3to21" is legacy -- the band is read from the file's own
#: ``band`` stamp, not from its name -- so a 6-month or 1-year set still
#: has to be called this or the page will not see it.
TRUTH_NAME = "gb_truth_3to21.npz"


def _store_tobs(run_dir):
    """Observation time of the run, or ``None``."""
    import glob as _glob

    try:
        import h5py

        cands = [p for p in _glob.glob(os.path.join(run_dir, "*.h5"))
                 if "testing" in os.path.basename(p) and "CORRUPT" not in p
                 and "backup" not in p]
        if not cands:
            return None
        with h5py.File(max(cands, key=os.path.getmtime), "r") as f:
            a = dict(f["global_fit/domain_settings/args"].attrs)
        return float(a["0"]) * float(a["1"]) * float(a["2"])
    except Exception:                             # noqa: BLE001
        return None


def check_truth(run_dir):
    """``(path, note)`` for the truth set this run can actually use.

    ``path`` is ``None`` when the page will have no detectability
    overlays -- no completeness denominator, no detectable-source target
    line on the leaf-count and occupancy panels.

    THE TWO WAYS IT GOES WRONG ARE DIFFERENT and the note says which:

    * absent -- nothing named :data:`TRUTH_NAME` beside the store or in
      the CWD;
    * MISMATCHED Tobs -- a set exists but was built for a different
      observation time. Detectability is per-Tobs (the FD bin width, the
      waveform duration and hence the optimal SNR all scale with it), so
      a 3-month set is simply the wrong denominator for a 6-month run and
      the page is right to refuse it.

    Also reports an ANALYTIC-ephemeris stamp, because that set is usable
    but wrong above ~5 mHz and the page will say so anyway.
    """
    import numpy as np

    tobs = _store_tobs(run_dir)
    for p in (os.path.join(run_dir, TRUTH_NAME), TRUTH_NAME):
        if not os.path.exists(p):
            continue
        try:
            T = np.load(p)
            t_have = float(T["tobs"]) if "tobs" in T.files else None
            orb = str(T["orbits"]) if "orbits" in T.files else "analytic"
        except Exception as e:                    # noqa: BLE001
            return None, f"{p} is unreadable ({type(e).__name__})"
        if tobs and t_have and abs(t_have - tobs) > 1.0:
            return None, (
                f"{os.path.basename(p)} is for Tobs {t_have / 86400:.0f} d "
                f"but this run is {tobs / 86400:.0f} d -- detectability is "
                "per-Tobs, so the page will refuse it")
        note = f"{os.path.basename(p)}, {int(T['det'].sum()):,} detectable"
        if not orb.startswith("mojito"):
            note += " (⚠ ANALYTIC ephemeris -- wrong above ~5 mHz)"
        return p, note
    return None, f"no {TRUTH_NAME} beside the store or in the CWD"


def build_truth_set(run_dir, extra_argv=()):
    """Generate :data:`TRUTH_NAME` INTO ``run_dir``. Returns the path.

    Straight python -- ``build_truth.main`` is called in this
    interpreter, no subprocess. Written into the RUN DIRECTORY on purpose:
    that is where the generator looks first, and it means the next
    snapshot tar carries its own truth set, so every page built from that
    tar has its overlays with nothing configured locally.

    Tens of minutes, CPU-only (~9k waveforms). Defaults now do the right
    thing: the newest usable iteration for the noise, and the INJECTED
    mojito L1 orbits rather than the analytic ephemeris.
    """
    import glob as _glob

    cands = [p for p in _glob.glob(os.path.join(run_dir, "*.h5"))
             if "testing" in os.path.basename(p) and "CORRUPT" not in p
             and "backup" not in p and "_extract" not in p]
    if not cands:
        # An extract IS usable here: build_truth reads the noise chains and
        # the domain settings, both of which the reduced store keeps in
        # full. Preferring a live store is about freshness, not capability.
        cands = [p for p in _glob.glob(os.path.join(run_dir, "*_extract.h5"))]
    if not cands:
        raise FileNotFoundError(f"no store under {run_dir} to build against")
    store = max(cands, key=os.path.getmtime)
    out = os.path.join(run_dir, TRUTH_NAME)

    # HAND IT THE MOJITO TREE. build_truth needs it TWICE -- once for the
    # GB catalogue and once for the L1 bricks the injected orbits come
    # from -- and it reads both from the environment. Without this the
    # caller had to export MOJITO_INFO_PATH by hand or the build fell back
    # to the ANALYTIC ephemeris and then died on the catalogue, which is
    # exactly what happened on the cluster (2026-09-27): "env set: NONE".
    _moj, _src = resolve_mojito_path(run_dir=run_dir)
    _saved = os.environ.get("MOJITO_INFO_PATH")
    if _moj:
        os.environ["MOJITO_INFO_PATH"] = _moj
        logger.info("build_truth: mojito data from %s (%s)", _moj, _src)

    from .build_truth import main as _bt_main

    argv = [store, "--out", out, *extra_argv]
    logger.info("building the truth set: build_truth %s", " ".join(argv))
    try:
        rc = _bt_main(argv)
    except SystemExit as e:
        # build_truth is a CLI: resolve_catalogue and friends raise
        # SystemExit, which `except Exception` does NOT catch. Left
        # alone it propagates through the monitor and kills the whole
        # command -- which is exactly what happened on the cluster
        # (2026-09-27): the catalogue was not found, the truth build
        # aborted, AND THE PAGE WAS NEVER WRITTEN. A missing truth set
        # must cost you overlays, not the report.
        raise RuntimeError(
            f"build_truth exited: {e.code}"
            if not isinstance(e.code, str) else str(e.code)) from e
    finally:
        if _saved is None:
            os.environ.pop("MOJITO_INFO_PATH", None)
        else:
            os.environ["MOJITO_INFO_PATH"] = _saved
    if rc not in (0, None):
        raise RuntimeError(f"build_truth exited {rc}")
    return out


def _add_info_env(add_info):
    """``GF_MONITOR_ADD_INFO`` for an explicit ``add_info`` list, or ``None``
    when the caller did not give one (the build then inherits the
    environment's -- which is how the in-run monitor picks up the launcher's
    export). See ``_add_info``."""
    if add_info is None:
        return None
    from ._add_info import add_info_env_value

    return add_info_env_value(add_info)


def build_monitor_in_process(run_dir: str, out_path: str,
                             mojito: Optional[str] = None,
                             add_info: Optional[list] = None) -> str:
    """Render the page in THIS interpreter -- one python, no child.

    ⚠ READ THIS BEFORE PREFERRING IT. The generator sets its own dark
    theme in ``plt.rcParams`` near the top and ~700 lines later imports
    ``...erebor.noise``, whose chain reaches eryn, which calls
    ``plt.style.use(["science"])`` as an IMPORT SIDE EFFECT. Run as a
    script, that side effect lands AFTER the generator's block and the
    science style wins. In this process ``lisatools`` is already imported,
    so the side effect has already happened and the generator's block wins
    instead.

    Measured consequence (2026-09-26): the "instrument noise posteriors"
    panel renders differently -- 25722 vs 27586 base64 chars, 25806 pixels
    across the whole plot area -- with all 760 other lines of the page
    identical. Nothing else changes.

    So this is NOT byte-identical to :func:`build_monitor` and cannot be
    made so without the generator taking control of its own style
    ordering. It is offered because one process is sometimes worth more
    than one panel, not because the difference is imaginary.

    It also keeps the build's ~2.5 GB peak RSS in the caller's heap and
    puts a matplotlib fault in the caller's process, which is why the
    saver rank -- the run's only writer -- must not use it.
    """
    import gc

    from ._add_info import ENV as _ADD_ENV

    argv = [generator_path(), str(run_dir), str(out_path)]
    saved_argv, saved_env = sys.argv, os.environ.get("MOJITO_INFO_PATH")
    saved_add = os.environ.get(_ADD_ENV)
    _moj, _src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    sys.argv = argv
    if _moj:
        os.environ["MOJITO_INFO_PATH"] = _moj
        logger.info("monitor: mojito data from %s (%s)", _moj, _src)
    # ADDED RUNS: an explicit list (the CLI's --add-info) is handed to the
    # generator through its knob for this build only; without one the
    # generator reads whatever the environment carries (the in-run case).
    _add_val = _add_info_env(add_info)
    if _add_val is not None:
        os.environ[_ADD_ENV] = _add_val
    # ⚠ OWN THE NAMESPACE -- do NOT go back to ``runpy.run_path``.
    # run_path hands the module globals back only on SUCCESS, so a
    # generator that RAISED left ``ns`` None, the teardown below closed
    # NOTHING, and the store stayed open read-only inside the run's only
    # writer. Refcounting did not save it either: a module globals dict is
    # a reference cycle (every function's ``__globals__`` points back at
    # it), so it outlives the traceback and waits for the cyclic GC. The
    # next save was then
    #   OSError: Unable to synchronously open file
    #            (file is already open for read-only)
    # which is how the 3-month run died on 2026-09-28. Executing the code
    # in a dict we created means the handles are reachable on EVERY path.
    # ``__name__`` must stay "__main__": the generator guards on it and
    # raises SystemExit(0) otherwise.
    # Covered by tests/test_monitor_in_process_handle_leak.py.
    _gen = generator_path()
    ns = {"__name__": "__main__", "__file__": _gen}
    try:
        with open(_gen, "rb") as _fh:
            _code = compile(_fh.read(), _gen, "exec")
        exec(_code, ns)                           # noqa: S102
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        # CLOSE THE GENERATOR'S HDF5 HANDLES. It is a script: it opens the
        # store at module level and never closes it, because a child
        # process exits and takes the handle with it. In THIS process the
        # handle survives, and h5py then refuses to create the snapshot's
        # ``*_extract.h5``:
        #
        #   OSError: Unable to synchronously create file (unable to
        #            truncate a file which is already open)
        #
        # which broke `--snapshot` the moment the render stopped being a
        # subprocess -- the page was written and the tar silently was not.
        # runpy hands back the module globals, so the handles are reachable
        # without the generator having to grow a teardown.
        for _v in list((ns or {}).values()):
            try:
                import h5py

                if isinstance(_v, h5py.File):
                    _v.close()
                # Dataset too, not only Group: the generator keeps
                # ``g["log_like"]`` style handles at module level and a
                # live Dataset holds its File open just as firmly.
                elif isinstance(_v, (h5py.Group, h5py.Dataset)) and _v.file:
                    _v.file.close()
            except Exception:                     # noqa: BLE001
                pass
        # Break the globals cycle now rather than leaving it to whenever
        # the collector next runs -- this process is the run's only writer
        # and the next thing it does is open the store for APPEND.
        ns.clear()
        gc.collect()
        sys.argv = saved_argv
        if saved_env is None:
            os.environ.pop("MOJITO_INFO_PATH", None)
        else:
            os.environ["MOJITO_INFO_PATH"] = saved_env
        if saved_add is None:
            os.environ.pop(_ADD_ENV, None)
        else:
            os.environ[_ADD_ENV] = saved_add
    return out_path


def build_monitor(run_dir: str, out_path: Optional[str] = None, *,
                  timeout: float = 1800.0,
                  mojito: Optional[str] = None,
                  in_process: bool = True,
                  check: bool = True,
                  add_info: Optional[list] = None) -> Optional[str]:
    """Build the HTML page for ``run_dir``.

    ``add_info``: run directories to overlay on the page's comparison figures
    (``_add_info``; user request 2026-10-12). ``None`` = whatever
    ``GF_MONITOR_ADD_INFO`` the environment carries (the in-run monitor's
    route); a list (the CLI's ``--add-info``) replaces it for this build, ``[]``
    meaning none.

    ⚠ DOCSTRING CORRECTED 2026-09-27. This used to say the build happens
    "in a FRESH interpreter" and that the subprocess was "a correctness
    requirement, not a preference". That stopped being true when
    ``in_process`` became the default: the reason a child was needed was
    that importing ``lisatools`` pulls eryn's ``plt.style.use(["science"])``
    before the generator sets its own rcParams, which restyled exactly one
    panel. That is now handled by pinning ``font.size`` explicitly
    (``_NOISE_PANEL_FONT``), and both paths were measured byte-identical.
    ``in_process=False`` keeps the child as an escape hatch.

    Returns the path written, or ``None`` when ``check=False`` and the
    build failed.

    Published atomically: the child writes a temp beside the target and
    this renames it into place, so a browser refreshing the page never
    catches half a document.
    """
    run_dir = os.path.abspath(str(run_dir))
    out_path = out_path or default_out_path(run_dir)
    tmp = out_path + ".tmp"
    st = time.perf_counter()
    # RESOLVE THE MOJITO TREE HERE, so "export MOJITO_INFO_PATH first" is
    # not a second step every caller has to remember. The child reads it
    # from the environment, so passing it down is all that is needed; an
    # already-set MOJITO_INFO_PATH still wins inside resolve_mojito_path.
    env = dict(os.environ)
    _moj, _src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    _add_val = _add_info_env(add_info)
    if _add_val is not None:
        from ._add_info import ENV as _ADD_ENV

        env[_ADD_ENV] = _add_val
    if _moj:
        env["MOJITO_INFO_PATH"] = _moj
        logger.info("monitor: mojito data from %s (%s)", _moj, _src)
    else:
        logger.warning(
            "monitor: %s. The page will fall back to the analytic PSD "
            "injection and DROP the residual-spectrum and "
            "data/template/residual panels, with no error on the page "
            "itself.", _src)
    # GF_MONITOR_IN_PROCESS=1 (or in_process=True) renders without a child
    # interpreter -- one python, at the cost of one restyled panel. See
    # build_monitor_in_process for the measurement.
    # STRAIGHT PYTHON FLOW, no child interpreter (user ruling
    # 2026-09-26: "no subprocesses!"). Safe since the font.size pin made
    # the in-process render byte-identical to the child one.
    # GF_MONITOR_SUBPROCESS=1 is the escape hatch for a caller that wants
    # the ~2.5 GB peak and any matplotlib fault kept out of its own
    # process -- the saver rank is the case that might.
    _inproc = in_process and os.environ.get(
        "GF_MONITOR_SUBPROCESS", "0").strip() not in ("1", "true", "yes", "on")
    try:
        if _inproc:
            build_monitor_in_process(run_dir, tmp, mojito=_moj, add_info=add_info)
        else:
            subprocess.run(
                [sys.executable, generator_path(), run_dir, tmp],
                check=True, timeout=timeout, env=env,
                # DO NOT SWALLOW THE GENERATOR'S OWN OUTPUT. It reports
                # the plot count, the VGB corners, the GB cloud size and
                # the missing-panel count -- the only running commentary
                # there is -- and DEVNULL here meant the subprocess path
                # silently showed less than the in-process one.
                stderr=subprocess.PIPE,
            )
        os.replace(tmp, out_path)
    except Exception as e:                        # noqa: BLE001
        _err = getattr(e, "stderr", b"") or b""
        if isinstance(_err, bytes):
            _err = _err.decode("utf-8", "replace")
        if _err:
            _detail = " Last stderr: " + _err.strip()[-800:]
        else:
            # THE IN-PROCESS PATH HAS NO ``.stderr``. Without this the
            # whole failure was one line naming the exception type, which
            # is why the 3-month page could fail on every save with nobody
            # able to say why -- and the same failing page was leaking the
            # store handle that killed the run. The subprocess path gets
            # the child's stderr; this is its equivalent.
            import traceback
            _detail = "\n" + traceback.format_exc().strip()[-4000:]
        logger.warning(
            "monitor page NOT refreshed (%s: %s). Any previous page is "
            "untouched.%s", type(e).__name__, e, _detail)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        if check:
            raise
        return None
    logger.info("monitor page built in %.1f s -> %s",
                time.perf_counter() - st, out_path)
    return out_path


# THE SHORT PAGE (user request 2026-10-03; see _short.py): the topline plus
# the lnL, leaf-count, other-sources, phase-maximised-overlap and current-noise
# panels, in well under a minute, written to RUN_DIR + "_monitor_short.html" --
# a separate file, so the full page above and its default name are unchanged.
# Since 2026-10-11 the same block (_short.build_glance) also opens the full
# page, above its collapsed detail groups.
# Imported last: _short reaches back into this module (generator_path,
# check_truth, resolve_mojito_path) only inside its functions.
from ._short import build_short_monitor, default_short_out_path  # noqa: E402
