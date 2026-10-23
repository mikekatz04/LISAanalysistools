r"""Build ``gb_truth_3to21.npz`` -- the frozen recovery denominator.

The status page (``monitor/_generator.py``, formerly
``scripts/diagnostics/gf_monitor_gen.py``) quotes every completeness / purity /
per-source recovery number against ONE frozen set: the catalogue galactic
binaries detectable (optimal SNR > 7) over the analysed GB band under the
run's own fitted noise at a chosen iteration. The original build script lived
in a session scratchpad and was lost -- which is why this one is IN the repo.
Committed 2026-08-19; validated by reproducing the original count (812
detectable over the then-hardcoded 3-21.94 mHz).

THE BAND IS SETTABLE, AND THE DEFAULT IS A JUDGEMENT CALL. It used to be
hardcoded at 3-21.94 mHz, which threw away most of the GB band: the run's
own ``sub_backend/gb/band_edges`` starts at 0.5556 mHz. ``--flo``/``--fhi``
now set the band and the value used is stamped into BOTH output npz files,
so the monitor quotes it rather than assuming.

The DEFAULT floor is **0.8 mHz** (2026-09-18), above the sampler's own
0.5556 mHz. That is deliberate: see the resolvability warning below. Pass
``--flo 5.5555555556e-4`` for the full analysed band.

A WARNING ABOUT THE LOW-FREQUENCY END -- the reason for the 0.8 mHz floor.
``det`` is and remains an SNR statement: optimal SNR > 7 against a
sensitivity that already includes the fitted galactic foreground, so the
confusion background is accounted for in the DENOMINATOR of the SNR. It is
NOT a resolvability statement. Below ~0.8 mHz the catalogue puts many sources
in every frequency bin, and a source can clear SNR 7 while being hopelessly
blended with its neighbours; charging the run for those in the completeness
denominator measures the catalogue, not the sampler. Even between 0.8 and
3 mHz, read this set as "carries enough signal power to matter" rather than
"is individually recoverable".

DETECTABILITY IS PER-TOBS. The observation time sets the FD bin width, the
waveform duration and hence the optimal SNR itself, so a truth set is only
valid for the run it was built against: at 1 year the same catalogue source
integrates 4x longer and the detectable count grows. ``Tobs`` is therefore
read from the STORE's own domain settings (``global_fit/domain_settings/args``
-> ``attrs["0"] * attrs["1"] * attrs["2"]``, the same expression the monitor
uses), with the 3-month value as the fallback and ``--tobs`` as an explicit
override, and the value used is stamped into both output npz files. The
monitor refuses a truth set whose stamped ``tobs`` does not match the run.

Nothing here is re-derived: the noise, waveform and SNR route are copied from
the monitor's own recovery section (GBGPU ``run_wave`` on CPU, lisatools
``A2TDISens``/``E2TDISens`` fed the run's sampled instrument + foreground).
That is a frequency-domain A/E optimal SNR on the EQUAL-ARM analytic TDI-2
sensitivity (:func:`sens_grids`), not the production likelihood (WDM domain,
unequal-arm noise model): ``snr`` is the monitor's detectability yardstick,
not the run's own SNR.

Shape of the output (the monitor's ``TRU`` contract)::

    f0    (N,)   Hz
    amp   (N,)   GW amplitude
    snr   (N,)   optimal SNR under the frozen noise
    det   (N,)   bool, snr > 7
    phys  (N,9)  GBGPU physical basis [amp, f0, fdot, fddot, phi0, iota,
                 psi, lam(icrs), beta(icrs)]

plus provenance arrays (``store``, ``iteration``, ``psd_params``,
``galfor_params``, ``band``, ``tobs``, ``nw`` and the ``orbits`` stamp,
``"mojito_l1"`` or ``"analytic"``). The ``lam``/``beta`` columns are the
catalogue's ICRS right ascension and declination. The ``phi0`` column is the
SAMPLING-basis phase of :func:`~lisatools.globalfit.recipe.gb_catalogue_to_sampling_basis`
(irrelevant for the optimal SNR).

The amplitude PREFILTER: running waveforms for every one of the 15.5M
catalogue rows is pointless when SNR/amp at fixed f is bounded. A unit-SNR
kappa curve (best-case orientation, coarse f grid, buffered 2x) cuts the
waveform set to the plausibly-detectable tail; the same curve is written to
``kappa_grid.npz`` for the monitor's population panels.

Usage (the monitor CLIs' ``--build-truth`` call :func:`main` in-process via
:func:`lisatools.globalfit.monitor.build_truth_set`, which writes into the run
directory; ``scripts/diagnostics/build_truth.py`` is a shim of this module)::

    OMP_NUM_THREADS=1 python -m lisatools.globalfit.monitor.build_truth \
        STORE.h5 [--iteration N] [--out gb_truth_3to21.npz] [--tobs SEC] \
        [--flo 0.8e-3] [--fhi 2.1944444e-2]

    python -m lisatools.globalfit.monitor RUN_DIR --build-truth

THE NOISE ITERATION DEFAULTS TO THE MOST RECENT SAMPLE (2026-09-26, user
ruling). It was a hardcoded ``78`` -- an iteration of whichever 3-month
store this script first ran against, silently reused for every truth set
since, including on stores that had never been near iteration 78. ``det``
is defined against "the run's own fitted noise", and that means the latest
fit, not a stranger's burn-in. ``--iteration N`` (or ``ITERATION=N``) pins
it; the resolved value is stamped into the npz either way, so a truth set
always says which noise it was built on. Resolution is rewind- and
torn-row aware -- see :func:`latest_iteration`.

The one thing this does NOT change: the truth set is still a FROZEN
denominator. Rebuilding it mid-campaign moves the denominator, so
completeness before and after are not the same measurement. Rebuild when
the noise has materially moved, not every time the page is regenerated.

Runs on CPU. THE WAVEFORM COUNT IS SET BY THE PREFILTER, NOT THE BAND --
measured on the 3-month store at iteration 208:

    floor       in-band rows   waveforms run   detectable
    0.5556 mHz     3 959 524           9 086         1048
    0.8    mHz     1 008 443           9 011         1048
    1      mHz       396 653           8 885         1042

Widening the band multiplies the catalogue rows by 10 and the waveforms by
2%, because `amp * kappa(f)` bounds the SNR and everything below the cut is
discarded before any waveform runs. The prefilter is a vectorised interp
over the catalogue array, so the extra rows cost seconds. Budget tens of
minutes for the ~9k waveforms. Keep the thread pins: laptop policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import h5py

# The band the truth set is built over. The ceiling is the runs' own
# ``sub_backend/gb/band_edges[-1]`` (21.9444 mHz); the FLOOR is a DELIBERATE
# 0.8 mHz, set above the sampler's band_edges[0] of 0.5556 mHz.
#
# Why not the full analysed band: ``det`` is an SNR statement, not a
# resolvability one. Below ~0.8 mHz the catalogue puts many sources in every
# frequency bin, so a binary clears SNR 7 while being inseparable from its
# neighbours. Counting those in the DENOMINATOR charges the run for sources
# nothing could individually recover, which depresses completeness by an
# amount that says nothing about the sampler. 0.8 mHz is the floor; use
# ``--flo 5.5555555556e-4`` to reinstate the full analysed band.
FLO, FHI = 0.8e-3, 2.1944444444e-2
SNR_DET = 7.0
TOBS_3MO = 7776000.0   # the original 3-month set; also the fallback Tobs
DT = 2.5
NW_3MO = 128      # FD points per waveform at 3 months (the monitor's own NW_)

# The GB catalogue this truth set is built against. The default is the
# mojito brick cache as it lands on a laptop; a cluster, a container or a
# second checkout will all put it somewhere else, so the path is
# overridable by --catalogue or MOJITO_CAT and the default is a SEARCH
# rather than a single guess.
MOJITO_CAT_DEFAULT = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/"
    "wdwd_cat_mojito_lite_processed.hdf5")
MOJITO_CAT_NAME = "wdwd_cat_mojito_lite_processed.hdf5"


def resolve_catalogue(explicit=None):
    """Locate the GB catalogue, most specific source first.

    ``--catalogue`` beats ``MOJITO_CAT`` beats ``MOJITO_CACHE_DIR`` beats
    the packaged default. Raises with every path it tried rather than
    failing later inside h5py with a bare "unable to open file", which is
    what made this hard to diagnose the first time.

    The full order is ``explicit``, ``MOJITO_INFO_PATH``, ``MOJITO_CAT``,
    ``MOJITO_CACHE_DIR`` (its nested brickmarket layout),
    :data:`MOJITO_CAT_DEFAULT`. A directory candidate is tried as
    ``<dir>/<name>`` and ``<dir>/catalogues/<name>``.

    Args:
        explicit (str, optional): ``--catalogue`` (file or directory).

    Returns:
        str: Path of ``wdwd_cat_mojito_lite_processed.hdf5``.

    Raises:
        SystemExit: Listing every path tried, when none exists.
    """
    tried = []
    for cand in (
        explicit,
        # MOJITO_INFO_PATH is the ONE knob the rest of the monitor uses,
        # and it names the tree that holds catalogues/ and data/ -- so it
        # locates this file too. It was missing from this chain, which is
        # how a cluster run with the tree at its standard location still
        # failed to find the catalogue (2026-09-27).
        os.environ.get("MOJITO_INFO_PATH"),
        os.environ.get("MOJITO_CAT"),
        (os.path.join(os.environ["MOJITO_CACHE_DIR"], "brickmarket",
                      "mojito_light_v1_0_0", "catalogues", MOJITO_CAT_NAME)
         if os.environ.get("MOJITO_CACHE_DIR") else None),
        MOJITO_CAT_DEFAULT,
    ):
        if not cand:
            continue
        cand = os.path.expanduser(cand)
        # A DIRECTORY IS ACCEPTED, AND IT MAY BE EITHER LEVEL. MOJITO_CAT
        # historically named the directory holding the file
        # (.../catalogues/), while MOJITO_INFO_PATH names the TREE that
        # holds catalogues/ and data/. Trying only <dir>/<name> meant a
        # correct MOJITO_INFO_PATH resolved to
        # /shared/data/mojito_cache/wdwd_...hdf5 -- one level short of the
        # real file -- and the build died having "tried" a path that never
        # existed (cluster, 2026-09-27).
        if os.path.isdir(cand):
            for _c in (os.path.join(cand, MOJITO_CAT_NAME),
                       os.path.join(cand, "catalogues", MOJITO_CAT_NAME)):
                tried.append(_c)
                if os.path.isfile(_c):
                    return _c
            continue
        tried.append(cand)
        if os.path.isfile(cand):
            return cand
    raise SystemExit(
        "could not find the GB catalogue "
        f"({MOJITO_CAT_NAME}).\nTried, in order:\n  "
        + "\n  ".join(tried)
        + "\n\nSet one of:\n"
        "  --catalogue /path/to/wdwd_cat_mojito_lite_processed.hdf5\n"
        "  MOJITO_CAT=/path/to/wdwd_cat_mojito_lite_processed.hdf5\n"
        "  MOJITO_CACHE_DIR=/path/to/.mojito_cache\n"
        "(--catalogue and MOJITO_CAT also accept the containing directory.)"
    )


def store_tobs(store, fallback=TOBS_3MO):
    """Observation time of the run, from the store's own domain settings.

    ``global_fit/domain_settings/args`` carries the WDM/FD grid arguments as
    attrs "0"/"1"/"2" = (Nt, Nf, dt) -- their product is Tobs. This is exactly
    how ``gf_monitor_gen.py`` derives ``SCI_TOBS``, so a truth set built here
    and a page rendered there agree by construction. Falls back to the 3-month
    value when the store predates the group.

    Args:
        store (str): The HDF5 store.
        fallback (float): Tobs [s] when the group is missing (warned).
            Default :data:`TOBS_3MO`.

    Returns:
        float: Tobs [s].
    """
    try:
        with h5py.File(store, "r") as f:
            a = dict(f["global_fit/domain_settings/args"].attrs)
        return float(a["0"]) * float(a["1"]) * float(a["2"])
    except Exception as e:
        print(f"WARNING: no domain settings in {store} ({type(e).__name__}); "
              f"falling back to Tobs = {fallback}")
        return float(fallback)


def nw_for(tobs):
    """FD points per waveform at this Tobs.

    ``N`` has to span the Doppler-modulated bandwidth of the source, which is
    a fixed number of HERTZ (a few tens of 1/yr sidebands); the FD bin is
    1/Tobs, so the bandwidth in BINS grows in proportion to Tobs. Hold the
    3-month value at exactly 128 (reproducing the frozen 812-source set) and
    scale up in powers of two from there -- 512 at one year. Over-sizing N is
    safe (the extra bins are ~zero and the slow part is merely sampled more
    finely); under-sizing truncates the waveform and silently loses SNR.

    Args:
        tobs (float): Observation time [s].

    Returns:
        int: ``128 * max(tobs / TOBS_3MO, 1)`` rounded up to a power of two,
        capped at 2048.

    Example:
        >>> from lisatools.globalfit.monitor.build_truth import nw_for
        >>> nw_for(7776000.0), nw_for(23328000.0), nw_for(31104000.0)
        (128, 512, 512)
    """
    n = NW_3MO * max(float(tobs) / TOBS_3MO, 1.0)
    return int(min(2 ** int(np.ceil(np.log2(n))), 2048))


def galfor_log_sampling_of(store) -> bool:
    """``noise_model_identity["galfor_log_sampling"]`` for this STORE.

    Read it off the store, never off the environment: the env var describes
    the current process, while the stored numbers were written in whatever
    basis that run used. A store predating the key is linear.

    Args:
        store (str): The HDF5 store.

    Returns:
        bool: Whether the store's galfor chain is in the log10 basis.
    """
    from lisatools.globalfit.stock.erebor.noise import read_noise_model_identity

    return bool(read_noise_model_identity(store).get("galfor_log_sampling", False))


def latest_iteration(store):
    """The newest stored row whose noise chains are actually written.

    The default used to be a hardcoded ``78`` -- an iteration of whichever
    3-month store this script was first run against, carried into every
    truth set built since. Nothing justified it: the detectability
    denominator is an SNR statement under "the run's own fitted noise", and
    the run's own fitted noise is the LATEST one, not one from a different
    store's early burn-in. (This file's own measurement table quotes
    iteration 208.)

    Resolving it means repeating the three guards ``gf_monitor_gen.py``
    applies to the same question, because all three produce a row that
    exists and reads as zeros rather than an error:

    1. FILLED EXTENT -- ``log_like`` rows are preallocated, so the dataset
       length is the capacity, not the progress.
    2. REWIND -- ``reset_recipe_stage`` moves the ``iteration`` attr back
       and leaves the rows beyond it holding the discarded pre-rewind
       trajectory until the next ``grow()`` truncates them. Those rows are
       filled and stale, so the attr wins when it is lower.
    3. TORN LAST ROW -- a save step writes datasets one at a time and takes
       ~20 s, so ``log_like`` can be present while ``chain/psd`` and
       ``chain/galfor`` for that row are still zeros. Reading it gives an
       all-zero PSD, which propagates to a NaN sensitivity and a silently
       all-False ``det`` -- the exact failure mode :func:`fitted_noise`
       already warns about, reached by a different route. Step back until
       both chains carry something.

    Raises ``RuntimeError`` rather than returning a guess when no row
    qualifies: a truth set built on zero noise is worse than no truth set.

    Args:
        store (str): The HDF5 store.

    Returns:
        int: The iteration (row index) to build on.

    Raises:
        RuntimeError: When no row is filled or none carries written psd and
            galfor chains.
    """
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        ll = g["log_like"][:, 0, 0, :]
        filled = np.where(np.any(ll != 0.0, axis=1))[0]
        if not filled.size:
            raise RuntimeError(f"{store}: no filled iterations to build on.")
        nit = int(filled.max()) + 1
        attr = g.attrs.get("iteration")
        if attr is not None and 0 < int(attr) < nit:
            print(f"store rewound: iteration attr {int(attr)} < filled rows "
                  f"{nit}; ignoring the discarded rows past it")
            nit = int(attr)
        for it in range(nit - 1, -1, -1):
            psd = np.asarray(g["chain"]["psd"][it, 0, 0, :, 0, :])
            gal = np.asarray(g["chain"]["galfor"][it, 0, 0, :, 0, :])
            if np.any(psd != 0.0) and np.any(gal != 0.0):
                if it != nit - 1:
                    print(f"row {nit - 1} is torn (noise chains unwritten); "
                          f"using iteration {it}")
                return it
    raise RuntimeError(
        f"{store}: no iteration carries written psd AND galfor chains.")


def fitted_noise(store, it):
    """Cold-chain median psd + galfor params at ``it``, in PHYSICAL units.

    The galfor row is de-log10'd when the store says so -- see
    :func:`~lisatools.globalfit.stock.erebor.noise.galfor_params_to_physical`.
    Handing the raw log10 row to the foreground model gives a negative
    ``amp``/``f_1`` and NaN sensitivity, which showed up here not as a crash
    but as an ALL-ZERO ``snr`` and an all-False ``det`` -- a silently empty
    truth set that still passes a "psd/galfor are nonzero" sanity check.

    The median is taken in the SAMPLING basis and converted afterwards,
    matching what the sampler and the monitor page both do; for the log
    columns that is a geometric median-of-logs, which is the meaningful
    centre for a quantity spanning decades.

    Args:
        store (str): The HDF5 store.
        it (int): The iteration (row) to read.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(psd, galfor)``: the walker median
        of the cold-chain psd parameters as sampled (:func:`sens_grids`
        squares the two into the ``LISAModel`` levels) and the physical
        galfor parameters.
    """
    from lisatools.globalfit.stock.erebor.noise import galfor_params_to_physical

    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        psd = np.median(np.asarray(g["chain"]["psd"][it, 0, 0, :, 0, :]),
                        axis=0)
        gal = np.median(np.asarray(g["chain"]["galfor"][it, 0, 0, :, 0, :]),
                        axis=0)
    return psd, galfor_params_to_physical(gal, galfor_log_sampling_of(store))


def sens_grids(psd_p, gal_p, df):
    """A / E noise PSDs (instrument + fitted foreground) on a uniform FD grid.

    Builds a ``LISAModel`` with levels ``psd_p[0]**2`` / ``psd_p[1]**2`` on
    :class:`~lisatools.detector.DefaultOrbits`, adds the
    :class:`~lisatools.stochastic.HyperbolicTangentGalacticForeground` with
    ``gal_p``, and evaluates :class:`~lisatools.sensitivity.A2TDISens` /
    :class:`~lisatools.sensitivity.E2TDISens` (EQUAL-ARM analytic TDI-2) at
    ``f = k * df`` for ``k = 0 .. int(2.35e-2 / df) + 1`` (``f = 0`` replaced
    by ``df``). Also used by the at-a-glance overlap (:mod:`._short`).

    Args:
        psd_p (array-like): The two psd parameters as sampled.
        gal_p (array-like): Physical galfor parameters.
        df (float): Frequency spacing [Hz], ``1 / Tobs``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(sa, se)``, the A and E PSDs on that
        grid (index ``k`` is frequency ``k * df``).
    """
    from lisatools import detector as lisa_models
    from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    lm = lisa_models.LISAModel(psd_p[0] ** 2, psd_p[1] ** 2,
                               lisa_models.DefaultOrbits(), "sampled")
    nk = dict(model=lm, stochastic_params=tuple(gal_p),
              stochastic_function=HTGF)
    ng = int(2.35e-2 / df) + 2
    fg = np.maximum(np.arange(ng) * df, df)
    sa = np.asarray(get_sensitivity(fg, sens_fn=A2TDISens, **nk), float)
    se = np.asarray(get_sensitivity(fg, sens_fn=E2TDISens, **nk), float)
    return sa, se


def _l1_usable(fp):
    """``(ok, reason)`` -- does this brick OPEN and carry the orbit tables?

    WHY THIS EXISTS (cluster failure 2026-09-26):

        OSError: Unable to synchronously open file (truncated file:
        eof = 5747410944, ..., stored_eof = 5932018145)

    A brick whose transfer was interrupted is ~184 MB short of what its
    own HDF5 superblock claims. ``find_l1_brick`` used to return
    ``hits[0]`` -- the first match alphabetically -- with no check, so ONE
    truncated file killed the whole truth build even when intact bricks
    sat beside it.

    Skipping it is not a papering-over: every mojito L1 file carries the
    SAME orbits/ltt tables (that is why any of them will do), so the next
    one is not a degraded substitute, it is the same data. What would be
    wrong is failing over SILENTLY, which is why the caller names every
    brick it skipped and why "found some, none opened" is reported
    differently from "found none".

    The group check matters as well as the open: a file can be complete
    and still be the wrong kind of brick.
    """
    try:
        import h5py

        with h5py.File(fp, "r") as fh:
            missing = [g for g in ("ltts", "orbits") if g not in fh]
            if missing:
                return False, f"opens, but has no {'/'.join(missing)} group"
        return True, ""
    except Exception as e:                       # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


#: Bricks already reported unusable in this process. ``find_l1_brick`` runs
#: several times per monitor build (the truth build, the at-a-glance overlap,
#: each ``--add-info`` run's overlap), and each used to print the same list
#: again (2026-10-13).
_REPORTED_BAD_BRICKS = set()


def brick_loss_note(fp):
    """What the MONITOR loses when L1 brick ``fp`` cannot be opened, by type.

    Orbits never: every L1 brick carries the same orbit / light-travel-time
    tables, so the truth build and the overlap waveforms take the next
    readable one. What a broken brick does cost is its OWN content:

    * ``NOISE_*`` -- the injected Soms_d / Sa_a the noise panels draw as truth
      (full page and at-a-glance) fall back to the round analytic injection
      (1.5e-11 / 3e-15; the brick refit agrees to 3e-6 / 4e-5), and the page
      says so in its Notes.
    * a source brick (``GB`` / ``VGB`` / ``MBHB`` / ``EMRI`` / ``SOBHB``) --
      the full page's data / template / residual rebuild leaves that source
      out of the data it sums, with a Notes line, but ONLY for a run that did
      not read mojito's ``COMBINED`` stream (a COMBINED run's page reads the
      combined brick and nothing else).
    * ``COMBINED`` -- the page sums the run's own per-type bricks instead
      (Notes line); with none of those readable either, the data / template /
      residual and whitened-residual panels are not drawn.

    Args:
        fp (str): The brick path (the type is the basename's prefix before
            the first ``_``).

    Returns:
        str: One clause for the warning line.
    """
    base = os.path.basename(str(fp))
    kind = base.split("_", 1)[0].upper()
    if kind == "NOISE":
        return ("noise truth lines fall back to the analytic injection "
                "(full page + at-a-glance)")
    if kind == "COMBINED":
        return ("the page's data / template / residual rebuild sums the run's per-type "
                "bricks instead")
    if kind in ("GB", "VGB", "MBHB", "EMRI", "SOBHB", "GALFOR"):
        return (f"this {kind} source is left out of the page's data sum, but only "
                "on a run that did not read COMBINED")
    return "nothing the monitor reads"


def report_skipped_bricks(skipped, used=None, out=print):
    """The one warning for unreadable L1 bricks, at most once per brick per
    process: a header, then ONE line per brick (path, error, what the page
    loses), then what to do. Returns the lines printed.

    Args:
        skipped (list[tuple[str, str]]): ``(path, reason)`` per unusable brick.
        used (str, optional): The brick read instead, named in the header.
        out (Callable[[str], None]): Printer. Default :func:`print`.

    Returns:
        list[str]: The lines printed (empty when every brick was already
        reported).
    """
    new = [(fp, why) for fp, why in skipped if fp not in _REPORTED_BAD_BRICKS]
    if not new:
        return []
    _REPORTED_BAD_BRICKS.update(fp for fp, _ in new)
    lines = [f"WARNING: skipped {len(new)} unusable mojito L1 brick(s)"
             + (f"; orbits read from {os.path.basename(used)} instead (every L1 "
                "brick carries the same orbit tables, so the truth set loses nothing)"
                if used else "") + ":"]
    for fp, why in new:
        lines.append(f"    {fp}: {why.splitlines()[0][:160] if why else 'unreadable'}"
                     f" -> {brick_loss_note(fp)}")
    lines.append("    These files are truncated or corrupt on disk: re-fetch them into "
                 "the mojito cache. The page notes each loss under Notes.")
    for ln in lines:
        out(ln)
    return lines


def find_l1_brick(path=None):
    """Any mojito L1 file -- they all carry the same orbits/ltt tables.

    THE ENV CHAIN IS THE SAME ONE THE PAGE USES (fixed 2026-09-26, user
    report: "I cannot get the html builder to find the proper orbits. Even
    when I provide MOJITO_CAT in both ways it does not work.").

    It used to read ONLY ``MOJITO_DATA_PATH`` and ``~/.mojito_cache``.
    ``MOJITO_CAT`` -- the variable this script documents for finding the
    catalogue, and the obvious one to reach for -- was never consulted
    here, so setting it resolved the catalogue perfectly and left the
    ORBITS on the analytic fallback. The only symptom was one ``print``
    that scrolls past in a tens-of-minutes build, and a truth set stamped
    ``analytic`` that the monitor then correctly complains about. Nothing
    told you which knob was the wrong one.

    Order: explicit path > MOJITO_INFO_PATH (the one knob) > MOJITO_CAT
    (directory, or the wdwd_cat file, in which case two levels up) >
    MOJITO_DATA_PATH > MOJITO_CACHE_DIR > the laptop default. Every entry
    is searched recursively for ``*_L1_*.h5``.

    Bricks that do not open or lack the ``ltts`` / ``orbits`` groups
    (:func:`_l1_usable`) are skipped and reported once
    (:func:`report_skipped_bricks`). An explicit ``path`` is returned even
    when unusable (with a warning), never replaced.

    Args:
        path (str, optional): An explicit brick (``--l1-brick``).

    Returns:
        str or None: The first usable brick in sorted order, or ``None``.
    """
    import glob
    if path:
        # An EXPLICIT --l1-brick is never silently replaced: the caller
        # named this file on purpose, so a broken one is an error to
        # report, not a reason to pick a different brick behind their back.
        ok, why = _l1_usable(path)
        if not ok:
            print(f"WARNING: --l1-brick {path} is not usable ({why}). "
                  "Not substituting another brick for a file you named "
                  "explicitly; drop the flag to search instead.")
        return path

    def _cat_root(v):
        # MOJITO_CAT accepts the containing directory OR the catalogue
        # file itself (.../catalogues/wdwd_cat_*.hdf5) -- match
        # _resolve_catalogue, which is what makes the same value work for
        # both lookups.
        v = os.path.expanduser(v)
        return os.path.dirname(os.path.dirname(v)) if os.path.isfile(v) else v

    roots = []
    for name in ("MOJITO_INFO_PATH", "MOJITO_CAT", "MOJITO_DATA_PATH",
                 "MOJITO_CACHE_DIR"):
        v = os.environ.get(name)
        if v:
            roots.append(_cat_root(v) if name == "MOJITO_CAT"
                         else os.path.expanduser(v))
    roots.append(os.path.expanduser("~/.mojito_cache"))

    skipped = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", "*_L1_*.h5"),
                                recursive=True))
        for fp in hits:
            ok, why = _l1_usable(fp)
            if ok:
                if skipped:
                    report_skipped_bricks(skipped, used=fp)
                return fp
            skipped.append((fp, why))
    if skipped:
        # Do NOT fail silently into the analytic fallback here: a
        # directory full of BROKEN bricks is a different problem from a
        # directory with none, and only this distinguishes them.
        print(f"WARNING: found {len(skipped)} mojito L1 brick(s) and NONE "
              f"could be opened; the orbits fall back to the analytic ephemeris:")
        report_skipped_bricks(skipped)
    return None


def l1_orbits(stride=200, path=None, frame="icrs"):
    """The INJECTED (mojito L1) orbits, with the light-travel-time table strided.

    THIS IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN RIGHT AND WRONG.
    Measured at 7.44-7.60 mHz on the 6mo store (2026-09-23), swapping the
    analytic ``DefaultOrbits`` ephemeris for these::

        template-vs-injection overlap   0.052 -> 0.580,  0.368 -> 0.982,
                                        0.262 -> 0.962
        optimal SNR                     7.96 -> 8.69,   32.81 -> 42.67

    The analytic ephemeris differs from the mojito orbits by the full
    annual-Doppler phase at the run epoch, so anything above a few mHz --
    SNRs, overlaps, the detectable denominator -- is simply a different
    number. Phase-MAXIMISED overlaps do not rescue it.

    Why striding is exact: the C++ side interpolates ltt linearly in
    ``(t - ltt_t0) / ltt_dt`` (``Detector.cu`` ``get_window`` ->
    ``interpolate``) and light travel times vary on ORBITAL timescales -- an
    8.3 s travel time changing by ~1% over a year. Sampling them every
    ``stride * 2.5`` s instead of every 2.5 s leaves a linear-interpolation
    error of ~1e-15 s, and turns a 1.2 GB table (plus its C++ copy) into
    1.5 MB. That is what lets this run on a laptop.

    Returns ``(orbits, path)``, or ``(None, reason)`` when no brick is found.

    The orbits are a CPU :class:`~lisatools.detector.L1Orbits` subclass
    (positions and velocities read as stored; the ltt table and its time
    sampling strided) with ``linear_interp_dt=500.0``, configured before
    return.
    The mojito L1 data are ICRS, so the default ``frame="icrs"`` matches the
    ICRS sky the truth build and the overlap waveforms feed GBGPU.

    Args:
        stride (int): Keep every ``stride``-th light-travel-time sample.
            Default ``200`` (500 s at the 2.5 s brick cadence).
        path (str, optional): Explicit brick, see :func:`find_l1_brick`.
        frame (str): Frame label passed to ``L1Orbits``. Default ``"icrs"``.

    Returns:
        tuple: ``(L1Orbits, str)`` with the brick path, or ``(None, str)``
        with the reason.
    """
    from lisatools import detector as lisa_models
    fp = find_l1_brick(path)
    if fp is None:
        return None, ("no mojito L1 brick under MOJITO_DATA_PATH or "
                      "~/.mojito_cache")

    class _StridedL1(lisa_models.L1Orbits):
        def _setup(self):
            with self.open() as f:
                self.ltt = np.ascontiguousarray(f.ltts.ltts[::stride])
                try:
                    self.ltt_t = f.ltts.time_sampling.t(slice(0, None, stride))
                except TypeError:
                    self.ltt_t = f.ltts.time_sampling.t()[::stride]
                self.x_base = f.orbits.positions[:]
                self.v_base = f.orbits.velocities[:]
                self.sc_t_base = f.orbits.time_sampling.t()
                self.size_base = self.sc_t_base.shape[0]
                self.dt_base = float(f.orbits.time_sampling.dt)
                self.ltt_dt = float(f.ltts.time_sampling.dt) * stride
                self.sc_dt = f.orbits.time_sampling.dt
                self.ltt_t0 = float(self.ltt_t[0])
                self.sc_t0 = float(self.sc_t_base[0])

    orb = _StridedL1(fp, force_backend="cpu", frame=frame,
                     linear_interp_dt=500.0)
    orb._ensure_configured()
    return orb, fp


def catalogue_phys(t_ref, flo=FLO, fhi=FHI, catalogue=None):
    """(N,9) GBGPU physical rows for every catalogue GB in [flo, fhi].

    Column conventions copied from the run's own catalogue path
    (``gb_catalogue_to_sampling_basis`` takes the parameters at the catalogue
    reference epoch, with no evolution, in the ICRS frame;
    ``both_transforms`` of the 8-col basis is what the sampler feeds
    GBGPU) -- here we only need the physical 9-vector, which
    ``gb_catalogue_to_sampling_basis`` exposes directly through the run's
    transform container.

    The rows are built from the 8-column sampling rows as ``[exp(lnA),
    f0 [Hz], fdot, 0, phi0 (sampling basis), arccos(cos_iota), psi,
    alpha (ICRS RA), arcsin(sin_delta) (ICRS Dec)]``. The catalogue is read
    with :func:`resolve_catalogue` and cut on ``GW22FrequencySSBFrame``.

    Args:
        t_ref (float): Accepted but not used (the catalogue parameters are
            taken at their reference epoch).
        flo (float): Band floor [Hz]. Default :data:`FLO`.
        fhi (float): Band ceiling [Hz]. Default :data:`FHI`.
        catalogue (str, optional): Forwarded to :func:`resolve_catalogue`.

    Returns:
        np.ndarray: Shape ``(N, 9)``.
    """
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    cat = resolve_catalogue(catalogue)
    print(f"catalogue: {cat}")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]
        f0 = np.asarray(b["GW22FrequencySSBFrame"][:], float)
        sel = (f0 >= flo) & (f0 <= fhi)
        entry = {k: np.asarray(b[k][sel]) for k in (
            "Amplitude", "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame", "InclinationAngle",
            "PolarisationAngle", "RightAscension", "Declination",
            "TrueAnomaly", "TimeReferenceSSBFrame", "ChirpMassSSBFrame",
            "LuminosityDistance", "Eccentricity")}
    # 8-col sampling basis [lnA, f0 mHz, fdot, phi0, cos_i, psi, lam, sinb]
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    phys = np.column_stack([
        np.exp(rows[:, 0]), rows[:, 1] * 1e-3, rows[:, 2],
        np.zeros(len(rows)), rows[:, 3], np.arccos(rows[:, 4]),
        rows[:, 5], rows[:, 6], np.arcsin(np.clip(rows[:, 7], -1, 1)),
    ])
    return phys


def snr_progress_due(done, total, shown):
    """``(print_now, shown)`` for the ``snr:`` progress line of :func:`opt_snr`.

    At most one line per QUARTER of ``total`` and always one at the end
    (user request 2026-10-13: the monitor log carried 51 lines of
    ``snr: 20000/1008443`` ... ``snr: 1008443/1008443``). ``shown`` is the
    number of quarters already reported.

    Args:
        done (int): Rows finished.
        total (int): Rows in all.
        shown (int): Quarters already reported.

    Returns:
        tuple[bool, int]: Whether to print now, and the updated ``shown``.

    Example:
        >>> from lisatools.globalfit.monitor.build_truth import snr_progress_due
        >>> snr_progress_due(250, 1000, 0), snr_progress_due(300, 1000, 1)
        ((True, 1), (False, 1))
        >>> snr_progress_due(1000, 1000, 1)
        (True, 4)
    """
    total = max(int(total), 1)
    q = min(4, (4 * int(done)) // total)
    if int(done) >= total:
        return True, 4
    if q > shown:
        return True, q
    return False, shown


def opt_snr(phys, sa, se, gbw, df, tobs, nw, batch=20000, progress=True):
    """Optimal SNR of each row: ``sqrt(4 df sum(|A|^2/SA + |E|^2/SE))``.

    ``tobs`` is the run's observation time -- it enters BOTH the waveform
    (``T=`` in ``run_wave``, which sets how long the source integrates) and
    the inner-product measure ``df = 1/tobs``; ``sa``/``se`` must already be
    sampled on that same ``df`` grid. ``progress=False`` prints nothing (the
    kappa prefilter's eight small calls); otherwise at most one ``snr:`` line
    per quarter of the rows, the last at completion.

    Waveforms are GBGPU TDI-2 A/E at ``dt = DT`` (2.5 s); a row whose
    waveform window falls outside the ``sa`` grid keeps SNR 0.

    Args:
        phys (np.ndarray): GBGPU physical rows, shape ``(N, 9)``
            (:func:`catalogue_phys` order).
        sa (np.ndarray): A-channel PSD on the ``k * df`` grid.
        se (np.ndarray): E-channel PSD on the same grid.
        gbw (gbgpu.gbgpu.GBGPU): A CPU instance with its orbits attached.
        df (float): Frequency spacing [Hz].
        tobs (float): Observation time [s].
        nw (int): FD points per waveform (:func:`nw_for`).
        batch (int): Rows per ``run_wave`` call. Default ``20000``.
        progress (bool): Print ``snr:`` progress lines. Default ``True``.

    Returns:
        np.ndarray: Optimal SNR per row, shape ``(N,)``.
    """
    out = np.zeros(len(phys))
    _shown = 0
    # One batch holds ``batch x nw`` complex A and E. That was harmless at
    # 4.7k rows and nw=128; over the full band the kept set is tens of
    # thousands and a 1-year build carries nw=512, so the knob is exposed --
    # drop it on a memory-constrained host rather than editing this file.
    B = int(batch)
    for lo in range(0, len(phys), B):
        p = phys[lo:lo + B]
        gbw.run_wave(*[np.ascontiguousarray(p[:, k]) for k in range(9)],
                     N=nw, T=tobs, dt=DT, tdi2=True, tdi_channel_setup="AE")
        A = np.asarray(gbw.A); E = np.asarray(gbw.E)
        s = np.asarray(gbw.start_inds).astype(int)
        for i in range(len(p)):
            if s[i] < 0 or s[i] + nw > sa.size:
                continue
            _sa = sa[s[i]:s[i] + nw]; _se = se[s[i]:s[i] + nw]
            out[lo + i] = np.sqrt(
                4 * df * (np.abs(A[i]) ** 2 / _sa
                          + np.abs(E[i]) ** 2 / _se).real.sum())
        _done = min(lo + B, len(phys))
        _due, _shown = snr_progress_due(_done, len(phys), _shown)
        if progress and _due:
            print(f"  snr: {_done}/{len(phys)}", flush=True)
    return out


def make_parser():
    """The CLI, split out so the resolved defaults can be tested.

    ``--iteration``'s default is evaluated HERE, at ``add_argument`` time,
    which is why this has to be a function rather than a module-level
    parser: a parser built at import would freeze whatever ``ITERATION``
    happened to be set to then.

    Returns:
        argparse.ArgumentParser: Positional ``store``; options
        ``--iteration``, ``--out`` (default ``gb_truth_3to21.npz``),
        ``--kappa-out`` (default ``kappa_grid.npz``), ``--tobs``, ``--nw``,
        ``--flo`` / ``--fhi`` (defaults :data:`FLO` / :data:`FHI`),
        ``--catalogue``, ``--l1-brick``, ``--analytic-orbits`` and
        ``--batch`` (default 20000).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store")
    ap.add_argument("--iteration", type=int,
                    default=(int(os.environ["ITERATION"])
                             if os.environ.get("ITERATION") else None),
                    help="store iteration whose fitted noise sets the "
                         "detectability denominator. DEFAULT: the most "
                         "recent usable sample (rewind- and torn-row "
                         "aware). Env ITERATION sets it too; the flag wins.")
    ap.add_argument("--out", default="gb_truth_3to21.npz")
    ap.add_argument("--kappa-out", default="kappa_grid.npz")
    ap.add_argument("--tobs", type=float, default=None,
                    help="observation time [s]; default: read from the "
                         "store's global_fit/domain_settings/args")
    ap.add_argument("--nw", type=int, default=None,
                    help="FD points per waveform; default: 128 scaled by "
                         "Tobs/3 months, rounded up to a power of two")
    ap.add_argument("--flo", type=float, default=FLO,
                    help=f"band floor [Hz]; default {FLO:.10g} = the GB "
                         "band_edges[0] of the erebor stores")
    ap.add_argument("--fhi", type=float, default=FHI,
                    help=f"band ceiling [Hz]; default {FHI:.10g} = the GB "
                         "band_edges[-1] of the erebor stores")
    ap.add_argument("--catalogue", default=None,
                    help="GB catalogue hdf5, or the directory holding it. "
                         "Overrides MOJITO_CAT and MOJITO_CACHE_DIR; "
                         f"default {MOJITO_CAT_DEFAULT}")
    ap.add_argument("--l1-brick", default=None,
                    help="explicit mojito L1 .h5 to read the orbits/ltt "
                         "tables from (any L1 file will do -- they all "
                         "carry the same tables). Overrides the "
                         "MOJITO_INFO_PATH / MOJITO_CAT / MOJITO_DATA_PATH "
                         "/ MOJITO_CACHE_DIR search.")
    ap.add_argument("--analytic-orbits", action="store_true",
                    help="build on DefaultOrbits instead of the injected "
                         "mojito L1 orbits. Reproduces truth sets made before "
                         "2026-09-23; wrong above a few mHz (see l1_orbits).")
    ap.add_argument("--batch", type=int, default=20000,
                    help="waveform rows per run_wave call (default 20000); "
                         "lower it if the build runs out of memory")
    return ap


def main(argv=None):
    """Build the truth set and the kappa grid for one store (the CLI).

    Steps: resolve Tobs, the FD points per waveform and the noise iteration
    (:func:`latest_iteration` unless ``--iteration``); take the cold-chain
    median noise there (:func:`fitted_noise`) and its A/E sensitivity
    (:func:`sens_grids`); load the injected mojito L1 orbits
    (:func:`l1_orbits`, analytic fallback below); read the catalogue GBs in
    ``[--flo, --fhi]`` (:func:`catalogue_phys`); build the best-orientation
    unit-SNR kappa curve on a log grid and write it to ``--kappa-out``; run
    :func:`opt_snr` on the rows whose ``amp * kappa(f0)`` exceeds half the
    detection threshold; write ``--out`` (``f0``, ``amp``, ``snr``, ``det``,
    ``phys`` and the provenance stamps).

    Args:
        argv (list[str], optional): CLI arguments (:func:`make_parser`).
            Default ``None`` reads :data:`sys.argv`.

    Returns:
        int: ``0``.

    Raises:
        SystemExit: On a usage error or when the catalogue is not found
            (:func:`resolve_catalogue`).
        RuntimeError: When the store has no usable noise row.

    Warning:
        The ANALYTIC-orbits path (no usable mojito L1 brick, or
        ``--analytic-orbits``) builds :class:`~lisatools.detector.DefaultOrbits`
        on CPU with the ``"icrs"`` frame label over the stock equal-arm orbit
        file, whose positions are SSB ECLIPTIC (the label does not convert
        them), and feeds GBGPU the catalogue's ICRS right ascension /
        declination.
        GBGPU reads the sky angles in the frame of the orbit positions, so on
        that path every SNR (and ``det``) is computed for a sky rotated by the
        obliquity (~23.4 deg), on top of the analytic-ephemeris error the
        banner already warns about. Such a set is stamped
        ``orbits="analytic"``. The default mojito L1 path (ICRS orbit data,
        ``frame="icrs"``) is frame-consistent.
    """
    ap = make_parser()
    a = ap.parse_args(argv)

    flo, fhi = float(a.flo), float(a.fhi)
    if not (0 < flo < fhi):
        ap.error(f"need 0 < --flo < --fhi (got {flo}, {fhi})")
    tobs = float(a.tobs) if a.tobs else store_tobs(a.store)
    nw = int(a.nw) if a.nw else nw_for(tobs)
    df = 1.0 / tobs
    print(f"Tobs = {tobs:.1f} s ({tobs / 86400.0:.1f} d), df = {df:.4g} Hz, "
          f"N per waveform = {nw}")
    print(f"band = [{flo:.10g}, {fhi:.10g}] Hz "
          f"({flo * 1e3:.4g}-{fhi * 1e3:.4g} mHz)")
    # Resolve BEFORE reading, and keep the resolved int: it is stamped into
    # the npz below, and a truth set whose provenance says "None" cannot be
    # audited against the store it came from.
    it_used = (latest_iteration(a.store) if a.iteration is None
               else int(a.iteration))
    psd_p, gal_p = fitted_noise(a.store, it_used)
    print(f"noise @ it {it_used}"
          f"{' (latest)' if a.iteration is None else ' (requested)'}: "
          f"psd={psd_p} galfor={gal_p}")
    sa, se = sens_grids(psd_p, gal_p, df)

    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
        GB_MOJITO_T_REF)
    # THE INJECTED ORBITS BY DEFAULT (2026-09-23). This file used to build
    # every truth set on the analytic ephemeris, which put the monitor in a
    # mixed state: the page's own overlap/SNR block uses the mojito orbits
    # when it can reach them, while the detectable DENOMINATOR, the SNR axis
    # of the recovery curve and column 1 of the #detect table came from here.
    # The two disagree by tens of percent above a few mHz (see l1_orbits).
    orb, orb_src = (None, "--analytic-orbits requested")
    if not a.analytic_orbits:
        orb, orb_src = l1_orbits(path=a.l1_brick)
    orbits_tag = "mojito_l1"
    if orb is None:
        orbits_tag = "analytic"
        if a.analytic_orbits:
            print(f"orbits: ANALYTIC by request ({orb_src})")
        else:
            # LOUD, and it names the knobs. This used to be one print line
            # in a tens-of-minutes build, so the usual way to discover it
            # was the monitor complaining days later about a truth set
            # stamped "analytic" (user report 2026-09-26). A wrong
            # ephemeris is not a detail: at 7.44-7.60 mHz it moves optimal
            # SNR 7.96 -> 8.69 and overlap 0.052 -> 0.580.
            _tried = [n for n in ("MOJITO_INFO_PATH", "MOJITO_CAT",
                                  "MOJITO_DATA_PATH", "MOJITO_CACHE_DIR")
                      if os.environ.get(n)]
            print(
                "\n"
                "########################################################\n"
                "##  NO MOJITO L1 BRICK FOUND -- BUILDING ON THE        \n"
                "##  ANALYTIC EPHEMERIS. THIS IS PROBABLY NOT WHAT YOU  \n"
                "##  WANT.                                              \n"
                "##                                                     \n"
                f"##  reason: {orb_src}\n"
                f"##  env set: {', '.join(_tried) if _tried else 'NONE'}\n"
                "##                                                     \n"
                "##  Point one of MOJITO_INFO_PATH / MOJITO_CAT /       \n"
                "##  MOJITO_DATA_PATH / MOJITO_CACHE_DIR at a directory \n"
                "##  containing *_L1_*.h5 (searched recursively), or    \n"
                "##  pass --l1-brick /path/to/any_L1_file.h5.           \n"
                "##                                                     \n"
                "##  Above a few mHz the analytic ephemeris is simply a \n"
                "##  different number: optimal SNR 7.96 -> 8.69 and     \n"
                "##  template overlap 0.052 -> 0.580 at 7.44-7.60 mHz.  \n"
                "##  Phase maximisation does NOT rescue it. The output  \n"
                "##  is stamped orbits='analytic' and the monitor page  \n"
                "##  will say so.                                       \n"
                "##                                                     \n"
                "##  Pass --analytic-orbits to silence this and mean it.\n"
                "########################################################\n")
        # WARNING (frames): the stock DefaultOrbits file is SSB ecliptic despite
        # the "icrs" label, while catalogue_phys hands GBGPU ICRS RA/Dec, so
        # every SNR on this path is for a sky rotated by the obliquity. See the
        # Warning in this function's docstring.
        orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    else:
        print(f"orbits: mojito L1 {orb_src}")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))

    phys = catalogue_phys(GB_MOJITO_T_REF, flo, fhi, a.catalogue)
    print(f"catalogue rows in [{flo:.10g}, {fhi:.10g}] Hz: {len(phys)}")

    # ---- kappa prefilter -------------------------------------------------
    # Best-case-orientation unit-SNR curve on a coarse grid: face-on,
    # psi=phi0=0, one sky draw per node maximised over 8 sky positions.
    # Node count scales with the LOG SPAN so the grid keeps the density the
    # 48-node 3-21.94 mHz grid had (48 nodes / 0.864 decades). A wider band on
    # a fixed node count would interpolate the kappa curve across the galactic
    # knee, and the prefilter reads that curve as a ceiling -- a sagging
    # interpolant there silently drops real sources before any waveform runs.
    _NK_REF, _SPAN_REF = 48, np.log10(21.94e-3 / 3e-3)
    _nk = max(int(round(_NK_REF * np.log10(fhi / flo) / _SPAN_REF)), _NK_REF)
    fgrid = np.geomspace(flo, fhi, _nk)
    kap = np.zeros(fgrid.size)
    A0 = 1e-22
    for isky in range(8):
        rng = np.random.default_rng(isky)
        lam = rng.uniform(0, 2 * np.pi); beta = np.arcsin(rng.uniform(-1, 1))
        proto = np.column_stack([
            np.full(fgrid.size, A0), fgrid, np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.full(fgrid.size, lam), np.full(fgrid.size, beta)])
        kap = np.maximum(kap, opt_snr(proto, sa, se, gbw, df, tobs, nw,
                                      progress=False) / A0)
    keep = phys[:, 0] * np.interp(phys[:, 1], fgrid, kap) > 0.5 * SNR_DET
    print(f"kappa prefilter keeps {keep.sum()} / {len(phys)}")
    # keys are the monitor's contract: kappa_grid.npz carries fgrid/fit. The
    # curve is a unit-SNR ceiling under THIS Tobs over THIS band, so stamp
    # both -- the monitor draws the curve only over the band it was built on.
    np.savez(a.kappa_out, fgrid=fgrid, fit=kap, tobs=np.array(tobs),
             band=np.array([flo, fhi]))

    snr = np.zeros(len(phys))
    snr[keep] = opt_snr(phys[keep], sa, se, gbw, df, tobs, nw, a.batch)
    det = snr > SNR_DET
    print(f"DETECTABLE (SNR > {SNR_DET}): {det.sum()}")

    np.savez_compressed(
        a.out, f0=phys[:, 1], amp=phys[:, 0], snr=snr, det=det, phys=phys,
        store=np.array(a.store), iteration=np.array(it_used),
        psd_params=psd_p, galfor_params=gal_p,
        band=np.array([flo, fhi]), tobs=np.array(tobs), nw=np.array(nw),
        # STAMPED so the monitor can refuse to mix ephemerides silently.
        orbits=np.array(orbits_tag))
    print(f"wrote {a.out}  (orbits: {orbits_tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
