#!/usr/bin/env python
"""Two ``noise_only`` runs on the mojito L1 bricks.

    instrument   data = NOISE                 -> samples psd (Soms_d, Sa_a)
    foreground   data = NOISE + GALFOR         -> samples psd + galfor (5 params)

Both bricks are mojito L1 files on the same 731 d / 2.5 s grid (25,246,480
samples), so combining them is a straight sum. GALFOR is the confusion
foreground — the GB brick with resolvable binaries subtracted — and carries no
instrument noise of its own, so the instrument run uses NOISE alone and drops
the galfor branch (with nothing to fit it would rail at its prior floor).

The bricks are loose files, not a mojito ``data/INSTRUMENT/L1`` tree, so this
reads them by path instead of going through ``L1ProcessingStep``.

Both noise branches are sampled in the LOG, in DIFFERENT bases: psd carries
``ln(Soms_d), ln(Sa_a)`` and galfor carries ``log10(amp), log10(f_k),
log10(f_1), log10(f_2)`` (alpha, an O(1) power-law index, stays linear). Each
is a uniform-in-log prior over the same physical range plus the matching
``exp`` / ``10**x`` transform back to linear, so the likelihood is unchanged
but the proposal steps are multiplicative — these parameters span 4 to 12
decades, and a linear-uniform prior puts almost all its mass in the top one.
``--linear-psd`` / ``--linear-galfor`` restore the old basis.

By default psd and galfor are sampled JOINTLY in one move on one ladder, so
the stretch proposal moves all 7 parameters together and avoids evaluating
the full covariance twice per outer iteration. ``--split-noise`` restores
separate Metropolis-within-Gibbs moves, each with the other branch frozen at
its cold row.

A per-iteration progress bar is on by default (``--no-progress`` to suppress
it); ``--verbose`` additionally streams the run's DEBUG logs to the console.

CPU runs are serial by default and dominated by the per-walker sensitivity
rebuild (these fits use a WDM basis, where the batched C++ likelihood kernel
does not apply). The main knobs are::

    --coarse-Q N       average N adjacent WDM time columns -> nominal Nx
                       smaller covariance builds/reductions (CPU only). The
                       Welch--Satterthwaite correction is on by default;
                       --no-coarse-ws selects plain Bartlett weighting
    --wdm-psd-method layer_calibrated
                       apply the fast calibrated layer policy to instrument,
                       foreground, and SGWB. Fixed instrument bases use their
                       cached correction; sampled spectra use a 64-node
                       window quadrature instead of Nt=16060 fold samples
    --joint-noise      one move over psd+galfor (the foreground default)
                       -> ~2x fewer covariance rebuilds per iteration
    --split-noise      opt back into separate psd and galfor Gibbs moves
    --parallel-modes   with --mode both, the two fits run concurrently -> 2x
    --build-threads N  spread one batch's builds over N threads -> ~1.8x
                       (saturates at N=4; the arrays are too small for more)

They compose. Coarsening, parallel modes, and build threading are off by
default; foreground noise sampling is joint by default.

Lite laptop-smoke preset by default; ``--full`` for production sampling and
``--two-years`` for the complete brick duration::

    python scripts/noise/run_noise_only.py
    python scripts/noise/run_noise_only.py --mode foreground --full --iterations 20000
    python scripts/noise/run_noise_only.py --mode foreground --full --two-years
    mpirun -n 4 python scripts/noise/run_noise_only.py --full --gpus 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

from lisatools.globalfit.preprocessing import BaseProcessingStep

HERE = os.path.dirname(os.path.abspath(__file__))
SPRINT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NOISE_FILE = os.path.join(SPRINT_ROOT, "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
GALFOR_FILE = os.path.join(SPRINT_ROOT, "GALFOR_731d_2.5s_L1.h5")
# Tabulated galactic-foreground time modulation (t, XX, YY, ZZ, XY, XZ, YZ),
# read by lisatools.sensitivity.GalForTimeModulation. Sits next to this script.
# NOT a default -- see --modulation.
MODULATION_FILE = os.path.join(HERE, "modulation_multi.dat")
RUN_DT = 5.0
RUN_NF = 768
# Per-end edge trim, matching the engine's preprocess default
# (globalfit/engine.py). --two-years sizes its WDM grid around it.
TRIM_DURATION = 200 * 3600.0


def _read_xyz(path):
    """``(xyz (3, N), times, fs)`` from a mojito L1 file."""
    from mojito import MojitoL1File

    with MojitoL1File(path) as f:
        xyz = np.asarray(f.tdis.xyz_doppler[:]).T  # file is (N, 3)
        times = np.asarray(f.tdis.time_sampling.t())
        fs = f.tdis.time_sampling.fs
    return xyz, times, fs


def _two_year_grid(path, dt=RUN_DT, nf=RUN_NF, trim_duration=TRIM_DURATION):
    """Return the largest even-``Nt`` WDM grid the conditioned brick covers.

    The WDM transform requires even ``Nf`` and ``Nt``.  ``trim_duration`` is
    the engine's per-end edge trim (seconds), removed after downsampling and
    before the ``Tobs`` cut, so the grid has to be sized on what survives it.
    At the stock 5 s / 768 layer grid the 731-day brick keeps everything past
    the stock 45.5-day observation, losing only the two trimmed edges and the
    remainder outside the last complete WDM block.
    """
    from mojito import MojitoL1File

    with MojitoL1File(path) as f:
        n_native = int(f.tdis.time_sampling.size)
        fs_native = float(f.tdis.time_sampling.fs)

    decimation = fs_native * dt
    decimation_int = int(round(decimation))
    if decimation_int < 1 or not np.isclose(decimation, decimation_int):
        raise SystemExit(
            "--two-years requires the input cadence to be an integer divisor "
            f"of the {dt:g} s run cadence; {path!r} has fs={fs_native:g} Hz"
        )

    # scipy.signal.resample_poly, used by the preprocessing path, returns the
    # ceiling of the input length divided by an integer decimation factor.
    n_downsampled = (n_native + decimation_int - 1) // decimation_int
    # SignalProcessor.trim takes int(duration / dt) off EACH end, on the
    # already-downsampled series.
    n_available = n_downsampled - 2 * int(trim_duration / dt)
    nt = n_available // nf
    nt -= nt % 2
    if nt < 2:
        raise SystemExit(
            f"--two-years input {path!r} is too short for an even {nf}xNt WDM "
            f"grid after trimming {trim_duration / 3600:g} h from each end"
        )
    return nf, nt, n_available


def _domain_t0(noise_file, gs):
    """Absolute time of the domain's ``t_arr[0]``.

    Domains hand out a 0-based ``t_arr``, and every absolute-clock table (the
    ``/ltts`` delays, the tabulated galactic modulation) is looked up at
    ``t_arr + data_t0``.  ``data_t0`` is therefore the first sample of the
    PROCESSED series, not of the file: the preprocess trims
    ``int(duration / dt)`` samples off the head before the WDM pour, so the
    file's own ``times[0]`` is early by exactly that much.  Missing the offset
    reads every column's delays from ~200 h in the past, which is a ~1.7e-3
    relative arm error -- small, but one-signed, and worth +0.0006 in
    ``ln Soms_d`` on a two-year fit.
    """
    # Only the first sample time is needed -- do NOT go through _read_xyz,
    # which pulls the whole 3 x N Doppler array (~600 MB on the two-year
    # brick) to reach it.
    from mojito import MojitoL1File

    with MojitoL1File(noise_file) as handle:
        t_first = float(handle.tdis.time_sampling.t()[0])
    trim = (gs.preprocess_kwargs or {}).get("trim_kwargs")
    head = 0.0
    if trim and trim.get("trimming_type") == "from_each_end":
        duration = float(trim["duration"])
        if trim.get("is_percent"):
            raise SystemExit(
                "is_percent trims are not supported here; the domain t0 offset "
                "cannot be resolved before the data is loaded."
            )
        head = int(duration / gs.dt) * gs.dt
    return t_first + head


class NoiseBrickStep(BaseProcessingStep):
    """The NOISE brick, optionally with GALFOR summed onto it.

    Orbits come from the NOISE file, as in ``L1DataLoader.load_data``.
    """

    def __init__(self, noise_file, galfor_file=None, orbits_kwargs=None, verbose=True):
        from lisatools.detector import L1Orbits

        xyz, times, fs = _read_xyz(noise_file)
        if galfor_file is not None:
            fg, _, _ = _read_xyz(galfor_file)
            if fg.shape != xyz.shape:
                raise ValueError(
                    f"GALFOR {fg.shape} and NOISE {xyz.shape} are not on the same grid"
                )
            xyz = xyz + fg

        super().__init__(times, xyz, fs, verbose=verbose)

        self.orbits_class = L1Orbits  # the engine rebuilds GPU orbits through this
        self.orbits = L1Orbits(noise_file, **(orbits_kwargs or {}))
        self.orbits._ensure_configured()


def check_resume(fit, branches):
    """Fail early if an existing backend was written with a different shape.

    The engine resumes any backend it finds at ``main_file_path``. Resuming one
    written under a different ladder size or walker count fails deep in the HDF
    write as a bare shape mismatch; catch it here with a readable message.
    """
    # Mirrors GeneralSetup.main_file_path (engine.py) -- that is a property on
    # the BUILT setup, and this runs pre-build, so rebuild it from the settings
    # fields. Concatenation, not os.path.join, to match the engine exactly.
    gs = fit.general
    path = gs.file_store_dir + gs.base_file_name + "_" + gs.main_file_key + ".h5"
    scheme_marker = path + ".noise_scheme"
    requested_scheme = (
        "joint" if bool(getattr(gs, "joint_noise_move", False)) else "split"
    )
    if not os.path.exists(path):
        if "galfor" in branches:
            with open(scheme_marker, "w", encoding="utf-8") as marker:
                marker.write(requested_scheme + "\n")
        return

    import h5py

    diffs = []
    with h5py.File(path, "r") as f:
        grp = f.get("global_fit") or f.get("mcmc")
        if grp is None or int(grp.attrs.get("iteration", 0)) <= 0:
            if "galfor" in branches:
                with open(scheme_marker, "w", encoding="utf-8") as marker:
                    marker.write(requested_scheme + "\n")
            return
        if "galfor" in branches:
            if os.path.exists(scheme_marker):
                with open(scheme_marker, encoding="utf-8") as marker:
                    stored_scheme = marker.read().strip()
            else:
                # Before the joint-default change, the unmarked driver default
                # was split. Treat old files conservatively so an explicit tag
                # such as ``try5`` cannot silently resume under the new scheme.
                stored_scheme = "split"
            if stored_scheme != requested_scheme:
                diffs.append(
                    "noise move scheme: stored "
                    f"{stored_scheme}, config {requested_scheme}"
                )
        stored_nwalkers = int(grp.attrs["nwalkers"])
        if stored_nwalkers != fit.general.nwalkers:
            diffs.append(f"nwalkers: stored {stored_nwalkers}, config {fit.general.nwalkers}")
        # The WDM time grid is NOT a sampler shape -- every chain array is
        # (iteration, temp, walker, leaf, ndim) regardless of Nt -- so a run
        # whose data grid moved would resume happily against the old walkers
        # and sample a different likelihood. --two-years changes Nt whenever
        # the conditioning changes (the edge trim shortens the record), which
        # is exactly the silent case. The stored ``omega`` is one entry per
        # WDM time column, so its length is the stored Nt.
        kwargs_grp = grp.get("domain_settings/kwargs")
        if kwargs_grp is not None:
            for name, want in (
                ("min_freq", fit.general.min_freq),
                ("max_freq", fit.general.max_freq),
            ):
                stored = kwargs_grp.attrs.get(name)
                if stored is None or want is None:
                    continue
                if not np.isclose(float(stored), float(want)):
                    diffs.append(
                        f"{name}: stored {float(stored):g}, config {float(want):g}"
                    )
        omega = grp.get("domain_settings/kwargs/omega")
        if omega is not None and getattr(fit.general, "nt", None) is not None:
            stored_nt = int(omega.shape[0])
            if stored_nt != int(fit.general.nt):
                diffs.append(
                    f"WDM Nt: stored {stored_nt}, config {int(fit.general.nt)}"
                )
        for name in branches:
            sub = grp.get(f"sub_backend/{name}")
            if sub is None:
                continue
            stored_ntemps = int(sub.attrs["ntemps"])
            want = getattr(fit, name).ntemps
            if stored_ntemps != want:
                diffs.append(f"{name}.ntemps: stored {stored_ntemps}, config {want}")

    if diffs:
        raise SystemExit(
            f"\n{path}\nwas written with a different configuration:\n"
            + "".join(f"  - {d}\n" for d in diffs)
            + "\nThe engine would resume it and fail with a shape mismatch. Use a "
            "different --tag (lite and full already differ), or delete the file to "
            "start fresh.\n"
        )
    if "galfor" in branches:
        with open(scheme_marker, "w", encoding="utf-8") as marker:
            marker.write(requested_scheme + "\n")


def sampled_branches(mode):
    """Noise branches this mode fits (instrument drops galfor entirely)."""
    return ["psd"] if mode == "instrument" else ["psd", "galfor"]


def _basis_tag(mode, args):
    """``""`` / ``"_log"`` / ``"_log<branch>"`` for the backend file name.

    A ``_joint`` suffix is appended when psd+galfor share one move: the two
    schemes produce IDENTICAL backend shapes (same branches, same ntemps,
    same nwalkers) but different per-branch ladders, so nothing downstream
    would notice a joint run resuming a split one.
    """
    log_branches = [b for b in sampled_branches(mode) if not getattr(args, f"linear_{b}")]
    if not log_branches:
        tag = ""
    elif len(log_branches) == len(sampled_branches(mode)):
        tag = "_log"  # the default: everything this mode samples is logged
    else:
        tag = "_log" + "_".join(log_branches)
    if getattr(args, "joint_noise", False) and mode != "instrument":
        tag += "_joint"
    # Same shapes again: a modulated and a stationary run differ only in the
    # covariance model, so nothing downstream would catch a resume across them.
    if getattr(args, "modulation", None) and mode != "instrument":
        tag += "_mod"
    # Same shapes yet again: the unequal-arm instrument model changes only the
    # covariance, so a resume across it would silently read equal-arm walkers.
    if getattr(args, "unequal_arm", False):
        tag += "_ua"
    _wdm_method = getattr(args, "wdm_psd_method", "fold")
    if _wdm_method == "layer_constant":
        tag += "_wdmcenter"
    elif _wdm_method == "layer_calibrated":
        tag += "_wdmcal"
    coarse_q = int(getattr(args, "coarse_Q", 1) or 1)
    if coarse_q > 1:
        tag += f"_cQ{coarse_q}"
        if getattr(args, "no_coarse_ws", False):
            tag += "_bartlett"
        else:
            tag += f"_ws_{getattr(args, 'coarse_fiducial', 'injection')}"
    return tag


def build_fit(mode, args):
    """Configure (not build) the ``noise_only`` fit for one of the two runs."""
    from lisatools.globalfit.stock import erebor

    # lite and full differ in grid (nt 256 vs 1024) AND ladder size
    # (psd/galfor ntemps 2 vs 12), so they must not share a backend file.
    # A sampling basis differs in no SHAPE at all -- a log run resuming a
    # linear backend would read ln-basis walkers out of linear coords and
    # never fail -- so the basis goes in the default tag too, naming the
    # branches this mode actually samples.
    profile = ("2yr_" if args.two_years else "") + ("full" if args.full else "lite")
    tag = args.tag or (profile + _basis_tag(mode, args))
    knobs = {"file_store_dir": args.out_dir, "base_file_name": f"noise_{mode}_{tag}"}
    if not args.full:
        knobs["lite"] = True
    if args.two_years:
        nf, nt, n_available = _two_year_grid(args.noise_file)
        knobs.update(nf=nf, nt=nt)
    for key, value in (
        ("nwalkers", args.nwalkers),
        ("num_iterations", args.iterations),
        ("gpus", args.gpus),
        ("psd_build_threads", args.build_threads),
        ("min_freq", args.min_freq),
        ("max_freq", args.max_freq),
    ):
        if value is not None:
            knobs[key] = value

    # ``progress`` is the bar alone; ``verbose`` additionally streams the run's
    # DEBUG logs to stdout. They are separate settings (engine.py) because the
    # bar is what you want on a laptop smoke run and the logs are not -- before
    # 2026-08 the sampler took ``progress=verbose`` and the two were welded
    # together. Passed explicitly so the CLI beats the PROGRESS / VERBOSE env.
    knobs["progress"] = args.progress
    knobs["verbose"] = args.verbose
    # One PSDMove over ["psd", "galfor"] instead of one per branch. The
    # variant rebuilds its default recipe from this (variants/noise.py), so
    # the stage carries a single ``noise_pe`` move.
    knobs["joint_noise_move"] = args.joint_noise
    knobs["coarse_Q"] = args.coarse_Q
    knobs["coarse_use_ws"] = not args.no_coarse_ws
    knobs["coarse_fiducial"] = args.coarse_fiducial
    # None -> stationary. A path is wrapped in GalForTimeModulation by the
    # variant's _resolve_modulation and threaded onto the GalacticForeground
    # component, which evaluates it on the domain's t_arr (0-based).
    if args.modulation is not None:
        if not os.path.isfile(args.modulation):
            raise SystemExit(f"--modulation {args.modulation!r} does not exist")
        _modulation_path = args.modulation
    else:
        _modulation_path = None

    fit = erebor.noise_only(**knobs)
    gs = fit.general
    # One policy for every component entering the WDM noise likelihood.  The
    # unequal-arm instrument has a specialized fixed-basis calibration; the
    # sampled foreground/SGWB use the generic proposal-dependent quadrature.
    gs.sensitivity_init_kwargs = dict(gs.sensitivity_init_kwargs or {})
    gs.sensitivity_init_kwargs["wdm_psd_method"] = args.wdm_psd_method

    if args.two_years:
        # Keep the engine's conditioning -- highpass then 200 h off each end --
        # and take the largest complete even WDM grid inside what survives.
        # The brick is NOT pre-conditioned: it carries a low-frequency wander
        # tens of sigma above the noise rms, and the run's Tukey taper is only
        # 2 wavelets per side, so tapering an unfiltered record turns that
        # wander into a broadband transient in the edge time columns. Measured
        # on this brick (X, 768x16436): dropping both left mean w^2/C = 5.7
        # over all pixels against 1.001 in the interior, with 100% of the
        # excess in the first and last ~5 columns and 243x in layer m=3 alone
        # -- enough to pull Sa_a up ~50x in power, and to rail galfor's
        # amplitude at its prior ceiling when that branch is free to absorb it.
        # Restoring both puts it back at 1.003.
        #
        # Tobs is explicit so a future preprocessing-default change cannot
        # silently shorten it; the trim duration is passed rather than left to
        # the default so the grid and the trim can never disagree.
        tobs = nf * nt * gs.dt
        gs.preprocess_kwargs = dict(
            highpass_kwargs=dict(cutoff=2e-5, order=2, zero_phase=True),
            trim_kwargs=dict(
                duration=TRIM_DURATION,
                is_percent=False,
                trimming_type="from_each_end",
            ),
            downsample_kwargs=dict(target_fs=1.0 / gs.dt),
            Tobs=tobs,
            normalize=False,
        )
        dropped = n_available - nf * nt
        print(
            f"[two-years] WDM grid Nf={nf}, Nt={nt}, dt={gs.dt:g} s; "
            f"Tobs={tobs / 86400.0:.6f} d (highpass 2e-5 Hz, "
            f"{TRIM_DURATION / 3600:g} h trimmed from each end, {dropped} "
            "trailing downsampled samples outside the final complete WDM block)",
            flush=True,
        )

    # The active band is a likelihood-content choice, not a grid detail, so it
    # goes in the banner next to the grid whenever it is not the default.
    if args.min_freq is not None or args.max_freq is not None:
        layer_df = 1.0 / (2.0 * gs.nf * gs.dt)
        ind_min = int(np.ceil(gs.min_freq / layer_df))
        ind_max = min(int(gs.max_freq / layer_df), gs.nf - 1)
        print(
            f"[band] active WDM layers {ind_min}-{ind_max} "
            f"({ind_min * layer_df * 1e3:.3f}-{ind_max * layer_df * 1e3:.3f} mHz, "
            f"{ind_max - ind_min + 1} of {gs.nf} layers; "
            f"layer_df={layer_df * 1e3:.4f} mHz) "
            f"from min_freq={gs.min_freq:g}, max_freq={gs.max_freq:g}",
            flush=True,
        )

    # Sample ln(Soms_d), ln(Sa_a) and log10 of the galfor scales. The
    # prepare_*_branch helpers (stock/erebor/noise.py) turn each flag into the
    # uniform-in-log prior, the exp / 10**x TransformContainer every consumer
    # applies, and (psd) the log-basis injection the truth overlays read. The
    # CLI flags beat PSD_LOG_SAMPLING / GALFOR_LOG_SAMPLING.
    fit.psd.log_sampling = not args.linear_psd
    fit.galfor.log_sampling = not args.linear_galfor

    if args.unequal_arm:
        # Swap the stock equal-arm instrument covariance (one constant L_SI for
        # all six links) for the orbit-informed one, which carries every link
        # delay independently. The arms differ by ~1% and the Sagnac splitting
        # makes d_ij != d_ji, both of which move the TDI transfer-function nulls
        # and give the cross-spectra an imaginary part -- structure the
        # equal-arm model cannot represent, so it lands in the residual.
        #
        # Read the delays straight out of the brick's /ltts group -- the ones
        # the data was generated with. (Going through L1Orbits gives the same
        # numbers to ~1e-11, but configures an orbits object we do not
        # otherwise need here.) The stride decimates a 2.5 s cadence that is
        # wildly finer than the month timescale the delays actually vary on.
        #
        # The exact method averages the table over every WDM time slice; the
        # layer_constant approximation interpolates it at coarse cell centers.
        # Either retains the ~1.5-1.8% breathing that one epoch cannot represent.
        # ``data_t0`` lines the file's absolute clock up with the domain's
        # 0-based t_arr -- via _domain_t0, so the head trim is included. The
        # --modulation path takes the same offset.
        #
        # Only plain arrays go onto the settings tree, never an orbits object:
        # the tree is deepcopied and pickled, and L1Orbits holds a C++/nanobind
        # wrap (sprint deepcopy/pickle rule).
        from lisatools.sensitivity import LinkDelayTable, UnequalArmInstrumentNoise

        table = LinkDelayTable.from_l1_file(
            args.noise_file, stride=200, data_t0=_domain_t0(args.noise_file, gs)
        )
        fit.psd.instrument_component_cls = UnequalArmInstrumentNoise
        fit.psd.instrument_component_kwargs = dict(
            ltts=table,
            # Shared by instrument and foreground modes.  The cache contains
            # exact coarse unit bases plus the three fine diagonals needed for
            # arbitrary channelwise WS fiducials; it is safe to reuse across
            # restarts and avoids rebuilding the breathing-arm transfer.
            coarse_cache_dir=os.path.join(args.out_dir, "unequal_arm_coarse_cache"),
            coarse_progress=True,
            wdm_psd_method=args.wdm_psd_method,
        )
        mean = table.run_average()
        span = table.ltts.max(axis=0) - table.ltts.min(axis=0)
        print(
            "[unequal-arm] link LTTs (s) from /ltts, order [12, 23, 31, 13, 32, 21]"
            f"\n  run mean : {np.array2string(mean, precision=9)}"
            f"\n  arm spread (of the means): {(mean.max() - mean.min()) / mean.mean():.3%}"
            f"\n  breathing over run, per link: "
            f"{np.array2string(100 * span / mean, precision=2)} %"
            "\n  WDM PSD method: "
            + (
                "exact DFT fold and fine-column covariance average"
                if args.wdm_psd_method == "fold"
                else "0.5*S(f_m) at WDM frequency/coarse-time cell centers"
            ),
            flush=True,
        )

    # Stay on the mojito data path without an L1 folder tree: resolve_data_source
    # only checks that mojito_data_path is a real directory, and the explicit
    # noise_file wins in resolve_noise_file (it seeds psd_injection off the
    # brick's tabulated noise_estimates).
    gs.data_mode = "mojito"
    gs.mojito_data_path = os.path.dirname(args.noise_file) or "."
    gs.noise_file = args.noise_file

    gs.data_processor_class = NoiseBrickStep
    gs.processor_init_kwargs = dict(
        noise_file=args.noise_file,
        galfor_file=args.galfor_file if mode == "foreground" else None,
        orbits_kwargs=dict(
            force_backend=gs.gpu_backend if gs.gpus is not None else "cpu",
            frame=gs.orbits_frame,
        ),
        verbose=gs.verbose,
    )

    if _modulation_path is not None:
        # Set the BRANCH-level modulation rather than
        # general.galfor_modulation_path: the branch override wins in
        # noise_sensitivity_init_kwargs, and only this route lets us pass the
        # epoch. The tables are tabulated on the ABSOLUTE mission clock while
        # the domain hands out a 0-based t_arr, so t0 = the brick's first
        # sample time is what lines them up. Reading it here (rather than
        # trusting the table to start at the data) keeps the two files as the
        # single source of their own time axes.
        from lisatools.sensitivity import GalForTimeModulation

        fit.galfor.modulation = GalForTimeModulation(
            _modulation_path, t0=_domain_t0(args.noise_file, gs)
        )

    if mode == "instrument":
        # One call drops the branch AND the moves declared against it: since
        # LAT bd02f94 remove_branch pops every Move(..., branch="galfor"), so
        # the split-noise galfor_pe goes with it. Popping it again here raises
        # KeyError.
        #
        # In joint mode there is nothing extra to drop either way -- the single
        # noise_pe move is declared on the psd branch and survives, degrading to
        # psd-only because setup_recipe filters each group to the branches the
        # run actually has.
        fit.remove_branch("galfor")

    return fit


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("instrument", "foreground", "both"), default="both")
    p.add_argument("--noise-file", default=NOISE_FILE)
    p.add_argument("--galfor-file", default=GALFOR_FILE)
    p.add_argument("--full", action="store_true", help="drop the lite laptop-smoke preset")
    p.add_argument(
        "--min-freq",
        type=float,
        default=None,
        help="lower edge of the active WDM band in Hz (default 3e-4). The "
        "band is CONTIGUOUS -- WDMSettings resolves it to layers "
        "[ceil(min_freq/layer_df), floor(max_freq/layer_df)] and there is no "
        "way to punch a hole in the middle. Cropping trades likelihood "
        "leverage for freedom from a region the model cannot describe",
    )
    p.add_argument(
        "--max-freq",
        type=float,
        default=None,
        help="upper edge of the active WDM band in Hz (default 8e-3). NOTE "
        "the 4-8 mHz layers are what anchor Soms_d -- the foreground is "
        "negligible there -- so lowering this loosens Soms_d markedly",
    )
    p.add_argument(
        "--two-years",
        action="store_true",
        help="process the complete 731-day (~2 year) input brick instead of "
        "the lite/full preset's short WDM time grid. Keeps the engine's "
        "highpass and 200-hour edge trim (the brick is NOT pre-conditioned -- "
        "without them the Tukey taper turns its low-frequency wander into a "
        "broadband edge transient that biases the noise posterior), then uses "
        "the largest complete even 768xNt grid inside what survives (713.8 "
        "days for the bundled brick). Sampling scale remains independent: "
        "combine with --full for production walkers/temperatures/iterations",
    )
    p.add_argument(
        "--linear-psd",
        action="store_true",
        help="sample the instrument levels linearly instead of in ln "
        "(the default is ln(Soms_d), ln(Sa_a))",
    )
    p.add_argument(
        "--linear-galfor",
        action="store_true",
        help="sample the foreground amplitude and frequency scales linearly "
        "instead of in log10 (the default is log10(amp), log10(f_k), "
        "log10(f_1), log10(f_2); alpha is linear either way)",
    )
    p.add_argument(
        "--modulation",
        nargs="?",
        const=MODULATION_FILE,
        default=None,
        metavar="PATH",
        help="tabulated galactic-foreground time modulation (columns "
        "t, XX, YY, ZZ, XY, XZ, YZ; t is seconds from the START of the data, "
        "matching the domain's 0-based t_arr). Bare --modulation uses "
        f"{os.path.basename(MODULATION_FILE)} next to this script. Default: "
        "omitted, i.e. the stationary isotropic limit (diag 1, off-diag -1/2). "
        "VERIFY THE PHASE against the brick before trusting a table -- an "
        "antiphased one is worse than stationary",
    )
    p.add_argument(
        "--unequal-arm",
        action="store_true",
        help="model the instrument covariance with the six per-link light "
        "travel times tabulated in the NOISE brick, instead of the stock "
        "equal-arm model's single constant L_SI. Captures the arm-to-arm "
        "spread, the ~1.5-1.8%% breathing over the run, and the Sagnac "
        "splitting d_ij != d_ji -- which shift the TDI nulls per arm and make "
        "the cross-spectra complex. The delays are averaged over each WDM time "
        "slice, so the noise PSD varies along the run. The exact fused build "
        "streams progress and is cached persistently under <out-dir>/"
        "unequal_arm_coarse_cache for restarts and the other mode; the "
        "per-proposal sampling cost is unchanged",
    )
    p.add_argument(
        "--wdm-psd-method",
        choices=("fold", "layer_constant", "layer_calibrated"),
        default="fold",
        help="WDM PSD construction for every noise component (instrument, "
        "galactic foreground, and SGWB). 'fold' (default) evaluates the "
        "DFT-grid PSD and folds it exactly. "
        "'layer_constant' uses the S_wdm[m]=0.5*S(f_m) approximation at WDM "
        "frequency-layer centers and, with --coarse-Q > 1, evaluates breathing "
        "arms at coarse time-cell centers -- it skips the expensive "
        "initialization fold but under-predicts the covariance by up to ~1%% in "
        "the lowest active layer, worth ~+0.0015 in ln Soms_d on a two-year "
        "instrument fit. 'layer_calibrated' keeps the specialized one-fold "
        "correction for the fixed instrument bases and uses a compact "
        "window-weighted quadrature for sampled stochastic spectra, retaining "
        "proposal-dependent curvature while keeping layer-center speed",
    )
    noise_move = p.add_mutually_exclusive_group()
    noise_move.add_argument(
        "--joint-noise",
        dest="joint_noise",
        action="store_true",
        default=True,
        help="sample psd and galfor in ONE move on ONE ladder (the default) -- "
        "proposes all 7 parameters together and evaluates the full covariance "
        "once per repeat. Requires PSD_NTEMPS == GALFOR_NTEMPS (both default "
        "to 12)",
    )
    noise_move.add_argument(
        "--split-noise",
        dest="joint_noise",
        action="store_false",
        help="use separate psd and galfor Metropolis-within-Gibbs moves, each "
        "with its own ladder and the other branch frozen at its cold row",
    )
    p.add_argument(
        "--coarse-Q",
        type=int,
        default=1,
        metavar="N",
        help="coarsen the real WDM likelihood over N adjacent time columns "
        "(default 1, disabled). N>1 is CPU-only for now and gives a nominal "
        "Nx reduction in covariance construction/inversion work. Unequal-arm "
        "covariances use an exact fine-grid cell average under "
        "--wdm-psd-method=fold, or a coarse cell-center evaluation under "
        "layer_constant/layer_calibrated",
    )
    p.add_argument(
        "--coarse-fiducial",
        choices=("injection", "initial"),
        default="injection",
        help="when WS is enabled, the fine-grid covariance at which the coarse "
        "effective degrees of freedom are frozen: injected/reference parameters "
        "(default) or the componentwise median physical parameters of the "
        "initial cold walkers",
    )
    p.add_argument(
        "--no-coarse-ws",
        action="store_true",
        help="opt out of the default channelwise Welch--Satterthwaite "
        "effective-dof correction and use plain Bartlett weights Qeff=n_q",
    )
    p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="suppress the per-iteration progress bar (on by default; turn it "
        "off when stdout is a log file)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="stream the run's DEBUG logs to the console as well (the progress "
        "bar alone does not need this)",
    )
    p.add_argument(
        "--build-threads",
        type=int,
        metavar="N",
        help="CPU threads for the per-walker sensitivity builds inside the "
        "noise move (default 1, i.e. serial). These fits run on a WDM basis, "
        "where the batched C++ likelihood kernel is unavailable and every "
        "proposal row rebuilds its covariance in Python -- the run's dominant "
        "cost. Scaling saturates at ~4 (3.11 ms/walker serial -> 1.73 at 4 -> "
        "1.78 at 6 on the stock 59x1024 active grid): the arrays are small, "
        "so Python orchestration, not the kernels, sets the floor. Output is "
        "bitwise-identical to the serial path. Ignored on GPU runs",
    )
    p.add_argument(
        "--parallel-modes",
        action="store_true",
        help="with --mode both, run the instrument and foreground fits as two "
        "concurrent subprocesses instead of one after the other. They share "
        "nothing but the input bricks and write separate backends, so this is "
        "a straight 2x on wall clock. Each child's output goes to "
        "<out-dir>/run_<mode>.log (two live progress bars on one terminal are "
        "unreadable). Ignored unless --mode both",
    )
    p.add_argument("--nwalkers", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--gpus", type=int, nargs="+", help="GPU device ids (omit for CPU)")
    p.add_argument("--out-dir", default="./gf_output_noise/")
    p.add_argument(
        "--tag",
        help="backend file tag (default: lite/full or 2yr_lite/2yr_full, plus "
        "the sampling basis, e.g. 2yr_full_log). Change it when you change "
        "nwalkers/ntemps/grid so the run does not resume an incompatible file.",
    )
    return p.parse_args(argv)


def _argv_for_mode(argv, mode):
    """``argv`` with any --mode / --parallel-modes stripped and ``mode`` set."""
    out = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok == "--mode":
            skip_next = True  # drop its value too
            continue
        if tok.startswith("--mode="):
            continue
        if tok == "--parallel-modes":
            continue
        out.append(tok)
    return out + ["--mode", mode]


def run_modes_in_parallel(args, argv):
    """Run instrument + foreground as concurrent subprocesses.

    The two fits share nothing but the input bricks — different branches,
    different backends (``base_file_name`` carries the mode), different
    artifact dirs — so there is nothing to coordinate. Output is redirected
    per mode because two live progress bars on one terminal interleave into
    noise.
    """
    procs = []
    logs = []
    for mode in ("instrument", "foreground"):
        log_path = os.path.join(args.out_dir, f"run_{mode}.log")
        log = open(log_path, "w")
        logs.append(log_path)
        cmd = [sys.executable, os.path.abspath(__file__), *_argv_for_mode(argv, mode)]
        print(f"[{mode}] -> {log_path}", flush=True)
        procs.append(
            (mode, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        )

    # The progress bars are still there, they are just in the logs now. Say so
    # and hand over the command -- otherwise this looks exactly like a hang,
    # and under --unequal-arm there is a one-time exact transfer build before
    # iteration 1 (now with its own progress/ETA output).
    print(
        "\nBoth runs are live; their progress bars are in the logs above.\n"
        f"  watch both:  tail -f {' '.join(logs)}\n"
        "Under --unequal-arm, watch the one-time breathing-arm build progress "
        "before the first iteration; the other mode/restarts reuse its cache.\n"
        "Drop --parallel-modes to get the bar on your terminal instead.",
        flush=True,
    )

    failed = []
    for mode, log, proc in procs:
        rc = proc.wait()
        log.close()
        status = "done" if rc == 0 else f"FAILED (exit {rc})"
        print(f"[{mode}] {status} — output under {args.out_dir}", flush=True)
        if rc != 0:
            failed.append(mode)

    if failed:
        raise SystemExit(
            f"{', '.join(failed)} failed; see <out-dir>/run_<mode>.log for the traceback"
        )


def main(argv=None):
    args = parse_args(argv)
    if args.coarse_Q < 1:
        raise SystemExit("--coarse-Q must be an integer >= 1")
    if args.coarse_Q > 1 and args.gpus is not None:
        raise SystemExit(
            "--coarse-Q > 1 is CPU-only for now; omit --gpus. "
            "Single-GPU support is planned."
        )
    # The engine builds paths by string concatenation
    # (``file_store_dir + base_file_name + ...``), so a missing trailing
    # separator silently writes "<dir>noise_instrument_..." NEXT TO the
    # directory instead of inside it.
    args.out_dir = os.path.join(args.out_dir, "")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "both" and args.parallel_modes:
        # Re-invoke this script once per mode. Done here, before any of the
        # heavy imports/build, so each child owns its own process from the
        # start (the two fits build independent data + backends anyway).
        run_modes_in_parallel(args, list(sys.argv[1:] if argv is None else argv))
        return

    for mode in ["instrument", "foreground"] if args.mode == "both" else [args.mode]:
        print(f"\n{f' {mode} ':=^72}\n", flush=True)
        fit = build_fit(mode, args)
        check_resume(fit, sampled_branches(mode))
        fit.run()
        print(f"[{mode}] done — output under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
