"""Per-source GB mismatch check: STFT (Fresnel) and FD paths vs the legacy GBGPU
FD waveform and vs the mojito L1 data.

For a set of catalogue GBs this script reports, per source:

  FD basis (rfft grid, band-sliced complex overlap):
    mm(legacy | data)     -- ``gbgpu.gbgpu.GBGPU.run_wave`` (the classic FD
                             generator) against the band-passed mojito data,
                             both under the same whole-span Tukey taper. This
                             is the gb_mojito_match.py convention (~1e-8 for
                             isolated sources when everything is consistent).
    mm(fdcomp | legacy)   -- ``GBFDComputations`` (the gb_fd heterodyne
                             kernels, the fit's FD path) against the legacy
                             waveform, both RECTANGULAR (no window modelling
                             enters this comparison).
    mm(fdcomp | data)     -- same fdcomp template against the rectangular
                             data rfft (shares the rect edge-leakage).

  STFT basis (per-segment window, column-sliced complex overlap):
    mm(stft | legacy-stft)-- ``STFTGBComputations`` Fresnel template against
                             the brute STFT of the legacy waveform (the exact
                             TF representation of the trusted generator).
    mm(stft | data-stft)  -- against the brute STFT of the raw data stream.
    ...in-stencil twins   -- the same overlaps restricted to the template's
                             populated pixels: removes the +/- n_side_bins
                             band-truncation leakage (the dominant term of
                             full-grid STFT mismatches, see the 2026-07
                             agreement studies).
    ...stencil-interior   -- additionally drops the first/last STFT segments
                             (observation/orbit-edge artifacts; for the
                             synthetic selftest also the irfft wrap of the
                             reference), isolating the per-column method
                             error + waveform-family difference.

Catalogue handling mirrors the global fit: ``L1ProcessingStep`` loads the
galaxy + catalogue, ``recipe.gb_catalogue_to_sampling_basis`` (params AT the
mojito reference epoch, no trim evolution) feeds the stock GB transform
(phi0 sign flip included), and every generator/comp receives
``t_ref = MOJITO_REFERENCE_TIME``. The analysis window is anchored at the
raw data start, which for mojito L1 coincides with that epoch (asserted).

Mismatches are UNWEIGHTED complex overlaps, 1 - Re(O) and 1 - |O| with
O = sum(conj(a) * b) / (||a|| ||b||), summed over channels and the slice.
Over the few-uHz / few-column slices used here a PSD weighting is constant
to <0.1% and changes nothing.

Usage (box with the mojito cache):

    python gb_mojito_stft_fd_mismatch.py --mojito-path \\
        /path/to/mojito_light_v1_0_0/ --n-days 90 --topn 6

Local plumbing check without any data (synthetic 2-source catalogue, data =
sum of legacy waveforms; legacy-vs-data must be ~exact):

    python gb_mojito_stft_fd_mismatch.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

import gbgpu  # noqa: F401  -- registers the gbgpu backends
import lisatools
from gbgpu.gbcomps import GBFDComputations, STFTGBComputations
from gbgpu.gbgpu import GBGPU
from eryn.utils import TransformContainer

from lisatools.detector import EqualArmlengthOrbits, L1Orbits
from lisatools.domains import FDSettings, TDSettings, TDSignal, get_stft_settings
from lisatools.globalfit.recipe import (
    MOJITO_REFERENCE_TIME,
    gb_catalogue_to_sampling_basis,
)
from lisatools.response.tdiconfig import TDIConfig

TDI_GEN = "2nd generation"
NCH = 3


# --------------------------------------------------------------------------
# catalogue -> physical params (the global fit's exact chain)
# --------------------------------------------------------------------------

def _f_ms_to_s(x):
    return x * 1e-3


def _neg(x):
    return -1.0 * x


def build_gb_transform() -> TransformContainer:
    """The run transform (GBSetup variant, stock/erebor.py): includes the
    phi0 sign flip that undoes the -TrueAnomaly stored by
    ``gb_catalogue_to_sampling_basis`` (physical phi0 = +TrueAnomaly)."""
    return TransformContainer(
        input_basis=["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"],
        output_basis=["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi",
                      "alpha", "sin_delta"],
        parameter_transforms={
            "A": np.exp,
            "f0": _f_ms_to_s,
            "phi0": _neg,
            "cos_iota": np.arccos,
            "sin_delta": np.arcsin,
        },
        fill_dict={"fddot": 0.0},
    )


def catalogue_to_physical(cat_entry: dict) -> np.ndarray:
    """(N_src, 9) physical params [amp, f0, fdot, fddot, phi0, iota, psi, ra, dec]."""
    sampling = np.atleast_2d(gb_catalogue_to_sampling_basis(cat_entry))
    return np.asarray(build_gb_transform().both_transforms(sampling))


# --------------------------------------------------------------------------
# data sources
# --------------------------------------------------------------------------

def load_mojito(args):
    """Load the mojito L1 GB galaxy + catalogue the way the global fit does."""
    from lisatools.globalfit.preprocessing import L1ProcessingStep

    loader = L1ProcessingStep(
        L1_folder=args.mojito_path,
        source_types=["gb"],
        source_ids=None,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=args.backend, frame="icrs"),
        verbose=True,
    )
    times = np.asarray(loader.times)
    data = np.asarray(loader.data)
    if data.shape[0] != NCH:
        data = data.T
    dt_native = float(loader.dt)
    deci = int(round(args.dt / dt_native))
    assert abs(deci * dt_native - args.dt) < 1e-12, (
        f"--dt {args.dt} must be an integer multiple of the native dt {dt_native}"
    )
    n_win = int(round(args.n_days * 86400.0 / args.dt))
    data_td = np.ascontiguousarray(data[:NCH, : n_win * deci : deci][:, :n_win],
                                   dtype=np.float64)
    t0_data = float(times[0])
    cat0 = loader.catalogue["GB"][0]
    orbits = loader.orbits if getattr(loader, "orbits", None) is not None else L1Orbits(
        force_backend=args.backend, frame="icrs"
    )
    if not orbits.configured:
        orbits.configure(linear_interp_setup=True)
    return data_td, t0_data, cat0, orbits


def make_selftest(args):
    """Synthetic 2-source 'catalogue' + data = sum of legacy GBGPU waveforms.

    Anchored 10 d into the (0-based) equal-armlength orbit so no leg sits on
    the orbit-file edge; the catalogue conversion itself is epoch-free, so the
    mojito reference only matters through t_ref = window start (delta = 0
    here by construction, same as the mojito convention).
    """
    t0_data = 10.0 * 86400.0
    # Whole-span taper modelling (run_wave's slow-part window) is approximate;
    # the selftest asserts EXACT legacy-vs-data agreement, so keep every leg
    # rectangular here.
    if args.data_tukey_alpha != 0.0:
        print("[selftest] forcing --data-tukey-alpha 0 (exactness check)")
        args.data_tukey_alpha = 0.0
    n_win = int(round(args.n_days * 86400.0 / args.dt))
    dfq = 1.0 / (args.stft_big_dt)  # separate sources by many STFT columns
    f0s = np.array([4.001e-3, 4.001e-3 + 20 * dfq])
    cat0 = dict(
        Amplitude=np.array([2e-22, 1e-22]),
        GW22FrequencySSBFrame=f0s,
        GW22FrequencyDerivativeSourceFrame=np.array([1e-17, 2e-17]),
        TrueAnomaly=np.array([1.1, 4.4]),
        InclinationAngle=np.array([0.7, 2.1]),
        PolarisationAngle=np.array([0.4, 2.6]),
        RightAscension=np.array([2.0, 5.1]),
        Declination=np.array([0.3, -0.7]),
    )
    orbits = EqualArmlengthOrbits(force_backend=args.backend)
    orbits.configure(linear_interp_setup=True)
    # EqualArmlength orbits are 0-based; shift the observation into the
    # interior by faking sc_t0 through the GBGPU t0 argument only. The
    # absolute epoch only enters through orbit evaluation, which is
    # periodic-ish and long enough here.
    params = catalogue_to_physical(cat0)
    gb = GBGPU(orbits=orbits, force_backend=args.backend, t0=t0_data)
    n_rfft = n_win // 2 + 1
    data_fd = np.zeros((NCH, n_rfft), dtype=np.complex128)
    for p in params:
        data_fd += legacy_fd_template(gb, p, n_win, args.dt, oversample=args.oversample)
    data_td = np.fft.irfft(data_fd / args.dt, n=n_win, axis=-1)
    return np.ascontiguousarray(data_td), t0_data, cat0, orbits


# --------------------------------------------------------------------------
# template legs
# --------------------------------------------------------------------------

def legacy_fd_template(gb: GBGPU, p9: np.ndarray, n_win: int, dt: float,
                       oversample: int = 2, window: str | None = None,
                       window_alpha: float = 0.0) -> np.ndarray:
    """Classic ``GBGPU.run_wave`` -> (NCH, n_rfft) on the full rfft grid.

    Conventions match the FD data ``rfft(x * win) * dt``.
    """
    gb.run_wave(
        *[np.atleast_1d(v) for v in p9],
        N=None, T=n_win * dt, dt=dt, oversample=oversample,
        tdi2=True, tdi_channel_setup="XYZ",
        window=window, window_alpha=window_alpha,
    )
    xyz = np.asarray(gb.XYZf)[0]              # (3, N_band)
    k0 = int(np.asarray(gb.start_inds)[0])
    n_rfft = n_win // 2 + 1
    out = np.zeros((NCH, n_rfft), dtype=np.complex128)
    lo, hi = max(k0, 0), min(k0 + xyz.shape[-1], n_rfft)
    out[:, lo:hi] = xyz[:, lo - k0 : hi - k0]
    return out


def make_stft_shim(settings, window_alpha: float, backend_name: str):
    """Minimal ``stft_comps`` stand-in for STFTGBComputations.fill_global_stft
    (the STFTEngineAccuracy pattern): only cpp_fresnel / cpp_domain are read
    by the fill kernel; the zero buffers must stay alive (dangling-ptr rule).
    """
    backend = lisatools.get_backend(backend_name)
    tdi_type = backend.TDITypeDict["XYZ"]
    NT, NF = settings.NT, settings.NF_active
    zdata = np.zeros((1, NCH, NT, NF), np.complex128)
    zinvC = np.zeros((1, NCH, NCH, NT, NF), np.complex128)
    s = settings
    domain = backend.STFTDomainWrap(NT, NF, NCH, s.t0, s.min_freq, s.max_freq,
                                    s.dt, s.df, zdata.reshape(-1),
                                    zinvC.reshape(-1), 1, 1, tdi_type)
    fres = backend.STFTFresnelWrap(NT, NF, NCH, s.t0, s.min_freq, s.max_freq,
                                   s.dt, s.df, window_alpha=window_alpha,
                                   use_midpoint=False)
    shim = type("_STFTShim", (), {})()
    shim.cpp_fresnel, shim.cpp_domain, shim.d_d = fres, domain, None
    shim._keepalive = (zdata, zinvC)
    return shim


def brute_stft(td_arr: np.ndarray, t0: float, dt: float, settings,
               seg_window: np.ndarray) -> np.ndarray:
    """Brute STFT (NCH, NT, NF_active) of a TD stream via the canonical
    ``TDSignal.stft`` (the reference construction of the STFT test suite)."""
    nper = seg_window.shape[0]
    used = settings.NT * nper
    td = TDSignal(np.ascontiguousarray(td_arr[:, :used]),
                  settings=TDSettings(used, dt, t0, force_backend="cpu"))
    return np.asarray(td.stft(window=seg_window, settings=settings).arr)


# --------------------------------------------------------------------------
# mismatches
# --------------------------------------------------------------------------

def overlap_mm(a: np.ndarray, b: np.ndarray) -> tuple:
    """(1 - Re O, 1 - |O|) with O the normalized complex overlap over all axes."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan"), float("nan")
    o = np.sum(np.conj(a) * b) / (na * nb)
    return float(1.0 - o.real), float(1.0 - abs(o))


def stft_mms(ref: np.ndarray, model: np.ndarray, c_lo: int, c_hi: int) -> dict:
    """Column-sliced STFT mismatches, decomposed the way the agreement
    studies established: ``full`` (everything), ``stencil`` (template's
    populated pixels only -- removes the +/- n_side band-truncation leakage),
    and ``stencil_interior`` (additionally drops the first/last segments,
    which carry the orbit/observation-edge artifacts) = the per-column
    method error proper."""
    R, M = ref[:, :, c_lo:c_hi], model[:, :, c_lo:c_hi]
    full = overlap_mm(R, M)
    mask = M != 0.0
    Rs = np.where(mask, R, 0.0)
    stencil = overlap_mm(Rs, M)
    interior = overlap_mm(Rs[:, 1:-1, :], M[:, 1:-1, :])
    return dict(full_re=full[0], full_abs=full[1],
                stencil_re=stencil[0], stencil_abs=stencil[1],
                stencil_interior_re=interior[0], stencil_interior_abs=interior[1],
                stencil_frac=float(np.abs(Rs).sum() / max(np.abs(R).sum(), 1e-300)))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mojito-path", default=None,
                   help="mojito_light folder (the global fit's MOJITO_DATA_PATH)")
    p.add_argument("--selftest", action="store_true",
                   help="no data needed: synthetic catalogue, data = legacy waveforms")
    p.add_argument("--n-days", type=float, default=90.0)
    p.add_argument("--dt", type=float, default=10.0,
                   help="analysis sample step (stride-decimated from native)")
    p.add_argument("--topn", type=int, default=6)
    p.add_argument("--f0-min", type=float, default=None,
                   help="Hz; with --f0-max, restrict candidates before ranking")
    p.add_argument("--f0-max", type=float, default=None)
    p.add_argument("--rank", choices=["hif", "amp"], default="hif",
                   help="rank candidates by frequency (isolation) or amplitude")
    p.add_argument("--band-uhz", type=float, default=5.0,
                   help="FD overlap half-window around f0 in microhertz")
    p.add_argument("--data-tukey-alpha", type=float, default=0.05,
                   help="whole-span taper for the FD legacy-vs-data leg")
    p.add_argument("--stft-big-dt", type=float, default=21600.0)
    p.add_argument("--stft-n-side", type=int, default=10)
    p.add_argument("--stft-window-alpha", type=float, default=0.0,
                   help="per-segment Tukey alpha (data STFT + Fresnel evaluator)")
    p.add_argument("--freq-from-tdi-phase", type=int, default=1)
    p.add_argument("--oversample", type=int, default=2)
    p.add_argument("--n-sparse", type=int, default=256)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if not args.selftest and not args.mojito_path:
        p.error("provide --mojito-path (or run --selftest)")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.selftest:
        data_td, t0_data, cat0, orbits = make_selftest(args)
    else:
        data_td, t0_data, cat0, orbits = load_mojito(args)

    if not args.selftest:
        delta_ref = t0_data - MOJITO_REFERENCE_TIME
        print(f"window t0 = {t0_data:.6f} s   t0 - REF = {delta_ref:+.3f} s")
        assert abs(delta_ref) < args.dt, (
            f"data start is {delta_ref:+.1f} s away from MOJITO_REFERENCE_TIME; "
            "the catalogue anchoring below assumes they coincide (mojito L1 "
            "convention). Evolve the params or trim differently before "
            "trusting the numbers."
        )
    t_ref = t0_data

    # ---- source selection (isolation gap vs the FULL catalogue) ----
    f_all = np.asarray(cat0["GW22FrequencySSBFrame"], dtype=float)
    a_all = np.asarray(cat0["Amplitude"], dtype=float)
    sel = np.ones(f_all.shape[0], dtype=bool)
    if args.f0_min is not None:
        sel &= f_all >= args.f0_min
    if args.f0_max is not None:
        sel &= f_all <= args.f0_max
    cand = np.where(sel)[0]
    order = np.argsort(f_all[cand] if args.rank == "hif" else a_all[cand])[::-1]
    picks = cand[order[: args.topn]]
    f_sorted = np.sort(f_all)

    params_all = catalogue_to_physical(cat0)

    # ---- FD grids ----
    n_win = data_td.shape[-1]
    T_win = n_win * args.dt
    df = 1.0 / T_win
    n_rfft = n_win // 2 + 1
    alpha = args.data_tukey_alpha
    if alpha > 0.0:
        from scipy.signal.windows import tukey
        win = tukey(n_win, alpha)
    else:
        win = np.ones(n_win)
    data_fd_w = np.fft.rfft(data_td * win[None, :], axis=-1) * args.dt
    data_fd_r = np.fft.rfft(data_td, axis=-1) * args.dt

    # Band margins: cover the FD heterodyne support (N_sparse bins) AND the
    # STFT stencil (n_side columns) so no per-source template clips the
    # active-band edge.
    df_stft_est = 1.0 / args.stft_big_dt
    margin = max(2 * args.n_sparse * df, (args.stft_n_side + 4) * df_stft_est)
    f_lo = float(f_all[picks].min()) - margin
    f_hi = float(f_all[picks].max()) + margin
    fd_settings = FDSettings(N=n_rfft, df=df, min_freq=f_lo, max_freq=f_hi,
                             force_backend=args.backend)
    tdi_config = TDIConfig(TDI_GEN, force_backend=args.backend)
    fd_comp = GBFDComputations(
        fd_settings, t_ref, N_sparse=args.n_sparse, orbits=orbits,
        tdi_config=tdi_config, force_backend=args.backend,
        tdi_type="XYZ", nchannels=NCH, tukey_alpha=0.0,
    )
    gb_legacy = GBGPU(orbits=orbits, force_backend=args.backend, t0=t0_data)

    # ---- STFT grid (whole selected band; per-source column slices later) ----
    nper = int(round(args.stft_big_dt / args.dt))
    n_stft = n_win // nper
    used = n_stft * nper
    stft_settings = get_stft_settings(
        t0_data + np.arange(used) * args.dt, args.stft_big_dt,
        min_freq=f_lo, max_freq=f_hi, force_backend=args.backend,
    )
    df_stft = float(stft_settings.df)
    if args.stft_window_alpha > 0.0:
        from scipy.signal.windows import tukey
        seg_win = tukey(nper, args.stft_window_alpha)
    else:
        seg_win = np.ones(nper)
    data_stft = brute_stft(data_td, t0_data, args.dt, stft_settings, seg_win)
    shim = make_stft_shim(stft_settings, args.stft_window_alpha, args.backend)
    stft_comp = STFTGBComputations(
        stft_comps=shim, T=n_stft * args.stft_big_dt, t_ref=t_ref,
        orbits=orbits, tdi_config=tdi_config, force_backend=args.backend,
        n_side_bins=args.stft_n_side, window_factor=1.0,
        freq_from_tdi_phase=bool(args.freq_from_tdi_phase),
    )

    print(f"FD grid: N={n_win} df={df:.4e}  band=[{f_lo:.6e},{f_hi:.6e}] Hz  "
          f"IP window +-{args.band_uhz} uHz  data taper alpha={alpha}")
    print(f"STFT grid: NT={stft_settings.NT} NF_active={stft_settings.NF_active} "
          f"big_dt={args.stft_big_dt:.0f}s df={df_stft:.4e}  n_side={args.stft_n_side} "
          f"seg alpha={args.stft_window_alpha} freq_from_tdi_phase={bool(args.freq_from_tdi_phase)}")

    w_bins = max(1, int(round(args.band_uhz * 1e-6 / df)))
    results = []
    hdr = (f"  {'idx':>6} {'f0(mHz)':>9} {'amp':>9} {'gap(uHz)':>9} |"
           f" {'leg|dat':>9} {'fdc|leg':>9} {'fdc|dat':>9} |"
           f" {'stft|leg':>9} {'insten':>9} {'sten-int':>9} {'stft|dat':>9} {'insten':>9}")
    print("\nmm = 1 - |O| per source (Re-variants + stencil fractions in the JSON):")
    print(hdr)

    for idx in picks:
        p9 = params_all[idx]
        f0 = float(p9[1])
        others = f_sorted[f_sorted != f_all[idx]]
        gap = float(np.min(np.abs(others - f_all[idx]))) if others.size else np.inf

        # legacy legs
        h_leg_w = legacy_fd_template(gb_legacy, p9, n_win, args.dt,
                                     oversample=args.oversample,
                                     window=("tukey" if alpha > 0 else None),
                                     window_alpha=alpha)
        h_leg_r = legacy_fd_template(gb_legacy, p9, n_win, args.dt,
                                     oversample=args.oversample)
        td_leg = np.fft.irfft(h_leg_r / args.dt, n=n_win, axis=-1)
        leg_stft = brute_stft(td_leg, t0_data, args.dt, stft_settings, seg_win)

        # fd-comp leg (rectangular); full-grid ndarray target -> row start 0
        tpl = np.zeros((1, NCH, n_rfft), dtype=np.complex128)
        fd_comp.fill_global(p9[None, :], tpl,
                            data_index=np.zeros(1, dtype=np.int32))
        h_fdc = tpl[0]

        # stft leg
        M = np.zeros((1, NCH, stft_settings.NT, stft_settings.NF_active),
                     np.complex128)
        stft_comp.fill_global_stft(p9[None, :], M,
                                   data_index=np.zeros(1, dtype=np.int32))
        M = M[0]

        # FD slice around f0
        k0 = int(round(f0 / df))
        s = slice(max(k0 - w_bins, 0), min(k0 + w_bins + 1, n_rfft))
        mm_leg_dat = overlap_mm(data_fd_w[:, s], h_leg_w[:, s])
        mm_fdc_leg = overlap_mm(h_leg_r[:, s], h_fdc[:, s])
        mm_fdc_dat = overlap_mm(data_fd_r[:, s], h_fdc[:, s])

        # STFT column slice around the carrier
        c0 = int(np.floor(f0 / df_stft)) - int(stft_settings.ind_min)
        c_lo = max(c0 - args.stft_n_side - 2, 0)
        c_hi = min(c0 + args.stft_n_side + 3, stft_settings.NF_active)
        mm_stft_leg = stft_mms(leg_stft, M, c_lo, c_hi)
        mm_stft_dat = stft_mms(data_stft, M, c_lo, c_hi)

        row = dict(
            index=int(idx), f0=f0, amp=float(p9[0]), gap_hz=gap,
            fd=dict(legacy_vs_data=dict(re=mm_leg_dat[0], abs=mm_leg_dat[1]),
                    fdcomp_vs_legacy=dict(re=mm_fdc_leg[0], abs=mm_fdc_leg[1]),
                    fdcomp_vs_data=dict(re=mm_fdc_dat[0], abs=mm_fdc_dat[1])),
            stft=dict(vs_legacy=mm_stft_leg, vs_data=mm_stft_dat),
            stft_cols=[int(c_lo), int(c_hi)],
        )
        results.append(row)
        flag = "" if gap > (args.stft_n_side + 1) * df_stft else "  <-- neighbor inside STFT stencil"
        print(f"  {idx:>6d} {f0*1e3:>9.5f} {p9[0]:>9.2e} {gap*1e6:>9.3f} |"
              f" {mm_leg_dat[1]:>9.2e} {mm_fdc_leg[1]:>9.2e} {mm_fdc_dat[1]:>9.2e} |"
              f" {mm_stft_leg['full_abs']:>9.2e} {mm_stft_leg['stencil_abs']:>9.2e}"
              f" {mm_stft_leg['stencil_interior_abs']:>9.2e}"
              f" {mm_stft_dat['full_abs']:>9.2e} {mm_stft_dat['stencil_abs']:>9.2e}{flag}")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gb_mojito_stft_fd_mismatch.json")
    with open(out_path, "w") as fp:
        json.dump(dict(config=vars(args), t0_data=t0_data,
                       t_ref=t_ref, results=results), fp, indent=2)
    print(f"\nJSON -> {out_path}")

    if args.selftest:
        # data IS the legacy sum: the legacy-vs-data leg must be ~exact for
        # the isolated sources, and every other leg must be sane.
        worst_leg = max(r["fd"]["legacy_vs_data"]["abs"] for r in results)
        worst_fdc = max(r["fd"]["fdcomp_vs_legacy"]["abs"] for r in results)
        worst_int = max(r["stft"]["vs_legacy"]["stencil_interior_abs"]
                        for r in results)
        print(f"[selftest] worst legacy|data={worst_leg:.3e}  "
              f"fdcomp|legacy={worst_fdc:.3e}  "
              f"stft-stencil-interior|legacy={worst_int:.3e}")
        assert worst_leg < 1e-6, "legacy waveform does not match its own data embed"
        assert worst_fdc < 5e-3, "FD comp diverges from the legacy waveform"
        # interior in-stencil = method error + waveform-family difference;
        # the edge segments additionally carry the irfft wrap of the
        # synthetic reference and are reported, not asserted.
        assert worst_int < 2e-3, "STFT Fresnel diverges in-stencil (interior)"
        print("[selftest] GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
