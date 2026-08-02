"""Prototype: RATIO-SPLINE candidate build for the sig-het in-model likelihood.

Idea (2026-07-30): the sig-het fold already consumes the heterodyne ratio
``r = c1/c0``.  Today ``c1`` (candidate sparse WDM coefficients) is built
exactly: control-point spline waveform -> N slow-time samples -> FFT ->
polyphase.  But every smooth/fast feature of the waveform lives in the
REFERENCE and cancels in the ratio, so instead model r directly:

  r_c(t) = h_cand_c(t) / h_ref_c(t)  =  exp(dlnA_c(t) + i dphi_c(t)),

fit a spline through r at a handful of node times (n_r raw evals instead of
n_cp), and either

  (i)  rebuild the slow-time series as r_hat(t) * s_ref(t) and run the
       EXACT FFT + polyphase on it (safe rung), or
  (ii) skip FFT + polyphase entirely: feed r_hat evaluated at the sparse
       WDM sample times straight into the bin-fold (fast rung).

This script threads both variants end to end WITHOUT touching interior code,
gating every hand-written stage against implemented machinery first:

  GATE F  (fold):   Python fold on r built from the IMPLEMENTED
                    ``gb_signal_het_make_reference`` c1 must reproduce the
                    production ``gb_signal_het_get_ll_in_kernel`` d_h/h_h.
  GATE X  (build):  Python slow-series + FFT + polyphase c1 must match the
                    implemented make_reference c1 at the coefficient level.

Scaffold (grid, sources, holder, engine) is copied from
``gb_sighet_inmodel_validate.py`` so the production path is the same one the
in-model gates validated.

Run:  python scripts/gb_chunked_het/gb_sighet_ratio_build_prototype.py
Env:  RATIO_PLOT_DIR   directory for output figures/npz (default ./ratio_proto_out)
      RATIO_NR_LIST    comma list of node counts    (default "4,8,16,32")
      RATIO_SWEEP      1 = full displacement sweep  (default 1)
"""
import os

# Pin every thread pool BEFORE numpy/scipy import (laptop CPU budget rule).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import numpy as np
from scipy.interpolate import CubicSpline

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine, PHYS_IDX_PHI0


class _FullGridWDMHolder:
    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Python mirrors of the kernel stages (formulas verbatim from
# gb_tdi_on_the_fly.cu::gb_signal_het_consume_one_source / gbfd_build_one_source;
# each is gated against the implemented C++ before use).
# ---------------------------------------------------------------------------

def m_active_for(f0_cand, g):
    """consume_one_source lines 2090-2097: clipped active m-band."""
    m_floor = int(math.floor(f0_cand / g["layer_df"]))
    m = m_floor + np.arange(-g["m_half"], g["m_half"] + 1)
    return np.clip(m, g["ind_min_f"], g["ind_min_f"] + g["Nf_active"] - 1)


def kernel_tukey_slow(N, alpha):
    """gbfd_build_one_source lines 428-439 with edge_frac = 0 (the in-kernel
    path passes 0.0): Tukey taper over the N slow samples."""
    w = np.ones(N)
    if alpha > 0.0:
        n_taper = 0.5 * alpha * (N - 1)
        if n_taper > 0.0:
            di = np.arange(N, dtype=float)
            dlast = float(N - 1)
            lo = di < n_taper
            w[lo] = 0.5 * (1.0 + np.cos(np.pi * (di[lo] / n_taper - 1.0)))
            hi = di > dlast - n_taper
            w[hi] = 0.5 * (1.0 + np.cos(np.pi * ((dlast - di[hi]) / n_taper - 1.0)))
    return w


def X_lin_from_slow(s_slow, g, N):
    """Builder tail: window, FFT, 0.5*dts scale; then the consumer's
    fft-order -> linear map with fft_order_scale = 1/dt folded in.
    X_lin[i] sits at absolute dense bin k_f0 + (i - N/2)."""
    w = kernel_tukey_slow(N, g["tukey_alpha"])
    Xf = np.fft.fft(s_slow * w[None, :], axis=-1)
    dts = g["Tobs"] / N
    Xf *= 0.5 * dts / g["dt"]
    i = np.arange(N)
    return Xf[:, (i - N // 2) % N]


def polyphase_py(X_lin, k_f0, m_active, g, window_full, n_sparse_local):
    """consume_one_source steps (1)-(2): gather fold + iFFT(Nt_layer) +
    per-pixel lisatools coefficient -> c1_sparse (3, M, N_sparse_t)."""
    N = X_lin.shape[-1]
    half_NS = N // 2
    Nt = g["Nt"]
    half_Nt = Nt // 2
    NL = g["nt_layer"]
    Nsp = g["N_sparse_t"]
    stride = g["stride"]
    n_start = g["ind_min_t"] + int(n_sparse_local[0])
    M = len(m_active)

    j = np.arange(Nt)
    j_off = j - half_Nt
    wpre = window_full * np.exp(1j * 2.0 * np.pi * j_off * n_start / Nt)

    fold = np.zeros((3, M, NL), dtype=np.complex128)
    for im, m_g in enumerate(m_active):
        j_base = k_f0 - half_NS + half_Nt - int(m_g) * half_Nt
        i = j - j_base
        valid = (i >= 0) & (i < N)
        contrib = np.zeros((3, Nt), dtype=np.complex128)
        contrib[:, valid] = X_lin[:, i[valid]] * wpre[valid]
        fold[:, im, :] = contrib.reshape(3, Nt // NL, NL).sum(axis=1)

    ifft_full = np.fft.ifft(fold, axis=-1)[:, :, :Nsp]     # numpy ifft == kernel
    n_layer = np.arange(Nsp)
    n_global = n_start + n_layer * stride
    sign_scale = np.where(n_global % 2 == 1, -1.0, 1.0) / stride
    kappa = 2.0 * math.sqrt(math.pi * g["dt"]) / g["Nf"]

    c1 = np.empty_like(ifft_full)
    for im, m_g in enumerate(m_active):
        m_plus_n = (int(m_g) + n_global) & 1
        conj_cmn = np.where(m_plus_n == 0, 1.0 + 0.0j, 0.0 - 1.0j)
        sign_mn = np.where(((int(m_g) + 1) * n_global) & 1 == 1, -1.0, 1.0)
        c1[:, im, :] = ifft_full[:, im, :] * (sign_scale * kappa * sign_mn) * conj_cmn
    return c1


def ratio_dr(c1_rows, c0_rows, stride, max_r=0.0):
    """consume_one_source step (3): floored ratio + centred-FD dr."""
    max_mag = np.abs(c0_rows).max(axis=-1, keepdims=True)
    floor = np.maximum(1e-12 * max_mag, 1e-300)
    mask = np.abs(c0_rows) > floor
    r = np.where(mask, c1_rows / np.where(mask, c0_rows, 1.0), 0.0 + 0.0j)
    if max_r > 0.0:
        a = np.abs(r)
        over = a > max_r
        r = np.where(over, r * (max_r / np.where(over, a, 1.0)), r)
    dr = fd_dr(r, stride)
    return r, dr, mask


def fd_dr(r, stride):
    """Kernel's centred/one-sided finite difference along b (Dn = stride)."""
    Dn = float(stride)
    dr = np.zeros_like(r)
    if r.shape[-1] >= 3:
        dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / (2.0 * Dn)
        dr[..., 0] = (r[..., 1] - r[..., 0]) / Dn
        dr[..., -1] = (r[..., -1] - r[..., -2]) / Dn
    elif r.shape[-1] == 2:
        dr[..., 0] = dr[..., 1] = (r[..., 1] - r[..., 0]) / Dn
    return dr


def fold_py(r, dr, m_active, sighet, data_idx=0):
    """consume_one_source step (4), project_real=1, XYZ: raw sums * 0.5."""
    g = sighet._g
    ml = np.asarray(m_active) - g["ind_min_f"]
    A0 = np.asarray(sighet.A0_all)[data_idx][:, ml, :]
    A1 = np.asarray(sighet.A1_all)[data_idx][:, ml, :]
    d_h_raw = (A0 * r + A1 * dr).sum()

    B0 = np.asarray(sighet.B0_all)[data_idx][:, :, ml, :]
    B1 = np.asarray(sighet.B1_all)[data_idx][:, :, ml, :]
    B0nc = np.asarray(sighet.B0nc_all)[data_idx][:, :, ml, :]
    B1nc = np.asarray(sighet.B1nc_all)[data_idx][:, :, ml, :]
    r_c, r_c2 = r[:, None], r[None, :]
    dr_c, dr_c2 = dr[:, None], dr[None, :]
    h_h_raw = (B0 * (np.conj(r_c) * r_c2)
               + B1 * (np.conj(r_c) * dr_c2 + np.conj(dr_c) * r_c2)
               + B0nc * (r_c * r_c2)
               + B1nc * (r_c * dr_c2 + dr_c * r_c2)).sum()
    return 0.5 * d_h_raw.real, 0.5 * h_h_raw.real


# ---------------------------------------------------------------------------
# Implemented-machinery helpers
# ---------------------------------------------------------------------------

def kernel_c1_full(sighet, params9):
    """EXACT candidate sparse/dense WDM coefficients from the IMPLEMENTED
    producer ``gb_signal_het_make_reference`` (full-band call, mirroring the
    class ctor).  This is the same build+polyphase the in-kernel scorer runs."""
    g = sighet._g
    c1_sparse = np.zeros((1, 3, g["Nf_active"], g["N_sparse_t"]),
                         dtype=np.complex128)
    c1_dense = np.zeros((1, 3, g["Nf_active"], g["Nt_active"]),
                        dtype=np.complex128)
    sighet.cpp.gb_signal_het_make_reference(
        sighet.tdi_wrap, c1_sparse, c1_dense,
        np.asarray(sighet.window_full), np.asarray(sighet.n_sparse_local),
        np.zeros(1, dtype=np.int32),
        np.ascontiguousarray(np.asarray(params9, dtype=float).reshape(1, 9)),
        1, 9, 1, 2,
        g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
        g["nt_layer"], g["N_sparse_t"], g["stride"],
        g["ind_min_t"], g["ind_min_f"], g["layer_df"], g["dt"],
        g["Tobs"], g["t0"],
        3, g["n_sparse_fd"], g["tukey_alpha"], g["n_cp_build"])
    return c1_sparse[0], c1_dense[0]


def build_spl(gen, params9):
    """One spline-object build per candidate (reusable across chunk records)."""
    amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = [
        np.array([v]) for v in params9]
    return gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
               convert_to_ra_dec=False, return_spline=True)


def slow_series(gen, params9, t_dense, N, g, kf0_pin=None, spl=None):
    """Complex heterodyned slow-time series from the INSTALLED generator's
    spline decomposition: ``eval_spline_vals`` gives (tdi_amp, tdi_phase,
    phase_ref) per channel -- the SAME amp/phase objects the kernel's
    control-point build evaluates (``eval_tdi`` is
    Re[amp * exp(-i(tdi_phase + phase_ref))], so the positive-frequency
    envelope is amp * exp(+i(...))).  No analytic-signal/Hilbert step: a
    pure phi0 displacement gives an EXACTLY constant ratio here.
    Evaluated directly at the builder's slow grid t = t_start + n*Tobs/N;
    UNWINDOWED.  (t_dense arg kept for call-site compatibility; unused.)"""
    del t_dense
    if spl is None:
        spl = build_spl(gen, params9)
    tau = np.arange(N) * (g["Tobs"] / N)
    t_eval = g["t0"] + tau
    a, tph, pref = spl.eval_spline_vals(t_eval)
    a, tph, pref = np.asarray(a)[0], np.asarray(tph)[0], np.asarray(pref)[0]
    z = a * np.exp(1j * (tph + pref[None, :]))           # (3, N) envelope
    kf0 = int(kf0_pin) if kf0_pin is not None else int(round(
        float(params9[1]) * g["Tobs"]))
    f0g = kf0 / g["Tobs"]
    return z * np.exp(-1j * 2.0 * np.pi * f0g * tau), kf0, f0g


def good_samples(s_ref, frac=0.02):
    """Samples where EVERY channel's reference envelope is well away from a
    null: r = s_cand/s_ref is numerically polluted where |s_ref| ~ 0 (the
    fold's |c0|^2 weighting makes those times irrelevant anyway)."""
    a = np.abs(s_ref)
    return (a / a.max(axis=-1, keepdims=True)).min(axis=0) >= frac


def fit_ratio(r_slow, tau_slow, n_r, kind, good=None, derot=None):
    """Fit (dlnA, dphi) per channel through n_r nodes (snapped to the nearest
    good sample if a validity mask is given); return a callable r_hat(tau).

    ``derot``: optional callable tau -> analytic carrier-difference phase
    (2*pi*(df0*tau + 0.5*dfdot*tau^2)) removed BEFORE the fit and restored at
    evaluation -- the second-level heterodyne: the known deterministic ramp
    never costs spline nodes and never risks the unwrap."""
    if derot is not None:
        r_fit = r_slow * np.exp(-1j * derot(tau_slow))[None, :]
    else:
        r_fit = r_slow
    dlnA = np.log(np.abs(r_fit))
    dphi = np.unwrap(np.angle(r_fit), axis=-1)
    idx = np.unique(np.round(np.linspace(0, len(tau_slow) - 1, n_r)).astype(int))
    if good is not None and not good.all():
        gi = np.flatnonzero(good)
        idx = np.unique(gi[np.abs(gi[None, :] - idx[:, None]).argmin(axis=1)])
    tn = tau_slow[idx]
    if kind == "cubic":
        fA = CubicSpline(tn, dlnA[:, idx], axis=1)
        fP = CubicSpline(tn, dphi[:, idx], axis=1)

        def r_base(tau):
            return np.exp(fA(tau) + 1j * fP(tau))
    else:
        def r_base(tau, _tn=tn, _A=dlnA[:, idx], _P=dphi[:, idx]):
            out = np.empty((3, len(tau)), dtype=np.complex128)
            for c in range(3):
                out[c] = np.exp(np.interp(tau, _tn, _A[c])
                                + 1j * np.interp(tau, _tn, _P[c]))
            return out

    if derot is not None:
        def r_hat(tau):
            return r_base(tau) * np.exp(1j * derot(tau))[None, :]
    else:
        r_hat = r_base
    return r_hat, len(idx)


def main():
    out_dir = os.environ.get("RATIO_PLOT_DIR", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    nr_list = [int(x) for x in
               os.environ.get("RATIO_NR_LIST", "4,8,16,32").split(",")]

    # ---- scaffold: identical to gb_sighet_inmodel_validate.py -------------
    # RATIO_NT stretches Tobs (Nt=512 -> 15.2 d, the validate default;
    # Nt=12288 -> 364 d) to measure how the required node count scales with
    # observation time -- the annual response modulation only completes full
    # cycles at ~1 yr.
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, int(os.environ.get("RATIO_NT", "512"))
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    edge = 40

    # RATIO_EDGE: WDM time-edge crop in layers.  The default 40 matches the
    # validate scaffold; the sig-het accuracy rule is edge >= Tukey taper
    # (= 0.025 * Nt layers at alpha=0.05) -- at 1 yr that is ~307 layers.
    edge = int(os.environ.get("RATIO_EDGE", str(edge)))
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    # WINDOW POLICY (user directive 2026-07-31, for ALL methods): a Tukey
    # taper of RATIO_TAPER_WAVELETS wavelets (fixed ABSOLUTE width set by
    # the wavelet grid, default 8) AND a time crop (min_time/max_time)
    # covering at least the taper -- the analyzed region always sees
    # window == 1, at every Tobs. alpha = 2*K/Nt is passed to BOTH engines
    # so builds and analysis share one window.
    _tk = int(os.environ.get("RATIO_TAPER_WAVELETS", "8"))
    _alpha_policy = 2.0 * _tk / Nt
    assert edge >= _tk + 4, (
        f"crop ({edge} wavelets) must cover the taper ({_tk}) + margin")
    print(f"[window] taper = {_tk} wavelets = {_tk * Nf * dt:.0f} s per side "
          f"(alpha={_alpha_policy:.5f}); crop = {edge} wavelets "
          f"= {edge * Nf * dt:.0f} s per side")
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
        tukey_alpha=_alpha_policy,
    )
    chunked.convert_to_ra_dec = False
    # RATIO_NCP: control-point count for the IMPLEMENTED build (0 = direct
    # per-point eval).  Default 0 so the stash reference and my dense Python
    # thread share the same waveform representation -- with the production
    # spline build (32) the known ~1e-4 spline-vs-dense coefficient
    # difference shows up as a floor in every ratio-arm comparison.
    _ncp = int(os.environ.get("RATIO_NCP", "0"))
    _ntl = int(os.environ.get("RATIO_NTL", "64"))
    _nsfd = int(os.environ.get("RATIO_NSFD", "512"))
    _mhalf = int(os.environ.get("RATIO_MHALF", "2"))
    sighet = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=_nsfd, n_cp_build=_ncp, nt_layer=_ntl,
        m_active_half_width=_mhalf)
    if os.environ.get("RATIO_TUKEY0", "0") == "1":
        # Discriminator: build candidates/references with NO slow-grid taper
        # (the engine's fill_global/dense convention is untapered; the
        # build's alpha-fraction taper grows with Tobs while the WDM edge
        # crop does not -- the measured 1-yr cold systematic).
        sighet._g["tukey_alpha"] = 0.0
        print("  [tukey0] sig-het build taper DISABLED")
    g = sighet._g
    N = g["n_sparse_fd"]
    print(f"[grid] Nf={Nf} Nt={Nt} Tobs={g['Tobs']:.3e}s "
          f"({g['Tobs']/86400:.1f} d) N_sparse_fd={N} nt_layer={g['nt_layer']} "
          f"stride={g['stride']} N_sparse_t={g['N_sparse_t']} "
          f"tukey_alpha={g['tukey_alpha']} n_cp_build={g['n_cp_build']}")

    eng_s = WDMBandLikelihoodEngine(sighet, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    B = A.copy()
    B[1] += 0.6 * layer_df
    B[4:7] = [2.9, 1.1, 0.9]

    h = np.zeros((3, Nf, Nt))
    chunked.fill_global_wdm(A[None, :], h, convert_to_ra_dec=False)
    chunked.fill_global_wdm(B[None, :], h, convert_to_ra_dec=False)
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    h_act = np.ascontiguousarray(h[:, ilo:ihi, wdm_set.active_slice_t])
    nch, nfa, nta = h_act.shape
    # RATIO_PSD: "identity" (validate-script default) or a lisatools noise
    # model name ("scirdv1") for a PHYSICAL inverse-sensitivity -- then the
    # delta columns are in real lnL units.
    psd_mode = os.environ.get("RATIO_PSD", "identity")
    if psd_mode == "identity":
        invC = np.zeros((nch, nch, nfa, nta))
        for c in range(nch):
            invC[c, c] = 1.0
    else:
        from lisatools.sensitivity import XYZ2SensitivityMatrix
        invC = np.ascontiguousarray(
            np.asarray(XYZ2SensitivityMatrix(wdm_set, model=psd_mode).invC),
            dtype=np.float64)
        assert invC.shape == (nch, nch, nfa, nta), invC.shape
        print(f"[psd] physical inverse-sensitivity: {psd_mode}")
    holder = _FullGridWDMHolder(h_act, invC)
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})

    sighet.setup_in_model(holder, A[None, :], zeros)
    print("[setup] in-model reference built at A "
          f"(f0={A[1]*1e3:.6f} mHz, {A[1]/layer_df - int(A[1]/layer_df):.2f} "
          "layer fraction)")

    gen = sighet._keep_alive["gb_gen"]
    Nobs = Nf * Nt
    t_dense = np.arange(Nobs) * dt + t_start
    tau_slow = np.arange(N) * (g["Tobs"] / N)
    n_sparse_local = np.asarray(sighet.n_sparse_local)
    window_full = np.asarray(sighet.window_full, dtype=np.float64)
    c0_all = np.asarray(sighet.c0_sparse_all)[0]         # (3, Nf_active, Nsp)

    def prod_delta(p):
        eng_s.get_ll(holder, p[None, :], phase_maximize=False, **kw)
        return float(eng_s.d_h_out[0]), float(eng_s.h_h_out[0])

    def c0_rows_for(m_act):
        return c0_all[:, np.asarray(m_act) - g["ind_min_f"], :]

    # ---- candidate battery for the gates ----------------------------------
    rng = np.random.default_rng(7)
    battery = [A.copy()]
    for _ in range(4):
        p = A.copy()
        p[0] *= 1.0 + 0.02 * rng.standard_normal()
        p[1] += 0.02 * layer_df * rng.standard_normal()
        p[2] *= 1.0 + 0.05 * rng.standard_normal()
        p[PHYS_IDX_PHI0] += 0.2 * rng.standard_normal()
        p[5] += 0.05 * rng.standard_normal()
        battery.append(p)
    for spec in [(0, "mul", 1.0), (PHYS_IDX_PHI0, "add", 1.0),
                 (1, "addf", 2e-4), (7, "add", 0.1)]:
        p = A.copy()
        idx, mode, s = spec
        if mode == "mul":
            p[idx] *= np.exp(s)
        elif mode == "addf":
            p[idx] += s * layer_df * Nt  # in dense-bin-ish units? no: layers
        else:
            p[idx] += s
        battery.append(p)
    # fix the f0 entry: displace by +0.2 layers (gate scale is ~3e-4 layers;
    # this is far out but exercises the machinery)
    battery[-2] = A.copy()
    battery[-2][1] += 2e-1 * layer_df

    # ---- GATE F: Python fold on implemented c1 vs production kernel -------
    print("\n[GATE F] fold_py(r = make_reference(cand)/c0_ref) vs "
          "in-kernel get_ll")
    worst_f = 0.0
    for i, p in enumerate(battery):
        dh_k, hh_k = prod_delta(p)
        m_act = m_active_for(p[1], g)
        c1_k, _ = kernel_c1_full(sighet, p)
        c1_rows = c1_k[:, np.asarray(m_act) - g["ind_min_f"], :]
        r, dr, _ = ratio_dr(c1_rows, c0_rows_for(m_act), g["stride"],
                            max_r=g["max_r"])
        dh_p, hh_p = fold_py(r, dr, m_act, sighet)
        rel_d = abs(dh_p - dh_k) / max(abs(dh_k), 1e-30)
        rel_h = abs(hh_p - hh_k) / max(abs(hh_k), 1e-30)
        worst_f = max(worst_f, rel_d, rel_h)
        print(f"  cand{i}: d_h rel={rel_d:.2e}  h_h rel={rel_h:.2e}")
    print(f"  GATE F worst rel = {worst_f:.2e}  "
          f"[{'OK' if worst_f < 1e-8 else 'FAIL'}]")

    # ---- GATE X: Python build thread vs implemented make_reference --------
    print("\n[GATE X] polyphase_py(X from slow-series) vs make_reference c1")
    worst_x = 0.0
    for i, p in enumerate(battery[:5]):
        m_act = m_active_for(p[1], g)
        c1_k, _ = kernel_c1_full(sighet, p)
        c1_k_rows = c1_k[:, np.asarray(m_act) - g["ind_min_f"], :]
        s, kf0, _ = slow_series(gen, p, t_dense, N, g)
        X_lin = X_lin_from_slow(s, g, N)
        c1_p = polyphase_py(X_lin, kf0, m_act, g, window_full, n_sparse_local)
        scale = np.abs(c1_k_rows).max()
        rel = np.abs(c1_p - c1_k_rows).max() / max(scale, 1e-300)
        rel_conj = np.abs(np.conj(c1_p) - c1_k_rows).max() / max(scale, 1e-300)
        worst_x = max(worst_x, min(rel, rel_conj))
        note = "" if rel <= rel_conj else "  [CONJ MATCHES BETTER -- sign flip!]"
        print(f"  cand{i}: c1 max rel={rel:.2e} (conj {rel_conj:.2e}){note}")
    print(f"  GATE X worst rel = {worst_x:.2e}  "
          f"[{'OK' if worst_x < 1e-5 else 'FAIL'}]")

    # ---- within-thread exact scorer --------------------------------------
    def thread_exact(p):
        m_act = m_active_for(p[1], g)
        s, kf0, _ = slow_series(gen, p, t_dense, N, g)
        X_lin = X_lin_from_slow(s, g, N)
        c1 = polyphase_py(X_lin, kf0, m_act, g, window_full, n_sparse_local)
        r, dr, _ = ratio_dr(c1, c0_rows_for(m_act), g["stride"], g["max_r"])
        return fold_py(r, dr, m_act, sighet)

    print("\n[thread fidelity] within-thread exact vs production kernel "
          "(delta = d_h - 0.5 h_h)")
    for i, p in enumerate(battery[:5]):
        dh_k, hh_k = prod_delta(p)
        dh_t, hh_t = thread_exact(p)
        dk, dtr = dh_k - 0.5 * hh_k, dh_t - 0.5 * hh_t
        print(f"  cand{i}: delta kernel={dk:+.6e} thread={dtr:+.6e} "
              f"absdiff={abs(dk - dtr):.2e}")

    # ---- reference slow series (common carrier for all ratio work) --------
    s_ref, kf0_ref, f0g_ref = slow_series(gen, A, t_dense, N, g)

    # ---- version (ii) time-mapping calibration ----------------------------
    # r_hat is fit in tau (slow-time seconds from t_start); the fold's sparse
    # samples sit at WDM pixel n_global = ind_min_t + n_sparse_local[b].
    # Calibrate tau_pix = (n_global + off) * Nf * dt against the EXACT sparse
    # ratio of a displaced candidate on the carrier row.
    n_global = g["ind_min_t"] + n_sparse_local
    p_cal = A.copy()
    p_cal[0] *= np.exp(0.5)
    p_cal[5] += 0.3
    p_cal[7] += 0.05
    m_act = m_active_for(p_cal[1], g)
    c1_k, _ = kernel_c1_full(sighet, p_cal)
    c1_rows = c1_k[:, np.asarray(m_act) - g["ind_min_f"], :]
    r_ex, _, mask_ex = ratio_dr(c1_rows, c0_rows_for(m_act), g["stride"])
    good = good_samples(s_ref)
    print(f"  [good-mask] {int(good.sum())}/{N} slow samples clear of "
          "reference-envelope nulls")
    s_cal, _, _ = slow_series(gen, p_cal, t_dense, N, g, kf0_pin=kf0_ref)
    r_slow_cal = s_cal / s_ref
    r_hat_cal, _ = fit_ratio(r_slow_cal, tau_slow, 64, "cubic", good=good)
    best_off, best_err = None, np.inf
    imc = g["m_half"]
    for off in (0.0, 0.5, 1.0, -0.5):
        tau_pix = (n_global + off) * Nf * dt
        rh = r_hat_cal(tau_pix)
        m = mask_ex[:, imc, :]
        err = np.abs(rh - r_ex[:, imc, :])[m].max()
        print(f"  [pix-off cal] off={off:+.1f}: max|r_hat - r_exact| = {err:.3e}")
        if err < best_err:
            best_off, best_err = off, err
    print(f"  [pix-off cal] using off={best_off:+.1f}")
    tau_pix = (n_global + best_off) * Nf * dt

    # ---- the two approximation arms ---------------------------------------
    def arm_i_from_rhat(p, r_hat):
        m_act = m_active_for(p[1], g)
        s_i = r_hat(tau_slow) * s_ref
        X_i = X_lin_from_slow(s_i, g, N)
        c1_i = polyphase_py(X_i, kf0_ref, m_act, g, window_full,
                            n_sparse_local)
        r_i, dr_i, _ = ratio_dr(c1_i, c0_rows_for(m_act), g["stride"],
                                g["max_r"])
        dh_i, hh_i = fold_py(r_i, dr_i, m_act, sighet)
        return dh_i - 0.5 * hh_i

    def arm_ii_from_rhat(p, r_hat):
        m_act = m_active_for(p[1], g)
        c0_rows = c0_rows_for(m_act)
        rh = r_hat(tau_pix)                              # (3, Nsp)
        max_mag = np.abs(c0_rows).max(axis=-1, keepdims=True)
        mask = np.abs(c0_rows) > np.maximum(1e-12 * max_mag, 1e-300)
        r_ii = np.where(mask, rh[:, None, :], 0.0 + 0.0j)
        dr_ii = fd_dr(r_ii, g["stride"])
        dh_ii, hh_ii = fold_py(r_ii, dr_ii, m_act, sighet)
        return dh_ii - 0.5 * hh_ii

    def arms(p, n_r, kind):
        """Return (delta_i, delta_ii, n_nodes) for candidate p."""
        s_cand, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
        r_slow = s_cand / s_ref
        r_hat, n_used = fit_ratio(r_slow, tau_slow, n_r, kind, good=good)
        return (arm_i_from_rhat(p, r_hat), arm_ii_from_rhat(p, r_hat),
                n_used)

    # within-thread exact but with the PINNED carrier (same as the arms use),
    # so the comparison isolates the ratio-spline approximation alone.
    def thread_exact_pinned(p):
        m_act = m_active_for(p[1], g)
        s, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
        X_lin = X_lin_from_slow(s, g, N)
        c1 = polyphase_py(X_lin, kf0_ref, m_act, g, window_full,
                          n_sparse_local)
        r, dr, _ = ratio_dr(c1, c0_rows_for(m_act), g["stride"], g["max_r"])
        dh, hh = fold_py(r, dr, m_act, sighet)
        return dh - 0.5 * hh

    # ---- CHUNKED FD-heterodyne build (RATIO_CHUNKED=1) --------------------
    # User design (2026-07-31): run the sig-het build + polyphase in TIME
    # RECORDS (mini-observations), exactly the chunked-het decomposition,
    # so the kernel scratch scales with the record instead of Tobs and the
    # stride/accuracy knob decouples from GPU shared memory.  Alignment
    # rules that make the assembly exact on interior pixels:
    #   * record start a_r == ind_min_t (mod stride) AND even -> the
    #     record's sparse-pixel comb lands ON the global comb and every
    #     parity-dependent WDM basis factor ((-1)^n, C_mn, sign_mn) is
    #     IDENTICAL, so no correction factors are needed;
    #   * pad P = stride pixels per side absorbs the record's own
    #     2-wavelet policy taper + the wavelet time support; only interior
    #     pixels are kept;
    #   * each record is a self-consistent mini-observation: own installed
    #     WDMSettings window, own carrier snap (the chain is exact for any
    #     snap), heterodyne referenced to the record start.
    # Gates: CH-1 assembled coefficients vs make_reference (bit-level on
    # interior pixels); CH-2 fold(ll) vs the production in-kernel engine.
    # Then v3 rung (i) THROUGH the chunked thread: r-hat x record
    # reference series -> record FFT -> record polyphase -> fold.
    if os.environ.get("RATIO_CHUNKED", "0") == "1":
        from lisatools.signal_het import sparse_time_grid as _stg

        stride_t = g["stride"]                  # from RATIO_NTL (main grid)
        P = int(os.environ.get("RATIO_CH_PAD", "0")) or max(stride_t, 8)
        P = int(math.ceil(P / stride_t)) * stride_t   # comb-aligned pad
        Nt_rec = int(os.environ.get("RATIO_CH_NTREC", "0"))
        if Nt_rec == 0:
            Nt_rec = max(8 * P, ((Nt // 8) // stride_t) * stride_t)
        Nt_rec -= Nt_rec % stride_t
        adv = Nt_rec - 2 * P
        assert adv > 0 and adv % stride_t == 0 and (2 * P) % stride_t == 0
        n_rec = int(math.ceil(g["Nt_active"] / adv))
        # Record taper: >= 2 wavelets AND >= 8 SLOW SAMPLES of the record's
        # N-point build grid.  A sub-sample taper quantizes into a near-hard
        # record edge whose spectral leakage contaminates interior
        # coefficients (measured: 7e-3 coeff / ~22 raw lnL at 45-d records
        # with a 0.66-sample taper) -- the taper-RESOLUTION rule, same
        # mechanism as the earlier policy-window confound.
        _tsamp = float(os.environ.get("RATIO_CH_TAPER_SAMPLES", "8"))
        taper_pix = max(2, int(math.ceil(_tsamp * Nt_rec / N)))
        rec_alpha = 2.0 * taper_pix / Nt_rec
        assert P >= taper_pix + 4, (
            f"pad {P} pixels must cover record taper {taper_pix} + margin")
        wdm_rec = WDMSettings(
            Nf, Nt_rec, dt, t0=t_start,
            min_freq=1e-4, max_freq=2e-2,
            min_time=0.0, max_time=Nt_rec * Nf * dt,
            force_backend=backend)
        win_rec = np.asarray(wdm_rec.window, dtype=np.float64)
        NL_rec = Nt_rec // stride_t
        _, Nsp_rec, nsl_rec = _stg(Nt_rec, Nt_rec, NL_rec)
        print(f"[chunk] {n_rec} records x Nt_rec={Nt_rec} pixels "
              f"({Nt_rec * Nf * dt / 86400:.1f} d), pad={P}, adv={adv}, "
              f"stride={stride_t}, NL_rec={NL_rec}, Nsp_rec={Nsp_rec}, "
              f"rec_alpha={rec_alpha:.5f} (2 wavelets)")
        # GPU scratch this geometry implies (fold + c1/r/dr, per block):
        M = 2 * g["m_half"] + 1
        scratch = (3 * M * NL_rec + 3 * 3 * M * Nsp_rec) * 16
        print(f"[chunk] kernel scratch at this geometry: {scratch/1024:.0f} KB "
              f"(A100 opt-in cap 160 KB)")

        def g_rec_for(a_r):
            gr = dict(g)
            gr.update(Nt=Nt_rec, Nt_active=Nt_rec, nt_layer=NL_rec,
                      N_sparse_t=Nsp_rec, stride=stride_t, ind_min_t=0,
                      Tobs=Nt_rec * Nf * dt,
                      t0=t_start + a_r * Nf * dt,
                      tukey_alpha=rec_alpha)
            return gr

        def chunked_c1(p, spl, m_act, r_hat=None, spl_ref=None,
                       p_ref=None):
            """Assembled (3, M, N_sparse_t_main) coefficients from records.
            r_hat=None -> exact build; else rung (i): candidate record
            series = r_hat(t_abs) * reference record series (spl_ref)."""
            out = np.full((3, len(m_act), g["N_sparse_t"]), np.nan,
                          dtype=np.complex128)
            for r in range(n_rec):
                a_r = g["ind_min_t"] + r * adv - P
                if a_r + Nt_rec > Nt:
                    # slide inward in STRIDE multiples: keeps the record's
                    # pixel comb on the global comb and a_r even.
                    over = a_r + Nt_rec - Nt
                    a_r -= int(math.ceil(over / stride_t)) * stride_t
                gr = g_rec_for(a_r)
                if r_hat is None:
                    s, kf0_r, _ = slow_series(gen, p, None, N, gr, spl=spl)
                else:
                    s_ref_r, kf0_r, _ = slow_series(
                        gen, A if p_ref is None else p_ref, None, N, gr,
                        spl=spl_ref)
                    tau_abs = (gr["t0"] - t_start
                               + np.arange(N) * (gr["Tobs"] / N))
                    s = r_hat(tau_abs) * s_ref_r
                X = X_lin_from_slow(s, gr, N)
                c1_r = polyphase_py(X, kf0_r, m_act, gr, win_rec, nsl_rec)
                # interior record pixels -> global comb slots
                n_glob = a_r + nsl_rec                     # global pixels
                keep = ((n_glob >= a_r + P) & (n_glob < a_r + Nt_rec - P)
                        & (n_glob >= g["ind_min_t"] + stride_t // 2)
                        & (n_glob < g["ind_min_t"] + g["Nt_active"]))
                b_g = (n_glob - g["ind_min_t"] - stride_t // 2) // stride_t
                ok = keep & (b_g >= 0) & (b_g < g["N_sparse_t"]) & (
                    (n_glob - g["ind_min_t"] - stride_t // 2) % stride_t == 0)
                out[:, :, b_g[ok]] = np.where(
                    np.isnan(out[:, :, b_g[ok]].real),
                    c1_r[:, :, ok], out[:, :, b_g[ok]])
                # overlap halo: first writer wins (interior-only anyway)
            return out

        # ---- GATE CH-1: coefficients vs implemented make_reference -------
        print("\n[GATE CH-1] chunk-assembled c1 vs make_reference "
              "(interior pixels, taper-free global zone)")
        for i, p in enumerate(battery[:4]):
            m_act = m_active_for(p[1], g)
            c1_k, _ = kernel_c1_full(sighet, p)
            c1_k_rows = c1_k[:, np.asarray(m_act) - g["ind_min_f"], :]
            spl = build_spl(gen, p)
            c1_ch = chunked_c1(p, spl, m_act)
            filled = ~np.isnan(c1_ch.real)
            scale = np.abs(c1_k_rows[filled]).max()
            rel = (np.abs(c1_ch - c1_k_rows)[filled].max()
                   / max(scale, 1e-300))
            print(f"  cand{i}: coverage {filled[0,0].sum()}/"
                  f"{g['N_sparse_t']}  max rel={rel:.2e}")

        # ---- GATE CH-2: fold(ll) vs production engine --------------------
        print("[GATE CH-2] chunked-build ll vs production in-kernel get_ll")
        worst = 0.0
        for i, p in enumerate(battery[:6]):
            dh_k, hh_k = prod_delta(p)
            m_act = m_active_for(p[1], g)
            spl = build_spl(gen, p)
            c1_ch = chunked_c1(p, spl, m_act)
            c1_ch = np.where(np.isnan(c1_ch.real), 0.0, c1_ch)
            r, dr, _ = ratio_dr(c1_ch, c0_rows_for(m_act), g["stride"],
                                g["max_r"])
            dh_p, hh_p = fold_py(r, dr, m_act, sighet)
            dk = dh_k - 0.5 * hh_k
            dp = dh_p - 0.5 * hh_p
            err = abs(dp - dk) / max(abs(dk), 1e-300)
            worst = max(worst, err)
            print(f"  cand{i}: delta prod={dk:+.6e} chunked={dp:+.6e} "
                  f"rel={err:.2e}")
        print(f"  GATE CH-2 worst rel = {worst:.2e}")

        # ---- rung (i) THROUGH the chunked thread on cold draws -----------
        print("\n[CHUNKED rung(i)] cold draws vs dense truth "
              "(raw lnL and rel), n_r nodes on the global ratio")
        from lisatools.sensitivity import XYZ2SensitivityMatrix as _XYZ2
        rngc = np.random.default_rng(11)

        def dense_terms(p):
            ht = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], ht, convert_to_ra_dec=False)
            ht_act = ht[:, ilo:ihi, wdm_set.active_slice_t]
            dh = np.einsum("cmn,cdmn,dmn->", h_act, invC, ht_act)
            hh = np.einsum("cmn,cdmn,dmn->", ht_act, invC, ht_act)
            return float(dh), float(hh)

        eng_c2 = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                         tdi_channel_setup="XYZ")
        eng_c2.get_ll(holder, A[None, :], phase_maximize=False, **kw)
        hh_ref_c = float(eng_c2.h_h_out[0])
        _, hh_dA = dense_terms(A)
        norm = hh_ref_c / hh_dA
        spl_ref = build_spl(gen, A)
        n_draws = int(os.environ.get("RATIO_CH_NDRAWS", "8"))
        nr_ch = [int(x) for x in
                 os.environ.get("RATIO_CH_NR", "16,32").split(",")]
        hdr = "  {:>4s} {:>12s} {:>11s}".format("#", "|delta_dense|",
                                                "exact-ch")
        for nr in nr_ch:
            hdr += f"  rung-i n{nr}"
        print(hdr + "   [raw lnL err]")
        stats = {k: [] for k in ["ex"] + nr_ch}
        for i in range(n_draws):
            p = A.copy()
            p[0] *= np.exp(0.02 * rngc.standard_normal())
            p[1] += 0.02 * rngc.standard_normal() / (2 * np.pi * g["Tobs"])
            p[4] += 0.05 * rngc.standard_normal()
            p[5] += 0.02 * rngc.standard_normal()
            p[6] += 0.02 * rngc.standard_normal()
            p[7] += 0.01 * rngc.standard_normal()
            p[8] += 0.01 * rngc.standard_normal()
            dh_d, hh_d = dense_terms(p)
            d_dense = norm * (dh_d - 0.5 * hh_d)
            m_act = m_active_for(p[1], g)
            c0r = c0_rows_for(m_act)
            spl = build_spl(gen, p)
            line = f"  {i:>4d} {abs(d_dense):12.4e}"
            c1_ch = np.where(np.isnan((tmp := chunked_c1(p, spl, m_act)).real),
                             0.0, tmp)
            r, dr, _ = ratio_dr(c1_ch, c0r, g["stride"], g["max_r"])
            dh_p, hh_p = fold_py(r, dr, m_act, sighet)
            e_ex = abs((dh_p - 0.5 * hh_p) - d_dense)
            stats["ex"].append(e_ex)
            line += f" {e_ex:11.3f}"
            s_cand_g, _, _ = slow_series(gen, p, None, N, g,
                                         kf0_pin=kf0_ref, spl=spl)
            r_slow = s_cand_g / s_ref
            for nr in nr_ch:
                rh, _ = fit_ratio(r_slow, tau_slow, nr, "cubic", good=good)
                c1_i = np.where(
                    np.isnan((tmp := chunked_c1(p, spl, m_act, r_hat=rh,
                                                spl_ref=spl_ref)).real),
                    0.0, tmp)
                ri, dri, _ = ratio_dr(c1_i, c0r, g["stride"], g["max_r"])
                dh_i, hh_i = fold_py(ri, dri, m_act, sighet)
                e_i = abs((dh_i - 0.5 * hh_i) - d_dense)
                stats[nr].append(e_i)
                line += f"  {e_i:10.3f}"
            print(line)
        print("  medians: exact-ch {:.3f}".format(
            np.median(stats["ex"])) + "".join(
            f"  n{nr} {np.median(stats[nr]):.3f}" for nr in nr_ch))

        # ---- MANY-REFERENCE test (RATIO_CH_REFS=1, user design) ----------
        # Draw a SOURCE from prior-scale ranges, make it the data AND the
        # heterodyne reference (production semantics: the reference is
        # always the current point), take SMALL production-scale deviations
        # around it, and compare rung-(i) | sig-het v2 | chunked het |
        # dense for each.  Population coverage is over REFERENCE GEOMETRY
        # (iota/psi/sky/f0 across the band), not over wild candidates.
        if os.environ.get("RATIO_CH_REFS", "0") == "1":
            n_ref = int(os.environ.get("RATIO_CH_REFS_N", "30"))
            nr_rf = int(os.environ.get("RATIO_CH_REFS_NR", "64"))
            rngr = np.random.default_rng(19)
            eng_c3 = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                             tdi_channel_setup="XYZ")
            drift_u = 1.0 / (2 * np.pi * g["Tobs"])   # 1 rad of carrier drift
            DEVS = [("f0+", 1, 0.1 * drift_u), ("f0-", 1, -0.3 * drift_u),
                    ("lnA", 0, 0.1), ("lnA2", 0, -1.0),
                    ("fdot", 2, 0.2 / (np.pi * g["Tobs"] ** 2)),
                    ("phi0", 4, 0.5), ("iota", 5, 0.2), ("psi", 6, -0.2),
                    ("lam", 7, 0.03), ("beta", 8, -0.08)]
            print(f"\n[MANY-REF] {n_ref} random reference sources x "
                  f"{len(DEVS)} small deviations; raw |dLL| vs dense; "
                  f"rung(i) n={nr_rf}")
            # ---- NULL-FIX arms (RATIO_CH_FIX=1, 2026-08-01) --------------
            # A: null-window patch -- in windows where the REFERENCE
            #    envelope dips below NP_THR (known free at setup), replace
            #    r_hat*s_ref with a direct Re/Im cubic through NP_NW node
            #    evals of the CANDIDATE envelope (smooth through its own
            #    null; no division anywhere).  Emulates +NP_NW raw evals
            #    per window in the kernel.
            # B: fold-side fix -- Wiener-regularized division
            #    r = c1 conj(c0)/(|c0|^2 + (REG_EPS*rowmax)^2) and ZERO the
            #    dr (A1/B1 sub-stride linear-interp) terms on flagged
            #    pixels (|c0| < REG_THR*rowmax, dilated +-1): the linear
            #    model of r near a pole is the catastrophic term.
            # Controls: iu = unchunked rung-i baseline (same thread as A),
            # xp = EXACT candidate build + plain fold (fold-error floor --
            # if xp fails the bar, no series-side fix alone can pass).
            fix_on = os.environ.get("RATIO_CH_FIX", "0") == "1"
            np_thr = float(os.environ.get("NP_THR", "0.08"))
            np_pad = int(os.environ.get("NP_PAD", "2"))
            np_nw = int(os.environ.get("NP_NW", "6"))
            reg_eps = float(os.environ.get("REG_EPS", "0.05"))
            reg_thr = float(os.environ.get("REG_THR", "0.1"))
            arm_keys = (("i", "v2", "ch")
                        + (("iu", "xp", "x0", "B", "BW", "A", "AB")
                           if fix_on else ()))
            if fix_on:
                print(f"[fix] NP_THR={np_thr} NP_PAD={np_pad} NP_NW={np_nw} "
                      f"REG_EPS={reg_eps} REG_THR={reg_thr}")

            def null_patch(s_model, s_cand, a_norm, tau):
                out = s_model.copy()
                nwin = 0
                for c in range(3):
                    bad = a_norm[c] < np_thr
                    for _ in range(np_pad):
                        b2 = bad.copy()
                        b2[1:] |= bad[:-1]
                        b2[:-1] |= bad[1:]
                        bad = b2
                    if not bad.any():
                        continue
                    db = np.diff(bad.astype(np.int8))
                    starts = list(np.flatnonzero(db == 1) + 1)
                    ends = list(np.flatnonzero(db == -1))
                    if bad[0]:
                        starts = [0] + starts
                    if bad[-1]:
                        ends = ends + [len(bad) - 1]
                    for i0, i1 in zip(starts, ends):
                        j0 = max(i0 - 1, 0)
                        j1 = min(i1 + 1, len(bad) - 1)
                        # node count scales with window width (a wide window
                        # at ~2 samples/node) -- the smoke showed a fixed 6
                        # under-resolves wide windows and makes A WORSE
                        k = min(max(np_nw, (j1 - j0 + 1) // 2 + 1), 16,
                                j1 - j0 + 1)
                        if k < 2:
                            continue
                        idx = np.unique(np.round(
                            np.linspace(j0, j1, k)).astype(int))
                        sl = slice(i0, i1 + 1)
                        if len(idx) >= 4:
                            out[c, sl] = (
                                CubicSpline(tau[idx],
                                            s_cand[c, idx].real)(tau[sl])
                                + 1j * CubicSpline(
                                    tau[idx], s_cand[c, idx].imag)(tau[sl]))
                        else:
                            out[c, sl] = (
                                np.interp(tau[sl], tau[idx],
                                          s_cand[c, idx].real)
                                + 1j * np.interp(tau[sl], tau[idx],
                                                 s_cand[c, idx].imag))
                        nwin += 1
                return out, nwin

            def fold_reg(c1_rows, c0_rows, m_act):
                # FLAG-GATED: plain division off-window (the global Wiener
                # denominator biased CLEAN refs by ~2 percent in r => tens
                # of raw lnL at high SNR); Wiener-suppressed r + zeroed dr
                # only on flagged pixels
                mx = np.abs(c0_rows).max(axis=-1, keepdims=True)
                r, dr, _ = ratio_dr(c1_rows, c0_rows, g["stride"],
                                    g["max_r"])
                flag = np.abs(c0_rows) < reg_thr * mx
                fl = flag.copy()
                fl[..., 1:] |= flag[..., :-1]
                fl[..., :-1] |= flag[..., 1:]
                rW = (c1_rows * np.conj(c0_rows)
                      / (np.abs(c0_rows) ** 2
                         + np.maximum((reg_eps * mx) ** 2, 1e-300)))
                r = np.where(fl, rW, r)
                dr = fd_dr(r, g["stride"])
                dr[fl] = 0.0
                dh, hh = fold_py(r, dr, m_act, sighet)
                return dh - 0.5 * hh

            def fold_win(c1_rows, c0_rows, m_act, p):
                """BW: remove flagged null windows from the sparse fold
                entirely (r = dr = 0 there) and ADD BACK their exact
                dense-pixel contribution using the implemented producer's
                dense candidate coefficients (in the kernel this comes
                free from the rung-ii envelope->pixel map)."""
                mxr = np.abs(c0_rows).max(axis=-1, keepdims=True)
                flag = (np.abs(c0_rows) < reg_thr * mxr).any(axis=(0, 1))
                fl = flag.copy()
                fl[1:] |= flag[:-1]
                fl[:-1] |= flag[1:]
                r, dr, _ = ratio_dr(c1_rows, c0_rows, g["stride"],
                                    g["max_r"])
                if not fl.any():
                    dh, hh = fold_py(r, dr, m_act, sighet)
                    return dh - 0.5 * hh
                r[..., fl] = 0.0
                dr[..., fl] = 0.0
                dh, hh = fold_py(r, dr, m_act, sighet)
                _, c1_d = kernel_c1_full(sighet, p)
                ml = np.asarray(m_act) - g["ind_min_f"]
                half = g["stride"] // 2
                dmask = np.zeros(h_act_r.shape[-1], bool)
                base = int(np.asarray(sighet.n_sparse_local)[0])
                for n_s in np.flatnonzero(fl):
                    ja = base + n_s * g["stride"]
                    dmask[max(ja - half, 0):
                          min(ja + half, len(dmask))] = True
                w1 = c1_d[:, ml][:, :, dmask].real
                dw = h_act_r[:, ml][:, :, dmask]
                iw = invC[:, :, ml][:, :, :, dmask]
                if not hasattr(fold_win, "_cal"):
                    fold_win._cal = True
                    w1f = c1_d[:, ml].real
                    hh_f = np.einsum("cmj,cdmj,dmj->", w1f,
                                     invC[:, :, ml], w1f)
                    print(f"        [BW cal] Re-basis dense h_h over "
                          f"m_act rows / engine h_h at ref: "
                          f"{hh_f:.6e} (engine-side check via truth col)")
                dh_w = np.einsum("cmj,cdmj,dmj->", w1, iw, dw)
                hh_w = np.einsum("cmj,cdmj,dmj->", w1, iw, w1)
                return (dh + dh_w) - 0.5 * (hh + hh_w)

            summ = {k: [] for k in arm_keys}
            geo = []
            errs_mat = []
            for rrr in range(n_ref):
                ref = np.array([
                    10 ** rngr.uniform(-22.5, -21.0),
                    rngr.uniform(1.5e-3, 1.5e-2),
                    rngr.uniform(0.0, 3e-16),
                    0.0,
                    rngr.uniform(0, 2 * np.pi),
                    np.arccos(rngr.uniform(-1, 1)),
                    rngr.uniform(0, np.pi),
                    rngr.uniform(0, 2 * np.pi),
                    np.arcsin(rngr.uniform(-1, 1)),
                ])
                href = np.zeros((3, Nf, Nt))
                chunked.fill_global_wdm(ref[None, :], href,
                                        convert_to_ra_dec=False)
                h_act_r = np.ascontiguousarray(
                    href[:, ilo:ihi, wdm_set.active_slice_t])
                holder_r = _FullGridWDMHolder(h_act_r, invC)

                def dense_r(p, _h=h_act_r):
                    ht = np.zeros((3, Nf, Nt))
                    chunked.fill_global_wdm(p[None, :], ht,
                                            convert_to_ra_dec=False)
                    ht_a = ht[:, ilo:ihi, wdm_set.active_slice_t]
                    return (float(np.einsum("cmn,cdmn,dmn->", _h, invC, ht_a)),
                            float(np.einsum("cmn,cdmn,dmn->", ht_a, invC,
                                            ht_a)))

                eng_c3.get_ll(holder_r, ref[None, :], phase_maximize=False,
                              **kw)
                hh_r = float(eng_c3.h_h_out[0])
                _, hh_dr = dense_r(ref)
                norm_r = hh_r / hh_dr
                sighet.clear_in_model()
                sighet.setup_in_model(holder_r, ref[None, :], zeros)
                spl_r = build_spl(gen, ref)
                gref = dict(g)
                s_ref_r, kf0_r, _ = slow_series(gen, ref, None, N, gref,
                                                spl=spl_r)
                good_r = good_samples(s_ref_r)
                ar = np.abs(s_ref_r)
                a_norm_r = ar / np.maximum(
                    ar.max(axis=1, keepdims=True), 1e-300)
                min_env = float(a_norm_r.min())
                errs_ref = {k: [] for k in arm_keys}
                nwin_ref = 0
                if fix_on:
                    # r == 1 self-check: fold the reference against itself.
                    # Any mismatch vs the engine h_h is a STASH defect at
                    # this geometry, independent of every candidate arm.
                    m_ar = m_active_for(ref[1], g)
                    c0r_ = np.asarray(sighet.c0_sparse_all)[0][
                        :, np.asarray(m_ar) - g["ind_min_f"], :]
                    r1, dr1, _ = ratio_dr(c0r_, c0r_, g["stride"], 0.0)
                    dh1, hh1 = fold_py(r1, dr1, m_ar, sighet)
                    print(f"        [self] r==1 fold vs engine: "
                          f"dh {dh1 - hh_r:+.4e}  hh {hh1 - hh_r:+.4e}  "
                          f"raw dLL {abs((dh1 - 0.5 * hh1) - 0.5 * hh_r):.3f}")
                for name, idx, dv in DEVS:
                    p = ref.copy()
                    if idx == 0:
                        p[0] *= np.exp(dv)
                    else:
                        p[idx] += dv
                    dh_d, hh_d = dense_r(p)
                    d_dense = norm_r * (dh_d - 0.5 * hh_d)
                    eng_c3.get_ll(holder_r, p[None, :],
                                  phase_maximize=False, **kw)
                    d_ch = float(eng_c3.d_h_out[0] - 0.5 * eng_c3.h_h_out[0])
                    eng_s.get_ll(holder_r, p[None, :],
                                 phase_maximize=False, **kw)
                    d_v2 = float(eng_s.d_h_out[0] - 0.5 * eng_s.h_h_out[0])
                    m_act = m_active_for(p[1], g)
                    spl_p = build_spl(gen, p)
                    s_c, _, _ = slow_series(gen, p, None, N, gref,
                                            kf0_pin=kf0_r, spl=spl_p)
                    rh, _ = fit_ratio(s_c / s_ref_r, tau_slow, nr_rf,
                                      "cubic", good=good_r)
                    c1_i = np.where(
                        np.isnan((tmp := chunked_c1(
                            p, spl_p, m_act, r_hat=rh,
                            spl_ref=spl_r, p_ref=ref)).real), 0.0, tmp)
                    ri, dri, _ = ratio_dr(
                        c1_i,
                        np.asarray(sighet.c0_sparse_all)[0][
                            :, np.asarray(m_act) - g["ind_min_f"], :],
                        g["stride"], g["max_r"])
                    dh_i, hh_i = fold_py(ri, dri, m_act, sighet)
                    d_i = dh_i - 0.5 * hh_i
                    extra = ()
                    if fix_on:
                        c0_rows_r = np.asarray(sighet.c0_sparse_all)[0][
                            :, np.asarray(m_act) - g["ind_min_f"], :]
                        # xp / B: EXACT candidate thread build
                        X_c = X_lin_from_slow(s_c, gref, N)
                        c1_x = polyphase_py(X_c, kf0_r, m_act, gref,
                                            window_full, n_sparse_local)
                        r_x, dr_x, _ = ratio_dr(c1_x, c0_rows_r,
                                                g["stride"], g["max_r"])
                        dh_x, hh_x = fold_py(r_x, dr_x, m_act, sighet)
                        d_xp = dh_x - 0.5 * hh_x
                        # x0: same but NO max_r clip -- discriminates
                        # clip-loss from dr-linearization at the pole
                        r_0, dr_0, _ = ratio_dr(c1_x, c0_rows_r,
                                                g["stride"], 0.0)
                        dh_0, hh_0 = fold_py(r_0, dr_0, m_act, sighet)
                        d_x0 = dh_0 - 0.5 * hh_0
                        d_B = fold_reg(c1_x, c0_rows_r, m_act)
                        d_BW = fold_win(c1_x, c0_rows_r, m_act, p)
                        # iu / A / AB: fitted-ratio series, unchunked thread
                        s_model = rh(tau_slow) * s_ref_r
                        X_u = X_lin_from_slow(s_model, gref, N)
                        c1_u = polyphase_py(X_u, kf0_r, m_act, gref,
                                            window_full, n_sparse_local)
                        r_u, dr_u, _ = ratio_dr(c1_u, c0_rows_r,
                                                g["stride"], g["max_r"])
                        dh_u, hh_u = fold_py(r_u, dr_u, m_act, sighet)
                        d_iu = dh_u - 0.5 * hh_u
                        s_patch, nwin_ref = null_patch(
                            s_model, s_c, a_norm_r, tau_slow)
                        X_a = X_lin_from_slow(s_patch, gref, N)
                        c1_a = polyphase_py(X_a, kf0_r, m_act, gref,
                                            window_full, n_sparse_local)
                        r_a, dr_a, _ = ratio_dr(c1_a, c0_rows_r,
                                                g["stride"], g["max_r"])
                        dh_a, hh_a = fold_py(r_a, dr_a, m_act, sighet)
                        d_A = dh_a - 0.5 * hh_a
                        d_AB = fold_reg(c1_a, c0_rows_r, m_act)
                        extra = (("iu", d_iu), ("xp", d_xp), ("x0", d_x0),
                                 ("B", d_B), ("BW", d_BW), ("A", d_A),
                                 ("AB", d_AB))
                        if os.environ.get("RATIO_CH_DIAG", "0") == "1":
                            print(f"          [dev {name:5s}] xp err "
                                  f"{abs(d_xp - d_dense):10.3f}  dh "
                                  f"{dh_x - norm_r * dh_d:+10.3f}  0.5hh "
                                  f"{0.5 * (hh_x - norm_r * hh_d):+10.3f}")
                    for k, d in (("i", d_i), ("v2", d_v2),
                                 ("ch", d_ch)) + extra:
                        errs_ref[k].append(abs(d - d_dense))
                        summ[k].append(abs(d - d_dense))
                geo.append((ref[1], ref[5], ref[6], min_env,
                            max(errs_ref["i"]), max(errs_ref["v2"]),
                            max(errs_ref["ch"]), np.sqrt(hh_r)))
                errs_mat.append([errs_ref[k] for k in arm_keys])
                print(f"  ref{rrr:02d} f0={ref[1]*1e3:7.3f}mHz SNR~"
                      f"{np.sqrt(hh_r):6.1f} min_env={min_env:.3f} | "
                      f"max raw err: rung-i {max(errs_ref['i']):8.3f} "
                      f"v2 {max(errs_ref['v2']):8.3f} "
                      f"chunk {max(errs_ref['ch']):8.2e}")
                if fix_on:
                    print(f"        fix: iu {max(errs_ref['iu']):9.3f} "
                          f"xp {max(errs_ref['xp']):9.3f} "
                          f"x0 {max(errs_ref['x0']):9.3f} "
                          f"B {max(errs_ref['B']):9.3f} "
                          f"BW {max(errs_ref['BW']):9.3f} "
                          f"A {max(errs_ref['A']):9.3f} "
                          f"AB {max(errs_ref['AB']):9.3f} "
                          f"nwin={nwin_ref}")
            lbls = {"i": "rung (i)", "v2": "sig-het v2", "ch": "chunked",
                    "iu": "rung-i unchunk", "xp": "exact+plainfold",
                    "x0": "exact+noclip", "B": "B exact+regfold",
                    "BW": "BW win-addback", "A": "A patch+plain",
                    "AB": "A+B patch+reg"}
            for k in arm_keys:
                v = np.array(summ[k])
                print(f"[MANY-REF] {lbls[k]:>16s}: median {np.median(v):.4f}"
                      f"  p95 {np.percentile(v, 95):.3f}  max {v.max():.3f} "
                      f"raw lnL over {v.size} (ref, dev) pairs")
            np.savez(os.path.join(out_dir, "ratio_manyref.npz"),
                     geo=np.array(geo))
            if fix_on:
                np.savez(os.path.join(out_dir, "ratio_manyref_fix.npz"),
                         geo=np.array(geo), errs=np.array(errs_mat),
                         arm_keys=np.array(arm_keys),
                         knobs=np.array([np_thr, np_pad, np_nw,
                                         reg_eps, reg_thr]))
            return

        # ---- MANY-PRIOR-DRAWS gate-coverage test (RATIO_CH_PRIOR=1) ------
        # Candidates from PRIOR-SCALE distributions (dlnA over decades,
        # full-sphere sky, f0 to +-1 layer) scored by rung (i) vs dense.
        # The claim under test: every draw INSIDE the trust region meets
        # raw |dLL| < bar; every large error lies OUTSIDE the gate (the
        # gate boundary == the validity boundary; zero in-gate violators).
        if os.environ.get("RATIO_CH_PRIOR", "0") == "1":
            n_pr = int(os.environ.get("RATIO_CH_PRIOR_N", "300"))
            nr_pr = int(os.environ.get("RATIO_CH_PRIOR_NR", "64"))
            R_LT = 499.00478
            rngp = np.random.default_rng(7)
            print(f"\n[PRIOR DRAWS] {n_pr} prior-scale draws, rung(i) "
                  f"n={nr_pr}, vs dense truth; gate: |dlnA|<=1.5, "
                  "drift<=0.5 rad, sky-Doppler<=1.0 rad, fdot<=0.5 rad")
            rows_pr = []
            for i in range(n_pr):
                p = A.copy()
                near = rngp.random() < 0.5   # 50% near-gate oversampling:
                # full-prior draws land in-gate ~0.1% of the time (the sky
                # Doppler gate is ~0.11 rad at 3 mHz), so the BOUNDARY
                # region needs its own population to test coverage.
                if near:
                    dlnA_d = rngp.uniform(-2.0, 2.0)
                    df0 = rngp.uniform(-1.0, 1.0) / (2 * np.pi * g["Tobs"])
                    p[2] += rngp.uniform(-2e-16, 2e-16)
                    ang = abs(rngp.normal(0.0, 0.12))
                    th = rngp.uniform(0, 2 * np.pi)
                    lam = A[7] + ang * np.cos(th) / max(np.cos(A[8]), 0.2)
                    bet = np.clip(A[8] + ang * np.sin(th), -1.4, 1.4)
                else:
                    dlnA_d = rngp.uniform(-5.0, 5.0)
                    if rngp.random() < 0.7:
                        df0 = rngp.uniform(-0.3, 0.3) / (2 * np.pi * g["Tobs"])
                    else:
                        df0 = rngp.uniform(-1.0, 1.0) * g["layer_df"]
                    p[2] += rngp.uniform(-1e-15, 1e-15)
                    lam = rngp.uniform(0, 2 * np.pi)
                    bet = np.arcsin(rngp.uniform(-1, 1))
                p[0] *= np.exp(dlnA_d)
                p[1] += df0
                p[4] = rngp.uniform(0, 2 * np.pi)
                p[5] = np.arccos(rngp.uniform(-1, 1))
                p[6] = rngp.uniform(0, np.pi)
                p[7], p[8] = lam, bet
                drift = 2 * np.pi * abs(df0) * g["Tobs"]
                fd_drift = np.pi * abs(p[2] - A[2]) * g["Tobs"] ** 2
                # angular separation on the sphere from the reference
                cs = (np.sin(A[8]) * np.sin(bet)
                      + np.cos(A[8]) * np.cos(bet) * np.cos(lam - A[7]))
                angsep = np.arccos(np.clip(cs, -1, 1))
                D_sky = 2 * np.pi * p[1] * R_LT * angsep
                in_gate = (abs(dlnA_d) <= 1.5 and drift <= 0.5
                           and D_sky <= 1.0 and fd_drift <= 0.5)
                dh_d, hh_d = dense_terms(p)
                d_dense = norm * (dh_d - 0.5 * hh_d)
                m_act = m_active_for(p[1], g)
                spl = build_spl(gen, p)
                s_cand_g, _, _ = slow_series(gen, p, None, N, g,
                                             kf0_pin=kf0_ref, spl=spl)
                r_slow = s_cand_g / s_ref
                rh, _ = fit_ratio(r_slow, tau_slow, nr_pr, "cubic",
                                  good=good)
                c1_i = np.where(
                    np.isnan((tmp := chunked_c1(p, spl, m_act, r_hat=rh,
                                                spl_ref=spl_ref)).real),
                    0.0, tmp)
                ri, dri, _ = ratio_dr(c1_i, c0_rows_for(m_act), g["stride"],
                                      g["max_r"])
                dh_i, hh_i = fold_py(ri, dri, m_act, sighet)
                err = abs((dh_i - 0.5 * hh_i) - d_dense)
                # polarization-geometry displacements (ungated so far) and a
                # cheap null-risk flag: minimum CANDIDATE envelope at the
                # ratio nodes relative to its own per-channel max -- an
                # envelope zero-crossing between ref and cand is the
                # log-polar representation's structural failure (amplitude
                # cusp + pi phase step the spline cannot represent).
                dio = abs(p[5] - A[5])
                dps = min(abs(p[6] - A[6]), np.pi - abs(p[6] - A[6]))
                ac = np.abs(s_cand_g)
                min_amp_rel = float(
                    (ac / np.maximum(ac.max(axis=1, keepdims=True), 1e-300))
                    .min())
                ar = np.abs(s_ref)
                min_ref_rel = float(
                    (ar / np.maximum(ar.max(axis=1, keepdims=True), 1e-300))
                    .min())
                rows_pr.append((in_gate, err, abs(dlnA_d), drift, D_sky,
                                fd_drift, abs(d_dense), dio, dps,
                                min_amp_rel, min_ref_rel))
                if (i + 1) % 50 == 0:
                    print(f"  ... {i+1}/{n_pr}")
            ig = np.array([r[0] for r in rows_pr])
            er = np.array([r[1] for r in rows_pr])
            print(f"[PRIOR] in-gate: {ig.sum()}/{n_pr}; "
                  f"in-gate err max={er[ig].max() if ig.any() else 0:.3f} "
                  f"median={np.median(er[ig]) if ig.any() else 0:.3f} "
                  f"(bar: raw < 1)")
            print(f"[PRIOR] out-gate err median={np.median(er[~ig]):.1f} "
                  f"max={er[~ig].max():.1f}")
            viol = ig & (er > 1.0)
            print(f"[PRIOR] IN-GATE VIOLATORS (err>1): {viol.sum()}")
            for r, v in zip(rows_pr, viol):
                if v:
                    print(f"   VIOL err={r[1]:.2f} dlnA={r[2]:.2f} "
                          f"drift={r[3]:.3f} Dsky={r[4]:.2f} fd={r[5]:.3f}")
            np.savez(os.path.join(out_dir, "ratio_prior_draws.npz"),
                     rows=np.array(rows_pr))
        return

    # ---- logL error vs waveform-spline density (RATIO_NCPSWEEP=1) ---------
    # Chunked het's N_cp_sig is nodes PER CHUNK (Nt_sub layers); sig-het
    # v2's n_cp_build is nodes PER TOBS.  Common axis = node spacing.
    # Truth = dense; anchor engine = the scaffold's exact chunked
    # (N_cp_sig=0, direct eval).  Cold posterior-width draws only.
    if os.environ.get("RATIO_NCPSWEEP", "0") == "1":
        rng4 = np.random.default_rng(11)
        eng_ex = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                         tdi_channel_setup="XYZ")

        def dense_terms(p):
            ht = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], ht, convert_to_ra_dec=False)
            ht_act = ht[:, ilo:ihi, wdm_set.active_slice_t]
            dh = np.einsum("cmn,cdmn,dmn->", h_act, invC, ht_act)
            hh = np.einsum("cmn,cdmn,dmn->", ht_act, invC, ht_act)
            return float(dh), float(hh)

        eng_ex.get_ll(holder, A[None, :], phase_maximize=False, **kw)
        hh_ref = float(eng_ex.h_h_out[0])
        _, hh_dA = dense_terms(A)
        norm = hh_ref / hh_dA
        n_draws = int(os.environ.get("RATIO_NCPSWEEP_N", "12"))
        draws = []
        for _ in range(n_draws):
            p = A.copy()
            p[0] *= np.exp(0.02 * rng4.standard_normal())
            p[1] += (0.02 * rng4.standard_normal()
                     / (2.0 * np.pi * g["Tobs"]))
            p[4] += 0.05 * rng4.standard_normal()
            p[5] += 0.02 * rng4.standard_normal()
            p[6] += 0.02 * rng4.standard_normal()
            p[7] += 0.01 * rng4.standard_normal()
            p[8] += 0.01 * rng4.standard_normal()
            dh_d, hh_d = dense_terms(p)
            draws.append((p, norm * (dh_d - 0.5 * hh_d)))

        def score(eng):
            raw = []
            for p, d_dense in draws:
                eng.get_ll(holder, p[None, :], phase_maximize=False, **kw)
                d = float(eng.d_h_out[0] - 0.5 * eng.h_h_out[0])
                raw.append(abs(d - d_dense))
            raw = np.array(raw)
            rel = raw / max(abs(hh_ref), 1e-300)
            return (np.median(raw), raw.max(), np.median(rel), rel.max())

        chunk_dur_s = 128 * Nf * dt   # Nt_sub = 128 layers (scaffold)
        n_chunks = g["Nt"] / 128.0
        print(f"\n[NCPSWEEP] {n_draws} cold draws; Tobs={g['Tobs']/86400:.1f} d;"
              f" chunk={chunk_dur_s/86400:.2f} d x {n_chunks:.0f} chunks;"
              f" raw err in lnL, rel = raw/h_h_ref({hh_ref:.3e})")
        print("  CHUNKED HET vs N_cp_sig (nodes/chunk):")
        print("   ncp  spacing[h]  evals/cand   raw med/max      rel med/max")
        m0, x0, rm0, rx0 = score(eng_ex)
        print(f"     0      exact  {'dense':>10s}  {m0:9.3f} {x0:9.3f}"
              f"  {rm0:.2e} {rx0:.2e}")
        for ncp in (16, 32, 48, 96):
            ch = GBWDMComputations(
                wdm_set, t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
                N_cp_sig=ncp, N_cp_orbit=0, orbits=orbits,
                tdi_config="2nd generation", force_backend=backend,
                d_d=0.0, tdi_type="XYZ", tukey_alpha=_alpha_policy)
            ch.convert_to_ra_dec = False
            eng = WDMBandLikelihoodEngine(ch, wdm_set, nchannels=3,
                                          tdi_channel_setup="XYZ")
            m, x, rm, rx = score(eng)
            print(f"  {ncp:4d} {chunk_dur_s/3600/max(ncp-1,1):10.2f} "
                  f"{int(ncp*n_chunks):10d}  {m:9.3f} {x:9.3f}"
                  f"  {rm:.2e} {rx:.2e}")
        print("  SIG-HET v2 vs n_cp_build (nodes/Tobs), "
              f"nt_layer={_ntl} nsfd={_nsfd} m_half={_mhalf}:")
        print("   ncp  spacing[h]  evals/cand   raw med/max      rel med/max")
        for ncp_b in (0, 48, 93, 128, 256):
            sh = GBSignalHetComputations.for_band_engine(
                chunked, n_sparse_fd=_nsfd, nt_layer=_ntl,
                m_active_half_width=_mhalf, n_cp_build=ncp_b)
            sh.setup_in_model(holder, A[None, :], zeros)
            eng = WDMBandLikelihoodEngine(sh, wdm_set, nchannels=3,
                                          tdi_channel_setup="XYZ")
            m, x, rm, rx = score(eng)
            sp = (g["Tobs"] / 3600 / max(ncp_b - 1, 1)) if ncp_b else 0.0
            lbl = f"{sp:10.2f}" if ncp_b else "    direct"
            ev = ncp_b if ncp_b else g["n_sparse_fd"]
            print(f"  {ncp_b:4d} {lbl} {ev:10d}  {m:9.3f} {x:9.3f}"
                  f"  {rm:.2e} {rx:.2e}")
        return

    # ---- four-way logL comparison (RATIO_COMPARE=1) -----------------------
    # dense lisatools truth | chunked het | current sig-het (production
    # in-kernel) | new sig-het (ratio build, adaptive nodes + derot), on
    # random 9-dim joint draws.  All four share the same data, grid and
    # inverse-sensitivity; the compared quantity is delta = d_h - 0.5 h_h
    # (the constant -0.5 d_d cancels).  Dense truth is normalized to the
    # engine convention once, at the reference.
    if os.environ.get("RATIO_COMPARE", "0") == "1":
        n_cmp = int(os.environ.get("RATIO_COMPARE_N", "24"))
        rng3 = np.random.default_rng(11)
        R_LT = 499.00478
        eng_c = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                        tdi_channel_setup="XYZ")

        def dense_terms(p):
            ht = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], ht, convert_to_ra_dec=False)
            ht_act = ht[:, ilo:ihi, wdm_set.active_slice_t]
            dh = np.einsum("cmn,cdmn,dmn->", h_act, invC, ht_act)
            hh = np.einsum("cmn,cdmn,dmn->", ht_act, invC, ht_act)
            return float(dh), float(hh)

        # Compiled sig_het_v3 (ratio build in-kernel), when the rebuilt
        # backend exposes it: a fifth column through the REAL class path.
        eng_v3 = None
        if hasattr(sighet.cpp, "gb_signal_het_v3_get_ll"):
            sighet_v3 = GBSignalHetComputations.for_band_engine(
                chunked, n_sparse_fd=512, n_cp_build=_ncp, nt_layer=_ntl,
                v3_n_nodes=int(os.environ.get("RATIO_V3NODES", "-1")))
            if os.environ.get("RATIO_TUKEY0", "0") == "1":
                sighet_v3._g["tukey_alpha"] = 0.0
            sighet_v3.setup_in_model(holder, A[None, :], zeros)
            eng_v3 = WDMBandLikelihoodEngine(sighet_v3, wdm_set, nchannels=3,
                                             tdi_channel_setup="XYZ")
            print("  [v3] compiled sig_het_v3 column enabled "
                  f"(v3_n_nodes={sighet_v3._g['v3_n_nodes']})")

        eng_c.get_ll(holder, A[None, :], phase_maximize=False, **kw)
        hh_ref_c = float(eng_c.h_h_out[0])
        _, hh_dA = dense_terms(A)
        norm = hh_ref_c / hh_dA
        print(f"\n[COMPARE] {n_cmp} draws; dense-truth norm calibrated at "
              f"ref: h_h_eng={hh_ref_c:.4e} (SNR_ref~{math.sqrt(max(hh_ref_c,0)):.1f})")

        def draw(rng, hot):
            s_ = (dict(lnA=.5, phi0=1.5, ang=.4, sky=.4, f0r=.25,
                       fdot=3e-17) if hot else
                  dict(lnA=.02, phi0=.05, ang=.02, sky=.01, f0r=.02,
                       fdot=1e-18))
            zf = rng.uniform(0.5, 2.0)
            p = A.copy()
            p[0] *= np.exp(np.clip(zf * s_["lnA"] * rng.standard_normal(),
                                   -1.5, 1.5))
            drift = float(np.clip(zf * s_["f0r"] * rng.standard_normal(),
                                  -0.5, 0.5))
            p[1] += drift / (2.0 * np.pi * g["Tobs"])
            p[2] += zf * s_["fdot"] * rng.standard_normal()
            p[4] += zf * s_["phi0"] * rng.standard_normal()
            dio = zf * s_["ang"] * rng.standard_normal()
            dps = zf * s_["ang"] * rng.standard_normal()
            p[5] += dio
            p[6] += dps
            dlam = zf * s_["sky"] * rng.standard_normal()
            dbet = zf * s_["sky"] * rng.standard_normal()
            p[7] += dlam
            p[8] = float(np.clip(p[8] + dbet, -1.3, 1.3))
            angsep = math.hypot(dlam * math.cos(A[8]), p[8] - A[8])
            D_pred = (2.0 * np.pi * p[1] * R_LT * angsep
                      + abs(dio) + abs(dps))
            return p, D_pred

        def policy_n_cmp(D_pred):
            return int(np.clip(
                math.ceil(16.0 * (max(D_pred, 1e-3) / 0.4) ** 0.25), 6, 64))

        def derot_cmp(df0, dfdot):
            def f(tau):
                return 2.0 * np.pi * (df0 * tau + 0.5 * dfdot * tau ** 2)
            return f

        hdr = ("  {:>3s} {:>4s} {:>6s} {:>13s} | {:>13s} {:>8s} | "
               "{:>13s} {:>8s} | {:>13s} {:>8s} {:>4s}").format(
            "#", "pop", "D", "dense delta", "chunked", "rel",
            "sig-het", "rel", "ratio(new)", "rel", "n_r")
        print(hdr)
        errs = {"chunk": [], "sig": [], "new": []}
        pops = []
        for i in range(n_cmp):
            hot = i >= n_cmp // 2
            p, D_pred = draw(rng3, hot)
            dh_d, hh_d = dense_terms(p)
            d_dense = norm * (dh_d - 0.5 * hh_d)
            scale = max(abs(d_dense), 1e-3 * hh_ref_c)
            eng_c.get_ll(holder, p[None, :], phase_maximize=False, **kw)
            d_chunk = float(eng_c.d_h_out[0] - 0.5 * eng_c.h_h_out[0])
            eng_s.get_ll(holder, p[None, :], phase_maximize=False, **kw)
            d_sig = float(eng_s.d_h_out[0] - 0.5 * eng_s.h_h_out[0])
            s_cand, _, _ = slow_series(gen, p, t_dense, N, g,
                                       kf0_pin=kf0_ref)
            r_slow = s_cand / s_ref
            n_ad = policy_n_cmp(D_pred)
            rh, n_used = fit_ratio(r_slow, tau_slow, n_ad, "cubic",
                                   good=good,
                                   derot=derot_cmp(p[1] - A[1],
                                                   p[2] - A[2]))
            d_new = arm_ii_from_rhat(p, rh)
            e_c = abs(d_chunk - d_dense) / scale
            e_s = abs(d_sig - d_dense) / scale
            e_n = abs(d_new - d_dense) / scale
            errs["chunk"].append(e_c)
            errs["sig"].append(e_s)
            errs["new"].append(e_n)
            line_v3 = ""
            if eng_v3 is not None:
                eng_v3.get_ll(holder, p[None, :], phase_maximize=False, **kw)
                d_v3 = float(eng_v3.d_h_out[0] - 0.5 * eng_v3.h_h_out[0])
                e_v3 = abs(d_v3 - d_dense) / scale
                errs.setdefault("v3", []).append(e_v3)
                line_v3 = " | v3 {:>+12.5e} {:>8.1e}".format(d_v3, e_v3)
            pops.append(hot)
            print(("  {:>3d} {:>4s} {:>6.2f} {:>+13.5e} | {:>+13.5e} "
                   "{:>8.1e} | {:>+13.5e} {:>8.1e} | {:>+13.5e} {:>8.1e} "
                   "{:>4d}").format(
                i, "hot" if hot else "cold", D_pred, d_dense,
                d_chunk, e_c, d_sig, e_s, d_new, e_n, n_used) + line_v3)
        print("\n[COMPARE SUMMARY] rel err vs dense truth (median / max)")
        pops = np.array(pops)
        for key, lbl in (("chunk", "chunked het"), ("sig", "current sig-het"),
                         ("new", "new sig-het (ratio)"),
                         ("v3", "COMPILED sig_het_v3")):
            if key not in errs:
                continue
            v = np.array(errs[key])
            print(f"  {lbl:>22s}: cold {np.median(v[~pops]):.2e} / "
                  f"{v[~pops].max():.2e}   hot {np.median(v[pops]):.2e} / "
                  f"{v[pops].max():.2e}")
        return

    # ---- combined-random-displacement STRESS TEST (RATIO_STRESS=1) --------
    # The MCMC-realism check: a few hundred 9-dim joint proposals from
    # cold-chain (posterior-width) and hot-chain (gate-limited) displacement
    # distributions, scored against the within-thread exact build.  Also
    # measures the adaptive-n_r policy and the analytic carrier-difference
    # de-rotation (fit only the residual after removing the known
    # 2*pi*(df0*tau + 0.5*dfdot*tau^2) ramp).
    if os.environ.get("RATIO_STRESS", "0") == "1":
        n_stress = int(os.environ.get("RATIO_STRESS_N", "240"))
        rng2 = np.random.default_rng(42)
        d_ref_scale = abs(thread_exact_pinned(A))
        R_LT = 499.00478   # 1 AU light time [s]

        def derot_fn(df0, dfdot):
            def f(tau):
                return 2.0 * np.pi * (df0 * tau + 0.5 * dfdot * tau ** 2)
            return f

        def policy_n(D_pred):
            # calibrated on the axis sweep: iota+0.5 (D~0.4) needs 16 nodes
            # for ~1e-4; err ~ D * n^-4  =>  n = 16 * (D/0.4)^(1/4).
            return int(np.clip(
                math.ceil(16.0 * (max(D_pred, 1e-3) / 0.4) ** 0.25), 6, 64))

        recs = []
        print(f"\n[STRESS] {n_stress} random 9-dim proposals "
              f"({n_stress//2} cold + {n_stress - n_stress//2} hot); "
              "rel err vs within-thread exact (scale-floored)")
        for i in range(n_stress):
            hot = i >= n_stress // 2
            s_ = (dict(lnA=.5, phi0=1.5, ang=.4, sky=.4, f0r=.25, fdot=3e-17)
                  if hot else
                  dict(lnA=.02, phi0=.05, ang=.02, sky=.01, f0r=.02,
                       fdot=1e-18))
            zf = rng2.uniform(0.5, 2.0)          # stretch-factor structure
            p = A.copy()
            dlnA_s = float(np.clip(zf * s_["lnA"] * rng2.standard_normal(),
                                   -1.5, 1.5))
            p[0] *= np.exp(dlnA_s)
            drift = float(np.clip(zf * s_["f0r"] * rng2.standard_normal(),
                                  -0.5, 0.5))    # carrier rad over Tobs (gate)
            p[1] += drift / (2.0 * np.pi * g["Tobs"])
            p[2] += zf * s_["fdot"] * rng2.standard_normal()
            p[4] += zf * s_["phi0"] * rng2.standard_normal()
            dio = zf * s_["ang"] * rng2.standard_normal()
            dps = zf * s_["ang"] * rng2.standard_normal()
            p[5] += dio
            p[6] += dps
            dlam = zf * s_["sky"] * rng2.standard_normal()
            dbet = zf * s_["sky"] * rng2.standard_normal()
            p[7] += dlam
            p[8] = float(np.clip(p[8] + dbet, -1.3, 1.3))
            angsep = math.hypot(dlam * math.cos(A[8]), p[8] - A[8])
            D_pred = (2.0 * np.pi * p[1] * R_LT * angsep
                      + abs(dio) + abs(dps))

            # within-thread exact (one slow_series eval, reused everywhere)
            s_cand, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
            m_act = m_active_for(p[1], g)
            X_l = X_lin_from_slow(s_cand, g, N)
            c1 = polyphase_py(X_l, kf0_ref, m_act, g, window_full,
                              n_sparse_local)
            r_e, dr_e, _ = ratio_dr(c1, c0_rows_for(m_act), g["stride"],
                                    g["max_r"])
            dh_e, hh_e = fold_py(r_e, dr_e, m_act, sighet)
            d_ex = dh_e - 0.5 * hh_e
            scale = max(abs(d_ex), 1e-3 * d_ref_scale)

            r_slow = s_cand / s_ref
            der = derot_fn(p[1] - A[1], p[2] - A[2])
            row = dict(hot=hot, D=D_pred, dlnA=dlnA_s, drift=drift,
                       angsep=angsep)
            for nr_fix in (8, 16, 32):
                rh, _ = fit_ratio(r_slow, tau_slow, nr_fix, "cubic",
                                  good=good, derot=der)
                row[f"ii_{nr_fix}"] = abs(arm_ii_from_rhat(p, rh) - d_ex) / scale
            n_ad = policy_n(D_pred)
            rh, n_used = fit_ratio(r_slow, tau_slow, n_ad, "cubic",
                                   good=good, derot=der)
            row["n_ad"] = n_used
            row["ii_ad"] = abs(arm_ii_from_rhat(p, rh) - d_ex) / scale
            row["i_ad"] = abs(arm_i_from_rhat(p, rh) - d_ex) / scale
            rh0, _ = fit_ratio(r_slow, tau_slow, n_ad, "cubic",
                               good=good, derot=None)
            row["ii_ad_noderot"] = abs(arm_ii_from_rhat(p, rh0) - d_ex) / scale
            recs.append(row)
            if (i + 1) % 40 == 0:
                print(f"  ... {i+1}/{n_stress}")

        def pctl(key, sel=None):
            v = np.array([r[key] for r in recs
                          if sel is None or r["hot"] == sel])
            return (np.percentile(v, 50), np.percentile(v, 90),
                    np.percentile(v, 99), v.max())

        print("\n[STRESS SUMMARY] rel err percentiles 50/90/99/max")
        for key in ("ii_8", "ii_16", "ii_32", "ii_ad", "i_ad",
                    "ii_ad_noderot"):
            for sel, lbl in ((False, "cold"), (True, "hot ")):
                p50, p90, p99, mx = pctl(key, sel)
                print(f"  {key:>13s} {lbl}: {p50:.2e} {p90:.2e} "
                      f"{p99:.2e} {mx:.2e}")
        n_ads = np.array([r["n_ad"] for r in recs])
        print(f"  adaptive nodes: mean={n_ads.mean():.1f} "
              f"median={np.median(n_ads):.0f} max={n_ads.max()} "
              f"(cold mean={n_ads[:n_stress//2].mean():.1f}, "
              f"hot mean={n_ads[n_stress//2:].mean():.1f})")

        # ---- f0 de-rotation ladder: the second-level heterodyne ----------
        # Beyond-gate carrier drifts, fit at n_r=16: with the analytic ramp
        # removed, arm (i) stays exact at ANY drift (the fit sees only the
        # residual); arm (ii)'s wavelet-support floor grows with drift and
        # is NOT repaired by de-rotation (it is a leakage term, not a fit
        # term) -- this measures how far the dphase gate could be relaxed
        # per rung.
        print("\n[f0-derot ladder] carrier drift (rad over Tobs), n_r=16, "
              "rel err arm(i)/arm(ii), derot ON vs OFF")
        for drift in (0.5, 2.0, 8.0, 32.0):
            p = A.copy()
            p[1] += drift / (2.0 * np.pi * g["Tobs"])
            s_cand, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
            m_act = m_active_for(p[1], g)
            X_l = X_lin_from_slow(s_cand, g, N)
            c1 = polyphase_py(X_l, kf0_ref, m_act, g, window_full,
                              n_sparse_local)
            r_e, dr_e, _ = ratio_dr(c1, c0_rows_for(m_act), g["stride"],
                                    g["max_r"])
            dh_e, hh_e = fold_py(r_e, dr_e, m_act, sighet)
            d_ex = dh_e - 0.5 * hh_e
            scale = max(abs(d_ex), 1e-3 * d_ref_scale)
            r_slow = s_cand / s_ref
            der = derot_fn(p[1] - A[1], 0.0)
            out = {}
            for lbl, dr_use in (("on", der), ("off", None)):
                rh, _ = fit_ratio(r_slow, tau_slow, 16, "cubic",
                                  good=good, derot=dr_use)
                out[f"i_{lbl}"] = abs(arm_i_from_rhat(p, rh) - d_ex) / scale
                out[f"ii_{lbl}"] = abs(arm_ii_from_rhat(p, rh) - d_ex) / scale
            print(f"  drift={drift:5.1f}: i on={out['i_on']:.2e} "
                  f"off={out['i_off']:.2e} | ii on={out['ii_on']:.2e} "
                  f"off={out['ii_off']:.2e}")

        np.savez(os.path.join(out_dir, "ratio_stress_results.npz"),
                 recs=np.array(recs, dtype=object))
        print(f"\n[npz] {out_dir}/ratio_stress_results.npz")
        return

    # ---- displacement sweep -----------------------------------------------
    if os.environ.get("RATIO_SWEEP", "1") != "1":
        return
    directions = [
        ("lnA",  0, "mul",  [0.1, 0.5, 1.0, 1.5]),
        ("phi0", PHYS_IDX_PHI0, "add", [0.1, 0.5, 1.0]),
        ("iota", 5, "add",  [0.05, 0.2, 0.5]),
        ("psi",  6, "add",  [0.2, 0.5, 1.0]),
        ("f0",   1, "addf", [1e-4, 3e-4]),
        ("fdot", 2, "add",  [1e-17, 1e-16]),
        ("lam",  7, "add",  [0.01, 0.05, 0.2, 1.0]),
        ("beta", 8, "add",  [0.01, 0.05, 0.2]),
    ]
    _dirs = os.environ.get("RATIO_DIRS")
    if _dirs:
        keep = set(_dirs.split(","))
        directions = [d for d in directions if d[0] in keep]
    rows = []
    print("\n[SWEEP] |delta_arm - delta_exact| (same thread, pinned carrier); "
          "delta = d_h - 0.5 h_h")
    hdr = "  {:>10s} {:>8s} {:>12s} {:>9s} {:>9s}".format(
        "dir", "step", "|delta_ex|", "dlnA_rng", "dphi_rng")
    for n_r in nr_list:
        hdr += f" | i:c{n_r:<3d} ii:c{n_r:<3d}"
    print(hdr)
    for name, idx, mode, steps in directions:
        for s_val in steps:
            p = A.copy()
            if mode == "mul":
                p[idx] *= np.exp(s_val)
            elif mode == "addf":
                p[idx] += s_val * layer_df
            else:
                p[idx] += s_val
            d_ex = thread_exact_pinned(p)
            s_cand, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
            r_slow = s_cand / s_ref
            # smoothness diagnostics on null-free samples only (division at
            # reference-envelope nulls is numerically meaningless there)
            dlnA_rng = np.ptp(np.log(np.abs(r_slow[:, good])))
            dphi_rng = np.ptp(np.unwrap(np.angle(r_slow), axis=-1)[:, good])
            scale = max(abs(d_ex), 1e-300)
            line = "  {:>10s} {:>8g} {:>12.4e} {:>9.2e} {:>9.2e}".format(
                name, s_val, abs(d_ex), dlnA_rng, dphi_rng)
            errs = {}
            for n_r in nr_list:
                d_i, d_ii, _ = arms(p, n_r, "cubic")
                errs[("cubic", n_r)] = (abs(d_i - d_ex), abs(d_ii - d_ex))
                d_i_l, d_ii_l, _ = arms(p, n_r, "linear")
                errs[("linear", n_r)] = (abs(d_i_l - d_ex), abs(d_ii_l - d_ex))
                line += " | {:.1e} {:.1e}".format(
                    errs[("cubic", n_r)][0] / scale,
                    errs[("cubic", n_r)][1] / scale)
            print(line)
            rows.append(dict(direction=name, step=s_val, delta_exact=d_ex,
                             dlnA_rng=dlnA_rng, dphi_rng=dphi_rng,
                             errs={f"{k[0]}_{k[1]}": v
                                   for k, v in errs.items()}))

    # linear-vs-cubic summary at genuinely time-varying displacements
    for lbl, build in [("phi0+0.5", lambda q: q.__setitem__(PHYS_IDX_PHI0,
                                                            A[4] + 0.5)),
                       ("lam+0.2", lambda q: q.__setitem__(7, A[7] + 0.2)),
                       ("combo lnA+1,iota+0.3,lam+0.2",
                        lambda q: (q.__setitem__(0, A[0] * np.e),
                                   q.__setitem__(5, A[5] + 0.3),
                                   q.__setitem__(7, A[7] + 0.2)))]:
        p = A.copy()
        build(p)
        d_ex = thread_exact_pinned(p)
        scale = max(abs(d_ex), 1e-300)
        print(f"\n[linear vs cubic] {lbl} REL arm errors by node count")
        for n_r in nr_list:
            ci, cii, nu = arms(p, n_r, "cubic")
            li, lii, _ = arms(p, n_r, "linear")
            print(f"  n_r={n_r:3d} (used {nu:3d}): "
                  f"cubic i={abs(ci-d_ex)/scale:.2e} "
                  f"ii={abs(cii-d_ex)/scale:.2e} | "
                  f"linear i={abs(li-d_ex)/scale:.2e} "
                  f"ii={abs(lii-d_ex)/scale:.2e}")

    # ---- figures + npz for the artifact -----------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = A.copy()
        p[0] *= np.exp(1.0)
        p[5] += 0.3
        p[7] += 0.2
        s_cand, _, _ = slow_series(gen, p, t_dense, N, g, kf0_pin=kf0_ref)
        r_slow = s_cand / s_ref
        r_hat8, _ = fit_ratio(r_slow, tau_slow, 8, "cubic")
        rh = r_hat8(tau_slow)
        td_days = tau_slow / 86400.0
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
        ch_names = "XYZ"
        for c in range(3):
            axes[0, 0].plot(td_days, np.log(np.abs(r_slow[c])),
                            lw=1.5, label=f"{ch_names[c]} true")
            axes[0, 0].plot(td_days, np.log(np.abs(rh[c])), "k--", lw=0.8)
            axes[1, 0].plot(td_days, np.unwrap(np.angle(r_slow[c])), lw=1.5)
            axes[1, 0].plot(td_days, np.unwrap(np.angle(rh[c])), "k--", lw=0.8)
            axes[0, 1].semilogy(td_days,
                                np.abs(rh[c] - r_slow[c]) + 1e-18, lw=1.0)
            axes[1, 1].semilogy(td_days, np.abs(s_ref[c]) + 1e-30, lw=1.0)
        axes[0, 0].set_ylabel("dlnA(t)")
        axes[1, 0].set_ylabel("dphi(t) [rad]")
        axes[1, 0].set_xlabel("t [days]")
        axes[0, 1].set_ylabel("|r_hat - r| (8 nodes)")
        axes[1, 1].set_ylabel("|h_ref slow|")
        axes[1, 1].set_xlabel("t [days]")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("ratio is SLOW: lnA+1, iota+0.3, lam+0.2")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ratio_smoothness.png"), dpi=120)
        print(f"\n[plot] {out_dir}/ratio_smoothness.png")

        np.savez(os.path.join(out_dir, "ratio_proto_results.npz"),
                 rows=np.array(rows, dtype=object),
                 grid=np.array([Nf, Nt, dt, g["Tobs"], N, g["nt_layer"]]),
                 gate_f=worst_f, gate_x=worst_x)
        print(f"[npz]  {out_dir}/ratio_proto_results.npz")
    except Exception as exc:  # plots are best-effort
        print(f"[plot] skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
