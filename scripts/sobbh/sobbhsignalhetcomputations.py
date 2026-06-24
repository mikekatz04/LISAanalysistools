"""SOBBHSignalHetComputations -- V2 polyphase signal-heterodyne SOBBH
likelihood. SOBBH duplicate of
``gb_chunked_het/gbsignalhetcomputations.py::GBSignalHetComputations``
(2026-06-18); the only differences are the source class (SOBBH PN
TDI-on-the-fly instead of GB UCB), the 11-element parameter vector, and the
backend routing (``bbhx_backend_*.cbbhx`` /
``SOBBHComputationGroupWrap`` / ``sobbh_signal_het_get_ll_in_kernel``).

All the heterodyne-coefficient setup happens inside ``__init__`` (data +
reference go in, the bin-fold coefficients are precomputed under the hood),
then ``get_ll(params)`` is a clean call into the C++
``sobbh_signal_het_get_ll_in_kernel`` kernel.

The polyphase + bin-fold helpers (:class:`GBSparseComplexWDMGen`,
:func:`python_bin_fold_real`) are SOURCE-AGNOSTIC -- they take a TD callable
and plain arrays -- so we reuse the GB versions verbatim from
``../gb_chunked_het/`` rather than duplicating them.

SOBBH 11-parameter order (matches lisatools.response.tdionfly.SOBBHTDIonTheFly):
    (m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)
carrier f0 = params[5] = f_low. There is no explicit fdot parameter; the
get_ll path ignores fdot_idx, so we pass fdot_idx=5 (harmless).
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

# Reuse the source-agnostic v2 helpers from the GB scripts dir.
_GB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "gb_chunked_het")
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen          # noqa: E402
from gb_signal_het_cpp_validate import python_bin_fold_real     # noqa: E402

NPARAMS_SOBBH = 11
F0_IDX_SOBBH = 5    # params[5] = f_low
# Must match SOBBH_SIGHET_MAX_M in BBHx/src/bbhx/cutils/sobbh_tdi_on_the_fly.cu
# (caps the active-m-band: M = 2*m_active_half_width+1 <= SOBBH_SIGHET_MAX_M).
SOBBH_SIGHET_MAX_M = 600


def _resolve_backend(name):
    name = {"gpu": "cuda12x", "cuda": "cuda12x"}.get(name.lower().strip(), name.lower().strip())
    if name == "cpu":
        return "cpu", "bbhx_backend_cpu", False
    if name in ("cuda11x", "cuda12x", "cuda13x"):
        return name, f"bbhx_backend_{name}", True
    raise ValueError(f"Unknown backend {name!r}.")


def tukey_taper_layers(Nt, tukey_alpha):
    """Width, in WDM time-LAYERS, of ONE end of a ``Tukey(alpha)`` taper.

    Identical to the GB helper -- the taper width is a property of the grid +
    window, not the source class.
    """
    return int(math.ceil(0.5 * float(tukey_alpha) * int(Nt)))


def recommended_edge_cut(Nt, tukey_alpha, method, margin=8):
    """Recommended WDM time-edge cut (layers), auto-derived from the Tukey
    taper. Same rule as the GB version (both methods collapse to
    ``max(20, taper + margin)`` since the real-WDM projection fix)."""
    taper = tukey_taper_layers(Nt, tukey_alpha)
    m = str(method).lower().replace("-", "_")
    if m in ("chunked", "chunk", "chunked_het",
             "sighet", "signal_het", "signalhet", "sig_het"):
        return max(20, taper + int(margin))
    raise ValueError(f"method must be 'chunked' or 'sighet', got {method!r}")


class SOBBHSignalHetComputations:
    """Signal-het SOBBH likelihood. ``__init__`` precomputes the heterodyne
    reference ``c0`` and bin-fold coefficients (A0/A1/B0/B1) from the data +
    reference params; ``get_ll(params)`` evaluates logL for candidate params.

    Args:
        data_td (np.ndarray): ``(3, Nf*Nt)`` time-domain TDI data.
        ref_params (np.ndarray): length-11 heterodyne reference
            ``(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)``.
        Nf, Nt, dt, t0 (int/float): WDM grid + data sampling.
        t_ref (float): SOBBH phase reference epoch.
        orbits, tdi_config: response orbits + TDI config.
        min_freq, max_freq (float): active WDM band [Hz].
        edge_cut, sens_model, nt_layer, n_sparse_fd, m_active_half_width,
        max_r, tukey_alpha, force_backend: as in the GB version.
    """

    def __init__(self, data_td, ref_params, *, Nf, Nt, dt, t0, t_ref,
                 orbits, tdi_config, min_freq, max_freq, sens_model="scirdv1",
                 edge_cut=None, nt_layer=64, n_sparse_fd=1024, m_active_half_width=2,
                 max_r=5.0, tukey_alpha=0.05, force_backend="cpu"):
        _, module_name, is_gpu = _resolve_backend(force_backend)
        _be = importlib.import_module(f"{module_name}.cbbhx")
        self.cpp = _be.SOBBHComputationGroupWrapGPU() if is_gpu else _be.SOBBHComputationGroupWrapCPU()

        self.taper_layers = tukey_taper_layers(Nt, tukey_alpha)
        if edge_cut is None:
            edge_cut = recommended_edge_cut(Nt, tukey_alpha, "sighet")
        self.edge_cut = int(edge_cut)

        Nobs = Nf * Nt
        Tobs = Nt * Nf * dt
        t_arr = np.arange(Nobs) * dt + t0
        if isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=force_backend)
        td_set = TDSettings(Nobs, dt, t0=t0, force_backend=force_backend)
        window = (_tukey(Nobs, alpha=tukey_alpha).astype(float) if tukey_alpha > 0
                  else np.ones(Nobs))

        wdm_kw = dict(t0=t0, min_freq=min_freq, max_freq=max_freq,
                      min_time=self.edge_cut * Nf * dt, max_time=(Nt - self.edge_cut) * Nf * dt,
                      force_backend=force_backend)
        wdm_set_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
        wdm_set_complex = WDMSettings(Nf, Nt, dt, is_complex=True, **wdm_kw)

        # dense TD generator (its wrap drives the per-call in-kernel FD build).
        t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
        sobbh_gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_ref, 1.0 / dt, 1,
                                     tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                                     force_backend=force_backend)
        self.tdi_wrap = sobbh_gen.wave_gen

        def real_td_cb(p):
            m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta = p
            sp = sobbh_gen(np.array([m1]), np.array([m2]), np.array([s1]),
                           np.array([s2]), np.array([distance]), np.array([f_low]),
                           np.array([phi_c]), np.array([inc]), np.array([psi]),
                           np.array([lam]), np.array([beta]),
                           convert_to_ra_dec=False, return_spline=True)
            return np.asarray(sp.eval_tdi(t_arr))[0]

        # --- data on the WDM band (real + complex), + d_d ---
        data_real = TDSignal(data_td, settings=td_set).transform(wdm_set_real, window=window)
        data_complex = np.asarray(
            TDSignal(data_td, settings=td_set).transform(wdm_set_complex, window=window).arr)
        sens_real = XYZ2SensitivityMatrix(wdm_set_real, model=sens_model)
        self.analysis = AnalysisContainer(data_real, sens_real)
        self.d_d = float(np.real(self.analysis.inner_product()))

        # --- heterodyne reference c0 at ref_params ---
        ref_params = np.asarray(ref_params, dtype=float).reshape(NPARAMS_SOBBH)
        td_ref = real_td_cb(ref_params)
        c0_dense = np.asarray(
            TDSignal(td_ref, settings=td_set).transform(wdm_set_complex, window=window).arr)

        # --- sparse grid + bin-fold COEFFICIENTS (source-agnostic helpers) ---
        ind_min_t = int(wdm_set_real.ind_min_t); ind_min_f = int(wdm_set_real.ind_min_f)
        Nt_active = int(wdm_set_real.Nt_active)
        Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)

        # MOVING-WINDOW band auto-sizing (2026-06-19): a SOBBH chirps UP across
        # many WDM layers over the observation, so the active band (centred on
        # f_low per candidate) must be WIDE enough to cover the carrier sweep.
        # Measure the reference's carrier track from |c0| and set the half-width
        # to cover (top layer - f_low layer) + margin. Off-track layers cost ~0
        # (c0 safe-divide floor). The band is the reference-guided "tube"; only
        # its frequency width is set here (time width = all sparse bins).
        # c0_dense is already the active-band array (3, Nf_active, Nt_active).
        ref_power = (np.abs(c0_dense) ** 2).sum(axis=0)            # (Nf_active, Nt_active)
        m_track = np.argmax(ref_power, axis=0)                     # local layer per active time
        m_floor_low = int(np.floor(float(ref_params[5]) / wdm_set_real.layer_df)) - ind_min_f
        span_up = int(m_track.max()) - m_floor_low
        span_dn = m_floor_low - int(m_track.min())
        margin = 5
        auto_half = max(int(span_up), int(span_dn)) + margin
        cap = SOBBH_SIGHET_MAX_M // 2 - 1
        m_active_half_width = int(min(cap, max(m_active_half_width, auto_half)))
        self.chirp_layers = int(m_track.max() - m_track.min())
        self.m_active_half_width = m_active_half_width
        print(f"[sighet] chirp spans {self.chirp_layers} layers; "
              f"m_active_half_width -> {m_active_half_width} "
              f"(M={2*m_active_half_width+1} band layers)", flush=True)

        sparse_gen = GBSparseComplexWDMGen(
            real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
            data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
            Nt_layer=nt_layer, m_active_half_width=m_active_half_width)
        stride = sparse_gen.stride; N_sparse_t = sparse_gen.N_sparse_t
        n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
        window_full = sparse_gen.window_full.astype(np.float64)
        c0_sparse = c0_dense[:, :, n_sparse_local]
        invC_complex = np.asarray(XYZ2SensitivityMatrix(wdm_set_complex, model=sens_model).invC)
        A0, A1, B0, B1, B0nc, B1nc = python_bin_fold_real(
            data_complex, c0_dense, invC_complex, n_sparse_local, stride, Nt_active, tdi_type="XYZ")

        # --- owned backend buffers + scalars for get_ll ---
        self.c0_sparse_all = c0_sparse[None, ...].copy()
        self.A0_all = A0[None].copy(); self.A1_all = A1[None].copy()
        self.B0_all = B0[None].copy(); self.B1_all = B1[None].copy()
        self.B0nc_all = B0nc[None].copy(); self.B1nc_all = B1nc[None].copy()
        self.window_full = window_full; self.n_sparse_local = n_sparse_local
        self.params_ref_all = ref_params.reshape(1, NPARAMS_SOBBH).copy()

        # KEEP-ALIVES: the kernel reads C++ pointers held inside ``self.tdi_wrap``
        # (sobbh_gen.wave_gen). If the Python objects below are GC'd those
        # pointers dangle and the first kernel call returns d_h=h_h=0 (or
        # SIGSEGVs). Mirror of the GB keep-alive fix (2026-06-16).
        self._keep_alive = dict(
            sobbh_gen=sobbh_gen, orbits=orbits, tdi_config=tdi_config,
            td_set=td_set, sparse_gen=sparse_gen,
            wdm_set_real=wdm_set_real, wdm_set_complex=wdm_set_complex,
            window=window, real_td_cb=real_td_cb,
        )
        self._g = dict(Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
                       nt_layer=nt_layer, N_sparse_t=N_sparse_t, stride=stride,
                       ind_min_t=ind_min_t, ind_min_f=ind_min_f, layer_df=wdm_set_real.layer_df,
                       dt=dt, Tobs=Tobs, t0=t0, n_sparse_fd=n_sparse_fd,
                       tukey_alpha=tukey_alpha, max_r=max_r,
                       m_half=int(m_active_half_width))
        self.nchannels = 3
        self.report_shared_mem()

    def report_shared_mem(self):
        """GPU memory budget monitor for the signal-het kernels.

        Two distinct pools (important for the moving-window wide band):

        * SHARED memory -- used ONLY by the FD generator (sobbh_run_fd_wave_tdi),
          sized by N_sparse_fd (NOT the band width M). This is the GPU
          per-block shared budget; it must fit under the device opt-in limit
          (A100 ~164 KB, H100 ~228 KB; >48 KB needs cudaFuncSetAttribute, which
          the wrap sets). It is INDEPENDENT of how wide the moving-window band
          gets -- that is the whole point of keeping the consumer buffers in
          global scratch.
        * GLOBAL scratch -- the consumer's per-binary fold/c1/r/dr buffers,
          which GROW with the band M = 2*m_active_half_width+1. cudaMalloc'd by
          the launcher; NOT shared, so the wide band never pressures shared mem.
        """
        g = self._g
        N = int(g["n_sparse_fd"]); nch = self.nchannels
        # FD-gen shared bytes = get_sobbh_fd_buffer_size(N, nch). get_tdi scratch
        # recovered from the wrap: get_sobbh_buffer_size(N) = N*8 + get_tdi(N).
        N_PARAMS_MAX = 20
        get_tdi = int(self.tdi_wrap.get_buffer_size(N)) - 8 * N
        fd_shared = (N_PARAMS_MAX * 8 + N * 8 + nch * N * 16
                     + 2 * nch * N * 8 + N * 8 + get_tdi)
        M = 2 * int(g["m_half"]) + 1
        Ntl = int(g["nt_layer"]); Nst = int(g["N_sparse_t"])
        per_bin_global = (nch * M * Ntl + 3 * nch * M * Nst) * 16  # fold + c1+r+dr cmplx
        self.fd_shared_bytes = int(fd_shared)
        self.consumer_global_per_bin_bytes = int(per_bin_global)
        a100, h100 = 164 * 1024, 228 * 1024
        flag = ("OK<48KB" if fd_shared <= 48 * 1024
                else f"opt-in (<A100 {a100//1024}KB: {'OK' if fd_shared<=a100 else 'OVER'})")
        print(f"[shared-mem] FD-gen shared = {fd_shared/1024:.1f} KB "
              f"(N_sparse_fd={N}) -> {flag}; band-INDEPENDENT", flush=True)
        print(f"[global-mem] consumer scratch = {per_bin_global/1024:.1f} KB/binary "
              f"(M={M} band layers) -> grows with band, NOT shared", flush=True)
        return self.fd_shared_bytes

    def get_ll(self, params):
        """logL for candidate ``params`` (length-11 vector or ``(N,11)``)."""
        x = np.asarray(params, dtype=float)
        x = x[None, :] if x.ndim == 1 else x
        N = x.shape[0]
        d_h = np.zeros(N, dtype=np.float64); h_h = np.zeros(N, dtype=np.float64)
        g = self._g
        self.cpp.sobbh_signal_het_get_ll_in_kernel(
            self.tdi_wrap, d_h, h_h, self.c0_sparse_all,
            self.A0_all, self.A1_all, self.B0_all, self.B1_all,
            self.B0nc_all, self.B1nc_all,
            self.window_full, self.n_sparse_local,
            np.ascontiguousarray(x), self.params_ref_all,
            np.zeros(N, dtype=np.int32),
            N, 1, NPARAMS_SOBBH, F0_IDX_SOBBH, F0_IDX_SOBBH,  # fdot_idx unused in get_ll
            g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["m_half"],
            g["layer_df"], g["dt"], g["Tobs"], g["t0"],
            3, 0, g["n_sparse_fd"],
            g["tukey_alpha"], g["max_r"], 1,  # project_real=1
        )
        self.last_d_h = np.asarray(d_h).copy()
        self.last_h_h = np.asarray(h_h).copy()
        return -0.5 * self.d_d + np.asarray(d_h) - 0.5 * np.asarray(h_h)
