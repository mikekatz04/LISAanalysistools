"""GBSignalHetComputations -- V2 polyphase signal-heterodyne GB likelihood,
with ALL heterodyne-coefficient setup done inside ``__init__`` (mirroring the
BBHx HeterodynedLikelihood pattern: data + reference go in, the heterodyne
coefficients are precomputed under the hood, then ``get_ll(params)`` is a clean
call into the existing ``gb_signal_het_get_ll_in_kernel`` kernel).

Refactored from the verified ``compare_signalhet_vs_chunked_mcmc.py:build_pack``
signal-het path -- no new likelihood logic, the kernel call is reused verbatim.
Lives here next to its v2 helpers (``GBSparseComplexWDMGen``, ``python_bin_fold``);
promotion to ``gbgpu/gbsignalhetcomputations.py`` is the dedicated V2-port step.
"""
from __future__ import annotations

import importlib
import math
import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_cpp_validate import python_bin_fold_real


def _resolve_backend(name):
    name = {"gpu": "cuda12x", "cuda": "cuda12x"}.get(name.lower().strip(), name.lower().strip())
    if name == "cpu":
        return "cpu", "gbgpu_backend_cpu", False
    if name in ("cuda11x", "cuda12x", "cuda13x"):
        return name, f"gbgpu_backend_{name}", True
    raise ValueError(f"Unknown backend {name!r}.")


def tukey_taper_layers(Nt, tukey_alpha):
    """Width, in WDM time-LAYERS, of ONE end of a ``Tukey(alpha)`` taper applied
    to the full ``Nf*Nt`` time series.

    A ``scipy.signal.windows.tukey(N, alpha)`` cosine-taper covers ``alpha/2`` of
    the ``N`` samples at each end; one WDM time-layer spans ``Nf`` samples, so the
    taper is ``alpha/2 * (Nf*Nt) / Nf = alpha*Nt/2`` layers. Automatically decided
    from the window (``alpha``) and grid (``Nt``) -- no hand-tuning.
    """
    return int(math.ceil(0.5 * float(tukey_alpha) * int(Nt)))


def recommended_edge_cut(Nt, tukey_alpha, method, margin=8):
    """Recommended WDM time-edge cut (layers) for a fast-likelihood ``method``,
    derived automatically from the Tukey taper (:func:`tukey_taper_layers`).

    The two GB fast-likelihoods want OPPOSITE edge regions w.r.t. the global
    taper:

    * ``method="chunked"`` -- the chunked-het cannot reproduce the global taper
      (it applies only a per-chunk stitching window), so the active region must
      EXCLUDE the taper: ``max(20, taper + margin)``.
    * ``method="sighet"`` -- SINCE the real-WDM projection fix (get_ll project_real
      =1) the sig-het has NO complex-vs-real bin-fold floor, so it no longer needs
      the old small-edge-cut workaround that masked that floor. It now shares the
      chunked rule (``EC >= taper``) so its active region MATCHES the dense/chunked
      one -- which is what makes its absolute logL track the dense (log-like chain:
      sig-het - dense ~2.5e-8 at injection, ~2e-2 at logL -500; a mismatched edge-
      cut instead leaves a ~13%-of-logL bias from different time coverage). The
      per-method edge split has collapsed; both return the same value.
    """
    taper = tukey_taper_layers(Nt, tukey_alpha)
    m = str(method).lower().replace("-", "_")
    if m in ("chunked", "chunk", "chunked_het",
             "sighet", "signal_het", "signalhet", "sig_het"):
        return max(20, taper + int(margin))
    raise ValueError(f"method must be 'chunked' or 'sighet', got {method!r}")


class GBSignalHetComputations:
    """Signal-het GB likelihood. ``__init__`` precomputes the heterodyne
    reference ``c0`` and bin-fold coefficients (A0/A1/B0/B1) from the data +
    reference params; ``get_ll(params)`` evaluates logL for candidate params.

    Args:
        data_td (np.ndarray): ``(3, Nf*Nt)`` time-domain TDI data (the band is
            selected by ``min_freq``/``max_freq`` in the WDM transform).
        ref_params (np.ndarray): length-9 heterodyne reference
            ``(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)`` -- the carrier
            the polyphase heterodyne is taken about (e.g. the catalogue params).
        Nf, Nt, dt, t0 (int/float): WDM grid + data sampling.
        t_ref (float): GB phase reference epoch.
        orbits, tdi_config: response orbits (e.g. ``L1Orbits``) + TDI config.
        min_freq, max_freq (float): active WDM band [Hz].
        edge_cut (int or None): WDM time-edge cut (layers). ``None`` (default)
            auto-decides from the window via
            :func:`recommended_edge_cut(Nt, tukey_alpha, "sighet")` -- a small cut
            BELOW the Tukey taper so the taper sits inside the active region and
            down-weights the noisy bin-fold edges (lowers the floor). Pass an int
            to override. The resolved value is stored on ``self.edge_cut`` and the
            taper width on ``self.taper_layers``.
        sens_model, nt_layer, n_sparse_fd, m_active_half_width, max_r,
        tukey_alpha, force_backend: as in ``build_pack`` (defaults match).
    """

    def __init__(self, data_td, ref_params, *, Nf, Nt, dt, t0, t_ref,
                 orbits, tdi_config, min_freq, max_freq, sens_model="scirdv1",
                 edge_cut=None, nt_layer=64, n_sparse_fd=1024, m_active_half_width=2,
                 max_r=5.0, tukey_alpha=0.05, force_backend="cpu"):
        # tukey_alpha in [0.01, 0.05]: the SAME Tukey taper is applied to the
        # data WDM window (below) AND to the FD-heterodyne sparse window via the
        # kernel's TUKEY_ALPHA arg (in get_ll) -- they must mirror, per the GB
        # chunked/sig-het notes (a mismatch shifts <d|h>/<h|h> by ~logL units).
        _, module_name, is_gpu = _resolve_backend(force_backend)
        _be = importlib.import_module(f"{module_name}.cgbgpu")
        self.cpp = _be.GBComputationGroupWrapGPU() if is_gpu else _be.GBComputationGroupWrapCPU()

        # Edge-cut auto-decided from the window: the Tukey taper is
        # tukey_taper_layers(Nt, tukey_alpha) WDM layers, and the sig-het floor is
        # smallest when that taper sits INSIDE the active region (it down-weights
        # the noisy bin-fold edges). edge_cut=None -> recommended_edge_cut(...,
        # "sighet") (a small cut below the taper). Pass an int to override.
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

        # dense TD generator (its GPU-side wrap drives the per-call template build)
        t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
        gb_gen = GBTDIonTheFly(t_tdi, Tobs, t_ref, 1.0 / dt, 1,
                               tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                               force_backend=force_backend)
        self.tdi_wrap = gb_gen.wave_gen

        def real_td_cb(p):
            amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
            sp = gb_gen(np.array([amp]), np.array([f0]), np.array([fdot]),
                        np.array([fddot]), np.array([phi0]), np.array([inc]),
                        np.array([psi]), np.array([lam]), np.array([beta]),
                        convert_to_ra_dec=False, return_spline=True)
            return np.asarray(sp.eval_tdi(t_arr))[0]

        # --- data on the WDM band (real + complex), + d_d --------------------
        data_real = TDSignal(data_td, settings=td_set).transform(wdm_set_real, window=window)
        data_complex = np.asarray(
            TDSignal(data_td, settings=td_set).transform(wdm_set_complex, window=window).arr)
        sens_real = XYZ2SensitivityMatrix(wdm_set_real, model=sens_model)
        self.analysis = AnalysisContainer(data_real, sens_real)
        self.d_d = float(np.real(self.analysis.inner_product()))

        # --- heterodyne reference c0 at ref_params ---------------------------
        ref_params = np.asarray(ref_params, dtype=float).reshape(9)
        td_ref = real_td_cb(ref_params)
        c0_dense = np.asarray(
            TDSignal(td_ref, settings=td_set).transform(wdm_set_complex, window=window).arr)

        # --- sparse grid + bin-fold COEFFICIENTS (built here, under the hood) -
        ind_min_t = int(wdm_set_real.ind_min_t); ind_min_f = int(wdm_set_real.ind_min_f)
        Nt_active = int(wdm_set_real.Nt_active)
        Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
        sparse_gen = GBSparseComplexWDMGen(
            real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
            data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
            Nt_layer=nt_layer, m_active_half_width=m_active_half_width)
        stride = sparse_gen.stride; N_sparse_t = sparse_gen.N_sparse_t
        n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
        window_full = sparse_gen.window_full.astype(np.float64)
        c0_sparse = c0_dense[:, :, n_sparse_local]
        invC_complex = np.asarray(XYZ2SensitivityMatrix(wdm_set_complex, model=sens_model).invC)
        # REAL-projection coefficients (match the REAL WDM likelihood exactly).
        # <d|h>: A0/A1 are the REPACKED complex coeffs (kernel's Re(A0*r) -> real,
        # no extra storage). <h|h>: B0/B1 (conj) + B0nc/B1nc (nonconj) so the
        # kernel forms 0.5*Re(conj + nonconj) = the real projection (project_real=1).
        A0, A1, B0, B1, B0nc, B1nc = python_bin_fold_real(
            data_complex, c0_dense, invC_complex, n_sparse_local, stride, Nt_active, tdi_type="XYZ")

        # --- owned backend buffers + scalars for get_ll ----------------------
        self.c0_sparse_all = c0_sparse[None, ...].copy()
        self.A0_all = A0[None].copy(); self.A1_all = A1[None].copy()
        self.B0_all = B0[None].copy(); self.B1_all = B1[None].copy()
        self.B0nc_all = B0nc[None].copy(); self.B1nc_all = B1nc[None].copy()
        self.window_full = window_full; self.n_sparse_local = n_sparse_local
        self.params_ref_all = ref_params.reshape(1, 9).copy()

        # KEEP-ALIVES: the signal-het kernel reads C++ pointers held inside
        # ``self.tdi_wrap`` (gb_gen.wave_gen) -- the orbits/TDI-config/spline
        # state lives on the Python objects below. If they are GC'd when
        # __init__ returns, those pointers dangle and the FIRST kernel call
        # silently returns d_h=h_h=0 (or SIGSEGVs nondeterministically).
        # Mirrors build_pack's explicit keep-alive list in
        # compare_signalhet_vs_chunked_mcmc.py.
        self._keep_alive = dict(
            gb_gen=gb_gen, orbits=orbits, tdi_config=tdi_config,
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

    def get_ll(self, params):
        """logL for candidate ``params`` (length-9 vector or ``(N,9)``)."""
        x = np.asarray(params, dtype=float)
        x = x[None, :] if x.ndim == 1 else x
        N = x.shape[0]
        d_h = np.zeros(N, dtype=np.float64); h_h = np.zeros(N, dtype=np.float64)
        g = self._g
        self.cpp.gb_signal_het_get_ll_in_kernel(
            self.tdi_wrap, d_h, h_h, self.c0_sparse_all,
            self.A0_all, self.A1_all, self.B0_all, self.B1_all,
            self.B0nc_all, self.B1nc_all,
            self.window_full, self.n_sparse_local,
            np.ascontiguousarray(x), self.params_ref_all,
            np.zeros(N, dtype=np.int32),
            N, 1, 9, 1, 2,
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
