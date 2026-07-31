#!/usr/bin/env python
"""Compare signal-het vs chunked-het MCMC posteriors on the SAME binary.

Both runs share, exactly:
  * injection params, TD signal, WDM data, sensitivity matrix
  * priors, periodic container, sampled-basis transform
  * RNG seed -- identical starting walker cloud
  * sampler config: stretch move, parallel tempering, nsteps, nwalkers, ntemps
  * <d|d> constant in the returned logL (we add it to chunked-het so the
    two paths are directly comparable in absolute logL, not just shape)

The ONLY thing that differs is the likelihood kernel:

  signal-het   : GBComputationGroupWrap.gb_signal_het_get_ll_in_kernel
                 (v2 polyphase + bin-fold + de-rotate + linear-interp)
  chunked-het  : GBWDMComputations.get_ll_wdm
                 (gb_wdm_het_get_ll: per-chunk heterodyne + WDM transform)

At convergence, the two posteriors should be identical up to MC noise
since they target the same Gaussian likelihood on the same injection.

Saves:
  * mcmc_compare_signalhet.h5  -- signal-het chain
  * mcmc_compare_chunked.h5    -- chunked-het chain
  * corner_compare_signalhet_chunked.png -- overlay corner (cold tail)
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import get_array_module

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from gbgpu.gbcomps import GBWDMComputations

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer, TransformContainer
from eryn.backends import HDFBackend

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_cpp_validate import python_bin_fold


# -------- chunked-het wdm_holder shim (mirrors gb_chunked_test_script.py) --
class _FullGridWDMHolder:
    """Minimal duck-type for GBWDMComputations.get_ll_wdm's wdm_holder arg.

    Copies the data and invC into contiguous flat buffers OWNED by the
    holder. This is important when the holder is built inside a function
    and returned/stored elsewhere -- a raw ``.ravel()`` view into a
    caller-local array would dangle once that function's frame goes away
    (observed: chunked-het kernel SIGSEGVs on first call).
    """
    def __init__(self, data_full, invC_diag_full):
        xp = get_array_module(data_full)
        self.linear_data_arr = [xp.ascontiguousarray(data_full).ravel().copy()]
        self.linear_psd_arr  = [xp.ascontiguousarray(invC_diag_full).ravel().copy()]
    def __len__(self): return 1


def _resolve_backend(name: str):
    """(force_backend_name, gbgpu_module_name, is_gpu) for ``cpu``/``cuda*``.

    Post Phase 3L.7f (2026-06-04) GBComputationGroupWrap +
    GBTDIonTheFlyWrap live in ``gbgpu_backend_<name>.cgbgpu``, not in
    ``fastlisaresponse_backend_<name>.tdionthefly``. We source the wrap
    class from there.
    """
    name = name.lower().strip()
    name = {"gpu": "cuda12x", "cuda": "cuda12x"}.get(name, name)
    if name == "cpu":
        return "cpu", "gbgpu_backend_cpu", False
    if name in ("cuda11x", "cuda12x", "cuda13x"):
        return name, f"gbgpu_backend_{name}", True
    raise ValueError(f"Unknown backend {name!r}.")


def build_pack(
    *,
    backend_name="cpu",
    f0_mhz=14.22,
    snr_target=50.0,
    seed=42,
    nt_layer=64,
    n_sparse_fd=1024,
    nt_sub=256,
    n_sparse=256,
    n_pad=None,
    # Default to the DIRECT (uncached) path so chunked-het matches lisatools
    # direct to machine precision, and therefore matches signal-het (which
    # was already validated against lisatools direct). ``gb_chunked_test_script.py``
    # and ``gb_chunked_prior_draws.py`` -- the canonical validation drivers --
    # also default to N_cp_sig=0 / N_cp_orbit=0 for this reason.
    #
    # Set these >0 to opt into the spline-cached chunked-het path for
    # production speed. With N_cp_sig=48 / N_cp_orbit=32 the per-chunk
    # spline approximation adds an ~mm ~ 4e-11 (GB) bias to <d|h>/<h|h>,
    # which shows up here as a small constant logL offset between
    # chunked-het and signal-het at injection. That bias is harmless for
    # MCMC posterior SHAPE but breaks the absolute-logL equivalence test.
    n_cp_sig=0,
    n_cp_orbit=0,
    # TUKEY_ALPHA = 0.0 -- NO TD-Tukey on the injection. This is the regime
    # ``gb_chunked_test_script.py`` validates as matching lisatools direct
    # to mm ~ 1e-9 (see its comment block around line 174). With TUKEY_ALPHA
    # > 0 the chunked-het template's intrinsic per-chunk Tukey (~0.05 at
    # N_sparse=256) doesn't mirror the global TD-Tukey applied to the data,
    # which shifts chunked-het's <d|h>/<h|h> by ~4 logL units at injection
    # while signal-het still matches because its bin-fold pipeline applies
    # an effective FD Tukey that does mirror the TD taper. Keep this at 0.0
    # for the absolute-logL equivalence test; raise it only for sampler-
    # robustness studies where the constant chunked-het offset doesn't matter.
    tukey_alpha=0.0,
    max_r=5.0,
    ll_ceiling=10.0,
    ll_reject=-1e30,
    verbose=True,
):
    """Build the signal-het + chunked-het kernel pack on the chosen backend.

    Returns a :class:`SimpleNamespace` with:

      - ``logl_signalhet(x, transform_fn=None)`` /
        ``logl_chunkedhet(x, transform_fn=None)`` -- the variable-batch
        likelihoods used by the MCMC.
      - ``cpp``, ``tdi_wrap``, ``gb_wdm_comp``, ``holder`` -- the underlying
        wrap objects + chunked-het pipeline.
      - ``c0_sparse_all``, ``c0_dense_cmplx_all``,
        ``A0_all``/``A1_all``/``B0_all``/``B1_all``,
        ``window_full``, ``n_sparse_local``, ``params_ref_all`` -- the
        signal-het reference buffers (already on the right device).
      - Geometry: ``Nf``, ``Nt``, ``Nf_active``, ``Nt_active``,
        ``Nt_layer``, ``N_sparse_t``, ``stride``, ``ind_min_t``,
        ``ind_min_f``, ``layer_df``, ``dt``, ``Tobs``, ``t_start``.
      - Injection: ``params_inj``, ``snr_inj``, ``d_d_lt``, ``amp_inj``.
      - Backend: ``xp``, ``backend_name``, ``is_gpu``,
        ``tukey_alpha``, ``max_r``, ``n_sparse_fd``.
    """
    backend_name, gbgpu_module_name, is_gpu = _resolve_backend(backend_name)
    if verbose:
        print(f"[backend] {backend_name}  gbgpu_module={gbgpu_module_name}  "
              f"is_gpu={is_gpu}", flush=True)
    _be = importlib.import_module(f"{gbgpu_module_name}.cgbgpu")
    if is_gpu:
        try:
            import cupy as xp
        except ImportError as e:
            raise RuntimeError(
                f"backend={backend_name} requires cupy "
                f"(cupy-{backend_name.replace('cuda', 'cuda')}).") from e
    else:
        xp = np

    # Allow tukey_alpha = 0.0 (validated regime in gb_chunked_test_script.py)
    # plus the documented [0.01, 0.05] band for sampler-robustness runs.
    assert tukey_alpha == 0.0 or 0.01 <= tukey_alpha <= 0.05, (
        f"tukey_alpha={tukey_alpha} not in {{0.0}} U [0.01, 0.05].")
    if n_pad is None:
        n_pad = nt_sub // 8

    np.random.seed(seed)

    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits = ESAOrbits(force_backend=backend_name)
    tdi_config = TDIConfig("2nd generation", force_backend=backend_name)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend_name,
    )
    tdi_wrap = gb_gen.wave_gen

    # The dense-TD kernel ``gb_run_wave_tdi_kernel`` allocates ``O(N)`` shared
    # memory per block (N = t_tdi grid length). At ``N=16384`` the per-block
    # buffer reaches ~475 KB, which exceeds the dynamic-shared-mem cap on
    # most NVIDIA parts (even with cudaFuncSetAttribute opt-in: A100 -> 99
    # KB, H100 -> 228 KB). The dense injection build is a one-time cost,
    # so we always do it on CPU and copy the resulting TD trace over to the
    # backend ``xp`` (cupy on GPU) before wrapping in TDSignal. The
    # signal-het kernel still uses the GPU-side ``tdi_wrap`` for the
    # per-call template build that runs every MCMC step.
    if is_gpu:
        orbits_cpu = ESAOrbits(force_backend="cpu")
        tdi_config_cpu = TDIConfig("2nd generation", force_backend="cpu")
        gb_gen_inj = GBTDIonTheFly(
            t_tdi, Tobs, t_start, 1.0 / dt, 1,
            tdi_config=tdi_config_cpu, orbits=orbits_cpu, tdi_chan="XYZ",
            force_backend="cpu",
        )
    else:
        gb_gen_inj = gb_gen

    def real_td_cb(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen_inj(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        td = np.asarray(spline.eval_tdi(t_arr))[0]
        # Transfer to backend xp so downstream ``TDSignal(td, settings=td_set)``
        # ``.transform(wdm_set, window=window)`` runs on the right device.
        return xp.asarray(td) if is_gpu else td

    td_set = TDSettings(Nobs, dt, force_backend=backend_name)
    window = _tukey(Nobs, alpha=tukey_alpha).astype(float)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend_name,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend_name,
    )
    layer_df  = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)

    # ----- injection -----
    f0_inj, fdot_inj = f0_mhz * 1e-3, 1e-16
    inc_inj, psi_inj, phi0_inj = np.pi / 3.0, 0.7, 1.4
    lam_inj, beta_inj = 2.1, 0.5
    amp_probe = 1e-22
    params_probe = np.array([amp_probe, f0_inj, fdot_inj, 0.0, phi0_inj,
                              inc_inj, psi_inj, lam_inj, beta_inj])
    td_probe = real_td_cb(params_probe)
    wdm_probe = TDSignal(td_probe, settings=td_set).transform(
        wdm_set_real, window=window)
    sens_mat_real = XYZ2SensitivityMatrix(
        DataResidualArray(wdm_probe).data_res_arr.settings, model="scirdv1")
    snr_probe = float(AnalysisContainer(
        DataResidualArray(wdm_probe), sens_mat_real).snr())
    amp_inj = amp_probe * (snr_target / max(snr_probe, 1e-30))
    params_inj = np.array([amp_inj, f0_inj, fdot_inj, 0.0, phi0_inj,
                            inc_inj, psi_inj, lam_inj, beta_inj])
    td_inj = real_td_cb(params_inj)
    wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
        wdm_set_real, window=window)
    analysis = AnalysisContainer(DataResidualArray(wdm_inj_real),
                                  sens_mat_real)
    snr_inj = float(analysis.snr())
    d_d_lt  = float(np.real(analysis.inner_product()))
    if verbose:
        print(f"[inject] amp={amp_inj:.3e} f0={f0_inj*1e3:.4f}mHz "
              f"fdot={fdot_inj:.2e} snr={snr_inj:.2f}  d_d={d_d_lt:.6e}",
              flush=True)

    # ----- signal-het reference -----
    wdm_inj_complex = np.asarray(
        TDSignal(td_inj, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
    c0_dense_active = wdm_inj_complex.copy()
    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=nt_layer, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local_host = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    window_full_host = sparse_gen.window_full.astype(np.float64)
    c0_sparse_active = c0_dense_active[:, :, n_sparse_local_host]
    sens_mat_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
    invC_complex = np.asarray(sens_mat_complex.invC)
    A0, A1, B0, B1 = python_bin_fold(
        wdm_inj_complex, c0_dense_active, invC_complex,
        n_sparse_local_host, stride, Nt_active, tdi_type="XYZ",
    )

    # Move all signal-het inputs to backend memory as OWNED buffers.
    # Mirror the original (pre-refactor) script's explicit `.copy()` calls
    # so the C++ kernel sees independent, contiguous arrays regardless of
    # the xp / view aliasing behavior.
    c0_sparse_all      = xp.asarray(c0_sparse_active[None, ...].copy())
    c0_dense_cmplx_all = xp.asarray(c0_dense_active[None, ...].copy())
    A0_all = xp.asarray(A0[None, ...].copy()); A1_all = xp.asarray(A1[None, ...].copy())
    B0_all = xp.asarray(B0[None, ...].copy()); B1_all = xp.asarray(B1[None, ...].copy())
    window_full    = xp.asarray(window_full_host.copy())
    n_sparse_local = xp.asarray(n_sparse_local_host.astype(np.int32).copy())
    params_ref_all = xp.asarray(params_inj.astype(np.float64).reshape(1, 9).copy())
    cpp = (_be.GBComputationGroupWrapGPU() if is_gpu
           else _be.GBComputationGroupWrapCPU())

    # ----- chunked-het construction -----
    gb_wdm_comp = GBWDMComputations(
        wdm_set_real, t_ref=t_start,
        Nt_sub=nt_sub, n_pad=n_pad, N_sparse=n_sparse,
        N_cp_sig=n_cp_sig, N_cp_orbit=n_cp_orbit,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend_name, d_d=0.0, tdi_type="XYZ",
    )
    if verbose:
        print(f"[chunked] n_chunks={gb_wdm_comp.n_chunks}  "
              f"T_chunk={gb_wdm_comp.T_chunk:.3e}s  "
              f"alpha={gb_wdm_comp.resolved_tukey_alpha}", flush=True)

    inj_active = xp.asarray(wdm_inj_real.arr)               # (3, Nf_a, Nt_a)
    invC_active = xp.asarray(sens_mat_real.invC)            # (3, 3, Nf_a, Nt_a)
    invC_active = xp.where(xp.isfinite(invC_active), invC_active, 0.0)
    holder = _FullGridWDMHolder(inj_active, invC_active)

    # ----- closures (variable batch) -----
    NT_LAYER = int(nt_layer)
    N_SPARSE_FD = int(n_sparse_fd)
    TUKEY_ALPHA = float(tukey_alpha)
    MAX_R = float(max_r)

    def logl_signalhet(x, transform_fn=None, **_kw):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        phys = (transform_fn.both_transforms(x_arr.copy())
                if transform_fn is not None else x_arr)
        N = phys.shape[0]
        params_cand_all = xp.asarray(
            np.ascontiguousarray(phys.astype(np.float64)))
        d_h_out = xp.zeros(N, dtype=np.float64)
        h_h_out = xp.zeros(N, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap, d_h_out, h_h_out,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_ref_all,
            xp.zeros(N, dtype=np.int32),
            N, 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride,
            ind_min_t, ind_min_f, 2,
            layer_df, dt, Tobs, t_start,
            3, 0, N_SPARSE_FD,
            TUKEY_ALPHA, MAX_R, 0)
        ll = -0.5 * d_d_lt + np.asarray(d_h_out) - 0.5 * np.asarray(h_h_out)
        bad = ~np.isfinite(ll) | (ll > ll_ceiling)
        return np.where(bad, ll_reject, ll)

    def logl_chunkedhet(x, transform_fn=None, **_kw):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        phys = (transform_fn.both_transforms(x_arr.copy())
                if transform_fn is not None else x_arr)
        ll = gb_wdm_comp.get_ll_wdm(
            phys, holder,
            convert_to_ra_dec=False,
            use_layer_groups=True,
            group_band_layers=5,
            margin_layers=0,
        )
        ll = np.asarray(ll) - 0.5 * d_d_lt
        bad = ~np.isfinite(ll) | (ll > ll_ceiling)
        return np.where(bad, ll_reject, ll)

    return SimpleNamespace(
        # KEEP-ALIVES: Python objects whose C++ pointers are read by the
        # kernels. Stashing them here keeps them from being GC'd when
        # build_pack returns -- otherwise tdi_wrap, gb_wdm_comp, and the
        # holder dangle into freed memory and the first kernel call SEGVs.
        _orbits=orbits, _tdi_config=tdi_config, _gb_gen=gb_gen,
        _wdm_set_real=wdm_set_real, _wdm_set_complex=wdm_set_complex,
        _td_set=td_set, _sparse_gen=sparse_gen,
        _sens_mat_real=sens_mat_real, _sens_mat_complex=sens_mat_complex,
        _wdm_inj_real=wdm_inj_real, _wdm_inj_complex=wdm_inj_complex,
        _inj_active=inj_active, _invC_active=invC_active,
        # backend
        xp=xp, backend_name=backend_name, is_gpu=is_gpu,
        # geometry
        Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
        Nt_layer=NT_LAYER, N_sparse_t=N_sparse_t, stride=stride,
        ind_min_t=ind_min_t, ind_min_f=ind_min_f,
        layer_df=layer_df, dt=dt, Tobs=Tobs, t_start=t_start,
        # knobs
        tukey_alpha=TUKEY_ALPHA, max_r=MAX_R, n_sparse_fd=N_SPARSE_FD,
        # injection
        params_inj=params_inj, amp_inj=amp_inj,
        snr_inj=snr_inj, d_d_lt=d_d_lt,
        # signal-het buffers
        cpp=cpp, tdi_wrap=tdi_wrap,
        c0_sparse_all=c0_sparse_all,
        c0_dense_cmplx_all=c0_dense_cmplx_all,
        A0_all=A0_all, A1_all=A1_all, B0_all=B0_all, B1_all=B1_all,
        window_full=window_full, n_sparse_local=n_sparse_local,
        params_ref_all=params_ref_all,
        # chunked-het
        gb_wdm_comp=gb_wdm_comp, holder=holder,
        # MCMC closures (variable batch)
        logl_signalhet=logl_signalhet,
        logl_chunkedhet=logl_chunkedhet,
        # raw reject sentinel
        ll_reject=ll_reject, ll_ceiling=ll_ceiling,
    )


def main():
    # ---- shared knobs ----
    SEED          = int(os.environ.get("SEED", "42"))
    NWALKERS      = int(os.environ.get("NWALKERS", "16"))
    NTEMPS        = int(os.environ.get("NTEMPS", "10"))
    NSTEPS        = int(os.environ.get("NSTEPS", "500"))
    BURNIN        = int(os.environ.get("BURNIN", "0"))
    F0_MHZ        = float(os.environ.get("F0_MHZ", "14.22"))
    SNR_TARGET    = float(os.environ.get("SNR_TARGET", "50.0"))
    START_FACTOR  = float(os.environ.get("START_FACTOR", "1e-3"))
    # Sprint convention (`recommended_tukey_alpha("heterodyne")`): EVERY
    # Tukey in the pipeline must land in the validated [0.01, 0.05] range.
    # Three Tukeys are in play:
    #   (1) GLOBAL Tukey on the injection TD before the WDM transform that
    #       builds the data both kernels score against (TUKEY_ALPHA below).
    #   (2) chunked-het's per-chunk Tukey on each chunk's sparse FD
    #       (alpha auto-resolved by GBWDMComputations to ~0.05 for
    #       N_sparse=256 -- already in range).
    #   (3) signal-het's sparse-FD Tukey passed as kernel arg
    #       (we forward TUKEY_ALPHA to it so template + data stay consistent
    #       on the signal-het side).
    #
    # Note: applying (1) shifts the chunked-het logL floor at injection
    # from ~1e-6 to ~4 (a constant ~4 unit offset that all candidates
    # share; affects absolute logL only, not posterior shape). The
    # signal-het floor stays ~0.025. The two kernels still target the
    # same posterior peak; only the absolute logL normalisation differs
    # by a constant -- harmless for MCMC posterior shape, but the
    # comparison-at-injection numerics will look asymmetric.
    # Default 0.0 = NO TD-Tukey on injection; the validated regime that
    # makes chunked-het match lisatools direct (and therefore signal-het).
    # See the comment block at ``build_pack`` for why.
    TUKEY_ALPHA   = float(os.environ.get("TUKEY_ALPHA", "0.0"))
    assert TUKEY_ALPHA == 0.0 or 0.01 <= TUKEY_ALPHA <= 0.05, (
        f"TUKEY_ALPHA={TUKEY_ALPHA} not in {{0.0}} U [0.01, 0.05].")
    # signal-het knobs
    NT_LAYER      = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD   = int(os.environ.get("N_SPARSE_FD", "1024"))
    MAX_R         = float(os.environ.get("MAX_R", "5.0"))
    # chunked-het knobs
    NT_SUB        = int(os.environ.get("NT_SUB", "256"))
    N_SPARSE      = int(os.environ.get("N_SPARSE", "256"))
    N_PAD         = int(os.environ.get("N_PAD", str(NT_SUB // 8)))
    # Default to N_cp_sig=0 / N_cp_orbit=0 (direct, uncached chunked-het).
    # See the comment at ``build_pack`` -- the cached path biases logL by
    # ~mm 4e-11 vs signal-het / lisatools direct. Opt in via env var for
    # production timing runs.
    N_CP_SIG      = int(os.environ.get("N_CP_SIG", "0"))
    N_CP_ORBIT    = int(os.environ.get("N_CP_ORBIT", "0"))
    # priors (wide angular priors by default, matching the stretch-clip run)
    PRIOR_FAC_F0      = float(os.environ.get("PRIOR_FAC_F0",     "0.4"))
    PRIOR_FAC_AMP     = float(os.environ.get("PRIOR_FAC_AMP",    "0.5"))
    PRIOR_FAC_FDOT    = float(os.environ.get("PRIOR_FAC_FDOT",   "1e-14"))
    PRIOR_FAC_PHI0    = float(os.environ.get("PRIOR_FAC_PHI0",   "3.14159"))
    PRIOR_FAC_COSINC  = float(os.environ.get("PRIOR_FAC_COSINC", "1.0"))
    PRIOR_FAC_PSI     = float(os.environ.get("PRIOR_FAC_PSI",    "1.57"))
    PRIOR_FAC_LAM     = float(os.environ.get("PRIOR_FAC_LAM",    "3.14159"))
    PRIOR_FAC_SINBETA = float(os.environ.get("PRIOR_FAC_SINBETA","1.0"))
    # outputs
    BACKEND_SH    = os.environ.get("BACKEND_SH", "mcmc_compare_signalhet.h5")
    BACKEND_CH    = os.environ.get("BACKEND_CH", "mcmc_compare_chunked.h5")
    OUT_PNG       = os.environ.get("OUT_PNG", "corner_compare_signalhet_chunked.png")
    SKIP_SH       = os.environ.get("SKIP_SH", "0") == "1"
    SKIP_CH       = os.environ.get("SKIP_CH", "0") == "1"

    BACKEND_NAME  = os.environ.get("BACKEND", "cpu")

    pack = build_pack(
        backend_name=BACKEND_NAME,
        f0_mhz=F0_MHZ, snr_target=SNR_TARGET, seed=SEED,
        nt_layer=NT_LAYER, n_sparse_fd=N_SPARSE_FD,
        nt_sub=NT_SUB, n_sparse=N_SPARSE, n_pad=N_PAD,
        n_cp_sig=N_CP_SIG, n_cp_orbit=N_CP_ORBIT,
        tukey_alpha=TUKEY_ALPHA, max_r=MAX_R,
    )
    # Re-bind locals from the pack so the surrounding MCMC code reads
    # identically to the pre-refactor version.
    logl_signalhet = pack.logl_signalhet
    logl_chunkedhet = pack.logl_chunkedhet
    layer_df = pack.layer_df
    amp_inj = pack.amp_inj
    f0_inj  = pack.params_inj[1]
    fdot_inj = pack.params_inj[2]
    phi0_inj = pack.params_inj[4]
    inc_inj  = pack.params_inj[5]
    psi_inj  = pack.params_inj[6]
    lam_inj  = pack.params_inj[7]
    beta_inj = pack.params_inj[8]
    snr_inj  = pack.snr_inj
    d_d_lt   = pack.d_d_lt
    LL_REJECT = pack.ll_reject

    # ------ Shared sampler setup ------
    full_basis    = ["amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"]
    sampled_basis = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"]
    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms={"cosinc": np.arccos, "sinbeta": np.arcsin},
        fill_dict={"fddot0": 0.0},
        key_map={"cosinc": "inc", "sinbeta": "beta"},
    )

    f0_layer_frac = (f0_inj / layer_df) - np.floor(f0_inj / layer_df)
    f0_safe_lo = max(-f0_layer_frac + 1e-3, -PRIOR_FAC_F0) * layer_df
    f0_safe_hi = min((1.0 - f0_layer_frac) - 1e-3,  PRIOR_FAC_F0) * layer_df
    priors = {"gb": ProbDistContainer({
        0: uniform_dist(amp_inj * (1.0 - PRIOR_FAC_AMP),
                        amp_inj * (1.0 + PRIOR_FAC_AMP)),
        1: uniform_dist(f0_inj + f0_safe_lo, f0_inj + f0_safe_hi),
        2: uniform_dist(fdot_inj - PRIOR_FAC_FDOT,
                        fdot_inj + PRIOR_FAC_FDOT),
        3: uniform_dist(phi0_inj - PRIOR_FAC_PHI0,
                        phi0_inj + PRIOR_FAC_PHI0),
        4: uniform_dist(max(-1.0, np.cos(inc_inj) - PRIOR_FAC_COSINC),
                        min( 1.0, np.cos(inc_inj) + PRIOR_FAC_COSINC)),
        5: uniform_dist(psi_inj - PRIOR_FAC_PSI, psi_inj + PRIOR_FAC_PSI),
        6: uniform_dist(lam_inj - PRIOR_FAC_LAM, lam_inj + PRIOR_FAC_LAM),
        7: uniform_dist(max(-1.0, np.sin(beta_inj) - PRIOR_FAC_SINBETA),
                        min( 1.0, np.sin(beta_inj) + PRIOR_FAC_SINBETA)),
    })}

    sf = START_FACTOR
    gen_dist = {"gb": ProbDistContainer({
        0: uniform_dist(amp_inj * (1.0 - sf), amp_inj * (1.0 + sf)),
        1: uniform_dist(f0_inj - sf * layer_df, f0_inj + sf * layer_df),
        2: uniform_dist(fdot_inj - 1e-17, fdot_inj + 1e-17),
        3: uniform_dist(phi0_inj - sf, phi0_inj + sf),
        4: uniform_dist(np.cos(inc_inj) - sf, np.cos(inc_inj) + sf),
        5: uniform_dist(psi_inj - sf, psi_inj + sf),
        6: uniform_dist(lam_inj - sf, lam_inj + sf),
        7: uniform_dist(np.sin(beta_inj) - sf, np.sin(beta_inj) + sf),
    })}

    ndims = {"gb": len(sampled_basis)}
    periodic_container = PeriodicContainer(
        {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}},
        key_order={"gb": list(range(len(sampled_basis)))},
    )

    # Shared starting cloud -- both samplers start from EXACTLY the same state.
    np.random.seed(SEED)
    start_coords = gen_dist["gb"].rvs(size=(NTEMPS, NWALKERS, 1))
    inj_in_sampled = np.array([
        amp_inj, f0_inj, fdot_inj, phi0_inj,
        np.cos(inc_inj), psi_inj, lam_inj, np.sin(beta_inj),
    ], dtype=float)

    # --------------------------------------------------------------------
    # PRE-FLIGHT: verify both kernels return the SAME logL on the SAME
    # candidates. The two kernels are different implementations of the
    # same Gaussian likelihood, so off the injection they should agree
    # to within their individual heterodyne floors (~1e-3 absolute on a
    # SNR=50 source by construction). If they don't, the posteriors
    # WILL differ; bail out and force a fix before burning sampler time.
    # --------------------------------------------------------------------
    ll_inj_sh = float(logl_signalhet(inj_in_sampled[None, :], transform_fn=tc)[0])
    ll_inj_ch = float(logl_chunkedhet(inj_in_sampled[None, :], transform_fn=tc)[0])
    print(f"[check] logL @ injection (analytic truth = 0):", flush=True)
    print(f"          signal-het  = {ll_inj_sh:+.6e}", flush=True)
    print(f"          chunked-het = {ll_inj_ch:+.6e}", flush=True)
    print(f"          difference  = {ll_inj_sh - ll_inj_ch:+.3e}", flush=True)

    # Per-call agreement at N test points drawn from the same prior the
    # sampler will see. Reports max abs/rel diff; refuses to start MCMC
    # if it exceeds a threshold.
    N_VERIFY = int(os.environ.get("N_VERIFY", "32"))
    SKIP_VERIFY = os.environ.get("SKIP_VERIFY", "0") == "1"
    if N_VERIFY > 0 and not SKIP_VERIFY:
        np.random.seed(SEED + 1)
        # draw from the start-cloud (tight around inj) so we're in the
        # regime both kernels are reliable; then add a few from the wider
        # prior so we see how they track when r(t) excursions matter.
        x_tight = gen_dist["gb"].rvs(size=(N_VERIFY // 2,))
        x_wide  = priors["gb"].rvs(size=(N_VERIFY - N_VERIFY // 2,))
        x_test  = np.vstack([x_tight, x_wide])
        # ProbDistContainer.rvs returns (size, 1, ndim); squeeze leaf axis.
        if x_test.ndim == 3:
            x_test = x_test.reshape(x_test.shape[0], -1)
        ll_sh = logl_signalhet(x_test, transform_fn=tc)
        ll_ch = logl_chunkedhet(x_test, transform_fn=tc)
        # Filter walkers either side rejected (==LL_REJECT)
        ok = (ll_sh > LL_REJECT * 0.5) & (ll_ch > LL_REJECT * 0.5) \
             & np.isfinite(ll_sh) & np.isfinite(ll_ch)
        n_ok = int(ok.sum())
        if n_ok == 0:
            raise RuntimeError("[verify] both kernels rejected every test point.")
        diff = (ll_sh - ll_ch)[ok]
        rel  = diff / np.maximum(np.abs(ll_ch[ok]), 1.0)
        print(f"\n[verify] {n_ok}/{N_VERIFY} test points usable", flush=True)
        print(f"[verify] ll_sh range  = [{ll_sh[ok].min():+.4e}, {ll_sh[ok].max():+.4e}]",
              flush=True)
        print(f"[verify] ll_ch range  = [{ll_ch[ok].min():+.4e}, {ll_ch[ok].max():+.4e}]",
              flush=True)
        print(f"[verify] diff abs:    median={np.median(np.abs(diff)):.3e}  "
              f"max={np.max(np.abs(diff)):.3e}", flush=True)
        print(f"[verify] diff rel:    median={np.median(np.abs(rel)):.3e}  "
              f"max={np.max(np.abs(rel)):.3e}", flush=True)
        TOL_REL = float(os.environ.get("VERIFY_TOL_REL", "0.05"))
        if np.max(np.abs(rel)) > TOL_REL:
            # First few worst offenders
            order = np.argsort(-np.abs(rel))
            print(f"\n[verify] top 5 mismatches (sh - ch):", flush=True)
            for i in order[:5]:
                print(f"  x = {x_test[ok][i]}", flush=True)
                print(f"  ll_sh = {ll_sh[ok][i]:+.4e}  "
                      f"ll_ch = {ll_ch[ok][i]:+.4e}  "
                      f"diff  = {ll_sh[ok][i] - ll_ch[ok][i]:+.3e}",
                      flush=True)
            raise RuntimeError(
                f"[verify] FAILED: max relative diff = {np.max(np.abs(rel)):.3e} "
                f"> tol {TOL_REL:.3e}. Fix the kernels before running MCMC. "
                f"Set SKIP_VERIFY=1 to bypass (NOT recommended).")
        print(f"[verify] PASSED (max rel diff {np.max(np.abs(rel)):.3e} "
              f"<= tol {TOL_REL:.3e})", flush=True)

    # ----- Run A: signal-het -----
    if not SKIP_SH:
        print(f"\n=== Run A: signal-het  ->  {BACKEND_SH} ===", flush=True)
        if os.path.exists(BACKEND_SH):
            os.remove(BACKEND_SH)
        backend_sh = HDFBackend(BACKEND_SH)
        state_sh = State({"gb": start_coords.copy()})
        sampler_sh = EnsembleSampler(
            NWALKERS, ndims, logl_signalhet, priors,
            tempering_kwargs=dict(ntemps=NTEMPS),
            kwargs=dict(transform_fn=tc),
            moves=StretchMove(live_dangerously=True),
            branch_names=["gb"],
            periodic=periodic_container,
            backend=backend_sh,
            vectorize=True,
        )
        state_sh.log_prior = sampler_sh.compute_log_prior(state_sh.branches_coords)
        state_sh.log_like = sampler_sh.compute_log_like(
            state_sh.branches_coords, logp=state_sh.log_prior)[0]
        print(f"[A] start cold ll: mean={np.asarray(state_sh.log_like)[0].mean():.3e}  "
              f"max={np.asarray(state_sh.log_like)[0].max():.3e}", flush=True)
        t0 = time.perf_counter()
        sampler_sh.run_mcmc(state_sh, NSTEPS, burn=BURNIN, progress=True)
        t1 = time.perf_counter()
        print(f"[A] elapsed = {t1-t0:.1f}s  ({(t1-t0)/NSTEPS*1000:.2f} ms/step)",
              flush=True)
    else:
        print(f"[A] SKIP_SH=1 -- using existing {BACKEND_SH}", flush=True)

    # ----- Run B: chunked-het -----
    if not SKIP_CH:
        print(f"\n=== Run B: chunked-het  ->  {BACKEND_CH} ===", flush=True)
        if os.path.exists(BACKEND_CH):
            os.remove(BACKEND_CH)
        backend_ch = HDFBackend(BACKEND_CH)
        state_ch = State({"gb": start_coords.copy()})
        sampler_ch = EnsembleSampler(
            NWALKERS, ndims, logl_chunkedhet, priors,
            tempering_kwargs=dict(ntemps=NTEMPS),
            kwargs=dict(transform_fn=tc),
            moves=StretchMove(live_dangerously=True),
            branch_names=["gb"],
            periodic=periodic_container,
            backend=backend_ch,
            vectorize=True,
        )
        state_ch.log_prior = sampler_ch.compute_log_prior(state_ch.branches_coords)
        state_ch.log_like = sampler_ch.compute_log_like(
            state_ch.branches_coords, logp=state_ch.log_prior)[0]
        print(f"[B] start cold ll: mean={np.asarray(state_ch.log_like)[0].mean():.3e}  "
              f"max={np.asarray(state_ch.log_like)[0].max():.3e}", flush=True)
        t0 = time.perf_counter()
        sampler_ch.run_mcmc(state_ch, NSTEPS, burn=BURNIN, progress=True)
        t1 = time.perf_counter()
        print(f"[B] elapsed = {t1-t0:.1f}s  ({(t1-t0)/NSTEPS*1000:.2f} ms/step)",
              flush=True)
    else:
        print(f"[B] SKIP_CH=1 -- using existing {BACKEND_CH}", flush=True)

    # ----- Overlay corner plot -----
    be_sh = HDFBackend(BACKEND_SH)
    be_ch = HDFBackend(BACKEND_CH)
    chain_sh = be_sh.get_chain()["gb"]        # (nsteps, ntemps, nwalkers, 1, ndim)
    chain_ch = be_ch.get_chain()["gb"]
    n_burn = max(1, int(0.3 * chain_sh.shape[0]))
    samples_sh = chain_sh[n_burn:, 0].reshape(-1, len(sampled_basis))
    samples_ch = chain_ch[n_burn:, 0].reshape(-1, len(sampled_basis))
    print(f"\n[overlay] tail samples: sh={samples_sh.shape[0]}  ch={samples_ch.shape[0]}",
          flush=True)

    # Per-dim posterior mean/std comparison
    print(f"\n  {'param':>9s} {'inj':>14s} "
          f"{'sh mean':>14s} {'ch mean':>14s} "
          f"{'sh std':>11s} {'ch std':>11s}  "
          f"{'|dmean|/std':>11s}", flush=True)
    for k, name in enumerate(sampled_basis):
        mean_sh, std_sh = samples_sh[:, k].mean(), samples_sh[:, k].std()
        mean_ch, std_ch = samples_ch[:, k].mean(), samples_ch[:, k].std()
        diff_norm = abs(mean_sh - mean_ch) / max(min(std_sh, std_ch), 1e-30)
        print(f"  {name:>9s} {inj_in_sampled[k]:>+14.6e} "
              f"{mean_sh:>+14.6e} {mean_ch:>+14.6e} "
              f"{std_sh:>11.3e} {std_ch:>11.3e}  "
              f"{diff_norm:>11.3f}", flush=True)

    fig = corner.corner(
        samples_sh, labels=sampled_basis, truths=inj_in_sampled,
        truth_color="black",
        color="C0",
        levels=(0.393, 0.865, 0.989),
        plot_datapoints=False, plot_density=False, no_fill_contours=True,
        smooth=1.0, contour_kwargs={"linewidths": 1.4},
        hist_kwargs={"density": True, "histtype": "step", "color": "C0", "lw": 1.0},
    )
    corner.corner(
        samples_ch, fig=fig,
        color="C1",
        levels=(0.393, 0.865, 0.989),
        plot_datapoints=False, plot_density=False, no_fill_contours=True,
        smooth=1.0, contour_kwargs={"linewidths": 1.4, "linestyles": "--"},
        hist_kwargs={"density": True, "histtype": "step", "color": "C1", "lw": 1.0},
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="C0", lw=1.4, label="signal-het"),
            plt.Line2D([0], [0], color="C1", lw=1.4, ls="--", label="chunked-het"),
        ],
        loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10,
    )
    fig.suptitle(
        f"signal-het vs chunked-het posterior overlay  "
        f"(NSTEPS={chain_sh.shape[0]}, burn={n_burn}, "
        f"{NWALKERS}w x {NTEMPS}T, SNR={snr_inj:.0f})",
        fontsize=11, y=1.01,
    )
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"\n[plot] saved -> {OUT_PNG}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
