#!/usr/bin/env python
"""Eryn GB MCMC using the Stage 2b signal-het get_ll as the vectorized
likelihood.

The C++ `GBComputationGroupWrapCPU.gb_signal_het_get_ll_in_kernel` takes a
batch of num_bin candidates and returns per-binary <d|h> and <h|h>, so it
maps directly to Eryn's vectorize=True pattern -- one likelihood call per
sampler step covers all walkers at once.

The bin-fold A0/A1/B0/B1 tables and c0_sparse are precomputed ONCE at
construction time from the injection's complex WDM transform. The C++
kernel reuses these for every walker; only the per-source FD generation
runs per walker.

Run::
    python test_gb_signal_het_mcmc.py

Env vars (knobs):
    SEED               default 42
    NWALKERS           default 16
    NTEMPS             default 1
    NSTEPS             default 50
    BURNIN             default 0
    NT_LAYER           default 64
    N_SPARSE_FD        default 1024
    TUKEY_ALPHA        default 0.05
    F0_MHZ             default 14.22   (injection f0 in mHz)
    SNR_TARGET         default 50.0    (rough target SNR for amp)
    START_FACTOR       default 1e-3    (init walkers within +/- this around inj)
    TIMER_REPEATS      default 5       (likelihood-timing reps before sampling)
    BACKEND_PATH       default ""      (HDF backend; empty -> in-memory only)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
import lisatools_backend_cpu.pycppdetector as _lat_pd
import gbgpu_backend_cpu.cgbgpu as _be  # GBComputationGroupWrap lives here post-3L.7g

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.moves.nuts import NUTSMove
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


def main():
    SEED            = int(os.environ.get("SEED", "42"))
    NWALKERS        = int(os.environ.get("NWALKERS", "16"))
    NTEMPS          = int(os.environ.get("NTEMPS", "1"))
    NSTEPS          = int(os.environ.get("NSTEPS", "50"))
    BURNIN          = int(os.environ.get("BURNIN", "0"))
    NT_LAYER        = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD     = int(os.environ.get("N_SPARSE_FD", "1024"))
    TUKEY_ALPHA     = float(os.environ.get("TUKEY_ALPHA", "0.05"))
    F0_MHZ          = float(os.environ.get("F0_MHZ", "14.22"))
    SNR_TARGET      = float(os.environ.get("SNR_TARGET", "50.0"))
    START_FACTOR    = float(os.environ.get("START_FACTOR", "1e-3"))
    TIMER_REPEATS   = int(os.environ.get("TIMER_REPEATS", "5"))
    BACKEND_PATH    = os.environ.get("BACKEND_PATH", "")
    MOVES           = os.environ.get("MOVES", "stretch").lower()
    CORNER_PATH     = os.environ.get("CORNER_PATH", "")
    NUTS_STEP_SIZE  = float(os.environ.get("NUTS_STEP_SIZE", "0.05"))
    NUTS_MAX_TREE   = int(os.environ.get("NUTS_MAX_TREE_DEPTH", "1"))
    PROGRESS        = os.environ.get("PROGRESS", "1") == "1"
    # max_r > 0 caps |r| per channel-cell inside the C++ bin-fold so the
    # positive-logL blowup at wild angle excursions is killed at its
    # source. Validated to be threshold-insensitive in {5, 10}.
    MAX_R           = float(os.environ.get("MAX_R", "5.0"))
    if MOVES not in ("stretch", "nuts"):
        raise ValueError(f"MOVES must be 'stretch' or 'nuts', got {MOVES!r}")
    print(f"[run] MOVES={MOVES}", flush=True)

    np.random.seed(SEED)

    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    tdi_wrap = gb_gen.wave_gen

    def real_td_cb(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=TUKEY_ALPHA).astype(float)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)

    # ----- Injection ----- #
    f0_inj   = F0_MHZ * 1e-3
    fdot_inj = 1e-16
    inc_inj  = np.pi / 3.0
    psi_inj  = 0.7
    phi0_inj = 1.4
    lam_inj  = 2.1
    beta_inj = 0.5
    # Pick amp to roughly hit SNR_TARGET (linear in amp).
    amp_probe = 1e-22
    params_probe = np.array([amp_probe, f0_inj, fdot_inj, 0.0,
                              phi0_inj, inc_inj, psi_inj,
                              lam_inj, beta_inj])
    td_probe = real_td_cb(params_probe)
    wdm_probe = TDSignal(td_probe, settings=td_set).transform(
        wdm_set_real, window=window)
    inj_arr_probe = DataResidualArray(wdm_probe)
    sens_mat_real = XYZ2SensitivityMatrix(
        inj_arr_probe.data_res_arr.settings, model="scirdv1")
    snr_probe = float(AnalysisContainer(inj_arr_probe, sens_mat_real).snr())
    amp_inj = amp_probe * (SNR_TARGET / max(snr_probe, 1e-30))

    params_inj = np.array([amp_inj, f0_inj, fdot_inj, 0.0,
                            phi0_inj, inc_inj, psi_inj,
                            lam_inj, beta_inj])
    td_inj = real_td_cb(params_inj)
    wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
        wdm_set_real, window=window)
    inj_data_arr = DataResidualArray(wdm_inj_real)
    analysis = AnalysisContainer(inj_data_arr, sens_mat_real)
    snr_inj = float(analysis.snr())
    d_d_lt  = float(np.real(analysis.inner_product()))
    print(f"[inject] amp={amp_inj:.3e} f0={f0_inj*1e3:.4f}mHz "
          f"fdot={fdot_inj:.2e} snr={snr_inj:.2f}", flush=True)
    print(f"[inject] <d|d>(lisatools) = {d_d_lt:.6e}", flush=True)

    # ----- Reference c0 + bin-fold tables (precomputed ONCE) ----- #
    wdm_inj_complex = np.asarray(
        TDSignal(td_inj, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
    c0_dense_active = wdm_inj_complex.copy()
    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=NT_LAYER, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    window_full = sparse_gen.window_full.astype(np.float64).copy()
    c0_sparse_active = c0_dense_active[:, :, n_sparse_local].copy()
    sens_mat_complex = XYZ2SensitivityMatrix(wdm_set_complex,
                                              model="scirdv1")
    invC_complex = np.asarray(sens_mat_complex.invC).copy()
    A0, A1, B0, B1 = python_bin_fold(
        wdm_inj_complex, c0_dense_active, invC_complex,
        n_sparse_local, stride, Nt_active, tdi_type="XYZ",
    )
    c0_sparse_all = c0_sparse_active[None, ...].copy()  # (1, 3, Nf_active, N_sparse_t)
    A0_all = A0[None, ...].copy()
    A1_all = A1[None, ...].copy()
    B0_all = B0[None, ...].copy()
    B1_all = B1[None, ...].copy()
    params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()

    print(f"[setup] NT_LAYER={NT_LAYER} stride={stride} N_sparse_t={N_sparse_t} "
          f"N_sparse_fd={N_SPARSE_FD} tukey_alpha={TUKEY_ALPHA} "
          f"max_r={MAX_R}", flush=True)

    cpp = _be.GBComputationGroupWrapCPU()

    # ---- Vectorized log-likelihood: one C++ call per (ntemps*nwalkers) batch ----
    def logl_vec(x, transform_fn=None, **_kw):
        """Stage 2b signal-het log-likelihood, batched over walkers.

        x: (N, ndim_sampled) -- Eryn passes the flattened (ntemps*nwalkers, ndim).
        transform_fn: TransformContainer mapping sampled -> physical 9-D.
        Returns: (N,) log-likelihood values
        (logL = -0.5*<d|d> + d_h - 0.5*h_h, full Gaussian).
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        if transform_fn is not None:
            phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
        else:
            phys = x_arr
        N = phys.shape[0]
        params_cand_all = np.ascontiguousarray(phys.astype(np.float64))
        data_index_all = np.zeros(N, dtype=np.int32)
        d_h_out = np.zeros(N, dtype=np.float64)
        h_h_out = np.zeros(N, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap,
            d_h_out, h_h_out,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_ref_all, data_index_all,
            N, 1,
            9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            Tobs, t_start,
            3, 0, N_SPARSE_FD,
            TUKEY_ALPHA, MAX_R,
        )
        # Full Gaussian: -0.5 <d|d> + <d|h> - 0.5 <h|h>.
        ll = -0.5 * d_d_lt + d_h_out - 0.5 * h_h_out
        # Guard: the TRUE logL is bounded above by 0 (achieved at injection).
        # The signal-het bin-fold can produce large POSITIVE values when r
        # excursions land near the null space of M_{c,c2} (the channel
        # Hermitian PSD matrix) -- h_h collapses while d_h stays large. Any
        # logL well above 0 is a numerical artifact, not a real likelihood
        # peak. Clamp to a strongly negative value so these proposals get
        # rejected.
        LOGL_CEILING = 10.0
        LOGL_REJECT  = -1e30
        bad = ~np.isfinite(ll) | (ll > LOGL_CEILING)
        if np.any(bad):
            ll = np.where(bad, LOGL_REJECT, ll)
        return ll

    # ----- Sampler setup ----- #
    # Sampled-basis: 8 dims (cosinc / sinbeta replace inc / beta; fddot=0).
    full_basis = ["amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi",
                  "lam", "beta"]
    sampled_basis = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam",
                     "sinbeta"]
    key_map = {"cosinc": "inc", "sinbeta": "beta"}
    parameter_transforms = {"cosinc": np.arccos, "sinbeta": np.arcsin}
    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={"fddot0": 0.0},
        key_map=key_map,
    )

    # Priors are NARROW around injection. The signal-het likelihood requires
    # the candidate to stay in a heterodyne-valid neighborhood of the
    # reference (= injection in this test). Two failure modes drive the
    # bounds:
    #   (a) m_floor crossing: f0 must stay in the same WDM layer as the
    #       reference (otherwise the active-band c0 coefficients are
    #       essentially zero where the candidate needs signal).
    #   (b) c0 dynamic range: even with the same m_floor, wide angle
    #       excursions make c0 vary by many orders of magnitude across
    #       sparse n; the per-layer safe-divide floor (1e-12 * max|c0|)
    #       doesn't catch borderline-small cells, so r = c1/c0 can still
    #       blow up.
    # Sizing each prior at ~5-10x the SNR=50 posterior width (rough
    # Fisher) gives the chain breathing room without crossing into the
    # blow-up region.
    PRIOR_FAC_F0     = float(os.environ.get("PRIOR_FAC_F0",     "0.4"))    # +/- layer_df fraction
    PRIOR_FAC_AMP    = float(os.environ.get("PRIOR_FAC_AMP",    "0.20"))   # +/- relative
    PRIOR_FAC_FDOT   = float(os.environ.get("PRIOR_FAC_FDOT",   "5e-16"))  # +/- Hz/s
    PRIOR_FAC_PHI0   = float(os.environ.get("PRIOR_FAC_PHI0",   "0.30"))   # +/- rad
    PRIOR_FAC_COSINC = float(os.environ.get("PRIOR_FAC_COSINC", "0.10"))   # +/- in cosinc
    PRIOR_FAC_PSI    = float(os.environ.get("PRIOR_FAC_PSI",    "0.20"))   # +/- rad
    PRIOR_FAC_LAM    = float(os.environ.get("PRIOR_FAC_LAM",    "0.20"))   # +/- rad
    PRIOR_FAC_SINBETA= float(os.environ.get("PRIOR_FAC_SINBETA","0.10"))   # +/- in sinbeta

    # Anchor f0 prior so m_floor cannot cross.
    f0_layer_frac = (f0_inj / layer_df) - np.floor(f0_inj / layer_df)
    f0_safe_lo = max(-f0_layer_frac + 1e-3,         -PRIOR_FAC_F0) * layer_df
    f0_safe_hi = min((1.0 - f0_layer_frac) - 1e-3,   PRIOR_FAC_F0) * layer_df
    print(f"[prior] f0 frac in layer = {f0_layer_frac:.3f}; "
          f"f0 prior = inj + [{f0_safe_lo:+.3e}, {f0_safe_hi:+.3e}] Hz "
          f"(<= +/- {PRIOR_FAC_F0:.2f} * layer_df) -- m_floor stays put",
          flush=True)
    print(f"[prior] angles: phi0 +/- {PRIOR_FAC_PHI0:.2f} rad, "
          f"cosinc +/- {PRIOR_FAC_COSINC:.2f}, "
          f"psi +/- {PRIOR_FAC_PSI:.2f} rad, "
          f"lam +/- {PRIOR_FAC_LAM:.2f} rad, "
          f"sinbeta +/- {PRIOR_FAC_SINBETA:.2f}", flush=True)
    print(f"[prior] amp +/- {PRIOR_FAC_AMP*100:.1f}%, "
          f"fdot +/- {PRIOR_FAC_FDOT:.1e}", flush=True)

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
        5: uniform_dist(psi_inj - PRIOR_FAC_PSI,
                        psi_inj + PRIOR_FAC_PSI),
        6: uniform_dist(lam_inj - PRIOR_FAC_LAM,
                        lam_inj + PRIOR_FAC_LAM),
        7: uniform_dist(max(-1.0, np.sin(beta_inj) - PRIOR_FAC_SINBETA),
                        min( 1.0, np.sin(beta_inj) + PRIOR_FAC_SINBETA)),
    })}

    # Tight starting cloud around the injection.
    sf = START_FACTOR
    gen_dist = {"gb": ProbDistContainer({
        0: uniform_dist(amp_inj * (1.0 - sf),  amp_inj * (1.0 + sf)),
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

    # Backend (HDF or in-memory).
    if BACKEND_PATH and os.path.exists(BACKEND_PATH):
        file_backend = HDFBackend(BACKEND_PATH)
        start_state = file_backend.get_last_sample()
        print(f"[backend] resuming from {BACKEND_PATH}", flush=True)
    else:
        file_backend = HDFBackend(BACKEND_PATH) if BACKEND_PATH else None
        start_state = State(
            {"gb": gen_dist["gb"].rvs(size=(NTEMPS, NWALKERS, 1))}
        )
        if BACKEND_PATH:
            print(f"[backend] new HDF backend at {BACKEND_PATH}", flush=True)

    # ---- Vectorized gradient (central-difference C++) for NUTS ----
    # sampled-basis -> phys-basis Jacobian. Only cosinc and sinbeta are
    # non-identity; phi0/psi/lam/amp/f0/fdot/fddot pass through.
    #
    # phys index : 0 amp 1 f0 2 fdot 3 fddot 4 phi0 5 inc 6 psi 7 lam 8 beta
    # samp index : 0 amp 1 f0 2 fdot 3 phi0 4 cosinc 5 psi 6 lam 7 sinbeta
    SAMP_TO_PHYS = [0, 1, 2, 4, 5, 6, 7, 8]   # samp k -> phys idx
    PARAM_EPS_PHYS = np.array([
        amp_inj * 1e-3,    # 0 amp
        1e-3 * layer_df,    # 1 f0
        1e-18,              # 2 fdot
        0.0,                # 3 fddot (frozen)
        1e-3,               # 4 phi0
        1e-3,               # 5 inc
        1e-3,               # 6 psi
        1e-3,               # 7 lam
        1e-3,               # 8 beta
    ], dtype=np.float64)

    def grad_logl_vec(x, transform_fn=None, **_kw):
        """Central-difference gradient of logL in the SAMPLED basis.

        x: (N, 8). Returns (N, 8) grad_logL_sampled.
        Cost per source: 1 central + 2*8 perturbed = 17 get_ll calls.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        N = x_arr.shape[0]
        if transform_fn is not None:
            phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
        else:
            phys = x_arr
        params_cand_all = np.ascontiguousarray(phys.astype(np.float64))
        data_index_all = np.zeros(N, dtype=np.int32)
        grad_phys = np.zeros((N, 9), dtype=np.float64)
        d_h_central = np.zeros(N, dtype=np.float64)
        h_h_central = np.zeros(N, dtype=np.float64)
        cpp.gb_signal_het_get_ll_grad_in_kernel(
            tdi_wrap,
            grad_phys, d_h_central, h_h_central,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_ref_all, data_index_all,
            PARAM_EPS_PHYS,
            N, 1,
            9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            Tobs, t_start,
            3, 0, N_SPARSE_FD,
            TUKEY_ALPHA, MAX_R,
        )
        # Chain rule into sampled basis. Most entries are identity;
        # cosinc and sinbeta carry the only Jacobian factors.
        grad_samp = np.zeros((N, 8), dtype=np.float64)
        for k_samp, k_phys in enumerate(SAMP_TO_PHYS):
            grad_samp[:, k_samp] = grad_phys[:, k_phys]
        # cosinc at samp idx 4 -> phys inc = arccos(cosinc). d(arccos(c))/dc
        # = -1/sqrt(1 - c^2). Clip cos to avoid division at the poles.
        cosinc = x_arr[:, 4]
        sin_inc = np.sqrt(np.clip(1.0 - cosinc * cosinc, 1e-12, None))
        grad_samp[:, 4] *= -1.0 / sin_inc
        # sinbeta at samp idx 7 -> phys beta = arcsin(sinbeta). d/ds = +1/sqrt(1-s^2)
        sinbeta = x_arr[:, 7]
        cos_beta = np.sqrt(np.clip(1.0 - sinbeta * sinbeta, 1e-12, None))
        grad_samp[:, 7] *= 1.0 / cos_beta
        return grad_samp

    # ---- JAX gradient path (analytic via jax.jit(jax.grad)) ----
    # ~4x faster than C++ central-diff per binary; the only path that
    # makes tempered NUTS at 10 temps tractable in this session.
    NUTS_BACKEND = os.environ.get("NUTS_BACKEND", "cpp").lower()  # cpp or jax

    def make_grad_logl_jax():
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        # Pytree-register OrbitsWrapJAX so the large jnp arrays flow
        # through jit as dynamic args (avoids XLA constant-folding
        # explosion on the 5.5M-element LTT table -- known JAX gotcha).
        from lisatools.jax.orbits import OrbitsWrapJAX
        from jax.tree_util import register_pytree_node
        def _flatten(o):
            return (
                (o.n, o.ltt, o.x),
                (o.ltt_t0, o.ltt_dt, o.ltt_N, o.sc_t0, o.sc_dt, o.sc_N,
                 o.armlength, o.links, o.sc_r, o.sc_e, o._link_to_ind),
            )
        def _unflatten(aux, children):
            obj = OrbitsWrapJAX.__new__(OrbitsWrapJAX)
            obj.n, obj.ltt, obj.x = children
            (obj.ltt_t0, obj.ltt_dt, obj.ltt_N, obj.sc_t0, obj.sc_dt,
             obj.sc_N, obj.armlength, obj.links, obj.sc_r, obj.sc_e,
             obj._link_to_ind) = aux
            return obj
        try:
            register_pytree_node(OrbitsWrapJAX, _flatten, _unflatten)
        except ValueError:
            pass

        from gbgpu.jax.wdm.signal_het_kernels import (
            gb_signal_het_get_ll_in_kernel_jax,
        )
        from gbgpu.jax.sources.ucb import JaxUCBSource
        from lisatools.jax.response.tdi_config import TDIConfigWrapJAX

        # Configure CPU orbits (needed to extract pycppdetector_args).
        try:
            orbits.configure(t_arr=t_arr, dt=dt, linear_interp_setup=True)
        except TypeError:
            orbits.configure(t_arr=t_arr)
        source_jax     = JaxUCBSource(t_ref=t_start)
        orbits_jax_obj = OrbitsWrapJAX(*orbits.pycppdetector_args)
        tdi_config_jax = TDIConfigWrapJAX(*tdi_config.pytdiconfig_args)

        c0_j  = jnp.asarray(c0_sparse_all)
        A0_j  = jnp.asarray(A0_all)
        A1_j  = jnp.asarray(A1_all)
        B0_j  = jnp.asarray(B0_all)
        B1_j  = jnp.asarray(B1_all)
        win_j = jnp.asarray(window_full)
        nsp_j = jnp.asarray(n_sparse_local)
        pref_j = jnp.asarray(params_ref_all)
        didx_j = jnp.asarray(np.zeros(1, dtype=np.int32))

        def _logL_one(params, orbits_jax):
            d_h, h_h = gb_signal_het_get_ll_in_kernel_jax(
                params[None, :],
                c0_j, A0_j, A1_j, B0_j, B1_j,
                win_j, nsp_j, pref_j, didx_j,
                source_jax, orbits_jax, tdi_config_jax,
                nparams=9, f0_idx=1,
                Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
                Nt_layer=NT_LAYER, N_sparse_t=N_sparse_t, stride=stride,
                ind_min_t=ind_min_t, ind_min_f=ind_min_f,
                m_active_half_width=2,
                layer_df=layer_df, dt=dt,
                T_obs=Tobs, t_start=t_start,
                nchannels=3, tdi_type=0,
                N_sparse_fd=N_SPARSE_FD,
                tukey_alpha=TUKEY_ALPHA,
                max_r=MAX_R,
            )
            return -0.5 * d_d_lt + d_h[0] - 0.5 * h_h[0]

        # Vmap over batch dim, jit the grad.
        grad_one = jax.jit(jax.grad(_logL_one, argnums=0))
        grad_batched = jax.jit(jax.vmap(jax.grad(_logL_one, argnums=0),
                                         in_axes=(0, None)))

        def grad_logl_jax_vec(x, transform_fn=None, **_kw):
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr[None, :]
            if transform_fn is not None:
                phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
            else:
                phys = x_arr
            phys_j = jnp.asarray(phys.astype(np.float64))
            grad_phys = np.asarray(
                grad_batched(phys_j, orbits_jax_obj).block_until_ready()
            )                                                    # (N, 9)
            # Chain rule into sampled basis -- same as the C++ path.
            N = x_arr.shape[0]
            grad_samp = np.zeros((N, 8), dtype=np.float64)
            for k_samp, k_phys in enumerate(SAMP_TO_PHYS):
                grad_samp[:, k_samp] = grad_phys[:, k_phys]
            cosinc = x_arr[:, 4]
            sin_inc = np.sqrt(np.clip(1.0 - cosinc * cosinc, 1e-12, None))
            grad_samp[:, 4] *= -1.0 / sin_inc
            sinbeta = x_arr[:, 7]
            cos_beta = np.sqrt(np.clip(1.0 - sinbeta * sinbeta, 1e-12, None))
            grad_samp[:, 7] *= 1.0 / cos_beta
            return grad_samp

        return grad_logl_jax_vec

    # ---- Build the requested move ----
    if MOVES == "stretch":
        moves = StretchMove(live_dangerously=True)
        print(f"[mcmc] move: stretch", flush=True)
    else:
        # NUTS with the C++ central-difference grad. Cost per leapfrog =
        # 17 get_ll evals per source; keep max_tree_depth modest.
        # NUTSMove.scale = per-parameter natural step (mass = 1/scale^2).
        nuts_scale = np.array([
            amp_inj * 1e-3,    # amp
            layer_df * 1e-3,    # f0
            1e-15,              # fdot
            1e-2,               # phi0
            1e-2,               # cosinc
            1e-2,               # psi
            1e-2,               # lam
            1e-2,               # sinbeta
        ], dtype=float)
        # NUTSMove takes a grad_log_like_fn(x) -> grad; bind transform_fn
        # into a closure so the chain rule still routes through tc.
        if NUTS_BACKEND == "jax":
            print(f"[mcmc] NUTS_BACKEND=jax -- building jax.jit(jax.grad)...",
                  flush=True)
            t_jax0 = time.perf_counter()
            grad_logl_jax_fn = make_grad_logl_jax()
            t_jax1 = time.perf_counter()
            print(f"[mcmc] JAX grad builder done in {t_jax1-t_jax0:.1f}s "
                  f"(first call triggers JIT compile ~5 min)", flush=True)
            def _grad_with_tc(x, **kwargs):
                return grad_logl_jax_fn(x, transform_fn=tc)
        else:
            print(f"[mcmc] NUTS_BACKEND=cpp -- using C++ central-difference grad",
                  flush=True)
            def _grad_with_tc(x, **kwargs):
                return grad_logl_vec(x, transform_fn=tc)
        NUTS_ADAPT = os.environ.get("NUTS_ADAPT", "0") == "1"
        NUTS_N_ADAPT = int(os.environ.get("NUTS_N_ADAPT", "50"))
        moves = NUTSMove(
            grad_log_like_fn=_grad_with_tc,
            ndim=len(sampled_basis),
            scale=nuts_scale,
            step_size=NUTS_STEP_SIZE,
            max_tree_depth=NUTS_MAX_TREE,
            adapt_step_size=NUTS_ADAPT,
            n_adapt=NUTS_N_ADAPT,
            target_accept=0.8,
            live_dangerously=True,
        )
        if NUTS_ADAPT:
            print(f"[mcmc] NUTS_ADAPT=True  n_adapt={NUTS_N_ADAPT}  target_accept=0.8",
                  flush=True)
        print(f"[mcmc] move: NUTS step_size={NUTS_STEP_SIZE} "
              f"max_tree_depth={NUTS_MAX_TREE}", flush=True)

    sampler = EnsembleSampler(
        NWALKERS,
        ndims,
        logl_vec,
        priors,
        tempering_kwargs=dict(ntemps=NTEMPS),
        kwargs=dict(transform_fn=tc),
        moves=moves,
        branch_names=["gb"],
        periodic=periodic_container,
        backend=file_backend,
        vectorize=True,
    )

    # Inject true params into the cloud-evaluation pre-step. At x_cand = x_inj
    # the analytic logL is exactly 0 (d_h = h_h = <d|d>, so
    # -0.5*d_d + d_h - 0.5*h_h = 0). The C++ result should be ~0 modulo the
    # heterodyne reconstruction residual.
    inj_in_sampled = np.array([
        amp_inj, f0_inj, fdot_inj, phi0_inj,
        np.cos(inc_inj), psi_inj, lam_inj, np.sin(beta_inj),
    ], dtype=float)
    ll_inj_vec = logl_vec(inj_in_sampled[None, :], transform_fn=tc)[0]
    print(f"[check] logL @ injection: C++ = {ll_inj_vec:+.6e}  "
          f"(analytic truth = 0; residual is heterodyne floor)",
          flush=True)
    print(f"[check] noise-only logL = -0.5 <d|d> = {-0.5 * d_d_lt:+.6e}",
          flush=True)

    # Initial likelihoods for the starting cloud.
    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(
            start_state.branches_coords)
        start_state.log_like  = sampler.compute_log_like(
            start_state.branches_coords, logp=start_state.log_prior)[0]
    print(f"[start] log_like.mean = {np.asarray(start_state.log_like).mean():.6e} "
          f"log_like.min/max = "
          f"{np.asarray(start_state.log_like).min():.6e} / "
          f"{np.asarray(start_state.log_like).max():.6e}", flush=True)

    # Timer.
    if TIMER_REPEATS > 0:
        bench_x = gen_dist["gb"].rvs(size=(NTEMPS * NWALKERS,))
        _ = logl_vec(bench_x, transform_fn=tc)   # warmup
        t0 = time.perf_counter()
        for _ in range(TIMER_REPEATS):
            _ = logl_vec(bench_x, transform_fn=tc)
        t1 = time.perf_counter()
        per_call_ms = (t1 - t0) / TIMER_REPEATS * 1e3
        per_src_us  = per_call_ms * 1000.0 / bench_x.shape[0]
        print(f"[timer] logl_vec batch={bench_x.shape[0]}: "
              f"{per_call_ms:.2f} ms / call  "
              f"({per_src_us:.1f} us / source)", flush=True)

    # ----- Run MCMC ----- #
    print(f"\n[mcmc] sampling NSTEPS={NSTEPS} burnin={BURNIN} "
          f"nwalkers={NWALKERS} ntemps={NTEMPS}", flush=True)
    t0 = time.perf_counter()
    sampler.run_mcmc(start_state, NSTEPS, burn=BURNIN, progress=PROGRESS,
                     thin_by=1)
    t1 = time.perf_counter()
    print(f"[mcmc] elapsed = {t1 - t0:.2f} s  "
          f"({(t1 - t0) / NSTEPS * 1000.0:.2f} ms / step)", flush=True)

    # ----- Diagnostics ----- #
    samples = sampler.get_chain()["gb"]   # (nsteps, ntemps, nwalkers, 1, ndim)
    log_like = sampler.get_log_like()      # (nsteps, ntemps, nwalkers)
    acc_frac = sampler.acceptance_fraction
    print(f"[diag] acceptance_fraction.mean = "
          f"{np.asarray(acc_frac).mean():.3f}", flush=True)
    final_ll = np.asarray(log_like[-1, 0])
    print(f"[diag] final log_like (cold) mean={final_ll.mean():.6e} "
          f"min={final_ll.min():.6e} max={final_ll.max():.6e}", flush=True)
    print(f"[diag] best-ever log_like = "
          f"{np.asarray(log_like[:, 0]).max():.6e}", flush=True)

    flat = samples[:, 0].reshape(-1, len(sampled_basis))  # cold temp
    mean_post = flat.mean(axis=0)
    std_post  = flat.std(axis=0)
    print(f"\n  {'param':>9s} {'inj':>14s} {'post mean':>14s} "
          f"{'post std':>11s} {'(mean - inj)/std':>16s}", flush=True)
    for k, name in enumerate(sampled_basis):
        s = std_post[k]
        bias = (mean_post[k] - inj_in_sampled[k]) / max(s, 1e-30)
        print(f"  {name:>9s} {inj_in_sampled[k]:>+14.6e} "
              f"{mean_post[k]:>+14.6e} {s:>11.3e} {bias:>+16.3f}",
              flush=True)

    # ---- Corner plot of cold chain ----
    if CORNER_PATH:
        # Drop the first 20% of samples as "burn" inside the corner plot.
        n_burn = max(1, int(0.2 * samples.shape[0]))
        flat_burn = samples[n_burn:, 0].reshape(-1, len(sampled_basis))
        try:
            fig = corner.corner(
                flat_burn,
                labels=sampled_basis,
                truths=inj_in_sampled,
                show_titles=True,
                title_kwargs={"fontsize": 9},
                label_kwargs={"fontsize": 9},
                quantiles=[0.16, 0.5, 0.84],
                color="C0" if MOVES == "stretch" else "C1",
            )
            fig.suptitle(
                f"GB signal-het MCMC ({MOVES}, nsteps={NSTEPS}, "
                f"nwalkers={NWALKERS}, ntemps={NTEMPS}, snr={snr_inj:.0f})",
                fontsize=10, y=1.02,
            )
            fig.savefig(CORNER_PATH, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"\n[corner] saved -> {CORNER_PATH}", flush=True)
        except Exception as e:
            print(f"[corner] FAILED: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
