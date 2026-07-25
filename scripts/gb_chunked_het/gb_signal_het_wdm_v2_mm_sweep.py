#!/usr/bin/env python
"""Nt_layer vs mm5/mm2 sweep for ``GBSparseComplexWDMGen`` (v2 polyphase).

For each Nt_layer in the sweep, draw N_DRAWS GB priors (same prior as
``gb_chunked_prior_draws.py``), build:

  1. The lisatools dense REAL-WDM injection ``inj`` at draw x_inj.
  2. The v2 sparse COMPLEX-WDM coefs ``c1_sparse`` at x_inj (active m-band,
     5 layers around m_floor).
  3. A reconstructed dense COMPLEX-WDM template by linearly interpolating
     ``c1_sparse`` across n within the active m-band (other m left at zero).
  4. Convert complex -> real WDM via ``Re(.)`` (lisatools convention --
     ``Re(complex_wdm) == real_wdm`` term-for-term, see
     ``LISAanalysistools/src/lisatools/domains.py:1169-1177``).

Then compute the canonical narrowband mismatches mm5 / mm2 vs ``inj``:

  - mm5: band ``[f0 - 3*layer_df, f0 + 2*layer_df]`` (5 layers around f0)
  - mm2: band ``[(m_floor - 0.5)*layer_df, (m_floor + 1.5)*layer_df]``

Polyphase is mathematically exact at the sparse n positions; the mm
quantifies the LINEAR-INTERPOLATION error between sparse n points. Plot
shows how aggressive stride (= ``Nt // Nt_layer``) can go while still
giving accurate dense templates.

Env vars:
  N_DRAWS              default 50
  NT_LAYER_LIST        comma-separated list of Nt_layer values
                       (must divide Nt; default uses [16, 32, 64, 128, 256, 512, Nt])
  NF, NT, EDGE_CUT     WDM grid; defaults match gb_chunked_prior_draws.py
  SNR_MIN, SNR_MAX     SNR rejection window for prior draws
  SEED                 RNG seed
  OUT_PNG              output plot path
  CACHE                if set, save raw mm5/mm2 arrays to NPZ alongside the plot
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
# GBTDIonTheFly re-prefixes its backend lookup to the ``gbgpu_*`` family
# (Phase 3L.7k); importing gbgpu registers those backends so force_backend
# resolution succeeds.
import gbgpu  # noqa: F401  (side-effect: registers gbgpu_cpu/cuda backends)
from lisatools.response.tdionfly import GBTDIonTheFly

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen


def build_gb_prior(*, A_lims, f0_lims_hz, fdot_lims, beta_lims=None):
    """Inline copy of ``gb_lookup_prior_draws.build_gb_prior`` (decoupled so
    we don't transitively import the stale lookup-table dependency)."""
    sampled_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    full_basis = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    parameter_transforms = {
        "A": np.exp,
        "f0": (lambda x: x * 1e-3),
        "cos_iota": np.arccos,
        "sin_beta": np.arcsin,
    }
    tc = TransformContainer(
        input_basis=sampled_basis, output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={"fddot": 0.0},
    )
    delta_safe = 1e-5
    iota_lims = (0.0 + delta_safe, np.pi - delta_safe)
    if beta_lims is None:
        beta_lims = (-0.05, 0.05)
    priors_gb = {
        "A":        uniform_dist(*(np.log(np.asarray(A_lims)))),
        "f0":       uniform_dist(f0_lims_hz[0] * 1e3, f0_lims_hz[1] * 1e3),
        "fdot":     uniform_dist(fdot_lims[0], fdot_lims[1]),
        "phi0":     uniform_dist(0.0, 2.0 * np.pi),
        # np.cos maps the increasing iota range to a DECREASING pair; newer
        # eryn validates max > min, so sort.
        "cos_iota": uniform_dist(*np.sort(np.cos(iota_lims))),
        "psi":      uniform_dist(0.0, np.pi),
        "lam":      uniform_dist(0.0, 2.0 * np.pi),
        "sin_beta": uniform_dist(*np.sin(beta_lims)),
    }
    return ProbDistContainer(priors_gb), tc


def linear_interp_sparse_to_dense(
    c_sparse: np.ndarray,
    n_sparse_local: np.ndarray,
    stride: int,
    Nt_active: int,
) -> np.ndarray:
    """Linearly interpolate sparse complex c[c, m, b] -> dense c[c, m, n].

    ``c_sparse``        : (nch, Nm, N_sparse) complex
    ``n_sparse_local``  : (N_sparse,) int, dense indices of the sparse bins
    ``stride``          : uniform spacing of ``n_sparse_local``
    ``Nt_active``       : output dense length

    Returns (nch, Nm, Nt_active) complex. Out-of-range pixels (left of the
    first / right of the last sparse bin) use linear extrapolation from the
    nearest interior segment.
    """
    nch, Nm, N_sparse = c_sparse.shape
    n_dense = np.arange(Nt_active)
    b_idx = np.searchsorted(n_sparse_local, n_dense, side="right") - 1
    b_idx = np.clip(b_idx, 0, N_sparse - 2)
    offsets = (n_dense - n_sparse_local[b_idx]).astype(float) / float(stride)
    # left = c_sparse[..., b_idx]; right = c_sparse[..., b_idx + 1]
    left = c_sparse[:, :, b_idx]
    right = c_sparse[:, :, b_idx + 1]
    out = left * (1.0 - offsets[None, None, :]) + right * offsets[None, None, :]
    return out


def build_v2_real_template(
    sparse_gen: GBSparseComplexWDMGen,
    fd_rfft: np.ndarray,
    params_cand: np.ndarray,
    params_ref: np.ndarray,
    Nf_active: int,
    Nt_active: int,
    c0_dense_complex: np.ndarray,
    c0_sparse_complex: np.ndarray,
    dt: float,
    Nf: int,
    ind_min_t: int,
) -> np.ndarray:
    """Return ``(3, Nf_active, Nt_active)`` real-WDM template via v2 heterodyne.

    Heterodyne reconstruction with analytic carrier de-rotation:

      r(t) = (smooth amp ratio)(t) * exp(+i 2*pi*Df0*t + i*pi*Dfdot*t^2)

    where ``Df0 = f0_cand - f0_ref`` and ``Dfdot = fdot_cand - fdot_ref``
    are KNOWN from the candidate params. We:

      1. Polyphase v2 -> c1_sparse at active m and sparse n.
      2. r_sparse = c1_sparse / c0_sparse.
      3. r_demod_sparse = r_sparse * exp(-i*phase_pred(n_sparse)) -- smooth.
      4. Linearly interpolate r_demod across n.
      5. r_dense = r_demod_dense * exp(+i*phase_pred(n_dense)) -- re-rotated.
      6. c1_dense = r_dense * c0_dense.
      7. Re -> real-WDM template.
    """
    c1_sparse, m_local = sparse_gen.sparse_from_rfft(fd_rfft, params_cand)
    c0_active_sparse = c0_sparse_complex[:, m_local, :]
    c0_mag = np.abs(c0_active_sparse)
    floor = 1e-12 * c0_mag.max(axis=-1, keepdims=True)
    floor = np.maximum(floor, 1e-300)
    mask = c0_mag > floor
    denom = np.where(mask, c0_active_sparse, 1.0)
    r_sparse = np.where(mask, c1_sparse / denom, 0.0 + 0.0j)

    # Analytic carrier de-rotation from Df0, Dfdot.
    Df0 = float(params_cand[1] - params_ref[1])
    Dfdot = float(params_cand[2] - params_ref[2])
    layer_dt = Nf * dt
    n_sparse = sparse_gen.n_sparse_local
    t_n_sparse = (ind_min_t + n_sparse) * layer_dt
    phase_sparse = (2.0 * np.pi * Df0 * t_n_sparse
                    + np.pi * Dfdot * t_n_sparse ** 2)
    r_demod_sparse = r_sparse * np.exp(-1j * phase_sparse)[None, None, :]

    # Interpolate the SMOOTH de-rotated ratio.
    r_demod_dense_active = linear_interp_sparse_to_dense(
        r_demod_sparse, sparse_gen.n_sparse_local, sparse_gen.stride, Nt_active,
    )

    # Re-apply the carrier at dense n.
    n_dense = np.arange(Nt_active)
    t_n_dense = (ind_min_t + n_dense) * layer_dt
    phase_dense = (2.0 * np.pi * Df0 * t_n_dense
                   + np.pi * Dfdot * t_n_dense ** 2)
    r_dense_active = r_demod_dense_active * np.exp(1j * phase_dense)[None, None, :]

    # Reconstruct c1_dense = r * c0_dense (active m only).
    c0_active_dense = c0_dense_complex[:, m_local, :]
    c1_dense_active = r_dense_active * c0_active_dense
    nch = c1_sparse.shape[0]
    tpl_complex = np.zeros((nch, Nf_active, Nt_active), dtype=np.complex128)
    tpl_complex[:, m_local, :] = c1_dense_active
    return tpl_complex.real


def compute_mm(
    wdm_inj_arr: np.ndarray,
    tpl_arr: np.ndarray,
    wdm_set: WDMSettings,
    f0: float,
    band: str,
    backend: str,
):
    """Compute canonical mm5 or mm2 narrowband mismatch.

    ``wdm_inj_arr`` and ``tpl_arr`` both have shape
    ``(nch, Nf_active, Nt_active)`` on ``wdm_set``'s active grid.
    """
    layer_df = wdm_set.layer_df
    m_floor = int(f0 / layer_df)
    if band == "mm5":
        # 5-layer symmetric band: m_floor +/- 2 (matches v2 active band).
        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=f0 - 2 * layer_df,
            max_freq=f0 + 2 * layer_df,
            force_backend=backend,
        )
    elif band == "mm2":
        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=(m_floor - 0.5) * layer_df,
            max_freq=(m_floor + 1 + 0.5) * layer_df,
            force_backend=backend,
        )
    else:
        raise ValueError(f"band must be 'mm5' or 'mm2', got {band!r}")
    lo = new_wdm_set.ind_min_f - wdm_set.ind_min_f
    hi = new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1
    inj_band = WDMSignal(wdm_inj_arr[:, lo:hi], new_wdm_set)
    tpl_band = WDMSignal(tpl_arr[:, lo:hi], new_wdm_set)
    analysis = AnalysisContainer(
        DataResidualArray(inj_band),
        XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1"),
    )
    mm = 1.0 - analysis.template_inner_product(
        DataResidualArray(tpl_band), normalize=True,
    )
    return float(mm)


def divisors_of(n: int) -> list:
    out = []
    for d in range(2, n + 1):
        if n % d == 0 and d % 2 == 0:
            out.append(d)
    return out


def main():
    backend = "cpu"

    # --- WDM grid -------------------------------------------------------
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    wavelet_duration = Nf * dt
    Nobs = Nf * Nt
    Tobs = Nt * wavelet_duration
    EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start
    min_time = EDGE_CUT * wavelet_duration
    max_time = (Nt - EDGE_CUT) * wavelet_duration
    min_freq = float(os.environ.get("MIN_FREQ_HZ", 1e-4))
    max_freq = float(os.environ.get("MAX_FREQ_HZ", 35.0e-3))

    print(
        f"[grid] Nf={Nf} Nt={Nt} dt={dt}s  Tobs={Tobs:.3e}s  "
        f"layer_dt={wavelet_duration:.3e}s",
        flush=True,
    )

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)

    N_inj_grid = 16384
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], N_inj_grid)
    gb_gen = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
    )

    def real_td_cb(params9):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params9
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    # Tukey window in the predetermined heterodyne alpha range
    # (ALPHA_HET_NARROW=0.05 for N_sparse < 512, ALPHA_HET_WIDE=0.01 else),
    # from lisa-on-gpu/src/fastlisaresponse/jax/wdm/fast_inner_heterodyne.py.
    # Applied to BOTH the data wdmtransform AND the rfft fed to v2 so they
    # see the same windowed signal.
    from scipy.signal.windows import tukey as _tukey
    TUKEY_ALPHA = float(os.environ.get("TUKEY_ALPHA", "0.05"))
    if TUKEY_ALPHA > 0:
        window = _tukey(Nobs, alpha=TUKEY_ALPHA).astype(float)
    else:
        window = np.ones(Nobs)
    print(f"[window] Tukey alpha={TUKEY_ALPHA} "
          f"(predetermined range: 0.01 <= alpha <= 0.05)", flush=True)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    print(
        f"[wdm]  ind_min_t={ind_min_t} Nt_active={Nt_active} Nf_active={Nf_active} "
        f"layer_df={layer_df:.3e}Hz",
        flush=True,
    )

    # --- prior ---------------------------------------------------------
    SEED = int(os.environ.get("SEED", "12345"))
    np.random.seed(SEED)
    buffer_layers = 7
    f0_lo_default = (wdm_set_real.ind_min_f + buffer_layers) * layer_df
    f0_hi_default = (wdm_set_real.ind_max_f - buffer_layers) * layer_df
    f0_lo_hz = float(os.environ.get("F0_LO_HZ", f0_lo_default))
    f0_hi_hz = float(os.environ.get("F0_HI_HZ", f0_hi_default))
    fdot_max = float(os.environ.get("FDOT_MAX", "1e-15"))
    A_lims = (float(os.environ.get("A_LO", "1e-23")),
              float(os.environ.get("A_HI", "1e-20")))
    prior, tc = build_gb_prior(
        A_lims=A_lims, f0_lims_hz=(f0_lo_hz, f0_hi_hz),
        fdot_lims=(-fdot_max, fdot_max), beta_lims=None,
    )

    # --- Nt_layer sweep list ------------------------------------------
    nt_layer_env = os.environ.get("NT_LAYER_LIST", "")
    if nt_layer_env:
        nt_layer_list = [int(s) for s in nt_layer_env.split(",")]
    else:
        # Reasonable default sweep -- even divisors of Nt up to Nt itself.
        cand = [d for d in divisors_of(Nt) if d % 2 == 0]
        # Pick a log-ish subset: 16, 32, 64, 128, 256, 512, 1280, Nt (filter to actual divisors).
        wanted = [16, 32, 64, 128, 256, 512, 1280, Nt]
        nt_layer_list = [v for v in wanted if v in cand or v == Nt]
        if not nt_layer_list:
            nt_layer_list = cand[: min(8, len(cand))]
    # Validate
    for ntl in nt_layer_list:
        if Nt % ntl != 0:
            raise ValueError(f"Nt_layer={ntl} doesn't divide Nt={Nt}")
    print(f"[sweep] Nt_layer list: {nt_layer_list}", flush=True)

    # --- prior draws (filter by SNR window) ----------------------------
    N_DRAWS = int(os.environ.get("N_DRAWS", "50"))
    SNR_MIN = float(os.environ.get("SNR_MIN", "5.0"))
    SNR_MAX = float(os.environ.get("SNR_MAX", "1100.0"))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", "500"))
    print(
        f"[draws] N_DRAWS={N_DRAWS}  SNR window=[{SNR_MIN}, {SNR_MAX}]",
        flush=True,
    )

    # Heterodyne perturbation scale for x_cand = x_ref + Delta.
    # Delta is applied to f0 (param idx 1) as a fraction of layer_df --
    # so the template parameters differ from the reference by some MCMC-
    # proposal-scale shift, forcing r(t) to oscillate in n and exercising
    # linear interpolation.
    DF0_FRAC = float(os.environ.get("DF0_FRAC", "0.05"))  # default 5% of layer_df
    print(f"[heterodyne] x_cand = x_ref + Delta f0 = {DF0_FRAC} * layer_df", flush=True)

    draws = []                                                  # list of (params_ref, params_cand, fd_rfft_cand, wdm_inj_real_arr_cand, c0_dense_complex_ref, snr_ref)
    t0_draws = time.perf_counter()
    sens_mat_full = None
    for di in range(N_DRAWS):
        chosen = None
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_i = tc.both_transforms(x_samp.copy())[0]
            td_i = real_td_cb(params_i)
            wdm_inj_real = TDSignal(td_i, settings=td_set).transform(
                wdm_set_real, window=window,
            )
            inj_data_arr = DataResidualArray(wdm_inj_real)
            if sens_mat_full is None:
                sens_mat_full = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1",
                )
            analysis = AnalysisContainer(inj_data_arr, sens_mat_full)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                # x_inj IS the reference x_ref (the heterodyne is centered on the
                # injection's true params). c0 = lisatools dense complex WDM(x_ref).
                wdm_ref_complex = np.asarray(
                    TDSignal(td_i, settings=td_set).transform(
                        wdm_set_complex, window=window,
                    ).arr
                ).copy()
                # x_cand = x_ref + Delta_f0; we then build the lisatools "ground
                # truth" template at x_cand (this is the 'data' for mm5/mm2),
                # and feed x_cand to v2 to reconstruct via heterodyne.
                params_cand = params_i.copy()
                params_cand[1] = params_i[1] + DF0_FRAC * layer_df
                td_cand = real_td_cb(params_cand)
                # Match the Tukey applied to data wdmtransform so v2's polyphase
                # input is consistent with what the dense reference sees.
                fd_rfft_cand = np.fft.rfft(
                    np.asarray(td_cand) * window, axis=-1,
                )
                wdm_cand_real = TDSignal(td_cand, settings=td_set).transform(
                    wdm_set_real, window=window,
                )
                wdm_cand_arr = np.asarray(wdm_cand_real.arr).copy()
                chosen = (params_i, params_cand, fd_rfft_cand,
                          wdm_cand_arr, wdm_ref_complex, snr_i)
                break
        if chosen is None:
            print(f"[warn] draw {di}: exhausted reject; keeping last (snr={snr_i:.2f})",
                  flush=True)
        draws.append(chosen)
        if (di + 1) % 5 == 0 or di == N_DRAWS - 1:
            print(f"  draw {di+1}/{N_DRAWS}  snr={chosen[5]:.1f}", flush=True)
    print(
        f"[draws] elapsed {time.perf_counter() - t0_draws:.1f}s for {N_DRAWS} draws",
        flush=True,
    )

    # --- sweep --------------------------------------------------------
    mm5_arr = np.zeros((len(nt_layer_list), N_DRAWS), dtype=float)
    mm2_arr = np.zeros((len(nt_layer_list), N_DRAWS), dtype=float)
    n_sparse_arr = np.zeros(len(nt_layer_list), dtype=int)
    stride_arr = np.zeros(len(nt_layer_list), dtype=int)

    for li, ntl in enumerate(nt_layer_list):
        t0_l = time.perf_counter()
        sparse_gen = GBSparseComplexWDMGen(
            real_td_callable=real_td_cb,
            wdm_set_complex=wdm_set_complex,
            data_dt=dt,
            ind_min_t=ind_min_t,
            Nt_active=Nt_active,
            Nt_layer=int(ntl),
            m_active_half_width=2,
        )
        n_sparse_arr[li] = sparse_gen.N_sparse_t
        stride_arr[li] = sparse_gen.stride

        for di, (params_ref, params_cand, fd_rfft_cand, wdm_cand_arr,
                 wdm_ref_complex, snr_i) in enumerate(draws):
            # c0_sparse at sparse positions (reference at x_ref)
            c0_sparse = wdm_ref_complex[:, :, sparse_gen.n_sparse_local]
            # v2 reconstructs the dense template at x_cand via heterodyne
            # (polyphase produces c1_sparse, then r=c1/c0 at sparse n,
            #  linear interp of r, multiply by c0_dense).
            tpl_real = build_v2_real_template(
                sparse_gen, fd_rfft_cand, params_cand, params_ref,
                Nf_active, Nt_active,
                wdm_ref_complex, c0_sparse, dt, Nf, ind_min_t,
            )
            f0 = float(params_cand[1])
            mm5_arr[li, di] = compute_mm(
                wdm_cand_arr, tpl_real, wdm_set_real, f0, "mm5", backend,
            )
            mm2_arr[li, di] = compute_mm(
                wdm_cand_arr, tpl_real, wdm_set_real, f0, "mm2", backend,
            )
        med5 = float(np.median(mm5_arr[li]))
        p90_5 = float(np.percentile(mm5_arr[li], 90))
        med2 = float(np.median(mm2_arr[li]))
        p90_2 = float(np.percentile(mm2_arr[li], 90))
        print(
            f"[sweep] Nt_layer={ntl:5d} stride={sparse_gen.stride:5d} "
            f"N_sparse_t={sparse_gen.N_sparse_t:5d}  "
            f"mm5 med={med5:.2e} 90%={p90_5:.2e}  "
            f"mm2 med={med2:.2e} 90%={p90_2:.2e}  "
            f"({time.perf_counter() - t0_l:.1f}s)",
            flush=True,
        )

    # --- save + plot --------------------------------------------------
    out_dir = os.environ.get("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.environ.get(
        "OUT_PNG", os.path.join(out_dir, "v2_polyphase_mm_sweep.png"),
    )
    if os.environ.get("CACHE"):
        np.savez(
            os.path.join(out_dir, "v2_polyphase_mm_sweep.npz"),
            nt_layer_list=np.array(nt_layer_list),
            stride=stride_arr, n_sparse=n_sparse_arr,
            mm5=mm5_arr, mm2=mm2_arr,
            params_ref=np.array([d[0] for d in draws]),
            params_cand=np.array([d[1] for d in draws]),
            snr=np.array([d[5] for d in draws]),
            df0_frac=DF0_FRAC,
            Nf=Nf, Nt=Nt, dt=dt, layer_df=layer_df,
            Nf_active=Nf_active, Nt_active=Nt_active,
        )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    xs = np.array(nt_layer_list, dtype=float)
    strides = np.array(stride_arr, dtype=float)
    for ax, mm, label in [(axs[0], mm5_arr, "mm5"), (axs[1], mm2_arr, "mm2")]:
        med = np.median(mm, axis=1)
        p10 = np.percentile(mm, 10, axis=1)
        p90 = np.percentile(mm, 90, axis=1)
        maxv = mm.max(axis=1)
        for di in range(mm.shape[1]):
            ax.plot(xs, mm[:, di], color="C7", alpha=0.15, linewidth=0.5)
        ax.plot(xs, med, "C0o-", label="median")
        ax.fill_between(xs, p10, p90, color="C0", alpha=0.2,
                        label="10/90 pct")
        ax.plot(xs, maxv, "C3^--", label="max", alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nt_layer (stride = Nt / Nt_layer)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Nt_layer (N_draws={N_DRAWS}, m_half_width=2)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
        # Secondary x: stride
        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(ax.get_xlim())
        ticks = ax.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{int(Nt // t)}" if t > 0 else "" for t in ticks])
        ax2.set_xlabel("stride (dense pixels)")
    fig.suptitle(
        f"v2 polyphase signal-het: mm of v2 heterodyne reconstruction vs "
        f"dense direct\n"
        f"(x_cand = x_ref + {DF0_FRAC} * layer_df in f0; N_draws={N_DRAWS}, "
        f"Nt={Nt})"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out_png}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
