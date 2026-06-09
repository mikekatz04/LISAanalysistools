#!/usr/bin/env python
"""SOBBH prior-draws validation -- new ``bbhx.sobbhcomps`` canonical home.

Direct sibling of ``gb_chunked_het/gb_chunked_prior_draws.py`` but for
SOBBHs. Uses the new architectural split (sprint rule 2026-06-06):

* ``bbhx.sobbhcomps.SOBBHWDMComputations`` -- SOBBH chunked-het frontend.
* ``lisatools.chunked_het.WDMComputationsBase``  -- shared base class.
* No ``gbgpu`` import. SOBBHs live in BBHx now.

Pipeline per draw:

  1. Sample an 11-vector ``(m1, m2, s1, s2, distance, f_low, phi_c,
     inc, psi, lam, beta)`` uniformly from a SOBBH-sensible prior.
  2. Build the dense TD signal via ``lisatools.response.tdionfly.SOBBHTDIonTheFly``.
  3. Forward-transform to WDM (``TDSignal.transform(wdm_set)``).
  4. Build the chunked-het template via ``SOBBHWDMComputations.fill_global_wdm``.
  5. Compute three mismatches:
     - full active band,
     - 5-layer (f_low +/- a few layers around the carrier),
     - 2-layer (m_floor, m_floor+1) -- tightest around carrier.

Env knobs:
    N_DRAWS, SNR_MIN, SNR_MAX, NF, NT, NT_SUB, N_SPARSE, N_PAD,
    F_LOW_LO_HZ, F_LOW_HI_HZ, M1_LO, M1_HI, M2_LO, M2_HI,
    N_CP_SIG, N_CP_ORBIT, OUTPUT_PREFIX, BACKEND, SEED

Output:
    ``<OUTPUT_PREFIX>.h5``     -- per-draw params + SNR + 3 mismatches.
    ``<OUTPUT_PREFIX>.png``    -- mismatch histograms.

Run::

    BACKEND=cpu N_DRAWS=20 SNR_MIN=20 SNR_MAX=200 \\
      python sobbh_chunked_prior_draws_v2.py
"""
from __future__ import annotations

import os
import sys
import time

import h5py
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Force bbhx import to register the bbhx_<flavor> backend family.
import bbhx  # noqa: F401
from bbhx.sobbhcomps import SOBBHWDMComputations

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly


def _sample_sobbh_params(
    rng: np.random.Generator,
    f_low_lo: float, f_low_hi: float,
    m1_lo: float, m1_hi: float,
    m2_lo: float, m2_hi: float,
) -> np.ndarray:
    """Uniform-prior SOBBH parameter vector, length 11.

    Order matches SOBBHTDIonTheFly: (m1, m2, s1, s2, distance, f_low,
    phi_c, inc, psi, lam, beta).
    """
    m1 = rng.uniform(m1_lo, m1_hi)
    m2 = rng.uniform(m2_lo, min(m2_hi, m1))  # mass ordering
    s1 = rng.uniform(-0.5, 0.5)
    s2 = rng.uniform(-0.5, 0.5)
    distance = 10 ** rng.uniform(8.0, 9.5)         # 100 Mpc -- 3 Gpc, in pc
    f_low = rng.uniform(f_low_lo, f_low_hi)
    phi_c = rng.uniform(0.0, 2 * np.pi)
    cos_inc = rng.uniform(-1.0, 1.0)
    psi = rng.uniform(0.0, np.pi)
    lam = rng.uniform(0.0, 2 * np.pi)
    sin_beta = rng.uniform(-1.0, 1.0)
    return np.array([
        m1, m2, s1, s2, distance, f_low, phi_c,
        np.arccos(cos_inc), psi, lam, np.arcsin(sin_beta),
    ])


def main() -> int:
    N_DRAWS    = int(os.environ.get("N_DRAWS", "20"))
    SNR_MIN    = float(os.environ.get("SNR_MIN", "10.0"))
    SNR_MAX    = float(os.environ.get("SNR_MAX", "300.0"))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", "60"))
    SEED       = int(os.environ.get("SEED", "12345"))
    BACKEND    = os.environ.get("BACKEND", "cpu")

    Nf = int(os.environ.get("NF", "1460"))
    Nt = int(os.environ.get("NT", "2560"))
    Nt_sub  = int(os.environ.get("NT_SUB",   "256"))
    n_pad   = int(os.environ.get("N_PAD",    str(Nt_sub // 8)))
    N_sparse = int(os.environ.get("N_SPARSE", "256"))
    N_cp_sig   = int(os.environ.get("N_CP_SIG",   "0"))  # direct path
    N_cp_orbit = int(os.environ.get("N_CP_ORBIT", "0"))

    M1_LO = float(os.environ.get("M1_LO", "20.0"))
    M1_HI = float(os.environ.get("M1_HI", "50.0"))
    M2_LO = float(os.environ.get("M2_LO", "10.0"))
    M2_HI = float(os.environ.get("M2_HI", "40.0"))

    OUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "sobbh_chunked_prior_draws_v2")

    dt = 10.0
    Nobs = Nf * Nt
    t_start = int(0.5 * YRSID_SI / dt) * dt
    Tobs = Nobs * dt
    EC = 20  # edge cut
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        force_backend=BACKEND,
    )
    layer_df = wdm_set.layer_df
    t_arr = np.arange(Nobs) * dt + t_start
    td_set = TDSettings(N=Nobs, dt=dt, force_backend=BACKEND)

    orbits = ESAOrbits()
    tdi_config = TDIConfig("2nd generation")
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)

    sobbh_gen = SOBBHTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=BACKEND,
    )

    chunked = SOBBHWDMComputations(
        wdm_set,
        t_ref=t_start,
        Nt_sub=Nt_sub,
        n_pad=n_pad,
        N_sparse=N_sparse,
        N_cp_sig=N_cp_sig,
        N_cp_orbit=N_cp_orbit,
        orbits=orbits,
        tdi_config="2nd generation",
        force_backend=BACKEND,
        d_d=0.0,
        tdi_type="XYZ",
    )

    print(f"[init] backend     : {BACKEND}", flush=True)
    print(f"[init] WDM grid    : Nf={Nf} Nt={Nt} layer_df={layer_df*1e3:.4f} mHz", flush=True)
    print(f"[init] n_chunks    : {chunked.n_chunks}", flush=True)
    print(f"[init] T_chunk     : {chunked.T_chunk:.3e} s", flush=True)
    print(f"[init] tukey alpha : {chunked.resolved_tukey_alpha}", flush=True)
    print(f"[init] N_cp_sig    : {N_cp_sig}  N_cp_orbit: {N_cp_orbit}", flush=True)

    # Prior bounds on f_low keep the source band away from active edges.
    buf_layers = 7
    f_low_lo = (wdm_set.ind_min_f + buf_layers) * layer_df
    f_low_hi = (wdm_set.ind_max_f - buf_layers) * layer_df
    f_low_lo = float(os.environ.get("F_LOW_LO_HZ", f_low_lo))
    f_low_hi = float(os.environ.get("F_LOW_HI_HZ", f_low_hi))
    print(f"[init] f_low band  : [{f_low_lo*1e3:.4f}, {f_low_hi*1e3:.4f}] mHz", flush=True)
    print(f"[init] mass band   : m1=[{M1_LO}, {M1_HI}] M_sun", flush=True)
    print(f"[init] SNR window  : [{SNR_MIN}, {SNR_MAX}]", flush=True)
    print(f"[init] N_DRAWS     : {N_DRAWS}", flush=True)

    rng = np.random.default_rng(SEED)

    sens_mat = None
    snr_list  = []
    mm_full_list = []
    mm5_list = []
    mm2_list = []
    params_list = []
    d_d_list  = []

    t0 = time.perf_counter()
    attempt_total = 0
    for i in range(N_DRAWS):
        chosen = None
        for _ in range(MAX_REJECT):
            attempt_total += 1
            params_i = _sample_sobbh_params(
                rng, f_low_lo, f_low_hi, M1_LO, M1_HI, M2_LO, M2_HI,
            )
            try:
                inj_spline = sobbh_gen(
                    *params_i.reshape(11, 1),
                    convert_to_ra_dec=False, return_spline=True,
                )
                td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
                if not np.all(np.isfinite(td_inj)):
                    continue
                wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(
                    wdm_set, window=None,
                )
                injection = DataResidualArray(wdm_inj_sig)
                if sens_mat is None:
                    sens_mat = XYZ2SensitivityMatrix(
                        injection.data_res_arr.settings, model="scirdv1",
                    )
                analysis = AnalysisContainer(injection, sens_mat)
                d_d = float(np.real(analysis.inner_product()))
                snr = float(analysis.snr())
            except Exception:
                continue
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: rejected {MAX_REJECT} attempts -- skipping",
                  flush=True)
            continue
        params_i, wdm_inj_sig, analysis, d_d, snr = chosen

        # Build the chunked-het template on the full grid then slice to the
        # WDM active band.
        template_full = np.zeros((3, Nf, Nt), dtype=float)
        chunked.fill_global_wdm(
            params_i.reshape(1, 11), template_full,
            convert_to_ra_dec=False, factors=None,
        )
        tpl_active = template_full[:,
                                    wdm_set.ind_min_f:wdm_set.ind_max_f + 1,
                                    wdm_set.active_slice_t]
        tpl_wdm = WDMSignal(tpl_active, wdm_set)

        mm_full = float(1.0 - analysis.template_inner_product(
            DataResidualArray(tpl_wdm), normalize=True,
        ))

        # 5-layer mismatch -- [f_low - 3*layer_df, f_low + 2*layer_df].
        f_low = float(params_i[5])
        m_floor = int(f_low / layer_df)
        wdm_set_5 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=f_low - 3 * layer_df,
            max_freq=f_low + 2 * layer_df,
            force_backend=BACKEND,
        )
        wdm_inj_arr = np.asarray(wdm_inj_sig.arr)
        i_lo_5 = wdm_set_5.ind_min_f - wdm_set.ind_min_f
        i_hi_5 = wdm_set_5.ind_max_f - wdm_set.ind_min_f + 1
        inj_5 = WDMSignal(wdm_inj_arr[:, i_lo_5:i_hi_5], wdm_set_5)
        tpl_5 = WDMSignal(tpl_active[:, i_lo_5:i_hi_5], wdm_set_5)
        a5 = AnalysisContainer(
            DataResidualArray(inj_5),
            XYZ2SensitivityMatrix(wdm_set_5, model="scirdv1"),
        )
        mm5 = float(1.0 - a5.template_inner_product(
            DataResidualArray(tpl_5), normalize=True,
        ))

        # 2-layer mismatch -- m_floor and m_floor+1 only.
        wdm_set_2 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=m_floor * layer_df,
            max_freq=(m_floor + 2) * layer_df - 0.5 * layer_df,
            force_backend=BACKEND,
        )
        i_lo_2 = wdm_set_2.ind_min_f - wdm_set.ind_min_f
        i_hi_2 = wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1
        inj_2 = WDMSignal(wdm_inj_arr[:, i_lo_2:i_hi_2], wdm_set_2)
        tpl_2 = WDMSignal(tpl_active[:, i_lo_2:i_hi_2], wdm_set_2)
        a2 = AnalysisContainer(
            DataResidualArray(inj_2),
            XYZ2SensitivityMatrix(wdm_set_2, model="scirdv1"),
        )
        mm2 = float(1.0 - a2.template_inner_product(
            DataResidualArray(tpl_2), normalize=True,
        ))

        snr_list.append(snr)
        d_d_list.append(d_d)
        params_list.append(params_i)
        mm_full_list.append(mm_full)
        mm5_list.append(mm5)
        mm2_list.append(mm2)

        elapsed = time.perf_counter() - t0
        print(f"[draw {i:3d}] m1={params_i[0]:5.1f} m2={params_i[1]:5.1f}  "
              f"f_low={f_low*1e3:6.3f} mHz  SNR={snr:7.2f}  "
              f"mm_full={mm_full:+.2e}  mm5={mm5:+.2e}  mm2={mm2:+.2e}  "
              f"({elapsed:5.1f}s)", flush=True)

    if not snr_list:
        print("[fail] no successful draws", flush=True)
        return 1

    snr_arr     = np.asarray(snr_list)
    mm_full_arr = np.asarray(mm_full_list)
    mm5_arr     = np.asarray(mm5_list)
    mm2_arr     = np.asarray(mm2_list)
    params_arr  = np.asarray(params_list)

    print(f"\n[summary] {len(snr_list)} / {N_DRAWS} draws kept "
          f"({attempt_total} attempts, {time.perf_counter() - t0:.1f}s)",
          flush=True)
    for label, arr in [("mm_full", mm_full_arr),
                        ("mm5", mm5_arr),
                        ("mm2", mm2_arr)]:
        abs_arr = np.abs(arr)
        med = float(np.nanmedian(abs_arr))
        p90 = float(np.nanpercentile(abs_arr, 90))
        p99 = float(np.nanpercentile(abs_arr, 99))
        mx = float(np.nanmax(abs_arr))
        print(f"  {label:9s}  median={med:.3e}  90%={p90:.3e}  "
              f"99%={p99:.3e}  max={mx:.3e}", flush=True)

    # Save data
    out_h5 = OUT_PREFIX + ".h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("snr",     data=snr_arr)
        f.create_dataset("d_d",     data=np.asarray(d_d_list))
        f.create_dataset("params",  data=params_arr)
        f.create_dataset("mm_full", data=mm_full_arr)
        f.create_dataset("mm5",     data=mm5_arr)
        f.create_dataset("mm2",     data=mm2_arr)
        f.attrs["N_DRAWS"]    = N_DRAWS
        f.attrs["BACKEND"]    = BACKEND
        f.attrs["NF"]         = Nf
        f.attrs["NT"]         = Nt
        f.attrs["NT_SUB"]     = Nt_sub
        f.attrs["N_SPARSE"]   = N_sparse
        f.attrs["N_CP_SIG"]   = N_cp_sig
        f.attrs["N_CP_ORBIT"] = N_cp_orbit
        f.attrs["SEED"]       = SEED
    print(f"\n[write] {out_h5}", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, arr, title in zip(
        axes,
        [np.abs(mm_full_arr), np.abs(mm5_arr), np.abs(mm2_arr)],
        ["|mm| full", "|mm5|", "|mm2|"],
    ):
        ax.hist(np.log10(np.clip(arr, 1e-18, None)), bins=20,
                color="C0", alpha=0.85, edgecolor="k")
        ax.set_xlabel("log10 |mm|")
        ax.set_ylabel("count")
        ax.set_title(f"{title}  (median = {np.median(arr):.2e})")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"SOBBHWDMComputations prior draws  "
        f"({len(snr_list)} kept, N_cp_sig={N_cp_sig}/N_cp_orbit={N_cp_orbit})",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = OUT_PREFIX + ".png"
    plt.savefig(out_png, dpi=120)
    print(f"[write] {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
