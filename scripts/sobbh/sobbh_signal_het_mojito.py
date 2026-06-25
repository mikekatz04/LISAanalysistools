#!/usr/bin/env python
"""Test the SOBBH signal-heterodyne likelihood against MOJITO L1 data.

SOBBH analogue of the GB mojito-comparison work, built on the same cached
mojito loader as ``sobbh_mojito_match_debug.py``. For a single mojito SOBHB
source it:

  1. Loads the L1 data window (cached at /tmp/sobbh_mojito_data_src{SRC}.npz)
     + the source's catalogue row.
  2. Builds the SOBBH 11-parameter heterodyne reference from the catalogue
     (m1, m2, s1, s2, distance[pc], f_low, phi_c, inc, psi, lam, beta).
  3. lisatools-dense reference logL  (AnalysisContainer over the WDM data,
     signal_gen = SOBBHTDIonTheFly -> WDM) -- the gold standard.
  4. chunked-het mm5 / mm2 / full-band of the catalogue-param template vs the
     mojito data (model fidelity; reproduces sobbh_chunked_prior_draws but on
     real data).
  5. SOBBH signal-het ``get_ll`` at the reference + a few perturbations,
     compared to the dense logL.

Two things are reported and they answer different questions:
  * sig-het vs dense agreement (and chunked vs dense) -- validates the C++
    port works on REAL, noisy mojito data, INDEPENDENT of the SOBBH<->mojito
    convention (both use the same SOBBHTDIonTheFly + same ref params).
  * absolute mm5/mm2 vs mojito -- how well the catalogue-param SOBBHTDIonTheFly
    template matches the real signal (convention-dependent; phase/time-maxed).

Run::
    SOBBH_SRC=0 python sobbh_signal_het_mojito.py
Env: SOBBH_SRC (0), BACKEND (cpu), NT_LAYER (64), N_SPARSE_FD (1024)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import L1Orbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.globalfit.preprocessing import find_file
from lisatools.sensitivity import XYZ2SensitivityMatrix

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401
from bbhx.sobbhcomps import SOBBHWDMComputations

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from sobbhsignalhetcomputations import SOBBHSignalHetComputations  # noqa: E402

MOJITO_REFERENCE_TIME = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")

DT = 10.0
NF, NT = 512, 512
N_WIN = NF * NT
F_MIN, F_MAX = 0.010, 0.030
BACKEND = os.environ.get("BACKEND", "cpu")
SRC = int(os.environ.get("SOBBH_SRC", "0"))
DATA_CACHE = f"/tmp/sobbh_mojito_data_src{SRC}.npz"


def load_mojito():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        print(f"[cache] {DATA_CACHE}", flush=True)
        return z["data_td"], float(z["data_t0"]), z["cat"].item()
    print("[cache] MISS -> reading mojito L1 (one time)...", flush=True)
    from lisatools.globalfit.preprocessing import L1ProcessingStep
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["sobhb"], source_ids={"sobhb": SRC},
        orbits_class=L1Orbits, orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True)
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    dt_native = float(loader.dt)
    data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["SOBHB"][SRC].items()
           if np.asarray(v).dtype.kind in "fi"}
    deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN * deci : deci][:, :N_WIN].copy()
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, cat=cat)
    return data_td, data_t0, cat


def cat_to_params(cat):
    """Catalogue row -> SOBBHTDIonTheFly 11-vector
    (m1, m2, s1, s2, distance[pc], f_low, phi_c, inc, psi, lam, beta).
    distance: LuminosityDistance is in Mpc -> *1e6 pc. ICRS sky (orbits
    frame='icrs'), so lam=RA, beta=Dec raw. phi_c = TrueAnomaly (the residual
    phase-reference convention is removed by the phase-maximised overlap)."""
    return np.array([
        cat["PrimaryMassSSBFrame"],
        cat["SecondaryMassSSBFrame"],
        cat["PrimarySpinCompZ"],
        cat["SecondarySpinCompZ"],
        cat["LuminosityDistance"] * 1e6,
        cat["GW22FrequencySSBFrame"],
        cat["TrueAnomaly"],
        cat["InclinationAngle"],
        cat["PolarisationAngle"],
        cat["RightAscension"] % (2 * np.pi),
        cat["Declination"],
    ])


def main():
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    n_sparse_fd = int(os.environ.get("N_SPARSE_FD", "1024"))
    EC = 20

    data_td, data_t0, cat = load_mojito()
    ref = cat_to_params(cat)
    f_low = float(ref[5])
    print(f"[src {SRC}] m1={ref[0]:.1f} m2={ref[1]:.1f} f_low={f_low*1e3:.4f}mHz "
          f"dist={ref[4]/1e6:.1f}Mpc inc={ref[7]:.3f} SNR_cat={cat.get('EstimatedSNR', float('nan')):.1f}",
          flush=True)

    Nobs = NF * NT
    Tobs = Nobs * DT
    t0 = data_t0
    t_ref = MOJITO_REFERENCE_TIME
    t_arr = np.arange(Nobs) * DT + t0

    orbits = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)
    td_set = TDSettings(Nobs, DT, t0=t0, force_backend=BACKEND)

    wdm_kw = dict(t0=t0, min_freq=F_MIN, max_freq=F_MAX,
                  min_time=EC * NF * DT, max_time=(NT - EC) * NF * DT, force_backend=BACKEND)
    wdm_set = WDMSettings(NF, NT, DT, is_complex=False, **wdm_kw)
    layer_df = wdm_set.layer_df

    # data on the WDM band + dense AnalysisContainer
    data_wdm = TDSignal(data_td, settings=td_set).transform(wdm_set, window=None)
    data_arr = DataResidualArray(data_wdm)
    sens_mat = XYZ2SensitivityMatrix(data_arr.data_res_arr.settings, model="scirdv1")
    analysis = AnalysisContainer(data_arr, sens_mat)
    d_d = float(np.real(analysis.inner_product()))
    print(f"[data] <d|d>={d_d:.4e}  SNR_data={np.sqrt(d_d):.2f}", flush=True)

    # dense SOBBH generator (for the gold-standard logL via signal_gen)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    sobbh_gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_ref, 1.0 / DT, 1,
                                 tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                                 force_backend=BACKEND)

    def dense_wdm(p):
        sp = sobbh_gen(*np.asarray(p, float).reshape(11, 1),
                       convert_to_ra_dec=False, return_spline=True)
        td = np.asarray(sp.eval_tdi(t_arr))[0]
        return TDSignal(td, settings=td_set).transform(wdm_set, window=None)

    analysis.signal_gen = lambda *p: dense_wdm(np.asarray(p, float).reshape(11))

    def dense_logL(p):
        return float(analysis.calculate_signal_likelihood(*np.asarray(p, float).reshape(11),
                                                          source_only=True))

    # ---- chunked-het model fidelity: mm5 / mm2 / full vs mojito data ----
    chunked = SOBBHWDMComputations(
        wdm_set, t_ref=t_ref, Nt_sub=256, n_pad=32, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
        force_backend=BACKEND, d_d=d_d, tdi_type="XYZ")
    template_full = np.zeros((3, NF, NT), dtype=float)
    chunked.fill_global_wdm(ref.reshape(1, 11), template_full,
                            convert_to_ra_dec=False, factors=None)
    tpl_active = template_full[:, wdm_set.ind_min_f:wdm_set.ind_max_f + 1, wdm_set.active_slice_t]
    tpl_wdm = WDMSignal(tpl_active, wdm_set)
    mm_full = float(1.0 - analysis.template_inner_product(DataResidualArray(tpl_wdm), normalize=True))

    def narrowband_mm(lo, hi):
        ws = WDMSettings(wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
                         min_time=wdm_set.min_time, max_time=wdm_set.max_time,
                         min_freq=lo, max_freq=hi, force_backend=BACKEND)
        i_lo = ws.ind_min_f - wdm_set.ind_min_f
        i_hi = ws.ind_max_f - wdm_set.ind_min_f + 1
        d_arr = np.asarray(data_wdm.arr)
        inj_b = WDMSignal(d_arr[:, i_lo:i_hi], ws)
        tpl_b = WDMSignal(tpl_active[:, i_lo:i_hi], ws)
        ac = AnalysisContainer(DataResidualArray(inj_b), XYZ2SensitivityMatrix(ws, model="scirdv1"))
        return float(1.0 - ac.template_inner_product(DataResidualArray(tpl_b), normalize=True))

    m_floor = int(f_low / layer_df)
    mm5 = narrowband_mm(f_low - 3 * layer_df, f_low + 2 * layer_df)
    mm2 = narrowband_mm(m_floor * layer_df, (m_floor + 2) * layer_df - 0.5 * layer_df)
    print(f"\n[chunked-het vs mojito]  mm_full={mm_full:+.3e}  mm5={mm5:+.3e}  mm2={mm2:+.3e}",
          flush=True)

    # ---- signal-het likelihood, referenced at the catalogue params ----
    sighet = SOBBHSignalHetComputations(
        data_td, ref, Nf=NF, Nt=NT, dt=DT, t0=t0, t_ref=t_ref,
        orbits=orbits, tdi_config=tdi_config, min_freq=F_MIN, max_freq=F_MAX,
        nt_layer=Nt_layer, n_sparse_fd=n_sparse_fd, force_backend=BACKEND)
    print(f"[sig-het] edge_cut={sighet.edge_cut} taper_layers={sighet.taper_layers} "
          f"<d|d>={sighet.d_d:.4e}", flush=True)

    PERTS = [
        ("ref (zero)",  None, 0.0),
        ("df0 +1e-2·lf", 5, 1e-2 * layer_df),
        ("df0 +1e-1·lf", 5, 1e-1 * layer_df),
        ("dphi_c +0.05", 6, 0.05),
        ("dinc +0.02",   7, 0.02),
    ]
    print(f"\n   {'perturbation':>14s} {'logL_sighet':>13s} {'logL_dense':>13s} "
          f"{'|sighet-dense|':>14s}", flush=True)
    worst = 0.0
    for label, idx, delta in PERTS:
        p = ref.copy()
        if idx is not None:
            p[idx] = ref[idx] + delta
        ll_s = float(np.asarray(sighet.get_ll(p)).reshape(()))
        ll_d = dense_logL(p)
        diff = abs(ll_s - ll_d)
        worst = max(worst, diff)
        print(f"   {label:>14s} {ll_s:+13.4e} {ll_d:+13.4e} {diff:14.3e}", flush=True)

    print(f"\n[summary] src={SRC}  chunked mm5={mm5:.2e} mm2={mm2:.2e}  "
          f"max|sighet-dense|={worst:.3e}", flush=True)
    # The sig-het<->dense agreement is the port-correctness metric on real data.
    ok = worst < max(5.0, 0.05 * abs(dense_logL(ref)))
    print("PASS" if ok else "CHECK", "(sig-het tracks dense on mojito data)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
