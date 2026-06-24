"""Decisive amplitude test: legacy pyResponse vs TOF, opt SNR vs mojito data, at
the SAME window/threshold. The opt-SNR-vs-threshold sweep showed opt is FLAT in
mode count (~5.95 at 1e-2..1e-7), so 1e-2 (22 modes, the dense FEW summation fits)
is representative -- no OOM, no cluster needed.

Builds the legacy all-ecliptic pyResponse (GenerateEMRIWaveform frame=detector ->
ResponseWrapper, FEW-correct, NO AMP_FACTOR) on N_WIN=16384 and reports opt/det/
data SNR. Compare to the TOF (emritdionfly AMP=0.5): opt=5.98 at 1e-2.

  legacy opt ~ 5.98 (== TOF) -> AMP=0.5 + TOF response == legacy end-to-end; both
      overshoot data 1.48x -> the gap is FEW-vs-mojito model/frame, and the TOF
      response carries a compensating 2x for the halved strain.
  legacy opt ~ 11.9 (== 2x TOF) -> the AMP=0.5 halves the TOF strain and it
      propagates; FEW-correct is 2.96x the data (the deep model gap is ~3x).
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from few.waveform import GenerateEMRIWaveform

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1
DT = 20.0; N_WIN = 16384; TOBS = N_WIN * DT
THRESH = float(os.environ.get("MODE_THRESH", "1e-2"))


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        M, mu, a = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter")
        p0, e0, dist = g("SemiLatusRectum"), g("Eccentricity"), g("LuminosityDistance") / 1e3
        ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
        qK, phiK = g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin")
        Pp, Pt, Pr = g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)
    base = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, qK_e, phiK_e, Pp, Pt, Pr]

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    print(f"  building legacy pyResponse (FEW-correct, no AMP_FACTOR), thresh={THRESH:.0e}...", flush=True)
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector",
        mode_selector_kwargs={"mode_selection_threshold": THRESH}, force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    wave_gen = ResponseWrapper(fg, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=True, tdi=tdi_config, tdi_chan="XYZ",
        order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
    lat = np.atleast_2d(np.asarray(wave_gen(*base, convert_to_ra_dec=False)))[:3]
    lat = (np.pad(lat, ((0, 0), (0, N_WIN - lat.shape[-1]))) if lat.shape[-1] < N_WIN else lat[:, :N_WIN])
    print(f"  legacy |X|max={np.max(np.abs(lat[0])):.3e}  finite={np.isfinite(lat).all()}", flush=True)

    # ---- noise-weighted opt/det/data SNR (full XYZ, SciRD) ----
    win = tukey(N_WIN, 0.1)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    tmpl_fd = TDSignal(lat, td_set).transform(fd_set, window=win)
    fb = np.asarray(fd_set.f_arr); base_arr = np.asarray(tmpl_fd.arr).copy()
    opt, _ = ac.template_snr(tmpl_fd)
    best_det = -9e99
    for t in np.linspace(-1500, 1500, 301):
        tmpl_fd.arr[:] = base_arr * np.exp(2j * np.pi * fb * t)[None, :]
        _, det = ac.template_snr(tmpl_fd)
        best_det = max(best_det, float(det))
    tmpl_fd.arr[:] = base_arr
    dd = float(ac.inner_product().real)
    print(f"\n  LEGACY pyResponse (thresh {THRESH:.0e}, FEW-correct, AMP=1.0-equiv):", flush=True)
    print(f"    opt SNR={opt:.3f}  det SNR={best_det:.3f}  data SNR={np.sqrt(dd):.3f}  "
          f"opt/data={opt/np.sqrt(dd):.3f}", flush=True)
    print(f"    -> TOF (emritdionfly AMP=0.5) opt was 5.98 at 1e-2.  "
          f"legacy/TOF = {opt/5.98:.3f}  (==1 -> same; ==2 -> AMP halves)", flush=True)


if __name__ == "__main__":
    main()
