"""ONE-OFF CHECK (not a permanent setup): EMRI vs mojito with the user-specified frame path.

SPECIAL: FEW intrinsic from ECLIPTIC-POLAR sky (qS_e,phiS_e) + FILE spin (qK,phiK) -> h+/hx.
         ResponseWrapper with is_ecliptic_latitude=False (so beta = pi/2 - qS_e = ecliptic
         latitude) and the call uses convert_to_ra_dec=True, so INSIDE the wrapper lam/beta
         are converted ecliptic->ICRS (ra,dec) just for the response, against ICRS orbits.
REF    : all-ecliptic (ecliptic sky + ecliptic spin), ecliptic orbits, is_ecliptic_latitude
         =False, no convert -- the corrected-convention data-reproducing baseline.

t0 = REF (catalogue epoch) for the intrinsic h+/hx; non-zero t0_shift_to_data; integer
offset sliced. Throwaway diagnostic.
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
REF = 97729089.327664; SRC = 1
DT = 20.0; N_WIN = 2 ** 17; TOBS_S = N_WIN * DT
ORDER = 25; T_BUFFER = 18000.0; THRESH = 1e-5


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.3)


def make_orbit(fp, frame, data_t0):
    orb = L1Orbits(fp, force_backend="cpu", frame=frame)
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)
    return orb


def make_fg():
    return GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, mode_selector_kwargs={"mode_selection_threshold": THRESH},
        frame="detector", force_backend="cpu")


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
    INTR = [M, mu, a, p0, e0, 1.0, dist]; PHASES = [Pp, Pt, Pr]
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])
    off = data_t0 - REF; offset_int = int(round(off / DT)); T0_SHIFT = off - offset_int * DT
    N_RESP = N_WIN + offset_int
    print(f"  off={off:.3f}s offset_int={offset_int} t0_shift_to_data={T0_SHIFT:+.4f}s  "
          f"qS_e(polar)={qS_e:.4f} phiS_e={phiS_e:.4f}  spin file qK={qK:.4f} phiK={phiK:.4f}", flush=True)

    n_buf = int(T_BUFFER / DT) + 4 * ORDER + 50
    win = np.zeros(N_WIN); win[n_buf:N_WIN - n_buf] = tukey(N_WIN - 2 * n_buf, 0.05)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    data_snr = np.sqrt(float(ac.inner_product().real))
    fb = np.asarray(fd_set.f_arr); taus = np.linspace(-2000.0, 2000.0, 161)
    print(f"  N_WIN={N_WIN} (Tobs={TOBS_S/86400:.1f}d)  data SNR={data_snr:.3f}", flush=True)

    def metrics(leg):
        tmpl = TDSignal(leg, td_set).transform(fd_set, window=win)
        base = np.asarray(tmpl.arr).copy()
        opt, det0 = ac.template_snr(tmpl); O0 = det0 / data_snr
        best_det, best_tau = -1.0, 0.0
        for t in taus:
            tmpl.arr[:] = base * np.exp(2j * np.pi * fb * t)[None, :]
            _, det = ac.template_snr(tmpl)
            if np.isfinite(det) and det > best_det:
                best_det, best_tau = float(det), float(t)
        return dict(O=best_det / data_snr, O0=O0, opt=float(opt) / data_snr, tau=best_tau)

    def run(label, orb_frame, vec, is_ecl_lat, convert):
        orb = make_orbit(fp, orb_frame, data_t0)
        wl = ResponseWrapper(make_fg(), orbits=orb, t0=REF, t0_shift_to_data=T0_SHIFT,
            Tobs=N_RESP * DT / YRSID_SI, dt=DT, index_lambda=8, index_beta=7, flip_hx=True,
            is_ecliptic_latitude=is_ecl_lat, tdi=TDIConfig("2nd generation", force_backend="cpu"),
            tdi_chan="XYZ", order=ORDER, remove_garbage="zero", t_buffer=T_BUFFER, force_backend="cpu")
        t0 = time.time()
        leg = np.atleast_2d(np.asarray(wl(*vec, convert_to_ra_dec=convert)))[:3]
        leg = leg[:, offset_int:offset_int + N_WIN]
        if leg.shape[-1] < N_WIN:
            leg = np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1])))
        m = metrics(leg)
        print(f"  [{label}] mm0(tau=0)={1-m['O0']:.4e} |O0|={m['O0']:.4f} | mm={1-m['O']:.4e} "
              f"|O|={m['O']:.4f}  opt/data={m['opt']:.3f}  tau={m['tau']:+.0f}s  [{time.time()-t0:.0f}s]", flush=True)
        del wl, orb; gc.collect()

    # SPECIAL: ecliptic-polar sky + FILE spin -> FEW ; is_ecliptic_latitude=False + convert_to_ra_dec=True -> ICRS resp
    print("\n  [SPECIAL] ecl-polar sky + file spin -> FEW; convert lam/beta->ra/dec for ICRS-orbit response", flush=True)
    run("SPECIAL", "icrs", INTR + [qS_e, phiS_e, float(qK), float(phiK)] + PHASES, is_ecl_lat=False, convert=True)

    # ALT: SAME as SPECIAL (file spin + ecliptic-polar sky) but ECLIPTIC orbits + convert=False.
    # Should EQUAL SPECIAL if the ecliptic<->ICRS frame handling is self-consistent.
    print("\n  [ALT] file spin + ecl-polar sky -> FEW; ECLIPTIC orbits, convert_to_ra_dec=False", flush=True)
    run("ALT-eclorb", "ecliptic", INTR + [qS_e, phiS_e, float(qK), float(phiK)] + PHASES, is_ecl_lat=False, convert=False)

    # REF: all-ecliptic (ecliptic sky + ecliptic-CONVERTED spin) -- shows the 1.49x (opt/data~1.47)
    print("\n  [REF all-ecliptic, ecliptic-converted spin] is_ecliptic_latitude=False", flush=True)
    run("REF-ecl", "ecliptic", INTR + [qS_e, phiS_e, qK_e, phiK_e] + PHASES, is_ecl_lat=False, convert=False)


if __name__ == "__main__":
    main()
