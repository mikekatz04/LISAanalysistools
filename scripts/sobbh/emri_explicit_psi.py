"""EMRI explicit-psi recipe (the TDI-on-the-fly recipe) through pyResponseTDI,
vs mojito. emritdionfly.py separates the pieces: source-frame modes via viewing
angles + EXPLICIT get_polarization_angle psi + ICRS response -- NOT FEW
frame="detector" (which bakes its own sky+psi into hp/hx and double-counts vs
the response u/v). Here we test that recipe with pyResponseTDI: FEW frame="source"
(no internal psi) + apply psi=get_polarization_angle explicitly, all-ecliptic.
If the residual amp-spread 1.55 collapses -> diagnosis proven.
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.signal import hilbert
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_polarization_angle, get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1
DT = 20.0; N_WIN = 65536; TOBS = N_WIN * DT
ETA_IJ = ["12", "23", "31", "13", "32", "21"]


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.3)


def best_match(a, b):
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    xc = np.fft.irfft(A * np.conj(B), n=len(a)); lag = int(np.argmax(np.abs(xc)))
    rho = xc[lag] / np.sqrt(np.sum(a**2) * np.sum(b**2))
    if lag > len(a) // 2: lag -= len(a)
    bb = hilbert(np.roll(b, lag)); c = np.vdot(bb, hilbert(a)) / np.vdot(bb, bb)
    return rho, lag, c


class _SourcePlusPsi:
    """FEW frame='source' h, then apply the explicit polarization rotation."""
    def __init__(self, fg_src, psi, sign):
        self.fg = fg_src; self.fac = np.exp(sign * 2j * psi)
    def __call__(self, *a, **k):
        return np.asarray(self.fg(*a, **k)) * self.fac


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
    theta, phi = get_viewing_angles(qS_e, phiS_e, qK_e, phiK_e)
    psi = float(get_polarization_angle(qS_e, phiS_e, qK_e, phiK_e))
    print(f"  ecliptic sky qS={qS_e:.3f} phiS={phiS_e:.3f} spin qK={qK_e:.3f} phiK={phiK_e:.3f}", flush=True)
    print(f"  viewing theta={float(theta):.4f} phi={float(phi):.4f}  psi={psi:.4f}", flush=True)
    base = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, qK_e, phiK_e, Pp, Pt, Pr]

    fp = find_file(EMRI_L1, "EMRI", SRC)
    mf = MojitoL1File(fp); ts = mf.tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        eta = {ij: np.asarray(f["tdis"][f"eta_{ij}"][: N_WIN * deci])[::deci][:N_WIN] / lf for ij in ETA_IJ}
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    fg_src = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="source",
        mode_selector_kwargs={"mode_selection_threshold": 1e-2}, force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    n = N_WIN; win = tukey(n, 0.1)
    eta_rms = [np.sqrt(np.mean((eta[ETA_IJ[i]][:n] * win) ** 2)) for i in range(6)]
    ff = np.fft.rfftfreq(n, d=DT); taus = np.linspace(-1500, 1500, 601)

    for tag, sign, flip in [("source+psi(+)", +1, True), ("source+psi(-)", -1, True),
                            ("source+psi(+)nohx", +1, False)]:
        wave_gen = ResponseWrapper(_SourcePlusPsi(fg_src, psi, sign), orbits=orb, t0=data_t0,
            Tobs=TOBS / YRSID_SI, dt=DT, index_lambda=8, index_beta=7, flip_hx=flip,
            tdi=tdi_config, tdi_chan="XYZ", order=40, remove_garbage="zero", t_buffer=3e4,
            force_backend="cpu")
        tdi = np.atleast_2d(np.asarray(wave_gen(*base, convert_to_ra_dec=False)))[:3]
        y = np.asarray(wave_gen.response_model.y_gw_flat).reshape(6, -1)
        rhos, amps = [], []
        for i in range(6):
            rho, lag, c = best_match(y[i, :n] * win, eta[ETA_IJ[i]][:n] * win)
            rhos.append(rho); amps.append(np.sqrt(np.mean((y[i, :n] * win) ** 2)) / eta_rms[i])
        Och = []
        for ci in range(3):
            A = np.fft.rfft(tdi[ci, :n] * win); B = np.fft.rfft(dXYZ[ci, :n] * win)
            nm = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum())
            Och.append(max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nm for t in taus))
        print(f"\n  [{tag}]  mean|rho|={np.mean(np.abs(rhos)):.3f}  amp-spread={max(amps)/min(amps):.2f}", flush=True)
        print(f"    per-arm rho: " + " ".join(f"{ETA_IJ[i]}={rhos[i]:+.2f}" for i in range(6)), flush=True)
        print(f"    per-arm amp: " + " ".join(f"{ETA_IJ[i]}={amps[i]:.2f}" for i in range(6)), flush=True)
        print(f"    time-scan TDI |O|: X={Och[0]:.4f} Y={Och[1]:.4f} Z={Och[2]:.4f}  "
              f"(all-ecl frame=detector was 0.92)", flush=True)


if __name__ == "__main__":
    main()
