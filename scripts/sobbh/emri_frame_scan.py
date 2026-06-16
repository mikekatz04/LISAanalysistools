"""EMRI sky/spin FRAME scan via the per-link (pre-TDI) diagnostic.

The LAT response == lisagwresponse (ICRS u,v from ra,dec), proven with the
SOBBH. The EMRI is equatorial (inc=0), so the per-arm response is set entirely
by the SPIN direction (qK,phiK) relative to the sky (qS,phiS). This scans which
frame FEW wants for (sky, spin) by matching y_gw to mojito eta_ij per arm
(|c| at best lag, so the 850.5s reference-time phase is removed).
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

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1
DT = 20.0; N_WIN = 65536; TOBS = N_WIN * DT
LAT_LINKS = [12, 23, 31, 13, 32, 21]
ETA_IJ = ["12", "23", "31", "13", "32", "21"]


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.3)


def best_match(a, b):
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    xc = np.fft.irfft(A * np.conj(B), n=len(a))
    lag = int(np.argmax(np.abs(xc)))
    rho = xc[lag] / np.sqrt(np.sum(a**2) * np.sum(b**2))
    if lag > len(a) // 2:
        lag -= len(a)
    bb = hilbert(np.roll(b, lag))
    c = np.vdot(bb, hilbert(a)) / np.vdot(bb, bb)
    return rho, lag, c


def dir_icrs_to_ecl(theta, phi):
    """Convert a direction (polar theta from ICRS z, azimuth phi) to ecliptic
    (theta', phi'). Treat (phi, pi/2-theta) as (ra, dec)."""
    lam, beta = icrs_to_ecliptic(float(phi) % (2 * np.pi), float(np.pi / 2 - theta))
    return float(np.pi / 2 - beta), float(lam) % (2 * np.pi)


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        M, mu, a = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter")
        p0, e0, inc = g("SemiLatusRectum"), g("Eccentricity"), g("InclinationAngle")
        dist = g("LuminosityDistance") / 1e3
        ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
        qK_i, phiK_i = g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin")
        Pp, Pt, Pr = g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")
    base = [M, mu, a, p0, e0, np.cos(inc), dist, None, None, None, None, Pp, Pt, Pr]
    print(f"EMRI src{SRC}: M={M:.2e} mu={mu:.1f} a={a:.3f} p0={p0:.3f} e0={e0:.3f} inc={inc:.3f}", flush=True)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    mf = MojitoL1File(fp); ts = mf.tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        eta = {ij: np.asarray(f["tdis"][f"eta_{ij}"][: N_WIN * deci])[::deci][:N_WIN] / lf for ij in ETA_IJ}

    print("  building orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="icrs")
    pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); del ltt_t; gc.collect()
    orb.configure(linear_interp_setup=True)

    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector",
        mode_selector_kwargs={"mode_selection_threshold": 1e-2}, force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    wave_gen = ResponseWrapper(fg, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=True, tdi=tdi_config, tdi_chan="XYZ",
        order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")

    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    qK_e, phiK_e = dir_icrs_to_ecl(qK_i, phiK_i)
    configs = {
        "icrs      (sky+spin ICRS)": (np.pi / 2 - dec, ra % (2 * np.pi), qK_i, phiK_i),
        "ecl_sky   (sky ecl, spin ICRS)": (qS_e, phiS_e, qK_i, phiK_i),
        "ecl_all   (sky+spin ecl)": (qS_e, phiS_e, qK_e, phiK_e),
    }
    n = N_WIN; win = tukey(n, 0.1)
    for name, (qS, phiS, qK, phiK) in configs.items():
        p = list(base); p[7], p[8], p[9], p[10] = qS, phiS, qK, phiK
        wave_gen(*p, convert_to_ra_dec=False)
        y = np.asarray(wave_gen.response_model.y_gw_flat).reshape(6, -1)
        cs = []
        for i in range(6):
            _, lag, c = best_match(y[i, :n] * win, eta[ETA_IJ[i]][:n] * win)
            cs.append(c)
        amps = [abs(c) for c in cs]
        print(f"\n  [{name}]  qS={qS:.3f} phiS={phiS:.3f} qK={qK:.3f} phiK={phiK:.3f}", flush=True)
        print(f"    per-arm |c| (LAT/eta): " + " ".join(f"{ETA_IJ[i]}={amps[i]:.3f}" for i in range(6)), flush=True)
        print(f"    spread (max/min)={max(amps)/min(amps):.2f}   mean|c|={np.mean(amps):.3f}   "
              f"signs={[int(np.sign(c.real)) for c in cs]}", flush=True)


if __name__ == "__main__":
    main()
