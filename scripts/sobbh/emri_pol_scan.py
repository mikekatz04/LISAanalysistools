"""EMRI: the response == lisagwresponse, so the per-arm mismatch must be the
FEW polarization-frame alignment to the ICRS (u,v) axes. Compute the FEW
waveform once, then scan a polarization rotation h -> h*exp(2i*psi) (and the
ICRS vs ecliptic sky), re-projecting each time, to find the alignment that
makes the per-link y_gw match the mojito eta_ij.
"""
import os, sys, time, threading, resource, gc
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


def rho_of(a, b):
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    xc = np.fft.irfft(A * np.conj(B), n=len(a))
    lag = int(np.argmax(np.abs(xc)))
    return xc[lag] / np.sqrt(np.sum(a**2) * np.sum(b**2))


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        ra = g("RightAscension") % (2 * np.pi); dec = g("Declination")
        base = [g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
                g("SemiLatusRectum"), g("Eccentricity"), np.cos(g("InclinationAngle")),
                g("LuminosityDistance") / 1e3, None, None, g("PolarAnglePrimarySpin"),
                g("AzimuthalAnglePrimarySpin"), g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")]

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
    rm = wave_gen.response_model
    lam = ra; beta = dec   # response uses ICRS (u,v) from (ra, dec)
    n = N_WIN; win = tukey(n, 0.1)

    def rms(a):
        return float(np.sqrt(np.mean(a ** 2)))

    def dir_icrs_to_ecl(theta, phi):  # convert a direction (polar theta, az phi) ICRS->ecl
        lam_, beta_ = icrs_to_ecliptic(float(phi) % (2 * np.pi), float(np.pi / 2 - theta))
        return float(np.pi / 2 - beta_), float(lam_) % (2 * np.pi)

    eta_rms = [rms(eta[ETA_IJ[i]][:n] * win) for i in range(6)]
    for config in ["icrs", "ecl_all"]:
        if config == "icrs":
            qS, phiS, qK, phiK = np.pi / 2 - dec, ra % (2 * np.pi), base[9], base[10]
        else:
            lam_e, beta_e = icrs_to_ecliptic(float(ra), float(dec))
            qS, phiS = np.pi / 2 - float(beta_e), float(lam_e) % (2 * np.pi)
            qK, phiK = dir_icrs_to_ecl(base[9], base[10])
        params = list(base); params[7], params[8], params[9], params[10] = qS, phiS, qK, phiK
        h = np.asarray(fg(*params, T=wave_gen.Tobs, dt=DT))     # FEW h+ - i hx, complex
        print(f"\n=== config={config}  FEW(qS={qS:.3f} phiS={phiS:.3f} qK={qK:.3f} phiK={phiK:.3f}),"
              f" response stays ICRS(ra,dec) ===", flush=True)
        best = (-9, None)
        for psi in np.linspace(0, np.pi, 25):
            hr = h * np.exp(2j * psi)                            # polarization rotation
            h_in = hr.real - 1j * hr.imag                        # flip_hx=True
            rm.get_projections(h_in, lam, beta, t0_shift_to_data=wave_gen.t0_shift_to_data,
                               t0=wave_gen.t0, t_buffer=wave_gen.t_buffer)
            y = np.asarray(rm.y_gw_flat).reshape(6, -1)
            rhos = [rho_of(y[i, :n] * win, eta[ETA_IJ[i]][:n] * win) for i in range(6)]
            amps = [rms(y[i, :n] * win) / eta_rms[i] for i in range(6)]
            score = np.mean(rhos)
            if score > best[0]:
                best = (score, np.degrees(psi), rhos, amps)
        sc, psd, rhos, amps = best
        spread = max(amps) / min(amps)
        print(f"  best psi={psd:.0f}deg  mean_rho={sc:+.3f}  amp-spread(max/min)={spread:.2f}", flush=True)
        print(f"    per-arm rho: " + " ".join(f"{ETA_IJ[i]}={rhos[i]:+.2f}" for i in range(6)), flush=True)
        print(f"    per-arm amp(LAT/eta): " + " ".join(f"{ETA_IJ[i]}={amps[i]:.2f}" for i in range(6)), flush=True)


if __name__ == "__main__":
    main()
