"""EMRI per-link (pre-TDI) comparison: LAT pyResponseTDI y_gw vs mojito eta_ij.

Localizes the SOBBH global -1: the response is SHARED with the EMRI, so if the
EMRI y_gw matches mojito eta_ij with +1 the -1 is the SOBBH waveform's A sign;
if -1, it is the shared response projection.
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
from few.waveform import GenerateEMRIWaveform
from lisatools.sources.utils import icrs_to_ecliptic


class _FewEclipticSky:
    def __init__(self, few, qS_ecl, phiS_ecl):
        self._few = few; self.qS_ecl = float(qS_ecl); self.phiS_ecl = float(phiS_ecl)
    def __call__(self, *a, **k):
        a = list(a); a[7] = self.qS_ecl; a[8] = self.phiS_ecl
        return self._few(*a, **k)

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1                       # source1 -> catalogue ROW 1
DT = 20.0; N_WIN = 65536; TOBS = N_WIN * DT
LAT_LINKS = [12, 23, 31, 13, 32, 21]
ETA_IJ = ["12", "23", "31", "13", "32", "21"]


def watchdog():
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
    c = np.vdot(hilbert(np.roll(b, lag)), hilbert(a)) / np.vdot(hilbert(np.roll(b, lag)), hilbert(np.roll(b, lag)))
    return rho, lag, c


def main():
    threading.Thread(target=watchdog, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        params = [g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
                  g("SemiLatusRectum"), g("Eccentricity"), np.cos(g("InclinationAngle")),
                  g("LuminosityDistance") / 1e3, np.pi / 2 - g("Declination"),
                  g("RightAscension") % (2 * np.pi), g("PolarAnglePrimarySpin"),
                  g("AzimuthalAnglePrimarySpin"), g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")]
    print(f"EMRI row{SRC}: M={params[0]:.0f} mu={params[1]:.2f} e0={params[4]:.3f}", flush=True)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    mf = MojitoL1File(fp); ts = mf.tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        eta = {ij: np.asarray(f["tdis"][f"eta_{ij}"][: N_WIN * deci])[::deci][:N_WIN] / lf for ij in ETA_IJ}

    print("  building orbit (ltt-sliced)...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
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
    if os.environ.get("EMRI_ECL_SKY"):
        lam_ecl, beta_ecl = icrs_to_ecliptic(float(params[8]), float(np.pi / 2 - params[7]))
        wave_gen.waveform_gen = _FewEclipticSky(wave_gen.waveform_gen,
                                                np.pi / 2 - float(beta_ecl), float(lam_ecl) % (2 * np.pi))
        print(f"  [sky] FEW ecliptic qS={np.pi/2-float(beta_ecl):.4f} phiS={float(lam_ecl)%(2*np.pi):.4f}", flush=True)
    print("  generating response...", flush=True)
    _ = wave_gen(*params)
    y_gw = np.asarray(wave_gen.response_model.y_gw_flat).reshape(6, -1)
    n = min(y_gw.shape[1], N_WIN); win = tukey(n, 0.1)

    print("\n  per-link match (diagonal LAT ij vs eta_ij): rho, lag, |ratio|, phase deg")
    for i, link in enumerate(LAT_LINKS):
        a = y_gw[i, :n] * win; bb = eta[ETA_IJ[i]][:n] * win
        rho, lag, c = best_match(a, bb)
        print(f"    LAT {link} vs eta_{ETA_IJ[i]}: rho={rho:+.3f} lag={lag} "
              f"|c|={abs(c):.3f} phase={np.degrees(np.angle(c)):+.1f}  "
              f"rms(y)={np.sqrt(np.mean(a**2)):.2e} rms(eta)={np.sqrt(np.mean(bb**2)):.2e}", flush=True)
    print("\n  => if rho ~ +1: -1 is the SOBBH waveform A sign.  if rho ~ -1: -1 is the shared response.")


if __name__ == "__main__":
    main()
