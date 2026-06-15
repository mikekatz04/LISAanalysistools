"""Per-link (pre-TDI) comparison: LAT pyResponseTDI y_gw vs mojito eta_ij.

Isolates the waveform + single-link projection convention from the TDI
combination.  For each LAT link [12,23,31,13,32,21] (receiver i, emitter j),
cross-correlate against all 6 mojito eta_ij to find the link map, sign,
amplitude ratio and phase.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import h5py
from scipy.signal import hilbert
from scipy.signal.windows import tukey

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.utils.constants import YRSID_SI
from mojito import MojitoL1File

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
SRC = int(os.environ.get("SOBBH_SRC", "0"))
DT = float(os.environ.get("SOBBH_DT", "10.0"))
_TOBS_FIXED = 2621440.0                      # ~30.3 days, dt-independent window
N_WIN = int(round(_TOBS_FIXED / DT))
TOBS = N_WIN * DT
LAT_LINKS = [12, 23, 31, 13, 32, 21]           # y_gw row order (recv i, emit j)
ETA_IJ = ["12", "23", "31", "13", "32", "21"]
FLIP_HX = False


def catalogue_params(src):
    cat = os.path.join(PATH, "catalogues", "sobhb_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][src])
        return [g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
                g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
                g("LuminosityDistance") / 1e3, g("InclinationAngle"),
                g("GW22FrequencySSBFrame"), g("RightAscension") % (2 * np.pi),
                g("Declination"), g("PolarisationAngle") % np.pi, g("TrueAnomaly")]


def best_match(a, b):
    """Normalized complex correlation of a vs b over time-lags (FFT)."""
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    xcorr = np.fft.irfft(A * np.conj(B), n=len(a))      # real cross-corr vs lag
    lag = int(np.argmax(np.abs(xcorr)))
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    rho = xcorr[lag] / norm                              # signed normalized corr
    if lag > len(a) // 2:
        lag -= len(a)
    # complex ratio (amplitude + phase) at best lag via analytic signals
    bsh = np.roll(b, lag)
    ah = hilbert(a); bh = hilbert(bsh)
    c = np.vdot(bh, ah) / np.vdot(bh, bh)                # a ~ c * b
    return rho, lag, c


def main():
    params = catalogue_params(SRC)
    print(f"SOBHB source {SRC}: m1={params[0]:.1f} m2={params[1]:.1f} "
          f"f_low={params[6]:.5g} inc={params[5]:.3f} cos(inc)={np.cos(params[5]):+.3f} "
          f"psi={params[9]:.3f}", flush=True)

    fp = find_file(SOBHB_L1, "SOBHB", SRC)
    mf = MojitoL1File(fp)
    ts = mf.tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        eta = {ij: np.asarray(f["tdis"][f"eta_{ij}"][: N_WIN * deci])[::deci][:N_WIN] / lf
               for ij in ETA_IJ}
        moj_tdi = {ch: np.asarray(f["tdis"][ch][: N_WIN * deci])[::deci][:N_WIN] / lf
                   for ch in ["X2", "Y2", "Z2", "A2", "E2", "T2"]}
    print(f"  data_t0={data_t0:.4f}  dt_native={ts.dt}  deci={deci}  lf={lf:.4e}", flush=True)

    print("  building L1Orbits...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="icrs")
    orb.configure(linear_interp_setup=True)

    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    gen = SOBBHWaveform(TOBS, DT, t0=data_t0, reference_time=REF, force_backend="cpu")
    wave_gen = ResponseWrapper(
        gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=7, index_beta=8, flip_hx=FLIP_HX, tdi=tdi_config, tdi_chan="XYZ",
        order=40, remove_garbage="zero", is_ecliptic_latitude=True, t_buffer=3e4,
        force_backend="cpu")
    print("  generating response (TDI + projections)...", flush=True)
    tdi_xyz = np.atleast_2d(np.asarray(wave_gen(*params, convert_to_ra_dec=False)))[:3]
    y_gw = np.asarray(wave_gen.response_model.y_gw_flat).reshape(6, -1)
    npts = y_gw.shape[1]
    print(f"  y_gw shape={y_gw.shape}  eta len={N_WIN}  (compare on {min(npts,N_WIN)})", flush=True)

    n = min(npts, N_WIN)
    win = tukey(n, 0.1)
    print(f"\n  per-link rms: LAT y_gw vs mojito eta (fractional freq)")
    for i, link in enumerate(LAT_LINKS):
        print(f"    LAT {link}: rms={np.sqrt(np.mean(y_gw[i,:n]**2)):.3e}   "
              f"mojito eta_{ETA_IJ[i]}: rms={np.sqrt(np.mean(eta[ETA_IJ[i]][:n]**2)):.3e}")

    print("\n  6x6 match (rho, lag, ratio|c|, phase deg) -- best per LAT link in CAPS:")
    print("        " + "  ".join(f"eta_{ij:>2}" for ij in ETA_IJ))
    for i, link in enumerate(LAT_LINKS):
        a = (y_gw[i, :n] * win)
        row = []
        best = (0, None)
        for ij in ETA_IJ:
            b = (eta[ij][:n] * win)
            rho, lag, c = best_match(a, b)
            row.append((ij, rho, lag, abs(c), np.degrees(np.angle(c))))
            if abs(rho) > abs(best[0]):
                best = (rho, ij)
        cells = []
        for (ij, rho, lag, ac, ph) in row:
            tag = "*" if ij == best[1] else " "
            cells.append(f"{tag}{rho:+.2f}")
        print(f"  LAT{link:>3} " + " ".join(cells) + f"   -> best eta_{best[1]} (rho={best[0]:+.3f})")
        # detail on the best
        ij = best[1]; b = (eta[ij][:n] * win)
        rho, lag, c = best_match(a, b)
        print(f"          best eta_{ij}: lag={lag} samp ({lag*DT:+.0f}s)  "
              f"y_gw ~ ({abs(c):.4f} e^i{np.degrees(np.angle(c)):+.1f}deg) * eta")

    # ---- TDI channel sign alignment: X/Y/Z and A/E/T vs mojito ----
    X, Y, Z = tdi_xyz[0, :n], tdi_xyz[1, :n], tdi_xyz[2, :n]
    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    lat_tdi = {"X2": X, "Y2": Y, "Z2": Z, "A2": A, "E2": E, "T2": T}
    print("\n  TDI channel sign alignment (LAT vs mojito, AET = standard from XYZ):")
    for ch in ["X2", "Y2", "Z2", "A2", "E2", "T2"]:
        a = lat_tdi[ch] * win; b = moj_tdi[ch][:n] * win
        rho, lag, c = best_match(a, b)
        print(f"    {ch:3s}: rho={rho:+.4f}  lag={lag:+d}  |c|={abs(c):.4f}  "
              f"phase={np.degrees(np.angle(c)):+.1f}  -> {'ALIGNED' if rho>0.5 else 'FLIPPED' if rho<-0.5 else '??'}")

    # ---- NON-phase-maximized (raw) overlap + the exact timing offset ----
    print(f"\n  timing: t0={wave_gen.t0:.4f}  t0_shift_to_data={wave_gen.t0_shift_to_data:.4f}  "
          f"data_t0={data_t0:.4f}", flush=True)
    f = np.fft.rfftfreq(n, d=DT)
    for ch in ["X2", "A2"]:
        a = lat_tdi[ch] * win; b = moj_tdi[ch][:n] * win
        A = np.fft.rfft(a); B = np.fft.rfft(b)
        norm = np.sqrt(np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2))
        raw = np.sum(np.conj(B) * A) / norm                       # lag=0, non-phase-max
        taus = np.linspace(-30, 30, 601); best = (-9, 0.0)
        for tau in taus:
            o = np.sum(np.conj(B) * A * np.exp(2j * np.pi * f * tau)) / norm
            if o.real > best[0]:
                best = (o.real, tau)
        print(f"    {ch} RAW(lag0): Re(O)={raw.real:+.4f} |O|={abs(raw):.4f} "
              f"ph={np.degrees(np.angle(raw)):+.1f}   best raw Re={best[0]:+.4f} at tau={best[1]:+.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
