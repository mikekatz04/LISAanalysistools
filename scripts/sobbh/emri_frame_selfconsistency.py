"""EMRI frame self-consistency test (NO mojito data).

Per user direction: start with sky+spin in the ecliptic (SSB) frame, generate the
TDI; transform ALL angles to ICRS (equatorial) AND load the orbit in ICRS; the
physical TDI must be identical. FEW (GenerateEMRIWaveform) computes psi and the
spherical-harmonic viewing angles internally from (qS,phiS,qK,phiK), so this
checks that FEW's psi convention is mutually consistent with LAT's response u/v
and the orbit-frame rotation. If TDI_ecl == TDI_icrs, the frame is NOT the
per-arm-null cause; if not, the frame handling is the bug -- iterate here first.

FEW takes POLAR angles for qS,qK (colatitude), so qS=pi/2-beta, qK=pi/2-beta_spin.
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper, ecliptic_to_icrs
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from few.waveform import GenerateEMRIWaveform

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1
DT = 20.0; N_WIN = 8192; TOBS = N_WIN * DT          # short window: self-consistency only


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.3)


def dir_ecl_to_icrs(theta_ecl, phi_ecl):
    """A direction given as ecliptic polar/azimuth -> ICRS polar/azimuth."""
    ra, dec = ecliptic_to_icrs(float(phi_ecl) % (2 * np.pi), float(np.pi / 2 - theta_ecl))
    return float(np.pi / 2 - dec), float(ra) % (2 * np.pi)


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        M, mu, a = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter")
        p0, e0 = g("SemiLatusRectum"), g("Eccentricity")
        dist = g("LuminosityDistance") / 1e3
        ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
        qK, phiK = g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin")
        Pp, Pt, Pr = g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")

    # config A: TREAT the catalogue sky/spin as ECLIPTIC (polar angles)
    qS_e, phiS_e = np.pi / 2 - dec, ra
    qK_e, phiK_e = qK, phiK
    # config B: rotate sky AND spin directions ecliptic -> ICRS
    qS_i, phiS_i = dir_ecl_to_icrs(qS_e, phiS_e)
    qK_i, phiK_i = dir_ecl_to_icrs(qK_e, phiK_e)
    print(f"  ECL: qS={qS_e:.4f} phiS={phiS_e:.4f} qK={qK_e:.4f} phiK={phiK_e:.4f}", flush=True)
    print(f"  ICRS:qS={qS_i:.4f} phiS={phiS_i:.4f} qK={qK_i:.4f} phiK={phiK_i:.4f}", flush=True)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    data_t0 = float(MojitoL1File(fp).tdis.time_sampling.t0)
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector",
        mode_selector_kwargs={"mode_selection_threshold": 1e-2}, force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")

    def build(frame, qS, phiS, qK_, phiK_):
        orb = L1Orbits(fp, force_backend="cpu", frame=frame)
        pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
        lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
        orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
        orb.configure(linear_interp_setup=True)
        wg = ResponseWrapper(fg, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
            index_lambda=8, index_beta=7, flip_hx=True, tdi=tdi_config, tdi_chan="XYZ",
            order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
        p = [M, mu, a, p0, e0, 1.0, dist, qS, phiS, qK_, phiK_, Pp, Pt, Pr]
        return np.atleast_2d(np.asarray(wg(*p, convert_to_ra_dec=False)))[:3]

    print("  building ECLIPTIC config...", flush=True)
    A = build("ecliptic", qS_e, phiS_e, qK_e, phiK_e)
    print("  building ICRS config...", flush=True)
    B = build("icrs", qS_i, phiS_i, qK_i, phiK_i)

    print("\n  TDI invariance (ecliptic vs ICRS, same physical source):", flush=True)
    for i, ch in enumerate("XYZ"):
        x, y = A[i], B[i]
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        denom = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)
        ov = np.vdot(x, y).real / denom if denom > 0 else 0.0
        reld = np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-300)
        print(f"    {ch}: overlap={ov:+.6f}  max-reldiff={reld:.3e}  "
              f"rmsA={np.sqrt(np.mean(x**2)):.3e} rmsB={np.sqrt(np.mean(y**2)):.3e}", flush=True)
    print("\n  => overlap~+1 & reldiff~0: FEW+response+orbit are frame-consistent (frame is NOT the data bug).")
    print("     => otherwise the frame handling (FEW psi vs response u/v vs orbit) is the bug; fix HERE first.")


if __name__ == "__main__":
    main()
