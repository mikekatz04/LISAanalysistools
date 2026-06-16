"""SOBBH frame self-consistency test (NO mojito data) -- isolates the shared
orbit+response frame handling from FEW.

Same physical SOBBH source in two frames: ICRS (lam=RA, beta=Dec, psi) with an
ICRS orbit, vs ecliptic (lam,beta,psi via icrs_to_ecliptic) with an ecliptic
orbit. The TDI MUST be invariant. If it is, the orbit+response frame machinery
is sound and the EMRI failure is FEW-specific (psi/viewing-angle). If it ALSO
fails, the frame bug is in the shared response/orbit, not FEW.
"""
import os, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.sources.utils import icrs_to_ecliptic
from lisatools.utils.constants import YRSID_SI

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
REF = 97729089.327664
SRC = 0
DT = 10.0; N_WIN = 32768; TOBS = N_WIN * DT


def main():
    cat = os.path.join(PATH, "catalogues", "sobhb_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        m1, m2 = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame")
        s1, s2 = g("PrimarySpinCompZ"), g("SecondarySpinCompZ")
        dist, inc = g("LuminosityDistance") / 1e3, g("InclinationAngle")
        f_low = g("GW22FrequencySSBFrame")
        ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
        psi, phi0 = g("PolarisationAngle") % np.pi, g("TrueAnomaly")

    # ICRS (validated SOBBH frame) and ecliptic via icrs_to_ecliptic (handles psi)
    lam_e, beta_e, psi_e = icrs_to_ecliptic(ra, dec, psi)
    lam_e = float(lam_e) % (2 * np.pi); beta_e = float(beta_e); psi_e = float(psi_e) % np.pi
    print(f"  ICRS: lam={ra:.4f} beta={dec:.4f} psi={psi:.4f}")
    print(f"  ECL : lam={lam_e:.4f} beta={beta_e:.4f} psi={psi_e:.4f}")

    fp = find_file(SOBHB_L1, "SOBHB", SRC)
    data_t0 = float(MojitoL1File(fp).tdis.time_sampling.t0)
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    gen = SOBBHWaveform(TOBS, DT, t0=data_t0, reference_time=REF, force_backend="cpu")

    def build(frame, lam, beta, ps):
        orb = L1Orbits(fp, force_backend="cpu", frame=frame)
        pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
        lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
        orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
        orb.configure(linear_interp_setup=True)
        wg = ResponseWrapper(gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
            index_lambda=7, index_beta=8, flip_hx=False, tdi=tdi_config, tdi_chan="XYZ",
            order=40, remove_garbage="zero", is_ecliptic_latitude=True, t_buffer=3e4,
            force_backend="cpu")
        p = [m1, m2, s1, s2, dist, inc, f_low, lam, beta, ps, phi0]
        return np.atleast_2d(np.asarray(wg(*p, convert_to_ra_dec=False)))[:3]

    print("  building ICRS config...", flush=True)
    B = build("icrs", ra, dec, psi)
    print("  building ECLIPTIC config...", flush=True)
    A = build("ecliptic", lam_e, beta_e, psi_e)

    print("\n  SOBBH TDI invariance (ICRS vs ecliptic, same physical source):")
    for i, ch in enumerate("XYZ"):
        x, y = A[i], B[i]
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        denom = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)
        ov = np.vdot(x, y).real / denom if denom > 0 else 0.0
        reld = np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-300)
        print(f"    {ch}: overlap={ov:+.6f}  max-reldiff={reld:.3e}")
    print("\n  => overlap~+1: shared orbit+response frame machinery is sound (EMRI fail is FEW-specific).")
    print("     => overlap<1: the frame bug is in the SHARED response/orbit, not FEW.")


if __name__ == "__main__":
    main()
