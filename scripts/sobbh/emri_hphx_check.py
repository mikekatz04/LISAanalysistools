"""LINE-BY-LINE: does EMRITDIonFly's mode combination reproduce FEW's ACTUAL h+/hx?
Compare the class's get_hp_hc(inc=0) per-sub reconstruction (summed, psi-rotated) to
the GROUND TRUTH FEW h+/hx -- captured by wrapping GenerateEMRIWaveform(frame=detector)
inside ResponseWrapper (the direct summed call OOMs on this box; the ResponseWrapper
path runs). Per single mode (incl odd l), |O| + RMS ratio for h+ and hx SEPARATELY.
If h+ matches but hx/odd-l does not -> the +/-m combination drops the true Y_{l,-m}
phase (the class feeds the -m sub as -pz = negated +m phase).
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.interpolate import CubicSpline
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from few.utils.utility import get_viewing_angles, get_polarization_angle

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
SRC = 1; DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
MODES = [(2, 2, 0, 0), (3, 3, 0, 0), (3, 2, 0, 0), (2, 2, 0, 1), (4, 4, 0, 0), (5, 5, 0, 0)]


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


class ModeCapture:
    """Force a single mode (+ -m) and CAPTURE the FEW h+/hx that ResponseWrapper pulls."""
    def __init__(self, fg, mode):
        self.fg = fg; self.mode = mode; self.last = None
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("mode_selection", [self.mode])
        kwargs.setdefault("include_minus_mkn", True)
        h = self.fg(*args, **kwargs)
        self.last = np.asarray(h)
        return h
    def __getattr__(self, name):
        return getattr(self.fg, name)


def class_recon_on(K, psi, t_target):
    """EXACT EMRITDIonFly combination -> (hp, hc) at t_target, the way the response does
    it: cubic-spline the SMOOTH per-sub amp + phase, THEN form the oscillation densely
    (NOT cos(pz) on the coarse holder grid + interp -- that aliases)."""
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms); nm = K.ms.shape[0]
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    AMP = 0.5; keep = K.ms != 0
    pz = (_mp - ylm_phase[:nm] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1)        # (n_tK, n_sub), SMOOTH
    az = AMP * mode_amp_amp * ylm_amp[:nm]; ab = (AMP * mode_amp_amp * ylm_amp[nm:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1)                    # (n_tK, n_sub), SMOOTH
    tK = np.asarray(K.t_arr)
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    hp = np.zeros(len(t_target)); hc = np.zeros(len(t_target))
    for s in range(mode_amp.shape[1]):
        amp_t = CubicSpline(tK, mode_amp[:, s])(t_target)
        ph_t = CubicSpline(tK, mode_phase[:, s])(t_target)         # smooth phase -> spline -> dense
        hSp = -np.cos(ph_t) * amp_t * 2.0; hSc = -np.sin(ph_t) * amp_t * 2.0
        hp += hSp * c2 - hSc * s2; hc += hSp * s2 + hSc * c2
    out = (t_target < tK[0]) | (t_target > tK[-1]); hp[out] = 0.0; hc[out] = 0.0
    return hp, hc


def omax(a, b):
    """time+phase-maximized |O| of COMPLEX a,b (a psi rotation is just *e^{-2i psi},
    absorbed by phase-max), so this is psi-invariant and alignment-independent."""
    xc = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))
    return float(np.max(np.abs(xc)) / np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)))


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
    inj = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, qK_e, phiK_e, Pp, Pt, Pr]
    insp = {"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"}

    fp = find_file(EMRI_L1, "EMRI", SRC); data_t0 = float(MojitoL1File(fp).tdis.time_sampling.t0)
    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        frame="detector", inspiral_kwargs=insp, sum_kwargs={"pad_output": True}, force_backend="cpu")
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu", inspiral_kwargs=insp, sum_kwargs={"pad_output": True})

    print(f"\n  {'mode':14s} {'|O|max':>8s} {'RMS|h|':>8s}   (recon vs FEW, complex, psi-inv)", flush=True)
    for mode in MODES:
        try:
            cap = ModeCapture(fg, mode)
            wl = ResponseWrapper(cap, orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
                index_lambda=8, index_beta=7, flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"),
                tdi_chan="XYZ", order=25, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
            _ = wl(*inj, convert_to_ra_dec=False)               # triggers cap to record h+/hx
            hF = cap.last; tF = np.arange(len(hF)) * DT          # FEW complex h = h+ - i hx
            K = wg(M, mu, a, p0, e0, 1.0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt, Phi_r0=Pr,
                   T=TOBS_S, dt=DT, mode_selection=[mode], include_minus_mkn=True, return_sparse_holder=True)
            tK = np.asarray(K.t_arr)
            hp_R, hc_R = class_recon_on(K, 0.0, tF)   # spline amp/phase -> dense, no aliasing
            good = (tF >= tK[0]) & (tF <= tK[-1])
            h_R = (hp_R - 1j * hc_R)[good]; h_F = hF[good]
            O = omax(h_R, h_F)
            rmag = np.sqrt(np.mean(np.abs(h_R) ** 2)) / np.sqrt(np.mean(np.abs(h_F) ** 2))
            print(f"  {str(mode):14s} {O:8.4f} {rmag:8.4f}", flush=True)
        except Exception as e:
            print(f"  {str(mode):14s}  SKIP {type(e).__name__}: {str(e)[:50]}", flush=True)


if __name__ == "__main__":
    main()
