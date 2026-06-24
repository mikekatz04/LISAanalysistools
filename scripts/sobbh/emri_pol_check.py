"""Polarization check: the strain MAGNITUDE matched FEW (|h_TOF|/|h_FEW|=1.0) but the
single-mode legacy-vs-TOF showed a per-mode ~1.48x + |O|=0.93. Hypothesis: inc=0 forces
each mode CIRCULAR (hp,hc equal-amp quadratures) so the +/-m feed mis-splits plus vs
cross -- right magnitude, wrong polarization. Compare hp vs FEW h+ and hc vs FEW hx
SEPARATELY (psi=0, source frame). If Im (cross) mismatches while |.| matched, that's it.
"""
import os, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
DT = 20.0; TOBS_S = 86400.0; AMP = 0.5
SEL = os.environ.get("SEL", "single")     # "single" -> [(2,2,0,0)] ; "full" -> threshold 1e-3


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 5.0:
            os._exit(42)
        time.sleep(0.1)


def directmodesum(K):
    nm = K.ms.shape[0]
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    w1 = (K.ylms[None, :nm] * K.teuk_modes * np.exp(-1j * _mp)).sum(1)
    keep = K.ms > 0
    sgn = ((-1.0) ** np.asarray(K.ls))[None, :]
    w2 = (sgn[:, keep] * K.ylms[None, nm:][:, keep] * np.conj(K.teuk_modes[:, keep])
          * np.exp(1j * _mp[:, keep])).sum(1)
    return w1 + w2


def tof_hp_hc(K, psi=0.0):
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    keep = K.ms != 0
    pz = (_mp - ylm_phase[:K.ms.shape[0]] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1)
    az = AMP * mode_amp_amp * ylm_amp[:_mp.shape[1]]
    ab = (AMP * mode_amp_amp * ylm_amp[_mp.shape[1]:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1)
    hSp = -np.cos(mode_phase) * mode_amp * 2.0
    hSc = -np.sin(mode_phase) * mode_amp * 2.0
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    hp = (hSp * c2 - hSc * s2).sum(1); hc = (hSp * s2 + hSc * c2).sum(1)
    return hp, hc      # source-frame plus, cross


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def overlap(a, b):
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    return float(abs((np.conj(B) * A).sum()) / np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum()))


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][1])
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

    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True})
    sel_kw = (dict(mode_selection=[(2, 2, 0, 0)]) if SEL == "single"
              else dict(mode_selection_threshold=1e-3))
    print(f"  SEL={SEL}  {sel_kw}", flush=True)
    K = wg(M, mu, a, p0, e0, 1.0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
           Phi_r0=Pr, T=TOBS_S, dt=DT, return_sparse_holder=True, include_minus_mkn=True, **sel_kw)

    hF = directmodesum(K)
    hp_F, hx_F = hF.real, -hF.imag                  # FEW source-frame plus, cross
    hp_T, hc_T = tof_hp_hc(K, psi=0.0)              # TOF (get_hp_hc inc=0) plus, cross
    print(f"  n_modes(m>=0)={K.ms.shape[0]}  ms={np.asarray(K.ms)} ns={np.asarray(K.ns)}", flush=True)
    print(f"  PLUS  : |O|(hp_TOF, h+_FEW)={overlap(hp_T, hp_F):.4f}   "
          f"RMS hp_TOF/h+_FEW={rms(hp_T)/rms(hp_F):.4f}", flush=True)
    print(f"  CROSS : |O|(hc_TOF, hx_FEW)={overlap(hc_T, hx_F):.4f}   "
          f"RMS hc_TOF/hx_FEW={rms(hc_T)/rms(hx_F):.4f}", flush=True)
    print(f"  cross/plus power  TOF={rms(hc_T)/rms(hp_T):.4f}   FEW={rms(hx_F)/rms(hp_F):.4f}  "
          f"(equal => polarization preserved)", flush=True)
    print(f"  |O|(h_TOF, h_FEW) complex={overlap(np.r_[hp_T, hc_T], np.r_[hp_F, hx_F]):.4f}", flush=True)


if __name__ == "__main__":
    main()
