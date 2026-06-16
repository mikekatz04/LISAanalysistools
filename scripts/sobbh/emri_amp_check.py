"""Isolate the EMRI amplitude error WITHOUT the dense FEW summation (which OOMs on
this box; the holder path is fine). Reconstruct the source-frame strain from the SAME
sparse holder two ways and compare:
  (T) emritdionfly:  (1/2)|teuk||ylm|, +m and -m, approximate phase  (AMP_FACTOR=1/2)
  (C) complex +m:    sum teuk*ylm*exp(-i*phase) over the holder's m>=0 modes (unambiguous)
  (F) complex full:  (C) + conjugate-symmetric -m partner  (the true legacy strain)
median |h_T|/|h_C| and |h_T|/|h_F| tell us whether the 1/2 reproduces the real modes
or over/under-shoots. No orbit, no response -- pure amplitude/mode test.
"""
import os, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
DT = 20.0; TOBS_S = 86400.0; THRESH = 1e-3


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 5.0:
            os._exit(42)
        time.sleep(0.1)


def emritdionfly_recon(K, AMP):
    """emritdionfly mode extraction -> summed source-frame strain on K.t_arr."""
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    keep = K.ms != 0
    pz = (_mp - ylm_phase[:K.ms.shape[0]] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1).T
    az = AMP * mode_amp_amp * ylm_amp[:_mp.shape[1]]
    ab = (AMP * mode_amp_amp * ylm_amp[_mp.shape[1]:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1).T
    return (mode_amp * np.exp(-1j * mode_phase)).sum(0)


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

    print("  building holder...", flush=True)
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True},
        mode_selector_kwargs={"mode_selection_threshold": THRESH})
    K = wg(M, mu, a, p0, e0, 1.0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
           Phi_r0=Pr, T=TOBS_S, dt=DT, mode_selection_threshold=THRESH,
           return_sparse_holder=True, include_minus_mkn=True)

    nm = K.ms.shape[0]
    print(f"  holder: n_modes={nm}  teuk_modes={np.asarray(K.teuk_modes).shape}  "
          f"ylms={np.asarray(K.ylms).shape}  phases={np.asarray(K.phases).shape}", flush=True)
    for attr in ("ls", "ms", "ks", "ns"):
        if hasattr(K, attr):
            v = np.asarray(getattr(K, attr))
            print(f"    K.{attr}: shape={v.shape} range=[{v.min()},{v.max()}]", flush=True)

    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    # (C) complex +m sum from the holder (uses the actual complex teuk & ylm, no AMP guess)
    h_C = (K.teuk_modes * K.ylms[None, :nm] * np.exp(-1j * _mp)).sum(1)
    # (F) full: add the conjugate-symmetric -m partner. FEW symmetry: h_{l,-m} contribution
    #     = (-1)^l conj(teuk) * Y_{l,-m} * exp(+i*phase). Y_{l,-m} is holder ylms[nm:].
    keep = K.ms != 0
    if hasattr(K, "ls"):
        sgn = ((-1.0) ** np.asarray(K.ls))[None, :]
    else:
        sgn = 1.0
    h_M = (sgn[:, keep] * np.conj(K.teuk_modes[:, keep]) * K.ylms[None, nm:][:, keep]
           * np.exp(1j * _mp[:, keep])).sum(1)
    h_F = h_C + h_M

    for nameR, hR in (("complex_+m (C)", h_C), ("complex_full (F)", h_F)):
        good = np.abs(hR) > 0.1 * np.abs(hR).max()
        print(f"  RMS |{nameR}| = {np.sqrt(np.mean(np.abs(hR)**2)):.4e}", flush=True)
        for AMP in (0.5, 1.0, 0.25):
            hT = emritdionfly_recon(K, AMP)
            r = float(np.median(np.abs(hT[good]) / np.abs(hR[good])))
            print(f"    AMP_FACTOR={AMP}: median |h_T|/|{nameR}| = {r:.4f}", flush=True)


if __name__ == "__main__":
    main()
