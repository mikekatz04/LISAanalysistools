"""Convention check for the EMRI TDI-on-the-fly source feed -- the EMRI analog of
sobbh_tdionfly_convention_check.py, and the strain-level check emritdionfly.py's
``AMP_FACTOR = 1/2  # CHECK THIS`` never had.

Verifies that the emritdionfly per-mode reconstruction reproduces FEW's OWN
``directmodesum`` waveform (w1 + w2, the full +/-m sum) EXACTLY at AMP_FACTOR=1.0,
and DOCUMENTS the production 0.5 as a factor-2 discrepancy vs FEW. Pure holder-level,
no orbit/TDI (the dense FEW summed path OOMs on small RAM; the holder path is fine).

FEW reference (FastEMRIWaveforms summation/directmodesum.py:90-126 + utils/ylm.py:330):
    w1 = sum  ylms[:nm] * teuk            * exp(-i*(m*Phi_phi + k*Phi_theta + n*Phi_r))
    w2 = sum  (-1)^l * ylms[nm:] * conj(teuk) * exp(+i*(...)),  over m > 0
    waveform = w1 + w2            # NO 1/2 and NO 2 anywhere

If a future edit to the emritdionfly mode extraction (phase signs, -m handling, ylm
indexing) drifts, the AMP=1.0 rel err blows up and this FAILS. See
project_emri_mojito_frame_debug / project_tdionfly_convention_check_lesson.
"""
import os, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
DT = 20.0; TOBS_S = 86400.0; THRESH = 1e-3
AMP_FACTOR_PRODUCTION = 0.5      # the value currently in scripts/emri/emritdionfly.py
TOL = 1e-8                       # ~2e-10 float accumulation over modes; an O(1) bug dwarfs this


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 5.0:
            os._exit(42)
        time.sleep(0.1)


def emritdionfly_recon(K, AMP):
    """scripts/emri/emritdionfly.py mode extraction -> summed strain on K.t_arr."""
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


def few_directmodesum(K):
    """FEW directmodesum w1 + w2 from the same holder (the legacy reference)."""
    nm = K.ms.shape[0]
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    w1 = (K.ylms[None, :nm] * K.teuk_modes * np.exp(-1j * _mp)).sum(1)        # m >= 0
    keep = K.ms > 0
    sgn = ((-1.0) ** np.asarray(K.ls))[None, :]
    w2 = (sgn[:, keep] * K.ylms[None, nm:][:, keep] * np.conj(K.teuk_modes[:, keep])
          * np.exp(1j * _mp[:, keep])).sum(1)                                 # m < 0 partner
    return w1 + w2


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

    h_ref = few_directmodesum(K)               # FEW's own legacy strain
    good = np.abs(h_ref) > 0.1 * np.abs(h_ref).max()
    rel = float(np.max(np.abs(emritdionfly_recon(K, 1.0)[good] - h_ref[good]) / np.abs(h_ref[good])))
    ratio_prod = float(np.median(np.abs(emritdionfly_recon(K, AMP_FACTOR_PRODUCTION)[good])
                                 / np.abs(h_ref[good])))
    ok = rel < TOL

    print(f"  n modes compared       : {K.ms.shape[0]}  (ylms 2*nm = {np.asarray(K.ylms).shape[0]})")
    print(f"  rel err recon(AMP=1.0) vs FEW directmodesum = {rel:.2e}  (tol {TOL:.0e})")
    print(f"  production AMP_FACTOR={AMP_FACTOR_PRODUCTION} -> strain is {ratio_prod:.3f}x FEW "
          f"(KNOWN factor-2; see project_emri_mojito_frame_debug)")
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(emritdionfly recon == FEW w1+w2 at AMP_FACTOR=1.0)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
