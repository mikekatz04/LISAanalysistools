"""Isolate the EMRI TOF amplitude bug at the STRAIN level (no orbit/TDI). Replicate
TDTDIonTheFly's per-mode get_hp_hc(inc=0, psi) feed on the emritdionfly modes and
compare the reconstructed h+ - i*hx to FEW's directmodesum (w1+w2). The mode-sum
reconstruction is known to be 0.5*FEW (AMP_FACTOR), and get_hp_hc(inc=0) multiplies
by (1+cos^2 0)=2 -- so IF the cancellation were clean, h_TOF == FEW. It isn't (the
end-to-end TOF is 1.49x the legacy). This shows WHY, by mode parity.

get_hp_hc (lat_tdi_on_the_fly.cu), inc=0 -> cosi=1:
    hSp = -cos(phase) * amp * (1 + cosi^2) = -2 amp cos(phase)
    hSc = -sin(phase) * 2 * amp * cosi     = -2 amp sin(phase)
    hp  = hSp cos2psi - hSc sin2psi ;  hc = hSp sin2psi + hSc cos2psi
The response consumes (hp, hc); the source-frame complex strain is hp - i*hc.

KEY suspicion: get_hp_hc carries NO (-1)^l, but FEW's w2 (the -m partner) DOES.
emritdionfly feeds +m AND -m as separate modes -> for odd l the -m sign flips vs
FEW, so the +/-m pair miscombines.
"""
import os, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
DT = 20.0; TOBS_S = 86400.0; THRESH = 1e-3; AMP = 0.5


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 5.0:
            os._exit(42)
        time.sleep(0.1)


def few_directmodesum(K, l_sel=None):
    """FEW w1 + w2 from the holder. l_sel: optional bool mask on the m>=0 modes."""
    nm = K.ms.shape[0]
    sel = np.ones(nm, bool) if l_sel is None else l_sel
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    w1 = (K.ylms[None, :nm][:, sel] * K.teuk_modes[:, sel] * np.exp(-1j * _mp[:, sel])).sum(1)
    keep = (K.ms > 0) & sel
    sgn = ((-1.0) ** np.asarray(K.ls))[None, :]
    w2 = (sgn[:, keep] * K.ylms[None, nm:][:, keep] * np.conj(K.teuk_modes[:, keep])
          * np.exp(1j * _mp[:, keep])).sum(1)
    return w1 + w2


def tof_via_get_hp_hc(K, psi=0.0):
    """emritdionfly modes -> per-mode get_hp_hc(inc=0) -> summed h+ - i*hx."""
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    keep = K.ms != 0
    pz = (_mp - ylm_phase[:K.ms.shape[0]] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1)         # (n_time, n_sub)
    az = AMP * mode_amp_amp * ylm_amp[:_mp.shape[1]]
    ab = (AMP * mode_amp_amp * ylm_amp[_mp.shape[1]:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1)                     # (n_time, n_sub)
    # get_hp_hc per mode, inc=0
    hSp = -np.cos(mode_phase) * mode_amp * 2.0
    hSc = -np.sin(mode_phase) * mode_amp * 2.0
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    hp = hSp * c2 - hSc * s2; hc = hSp * s2 + hSc * c2
    return (hp - 1j * hc).sum(1)


def rmag(x, ref):
    good = np.abs(ref) > 0.1 * np.abs(ref).max()
    return float(np.median(np.abs(x[good]) / np.abs(ref[good])))


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
        sum_kwargs={"pad_output": True}, mode_selector_kwargs={"mode_selection_threshold": THRESH})
    K = wg(M, mu, a, p0, e0, 1.0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
           Phi_r0=Pr, T=TOBS_S, dt=DT, mode_selection_threshold=THRESH,
           return_sparse_holder=True, include_minus_mkn=True)
    ls = np.asarray(K.ls); even = (ls % 2 == 0); odd = ~even

    h_FEW = few_directmodesum(K)
    h_TOF = tof_via_get_hp_hc(K, psi=0.0)
    print(f"  n_modes={K.ms.shape[0]}  (even-l {even.sum()}, odd-l {odd.sum()})", flush=True)
    print(f"  |h_TOF(get_hp_hc,inc=0,AMP=0.5)| / |h_FEW| = {rmag(h_TOF, h_FEW):.4f}", flush=True)
    print(f"     (clean cancellation would be 1.0; the end-to-end TOF was 1.49x legacy)", flush=True)
    # parity split: does the discrepancy live in odd-l (the (-1)^l sign get_hp_hc lacks)?
    h_FEW_e = few_directmodesum(K, even); h_FEW_o = few_directmodesum(K, odd)
    print(f"  even-l: |h_FEW_even| RMS={np.sqrt(np.mean(np.abs(h_FEW_e)**2)):.3e}  "
          f"odd-l: |h_FEW_odd| RMS={np.sqrt(np.mean(np.abs(h_FEW_o)**2)):.3e}  "
          f"(odd/even power {np.mean(np.abs(h_FEW_o)**2)/np.mean(np.abs(h_FEW_e)**2):.3f})", flush=True)
    # what if we DROP the -m feed (only m>=0 through get_hp_hc)? get_hp_hc's cos/sin
    # already supplies both quadratures, so the explicit -m may be the double-count.
    print(f"  RMS |h_FEW|={np.sqrt(np.mean(np.abs(h_FEW)**2)):.3e}  "
          f"RMS |h_TOF|={np.sqrt(np.mean(np.abs(h_TOF)**2)):.3e}", flush=True)


if __name__ == "__main__":
    main()
