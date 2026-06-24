"""Validate the EMRI TDI-on-the-fly against mojito with a FINE upsampled trajectory.

Uses EMRITDIonFly's exact mode extraction + out.eval_tdi mixing, but trims the
per-mode response eval grid INSIDE the spline by the k*x delay (the fixed 1-sample
trim is too small for a fine trajectory) and zeros the TDI outside the spline
window. Fine trajectory (many points) is needed because the phase wants ~8th-order
accuracy and the response uses cubic splines.
"""
import os, sys, types, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.tdionfly import TDTDIonTheFly
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.directresponse import ecliptic_to_icrs
from lisatools.sources.utils import icrs_to_ecliptic
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_polarization_angle, get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664
SRC = 1
DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
T_BUF = 1000.0
N_PTS = int(os.environ.get("N_PTS", "4096"))   # sweepable: tests get_tdi spline/phi_ref density
DELAY = 800.0                 # > k*x SSB-projection delay (~472s) + TDI/interp margin
MODE_THRESH = float(os.environ.get("MODE_THRESH", "1e-7"))   # sweepable for the opt-SNR-vs-
#  modes dig. CALL-TIME arg (base.py:143). 1e-5(def)->180->|O|0.967; 1e-7->388->0.975;
#  1e-10->724->0.975 (PLATEAUED). opt SNR @1e-7 = 5.95 vs data 4.02 (1.48x).


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.5:
            os._exit(42)
        time.sleep(0.3)


def emri_tof(wave_gen, orbits, tdi_config, t0, p, n_trim):
    """EMRITDIonFly.__call__ verbatim, except t_arr_tdi trims by n_trim (not 1)."""
    M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = p
    theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
    psi = get_polarization_angle(qS, phiS, qK, phiK)
    ra, dec = ecliptic_to_icrs(phiS, np.pi / 2 - qS)
    K = wave_gen(M, mu, a, p0, e0, x0, theta, phi, dist=dist, Phi_phi0=Pp,
                 Phi_theta0=Pt, Phi_r0=Pr, T=TOBS_S, dt=DT, return_sparse_holder=True,
                 include_minus_mkn=True, mode_selection_threshold=MODE_THRESH)
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    AMP = 1 / 2.; keep = K.ms != 0
    pz = (_mp - ylm_phase[:K.ms.shape[0]] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1).T
    az = AMP * mode_amp_amp * ylm_amp[:_mp.shape[1]]
    ab = (AMP * mode_amp_amp * ylm_amp[_mp.shape[1]:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1).T
    t_arr_in = t0 + np.repeat(K.t_arr[:, None], mode_phase.shape[0], axis=-1).T
    t_arr_tdi = t_arr_in[:, n_trim:-n_trim]                      # MODIFIED: trim by delay
    num_sub = mode_amp.shape[0]
    tdi_gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, 1.0, num_sub, t_input=t_arr_in,
                            tdi_config=tdi_config, orbits=orbits)
    inc = np.zeros(num_sub); psi_in = np.full(num_sub, psi)
    ra_in = np.full(num_sub, ra); dec_in = np.full(num_sub, dec)
    return tdi_gen(inc, psi_in, ra_in, dec_in, return_spline=True), num_sub


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
    # SPECIAL EMRITDIonFly setup: ecliptic-POLAR sky (for FEW viewing/psi) + RAW FILE spin
    # (qK,phiK straight from catalogue, NOT ecliptic-converted); the emri_tof reconstruction
    # already feeds the response ra,dec = ecliptic_to_icrs(phiS_e, pi/2-qS_e) with ICRS orbits.
    inj = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, float(qK), float(phiK), Pp, Pt, Pr]

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf for c in ("X2", "Y2", "Z2")])

    print("  building ICRS orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="icrs")
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    off = data_t0 - REF
    new_t = np.linspace(off + T_BUF, off + TOBS_S - T_BUF, N_PTS)
    dt_traj = new_t[1] - new_t[0]; n_trim = int(np.ceil(DELAY / dt_traj)) + 2
    print(f"  N_PTS={N_PTS} dt_traj~{dt_traj:.1f}s  n_trim={n_trim} (~{n_trim*dt_traj:.0f}s inside)", flush=True)
    wave_gen = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                         "upsample": True, "fix_t": True, "new_t": new_t},
        sum_kwargs={"pad_output": True}, mode_selector_kwargs={"mode_selection_threshold": MODE_THRESH})
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")

    print("  running EMRI TDI-on-the-fly (fine traj, delay-trim, zero outside)...", flush=True)
    out, num_sub = emri_tof(wave_gen, orb, tdi_config, REF, inj, n_trim)
    print(f"  num modes={num_sub}", flush=True)

    tgrid = data_t0 + np.arange(N_WIN) * DT
    tg = np.asarray(out.x); t_lo, t_hi = float(np.max(tg[:, 0])), float(np.min(tg[:, -1]))
    inside = (tgrid > t_lo) & (tgrid < t_hi)
    _tdi = out.eval_tdi(tgrid[inside]); tdi_sum = np.real(np.sum(np.asarray(_tdi), axis=0))
    lat = np.zeros((3, N_WIN)); lat[:, inside] = tdi_sum               # ZERO outside spline window
    print(f"  eval window inside spline: {int(inside.sum())}/{N_WIN} pts", flush=True)

    win = tukey(N_WIN, 0.1); ff = np.fft.rfftfreq(N_WIN, d=DT); taus = np.linspace(-1500, 1500, 601)
    for ci, ch in enumerate("XYZ"):
        A = np.fft.rfft(lat[ci] * win); B = np.fft.rfft(dXYZ[ci] * win)
        nm = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum())
        O = max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nm for t in taus)
        print(f"    {ch}: time-scan TDI |O|={O:.4f}  (coarse-traj was ~0.967, pyResponse ~0.92)", flush=True)

    # ---- noise-weighted LIKELIHOOD at injection (full XYZ, SciRD) ----
    from lisatools.domains import TDSettings, FDSettings, TDSignal
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    tmpl_fd = TDSignal(lat, td_set).transform(fd_set, window=win)
    fb = np.asarray(fd_set.f_arr); base = np.asarray(tmpl_fd.arr).copy()
    opt, _ = ac.template_snr(tmpl_fd)                    # optimal SNR (tau-invariant)
    best = (-9e99, 0.0, 0.0)
    for t in np.linspace(-1500, 1500, 301):
        tmpl_fd.arr[:] = base * np.exp(2j * np.pi * fb * t)[None, :]
        _, det = ac.template_snr(tmpl_fd)               # detected SNR <d|h>/sqrt<h|h>
        logL = det * opt - 0.5 * opt ** 2
        if logL > best[0]:
            best = (logL, t, det)
    tmpl_fd.arr[:] = base
    dd = float(ac.inner_product().real); logL, tau_b, det = best
    print(f"  LIKELIHOOD (SciRD, full XYZ, best tau={tau_b:+.0f}s):", flush=True)
    print(f"    opt SNR={opt:.3f}  det SNR={det:.3f}  data SNR={np.sqrt(dd):.3f}  "
          f"SNR recovered={det/np.sqrt(dd):.4f}", flush=True)
    # EMRI is low SNR -> small SNR^2 in the likelihood, so dlogL(loss) is modest even at |O|~0.97
    print(f"    logL={logL:.2f}  inj logL=0.5<d|d>={0.5*dd:.2f}  dlogL(loss vs inj)={0.5*dd-logL:.3f}", flush=True)


if __name__ == "__main__":
    main()
