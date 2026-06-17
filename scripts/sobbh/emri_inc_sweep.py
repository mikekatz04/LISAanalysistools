"""Sweep amp/inc on the REAL EMRI: run the actual EMRITDIonFly feed (FEW modes ->
TDTDIonTheFly) but vary the inc fed to the response and the AMP_FACTOR, vs the mojito
data. See how opt/data and |O|(TOF,data) move -- does some (inc, amp) match the data
(legacy opt/data = 0.994)? Reference: at (inc=0, AMP=0.5) it's the production 1.49x.
"""
import os, time, threading, resource, gc
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
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_polarization_angle, get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664; SRC = 1
DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
T_BUF = 1000.0; N_PTS = 4096; DELAY = 800.0; THRESH = 1e-3


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


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
    M_, mu_, a_, p0_, e0_ = M, mu, a, p0, e0
    theta, phi = get_viewing_angles(qS_e, phiS_e, qK_e, phiK_e)
    psi = float(get_polarization_angle(qS_e, phiS_e, qK_e, phiK_e))
    lam, beta = phiS_e, float(np.pi / 2 - qS_e)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    off = data_t0 - REF
    new_t = np.linspace(off + T_BUF, off + TOBS_S - T_BUF, N_PTS); dt_traj = new_t[1] - new_t[0]
    n_trim = int(np.ceil(DELAY / dt_traj)) + 1
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                         "upsample": True, "fix_t": True, "new_t": new_t},
        sum_kwargs={"pad_output": True}, mode_selector_kwargs={"mode_selection_threshold": THRESH})
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    print("  generating FEW holder (once)...", flush=True)
    K = wg(M_, mu_, a_, p0_, e0_, 1.0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt, Phi_r0=Pr,
           T=TOBS_S, dt=DT, mode_selection_threshold=THRESH, return_sparse_holder=True, include_minus_mkn=True)
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms); nm = K.ms.shape[0]
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    keep = K.ms != 0
    pz = (_mp - ylm_phase[:nm] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1).T
    t_arr_in = REF + np.repeat(K.t_arr[:, None], mode_phase.shape[0], axis=-1).T
    t_arr_tdi = t_arr_in[:, n_trim:-n_trim]; num_sub = mode_phase.shape[0]
    print(f"  num_sub={num_sub}", flush=True)

    win = tukey(N_WIN, 0.1); tgrid = data_t0 + np.arange(N_WIN) * DT
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    dd = float(ac.inner_product().real); dsnr = np.sqrt(dd)
    ff = np.fft.rfftfreq(N_WIN, d=DT); taus = np.linspace(-1500, 1500, 301)

    def run(inc_val, AMP):
        az = AMP * mode_amp_amp * ylm_amp[:nm]; ab = (AMP * mode_amp_amp * ylm_amp[nm:])[:, keep]
        mode_amp = np.concatenate([az, ab], axis=-1).T
        gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, 1.0 / DT, num_sub, t_input=t_arr_in,
                            tdi_config=tdi_config, orbits=orb)
        out = gen(np.full(num_sub, inc_val), np.full(num_sub, psi), np.full(num_sub, lam),
                  np.full(num_sub, beta), return_spline=True)
        tg = np.asarray(out.x); inside = (tgrid > float(tg[:, 0].max()) + DELAY) & (tgrid < float(tg[:, -1].min()) - DELAY)
        tdi = np.zeros((3, N_WIN)); tdi[:, inside] = np.real(np.sum(np.asarray(out.eval_tdi(tgrid[inside])), axis=0))
        opt = float(ac.template_snr(TDSignal(tdi, td_set).transform(fd_set, window=win))[0])
        O = []
        for ci in range(3):
            A = np.fft.rfft(tdi[ci] * win); B = np.fft.rfft(dXYZ[ci] * win)
            nmz = np.sqrt((np.abs(A)**2).sum() * (np.abs(B)**2).sum())
            O.append(max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nmz for t in taus))
        return opt, float(np.mean(O))

    print(f"\n  data SNR={dsnr:.3f}  (legacy opt/data=0.994 reference)", flush=True)
    print(f"\n  INC sweep at AMP_FACTOR=0.5:", flush=True)
    print(f"  {'inc[deg]':>9s} {'opt/data':>9s} {'|O|':>7s}", flush=True)
    for inc in [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2 - 0.02]:
        o, O = run(inc, 0.5)
        print(f"  {np.degrees(inc):9.1f} {o/dsnr:9.4f} {O:7.4f}", flush=True)
    print(f"\n  AMP_FACTOR sweep at inc=0:", flush=True)
    for AMP in [0.5, 0.34, 0.25]:
        o, O = run(0.0, AMP)
        print(f"  AMP={AMP:.3f}  opt/data={o/dsnr:.4f}  |O|={O:.4f}", flush=True)


if __name__ == "__main__":
    main()
