"""Single-mode (2,2,0) diagnostic: legacy pyResponse vs TOF on the SAME ecliptic
orbit/window/data, but FORCING mode_selection=[(2,2,0)] on both. The shared response
is validated by single-mode sources (SOBBH/MBH), so:
  legacy == TOF on one mode  -> the 1.49x is in emritdionfly's MULTI-MODE handling
                               (how the set of modes / +-m subs is built & fed).
  legacy != TOF on one mode  -> the per-mode amp/phase construction in emritdionfly
                               is wrong (mode extraction itself).
Reports per-channel RMS(TOF)/RMS(legacy), opt SNR each vs data, |O|(TOF,legacy).
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdionfly import TDTDIonTheFly
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from few.utils.utility import get_polarization_angle, get_viewing_angles

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664; SRC = 1
DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
T_BUF = 1000.0; N_PTS = 4096; DELAY = 800.0
MODE_SEL = [(2, 2, 0, 0)]                     # (l, m, k, n) -- FEW wants 4 indices


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


class FixedModeGen:
    """Wrap a FEW generator to force a fixed mode_selection (for the legacy path)."""
    def __init__(self, fg, mode_sel):
        self.fg = fg; self.mode_sel = mode_sel
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("mode_selection", self.mode_sel)
        return self.fg(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.fg, name)


def emri_tof_ecl(wave_gen, orbits, tdi_config, t0, p, n_trim):
    M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = p
    theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
    psi = float(get_polarization_angle(qS, phiS, qK, phiK))
    lam, beta = float(phiS), float(np.pi / 2 - qS)
    K = wave_gen(M, mu, a, p0, e0, x0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                 Phi_r0=Pr, T=TOBS_S, dt=DT, return_sparse_holder=True,
                 include_minus_mkn=True, mode_selection=MODE_SEL)
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    AMP = 0.5; keep = K.ms != 0
    pz = (_mp - ylm_phase[:K.ms.shape[0]] - mode_amp_phase)
    mode_phase = np.concatenate([pz, -pz[:, keep]], axis=-1).T
    az = AMP * mode_amp_amp * ylm_amp[:_mp.shape[1]]
    ab = (AMP * mode_amp_amp * ylm_amp[_mp.shape[1]:])[:, keep]
    mode_amp = np.concatenate([az, ab], axis=-1).T
    t_arr_in = t0 + np.repeat(K.t_arr[:, None], mode_phase.shape[0], axis=-1).T
    t_arr_tdi = t_arr_in[:, n_trim:-n_trim]
    num_sub = mode_amp.shape[0]
    print(f"    TOF holder: n_modes(m>=0)={K.ms.shape[0]}  subs(incl -m)={num_sub}  "
          f"ms={np.asarray(K.ms)} ns={np.asarray(K.ns)}", flush=True)
    tdi_gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, 1.0, num_sub, t_input=t_arr_in,
                            tdi_config=tdi_config, orbits=orbits)
    return tdi_gen(np.zeros(num_sub), np.full(num_sub, psi), np.full(num_sub, lam),
                   np.full(num_sub, beta), return_spline=True), num_sub


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
    inj = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, qK_e, phiK_e, Pp, Pt, Pr]

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit (shared)...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    off = data_t0 - REF
    new_t = np.linspace(off + T_BUF, off + TOBS_S - T_BUF, N_PTS)
    dt_traj = new_t[1] - new_t[0]; n_trim = int(np.ceil(DELAY / dt_traj)) + 2
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                         "upsample": True, "fix_t": True, "new_t": new_t},
        sum_kwargs={"pad_output": True})
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    print(f"  building TOF, mode_selection={MODE_SEL}...", flush=True)
    out, num_sub = emri_tof_ecl(wg, orb, tdi_config, REF, inj, n_trim)
    tgrid = data_t0 + np.arange(N_WIN) * DT
    tg = np.asarray(out.x); t_lo, t_hi = float(np.max(tg[:, 0])), float(np.min(tg[:, -1]))
    inside = (tgrid > t_lo) & (tgrid < t_hi)
    _tdi = out.eval_tdi(tgrid[inside]); tof = np.zeros((3, N_WIN)); tof[:, inside] = np.real(np.sum(np.asarray(_tdi), axis=0))

    print(f"  building LEGACY pyResponse, mode_selection={MODE_SEL}...", flush=True)
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector", force_backend="cpu")
    wl = ResponseWrapper(FixedModeGen(fg, MODE_SEL), orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ", order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
    leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
    leg = (np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1]))) if leg.shape[-1] < N_WIN else leg[:, :N_WIN])

    win = tukey(N_WIN, 0.1)
    print("\n  per-channel RMS(TOF)/RMS(legacy):", flush=True)
    for ci, ch in enumerate("XYZ"):
        rT = np.sqrt(np.mean((tof[ci, inside]) ** 2)); rL = np.sqrt(np.mean((leg[ci, inside]) ** 2))
        print(f"    {ch}: RMS_TOF={rT:.3e}  RMS_leg={rL:.3e}  ratio={rT/rL:.4f}", flush=True)

    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    dd = float(ac.inner_product().real)
    print(f"\n  opt SNR vs data (data SNR={np.sqrt(dd):.3f}):", flush=True)
    for tag, sig in [("TOF", tof), ("legacy", leg)]:
        s_fd = TDSignal(sig, td_set).transform(fd_set, window=win)
        opt, _ = ac.template_snr(s_fd)
        print(f"    {tag:7s} opt SNR={float(opt):.3f}  opt/data={float(opt)/np.sqrt(dd):.3f}", flush=True)

    ff = np.fft.rfftfreq(N_WIN, d=DT); taus = np.linspace(-1500, 1500, 601)
    print("\n  |O|(TOF vs legacy) time-scanned  [single mode -> should be ~1 if per-mode OK]:", flush=True)
    for ci, ch in enumerate("XYZ"):
        A = np.fft.rfft(tof[ci] * win); B = np.fft.rfft(leg[ci] * win)
        nm = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum())
        O = max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nm for t in taus)
        print(f"    {ch}: |O|(TOF,legacy)={O:.4f}", flush=True)


if __name__ == "__main__":
    main()
