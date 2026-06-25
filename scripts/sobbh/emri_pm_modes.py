"""Isolate the +m vs -m handling. With include_minus_m=False, force a SINGLE mode on
BOTH legacy and TOF and compare:
   (A) (2, 2,0,0)  -- pure +m
   (B) (2,-2,0,0)  -- pure -m  (different SWSH Y_{l,-m}; this is why +/-m are fed
                                 separately with inc=0)
If +m matches (legacy==TOF) but -m does NOT -> the bug is the -m branch of the
emritdionfly feed (a missing complex conjugate in how the -m mode amp/phase is built).
Reports per mode: source |teuk||ylm| envelope, opt SNR each, TOF/legacy, |O|(TOF,legacy).
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
T_BUF = 1000.0; N_PTS = 4096; DELAY = 800.0; AMP = 0.5


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


class FixedModeGen:
    def __init__(self, fg, mode_sel):
        self.fg = fg; self.mode_sel = mode_sel
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("mode_selection", self.mode_sel)
        kwargs.setdefault("include_minus_mkn", False)
        return self.fg(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.fg, name)


def tof_one(wave_gen, orbits, tdi_config, t0, p, n_trim, mode_sel):
    """emritdionfly feed, include_minus_m=False: feed ONLY the selected mode(s), no -m concat."""
    M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = p
    theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
    psi = float(get_polarization_angle(qS, phiS, qK, phiK))
    lam, beta = float(phiS), float(np.pi / 2 - qS)
    K = wave_gen(M, mu, a, p0, e0, x0, theta, phi, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                 Phi_r0=Pr, T=TOBS_S, dt=DT, return_sparse_holder=True,
                 include_minus_mkn=False, mode_selection=mode_sel)
    mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0); mode_amp_amp = np.abs(K.teuk_modes)
    ylm_phase = np.angle(K.ylms); ylm_amp = np.abs(K.ylms)
    nm = K.ms.shape[0]
    _mp = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
           + K.ns[None, :] * K.phases[:, 2][:, None])
    pz = (_mp - ylm_phase[:nm] - mode_amp_phase)
    mode_phase = pz.T                                       # only the selected mode(s)
    mode_amp = (AMP * mode_amp_amp * ylm_amp[:nm]).T
    env = float(np.sqrt(np.mean((mode_amp_amp * ylm_amp[:nm]) ** 2)))   # source |teuk||ylm| RMS
    t_arr_in = t0 + np.repeat(K.t_arr[:, None], mode_phase.shape[0], axis=-1).T
    t_arr_tdi = t_arr_in[:, n_trim:-n_trim]; num_sub = mode_amp.shape[0]
    tdi_gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, 1.0, num_sub, t_input=t_arr_in,
                            tdi_config=tdi_config, orbits=orbits)
    out = tdi_gen(np.zeros(num_sub), np.full(num_sub, psi), np.full(num_sub, lam),
                  np.full(num_sub, beta), return_spline=True)
    return out, num_sub, env, (np.asarray(K.ms), np.asarray(K.ns))


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

    win = tukey(N_WIN, 0.1); tgrid = data_t0 + np.arange(N_WIN) * DT
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    ff = np.fft.rfftfreq(N_WIN, d=DT); taus = np.linspace(-1500, 1500, 601)

    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector", force_backend="cpu")

    for mode_sel in ([(2, 2, 0, 0)], [(2, -2, 0, 0)]):
        print(f"\n========== mode {mode_sel}  (include_minus_m=False) ==========", flush=True)
        out, num_sub, env, (ms, ns) = tof_one(wg, orb, tdi_config, REF, inj, n_trim, mode_sel)
        tg = np.asarray(out.x); t_lo, t_hi = float(np.max(tg[:, 0])), float(np.min(tg[:, -1]))
        inside = (tgrid > t_lo) & (tgrid < t_hi)
        _tdi = out.eval_tdi(tgrid[inside]); tof = np.zeros((3, N_WIN)); tof[:, inside] = np.real(np.sum(np.asarray(_tdi), axis=0))
        print(f"  TOF: subs={num_sub} ms={ms} ns={ns}  source |teuk||ylm| RMS={env:.4e}", flush=True)

        wl = ResponseWrapper(FixedModeGen(fg, mode_sel), orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
            index_lambda=8, index_beta=7, flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"),
            tdi_chan="XYZ", order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
        leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
        leg = (np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1]))) if leg.shape[-1] < N_WIN else leg[:, :N_WIN])

        s_tof = TDSignal(tof, td_set).transform(fd_set, window=win); s_leg = TDSignal(leg, td_set).transform(fd_set, window=win)
        oT, _ = ac.template_snr(s_tof); oL, _ = ac.template_snr(s_leg)
        rms_r = [np.sqrt(np.mean(tof[c, inside] ** 2)) / np.sqrt(np.mean(leg[c, inside] ** 2)) for c in range(3)]
        Os = []
        for c in range(3):
            A = np.fft.rfft(tof[c] * win); B = np.fft.rfft(leg[c] * win)
            nmz = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum())
            Os.append(max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nmz for t in taus))
        print(f"  opt SNR: TOF={float(oT):.3f}  legacy={float(oL):.3f}  TOF/legacy={float(oT)/float(oL):.4f}", flush=True)
        print(f"  RMS(TOF)/RMS(legacy): X={rms_r[0]:.4f} Y={rms_r[1]:.4f} Z={rms_r[2]:.4f}", flush=True)
        print(f"  |O|(TOF,legacy): X={Os[0]:.4f} Y={Os[1]:.4f} Z={Os[2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
