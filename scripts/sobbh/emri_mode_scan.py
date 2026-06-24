"""Scan individual single modes (one at a time, include_minus_m=False) through the
package EMRITDIonFly vs legacy pyResponse. If TOF/legacy opt ratio is CONSTANT across
modes (odd l/m, non-zero n) -> a uniform response factor (inc=0/circular). If it VARIES
by mode -> the SWSH is applied differently than FEW (per-mode). |O| flags shape diffs.
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.sources.emri import EMRITDIonFly
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664; SRC = 1
DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
T_BUF = 1000.0; N_PTS = 4096; DELAY = 800.0
# (l, m, k, n): even/odd l, even/odd m, n = 0 / +-1 / +-2
MODES = [(2, 2, 0, 0), (3, 3, 0, 0), (3, 2, 0, 0), (4, 4, 0, 0),
         (2, 2, 0, 1), (2, 2, 0, -1), (3, 3, 0, 1), (3, 2, 0, -1),
         (4, 3, 0, 1), (5, 5, 0, 0)]


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
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    off = data_t0 - REF
    new_t = np.linspace(off + T_BUF, off + TOBS_S - T_BUF, N_PTS)
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                         "upsample": True, "fix_t": True, "new_t": new_t}, sum_kwargs={"pad_output": True})
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    fly = EMRITDIonFly(wg, orb, tdi_config, DT, TOBS_S, REF)
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector", force_backend="cpu")

    win = tukey(N_WIN, 0.1); tgrid = data_t0 + np.arange(N_WIN) * DT
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    f = np.fft.rfftfreq(N_WIN, d=DT)

    print(f"\n  {'mode':14s} {'opt_TOF':>8s} {'opt_leg':>8s} {'TOF/leg':>8s} {'|O|':>7s}", flush=True)
    for mode in MODES:
        try:
            out = fly(*inj, include_minus_mkn=False, mode_selection=[mode])
            tg = np.asarray(out.x); t_lo, t_hi = float(np.max(tg[:, 0])) + DELAY, float(np.min(tg[:, -1])) - DELAY
            inside = (tgrid > t_lo) & (tgrid < t_hi)
            _t = out.eval_tdi(tgrid[inside]); tof = np.zeros((3, N_WIN)); tof[:, inside] = np.real(np.sum(np.asarray(_t), axis=0))
            wl = ResponseWrapper(FixedModeGen(fg, [mode]), orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
                index_lambda=8, index_beta=7, flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"),
                tdi_chan="XYZ", order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
            leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
            leg = (np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1]))) if leg.shape[-1] < N_WIN else leg[:, :N_WIN])
            oT = float(ac.template_snr(TDSignal(tof, td_set).transform(fd_set, window=win))[0])
            oL = float(ac.template_snr(TDSignal(leg, td_set).transform(fd_set, window=win))[0])
            A = np.fft.rfft(tof[0] * win); B = np.fft.rfft(leg[0] * win)
            xc = np.fft.irfft(A * np.conj(B), n=N_WIN); k = int(np.argmax(np.abs(xc)))
            tau = (k if k <= N_WIN // 2 else k - N_WIN) * DT
            O = abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * f * tau))) / np.sqrt((np.abs(A)**2).sum()*(np.abs(B)**2).sum())
            print(f"  {str(mode):14s} {oT:8.4f} {oL:8.4f} {oT/oL:8.4f} {O:7.4f}", flush=True)
        except Exception as e:
            print(f"  {str(mode):14s}  SKIP: {type(e).__name__}: {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    main()
