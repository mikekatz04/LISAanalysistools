"""VGB fidelity vs mojito: verification binaries from the VGB stream, compared
against the GBTDIonTheFly time-domain template.

The VGB L1 stream contains ONLY the 55 verification binaries (no galactic
confusion), and the top-frequency VGBs are isolated by >100 uHz from their
nearest neighbour -- so a tight band-pass gives a genuinely single-source GB
data segment. This removes the contamination ambiguity of the whole-galaxy
GB stream used by ``gb_mojito_match.py``: for e.g. HM Cnc at 6.22 mHz with
fdot = 7.5e-16, the fddot / amplitude-drift terms are irrelevant over 90 d,
so the measured mismatch here is purely response + sky-frame + phase
convention fidelity.

Same conventions as gb_mojito_match.py (the validated injection convention):
params consumed AT the catalogue epoch REF, phi0 = +TrueAnomaly, f0 =
GW22FrequencySSBFrame, RA/Dec direct with icrs-frame orbits, kernel
t_ref = REF.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
MEM_CAP_GB = float(os.environ.get("GB_MEM_CAP_GB", "8.2")); _IS_MAC = sys.platform == "darwin"
def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6
def _wd():
    while True:
        if rss_gb() > MEM_CAP_GB: os._exit(42)
        time.sleep(0.3)
def mark(m): print(f"[RSS {rss_gb():5.2f} GB] {m}", flush=True)

import gbgpu  # noqa: F401 -- registers the gbgpu_<flavor> backend GBTDIonTheFly needs
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
VGB_L1 = os.path.join(PATH, "data", "VGB", "L1")
BACKEND = "cpu"; DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3; SENS = "scirdv1"
N_DAYS = float(os.environ.get("GB_DAYS", "90")); N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
TOPN = int(os.environ.get("GB_TOPN", "3"))
BAND_UHZ = float(os.environ.get("GB_BAND_UHZ", "5.0"))  # +/- microHertz band half-width
TUKEY_ALPHA = float(os.environ.get("GB_TUKEY_ALPHA", "0.05"))
DATA_CACHE = f"/tmp/vgb_mojito_data_{int(N_DAYS)}d.npz"
# catalogue field -> UCB param (amp,f0,fdot,fddot=0,phi0,inc,psi,lam=RA,beta=Dec)
FIELDS = ["Amplitude", "GW22FrequencySSBFrame", "GW22FrequencyDerivativeSourceFrame",
          "TrueAnomaly", "InclinationAngle", "PolarisationAngle", "RightAscension", "Declination"]


def banner(s): print("\n" + "=" * 84 + f"\n {s}\n" + "=" * 84, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_vgb_cached(n_keep=55):
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE)
        print(f"[cache] {DATA_CACHE}", flush=True)
        return np.asarray(z["data_td"]), float(z["data_t0"]), np.asarray(z["top_params"]), np.asarray(z["top_freqs"])
    print(f"[cache] MISS -> VGB stream read ({N_DAYS:.0f} d window)...", flush=True)
    loader = L1ProcessingStep(L1_folder=PATH, source_types=["vgb"], source_ids=None,
                              orbits_class=L1Orbits, orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
                              verbose=True)
    times = np.asarray(loader.times); data_full = np.asarray(loader.data).T if np.asarray(loader.data).shape[0] != 3 else np.asarray(loader.data)
    dt_native = float(loader.dt); data_t0 = float(times[0]); deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN * deci : deci][:, :N_WIN].copy()
    cat0 = loader.catalogue["VGB"][0]
    f = np.asarray(cat0["GW22FrequencySSBFrame"], float)
    order = np.argsort(f)[::-1][:n_keep]
    top_params = np.array([[float(np.asarray(cat0[k])[i]) for k in FIELDS] for i in order])
    top_freqs = f[order]
    del data_full, loader, cat0; gc.collect()
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, top_params=top_params, top_freqs=top_freqs)
    print(f"[cache] wrote {DATA_CACHE}", flush=True)
    return data_td, data_t0, top_params, top_freqs


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"VGB fidelity vs mojito: top {TOPN} hi-f, +/-{BAND_UHZ} uHz band-pass ({N_DAYS:.0f} d)")
    D, data_t0, top_params, top_freqs = load_vgb_cached(); D = D[:NCH]; mark("data loaded")
    print(f"  N_WIN={N_WIN} ({N_WIN*DT/86400:.0f} d)  data_t0-REF={data_t0-REF:.2f}s", flush=True)
    fs = np.sort(top_freqs)[::-1]
    print(f"  highest f's (mHz): {', '.join(f'{x*1e3:.4f}' for x in fs[:TOPN+2])}", flush=True)

    orb = L1Orbits(find_file(VGB_L1, "VGB", 0), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True); mark("orbit ready")

    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)
    grid = np.arange(N_WIN) * DT + data_t0
    t_nodes = np.linspace(grid[0], grid[-1], 16384)
    gb_gen = GBTDIonTheFly(t_nodes, TOBS, REF, 1.0 / DT, 1,
                           tdi_config=tdi_config, orbits=orb, tdi_chan="XYZ", force_backend=BACKEND)
    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    delta = BAND_UHZ * 1e-6

    fig, axes = plt.subplots(TOPN, 1, figsize=(10, 3 * TOPN)); axes = np.atleast_1d(axes)
    print(f"\n  {'rank':>4} {'f0(mHz)':>10} {'amp':>10} {'1-Re(O)':>11} {'1-|O|':>11} {'tau*(s)':>9} {'mm@tau*':>11} {'SNRdet':>8}", flush=True)
    for rank in range(TOPN):
        A0, f0, fdot, phi0, inc, psi, ra, dec = top_params[rank]
        params9 = np.array([A0, f0, fdot, 0.0, phi0, inc, psi, ra, dec]).reshape(9, 1)
        out = gb_gen(*params9, convert_to_ra_dec=False, return_spline=True)
        B = np.asarray(out.eval_tdi(grid)); B = (B[0] if B.ndim == 3 else B)[:NCH]
        flo, fhi = f0 - delta, f0 + delta
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=flo, max_freq=fhi, force_backend=BACKEND)
        f = np.asarray(fd.f_arr)
        d = TDSignal(D, td).transform(fd, window=win); t = TDSignal(B, td).transform(fd, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))
        O = ac.template_inner_product(t, normalize=True, complex=True)
        opt, det = ac.template_snr(t); t0a = np.asarray(t.arr).copy()
        def mmt(tau):
            t.arr[:] = t0a * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(t, normalize=True, complex=True))
        per = 1.0 / f0; bc = min(((mmt(x), x) for x in np.linspace(-1.5 * per, 1.5 * per, 121)))
        bf = min(((mmt(x), x) for x in bc[1] + np.linspace(-1.0, 1.0, 121))); t.arr[:] = t0a
        print(f"  {rank:>4} {f0*1e3:>10.5f} {A0:>10.3e} {1-O.real:>+11.3e} {1-abs(O):>11.3e} "
              f"{bf[1]:>+9.3f} {bf[0]:>11.3e} {float(det):>+8.2f}", flush=True)
        # Machine-parseable line for scripts/validation/run_mojito_null_checks.sh.
        print(f"[RESULT] class=VGB rank={rank} f0_mhz={f0*1e3:.5f} "
              f"overlap={abs(O):.8f} mismatch={1-abs(O):.3e} "
              f"mm_taustar={bf[0]:.3e} det_snr={float(det):.2f}", flush=True)
        # plot band
        ff = np.fft.rfftfreq(N_WIN, DT); sel = (ff >= flo) & (ff <= fhi)
        FD = np.fft.rfft(D[0] * win) * DT; FB = np.fft.rfft(B[0] * win) * DT
        axes[rank].semilogy((ff[sel] - f0) * 1e6, np.abs(FD[sel]), label="mojito data", lw=1.0)
        axes[rank].semilogy((ff[sel] - f0) * 1e6, np.abs(FB[sel]), ":", label="GBTDIonTheFly", lw=1.2)
        axes[rank].set_title(f"VGB rank {rank}: f0={f0*1e3:.4f} mHz, 1-|O|={1-abs(O):.2e}")
        axes[rank].set_xlabel("f - f0 [uHz]"); axes[rank].legend(fontsize=8)
    fig.tight_layout(); out_png = "/tmp/vgb_mojito_match.png"; fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"\nDONE.  plot -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
