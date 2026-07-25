"""A-vs-B (pyResponse vs TDI-on-the-fly) TDI mismatch resolved in FREQUENCY.

Generates A (pyResponseTDI) and B (on-the-fly) at edge-on (where A-vs-B ~ 11%,
driven by higher modes) and computes the noise-weighted mismatch in frequency
bands + cumulative, to localize where the per-mode response discrepancy lives.
Marks the (2,2)/(3,3)/(4,4) merger frequencies.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MEM_CAP_GB = 6.5; _IS_MAC = sys.platform == "darwin"
def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6
def _wd():
    while True:
        if rss_gb() > MEM_CAP_GB: os._exit(42)
        time.sleep(0.3)

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file, L1DataLoader
from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import TDTDIonTheFly
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal, place_td_signal_on_grid
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.utility import get_array_module
import phentax.waveform as pw


def to_host(x):
    """Backend array (cupy/numpy) -> numpy host array. On the cuda backend the
    waveform/response arrays are cupy; A and B are consumed as numpy (np.fft,
    TDSignal input), so convert at the gen_A/gen_B boundary. No-op on CPU."""
    return x.get() if hasattr(x, "get") else np.asarray(x)

REF = 97729089.327664
# Mojito brick location. Same knob as the stock global fits (fit.py
# GeneralSettings.mojito_data_path): env MOJITO_DATA_PATH, home-relative
# default so it resolves on any machine (laptop /Users/<u>, cluster /home/<u>).
PATH = os.environ.get(
    "MOJITO_DATA_PATH",
    os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
)
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")


def _first_available_mbhb_id():
    """Lowest MBHB source id whose L1 data file is present in MBHB_L1. Not every
    source's L1 file is downloaded on every machine (e.g. id 0 may be absent on
    the cluster), so when MBHB_ID is unset we scan ids in order and take the
    first that actually has a file. Filenames look like ``MBHB_..._source{id}_``
    (same convention find_file matches)."""
    ids = set()
    for f in os.listdir(MBHB_L1):
        if not f.startswith("MBHB_") or "source" not in f:
            continue
        try:
            ids.add(int(f.split("source", 1)[1].split("_", 1)[0]))
        except (IndexError, ValueError):
            continue
    if not ids:
        raise FileNotFoundError(f"No MBHB L1 source files found in {MBHB_L1}")
    return min(ids)


# Explicit MBHB_ID wins; otherwise auto-pick the first id present on this box.
MBHB_ID = int(os.environ["MBHB_ID"]) if "MBHB_ID" in os.environ else _first_available_mbhb_id()
print(f"[mbhb] using source id {MBHB_ID}  (MBHB_L1={MBHB_L1})", flush=True)
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
# The A-vs-B mismatch is backend-agnostic: generate the waveforms on BACKEND
# (GPU exercises the spline TDI-on-the-fly kernel), bring A/B to host, and run
# the FD/inner-product metric on CPU. Keeps the metric pipeline off the
# host/device-consistency edges in TDSignal/transform (numpy window etc.).
METRIC_BACKEND = "cpu"
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0; DTMIN = 0.1
WIN_DAYS = 6.0; DUR_DAYS = 5.0
INC = float(os.environ.get("MBH_INC", str(np.pi / 2)))   # edge-on default
CG_SCALE = float(os.environ.get("MBH_CG_SCALE", "12.0"))  # coarse-grain density (pts/cycle)
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def load_cat():
    """MBH catalogue params for MBHB_ID. Only the catalogue is needed here (A and
    B are generated fresh), so read it straight from the mojito brick when the
    /tmp cache is absent -- no full L1 data-stream load. Mirrors the loader's
    load_single_binary semantics (group[key][id], numeric fields only)."""
    if os.path.exists(DATA_CACHE):
        return np.load(DATA_CACHE, allow_pickle=True)["cat"].item()
    import h5py
    ld = L1DataLoader(L1_folder=PATH, source_types=["MBHB"],
                      source_ids={"MBHB": [MBHB_ID]}, verbose=False)
    catf = os.path.join(ld.catalogues_folder, ld.catalogues_map["MBHB"])
    print(f"[cat] reading {catf} (id={MBHB_ID})", flush=True)
    with h5py.File(catf, "r") as f:
        b = f["Binaries"]
        return {k: float(np.asarray(b[k][MBHB_ID])) for k in b.keys()
                if np.asarray(b[k][MBHB_ID]).dtype.kind in "fi"}


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def build_orbit(window_t0, TOBS):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del ltt_t; gc.collect()
    orb.configure(linear_interp_setup=True, dt=POS_DT)
    return orb


def gen_A(wf, window_t0, N_WIN, dur_s, orbit):
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orbit, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    times, ch = gen.compute_tdi_channels(*wf)
    xp = get_array_module(ch)   # cupy on the GPU backend; keep the placement on-backend
    arr = to_host(place_td_signal_on_grid(xp.atleast_2d(ch)[:NCH], grid, times=times).arr)
    del gen; gc.collect(); return arr


def gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s):
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    nt, nm, sca, scp = wave_gen.compute_strain_components_amp_phase(
        m1, m2, s1z, s2z, dist, phi_ref, inc, psi, delta_t=DTMIN, t_min=-dur_s, t_ref=0.0)
    amp_m = np.asarray(sca) / 2.0; ph_m = np.pi - np.asarray(scp)
    nmodes = wave_gen.num_modes
    _nt = np.asarray(nt[nm] + t_plunge + REF)
    amp = np.asarray(amp_m[0][:, nm[0]]); phase = np.asarray(ph_m[0][:, nm[0]])
    # Zero-amp / held-phase tail past the last node. The on-the-fly TDI reads
    # amp/phase at RETARDED times slightly past the final node (sky-dependent),
    # running off the end of the spline -- silent garbage on CPU but an ILLEGAL
    # MEMORY ACCESS on GPU. Mirror MBHTDIonFly's n_tail padding (bbhx
    # mbhtdionfly.py). The eval window is unchanged; only the spline domain
    # (t_input/amp/phase) is extended, so the mismatch is unaffected.
    n_tail = 120; dt_tail = 10.0
    tail_t = _nt[-1] + dt_tail * np.arange(1, n_tail + 1)
    _nt = np.concatenate([_nt, tail_t])
    amp = np.concatenate([amp, np.zeros((amp.shape[0], n_tail))], axis=1)
    phase = np.concatenate([phase, np.repeat(phase[:, -1:], n_tail, axis=1)], axis=1)
    nta = np.repeat(_nt[None, :], nmodes, axis=0)
    tb = int(1000 / DT); eval_t = nta[:, tb:-(tb + n_tail)]
    g = TDTDIonTheFly(eval_t, amp, phase, sampling_frequency=1.0 / DT, num_sub=nmodes,
                      t_input=nta, tdi_config=tdi_config, orbits=orbit, force_backend=BACKEND)
    out = g(np.full(nmodes, 0.0), np.full(nmodes, psi), np.full(nmodes, ra), np.full(nmodes, dec),
            return_spline=True)
    grid_t = np.arange(N_WIN) * DT + window_t0
    ntdi = np.zeros((out.t_arr.shape[0], 3, N_WIN))
    keep = (grid_t >= out.t_arr.min().item()) & (grid_t <= out.t_arr.max().item())
    xp = get_array_module(out.t_arr)   # cupy on the GPU backend
    ntdi[:, :, keep] = to_host(out.eval_tdi(xp.asarray(grid_t[keep])))
    arr = ntdi.sum(axis=0)[:NCH]
    del g, out; gc.collect(); return arr


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"A-vs-B TDI mismatch vs FREQUENCY (id={MBHB_ID}, inc={INC:.4f})")
    cat = load_cat()
    wf = np.asarray(MBH_TRANSFORM.both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    wf[6] = INC
    t_plunge = wf[10]; abs_merger = REF + t_plunge
    N_WIN = int(round(WIN_DAYS * 86400 / DT)); window_t0 = abs_merger - 0.72 * N_WIN * DT
    TOBS = N_WIN * DT; dur_s = DUR_DAYS * 86400.0
    # 22-mode merger (ISCO-ish) frequency, and 33/44 scaled
    M = wf[0] + wf[1]; f22 = 0.018 / (M / 1e6); fmodes = {"f22": f22, "f33": 1.5 * f22, "f44": 2.0 * f22}
    print(f"  M={M:.3e} Msun  approx merger f22~{f22:.2e} Hz", flush=True)

    orbit = build_orbit(window_t0, TOBS)
    A = gen_A(wf, window_t0, N_WIN, dur_s, orbit)
    print(f"  coarse_graining_scale_factor = {CG_SCALE}", flush=True)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL,
                               coarse_graining_scale_factor=CG_SCALE)
    B = gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s)
    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=METRIC_BACKEND)

    def mm_band(lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=METRIC_BACKEND)
        a = AnalysisContainer(TDSignal(A, td_set).transform(fd, window=win),
                              XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        t = TDSignal(B, td_set).transform(fd, window=win)
        return 1 - abs(a.template_inner_product(t, normalize=True, complex=True))

    banner("mismatch in frequency BANDS (noise-weighted, phase-max)")
    edges = np.array([5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1.2e-2, F_MAX])
    print(f"  full [5e-4, {F_MAX}] : {mm_band(5e-4, F_MAX):.4e}", flush=True)
    print(f"  {'band [Hz]':>22}   {'mm':>12}", flush=True)
    for i in range(len(edges) - 1):
        print(f"  [{edges[i]:.1e}, {edges[i+1]:.1e}]   {mm_band(edges[i], edges[i+1]):12.4e}", flush=True)
    banner("cumulative mismatch from f_lo -> F_MAX")
    for flo in [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3]:
        print(f"  [{flo:.1e}, {F_MAX:.1e}] : {mm_band(flo, F_MAX):.4e}", flush=True)

    # plot FD spectra + |A-B|
    f = np.fft.rfftfreq(N_WIN, DT); w = win
    FAx = np.fft.rfft(A[0] * w) * DT; FBx = np.fft.rfft(B[0] * w) * DT
    sel = (f >= 5e-4) & (f <= F_MAX)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.loglog(f[sel], np.abs(FAx[sel]), label="A pyResponse |X|", lw=1.0)
    ax.loglog(f[sel], np.abs(FBx[sel]), "--", label="B on-the-fly |X|", lw=1.0, alpha=.8)
    ax.loglog(f[sel], np.abs((FAx - FBx)[sel]), ":", color="crimson", label="|A - B|", lw=1.2)
    for nm, fv in fmodes.items():
        ax.axvline(fv, color="k", ls=":", alpha=.4); ax.text(fv, ax.get_ylim()[1], nm, fontsize=8)
    ax.set_title(f"MBH TDI A vs B (inc={INC:.3f}): FD |X| + residual"); ax.set_xlabel("f [Hz]"); ax.legend()
    out = f"/tmp/mbh_mm_vs_freq_id{MBHB_ID}.png"; fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\nDONE.  plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
