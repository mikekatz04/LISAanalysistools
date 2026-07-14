"""Splice test: measure the acceptance ceiling of leaf-independence proposals.

For each branch/leaf, rebuild the run's residuals from the latest cold-chain
state, then score candidate leaf-parameter vectors against each walker's own
residual through the SAME compute_like path the moves use:

  kind "own"    : walker w's own params        -> Delta ll = 0 (bookkeeping check)
  kind "walker" : another walker v's params    -> ceiling of ANY marginal
                                                  independence proposal
  kind "lagK"   : walker w's own params K steps ago -> staleness cost
  kind "flow"   : draws from the live flow checkpoint -> reproduces the
                                                  observed lnpdiff offline

If "walker" splices already lose 10^2-10^3 nats, no per-leaf independence
proposal can work (conditional-vs-marginal gap / target ruggedness); if they
sit at O(1) while "flow" loses big, it is a flow-fit problem.
"""
import os, sys, time, json, pickle
TMP = "/home/asantini/.claude/jobs/f7a00a42/tmp"
sys.path.insert(0, TMP)

import numpy as np
import cupy as cp
cp.cuda.runtime.setDevice(7)  # spare GPU; run uses 5,6 + trainers on 4
import h5py
import torch
torch.set_num_threads(4)

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)

log("importing settings module (patched: gpus=[7], scratch head_dir)")
import splice_settings as S
from copy import deepcopy
from functools import partial
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sources.emri.waveform import EMRITDIWaveform
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.globalfit.moves import TDMBHSpecialMove, EMRISpecialMove
from eryn.moves.tempering import make_ladder
from eryn.flows import ZukoFlow
from eryn.flows.torch.flows import _strip_pickled_keys

log("building settings + data (L1 processing)...")
curr = S.get_global_fit_settings()
gi = curr.general_info
emri_info = curr.source_info["emri"]
mbh_info = curr.source_info["mbh"]
psd_info = curr.source_info["psd"]
nwalkers = gi.nwalkers
log(f"settings built. nwalkers={nwalkers}, gpus={gi.gpus}")

# ---------- state from backend copy ----------
fb = h5py.File(f"{TMP}/main_backend.h5", "r")
ll_all = fb["mcmc/log_like"][:]
filled = np.where(np.any(ll_all != 0.0, axis=(1, 2)))[0]
nstep = int(filled[-1])
branches = ("emri", "mbh", "psd")
class Shim: pass
state = Shim()
state.branches_coords = {b: fb[f"mcmc/chain/{b}"][nstep, 0:1] for b in branches}
state.branches_inds = {b: fb[f"mcmc/inds/{b}"][nstep, 0:1] for b in branches}
stored_logl = ll_all[nstep, 0].copy()
LAGS = (3, 6, 12)
mbh_hist = {k: fb["mcmc/chain/mbh"][nstep - k, 0] for k in LAGS}
fb.close()
log(f"state loaded from step {nstep} (cold chain)")

# ---------- ACS (setup_acs replica, cold temp) ----------
acs_tmp = []
for w in range(nwalkers):
    dra = deepcopy(gi.input_data_residual_array)
    pp = state.branches_coords["psd"][0, w, 0]
    pp = psd_info.transform.both_transforms(pp) if psd_info.transform is not None else pp
    sens = gi.sensitivity_backend(f"walker_{w}", pp, galfor_params=None)
    acs_tmp.append(AnalysisContainer(dra, deepcopy(sens), signal_gen=None))
acs = AnalysisContainerArray(acs_tmp, gpus=gi.gpus, complex_psd=True)
log("ACS built (24 walkers on cuda:7)")

# ---------- waveform generators + residual build ----------
emri_wave_gen = EMRITDIWaveform(**emri_info.initialize_kwargs)
S.subtract_initial_signal(acs, state, emri_wave_gen.get_signals_for_residuals, "emri", emri_info)
log("EMRI templates subtracted")
mbh_wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
S.subtract_initial_signal(acs, state, mbh_wave_gen.get_signals_for_residuals, "mbh", mbh_info)
log("MBH templates subtracted")

# ---------- validation vs stored log_like ----------
my_logl = np.array([float(acs.acs[w].likelihood().real) for w in range(nwalkers)])
diff = my_logl - stored_logl
log(f"validation: my_logl - stored_logl: median {np.median(diff):+.3f}, "
    f"spread (max-min) {np.ptp(diff):.3e}, |max| {np.abs(diff).max():.3e}")

# ---------- moves (for their exact compute_like path) ----------
if mbh_info.betas is None:
    mbh_info.betas = make_ladder(mbh_info.ndim, ntemps=gi.ntemps)
mbh_move = TDMBHSpecialMove(
    dcga=acs, waveform_gen=mbh_wave_gen, branch_name="mbh",
    coords_shape=(gi.ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim),
    waveform_gen_kwargs=mbh_info.waveform_kwargs.copy(), waveform_like_kwargs={},
    num_repeats=1, transform_fn=mbh_info.transform, priors=mbh_info.priors,
    inner_moves=mbh_info.inner_moves,
    betas_all=np.tile(mbh_info.betas, (mbh_info.nleaves_max, 1)),
    permute_every=100, pad_out_of_prior=True, run_async=True, run_threaded=True,
    randomize_split=True, batch_size_per_gpu=48,
)
if emri_info.betas is None:
    emri_info.betas = make_ladder(emri_info.ndim, ntemps=gi.ntemps)
emri_betas_all = np.tile(emri_info.betas, (emri_info.nleaves_max, 1))
emri_move = EMRISpecialMove(
    dcga=acs, waveform_gen=emri_wave_gen, branch_name="emri",
    coords_shape=(emri_betas_all.shape[1], nwalkers, emri_info.nleaves_max, emri_info.ndim),
    waveform_gen_method="get_signals_for_residuals",
    waveform_gen_kwargs=emri_info.waveform_kwargs.copy(),
    waveform_like_method="__call__",
    waveform_like_kwargs=emri_info.waveform_kwargs.copy(),
    num_repeats=1, transform_fn=emri_info.transform, priors=emri_info.priors,
    inner_moves=emri_info.inner_moves, betas_all=emri_betas_all,
    permute_every=25, pad_out_of_prior=True, run_async=True, run_threaded=True,
    randomize_split=True, batch_size_per_gpu=5,
)
log("moves built")

# ---------- flow checkpoints (CPU) ----------
def load_flow(path):
    with h5py.File(path, "r") as h:
        grp = h["flow"]
        cfg = _strip_pickled_keys(json.loads(grp.attrs["config"]))
        cfg["device"] = "cpu"
        cfg["data_transform"] = pickle.loads(bytes(grp["data_transform"][()]))
        cfg["conditioning"] = pickle.loads(bytes(grp["conditioning"][()]))
        fl = ZukoFlow(**cfg)
        fl.set_weights({k: torch.tensor(np.array(v)) for k, v in grp["weights"].items()})
    return fl
mbh_flow = load_flow(f"{TMP}/mbh_flow_latest.h5")
emri_flow = load_flow(f"{TMP}/emri_flow_latest.h5")
log("flow checkpoints loaded")

# ---------- splice machinery ----------
records = dict(branch=[], leaf=[], w=[], v=[], kind=[], ll=[])
def add_rec(branch, leaf, w, v, kind, ll):
    records["branch"].append(branch); records["leaf"].append(leaf)
    records["w"].append(w); records["v"].append(v)
    records["kind"].append(kind); records["ll"].append(float(ll))

def leaf_addback(info, gen, coords_leaf, sign):
    """sign=+1: put leaf back into residual (pre-scoring); -1: restore."""
    cin = np.asarray(info.transform.both_transforms(np.asarray(coords_leaf)))
    for w in range(nwalkers):
        sig = gen.get_signals_for_residuals(*cin[w], **info.waveform_kwargs)
        if sign > 0:
            acs.remove_signal_from_residual(sig, data_index=np.array([w]))
        else:
            acs.add_signal_to_residual(sig, data_index=np.array([w]))
    acs.free_gpu_memory()

def score(move, info, coords, widx, prior_key):
    """Mirror the move: out-of-prior candidates get -1e300, never a waveform call."""
    coords = np.asarray(coords)
    lp = info.priors[prior_key].logpdf(coords)
    ok = np.isfinite(lp)
    out = np.full(len(coords), -1e300)
    if ok.any():
        cin = np.asarray(info.transform.both_transforms(coords[ok]))
        out[ok] = move.compute_like(cin, data_index=np.asarray(widx)[ok].astype(np.int32))
    return out

# ---------- sky-mode ids (computed up front; needed for the final analysis) ----------
coords_mbh = state.branches_coords["mbh"][0]  # (24, 6, 11)
cat = gi.catalogue["MBHB"]
tlf = partial(S.to_lisa_frame, orbits=gi.orbits, t_ref=S.MOJITO_REFERENCE_TIME)
truths = np.array([S.mbh_catalogue_to_sampling_basis(cat[sid], to_lisa_frame=tlf)
                   for sid in sorted(cat.keys())])
def circd(a, b, per):
    d = np.abs(a - b) % per
    return np.minimum(d, per - d)
mode_id = np.zeros((nwalkers, mbh_info.nleaves_max), dtype=int)
for leaf in range(mbh_info.nleaves_max):
    imgs = []
    for k in range(4):
        for refl in (0, 1):
            lam = (truths[leaf, 8] + k * np.pi / 2) % (2 * np.pi)
            sb = -truths[leaf, 9] if refl else truths[leaf, 9]
            imgs.append((lam, sb))
    for w in range(nwalkers):
        d = [np.hypot(circd(coords_mbh[w, leaf, 8], lam, 2 * np.pi),
                      coords_mbh[w, leaf, 9] - sb) for lam, sb in imgs]
        mode_id[w, leaf] = int(np.argmin(d))
log("sky-mode ids computed")

def save_partial():
    np.savez(f"{TMP}/splice_results.npz",
             branch=np.array(records["branch"]), leaf=np.array(records["leaf"]),
             w=np.array(records["w"]), v=np.array(records["v"]),
             kind=np.array(records["kind"]), ll=np.array(records["ll"]),
             my_logl=my_logl, stored_logl=stored_logl, mode_id=mode_id,
             nstep=nstep)

# ---------- MBH splices ----------
rng = np.random.default_rng(20260713)
for leaf in range(mbh_info.nleaves_max):
    log(f"MBH leaf {leaf}: add-back")
    leaf_addback(mbh_info, mbh_wave_gen, coords_mbh[:, leaf], +1)
    cands, widx, kinds, vsrc = [], [], [], []
    fdraws, _ = mbh_flow.sample_and_log_prob(8 * nwalkers, context=leaf)
    for w in range(nwalkers):
        cands.append(coords_mbh[w, leaf]); widx.append(w); kinds.append("own"); vsrc.append(w)
        for v in range(nwalkers):
            if v == w: continue
            cands.append(coords_mbh[v, leaf]); widx.append(w); kinds.append("walker"); vsrc.append(v)
        for k in LAGS:
            cands.append(mbh_hist[k][w, leaf]); widx.append(w); kinds.append(f"lag{k}"); vsrc.append(w)
        for d in fdraws[8 * w:8 * (w + 1)]:
            cands.append(d); widx.append(w); kinds.append("flow"); vsrc.append(-1)
    ll = score(mbh_move, mbh_info, np.array(cands), np.array(widx), "mbh")
    for i in range(len(cands)):
        add_rec("mbh", leaf, widx[i], vsrc[i], kinds[i], ll[i])
    log(f"MBH leaf {leaf}: scored {len(cands)} candidates; restoring residual")
    leaf_addback(mbh_info, mbh_wave_gen, coords_mbh[:, leaf], -1)
    chk = float(acs.acs[0].likelihood().real)
    log(f"  restoration check walker0: {chk - my_logl[0]:+.6e}")
    save_partial()

# ---------- EMRI splices ----------
coords_emri = state.branches_coords["emri"][0]  # (24, 2, 12)
N_PAIR = 8
for leaf in range(emri_info.nleaves_max):
    log(f"EMRI leaf {leaf}: add-back")
    leaf_addback(emri_info, emri_wave_gen, coords_emri[:, leaf], +1)
    cands, widx, kinds, vsrc = [], [], [], []
    fdraws, _ = emri_flow.sample_and_log_prob(4 * nwalkers, context=leaf)
    for w in range(nwalkers):
        cands.append(coords_emri[w, leaf]); widx.append(w); kinds.append("own"); vsrc.append(w)
        for v in rng.choice([v for v in range(nwalkers) if v != w], N_PAIR, replace=False):
            cands.append(coords_emri[v, leaf]); widx.append(w); kinds.append("walker"); vsrc.append(int(v))
        for d in fdraws[4 * w:4 * (w + 1)]:
            cands.append(d); widx.append(w); kinds.append("flow"); vsrc.append(-1)
    ll = score(emri_move, emri_info, np.array(cands), np.array(widx), "emri")
    for i in range(len(cands)):
        add_rec("emri", leaf, widx[i], vsrc[i], kinds[i], ll[i])
    log(f"EMRI leaf {leaf}: scored {len(cands)} candidates; restoring residual")
    leaf_addback(emri_info, emri_wave_gen, coords_emri[:, leaf], -1)
    save_partial()

save_partial()
log("results saved to splice_results.npz")

# ---------- quick summary ----------
br = np.array(records["branch"]); kd = np.array(records["kind"])
lf = np.array(records["leaf"]); ww = np.array(records["w"]); vv = np.array(records["v"])
ll = np.array(records["ll"])
own_map = {(b, l, w_): val for b, l, w_, k_, val in zip(br, lf, ww, kd, ll) if k_ == "own"}
dll = np.array([ll[i] - own_map[(br[i], lf[i], ww[i])] for i in range(len(ll))])
print("\n=== Delta logl summary (candidate - own), medians [16th, 84th pct] ===", flush=True)
for b in ("mbh", "emri"):
    for kind in sorted(set(kd[br == b])):
        if kind == "own": continue
        m = (br == b) & (kd == kind)
        if b == "mbh" and kind == "walker":
            same = np.array([mode_id[vv[i], lf[i]] == mode_id[ww[i], lf[i]] if vv[i] >= 0 else False
                             for i in np.where(m)[0]])
            for tag, sel in (("same-mode", same), ("cross-mode", ~same)):
                x = dll[np.where(m)[0][sel]]
                if len(x):
                    print(f"  {b:5s} {kind}/{tag:10s}: n={len(x):4d} median {np.median(x):10.1f} "
                          f"[{np.percentile(x,16):10.1f}, {np.percentile(x,84):10.1f}] "
                          f"frac>-3: {np.mean(x > -3):.3f}", flush=True)
        else:
            x = dll[m]
            print(f"  {b:5s} {kind:18s}: n={len(x):4d} median {np.median(x):10.1f} "
                  f"[{np.percentile(x,16):10.1f}, {np.percentile(x,84):10.1f}] "
                  f"frac>-3: {np.mean(x > -3):.3f}", flush=True)
x = dll[kd == "own"]
print(f"  own-consistency: |max| {np.abs(x).max():.3e} (should be ~0)", flush=True)
log("DONE")
