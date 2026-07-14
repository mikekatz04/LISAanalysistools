"""Phase 0/1 driver, part 2: score candidate flow checkpoints against the
splice-test ceiling, through the moves' own compute_like path.

For each leaf: add the leaf's template back into each walker's residual, score
N draws per walker from every candidate flow (prior-filtered, like the move),
plus the walker's own params (baseline), then restore. Reports Delta logl
distributions and the implied independence-MH acceptance E[min(1, e^dll)].
"""
import os, sys, time, json, pickle, glob
TMP = "/home/asantini/.claude/jobs/f7a00a42/tmp"
sys.path.insert(0, TMP)

import numpy as np
import cupy as cp
cp.cuda.runtime.setDevice(7)
import h5py
import torch
torch.set_num_threads(4)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

import splice_settings as S
from copy import deepcopy
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

fb = h5py.File(f"{TMP}/main_backend.h5", "r")
ll_all = fb["mcmc/log_like"][:]
nstep = int(np.where(np.any(ll_all != 0.0, axis=(1, 2)))[0][-1])
branches = ("emri", "mbh", "psd")
class Shim: pass
state = Shim()
state.branches_coords = {b: fb[f"mcmc/chain/{b}"][nstep, 0:1] for b in branches}
state.branches_inds = {b: fb[f"mcmc/inds/{b}"][nstep, 0:1] for b in branches}
stored_logl = ll_all[nstep, 0].copy()
fb.close()
log(f"state from step {nstep} (cold)")

acs_tmp = []
for w in range(nwalkers):
    dra = deepcopy(gi.input_data_residual_array)
    pp = state.branches_coords["psd"][0, w, 0]
    pp = psd_info.transform.both_transforms(pp) if psd_info.transform is not None else pp
    sens = gi.sensitivity_backend(f"walker_{w}", pp, galfor_params=None)
    acs_tmp.append(AnalysisContainer(dra, deepcopy(sens), signal_gen=None))
acs = AnalysisContainerArray(acs_tmp, gpus=gi.gpus, complex_psd=True)
log("ACS built")

emri_wave_gen = EMRITDIWaveform(**emri_info.initialize_kwargs)
S.subtract_initial_signal(acs, state, emri_wave_gen.get_signals_for_residuals, "emri", emri_info)
mbh_wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
S.subtract_initial_signal(acs, state, mbh_wave_gen.get_signals_for_residuals, "mbh", mbh_info)
my_logl = np.array([float(acs.acs[w].likelihood().real) for w in range(nwalkers)])
d = my_logl - stored_logl
log(f"validation: median {np.median(d):+.3f}, |max| {np.abs(d).max():.3e}")

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

def load_flow_cpu(path):
    with h5py.File(path, "r") as h:
        grp = h["flow"]
        cfg = _strip_pickled_keys(json.loads(grp.attrs["config"]))
        cfg["device"] = "cpu"
        cfg["data_transform"] = pickle.loads(bytes(grp["data_transform"][()]))
        cfg["conditioning"] = pickle.loads(bytes(grp["conditioning"][()]))
        fl = ZukoFlow(**cfg)
        fl.set_weights({k: torch.tensor(np.array(v)) for k, v in grp["weights"].items()})
    return fl

ART = "/data/asantini/globalfit/erebor_org_setup/mojito_runs/test_flow_joint_sources_stft_artifacts"
cands = {"mbh": {}, "emri": {}}
for b in ("mbh", "emri"):
    for p in sorted(glob.glob(f"{TMP}/offline_flows/{b}_*.h5")):
        tag = os.path.basename(p)[len(b) + 1:-3]
        cands[b][tag] = load_flow_cpu(p)
    live = f"{TMP}/{b}_flow_live_now.h5"
    os.system(f"cp {ART}/{b}_flow/{b}_flow_latest.h5 {live}")
    cands[b]["live"] = load_flow_cpu(live)
log(f"candidates: mbh={list(cands['mbh'])}, emri={list(cands['emri'])}")

def score(move, info, coords, widx, key):
    coords = np.asarray(coords)
    lp = info.priors[key].logpdf(coords)
    ok = np.isfinite(lp)
    out = np.full(len(coords), -1e300)
    if ok.any():
        cin = np.asarray(info.transform.both_transforms(coords[ok]))
        out[ok] = move.compute_like(cin, data_index=np.asarray(widx)[ok].astype(np.int32))
    return out

def leaf_addback(info, gen, coords_leaf, sign):
    cin = np.asarray(info.transform.both_transforms(np.asarray(coords_leaf)))
    for w in range(nwalkers):
        sig = gen.get_signals_for_residuals(*cin[w], **info.waveform_kwargs)
        if sign > 0:
            acs.remove_signal_from_residual(sig, data_index=np.array([w]))
        else:
            acs.add_signal_to_residual(sig, data_index=np.array([w]))
    acs.free_gpu_memory()

records = dict(branch=[], leaf=[], cand=[], w=[], ll=[], own=[], logq=[], logq_own=[])
NDRAW = {"mbh": 8, "emri": 4}
for branch, info, move, gen, nleaves in (
    ("mbh", mbh_info, mbh_move, mbh_wave_gen, mbh_info.nleaves_max),
    ("emri", emri_info, emri_move, emri_wave_gen, emri_info.nleaves_max),
):
    coords_all = state.branches_coords[branch][0]
    nd = NDRAW[branch]
    for leaf in range(nleaves):
        log(f"{branch} leaf {leaf}: add-back + scoring {len(cands[branch])} candidates")
        leaf_addback(info, gen, coords_all[:, leaf], +1)
        own_ll = score(move, info, coords_all[:, leaf], np.arange(nwalkers), branch)
        for tag, fl in cands[branch].items():
            draws, logq = fl.sample_and_log_prob(nd * nwalkers, context=leaf)
            logq_own = fl.log_prob(coords_all[:, leaf], context=leaf)  # q at current walkers
            widx = np.repeat(np.arange(nwalkers), nd)
            ll = score(move, info, draws, widx, branch)
            for i in range(len(ll)):
                records["branch"].append(branch); records["leaf"].append(leaf)
                records["cand"].append(tag); records["w"].append(int(widx[i]))
                records["ll"].append(float(ll[i])); records["own"].append(float(own_ll[widx[i]]))
                records["logq"].append(float(logq[i])); records["logq_own"].append(float(logq_own[widx[i]]))
        leaf_addback(info, gen, coords_all[:, leaf], -1)
        np.savez(f"{TMP}/score_results.npz", **{k: np.array(v) for k, v in records.items()})

br = np.array(records["branch"]); cd = np.array(records["cand"])
lf = np.array(records["leaf"])
dll = np.array(records["ll"]) - np.array(records["own"])
# exact independence-MH statistic: lnpdiff = [ll(y)-logq(y)] - [ll(x)-logq(x)]
lnp = dll - (np.array(records["logq"]) - np.array(records["logq_own"]))
oop = np.array(records["ll"]) <= -1e299
print("\n=== exact-MH scores: lnpdiff = dll - (logq_new - logq_old); acc = E[min(1,e^lnpdiff)] ===", flush=True)
for b in ("mbh", "emri"):
    print(f"--- {b} ---", flush=True)
    for tag in sorted(set(cd[br == b])):
        m = (br == b) & (cd == tag)
        acc = np.mean(np.minimum(1.0, np.exp(np.clip(lnp[m], -700, 0))) * ~oop[m])
        acc_dll = np.mean(np.minimum(1.0, np.exp(np.clip(dll[m], -700, 0))) * ~oop[m])
        xin = lnp[m & ~oop]
        med = np.median(xin) if len(xin) else float("nan")
        per_leaf = " ".join(f"L{l}:{np.mean(np.minimum(1.0, np.exp(np.clip(lnp[m & (lf == l)], -700, 0))) * ~oop[m & (lf == l)]):.3f}"
                            for l in sorted(set(lf[m])))
        print(f"  {tag:16s}: MH acc {acc:.3f} (q-uncorr {acc_dll:.3f}) med lnpdiff {med:9.1f} "
              f"oop {oop[m].mean():.2f} | per-leaf acc {per_leaf}", flush=True)
log("DONE")
