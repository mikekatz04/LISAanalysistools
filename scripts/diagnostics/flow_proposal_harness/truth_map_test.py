"""Cleanest possible test of the sky-mode symmetry: apply the group maps to the
TRUE injection parameters (which sit at the mode peak by construction) and score
against a walker's residual. An exact symmetry => Delta logl ~ 0 on all 8."""
import os, sys, time
TMP="/home/asantini/.claude/jobs/f7a00a42/tmp"; sys.path.insert(0, TMP)
import numpy as np, cupy as cp
cp.cuda.runtime.setDevice(7)
import h5py, torch, glob
from functools import partial
torch.set_num_threads(4)
t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
import splice_settings as S
from copy import deepcopy
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sources.emri.waveform import EMRITDIWaveform
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.globalfit.moves import TDMBHSpecialMove
from eryn.moves.tempering import make_ladder
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis, MOJITO_REFERENCE_TIME
from lisatools.sources.utils import to_lisa_frame
from fold_utils import orbit

curr=S.get_global_fit_settings(); gi=curr.general_info
emri_info=curr.source_info["emri"]; mbh_info=curr.source_info["mbh"]; psd_info=curr.source_info["psd"]
nw=gi.nwalkers
fb=h5py.File(f"{TMP}/main_backend.h5","r"); lla=fb["mcmc/log_like"][:]
ns=int(np.where(np.any(lla!=0.,axis=(1,2)))[0][-1])
class Sh: pass
st=Sh(); st.branches_coords={b: fb[f"mcmc/chain/{b}"][ns,0:1] for b in ("emri","mbh","psd")}
st.branches_inds={b: fb[f"mcmc/inds/{b}"][ns,0:1] for b in ("emri","mbh","psd")}
fb.close()
acs_tmp=[]
for w in range(nw):
    dra=deepcopy(gi.input_data_residual_array)
    pp=st.branches_coords["psd"][0,w,0]
    pp=psd_info.transform.both_transforms(pp) if psd_info.transform is not None else pp
    acs_tmp.append(AnalysisContainer(dra, deepcopy(gi.sensitivity_backend(f"walker_{w}",pp,galfor_params=None)), signal_gen=None))
acs=AnalysisContainerArray(acs_tmp,gpus=gi.gpus,complex_psd=True)
ewg=EMRITDIWaveform(**emri_info.initialize_kwargs)
S.subtract_initial_signal(acs,st,ewg.get_signals_for_residuals,"emri",emri_info)
mwg=PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
S.subtract_initial_signal(acs,st,mwg.get_signals_for_residuals,"mbh",mbh_info)
if mbh_info.betas is None: mbh_info.betas=make_ladder(mbh_info.ndim,ntemps=gi.ntemps)
mv=TDMBHSpecialMove(dcga=acs,waveform_gen=mwg,branch_name="mbh",
    coords_shape=(gi.ntemps,nw,mbh_info.nleaves_max,mbh_info.ndim),
    waveform_gen_kwargs=mbh_info.waveform_kwargs.copy(),waveform_like_kwargs={},
    num_repeats=1,transform_fn=mbh_info.transform,priors=mbh_info.priors,
    inner_moves=mbh_info.inner_moves,betas_all=np.tile(mbh_info.betas,(mbh_info.nleaves_max,1)),
    permute_every=100,pad_out_of_prior=True,run_async=True,run_threaded=True,
    randomize_split=True,batch_size_per_gpu=48)
def score(coords,widx):
    coords=np.asarray(coords,dtype=np.float64); lp=mbh_info.priors["mbh"].logpdf(coords); ok=np.isfinite(lp)
    out=np.full(len(coords),-1e300)
    if ok.any():
        cin=np.asarray(mbh_info.transform.both_transforms(coords[ok]))
        out[ok]=mv.compute_like(cin,data_index=np.asarray(widx)[ok].astype(np.int32))
    return out
def addback(coords_leaf,sign):
    cin=np.asarray(mbh_info.transform.both_transforms(np.asarray(coords_leaf)))
    for w in range(nw):
        sig=mwg.get_signals_for_residuals(*cin[w],**mbh_info.waveform_kwargs)
        (acs.remove_signal_from_residual if sign>0 else acs.add_signal_to_residual)(sig,data_index=np.array([w]))
    acs.free_gpu_memory()
# truths
orb=gi.orbits
tlf=partial(to_lisa_frame,orbits=orb,t_ref=MOJITO_REFERENCE_TIME)
cat=gi.catalogue["MBHB"]; IDS=sorted(cat.keys())
truths=np.array([mbh_catalogue_to_sampling_basis(cat[s],to_lisa_frame=tlf) for s in IDS])
C=st.branches_coords["mbh"][0]
print("\nDelta logl of the TRUE params under each sky-mode map (vs unmapped truth),")
print("scored against walker 0's residual. Exact symmetry => ~0 everywhere.")
print(f"{'leaf':4s} " + " ".join(f"{'g'+str(i):>10s}" for i in range(8)))
for leaf in range(6):
    addback(C[:,leaf],+1)
    T=np.tile(truths[leaf],(nw,1))
    base=score(T,np.arange(nw))[0]
    row=[]
    for g in range(8):
        im=orbit(T)[g]
        row.append(score(im,np.arange(nw))[0]-base)
    addback(C[:,leaf],-1)
    print(f"{leaf:<4d} " + " ".join(f"{v:10.1f}" for v in row))
log("DONE")
