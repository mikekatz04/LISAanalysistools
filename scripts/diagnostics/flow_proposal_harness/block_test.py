"""Does a BLOCK proposal on the non-sky parameters work?

Tests two variants against the run's own likelihood, using a plain GAUSSIAN
fitted per (leaf, mode) -- no flow needed. If the conditional variant works,
the design is sound and a flow can only do better; if it fails, the blocks are
inseparable and the idea is dead.

  marginal   : safe' ~ N(mu_safe, Sig_safe)          , sky kept
  conditional: safe' ~ N(mu_safe|sky_w, Sig_safe|sky), sky kept
"""
import os, sys, time
TMP="/home/asantini/.claude/jobs/f7a00a42/tmp"; sys.path.insert(0, TMP)
import numpy as np, cupy as cp
cp.cuda.runtime.setDevice(7)
import h5py, torch
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

SAFE=[0,1,2,3,4,5,10]; SKY=[6,7,8,9]
curr=S.get_global_fit_settings(); gi=curr.general_info
emri_info=curr.source_info["emri"]; mbh_info=curr.source_info["mbh"]; psd_info=curr.source_info["psd"]
nw=gi.nwalkers
fb=h5py.File(f"{TMP}/main_backend.h5","r"); lla=fb["mcmc/log_like"][:]
ns=int(np.where(np.any(lla!=0.,axis=(1,2)))[0][-1])
hist=fb["mcmc/chain/mbh"][ns-168:ns+1,0]      # (169,24,6,11) for the covariances
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
def addback(c,sign):
    cin=np.asarray(mbh_info.transform.both_transforms(np.asarray(c)))
    for w in range(nw):
        sig=mwg.get_signals_for_residuals(*cin[w],**mbh_info.waveform_kwargs)
        (acs.remove_signal_from_residual if sign>0 else acs.add_signal_to_residual)(sig,data_index=np.array([w]))
    acs.free_gpu_memory()
C=st.branches_coords["mbh"][0]
rng=np.random.default_rng(0)
PERD={5:2*np.pi, 7:np.pi, 8:2*np.pi}
def unwrap_to(R, ref):
    """Unwrap periodic dims of R (N,11) onto the branch centred on ref."""
    R=R.copy()
    for d,per in PERD.items():
        R[:,d]=ref[d]+((R[:,d]-ref[d]+per/2)%per)-per/2
    return R
NDRAW=8
print("\nBlock-proposal acceptance (Gaussian, per-MODE pooled cov, periodic-unwrapped)")
print(f"{'leaf':4s} {'marginal safe-block':>20s} {'conditional safe|sky':>22s}")
for leaf in range(6):
    addback(C[:,leaf],+1)
    base=score(C[:,leaf],np.arange(nw))
    # per-walker: fit mu/Sigma on the rows of that walker's OWN track (its mode)
    accs={"marg":[], "cond":[]}
    for kind in ("marg","cond"):
        cands=[]; widx=[]
        for w in range(nw):
            # pool ALL rows in this walker's own MODE (nearest-image label), and
            # unwrap periodic dims onto the walker's branch -> a real covariance
            ref=C[w,leaf]
            allr=hist.reshape(-1,6,11)[:,leaf,:].astype(np.float64)
            allr=unwrap_to(allr, ref)
            dsky=np.linalg.norm((allr[:,SKY]-ref[SKY])/(allr[:,SKY].std(0)+1e-30),axis=1)
            sel=dsky<np.percentile(dsky,35)              # same mode as walker w
            R=allr[sel]
            if len(R)<200: R=allr[np.argsort(dsky)[:400]]
            mu=R.mean(0); Sg=np.cov(R.T)+1e-12*np.eye(11)
            iS=np.linalg.pinv(Sg[np.ix_(SKY,SKY)])
            for _ in range(NDRAW):
                x=unwrap_to(C[w:w+1,leaf].copy(), ref)[0]
                if kind=="marg":
                    s=rng.multivariate_normal(mu[SAFE], Sg[np.ix_(SAFE,SAFE)])
                else:
                    d=x[SKY]-mu[SKY]
                    mc=mu[SAFE]+Sg[np.ix_(SAFE,SKY)]@iS@d
                    Sc=Sg[np.ix_(SAFE,SAFE)]-Sg[np.ix_(SAFE,SKY)]@iS@Sg[np.ix_(SKY,SAFE)]
                    s=rng.multivariate_normal(mc, (Sc+Sc.T)/2 + 1e-14*np.eye(7))
                x[SAFE]=s
                for d,per in PERD.items(): x[d]=x[d]%per   # back into prior range
                cands.append(x); widx.append(w)
        ll=score(np.array(cands),np.array(widx))
        d=ll-base[np.array(widx)]
        accs[kind]=np.mean(np.minimum(1,np.exp(np.clip(d,-700,0))))
    addback(C[:,leaf],-1)
    print(f"{leaf:<4d} {accs['marg']:20.3f} {accs['cond']:22.3f}")
log("DONE")
