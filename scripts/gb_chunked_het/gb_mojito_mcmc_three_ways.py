#!/usr/bin/env python
"""MCMC on ONE high-frequency mojito galactic binary, THREE ways that differ
ONLY by the analysis domain:

  (1) "fd"          -- FREQUENCY DOMAIN  : FDSettings  + GBFDHetComputations.get_ll
  (2) "wdm_chunked" -- chunked-het WDM   : WDMSettings + GBWDMComputations.get_ll_wdm
  (3) "wdm_sighet"  -- signal-het WDM    : WDMSettings + GBSignalHetComputations.get_ll

The FD path uses the SAME GBTDIonTheFly waveform the WDM paths start with (the
full rfft of the TD-on-the-fly signal), so its mismatch matches the WDM domain
(mm ~ 1e-11 at injection) -- NOT GBGPU's separate fast-FD model. GBFDHetComputations
exposes the full comp surface (get_ll / fill_global / get_swap_ll). It is the
EXACT (dense full-rfft) FD path, so it is slow (~2 s/template on CPU); the WDM
paths are ~100x faster via their heterodyne kernels.

Design goals (per user direction):
  * MINIMAL new code -- the log-likelihood VALUE comes straight out of the
    installed ``get_ll`` methods; nothing is re-derived here.
  * The ONLY thing that changes between the three runs is the DOMAIN object
    (FDSettings vs WDMSettings) + the matching installed computations class,
    built in ``build_method``. The sampler, priors, transform, periodicity,
    start cloud and corner overlay are shared, identical code.
  * Priors / transform / periodicity are taken from the GB global-fit settings
    (``erebor.make_gb_transform_container`` + the ``GBSetup`` prior recipe +
    ``get_fdot_mojito``).
  * Uses Eryn end to end: EnsembleSampler, StretchMove, parallel tempering,
    ProbDistContainer/uniform_dist, TransformContainer, PeriodicContainer,
    State, HDFBackend.

ONE source at a time: GB_RANK selects the rank-N highest-frequency mojito GB
(0 = loudest/highest f). All three target the SAME source, band and start cloud.

Sampling basis (Eryn, all three methods, REF epoch):
  [logA, f0(mHz), fdot, phi0, cos_iota, psi, alpha, sin_delta]
All three reference the GB phase to REF (t_ref = REF) and analyse the same
mojito band, so the posteriors are in the SAME basis and overlay directly.

Run (one method or all):
  GB_RANK=0 METHODS=fd,wdm_chunked,wdm_sighet python gb_mojito_mcmc_three_ways.py
Env: GB_RANK, METHODS, NWALKERS, NTEMPS, NSTEPS, BURNIN, GB_NT, PRIOR_F0_LAYERS,
     START_FACTOR, FD_BATCH, FD_N_SPARSE, FD_IMPL, FD_EDGE_MODE, BACKEND, OUT_DIR.
     FD_EDGE_MODE = "active" (FD over the WDM active time [EC,Nt-EC], matches the
     WDM methods) | "full" (FD over the full Tobs; uses the edge data the WDM drops).
"""
from __future__ import annotations
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gbgpu  # noqa: F401  (registers gbgpu backends)
from lisatools.detector import L1Orbits
from lisatools.domains import TDSettings, TDSignal, FDSettings, WDMSettings
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.stock.erebor import make_gb_transform_container
from lisatools.globalfit.priors.gbpriors import get_fdot_mojito
from lisatools.sources.utils import evolve_galactic_binary
from gbgpu.gbcomps import GBWDMComputations, GBFDComputations

from gbsignalhetcomputations import recommended_edge_cut          # dev helper
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations  # INSTALLED class
# ^ sig-het reference now comes from the GBGPU backend (gb_signal_het_make_reference)
#   + lisatools window/bin-fold; no Python polyphase. The dev gbsignalhetcomputations
#   .py is kept only for recommended_edge_cut + the walkthrough explainer's internals.
from gbfdhetcomputations import GBFDHetComputations  # dense reference (FD_IMPL=dense)

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer
from eryn.backends import HDFBackend

REF = 97729089.327664
MOJ_PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MOJ_CACHE = "/tmp/gb_mojito_data_365d.npz"
MEM_CAP_GB = float(os.environ.get("GB_MEM_CAP_GB", "12.0")); _IS_MAC = sys.platform == "darwin"


def _watchdog():
    while True:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if (r / 1e9 if _IS_MAC else r / 1e6) > MEM_CAP_GB:
            os._exit(42)
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Shared geometry / data, built once and reused by all three methods.
# ---------------------------------------------------------------------------
def build_shared():
    backend = os.environ.get("BACKEND", "cpu"); dt = 10.0; Nf = 1460
    Nt = int(os.environ.get("GB_NT", "512"))
    Nobs = Nf * Nt; Tobs = Nobs * dt
    layer_df = 1.0 / (2.0 * Nf * dt); TUK = 0.05; M_HALF = 2
    EC = recommended_edge_cut(Nt, TUK, "chunked")

    z = np.load(MOJ_CACHE)
    data_td = np.ascontiguousarray(np.asarray(z["data_td"])[:3, :Nobs])
    data_t0 = float(z["data_t0"])
    rank = int(os.environ.get("GB_RANK", "0"))
    amp, f0, fdot, phi0, inc, psi, ra, dec = np.asarray(z["top_params"])[rank]
    # physical 9-vector at REF: [amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]
    p_inj = np.array([amp, f0, fdot, 0.0, phi0, inc, psi, ra, dec])
    # sampled 8-vector (REF): [logA, f0(mHz), fdot, phi0, cos_iota, psi, alpha, sin_delta]
    # phi0 (2pi), psi (pi) and alpha (2pi) are periodic; the cache stores them in a
    # signed range, so wrap into the prior support [0, period) (physically identical).
    inj_sampled = np.array([np.log(amp), f0 * 1e3, fdot, phi0 % (2 * np.pi),
                            np.cos(inc), psi % np.pi, ra % (2 * np.pi), np.sin(dec)])

    m_floor = int(f0 / layer_df); BAND = 15
    lo_f = (m_floor - BAND) * layer_df; hi_f = (m_floor + BAND + 1) * layer_df

    orbits = L1Orbits(find_file(os.path.join(MOJ_PATH, "data", "GB", "L1"), "GB", 0),
                      force_backend=backend, frame="icrs")
    pad = 1e5; lo = max(data_t0 - pad, float(orbits.sc_t0))
    hi = min(data_t0 + Tobs + pad, float(orbits._sc_t_base[-1]))
    lt = np.asarray(orbits.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orbits.ltt = np.asarray(orbits.ltt)[mk].copy(); orbits.ltt_t = lt[mk].copy()
    orbits.ltt_t0 = float(orbits.ltt_t[0]); del lt; gc.collect()
    orbits.configure(linear_interp_setup=True)

    return dict(backend=backend, dt=dt, Nf=Nf, Nt=Nt, Nobs=Nobs, Tobs=Tobs,
                layer_df=layer_df, TUK=TUK, M_HALF=M_HALF, EC=EC, data_td=data_td,
                data_t0=data_t0, rank=rank, p_inj=p_inj, inj_sampled=inj_sampled,
                f0=f0, m_floor=m_floor, lo_f=lo_f, hi_f=hi_f, orbits=orbits)


def _clean(ll, ceiling=10.0, reject=-1e30):
    ll = np.asarray(getattr(ll, "get", lambda: ll)(), dtype=float)  # cupy-safe
    return np.where(~np.isfinite(ll) | (ll > ceiling), reject, ll)


# ---------------------------------------------------------------------------
# THE only method-specific code: build the domain object + installed comp +
# a logl(x_sampled, transform_fn) closure that returns get_ll straight out.
# ---------------------------------------------------------------------------
def build_method(kind, s):
    backend, dt, Nf, Nt = s["backend"], s["dt"], s["Nf"], s["Nt"]
    Nobs, Tobs, data_td, data_t0 = s["Nobs"], s["Tobs"], s["data_td"], s["data_t0"]
    lo_f, hi_f, EC, TUK, M_HALF = s["lo_f"], s["hi_f"], s["EC"], s["TUK"], s["M_HALF"]
    p_inj, orbits = s["p_inj"], s["orbits"]
    td_set = TDSettings(Nobs, dt, t0=data_t0, force_backend=backend)
    from scipy.signal.windows import tukey
    window = tukey(Nobs, TUK).astype(float)
    keep = {}

    if kind == "fd":
        # ---- C++ FD HETERODYNE inner product (GBFDComputations.gb_fd_get_ll) --
        # built on gbfd_build_one_source, the SAME C++ FD heterodyne the WDM
        # sig-het kernel uses. gb_fd_get_ll requires t_start==t_ref, so t_ref=
        # data_t0 and the closure re-references each candidate REF->data_t0
        # (phase_sign=-1, exact). Set FD_IMPL=dense for the slow Python full-rfft.
        #
        # FD_EDGE_MODE selects the FD analysis TIME width -- applied identically
        # to the data->FD transform AND the FD likelihood (dense + fast kernel):
        #   "active" (default): restrict to the WDM grid's [EC, Nt-EC] active
        #       region (tukey x edge-cut boxcar; edge_frac=EC/Nt) so the FD logL
        #       MATCHES the WDM methods (chunked/sig-het) for a direct comparison.
        #   "full":  full Tobs (tukey only, edge_frac=0) -- the FD uses all the
        #       data the WDM throws away at the edges; it then does NOT match the
        #       (edge-cut) WDM but is the more informative standalone FD analysis.
        from scipy.signal.windows import tukey as _tk
        FD_EDGE_MODE = os.environ.get("FD_EDGE_MODE", "active").lower()
        ec_layers = EC if FD_EDGE_MODE == "active" else 0
        domain = FDSettings(Nobs // 2 + 1, 1.0 / Tobs, force_backend=backend)
        if os.environ.get("FD_IMPL", "cpp") == "dense":
            comp = GBFDHetComputations(
                data_td, p_inj, Nf=Nf, Nt=Nt, dt=dt, t0=data_t0, t_ref=REF,
                orbits=orbits, tdi_config="2nd generation", min_freq=lo_f, max_freq=hi_f,
                tukey_alpha=TUK, edge_cut=ec_layers, batch=int(os.environ.get("FD_BATCH", "8")),
                force_backend=backend, tdi_type="XYZ")
            d_d = comp.d_d; keep["comp"] = comp

            def logl(x, transform_fn=None, **_kw):
                x = np.atleast_2d(np.asarray(x, dtype=float))
                phys = transform_fn.both_transforms(x.copy()) if transform_fn is not None else x
                return _clean(comp.get_ll(phys))
        else:
            # window = tukey, x edge-cut boxcar in "active" mode (matches the WDM
            # active region); the edge_frac below makes the FD-het TEMPLATE use
            # the same window. Both data and template share this window.
            win = _tk(Nobs, TUK).astype(float)
            if ec_layers > 0:
                ec_mask = np.zeros(Nobs); ec_mask[ec_layers * Nf:(Nt - ec_layers) * Nf] = 1.0
                win = win * ec_mask
            edge_frac = ec_layers / float(Nt)
            data_fd = np.ascontiguousarray(np.fft.rfft(data_td * win[None, :], axis=-1) * dt)
            n_rfft = data_fd.shape[-1]; df = 1.0 / Tobs
            invC = np.asarray(XYZ2SensitivityMatrix(
                FDSettings(n_rfft, df, force_backend=backend), model="scirdv1").invC).copy()
            invC = np.where(np.isfinite(invC), invC, 0.0); invC[:, :, 0] = 0.0
            ind_min = max(1, int(lo_f / df)); ind_max = min(n_rfft - 1, int(hi_f / df))
            sl = slice(ind_min, ind_max + 1)
            d_d = 4.0 * df * float(np.real(np.einsum(
                "ck,cdk,dk->", np.conj(data_fd[:, sl]), invC[:, :, sl], data_fd[:, sl])))
            comp = GBFDComputations(
                T=Tobs, t_ref=data_t0, t_start=data_t0,
                N_sparse=int(os.environ.get("FD_N_SPARSE", "8192")), df=df,
                data_fd=data_fd[None].astype(complex), invC=invC[None].astype(float),
                orbits=orbits, tdi_config="2nd generation", force_backend=backend,
                d_d=d_d, tdi_type="XYZ", ind_min=ind_min, ind_max=ind_max,
                tukey_alpha=TUK, edge_frac=edge_frac)
            keep["comp"] = comp

            def reref(phys):   # REF -> data_t0 (gb_fd_get_ll needs t_ref=data_t0)
                out = np.array(phys, dtype=float, copy=True)
                f_new, phi_new, _ = evolve_galactic_binary(
                    REF, data_t0, phys[:, 1], phys[:, 4], phys[:, 2],
                    fddot=phys[:, 3], phase_sign=-1)
                out[:, 1] = f_new; out[:, 4] = phi_new
                return out

            def logl(x, transform_fn=None, **_kw):
                x = np.atleast_2d(np.asarray(x, dtype=float))
                phys = transform_fn.both_transforms(x.copy()) if transform_fn is not None else x
                return _clean(comp.get_ll_fd(reref(phys), convert_to_ra_dec=False))

    elif kind == "wdm_chunked":
        # ---- CHUNKED-HET WDM -------------------------------------------------
        domain = WDMSettings(Nf, Nt, dt, t0=data_t0, min_freq=lo_f, max_freq=hi_f,
                             min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                             is_complex=False, force_backend=backend)
        data_wdm = TDSignal(data_td, settings=td_set).transform(domain, window=window)
        sens = XYZ2SensitivityMatrix(domain, model="scirdv1")
        an = AnalysisContainer(DataResidualArray(data_wdm), sens)
        d_d = float(np.real(an.inner_product()))
        holder = AnalysisContainerArray([an])
        comp = GBWDMComputations(domain, t_ref=REF, Nt_sub=256, n_pad=32, N_sparse=256,
                                 N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
                                 force_backend=backend, d_d=d_d, tdi_type="XYZ")
        keep.update(comp=comp, holder=holder, an=an)

        def logl(x, transform_fn=None, **_kw):
            x = np.atleast_2d(np.asarray(x, dtype=float))
            phys = transform_fn.both_transforms(x.copy()) if transform_fn is not None else x
            return _clean(comp.get_ll_wdm(phys, holder, convert_to_ra_dec=False,
                                          use_layer_groups=True, group_band_layers=5, margin_layers=0))

    elif kind == "wdm_sighet":
        # ---- SIGNAL-HET WDM --------------------------------------------------
        domain = WDMSettings(Nf, Nt, dt, t0=data_t0, min_freq=lo_f, max_freq=hi_f,
                             min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                             is_complex=False, force_backend=backend)
        # SIGHET_REF_IMPL toggles HOW the sig-het reference c0 is built:
        #   "chunked" (default) -- the chunked-het/polyphase WDM method (active band,
        #                          full Nt), proven == dense to ~3.7e-16 logL.
        #   "dense"             -- full-TD reference -> TDSignal.transform(wdm_complex).
        comp = GBSignalHetComputations(
            data_td, p_inj, Nf=Nf, Nt=Nt, dt=dt, t0=data_t0, t_ref=REF,
            orbits=orbits, tdi_config="2nd generation", min_freq=lo_f, max_freq=hi_f,
            edge_cut=EC, m_active_half_width=M_HALF, nt_layer=64, tukey_alpha=TUK,
            force_backend=backend)
        d_d = comp.d_d; keep["comp"] = comp

        def logl(x, transform_fn=None, **_kw):
            x = np.atleast_2d(np.asarray(x, dtype=float))
            phys = transform_fn.both_transforms(x.copy()) if transform_fn is not None else x
            return _clean(comp.get_ll(phys))

    else:
        raise ValueError(f"unknown method {kind!r}")

    return dict(kind=kind, domain=domain, logl=logl, d_d=d_d, keep=keep)


# ---------------------------------------------------------------------------
# Shared Eryn pieces: priors / transform / periodic from the GB global-fit
# settings, and one EnsembleSampler run.
# ---------------------------------------------------------------------------
def build_eryn(s):
    amp, f0, fdot = s["p_inj"][0], s["p_inj"][1], s["p_inj"][2]
    layer_df = s["layer_df"]
    F0L = float(os.environ.get("PRIOR_F0_LAYERS", "2.0"))
    fdot_lims = [get_fdot_mojito(f0, "-"), get_fdot_mojito(f0, "+")]
    # GBSetup prior recipe (uniform on logA, f0[mHz], fdot, phi0, cos_iota, psi,
    # alpha, sin_delta), band-localised f0 prior.
    priors = {"gb": ProbDistContainer({
        0: uniform_dist(np.log(amp * 0.2), np.log(amp * 5.0)),
        1: uniform_dist((f0 - F0L * layer_df) * 1e3, (f0 + F0L * layer_df) * 1e3),
        2: uniform_dist(min(fdot_lims), max(fdot_lims)),
        3: uniform_dist(0.0, 2 * np.pi),
        4: uniform_dist(-1.0, 1.0),
        5: uniform_dist(0.0, np.pi),
        6: uniform_dist(0.0, 2 * np.pi),
        7: uniform_dist(-1.0, 1.0),
    })}
    transform = make_gb_transform_container()
    periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}},
                                 key_order={"gb": list(range(8))})
    return priors, transform, periodic


def start_cloud(s, ntemps, nwalkers, seed):
    inj = s["inj_sampled"]; layer_df = s["layer_df"]
    sf = float(os.environ.get("START_FACTOR", "1e-3"))
    np.random.seed(seed)
    # tight Eryn ProbDistContainer ball around the injection (sub-layer in f0)
    gen = ProbDistContainer({
        0: uniform_dist(inj[0] - sf, inj[0] + sf),
        1: uniform_dist(inj[1] - sf * layer_df * 1e3, inj[1] + sf * layer_df * 1e3),
        2: uniform_dist(inj[2] - 1e-18, inj[2] + 1e-18),
        3: uniform_dist(inj[3] - sf, inj[3] + sf),
        4: uniform_dist(max(-0.999, inj[4] - sf), min(0.999, inj[4] + sf)),
        5: uniform_dist(inj[5] - sf, inj[5] + sf),
        6: uniform_dist(inj[6] - sf, inj[6] + sf),
        7: uniform_dist(max(-0.999, inj[7] - sf), min(0.999, inj[7] + sf)),
    })
    return gen.rvs(size=(ntemps, nwalkers, 1))


def run_one(method, s, priors, transform, periodic, start_coords, out_h5,
            nwalkers, ntemps, nsteps, burnin):
    if os.path.exists(out_h5):
        os.remove(out_h5)
    sampler = EnsembleSampler(
        nwalkers, {"gb": 8}, method["logl"], priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(transform_fn=transform),
        moves=StretchMove(live_dangerously=True),
        branch_names=["gb"], periodic=periodic,
        backend=HDFBackend(out_h5), vectorize=True,
    )
    state = State({"gb": start_coords.copy()})
    state.log_prior = sampler.compute_log_prior(state.branches_coords)
    state.log_like = sampler.compute_log_like(state.branches_coords, logp=state.log_prior)[0]
    cold = np.asarray(state.log_like)[0]
    print(f"  [{method['kind']}] start cold logL: mean={cold.mean():+.3e} max={cold.max():+.3e}",
          flush=True)
    t0 = time.perf_counter()
    sampler.run_mcmc(state, nsteps, burn=burnin, progress=True)
    el = time.perf_counter() - t0
    print(f"  [{method['kind']}] {el:.1f}s ({el/nsteps*1e3:.1f} ms/step)  -> {out_h5}", flush=True)


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    METHODS = os.environ.get("METHODS", "fd,wdm_chunked,wdm_sighet").split(",")
    METHODS = [m.strip() for m in METHODS if m.strip()]
    NWALKERS = int(os.environ.get("NWALKERS", "20"))
    NTEMPS = int(os.environ.get("NTEMPS", "10"))
    NSTEPS = int(os.environ.get("NSTEPS", "1500"))
    BURNIN = int(os.environ.get("BURNIN", "0"))
    SEED = int(os.environ.get("SEED", "42"))
    OUT_DIR = os.environ.get("OUT_DIR", ".")
    os.makedirs(OUT_DIR, exist_ok=True)

    s = build_shared()
    print(f"=== mojito GB rank {s['rank']}: f0={s['f0']*1e3:.4f} mHz  m_floor={s['m_floor']}  "
          f"Nt={s['Nt']} EC={s['EC']}  band=[{s['lo_f']*1e3:.3f},{s['hi_f']*1e3:.3f}] mHz ===", flush=True)
    priors, transform, periodic = build_eryn(s)
    start_coords = start_cloud(s, NTEMPS, NWALKERS, SEED)

    # ---- build each method (the ONLY domain-specific step) + pre-flight ------
    built = {}
    print("\n[pre-flight] logL at injection (installed get_ll, REF basis):", flush=True)
    inj_x = s["inj_sampled"][None, :]
    for kind in METHODS:
        m = build_method(kind, s)
        built[kind] = m
        ll0 = float(m["logl"](inj_x, transform_fn=transform)[0])
        print(f"    {kind:>12s}:  logL_inj = {ll0:+.4e}   (d|d) = {m['d_d']:.4e}", flush=True)

    # ---- run each method with the SHARED sampler config ----------------------
    chains = {}
    for kind in METHODS:
        out_h5 = os.path.join(OUT_DIR, f"mcmc_mojito_rank{s['rank']}_{kind}.h5")
        print(f"\n=== run {kind} -> {out_h5} ===", flush=True)
        run_one(built[kind], s, priors, transform, periodic, start_coords, out_h5,
                NWALKERS, NTEMPS, NSTEPS, BURNIN)
        chain = HDFBackend(out_h5).get_chain()["gb"]              # (nsteps, ntemps, nwalkers, 1, 8)
        nb = max(1, int(0.3 * chain.shape[0]))
        chains[kind] = chain[nb:, 0].reshape(-1, 8)

    # ---- posterior overlay ---------------------------------------------------
    labels = ["logA", "f0 [mHz]", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    print(f"\n  {'param':>10s} {'inj':>13s}" + "".join(f" {k+' mean':>14s}" for k in METHODS), flush=True)
    for j, lab in enumerate(labels):
        row = f"  {lab:>10s} {s['inj_sampled'][j]:>+13.5e}"
        for kind in METHODS:
            row += f" {chains[kind][:, j].mean():>+14.5e}"
        print(row, flush=True)

    try:
        import corner
        colors = {"fd": "C0", "wdm_chunked": "C1", "wdm_sighet": "C2"}
        fig = None
        for kind in METHODS:
            fig = corner.corner(
                chains[kind], labels=labels, fig=fig, color=colors.get(kind, "C3"),
                truths=s["inj_sampled"], truth_color="black",
                levels=(0.393, 0.865, 0.989), plot_datapoints=False, plot_density=False,
                no_fill_contours=True, smooth=1.0, hist_kwargs={"density": True})
        fig.legend(handles=[plt.Line2D([0], [0], color=colors.get(k, "C3"), lw=2, label=k)
                            for k in METHODS], loc="upper right", fontsize=11)
        fig.suptitle(f"mojito GB rank {s['rank']} (f0={s['f0']*1e3:.3f} mHz): FD vs chunked-het vs "
                     f"signal-het posteriors\n({NWALKERS}w x {NTEMPS}T x {NSTEPS} steps)", y=1.02)
        out_png = os.path.join(OUT_DIR, f"corner_mojito_rank{s['rank']}_three_ways.png")
        fig.savefig(out_png, bbox_inches="tight", dpi=140); plt.close(fig)
        print(f"\n[plot] {out_png}", flush=True)
    except Exception as e:
        print(f"[plot] skipped corner overlay: {e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
