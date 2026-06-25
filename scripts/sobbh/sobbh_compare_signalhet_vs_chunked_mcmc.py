#!/usr/bin/env python
"""Compare SOBBH signal-het vs chunked-het MCMC posteriors on one binary.

SOBBH analogue of
``gb_chunked_het/compare_signalhet_vs_chunked_mcmc.py``. Same synthetic SOBBH
injection, same priors / RNG / sampler config; only the likelihood kernel
differs:

  * signal-het  : SOBBHSignalHetComputations.get_ll        (v2 polyphase + bin-fold)
  * chunked-het : SOBBHWDMComputations.get_ll_wdm          (per-chunk WDM heterodyne)

Both are fast C++ CPU likelihoods, so the run is quick. To keep the corner
readable and the sampler well-mixed we vary a 4-parameter subset
(f_low, phi_c, cos_inc, ln_distance) around the injection with the other 7
parameters frozen at truth; both likelihoods see the SAME reduced space.

If the C++ port is correct the two posteriors must overlap (they are the same
physical likelihood computed two ways).

Run::
    N_STEPS=200 NWALKERS=20 python sobbh_compare_signalhet_vs_chunked_mcmc.py
Env: N_STEPS (200), NWALKERS (20), SEED (7), OUT_PNG, BACKEND (cpu)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import get_array_module

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401
from bbhx.sobbhcomps import SOBBHWDMComputations

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from sobbhsignalhetcomputations import SOBBHSignalHetComputations  # noqa: E402


class _FullGridWDMHolder:
    """Minimal duck-type for SOBBHWDMComputations.get_ll_wdm's wdm_holder."""
    def __init__(self, data_full, invC_full):
        xp = get_array_module(data_full)
        self.linear_data_arr = [xp.ascontiguousarray(data_full).ravel().copy()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC_full).ravel().copy()]

    def __len__(self):
        return 1


# Subset sampled: indices into the 11-vec, plus (lo, hi) prior widths.
SUB_IDX = [5, 6, 7, 9]   # f_low, phi_c, inc, lam


def main():
    N_STEPS = int(os.environ.get("N_STEPS", "200"))
    NWALK = int(os.environ.get("NWALKERS", "20"))
    SEED = int(os.environ.get("SEED", "7"))
    OUT_PNG = os.environ.get("OUT_PNG", "sobbh_compare_signalhet_chunked.png")
    backend = os.environ.get("BACKEND", "cpu")

    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nobs * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    sobbh_gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                                 tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                                 force_backend=backend)

    def real_td_cb(p):
        sp = sobbh_gen(*np.asarray(p, float).reshape(11, 1),
                       convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)
    wdm_kw = dict(t0=t_start, min_freq=1e-4, max_freq=35e-3,
                  min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt, force_backend=backend)
    wdm_set_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
    layer_df = wdm_set_real.layer_df
    ind_min_f = int(wdm_set_real.ind_min_f)

    f_low = float((ind_min_f + 80) * layer_df)
    truth = np.array([42.0, 38.0, 0.1, 0.2, 5.0e8, f_low, 1.1,
                      np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = real_td_cb(truth)
    data_real = TDSignal(td_inj, settings=td_set).transform(wdm_set_real, window=window)
    sens = XYZ2SensitivityMatrix(data_real.settings, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_real), sens)
    d_d = float(np.real(analysis.inner_product()))
    snr = float(analysis.snr())
    print(f"[inj] f_low={f_low*1e3:.4f}mHz SNR={snr:.1f}", flush=True)

    # signal-het likelihood
    sighet = SOBBHSignalHetComputations(
        td_inj, truth, Nf=Nf, Nt=Nt, dt=dt, t0=t_start, t_ref=t_start,
        orbits=orbits, tdi_config=tdi_config, min_freq=1e-4, max_freq=35e-3,
        force_backend=backend)

    # chunked-het likelihood
    chunked = SOBBHWDMComputations(
        wdm_set_real, t_ref=t_start, Nt_sub=256, n_pad=32, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ")
    inj_active = np.asarray(data_real.arr)
    invC_active = np.asarray(sens.invC)
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)
    holder = _FullGridWDMHolder(inj_active, invC_active)

    def to_full(x_sub):
        x_sub = np.atleast_2d(x_sub)
        full = np.tile(truth, (x_sub.shape[0], 1))
        full[:, SUB_IDX] = x_sub
        return full

    def logl_sighet(x, **_kw):
        full = to_full(x)
        ll = np.asarray(sighet.get_ll(full)) - 0.0
        return np.where(np.isfinite(ll) & (ll < 50.0), ll, -1e10)

    def logl_chunked(x, **_kw):
        full = to_full(x)
        ll = np.asarray(chunked.get_ll_wdm(
            full, holder, convert_to_ra_dec=False,
            use_layer_groups=True, group_band_layers=5, margin_layers=0)) - 0.5 * d_d
        return np.where(np.isfinite(ll) & (ll < 50.0), ll, -1e10)

    # sanity: both logL at truth
    ll_s0 = float(np.asarray(logl_sighet(truth[SUB_IDX]))[0])
    ll_c0 = float(np.asarray(logl_chunked(truth[SUB_IDX]))[0])
    print(f"[truth] logL sig-het={ll_s0:+.4e}  chunked={ll_c0:+.4e}  "
          f"diff={abs(ll_s0 - ll_c0):.3e}", flush=True)

    # priors (tight around truth)
    pri = {0: {
        0: uniform_dist(f_low - 0.3 * layer_df, f_low + 0.3 * layer_df),
        1: uniform_dist(truth[6] - 0.6, truth[6] + 0.6),
        2: uniform_dist(max(0.0, truth[7] - 0.5), min(np.pi, truth[7] + 0.5)),
        3: uniform_dist(truth[9] - 0.6, truth[9] + 0.6),
    }}
    priors = {"sobbh": ProbDistContainer(pri[0])}
    ndim = len(SUB_IDX)

    rng = np.random.default_rng(SEED)
    start = truth[SUB_IDX][None, :] + np.array([0.02 * layer_df, 0.05, 0.05, 0.05]) * \
        rng.standard_normal((NWALK, ndim))
    start = start[None, :, None, :]  # (ntemps=1, nwalkers, nleaves=1, ndim)

    chains = {}
    for tag, logl in [("sighet", logl_sighet), ("chunked", logl_chunked)]:
        sampler = EnsembleSampler(
            NWALK, {"sobbh": ndim}, logl, priors,
            nleaves_max={"sobbh": 1}, nleaves_min={"sobbh": 1},
            tempering_kwargs=dict(ntemps=1),
        )
        np.random.seed(SEED)
        sampler.run_mcmc(start, N_STEPS, progress=False, burn=N_STEPS // 4)
        ch = sampler.get_chain()["sobbh"][:, 0, :, 0, :].reshape(-1, ndim)
        chains[tag] = ch
        print(f"[{tag}] chain {ch.shape}  acc={np.mean(sampler.acceptance_fraction):.3f}",
              flush=True)

    # corner overlay
    labels = ["f_low", "phi_c", "inc", "lam"]
    fig, axes = plt.subplots(ndim, ndim, figsize=(10, 10))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                for tag, c in [("sighet", "C0"), ("chunked", "C1")]:
                    ax.hist(chains[tag][:, i], bins=30, histtype="step", color=c, density=True,
                            label=tag)
                ax.axvline(truth[SUB_IDX][i], color="k", ls="--", lw=1)
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                ax.scatter(chains["sighet"][:, j], chains["sighet"][:, i], s=2, alpha=0.3, color="C0")
                ax.scatter(chains["chunked"][:, j], chains["chunked"][:, i], s=2, alpha=0.3, color="C1")
                ax.plot(truth[SUB_IDX][j], truth[SUB_IDX][i], "k*", ms=8)
            if i == ndim - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
    fig.suptitle(f"SOBBH sig-het (C0) vs chunked-het (C1) posteriors  SNR={snr:.0f}", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_PNG, dpi=110)
    print(f"\n[write] {OUT_PNG}", flush=True)

    # quantitative overlap: 1D median/std agreement
    print("\n[posterior medians (sig-het vs chunked)]", flush=True)
    for i, lab in enumerate(labels):
        ms, mc = np.median(chains["sighet"][:, i]), np.median(chains["chunked"][:, i])
        ss = np.std(chains["sighet"][:, i])
        print(f"  {lab:>6s}: sighet={ms:+.5e}  chunked={mc:+.5e}  "
              f"|Δ|/σ={abs(ms - mc)/max(ss, 1e-30):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
