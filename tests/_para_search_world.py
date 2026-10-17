"""Tiny synthetic WDM GB world for the parallel-search tests (CPU, laptop-sized).

A short WDM grid (default 15 d: Nf=256, Nt=512, dt=10 s) with the orbits
configured on a coarse window (the GBGPU psd-mirror fixture's trick: the dense
full-file orbit grid is ~3 GB), a cheap chunked-het comp, an analytic XYZ
inverse covariance (scirdv1, with unequal channel gains), GBs injected at
target OPTIMAL SNRs in chosen sub-bands, and optional WDM-domain Gaussian
noise drawn with the same covariance.

Not a test module (leading underscore): imported by
``tests/test_gb_para_search*.py``.
"""

from __future__ import annotations

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.sensitivity import X2TDISens, XY2TDISens
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations


class OneRowHolder:
    def __init__(self, res, invC):
        self.linear_data_arr = [np.ascontiguousarray(res[None]).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC[None]).ravel()]

    def __len__(self):
        return 1


#: (interior band, offset in bins from the band's low edge, target opt SNR)
DEFAULT_SOURCES = (
    (1, 30.0, 4.0),
    (2, 30.0, 6.0),
    (4, 22.0, 9.0),
    (5, 18.0, 7.0),
    (5, 26.0, 6.0),      # 8 bins above its band-mate
    (6, 33.0, 15.0),
)


def covariance(wdm):
    nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
    fl = np.maximum((wdm.ind_min_f + np.arange(nfa)) * wdm.layer_df, 1e-5)
    sxx = np.asarray(X2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
    sxy = np.asarray(XY2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
    C = np.empty((nfa, 3, 3))
    for c in range(3):
        for d in range(3):
            C[:, c, d] = sxx if c == d else sxy
    D = np.diag([0.99, 1.0, 1.01])
    return D[None] @ C @ D[None]                     # (nfa, 3, 3)


def build_world(*, nf=256, nt=256, dt=10.0, edge=24, f_start=3.0e-3,
                band_bins=64, n_bands=10, sources=DEFAULT_SOURCES, noise=False,
                noise_seed=3, sighet=True, comp_kw=None, mc=0.3, src_seed=11):
    t_start = int(0.5 * YRSID_SI / dt) * dt
    orbits = ESAOrbits(force_backend="cpu")
    T = nf * nt * dt
    o_dt = 1000.0
    margin = 64 * nf * dt
    t_arr = np.arange(-margin, T + margin + o_dt, o_dt) + t_start
    orbits._configure(t_arr=t_arr)
    wdm = WDMSettings(nf, nt, dt, t0=t_start, min_freq=1e-4, max_freq=2e-2,
                      min_time=edge * nf * dt, max_time=(nt - edge) * nf * dt,
                      force_backend="cpu")
    kw = dict(Nt_sub=64, n_pad=8, N_sparse=128, N_cp_sig=0, N_cp_orbit=0)
    kw.update(comp_kw or {})
    ch = GBWDMComputations(wdm, t_ref=t_start, orbits=orbits,
                           tdi_config="2nd generation", force_backend="cpu",
                           d_d=0.0, tdi_type="XYZ", **kw)
    ch.convert_to_ra_dec = False
    nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
    nta = int(wdm.Nt_active)
    C = covariance(wdm)
    iC = np.broadcast_to(np.linalg.inv(C).transpose(1, 2, 0)[:, :, :, None],
                         (3, 3, nfa, nta)).copy()
    tobs = float(wdm.Tobs)
    df = 1.0 / tobs
    edges = f_start + np.arange(n_bands + 1) * band_bins * df

    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

    tf = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                     use_distance=True, mc_lims=(0.001, 1.0))
    rng = np.random.default_rng(src_seed)
    data = np.zeros((3, nfa, nta))
    truth = []
    for band, off, snr in sources:
        f0 = edges[band] + off * df
        x = np.array([[1.0, f0 * 1e3, mc, rng.uniform(0, 2 * np.pi),
                       rng.uniform(-0.9, 0.9), rng.uniform(0, np.pi),
                       rng.uniform(0, 2 * np.pi), rng.uniform(-0.9, 0.9), 0.0]])
        p = tf.both_transforms(x)
        h = np.zeros((1, 3, nfa, nta))
        ch.fill_global_wdm(p, h.reshape(-1), data_index=np.zeros(1, dtype=np.int32),
                           factors=np.ones(1))
        s2 = float(np.einsum("cft,cdft,dft->", h[0], iC, h[0]))
        if not s2 > 0:
            raise RuntimeError("zero template in the para-search world (orbits?)")
        scale = snr / np.sqrt(s2)
        h *= scale
        x[0, 0] = x[0, 0] / scale          # A propto 1/d
        data += h[0]
        truth.append(dict(band=band, f0_hz=f0, snr=snr, x=x[0],
                          phys=np.asarray(tf.both_transforms(x))[0],
                          template=h[0].copy()))
    if noise:
        nrng = np.random.default_rng(noise_seed)
        L = np.linalg.cholesky(C)                       # (nfa, 3, 3)
        z = nrng.standard_normal((nfa, nta, 3))
        data = data + np.einsum("fcd,ftd->cft", L, z)
    holder = OneRowHolder(data, iC)
    sig = None
    if sighet:
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

        sig = GBSignalHetComputations.for_band_engine(
            ch, v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)
    return dict(chunked=ch, sighet=sig, wdm=wdm, holder=holder, data=data, iC=iC,
                C=C, edges=edges, truth=truth, tobs=tobs, df=df, transform=tf,
                nfa=nfa, nta=nta)


def write_synthetic_store(run_dir, world, residuals, *, name="gf_para_test",
                          leaves=None):
    """A KB-sized store + residual snapshot the CLI reads.

    ``residuals`` (nw, 3, Nf_active, Nt_active). The store carries what the
    CLI reads and nothing else: ``domain_settings``, ``noise_model_identity``
    (``data_t0``), ``sub_backend/gb/band_edges``, one cold row of
    ``chain/gb`` + ``inds/gb`` (``leaves`` = per-walker lists of 9-column
    sampling rows, default none alive) and the ``iteration`` attr.
    """
    import os

    import h5py

    from lisatools.globalfit.hdfbackend import _write_domain_settings

    os.makedirs(run_dir, exist_ok=True)
    store = os.path.join(run_dir, f"{name}.h5")
    residuals = np.asarray(residuals)
    nw = residuals.shape[0]
    nl = 4
    chain = np.zeros((1, 1, 1, nw, nl, 9))
    inds = np.zeros((1, 1, 1, nw, nl), dtype=bool)
    for w, rows in enumerate(leaves or []):
        for j, r in enumerate(rows[:nl]):
            chain[0, 0, 0, w, j] = r
            inds[0, 0, 0, w, j] = True
    with h5py.File(store, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = 1
        _write_domain_settings(g, world["wdm"])
        nmi = g.create_group("noise_model_identity")
        nmi.attrs["data_t0"] = float(world["wdm"].t0)
        g.create_dataset("sub_backend/gb/band_edges", data=np.asarray(world["edges"]))
        g.create_dataset("chain/gb", data=chain)
        g.create_dataset("inds/gb", data=inds)
    wdm = world["wdm"]
    np.savez(store[:-3] + "_residual_snapshot.npz",
             residuals=residuals.astype(np.float32),
             walkers=np.arange(nw), residual=residuals[0].astype(np.float32),
             walker=0, lnl=np.zeros(nw), store_row=0, iteration=0,
             nf=int(wdm.Nf), nt=int(wdm.Nt), dt=float(wdm.data_dt),
             data=residuals[0].astype(np.float32))
    return store


def make_builder(world, cfg, *, mode="auto"):
    from lisatools.globalfit.warmstart.para_search import (
        ParaSearchBuilder,
        ResidualScorer,
        build_search_prior,
        default_search_priors_in,
    )

    scorer = ResidualScorer(world["chunked"], world["holder"], row=0,
                            sighet=world["sighet"], tobs=world["tobs"], mode=mode)
    prior = build_search_prior(default_search_priors_in(
        dist_lims=(0.001, 1e6), ratio_max=1.0))
    return ParaSearchBuilder(scorer, band_edges_hz=world["edges"],
                             transform_fn=world["transform"], prior=prior,
                             mc_lims=(0.001, 1.0), ratio_max=1.0, cfg=cfg,
                             tag="test")
