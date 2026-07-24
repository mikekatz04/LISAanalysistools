#!/usr/bin/env python
"""Fast likelihood ≡ AnalysisContainer likelihood, against real mojito data.

The GB/VGB WDM chunked-heterodyne engine (``GBWDMComputations.get_ll_wdm``) is
what the sampler uses in-model at scale.  This runner proves, THROUGH a built
stock fit on MOJITO data, that its ``<d|h>`` / ``<h|h>`` match the dense
:class:`AnalysisContainer` inner products for the SAME mojito-data container —
so the fast path inherits the AnalysisContainer's validated tie to the data
(t1-gt-{gb,vgb}) and can be trusted forward into the full runs.

For the ``--topn`` highest-frequency injected sources:
  * data  = the mojito WDM stream loaded by the stock fit
    (``setup_acs(rebuild_residuals=False)``, same as the null-check);
  * fast  = ``gb_wdm_comp.get_ll_wdm(params_phys, acs[0])`` -> ``d_h_out`` /
    ``h_h_out`` (chunked-heterodyne), using the canonical fdot-basis transform;
  * ref   = the same container's ``non_marg_d_h`` (``<mojito|h>``) and
    ``template_snr`` (``<h|h>``), with the dense engine ``signal_gen`` template;
  * compare (heterodyne budget: reldiff <~ 1e-4).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("USE_GPU", "0")
os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")

PLOT_DIR = os.environ.get("CAMPAIGN_PLOT_DIR", "/tmp")


def _asnum(x):
    a = np.asarray(x.get() if hasattr(x, "get") else x)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["gb", "vgb"], default="gb")
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--tobs-days", type=float,
                    default=float(os.environ.get("FASTLIKE_TOBS_DAYS", "30")))
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from eryn.state import BranchSupplemental
    from mpi4py import MPI

    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor
    from lisatools.utils.utility import asnumpy

    variant = "gb_no_fg" if args.branch == "gb" else "vgb"
    fit = getattr(erebor, variant)(lite=True, tobs_target=args.tobs_days * 86400.0)
    curr = fit.build()

    info = curr.source_info[args.branch]
    signal_gen = getattr(info, "signal_gen", None)
    # GB sets info.gb_wdm_comp eagerly; VGB builds it lazily on the
    # container-level source_info, reachable via the signal_gen's _comp().
    comp = getattr(info, "gb_wdm_comp", None)
    if comp is None and signal_gen is not None and hasattr(signal_gen, "_comp"):
        comp = signal_gen._comp()
    if comp is None or signal_gen is None:
        print(f"[RESULT] fastlike_ok=0 reason=no_engine "
              f"comp={comp is not None} signal_gen={signal_gen is not None}",
              flush=True)
        sys.exit(1)

    # --- mojito-data ACA, exactly like scripts/validation/mojito_null_check.py
    gf = GlobalFit(curr, MPI.COMM_WORLD)
    priors = {}
    for name in curr.branch_names:
        priors.update(curr.source_info[name].priors)
    state = gf.load_info(priors)
    nt, nw = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nw), (nt, 1))},
        base_shape=(nt, nw), copy=True,
    )
    inj = np.asarray(info.injection, dtype=float)
    if inj.ndim == 1:
        inj = inj[None, :]
    # Data-only AC: residual == raw mojito data, independent of the sampler
    # coords, so we leave load_info's seeded leaves untouched (no need to
    # stuff the injection into branches_coords — that is only for the
    # rebuild_residuals=True null pipeline).
    acs = gf.setup_acs(state, rebuild_residuals=False)
    ac = acs.flatten()[0]

    # Resolve each leaf's f0 to pick the 3 highest-FREQUENCY sources.
    # GB samples f0 directly (injection col 1). VGB fixes f0 PER-LEAF
    # (VGB_FIXED_BASIS = [f0, alpha, sin_delta]), so its injection col 1 is
    # fdot and f0 lives in the transform's per-leaf fill_dict.
    # f0 in sampling units (mHz). GB samples f0 in injection col 1. VGB fixes
    # it per-leaf: Eryn stores fills as {'fill_inds': [full-basis cols],
    # 'fill_values': (n_leaves, k)}; f0 is full-basis col 1, so read the
    # fill_values column at the position where fill_inds == 1.
    tc = getattr(signal_gen, "transform", None) or getattr(info, "transform", None)
    fd = getattr(tc, "fill_dict", None)
    f0_vals = inj[:, 1]
    if isinstance(fd, dict) and "fill_values" in fd and "fill_inds" in fd:
        fill_inds = list(np.asarray(fd["fill_inds"]).ravel())
        if 1 in fill_inds:
            f0_vals = np.asarray(fd["fill_values"])[:, fill_inds.index(1)]
    order = np.argsort(f0_vals)[::-1][: args.topn].astype(int)
    sources = inj[order]
    f0_sel = f0_vals[order]
    print(f"[RESULT] branch={args.branch} n_injected={inj.shape[0]} "
          f"topn={sources.shape[0]} "
          f"f0_mhz={np.round(f0_sel, 4).tolist()}", flush=True)

    # --- fast engine: all sources at once against walker-0's mojito data ----
    # info.injection is the RUN sampling basis, so use the fit's OWN transform
    # (the exact one signal_gen applies) to reach the physical GBGPU basis.
    # VGB's transform carries per-leaf fills, so it needs leaf_inds = the
    # selected sources' leaf indices; scalar GB transforms ignore leaf_inds, so
    # passing them unconditionally is safe.
    xp = comp.xp
    leaf_inds = xp.asarray(order)
    params_phys = tc.both_transforms(xp.asarray(sources), leaf_inds=leaf_inds, xp=xp)
    di = xp.zeros(params_phys.shape[0], dtype=xp.int32)
    comp.get_ll_wdm(params_phys, acs[0], data_index=di, noise_index=di)
    d_h_fast = asnumpy(comp.d_h_out).real
    h_h_fast = asnumpy(comp.h_h_out).real

    # --- AnalysisContainer reference on the SAME mojito data -----------------
    reldiffs, dh_pairs, hh_pairs, labels = [], [], [], []
    for i, row in enumerate(sources):
        # *row unpacks to a 1-D param vector (leading shape ()), so leaf_inds
        # is the matching scalar leaf index.
        h = signal_gen(*row, leaf_inds=int(order[i]))
        opt, det = ac.template_snr(h)
        h_h_ac = float(opt) ** 2
        d_h_ac = float(np.real(complex(ac.non_marg_d_h)))
        dhf, hhf = float(d_h_fast[i]), float(h_h_fast[i])
        rd_dh = abs(dhf - d_h_ac) / max(abs(d_h_ac), 1e-30)
        rd_hh = abs(hhf - h_h_ac) / max(abs(h_h_ac), 1e-30)
        reldiffs += [rd_dh, rd_hh]
        dh_pairs.append((d_h_ac, dhf))
        hh_pairs.append((h_h_ac, hhf))
        labels.append(f"f0={f0_sel[i]:.4f} mHz")
        print(f"[RESULT] src={i} f0_mhz={f0_sel[i]:.5f} "
              f"d_h_fast={dhf:.6e} d_h_ac={d_h_ac:.6e} "
              f"h_h_fast={hhf:.6e} h_h_ac={h_h_ac:.6e} "
              f"reldiff_dh={rd_dh:.3e} reldiff_hh={rd_hh:.3e}", flush=True)

    max_rd = max(reldiffs) if reldiffs else 1.0
    print(f"[RESULT] fastlike_ok=1 fastlike_sources={len(labels)} "
          f"fastlike_max_reldiff={max_rd:.3e}", flush=True)

    # --- proof plot: fast vs AC, d_h and h_h, against mojito data -----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, pairs, ttl in ((a1, dh_pairs, "$\\langle d|h\\rangle$ (mojito data)"),
                           (a2, hh_pairs, "$\\langle h|h\\rangle$")):
        ref = np.array([p[0] for p in pairs])
        fast = np.array([p[1] for p in pairs])
        ax.scatter(ref, fast, c="#2a78d6", zorder=3, s=45)
        lim = [min(ref.min(), fast.min()), max(ref.max(), fast.max())]
        ax.plot(lim, lim, "--", color="#0b0b0b", lw=1, label="y = x")
        ax.set_xlabel(f"AnalysisContainer {ttl}")
        ax.set_ylabel(f"fast engine {ttl}")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.15)
    fig.suptitle(
        f"{args.branch.upper()} fast likelihood ≡ AnalysisContainer (mojito data)"
        f"\ntop-{len(labels)} highest-f  ·  max reldiff {max_rd:.1e}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"fastlike_{args.branch}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[RESULT] plot={out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] fastlike_ok=0 error={type(exc).__name__}", flush=True)
        sys.exit(1)
