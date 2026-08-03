"""Settings sweep: accuracy AND speed for every knob combination, per Tobs.

The proof figure answers "which ENGINE wins".  This answers "which SETTINGS
should it run with" -- the tunables that trade waveform evaluations and
shared memory against likelihood error:

    nr    fit nodes   (raw TDI evaluations per candidate; the dominant cost)
    K     fixed knots (resampling resolution; exponentials only, no evals)
    band  half-band   (cardinal-weight truncation; 0 = cooperative solve)

For each Tobs and each combination it measures the tiered error against the
chunked reference and the per-candidate cost, then renders one accuracy-vs-
cost panel per Tobs (the Pareto view: bottom-left dominates) plus marginal
panels for the individual knobs.

Run (GPU):
    USE_GPU=1 GPU_BACKEND=cuda12x python gb_sighet_settings_sweep.py
Env: SWEEP_NT_LIST ("1080,2160,4320"), SWEEP_NR ("16,32,64"),
     SWEEP_K ("64,128,256"), SWEEP_BAND ("0,16"), SWEEP_NREF (3),
     SWEEP_T (100 -- the tier the panels report), SWEEP_BATCH (1024),
     SWEEP_REPS (3), ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_speed_shootout as shoot
import gb_sighet_proof_figure as proof

from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine

USE_GPU, BACKEND = shoot.USE_GPU, shoot.BACKEND
INK, INK2, GRID = proof.INK, proof.INK2, proof.GRID
# sequential ramp for node count (magnitude), marker for band mode
# sequential ramp for node count (magnitude, light -> dark)
_RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
         "#08306b"]


def nr_colour(nr, all_nrs):
    """Light->dark by node count; magnitude gets a sequential ramp."""
    order = sorted(set(all_nrs))
    i = order.index(nr) if nr in order else 0
    return _RAMP[min(int(round(i * (len(_RAMP) - 1)
                               / max(1, len(order) - 1))), len(_RAMP) - 1)]
K_MRK = {64: "o", 128: "s", 256: "^"}



def nr_law(nt, Nf=1440, dt=10.0, base=64, base_yr=1.0):
    """Predicted fit-node count at this baseline.

    The ratio r = h_cand/h_ref is structured by the ANNUAL modulation
    (sky-Doppler + antenna pattern), so the number of oscillations a spline
    must represent scales as T_obs / 1 yr; polynomial phase (df0, dfdot)
    costs no nodes at any baseline.  Hence n_r ~ base * (T_obs / 1 yr),
    with ``base`` the value calibrated at 1 yr.
    """
    tobs_yr = nt * Nf * dt / 86400.0 / 365.25
    return int(np.clip(round(base * tobs_yr / base_yr), 4, 256))


def nr_ladder(nt, Nf=1440, dt=10.0, base=64):
    """Factor-of-two ladder bracketing the law -- so the sweep MEASURES the
    minimum passing node count at each baseline instead of assuming it."""
    n = nr_law(nt, Nf, dt, base)
    return sorted({max(4, n // 4), max(4, n // 2), n, min(256, 2 * n)})


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    nts = [int(x) for x in os.environ.get(
        "SWEEP_NT_LIST", "1080,2160,4320" if USE_GPU else "1024").split(",")]
    # SWEEP_NR="auto" (default): a Tobs-scaled ladder per baseline -- short
    # observations get FEWER nodes, which is where the cheap speedup lives.
    nr_env = os.environ.get("SWEEP_NR", "auto")
    Ks = [int(x) for x in os.environ.get("SWEEP_K", "64,128,256").split(",")]
    bands = [int(x) for x in os.environ.get("SWEEP_BAND", "0,16").split(",")]
    n_ref = int(os.environ.get("SWEEP_NREF", "3"))
    T_rep = float(os.environ.get("SWEEP_T", "100"))
    nb = int(os.environ.get("SWEEP_BATCH", "1024" if USE_GPU else "8"))
    reps = int(os.environ.get("SWEEP_REPS", "3"))
    tiers = [T_rep]

    res = {}      # (nt, nr, K, band) -> dict(cost, med, worst)
    for nt in nts:
        # node ladder FIRST (the scaffold sizes itself for the largest nr)
        nrs = (nr_ladder(nt) if nr_env == "auto"
               else [int(x) for x in nr_env.split(",")])
        # one scaffold per Tobs; engines are rebuilt per setting combination
        ws, chunked, _base, Nf, dt, t0 = shoot.build(nt, max(nrs), max(Ks),
                                                     max(bands) or 16)
        xp = chunked.xp
        ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
        ref0 = np.array([1e-22, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 0.4, 2.0, 0.3])
        href = xp.zeros((3, Nf, nt))
        chunked.fill_global_wdm(xp.asarray(ref0)[None, :], href,
                                convert_to_ra_dec=False)
        h_act = xp.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
        invC = xp.zeros((3, 3) + h_act.shape[1:])
        for c in range(3):
            invC[c, c] = 1.0
        holder0 = shoot.XpGridWDMHolder(xp, h_act, invC)
        g0 = chunked
        ntl = _base["v3"]._g["nt_layer"]

        rng = np.random.default_rng(7)
        cands = np.repeat(ref0[None, :], nb, axis=0)
        cands[:, 0] *= np.exp(0.01 * rng.standard_normal(nb))
        cands[:, 5] += 0.01 * rng.standard_normal(nb)
        cands = xp.asarray(cands)
        z = np.zeros(nb, dtype=np.int32)
        kw = dict(data_index=z, noise_index=z, N_vals=None,
                  waveform_kwargs={})

        print(f"[sweep] Nt={nt} ({nt*Nf*dt/86400/365.25:.2f} yr) "
              f"node ladder: {nrs} (law predicts {nr_law(nt, Nf, dt)})")
        for nr in nrs:
            for K in Ks:
                for band in bands:
                    try:
                        sh = GBSignalHetComputations.for_band_engine(
                            chunked, n_sparse_fd=512, n_cp_build=93,
                            nt_layer=ntl, m_active_half_width=2,
                            v3_n_nodes=nr, v4_knots=K, v4_band=band)
                        sh.clear_in_model()
                        sh.setup_in_model(holder0, xp.asarray(ref0)[None, :],
                                          np.zeros(1, np.int32))
                        eng = WDMBandLikelihoodEngine(
                            sh, ws, nchannels=3, tdi_channel_setup="XYZ")
                        cost = proof.timeit(eng, holder0, cands, kw, reps)
                        err = proof.accuracy_at(nt, ws, chunked,
                                                {"cfg": sh}, Nf, dt,
                                                tiers, n_ref)["cfg"][T_rep]
                        res[(nt, nr, K, band)] = dict(
                            cost=cost, med=float(np.median(err)),
                            worst=float(np.max(err)))
                        print(f"[sweep] Nt={nt} nr={nr} K={K} band={band}: "
                              f"{cost:7.2f} us  med={np.median(err):.3e} "
                              f"worst={np.max(err):.3e}")
                    except Exception as exc:              # noqa: BLE001
                        print(f"[sweep] Nt={nt} nr={nr} K={K} band={band} "
                              f"FAILED: {type(exc).__name__}: {exc}")
    all_nrs = sorted({k[1] for k in res})
    # persist every (setting -> cost, error) measurement + print the table
    rows = np.array([[nt, nr, K, band, r["cost"], r["med"], r["worst"]]
                     for (nt, nr, K, band), r in sorted(res.items())],
                    dtype=float)
    np.savez(os.path.join(out_dir, "settings_sweep.npz"), rows=rows,
             columns=np.array(["Nt", "nr", "K", "band", "cost_us",
                               "median_err", "worst_err"]),
             T_rep=np.array([T_rep]), batch=np.array([nb]))
    allowed = max(0.1, T_rep / 100.0)
    print(f"\n[SETTINGS] cost and error at T={T_rep:.0f} "
          f"(allowed {allowed:.2f}); PASS = worst <= allowed")
    print("   Tobs[yr]   nr    K  band     cost[us]     worst      median  ")
    for (nt, nr, K, band), r in sorted(res.items()):
        flag = "PASS" if r["worst"] <= allowed else "over"
        print(f"   {nt*1440*10/86400/365.25:7.2f} {nr:4d} {K:4d} "
              f"{('band' if band else 'pcr'):>5s} {r['cost']:10.2f} "
              f"{r['worst']:10.3e} {r['med']:10.3e}  {flag}")
    # cheapest PASSING configuration per Tobs -- the operational answer
    print("\n[RECOMMEND] cheapest configuration meeting the bar, per Tobs:")
    for nt in nts:
        cands = [((nr, K, band), r) for (n2, nr, K, band), r in res.items()
                 if n2 == nt and r["worst"] <= allowed]
        if cands:
            (nr, K, band), r = min(cands, key=lambda kv: kv[1]["cost"])
            print(f"   {nt*1440*10/86400/365.25:7.2f} yr: nr={nr} K={K} "
                  f"{'banded' if band else 'pcr'} -> {r['cost']:.2f} us "
                  f"(worst {r['worst']:.2e})")
        else:
            print(f"   {nt*1440*10/86400/365.25:7.2f} yr: NO configuration "
                  f"met the bar -- widen the ladder")
    make_figure(out_dir, nts, all_nrs, Ks, bands, T_rep, res, nb)


def make_figure(out_dir, nts, nrs, Ks, bands, T_rep, res, nb):
    ncol = len(nts)
    fig = plt.figure(figsize=(4.6 * ncol, 8.4))
    gs = fig.add_gridspec(2, ncol, hspace=0.32, wspace=0.24,
                          left=0.07, right=0.98, top=0.87, bottom=0.08)
    allowed = max(0.1, T_rep / 100.0)

    # --- top row: accuracy vs cost, one panel per Tobs -------------------
    for c, nt in enumerate(nts):
        ax = fig.add_subplot(gs[0, c]); proof.style(ax)
        for nr in nrs:
            for K in Ks:
                for band in bands:
                    r = res.get((nt, nr, K, band))
                    if r is None:
                        continue
                    ax.scatter([r["cost"]], [r["worst"]], s=95,
                               color=nr_colour(nr, nrs),
                               marker=K_MRK.get(K, "o"),
                               edgecolor="white" if band else INK2,
                               linewidth=1.6 if band else 1.0, zorder=3)
        ax.axhline(allowed, color="#2f7d4f", lw=1.6, ls="--")
        if c == 0:
            ax.annotate(f"allowed at T={T_rep:.0f}", (ax.get_xlim()[0],
                                                      allowed),
                        color="#2f7d4f", fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        tobs_yr = nt * 1440 * 10 / 86400.0 / 365.25
        ax.set_title(f"$T_{{obs}}$ = {tobs_yr:.2f} yr", color=INK,
                     fontsize=10, loc="left")
        ax.set_xlabel("per-candidate cost [$\\mu$s]")
        if c == 0:
            ax.set_ylabel(f"worst error at T={T_rep:.0f}")

    # --- bottom row: marginals -------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); proof.style(ax)
    for K in Ks:
        for band in bands:
            xs, ys = [], []
            for nr in nrs:
                r = res.get((nts[-1], nr, K, band))
                if r:
                    xs.append(nr); ys.append(r["worst"])
            if xs:
                ax.plot(xs, ys, marker=K_MRK.get(K, "o"), lw=1.8,
                        color="#08519c" if band else "#9ecae1",
                        ls="-" if band else "--",
                        label=f"K={K}, {'band' if band else 'pcr'}")
    ax.axhline(allowed, color="#2f7d4f", lw=1.6, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("fit nodes  $n_r$")
    ax.set_ylabel(f"worst error at T={T_rep:.0f}")
    ax.set_title("accuracy vs fit nodes (longest $T_{obs}$)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, ncol=2)

    if ncol > 1:
        ax = fig.add_subplot(gs[1, 1]); proof.style(ax)
        for nr in nrs:
            xs, ys = [], []
            for K in Ks:
                r = res.get((nts[-1], nr, K, bands[-1]))
                if r:
                    xs.append(K); ys.append(r["worst"])
            if xs:
                ax.plot(xs, ys, marker="o", lw=1.8,
                        color=nr_colour(nr, nrs), label=f"$n_r$={nr}")
        ax.axhline(allowed, color="#2f7d4f", lw=1.6, ls="--")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("fixed knots  K")
        ax.set_ylabel(f"worst error at T={T_rep:.0f}")
        ax.set_title("accuracy vs knots — knots are cheap", color=INK,
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    if ncol > 2:
        ax = fig.add_subplot(gs[1, 2]); proof.style(ax)
        for nr in nrs:
            xs, ys = [], []
            for nt in nts:
                r = res.get((nt, nr, Ks[len(Ks) // 2], bands[-1]))
                if r:
                    xs.append(nt * 1440 * 10 / 86400.0 / 365.25)
                    ys.append(r["cost"])
            if xs:
                ax.plot(xs, ys, marker="o", lw=1.8,
                        color=nr_colour(nr, nrs), label=f"$n_r$={nr}")
        ax.set_xlabel("observation time [yr]")
        ax.set_ylabel("per-candidate cost [$\\mu$s]")
        ax.set_title("cost is set by fit nodes, not $T_{obs}$", color=INK,
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    fig.suptitle("sig-het settings: what to run with",
                 fontsize=14, color=INK, x=0.07, ha="left", y=0.955)
    fig.text(0.07, 0.915,
             f"{BACKEND} · colour = fit nodes $n_r$ (light→dark), marker = "
             f"knots K, filled ring = banded / hollow = PCR solve · "
             f"batch={nb} · error vs chunked-het reference",
             fontsize=9, color=INK2, ha="left")
    fp = os.path.join(out_dir, "gb_settings_sweep.png")
    fig.savefig(fp, dpi=160, facecolor="white")
    print(f"\n[out] {fp}")


if __name__ == "__main__":
    main()
