"""SOBBH REMOVAL-NULL validation at 6 months (campaign gate S3 evidence).

Builds the stock 6-mo SOBBH-only synthetic fit (erebor.full_year_combined,
DATA_MODE=synthetic, no noise, likelihood source-only), puts every leaf at
TRUTH, and quantifies how well the CHUNKED engine fill path
(``fill_global_wdm``, the production SOBBH_LIKELIHOOD=chunked residual
generator) removes each source, plus how well the chunked scoring kernel
(``get_ll_wdm``) matches the slow tdionfly reference.

Measurements
------------
1. Production null: cold source-only lnL at truth after the engine's own
   ``setup_acs(rebuild_residuals=True)`` (chunked fill removal from data
   built with the EXACT tdionfly injection path).  July flip-book criterion:
   ~ -0.0000 when removal is exact.
2. Per-source removal residual r_i = h_exact_i - h_chunk_i (equivalent to
   removing source i alone from data containing only source i, since the
   synthetic data is the linear sum of exact tdionfly templates and no
   noise is added):  <r|r> (abs lnL units: deficit = -1/2 <r|r>) and
   <r|r>/<h|h> (fraction of SNR^2), plus the normalized match.
3. Knob sweeps on selected ids: SOBBH_NT_SUB x SOBBH_FILL_M_BAND_HALF_WIDTH
   removal-fraction table, and the scoring fast-vs-slow lnL match over
   SOBBH_M_BAND_HALF_WIDTH at truth + displaced points.

Laptop discipline: CPU-only, single-threaded BLAS (env pinned below before
numpy import), templates cached to --outdir so re-runs skip the slow
dense/tdionfly generation.

Run (from the LAT repo root)::

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /Users/mkatz/miniconda3/envs/deving/bin/python \
        scripts/sobbh/sobbh_removal_null_6mo.py --outdir /tmp/sobbh_null

All the stock-run env knobs (TOBS_TARGET, SOBHB_IDS, NWALKERS, ...) are
set as *defaults* below -- an exported env var still wins.
"""

from __future__ import annotations

import os

# ---- env FIRST (before numpy / lisatools imports) --------------------------
# Laptop CPU budget: pin every thread pool (sprint policy).
for _var in (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Stock 6-mo SOBBH-only synthetic fit (explicit export still wins).
_RUN_ENV_DEFAULTS = {
    "TOBS_TARGET": "15552000",       # 6 months (grid rounds to Nf*Nt*dt)
    "DATA_MODE": "synthetic",        # in-process data; no mojito load
    "SOBHB_IDS": "0,1,2,3,4,5",
    "EMRI_IDS": "",                  # env-empty -> zero leaves -> branch pruned
    "MBHB_IDS": "",
    "NWALKERS": "4",
    "SOBBH_START_FACTOR": "0",       # exact truth starting coords
    "MAKE_DIAGNOSTIC_PLOTS": "0",
}
for _k, _v in _RUN_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import argparse
import json
import resource
import time

import numpy as np

GSUN_SEC = 4.925490947641267e-06  # G*Msun/c^3 [s] (diagnostic kappa only)


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if os.uname().sysname == "Darwin" else ru / 1e3


class Timer:
    """Named stage timer collecting (name, seconds, rss_mb)."""

    def __init__(self):
        self.records = []

    def __call__(self, name):
        return _TimerCtx(self, name)


class _TimerCtx:
    def __init__(self, parent, name):
        self.parent, self.name = parent, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.parent.records.append((self.name, dt, rss_mb()))
        print(f"[timing] {self.name}: {dt:.1f} s (peak RSS {rss_mb():.0f} MB)",
              flush=True)
        return False


def sobbh_fdot(m1, m2, f):
    """Leading-order PN fdot [Hz/s] (diagnostic only -- kappa bookkeeping)."""
    mc = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    return (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * (
        mc * GSUN_SEC) ** (5.0 / 3.0) * f ** (11.0 / 3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="./sobbh_removal_null_6mo_out")
    ap.add_argument("--sweep-ids", default="0,3",
                    help="source ids for the knob sweeps")
    ap.add_argument("--nt-subs", default="8,16,32,64,128")
    ap.add_argument("--fill-widths", default="2,4,8")
    ap.add_argument("--score-widths", default="1,2,3,6")
    ap.add_argument("--budget-seconds", type=float, default=2400.0,
                    help="soft wall budget; slow displaced-point generation "
                         "is trimmed once exceeded")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    os.environ.setdefault("FILE_STORE_DIR", os.path.join(outdir, "gf_out/"))

    t_wall0 = time.perf_counter()
    timer = Timer()
    results = {"env": {k: os.environ.get(k) for k in _RUN_ENV_DEFAULTS},
               "outdir": outdir}
    out_json = os.path.join(outdir, "results.json")

    def dump():
        results["timings"] = [
            dict(stage=n, seconds=s, rss_mb=r) for (n, s, r) in timer.records
        ]
        results["total_wall_seconds"] = time.perf_counter() - t_wall0
        results["peak_rss_mb"] = rss_mb()
        with open(out_json, "w") as f:
            json.dump(results, f, indent=1)

    # ---- 1. build the stock fit + truth state + acs (production path) ------
    with timer("import lisatools + erebor"):
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.source_runtime import (
            find_source_cfg,
            get_sobbh_chunked_comp,
            get_sobbh_wave_wrap,
        )
        from lisatools.globalfit.moves.sobbhspecialmove import (
            SOBBHChunkedLikeMove,
        )
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product
        from lisatools.domains import WDMSignal
        from lisatools.response.tdiconfig import TDIConfig
        from bbhx.sobbhcomps import SOBBHWDMComputations

    with timer("fit.build() (synthetic data: exact tdionfly injections)"):
        fit = erebor.full_year_combined()
        assert fit.general.data_mode == "synthetic"
        assert fit.general.add_instrument_noise is False
        assert fit.general.likelihood_source_only is True
        fit.build()

    general_info = fit.general_info
    wdm = general_info.domain_settings
    cfg = find_source_cfg(fit)
    if cfg is None:
        raise RuntimeError("no SourceSignalGen found post-build")
    assert cfg.get("sobbh_likelihood") == "chunked", cfg.get("sobbh_likelihood")

    nch = int(cfg["nchannels"])
    Nfa, Nta = int(wdm.Nf_active), int(wdm.Nt_active)
    layer_df = float(wdm.layer_df)
    layer_dt = float(wdm.layer_dt)
    grid_info = dict(
        Nf=int(wdm.Nf), Nt=int(wdm.Nt), Nf_active=Nfa, Nt_active=Nta,
        dt=float(general_info.dt), Tobs=float(general_info.Tobs),
        data_t0=float(general_info.data_t0),
        layer_df=layer_df, layer_dt=layer_dt,
        nt_sub_default=int(cfg["sobbh_nt_sub"]),
        n_pad=int(cfg["sobbh_n_pad"]), n_sparse=int(cfg["sobbh_n_sparse"]),
        fill_m_default=int(cfg["sobbh_fill_m_band_half_width"]),
        score_m_default=int(cfg["sobbh_m_band_half_width"]),
    )
    results["grid"] = grid_info
    print("[grid]", json.dumps(grid_info), flush=True)

    with timer("GlobalFit + load_info(truth) + setup_acs(rebuild)"):
        gf = GlobalFit(fit, comm=None)
        priors = {}
        for name in fit.engine_info.branch_names:
            priors.update(fit.source_info[name].priors)
        state = gf.load_info(priors)
        acs = gf.setup_acs(state, rebuild_residuals=True)

    lnl_truth = np.asarray(acs.likelihood(complex=False)).real
    results["lnl_truth_production"] = [float(x) for x in np.ravel(lnl_truth)]
    print(f"[null] production cold source-only lnL at truth (per walker): "
          f"{np.ravel(lnl_truth)}", flush=True)
    dump()

    ac0 = acs.flatten()[0]
    sens = ac0.sens_mat            # shared read-only for scratch containers

    # waveform-basis truth rows via the branch transform (roundtrip check
    # against the deterministic injection maker)
    tf = fit.source_info["sobbh"].transform
    inj_sampling = np.asarray(fit.source_info["sobbh"].injection, dtype=float)
    wf_rows = np.stack(
        [tf.both_transforms(row) for row in inj_sampling], axis=0
    )
    from lisatools.globalfit.stock.erebor.injections import (
        make_sobbh_injections,
    )
    wf_direct = make_sobbh_injections(wf_rows.shape[0], mode="stock")
    if not np.allclose(wf_rows, wf_direct, rtol=1e-12, atol=1e-12):
        print("[warn] transform-roundtrip rows differ from "
              "make_sobbh_injections(stock); using the transform rows.",
              flush=True)
    n_src = wf_rows.shape[0]
    p_chunk = SOBBHChunkedLikeMove.to_chunked_basis(wf_rows)  # (n, 11)

    # per-source diagnostic sweep rates
    fdots = sobbh_fdot(wf_rows[:, 0], wf_rows[:, 1], wf_rows[:, 6])
    results["sources"] = [
        dict(id=i, f_low=float(wf_rows[i, 6]), m1=float(wf_rows[i, 0]),
             m2=float(wf_rows[i, 1]), dist_gpc=float(wf_rows[i, 4]),
             fdot=float(fdots[i]),
             layers_per_6mo=float(fdots[i] * general_info.Tobs / layer_df))
        for i in range(n_src)
    ]

    # ---- 2. exact (slow tdionfly) per-source WDM templates -----------------
    slow_wrap = get_sobbh_wave_wrap(general_info, cfg)
    h_exact = []
    slow_secs = []
    for i in range(n_src):
        cache = os.path.join(outdir, f"h_exact_src{i}.npy")
        if os.path.exists(cache):
            arr = np.load(cache)
            print(f"[cache] exact template src {i} loaded", flush=True)
            slow_secs.append(0.0)
        else:
            with timer(f"exact tdionfly template src {i}") as _t:
                arr = np.asarray(slow_wrap(*wf_rows[i]).arr, dtype=float)
            np.save(cache, arr)
            slow_secs.append(timer.records[-1][1])
        h_exact.append(arr)
    results["slow_template_seconds"] = slow_secs
    dump()

    def wsig(arr):
        return WDMSignal(np.ascontiguousarray(arr), wdm)

    def ip(a, b):
        """<a|b> with the run's fixed sensitivity (installed diagnostic)."""
        return float(np.real(inner_product(wsig(a), wsig(b), psd=sens,
                                           complex=False)))

    # data-vs-sum-of-exact-templates null (injection vs PE convention pin)
    data_arr = np.asarray(general_info.input_data_residual_array.arr,
                          dtype=float)
    h_sum = np.sum(h_exact, axis=0)
    d_d = ip(data_arr, data_arr)
    s_s = ip(h_sum, h_sum)
    d_s = ip(data_arr, h_sum)
    resid_ds = ip(data_arr - h_sum, data_arr - h_sum)
    results["data_vs_exact_sum"] = dict(
        d_d=d_d, h_h=s_s, match=d_s / np.sqrt(d_d * s_s),
        resid_over_dd=resid_ds / d_d)
    print(f"[pin] data vs sum(exact templates): match="
          f"{d_s / np.sqrt(d_d * s_s):+.9f}  <r|r>/<d|d>={resid_ds / d_d:.3e}",
          flush=True)

    # ---- 3. per-source removal residual at production defaults -------------
    comp_default = get_sobbh_chunked_comp(general_info, cfg)
    fill_m_default = int(cfg["sobbh_fill_m_band_half_width"])

    def chunk_fill(comp, row_idx, m_fill):
        buf = np.zeros((nch, Nfa, Nta), dtype=float)
        comp.fill_global_wdm(p_chunk[row_idx][None, :], buf,
                             convert_to_ra_dec=False,
                             m_band_half_width=int(m_fill))
        return buf

    hh_exact = [ip(h, h) for h in h_exact]

    def removal_row(comp, i, m_fill):
        hc = chunk_fill(comp, i, m_fill)
        r = h_exact[i] - hc
        rr = ip(r, r)
        hchc = ip(hc, hc)
        ehc = ip(h_exact[i], hc)
        return dict(
            id=i, rr=rr, hh=hh_exact[i], frac=rr / hh_exact[i],
            lnl_deficit=-0.5 * rr,
            match=ehc / np.sqrt(hh_exact[i] * hchc),
        )

    with timer("per-source removal residuals (defaults 32/8)"):
        removal_default = [removal_row(comp_default, i, fill_m_default)
                           for i in range(n_src)]
    results["removal_default"] = removal_default
    print("\n== per-source removal residual, production defaults "
          f"(Nt_sub={grid_info['nt_sub_default']}, "
          f"fill_m={fill_m_default}) ==")
    print(f"{'id':>3} {'SNR':>8} {'<r|r>':>12} {'<r|r>/<h|h>':>12} "
          f"{'lnL deficit':>12} {'match':>12}")
    for row in removal_default:
        print(f"{row['id']:>3} {np.sqrt(row['hh']):>8.2f} {row['rr']:>12.4e} "
              f"{row['frac']:>12.4e} {row['lnl_deficit']:>12.4e} "
              f"{row['match']:>12.9f}")
    tot_rr = sum(r["rr"] for r in removal_default)
    print(f"[null] sum_i -0.5<r_i|r_i> = {-0.5 * tot_rr:.4e} "
          f"(production lnL at truth = {float(np.ravel(lnl_truth)[0]):.4e}; "
          "difference = cross terms)", flush=True)
    dump()

    # ---- 4. knob sweep: Nt_sub x fill width (removal) ----------------------
    sweep_ids = [int(x) for x in args.sweep_ids.split(",") if x.strip()]
    nt_subs = [int(x) for x in args.nt_subs.split(",") if x.strip()]
    fill_widths = [int(x) for x in args.fill_widths.split(",") if x.strip()]

    t_ref = cfg["sobbh_reference_time"]
    t_ref = float(general_info.data_t0 if t_ref is None else t_ref)
    tdi_config = TDIConfig(cfg["tdi_gen_str"],
                           force_backend=general_info.force_backend)

    comps = {}

    def get_comp(nt_sub):
        if nt_sub in comps:
            return comps[nt_sub]
        n_pad = int(cfg["sobbh_n_pad"])
        if nt_sub - 2 * n_pad <= 0:      # keep-step must stay positive/even
            n_pad = max(1, nt_sub // 4)
        comp = SOBBHWDMComputations(
            wdm, t_ref=t_ref, Nt_sub=int(nt_sub), n_pad=n_pad,
            N_sparse=int(cfg["sobbh_n_sparse"]),
            orbits=general_info.orbits, tdi_config=tdi_config,
            tdi_type=cfg["tdi_chan"], d_d=0.0,
            force_backend=general_info.force_backend,
            t_obs_start=float(general_info.data_t0),
        )
        comp._used_n_pad = n_pad
        comps[nt_sub] = comp
        return comp

    sweep_rows = []
    with timer("removal knob sweep (Nt_sub x fill width)"):
        for i in sweep_ids:
            for nt_sub in nt_subs:
                comp = get_comp(nt_sub)
                kappa = float(fdots[i] * (nt_sub * layer_dt) / layer_df)
                for mf in fill_widths:
                    t0 = time.perf_counter()
                    row = removal_row(comp, i, mf)
                    row.update(nt_sub=nt_sub, fill_m=mf, kappa=kappa,
                               n_pad=comp._used_n_pad,
                               fill_seconds=time.perf_counter() - t0)
                    sweep_rows.append(row)
    results["removal_sweep"] = sweep_rows
    dump()

    print("\n== removal sweep:  residual fraction <r|r>/<h|h>  "
          "(kappa = intra-chunk sweep in layers) ==")
    for i in sweep_ids:
        print(f"-- source id {i} (f_low={wf_rows[i, 6] * 1e3:.3f} mHz, "
              f"SNR={np.sqrt(hh_exact[i]):.1f}) --")
        hdr = f"{'Nt_sub':>7} {'kappa':>8} " + " ".join(
            f"{'fill_m=' + str(m):>12}" for m in fill_widths)
        print(hdr + f" {'match(m=max)':>13}")
        for nt_sub in nt_subs:
            rr = [r for r in sweep_rows
                  if r["id"] == i and r["nt_sub"] == nt_sub]
            rr.sort(key=lambda r: r["fill_m"])
            line = f"{nt_sub:>7} {rr[0]['kappa']:>8.3f} " + " ".join(
                f"{r['frac']:>12.4e}" for r in rr)
            print(line + f" {rr[-1]['match']:>13.9f}")

    # ---- 5. scoring match: get_ll (fast) vs slow tdionfly reference --------
    score_widths = [int(x) for x in args.score_widths.split(",") if x.strip()]

    def displaced_points(i):
        # dist x1.2 is FREE on the slow side (h scales as 1/dL exactly), so
        # only two displaced points cost a slow tdionfly call each.
        base = wf_rows[i]
        pts = [("truth", base.copy())]
        p = base.copy(); p[4] *= 1.2
        pts.append(("dist x1.2", p))
        p = base.copy(); p[6] += 0.25 * layer_df
        pts.append(("f_low +0.25 layer", p))
        p = base.copy(); p[10] = (p[10] + 0.5) % (2 * np.pi)
        pts.append(("phi0 +0.5", p))
        return pts

    scoring = []
    for i in sweep_ids:
        d_arr = h_exact[i]                       # single-source dataset
        ac_i = AnalysisContainer(wsig(d_arr.copy()), sens)
        holder = comp_default._as_wdm_holder(ac_i)
        pts = displaced_points(i)
        for label, wrow in pts:
            over_budget = (time.perf_counter() - t_wall0) > args.budget_seconds
            cache = os.path.join(
                outdir,
                f"h_slow_src{i}_{label.replace(' ', '_').replace('+', 'p').replace('.', '')}.npy",
            )
            if label == "truth":
                h_slow = d_arr
            elif label == "dist x1.2":
                h_slow = d_arr / 1.2      # exact amplitude scaling, free
            elif os.path.exists(cache):
                h_slow = np.load(cache)
            elif over_budget:
                print(f"[budget] skipping slow template for src {i} "
                      f"'{label}' (wall budget exceeded)", flush=True)
                continue
            else:
                with timer(f"slow displaced template src {i} '{label}'"):
                    h_slow = np.asarray(slow_wrap(*wrow).arr, dtype=float)
                np.save(cache, h_slow)
            d_h_s = ip(d_arr, h_slow)
            h_h_s = ip(h_slow, h_slow)
            ll_slow = d_h_s - 0.5 * h_h_s
            prow = SOBBHChunkedLikeMove.to_chunked_basis(wrow[None, :])
            for m in score_widths:
                ll_fast = float(np.asarray(comp_default.get_ll_wdm(
                    prow, holder,
                    data_index=np.zeros(1, dtype=np.int32),
                    noise_index=np.zeros(1, dtype=np.int32),
                    m_band_half_width=int(m),
                ))[0])
                d_h_f = float(np.real(np.asarray(comp_default.d_h_out))[0])
                h_h_f = float(np.real(np.asarray(comp_default.h_h_out))[0])
                scoring.append(dict(
                    id=i, point=label, m=m,
                    ll_fast=ll_fast, ll_slow=ll_slow,
                    dll=ll_fast - ll_slow,
                    d_h_fast=d_h_f, h_h_fast=h_h_f,
                    d_h_slow=d_h_s, h_h_slow=h_h_s,
                ))
    results["scoring"] = scoring

    print("\n== scoring match: fast (chunked get_ll, Nt_sub="
          f"{grid_info['nt_sub_default']}) vs slow tdionfly, "
          "ll = <d|h> - 1/2<h|h> ==")
    for i in sweep_ids:
        rows_i = [r for r in scoring if r["id"] == i]
        pts_seen = []
        for r in rows_i:
            if r["point"] not in pts_seen:
                pts_seen.append(r["point"])
        print(f"-- source id {i} --")
        print(f"{'point':>18} {'ll_slow':>14} " + " ".join(
            f"{'dll m=' + str(m):>12}" for m in score_widths))
        for pt in pts_seen:
            rr = {r["m"]: r for r in rows_i if r["point"] == pt}
            anyr = rr[score_widths[0]]
            print(f"{pt:>18} {anyr['ll_slow']:>14.4f} " + " ".join(
                f"{rr[m]['dll']:>12.4e}" if m in rr else f"{'--':>12}"
                for m in score_widths))

    # ---- wrap up ------------------------------------------------------------
    dump()
    print(f"\n[done] wall {results['total_wall_seconds']:.0f} s, "
          f"peak RSS {results['peak_rss_mb']:.0f} MB -> {out_json}",
          flush=True)


if __name__ == "__main__":
    main()
