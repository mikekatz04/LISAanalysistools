#!/usr/bin/env python
"""Time-frequency localization of the MBH null residual, INSIDE the stock fit.

Answers: *where in the waveform* (inspiral / early-low-freq vs merger-ringdown /
late-high-freq) does the stock MBH template fail to null the mojito data, and how
does that differ between the **legacy** phentax response and **TDI-on-the-fly**?

`erebor.full_year_combined`'s analysis domain is WDM (time x frequency wavelets),
so the residual is natively localized: the noise-weighted inner product that
drives the mismatch, `<r|r> = 4 dphi Sum_ij Re[r_i* r_j invC_ij]`, is a per-(f,t)
element sum (diagnostic.inner_product). We build the residual through the stock
objects (erebor -> setup_acs data -> signal_gen template), form the per-element
contribution `g_ft`, VALIDATE `Sum g_ft == <r|r>` (the null-check number), and
collapse it to residual-vs-time and residual-vs-frequency.

Two modes:

  extract:  MBHB_ID=<id> RESPONSE=legacy|tof  python mbh_td_residual.py extract
            -> builds one config, saves <outdir>/td_<id>_<resp>.npz (summaries only)
  analyze:  python mbh_td_residual.py analyze --id <id> [--fcut 5e-4]
            -> loads the two npz, plots the comparison, prints the inspiral/merger
               and <fcut split for legacy vs TOF.

Heavy build (~1-2 min + swap on an 8GB laptop); run one config at a time.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import threading
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
OUTDIR = os.environ.get("CAMPAIGN_PLOT_DIR", "/tmp/mbh_td")


def _watchdog(limit_gb=float(os.environ.get("NULL_CHECK_MEM_GB", "20"))):
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > limit_gb:
            print(f"[watchdog] RSS over {limit_gb} GB; aborting", flush=True)
            os._exit(42)
        time.sleep(0.5)


# ----------------------------------------------------------------------------- extract
def extract():
    threading.Thread(target=_watchdog, daemon=True).start()
    mbh_id = int(os.environ["MBHB_ID"])
    resp = os.environ.get("RESPONSE", "legacy").lower()
    use_tof = resp in ("tof", "tdionfly", "1", "true")

    # Stock knobs, same resolution as the production runs / mojito_null_check.
    os.environ["MBHB_IDS"] = str(mbh_id)
    os.environ.setdefault("SOBHB_IDS", "")
    os.environ.setdefault("EMRI_IDS", "")
    os.environ.setdefault("CHOP_WINDOW", "1")
    os.environ["USE_TDIONFLY"] = "1" if use_tof else "0"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MAKE_PLOTS", "0")
    os.environ.setdefault("NWALKERS", "1")
    os.environ.setdefault("NTEMPS", "1")
    os.environ.setdefault(
        "MOJITO_DATA_PATH",
        "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/")

    from mpi4py import MPI
    from eryn.state import BranchSupplemental
    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor

    fit = erebor.full_year_combined(nwalkers=1, ntemps=1)
    gs = fit.general
    print(f"[cfg] id={mbh_id} response={'TOF' if use_tof else 'legacy'} "
          f"use_tdionfly={use_tof} dt={gs.dt} chop={gs.chop_window}", flush=True)

    curr = fit.build()
    gi = curr.general_info
    branch = [n for n in curr.source_info if n not in ("psd", "galfor")][0]
    setup = curr.source_info[branch]
    print(f"[branch] {branch} signal_gen={type(setup.signal_gen).__name__} "
          f"Tobs={gi.Tobs:.4e}s ({gi.Tobs/86400:.2f}d)", flush=True)

    comm = MPI.COMM_WORLD
    if os.path.exists(gi.main_file_path):
        os.remove(gi.main_file_path)
    gf = GlobalFit(curr, comm)
    priors = {}
    for name in gf.engine_info.branch_names:
        si = curr.source_info.get(name)
        p = getattr(si, "priors", None) if si is not None else None
        if p:
            priors.update(p)
    state = gf.load_info(priors)
    ntemps, nwalkers = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers), copy=True)
    inj = np.asarray(setup.injection, dtype=float)
    if inj.ndim == 1:
        inj = inj[None, :]
    state.branches_coords[branch][:] = inj[None, None]

    # data-only AC + template at injection (the null-check primary path)
    ac = gf.setup_acs(state, rebuild_residuals=False).flatten()[0]
    d_d = float(np.ravel(np.asarray(ac.inner_product(complex=False)).real)[0])
    _t = time.time()
    h = setup.signal_gen(*inj[0], leaf_inds=0)
    print(f"[template] built in {time.time()-_t:.1f}s", flush=True)
    opt, det = ac.template_snr(h)
    h_h = float(opt) ** 2
    d_h = complex(ac.non_marg_d_h).real
    r_r = d_d + h_h - 2.0 * d_h
    mm = 1.0 - d_h / np.sqrt(d_d * h_h)
    print(f"[inner] dd={d_d:.6e} hh={h_h:.6e} dh={d_h:.6e} rr={r_r:.6e} "
          f"mm={mm:.3e}  (SNR {np.sqrt(d_d):.1f})", flush=True)

    # WDM coefficient arrays (nch, Nf, Nt) and the noise inverse-covariance.
    d_arr = np.asarray(ac.data_res_arr.arr)
    h_arr = np.asarray(h.arr)
    while d_arr.ndim > 3:            # squeeze any (nbatch,) leading dim
        d_arr = d_arr[0]
    while h_arr.ndim > 3:
        h_arr = h_arr[0]
    r_arr = d_arr - h_arr                            # (nch, Nf, Nt)
    sens = ac.sens_mat
    invC = np.asarray(sens.invC)                     # (nch, nch, Nf, Nt)
    dphi = float(sens.differential_component)
    nch = r_arr.shape[0]
    Nf, Nt = r_arr.shape[-2], r_arr.shape[-1]
    print(f"[wdm] arr shape={r_arr.shape} invC shape={invC.shape} "
          f"dphi={dphi:.6e} channel_shape={sens.channel_shape}", flush=True)

    # per-(f,t) contribution to <r|r>: 4 dphi Re[sum_ij r_i* r_j invC_ij]
    g_ft = np.zeros((Nf, Nt), dtype=np.float64)
    if invC.ndim == 4:              # (nch,nch,Nf,Nt) full cross-channel
        for i in range(nch):
            for j in range(nch):
                g_ft += np.real(np.conj(r_arr[i]) * r_arr[j]
                                * np.nan_to_num(invC[i, j]))
    else:                           # (nch,Nf,Nt) diagonal
        for i in range(nch):
            g_ft += np.real(np.conj(r_arr[i]) * r_arr[i]
                            * np.nan_to_num(invC[i]))
    g_ft *= 4.0 * dphi
    rr_check = float(g_ft.sum())
    print(f"[validate] sum(g_ft)={rr_check:.6e} vs <r|r>={r_r:.6e}  "
          f"ratio={rr_check/r_r:.5f}", flush=True)

    # data power per time layer (unweighted) -> locate merger; residual maps.
    dpow_t = np.sum(np.abs(d_arr) ** 2, axis=(0, 1))          # (Nt,)
    dpow_f = np.sum(np.abs(d_arr) ** 2, axis=(0, 2))          # (Nf,)
    rawres_t = np.sum(np.abs(r_arr) ** 2, axis=(0, 1))        # (Nt,) unweighted
    rr_time = g_ft.sum(axis=0)                                # (Nt,) weighted
    rr_freq = g_ft.sum(axis=1)                                # (Nf,) weighted

    # physical layer axes: freq-layer centers, time-layer centers.
    st = ac.data_res_arr.settings
    layer_df = float(getattr(st, "layer_df", np.nan))
    wav_dur = float(getattr(st, "layer_dt", getattr(st, "wavelet_duration",
                    Nf * gs.dt)))
    # active freq-layer indices (min_freq/max_freq crop) if exposed
    fmask = getattr(st, "frequency_layer_mask", None)
    if fmask is not None and int(np.sum(np.asarray(fmask))) == Nf:
        f_idx = np.where(np.asarray(fmask))[0]
    else:
        f_idx = np.arange(Nf)
    f_centers = f_idx * layer_df
    t_centers = np.arange(Nt) * wav_dur

    # downsample g_ft for a compact 2D map (block-sum)
    def _block(a2d, nf=160, nt=220):
        Ff = max(1, a2d.shape[0] // nf)
        Tt = max(1, a2d.shape[1] // nt)
        f2 = (a2d.shape[0] // Ff) * Ff
        t2 = (a2d.shape[1] // Tt) * Tt
        return (a2d[:f2, :t2].reshape(f2 // Ff, Ff, t2 // Tt, Tt)
                .sum(axis=(1, 3)),
                f_centers[:f2:Ff], t_centers[:t2:Tt])
    g_map, f_ax, t_ax = _block(g_ft)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"td_{mbh_id}_{'tof' if use_tof else 'legacy'}.npz")
    np.savez_compressed(
        out, id=mbh_id, response=("tof" if use_tof else "legacy"),
        dd=d_d, hh=h_h, dh=d_h, rr=r_r, rr_check=rr_check, mm=mm,
        snr=np.sqrt(d_d), Tobs=gi.Tobs, dt=gs.dt, Nf=Nf, Nt=Nt,
        layer_df=layer_df, wav_dur=wav_dur,
        rr_time=rr_time, rr_freq=rr_freq, dpow_t=dpow_t, dpow_f=dpow_f,
        rawres_t=rawres_t, f_centers=f_centers, t_centers=t_centers,
        g_map=g_map, f_ax=f_ax, t_ax=t_ax)
    print(f"[RESULT] extract_ok=1 id={mbh_id} "
          f"resp={'tof' if use_tof else 'legacy'} mm={mm:.3e} rr={r_r:.6e} "
          f"validate_ratio={rr_check/r_r:.5f} out={out}", flush=True)


# ----------------------------------------------------------------------------- analyze
def analyze():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--fcut", type=float, default=5e-4)
    ap.add_argument("--dir", default=OUTDIR)
    args, _ = ap.parse_known_args(sys.argv[2:])
    import matplotlib.pyplot as plt

    def _load(resp):
        p = os.path.join(args.dir, f"td_{args.id}_{resp}.npz")
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    L, T = _load("legacy"), _load("tof")
    have = [(n, d) for n, d in (("legacy", L), ("TOF", T)) if d is not None]
    if not have:
        print("[RESULT] analyze_ok=0 reason=no_npz", flush=True)
        sys.exit(1)

    # merger time layer = peak of data power (same data both configs)
    ref = have[0][1]
    merg = int(np.argmax(ref["dpow_t"]))
    t_c = ref["t_centers"]
    merg_t = t_c[merg]

    def _split(d):
        rr_t = d["rr_time"]
        insp = float(rr_t[:merg].sum())
        mrg = float(rr_t[merg:].sum())
        rr_f = d["rr_freq"]
        fc = d["f_centers"]
        below = float(rr_f[fc < args.fcut].sum())
        return insp, mrg, below, float(rr_t.sum())

    print(f"\n  MBH id={args.id}   merger at t-layer {merg} "
          f"(t={merg_t/86400:.2f} d of {t_c[-1]/86400:.2f} d)   "
          f"fcut={args.fcut:.1e} Hz")
    print(f"  {'config':>7} {'mm':>10} {'<r|r>':>11} {'inspiral':>11} "
          f"{'merger+rd':>11} {'insp %':>7} {'<fcut %':>8}")
    for name, d in have:
        insp, mrg, below, tot = _split(d)
        print(f"  {name:>7} {float(d['mm']):>10.3e} {tot:>11.3e} "
              f"{insp:>11.3e} {mrg:>11.3e} {100*insp/tot:>6.1f}% "
              f"{100*below/tot:>7.1f}%")

    # ---- figure: per-config TF map (top row) + shared time/freq collapses ----
    ncfg = len(have)
    fig = plt.figure(figsize=(13, 4.2 + 3.0))
    gs_ = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42,
                           wspace=0.22)
    # top: TF residual maps
    tf_ax = [fig.add_subplot(gs_[0, k]) for k in range(2)]
    import matplotlib.colors as mc
    vmax = max(float(np.nanmax(d["g_map"])) for _, d in have)
    vmin = vmax * 1e-6
    for k in range(2):
        ax = tf_ax[k]
        if k < ncfg:
            name, d = have[k]
            gmap = np.clip(d["g_map"], vmin, None)
            im = ax.pcolormesh(d["t_ax"] / 86400, d["f_ax"] * 1e3, gmap,
                               norm=mc.LogNorm(vmin=vmin, vmax=vmax),
                               shading="auto", cmap="magma")
            ax.axhline(args.fcut * 1e3, color="#39d6ff", lw=1.0, ls="--")
            ax.axvline(merg_t / 86400, color="w", lw=0.9, ls=":")
            ax.set_title(f"{name}: TF residual  |g(f,t)|  (mm {float(d['mm']):.1e})",
                         fontsize=9.5)
            ax.set_xlabel("time [day]"); ax.set_ylabel("freq [mHz]")
            ax.set_yscale("log")
            fig.colorbar(im, ax=ax, pad=0.01, fraction=0.045)
        else:
            ax.axis("off")
    # bottom-left: residual vs time
    axt = fig.add_subplot(gs_[1, 0])
    for name, d in have:
        c = "#d62728" if name == "legacy" else "#1f77b4"
        axt.plot(d["t_centers"] / 86400, np.clip(d["rr_time"], 1e-12, None),
                 color=c, lw=1.6, label=f"{name}  <r|r>={float(d['rr']):.2e}")
    axt.axvline(merg_t / 86400, color="k", lw=1.0, ls=":", label="merger")
    # data-power envelope (scaled) to show inspiral->merger
    dp = ref["dpow_t"] / ref["dpow_t"].max()
    axt.plot(t_c / 86400, dp * axt.get_ylim()[1], color="#888", lw=0.9,
             alpha=0.5, label="data power (scaled)")
    axt.set_yscale("log"); axt.set_xlabel("time [day]")
    axt.set_ylabel(r"$<r|r>$ per time-layer")
    axt.set_title("residual vs time (inspiral -> merger/ringdown)", fontsize=9.5)
    axt.legend(frameon=False, fontsize=7.5); axt.grid(alpha=0.15, which="both")
    # bottom-right: residual vs frequency
    axf = fig.add_subplot(gs_[1, 1])
    for name, d in have:
        c = "#d62728" if name == "legacy" else "#1f77b4"
        fc = d["f_centers"]
        good = fc > 0
        axf.plot(fc[good] * 1e3, np.clip(d["rr_freq"][good], 1e-12, None),
                 color=c, lw=1.6, label=name)
    axf.axvline(args.fcut * 1e3, color="#0a8", lw=1.0, ls="--",
                label=f"{args.fcut:.0e} Hz")
    axf.set_xscale("log"); axf.set_yscale("log")
    axf.set_xlabel("freq [mHz]"); axf.set_ylabel(r"$<r|r>$ per freq-layer")
    axf.set_title("residual vs frequency (excess < cut?)", fontsize=9.5)
    axf.legend(frameon=False, fontsize=7.5); axf.grid(alpha=0.15, which="both")

    fig.suptitle(
        f"MBH id={args.id} null residual localized in the WDM domain "
        f"(legacy vs TDI-on-the-fly)", fontsize=12, y=0.99)
    os.makedirs(args.dir, exist_ok=True)
    out = os.path.join(args.dir, f"mbh_td_residual_id{args.id}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[RESULT] analyze_ok=1 id={args.id} plot={out}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if mode == "analyze":
        analyze()
    else:
        try:
            extract()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[RESULT] extract_ok=0 error={type(exc).__name__}", flush=True)
            sys.exit(1)
