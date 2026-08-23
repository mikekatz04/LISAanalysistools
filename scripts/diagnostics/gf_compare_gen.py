#!/usr/bin/env python
"""Three-run GF comparison page: v4 / v5 / v6 on shared axes.

    python scripts/diagnostics/gf_compare_gen.py \
        <v4_h5_or_dir> <v5_dir> <v6_dir> <out.html> [--truth gb_truth_3to21.npz]

One page, sibling of the gf_monitor_gen pages (same tokens, same
one-meaning-per-hue palette, per-run arm colours):

* overplotted trajectories (cold logL max, delta vs v4 at matched
  iteration, leaves/walker) — one line per run on ONE axis;
* source counts vs frequency at each run's latest iteration, overstepped;
* THE STACKED ZOOM EXPLORER: three flush canvases (no vertical gap), one
  per run, IDENTICAL margins/axes and a single shared view-state — every
  pan / wheel-zoom on any panel applies to all three, so a zoom-in is a
  direct like-for-like comparison. Catalogue truths under each cloud.

Rerunnable on every snapshot (make_snapshots.sh zips / *_extract.h5
reduced stores are handled: the live h5 in a dir is the newest
*testing*.h5 that is not a backup/CORRUPT copy, extracts preferred last).
"""
import argparse
import base64
import io
import json
import os
import sys
import warnings

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_META = [
    ("v4", "#58C48A"),
    ("v5", "#FF7BAC"),
    ("v6", "#9B7BFF"),
    ("1yr", "#F2E14C"),   # optional 4th run; different Tobs -- shown for
                          # contrast, excluded from matched-iteration deltas
]
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
TRUTHRED = "#FF2E3E"
TG = 4.925490947641267e-6

# flagship: the highest-frequency catalogue source in the 3-21 mHz set
FLAG_F0, FLAG_TOL, FLAG_FDOT = 20.380377e-3, 5.144e-6, 1.0245e-13
PSD_NAMES = ["Soms_d  [m]", "Sa_a  [m/s$^2$]"]
PSD_INJ = [1.496182e-11, 2.982412e-15]
GAL_NAMES = ["log10 amp", "p1", "log10 fknee", "p2", "slope"]
MOJITO_CAT_DIR = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0")


def live_h5(path):
    if os.path.isfile(path):
        return path
    cands = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith(".h5") and "testing" in f
             and "backup" not in f and "CORRUPT" not in f]
    if not cands:
        raise SystemExit(f"no live h5 in {path}")
    # prefer an _extract if it is the freshest artefact of its store
    cands.sort(key=lambda p: (os.path.getmtime(p), p.endswith("_extract.h5")))
    return cands[-1]


def _trim(a):
    """Drop the preallocated all-zero tail of a stored (it, ...) array.

    The h5 stores are preallocated to the *requested* iteration count, so a
    naive ``[:it]`` slice can still end on rows the sub-backend never
    flushed (galfor lags psd by one row in v5, for example).  Scan backwards
    for the last row that is finite and not identically zero.
    """
    a = np.asarray(a)
    for r in range(a.shape[0] - 1, -1, -1):
        v = a[r]
        if np.isfinite(v).any() and np.abs(np.nan_to_num(v)).sum() > 0:
            return a[:r + 1]
    return a[:0]


def load_run(path):
    """Series + latest-iteration cold points for one run."""
    from lisatools.globalfit.stock.erebor.transforms import (
        make_gb_transform_container)
    tc = make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True)
    extra = {}
    with h5py.File(path, "r") as f:
        g = f["global_fit"]
        it = int(g.attrs["iteration"])
        ll = np.asarray(g["log_like"][:it])
        while ll.ndim > 3:
            ll = ll[:, 0]
        cold = ll[:, 0]                     # (it, nwalkers)
        inds = g["inds/gb"]
        counts = np.empty(it)
        for r in range(it):                 # row reads: extract-safe
            a = np.asarray(inds[r])
            while a.ndim > 3:
                a = a[:, 0]
            counts[r] = a[0].sum(-1).mean()
        ch = np.asarray(g["chain/gb"][it - 1])
        ii = np.asarray(inds[it - 1])
        while ch.ndim > 4:
            ch = ch[0]
        while ii.ndim > 3:
            ii = ii[0]
        rows = ch[0][ii[0]]                 # pooled cold leaves, (n, 9)

        # ---- small branches: full history, cold chain -------------------
        # chain/psd (it,1,1,nw,1,2) and chain/galfor (it,1,1,nw,1,5) are
        # written every iteration in all four stores.
        for key, name in (("chain/psd", "psd"), ("chain/galfor", "galfor")):
            try:
                a = np.asarray(g[key][:it])
                while a.ndim > 4:           # -> (it, nw, nleaf, ndim)
                    a = a[:, 0]
                extra[name] = _trim(a[:, :, 0])      # (it, nw, ndim)
            except Exception as e:          # pragma: no cover - store shape
                extra[name] = None
                extra[name + "_why"] = f"{type(e).__name__}: {e}"
        # chain/vgb (it,1,1,nw,55,5) -- full history for v4/v5/1yr, but a
        # keep-window only for v6, so take the LAST written row only.
        try:
            a = np.asarray(g["chain/vgb"][it - 1])
            while a.ndim > 3:
                a = a[0]
            extra["vgb"] = a                          # (nw, 55, 5)
        except Exception as e:              # pragma: no cover
            extra["vgb"] = None
            extra["vgb_why"] = f"{type(e).__name__}: {e}"
        # ---- leaf-cap tracker: full history, per band -------------------
        try:
            caps = _trim(np.asarray(g["sub_backend/gb/band_leaf_cap"][:it]))
            m = np.where(caps >= 0, caps, np.nan)     # -1 = not yet armed
            with warnings.catch_warnings():           # all-sentinel rows -> nan
                warnings.simplefilter("ignore", RuntimeWarning)
                extra["cap_mean"] = np.nanmean(m, axis=1)
            extra["cap_armed"] = np.isfinite(m).sum(axis=1)
            extra["nband"] = int(caps.shape[1])
        except Exception as e:              # pragma: no cover
            extra["cap_mean"] = None
            extra["cap_why"] = f"{type(e).__name__}: {e}"
        # ---- Tobs, for captions -----------------------------------------
        try:
            da = dict(g["domain_settings/args"].attrs)
            extra["tobs"] = float(da["0"]) * float(da["1"]) * float(da["2"])
        except Exception:
            extra["tobs"] = np.nan

    phys = np.asarray(tc.both_transforms(rows))
    f0_mhz = phys[:, 1] * 1e3
    logA = np.log10(np.maximum(phys[:, 0], 1e-30))
    hi = phys[:, 1] > 10e-3
    neg = float((phys[hi, 2] < 0).mean()) if hi.any() else np.nan
    # flagship: the highest-f catalogue source, 20.380377 mHz
    win = np.abs(phys[:, 1] - FLAG_F0) <= FLAG_TOL
    nw_ = int(cold.shape[1])
    fr = phys[win, 2] / FLAG_FDOT if win.any() else np.array([])
    extra.update(
        flag_n=int(win.sum()),
        flag_mult=float(win.sum()) / max(nw_, 1),
        flag_fdot_ratio=float(np.median(fr)) if fr.size else np.nan,
        nwalkers=int(cold.shape[1]),
        # raw SAMPLING-basis sky columns (6 = alpha, 7 = sin_delta)
        sky=np.column_stack([np.mod(rows[:, 6], 2 * np.pi), rows[:, 7]]),
        fdot=phys[:, 2],
        ll_spread=cold.max(1) - cold.min(1),
    )
    return dict(it=it, ll_max=cold.max(1), ll_mean=cold.mean(1),
                leaves=counts, pts=np.column_stack([f0_mhz, logA]),
                f0=phys[:, 1], n_hi=int(hi.sum()), neg_hi=neg, **extra)


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=PANEL,
                bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def styled_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.grid(color=LINE, lw=0.5, alpha=0.6)


def leg(ax, **kw):
    ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8, **kw)


def lab(ax, x=None, y=None, fs=9):
    """Axis labels at a size that fits these short panels without clipping."""
    if x is not None:
        ax.set_xlabel(x, fontsize=fs, color=DIM)
    if y is not None:
        ax.set_ylabel(y, fontsize=fs, color=DIM)


def grid_axes(n, w=10.5, h=6.4, sharex=True, sharey=True):
    """2 x 2 (or 1 x n) small-multiple grid with SHARED axes and limits."""
    nr, nc = (2, 2) if n > 2 else (1, max(n, 1))
    fig, axs = plt.subplots(nr, nc, figsize=(w, h if nr == 2 else h / 2),
                            facecolor=PANEL, sharex=sharex, sharey=sharey)
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        styled_ax(ax)
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs[:n]


def outer_labels(axs, xl, yl):
    """Label only the outer edge of a shared-axis grid (2x2 or 1xn)."""
    n = len(axs)
    nc = 2 if n > 2 else n
    for i, ax in enumerate(axs):
        lab(ax, xl if i >= n - nc else None, yl if i % nc == 0 else None)


def panel_tag(ax, tag, col, extra=""):
    ax.set_title(f"{tag}{extra}", color=col, fontsize=10, fontweight="bold",
                 loc="left", pad=3)


def hole(why):
    """A labeled placeholder div -- never a silent drop."""
    return (f'<div class="hole"><b>panel unavailable</b> &mdash; {why}</div>')


def load_vgb_cat():
    """(f0 in mHz, truth distance in kpc, IDs) for the 55 fixed VGB leaves."""
    from lisatools.globalfit.stock.erebor.vgb import load_vgb_catalogue_file
    v = np.asarray(load_vgb_catalogue_file(MOJITO_CAT_DIR)["vgb"]).item()
    f0 = np.asarray(v["GW22FrequencySSBFrame"], float) * 1e3
    dist = np.asarray(v["LuminosityDistance"], float) * 1e3      # Mpc -> kpc
    ids = [i.decode() if isinstance(i, bytes) else str(i) for i in v["ID"]]
    return f0, dist, ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("v4"); ap.add_argument("v5"); ap.add_argument("v6")
    ap.add_argument("out")
    ap.add_argument("--yr1", default=None,
                    help="optional 1-yr run dir/h5 -- adds a 4th panel/line")
    ap.add_argument("--truth", default="gb_truth_3to21.npz")
    a = ap.parse_args(argv)

    runs = {}
    srcs = [a.v4, a.v5, a.v6] + ([a.yr1] if a.yr1 else [])
    for (tag, col), src in zip(RUN_META, srcs):
        p = live_h5(src)
        print(f"[{tag}] {p}")
        runs[tag] = load_run(p)
        runs[tag]["col"] = col
        runs[tag]["src"] = os.path.basename(p)

    truth_det = truth_sub = []
    TR = None
    if os.path.exists(a.truth):
        z = np.load(a.truth)
        t_f0, t_amp, det = z["f0"] * 1e3, np.log10(z["amp"]), z["det"] > 0
        truth_det = np.column_stack([t_f0[det], t_amp[det]]).round(5).tolist()
        truth_sub = np.column_stack([t_f0[~det], t_amp[~det]]).round(5).tolist()
        TR = dict(f0=t_f0, logA=t_amp, det=det,
                  phys=z["phys"] if "phys" in z.files else None)

    # ---- overplot figures ------------------------------------------------
    figs = {}
    fig, ax = plt.subplots(figsize=(10.5, 3.2), facecolor=PANEL)
    styled_ax(ax)
    for tag in runs:
        r = runs[tag]
        ax.plot(r["ll_max"] - r["ll_max"][0], color=r["col"], lw=1.8,
                label=f"{tag} (it {r['it']})")
    ax.set_xlabel("iteration"); ax.set_ylabel("cold logL max − start")
    ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8)
    figs["ll"] = b64fig(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.0), facecolor=PANEL)
    styled_ax(ax)
    n4 = runs["v4"]["ll_max"]
    for tag in ("v5", "v6"):
        r = runs[tag]
        n = min(len(n4), len(r["ll_max"]))
        ax.plot(np.arange(n), r["ll_max"][:n] - n4[:n], color=r["col"],
                lw=1.8, label=f"{tag} − v4")
    ax.axhline(0, color=runs["v4"]["col"], lw=1.2, ls="--", label="v4")
    ax.set_xlabel("iteration (matched)"); ax.set_ylabel("Δ cold logL max")
    ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8)
    figs["dll"] = b64fig(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.0), facecolor=PANEL)
    styled_ax(ax)
    for tag in runs:
        r = runs[tag]
        ax.plot(r["leaves"], color=r["col"], lw=1.8, label=tag)
    ax.set_xlabel("iteration"); ax.set_ylabel("alive GB leaves / walker")
    ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8)
    figs["lv"] = b64fig(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.0), facecolor=PANEL)
    styled_ax(ax)
    bins = np.logspace(np.log10(4e-4), np.log10(2.3e-2), 70)
    for tag in runs:
        r = runs[tag]
        ax.hist(r["f0"], bins=bins, histtype="step", lw=1.6,
                color=r["col"], label=f"{tag} @ it {r['it']}")
    ax.set_xscale("log")
    ax.set_xlabel("f0 [Hz]"); ax.set_ylabel("cold leaves (pooled)")
    ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8)
    figs["cnt"] = b64fig(fig)

    # ================= comparison replicas of the monitor sections =======
    # Every section below is the comparison form of one gf_monitor_gen
    # section: overplotted lines where the monitor draws lines, shared-axis
    # small multiples where the monitor draws an image / 2-D distribution.
    holes = {k: "" for k in ("pop", "sky", "hif", "psd", "gal", "vgb", "cap")}
    tags = list(runs)

    # ---- (g) RESIDUAL PROXY: cold logL mean + walker spread -------------
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2), facecolor=PANEL)
    for ax in axs:
        styled_ax(ax)
    for t in tags:
        r = runs[t]
        axs[0].plot(r["ll_mean"] - r["ll_mean"][0], color=r["col"], lw=1.6,
                    label=t)
        axs[1].plot(r["ll_spread"], color=r["col"], lw=1.4, label=t)
    lab(axs[0], "iteration", "cold logL mean − start")
    lab(axs[1], "iteration", "walker lnL spread (max − min)")
    axs[1].set_yscale("log")
    leg(axs[0])
    fig.tight_layout()
    figs["llx"] = b64fig(fig)

    # ---- (f) CAP / LEAF-CAP tracker -------------------------------------
    if all(runs[t].get("cap_mean") is not None for t in tags):
        fig, ax = plt.subplots(figsize=(10.5, 3.0), facecolor=PANEL)
        styled_ax(ax)
        for t in tags:
            r = runs[t]
            ax.plot(r["cap_mean"], color=r["col"], lw=1.8,
                    label=f"{t} ({r['nband']} bands)")
        lab(ax, "iteration", "mean leaf cap per band")
        leg(ax)
        fig.tight_layout()
        figs["cap"] = b64fig(fig)
    else:
        figs["cap"] = ""
        holes["cap"] = hole("sub_backend/gb/band_leaf_cap unreadable in "
                            + ", ".join(t for t in tags
                                        if runs[t].get("cap_mean") is None))

    # ---- (b) RECOVERED POPULATION, side by side -------------------------
    fig, axs = grid_axes(len(tags), h=6.6)
    xall = np.concatenate([runs[t]["pts"][:, 0] for t in tags])
    yall = np.concatenate([runs[t]["pts"][:, 1] for t in tags])
    for ax, t in zip(axs, tags):
        r = runs[t]
        if TR is not None:
            ax.scatter(TR["f0"][~TR["det"]], TR["logA"][~TR["det"]], s=2,
                       marker="x", lw=0.35, color=TRUTHRED, alpha=0.10)
            ax.scatter(TR["f0"][TR["det"]], TR["logA"][TR["det"]], s=9,
                       marker="x", lw=0.7, color=TRUTHRED, alpha=0.85)
        ax.scatter(r["pts"][:, 0], r["pts"][:, 1], s=2.5, lw=0,
                   color=r["col"], alpha=0.45)
        panel_tag(ax, t, r["col"], f"   it {r['it']}   {len(r['pts'])} leaves")
        ax.set_xscale("log")
    axs[0].set_xlim(max(xall.min() * 0.9, 0.2), xall.max() * 1.1)
    axs[0].set_ylim(np.percentile(yall, 0.02) - 0.15, yall.max() + 0.15)
    outer_labels(axs, "f0 [mHz]", "log10 strain amplitude")
    fig.tight_layout()
    figs["pop"] = b64fig(fig)

    # ---- (c) SKY DISTRIBUTION, side by side ------------------------------
    fig, axs = grid_axes(len(tags), h=6.0)
    for ax, t in zip(axs, tags):
        r = runs[t]
        if TR is not None and TR["phys"] is not None:
            ph = TR["phys"][TR["det"]]
            ax.scatter(np.mod(ph[:, 7], 2 * np.pi), np.sin(ph[:, 8]), s=5,
                       marker="x", lw=0.6, color=TRUTHRED, alpha=0.7)
        ax.scatter(r["sky"][:, 0], r["sky"][:, 1], s=2.0, lw=0,
                   color=r["col"], alpha=0.30)
        panel_tag(ax, t, r["col"], f"   it {r['it']}")
    axs[0].set_xlim(0, 2 * np.pi)
    axs[0].set_ylim(-1.02, 1.02)
    outer_labels(axs, "alpha [rad]", "sin_delta")
    fig.tight_layout()
    figs["sky"] = b64fig(fig)

    # ---- (d) HIGH-F FOCUS ------------------------------------------------
    hb = np.arange(10.0, 22.0 + 1e-9, 0.1)
    fig, ax = plt.subplots(figsize=(10.5, 3.2), facecolor=PANEL)
    styled_ax(ax)
    for t in tags:
        r = runs[t]
        h, _ = np.histogram(r["f0"] * 1e3, bins=hb)
        ax.step(hb[:-1], h / max(r["nwalkers"], 1), where="post", lw=1.5,
                color=r["col"], label=f"{t} (n={r['n_hi']})")
    if TR is not None:
        h, _ = np.histogram(TR["f0"][TR["det"]], bins=hb)
        ax.step(hb[:-1], h, where="post", lw=1.2, color=TRUTHRED, ls="--",
                label="detectable truths")
    ax.axvline(FLAG_F0 * 1e3, color=FG, lw=0.8, ls=":")
    lab(ax, "f0 [mHz]  (0.1 mHz bins)", "cold leaves / walker")
    leg(ax)
    fig.tight_layout()
    figs["hif"] = b64fig(fig)

    hirows = "".join(
        f"<tr><td><span class='chip' style='background:{runs[t]['col']}'>"
        f"</span>{t}</td><td>{runs[t]['n_hi']}</td>"
        f"<td>{100 * runs[t]['neg_hi']:.1f}%</td>"
        f"<td>{runs[t]['flag_n']}</td>"
        f"<td>{runs[t]['flag_mult']:.2f}</td>"
        f"<td>{runs[t]['flag_fdot_ratio']:.3f}</td></tr>" for t in tags)

    # ---- (a) NOISE MODEL: traces + latest-iteration posteriors ----------
    for br, names, inj, key in (("psd", PSD_NAMES, PSD_INJ, "psd"),
                                ("galfor", GAL_NAMES, None, "gal")):
        bad = [t for t in tags if runs[t].get(br) is None
               or not len(runs[t][br])]
        if bad:
            figs[key + "tr"] = figs[key + "h"] = ""
            holes[key] = hole(f"chain/{br} unreadable / empty in "
                              + ", ".join(bad))
            continue
        np_ = len(names)
        fig, axs = plt.subplots(1, np_, figsize=(10.5, 2.5), facecolor=PANEL)
        axs = np.atleast_1d(axs)
        for ax in axs:
            styled_ax(ax)
        for j in range(np_):
            for t in tags:
                c = runs[t][br]
                ax = axs[j]
                ax.plot(np.median(c[:, :, j], axis=1), color=runs[t]["col"],
                        lw=1.3, label=t)
            if inj is not None:
                axs[j].axhline(inj[j], color=TRUTHRED, lw=1.0, ls=":")
            axs[j].set_title(names[j], color=DIM, fontsize=9)
            lab(axs[j], "iteration", "cold-walker median" if j == 0 else None)
            axs[j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                                    useOffset=False)
        leg(axs[0])
        fig.tight_layout()
        figs[key + "tr"] = b64fig(fig)

        fig, axs = plt.subplots(1, np_, figsize=(10.5, 2.4), facecolor=PANEL)
        axs = np.atleast_1d(axs)
        for ax in axs:
            styled_ax(ax)
        for j in range(np_):
            lo = min(runs[t][br][-1, :, j].min() for t in tags)
            hiv = max(runs[t][br][-1, :, j].max() for t in tags)
            if not np.isfinite(lo) or hiv <= lo:
                lo, hiv = lo - 1e-30, lo + 1e-30
            bins = np.linspace(lo, hiv, 22)
            for t in tags:
                axs[j].hist(runs[t][br][-1, :, j], bins=bins, histtype="step",
                            lw=1.4, color=runs[t]["col"], label=t)
            if inj is not None:
                axs[j].axvline(inj[j], color=TRUTHRED, lw=1.0, ls=":")
            axs[j].set_title(names[j], color=DIM, fontsize=9)
            axs[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                                    useOffset=False)
            axs[j].tick_params(axis="x", labelrotation=30)
        leg(axs[0])
        fig.tight_layout()
        figs[key + "h"] = b64fig(fig)

    # ---- (e) VGB: 55 per-leaf cold-chain distance medians ---------------
    bad = [t for t in tags if runs[t].get("vgb") is None]
    if bad:
        figs["vgb"] = ""
        holes["vgb"] = hole("chain/vgb unreadable in " + ", ".join(bad))
    else:
        try:
            vf0, vtrue, vids = load_vgb_cat()
            xlab, xlog = "catalogue f0 [mHz]", True
        except Exception as e:
            vf0 = np.arange(runs[tags[0]]["vgb"].shape[1])
            vtrue, vids, xlog = None, None, False
            xlab = "VGB leaf index"
            holes["vgb"] = hole(
                "catalogue f0 / truth distances unavailable "
                f"({type(e).__name__}: {e}) &mdash; x is the leaf index and "
                "no truth markers are drawn")
        o = np.argsort(vf0)
        fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor=PANEL)
        styled_ax(ax)
        if vtrue is not None:
            ax.plot(vf0[o], vtrue[o], marker="_", ms=13, lw=0, mew=1.8,
                    color=TRUTHRED, label="catalogue truth")
        for t in tags:
            med = np.median(runs[t]["vgb"][:, :, 0], axis=0)
            # markers only: the 55 leaves are independent, a joining line
            # would read as a spectrum that does not exist
            ax.plot(vf0[o], med[o], marker="o", ms=4.0, lw=0, alpha=0.8,
                    color=runs[t]["col"], label=f"{t} @ it {runs[t]['it']}")
        if xlog:
            ax.set_xscale("log")
        ax.set_yscale("log")
        lab(ax, xlab, "distance [kpc]")
        leg(ax, ncol=2)
        fig.tight_layout()
        figs["vgb"] = b64fig(fig)

    # new sections are injected as ready-made HTML blocks so a missing panel
    # renders its placeholder instead of a broken <img>
    def block(key, alt, hole_key=None):
        b = figs.pop(key, "")
        img = f'<img alt="{alt}" src="data:image/png;base64,{b}">' if b else ""
        return img + holes.get(hole_key, "")

    figs["b_llx"] = block("llx", "cold logL mean and walker spread, all runs")
    figs["b_cap"] = block("cap", "mean leaf cap per band vs iteration", "cap")
    figs["b_pop"] = block("pop", "amplitude vs frequency, one panel per run",
                          "pop")
    figs["b_sky"] = block("sky", "sky distribution, one panel per run", "sky")
    figs["b_hif"] = block("hif", "leaves per 0.1 mHz bin above 10 mHz", "hif")
    figs["b_psdtr"] = block("psdtr", "psd parameter traces", "psd")
    figs["b_psdh"] = block("psdh", "psd posteriors at the latest iteration")
    figs["b_galtr"] = block("galtr", "galfor parameter traces", "gal")
    figs["b_galh"] = block("galh", "galfor posteriors at latest iteration")
    figs["b_vgb"] = block("vgb", "VGB per-leaf distance medians", "vgb")
    figs["hirows"] = hirows
    figs["tobsnote"] = "; ".join(
        f"{t} Tobs = {runs[t]['tobs'] / 86400:.0f} d" for t in tags)

    data = {t: dict(pts=runs[t]["pts"].round(5).tolist(), col=runs[t]["col"],
                    it=runs[t]["it"]) for t in runs}
    payload = json.dumps(dict(runs=data, truth=truth_det, truth_sub=truth_sub),
                         separators=(",", ":"))

    rows = "".join(
        f"<tr><td><span class='chip' style='background:{runs[t]['col']}'></span>"
        f"{t}</td><td>{runs[t]['src']}</td><td>{runs[t]['it']}</td>"
        f"<td>{runs[t]['leaves'][-1]:.0f}</td>"
        f"<td>{runs[t]['ll_max'][-1]:,.0f}</td><td>{runs[t]['n_hi']}</td>"
        f"<td>{100 * runs[t]['neg_hi']:.1f}%</td></tr>" for t in runs)

    html = HTML_TMPL.format(payload=payload, rows=rows, **figs)
    with open(a.out, "w") as fh:
        fh.write(html)
    print(f"wrote {a.out}: {os.path.getsize(a.out)/1e6:.1f} MB")


HTML_TMPL = """<title>LISA Global Fit Compare</title>
<style>
:root {{ --bg:{BG}; --panel:{PANEL}; --line:{LINE}; --fg:{FG}; --dim:{DIM};
        --truthred:{TRUTHRED}; }}
:root[data-theme="light"] {{ --bg:#EEF1F5; --panel:#FFFFFF; --line:#D4DBE3;
        --fg:#25313D; --dim:#5D6B7A; --truthred:#E00016; }}
body {{ background:var(--bg); color:var(--fg); margin:0;
  font:14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width:1060px; margin:0 auto; padding:20px 16px 60px; }}
h1 {{ font-size:19px; font-weight:600; margin:6px 0 2px; }}
h2 {{ font-size:13px; font-weight:600; color:var(--dim);
     text-transform:uppercase; letter-spacing:.08em; margin:30px 0 8px; }}
p.note {{ color:var(--dim); font-size:12px; margin:4px 0 10px; }}
img {{ width:100%; border:1px solid var(--line); border-radius:4px;
      background:var(--panel); display:block; margin:6px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
        font-variant-numeric:tabular-nums; }}
td, th {{ border:1px solid var(--line); padding:5px 9px; text-align:right; }}
td:first-child, th:first-child {{ text-align:left; }}
th {{ color:var(--dim); font-weight:600; }}
.chip {{ display:inline-block; width:10px; height:10px; border-radius:2px;
        margin-right:7px; }}
#stack {{ border:1px solid var(--line); border-radius:4px; overflow:hidden; }}
#stack canvas {{ display:block; width:100%; height:250px; background:var(--panel);
        touch-action:none; cursor:grab; border:0;
        border-bottom:1px solid var(--line); }}
#stack canvas:last-child {{ border-bottom:0; }}
.bar {{ display:flex; gap:8px; align-items:center; margin:8px 0; }}
button {{ background:var(--panel); color:var(--fg); border:1px solid var(--line);
        border-radius:4px; padding:4px 12px; font-size:12px; cursor:pointer; }}
button:focus-visible {{ outline:2px solid var(--dim); }}
button.armed {{ outline:2px solid var(--truthred); }}
.bar label {{ color:var(--dim); font-size:12px; display:flex; gap:4px;
        align-items:center; }}
.bar input[type=text] {{ width:86px; background:var(--panel); color:var(--fg);
        border:1px solid var(--line); border-radius:3px; padding:2px 5px;
        font:12px monospace; }}
.bar input[type=range] {{ width:110px; }}
.hole {{ border:1px dashed var(--line); border-radius:4px; padding:10px 12px;
        color:var(--dim); font-size:12px; background:var(--panel);
        margin:6px 0; }}
.hole b {{ color:var(--truthred); font-weight:600; }}
</style>
<main>
<h1>LISA Global Fit Compare</h1>
<p class="note">Run comparison on shared axes. Same colour per run
everywhere: <span class="chip" style="background:#58C48A"></span>v4
<span class="chip" style="background:#FF7BAC"></span>v5
<span class="chip" style="background:#9B7BFF"></span>v6
<span class="chip" style="background:#F2E14C"></span>1yr (different Tobs
-- shown for contrast, not matched-iteration comparison); red is reserved
for catalogue truths. Truth crosses are the 3-month detectability set;
the 1-yr run's own detectable set is larger.</p>
<table><tr><th>run</th><th>store</th><th>iterations</th>
<th>leaves/walker</th><th>cold logL max</th><th>leaves &gt; 10 mHz</th>
<th>fdot &lt; 0 above 10 mHz</th></tr>{rows}</table>

<h2>Cold-chain log-likelihood</h2>
<img alt="cold logL max vs iteration, three runs" src="data:image/png;base64,{ll}">
<img alt="delta logL vs v4 at matched iteration" src="data:image/png;base64,{dll}">
<h2>Model size</h2>
<img alt="alive leaves per walker vs iteration" src="data:image/png;base64,{lv}">
<h2>Source counts vs frequency (latest iteration)</h2>
<img alt="cold leaves vs frequency, three runs overstepped" src="data:image/png;base64,{cnt}">

<h2>Synced zoom — amplitude vs frequency, one panel per run</h2>
<p class="note">Drag to pan, scroll to zoom, on ANY panel — the view is
shared, so all three windows show exactly the same axes at all times.
Red crosses: detectable catalogue truths (SNR&gt;7); faint crosses: below
threshold. Cloud: last-iteration cold-chain leaves, all walkers.</p>
<div class="bar">
  <button id="reset">reset view</button>
  <button id="truthbtn">toggle catalogue truths</button>
  <button id="cmp_pick" title="arm, then click any panel to set the view center">set center by click</button>
  <label>cx <input id="cmp_cx" type="text"></label>
  <label>cy <input id="cmp_cy" type="text"></label>
  <label>width <input id="cmp_wsl" type="range" min="0" max="1000" step="1"><input id="cmp_w" type="text"></label>
  <label>height <input id="cmp_hsl" type="range" min="0" max="1000" step="1"><input id="cmp_h" type="text"></label>
  <span class="note" id="viewcap"></span>
</div>
<div id="stack"></div>

<h2>Residual proxy — cold logL mean &amp; walker spread</h2>
<p class="note">The monitor's status log-likelihood panel in comparison form.
Left: cold-chain <em>mean</em> lnL over the 24 walkers, offset to its own
start so runs with different absolute lnL floors overlay. Right: the
max−min walker spread (log axis) — a converging model narrows it. The 1-yr
run's absolute lnL is not comparable to the 3-month runs ({tobsnote}); its
<em>shape</em> is.</p>
{b_llx}

<h2>Recovered population — one panel per run</h2>
<p class="note">Static counterpart of the synced zoom stack, for printing
and scanning: identical x/y limits on all panels, log frequency axis. Red
crosses are catalogue truths (bold = detectable at SNR&nbsp;&gt;&nbsp;7,
faint = below threshold, 3-month set); the coloured cloud is that run's
last-iteration cold-chain leaves pooled over all walkers.</p>
{b_pop}

<h2>Sky distribution — one panel per run</h2>
<p class="note">Sampling-basis columns 6 (alpha) and 7 (sin_delta) at each
run's latest coordinate row, shared axes. Red crosses: detectable
catalogue truths. A uniform-in-sin_delta, uniform-in-alpha cloud with a
galactic-plane overdensity is the expected shape; structure beyond that is
a sampler artefact, not astrophysics.</p>
{b_sky}

<h2>High-frequency focus (&gt; 10 mHz)</h2>
<p class="note">Leaves per 0.1&nbsp;mHz bin above 10&nbsp;mHz, per cold
walker, one step line per run; dashed red is the detectable-truth count in
the same bins (an absolute count, not per walker — the shape is the
comparison, not the height). The dotted vertical marks the flagship source
at 20.380&nbsp;mHz. <em>Flagship multiplicity</em> counts cold leaves
within ±5.144e−6&nbsp;Hz of it, divided by the walker count: 1.0 means each
walker holds exactly one; &gt;&nbsp;1 means the run is splitting one source
across several leaves. <em>fdot ratio</em> is the median recovered fdot in
that window over the true 1.0245e−13&nbsp;Hz/s.</p>
{b_hif}
<table><tr><th>run</th><th>leaves &gt; 10 mHz</th>
<th>fdot &lt; 0 above 10 mHz</th><th>flagship leaves (pooled)</th>
<th>flagship multiplicity</th><th>median fdot / truth</th></tr>{hirows}</table>

<h2>Leaf-cap tracker</h2>
<p class="note">Mean enforced leaf cap per band vs iteration, one line per
run. Bands whose cap is still the −1 sentinel (not yet armed) are excluded
from the mean, so early iterations average over fewer bands. v6 runs a
finer band grid than v4/v5/1-yr — the band count per run is in the legend,
so the level is comparable only in shape.</p>
{b_cap}

<h2>Noise model — instrument PSD</h2>
<p class="note">Left-to-right, one panel per sampled parameter. Traces:
cold-chain median over walkers per iteration, one line per run. Histograms:
the walker posterior at each run's latest stored iteration, overlaid step
histograms on shared bins. Dotted red = injected value.</p>
{b_psdtr}
{b_psdh}

<h2>Noise model — galactic foreground</h2>
<p class="note">Same construction for the 5 galfor parameters (no injected
truth exists for these — the foreground model is fitted, not injected).
The 1-yr run sees a different confusion level than the 3-month runs, so a
genuine offset there is expected, not a bug.</p>
{b_galtr}
{b_galh}

<h2>Verification binaries — per-leaf distance</h2>
<p class="note">All 55 fixed VGB leaves: cold-walker median of the sampled
distance at each run's latest iteration, plotted against catalogue
frequency. Red dashes are the catalogue distances. Leaves that sit on the
red marks are recovered; leaves that wander are either genuinely
uninformative (low SNR) or being pulled by a neighbouring GB.</p>
{b_vgb}
</main>
<script>
const D = {payload};
const stack = document.getElementById("stack");
const css = getComputedStyle(document.documentElement);
const C = n => css.getPropertyValue(n).trim();
const tags = Object.keys(D.runs);
let showT = true;
// shared view-state
let X0, X1, Y0, Y1;
const allx = [], ally = [];
for (const t of tags) for (const p of D.runs[t].pts) {{ allx.push(p[0]); ally.push(p[1]); }}
const pad = (a, b) => [a - (b - a) * .04, b + (b - a) * .04];
const full = () => {{ [X0, X1] = pad(Math.min(...allx), Math.max(...allx));
                     [Y0, Y1] = pad(Math.min(...ally), Math.max(...ally)); }};
full();
const ML = 56, MR = 10, MT = 6, MB = 24;
const dpr = window.devicePixelRatio || 1;
const cvs = tags.map((t, i) => {{
  const cv = document.createElement("canvas");
  cv.dataset.tag = t;
  cv.setAttribute("aria-label", "zoom panel " + t);
  stack.appendChild(cv);
  return cv;
}});
function drawOne(cv, isLast) {{
  const t = cv.dataset.tag, run = D.runs[t];
  const w = cv.clientWidth, h = cv.clientHeight;
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (cv.width !== bw || cv.height !== bh) {{ cv.width = bw; cv.height = bh; }}
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = C("--panel"); g.fillRect(0, 0, w, h);
  const mb = isLast ? MB : 6;
  const sx = x => ML + (x - X0) / (X1 - X0) * (w - ML - MR);
  const sy = y => h - mb - (y - Y0) / (Y1 - Y0) * (h - mb - MT);
  g.strokeStyle = C("--line"); g.fillStyle = C("--dim"); g.font = "10px monospace";
  for (let i = 0; i <= 6; i++) {{
    const xv = X0 + (X1 - X0) * i / 6, yv = Y0 + (Y1 - Y0) * i / 6;
    g.beginPath(); g.moveTo(sx(xv), MT); g.lineTo(sx(xv), h - mb); g.stroke();
    if (i > 0 && i < 6) {{
      g.beginPath(); g.moveTo(ML, sy(yv)); g.lineTo(w - MR, sy(yv)); g.stroke();
      g.fillText(yv.toPrecision(3), 4, sy(yv) + 3);
    }}
    if (isLast) g.fillText(xv.toPrecision(4), sx(xv) - 14, h - 8);
  }}
  const draw_t = (T, alpha, lw) => {{
    g.strokeStyle = C("--truthred"); g.globalAlpha = alpha;
    g.lineWidth = lw; g.lineCap = "round"; const r = 3.2;
    g.beginPath();
    for (const p of T) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ML || x > w - MR || y < MT || y > h - mb) continue;
      g.moveTo(x - r, y - r); g.lineTo(x + r, y + r);
      g.moveTo(x - r, y + r); g.lineTo(x + r, y - r);
    }}
    g.stroke(); g.globalAlpha = 1;
  }};
  if (showT) {{ draw_t(D.truth_sub, .22, 1); draw_t(D.truth, .9, 1.4); }}
  g.globalAlpha = .6; g.fillStyle = run.col;
  g.beginPath();
  for (const p of run.pts) {{
    const x = sx(p[0]), y = sy(p[1]);
    if (x < ML || x > w - MR || y < MT || y > h - mb) continue;
    g.moveTo(x + 2, y); g.arc(x, y, 2, 0, 6.29);
  }}
  g.fill(); g.globalAlpha = 1;
  g.fillStyle = run.col; g.font = "bold 12px monospace";
  g.fillText(t + "  it " + run.it + "  (" + run.pts.length + " leaves)", ML + 8, MT + 14);
}}
function draw() {{
  cvs.forEach((cv, i) => drawOne(cv, i === cvs.length - 1));
  document.getElementById("viewcap").textContent =
    X0.toPrecision(5) + " – " + X1.toPrecision(5) + " mHz";
}}
let raf = 0;
const redraw = () => {{ if (raf) return;
  raf = requestAnimationFrame(() => {{ raf = 0; draw(); syncCtl(); }}); }};
for (const cv of cvs) {{
  let drag = null;
  cv.addEventListener("pointerdown", e => {{
    drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - ML - MR) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - MT - MB) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy;
    drag = [e.clientX, e.clientY]; redraw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const w = cv.clientWidth;
    const fx = X0 + (e.offsetX - ML) / (w - ML - MR) * (X1 - X0);
    const h = cv.clientHeight;
    const fy = Y0 + (h - MB - e.offsetY) / (h - MT - MB) * (Y1 - Y0);
    const k = Math.exp(e.deltaY * 0.0015);
    X0 = fx + (X0 - fx) * k; X1 = fx + (X1 - fx) * k;
    Y0 = fy + (Y0 - fy) * k; Y1 = fy + (Y1 - fy) * k;
    redraw();
  }}, {{ passive: false }});
}}
document.getElementById("reset").onclick = () => {{ full(); draw(); }};
document.getElementById("truthbtn").onclick = () => {{ showT = !showT; draw(); }};
// ---- slider / typed / click-to-center view controls (ported from the
// monitor's viewCtl; one control strip drives the SHARED view-state) ----
const FW0 = X1 - X0, FH0 = Y1 - Y0;
const $ = id => document.getElementById("cmp_" + id);
const cxI = $("cx"), cyI = $("cy"), wI = $("w"), hI = $("h"),
      wS = $("wsl"), hS = $("hsl"), pk = $("pick");
const clampv = (v, a, b) => Math.max(a, Math.min(b, v));
const s2span = (sv, fullv) => {{
  const mn = Math.log(fullv / 2000), mx = Math.log(fullv * 1.2);
  return Math.exp(mn + (mx - mn) * sv / 1000);
}};
const span2s = (W, fullv) => {{
  const mn = Math.log(fullv / 2000), mx = Math.log(fullv * 1.2);
  return clampv(Math.round(1000 * (Math.log(W) - mn) / (mx - mn)), 0, 1000);
}};
let busy = false;
function syncCtl() {{
  if (busy) return;
  cxI.value = ((X0 + X1) / 2).toPrecision(7);
  cyI.value = ((Y0 + Y1) / 2).toPrecision(7);
  wI.value = (X1 - X0).toPrecision(5);
  hI.value = (Y1 - Y0).toPrecision(5);
  wS.value = span2s(X1 - X0, FW0);
  hS.value = span2s(Y1 - Y0, FH0);
}}
function applyTyped() {{
  let cx = parseFloat(cxI.value), cy = parseFloat(cyI.value);
  let W = parseFloat(wI.value), H = parseFloat(hI.value);
  if (!isFinite(cx)) cx = (X0 + X1) / 2;
  if (!isFinite(cy)) cy = (Y0 + Y1) / 2;
  if (!isFinite(W) || W <= 0) W = X1 - X0;
  if (!isFinite(H) || H <= 0) H = Y1 - Y0;
  busy = true;
  X0 = cx - W / 2; X1 = cx + W / 2; Y0 = cy - H / 2; Y1 = cy + H / 2;
  draw(); busy = false; syncCtl();
}}
cxI.onchange = cyI.onchange = wI.onchange = hI.onchange = applyTyped;
wS.oninput = () => {{
  const cx = (X0 + X1) / 2, W = s2span(+wS.value, FW0);
  busy = true; X0 = cx - W / 2; X1 = cx + W / 2; draw(); busy = false;
  wI.value = W.toPrecision(5);
}};
hS.oninput = () => {{
  const cy = (Y0 + Y1) / 2, H = s2span(+hS.value, FH0);
  busy = true; Y0 = cy - H / 2; Y1 = cy + H / 2; draw(); busy = false;
  hI.value = H.toPrecision(5);
}};
let picking = false;
pk.onclick = () => {{ picking = !picking; pk.classList.toggle("armed", picking);
  for (const cv of cvs) cv.style.cursor = picking ? "crosshair" : "grab"; }};
cvs.forEach((cv, i) => {{
  cv.addEventListener("pointerdown", e => {{
    if (!picking) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const mb = (i === cvs.length - 1) ? MB : 6;
    const cx = X0 + ((e.clientX - r.left) - ML) / (w - ML - MR) * (X1 - X0);
    const cy = Y0 + ((h - mb) - (e.clientY - r.top)) / (h - mb - MT) * (Y1 - Y0);
    const W = X1 - X0, H = Y1 - Y0;
    X0 = cx - W / 2; X1 = cx + W / 2; Y0 = cy - H / 2; Y1 = cy + H / 2;
    picking = false; pk.classList.remove("armed");
    for (const c of cvs) c.style.cursor = "grab";
    draw(); syncCtl();
  }}, {{ capture: true }});
}});
new ResizeObserver(redraw).observe(stack);
draw();
syncCtl();
</script>
"""
HTML_TMPL = HTML_TMPL.replace("{BG}", BG).replace("{PANEL}", PANEL) \
    .replace("{LINE}", LINE).replace("{FG}", FG).replace("{DIM}", DIM) \
    .replace("{TRUTHRED}", TRUTHRED)

if __name__ == "__main__":
    sys.exit(main())
