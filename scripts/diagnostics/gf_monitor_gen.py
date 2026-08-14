#!/usr/bin/env python
"""GF run monitor generator: <run_dir> -> self-contained monitor.html.

Reusable: point RUN_DIR at any unzipped gf_prod_* snapshot and rerun; the
artifact redeploys to the same URL. Sections degrade to labeled
placeholders when a snapshot lacks their inputs.
"""
import base64, io, json, os, re, sys
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "prod3mo/gf_prod_3mo"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gf_monitor.html"

# ---- mission-control plot style -------------------------------------------
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
CYAN, AMBER, GREEN, RED, VIOLET = "#4FD8EB", "#F5A623", "#58C48A", "#E5484D", "#9B7BFF"
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
})

IMGS, MISSING = {}, []

def fig_b64(fig, key):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    IMGS[key] = base64.b64encode(buf.getvalue()).decode()

def img(key, alt=""):
    if key not in IMGS:
        return f'<div class="missing">plot unavailable in this snapshot: {alt or key}</div>'
    return f'<img src="data:image/png;base64,{IMGS[key]}" alt="{alt or key}">'

# ============================ LOAD ==========================================
h5path = None
for fn in os.listdir(RUN_DIR):
    if fn.endswith(".h5"):
        h5path = os.path.join(RUN_DIR, fn)
f = h5py.File(h5path, "r")
g = f["global_fit"]
ll_all = g["log_like"][:, 0, 0, :]
filled = np.where(np.any(ll_all != 0.0, axis=1))[0]
NIT = int(filled.max()) + 1 if filled.size else 0
it = np.arange(NIT)
ll = ll_all[:NIT]                                   # (it, 24)
recipe = {}
rg = f.get("global_fit/recipe", f.get("recipe"))
if rg is not None:
    for k in rg:
        recipe[k] = (int(rg[k].attrs.get("order num", 0)), bool(rg[k].attrs.get("status", False)))
nwalk = ll.shape[1]

sub = g["sub_backend"]
psd_c = sub["psd/chain"][:NIT]                      # (it, 12, 24, 1, 2)
gal_c = sub["galfor/chain"][:NIT]                   # (it, 12, 24, 1, 5)
vgb_c = sub["vgb/chain"][:NIT, 0]                   # (it, 24, 55, 5)
vgb_hh = sub["vgb/h_h"][:NIT]                       # (it, 24, 55)
gb_inds = g["inds/gb"][:NIT, 0, 0]                  # (it, 24, 10000)
gb_chain_cold = g["chain/gb"][NIT-1, 0, 0]          # (24, 10000, 9) last iter
gb_alive_last = g["inds/gb"][NIT-1, 0, 0]           # (24, 10000)
caps = sub["gb/band_leaf_cap"][:NIT]                # (it, 154)
band_edges = sub["gb/band_edges"][:]
psd_sw_a = sub["psd/swaps_accepted"][:NIT]; psd_sw_p = sub["psd/swaps_proposed"][:NIT]
gal_sw_a = sub["galfor/swaps_accepted"][:NIT]; gal_sw_p = sub["galfor/swaps_proposed"][:NIT]

logpath = None
for root, _, fns in os.walk(RUN_DIR):
    for fn in fns:
        if fn == "globalfit_run.log":
            logpath = os.path.join(root, fn)
log_text = open(logpath, errors="replace").read() if logpath else ""

# ============================ PLOTS =========================================
# ---- 1. likelihood ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for w in range(nwalk):
    ax[0].plot(it, ll[:, w], color=CYAN, alpha=0.25, lw=0.8)
ax[0].plot(it, ll.max(axis=1), color=AMBER, lw=1.8, label="max")
ax[0].plot(it, np.median(ll, axis=1), color=FG, lw=1.2, ls="--", label="median")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("cold-chain lnL"); ax[0].legend()
ax[0].set_title("total log-likelihood (24 walkers)")
ax[1].plot(it, ll.max(axis=1) - ll.min(axis=1), color=VIOLET, lw=1.5)
ax[1].set_xlabel("iteration"); ax[1].set_title("walker lnL spread (max - min)")
fig_b64(fig, "ll")

# ---- 2. PSD params ----
psd_cold = psd_c[:, 0, :, 0, :]                     # (it, 24, 2)
SOMS_INJ, SA_INJ = 1.496182e-11, 2.982412e-15
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    for w in range(nwalk):
        ax[j].plot(it, psd_cold[:, w, j], color=CYAN, alpha=0.3, lw=0.8)
    ax[j].axhline(inj, color=RED, lw=1.4, ls=":", label="injected")
    ax[j].set_title(f"psd: {name}"); ax[j].set_xlabel("iteration"); ax[j].legend()
fig_b64(fig, "psd_trace")

fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    v = psd_cold[-min(3, NIT):, :, j].ravel()
    ax[j].hist(v, bins=24, color=CYAN, alpha=0.85)
    ax[j].axvline(inj, color=RED, lw=1.4, ls=":")
    ax[j].set_title(f"{name} posterior (last {min(3,NIT)} iters x 24 walkers)")
fig_b64(fig, "psd_hist")

# ---- 3. galfor params ----
gal_cold = gal_c[:, 0, :, 0, :]                     # (it, 24, 5)
GAL_NAMES = ["log10 amp", "p1", "log10 fknee", "p2", "slope"]
fig, ax = plt.subplots(1, 5, figsize=(14, 2.7))
for j in range(5):
    for w in range(nwalk):
        ax[j].plot(it, gal_cold[:, w, j], color=AMBER, alpha=0.3, lw=0.8)
    ax[j].set_title(GAL_NAMES[j], fontsize=9); ax[j].set_xlabel("iter")
fig_b64(fig, "gal_trace")
fig, ax = plt.subplots(1, 5, figsize=(14, 2.5))
for j in range(5):
    ax[j].hist(gal_cold[-min(3, NIT):, :, j].ravel(), bins=20, color=AMBER, alpha=0.85)
    ax[j].set_title(GAL_NAMES[j], fontsize=9)
fig_b64(fig, "gal_hist")

# ---- 4. PSD curves: lisatools LISASens + HyperbolicTangent foreground ----
try:
    from lisatools.sensitivity import get_sensitivity, LISASens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)

    fr = np.logspace(np.log10(2e-4), np.log10(2.6e-2), 500)

    def sens_curves(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms**2, sa**2,
                                      lisa_models.DefaultOrbits(), "mon")
        if galp is None:
            return get_sensitivity(fr, sens_fn=LISASens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=LISASens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    # panel 1: latest iteration, PSD-only vs PSD+foreground vs injected
    pm = np.median(psd_cold[-1], axis=0)
    gm = np.median(gal_cold[-1], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_curves(*pm), color=CYAN, lw=1.6, label="instrument PSD")
    ax.plot(fr, sens_curves(pm[0], pm[1], gm), color=AMBER, lw=1.6,
            label="PSD + galactic foreground")
    ax.plot(fr, sens_curves(SOMS_INJ, SA_INJ), color=RED, ls=":", lw=1.3,
            label="injected instrument")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        "sensitivity, cold-chain walker-median, latest stored iteration")
    fig_b64(fig, "psd_curves")

    # panel 2: foreground evolution over stored iterations (decline watch)
    import matplotlib.colors as mcolors
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_curves(*pm), color=CYAN, lw=1.4,
            label="instrument PSD (latest)")
    for k in range(NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold[k], axis=0)
        ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                color=ramp(k / max(NIT - 1, 1)), lw=1.1,
                label=f"iter {k}" if k in (0, NIT - 1) else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        "PSD + foreground per stored iteration (light -> dark = later)")
    fig_b64(fig, "psd_evolution")
except Exception as e:
    MISSING.append(f"PSD curve render failed: {e!r}")

# ---- 5. GB leaves + caps ----
gb_counts = gb_inds.sum(axis=-1)                    # (it, 24)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for w in range(nwalk):
    ax[0].plot(it, gb_counts[:, w], color=GREEN, alpha=0.35, lw=0.9)
ax[0].plot(it, gb_counts.max(axis=1), color=GREEN, lw=1.8)
ax[0].set_title("GB leaf count (cold walkers)"); ax[0].set_xlabel("iteration")
ax[0].set_ylim(bottom=-0.5)
im = ax[1].imshow(caps.T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, NIT, 0, caps.shape[1]])
ax[1].set_title("per-band leaf cap"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("band")
fig.colorbar(im, ax=ax[1], shrink=0.85)
# High-f barren-band birth shutoff (GB_RJ_BAND_SHUTOFF_*): each shutoff
# emits "[GB_BAND_SHUTOFF <move>] band <b> ... births OFF ..." -- mark
# those band rows in red on the cap plot (marker at the right edge +
# translucent row line; the log carries the when, the plot the which).
shutoff_bands = sorted({int(b) for b in re.findall(
    r"\[GB_BAND_SHUTOFF[^\]]*\] band (\d+)", log_text)})
for b in shutoff_bands:
    ax[1].axhline(b + 0.5, color=RED, lw=1.0, alpha=0.55)
    ax[1].plot([NIT * 0.99], [b + 0.5], marker="<", color=RED, ms=6,
               clip_on=False)
if shutoff_bands:
    ax[1].set_title(
        f"per-band leaf cap ({len(shutoff_bands)} bands birth-OFF, red)")
fig_b64(fig, "gb_leaves")

# ---- 6. f-stat fit ----
fdir = os.path.join(RUN_DIR, "gb_fstat_fit", "shared")
epochs = sorted([d for d in os.listdir(fdir)] if os.path.isdir(fdir) else [])
fstat_meta = {}
if epochs:
    ed = os.path.join(fdir, epochs[-1])
    comb = np.load(os.path.join(ed, "fstat_grid_comb.npz"), allow_pickle=True)
    pk = np.load(os.path.join(ed, "fstat_grid_peaks_stacked.npz"), allow_pickle=True)
    fstat_meta = {"epoch": epochs[-1], "comb_keys": list(comb.keys()), "pk_keys": list(pk.keys())}
    f0n = None; Fv = None
    for k in ("f0_nodes_mHz", "f0_nodes", "f0s", "f0"):
        if k in comb: f0n = np.asarray(comb[k]); break
    for k in ("F_max", "F", "F_vals"):
        if k in comb: Fv = np.asarray(comb[k]); break
    IN_MHZ = "f0_nodes_mHz" in comb
    fig, ax = plt.subplots(figsize=(12, 3.6))
    if f0n is not None and Fv is not None and f0n.shape == Fv.shape:
        n = len(f0n); step = max(1, n // 6000)
        # max-decimate so peaks survive
        m = (n // step) * step
        fd = f0n[:m].reshape(-1, step); Fd = Fv[:m].reshape(-1, step)
        ax.plot(fd[:, 0] * (1.0 if IN_MHZ else 1e3), Fd.max(axis=1), color=CYAN, lw=0.6)
        ax.set_yscale("log"); ax.set_xlabel("f0 [mHz]"); ax.set_ylabel("F")
        ax.set_title(f"F-stat comb scan ({fstat_meta['epoch']}, {n} nodes, max-decimated)")
    fig_b64(fig, "fstat_comb")
    pf0 = None; pF = None
    for k in ("peak_f0_mHz", "peak_f0", "f0", "f0s"):
        if k in pk: pf0 = np.asarray(pk[k]); break
    for k in ("peak_F", "F"):
        if k in pk: pF = np.asarray(pk[k]); break
    PK_MHZ = "peak_f0_mHz" in pk
    if pf0 is not None and pF is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
        ax[0].scatter(pf0 * (1.0 if PK_MHZ else 1e3), np.clip(pF, 1, None), s=4, color=AMBER, alpha=0.6)
        ax[0].set_yscale("log"); ax[0].set_xlabel("f0 [mHz]"); ax[0].set_ylabel("F")
        ax[0].set_title(f"{len(pf0)} peaks (birth-grid anchors)")
        ax[1].hist(pf0 * (1.0 if PK_MHZ else 1e3), bins=80, color=AMBER, alpha=0.85)
        ax[1].set_xlabel("f0 [mHz]"); ax[1].set_title("peak density vs frequency")
        fig_b64(fig, "fstat_peaks")
else:
    MISSING.append("No fstat epoch caches found under gb_fstat_fit/shared.")

# ---- 7. VGB ----
VGB_NAMES = ["dist [kpc]", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
# Per-leaf FIXED frequencies + names from the mojito catalogue (leaf i =
# catalogue row i, fixed-leaf branch).
VGB_F0 = None
VGB_IDS = None
try:
    from lisatools.globalfit.stock.erebor.vgb import load_vgb_catalogue_file
    _cat = load_vgb_catalogue_file(os.path.expanduser(
        "~/.mojito_cache/brickmarket/mojito_light_v1_0_0"))
    _v = np.asarray(_cat["vgb"]).item()
    VGB_F0 = np.asarray(_v["GW22FrequencySSBFrame"]) * 1e3   # mHz
    VGB_IDS = [i.decode() if isinstance(i, bytes) else str(i)
               for i in _v["ID"]]
except Exception as e:
    MISSING.append(f"VGB catalogue f0 axis unavailable locally: {e!r}")

vgb_last = vgb_c[-min(3, NIT):].reshape(-1, 55, 5)   # (S, 55, 5)
snr = np.sqrt(np.clip(np.nanmean(vgb_hh[-1], axis=0), 0, None))  # (55,)
order = np.argsort(snr)[::-1]
med = np.median(vgb_last[:, :, 0], axis=0); lo = np.percentile(vgb_last[:, :, 0], 16, axis=0)
hi = np.percentile(vgb_last[:, :, 0], 84, axis=0)
if VGB_F0 is not None:
    xs, xlab_vgb = VGB_F0, "catalogue f0 [mHz]"
else:
    xs, xlab_vgb = np.arange(55), "VGB leaf index"
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.errorbar(xs, med, yerr=[med - lo, hi - med],
            fmt="o", ms=3, color=VIOLET, ecolor=VIOLET, alpha=0.9, capsize=2)
if VGB_F0 is not None:
    ax.set_xscale("log")
    for k in order[:3]:
        ax.annotate(VGB_IDS[k], (xs[k], med[k]), fontsize=7, color=FG,
                    xytext=(2, 6), textcoords="offset points")
ax.set_xlabel(xlab_vgb); ax.set_ylabel("dist [kpc]")
ax.set_title("VGB distance posteriors (median +/- 1 sigma; last iters x walkers)")
fig_b64(fig, "vgb_dist")

# SNR evolution over iterations: noise-weighted sqrt(<h|h>) rises as the
# galactic foreground is fit/subtracted down -- the source-side twin of
# the PSD decline-watch panel.
snr_it = np.sqrt(np.clip(np.nanmean(vgb_hh[:NIT], axis=1), 0, None))  # (it, 55)
import matplotlib.colors as _mc
_vramp = _mc.LinearSegmentedColormap.from_list(
    "violet", ["#E3D9FF", "#9B7BFF", "#4A2FA8"])
fig, ax = plt.subplots(figsize=(12, 3.4))
if VGB_F0 is not None:
    _of = np.argsort(VGB_F0)
    _xs = VGB_F0[_of]
    ax.set_xscale("log")
else:
    _of = np.arange(55)
    _xs = np.arange(55)
for k in range(NIT):
    ax.plot(_xs, snr_it[k][_of], color=_vramp(k / max(NIT - 1, 1)),
            lw=1.0, alpha=0.9,
            label=(f"iter {k}" if k in (0, NIT - 1) else None))
ax.plot(_xs, snr_it[-1][_of], "o", ms=3.5, color="#4A2FA8")
if VGB_F0 is not None:
    for kk in order[:3]:
        ax.annotate(VGB_IDS[kk], (VGB_F0[kk], snr[kk]), fontsize=7,
                    color=FG, xytext=(2, 4), textcoords="offset points")
ax.set_yscale("log"); ax.set_xlabel(xlab_vgb)
ax.set_ylabel("sqrt(<h|h>)")
ax.legend(fontsize=8)
ax.set_title("VGB optimal SNR per stored iteration (light -> dark = later; "
             "watch these RISE as the foreground comes down)")
fig_b64(fig, "vgb_snr")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.0))
for k in range(3):
    leaf = order[k]
    nm = VGB_IDS[leaf] if VGB_IDS else f"leaf {leaf}"
    for w in range(nwalk):
        ax[k].plot(it, vgb_c[:, w, leaf, 0], color=VIOLET, alpha=0.35, lw=0.8)
    _f0txt = f", f0={VGB_F0[leaf]:.3f} mHz" if VGB_F0 is not None else ""
    ax[k].set_title(f"{nm} (SNR~{snr[leaf]:.0f}{_f0txt}) dist", fontsize=9)
    ax[k].set_xlabel("iter")
fig_b64(fig, "vgb_traces")

fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
for j in range(1, 5):
    ax[j-1].hist(vgb_last[:, :, j].ravel(), bins=30, color=VIOLET, alpha=0.85)
    ax[j-1].set_title(VGB_NAMES[j], fontsize=9)
fig_b64(fig, "vgb_hists")

# GB/VGB explorer data (interactive)
expl = {"gb": [], "vgb": []}
nz = np.nonzero(gb_alive_last.sum(axis=0))[0]
if nz.size:
    for w in range(nwalk):
        al = np.nonzero(gb_alive_last[w])[0]
        for i_ in al:
            row = gb_chain_cold[w, i_]
            expl["gb"].append([float(row[1]), float(1.0 / max(row[0], 1e-6)), int(w)])
S = vgb_c[-1]                                        # (24, 55, 5)
for w in range(nwalk):
    for leaf in range(55):
        _x = float(VGB_F0[leaf]) if VGB_F0 is not None else int(leaf)
        expl["vgb"].append([_x, float(1.0 / max(S[w, leaf, 0], 1e-6)),
                            float(snr[leaf])])
expl["vgb_axis"] = "f0 [mHz] (catalogue)" if VGB_F0 is not None else "VGB leaf index"
EXPL_JSON = json.dumps(expl)

# zoomable dist-f0 posterior cloud: every sample (last iters x walkers x leaf)
_xs_axis = VGB_F0 if VGB_F0 is not None else np.arange(55).astype(float)
vgb_post = [[float(_xs_axis[leaf]), float(v)]
            for leaf in range(55) for v in vgb_last[:, leaf, 0]]
VGB_POST_JSON = json.dumps({
    "pts": vgb_post,
    "xlab": "catalogue f0 [mHz]" if VGB_F0 is not None else "VGB leaf index",
})

# ---- 8. efficiency: proposals/s + wall per propose (GB_TIMING records) ----
TIM_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?\[GB_TIMING (\w+)\] "
    r"total=([\d.]+)s.*\| ([^|]+)$")
recs = []
for line in log_text.splitlines():
    m = TIM_RE.match(line)
    if m:
        cnt = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)\b", m.group(4)))
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        recs.append((t, m.group(2), float(m.group(3)), cnt))
if recs:
    t0r = recs[0][0]
    MOVE_COLOR = {"rj_fstat_search": GREEN, "rj_fstat_pe": GREEN,
                  "vgb_pe": VIOLET, "rj_prior_removal": AMBER}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    series = {}
    for t, name, tot, cnt in recs:
        picked = cnt.get("picked_sources", 0)
        blocks = max(cnt.get("inmodel_blocks", 0), 1)
        # in-model proposals ~ repeat calls x mean block size (each repeat
        # call proposes once for every source in its block)
        inm = cnt.get("inmodel_repeat_calls", 0) * (
            cnt.get("inmodel_sources", 0) / blocks)
        props = picked + inm
        series.setdefault(name, {"t": [], "rate": [], "wall": []})
        series[name]["t"].append((t - t0r).total_seconds() / 60)
        series[name]["rate"].append(props / max(tot, 1e-9))
        series[name]["wall"].append(tot)
    for name, d in sorted(series.items()):
        c = MOVE_COLOR.get(name, CYAN)
        ax[0].plot(d["t"], d["rate"], "o-", ms=3.5, lw=1.0, color=c,
                   label=f"{name} (n={len(d['t'])})")
        ax[1].plot(d["t"], d["wall"], "o-", ms=3.5, lw=1.0, color=c)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("proposals / s"); ax[0].set_xlabel("minutes since first record")
    ax[0].set_title("proposal throughput per propose"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("move wall [s]"); ax[1].set_xlabel("minutes since first record")
    ax[1].set_title("wall time per propose")
    fig_b64(fig, "timing_moves")

# gpu util CSVs (latest three jobs only -- earlier ones are archived attempts)
csvs = sorted([fn for fn in os.listdir(RUN_DIR) if fn.startswith("gpu_util")])[-3:]
if csvs:
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.0), sharex=False)
    for ci, fn in enumerate(csvs):
        rows = [l.split(",") for l in open(os.path.join(RUN_DIR, fn)) if l.strip()]
        try:
            t0 = None
            per = {}
            for r in rows:
                ts = datetime.strptime(r[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S")
                if t0 is None: t0 = ts
                gpu = int(r[1]); per.setdefault(gpu, {"t": [], "u": [], "m": []})
                per[gpu]["t"].append((ts - t0).total_seconds() / 60)
                per[gpu]["u"].append(float(r[3])); per[gpu]["m"].append(float(r[5]) / 1024)
            for gpu, d in per.items():
                c = CYAN if gpu == 0 else AMBER
                ls = "-" if ci == len(csvs)-1 else ":"
                ax[0].plot(d["t"], d["u"], color=c, ls=ls, lw=0.9,
                           label=f"{fn} gpu{gpu}")
                ax[1].plot(d["t"], d["m"], color=c, ls=ls, lw=0.9)
        except Exception:
            continue
    ax[0].set_ylabel("util [%]"); ax[0].legend(fontsize=7, ncols=2)
    ax[1].set_ylabel("mem [GiB]"); ax[1].set_xlabel("minutes since job start")
    ax[0].set_title("nvidia-smi telemetry (dotted = earlier job)")
    fig_b64(fig, "gpu_util")

# ---- 9. swaps (last ACTIVE iteration per branch: a stored iteration can
# record zero proposals for one branch at a stage handoff) ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.0))
for arrs, name, a in [((psd_sw_a, psd_sw_p), "psd", 0), ((gal_sw_a, gal_sw_p), "galfor", 1)]:
    sa, sp = arrs
    nz = np.where(sp.sum(axis=1) > 0)[0]
    k_it = int(nz.max()) if nz.size else sp.shape[0] - 1
    rate = sa[k_it] / np.maximum(sp[k_it], 1)
    ax[a].bar(np.arange(len(rate)), rate, color=CYAN if a == 0 else AMBER, alpha=0.85)
    ax[a].set_title(f"{name} swap acceptance per rung (iter {k_it})")
    ax[a].set_xlabel("rung"); ax[a].set_ylim(0, 1)
fig_b64(fig, "swaps")

# ---- 10. grouped-RJ stats + device-memory telemetry (new-code lines) -------
RJ_STATS = {}
m = re.findall(r"band unit complete after (\d+) pick rounds \((\d+) cells\)",
               log_text)
big_units = [(int(r), int(c)) for r, c in m if int(c) > 5000]
if big_units:
    RJ_STATS["cells"] = big_units[-1][1]
    RJ_STATS["rounds"] = big_units[-1][0]
m = re.findall(r"grouped in-model \S+ (\d+) flushes, mean batch ([\d.]+) "
               r"sources \((\d+) buffer slots\)", log_text)
if m:
    RJ_STATS["flushes"], RJ_STATS["batch"], RJ_STATS["slots"] = (
        int(m[-1][0]), float(m[-1][1]), int(m[-1][2]))
m = re.findall(r"at-cap skip -- (\d+) dead \(birth\) slots excluded across "
               r"(\d+) at-cap cells", log_text)
if m:
    RJ_STATS["atcap_cells"] = int(m[-1][1])

# last full rj GB_TIMING record -> breakdown bar
tm_rj = re.findall(r"\[GB_TIMING (rj_\w+)\] (total=[^|]+)\|([^|]+)\|(.*)", log_text)
if tm_rj:
    name, head, body, tail = tm_rj[-1]
    parts = dict(re.findall(r"(\w+)=([\d.]+)s", head + body))
    tot = float(parts.pop("total", 0)); parts.pop("tracked", None)
    parts.pop("untracked", None)
    top = sorted(parts.items(), key=lambda kv: -float(kv[1]))[:9]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    labels = [k for k, _ in top][::-1]
    vals = [float(v) for _, v in top][::-1]
    ax.barh(labels, vals, color=GREEN, alpha=0.9, height=0.6)
    for y_, v in enumerate(vals):
        ax.text(v, y_, f" {v:,.0f}s ({100*v/max(tot,1e-9):.0f}%)",
                va="center", fontsize=8, color=FG)
    ax.set_xlabel("seconds"); ax.set_title(
        f"{name} propose breakdown (total {tot:,.0f}s; last full record)")
    counters = dict(re.findall(r"(\w+)=(\d+)\b", tail))
    RJ_STATS["gbt_counters"] = counters
    fig_b64(fig, "rj_breakdown")

# device-memory telemetry series (buffer build / lifecycle / unit-open lines)
mem_re = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?GPU pool used ([\d.]+) / "
    r"total ([\d.]+) GB; device used/total GB: dev0 ([\d.]+)/[\d.]+, "
    r"dev1 ([\d.]+)")
mem_pts = []
for line in log_text.splitlines():
    mm_ = mem_re.match(line)
    if mm_:
        t = datetime.strptime(mm_.group(1), "%Y-%m-%d %H:%M:%S")
        mem_pts.append((t, *(float(x) for x in mm_.groups()[1:])))
if mem_pts:
    t0m = mem_pts[0][0]
    tm_min = np.array([(p[0] - t0m).total_seconds() / 60 for p in mem_pts])
    arr = np.array([p[1:] for p in mem_pts])
    # NaN-break across attempt gaps so restarts don't draw false ramps
    gaps = np.where(np.diff(tm_min) > 5.0)[0]
    for gi in gaps[::-1]:
        tm_min = np.insert(tm_min, gi + 1, np.nan)
        arr = np.insert(arr, gi + 1, np.nan, axis=0)
    tm_min = tm_min.tolist()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(tm_min, arr[:, 2], color=CYAN, lw=1.4, label="dev0 device used")
    ax.plot(tm_min, arr[:, 3], color=AMBER, lw=1.4, label="dev1 device used")
    ax.plot(tm_min, arr[:, 1], color=GREEN, lw=1.0, ls="--",
            label="cupy pool total (current dev)")
    ax.set_xlabel(f"minutes since {t0m:%m-%d %H:%M}")
    ax.set_ylabel("GB (of 99.9/device)")
    ax.legend(); ax.set_title(
        "device-wide memory from the in-run telemetry lines (memGetInfo)")
    fig_b64(fig, "mem_telemetry")

# ============================ HTML ==========================================
stage_now = "?"
for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]):
    if not s_:
        stage_now = k; break
chips = "".join(
    f'<span class="chip {"done" if s_ else ("now" if k == stage_now else "")}">'
    f'{o}. {k}{" &#10003;" if s_ else ""}</span>'
    for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]))

rj_kpis = ""
if RJ_STATS:
    c_ = RJ_STATS
    rj_kpis = f"""
<div class="kpi">
  <div><b>{c_.get('cells','?'):,}</b><span>cells / rj unit</span></div>
  <div><b>{c_.get('slots','?'):,}</b><span>buffer slots (staged)</span></div>
  <div><b>{c_.get('rounds','?'):,}</b><span>pick rounds / unit</span></div>
  <div><b>{c_.get('flushes','?')}</b><span>in-model flushes</span></div>
  <div><b>{c_.get('batch',0):,.0f}</b><span>mean flush batch [sources]</span></div>
  <div><b>{c_.get('atcap_cells',0):,}</b><span>at-cap cells skipped</span></div>
</div>"""

alert = """
<div class="alert">
<strong>JOB 187 SYNC AUTOPSY (14:25 record) &mdash; the rj black box is split, and half of
it was invisible:</strong> the old 1,461 s "rj_kernel" aggregate decomposes into
<strong>rj_getll 781 s</strong> (the scoring-call path; true device compute per the bench
is ~10&ndash;40 s of that, so &gt;90% is host/launch overhead inside get_ll) and
<strong>rj_fstat_centers 735 s</strong> &mdash; the F-stat distance-center chain for
births/deaths costs as much as ALL the scoring calls (the hidden hog). The statistics are
fully exonerated: rj_birth_prior 2.2 s, rj_score_rest 0.5 s, prior gate 6 s, accept 0.8 s.
Sync tax was only +2.5% wall. Both hogs are per-round costs, so the deployed restructure
(LAT 1fd9e08: early birth flip &rarr; ~5&times; fewer rounds; full-width in-model flush)
attacks both directly; a further lever is now visible &mdash; birth coords are pre-drawn at
sorter build, so the centers chain could be computed ONCE per propose instead of per round.
Steady elsewhere: [SAVE] 64.4 s again (~2%), info-mat cold 0.59 at jump 0.2 (0.4 next
restart), rj cold 0.0017, leaves ~88, dev0 4% / dev1 30% during rj. The single sync record
is clean and decisive &mdash; <strong>restart onto 1fd9e08 now</strong> (script already
carries jump 0.4 + no SYNC).
</div>"""

missing_html = "".join(f"<li>{m}</li>" for m in MISSING)
wanted_next = """
<li>The sbatch stdout log (<code>gf3mo_&lt;jobid&gt;.log</code>) — carries the
<code>[MAXLOGL]</code> / <code>[BENCH]</code> / stage-banner lines (per-iteration wall).</li>
<li>VGB fixed params (f0, sky) or the catalogue slice — unlocks the true frequency axis
+ truth overlays in the explorer below.</li>
<li>GB injection catalogue (f0, dist) for the truth overlay once GB births land.</li>"""

html = f"""<title>GF 3-Month Run Monitor</title>
<style>
:root {{
  --bg:#0A0E14; --panel:#10161F; --line:#223041; --fg:#B8C6D4; --dim:#67788A;
  --cyan:#4FD8EB; --amber:#F5A623; --green:#58C48A; --red:#E5484D; --violet:#9B7BFF;
}}
:root[data-theme="light"] {{
  --bg:#EEF1F5; --panel:#FFFFFF; --line:#D4DBE3; --fg:#25313D; --dim:#5D6B7A;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0; }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 20px; z-index:5; display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; }}
header h1 {{ font-size:15px; margin:0; letter-spacing:.06em; color:var(--cyan); text-transform:uppercase; }}
header .stamp {{ color:var(--dim); font-size:12px; }}
nav {{ display:flex; flex-wrap:wrap; gap:6px; padding:8px 20px; border-bottom:1px solid var(--line); }}
nav a {{ color:var(--dim); text-decoration:none; font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:3px; }}
nav a:hover, nav a:focus {{ color:var(--cyan); border-color:var(--cyan); outline:none; }}
main {{ max-width:1240px; margin:0 auto; padding:16px 20px 60px; }}
section {{ margin-top:28px; }}
h2 {{ font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--fg);
  border-bottom:1px solid var(--line); padding-bottom:6px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:12px; margin-top:12px; overflow-x:auto; }}
.panel img {{ max-width:100%; display:block; margin:0 auto; }}
.caption {{ color:var(--dim); font-size:12px; margin-top:6px; }}
.chip {{ display:inline-block; border:1px solid var(--line); border-radius:3px; padding:2px 9px; font-size:12px; color:var(--dim); }}
.chip.done {{ color:var(--green); border-color:var(--green); }}
.chip.now {{ color:var(--amber); border-color:var(--amber); }}
.kpi {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
.kpi div {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 14px; }}
.kpi b {{ display:block; font-size:18px; color:var(--cyan); font-variant-numeric:tabular-nums; }}
.kpi span {{ font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }}
.alert {{ background:color-mix(in srgb, var(--red) 12%, var(--panel)); border:1px solid var(--red);
  border-radius:4px; padding:12px 14px; margin-top:14px; font-size:13px; }}
.missing {{ color:var(--amber); border:1px dashed var(--amber); border-radius:4px; padding:10px; font-size:12px; }}
code {{ color:var(--cyan); }}
canvas {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; width:100%; height:380px; display:block; touch-action:none; cursor:grab; }}
.btnrow {{ display:flex; gap:8px; margin:8px 0; }}
button {{ background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:3px;
  padding:4px 10px; font:12px ui-monospace,monospace; cursor:pointer; }}
button:hover, button:focus {{ border-color:var(--cyan); color:var(--cyan); outline:none; }}
ul {{ color:var(--dim); font-size:13px; }}
</style>
<header>
  <h1>GF 3-Month Run Monitor</h1>
  <span class="stamp">snapshot: {os.path.basename(RUN_DIR)} &middot; generated {datetime.now():%Y-%m-%d %H:%M}</span>
  <span>{chips}</span>
</header>
<nav>
  <a href="#status">status</a><a href="#ll">likelihood</a><a href="#noise">psd + foreground</a>
  <a href="#gb">gb search</a><a href="#fstat">f-stat fit</a><a href="#vgb">vgb</a>
  <a href="#explorer">1/d explorer</a><a href="#timing">timing</a><a href="#next">next snapshot</a>
</nav>
<main>

<section id="status"><h2>Status</h2>
<div class="kpi">
  <div><b>{NIT}</b><span>iterations stored</span></div>
  <div><b>{ll[-1].max():,.1f}</b><span>max cold lnL</span></div>
  <div><b>{ll[-1].max()-ll[-1].min():.2f}</b><span>walker spread</span></div>
  <div><b>{int(gb_counts[-1].max())}</b><span>GB sources (max walker)</span></div>
  <div><b>55</b><span>VGB leaves</span></div>
  <div><b>{stage_now}</b><span>active stage</span></div>
</div>
{alert}
</section>

<section id="ll"><h2>Likelihood</h2>
<div class="panel">{img("ll")}
<div class="caption">Cold-chain total lnL. The spread panel is the tempering health check:
at equilibrium it sits at a few units.</div></div>
</section>

<section id="noise"><h2>PSD + Galactic Foreground</h2>
<div class="panel">{img("psd_curves", "PSD + foreground curves")}
<div class="caption">Instrument-only vs instrument+foreground sensitivity
(lisatools LISASens + the sampled 5-parameter hyperbolic-tangent foreground),
cold-chain walker medians, latest stored iteration.</div></div>
<div class="panel">{img("psd_evolution", "foreground evolution")}
<div class="caption">The decline watch: PSD+foreground per stored iteration
(light&rarr;dark amber = later). All stored iterations predate GB subtraction, so
this is the baseline &mdash; once gb_search iterations store, the high-frequency
shoulder of the foreground should walk DOWN as detectable sources leave the
residual.</div></div>
<div class="panel">{img("psd_trace")}<div class="caption">PSD parameter traces
(dotted red = mojito injection values).</div></div>
<div class="panel">{img("psd_hist")}</div>
<div class="panel">{img("gal_trace")}<div class="caption">Galactic-foreground parameters
(5-parameter hyperbolic-tangent model; labels are positional).</div></div>
<div class="panel">{img("gal_hist")}</div>
<div class="panel">{img("swaps")}<div class="caption">Noise-branch tempering swap acceptance
(identity naive + fancy every 10, tallied together). NOTE: the final stored iteration
recorded ZERO psd proposals (swaps and in-model) while galfor sampled fully &mdash; the
panel therefore shows each branch's last ACTIVE iteration. Whether psd stays frozen
through gb_search is an open check; if so the PSD cannot re-equilibrate as GBs are
subtracted until full_pe.</div></div>
</section>

<section id="gb"><h2>GB Search</h2>
<div class="panel">{img("gb_leaves")}
<div class="caption">Left: cold-chain GB leaf counts in the STORED iterations. Right:
per-band progressive leaf caps (D/2 gate); bands marked RED have had their RJ births shut
off by the high-frequency barren-band rule (GB_RJ_BAND_SHUTOFF_*: &gt;10 mHz default, 5
consecutive proposes with zero accepted births — deaths and in-model continue; each
shutoff is also an INFO line in the log).</div></div>
</section>

<section id="fstat"><h2>Last F-stat Fit</h2>
<div class="panel">{img("fstat_comb")}
<div class="caption">Comb scan of the maximized F-statistic over the full GB band
(epoch {fstat_meta.get("epoch","?")}, fit against the live residual, walker_ref lnL in the log).</div></div>
<div class="panel">{img("fstat_peaks")}
<div class="caption">Selected peaks = the birth-proposal anchors (cap 200/band). Job 185
loads this grid from the epoch-0 checkpoint; 9 interior sub-bands (68, 78, 82, 87, 88, 91,
93, 94, 99) fit ZERO peaks — births there ride the comb/floor components only until the
first refit against the GB-subtracted residual.</div></div>
</section>

<section id="vgb"><h2>VGB Posteriors</h2>
<div class="panel">{img("vgb_dist")}</div>
<div class="panel">
<div class="btnrow">
  <button id="vgbpost_reset">reset zoom</button>
  <span class="caption" style="align-self:center">distance-f0 posterior cloud: every
  sample (last iters &times; 24 walkers per leaf) &middot; drag = pan &middot;
  wheel/pinch = zoom</span>
</div>
<canvas id="vgbpost" style="height:340px"></canvas>
</div>
<div class="panel">{img("vgb_snr")}</div>
<div class="panel">{img("vgb_traces")}
<div class="caption">Distance traces for the three loudest VGBs, all 24 walkers.</div></div>
<div class="panel">{img("vgb_hists")}
<div class="caption">Pooled posteriors of the remaining sampled parameters.</div></div>
</section>

<section id="explorer"><h2>1/Distance vs Frequency Explorer</h2>
<div class="panel">
<div class="btnrow">
  <button id="btn_all">full band</button>
  <button id="btn_top3">3 highest-frequency sources</button>
  <button id="btn_reset">reset zoom</button>
  <span class="caption" style="align-self:center">drag = pan &middot; wheel/pinch = zoom</span>
</div>
<canvas id="expl"></canvas>
<div class="caption" id="expl_cap"></div>
</div>
</section>

<section id="timing"><h2>Timing + Memory</h2>
{rj_kpis}
<div class="caption">Grouped RJ&rarr;in-model throughput: job 185's first full-band unit
swept 43,776 cells in 1,238 s = <strong>0.028 s/cell</strong>, vs 0.080 s/cell for the same
unit pre-fix (job 177) and ~0.29 s/cell pre-grouped &mdash; the 16,384-slot staging
(8,192/GPU) plus the GIL-released kernels bought ~2.8&times; on top of the grouped-scheduler
win.</div>
<div class="panel">{img("rj_breakdown", "rj propose breakdown")}
<div class="caption">Where the rj propose spends its wall time (job 187's SYNC-ATTRIBUTED
record: every mark carries exactly its own kernel time). rj_getll and rj_fstat_centers
split the old rj_kernel aggregate ~50/50; buffer_build + temper_buffer &asymp; 780 s is
the churn lever; the statistics marks are negligible.</div></div>
<div class="panel">{img("mem_telemetry", "device memory telemetry")}
<div class="caption">Per-device used memory from the in-run memGetInfo telemetry.
Flat sawtooth = the bounded-buffer behavior; a monotonic ramp here is the leak alarm.</div></div>
<div class="panel">{img("timing_moves", "per-move timing")}</div>
<div class="panel">{img("gpu_util", "gpu telemetry")}
<div class="caption">nvidia-smi utilization + memory, latest three jobs. Job 185's rj phase:
dev0 6% / dev1 28% mean (peaks 100%) &mdash; the old 5%/57% single-device split is gone but
duty cycle is now LOW on both: the launch-width-bound kernels finish faster, leaving
per-lane Python as the residual. Next lever = batching picks into wider launches, not more
devices.</div></div>
<div class="missing">Per-iteration wall times ([MAXLOGL]/[BENCH]) go to sbatch stdout,
which was not in this zip — include <code>gf3mo_&lt;jobid&gt;.log</code> next time. [SAVE]
question ANSWERED: 65.2 s sync write vs ~56 min iteration = ~2%, below the 5–10% mpiexec
threshold — single-process stands at 3 months; revisit with the 23-mo store sizes.</div>
</section>

<section id="next"><h2>For the Next Snapshot</h2>
<ul>{missing_html}{wanted_next}</ul>
</section>
</main>

<script>
const DATA = {EXPL_JSON};
const VPOST = {VGB_POST_JSON};
(() => {{
  const cv = document.getElementById("vgbpost");
  if (!cv) return;
  const css = getComputedStyle(document.documentElement);
  const C = n => css.getPropertyValue(n).trim();
  const pts = VPOST.pts;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let X0, X1, Y0, Y1;
  const pad = (a, b) => [(a - (b - a) * 0.05), (b + (b - a) * 0.05)];
  const full = () => {{
    [X0, X1] = pad(Math.min(...xs), Math.max(...xs));
    [Y0, Y1] = pad(0, Math.max(...ys));
  }};
  full();
  const dpr = window.devicePixelRatio || 1;
  function draw() {{
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = C("--panel"); g.fillRect(0, 0, w, h);
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const sx = x => ml + (x - X0) / (X1 - X0) * (w - ml - mr);
    const sy = y => h - mb - (y - Y0) / (Y1 - Y0) * (h - mb - mt);
    g.strokeStyle = C("--line"); g.fillStyle = C("--dim"); g.font = "10px monospace";
    for (let i = 0; i <= 6; i++) {{
      const xv = X0 + (X1 - X0) * i / 6, yv = Y0 + (Y1 - Y0) * i / 6;
      g.beginPath(); g.moveTo(sx(xv), mt); g.lineTo(sx(xv), h - mb); g.stroke();
      g.beginPath(); g.moveTo(ml, sy(yv)); g.lineTo(w - mr, sy(yv)); g.stroke();
      g.fillText(xv.toPrecision(4), sx(xv) - 14, h - 12);
      g.fillText(yv.toPrecision(3), 4, sy(yv) + 3);
    }}
    g.fillText(VPOST.xlab, w / 2 - 50, h - 2);
    g.save(); g.translate(10, h / 2); g.rotate(-Math.PI / 2);
    g.fillText("dist [kpc]", -25, 0); g.restore();
    g.fillStyle = C("--violet");
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.globalAlpha = 0.45;
      g.beginPath(); g.arc(x, y, 1.4, 0, 6.29); g.fill();
    }}
    g.globalAlpha = 1;
  }}
  let drag = null;
  cv.addEventListener("pointerdown", e => {{ drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - 66) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - 38) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy; drag = [e.clientX, e.clientY]; draw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const s = e.deltaY > 0 ? 1.15 : 0.87;
    const cx = (X0 + X1) / 2, cyy = (Y0 + Y1) / 2;
    X0 = cx + (X0 - cx) * s; X1 = cx + (X1 - cx) * s;
    Y0 = cyy + (Y0 - cyy) * s; Y1 = cyy + (Y1 - cyy) * s; draw();
  }}, {{ passive: false }});
  document.getElementById("vgbpost_reset").onclick = () => {{ full(); draw(); }};
  new ResizeObserver(draw).observe(cv);
  draw();
}})();
(() => {{
  const cv = document.getElementById("expl"), cap = document.getElementById("expl_cap");
  const css = getComputedStyle(document.documentElement);
  const C = n => css.getPropertyValue(n).trim();
  const hasGB = DATA.gb.length > 0;
  // points: [x, y(=1/d), tag]
  const pts = hasGB ? DATA.gb : DATA.vgb;
  const xlab = hasGB ? "f0 [mHz]" : (DATA.vgb_axis || "VGB leaf index");
  cap.textContent = (hasGB
    ? `GB samples: ${{DATA.gb.length}} alive-source rows (last iteration, all cold walkers). Truth overlay pending catalogue in a future snapshot.`
    : `No GB sources alive yet - showing the 55 VGBs (24 walker samples each) as 1/dist vs leaf index. GB samples take over automatically once births land.`);
  let X0, X1, Y0, Y1;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = (a, b) => [(a - (b - a) * 0.05) , (b + (b - a) * 0.05)];
  const full = () => {{
    [X0, X1] = pad(Math.min(...xs), Math.max(...xs));
    [Y0, Y1] = pad(0, Math.max(...ys));
  }};
  full();
  const dpr = window.devicePixelRatio || 1;
  function draw() {{
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = C("--panel"); g.fillRect(0, 0, w, h);
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const sx = x => ml + (x - X0) / (X1 - X0) * (w - ml - mr);
    const sy = y => h - mb - (y - Y0) / (Y1 - Y0) * (h - mb - mt);
    g.strokeStyle = C("--line"); g.fillStyle = C("--dim"); g.font = "10px monospace";
    for (let i = 0; i <= 6; i++) {{
      const xv = X0 + (X1 - X0) * i / 6, yv = Y0 + (Y1 - Y0) * i / 6;
      g.beginPath(); g.moveTo(sx(xv), mt); g.lineTo(sx(xv), h - mb); g.stroke();
      g.beginPath(); g.moveTo(ml, sy(yv)); g.lineTo(w - mr, sy(yv)); g.stroke();
      g.fillText(xv.toPrecision(4), sx(xv) - 14, h - 12);
      g.fillText(yv.toPrecision(3), 4, sy(yv) + 3);
    }}
    g.fillStyle = C("--dim");
    g.fillText(xlab, w / 2 - 40, h - 2);
    g.save(); g.translate(10, h / 2); g.rotate(-Math.PI / 2);
    g.fillText("1 / dist [1/kpc]", -30, 0); g.restore();
    const col = hasGB ? C("--green") : C("--violet");
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.globalAlpha = 0.55; g.fillStyle = col;
      g.beginPath(); g.arc(x, y, 2.2, 0, 6.29); g.fill();
    }}
    g.globalAlpha = 1;
  }}
  // pan/zoom
  let drag = null;
  cv.addEventListener("pointerdown", e => {{ drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - 66) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - 38) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy; drag = [e.clientX, e.clientY]; draw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const s = e.deltaY > 0 ? 1.15 : 0.87;
    const cx = (X0 + X1) / 2, cyy = (Y0 + Y1) / 2;
    X0 = cx + (X0 - cx) * s; X1 = cx + (X1 - cx) * s;
    Y0 = cyy + (Y0 - cyy) * s; Y1 = cyy + (Y1 - cyy) * s; draw();
  }}, {{ passive: false }});
  document.getElementById("btn_all").onclick = () => {{ full(); draw(); }};
  document.getElementById("btn_reset").onclick = () => {{ full(); draw(); }};
  document.getElementById("btn_top3").onclick = () => {{
    const srt = [...pts].sort((a, b) => b[0] - a[0]);
    const top = srt.slice(0, Math.min(3 * 24, srt.length));
    const tx = top.map(p => p[0]), ty = top.map(p => p[1]);
    [X0, X1] = pad(Math.min(...tx), Math.max(...tx) || 1);
    [Y0, Y1] = pad(0, Math.max(...ty) || 1);
    draw();
  }};
  new ResizeObserver(draw).observe(cv);
  draw();
}})();
</script>
"""
open(OUT, "w").write(html)
print(f"wrote {OUT}: {len(html)//1024} KB, {len(IMGS)} plots, missing={len(MISSING)}")
