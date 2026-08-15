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

def fig_b64(fig, key, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
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

# RUN IDENTITY (2026-08-15): one generator now serves several runs (3-mo
# production, 23-mo scaling). Derive the label from the store rather than
# hard-coding it, so a 23-mo page can never be mislabelled as the 3-mo one.
_base = os.path.basename(os.path.normpath(RUN_DIR))
if "23mo" in _base:
    RUN_LABEL, RUN_KIND = "23-Month", "23mo"
elif "6mo" in _base:
    RUN_LABEL, RUN_KIND = "6-Month", "6mo"
else:
    RUN_LABEL, RUN_KIND = "3-Month", "3mo"

sub = g["sub_backend"]
psd_c = sub["psd/chain"][:NIT]                      # (it, 12, 24, 1, 2)
gal_c = sub["galfor/chain"][:NIT]                   # (it, 12, 24, 1, 5)
vgb_c = sub["vgb/chain"][:NIT, 0]                   # (it, 24, 55, 5)
vgb_hh = sub["vgb/h_h"][:NIT]                       # (it, 24, 55)
# TRAILING INCOMPLETE SUB-BACKEND ROWS (2026-08-15). The main backend and a
# sub-backend are not flushed atomically: a snapshot can hold a row where
# log_like / inds / chain are written but sub_backend/vgb/* is still all
# zeros. Taken at face value that row makes EVERY VGB look like it has zero
# amplitude -- it is what made HM Cnc, the loudest VGB in the catalogue,
# render with SNR 0. Trim the VGB arrays to their last row that carries
# actual signal; GB panels keep the full NIT because their datasets are
# complete. (Leading rows are legitimately NaN -- the VGB branch only starts
# sampling at stage 2 -- so test for "has any nonzero finite value".)
def _last_written(a):
    for i in range(a.shape[0] - 1, -1, -1):
        v = a[i]
        if np.isfinite(v).any() and np.abs(np.nan_to_num(v)).sum() > 0:
            return i + 1
    return 0

psd_sw_a = sub["psd/swaps_accepted"][:NIT]; psd_sw_p = sub["psd/swaps_proposed"][:NIT]
gal_sw_a = sub["galfor/swaps_accepted"][:NIT]; gal_sw_p = sub["galfor/swaps_proposed"][:NIT]
# EVERY sub-backend shares the flush, so one trailing row can be missing
# from all of them at once -- it is not a VGB quirk. Trim each branch to its
# own last written row (they can differ) and report it.
SUB_NIT = min(_last_written(vgb_hh), _last_written(vgb_c),
              _last_written(psd_c), _last_written(gal_c))
if SUB_NIT < NIT:
    MISSING.append(
        f"sub-backends (psd / galfor / vgb) written through iteration "
        f"{SUB_NIT - 1} while the main backend reached {NIT - 1} (snapshot "
        f"caught mid-flush); those panels use the last COMPLETE row. Taken "
        f"raw, the unwritten row reads as zero noise parameters and zero VGB "
        f"amplitudes.")
    vgb_c = vgb_c[:SUB_NIT]
    vgb_hh = vgb_hh[:SUB_NIT]
    psd_c = psd_c[:SUB_NIT]
    gal_c = gal_c[:SUB_NIT]
    psd_sw_a = psd_sw_a[:SUB_NIT]; psd_sw_p = psd_sw_p[:SUB_NIT]
    gal_sw_a = gal_sw_a[:SUB_NIT]; gal_sw_p = gal_sw_p[:SUB_NIT]
VGB_NIT = SUB_NIT
gb_inds = g["inds/gb"][:NIT, 0, 0]                  # (it, 24, 10000)
gb_chain_cold = g["chain/gb"][NIT-1, 0, 0]          # (24, 10000, 9) last iter
gb_alive_last = g["inds/gb"][NIT-1, 0, 0]           # (24, 10000)
# TORN-SNAPSHOT TOLERANCE (2026-08-15): a store copied while the run is
# mid-[SAVE] can carry truncated gzip chunks -- the dataset OPENS fine and
# only fails when READ ("filter returned failure during read"). That is a
# snapshot artifact, not a run fault, and it must not cost the whole page:
# on the v2 zip the tear was confined to sub_backend/gb/* while chain,
# inds and log_like -- i.e. every science panel -- were perfectly readable.
# Degrade per-dataset instead of dying.
_BACKUP_G = None  # lazily-opened *_running_backup_copy.h5 root group


def _backup_group():
    """The run's own backup copy, opened on demand.

    The engine keeps ``*_running_backup_copy.h5`` alongside the live store
    precisely so a torn live copy is recoverable. It lags the main file (it
    is written between saves), so it is a FALLBACK, never the default.
    """
    global _BACKUP_G
    if _BACKUP_G is None:
        _BACKUP_G = False
        for fn in os.listdir(RUN_DIR):
            if fn.endswith("_running_backup_copy.h5"):
                try:
                    _BACKUP_G = h5py.File(
                        os.path.join(RUN_DIR, fn), "r")["global_fit"]
                except Exception:
                    _BACKUP_G = False
                break
    return _BACKUP_G or None


def _safe(node, key, default=None, label=None):
    try:
        return node[key][:NIT] if NIT else node[key][()]
    except Exception as e:
        # Torn in the live copy -> try the run's backup before giving up.
        bg = _backup_group()
        if bg is not None:
            try:
                sub_key = key if node is g else "sub_backend/" + key
                # The backup is PREALLOCATED to the full run length, so slice
                # to the iterations it has actually written -- otherwise a
                # 5-row store returns 2000 rows of zeros and every plot built
                # on it is mostly empty padding.
                _bn = int(bg.attrs.get("iteration", 0))
                arr = bg[sub_key][:_bn] if _bn else bg[sub_key][()]
                MISSING.append(
                    f"{label or key}: torn in the live store, recovered from "
                    f"the run's backup copy, which holds {arr.shape[0]} "
                    f"iteration(s) vs {NIT} in the main file.")
                return arr
            except Exception:
                pass
        MISSING.append(
            f"{label or key}: unreadable in this snapshot "
            f"(likely copied mid-save) -- {type(e).__name__}")
        return default

caps = _safe(sub, "gb/band_leaf_cap", None, "per-band leaf caps")  # (it, 154)
band_edges = sub["gb/band_edges"][:]


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
        ax[j].plot(np.arange(SUB_NIT), psd_cold[:, w, j], color=CYAN, alpha=0.3, lw=0.8)
    ax[j].axhline(inj, color=RED, lw=1.4, ls=":", label="injected")
    ax[j].set_title(f"psd: {name}"); ax[j].set_xlabel("iteration"); ax[j].legend()
fig_b64(fig, "psd_trace")

fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    v = psd_cold[-min(3, SUB_NIT):, :, j].ravel()
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
        ax[j].plot(np.arange(SUB_NIT), gal_cold[:, w, j], color=AMBER, alpha=0.3, lw=0.8)
    ax[j].set_title(GAL_NAMES[j], fontsize=9); ax[j].set_xlabel("iter")
fig_b64(fig, "gal_trace")
fig, ax = plt.subplots(1, 5, figsize=(14, 2.5))
for j in range(5):
    ax[j].hist(gal_cold[-min(3, SUB_NIT):, :, j].ravel(), bins=20, color=AMBER, alpha=0.85)
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
    for k in range(SUB_NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold[k], axis=0)
        ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                color=ramp(k / max(SUB_NIT - 1, 1)), lw=1.1,
                label=f"iter {k}" if k in (0, SUB_NIT - 1) else None)
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
if caps is not None and caps.size:
    _cn = caps.shape[0]                      # may lag NIT (backup fallback)
    im = ax[1].imshow(caps.T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, _cn, 0, caps.shape[1]])
    _u = np.unique(caps[-1])
    _lab = ("cap cell" if caps.shape[1] > 200 else "band")
    ax[1].set_title(
        f"leaf cap per {_lab}"
        + (f" (ALL at {_u[0]:.0f})" if _u.size == 1 else ""))
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(_lab)
    fig.colorbar(im, ax=ax[1], shrink=0.85)
else:
    ax[1].text(0.5, 0.5, "leaf caps unreadable\n(snapshot copied mid-save)",
               ha="center", va="center", transform=ax[1].transAxes,
               color=DIM, fontsize=9)
    ax[1].set_xticks([]); ax[1].set_yticks([])
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

# ---- 5b. GB BIRTH FATE (from [GB_ACCEPT rj-split]) ----
# Every RJ propose reports where its birth proposals died. Nothing plotted
# this before, and on the v2 run it is the clearest single view of what the
# new machinery is doing: the SNR-truncated distance proposal should keep
# "snr-clamped" small (it was 59% of scored births before that lever), and
# the cap-cell grid shows up as "capped" (0 before the grid existed).
# Fates are DISJOINT and sum to the reported birth count:
#   capped / oob / prior  -> gated before scoring (cheap)
#   snr / kernel          -> scored, then dropped
#   viable-rejected       -> scored, offered to MH, rejected
#   accepted              -> became a source
RJ_SPLIT_RE = re.compile(
    r"\[GB_ACCEPT rj-split (\w+)\] births (\d+): viable (\d+) "
    r"\(acc (\d+)[^|]*\| gated: prior (\d+) oob (\d+) capped (\d+) "
    r"\| scored-dropped: snr (\d+) kernel (\d+)")
_splits = {}
for m in RJ_SPLIT_RE.finditer(log_text):
    mv = m.group(1)
    births, viable, acc, prior, oob, capped, snr, kern = (
        int(m.group(i)) for i in range(2, 10))
    if births == 0:
        continue  # removal-only moves propose no births
    _splits.setdefault(mv, []).append(
        dict(births=births, accepted=acc, gated=prior + oob + capped,
             capped=capped, snr=snr, kernel=kern,
             viable_rej=max(viable - acc, 0)))
if _splits:
    mv = max(_splits, key=lambda k: len(_splits[k]))
    rows = _splits[mv]
    x = np.arange(len(rows))
    order = [("gated (cap/oob/prior)", "gated", AMBER),
             ("scored, SNR-clamped", "snr", RED),
             ("scored, MH-rejected", "viable_rej", DIM),
             ("ACCEPTED", "accepted", GREEN)]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4))
    # left: absolute counts, stacked
    bot = np.zeros(len(rows))
    for lab, key, col in order:
        v = np.array([r[key] for r in rows], dtype=float)
        ax[0].bar(x, v, bottom=bot, color=col, label=lab, width=0.85)
        bot += v
    ax[0].set_title(f"{mv}: birth proposals by fate")
    ax[0].set_xlabel("rj propose"); ax[0].set_ylabel("proposals")
    ax[0].legend(fontsize=7, loc="upper left")
    # right: fractions, so the trend is readable as the model fills
    tot = np.array([max(r["births"], 1) for r in rows], dtype=float)
    bot = np.zeros(len(rows))
    for lab, key, col in order:
        v = np.array([r[key] for r in rows], dtype=float) / tot * 100.0
        ax[1].bar(x, v, bottom=bot, color=col, width=0.85)
        bot += v
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of births")
    ax[1].set_xlabel("rj propose")
    _last = rows[-1]
    ax[1].set_title(
        f"fate share (last: capped {_last['capped']/tot[-1]*100:.0f}%, "
        f"snr {_last['snr']/tot[-1]*100:.1f}%)")
    fig_b64(fig, "gb_birth_fate")
    GB_FATE_TXT = (
        f"Latest {mv} propose: {_last['births']:,} birth proposals -> "
        f"{_last['gated']:,} gated before scoring "
        f"({_last['capped']:,} by the cap-cell grid), "
        f"{_last['snr']:,} scored-then-SNR-clamped "
        f"({_last['snr']/tot[-1]*100:.1f}%), "
        f"{_last['viable_rej']:,} scored and MH-rejected, "
        f"{_last['accepted']:,} accepted.")
else:
    GB_FATE_TXT = ""

# ---- 6. f-stat fit ----
fdir = os.path.join(RUN_DIR, "gb_fstat_fit", "shared")
epochs = sorted([d for d in os.listdir(fdir)] if os.path.isdir(fdir) else [])
# COMPLETE epochs only (2026-08-15): an epoch directory exists as soon as the
# fit STARTS, and the 23-mo comb alone is a 1.19-billion-evaluation sweep, so
# a snapshot very often catches a half-written epoch. Requiring both cache
# files keeps a mid-fit run from killing the whole page.
epochs = [d for d in epochs
          if os.path.exists(os.path.join(fdir, d, "fstat_grid_comb.npz"))
          and os.path.exists(
              os.path.join(fdir, d, "fstat_grid_peaks_stacked.npz"))]
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
    MISSING.append(
        "No COMPLETE fstat epoch cache under gb_fstat_fit/shared -- the grid "
        "fit is still running (its epoch dir appears at fit start, the "
        "comb/peaks caches only when it finishes).")

# ---- 7. VGB ----
VGB_NAMES = ["dist [kpc]", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
# Per-leaf FIXED frequencies + names from the mojito catalogue (leaf i =
# catalogue row i, fixed-leaf branch).
VGB_F0 = None
VGB_IDS = None

MOJITO_CAT_DIR = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0")


def cat_to_sampled9(entry):
    """Catalogue columns -> the run's 9-column GB sampling basis + an
    amplitude self-check.

    Conventions are NEVER re-derived here. The physical->sampling map is
    ``recipe.gb_catalogue_to_sampling_basis`` (the single source of the
    phi0 SIGN -- sampling phi0 = -TrueAnomaly mod 2pi -- the ICRS
    ``alpha``/``sin_delta`` sky frame and the psi mod-pi wrap), and the
    ``(dist, Mc, fdot_astro_ratio)`` split follows the run's own catalogue
    path in ``stock/erebor/vgb.py``: ``dist = LuminosityDistance`` (Mpc ->
    kpc), ``Mc = ChirpMassSSBFrame``, ``r = fdot_cat / fdot_gr(f0, Mc) - 1``.
    That split reproduces the catalogue Amplitude and fdot EXACTLY (the
    returned ``rel`` is the amplitude residual, asserted small by the run's
    own VGB path).

    Returns ``(rows9, rel_max)`` with rows9 columns
    ``[dist kpc, f0 mHz, Mc, phi0, cos_iota, psi, alpha, sin_delta, ratio]``
    -- the run's ``key_order`` for the gb branch.
    """
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    from lisatools.globalfit.stock.erebor.transforms import (
        McDistFdotAstroQuad, gb_amp_from_dist)
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    d_kpc = np.asarray(entry["LuminosityDistance"], dtype=float).ravel() * 1e3
    mc = np.asarray(entry["ChirpMassSSBFrame"], dtype=float).ravel()
    f0_hz = rows[:, 1] * 1e-3
    _, _, fdot_gr, _ = McDistFdotAstroQuad()(
        d_kpc, f0_hz, mc, np.zeros_like(d_kpc))
    ratio = rows[:, 2] / fdot_gr - 1.0
    rel = float(np.abs(
        gb_amp_from_dist(f0_hz, mc, d_kpc) / np.exp(rows[:, 0]) - 1.0).max())
    return np.column_stack([d_kpc, rows[:, 1], mc, rows[:, 3], rows[:, 4],
                            rows[:, 5], rows[:, 6], rows[:, 7], ratio]), rel


VGB_TRUTH = None        # (55, 5) in the VGB sampled basis
VGB_TRUTH_REL = None
try:
    from lisatools.globalfit.stock.erebor.vgb import load_vgb_catalogue_file
    _cat = load_vgb_catalogue_file(MOJITO_CAT_DIR)
    _v = np.asarray(_cat["vgb"]).item()
    VGB_F0 = np.asarray(_v["GW22FrequencySSBFrame"]) * 1e3   # mHz
    VGB_IDS = [i.decode() if isinstance(i, bytes) else str(i)
               for i in _v["ID"]]
    # VGB sampled basis = ["dist", "phi0", "cos_iota", "psi",
    # "fdot_astro_ratio"] (VGB_SAMPLED_BASIS_DIST) = columns 0, 3, 4, 5, 8
    # of the 9-column GB basis.
    _r9, VGB_TRUTH_REL = cat_to_sampled9(_v)
    VGB_TRUTH = _r9[:, [0, 3, 4, 5, 8]]
except Exception as e:
    MISSING.append(f"VGB catalogue f0 axis unavailable locally: {e!r}")

# ---- 7b. DATA / TEMPLATE SUM / RESIDUAL, frequency + WDM domains ----------
# The one panel that answers "is the fit actually subtracting anything".
# Layout mirrors the run's OWN debug convention,
# ``gbspecialstretch.py::_debug_plot_band_sequence`` (and its addremove twin
# ``addremovemove.py::_debug_plot_source_sequence``): rows = the TDI channels
# the run analyses (X/Y/Z), columns = the three states, WDM panels rendered as
# ``imshow(|arr|, origin="lower")`` under ONE shared color scale per channel
# row so a shrinking residual renders DARK instead of autoscaling back up.
# Column order follows the request: data | template sum | residual.
#
# SOURCING (documented choice; nothing here is hand-rolled):
#   * data      -- the mojito L1 bricks the run itself loaded (NOISE + GB +
#                  VGB, from ``processor_init_kwargs`` in run_settings.log),
#                  read with the mojito reader's LAZY slicing so only the
#                  analysis window's samples ever enter memory, summed exactly
#                  as ``L1DataLoader.load_data`` sums them, then pushed
#                  through the installed ``TDSignal.fft`` / ``FDSignal
#                  .transform(WDMSettings)`` with the run's own Tukey window.
#                  The snapshot itself carries NO data/residual arrays (the
#                  only rendered one is artifacts/wdm_data.png, produced by
#                  ``engine.py``'s ``WDMSignal.heatmap`` -- an image, not
#                  numbers), so the streams must be rebuilt.
#   * templates -- the LAST stored cold-chain iteration's coordinates for the
#                  MAX-lnL walker, run through the run's own sampling->physical
#                  ``make_gb_transform_container`` and the installed
#                  ``gbgpu.gbcomps.GBFDComputations`` FD chunked-heterodyne
#                  kernel (the same kernel family the WDM analysis path uses;
#                  no phase-convention fixup, unlike the legacy fastGB path).
#   * residual  -- data - (GB + VGB), per channel, in BOTH domains.
#
# TIME-ORIGIN PHASE FACTOR: the GB/VGB branches anchor source phases at
# ``recipe.MOJITO_REFERENCE_TIME`` while the data grid starts at the brick's
# own t0, 850.5 s later. The WDM comp carries that offset internally
# (``_wdm.t0 = data_t0`` with ``t_ref = si.t0``); ``GBFDComputations`` instead
# REQUIRES ``t_start == t_ref``, so the generated template lands on a grid
# anchored at t_ref and must be advanced onto the data grid by the exact
# time-shift factor exp(+2 pi i f dt). Verified end-to-end, not assumed: with
# the CATALOGUE-TRUTH VGB parameters this route reproduces the VGB-only mojito
# brick at complex overlap 0.9999 per channel (residual power 1.8e-4 of the
# brick); without the factor the same comparison reads 0.77 with a ~90 deg
# phase.
DTR = {}
try:
    import glob as _glob
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
    from lisatools.detector import L1Orbits
    from lisatools.domains import (FDSettings, FDSignal, TDSettings, TDSignal,
                                   WDMSettings)
    from lisatools.utils.utility import windowfun
    from lisatools.globalfit.stock.erebor.transforms import (
        make_gb_transform_container)
    from lisatools.response.tdiconfig import TDIConfig
    from gbgpu.gbcomps import GBFDComputations
    from mojito import MojitoL1File

    if VGB_TRUTH is None:
        raise RuntimeError("VGB catalogue unavailable; cannot fix VGB leaves")

    # --- the run's own grid, straight off the stored domain settings ---
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    _k = dict(f["global_fit/domain_settings/kwargs"].attrs)
    W_NF, W_NT, W_DT = int(_a["0"]), int(_a["1"]), float(_a["2"])
    N_TD = W_NF * W_NT
    # window: the engine builds tukey(Nt_td, alpha=window_taper_duration/Tobs)
    _settings_txt = open(os.path.join(
        RUN_DIR, os.path.basename(RUN_DIR) + "_artifacts",
        "run_settings.log"), errors="replace").read()
    _mt = re.search(r"window_taper_duration:\s*([\d.eE+-]+)", _settings_txt)
    WIN_ALPHA = (float(_mt.group(1)) / (N_TD * W_DT)) if _mt else 0.0
    _mt0 = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", _settings_txt, re.S)
    T_REF = float(_mt0.group(1)) if _mt0 else float(MOJITO_REFERENCE_TIME)

    # --- which mojito bricks the run summed (source_ids + instrument noise) --
    _pk = re.search(r"processor_init_kwargs:\s*\{(.*)\}\s*$", _settings_txt,
                    re.M)
    _pk = _pk.group(1) if _pk else ""
    _types = [t for t in ("GB", "VGB", "MBHB", "EMRI", "SOBHB")
              if re.search(rf"'{t}':\s*\[", _pk)]
    if re.search(r"'add_instrument_noise':\s*'mojito'", _pk):
        _types = ["NOISE"] + _types
    if not _types:
        _types = ["NOISE", "GB", "VGB"]
    _SUBDIR = {"NOISE": "INSTRUMENT"}
    _files = {}
    for _t in _types:
        _hits = sorted(_glob.glob(os.path.join(
            MOJITO_CAT_DIR, "data", _SUBDIR.get(_t, _t), "L1", f"{_t}_*.h5")))
        if not _hits:
            raise FileNotFoundError(f"no local mojito {_t} L1 brick")
        _files[_t] = _hits[0]

    # --- orbits: the run's L1Orbits, but only the window's light-travel times
    class _WindowedL1Orbits(L1Orbits):
        """L1Orbits that reads only the analysis window's ltt table.

        Numerically identical to the stock class over [t0, t0 + Tobs): the
        C++ detector addresses the table as (t - ltt_t0)/ltt_dt, so dropping
        the tail past the window changes nothing that is ever evaluated. It
        avoids holding the full 731-day 25.2M x 6-link table (1.2 GB) plus
        its C++ copy for a 90-day analysis -- this generator has to run on a
        laptop.
        """
        n_ltt = int(N_TD + 20000)

        def _setup(self):
            with self.open() as _fh:
                _n = self.n_ltt
                self.ltt = _fh.ltts.ltts[:_n]
                self.ltt_t = _fh.ltts.time_sampling.t(slice(0, _n))
                self.x_base = _fh.orbits.positions[:]      # frame == "icrs"
                self.v_base = _fh.orbits.velocities[:]
                self.sc_t_base = _fh.orbits.time_sampling.t()
                self.size_base = self.sc_t_base.shape[0]
                self.dt_base = float(_fh.orbits.time_sampling.dt)
                self.ltt_dt = _fh.ltts.time_sampling.dt
                self.sc_dt = _fh.orbits.time_sampling.dt
                self.ltt_t0 = float(self.ltt_t[0])
                self.sc_t0 = float(self.sc_t_base[0])

    _orb = _WindowedL1Orbits(_files[_types[0]], force_backend="cpu",
                             frame="icrs", linear_interp_dt=500.0)
    _orb._ensure_configured()

    # --- data: partial (lazy) brick reads, summed on the analysis window ----
    _td = np.zeros((3, N_TD), dtype=np.float64)
    _t0_data = None
    for _t, _fp in _files.items():
        with MojitoL1File(_fp) as _fh:
            _chunk = _fh.tdis.xyz_doppler[:N_TD]          # lazy -> partial IO
            _td += np.asarray(_chunk).T
            del _chunk
            if _t0_data is None:
                _t0_data = float(_fh.tdis.time_sampling.t0)
    _win, _ = windowfun("tukey", N_TD, alpha=WIN_ALPHA)
    _fd_data = TDSignal(_td, TDSettings(t0=_t0_data, dt=W_DT, N=N_TD,
                                        force_backend="cpu")
                        ).fft(settings=None, window=_win)
    FDS = _fd_data.settings
    data_fd = _fd_data.arr
    del _td, _fd_data, _win

    # --- templates from the max-lnL cold walker's last stored coordinates ---
    WBEST = int(np.argmax(ll[-1]))
    _gb9 = gb_chain_cold[WBEST][gb_alive_last[WBEST]]        # (n_gb, 9)
    _v5 = vgb_c[-1, WBEST]                                   # (55, 5)
    # VGB sampled 5 columns + the 4 catalogue-FIXED ones (f0, Mc, alpha,
    # sin_delta) reassembled into the same 9-column basis the gb branch uses,
    # so ONE transform container serves both branches.
    _vgb9 = np.column_stack([_v5[:, 0], _r9[:, 1], _r9[:, 2], _v5[:, 1],
                             _v5[:, 2], _v5[:, 3], _r9[:, 6], _r9[:, 7],
                             _v5[:, 4]])
    _tf = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                      use_distance=True, mc_lims=(0.001, 1.0))
    _comp = GBFDComputations(
        FDSettings(FDS.N, FDS.df, min_freq=0.0, max_freq=None,
                   force_backend="cpu"),
        T_REF, t_start=T_REF, N_sparse=2048, orbits=_orb,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_type="XYZ", nchannels=3, force_backend="cpu",
        tukey_alpha=WIN_ALPHA, edge_frac=0.0)
    _fr = np.asarray(FDS.f_arr)
    _shift = np.exp(2j * np.pi * _fr * (_t0_data - T_REF))[None, :]
    _tmpl = {}
    for _nm, _rows in (("gb", _gb9), ("vgb", _vgb9)):
        _arr = np.zeros((1, 3, FDS.N), dtype=np.complex128)
        _comp.fill_global(_tf.both_transforms(_rows.copy()), _arr,
                          convert_to_ra_dec=False)
        _tmpl[_nm] = _arr[0] * _shift
        del _arr
    resid_fd = data_fd - _tmpl["gb"] - _tmpl["vgb"]
    DTR.update(n_gb=int(_gb9.shape[0]), n_vgb=int(_vgb9.shape[0]),
               walker=WBEST, lnl=float(ll[-1, WBEST]),
               dt_shift=float(_t0_data - T_REF))

    # --- numeric sanity: the loudest recovered GB, data vs residual ---------
    _ig = int(np.argmax(np.abs(_tmpl["gb"][0])))
    _w = slice(max(_ig - 40, 0), _ig + 41)
    DTR.update(
        chk_f0=float(_fr[_ig] * 1e3),
        chk_dpk=float(np.abs(data_fd[0, _ig])),
        chk_rpk=float(np.abs(resid_fd[0, _ig])),
        chk_dp=float(np.sum(np.abs(data_fd[:, _w]) ** 2)),
        chk_rp=float(np.sum(np.abs(resid_fd[:, _w]) ** 2)))
    _mb = (_fr >= band_edges[0]) & (_fr <= band_edges[-1])
    DTR.update(band_dp=float(np.sum(np.abs(data_fd[:, _mb]) ** 2)),
               band_rp=float(np.sum(np.abs(resid_fd[:, _mb]) ** 2)))

    # ===================== FD figure ======================================
    CHN = ["X", "Y", "Z"]
    _sel = (_fr >= 1e-4) & (_fr <= 2.6e-2)
    _fs = _fr[_sel] * 1e3
    _FLO, _FHI = band_edges[0] * 1e3, band_edges[-1] * 1e3
    _YLO = 1e-19

    def _maxdec(x, y, npts=2200):
        """Block-MAX decimation: a 1.55M-bin spectrum has to lose 99.9% of
        its points to fit a PNG, and taking every Nth bin would drop exactly
        the narrow GB lines this panel exists to show."""
        st = max(1, len(x) // npts)
        m = (len(x) // st) * st
        return (x[:m].reshape(-1, st)[:, 0],
                np.abs(y[:m]).reshape(-1, st).max(axis=1))

    _cols = [
        ("data (mojito " + " + ".join(_types) + ")",
         [("data", data_fd, DIM, 1.0, 0.9)]),
        ("template sum (GB + VGB)",
         [("data", data_fd, DIM, 0.35, 0.9), ("GB", _tmpl["gb"], GREEN, 1.0, 0.7),
          ("VGB", _tmpl["vgb"], VIOLET, 1.0, 0.7)]),
        ("residual = data - templates",
         [("data", data_fd, DIM, 0.55, 1.1),
          ("residual", resid_fd, CYAN, 1.0, 0.7)]),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 8.2), sharex=True, sharey=True)
    for r in range(3):
        for c, (ttl, series) in enumerate(_cols):
            a_ = ax[r][c]
            for lab, arr, col, al, lw in series:
                x_, y_ = _maxdec(_fs, arr[r][_sel])
                y_ = np.where(y_ > _YLO, y_, np.nan)   # no off-scale spikes
                a_.plot(x_, y_, color=col, lw=lw, alpha=al,
                        label=(lab if r == 0 else None))
            a_.set_xscale("log"); a_.set_yscale("log")
            a_.axvline(_FLO, color=RED, ls=":", lw=1.0)
            a_.axvline(_FHI, color=RED, ls=":", lw=1.0)
            if r == 0:
                a_.set_title(ttl, fontsize=10)
                # lower-left is the one corner empty in all three columns
                _lg = a_.legend(fontsize=8, loc="lower left")
                for _lh in _lg.get_lines():
                    _lh.set_linewidth(2.2)
            if c == 0:
                a_.set_ylabel(f"{CHN[r]}\n|TDI(f)|  [1/Hz]", fontsize=9)
            if r == 2:
                a_.set_xlabel("f [mHz]", fontsize=9)
    ax[0][0].set_ylim(_YLO, 3e-15)
    fig.suptitle(
        f"frequency domain - cold walker {WBEST} (max lnL), stored iteration "
        f"{NIT - 1} - dotted red = the run's GB band edges", fontsize=10,
        color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "dtr_fd")

    # ===================== WDM figure =====================================
    _wdm = WDMSettings(W_NF, W_NT, W_DT, t0=float(_k["t0"]),
                       oversample=int(_k["oversample"]),
                       min_freq=float(_k["min_freq"]),
                       max_freq=float(_k["max_freq"]),
                       min_time=float(_k["min_time"]),
                       max_time=float(_k["max_time"]),
                       is_complex=bool(_k["is_complex"]),
                       force_backend="cpu")
    DEC = 5                       # WDM time pixels pooled per plotted column
    _wp = {}
    for _nm, _arr in (("data", data_fd), ("tmpl", _tmpl["gb"] + _tmpl["vgb"]),
                      ("res", resid_fd)):
        _m = np.abs(np.asarray(FDSignal(_arr.copy(), FDS).transform(_wdm).arr))
        _n = (_m.shape[-1] // DEC) * DEC
        # MAX-pool the time axis down to plot resolution immediately; the
        # full-resolution map is never carried past this line.
        _wp[_nm] = _m[..., :_n].reshape(_m.shape[0], _m.shape[1], -1,
                                        DEC).max(axis=-1)
        del _m
    _te = np.asarray(_wdm.t_arr_edges); _fe = np.asarray(_wdm.f_arr_edges)
    _ext = [0.0, (_te[-1] - _te[0]) / 86400.0, _fe[0] * 1e3, _fe[-1] * 1e3]
    DTR.update(wdm_shape=tuple(int(s) for s in _wp["data"].shape), wdm_dec=DEC,
               layer_df=float(_wdm.layer_df), layer_dt=float(_wdm.layer_dt))
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    _tt = ["data", "template sum (GB + VGB)", "residual = data - templates"]
    for r in range(3):
        # ONE linear scale per channel row, keyed to that row's DATA panel
        # (the addremove debug convention: norm from the total-data column,
        # one per channel, shared across every frame) -- so the residual
        # panel darkens as sources leave instead of re-autoscaling.
        vmax = float(np.percentile(_wp["data"][r], 99.5))
        for c, kk in enumerate(("data", "tmpl", "res")):
            a_ = ax[r][c]
            # plotted as a FRACTION of the row scale: 0-1 ticks keep the
            # colorbar free of a floating 1e-20 offset label, and the shared
            # per-row normalization becomes explicit rather than implied.
            im = a_.imshow(_wp[kk][r] / vmax, aspect="auto", origin="lower",
                           extent=_ext, vmin=0.0, vmax=1.0, cmap="viridis",
                           interpolation="nearest")
            a_.grid(False)
            if r == 0:
                a_.set_title(_tt[c], fontsize=10)
            if c == 0:
                a_.set_ylabel(f"{CHN[r]}\nfrequency [mHz]", fontsize=9)
            if r == 2:
                a_.set_xlabel("time [days from data start]", fontsize=9)
            if c == 2:
                cb = fig.colorbar(im, ax=a_, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                cb.set_label(f"|w| / {vmax:.2e}", fontsize=7, color=DIM)
    fig.suptitle(
        f"WDM domain - |w_mn| on the run's own grid "
        f"({_wdm.Nf_active} layers x {_wdm.layer_df * 1e3:.4f} mHz, "
        f"{int(_wdm.layer_dt)} s pixels) - shared linear scale per channel row",
        fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    # 9 dense speckle images are the single most expensive PNG on the page;
    # 88 dpi still oversamples the pooled (layer x pooled-time) grid.
    fig_b64(fig, "dtr_wdm", dpi=88)
    del _wp, data_fd, resid_fd, _tmpl, _orb, _comp
except Exception as e:
    MISSING.append(
        f"data/template/residual panels unavailable: {type(e).__name__}: {e}")

vgb_last = vgb_c[-min(3, VGB_NIT):].reshape(-1, 55, 5)   # (S, 55, 5)
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
if VGB_TRUTH is not None:
    ax.plot(xs, VGB_TRUTH[:, 0], "_", ms=9, mew=1.4, color=RED, ls="none",
            label="catalogue truth")
    ax.legend(fontsize=8)
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
snr_it = np.sqrt(np.clip(np.nanmean(vgb_hh[:VGB_NIT], axis=1), 0, None))  # (it, 55)
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
# markers only (user request 2026-08-15): connecting lines between
# unrelated VGBs on a frequency axis implied a spectrum that isn't there
for k in range(VGB_NIT):
    ax.plot(_xs, snr_it[k][_of], "o", ms=3.0, ls="none",
            color=_vramp(k / max(VGB_NIT - 1, 1)), alpha=0.85,
            label=(f"iter {k}" if k in (0, VGB_NIT - 1) else None))
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
        ax[k].plot(np.arange(VGB_NIT), vgb_c[:, w, leaf, 0],
                   color=VIOLET, alpha=0.35, lw=0.8)
    if VGB_TRUTH is not None:
        ax[k].axhline(VGB_TRUTH[leaf, 0], color=RED, lw=1.4, ls=":",
                      label="catalogue truth")
        ax[k].legend(fontsize=7)
    _f0txt = f", f0={VGB_F0[leaf]:.3f} mHz" if VGB_F0 is not None else ""
    ax[k].set_title(f"{nm} (SNR~{snr[leaf]:.0f}{_f0txt}) dist", fontsize=9)
    ax[k].set_xlabel("iter")
fig_b64(fig, "vgb_traces")

# Pooled over all 55 leaves, so the truth is a DISTRIBUTION, not a line: a
# red step histogram of the 55 catalogue values on the same axis (scaled to
# the posterior's peak). fdot_astro_ratio is the exception -- every truth is
# identically 0 (GR-chirp binaries), so a single dotted line is the honest
# overlay there.
fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
for j in range(1, 5):
    a = ax[j-1]
    n_, edges_, _ = a.hist(vgb_last[:, :, j].ravel(), bins=30, color=VIOLET,
                           alpha=0.85)
    if VGB_TRUTH is not None:
        tv = VGB_TRUTH[:, j]
        if VGB_NAMES[j] == "fdot_astro_ratio":
            a.axvline(0.0, color=RED, lw=1.4, ls=":", label="truth = 0 (GR)")
        else:
            tn, te = np.histogram(tv, bins=edges_)
            a.step(te[:-1], tn * (n_.max() / max(tn.max(), 1)), where="post",
                   color=RED, lw=1.2, ls=":", label="catalogue truths (55)")
            a.plot(tv, np.full(tv.shape, -0.03 * n_.max()), "|", ms=6,
                   color=RED, alpha=0.8, clip_on=False)
        a.legend(fontsize=7)
    a.set_title(VGB_NAMES[j], fontsize=9)
fig_b64(fig, "vgb_hists")

# GB/VGB explorer data (interactive)
expl = {"gb": [], "vgb": []}
nz = np.nonzero(gb_alive_last.sum(axis=0))[0]
if nz.size:
    # f0-AMPLITUDE (user request 2026-08-15): amplitude derived from the
    # sampled (dist, f0, Mc) via the stock transform; plotted as log10(A).
    try:
        from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist
    except Exception:
        gb_amp_from_dist = None
    for w in range(nwalk):
        al = np.nonzero(gb_alive_last[w])[0]
        for i_ in al:
            row = gb_chain_cold[w, i_]
            if gb_amp_from_dist is not None:
                _amp = float(gb_amp_from_dist(
                    row[1] * 1e-3, row[2], max(row[0], 1e-6)))
                _y = float(np.log10(max(_amp, 1e-30)))
            else:
                _y = float(1.0 / max(row[0], 1e-6))
            expl["gb"].append([float(row[1]), _y, int(w)])
S = vgb_c[-1]                                        # (24, 55, 5)
for w in range(nwalk):
    for leaf in range(55):
        _x = float(VGB_F0[leaf]) if VGB_F0 is not None else int(leaf)
        expl["vgb"].append([_x, float(1.0 / max(S[w, leaf, 0], 1e-6)),
                            float(snr[leaf])])
expl["vgb_axis"] = "f0 [mHz] (catalogue)" if VGB_F0 is not None else "VGB leaf index"

# ---- GB catalogue truth cloud for the explorer (Task 3) -------------------
# The key science overlay: recovered (f0, log10 A) cloud vs the injected
# catalogue in the SAME plane reads completeness / faint tail directly.
# The 2.3 GB wdwd catalogue is opened lazily -- only the two columns needed
# for the cloud are ever pulled into memory in full.
# The band is the RUN's own gb band structure (sub_backend/gb/band_edges),
# never a hard-coded pair.
WDWD_PATH = os.path.join(MOJITO_CAT_DIR, "catalogues",
                         "wdwd_cat_mojito_lite_processed.hdf5")
TRUTH_CAP = 30000
gb_truth_pts, gb_truth_meta = [], {}
cf = f0_band = cat_gidx = None
try:
    _wd = h5py.File(WDWD_PATH, "r")
    cf = _wd["Binaries"]
    _f0_all = cf["GW22FrequencySSBFrame"][:]
    cat_gidx = np.nonzero(
        (_f0_all >= band_edges[0]) & (_f0_all <= band_edges[-1]))[0]
    f0_band = _f0_all[cat_gidx] * 1e3            # mHz, in-band catalogue
    del _f0_all
    la_band = np.log10(np.maximum(cf["Amplitude"][:][cat_gidx], 1e-30))
    gb_truth_meta["in_band"] = int(cat_gidx.size)
    if expl["gb"]:
        _rec_lo = float(min(p[1] for p in expl["gb"]))
        cut = _rec_lo - 0.5      # 0.5 dex below the faintest recovered source
        keep = np.nonzero(la_band >= cut)[0]
        gb_truth_meta.update(cut=cut, rec_lo=_rec_lo, above_cut=int(keep.size))
        if keep.size > TRUTH_CAP:
            # tiered decimation: keep EVERY truth in the bright half (where
            # the completeness statement lives), uniformly subsample the
            # faint remainder so the tail's SHAPE survives at a known,
            # quoted density.
            o = keep[np.argsort(la_band[keep])[::-1]]
            bright, rest = o[:TRUTH_CAP // 2], o[TRUTH_CAP // 2:]
            sub = np.random.default_rng(0).choice(
                rest, size=TRUTH_CAP - bright.size, replace=False)
            sel = np.concatenate([bright, sub])
            gb_truth_meta["bright_cut"] = float(la_band[bright].min())
            gb_truth_meta["faint_frac"] = float(sub.size / max(rest.size, 1))
        else:
            sel = keep
        gb_truth_pts = [[float(f"{f0_band[i]:.7g}"), round(float(la_band[i]), 4)]
                        for i in sel]
        gb_truth_meta["shown"] = len(gb_truth_pts)
        _tm = gb_truth_meta
        expl["truth_cap"] = (
            f"Catalogue truths (red): {_tm['shown']:,} points shown of "
            f"{_tm['above_cut']:,} passing the cut log10 A >= {_tm['cut']:.2f} "
            f"(0.5 dex below the faintest recovered source, {_tm['rec_lo']:.2f}); "
            f"{_tm['in_band']:,} catalogue sources lie in the GB band in total. "
            + (f"Every truth brighter than log10 A = {_tm['bright_cut']:.2f} is "
               f"kept; the fainter remainder is a uniform "
               f"{100 * _tm['faint_frac']:.1f}% random subsample -- the faint "
               f"tail's DENSITY is diluted by that factor, its shape is not."
               if "bright_cut" in _tm else "No decimation was needed."))
except Exception as e:
    MISSING.append(f"GB injection catalogue truth overlay unavailable: {e!r}")

expl["truth"] = gb_truth_pts
expl["truth_meta"] = gb_truth_meta
EXPL_JSON = json.dumps(expl)

# zoomable dist-f0 posterior cloud: every sample (last iters x walkers x leaf)
_xs_axis = VGB_F0 if VGB_F0 is not None else np.arange(55).astype(float)
vgb_post = [[float(_xs_axis[leaf]), float(v)]
            for leaf in range(55) for v in vgb_last[:, leaf, 0]]
VGB_POST_JSON = json.dumps({
    "pts": vgb_post,
    "truth": ([[float(_xs_axis[leaf]), float(VGB_TRUTH[leaf, 0])]
               for leaf in range(55)] if VGB_TRUTH is not None else []),
    "xlab": "catalogue f0 [mHz]" if VGB_F0 is not None else "VGB leaf index",
})


# ---- per-source posterior panels (Tasks 2 + 4) ---------------------------
def col_decimals(v):
    """Decimals giving ~5 significant digits OF THE COLUMN'S OWN SPREAD.

    A flat 5-significant-digit round on the absolute value would quantize a
    narrow posterior out of existence -- the highest-f GB sits at 20.3812
    mHz with a 0.002 mHz spread, which 5 significant digits collapses into
    two spikes. Precision therefore tracks the spread, not the magnitude.
    """
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return 5
    span = float(a.max() - a.min())
    if span <= 0:
        span = float(np.abs(a).max()) or 1.0
    return int(np.clip(np.ceil(-np.log10(span)) + 4, 0, 12))


def jnum(x, d):
    """JSON-safe rounded float (NaN/inf -> None)."""
    v = float(x)
    return None if not np.isfinite(v) else round(v, d)


def src_blob(label, sub, samples, truth, names, note="", note_bad=False):
    """One entry of the shared small-multiple-histogram blob."""
    dec = [col_decimals(samples[:, j]) for j in range(samples.shape[1])]
    return {
        "label": label, "sub": sub, "note": note, "bad": bool(note_bad),
        "samples": [[jnum(v, dec[j]) for v in samples[:, j]]
                    for j in range(samples.shape[1])],
        "truth": [None if truth is None else jnum(truth[j], dec[j])
                  for j in range(len(names))],
    }


# Task 2: every VGB, SNR-descending in the selector -- rendered as
# ChainConsumer CORNER plots (2026-08-15, user request), one PNG per leaf,
# base64'd into the page. The selector swaps the src of a single <img>, so
# only the selected corner is ever in the DOM's visible flow; the JS
# histogram panel (srcPanel) now serves the GB panel only.
#
# Samples: the last min(10, NIT) stored iterations x every cold walker
# (~240 rows), wider than the 3-iteration window the marginal panels use --
# a corner needs the extra rows for its 2-D contours to mean anything.
CORNER_ITS = min(10, VGB_NIT)
CORNER_DPI, CORNER_IN = 68, 7.0        # size-budget tuned (see below)
vgb_corner = vgb_c[-CORNER_ITS:].reshape(-1, 55, 5)      # (S, 55, 5)
VGB_CORNER = {"src": [], "nsamp": int(vgb_corner.shape[0]),
              "nits": int(CORNER_ITS), "nwalk": int(nwalk)}
CORNER_BYTES = []
try:
    import logging
    import warnings as _warnings

    import pandas as pd
    from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth

    # chainconsumer logs "Parameter <p> in chain ... is not constrained"
    # once per unconstrained column per leaf (hundreds of lines here, and
    # informational -- an unconstrained VGB angle is a RESULT, not a fault).
    logging.getLogger("chainconsumer").setLevel(logging.ERROR)

    def vgb_corner_png(leaf, title):
        """One leaf -> a base64 PNG of its 5x5 ChainConsumer corner.

        Extents are widened to contain the catalogue truth so a truth line
        that falls OUTSIDE the posterior is still visible (the same rule
        the JS histogram panel used); without it a badly-recovered leaf
        would silently show no truth at all.
        """
        Sm = np.asarray(vgb_corner[:, leaf, :], dtype=float)
        truth = None if VGB_TRUTH is None else VGB_TRUTH[leaf]
        ext = {}
        for j, nm_ in enumerate(VGB_NAMES):
            lo, hi = float(np.min(Sm[:, j])), float(np.max(Sm[:, j]))
            if truth is not None and np.isfinite(truth[j]):
                lo = min(lo, float(truth[j])); hi = max(hi, float(truth[j]))
            if not hi > lo:
                hi = lo + max(abs(lo) * 1e-6, 1e-12)
            pd_ = (hi - lo) * 0.06
            ext[nm_] = (lo - pd_, hi + pd_)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            cc = ChainConsumer()
            cc.add_chain(Chain(
                samples=pd.DataFrame(Sm, columns=VGB_NAMES), name="posterior",
                color=VIOLET, shade=True, shade_alpha=0.35, bar_shade=True,
                plot_point=False, smooth=1, bins=18,
                show_label_in_legend=False))
            if truth is not None:
                cc.add_truth(Truth(
                    location={nm_: float(truth[j])
                              for j, nm_ in enumerate(VGB_NAMES)},
                    color=RED, line_style=":", line_width=1.4))
            # diagonal_tick_labels defaults ON and is unreadable at this
            # dpi; 3 upright ticks per axis is what survives the size budget.
            cc.set_plot_config(PlotConfig(
                labels={nm_: nm_ for nm_ in VGB_NAMES}, extents=ext,
                label_font_size=8, tick_font_size=7, max_ticks=3,
                diagonal_tick_labels=False,
                show_legend=False, summarise=False, dpi=CORNER_DPI))
            fig_ = cc.plotter.plot(figsize=(CORNER_IN, CORNER_IN))
        fig_.suptitle(title, fontsize=9, color=FG)
        buf = io.BytesIO()
        fig_.savefig(buf, format="png", dpi=CORNER_DPI, bbox_inches="tight")
        plt.close(fig_)
        raw = buf.getvalue()
        CORNER_BYTES.append(len(raw))
        return base64.b64encode(raw).decode()

    for leaf in order:
        nm = VGB_IDS[leaf] if VGB_IDS else f"leaf {leaf}"
        _f0 = VGB_F0[leaf] if VGB_F0 is not None else float("nan")
        _lab = f"{nm} - {_f0:.4f} mHz - SNR~{snr[leaf]:.0f}"
        VGB_CORNER["src"].append({
            "label": _lab,
            "sub": (f"leaf {int(leaf)} - {vgb_corner.shape[0]} samples "
                    f"(last {CORNER_ITS} stored iterations x {nwalk} cold "
                    f"walkers) - dotted red = catalogue truth"),
            "png": vgb_corner_png(int(leaf), _lab),
        })
    print(f"[corner] {len(CORNER_BYTES)} VGB corners, "
          f"png min/med/max = {min(CORNER_BYTES)/1024:.1f} / "
          f"{np.median(CORNER_BYTES)/1024:.1f} / {max(CORNER_BYTES)/1024:.1f} KB, "
          f"total png {sum(CORNER_BYTES)/1024**2:.2f} MB "
          f"(base64 {sum(CORNER_BYTES)*4/3/1024**2:.2f} MB) "
          f"@ dpi={CORNER_DPI}, figsize={CORNER_IN}")
except Exception as e:
    MISSING.append(f"VGB ChainConsumer corner plots unavailable: {e!r}")
VGB_CORNER_JSON = json.dumps(VGB_CORNER)

# Task 4: the 3 highest-frequency RECOVERED GBs.
# Tobs from the stored domain settings (WDMSettings args = Nt, Nf, dt) ->
# the FD bin width that sets both the cluster window and the Delta-f0 unit.
TOBS = 3 * 30 * 86400.0
try:
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    TOBS = float(_a["0"]) * float(_a["1"]) * float(_a["2"])
except Exception as e:
    MISSING.append(f"Tobs not readable from domain_settings ({e!r}); "
                   "using 90 d for the f0 bin width.")
DF_MHZ = 1e3 / TOBS                       # one FD bin, in mHz
CLUSTER_BINS, MATCH_BINS = 20.0, 100.0
GB_NAMES = ["dist [kpc]", "f0 [mHz]", "Mc [Msol]", "phi0", "cos_iota",
            "psi", "alpha", "sin_delta", "fdot_astro_ratio"]
GB1 = {"params": GB_NAMES, "src": []}
gb1_meta = {}
_rows = sorted(((float(gb_chain_cold[w, i, 1]), w, i) for w in range(nwalk)
                for i in np.nonzero(gb_alive_last[w])[0]), key=lambda r: -r[0])
if _rows:
    clusters, cur = [], [_rows[0]]
    for r in _rows[1:]:
        if cur[-1][0] - r[0] <= CLUSTER_BINS * DF_MHZ:
            cur.append(r)
        else:
            clusters.append(cur); cur = [r]
    clusters.append(cur)
    # A "recovered source" must live in at least 3 of the 24 cold walkers;
    # 1-2-walker clusters at the top of the band are transient births, not
    # sources (their count is quoted in the caption -- it is itself a
    # readout of high-f birth churn).
    solid = [c for c in clusters if len({x[1] for x in c}) >= 3]
    gb1_meta["n_clusters"] = len(clusters)
    gb1_meta["n_solid"] = len(solid)
    gb1_meta["transient_above"] = (
        len([c for c in clusters
             if c[0][0] > solid[0][0][0] and len({x[1] for x in c}) < 3])
        if solid else len(clusters))
    for c in solid[:3]:
        P = np.array([gb_chain_cold[w, i] for _, w, i in c])
        f0_med = float(np.median(P[:, 1]))
        truth, note, bad = None, "", False
        if f0_band is not None and f0_band.size:
            j = int(np.argmin(np.abs(f0_band - f0_med)))
            d_bins = (f0_med - f0_band[j]) / DF_MHZ
            if abs(d_bins) <= MATCH_BINS:
                gidx = int(cat_gidx[j])   # row in the full 15.5M catalogue
                entry = {k: np.atleast_1d(float(cf[k][gidx]))
                         for k in ("Amplitude", "GW22FrequencySSBFrame",
                                   "GW22FrequencyDerivativeSourceFrame",
                                   "TrueAnomaly", "InclinationAngle",
                                   "PolarisationAngle", "RightAscension",
                                   "Declination", "LuminosityDistance",
                                   "ChirpMassSSBFrame")}
                truth = cat_to_sampled9(entry)[0][0]
                from lisatools.globalfit.stock.erebor.transforms import (
                    gb_amp_from_dist as _amp)
                a_rec = float(np.median(_amp(P[:, 1] * 1e-3, P[:, 2],
                                            np.maximum(P[:, 0], 1e-6))))
                a_cat = float(entry["Amplitude"][0])
                n_near = int(np.sum(np.abs(f0_band - f0_med)
                                    <= MATCH_BINS * DF_MHZ))
                note = (f"catalogue match ID {int(cf['ID'][gidx])}: "
                        f"df0 = {d_bins:+.1f} bins ({(f0_med - f0_band[j])*1e3:+.3f} "
                        f"uHz), A_rec/A_cat = {a_rec / a_cat:.2f}, "
                        f"{n_near} catalogue source(s) within "
                        f"{MATCH_BINS:.0f} bins")
            else:
                note = (f"NO catalogue match within {MATCH_BINS:.0f} bins "
                        f"(nearest is {d_bins:+.0f} bins away) - this "
                        f"recovery has no injected counterpart")
                bad = True
        GB1["src"].append(src_blob(
            f"GB @ {f0_med:.5f} mHz (n={len(c)} samples, "
            f"{len({x[1] for x in c})} walkers)",
            f"cold-chain iteration {NIT - 1}; cluster window "
            f"{CLUSTER_BINS:.0f} FD bins = {CLUSTER_BINS * DF_MHZ * 1e3:.2f} uHz",
            P, truth, GB_NAMES, note, bad))
GB1_JSON = json.dumps(GB1)

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
# direct-batch mode (GB_RJ_DIRECT_BATCH, 2026-08-14): rigid rj batches +
# one end-of-unit in-model phase in capacity chunks
m = re.findall(r"direct batches \S+ (\d+) rj batch\(es\), (\d+) survivors "
               r"polished in (\d+) in-model chunk\(s\) \((\d+) buffer slots\)",
               log_text)
if m:
    RJ_STATS["flushes"] = int(m[-1][2])
    RJ_STATS["batch"] = int(m[-1][1]) / max(int(m[-1][2]), 1)
    RJ_STATS["slots"] = int(m[-1][3])
m = re.findall(r"at-cap skip -- (\d+) dead \(birth\) slots excluded across "
               r"(\d+) at-cap cells", log_text)
if m:
    RJ_STATS["atcap_cells"] = int(m[-1][1])

# last full rj GB_TIMING record PER MOVE -> one breakdown bar panel each.
# Before 2026-08-15 this took tm_rj[-1] only, so whichever rj move happened
# to log LAST (rj_prior_removal, 75 s) was the only breakdown on the page
# and rj_fstat_search (1,041 s, the actual hog) never appeared. Every rj
# move now gets its OWN panel, so the moves are never lumped or shadowed.
tm_rj = re.findall(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?\[GB_TIMING (rj_\w+)\] "
    r"(total=[^|]+)\|([^|]+)\|(.*)$", log_text, re.M)
RJ_LAST = {}                       # move -> (stamp, head, body, tail)
for stamp, name, head, body, tail in tm_rj:
    RJ_LAST[name] = (stamp, head, body, tail)
# rj_fstat_search (the F-stat birth proposal) leads, then rj_prior_removal;
# any other rj move follows in first-seen order.
RJ_ORDER = [n for n in ("rj_fstat_search", "rj_prior_removal") if n in RJ_LAST]
RJ_ORDER += [n for n in RJ_LAST if n not in RJ_ORDER]
RJ_MOVE_COLOR = {"rj_fstat_search": GREEN, "rj_fstat_pe": GREEN,
                 "rj_prior_removal": AMBER}
# run_proposal / run_tempering are ENCLOSING phase marks -- they contain the
# leaf spans, so leaving them in the bars just reprints the total (1,037 of
# 1,041 s for rj_fstat_search) and buries the actual hog. They are quoted in
# each panel's title instead; the bars are leaf spans only.
RJ_WRAPPERS = ("run_proposal", "run_tempering")
RJ_BREAK = {}                      # move -> (stamp, total, [(span, s), ...])
if RJ_ORDER:
    NTOP = 9
    fig, axs = plt.subplots(1, len(RJ_ORDER), figsize=(6.0 * len(RJ_ORDER), 3.6))
    axs = np.atleast_1d(axs)
    for a_, name in zip(axs, RJ_ORDER):
        stamp, head, body, tail = RJ_LAST[name]
        parts = dict(re.findall(r"(\w+)=([\d.]+)s", head + body))
        tot = float(parts.pop("total", 0)); parts.pop("tracked", None)
        parts.pop("untracked", None)
        wrap = {w: float(parts.pop(w)) for w in RJ_WRAPPERS if w in parts}
        top = sorted(parts.items(), key=lambda kv: -float(kv[1]))[:NTOP]
        RJ_BREAK[name] = (stamp, tot, [(k, float(v)) for k, v in top])
        labels = [k for k, _ in top][::-1]
        vals = [float(v) for _, v in top][::-1]
        a_.barh(labels, vals, color=RJ_MOVE_COLOR.get(name, CYAN), alpha=0.9,
                height=0.6)
        for y_, v in enumerate(vals):
            a_.text(v, y_, f"  {v:,.1f}s ({100 * v / max(tot, 1e-9):.0f}%)",
                    va="center", fontsize=8, color=FG)
        a_.set_xlim(0, max(vals + [1.0]) * 1.34)
        a_.tick_params(axis="y", labelsize=8)
        a_.set_xlabel("seconds", fontsize=9)
        _wtxt = ", ".join(f"{w} {v:,.1f}s" for w, v in wrap.items())
        a_.set_title(f"{name}: total {tot:,.1f}s (last full record, "
                     f"{stamp[5:]})\nleaf spans only; enclosing: "
                     f"{_wtxt or 'none'}", fontsize=9)
    fig.tight_layout()
    # counters of the last record overall (unchanged behavior)
    RJ_STATS["gbt_counters"] = dict(
        re.findall(r"(\w+)=(\d+)\b", tm_rj[-1][4]))
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

    def _kpi(key, dec=0):
        """Thousands-separated KPI, or an em dash when the log lacks it.

        A missing key used to reach an f-string ``{...:,}`` as the string
        "?", which raises ValueError ("Cannot specify ',' with 's'") and
        killed the whole page. Snapshots legitimately miss KPIs -- an early
        fresh run has no completed RJ unit yet -- so degrade, never crash.
        """
        v = c_.get(key)
        if v is None:
            return "&mdash;"
        try:
            return f"{float(v):,.{dec}f}"
        except (TypeError, ValueError):
            return str(v)

    rj_kpis = f"""
<div class="kpi">
  <div><b>{_kpi('cells')}</b><span>cells / rj unit</span></div>
  <div><b>{_kpi('slots')}</b><span>buffer slots (staged)</span></div>
  <div><b>{_kpi('rounds')}</b><span>pick rounds / unit</span></div>
  <div><b>{_kpi('flushes')}</b><span>in-model flushes</span></div>
  <div><b>{_kpi('batch')}</b><span>mean flush batch [sources]</span></div>
  <div><b>{_kpi('atcap_cells')}</b><span>at-cap cells skipped</span></div>
</div>"""

# one-line-per-move readout under the breakdown panel (built from the SAME
# records the figure is built from -- never hand-typed numbers)
rj_break_txt = ""
rj_break_stamps = ""
if RJ_BREAK:
    _st = sorted({v[0] for v in RJ_BREAK.values()})
    rj_break_stamps = f" ({_st[0]} &ndash; {_st[-1]})" if _st else ""
    rj_break_txt = "<br>" + "<br>".join(
        f"<code>{n}</code>: total <strong>{RJ_BREAK[n][1]:,.1f} s</strong>, "
        f"top span <code>{RJ_BREAK[n][2][0][0]}</code> "
        f"{RJ_BREAK[n][2][0][1]:,.1f} s "
        f"({100 * RJ_BREAK[n][2][0][1] / max(RJ_BREAK[n][1], 1e-9):.0f}%)"
        for n in RJ_ORDER) + "<br>"

_alert_3mo = """
<div class="alert">
<strong>FRESH RUN (gf_prod_3mo_v2) &mdash; the rebuilt stack, measured from iteration 0.</strong>
Everything below is a NEW baseline: new store, all state built from config, F-stat grid and
epoch center table fitted against this run&rsquo;s own residual, and the VGB likelihood
working for the first time. Headline: <strong>the iteration wall fell from 53&ndash;77 min
to ~11 min while carrying MORE signal</strong> (160 GB leaves/cold walker at iteration 9 vs
~140 at iteration 19 of the old run), and cold lnL reached <strong>52,586,123 by iteration
9</strong> &mdash; past where the previous run sat at iteration 19.
<br><br>
<strong>Why the leaf count climbs so fast now: the cap-cell grid removed a ceiling the old
run was pressed against.</strong> Leaf caps used to be applied per SUB-BAND, a width chosen
for COMPUTE, not for how close two sources can sit before they are confused. With 154 bands
at a mean cap of 1.47, the old run's whole-model ceiling was <strong>227 leaves per
walker</strong> &mdash; and it was sitting at 137, i.e. 60% of the way into it, with more
room only arriving as the slow per-band cap increments fired. Caps now live on an
INDEPENDENT band/8 grid at the confusion scale (1,232 cells), so a band holds up to 8&times;
what it did and the model is free to populate at the rate the data actually supports:
0 &rarr; 86 &rarr; 121 &rarr; 144 &rarr; 160 leaves/walker over iterations 4&ndash;8. The
growth is the throttle coming off, not a birth pathology &mdash; and the per-cell cap still
does the job the band cap was standing in for, visible as the amber band in the birth-fate
panel below.
<br><br>
<strong>What each change actually bought.</strong> (1) EPOCH CENTERS: the 953 s/propose
F-stat centre chain &mdash; 92% of the old search propose &mdash; is now a
<strong>529,641-node table built once per epoch in 0.5 s</strong>, with ZERO lookup
fallbacks; <code>rj_fstat_centers</code> has vanished from the breakdown entirely and
<code>rj_fstat_search</code> is <strong>137&ndash;151 s (was 1,041 s)</strong>. (2) SAVER
RANK: <code>[SAVE]</code> 60 s &rarr; <strong>4 s</strong> (&ldquo;handoff to rank 2&rdquo;)
&mdash; the dedicated mpiexec saver works. (3) JUMP 1.2: in-model cold acceptance
<strong>0.349</strong>, squarely in the 0.15&ndash;0.4 target (it was 0.71&ndash;0.80 at
0.6). (4) SNR-TRUNCATED BIRTHS: only <strong>6.2%</strong> of births now die at the SNR
clamp, against 59% before the lever. (5) GPU BALANCE: dev0 19% / dev1 18% mean, versus the
3%/79% single-device signature that dominated the old run.
<br><br>
<strong>Correctness &mdash; the top watch is resolved.</strong> Cold-rung
<code>[GB_CELL_LL]</code> credit discrepancies are now ~1e-4/rep against allowances of
5e-2&ndash;1e-1; the old run was running 20&ndash;60/rep at temp 0. The new
<code>[GB_ORTHO_LL]</code> bilinearity monitor sits at <strong>1e-6&ndash;2e-5</strong>
across every unit, confirming that per-buffer likelihood deltas sum to the realized
residual change. Hot rungs (temp 4&ndash;10) still show excesses, which remains parked.
<br><br>
<strong>THE NEW BOTTLENECK IS vgb_pe.</strong> One iteration is ~673 s and breaks down as
<code>rj_fstat_search</code> ~151 s + <code>rj_prior_removal</code> ~127 s + a noise+VGB
block of ~542 s made of <strong>18 rounds &times; ~30 s</strong>. Within each round
<code>vgb_pe</code> is ~19 s, so <strong>VGB sampling alone is ~340 s &mdash; about half the
iteration</strong>, and psd+galfor together are only ~11 s/round. The cause is structural,
not a bug: <code>VGB_NTEMPS=8</code> replaced what was effectively a 1-rung ladder, so the
VGB buffer is 8&times; larger (2,304 slots) and rebuilt every round &mdash; inside
<code>vgb_pe</code> the top spans are <code>run_tempering</code> 8.7 s,
<code>bufbuild_alloc</code> 7.0 s and <code>temper_buffer</code> 6.3 s. This is the correct
sampling we previously were not doing at all; the question is now whether 8 rungs and 18
rounds per iteration are both needed for 55 KNOWN, unimodal sources.
<br><br>
<strong>Open items.</strong> The cap-cell grid is actively gating &mdash; 353k of 1.15M
births blocked as <code>capped</code> (it was 0 before), which is the intended
anti-stacking behaviour but deserves a look at whether it is too aggressive.
<code>[GB_CAP_LL]</code> emitted ZERO lines despite being armed, so that monitor needs
debugging. Per-process GPU telemetry answers an old question: <strong>all three MPI ranks
hold ~2.3 GB each</strong>, so the saver/spare ranks really do allocate GPU memory and the
rank-gated build is worth building. Finally, <code>GF_MOVE_TIMING</code> writes to sbatch
stdout rather than the run log, so <code>gf3mo_&lt;jobid&gt;.log</code> is needed to split
psd_pe from galfor_pe directly.
</div>"""

_alert_23mo = f"""
<div class="alert">
<strong>23-MONTH SCALING RUN &mdash; the same stack, 7.8&times; the grid.</strong> This page
is the exact 3-month monitor pointed at <code>gf_prod_23mo</code>; read it as a SCALING
readout, not a science posterior. Tobs = 6.048e7 s (Nf 1440 &times; Nt 16800), so the WDM
residual is ~0.58 GB per walker against ~0.075 GB at 3 months, and the run carries the
23-mo sizing (<code>GB_N_SUBBANDS</code> 2048/GPU, <code>GB_NLEAVES_MAX</code> 25000,
<code>FSTAT_PEAKS_PER_BAND</code> 500, <code>SIGHET_NT_LAYER</code> 525) with every
Tobs-independent improvement from the 3-mo work.
<br><br>
<strong>Where it is.</strong> {NIT} stored iteration(s); cold lnL {ll[NIT-1].max():,.0f}.
The GB panels are EMPTY BY CONSTRUCTION so far &mdash; the staged recipe runs
noise_search &rarr; noise_vgb_search &rarr; gb_search, and this run is still in the noise +
VGB stages while the F-stat grid is fitted. That fit is the headline scaling number: the
comb is a <strong>1.19-billion-evaluation</strong> level-5 sweep here (2.3M nodes &times;
512 sky) against 38.7M at 3 months, which is the ~8&times; Tobs scaling the design
predicted, landing on top of an already-8&times; per-evaluation grid.
<br><br>
<strong>What to compare against the 3-month page.</strong> Per-source sig-het scoring is
FLAT in Tobs by design (the v4/v5 bench-off), so the GB propose spans should NOT scale with
Tobs once gb_search opens &mdash; buffer fills, band likelihoods and the F-stat fit are the
terms that do. The VGB branch is a clean early check: 55 fixed sources at 8 ladder rungs,
identical to the 3-mo configuration, so any <code>vgb_pe</code> cost difference is pure
Tobs scaling of the fills. Watch device memory against the 96 GB cards: the 23-mo per-slot
slab is ~8&times; the 3-mo one, which is exactly why the buffer is sized at 2048/GPU here
instead of 8192.
</div>"""

alert = _alert_23mo if RUN_KIND == "23mo" else _alert_3mo

# ---- data/template/residual captions (numbers come from the arrays) -------
if DTR:
    _d = DTR
    dtr_fd_cap = (
        f"Rows = the TDI channels the run analyses (X / Y / Z); columns = "
        f"<strong>data</strong> | <strong>template sum</strong> | "
        f"<strong>residual</strong> &mdash; the same 3-column layout the run's "
        f"own debug plots use (<code>gbspecialstretch.py::"
        f"_debug_plot_band_sequence</code>). Amplitude spectra, log&ndash;log, "
        f"block-MAX decimated to {2200} points per curve (taking every Nth bin "
        f"instead would drop exactly the narrow GB lines this panel exists to "
        f"show). Dim grey = data, repeated faintly under the template and "
        f"residual columns so the comparison is direct; <span style="
        f"'color:var(--green)'>green</span> = the {_d['n_gb']} live GB leaves, "
        f"<span style='color:var(--violet)'>violet</span> = the {_d['n_vgb']} "
        f"VGB leaves, <span style='color:var(--cyan)'>cyan</span> = the "
        f"residual. Dotted red = the run's own GB band edges "
        f"(<code>sub_backend/gb/band_edges</code>). Curves are clipped at "
        f"1e-19; the template is zero between sources, so its line simply "
        f"stops there rather than diving off-scale."
        f"<br><br><strong>Numeric check &mdash; the subtraction is real.</strong> "
        f"At the loudest recovered GB, f0 = {_d['chk_f0']:.5f} mHz, the X-channel "
        f"peak bin goes from |data| = {_d['chk_dpk']:.3e} to |residual| = "
        f"{_d['chk_rpk']:.3e} (a factor {_d['chk_dpk']/max(_d['chk_rpk'],1e-99):.0f} "
        f"down), and over the &plusmn;40-bin window around it the 3-channel power "
        f"falls {_d['chk_dp']:.3e} &rarr; {_d['chk_rp']:.3e}, i.e. "
        f"{100*_d['chk_rp']/max(_d['chk_dp'],1e-99):.1f}% of the data power left. "
        f"Across the WHOLE GB band, by contrast, only "
        f"{100*(1 - _d['band_rp']/max(_d['band_dp'],1e-99)):.1f}% of the power "
        f"is removed ({100*_d['band_rp']/max(_d['band_dp'],1e-99):.1f}% remains) "
        f"&mdash; expected at stored iteration {NIT-1}, which is still in the "
        f"noise/VGB stages with only {_d['n_gb']} GB leaves alive out of a "
        f"whole confusion foreground.")
    dtr_wdm_cap = (
        f"The same three states on the run's OWN WDM grid, read straight off "
        f"<code>global_fit/domain_settings</code>: {_d['wdm_shape'][1]} active "
        f"layers of {_d['layer_df']*1e3:.4f} mHz &times; "
        f"{int(_d['layer_dt'])} s time pixels, MAX-pooled "
        f"{_d['wdm_dec']}&times; along time to plot resolution "
        f"({_d['wdm_shape'][2]} columns) before anything is drawn. Panels are "
        f"|w<sub>mn</sub>| under <strong>one shared LINEAR color scale per "
        f"channel row</strong>, keyed to the 99.5th percentile of that row's "
        f"DATA panel &mdash; the run's own convention (the addremove debug "
        f"plot takes its norm from the total-data column, one per channel, "
        f"shared across every frame). Shared-per-row is the right choice here: "
        f"it is what makes a shrinking residual render DARKER instead of "
        f"autoscaling itself back to full brightness, which a per-panel scale "
        f"would do. The template column's horizontal tracks with their annual "
        f"brightness modulation are the recovered sources; read the residual "
        f"panel against the data panel at those same layers.")
    dtr_note = (
        f"Both figures are built for cold walker <strong>{_d['walker']}</strong> "
        f"&mdash; the MAX-lnL walker of stored iteration {NIT-1} "
        f"(lnL = {_d['lnl']:,.1f}), not walker 0. Data = the mojito L1 bricks "
        f"the run itself loaded, re-poured through the installed "
        f"<code>TDSignal.fft</code> / <code>FDSignal.transform(WDMSettings)</code> "
        f"with the run's Tukey window; templates = the stored cold-chain "
        f"coordinates through the run's <code>make_gb_transform_container</code> "
        f"and the installed <code>gbgpu.gbcomps.GBFDComputations</code> kernel "
        f"(the FD twin of the WDM chunked-heterodyne comp the GB branch "
        f"analyses with). The GB/VGB branches anchor source phase at "
        f"<code>MOJITO_REFERENCE_TIME</code> while the data grid starts "
        f"{_d['dt_shift']:.1f} s later, and the FD comp requires "
        f"<code>t_start == t_ref</code>, so the template is advanced onto the "
        f"data grid by exp(+2&pi;i&middot;f&middot;&Delta;t); that route was "
        f"VERIFIED, not assumed &mdash; at catalogue-truth VGB parameters it "
        f"reproduces the VGB-only mojito brick at complex overlap 0.9999 per "
        f"channel (residual power 1.8e-4 of the brick). <strong>psd and galfor "
        f"are NOT in the template sum</strong>: they are noise-model branches "
        f"&mdash; they shape the sensitivity the likelihood weights by, they "
        f"are not subtracted from the data, so the unresolved galaxy stays in "
        f"the residual by construction.")
else:
    dtr_fd_cap = dtr_wdm_cap = dtr_note = ""

missing_html = "".join(f"<li>{m}</li>" for m in MISSING)
wanted_next = """
<li>The sbatch stdout log (<code>gf3mo_&lt;jobid&gt;.log</code>) — carries the
<code>[MAXLOGL]</code> / <code>[BENCH]</code> / stage-banner lines (per-iteration wall).</li>
<li>Per-iteration <code>[FSTAT_CTR]</code> census lines across a longer stretch — the
two-state per-row center cost (1.64 vs 0.31 ms/row) needs several proposes to correlate
against cell counts / caps.</li>"""

html = f"""<title>GF {RUN_LABEL} Run Monitor</title>
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
button.armed {{ border-color:var(--amber); color:var(--amber); }}
.viewctl {{ flex-wrap:wrap; align-items:center; gap:6px 12px; }}
.viewctl label {{ color:var(--dim); font-size:11px; display:inline-flex; align-items:center; gap:4px;
  text-transform:uppercase; letter-spacing:.04em; }}
.viewctl input[type=text] {{ background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:3px; font:11px ui-monospace,monospace; padding:2px 4px; width:86px; }}
.viewctl input[type=text]:focus {{ border-color:var(--cyan); outline:none; }}
.viewctl input[type=range] {{ width:110px; accent-color:var(--cyan); }}
.viewctl select {{ background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:3px; font:11px ui-monospace,monospace; padding:2px 4px; max-width:520px; }}
.viewctl select:focus {{ border-color:var(--cyan); outline:none; }}
button.armed.truth {{ border-color:var(--red); color:var(--red); }}
ul {{ color:var(--dim); font-size:13px; }}
</style>
<header>
  <h1>GF {RUN_LABEL} Run Monitor</h1>
  <span class="stamp">snapshot: {os.path.basename(RUN_DIR)} &middot; generated {datetime.now():%Y-%m-%d %H:%M}</span>
  <span>{chips}</span>
</header>
<nav>
  <a href="#status">status</a><a href="#dtr">data / template / residual</a>
  <a href="#ll">likelihood</a><a href="#noise">psd + foreground</a>
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

<section id="dtr"><h2>Data / Template / Residual</h2>
<div class="panel">{img("dtr_fd", "data / template / residual, frequency domain")}
<div class="caption">{dtr_fd_cap}</div></div>
<div class="panel">{img("dtr_wdm", "data / template / residual, WDM domain")}
<div class="caption">{dtr_wdm_cap}</div></div>
<div class="caption">{dtr_note}</div>
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
<div class="panel">{img("gb_birth_fate", "birth-fate breakdown")}
<div class="caption">Where every RJ birth proposal ends up &mdash; the clearest single view
of the new machinery, and nothing plotted it before. The fates are disjoint and sum to the
proposed count. <strong>AMBER</strong> = gated before scoring (cap-cell grid, out-of-band,
prior): cheap rejections that never touch a likelihood kernel, and dominated by the new
band/8 cap grid, which blocks a birth whose own cell is already full &mdash; this was
exactly 0 before the grid existed, so a large amber band is the anti-stacking rule
working, not a fault. <strong>RED</strong> = scored, then dropped at the optimal-SNR
clamp; the SNR-truncated distance proposal exists to shrink this, and it fell from 59% of
scored births to a few percent. <strong>GREY</strong> = scored, offered to
Metropolis&ndash;Hastings, rejected. <strong>GREEN</strong> = accepted, i.e. a new source.
Left is absolute counts, right the same data as percentages so the trend stays readable as
the model fills. {GB_FATE_TXT}</div></div>
<div class="panel">{img("gb_leaves")}
<div class="caption">Left: cold-chain GB leaf counts in the STORED iterations. Right:
per-band progressive leaf caps (D/2 gate). NOTE (2026-08-15): leaf caps now live on a
FINER grid than this panel shows &mdash; each band is split into GB_CAP_DIVISOR (8) cap
cells at the confusion scale, so a band's displayed cap is the MAX over its cells and the
band can hold up to 8x that in total. Bands marked RED have had their RJ births shut
off by the high-frequency barren-band rule (GB_RJ_BAND_SHUTOFF_*: &gt;10 mHz default, 5
consecutive proposes with zero accepted births — deaths and in-model continue; each
shutoff is also an INFO line in the log).</div></div>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="gb1_sel"></select></label>
  <span class="caption" style="align-self:center">posteriors of the 3 highest-frequency
  recovered GBs</span>
</div>
<canvas id="gb1" style="height:330px"></canvas>
<div class="caption" id="gb1_cap"></div>
<div class="caption">Sources are clustered out of the last stored cold-chain iteration by
f0 ({CLUSTER_BINS:.0f} FD bins = {CLUSTER_BINS * DF_MHZ * 1e3:.2f} &micro;Hz link window,
1/T<sub>obs</sub> = {DF_MHZ * 1e3:.3f} &micro;Hz); a cluster counts as a SOURCE only if it
appears in &ge;3 of the {nwalk} cold walkers. {gb1_meta.get("n_solid", 0)} of
{gb1_meta.get("n_clusters", 0)} clusters clear that bar, and
{gb1_meta.get("transient_above", 0)} single-/two-walker clusters sit ABOVE the highest
one shown &mdash; transient high-f births, not recoveries. Truth = the nearest catalogue
source in f0 within {MATCH_BINS:.0f} bins, mapped through the same
<code>gb_catalogue_to_sampling_basis</code> route as the VGBs. CAVEAT on
(dist, Mc, fdot_astro_ratio): the likelihood constrains only the combinations
<code>A(dist, Mc, f0)</code> and <code>fdot_gr(f0, Mc)&middot;(1+r)</code>, so the truth
shown is the catalogue's own physical split &mdash; a posterior displaced ALONG that
degeneracy is not an error.</div>
</div>
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
<div class="btnrow viewctl">
  <button id="vgbpost_pick" title="arm, then click the plot to set the view center">set center by click</button>
  <label>cx <input id="vgbpost_cx" type="text"></label>
  <label>cy <input id="vgbpost_cy" type="text"></label>
  <label>width <input id="vgbpost_wsl" type="range" min="0" max="1000" step="1"><input id="vgbpost_w" type="text"></label>
  <label>height <input id="vgbpost_hsl" type="range" min="0" max="1000" step="1"><input id="vgbpost_h" type="text"></label>
</div>
<canvas id="vgbpost" style="height:340px"></canvas>
</div>
<div class="panel">{img("vgb_snr")}</div>
<div class="panel">{img("vgb_traces")}
<div class="caption">Distance traces for the three loudest VGBs, all 24 walkers.</div></div>
<div class="panel">{img("vgb_hists")}
<div class="caption">Pooled posteriors of the remaining sampled parameters, all 55 leaves
together &mdash; so the truth overlay is a DISTRIBUTION, not a line: the dotted red step
(plus the rug under the axis) is the histogram of the 55 catalogue truths on the same bins,
rescaled to the posterior peak. <code>fdot_astro_ratio</code> is the exception: every
catalogue VGB is a pure GR chirper, so its truth is the single dotted line at exactly 0.
Conventions come from the run's own code &mdash;
<code>recipe.gb_catalogue_to_sampling_basis</code> (sampling
<code>phi0 = -TrueAnomaly mod 2&pi;</code>, <code>psi mod &pi;</code>) plus the distance-basis
split in <code>stock/erebor/vgb.py</code>; the (f0, Mc, dist) &rarr; A round trip reproduces
the catalogue Amplitude to {("%.1e" % VGB_TRUTH_REL) if VGB_TRUTH_REL is not None else "n/a"}
relative.</div></div>
<div class="panel">
<div class="btnrow viewctl">
  <label>vgb <select id="vgb1_sel"></select></label>
  <span class="caption" style="align-self:center">single-VGB corner posterior
  (ChainConsumer), SNR-ordered</span>
</div>
<img id="vgb1_img" alt="single-VGB corner posterior">
<div class="caption" id="vgb1_cap"></div>
<div class="caption">Full 5-parameter corner per leaf, pre-rendered with
ChainConsumer over the last {CORNER_ITS} stored iterations &times; {nwalk} cold walkers
({vgb_corner.shape[0]} samples); violet = posterior (1/2&sigma; contours + shaded
marginals), dotted red = the catalogue truth in the same sampled basis as every other
truth overlay on this page. Axis extents are widened to CONTAIN the truth, so a truth
line sitting outside the posterior is visible rather than clipped away.</div>
</div>
</section>

<section id="explorer"><h2>Amplitude vs Frequency Explorer (log10 A from sampled dist, f0, Mc)</h2>
<div class="panel">
<div class="btnrow">
  <button id="btn_all">full band</button>
  <button id="btn_top3">3 highest-frequency sources</button>
  <button id="btn_truth">show catalogue truths</button>
  <button id="btn_reset">reset zoom</button>
  <span class="caption" style="align-self:center">drag = pan &middot; wheel/pinch = zoom</span>
</div>
<div class="btnrow viewctl">
  <button id="expl_pick" title="arm, then click the plot to set the view center">set center by click</button>
  <label>cx <input id="expl_cx" type="text"></label>
  <label>cy <input id="expl_cy" type="text"></label>
  <label>width <input id="expl_wsl" type="range" min="0" max="1000" step="1"><input id="expl_w" type="text"></label>
  <label>height <input id="expl_hsl" type="range" min="0" max="1000" step="1"><input id="expl_h" type="text"></label>
</div>
<canvas id="expl"></canvas>
<div class="caption" id="expl_cap"></div>
</div>
</section>

<section id="timing"><h2>Timing + Memory</h2>
{rj_kpis}
<div class="caption">Direct-batch RJ&rarr;in-model throughput, job 197 (the 07:14 restart on
the full perf mega-batch; its records start after line 12405 of globalfit_run.log, job 196
is everything before). The batch collapsed every remaining dispatch/fill tax: removal
propose 658&ndash;843&rarr;75 s, search propose excluding centers 537&rarr;88 s, buffer fills
430&rarr;11.6 s (buffill_resid_psd 419&rarr;1.2 s), temper_buffer 388&rarr;8.6 s (45&times;),
proposal tables/infomat_kernel 42&ndash;78&rarr;9.3 s. What is left is
<strong>rj_fstat_centers</strong>, now instrumented by the [FSTAT_CTR] census: 55% of the
636,697 unit rows were at-cap reserve and are no longer computed, but the per-row cost
oscillates between two states (1.64 ms/row for 285k rows vs 0.31 ms/row for 346k rows) &mdash;
the last big lever.</div>
<div class="panel">{img("rj_breakdown", "rj propose breakdown")}
<div class="caption">Where each rj move spends its wall time &mdash; <strong>one panel per
move, never lumped</strong> (SYNC-ATTRIBUTED records: every mark carries exactly its own
kernel time). Bars are LEAF spans, largest first; the enclosing phase marks
<code>run_proposal</code> / <code>run_tempering</code> wrap them and are quoted in each
panel title instead of competing for a bar. These are the FIRST post-mega-batch
records{rj_break_stamps}.
{rj_break_txt}
<strong>rj_fstat_search</strong> is the F-stat birth proposal and it is now the whole story:
<code>rj_fstat_centers</code> alone is 92% of it, while every other span &mdash; rj_step,
route_dispatch, gll_engine, rj_getll, the buffer fills, the proposal tables &mdash; sits in
the tens of seconds. <strong>rj_prior_removal</strong> collapsed from 658&ndash;843 s to
75 s and its own profile is now tempering-dominated (run_tempering / temper_swap_score /
sorter_build), i.e. no kernel hog left. Centers is the last lever on the page.</div></div>
<div class="panel">{img("mem_telemetry", "device memory telemetry")}
<div class="caption">Per-device used memory from the in-run memGetInfo telemetry.
Flat sawtooth = the bounded-buffer behavior; a monotonic ramp here is the leak alarm.</div></div>
<div class="panel">{img("timing_moves", "per-move timing")}</div>
<div class="panel">{img("gpu_util", "gpu telemetry")}
<div class="caption">nvidia-smi utilization + memory, latest three jobs &mdash; the solid
traces are now job 197, the post-mega-batch run. The single-device signature is GONE:
job-197 means are <strong>dev0 26% / dev1 62%</strong> (job 196 read 3% / 79%), with peaks
of 69 and 43 GB out of 96. The residual imbalance is the F-stat center precompute, the one
phase the batch did not de-serialize.</div></div>
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
const VGBC = {VGB_CORNER_JSON};
const GB1 = {GB1_JSON};
// Single-VGB corner panel: the <select> swaps the src of ONE <img> between
// 55 pre-rendered ChainConsumer PNGs (base64 data URIs held in JS, so only
// the selected corner is ever in the document's visible flow).
function cornerPanel(px, blob) {{
  const im = document.getElementById(px + "_img"),
        sel = document.getElementById(px + "_sel"),
        cap = document.getElementById(px + "_cap");
  if (!im || !sel) return;
  if (!blob.src || !blob.src.length) {{
    im.remove();
    if (cap) cap.textContent = "no corner plots available in this snapshot";
    return;
  }}
  blob.src.forEach((s, i) => {{
    const o = document.createElement("option");
    o.value = i; o.textContent = s.label; sel.appendChild(o);
  }});
  function show() {{
    const S = blob.src[+sel.value || 0];
    im.src = "data:image/png;base64," + S.png;
    im.alt = S.label;
    cap.textContent = S.sub;
  }}
  sel.onchange = show;
  show();
}}
cornerPanel("vgb1", VGBC);
// Shared single-source posterior panel: a <select> of sources driving a
// canvas of one histogram per sampled parameter, each with the catalogue
// truth as a dotted RED vertical line (the file-wide truth convention).
// Used by the GB panel (9 params; the VGB panel moved to corner plots
// 2026-08-15); every color is read from the CSS custom properties AT DRAW
// TIME, so the panel follows a light/dark theme switch without
// regenerating anything.
function srcPanel(px, blob) {{
  const cv = document.getElementById(px), sel = document.getElementById(px + "_sel"),
        cap = document.getElementById(px + "_cap");
  if (!cv || !sel) return;
  if (!blob.src.length) {{
    cap.textContent = "no sources available in this snapshot"; return;
  }}
  blob.src.forEach((s, i) => {{
    const o = document.createElement("option");
    o.value = i; o.textContent = s.label; sel.appendChild(o);
  }});
  const NB = 24, dpr = window.devicePixelRatio || 1;
  const fmt = v => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3))
    ? v.toExponential(1) : (+v).toPrecision(4);
  function draw() {{
    const S = blob.src[+sel.value || 0];
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const C = n => css.getPropertyValue(n).trim();
    g.fillStyle = C("--panel"); g.fillRect(0, 0, w, h);
    g.font = "10px ui-monospace,monospace";
    const n = blob.params.length;
    const cols = Math.min(5, n), rows = Math.ceil(n / cols);
    const cw = w / cols, chh = h / rows;
    for (let k = 0; k < n; k++) {{
      const gx = (k % cols) * cw, gy = Math.floor(k / cols) * chh;
      const x0 = gx + 8, x1 = gx + cw - 8, y0 = gy + 17, y1 = gy + chh - 24;
      const v = S.samples[k].filter(a => a !== null);
      if (!v.length) continue;
      let lo = Math.min(...v), hi = Math.max(...v);
      const t = S.truth[k];
      const hasT = (t !== null && isFinite(t));
      if (hasT) {{ lo = Math.min(lo, t); hi = Math.max(hi, t); }}
      if (!(hi > lo)) hi = lo + Math.max(Math.abs(lo) * 1e-6, 1e-12);
      const pd = (hi - lo) * 0.06; lo -= pd; hi += pd;
      const cnt = new Array(NB).fill(0);
      for (const a of v)
        cnt[Math.min(NB - 1, Math.floor((a - lo) / (hi - lo) * NB))]++;
      const mx = Math.max(...cnt, 1);
      g.fillStyle = C("--violet");
      for (let b = 0; b < NB; b++) {{
        if (!cnt[b]) continue;
        const bw = (x1 - x0) / NB, bh = (y1 - y0) * cnt[b] / mx;
        g.fillRect(x0 + bw * b, y1 - bh, Math.max(bw - 0.6, 0.6), bh);
      }}
      g.strokeStyle = C("--line"); g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0, y1); g.lineTo(x1, y1); g.stroke();
      if (hasT) {{
        const tx = x0 + (x1 - x0) * (t - lo) / (hi - lo);
        g.strokeStyle = C("--red"); g.lineWidth = 1.5; g.setLineDash([3, 2]);
        g.beginPath(); g.moveTo(tx, y0 - 3); g.lineTo(tx, y1); g.stroke();
        g.setLineDash([]); g.lineWidth = 1;
      }}
      g.fillStyle = C("--fg"); g.fillText(blob.params[k], gx + 8, gy + 11);
      g.fillStyle = C("--dim");
      g.fillText(fmt(lo), x0, y1 + 11);
      const rt = fmt(hi);
      g.fillText(rt, Math.max(x0, x1 - g.measureText(rt).width), y1 + 11);
      if (hasT) {{
        g.fillStyle = C("--red");
        const tt = "truth " + fmt(t);
        g.fillText(tt, x0 + Math.max(0, (x1 - x0 - g.measureText(tt).width) / 2),
                   y1 + 21);
      }}
    }}
    const note = S.note
      ? ` &middot; <span style="color:var(${{S.bad ? "--red" : "--fg"}})">${{S.note}}</span>`
      : "";
    cap.innerHTML = S.sub + note;
  }}
  sel.onchange = draw;
  new ResizeObserver(draw).observe(cv);
  draw();
}}
srcPanel("gb1", GB1);
// Shared view controls: numeric center (cx, cy) + log-scale width/height
// sliders, all around a FIXED center -- plus a click-to-set-center mode.
// api: get() -> [X0,X1,Y0,Y1]; set(x0,x1,y0,y1) (must redraw); fullW/fullH
// = the reset-view spans (slider range = [full/2000, full*1.2], log-mapped).
function viewCtl(px, cv, api) {{
  const el = id => document.getElementById(px + "_" + id);
  const cxI = el("cx"), cyI = el("cy"), wI = el("w"), hI = el("h"),
        wS = el("wsl"), hS = el("hsl"), pk = el("pick");
  if (!cxI) return {{ sync: () => {{}} }};
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const s2span = (s, full) => {{
    const mn = Math.log(full / 2000), mx = Math.log(full * 1.2);
    return Math.exp(mn + (mx - mn) * s / 1000);
  }};
  const span2s = (W, full) => {{
    const mn = Math.log(full / 2000), mx = Math.log(full * 1.2);
    return clamp(Math.round(1000 * (Math.log(W) - mn) / (mx - mn)), 0, 1000);
  }};
  let busy = false;
  function sync() {{
    if (busy) return;
    const [X0, X1, Y0, Y1] = api.get();
    cxI.value = ((X0 + X1) / 2).toPrecision(7);
    cyI.value = ((Y0 + Y1) / 2).toPrecision(7);
    wI.value = (X1 - X0).toPrecision(5);
    hI.value = (Y1 - Y0).toPrecision(5);
    wS.value = span2s(X1 - X0, api.fullW);
    hS.value = span2s(Y1 - Y0, api.fullH);
  }}
  function applyTyped() {{
    const [X0, X1, Y0, Y1] = api.get();
    let cx = parseFloat(cxI.value), cy = parseFloat(cyI.value);
    let W = parseFloat(wI.value), H = parseFloat(hI.value);
    if (!isFinite(cx)) cx = (X0 + X1) / 2;
    if (!isFinite(cy)) cy = (Y0 + Y1) / 2;
    if (!isFinite(W) || W <= 0) W = X1 - X0;
    if (!isFinite(H) || H <= 0) H = Y1 - Y0;
    busy = true;
    api.set(cx - W / 2, cx + W / 2, cy - H / 2, cy + H / 2);
    busy = false; sync();
  }}
  cxI.onchange = cyI.onchange = wI.onchange = hI.onchange = applyTyped;
  wS.oninput = () => {{
    const [X0, X1, Y0, Y1] = api.get();
    const cx = (X0 + X1) / 2, W = s2span(+wS.value, api.fullW);
    busy = true; api.set(cx - W / 2, cx + W / 2, Y0, Y1); busy = false;
    wI.value = W.toPrecision(5);
  }};
  hS.oninput = () => {{
    const [X0, X1, Y0, Y1] = api.get();
    const cy = (Y0 + Y1) / 2, H = s2span(+hS.value, api.fullH);
    busy = true; api.set(X0, X1, cy - H / 2, cy + H / 2); busy = false;
    hI.value = H.toPrecision(5);
  }};
  let picking = false;
  pk.onclick = () => {{ picking = !picking; pk.classList.toggle("armed", picking);
                        cv.style.cursor = picking ? "crosshair" : "grab"; }};
  // Capture-phase so an armed pick swallows the click before the pan handler.
  cv.addEventListener("pointerdown", e => {{
    if (!picking) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const [X0, X1, Y0, Y1] = api.get();
    const cx = X0 + ((e.clientX - r.left) - ml) / (w - ml - mr) * (X1 - X0);
    const cy = Y0 + ((h - mb) - (e.clientY - r.top)) / (h - mb - mt) * (Y1 - Y0);
    const W = X1 - X0, H = Y1 - Y0;
    api.set(cx - W / 2, cx + W / 2, cy - H / 2, cy + H / 2);
    picking = false; pk.classList.remove("armed"); cv.style.cursor = "grab";
    sync();
  }}, {{ capture: true }});
  sync();
  return {{ sync }};
}}
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
  const FW = X1 - X0, FH = Y1 - Y0;
  let syncCtl = () => {{}};
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
    // catalogue truth distance per leaf (red, the file-wide truth color)
    g.globalAlpha = 1; g.fillStyle = C("--red");
    for (const p of (VPOST.truth || [])) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.beginPath(); g.arc(x, y, 2.6, 0, 6.29); g.fill();
    }}
    g.globalAlpha = 1;
    syncCtl();
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
  syncCtl = viewCtl("vgbpost", cv, {{
    get: () => [X0, X1, Y0, Y1],
    set: (a, b, c, d) => {{ X0 = a; X1 = b; Y0 = c; Y1 = d; draw(); }},
    fullW: FW, fullH: FH,
  }}).sync;
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
  const TRUTH = DATA.truth || [];
  let showT = false;
  const baseCap = (hasGB
    ? `GB samples: ${{DATA.gb.length}} alive-source rows (last iteration, all cold walkers; y = log10 amplitude from (dist, f0, Mc)).`
    : `No GB sources alive yet - showing the 55 VGBs (24 walker samples each) as 1/dist vs leaf index. GB samples take over automatically once births land.`);
  const setCap = () => {{
    cap.textContent = baseCap + (TRUTH.length
      ? (showT ? " " + (DATA.truth_cap || "") : ` ${{TRUTH.length}} catalogue truth points available - press "show catalogue truths".`)
      : " No catalogue truth overlay in this snapshot.");
  }};
  let X0, X1, Y0, Y1;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = (a, b) => [(a - (b - a) * 0.05) , (b + (b - a) * 0.05)];
  const full = () => {{
    [X0, X1] = pad(Math.min(...xs), Math.max(...xs));
    [Y0, Y1] = hasGB ? pad(Math.min(...ys), Math.max(...ys)) : pad(0, Math.max(...ys));
  }};
  full();
  const FW = X1 - X0, FH = Y1 - Y0;
  let syncCtl = () => {{}};
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
    g.fillText(hasGB ? "log10 A" : "1 / dist [1/kpc]", -30, 0); g.restore();
    // truths UNDER the recovered cloud so the recoveries stay readable
    if (showT) {{
      g.fillStyle = C("--red"); g.globalAlpha = 0.5;
      for (const p of TRUTH) {{
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.fillRect(x - 0.7, y - 0.7, 1.4, 1.4);
      }}
    }}
    const col = hasGB ? C("--green") : C("--violet");
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.globalAlpha = 0.55; g.fillStyle = col;
      g.beginPath(); g.arc(x, y, 2.2, 0, 6.29); g.fill();
    }}
    g.globalAlpha = 1;
    syncCtl();
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
  const bt = document.getElementById("btn_truth");
  if (!TRUTH.length) bt.disabled = true;
  bt.onclick = () => {{
    showT = !showT;
    bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
    bt.textContent = showT ? "hide catalogue truths" : "show catalogue truths";
    setCap(); draw();
  }};
  setCap();
  document.getElementById("btn_top3").onclick = () => {{
    const srt = [...pts].sort((a, b) => b[0] - a[0]);
    const top = srt.slice(0, Math.min(3 * 24, srt.length));
    const tx = top.map(p => p[0]), ty = top.map(p => p[1]);
    [X0, X1] = pad(Math.min(...tx), Math.max(...tx) || 1);
    [Y0, Y1] = hasGB ? pad(Math.min(...ty), Math.max(...ty)) : pad(0, Math.max(...ty) || 1);
    draw();
  }};
  syncCtl = viewCtl("expl", cv, {{
    get: () => [X0, X1, Y0, Y1],
    set: (a, b, c, d) => {{ X0 = a; X1 = b; Y0 = c; Y1 = d; draw(); }},
    fullW: FW, fullH: FH,
  }}).sync;
  new ResizeObserver(draw).observe(cv);
  draw();
}})();
</script>
"""
open(OUT, "w").write(html)
print(f"wrote {OUT}: {len(html)//1024} KB ({len(html)/1024**2:.2f} MB), "
      f"{len(IMGS)} plots + {len(VGB_CORNER['src'])} VGB corners, "
      f"missing={len(MISSING)}")
