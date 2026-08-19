#!/usr/bin/env python
"""GF run status page: <run_dir> -> one self-contained HTML file.

Reusable: point RUN_DIR at any unzipped gf_prod_* snapshot and rerun; the
artifact redeploys to the same URL. Sections degrade to labeled
placeholders when a snapshot lacks their inputs.

The page is a STATUS OBJECT for collaborators, not an engineering worklog.
Its spine is one frozen denominator -- the 812 catalogue galactic binaries
detectable (optimal SNR > 7) over 3-21.94 mHz under this run's own fitted
noise -- against which completeness, purity and per-source recovery are all
quoted. Run-mechanics forensics live in the collapsed appendix, never in the
body.

Optional inputs, read from the run directory or the working directory:
  gb_truth_3to21.npz  the frozen detectability set + full catalogue
                      parameters (built once by build_truth.py)
  kappa_grid.npz      SNR/amplitude ceiling per frequency, for the
                      sensitivity curve on the population panels
  gf_arm_<tag>.npz    written by THIS script on every run; holds one arm's
                      per-iteration recovery series so the v2/v3 comparison
                      panels can draw both arms.
Without them the recovery section degrades to a note; every other section
still builds.

COLOUR CONVENTION, one meaning per hue (the previous page used red for five
different things):
  cyan   injected data / the injected catalogue / arm v2
  amber  noise + foreground / arm v3
  green  recovered AND matched to an injection
  violet recovered with NO matching injection
  red    detectable but NOT recovered  (and nothing else)
  white  reference lines: noise model, y = x, sensitivity
"""
import base64, glob, io, json, os, re, sys
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

# MATCH-CRITERION CONTENT GATE (user ruling 2026-08-19). The page's
# completeness / purity / matched-pair numbers all come from the 2-bin f0
# PROXY match, not the real phase-maximised overlap statistic (too heavy to
# compute at page-build time). Ruling: catalogue TRUTHS stay on every visual
# overlay, but nothing derived from the page's own match criterion is shown
# -- no completeness/purity, no matched counts, no matched-pair deltas, no
# recovery split/census. GF_MONITOR_MATCH_STATS=1 restores those panels.
SHOW_MATCH_STATS = os.environ.get("GF_MONITOR_MATCH_STATS", "0") == "1"

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
# PICK THE LIVE STORE DETERMINISTICALLY (2026-08-16). A recovered run dir
# holds THREE .h5 files -- the live store, ``*_running_backup_copy.h5`` and
# one or more ``*_CORRUPT*.h5`` kept for forensics -- and the old
# "last name os.listdir happens to yield" pick could land on any of them.
# os.listdir order is filesystem-dependent, so the page could silently
# render the damaged store (missing iterations) or the lagging backup on one
# machine and the right file on another, with nothing in the output saying
# which. Name the live store explicitly and keep the others as fallbacks.
_h5s = sorted(fn for fn in os.listdir(RUN_DIR) if fn.endswith(".h5"))
_live = [fn for fn in _h5s
         if not fn.endswith("_running_backup_copy.h5") and "_CORRUPT" not in fn]
if not _live:                                   # only a backup survived
    _live = [fn for fn in _h5s if fn.endswith("_running_backup_copy.h5")]
    if _live:
        MISSING.append(
            "no live store in this snapshot -- rendering the running backup "
            "copy, which lags the run by up to one save step.")
h5path = os.path.join(RUN_DIR, (_live or _h5s)[0])
f = h5py.File(h5path, "r")
g = f["global_fit"]
ll_all = g["log_like"][:, 0, 0, :]
filled = np.where(np.any(ll_all != 0.0, axis=1))[0]
NIT = int(filled.max()) + 1 if filled.size else 0
# REWIND-AWARE (2026-08-19): the live row count is the store's ``iteration``
# attr, NOT the filled-row extent. reset_recipe_stage rewinds by moving that
# one attr back; the rows beyond it keep their old (dead) contents until the
# run's next grow() truncates them. Rendering by filled rows alone mixed the
# discarded pre-rewind trajectory into every panel of a freshly rewound
# store (leaves climbing to the old values, x-axes past the live range).
_it_attr = g.attrs.get("iteration")
REWOUND = _it_attr is not None and 0 < int(_it_attr) < NIT
if REWOUND:
    MISSING.append(
        f"store rewound: iteration attr {int(_it_attr)} < filled rows {NIT}; "
        f"rendering the live {int(_it_attr)} rows only (the rest is the "
        "discarded pre-rewind trajectory awaiting truncation).")
    NIT = int(_it_attr)
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
elif _base.endswith("_v4") or "3mo_v4" in _base:
    RUN_LABEL, RUN_KIND = "3-Month v4", "3mo_v4"
elif _base.endswith("_v3") or "3mo_v3" in _base:
    # The v3 A/B carries the same Tobs as v2, so the label has to come from
    # the VARIANT or the two pages are indistinguishable in a browser tab --
    # which is the whole point of running them side by side.
    RUN_LABEL, RUN_KIND = "3-Month v3", "3mo_v3"
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

# THE CAP PANEL MUST PLOT THE ENFORCED ARRAY (2026-08-16). This used to
# read ``gb/band_leaf_cap`` -- the LEGACY MIRROR that
# ``_mirror_band_leaf_cap`` keeps equal to the MAX over each band's cap
# cells -- while the births are actually gated per CAP CELL
# (``gbspecialstretch._run_rj_step``: "THE EXACT PER-CELL ENFORCEMENT
# POINT ... it is per cell"). On this run the mirror overstates the real
# cap for 515 of 1,232 cells (42%), by up to 12, because 133 of 154 bands
# carry a non-zero spread across their own cells. The old label
# "leaf cap per band" was therefore TRUE of the array being drawn and
# false about the run -- fixing only the string would have made the plot
# lie. Prefer the cell array; fall back to the band array only where the
# cell array does not exist (``cap_divisor == 1`` stores never allocate
# it), and label from WHICH array was used rather than guessing from the
# column count.
band_edges = sub["gb/band_edges"][:]
try:
    cap_edges_static = sub["gb/cap_edges"][:]
except Exception:
    cap_edges_static = band_edges
CAP_K = max(int(round((cap_edges_static.size - 1) / max(band_edges.size - 1, 1))), 1)
caps = _safe(sub, "gb/cap_cell_leaf_cap", None, "per-cell leaf caps")
CAP_UNIT = "cap cell"
if caps is None or not getattr(caps, "size", 0):
    caps = _safe(sub, "gb/band_leaf_cap", None, "per-band leaf caps")
    CAP_UNIT, CAP_K = "band", 1


logpath = None
for root, _, fns in os.walk(RUN_DIR):
    for fn in fns:
        if fn == "globalfit_run.log":
            logpath = os.path.join(root, fn)
log_text = open(logpath, errors="replace").read() if logpath else ""
# REWIND-AWARE (2026-08-19): the run log is CUMULATIVE across launches. On a
# rewound store the segments before the final relaunch describe the DISCARDED
# trajectory, and every log-parsed panel (band shutoffs, RJ split, acceptance,
# timing) would mix it into the live run's series. Cut at the last resume
# marker; non-rewound stores keep the full log (normal resumes continue the
# same trajectory, so their history is valid).
if REWOUND and log_text:
    # Per-launch marker actually present in THIS file: the results rank
    # prints it once at every process start ("RESUMING from existing
    # backend" goes to the slurm stdout, not here).
    _pos = log_text.rfind("starting async save/plot loop")
    if _pos > 0:
        _pos = log_text.rfind("\n", 0, _pos) + 1
        MISSING.append(
            "log-parsed panels cut to the final launch segment (pre-rewind "
            f"log history discarded, {_pos/1e6:.1f} MB skipped).")
        log_text = log_text[_pos:]

# ---- the artifacts directory, found rather than guessed --------------------
# This used to be built as ``basename(RUN_DIR) + "_artifacts"``, which is only
# right when the run directory happens to be named after the store. It is not:
# both production arms live in ``gf_prod_3mo_v2`` / ``gf_prod_3mo_v3`` while
# the artifacts directory inside each is ``gf_prod_3mo_artifacts`` -- named for
# the STORE. The guess therefore missed on every run, run_settings.log was
# never read, and the whole data/template/residual section rendered as "plot
# unavailable" with no indication that a path was at fault. Glob for it.
ART_DIR = None
_cands = sorted(glob.glob(os.path.join(RUN_DIR, "*_artifacts")))
_cands += [os.path.join(RUN_DIR, os.path.basename(os.path.normpath(RUN_DIR))
                        + "_artifacts")]
for _c in _cands:
    if os.path.exists(os.path.join(_c, "run_settings.log")):
        ART_DIR = _c
        break
SETTINGS_TXT = ""
if ART_DIR:
    SETTINGS_TXT = open(os.path.join(ART_DIR, "run_settings.log"),
                        errors="replace").read()
else:
    MISSING.append("no *_artifacts/run_settings.log under the run directory; "
                   "the residual spectrum cannot be rebuilt.")

# The GB branch anchors source phase at this epoch; the data grid starts a
# little later, and the offset is a real phase factor, so it is read from the
# run rather than defaulted.
_m0 = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", SETTINGS_TXT, re.S)
T_REF_SCI = float(_m0.group(1)) if _m0 else 97729089.327664

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

# ---- F11: the noise model, in two panels ----------------------------------
# The page used to carry six: two instrument traces, two instrument
# histograms, five foreground traces and five foreground histograms, none of
# which said whether the noise model is RIGHT. What matters is (a) do the two
# instrument parameters recover their injected values, and (b) is the
# foreground coming down as sources leave the residual. Two panels, both
# answerable at a glance.
psd_cold = psd_c[:, 0, :, 0, :]                     # (it, 24, 2)
gal_cold = gal_c[:, 0, :, 0, :]                     # (it, 24, 5)
SOMS_INJ, SA_INJ = 1.496182e-11, 2.982412e-15
GAL_NAMES = ["log10 amp", "p1", "log10 fknee", "p2", "slope"]

_nsh = min(3, SUB_NIT)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.0))
for j, (name, inj, unit) in enumerate(
        [("Soms_d", SOMS_INJ, "m"), ("Sa_a", SA_INJ, "m/s$^2$")]):
    v = psd_cold[-_nsh:, :, j].ravel()
    ax[j].hist(v, bins=26, color=VIOLET, alpha=0.85)
    ax[j].axvline(inj, color=CYAN, lw=1.6, ls="--", label="injected")
    ax[j].set_title(f"{name}  [{unit}]", fontsize=10)
    ax[j].legend(fontsize=8)
    ax[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
NOISE_BIAS = [float(np.median(psd_cold[-_nsh:, :, j]) / inj - 1.0)
              for j, inj in enumerate((SOMS_INJ, SA_INJ))]
fig.suptitle(f"instrument-noise posteriors, last {_nsh} stored iterations "
             f"x {nwalk} cold walkers", fontsize=10, color=FG)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_b64(fig, "f11_psd")

try:
    from lisatools.sensitivity import get_sensitivity, A2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    import matplotlib.colors as mcolors

    fr = np.logspace(np.log10(3e-4), np.log10(2.5e-2), 400)

    def sens_curves(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms ** 2, sa ** 2,
                                      lisa_models.DefaultOrbits(), "sampled")
        if galp is None:
            return get_sensitivity(fr, sens_fn=A2TDISens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=A2TDISens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    ramp = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    pm = np.median(psd_cold[-1], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for k in range(SUB_NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold[k], axis=0)
        ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                color=ramp(k / max(SUB_NIT - 1, 1)), lw=1.1,
                label=(f"iteration {k}" if k in (0, SUB_NIT - 1) else None))
    ax.plot(fr, sens_curves(*pm), color=FG, lw=1.4, ls=":",
            label="instrument only (latest)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD, TDI A channel  [1/Hz]")
    ax.legend(fontsize=8, loc="upper left")
    fig_b64(fig, "f11_fg")
except Exception as e:
    MISSING.append(f"foreground curve render failed: {e!r}")


# ---- RESTORED: per-parameter noise traces and histograms -------------
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

# ---- RESTORED: LISASens curve pair (instrument vs instrument+foreground) ---
# Kept alongside F11 rather than folded into it: F11 plots the A-channel PSD
# the likelihood is weighted by, this pair plots the LISASens sky-averaged
# sensitivity the mission documents quote, and the injected instrument curve
# only exists on this one.
try:
    from lisatools.sensitivity import get_sensitivity, LISASens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    import matplotlib.colors as mcolors

    fr = np.logspace(np.log10(2e-4), np.log10(2.6e-2), 500)

    def sens_lisasens(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms**2, sa**2,
                                      lisa_models.DefaultOrbits(), "mon")
        if galp is None:
            return get_sensitivity(fr, sens_fn=LISASens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=LISASens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    pm = np.median(psd_cold[-1], axis=0)
    gm = np.median(gal_cold[-1], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.6, label="instrument PSD")
    ax.plot(fr, sens_lisasens(pm[0], pm[1], gm), color=AMBER, lw=1.6,
            label="PSD + galactic foreground")
    ax.plot(fr, sens_lisasens(SOMS_INJ, SA_INJ), color=RED, ls=":", lw=1.3,
            label="injected instrument")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        "sensitivity, cold-chain walker-median, latest stored iteration")
    fig_b64(fig, "psd_curves")

    ramp2 = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.4,
            label="instrument PSD (latest)")
    for k in range(SUB_NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold[k], axis=0)
        ax.plot(fr, sens_lisasens(pk_[0], pk_[1], gk),
                color=ramp2(k / max(SUB_NIT - 1, 1)), lw=1.1,
                label=f"iter {k}" if k in (0, SUB_NIT - 1) else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        "PSD + foreground per stored iteration (light -> dark = later)")
    fig_b64(fig, "psd_evolution")
except Exception as e:
    MISSING.append(f"LISASens curve render failed: {e!r}")

# ---- GB leaf count (the raw model size; the recovery section below is
# where it is given a denominator) ----------------------------------------
gb_counts = gb_inds.sum(axis=-1)                    # (it, 24)

# ---- detectable-truth set for the overlays (from the census npz) ----------
# `census.py` computes optimal SNR for every catalogue GB over 3-21 mHz
# against the run's OWN sampled sensitivity; the detectable (SNR>7) subset
# is the natural TARGET line for the leaf-count and occupancy panels. It is
# optional -- without the npz these overlays simply do not draw.
DET_F0 = None
for _cp in (os.path.join(RUN_DIR, "gb_hi_f_census.npz"), "gb_hi_f_census.npz"):
    if os.path.exists(_cp):
        try:
            _c = np.load(_cp)
            if "det_f0" in _c:
                DET_F0 = np.asarray(_c["det_f0"], dtype=float)
                DET_LO = float(_c["det_lo"]); DET_HI = float(_c["det_hi"])
        except Exception:
            DET_F0 = None
        break

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for w in range(nwalk):
    ax[0].plot(it, gb_counts[:, w], color=GREEN, alpha=0.35, lw=0.9)
ax[0].plot(it, gb_counts.max(axis=1), color=GREEN, lw=1.8, label="all f")
# TRUTH TARGET. The raw leaf count has no scale -- 577 leaves is only
# meaningful against how many sources are actually THERE. Overlay the
# detectable (SNR>7) catalogue count over the range where SNRs exist, and
# the model's leaf count restricted to that SAME range, so the two lines
# are comparable rather than merely adjacent.
if DET_F0 is not None:
    _f0_it = g["chain/gb"][:NIT, 0, 0][..., 1] * 1e-3     # (it, walker, leaf)
    _in = (_f0_it >= DET_LO) & (_f0_it <= DET_HI) & gb_inds
    _cnt = _in.sum(axis=2)                                # (it, walker)
    ax[0].plot(it, _cnt.max(axis=1), color=CYAN, lw=1.6,
               label=f"{DET_LO*1e3:.0f}-{DET_HI*1e3:.0f} mHz")
    ax[0].axhline(DET_F0.size, color=RED, ls=":", lw=1.5)
    ax[0].text(NIT * 0.98, DET_F0.size,
               f"{DET_F0.size} detectable (SNR>7), "
               f"{DET_LO*1e3:.0f}-{DET_HI*1e3:.0f} mHz ", color=RED,
               fontsize=8, va="bottom", ha="right")
    ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("GB leaf count (cold walkers)"); ax[0].set_xlabel("iteration")
ax[0].set_ylim(bottom=-0.5)
if caps is not None and caps.size:
    _cn = caps.shape[0]                      # may lag NIT (backup fallback)
    im = ax[1].imshow(caps.T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, _cn, 0, caps.shape[1]])
    _u = np.unique(caps[-1])
    _lab = CAP_UNIT
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
# The SHUTOFF is per BAND by design, but the image is now per CAP CELL --
# band b spans rows [b*K, (b+1)*K), so an unscaled axhline would land at
# 1/K of its true height. Also anchor the marker to the image's OWN x
# extent: caps can come from the backup copy and be LONGER than NIT.
_cx = caps.shape[0]
for b in shutoff_bands:
    _y = (b + 0.5) * CAP_K
    ax[1].axhline(_y, color=RED, lw=1.0, alpha=0.55)
    ax[1].plot([_cx * 0.99], [_y], marker="<", color=RED, ms=6, clip_on=False)
if shutoff_bands:
    ax[1].set_title(
        f"leaf cap per {_lab} ({len(shutoff_bands)} bands birth-OFF, red)")
fig_b64(fig, "gb_leaves")

# ---- 5a. CAP-CELL OCCUPANCY ----------------------------------------------
# THE QUESTION THIS PANEL EXISTS TO ANSWER (user, 2026-08-16): "how many of
# the 1232 cap cells have 1 source in them? It should be many more if there
# are really 1232 sub-bands." The band panel above cannot answer it -- it
# shows the cap, not the OCCUPANCY, and it shows bands, not cells. Without
# this the only way to tell whether the cap is binding, and whether the cap
# grid is being followed at all rather than quietly collapsing back onto the
# 154-band grid, is to open the store by hand.
#
# The three questions, one axes each: what does the occupancy distribution
# look like against the cap (is the cap binding?); how do occupancy and the
# at-cap count evolve (is the ramp keeping ahead of the model?); and WHERE in
# frequency the occupied cells sit (which is why the occupied FRACTION is
# small even when the packing is working -- the sources are all in the
# galaxy, not spread over 0.56-21.9 mHz).
cap_cells = _safe(sub, "gb/cap_cell_leaf_cap", None, "per-cell leaf caps")
cap_edges_arr = None
try:
    cap_edges_arr = sub["gb/cap_edges"][:]
except Exception:
    pass
CAP_TXT = ""
if cap_cells is not None and cap_cells.size and cap_edges_arr is not None:
    ncell = cap_edges_arr.size - 1
    # The cap arrays can lag the main backend by a row (separate flush), and
    # an unwritten row reads as all-zero -- which would render as "every cell
    # capped at 0". Use the last row that carries a real cap.
    _crow = cap_cells.shape[0] - 1
    while _crow > 0 and not np.any(cap_cells[_crow] > 0):
        _crow -= 1

    def _cell_counts(iteration):
        """Sources per cap cell, per cold walker, at one stored iteration."""
        alive = g["inds/gb"][iteration, 0, 0]                    # (nw, nleaf)
        f0 = g["chain/gb"][iteration, 0, 0][..., 1] * 1e-3       # mHz -> Hz
        out = np.zeros((alive.shape[0], ncell), dtype=np.int32)
        for w in range(alive.shape[0]):
            fv = f0[w][alive[w]]
            if not fv.size:
                continue
            ci = np.searchsorted(cap_edges_arr, fv, side="right") - 1
            ci = ci[(ci >= 0) & (ci < ncell)]
            out[w] = np.bincount(ci, minlength=ncell)
        return out

    cc_last = _cell_counts(NIT - 1)
    nw_ = cc_last.shape[0]
    cap_row = cap_cells[_crow]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.6),
                           gridspec_kw=dict(wspace=0.42))

    # (a) occupancy distribution vs the cap. Log-scaled counts, because the
    # empty bar is ~5000x the tallest occupied one and a linear axis would
    # render every bar this panel exists to compare as a flat line.
    # Split each bar by whether those cells are AT their own cap, rather
    # than colouring a whole occupancy level: cells holding one source are a
    # mix of cap-1 cells (full) and cap-2 cells (room for one more), and
    # painting the level uniformly would claim ~190 saturated cells where
    # there are ~40.
    _atc = cc_last >= cap_row[None, :]
    kmax = int(max(cc_last.max(), cap_row.max())) + 1
    hist = np.array([(cc_last == k).sum() / nw_ for k in range(kmax + 1)])
    h_at = np.array([((cc_last == k) & _atc).sum() / nw_
                     for k in range(kmax + 1)])
    h_ov = np.array([((cc_last == k) & (cc_last > cap_row[None, :])).sum()
                     / nw_ for k in range(kmax + 1)])
    _x = np.arange(kmax + 1)
    # Log-axis floor so a fractional bar (0.08 cells/walker) is still
    # visible -- but only where the level EXISTS, otherwise an empty level
    # renders as a red stub reading "over cap" where nothing is.
    _F = np.where(hist > 0, 1e-2, 0.0)
    ax[0].bar(_x, np.maximum(hist - h_at, _F), color=CYAN, width=0.72,
              label="below cap")
    ax[0].bar(_x, np.maximum(h_at - h_ov, _F), width=0.72, color=AMBER,
              bottom=np.maximum(hist - h_at, _F), label="at cap")
    if h_ov.sum():
        ax[0].bar(_x, np.maximum(h_ov, _F), width=0.72, color=RED,
                  bottom=np.maximum(hist - h_ov, _F), label="over cap")
    ax[0].legend(loc="upper right", fontsize=8)
    for k, v in enumerate(hist):
        if v <= 0:
            continue
        ax[0].text(k, max(v, 1e-2), f"{v:.0f}" if v >= 1 else f"{v:.2f}",
                   ha="center", va="bottom", color=FG, fontsize=8.5)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-2, hist.max() * 4)
    _st = 1 if kmax <= 8 else 2
    ax[0].set_xticks(np.arange(0, kmax + 1, _st))
    ax[0].set_xlabel("sources in the cell")
    ax[0].set_ylabel(f"cap cells (of {ncell})")
    ax[0].set_title(f"cell occupancy @ iter {NIT-1} (cap {int(cap_row.min())}"
                    f"-{int(cap_row.max())})")

    # (b) the ramp: occupied and at-cap cells per stored iteration. Caps are
    # a SENTINEL (-1) until the GB stage arms them, and "count >= -1" is true
    # of every empty cell -- plotted raw that reads as all 1,232 cells capped
    # before the search even starts. Only plot iterations with a real cap.
    _its, _occ, _tot = [], [], []
    _cap_its, _atcap = [], []
    for i in range(NIT):
        cc = _cell_counts(i)
        capi = cap_cells[min(i, _crow)]
        _its.append(i)
        _occ.append((cc > 0).sum() / nw_)
        _tot.append(cc.sum() / nw_)
        if np.all(capi >= 1):
            _cap_its.append(i)
            _atcap.append((cc >= capi[None, :]).sum() / nw_)
    ax[1].plot(_its, _occ, color=CYAN, lw=2, label="occupied cells")
    ax[1].plot(_cap_its, _atcap, color=AMBER, lw=2, label="at/over cap")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("cap cells")
    ax[1].legend(loc="upper left", fontsize=9)
    axr = ax[1].twinx()
    axr.plot(_its, _tot, color=GREEN, lw=1.4, ls="--")
    axr.set_ylabel("sources / walker", color=GREEN, fontsize=9)
    axr.tick_params(axis="y", colors=GREEN, labelsize=8); axr.grid(False)
    ax[1].set_title("occupancy vs the model")

    # (c) where the occupied cells actually are. Plotted per BAND (8 cells)
    # rather than per cell: 1,232 hairlines across 21 mHz is a moire pattern,
    # not a distribution, and the question here is only "which part of the
    # spectrum is populated".
    K = max(int(round(ncell / max(len(band_edges) - 1, 1))), 1)
    nblk = ncell // K
    occ_cell = (cc_last > 0).mean(axis=0)[:nblk * K].reshape(nblk, K).sum(1)
    fblk = (0.5 * (cap_edges_arr[:-1] + cap_edges_arr[1:])
            )[:nblk * K].reshape(nblk, K).mean(1) * 1e3
    ax[2].fill_between(fblk, 0, occ_cell, color=CYAN, alpha=0.9, lw=0,
                       step="mid", label="model")
    # TRUTH: how many cap cells per band actually CONTAIN a detectable
    # source. The gap between the two curves is the remaining search work,
    # localised in frequency -- which the model curve alone cannot show.
    if DET_F0 is not None:
        _dc = np.clip(np.searchsorted(cap_edges_arr, DET_F0, side="right") - 1,
                      0, ncell - 1)
        _hasdet = np.zeros(ncell, dtype=bool)
        _hasdet[_dc] = True
        _tb = _hasdet[:nblk * K].reshape(nblk, K).sum(1)
        _rng = (fblk >= DET_LO * 1e3) & (fblk <= DET_HI * 1e3)
        ax[2].step(fblk[_rng], _tb[_rng], where="mid", color=RED, lw=1.3,
                   ls=":", label=f"has detectable source "
                                 f"({DET_LO*1e3:.0f}-{DET_HI*1e3:.0f} mHz)")
        ax[2].legend(fontsize=8, loc="upper right")
    ax[2].axhline(K, color=DIM, ls=":", lw=1)
    ax[2].text(fblk[-1], K, f" all {K} cells", color=DIM, fontsize=8,
               va="bottom", ha="right")
    ax[2].set_xlabel("f0 [mHz]")
    ax[2].set_ylabel(f"occupied cells per band")
    ax[2].set_title("where the occupied cells are")
    fig_b64(fig, "gb_cap_cells")

    # A young run may have NO iteration with an armed cap yet (_atcap empty:
    # the caps stay at the -1 sentinel until the GB stage arms them) -- report
    # zero at-cap rather than crashing on the empty list.
    _occ_last = _occ[-1] if _occ else 0.0
    _atcap_last = _atcap[-1] if _atcap else 0.0
    _exact1 = float((cc_last == 1).sum() / nw_)
    CAP_TXT = (
        f"At iteration {NIT-1} the median cold walker holds "
        f"<strong>{_tot[-1]:.0f}</strong> GB sources spread over "
        f"<strong>{_occ_last:.0f} of {ncell}</strong> cap cells "
        f"({100*_occ_last/ncell:.0f}%): <strong>{_exact1:.0f}</strong> cells "
        f"hold exactly one source and <strong>{_atcap_last:.0f}</strong> sit "
        f"at or over their cap.")

# ---- 5a2. HIGH-FREQUENCY RECOVERY CENSUS ----------------------------------
# Injection-vs-recovery above 5 mHz, the direct test of whether source
# ADDING is trustworthy. Not computed here: the optimal SNR of every
# catalogue source against the run's sampled sensitivity needs the 2.3 GB
# WDWD catalogue plus a GBGPU waveform each, which does not belong in a
# monitor that has to run in seconds. `census.py` in the scratchpad
# produces `gb_hi_f_census.npz`; drop it beside the store (or in CWD) and
# this section appears. Without it the page degrades to a note.
CENSUS = None
for _cp in (os.path.join(RUN_DIR, "gb_hi_f_census.npz"),
            "gb_hi_f_census.npz"):
    if os.path.exists(_cp):
        try:
            CENSUS = np.load(_cp)
        except Exception:
            CENSUS = None
        break
CENSUS_TXT = ""
if CENSUS is not None:
    t_f0 = CENSUS["t_f0"]; t_snr = CENSUS["t_snr"]; found = CENSUS["found"]
    c_rec = CENSUS["rec_f0"]; c_occ = CENSUS["occ"]; c_cap = CENSUS["cap"]
    c_ce = CENSUS["cap_edges"]; FCUT = float(CENSUS["FCUT"])
    nc = c_ce.size - 1
    fig, ax = plt.subplots(1, 3, figsize=(15.4, 3.9),
                           gridspec_kw=dict(wspace=0.30))

    # (a) the census itself: every catalogue source, found or not.
    ax[0].scatter(t_f0[~found] * 1e3, t_snr[~found], s=9, color=RED,
                  alpha=0.55, lw=0, label=f"missed ({(~found).sum()})")
    ax[0].scatter(t_f0[found] * 1e3, t_snr[found], s=14, color=GREEN,
                  alpha=0.95, lw=0, label=f"recovered ({found.sum()})")
    ax[0].axhline(7, color=DIM, ls=":", lw=1)
    ax[0].text(t_f0.max() * 1e3, 7, " SNR 7", color=DIM, fontsize=8,
               ha="right", va="bottom")
    ax[0].set_yscale("log"); ax[0].set_xlabel("f0 [mHz]", fontsize=9)
    ax[0].set_ylabel("optimal SNR", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"catalogue GBs above {FCUT*1e3:.0f} mHz")

    # (b) recovery rate vs SNR -- the shape that says whether adding is
    # SNR-ordered (healthy) or arbitrary (not).
    eds = np.array([3, 5, 7, 10, 15, 25, 40, 1e9])
    xs, ys, ns = [], [], []
    for a, b in zip(eds[:-1], eds[1:]):
        m = (t_snr >= a) & (t_snr < b)
        if m.sum() >= 4:
            xs.append(np.sqrt(a * min(b, 60))); ys.append(100 * found[m].mean())
            ns.append(m.sum())
    ax[1].plot(xs, ys, "o-", color=CYAN, lw=2, ms=6)
    for x, y, n in zip(xs, ys, ns):
        ax[1].text(x, y, f" {n}", color=DIM, fontsize=8, va="bottom")
    ax[1].set_xscale("log"); ax[1].set_xlabel("optimal SNR", fontsize=9)
    ax[1].set_ylabel("recovered [%]", fontsize=9); ax[1].set_ylim(0, 100)
    ax[1].set_title("recovery vs SNR (labels = N in bin)")

    # (c) THE CEILING. Detectable sources per cap cell against the cap the
    # cell actually carries: everything to the right of the cap line cannot
    # be represented no matter how well the sampler works.
    ci = np.clip(np.searchsorted(c_ce, t_f0, side="right") - 1, 0, nc - 1)
    loud = t_snr > 7
    nloud = np.bincount(ci[loud], minlength=nc)
    kmax = int(nloud.max())
    # k=0 (cells with nothing detectable in them) is ~80% of the grid and
    # says nothing about the ceiling -- start at 1 so the bars that matter
    # are not flattened against it.
    ks = np.arange(1, kmax + 1)
    hh = np.array([(nloud == k).sum() for k in ks])
    capmax = int(np.max(c_cap))
    cols = [AMBER if k > capmax else CYAN for k in ks]
    ax[2].bar(ks, hh, color=cols, width=0.72)
    for k, v in zip(ks, hh):
        ax[2].text(k, v, f"{v}", ha="center", va="bottom", color=FG,
                   fontsize=8.5)
    ax[2].axvline(capmax + 0.5, color=RED, lw=1.5, ls="--")
    ax[2].text(capmax + 0.62, hh.max() * 0.6, f"cap {capmax}", color=RED,
               fontsize=9)
    ax[2].set_ylim(0, hh.max() * 1.35)
    ax[2].set_xticks(ks)
    ax[2].set_xlabel("detectable sources in the cell", fontsize=9)
    ax[2].set_ylabel("cap cells", fontsize=9)
    ax[2].set_title("what the cap can hold vs what is there")
    fig_b64(fig, "gb_hi_f_census")

    _excl = int(np.clip(nloud - c_cap, 0, None).sum())
    _nloud = int(loud.sum())
    CENSUS_TXT = (
        f"Above {FCUT*1e3:.0f} mHz the catalogue holds <strong>{t_f0.size}</strong> "
        f"sources, <strong>{_nloud}</strong> of them detectable (SNR&gt;7) against "
        f"this run's own sampled sensitivity. The max-logL cold walker has recovered "
        f"<strong>{int(found.sum())}</strong>. Of the {_nloud - int(found[loud].sum())} "
        f"detectable ones still missing, <strong>{_excl}</strong> "
        f"({100*_excl/max(_nloud,1):.0f}% of all detectable sources) are excluded BY "
        f"CONSTRUCTION &mdash; they sit in cap cells that already contain more "
        f"detectable sources than the cap permits.")

# ---- 5a3. CAP-DIVISOR STUDY (pre-rendered) --------------------------------
# A STATIC study, not a per-snapshot panel: it asks what the cap grid could
# represent of the mojito galaxy, independent of how far this run has got.
# `divisor_study.py` in the scratchpad renders it; drop the PNG beside the
# store and it appears here.
for _dp in (os.path.join(RUN_DIR, "gb_cap_divisor_study.png"),
            "gb_cap_divisor_study.png"):
    if os.path.exists(_dp):
        with open(_dp, "rb") as _fh:
            IMGS["gb_cap_divisor"] = base64.b64encode(_fh.read()).decode()
        break

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
        ax[0].set_yscale("log"); ax[0].set_xlabel("f0 [mHz]", fontsize=9); ax[0].set_ylabel("F")
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
    _settings_txt = SETTINGS_TXT
    if not _settings_txt:
        raise FileNotFoundError("run_settings.log not found beside the store")
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
    _win_keep = _win
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

    # ================= F1: the residual spectrum ==========================
    # The field's canonical "is the fit subtracting anything" figure (Katz+
    # 2405.04690 Fig 5): log-log power spectral density against frequency in
    # Hz, with the injected data, the template sum and the residual on one
    # axes, and the noise decomposition of ESA Red Book Fig 2.2 underneath --
    # instrument noise alone, the unresolved galactic foreground alone, and
    # their sum. The GAP between the instrument curve and the sum IS the
    # galactic confusion, so the residual's position relative to those two
    # lines reads directly as "how much of the galaxy is still unmodelled".
    #
    # ORDINATE, stated because the field is not consistent about it: this is
    # the PSD of the TDI *A* channel in 1/Hz. It is not a strain amplitude and
    # it is not an ASD. The run analyses X/Y/Z; A = (Z-X)/sqrt(2) is formed
    # here purely so the data can be compared against lisatools' A2TDISens
    # noise model, which is what the likelihood is weighted by.
    def _AE(x):
        return ((x[2] - x[0]) / np.sqrt(2.0),
                (x[0] - 2 * x[1] + x[2]) / np.sqrt(6.0))

    _W2 = float(np.mean(_win_keep ** 2))

    def _PSD(x):
        return 2.0 * np.abs(x) ** 2 / (N_TD * W_DT * _W2)

    _dA, _ = _AE(data_fd)
    _tA, _ = _AE(_tmpl["gb"] + _tmpl["vgb"])
    _rA, _ = _AE(resid_fd)

    from lisatools.sensitivity import get_sensitivity, A2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as _HTGF)
    _pm = np.median(psd_cold[-1], axis=0)
    _gm = np.median(gal_cold[-1], axis=0)
    _lmod = lisa_models.LISAModel(_pm[0] ** 2, _pm[1] ** 2,
                                  lisa_models.DefaultOrbits(), "sampled")
    _fpos = np.maximum(_fr, FDS.df)
    _Sinst = np.asarray(get_sensitivity(_fpos, sens_fn=A2TDISens,
                                        model=_lmod, stochastic_params=()),
                        float)
    _Ssum = np.asarray(get_sensitivity(_fpos, sens_fn=A2TDISens, model=_lmod,
                                       stochastic_params=tuple(_gm),
                                       stochastic_function=_HTGF), float)
    _Sgal = np.maximum(_Ssum - _Sinst, 1e-60)

    # log-f binning. A PSD is an AVERAGE of periodogram bins, so the reduction
    # is a mean per log-frequency bin -- not the block-MAX a line spectrum
    # wants, and not a median, which would read the template (exactly zero
    # between sources) as identically zero everywhere.
    # Below the run's own GB band the windowed data is dominated by
    # spectral leakage from the enormous sub-mHz TDI content -- a sinc
    # pattern that is real but is not analysed and reads as structure.
    _sel = (_fr >= max(float(band_edges[0]), 3e-4)) & (_fr <= 2.5e-2)
    _fs = _fr[_sel]
    _NB = 300
    _ed = np.logspace(np.log10(_fs[0]), np.log10(_fs[-1]), _NB + 1)
    _bi = np.clip(np.searchsorted(_ed, _fs, side="right") - 1, 0, _NB - 1)
    _cnt = np.bincount(_bi, minlength=_NB).astype(float)
    _fcb = np.sqrt(_ed[:-1] * _ed[1:])
    _ok = _cnt > 0

    def _bmean(v):
        out = np.full(_NB, np.nan)
        out[_ok] = np.bincount(_bi, weights=v, minlength=_NB)[_ok] / _cnt[_ok]
        return out

    _Pd, _Pt = _bmean(_PSD(_dA[_sel])), _bmean(_PSD(_tA[_sel]))
    _Pr = _bmean(_PSD(_rA[_sel]))
    _Ni = _bmean(_Sinst[_sel]); _Ng = _bmean(_Sgal[_sel]); _Ns = _bmean(_Ssum[_sel])

    # whitened residual: real and imaginary parts are each N(0,1) when the
    # noise model is right, so the ratio below sits at 1 and the
    # Anderson-Darling test on them has nothing to reject.
    _zA = _rA[_sel] / np.sqrt(np.maximum(_Ssum[_sel], 1e-60)
                              * N_TD * W_DT * _W2 / 4.0)
    _rat = _bmean(np.abs(_zA) ** 2 / 2.0)

    # Rosati & Littenberg (2410.17180) Fig 4: colour the residual trace by the
    # Anderson-Darling Gaussianity p-value of the whitened residual, so the
    # confusion region identifies ITSELF as the non-Gaussian zone instead of
    # being annotated by hand. The test is run on a fixed 600-sample draw per
    # bin: with 10^4 samples it rejects on effect sizes far too small to care
    # about, and the bins would no longer be comparable to each other.
    _adp = np.full(_NB, np.nan)
    try:
        from scipy.stats import anderson as _anderson
        _rng = np.random.default_rng(0)
        _ord = np.argsort(_bi, kind="stable")
        _zs = _zA[_ord]
        _bnd = np.searchsorted(_bi[_ord], np.arange(_NB + 1))
        for _b in range(_NB):
            _v = _zs[_bnd[_b]:_bnd[_b + 1]]
            if _v.size < 40:
                continue
            _v = np.concatenate([_v.real, _v.imag])
            if _v.size > 600:
                _v = _rng.choice(_v, 600, replace=False)
            _A2 = float(_anderson(_v, dist="norm").statistic)
            _n = _v.size
            _As = _A2 * (1 + 0.75 / _n + 2.25 / _n ** 2)
            if _As < 0.2:
                _p = 1 - np.exp(-13.436 + 101.14 * _As - 223.73 * _As ** 2)
            elif _As < 0.34:
                _p = 1 - np.exp(-8.318 + 42.796 * _As - 59.938 * _As ** 2)
            elif _As < 0.6:
                _p = np.exp(0.9177 - 4.279 * _As - 1.38 * _As ** 2)
            else:
                _p = np.exp(1.2937 - 5.709 * _As + 0.0186 * _As ** 2)
            _adp[_b] = np.log10(float(np.clip(_p, 1e-12, 1.0)))
    except Exception as _e:
        MISSING.append(f"residual Gaussianity test unavailable: {_e!r}")

    fig, ax = plt.subplots(2, 1, figsize=(11.6, 6.8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.5, 1],
                                            hspace=0.07))
    ax[0].plot(_fcb, _Pd, color=CYAN, lw=1.6, label="injected data")
    ax[0].plot(_fcb, _Pt, color=GREEN, lw=1.2, label="template sum (GB + VGB)")
    ax[0].plot(_fcb, _Pr, color=VIOLET, lw=1.4, label="residual")
    ax[0].plot(_fcb, _Ni, color=FG, lw=1.2, ls=":", label="instrument noise")
    ax[0].plot(_fcb, _Ng, color=AMBER, lw=1.2, ls=":",
               label="galactic foreground")
    ax[0].plot(_fcb, _Ns, color=FG, lw=1.3, ls="--", label="their sum")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylabel("PSD, TDI A channel  [1/Hz]")
    ax[0].set_ylim(1e-45, 2e-37)
    ax[0].legend(fontsize=8, loc="upper left", ncols=2)
    _sc = ax[1].scatter(_fcb, _rat, c=_adp, cmap="magma", s=11, vmin=-8,
                        vmax=0, lw=0)
    ax[1].axhline(1.0, color=FG, ls="--", lw=1.0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_ylim(0.25, 40)
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("residual PSD\n/ (noise + foreground)", fontsize=9)
    _cb = fig.colorbar(_sc, ax=ax[1], pad=0.012)
    _cb.set_label("log$_{10}$ p, Anderson-Darling", fontsize=8)
    _cb.ax.tick_params(labelsize=7)
    fig_b64(fig, "f1_resid")

    _gb_band = (_fcb >= 3e-3) & (_fcb <= 1e-2)
    _cr = _Ns / np.maximum(_Ni, 1e-60)
    _ci = int(np.nanargmax(np.where(_ok, _cr, np.nan)))
    DTR.update(
        rat_gal=float(np.nanmedian(_rat[_gb_band])),
        conf_ratio=float(_cr[_ci]), conf_f=float(_fcb[_ci]),
        undersub=int(np.nansum(_Pr[_ok] < _Ni[_ok])),
        # WHERE those bins are decides whether they mean anything. Below the
        # analysed band the windowed data is dominated by leakage from the
        # enormous sub-mHz TDI content -- a sinc pattern whose nulls dip under
        # any smooth noise curve while carrying no model at all. A bin under
        # the instrument curve THERE is a window artefact; one inside the
        # band, where templates are actually subtracted, would be real
        # over-subtraction. Report the split rather than asserting either.
        undersub_hi=float(np.nanmax(np.where(
            _ok & (_Pr < _Ni), _fcb, np.nan))) if np.any(
                _ok & (_Pr < _Ni)) else float("nan"),
        undersub_lo=float(np.nanmin(np.where(
            _ok & (_Pr < _Ni), _fcb, np.nan))) if np.any(
                _ok & (_Pr < _Ni)) else float("nan"),
        undersub_inband=int(np.nansum(
            _ok & (_Pr < _Ni) & (_fcb >= 3e-3))),
        # The deepest shortfall, as a RATIO. A residual that is 3% under a
        # noise curve carrying a 3.6% parameter bias is explained; one that
        # is 2x under it is not, and the two must not read the same.
        undersub_worst=float(np.nanmin(np.where(
            _ok & (_Pr < _Ni), _Pr / np.maximum(_Ni, 1e-60), np.nan)))
        if np.any(_ok & (_Pr < _Ni)) else float("nan"),
        nbins=int(_ok.sum()),
        adp_bad=float(np.nanmean(_adp[np.isfinite(_adp)] < -3.0))
        if np.isfinite(_adp).any() else float("nan"))
    del data_fd, resid_fd, _tmpl, _orb, _comp
except Exception as e:
    MISSING.append(
        f"data/template/residual panels unavailable: {type(e).__name__}: {e}")

# ======================= GB RECOVERY (the science block) ====================
# ONE FROZEN DENOMINATOR. ``gb_truth_3to21.npz`` holds every catalogue GB that
# survives the kappa_max amplitude prefilter, with its exact optimal SNR under
# the run's own fitted noise at iteration 78 and its full parameter vector in
# both the run's 9-column sampling basis and GBGPU's physical basis. The
# detectable (SNR>7) subset over 3-21.94 mHz is 812 sources, and that number is
# used as the denominator of EVERY recovery statement on this page. It is
# deliberately frozen: "detectable" moves as the foreground estimate drops, and
# a denominator that moves with the numerator cannot measure progress.
#
# Everything downstream is recomputed HERE, at the last stored iteration --
# none of it is read from the iteration-15 caches the earlier page was built
# on, which were three times smaller in model size.
SCI = {}
TRU = None
for _tp in (os.path.join(RUN_DIR, "gb_truth_3to21.npz"), "gb_truth_3to21.npz"):
    if os.path.exists(_tp):
        try:
            TRU = np.load(_tp)
        except Exception:
            TRU = None
        break

SCI_TOBS = 7776000.0
try:
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    SCI_TOBS = float(_a["0"]) * float(_a["1"]) * float(_a["2"])
except Exception:
    pass
SCI_DF = 1.0 / SCI_TOBS
FLO, FHI = 3e-3, 21.94e-3          # the band the frozen denominator covers
TOL_BINS = 2.0
NBAND = (FHI - FLO) / SCI_DF       # FD bins in the band

# The truth set is tied to this run's Tobs (SNRs, and the FD bin the match
# tolerance is quoted in). Refuse it on a different observation time rather
# than silently comparing a 23-month model against 3-month detectability.
if TRU is not None and abs(SCI_TOBS - 7776000.0) > 1.0:
    MISSING.append(
        "recovery panels skipped: the frozen detectability set is built for a "
        "3-month observation and this run is not one.")
    TRU = None


def _match_pairs(rf, tf, tol):
    """Globally-greedy ONE-TO-ONE nearest match in f0 within ``tol`` (Hz).

    Greedy on |df| ascending, so the result is independent of input order. A
    per-source nearest-neighbour match is not: it lets two model leaves claim
    the same injection, which inflates completeness by exactly the duplicate
    rate this page is trying to measure.
    """
    if rf.size == 0 or tf.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    o = np.argsort(tf)
    tfs = tf[o]
    lo = np.searchsorted(tfs, rf - tol)
    hi = np.searchsorted(tfs, rf + tol)
    ri, ti = [], []
    for i in range(rf.size):
        for j in o[lo[i]:hi[i]]:
            ri.append(i); ti.append(j)
    if not ri:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    ri = np.asarray(ri); ti = np.asarray(ti)
    d = rf[ri] - tf[ti]
    ur = np.zeros(rf.size, bool); ut = np.zeros(tf.size, bool)
    out = []
    for m in np.argsort(np.abs(d)):
        a_, b_ = ri[m], ti[m]
        if ur[a_] or ut[b_]:
            continue
        ur[a_] = ut[b_] = True
        out.append((a_, b_, d[m]))
    out.sort()
    return (np.array([p[0] for p in out], int),
            np.array([p[1] for p in out], int),
            np.array([p[2] for p in out], float))


def _wilson(k, n, z=1.0):
    """Wilson score interval (z=1 -> 68%); correct at k=0 and k=n, unlike the
    normal approximation, which is what a 0%-recovery bin needs."""
    k = np.asarray(k, float); n = np.maximum(np.asarray(n, float), 1e-9)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(np.maximum(p * (1 - p) / n + z * z / (4 * n * n), 0)) / den
    return c - h, c + h


def _survival(x):
    """UN-NORMALISED survival count: (sorted x, number of sources at or above).

    Unnormalised is the field convention (Littenberg Fig 13, Katz Fig 8): the
    y-intercept then reads directly as the size of the set.
    """
    s = np.sort(np.asarray(x, float))
    return s, np.arange(s.size, 0, -1)


def _nn_bins(fv):
    """Nearest-neighbour |Delta f0| in FD bins within one frequency set."""
    if fv.size < 2:
        return np.zeros(0)
    s = np.sort(fv)
    d = np.diff(s)
    return np.minimum(np.r_[d, np.inf], np.r_[np.inf, d]) / SCI_DF


def _leaf_f0(it_, w):
    """Alive-leaf f0 [Hz] for one cold walker at one stored iteration.

    Sliced column-wise: the chain is chunked (..., 1) on the parameter axis,
    so pulling only f0 reads ~1/9 of the bytes a full-row read would.
    """
    al = g["inds/gb"][it_, 0, 0, w]
    return (g["chain/gb"][it_, 0, 0, w, :, 1] * 1e-3)[al]


if TRU is not None:
    _sel = (TRU["det"] & (TRU["f0"] >= FLO) & (TRU["f0"] <= FHI))
    T_F0 = TRU["f0"][_sel]
    T_SNR = TRU["snr"][_sel]
    T_AMP = TRU["amp"][_sel]
    T_PHYS = TRU["phys"][_sel]
    NDET = int(_sel.sum())
    # Chance rate: the fraction of the band covered by the +-2-bin acceptance
    # windows of the truth set. Any purity number is only meaningful against
    # it, and it is the first thing a reviewer asks for.
    CHANCE = NDET * (2 * TOL_BINS * SCI_DF) / (FHI - FLO)
    _tol = TOL_BINS * SCI_DF

    # ---- per-iteration progress (the max-lnL cold walker OF EACH ITERATION) -
    _wb = np.argmax(ll[:NIT], axis=1).astype(int)
    n_all = np.zeros(NIT, int); n_band = np.zeros(NIT, int)
    n_match = np.zeros(NIT, int)
    for _i in range(NIT):
        _fv = _leaf_f0(_i, int(_wb[_i]))
        n_all[_i] = _fv.size
        _fv = _fv[(_fv >= FLO) & (_fv <= FHI)]
        n_band[_i] = _fv.size
        n_match[_i] = _match_pairs(_fv, T_F0, _tol)[0].size
    _nz = np.nonzero(n_all > 0)[0]
    IT0 = int(_nz[0]) if _nz.size else 0          # the GB-search origin
    NGBIT = NIT - 1 - IT0

    # ---- the last stored iteration, in full ------------------------------
    WB = int(_wb[-1])
    _alive = g["inds/gb"][NIT - 1, 0, 0, WB]
    REC9 = g["chain/gb"][NIT - 1, 0, 0, WB][_alive]
    _inb = (REC9[:, 1] * 1e-3 >= FLO) & (REC9[:, 1] * 1e-3 <= FHI)
    REC9 = REC9[_inb]
    MI, TI, DFH = _match_pairs(REC9[:, 1] * 1e-3, T_F0, _tol)
    MATCHED = np.zeros(REC9.shape[0], bool); MATCHED[MI] = True
    FOUND = np.zeros(NDET, bool); FOUND[TI] = True

    SCI.update(ndet=NDET, chance=CHANCE, it0=IT0, ngbit=NGBIT, walker=WB,
               n_all=int(n_all[-1]), n_band=int(REC9.shape[0]),
               n_match=int(MI.size),
               completeness=MI.size / NDET,
               purity=MI.size / max(REC9.shape[0], 1))

    # ---- waveform-level quantities ---------------------------------------
    # Optimal SNRs and template overlaps come from GBGPU's OWN run_wave and a
    # noise-weighted inner product against lisatools A2TDISens/E2TDISens fed
    # this run's sampled instrument + foreground -- the same route the run's
    # own likelihood uses. Nothing about the waveform or the noise is
    # re-derived here.
    RPHYS = None
    try:
        from gbgpu.gbgpu import GBGPU
        from lisatools import detector as lisa_models
        from lisatools.sensitivity import (get_sensitivity, A2TDISens,
                                           E2TDISens)
        from lisatools.stochastic import (
            HyperbolicTangentGalacticForeground as HTGF)
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container)

        _psd_p = np.median(psd_cold[-1], axis=0)
        _gal_p = np.median(gal_cold[-1], axis=0)
        _lm = lisa_models.LISAModel(_psd_p[0] ** 2, _psd_p[1] ** 2,
                                    lisa_models.DefaultOrbits(), "sampled")
        _nk = dict(model=_lm, stochastic_params=tuple(_gal_p),
                   stochastic_function=HTGF)
        # The noise is a pure function of the FD bin index, so evaluate it
        # ONCE on the whole grid: per-source calls would rebuild the model
        # 3,000 times for numbers that never change.
        _ng = int(2.35e-2 / SCI_DF) + 2
        _fg = np.maximum(np.arange(_ng) * SCI_DF, SCI_DF)
        SA_G = np.asarray(get_sensitivity(_fg, sens_fn=A2TDISens, **_nk), float)
        SE_G = np.asarray(get_sensitivity(_fg, sens_fn=E2TDISens, **_nk), float)

        _orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
        _gbw = GBGPU(force_backend="cpu", orbits=_orb, t0=float(T_REF_SCI))
        _tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
            mc_lims=(0.001, 1.0))
        RPHYS = _tc.both_transforms(np.asarray(REC9, float).copy())

        NW_ = 1024

        def _ae(phys):
            _gbw.run_wave(*[np.ascontiguousarray(phys[:, k]) for k in range(9)],
                          N=NW_, T=SCI_TOBS, dt=2.5, tdi2=True,
                          tdi_channel_setup="AE")
            return (np.asarray(_gbw.A), np.asarray(_gbw.E),
                    np.asarray(_gbw.start_inds).astype(int))

        _A, _E, _s = _ae(RPHYS)
        REC_SNR = np.zeros(RPHYS.shape[0])
        for _i in range(RPHYS.shape[0]):
            if _s[_i] < 0 or _s[_i] + NW_ > SA_G.size:
                continue
            _sa = SA_G[_s[_i]:_s[_i] + NW_]; _se = SE_G[_s[_i]:_s[_i] + NW_]
            REC_SNR[_i] = np.sqrt(max(4.0 * SCI_DF * float(np.sum(
                np.abs(_A[_i]) ** 2 / _sa + np.abs(_E[_i]) ** 2 / _se)), 0.0))

        # phase-maximised, noise-weighted overlap: |<a|b>| / sqrt(<a|a><b|b>).
        # The modulus IS the maximum over an overall phase, so no phase grid
        # is needed.
        _Ar, _Er, _sr = _ae(RPHYS[MI])
        _At, _Et, _st = _ae(T_PHYS[TI])
        MM = np.zeros(MI.size)
        for _i in range(MI.size):
            _o = min(_sr[_i], _st[_i])
            _sp = max(_sr[_i], _st[_i]) + NW_ - _o
            if _o < 0 or _o + _sp > SA_G.size:
                continue
            _sa = SA_G[_o:_o + _sp]; _se = SE_G[_o:_o + _sp]
            _a1 = np.zeros(_sp, complex); _e1 = np.zeros(_sp, complex)
            _a2 = np.zeros(_sp, complex); _e2 = np.zeros(_sp, complex)
            _k = _sr[_i] - _o; _a1[_k:_k + NW_] = _Ar[_i]; _e1[_k:_k + NW_] = _Er[_i]
            _k = _st[_i] - _o; _a2[_k:_k + NW_] = _At[_i]; _e2[_k:_k + NW_] = _Et[_i]
            _n = (np.sum(np.conj(_a1) * _a2 / _sa)
                  + np.sum(np.conj(_e1) * _e2 / _se))
            _n1 = (np.sum(np.abs(_a1) ** 2 / _sa)
                   + np.sum(np.abs(_e1) ** 2 / _se))
            _n2 = (np.sum(np.abs(_a2) ** 2 / _sa)
                   + np.sum(np.abs(_e2) ** 2 / _se))
            MM[_i] = float(np.abs(_n) / np.sqrt(max(_n1 * _n2, 1e-300)))
        MM = np.clip(MM, 0.0, 1.0)
        SCI.update(mm_med=float(np.median(MM)) if MM.size else float("nan"),
                   mm_hi=float(np.mean(MM > 0.9)) if MM.size else 0.0,
                   snr_med=float(np.median(REC_SNR)))
    except Exception as e:
        MISSING.append(f"waveform-level recovery statistics unavailable: "
                       f"{type(e).__name__}: {e}")
        RPHYS, REC_SNR, MM = None, None, np.zeros(0)

    # ---- cross-arm cache -------------------------------------------------
    # The v2/v3 comparison must be made on GB-SEARCH iterations, not absolute
    # ones (v2's first GB leaf lands at iteration 5, v3's at 16), so each arm
    # publishes its own series keyed by that origin and every comparison panel
    # subtracts it.
    ARM_TAG = {"3mo_v3": "v3", "3mo_v4": "v4"}.get(RUN_KIND, "v2")
    try:
        np.savez(f"gf_arm_{ARM_TAG}.npz", n_all=n_all, n_band=n_band,
                 n_match=n_match, it0=IT0, ti=TI, mm=MM,
                 rec_f0=REC9[:, 1] * 1e-3, ndet=NDET)
    except Exception:
        pass
    ARMS = {}
    for _fn in sorted(glob.glob("gf_arm_*.npz")):
        _tag = os.path.basename(_fn)[7:-4]
        try:
            _z = np.load(_fn)
            # An arm cache stores TRUTH-SET INDICES (`ti`), so it is only
            # meaningful against the truth set it was matched to. A cache
            # built against a different denominator (e.g. the original
            # 812-source set vs a rebuilt one) would overlay silently WRONG
            # per-source recovery -- and index out of bounds when the sets
            # differ in size. `ndet` stamped in the cache is the guard.
            if "ndet" in _z and int(_z["ndet"]) != NDET:
                MISSING.append(
                    f"arm cache {_fn} skipped: built against a "
                    f"{int(_z['ndet'])}-source truth set, this page uses "
                    f"{NDET}. Regenerate it by rerunning this script on "
                    "that arm's run dir with the current truth npz.")
                continue
            ARMS[_tag] = _z
        except Exception:
            pass
    ARM_COL = {"v2": CYAN, "v3": AMBER, "v4": GREEN}

    # ================= FIGURES ==========================================
    import matplotlib.colors as _mcol

    if SHOW_MATCH_STATS:
        # ---- F2: completeness AND purity vs GB-search iteration -------------
        fig, ax = plt.subplots(figsize=(11, 3.8))
        axr = ax.twinx(); axr.grid(False)
        for _tag in sorted(ARMS):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _x = np.arange(_D["n_match"].size) - int(_D["it0"])
            _k = _x >= 0
            ax.plot(_x[_k], 100 * _D["n_match"][_k] / NDET, color=_c, lw=2.0,
                    label=f"{_tag} completeness")
            axr.plot(_x[_k], 100 * _D["n_match"][_k]
                     / np.maximum(_D["n_band"][_k], 1), color=_c, lw=1.2, ls="--")
        axr.axhline(100 * CHANCE, color=RED, ls=":", lw=1.2)
        axr.text(1, 100 * CHANCE, f" {100*CHANCE:.1f}% by chance", color=RED,
                 fontsize=8, va="bottom")
        ax.set_xlabel("iterations since the first galactic-binary leaf")
        ax.set_ylabel(f"completeness  [% of {NDET}]")
        axr.set_ylabel("purity  [% of leaves]  (dashed)", color=DIM, fontsize=9)
        axr.tick_params(axis="y", colors=DIM, labelsize=8)
        ax.set_ylim(0, 60); axr.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="lower right")
        fig_b64(fig, "f2_progress")

        # ---- F3: match CDF + survival COUNT ---------------------------------
        fig, ax = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True,
                               gridspec_kw=dict(hspace=0.08))
        for _tag in sorted(ARMS):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _m = np.sort(np.asarray(_D["mm"], float))
            if not _m.size:
                continue
            ax[0].plot(_m, np.arange(1, _m.size + 1) / _m.size, color=_c, lw=1.8,
                       label=f"{_tag}  ({_m.size} matched)")
            _s, _n = _survival(_m)
            ax[1].plot(_s, _n, color=_c, lw=1.8)
        ax[0].set_ylabel("cumulative fraction"); ax[0].set_ylim(0, 1)
        ax[0].legend(fontsize=8, loc="upper left")
        ax[1].axhline(NDET, color=FG, ls="--", lw=1.0)
        ax[1].text(0.02, NDET, f" {NDET} detectable injections", color=FG,
                   fontsize=8, va="bottom")
        ax[1].set_yscale("log"); ax[1].set_ylim(1, 2500); ax[1].set_xlim(0, 1)
        ax[1].set_xlabel("phase-maximised overlap with the matched injection")
        ax[1].set_ylabel("sources at or above")
        fig_b64(fig, "f3_match")

    if RPHYS is not None:
        _rf = RPHYS[:, 1]; _ra_ = RPHYS[:, 0]
        _fgr = np.logspace(np.log10(FLO), np.log10(FHI), 220)
        _athr = None
        for _kp in (os.path.join(RUN_DIR, "kappa_grid.npz"), "kappa_grid.npz"):
            if os.path.exists(_kp):
                _kg = np.load(_kp)
                _athr = 7.0 / np.interp(_fgr, _kg["fgrid"], _kg["fit"])
                break

        # ---- F4: amplitude vs frequency + the three-way recovery scatter --
        fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.5))
        a_ = ax[0]
        a_.scatter(T_F0, T_AMP, s=5, color=DIM, alpha=0.35, lw=0,
                   label="detectable injections")
        _sc = a_.scatter(_rf, _ra_, c=np.clip(REC_SNR, 4, None), s=14,
                         marker=".", cmap="cool", lw=0,
                         norm=_mcol.LogNorm(vmin=4, vmax=200),
                         label="resolved GBs")
        if _athr is not None:
            a_.plot(_fgr, _athr, color=FG, lw=1.5,
                    label="instrument sensitivity (SNR = 7)")
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_xlabel("Frequency [Hz]"); a_.set_ylabel("Strain amplitude")
        a_.legend(fontsize=8, loc="lower left")
        _cb = fig.colorbar(_sc, ax=a_, pad=0.015); _cb.set_label("SNR", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        b_ = ax[1]
        if SHOW_MATCH_STATS:
            b_.scatter(T_F0[~FOUND], T_AMP[~FOUND], s=17, marker="x", color=RED,
                       lw=0.8, alpha=0.7,
                       label=f"detectable, not recovered ({int((~FOUND).sum())})")
            b_.scatter(_rf[~MATCHED], _ra_[~MATCHED], s=22, facecolors="none",
                       edgecolors=VIOLET, lw=0.8, alpha=0.85,
                       label=f"recovered, no match ({int((~MATCHED).sum())})")
            b_.scatter(_rf[MATCHED], _ra_[MATCHED], s=12, color=GREEN, lw=0,
                       alpha=0.95, label=f"recovered and matched ({int(MATCHED.sum())})")
        else:
            # Neutral overlay -- injections and model in one plane, the eye
            # does the comparison; no proxy-match classification.
            b_.scatter(T_F0, T_AMP, s=17, marker="x", color=DIM, lw=0.8,
                       alpha=0.6, label=f"detectable injections ({NDET})")
            b_.scatter(_rf, _ra_, s=12, color=GREEN, lw=0, alpha=0.9,
                       label=f"model sources ({_rf.size})")
        if _athr is not None:
            b_.plot(_fgr, _athr, color=FG, lw=1.5)
        b_.set_xscale("log"); b_.set_yscale("log")
        b_.set_xlabel("Frequency [Hz]"); b_.set_ylim(*a_.get_ylim())
        b_.legend(fontsize=8, loc="lower left")
        fig_b64(fig, "f4_pop")

        # ---- F8: sky map ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        ax.scatter(np.mod(T_PHYS[:, 7], 2 * np.pi), T_PHYS[:, 8], s=4,
                   color=DIM, alpha=0.35, lw=0, label="detectable injections")
        _sc = ax.scatter(np.mod(RPHYS[:, 7], 2 * np.pi), RPHYS[:, 8],
                         c=np.clip(REC_SNR, 4, None), s=13, cmap="cool", lw=0,
                         norm=_mcol.LogNorm(vmin=4, vmax=200))
        ax.set_xlabel("right ascension [rad]")
        ax.set_ylabel("declination [rad]")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(-1.6, 1.6)
        _cb = fig.colorbar(_sc, ax=ax, pad=0.012); _cb.set_label("SNR", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc="upper left")
        fig_b64(fig, "f8_sky")

        if SHOW_MATCH_STATS:
            # ---- F7: recovered vs injected parameters --------------------------
            _R = RPHYS[MI]; _T = T_PHYS[TI]
            _db = DFH / SCI_DF
            _lnA = np.log(np.maximum(_R[:, 0], 1e-40) / np.maximum(_T[:, 0], 1e-40))
            _cc = (np.sin(_R[:, 8]) * np.sin(_T[:, 8])
                   + np.cos(_R[:, 8]) * np.cos(_T[:, 8])
                   * np.cos(_R[:, 7] - _T[:, 7]))
            _sep = np.degrees(np.arccos(np.clip(_cc, -1, 1)))
            fig, ax = plt.subplots(2, 3, figsize=(13.6, 6.0))
            ax[0][0].hist(np.clip(_db, -2, 2), bins=40, color=GREEN, alpha=0.9)
            ax[0][0].axvline(0, color=FG, ls="--", lw=1)
            ax[0][0].set_xlabel(r"$\Delta f_0$  [FD bins]")
            ax[0][1].hist(np.clip(_lnA, -2, 2), bins=40, color=GREEN, alpha=0.9)
            ax[0][1].axvline(0, color=FG, ls="--", lw=1)
            ax[0][1].set_xlabel(r"$\ln(A_{\rm rec}/A_{\rm cat})$")
            ax[0][2].hist(_sep, bins=40, color=GREEN, alpha=0.9)
            ax[0][2].set_xlabel("sky separation [deg]")
            for _k in range(3):
                ax[0][_k].set_ylabel("matched sources", fontsize=9)
            # BOTTOM ROW: error histograms centred on zero, NOT a
            # recovered-vs-injected scatter with a y = x diagonal -- that idiom
            # does not appear anywhere in this literature. Zero-centred residual
            # histograms are what Littenberg 2011 Fig 6 / Strub 2403.15318 Fig 5
            # use, and they put the bias and the spread on the same axis.
            _dpsi = (_R[:, 6] - _T[:, 6] + np.pi / 4) % (np.pi / 2) - np.pi / 4
            _panels = [
                (np.cos(_R[:, 5]) - np.cos(_T[:, 5]), r"$\Delta$ cos $\iota$", 2.0),
                (_dpsi, r"$\Delta\psi$ wrapped to $\pi/2$  [rad]", np.pi / 4),
                ((_R[:, 2] - _T[:, 2]) * 1e16,
                 r"$\Delta\dot f_0$  [$10^{-16}$ Hz/s]", None)]
            for _ax, (_dv, _lb, _cl) in zip(ax[1], _panels):
                if _cl is None:
                    _cl = float(np.percentile(np.abs(_dv), 90)) or 1.0
                _ax.hist(np.clip(_dv, -_cl, _cl), bins=40, color=GREEN, alpha=0.9)
                _ax.axvline(0, color=FG, ls="--", lw=1)
                _ax.set_xlabel(_lb)
                _ax.set_ylabel("matched sources", fontsize=9)
            fig.tight_layout()
            fig_b64(fig, "f7_params")
            # A SECOND encoding of the same matched set, kept deliberately even
            # though this idiom does not appear in the LISA galactic-binary
            # literature (which uses zero-centred error histograms, above). It
            # answers a different question: the histogram shows the size of the
            # error, the diagonal shows whether the parameter is being RECOVERED
            # at all, i.e. whether the points know about the injected value or
            # merely scatter over the prior. For cos iota at these signal
            # strengths those are visibly different statements.
            fig, ax2 = plt.subplots(1, 3, figsize=(13.6, 3.5))
            _sc_panels = [
                (np.cos(_R[:, 5]), np.cos(_T[:, 5]), r"cos $\iota$", None),
                (_R[:, 6] % (np.pi / 2), _T[:, 6] % (np.pi / 2),
                 r"$\psi$ mod $\pi/2$  [rad]", None),
                (_R[:, 2] * 1e16, _T[:, 2] * 1e16,
                 r"$\dot f_0$  [$10^{-16}$ Hz/s]", 98.0)]
            for _ax, (_rv, _tv, _lb, _q) in zip(ax2, _sc_panels):
                _ax.scatter(_tv, _rv, s=10, color=GREEN, alpha=0.6, lw=0)
                if _q is None:
                    _l2 = min(np.min(_tv), np.min(_rv))
                    _h2 = max(np.max(_tv), np.max(_rv))
                else:
                    _l2, _h2 = np.percentile(_tv, [100 - _q, _q])
                    _pd2 = 0.8 * (_h2 - _l2); _l2 -= _pd2; _h2 += _pd2
                _ax.plot([_l2, _h2], [_l2, _h2], color=FG, ls="--", lw=1.1)
                _ax.set_xlim(_l2, _h2); _ax.set_ylim(_l2, _h2)
                _ax.set_xlabel("injected  " + _lb)
                _ax.set_ylabel("recovered", fontsize=9)
            fig.tight_layout()
            fig_b64(fig, "f7_scatter")
            SCI["ci_corr"] = float(np.corrcoef(np.cos(_R[:, 5]),
                                               np.cos(_T[:, 5]))[0, 1])
            SCI.update(df_tight=float(np.mean(np.abs(_db) < 0.5)),
                       lnA_med=float(np.median(np.abs(_lnA))),
                       sep_med=float(np.median(_sep)),
                       sep_tight=float(np.mean(_sep < 10.0)))

    # ---- F5: source counts vs frequency --------------------------------
    _f0_cat = None
    try:
        with h5py.File(os.path.join(
                MOJITO_CAT_DIR, "catalogues",
                "wdwd_cat_mojito_lite_processed.hdf5"), "r") as _wf:
            _f0_cat = _wf["Binaries"]["GW22FrequencySSBFrame"][:]
    except Exception:
        _f0_cat = None
    _eds = np.logspace(np.log10(FLO), np.log10(FHI), 26)
    _hdet, _ = np.histogram(T_F0, bins=_eds)
    _hrec, _ = np.histogram(T_F0[FOUND], bins=_eds)
    _fc = np.sqrt(_eds[:-1] * _eds[1:])
    if SHOW_MATCH_STATS:
        fig, ax = plt.subplots(2, 1, figsize=(10.6, 5.3), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1],
                                                hspace=0.08))
        ax0, ax1 = ax[0], ax[1]
    else:
        # Match-free variant: density comparison only (model leaves vs the
        # truth populations), no recovered-fraction panel.
        fig, ax0 = plt.subplots(figsize=(10.6, 3.9))
        ax1 = None
    if _f0_cat is not None:
        _m = (_f0_cat >= FLO) & (_f0_cat <= FHI)
        _hall, _ = np.histogram(_f0_cat[_m], bins=_eds)
        ax0.stairs(_hall, _eds, color=DIM, fill=True, alpha=0.4,
                   label=f"all catalogue ({int(_m.sum()):,})")
        SCI["n_cat_band"] = int(_m.sum())
    ax0.stairs(_hdet, _eds, color=CYAN, lw=1.7, label=f"detectable ({NDET})")
    if SHOW_MATCH_STATS:
        ax0.stairs(_hrec, _eds, color=GREEN, fill=True, alpha=0.9,
                   label=f"recovered ({int(FOUND.sum())})")
    else:
        _hmod, _ = np.histogram(_rf, bins=_eds)
        ax0.stairs(_hmod, _eds, color=GREEN, fill=True, alpha=0.9,
                   label=f"model sources ({int(_rf.size)})")
    ax0.set_yscale("log"); ax0.set_ylim(0.5, None)
    ax0.set_ylabel("sources per bin"); ax0.legend(fontsize=8)
    if ax1 is not None:
        _lo, _hi = _wilson(_hrec, _hdet)
        _fr = np.where(_hdet > 0, _hrec / np.maximum(_hdet, 1), np.nan)
        ax1.errorbar(_fc, 100 * _fr,
                     yerr=[100 * (_fr - _lo), 100 * (_hi - _fr)], fmt="o",
                     ms=3.5, color=GREEN, ecolor=GREEN, alpha=0.9, capsize=2)
        ax1.set_xscale("log"); ax1.set_ylim(0, 100)
        ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("recovered [%]")
    else:
        ax0.set_xscale("log"); ax0.set_xlabel("Frequency [Hz]")
    fig_b64(fig, "f5_counts")

    if SHOW_MATCH_STATS:
        # ---- F6: completeness vs SNR, with Wilson intervals ------------------
        _seds = np.array([7, 10, 15, 25, 1e9])
        _lab = ["7-10", "10-15", "15-25", "25+"]
        fig, ax = plt.subplots(figsize=(9.0, 3.7))
        _tags = sorted(ARMS)
        for _q, _tag in enumerate(_tags):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _fnd = np.zeros(NDET, bool); _fnd[np.asarray(_D["ti"], int)] = True
            _x, _y, _el, _eh = [], [], [], []
            for _j, (_a2, _b2) in enumerate(zip(_seds[:-1], _seds[1:])):
                _m = (T_SNR >= _a2) & (T_SNR < _b2)
                if not _m.sum():
                    continue
                _p = _fnd[_m].mean()
                _l, _h = _wilson(_fnd[_m].sum(), _m.sum())
                _x.append(_j + (_q - (len(_tags) - 1) / 2) * 0.10)
                _y.append(100 * _p); _el.append(100 * (_p - _l))
                _eh.append(100 * (_h - _p))
            _n = int(_D["n_match"][-1])
            _g = int(_D["n_match"].size) - 1 - int(_D["it0"])
            ax.errorbar(_x, _y, yerr=[_el, _eh], fmt="o-", ms=5, color=_c, lw=1.6,
                        capsize=3, label=f"{_tag}, {_g} GB-search iterations")
        for _j, (_a2, _b2) in enumerate(zip(_seds[:-1], _seds[1:])):
            _m = (T_SNR >= _a2) & (T_SNR < _b2)
            ax.text(_j, 3, f"n={int(_m.sum())}", ha="center", color=DIM, fontsize=8)
        ax.set_xticks(range(len(_lab))); ax.set_xticklabels(_lab)
        ax.set_xlabel("optimal SNR of the injection")
        ax.set_ylabel("recovered [%]"); ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper left")
        fig_b64(fig, "f6_snr")

    # ---- F10: nearest-neighbour separation survival ----------------------
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    for _tag in sorted(ARMS):
        _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
        _s, _n = _survival(_nn_bins(np.asarray(_D["rec_f0"], float)))
        ax.plot(np.maximum(_s, 1e-2), _n, color=_c, lw=1.8,
                label=f"{_tag} model leaves")
    _s, _n = _survival(_nn_bins(T_F0))
    ax.plot(np.maximum(_s, 1e-2), _n, color=FG, lw=1.4, ls="--",
            label=f"detectable injections ({NDET})")
    ax.axvline(TOL_BINS, color=RED, ls=":", lw=1.2)
    ax.text(TOL_BINS * 1.1, 1.5, " match tolerance", color=RED, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e4); ax.set_ylim(1, 2000)
    ax.set_xlabel(r"nearest-neighbour $|\Delta f_0|$  [FD bins]")
    ax.set_ylabel("sources at or above")
    ax.legend(fontsize=8, loc="lower left")
    fig_b64(fig, "f10_nn")
    _nn = _nn_bins(REC9[:, 1] * 1e-3)
    SCI.update(nn_half=float(np.mean(_nn < 0.5)) if _nn.size else 0.0,
               nn_tol=float(np.mean(_nn < TOL_BINS)) if _nn.size else 0.0,
               nn_half_t=float(np.mean(_nn_bins(T_F0) < 0.5)))
    # unmatched leaves sitting on an injection another leaf already claimed --
    # the "two templates, one source" blending mode, counted rather than
    # inferred from a duplicate percentage.
    if TI.size and int((~MATCHED).sum()):
        _d = np.abs((REC9[~MATCHED, 1] * 1e-3)[:, None]
                    - T_F0[TI][None, :]).min(axis=1) / SCI_DF
        SCI["blend_b"] = int((_d <= TOL_BINS).sum())
        SCI["n_unmatched"] = int((~MATCHED).sum())


# ---- F9: verification binaries, restricted to the ones that are there -----
# At three months only ELEVEN of the 55 catalogue verification binaries clear
# SNR 7; the median VGB optimal SNR is 1.6. Showing all 55 posteriors, as the
# page used to, showed 44 prior-dominated distributions next to 11 real ones
# with nothing distinguishing them -- a reader inevitably read the flat ones
# as failures of the fit rather than as an absence of signal. The headline
# panel is therefore the detectable subset; the rest are stated, not plotted.
vgb_last = vgb_c[-min(3, VGB_NIT):].reshape(-1, 55, 5)   # (S, 55, 5)
snr = np.sqrt(np.clip(np.nanmean(vgb_hh[-1], axis=0), 0, None))  # (55,)
order = np.argsort(snr)[::-1]
VGB_DET = np.nonzero(snr > 7.0)[0]
VGB_DET = VGB_DET[np.argsort(snr[VGB_DET])[::-1]]
VGB_N_DET = int(VGB_DET.size)
med = np.median(vgb_last[:, :, 0], axis=0)
lo = np.percentile(vgb_last[:, :, 0], 16, axis=0)
hi = np.percentile(vgb_last[:, :, 0], 84, axis=0)

_lab = [(VGB_IDS[k] if VGB_IDS else f"leaf {k}") for k in VGB_DET]
_y = np.arange(VGB_N_DET)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12.4, 0.34 * max(VGB_N_DET, 8) + 1.6),
                       sharey=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
ax[0].errorbar(med[VGB_DET], _y,
               xerr=[med[VGB_DET] - lo[VGB_DET], hi[VGB_DET] - med[VGB_DET]],
               fmt="o", ms=4, color=GREEN, ecolor=GREEN, capsize=2, lw=1.2)
if VGB_TRUTH is not None:
    ax[0].plot(VGB_TRUTH[VGB_DET, 0], _y, "|", ms=13, mew=1.8, color=CYAN,
               ls="none", label="catalogue distance")
    ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_yticks(_y); ax[0].set_yticklabels(_lab, fontsize=8)
ax[0].set_xlabel("distance [kpc]")
ax[0].set_title(f"the {VGB_N_DET} verification binaries with SNR > 7  "
                f"(median +/- 1 sigma)", fontsize=10)
ax[1].barh(_y, snr[VGB_DET], color=VIOLET, alpha=0.9, height=0.55)
ax[1].axvline(7, color=RED, ls=":", lw=1.2)
ax[1].set_xscale("log"); ax[1].set_xlabel("optimal SNR")
ax[1].set_title("SNR against the fitted noise", fontsize=10)
if VGB_F0 is not None:
    for _i, _k in enumerate(VGB_DET):
        ax[1].text(snr[_k] * 1.06, _y[_i], f" {VGB_F0[_k]:.2f} mHz",
                   fontsize=7, color=DIM, va="center")
fig.tight_layout()
fig_b64(fig, "f9_vgb")

# SNR against frequency for ALL 55, with the detection threshold, so the
# "44 are prior-dominated" statement is visible rather than asserted.
fig, ax = plt.subplots(figsize=(11, 3.2))
_x = VGB_F0 if VGB_F0 is not None else np.arange(55).astype(float)
ax.plot(_x[snr <= 7], snr[snr <= 7], "o", ms=4, color=DIM, alpha=0.75,
        label=f"prior-dominated ({55 - VGB_N_DET})")
ax.plot(_x[snr > 7], snr[snr > 7], "o", ms=6, color=GREEN,
        label=f"SNR > 7 ({VGB_N_DET})")
ax.axhline(7, color=RED, ls=":", lw=1.2)
if VGB_F0 is not None:
    ax.set_xscale("log")
    ax.set_xlabel("catalogue f0 [mHz]")
else:
    ax.set_xlabel("VGB leaf index")
ax.set_yscale("log"); ax.set_ylabel("optimal SNR")
ax.legend(fontsize=8, loc="upper left")
fig_b64(fig, "f9_vgb_snr")
VGB_SNR_MED = float(np.median(snr))

# ---- RESTORED: the full 55-leaf VGB panels ---------------------------
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

    # ALL 55 leaves (2026-08-16): the redesign cut this to the 11
    # detectable ones. Detectable first so the picker still opens on a
    # real posterior, then the rest in descending SNR.
    _corner_order = list(VGB_DET) + [k for k in order if k not in set(VGB_DET)]
    for leaf in _corner_order:
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

# ---- RESTORED: run mechanics. Engineering instrumentation, so it lives
# in the collapsed appendix rather than the body -- but it is the only
# view of where the wall time goes and what the devices are holding.
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
# ============================ HTML ==========================================
stage_now = "?"
for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]):
    if not s_:
        stage_now = k; break
chips = "".join(
    f'<span class="chip {"done" if s_ else ("now" if k == stage_now else "")}">'
    f'{o}. {k}{" &#10003;" if s_ else ""}</span>'
    for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---- the like-for-like arm table ------------------------------------------
# The two arms started their galactic-binary search at DIFFERENT absolute
# iterations (v2's first GB leaf lands at iteration 5, v3's at 16), so any
# comparison at a shared absolute iteration silently gives v2 eleven extra
# search steps. Everything below is indexed on iterations SINCE the first GB
# leaf, and the table is cut at the shorter arm's length.
ARM_TABLE = ""
# ARMS is built inside the GB-analysis section, which a very young store
# (too few GB iterations for the match machinery) skips entirely -- the
# cross-arm table then simply has no data to show.
if len(globals().get("ARMS", {})) >= 2 and SCI and SHOW_MATCH_STATS:
    _K = min(int(D["n_match"].size) - 1 - int(D["it0"]) for D in ARMS.values())
    _rows = []
    for _t in sorted(ARMS):
        _D = ARMS[_t]
        _i = int(_D["it0"]) + _K
        _rows.append((_t, int(_D["n_all"][_i]), int(_D["n_match"][_i]),
                      int(_D["n_match"][_i]) / SCI["ndet"],
                      int(_D["n_match"][_i]) / max(int(_D["n_band"][_i]), 1),
                      int(_D["n_match"].size) - 1 - int(_D["it0"])))
    _hdr = "".join(f'<th style="text-align:right;padding:4px 0 4px 20px">{r[0]}'
                   f'</th>' for r in _rows)
    def _row(lbl, fn):
        return ("<tr><td style='padding:3px 0'>" + lbl + "</td>"
                + "".join("<td style='text-align:right;padding:3px 0 3px 20px'>"
                          + fn(r) + "</td>" for r in _rows) + "</tr>")
    ARM_TABLE = f"""
<table style="border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:10px">
<tr style="border-bottom:1px solid var(--line)"><th style="text-align:left;padding:4px 0">
at {_K} galactic-binary search iterations</th>{_hdr}</tr>
{_row("model sources", lambda r: f"{r[1]:,}")}
{_row("matched to a detectable injection", lambda r: f"{r[2]:,}")}
{_row("completeness", lambda r: pct(r[3]))}
{_row("purity", lambda r: pct(r[4]))}
{_row("search iterations completed in total", lambda r: f"{r[5]}")}
</table>"""

# ---- captions, every number read off the arrays that made the figure ------
if SCI:
    cap_f2 = (
        f"Completeness (solid, left axis) is the share of the {SCI['ndet']} "
        f"detectable injections matched by a model source within "
        f"{TOL_BINS:.0f} frequency bins; purity (dashed, right axis) is the "
        f"share of model sources that so match. Now "
        f"{pct(SCI['completeness'])} and {pct(SCI['purity'])}.")
    cap_f3 = (
        f"Noise-weighted overlap between each matched pair, maximised over an "
        f"overall phase. Read the lower panel's height at any overlap as the "
        f"NUMBER of sources recovered that well. Median "
        f"{SCI.get('mm_med', float('nan')):.2f}; "
        f"{pct(SCI.get('mm_hi', 0))} exceed 0.9.")
    cap_f4 = (
        f"Left: every model source in the amplitude-frequency plane, coloured "
        f"by optimal SNR. Right: the same plane split three ways. Recovery "
        f"tracks amplitude, and the misses concentrate along the faint edge "
        f"rather than anywhere structural.")
    cap_f5 = (
        f"Where the sources are and where they are being found. The lower "
        f"panel is the per-bin recovery fraction with 68% Wilson intervals. "
        f"Recovery climbs steeply above 8 mHz, which is the galaxy thinning "
        f"out rather than the search improving there.")
    cap_f6 = (
        f"Recovery against injected SNR, with 68% Wilson intervals, both arms "
        f"on equal footing. The monotone rise is the health check: a search "
        f"that adds sources arbitrarily would be flat here. This axis is "
        f"ours, not a field convention.")
    cap_f7 = (
        f"Recovered minus injected for the matched pairs. Frequency is in "
        f"bins, amplitude is a log ratio, angles are raw. "
        f"{pct(SCI.get('df_tight', 0), 0)} of matches sit within half a "
        f"frequency bin; the median sky offset is "
        f"{SCI.get('sep_med', float('nan')):.0f} degrees.")
    cap_f8 = (
        f"Recovered sources over the injected population, coloured by SNR. "
        f"The bulge dominates both. Sky is the weakest-constrained "
        f"coordinate at these signal strengths, so scatter here is expected "
        f"and is quantified in the parameter panel.")
    cap_f10 = (
        f"How close model sources sit to each other in frequency. Below the "
        f"match tolerance the model is denser than the injections, which is "
        f"two templates sharing one source; "
        f"{pct(SCI.get('nn_half', 0), 1)} of leaves have a neighbour within "
        f"half a bin against {pct(SCI.get('nn_half_t', 0), 1)} of injections.")
else:
    cap_f2 = cap_f3 = cap_f4 = cap_f5 = cap_f6 = cap_f7 = cap_f8 = cap_f10 = ""
if SCI and not SHOW_MATCH_STATS:
    # Match-criterion content is gated off this page (user ruling
    # 2026-08-19): truths stay in every overlay, but nothing is classified
    # or counted by the page's own 2-bin proxy match.
    cap_f4 = (
        "Left: every model source in the amplitude-frequency plane, "
        "coloured by optimal SNR, over the detectable injections (grey). "
        "Right: injections and model overlaid in the same plane. The "
        "comparison is visual; no match criterion is applied on this page.")
    cap_f5 = (
        "Source density per frequency bin: the full catalogue, its "
        "detectable subset, and the model population. The gap between the "
        "green and cyan curves is read by eye; no per-source matching is "
        "applied.")
    cap_f10 = (
        "How close model sources sit to each other in frequency, against "
        "the same distribution for the injections. Sub-bin spacing in the "
        "model relative to the injections indicates template sharing.")

if DTR:
    _d = DTR
    cap_f1 = (
        f"Power spectral density of the TDI A channel &mdash; not a strain "
        f"amplitude, and not an ASD. The gap between the instrument curve and "
        f"their sum is the galactic confusion, at most a factor "
        f"{_d.get('conf_ratio', float('nan')):.2f} in power, near "
        f"{_d.get('conf_f', float('nan')) * 1e3:.1f} mHz. {_d['n_gb']} "
        f"galactic-binary and {_d['n_vgb']} verification-binary templates are "
        f"subtracted here.")
    cap_f1b = (
        f"Lower panel: residual power over the fitted noise-plus-foreground "
        f"model, coloured by the Anderson&ndash;Darling Gaussianity p-value of "
        f"the whitened residual &mdash; dark bins are where the model is still "
        f"incomplete. The residual stays above the instrument-only curve in "
        f"{_d['nbins'] - _d['undersub']} of {_d['nbins']} bins; "
        f"{_d['undersub']} dip below it, spanning "
        f"{_d.get('undersub_lo', float('nan')) * 1e3:.1f}&ndash;"
        f"{_d.get('undersub_hi', float('nan')) * 1e3:.1f} mHz, and the "
        f"deepest is {100 * (1 - _d.get('undersub_worst', 1)):.0f}% under. "
        f"Read that against the instrument model itself: the fitted "
        f"Soms sits {100 * NOISE_BIAS[0]:+.1f}% from injection, which is "
        f"{100 * ((1 + NOISE_BIAS[0]) ** 2 - 1):+.1f}% in power, so a curve "
        f"drawn a few percent high will sit above a correct residual over "
        f"exactly the band where that parameter dominates. Shortfalls of "
        f"that size are the noise model, not over-subtraction; a bin far "
        f"under would be a different statement.")
else:
    cap_f1 = cap_f1b = ""

# ---- captions for the restored data/template/residual panels --------------
# Condensed from the pre-redesign page: same numbers, read off the same
# arrays, without the method narrative that now lives in the appendix.
if DTR:
    _d = DTR
    dtr_fd_cap = (
        f"Rows are the TDI channels the run analyses; columns are data, "
        f"template sum and residual. Grey is the data, repeated faintly under "
        f"the other two columns so the comparison is direct; green is the "
        f"{_d['n_gb']} galactic-binary templates, violet the {_d['n_vgb']} "
        f"verification binaries, cyan the residual. Dotted red marks the "
        f"run's own band edges. At the loudest recovered source, "
        f"{_d['chk_f0']:.5f} mHz, the peak bin falls by a factor "
        f"{_d['chk_dpk'] / max(_d['chk_rpk'], 1e-99):.0f} and the power in the "
        f"surrounding 81 bins drops to "
        f"{100 * _d['chk_rp'] / max(_d['chk_dp'], 1e-99):.1f}% of the data. "
        f"Across the whole band only "
        f"{100 * (1 - _d['band_rp'] / max(_d['band_dp'], 1e-99)):.1f}% of the "
        f"power has been removed &mdash; the unresolved galaxy is still there.")
    dtr_wdm_cap = (
        f"The same three states on the run's own time-frequency grid: "
        f"{_d['wdm_shape'][1]} layers of {_d['layer_df'] * 1e3:.4f} mHz by "
        f"{int(_d['layer_dt'])} s, max-pooled {_d['wdm_dec']}&times; in time. "
        f"One shared linear scale per channel row, keyed to that row's data "
        f"panel, so a shrinking residual renders darker instead of "
        f"rescaling itself back to full brightness. The horizontal tracks with "
        f"annual brightness modulation are the recovered sources.")
    dtr_note = (
        f"Both are built for cold walker {_d['walker']}, the maximum-likelihood "
        f"walker of the last stored iteration. Noise and foreground are not in "
        f"the template sum &mdash; they shape the sensitivity the likelihood "
        f"weights by, they are not subtracted, so the unresolved galaxy stays "
        f"in the residual by construction.")
else:
    dtr_fd_cap = dtr_wdm_cap = dtr_note = ""

NOISE_TXT = " and ".join(f"{100 * b:+.1f}%" for b in NOISE_BIAS)

# ---- ONE run-health line, in place of ~1,900 words of failure forensics ----
# The OOM, cap-ramp, ghost-guard and Doppler-offset investigations that used to
# open this page are engineering history: they belong in the run log and in the
# tracker, not in the first thing a collaborator reads. What a reader needs
# from the top of a status page is how far each arm got and whether to trust it
# as converged.
_arm_bits = []
for _t in sorted(globals().get("ARMS", {})):
    _D = ARMS[_t]
    _arm_bits.append(f"{_t} has completed "
                     f"{int(_D['n_match'].size) - 1 - int(_D['it0'])} "
                     f"galactic-binary search iterations")
_ended = ("ended at iteration 80 on a GPU memory limit"
          if RUN_KIND == "3mo" and NIT >= 80 else
          f"has stored {NIT} iterations")
RUN_HEALTH = (
    f"<strong>Run health.</strong> This arm {_ended}. "
    + ("; ".join(_arm_bits) + ". " if _arm_bits else "")
    + "Neither arm has converged, so every number here is a progress readout "
      "rather than a result.")

missing_html = "".join(f"<li>{m}</li>" for m in MISSING)


# ---- match-criterion fragments (SHOW_MATCH_STATS gate) --------------------
if SHOW_MATCH_STATS:
    KPI_MATCH = f"""  <div><b>{SCI.get("n_match", 0):,}</b><span>matched to an injection</span></div>
  <div><b>{pct(SCI["completeness"]) if SCI else "&mdash;"}</b><span>completeness</span></div>
  <div><b>{pct(SCI["purity"]) if SCI else "&mdash;"}</b><span>purity</span></div>"""
    REC_MATCH_PANELS = f"""<div class="panel">{img("f2_progress", "completeness and purity vs GB-search iteration")}
<div class="caption">{cap_f2}</div></div>
<div class="panel">{img("f3_match", "overlap CDF and survival count")}
<div class="caption">{cap_f3}</div></div>
<div class="panel">{img("f6_snr", "completeness vs SNR")}
<div class="caption">{cap_f6}</div></div>"""
    PARAMS_SECTION = f"""<section id="params"><h2>Parameter Recovery</h2>
<div class="panel">{img("f7_params", "recovered minus injected parameters")}
<div class="caption">{cap_f7}<br><em>Distance, chirp mass and the frequency-derivative
ratio are not shown: the likelihood constrains only their combinations, so scatter
along that direction is degeneracy, not error.</em></div></div>
<div class="panel">{img("f7_scatter", "recovered vs injected")}
<div class="caption">The same matched sources as recovered against injected, with the
diagonal. The histograms above size the error; this asks whether the parameter is
constrained at all. Inclination correlates at {SCI.get("ci_corr", float("nan")):.2f}.
<em>Our encoding, not a field convention.</em></div></div>
</section>"""
    CENSUS_PANEL = f"""<div class="panel">{img("gb_hi_f_census", "high-frequency recovery census")}
<div class="caption">{CENSUS_TXT} Left is the raw census: every catalogue source above the
cut, green where the maximum-likelihood walker holds a leaf on it. Middle is the health
test &mdash; recovery must be monotonic in signal-to-noise, and it is, which says adding is
signal-ordered rather than arbitrary. Right is the ceiling: bars to the right of the dashed
line are cells holding more detectable sources than the cap allows.</div></div>"""
    F10_BLEND_NOTE = ("<br><em>Two blending modes sit at the two ends: below "
                      "the tolerance, several templates share one injection; "
                      "far above it, one template can still straddle a pair "
                      "of injections that the catalogue resolves.</em>")
    NAV_PARAMS = '<a href="#params">parameters</a>'
    OPEN_ITEMS = f"""<strong>Open items.</strong> Purity is {pct(SCI["purity"]) if SCI else "&mdash;"}
against a {pct(SCI["chance"], 1) if SCI else "2.2%"} chance rate, so the matches are
real, but {SCI.get("n_unmatched", 0)} model sources have no detectable counterpart and
{SCI.get("blend_b", 0)} of those sit on an injection another source already claims.
Neither arm has converged."""
else:
    KPI_MATCH = REC_MATCH_PANELS = PARAMS_SECTION = CENSUS_PANEL = ""
    F10_BLEND_NOTE = NAV_PARAMS = ""
    OPEN_ITEMS = (
        "<strong>Open items.</strong> Quantitative match-vs-catalogue "
        "statistics are intentionally absent from this page: the physical "
        "phase-maximised overlap match is computed offline, and the page's "
        "own 2-bin frequency proxy is not quoted as a number. The catalogue "
        "truths appear in the visual overlays only. The run has not "
        "converged.")

html = f"""<title>LISA Global Fit {RUN_LABEL}</title>
<style>
:root {{
  --bg:#0A0E14; --panel:#10161F; --line:#223041; --fg:#B8C6D4; --dim:#67788A;
  --cyan:#4FD8EB; --amber:#F5A623; --green:#58C48A; --red:#E5484D; --violet:#9B7BFF;
  /* Catalogue-truth marks only. BRIGHT, not deep (2026-08-19): the earlier
     #C41220 was chosen to keep 30k crosses from reading as a pink haze, but
     on the #0A0E14 panel it went muddy and the crosses were unreadable
     against the green recovery circles. Legibility of the overlay wins --
     the haze worry is handled by the per-cross alpha instead. */
  --truthred:#FF2E3E;
}}
:root[data-theme="light"] {{
  --bg:#EEF1F5; --panel:#FFFFFF; --line:#D4DBE3; --fg:#25313D; --dim:#5D6B7A;
  --truthred:#E00016;
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
button.armed.truth {{ border-color:var(--dim); color:var(--dim); }}
ul {{ color:var(--dim); font-size:13px; }}
</style>
<header>
  <h1>LISA Global Fit &middot; {RUN_LABEL} Status</h1>
  <span class="stamp">{os.path.basename(RUN_DIR)} &middot; {datetime.now():%Y-%m-%d}</span>
  <span>{chips}</span>
</header>
<nav>
  <a href="#status">status</a><a href="#resid">residual</a>
  <a href="#recovery">recovery</a><a href="#population">population</a><a href="#vgb">vgb zoom</a>
  {NAV_PARAMS}<a href="#search">search &amp; cap cells</a>
  <a href="#fstat">f-stat</a><a href="#noise">noise</a>
  <a href="#vgb">verification binaries</a><a href="#detect">detectability</a>
  <a href="#appendix">appendix</a>
</nav>
<main>

<section id="status"><h2>Status</h2>
<div class="kpi">
  <div><b>{SCI.get("ngbit", 0)}</b><span>GB search iterations</span></div>
  <div><b>{SCI.get("n_all", int(gb_counts[-1].max())):,}</b><span>model sources</span></div>
{KPI_MATCH}
  <div><b>{VGB_N_DET}</b><span>verification binaries above SNR 7</span></div>
</div>
<p style="font-size:13px">{RUN_HEALTH}</p>
<p style="font-size:13px"><strong>The denominator, stated once.</strong> Every
recovery number on this page is against <strong>{SCI.get("ndet", 812)} galactic
binaries</strong> &mdash; those in the injected catalogue with optimal
signal-to-noise above 7 over 3&ndash;21.94 mHz, evaluated under this run&rsquo;s own
fitted noise. A model source counts as a recovery when it lies within
{TOL_BINS:.0f} frequency bins ({TOL_BINS / SCI_TOBS * 1e6:.3f} &micro;Hz) of one,
one-to-one. Those windows cover {pct(SCI["chance"], 1) if SCI else "2.2%"} of the
band, so that is the rate at which an arbitrary source would match by accident.</p>
{ARM_TABLE}
</section>

<section id="resid"><h2>Residual Spectrum</h2>
<div class="panel">{img("f1_resid", "residual spectrum")}
<div class="caption">{cap_f1}<br>{cap_f1b}</div></div>
<div class="panel">{img("dtr_fd", "data / template / residual, frequency domain")}
<div class="caption">{dtr_fd_cap}</div></div>
<div class="panel">{img("dtr_wdm", "data / template / residual, time-frequency")}
<div class="caption">{dtr_wdm_cap}</div></div>
<div class="caption">{dtr_note}</div>
<div class="caption">Data are the mojito Level-1 products the run itself loaded,
re-transformed with the run&rsquo;s own window; templates are the last stored
cold-chain coordinates of the highest-likelihood walker through the run&rsquo;s own
transform and waveform generator. The noise branches shape the sensitivity the
likelihood weights by and are never subtracted, so the unresolved galaxy stays in
the residual by construction.</div>
</section>

<section id="recovery"><h2>Recovery</h2>
{REC_MATCH_PANELS}
<div class="panel">{img("f5_counts", "source counts vs frequency")}
<div class="caption">{cap_f5}</div></div>
</section>

<section id="population"><h2>Recovered Population</h2>
<div class="panel">{img("f4_pop", "amplitude vs frequency, and the three-way recovery split")}
<div class="caption">{cap_f4}</div></div>
<div class="panel">{img("f8_sky", "sky distribution")}
<div class="caption">{cap_f8}</div></div>
<div class="panel">{img("f10_nn", "nearest-neighbour separation")}
<div class="caption">{cap_f10}{F10_BLEND_NOTE}</div></div>

<div class="caption" style="margin-top:18px"><strong>Zoom in.</strong> The static panels above are the whole band at once; these two are pannable and zoomable, which is the only way to read an individual galactic binary against its injected counterpart.</div>
<div class="panel">
<div class="btnrow">
  <button id="btn_all">full band</button>
  <button id="btn_top3">highest-frequency sources</button>
  <button id="btn_truth">show catalogue</button>
  <button id="btn_reset">reset zoom</button>
  <span class="caption" style="align-self:center">drag to pan &middot; wheel to zoom</span>
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
<div class="caption">Zoomable version of the amplitude-frequency plane: green =
model sources, red crosses = the injected catalogue. Drag to pan and use the
wheel to zoom, or set the view numerically with the centre and width/height
controls above &mdash; those hold the window size fixed and slide it across
the band, which is the steadier way to walk through frequency.</div>
</div>

<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="gb1_sel"></select></label>
  <span class="caption" style="align-self:center">posteriors of the highest-frequency
  recovered galactic binaries</span>
</div>
<canvas id="gb1" style="height:330px"></canvas>
<div class="caption" id="gb1_cap"></div>
<div class="caption">Sources are clustered out of the last stored iteration by
frequency ({CLUSTER_BINS:.0f} bins) and counted only if they appear in at least three
of the {nwalk} cold walkers; {gb1_meta.get("n_solid", 0)} of
{gb1_meta.get("n_clusters", 0)} clusters clear that bar. Catalogue values are shown
where a source lies within {MATCH_BINS:.0f} bins.</div>
</div>

</section>

{PARAMS_SECTION}

<section id="search"><h2>Search &amp; Cap Cells</h2>
<div class="panel">{img("gb_birth_fate", "birth-fate breakdown")}
<div class="caption">Where every trans-dimensional birth proposal ends up. The fates are
disjoint and sum to the proposed count. <span style="color:var(--amber)">Amber</span> is
gated before scoring &mdash; the cell already holds its allowance, or the draw is out of
band or out of prior &mdash; cheap rejections that never touch a likelihood kernel.
<span style="color:var(--red)">Red</span> is scored and then dropped at the optimal-SNR
clamp. Grey is scored, offered to Metropolis&ndash;Hastings and rejected.
<span style="color:var(--green)">Green</span> is accepted, i.e. a new source. Left is
absolute counts, right the same data as percentages so the trend stays readable as the
model fills. {GB_FATE_TXT}</div></div>
<div class="panel">{img("gb_leaves")}
<div class="caption">Left: galactic-binary leaf count per cold walker across the stored
iterations, against the detectable-injection target. Right: the enforced per-cell leaf
cap over time. Rows marked in red have had their births shut off by the barren-band rule
&mdash; deaths and in-model moves continue there.</div></div>
<div class="panel">{img("gb_cap_cells", "cap-cell occupancy")}
<div class="caption">{CAP_TXT} Left is the direct test of whether the cap is being
respected: bars at or above the cap are amber, and a bar past it would mean sources are
stacking. Middle is the race that matters &mdash; occupied cells must stay ahead of cells
at their cap, or the model is queuing against the ceiling rather than filling. Right
explains why the occupied fraction looks small: the cells tile the whole band uniformly
while the sources are concentrated in the galaxy, so most cells are empty because there is
nothing in them yet.</div></div>
<div class="panel">{img("gb_cap_divisor", "cap-divisor study")}
<div class="caption">What the cap grid can represent, independent of how far this run has
got: summing the detectable sources a cell cannot admit gives the ceiling the sampler can
never beat. A finer grid beats simply raising the cap, because a higher cap permits several
sources inside one cell, which is the stacking the rule exists to stop.</div></div>
{CENSUS_PANEL}
</section>

<section id="fstat"><h2>F-statistic Fit</h2>
<div class="panel">{img("fstat_comb")}
<div class="caption">Comb scan of the maximised F-statistic across the band
(epoch {fstat_meta.get("epoch", "?")}), fitted against the live residual. This is what the
birth proposal draws from.</div></div>
<div class="panel">{img("fstat_peaks")}
<div class="caption">The selected peaks are the birth-proposal anchors. Left is where they
sit in the plane; right is their density against frequency &mdash; the shape that decides
where the search spends its proposals.</div></div>
</section>

<section id="noise"><h2>Noise Model</h2>
<div class="panel">{img("f11_psd", "instrument noise posteriors")}
<div class="caption">The two instrument-noise parameters against their injected
values. Medians sit {NOISE_TXT} from injection. These are the only noise parameters
with a truth to compare against.</div></div>
<div class="panel">{img("f11_fg", "foreground evolution")}
<div class="caption">The fitted noise-plus-foreground curve at every stored
iteration, light to dark with time, over the instrument-only curve. The galactic
shoulder should walk down as resolved sources leave the residual.</div></div>
<div class="panel">{img("psd_curves", "sensitivity curves")}
<div class="caption">The same model as the sky-averaged sensitivity the mission documents
quote: instrument only, instrument plus the fitted foreground, and the injected instrument
curve. This is the only panel carrying the injected curve.</div></div>
<div class="panel">{img("psd_evolution", "sensitivity evolution")}
<div class="caption">The decline watch on the same axes, one curve per stored iteration,
light to dark with time.</div></div>
<div class="panel">{img("psd_trace")}
<div class="caption">Instrument-parameter traces per cold walker; dotted red is the
injected value.</div></div>
<div class="panel">{img("psd_hist")}</div>
<div class="panel">{img("gal_trace")}
<div class="caption">The five foreground parameters. There is no truth line: the injection
is a source population, not a hyperbolic-tangent model, so these are checked through the
curve above and through the residual, never against a number.</div></div>
<div class="panel">{img("gal_hist")}</div>
</section>

<section id="vgb"><h2>Verification Binaries</h2>
<div class="panel">{img("f9_vgb", "detectable verification binaries")}
<div class="caption">The {VGB_N_DET} of 55 catalogue verification binaries that clear
SNR 7 at three months, with their distance posteriors against the catalogue value.
The other {55 - VGB_N_DET} are prior-dominated &mdash; the median verification-binary
SNR is {VGB_SNR_MED:.1f}.</div></div>
<div class="panel">{img("f9_vgb_snr", "verification binary SNRs")}
<div class="caption">Why: optimal SNR of all 55 against the fitted noise. Three
months is simply not long enough for most of this set, and their flat posteriors are
an absence of signal, not a failure of the fit.</div></div>
<div class="panel">{img("vgb_dist")}
<div class="caption">Distance posteriors for all 55 leaves against catalogue truth, median
and 1&sigma;. The 44 prior-dominated ones are the flat error bars.</div></div>
<div class="panel">
<div class="btnrow">
  <button id="vgbpost_reset">reset zoom</button>
  <span class="caption" style="align-self:center">distance&ndash;f0 posterior cloud:
  EVERY sample (last stored iterations &times; {nwalk} walkers per leaf), catalogue truth
  in red &middot; drag = pan &middot; wheel/pinch = zoom</span>
</div>
<div class="btnrow viewctl">
  <button id="vgbpost_pick" title="arm, then click the plot to set the view center">set center by click</button>
  <label>cx <input id="vgbpost_cx" type="text"></label>
  <label>cy <input id="vgbpost_cy" type="text"></label>
  <label>width <input id="vgbpost_wsl" type="range" min="0" max="1000" step="1"><input id="vgbpost_w" type="text"></label>
  <label>height <input id="vgbpost_hsl" type="range" min="0" max="1000" step="1"><input id="vgbpost_h" type="text"></label>
</div>
<canvas id="vgbpost" style="height:340px"></canvas>
<div class="caption">The zoom is the point: at full extent the 55 posteriors overlap into a
band, and only zoomed in can an individual leaf be read against its red truth mark. The
JS for this canvas survived the redesign but its markup did not, so the handler was
dereferencing a null and killing every later script block on the page.</div>
</div>
<div class="panel">{img("vgb_snr")}
<div class="caption">Optimal signal-to-noise per stored iteration, light to dark with time.
These should <em>rise</em> as the galactic foreground is fitted down &mdash; the source-side
twin of the sensitivity decline watch.</div></div>
<div class="panel">{img("vgb_traces")}
<div class="caption">Distance traces for the three loudest verification binaries, all cold
walkers, against catalogue truth.</div></div>
<div class="panel">{img("vgb_hists")}
<div class="caption">The four remaining sampled parameters pooled over all 55 leaves, so
the truth is a distribution rather than a line. The frequency-derivative ratio is the
exception: every catalogue value is identically zero.</div></div>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="vgb1_sel"></select></label>
  <span class="caption" style="align-self:center">full posterior, one verification
  binary at a time &mdash; all 55, detectable first</span>
</div>
<img id="vgb1_img" alt="verification binary corner posterior">
<div class="caption" id="vgb1_cap"></div>
<div class="caption">Existence proof that the machinery produces real posteriors:
five sampled parameters over the last {CORNER_ITS} stored iterations &times; {nwalk}
cold walkers, with the catalogue value in cyan. Axis ranges are widened to contain
the truth, so a truth line outside the posterior stays visible.</div>
</div>
</section>

<section id="detect"><h2>How Many Are Detectable At All</h2>
<div class="panel">
<div class="caption" style="margin:0 0 10px 0">Optimal SNR of the whole injected
catalogue at this observation time, under two noise models: the run&rsquo;s own fitted
instrument and foreground, and the injected instrument noise with the legacy fitted
foreground. Full band, so these are larger than the 3&ndash;21.94 mHz denominator
above.</div>
<table style="border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums">
<tr style="border-bottom:1px solid var(--line)">
  <th style="text-align:left;padding:4px 18px 4px 0">SNR &gt;</th>
  <th style="text-align:right;padding:4px 18px 4px 0">fitted noise</th>
  <th style="text-align:right;padding:4px 0">injected + legacy foreground</th></tr>
<tr><td style="padding:3px 18px 3px 0">5</td><td style="text-align:right;padding:3px 18px 3px 0">1,661</td><td style="text-align:right">1,749</td></tr>
<tr><td style="padding:3px 18px 3px 0"><strong>7</strong></td><td style="text-align:right;padding:3px 18px 3px 0"><strong>1,001</strong></td><td style="text-align:right"><strong>1,103</strong></td></tr>
<tr><td style="padding:3px 18px 3px 0">10</td><td style="text-align:right;padding:3px 18px 3px 0">560</td><td style="text-align:right">647</td></tr>
<tr><td style="padding:3px 18px 3px 0">15</td><td style="text-align:right;padding:3px 18px 3px 0">259</td><td style="text-align:right">297</td></tr>
</table>
<div class="caption" style="margin-top:12px">Detectability dies below about 1 mHz,
where the foreground swamps everything: eight detectable sources across the whole
0.1&ndash;1 mHz decade, out of roughly 13 million catalogue entries there. The two
models disagree by 10% overall and, more usefully, in opposite directions either side
of the galactic peak &mdash; treat the second column as a reference point, not truth.
<br><br>
Detectability is also a moving target: the same calculation gives
<strong>{SCI.get("ndet", 812)}</strong> over 3&ndash;21.94 mHz at this run&rsquo;s
late-iteration noise against 694 at iteration 15, because the foreground estimate
dropped as sources left the residual. That is exactly why the denominator on this
page is frozen at one iteration and stated.</div>
</div>
</section>

<section id="appendix"><h2>Appendix</h2>
<details><summary style="cursor:pointer;color:var(--dim);font-size:13px">
method, sampler health, interactive views and run mechanics</summary>

<div class="panel">{img("ll")}
<div class="caption">Cold-chain total log-likelihood across the {nwalk} walkers, and
the max-minus-min spread. At equilibrium the spread sits at a few units.</div></div>

<div class="caption"><strong>How the numbers are produced.</strong> Optimal SNRs and
template overlaps use the injected catalogue&rsquo;s own parameters through the
run&rsquo;s catalogue-to-sampling map, the same waveform generator the run samples
with, and a noise-weighted inner product against the run&rsquo;s own fitted
instrument and foreground. Catalogue detectability is computed by bounding
signal-to-noise per unit amplitude over orientation, which can only over-include, then
evaluating exact SNRs for the survivors; an audit of 400 rejected sources weighted
toward the cut found a loudest SNR of 4.36 and none above 7. Overlaps and parameter
comparisons on this page are recomputed at the last stored iteration, not carried over
from earlier snapshots.
<br><br>
{OPEN_ITEMS}</div>

<div class="caption" style="margin-top:22px"><strong>Run mechanics.</strong> Engineering
instrumentation &mdash; where the wall time goes, what the devices are holding, and whether
the tempering ladder is exchanging. None of it is a science result; all of it is the first
thing to look at when a run stalls.</div>
{rj_kpis}
<div class="panel">{img("swaps")}
<div class="caption">Tempering swap acceptance per rung for the two noise branches, at each
branch&rsquo;s last active iteration &mdash; a stored iteration can record zero proposals
for one branch at a stage handoff.</div></div>
<div class="panel">{img("rj_breakdown", "trans-dimensional move breakdown")}
<div class="caption">Wall-time breakdown of the last complete record of each
trans-dimensional move, leaf spans only; the enclosing phase marks are quoted in each
title instead of drawn, since they merely reprint the total.{rj_break_txt}</div></div>
<div class="panel">{img("timing_moves", "per-move throughput")}
<div class="caption">Proposal throughput and wall time per propose, per move, against
elapsed run time.</div></div>
<div class="panel">{img("mem_telemetry", "device memory")}
<div class="caption">Device-wide memory from the in-run telemetry, with breaks across
restart gaps so an attempt boundary does not draw a false ramp.</div></div>
<div class="panel">{img("gpu_util", "gpu telemetry")}
<div class="caption">Utilisation and memory sampled by nvidia-smi; dotted traces are
earlier jobs.</div></div>

<div class="caption"><strong>Not reproduced in this snapshot:</strong></div>
<ul>{missing_html}</ul>
</details>
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
    // RESIZE ONLY WHEN THE SIZE ACTUALLY CHANGES (2026-08-19). Assigning to
    // cv.width reallocates the backing store and resets all context state.
    // Doing it every frame -- while panning, at pointer-event rate -- was
    // the biggest cost in these canvases: a multi-megabyte buffer thrown
    // away and rebuilt per pointermove.
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (cv.width !== bw || cv.height !== bh) {{ cv.width = bw; cv.height = bh; }}
    const g = cv.getContext("2d");
    // setTransform, not scale: scale() would compound now that the buffer
    // (and with it the identity transform) survives between frames.
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
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
  // See the explorer canvas below for why pan/zoom redraws are coalesced
  // into animation frames rather than run per pointer event.
  let raf = 0;
  const redraw = () => {{
    if (raf) return;
    raf = requestAnimationFrame(() => {{ raf = 0; draw(); }});
  }};
  let drag = null;
  cv.addEventListener("pointerdown", e => {{ drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - 66) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - 38) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy; drag = [e.clientX, e.clientY]; redraw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  // ZOOM ABOUT THE CURSOR, NOT THE VIEW CENTRE (2026-08-19). Centre-anchored
  // zoom is what made this canvas hard to drive: the source you were aiming
  // at slid out of frame as you zoomed, so reaching one binary meant
  // alternating zoom and pan several times. Anchoring on the pointer holds
  // whatever is under the cursor still, the way every map-style UI behaves.
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const s = e.deltaY > 0 ? 1.15 : 0.87;
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const cl = v => Math.max(0, Math.min(1, v));
    // fraction across the plot box; clamped so a cursor sitting in the axis
    // margins anchors at the edge instead of flinging the view sideways
    const fx = cl((e.clientX - r.left - ml) / (w - ml - mr));
    const fy = cl((h - mb - (e.clientY - r.top)) / (h - mb - mt));
    const ax = X0 + fx * (X1 - X0), ay = Y0 + fy * (Y1 - Y0);
    X0 = ax + (X0 - ax) * s; X1 = ax + (X1 - ax) * s;
    Y0 = ay + (Y0 - ay) * s; Y1 = ay + (Y1 - ay) * s; redraw();
  }}, {{ passive: false }});
  document.getElementById("vgbpost_reset").onclick = () => {{ full(); draw(); }};
  syncCtl = viewCtl("vgbpost", cv, {{
    get: () => [X0, X1, Y0, Y1],
    set: (a, b, c, d) => {{ X0 = a; X1 = b; Y0 = c; Y1 = d; draw(); }},
    fullW: FW, fullH: FH,
  }}).sync;
  new ResizeObserver(redraw).observe(cv);
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
  // DEFAULT ON (2026-08-16). The catalogue overlay is the point of this
  // panel -- the recovered cloud alone cannot show completeness or the
  // faint tail -- and defaulting it to hidden meant it went unnoticed.
  let showT = TRUTH.length > 0;
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
    // RESIZE ONLY WHEN THE SIZE ACTUALLY CHANGES (2026-08-19). Assigning to
    // cv.width reallocates the backing store and resets all context state.
    // Doing it every frame -- while panning, at pointer-event rate -- was
    // the biggest cost in these canvases: a multi-megabyte buffer thrown
    // away and rebuilt per pointermove.
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (cv.width !== bw || cv.height !== bh) {{ cv.width = bw; cv.height = bh; }}
    const g = cv.getContext("2d");
    // setTransform, not scale: scale() would compound now that the buffer
    // (and with it the identity transform) survives between frames.
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
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
      // Drawn as CROSSES, not 1.4px dots (user request 2026-08-16): the
      // dots were the same visual weight as a rendering artefact, so the
      // injected population read as background noise rather than as the
      // thing the recovered cloud is being judged against. An open glyph
      // also stays legible UNDER the filled recovery circles, which a
      // solid marker of this size would not.
      // --truthred, NOT --dim (2026-08-19). The colour was defined for
      // exactly these marks and never wired up, so the crosses rendered
      // grey and the caption's "Catalogue truths (red)" was a lie.
      g.strokeStyle = C("--truthred"); g.globalAlpha = 0.9;
      g.lineWidth = 1.5; g.lineCap = "round";
      const r = 3.4;
      g.beginPath();
      for (const p of TRUTH) {{
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x - r, y - r); g.lineTo(x + r, y + r);
        g.moveTo(x - r, y + r); g.lineTo(x + r, y - r);
      }}
      g.stroke();
    }}
    // ONE PATH, ONE FILL (2026-08-19). This used to open a path and issue a
    // separate fill() per point -- 5k+ draw calls per frame. Batching every
    // dot into a single path costs one fill. The moveTo before each arc is
    // required: without it consecutive arcs are joined by a straight line.
    g.globalAlpha = 0.55;
    g.fillStyle = hasGB ? C("--green") : C("--violet");
    g.beginPath();
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.moveTo(x + 2.2, y); g.arc(x, y, 2.2, 0, 6.29);
    }}
    g.fill();
    g.globalAlpha = 1;
    syncCtl();
  }}
  // COALESCE REDRAWS TO ANIMATION FRAMES (2026-08-19). pointermove fires at
  // up to the pointer's sample rate (120 Hz+ on a trackpad) and each event
  // used to force a full synchronous redraw of 30k truth crosses plus the
  // recovery cloud. Collapsing every burst of events into at most one draw
  // per frame is what makes dragging track the cursor instead of lagging
  // behind it. Slider/typed input still calls draw() directly -- viewCtl's
  // `busy` guard depends on sync() running inside its own call.
  let raf = 0;
  const redraw = () => {{
    if (raf) return;
    raf = requestAnimationFrame(() => {{ raf = 0; draw(); }});
  }};
  // pan/zoom
  let drag = null;
  cv.addEventListener("pointerdown", e => {{ drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - 66) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - 38) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy; drag = [e.clientX, e.clientY]; redraw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  // ZOOM ABOUT THE CURSOR, NOT THE VIEW CENTRE (2026-08-19). Centre-anchored
  // zoom is what made this canvas hard to drive: the source you were aiming
  // at slid out of frame as you zoomed, so reaching one binary meant
  // alternating zoom and pan several times. Anchoring on the pointer holds
  // whatever is under the cursor still, the way every map-style UI behaves.
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const s = e.deltaY > 0 ? 1.15 : 0.87;
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const cl = v => Math.max(0, Math.min(1, v));
    // fraction across the plot box; clamped so a cursor sitting in the axis
    // margins anchors at the edge instead of flinging the view sideways
    const fx = cl((e.clientX - r.left - ml) / (w - ml - mr));
    const fy = cl((h - mb - (e.clientY - r.top)) / (h - mb - mt));
    const ax = X0 + fx * (X1 - X0), ay = Y0 + fy * (Y1 - Y0);
    X0 = ax + (X0 - ax) * s; X1 = ax + (X1 - ax) * s;
    Y0 = ay + (Y0 - ay) * s; Y1 = ay + (Y1 - ay) * s; redraw();
  }}, {{ passive: false }});
  document.getElementById("btn_all").onclick = () => {{ full(); draw(); }};
  document.getElementById("btn_reset").onclick = () => {{ full(); draw(); }};
  const bt = document.getElementById("btn_truth");
  if (!TRUTH.length) bt.disabled = true;
  // reflect the ON default in the control the moment the page loads
  bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
  bt.textContent = showT ? "hide catalogue truths" : "show catalogue truths";
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
  new ResizeObserver(redraw).observe(cv);
  draw();
}})();
</script>
"""
open(OUT, "w").write(html)
print(f"wrote {OUT}: {len(html)//1024} KB ({len(html)/1024**2:.2f} MB), "
      f"{len(IMGS)} plots + {len(VGB_CORNER['src'])} VGB corners, "
      f"missing={len(MISSING)}")
