#!/usr/bin/env python
"""GF run status page: <run_dir> -> one self-contained HTML file.

Reusable: point RUN_DIR at any unzipped gf_prod_* snapshot and rerun; the
artifact redeploys to the same URL. Sections degrade to labeled
placeholders when a snapshot lacks their inputs.

The page is a STATUS OBJECT for collaborators, not an engineering worklog.
Its spine is one frozen denominator -- the catalogue galactic binaries
detectable (optimal SNR > 7) over the analysed GB band under this run's own
fitted noise -- against which completeness, purity and per-source recovery are
all quoted. The band is not hardcoded here: it is read from the ``band`` field
the truth npz stamps (``build_truth.py``), so widening the truth set to the
full 0.5556-21.94 mHz GB band re-labels every caption automatically. Run-mechanics forensics live in the collapsed appendix, never in the
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
Per-walker traces (lnL, GB leaf count, both pages) are the one exception:
there a hue names a cold walker index (WALKER_COLORS, never red) and the
thick light-grey line is the walker mean.
"""
import base64, glob, io, json, os, re, sys
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# KEPT HERE, NOT IMPORTED FROM ._logs -- and that is load-bearing.
#
# The obvious tidy-up is `from lisatools.globalfit.monitor._logs import
# discover_run_logs`. It was tried (2026-09-26) and it CHANGED THE PAGE.
# That import pulls in lisatools.globalfit, whose chain reaches eryn,
# which calls plt.style.use(["science"]) as an import side effect. In the
# original file that happens at the `from ...erebor.noise import
# psd_truth_levels` line ~800, i.e. AFTER this file's own
# plt.rcParams.update() block -- so the science style wins. Importing the
# package up here moves it BEFORE that block, so the local style wins
# instead, and the "instrument noise posteriors" panel renders
# differently (measured: 25722 vs 27586 base64 chars, 25806 pixels
# changed across the whole plot area).
#
# So ._logs holds an identical copy for importers (the
# scripts/diagnostics/gf_monitor_gen.py shim has no body of its own), and
# MonitorLogsCopyIsInSyncTest compares the two sources so they cannot
# drift. A duplicate that a test pins beats an import that silently
# restyles the page.
def discover_run_logs(run_dir):
    """Every rank's run log under ``run_dir``: the head's ``globalfit_run.log``
    first, then ``globalfit_run.rank<k>.log`` files sorted by rank number.

    Each rank writes its own log file under the walker-block layout
    (``run.py::_rank_log_filenames``); concatenating them (head first) is
    what lets the regex scans below (e.g. ``RJ_SPLIT_RE``) see every rank's
    ``[GB_ACCEPT rj-split]`` lines, not just the head's (Plan 5 Task 4).

    The walk is RECURSIVE and deterministic (directories and file names
    sorted), identical to ``gf_run_log_digest.py``'s copy of this helper. A
    second file for a rank already seen (the same run unpacked twice under
    ``run_dir``) is NOT silently dropped -- the first one found wins and the
    duplicate is named on stderr.
    """
    found = {}
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if fn == "globalfit_run.log":
                key = -1
            else:
                m = re.match(r"^globalfit_run\.rank(\d+)\.log$", fn)
                if m is None:
                    continue
                key = int(m.group(1))
            path = os.path.join(root, fn)
            if key in found:
                label = "head" if key < 0 else f"rank {key}"
                print(
                    f"# WARNING: duplicate {label} run log {path}; keeping {found[key]}",
                    file=sys.stderr,
                )
                continue
            found[key] = path

    # ---- slurm stdout: the SUPERSET, and the only source of some lines ---
    #
    # ⚠ WITHOUT THIS THE PAGE IS BLIND ON THE CLUSTER. Measured on the 6mo
    # v9 tars (2026-09-28): ``slurm_stdout_<job>.log`` is a verbatim
    # superset of ``globalfit_run.log`` plus the per-rank logs (3000/3000
    # and 2960/3000 sampled lines found in it), AND it alone carries
    # ``[GF_TIMING]``, ``[V9-SEED]``, the ``[r4/saver]`` lines and the
    # stage table -- those never reach the run logs at all. So:
    #
    #  * ``GFT_RE`` could never match on a cluster-built page, which is
    #    why the "search efficiency" panel has been dead there, and
    #  * a page built from a snapshot whose head log was shipped as the
    #    filtered + tail pair (``globalfit_run_filtered.log`` /
    #    ``_tail.log``, which the names above do not match) saw NO head
    #    text at all, and a short tar gave it nothing whatsoever.
    #
    # Appended LAST so the head log keeps priority for everything it does
    # carry, and only the NEWEST stdout is taken: older ones are dead jobs
    # whose lines would be replayed as if current.
    _stdouts = []
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if re.match(r"^slurm_stdout_\d+\.log$", fn):
                p = os.path.join(root, fn)
                try:
                    _stdouts.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    out = [found[k] for k in sorted(found)]
    if _stdouts:
        out.append(max(_stdouts)[1])
    elif not out:
        # Neither kind present: say so rather than silently rendering a
        # page with every log-derived panel empty.
        print(f"# WARNING: no run log and no slurm_stdout_*.log under "
              f"{run_dir}; every log-derived panel will be empty",
              file=sys.stderr)
    return out


if __name__ != "__main__":
    # Standalone report generator, not a library; guard the rest of the file
    # (which reads sys.argv / real snapshot files unconditionally) so tests
    # can import discover_run_logs() above without running it.
    raise SystemExit(0)

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "prod3mo/gf_prod_3mo"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gf_monitor.html"

# ---- mission-control plot style -------------------------------------------
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
CYAN, AMBER, GREEN, RED, VIOLET = "#4FD8EB", "#F5A623", "#58C48A", "#E5484D", "#9B7BFF"
# text.usetex is pinned OFF rather than left to inherit. Nothing on this
# page needs a real LaTeX installation -- the ~19 math labels
# (r"$\Delta f_0$", r"$\ln(A_{\rm rec}/A_{\rm cat})$", "m/s$^2$", ...) are
# all plain mathtext, which matplotlib renders internally. Left unset the
# value comes from whatever matplotlibrc the machine happens to carry, so a
# host with usetex on would shell out to LaTeX for every one of the 30
# figures: far slower, and a hard failure on any box without a TeX
# distribution -- which is most cluster nodes and every fresh container.
# Set GF_MONITOR_USETEX=1 to opt back in (real LaTeX must be installed).
_USETEX = os.environ.get("GF_MONITOR_USETEX", "0") not in ("0", "", "false", "no")
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})

# ---- one colour per cold walker (user request 2026-10-12) ------------------
# "Adjust the logL and leaf plots over iteration from min-max-mean to all 4
# walkers + mean". Every each-walker-against-iteration trace -- this page's
# total log-likelihood and GB leaf count, and the at-a-glance block's two
# history figures (_short.fig_history, which compiles these three constants
# and the two functions below by name) -- draws cold walker k in
# WALKER_COLORS[k] and the walker mean as one thick WALKER_MEAN_COLOR line,
# so a walker keeps its colour on every panel, inset and page.
# The hues are the data-viz reference palette's dark steps (green, magenta,
# blue, yellow, violet), in an order validated on this panel colour
# (#10161F): adjacent pairs clear CVD dE 13.0 and normal-vision dE 26.5; the
# first four -- today's four walkers -- clear normal-vision dE 19.3 for EVERY
# pair (CVD 6.9 at worst, yellow / green, where the legend label carries it);
# all five clear 3:1 contrast. Its orange and red are left out (normal-vision
# dE < 6 from RED, this page's detectable-target colour); its aqua too, at
# dE 11.9 from green. Magenta is kept although it sits at dE 7.3 from RED: on
# these figures red is only the dotted, labelled target line, and telling the
# walkers apart is the point. A walker past the fifth folds into one muted
# line labelled once, never a cycled hue, which would hand it an earlier
# walker's identity.
WALKER_COLORS = ("#008300", "#d55181", "#3987e5", "#c98500", "#9085e9")
WALKER_OTHER_COLOR = "#67788A"
WALKER_MEAN_COLOR = "#B8C6D4"


def walker_color(w):
    """Line colour of cold walker ``w`` on every per-walker trace."""
    return WALKER_COLORS[w] if 0 <= w < len(WALKER_COLORS) else WALKER_OTHER_COLOR


def plot_walker_lines(ax, x, y, *, label=True, lw=1.0, mean_lw=2.4, mean_label="mean"):
    """Each cold walker of ``y`` (rows, walkers) as its own thin line in
    :func:`walker_color`, and the mean over walkers as one thick line on top.

    The walker count is ``y.shape[1]``, never assumed. A NaN row (not written
    yet, or unreadable) is a gap in every line, the mean's included: the mean
    is the plain one, as the min / max it replaces were, so a row missing a
    walker is not averaged over whichever walkers happen to be there.
    ``label=False`` (the zoom panel, an inset) adds no legend entries, so a
    figure's legend names each walker once. Returns the mean.
    """
    import matplotlib.patheffects as _pe

    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"y must be (rows, walkers), got shape {y.shape}")
    nw, nc = y.shape[1], len(WALKER_COLORS)
    for w in range(nw):
        lab = None
        if label and w < nc:
            lab = f"walker {w}"
        elif label and w == nc:
            lab = f"walker {w}" if nw == nc + 1 else f"walkers {nc}-{nw - 1}"
        ax.plot(x, y[:, w], color=walker_color(w), lw=lw, alpha=0.9, label=lab, zorder=2)
    mean = y.mean(axis=1)
    # a ring in the panel colour keeps the mean legible where it crosses walkers
    ax.plot(x, mean, color=WALKER_MEAN_COLOR, lw=mean_lw, zorder=3,
            label=mean_label if label else None,
            path_effects=[_pe.Stroke(linewidth=mean_lw + 1.2, foreground=ax.get_facecolor()),
                          _pe.Normal()])
    return mean


IMGS, MISSING = {}, []

# MATCH-CRITERION CONTENT GATE. History: the 2026-08-19 ruling gated every
# match-derived number OFF because the page's only criterion then was the
# 2-df-bin f0 PROXY. Since the phase-maximised, noise-weighted overlap is
# computed for every proxy pair at the last stored row (see the waveform-
# level block below), the HEADLINE numbers -- matched count, completeness,
# purity, the F4 three-way split, completeness-vs-SNR, the zoomable-plot
# classes -- are the OVERLAP-refined match (>= GF_MONITOR_MATCH_MM, default
# 0.8). User request 2026-09-30 ("get back to including the phase maximized
# computations of match"): the gate is ON by default. What stays proxy, and
# says so in its caption: the per-iteration completeness/purity CURVES
# (F2) and the cross-arm table, which run on every stored row where a
# waveform per pair is not affordable. GF_MONITOR_MATCH_STATS=0 restores
# the match-free page.
SHOW_MATCH_STATS = os.environ.get("GF_MONITOR_MATCH_STATS", "1") != "0"
if not SHOW_MATCH_STATS:
    # SAY SO (2026-09-26). The gate removes NINE panels and the whole
    # Parameter Recovery section, and it used to do it in total silence --
    # they simply were not there, with no notice and no placeholder. A page
    # built without them therefore looks like a page whose run produced
    # nothing, and the only way to tell the two apart was to diff against an
    # older page, which is exactly how this was reported ("does not look
    # right ... make sure it did not drop anything").
    MISSING.append(
        "match-criterion panels are SUPPRESSED (GF_MONITOR_MATCH_STATS=0): "
        "no completeness/purity curve, no completeness-vs-SNR, no overlap "
        "CDF, no recovery split, no source-counts / amplitude / sky / "
        "nearest-neighbour population panels, and no Parameter Recovery "
        "section. Nothing is wrong with the run or the snapshot. Unset the "
        "knob (default ON since 2026-09-30) to include them.")

# Threshold for "matched" once the phase-maximised, noise-weighted overlap MM
# is computed for each 2-df pair (see below). Default 0.8; override via
# GF_MONITOR_MATCH_MM. Only affects F4's three-way split and the zoomable-plot
# marker classes -- the F2/F3/F6 panels intentionally still key off the 2-df
# proxy so per-iteration progress does not require re-running waveforms per
# stored row.
MATCH_MM_THRESH = float(os.environ.get("GF_MONITOR_MATCH_MM", "0.8"))


# ---- THE ONE MOJITO PATH ---------------------------------------------------
# USER RULING 2026-09-26: "the one path you need for the mojito data [is] the
# folder that contains catalogues/ and data/. Make that MOJITO_INFO_PATH."
#
# This page used to need TWO knobs for one directory, and they resolved
# differently, which is precisely how a render silently loses panels:
#
#   MOJITO_DATA_PATH   -> data/INSTRUMENT/L1/NOISE_*   (PSD truth levels)
#                         NO default, NO expanduser; unset = silent fallback
#                         to the analytic injection.
#   MOJITO_CAT /
#   MOJITO_CACHE_DIR   -> data/{GB,VGB,COMBINED}/L1/* + catalogues/wdwd_cat_*
#                         defaulted to the laptop's nested brickmarket path,
#                         so on a cluster it missed and the residual-spectrum
#                         and data/template/residual panels just vanished.
#
# Both want the SAME directory. One knob, one meaning, and it is the level
# holding BOTH subdirectories -- the level people actually get wrong, since
# the laptop cache nests two deeper than the cluster's
# (~/.mojito_cache/brickmarket/mojito_light_v1_0_0 vs /shared/data/
# mojito_cache). The legacy vars still work as fallbacks so existing
# runbooks keep rendering, but they are no longer the documented route.
_MOJITO_SUBDIRS = ("catalogues", "data")


def _mojito_info_ok(p):
    return p and all(
        os.path.isdir(os.path.join(p, d)) for d in _MOJITO_SUBDIRS)


def _resolve_mojito_info_path():
    """``MOJITO_INFO_PATH``, validated, or ``None``.

    Pointing this one level too high is the expected mistake, so a value
    that does not hold both subdirectories gets ONE rescue attempt down
    build_truth.py's nested ``brickmarket/mojito_light_v1_0_0`` layout
    before being reported. Never guesses further and never fails silently:
    a wrong path here costs whole panels, and a page that quietly drops
    them looks exactly like a page whose run produced nothing.
    """
    v = os.environ.get("MOJITO_INFO_PATH")
    if not v:
        return None
    v = os.path.expanduser(v)
    if _mojito_info_ok(v):
        return v
    nested = os.path.join(v, "brickmarket", "mojito_light_v1_0_0")
    if _mojito_info_ok(nested):
        MISSING.append(
            f"MOJITO_INFO_PATH={v!r} does not hold catalogues/ and data/, "
            f"but {nested!r} does -- using that. Point the variable there "
            "directly to silence this.")
        return nested
    _absent = [d for d in _MOJITO_SUBDIRS
               if not os.path.isdir(os.path.join(v, d))]
    MISSING.append(
        f"MOJITO_INFO_PATH={v!r} is missing {'/ '.join(_absent)}. It must be "
        "the directory that CONTAINS catalogues/ and data/ (cluster: "
        "/shared/data/mojito_cache; laptop: "
        "~/.mojito_cache/brickmarket/mojito_light_v1_0_0). Falling back to "
        "MOJITO_DATA_PATH / MOJITO_CAT.")
    return None


MOJITO_INFO_PATH = _resolve_mojito_info_path()


def fig_b64(fig, key, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    # 8-bit palette PNG (_png.py): about a third of the bytes for the same figure,
    # which is what keeps the page inside the 16 MB budget with the glance block on top
    IMGS[key] = base64.b64encode(_shrink_png(buf.getvalue())).decode()

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
# NEWEST mtime wins, not alphabetical (2026-08-22): a make_snapshots.sh
# tar refreshes the *_extract.h5 but cannot delete a previous format's
# full store left in the dir, and "testing.h5" sorts before
# "testing_extract.h5" -- the first tar-format page silently rendered a
# 384-iteration-stale store. Freshness is the only defensible tiebreak.
_h5s = sorted((fn for fn in os.listdir(RUN_DIR) if fn.endswith(".h5")),
              key=lambda fn: os.path.getmtime(os.path.join(RUN_DIR, fn)),
              reverse=True)
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

# TORN LAST ROW (2026-09-24). A save step writes the row's datasets one at a
# time and takes ~23 s on this store; the GB branch alone is 4 walkers x
# 15000 leaves x 9 columns. Read the file mid-save and the log_like row -- and
# so ``filled`` and often the ``iteration`` attr -- can already be there while
# ``inds/gb`` is still zeros. EVERY GB panel then renders an empty model: 0
# model sources, 0% completeness, no templates on the residual spectrum, and
# the leaf-count trace dropping off a cliff on its final point. Seen on the
# 2026-09-24 render at iteration attr 811, where row 810 was empty and 809
# held 3100 leaves. Step back to the last row that actually carries GB
# content and say so, rather than publishing a page that reads as if the
# search had lost every source.
_gb_inds_ds = g.get("inds/gb")


def _gb_alive(j):
    """Live GB leaves in stored row ``j``; 0 when the row cannot be read.

    An unreadable row is the same torn save caught one step earlier: the
    saver is rewriting that chunk under this reader ("filter returned
    failure during read", 6mo job 748 2026-10-09, which cost the whole
    page because this check ran outside any try).
    """
    try:
        return int(np.count_nonzero(_gb_inds_ds[j, 0, 0]))
    except OSError:
        return 0


if NIT > 0 and _gb_inds_ds is not None:
    _alive_last = _gb_alive(NIT - 1)
    if _alive_last == 0:
        _back = 0
        for _j in range(NIT - 2, max(NIT - 40, -1), -1):
            if _gb_alive(_j):
                _back = NIT - 1 - _j
                break
        if _back:
            MISSING.append(
                f"stored row {NIT - 1} carries NO GB leaves (or is unreadable "
                f"mid-save) while row {NIT - 1 - _back} carries "
                f"{_gb_alive(NIT - 1 - _back)}"
                f" -- a torn save, not a lost model. Rendering through row "
                f"{NIT - 1 - _back} ({_back} row(s) dropped); re-run once the "
                "save completes to pick them up.")
            NIT -= _back
        else:
            MISSING.append(
                f"stored row {NIT - 1} carries no GB leaves and neither does "
                "any of the 40 rows before it. Every GB panel below will read "
                "as an empty model -- treat that as a store problem, not a "
                "search result, until the leaf-count trace is checked.")
it = np.arange(NIT)
ll = ll_all[:NIT]                                   # (it, 24)
recipe = {}
rg = f.get("global_fit/recipe", f.get("recipe"))
if rg is not None:
    for k in rg:
        recipe[k] = (int(rg[k].attrs.get("order num", 0)), bool(rg[k].attrs.get("status", False)))

# ---- RECIPE STAGE BOUNDARIES ---------------------------------------------
# ``(iteration, stage name)`` for every step that has COMPLETED, in order.
# The stamp is ``completed_iteration``, written by
# hdfbackend.completed_recipe_step since 2026-09-24; every store older than
# that carries the status flag alone, so this list comes back empty and every
# panel below simply draws no boundary.
#
# WHY IT MATTERS NOW. The v9 restructure replaced one ``gb_search`` with
# ``gb_search_1/2/3``, and the three differ in phase maximization, the
# opt-SNR prior boundary (8 -> 5), the F-stat peak floor (8 -> 6.25) and
# whether the noise model is sampled at all. Every per-iteration trace on
# this page therefore spans several regimes. A leaf count that jumps at the
# boundary is a stage lowering its floors, not a discovery -- and with no
# boundary drawn there is nothing on the page that can tell those apart.
STAGE_BOUNDS = []
if rg is not None:
    for k in rg:
        _ci = rg[k].attrs.get("completed_iteration", None)
        if _ci is not None and bool(rg[k].attrs.get("status", False)):
            STAGE_BOUNDS.append((int(_ci), str(k)))
    STAGE_BOUNDS.sort()


def mark_stages(ax, *, label=True, max_it=None):
    """Draw the completed recipe-stage boundaries as vertical rules on ``ax``.

    No-op on a store with no stamps (everything written before 2026-09-24),
    which is what keeps this safe to call from every time-series panel.
    ``max_it`` drops boundaries past the plotted range.
    """
    if not STAGE_BOUNDS:
        return
    for _i, (_it, _name) in enumerate(STAGE_BOUNDS):
        if max_it is not None and _it > max_it:
            continue
        ax.axvline(_it, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=0)
        if label:
            ax.annotate(
                _name, xy=(_it, 1.0), xycoords=("data", "axes fraction"),
                xytext=(2, -2), textcoords="offset points",
                fontsize=6, color="0.35", rotation=90,
                ha="left", va="top", zorder=0,
            )
nwalk = ll.shape[1]

# Observation time from domain settings (needed by multiple plot sections)
SCI_TOBS = 7776000.0  # 3-month default
try:
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    SCI_TOBS = float(_a["0"]) * float(_a["1"]) * float(_a["2"])
except Exception:
    pass

# RUN IDENTITY (2026-08-15): one generator now serves several runs (3-mo
# production, 23-mo scaling). Derive the label from the store rather than
# hard-coding it, so a 23-mo page can never be mislabelled as the 3-mo one.
# 2026-10-07: the name ladder lives in ``_identity.run_identity`` (shared
# with the lean generator), every historical branch verbatim plus ONE
# generic ``<N>mo_v<K>`` / ``<N>yr_v<K>`` rule -- so the 9mo v9 store gets
# its own banner and arm cache instead of the 3-Month / v2 fall-through that
# clobbered gf_arm_v2.npz four times. ``_arm_tag`` is the cache-name map.
from lisatools.globalfit.monitor._identity import (  # noqa: E402
    arm_tag as _arm_tag,
    run_identity as _run_identity,
)
from lisatools.globalfit.monitor._png import shrink_png as _shrink_png  # noqa: E402

_base = os.path.basename(os.path.normpath(RUN_DIR))
RUN_LABEL, RUN_KIND = _run_identity(_base)

# ---- persistent commentary sidecar (2026-08-27) --------------------------
# Hand-written notes must survive page regeneration (the page is rebuilt
# from scratch every snapshot): drop HTML into commentary_{RUN_KIND}.html
# NEXT TO the output file and it renders as a "Commentary" section at the
# top of <main>. Absent file -> no section, no cost.
_comm_path = os.path.join(
    os.path.dirname(os.path.abspath(OUT)) or ".",
    f"commentary_{RUN_KIND}.html")
if os.path.exists(_comm_path):
    with open(_comm_path, "r") as _cf:
        COMMENTARY = (
            '<section id="commentary"><h2>Commentary</h2>\n'
            + _cf.read() + "\n</section>\n")
    print(f"[commentary] injected {_comm_path}")
else:
    COMMENTARY = ""

sub = g["sub_backend"]
psd_c = sub["psd/chain"][:NIT]                      # (it, 12, 24, 1, 2)
gal_c = sub["galfor/chain"][:NIT]                   # (it, 12, 24, 1, 5)
# Two VGB sampling bases are live in production (stock/erebor/vgb.py):
#   * legacy 5-param DIST basis: [dist, phi0, cos_iota, psi, fdot_astro_ratio]
#     with per-leaf FIXED Mc from the catalogue
#   * 6-param CHIRP basis (VGB_CHIRP_MASS_BASIS=1, used by 6-month runs):
#     [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]  -- Mc sampled
# The panel that reassembles the 9-col GB basis (F1 residual, DTR panels)
# below MUST know which basis this store used or it will read a chirp mass
# out of the fdot_astro_ratio slot and vice versa, collapsing every VGB
# template's amplitude to ~zero (silent -- no crash, the residual just
# reads as if VGB isn't being subtracted). Store's last chain dim tells
# them apart. Keep the FULL sampled row; downstream reads by column index.
_vgb_chain_ndim = int(sub["vgb/chain"].shape[-1])
VGB_SAMPLED_DIST = _vgb_chain_ndim == 5
VGB_SAMPLED_CHIRP = _vgb_chain_ndim == 6
if not (VGB_SAMPLED_DIST or VGB_SAMPLED_CHIRP):
    raise RuntimeError(
        f"unrecognised VGB chain last-dim {_vgb_chain_ndim} "
        f"(expected 5 for DIST basis or 6 for CHIRP basis)")
vgb_c = sub["vgb/chain"][:NIT, 0]                   # (it, W, 55, 5 or 6)
vgb_hh = sub["vgb/h_h"][:NIT]                       # (it, W, 55)
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
# gb_inds / gb_chain_cold / gb_alive_last are loaded below _row, which they
# read through.

# ---- POOLED SAMPLE WINDOWS (2026-08-27, user request) --------------------
# Every panel that SCATTERS per-leaf cold-chain samples used to draw ONE
# stored iteration -- the latest. A single iteration of a trans-dimensional
# branch is not a posterior, it is a snapshot of where the walkers happen to
# sit at one step: a source alive in 23 of the 24 cold walkers draws 23
# points, and their spread is walker scatter, not sampler uncertainty.
# Pooling the last few stored iterations makes each panel the thing the
# reader already believes they are looking at.
#
# Two windows, deliberately different:
#   * POOL_ITS_SAMPLES    -- scatter / explorer / marginal panels.
#   * POOL_ITS_POSTERIOR  -- corner plots, which need the extra rows before
#     their 2-D contours mean anything (this is the window the VGB and GB
#     corners already used; it is routed through the helper now so every
#     pooling site in the file reads from ONE place).
#
# THREE RULES, none of them optional:
#   1. min(requested, available) -- a store with three written rows pools
#      three, and nothing anywhere may assume the full window exists.
#   2. The alive mask is applied PER ITERATION. ``inds`` at iteration N says
#      nothing about N-4: the branch is trans-dimensional, a leaf alive now
#      may be dead then, and a dead row's coordinates are stale-or-zero
#      garbage that draws as a phantom source (f0 = 0 with the zero fill).
#   3. A reduced ``*_extract.h5`` store carries real coordinates ONLY inside
#      its final keep-window -- earlier rows are zero-filled placeholders
#      that survive as valid-looking arrays. Clamp the window to rows whose
#      coordinates are actually populated. (The full production store this
#      was validated on is NOT an extract, so the clamp is a no-op there;
#      it exists because the same generator is pointed at extracts.)
POOL_ITS_SAMPLES = 30
POOL_ITS_POSTERIOR = 30


def _zoom_its(n_avail):
    """Trailing stored iterations a "recent history" zoom should cover.

    Every in-pane inset -- cold-chain lnL, its walker spread, the GB leaf
    count, the two instrument-parameter traces -- plus the full-size
    ``psd_evolution_zoom`` panel, which is the same idea at panel scale,
    used to ask for a flat ``min(50, N)``.

    That is not a zoom early in a run. At N = 14 the inset redraws the
    parent panel's entire x-range on 35% of its area, so the two carry the
    same curve and the inset costs its host a third of the plot for
    nothing. Worse, it is exactly when the run is young -- when the last
    few iterations are the only ones anyone is reading -- that the zoom
    stops working.

    Scale the window to what exists (user ruling 2026-09-27):

        N < 25  -> 5      N < 50   -> 10
        N < 100 -> 25     N >= 100 -> 50

    Clamped to ``N`` so a two-row store still returns something drawable;
    callers keep their own ``>= 3`` floor on whether to draw at all.
    """
    n = int(max(n_avail, 0))
    if n < 25:
        w = 5
    elif n < 50:
        w = 10
    elif n < 100:
        w = 25
    else:
        w = 50
    return min(w, n)


# Bound ONCE, here, because the panel and its caption are built ~4500 lines
# apart and the panel sits inside a try/except that may never run. A
# caption that says "last 50" over a 5-iteration plot is worse than no
# caption, and that is exactly what a second min(50, ...) at the caption
# site would produce the moment the two drifted.
PSD_ZOOM_N = _zoom_its(SUB_NIT)
PSD_ZOOM_ALT = f"sensitivity evolution (last {PSD_ZOOM_N} iterations)"

EXTRACT_STORE = "_extract" in os.path.basename(h5path)
_POOL_FLOOR = {}


def _pool_floor(branch="gb"):
    """Earliest stored row of ``branch`` whose coordinates are real.

    0 for a full store (every written row carries its coordinates). For an
    extract, walk backwards from the newest row and stop at the first one
    whose alive leaves do NOT carry a non-zero f0 -- that is the edge of the
    keep-window. Probes ONE walker (the one holding the most alive leaves)
    and ONE parameter column, so the cost is a couple of chunks per row.
    """
    if branch in _POOL_FLOOR:
        return _POOL_FLOOR[branch]
    floor = 0
    if EXTRACT_STORE:
        floor = max(NIT - 1, 0)
        for _i in range(NIT - 1, -1, -1):
            try:
                _al = g[f"inds/{branch}"][_i, 0, 0]
                if not _al.any():
                    break
                _w = int(np.argmax(_al.sum(axis=1)))
                _f0 = g[f"chain/{branch}"][_i, 0, 0, _w, :, 1][_al[_w]]
            except Exception:
                break
            if not np.any(_f0 != 0.0):
                break
            floor = _i
        if floor > 0:
            MISSING.append(
                f"reduced *_extract.h5 store: {branch} coordinates are real "
                f"only from stored iteration {floor} onwards, so the pooled "
                f"sample panels use that keep-window, not the full "
                f"{POOL_ITS_POSTERIOR}-iteration request.")
    _POOL_FLOOR[branch] = floor
    return floor


def _pool_its(nwant, branch="gb", nit=None):
    """Stored-iteration indices a pooled panel should read, oldest first.

    ``nwant`` is the REQUEST (5 or 10); what comes back is
    ``min(nwant, available)`` indices, floored at the extract keep-window.
    """
    n = int(NIT if nit is None else nit)
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.arange(max(_pool_floor(branch), n - int(nwant)), n)


def _pool_gb_iter(nwant, cols=None):
    """Yield ``(it, alive, chain)`` for each pooled GB iteration, oldest first.

    ``alive`` is THAT iteration's own ``inds`` row -- rule 2 above. ``chain``
    is ``(nwalk, nleaf, len(cols))``, sliced column-wise because the chain is
    chunked ``(..., 1)`` on the parameter axis: pulling three columns
    decompresses a third of the bytes a full-row read would. A row that
    cannot be read (see ``_row``) is left out of the pool.
    """
    for _i in _pool_its(nwant, "gb"):
        _al = gb_inds[_i]
        if not _al.any():
            continue
        if cols is None:
            _ch = _row("chain/gb", (_i, 0, 0))
        else:
            _cs = [_row("chain/gb", (_i, 0, 0, slice(None), slice(None), _c))
                   for _c in cols]
            _ch = None if any(_c is None for _c in _cs) else np.stack(_cs, axis=-1)
        if _ch is None:
            continue
        yield int(_i), _al, _ch
        del _ch


def _vgb_pool_rows(nwant):
    """Row indices of ``vgb_c`` a pooled VGB panel should use, oldest first.

    Same three rules. The VGB branch is fixed-dimension, so there is no
    alive mask to apply per iteration -- what there IS is a leading block of
    rows the branch had not started sampling yet (it comes online at stage
    2) and, on an extract, a zero-filled pre-keep-window block. Both read as
    "all zero / all NaN" and both would drag every posterior to 0 kpc, so
    unpopulated rows are dropped from the FRONT of the window.
    """
    n = int(VGB_NIT)
    if n <= 0:
        return np.zeros(0, dtype=int)
    lo = max(0, n - int(nwant))
    while lo < n - 1:
        _r = vgb_c[lo]
        if np.isfinite(_r).any() and np.abs(np.nan_to_num(_r)).sum() > 0:
            break
        lo += 1
    return np.arange(lo, n)


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


# PER-ROW READS GET THE SAME LADDER (2026-10-02). ``_safe`` guards whole
# datasets; the GB panels read ``chain/gb`` one stored row at a time,
# straight off ``g`` and outside any try, so one unreadable row killed the
# whole page (6mo cluster, ``python -m lisatools.globalfit.monitor`` on the
# LIVE run dir):
#   _leaf_f0 -> g["chain/gb"][it_, 0, 0, w, :, 1]
#   OSError: Can't synchronously read data (filter returned failure during read)
# That build had already read every chain/gb row cleanly at the leaf-count
# panel, so the row was not torn on disk: the saver appended in between, and
# a reader without SWMR is reading a file that is changing under it.
# _store_extract met the same thing on 2026-08-22 and takes the same ladder:
# the live row, else the same row from the run's backup copy (closed,
# atomically replaced, one save behind -- the row is identical, not an
# approximation), else skip that row and say so on the page.
_TORN_ROWS = []


def _row(name, sel):
    """``g[name][sel]`` for stored row(s) ``sel[0]``, or ``None`` if unreadable.

    ``sel[0]`` is a row index or a slice of rows. The backup copy only
    supplies rows below its own ``iteration`` attr: it is preallocated, and a
    row it has not written yet reads back as zeros -- which every GB panel
    would render as the model losing all of its sources.
    """
    try:
        return g[name][sel]
    except OSError:
        pass
    r0 = sel[0]
    hi = (r0.stop if r0.stop is not None else NIT) if isinstance(r0, slice) else int(r0) + 1
    where = f"rows {r0.start or 0}-{hi - 1}" if isinstance(r0, slice) else f"row {int(r0)}"
    bg = _backup_group()
    if bg is not None:
        try:
            if hi <= int(bg.attrs.get("iteration", 0)):
                val = bg[name][sel]
                _TORN_ROWS.append(f"{name} {where} <- backup copy")
                return val
        except Exception:
            pass
    _TORN_ROWS.append(f"{name} {where} SKIPPED")
    return None


# Every GB panel stands on these two; without them there is no page to degrade to.
gb_inds = _row("inds/gb", (slice(0, NIT), 0, 0))     # (it, 24, 10000)
gb_chain_cold = _row("chain/gb", (NIT - 1, 0, 0))    # (24, 10000, 9) last iter
if gb_inds is None or gb_chain_cold is None:
    raise RuntimeError(
        "chain/gb or inds/gb is unreadable in the store AND its running backup "
        f"copy ({'; '.join(_TORN_ROWS)}). If the run is live, rebuild after "
        "the save completes.")
gb_alive_last = gb_inds[NIT - 1]                     # (24, 10000)


def _opt(node, key):
    """A dataset that MAY SIMPLY NOT EXIST, read quietly.

    ``_safe`` is for datasets every store is supposed to carry: an absent one
    is a fault and earns a MISSING line. The per-walker cap / search-stage /
    RJ-valve families are the opposite case -- they are allocated only when
    their (default-off) feature flag is set, so every store written before
    2026-09-24 and every run with the flags off legitimately lacks them. Those
    must cost NO MISSING line and NO placeholder: the panels they feed are
    simply not emitted. Returns ``None`` for absent, unreadable or empty.
    """
    try:
        ds = node.get(key)
    except Exception:
        return None
    if ds is None:
        return None
    try:
        arr = ds[:NIT] if NIT else ds[()]
    except Exception:
        # Torn chunk in a snapshot copied mid-save. A per-walker diagnostic
        # panel is not worth a MISSING line of its own; drop it silently,
        # exactly as if the feature had been off.
        return None
    arr = np.asarray(arr)
    return arr if arr.size else None


def _per_cell(arr, how="max"):
    """Legacy per-(iteration, cell) view of a cap array of EITHER layout.

    PER-WALKER CAPS (2026-09-22, ``GB_LEAF_CAP_PER_WALKER``). The gate's state
    became ``cap_cell_leaf_cap_w`` ``(nwalkers, ncells)``; the 1-D arrays this
    page has always read are kept live as MAX-OVER-WALKERS mirrors
    (``gbspecialstretch._mirror_band_leaf_cap``), so in practice the stored
    ``cap_cell_leaf_cap`` is still ``(nit, ncells)`` and every panel below is
    unchanged. This normalizer exists so that a store which ever DOES carry a
    walker axis in the mirror slot -- a hand-built one, a future layout, or a
    mirror that a cap-free branch left unrefreshed -- renders the envelope
    instead of raising inside ``imshow``. ``how="max"`` matches the mirror's
    own reduction, so the collapsed array is the same number the run wrote.
    """
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a.max(axis=1) if how == "max" else a.min(axis=1)
    return a

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
if caps is not None and getattr(caps, "size", 0):
    caps = _per_cell(caps)
# The per-walker cap table itself (``GB_LEAF_CAP_PER_WALKER``, default off).
# ``None`` on every store written before 2026-09-22 and on every run with the
# flag off -- the spread panel is then simply not emitted. Stored layout is
# ``(nit, nwalkers, ncells)``.
CAPS_W = _opt(sub, "gb/cap_cell_leaf_cap_w")


logpaths = discover_run_logs(RUN_DIR)
log_text = "".join(open(p, errors="replace").read() for p in logpaths)
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
# Each cold walker in its own colour plus the walker mean (user request
# 2026-10-12; was every walker in one cyan + max + median).
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
plot_walker_lines(ax[0], it, ll)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("cold-chain lnL")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title(f"total log-likelihood ({nwalk} walkers)")
ax[1].plot(it, ll.max(axis=1) - ll.min(axis=1), color=VIOLET, lw=1.5)
ax[1].set_xlabel("iteration"); ax[1].set_title("walker lnL spread (max - min)")

# In-pane zoom over the recent history (user request 2026-09-20; window
# made adaptive 2026-09-27 -- see _zoom_its). Shared with the GB leaf-count
# inset further down, which reuses _n_zoom/_it_zoom.
_n_zoom = _zoom_its(NIT)
_it_zoom = it[-_n_zoom:]
_ll_zoom = ll[-_n_zoom:]
if _n_zoom >= 3:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Left panel zoom: last 50 iterations of all traces
    axins0 = inset_axes(ax[0], width="35%", height="30%", loc="lower right",
                        borderpad=1.2)
    plot_walker_lines(axins0, _it_zoom, _ll_zoom, label=False, lw=0.8, mean_lw=1.6)
    axins0.tick_params(labelsize=7)
    axins0.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins0.grid(True, alpha=0.25, lw=0.5)
    # Right panel zoom: last 50 iterations of spread
    axins1 = inset_axes(ax[1], width="35%", height="30%", loc="upper right",
                        borderpad=1.2)
    axins1.plot(_it_zoom, _ll_zoom.max(axis=1) - _ll_zoom.min(axis=1),
                color=VIOLET, lw=1.3)
    axins1.tick_params(labelsize=7)
    axins1.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins1.grid(True, alpha=0.25, lw=0.5)

fig_b64(fig, "ll")

# ---- F11: the noise model, in two panels ----------------------------------
# The page used to carry six: two instrument traces, two instrument
# histograms, five foreground traces and five foreground histograms, none of
# which said whether the noise model is RIGHT. What matters is (a) do the two
# instrument parameters recover their injected values, and (b) is the
# foreground coming down as sources leave the residual. Two panels, both
# answerable at a glance.
psd_cold = psd_c[:, 0, :, 0, :]                     # (it, 24, 2)
gal_cold = gal_c[:, 0, :, 0, :]                     # (it, 24, 5)  SAMPLING basis
# Resolved, never a literal. The pair used to be pasted in as
# 1.496182e-11 / 2.982412e-15, which is the EQUAL-arm fit of the brick's
# tabulated estimates and sits 0.26% / 0.59% BELOW what mojito-light actually
# injects (1.5e-11 / 3e-15). Drawing that as "truth" made a 6-month
# noise-only fit accurate to -0.02% / -0.22% read as biased by +0.23% /
# +0.37% (2026-09-23). psd_truth_levels honors PSD_TRUTH, else refits the
# brick with the unequal-arm model, else falls back to the round injection.
from lisatools.globalfit.stock.erebor.noise import psd_truth_levels

SOMS_INJ, SA_INJ = psd_truth_levels(
    # MOJITO_INFO_PATH is the documented knob; MOJITO_DATA_PATH is the
    # legacy fallback. Note this call is the SILENT one -- with neither set
    # it returns the round injection and says nothing -- so the page has to
    # say it instead, or "analytic fallback" is invisible on the page and
    # only appears in whatever scrollback the operator happens to keep.
    mojito_data_path=MOJITO_INFO_PATH or os.environ.get("MOJITO_DATA_PATH")
)
if not (MOJITO_INFO_PATH or os.environ.get("MOJITO_DATA_PATH")):
    MISSING.append(
        "no MOJITO_INFO_PATH: the Soms_d / Sa_a truth lines are the analytic "
        f"injection ({SOMS_INJ:.4g}, {SA_INJ:.4g}), not a fit to the brick. "
        "The brick refit agrees to 3e-6 / 4e-5, so this is cosmetic here -- "
        "but set MOJITO_INFO_PATH to the folder holding catalogues/ and "
        "data/ and the page stops guessing.")

# Under GALFOR_LOG_SAMPLING the four log columns are stored as log10 while
# alpha stays linear. The trace/histogram panels want the SAMPLING basis
# (that is what the chain actually explores, and what its prior is uniform
# in); every sensitivity evaluation wants the PHYSICAL one. Keeping both and
# labelling them correctly is the whole fix -- feeding the raw log10 row to
# the foreground model makes amp and f_1 negative, so (f/f_1)**alpha is NaN
# and the entire data/template/residual block raises. That surfaced as a
# fatal KeyError: 'nbins' far downstream, with no page written at all.
from lisatools.globalfit.stock.erebor.noise import (
    GALFOR_BASIS, GALFOR_LOG_PARAMS, galfor_params_to_physical,
    read_noise_model_identity,
)

# The store's OWN basis flag -- never an env var, which describes the current
# process rather than the run that wrote these numbers.
GALFOR_LOG = bool(
    read_noise_model_identity(h5path).get("galfor_log_sampling", False))
gal_cold_phys = galfor_params_to_physical(gal_cold, GALFOR_LOG)
GAL_NAMES = [("log10 " + n) if (GALFOR_LOG and n in GALFOR_LOG_PARAMS) else n
             for n in GALFOR_BASIS]

# The lisatools/eryn import chain above (via erebor.noise -> eryn.utils.plot)
# calls plt.style.use(["science"]), which flips text.usetex=True on any host
# that has scienceplots installed. Every Text object created after this point
# captures rcParams["text.usetex"] at __init__, so we MUST pin it back to
# _USETEX BEFORE the next plt.subplots() call -- the later rcParams.update
# block only takes effect at figure creation time for new Text, not for
# already-existing ones. Fixing it here keeps the page tex-free even when the
# host has no LaTeX (i.e., every cluster node and every fresh container).
# ---- OWN THE FONT SIZE, DO NOT INHERIT IT (2026-09-26) -------------------
# MEASURED, not assumed: importing the erebor.noise chain above changes
# EXACTLY ONE rcParam -- font.size 10.0 -> 16.0. Nothing else moves. (The
# comment below about plt.style.use(["science"]) describes a real hazard on
# hosts that HAVE scienceplots, but this one does not have it, so that call
# fails and the font is the whole story here.)
#
# That one number made the page depend on the caller's import history:
#
#   fresh interpreter -> this file sets font.size 10 far above, the import
#                        HERE raises it to 16, and the next figure renders
#                        at 16.
#   already imported  -> it was 16 before this file ran, this file's own
#                        block set it to 10, and the next figure renders
#                        at 10.
#
# One figure is drawn immediately below (the instrument-noise posteriors),
# so that panel alone came out different: 25722 vs 27586 base64 chars,
# 25806 pixels across the whole plot area, every other line of the page
# identical. It is why the page was reproducible only in a child process.
#
# Pinned to 16 because that is what every page produced to date rendered
# at, and reproduction is the requirement. It is NOT a considered choice
# of font size -- it is an inherited side effect, now stated. If the page
# should own its typography, change this to _BASE_FONT and expect every
# noise panel to change with it.
_NOISE_PANEL_FONT = 16.0
plt.rcParams["font.size"] = _NOISE_PANEL_FONT
plt.rcParams["text.usetex"] = _USETEX

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
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_b64(fig, "f11_psd")

try:
    # XYZ EVERYWHERE (user ruling 2026-08-19): the run analyses X/Y/Z, so
    # every channel-PSD display uses the X channel and X2TDISens -- no AET
    # projections anywhere on this page.
    from lisatools.sensitivity import get_sensitivity, X2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    import matplotlib.colors as mcolors

    fr = np.logspace(np.log10(3e-4), np.log10(2.5e-2), 400)

    def sens_curves(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms ** 2, sa ** 2,
                                      lisa_models.DefaultOrbits(), "sampled")
        if galp is None:
            return get_sensitivity(fr, sens_fn=X2TDISens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=X2TDISens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    ramp = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    pm = np.median(psd_cold[-1], axis=0)

    # EVERY WALKER, NEVER THE PARAMETER-WISE MEDIAN (user ruling 2026-09-24).
    # The foreground posterior is bimodal in alpha: walkers sit near 1.6 or
    # near 7 on one degeneracy ridge, and both ends produce curves that agree
    # to 4%. Taking the median COLUMN BY COLUMN across four walkers does not
    # give either -- with a 2/2 split it builds a hybrid parameter vector
    # (alpha ~ 4.8) whose hyperbolic tangent is not the average of two
    # hyperbolic tangents. Measured on the 6mo store over the last 50
    # iterations: the walkers spanned 1.04x at 3 mHz while the median curve
    # spanned 1.81x, and the upper branch it drew sat 1.76x above anything
    # any walker held. That branch was an artifact of this line, not the
    # foreground moving. Plot the walkers; let the reader see the spread.
    def _walker_noise(k):
        """``[(psd_params, galfor_params), ...]`` for every cold walker at k."""
        return [(psd_cold[k, w], gal_cold_phys[k, w]) for w in range(nwalk)]

    # Full-run panels draw nwalk curves per iteration, so decimate the
    # iteration axis to keep the figure legible and the PNG small.
    _step = max(1, SUB_NIT // 250)
    _its = list(range(0, SUB_NIT, _step))
    if _its[-1] != SUB_NIT - 1:
        _its.append(SUB_NIT - 1)

    fig, ax = plt.subplots(figsize=(11, 4.0))
    for k in _its:
        for _w, (pk_, gk) in enumerate(_walker_noise(k)):
            ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                    color=ramp(k / max(SUB_NIT - 1, 1)), lw=0.8, alpha=0.55,
                    label=(f"iteration {k}"
                           if (_w == 0 and k in (0, SUB_NIT - 1)) else None))
    ax.plot(fr, sens_curves(*pm), color=FG, lw=1.4, ls=":",
            label="instrument only (latest)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD, TDI X channel  [1/Hz]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"PSD + foreground, all {nwalk} cold walkers per stored iteration")
    fig_b64(fig, "f11_fg")

    # Last-50-iterations zoom of the same plot (user request 2026-09-21).
    _last_n = min(50, SUB_NIT)
    if _last_n >= 3:
        fig, ax = plt.subplots(figsize=(11, 4.0))
        for k in range(SUB_NIT - _last_n, SUB_NIT):
            # Normalize color index to the last-50 window
            _color_idx = (k - (SUB_NIT - _last_n)) / max(_last_n - 1, 1)
            for _w, (pk_, gk) in enumerate(_walker_noise(k)):
                ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                        color=ramp(_color_idx), lw=0.8, alpha=0.6,
                        label=(f"iteration {k}"
                               if (_w == 0 and k in (SUB_NIT - _last_n,
                                                     SUB_NIT - 1)) else None))
        ax.plot(fr, sens_curves(*pm), color=FG, lw=1.4, ls=":",
                label="instrument only (latest)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD, TDI X channel  [1/Hz]")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"PSD + foreground, all {nwalk} cold walkers "
                     f"(last {_last_n} iterations)")
        fig_b64(fig, "f11_fg_zoom")
except Exception as e:
    MISSING.append(f"foreground curve render failed: {e!r}")


# ---- RESTORED: per-parameter noise traces and histograms -------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    for w in range(nwalk):
        ax[j].plot(np.arange(SUB_NIT), psd_cold[:, w, j], color=CYAN, alpha=0.3, lw=0.8)
    ax[j].axhline(inj, color=RED, lw=1.4, ls=":", label="injected")
    ax[j].set_title(f"psd: {name}"); ax[j].set_xlabel("iteration"); ax[j].legend()

    # Inset: the recent-history window (adaptive, see _zoom_its)
    _last_n = _zoom_its(SUB_NIT)
    if _last_n >= 3:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_in = inset_axes(ax[j], width="40%", height="35%", loc="center")
        for w in range(nwalk):
            ax_in.plot(np.arange(SUB_NIT - _last_n, SUB_NIT),
                      psd_cold[SUB_NIT - _last_n:, w, j],
                      color=CYAN, alpha=0.3, lw=0.8)
        ax_in.axhline(inj, color=RED, lw=1.2, ls=":")
        ax_in.set_title(f"last {_last_n} iter", fontsize=7)
        ax_in.tick_params(labelsize=6)
        ax_in.grid(True, alpha=0.3)
fig_b64(fig, "psd_trace")

fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    v = psd_cold[-min(3, SUB_NIT):, :, j].ravel()
    ax[j].hist(v, bins=min(30, nwalk), color=CYAN, alpha=0.85)
    ax[j].axvline(inj, color=RED, lw=1.4, ls=":")
    ax[j].set_title(f"{name} posterior (last {min(3,NIT)} iters x {nwalk} walkers)")
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
    # gm: the parameter-wise walker median. NO LONGER DRAWN (2026-09-24) --
    # kept only because the _detect table and the SNR machinery still take
    # a single representative noise model. See the f11_fg note.
    gm = np.median(gal_cold_phys[-1], axis=0)  # noqa: F841

    # Compute injected + FittedHT foreground for comparison
    from lisatools.stochastic import (
        FittedHyperbolicTangentGalacticForeground as _FHT_sens)
    _gal_fht = _FHT_sens.specific_Sh_function(fr, SCI_TOBS)

    # One curve per cold walker, not the parameter-wise median -- see the
    # note at the f11_fg panel for why that median draws a foreground no
    # walker holds when the alpha posterior is split.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.6, label="instrument PSD (sampled)")
    for w in range(nwalk):
        ax.plot(fr, sens_lisasens(psd_cold[-1, w, 0], psd_cold[-1, w, 1],
                                  gal_cold_phys[-1, w]),
                color=AMBER, lw=1.4, alpha=0.75,
                label="PSD + galactic foreground (per walker)" if w == 0 else None)
    ax.plot(fr, sens_lisasens(SOMS_INJ, SA_INJ), color=RED, ls=":", lw=1.3,
            label="injected instrument")
    ax.plot(fr, sens_lisasens(SOMS_INJ, SA_INJ) + _gal_fht, color=RED, ls="-.",
            lw=1.3, alpha=0.8, label="injected + FittedHT foreground")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(fontsize=9); ax.set_title(
        f"sensitivity, all {nwalk} cold walkers, latest stored iteration")
    fig_b64(fig, "psd_curves")

    ramp2 = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    _step2 = max(1, SUB_NIT // 250)
    _its2 = list(range(0, SUB_NIT, _step2))
    if _its2[-1] != SUB_NIT - 1:
        _its2.append(SUB_NIT - 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.4,
            label="instrument PSD (latest)")
    for k in _its2:
        for w in range(nwalk):
            ax.plot(fr, sens_lisasens(psd_cold[k, w, 0], psd_cold[k, w, 1],
                                      gal_cold_phys[k, w]),
                    color=ramp2(k / max(SUB_NIT - 1, 1)), lw=0.8, alpha=0.55,
                    label=(f"iter {k}"
                           if (w == 0 and k in (0, SUB_NIT - 1)) else None))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        f"PSD + foreground, all {nwalk} cold walkers per stored iteration "
        "(light -> dark = later)")
    fig_b64(fig, "psd_evolution")

    # Recent-history zoom. Same adaptive window as the insets (_zoom_its):
    # this is the one full-size panel built to the inset's pattern, so a
    # flat 50 made it a duplicate of the panel above it on a young run.
    _last_n = PSD_ZOOM_N
    if _last_n >= 3:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.4,
                label="instrument PSD (latest)")
        for k in range(SUB_NIT - _last_n, SUB_NIT):
            _color_idx = (k - (SUB_NIT - _last_n)) / max(_last_n - 1, 1)
            for w in range(nwalk):
                ax.plot(fr, sens_lisasens(psd_cold[k, w, 0], psd_cold[k, w, 1],
                                          gal_cold_phys[k, w]),
                        color=ramp2(_color_idx), lw=0.8, alpha=0.6,
                        label=(f"iter {k}"
                               if (w == 0 and k in (SUB_NIT - _last_n,
                                                    SUB_NIT - 1)) else None))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
        ax.legend(); ax.set_title(
            f"PSD + foreground, all {nwalk} cold walkers, last {_last_n} "
            "stored iterations (light -> dark = later)")
        fig_b64(fig, "psd_evolution_zoom")
except Exception as e:
    MISSING.append(f"LISASens curve render failed: {e!r}")

# ---- GB leaf count (the raw model size; the recovery section below is
# where it is given a denominator) ----------------------------------------
gb_counts = gb_inds.sum(axis=-1)                    # (it, 24)

# ---- detectable-truth set for the overlays (from the census npz) ----------
# `census.py` computes optimal SNR for every catalogue GB over the analysed
# band against the run's OWN sampled sensitivity; the detectable (SNR>7) subset
# is the natural TARGET line for the leaf-count and occupancy panels. It is
# optional -- without the npz these overlays simply do not draw.


def _mhz(x_hz):
    """A band edge as an mHz string, at a precision that survives the low end.

    The band floor is 0.5556 mHz, and the ``:.0f`` these labels used to carry
    renders that as "1" -- a label that is wrong by a factor of two and reads
    as a different band. ``:.4g`` gives "0.5556" and "21.94" from the same
    format, so every band string on the page is produced HERE and the low and
    high edges cannot drift apart in precision.
    """
    return f"{float(x_hz) * 1e3:.4g}"


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

# FALLBACK to the frozen truth npz (2026-08-22). `gb_hi_f_census.npz` is
# written by a `census.py` that lives in a SCRATCH directory, so it does not
# survive a session -- and without it BOTH of the red detectability overlays
# below (the "N detectable (SNR>7)" target line on the leaf-count panel and
# the "has detectable source" curve on the occupancy panel) vanish with no
# entry in MISSING. `gb_truth_3to21.npz` carries the same column: `det` is
# SNR > 7 against the run's own sampled sensitivity over the band stamped in
# `band`, which is exactly what `det_f0` was. Same Tobs guard as the recovery
# section further down -- an unstamped (legacy) npz is treated as 3-month.
if DET_F0 is None:
    _run_tobs = 7776000.0
    try:
        _a0 = dict(f["global_fit/domain_settings/args"].attrs)
        _run_tobs = float(_a0["0"]) * float(_a0["1"]) * float(_a0["2"])
    except Exception:
        pass
    for _tp in (os.path.join(RUN_DIR, "gb_truth_3to21.npz"),
                "gb_truth_3to21.npz"):
        if os.path.exists(_tp):
            try:
                _t = np.load(_tp)
                _t_tobs = (float(np.asarray(_t["tobs"]).reshape(-1)[0])
                           if "tobs" in _t.files else 7776000.0)
                if abs(_t_tobs - _run_tobs) > 1.0:
                    MISSING.append(
                        "detectability overlays skipped: no gb_hi_f_census.npz "
                        f"and the truth set is built for a "
                        f"{_t_tobs / 86400.0:.4g}-day observation while this "
                        f"run is {_run_tobs / 86400.0:.4g} days.")
                elif "det" in _t.files and "f0" in _t.files:
                    _d = np.asarray(_t["det"], dtype=bool)
                    DET_F0 = np.asarray(_t["f0"], dtype=float)[_d]
                    _b = np.asarray(_t["band"], dtype=float).reshape(-1)
                    DET_LO, DET_HI = float(_b[0]), float(_b[1])
            except Exception as _e:
                DET_F0 = None
                MISSING.append(
                    f"detectability overlays unavailable: {_e!r}")
            break
    else:
        MISSING.append(
            "detectability overlays skipped: neither gb_hi_f_census.npz nor "
            "gb_truth_3to21.npz is beside the store or in the CWD, so the "
            "leaf-count and occupancy panels carry no detectable-source "
            "target line.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
# Each cold walker in its own colour plus the walker mean (user request
# 2026-10-12; was every walker in one green + the max over walkers for each
# frequency selection). Two selections per walker would be eight thin lines
# in two near-parallel bundles, so when the analysed band is known the walkers
# are drawn over THAT band (the one the detectable target counts) and the
# all-frequency count is its walker mean alone, dashed; the legend says so.
# _gb_walk is the per-walker selection, reused by the zoom inset below.
# TRUTH TARGET. The raw leaf count has no scale -- 577 leaves is only
# meaningful against how many sources are actually THERE. Overlay the
# detectable (SNR>7) catalogue count over the range where SNRs exist, and
# the model's leaf count restricted to that SAME range, so the two lines
# are comparable rather than merely adjacent.
if DET_F0 is not None:
    # row by row, f0 column only; an unreadable row is a gap (NaN), not a 0
    _cnt = np.full((NIT, gb_inds.shape[1]), np.nan)       # (it, walker)
    for _i in range(NIT):
        _f0_i = _row("chain/gb", (_i, 0, 0, slice(None), slice(None), 1))
        if _f0_i is not None:
            _f0_i = _f0_i * 1e-3
            _cnt[_i] = ((_f0_i >= DET_LO) & (_f0_i <= DET_HI)
                        & gb_inds[_i]).sum(axis=1)
    _gb_walk = _cnt
    plot_walker_lines(ax[0], it, _cnt)
    ax[0].plot(it, gb_counts.mean(axis=1), color=WALKER_MEAN_COLOR, lw=1.6,
               ls=(0, (5, 2)), zorder=3, label="all f, mean only")
    ax[0].axhline(DET_F0.size, color=RED, ls=":", lw=1.5)
    ax[0].text(NIT * 0.98, DET_F0.size,
               f"{DET_F0.size} detectable (SNR>7), "
               f"{_mhz(DET_LO)}-{_mhz(DET_HI)} mHz ", color=RED,
               fontsize=8, va="bottom", ha="right")
    _gb_sel = f"{_mhz(DET_LO)}-{_mhz(DET_HI)} mHz"
else:
    _gb_walk = gb_counts
    plot_walker_lines(ax[0], it, gb_counts)
    _gb_sel = "all f"
ax[0].legend(fontsize=7.5, loc="lower right", ncol=3, handlelength=1.4, columnspacing=1.0,
             title=f"{_gb_sel}: walkers + mean", title_fontsize=7.5)
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

# In-pane zoom for GB leaf count: last 50 iterations (user request 2026-09-20).
# The per-walker selection of the main panel and its mean; the all-f mean is
# left out here, where its offset would squash the walkers into a flat band.
_gb_zoom = _gb_walk[-_n_zoom:]
if _n_zoom >= 3:
    axins_gb = inset_axes(ax[0], width="35%", height="30%", loc="center",
                          borderpad=1.2)
    plot_walker_lines(axins_gb, _it_zoom, _gb_zoom, label=False, lw=0.8, mean_lw=1.6)
    axins_gb.tick_params(labelsize=7)
    axins_gb.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins_gb.grid(True, alpha=0.25, lw=0.5)

# Recipe-stage boundaries on the leaf trace. THE reason this panel needs
# them: under the v9 three-stage search the opt-SNR prior boundary drops
# 8 -> 5 and the F-stat peak floor 8 -> 6.25 at the gb_search_1 -> _2
# boundary, so the leaf count is EXPECTED to step up there. Without the rule
# drawn, that step reads as a discovery burst.
mark_stages(ax[0], max_it=int(it[-1]) if len(it) else None)
mark_stages(ax[1], label=False, max_it=int(it[-1]) if len(it) else None)

fig_b64(fig, "gb_leaves")

# ---- 5a-0. GB LOG-LIKELIHOOD PER SUB-BAND, PER WALKER (2026-10-09) --------
# User request: "the logl per subband per walker over time", all walkers
# drawn alike, each sub-band told apart by frequency, plus the last 100
# iterations. Per stored row, cold walker and GB band: the sum over the
# band's live leaves of d_h - h_h/2, i.e. the log-likelihood the band's
# sources buy against the residual with them taken out -- the
# source-attributed statistic the cap-cell gate already uses
# (GBSpecial*._cap_cell_source_lls). On the 6mo store it spans 6 to 4e4
# nats across bands, hence the log axis.
# NOT the residual -1/2<r|r> per band: on the WDM grid a GB band is
# narrower than a layer, so most bands own no residual window
# (band_cold_logl_w is 0 in 3/4 of its cells) and that array stops moving
# once the search valve retires.
BAND_LL_ZOOM = 100            # the right-hand column's window, in rows
BAND_LL_MAX_ROWS = 300        # strided rows before that window (I/O bound)
BAND_LL_F_EDGES_MHZ = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0)   # colour classes
GB_BAND_LL_CAP = ""


def gb_band_ll_rows(nit, zoom=100, max_rows=300):
    """Stored rows the per-band panel reads: every row of the last ``zoom``,
    and at most ``max_rows`` evenly strided rows before them."""
    z0 = max(0, nit - zoom)
    stride = max(1, int(np.ceil(z0 / max_rows))) if z0 else 1
    return list(range(0, z0, stride)) + list(range(z0, nit))


def gb_band_ll(d_h, h_h, f0_hz, alive, edges):
    """``(nwalkers, nbands)`` sum of ``d_h - h_h/2`` over each band's live
    leaves; NaN where a walker holds no live leaf in the band."""
    nb = len(edges) - 1
    contrib = np.asarray(d_h, float) - 0.5 * np.asarray(h_h, float)
    ok = np.asarray(alive, bool) & np.isfinite(contrib) & np.isfinite(f0_hz)
    band = np.clip(np.searchsorted(edges, f0_hz, side="right") - 1, 0, nb - 1)
    out = np.zeros((contrib.shape[0], nb))
    cnt = np.zeros((contrib.shape[0], nb), dtype=int)
    for w in range(contrib.shape[0]):
        m = ok[w]
        np.add.at(out[w], band[w][m], contrib[w][m])
        np.add.at(cnt[w], band[w][m], 1)
    out[cnt == 0] = np.nan
    return out


try:
    if _opt(sub, "gb/d_h") is None or _opt(sub, "gb/h_h") is None:
        raise ValueError("the store carries no gb d_h / h_h")
    _bl_rows = gb_band_ll_rows(NIT, BAND_LL_ZOOM, BAND_LL_MAX_ROWS)
    _bl = np.full((len(_bl_rows), gb_inds.shape[1], len(band_edges) - 1), np.nan)
    for _k, _i in enumerate(_bl_rows):
        _dh = _row("sub_backend/gb/d_h", (_i,))
        _hh = _row("sub_backend/gb/h_h", (_i,))
        _f0 = _row("chain/gb", (_i, 0, 0, slice(None), slice(None), 1))
        if _dh is None or _hh is None or _f0 is None or not np.any(_dh):
            continue                     # unreadable or unwritten: a gap
        _bl[_k] = gb_band_ll(_dh, _hh, _f0 * 1e-3, gb_inds[_i], band_edges)
    _bl_have = np.isfinite(_bl).any(axis=(1, 2))
    if _bl_have.sum() < 2:
        raise ValueError(
            f"d_h / h_h are written in only {int(_bl_have.sum())} of the "
            f"{len(_bl_rows)} rows read (a short snapshot keeps the last row only)")
    _bl_x = np.asarray(_bl_rows)
    _fc = 0.5 * (band_edges[:-1] + band_edges[1:]) * 1e3          # mHz
    import matplotlib.colors as _blc
    from matplotlib.collections import LineCollection

    # DISCRETE frequency classes, not a continuous ramp: ~490 occupied bands
    # sit mostly at 2-10 mHz, and on a continuous log ramp they all come out
    # orange. Fixed edges keep a colour meaning the same thing on every page.
    _bl_edges = np.array([b for b in BAND_LL_F_EDGES_MHZ
                          if _fc.min() < b < _fc.max()], float)
    _bl_edges = np.concatenate([[_fc.min()], _bl_edges, [_fc.max()]])
    _bl_cmap = _blc.ListedColormap(
        plt.get_cmap("turbo")(np.linspace(0.10, 0.95, len(_bl_edges) - 1)))
    _bl_norm = _blc.BoundaryNorm(_bl_edges, _bl_cmap.N)
    _pos = np.where(_bl > 0, _bl, np.nan)       # log axis: positive gains only

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True,
                           gridspec_kw=dict(width_ratios=[1.7, 1.0], wspace=0.06))
    _z0 = max(0, NIT - BAND_LL_ZOOM)
    for _a, _sel in ((ax[0], slice(None)), (ax[1], _bl_x >= _z0)):
        _xs = _bl_x[_sel]
        _ys = _pos[_sel]                                  # (rows, walker, band)
        _segs, _cols = [], []
        for _b in np.nonzero(np.isfinite(_ys).any(axis=(0, 1)))[0]:
            for _w in range(_ys.shape[1]):
                _yb = _ys[:, _w, _b]
                if np.isfinite(_yb).sum() >= 1:
                    _segs.append(np.column_stack([_xs, _yb]))
                    _cols.append(_bl_cmap(_bl_norm(_fc[_b])))
        # shuffled draw order: band order would paint the high-f classes over
        # everything below them
        _ord = np.random.default_rng(0).permutation(len(_segs))
        _lc = LineCollection([_segs[j] for j in _ord], colors=[_cols[j] for j in _ord],
                             linewidths=0.6, alpha=0.45)
        _a.add_collection(_lc)
        _a.set_yscale("log")
        _a.set_xlim(float(_xs.min()), float(max(_xs.max(), _xs.min() + 1)))
        _a.set_xlabel("iteration")
    _fin = _pos[np.isfinite(_pos)]
    ax[0].set_ylim(max(float(_fin.min()) * 0.7, 1e-1), float(_fin.max()) * 1.5)
    ax[0].set_ylabel(r"band $\sum\,(\langle d|h\rangle - \langle h|h\rangle/2)$  [nats]")
    ax[0].set_title(f"GB log-likelihood per sub-band, {gb_inds.shape[1]} cold walkers")
    ax[1].set_title(f"last {min(BAND_LL_ZOOM, NIT)} iterations")
    mark_stages(ax[0], max_it=int(NIT - 1))
    _sm = plt.cm.ScalarMappable(norm=_bl_norm, cmap=_bl_cmap)
    _cb = fig.colorbar(_sm, ax=ax, pad=0.01, fraction=0.035,
                       ticks=_bl_edges, spacing="uniform")
    _cb.ax.set_yticklabels([f"{v:.3g}" for v in _bl_edges])
    _cb.set_label("sub-band centre frequency [mHz]")
    fig_b64(fig, "gb_band_ll")

    _last = _bl[np.nonzero(_bl_have)[0][-1]]                  # (walker, band)
    _occ = np.isfinite(_last).sum(axis=1)
    _neg = int((np.nan_to_num(_last, nan=1.0) <= 0).sum())
    _bmax = int(np.nanargmax(np.nanmax(_last, axis=0)))
    GB_BAND_LL_CAP = (
        f"One line per (cold walker, sub-band), all {gb_inds.shape[1]} walkers drawn "
        "alike and coloured by the sub-band's centre frequency in fixed classes "
        "(colour bar, mHz). "
        "The value is the log-likelihood the band's live GB leaves buy: the sum over "
        "them of &lang;d|h&rang; &minus; &lang;h|h&rang;/2, with d the residual plus "
        "that leaf's own template, so roughly half the band's summed SNR&sup2;. "
        f"At the last row {int(np.median(_occ))} of {_last.shape[1]} sub-bands are "
        f"occupied (median over walkers); the largest is "
        f"{np.nanmax(_last):,.0f} nats at {_fc[_bmax]:.2f} mHz. A band appears when "
        "it gains its first leaf and drops out when it loses its last. Left: "
        f"{len(_bl_rows) - min(BAND_LL_ZOOM, NIT)} strided rows plus the last "
        f"{min(BAND_LL_ZOOM, NIT)}; right: every one of the last "
        f"{min(BAND_LL_ZOOM, NIT)}. Flat lines in the right column are converged "
        "bands; steps are births, deaths or a source moving between bands."
        + (f" {_neg} (walker, band) value(s) are &le; 0 and fall off the log axis."
           if _neg else ""))
    del _bl, _pos, _segs, _cols
except Exception as _exc:  # noqa: BLE001 - a missing panel, never a dead page
    MISSING.append(f"GB log-likelihood per sub-band unavailable: {_exc}")

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
if cap_cells is not None and getattr(cap_cells, "size", 0):
    # Same both-layouts normalisation as ``caps`` above: this panel indexes
    # ``cap_cells[row]`` as a 1-D per-cell vector and broadcasts it against
    # ``(nwalkers, ncell)`` occupancy, which a walker axis would silently
    # misalign rather than raise on.
    cap_cells = _per_cell(cap_cells)
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
        """Sources per cap cell, per cold walker, at one stored iteration.

        ``None`` when that row of the chain cannot be read (see ``_row``).
        """
        alive = gb_inds[iteration]                               # (nw, nleaf)
        if iteration == NIT - 1:
            f0 = gb_chain_cold[..., 1]                           # already read
        else:
            f0 = _row("chain/gb", (iteration, 0, 0, slice(None), slice(None), 1))
            if f0 is None:
                return None
        f0 = f0 * 1e-3                                           # mHz -> Hz
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
        if cc is None:
            continue
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
                                 f"({_mhz(DET_LO)}-{_mhz(DET_HI)} mHz)")
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
    _tot_last = _tot[-1] if _tot else 0.0  # same NIT=0 guard as _occ/_atcap
    _exact1 = float((cc_last == 1).sum() / nw_)
    CAP_TXT = (
        f"At iteration {NIT-1} the median cold walker holds "
        f"<strong>{_tot_last:.0f}</strong> GB sources spread over "
        f"<strong>{_occ_last:.0f} of {ncell}</strong> cap cells "
        f"({100*_occ_last/ncell:.0f}%): <strong>{_exact1:.0f}</strong> cells "
        f"hold exactly one source and <strong>{_atcap_last:.0f}</strong> sit "
        f"at or over their cap.")

# ---- 5b. PER-(WALKER, BAND) SEARCH STATE ----------------------------------
# Three families landed on 2026-09-22/24, all indexed by (walker, band or
# cell) and all PERSISTED in band_info, which is what makes them plottable
# over the course of a stage and across stages rather than only inside one
# propose:
#
#   * cap_cell_leaf_cap_w  (GB_LEAF_CAP_PER_WALKER)  -- each walker earns,
#     holds and is enforced against its OWN per-cell allowance. The 1-D
#     arrays the panels above read are max-over-walker mirrors, so they show
#     the ENVELOPE; the spread is only in the _w array, and whether the
#     spread ever opens at all is the whole question the feature poses.
#     Design: docs/superpowers/specs/2026-09-22-gb-cap-per-walker-design.md
#   * band_rj_shutoff_w    (GB_SEARCH_BAND_SHUTOFF_PER_WALKER)  -- a
#     (walker, band) pair whose cold-chain lnL converged WITHIN the current
#     recipe step takes no further RJ until the next step releases it.
#     "every OCCUPIED pair has shut off" is the stage-convergence criterion
#     (recipe.band_shutoff_w_pending_total == 0), so the pending count going
#     to zero is a stage ending, and the reset at the step boundary is the
#     next one starting.
#   * band_stage_w         (GB_SEARCH_STAGE_PER_WALKER)  -- stage 0 COARSE /
#     stage 1 FINE per (walker, band); a one-way latch on the opt-SNR and
#     F-stat floors.
#     Design: docs/superpowers/specs/2026-09-24-gb-search-stage-schedule-design.md
#
# EVERY ONE OF THEM IS OPTIONAL. All three flags default to False and none of
# the arrays exists in a store written before the feature landed, which is
# most stores on disk. ``_opt`` returns None quietly in that case and each
# panel below is skipped whole -- no figure, no ``img()`` placeholder (the
# HTML fragment collapses to ""), no MISSING line. A store without these keys
# must render EXACTLY as it did before this section existed.
CAP_PW_TXT = SHUTOFF_W_TXT = STAGE_W_TXT = ""
CAP_PW_PANEL = SHUTOFF_W_PANEL = STAGE_W_PANEL = ""

# The cold-chain per-(walker, band) source census, ``band_counts[0]``, is the
# occupancy the valve and the stage latch are both judged against -- an EMPTY
# pair can never shut off and can never promote, so "all pairs shut" would be
# unreachable without it. Stored as (nit, ntemps, nwalkers, nbands); the
# temp-0 hyperslab is 1/ntemps of the bytes a full read would pull, which
# matters on a 1-yr store. Read ONLY when one of the new families is present,
# so an old store pays nothing for this section at all.
_occ_wb = None
if (CAPS_W is not None or _opt(sub, "gb/band_rj_shutoff_w") is not None
        or _opt(sub, "gb/band_stage_w") is not None):
    try:
        _occ_wb = np.asarray(sub["gb/band_num_binaries"][:NIT, 0])  # (it,w,b)
    except Exception:
        _occ_wb = None

# ---- 5b-1. per-walker leaf caps: did the allowances actually diverge? -----
if CAPS_W is not None and CAPS_W.ndim == 3:
    _cw = np.asarray(CAPS_W, dtype=float)            # (it, nw, ncell)
    _cn, _cnw, _cnc = _cw.shape
    # The cap is a SENTINEL (-1) until the GB search stage arms it, and a
    # disarmed row would read as a perfectly agreed cap of -1. Only rows where
    # some walker holds a real (>= 1) cap carry a spread worth quoting.
    _armed = np.any(_cw >= 1, axis=(1, 2))           # (it,)
    _spread = _cw.max(axis=1) - _cw.min(axis=1)      # (it, ncell)
    _ndiff = (_spread > 0).sum(axis=1)               # cells where walkers differ
    _mxsp = _spread.max(axis=1)
    _cit = np.arange(_cn)

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.6),
                           gridspec_kw=dict(wspace=0.42))

    # (a) is the feature doing anything? This is the plotted form of the
    # run log's once-per-iteration "[GB_CAP_PW] cap spread" line: a flat zero
    # means every walker ramped in lockstep and the per-walker caps bought
    # nothing, which is a real result and the first thing to check.
    ax[0].plot(_cit, _ndiff, color=VIOLET, lw=2, label="cells where walkers differ")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(f"cap cells (of {_cnc})")
    axr = ax[0].twinx()
    axr.plot(_cit, _mxsp, color=AMBER, lw=1.4, ls="--")
    axr.set_ylabel("max spread [leaves]", color=AMBER, fontsize=9)
    axr.tick_params(axis="y", colors=AMBER, labelsize=8); axr.grid(False)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("per-walker cap divergence"
                    + ("" if _ndiff[-1] else " (INERT: all walkers agree)"))

    # (b) each walker's own total allowance over the run. The mirror panel
    # above can only ever show the leading walker; this is the ensemble.
    _tot_w = _cw.clip(min=0).sum(axis=2)             # (it, nw)
    for _w in range(_cnw):
        ax[1].plot(_cit, _tot_w[:, _w], color=CYAN, alpha=0.45, lw=0.9)
    ax[1].plot(_cit, _tot_w.max(axis=1), color=AMBER, lw=1.8, label="max walker")
    ax[1].plot(_cit, _tot_w.min(axis=1), color=GREEN, lw=1.4, label="min walker")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("total allowance [leaves]")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title(f"allowance per cold walker ({_cnw} walkers)")

    # (c) WHERE the disagreement sits. Blocked to the band grid for the same
    # reason panel 5a(c) is: 1,232 hairlines is a moire, not a distribution.
    _last = int(np.max(np.nonzero(_armed)[0])) if _armed.any() else _cn - 1
    _K = max(int(round(_cnc / max(len(band_edges) - 1, 1))), 1)
    _nblk = _cnc // _K
    if _nblk >= 1:
        _sp_blk = _spread[_last][:_nblk * _K].reshape(_nblk, _K).max(axis=1)
        _fblk = (0.5 * (cap_edges_static[:-1] + cap_edges_static[1:])
                 )[:_nblk * _K].reshape(_nblk, _K).mean(axis=1) * 1e3
        ax[2].fill_between(_fblk, 0, _sp_blk, color=VIOLET, alpha=0.9, lw=0,
                           step="mid")
        ax[2].set_xlabel("f0 [mHz]")
        ax[2].set_ylabel("max cap spread in band")
        ax[2].set_title(f"where the walkers disagree @ iter {_last}")
    fig_b64(fig, "gb_cap_per_walker")

    _ndiff_last = int(_ndiff[_last]) if _cn else 0
    CAP_PW_TXT = (
        f"<code>GB_LEAF_CAP_PER_WALKER</code> is ON in this run: each of the "
        f"{_cnw} cold walkers carries its own allowance in each of the "
        f"{_cnc} cap cells. At iteration {_last}, "
        f"<strong>{_ndiff_last}</strong> cell(s) hold different caps for "
        f"different walkers, the largest disagreement being "
        f"<strong>{_mxsp[_last]:.0f}</strong> leaves."
        if _ndiff_last else
        f"<code>GB_LEAF_CAP_PER_WALKER</code> is ON in this run, but at "
        f"iteration {_last} every one of the {_cnw} cold walkers still holds "
        f"the SAME cap in all {_cnc} cells &mdash; the per-walker caps are "
        f"running and, so far, buying nothing.")
    CAP_PW_PANEL = f"""<div class="panel">{img("gb_cap_per_walker", "per-walker leaf caps")}
<div class="caption">{CAP_PW_TXT} The cap panels above read the legacy 1-D arrays, which
the run keeps as <em>max-over-walker</em> mirrors &mdash; they show the envelope, never the
spread, so this panel is the only place the feature can be judged. Left is whether the
allowances diverged at all and by how much; middle is each walker's own total allowance,
which is what the enforcement actually gates against; right localises the disagreement in
frequency.</div></div>"""
    del _cw, _spread

# ---- 5b-2. the per-(walker, band) RJ shutoff valve ------------------------
_shut_w = _opt(sub, "gb/band_rj_shutoff_w")
if _shut_w is not None and _shut_w.ndim == 3 and _occ_wb is not None:
    _sw = np.asarray(_shut_w, dtype=bool)                  # (it, nw, nb)
    _n = min(_sw.shape[0], _occ_wb.shape[0])
    _sw, _ow = _sw[:_n], np.asarray(_occ_wb[:_n])
    if _sw.shape[1:] != _ow.shape[1:]:
        # A (walker, band) grid that does not match the census cannot be
        # scored against occupancy, and OCCUPIED is the load-bearing half of
        # the criterion. Say so rather than plotting a wrong denominator.
        MISSING.append(
            f"per-walker RJ shutoff panel skipped: the valve is "
            f"{_sw.shape[1:]} but the cold-chain band census is "
            f"{_ow.shape[1:]}; without a matching occupancy grid the "
            "'fraction of OCCUPIED pairs shut' denominator would be wrong.")
    else:
        _occ = _ow > 0                                     # (it, nw, nb)
        _nocc = _occ.sum(axis=(1, 2)).astype(float)        # occupied pairs
        _nshut = (_sw & _occ).sum(axis=(1, 2)).astype(float)
        _npend = _nocc - _nshut                            # the convergence stat
        _frac = np.divide(_nshut, _nocc, out=np.zeros_like(_nshut),
                          where=_nocc > 0)
        _sit = np.arange(_n)
        # Recipe-step boundaries: the valve is RELEASED at each new step, so
        # the step stamp is what turns a sawtooth into a readable series.
        _step = _opt(sub, "gb/band_shutoff_w_step")
        _steps = None
        _bnds = []
        if _step is not None:
            _steps = np.asarray(_step).reshape(_step.shape[0], -1)[:_n, 0]
            _bnds = [int(i) for i in range(1, _n) if _steps[i] != _steps[i - 1]]

        fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.6),
                               gridspec_kw=dict(wspace=0.42))

        # (a) THE stage-convergence readout. 100% = every occupied pair has
        # shut off = the stage's own stopping criterion is met.
        ax[0].plot(_sit, 100.0 * _frac, color=AMBER, lw=2)
        ax[0].axhline(100.0, color=GREEN, ls=":", lw=1.2)
        ax[0].text(_sit[-1] if _n else 0, 100.0, " stage converged ",
                   color=GREEN, fontsize=8, va="top", ha="right")
        for _b in _bnds:
            ax[0].axvline(_b, color=VIOLET, lw=1.0, alpha=0.7, ls="--")
        # ⚠ TWO DIFFERENT BOUNDARIES on this axis, and they must not be
        # confused. The violet dashes are the valve's own RECIPE STEP serial
        # changing (every step releases the valve wholesale); the grey ones
        # mark where a recipe STAGE completed. They coincide when every stage
        # is one step, and diverge the moment anything else advances a step.
        mark_stages(ax[0], max_it=_sit[-1] if _n else None)
        ax[0].set_ylim(-2, 104)
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("% of occupied (walker, band) pairs")
        ax[0].set_title("RJ valve: shut fraction"
                        + (f" ({len(_bnds)} step boundaries)" if _bnds else ""))

        # (b) the counts behind the fraction. PENDING is the number the
        # recipe's own band_shutoff_w_pending_total returns, and a stage ends
        # when it touches zero.
        ax[1].plot(_sit, _nocc, color=CYAN, lw=1.6, label="occupied pairs")
        ax[1].plot(_sit, _nshut, color=AMBER, lw=2, label="shut off")
        ax[1].plot(_sit, _npend, color=RED, lw=1.6, label="pending (= 0 ends the stage)")
        for _b in _bnds:
            ax[1].axvline(_b, color=VIOLET, lw=1.0, alpha=0.7, ls="--")
        mark_stages(ax[1], label=False, max_it=_sit[-1] if _n else None)
        ax[1].set_xlabel("iteration"); ax[1].set_ylabel("(walker, band) pairs")
        ax[1].legend(fontsize=8, loc="upper left")
        ax[1].set_title("occupied / shut / pending")

        # (c) WHICH bands are holding the stage open, at the last row: per
        # band, how many walkers are occupied there and still unfrozen.
        _nb = _sw.shape[2]
        _pend_b = (_occ[-1] & ~_sw[-1]).sum(axis=0)        # (nb,) walkers
        _shut_b = (_occ[-1] & _sw[-1]).sum(axis=0)
        _fc = 0.5 * (band_edges[:-1] + band_edges[1:])[:_nb] * 1e3
        ax[2].fill_between(_fc, 0, _shut_b, color=AMBER, alpha=0.85, lw=0,
                           step="mid", label="shut")
        ax[2].fill_between(_fc, _shut_b, _shut_b + _pend_b, color=RED,
                           alpha=0.85, lw=0, step="mid", label="pending")
        ax[2].set_xlabel("f0 [mHz]"); ax[2].set_ylabel("occupied walkers in band")
        ax[2].legend(fontsize=8, loc="upper right")
        ax[2].set_title(f"where the stage is still open @ iter {_n - 1}")
        fig_b64(fig, "gb_shutoff_w")

        _step_txt = ""
        if _steps is not None:
            _step_txt = (f" The valve is stamped with recipe step "
                         f"<strong>{int(_steps[-1])}</strong>"
                         + (f" and has been released at {len(_bnds)} step "
                            f"boundary(ies) in this store." if _bnds
                            else ", which has not changed within this store."))
        SHUTOFF_W_TXT = (
            f"<code>GB_SEARCH_BAND_SHUTOFF_PER_WALKER</code> is ON. At "
            f"iteration {_n - 1}, <strong>{_nshut[-1]:.0f} of "
            f"{_nocc[-1]:.0f}</strong> occupied (walker, band) pairs have shut "
            f"off ({100 * _frac[-1]:.0f}%), leaving "
            f"<strong>{_npend[-1]:.0f}</strong> pending." + _step_txt)
        SHUTOFF_W_PANEL = f"""<div class="panel">{img("gb_shutoff_w", "per-walker RJ shutoff valve")}
<div class="caption">{SHUTOFF_W_TXT} A pair freezes when its own cold-chain lnL stops
improving <em>within the current recipe step</em>, and the valve is released wholesale when
the next step begins &mdash; the dashed violet lines. This is additional to the per-band
valve marked in red on the cap panel, and the two compose with OR. The denominator is
OCCUPIED pairs on purpose: an empty band can never shut off, so counting it would put
&ldquo;all pairs shut&rdquo; permanently out of reach. Middle is the literal stage-stopping
criterion &mdash; when the red pending line reaches zero the stage has converged; right says
which bands are holding it open.</div></div>"""
        del _sw, _occ


def gb_gate_valve_settings(log_text, environ):
    """``(tol, conv_iter, tol_from_log, ci_from_log)`` for the valve panel.

    The run's own values come from the NEWEST ``[GB_GATE <move>] occupied
    ...`` summary line, which prints ``streak ... of <conv_iter>`` and
    ``(tol <tol>)``. Without one, the env on the machine building the page
    (``GB_SEARCH_BAND_SHUTOFF_LL_TOL`` default 4.0,
    ``GB_SEARCH_BAND_SHUTOFF_CONV_ITER`` default 3) -- which need not be
    what the run used (the 9mo run gated at tol 20 while the laptop default
    read 4).
    """
    import re as _re

    tol = ci = None
    lines = _re.findall(r"\[GB_GATE [^\]\n]+\] occupied \d+[^\n]*",
                        log_text or "")
    if lines:
        last = lines[-1]
        m = _re.search(r"\(tol ([-+0-9.eEinfa]+)\)", last)
        if m:
            try:
                tol = float(m.group(1))
            except ValueError:
                tol = None
        m = _re.search(r"streak p50 \S+ p90 \S+ of (\d+);", last)
        if m:
            ci = float(m.group(1))
    tol_log, ci_log = tol is not None, ci is not None
    if tol is None:
        tol = float(environ.get("GB_SEARCH_BAND_SHUTOFF_LL_TOL", "4.0"))
    if ci is None:
        ci = float(environ.get("GB_SEARCH_BAND_SHUTOFF_CONV_ITER", "3"))
    return tol, ci, tol_log, ci_log


# ---- 5b-2b. WHY the valve has or has not latched (2026-09-27) ------------
# The shutoff share above says WHETHER pairs are shutting. These three say
# WHY the rest are not, which is the question the 6mo run spent a day
# unable to answer: the statistic the valve judges was computed fresh each
# iteration and thrown away, and the only per-band cold-lnL array in the
# store (``band_cold_ll``) is written by the leaf-CAP gate -- so with caps
# disarmed it is -inf in every cell of every row.
GATE_WHY_PANEL = ""
_cmax = _opt(sub, "gb/band_cold_logl_max_w")
_cnow = _opt(sub, "gb/band_cold_logl_w")
_strk = _opt(sub, "gb/band_shutoff_streak_w")
_rset = _opt(sub, "gb/band_shutoff_reset_w")
if (_cmax is not None and _cnow is not None and _strk is not None
        and _rset is not None and _shut_w is not None and _occ_wb is not None
        and np.ndim(_cmax) == 3):
    try:
        _n2 = min(len(_cmax), len(_cnow), len(_strk), len(_rset),
                  len(_shut_w), len(_occ_wb))
        _mx = np.asarray(_cmax[:_n2], float)
        _nw_ = np.asarray(_cnow[:_n2], float)
        _sk = np.asarray(_strk[:_n2], float)
        _rs = np.asarray(_rset[:_n2], float)
        _sh = np.asarray(_shut_w[:_n2], bool)
        _oc = np.asarray(_occ_wb[:_n2]) > 0
        _last = _n2 - 1
        _act = _oc[_last] & ~_sh[_last]                  # occupied, not shut
        # The RUN's tolerance and clock length, from its newest [GB_GATE]
        # summary line; the env here is the page builder's, not the run's
        # (the 9mo run gated at tol 20 while this read the laptop's 4).
        _tol, _ci, _tol_log, _ci_log = gb_gate_valve_settings(log_text, os.environ)
        _tol_src = ("from the run's [GB_GATE] log line" if _tol_log else
                    "GB_SEARCH_BAND_SHUTOFF_LL_TOL on the page-building machine, "
                    "NOT read from the run (no [GB_GATE] summary line in its logs)")
        _ci_src = ("from the run's [GB_GATE] log line" if _ci_log else
                   "GB_SEARCH_BAND_SHUTOFF_CONV_ITER on the page-building machine, "
                   "NOT read from the run")
        if not _act.any():
            raise ValueError("no active occupied pairs in the last row")

        fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.6),
                               gridspec_kw=dict(wspace=0.40))

        # (a) HOW FAR each still-active pair is from its own all-time best.
        # A pair converges when this stays above -tol for conv_iter
        # iterations; a pair sitting near zero is one that keeps matching
        # its own record, which is what a chronic resetter looks like.
        _gap = (_mx[_last] - _nw_[_last])[_act]
        _gap = _gap[np.isfinite(_gap)]
        if _gap.size:
            ax[0].hist(np.clip(_gap, -2.0 * _tol, 6.0 * _tol), bins=50,
                       color=AMBER, alpha=0.9)
            ax[0].axvline(_tol, color=RED, ls="--", lw=1.4)
            ax[0].text(_tol, ax[0].get_ylim()[1],
                       f" tol {_tol:g} ({'run log' if _tol_log else 'local env, not the run'}) ",
                       color=RED, fontsize=8, va="top")
            ax[0].axvline(0.0, color=FG, ls=":", lw=1.0)
        ax[0].set_xlabel("all-time max  -  this iteration  [lnL]")
        ax[0].set_ylabel("active (walker, sub-band) pairs")
        ax[0].set_title("distance from each pair's own record", fontsize=10)

        # (b) THE PATIENCE CLOCK. Everything at conv_iter-1 is one quiet
        # iteration from shutting; a pile at 0 is the resetting population.
        _skv = _sk[_last][_act]
        ax[1].hist(_skv, bins=np.arange(0, max(_skv.max(), _ci) + 2) - 0.5,
                   color=CYAN, alpha=0.9)
        ax[1].axvline(_ci, color=GREEN, ls="--", lw=1.4)
        ax[1].text(_ci, ax[1].get_ylim()[1],
                   f" shuts at {_ci:g} ({'run log' if _ci_log else 'local env'}) ",
                   color=GREEN, fontsize=8, va="top", ha="right")
        ax[1].set_xlabel("consecutive non-improving iterations")
        ax[1].set_ylabel("active pairs")
        ax[1].set_title("patience clock", fontsize=10)

        # (c) THE THREE POPULATIONS over time. Streak alone cannot tell a
        # chronic resetter (high reset count, streak near 0) from a pair
        # that was simply occupied last iteration (both zero).
        _ch, _yo, _fr = [], [], []
        for _i in range(_n2):
            _a = _oc[_i] & ~_sh[_i]
            if not _a.any():
                _ch.append(0); _yo.append(0); _fr.append(0); continue
            _r, _k = _rs[_i][_a], _sk[_i][_a]
            _ch.append(int(((_r > 0) & (_k < _ci)).sum()))
            _yo.append(int(((_r == 0) & (_k > 0)).sum()))
            _fr.append(int(((_r == 0) & (_k == 0)).sum()))
        _x2 = np.arange(_n2)
        ax[2].stackplot(_x2, _ch, _yo, _fr,
                        labels=["chronic resetter", "still young",
                                "newly occupied"],
                        colors=[RED, AMBER, DIM], alpha=0.9)
        ax[2].legend(fontsize=7, loc="upper left")
        ax[2].set_xlabel("stored iteration")
        ax[2].set_ylabel("active pairs")
        ax[2].set_title("why each active pair is still open", fontsize=10)
        fig_b64(fig, "gate_why")

        GATE_WHY_PANEL = f"""<div class="panel">{img("gate_why", "why the valve has not latched")}
<div class="caption">The panel above says WHETHER pairs are shutting; these say
WHY the rest are not. <b>Left:</b> how far each still-active pair sits below its
own all-time cold-chain record. A pair only advances its patience clock while
this stays above the tolerance (dashed red); a pile near zero is a pair that
keeps matching its own record. <b>Middle:</b> the patience clock itself &mdash;
anything one short of the green line shuts on its next quiet iteration.
<b>Right:</b> the three reasons a pair can still be open, which the streak alone
cannot separate: a <em>chronic resetter</em> keeps beating its record, one that is
<em>still young</em> simply has not had enough consecutive quiet iterations yet,
and a <em>newly occupied</em> pair started its clock this iteration. A run that is
converging shows the red band shrinking; a run whose gate is mis-specified shows
it flat. The tolerance ({_tol:g} lnL) is {_tol_src}; the clock length ({_ci:g}
iterations) is {_ci_src}.</div></div>"""
    except Exception as _e:   # noqa: BLE001
        MISSING.append(f"gate-diagnosis panel unavailable: {_e!r}")
elif _shut_w is not None:
    MISSING.append(
        "gate-diagnosis panel skipped: this store predates the "
        "band_cold_logl_max_w / _w / streak / reset telemetry (added "
        "2026-09-27). Run scripts/fstat_proposal/migrate_gb_shutoff_persist.py "
        "with the job down to add them; the run does not need it to keep "
        "going, but nothing about the gate is recorded without it.")

# ---- 5b-3. the per-(walker, band) search STAGE (coarse -> fine) -----------
_stage_w = _opt(sub, "gb/band_stage_w")
if _stage_w is not None and _stage_w.ndim == 3:
    _st = np.asarray(_stage_w, dtype=float)                # (it, nw, nb)
    _n = _st.shape[0]
    _stit = np.arange(_n)
    _fine = _st >= 1
    if _occ_wb is not None and _occ_wb.shape[1:] == _st.shape[1:]:
        _m = min(_n, _occ_wb.shape[0])
        _occ_s = np.asarray(_occ_wb[:_m]) > 0
        _st, _fine, _n, _stit = _st[:_m], _fine[:_m], _m, _stit[:_m]
        _den = _occ_s.sum(axis=(1, 2)).astype(float)
        _num = (_fine & _occ_s).sum(axis=(1, 2)).astype(float)
        _den_lab = "occupied (walker, band) pairs"
    else:
        # No census to pair with -- fall back to ALL pairs and label it, so
        # the number is never silently a different denominator than 5b-2's.
        _den = np.full(_n, float(_st.shape[1] * _st.shape[2]))
        _num = _fine.sum(axis=(1, 2)).astype(float)
        _den_lab = "all (walker, band) pairs"
    _fr = np.divide(_num, _den, out=np.zeros_like(_num), where=_den > 0)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.4))
    ax[0].plot(_stit, 100.0 * _fr, color=GREEN, lw=2)
    ax[0].set_ylim(-2, 104)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(f"% of {_den_lab}")
    ax[0].set_title("promoted to FINE (stage 1)"
                    + ("" if _num[-1] else " -- none yet"))
    # Per band, the fraction of walkers at FINE. The stage is a ONE-WAY latch,
    # so this image can only ever fill in, and where it fills first is where
    # the search settled first.
    _im = ax[1].imshow(_fine.mean(axis=1).T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=1,
                       extent=[0, _n, 0, _st.shape[2]])
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("band")
    ax[1].set_title("fraction of walkers at FINE, per band")
    fig.colorbar(_im, ax=ax[1], shrink=0.85)
    fig_b64(fig, "gb_search_stage")

    STAGE_W_TXT = (
        f"<code>GB_SEARCH_STAGE_PER_WALKER</code> is ON. At iteration "
        f"{_n - 1}, <strong>{_num[-1]:.0f} of {_den[-1]:.0f}</strong> "
        f"{_den_lab} ({100 * _fr[-1]:.0f}%) have promoted from the COARSE "
        f"floor to the FINE one.")
    STAGE_W_PANEL = f"""<div class="panel">{img("gb_search_stage", "per-walker search stage")}
<div class="caption">{STAGE_W_TXT} Stage 0 (COARSE) runs the run's starting opt-SNR and
F-stat floors &mdash; hunt bright sources, do not poison the model with noise births. A
(walker, band) pair promotes to stage 1 (FINE, relaxed floors, dig the faint tail) once its
own cold-chain source count has sat unchanged for the configured number of iterations, and
an empty band never promotes. The latch is <em>one-way</em>, so the right-hand image can
only fill in: where it fills first is where the search settled first.</div></div>"""
    del _st, _fine

if _occ_wb is not None:
    del _occ_wb

# One injection point, and it is CONCATENATED onto the previous panel's line
# in the HTML rather than given lines of its own: with all three features off
# (every store written before 2026-09-24) this is the empty string, and the
# page that comes out is then BYTE-IDENTICAL to one built before this section
# existed -- which is the property that matters most here, since almost every
# store on disk predates the features. Verified with ``cmp`` against
# ``git show HEAD:...`` on gf_prod_3mo_v7.
PER_WALKER_PANELS = "\n".join(
    p for p in (CAP_PW_PANEL, SHUTOFF_W_PANEL, STAGE_W_PANEL) if p)
if PER_WALKER_PANELS:
    PER_WALKER_PANELS = "\n" + PER_WALKER_PANELS

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
if CENSUS is None:
    # SAY SO (2026-08-22). This used to degrade to a bare "plot unavailable"
    # placeholder with nothing in MISSING, so the page gave no reason why the
    # one panel whose RED layer means "detectable but NOT recovered" had
    # disappeared. The npz is written by census.py into a scratch directory
    # that does not survive a session, so its absence is the normal case.
    MISSING.append(
        "high-frequency recovery census skipped: no gb_hi_f_census.npz "
        "beside the store or in the CWD. It is produced by census.py (a "
        "scratch-directory script), not by this generator, so a fresh "
        "scratchpad loses it. The red 'missed' scatter, the red 'N "
        "detectable' line and the red 'has detectable source' curve all "
        "come from that file.")
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
# ⚠ THE POOL DIRECTORY IS NOT ALWAYS CALLED "shared". Production writes
# per-stage pools -- 6mo v9 uses ``gb_fstat_fit/shared_search`` -- so the
# hard-coded "shared" resolved to a path that does not exist on the
# cluster and the comb / peaks panels have been silently missing from
# every cluster-built page. Take any ``gb_fstat_fit/*`` pool, newest
# usable one wins, with the legacy name still preferred when present so
# an older run renders exactly as before.
_froot = os.path.join(RUN_DIR, "gb_fstat_fit")
_pools = []
if os.path.isdir(_froot):
    for _d in sorted(os.listdir(_froot)):
        _p = os.path.join(_froot, _d)
        if os.path.isdir(_p):
            _pools.append(_p)
_legacy = os.path.join(_froot, "shared")
if _legacy in _pools:
    _pools.remove(_legacy)
    _pools.insert(0, _legacy)


def _complete_epochs(d):
    """Epoch dirs under ``d`` carrying BOTH cache files."""
    if not os.path.isdir(d):
        return []
    return [e for e in sorted(os.listdir(d))
            if os.path.exists(os.path.join(d, e, "fstat_grid_comb.npz"))
            and os.path.exists(
                os.path.join(d, e, "fstat_grid_peaks_stacked.npz"))]


fdir, epochs = _legacy, []
for _p in _pools:
    _e = _complete_epochs(_p)
    if _e:
        fdir, epochs = _p, _e
        break
if not epochs and _pools:
    print(f"# WARNING: gb_fstat_fit pools {[os.path.basename(p) for p in _pools]}"
          f" hold no COMPLETE epoch (both fstat_grid_comb.npz and "
          f"fstat_grid_peaks_stacked.npz)", file=sys.stderr)
# (COMPLETE epochs only -- 2026-08-15 -- is enforced by _complete_epochs
# above: an epoch directory exists as soon as the fit STARTS, and the
# 23-mo comb alone is a 1.19-billion-evaluation sweep, so a snapshot very
# often catches a half-written epoch. Requiring both cache files keeps a
# mid-fit run from killing the whole page.)
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
        "No COMPLETE fstat epoch cache under gb_fstat_fit/* -- either the "
        "grid fit is still running (its epoch dir appears at fit start, "
        "the comb/peaks caches only when it finishes), or this page was "
        "built from a SNAPSHOT: the tar drops everything under "
        "gb_fstat_fit except DONE.json unless it was built with "
        "include_fstat.")

# ---- 7. VGB ----
# Matches the sampled columns in the same order the store writes them, so
# vgb_last[..., j] always corresponds to VGB_NAMES[j].
VGB_NAMES = (
    ["dist [kpc]", "phi0", "cos_iota", "psi", "Mc [Msol]", "fdot_astro_ratio"]
    if VGB_SAMPLED_CHIRP else
    ["dist [kpc]", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
)
# Per-leaf FIXED frequencies + names from the mojito catalogue (leaf i =
# catalogue row i, fixed-leaf branch).
VGB_F0 = None
VGB_IDS = None

# The monitor rebuilds the data / template / residual stack on the fly and
# reads the VGB, GB and WDWD catalogues on the way; both need a "brick base
# dir" that contains ``data/{GB,VGB,COMBINED,...}/L1/*.h5`` and
# ``catalogues/wdwd_cat_mojito_lite_processed.hdf5``. That directory lives
# somewhere different on every machine, so it must be discoverable via env
# rather than hard-coded.
#
# Chain: MOJITO_INFO_PATH > MOJITO_CAT > MOJITO_CACHE_DIR > laptop default.
# MOJITO_INFO_PATH is THE knob (see _resolve_mojito_info_path); the two
# below it are kept so runbooks that predate it keep rendering, and they
# still match build_truth.py's own ``_resolve_catalogue()`` order
# (--catalogue > MOJITO_CAT > MOJITO_CACHE_DIR > default).
#
# What the default costs when it misses: cluster runs (bricks under
# /shared/data/mojito_cache) fell through to the laptop path and silently
# dropped the residual-spectrum and data/template/residual panels.
def _resolve_mojito_cat_dir():
    # MOJITO_INFO_PATH first (user ruling 2026-09-26): it is already
    # validated to hold BOTH catalogues/ and data/, which is exactly this
    # function's contract, so nothing below can improve on it.
    if MOJITO_INFO_PATH:
        return MOJITO_INFO_PATH
    v = os.environ.get("MOJITO_CAT")
    if v:
        v = os.path.expanduser(v)
        # A file argument (typically .../catalogues/wdwd_cat_*.hdf5) implies
        # the brick base dir is two levels up.
        if os.path.isfile(v):
            return os.path.dirname(os.path.dirname(v))
        if os.path.isdir(v):
            return v
    v = os.environ.get("MOJITO_CACHE_DIR")
    if v:
        v = os.path.expanduser(v)
        # build_truth.py's convention nests the bricks under
        # brickmarket/mojito_light_v1_0_0/; a cluster layout may store them
        # flat under the cache root. Prefer the nested layout when present,
        # else fall through to the flat cache.
        nested = os.path.join(v, "brickmarket", "mojito_light_v1_0_0")
        return nested if os.path.isdir(nested) else v
    return os.path.expanduser(
        "~/.mojito_cache/brickmarket/mojito_light_v1_0_0")


MOJITO_CAT_DIR = _resolve_mojito_cat_dir()


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


VGB_TRUTH = None        # (55, ndim) in the VGB sampled basis (5 or 6 cols)
VGB_TRUTH_REL = None
try:
    from lisatools.globalfit.stock.erebor.vgb import load_vgb_catalogue_file
    _cat = load_vgb_catalogue_file(MOJITO_CAT_DIR)
    _v = np.asarray(_cat["vgb"]).item()
    VGB_F0 = np.asarray(_v["GW22FrequencySSBFrame"]) * 1e3   # mHz
    VGB_IDS = [i.decode() if isinstance(i, bytes) else str(i)
               for i in _v["ID"]]
    # Legacy VGB sampled basis (5 params) = ["dist", "phi0", "cos_iota",
    # "psi", "fdot_astro_ratio"] = 9-col GB indices [0, 3, 4, 5, 8].
    # Chirp basis (6 params, 6-month runs) = ["dist", "phi0", "cos_iota",
    # "psi", "Mc", "fdot_astro_ratio"] = 9-col GB indices
    # [0, 3, 4, 5, 2, 8] -- Mc slot goes IN THE MIDDLE, matching the store
    # column order in ``vgb_c[..., 4]``.
    _r9, VGB_TRUTH_REL = cat_to_sampled9(_v)
    _truth_cols = ([0, 3, 4, 5, 2, 8] if VGB_SAMPLED_CHIRP
                   else [0, 3, 4, 5, 8])
    VGB_TRUTH = _r9[:, _truth_cols]
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


# ---- TIME-FREQUENCY IMAGES: POOL TO THE PIXEL GRID (2026-10-11) -----------
# An MBHB merger is ONE WDM time column (3600 s). The 9mo panel used to
# max-pool 5 columns per drawn column (6361 -> 1272) and hand that to a
# ~347 px wide axes, so matplotlib's own resampling kept one column in ~3.7
# and dropped the rest -- a different subset in every panel. The 102 d
# merger (SNR ~1010) was bright in the data panel and faint in the
# fitted-signals panel, while 109 d and 171 d showed the other way round:
# "spikes in the wrong panels", raster aliasing rather than data. The fix is
# to pool to AT MOST ONE COLUMN PER SCREEN PIXEL with MAX, so every drawn
# column is drawn and a one-column transient survives in every panel. These
# helpers are display only; nothing here touches a likelihood.


def tf_pool_block(n, target):
    """Contiguous block length that pools ``n`` samples to at most ``target``.

    ``ceil(n / target)``, and 1 (no pooling) when ``target >= n``.
    """
    n, target = int(n), int(target)
    if target < 1:
        raise ValueError(f"pooling target must be >= 1, got {target}")
    return max(1, -(-n // target))


def pool_tf_for_display(abs_w, n_cols, n_rows=None, how="max"):
    """Pool a ``(..., layers, time)`` ``|w|`` array down to a display size.

    The time axis (last) is cut into contiguous blocks of
    ``tf_pool_block(Nt, n_cols)`` columns, so the result has at most
    ``n_cols`` columns; likewise the layer axis (second to last) when
    ``n_rows`` is given. Each block is reduced with ``how``: ``"max"``
    (default) keeps a one-column transient -- an MBHB merger -- at full
    height in whichever block holds it; ``"mean"`` dilutes it by the block
    length. The last block may be RAGGED (shorter): it is reduced over the
    samples it has. NaNs are ignored; an all-NaN block gives NaN. A target
    at least as large as the axis leaves that axis untouched. Leading axes
    (channels) pass through.
    """
    if how not in ("max", "mean"):
        raise ValueError(f"how must be 'max' or 'mean', got {how!r}")
    a = np.asarray(abs_w)
    if a.ndim < 2:
        raise ValueError(f"need a (..., layers, time) array, got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(float)

    def _pool_last(x, target):
        n = x.shape[-1]
        b = tf_pool_block(n, target)
        if b == 1:
            return x
        nout = -(-n // b)
        pad = nout * b - n
        if pad:     # ragged last block: NaN padding, ignored by the reductions
            x = np.concatenate(
                [x, np.full(x.shape[:-1] + (pad,), np.nan, dtype=x.dtype)], axis=-1)
        x = x.reshape(x.shape[:-1] + (nout, b))
        if how == "max":
            return np.fmax.reduce(x, axis=-1)       # NaN-ignoring, no warning
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(x, axis=-1) / np.sum(~np.isnan(x), axis=-1)

    out = _pool_last(a, n_cols)
    if n_rows is not None:
        out = np.swapaxes(_pool_last(np.swapaxes(out, -1, -2), n_rows), -1, -2)
    return out


def tf_display_grid(axes, dpi):
    """``(n_cols, n_rows)``: the largest image that maps <= 1 column (row) per
    screen pixel in EVERY one of ``axes`` when the figure is saved at ``dpi``.

    Read from the axes positions, so call it after the layout is final
    (``tight_layout``, colorbars). One pixel is kept in hand for Agg's
    snapping of the image box to whole pixels. ``bbox_inches="tight"`` only
    crops the canvas; it never rescales it, so the positions hold.
    """
    axes = list(axes)
    fw, fh = axes[0].figure.get_size_inches()
    w = min(a.get_position().width for a in axes) * fw * float(dpi)
    h = min(a.get_position().height for a in axes) * fh * float(dpi)
    return max(int(np.floor(w)) - 1, 1), max(int(np.floor(h)) - 1, 1)


#: sampling-basis column of ``t_plunge`` in ``chain/mbh``
#: (``stock.erebor.transforms.make_mbh_transform_container``: the last of 11)
MBH_T_PLUNGE_COL = 10


def mbh_merger_plot_days(chain_row, inds_row, t_offset, col=MBH_T_PLUNGE_COL):
    """Merger times of the MBH leaves on a time axis, in days, sorted.

    ``chain_row`` is ONE stored row of the MBH cold chain,
    ``(nwalkers, nleaves, ndim)``; ``inds_row`` its ``(nwalkers, nleaves)``
    mask (``None`` = all live). Per leaf, the median ``t_plunge`` over its
    live walkers (the walker spread is seconds; a drawn column is hours),
    minus ``t_offset`` seconds -- the axis origin in the ``t_plunge`` frame.
    Leaves with no live walker are skipped.
    """
    c = np.asarray(chain_row, dtype=float)
    if c.ndim != 3 or c.shape[1] == 0 or c.shape[2] <= col:
        return []
    t = c[..., col]
    live = (np.ones(t.shape, bool) if inds_row is None
            else np.asarray(inds_row, dtype=bool))
    out = []
    for j in range(t.shape[1]):
        v = t[:, j][live[:, j] & np.isfinite(t[:, j])]
        if v.size:
            out.append((float(np.median(v)) - float(t_offset)) / 86400.0)
    return sorted(out)


def draw_wdm_tf_panels(sources, *, layer_dt, layer_df, t_edge0, f_edge0, nt, nf,
                       col_titles, row_names=("X", "Y", "Z"), suptitle="",
                       mergers_d=(), merger_src="", dpi=88, figsize=(13.5, 8.6)):
    """The data | fitted signals | residual time-frequency figure.

    ``sources`` is three ``(key, thunk)`` pairs in column order; each thunk
    returns that column's full-resolution ``|w|``, ``(channels, nf, nt)``.
    The axes are laid out FIRST, so the pixel grid is known before any
    array is built; each array is then max-pooled to that grid
    (:func:`pool_tf_for_display`, the SAME target for all three columns) the
    moment it exists, and the full-resolution map is dropped. Drawn with
    ``interpolation="nearest"`` on at most one pooled column per pixel, so
    no column is skipped and a one-column merger shows in every panel.

    Colour: ONE linear scale per channel row, the 99.5th percentile of that
    row's POOLED data panel, shared across the row. Max pooling lifts the
    apparent noise floor a little against an unpooled view; the subtitle
    says so. The time axis is days from ``t_edge0`` (seconds, the first
    active edge); ``mergers_d`` (days, same origin) are drawn as short white
    ticks on the top and bottom edge of every panel, off the image's middle
    so they never cover the feature they mark.

    ``figsize`` height 8.6 in (8.0 before 2026-10-11) gives each row ~185 px
    at 88 dpi, so the 179-layer grid is drawn at one layer per pixel or
    better (``bf == 1``) instead of being pooled 2x in frequency.

    Returns ``(fig, info)``; ``info`` holds the pooled grid and blocks.
    """
    import matplotlib.patheffects as _pe

    fig, ax = plt.subplots(len(row_names), len(sources), figsize=figsize,
                           sharex=True, sharey=True, squeeze=False)
    ims, cbs = {}, {}
    nr, nc = ax.shape
    for r in range(nr):
        for c in range(nc):
            a_ = ax[r][c]
            # placeholder: the real (pooled) image needs the final layout
            ims[r, c] = a_.imshow(np.zeros((1, 1)), aspect="auto", origin="lower",
                                  extent=[0.0, 1.0, 0.0, 1.0], vmin=0.0, vmax=1.0,
                                  cmap="viridis", interpolation="nearest")
            a_.grid(False)
            if r == 0:
                a_.set_title(col_titles[c], fontsize=10)
            if c == 0:
                a_.set_ylabel(f"{row_names[r]}\nfrequency [mHz]", fontsize=9)
            if r == nr - 1:
                # the axis starts at the first ACTIVE edge, not at the data
                # start: the edge crop (min_time) sits in between
                a_.set_xlabel(f"time [d], 0 = data start + {t_edge0 / 86400.0:.3g} d "
                              f"(edge crop)", fontsize=9)
            if c == nc - 1:
                cbs[r] = fig.colorbar(ims[r, c], ax=a_, fraction=0.046, pad=0.02)
                cbs[r].ax.tick_params(labelsize=7)
                cbs[r].set_label(f"|w| / {1.0:.2e}", fontsize=7, color=DIM)
    fig.suptitle(suptitle, fontsize=10, color=FG, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    n_cols, n_rows = tf_display_grid(ax.ravel(), dpi)
    bt, bf = tf_pool_block(nt, n_cols), tf_pool_block(nf, n_rows)
    pooled = {}
    for key, thunk in sources:
        m = thunk()
        if tuple(m.shape[-2:]) != (int(nf), int(nt)):
            raise ValueError(f"{key}: |w| shape {m.shape} != (.., {nf}, {nt})")
        pooled[key] = pool_tf_for_display(m, n_cols, n_rows, how="max")
        del m
    keys = [k for k, _ in sources]
    pshape = pooled[keys[0]].shape
    # EXACT cell edges: every drawn cell spans bt pixels (bf layers), the
    # ragged last one included, so a merger tick lands on its own cell. The
    # axis may run up to one cell past the analysed span; nothing is cut.
    ext = [0.0, pshape[-1] * bt * layer_dt / 86400.0,
           f_edge0 * 1e3, (f_edge0 + pshape[-2] * bf * layer_df) * 1e3]
    vmaxes = []
    for r in range(nr):
        vmax = float(np.nanpercentile(pooled[keys[0]][r], 99.5))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        vmaxes.append(vmax)
        for c, key in enumerate(keys):
            ims[r, c].set_data(pooled[key][r] / vmax)
            ims[r, c].set_extent(ext)
        cbs[r].set_label(f"|w| / {vmax:.2e}", fontsize=7, color=DIM)
    ax[0][0].set_xlim(ext[0], ext[1])
    ax[0][0].set_ylim(ext[2], ext[3])

    drawn = [float(t) for t in mergers_d if ext[0] <= float(t) <= ext[1]]
    _halo = [_pe.Stroke(linewidth=2.8, foreground="#0A0E14"), _pe.Normal()]
    for a_ in ax.ravel():
        _tx = a_.get_xaxis_transform()
        for t in drawn:
            for y0, y1 in ((0.0, 0.07), (0.93, 1.0)):
                a_.plot([t, t], [y0, y1], transform=_tx, color="white", lw=1.3,
                        solid_capstyle="butt", path_effects=_halo, zorder=5)
    sub = (f"max-pooled to the pixel grid: one drawn column = {bt} x "
           f"{layer_dt:.0f} s" + (f", one drawn row = {bf} layers" if bf > 1 else "")
           + " (MAX keeps one-pixel transients in every panel; the noise floor "
           "reads slightly higher)")
    if drawn:
        sub += (f"\nwhite edge ticks = the {len(drawn)} MBHB merger times"
                + (f" ({merger_src})" if merger_src else ""))
    fig.text(0.5, 0.973, sub, ha="center", va="top", fontsize=8, color=DIM)
    info = dict(n_cols=int(n_cols), n_rows=int(n_rows), bt=int(bt), bf=int(bf),
                pooled_shape=tuple(int(s) for s in pshape), vmax=vmaxes,
                mergers_d=drawn, extent=ext)
    return fig, info


def snapshot_wdm_to_fd(arr, wdm, full, fds, channels=None):
    """One residual-snapshot array ``(3, Nf_active, Nt_active)`` -> FD.

    Zero-pads the active band and span into the full ``full`` WDM grid and
    inverts with ``WDMSignal.wdm_to_fd`` onto ``fds``: band-limited to the
    analysed band and the edge-cropped span, which is all the likelihood
    sees. ``channels`` (a slice) converts only those channels -- the
    X-only call (``slice(0, 1)``) builds a third of the transform's
    full-length temporaries. Returns ``(nchannels, fds.N)`` complex.
    """
    from lisatools.domains import WDMSignal

    a = np.asarray(arr)
    if channels is not None:
        a = a[channels]
    pad = np.zeros((a.shape[0], int(full.Nf), int(full.Nt)))
    pad[:, wdm.active_slice_f, wdm.active_slice_t] = a
    return np.asarray(WDMSignal(pad, full).wdm_to_fd(settings=fds).arr)


def load_residual_snapshot(path, keep_walkers=None):
    """The page's view of a ``*_residual_snapshot.npz`` sidecar.

    Always: ``data``, ``residual`` (the snapshot's max-lnL walker),
    ``iteration``, ``walker`` and ``row`` (the STORE row; ``None`` for
    sidecars before 2026-10-08, which carry only the job's own iteration
    count, restarting on every resume). Sidecars from 2026-10-11 on also
    carry every cold walker: ``residuals`` + ``walkers`` are then set, cut
    to ``keep_walkers`` (cold walker indices; ``None`` keeps all) so the
    page holds only the rows it draws. An older sidecar simply lacks them.
    """
    with np.load(path) as z:
        snap = {k: z[k] for k in ("data", "residual", "iteration", "walker")}
        snap["row"] = int(z["store_row"]) if "store_row" in z.files else None
        if "residuals" in z.files and "walkers" in z.files:
            ws_all = [int(v) for v in np.asarray(z["walkers"]).ravel()]
            want = (set(ws_all) if keep_walkers is None
                    else {int(v) for v in keep_walkers})
            ix = [i for i, v in enumerate(ws_all) if v in want]
            rs_all = z["residuals"]
            snap["residuals"] = rs_all[ix]          # fancy index: a copy
            snap["walkers"] = [ws_all[i] for i in ix]
            del rs_all
    return snap


#: the three ways the page can build a walker's residual, best first (see
#: :func:`walker_residual_x`); these short tags go in figure legends, the
#: full label on the figure's title lines
RESID_KIND_SHORT = {
    "run": "run's residual",
    "swap": "GB+VGB swapped in",
    "gbvgb": "GB+VGB only",
}


def walker_residual_x(w, wbest, snap, page_resid_x, tmpl_x, snap_fd_x, data_x):
    """The X-channel FD residual that stands for cold walker ``w`` on the
    "whitened residual per GB band" panels, and how it was built.

    Returns ``(r_x, kind, label)``, ``kind`` one of (best first):

    ``"run"``    the run's own residual for ``w``: the snapshot's
                 ``residuals`` row for ``w`` (sidecars written from
                 2026-10-11 on), or its ``residual`` when ``w`` IS the
                 snapshot's walker;
    ``"swap"``   an OLD single-walker snapshot of walker ``s != w``:
                 ``r_s + T(s) - T(w)``, i.e. ``s``'s residual with ``w``'s GB
                 + VGB templates swapped in for ``s``'s. Every other class
                 (MBH, EMRI, SOBBH) is still ``s``'s fit, not ``w``'s;
    ``"gbvgb"``  no snapshot: the data minus ``w``'s GB + VGB templates only,
                 so every MBH / EMRI / SOBBH in the data is still in it.

    Before 2026-10-11 every walker but the snapshot's own got "data minus
    its GB + VGB" even WITH a snapshot, so the min-lnL curve carried every
    MBH / EMRI / SOBBH (a 5-7x "excess" below 1.5 mHz on the 9mo page).

    ``snap`` is ``None`` or the page's snapshot dict (``walker``; optionally
    ``residuals`` + ``walkers``). ``page_resid_x`` is the page's own residual
    ``resid_fd[0]``: the snapshot walker's when ``snap`` is set, else
    ``data_x - tmpl_x(wbest)``. ``tmpl_x(v)`` returns a NEW array, cold walker
    ``v``'s X-channel GB + VGB template sum at the last stored row;
    ``snap_fd_x(a)`` turns one ``(3, Nf, Nt)`` snapshot array into its
    X-channel FD series. Each is called only when its construction needs it,
    and besides the result at most one extra full-length series is alive
    here at a time.
    """
    w = int(w)
    if snap is None:
        r = page_resid_x if w == int(wbest) else data_x - tmpl_x(w)
        return r, "gbvgb", ("GB+VGB subtraction only (MBH / EMRI / SOBBH not "
                            "subtracted)")
    s = int(snap["walker"])
    if w == s:
        return page_resid_x, "run", "run's own residual"
    ws, rs = snap.get("walkers"), snap.get("residuals")
    if ws is not None and rs is not None:
        ws = [int(v) for v in np.asarray(ws).ravel()]
        if w in ws:
            return snap_fd_x(rs[ws.index(w)]), "run", "run's own residual"
    r = page_resid_x + tmpl_x(s)
    r -= tmpl_x(w)
    return r, "swap", (f"snapshot residual of walker {s} with this walker's "
                       f"GB+VGB templates swapped in (other classes from "
                       f"walker {s})")


DTR = {}
FGW = {}        # whitened-residual-per-band readout (filled inside the block below)
FG_RESID_SRC = ""   # one sentence: how the whitened panels' residuals were built
try:
    import glob as _glob
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
    from lisatools.detector import L1Orbits
    from lisatools.domains import (FDSettings, FDSignal, TDSettings, TDSignal,
                                   WDMSettings, WDMSignal)
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

    # DISPLAY TAPER: fill the edge crop, do not inherit the run's data taper.
    #
    # The run's taper is deliberately tiny (14400 s of 1.5552e7 = alpha
    # 1.85e-3) and that is CORRECT for the likelihood, which works in the WDM
    # domain and throws the tapered edges away wholesale via
    # EDGE_CROP_WAVELETS. There is no global FFT there and so no global
    # leakage. This panel is a different representation: one FFT over the
    # whole span, where a 2-wavelet taper leaks catastrophically. Root-caused
    # 2026-08-18: the smooth lobes below ~1.4 mHz are the taper's own
    # transfer function convolved with the enormous sub-mHz TDI content
    # (highpass is off), with measured null spacing 0.1366 mHz = 1/7320 s
    # against 1/7200 s for the taper ramp. They are a picture of the window,
    # not of the data, and they drove the residual/model ratio in the lower
    # panel to 10-40x below 0.8 mHz.
    #
    # The cure obeys the standing EDGE-EXCLUSION INVARIANT (2026-08-19): any
    # region where taper error is created must be one the crop removes. So
    # the display taper is set to the crop duration itself -- it lives
    # entirely inside [t_start, min_time] and [max_time, t_end], which the
    # likelihood already ignores, leaving the analysed span untouched while
    # lengthening the ramp ~15x and collapsing the leakage.
    _t_span = N_TD * W_DT
    try:
        _crop_lo = float(_k["min_time"])
        _crop_hi = _t_span - float(_k["max_time"])
        _taper_sec = max(min(_crop_lo, _crop_hi), 0.0)
    except Exception:
        _taper_sec = 0.0
    WIN_ALPHA_DISP = (2.0 * _taper_sec / _t_span) if _taper_sec > 0 else WIN_ALPHA
    if WIN_ALPHA_DISP <= WIN_ALPHA:
        WIN_ALPHA_DISP = WIN_ALPHA
        MISSING.append(
            "residual-spectrum display taper fell back to the run's own "
            f"alpha ({WIN_ALPHA:.4g}): no usable edge crop in the stored "
            "domain settings. Expect window-leakage lobes below ~1.4 mHz; "
            "they are the window, not the data.")
    WIN_TAPER_NOTE = (
        f"display taper {_taper_sec:.0f} s per side (alpha "
        f"{WIN_ALPHA_DISP:.4g}), sized to the edge crop the likelihood "
        f"already discards; the run's own data taper is alpha {WIN_ALPHA:.4g}")
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
    # THE RUN'S OWN DATA STREAM (2026-10-07). A run that lists COMBINED in
    # its source_types read mojito's pre-summed brick (noise + every source
    # class) and nothing else, so that is the data here too. Otherwise the
    # bricks of the run's own source_ids are summed. Until 2026-10-07 this
    # summed the FIRST local brick of each type instead -- an arbitrary
    # source (on the laptop cache MBHB 16, merging at 111 d), so the page
    # showed a "residual" MBH merger the run had in fact fitted.
    from lisatools.globalfit.preprocessing import find_combined_file, find_file
    _stypes = re.search(r"'source_types':\s*\[([^\]]*)\]", _pk)
    _stypes = re.findall(r"'(\w+)'", _stypes.group(1)) if _stypes else []
    _files = {}
    if "COMBINED" in _stypes:
        try:
            _files["COMBINED"] = find_combined_file(
                os.path.join(MOJITO_CAT_DIR, "data", "COMBINED", "L1"))
            _types = ["COMBINED"]
        except (FileNotFoundError, OSError) as _exc:
            MISSING.append(
                f"the run read mojito's COMBINED stream, which is not in this "
                f"machine's cache ({_exc}); the data here is the sum of the "
                f"run's own per-type bricks that ARE cached instead.")
    if not _files:
        for _t in _types:
            _folder = os.path.join(MOJITO_CAT_DIR, "data", _SUBDIR.get(_t, _t), "L1")
            _ids = re.search(rf"'{_t}':\s*\[([^\]]*)\]", _pk)
            _ids = ([int(x) for x in re.findall(r"\d+", _ids.group(1))]
                    if _ids else [0])
            for _sid in _ids:
                try:
                    _files[f"{_t}{_sid}" if len(_ids) > 1 else _t] = find_file(
                        _folder, _t, _sid)
                except (FileNotFoundError, OSError):
                    MISSING.append(f"mojito {_t} source {_sid} brick not in this "
                                   f"machine's cache; left out of the data.")
        if not _files:
            raise FileNotFoundError("no local mojito L1 brick for this run")

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

    _orb = _WindowedL1Orbits(next(iter(_files.values())), force_backend="cpu",
                             frame="icrs", linear_interp_dt=500.0)
    _orb._ensure_configured()
    # stash for the SNR machinery below: the analytic DefaultOrbits ephemeris
    # differs from the mojito L1 orbits by the FULL annual-Doppler phase at
    # the run epoch (+-16 FD bins at 20 mHz) -- high-f SNRs/overlaps computed
    # on DefaultOrbits are wrong (2026-08-19 finding).
    _L1_ORB_CPU = _orb

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
    # WIN_ALPHA_DISP, not WIN_ALPHA -- see the note where it is derived. The
    # same window is applied to the data, the templates and the residual, so
    # every trace on this panel stays directly comparable.
    _win, _ = windowfun("tukey", N_TD, alpha=WIN_ALPHA_DISP)
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
    _v5 = vgb_c[-1, WBEST]                                   # (55, 5) or (55, 6)
    # Reassemble to the 9-column GB basis
    # [dist, f0, Mc, phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio].
    # LEGACY (5-param DIST basis): Mc is catalogue-fixed (_r9[:, 2]),
    #   fdot_astro_ratio is sampled at _v5[:, 4].
    # CHIRP (6-param): Mc is SAMPLED at _v5[:, 4], and fdot_astro_ratio
    #   moves to _v5[:, 5]. Getting this wrong silently zeroes the VGB
    #   template amplitude (Mc value ~0.3 winds up in the ratio slot and
    #   the amplitude transform d(A)/d(Mc,dist,...) collapses) -- the
    #   sampler subtracts as normal, but the monitor's rebuild does not,
    #   and the DTR residual reads as if HM Cnc etc. were never fit.
    if VGB_SAMPLED_CHIRP:
        _vgb9 = np.column_stack([_v5[:, 0], _r9[:, 1], _v5[:, 4],
                                 _v5[:, 1], _v5[:, 2], _v5[:, 3],
                                 _r9[:, 6], _r9[:, 7], _v5[:, 5]])
    else:
        _vgb9 = np.column_stack([_v5[:, 0], _r9[:, 1], _r9[:, 2],
                                 _v5[:, 1], _v5[:, 2], _v5[:, 3],
                                 _r9[:, 6], _r9[:, 7], _v5[:, 4]])
    _tf = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                      use_distance=True, mc_lims=(0.001, 1.0))
    _comp = GBFDComputations(
        FDSettings(FDS.N, FDS.df, min_freq=0.0, max_freq=None,
                   force_backend="cpu"),
        T_REF, t_start=T_REF, N_sparse=2048, orbits=_orb,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_type="XYZ", nchannels=3, force_backend="cpu",
        # MUST match the window applied to the data above: the residual is
        # formed as data_fd - templates, so a different taper here would
        # subtract a differently-windowed signal and manufacture a residual
        # component out of the mismatch -- worse than the leakage this is
        # fixing. One alpha, both sides.
        tukey_alpha=WIN_ALPHA_DISP, edge_frac=0.0)
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

    # --- THE RUN'S OWN RESIDUAL, when the run wrote one (2026-10-07) -------
    # GF_RESIDUAL_SNAPSHOT_EVERY makes the head write the max-lnL walker's
    # residual -- the data minus EVERY fitted signal (GB, VGB, MBH, EMRI,
    # SOBBH), on the run's WDM grid -- next to the store
    # (lisatools.globalfit.residual_snapshot). When it is here, the data,
    # template and residual on this page are the run's: data and residual
    # come back to the frequency domain through WDMSignal.wdm_to_fd
    # (band-limited to the analysed band and edge-cropped span, which is all
    # the likelihood sees), and "template" is data - residual, i.e. every
    # signal class. Without it the page falls back to its own GB + VGB
    # subtraction and says so.
    _wdm = WDMSettings(W_NF, W_NT, W_DT, t0=float(_k["t0"]),
                       oversample=int(_k["oversample"]),
                       min_freq=float(_k["min_freq"]),
                       max_freq=float(_k["max_freq"]),
                       min_time=float(_k["min_time"]),
                       max_time=float(_k["max_time"]),
                       is_complex=bool(_k["is_complex"]),
                       force_backend="cpu")
    SNAP = None
    _snaps = sorted(_glob.glob(os.path.join(RUN_DIR, "*_residual_snapshot.npz")),
                    key=os.path.getmtime, reverse=True)
    if _snaps:
        try:
            # only the two walkers the whitened panels show are kept: the
            # max- and the min-lnL of the last stored row
            SNAP = load_residual_snapshot(
                _snaps[0], keep_walkers=(WBEST, int(np.argmin(ll[-1]))))
            _want = (3, int(_wdm.Nf_active), int(_wdm.Nt_active))
            if tuple(SNAP["data"].shape) != _want:
                raise ValueError(f"snapshot shape {SNAP['data'].shape} != the "
                                 f"store's WDM grid {_want}")
            if ("residuals" in SNAP
                    and tuple(SNAP["residuals"].shape[1:]) != _want):
                raise ValueError(f"snapshot residuals {SNAP['residuals'].shape} "
                                 f"!= the store's WDM grid {_want}")
            _full = WDMSettings(W_NF, W_NT, W_DT, oversample=int(_k["oversample"]),
                                force_backend="cpu")

            def _snap_fd(arr):
                return snapshot_wdm_to_fd(arr, _wdm, _full, FDS)

            data_fd = _snap_fd(SNAP["data"].astype(float))
            resid_fd = _snap_fd(SNAP["residual"].astype(float))
            DTR.update(snap=True, snap_it=(f"store row {SNAP['row']}"
                                           if SNAP["row"] is not None else
                                           f"job iteration {int(SNAP['iteration'])}"),
                       snap_walker=int(SNAP["walker"]),
                       snap_file=os.path.basename(_snaps[0]))
        except Exception as _exc:  # noqa: BLE001 - fall back, never fail the page
            MISSING.append(f"residual snapshot {os.path.basename(_snaps[0])} "
                           f"unusable ({_exc!r}); data/template/residual fall "
                           f"back to this page's own GB + VGB subtraction.")
            SNAP = None
    if SNAP is None:
        MISSING.append(
            "no run residual snapshot beside the store (GF_RESIDUAL_SNAPSHOT_EVERY): "
            "the data/template/residual panels subtract only the GB + VGB templates "
            "this page rebuilds, so every MBH / EMRI / SOBBH in the data shows in "
            "the residual.")
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
        """Log-frequency RMS binning: a 1.55M-bin spectrum has to lose 99.9%
        of its points to fit a PNG. Earlier this was a linear block-MAX of
        |y|; that preserved narrow GB lines but the sub-band region reads
        as visible waviness because Tukey-window sidelobes from strong
        sub-mHz TDI content sinc-leak across every bin below FLO and the
        MAX picks up their peaks. Log-frequency RMS (sqrt of the mean of
        |y|^2 per log-f bin) collapses the sinc envelope to its RMS while
        narrow GB lines still show as bumps above the residual floor
        (in-band, thousands of noise bins per log-f pixel drop the RMS
        floor well below the GB peak). Matches what the F1 residual-
        spectrum panel already does (see the ``_bmean`` block below).
        """
        x = np.asarray(x, float)
        y = np.abs(np.asarray(y))
        pos = x > 0
        if not pos.any():
            return x[:0], y[:0]
        xL, yL = x[pos], y[pos]
        ed = np.logspace(np.log10(xL[0]), np.log10(xL[-1]), npts + 1)
        bi = np.clip(np.searchsorted(ed, xL, side="right") - 1, 0, npts - 1)
        cnt = np.bincount(bi, minlength=npts).astype(float)
        s2 = np.bincount(bi, weights=yL ** 2, minlength=npts)
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(s2 / np.maximum(cnt, 1))
        fc = np.sqrt(ed[:-1] * ed[1:])
        ok = cnt > 0
        return fc[ok], rms[ok]

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
    if SNAP is not None:
        _cols = [
            ("data (as the run loaded it)", [("data", data_fd, DIM, 1.0, 0.9)]),
            ("all fitted signals (data - residual)",
             [("data", data_fd, DIM, 0.35, 0.9),
              ("all signals", data_fd - resid_fd, GREEN, 1.0, 0.7)]),
            ("residual (the run's own)",
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
    plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "dtr_fd")

    # ===================== WDM figure =====================================
    # (``_wdm`` is built above, beside the residual-snapshot read.)
    # POOLED TO THE PIXEL GRID (2026-10-11), see draw_wdm_tf_panels: the old
    # fixed 5x pool left ~3.7 columns per screen pixel and matplotlib's
    # resampling dropped single-column MBHB mergers at random per panel.
    # Each thunk builds ONE full-resolution |w| map; the figure pools it the
    # moment it exists, so no full-resolution map outlives its own pooling.
    if SNAP is not None:
        # the run's own arrays, on its own grid: no transform at all
        _wdm_src = (
            ("data", lambda: np.abs(np.asarray(SNAP["data"], dtype=float))),
            ("tmpl", lambda: np.abs(np.asarray(SNAP["data"], dtype=float)
                                    - SNAP["residual"])),
            ("res", lambda: np.abs(np.asarray(SNAP["residual"], dtype=float))))
    else:
        def _wdm_abs(arr):
            return lambda: np.abs(np.asarray(
                FDSignal(arr.copy(), FDS).transform(_wdm).arr))
        _wdm_src = (("data", _wdm_abs(data_fd)),
                    ("tmpl", _wdm_abs(_tmpl["gb"] + _tmpl["vgb"])),
                    ("res", _wdm_abs(resid_fd)))
    _te = np.asarray(_wdm.t_arr_edges); _fe = np.asarray(_wdm.f_arr_edges)
    _tt = (["data (as the run loaded it)",
            "all fitted signals (data - residual)",
            f"residual (the run's own, walker {int(SNAP['walker'])}, "
            + (f"store row {SNAP['row']})" if SNAP["row"] is not None else
               f"job iteration {int(SNAP['iteration'])})")]
           if SNAP is not None else
           ["data", "template sum (GB + VGB)", "residual = data - templates"])
    # MBHB MERGER TICKS. ``t_plunge`` is referenced to the MBH waveform epoch
    # (stock ``mbh_waveform_t0`` = MOJITO_REFERENCE_TIME on mojito data); WDM
    # layer 0 is the data start ``_t0_data`` (850.5 s later), and this axis
    # starts at the first ACTIVE edge ``_te[0]`` (the edge crop, 2.5 d on the
    # 6mo/9mo/1yr grids). So axis time = t_plunge - (data_t0 - epoch) - _te[0].
    _merge_d, _merge_src = [], ""
    try:
        _mbh_c = _row("chain/mbh", (NIT - 1, 0, 0))
        _mbh_i = _row("inds/mbh", (NIT - 1, 0, 0))
        if _mbh_c is not None:
            _merge_d = mbh_merger_plot_days(
                _mbh_c, _mbh_i,
                t_offset=(_t0_data - MOJITO_REFERENCE_TIME) + float(_te[0]))
            _merge_src = f"MBH cold chain, store row {NIT - 1}"
    except (KeyError, ValueError, IndexError):
        pass        # no MBH branch in this store: no ticks, nothing to say
    plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
    # ONE linear scale per channel row, keyed to that row's DATA panel (the
    # addremove debug convention), plotted as a FRACTION of the row scale --
    # so the residual panel darkens as sources leave instead of
    # re-autoscaling; see draw_wdm_tf_panels.
    _WDM_DPI = 88   # 9 dense images: the single most expensive PNG on the page
    fig, _tfi = draw_wdm_tf_panels(
        _wdm_src, layer_dt=float(_wdm.layer_dt), layer_df=float(_wdm.layer_df),
        t_edge0=float(_te[0]), f_edge0=float(_fe[0]),
        nt=int(_wdm.Nt_active), nf=int(_wdm.Nf_active),
        col_titles=_tt, row_names=CHN,
        suptitle=(f"WDM domain - |w_mn| on the run's own grid "
                  f"({_wdm.Nf_active} layers x {_wdm.layer_df * 1e3:.4f} mHz, "
                  f"{int(_wdm.layer_dt)} s pixels) - shared linear scale per "
                  f"channel row"),
        mergers_d=_merge_d, merger_src=_merge_src, dpi=_WDM_DPI)
    DTR.update(wdm_shape=(3, int(_wdm.Nf_active), int(_wdm.Nt_active)),
               wdm_pooled=_tfi["pooled_shape"], wdm_bt=_tfi["bt"], wdm_bf=_tfi["bf"],
               wdm_mergers=len(_tfi["mergers_d"]),
               layer_df=float(_wdm.layer_df), layer_dt=float(_wdm.layer_dt))
    fig_b64(fig, "dtr_wdm", dpi=_WDM_DPI)

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
    # the PSD of the TDI *X* channel in 1/Hz. It is not a strain amplitude and
    # it is not an ASD. XYZ everywhere (user ruling 2026-08-19): the run
    # analyses X/Y/Z, so the display stays in X against the X2TDISens noise
    # model -- no AET projection is formed anywhere on this page.
    def _AE(x):
        return (x[0], None)

    _W2 = float(np.mean(_win_keep ** 2))

    def _PSD(x):
        return 2.0 * np.abs(x) ** 2 / (N_TD * W_DT * _W2)

    _dA, _ = _AE(data_fd)
    _tA, _ = _AE(_tmpl["gb"] + _tmpl["vgb"])
    _rA, _ = _AE(resid_fd)

    from lisatools.sensitivity import get_sensitivity, X2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as _HTGF,
        FittedHyperbolicTangentGalacticForeground as _FHT)
    # WBEST's OWN noise, not the walker median (2026-09-24). The data,
    # templates and residual on this panel are all built from cold walker
    # WBEST, so the noise curve drawn over them has to be the one THAT
    # walker's likelihood used. The parameter-wise median is a different
    # model -- with the alpha posterior split it is not even a model any
    # walker holds -- so overlaying it invited exactly the "residual sits
    # above the curve" reading that is really a mismatched pairing.
    _pm = psd_cold[-1, WBEST]
    _gm = gal_cold_phys[-1, WBEST]
    _lmod = lisa_models.LISAModel(_pm[0] ** 2, _pm[1] ** 2,
                                  lisa_models.DefaultOrbits(), "sampled")
    _fpos = np.maximum(_fr, FDS.df)
    _Sinst = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                        model=_lmod, stochastic_params=()),
                        float)
    _Ssum = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens, model=_lmod,
                                       stochastic_params=tuple(_gm),
                                       stochastic_function=_HTGF), float)
    _Sgal = np.maximum(_Ssum - _Sinst, 1e-60)

    # Injected instrument + FittedHyperbolicTangent foreground for comparison.
    _lm_inj = lisa_models.LISAModel(SOMS_INJ ** 2, SA_INJ ** 2,
                                    lisa_models.DefaultOrbits(), "injected")
    _Sinst_inj = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                            model=_lm_inj, stochastic_params=()),
                            float)
    _Ssum_inj = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                           model=_lm_inj,
                                           stochastic_params=(SCI_TOBS,),
                                           stochastic_function=_FHT), float)
    _Sgal_inj = np.maximum(_Ssum_inj - _Sinst_inj, 1e-60)

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
    _Ni_inj = _bmean(_Sinst_inj[_sel])
    _Ng_inj = _bmean(_Sgal_inj[_sel])
    _Ns_inj = _bmean(_Ssum_inj[_sel])

    # Welch-in-frequency-domain smoothing on the log-f binned periodogram.
    # A single-windowed periodogram bin has Chi^2(2) statistics
    # (sigma/mu = 1 per FD bin), and the log-f block-MEAN above beats that
    # to sigma/mu = 1/sqrt(N) where N is the number of FD bins per log-f
    # pixel. At the low-f end where N is small, ~5-10% variance is still
    # visible on log-y over five decades of PSD range and reads as
    # waviness. Convolving with a Hann kernel of half-width H log-f bins
    # is equivalent to Welch's method with a Hann synthesis window across
    # neighbouring subbands: it multiplies the effective DOF by
    # (2H+1)*<w^2>/<w>^2, so H=2 (5-tap Hann) roughly triples the DOF and
    # drops sigma/mu by ~sqrt(3). Total power is preserved because the
    # kernel is normalised, and the model curves (_Ni/_Ng/_Ns) are already
    # smooth so the same kernel on them is a no-op up to floating point.
    # NaN-safe: bins the block-mean flagged as empty stay NaN.
    def _smooth_logf(y, half=2):
        w = np.hanning(2 * half + 1)
        w = w / w.sum()
        m = np.isfinite(y).astype(float)
        yz = np.where(np.isfinite(y), y, 0.0)
        yc = np.convolve(yz, w, mode="same")
        mc = np.convolve(m, w, mode="same")
        return np.where(mc > 1e-9, yc / np.maximum(mc, 1e-30), np.nan)
    _Pd = _smooth_logf(_Pd); _Pt = _smooth_logf(_Pt); _Pr = _smooth_logf(_Pr)
    _Ni = _smooth_logf(_Ni); _Ng = _smooth_logf(_Ng); _Ns = _smooth_logf(_Ns)
    _Ni_inj = _smooth_logf(_Ni_inj); _Ng_inj = _smooth_logf(_Ng_inj)
    _Ns_inj = _smooth_logf(_Ns_inj)

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
    ax[0].plot(_fcb, _Ni, color=FG, lw=1.2, ls=":", label="instrument noise (sampled)")
    ax[0].plot(_fcb, _Ng, color=AMBER, lw=1.2, ls=":",
               label="galactic foreground (sampled)")
    ax[0].plot(_fcb, _Ns, color=FG, lw=1.3, ls="--", label="their sum (sampled)")
    ax[0].plot(_fcb, _Ni_inj, color=RED, lw=1.0, ls=":", alpha=0.7,
               label="instrument noise (injected)")
    ax[0].plot(_fcb, _Ng_inj, color=RED, lw=1.0, ls="-.", alpha=0.7,
               label="FittedHyperbolicTangent foreground")
    ax[0].plot(_fcb, _Ns_inj, color=RED, lw=1.2, ls="--", alpha=0.7,
               label="their sum (injected)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylabel("PSD, TDI X channel  [1/Hz]")
    ax[0].set_ylim(1e-45, 2e-37)
    ax[0].legend(fontsize=7.5, loc="upper left", ncols=2)
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

    # ======== FOREGROUND DIAGNOSIS: the whitened residual per GB band =========
    # User request 2026-10-02 (6mo job 675, after the first ratchet cycle
    # released and the foreground climbed back): "whitened (included
    # foreground) residuals to visualize any excess power left over ... could
    # the rigidity of the model have anything to do with somewhat averaging
    # over sub-bands with a non-smooth, generally non-gaussian residual?"
    #
    # Per GB band (the run's OWN band_edges, ~17 uHz at 6 months => ~270 FD
    # bins per band in X): u = |r_X(f)|^2 / (S_total(f) N dt <w^2> / 4) / 2 is
    # Exp(1) when the noise model is right, so the band MEAN sits at 1, the
    # band mean/median at 1/ln2 = 1.443 and the fraction of bins above 4.605
    # at 1%. A smooth drift of the mean away from 1 that follows frequency is
    # the 5-parameter tanh x power-law failing to follow the residual's shape
    # (model rigidity); a band whose mean is 1 but whose mean/median and tail
    # are high holds its power in a few loud bins (unresolved or missed lines),
    # not in a Gaussian floor. Every cold walker is whitened by ITS OWN noise
    # (the whitening its likelihood used); the two reference ratios whiten the
    # same residual by instrument-only and by instrument + the FittedHT
    # foreground estimate, so the reader sees what the fit added and what the
    # estimate would have said.
    try:
        _nb = int(band_edges.size - 1)
        _bix = np.searchsorted(band_edges, _fr, side="right") - 1
        _inb = (_bix >= 0) & (_bix < _nb)
        _bix_in = _bix[_inb]
        _bfc = 0.5 * (band_edges[:-1] + band_edges[1:]) * 1e3          # mHz
        _cnt_b = np.bincount(_bix_in, minlength=_nb).astype(float)
        _okb = _cnt_b > 0
        _norm_w = N_TD * W_DT * _W2 / 4.0

        def _bmean_band(v):
            out = np.full(_nb, np.nan)
            out[_okb] = np.bincount(_bix_in, weights=v[_inb], minlength=_nb)[_okb] / _cnt_b[_okb]
            return out

        _ord_b = np.argsort(_bix_in, kind="stable")
        _bnd_b = np.searchsorted(_bix_in[_ord_b], np.arange(_nb + 1))

        def _bmedian_band(v):
            vs = v[_inb][_ord_b]
            out = np.full(_nb, np.nan)
            for _b in range(_nb):
                _lo, _hi = _bnd_b[_b], _bnd_b[_b + 1]
                if _hi > _lo:
                    out[_b] = np.median(vs[_lo:_hi])
            return out

        def _run_mean(y, half=7):
            """uniform running mean over 2*half+1 bands, NaN-aware"""
            w = np.ones(2 * half + 1)
            m = np.isfinite(y).astype(float)
            yz = np.where(np.isfinite(y), y, 0.0)
            yc = np.convolve(yz, w, mode="same")
            mc = np.convolve(m, w, mode="same")
            return np.where(mc > 0, yc / np.maximum(mc, 1e-30), np.nan)

        def _gbvgb_X(v):
            """NEW X-channel FD GB + VGB template sum of cold walker v at the
            last stored row, advanced onto the data grid (``_shift``). One
            3-channel generation buffer is alive at a time."""
            if v == WBEST:            # built above for the DTR panels
                return _tmpl["gb"][0] + _tmpl["vgb"][0]     # a new array
            _g9 = gb_chain_cold[v][gb_alive_last[v]]
            _v5w = vgb_c[-1, v]
            if VGB_SAMPLED_CHIRP:
                _vg9 = np.column_stack([_v5w[:, 0], _r9[:, 1], _v5w[:, 4],
                                        _v5w[:, 1], _v5w[:, 2], _v5w[:, 3],
                                        _r9[:, 6], _r9[:, 7], _v5w[:, 5]])
            else:
                _vg9 = np.column_stack([_v5w[:, 0], _r9[:, 1], _r9[:, 2],
                                        _v5w[:, 1], _v5w[:, 2], _v5w[:, 3],
                                        _r9[:, 6], _r9[:, 7], _v5w[:, 4]])
            _tX = np.zeros(FDS.N, dtype=np.complex128)
            for _rows in (_g9, _vg9):
                _a = np.zeros((1, 3, FDS.N), dtype=np.complex128)
                _comp.fill_global(_tf.both_transforms(_rows.copy()), _a,
                                  convert_to_ra_dec=False)
                _tX += _a[0, 0]
                del _a
            _tX *= _shift[0]
            return _tX

        def _snap_fd_x(arr):
            """X channel only of one snapshot WDM array -> FD."""
            return snapshot_wdm_to_fd(arr, _wdm, _full, FDS,
                                      channels=slice(0, 1))[0]

        # TWO WALKERS, NOT THE MEAN OF ALL OF THEM (user request 2026-10-02):
        # the max- and min-lnL cold walkers of the last stored row, each
        # whitened by its own noise. A walker mean blends several noise models
        # into a curve no walker holds; the best and the worst walker bracket
        # what the cold chain actually carries. Index 0 = max lnL, 1 = min lnL
        # in every array below.
        #
        # WHICH RESIDUAL each one is (2026-10-11): walker_residual_x picks the
        # run's own residual when the snapshot carries that walker, else the
        # snapshot walker's residual with this walker's GB + VGB swapped in,
        # else (no snapshot) data minus GB + VGB only. Until 2026-10-11 the
        # min-lnL walker always got "data minus its GB + VGB", i.e. every
        # MBH / EMRI / SOBBH left in, while the max-lnL walker had the run's
        # full residual -- the 9mo page's 5-7x amber "excess" below 1.5 mHz.
        WWORST = int(np.argmin(ll[-1]))
        _WSHOW = ((WBEST, CYAN, "max lnL"), (WWORST, AMBER, "min lnL"))
        _wkind, _wlab = [], []
        _gal_inj_curve = np.maximum(_Ssum_inj - _Sinst_inj, 0.0)
        _rat_tot, _rat_inst, _rat_est, _mm, _tail, _share = [], [], [], [], [], []
        _psd_b, _Stot_b, _Sinst_b = [], [], []
        for _w, _wc, _wl in _WSHOW:
            _pmw = psd_cold[-1, _w]
            _gmw = gal_cold_phys[-1, _w]
            _lmw = lisa_models.LISAModel(_pmw[0] ** 2, _pmw[1] ** 2,
                                         lisa_models.DefaultOrbits(), "sampled")
            _Si_w = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens, model=_lmw,
                                               stochastic_params=()), float)
            _St_w = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens, model=_lmw,
                                               stochastic_params=tuple(_gmw),
                                               stochastic_function=_HTGF), float)
            _rX, _kd, _lb = walker_residual_x(
                _w, WBEST, SNAP, resid_fd[0], _gbvgb_X, _snap_fd_x, data_fd[0])
            _wkind.append(_kd); _wlab.append(_lb)
            _p = np.abs(_rX) ** 2 / _norm_w / 2.0
            _u = _p / np.maximum(_St_w, 1e-60)
            _rat_tot.append(_bmean_band(_u))
            _rat_inst.append(_bmean_band(_p / np.maximum(_Si_w, 1e-60)))
            _rat_est.append(_bmean_band(_p / np.maximum(_Si_w + _gal_inj_curve, 1e-60)))
            _mm.append(_rat_tot[-1] / _bmedian_band(_u))
            _tail.append(_bmean_band((_u > 4.605).astype(float)))
            _share.append(_bmean_band(np.maximum(_St_w - _Si_w, 0.0) / np.maximum(_St_w, 1e-60)))
            _psd_b.append(_bmean_band(_p)); _Stot_b.append(_bmean_band(_St_w))
            _Sinst_b.append(_bmean_band(_Si_w))
            del _rX, _p, _u, _Si_w, _St_w
        _rat_tot = np.array(_rat_tot); _rat_inst = np.array(_rat_inst)
        _rat_est = np.array(_rat_est); _mm = np.array(_mm); _tail = np.array(_tail)
        _share = np.array(_share)
        _psd_b = np.array(_psd_b); _Stot_b = np.array(_Stot_b); _Sinst_b = np.array(_Sinst_b)
        _Sest_b = _bmean_band(_Sinst_inj + _gal_inj_curve)
        # stored row 0's noise (under a pinned launch that is the start vector
        # the search ran against until the first release), for the PSD zoom
        try:
            _lm0 = lisa_models.LISAModel(float(psd_cold[0, 0, 0]) ** 2,
                                         float(psd_cold[0, 0, 1]) ** 2,
                                         lisa_models.DefaultOrbits(), "row0")
            _S0_b = _bmean_band(np.asarray(get_sensitivity(
                _fpos, sens_fn=X2TDISens, model=_lm0,
                stochastic_params=tuple(gal_cold_phys[0, 0]),
                stochastic_function=_HTGF), float))
        except Exception:
            _S0_b = None
        _wtitle = (f"cold walkers {WBEST} (max lnL, cyan) and {WWORST} (min lnL, amber) "
                   f"of stored row {NIT - 1}")
        # HOW each curve's residual was built, said on every figure and in
        # the prose -- the two curves need not be the same construction.
        _snap_at = ""
        if SNAP is not None:
            _snap_at = (f"snapshot at store row {SNAP['row']}" if SNAP["row"] is not None
                        else f"snapshot at job iteration {int(SNAP['iteration'])}")
        _wsrc_l = [f"{('cyan', 'amber')[_k]} walker {_w}: {_wlab[_k]}"
                   for _k, (_w, _wc, _wl) in enumerate(_WSHOW)]
        if _snap_at:
            _wsrc_l[-1] += f" ({_snap_at})"
        _wsrc = "; ".join(_wsrc_l)               # one line, for the captions
        _wsrc_t = "\n".join(_wsrc_l)             # one line per walker, for titles
        _wleg = [f"walker {_w} ({_wl}; {RESID_KIND_SHORT[_wkind[_k]]})"
                 for _k, (_w, _wc, _wl) in enumerate(_WSHOW)]
        FG_RESID_SRC = f" Residuals: {_wsrc}."

        fig, ax = plt.subplots(4, 1, figsize=(11.6, 10.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.4, 1.2, 1.2, 0.9],
                                                hspace=0.08))
        for a_ in ax:
            a_.axvspan(3.0, 5.0, color=FG, alpha=0.05, lw=0)
            a_.set_xscale("log")
        for _k, (_w, _wc, _wl) in enumerate(_WSHOW):
            ax[0].scatter(_bfc, _rat_tot[_k], s=5, color=_wc, alpha=0.3, lw=0)
            ax[0].plot(_bfc, _run_mean(_rat_tot[_k]), color=_wc, lw=1.8,
                       label=f"{_wleg[_k]} against its own noise: per band (points), "
                             "running mean over 15 bands")
            ax[0].plot(_bfc, _run_mean(_rat_inst[_k]), color=_wc, lw=1.0, ls="--",
                       label=("dashed: same residual / instrument-only (what the foreground "
                              "carries)") if _k == 0 else None)
            ax[0].plot(_bfc, _run_mean(_rat_est[_k]), color=_wc, lw=1.0, ls="-.",
                       label=("dash-dot: same residual / (instrument + FittedHT foreground "
                              "estimate)") if _k == 0 else None)
            ax[1].scatter(_bfc, _mm[_k], s=5, color=_wc, alpha=0.3, lw=0)
            ax[1].plot(_bfc, _run_mean(_mm[_k]), color=_wc, lw=1.6)
            ax[2].scatter(_bfc, 100 * _tail[_k], s=5, color=_wc, alpha=0.3, lw=0)
            ax[2].plot(_bfc, 100 * _run_mean(_tail[_k]), color=_wc, lw=1.6)
            ax[3].plot(_bfc, 100 * _share[_k], color=_wc, lw=1.6)
        ax[0].axhline(1.0, color=FG, lw=1.0, ls=":")
        ax[0].set_yscale("log"); ax[0].set_ylim(0.3, 30)
        ax[0].set_ylabel("whitened residual power\nper GB band (1 = model)", fontsize=9)
        ax[0].legend(fontsize=7.5, loc="upper right")
        ax[1].axhline(1.0 / np.log(2.0), color=FG, lw=1.0, ls=":",
                      label="Gaussian: mean/median = 1/ln 2 = 1.443")
        ax[1].set_ylim(1.0, 4.0)
        ax[1].set_ylabel("mean / median\nof whitened power", fontsize=9)
        ax[1].legend(fontsize=7.5, loc="upper right")
        ax[2].axhline(1.0, color=FG, lw=1.0, ls=":", label="Gaussian: 1% of bins above 4.605")
        ax[2].set_ylim(0, 8)
        ax[2].set_ylabel("bins above 4.605\n[% per band]", fontsize=9)
        ax[2].legend(fontsize=7.5, loc="upper right")
        ax[3].set_ylim(0, 100)
        ax[3].set_ylabel("foreground share\nof model noise [%]", fontsize=9)
        ax[3].set_xlabel("frequency [mHz]  (shaded: 3-5 mHz)")
        ax[3].set_xlim(max(_bfc[0], 0.5), _bfc[-1])
        fig.suptitle(
            f"whitened residual per GB band, {_wtitle}, TDI X -- 1 means the residual "
            f"carries exactly the power the fitted noise says\nresiduals -- {_wsrc_t}",
            fontsize=9, color=FG)
        fig.tight_layout(rect=[0, 0, 1, 0.945])
        fig_b64(fig, "fg_whitened_bands")

        # ---- ZOOM 2.5-5.5 mHz, linear frequency (user request 2026-10-02) ----
        _zm = (_bfc >= 2.5) & (_bfc <= 5.5)
        fig, ax = plt.subplots(3, 1, figsize=(11.6, 8.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.0, 1.1, 1.1], hspace=0.08))
        for _k, (_w, _wc, _wl) in enumerate(_WSHOW):
            ax[0].scatter(_bfc[_zm], _rat_tot[_k][_zm], s=6, color=_wc, alpha=0.3, lw=0)
            ax[0].plot(_bfc[_zm], _run_mean(_rat_tot[_k], half=2)[_zm], color=_wc, lw=2.0,
                       label=f"{_wleg[_k]} against its own noise: per band (points), "
                             "running mean over 5 bands")
            ax[0].plot(_bfc[_zm], _run_mean(_rat_inst[_k], half=2)[_zm], color=_wc, lw=1.0,
                       ls="--", label="dashed: same residual / instrument-only"
                       if _k == 0 else None)
            ax[0].plot(_bfc[_zm], _run_mean(_rat_est[_k], half=2)[_zm], color=_wc, lw=1.0,
                       ls="-.", label=("dash-dot: same residual / (instrument + FittedHT "
                                       "foreground estimate)") if _k == 0 else None)
            ax[1].scatter(_bfc[_zm], _mm[_k][_zm], s=6, color=_wc, alpha=0.3, lw=0)
            ax[1].plot(_bfc[_zm], _run_mean(_mm[_k], half=2)[_zm], color=_wc, lw=1.8)
            ax[2].scatter(_bfc[_zm], 100 * _tail[_k][_zm], s=6, color=_wc, alpha=0.3, lw=0)
            ax[2].plot(_bfc[_zm], 100 * _run_mean(_tail[_k], half=2)[_zm], color=_wc, lw=1.8)
        ax[0].axhline(1.0, color=FG, lw=1.0, ls=":")
        ax[0].set_yscale("log"); ax[0].set_ylim(0.5, 20)
        ax[0].set_ylabel("whitened residual power\nper GB band (1 = model)", fontsize=9)
        ax[0].legend(fontsize=7.5, loc="upper right")
        ax[1].axhline(1.0 / np.log(2.0), color=FG, lw=1.0, ls=":", label="Gaussian: 1.443")
        ax[1].set_ylim(1.0, 4.0)
        ax[1].set_ylabel("mean / median\nof whitened power", fontsize=9)
        ax[1].legend(fontsize=7.5, loc="upper right")
        ax[2].axhline(1.0, color=FG, lw=1.0, ls=":", label="Gaussian: 1% above 4.605")
        ax[2].set_ylim(0, 10)
        ax[2].set_ylabel("bins above 4.605\n[% per band]", fontsize=9)
        ax[2].legend(fontsize=7.5, loc="upper right")
        ax[2].set_xlabel("frequency [mHz]"); ax[2].set_xlim(2.5, 5.5)
        fig.suptitle(f"ZOOM 2.5-5.5 mHz: whitened residual per GB band, {_wtitle}, each "
                     f"against its own fitted noise (linear frequency)\nresiduals -- {_wsrc_t}",
                     fontsize=9, color=FG)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig_b64(fig, "fg_whitened_zoom")

        # ---- ZOOM 2.5-5.5 mHz: the UNWHITENED residual against the curves ----
        # colour = walker; solid = its residual, dashed = its fitted noise,
        # dotted = its fitted instrument alone
        fig, ax = plt.subplots(1, 1, figsize=(11.6, 5.4))
        for _k, (_w, _wc, _wl) in enumerate(_WSHOW):
            ax.plot(_bfc[_zm], _psd_b[_k][_zm], color=_wc, lw=1.5,
                    label=f"{_wleg[_k]}: residual PSD per GB band")
            ax.plot(_bfc[_zm], _Stot_b[_k][_zm], color=_wc, lw=1.3, ls="--",
                    label=f"walker {_w}: its fitted noise, instrument + galfor")
            ax.plot(_bfc[_zm], _Sinst_b[_k][_zm], color=_wc, lw=0.9, ls=":",
                    label=f"walker {_w}: its fitted instrument only")
        ax.plot(_bfc[_zm], _Sest_b[_zm], color=RED, lw=1.1, ls="-.",
                label="injected instrument + FittedHT foreground estimate")
        if _S0_b is not None:
            ax.plot(_bfc[_zm], _S0_b[_zm], color=VIOLET, lw=1.3, ls=(0, (4, 2)),
                    label="stored row 0 noise, walker 0 (the curve the search ran against "
                          "before the first release)")
        ax.set_yscale("log"); ax.set_xlim(2.5, 5.5)
        ax.set_xlabel("frequency [mHz]")
        ax.set_ylabel("PSD of TDI X [1/Hz], mean per GB band", fontsize=9)
        _zz = _psd_b[:, _zm]; _zz = _zz[np.isfinite(_zz)]
        _zi = _Sinst_b[:, _zm]; _zi = _zi[np.isfinite(_zi)]
        if _zz.size and _zi.size:
            ax.set_ylim(0.5 * float(np.min(_zi)), 3.0 * float(np.max(_zz)))
        ax.legend(fontsize=7.5, loc="upper right", ncols=2)
        fig.suptitle("ZOOM 2.5-5.5 mHz: residual spectrum against the noise curves (unwhitened)"
                     f"\nresiduals -- {_wsrc_t}", fontsize=9, color=FG)
        fig.tight_layout(rect=[0, 0, 1, 0.885])
        fig_b64(fig, "fg_resid_psd_zoom")

        def _at(arr, f_mhz):
            _i = int(np.clip(np.searchsorted(_bfc, f_mhz), 0, _nb - 1))
            return float(arr[_i]) if np.isfinite(arr[_i]) else float("nan")

        _probe_f = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0)
        _m35 = (_bfc >= 3.0) & (_bfc <= 5.0)
        _m12 = (_bfc >= 1.0) & (_bfc <= 2.0)
        _m610 = (_bfc >= 6.0) & (_bfc <= 10.0)
        # readouts per shown walker: bare keys = max lnL, "_min" = min lnL
        FGW.update(walker=WBEST, walker_min=WWORST, bands_35=int(_m35.sum()),
                   kind=_wkind[0], kind_min=_wkind[1],
                   src=_wlab[0], src_min=_wlab[1], snap_at=_snap_at)
        # the prose for the caption's caveat (1): what each curve's residual is
        _snap_w = int(SNAP["walker"]) if SNAP is not None else None
        _why = {
            "run": ("the run's own residual: the data it loaded minus every fitted "
                    "signal class (GB, VGB, MBH, EMRI, SOBBH), band-limited to the "
                    "analysed band"),
            "swap": (f"walker {_snap_w}'s own residual from the run with that walker's "
                     f"GB + VGB templates swapped in for walker {_snap_w}'s, so its "
                     f"MBH / EMRI / SOBBH subtraction is walker {_snap_w}'s fit, not "
                     "its own (the snapshot predates per-walker residuals, "
                     "2026-10-11); a GB + VGB difference between the two walkers "
                     "shows, a difference in the other classes does not"),
            "gbvgb": ("the data minus that walker's GB + VGB templates only (the run "
                      "wrote no residual snapshot; GF_RESIDUAL_SNAPSHOT_EVERY), so "
                      "every MBH / EMRI / SOBBH in its data ("
                      + (" + ".join(sorted(globals().get("_files", {}))) or "?")
                      + ") is still in it"),
        }
        if _wkind[0] == _wkind[1]:
            _src_txt = f"Both curves are {_why[_wkind[0]]}"
        else:
            _src_txt = (f"The cyan curve (walker {WBEST}) is {_why[_wkind[0]]}; the "
                        f"amber curve (walker {WWORST}) is {_why[_wkind[1]]}")
        FGW["src_txt"] = _src_txt + (f" ({_snap_at})" if _snap_at else "") + ". "
        for _k, _sfx in ((0, ""), (1, "_min")):
            _rm = _run_mean(_rat_tot[_k])
            FGW.update({
                "rat_at" + _sfx: {f_: _at(_rm, f_) for f_ in _probe_f},
                "inst_at" + _sfx: {f_: _at(_run_mean(_rat_inst[_k]), f_) for f_ in _probe_f},
                "est_at" + _sfx: {f_: _at(_run_mean(_rat_est[_k]), f_) for f_ in _probe_f},
                "share_at" + _sfx: {f_: _at(_share[_k], f_) for f_ in _probe_f},
                "rat_35" + _sfx: float(np.nanmean(_rat_tot[_k][_m35])),
                "rat_12" + _sfx: float(np.nanmean(_rat_tot[_k][_m12])),
                "rat_610" + _sfx: float(np.nanmean(_rat_tot[_k][_m610])),
                "mm_35" + _sfx: float(np.nanmedian(_mm[_k][_m35])),
                "mm_12" + _sfx: float(np.nanmedian(_mm[_k][_m12])),
                "mm_610" + _sfx: float(np.nanmedian(_mm[_k][_m610])),
                "tail_35" + _sfx: float(np.nanmean(_tail[_k][_m35])),
                "tail_12" + _sfx: float(np.nanmean(_tail[_k][_m12])),
                "tail_610" + _sfx: float(np.nanmean(_tail[_k][_m610])),
                "bands_nongauss_35" + _sfx: int(np.sum(_tail[_k][_m35] > 0.02)),
                "spread_35" + _sfx: float(np.nanmax(_rm[_m35]) / np.nanmin(_rm[_m35])),
            })
        del _rat_tot, _rat_inst, _rat_est, _mm, _tail, _share
    except Exception as _e:
        MISSING.append(f"whitened residual per band unavailable: {type(_e).__name__}: {_e}")
    del data_fd, resid_fd, _tmpl, _orb, _comp
except Exception as e:
    MISSING.append(
        f"data/template/residual panels unavailable: {type(e).__name__}: {e}")

# ============ FOREGROUND RATCHET: the timeline and where the leaves went =====
# (user request 2026-10-02.) Under GALFOR_RATCHET=1 + GB_SEARCH_LEGS=1 every
# noise CHANGE (nudge or release) saves a row right after the gated noise
# step, so the store carries the experiment itself: ``saved_after`` names the
# gate rows, the galfor cold chain tells a nudge (alpha and f_1 untouched,
# amplitude stepped) from a release (everything moves), and the lnL / leaf
# trajectories around them are the readout. REF is the last row BEFORE the
# first nudge; everything in this block is relative to it.
RATCHET = {}
cap_fgA = cap_fgB = cap_fgC = ""
cap_fgD = cap_fgE = ""      # filled by the ratchet scorecard after the truth set loads
try:
    _sa = g.get("saved_after")
    SAVED_AFTER = ([s.decode() if isinstance(s, bytes) else str(s) for s in _sa[:NIT]]
                   if _sa is not None else [])
    _gate_rows = [i for i, s in enumerate(SAVED_AFTER) if s == "noise_ratchet_search"]
    if not _gate_rows or _gate_rows[0] < 1:
        raise RuntimeError("no row saved after the gated noise step (ratchet or legs off)")
    REF = _gate_rows[0] - 1
    _gn = int(min(NIT, gal_cold_phys.shape[0]))
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as _HTr
    _pf = np.array([2e-3, 3e-3, 4e-3, 5e-3])

    def _gal_mean_curve(i, f_hz):
        """cold-walker-mean S_gal(f) of stored row i (physical galfor params)"""
        return np.mean([np.asarray(_HTr.specific_Sh_function(f_hz, *gal_cold_phys[i, w]))
                        for w in range(nwalk)], axis=0)

    _ref_curve = _gal_mean_curve(REF, _pf)
    _ratio_rows = np.array([_gal_mean_curve(i, _pf) / _ref_curve for i in range(_gn)])
    _kind = {}
    for _r in _gate_rows:
        if _r >= _gn:
            continue
        _a_same = np.allclose(gal_cold_phys[_r, :, 2], gal_cold_phys[_r - 1, :, 2],
                              rtol=0.0, atol=1e-9)
        _f1_same = np.allclose(gal_cold_phys[_r, :, 3], gal_cold_phys[_r - 1, :, 3],
                               rtol=1e-9, atol=0.0)
        _amp_moved = not np.allclose(gal_cold_phys[_r, :, 0], gal_cold_phys[_r - 1, :, 0],
                                     rtol=1e-9, atol=0.0)
        _kind[_r] = "nudge" if (_a_same and _f1_same and _amp_moved) else "release"
    _nudges = [r for r, k in _kind.items() if k == "nudge"]
    _releases = [r for r, k in _kind.items() if k == "release"]
    leaves_w = gb_inds.sum(-1).astype(int)                     # (it, nw)
    _dll = ll - ll[REF][None, :]                                 # (it, nw)
    _x0 = max(REF - 3, 0)
    fig, ax = plt.subplots(3, 1, figsize=(11.6, 8.8), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.6, 1.2, 1.2], hspace=0.08))
    _bounds = sorted(_kind)
    for _j, _r in enumerate(_bounds):
        _r_next = _bounds[_j + 1] if _j + 1 < len(_bounds) else NIT
        _col = AMBER if _kind[_r] == "nudge" else GREEN
        for a_ in ax:
            a_.axvspan(_r - 0.5, _r_next - 0.5, color=_col, alpha=0.08, lw=0)
            a_.axvline(_r, color=_col, lw=0.9, ls=(0, (3, 2)))
        ax[0].annotate(_kind[_r], xy=(_r, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(2, -2), textcoords="offset points", fontsize=7,
                       color=_col, rotation=90, ha="left", va="top")
    for _w in range(nwalk):
        ax[0].plot(it[_x0:], _dll[_x0:, _w], color=CYAN, lw=0.7, alpha=0.4)
    ax[0].plot(it[_x0:], _dll[_x0:].mean(axis=1), color=CYAN, lw=1.8,
               label=f"walker mean (thin: each of {nwalk})")
    ax[0].axhline(0.0, color=FG, lw=1.0, ls=":")
    ax[0].set_ylabel(f"cold lnL minus row {REF}\n(the row before the first nudge)", fontsize=9)
    ax[0].legend(fontsize=7.5, loc="lower right")
    for _k, (_fm, _col) in enumerate(zip((2, 3, 4, 5), (FG, VIOLET, CYAN, AMBER))):
        ax[1].plot(it[_x0:_gn], _ratio_rows[_x0:_gn, _k], color=_col, lw=1.4,
                   label=f"{_fm} mHz")
    ax[1].axhline(1.0, color=FG, lw=1.0, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(f"S_gal / S_gal at row {REF}\n(cold-walker mean)", fontsize=9)
    ax[1].legend(fontsize=7.5, loc="lower right", ncols=4)
    for _w in range(nwalk):
        ax[2].plot(it[_x0:], leaves_w[_x0:, _w] - leaves_w[REF, _w], color=GREEN,
                   lw=0.7, alpha=0.45)
    ax[2].plot(it[_x0:], (leaves_w[_x0:] - leaves_w[REF][None, :]).mean(axis=1),
               color=GREEN, lw=1.8)
    ax[2].axhline(0.0, color=FG, lw=1.0, ls=":")
    ax[2].set_ylabel(f"cold GB leaves minus row {REF}", fontsize=9)
    ax[2].set_xlabel("stored row (one per search leg; amber = nudge + hold, "
                     "green = release)")
    for _i in range(_x0, NIT):
        if _i < len(SAVED_AFTER) and SAVED_AFTER[_i]:
            ax[2].annotate(SAVED_AFTER[_i].replace("noise_ratchet_search", "noise")
                           .replace("in_model", "im"),
                           xy=(_i, 0.0), xycoords=("data", "axes fraction"),
                           xytext=(0, 2), textcoords="offset points", fontsize=5.5,
                           color=DIM, rotation=90, ha="center", va="bottom")
    fig.suptitle(f"galfor ratchet: stored rows {_x0}-{NIT - 1}, reference row {REF}",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "fg_ratchet_timeline")
    _mean_dll = _dll[REF:].mean(axis=1)
    RATCHET.update(
        ref=REF, nudges=_nudges, releases=_releases,
        dll_last=float(_dll[-1].mean()),
        dll_min=float(_mean_dll.min()), dll_min_row=int(REF + np.argmin(_mean_dll)),
        dll_first_release=(float(_dll[_releases[0]].mean()) if _releases else float("nan")),
        dll_before_release=(float(_dll[_releases[0] - 1].mean()) if _releases else float("nan")),
        leaves_ref=float(leaves_w[REF].mean()), leaves_last=float(leaves_w[-1].mean()),
        leaves_before_release=(float(leaves_w[_releases[0] - 1].mean()) if _releases
                               else float("nan")),
        ratio_last={fm: float(_ratio_rows[_gn - 1, k]) for k, fm in enumerate((2, 3, 4, 5))},
        ratio_held=({fm: float(_ratio_rows[_nudges[0], k]) for k, fm in enumerate((2, 3, 4, 5))}
                    if _nudges else {}),
        gal_ref=[float(v) for v in gal_cold[REF].mean(axis=0)],
        gal_last=[float(v) for v in gal_cold[_gn - 1].mean(axis=0)],
    )
except Exception as _e:
    MISSING.append(f"ratchet timeline unavailable: {type(_e).__name__}: {_e}")

try:
    if "REF" not in globals():
        raise RuntimeError("no ratchet reference row")
    _chg = g["chain/gb"]
    _ing = g["inds/gb"]
    _cref = None
    for _i in range(REF, NIT - 1):
        # an extract carries coordinates only in its keep window; the earliest
        # row that has them is the honest reference, named in the caption
        if np.any(_chg[_i, 0, 0, :, :, 1] != 0.0):
            _cref = _i
            break
    if _cref is None:
        raise RuntimeError("no coordinate-bearing row before the last one (extract keep window)")

    def _f0s(i):
        return [_chg[i, 0, 0, w][_ing[i, 0, 0, w].astype(bool)][:, 1] for w in range(nwalk)]

    _f_ref = _f0s(_cref)
    _f_last = _f0s(NIT - 1)
    _lo_m, _hi_m = float(band_edges[0]) * 1e3, float(band_edges[-1]) * 1e3
    _edges_m = np.arange(np.floor(_lo_m * 4) / 4, _hi_m + 0.25, 0.25)
    _fc_m = 0.5 * (_edges_m[:-1] + _edges_m[1:])
    _Href = np.array([np.histogram(_f_ref[w], bins=_edges_m)[0] for w in range(nwalk)])
    _Hlast = np.array([np.histogram(_f_last[w], bins=_edges_m)[0] for w in range(nwalk)])
    _H = _Hlast - _Href
    _bl = sub.get("gb/band_cold_logl_w")
    _dB = None
    if _bl is not None and _bl.shape[0] >= NIT:
        _d = _bl[NIT - 1].astype(float) - _bl[REF].astype(float)
        _d = np.where(np.isfinite(_d), _d, 0.0)                     # (nw, nbands)
        _bfc_m = 0.5 * (band_edges[:-1] + band_edges[1:]) * 1e3
        _bb = np.clip(np.searchsorted(_edges_m, _bfc_m, side="right") - 1, 0, _fc_m.size - 1)
        _dB = np.array([np.bincount(_bb, weights=_d[w], minlength=_fc_m.size)
                        for w in range(nwalk)])
    _rat_curve = (_gal_mean_curve(min(NIT - 1, _gn - 1), _fc_m * 1e-3)
                  / _gal_mean_curve(REF, _fc_m * 1e-3))
    _nrow = 2 if _dB is not None else 1
    fig, ax = plt.subplots(_nrow, 1, figsize=(11.6, 7.2 if _nrow == 2 else 4.6),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    for _w in range(nwalk):
        ax[0].step(_edges_m[:-1], _H[_w], where="post", color=GREEN, lw=0.7, alpha=0.4)
    ax[0].step(_edges_m[:-1], _H.mean(axis=0), where="post", color=GREEN, lw=1.8,
               label=f"leaves added per 0.25 mHz, walker mean (thin: each of {nwalk})")
    ax[0].axhline(0, color=FG, lw=1.0, ls=":")
    ax[0].axvspan(3.0, 5.0, color=AMBER, alpha=0.07, lw=0)
    ax[0].set_ylabel(f"cold GB leaves, row {NIT - 1} minus row {_cref}", fontsize=9)
    _ax0b = ax[0].twinx()
    _ax0b.plot(_fc_m, _rat_curve, color=AMBER, lw=1.5,
               label=f"foreground S_gal, row {min(NIT - 1, _gn - 1)} / row {REF}")
    _ax0b.axhline(1.0, color=AMBER, lw=0.8, ls=":")
    _ax0b.set_yscale("log"); _ax0b.set_ylim(0.3, 3.0)
    _ax0b.set_ylabel("S_gal ratio (cold mean)", fontsize=9, color=AMBER)
    _ax0b.tick_params(axis="y", colors=AMBER)
    _ax0b.grid(False)
    _h1, _l1 = ax[0].get_legend_handles_labels()
    _h2, _l2 = _ax0b.get_legend_handles_labels()
    ax[0].legend(_h1 + _h2, _l1 + _l2, fontsize=7.5, loc="upper right")
    if _dB is not None:
        ax[1].step(_edges_m[:-1], _dB.mean(axis=0), where="post", color=VIOLET, lw=1.6,
                   label=f"per-band cold lnL, row {NIT - 1} minus row {REF} "
                         "(walker mean, summed per 0.25 mHz; carries the noise-term "
                         "change as well as the sources)")
        ax[1].axhline(0, color=FG, lw=1.0, ls=":")
        ax[1].axvspan(3.0, 5.0, color=AMBER, alpha=0.07, lw=0)
        ax[1].set_ylabel("band lnL change", fontsize=9)
        ax[1].legend(fontsize=7.5, loc="lower right")
    ax[-1].set_xlabel("frequency [mHz]  (shaded: 3-5 mHz)")
    ax[-1].set_xlim(max(_lo_m, 0.5), min(_hi_m, 12.0))
    fig.suptitle("where the leaves went across the ratchet (cold chain, per walker)",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "fg_leaf_delta")

    # ---- ZOOM 2.5-5.5 mHz at 0.05 mHz bins (user request 2026-10-02) ----
    _ez = np.arange(2.5, 5.5 + 1e-9, 0.05)
    _fz = 0.5 * (_ez[:-1] + _ez[1:])
    _Hz_ref = np.array([np.histogram(_f_ref[w], bins=_ez)[0] for w in range(nwalk)])
    _Hz = np.array([np.histogram(_f_last[w], bins=_ez)[0] for w in range(nwalk)]) - _Hz_ref
    _nrz = 2 if _dB is not None else 1
    fig, ax = plt.subplots(_nrz, 1, figsize=(11.6, 7.0 if _nrz == 2 else 4.4),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    ax[0].bar(_fz, _Hz_ref.mean(axis=0), width=0.05, color=DIM, alpha=0.35, lw=0,
              label=f"leaves per 0.05 mHz at row {_cref} (walker mean)")
    ax[0].bar(_fz, _Hz.mean(axis=0), width=0.05, color=GREEN, alpha=0.9, lw=0,
              label=f"added since row {_cref} (walker mean; negative = removed)")
    ax[0].axhline(0, color=FG, lw=1.0, ls=":")
    ax[0].set_ylabel("cold GB leaves", fontsize=9)
    _axz = ax[0].twinx()
    _rz = (_gal_mean_curve(min(NIT - 1, _gn - 1), _fz * 1e-3)
           / _gal_mean_curve(REF, _fz * 1e-3))
    _axz.plot(_fz, _rz, color=AMBER, lw=1.5,
              label=f"S_gal row {min(NIT - 1, _gn - 1)} / row {REF} (cold mean)")
    _axz.axhline(1.0, color=AMBER, lw=0.8, ls=":")
    _axz.set_ylim(0.5, 1.6); _axz.set_ylabel("S_gal ratio", color=AMBER, fontsize=9)
    _axz.tick_params(axis="y", colors=AMBER); _axz.grid(False)
    _h1, _l1 = ax[0].get_legend_handles_labels()
    _h2, _l2 = _axz.get_legend_handles_labels()
    ax[0].legend(_h1 + _h2, _l1 + _l2, fontsize=7.5, loc="upper right")
    if _dB is not None:
        _bbz = np.searchsorted(_ez, _bfc_m, side="right") - 1
        _inz = (_bbz >= 0) & (_bbz < _fz.size)
        _dBz = np.array([np.bincount(_bbz[_inz], weights=_d[w][_inz], minlength=_fz.size)
                         for w in range(nwalk)])
        ax[1].bar(_fz, _dBz.mean(axis=0), width=0.05, color=VIOLET, alpha=0.9, lw=0,
                  label=f"per-band cold lnL, row {NIT - 1} minus row {REF}, summed per "
                        "0.05 mHz (walker mean; sources added AND the noise-term change)")
        ax[1].axhline(0, color=FG, lw=1.0, ls=":")
        ax[1].set_ylabel("band lnL change", fontsize=9)
        ax[1].legend(fontsize=7.5, loc="lower right")
    ax[-1].set_xlabel("frequency [mHz]"); ax[-1].set_xlim(2.5, 5.5)
    fig.suptitle("ZOOM 2.5-5.5 mHz: where the leaves went, 0.05 mHz bins", fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "fg_leaf_delta_zoom")
    _bins_m = {"1-2": (_fc_m >= 1.0) & (_fc_m < 2.0), "2-3": (_fc_m >= 2.0) & (_fc_m < 3.0),
               "3-4": (_fc_m >= 3.0) & (_fc_m < 4.0), "4-5": (_fc_m >= 4.0) & (_fc_m < 5.0),
               "5-7": (_fc_m >= 5.0) & (_fc_m < 7.0), "7+": _fc_m >= 7.0}
    RATCHET.update(
        cref=_cref, cref_name=(SAVED_AFTER[_cref] if _cref < len(SAVED_AFTER) else ""),
        dleaves_total=float(_H.sum(axis=1).mean()),
        dleaves={k: float(_H[:, m].sum(axis=1).mean()) for k, m in _bins_m.items()},
        leaves_ref_bins={k: float(_Href[:, m].sum(axis=1).mean()) for k, m in _bins_m.items()},
        dband_lnl=({k: float(_dB[:, m].sum(axis=1).mean()) for k, m in _bins_m.items()}
                   if _dB is not None else {}),
    )
except Exception as _e:
    MISSING.append(f"leaf-delta panel unavailable: {type(_e).__name__}: {_e}")

if RATCHET:
    _R = RATCHET
    cap_fgA = (
        f"<strong>The ratchet, read off the store.</strong> Row {_R['ref']} is the last "
        f"row before the first nudge; nudge rows {_R['nudges']}, release rows "
        f"{_R['releases']}. Cold log-likelihood (walker mean) relative to that row: "
        f"minimum {_R['dll_min']:+,.0f} at row {_R['dll_min_row']}, "
        f"{_R['dll_before_release']:+,.0f} on the last held row, "
        f"{_R['dll_first_release']:+,.0f} right after the first release, "
        f"{_R['dll_last']:+,.0f} at the last row. Cold leaves per walker: "
        f"{_R['leaves_ref']:.0f} before the nudge, {_R['leaves_before_release']:.0f} on "
        f"the last held row, {_R['leaves_last']:.0f} now. Foreground power relative to the "
        f"pre-nudge fit at 2 / 3 / 4 / 5 mHz: held at "
        + " / ".join(f"{v:.2f}" for v in _R.get("ratio_held", {}).values())
        + ", now "
        + " / ".join(f"{v:.2f}" for v in _R["ratio_last"].values())
        + ". A release that climbs back to 1 says the model was honest there; one that stays "
        "below says the fit had been carrying resolvable power; one that goes ABOVE 1 says "
        "the released fit reshaped the curve rather than returning to it.")
    if "cref" in _R:
        cap_fgC = (
            f"<strong>Where the leaves went.</strong> Cold-chain leaf histogram at the last row "
            f"minus row {_R['cref']}"
            + (f" (saved after {_R['cref_name']})" if _R.get("cref_name") else "")
            + (" &mdash; the earliest row of this snapshot that still carries coordinates, "
               "not the pre-nudge row itself" if _R["cref"] != _R["ref"] else "")
            + f". Net leaves per walker: {_R['dleaves_total']:+.0f} in total; by band (mHz): "
            + ", ".join(f"{k}: {v:+.0f} on {_R['leaves_ref_bins'][k]:.0f}"
                        for k, v in _R["dleaves"].items())
            + ". The amber curve is the cold-mean foreground of the last row over the pre-nudge "
            "one."
            + (" Lower panel: the per-band cold log-likelihood change over the same rows. This "
               "is the DATA term of the band likelihood only (no log-determinant), so it tracks "
               "the change in whitened residual power in each band: it goes down wherever the "
               "noise curve was lowered or sources were removed and up wherever the curve was "
               "raised or sources were added, and it does NOT by itself say whether the new "
               "curve fits better: "
               + ", ".join(f"{k}: {v:+,.0f}" for k, v in _R["dband_lnl"].items())
               if _R.get("dband_lnl") else ""))
if FGW:
    _F = FGW
    _fp = (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)
    cap_fgB = (
        "<strong>Is the residual what the noise model says it is?</strong> Top: the "
        "whitened residual power per GB band (1 = the fitted instrument + foreground "
        f"exactly) for two cold walkers of the last stored row, each against its own fit: "
        f"walker {_F['walker']}, the highest log-likelihood (cyan), and walker "
        f"{_F['walker_min']}, the lowest (amber). Running mean at 1 / 2 / 3 / 3.5 / 4 / "
        "4.5 / 5 / 6 / 8 mHz: max lnL "
        + " / ".join(f"{_F['rat_at'][f_]:.2f}" for f_ in _fp)
        + ", min lnL "
        + " / ".join(f"{_F['rat_at_min'][f_]:.2f}" for f_ in _fp)
        + f"; band means over 1-2 / 3-5 / 6-10 mHz are {_F['rat_12']:.2f} / "
        f"{_F['rat_35']:.2f} / {_F['rat_610']:.2f} (max lnL) and {_F['rat_12_min']:.2f} / "
        f"{_F['rat_35_min']:.2f} / {_F['rat_610_min']:.2f} (min lnL), and the running mean "
        f"spans a factor {_F['spread_35']:.2f} / {_F['spread_35_min']:.2f} across 3-5 mHz. "
        "Against instrument-only the max-lnL residual reads "
        + " / ".join(f"{_F['inst_at'][f_]:.1f}" for f_ in (2.0, 3.0, 4.0, 5.0))
        + " at 2 / 3 / 4 / 5 mHz (dashed: that is what the foreground term carries), and "
        "against instrument + the FittedHT estimate "
        + " / ".join(f"{_F['est_at'][f_]:.2f}" for f_ in (2.0, 3.0, 4.0, 5.0))
        + " (dash-dot: the estimate, used as the noise, would leave that much excess). "
        "Second and third rows: Gaussianity per band. For Gaussian noise the mean/median of "
        "the whitened power is 1.443 and 1% of bins exceed 4.605; for the max / min lnL "
        f"walker the medians of the per-band mean/median are {_F['mm_12']:.2f} / "
        f"{_F['mm_12_min']:.2f} (1-2 mHz), {_F['mm_35']:.2f} / {_F['mm_35_min']:.2f} (3-5 mHz), "
        f"{_F['mm_610']:.2f} / {_F['mm_610_min']:.2f} (6-10 mHz), and the tail fractions "
        f"{100 * _F['tail_12']:.1f}% / {100 * _F['tail_35']:.1f}% / {100 * _F['tail_610']:.1f}% "
        f"(max lnL) and {100 * _F['tail_12_min']:.1f}% / {100 * _F['tail_35_min']:.1f}% / "
        f"{100 * _F['tail_610_min']:.1f}% (min lnL); {_F['bands_nongauss_35']} / "
        f"{_F['bands_nongauss_35_min']} of {_F['bands_35']} bands in 3-5 mHz have more than "
        "2% of their bins in the tail. A band at mean 1 with a heavy tail holds its power in a few "
        "loud bins (unresolved or missed lines) rather than a Gaussian floor; a smooth drift of "
        "the mean away from 1 that follows frequency is the rigid tanh x power-law failing to "
        "follow the residual's shape. Bottom: the share of the model noise that is foreground. "
        "<em>Caveats.</em> (1) " + _F["src_txt"] +
        "(2) The noise curves here are the "
        "static equal-arm model on each walker's parameters; the run multiplies the foreground "
        "by a tabulated time modulation (GalForTimeModulation) that these curves do not carry, "
        "so in foreground-dominated bands the page under-states the run's effective noise by "
        "the window-mean of that table's XX element minus one, times the foreground share.")

# ======================= GB RECOVERY (the science block) ====================
# ONE FROZEN DENOMINATOR. ``gb_truth_3to21.npz`` holds every catalogue GB that
# survives the kappa_max amplitude prefilter, with its exact optimal SNR under
# the run's own fitted noise at the iteration it was built at and its full
# parameter vector in both the run's 9-column sampling basis and GBGPU's
# physical basis. Its detectable (SNR>7) subset is used as the denominator of
# EVERY recovery statement on this page. It is deliberately frozen:
# "detectable" moves as the foreground estimate drops, and a denominator that
# moves with the numerator cannot measure progress.
#
# Everything downstream is recomputed HERE, at the last stored iteration --
# none of it is read from the iteration-15 caches the earlier page was built
# on, which were three times smaller in model size.
SCI = {}
TRU = None
# Fallbacks for the live-noise SNR rescale (populated inside the TRU-truthy
# branch below). Keeping module-level defaults means the F4 caption and
# the zoomable-canvas tier code can reference them unconditionally.
T_SNR_LIVE = None
_T_SNR_LIVE_FULL = None
_LIVE_SNR_NOTE = ""
for _tp in (os.path.join(RUN_DIR, "gb_truth_3to21.npz"), "gb_truth_3to21.npz"):
    if os.path.exists(_tp):
        try:
            TRU = np.load(_tp)
        except Exception:
            TRU = None
        break

# WHICH EPHEMERIS IS THE TRUTH SET ON? (2026-09-23.) This page used to run in
# a silently MIXED state: the recovery block below builds waveforms on the
# mojito L1 orbits whenever it can reach them, while the detectable
# denominator, the injected-SNR axis of the recovery curve and column 1 of
# the #detect table all come straight out of this npz -- which build_truth.py
# built on the ANALYTIC ephemeris until today. Measured at 7.44-7.60 mHz on
# the 6mo store, that is not a rounding difference: optimal SNRs moved
# 7.96 -> 8.69 and 32.81 -> 42.67, and template overlaps 0.052 -> 0.580.
# build_truth now stamps ``orbits``; an unstamped file predates the stamp and
# is therefore analytic.
TRU_ORBITS = "unstamped (pre-2026-09-23 => analytic)"
if TRU is not None:
    try:
        TRU_ORBITS = str(TRU["orbits"]) if "orbits" in TRU.files else TRU_ORBITS
    except Exception:
        pass
    if not TRU_ORBITS.startswith("mojito"):
        MISSING.append(
            "the truth set was built on the ANALYTIC ephemeris "
            f"({TRU_ORBITS}) while this page's overlap/SNR block uses the "
            "injected mojito L1 orbits. Everything keyed to the truth set -- "
            "the detectable denominator, completeness/purity, the injected-SNR "
            "axis and column 1 of the detectability table -- carries that "
            "ephemeris and is NOT comparable to column 2 or to the overlaps. "
            "Rebuild with scripts/diagnostics/build_truth.py (it now defaults "
            "to the injected orbits).")

# SCI_TOBS already defined near line 200 after file open
SCI_DF = 1.0 / SCI_TOBS

# THE BAND COMES FROM THE TRUTH SET (2026-08-23). It used to be hardcoded at
# 3-21.94 mHz here AND spelled out in three captions, while the truth npz
# already stamped a ``band`` field -- so widening the truth set to the run's
# real GB band (band_edges[0] = 0.5556 mHz, not 3 mHz) would have left the
# page quoting the old band over the new counts, which is worse than either.
# Read the stamp, with the old hardcoded pair as the fallback for a pre-stamp
# npz -- the same backward-compatible pattern as the ``tobs`` stamp below.
# Read BEFORE the Tobs guard, which sets TRU to None: FLO/FHI still label the
# population panels on a run whose recovery block is skipped.
FLO, FHI = 3e-3, 21.94e-3          # fallback: the pre-2026-08-23 band
if TRU is not None and "band" in getattr(TRU, "files", []):
    try:
        _tband = np.asarray(TRU["band"], float).reshape(-1)
        if _tband.size >= 2 and 0 < _tband[0] < _tband[1]:
            FLO, FHI = float(_tband[0]), float(_tband[1])
    except Exception:
        pass
BAND_TXT = f"{_mhz(FLO)}&ndash;{_mhz(FHI)} mHz"       # HTML captions
TOL_BINS = 2.0
NBAND = (FHI - FLO) / SCI_DF       # FD bins in the band

# Set the overlap-match caption fragments UP FRONT so they exist even when
# the recovery-section machinery below never runs (TRU missing, Tobs
# mismatch, or the RPHYS try-block failed). The values are overwritten in
# the RPHYS block below once MM is actually computed.
MATCH_CRIT_HTML = (
    f"&ldquo;Matched&rdquo; here = phase-maximised overlap "
    f"&ge; {MATCH_MM_THRESH:.2f} on top of a "
    f"{TOL_BINS:.0f}-df-bin f<sub>0</sub> pair "
    f"(threshold <code>GF_MONITOR_MATCH_MM</code>).")
MATCH_CRIT_TXT = f"matched = phase-max overlap >= {MATCH_MM_THRESH:.2f}"

# The truth set is tied to A Tobs (SNRs, and the FD bin the match tolerance is
# quoted in) -- not to THE 3-month one: ``build_truth.py`` now stamps the
# observation time it was built at into the npz. Compare against that stamp,
# defaulting to the 3-month value for the pre-stamp npz files, so a 1-year set
# passes on a 1-year run while a 3-month set on a 1-year run still refuses
# rather than silently comparing a 1-year model against 3-month detectability.
TRU_TOBS = 7776000.0
if TRU is not None and "tobs" in getattr(TRU, "files", []):
    try:
        TRU_TOBS = float(np.asarray(TRU["tobs"]).reshape(-1)[0])
    except Exception:
        TRU_TOBS = 7776000.0
if TRU is not None and abs(SCI_TOBS - TRU_TOBS) > 1.0:
    MISSING.append(
        "recovery panels skipped: the frozen detectability set is built for a "
        f"{TRU_TOBS / 86400.0:.4g}-day observation and this run is "
        f"{SCI_TOBS / 86400.0:.4g} days.")
    TRU = None

# ============ RATCHET SCORECARD (user request 2026-10-02) ====================
# "the ratchet should have done this or shown something along these lines":
# the number the ratchet exists to drive down is the count of DETECTABLE
# catalogue sources (SNR > 7 in the frozen set) that no cold walker holds, per
# band -- and the power those sources carry is what the fitted foreground is
# made of. Two panels: (D) that count per band group against stored row, on
# every row that still carries coordinates; (E) the fitted foreground share of
# the model noise per 0.5 mHz against the periodogram power of the catalogue
# sources the model has NOT resolved, split by their SNR. A source confined to
# ~one FD bin carries SNR^2 S_X / 4 of X-channel power (optimal SNR is over
# the two independent TDI combinations), so the unresolved set's mean PSD over
# a band of N_bin bins is S_tot * sum(SNR^2) / (4 N_bin).
RSC = {}
try:
    if TRU is None:
        raise RuntimeError("no truth set in this snapshot")
    _tf0_m = np.asarray(TRU["f0"], float).reshape(-1) * 1e3          # Hz -> mHz
    _tsnr_r = np.asarray(TRU["snr"], float).reshape(-1)
    _det7 = _tsnr_r > 7.0
    _tol_m = 2.0 / SCI_TOBS * 1e3                                     # two FD bins
    _chg_r = g["chain/gb"]; _ing_r = g["inds/gb"]
    _rows_r = [i for i in range(NIT) if np.any(_chg_r[i, 0, 0, :, :, 1] != 0.0)]
    if not _rows_r:
        raise RuntimeError("no coordinate-bearing rows")

    def _hits_r(i):
        """how many cold walkers hold a leaf within two bins of each truth f0"""
        h = np.zeros(_tf0_m.size, dtype=int)
        for _w in range(nwalk):
            lf = np.sort(_chg_r[i, 0, 0, _w][_ing_r[i, 0, 0, _w].astype(bool)][:, 1])
            if lf.size == 0:
                continue
            j = np.searchsorted(lf, _tf0_m)
            near = np.full(_tf0_m.size, np.inf)
            for off in (-1, 0):
                jj = np.clip(j + off, 0, lf.size - 1)
                near = np.minimum(near, np.abs(lf[jj] - _tf0_m))
            h += (near <= _tol_m)
        return h

    _groups_r = [("2-3", 2.0, 3.0), ("3-4", 3.0, 4.0), ("4-5", 4.0, 5.0), ("5-7", 5.0, 7.0)]
    _none_r = {k: [] for k, *_ in _groups_r}
    _one_r = {k: [] for k, *_ in _groups_r}
    _h_last = None
    for i in _rows_r:
        h = _hits_r(i)
        _h_last = h
        for k, lo, hi in _groups_r:
            m = _det7 & (_tf0_m >= lo) & (_tf0_m < hi)
            _none_r[k].append(int(np.sum(m & (h == 0))))
            _one_r[k].append(int(np.sum(m & (h == 1))))
    _tot7_r = {k: int(np.sum(_det7 & (_tf0_m >= lo) & (_tf0_m < hi))) for k, lo, hi in _groups_r}
    # one y axis for both rows (user request 2026-10-02): "held by none" and
    # "held by one" read against each other only on the same scale
    fig, ax = plt.subplots(2, 1, figsize=(11.6, 7.2), sharex=True, sharey=True)
    for (k, lo, hi), c in zip(_groups_r, (FG, VIOLET, CYAN, AMBER)):
        ax[0].plot(_rows_r, _none_r[k], "o-", color=c, lw=1.4, ms=3.5,
                   label=f"{k} mHz (of {_tot7_r[k]} catalogue sources with SNR > 7)")
        ax[1].plot(_rows_r, _one_r[k], "o-", color=c, lw=1.4, ms=3.5)
    for _r in RATCHET.get("nudges", []):
        for a_ in ax:
            a_.axvline(_r, color=AMBER, lw=0.9, ls=(0, (3, 2)))
    for _r in RATCHET.get("releases", []):
        for a_ in ax:
            a_.axvline(_r, color=GREEN, lw=0.9, ls=(0, (3, 2)))
    ax[0].set_ylabel("SNR > 7 catalogue sources\nheld by NO cold walker", fontsize=9)
    ax[1].set_ylabel("held by exactly\nONE cold walker", fontsize=9)
    ax[1].set_xlabel("stored row (coordinate-bearing rows only; amber = nudge, green = release)")
    ax[0].legend(fontsize=7.5, loc="upper right")
    for a_ in ax:
        a_.set_ylim(bottom=0)
    fig.suptitle("ratchet scorecard: detectable catalogue sources still unresolved, per band "
                 "(a leaf within two FD bins of the catalogue f0 counts as held)",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_b64(fig, "fg_missed_vs_row")
    RSC.update(rows=(int(_rows_r[0]), int(_rows_r[-1])),
               none_first={k: v[0] for k, v in _none_r.items()},
               none_last={k: v[-1] for k, v in _none_r.items()},
               one_last={k: v[-1] for k, v in _one_r.items()}, tot7=_tot7_r)
    _m35 = (_tf0_m >= 3.0) & (_tf0_m < 5.5) & _det7 & (_h_last == 0)
    if np.any(_m35):
        RSC["snr_pct_35"] = [float(x) for x in np.percentile(_tsnr_r[_m35], [10, 50, 90])]
        RSC["n_none_35"] = int(_m35.sum())

    # ---- (E) what the fitted foreground is made of, per 0.5 mHz --------------
    from lisatools.sensitivity import get_sensitivity as _gs_r, X2TDISens as _X2_r
    from lisatools import detector as _lm_r
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as _HT_r
    _ed_r = np.arange(2.0, 7.0 + 1e-9, 0.5)
    _fc_r = 0.5 * (_ed_r[:-1] + _ed_r[1:])
    _nbin_r = 0.5e-3 * SCI_TOBS
    # the max- and min-lnL cold walkers of the last stored row, not the mean
    _WSH_R = (int(np.argmax(ll[-1])), int(np.argmin(ll[-1])))
    _share_r = np.zeros((2, _fc_r.size))
    for _k, _w in enumerate(_WSH_R):
        _pmw = psd_cold[-1, _w]; _gmw = gal_cold_phys[-1, _w]
        _lmw = _lm_r.LISAModel(float(_pmw[0]) ** 2, float(_pmw[1]) ** 2,
                               _lm_r.DefaultOrbits(), "sampled")
        _si = np.asarray(_gs_r(_fc_r * 1e-3, sens_fn=_X2_r, model=_lmw, stochastic_params=()), float)
        _st = np.asarray(_gs_r(_fc_r * 1e-3, sens_fn=_X2_r, model=_lmw,
                               stochastic_params=tuple(_gmw), stochastic_function=_HT_r), float)
        _share_r[_k] = 1.0 - _si / np.maximum(_st, 1e-60)
    _share_m = _share_r[0]                              # max lnL: what the red labels read
    _un = _h_last < 2                                   # not held by >= 2 walkers
    _p7 = np.zeros(_fc_r.size); _p57 = np.zeros(_fc_r.size); _p5 = np.zeros(_fc_r.size)
    for i in range(_fc_r.size):
        m = (_tf0_m >= _ed_r[i]) & (_tf0_m < _ed_r[i + 1]) & _un
        _p7[i] = np.sum(_tsnr_r[m & (_tsnr_r > 7)] ** 2) / (4 * _nbin_r)
        _p57[i] = np.sum(_tsnr_r[m & (_tsnr_r > 5) & (_tsnr_r <= 7)] ** 2) / (4 * _nbin_r)
        _p5[i] = np.sum(_tsnr_r[m & (_tsnr_r <= 5)] ** 2) / (4 * _nbin_r)
    fig, ax = plt.subplots(1, 1, figsize=(11.6, 5.2))
    _wd = 0.14
    ax.bar(_fc_r - 0.15, _p5, width=_wd, color=DIM, label="unresolved catalogue sources, SNR <= 5")
    ax.bar(_fc_r - 0.15, _p57, width=_wd, bottom=_p5, color=VIOLET, label="unresolved, SNR 5-7")
    ax.bar(_fc_r - 0.15, _p7, width=_wd, bottom=_p5 + _p57, color=RED,
           label="unresolved, SNR > 7 (the ratchet's targets)")
    for _k, (_w, _wc, _wl) in enumerate(((_WSH_R[0], CYAN, "max lnL"),
                                         (_WSH_R[1], AMBER, "min lnL"))):
        ax.bar(_fc_r + 0.15 * _k, _share_r[_k], width=_wd, color=_wc, alpha=0.85,
               label=f"fitted foreground share of the model noise, walker {_w} ({_wl}), "
                     f"row {NIT - 1}")
    for i in range(_fc_r.size):
        if _share_m[i] > 0.02:
            ax.annotate(f"{100 * _p7[i] / _share_m[i]:.0f}%",
                        xy=(_fc_r[i], max(_share_r[:, i].max(), _p5[i] + _p57[i] + _p7[i]) + 0.02),
                        ha="center", fontsize=7, color=RED)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("frequency [mHz]  (red labels: the SNR > 7 misses' share of the fitted foreground)")
    ax.set_ylabel("fraction of the model noise S_tot (TDI X)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("what the fitted foreground is made of: unresolved catalogue power vs the fit, "
                 "per 0.5 mHz (truth set is amplitude-prefiltered: a lower bound)",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_b64(fig, "fg_truth_decomp")
    RSC.update(share_walkers=_WSH_R,
               share={f"{_ed_r[i]:.1f}-{_ed_r[i + 1]:.1f}": float(_share_m[i]) for i in range(_fc_r.size)},
               share_min={f"{_ed_r[i]:.1f}-{_ed_r[i + 1]:.1f}": float(_share_r[1, i]) for i in range(_fc_r.size)},
               unres={f"{_ed_r[i]:.1f}-{_ed_r[i + 1]:.1f}": float(_p5[i] + _p57[i] + _p7[i]) for i in range(_fc_r.size)},
               miss7_frac={f"{_ed_r[i]:.1f}-{_ed_r[i + 1]:.1f}": (float(_p7[i] / _share_m[i]) if _share_m[i] > 0.02 else float("nan"))
                           for i in range(_fc_r.size)})
    del _chg_r, _ing_r
except Exception as _e:
    MISSING.append(f"ratchet scorecard unavailable: {type(_e).__name__}: {_e}")

# ---- (G) where the RJ births actually landed, per leg and per 0.5 mHz --------
# ``band_num_proposed_rj`` / ``band_num_accepted_rj`` are CUMULATIVE per
# (row, band, rung) counters; the difference between consecutive stored rows
# is that leg's traffic. Under search legs one row lands per leg, so this is
# births (the in_model_fstat leg) and deaths (the in_model_removal leg) per
# band, which is the direct test of "did the births go where the unresolved
# detectable sources are". An extract carries the counters only in its keep
# window, so a short snapshot shows the last few legs.
RJB = {}
try:
    _pr = sub["gb/band_num_proposed_rj"]; _ac = sub["gb/band_num_accepted_rj"]
    _rows_c = [i for i in range(1, NIT) if np.any(_pr[i, :, 0] > 0) and np.any(_pr[i - 1, :, 0] > 0)]
    if not _rows_c:
        raise RuntimeError("fewer than two rows carry the per-band RJ counters")
    _ed_g = np.arange(2.0, 7.0 + 1e-9, 0.5)
    _fc_g = 0.5 * (_ed_g[:-1] + _ed_g[1:])
    _bcen = 0.5 * (band_edges[:-1] + band_edges[1:]) * 1e3
    _gi = np.searchsorted(_ed_g, _bcen, side="right") - 1
    _ing_g = (_gi >= 0) & (_gi < _fc_g.size)
    _legs = []
    for i in _rows_c:
        dp = _pr[i, :, 0] - _pr[i - 1, :, 0]
        da = _ac[i, :, 0] - _ac[i - 1, :, 0]
        if not np.any(dp > 0):
            continue
        _legs.append((i, (SAVED_AFTER[i] if i < len(SAVED_AFTER) and SAVED_AFTER[i] else f"row {i}"),
                      np.bincount(_gi[_ing_g], weights=dp[_ing_g], minlength=_fc_g.size),
                      np.bincount(_gi[_ing_g], weights=da[_ing_g], minlength=_fc_g.size)))
    if not _legs:
        raise RuntimeError("no leg with RJ traffic in the counters' window")
    fig, ax = plt.subplots(2, 1, figsize=(11.6, 7.0), sharex=True)
    _nl = len(_legs); _wd = 0.44 / _nl
    _pal = [CYAN, GREEN, AMBER, VIOLET, RED, FG, DIM] * 4
    for k, (i, nm, dp, da) in enumerate(_legs):
        _x = _fc_g - 0.22 + (k + 0.5) * _wd
        ax[0].bar(_x, da, width=_wd, color=_pal[k], label=f"row {i} (after {nm})")
        ax[1].bar(_x, 100 * da / np.maximum(dp, 1), width=_wd, color=_pal[k])
    ax[0].set_ylabel("accepted RJ moves, cold rung,\nall walkers, per 0.5 mHz", fontsize=9)
    ax[1].set_ylabel("acceptance [%]", fontsize=9)
    ax[1].set_xlabel("frequency [mHz]")
    ax[0].legend(fontsize=7.5, loc="upper right", ncols=2)
    fig.suptitle("RJ traffic per leg (cumulative per-band counters differenced between stored rows)",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_b64(fig, "fg_rj_per_band")
    RJB.update(legs=[(int(i), nm, float(dp.sum()), float(da.sum())) for i, nm, dp, da in _legs])
except Exception as _e:
    MISSING.append(f"RJ-per-band panel unavailable: {type(_e).__name__}: {_e}")

if RSC:
    _S = RSC
    cap_fgD = (
        f"<strong>The ratchet's scorecard.</strong> Catalogue sources with SNR &gt; 7 (frozen set) "
        f"that NO cold walker holds, rows {_S['rows'][0]} to {_S['rows'][1]}: "
        + ", ".join(f"{k} mHz {_S['none_first'][k]} &rarr; {_S['none_last'][k]} of {_S['tot7'][k]}"
                    for k in _S["tot7"])
        + ". Held by exactly one walker at the last row: "
        + ", ".join(f"{k}: {_S['one_last'][k]}" for k in _S["tot7"])
        + " (those are a cross-walker exchange away from resolved)."
        + (f" The {_S['n_none_35']} still unresolved at 3&ndash;5.5 mHz have SNR "
           f"{_S['snr_pct_35'][0]:.1f} / {_S['snr_pct_35'][1]:.1f} / {_S['snr_pct_35'][2]:.1f} "
           "at the 10th / 50th / 90th percentile: threshold sources." if "snr_pct_35" in _S else "")
        + " SNRs are the frozen set's, under the noise it was built at; an extract carries "
        "coordinates only in its keep window, so earlier rows need an earlier snapshot.")
    cap_fgE = (
        "<strong>What the fitted foreground is made of.</strong> Per 0.5 mHz: the fitted "
        "foreground's share of the model noise for the max-lnL (cyan, walker "
        f"{_S['share_walkers'][0]}) and min-lnL (amber, walker {_S['share_walkers'][1]}) "
        "cold walkers of the last stored row, against the "
        "X-channel periodogram power of the catalogue sources the model has not resolved "
        "(stacked by SNR; a source in one FD bin carries SNR&sup2; S_X / 4). Fit (max lnL / "
        "min lnL) vs unresolved total: "
        + ", ".join(f"{k}: {_S['share'][k]:.2f} / {_S['share_min'][k]:.2f} vs {_S['unres'][k]:.2f}"
                    for k in _S["share"])
        + ". Share of the max-lnL walker's fitted foreground carried by the SNR &gt; 7 misses: "
        + ", ".join(f"{k}: {100 * v:.0f}%" for k, v in _S["miss7_frac"].items() if np.isfinite(v))
        + ". Where the two bars agree the fitted foreground IS the unresolved galaxy; the red "
        "fraction is the part a search could still remove.")


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
    ``None`` when that row cannot be read (see ``_row``).
    """
    f0 = _row("chain/gb", (it_, 0, 0, w, slice(None), 1))
    if f0 is None:
        return None
    return (f0 * 1e-3)[gb_inds[it_, w]]


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
        if _fv is None:
            # unreadable row: hold the previous row's counts rather than plot
            # a drop to zero; the row is named on the page via _TORN_ROWS
            if _i:
                n_all[_i] = n_all[_i - 1]; n_band[_i] = n_band[_i - 1]
                n_match[_i] = n_match[_i - 1]
            continue
        n_all[_i] = _fv.size
        _fv = _fv[(_fv >= FLO) & (_fv <= FHI)]
        n_band[_i] = _fv.size
        n_match[_i] = _match_pairs(_fv, T_F0, _tol)[0].size
    _nz = np.nonzero(n_all > 0)[0]
    IT0 = int(_nz[0]) if _nz.size else 0          # the GB-search origin
    NGBIT = NIT - 1 - IT0

    # ---- the last stored iteration, in full ------------------------------
    WB = int(_wb[-1])
    REC9 = gb_chain_cold[WB][gb_alive_last[WB]]           # row NIT-1, read once
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

    # ---- live-noise rescale of catalogue SNR (user ruling 2026-10-05) ----
    # The missed-detection X markers in the F4 right panel and the
    # zoomable canvas are TIERED by injected SNR (yellow 7-10, orange
    # 10-20, red >20). The frozen ``gb_truth_3to21.npz`` carries the
    # SNR evaluated under the TRUTH-BUILD iteration's PSD + foreground
    # (``psd_params``, ``galfor_params`` fields of the npz), and that
    # noise is NOT the noise of the last stored iteration -- a run that
    # has pulled the foreground down keeps classifying its loudest misses
    # with the old PSD unless the SNR is rescaled.
    #
    # For a quasi-monochromatic GB source the catalogue SNR scales as
    # SNR = h(f0) / sqrt(S(f0)) up to a weak f-derivative term that
    # cancels in a ratio, so rescaling by sqrt(S_truth(f0) / S_live(f0))
    # at the catalogue f0 is the right one-liner (O(N) in catalogue
    # size, no waveform recomputation). The frozen ``TRU["det"]`` set is
    # LEFT ALONE -- the headline denominator and completeness numbers
    # are deliberately pinned to one iteration (CLAUDE.md: "frozen
    # denominator, stated once"), so this rescale touches ONLY the X
    # marker tiering, not which sources count as detectable.
    T_SNR_LIVE = T_SNR               # fallback: frozen SNR
    _T_SNR_LIVE_FULL = None          # for the zoomable canvas (full cat)
    _LIVE_SNR_NOTE = ""
    try:
        from lisatools import detector as _lisa_models_lsnr
        from lisatools.sensitivity import (get_sensitivity as _gs_lsnr,
                                            A2TDISens as _A2_lsnr)
        from lisatools.stochastic import (
            HyperbolicTangentGalacticForeground as _HTGF_lsnr)
        _wb_last = int(np.argmax(ll[-1]))
        _psd_live = np.asarray(psd_cold[-1, _wb_last], float)
        _gal_live = np.asarray(gal_cold_phys[-1, _wb_last], float)
        _psd_tru = (np.asarray(TRU["psd_params"], float)
                    if "psd_params" in TRU.files else None)
        _gal_tru = (np.asarray(TRU["galfor_params"], float)
                    if "galfor_params" in TRU.files else None)
        if (_psd_tru is None or _gal_tru is None or _psd_tru.size != 2
                or _gal_tru.size != 5):
            _LIVE_SNR_NOTE = (" (fallback: frozen SNR -- the truth npz was "
                              "built before psd_params/galfor_params were "
                              "stamped)")
            raise RuntimeError("truth npz missing psd/galfor stamps")
        _f0_hz_full = np.asarray(TRU["f0"], float)
        _mdl_live = _lisa_models_lsnr.LISAModel(
            _psd_live[0] ** 2, _psd_live[1] ** 2,
            _lisa_models_lsnr.DefaultOrbits(), "live-last-maxLL")
        _mdl_tru = _lisa_models_lsnr.LISAModel(
            _psd_tru[0] ** 2, _psd_tru[1] ** 2,
            _lisa_models_lsnr.DefaultOrbits(), "truth-build")
        _nk_live = dict(model=_mdl_live, stochastic_params=tuple(_gal_live),
                        stochastic_function=_HTGF_lsnr)
        _nk_tru = dict(model=_mdl_tru, stochastic_params=tuple(_gal_tru),
                       stochastic_function=_HTGF_lsnr)
        _S_live = np.asarray(_gs_lsnr(_f0_hz_full, sens_fn=_A2_lsnr,
                                       **_nk_live), float)
        _S_tru = np.asarray(_gs_lsnr(_f0_hz_full, sens_fn=_A2_lsnr,
                                      **_nk_tru), float)
        _ratio = np.sqrt(np.maximum(_S_tru, 1e-60)
                         / np.maximum(_S_live, 1e-60))
        _T_SNR_LIVE_FULL = np.asarray(TRU["snr"], float) * _ratio
        T_SNR_LIVE = _T_SNR_LIVE_FULL[_sel]
        # one-line summary: how many tier swaps did the rescale cause?
        _old_tier = np.where(T_SNR >= 20, 2, np.where(T_SNR >= 10, 1, 0))
        _new_tier = np.where(T_SNR_LIVE >= 20, 2,
                              np.where(T_SNR_LIVE >= 10, 1, 0))
        _n_tier_swap = int((_old_tier != _new_tier).sum())
        _LIVE_SNR_NOTE = (
            f" (live rescale: {_n_tier_swap} of {int(_sel.sum())} "
            f"catalogue sources changed SNR tier vs the truth-build noise; "
            f"iter {NIT - 1} walker {_wb_last})")
    except Exception as _e:
        if not _LIVE_SNR_NOTE:
            MISSING.append(
                "live-noise SNR rescale for the X marker tiers unavailable "
                f"({type(_e).__name__}: {_e}); tiers fall back to the "
                "truth-build frozen SNR.")

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
        _gal_p = np.median(gal_cold_phys[-1], axis=0)
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

        # L1 orbits when available (mandatory for high-f accuracy); the
        # analytic fallback is fine below a few mHz only.
        _orb = globals().get("_L1_ORB_CPU")
        if _orb is None:
            MISSING.append(
                "SNR machinery fell back to DefaultOrbits (no mojito L1 "
                "orbit file reachable): SNRs/overlaps above ~5 mHz carry an "
                "annual-Doppler phase error.")
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

    # ---- overlap-refined match: the criterion the science panels use ----
    # Above, MI/TI/MATCHED/FOUND come from the 2-df-bin proxy match: a pair
    # is anything within TOL_BINS FD bins of f0. That is what the F2 progress
    # panel and completeness/purity KPIs use, because they run every stored
    # row and can't afford a waveform per pair. Here we tighten "matched" to
    # a real phase-maximised overlap threshold (MATCH_MM_THRESH, default
    # 0.8) for the panels that only need it at the last row -- F4 and the
    # zoomable plot. Everything upstream keeps the 2-df definition.
    if MM is not None and MM.size:
        _kept = np.asarray(MM, float) >= MATCH_MM_THRESH
        _MI_ov, _TI_ov = MI[_kept], TI[_kept]
        MATCHED_MM = np.zeros(REC9.shape[0], bool); MATCHED_MM[_MI_ov] = True
        FOUND_MM   = np.zeros(NDET, bool);           FOUND_MM[_TI_ov]  = True
        MATCH_CRIT_HTML = (
            f"&ldquo;Matched&rdquo; here = phase-maximised, noise-weighted "
            f"overlap &ge; {MATCH_MM_THRESH:.2f} on top of a "
            f"{TOL_BINS:.0f}-df-bin f<sub>0</sub> pair "
            f"(threshold <code>GF_MONITOR_MATCH_MM</code>).")
        MATCH_CRIT_TXT = f"matched = phase-max overlap >= {MATCH_MM_THRESH:.2f}"
    else:
        MATCHED_MM = MATCHED
        FOUND_MM   = FOUND
        MATCH_CRIT_HTML = (
            f"&ldquo;Matched&rdquo; here = within {TOL_BINS:.0f} "
            f"f<sub>0</sub> bins ONLY &mdash; the phase-maximised overlap "
            f"machinery was unavailable in this snapshot, so this panel "
            f"falls back to the 2-df proxy.")
        MATCH_CRIT_TXT = f"matched = within {TOL_BINS:.0f} df bins (2-df proxy)"

    # ---- HEADLINE NUMBERS = the overlap-refined match when it exists ------
    # (user request 2026-09-30). The proxy values are kept alongside, named
    # as such, for the caption that has to explain the per-iteration curves.
    SCI.update(n_match_proxy=int(MI.size),
               completeness_proxy=MI.size / NDET,
               purity_proxy=MI.size / max(REC9.shape[0], 1))
    if MM is not None and MM.size:
        TI_OV = _TI_ov
        SCI.update(n_match=int(MATCHED_MM.sum()),
                   completeness=float(FOUND_MM.sum()) / NDET,
                   purity=float(MATCHED_MM.sum()) / max(REC9.shape[0], 1),
                   match_is_overlap=True)
    else:
        TI_OV = TI
        SCI.update(match_is_overlap=False)

    # ---- cross-arm cache -------------------------------------------------
    # The v2/v3 comparison must be made on GB-SEARCH iterations, not absolute
    # ones (v2's first GB leaf lands at iteration 5, v3's at 16), so each arm
    # publishes its own series keyed by that origin and every comparison panel
    # subtracts it.
    # Every run kind gets its OWN arm cache (2026-08-22): the old
    # {v3, v4}-else-v2 map sent v5, v6 AND the 1-yr run all to
    # gf_arm_v2.npz, silently clobbering the shared v2 arm. The map is
    # ``_identity.arm_tag`` since 2026-10-07 (shared with the lean twin).
    ARM_TAG = _arm_tag(RUN_KIND)
    try:
        # ``ti`` = the truth indices FOUND at the last row under the page's
        # headline criterion (overlap-refined when available): F6's
        # completeness-vs-SNR reads it. ``n_match`` stays the per-row proxy.
        np.savez(f"gf_arm_{ARM_TAG}.npz", n_all=n_all, n_band=n_band,
                 n_match=n_match, it0=IT0, ti=TI_OV, mm=MM,
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
    # Every arm needs its OWN hue (2026-08-22): the per-run arm tags
    # (gf_arm_3mo_v5 / _3mo_v6 / _1yr_v5) all fell through to the GREEN
    # default, drawing indistinguishable comparison lines. Known tags get
    # fixed hues; anything new draws from a deterministic cycle keyed by
    # sorted position so two unknown arms can never share a colour.
    ARM_COL = {
        "v2": CYAN, "v3": AMBER, "v4": GREEN,
        "3mo_v5": "#FF7BAC",   # pink
        "3mo_v6": VIOLET,
        "1yr_v5": "#F2E14C",   # yellow
    }
    _ARM_CYCLE = [c for c in ("#FF7BAC", VIOLET, "#F2E14C", "#7BE0FF",
                              "#C08B5C", "#8FE388", "#B0B7C3")
                  if c not in {ARM_COL[t] for t in ARMS if t in ARM_COL}]
    for _i, _tag in enumerate(sorted(t for t in ARMS if t not in ARM_COL)):
        ARM_COL[_tag] = _ARM_CYCLE[_i % max(len(_ARM_CYCLE), 1)]

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
        # both axes 0-100%: a fixed 0-60 cap hid the 9mo curve (78.6%) entirely
        ax.set_ylim(0, 100); axr.set_ylim(0, 100)
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
            # MISSED-SOURCE TIER COLORS (2026-10-05, user rulings):
            #   loud misses have to pop against quiet ones because a
            #   missed SNR 50 source is a worse story than a missed SNR
            #   7.5 one, and this panel is where that story has to land.
            #   Three tiers (yellow 7-10, orange 10-20, red >20); the
            #   colored X's are drawn loudest-last so the red overwrites
            #   nothing it needs to show.  The SNR itself is the
            #   LIVE-NOISE rescale (``T_SNR_LIVE``) rather than the
            #   frozen catalogue number -- a run that pulled the
            #   foreground down should re-tier its misses accordingly.
            #   Fallback to the frozen SNR when the rescale is
            #   unavailable (``T_SNR_LIVE`` is just ``T_SNR`` then).
            _mis_mask = ~FOUND_MM
            _mis_snr = np.asarray(T_SNR_LIVE)[_mis_mask]
            _mis_f0 = np.asarray(T_F0)[_mis_mask]
            _mis_amp = np.asarray(T_AMP)[_mis_mask]
            _TIERS = (
                ("#F7D33C", _mis_snr < 10,
                 "missed SNR 7-10"),
                ("#FF8A1A", (_mis_snr >= 10) & (_mis_snr < 20),
                 "missed SNR 10-20"),
                ("#FF2E3E", _mis_snr >= 20,
                 "missed SNR > 20"),
            )
            for _col, _mk, _lbl in _TIERS:
                _n = int(_mk.sum())
                if _n == 0:
                    continue
                b_.scatter(_mis_f0[_mk], _mis_amp[_mk], s=17, marker="x",
                           color=_col, lw=0.8, alpha=0.85,
                           label=f"{_lbl} ({_n})")
            b_.scatter(_rf[~MATCHED_MM], _ra_[~MATCHED_MM], s=22,
                       facecolors="none", edgecolors=VIOLET, lw=0.8, alpha=0.85,
                       label=f"recovered, no match ({int((~MATCHED_MM).sum())})")
            b_.scatter(_rf[MATCHED_MM], _ra_[MATCHED_MM], s=12, color=GREEN,
                       lw=0, alpha=0.95,
                       label=f"recovered and matched ({int(MATCHED_MM.sum())})")
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

        if SHOW_MATCH_STATS and len(RPHYS[MI]) == 0:
            # A store with no GB leaves yet (young run, noise stages only)
            # has ZERO matched sources; every panel below percentiles /
            # histograms empty arrays and np.percentile raises. Skip with a
            # reason instead of crashing the whole page (hit on the v7
            # first snapshot, 2026-08-26).
            MISSING.append("match-stats figures skipped: zero matched "
                           "sources in this snapshot (no gb leaves yet).")
        elif SHOW_MATCH_STATS:
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
        # F5 sits OUTSIDE the ``if RPHYS is not None`` block that defines
        # ``_rf = RPHYS[:, 1]`` for the F4 scatter, so when the RPHYS build
        # try-block raises (unreachable orbits, missing waveform backend,
        # zero recovered sources with an "empty slice" warning cascade,
        # etc.) this branch used to hit ``NameError: name '_rf' is not
        # defined``. Rebuild from REC9 -- always populated before F5 --
        # instead of chaining off the F4 side-effect.
        _rf_hz = REC9[:, 1] * 1e-3
        _hmod, _ = np.histogram(_rf_hz, bins=_eds)
        ax0.stairs(_hmod, _eds, color=GREEN, fill=True, alpha=0.9,
                   label=f"model sources ({int(_rf_hz.size)})")
    ax0.set_yscale("log"); ax0.set_ylim(0.5, None)
    ax0.set_ylabel("sources per bin"); ax0.legend(fontsize=8)
    if ax1 is not None:
        _lo, _hi = _wilson(_hrec, _hdet)
        _fr = np.where(_hdet > 0, _hrec / np.maximum(_hdet, 1), np.nan)
        # maximum(.., 0): at p=0 (a zero-recovery bin -- the full band has
        # them, the old 3-21 mHz band never did) the Wilson lower bound
        # equals p algebraically but can land ~1e-17 ABOVE it in floating
        # point, and errorbar refuses negative yerr.
        ax1.errorbar(_fc, 100 * _fr,
                     yerr=[100 * np.maximum(_fr - _lo, 0),
                           100 * np.maximum(_hi - _fr, 0)], fmt="o",
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
            if not _x:
                continue
            _n = int(_D["n_match"][-1])
            _g = int(_D["n_match"].size) - 1 - int(_D["it0"])
            # An arm from a zero-match store (young run) can produce
            # degenerate Wilson bounds; matplotlib refuses negative yerr.
            _el = np.clip(_el, 0.0, None)
            _eh = np.clip(_eh, 0.0, None)
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
# POOLED (2026-08-27): was a hard-coded 3-iteration window; it is now the
# shared POOL_ITS_SAMPLES window, so every VGB sample panel below (the
# detectable-subset credible intervals, the full-55 distance panel, the
# pooled marginals and the zoomable posterior cloud) draws the same rows.
VGB_SAMP_ROWS = _vgb_pool_rows(POOL_ITS_SAMPLES)
VGB_SAMP_ITS = int(VGB_SAMP_ROWS.size)
vgb_last = vgb_c[VGB_SAMP_ROWS].reshape(-1, 55, _vgb_chain_ndim)  # (S, 55, 5 or 6)
# ``vgb_hh`` gets trimmed to SUB_NIT rows above, and SUB_NIT can land at 0
# on a very young store or one whose mid-flush left the whole VGB column
# unwritten. ``vgb_hh[-1]`` then raises IndexError and the whole page dies.
# Same for the all-NaN case: nanmean on all-NaN returns NaN with a warning,
# but we don't want a NaN cascade downstream. Fall back to a zero-SNR
# vector so ``VGB_DET`` is empty and the panel renders with no bars.
_nv = int(vgb_hh.shape[-1]) if vgb_hh.ndim >= 1 else 55
try:
    if vgb_hh.shape[0] == 0:
        raise ValueError("no VGB h_h rows retained (SUB_NIT trimmed to 0)")
    with np.errstate(invalid="ignore"):
        snr = np.sqrt(np.clip(np.nanmean(vgb_hh[-1], axis=0), 0, None))
    if snr.ndim != 1 or snr.shape[0] != _nv:
        raise ValueError(f"unexpected SNR shape {snr.shape}")
    snr = np.where(np.isfinite(snr), snr, 0.0)
except Exception as _e:
    snr = np.zeros(_nv)
    MISSING.append(f"VGB SNR>7 panel: could not compute per-VGB SNRs "
                   f"({type(_e).__name__}: {_e}); the detectable-subset "
                   f"strip shows as empty (VGB branch not populated yet, "
                   f"or vgb_hh malformed in this snapshot).")
order = np.argsort(snr)[::-1]
VGB_DET = np.nonzero(snr > 7.0)[0]
VGB_DET = VGB_DET[np.argsort(snr[VGB_DET])[::-1]]
VGB_N_DET = int(VGB_DET.size)
# ``vgb_last`` can be shape (0, 55, 5) on the same young / mid-flush
# snapshots the vgb_hh guard above already handles. np.median on an empty
# slice merely warns and returns NaN, but np.percentile crashes inside
# ``_quantile`` (arr[-1, ...] on a size-0 axis). Fall back to NaN arrays so
# the panel still lays out with empty errorbars.
if vgb_last.shape[0] == 0:
    _shp = vgb_last.shape[1]
    med = np.full(_shp, np.nan)
    lo = np.full(_shp, np.nan)
    hi = np.full(_shp, np.nan)
else:
    with np.errstate(invalid="ignore"):
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
                f"(median +/- 1 sigma, last {VGB_SAMP_ITS} its x {nwalk} "
                f"walkers)", fontsize=10)
ax[1].barh(_y, snr[VGB_DET], color=VIOLET, alpha=0.9, height=0.55)
ax[1].axvline(7, color=RED, ls=":", lw=1.2)
ax[1].set_xscale("log"); ax[1].set_xlabel("optimal SNR")
ax[1].set_title("SNR against the fitted noise", fontsize=10)
if VGB_F0 is not None:
    for _i, _k in enumerate(VGB_DET):
        ax[1].text(snr[_k] * 1.06, _y[_i], f" {VGB_F0[_k]:.2f} mHz",
                   fontsize=7, color=DIM, va="center")
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
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
ax.set_title(f"VGB distance posteriors (median +/- 1 sigma; last "
             f"{VGB_SAMP_ITS} its x {nwalk} walkers)")
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
# Non-distance marginals: cols 1..(ndim-1), so 4 panels on the legacy
# 5-param basis and 5 on the chirp 6-param basis.
_marg_cols = list(range(1, _vgb_chain_ndim))
fig, ax = plt.subplots(1, len(_marg_cols), figsize=(13, 2.8))
if len(_marg_cols) == 1:
    ax = [ax]
for _panel_i, j in enumerate(_marg_cols):
    a = ax[_panel_i]
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
fig.suptitle(f"VGB marginals pooled over 55 leaves x the last "
             f"{VGB_SAMP_ITS} stored iterations x {nwalk} cold walkers "
             f"({vgb_last.shape[0] * 55} samples)", fontsize=9, y=1.02)
fig_b64(fig, "vgb_hists")

# GB/VGB explorer data (interactive)
#
# POOLED (2026-08-27): this canvas -- the zoomable amplitude-frequency plane,
# including its "highest-frequency sources" preset -- used to draw the single
# latest stored iteration. It now pools POOL_ITS_SAMPLES iterations, with the
# alive mask taken from EACH iteration's own inds row.
expl = {"gb": [], "vgb": []}
EXPL_ITS, EXPL_RAW, EXPL_STRIDE = 0, 0, 1
nz = np.nonzero(gb_alive_last.sum(axis=0))[0]
if nz.size:
    # f0-AMPLITUDE (user request 2026-08-15): amplitude derived from the
    # sampled (dist, f0, Mc) via the stock transform; plotted as log10(A).
    try:
        from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist
    except Exception:
        gb_amp_from_dist = None
    # SPLIT POOLED SAMPLES INTO (older iterations, LATEST iteration)
    # (2026-10-05, user ruling). The canvas overlays an amber ring on the
    # most-recent-iteration rows so the reader can see where the sampler
    # just WAS versus where it has been. Two stacks here, concatenated
    # after decimation, so the latest-iteration rows are NEVER decimated
    # out even on a dense store: the stride is applied only to the older
    # pool.
    _pool_rows_cache = list(_pool_gb_iter(POOL_ITS_SAMPLES, cols=(0, 1, 2)))
    _latest_it_in_pool = max((it_ for it_, _, _ in _pool_rows_cache),
                              default=-1)
    _gbstack_old = []
    _gbstack_latest = []
    for _it_, _al_, _ch_ in _pool_rows_cache:
        EXPL_ITS += 1
        _is_latest = (_it_ == _latest_it_in_pool)
        for w in range(nwalk):
            al = np.nonzero(_al_[w])[0]
            if not al.size:
                continue
            _d = np.maximum(np.asarray(_ch_[w, al, 0], dtype=float), 1e-6)
            _f = np.asarray(_ch_[w, al, 1], dtype=float)
            _mc = np.asarray(_ch_[w, al, 2], dtype=float)
            if gb_amp_from_dist is not None:
                _amp = np.asarray(gb_amp_from_dist(_f * 1e-3, _mc, _d),
                                  dtype=float)
                _y = np.log10(np.maximum(_amp, 1e-30))
            else:
                _y = 1.0 / _d
            _rows = np.column_stack([
                _f, _y,
                np.full(al.size, float(w)),
                np.full(al.size, 1.0 if _is_latest else 0.0),
            ])
            (_gbstack_latest if _is_latest else _gbstack_old).append(_rows)
    _R_old = (np.concatenate(_gbstack_old) if _gbstack_old
              else np.zeros((0, 4), dtype=float))
    _R_latest = (np.concatenate(_gbstack_latest) if _gbstack_latest
                  else np.zeros((0, 4), dtype=float))
    EXPL_RAW = int(_R_old.shape[0] + _R_latest.shape[0])
    N_LATEST = int(_R_latest.shape[0])
    # PAGE-WEIGHT GUARD. Pooling multiplies this array by the window, and
    # every row is JSON text in the page. On the 3-month store the pooled
    # cloud is ~47k rows (~1 MB of JSON) and needs no decimation at all; a
    # denser store would, so the budget is enforced here -- AFTER pooling,
    # exactly like the truth-cross decimation below, and by striding whole
    # rows so the f0 coverage stays uniform. The latest-iteration rows are
    # EXEMPT from the stride so the "where is the sampler right now?"
    # overlay cannot degrade on a dense store; the budget is spent against
    # the older rows alone. The truth-cross rules (recovery-proximity
    # protection + per-window floor) are untouched.
    EXPL_CAP = 150000
    if EXPL_RAW > EXPL_CAP:
        _budget_old = max(1, EXPL_CAP - N_LATEST)
        if _R_old.shape[0] > _budget_old:
            EXPL_STRIDE = int(np.ceil(_R_old.shape[0] / _budget_old))
            _R_old = _R_old[::EXPL_STRIDE]
    # Older rows first, latest rows last -- the canvas draws the base
    # cyan layer in order and then overlays rings on p[3] == 1, so the
    # order here does not affect the overlay pass, but keeping the
    # latest at the tail matches the "latest = most recent" reading
    # convention and makes the row-slice easier to inspect.
    _R = (np.concatenate([_R_old, _R_latest])
          if (_R_old.size or _R_latest.size)
          else np.zeros((0, 4), dtype=float))
    # Coordinates are ROUNDED before serialization: f0 to 1e-7 mHz (about
    # 1e-3 of an FD bin at three months) and log10 A to 1e-4 dex. Both are
    # far below anything the canvas or the match tolerance can resolve, and
    # they cut the JSON from ~40 to ~23 bytes a row, which is what keeps the
    # pooled cloud from ballooning the page.
    expl["gb"] = [[round(float(_a), 7), round(float(_b), 4),
                   int(_c), int(_d_flag)]
                  for _a, _b, _c, _d_flag in _R]
    del _gbstack_old, _gbstack_latest, _R_old, _R_latest, _R
    expl["gb_latest_iter"] = int(_latest_it_in_pool)
    expl["gb_n_latest"] = N_LATEST
expl["gb_its"] = int(EXPL_ITS)
expl["gb_raw"] = int(EXPL_RAW)
expl["gb_stride"] = int(EXPL_STRIDE)
EXPL_DEC_NOTE = ("" if EXPL_STRIDE <= 1 else
                 f", decimated 1-in-{EXPL_STRIDE} to {len(expl['gb']):,} "
                 f"drawn for page weight")
print(f"[explorer] GB cloud: {EXPL_RAW} alive-leaf rows pooled over "
      f"{EXPL_ITS} stored iteration(s) x {nwalk} cold walkers; "
      f"{len(expl['gb'])} drawn (stride {EXPL_STRIDE}).")
# The VGB side of the explorer pools the same window (fixed-dimension branch,
# so every leaf contributes one row per pooled iteration per walker).
_Spool = vgb_c[VGB_SAMP_ROWS]                        # (P, 24, 55, 5)
for _p in range(_Spool.shape[0]):
    S = _Spool[_p]                                   # (24, 55, 5)
    for w in range(nwalk):
        for leaf in range(55):
            _x = float(VGB_F0[leaf]) if VGB_F0 is not None else int(leaf)
            expl["vgb"].append([_x, float(1.0 / max(S[w, leaf, 0], 1e-6)),
                                float(snr[leaf])])
expl["vgb_its"] = int(VGB_SAMP_ITS)
expl["vgb_axis"] = "f0 [mHz] (catalogue)" if VGB_F0 is not None else "VGB leaf index"
expl["nwalk"] = int(nwalk)

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
gb_truth_pts_grey, gb_truth_pts_red = [], []
cf = f0_band = cat_gidx = None
try:
    _wd = h5py.File(WDWD_PATH, "r")
    cf = _wd["Binaries"]
    _f0_all = cf["GW22FrequencySSBFrame"][:]
    cat_gidx = np.nonzero(
        (_f0_all >= band_edges[0]) & (_f0_all <= band_edges[-1]))[0]
    f0_band_hz = _f0_all[cat_gidx].astype(float, copy=True)   # Hz copy for
                                                              # bit-exact lookups
    f0_band = f0_band_hz * 1e3                    # mHz, in-band catalogue
    del _f0_all
    la_band = np.log10(np.maximum(cf["Amplitude"][:][cat_gidx], 1e-30))
    gb_truth_meta["in_band"] = int(cat_gidx.size)

    # DETECTABILITY per catalogue row (user ask 2026-09-19): the overlay used
    # to draw every catalogue source in one red, which made a cross under a
    # recovered dot indistinguishable from a cross the run could never have
    # found. Detectable stays red; everything below threshold goes grey.
    #
    # Matched on f0 against DET_F0 -- the same detectable-f0 list the leaf
    # count and occupancy overlays already use, carrying its own Tobs guard
    # (an npz built for a different observation time never reaches here).
    # Both arrays are the SAME catalogue column, so the only discrepancy is
    # float round-trip through the npz and the *1e3 to mHz; a relative
    # tolerance well below the catalogue's own f0 spacing settles it without
    # ever matching a neighbouring source.
    det_band = np.ones(f0_band.size, dtype=bool)
    if DET_F0 is not None and DET_F0.size:
        _dref = np.sort(np.asarray(DET_F0, dtype=float))
        _fq = f0_band * 1e-3                       # back to Hz for the match
        _j = np.clip(np.searchsorted(_dref, _fq), 0, _dref.size - 1)
        _jm = np.clip(_j - 1, 0, _dref.size - 1)
        _near = np.minimum(np.abs(_dref[_j] - _fq), np.abs(_dref[_jm] - _fq))
        det_band = _near <= (1e-9 * np.maximum(_fq, 1e-30))
        gb_truth_meta["n_det"] = int(det_band.sum())
        gb_truth_meta["n_undet"] = int((~det_band).sum())
        del _dref, _fq, _j, _jm, _near
    if expl["gb"]:
        _rec_lo = float(min(p[1] for p in expl["gb"]))
        cut = _rec_lo - 0.5      # 0.5 dex below the faintest recovered source
        keep = np.nonzero(la_band >= cut)[0]
        gb_truth_meta.update(cut=cut, rec_lo=_rec_lo, above_cut=int(keep.size))
        if keep.size > TRUTH_CAP:
            # PER-WINDOW decimation (2026-08-23). The old scheme kept the
            # globally brightest half of the budget outright, which drew a
            # hard horizontal red edge (one global brightness cut) across
            # the whole band with the recovered cloud floating above it --
            # on the zoom plots that edge read as "truth amplitudes are
            # wrong". Instead the budget is split across log-spaced f0
            # windows, and each window keeps the brightest half of ITS quota
            # outright plus a uniform subsample of the rest: the drawn
            # fraction is uniform in f0, any brightness cut is local, and no
            # band-wide edge exists to misread.
            #
            # THE QUOTA IS NOT A FLAT FRACTION (2026-08-27). It used to be
            # ``max(round(pop * TRUTH_CAP / above_cut), 1)``, which decimated
            # a window holding 2 sources exactly as hard as one holding
            # 12,000. Two things then broke at the sparse top of the band:
            #   * a window whose quota rounded to 1 got ``_nb = _q // 2 == 0``
            #     -- the "brightest half kept outright" guarantee degenerated
            #     to nothing, and the single point drawn was a COIN FLIP among
            #     the window's members;
            #   * so recovered, catalogue-present sources lost that flip and
            #     drew no red cross under their green dot. Measured on the v7
            #     3-month page: 470 of the 1,064 detectable catalogue sources
            #     carried no truth mark, only 12 of the 45 sources above
            #     10 mHz were drawn at all, and the flagship at 20.380377 mHz
            #     (SNR 46, recovered in 23/24 walkers) lost a 1-of-2 draw in
            #     the top window and vanished.
            # The repair is two rules, both general and budget-neutral:
            #   1. PROTECT: a catalogue row that a model source sits on (within
            #      the page's own TOL_BINS match tolerance) is always drawn.
            #      The panel exists to judge recovered against injected, so a
            #      recovered dot may never sit on an empty patch of truth.
            #   2. FLOOR: every window is drawn IN FULL up to _WIN_FULL members
            #      before any fraction is applied; only what the floor and the
            #      protected set leave over is spread proportionally. Sparse
            #      high-f windows cost a few hundred points in total, so there
            #      is no reason to sample them at all.
            _rng = np.random.default_rng(0)
            _NWIN, _WIN_FULL = 64, 200
            _wedges = np.geomspace(f0_band[keep].min() * (1 - 1e-12),
                                   f0_band[keep].max() * (1 + 1e-12),
                                   _NWIN + 1)
            _wid = np.clip(np.searchsorted(_wedges, f0_band[keep],
                                           side="right") - 1, 0, _NWIN - 1)

            # rule 1 -- rows a recovered source sits on, by f0 proximity in
            # the SAME tolerance the recovery panels call a match.
            _pro = np.zeros(keep.size, dtype=bool)
            _rf0 = np.unique(np.asarray([p[0] for p in expl["gb"]], dtype=float))
            if _rf0.size:
                _tolm = TOL_BINS * SCI_DF * 1e3           # mHz
                _srt = np.argsort(f0_band[keep])
                _ks = f0_band[keep][_srt]
                _plo = np.searchsorted(_ks, _rf0 - _tolm, side="left")
                _phi = np.searchsorted(_ks, _rf0 + _tolm, side="right")
                for _a, _b in zip(_plo, _phi):
                    if _b > _a:
                        _pro[_srt[_a:_b]] = True

            # ...and PROTECT EVERY DETECTABLE SOURCE too (2026-09-19). Rule 1
            # above only spares a catalogue row that a MODEL source already
            # sits on, so a detectable source the run has NOT found could be
            # decimated away -- and with red now meaning "detectable", a
            # missing red cross would read as "nothing to find here" when the
            # truth is "here is one it missed", which is the single most
            # important thing this panel can show. They are 1,064 rows out of
            # ~4M on the 3-month set, so protecting all of them is free.
            # Gated on "n_det", NOT on det_band itself: with no DET_F0 the
            # mask defaults to ALL-TRUE (everything drawn red, the old
            # behaviour), and OR-ing that in would protect every row and
            # silently disable decimation entirely.
            if "n_det" in gb_truth_meta:
                _pro |= det_band[keep]

            # rule 2 -- floor first, then spread what is left proportionally.
            # The floor is raised to the protected count where a window holds
            # more protected rows than the floor, so the per-window quota can
            # never be smaller than the rows rule 1 pins.
            _pop = np.bincount(_wid, minlength=_NWIN)
            _npro = np.bincount(_wid[_pro], minlength=_NWIN)
            _base = np.maximum(np.minimum(_pop, _WIN_FULL), _npro)
            _left = _pop - _base
            _extra = max(TRUTH_CAP - int(_base.sum()), 0)
            _frac = _extra / int(_left.sum()) if _left.sum() else 0.0
            _quota = _base + np.rint(_left * _frac).astype(int)

            _parts = []
            for _wi in range(_NWIN):
                _m = _wid == _wi
                if not _m.any():
                    continue
                _kw, _pw = keep[_m], _pro[_m]
                _q = int(min(_quota[_wi], _kw.size))
                if _kw.size <= _q:
                    _parts.append(_kw)
                    continue
                _must, _rest = _kw[_pw], _kw[~_pw]
                _r = _q - _must.size                     # >= 0 by _base above
                if _r <= 0:
                    _parts.append(_must)
                    continue
                _o = _rest[np.argsort(la_band[_rest])[::-1]]
                # never 0: a quota of 1 must spend it on the BRIGHTEST member
                # of the window, not on a random one.
                _nb = int(min(max(_r // 2, 1), _o.size))
                _ns = int(min(_r - _nb, _o.size - _nb))
                _sub = (_rng.choice(_o[_nb:], size=_ns, replace=False)
                        if _ns > 0 else np.empty(0, dtype=_o.dtype))
                _parts.append(np.concatenate([_must, _o[:_nb], _sub]))
            sel = np.unique(np.concatenate(_parts))
            gb_truth_meta["nwin"] = _NWIN
            gb_truth_meta["win_full"] = int(_WIN_FULL)
            gb_truth_meta["protected"] = int(_pro.sum())
            gb_truth_meta["drawn_frac"] = float(_frac)
        else:
            sel = keep
        # Third column: 1 = detectable (red), 0 = sub-threshold (grey). An
        # int costs 2 bytes a row in the JSON, against the ~23 the pair
        # already costs, and saves the canvas a second lookup structure.
        gb_truth_pts = [[float(f"{f0_band[i]:.7g}"), round(float(la_band[i]), 4),
                         int(det_band[i])] for i in sel]
        gb_truth_meta["shown"] = len(gb_truth_pts)

        # SPLIT the truth crosses into (grey: undetectable) and (red:
        # detectable but not recovered) so the zoomable canvas can render
        # the four required classes. Rows that are detectable AND recovered
        # are dropped here -- the green filled-circle source marker sits at
        # the recovered coordinate on top, which is what the reader is
        # trying to see.
        #
        # Classification is by f0 lookup with a small absolute tolerance
        # (1e-12 Hz, ~3 orders below the FD bin). `np.isin` would be
        # unreliable: TRU["f0"] goes through the mHz sampling basis in
        # build_truth (rows[:, 1] * 1e-3), so it does NOT bit-exactly match
        # the raw wdwd `GW22FrequencySSBFrame` in f0_band_hz.
        gb_truth_pts_grey = gb_truth_pts   # fallback: no TRU -> all grey
        gb_truth_pts_red = []
        if TRU is not None:
            def _within_tol_lookup(vals_hz):
                if vals_hz.size == 0:
                    return lambda q: np.zeros_like(q, dtype=bool)
                _s = np.sort(np.asarray(vals_hz, float))
                _n = _s.size

                def _fn(q):
                    q = np.asarray(q, float)
                    i = np.searchsorted(_s, q)
                    il = np.clip(i - 1, 0, _n - 1)
                    ir = np.clip(i, 0, _n - 1)
                    dl = np.abs(_s[il] - q); dr = np.abs(_s[ir] - q)
                    return np.minimum(dl, dr) < 1e-12
                return _fn

            _det_full_hz = np.asarray(
                TRU["f0"][ np.asarray(TRU["det"], bool)
                          & (np.asarray(TRU["f0"], float) >= band_edges[0])
                          & (np.asarray(TRU["f0"], float) <= band_edges[-1]) ],
                float)
            _rec_full_hz = (np.asarray(T_F0, float)[FOUND_MM]
                            if (isinstance(FOUND_MM, np.ndarray)
                                and FOUND_MM.size)
                            else np.zeros(0))

            _sel_hz = f0_band_hz[sel]
            _is_det = _within_tol_lookup(_det_full_hz)(_sel_hz)
            _is_rec = _within_tol_lookup(_rec_full_hz)(_sel_hz)
            # grey = undetectable within the decimated `sel`
            _grey_mask = ~_is_det
            gb_truth_pts_grey = [
                [float(f"{f0_band[sel[i]]:.7g}"),
                 round(float(la_band[sel[i]]), 4)]
                for i in np.where(_grey_mask)[0]]
            # red = detectable, not recovered -- pulled from the FULL
            # detectable set in-band, no decimation. There are only ~10^3 of
            # these even on a 1-year run, so this is cheap and complete.
            _found_arr = (np.asarray(FOUND_MM, bool)
                          if (isinstance(FOUND_MM, np.ndarray) and FOUND_MM.size)
                          else np.zeros(NDET, bool))
            _mis_f = np.asarray(T_F0, float)[~_found_arr]
            _mis_a = np.log10(np.maximum(
                np.asarray(T_AMP, float)[~_found_arr], 1e-40))
            # SNR tier per missed source (user ruling 2026-10-05):
            # 0 = SNR 7-10 (yellow), 1 = SNR 10-20 (orange), 2 = SNR > 20 (red).
            # Carrying ONE int per point in the JSON blob (3 bytes vs the
            # ~22 the pair already costs) is what lets the canvas color
            # tier by tier without a second lookup table.
            #
            # SNR is the LIVE-NOISE rescale (``T_SNR_LIVE``, max-lnL cold
            # walker of the last stored iteration) rather than the frozen
            # catalogue number -- the three tiers reflect what the
            # current run is missing, not what the truth-build iteration
            # was. ``T_SNR_LIVE`` is already filtered to the detectable
            # subset (same ``_sel`` mask as ``T_SNR``), so the per-source
            # indexing matches ``~_found_arr`` directly. Falls back to
            # ``T_SNR`` when the rescale was unavailable (that assignment
            # is already done above).
            _mis_s = np.asarray(T_SNR_LIVE, float)[~_found_arr]
            _mis_tier = np.where(_mis_s >= 20, 2,
                                 np.where(_mis_s >= 10, 1, 0)).astype(int)
            gb_truth_pts_red = [
                [float(f"{f * 1e3:.7g}"), round(float(a), 4), int(t)]
                for f, a, t in zip(_mis_f, _mis_a, _mis_tier)]
            gb_truth_meta["shown_grey"] = len(gb_truth_pts_grey)
            gb_truth_meta["shown_red"] = len(gb_truth_pts_red)
            gb_truth_meta["red_tier_counts"] = {
                "7-10 (yellow)":   int((_mis_tier == 0).sum()),
                "10-20 (orange)":  int((_mis_tier == 1).sum()),
                ">20 (red)":       int((_mis_tier == 2).sum()),
            }
        _tm = gb_truth_meta
        _detbit = (
            f"{_tm.get('shown_det', _tm['shown']):,} of them RED (detectable, "
            f"SNR > 7), the rest GREY (sub-threshold -- drawn so a patch with "
            f"no red mark reads as 'nothing findable here' rather than as a "
            f"gap in the overlay). "
            if "shown_det" in _tm else "")
        expl["truth_cap"] = (
            f"Catalogue truths: {_tm['shown']:,} points shown of "
            f"{_tm['above_cut']:,} passing the cut log10 A >= {_tm['cut']:.2f} "
            f"(0.5 dex below the faintest recovered source, {_tm['rec_lo']:.2f}); "
            f"{_tm['in_band']:,} catalogue sources lie in the GB band in total. "
            + _detbit
            + (f"Decimation is per frequency window ({_tm['nwin']} log-spaced "
               f"windows). Any window holding at most {_tm['win_full']:,} "
               f"sources is drawn IN FULL -- which is every window above a few "
               f"mHz -- and the {_tm['protected']:,} catalogue rows lying "
               f"within {TOL_BINS:.0f} frequency bins of a model source are "
               f"always drawn, so no recovered dot can sit on a patch with no "
               f"truth mark under it. Only what those two rules leave over is "
               f"decimated, at ~{100 * _tm['drawn_frac']:.1f}% of each crowded "
               f"window's remainder (brightest half of that share kept "
               f"outright, the rest a uniform random subsample) -- the drawn "
               f"density is uniform in f0 and there is NO band-wide "
               f"brightness edge."
               if "nwin" in _tm else "No decimation was needed."))
except Exception as e:
    MISSING.append(f"GB injection catalogue truth overlay unavailable: {e!r}")

expl["truth"] = gb_truth_pts
# When TRU is available the classification split populated these two; when
# it was not, the grey list falls back to the full unclassified truth so
# the canvas still shows something recognisable.
expl["truth_grey"] = gb_truth_pts_grey or gb_truth_pts
expl["truth_red"]  = gb_truth_pts_red
expl["truth_meta"] = gb_truth_meta

# Recovered-source markers for the zoomable canvas: one per source at the
# last stored iteration, classified matched/unmatched under the current
# match criterion (MATCHED_MM). Coords are (f0 mHz, log10 amplitude) so
# they land on the same axes as expl["gb"].
expl["sources"] = []
try:
    if (RPHYS is not None and RPHYS.shape[0] > 0
            and isinstance(MATCHED_MM, np.ndarray)
            and MATCHED_MM.size == RPHYS.shape[0]):
        _rf_hz = np.asarray(RPHYS[:, 1], float)      # Hz from GBGPU
        _rf_mhz = _rf_hz * 1e3                       # convert to mHz for plot
        _ra_lg = np.log10(np.maximum(np.asarray(RPHYS[:, 0], float), 1e-40))
        expl["sources"] = [
            [round(float(_rf_mhz[i]), 7),
             round(float(_ra_lg[i]), 4),
             1 if bool(MATCHED_MM[i]) else 0]
            for i in range(RPHYS.shape[0])]
except NameError:
    pass
expl["match_note"] = MATCH_CRIT_TXT

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
# Task 2: every VGB, SNR-descending in the selector -- rendered as
# ChainConsumer CORNER plots (2026-08-15, user request), one PNG per leaf,
# base64'd into the page. The selector swaps the src of a single <img>, so
# only the selected corner is ever in the DOM's visible flow; the JS
# histogram panel it replaced is gone; the GB panel moved to corner
# plots too (2026-08-20), so both selectors are now image swaps.
#
# Samples: the last POOL_ITS_POSTERIOR stored iterations x every cold walker
# (~240 rows), wider than the POOL_ITS_SAMPLES window the scatter/marginal
# panels use -- a corner needs the extra rows for its 2-D contours to mean
# anything. Both windows come from the shared pooling helper.
VGB_CORNER_ROWS = _vgb_pool_rows(POOL_ITS_POSTERIOR)
CORNER_ITS = int(VGB_CORNER_ROWS.size)
CORNER_DPI, CORNER_IN = 68, 7.0        # size-budget tuned (see below)
vgb_corner = vgb_c[VGB_CORNER_ROWS].reshape(-1, 55, _vgb_chain_ndim)  # (S, 55, 5 or 6)
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

    def corner_png(Sm, names, truth, title, size_in=None, tally=None,
                   dpi=None, label_fs=8, tick_fs=7, smooth=None, bins=None):
        """Samples -> a base64 PNG of their ChainConsumer corner.

        Shared by the VGB leaf panel and the GB per-source panel; the only
        thing that differs between them is the parameter list, the figure
        size, and how the samples were gathered.

        Extents are widened to contain the catalogue truth so a truth line
        that falls OUTSIDE the posterior is still visible; without it a
        badly-recovered source would silently show no truth at all.

        SMOOTHING TRACKS SAMPLE COUNT. chainconsumer's default smooth=1 on a
        few dozen rows does not produce a posterior, it produces a field of
        spurious closed rings that read as multimodality. Below ~200 rows
        the kernel is widened and the binning coarsened so the contours
        cannot claim structure the sample count does not support.
        """
        Sm = np.asarray(Sm, dtype=float)
        n = int(Sm.shape[0])
        smooth_, bins_ = (3, 12) if n < 200 else (1, 18)
        if smooth is not None:
            smooth_ = smooth
        if bins is not None:
            bins_ = bins
        _dpi = dpi or CORNER_DPI
        ext = {}
        for j, nm_ in enumerate(names):
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
                samples=pd.DataFrame(Sm, columns=names), name="posterior",
                color=VIOLET, shade=True, shade_alpha=0.35, bar_shade=True,
                plot_point=False, smooth=smooth_, bins=bins_,
                show_label_in_legend=False))
            if truth is not None:
                cc.add_truth(Truth(
                    location={nm_: float(truth[j])
                              for j, nm_ in enumerate(names)},
                    color=RED, line_style=":", line_width=1.4))
            # diagonal_tick_labels defaults ON and is unreadable at this
            # dpi; 3 upright ticks per axis is what survives the size budget.
            cc.set_plot_config(PlotConfig(
                labels={nm_: nm_ for nm_ in names}, extents=ext,
                label_font_size=label_fs, tick_font_size=tick_fs, max_ticks=3,
                diagonal_tick_labels=False,
                show_legend=False, summarise=False, dpi=_dpi))
            _sz = size_in or CORNER_IN
            fig_ = cc.plotter.plot(figsize=(_sz, _sz))
        fig_.suptitle(title, fontsize=9 if _dpi <= 80 else 11, color=FG)
        buf = io.BytesIO()
        fig_.savefig(buf, format="png", dpi=_dpi, bbox_inches="tight")
        plt.close(fig_)
        raw = buf.getvalue()
        (CORNER_BYTES if tally is None else tally).append(len(raw))
        return base64.b64encode(raw).decode()

    def vgb_corner_png(leaf, title):
        """One leaf -> a base64 PNG of its 5x5 ChainConsumer corner."""
        return corner_png(vgb_corner[:, leaf, :], VGB_NAMES,
                          None if VGB_TRUTH is None else VGB_TRUTH[leaf],
                          title)

    # DETECTABLE-ONLY (2026-09-05): render corner PNGs only for the SNR>7
    # detectable set. The 44 prior-dominated leaves render flat, useless
    # corners and were the dominant page-weight cost after POOL_ITS->30 --
    # dropping them is the main 16 MB size win. The per-leaf corner selector
    # panel + JS stay; the picker now lists exactly this detectable set.
    _corner_order = list(VGB_DET)
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
    # corner_png IS DEFINED INSIDE THE try ABOVE, so when chainconsumer is
    # missing it never exists -- and the GB per-source corner block far
    # below then failed with NameError("name 'corner_png' is not defined")
    # ONCE PER SOURCE. Three identical NameErrors on the page, none of
    # which named the actual cause, which was reported as "the html
    # generator still not working" (2026-09-26). Bind the name so that
    # block can say the real reason once instead.
    corner_png = None
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
    # POOL SAMPLES ACROSS ITERATIONS (2026-08-20). The marginal-histogram
    # panel this replaced drew a single stored iteration -- ~33 rows, which
    # is enough for a 1-D bar chart and nowhere near enough for a 2-D
    # contour. The GB branch is trans-dimensional, so leaf index i is NOT
    # the same source from one iteration to the next; rows are associated by
    # f0 proximity instead, at most one per (iteration, walker) cell, taking
    # the alive leaf nearest the cluster centre. Same widened window the VGB
    # corners already use, and the same reason for it. Routed through the
    # shared pooling helper (2026-08-27) so the window, the young-store
    # clamp and the extract keep-window clamp are defined in ONE place.
    GB_CORNER_ROWS = _pool_its(POOL_ITS_POSTERIOR, "gb")
    GB_CORNER_ITS = int(GB_CORNER_ROWS.size)
    # 9 params need far more canvas than the VGB panel's 5. At the VGB's
    # 7in/68dpi the axis labels of a 9x9 collide into an unreadable smear.
    GB_CORNER_IN, GB_CORNER_DPI = 13.0, 96
    # SHORT AXIS LABELS, UNITS IN THE CAPTION. "dist [kpc]" / "Mc [Msol]" /
    # "fdot_astro_ratio" overrun their panels at 9 across.
    GB_CORNER_LABELS = ["dist", "d_f0 [uHz]", "Mc", "phi0", "cos_iota",
                        "psi", "alpha", "sin_delta", "fdot_ratio"]
    # f0 is plotted as an OFFSET from the cluster centre. Absolute f0 has a
    # ~0.002 mHz spread on a ~20 mHz value, so matplotlib renders it with an
    # "[1e-3+2.038e1]" offset tag glued to the axis label -- the single worst
    # contributor to the collisions. The centre is in the panel title, and
    # the offset is the same Delta-f0 the catalogue-match note quotes.
    GB_CORNER_BYTES = []
    _gb_win = CLUSTER_BINS * DF_MHZ
    _centers = [float(np.median([gb_chain_cold[w, i, 1] for _, w, i in c]))
                for c in solid[:3]]
    _pool = [[] for _ in _centers]
    _pool_its = [0 for _ in _centers]
    for _it in GB_CORNER_ROWS:
        _al = gb_inds[_it]                             # THIS iteration's mask
        if not _al.any():
            continue
        _ch = _row("chain/gb", (_it, 0, 0))            # (nwalk, nleaf, 9)
        if _ch is None:
            continue
        for _k, _c0 in enumerate(_centers):
            _got = 0
            for _w in range(nwalk):
                _idx = np.nonzero(_al[_w])[0]
                if not _idx.size:
                    continue
                _d = np.abs(_ch[_w, _idx, 1] - _c0)
                _j = int(np.argmin(_d))
                if _d[_j] <= _gb_win:
                    _pool[_k].append(np.array(_ch[_w, _idx[_j]], dtype=float))
                    _got += 1
            _pool_its[_k] += 1 if _got else 0
        del _ch
    gb1_meta["corner_its"] = int(GB_CORNER_ITS)

    for _k, c in enumerate(solid[:3]):
        # pooled rows if the association found any, else the single-iteration
        # cluster (so the panel still renders on a one-iteration snapshot)
        P = (np.array(_pool[_k]) if _pool[_k]
             else np.array([gb_chain_cold[w, i] for _, w, i in c]))
        f0_med = float(np.median([gb_chain_cold[w, i, 1] for _, w, i in c]))
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
        _lab = (f"GB @ {f0_med:.5f} mHz ({len(P)} samples, "
                f"{len({x[1] for x in c})} walkers)")
        # centre the f0 column (and its truth) on the cluster, in uHz
        _Pc = np.array(P, dtype=float, copy=True)
        _Pc[:, 1] = (_Pc[:, 1] - f0_med) * 1e3
        _tc = None
        if truth is not None:
            _tc = np.array(truth, dtype=float, copy=True)
            _tc[1] = (_tc[1] - f0_med) * 1e3
        if corner_png is None:
            # One honest line for the whole block, not one NameError per
            # source -- which is what this produced before, naming a
            # symptom instead of the cause.
            MISSING.append(
                "GB per-source corner plots unavailable: chainconsumer is "
                "not installed, so the shared corner_png helper was never "
                "defined. pip install chainconsumer to get them (the VGB "
                "corners above need it too).")
            break
        try:
            _png = corner_png(_Pc, GB_CORNER_LABELS, _tc, _lab,
                              size_in=GB_CORNER_IN, dpi=GB_CORNER_DPI,
                              label_fs=11, tick_fs=9, smooth=3, bins=14,
                              tally=GB_CORNER_BYTES)
        except Exception as _e:
            MISSING.append(f"GB corner plot for {_lab} unavailable: {_e!r}")
            continue
        GB1["src"].append({
            "label": _lab,
            "sub": (f"{len(P)} samples pooled over the last "
                    f"{_pool_its[_k]} GB-active stored iterations x {nwalk} "
                    f"cold walkers, associated by f0 within the "
                    f"{CLUSTER_BINS:.0f}-bin cluster window "
                    f"({CLUSTER_BINS * DF_MHZ * 1e3:.2f} uHz); "
                    f"dotted red = catalogue truth. Axes: dist [kpc], "
                    f"d_f0 = f0 - {f0_med:.5f} mHz [uHz], Mc [Msol], "
                    f"phi0, cos_iota, psi, alpha, sin_delta, "
                    f"fdot_astro_ratio"),
            "note": note, "bad": bad, "png": _png,
        })
    if GB_CORNER_BYTES:
        print(f"[corner] {len(GB_CORNER_BYTES)} GB corners, png "
              f"min/max = {min(GB_CORNER_BYTES)/1024:.1f} / "
              f"{max(GB_CORNER_BYTES)/1024:.1f} KB, total "
              f"{sum(GB_CORNER_BYTES)/1024**2:.2f} MB")
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

# ---- 8b. SEARCH EFFICIENCY: what each move COSTS against what it BUYS ----
# Added 2026-09-25 after job 628's first two full gb_search_1 cycles. The
# panel above answers "how long did this move take"; it cannot answer the
# question that actually decides whether the recipe is worth its wall time:
#
#   * cost per NEW LEAF, which is the number that degrades as the search
#     saturates. 628 went 5.5 s/leaf on iteration 1 to 70.6 s/leaf on
#     iteration 2 at an essentially FLAT per-iteration cost -- the
#     iteration timing alone shows none of that;
#   * the rj_step vs in-model-polish split inside each move. In 628 the
#     RJ proposals are ~10% of the GB wall and the convergence polish is
#     ~85%, so "the RJ schedule is expensive" was the wrong diagnosis;
#   * per-RJ-move YIELD. rj_replace cost ~1420 s/iteration and accepted
#     ZERO swaps in 122k proposals x 2 passes x 2 iterations. Nothing on
#     the page said so.
#
# Sources: the ``[GF_TIMING] ... move=<m> it=<k> wall_s=<w>`` records (per
# ITERATION, unlike the ``[GB_TIMING]`` ones above), ``[GB_TIMING <m> head
# r0]`` for the internal split, ``[GB_ACCEPT <m>] rj cold`` for the yield,
# and ``gb_counts`` for the leaf series. Every piece is optional: a run
# that predates any of them simply drops that subplot.
GFT_RE = re.compile(
    r"\[GF_TIMING\] stage=(\S+) move=(\S+) it=(\d+) wall_s=([\d.]+)")
SPLIT_RE = re.compile(r"\[GB_TIMING (\w+) head r0\] (.*)")
ACC_RE = re.compile(
    r"\[GB_ACCEPT (rj_\w+)\] rj cold ([\d.]+) \(n=(\d+)\)")
try:
    per_it = {}                      # it -> {move: wall_s}
    for _m in GFT_RE.finditer(log_text):
        if _m.group(2) == "__total__":
            continue
        per_it.setdefault(int(_m.group(3)), {})[_m.group(2)] = float(_m.group(4))
    splits = {}                      # move -> (rj_step, inmodel_repeats)
    for _m in SPLIT_RE.finditer(log_text):
        _d = {}
        for _kv in _m.group(2).split():
            if "=" in _kv and _kv.endswith("s"):
                _k, _v = _kv.split("=", 1)
                try:
                    _d[_k] = float(_v[:-1])
                except ValueError:
                    pass
        splits[_m.group(1)] = (_d.get("rj_step", 0.0),
                               _d.get("inmodel_repeats", 0.0))
    yields = {}                      # move -> [cold accepts per propose]
    for _m in ACC_RE.finditer(log_text):
        yields.setdefault(_m.group(1), []).append(
            float(_m.group(2)) * int(_m.group(3)))

    if per_it:
        its = sorted(per_it)
        moves = sorted({mv for d in per_it.values() for mv in d},
                       key=lambda m: -sum(per_it[i].get(m, 0.0) for i in its))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))

        # (a) stacked wall per move per iteration
        bot = np.zeros(len(its))
        for _i, mv in enumerate(moves[:10]):
            h = np.array([per_it[i].get(mv, 0.0) for i in its])
            ax[0].bar(its, h, bottom=bot, label=mv,
                      color=plt.cm.viridis(_i / max(len(moves[:10]) - 1, 1)))
            bot += h
        ax[0].set_xlabel("iteration"); ax[0].set_ylabel("wall [s]")
        ax[0].set_title("where the iteration goes, per move")
        ax[0].legend(fontsize=6, ncol=2)

        # (b) seconds per NEW leaf -- the saturation curve
        if "gb_counts" in dir() and len(gb_counts) >= 2:
            lv = np.asarray(gb_counts, dtype=float).mean(axis=-1)
            n = min(len(lv), len(its))
            tot = np.array([sum(per_it[i].values()) for i in its[:n]])
            d_lv = np.diff(np.concatenate([[0.0], lv[:n]]))
            with np.errstate(divide="ignore", invalid="ignore"):
                spl = np.where(d_lv > 0, tot / d_lv, np.nan)
            ax[1].plot(its[:n], spl, "o-", color=AMBER, lw=1.2, ms=4)
            ax[1].set_yscale("log")
            ax[1].set_ylabel("wall [s] per NEW leaf/walker")
            ax[1].set_xlabel("iteration")
            ax[1].set_title("search efficiency (lower is better)")
            for _x, _y, _d in zip(its[:n], spl, d_lv):
                if np.isfinite(_y):
                    ax[1].annotate(f"+{_d:.0f}", (_x, _y), fontsize=6,
                                   textcoords="offset points", xytext=(0, 5))
        else:
            ax[1].text(0.5, 0.5, "no leaf series", ha="center",
                       transform=ax[1].transAxes)

        # (c) per-RJ-move yield: cold accepts on the most recent propose,
        #     against that move's wall. A bar at zero with a long wall is
        #     the signature that put rj_replace on the table.
        rj = [m for m in moves if m.startswith("rj_")]
        if rj:
            wall = [per_it[its[-1]].get(m, 0.0) for m in rj]
            got = [yields.get(m, [0.0])[-1] for m in rj]
            x = np.arange(len(rj))
            ax[2].bar(x - 0.2, wall, 0.4, color=CYAN, label="wall [s]")
            ax2b = ax[2].twinx()
            ax2b.bar(x + 0.2, got, 0.4, color=GREEN, label="cold accepts")
            ax2b.set_ylabel("cold RJ accepts", color=GREEN)
            ax[2].set_xticks(x)
            ax[2].set_xticklabels([m.replace("rj_", "") for m in rj],
                                  rotation=30, ha="right", fontsize=7)
            ax[2].set_ylabel("wall [s]", color=CYAN)
            ax[2].set_title("cost vs yield, latest iteration")
        fig_b64(fig, "search_efficiency")

        # The split is a table, not a plot -- three numbers per move.
        _rows = []
        for mv in moves:
            _rs, _im = splits.get(mv, (0.0, 0.0))
            _w = per_it[its[-1]].get(mv, 0.0)
            if _w <= 0:
                continue
            _rows.append((mv, _w, _rs, _im, 100.0 * _im / max(_w, 1e-9)))
        SEARCH_SPLIT_ROWS = _rows
    else:
        SEARCH_SPLIT_ROWS = []
except Exception as _e:                       # never break the page
    SEARCH_SPLIT_ROWS = []
    MISSING.append(f"search-efficiency panel failed: {_e!r}")

if SEARCH_SPLIT_ROWS:
    _tot_w = sum(r[1] for r in SEARCH_SPLIT_ROWS)
    _tot_rs = sum(r[2] for r in SEARCH_SPLIT_ROWS)
    _tot_im = sum(r[3] for r in SEARCH_SPLIT_ROWS)
    _cells = "".join(
        f"<tr><td>{mv}</td><td style='text-align:right'>{w:,.0f}</td>"
        f"<td style='text-align:right'>{rs:,.0f}</td>"
        f"<td style='text-align:right'>{im:,.0f}</td>"
        f"<td style='text-align:right'>{pct:.0f}%</td></tr>"
        for mv, w, rs, im, pct in SEARCH_SPLIT_ROWS)
    SEARCH_SPLIT_HTML = (
        "<table style='width:100%;border-collapse:collapse;font-size:12px'>"
        "<tr><th style='text-align:left'>move</th>"
        "<th style='text-align:right'>wall [s] / iteration</th>"
        "<th style='text-align:right'>rj_step [s] / propose</th>"
        "<th style='text-align:right'>polish [s] / propose</th>"
        "<th style='text-align:right'>polish %</th></tr>"
        + _cells
        + f"<tr><td><b>total</b></td>"
          f"<td style='text-align:right'><b>{_tot_w:,.0f}</b></td>"
          f"<td style='text-align:right'><b>{_tot_rs:,.0f}"
          f" ({100.0 * _tot_rs / max(_tot_w, 1e-9):.0f}%)</b></td>"
          f"<td style='text-align:right'><b>{_tot_im:,.0f}"
          f" ({100.0 * _tot_im / max(_tot_w, 1e-9):.0f}%)</b></td>"
          f"<td></td></tr></table>")
else:
    SEARCH_SPLIT_HTML = "<i>no [GF_TIMING] per-iteration records in this log</i>"

# ======================= SEARCH GATES (2026-09-26) =========================
# User ask: "update the html to be more reflective of the new gates and
# settings in the run. Like the shutoffs in a given recipe stage and the
# status of those as the run evolves."
#
# The v9 search is governed by four gates that the page could not see, and
# each of them can be ARMED AND REACHING NOTHING -- the failure shape this
# run has hit repeatedly. Every panel below is therefore built to answer
# "is this gate doing anything", not just "what is it set to".
#
#   1. per-(walker, band) RJ shutoff  -- and since GB_SEARCH_STAGE_END_ON_
#      SHUTOFF it is THE stage-end criterion, so its progress is progress
#      toward the next recipe stage.
#   2. [GB_CELL_LL] reconciliation    -- the only instrument that catches a
#      mis-credited cell ledger, and the one the all-rungs vertical swap
#      was gated on.
#   3. the vertical swap ladder       -- per rung pair, plus the all-rungs
#      unpriceable drops.
#   4. in-model convergence           -- newborn vs survivor repeats and
#      how many rows sit exactly on the floor.
_GATE_SHUT = re.compile(
    r"\[GB_STAGE (\w+)\] per-walker RJ shutoff: (\d+) \(walker, band\) pairs "
    r"converged.*?(\d+) of (\d+) pairs now shut")
_GATE_INERT = re.compile(
    r"\[GB_STAGE (\w+)\] per-walker RJ valve is currently inert")
_GATE_LIVE = re.compile(
    r"\[GB_STAGE (\w+)\] per-walker RJ shutoff valve is LIVE "
    r"\((\d+) walkers x (\d+) bands, window (\d+)")
_GATE_CELLLL_UNIT = re.compile(r"\[GB_CELL_LL (\w+)\] unit:")
_GATE_CELLLL_EXC = re.compile(
    r"\[GB_CELL_LL (\w+)\] per-repeat sampled-vs-actual diff ([\d.eE+-]+) "
    r"exceeds its temperature-scaled allowance ([\d.eE+-]+) \(temp (\d+)")
_GATE_VERT = re.compile(
    r"\[GB_VERT (\w+)\][^\n]*?proposed (\d+) accepted (\d+)")
_GATE_VERT_DROP = re.compile(r"ALL-RUNGS on, (\d+) pair\(s\) dropped")
_GATE_RUNGPAIR = re.compile(r"T(\d+)-T(\d+): (\d+)/(\d+)")
_GATE_IMCONV = re.compile(
    r"\[GB_IMCONV (\w+)\] (\w+) calibration: repeats p10/p50/p90/max = "
    r"(\d+)/(\d+)/(\d+)/(\d+), AT FLOOR \(<=(\d+)\) (\d+) \(([\d.]+)%\)")
_GATE_ARMED = re.compile(r"\[GB_IMCONV (\w+)\] armed \(on\): ([^\n]+)")
_GATE_STAGE_END = re.compile(r"\[V9-STAGE ([\w-]+)\] ([^\n]{0,150})")
# The richest progress series there is: it fires every 10 checks and counts
# the OCCUPIED pairs still paying, i.e. exactly what has to reach zero for
# the stage to end under GB_SEARCH_STAGE_END_ON_SHUTOFF.
_GATE_ACTIVE = re.compile(
    r"\[V9-STAGE ([\w-]+)\] running: (\d+) occupied \(walker, band\) "
    r"pair\(s\) still active")

GATES_HTML = ""
NAV_GATES = ""
_g_shut = {}          # move -> [(shut, total), ...] in log order
_g_live = {}          # move -> (nwalkers, nbands, window)
_g_ll_unit = {}       # move -> count of unit reports
_g_ll_exc = {}        # move -> [ratio, ...]
_g_ll_cold = {}       # move -> cold-rung exceedances
_g_vert = {}          # move -> [(proposed, accepted), ...]
_g_drop = 0
_g_rung = {}          # cold rung -> [accepted, proposed]
_g_imconv = {}        # (move, cls) -> [(p50, floor, atfloor_pct), ...]
_g_armed = {}
_g_stage_lines = []
_g_active = {}       # stage -> [occupied-still-active, ...]
if log_text:
    for _m in _GATE_SHUT.finditer(log_text):
        _g_shut.setdefault(_m.group(1), []).append(
            (int(_m.group(3)), int(_m.group(4))))
    for _m in _GATE_LIVE.finditer(log_text):
        _g_live[_m.group(1)] = (int(_m.group(2)), int(_m.group(3)),
                                int(_m.group(4)))
    for _m in _GATE_CELLLL_UNIT.finditer(log_text):
        _g_ll_unit[_m.group(1)] = _g_ll_unit.get(_m.group(1), 0) + 1
    for _m in _GATE_CELLLL_EXC.finditer(log_text):
        _mv = _m.group(1)
        try:
            _g_ll_exc.setdefault(_mv, []).append(
                float(_m.group(2)) / max(float(_m.group(3)), 1e-30))
        except ValueError:
            pass
        if int(_m.group(4)) == 0:
            _g_ll_cold[_mv] = _g_ll_cold.get(_mv, 0) + 1
    for _m in _GATE_VERT.finditer(log_text):
        _g_vert.setdefault(_m.group(1), []).append(
            (int(_m.group(2)), int(_m.group(3))))
    for _m in _GATE_VERT_DROP.finditer(log_text):
        _g_drop += int(_m.group(1))
    for _m in _GATE_RUNGPAIR.finditer(log_text):
        _r = _g_rung.setdefault(int(_m.group(1)), [0, 0])
        _r[0] += int(_m.group(3)); _r[1] += int(_m.group(4))
    for _m in _GATE_IMCONV.finditer(log_text):
        _g_imconv.setdefault((_m.group(1), _m.group(2)), []).append(
            (int(_m.group(4)), int(_m.group(7)), float(_m.group(9))))
    for _m in _GATE_ARMED.finditer(log_text):
        _g_armed[_m.group(1)] = _m.group(2).strip()
    for _m in _GATE_ACTIVE.finditer(log_text):
        _g_active.setdefault(_m.group(1), []).append(int(_m.group(2)))
    for _m in _GATE_STAGE_END.finditer(log_text):
        _t = _m.group(2).strip()
        if ("STAGE COMPLETE" in _t or "holding the stage open" in _t
                or "still active" in _t):
            _g_stage_lines.append(f"{_m.group(1)}: {_t}")

if _g_shut or _g_vert or _g_ll_unit or _g_imconv:
    _np_ = 4
    fig, ax = plt.subplots(1, _np_, figsize=(16.5, 3.4))

    # --- 1. shutoff progress = progress toward the next recipe stage ----
    # The ACTIVE count is the real progress series: it is what has to
    # reach ZERO for the stage to end, it is reported every 10 checks, and
    # -- unlike "% shut" -- it moves when new births create new occupied
    # pairs, which is the thing that can push the stage end AWAY.
    if _g_active:
        for _st, _seq in sorted(_g_active.items()):
            ax[0].plot(range(1, len(_seq) + 1), _seq, marker="o", ms=3,
                       color=AMBER, label=f"{_st} still active")
            if len(_seq) >= 2:
                _tr = _seq[-1] - _seq[0]
                ax[0].text(
                    0.03, 0.06,
                    ("RISING +%d over %d checks: births are opening pairs "
                     "faster than\nthey shut, so the stage end is moving "
                     "AWAY" % (_tr, len(_seq))) if _tr > 0 else
                    ("falling %d over %d checks" % (_tr, len(_seq))),
                    color=(RED if _tr > 0 else GREEN), fontsize=7,
                    transform=ax[0].transAxes)
        ax[0].axhline(0.0, color=GREEN, ls="--", lw=1)
        ax[0].set_ylabel("occupied pairs still active")
        ax[0].set_xlabel("stage check")
        ax[0].legend(fontsize=7)
        ax[0].set_title("progress to stage end (0 = stage ends)")
    elif _g_shut:
        for _mv, _seq in sorted(_g_shut.items()):
            _y = [100.0 * s / max(t, 1) for s, t in _seq]
            ax[0].plot(range(1, len(_y) + 1), _y, marker="o", ms=3,
                       label=_mv)
        ax[0].axhline(100.0, color=GREEN, ls="--", lw=1)
        ax[0].set_ylim(0, 105)
        ax[0].legend(fontsize=7)
        ax[0].set_ylabel("% of pairs shut")
        ax[0].set_xlabel("shutoff report")
        ax[0].set_title("(walker, band) pairs shut off")
    else:
        ax[0].text(0.5, 0.5, "no shutoff reports", ha="center",
                   va="center", color=DIM, transform=ax[0].transAxes)
        ax[0].set_title("progress to stage end")

    # --- 2. the [GB_CELL_LL] instrument ---------------------------------
    if _g_ll_unit:
        _mvs = sorted(set(_g_ll_unit) | set(_g_ll_exc))
        _rate = [100.0 * len(_g_ll_exc.get(m, [])) / max(_g_ll_unit.get(m, 1), 1)
                 for m in _mvs]
        _xx = np.arange(len(_mvs))
        _cols = [RED if r > 50 else (AMBER if r > 5 else GREEN)
                 for r in _rate]
        ax[1].bar(_xx, _rate, color=_cols, alpha=0.85)
        for _i, m in enumerate(_mvs):
            _c = _g_ll_cold.get(m, 0)
            ax[1].text(_i, min(_rate[_i] + 3, 100), f"{_c} cold",
                       ha="center", fontsize=7, color=FG)
        ax[1].set_xticks(_xx)
        ax[1].set_xticklabels([m[:13] for m in _mvs], rotation=35,
                              ha="right", fontsize=7)
        ax[1].set_ylim(0, 112)
    else:
        ax[1].text(0.5, 0.5, "no [GB_CELL_LL] reports", ha="center",
                   va="center", color=DIM, transform=ax[1].transAxes)
    ax[1].set_title("cell-ll reconciliation: % of units over allowance")
    ax[1].set_ylabel("% exceeding")

    # --- 3. the vertical ladder, per rung pair --------------------------
    if _g_rung:
        _ks = sorted(_g_rung)
        _acc = [100.0 * _g_rung[k][0] / max(_g_rung[k][1], 1) for k in _ks]
        ax[2].plot(_ks, _acc, marker="o", ms=3, color=CYAN)
        ax[2].axhline(95.0, color=RED, ls="--", lw=1)
        ax[2].text(0.02, 0.05, "→95% would mean empty↔empty leaked in",
                   color=RED, fontsize=7, transform=ax[2].transAxes)
        ax[2].set_ylim(0, 105)
        ax[2].set_xlabel("cold rung of the pair (T→T+1)")
    else:
        ax[2].text(0.5, 0.5, "no [GB_VERT] rung-pair data", ha="center",
                   va="center", color=DIM, transform=ax[2].transAxes)
    ax[2].set_title("vertical swap acceptance per rung pair")
    ax[2].set_ylabel("% accepted")

    # --- 4. in-model convergence: newborn vs survivor -------------------
    if _g_imconv:
        _lbl, _p50, _flr, _atf = [], [], [], []
        for (_mv, _cls), _rows in sorted(_g_imconv.items()):
            _lbl.append(f"{_mv[:11]}\n{_cls}")
            _p50.append(float(np.median([r[0] for r in _rows])))
            _flr.append(float(np.median([r[1] for r in _rows])))
            _atf.append(float(np.median([r[2] for r in _rows])))
        _xx = np.arange(len(_lbl))
        ax[3].bar(_xx, _p50, color=VIOLET, alpha=0.85, label="median repeats")
        ax[3].plot(_xx, _flr, "_", color=AMBER, ms=18, mew=2,
                   label="the floor (window+1)")
        for _i, _a in enumerate(_atf):
            ax[3].text(_i, _p50[_i], f"{_a:.0f}% at floor", ha="center",
                       va="bottom", fontsize=6.5, color=FG)
        ax[3].set_xticks(_xx)
        ax[3].set_xticklabels(_lbl, fontsize=6.5)
        ax[3].legend(fontsize=7)
    else:
        ax[3].text(0.5, 0.5, "no [GB_IMCONV] calibration", ha="center",
                   va="center", color=DIM, transform=ax[3].transAxes)
    ax[3].set_title("in-model repeats vs the floor")
    ax[3].set_ylabel("repeats")

    fig.tight_layout()
    fig_b64(fig, "search_gates")

    # ---- the companion table: what each gate is SET to, and reaching ---
    _rows_html = []
    # ONE ROW, NOT ONE PER MOVE. Every armed move reads the SAME shared
    # band_rj_shutoff_w table (recipe._band_shutoff_w_pending dedupes on
    # id(table)), and only one of them happens to emit the count line. A
    # per-move row therefore reported "armed, 0 converged" for moves that
    # were in fact sharing the 31% -- which reads as a broken gate.
    if _g_live or _g_shut:
        _shapes = set(_g_live.values())
        _all_seq = [v for seq in _g_shut.values() for v in seq]
        _nw, _nb, _win = (sorted(_shapes)[0] if _shapes else (0, 0, 0))
        if _all_seq:
            _s, _t = _all_seq[-1]
            _pct = 100.0 * _s / max(_t, 1)
            _active = (_g_active.get(sorted(_g_active)[0], [])
                       if _g_active else [])
            _trend = ""
            if len(_active) >= 2:
                _d = _active[-1] - _active[0]
                _trend = (
                    f" &mdash; <span style='color:var(--red)'>still-active "
                    f"RISING +{_d}</span>" if _d > 0 else
                    f" &mdash; still-active falling {_d}")
            _state = (f"<span style='color:var(--green)'>{_s:,} / {_t:,} "
                      f"({_pct:.0f}%)</span>{_trend}")
        else:
            _state = ("<span style='color:var(--amber)'>armed; no pair has "
                      "converged yet</span>")
        _shared = ", ".join(sorted(set(_g_live) | set(_g_shut)))
        _rows_html.append(
            f"<tr><td>{_shared}</td><td>RJ shutoff "
            f"<i>(one shared table)</i></td>"
            f"<td>{_nw}w x {_nb}b, window {_win}</td><td>{_state}</td></tr>")
    for _mv, _txt in sorted(_g_armed.items()):
        _win = re.search(r"window (\d+)[^/]*/ (\d+)", _txt)
        _dll = re.search(r"dll ([\d.]+)", _txt)
        _rows_html.append(
            f"<tr><td>{_mv}</td><td>in-model convergence</td>"
            f"<td>window {_win.group(1)}/{_win.group(2)} (newborn/survivor)"
            f", dll {_dll.group(1) if _dll else '?'}</td>"
            f"<td>{len(_g_imconv.get((_mv, 'newborn'), [])):,} newborn / "
            f"{len(_g_imconv.get((_mv, 'mature'), [])):,} mature reports</td>"
            f"</tr>")
    if _g_vert:
        _vp = sum(p for v in _g_vert.values() for p, a in v)
        _va = sum(a for v in _g_vert.values() for p, a in v)
        _rows_html.append(
            f"<tr><td>all gb moves</td><td>vertical swap</td>"
            f"<td>{'ALL RUNGS' if _g_drop else 'picked rows only'}</td>"
            f"<td>{_va:,} / {_vp:,} accepted ({100.0 * _va / max(_vp, 1):.1f}%)"
            + (f"; {_g_drop:,} unpriceable pair(s) dropped" if _g_drop else "")
            + "</td></tr>")
    GATES_TABLE = (
        "<table style='width:100%;border-collapse:collapse;font-size:12px'>"
        "<tr><th style='text-align:left'>move</th>"
        "<th style='text-align:left'>gate</th>"
        "<th style='text-align:left'>configured</th>"
        "<th style='text-align:left'>reaching</th></tr>"
        + "".join(_rows_html) + "</table>") if _rows_html else ""
    _stage_html = ""
    if _g_stage_lines:
        _stage_html = (
            "<div class='panel'><b>Stage-end decisions, most recent last</b>"
            "<div class='caption'>" + "<br>".join(
                s.replace("&", "&amp;").replace("<", "&lt;")
                for s in _g_stage_lines[-8:])
            + "</div></div>")
    NAV_GATES = '<a href="#gates">search gates</a>'
    GATES_HTML = f"""
<section id="gates"><h2>Search Gates</h2>
{GATE_WHY_PANEL}
<div class="panel">{img("search_gates", "search gate status")}
<div class="caption"><b>Every one of these gates can be armed and reaching
nothing</b>, which is the failure shape this run has hit repeatedly, so each
panel answers "is it doing anything" rather than "what is it set to".
<b>Left:</b> the share of occupied (walker, band) pairs that have shut off.
Since <code>GB_SEARCH_STAGE_END_ON_SHUTOFF</code> this IS the stage-end
criterion &mdash; 100% ends the recipe stage, so this curve is progress
toward the next stage, not just a diagnostic.
<b>Second:</b> the <code>[GB_CELL_LL]</code> sampled-vs-realized check, the
only instrument that catches a mis-credited cell ledger. The bar is the share
of unit reports over their temperature-scaled allowance and the annotation is
the count at the COLD rung, which is the half that reaches the catalogue.
&#9888; a high bar is NOT news by itself &mdash; it was already 98&ndash;100%
for the in-model and RJ moves before the all-rungs swap existed; what matters
is this moving relative to that baseline.
<b>Third:</b> vertical-swap acceptance per rung pair. It should rise smoothly
with temperature; a pair approaching 100% would mean empty&harr;empty pairs
leaked into the proposal set, which are accepted unconditionally and would
collapse that band's ladder.
<b>Right:</b> in-model repeats per provenance class against the floor
(window+1). A bar sitting ON its floor marker means the window is a fixed
budget those rows serve out rather than a measurement &mdash; the reason the
survivor window went 100&nbsp;&rarr;&nbsp;50.</div></div>
{GATES_TABLE}
{_stage_html}
</section>
"""

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
_empty_branches = []
for arrs, name, a in [((psd_sw_a, psd_sw_p), "psd", 0), ((gal_sw_a, gal_sw_p), "galfor", 1)]:
    sa, sp = arrs
    # Empty swap-count arrays happen on a very young / mid-flush snapshot
    # where SUB_NIT has trimmed the whole stream away. The prior fallback
    # ``sp.shape[0] - 1`` then landed at -1 and sa[-1] raised IndexError.
    # Skip the branch with an empty panel + a MISSING reason instead.
    if sp.shape[0] == 0:
        ax[a].set_title(f"{name} swap acceptance (no rows in this snapshot)")
        ax[a].set_xlabel("rung"); ax[a].set_ylim(0, 1)
        _empty_branches.append(name)
        continue
    nz = np.where(sp.sum(axis=1) > 0)[0]
    k_it = int(nz.max()) if nz.size else sp.shape[0] - 1
    rate = sa[k_it] / np.maximum(sp[k_it], 1)
    ax[a].bar(np.arange(len(rate)), rate, color=CYAN if a == 0 else AMBER, alpha=0.85)
    ax[a].set_title(f"{name} swap acceptance per rung (iter {k_it})")
    ax[a].set_xlabel("rung"); ax[a].set_ylim(0, 1)
if _empty_branches:
    MISSING.append(
        f"swap-acceptance panel: {'/'.join(_empty_branches)} sub-backend "
        f"has no rows in this snapshot (young run or mid-flush trim to "
        f"SUB_NIT=0); those panels render empty.")
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

# ======================= SOURCE BRANCH DROPDOWNS ============================
# Per-leaf chain-trace + marginal-posterior PNGs for the mbh / emri / sobbh
# branches (user ruling 2026-10-05: "chain plots and posterior estimates for
# all those new sources ... same style with parameter chains and estimated
# posteriors"). The PNGs are pre-rendered server-side and swapped in by the
# same ``cornerPanel(px, blob)`` JS helper the VGB and GB top-3 dropdowns
# use, so the HTML only carries one <img> per branch regardless of how many
# leaves the run holds.
#
# Truth lines come from the mojito catalogues under MOJITO_CAT_DIR, keyed by
# the per-branch ``source_ids`` list parsed out of
# ``processor_init_kwargs`` in run_settings.log (the ``source_ids`` entry is
# the mojito-loader 0-based dict key, i.e. the positional row in the
# ID-sorted catalogue). A branch with an unreadable chain, unreadable
# catalogue, or no run_settings source_ids degrades to an empty blob and a
# MISSING line -- nothing on the page raises.
_SRC_BRANCH_SPECS = {
    "mbh": dict(
        cls="MBHB",
        param_defs=[
            ("logM",      r"$\log M_{\rm tot}$"),
            ("Q",         r"$Q=m_1/m_2$"),
            ("s1z",       r"$\chi_1$"),
            ("s2z",       r"$\chi_2$"),
            ("dist",      r"$d_L$ [Gpc]"),
            ("phi_ref",   r"$\phi_{\rm ref}$"),
            ("cos_iota",  r"$\cos\iota$"),
            ("psi",       r"$\psi$"),
            ("alpha",     r"$\alpha$ (RA)"),
            ("sin_delta", r"$\sin\delta$"),
            ("t_plunge",  r"$t_c$ [s]"),
        ],
        key_idx=[0, 1, 2, 3, 10, 4, 8, 9],   # M, Q, chi1, chi2, tc, dist, RA, sin_dec
        cat_files=(
            "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5",
            "mbhb_cat_mojito_lite_processed_MT.hdf5",
        ),
    ),
    "emri": dict(
        cls="EMRI",
        param_defs=[
            ("logm1",    r"$\log m_1$"),
            ("m2",       r"$m_2$"),
            ("a",        r"$a$"),
            ("p0",       r"$p_0$"),
            ("e0",       r"$e_0$"),
            ("dist",     r"$d_L$"),
            ("qS",       r"$q_S$"),
            ("phiS",     r"$\phi_S$"),
            ("qK",       r"$q_K$"),
            ("phiK",     r"$\phi_K$"),
            ("Phi_phi0", r"$\Phi_{\phi,0}$"),
            ("Phi_r0",   r"$\Phi_{r,0}$"),
        ],
        key_idx=[0, 2, 3, 4, 5],   # logm1 (M), a, p0, e0, dist
        cat_files=("emri_cat_mojito_lite_processed_MT.hdf5",),
    ),
    "sobbh": dict(
        cls="SOBHB",
        param_defs=[
            ("logm1",  r"$\log m_1$"),
            ("logm2",  r"$\log m_2$"),
            ("s1",     r"$\chi_1$"),
            ("s2",     r"$\chi_2$"),
            ("dist",   r"$d_L$"),
            ("cosinc", r"$\cos\iota$"),
            ("f_low",  r"$f_{\rm low}$ [Hz]"),
            ("phiS",   r"$\phi_S$"),
            ("cosqS",  r"$\cos q_S$"),
            ("psi",    r"$\psi$"),
            ("phi0",   r"$\phi_0$"),
        ],
        key_idx=[0, 1, 6, 4, 7, 8],   # logm1, logm2, f_low, dist, phiS, cosqS
        cat_files=("sobhb_cat_mojito_lite_processed_MT.hdf5",),
    ),
}


def _parse_src_ids_from_settings(txt):
    """Dict {cls: [source_ids]} from processor_init_kwargs in run_settings.log.

    These are mojito-loader 0-based dict keys, i.e. positional rows in the
    ID-sorted catalogue -- the same thing the engine iterates when it builds
    per-leaf injections (``sorted(cat.keys())``).  Returns ``{}`` on parse
    failure so the branch panels degrade to no-truth without raising.
    """
    m = re.search(r"'source_ids':\s*\{([^}]*)\}", txt, re.S)
    if not m:
        return {}
    blob = m.group(1)
    out = {}
    for mm in re.finditer(r"'(MBHB|EMRI|SOBHB)':\s*\[([^\]]*)\]", blob):
        try:
            out[mm.group(1)] = [int(x) for x in mm.group(2).replace(" ", "").split(",") if x]
        except ValueError:
            continue
    return out


_SRC_IDS = _parse_src_ids_from_settings(SETTINGS_TXT)


def _src_cat_to_sampling_mbh(row):
    """MBHB catalogue entry -> MBH sampling basis (direct ICRS)."""
    m1 = float(row["PrimaryMassSSBFrame"])
    m2 = float(row["SecondaryMassSSBFrame"])
    if m2 > m1:
        m1, m2 = m2, m1
    logM = np.log(m1 + m2)
    Q = m1 / m2
    s1z = float(row["PrimarySpinCompZ"])
    s2z = float(row["SecondarySpinCompZ"])
    dist = float(row["LuminosityDistance"]) / 1e3        # Mpc -> Gpc
    phi_ref = float(row["PhaseReferenceSourceFrame"]) % (2 * np.pi)
    cos_iota = float(np.cos(float(row["InclinationAngle"])))
    ra = float(row["RightAscension"]) % (2 * np.pi)
    sin_dec = float(np.sin(float(row["Declination"])))
    psi = float(row["PolarisationAngle"]) % np.pi
    tphenom = float(row.get("TimeCoalescencePhenomTPHMSSBFrame",
                            row.get("TimeCoalescenceSSBFrame", np.nan)))
    return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota,
                     psi, ra, sin_dec, tphenom])


def _src_cat_to_sampling_emri(row):
    """EMRI catalogue entry -> EMRI sampling basis (ecliptic colat sky)."""
    logm1 = np.log(float(row["PrimaryMassSSBFrame"]))
    m2 = float(row["SecondaryMassSSBFrame"])
    a = float(row["PrimarySpinParameter"])
    p0 = float(row["SemiLatusRectum"])
    e0 = float(row["Eccentricity"])
    dist = float(row["LuminosityDistance"]) / 1e3
    qS = np.pi / 2 - float(row["Declination"])            # ecliptic colatitude
    phiS = float(row["RightAscension"]) % (2 * np.pi)
    qK = float(row["PolarAnglePrimarySpin"])
    phiK = float(row["AzimuthalAnglePrimarySpin"]) % (2 * np.pi)
    Phi_phi0 = float(row["AzimuthalPhase"]) % (2 * np.pi)
    Phi_r0 = float(row["RadialPhase"]) % (2 * np.pi)
    return np.array([logm1, m2, a, p0, e0, dist, qS, phiS, qK, phiK,
                     Phi_phi0, Phi_r0])


def _src_cat_to_sampling_sobbh(row):
    """SOBHB catalogue entry -> SOBBH sampling basis."""
    m1 = float(row["PrimaryMassSSBFrame"])
    m2 = float(row["SecondaryMassSSBFrame"])
    logm1 = np.log(m1)
    logm2 = np.log(m2)
    s1 = float(row["PrimarySpinCompZ"])
    s2 = float(row["SecondarySpinCompZ"])
    dist = float(row["LuminosityDistance"]) / 1e3
    cosinc = float(np.cos(float(row["InclinationAngle"])))
    f_low = float(row["GW22FrequencySSBFrame"])
    phiS = float(row["RightAscension"]) % (2 * np.pi)
    cosqS = float(np.cos(np.pi / 2 - float(row["Declination"])))
    psi = float(row["PolarisationAngle"]) % np.pi
    phi0 = float(row["TrueAnomaly"]) % (2 * np.pi)
    return np.array([logm1, logm2, s1, s2, dist, cosinc, f_low,
                     phiS, cosqS, psi, phi0])


_SRC_CAT_FNS = {
    "mbh": _src_cat_to_sampling_mbh,
    "emri": _src_cat_to_sampling_emri,
    "sobbh": _src_cat_to_sampling_sobbh,
}


def _src_load_catalogue(cat_file_candidates):
    """Open the first catalogue file that exists -> ``{col: ndarray}`` or None."""
    for fn in cat_file_candidates:
        p = os.path.join(MOJITO_CAT_DIR, "catalogues", fn)
        if os.path.exists(p):
            try:
                with h5py.File(p, "r") as _fh:
                    B = _fh["Binaries"]
                    return {k: np.asarray(B[k][:]) for k in B.keys()}
            except Exception as e:
                MISSING.append(f"source catalogue {fn} unreadable: {e!r}")
    return None


def _src_cat_row(cat, loader_key):
    """Row at position ``loader_key`` of the ID-sorted catalogue, as a dict.

    The mojito loader's dict key is the 0-based positional index into
    sorted(IDs), not the ID column itself (EMRI's ID column is 1-indexed;
    MBHB / SOBHB use 0-indexed dense IDs).  Positional lookup absorbs all
    three conventions and matches ``[sorted(cat.keys())][leaf]``.
    """
    if cat is None:
        return None
    ids = np.asarray(cat["ID"]).astype(int)
    order = np.argsort(ids)
    if not (0 <= int(loader_key) < order.size):
        return None
    i = int(order[int(loader_key)])
    return {k: v[i] for k, v in cat.items()}


# full_pe start iteration -- the recipe stamp written by the engine when the
# full_pe stage begins. Panels mark it as a vertical line on the trace and
# use ``samples[start_iteration:]`` as the posterior window. ``None`` when
# full_pe has not started: the posterior panel then says so, and the trace
# simply has no mark.
_SRC_FULL_PE_START = None
if rg is not None and "full_pe" in rg:
    _si = rg["full_pe"].attrs.get("start_iteration", None)
    if _si is not None and 0 <= int(_si) < NIT:
        _SRC_FULL_PE_START = int(_si)


def _src_pool_window(nit):
    """Trailing iterations the corner-plot pool draws from.

    User ruling 2026-10-05: last 50 iterations (x 4 walkers = 200 pooled
    samples) when the branch has < 500 rows; last 250 (x 4 = 1000) once
    it is past that mark.  The step is a BURN-IN GATE, not a threshold:
    at <500 iterations the trailing 50 is where the branch currently sits
    and anything older is adaptation; once there are >=500 rows, the
    first 250 are the burn-in to drop and the second half is the
    posterior.  Clamped to the available rows so very young stores still
    show something -- an empty window would make the corner panel claim
    "no samples" at a time when the user is actively watching the
    sampler fire.
    """
    pool = 250 if nit > 500 else 50
    return min(pool, max(1, int(nit)))


def _src_make_leaf_png(
    *,
    branch,
    leaf_idx,
    label,
    chain,          # (nit, nwalkers, ndim) cold-chain
    inds,           # (nit, nwalkers) bool
    truth,          # (ndim,) or None
    param_defs,
    key_idx,
    dh=None,        # (nit, nwalkers) or None
    hh=None,
    dpi=90,
    tally=None,
):
    """Render one trace + corner-posterior PNG for one leaf.

    Layout: ``(1 + npar) x (npar + 1)`` grid.

      row 0:        chain traces per key param (full iteration range),
                    right-column pane = <d|h> / <h|h> diagnostic.
      rows 1..npar: corner panel.  Diagonal cell (i, i) is a 1-D histogram
                    of pooled samples for key param i; cell (i, j) with
                    j < i is a scatter of those samples (x = param j,
                    y = param i).  Upper triangle (j > i) is hidden
                    except for the right-column summary table that spans
                    all corner rows.

    Pooled samples come from ``_src_pool_window(nit)`` trailing
    iterations x all cold walkers (user ruling 2026-10-05: last 50 iters
    when < 500 stored, last 250 once past that; the step is the burn-in
    gate for the full_pe stage).  Dead rows (``inds == False``) are
    dropped per iteration so a trans-dimensional branch cannot draw a
    phantom source at coordinate zero.

    Returns a base64-encoded PNG string.
    """
    full_pe_start = _SRC_FULL_PE_START
    nit, nwalkers, ndim = chain.shape
    it_axis = np.arange(nit)
    npar = len(key_idx)
    _walker_colors = [CYAN, AMBER, GREEN, VIOLET]
    # ---- pool: last W iterations, mask dead rows per iteration -----------
    pool_W = _src_pool_window(nit)
    pool_start = max(0, nit - pool_W)
    pool_chain = chain[pool_start:nit]        # (W, nwalkers, ndim)
    pool_inds = inds[pool_start:nit].astype(bool)
    pool_mask_flat = pool_inds.reshape(-1)    # (W * nwalkers,)
    pool_flat = pool_chain.reshape(-1, ndim)[pool_mask_flat]
    n_pooled = int(pool_flat.shape[0])

    # Figure geometry: traces row (fat) + npar corner rows (square-ish).
    # Width per column: 1.75" param + 2.0" diagnostic.
    _col_w = 1.75
    _diag_w = 2.1
    _trace_h = 1.5
    _corner_h = 0.75
    w_total = max(_col_w * npar + _diag_w, 8.0)
    h_total = _trace_h + _corner_h * npar + 0.6     # +0.6 for suptitle margin
    fig = plt.figure(figsize=(w_total, h_total), dpi=dpi)
    width_ratios = [1.0] * npar + [_diag_w / _col_w]
    height_ratios = [_trace_h] + [_corner_h] * npar
    gs = fig.add_gridspec(
        1 + npar, npar + 1,
        width_ratios=width_ratios, height_ratios=height_ratios,
        hspace=0.18, wspace=0.18,
        left=0.055, right=0.985,
        top=1.0 - 0.55 / max(h_total, 1.0),
        bottom=0.55 / max(h_total, 1.0),
    )

    # ---- row 0: chain traces per key param ------------------------------
    for col, pcol in enumerate(key_idx):
        _, lbl = param_defs[pcol]
        ax = fig.add_subplot(gs[0, col])
        for w in range(nwalkers):
            vals = chain[:, w, pcol].astype(float)
            mask = inds[:, w].astype(bool)
            if mask.any():
                ax.plot(it_axis[mask], vals[mask],
                        color=_walker_colors[w % 4], lw=0.6, alpha=0.85)
        if truth is not None and np.isfinite(truth[pcol]):
            ax.axhline(float(truth[pcol]), color=RED, ls=":",
                       lw=1.0, alpha=0.95)
        # mark the pool window start (burn-in cut-off that feeds the corner)
        ax.axvline(pool_start, color=CYAN, ls="-", lw=0.6, alpha=0.6)
        if full_pe_start is not None and full_pe_start < nit:
            ax.axvline(full_pe_start, color=DIM, ls="--", lw=0.6, alpha=0.6)
        ax.set_title(lbl, fontsize=9, pad=2)
        ax.tick_params(axis="both", labelsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4),
                             useOffset=False, useMathText=True)
        if col == 0:
            ax.set_ylabel("value", fontsize=8)

    # Row 0 right-column: d|h / h|h diagnostic trace
    ax = fig.add_subplot(gs[0, -1])
    if dh is not None or hh is not None:
        for w in range(nwalkers):
            if dh is not None:
                ax.plot(it_axis, dh[:, w], color=_walker_colors[w % 4],
                        lw=0.6, alpha=0.85)
            if hh is not None:
                ax.plot(it_axis, hh[:, w], color=_walker_colors[w % 4],
                        lw=0.6, alpha=0.4, ls="--")
        ax.axvline(pool_start, color=CYAN, ls="-", lw=0.6, alpha=0.6)
        if full_pe_start is not None and full_pe_start < nit:
            ax.axvline(full_pe_start, color=DIM, ls="--", lw=0.6, alpha=0.6)
        ax.set_title(r"$\langle d|h\rangle$ solid, $\langle h|h\rangle$ dashed",
                     fontsize=8, pad=2)
        ax.tick_params(axis="both", labelsize=7)
    else:
        ax.axis("off")

    # ---- rows 1..npar: corner panel -------------------------------------
    # Compute per-param axis extents from pooled samples (padded by 5%).
    extents = []
    for pcol in key_idx:
        vals = pool_flat[:, pcol] if n_pooled else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        else:
            lo, hi = 0.0, 1.0
        if truth is not None and np.isfinite(truth[pcol]):
            tv = float(truth[pcol])
            lo = min(lo, tv); hi = max(hi, tv)
        if not hi > lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-9)
        pad = (hi - lo) * 0.06
        extents.append((lo - pad, hi + pad))

    for i, pcol_y in enumerate(key_idx):
        for j, pcol_x in enumerate(key_idx):
            ax = fig.add_subplot(gs[1 + i, j])
            if j > i:
                ax.axis("off")
                continue
            if j == i:
                # Diagonal: 1-D histogram of pooled samples
                vals = (pool_flat[:, pcol_x] if n_pooled else np.array([]))
                vals = vals[np.isfinite(vals)] if vals.size else vals
                if vals.size >= 3:
                    ax.hist(vals, bins=24, color=VIOLET, alpha=0.75,
                            edgecolor=DIM, lw=0.3)
                else:
                    ax.text(0.5, 0.5, "n=%d" % n_pooled,
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=7, color=DIM)
                if truth is not None and np.isfinite(truth[pcol_x]):
                    ax.axvline(float(truth[pcol_x]), color=RED, ls=":", lw=1.0)
                ax.set_xlim(*extents[j])
                ax.set_yticks([])
            else:
                # Lower triangle: 2-D scatter dots
                if n_pooled:
                    xv = pool_flat[:, pcol_x]
                    yv = pool_flat[:, pcol_y]
                    ok = np.isfinite(xv) & np.isfinite(yv)
                    xv, yv = xv[ok], yv[ok]
                    if xv.size:
                        ax.scatter(xv, yv, s=3.2, color=VIOLET, alpha=0.55,
                                   lw=0)
                if truth is not None and np.isfinite(truth[pcol_x]) and \
                        np.isfinite(truth[pcol_y]):
                    ax.plot(float(truth[pcol_x]), float(truth[pcol_y]),
                            "+", color=RED, mew=1.2, ms=8)
                ax.set_xlim(*extents[j]); ax.set_ylim(*extents[i])

            # Axis labels: left column -> y-axis param label; bottom row ->
            # x-axis param label.  Hide ticks elsewhere so the N x N grid
            # does not drown in numbers.
            if j == 0:
                ax.set_ylabel(param_defs[pcol_y][1], fontsize=7.5)
                ax.tick_params(axis="y", labelsize=6)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4),
                                     useOffset=False, useMathText=True)
            else:
                ax.set_yticklabels([])
            if i == npar - 1:
                ax.set_xlabel(param_defs[pcol_x][1], fontsize=7.5)
                ax.tick_params(axis="x", labelsize=6, rotation=0)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4),
                                     useOffset=False, useMathText=True)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3, lw=0.4)

    # ---- summary / diagnostic column (spans corner rows) -----------------
    ax = fig.add_subplot(gs[1:, -1])
    ax.axis("off")
    lines = [
        f"iter range: 0 .. {nit - 1}",
        f"pool window: last {pool_W} iters (start {pool_start}, "
        f"cutoff at <500 -> 50, >=500 -> 250)",
        f"pooled samples: {n_pooled} ({pool_W} x {nwalkers} nominal, "
        f"minus dead-leaf rows)",
    ]
    if full_pe_start is not None:
        lines.append(f"full_pe: {max(0, nit - full_pe_start)} iters "
                     f"(start {full_pe_start})")
    if dh is not None and hh is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = dh[-1] / np.sqrt(np.maximum(hh[-1], 1e-30))
        lines.append(f"SNR  last: {np.round(snr, 1).tolist()}")
        lines.append(f"d|h  last: {np.round(dh[-1], 0).tolist()}")
        lines.append(f"h|h  last: {np.round(hh[-1], 0).tolist()}")
    if truth is not None:
        lines.append("")
        lines.append(" param    median    truth   (m-t)/sig")
        for pcol in key_idx:
            name = param_defs[pcol][0]
            vals_p = (pool_flat[:, pcol] if n_pooled else np.array([]))
            vals_p = vals_p[np.isfinite(vals_p)] if vals_p.size else vals_p
            if vals_p.size:
                med = float(np.median(vals_p))
                sd = float(np.std(vals_p)) or np.nan
                tv = (float(truth[pcol])
                      if np.isfinite(truth[pcol]) else np.nan)
                diff_sigma = ((med - tv) / sd
                              if (sd and np.isfinite(sd)) else np.nan)
                lines.append(
                    f" {name:<8} {med: .3g}  {tv: .3g}  {diff_sigma: .2f}")
    ax.text(0.0, 1.0, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=FG, family="monospace")

    fig.suptitle(f"{branch.upper()} leaf {leaf_idx} — {label}",
                 fontsize=10, color=FG, y=0.995)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    raw = _shrink_png(buf.getvalue())
    if tally is not None:
        tally.append(len(raw))
    return base64.b64encode(raw).decode()


SRC_PANEL_JSON = {"mbh": '{"src":[]}', "emri": '{"src":[]}',
                  "sobbh": '{"src":[]}'}
SRC_PANEL_META = {}
_SRC_PNG_BYTES = []
_src_t0 = time.perf_counter() if False else None  # time tracked below via `time`
import time as _src_time_mod  # local import, keep page-build time clean
_src_build_start = _src_time_mod.perf_counter()
for _branch, _spec in _SRC_BRANCH_SPECS.items():
    _cls = _spec["cls"]
    _chain_ds = _safe(g, f"chain/{_branch}", label=f"chain/{_branch}")
    _inds_ds = _safe(g, f"inds/{_branch}", label=f"inds/{_branch}")
    if _chain_ds is None or _inds_ds is None:
        SRC_PANEL_META[_branch] = {"status": "chain or inds unreadable"}
        continue
    try:
        _chain_all = _chain_ds[:, 0, 0]   # (nit, nwalkers, nleaves, ndim)
        _inds_all = _inds_ds[:, 0, 0]
    except Exception as e:
        MISSING.append(f"{_branch} chain reshape failed: {e!r}")
        SRC_PANEL_META[_branch] = {"status": f"shape error: {e!r}"}
        continue
    if _chain_all.ndim != 4 or _inds_all.ndim != 3:
        MISSING.append(
            f"{_branch} chain/inds shape unexpected: chain={_chain_all.shape} "
            f"inds={_inds_all.shape}; dropdown skipped")
        continue
    _nwalkers = _chain_all.shape[1]
    _nleaves = _chain_all.shape[2]
    _sub_dh = _safe(g, f"sub_backend/{_branch}/d_h",
                     label=f"sub_backend/{_branch}/d_h")
    _sub_hh = _safe(g, f"sub_backend/{_branch}/h_h",
                     label=f"sub_backend/{_branch}/h_h")
    _dh_all = _sub_dh if _sub_dh is not None else None
    _hh_all = _sub_hh if _sub_hh is not None else None
    _cat = _src_load_catalogue(_spec["cat_files"])
    _ids = _SRC_IDS.get(_cls, list(range(_nleaves)))
    if _cat is None:
        MISSING.append(
            f"{_branch} truth lines unavailable: no catalogue under "
            f"{MOJITO_CAT_DIR}/catalogues matching {_spec['cat_files']!r}")
    _truths = []
    for _leaf in range(_nleaves):
        if _cat is None or _leaf >= len(_ids):
            _truths.append(None); continue
        _row = _src_cat_row(_cat, _ids[_leaf])
        if _row is None:
            _truths.append(None); continue
        try:
            _truths.append(_SRC_CAT_FNS[_branch](_row))
        except Exception as e:
            MISSING.append(
                f"{_branch} leaf {_leaf} truth build failed: {e!r}")
            _truths.append(None)
    _blob = {"src": [], "nleaves": int(_nleaves), "nwalkers": int(_nwalkers)}
    _n_fail = 0
    _per_branch_bytes = []
    for _leaf in range(_nleaves):
        _chain = _chain_all[:, :, _leaf, :]
        _inds = _inds_all[:, :, _leaf]
        _dh = _dh_all[:, :, _leaf] if _dh_all is not None else None
        _hh = _hh_all[:, :, _leaf] if _hh_all is not None else None
        _src_id = _ids[_leaf] if _leaf < len(_ids) else _leaf
        _label = (f"src id {_src_id}" if _cat is not None
                  else f"src idx {_leaf}")
        if _dh is not None and _hh is not None and _dh.size and _hh.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                _snr = float(np.median(
                    _dh[-1] / np.sqrt(np.maximum(_hh[-1], 1e-30))))
            if np.isfinite(_snr):
                _label += f" · SNR~{_snr:.1f}"
        try:
            _png = _src_make_leaf_png(
                branch=_branch, leaf_idx=_leaf, label=_label,
                chain=_chain, inds=_inds, truth=_truths[_leaf],
                param_defs=_spec["param_defs"], key_idx=_spec["key_idx"],
                dh=_dh, hh=_hh, tally=_per_branch_bytes,
            )
        except Exception as e:
            _n_fail += 1
            MISSING.append(
                f"{_branch} leaf {_leaf} PNG render failed: {e!r}")
            continue
        _pool_W = _src_pool_window(int(_chain.shape[0]))
        _n_pe = (max(0, NIT - _SRC_FULL_PE_START)
                 if _SRC_FULL_PE_START is not None else 0)
        _sub = (f"leaf {int(_leaf)} · {int(_chain.shape[0])} iterations · "
                f"corner pool: last {_pool_W} iters x {int(_nwalkers)} cold "
                f"walkers (burn-in gate: <500 -> 50, &ge;500 -> 250) · "
                f"full_pe {_n_pe} rows (start "
                f"{_SRC_FULL_PE_START if _SRC_FULL_PE_START is not None else 'n/a'}) · "
                f"dotted red = catalogue truth")
        _blob["src"].append({"label": _label, "sub": _sub, "png": _png})
    SRC_PANEL_JSON[_branch] = json.dumps(_blob)
    SRC_PANEL_META[_branch] = {
        "nleaves": int(_nleaves),
        "n_rendered": len(_blob["src"]),
        "n_fail": _n_fail,
        "png_total_mb": sum(_per_branch_bytes) / 1024 ** 2,
    }
    _SRC_PNG_BYTES += _per_branch_bytes
_src_build_sec = _src_time_mod.perf_counter() - _src_build_start
if _SRC_PNG_BYTES:
    print(f"[source-dropdown] built {len(_SRC_PNG_BYTES)} leaf PNGs across "
          f"mbh/emri/sobbh in {_src_build_sec:.1f} s, "
          f"total {sum(_SRC_PNG_BYTES) / 1024**2:.2f} MB "
          f"(base64 ~{sum(_SRC_PNG_BYTES) * 4 / 3 / 1024**2:.2f} MB)")

# Captions / counts for the HTML side (used in the three new <section> blocks).
SRC_PANEL_COUNT_TXT = {}
for _b in ("mbh", "emri", "sobbh"):
    _m = SRC_PANEL_META.get(_b, {})
    _n = _m.get("n_rendered", 0)
    _nl = _m.get("nleaves", 0)
    _fail = _m.get("n_fail", 0)
    if _nl == 0:
        SRC_PANEL_COUNT_TXT[_b] = (
            "no chain found in this snapshot (branch not sampled yet, or "
            "chain/inds unreadable); dropdown is empty.")
    else:
        _fp = (f"the last {NIT - _SRC_FULL_PE_START} full_pe iteration(s)"
               if _SRC_FULL_PE_START is not None else "no full_pe rows yet")
        SRC_PANEL_COUNT_TXT[_b] = (
            f"{_n} of {_nl} leaves rendered; marginals are over {_fp}."
            + (f" {_fail} leaf render(s) failed." if _fail else ""))

MBH_PANEL_JSON = SRC_PANEL_JSON["mbh"]
EMRI_PANEL_JSON = SRC_PANEL_JSON["emri"]
SOBBH_PANEL_JSON = SRC_PANEL_JSON["sobbh"]
MBH_PANEL_COUNT = SRC_PANEL_COUNT_TXT.get("mbh", "")
EMRI_PANEL_COUNT = SRC_PANEL_COUNT_TXT.get("emri", "")
SOBBH_PANEL_COUNT = SRC_PANEL_COUNT_TXT.get("sobbh", "")

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
{_row("matched to a detectable injection (2-df proxy, per iteration)", lambda r: f"{r[2]:,}")}
{_row("completeness (proxy)", lambda r: pct(r[3]))}
{_row("purity (proxy)", lambda r: pct(r[4]))}
{_row("search iterations completed in total", lambda r: f"{r[5]}")}
</table>"""

# ---- captions, every number read off the arrays that made the figure ------
if SCI:
    cap_f2 = (
        f"Per stored iteration, completeness (solid, left axis) is the share "
        f"of the {SCI['ndet']} detectable injections with a model source "
        f"within {TOL_BINS:.0f} frequency bins and purity (dashed, right "
        f"axis) the share of model sources that so pair: the 2-df proxy, "
        f"because a waveform per pair per row is not affordable. At the last "
        f"row the headline criterion is {MATCH_CRIT_TXT}: completeness "
        f"{pct(SCI['completeness'])} and purity {pct(SCI['purity'])} "
        f"(proxy {pct(SCI.get('completeness_proxy', SCI['completeness']))} "
        f"and {pct(SCI.get('purity_proxy', SCI['purity']))}).")
    cap_f3 = (
        f"Noise-weighted overlap between each matched pair, maximised over an "
        f"overall phase. Read the lower panel's height at any overlap as the "
        f"NUMBER of sources recovered that well. Median "
        f"{SCI.get('mm_med', float('nan')):.2f}; "
        f"{pct(SCI.get('mm_hi', 0))} exceed 0.9.")
    cap_f4 = (
        f"Left: every model source in the amplitude-frequency plane, coloured "
        f"by optimal SNR. Right: the same plane split three ways; the "
        f"missed-detection X&apos;s are tiered by the injected source&rsquo;s "
        f"SNR re-evaluated under the LAST stored iteration&rsquo;s "
        f"max-ln L cold-walker PSD + foreground "
        f"(<span style='color:var(--truthyellow)'>yellow</span> 7&ndash;10, "
        f"<span style='color:var(--truthorange)'>orange</span> 10&ndash;20, "
        f"<span style='color:var(--truthred)'>red</span> &gt; 20) so the loud "
        f"misses pop first on a crowded plane AND the tiering tracks the "
        f"run as its foreground fit evolves{_LIVE_SNR_NOTE}. Recovery tracks "
        f"amplitude, and the misses concentrate along the faint edge rather "
        f"than anywhere structural. {MATCH_CRIT_HTML}")
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
    # or counted by the page's own 2-bin proxy match. Match-criterion note
    # still emitted so the reader knows what would be applied were the gate
    # opened.
    cap_f4 = (
        "Left: every model source in the amplitude-frequency plane, "
        "coloured by optimal SNR, over the detectable injections (grey). "
        "Right: injections and model overlaid in the same plane. The "
        "comparison is visual; no match criterion is applied on this page. "
        f"(Would be: {MATCH_CRIT_TXT}.)")
    cap_f5 = (
        "Source density per frequency bin: the full catalogue, its "
        "detectable subset, and the model population. The gap between the "
        "green and cyan curves is read by eye; no per-source matching is "
        "applied.")
    cap_f10 = (
        "How close model sources sit to each other in frequency, against "
        "the same distribution for the injections. Sub-bin spacing in the "
        "model relative to the injections indicates template sharing.")

# A PARTIAL DTR means the data/template/residual analysis raised partway
# through -- it is truthy but missing the keys the captions read. Rendering
# "nan of 0 bins" would be worse than saying nothing, so report it as a
# MISSING section instead. (Before 2026-09-18 this path hard-indexed
# _d['nbins'] and killed the whole page with a KeyError, which is how the
# galfor log-sampling bug above surfaced.)
if DTR and "nbins" not in DTR:
    MISSING.append(
        "data/template/residual captions skipped: the DTR analysis did not "
        "complete (no 'nbins'), so its summary numbers do not exist. The "
        "panels above are whatever it managed to produce. Most likely cause "
        "is a NaN sensitivity curve -- check the galfor sampling basis.")
if DTR and "nbins" in DTR:
    _d = DTR
    cap_f1 = (
        f"Power spectral density of the TDI X channel &mdash; not a strain "
        f"amplitude, and not an ASD. The gap between the instrument curve and "
        f"their sum is the galactic confusion, at most a factor "
        f"{_d.get('conf_ratio', float('nan')):.2f} in power, near "
        f"{_d.get('conf_f', float('nan')) * 1e3:.1f} mHz. "
        f"{_d.get('n_gb', '?')} "
        f"galactic-binary and {_d.get('n_vgb', '?')} verification-binary "
        f"templates are subtracted here. "
        f"Data, templates and residual share one window: "
        f"{globals().get('WIN_TAPER_NOTE', 'the run&rsquo;s own data taper')}. "
        f"Until 2026-09-24 this panel inherited the run&rsquo;s 2-wavelet "
        f"taper, which is right for the likelihood &mdash; it works in the "
        f"WDM domain and discards the tapered edges &mdash; but leaks badly "
        f"through a single whole-span FFT, putting scalloped lobes below "
        f"~1.4 mHz that were the window, not the data (146&times; "
        f"peak-to-trough, now 1.7&times;; in-band unchanged to 0.001%).")
    cap_f1b = (
        f"Lower panel: residual power over the fitted noise-plus-foreground "
        f"model, coloured by the Anderson&ndash;Darling Gaussianity p-value of "
        f"the whitened residual &mdash; dark bins are where the model is still "
        f"incomplete. The residual stays above the instrument-only curve in "
        f"{_d.get('nbins', 0) - _d.get('undersub', 0)} of "
        f"{_d.get('nbins', 0)} bins; "
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
if DTR and "nbins" in DTR:
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
        f"{int(_d['layer_dt'])} s, max-pooled to the image's pixel grid "
        f"({_d['wdm_bt']} time pixels"
        + (f" &times; {_d['wdm_bf']} layers" if _d.get("wdm_bf", 1) > 1 else "")
        + f" per drawn cell, the same in all three columns). Taking the "
        f"MAXIMUM, not a sample, keeps a one-pixel transient &mdash; an MBHB "
        f"merger is a single {int(_d['layer_dt'])} s column &mdash; visible in "
        f"every panel; it "
        f"also lifts the noise floor slightly against an unpooled view. "
        + (f"White ticks on the top and bottom edges mark the "
           f"{_d['wdm_mergers']} MBHB merger times from the MBH cold chain. "
           if _d.get("wdm_mergers") else "")
        + f"One shared linear scale per channel row, keyed to that row's data "
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

if _TORN_ROWS:
    _tr = sorted(set(_TORN_ROWS))
    MISSING.append(
        f"{len(_tr)} chain row read(s) failed in the store (\"filter returned "
        "failure during read\": a run writing the file while this page read "
        "it, or a save cut short): " + "; ".join(_tr[:12])
        + (f"; ... and {len(_tr) - 12} more" if len(_tr) > 12 else "")
        + ". Rows from the backup copy are exact; a SKIPPED row is left out "
          "of its panel (the per-iteration recovery trace holds the previous "
          "row's value there).")

missing_html = "".join(f"<li>{m}</li>" for m in MISSING)


# ---- match-criterion fragments (SHOW_MATCH_STATS gate) --------------------
if SHOW_MATCH_STATS:
    _crit_kpi = ("phase-max overlap" if SCI and SCI.get("match_is_overlap")
                 else "2-df proxy")
    KPI_MATCH = f"""  <div><b>{SCI.get("n_match", 0):,}</b><span>matched to an injection ({_crit_kpi})</span></div>
  <div><b>{pct(SCI["completeness"]) if SCI else "&mdash;"}</b><span>completeness ({_crit_kpi})</span></div>
  <div><b>{pct(SCI["purity"]) if SCI else "&mdash;"}</b><span>purity ({_crit_kpi})</span></div>"""
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

# ---- "How Many Are Detectable At All" table (live, this Tobs) --------------
# The block below builds the two columns of the #detect table at this run's
# actual observation time -- until 2026-09-20 the table was HARDCODED to the
# 3-month values (1,001 / 1,103 at SNR>7), so a 1-year page silently claimed
# the 3-month numbers as if they were current.
#   Column 1 (fitted noise): free -- TRU already carries per-source SNR under
#     the run's own fitted PSD + foreground at this Tobs. Just count.
#   Column 2 (injected + FittedHyperbolicTangent): re-run the catalogue
#     optimal-SNR calculation against a fixed reference noise -- the injected
#     instrument (SOMS_INJ, SA_INJ) plus the lisatools
#     FittedHyperbolicTangentGalacticForeground, both functions of Tobs alone.
#     Reuses build_truth.py's opt_snr helper for consistency.
DETECT_CUTS = (5, 7, 10, 15)
_col1 = _col2 = None
_col2_err = None
if TRU is not None:
    try:
        _snr_fit = np.asarray(TRU["snr"], float)
        _col1 = {c: int((_snr_fit > c).sum()) for c in DETECT_CUTS}
    except Exception as _e:
        MISSING.append(f"#detect column 1 (fitted noise) unavailable: "
                       f"{type(_e).__name__}: {_e}")

    # Column 2 -- compute lazily. Skipped when GBGPU can't import or when
    # any prerequisite is missing; the caption below reflects the state.
    try:
        import sys as _sys
        _bt_dir = os.path.dirname(os.path.abspath(__file__))
        if _bt_dir not in _sys.path:
            _sys.path.insert(0, _bt_dir)
        from build_truth import opt_snr as _opt_snr_bt   # local sibling

        from gbgpu.gbgpu import GBGPU as _GBGPU2
        from lisatools import detector as _lm2
        from lisatools.sensitivity import (
            get_sensitivity as _gs2, A2TDISens as _A2T, E2TDISens as _E2T)
        from lisatools.stochastic import (
            FittedHyperbolicTangentGalacticForeground as _FHT)
        from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
            GB_MOJITO_T_REF as _T_REF_BT)

        _lm_inj = _lm2.LISAModel(SOMS_INJ ** 2, SA_INJ ** 2,
                                 _lm2.DefaultOrbits(force_backend="cpu",
                                                    frame="icrs"),
                                 "injected")
        _nk_inj = dict(model=_lm_inj, stochastic_params=(SCI_TOBS,),
                       stochastic_function=_FHT)
        _ng2 = int(2.35e-2 / SCI_DF) + 2
        _fg2 = np.maximum(np.arange(_ng2) * SCI_DF, SCI_DF)
        _sa2 = np.asarray(_gs2(_fg2, sens_fn=_A2T, **_nk_inj), float)
        _se2 = np.asarray(_gs2(_fg2, sens_fn=_E2T, **_nk_inj), float)
        _nw2 = int(TRU["nw"]) if "nw" in getattr(TRU, "files", []) else 128
        _phys2 = np.asarray(TRU["phys"], float)
        _orb2 = globals().get("_L1_ORB_CPU")
        if _orb2 is None:
            _orb2 = _lm2.DefaultOrbits(force_backend="cpu", frame="icrs")
        _gbw2 = _GBGPU2(force_backend="cpu", orbits=_orb2,
                        t0=float(_T_REF_BT))
        _snr_inj = _opt_snr_bt(_phys2, _sa2, _se2, _gbw2, SCI_DF,
                               SCI_TOBS, _nw2, batch=20000)
        _col2 = {c: int((_snr_inj > c).sum()) for c in DETECT_CUTS}
    except Exception as _e:
        _col2_err = f"{type(_e).__name__}: {_e}"
        MISSING.append(f"#detect column 2 (injected + FittedHT) unavailable: "
                       f"{_col2_err}")

def _detect_tbl_row(cut):
    _bold = (cut == 7)
    _tc = "font-weight:bold" if _bold else ""
    _v1 = (f"{_col1[cut]:,}" if _col1 is not None else "&mdash;")
    _v2 = (f"{_col2[cut]:,}" if _col2 is not None else "&mdash;")
    _lb = f"<strong>{cut}</strong>" if _bold else f"{cut}"
    _cs = "padding:3px 18px 3px 0"
    return (f'<tr><td style="{_cs}">{_lb}</td>'
            f'<td style="text-align:right;{_cs};{_tc}">{_v1}</td>'
            f'<td style="text-align:right;{_tc}">{_v2}</td></tr>')

TBL_DETECT_ROWS = "\n".join(_detect_tbl_row(c) for c in DETECT_CUTS)
_col2_note = ("" if _col2 is not None else
              f"<br><em>Column 2 not computed in this snapshot "
              f"({'no TRU npz' if TRU is None else _col2_err}).</em>")

# ---- AT A GLANCE (user request 2026-10-11) -------------------------------
# "We really want the plots and info that are most important/succinct in
# capturing information appearing first (this should be the 'short' html as
# well). Then after grouped by type with drop down options defaulting the more
# detailed information ... to 'collapsed'", and "stack the other sources closer
# to the top". The block is the SHORT page's, rendered by the short page's own
# code (_short.build_glance), so the two pages cannot disagree; everything this
# script draws follows it in collapsed <details> groups. A failure costs the
# block and a Notes line, never the page.
import html as _html_mod  # noqa: E402

from lisatools.globalfit.monitor._short import GLANCE_CSS, build_glance  # noqa: E402

GLANCE_HTML = GLANCE_NAV = GLANCE_NOTES_HTML = ""
try:
    _G = build_glance(RUN_DIR, mojito=MOJITO_INFO_PATH or (
        MOJITO_CAT_DIR if os.path.isdir(os.path.join(MOJITO_CAT_DIR, "catalogues")) else None))
    GLANCE_HTML = _G["html"]
    GLANCE_NAV = "".join(f'<a href="#{_i}">{_t}</a>' for _i, _t in _G["nav"])
    GLANCE_NOTES_HTML = "".join(f"<li>at a glance: {_n}</li>" for _n in _G["notes"])
    print(f"[glance] at-a-glance block built in {_G['elapsed']:.1f} s")
except Exception as _e:  # noqa: BLE001
    GLANCE_HTML = ('<div class="missing">the at-a-glance block could not be built: '
                   f"{type(_e).__name__}: {_html_mod.escape(str(_e))} (the short page "
                   "shares its code; the groups below are unaffected)</div>")
    MISSING.append(f"at-a-glance block failed: {type(_e).__name__}: "
                   f"{_html_mod.escape(str(_e))}")
# a note can quote an exception, and exceptions quote paths like <mojito_data_path>/...:
# escape the angle brackets (no note carries markup of its own; entities pass through)
missing_html = "".join("<li>" + str(m).replace("<", "&lt;").replace(">", "&gt;") + "</li>"
                       for m in MISSING)
N_NOTES = len(MISSING) + GLANCE_NOTES_HTML.count("<li>")
# what the closed sources group holds, so its title says whether it is worth opening
SRC_GROUP_SUB = "per-source chain traces and posteriors: " + " &middot; ".join(
    [f"{SRC_PANEL_META.get(_b, {}).get('n_rendered', 0)} {_c}"
     for _b, _c in (("mbh", "MBHB"), ("emri", "EMRI"), ("sobbh", "SOBHB"))]
    + [f"verification binaries ({VGB_N_DET} of 55 above SNR 7)"])

RATCHET_GROUP = ""
NAV_RATCHET = ""
if "fg_ratchet_timeline" in IMGS:
    # the galfor ratchet ran on this store (no current recipe runs it, 2026-10-03):
    # its own panels, collapsed like every detail group
    NAV_RATCHET = '<span class="ng"><a class="g" href="#grp-ratchet">ratchet</a>'\
        '<a href="#ratchet">timeline</a></span>'
    RATCHET_GROUP = f"""<details class="group" id="grp-ratchet"><summary><span class="gt">Foreground ratchet</span><span class="gs">the gated noise steps and where the leaves went across them (this store ran the galfor ratchet)</span></summary>
<div class="gbody">
<section id="ratchet"><h2>Foreground Ratchet Diagnosis</h2>
<div class="panel">{img("fg_ratchet_timeline", "ratchet timeline (needs GALFOR_RATCHET + GB_SEARCH_LEGS rows)")}
<div class="caption">{cap_fgA}</div></div>
<div class="panel">{img("fg_leaf_delta", "where the leaves went across the ratchet")}
<div class="caption">{cap_fgC}</div></div>
<div class="panel">{img("fg_leaf_delta_zoom", "zoom 2.5-5.5 mHz: where the leaves went")}
<div class="caption">Zoom on 2.5&ndash;5.5 mHz at 0.05 mHz bins (about three GB bands each):
the grey bars are the leaves already there at the reference row, the green bars what was
added or removed since, the amber curve the foreground ratio, and the lower panel the
per-band cold log-likelihood change.</div></div>
</section>
</div>
</details>
"""

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
  /* Missed-detection SNR tier colors (2026-10-05, user ruling): yellow
     for 7-10, orange for 10-20, red for >20. The idea is that a loud
     missed source is a worse story than a quiet one, and the eye should
     see the loud ones FIRST on a crowded panel. */
  --truthorange:#FF8A1A;
  --truthyellow:#F7D33C;
  /* Zoom-canvas latest-iteration overlay ring (2026-10-05, user ruling).
     MAGENTA, held far from --cyan (base cloud), --amber, and the
     three missed-detection tiers so it reads as "where the sampler
     IS RIGHT NOW" without blending into the orange X's that sit
     elsewhere on the plane. */
  --latest-ring:#FF5CEC;
  /* Sub-threshold catalogue sources on the zoom canvas. Light enough to
     read as a population against --bg, dark enough that the red
     detectable crosses stay the thing the eye lands on. */
  --truthgrey:#5A6878;
}}
:root[data-theme="light"] {{
  --bg:#EEF1F5; --panel:#FFFFFF; --line:#D4DBE3; --fg:#25313D; --dim:#5D6B7A;
  --truthred:#E00016;
  --truthorange:#D16500;
  --truthyellow:#B38F00;
  --latest-ring:#C300A8;
  --truthgrey:#9AA7B4;
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
details.group {{ margin-top:16px; border:1px solid var(--line); border-radius:4px;
  background:color-mix(in srgb, var(--panel) 45%, var(--bg)); }}
details.group > summary {{ cursor:pointer; list-style:none; padding:10px 14px; display:flex;
  flex-wrap:wrap; gap:2px 14px; align-items:baseline; }}
details.group > summary::-webkit-details-marker {{ display:none; }}
details.group > summary::before {{ content:"\\25B8"; color:var(--cyan); width:1em; }}
details.group[open] > summary::before {{ content:"\\25BE"; }}
details.group > summary:hover .gt, details.group > summary:focus .gt {{ color:var(--cyan); }}
details.group > summary .gt {{ font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--fg); }}
details.group > summary .gs {{ font-size:12px; color:var(--dim); }}
details.group > .gbody {{ padding:0 14px 14px; }}
details.group > .gbody > section:first-child {{ margin-top:4px; }}
#glance > h2.glance-title {{ color:var(--cyan); }}
#glance > h2.glance-title span {{ color:var(--dim); text-transform:none; letter-spacing:0; font-weight:normal; }}
nav .ng {{ display:inline-flex; flex-wrap:wrap; gap:4px; align-items:center; padding-right:8px;
  border-right:1px solid var(--line); }}
nav .ng a.g {{ color:var(--fg); }}
nav .ng a:not(.g) {{ font-size:11px; padding:1px 6px; }}
nav button {{ font-size:11px; padding:1px 8px; }}
{GLANCE_CSS}
</style>
<header>
  <h1>LISA Global Fit &middot; {RUN_LABEL} Status</h1>
  <span class="stamp">{os.path.basename(RUN_DIR)} &middot; {datetime.now():%Y-%m-%d}</span>
  <span>{chips}</span>
</header>
<nav>
  <span class="ng"><a class="g" href="#glance">at a glance</a>{GLANCE_NAV}</span>
  <span class="ng"><a class="g" href="#grp-sources">other sources</a><a href="#mbh">MBHB</a><a href="#emri">EMRI</a><a href="#sobbh">SOBHB</a><a href="#vgb">VGB</a></span>
  <span class="ng"><a class="g" href="#grp-gbrec">GB recovery</a><a href="#recovery">recovery</a><a href="#population">population</a>{NAV_PARAMS}</span>
  <span class="ng"><a class="g" href="#grp-noise">noise &amp; residual</a><a href="#noise">noise</a><a href="#resid">residual</a><a href="#foreground">foreground</a></span>
  <span class="ng"><a class="g" href="#grp-gbsearch">GB search</a><a href="#trackers">trackers</a><a href="#search">search &amp; cap cells</a>{NAV_GATES}<a href="#fstat">f-stat</a></span>
  <span class="ng"><a class="g" href="#grp-method">detectability &amp; method</a><a href="#status">summary</a><a href="#detect">detectability</a></span>
  <span class="ng"><a class="g" href="#grp-mech">run mechanics</a></span>
  {NAV_RATCHET}
  <span class="ng"><a class="g" href="#grp-notes">notes ({N_NOTES})</a></span>
  <button type="button" id="grp_open">expand all</button><button type="button" id="grp_close">collapse all</button>
</nav>
<main>
{COMMENTARY}
<div id="glance">
<h2 class="glance-title">At a glance <span>&middot; the same block is the short page,
<code>{os.path.basename(os.path.normpath(RUN_DIR))}_monitor_short.html</code>; the detail
groups below start collapsed &mdash; click a heading to open one</span></h2>
{GLANCE_HTML}
</div>

<details class="group" id="grp-sources"><summary><span class="gt">Other sources</span><span class="gs">{SRC_GROUP_SUB}</span></summary>
<div class="gbody">
<section id="mbh"><h2>Massive Black-Hole Binaries</h2>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="mbh1_sel"></select></label>
  <span class="caption" style="align-self:center">cold-chain trace (top) and
  posterior over full_pe rows (bottom) per injected MBHB, one source at a
  time; dotted red is the catalogue value</span>
</div>
<img id="mbh1_img" alt="MBHB chain + posterior">
<div class="caption" id="mbh1_cap"></div>
<div class="caption">{MBH_PANEL_COUNT} Key parameters shown: total mass
M<sub>tot</sub>, mass ratio Q = m<sub>1</sub>/m<sub>2</sub> &ge; 1, aligned
spins &chi;<sub>1</sub> / &chi;<sub>2</sub>, coalescence time t<sub>c</sub>,
distance d<sub>L</sub>, sky (&alpha;, sin&delta;). The right-hand diagnostic
pane is the per-walker &lang;d|h&rang; / &lang;h|h&rang; from
<code>sub_backend/mbh</code>; the summary table at the bottom quotes median,
truth, and (median &minus; truth) / &sigma; per parameter over the
cold-walker-pooled full_pe window.</div>
</div>
</section>

<section id="emri"><h2>Extreme Mass-Ratio Inspirals</h2>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="emri1_sel"></select></label>
  <span class="caption" style="align-self:center">cold-chain trace (top) and
  posterior over full_pe rows (bottom) per injected EMRI; dotted red is the
  catalogue value</span>
</div>
<img id="emri1_img" alt="EMRI chain + posterior">
<div class="caption" id="emri1_cap"></div>
<div class="caption">{EMRI_PANEL_COUNT} Key parameters shown: primary mass
log&nbsp;m<sub>1</sub>, spin a, semi-latus rectum p<sub>0</sub>, eccentricity
e<sub>0</sub>, distance d<sub>L</sub>. The sky is kept in each sampling
frame &mdash; the EMRI stock basis uses ecliptic colatitude q<sub>S</sub>
and &phi;<sub>S</sub>, so the truth sky line comes from &pi;/2 &minus; Dec
and RA.</div>
</div>
</section>

<section id="sobbh"><h2>Stellar-Origin Black-Hole Binaries</h2>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="sobbh1_sel"></select></label>
  <span class="caption" style="align-self:center">cold-chain trace (top) and
  posterior over full_pe rows (bottom) per injected SOBHB; dotted red is the
  catalogue value</span>
</div>
<img id="sobbh1_img" alt="SOBHB chain + posterior">
<div class="caption" id="sobbh1_cap"></div>
<div class="caption">{SOBBH_PANEL_COUNT} Key parameters shown: component
log-masses log&nbsp;m<sub>1</sub>, log&nbsp;m<sub>2</sub>, starting GW
frequency f<sub>low</sub>, distance d<sub>L</sub>, and sky (&phi;<sub>S</sub>,
cos&nbsp;q<sub>S</sub>). SOBHB uses eight tempering rungs per leaf
(<code>SOBBH_NTEMPS</code>); only the cold rung is shown here.</div>
</div>
</section>

<section id="vgb"><h2>Verification Binaries</h2>
<div class="panel">{img("f9_vgb", "detectable verification binaries")}
<div class="caption">The {VGB_N_DET} of 55 catalogue verification binaries that clear
SNR 7 at three months, with their distance posteriors against the catalogue value.
The other {55 - VGB_N_DET} are prior-dominated. Full-55 SNR-vs-frequency, all-55
distance posteriors, the SNR-per-iteration trend, and the interactive zoom cloud were
removed for the 16 MB size budget; the median VGB SNR is {VGB_SNR_MED:.1f} and the
{55 - VGB_N_DET} prior-dominated leaves are an absence of signal, not a fit failure.</div></div>
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
</div>
</details>

<details class="group" id="grp-gbrec"><summary><span class="gt">Galactic binaries: recovery</span><span class="gs">completeness and purity, the recovered population, the zoomable plane, parameter recovery</span></summary>
<div class="gbody">
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

<div class="caption" style="margin-top:18px"><strong>Zoom in.</strong> The static panels above are the whole band at once; these two are pannable and zoomable, which is the only way to read an individual galactic binary against its injected counterpart. The scatter pools the last {EXPL_ITS} stored iterations of the cold chain; the corner posteriors under it pool {gb1_meta.get("corner_its", 0)}. Every pooled iteration is masked by its OWN <code>inds</code> row &mdash; the branch is trans-dimensional, so a leaf alive now was not necessarily alive four iterations ago.</div>
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
<div class="caption">Zoomable version of the amplitude-frequency plane, with
four classification layers on top of the posterior cloud:
<span style="color:var(--cyan)">cyan</span> = pooled posterior samples of
the current model,
<span style="color:var(--latest-ring)">faint magenta rings</span> = samples from
the MOST RECENT stored iteration (visible on close inspection; deliberately
subdued so the recovery markers stay the thing the eye lands on),
<span style="color:var(--dim)">grey X</span> = undetectable catalogue rows,
missed-detection X&apos;s tiered by injected SNR
(<span style="color:var(--truthyellow)">yellow</span> 7&ndash;10,
<span style="color:var(--truthorange)">orange</span> 10&ndash;20,
<span style="color:var(--truthred)">red</span> &gt; 20) &mdash;
loud misses pop before quiet ones,
<span style="color:var(--green)">closed green circle</span> = recovered and
matched,
<span style="color:var(--violet)">open violet circle</span> = recovered with
no matching injection. Drag to pan and use the wheel to zoom, or set the
view numerically with the centre and width/height controls above &mdash;
those hold the window size fixed and slide it across the band, which is the
steadier way to walk through frequency.
<strong>The cyan posterior cloud is pooled over the last {EXPL_ITS} stored iterations</strong>
&times; {nwalk} cold walkers ({EXPL_RAW:,} alive-leaf rows{EXPL_DEC_NOTE}),
with each iteration&rsquo;s own alive mask applied, so a single source draws a
cloud whose width is the sampler&rsquo;s spread rather than one snapshot of
where the walkers happened to sit. The catalogue overlay is decimated per
frequency window on the undetectable rows; every detectable-not-recovered
row is drawn in full. {MATCH_CRIT_HTML}</div>
</div>

<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="gb1_sel"></select></label>
  <span class="caption" style="align-self:center">posteriors of the highest-frequency
  recovered galactic binaries</span>
</div>
<img id="gb1_img" alt="galactic binary corner posterior">
<div class="caption" id="gb1_cap"></div>
<div class="caption">Sources are clustered out of the last stored iteration by
frequency ({CLUSTER_BINS:.0f} bins) and counted only if they appear in at least three
of the {nwalk} cold walkers; {gb1_meta.get("n_solid", 0)} of
{gb1_meta.get("n_clusters", 0)} clusters clear that bar. Catalogue values are shown
where a source lies within {MATCH_BINS:.0f} bins. The corner itself is built from a
wider window than that clustering step &mdash; up to the last
{gb1_meta.get("corner_its", 0)} stored iterations, pooled by frequency association
because the GB branch is trans-dimensional and a leaf index does not track one
source across iterations. Contours are 1, 2 and 3 sigma; below 200 pooled rows the
kernel is widened deliberately, so read these as where the walkers currently sit
rather than as credible intervals.</div>
</div>

</section>

{PARAMS_SECTION}
</div>
</details>

<details class="group" id="grp-noise"><summary><span class="gt">Noise, foreground &amp; residual</span><span class="gs">noise posteriors and traces, the residual spectrum, data / template / residual, the whitened residual per band</span></summary>
<div class="gbody">
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
<div class="panel">{img("psd_evolution_zoom", PSD_ZOOM_ALT)}
<div class="caption">Same as above, zoomed to the last {PSD_ZOOM_N} stored iterations.</div></div>
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

<section id="foreground"><h2>Foreground &amp; Whitened Residual</h2>
<div class="panel">{img("fg_whitened_bands", "whitened residual per GB band")}
<div class="caption">{cap_fgB}</div></div>
<div class="panel">{img("fg_whitened_zoom", "zoom 2.5-5.5 mHz: whitened residual per GB band")}
<div class="caption">Zoom of the panel above on 2.5&ndash;5.5 mHz with a linear frequency
axis. Points are every GB band for the max-lnL (cyan) and min-lnL (amber) cold walkers
of the last stored row, each against its own fitted noise; the running mean spans 5 bands
(~85 &micro;Hz) instead of 15.{FG_RESID_SRC}</div></div>
<div class="panel">{img("fg_resid_psd_zoom", "zoom 2.5-5.5 mHz: residual spectrum against the noise curves")}
<div class="caption">The same residual unwhitened, for the max-lnL (cyan) and min-lnL
(amber) cold walkers of the last stored row: each walker&rsquo;s residual PSD per GB band
(solid) against its own fitted noise (instrument + galfor, dashed) and its fitted
instrument alone (dotted), with the injected instrument + FittedHT estimate and stored row
0&rsquo;s noise (violet) &mdash; under a pinned launch that is the start vector the search
ran against until the first release.
Where the residual sits above the fitted curve the model is under-estimating the power
there; where the fitted curve sits above row 0 the release raised the foreground.{FG_RESID_SRC}</div></div>
<div class="panel">{img("fg_missed_vs_row", "ratchet scorecard: unresolved detectable catalogue sources per band vs stored row")}
<div class="caption">{cap_fgD}</div></div>
<div class="panel">{img("fg_truth_decomp", "what the fitted foreground is made of: unresolved catalogue power vs the fit")}
<div class="caption">{cap_fgE}</div></div>
</section>

</div>
</details>

<details class="group" id="grp-gbsearch"><summary><span class="gt">Galactic binaries: search</span><span class="gs">leaf count and lnL in detail, lnL per sub-band, birth fates, efficiency, cap cells, the search gates, the F-statistic</span></summary>
<div class="gbody">
<section id="trackers"><h2>Model Size &amp; Likelihood</h2>
<!-- The detailed trackers (leaf count with the cap-cell map, lnL with the walker
     spread, lnL per sub-band). The at-a-glance block carries the short versions
     (user request 2026-10-11: succinct first, detail collapsed). -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">
<div class="panel">{img("gb_leaves")}
<div class="caption">Left: galactic-binary leaf count per cold walker across the stored
iterations, each walker its own colour (the same on every per-walker panel) and the thick
light line their mean, against the detectable-injection target; with a truth set the walkers
count the analysed band and the dashed line is the all-frequency mean. Right: the enforced per-cell leaf
cap over time. Rows marked in red have had their births shut off by the barren-band rule
&mdash; deaths and in-model moves continue there.</div></div>
<div class="panel">{img("ll")}
<div class="caption">Cold-chain total log-likelihood of each of the {nwalk} walkers in
its own colour (the same as on the leaf count) with their mean as the thick light line, and
the max-minus-min spread. At equilibrium the spread sits at a few units.</div></div>
</div>
<div class="panel">{img("gb_band_ll", "GB log-likelihood per sub-band")}
<div class="caption">{GB_BAND_LL_CAP}</div></div>
</section>

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
<!-- gb_leaves + ll grid MOVED to the top of the page (user request
     2026-08-22, twice: on a 12 MB page these two headline trackers were
     effectively invisible this deep in the scroll). See TRACKERS_TOP. -->
<div class="panel">{img("timing_moves", "per-move throughput")}
<div class="caption">Proposal throughput and wall time per propose, per move, against
elapsed run time.</div></div>
<div class="panel">{img("search_efficiency", "search efficiency")}
<div class="caption">What the recipe COSTS against what it BUYS. Left: where an
iteration's wall time goes, per move. Middle: seconds per NEW cold leaf per walker,
annotated with the leaf gain &mdash; this is the number that degrades as the search
saturates, and it moves even when the per-iteration cost is flat. Right: each RJ
move's wall time (cyan, left axis) beside the cold RJ accepts it actually landed
(green, right axis) on the latest iteration; a tall cyan bar over a zero green bar
is a move paying full price for nothing.</div></div>
<div class="panel">{SEARCH_SPLIT_HTML}
<div class="caption">Inside each move: how much of its wall is the RJ step itself
versus the in-model convergence polish that follows it. The polish dominating every
row is the expected shape once the convergence rule is armed &mdash; it is what makes
"the RJ schedule got expensive" the wrong reading of a slow iteration.
&#9888; The wall column is PER ITERATION, the split columns are PER PROPOSE, taken
from the most recent <code>[GB_TIMING &lt;move&gt; head r0]</code> record. They agree
for a one-propose move; <code>rj_replace</code> runs TWO passes per iteration, so its
split row is one pass and its true polish share is about double what the row shows
(98%, not 49%).</div></div>
<div class="panel">{img("gb_cap_cells", "cap-cell occupancy")}
<div class="caption">{CAP_TXT} Left is the direct test of whether the cap is being
respected: bars at or above the cap are amber, and a bar past it would mean sources are
stacking. Middle is the race that matters &mdash; occupied cells must stay ahead of cells
at their cap, or the model is queuing against the ceiling rather than filling. Right
explains why the occupied fraction looks small: the cells tile the whole band uniformly
while the sources are concentrated in the galaxy, so most cells are empty because there is
nothing in them yet.</div></div>{PER_WALKER_PANELS}
<div class="panel">{img("gb_cap_divisor", "cap-divisor study")}
<div class="caption">What the cap grid can represent, independent of how far this run has
got: summing the detectable sources a cell cannot admit gives the ceiling the sampler can
never beat. A finer grid beats simply raising the cap, because a higher cap permits several
sources inside one cell, which is the stacking the rule exists to stop.</div></div>
{CENSUS_PANEL}
<div class="panel">{img("fg_rj_per_band", "RJ traffic per leg and per 0.5 mHz")}
<div class="caption">Accepted RJ moves on the cold rung per 0.5 mHz for each stored leg the
counters cover, and the acceptance fraction. The in_model_fstat legs carry the F-stat
births, the in_model_removal legs the prior removals. The counters are cumulative per band
and are differenced between consecutive stored rows; an extract keeps them only in its keep
window, so a short snapshot shows the last few legs. Legs covered:
{", ".join(f"row {i} ({nm}): {int(a):,} accepted of {int(p):,}" for i, nm, p, a in RJB.get("legs", [])) or "none"}.</div></div>
</section>

{GATES_HTML}
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
</div>
</details>

<details class="group" id="grp-method"><summary><span class="gt">Detectability &amp; method</span><span class="gs">the frozen denominator, the run summary, how many sources are detectable at all, how the numbers are produced</span></summary>
<div class="gbody">
<section id="status"><h2>Run Summary &amp; Denominator</h2>
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
signal-to-noise above 7 over {BAND_TXT}, the full band the sampler analyses,
evaluated under this run&rsquo;s own fitted noise. A model source counts as a
recovery when it lies within
{TOL_BINS:.0f} frequency bins ({TOL_BINS / SCI_TOBS * 1e6:.3f} &micro;Hz) of one,
one-to-one. Those windows cover {pct(SCI["chance"], 1) if SCI else "2.2%"} of the
band, so that is the rate at which an arbitrary source would match by accident.
Below roughly 3 mHz that threshold is an SNR statement and not a resolvability
one: the sensitivity it is measured against already carries the fitted galactic
foreground, but the catalogue puts many sources in every frequency bin down
there, so a binary can clear signal-to-noise 7 and still be inseparable from
its neighbours.</p>
{ARM_TABLE}
</section>

<section id="detect"><h2>How Many Are Detectable At All</h2>
<div class="panel">
<div class="caption" style="margin:0 0 10px 0">Optimal SNR of the injected
catalogue over {BAND_TXT} at this run&rsquo;s observation time
({SCI_TOBS / 86400.0:.4g} d), under two noise models:
<em>col 1</em> = the run&rsquo;s own fitted instrument + foreground at the
truth-build iteration; <em>col 2</em> = the injected instrument
(S<sub>oms</sub>, S<sub>a</sub>) plus the lisatools
<code>FittedHyperbolicTangentGalacticForeground</code> at this T<sub>obs</sub>.
Both columns re-compute at every snapshot render &mdash; they do not carry
over from previous pages.{_col2_note}</div>
<table style="border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums">
<tr style="border-bottom:1px solid var(--line)">
  <th style="text-align:left;padding:4px 18px 4px 0">SNR &gt;</th>
  <th style="text-align:right;padding:4px 18px 4px 0">fitted noise (col 1)</th>
  <th style="text-align:right;padding:4px 0">injected + FittedHT (col 2)</th></tr>
{TBL_DETECT_ROWS}
</table>
<div class="caption" style="margin-top:12px">Detectability dies below about 1 mHz,
where the foreground swamps everything. The two models disagree by ~10% and, more
usefully, in opposite directions either side of the galactic peak &mdash; treat
column 2 as a reference point, not truth.
<br><br>
Detectability is also a moving target: the same calculation gives
<strong>{SCI.get("ndet", 0)}</strong> over {BAND_TXT} at this run&rsquo;s
late-iteration noise, and it moved as the foreground estimate dropped and sources
left the residual. That is exactly why the denominator on this page is frozen at
one iteration and stated.</div>
</div>
</section>

<section id="method"><h2>Method</h2>
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
</section>

</div>
</details>

<details class="group" id="grp-mech"><summary><span class="gt">Run mechanics</span><span class="gs">tempering swaps, the RJ move breakdown, device memory, GPU telemetry</span></summary>
<div class="gbody">
<section id="mechanics"><h2>Run Mechanics</h2>
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
<div class="panel">{img("mem_telemetry", "device memory")}
<div class="caption">Device-wide memory from the in-run telemetry, with breaks across
restart gaps so an attempt boundary does not draw a false ramp.</div></div>
<div class="panel">{img("gpu_util", "gpu telemetry")}
<div class="caption">Utilisation and memory sampled by nvidia-smi; dotted traces are
earlier jobs.</div></div>
</section>

</div>
</details>

{RATCHET_GROUP}
<details class="group" id="grp-notes"><summary><span class="gt">Notes</span>
<span class="gs">{N_NOTES} item(s): what this snapshot could not reproduce, and why</span></summary>
<div class="gbody">
<ul>{missing_html}{GLANCE_NOTES_HTML}</ul>
</div>
</details>
</main>

<script>
const DATA = {EXPL_JSON};
const VPOST = {VGB_POST_JSON};
const VGBC = {VGB_CORNER_JSON};
const GB1 = {GB1_JSON};
const MBHP = {MBH_PANEL_JSON};
const EMRIP = {EMRI_PANEL_JSON};
const SOBBHP = {SOBBH_PANEL_JSON};
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
    // the GB panel carries a catalogue-match note (red when the source has
    // no injected counterpart); the VGB panel leaves it empty
    const note = S.note
      ? ` &middot; <span style="color:var(${{S.bad ? "--red" : "--fg"}})">${{S.note}}</span>`
      : "";
    cap.innerHTML = S.sub + note;
  }}
  sel.onchange = show;
  show();
}}
cornerPanel("vgb1", VGBC);
cornerPanel("gb1", GB1);
cornerPanel("mbh1", MBHP);
cornerPanel("emri1", EMRIP);
cornerPanel("sobbh1", SOBBHP);
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
// vgbpost dist-f0 posterior cloud REMOVED 2026-09-05 (16 MB size budget); its
// canvas markup + reset/pick controls were removed, so the whole handler IIFE
// is deleted here to avoid dereferencing a null #vgbpost and killing later blocks.
(() => {{
  const cv = document.getElementById("expl"), cap = document.getElementById("expl_cap");
  const css = getComputedStyle(document.documentElement);
  const C = n => css.getPropertyValue(n).trim();
  const hasGB = DATA.gb.length > 0;
  // points: [x, y(=1/d), tag]
  const pts = hasGB ? DATA.gb : DATA.vgb;
  const xlab = hasGB ? "f0 [mHz]" : (DATA.vgb_axis || "VGB leaf index");
  // FOUR-CLASS OVERLAY (2026-09-20). truth_grey = undetectable catalogue,
  // truth_red = detectable catalogue that isn't recovered under the current
  // match criterion, sources = recovered sources (last iter) with a
  // matched/unmatched flag. Legacy DATA.truth is retained for the VGB
  // fallback path and any external consumers of this JSON blob.
  const T_GREY = DATA.truth_grey || DATA.truth || [];
  const T_RED  = DATA.truth_red  || [];
  const SRC    = DATA.sources    || [];
  const N_OVERLAY = T_GREY.length + T_RED.length + SRC.length;
  const MATCH_NOTE = DATA.match_note || "";
  // DEFAULT ON. The classification overlay IS the point of this panel --
  // catalogue completeness cannot be read off the recovered cloud alone.
  let showT = N_OVERLAY > 0;
  const baseCap = (hasGB
    ? `GB samples: ${{DATA.gb.length}} alive-source rows pooled over the last ${{DATA.gb_its}} stored iterations x all cold walkers${{DATA.gb_stride > 1 ? ` (1-in-${{DATA.gb_stride}} of ${{DATA.gb_raw}} for the OLDER pool; the ${{DATA.gb_n_latest || 0}} latest-iteration rows are never decimated)` : ""}}; y = log10 amplitude from (dist, f0, Mc). Posterior cloud = cyan, with faint magenta rings on the ${{DATA.gb_n_latest || 0}} samples from the MOST RECENT iteration (${{DATA.gb_latest_iter >= 0 ? "iter " + DATA.gb_latest_iter : "n/a"}}); catalogue: grey X undetectable, missed-detection X tiered by injected SNR (yellow 7-10, orange 10-20, red >20); recovered source markers: closed green circle (matched) or open violet circle (not matched). ${{MATCH_NOTE}}`
    : `No GB sources alive yet - showing the 55 VGBs (${{DATA.vgb_its}} stored iterations x ${{DATA.nwalk}} walker samples each) as 1/dist vs leaf index. GB samples take over automatically once births land.`);
  const setCap = () => {{
    cap.textContent = baseCap + (N_OVERLAY
      ? (showT ? " " + (DATA.truth_cap || "") : ` ${{N_OVERLAY.toLocaleString()}} classification points available - press "show catalogue & recovered".`)
      : " No catalogue/recovered overlay in this snapshot.");
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
    // LAYERING (2026-09-20, updated 2026-10-05). Draw order, bottom to top:
    //   1. grey X's for undetectable catalogue rows            (--dim)
    //   2. cyan posterior cloud                                (--cyan)
    //   3. FAINT magenta rings on the most recent iteration    (--latest-ring, alpha 0.35)
    //   4. SNR-tiered X's for detectable-not-recovered         (--truthyellow/orange/red)
    //   5. green filled / violet open circles for recovered sources
    // Order (3) sits above the cyan cloud so the latest-iter rows can be
    // picked out among older samples, but UNDER (4) and (5) so the
    // recovery markers stay the thing the eye lands on.
    // Steps 1, 3, 4 are gated by `showT`. Step 2 always draws (that's the
    // "posterior of the current model" and reads even without an overlay).
    // The cloud sits between the two truth layers so the red misses stay
    // legible over crowded amber patches, but the grey undetectable rows
    // still show up under the cloud where the model has no sources.
    if (showT) {{
      // Layer 1: grey X's for undetectable catalogue rows.
      g.strokeStyle = C("--dim"); g.globalAlpha = 0.55;
      g.lineWidth = 1.5; g.lineCap = "round";
      const r = 3.4;
      g.beginPath();
      for (const p of T_GREY) {{
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
    // AMBER for the GB posterior cloud (2026-09-20): green now belongs to
    // the "recovered + matched" source markers layered on top. VGB fallback
    // keeps violet.
    // Posterior sample cloud: cyan (2026-10-05, user ruling) -- the earlier
    // amber clashed with the SNR-tiered yellow / orange missed-detection
    // X's; cyan sits far from every X tier and keeps the cloud readable
    // on top of the grey undetectable layer.
    g.fillStyle = hasGB ? C("--cyan") : C("--violet");
    g.beginPath();
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.moveTo(x + 2.2, y); g.arc(x, y, 2.2, 0, 6.29);
    }}
    g.fill();
    g.globalAlpha = 1;
    // LATEST-ITER RING (2026-10-05, user ruling): a FAINT magenta ring
    // sits just over the cyan cloud so the eye can tell where the
    // sampler is right now versus where it has been.  Deliberately
    // UNDER the X markers and the recovered-source circles so the
    // recovery markers stay prominent -- the earlier top-layer
    // placement at full alpha drowned the green dots and read as the
    // headline, which it is not.  Alpha 0.35 + 0.9 px line + 2.6 px
    // radius: visible on close inspection, invisible from across the
    // panel.
    if (hasGB) {{
      g.strokeStyle = C("--latest-ring");
      g.lineWidth = 0.9; g.lineCap = "round";
      g.globalAlpha = 0.35;
      const rl = 2.6;
      g.beginPath();
      for (const p of pts) {{
        if (p.length < 4 || p[3] !== 1) continue;
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x + rl, y); g.arc(x, y, rl, 0, 6.29);
      }}
      g.stroke();
      g.globalAlpha = 1;
    }}
    if (showT) {{
      // Missed-detection X's, tiered by SNR (2026-10-05): yellow 7-10,
      // orange 10-20, red >20. Three passes so each tier's strokeStyle
      // is set once; loud tier last so it overdraws nothing it needs to
      // show. Legacy T_RED rows have no tier byte -- treated as >20 so
      // an old blob still reads as the original "all red" page.
      const TIER_COLOR = ["--truthyellow", "--truthorange", "--truthred"];
      g.globalAlpha = 0.95; g.lineWidth = 1.6; g.lineCap = "round";
      const rr = 3.6;
      for (let tier = 0; tier < 3; tier++) {{
        g.strokeStyle = C(TIER_COLOR[tier]);
        g.beginPath();
        for (const p of T_RED) {{
          const _t = (p.length > 2 ? p[2] : 2);
          if (_t !== tier) continue;
          const x = sx(p[0]), y = sy(p[1]);
          if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
          g.moveTo(x - rr, y - rr); g.lineTo(x + rr, y + rr);
          g.moveTo(x - rr, y + rr); g.lineTo(x + rr, y - rr);
        }}
        g.stroke();
      }}
      // recovered-source markers: filled green (matched) then open violet
      // (unmatched). Two passes so fill / stroke styles need not toggle
      // inside the loop.
      const rs = 3.2;
      g.globalAlpha = 0.95;
      g.fillStyle = C("--green");
      g.beginPath();
      for (const p of SRC) {{
        if (p[2] !== 1) continue;
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x + rs, y); g.arc(x, y, rs, 0, 6.29);
      }}
      g.fill();
      g.strokeStyle = C("--violet");
      g.lineWidth = 1.4;
      g.beginPath();
      for (const p of SRC) {{
        if (p[2] !== 0) continue;
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x + rs, y); g.arc(x, y, rs, 0, 6.29);
      }}
      g.stroke();
      g.globalAlpha = 1;
    }}
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
  if (!N_OVERLAY) bt.disabled = true;
  // reflect the ON default in the control the moment the page loads
  bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
  bt.textContent = showT ? "hide catalogue & recovered" : "show catalogue & recovered";
  bt.onclick = () => {{
    showT = !showT;
    bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
    bt.textContent = showT ? "hide catalogue & recovered" : "show catalogue & recovered";
    setCap(); draw();
  }};
  setCap();
  document.getElementById("btn_top3").onclick = () => {{
    const srt = [...pts].sort((a, b) => b[0] - a[0]);
    const top = srt.slice(0, Math.min(3 * DATA.nwalk, srt.length));
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
// ---- the collapsed detail groups (2026-10-11) ------------------------------
// Every group starts closed. A link to anything inside one opens it (and any
// group around it) before the browser scrolls; expand / collapse all in the nav;
// each viewer's open groups are remembered across reloads (a refreshed snapshot
// page reopens where the reader left it). Storage can be absent: then nothing
// is remembered, and the page works the same.
(() => {{
  const groups = [...document.querySelectorAll("details.group")];
  const KEY = "gfmon-open:";
  const remember = (id, open) => {{
    try {{ if (open) localStorage.setItem(KEY + id, "1"); else localStorage.removeItem(KEY + id); }}
    catch (e) {{}}
  }};
  groups.forEach(d => {{
    try {{ if (localStorage.getItem(KEY + d.id) === "1") d.open = true; }} catch (e) {{}}
    d.addEventListener("toggle", () => remember(d.id, d.open));
  }});
  const openFor = id => {{
    const el = id ? document.getElementById(id) : null;
    if (!el) return null;
    for (let d = el.closest("details"); d; d = d.parentElement ? d.parentElement.closest("details") : null)
      d.open = true;
    return el;
  }};
  document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener("click", () =>
    openFor(decodeURIComponent(a.getAttribute("href").slice(1)))));
  window.addEventListener("hashchange", () => {{
    const el = openFor(decodeURIComponent(location.hash.slice(1)));
    if (el) el.scrollIntoView();
  }});
  if (location.hash) {{
    const el = openFor(decodeURIComponent(location.hash.slice(1)));
    if (el) requestAnimationFrame(() => el.scrollIntoView());
  }}
  const setAll = open => groups.forEach(d => {{ d.open = open; }});
  const bo = document.getElementById("grp_open"), bc = document.getElementById("grp_close");
  if (bo) bo.onclick = () => setAll(true);
  if (bc) bc.onclick = () => setAll(false);
}})();
</script>
"""
open(OUT, "w").write(html)
print(f"wrote {OUT}: {len(html)//1024} KB ({len(html)/1024**2:.2f} MB), "
      f"{len(IMGS)} plots + {len(VGB_CORNER['src'])} VGB corners, "
      f"missing={len(MISSING)}")
