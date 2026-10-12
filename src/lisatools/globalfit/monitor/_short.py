"""The SHORT monitor page, which is also the full page's "at a glance" block:
the topline and the handful of panels that capture the run, built in well
under a minute.

USER REQUEST 2026-10-03: "let's make a short version of the html as well.
That picks out the most important plots and topline info. phase maximized
overlap. current psd measurements. nleaves over time, likelihood over time.
Add what you think should be there but we want to keep it contained and easy
to generate."

USER REQUEST 2026-10-11: "We really want the plots and info that are most
important/succinct in capturing information appearing first (this should be
the 'short' html as well). Then after grouped by type with drop down options
defaulting the more detailed information ... to 'collapsed'." and "stack the
other sources closer to the top". So :func:`build_glance` renders the block
ONCE and both pages carry it: this module wraps it in a page of its own, and
``_generator.py`` puts the same block at the top of the full page, above its
collapsed detail groups.

    python -m lisatools.globalfit.monitor --short-page RUN_DIR [OUT.html]
    python -m lisatools.globalfit.monitor.from_tar --short-page SNAP.tar.gz
    GF_MONITOR_AFTER_SAVE=1 GF_MONITOR_PAGE_SHORT=1      (in-run, saver rank)

The page lands BESIDE the run folder as ``RUN_DIR_monitor_short.html`` (the
run directory's path plus ``_monitor_short.html``) -- a SEPARATE file. The
full page keeps its own name (``..._monitor.html``) and this module never
writes it.

CONTENT, in this order
----------------------
1. TOPLINE. Store, last saved row, stage (step k of N, rows in it, how long
   ago it started), the leg the row was saved after and the next one, the
   pace (log age, last save, last full cycle), the leg in flight against its
   last wall time, the stage-end clock (the per-(walker, band) RJ valve's
   active pairs and their trend), the empty-band valve, the newest F-stat
   epoch and the peak trend, head-node GPU utilisation and memory (job peak
   too) from the newest ``gpu_util_JOB.csv`` (the head node only -- a second
   node is not sampled), cold lnL max / mean / min at the last row and the
   change against the row before, cold leaf counts per branch, the VGBs
   above SNR 7, the overlap headline and the instrument-noise bias.
2. Cold log-likelihood against stored row: min / mean / max over walkers, the
   whole run and the recent window, with the recipe stage boundaries.
3. GB leaf count against stored row, same layout, against the detectable
   target when a truth set is present.
4. The other sources (MBHB, EMRI, SOBHB): every source's matched SNR per cold
   walker at the last row, a walker sitting on a different mode named, and
   each key parameter's (median - catalogue truth) / sigma over the full
   page's source-panel pool window.
5. The phase-maximised, noise-weighted overlap of every proxy-matched
   (recovered, injected) pair at the last row for the max-lnL walker -- the
   full page's headline match criterion, computed the same way.
6. The CURRENT noise measurement: every cold walker's sensitivity at the last
   row against the injected instrument and the injected instrument + FittedHT
   foreground (and the ratio to the latter), plus the instrument-noise
   posteriors against their injected values.
(The galfor-ratchet rows and table are gone: no current recipe runs the
ratchet, dropped 2026-10-03.)

WHY IT DOES NOT CALL THE FULL PAGE'S PANELS
-------------------------------------------
There are none to call. ``_generator.py`` is a script: it reads ``sys.argv``
and raises ``SystemExit(0)`` when imported, and every panel is a block of
MODULE-LEVEL statements closing over ~200 globals, not a function. Running it
means running all of it -- every per-rank log (about 2 GB on the 6-month
run), the mojito data bricks for the residual spectrum, the VGB catalogue.

The smallest adapter that reuses its code anyway: :func:`_generator_functions`
compiles NAMED top-level ``def`` statements (and named plain-data constants) out
of the generator's syntax tree into a namespace this module supplies. The
generator's own torn-read ladder (``_row`` / ``_backup_group``), pairing
(``_match_pairs``), survival count (``_survival``), zoom window
(``_zoom_its``), sub-backend trim (``_last_written``), stage marker
(``mark_stages``) and the source panels' catalogue reading, truth maps and
pool window (``_SRC_BRANCH_SPECS``, ``_src_*``) run here unchanged, and
nothing else of the script executes. The physics goes through the same
importable routes the full page uses: ``build_truth.l1_orbits`` /
``build_truth.sens_grids`` (byte-for-byte the generator's ``SA_G``/``SE_G``),
GBGPU ``run_wave``, ``make_gb_transform_container``, ``get_sensitivity``. The
one inline block it cannot reach -- the per-pair overlap loop -- is mirrored in
:func:`phase_max_overlaps`, and ``tests/test_monitor_short_page.py`` executes
the generator's own loop against it so the two cannot drift.

WHAT IT DOES NOT READ: the per-rank logs, the mojito data bricks, the F-stat
caches, the VGB catalogue. The head log (``*_artifacts/globalfit_run.log``,
its ``_tail`` copy in a snapshot, else the newest ``slurm_stdout_JOB.log``)
is read from the end in growing windows, exactly as
``scripts/diagnostics/gf_last_iteration.py`` reads it. Its readers are
mirrored here rather than imported because that script lives outside the
package and does not ship in the wheel; the test suite runs both parsers on
one log and requires the same answer.

EVERY SOURCE IS OPTIONAL except the store. A missing head log, GPU CSV, truth
set, mojito tree, catalogue or GBGPU costs its own row or panel and a line
under "Notes", never the page.

Knobs (2026-10-03): ``GF_MONITOR_PAGE_SHORT=1`` makes the in-run hook build
this page INSTEAD of the full one (``hooks.py``); ``GF_MONITOR_MATCH_STATS=0``
drops the overlap section, as it does on the full page;
``GF_MONITOR_MATCH_MM`` sets the overlap threshold for "matched" (default
0.8, as on the full page); ``MOJITO_INFO_PATH`` locates the L1 orbits, the
noise brick and the source catalogues, resolved exactly as for the full page;
``GF_MONITOR_PNG_QUANTIZE=0`` keeps the figures 32-bit (``_png.py``).
"""

from __future__ import annotations

import base64
import glob
import html as _html
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np

from ._png import shrink_png

logger = logging.getLogger(__name__)

__all__ = ["build_short_monitor", "default_short_out_path", "render_short_page",
           "build_glance", "phase_max_overlaps", "GLANCE_CSS"]

#: The generator's own top-level helpers this page runs unchanged.
GEN_HELPERS = ("mark_stages", "_zoom_its", "_last_written", "_backup_group",
               "_row", "_match_pairs", "_survival",
               "_parse_src_ids_from_settings", "_src_cat_to_sampling_mbh",
               "_src_cat_to_sampling_emri", "_src_cat_to_sampling_sobbh",
               "_src_cat_row", "_src_load_catalogue", "_src_pool_window")
#: ... and its top-level plain-data constants (literals, ``dict(...)`` of literals).
GEN_CONSTS = ("_SRC_BRANCH_SPECS",)
#: The fixed-dimension source branches the "Other Sources" table covers.
SRC_BRANCHES = ("mbh", "emri", "sobbh")
#: A cold walker whose matched SNR sits this far off the walker median -- by this
#: fraction AND by this many SNR units, so a quiet source's ordinary scatter is not
#: flagged -- is named in the source table (another mode, or the source lost).
SRC_SNR_OFF = 0.2
SRC_SNR_OFF_ABS = 3.0

#: Stores that are never the live one (same list as gf_last_iteration.py).
SKIP = ("backup", "CORRUPT", "pre_rerung", "pre_repair", "midit", ".stale")
#: The head log is read from its end in windows of this size, doubled until the
#: window holds what the topline shows (gf_last_iteration.py's rule).
TAIL_BYTES = 32 * 1024 * 1024
TAIL_BYTES_MAX = 256 * 1024 * 1024
#: Valve judges / F-stat epochs in the topline's trend lines.
TREND = 8
#: The generator's f0 pairing tolerance, in FD bins, and its overlap waveform length.
TOL_BINS = 2.0
NW_OVERLAP = 1024
#: The generator's fallback GB phase epoch when run_settings.log is absent.
T_REF_DEFAULT = 97729089.327664

# The full page's palette, one meaning per hue (see _generator.py's docstring).
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
CYAN, AMBER, GREEN, RED, VIOLET = "#4FD8EB", "#F5A623", "#58C48A", "#E5484D", "#9B7BFF"

#: The full page's figure style, applied through ``rc_context`` so building
#: this page never changes the CALLER's rcParams (it runs on the saver rank).
#: Set in full, font size included: importing lisatools reaches eryn, which
#: restyles matplotlib as an import side effect (font.size 10 -> 16).
STYLE = {
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110, "text.usetex": False,
}

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def default_short_out_path(run_dir: str) -> str:
    """``RUN_DIR`` + ``_monitor_short.html``: a sibling of the run folder, like
    the full page, under a name of its own so neither overwrites the other."""
    return os.path.abspath(str(run_dir).rstrip("/")) + "_monitor_short.html"


# --------------------------------------------------------------------------
# the adapter onto the generator's own helpers
# --------------------------------------------------------------------------
def _plain_data(node) -> bool:
    """``node`` builds plain data only: literals, containers of them, and
    ``dict(key=...)`` calls -- nothing that can run code when executed."""
    import ast

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_plain_data(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _plain_data(k) for k in node.keys) and \
            all(_plain_data(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _plain_data(node.operand)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "dict" and not node.args):
        return all(k.arg is not None and _plain_data(k.value) for k in node.keywords)
    return False


def _generator_functions(ns: dict, names=GEN_HELPERS, consts=None) -> dict:
    """Compile the named top-level ``def``s of ``_generator.py`` into ``ns``,
    plus the named top-level constants whose value is plain data.

    Only those definitions are executed -- defining a function runs none of
    its body, and a plain-data assignment runs no code -- so nothing of the
    script's module-level flow happens. Each function resolves its globals in
    ``ns``: ``_row`` wants ``g``, ``NIT``, ``RUN_DIR``, ``_BACKUP_G``,
    ``_TORN_ROWS``; ``mark_stages`` wants ``STAGE_BOUNDS``;
    ``_src_load_catalogue`` wants ``MOJITO_CAT_DIR``, ``MISSING`` and
    ``h5py``; the rest want ``np`` / ``re`` at most. Raises if a name is no
    longer a top-level ``def`` (or plain-data constant) there, rather than
    silently running without it.
    """
    import ast

    from . import generator_path

    if consts is None:                    # a caller naming its own helpers names its constants
        consts = GEN_CONSTS if names is GEN_HELPERS else ()
    path = generator_path()
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    defs = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    lits = []
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in consts):
            if not _plain_data(n.value):
                raise RuntimeError(f"_generator.py's {n.targets[0].id} is no longer plain "
                                   "data; the short page will not run it")
            lits.append(n)
    missing = sorted((set(names) - {n.name for n in defs})
                     | (set(consts) - {n.targets[0].id for n in lits}))
    if missing:
        raise RuntimeError(
            f"_generator.py no longer defines {missing} at top level; the "
            "short page reuses them by name")
    ns.setdefault("re", re)
    exec(compile(ast.Module(body=lits + defs, type_ignores=[]), path, "exec"), ns)  # noqa: S102
    return ns


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def find_store(run_dir: str) -> str:
    """The newest ``*.h5`` in ``run_dir`` that is not a backup or quarantined
    copy. A reduced ``*_extract.h5`` counts: a snapshot ships nothing else."""
    cands = [p for p in glob.glob(os.path.join(run_dir, "*.h5"))
             if not any(s in os.path.basename(p) for s in SKIP)]
    if not cands:
        raise FileNotFoundError(f"no run store (*.h5) in {run_dir}")
    return max(cands, key=os.path.getmtime)


def _s(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _attr(a, key, default=None):
    v = a.get(key, default)
    if v is None:
        return None
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:  # noqa: BLE001
        return v


def _recipe_steps(g) -> list:
    """Every recipe step as a dict, in order. Empty without a recipe group."""
    rg = g.get("recipe")
    if rg is None:
        return []
    steps = []
    for name in rg:
        a = rg[name].attrs
        mo = _attr(a, "move_order")
        try:
            mo = list(json.loads(_s(mo))) if mo is not None else None
        except Exception:  # noqa: BLE001
            mo = None
        steps.append(dict(
            name=str(name), order=int(_attr(a, "order num", 0) or 0),
            status=bool(_attr(a, "status", False)),
            start=_attr(a, "start_iteration"),
            completed=_attr(a, "completed_iteration"), move_order=mo))
    steps.sort(key=lambda s: s["order"])
    return steps


def stage_bounds(steps: list) -> list:
    """``[(iteration, label), ...]`` for ``mark_stages``.

    A step's ``start_iteration`` is drawn when the step is done or current,
    its ``completed_iteration`` only when the step is done -- a step reset by
    a stage migration keeps stale stamps from the abandoned trajectory (the
    6-month store carries ``full_pe start_iteration=55`` while it is still in
    gb_search_3), and drawing those would mark a boundary that never happened.
    Boundaries at the same row share one label.
    """
    cur = next((s for s in steps if not s["status"]), None)
    lim = cur["order"] if cur is not None else float("inf")
    at = {}
    for s in steps:
        if s["order"] > lim:
            continue
        if s["status"] and s["completed"] is not None:
            at.setdefault(int(s["completed"]), [[], []])[0].append(s["name"])
        if s["start"] is not None:
            at.setdefault(int(s["start"]), [[], []])[1].append(s["name"])
    out = []
    for it, (done, start) in sorted(at.items()):
        lab = " / ".join(f"{n} done" for n in done)
        if start:
            lab = (lab + " -> " if lab else "-> ") + " / ".join(start)
        out.append((it, lab))
    return out


def _source_rows(g, _row, branch, nit, pool_w):
    """One fixed-dimension branch's last ``pool_w`` rows: cold-chain coordinates
    ``(W, nw, nleaves, ndim)``, alive mask ``(W, nw, nleaves)``, and the last
    row's ``<d|h>`` / ``<h|h>`` ``(nw, nleaves)``; None when unreadable."""
    lo = max(0, nit - pool_w)

    def rd(name, sel):           # _row reads through a torn chunk, not a missing dataset
        return _row(name, sel) if name in g else None

    ch = rd(f"chain/{branch}", (slice(lo, nit), 0, 0))
    ind = rd(f"inds/{branch}", (slice(lo, nit), 0, 0))
    dh = rd(f"sub_backend/{branch}/d_h", (nit - 1,))
    hh = rd(f"sub_backend/{branch}/h_h", (nit - 1,))
    if ch is None or ind is None:
        return None
    return dict(chain=np.asarray(ch, float), inds=np.asarray(ind, bool), lo=lo,
                dh=None if dh is None else np.asarray(dh, float),
                hh=None if hh is None else np.asarray(hh, float))


def read_store(store: str, run_dir: str, ns: dict, notes: list) -> dict:
    """Everything the page needs from the store, as plain arrays.

    Row-count rules are the full page's: the filled extent of ``log_like``,
    lowered to the ``iteration`` attr on a rewound store, then stepped back
    past a torn last row whose ``inds/gb`` is still empty. Every dataset read
    goes through the generator's ``_row``, so a chunk torn by a racing saver
    falls back to the run's backup copy and is named under Notes.
    """
    import h5py

    S = dict(store=store, extract="_extract" in os.path.basename(store))
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        ns.update(g=g, NIT=0, RUN_DIR=run_dir, _BACKUP_G=None, _TORN_ROWS=[],
                  h5py=h5py, os=os, np=np)
        try:
            _row = ns["_row"]
            ll_all = _row("log_like", (slice(None), 0, 0))
            if ll_all is None:
                raise RuntimeError("log_like is unreadable in the store and its backup")
            ll_all = np.asarray(ll_all, float)
            filled = np.where(np.any(ll_all != 0.0, axis=1))[0]
            if not filled.size:
                raise RuntimeError(f"{os.path.basename(store)}: no filled rows yet")
            nit = int(filled.max()) + 1
            it_attr = g.attrs.get("iteration")
            S["it_attr"] = int(it_attr) if it_attr is not None else None
            if it_attr is not None and 0 < int(it_attr) < nit:
                notes.append(f"store rewound: iteration attr {int(it_attr)} &lt; "
                             f"filled rows {nit}; plotting the live {int(it_attr)} rows.")
                nit = int(it_attr)
            ns["NIT"] = nit

            branches = sorted(g["inds"].keys()) if "inds" in g else []
            inds_gb = (_row("inds/gb", (slice(0, nit), 0, 0))
                       if "gb" in branches else None)
            if inds_gb is not None and nit > 0 and not inds_gb[nit - 1].any():
                back = 0
                for j in range(nit - 2, max(nit - 40, -1), -1):
                    if inds_gb[j].any():
                        back = nit - 1 - j
                        break
                if back:
                    notes.append(f"stored row {nit - 1} carries no GB leaves (a torn "
                                 f"save); plotting through row {nit - 1 - back}.")
                    nit -= back
                    ns["NIT"] = nit
                else:
                    notes.append(f"stored row {nit - 1} and the 40 before it carry no "
                                 "GB leaves; check the store before reading the GB panels.")
            S["nit"] = nit
            S["ll"] = ll_all[:nit]
            S["nwalk"] = int(S["ll"].shape[1])

            leaves = {}
            for b in branches:
                ind = ((inds_gb[:nit] if inds_gb is not None else None) if b == "gb"
                       else _row(f"inds/{b}", (slice(0, nit), 0, 0)))
                if ind is not None:
                    leaves[b] = np.asarray(ind).sum(axis=-1).astype(int)
            S["leaves"] = leaves

            S["saved_after"] = []
            if "saved_after" in g:
                sa = _row("saved_after", (slice(0, nit),))
                if sa is not None:
                    S["saved_after"] = [_s(x) for x in sa]

            S["steps"] = _recipe_steps(g)

            # noise chains: the sub-backend's cold rung, trimmed to the last
            # row every noise branch has actually written (_last_written)
            def _noise(b):
                if f"sub_backend/{b}/chain" in g:
                    a = _row(f"sub_backend/{b}/chain", (slice(0, nit),))
                    return None if a is None else np.asarray(a)[:, 0, :, 0, :]
                if f"chain/{b}" in g:
                    a = _row(f"chain/{b}", (slice(0, nit), 0, 0))
                    return None if a is None else np.asarray(a)[:, :, 0, :]
                return None

            S["psd"], S["gal"] = _noise("psd"), _noise("galfor")
            if S["psd"] is not None and S["gal"] is not None:
                sub_nit = min(ns["_last_written"](S["psd"]), ns["_last_written"](S["gal"]))
                if sub_nit < nit:
                    notes.append(f"noise chains written through row {sub_nit - 1} while "
                                 f"the main backend reached {nit - 1} (caught mid-flush); "
                                 "the noise panels use the last complete row.")
                S["psd"], S["gal"] = S["psd"][:sub_nit], S["gal"][:sub_nit]

            try:
                a = dict(g["domain_settings/args"].attrs)
                S["tobs"] = float(a["0"]) * float(a["1"]) * float(a["2"])
            except Exception:  # noqa: BLE001
                S["tobs"] = None
                notes.append("no domain_settings in the store; Tobs unknown, so "
                             "no overlap panel.")

            # the max-lnL walker's GB leaves at the last row, for the overlap
            S["wb"] = int(np.argmax(S["ll"][nit - 1]))
            S["rec9"] = None
            if inds_gb is not None:
                row = _row("chain/gb", (nit - 1, 0, 0, S["wb"]))
                if row is not None:
                    S["rec9"] = np.asarray(row, float)[np.asarray(inds_gb[nit - 1, S["wb"]], bool)]

            # the VGBs above SNR 7: the full page's rule (walker-mean <h|h>)
            S["vgb_snr"] = None
            if "sub_backend/vgb/h_h" in g:
                hh = _row("sub_backend/vgb/h_h", (slice(0, nit),))
                if hh is not None:
                    hh = np.asarray(hh, float)
                    k = ns["_last_written"](hh)
                    if k:
                        import warnings

                        with warnings.catch_warnings():       # an unsampled VGB is all-NaN
                            warnings.simplefilter("ignore", RuntimeWarning)
                            snr = np.sqrt(np.clip(np.nanmean(hh[k - 1], axis=0), 0, None))
                        S["vgb_snr"] = np.where(np.isfinite(snr), snr, 0.0)

            # the fixed-dimension source branches, over the source panels' pool window
            S["src"] = {}
            pool_w = int(ns["_src_pool_window"](nit))
            for b in SRC_BRANCHES:
                rows = _source_rows(g, _row, b, nit, pool_w)
                if rows is not None:
                    S["src"][b] = rows
            S["torn_rows"] = list(ns["_TORN_ROWS"])
        finally:
            bg = ns.get("_BACKUP_G")
            if bg:
                try:
                    bg.file.close()
                except Exception:  # noqa: BLE001
                    pass
            ns.update(g=None, _BACKUP_G=None)

    from lisatools.globalfit.stock.erebor.noise import (
        galfor_params_to_physical, read_noise_model_identity)

    S["galfor_log"] = bool(read_noise_model_identity(store).get("galfor_log_sampling", False))
    S["gal_phys"] = (galfor_params_to_physical(S["gal"], S["galfor_log"])
                     if S["gal"] is not None else None)
    if S["torn_rows"]:
        notes.append("rows read around a racing saver: " + "; ".join(S["torn_rows"]))
    return S


def current_stage(steps: list, nit: int, saved_after: list) -> Optional[dict]:
    """The step a resume would run, with rows-in and the next leg."""
    if not steps:
        return None
    cur = next((s for s in steps if not s["status"]), None)
    out = dict(total=len(steps), done=[s["name"] for s in steps if s["status"]],
               name=None, order=None, start=None, rows_in=None, next_leg=None)
    if cur is None:
        return out
    out.update(name=cur["name"], order=cur["order"], start=cur["start"])
    if cur["start"] is not None:
        out["rows_in"] = nit - int(cur["start"])
    mo, last = cur["move_order"], (saved_after[-1] if saved_after else None)
    if mo and last in mo:
        out["next_leg"] = mo[(mo.index(last) + 1) % len(mo)]
    return out


def _artifacts_file(run_dir: str, name: str) -> Optional[str]:
    for p in glob.glob(os.path.join(run_dir, "*_artifacts", name)):
        return p
    return None


def run_settings_t0(run_dir: str) -> float:
    """The GB branch's phase epoch from run_settings.log (the generator's regex)."""
    for p in glob.glob(os.path.join(run_dir, "*_artifacts", "run_settings.log")):
        try:
            with open(p, errors="replace") as fh:
                m = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", fh.read(), re.S)
            if m:
                return float(m.group(1))
        except OSError:
            continue
    return T_REF_DEFAULT


# --------------------------------------------------------------------------
# the other sources: SNR per walker and the pulls against the catalogue
# --------------------------------------------------------------------------
def source_summary(S: dict, run_dir: str, ns: dict, mojito_tree, notes: list) -> dict:
    """Per fixed-dimension branch: ``dict(params, pool, rows=[...])``, a row per
    source with its catalogue id, the matched SNR ``<d|h>/sqrt(<h|h>)`` per cold
    walker at the last row, the walkers more than ``SRC_SNR_OFF`` off the walker
    median, and each key parameter's ``(median - truth) / sigma`` over the pool
    window (the full page's source panels: the same pool, mask and truth map).
    """
    specs = ns["_SRC_BRANCH_SPECS"]
    settings = ""
    p = _artifacts_file(run_dir, "run_settings.log")
    if p:
        try:
            with open(p, errors="replace") as fh:
                settings = fh.read()
        except OSError:
            pass
    ids_all = ns["_parse_src_ids_from_settings"](settings) if settings else {}
    missing = []
    ns.update(MOJITO_CAT_DIR=mojito_tree or "", MISSING=missing, h5py=__import__("h5py"))
    out = {}
    for b, rows in S.get("src", {}).items():
        spec = specs[b]
        chain, inds = rows["chain"], rows["inds"]           # (W, nw, nl, nd), (W, nw, nl)
        nw, nleaves = chain.shape[1], chain.shape[2]
        cat = ns["_src_load_catalogue"](spec["cat_files"]) if mojito_tree else None
        ids = ids_all.get(spec["cls"], list(range(nleaves)))
        to_samp = ns[f"_src_cat_to_sampling_{b}"]
        key = [spec["param_defs"][k][0] for k in spec["key_idx"]]
        recs = []
        for leaf in range(nleaves):
            truth = None
            if cat is not None and leaf < len(ids):
                r = ns["_src_cat_row"](cat, ids[leaf])
                if r is not None:
                    try:
                        truth = to_samp(r)
                    except Exception as e:  # noqa: BLE001
                        missing.append(f"{b} leaf {leaf} truth: {type(e).__name__}: {e}")
            snr = None
            if rows["dh"] is not None and rows["hh"] is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr = rows["dh"][:, leaf] / np.sqrt(np.maximum(rows["hh"][:, leaf], 1e-30))
            # the pool: every (row, walker) the leaf is alive in, populated rows only
            # (an extract zero-fills rows outside its keep window)
            m = inds[:, :, leaf] & np.any(chain[:, :, leaf, :] != 0.0, axis=-1)
            pool = chain[:, :, leaf, :][m]
            pulls = []
            for k in spec["key_idx"]:
                v = pool[:, k][np.isfinite(pool[:, k])] if pool.size else np.array([])
                if truth is None or not v.size or not np.isfinite(truth[k]):
                    pulls.append(None)
                    continue
                sd = float(np.std(v))
                pulls.append((float(np.median(v)) - float(truth[k])) / sd if sd > 0 else None)
            off = []
            if snr is not None and np.all(np.isfinite(snr)):
                med = float(np.median(snr))
                off = [w for w in range(nw)
                       if abs(snr[w] - med) > max(SRC_SNR_OFF * abs(med), SRC_SNR_OFF_ABS)]
            recs.append(dict(leaf=leaf, id=ids[leaf] if leaf < len(ids) else None,
                             snr=snr, off=off, pulls=pulls, n_pool=int(pool.shape[0]),
                             truth=truth is not None))
        out[b] = dict(cls=spec["cls"], params=key, rows=recs, nw=nw,
                      pool=int(chain.shape[0]), has_cat=cat is not None)
        if cat is None:
            notes.append(f"{spec['cls']}: no catalogue under "
                         f"{_html.escape(str(mojito_tree or 'no mojito tree'))}/catalogues "
                         "-- SNRs only, no pulls.")
    notes.extend(_html.escape(m_) for m_ in missing)
    return out


# --------------------------------------------------------------------------
# the head log and the GPU sampler (mirrors gf_last_iteration.py: same tags,
# same windows, same parse; the test suite holds the two to one answer)
# --------------------------------------------------------------------------
_TAGS = (b"[SAVE] save_step", b"[LEG ", b"[GB_GATE ", b"[GB_BAND_SHUTOFF status ",
         b"[FANOUT] op=", b"[stageA] ", b"[sweep:", b"] STAGE COMPLETE", b"] stored row",
         b"F-stat grid fit epoch ")
_RE_LEG = re.compile(r"\[LEG (\S+)\] cycle (\d+) leg (\d+)/(\d+): ran \[(.*?)\] -> "
                     r"row saved after '([^']*)'")
_RE_RESUME = re.compile(r"\[LEG (\S+)\] resume:")
_RE_FANOUT = re.compile(r"\[FANOUT\] op=(\S+) move=(\S+)")
_RE_GATE = re.compile(r"occupied (\d+) \| shut (\d+) \(([\d.]+)%\) \| active (\d+)"
                      r"(?: -- streak p50 (\d+) p90 (\d+) of (\d+))?")
_RE_TOL = re.compile(r"\(tol ([\d.]+)\)")
_RE_EMPTY = re.compile(r"floor ([\d.]+) mHz: (\d+)/(\d+) bands eligible.*?; (\d+) off \(.*?\); "
                       r"(\d+)/(\d+) iters since revive")
_RE_EPOCH = re.compile(r"F-stat grid fit epoch (\d+) (?:done in ([\d.]+)s \((\d+) peaks\)|starting)")
_RE_STAGEA = re.compile(r"\[stageA\] comb \+ peak selection: (\d+) peaks in ([\d.]+)m .*?; "
                        r"(\d+) of (\d+) bands skipped")
_RE_SWEEP = re.compile(r"\[sweep:(\S+)\] \d+/\d+ evals \(([\d.]+)%\).*?ETA (\S+)")
_RE_DONE = re.compile(r"\[V9-STAGE (\S+)\] (STAGE COMPLETE.*)")


def _tail_lines(path, nbytes):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size > nbytes:
            fh.seek(size - nbytes)
            fh.readline()                               # drop the torn first line
        data = fh.read()
    return data.decode("utf-8", "replace").splitlines()


def find_head_log(run_dir: str) -> Optional[str]:
    """The head's run log, its snapshot ``_tail`` copy, or the newest job stdout
    (a superset of it, and the only log a short tar carries)."""
    for name in ("globalfit_run.log", "globalfit_run_tail.log"):
        hits = glob.glob(os.path.join(run_dir, "*_artifacts", name))
        if hits:
            return max(hits, key=os.path.getmtime)
    hits = [p for p in glob.glob(os.path.join(run_dir, "slurm_stdout_*.log"))
            if not p.endswith(("_filtered.log", "_tail.log"))]
    return max(hits, key=os.path.getmtime) if hits else None


def _ts(line):
    m = _TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _mins(a, b):
    return (b - a).total_seconds() / 60.0


def _scan(path, lo, hi):
    """The tagged lines of ``path[lo:hi]`` in file order, and the offset of the
    chunk's first whole line (the torn first line is dropped unless ``lo`` is 0)."""
    with open(path, "rb") as fh:
        fh.seek(lo)
        data = fh.read(hi - lo)
    start = data.find(b"\n") + 1 if lo > 0 else 0
    starts = set()
    for tag in _TAGS:
        i = data.find(tag, start)
        while i >= 0:
            starts.add(data.rfind(b"\n", 0, i) + 1)
            e = data.find(b"\n", i)
            if e < 0:
                break
            i = data.find(tag, e)
    lines = []
    for s in sorted(starts):
        e = data.find(b"\n", s)
        lines.append(data[s:e if e >= 0 else len(data)].decode("utf-8", "replace"))
    return lines, lo + start


def _last_ts(path):
    for line in reversed(_tail_lines(path, 256 * 1024)):
        t = _ts(line)
        if t is not None:
            return t
    return None


def _follows(prev, rec):
    if prev["stage"] != rec["stage"]:
        return False
    if rec["cycle"] == prev["cycle"]:
        return rec["leg"] == prev["leg"] + 1
    return prev["leg"] == prev["nlegs"] and rec["leg"] == 1 and rec["cycle"] == prev["cycle"] + 1


def _move_walls(fan, ran, t0):
    """Each move's wall seconds inside one leg, from its ``[FANOUT]`` lines (see
    gf_last_iteration.py: an inner move of a composite belongs to the next leg
    move that logs none of its own)."""
    direct = {m for _, m in fan if m in ran}
    end, cur = {}, None
    for t, m in fan:
        if m in ran:
            cur = m
        else:
            i0 = ran.index(cur) + 1 if cur in ran else 0
            cur = next((r for r in ran[i0:] if r not in direct), cur)
        if cur is not None:
            end[cur] = t
    walls, t_prev = {}, t0
    for m in ran:
        if m in end:
            walls[m] = (end[m] - t_prev).total_seconds()
            t_prev = end[m]
    return walls


def parse_head_log(lines):
    """The tagged head-log lines (file order) -> saves, legs (with walls), valve
    judges, empty-band status, F-stat epochs, the open epoch and its sweep, the
    last STAGE COMPLETE. Valve, empty-band and epoch lists restart at a STAGE
    COMPLETE line; a ``[LEG ...] resume:`` line breaks the leg chain."""
    out = dict(saves=[], legs=[], gates=[], empty=None, epochs=[], epoch_open=None, sweep=None,
               stage_done=None, replica=None)
    prev, fan, stage_a, seg = None, [], None, 0
    for line in lines:
        t = _ts(line)
        if t is None:
            continue
        if "[FANOUT] op=" in line:
            m = _RE_FANOUT.search(line)
            if m and m.group(2) != "None":
                fan.append((t, m.group(2)))
        elif "[SAVE] save_step" in line:
            out["saves"].append(t)
        elif "[LEG " in line:
            m = _RE_LEG.search(line)
            if m:
                rec = dict(t=t, seg=seg, stage=m.group(1), cycle=int(m.group(2)),
                           leg=int(m.group(3)), nlegs=int(m.group(4)),
                           ran=re.findall(r"'([^']*)'", m.group(5)), after=m.group(6),
                           wall=None, moves={})
                if prev is not None and _follows(prev, rec):
                    rec["wall"] = (t - prev["t"]).total_seconds()
                    rec["moves"] = _move_walls(fan, rec["ran"], prev["t"])
                out["legs"].append(rec)
                prev, fan = rec, []
            elif _RE_RESUME.search(line):
                prev, fan, seg = None, [], seg + 1
        elif "[GB_GATE " in line:
            m = _RE_GATE.search(line)
            if m:
                tol = _RE_TOL.search(line)
                out["gates"].append(dict(
                    t=t, occupied=int(m.group(1)), shut=int(m.group(2)),
                    shut_pct=float(m.group(3)), active=int(m.group(4)),
                    streak=None if m.group(5) is None else tuple(int(m.group(i)) for i in (5, 6, 7)),
                    tol=float(tol.group(1)) if tol else None))
        elif "[GB_BAND_SHUTOFF status" in line:
            m = _RE_EMPTY.search(line)
            if m:
                out["empty"] = dict(t=t, floor_mhz=float(m.group(1)), eligible=int(m.group(2)),
                                    nbands=int(m.group(3)), off=int(m.group(4)),
                                    since=int(m.group(5)), every=int(m.group(6)))
        elif "[stageA] " in line:
            m = _RE_STAGEA.search(line)
            if m:
                stage_a = dict(minutes=float(m.group(2)), skipped=int(m.group(3)),
                               nbands=int(m.group(4)))
        elif "F-stat grid fit epoch " in line:
            m = _RE_EPOCH.search(line)
            if m and m.group(2) is not None:
                out["epochs"].append(dict(t=t, epoch=int(m.group(1)), wall_s=float(m.group(2)),
                                          peaks=int(m.group(3)), stage_a=stage_a))
                out["epoch_open"], out["sweep"], stage_a = None, None, None
            elif m:
                out["epoch_open"], out["sweep"] = dict(t=t, epoch=int(m.group(1))), None
        elif "[sweep:" in line:
            m = _RE_SWEEP.search(line)
            if m and out["epoch_open"] is not None:
                out["sweep"] = dict(t=t, name=m.group(1), pct=float(m.group(2)), eta=m.group(3))
        elif "] STAGE COMPLETE" in line:
            m = _RE_DONE.search(line)
            if m:
                out["stage_done"] = dict(t=t, stage=m.group(1), text=m.group(2))
                out["gates"], out["empty"], out["epochs"] = [], None, []
        elif "[REPLICA_PE" in line and "] stored row" in line:
            out["replica"] = line.split("] ", 1)[-1]
    return out


def last_cycle(legs, stage):
    """``(cycle, total_s, walls)`` of the newest fully timed cycle of ``stage``."""
    by = {}
    for r in legs:
        if r["stage"] == stage:
            by.setdefault((r["seg"], r["cycle"]), {})[r["leg"]] = r
    for key in sorted(by, key=lambda k: max(r["t"] for r in by[k].values()), reverse=True):
        d = by[key]
        n = next(iter(d.values()))["nlegs"]
        if all(i in d and d[i]["wall"] is not None for i in range(1, n + 1)):
            walls = [d[i]["wall"] for i in range(1, n + 1)]
            return key[1], sum(walls), walls
    return None


def _enough(out):
    if out["stage_done"] is not None:
        return True
    if len(out["saves"]) < 7:
        return False
    if out["legs"] and last_cycle(out["legs"], out["legs"][-1]["stage"]) is None:
        return False
    if out["gates"] and len(out["gates"]) < TREND:
        return False
    if out["epochs"] and len(out["epochs"]) < TREND:
        return False
    return True


def read_live(run_dir: str) -> Optional[dict]:
    """The head log's reading (:func:`parse_head_log`) plus ``log``,
    ``log_age_min`` (now minus the log's mtime, so a tar read in another time
    zone reads right), ``t_end`` (its newest timestamp, its own clock);
    ``None`` without a log."""
    path = find_head_log(run_dir)
    if path is None:
        return None
    size = os.path.getsize(path)
    lines, lo, nbytes = [], size, TAIL_BYTES
    while True:
        new_lo = max(0, size - nbytes)
        chunk, lo = _scan(path, new_lo, lo)
        lines = chunk + lines
        out = parse_head_log(lines)
        if new_lo == 0 or nbytes >= TAIL_BYTES_MAX or _enough(out):
            break
        nbytes *= 2
    out.update(log=path, log_age_min=(time.time() - os.path.getmtime(path)) / 60.0,
               t_end=_last_ts(path))
    return out


def read_gpu(run_dir: str, window_s: float = 60.0) -> Optional[dict]:
    """The newest ``gpu_util_JOB.csv``: per GPU its last sample, the mean
    utilisation over the trailing ``window_s`` and the job's peak memory. The
    launcher's sampler runs on the HEAD node only."""
    cands = glob.glob(os.path.join(run_dir, "gpu_util_*.csv"))
    if not cands:
        return None
    path = max(cands, key=os.path.getmtime)
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", "replace").splitlines()
    peak = {}
    for line in text:
        p = line.split(",")
        if len(p) < 9:
            continue
        try:
            idx, used = int(p[1]), float(p[5]) / 1024.0
        except ValueError:
            continue
        peak[idx] = max(peak.get(idx, 0.0), used)
    rows = []
    for line in text[-400:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        try:
            rows.append(dict(t=datetime.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S"),
                             idx=int(p[1]), util=float(p[3]), used=float(p[5]) / 1024.0,
                             total=float(p[6]) / 1024.0))
        except ValueError:
            continue
    if not rows:
        return None
    t_last = max(r["t"] for r in rows)
    per = {}
    for r in rows:
        if (t_last - r["t"]).total_seconds() <= window_s:
            per.setdefault(r["idx"], []).append(r)
    gpus = {}
    for idx, rs in sorted(per.items()):
        last = max(rs, key=lambda r: r["t"])
        gpus[idx] = dict(util=last["util"], util_avg=sum(r["util"] for r in rs) / len(rs),
                         mem_used_gb=last["used"], mem_total_gb=last["total"],
                         mem_peak_gb=peak.get(idx, last["used"]))
    return dict(csv=path, t_last=t_last,
                age_min=(time.time() - os.path.getmtime(path)) / 60.0, gpus=gpus)


# --------------------------------------------------------------------------
# the phase-maximised overlap
# --------------------------------------------------------------------------
def phase_max_overlaps(Ar, Er, sr, At, Et, st, SA, SE, nw):
    """``|<a|b>| / sqrt(<a|a><b|b>)`` over the A and E channels, per pair.

    The modulus IS the maximum over an overall phase, so no phase grid is
    needed. ``Ar/Er`` (``At/Et``) are the recovered (injected) waveforms,
    ``nw`` FD points each starting at FD bin ``sr`` (``st``); ``SA/SE`` the
    noise on the full FD grid. A pair whose window leaves the grid scores 0.

    THIS MIRRORS THE GENERATOR'S INLINE LOOP (the ``MM = np.zeros(MI.size)``
    block of ``_generator.py``'s recovery section) statement for statement;
    ``tests/test_monitor_short_page.py`` executes that block against this
    function on the same inputs.
    """
    n = len(sr)
    MM = np.zeros(n)
    for i in range(n):
        o = min(sr[i], st[i])
        sp = max(sr[i], st[i]) + nw - o
        if o < 0 or o + sp > SA.size:
            continue
        sa = SA[o:o + sp]
        se = SE[o:o + sp]
        a1 = np.zeros(sp, complex)
        e1 = np.zeros(sp, complex)
        a2 = np.zeros(sp, complex)
        e2 = np.zeros(sp, complex)
        k = sr[i] - o
        a1[k:k + nw] = Ar[i]
        e1[k:k + nw] = Er[i]
        k = st[i] - o
        a2[k:k + nw] = At[i]
        e2[k:k + nw] = Et[i]
        num = np.sum(np.conj(a1) * a2 / sa) + np.sum(np.conj(e1) * e2 / se)
        n1 = np.sum(np.abs(a1) ** 2 / sa) + np.sum(np.abs(e1) ** 2 / se)
        n2 = np.sum(np.abs(a2) ** 2 / sa) + np.sum(np.abs(e2) ** 2 / se)
        MM[i] = float(np.abs(num) / np.sqrt(max(n1 * n2, 1e-300)))
    return np.clip(MM, 0.0, 1.0)


def _l1_orbits_from(mojito_tree):
    """``(orbits, label)``: the injected mojito L1 orbits via build_truth, the
    tree handed over through ``MOJITO_INFO_PATH`` for the duration of the call."""
    from .build_truth import l1_orbits

    saved = os.environ.get("MOJITO_INFO_PATH")
    try:
        if mojito_tree:
            os.environ["MOJITO_INFO_PATH"] = mojito_tree
        orb, where = l1_orbits()
    finally:
        if saved is None:
            os.environ.pop("MOJITO_INFO_PATH", None)
        else:
            os.environ["MOJITO_INFO_PATH"] = saved
    return orb, where


def compute_overlap(S: dict, truth_path: str, ns: dict, mojito_tree, t_ref: float,
                    notes: list, thresh: float) -> Optional[dict]:
    """The full page's headline match at the last row, for the max-lnL walker.

    Same route as ``_generator.py``'s recovery section: the frozen truth set's
    detectable sources in its stamped band, a globally-greedy one-to-one f0
    pairing within 2 FD bins (``_match_pairs``), then the phase-maximised,
    noise-weighted overlap of each pair under the walker-median noise of the
    last row, with GBGPU waveforms on the injected L1 orbits. ``matched`` =
    overlap at or above ``thresh`` (``GF_MONITOR_MATCH_MM``).
    """
    T = np.load(truth_path)
    tobs = S["tobs"]
    df = 1.0 / tobs
    flo, fhi = 3e-3, 21.94e-3                     # the generator's pre-stamp fallback
    if "band" in T.files:
        b = np.asarray(T["band"], float).reshape(-1)
        if b.size >= 2 and 0 < b[0] < b[1]:
            flo, fhi = float(b[0]), float(b[1])
    sel = np.asarray(T["det"], bool) & (T["f0"] >= flo) & (T["f0"] <= fhi)
    t_f0 = np.asarray(T["f0"], float)[sel]
    ndet = int(sel.sum())
    out = dict(ndet=ndet, band=(flo, fhi), walker=S["wb"], row=S["nit"] - 1,
               thresh=thresh, chance=ndet * (2 * TOL_BINS * df) / (fhi - flo))
    rec9 = S["rec9"]
    if rec9 is None or not rec9.size or not np.any(rec9[:, 1] != 0.0):
        notes.append("overlap: the max-lnL walker carries no GB coordinates at the "
                     "last row (unreadable, or outside an extract's keep window).")
        return None
    f_rec = rec9[:, 1] * 1e-3
    inb = (f_rec >= flo) & (f_rec <= fhi)
    rec9, f_rec = rec9[inb], f_rec[inb]
    MI, TI, _ = ns["_match_pairs"](f_rec, t_f0, TOL_BINS * df)
    out.update(n_band=int(rec9.shape[0]), n_proxy=int(MI.size),
               completeness_proxy=MI.size / max(ndet, 1),
               purity_proxy=MI.size / max(rec9.shape[0], 1), mm=None)
    try:
        from gbgpu.gbgpu import GBGPU
        from lisatools import detector as lisa_models
        from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

        from .build_truth import sens_grids

        SA, SE = sens_grids(np.median(S["psd"][-1], axis=0),
                            np.median(S["gal_phys"][-1], axis=0), df)
        orb, where = _l1_orbits_from(mojito_tree)
        if orb is None:
            notes.append("overlap: no mojito L1 brick reachable, so the waveforms use "
                         "the analytic DefaultOrbits ephemeris -- overlaps above ~5 mHz "
                         "carry an annual-Doppler phase error.")
            orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
            out["orbits"] = "analytic DefaultOrbits"
        else:
            out["orbits"] = "mojito L1 (" + os.path.basename(str(where)) + ")"
        gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(t_ref))
        tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                         use_distance=True, mc_lims=(0.001, 1.0))
        rphys = tc.both_transforms(np.asarray(rec9[MI], float).copy())
        tphys = np.asarray(T["phys"], float)[sel][TI]

        def _ae(phys):
            gbw.run_wave(*[np.ascontiguousarray(phys[:, k]) for k in range(9)],
                         N=NW_OVERLAP, T=tobs, dt=2.5, tdi2=True, tdi_channel_setup="AE")
            return (np.array(gbw.A), np.array(gbw.E),
                    np.asarray(gbw.start_inds).astype(int).copy())

        Ar, Er, sr = _ae(rphys)
        At, Et, st = _ae(tphys)
        mm = phase_max_overlaps(Ar, Er, sr, At, Et, st, SA, SE, NW_OVERLAP)
    except Exception as e:  # noqa: BLE001
        notes.append(f"overlap: waveform-level statistics unavailable "
                     f"({type(e).__name__}: {_html.escape(str(e))}); the headline falls "
                     "back to the 2-bin f0 proxy.")
        return out
    kept = mm >= thresh
    out.update(mm=mm, t_f0=t_f0[TI], n_match=int(kept.sum()),
               completeness=float(kept.sum()) / max(ndet, 1),
               purity=float(kept.sum()) / max(rec9.shape[0], 1),
               mm_med=float(np.median(mm)) if mm.size else float("nan"),
               mm_hi=float(np.mean(mm > 0.9)) if mm.size else 0.0)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _png(fig, dpi=None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    return base64.b64encode(shrink_png(buf.getvalue())).decode()


def _band_panel(ax, x, y, color, ylabel):
    ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=color, alpha=0.22, lw=0,
                    label="min-max over walkers")
    ax.plot(x, y.max(axis=1), color=color, lw=1.0)
    ax.plot(x, y.min(axis=1), color=color, lw=1.0)
    ax.plot(x, y.mean(axis=1), color=FG, lw=1.5, label="mean")
    ax.set_xlabel("stored row")
    ax.set_ylabel(ylabel)


def fig_history(ns, y, color, ylabel, title, target=None):
    """Whole run (left) and the recent window (right): min / mean / max over
    walkers, stage boundaries via the generator's ``mark_stages``; ``target``
    = ``(value, label)`` draws a dotted reference on the whole-run panel."""
    from matplotlib.figure import Figure

    nit = y.shape[0]
    x = np.arange(nit)
    nz = max(int(ns["_zoom_its"](nit)), 1)
    fig = Figure(figsize=(11.6, 3.5))
    ax = fig.subplots(1, 2)
    every = ns["STAGE_BOUNDS"]
    for a, sl, ttl in ((ax[0], slice(0, nit), f"{title}, all {nit} rows"),
                       (ax[1], slice(nit - nz, nit), f"last {nz} rows")):
        _band_panel(a, x[sl], y[sl], color, ylabel)
        lo, hi = int(x[sl][0]), int(x[sl][-1])
        # mark_stages has a max_it but no min_it, and its labels use a blended
        # (data, axes) transform that is NOT clipped -- a boundary left of the
        # zoom window would print its label beside the panel. Hand it only the
        # boundaries inside this window.
        ns["STAGE_BOUNDS"] = [b for b in every if lo <= b[0] <= hi]
        try:
            ns["mark_stages"](a, label=True, max_it=hi)
        finally:
            ns["STAGE_BOUNDS"] = every
        a.set_xlim(lo - 0.5, hi + 0.5)
        a.set_title(ttl)
    if target is not None:
        ax[0].axhline(target[0], color=RED, ls=":", lw=1.1)
        ax[0].text(0.99, target[0], f"{target[1]} ", color=RED, fontsize=8, va="bottom",
                   ha="right", transform=ax[0].get_yaxis_transform())
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_overlap(ns, ov):
    from matplotlib.figure import Figure

    mm = np.asarray(ov["mm"], float)
    fig = Figure(figsize=(11.6, 3.8))
    ax = fig.subplots(1, 2)
    s, n = ns["_survival"](mm)
    ax[0].plot(s, n, color=GREEN, lw=1.8, label=f"{mm.size} f0-paired sources")
    ax[0].axhline(ov["ndet"], color=FG, ls="--", lw=1.0)
    ax[0].text(0.02, ov["ndet"], f" {ov['ndet']} detectable injections", color=FG,
               fontsize=8, va="bottom")
    ax[0].axvline(ov["thresh"], color=RED, ls=":", lw=1.2)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, max(2500, 1.5 * ov["ndet"]))
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("phase-maximised overlap with the paired injection")
    ax[0].set_ylabel("sources at or above")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].scatter(ov["t_f0"] * 1e3, mm, s=5, color=CYAN, alpha=0.6, lw=0)
    ax[1].axhline(ov["thresh"], color=RED, ls=":", lw=1.2)
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("injected f0 [mHz]")
    ax[1].set_ylabel("overlap")
    ax[1].set_title(f"row {ov['row']}, max-lnL walker {ov['walker']}")
    fig.tight_layout()
    return _png(fig)


def noise_figures(S, soms_inj, sa_inj):
    """``(psd_now, psd_params)`` PNGs: the current sensitivity per cold walker
    and the instrument posteriors. (The per-parameter traces are the full
    page's Noise group: ``psd_trace`` / ``gal_trace``.)"""
    from matplotlib.figure import Figure

    from lisatools import detector as lisa_models
    from lisatools.sensitivity import LISASens, get_sensitivity
    from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground as FHT
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF

    psd, gph = S["psd"], S["gal_phys"]
    nw, nsub = psd.shape[1], psd.shape[0]
    orb0 = lisa_models.DefaultOrbits()
    fr = np.logspace(np.log10(2e-4), np.log10(2.6e-2), 500)

    def sens(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms ** 2, sa ** 2, orb0, "mon")
        if galp is None:
            return np.asarray(get_sensitivity(fr, sens_fn=LISASens, model=model,
                                              stochastic_params=()), float)
        return np.asarray(get_sensitivity(fr, sens_fn=LISASens, model=model,
                                          stochastic_params=tuple(galp),
                                          stochastic_function=HTGF), float)

    pm = np.median(psd[-1], axis=0)
    inj = sens(soms_inj, sa_inj)
    ref = inj + np.asarray(FHT.specific_Sh_function(fr, S["tobs"] or 7776000.0), float)
    fig = Figure(figsize=(11.6, 4.0))
    ax = fig.subplots(1, 2)
    ax[0].plot(fr, sens(*pm), color=CYAN, lw=1.6, label="instrument (walker median)")
    for w in range(nw):
        cw = sens(psd[-1, w, 0], psd[-1, w, 1], gph[-1, w])
        ax[0].plot(fr, cw, color=AMBER, lw=1.2, alpha=0.75,
                   label="instrument + foreground, per walker" if w == 0 else None)
        ax[1].plot(fr, cw / ref, color=AMBER, lw=1.2, alpha=0.75)
    ax[0].plot(fr, inj, color=RED, ls=":", lw=1.3, label="injected instrument")
    ax[0].plot(fr, ref, color=RED, ls="-.", lw=1.3, label="injected + FittedHT")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("f [Hz]")
    ax[0].set_ylabel("Sn(f) [LISASens]")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"sensitivity at row {nsub - 1}, all {nw} cold walkers")
    ax[1].axhline(1.0, color=FG, lw=1.0, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlim(2e-4, 2.6e-2)
    ax[1].set_xlabel("f [Hz]")
    ax[1].set_ylabel("fitted / (injected + FittedHT)")
    ax[1].set_title("each walker over the injected reference")
    fig.tight_layout()
    psd_now = _png(fig)

    nsh = min(3, nsub)
    fig = Figure(figsize=(11.6, 2.9))
    ax = fig.subplots(1, 2)
    for j, (name, injv, unit) in enumerate((("Soms_d", soms_inj, "m"),
                                            ("Sa_a", sa_inj, "m/s$^2$"))):
        v = psd[-nsh:, :, j].ravel()
        bias = float(np.median(v) / injv - 1.0)
        ax[j].hist(v, bins=max(8, min(26, v.size)), color=VIOLET, alpha=0.85)
        ax[j].axvline(injv, color=CYAN, lw=1.6, ls="--", label="injected")
        ax[j].set_title(f"{name} [{unit}]: median {100 * bias:+.2f}% from injected",
                        fontsize=10)
        ax[j].legend(fontsize=8)
        ax[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(f"instrument-noise posteriors, last {nsh} rows x {nw} cold walkers",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return psd_now, _png(fig)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
#: The base page style (the full page carries its own copy of these rules).
BASE_CSS = """
:root {
  --bg:#0A0E14; --panel:#10161F; --line:#223041; --fg:#B8C6D4; --dim:#67788A;
  --cyan:#4FD8EB; --amber:#F5A623; --green:#58C48A; --red:#E5484D; --violet:#9B7BFF;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 16px; z-index:5; display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; }
header h1 { font-size:15px; margin:0; letter-spacing:.06em; color:var(--cyan); text-transform:uppercase; }
header .stamp { color:var(--dim); font-size:12px; }
nav { display:flex; flex-wrap:wrap; gap:6px; padding:8px 16px; border-bottom:1px solid var(--line); }
nav a { color:var(--dim); text-decoration:none; font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:3px; }
nav a:hover, nav a:focus { color:var(--cyan); border-color:var(--cyan); outline:none; }
main { max-width:1240px; margin:0 auto; padding:16px 16px 60px; }
section { margin-top:26px; }
h2 { font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--fg);
  border-bottom:1px solid var(--line); padding-bottom:6px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:12px; margin-top:12px; overflow-x:auto; }
.panel img { max-width:100%; display:block; margin:0 auto; }
.caption { color:var(--dim); font-size:12px; margin-top:6px; }
.chip { display:inline-block; border:1px solid var(--line); border-radius:3px; padding:2px 9px; font-size:12px; color:var(--dim); }
.chip.done { color:var(--green); border-color:var(--green); }
.chip.now { color:var(--amber); border-color:var(--amber); }
.kpi { display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }
.kpi div { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 14px; }
.kpi b { display:block; font-size:18px; color:var(--cyan); font-variant-numeric:tabular-nums; }
.kpi span { font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }
.missing { color:var(--amber); border:1px dashed var(--amber); border-radius:4px; padding:10px; font-size:12px; margin-top:12px; }
code { color:var(--cyan); }
ul { color:var(--dim); font-size:13px; }
"""

#: What the at-a-glance block needs beyond the base style; the full page adds it to its own.
GLANCE_CSS = """
table.top { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; width:100%; }
table.top td { padding:4px 10px 4px 0; vertical-align:top; border-bottom:1px solid var(--line); }
table.top td.k { color:var(--dim); white-space:nowrap; width:1%; text-transform:uppercase; font-size:11px; letter-spacing:.05em; }
table.src { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; margin:4px 0 14px; }
table.src th, table.src td { padding:2px 10px 2px 0; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
table.src th { color:var(--dim); font-weight:normal; }
table.src th.l, table.src td.l { text-align:left; }
table.src td.p2 { color:var(--amber); }
table.src td.p3 { color:var(--red); font-weight:bold; }
table.src td.off { color:var(--red); text-align:left; white-space:normal; }
.warn { color:var(--red); }
.dim { color:var(--dim); }
"""


def _e(x) -> str:
    return _html.escape(str(x))


def _f(x) -> str:
    return f"{x:,.1f}"


def _pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def _age(m) -> str:
    if m is None:
        return "n/a"
    if m < -1.0:
        return f"{-m:.0f} min in the future (clock skew between the run's host and this one)"
    if m < 1.0:
        return f"{max(m, 0.0) * 60:.0f} s ago"
    if m < 120:
        return f"{m:.0f} min ago"
    return f"{m / 60:.1f} h ago"


def _dur(s) -> str:
    if s is None:
        return "-"
    if s < 100:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _trend(vals) -> str:
    return " &gt; ".join(str(v) for v in vals[-TREND:])


def _img(b64: Optional[str], alt: str) -> str:
    if not b64:
        return f'<div class="missing">plot unavailable: {_e(alt)}</div>'
    return f'<img src="data:image/png;base64,{b64}" alt="{_e(alt)}">'


def _live_rows(row, live, stage):
    """The topline rows read off the head log: pace, the leg in flight, the
    stage-end clock, the empty-band valve, the F-stat epochs."""
    if live is None:
        row("pace", "<span class='dim'>n/a (no head log or job stdout in the run dir)</span>")
        return
    t_end, log_age = live["t_end"], live["log_age_min"]

    def ago(t):
        return None if t is None or t_end is None else log_age + _mins(t, t_end)

    name = stage["name"] if stage else None
    recs = [r for r in live["legs"] if r["stage"] == name]
    bits = [f"log written {_age(log_age)}"]
    if live["saves"]:
        bits.append(f"last save {_age(ago(live['saves'][-1]))}")
    cyc = last_cycle(live["legs"], name) if name else None
    if cyc is not None:
        bits.append(f"last full cycle {_dur(cyc[1])} (cycle {cyc[0]}: "
                    + " + ".join(_dur(w) for w in cyc[2]) + ")")
    elif len(live["saves"]) > 1:
        s = live["saves"][-7:]
        c = [_mins(a, b) for a, b in zip(s[:-1], s[1:])]
        bits.append(f"row cadence: last {c[-1]:.0f} min, median of {len(c)} = "
                    f"{sorted(c)[len(c) // 2]:.0f} min")
    quiet = log_age > 30
    row("pace", (f"<span class='warn'>{' &middot; '.join(bits)}</span>" if quiet
                 else " &middot; ".join(bits))
        + (" <span class='dim'>(quiet log: a snapshot or a stopped job)</span>" if quiet else "")
        + f" <span class='dim'>({_e(os.path.basename(live['log']))})</span>")
    if recs and t_end is not None:
        last = recs[-1]
        k = last["leg"] % last["nlegs"] + 1
        timed = [r for r in recs if r["leg"] == k and r["wall"] is not None]
        prev = timed[-1] if timed else None
        ran = prev["ran"] if prev else next((r["ran"] for r in reversed(recs) if r["leg"] == k), [])
        names = f" ({_e(ran[0])} .. {_e(ran[-1])})" if len(ran) > 1 else (
            f" ({_e(ran[0])})" if ran else "")
        run_s = (t_end - last["t"]).total_seconds()
        s = f"leg {k}/{last['nlegs']}{names}: {_dur(run_s)} in"
        if prev is not None:
            s += f", {_dur(prev['wall'])} last time"
            if run_s > 1.3 * prev["wall"] + 300:
                s = f"<span class='warn'>{s} -- {_dur(run_s - prev['wall'])} OVER</span>"
        row("leg in flight", s + (" <span class='dim'>(as of the log's last line)</span>"
                                  if quiet else ""))
    if live["gates"]:
        g = live["gates"][-1]
        s = (f"<b>{g['active']:,}</b> active of {g['occupied']:,} occupied (walker, band) "
             f"pairs ({g['shut_pct']:.1f}% shut; the stage ends at 0)")
        if g["streak"]:
            s += f" &middot; streak p50 {g['streak'][0]} p90 {g['streak'][1]} of {g['streak'][2]}"
        if g["tol"] is not None:
            s += f" &middot; tol {g['tol']:g} nats"
        s += f" &middot; judged {_age(ago(g['t']))}"
        if len(live["gates"]) > 1:
            s += (f"<br><span class='dim'>active, last {min(TREND, len(live['gates']))} "
                  f"judges:</span> {_trend([x['active'] for x in live['gates']])}")
        row("rj valve", s)
    if live["empty"]:
        e = live["empty"]
        row("empty bands", f"{e['off']} of {e['eligible']} eligible bands off "
            f"(f &gt; {e['floor_mhz']:g} mHz, of {e['nbands']}) &middot; next revive check in "
            f"{max(e['every'] - e['since'], 0)} iteration(s)")
    if live["epochs"] or live["epoch_open"]:
        parts = []
        if live["epoch_open"] is not None:
            o = live["epoch_open"]
            s = f"epoch {o['epoch']} <b>FITTING</b> for {_dur((t_end - o['t']).total_seconds())}" \
                if t_end is not None else f"epoch {o['epoch']} fitting"
            if live["sweep"]:
                w = live["sweep"]
                s += f" &middot; {_e(w['name'])} {w['pct']:.0f}% ETA {_e(w['eta'])}"
            parts.append(s)
        if live["epochs"]:
            ep = live["epochs"][-1]
            s = (f"epoch {ep['epoch']}, {_age(ago(ep['t']))}: <b>{ep['peaks']}</b> peaks "
                 f"&middot; fit {_dur(ep['wall_s'])}")
            if ep["stage_a"]:
                a = ep["stage_a"]
                s += f" (stage A {a['minutes']:.1f} min, {a['skipped']}/{a['nbands']} bands skipped)"
            parts.append(s)
            if len(live["epochs"]) > 1:
                parts.append(f"<span class='dim'>peaks, last {min(TREND, len(live['epochs']))} "
                             f"epochs:</span> {_trend([x['peaks'] for x in live['epochs']])}")
        row("f-stat", "<br>".join(parts))
    if live.get("replica"):
        row("replica pe", _e(live["replica"][:160]))


def _topline(S, stage, live, gpu, ov, noise_bias, run_dir) -> str:
    nit, ll = S["nit"], S["ll"]
    R = []

    def row(k, v):
        R.append(f'<tr><td class="k">{k}</td><td>{v}</td></tr>')

    store = os.path.basename(S["store"])
    row("store", f"<code>{_e(store)}</code> <span class='dim'>in {_e(run_dir)}"
        + (" &middot; reduced extract" if S["extract"] else "") + "</span>")
    last_after = S["saved_after"][-1] if S["saved_after"] else None
    row("last saved row", f"<b>{nit - 1}</b> <span class='dim'>(rows are numbered from 0 "
        f"on this page, as on the full page: {nit} rows stored, iteration attr "
        f"{S.get('it_attr')})</span> &middot; saved after "
        f"<code>{_e(last_after or 'n/a')}</code>"
        + (f" &middot; next leg <code>{_e(stage['next_leg'])}</code>"
           if stage and stage.get("next_leg") else ""))
    if stage is None:
        row("stage", "none (no recipe group in the store)")
    elif stage["name"] is None:
        row("stage", f"<b>FINISHED</b>: all {stage['total']} steps complete")
    else:
        bits = [f"step {stage['order']} of {stage['total']}"]
        if stage["start"] is not None:
            sd = live["stage_done"] if live else None
            when = ""
            # the logged end of the step before this one is this one's start
            if (sd is not None and stage["done"] and sd["stage"] == stage["done"][-1]
                    and live["t_end"] is not None):
                when = f" ({_age(live['log_age_min'] + _mins(sd['t'], live['t_end']))})"
            bits.append(f"started row {int(stage['start'])}{when}, {stage['rows_in']} rows in")
        row("stage", f"<b>{_e(stage['name'])}</b> &middot; " + " &middot; ".join(bits)
            + (f" <span class='dim'>(done: {_e(', '.join(stage['done']))})</span>"
               if stage["done"] else ""))
    _live_rows(row, live, stage)
    if gpu is None:
        row("gpu (head node)", "<span class='dim'>n/a (no gpu_util_*.csv in the run dir)</span>")
    else:
        parts = [f"GPU{i} util {d['util']:.0f}% (60 s {d['util_avg']:.0f}%), mem "
                 f"{d['mem_used_gb']:.1f}/{d['mem_total_gb']:.1f} GiB, job peak "
                 f"{d['mem_peak_gb']:.1f}"
                 for i, d in gpu["gpus"].items()]
        stale = gpu["age_min"] is not None and gpu["age_min"] > 5
        row("gpu (head node)", " &middot; ".join(parts)
            + f" <span class='{'warn' if stale else 'dim'}'>sample "
            f"{_e(_age(gpu['age_min']))}{' -- STALE' if stale else ''}; head node only, a "
            f"second node is not sampled ({_e(os.path.basename(gpu['csv']))})</span>")
    L = ll[nit - 1]
    txt = (f"max <b>{_f(L.max())}</b> &middot; mean {_f(L.mean())} &middot; min "
           f"{_f(L.min())} &middot; spread {_f(L.max() - L.min())}")
    if nit > 1:
        P = ll[nit - 2]
        txt += (f"<br><span class='dim'>change vs row {nit - 2}:</span> max "
                f"{L.max() - P.max():+,.1f} &middot; mean {L.mean() - P.mean():+,.1f} "
                f"&middot; min {L.min() - P.min():+,.1f}")
    row(f"cold lnL, row {nit - 1}", txt)
    lv = S["leaves"]
    if lv:
        main = "gb" if "gb" in lv else sorted(lv)[0]
        v = lv[main][nit - 1]
        txt = (f"{main} min <b>{int(v.min()):,}</b> &middot; mean {v.mean():,.1f} "
               f"&middot; max {int(v.max()):,}")
        if nit > 1:
            pv = lv[main][nit - 2]
            txt += f" <span class='dim'>(mean {v.mean() - pv.mean():+.1f} vs row {nit - 2})</span>"
        others = []
        for b in sorted(lv):
            if b == main:
                continue
            w = lv[b][nit - 1]
            others.append(f"{b} {int(w.min())}" if w.min() == w.max()
                          else f"{b} {int(w.min())}-{int(w.max())}")
        if others:
            txt += "<br><span class='dim'>fixed-size branches:</span> " + " &middot; ".join(others)
        row("cold leaves", txt)
    if S.get("vgb_snr") is not None:
        snr = S["vgb_snr"]
        row("verification binaries", f"<b>{int((snr > 7.0).sum())}</b> of {snr.size} above "
            f"SNR 7 <span class='dim'>(walker-mean optimal SNR; median "
            f"{float(np.median(snr)):.1f})</span>")
    if ov is not None:
        if ov.get("mm") is not None:
            row("overlap match", f"<b>{ov['n_match']:,}</b> of {ov['ndet']:,} detectable "
                f"matched at overlap &ge; {ov['thresh']:.2f} &middot; completeness "
                f"{_pct(ov['completeness'])} &middot; purity {_pct(ov['purity'])} &middot; "
                f"median overlap {ov['mm_med']:.3f} <span class='dim'>(row {ov['row']}, "
                f"max-lnL walker {ov['walker']})</span>")
        else:
            row("overlap match", f"2-bin f0 proxy only: {ov['n_proxy']:,} of "
                f"{ov['ndet']:,} &middot; completeness {_pct(ov['completeness_proxy'])} "
                f"&middot; purity {_pct(ov['purity_proxy'])}")
    if noise_bias:
        row("instrument noise", " &middot; ".join(
            f"{k} median {100 * b:+.2f}% from injected" for k, b in noise_bias.items()))
    return '<table class="top">' + "".join(R) + "</table>"


def _sources_html(src: dict, nit: int) -> str:
    """One table per fixed-dimension branch: a row per source."""
    if not src:
        return ('<div class="missing">No MBHB / EMRI / SOBHB branch in this store '
                '(or its chains are unreadable).</div>')
    parts = []
    pool = nw = None
    for b in SRC_BRANCHES:
        if b not in src:
            continue
        d = src[b]
        pool, nw = d["pool"], d["nw"]
        head = (f"<tr><th class='l'>{_e(d['cls'])}</th><th>SNR</th>"
                f"<th>walkers</th>"
                + "".join(f"<th>{_e(p)}</th>" for p in d["params"])
                + "<th class='l'>note</th></tr>")
        body = []
        for r in d["rows"]:
            sid = f"src {r['id']}" if r["id"] is not None else f"leaf {r['leaf']}"
            if r["snr"] is not None and np.all(np.isfinite(r["snr"])):
                snr = f"<b>{float(np.median(r['snr'])):.1f}</b>"
                rng_ = f"{float(np.min(r['snr'])):.1f}-{float(np.max(r['snr'])):.1f}"
            else:
                snr, rng_ = "-", ""
            cells = []
            for p in r["pulls"]:
                if p is None or not np.isfinite(p):
                    cells.append("<td class='dim'>-</td>")
                    continue
                cls = " class='p3'" if abs(p) >= 3 else (" class='p2'" if abs(p) >= 2 else "")
                cells.append(f"<td{cls}>{p:+.1f}</td>")
            note = ""
            if r["off"]:
                med = float(np.median(r["snr"]))
                note = "; ".join(f"w{w} at SNR {float(r['snr'][w]):.1f} (median {med:.1f})"
                                 for w in r["off"])
            elif not r["truth"]:
                note = "<span class='dim'>no catalogue truth</span>"
            body.append(f"<tr><td class='l'>{_e(sid)}</td><td>{snr}</td><td class='dim'>{rng_}</td>"
                        + "".join(cells)
                        + f"<td class='off'>{note}</td></tr>")
        parts.append("<table class='src'>" + head + "".join(body) + "</table>")
    return ('<div class="panel">' + "".join(parts)
            + f'<div class="caption">Every MBHB, EMRI and SOBHB source at row {nit - 1}. '
            "<b>SNR</b> is the matched SNR &lang;d|h&rang;/&radic;&lang;h|h&rang; from the "
            f"sub-backend, median over the {nw} cold walkers, with their range; a walker more "
            f"than {SRC_SNR_OFF:.0%} (and {SRC_SNR_OFF_ABS:g} SNR units) off the median is named in "
            "the note column &mdash; it sits on a different mode or has lost the source, which "
            "in-model sampling rarely fixes by itself (tempering swaps or a leaf copy do). The "
            "parameter columns are each key parameter&rsquo;s (median &minus; catalogue truth) / "
            f"&sigma; over the last {pool} stored rows &times; the cold walkers, the pool the full "
            "page&rsquo;s per-source panels draw: |pull| &ge; 2 amber, &ge; 3 red. Early in a "
            "run that pool is the walkers&rsquo; scatter, not a posterior, so a large pull there "
            "reads as &ldquo;not there yet&rdquo; rather than as a bias.</div></div>")


def _chips(steps, stage) -> str:
    out = ""
    for s in steps:
        cls = "done" if s["status"] else ("now" if stage and s["name"] == stage["name"] else "")
        out += (f'<span class="chip {cls}">{s["order"]}. {_e(s["name"])}'
                f'{" &#10003;" if s["status"] else ""}</span> ')
    return out


def build_glance(run_dir: str, *, mojito: Optional[str] = None) -> dict:
    """Render the at-a-glance block for ``run_dir``.

    Returns ``dict(html, notes, phases, elapsed, chips, label, tobs_txt, nit,
    store, nav)``: ``html`` is the block's ``<section>`` elements (topline,
    likelihood, leaves, other sources, overlap, current noise) and ``nav``
    their ``(id, label)`` pairs. The short page wraps it in a page of its own;
    the full page puts it first. Uses ``GLANCE_CSS`` on top of the base style.
    """
    import matplotlib

    from . import check_truth, resolve_mojito_path

    t0 = time.perf_counter()
    phases, mark = {}, [t0]

    def _lap(name):
        """Charge the time since the previous lap to ``name`` (footer + log)."""
        now_ = time.perf_counter()
        phases[name] = phases.get(name, 0.0) + now_ - mark[0]
        mark[0] = now_

    run_dir = os.path.abspath(str(run_dir))
    notes = []
    ns = _generator_functions({"np": np})
    store = find_store(run_dir)
    S = read_store(store, run_dir, ns, notes)
    _lap("store")
    nit, nw = S["nit"], S["nwalk"]
    ns["STAGE_BOUNDS"] = stage_bounds(S["steps"])
    stage = current_stage(S["steps"], nit, S["saved_after"])
    gb_leaves = S["leaves"].get("gb")

    live = gpu = None
    try:
        live = read_live(run_dir)
    except Exception as e:  # noqa: BLE001
        notes.append(f"head log unreadable ({type(e).__name__}: {_e(e)})")
    if live is None:
        notes.append("no head log (*_artifacts/globalfit_run.log or _tail) and no "
                     "slurm_stdout_*.log: no pace, leg, valve or F-stat lines.")
    try:
        gpu = read_gpu(run_dir)
    except Exception as e:  # noqa: BLE001
        notes.append(f"gpu_util csv unreadable ({type(e).__name__}: {_e(e)})")
    if gpu is None:
        notes.append("no gpu_util_*.csv in the run dir: no GPU row.")
    _lap("log+gpu")

    moj, moj_src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    from lisatools.globalfit.stock.erebor.noise import psd_truth_levels

    soms_inj, sa_inj = psd_truth_levels(mojito_data_path=moj)
    if not moj:
        notes.append(f"no mojito tree ({_e(moj_src)}): the injected Soms_d / Sa_a are "
                     "the round analytic values, not the brick fit.")
    _lap("noise truth")

    src = {}
    try:
        src = source_summary(S, run_dir, ns, moj, notes)
    except Exception as e:  # noqa: BLE001
        notes.append(f"source table failed: {type(e).__name__}: {_e(e)}")
    _lap("sources")

    show_match = os.environ.get("GF_MONITOR_MATCH_STATS", "1") != "0"
    thresh = float(os.environ.get("GF_MONITOR_MATCH_MM", "0.8"))
    ov = None
    ov_note = ""
    if not show_match:
        ov_note = ("Overlap section SUPPRESSED (GF_MONITOR_MATCH_STATS=0). Nothing is "
                   "wrong with the run; unset the knob to include it.")
    elif S["tobs"] is None:
        ov_note = "No Tobs in the store, so no truth set can be matched."
    else:
        truth, tnote = check_truth(run_dir)
        if truth is None:
            ov_note = (f"No usable truth set: {_e(tnote)}. Build one with "
                       "<code>python -m lisatools.globalfit.monitor --build-truth "
                       "--short-page RUN_DIR</code> (tens of minutes, CPU-only).")
        else:
            try:
                ov = compute_overlap(S, truth, ns, moj, run_settings_t0(run_dir),
                                     notes, thresh)
            except Exception as e:  # noqa: BLE001
                ov_note = f"overlap failed: {type(e).__name__}: {_e(e)}"
            if ov is None and not ov_note:
                ov_note = "No GB coordinates to match at the last row (see Notes)."
    _lap("overlap")

    psd_bias = {}
    imgs = {}
    with matplotlib.rc_context(STYLE):
        imgs["lnl"] = fig_history(ns, S["ll"] - S["ll"][nit - 1].max(), CYAN,
                                  f"cold lnL - {S['ll'][nit - 1].max():,.1f}",
                                  "cold log-likelihood")
        if gb_leaves is not None:
            tgt = ((ov["ndet"], f"{ov['ndet']} detectable (SNR > 7)")
                   if ov is not None else None)
            imgs["leaves"] = fig_history(ns, gb_leaves, GREEN, "cold GB leaves",
                                         "GB leaf count", target=tgt)
        if ov is not None and ov.get("mm") is not None and ov["mm"].size:
            imgs["overlap"] = fig_overlap(ns, ov)
        _lap("history+overlap figures")
        if S["psd"] is not None and S["gal_phys"] is not None and S["psd"].shape[0]:
            try:
                imgs["psd_now"], imgs["psd_params"] = noise_figures(S, soms_inj, sa_inj)
            except Exception as e:  # noqa: BLE001
                notes.append(f"noise panels failed: {type(e).__name__}: {_e(e)}")
            _lap("noise figures")
            nsh = min(3, S["psd"].shape[0])
            for j, (k, injv) in enumerate((("Soms_d", soms_inj), ("Sa_a", sa_inj))):
                psd_bias[k] = float(np.median(S["psd"][-nsh:, :, j]) / injv - 1.0)
        else:
            notes.append("no psd/galfor chains in the store: no noise panels.")

    if ov is not None and ov.get("mm") is not None:
        kpi = (f'<div class="kpi"><div><b>{ov["n_match"]:,}</b><span>matched, overlap '
               f'&ge; {ov["thresh"]:.2f}</span></div>'
               f'<div><b>{_pct(ov["completeness"])}</b><span>completeness of '
               f'{ov["ndet"]:,}</span></div>'
               f'<div><b>{_pct(ov["purity"])}</b><span>purity of {ov["n_band"]:,} '
               f'in band</span></div>'
               f'<div><b>{ov["mm_med"]:.3f}</b><span>median overlap</span></div>'
               f'<div><b>{_pct(ov["mm_hi"])}</b><span>pairs above 0.9</span></div></div>')
        ov_html = (kpi + '<div class="panel">'
                   f'{_img(imgs.get("overlap"), "phase-maximised overlap")}'
                   f'<div class="caption">Every recovered source of the max-lnL cold walker '
                   f'(walker {ov["walker"]}) at row {ov["row"]} in '
                   f'{ov["band"][0] * 1e3:.4g}&ndash;{ov["band"][1] * 1e3:.4g} mHz is paired '
                   f'one-to-one with a detectable injection within {TOL_BINS:.0f} FD bins '
                   f'({ov["n_proxy"]:,} pairs; the windows cover {_pct(ov["chance"])} of the '
                   f'band, the chance rate), then scored by the noise-weighted overlap over A '
                   f'and E maximised over an overall phase, under the walker-median noise of '
                   f'that row, waveforms on {_e(ov.get("orbits", "?"))}. Left: how many pairs '
                   f'reach each overlap (the red line is the threshold); right: where the poor '
                   f'ones sit in frequency. Same criterion and numbers as the full page&rsquo;s '
                   f'recovery group.</div></div>')
    elif ov is not None:
        ov_html = (f'<div class="missing">Waveform overlaps unavailable (see Notes). '
                   f'2-bin f0 proxy: {ov["n_proxy"]:,} of {ov["ndet"]:,} detectable paired, '
                   f'completeness {_pct(ov["completeness_proxy"])}, purity '
                   f'{_pct(ov["purity_proxy"])}.</div>')
    else:
        ov_html = f'<div class="missing">{ov_note}</div>'

    html = f"""<section id="topline"><h2>Topline</h2>
{_topline(S, stage, live, gpu, ov, psd_bias, run_dir)}
</section>

<section id="lnl"><h2>Log-likelihood</h2>
<div class="panel">{_img(imgs.get("lnl"), "cold log-likelihood")}
<div class="caption">Cold-chain total log-likelihood per stored row, relative to the best
walker at the last row: the band spans min to max over the {nw} walkers, the white line is
the mean. Left the whole run, right the recent window. Grey dashed verticals are recipe stage
boundaries (completed and start iterations from the store's recipe group).</div></div>
</section>

<section id="leaves"><h2>GB Leaf Count</h2>
<div class="panel">{_img(imgs.get("leaves"), "GB leaf count")}
<div class="caption">Galactic-binary leaves per cold walker per stored row, min / mean /
max over walkers, same layout and marks{", against the detectable target (dotted red)" if ov else ""}.
The fixed-size branches are listed in the topline.</div></div>
</section>

<section id="sources"><h2>Other Sources</h2>
{_sources_html(src, nit)}
</section>

<section id="overlap"><h2>Phase-Maximised Overlap</h2>
{ov_html}
</section>

<section id="noise-now"><h2>Noise: Current Measurement</h2>
<div class="panel">{_img(imgs.get("psd_now"), "current sensitivity per walker")}
<div class="caption">The fitted noise at the last stored row as the sky-averaged
sensitivity: every cold walker's instrument + galactic foreground (amber) against the
injected instrument (red dotted) and the injected instrument + the FittedHT foreground
estimate (red dash-dot); right, each walker divided by that reference. The run multiplies
the foreground by a tabulated time modulation these static curves do not carry.</div></div>
<div class="panel">{_img(imgs.get("psd_params"), "instrument-noise posteriors")}
<div class="caption">The two instrument-noise parameters, the only noise parameters with a
truth to compare against.</div></div>
</section>
"""
    label = os.path.basename(run_dir)
    return dict(
        html=html, notes=notes, phases=phases, elapsed=time.perf_counter() - t0,
        chips=_chips(S["steps"], stage), label=label, nit=nit, store=store,
        tobs_txt=f"{S['tobs'] / 86400.0:.0f}-day" if S["tobs"] else "unknown-Tobs",
        nav=(("topline", "topline"), ("lnl", "likelihood"), ("leaves", "leaves"),
             ("sources", "other sources"), ("overlap", "overlap"),
             ("noise-now", "noise now")))


def render_short_page(run_dir: str, *, mojito: Optional[str] = None, now=None) -> str:
    """The short page for ``run_dir`` as one self-contained HTML string: the
    at-a-glance block (:func:`build_glance`) and its notes."""
    G = build_glance(run_dir, mojito=mojito)
    phase_txt = ", ".join(f"{k} {v:.1f}" for k, v in G["phases"].items())
    logger.info("short page phases (s): %s", phase_txt)
    notes_html = "".join(f"<li>{n}</li>" for n in G["notes"]) or "<li>none</li>"
    built = now or datetime.now()
    full_page = G["label"] + "_monitor.html"
    nav = "".join(f'<a href="#{i}">{_e(t)}</a>' for i, t in G["nav"]) + '<a href="#notes">notes</a>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GF short &middot; {_e(G["label"])} &middot; row {G["nit"] - 1}</title>
<style>{BASE_CSS}{GLANCE_CSS}</style>
</head>
<body>
<header>
  <h1>LISA Global Fit &middot; short status</h1>
  <span class="stamp">{_e(G["label"])} &middot; {G["tobs_txt"]} &middot; row {G["nit"] - 1} &middot; built {built:%Y-%m-%d %H:%M}</span>
  <span>{G["chips"]}</span>
</header>
<nav>
  {nav}
</nav>
<main>
{G["html"]}
<section id="notes"><h2>Notes</h2>
<ul>{notes_html}</ul>
<p class="caption">This is the SHORT page, and the same block opens the full page,
<code>{_e(full_page)}</code> beside the run folder
(<code>python -m lisatools.globalfit.monitor RUN_DIR</code>), where the residual spectrum,
recovered population, search gates, F-statistic, per-source posteriors, verification binaries
and detectability table follow in collapsed groups. Built in {G["elapsed"]:.1f} s
({_e(phase_txt)}) from <code>{_e(os.path.basename(G["store"]))}</code>.</p>
</section>
</main>
</body>
</html>
"""


def build_short_monitor(run_dir: str, out_path: Optional[str] = None, *,
                        mojito: Optional[str] = None, check: bool = True,
                        now=None) -> Optional[str]:
    """Write the short page; returns its path, or ``None`` when ``check=False``
    and the build failed.

    Published atomically (temp beside the target, then rename), so a browser
    refreshing the page never catches half a document, and a failed build
    leaves any previous page in place.
    """
    run_dir = os.path.abspath(str(run_dir))
    out_path = out_path or default_short_out_path(run_dir)
    tmp = out_path + ".tmp"
    st = time.perf_counter()
    try:
        page = render_short_page(run_dir, mojito=mojito, now=now)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(page)
        os.replace(tmp, out_path)
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning("short monitor page NOT refreshed (%s: %s). Any previous page "
                       "is untouched.\n%s", type(e).__name__, e,
                       traceback.format_exc().strip()[-4000:])
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        if check:
            raise
        return None
    logger.info("short monitor page built in %.1f s -> %s",
                time.perf_counter() - st, out_path)
    return out_path
