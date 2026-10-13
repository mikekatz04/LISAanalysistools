"""Other runs overlaid on a few panels of the full monitor page.

USER REQUEST 2026-10-12 (Mike): "make a change to the monitor call to add a
second run to the same html only in specific places where it is reasonable
visually to compare between runs, so the base of everything stays the specific
one you want (9 mo in this case), but I want to plot the phase maximized
overlap plots, a comparison of the final psd and foreground measurements
(final for the previous run and current for the ongoing run). In this case, we
want to be able to add the 6mo v9 run but it should be able to add a number of
runs. It should be something like --add-info /path/to/gf_prod_6mo_v9_4gpu."

THE PAGE STAYS ONE RUN. Every panel, table and sentence is the base run's, as
before. Each ADDED run is read at ITS LAST STORED ROW and drawn onto exactly
these figures of the full page, one legend entry per run:

* Recovery: the phase-maximised overlap CDF and its "sources at or above"
  twin (``f3_match``), recovered [%] vs the injection's optimal SNR
  (``f6_snr``), and recovered [%] per frequency bin (the lower panel of
  ``f5_counts``);
* Noise: the Soms_d / Sa_a posterior histograms (``f11_psd``, ``psd_hist``),
  the galfor parameter histograms (``gal_hist``), the PSD + foreground curves
  (``f11_fg``) and the sensitivity curves (``psd_curves``).

Nothing else of an added run is read or drawn. The short page and the
at-a-glance block stay single-run.

HOW AN ADDED RUN IS READ (one at a time; memory matters on the monitor rank):

1. :func:`load_added_runs` (before the noise figures): the store
   (:func:`._short.find_store`), its row count by the full page's rules
   (filled ``log_like`` extent, the ``iteration`` attr on a rewound store, a
   torn last row stepped back over), then ONLY the last row's ``inds/gb`` and
   the max-lnL cold walker's ``chain/gb`` row, the cold rung of the psd and
   galfor chains (trimmed to their last written row; the last <= 3 rows kept),
   the domain settings (Tobs), the noise identity (galfor basis) and the GB
   phase epoch from its own ``run_settings.log``. Every read goes through the
   generator's own torn-read ladder (``_row`` / ``_backup_group``, compiled
   out of ``_generator.py`` by name), so a racing saver falls back to THAT
   run's backup copy.
2. :func:`compute_recoveries` (inside the page's recovery section): the added
   run's OWN truth set (``gb_truth_3to21.npz``, else the newest
   ``gb_truth_*.npz``, in its run dir; never the CWD, never the base run's),
   refused on a Tobs mismatch, then the full page's headline match through
   :func:`._short.compute_overlap` -- the same pairing (``_match_pairs``,
   compiled from the generator), the same phase-maximised overlap loop (pinned
   to the generator's inline block by tests/test_monitor_short_page.py), under
   the added run's own walker-median noise, Tobs and GB epoch.

An added run that is missing, has no store, has no truth set, or whose matching
fails costs its own overlay only: a note in each affected caption, a MISSING
line on the page and an ``[add-info] MISSING`` log line.

Knob: ``GF_MONITOR_ADD_INFO`` (the ``add_info`` attribute, CLAUDE.md knob
rule): run directories joined by ``os.pathsep`` (``:``). The CLI's repeatable
``--add-info RUN_DIR`` wins over it when given; ``none`` / ``0`` / ``off``
means no added run (so ``--add-info none`` silences an exported list).
"""

from __future__ import annotations

import contextlib
import glob
import html as _html
import os
import time
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["ENV", "parse_add_info", "add_info_env_value", "run_label", "find_truth",
           "load_added_runs", "compute_recoveries", "overlay_overlap",
           "overlay_snr_completeness", "overlay_recovery_vs_freq", "overlay_hist",
           "overlay_curves", "galfor_in_basis", "caption_html", "STYLES"]

#: The env knob (attribute ``add_info``; GF_MONITOR_ is the monitor's namespace).
ENV = "GF_MONITOR_ADD_INFO"
#: Values meaning "no added run" (so an exported list can be silenced).
OFF = ("none", "0", "off", "false", "no")
#: The fallback truth-set Tobs for an unstamped npz (the generator's rule).
TOBS_UNSTAMPED = 7776000.0

#: One style per REQUESTED position (colour follows the entity: a skipped run
#: never repaints the others). Muted hues chosen against everything these
#: figures already draw on the dark panel (#10161F): cyan, amber and its ramp,
#: green, red, violet, the light reference grey and the base arm's pink.
#: Computed in OKLab (x100): the steel blue and the plum clear every base data
#: colour by >= 16 (normal vision) and >= 13 (protan / deutan); the pale lime
#: by >= 14 except the faint early end of the amber ramp. Adjacent added runs
#: (1-2, 2-3) are >= 39 apart; 1 and 3 are 13.6 apart (4.1 under CVD), so the
#: line style is the second channel: dashed / dotted / dash-dot. A fourth run
#: and later fold into one muted grey with its own dash pattern, never a
#: cycled hue.
STYLES = (
    dict(color="#3A74AC", ls="--", word="dashed"),
    dict(color="#C9E89A", ls=":", word="dotted"),
    dict(color="#93589A", ls="-.", word="dash-dot"),
)
STYLE_OTHER = dict(color="#8C96A3", ls=(0, (5, 1.5, 1, 1.5, 1, 1.5)), word="dash-dot-dot")


def style(k: int) -> dict:
    """The line style of the ``k``-th REQUESTED added run."""
    return dict(STYLES[k]) if 0 <= k < len(STYLES) else dict(STYLE_OTHER)


#: Drawing failures of the current page build (reset by :func:`load_added_runs`;
#: the page lists them under its MISSING notes). The overlay calls sit outside
#: any ``try`` in the generator, so they must never raise: an overlay that
#: fails costs that overlay, never the page.
OVERLAY_ERRORS: list = []


@contextlib.contextmanager
def _guard(where: str, A: dict):
    """One added run's drawing: an error is recorded and swallowed, so the
    other runs on the same figure still draw."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        msg = f"added run {A.get('label')}: {where} failed: {type(e).__name__}: {e}"
        OVERLAY_ERRORS.append(_html.escape(msg))
        print(f"[add-info] {msg}")


def _never_raises(default):
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                msg = f"added-run overlay {fn.__name__} failed: {type(e).__name__}: {e}"
                OVERLAY_ERRORS.append(_html.escape(msg))
                print(f"[add-info] {msg}")
                return default
        return wrapper
    return deco


# --------------------------------------------------------------------------
# the knob
# --------------------------------------------------------------------------
def _split(value) -> list:
    if value is None:
        return []
    return [p.strip() for p in str(value).split(os.pathsep) if p.strip()]


def _key(path) -> str:
    return os.path.realpath(os.path.expanduser(str(path).rstrip("/") or "/"))


def parse_add_info(cli: Optional[Sequence[str]] = None, environ=None,
                   base_run_dir: Optional[str] = None) -> list:
    """The added run directories, in the order given, duplicates dropped.

    ``cli`` is the list of ``--add-info`` values (each may itself be an
    ``os.pathsep``-joined list); when it is given (not ``None`` and not empty)
    it WINS over the environment (explicit > env, the stock-fit rule).
    Otherwise ``GF_MONITOR_ADD_INFO`` is read. ``none`` / ``0`` / ``off``
    entries mean "no added run" and are dropped; the base run itself is
    dropped too (a page never overlays itself). Paths are returned absolute,
    as given otherwise; two spellings of one directory count once.
    """
    env = os.environ if environ is None else environ
    raw = []
    if cli:
        for v in cli:
            raw += _split(v)
    else:
        raw = _split(env.get(ENV))
    seen = {_key(base_run_dir)} if base_run_dir else set()
    out = []
    for r in raw:
        if r.lower() in OFF:
            continue
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(os.path.abspath(os.path.expanduser(r)))
    return out


def add_info_env_value(dirs: Iterable[str]) -> str:
    """The ``GF_MONITOR_ADD_INFO`` value for ``dirs`` (``none`` for an empty
    list, so an explicit empty CLI choice also silences an inherited export)."""
    dirs = [str(d) for d in dirs]
    return os.pathsep.join(dirs) if dirs else "none"


def run_label(run_dir: str) -> str:
    """The short label the page gives a run: its arm tag (``9mo_v9``, ``6mo``),
    from the run directory's name through ``_identity`` -- the same mechanism
    the page uses for its own run, read-only."""
    from ._identity import arm_tag, run_identity

    return arm_tag(run_identity(os.path.basename(os.path.normpath(run_dir)))[1])


def find_truth(run_dir: str) -> Optional[str]:
    """The run's OWN truth set: ``gb_truth_3to21.npz`` in its run dir, else the
    newest ``gb_truth_*.npz`` there. Never the CWD (that is the base run's
    lookup, and a page renders from the base run's directory)."""
    from . import TRUTH_NAME

    p = os.path.join(run_dir, TRUTH_NAME)
    if os.path.isfile(p):
        return p
    cands = sorted(glob.glob(os.path.join(run_dir, "gb_truth_*.npz")), key=os.path.getmtime)
    return cands[-1] if cands else None


# --------------------------------------------------------------------------
# reading an added run (phase 1: the last row and the noise)
# --------------------------------------------------------------------------
_HELPERS = ("_row", "_backup_group", "_last_written", "_match_pairs")


def _helpers() -> dict:
    """The generator's torn-read ladder, sub-backend trim and pairing, compiled
    by name into a namespace of our own (never the running page's globals)."""
    from ._short import _generator_functions

    return _generator_functions({"np": np}, names=_HELPERS)


def _find_store(run_dir: str) -> str:
    """The added run's store: the newest ``*testing*.h5`` that is not a backup
    or quarantined copy (an ``_extract`` counts), else :func:`._short.find_store`.
    The ``testing`` filter matters for a production run directory, which also
    holds the EMRI / SOBBH ``wdm_lookup_*.h5`` tables: once that run stops
    saving, a newest-``*.h5`` rule alone could land on one of them."""
    from ._short import SKIP, find_store

    cands = [p for p in glob.glob(os.path.join(run_dir, "*.h5"))
             if "testing" in os.path.basename(p)
             and not any(s in os.path.basename(p) for s in SKIP)]
    return max(cands, key=os.path.getmtime) if cands else find_store(run_dir)


def _read_last(run_dir: str, ns: dict, notes: list, noise_rows: int = 3) -> dict:
    """What the overlays need from one added run, and nothing else."""
    import h5py

    from ._short import run_settings_t0

    store = _find_store(run_dir)
    S = dict(store=store)
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        ns.update(g=g, NIT=0, RUN_DIR=run_dir, _BACKUP_G=None, _TORN_ROWS=[],
                  h5py=h5py, os=os, np=np)
        try:
            _row = ns["_row"]
            ll = _row("log_like", (slice(None), 0, 0))
            if ll is None:
                raise RuntimeError("log_like is unreadable in the store and its backup")
            ll = np.asarray(ll, float)
            filled = np.where(np.any(ll != 0.0, axis=1))[0]
            if not filled.size:
                raise RuntimeError("no filled rows yet")
            nit = int(filled.max()) + 1
            it_attr = g.attrs.get("iteration")
            if it_attr is not None and 0 < int(it_attr) < nit:
                notes.append(f"store rewound: iteration attr {int(it_attr)} < filled rows {nit}")
                nit = int(it_attr)
            ns["NIT"] = nit

            def _alive(j):
                r = _row("inds/gb", (j, 0, 0))
                return None if r is None else np.asarray(r, bool)

            alive = _alive(nit - 1)
            if alive is None or not alive.any():
                # the full page's torn-last-row rule: step back to the last
                # row that carries GB leaves (one row read at a time)
                for j in range(nit - 2, max(nit - 40, -1), -1):
                    a = _alive(j)
                    if a is not None and a.any():
                        notes.append(f"stored row {nit - 1} carries no GB leaves (a torn "
                                     f"save); read row {j}")
                        nit, alive = j + 1, a
                        break
                else:
                    raise RuntimeError(f"stored row {nit - 1} and the 40 before it carry "
                                       "no GB leaves")
                ns["NIT"] = nit
            wb = int(np.argmax(ll[nit - 1]))
            row = _row("chain/gb", (nit - 1, 0, 0, wb))
            if row is None:
                raise RuntimeError(f"chain/gb row {nit - 1} is unreadable")
            S.update(nit=nit, row=nit - 1, wb=wb, nwalk=int(ll.shape[1]),
                     rec9=np.asarray(row, float)[alive[wb]])
            del row

            def _noise(b):
                if f"sub_backend/{b}/chain" in g:
                    a = _row(f"sub_backend/{b}/chain", (slice(0, nit), 0))
                    return None if a is None else np.asarray(a, float)[:, :, 0, :]
                if f"chain/{b}" in g:
                    a = _row(f"chain/{b}", (slice(0, nit), 0, 0))
                    return None if a is None else np.asarray(a, float)[:, :, 0, :]
                return None

            psd, gal = _noise("psd"), _noise("galfor")
            if psd is None or gal is None:
                raise RuntimeError("no readable psd / galfor chain")
            sub_nit = min(ns["_last_written"](psd), ns["_last_written"](gal))
            if sub_nit < 1:
                raise RuntimeError("the psd / galfor chains carry no written row")
            k = min(int(noise_rows), sub_nit)
            S.update(psd=psd[sub_nit - k:sub_nit].copy(), gal=gal[sub_nit - k:sub_nit].copy(),
                     noise_row=sub_nit - 1)
            del psd, gal
            try:
                a = dict(g["domain_settings/args"].attrs)
                S["tobs"] = float(a["0"]) * float(a["1"]) * float(a["2"])
            except Exception:  # noqa: BLE001
                S["tobs"] = None
            if ns["_TORN_ROWS"]:
                notes.append("rows read around a racing saver: " + "; ".join(ns["_TORN_ROWS"]))
        finally:
            bg = ns.get("_BACKUP_G")
            if bg:
                try:
                    bg.file.close()
                except Exception:  # noqa: BLE001
                    pass
            ns.update(g=None, _BACKUP_G=None)

    from lisatools.globalfit.stock.erebor.noise import (galfor_params_to_physical,
                                                        read_noise_model_identity)

    S["galfor_log"] = bool(read_noise_model_identity(store).get("galfor_log_sampling", False))
    S["gal_phys"] = galfor_params_to_physical(S["gal"], S["galfor_log"])
    S["t_ref"] = run_settings_t0(run_dir)
    return S


def _log(log, msg):
    if log is not None:
        log(msg)


def _miss(A, reason, missing, log, what="not overlaid"):
    line = (f"added run {A['label']} ({A['dir']}) {what}: {reason}")
    if missing is not None:
        missing.append(_html.escape(line))
    _log(log, f"[add-info] MISSING {A['label']}: {reason}")


def load_added_runs(dirs: Sequence[str], *, base_label: str, missing: Optional[list] = None,
                    log=print) -> list:
    """Phase 1: one dict per REQUESTED run, in order (``skip`` is the reason a
    run could not be read, else ``None``). Reads each store's last row and
    noise chains, one store open at a time.

    Runs inside ``matplotlib.rc_context``: the noise helpers come from
    ``erebor.noise``, whose import chain restyles matplotlib (eryn's
    ``plt.style.use``, font.size 10 -> 16) -- on the page that runs this before
    its first figure, that would restyle the likelihood panel.
    """
    import matplotlib

    del OVERLAY_ERRORS[:]                # a new page build (the monitor rank reuses this process)
    out = []
    used = {base_label}
    ns = None
    with matplotlib.rc_context():
        for k, d in enumerate(dirs):
            # one legend entry per run: a label already taken (the base run's,
            # or another added run's) falls back to the directory name, then
            # to a numbered copy of it
            label = run_label(d)
            if label in used:
                label = os.path.basename(os.path.normpath(d))
            stem, i = label, 2
            while label in used:
                label, i = f"{stem} ({i})", i + 1
            used.add(label)
            A = dict(dir=d, label=label, style=style(k), skip=None, rec=None, rec_skip=None,
                     notes=[])
            out.append(A)
            if not os.path.isdir(d):
                A["skip"] = "no such run directory"
                _miss(A, A["skip"], missing, log)
                continue
            st = time.perf_counter()
            try:
                if ns is None:
                    ns = _helpers()
                S = _read_last(d, ns, A["notes"])
            except FileNotFoundError:
                A["skip"] = "no store (*.h5) in its run directory"
            except Exception as e:  # noqa: BLE001
                A["skip"] = f"store unreadable ({type(e).__name__}: {e})"
            if A["skip"]:
                _miss(A, A["skip"], missing, log)
                continue
            A.update(S)
            A["store_name"] = os.path.basename(S["store"])
            _log(log, f"[add-info] {label}: row {S['row']} of {A['store_name']} (max-lnL walker "
                      f"{S['wb']}, {int(S['rec9'].shape[0])} GB leaves; noise row "
                      f"{S['noise_row']}; Tobs {(S['tobs'] or 0) / 86400.0:.0f} d) in "
                      f"{time.perf_counter() - st:.1f} s")
    return out


# --------------------------------------------------------------------------
# phase 2: the match against the added run's own truth set
# --------------------------------------------------------------------------
def compute_recoveries(runs: list, *, mojito_tree=None, thresh: float = 0.8,
                       missing: Optional[list] = None, log=print) -> int:
    """Phase 2, one run at a time: the full page's headline match for every
    readable added run (``A["rec"]``), or the reason it has none
    (``A["rec_skip"]``). The L1 orbits are built once for all of them. Returns
    the number of runs matched. Never raises."""
    from ._short import _l1_orbits_from, compute_overlap

    orbits = None
    n = 0
    ns = None
    for A in runs:
        if A["skip"] is not None:
            continue
        st = time.perf_counter()
        try:
            truth = find_truth(A["dir"])
            if truth is None:
                raise _Skip("no gb_truth_*.npz in its run directory (build one with "
                            "python -m lisatools.globalfit.monitor --build-truth on that run)")
            if not A.get("tobs"):
                raise _Skip("no domain settings in its store, so no Tobs to match at")
            T = np.load(truth)
            t_have = (float(np.asarray(T["tobs"]).reshape(-1)[0]) if "tobs" in T.files
                      else TOBS_UNSTAMPED)
            orb_stamp = str(T["orbits"]) if "orbits" in T.files else "unstamped (analytic)"
            del T
            if abs(t_have - A["tobs"]) > 1.0:
                raise _Skip(f"its truth set {os.path.basename(truth)} is for Tobs "
                            f"{t_have / 86400.0:.0f} d but the run is {A['tobs'] / 86400.0:.0f} d")
            if ns is None:
                ns = _helpers()
            if orbits is None:
                orbits = _l1_orbits_from(mojito_tree)
            notes = A["notes"]
            ov = compute_overlap(A, truth, ns, mojito_tree, A["t_ref"], notes, thresh,
                                 orbits=orbits)
            if ov is None:
                raise _Skip("no GB coordinates at its last row")
            if ov.get("mm") is None:
                notes.append("waveform overlaps unavailable; the recovery curves use the "
                             "2-bin f0 proxy")
            ov["truth"] = os.path.basename(truth)
            ov["truth_orbits"] = orb_stamp
            A["rec"] = ov
            n += 1
            _log(log, f"[add-info] {A['label']}: "
                      + (f"{ov['n_match']:,} matched (phase-max overlap >= {thresh:.2f})"
                         if ov.get("mm") is not None else f"{ov['n_proxy']:,} f0-paired (proxy)")
                      + f" of {ov['ndet']:,} detectable, own truth {ov['truth']}, in "
                        f"{time.perf_counter() - st:.1f} s")
        except _Skip as e:
            A["rec_skip"] = str(e)
        except Exception as e:  # noqa: BLE001
            A["rec_skip"] = f"matching failed ({type(e).__name__}: {e})"
        finally:
            A.pop("rec9", None)                 # the leaves are no longer needed
        if A["rec_skip"]:
            _miss(A, A["rec_skip"], missing, log, what="has no recovery overlay")
    return n


class _Skip(Exception):
    """A reason an added run's recovery overlay is not drawn."""


# --------------------------------------------------------------------------
# drawing (each returns the number of runs it drew; the base is never touched)
# --------------------------------------------------------------------------
def _drawable_rec(runs):
    return [A for A in runs if A.get("rec") is not None]


def _drawable_noise(runs):
    return [A for A in runs if A["skip"] is None and A.get("psd") is not None]


@_never_raises(0)
def overlay_overlap(ax_cdf, ax_surv, runs, *, survival) -> int:
    """``f3_match``: each added run's phase-maximised overlap CDF (top) and its
    survival count (bottom), labelled as the page labels its own run
    (``"<tag>  (<n> matched)"``). ``survival`` is the generator's ``_survival``."""
    n = 0
    for A in _drawable_rec(runs):
        with _guard("overlay_overlap", A):
            mm = A["rec"].get("mm")
            if mm is None or not np.size(mm):
                continue
            m = np.sort(np.asarray(mm, float))
            x, c = survival(m)
            s = A["style"]
            ax_cdf.plot(m, np.arange(1, m.size + 1) / m.size, color=s["color"], ls=s["ls"],
                        lw=1.5, alpha=0.95, label=f"{A['label']}  ({m.size} matched)")
            ax_surv.plot(x, c, color=s["color"], ls=s["ls"], lw=1.5, alpha=0.95)
            n += 1
    return n


def _found(R, overlap=True):
    """Truth-index mask of the detectable set found: the overlap-refined match
    when computed (and asked for), else the 2-bin proxy -- the base's rule."""
    ti = R["ti_ov"] if (overlap and R.get("ti_ov") is not None) else R["ti"]
    f = np.zeros(int(R["ndet"]), bool)
    f[np.asarray(ti, int)] = True
    return f


@_never_raises(0)
def overlay_snr_completeness(ax, runs, seds, *, wilson, n_base: int = 1) -> int:
    """``f6_snr``: each added run's recovered fraction per injected-SNR bin
    (its own truth set's SNRs, its own overlap-refined matches), offset to the
    right of the base arms. ``wilson`` is the generator's ``_wilson``."""
    n = 0
    for q, A in enumerate(_drawable_rec(runs)):
        with _guard("overlay_snr_completeness", A):
            R = A["rec"]
            snr = np.asarray(R["det_snr"], float)
            fnd = _found(R, overlap=True)
            x, y, el, eh = [], [], [], []
            for j, (a2, b2) in enumerate(zip(seds[:-1], seds[1:])):
                m = (snr >= a2) & (snr < b2)
                if not m.sum():
                    continue
                p = fnd[m].mean()
                lo, hi = wilson(fnd[m].sum(), m.sum())
                x.append(j + ((n_base - 1) / 2 + 1 + q) * 0.10)
                y.append(100 * p)
                el.append(100 * (p - lo))
                eh.append(100 * (hi - p))
            if not x:
                continue
            s = A["style"]
            ax.errorbar(x, y, yerr=[np.clip(el, 0, None), np.clip(eh, 0, None)], fmt="o",
                        ls=s["ls"], ms=4, mfc="none", color=s["color"], lw=1.3, capsize=2,
                        alpha=0.95, label=f"{A['label']}, row {A['row']}")
            n += 1
    return n


@_never_raises(0)
def overlay_recovery_vs_freq(ax, runs, eds, *, base_color: str, base_label: str) -> int:
    """``f5_counts`` lower panel: each added run's recovered fraction per
    frequency bin (the same bins), its own detectable set under the base
    panel's criterion (the 2-bin f0 proxy pairing). Adds a legend (the base
    panel has none) naming the base run too."""
    from matplotlib.lines import Line2D

    fc = np.sqrt(eds[:-1] * eds[1:])
    handles = []
    for A in _drawable_rec(runs):
        with _guard("overlay_recovery_vs_freq", A):
            R = A["rec"]
            f0 = np.asarray(R["det_f0"], float)
            hdet, _ = np.histogram(f0, bins=eds)
            hrec, _ = np.histogram(f0[_found(R, overlap=False)], bins=eds)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = np.where(hdet > 0, hrec / np.maximum(hdet, 1), np.nan)
            s = A["style"]
            (h,) = ax.plot(fc, 100 * fr, color=s["color"], ls=s["ls"], lw=1.4, marker="o",
                           ms=3.5, mfc="none", alpha=0.95, label=f"{A['label']}, row {A['row']}")
            handles.append(h)
    if handles:
        base = Line2D([], [], color=base_color, marker="o", ls="none", ms=3.5,
                      label=base_label)
        # lower right: low recovery at high frequency is the rare corner
        ax.legend(handles=[base] + handles, fontsize=7, loc="lower right",
                  ncol=len(handles) + 1)
    return len(handles)


def _median_mark(ax, x, color):
    """A downward triangle at the top edge of ``ax`` at ``x`` (data)."""
    ax.plot([x], [1.0], marker="v", ms=7, color=color, ls="none", clip_on=False,
            transform=ax.get_xaxis_transform(), zorder=5)


@_never_raises(0)
def overlay_hist(axes, runs, values, *, bins, base_color: str, base_label: str,
                 legend_axes=(0,), legend_kw=None, figure_legend=False) -> int:
    """An outlined histogram per added run on each panel of ``axes`` (``values(A,
    j)`` -> that run's samples for panel ``j``), with its median marked at the
    top edge; each added run in its own colour and line style, the base run's
    filled histogram untouched. The legend of each panel in ``legend_axes``
    becomes: the base run (a filled patch), whatever it already carried, then
    the added runs. ``figure_legend=True`` puts that legend ONCE under the
    panels instead (small panels, where it would cover a histogram)."""
    from matplotlib.patches import Patch

    axes = list(np.atleast_1d(axes))
    if figure_legend:
        legend_axes = ()
    n = 0
    for A in _drawable_noise(runs):
        with _guard("overlay_hist", A):
            # every panel's samples first, so a failing run draws nothing at all
            vals = []
            for j in range(len(axes)):
                v = np.asarray(values(A, j), float).ravel()
                vals.append(v[np.isfinite(v)])
            s = A["style"]
            lab = f"{A['label']}, last {A['psd'].shape[0]} rows to {A['noise_row']}"
            for ax, v in zip(axes, vals):
                if not v.size:
                    continue
                ax.hist(v, bins=bins, histtype="step", color=s["color"], ls=s["ls"], lw=1.5,
                        label=lab)
                _median_mark(ax, float(np.median(v)), s["color"])
            n += 1
    if n:
        for j in legend_axes:
            ax = axes[j]
            h, l_ = ax.get_legend_handles_labels()
            base = Patch(color=base_color, alpha=0.85, label=base_label)
            ax.legend(handles=[base] + h, **(legend_kw or {}))
        if figure_legend:
            h, l_ = axes[0].get_legend_handles_labels()
            base = Patch(color=base_color, alpha=0.85, label=base_label)
            axes[0].figure.legend(handles=[base] + h, loc="upper center",
                                  bbox_to_anchor=(0.5, 0.0), ncol=len(h) + 1,
                                  **(legend_kw or {}))
    return n


@_never_raises(0)
def overlay_curves(ax, runs, curve, *, lw=1.4) -> int:
    """Each added run's last-row noise as ``curve(psd_row, galfor_phys_row)``
    per cold walker, in its colour and line style, labelled once per run."""
    n = 0
    for A in _drawable_noise(runs):
        with _guard("overlay_curves", A):
            nw = A["psd"].shape[1]
            xy = [curve(A["psd"][-1, w], A["gal_phys"][-1, w]) for w in range(nw)]
            s = A["style"]
            for w, (x, y) in enumerate(xy):
                ax.plot(x, y, color=s["color"], ls=s["ls"], lw=lw, alpha=0.95, zorder=4,
                        label=(f"{A['label']}, row {A['noise_row']} ({nw} walkers)"
                               if w == 0 else None))
            n += 1
    return n


def galfor_in_basis(gal_phys, log_sampling: bool):
    """Physical galfor parameters -> a run's sampling basis (log10 on the
    ``GALFOR_LOG_PARAMS`` columns when it samples them in log): the inverse of
    ``erebor.noise.galfor_params_to_physical``, so an added run's foreground
    histograms land in the BASE run's basis."""
    from lisatools.globalfit.stock.erebor.noise import GALFOR_BASIS, GALFOR_LOG_PARAMS

    out = np.array(gal_phys, dtype=float, copy=True)
    if log_sampling:
        idx = [GALFOR_BASIS.index(p) for p in GALFOR_LOG_PARAMS]
        out[..., idx] = np.log10(out[..., idx])
    return out


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------
def _e(x) -> str:
    return _html.escape(str(x))


@_never_raises("")
def caption_html(runs, kind: str, *, base_label: str, base_row: int,
                 thresh: float = 0.8) -> str:
    """The sentence each overlay figure's caption carries: which runs are
    drawn, at which stored row each added run was read, and which requested
    runs are NOT drawn and why. ``""`` when no run was requested (the page is
    then exactly as before). ``kind`` is ``"recovery"`` or ``"noise"``."""
    if not runs:
        return ""
    shown, gone = [], []
    for A in runs:
        s = A["style"]
        if kind == "recovery":
            R = A.get("rec")
            if R is None:
                gone.append(f"<b>{_e(A['label'])}</b> ({_e(A['skip'] or A['rec_skip'])})")
                continue
            crit = (f"phase-max overlap &ge; {thresh:.2f}" if R.get("mm") is not None
                    else "2-bin f<sub>0</sub> proxy only: waveform overlaps unavailable")
            orb = "" if str(R.get("truth_orbits", "")).startswith("mojito") else (
                f"; &#9888; its truth set is on the {_e(R.get('truth_orbits'))} ephemeris")
            shown.append(
                f"<b>{_e(A['label'])}</b> ({s['word']}): its stored row {A['row']} (its last) of "
                f"<code>{_e(A['store_name'])}</code>, max-lnL walker {R['walker']}, matched "
                f"against its own <code>{_e(R['truth'])}</code> ({R['ndet']:,} detectable in "
                f"{R['band'][0] * 1e3:.4g}&ndash;{R['band'][1] * 1e3:.4g} mHz, Tobs "
                f"{A['tobs'] / 86400.0:.0f} d; {crit}{orb})")
        else:
            if A["skip"] is not None:
                gone.append(f"<b>{_e(A['label'])}</b> ({_e(A['skip'])})")
                continue
            shown.append(
                f"<b>{_e(A['label'])}</b> ({s['word']}): its last written row "
                f"{A['noise_row']} of <code>{_e(A['store_name'])}</code> for the curves (the "
                f"histograms pool its last {A['psd'].shape[0]} rows, as this run's do), all "
                f"{A['psd'].shape[1]} cold walkers, its own parameters")
    out = (f" <em>Overlaid runs:</em> <b>{_e(base_label)}</b> (this run, as drawn, row "
           f"{base_row})")
    if shown:
        out += "; " + "; ".join(shown)
    out += "."
    if kind == "noise" and shown:
        out += (" Curves and histograms only; the injected references are this run's. The "
                "fitted tanh foreground carries no Tobs of its own.")
    if gone:
        out += " <em>Not overlaid:</em> " + "; ".join(gone) + "."
    return out
