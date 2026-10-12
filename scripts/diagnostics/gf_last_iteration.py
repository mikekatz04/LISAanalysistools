"""A one-screen dashboard of where a global-fit run is RIGHT NOW: last saved row,
stage, the live pace (log age, last save, the leg in flight against its last
wall time), the stage-end clock (the per-(walker, band) RJ valve's active
pairs and their trend), the empty-band valve, the F-stat epochs, head-node
GPU utilization and memory, cold-chain lnL and leaf counts, the last rows, and
the stage's proposal cycle grouped by leg with each move's last wall time, the
last saved leg marked and the next proposal named.

    python scripts/diagnostics/gf_last_iteration.py /shared/data/global_fit_output/gf_prod_9mo_v9

    ================================================================================================
     gf_prod_9mo_testing.h5                  /shared/data/global_fit_output/gf_prod_9mo_v9
    ------------------------------------------------------------------------------------------------
     last saved row   90              saved after   in_model
     stage            gb_search_2     step 4 of 5 . started row 56, 21.8 h ago . 34 rows in
     done             gb_search_seed, source_search, gb_search_1
    ------------------------------------------------------------------------------------------------
     live        log written 4 s ago . last save 17 min ago . last full cycle 102.7 min (cycle 10)
     next row    leg 2/3 (rj_fstat_search .. in_model_fstat): 16.9 min in, 50.2 min last time
     rj valve    124 active of 1999 occupied pairs (93.8% shut; stage end at 0) . judged 1.3 h ago
                 streak p50 0 p90 2 of 3, tol 20 nats
                 active, last 8 judges: 377 > 313 > 286 > 232 > 195 > 183 > 136 > 124
     empty bands 647 of 688 eligible bands off (f > 10 mHz) . next revive check in 9 iteration(s)
     fstat       epoch 28, 14 min ago: 242 peaks . fit 3.2 min (stage A 2.4, 1085/1232 bands skipped)
                 peaks, last 8 epochs: 463 > 410 > 382 > 342 > 287 > 263 > 377 > 242
     gpu (head)  GPU0 util 84% (60 s 61%) . 41.3/93.6 GiB, job peak 49.7
                 GPU1 util 80% (60 s 65%) . 42.8/93.6 GiB, job peak 48.0   sample 4 s ago
    ------------------------------------------------------------------------------------------------
     cold lnL   row 90    max 158,033,409.7   mean 158,032,894.8   min 158,032,544.7   spread 865.0
                vs 89     max +57.3   mean +8.4   min -61.1
     cold leaves          gb  min 2656  mean 2674.0  max 2684    emri 8 . mbh 6 . sobbh 6 . vgb 55
    ------------------------------------------------------------------------------------------------
     last rows   row  saved after            lnL max          lnL min        gb leaves
                  86  in_model_removal       158,033,360.5    158,032,508.0  2646-2680
                  ...
    ------------------------------------------------------------------------------------------------
     stage cycle (gb_search_2.move_order)   * = a row lands after this leg          last wall
       leg 1/3                                                                       18.4 min
          1    vgb_pe                         10 s
          2    rj_warm_search                9.2 min
          3  * in_model                      9.1 min  <<< LAST SAVED (row 90)
       leg 2/3                                                                       50.2 min
          4    rj_fstat_search              41.0 min  <-- NEXT
       ...
    ================================================================================================

``--line`` prints the old one-line form (``90  stage=gb_search_2 (step 4 of 5;
...)  saved_after=in_model  store=...``); ``-v`` is accepted and means the
dashboard. ``--rows N`` sets the recent-rows table length (default 5);
``--no-color`` / ``--color`` override the tty detection.

Sources, all read-only and lock-free (safe while the saver writes): the live
store (newest ``*.h5`` in the folder that is not a running backup, a
pre-migration copy or a quarantined file; an ``.h5`` path works too); the
head's run log ``<run_dir>/*_artifacts/globalfit_run.log`` (or the tar's
``globalfit_run_tail.log``, else the newest job stdout), read from the end in
growing windows (32 MB, doubled up to 256 MB) until the window holds a full
timed cycle, 8 valve judges and 8 F-stat epochs or reaches the previous
stage's end; the launcher's nvidia-smi sampler ``<run_dir>/gpu_util_<job>.csv``
(newest; the HEAD node's GPUs only -- a 2-node layout's second node is not
sampled). Ages are anchored on the files' mtimes and the log's own clock, so a
tar read on a laptop in another time zone reads right (its "log written" age
is the snapshot's age). A dataset read that races the saver is reported as
unreadable on its own line, never as a crash; a missing log or CSV prints n/a.
"""

import glob
import json
import os
import re
import sys
import time
from datetime import datetime

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SKIP = ("backup", "CORRUPT", "pre_rerung", "pre_repair", "midit", ".stale")
WIDTH = 96
TAIL_BYTES = 32 * 1024 * 1024      # the first log window; doubled until the dashboard has its lines
TAIL_BYTES_MAX = 256 * 1024 * 1024
TREND = 8                          # valve judges / F-stat epochs in the trend lines
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# the head-log lines the dashboard reads, found with bytes.find (memchr speed,
# ~10x a per-line Python scan of the same window)
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


# --------------------------------------------------------------------------
# store access
# --------------------------------------------------------------------------
def find_store(path):
    if os.path.isfile(path):
        return path
    cands = [p for p in glob.glob(os.path.join(path, "*.h5"))
             if not any(s in os.path.basename(p) for s in SKIP)]
    if not cands:
        raise FileNotFoundError(f"no run store (*.h5) in {path}")
    return max(cands, key=os.path.getmtime)


def _attr(a, key, default=None):
    v = a.get(key, default)
    if v is None:
        return None
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:  # noqa: BLE001
        return v


def _str(raw):
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def current_stage(g):
    """The recipe step a resume would run: ``dict(name, order, total, start, done,
    move_order)``.

    ``None`` when the store carries no recipe group. ``name`` is None when
    every step is complete (the recipe has finished).
    """
    if "recipe" not in g:
        return None
    steps = []
    for name in g["recipe"]:
        a = g["recipe"][name].attrs
        mo = _attr(a, "move_order")
        try:
            mo = list(json.loads(_str(mo))) if mo is not None else None
        except Exception:  # noqa: BLE001
            mo = None
        steps.append(dict(name=name, order=int(_attr(a, "order num", 0) or 0),
                          status=bool(_attr(a, "status", False)),
                          start=_attr(a, "start_iteration"), move_order=mo))
    steps.sort(key=lambda s: s["order"])
    done = [s["name"] for s in steps if s["status"]]
    cur = next((s for s in steps if not s["status"]), None)
    return dict(name=None if cur is None else cur["name"],
                order=None if cur is None else cur["order"], total=len(steps),
                start=None if cur is None else cur["start"],
                move_order=None if cur is None else cur["move_order"], done=done)


def last_iteration(path, group="global_fit"):
    """``(iteration, store_path, saved_after, stage)``; ``saved_after`` / ``stage`` None when absent."""
    import h5py

    store = find_store(path)
    with h5py.File(store, "r") as f:
        g = f[group] if group in f else f
        it = int(g.attrs["iteration"])
        after = None
        if it > 0 and "saved_after" in g and g["saved_after"].shape[0] >= it:
            after = _str(g["saved_after"][it - 1])
        stage = current_stage(g)
    return it, store, after, stage


def _cold(arr):
    """Collapse eryn's leading (nsamplers, ntemps) axes of one stored row to the
    cold walkers: ``(nw,)`` for log_like, ``(nw, nleaves)`` for inds."""
    import numpy as np

    a = np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim > 2:                      # ntemps > 1 in the main group: cold rung
        a = a[0]
    return a


def read_rows(path, it, n_rows, group="global_fit"):
    """Per-row cold-chain summary for rows ``it-n_rows .. it-1``: a list of
    ``dict(row, after, ll (nw,), leaves {branch: (nw,)})``; a row that cannot
    be read carries ``error``."""
    import h5py
    import numpy as np

    out = []
    lo = max(0, it - n_rows)
    with h5py.File(path, "r") as f:
        g = f[group] if group in f else f
        branches = list(g["inds"].keys()) if "inds" in g else []
        for r in range(lo, it):
            rec = dict(row=r + 1, after=None, ll=None, leaves={})
            try:
                if "saved_after" in g and g["saved_after"].shape[0] > r:
                    rec["after"] = _str(g["saved_after"][r])
                rec["ll"] = np.asarray(_cold(g["log_like"][r]), dtype=float).ravel()
                for b in branches:
                    rec["leaves"][b] = _cold(g["inds"][b][r]).sum(axis=-1).ravel()
            except Exception as exc:  # noqa: BLE001 -- a read racing the saver
                rec["error"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# live sidecars: the head's run log and the nvidia-smi CSV
# --------------------------------------------------------------------------
def _tail_lines(path, nbytes):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size > nbytes:
            fh.seek(size - nbytes)
            fh.readline()                               # drop the torn first line
        data = fh.read()
    return data.decode("utf-8", "replace").splitlines()


def find_run_log(run_dir):
    """The head's run log, or (a short tar / a run dir without the artifacts
    log) the newest job stdout, which is a superset of it."""
    for name in ("globalfit_run.log", "globalfit_run_tail.log"):
        hits = glob.glob(os.path.join(run_dir, "*_artifacts", name))
        if hits:
            return max(hits, key=os.path.getmtime)
    hits = [p for p in glob.glob(os.path.join(run_dir, "slurm_stdout_*.log"))
            if not p.endswith(("_filtered.log", "_tail.log"))]
    if hits:
        return max(hits, key=os.path.getmtime)
    return None


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
    """The newest timestamp in the log (its own clock)."""
    for line in reversed(_tail_lines(path, 256 * 1024)):
        t = _ts(line)
        if t is not None:
            return t
    return None


def _follows(prev, rec):
    """``rec`` is the leg right after ``prev`` in the same stage's cycle."""
    if prev["stage"] != rec["stage"]:
        return False
    if rec["cycle"] == prev["cycle"]:
        return rec["leg"] == prev["leg"] + 1
    return prev["leg"] == prev["nlegs"] and rec["leg"] == 1 and rec["cycle"] == prev["cycle"] + 1


def _move_walls(fan, ran, t0):
    """Each move's wall seconds inside one leg, from the leg's ``[FANOUT]`` lines
    (logged when an op ENDS): a move runs from the previous move's last line
    (the leg start ``t0`` for the first) to its own last line. A FANOUT name
    the leg did not run is an inner move of a composite (the noise block's
    psd_pe / galfor_pe) and belongs to the next leg move that logs no FANOUT
    line of its own; a move with none at all (a skipped Gibbs) has no wall."""
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


def _parse_log(lines):
    """The dashboard's reading of the tagged head-log lines (in file order).

    Valve judges, empty-band status and F-stat epochs restart at a ``STAGE
    COMPLETE`` line (they belong to the stage that ended). A leg's wall is the
    time from the previous ``[LEG]`` line of the same stage's cycle; a resume
    line breaks the chain (the gap holds a restart)."""
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
    """``(cycle, total_s, walls)`` of the newest cycle of ``stage`` whose every leg
    carries a wall, from one job (no resume inside it); None when there is none."""
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
    """The window holds what the dashboard shows, or it reached the previous stage."""
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


def read_live(run_dir):
    """What the head's run log says about the pace, the gates and the F-stat fit:
    :func:`_parse_log`'s dict plus ``log``, ``log_age_min`` (now minus the log's
    mtime), ``t_end`` (the log's newest timestamp, its own clock) and
    ``scanned_mb``; None without a log."""
    path = find_run_log(run_dir)
    if path is None:
        return None
    size = os.path.getsize(path)
    lines, lo, nbytes = [], size, TAIL_BYTES
    while True:
        new_lo = max(0, size - nbytes)
        chunk, lo = _scan(path, new_lo, lo)
        lines = chunk + lines
        out = _parse_log(lines)
        if new_lo == 0 or nbytes >= TAIL_BYTES_MAX or _enough(out):
            break
        nbytes *= 2
    out.update(log=path, log_age_min=(time.time() - os.path.getmtime(path)) / 60.0,
               t_end=_last_ts(path), scanned_mb=(size - lo) / 2.0 ** 20)
    return out


def read_gpu(run_dir, window_s=60.0):
    """The newest ``gpu_util_<job>.csv``: per GPU index the last sample
    ``dict(util, mem_used_gb, mem_total_gb, power_w, temp_c, util_avg, mem_peak_gb)``
    with ``util_avg`` over the trailing ``window_s`` seconds and ``mem_peak_gb``
    over the whole job; plus ``age_min`` (now minus the CSV's mtime) and the
    CSV's name. None without a CSV."""
    cands = glob.glob(os.path.join(run_dir, "gpu_util_*.csv"))
    if not cands:
        return None
    path = max(cands, key=os.path.getmtime)
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", "replace").splitlines()
    peak = {}
    for line in text:                                   # whole job: memory only, no time parse
        parts = line.split(",")
        if len(parts) < 9:
            continue
        try:
            idx, used = int(parts[1]), float(parts[5]) / 1024.0
        except ValueError:
            continue
        peak[idx] = max(peak.get(idx, 0.0), used)
    rows = []
    for line in text[-400:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            t = datetime.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S")
            rows.append(dict(t=t, idx=int(parts[1]), util=float(parts[3]),
                             used=float(parts[5]) / 1024.0, total=float(parts[6]) / 1024.0,
                             power=float(parts[7]), temp=float(parts[8])))
        except ValueError:
            continue
    if not rows:
        return None
    t_last = max(r["t"] for r in rows)
    recent = {}
    for r in rows:
        if (t_last - r["t"]).total_seconds() <= window_s:
            recent.setdefault(r["idx"], []).append(r)
    gpus = {}
    for idx, rs in recent.items():
        last = max(rs, key=lambda r: r["t"])
        gpus[idx] = dict(util=last["util"], util_avg=sum(r["util"] for r in rs) / len(rs),
                         mem_used_gb=last["used"], mem_total_gb=last["total"],
                         power_w=last["power"], temp_c=last["temp"],
                         mem_peak_gb=peak.get(idx, last["used"]))
    return dict(csv=path, age_min=(time.time() - os.path.getmtime(path)) / 60.0, gpus=gpus)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
class Style:
    def __init__(self, color):
        self.color = color

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s):
        return self._w("1", s)

    def dim(self, s):
        return self._w("2", s)

    def hi(self, s):          # the last saved leg
        return self._w("1;33", s)

    def nxt(self, s):         # the next proposal
        return self._w("1;36", s)

    def warn(self, s):
        return self._w("1;31", s)


def _fmt(x):
    return f"{x:,.1f}"


def _age(m):
    if m is None:
        return "n/a"
    if m < -1.0:
        # the file's mtime is AHEAD of this host's clock (NFS skew): the age is
        # meaningless, say so instead of a negative
        return f"{-m:.0f} min in the future (clock skew)"
    if m < 1.0:
        return f"{max(m, 0.0) * 60:.0f} s ago"
    if m < 120:
        return f"{m:.0f} min ago"
    return f"{m / 60:.1f} h ago"


def _dur(s):
    if s is None:
        return "-"
    if s < 100:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _rule(ch="-"):
    return ch * WIDTH


def _trend(vals):
    return " > ".join(str(v) for v in vals[-TREND:])


def describe_stage(it, stage, verbose):
    if stage is None:
        return "stage=none (no recipe group)"
    if stage["name"] is None:
        return f"stage=FINISHED (all {stage['total']} steps complete)"
    s = f"stage={stage['name']}"
    if not verbose:
        return s
    bits = [f"step {stage['order']} of {stage['total']}"]
    if stage["start"] is not None:
        bits.append(f"started at row {int(stage['start'])}, {it - int(stage['start'])} rows in")
    if stage["done"]:
        bits.append("done: " + ", ".join(stage["done"]))
    return s + " (" + "; ".join(bits) + ")"


def _leg_view(live, stage_name):
    """The current stage's legs: the newest record per leg index (with a wall
    when one exists), the leg each move belongs to, and the newest record."""
    recs = [r for r in (live or {}).get("legs", []) if r["stage"] == stage_name]
    newest, timed = {}, {}
    for r in recs:
        newest[r["leg"]] = r
        if r["wall"] is not None:
            timed[r["leg"]] = r
    leg_of = {m: k for k, r in sorted(newest.items()) for m in r["ran"]}
    return dict(timed=timed, leg_of=leg_of, last=recs[-1] if recs else None,
                nlegs=recs[-1]["nlegs"] if recs else None)


def _render_live(L, live, gpu, stage, legs, st):
    """The live block: pace, the leg in flight, the gates, the F-stat fit, gpu."""
    lab = lambda s: f" {s:<11} "        # noqa: E731
    pad = " " * 13
    if live is None:
        L.append(lab("live") + st.dim("n/a (no run log in the run dir)"))
    else:
        t_end, log_age = live["t_end"], live["log_age_min"]

        def ago(t):                      # an event on the log's clock -> minutes before now
            return None if t is None or t_end is None else log_age + _mins(t, t_end)

        bits = [f"log written {_age(log_age)}"]
        if live["saves"]:
            bits.append(f"last save {_age(ago(live['saves'][-1]))}")
        cyc = last_cycle(live["legs"], stage["name"]) if stage and stage["name"] else None
        if cyc is not None:
            bits.append(f"last full cycle {_dur(cyc[1])} (cycle {cyc[0]})")
        elif len(live["saves"]) > 1:
            s = live["saves"][-7:]
            c = [_mins(a, b) for a, b in zip(s[:-1], s[1:])]
            bits.append(f"row cadence: last {c[-1]:.0f} min, median of {len(c)} = "
                        f"{sorted(c)[len(c) // 2]:.0f} min")
        line = lab("live") + " . ".join(bits)
        L.append(st.warn(line) if log_age > 30 else line)
        if log_age > 30:
            L.append(pad + st.dim("quiet log: a snapshot or a stopped job"
                                  + ("; the leg time below is at its last line"
                                     if legs["last"] is not None else "")))
        # ---- the leg in flight against its last wall time ----------------
        last = legs["last"]
        if last is not None and t_end is not None:
            k = last["leg"] % last["nlegs"] + 1
            nxt = legs["timed"].get(k)
            ran = nxt["ran"] if nxt else [m for m, j in legs["leg_of"].items() if j == k]
            names = f" ({ran[0]} .. {ran[-1]})" if len(ran) > 1 else (f" ({ran[0]})" if ran else "")
            run_s = (t_end - last["t"]).total_seconds()
            s = f"leg {k}/{last['nlegs']}{names}: {_dur(run_s)} in"
            if nxt is not None:
                s += f", {_dur(nxt['wall'])} last time"
                if run_s > 1.3 * nxt["wall"] + 300:
                    s = st.warn(s + f" -- {_dur(run_s - nxt['wall'])} OVER")
            L.append(lab("next row") + s)
        # ---- the stage-end clock: the per-(walker, band) RJ valve --------
        if live["gates"]:
            g = live["gates"][-1]
            L.append(lab("rj valve") + f"{g['active']} active of {g['occupied']} occupied pairs "
                     f"({g['shut_pct']:.1f}% shut; stage end at 0) . judged {_age(ago(g['t']))}")
            bits = []
            if g["streak"]:
                bits.append(f"streak p50 {g['streak'][0]} p90 {g['streak'][1]} of {g['streak'][2]}")
            if g["tol"] is not None:
                bits.append(f"tol {g['tol']:g} nats")
            if bits:
                L.append(pad + ", ".join(bits))
            if len(live["gates"]) > 1:
                L.append(pad + f"active, last {min(TREND, len(live['gates']))} judges: "
                         + _trend([x["active"] for x in live["gates"]]))
        if live["empty"]:
            e = live["empty"]
            L.append(lab("empty bands") + f"{e['off']} of {e['eligible']} eligible bands off "
                     f"(f > {e['floor_mhz']:g} mHz) . next revive check in "
                     f"{max(e['every'] - e['since'], 0)} iteration(s)")
        # ---- F-stat epochs ------------------------------------------------
        if live["epochs"] or live["epoch_open"]:
            if live["epoch_open"] is not None:
                o = live["epoch_open"]
                s = f"epoch {o['epoch']} FITTING for {_dur((t_end - o['t']).total_seconds())}"
                if live["sweep"]:
                    w = live["sweep"]
                    s += f" . {w['name']} {w['pct']:.0f}% ETA {w['eta']}"
                L.append(lab("fstat") + s)
            if live["epochs"]:
                ep = live["epochs"][-1]
                s = (f"epoch {ep['epoch']}, {_age(ago(ep['t']))}: {ep['peaks']} peaks . "
                     f"fit {_dur(ep['wall_s'])}")
                if ep["stage_a"]:
                    a = ep["stage_a"]
                    s += f" (stage A {a['minutes']:.1f}, {a['skipped']}/{a['nbands']} bands skipped)"
                L.append((pad if live["epoch_open"] else lab("fstat")) + s)
                if len(live["epochs"]) > 1:
                    L.append(pad + f"peaks, last {min(TREND, len(live['epochs']))} epochs: "
                             + _trend([x["peaks"] for x in live["epochs"]]))
        if live["replica"]:
            L.append(lab("replica pe") + live["replica"][:WIDTH - 14])
    # ---- head-node GPUs -------------------------------------------------
    if gpu is None:
        L.append(lab("gpu (head)") + st.dim("n/a (no gpu_util_*.csv in the run dir)"))
        return
    stale = gpu["age_min"] is not None and gpu["age_min"] > 5
    tag = f"sample {_age(gpu['age_min'])}"
    for j, idx in enumerate(sorted(gpu["gpus"])):
        d = gpu["gpus"][idx]
        s = (f"GPU{idx} util {d['util']:.0f}% (60 s {d['util_avg']:.0f}%) . "
             f"{d['mem_used_gb']:.1f}/{d['mem_total_gb']:.1f} GiB, job peak {d['mem_peak_gb']:.1f}")
        if j == len(gpu["gpus"]) - 1:
            s += "   " + (st.warn(tag + " -- STALE") if stale else st.dim(tag))
        L.append((lab("gpu (head)") if j == 0 else pad) + s)


def render(it, store, after, stage, rows, live, gpu, st):
    L = []
    run_dir = os.path.dirname(os.path.abspath(store))
    name = os.path.basename(store)
    L.append(_rule("="))
    L.append(f" {st.bold(name)}" + " " * max(1, 39 - len(name)) + f"  {run_dir}")
    L.append(_rule())
    legs = _leg_view(live, stage["name"] if stage else None)
    # ---- where ----------------------------------------------------------
    L.append(f" {'last saved row':<16} {st.bold(str(it))}{' ' * max(1, 14 - len(str(it)))}  "
             f"saved after   {st.hi(after) if after else st.dim('(none recorded)')}")
    if stage is None:
        L.append(f" {'stage':<16} none (no recipe group)")
    elif stage["name"] is None:
        L.append(f" {'stage':<16} {st.bold('FINISHED')}  all {stage['total']} steps complete")
    else:
        bits = [f"step {stage['order']} of {stage['total']}"]
        sd = live["stage_done"] if live else None
        if stage["start"] is not None:
            s0 = int(stage["start"])
            when = ""
            # the logged end of the step right before this one is this one's start
            # (a source_search in between logs no STAGE COMPLETE of its own)
            if (sd is not None and stage["done"] and sd["stage"] == stage["done"][-1]
                    and live["t_end"] is not None):
                when = f", {_age(live['log_age_min'] + _mins(sd['t'], live['t_end']))}"
            bits.append(f"started row {s0}{when}")
            bits.append(f"{it - s0} rows in")
        L.append(f" {'stage':<16} {st.bold(stage['name'])}"
                 f"{' ' * max(1, 15 - len(stage['name']))} " + " . ".join(bits))
        if stage["done"]:
            L.append(f" {'done':<16} {', '.join(stage['done'])}")
        if sd is not None and sd["stage"] == stage["name"]:
            L.append(f" {'':<16} " + st.warn("the log says STAGE COMPLETE; the store's stamp "
                                             "is still pending"))
    L.append(_rule())
    _render_live(L, live, gpu, stage, legs, st)
    L.append(_rule())
    # ---- cold chain -----------------------------------------------------
    last = rows[-1] if rows else None
    prev = rows[-2] if len(rows) > 1 else None
    if last is None or last.get("error") or last["ll"] is None:
        L.append(f" {'cold lnL':<10} " + st.warn(
            f"row {it} unreadable" + (f" ({last['error']})" if last and last.get('error') else "")))
    else:
        ll = last["ll"]
        L.append(f" {'cold lnL':<10} row {it:<5} max {_fmt(ll.max()):>16}   mean {_fmt(ll.mean()):>16}   "
                 f"min {_fmt(ll.min()):>16}   spread {_fmt(ll.max() - ll.min())}")
        if prev is not None and not prev.get("error") and prev["ll"] is not None:
            p = prev["ll"]
            L.append(f" {'':<10} vs {prev['row']:<6} "
                     f"max {ll.max() - p.max():>+16,.1f}   mean {ll.mean() - p.mean():>+16,.1f}   "
                     f"min {ll.min() - p.min():>+16,.1f}")
        lv = last["leaves"]
        if lv:
            main = "gb" if "gb" in lv else sorted(lv)[0]
            v = lv[main]
            s = (f" {'cold leaves':<10} {'':<9} {main}  min {int(v.min()):<6} mean {v.mean():<8.1f} "
                 f"max {int(v.max()):<6}")
            others = []
            for b in sorted(lv):
                if b == main or b in ("psd", "galfor"):
                    continue
                w = lv[b]
                others.append(f"{b} {int(w.min())}" if w.min() == w.max()
                              else f"{b} {int(w.min())}-{int(w.max())}")
            if others:
                s += "   " + " . ".join(others)
            L.append(s)
    L.append(_rule())
    # ---- recent rows ----------------------------------------------------
    if rows:
        L.append(f" {'last rows':<10} {'row':>5}  {'saved after':<22} {'lnL max':>16} {'lnL min':>16}"
                 f"  {'gb leaves':<11}")
        for rec in rows:
            if rec.get("error") or rec["ll"] is None:
                L.append(f" {'':<10} {rec['row']:>5}  " + st.warn(f"unreadable ({rec.get('error', '?')})"))
                continue
            gb = rec["leaves"].get("gb")
            gbs = f"{int(gb.min())}-{int(gb.max())}" if gb is not None else ""
            line = (f" {'':<10} {rec['row']:>5}  {(rec['after'] or '-'):<22} "
                    f"{_fmt(rec['ll'].max()):>16} {_fmt(rec['ll'].min()):>16}  {gbs:<11}")
            L.append(st.bold(line) if rec is last else line)
        L.append(_rule())
    # ---- the stage's proposal cycle, grouped by leg ----------------------
    if stage is not None and stage["name"] is not None:
        mo = stage["move_order"]
        if not mo:
            L.append(f" {'stage cycle':<10} " + st.dim("(no move_order recorded for this stage)"))
        else:
            marks = {rec["after"] for rec in rows if rec.get("after")}
            marks |= {r["after"] for r in legs["timed"].values()}
            if after:
                marks.add(after)
            head = (f" stage cycle ({stage['name']}.move_order)   "
                    f"* = a row lands after this leg")
            L.append(head + ("last wall".rjust(WIDTH - len(head) - 1) if legs["timed"] else ""))
            i_last = mo.index(after) if after in mo else None
            i_next = (i_last + 1) % len(mo) if i_last is not None else None
            cur_leg = None
            for i, m in enumerate(mo):
                k = legs["leg_of"].get(m)
                if k is not None and k != cur_leg:
                    r = legs["timed"].get(k)
                    lh = f"   leg {k}/{legs['nlegs']}"
                    L.append(st.dim(lh + (_dur(r["wall"]).rjust(WIDTH - len(lh) - 1) if r else "")))
                    cur_leg = k
                r = legs["timed"].get(k)
                w = r["moves"].get(m) if r else None
                line = (f"   {'  ' if k else ''}{i + 1:>2}  {'*' if m in marks else ' '} {m:<24}"
                        f"{_dur(w) if w is not None else '':>9}")
                if i == i_last:
                    line = st.hi(line + f"  <<< LAST SAVED (row {it})")
                elif i == i_next:
                    line = st.nxt(line + "  <-- NEXT")
                L.append(line)
            if i_last is not None and i_next == 0:
                L.append(st.dim("       (the cycle wraps: the next proposal is its first move)"))
            elif after and i_last is None:
                L.append(st.warn(f"       saved_after={after!r} is not in this stage's move_order "
                                 "(the row was saved by the previous stage)"))
    L.append(_rule("="))
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    one_line = "--line" in argv
    color = None
    if "--no-color" in argv:
        color = False
    if "--color" in argv:
        color = True
    n_rows = 5
    if "--rows" in argv:
        i = argv.index("--rows")
        try:
            n_rows = max(1, int(argv[i + 1]))
        except (IndexError, ValueError):
            print("usage: --rows N")
            return 2
        del argv[i:i + 2]
    args = [a for a in argv if a not in ("-v", "--line", "--no-color", "--color")]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: gf_last_iteration.py <run_dir | store.h5> [--line] [--rows N] "
              "[--no-color | --color]")
        return 2
    it, store, after, stage = last_iteration(args[0])
    if one_line:
        print(f"{it}  {describe_stage(it, stage, True)}  saved_after={after}  "
              f"store={os.path.basename(store)}")
        return 0
    if color is None:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    run_dir = os.path.dirname(os.path.abspath(store))
    rows = read_rows(store, it, n_rows) if it > 0 else []
    try:
        live = read_live(run_dir)
    except Exception as exc:  # noqa: BLE001 -- a sidecar never takes the page down
        live = None
        print(f"(run log unreadable: {exc})", file=sys.stderr)
    try:
        gpu = read_gpu(run_dir)
    except Exception as exc:  # noqa: BLE001
        gpu = None
        print(f"(gpu csv unreadable: {exc})", file=sys.stderr)
    print(render(it, store, after, stage, rows, live, gpu, Style(color)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
