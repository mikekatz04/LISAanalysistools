"""Harvest campaign metrics from run logs.

Understands the parseable line families the global fit already emits (or that
campaign runners print):

- ``[RESULT] key=value key=value ...``            (runners, validation scripts)
- ``[GF_TIMING] stage=.. move=.. it=.. wall_s=..`` (GFCombineMove instrumentation)
- ``[GB_TIMING <move>] total=..s ...``             (gbspecialstretch _ProposeTimer)
- ``SubBandBuffer: <N> cells x ... ~ <MB> MB``     (buffer geometry/telemetry)
- unittest tails: ``Ran N tests`` / ``OK`` / ``FAILED (failures=x, errors=y)``

``harvest(text) -> dict`` returns a flat metrics dict; keys are stable and are
what gate criteria in ``gates.py`` refer to.
"""

from __future__ import annotations

import math
import re

_RESULT_RE = re.compile(r"^\s*\[RESULT\]\s+(.*)$", re.M)
_GF_TIMING_RE = re.compile(
    r"^\s*\[GF_TIMING\]\s+stage=(?P<stage>\S+)\s+move=(?P<move>\S+)\s+it=(?P<it>\d+)"
    r"\s+wall_s=(?P<wall>[-\d.eE+]+)"
    r"(?:\s+rss_mb=(?P<rss>[-\d.eE+]+))?"
    r"(?:\s+d_rss_mb=(?P<drss>[-\d.eE+]+))?"
    r"(?:\s+gpu_used_mb=(?P<gused>[-\d.eE+]+))?"
    r"(?:\s+gpu_pool_mb=(?P<gpool>[-\d.eE+]+))?",
    re.M,
)
_GB_TIMING_RE = re.compile(
    r"^\s*\[GB_TIMING\s+(?P<move>[^\]]+)\]\s+total=(?P<total>[\d.eE+-]+)s", re.M
)
_SUBBAND_RE = re.compile(
    r"SubBandBuffer:\s+(?P<cells>\d+)\s+cells.*?~\s*(?P<mb>[\d.]+)\s*MB", re.S
)
_GPU_POOL_RE = re.compile(
    r"GPU pool used\s+(?P<used>[\d.]+)\s*/\s*total\s+(?P<total>[\d.]+)\s*GB"
)
_MAXRSS_RE = re.compile(r"host maxRSS\s+(?P<rss>[\d.]+)\s*GB")
_UNITTEST_RAN_RE = re.compile(r"^Ran\s+(\d+)\s+tests?", re.M)
_UNITTEST_FAIL_RE = re.compile(
    r"^FAILED\s*\((?:failures=(\d+))?,?\s*(?:errors=(\d+))?\)", re.M
)
_UNITTEST_OK_RE = re.compile(r"^OK(?:\s|$)", re.M)
# scripts/validation null-check result lines, e.g.
# [RESULT] branch=mbh ... rr=1.2e-4 dd=3.4e2 ... rr/dd=3.5e-7 ...
_KV_RE = re.compile(r"(\S+?)=([^\s]+)")


def _num(s):
    try:
        v = float(s)
        return int(v) if v.is_integer() and "e" not in s.lower() and "." not in s else v
    except (TypeError, ValueError):
        return s


def harvest(text: str) -> dict:
    m: dict = {}

    # ---- [RESULT] key=value lines -------------------------------------
    result_rows = []
    for line in _RESULT_RE.findall(text):
        row = {k: _num(v) for k, v in _KV_RE.findall(line)}
        result_rows.append(row)
        for k, v in row.items():
            m[f"result_{k}"] = v  # last one wins
    if result_rows:
        m["result_rows"] = result_rows

    # per-class null-check aggregation: [RESULT] ... class=MBHB ... rr/dd=x
    per_class: dict = {}
    for row in result_rows:
        cls = row.get("class") or row.get("branch")
        val = row.get("rr_over_dd", row.get("rr/dd", row.get("rr_dd")))
        mmv = row.get("mismatch", row.get("mm"))
        if cls is None:
            continue
        cls = str(cls).upper()
        if isinstance(val, (int, float)):
            per_class.setdefault(cls, []).append(float(val))
        if isinstance(mmv, (int, float)):
            per_class.setdefault(f"{cls}_MM", []).append(float(mmv))
    for cls, vals in per_class.items():
        if cls.endswith("_MM"):
            m[f"{cls[:-3].lower()}_mismatch_max"] = max(vals)
        else:
            m[f"null_rr_dd_{cls}_max"] = max(vals)

    # ---- [GF_TIMING] ---------------------------------------------------
    totals, per_move, rss_seen, pool_seen = [], {}, [], []
    for g in _GF_TIMING_RE.finditer(text):
        wall = float(g.group("wall"))
        move = g.group("move")
        if move == "__total__":
            totals.append(wall)
        else:
            per_move.setdefault(move, []).append(wall)
        if g.group("rss"):
            rss_seen.append(float(g.group("rss")))
        if g.group("gpool") and float(g.group("gpool")) >= 0:
            pool_seen.append(float(g.group("gpool")))
    if totals:
        m["s_per_it"] = sum(totals) / len(totals)
        m["s_per_it_max"] = max(totals)
        m["iterations_timed"] = len(totals)
    if per_move:
        m["timed_moves"] = len(per_move)
        m["move_wall_s_mean"] = {
            k: sum(v) / len(v) for k, v in sorted(per_move.items())
        }
    if rss_seen:
        m["peak_rss_mb"] = max(rss_seen)
    if pool_seen:
        m["peak_gpu_pool_mb"] = max(pool_seen)

    # ---- [GB_TIMING] ---------------------------------------------------
    gb_tot: dict = {}
    for g in _GB_TIMING_RE.finditer(text):
        gb_tot.setdefault(g.group("move").strip(), []).append(float(g.group("total")))
    if gb_tot:
        m["gb_timing_total_s_mean"] = {
            k: sum(v) / len(v) for k, v in sorted(gb_tot.items())
        }

    # ---- SubBandBuffer / pool / maxRSS telemetry ----------------------
    sb = [_num(x.group("mb")) for x in _SUBBAND_RE.finditer(text)]
    if sb:
        m["subband_buffer_mb_max"] = max(float(x) for x in sb)
    pools = [
        (float(x.group("used")), float(x.group("total")))
        for x in _GPU_POOL_RE.finditer(text)
    ]
    if pools:
        m["gpu_pool_used_gb_max"] = max(p[0] for p in pools)
        m["gpu_pool_total_gb_max"] = max(p[1] for p in pools)
    rss = [float(x.group("rss")) for x in _MAXRSS_RE.finditer(text)]
    if rss:
        m["host_maxrss_gb"] = max(rss)

    # ---- unittest ------------------------------------------------------
    ran = [int(x) for x in _UNITTEST_RAN_RE.findall(text)]
    if ran:
        m["tests_ran"] = sum(ran)
        fails = 0
        for f, e in _UNITTEST_FAIL_RE.findall(text):
            fails += int(f or 0) + int(e or 0)
        # any FAILED line without counts still counts as failure
        if fails == 0 and "FAILED" in text and not _UNITTEST_OK_RE.search(text):
            fails = 1
        m["tests_failed"] = fails

    # ---- derived convenience flags ------------------------------------
    lls = [
        row.get("ll_max")
        for row in result_rows
        if isinstance(row.get("ll_max"), (int, float))
    ]
    if lls:
        m["ll_max_last"] = lls[-1]
        m["ll_finite"] = int(all(math.isfinite(v) and v > -1e290 for v in lls))

    # ---- bare [RESULT] keys -------------------------------------------
    # Gate criteria reference bare names (``variants_built``, ``debug_pngs``);
    # expose every RESULT key unprefixed too — last line wins (matching the
    # ``result_<k>`` convention) — without clobbering aggregates computed
    # above (those win via setdefault-after).
    last_kv: dict = {}
    for row in result_rows:
        last_kv.update(row)
    for k, v in last_kv.items():
        m.setdefault(k, v)
    return m


_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}


def evaluate(criteria, metrics: dict):
    """Evaluate criterion dicts against harvested metrics.

    Returns (all_quantitative_pass, unmet_list, manual_list). A missing metric
    counts as unmet (the run did not produce the evidence the gate demands).
    """
    unmet, manual = [], []
    for c in criteria:
        if "manual" in c:
            manual.append(c["manual"])
            continue
        key, op, want = c["metric"], c["op"], c["value"]
        have = metrics.get(key)
        if have is None or not isinstance(have, (int, float)):
            unmet.append(f"{key} missing (wanted {op} {want})")
        elif not _OPS[op](have, want):
            unmet.append(f"{key}={have!r} fails {op} {want!r}")
    return (not unmet), unmet, manual
