"""Render the campaign ledger into a single self-contained HTML dashboard.

Stdlib only.  The page must satisfy a strict CSP when published as an
artifact: no external requests, so proof plots are embedded as base64 data
URIs and all CSS is inline.  Status is never color alone — every state ships
an icon + text badge (colors from the validated status palette; chrome tokens
themed for light and dark).
"""

from __future__ import annotations

import base64
import html
import json
import os

STATUS = {
    # state -> (icon, label, css class)
    "green": ("✓", "green — verified", "st-green"),
    "running": ("●", "running…", "st-running"),
    "yellow": ("◌", "yellow — unverified", "st-yellow"),
    "red": ("✗", "red — broken", "st-red"),
    "pending": ("⋯", "pending", "st-pending"),
}
_COUNT_ORDER = ("green", "running", "yellow", "red", "pending")

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --warn: #fab219; --crit: #d03b3b; --series-1: #2a78d6;
  --running: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --running: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --running: #3987e5;
}
* { box-sizing: border-box; }
body { background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 20px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--ink-2); margin-bottom: 14px; }
.counts { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.count { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 12px; }
.count b { font-size: 18px; }
.tiers { display: flex; gap: 14px; align-items: flex-start;
  overflow-x: auto; padding-bottom: 8px; }
.tier { min-width: 250px; max-width: 300px; flex: 1 0 250px; }
.tier h2 { font-size: 13px; color: var(--ink-2); text-transform: uppercase;
  letter-spacing: 0.04em; margin: 0 0 8px; }
.card { position: relative; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px; margin-bottom: 10px; }
/* Circular status indicator, sitting in the card's clear top-right corner.
   A 2px surface ring lifts it off the card; a faint outer hairline keeps a
   pending (muted) dot visible on both themes. */
.status-dot { position: absolute; top: 11px; right: 11px;
  width: 11px; height: 11px; border-radius: 50%;
  box-shadow: 0 0 0 2px var(--surface), 0 0 0 3px var(--border); }
.hdr { display: flex; flex-wrap: wrap; gap: 5px; align-items: center;
  padding-right: 16px; }
.st-green  .status-dot { background: var(--good); }
.st-yellow .status-dot { background: var(--warn); }
.st-red    .status-dot { background: var(--crit); }
.st-pending .status-dot { background: var(--muted); }
.st-running .status-dot { background: var(--running);
  animation: pulse 1.3s ease-in-out infinite; }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 2px var(--surface), 0 0 0 3px var(--running); }
  50%      { box-shadow: 0 0 0 2px var(--surface),
             0 0 0 6px color-mix(in srgb, var(--running) 30%, transparent); }
}
@media (prefers-reduced-motion: reduce) {
  .st-running .status-dot { animation: none; }
}
.badge { display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 600; border-radius: 999px;
  padding: 1px 8px; border: 1px solid var(--border); color: var(--ink); }
.badge .ic { font-size: 12px; }
.st-green  .badge .ic { color: var(--good); }
.st-running .badge .ic { color: var(--running); }
.st-yellow .badge .ic { color: var(--warn); }
.st-red    .badge .ic { color: var(--crit); }
.st-pending .badge .ic { color: var(--muted); }
.gid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: var(--muted); }
.title { font-weight: 600; margin: 2px 0; }
.obj { color: var(--ink-2); font-size: 12.5px; margin-bottom: 6px; }
.where { font-size: 10.5px; color: var(--muted); border: 1px solid var(--grid);
  border-radius: 4px; padding: 0 4px; margin-left: 6px; }
dl.metrics { display: grid; grid-template-columns: auto 1fr; gap: 1px 10px;
  margin: 6px 0; font-size: 12px; }
dl.metrics dt { color: var(--muted); }
dl.metrics dd { margin: 0; font-variant-numeric: tabular-nums; }
.deps { font-size: 11px; color: var(--muted); }
.note { font-size: 11.5px; color: var(--ink-2); margin-top: 4px; }
details { margin-top: 6px; }
summary { cursor: pointer; font-size: 12px; color: var(--ink-2); }
.proof img { max-width: 100%; border: 1px solid var(--grid);
  border-radius: 6px; margin-top: 6px; display: block; }
.ev { font-size: 11px; }
.ev code { color: var(--ink-2); }
table.moves { border-collapse: collapse; font-size: 11.5px; margin-top: 4px;
  width: 100%; }
table.moves td, table.moves th { padding: 1px 6px; text-align: right;
  font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--grid); }
table.moves th { color: var(--muted); font-weight: 500; text-align: right; }
table.moves td:first-child, table.moves th:first-child { text-align: left;
  font-family: ui-monospace, Menlo, monospace; }
.spark { display: block; margin-top: 4px; }
.foot { color: var(--muted); font-size: 11.5px; margin-top: 16px; }
"""


def _spark(vals, w=140, h=26):
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = " ".join(
        f"{i * w / (len(vals) - 1):.1f},{h - 3 - (v - lo) / rng * (h - 6):.1f}"
        for i, v in enumerate(vals)
    )
    return (
        f'<svg class="spark" width="{w}" height="{h}" role="img" '
        f'aria-label="history {lo:g} to {hi:g}">'
        f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" '
        f'stroke-width="2" stroke-linecap="round"/></svg>'
    )


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return html.escape(str(v))


_METRIC_ORDER = (
    "s_per_it", "peak_rss_mb", "peak_gpu_pool_mb", "host_maxrss_gb",
    "subband_buffer_mb_max", "tests_ran", "tests_failed", "iterations_timed",
    "timed_moves", "ll_max_last",
)


def _card(gate, entry, base_dir):
    ic, label, cls = STATUS[entry["state"]]
    parts = [f'<div class="card {cls}">']
    # Circular status indicator in the clear top-right corner (title attr keeps
    # it labeled for hover / screen readers; the badge below carries the word).
    parts.append(f'<span class="status-dot" title="{html.escape(label)}"></span>')
    parts.append(
        '<div class="hdr">'
        f'<span class="badge"><span class="ic">{ic}</span>{html.escape(label)}</span>'
        f'<span class="where">{html.escape(gate.branch)}</span>'
        f'<span class="where">{html.escape(gate.where)}</span>'
        '</div>'
    )
    parts.append(f'<div class="gid">{gate.id}</div>')
    parts.append(f'<div class="title">{html.escape(gate.title)}</div>')
    parts.append(f'<div class="obj">{html.escape(gate.objective)}</div>')
    if gate.depends_on:
        parts.append(
            f'<div class="deps">needs: {html.escape(", ".join(gate.depends_on))}</div>'
        )

    metrics = entry.get("metrics", {})
    shown = [(k, metrics[k]) for k in _METRIC_ORDER if k in metrics]
    if shown:
        parts.append('<dl class="metrics">')
        for k, v in shown:
            parts.append(f"<dt>{html.escape(k)}</dt><dd>{_fmt(v)}</dd>")
        parts.append("</dl>")

    mv = metrics.get("move_wall_s_mean")
    if isinstance(mv, dict) and mv:
        rows = sorted(mv.items(), key=lambda kv: -kv[1])[:8]
        parts.append('<details><summary>per-move wall time</summary>'
                     '<table class="moves"><tr><th>move</th><th>s/it</th></tr>')
        for name, s in rows:
            parts.append(f"<tr><td>{html.escape(name)}</td><td>{s:.3f}</td></tr>")
        parts.append("</table></details>")

    # sparkline from history of s_per_it if present in history notes-free form
    hist_vals = [
        h.get("s_per_it") for h in entry.get("history", [])
        if isinstance(h.get("s_per_it"), (int, float))
    ]
    if hist_vals:
        parts.append(_spark(hist_vals))

    # proof plots (embedded) for yellow/green
    if entry["state"] in ("green", "yellow"):
        imgs = []
        for rel in entry.get("evidence", []):
            if not rel.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = "jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
            imgs.append(
                f'<img src="data:image/{ext};base64,{b64}" '
                f'alt="{html.escape(os.path.basename(path))}">'
            )
        if imgs:
            parts.append(
                f"<details open><summary>proof ({len(imgs)})</summary>"
                f'<div class="proof">{"".join(imgs)}</div></details>'
            )

    evs = [e for e in entry.get("evidence", []) if e.endswith(".log")]
    if evs:
        items = "".join(f"<li><code>{html.escape(e)}</code></li>" for e in evs)
        parts.append(f'<details><summary>evidence logs</summary>'
                     f'<ul class="ev">{items}</ul></details>')

    hist = entry.get("history", [])
    if hist:
        last = hist[-1]
        parts.append(
            f'<div class="note">{html.escape(last["ts"])} — '
            f'{html.escape(last["note"][:160])}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def render(gates, ledger, base_dir, out_path=None):
    out_path = out_path or os.path.join(base_dir, "artifacts", "dashboard.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    entries = {g.id: ledger["gates"].get(g.id, {"state": "pending", "metrics": {},
                                               "evidence": [], "history": []})
               for g in gates}
    counts = {s: 0 for s in _COUNT_ORDER}
    for e in entries.values():
        counts[e["state"]] = counts.get(e["state"], 0) + 1

    tiers: dict = {}
    for g in gates:
        tiers.setdefault(g.tier, []).append(g)

    tier_titles = {
        0: "T0 · Foundation", 1: "T1 · Mojito ground truth",
        2: "T2 · Lite CPU sampling", 3: "T3 · Single GPU",
        4: "T4 · Heavy branches", 5: "T5 · Multi-GPU", 6: "T6 · Full scale",
    }

    cols = []
    for tier in sorted(tiers):
        cards = "".join(_card(g, entries[g.id], base_dir) for g in tiers[tier])
        cols.append(
            f'<div class="tier"><h2>{html.escape(tier_titles.get(tier, f"T{tier}"))}'
            f"</h2>{cards}</div>"
        )

    count_html = "".join(
        f'<div class="count"><b>{counts[s]}</b> '
        f'{STATUS[s][0]} {html.escape(s)}</div>'
        for s in _COUNT_ORDER
    )

    doc = f"""<title>LISA Global Fit — Testing Campaign</title>
<style>{_CSS}</style>
<h1>LISA Global Fit — Testing Campaign</h1>
<div class="sub">Chain of custody: mojito ground truth &rarr; all_sources, 2 years,
multi-GPU. Every check runs through the stock fit infrastructure.</div>
<div class="counts">{count_html}</div>
<div class="tiers">{"".join(cols)}</div>
<div class="foot">ledger updated {html.escape(str(ledger.get("updated")))} ·
states: &#10003; verified science · &#9676; runs, science unverified ·
&#10007; broken · &#8943; not yet run</div>
"""
    with open(out_path, "w") as f:
        f.write(doc)
    return out_path


if __name__ == "__main__":
    import gates as G

    here = os.path.dirname(os.path.abspath(__file__))
    lp = os.path.join(here, "ledger.json")
    with open(lp) as f:
        led = json.load(f)
    print(render(G.GATES, led, here))
