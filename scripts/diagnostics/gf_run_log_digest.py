"""Digest globalfit_run.log + gpu_util CSVs for the 3-mo production run."""
import re
import sys
from datetime import datetime

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else None
LOG = f"{RUN}/gf_prod_3mo_artifacts/globalfit_run.log"

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+) - (\S+) - (\w+) - (.*)$")

def parse_ts(s, ms):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp() + int(ms) / 1e3

events = []
for line in open(LOG, errors="replace"):
    m = TS.match(line)
    if m:
        events.append((parse_ts(m.group(1), m.group(2)), m.group(3),
                       m.group(4), m.group(5)))

# attempt boundaries: 'Multiple GPUs detected' warnings ~ startup
starts = [t for t, mod, lvl, msg in events
          if "Multiple GPUs detected" in msg and "analysiscontainer" in mod]
print("attempt starts:", [datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")
                          for t in starts])

# last attempt slice
t_last = starts[-1]
ev = [e for e in events if e[0] >= t_last - 60]
print(f"\n=== LAST ATTEMPT ({datetime.fromtimestamp(t_last)}) "
      f"-> {datetime.fromtimestamp(ev[-1][0])} "
      f"({(ev[-1][0]-t_last)/3600:.2f} h of log) ===")

# ---- per-move propose boundaries: 'buffer lifecycle' = propose exit --------
lif = [(t, msg) for t, _, _, msg in ev if "buffer lifecycle" in msg]
moves = {}
for t, msg in lif:
    mv = msg.split(":")[0]
    moves.setdefault(mv, []).append(t)
for mv, ts in moves.items():
    dt = np.diff(ts)
    print(f"propose cadence {mv}: n={len(ts)}, median gap "
          f"{np.median(dt) if len(dt) else float('nan'):.1f} s")

# ---- rj band units: 'band unit complete after N pick rounds (M cells)' -----
unit_re = re.compile(r"(\w+): band unit complete after (\d+) pick rounds \((\d+) cells\)")
units = []
for t, _, _, msg in ev:
    m = unit_re.search(msg)
    if m:
        units.append((t, m.group(1), int(m.group(2)), int(m.group(3))))
rj_units = [(t, r, c) for t, mv, r, c in units if "rj" in mv]
if rj_units:
    ts_u = np.array([u[0] for u in rj_units])
    walls = np.diff(np.concatenate(([t_last], ts_u)))
    print(f"\nrj units: {len(rj_units)}; cells/unit "
          f"{[u[2] for u in rj_units[:8]]}...; pick rounds "
          f"{[u[1] for u in rj_units[:8]]}...")
    print(f"rj unit walls [s]: {np.round(walls[:10], 1)}")
    print(f"  median {np.median(walls):.1f}, total {walls.sum()/3600:.2f} h")

# ---- at-cap skip lines -----------------------------------------------------
cap = [msg for t, _, _, msg in ev if "at-cap skip" in msg]
if cap:
    print(f"\nat-cap skip lines: {len(cap)}; last: ...{cap[-1][-130:]}")

# ---- in-model flush / grouped lines ---------------------------------------
for key in ("flush", "grouped", "pool", "polish"):
    hits = [msg for t, _, _, msg in ev
            if key in msg.lower() and "GPU pool" not in msg]
    if hits:
        print(f"\n'{key}' lines: {len(hits)}; e.g. {hits[len(hits)//2][:150]}")

# ---- leaves growth (gb move: after-proposal cold-chain leaves) -------------
leaves = []
for t, _, _, msg in ev:
    if "active leaves in cold chain after proposal" in msg:
        arr = re.findall(r"\d+", msg.split("[")[-1])
        if arr:
            leaves.append((t, int(np.mean([int(a) for a in arr]))))
if leaves:
    lv = [(datetime.fromtimestamp(t).strftime("%H:%M"), n) for t, n in leaves]
    print(f"\nleaves (mean over walkers) samples: {lv[::max(1,len(lv)//10)]}")

# ---- memory series from buffer-build lines ---------------------------------
mem_re = re.compile(
    r"GPU pool used ([\d.]+) / total ([\d.]+) GB; device used/total GB: "
    r"dev0 ([\d.]+)/[\d.]+, dev1 ([\d.]+)")
mems = []
for t, _, _, msg in ev:
    m = mem_re.search(msg)
    if m:
        mems.append((t, *[float(g) for g in m.groups()]))
if mems:
    a = np.array(mems)
    print(f"\nmemory series ({len(a)} pts): pool used "
          f"{a[:, 1].min():.1f}->{a[:, 1].max():.1f} GB; dev0 "
          f"{a[:, 3].min():.1f}->{a[:, 3].max():.1f} GB; dev1 "
          f"{a[:, 4].min():.1f}->{a[:, 4].max():.1f} GB")
    np.save(f"{RUN}/../mem_series_last.npy", a)

# ---- host RSS from SubBandBuffer lines --------------------------------------
rss = [(t, float(m.group(1))) for t, _, _, msg in ev
       if (m := re.search(r"host maxRSS ([\d.]+) GB", msg))]
if rss:
    r = np.array(rss)
    print(f"host maxRSS: {r[:,1].min():.1f} -> {r[:,1].max():.1f} GB")

# ---- fstat fit + [SAVE] + warnings ------------------------------------------
for t, _, lvl, msg in ev:
    if "grid fit epoch" in msg or "[SAVE]" in msg or "multi-device scorer" in msg:
        print(f"  {datetime.fromtimestamp(t).strftime('%H:%M:%S')} {msg[:150]}")
warns = {}
for t, _, lvl, msg in ev:
    if lvl in ("WARNING", "ERROR"):
        key = msg[:60]
        warns[key] = warns.get(key, 0) + 1
print("\nwarnings/errors (last attempt):")
for k, v in sorted(warns.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v:4d}x {k}")
