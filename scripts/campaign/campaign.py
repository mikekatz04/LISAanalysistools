#!/usr/bin/env python
"""Testing-campaign CLI: run gates, ingest cluster logs, track the ledger,
render the dashboard.  Stdlib only (PIL/matplotlib used opportunistically for
proof-plot downscaling).

  campaign.py list [--tier N] [--state S]
  campaign.py run <gate-id> [--check CHECK_ID] [--dry-run]
  campaign.py ingest <gate-id> <log> [<log> ...] [--confirm TEXT]...
  campaign.py set <gate-id> <state> --note TEXT [--evidence PATH]
  campaign.py batch <N>
  campaign.py render

State machine: pending -> red | yellow | green.  A gate goes green only when
every quantitative criterion passes AND every manual criterion has been
confirmed AND all depends_on gates are green (--force overrides, logged).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import gates as G  # noqa: E402
import parse as P  # noqa: E402

LEDGER = os.path.join(HERE, "ledger.json")
EVIDENCE = os.path.join(HERE, "evidence")
RAW_ROOT = os.path.join(REPO, "gf_output", "campaign")
LOCK = os.path.join(RAW_ROOT, ".running.lock")

# Laptop CPU budget (user directive): campaign work stays well below 50% of the
# machine. Enforced three ways — a lockfile so only ONE gate runs at a time,
# `nice` so an interactive session always outranks campaign work, and every
# thread pool pinned to 1 in the child env.
NICE = os.environ.get("CAMPAIGN_NICE", "10")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _acquire_lock(gate_id: str, force: bool = False):
    """Refuse to start a second gate while one is running."""
    os.makedirs(RAW_ROOT, exist_ok=True)
    if os.path.exists(LOCK):
        try:
            with open(LOCK) as f:
                held = json.load(f)
        except Exception:
            held = {}
        pid = int(held.get("pid", -1))
        if pid > 0 and _pid_alive(pid):
            if not force:
                sys.exit(
                    f"[campaign] {held.get('gate')} is already running (pid {pid}, "
                    f"since {held.get('started')}). One gate at a time — wait for it, "
                    f"or `kill {pid}` first."
                )
            print(f"[campaign] --force: taking the lock from pid {pid}")
        else:
            print(f"[campaign] clearing stale lock from {held.get('gate')} (pid {pid} gone)")
    with open(LOCK, "w") as f:
        json.dump({"gate": gate_id, "pid": os.getpid(), "started": _now()}, f)


def _release_lock():
    try:
        os.remove(LOCK)
    except FileNotFoundError:
        pass


def _cpu_snapshot() -> str:
    """Total CPU% of python processes, as a fraction of the whole machine."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pcpu,comm"], capture_output=True, text=True, timeout=10
        ).stdout
        total = sum(
            float(ln.split()[0])
            for ln in out.splitlines()[1:]
            if len(ln.split()) > 1 and "python" in ln.split()[1].lower()
        )
        ncpu = int(
            subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True,
                           text=True, timeout=10).stdout.strip()
            or 1
        )
        return f"{total:.0f}% of one core = {total / ncpu:.0f}% of machine"
    except Exception:
        return "unavailable"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            return json.load(f)
    return {"version": 1, "updated": None, "gates": {}}


def save_ledger(led):
    led["updated"] = _now()
    for g in G.GATES:  # keep every defined gate present
        led["gates"].setdefault(
            g.id,
            {"state": "pending", "metrics": {}, "evidence": [], "history": [],
             "confirmed": []},
        )
    # DAG restructures rename gates; park entries whose id no longer exists so
    # their evidence/history stay findable without cluttering the dashboard.
    for gid in [k for k in led["gates"] if k not in G.GATES_BY_ID]:
        led.setdefault("retired", {})[gid] = led["gates"].pop(gid)
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2, sort_keys=True)
        f.write("\n")


def _entry(led, gid):
    return led["gates"].setdefault(
        gid,
        {"state": "pending", "metrics": {}, "evidence": [], "history": [],
         "confirmed": []},
    )


def _transition(led, gid, state, note):
    e = _entry(led, gid)
    e["state"] = state
    e["history"].append({"ts": _now(), "state": state, "note": note})
    print(f"[campaign] {gid} -> {state}: {note}")


def _deps_green(led, gate):
    return [d for d in gate.depends_on if _entry(led, d)["state"] != "green"]


def _trim_log(text, keep_head=60, keep_tail=120):
    lines = text.splitlines()
    keep = []
    marked = [
        ln
        for ln in lines
        if ln.startswith(("[RESULT]", "[GF_TIMING]", "[GB_TIMING"))
        or "SubBandBuffer:" in ln
        or "GPU pool used" in ln
        or "host maxRSS" in ln
        or ln.startswith(("Ran ", "OK", "FAILED"))
    ]
    keep.extend(lines[:keep_head])
    if len(lines) > keep_head + keep_tail:
        keep.append(f"... [{len(lines) - keep_head - keep_tail} lines trimmed] ...")
    keep.extend(lines[-keep_tail:] if len(lines) > keep_head else [])
    body = "\n".join(keep)
    marks = "\n".join(marked)
    return f"{body}\n\n===== parseable lines =====\n{marks}\n"


def _downscale(src, dst, max_px=800):
    """Copy src PNG to dst, downscaled if possible; fall back to raw copy."""
    try:
        from PIL import Image

        im = Image.open(src)
        im.thumbnail((max_px, max_px))
        im.save(dst, optimize=True)
        return
    except Exception:
        pass
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        arr = mpimg.imread(src)
        h, w = arr.shape[:2]
        scale = min(1.0, max_px / max(h, w))
        fig = plt.figure(figsize=(w * scale / 100, h * scale / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr)
        ax.axis("off")
        fig.savefig(dst, dpi=100)
        plt.close(fig)
        return
    except Exception:
        import shutil

        shutil.copyfile(src, dst)


def _capture_proof_plots(gate):
    """Collect proof plots from the gate's raw dir into evidence/<gate>/."""
    raw_dir = os.path.join(RAW_ROOT, gate.id)
    out_dir = os.path.join(EVIDENCE, gate.id)
    captured = []
    for pat in gate.proof_plots:
        for src in sorted(glob.glob(os.path.join(raw_dir, pat)) +
                          glob.glob(os.path.join(raw_dir, "**", pat), recursive=True)):
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, os.path.basename(src))
            _downscale(src, dst)
            rel = os.path.relpath(dst, HERE)
            if rel not in captured:
                captured.append(rel)
    return captured


def _evaluate_gate(led, gate, metrics, confirmed):
    """Combine all checks' criteria; decide red/yellow/green."""
    all_criteria = [c for ch in gate.checks for c in ch.criteria]
    ok, unmet, manual = P.evaluate(all_criteria, metrics)
    missing_manual = [t for t in manual if t not in confirmed]
    if not ok:
        return "red", f"criteria unmet: {'; '.join(unmet[:4])}"
    if missing_manual:
        return "yellow", f"quantitative pass; awaiting manual: {missing_manual[0][:80]}..."
    deps = _deps_green(led, gate)
    if deps:
        return "yellow", f"criteria pass but deps not green: {deps}"
    return "green", "all criteria pass, manual confirmed, deps green"


def _record(led, gate, text, confirmed_new, evidence_paths, note_prefix):
    e = _entry(led, gate.id)
    metrics = P.harvest(text)
    # merge, preferring the new run's values
    e["metrics"].update({k: v for k, v in metrics.items() if not isinstance(v, list)})
    for t in confirmed_new:
        if t not in e["confirmed"]:
            e["confirmed"].append(t)
    proofs = _capture_proof_plots(gate)
    for p in evidence_paths + proofs:
        if p not in e["evidence"]:
            e["evidence"].append(p)
    state, why = _evaluate_gate(led, gate, e["metrics"], e["confirmed"])
    _transition(led, gate.id, state, f"{note_prefix}: {why}")
    return state


# --------------------------------------------------------------------- cmds
def cmd_list(args):
    led = load_ledger()
    for g in G.GATES:
        e = _entry(led, g.id)
        if args.tier is not None and g.tier != args.tier:
            continue
        if args.state and e["state"] != args.state:
            continue
        deps = f" needs:{','.join(g.depends_on)}" if g.depends_on else ""
        print(f"T{g.tier} [{g.where[0].upper()}] {e['state']:7s} {g.id:24s} {g.title}{deps}")


def cmd_run(args):
    led = load_ledger()
    gate = G.GATES_BY_ID[args.gate]
    if gate.where != "laptop":
        sys.exit(f"{gate.id} is a cluster gate — use `campaign.py batch` to emit "
                 f"its checklist and `campaign.py ingest` for the logs.")
    deps = _deps_green(led, gate)
    if deps and not args.force:
        print(f"[campaign] warning: deps not green: {deps} (running anyway, "
              f"gate cannot go green until they are)")
    if not args.dry_run:
        _acquire_lock(gate.id, force=args.force)
    raw_dir = os.path.join(RAW_ROOT, gate.id)
    os.makedirs(raw_dir, exist_ok=True)
    env = dict(os.environ)
    # Every check drops its proof plots into the gate's raw dir; runners and
    # match scripts honor CAMPAIGN_PLOT_DIR so proof_plots capture finds them.
    env["CAMPAIGN_PLOT_DIR"] = raw_dir
    # Laptop CPU budget: campaign work stays below 50% of the machine. Every
    # thread pool is pinned to 1 (macOS Accelerate reads VECLIB_*, not OMP_*)
    # and gates run strictly one at a time.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(var, "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")  # child flushes -> log streams live
    combined = []
    failed_check = None
    for ch in gate.checks:
        if args.check and ch.id != args.check:
            continue
        if not ch.command:
            print(f"[campaign] {gate.id}/{ch.id}: manual/aggregation check, skipping run")
            continue
        cmd = ch.command.format(py=sys.executable)
        # `nice` the whole check so an interactive session always wins the CPU.
        niced = f"nice -n {NICE} sh -c {shlex.quote(cmd)}"
        print(f"[campaign] {gate.id}/{ch.id}: {cmd}")
        if args.dry_run:
            continue
        print(f"[campaign]   cpu before: {_cpu_snapshot()}")
        t0 = time.time()
        log_path = os.path.join(raw_dir, f"{ch.id}.log")
        # Stream straight to the log file (line-buffered) so a long build is
        # observable live via `tail -f` and never lost if the wrapper dies.
        with open(log_path, "w", buffering=1) as lf:
            rc = subprocess.run(
                niced, shell=True, cwd=REPO, env=env,
                stdout=lf, stderr=subprocess.STDOUT, text=True,
            ).returncode
        with open(log_path) as lf:
            out = lf.read()

        class _P:
            returncode = rc
        proc = _P()
        combined.append(out)
        tail = "\n".join(out.splitlines()[-8:])
        print(tail)
        print(f"[campaign]   {ch.id} took {time.time() - t0:.0f}s; "
              f"cpu after: {_cpu_snapshot()}")
        if proc.returncode != 0:
            failed_check = (ch.id, proc.returncode)
            print(f"[campaign] {gate.id}/{ch.id} FAILED (exit {proc.returncode})")
            break
    if args.dry_run:
        return
    text = "\n".join(combined)
    trimmed = os.path.join(EVIDENCE, f"{gate.id}.log")
    os.makedirs(EVIDENCE, exist_ok=True)
    with open(trimmed, "w") as f:
        f.write(_trim_log(text))
    if failed_check:
        _entry(led, gate.id)["metrics"].update(
            {k: v for k, v in P.harvest(text).items() if not isinstance(v, list)}
        )
        _transition(led, gate.id, "red",
                    f"check {failed_check[0]} exited {failed_check[1]}")
    else:
        _record(led, gate, text, [], [os.path.relpath(trimmed, HERE)], "run")
    save_ledger(led)
    _release_lock()
    cmd_render(args)


def cmd_ingest(args):
    led = load_ledger()
    gate = G.GATES_BY_ID[args.gate]
    text = ""
    ev = []
    for lp in args.logs:
        with open(lp) as f:
            text += f.read() + "\n"
        dst = os.path.join(EVIDENCE, f"{gate.id}__{os.path.basename(lp)}")
        os.makedirs(EVIDENCE, exist_ok=True)
        with open(dst, "w") as f:
            f.write(_trim_log(text))
        ev.append(os.path.relpath(dst, HERE))
    _record(led, gate, text, args.confirm or [], ev, "ingest")
    save_ledger(led)
    cmd_render(args)


def cmd_set(args):
    led = load_ledger()
    if args.gate not in G.GATES_BY_ID:
        sys.exit(f"unknown gate {args.gate}")
    gate = G.GATES_BY_ID[args.gate]
    if args.state == "green":
        deps = _deps_green(led, gate)
        if deps and not args.force:
            sys.exit(f"cannot set green: deps not green {deps} (use --force)")
    e = _entry(led, args.gate)
    if args.evidence:
        e["evidence"].append(args.evidence)
    _transition(led, args.gate, args.state, f"manual: {args.note}")
    save_ledger(led)
    cmd_render(args)


def cmd_batch(args):
    gs = G.gates_for_batch(args.n)
    print(f"# Cluster batch #{args.n}\n")
    print("Run from the LAT repo root on the cluster (`/home/mlkatz/new_dev/"
          "LISAanalysistools`), your usual GPU env. For each check: run the "
          "command, save ALL stdout+stderr, hand the log back.\n")
    for g in gs:
        print(f"## {g.id} — {g.title}\n\n{g.objective}\n")
        if g.notes:
            print(f"_{g.notes}_\n")
        for ch in g.checks:
            print(f"### {g.id}/{ch.id}")
            if ch.command:
                print(f"```bash\n{ch.command.format(py='python')} 2>&1 | tee {g.id}__{ch.id}.log\n```")
            else:
                print("_(manual/aggregation check — see criteria)_")
            if ch.notes:
                print(f"- note: {ch.notes}")
            for c in ch.criteria:
                if "manual" in c:
                    print(f"- expect (manual): {c['manual']}")
                else:
                    print(f"- expect: {c['metric']} {c['op']} {c['value']}")
            print(f"- then: `python scripts/campaign/campaign.py ingest {g.id} "
                  f"{g.id}__{ch.id}.log`  (Claude runs this on ingest)")
            print()


def cmd_render(args):
    import dashboard

    led = load_ledger()
    save_ledger(led)
    out = dashboard.render(G.GATES, led, HERE)
    print(f"[campaign] dashboard -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="campaign.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--tier", type=int)
    p.add_argument("--state", choices=G.STATES)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("run")
    p.add_argument("gate", choices=list(G.GATES_BY_ID))
    p.add_argument("--check")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ingest")
    p.add_argument("gate", choices=list(G.GATES_BY_ID))
    p.add_argument("logs", nargs="+")
    p.add_argument("--confirm", action="append",
                   help="text of a manual criterion now confirmed (repeatable)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("set")
    p.add_argument("gate")
    p.add_argument("state", choices=G.STATES)
    p.add_argument("--note", required=True)
    p.add_argument("--evidence")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("batch")
    p.add_argument("n", type=int, choices=sorted(G.BATCH_TIERS))
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("render")
    p.set_defaults(fn=cmd_render)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
