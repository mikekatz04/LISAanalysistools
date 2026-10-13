#!/usr/bin/env python
"""Build the flagship launchers: launcher.sh.in + runs.py -> submit_gf_<run>.sh.

    python scripts/flagship/build.py                 # all six, in place
    python scripts/flagship/build.py 12mo            # one run, in place
    python scripts/flagship/build.py --check         # exit 1 if any committed
                                                     # launcher differs (CI / tests)
    python scripts/flagship/build.py --list          # the table, resolved
    python scripts/flagship/build.py 9mo --set GB_NTEMPS=12 --set NWALKERS=8 \\
        --out scripts/flagship/oneoff/submit_gf_9mo_ntemps12.sh
                                                     # LEVEL 3: a one-off launcher
                                                     # (never committed; oneoff/ is
                                                     # git-ignored)

Template language (deliberately tiny):

* ``{{NAME}}``                  a value from :func:`tokens` (unknown name = error);
* ``{{#if FLAG}}`` / ``{{#else}}`` / ``{{#endif}}`` on their own lines, nestable;
* ``{{RUN_EXTRA_EXPORTS}}``     the hook for ``Run.extra_exports``.

After the fill, ``Run.overrides`` (level 2) and ``--set`` (level 3) rewrite
export lines IN PLACE with :func:`set_export`: ``export K=v`` -> ``export
K=VALUE``, ``export K=${K:-v}`` -> ``export K=${K:-VALUE}`` (the ``${K-v}``
form likewise), trailing comment kept. A knob that is not an export line of
the built text is refused -- it belongs in ``extra_exports`` (level 2) or on
the launch line (level 3).
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runs as _runs  # noqa: E402

TEMPLATE = os.path.join(HERE, "launcher.sh.in")
_TOKEN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_EXPORT = re.compile(r"^(export\s+)([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def launcher_path(name: str) -> str:
    return os.path.join(HERE, f"submit_gf_{name}.sh")


# ---- the per-run values ----------------------------------------------------------
def _mbhb_note(run: "_runs.Run") -> str:
    ids = [int(i) for i in run.mbhb_ids.split(",")]
    tc = _runs.MBHB_TC_DAYS
    inside = [i for i in ids if tc[i] < run.days]
    past_end = [i for i in ids if tc[i] >= run.days]
    nxt = sorted((i for i in tc if i not in ids), key=tc.get)[:3]
    parts = ["t_c " + "/".join(f"{tc[i]:.1f}" for i in ids) + " d from the data start"
             " (mojito-lite catalogue anchored on the MBH session table 2026-10-03):"
             f" every catalogue MBHB merging inside {run.days} d + the"
             f" {_runs.MBH_MERGER_BUFFER_DAYS}-d MBH_MERGER_TIME_BUFFER below"]
    if past_end:
        parts.append("in-window inspiral only, merger past the data end: "
                     + ", ".join(f"{i} ({tc[i]:.1f} d)" for i in past_end))
    else:
        parts.append(f"all merge BEFORE the end; none lands in ({run.days}, "
                     f"{run.days + _runs.MBH_MERGER_BUFFER_DAYS}] d")
    if nxt:
        parts.append("next out: " + ", ".join(f"{i} ({tc[i]:.1f} d)" for i in nxt))
    else:
        parts.append("the whole catalogue")
    _ = inside
    return "; ".join(parts)


def _emri_note(run: "_runs.Run") -> str:
    pl = _runs.EMRI_PLUNGE_DAYS
    ins = [i for i in sorted(pl, key=pl.get) if pl[i] < run.days]
    if not ins:
        return "no EMRI plunges inside this window"
    return ("plunging INSIDE this window, on the direct template's plunge path: "
            + ", ".join(f"{i} ({pl[i]:.0f} d)" for i in ins))


def tokens(run: "_runs.Run") -> dict:
    parent = run.resolved_parent()
    tok = {
        "RUN": run.name, "SERIES": _runs.SERIES, "OUTPUT_ROOT": _runs.OUTPUT_ROOT,
        "DAYS": str(run.days), "TOBS": str(run.tobs), "NT": str(run.nt),
        "SIGHET_NT_LAYER": str(run.sighet_nt_layer),
        "EDGE_CROP_WAVELETS": str(run.edge_crop_wavelets),
        "GB_NLEAVES_MAX": str(run.gb_nleaves_max),
        "GB_N_SUBBANDS": str(run.gb_n_subbands),
        "GB_RJ_INMODEL_CHUNK": str(run.gb_rj_inmodel_chunk),
        "MBHB_IDS": run.mbhb_ids, "MBHB_IDS_NOTE": _mbhb_note(run),
        "EMRI_PLUNGE_NOTE": _emri_note(run),
        "JOB_NAME": run.resolved_job_name, "LOG_STEM": run.resolved_log_stem,
        "STORE_DIR": run.store_dir, "BASE_FILE_NAME": run.base_file_name,
        "NGPUS_DEFAULT": str(run.ngpus_default),
        "RUN_EXTRA_EXPORTS": "\n".join(run.extra_exports) if run.extra_exports
        else "# (none for this run)",
    }
    if parent is not None:
        tok.update({
            "PARENT_NAME": parent.name, "PARENT_DAYS": str(parent.days),
            "PARENT_TOBS": str(parent.tobs),
            "PARENT_SEED_STORE": parent.seed_store,
            "PARENT_STORE_DIRNAME": parent.store_dirname,
            "WARM_NPZ": parent.warm_npz,
            "PARENT_MBHB_IDS": parent.mbhb_ids, "PARENT_EMRI_IDS": parent.emri_ids,
            "PARENT_SOBHB_IDS": parent.sobhb_ids,
        })
    tok["HEADER"] = header(run, tok)
    return tok


def header(run: "_runs.Run", tok: dict) -> str:
    parent = run.resolved_parent()
    need = _runs.sighet_guard_min_crop(run.nt)
    L = []
    L.append("# ============================================================================")
    L.append(f"# FLAGSHIP global fit -- {run.name} (Tobs = {run.days} d). GENERATED FILE, DO NOT")
    L.append("# EDIT: python scripts/flagship/build.py rewrites it from")
    L.append("#   scripts/flagship/launcher.sh.in   (every run  -- level 1)")
    L.append(f"#   scripts/flagship/runs.py :: RUNS[\"{run.name}\"]   (this run -- level 2)")
    L.append("# One-off changes (level 3): KNOB=value ./scripts/flagship/submit_gf_"
             f"{run.name}.sh for any")
    L.append(f"# ${{K:-v}} knob, or  python scripts/flagship/build.py {run.name} --set KNOB=value")
    L.append("# --out scripts/flagship/oneoff/<file>.sh  (rewrites a hard export too).")
    L.append("# Guard: tests/test_flagship_launchers.py (committed == rebuilt; the per-run")
    L.append("# deltas; the 9mo == scripts/fstat_proposal/submit_gf_9mo_v9.sh in every line")
    L.append("# bash executes). The 9mo v9 is the base of the series (user ruling 2026-10-11):")
    L.append("# every run follows its pattern and differs ONLY in the lines listed here.")
    L.append("#")
    L.append("# THIS RUN'S VALUES (runs.py):")
    L.append(f"#   TOBS_TARGET={run.tobs} ({run.days} d) -> Nf 1440 x Nt {run.nt} x dt 2.5 s;")
    L.append(f"#   SIGHET_NT_LAYER={run.sighet_nt_layer} (Tobs / 36 h); EDGE_CROP_WAVELETS="
             f"{run.edge_crop_wavelets} (sig-het guard needs >= {need});")
    L.append(f"#   GB_NLEAVES_MAX={run.gb_nleaves_max}, GB_N_SUBBANDS={run.gb_n_subbands} per GPU,"
             f" GB_RJ_INMODEL_CHUNK={run.gb_rj_inmodel_chunk};")
    L.append(f"#   MBHB_IDS={run.mbhb_ids} (t_c < {run.days} + "
             f"{_runs.MBH_MERGER_BUFFER_DAYS} d); EMRI / SOBHB: all;")
    L.append(f"#   STORE_DIR {run.store_dir}, BASE_FILE_NAME {run.base_file_name},")
    L.append(f"#   job {run.resolved_job_name}, log {run.resolved_log_stem}_<jobid>.log;"
             f" NGPUS default {run.ngpus_default}.")
    if parent is not None:
        L.append(f"#   PARENT = the {parent.name} ({parent.days} d): GF_SEED_STORE "
                 f"{parent.seed_store}")
        L.append(f"#     (GB warm start fitted from its last 300 cold-chain rows -> "
                 f"warmstart/{parent.warm_npz}, auto-built")
        L.append("#     on the first launch when missing; psd + galfor pinned at its maxlogL "
                 "cold walker; the source")
        L.append(f"#     warm start copies its SNR > 10 leaves by catalogue id: MBHB "
                 f"{parent.mbhb_ids}, every EMRI / SOBHB; the monitor page overlays")
        L.append(f"#     it: GF_MONITOR_ADD_INFO={_runs.OUTPUT_ROOT}/{parent.store_dirname}).")
        L.append("#     The parent store must be COMPLETE and readable when this launches.")
    else:
        L.append("#   NO PARENT: the no-parent branches of launcher.sh.in (no GB warm start, no")
        L.append("#     gb_search_seed, no seed store, psd start = the unequal-arm reference fit,")
        L.append("#     galfor start = the INFLATED fitted tanh, noise_search forced after")
        L.append("#     source_search, gb_search_1 noise once per cycle end, sources from truth).")
    if run.overrides:
        L.append("#   OVERRIDES (level 2, rewritten export lines): "
                 + ", ".join(f"{k}={v}" for k, v in run.overrides.items()))
    if run.extra_exports:
        L.append(f"#   EXTRA EXPORTS (level 2): {len(run.extra_exports)} line(s) at the hook.")
    if run.notes:
        L.append("#")
        L.append("# NOTES:")
        for n in run.notes:
            L.append(f"#   {n}")
    L.append("#")
    L.append("# RECIPE (the 9mo pattern, ruling 2026-10-07; resolved by")
    L.append("#   python scripts/fstat_proposal/run_combined_staged.py --print-recipe):")
    if parent is not None:
        L.append("#   gb_search_seed x3 -> source_search -> gb_search_1 -> gb_search_2 -> full_pe")
    else:
        L.append("#   source_search -> noise_search -> gb_search_1 -> gb_search_2 -> full_pe")
    L.append("#")
    L.append("# LAUNCH (from the LAT root; a FRESH store on the first submit, a RESUME after):")
    L.append(f"#   ./scripts/flagship/submit_gf_{run.name}.sh")
    L.append("# History of every knob below: scripts/fstat_proposal/submit_gf_9mo_v9.sh (the")
    L.append("# launched 9mo, with the 6mo v9 / v8 headers it inherited) and git log.")
    L.append("# ============================================================================")
    return "\n".join(L)


# ---- the template engine -----------------------------------------------------------
def render(template: str, tok: dict, flags: dict) -> str:
    out = []
    stack = []   # (flag_value, else_seen)
    for lineno, line in enumerate(template.split("\n"), 1):
        s = line.strip()
        if s.startswith("{{#if "):
            flag = s[len("{{#if "):-2].strip()
            if flag not in flags:
                raise KeyError(f"launcher.sh.in:{lineno}: unknown flag {flag!r}")
            stack.append([bool(flags[flag]), False])
            continue
        if s == "{{#else}}":
            if not stack or stack[-1][1]:
                raise SyntaxError(f"launcher.sh.in:{lineno}: stray {{{{#else}}}}")
            stack[-1][1] = True
            continue
        if s == "{{#endif}}":
            if not stack:
                raise SyntaxError(f"launcher.sh.in:{lineno}: stray {{{{#endif}}}}")
            stack.pop()
            continue
        if not all((v if not e else not v) for v, e in stack):
            continue

        def sub(m):
            name = m.group(1)
            if name not in tok:
                raise KeyError(f"launcher.sh.in:{lineno}: unknown token {{{{{name}}}}}")
            return tok[name]

        out.append(_TOKEN.sub(sub, line))
    if stack:
        raise SyntaxError("launcher.sh.in: unterminated {{#if}}")
    text = "\n".join(out)
    left = _TOKEN.findall(text)
    if left:
        raise KeyError(f"unresolved tokens after the fill: {sorted(set(left))}")
    return text


def set_export(text: str, knob: str, value: str) -> str:
    """Rewrite the ONE export line of ``knob`` in ``text`` to carry ``value``."""
    hits = [i for i, l in enumerate(text.split("\n")) if _EXPORT.match(l)
            and _EXPORT.match(l).group(2) == knob]
    if not hits:
        raise KeyError(f"{knob} is not an export line of this launcher: add it as an "
                       "extra_exports line (runs.py, level 2) or on the launch line")
    if len(hits) > 1:
        raise KeyError(f"{knob} is exported {len(hits)} times (lines "
                       f"{[h + 1 for h in hits]}); rewrite it by hand or on the launch line")
    lines = text.split("\n")
    m = _EXPORT.match(lines[hits[0]])
    rhs = m.group(3)
    # split a trailing comment off an unquoted value
    body, comment = rhs, ""
    cm = re.match(r"^([^#\"']*?)(\s+#.*)$", rhs)
    if cm:
        body, comment = cm.group(1), cm.group(2)
    form = re.match(r"^\$\{" + re.escape(knob) + r"(:?-)(.*)\}$", body)
    if form:
        new_body = "${" + knob + form.group(1) + value + "}"
    elif body.startswith('"') and body.endswith('"') and len(body) >= 2:
        new_body = '"' + value + '"'
    else:
        new_body = value
    lines[hits[0]] = f"{m.group(1)}{knob}={new_body}{comment}"
    return "\n".join(lines)


def build(name: str, sets: dict | None = None, template: str | None = None) -> str:
    run = _runs.RUNS[name]
    if template is None:
        with open(TEMPLATE) as fh:
            template = fh.read()
    tok = tokens(run)
    flags = {"HAS_PARENT": run.resolved_parent() is not None}
    text = render(template, tok, flags)
    for k, v in run.overrides.items():
        text = set_export(text, k, v)
    if sets:
        for k, v in sets.items():
            text = set_export(text, k, v)
        text = text.replace(
            "# ============================================================================\n"
            f"# FLAGSHIP global fit -- {run.name}",
            "# ============================================================================\n"
            "# ⚠ ONE-OFF LAUNCHER (level 3): built with --set "
            + " ".join(f"{k}={v}" for k, v in sets.items()) + " -- NOT the committed\n"
            "# launcher; never commit this file.\n"
            f"# FLAGSHIP global fit -- {run.name}", 1)
    if not text.endswith("\n"):
        text += "\n"
    return text


def write(name: str, text: str, out: str | None = None) -> str:
    path = out or launcher_path(name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o755)
    return path


def check(names) -> int:
    rc = 0
    for name in names:
        want = build(name)
        path = launcher_path(name)
        have = open(path).read() if os.path.exists(path) else ""
        if have != want:
            rc = 1
            print(f"[flagship] {os.path.relpath(path)} is STALE (rebuild with build.py):")
            for l in difflib.unified_diff(have.splitlines(), want.splitlines(),
                                          "committed", "rebuilt", lineterm="", n=1):
                print("   ", l)
        else:
            print(f"[flagship] {os.path.relpath(path)} up to date")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="*", help="run name(s); default: all")
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                    help="level 3: rewrite one export line (one run only; needs --out)")
    ap.add_argument("--out", help="write the one-off launcher here (with --set)")
    ap.add_argument("--check", action="store_true", help="compare, do not write")
    ap.add_argument("--list", action="store_true", help="print the resolved table")
    a = ap.parse_args(argv)
    names = a.run or list(_runs.ORDER)
    for n in names:
        if n not in _runs.RUNS:
            ap.error(f"unknown run {n!r}; known: {', '.join(_runs.ORDER)}")
    if a.list:
        for n in names:
            r = _runs.RUNS[n]
            p = r.resolved_parent()
            print(f"{n:>5}: {r.days:4d} d  Tobs {r.tobs:9d}  Nt {r.nt:6d}  stride "
                  f"{r.sighet_nt_layer:4d}  crop {r.edge_crop_wavelets:4d}  leaves "
                  f"{r.gb_nleaves_max}  subbands {r.gb_n_subbands:6d}  chunk "
                  f"{r.gb_rj_inmodel_chunk:6d}  MBHB {r.mbhb_ids}  parent "
                  f"{p.store_dirname if p else '-'}  store {r.store_dir}")
        return 0
    if a.check:
        return check(names)
    sets = dict(s.split("=", 1) for s in a.set)
    if sets:
        if len(names) != 1 or not a.out:
            ap.error("--set takes exactly one run and --out")
        path = write(names[0], build(names[0], sets), a.out)
        print(f"[flagship] wrote the ONE-OFF launcher {path} (never commit it)")
        return 0
    if a.out and len(names) != 1:
        ap.error("--out takes exactly one run")
    for n in names:
        path = write(n, build(n), a.out)
        print(f"[flagship] wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
