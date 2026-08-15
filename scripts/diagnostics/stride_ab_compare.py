#!/usr/bin/env python
"""Compare two stride A/B runs (GB_BAND_UNIT_STRIDE=2 vs 3) from their logs.

Usage:
    python scripts/diagnostics/stride_ab_compare.py <log_stride2> <log_stride3>

Parses ``globalfit_run.log`` (or the sbatch stdout) from each run and prints
a side-by-side table of the quantities the A/B is about:

* per-move [GB_TIMING] span means (total, unit_open_close, buffer_build,
  buffill_resid_psd, run_tempering, temper_buffer, rj_step, rj_getll,
  rj_fstat_centers, inmodel_repeats) -- the COST side of stride 3
  (3 units/propose + 3 tempering passes vs 2);
* [GB_ORTHO_LL] worst per unit -- must sit at the same ~1e-5 floor at both
  strides (the bilinearity monitor; a stride-correlated growth here is a
  correctness stop signal);
* [GB_CELL_LL] worst per-repeat excess -- stochastic, same family expected;
* [GB_ACCEPT] cold + all-T rates per move -- statistically unchanged
  expected (the stride reorders proposal sequencing, so compare rates, not
  samples);
* [FSTAT_CTR] census (rows, wall) -- unaffected by stride expected.

Exit code 1 if a correctness gate trips (ortho-ll floor ratio > 10x or an
acceptance rate moved by more than 3 sigma of its binomial error).
"""
import math
import re
import sys

TIMING_RE = re.compile(r"\[GB_TIMING (\w+)\] total=([\d.]+)s.*?\| (.*?) \|")
SPAN_RE = re.compile(r"(\w+)=([\d.]+)s")
ORTHO_RE = re.compile(r"\[GB_ORTHO_LL (\w+)\].*?max ([\d.eE+-]+)")
CELL_RE = re.compile(
    r"\[GB_CELL_LL (\w+)\] unit:.*?worst per-repeat ([\d.eE+-]+)/rep")
ACC_RE = re.compile(
    r"\[GB_ACCEPT (\w+)\] rj cold (\d+)/(\d+) \(([\d.]+)\) "
    r"all (\d+)/(\d+) \(([\d.]+)\)")
CTR_RE = re.compile(
    r"\[FSTAT_CTR (\w+)\] unit precompute: (\d+) rows.*? in ([\d.]+)s")

SPANS = ["unit_open_close", "buffer_build", "buffill_resid_psd",
         "run_tempering", "temper_buffer", "rj_step", "rj_getll",
         "rj_fstat_centers", "inmodel_repeats"]


def parse(path):
    out = {"timing": {}, "ortho": {}, "cell": {}, "acc": {}, "ctr": {}}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TIMING_RE.search(line)
            if m:
                move, total, spans = m.group(1), float(m.group(2)), m.group(3)
                rec = out["timing"].setdefault(
                    move, {"total": [], **{s: [] for s in SPANS}})
                rec["total"].append(total)
                got = dict(SPAN_RE.findall(spans))
                for s in SPANS:
                    if s in got:
                        rec[s].append(float(got[s]))
                continue
            m = ORTHO_RE.search(line)
            if m:
                out["ortho"].setdefault(m.group(1), []).append(
                    float(m.group(2)))
                continue
            m = CELL_RE.search(line)
            if m:
                out["cell"].setdefault(m.group(1), []).append(
                    float(m.group(2)))
                continue
            m = ACC_RE.search(line)
            if m:
                move = m.group(1)
                out["acc"][move] = {
                    "cold": (int(m.group(2)), int(m.group(3))),
                    "all": (int(m.group(5)), int(m.group(6))),
                }
                continue
            m = CTR_RE.search(line)
            if m:
                out["ctr"].setdefault(m.group(1), []).append(
                    (int(m.group(2)), float(m.group(3))))
    return out


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def fmt(v):
    return f"{v:10.2f}" if v == v else "         -"


def rate_sigma(a, b):
    """Two-proportion z between (k, n) tuples."""
    (k1, n1), (k2, n2) = a, b
    if min(n1, n2) == 0:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-30))
    return abs(k1 / n1 - k2 / n2) / se


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = parse(sys.argv[1]), parse(sys.argv[2])
    failed = []

    print(f"{'':34s}{'stride A':>12s}{'stride B':>12s}{'B/A':>8s}")
    print("=" * 66)
    for move in sorted(set(a["timing"]) | set(b["timing"])):
        ra, rb = a["timing"].get(move, {}), b["timing"].get(move, {})
        na = len(ra.get("total", []))
        nb = len(rb.get("total", []))
        print(f"[{move}]  (n={na} vs {nb} proposes, span means in s)")
        for s in ["total"] + SPANS:
            va, vb = mean(ra.get(s, [])), mean(rb.get(s, []))
            if va != va and vb != vb:
                continue
            ratio = vb / va if va and va == va and vb == vb else float("nan")
            rs = f"{ratio:7.2f}x" if ratio == ratio else "      -"
            print(f"  {s:32s}{fmt(va)}{fmt(vb)}{rs}")

    print("\n[GB_ORTHO_LL] worst-per-unit max (bilinearity monitor)")
    for move in sorted(set(a["ortho"]) | set(b["ortho"])):
        va = max(a["ortho"].get(move, [float('nan')]))
        vb = max(b["ortho"].get(move, [float('nan')]))
        flag = ""
        if va == va and vb == vb and va > 0 and vb / va > 10:
            flag = "  <-- STOP: stride-correlated growth"
            failed.append(f"ortho_ll {move} grew {vb/va:.1f}x")
        print(f"  {move:32s}{va:12.3e}{vb:12.3e}{flag}")

    print("\n[GB_CELL_LL] worst per-repeat excess (stochastic; same family expected)")
    for move in sorted(set(a["cell"]) | set(b["cell"])):
        va = max(a["cell"].get(move, [float('nan')]))
        vb = max(b["cell"].get(move, [float('nan')]))
        print(f"  {move:32s}{va:12.3e}{vb:12.3e}")

    print("\n[GB_ACCEPT] rates (last record per move; 2-proportion z)")
    for move in sorted(set(a["acc"]) | set(b["acc"])):
        if move not in a["acc"] or move not in b["acc"]:
            continue
        for kind in ("cold", "all"):
            ta, tb = a["acc"][move][kind], b["acc"][move][kind]
            z = rate_sigma(ta, tb)
            flag = ""
            if z > 3:
                flag = "  <-- >3 sigma"
                failed.append(f"accept {move}/{kind} z={z:.1f}")
            print(f"  {move}/{kind:28s}"
                  f"{ta[0]/max(ta[1],1):12.4f}{tb[0]/max(tb[1],1):12.4f}"
                  f"   z={z:5.2f}{flag}")

    for name, d in (("A", a), ("B", b)):
        for move, recs in d["ctr"].items():
            rows = mean([r[0] for r in recs])
            wall = mean([r[1] for r in recs])
            print(f"\n[FSTAT_CTR {move}] {name}: mean {rows:,.0f} rows "
                  f"in {wall:.1f}s per unit ({len(recs)} units)")

    if failed:
        print("\nGATES TRIPPED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nAll correctness gates clean (timing differences are the "
          "expected stride cost).")


if __name__ == "__main__":
    main()
