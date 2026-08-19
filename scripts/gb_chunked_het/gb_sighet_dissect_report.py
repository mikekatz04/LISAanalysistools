"""Verdict report for GB_SIGHET_DISSECT dumps.

Reads the per-block npz files the move writes under ``GB_SIGHET_DISSECT=<dir>``
(see ``gbspecialstretch._sighet_dissect_dump``) and answers, in order, the
three questions that keep going in circles:

1. **Is the reference build itself broken, and for whom?** ``eps0 =
   |het0 - ex0|`` is sig-het vs exact AT the expansion point on the frozen
   residual -- r(t) = 1, so displacement, sparse-grid resolution and drift are
   all structurally excluded. Any eps0 above a few lnL is a setup/fold defect.
   The d_h/h_h split attributes it: ``hh_het0 != hh_ex0`` is the
   template-side fold (reference build / B-moments); ``dh_het0 != dh_ex0`` is
   the data side (residual slab / A-moments); both matching while ll differs
   points at the composition (phase max, kept mask).

2. **Does the error track the deep-null hypothesis?** eps0 and worst
   eps_delta/T are correlated against ``null_depth`` (min/max supported c0
   row power) and ``frac_masked`` (pixels under the scorer's own row floor).

3. **What does displacement actually cost?** The tier ladder in the
   delta-vs-delta lens, per source -- compared ACROSS dump dirs when several
   are given, which is how the knob matrix (nt_layer 60 vs 270, v5 on/off,
   n_r 32/64/128) becomes attributable: same store, same frozen state, one
   knob per run.

Usage::

    python scripts/gb_chunked_het/gb_sighet_dissect_report.py DIR [DIR2 ...]

Each DIR is one engine configuration's dump directory. With multiple DIRs the
final section prints the cross-config table. Plain numpy; runs anywhere.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

EPS0_BAD = 10.0          # lnL; anchor disagreement above this = broken ref
COLD_BETA = 0.999


def load_dir(d):
    files = sorted(glob.glob(os.path.join(d, "dissect_*.npz")))
    if not files:
        raise SystemExit(f"no dissect_*.npz under {d}")
    out = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            out.setdefault(k, []).append(z[k])
    blocks = len(files)
    # per-source arrays concatenate; per-block scalars keep the first
    cat = {}
    for k, v in out.items():
        if v[0].ndim == 0:
            cat[k] = v[0]
        elif k in ("tiers",):
            cat[k] = v[0]
        elif v[0].ndim == 2:      # (tiers, n) stacks -> concat on axis 1
            cat[k] = np.concatenate(v, axis=1)
        else:
            cat[k] = np.concatenate(v)
    cat["_blocks"] = blocks
    return cat


def pct(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def report_one(d, c):
    n = c["f0_hz"].size
    tiers = np.asarray(c["tiers"], dtype=float)
    print(f"\n{'='*76}\n{d}\n  blocks={c['_blocks']}  sources={n}  "
          f"tiers={tiers.tolist()}")
    if str(c.get("config", "")):
        print(f"  engine: {c['config']}")

    # ---- 1. anchor fidelity ------------------------------------------------
    eps0 = np.abs(c["het0"] - c["ex0"])
    fin = np.isfinite(eps0)
    bad = fin & (eps0 > EPS0_BAD)
    print(f"\n  [1] ANCHOR FIDELITY  eps0 = |sighet - exact| at r(t)=1")
    print(f"      median={pct(eps0,50):.3g}  p90={pct(eps0,90):.3g}  "
          f"p99={pct(eps0,99):.3g}  max={np.nanmax(eps0):.3g}")
    print(f"      broken refs (eps0 > {EPS0_BAD:g}): {bad.sum()}/{fin.sum()} "
          f"({100*bad.sum()/max(fin.sum(),1):.1f}%)")
    if bad.any():
        # attribution split
        dhh = np.abs(c["hh_het0"] - c["hh_ex0"])
        ddh = np.abs(c["dh_het0"] - c["dh_ex0"])
        hh_dom = bad & (dhh > 2 * ddh)
        dh_dom = bad & (ddh > 2 * dhh)
        both = bad & ~hh_dom & ~dh_dom
        print(f"      attribution: h_h-dominated {hh_dom.sum()} "
              f"(reference/B-moment build) | d_h-dominated {dh_dom.sum()} "
              f"(residual slab/A-moments) | mixed {both.sum()}")
        # the recurring-frequency census
        f0b = np.round(c["f0_hz"][bad] * 1e6)   # to ~binned uHz
        uf, cnt = np.unique(f0b, return_counts=True)
        order = np.argsort(cnt)[::-1][:8]
        print("      worst frequencies (count | f0 mHz | band | "
              "med eps0 | med hh_het/hh_ex | cold?):")
        for j in order:
            m = bad & (np.round(c["f0_hz"] * 1e6) == uf[j])
            r = np.nanmedian(c["hh_het0"][m] / np.maximum(c["hh_ex0"][m],
                                                          1e-300))
            ncold = int((c["beta"][m] > COLD_BETA).sum())
            print(f"        {cnt[j]:4d} | {uf[j]/1e3:9.4f} | "
                  f"{int(np.median(c['band'][m])):4d} | "
                  f"{np.nanmedian(eps0[m]):9.3g} | {r:9.3g} | "
                  f"{ncold}/{m.sum()} cold")

    # ---- 2. null correlation ----------------------------------------------
    nd = c["null_depth"]
    have = np.isfinite(nd) & fin
    print(f"\n  [2] DEEP-NULL HYPOTHESIS  (c0 stats on {have.sum()} sources)")
    if have.sum() > 50:
        lo = have & (nd < pct(nd[have], 10))
        hi = have & (nd > pct(nd[have], 50))
        print(f"      eps0 median: deepest-null decile {pct(eps0[lo],50):.3g}"
              f"  vs shallow half {pct(eps0[hi],50):.3g}"
              f"  (ratio {pct(eps0[lo],50)/max(pct(eps0[hi],50),1e-30):.2f}x)")
        r = np.corrcoef(np.log10(np.maximum(nd[have], 1e-30)),
                        np.log10(np.maximum(eps0[have], 1e-30)))[0, 1]
        print(f"      corr[log null_depth, log eps0] = {r:+.3f} "
              f"(negative = deeper null, bigger error)")
    else:
        print("      c0 stats unavailable (engine layout not probed); "
              "re-run with a single-shard engine or extend _dissect_comps.")

    # ---- 3. displacement ladder -------------------------------------------
    print(f"\n  [3] DISPLACEMENT (delta-vs-delta, per source; cold only)")
    cold = c["beta"] > COLD_BETA
    het, exa = c["het"], c["exa"]
    rows = []
    for i, dph in enumerate(tiers):
        eps = np.abs((het[i] - c["ll_ref"]) - (exa[i] - c["ex0"]))[cold]
        T = np.abs(exa[i] - c["ex0"])[cold]
        ratio = eps / np.maximum(T, 1e-300)
        ok = np.isfinite(ratio)
        rows.append((dph, pct(ratio[ok], 50), pct(ratio[ok], 90),
                     float(np.nanmax(ratio[ok])) if ok.any() else np.nan))
        print(f"      dphase {dph:6.3f} rad: eps/T med {rows[-1][1]:8.3g}  "
              f"p90 {rows[-1][2]:8.3g}  max {rows[-1][3]:8.3g}")
    return dict(eps0=eps0, rows=rows, cold=cold)


def report_sweeps(d):
    """The in-run sweep (GB_SIGHET_SWEEP): per-arm anchor + tier table.

    Every arm scored the SAME sources on the SAME frozen residual against
    the SAME shared exact side -- so a column that moves is caused by that
    arm's configuration and nothing else. Two rows carry the verdict:
    anchor |dll| (a knob that only changes RESOLUTION cannot move it) and
    the per-tier eps/T (where resolution legitimately acts)."""
    files = sorted(glob.glob(os.path.join(d, "sweep_*.npz")))
    if not files:
        return
    for f in files:
        z = np.load(f, allow_pickle=False)
        tiers = z["tiers"]; ex0 = z["ex0"]; exa = z["exa"]
        cold = z["beta"] > COLD_BETA
        print(f"\n  [SWEEP] {os.path.basename(f)}  "
              f"({z['sub'].size} sources, {cold.sum()} cold, "
              f"arms: {list(z['arms'])})")
        hdr = (f"      {'arm':24s} {'wall':>6s} {'anchor med':>10s} "
               f"{'anchor max':>10s} {'vs base':>9s}"
               + "".join(f" {'d=%g' % t:>9s}" for t in tiers))
        print(hdr + "   (anchor=|dll| vs exact; tiers=eps/T med, cold)")
        base0 = z.get("a00_het0")
        for i, arm in enumerate(z["arms"]):
            a = f"a{i:02d}"
            if str(z["arm_error"][i]):
                print(f"      {str(arm):24s} FAILED: {z['arm_error'][i]}")
                continue
            h0 = z[f"{a}_het0"]; ht = z[f"{a}_het"]
            e0 = np.abs(h0 - ex0)
            # THE SCORING-PATH SENTINEL. max |het0_arm - het0_base|: a real
            # engine change must differ from base at least in round-off on
            # SOME source. Exactly 0.0 on every source means the arm scored
            # through the base engine (a sweep wiring bug), and its row
            # proves nothing. Small-but-nonzero + identical anchor columns
            # is the meaningful verdict: corruption upstream of the knob.
            vsb = (float(np.nanmax(np.abs(h0 - base0)))
                   if (i > 0 and base0 is not None) else 0.0)
            flag = "  <-- BITWISE base; arm did NOT rescore!" \
                if (i > 0 and vsb == 0.0) else ""
            cells = []
            for j in range(len(tiers)):
                eps = np.abs((ht[j] - h0) - (exa[j] - ex0))[cold]
                T = np.abs(exa[j] - ex0)[cold]
                cells.append(f"{np.nanmedian(eps/np.maximum(T,1e-300)):9.3g}")
            print(f"      {str(arm):24s} {z['arm_wall'][i]:5.0f}s "
                  f"{np.nanmedian(e0):10.3g} {np.nanmax(e0):10.3g} "
                  f"{vsb:9.3g}"
                  + " ".join([""] + cells) + flag)
        if base0 is not None:
            print("      (arm rows share sources/residual/exact side; only "
                  "the engine differs)")


def main(argv):
    dirs = argv or ["."]
    results = {}
    for d in dirs:
        results[d] = report_one(d, load_dir(d))
        report_sweeps(d)
    if len(dirs) > 1:
        print(f"\n{'='*76}\nCROSS-CONFIG  (eps/T median per tier; a knob that "
              "matters moves its column)")
        tiers = None
        for d, r in results.items():
            t = [f"{x[1]:.3g}" for x in r["rows"]]
            if tiers is None:
                tiers = [f"{x[0]:g}" for x in r["rows"]]
                print(f"  {'dir':40s} " + " ".join(f"{x:>8s}" for x in tiers))
            print(f"  {os.path.basename(os.path.normpath(d)):40s} "
                  + " ".join(f"{x:>8s}" for x in t))
        print("\n  eps0 (anchor) median/max per config -- resolution knobs "
              "CANNOT move this row; if it moves, the knob touches setup:")
        for d, r in results.items():
            e = r["eps0"]
            print(f"  {os.path.basename(os.path.normpath(d)):40s} "
                  f"{np.nanmedian(e):9.3g} / {np.nanmax(e):9.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
