"""Aggregate GB_RJ_TRACE lines from a global-fit run log.

The trace (gbspecialstretch._run_rj_step, env GB_RJ_TRACE=1) logs every
cold-chain DEATH proposal (accepted or not) plus every accepted
cold-chain move:

  RJTRACE BIRTH t=0 w=1 b=1 slot=12 f0=1.020e+01 mHz N=1024 \
      d_h=... h_h=... delta=... beta=... lnp=... factors=... \
      curr_lp=... prev_lp=... accept=1

Focus of the analysis: accepted T0 events whose template is ~zero
(h_h ~ 0) with delta ~ 0 -- the "blank-template accept" signature seen
in the seq0 figures. For each event class we want to know whether the
accept was driven by the prior/proposal factors (amplitude-floor birth
slipping the opt_snr clamp) or by a kernel-side h_h == 0 (band-edge
N/window gating).

Usage: python analyze_rj_trace.py <run.log> [--h-h-floor 1.0]
"""

import argparse
import re
import sys

PAT = re.compile(
    r"RJTRACE (?P<kind>BIRTH|DEATH) t=(?P<t>\d+) w=(?P<w>\d+) b=(?P<b>\d+) "
    r"slot=(?P<slot>-?\d+) f0=(?P<f0>[-\d.eE+]+) mHz N=(?P<N>\d+) "
    r"d_h=(?P<d_h>[-\d.eE+]+) h_h=(?P<h_h>[-\d.eE+]+) delta=(?P<delta>[-\d.eE+]+) "
    r"beta=(?P<beta>[-\d.eE+]+) lnp=(?P<lnp>[-\d.eE+]+) factors=(?P<factors>[-\d.eE+]+) "
    r"curr_lp=(?P<curr_lp>[-\d.eE+]+) prev_lp=(?P<prev_lp>[-\d.eE+]+) accept=(?P<acc>[01])"
)

FLOATS = ("f0", "d_h", "h_h", "delta", "beta", "lnp", "factors", "curr_lp", "prev_lp")
INTS = ("t", "w", "b", "slot", "N")


def parse(path):
    events = []
    with open(path) as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            ev = {"kind": m["kind"], "accept": m["acc"] == "1"}
            for k in FLOATS:
                ev[k] = float(m[k])
            for k in INTS:
                ev[k] = int(m[k])
            events.append(ev)
    return events


def fmt_row(ev):
    snr = (max(ev["h_h"], 0.0)) ** 0.5
    return (
        f"  {ev['kind']:<5} w={ev['w']} slot={ev['slot']:>3} "
        f"f0={ev['f0']:.6f} N={ev['N']:>5} snr_opt={snr:8.3f} "
        f"d_h={ev['d_h']:+.3e} h_h={ev['h_h']:.3e} delta={ev['delta']:+.3e} "
        f"factors={ev['factors']:+.3e} dlp={ev['curr_lp'] - ev['prev_lp']:+.3e} "
        f"lnp={ev['lnp']:+.3e} acc={int(ev['accept'])}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument(
        "--h-h-floor", type=float, default=1.0,
        help="h_h below this counts as a 'blank template' (opt SNR < 1)",
    )
    args = ap.parse_args()

    events = parse(args.log)
    if not events:
        print("no RJTRACE lines found")
        return 1

    t0 = [e for e in events if e["t"] == 0]
    print(f"{len(events)} RJTRACE events, {len(t0)} at T0\n")

    acc = [e for e in t0 if e["accept"]]
    blank_acc = [e for e in acc if e["h_h"] < args.h_h_floor]
    real_acc = [e for e in acc if e["h_h"] >= args.h_h_floor]
    deaths = [e for e in t0 if e["kind"] == "DEATH"]
    deaths_acc = [e for e in deaths if e["accept"]]

    print(f"T0 accepted moves: {len(acc)} "
          f"({sum(e['kind'] == 'BIRTH' for e in acc)} births, "
          f"{sum(e['kind'] == 'DEATH' for e in acc)} deaths)")
    print(f"T0 death proposals: {len(deaths)} ({len(deaths_acc)} accepted)")
    print(f"T0 BLANK-template accepts (h_h < {args.h_h_floor:g}): {len(blank_acc)}\n")

    if blank_acc:
        print("--- blank-template accepts ---")
        for e in blank_acc:
            print(fmt_row(e))
        # What drove the accept: delta*beta vs factors vs prior diff.
        print("\n  accept drivers (lnpdiff = beta*delta + dlp + factors):")
        for e in blank_acc:
            print(
                f"    {e['kind']:<5} slot={e['slot']:>3}: "
                f"beta*delta={e['beta'] * e['delta']:+.3e} "
                f"dlp={e['curr_lp'] - e['prev_lp']:+.3e} "
                f"factors={e['factors']:+.3e}"
            )
        print()

    if deaths_acc:
        print("--- accepted deaths ---")
        for e in deaths_acc:
            print(fmt_row(e))
        print()

    if real_acc:
        hs = sorted((e["h_h"] ** 0.5 for e in real_acc))
        print(f"real-template accepts: {len(real_acc)}, "
              f"opt-SNR range [{hs[0]:.2f}, {hs[-1]:.2f}]")

    # Rejected deaths of bright sources should dominate: sanity print.
    bright_death_rej = [
        e for e in deaths if not e["accept"] and e["h_h"] > 25.0
    ]
    if deaths:
        print(f"bright (snr>5) death proposals rejected: {len(bright_death_rej)}"
              f" / {len(deaths)} total death proposals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
