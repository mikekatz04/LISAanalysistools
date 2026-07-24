#!/usr/bin/env python
"""t0-foundation/campaign-selftest: prove the campaign plumbing round-trips.

Feeds a fixture log through parse.harvest, checks the harvested metrics and
criteria evaluation, and drops a small PNG into the gate's raw dir so the
proof-plot capture + dashboard embedding path is exercised end to end.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import parse as P  # noqa: E402

FIXTURE = """
Ran 99 tests in 12.345s

OK
[RESULT] variant=gb_no_fg_lite build=ok moves=3 rss_mb=1200
[RESULT] it=0 ll_max=-1.234000e+05 it_wall_s=3.10
[RESULT] it=1 ll_max=-1.230000e+05 it_wall_s=2.90
[GF_TIMING] stage=gb_pe move=rj_prior it=1 wall_s=2.5000 rss_mb=1250 d_rss_mb=10 gpu_used_mb=-1 gpu_pool_mb=-1
[GF_TIMING] stage=gb_pe move=__total__ it=1 wall_s=3.1000 rss_mb=1251
[GF_TIMING] stage=gb_pe move=rj_prior it=2 wall_s=2.4000 rss_mb=1252 d_rss_mb=2 gpu_used_mb=-1 gpu_pool_mb=-1
[GF_TIMING] stage=gb_pe move=__total__ it=2 wall_s=2.9000 rss_mb=1253
[GB_TIMING rj_prior] total=2.45s tracked=2.40s untracked=0.05s | run_proposal=1.9s
SubBandBuffer: 258 cells x (3, 7, 433) per-cell (float64) ~ 33.4 MB [GPU pool used 1.2 / total 2.0 GB]  [host maxRSS 5.1 GB]
[RESULT] branch=mbh data_snr=310.0 rr_over_dd=8.5e-6 xcheck=True
[RESULT] branch=mbh data_snr=1009.0 rr_over_dd=1.5e-3 xcheck=True
[RESULT] class=VGB rank=0 mismatch=1.7e-12
"""

EXPECT = {
    "tests_ran": 99,
    "tests_failed": 0,
    "s_per_it": 3.0,
    "timed_moves": 1,
    "peak_rss_mb": 1253.0,
    "ll_finite": 1,
    "null_rr_dd_MBH_max": 1.5e-3,
    "vgb_mismatch_max": 1.7e-12,
    "subband_buffer_mb_max": 33.4,
    "gpu_pool_used_gb_max": 1.2,
    "host_maxrss_gb": 5.1,
    # bare [RESULT] keys must be directly addressable by gate criteria
    # (regression check: they were once only stored as result_<k>);
    # last line wins, so rank comes from the final VGB row.
    "moves": 3,
    "rank": 0,
}


def main() -> None:
    m = P.harvest(FIXTURE)
    bad = []
    for k, want in EXPECT.items():
        have = m.get(k)
        if have is None or (
            isinstance(want, float) and abs(have - want) > 1e-9 * max(1, abs(want))
        ) or (isinstance(want, int) and have != want):
            bad.append(f"{k}: want {want!r} got {have!r}")

    ok, unmet, manual = P.evaluate(
        (
            {"metric": "tests_failed", "op": "==", "value": 0},
            {"metric": "s_per_it", "op": "<=", "value": 5.0},
            {"manual": "example manual criterion"},
        ),
        m,
    )
    if not ok:
        bad.append(f"criteria evaluation failed: {unmet}")
    if manual != ["example manual criterion"]:
        bad.append("manual criterion not routed")

    # proof-plot path: drop a tiny PNG where campaign.py will capture it
    raw_dir = os.path.join(
        os.path.dirname(os.path.dirname(HERE)), "gf_output", "campaign", "t0-foundation"
    )
    os.makedirs(raw_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 2.2), dpi=100)
        vals = [3.4, 3.1, 2.9, 3.0, 2.8]
        ax.plot(range(len(vals)), vals, marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("s / iteration")
        ax.set_title("campaign selftest: parse->ledger->dashboard")
        fig.tight_layout()
        fig.savefig(os.path.join(raw_dir, "selftest_parse_roundtrip.png"))
        plt.close(fig)
    except Exception as e:  # matplotlib missing is a selftest failure
        bad.append(f"proof PNG generation failed: {e}")

    for b in bad:
        print(f"[selftest] FAIL {b}")
    print(f"[RESULT] selftest_failed={len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
