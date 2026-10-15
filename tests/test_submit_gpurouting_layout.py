"""The GPU-routing submit variant: layout dispatch and the layout report.

``submit_gf_6mo_v8_gpurouting.sh`` is a MIRROR of ``submit_gf_6mo_v8.sh``
carrying only the rank-layout changes that make ``n_compute = n_blocks x R``
launchable. Two things are pinned here:

* the four routing differences behave (this module), and
* **nothing else drifted from the parent** -- ``ParentMirrorTest`` diffs the
  two scripts and fails on any hunk outside those four blocks. That guard is
  the whole reason the fork is safe: the parent is live production and is
  edited by hand, so a silent physics-knob divergence between the two is the
  failure mode worth catching.
"""

import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts", "fstat_proposal")
PARENT = os.path.join(SCRIPTS, "submit_gf_6mo_v8.sh")
CHILD = os.path.join(SCRIPTS, "submit_gf_6mo_v8_gpurouting.sh")


def _text(path):
    with open(path) as fh:
        return fh.read()


class ScriptExistsTest(unittest.TestCase):
    def test_both_scripts_are_present_and_executable(self):
        for p in (PARENT, CHILD):
            self.assertTrue(os.path.exists(p), p)
        self.assertTrue(os.access(CHILD, os.X_OK), "child must be executable")

    def test_bash_parses_the_child(self):
        r = subprocess.run(["bash", "-n", CHILD], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class ParentMirrorTest(unittest.TestCase):
    """Every hunk of the diff against the parent must be a routing change."""

    #: substrings that identify a legitimate routing hunk
    ALLOWED = (
        "GPU-ROUTING VARIANT",          # the child's header
        "NGPUS",                        # the case table
        "_NODES", "_GRES", "_NGPU_PART",
        "_DIST_FLAG", "distribution",
        "RANKS_PER_BLOCK",
        "N_COMPUTE_EFF", "NWALKERS",
        "_gcd", "_NBLOCKS", "_R",
        "block(s)", "rank(s) per block",
        "walker parallelism", "one-walker replica mode",
        "GPUs > walkers",
        "NO ROUND-UP", "build_layout", "gcd",
        "unified factorization", "layout",
        "GF_GPU_ROUTING", "_ROUTING_ON",   # the opt-in this variant turns on
    )

    def test_no_hunk_outside_the_routing_blocks(self):
        r = subprocess.run(["diff", PARENT, CHILD], capture_output=True, text=True)
        # 1 == "files differ", which is expected; 2 == trouble
        self.assertIn(r.returncode, (0, 1), r.stderr)
        offenders = []
        for line in r.stdout.splitlines():
            if not line.startswith(("<", ">")):
                continue
            body = line[1:].strip()
            if not body or body.startswith("#"):
                continue  # comments are free
            # bare shell structure carries no semantics -- a `fi` cannot be a
            # physics-knob divergence, and the routing blocks introduce
            # several of them
            if body in ("else", "fi", "esac", "then", "done", ";;", "{", "}"):
                continue
            if not any(tok in body for tok in self.ALLOWED):
                offenders.append(body)
        self.assertEqual(
            offenders, [],
            "the gpurouting variant drifted from submit_gf_6mo_v8.sh outside "
            "its four routing blocks:\n  " + "\n  ".join(offenders[:20]),
        )

    def test_the_parent_still_has_its_round_up(self):
        # The fork exists so the parent is UNTOUCHED. If this fails, someone
        # edited production instead of the variant.
        self.assertIn("(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF",
                      _text(PARENT))

    def test_the_child_replaced_it_with_the_gcd_report(self):
        t = _text(CHILD)
        self.assertNotIn("(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF", t)
        self.assertIn("_gcd()", t)
        self.assertIn("rank(s) per block", t)


class DispatchTest(unittest.TestCase):
    def test_ngpus_table_accepts_the_new_counts(self):
        t = _text(CHILD)
        self.assertIn("8|16|32)", t)
        self.assertIn("unsupported (2, 4, 8, 16 or 32)", t)
        # nodes = NGPUS/2 on this cluster's 2-GPU nodes
        self.assertIn("_NODES=$(( NGPUS / 2 ))", t)

    def test_distribution_is_blocked_only_when_a_block_is_shared(self):
        t = _text(CHILD)
        self.assertIn('[ "${RANKS_PER_BLOCK}" -gt 1 ]', t)
        self.assertIn("${DIST:-block:block}", t)
        self.assertIn("${DIST:-cyclic}", t)

    def test_ranks_per_block_is_exported_to_the_job(self):
        self.assertIn('RANKS_PER_BLOCK="${RANKS_PER_BLOCK:-}"', _text(CHILD))


class LayoutReportTest(unittest.TestCase):
    """Run the report block in isolation, as the parent suite does."""

    _START = ('if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
              '[ "${N_COMPUTE_EFF}" -gt 0 ]; then')

    def _block(self):
        text = _text(CHILD)
        start = text.index(self._START)
        end = start
        while True:
            end = text.index("\nfi", end + 1)
            if text[end + 1: end + 4] == "fi\n":
                break
        return text[start: end + len("\nfi")]

    def _run(self, **env):
        block = self._block()
        full = {"PATH": os.environ.get("PATH", "")}
        full.update({k: str(v) for k, v in env.items()})
        r = subprocess.run(["bash", "-c", block + "\necho NW=${NWALKERS}"],
                           env=full, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_todays_shape_is_one_block_per_rank(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=4, NWALKERS=4)
        self.assertIn("4 block(s) of 1 walker(s), 1 rank(s) per block", out)
        self.assertNotIn("GPUs > walkers", out)
        self.assertIn("NW=4", out.splitlines())

    def test_gpus_greater_than_walkers(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=16, NWALKERS=4)
        self.assertIn("4 block(s) of 1 walker(s), 4 rank(s) per block", out)
        self.assertIn("GPUs > walkers", out)

    def test_pe_shape_blocks_wider_than_one_walker(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=8, NWALKERS=24)
        self.assertIn("8 block(s) of 3 walker(s), 1 rank(s) per block", out)

    def test_one_walker_is_still_named_explicitly(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=4, NWALKERS=1)
        self.assertIn("one-walker replica mode", out)
        self.assertIn("NW=1", out.splitlines())

    def test_nwalkers_is_never_rewritten(self):
        # 6 on 4 ranks used to be rounded up to 8
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=4, NWALKERS=6)
        self.assertIn("NW=6", out.splitlines())
        self.assertIn("2 block(s) of 3 walker(s), 2 rank(s) per block", out)
        self.assertNotIn("is not a multiple of", out)

    def test_a_coprime_shape_warns(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=4, NWALKERS=3)
        self.assertIn("no walker parallelism", out)
        self.assertIn("NW=3", out.splitlines())

    def test_explicit_ranks_per_block_is_honoured(self):
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=16,
                        NWALKERS=4, RANKS_PER_BLOCK=2)
        self.assertIn("8 block(s)", out)

    def test_an_impossible_explicit_ranks_per_block_warns(self):
        # R=8 over 16 ranks -> 2 blocks, which does not divide 3 walkers
        out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=16,
                        NWALKERS=3, RANKS_PER_BLOCK=8)
        self.assertIn("build_layout will refuse", out)

    def test_the_shell_gcd_matches_pythons(self):
        import math
        for nw, nc in ((4, 4), (4, 16), (24, 8), (24, 32), (1, 4),
                       (10, 4), (3, 2), (32, 16), (6, 4)):
            out = self._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=nc, NWALKERS=nw)
            m = re.search(r"(\d+) block\(s\) of (\d+) walker\(s\), (\d+) rank", out)
            self.assertIsNotNone(m, f"no report for ({nw}, {nc}): {out}")
            nb, per, r = (int(x) for x in m.groups())
            self.assertEqual(nb, math.gcd(nw, nc), f"({nw}, {nc})")
            self.assertEqual(nb * per, nw)
            self.assertEqual(nb * r, nc)


class ProductionLaunchersAgreeTest(unittest.TestCase):
    """The v9 production launchers wire the SAME rule this variant reports.

    Since 2026-10-15 ``GF_GPU_ROUTING=1`` is wired into the v9 production
    launchers themselves (``submit_gf_6mo_v9_4gpu.sh`` and the 9mo / 1yr /
    flagship scripts derived from it): the knob rides ``--export``, NWALKERS
    is never rounded, and the factorization is printed by
    ``_gf_routing_layout``. Two shell copies of one rule must not drift, so
    the production function is checked against this variant's report block
    on a sweep (both are already checked against ``factorize_layout``).
    """

    V9 = os.path.join(SCRIPTS, "submit_gf_9mo_v9.sh")

    def _v9(self, nw, nc):
        text = _text(self.V9)
        i = text.index("_gf_routing_on() {")
        funcs = text[i: text.index('if [ -z "${SLURM_JOB_ID:-}" ]; then', i)]
        r = subprocess.run(["bash", "-c", funcs + f"\n_gf_routing_layout {nw} {nc} 1"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        m = re.search(r"n_blocks=(\d+) walker block\(s\) of (\d+) walker\(s\) x R=(\d+)",
                      r.stdout)
        return tuple(int(x) for x in m.groups())

    def test_the_two_reports_agree(self):
        report = LayoutReportTest()
        for nw, nc in ((4, 8), (4, 4), (4, 16), (24, 8), (24, 32), (1, 4), (6, 4), (3, 2)):
            out = report._run(GF_LEGACY_RANK_LAYOUT=0, N_COMPUTE_EFF=nc, NWALKERS=nw)
            m = re.search(r"(\d+) block\(s\) of (\d+) walker\(s\), (\d+) rank", out)
            self.assertIsNotNone(m, out)
            self.assertEqual(tuple(int(x) for x in m.groups()), self._v9(nw, nc),
                             (nw, nc))


if __name__ == "__main__":
    unittest.main()
