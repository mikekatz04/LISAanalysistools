"""The flagship launchers: ONE template + ONE run table -> six submit scripts.

``scripts/flagship/launcher.sh.in`` (level 1, every run) and
``scripts/flagship/runs.py`` (level 2, one run) build ``submit_gf_<run>.sh``
for 3mo / 6mo / 9mo / 12mo / 18mo / 24mo through ``scripts/flagship/build.py``
(user ruling 2026-10-11: "build the submit scripts from a singular consistent
structure across launches that is then adjusted particular to each launch
only where necessary ... The 9mo v9 script is the base").

Five guards:

* DRIFT: every committed launcher equals what the table + template build
  (an edit to either without a rebuild, or a hand edit of a launcher, fails);
* IDENTITY: the flagship 9mo behaves exactly as the launched
  ``scripts/fstat_proposal/submit_gf_9mo_v9.sh`` -- every line bash executes
  (exports, assignments, control flow, heredoc preflights, SBATCH directives)
  is identical, with three stale operator-hint lines named explicitly;
* STRUCTURE: tokens live only on the declared per-run lines of the template,
  and every run's resolved exports differ from the 9mo's only in the declared
  Tobs-driven / parent-driven / no-parent sets;
* VALUES: the per-run numbers follow their rules (grid, sig-het stride, edge
  crop guard, MBHB two-week rule, the parent chain, the 3mo's no-parent
  overlay, NGPUS default);
* COMPOSITION: the driver's ``--print-recipe`` composes each launcher's
  exports into the intended stage list (needs lisatools importable).
"""
import importlib.util
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLAGSHIP = os.path.join(ROOT, "scripts", "flagship")
NINE_MO_V9 = os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_9mo_v9.sh")
DRIVER = os.path.join(ROOT, "scripts", "fstat_proposal", "run_combined_staged.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fb = _load("flagship_build", os.path.join(FLAGSHIP, "build.py"))
fr = fb._runs
ORDER = fr.ORDER
PARENTED = tuple(n for n in ORDER if fr.RUNS[n].parent is not None)
NO_PARENT = tuple(n for n in ORDER if fr.RUNS[n].parent is None)


def _path(name):
    return fb.launcher_path(name)


def _text(name):
    with open(_path(name)) as fh:
        return fh.read()


def _exports(path):
    """``{KNOB: value}`` as bash resolves the script's export lines in order
    (the same helper as tests/test_submit_scripts_layout.py)."""
    src = open(path).read()
    lines = [l for l in src.split("\n") if re.match(r"^export [A-Z0-9_]+=", l)]
    env = dict(os.environ)
    for k in list(env):
        if k.isupper():
            env.pop(k, None)
    out = subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv | sort\n"],
                         capture_output=True, text=True, env=env).stdout
    return dict(l.split("=", 1) for l in out.split("\n") if "=" in l)


_ASSIGN_COMMENT = re.compile(r"^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=[^#\"']*?)\s+#.*$")
_SBATCH_COMMENT = re.compile(r"^(#SBATCH\s+\S+)\s+#.*$")
_HEREDOC = re.compile(r"<<-?\s*'?(\w+)'?")


def executable_lines(text):
    """What bash runs: every non-comment line (trailing comments stripped from
    assignments and SBATCH directives), heredoc bodies kept verbatim."""
    out, hd = [], None
    for line in text.split("\n"):
        if hd is not None:
            out.append(line)
            if line.strip() == hd:
                hd = None
            continue
        s = line.strip()
        if s == "" or (s.startswith("#") and not s.startswith("#SBATCH")
                       and not s.startswith("#!")):
            continue
        m = _ASSIGN_COMMENT.match(line)
        if m:
            line = m.group(1)
        m = _SBATCH_COMMENT.match(line)
        if m:
            line = m.group(1)
        out.append(line.rstrip())
        m = _HEREDOC.search(line)
        if m and not line.lstrip().startswith("#"):
            hd = m.group(1)
    return out


_STUB_SBATCH = """#!/usr/bin/env bash
for a in "$@"; do printf '%s\\n' "$a"; done
exit 0
"""
_DISPATCH_ENV_KEYS = ("SLURM_JOB_ID", "NGPUS", "NODES", "GPUS_PER_RANK", "RANKS_PER_GPU",
                      "GF_LEGACY_RANK_LAYOUT", "GF_MONITOR_RANK")


class _Dispatch:
    """Run a launcher's pre-submit block against a stub ``sbatch`` (argv echo)."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="flagship_sbatch_stub_")
        stub = os.path.join(self.tmp.name, "sbatch")
        with open(stub, "w") as fh:
            fh.write(_STUB_SBATCH)
        os.chmod(stub, 0o755)

    def run(self, script, overrides):
        env = dict(os.environ)
        for key in _DISPATCH_ENV_KEYS:
            env.pop(key, None)
        env["PATH"] = self.tmp.name + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        r = subprocess.run(["bash", script], env=env, capture_output=True, text=True,
                           timeout=60)
        if r.returncode != 0:
            raise AssertionError(f"{script} {overrides}: exit {r.returncode}\n"
                                 f"{r.stdout}\n{r.stderr}")
        return r.stdout.splitlines()


# ---------------------------------------------------------------------------
class DriftTest(unittest.TestCase):
    """Every committed launcher is exactly what build.py builds today."""

    def test_committed_launchers_match_the_table_and_template(self):
        for name in ORDER:
            with self.subTest(run=name):
                self.assertTrue(os.path.exists(_path(name)), _path(name))
                self.assertEqual(_text(name), fb.build(name),
                                 f"scripts/flagship/submit_gf_{name}.sh is stale: run "
                                 "python scripts/flagship/build.py")

    def test_the_check_mode_agrees(self):
        r = subprocess.run([sys.executable, os.path.join(FLAGSHIP, "build.py"), "--check"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_no_unresolved_tokens_or_conditionals_leak(self):
        for name in ORDER:
            with self.subTest(run=name):
                t = _text(name)
                self.assertNotIn("{{", t)        # ("}}" is legal bash: ${A-${B}})
                self.assertNotRegex(t, r"(?m)^\{\{#")

    def test_bash_syntax(self):
        for name in ORDER:
            with self.subTest(run=name):
                r = subprocess.run(["bash", "-n", _path(name)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(os.access(_path(name), os.X_OK), "not executable")

    def test_the_files_are_marked_generated(self):
        for name in ORDER:
            with self.subTest(run=name):
                head = _text(name).split("\n", 3)
                self.assertEqual(head[0], "#!/bin/bash")
                self.assertIn("GENERATED FILE, DO NOT", head[2])


# ---------------------------------------------------------------------------
class NineMonthIdentityTest(unittest.TestCase):
    """``scripts/flagship/submit_gf_9mo.sh`` == the launched 9mo v9 launcher in
    every line bash executes (user ruling 2026-10-11: "byte-identical or
    whatever where it will behave exactly the same in the new structure")."""

    # the ONLY executable lines allowed to differ: three stale operator hints
    # that the 9mo had inherited verbatim from the 6mo (they name the 6mo's
    # 3mo parent); the flagship names the run's actual parent. All three sit on
    # the "seed store missing / npz cannot be built" diagnostic paths.
    ALLOWED = (
        ('    echo "[V9-SEED] Candidate 3mo run directories on this filesystem:"',
         '    echo "[V9-SEED] Candidate 6mo run directories on this filesystem:"'),
        ("    find /shared/data/global_fit_output -maxdepth 1 -type d -name 'gf_prod_3mo*' \\",
         "    find /shared/data/global_fit_output -maxdepth 1 -type d -name 'gf_prod_6mo*' \\"),
        ('    echo "[WARMSTART]       --last-k 10 --tobs 7776000 --out <dir>/gf_prod_3mo_v8_10w_fit.npz"',
         '    echo "[WARMSTART]       --last-k ${GB_WARM_START_LAST_K} --tobs ${GB_WARM_START_SOURCE_TOBS} --out <dir>/gf_prod_6mo_v9_4gpu_fit.npz"'),
    )

    def setUp(self):
        self.old = open(NINE_MO_V9).read()
        self.new = _text("9mo")

    def test_every_line_bash_executes_is_identical(self):
        a, b = executable_lines(self.old), executable_lines(self.new)
        self.assertEqual(len(a), len(b), "different executable line counts")
        seen = set()
        for i, (x, y) in enumerate(zip(a, b)):
            if x == y:
                continue
            self.assertIn((x, y), self.ALLOWED,
                          f"executable line {i} drifted:\n  old: {x}\n  new: {y}")
            seen.add((x, y))
        self.assertEqual(seen, set(self.ALLOWED),
                         "the allowlist carries a pair that no longer differs")

    def test_the_resolved_exports_are_identical(self):
        self.assertEqual(_exports(NINE_MO_V9), _exports(_path("9mo")))

    def test_the_export_line_forms_are_identical(self):
        # ${K:-v} vs ${K-v} vs a hard pin decides what the launch line can reach
        form = lambda t: [l.split("#")[0].rstrip() for l in t.split("\n")
                          if re.match(r"^export [A-Z0-9_]+=", l)]
        self.assertEqual(form(self.old), form(self.new))

    def test_the_sbatch_directives_are_identical(self):
        sb = lambda t: [_SBATCH_COMMENT.sub(r"\1", l).rstrip() for l in t.split("\n")
                        if l.startswith("#SBATCH")]
        self.assertEqual(sb(self.old), sb(self.new))

    def test_the_heredoc_preflights_are_identical(self):
        def heredocs(t):
            out, cur, hd = [], None, None
            for line in t.split("\n"):
                if hd is not None:
                    if line.strip() == hd:
                        out.append("\n".join(cur)); cur, hd = None, None
                    else:
                        cur.append(line)
                    continue
                m = _HEREDOC.search(line)
                if m and not line.lstrip().startswith("#"):
                    hd, cur = m.group(1), []
            return out
        self.assertEqual(heredocs(self.old), heredocs(self.new))
        self.assertGreaterEqual(len(heredocs(self.new)), 6)

    def test_the_dispatch_is_identical(self):
        d = _Dispatch()
        try:
            for ov in ({}, {"NGPUS": "2"}, {"NGPUS": "4"}, {"NGPUS": "2", "NODES": "2"},
                       {"GF_LEGACY_RANK_LAYOUT": "1", "NGPUS": "2"}):
                with self.subTest(overrides=ov):
                    # the stub echoes argv; the last element is the script itself
                    a = [l for l in d.run(NINE_MO_V9, ov) if l != NINE_MO_V9]
                    b = [l for l in d.run(_path("9mo"), ov) if l != _path("9mo")]
                    self.assertEqual(a, b)
                    self.assertTrue(any(l.startswith("--export=") for l in a))
        finally:
            d.tmp.cleanup()

    def test_the_names_are_the_live_runs(self):
        e = _exports(_path("9mo"))
        self.assertEqual(e["BASE_FILE_NAME"], "gf_prod_9mo")
        self.assertIn("\nSTORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_9mo_v9/}\n",
                      self.new)
        self.assertIn("#SBATCH --job-name=gf9mo_v9", self.new)
        self.assertIn("/shared/data/global_fit_output/gf9mo_v9_%j.log", self.new)
        self.assertEqual(e["GF_SEED_STORE"],
                         "/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5")
        self.assertNotIn("flagship", "\n".join(executable_lines(self.new)))


# ---------------------------------------------------------------------------
class TemplateStructureTest(unittest.TestCase):
    """Tokens live ONLY on the declared per-run lines of the template."""

    TOKEN_EXPORTS = {
        "TOBS_TARGET", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS", "GB_NLEAVES_MAX",
        "GB_N_SUBBANDS", "GB_RJ_INMODEL_CHUNK", "MBHB_IDS", "EMRI_IDS", "BASE_FILE_NAME",
        "GF_SEED_STORE", "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_TOBS",
        "GB_WARM_START_LAST_K", "MBH_SOURCE_WARM_START_IDS", "EMRI_SOURCE_WARM_START_IDS",
        "SOBBH_SOURCE_WARM_START_IDS",
        "GF_MONITOR_ADD_INFO",      # 2026-10-12: the parent run overlaid on the monitor page
    }
    OTHER_LINES = (
        "STORE_DIR=${STORE_DIR:-{{STORE_DIR}}}",
        "SLURM_LOG={{OUTPUT_ROOT}}/{{LOG_STEM}}_${SLURM_JOB_ID:-manual}.log",
        "  NGPUS=${NGPUS:-{{NGPUS_DEFAULT}}}",
        'if [ "${NGPUS:-{{NGPUS_DEFAULT}}}" = "4" ]; then',
        "        mbh, domain_settings=WDMSettings.make_factory(1440, {{NT}}))",
        "        emri, domain_settings=WDMSettings.make_factory(1440, {{NT}}), tdi_chan=tdi_chan)",
        '    echo "[V9-SEED] Candidate {{PARENT_NAME}} run directories on this filesystem:"',
        "    find {{OUTPUT_ROOT}} -maxdepth 1 -type d -name 'gf_prod_{{PARENT_NAME}}*' \\",
        '    echo "[WARMSTART]       --last-k ${GB_WARM_START_LAST_K} --tobs ${GB_WARM_START_SOURCE_TOBS} --out <dir>/{{PARENT_STORE_DIRNAME}}_fit.npz"',
        "{{HEADER}}", "{{RUN_EXTRA_EXPORTS}}",
    )

    def setUp(self):
        with open(fb.TEMPLATE) as fh:
            self.template = fh.read()

    def test_tokens_only_on_declared_lines(self):
        bad = []
        for lineno, line in enumerate(self.template.split("\n"), 1):
            if "{{" not in line or line.strip().startswith("{{#"):
                continue
            if line.lstrip().startswith("#"):
                continue                      # comments and SBATCH directives
            m = re.match(r"^export ([A-Z0-9_]+)=", line)
            if m and m.group(1) in self.TOKEN_EXPORTS:
                continue
            bare = _ASSIGN_COMMENT.sub(r"\1", line).rstrip()
            if bare in self.OTHER_LINES:
                continue
            bad.append(f"{lineno}: {line}")
        self.assertEqual(bad, [], "tokens on undeclared executable lines:\n" + "\n".join(bad))

    def test_every_declared_line_exists(self):
        for want in self.OTHER_LINES:
            self.assertIn(want, self.template, want)
        for knob in self.TOKEN_EXPORTS:
            self.assertRegex(self.template, rf"(?m)^export {knob}=.*\{{\{{", knob)

    def test_the_conditionals_are_balanced_and_all_one_flag(self):
        opens = re.findall(r"(?m)^\{\{#if (\w+)\}\}$", self.template)
        self.assertEqual(set(opens), {"HAS_PARENT"})
        self.assertEqual(len(opens), self.template.count("{{#else}}"))
        self.assertEqual(len(opens), self.template.count("{{#endif}}"))
        self.assertEqual(len(opens), 8)

    def test_the_no_parent_branches_carry_the_overlay(self):
        text = fb.render(self.template, fb.tokens(fr.RUNS["3mo"]), {"HAS_PARENT": False})
        for frag in ("export GB_WARM_START_COMPONENTS=\n", "export GB_SEARCH_SEED_ITERS=0\n",
                     "export GB_REPLACE_WARM_PASS=0", "SOURCE_SEARCH_POSITION:-first}",
                     "export SOURCE_WARM_START_STORE=\n", "STAGE_FORCE_NOISE_SEARCH:-1}",
                     "GB_SEARCH_1_NOISE_MODE:-cycle_end}"):
            self.assertIn(frag, text, frag)
        for frag in ("export GF_SEED_STORE=", "export GB_WARM_START_SOURCE_TOBS=",
                     "export GB_SEARCH_3_WARM_EVERY=", "export MBH_SOURCE_WARM_START_IDS=",
                     "export GF_MONITOR_ADD_INFO="):
            self.assertNotIn(frag, text, frag)


# ---------------------------------------------------------------------------
class PerRunValuesTest(unittest.TestCase):
    """The per-run numbers follow their rules."""

    @classmethod
    def setUpClass(cls):
        cls.ex = {n: _exports(_path(n)) for n in ORDER}
        cls.tx = {n: _text(n) for n in ORDER}

    def test_tobs_grid_stride_and_crop(self):
        for name in ORDER:
            run = fr.RUNS[name]
            with self.subTest(run=name):
                e = self.ex[name]
                self.assertEqual(int(e["TOBS_TARGET"]), run.days * 86400)
                self.assertEqual(int(e["TOBS_TARGET"]), 1440 * run.nt * 25 // 10)
                self.assertEqual(int(e["SIGHET_NT_LAYER"]), int(e["TOBS_TARGET"]) // (36 * 3600))
                self.assertEqual(run.nt % int(e["SIGHET_NT_LAYER"]), 0)
                need = math.ceil(0.005 * run.nt) + 8          # check_sighet_build_config
                self.assertGreaterEqual(int(e["EDGE_CROP_WAVELETS"]), need)
                self.assertGreaterEqual(int(e["EDGE_CROP_WAVELETS"]), 20)   # the code default
                self.assertEqual(self.tx[name].count(f"make_factory(1440, {run.nt})"), 2)
                self.assertEqual(self.tx[name].count(f"Nf 1440 x Nt {run.nt} at dt 2.5 s"), 3)
                self.assertEqual(len(re.findall(r"make_factory\(1440, \d+\)", self.tx[name])), 2)

    def test_the_grid_is_what_the_stock_wdm_settings_resolve(self):
        try:
            from lisatools.domains import WDMSettings
            from lisatools.globalfit.stock.erebor.fit import EreborGeneralSettings
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"lisatools not importable here: {exc!r}")
        lo, hi = EreborGeneralSettings().wavelet_duration_bounds
        for name in ORDER:
            run = fr.RUNS[name]
            with self.subTest(run=name):
                Nf, Nt, _ = WDMSettings.adjust_to_even_bins(t_min=lo, t_max=hi, dt=2.5,
                                                            Tobs=run.tobs)
                self.assertEqual((Nf, Nt), (1440, run.nt))

    def test_the_known_values(self):
        want = {  # (days, Nt, stride, crop, leaves, subbands, chunk)
            "3mo": (90, 2160, 60, 20, 10000, 32768, 65536),
            "6mo": (180, 4320, 120, 60, 15000, 8192, 32768),
            "9mo": (270, 6480, 180, 60, 20000, 4096, 16384),
            "12mo": (360, 8640, 240, 60, 20000, 4096, 16384),
            "18mo": (540, 12960, 360, 90, 25000, 2048, 8192),
            "24mo": (720, 17280, 480, 120, 25000, 2048, 8192),
        }
        for name, (d, nt, st, cr, lv, sb, ch) in want.items():
            e = self.ex[name]
            with self.subTest(run=name):
                self.assertEqual(fr.RUNS[name].days, d)
                self.assertEqual(fr.RUNS[name].nt, nt)
                self.assertEqual(e["SIGHET_NT_LAYER"], str(st))
                self.assertEqual(e["EDGE_CROP_WAVELETS"], str(cr))
                self.assertEqual(e["GB_NLEAVES_MAX"], str(lv))
                self.assertEqual(e["GB_N_SUBBANDS"], str(sb))
                self.assertEqual(e["GB_RJ_INMODEL_CHUNK"], str(ch))

    def test_capacity_knobs_are_monotone_in_tobs(self):
        runs = [fr.RUNS[n] for n in ORDER]
        for a, b in zip(runs, runs[1:]):
            self.assertLess(a.days, b.days)
            self.assertLessEqual(a.gb_nleaves_max, b.gb_nleaves_max, (a.name, b.name))
            self.assertGreaterEqual(a.gb_n_subbands, b.gb_n_subbands, (a.name, b.name))
            self.assertGreaterEqual(a.gb_rj_inmodel_chunk, b.gb_rj_inmodel_chunk, (a.name, b.name))
            self.assertLessEqual(a.edge_crop_wavelets, b.edge_crop_wavelets, (a.name, b.name))

    def test_mbhb_ids_follow_the_two_week_rule(self):
        tc = fr.MBHB_TC_DAYS
        self.assertEqual(len(tc), 20)
        # the MBH session table 2026-10-03, verbatim
        for i, d in ((18, 92.0), (5, 104.7), (16, 111.4), (2, 173.3), (12, 243.8), (7, 263.8),
                     (9, 285.9), (4, 286.4), (0, 300.4), (15, 318.0), (3, 336.8), (10, 369.5)):
            self.assertAlmostEqual(tc[i], d, places=1)
        for name in ORDER:
            run = fr.RUNS[name]
            with self.subTest(run=name):
                e = self.ex[name]
                want = ",".join(str(i) for i in sorted(tc) if tc[i] < run.days + 14)
                self.assertEqual(e["MBHB_IDS"], want)
                self.assertEqual(e["MBH_MERGER_TIME_BUFFER"], str(14 * 86400))
                self.assertEqual(e["EMRI_IDS"], "0,1,2,3,4,5,6,7")
                self.assertEqual(e["SOBHB_IDS"], "0,1,2,3,4,5")
                self.assertEqual(e["SOURCE_TYPES"], "COMBINED,GB,VGB,MBHB,EMRI,SOBHB")
        self.assertEqual(self.ex["3mo"]["MBHB_IDS"], "18")              # 5 at 104.7 d is out
        self.assertEqual(self.ex["6mo"]["MBHB_IDS"], "2,5,16,18")
        self.assertEqual(self.ex["9mo"]["MBHB_IDS"], "2,5,7,12,16,18")  # the launched list
        self.assertIn("10", self.ex["12mo"]["MBHB_IDS"].split(","))     # 369.5 d < 374 d
        self.assertNotIn("17", self.ex["18mo"]["MBHB_IDS"].split(","))  # 585.5 d > 554 d
        self.assertEqual(self.ex["24mo"]["MBHB_IDS"], ",".join(map(str, range(20))))

    def test_naming(self):
        stores, jobs = set(), set()
        for name in ORDER:
            run = fr.RUNS[name]
            with self.subTest(run=name):
                t, e = self.tx[name], self.ex[name]
                self.assertEqual(e["BASE_FILE_NAME"], f"gf_prod_{name}")
                self.assertIn(f"\nSTORE_DIR=${{STORE_DIR:-{run.store_dir}}}\n", t)
                self.assertIn(f"#SBATCH --job-name={run.resolved_job_name} ", t)
                self.assertIn(f"#SBATCH --output=/shared/data/global_fit_output/"
                              f"{run.resolved_log_stem}_%j.log", t)
                self.assertIn(f"\nSLURM_LOG=/shared/data/global_fit_output/"
                              f"{run.resolved_log_stem}_${{SLURM_JOB_ID:-manual}}.log\n", t)
                self.assertTrue(run.store_dir.endswith("/"))
                stores.add(run.store_dir)
                jobs.add(run.resolved_job_name)
                if name != "9mo":
                    self.assertIn(f"gf_prod_{name}_{fr.SERIES}/", run.store_dir)
                    self.assertNotIn("_v9", t.split("set -euo pipefail")[1].split("# ----")[0])
        self.assertEqual(len(stores), len(ORDER), "two runs share a store")
        self.assertEqual(len(jobs), len(ORDER), "two runs share a job name")
        # the 9mo keeps the LIVE run's names; the new runs never touch a live store
        self.assertEqual(fr.RUNS["9mo"].store_dir, "/shared/data/global_fit_output/gf_prod_9mo_v9/")
        for name in ORDER:
            if name != "9mo":
                for live in ("gf_prod_9mo_v9/", "gf_prod_6mo_v9_4gpu/", "gf_prod_3mo_v9"):
                    self.assertNotIn(live, fr.RUNS[name].store_dir)

    def test_ngpus_defaults_to_four_everywhere(self):
        d = _Dispatch()
        try:
            for name in ORDER:
                with self.subTest(run=name):
                    lines = d.run(_path(name), {})
                    self.assertIn("--nodes=2", lines)
                    self.assertIn("--gres=gpu:2", lines)
                    self.assertIn("--ntasks=6", lines)      # 4 compute + saver + monitor
                    self.assertIn("--partition=gpu-80-ondemand", lines)
                    self.assertIn("--distribution=cyclic", lines)
                    self.assertIn("--cpus-per-task=10", lines)   # 2 + EMRI_TRAJ_WORKERS 8
                    self.assertTrue(any("NGPUS=4" in l for l in lines if l.startswith("--export=")))
                    two = d.run(_path(name), {"NGPUS": "2"})
                    self.assertIn("--nodes=1", two)
                    self.assertIn("--ntasks=4", two)
                    self.assertIn("--partition=gpu-80-spot", two)
        finally:
            d.tmp.cleanup()


# ---------------------------------------------------------------------------
class ParentChainTest(unittest.TestCase):
    """3mo -> 6mo -> 9mo -> 12mo -> 18mo -> 24mo: each run reads the previous one."""

    @classmethod
    def setUpClass(cls):
        cls.ex = {n: _exports(_path(n)) for n in ORDER}

    def test_the_chain(self):
        self.assertEqual(ORDER, ("3mo", "6mo", "9mo", "12mo", "18mo", "24mo"))
        self.assertEqual(fr.RUNS["3mo"].parent, None)
        self.assertEqual(fr.RUNS["6mo"].parent, "3mo")
        self.assertIsInstance(fr.RUNS["9mo"].parent, fr.Parent)   # the LIVE 6mo v9 store
        self.assertIsInstance(fr.RUNS["12mo"].parent, fr.Parent)  # a COPY of the 9mo folder
        self.assertEqual(fr.RUNS["12mo"].parent.store_dirname, "gf_prod_9mo_v9_copy")
        self.assertEqual(fr.RUNS["12mo"].parent.mbhb_ids, fr.RUNS["9mo"].mbhb_ids)
        self.assertEqual(fr.RUNS["12mo"].parent.days, fr.RUNS["9mo"].days)
        self.assertEqual(fr.RUNS["18mo"].parent, "12mo")
        self.assertEqual(fr.RUNS["24mo"].parent, "18mo")

    def test_every_parented_run_reads_its_parent(self):
        for name in PARENTED:
            run = fr.RUNS[name]
            p = run.resolved_parent()
            with self.subTest(run=name):
                e = self.ex[name]
                self.assertEqual(e["GF_SEED_STORE"], p.seed_store)
                self.assertEqual(e["GB_WARM_START_SOURCE_STORE"], e["GF_SEED_STORE"])
                # 2026-10-22 (Mike): the non-GB sources always start at the
                # catalogue injection -- the parent-run source warm start is
                # OFF by default (opt in with SOURCE_WARM_START_STORE on the line)
                self.assertEqual(e["SOURCE_WARM_START_STORE"], "")
                self.assertEqual(int(e["GB_WARM_START_SOURCE_TOBS"]), p.days * 86400)
                self.assertEqual(e["GB_WARM_START_LAST_K"], "300")
                self.assertTrue(e["GB_WARM_START_COMPONENTS"].endswith(
                    f"/warmstart/{p.store_dirname}_refereed.npz"), e["GB_WARM_START_COMPONENTS"])
                self.assertEqual(e["MBH_SOURCE_WARM_START_IDS"], p.mbhb_ids)
                self.assertEqual(e["EMRI_SOURCE_WARM_START_IDS"], e["EMRI_IDS"])
                self.assertEqual(e["SOBBH_SOURCE_WARM_START_IDS"], e["SOBHB_IDS"])
                self.assertEqual(e["SOURCE_WARM_START_SNR_MIN"], "10")
                self.assertEqual(e["GB_SEARCH_SEED_ITERS"], "3")
                self.assertEqual(e["GB_SEARCH_3_WARM_EVERY"], "5")
                self.assertEqual(e["GB_REPLACE_WARM_PASS"], "1")
                self.assertEqual(e["SOURCE_SEARCH_POSITION"], "after_seed")
                self.assertEqual(e["STAGE_SKIP_SOURCE_SEARCH"], "0")
                # the parent run overlaid on this run's monitor page (2026-10-12)
                self.assertEqual(e["GF_MONITOR_ADD_INFO"],
                                 f"/shared/data/global_fit_output/{p.store_dirname}")
                self.assertNotIn("PSD_START_PARAMS", e)           # the pin comes from the store
                self.assertNotIn("STAGE_FORCE_NOISE_SEARCH", e)
                self.assertNotIn("GB_SEARCH_1_NOISE_MODE", e)
                # the parent's MBHB_IDS are what its launcher exports
                if isinstance(run.parent, str):
                    self.assertEqual(e["MBH_SOURCE_WARM_START_IDS"],
                                     self.ex[run.parent]["MBHB_IDS"])
                    self.assertEqual(e["GF_SEED_STORE"],
                                     fr.RUNS[run.parent].store_dir
                                     + self.ex[run.parent]["BASE_FILE_NAME"] + "_testing.h5")

    def test_the_9mo_parent_is_the_live_6mo_store_and_the_6mo_parent_is_the_flagship_3mo(self):
        self.assertEqual(self.ex["9mo"]["GF_SEED_STORE"],
                         "/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5")
        self.assertEqual(self.ex["9mo"]["MBH_SOURCE_WARM_START_IDS"], "2,5,16,18")
        self.assertEqual(self.ex["6mo"]["GF_SEED_STORE"],
                         "/shared/data/global_fit_output/gf_prod_3mo_flagship/gf_prod_3mo_testing.h5")
        self.assertEqual(self.ex["6mo"]["MBH_SOURCE_WARM_START_IDS"], "18")
        # the 12mo reads a COPY of the 9mo folder, never the live store (Mike 2026-10-13)
        self.assertEqual(self.ex["12mo"]["GF_SEED_STORE"],
                         "/shared/data/global_fit_output/gf_prod_9mo_v9_copy/gf_prod_9mo_testing.h5")
        self.assertEqual(self.ex["12mo"]["GF_MONITOR_ADD_INFO"],
                         "/shared/data/global_fit_output/gf_prod_9mo_v9_copy")
        self.assertEqual(self.ex["12mo"]["MBH_SOURCE_WARM_START_IDS"], self.ex["9mo"]["MBHB_IDS"])
        self.assertNotIn("gf_prod_9mo_v9/", _text("12mo").split("set -euo pipefail")[1])


# ---------------------------------------------------------------------------
class NoParentTest(unittest.TestCase):
    """The 3mo: the first run of the chain (the 3mo v9 2gpu twin's lessons)."""

    GALFOR_BASE_AMP = 1.436180605904e-44
    GALFOR_BASE_FK = 2.533915614978e-03

    def setUp(self):
        self.e = _exports(_path("3mo"))
        self.t = _text("3mo")

    def test_no_warm_start_no_seed_stage_no_seed_store(self):
        self.assertEqual(self.e.get("GB_WARM_START_COMPONENTS", "<unset>"), "")
        self.assertEqual(self.e["GB_SEARCH_SEED_ITERS"], "0")
        self.assertEqual(self.e["GB_REPLACE_WARM_PASS"], "0")
        for knob in ("GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE", "GB_WARM_START_SOURCE_TOBS",
                     "GB_WARM_START_LAST_K", "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES",
                     "GB_SEARCH_3_WARM_EVERY", "MBH_SOURCE_WARM_START_IDS",
                     "EMRI_SOURCE_WARM_START_IDS", "SOBBH_SOURCE_WARM_START_IDS"):
            self.assertNotIn(knob, self.e, knob)
        self.assertEqual(self.e.get("SOURCE_WARM_START_STORE", "<unset>"), "")
        self.assertNotIn("GF_MONITOR_ADD_INFO", self.e)     # no previous run to overlay
        self.assertNotIn("[V9-SEED]", "\n".join(executable_lines(self.t)))
        self.assertNotIn("noise_pin", self.t.split("set -euo pipefail")[1])

    def test_the_sources_seed_first_from_truth(self):
        self.assertEqual(self.e["SOURCE_SEARCH_POSITION"], "first")
        self.assertEqual(self.e["STAGE_SKIP_SOURCE_SEARCH"], "0")
        for b in ("MBH", "EMRI", "SOBBH", "VGB"):
            self.assertEqual(self.e[f"{b}_START_FACTOR"], "0.0")

    def test_the_noise_starts_pinned_and_the_noise_stage_is_forced(self):
        self.assertEqual(self.e["PSD_START_PARAMS"], "1.500004011496e-11,3.000107254658e-15")
        self.assertEqual(self.e["STAGE_FORCE_NOISE_SEARCH"], "1")
        self.assertEqual(self.e["STAGE_SKIP_NOISE_VGB"], "1")
        self.assertEqual(self.e["GB_SEARCH_1_NOISE_MODE"], "cycle_end")
        self.assertEqual(self.e["GB_SEARCH_2_NOISE_MODE"], "cycle_end")
        self.assertEqual(self.e["NOISE_SEARCH_CHECKS"], "5")
        self.assertEqual(self.e["MAXLOGL_PER_WALKER"], "1")
        self.assertEqual(self.e["MAXLOGL_FREEZE_CONVERGED"], "0")
        self.assertEqual(self.e["MAXLOGL_TOL"], "20")
        self.assertNotIn("STAGE_NOISE_PSD_ONLY", self.e)
        self.assertNotIn("GB_SEARCH_SAMPLE_NOISE_ALL_STAGES", self.e)
        self.assertEqual(self.e["MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM"], "1")
        self.assertEqual(self.e["UNEQUAL_ARM"], "1")

    def test_the_foreground_starts_at_the_inflated_fitted_tanh(self):
        vals = [float(x) for x in self.e["GALFOR_START_PARAMS"].split(",")]
        self.assertEqual(len(vals), 5)
        amp, fk, alpha, f_1, f_2 = vals
        self.assertAlmostEqual(amp / self.GALFOR_BASE_AMP, 1.25, places=9)
        self.assertAlmostEqual(fk / self.GALFOR_BASE_FK, 1.10, places=9)
        self.assertEqual(alpha, 5.0)
        self.assertAlmostEqual(f_2, 1.405721657329e-03, places=15)
        for name, v, lo, hi in (("amp", amp, 1e-47, 1e-41),
                                ("fk", fk, 0.8e-3, 1e-2),
                                ("f_1", f_1, 1e-4, 1e-2),
                                ("f_2", f_2, 0.8e-3, 1e-2)):
            self.assertGreater(v, lo, name)
            self.assertLess(v, hi, f"{name} is ON or past its prior rail")
        # the parented runs keep the uninflated offline estimate (the store pin wins)
        for name in PARENTED:
            p = [float(x) for x in _exports(_path(name))["GALFOR_START_PARAMS"].split(",")]
            self.assertAlmostEqual(p[0], self.GALFOR_BASE_AMP)
            self.assertAlmostEqual(p[1], self.GALFOR_BASE_FK)

    def test_the_driver_reads_the_overlay_knobs(self):
        src = open(DRIVER).read()
        for knob in ("STAGE_FORCE_NOISE_SEARCH", "STAGE_SKIP_NOISE_VGB", "GB_SEARCH_{N}_NOISE_MODE",
                     "SOURCE_SEARCH_POSITION", "GB_SEARCH_SEED_ITERS"):
            self.assertIn(knob, src, knob)


# ---------------------------------------------------------------------------
class EveryOtherKnobIsTheNineMonthsTest(unittest.TestCase):
    """Each run's resolved exports differ from the 9mo's ONLY in the declared sets."""

    TOBS_DRIVEN = {"TOBS_TARGET", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS", "GB_NLEAVES_MAX",
                   "GB_N_SUBBANDS", "GB_RJ_INMODEL_CHUNK", "MBHB_IDS", "BASE_FILE_NAME"}
    PARENT_DRIVEN = {"GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE", "GB_WARM_START_SOURCE_TOBS",
                     "GB_WARM_START_COMPONENTS", "SOURCE_WARM_START_STORE",
                     "MBH_SOURCE_WARM_START_IDS", "GF_MONITOR_ADD_INFO"}
    NO_PARENT = {"GB_WARM_START_LAST_K", "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES",
                 "EMRI_SOURCE_WARM_START_IDS", "SOBBH_SOURCE_WARM_START_IDS",
                 "SOURCE_WARM_START_SNR_MIN", "GB_SEARCH_SEED_ITERS", "GB_SEARCH_3_WARM_EVERY",
                 "GB_REPLACE_WARM_PASS", "SOURCE_SEARCH_POSITION", "PSD_START_PARAMS",
                 "GALFOR_START_PARAMS", "STAGE_FORCE_NOISE_SEARCH", "STAGE_SKIP_NOISE_VGB",
                 "GB_SEARCH_1_NOISE_MODE", "NOISE_SEARCH_CHECKS", "MAXLOGL_PER_WALKER",
                 "MAXLOGL_FREEZE_CONVERGED",
                 # 2026-10-13: the GB warm-start SELF refit lives in the parent
                 # warm-start branch of the template -- it rebuilds the parent-
                 # derived mixture from the run's own rows and the warm moves
                 # only exist with GB_WARM_START_COMPONENTS, so a no-parent run
                 # has nothing to refit and does not export the block.
                 "GB_WARM_SELF", "GB_WARM_SELF_AT_ENTRY", "GB_WARM_SELF_EVERY_SEARCH",
                 "GB_WARM_SELF_EVERY_PE", "GB_WARM_SELF_LAST_K",
                 "GB_WARM_SELF_SPLIT_SEARCH", "GB_WARM_SELF_SPLIT_PE",
                 "GB_WARM_SELF_CONSENSUS"}
    MUST_DIFFER = {"TOBS_TARGET", "SIGHET_NT_LAYER", "MBHB_IDS", "BASE_FILE_NAME",
                   "GF_SEED_STORE", "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_COMPONENTS",
                   "MBH_SOURCE_WARM_START_IDS", "GF_MONITOR_ADD_INFO"}

    @classmethod
    def setUpClass(cls):
        cls.ex = {n: _exports(_path(n)) for n in ORDER}

    def test_resolved_exports(self):
        nine = self.ex["9mo"]
        for name in ORDER:
            if name == "9mo":
                continue
            allowed = self.TOBS_DRIVEN | self.PARENT_DRIVEN
            # a level-2 override is declared by definition, on either side (the
            # 9mo keeps the chunked GB likelihood, Mike 2026-10-24)
            allowed |= set(fr.RUNS[name].overrides) | set(fr.RUNS["9mo"].overrides)
            if fr.RUNS[name].parent is None:
                allowed |= self.NO_PARENT
            with self.subTest(run=name):
                e = self.ex[name]
                keys = (set(e) | set(nine)) - {"_", "SHLVL", "PWD"}
                drift = {k: (nine.get(k), e.get(k)) for k in sorted(keys - allowed)
                         if nine.get(k) != e.get(k)}
                self.assertEqual(drift, {}, f"undeclared 9mo -> {name} drift: {drift}")
                same = sorted(k for k in self.MUST_DIFFER if nine.get(k) == e.get(k))
                self.assertEqual(same, [], f"{name}: declared deltas that do not differ: {same}")

    def test_the_v930_production_defaults_hold_everywhere(self):
        line = {"MIDIT_CHECKPOINT": "0", "MBH_NTEMPS": "8", "EMRI_NTEMPS": "12",
                "EMRI_TRAJ_WORKERS": "8", "MBH_WINDOW_DECIMATE": "2",
                "SOBBH_EIGEN_INFO": "gram", "EMRI_EIGEN_INFO": "gram", "MBH_EIGEN_INFO": "gram",
                "PSD_EIGEN_INFO": "fisher", "GALFOR_EIGEN_INFO": "fisher",
                "SOBBH_CHECK_LL": "0", "MBH_CHECK_LL": "0", "EMRI_CHECK_LL": "0",
                "GF_PERSIST_STAGE_START": "1", "GB_LADDER_HOT_RATIO": "1.35",
                "GB_LADDER_HOT_FROM": "10", "GB_LADDER_PIN_LAST": "0",
                "MBH_EIGEN_REFRESH": "10", "EMRI_EIGEN_REFRESH": "10",
                "GB_SEARCH_STAGES": "1,2", "GB_SEARCH_1_SOURCE_EVERY": "1",
                "GB_SEARCH_2_SOURCE_EVERY": "1", "GB_SEARCH_2_NOISE_MODE": "cycle_end",
                "GB_SEARCH_2_RESET_VALVES": "1", "STAGE_REPLICA_PE": "0", "GALFOR_RATCHET": "0",
                "FSTAT_STAGEB_NSKY_MAX": "512", "FSTAT_STAGEB_GROUP_MAX_GB": "1.0",
                # 2026-10-12: PE F-stat refit 150, the two PE F-stat moves, MBH sky-mode hops
                "GB_FSTAT_REFIT_EVERY_PE": "150", "GB_FSTAT_REFIT_EVERY_PE_GB_FREE": "0",
                "GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE": "6.5",
                # 2026-10-18 (Mike): split 0.5 -> 0.9, warm 0.1 -> 0.4
                "GB_PE_RJ_FSTAT_RESIDUAL_SPLIT": "0.9", "GB_PE_RJ_WARM_FRACTION": "0.4",
                "MBH_SKY_MOVE_BOTH_WEIGHT": "0.02", "MBH_SKY_MOVE_LAT_WEIGHT": "0.04",
                "MBH_SKY_MOVE_LONG_WEIGHT": "0.02",
                # 2026-10-13: the eigen jump mixture (10 % of the draws at 3 sigma)
                "EMRI_EIGEN_JUMP_MIX": "0.1", "EMRI_EIGEN_JUMP_FACTOR": "3.0",
                "MBH_EIGEN_JUMP_MIX": "0.1", "MBH_EIGEN_JUMP_FACTOR": "3.0",
                "SOBBH_EIGEN_JUMP_MIX": "0.1", "SOBBH_EIGEN_JUMP_FACTOR": "3.0",
                # 2026-10-14: per-axis step-scale adaptation of the source eigen tables
                "EMRI_EIGEN_AXIS_ADAPT": "1", "MBH_EIGEN_AXIS_ADAPT": "1",
                "SOBBH_EIGEN_AXIS_ADAPT": "1",
                # 2026-10-15: pooled / tempered / factor-capped adaptation
                "EMRI_EIGEN_ADAPT_POOL": "1", "MBH_EIGEN_ADAPT_POOL": "1",
                "SOBBH_EIGEN_ADAPT_POOL": "1",
                "EMRI_EIGEN_TEMPER_SIGMAS": "1", "MBH_EIGEN_TEMPER_SIGMAS": "1",
                "SOBBH_EIGEN_TEMPER_SIGMAS": "1",
                "EMRI_EIGEN_ADAPT_MAX_FACTOR": "100", "MBH_EIGEN_ADAPT_MAX_FACTOR": "100",
                "SOBBH_EIGEN_ADAPT_MAX_FACTOR": "100",
                # 2026-10-18: the EMRI sampling audit (docs/emri-sampling-audit.md)
                "EMRI_EIGEN_AXIS_ACC_LOG": "1", "MBH_EIGEN_AXIS_ACC_LOG": "1",
                "SOBBH_EIGEN_AXIS_ACC_LOG": "1", "EMRI_EIGEN_AXIS_ACC_TUNE": "1",
                "EMRI_EIGEN_AXIS_ACC_MIN": "0.1", "EMRI_EIGEN_AXIS_ACC_MAX": "0.5",
                "EMRI_EIGEN_AXIS_ACC_FACTOR": "2", "EMRI_DE_JUMP_MIX": "0.2",
                "EMRI_DE_JUMP_MODE_FRAC": "0.5", "EMRI_LADDER_REANCHOR_NATS": "100",
                # 2026-10-18 (Mike, "do this for the other sources"): the
                # controller on for MBH / SOBBH, the jump at 0.1 / 0.2, the
                # SOBBH guard 20 -> 60
                "MBH_EIGEN_AXIS_ACC_TUNE": "1", "SOBBH_EIGEN_AXIS_ACC_TUNE": "1",
                "MBH_DE_JUMP_MIX": "0.1", "SOBBH_DE_JUMP_MIX": "0.2",
                "SOBBH_LADDER_REANCHOR_NATS": "60",
                # 2026-10-19 (Mike, "move in the LISA reference frame"): the MBH
                # inner moves propose in the LISA frame
                "MBH_PROPOSAL_FRAME": "lisa",
                "GALFOR_FK_PRIOR": "0.8e-3,1e-2", "GALFOR_F1_PRIOR": "1e-4,1e-2",
                "GALFOR_F2_PRIOR": "0.8e-3,1e-2", "NWALKERS": "4", "GB_NTEMPS": "24",
                "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM": "1", "UNEQUAL_ARM": "1"}
        for name in ORDER:
            with self.subTest(run=name):
                for knob, want in line.items():
                    self.assertEqual(self.ex[name].get(knob), want, f"{name}: {knob}")
                for knob in ("GB_FSTAT_FORCE_REFIT", "GB_FSTAT_PE_REF"):
                    self.assertNotIn(knob, self.ex[name], knob)


# ---------------------------------------------------------------------------
class LevelsTest(unittest.TestCase):
    """The three adjustment levels through build.py."""

    def test_set_export_rewrites_each_form(self):
        text = ("export A=1   # c\nexport B=${B:-2}\nexport C=${C-3}   # d\n"
                'export D="x y"\nexport E=5\n')
        text = fb.set_export(text, "A", "9")
        text = fb.set_export(text, "B", "8")
        text = fb.set_export(text, "C", "")
        text = fb.set_export(text, "D", "z")
        self.assertEqual(text, ("export A=9   # c\nexport B=${B:-8}\nexport C=${C-}   # d\n"
                                'export D="z"\nexport E=5\n'))
        with self.assertRaises(KeyError):
            fb.set_export(text, "NOPE", "1")
        with self.assertRaises(KeyError):
            fb.set_export(text + "export A=2\n", "A", "3")      # exported twice

    def test_level_3_one_off(self):
        one = fb.build("9mo", {"GB_NTEMPS": "12", "GB_INMODEL_CONVERGE_ITERS": "100"})
        self.assertIn("\nexport GB_NTEMPS=12\n", one)
        self.assertIn("\nexport GB_INMODEL_CONVERGE_ITERS=100\n", one)
        self.assertIn("ONE-OFF LAUNCHER (level 3)", one)
        # everything else is the committed 9mo
        a = [l for l in executable_lines(_text("9mo")) if not l.startswith("export GB_NTEMPS=")
             and not l.startswith("export GB_INMODEL_CONVERGE_ITERS=")]
        b = [l for l in executable_lines(one) if not l.startswith("export GB_NTEMPS=")
             and not l.startswith("export GB_INMODEL_CONVERGE_ITERS=")]
        self.assertEqual(a, b)
        with self.assertRaises(KeyError):
            fb.build("3mo", {"GB_WARM_START_LAST_K": "10"})   # not an export of the 3mo

    def test_level_2_overrides_and_extra_exports(self):
        import dataclasses
        base = fr.RUNS["12mo"]
        trial = dataclasses.replace(base, name="12mo_trial",
                                    overrides={"NWALKERS": "8", "GB_NTEMPS": "12"},
                                    extra_exports=("export FLAGSHIP_TRIAL_KNOB=1",))
        fr.RUNS["12mo_trial"] = trial
        try:
            text = fb.build("12mo_trial")
        finally:
            del fr.RUNS["12mo_trial"]
        e = {}
        for l in text.split("\n"):
            m = re.match(r"^export ([A-Z0-9_]+)=(.*)$", l)
            if m:
                e[m.group(1)] = m.group(2)
        self.assertEqual(e["NWALKERS"], "${NWALKERS:-8}")
        self.assertEqual(e["GB_NTEMPS"], "12")
        self.assertEqual(e["FLAGSHIP_TRIAL_KNOB"], "1")
        self.assertIn("OVERRIDES (level 2", text)
        self.assertIn("gf_prod_12mo_trial_flagship/", text)
        self.assertIn(text.index("export FLAGSHIP_TRIAL_KNOB=1"), range(
            text.index("[SOBBH-PREFLIGHT]"), text.index("FRESH-RUN GUARD")))

    def test_the_run_table_refuses_a_bad_crop_or_an_unknown_parent(self):
        import dataclasses
        with self.assertRaises(ValueError):
            dataclasses.replace(fr.RUNS["24mo"], edge_crop_wavelets=60).check()
        with self.assertRaises(ValueError):
            dataclasses.replace(fr.RUNS["24mo"], parent="99mo").check()
        with self.assertRaises(ValueError):
            dataclasses.replace(fr.RUNS["24mo"], ngpus_default=8).check()

    def test_the_launch_line_reaches_the_overridable_knobs(self):
        # the V9-30 line knobs keep the ${K:-v} form in every launcher
        for name in ORDER:
            src = _text(name)
            with self.subTest(run=name):
                for knob, want in (("MIDIT_CHECKPOINT", "0"), ("MBH_NTEMPS", "8"),
                                   ("NWALKERS", "4"), ("GB_SEARCH_STAGES", "1,2"),
                                   ("FSTAT_STAGEB_NSKY_MAX", "512")):
                    self.assertRegex(src, rf"(?m)^export {knob}=\$\{{{knob}:-{re.escape(want)}\}}")


# ---------------------------------------------------------------------------
def _lisatools_importable():
    try:
        import lisatools.globalfit  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_lisatools_importable(), "lisatools (the driver's imports) not importable")
@unittest.skipIf(os.environ.get("FLAGSHIP_SKIP_RECIPE") == "1", "FLAGSHIP_SKIP_RECIPE=1")
class RecipeCompositionTest(unittest.TestCase):
    """``run_combined_staged.py --print-recipe`` under each launcher's exports."""

    @classmethod
    def _recipe(cls, name, extra):
        src = _text(name)
        lines = [l for l in src.split("\n") if re.match(r"^export [A-Z0-9_]+=", l)]
        script = ("set -a\nSTORE_DIR=/tmp/flagship_recipe_probe/\n" + "\n".join(lines) + "\n"
                  + "".join(f"{k}={v}\n" for k, v in extra.items())
                  + f"exec {sys.executable} {DRIVER} --print-recipe\n")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env,
                           cwd=ROOT, timeout=600)
        if r.returncode != 0:
            raise AssertionError(f"{name}: --print-recipe exited {r.returncode}\n"
                                 f"{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
        return r.stdout

    def test_every_launcher_composes_the_intended_recipe(self):
        for name in ORDER:
            run = fr.RUNS[name]
            with self.subTest(run=name):
                extra = {}
                if run.parent is not None:
                    # on the cluster the seed store supplies this pin before the
                    # driver runs; without it the driver keeps the noise stages
                    extra["PSD_START_PARAMS"] = "1.500004011496e-11,3.000107254658e-15"
                out = self._recipe(name, extra)
                stages = re.findall(r"(?m)^stage \d+/\d+\s+(\w+)\s+kind=", out)
                if run.parent is not None:
                    self.assertEqual(stages, ["gb_search_seed", "source_search", "gb_search_1",
                                              "gb_search_2", "full_pe"])
                else:
                    self.assertEqual(stages, ["source_search", "noise_search", "gb_search_1",
                                              "gb_search_2", "full_pe"])
                mbh = re.search(r"armed sources: .*mbh=\[([^\]]*)\]", out).group(1)
                self.assertEqual(mbh.replace(" ", ""), run.mbhb_ids)
                blocks = re.split(r"(?m)^stage \d+/\d+\s+", out)[1:]
                by = {b.split()[0]: b for b in blocks}
                self.assertIn("SAMPLED at the END of the cycle", by["gb_search_2"])
                if run.parent is not None:
                    self.assertIn("noise:    FIXED", by["gb_search_1"])
                    self.assertIn("rj_warm_search", by["gb_search_1"])
                else:
                    self.assertIn("SAMPLED at the END of the cycle", by["gb_search_1"])
                    self.assertNotIn("rj_warm", out)
                self.assertIn("sobbh_pe/mbh_pe/emri_pe EVERY iteration", by["gb_search_1"])


if __name__ == "__main__":
    unittest.main()
