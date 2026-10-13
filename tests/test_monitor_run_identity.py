"""The monitor page's run identity (banner label, run kind, arm-cache tag).

``lisatools.globalfit.monitor._identity.run_identity`` replaced the inline
name ladders of ``_generator.py`` and ``scripts/diagnostics/gf_monitor_gen_lean.py``
on 2026-10-07. Two things are pinned:

* every HISTORICAL name resolves exactly as before (their ``gf_arm_<tag>.npz``
  caches already exist on the cluster pages, so a tag that moved would leave
  a duplicate overlay arm behind);
* a NEW versioned name -- the 9mo v9 run, the 3mo v9 twin, the 1yr v9 -- gets
  its own banner and arm tag instead of the ``("3-Month", "3mo")`` / ``v2``
  fall-through that silently overwrote ``gf_arm_v2.npz`` four times (1yr
  2026-08-22, v7 08-26, v8 09-02, 3mo v9 09-28).

The two generators must consume the helper (a text pin: putting the ladder
back inline in either file is the regression).
"""
import os
import unittest

from lisatools.globalfit.monitor._identity import arm_tag, run_identity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GENERATOR = os.path.join(ROOT, "src", "lisatools", "globalfit", "monitor", "_generator.py")
LEAN = os.path.join(ROOT, "scripts", "diagnostics", "gf_monitor_gen_lean.py")


class RunIdentityTest(unittest.TestCase):
    # (store directory basename, RUN_LABEL, RUN_KIND) -- the historical ladder, verbatim
    HISTORICAL = (
        ("gf_prod_23mo", "23-Month", "23mo"),
        ("gf_prod_6mo_v9_4gpu", "6-Month", "6mo"),          # deliberately the plain 6mo tag
        ("gf_prod_6mo_v8", "6-Month", "6mo"),
        ("gf_prod_3mo_v4", "3-Month v4", "3mo_v4"),
        ("gf_prod_3mo_v3", "3-Month v3", "3mo_v3"),
        ("gf_prod_1yr_v8_4gpu", "1-Year v8", "1yr_v8"),
        ("gf_prod_1yr", "1-Year v5", "1yr_v5"),
        ("gf_prod_3mo_v5", "3-Month v5", "3mo_v5"),
        ("gf_prod_3mo_v6", "3-Month v6", "3mo_v6"),
        ("gf_prod_3mo_v7", "3-Month v7", "3mo_v7"),
        ("gf_prod_3mo_v8_10walkers", "3-Month v8 · 10 Walkers", "3mo_v8_10w"),
        ("gf_prod_3mo_v8", "3-Month v8", "3mo_v8"),
        ("gf_prod_3mo", "3-Month", "3mo"),
    )

    # the names that used to fall through (or mislabel) and now get their own identity
    NEW = (
        ("gf_prod_9mo_v9", "9-Month v9", "9mo_v9"),
        ("gf_prod_3mo_v9_2gpu", "3-Month v9", "3mo_v9"),
        ("gf_prod_1yr_v9", "1-Year v9", "1yr_v9"),
        ("gf_prod_9mo_v10_4gpu", "9-Month v10", "9mo_v10"),
        ("gf_prod_2yr_v9", "2-Year v9", "2yr_v9"),
    )

    # the flagship series (scripts/flagship, 2026-10-11): its own tag per run
    FLAGSHIP = (
        ("gf_prod_3mo_flagship", "3-Month flagship", "3mo_flagship"),
        ("gf_prod_6mo_flagship", "6-Month flagship", "6mo_flagship"),
        ("gf_prod_12mo_flagship", "12-Month flagship", "12mo_flagship"),
        ("gf_prod_18mo_flagship", "18-Month flagship", "18mo_flagship"),
        ("gf_prod_24mo_flagship", "24-Month flagship", "24mo_flagship"),
        ("gf_prod_2yr_flagship", "2-Year flagship", "2yr_flagship"),
    )

    def test_the_flagship_series_gets_its_own_identity(self):
        for base, label, kind in self.FLAGSHIP:
            with self.subTest(base=base):
                self.assertEqual(run_identity(base), (label, kind))
                self.assertEqual(arm_tag(kind), kind)
        # never the live 6mo's arm cache, never the v2 fall-through
        self.assertNotEqual(run_identity("gf_prod_6mo_flagship")[1], "6mo")
        self.assertNotEqual(arm_tag(run_identity("gf_prod_3mo_flagship")[1]), "v2")
        self.assertNotEqual(run_identity("gf_prod_12mo_flagship")[1], "3mo")
        # the flagship 9mo keeps the live run's store and therefore its identity
        self.assertEqual(run_identity("gf_prod_9mo_v9"), ("9-Month v9", "9mo_v9"))
        # the series tag must be a whole word
        self.assertEqual(run_identity("gf_prod_3mo_flagshipx"), ("3-Month", "3mo"))

    def test_every_historical_name_resolves_as_before(self):
        for base, label, kind in self.HISTORICAL:
            with self.subTest(base=base):
                self.assertEqual(run_identity(base), (label, kind))

    def test_new_versioned_names_get_their_own_identity(self):
        for base, label, kind in self.NEW:
            with self.subTest(base=base):
                self.assertEqual(run_identity(base), (label, kind))

    def test_the_9mo_run_never_lands_on_the_v2_arm_cache(self):
        # the clobber: RUN_KIND "3mo" -> arm tag "v2" -> gf_arm_v2.npz overwritten
        for base in ("gf_prod_9mo_v9", "gf_prod_3mo_v9_2gpu", "gf_prod_1yr_v9"):
            with self.subTest(base=base):
                _, kind = run_identity(base)
                self.assertNotEqual(kind, "3mo")
                self.assertNotEqual(arm_tag(kind), "v2")

    def test_a_trailing_path_separator_is_harmless_for_the_callers(self):
        # the generators normpath + basename before calling; the helper itself
        # only sees the basename, so a full path must not be fed to it
        base = os.path.basename(os.path.normpath("/shared/data/global_fit_output/gf_prod_9mo_v9/"))
        self.assertEqual(run_identity(base), ("9-Month v9", "9mo_v9"))

    def test_the_23mo_name_is_not_read_as_a_3mo_variant(self):
        self.assertEqual(run_identity("gf_prod_23mo_v1"), ("23-Month", "23mo"))


class ArmTagTest(unittest.TestCase):
    def test_legacy_kinds_keep_their_historical_file_names(self):
        self.assertEqual(arm_tag("3mo"), "v2")
        self.assertEqual(arm_tag("3mo_v3"), "v3")
        self.assertEqual(arm_tag("3mo_v4"), "v4")

    def test_every_other_kind_is_its_own_tag(self):
        for kind in ("3mo_v8", "3mo_v8_10w", "6mo", "1yr_v8", "9mo_v9", "3mo_v9"):
            self.assertEqual(arm_tag(kind), kind)

    def test_the_lean_generators_env_override_wins(self):
        self.assertEqual(arm_tag("9mo_v9", "custom"), "custom")
        self.assertEqual(arm_tag("9mo_v9", ""), "9mo_v9")
        self.assertEqual(arm_tag("9mo_v9", None), "9mo_v9")


class GeneratorsUseTheHelperTest(unittest.TestCase):
    def test_both_generators_call_run_identity_and_arm_tag(self):
        for path in (GENERATOR, LEAN):
            with self.subTest(path=os.path.basename(path)):
                src = open(path).read()
                self.assertIn("run_identity as _run_identity", src)
                self.assertIn("RUN_LABEL, RUN_KIND = _run_identity(_base)", src)
                self.assertIn("_arm_tag(RUN_KIND", src)
                # the inline ladder is gone: its first branch and its
                # fall-through must not be in the file
                self.assertNotIn('elif "6mo" in _base:', src)
                self.assertNotIn('RUN_LABEL, RUN_KIND = "3-Month", "3mo"', src)
                self.assertNotIn('{"3mo_v3": "v3", "3mo_v4": "v4", "3mo": "v2"}', src)


if __name__ == "__main__":
    unittest.main()
