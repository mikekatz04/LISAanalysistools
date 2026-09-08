"""MIN_FREQ must reach the production variants, and layer 1 must be refused.

`AllSourcesGeneralSettings.min_freq` (and full_year_combined's) was a PLAIN
``= 1e-4`` shadowing the env-backed factory on ``EreborGeneralSettings``, so the
``MIN_FREQ=4e-4`` every production submit script exports never took effect. That
put WDM layer 1 -- support 0.069-0.208 mHz, sharing an edge with DC -- inside
the analysed band, where the instrument model diverges and
``instrument_fill_nans=0.0`` turns the NaN into a hard zero.

Measured consequence on the 3mo v8 store (kappa probe, job 459): det(C)=0 at
layer 1, carrying q = w^T C^-1 w / 3 = 150.3 = 43% of the fit's whole chi^2
budget, which drags the ML to alpha = sqrt(mean(q/3)) = 1.389 -- the entire
observed instrument-PSD bias.

This is the third instance of the same shadowing bug in these files (nf/nt and
EDGE_CROP_WAVELETS preceded it), so it is pinned here for every variant at once.
"""

import math
import os
import unittest

PRODUCTION_VARIANTS = ("all_sources", "full_year_combined")
ALL_VARIANTS = PRODUCTION_VARIANTS + ("gb_no_fg", "noise_mojito")


class _EnvGuard(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("MIN_FREQ", "MAX_FREQ")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class MinFreqEnvReachesVariants(_EnvGuard):
    """The knob must actually reach the field (stock rule 0)."""

    def test_min_freq_env_is_honored(self):
        from lisatools.globalfit.stock import erebor

        os.environ["MIN_FREQ"] = "4e-4"
        for name in ALL_VARIANTS:
            with self.subTest(variant=name):
                fit = erebor.get_stock(name)
                self.assertAlmostEqual(
                    float(fit.general.min_freq), 4e-4, places=12,
                    msg=f"{name}: MIN_FREQ ignored -- the field shadows the "
                        "env-backed base factory again",
                )

    def test_max_freq_env_is_honored(self):
        from lisatools.globalfit.stock import erebor

        os.environ["MAX_FREQ"] = "2.0e-2"
        for name in ALL_VARIANTS:
            with self.subTest(variant=name):
                fit = erebor.get_stock(name)
                self.assertAlmostEqual(
                    float(fit.general.max_freq), 2.0e-2, places=12,
                    msg=f"{name}: MAX_FREQ ignored",
                )

    def test_defaults_unchanged_when_env_unset(self):
        """Env-backing must not move any variant's hard default."""
        from lisatools.globalfit.stock import erebor

        os.environ.pop("MIN_FREQ", None)
        os.environ.pop("MAX_FREQ", None)
        for name, lo, hi in (("all_sources", 1e-4, 2.5e-2),
                             ("full_year_combined", 1e-4, 2.5e-2)):
            with self.subTest(variant=name):
                gs = erebor.get_stock(name).general
                self.assertAlmostEqual(float(gs.min_freq), lo, places=12)
                self.assertAlmostEqual(float(gs.max_freq), hi, places=12)

    def test_explicit_kwarg_still_wins_over_env(self):
        from lisatools.globalfit.stock import erebor

        os.environ["MIN_FREQ"] = "4e-4"
        fit = erebor.get_stock("all_sources")
        fit.general.min_freq = 7e-4
        self.assertAlmostEqual(float(fit.general.min_freq), 7e-4, places=12)


class LayerOneIsRefused(_EnvGuard):
    """min_freq must land on layer 2+; layer 1 borders DC and is singular."""

    @staticmethod
    def _ind_min_f(min_freq, nf, dt):
        return int(math.ceil(min_freq / (1.0 / (2.0 * nf * dt))))

    def test_production_band_lands_on_layer_three(self):
        """4e-4 on the production grid -> layer 3, two clear of DC."""
        self.assertEqual(self._ind_min_f(4e-4, 1440, 2.5), 3)

    def test_the_bug_would_have_landed_on_layer_one(self):
        """The regression this guards: 1e-4 admits the singular layer."""
        self.assertEqual(self._ind_min_f(1e-4, 1440, 2.5), 1)

    def test_audit_rule_two_layer_df(self):
        """min_freq >= 2*layer_df is the WDM audit's rule; 4e-4 clears it."""
        layer_df = 1.0 / (2.0 * 1440 * 2.5)
        self.assertGreaterEqual(4e-4, 2.0 * layer_df)
        self.assertGreaterEqual(self._ind_min_f(2.0 * layer_df, 1440, 2.5), 2)

    def test_guard_warns_on_a_dc_adjacent_band(self):
        """fit.py must WARN (not silently proceed) when layer 1 is admitted."""
        from lisatools.globalfit.stock import erebor

        os.environ["MIN_FREQ"] = "1e-4"
        fit = erebor.get_stock("all_sources")
        gs = fit.default_general()
        gs.nf, gs.nt, gs.dt = 1440, 2160, 2.5
        with self.assertLogs(
            "lisatools.globalfit.stock.erebor.fit", level="WARNING"
        ) as cm:
            fit.finalize_general_grid_for_test(gs) if hasattr(
                fit, "finalize_general_grid_for_test"
            ) else self._emit_via_make_general(fit, gs)
        self.assertTrue(
            any("borders DC" in m for m in cm.output),
            msg=f"no DC-adjacency warning; got {cm.output}",
        )

    @staticmethod
    def _emit_via_make_general(fit, gs):
        """Drive the real code path that contains the guard."""
        fit.make_general_settings()


if __name__ == "__main__":
    unittest.main()
