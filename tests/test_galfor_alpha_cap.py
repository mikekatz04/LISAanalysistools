"""The galfor alpha-cap override (GALFOR_ALPHA_MAX / alpha_max kwarg).

Diagnostic 2026-09-04: the galactic-foreground slope index alpha rails at its
stock prior cap 5.0 in the 3-month v8 run while the instrument PSD is biased
~1.4x. This knob raises ONLY the alpha upper bound so the slope can explore;
the default must leave every other prior bound bit-identical.
"""
import os
import unittest

from lisatools.globalfit.stock.erebor.noise import galfor_prior_dict, GALFOR_BASIS

IA = GALFOR_BASIS.index("alpha")


class GalforAlphaCapTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("GALFOR_ALPHA_MAX", None)

    def tearDown(self):
        os.environ.pop("GALFOR_ALPHA_MAX", None)
        if self._saved is not None:
            os.environ["GALFOR_ALPHA_MAX"] = self._saved

    def test_default_cap_is_five(self):
        self.assertEqual(galfor_prior_dict()[IA].maximum, 5.0)

    def test_kwarg_widens_only_upper(self):
        d = galfor_prior_dict(alpha_max=20.0)
        self.assertEqual(d[IA].maximum, 20.0)
        self.assertEqual(d[IA].minimum, 0.001)  # lower untouched

    def test_env_widens(self):
        os.environ["GALFOR_ALPHA_MAX"] = "12.5"
        self.assertEqual(galfor_prior_dict()[IA].maximum, 12.5)

    def test_kwarg_beats_env(self):
        os.environ["GALFOR_ALPHA_MAX"] = "12.5"
        self.assertEqual(galfor_prior_dict(alpha_max=30.0)[IA].maximum, 30.0)

    def test_other_bounds_unchanged_when_widened(self):
        base = galfor_prior_dict()
        wide = galfor_prior_dict(alpha_max=20.0)
        for name in GALFOR_BASIS:
            i = GALFOR_BASIS.index(name)
            if name == "alpha":
                continue
            self.assertEqual(wide[i].minimum, base[i].minimum, name)
            self.assertEqual(wide[i].maximum, base[i].maximum, name)

    def test_log_sampling_still_applies_to_log_params(self):
        # widening alpha must not disturb the log10 remap of amp/fk/f_1/f_2
        lin = galfor_prior_dict(log_sampling=False, alpha_max=20.0)
        log = galfor_prior_dict(log_sampling=True, alpha_max=20.0)
        iamp = GALFOR_BASIS.index("amp")
        self.assertGreater(lin[iamp].maximum, 0.0)          # linear ~1e-41
        self.assertLess(log[iamp].maximum, 0.0)             # log10(1e-41) < 0
        self.assertEqual(log[IA].maximum, 20.0)             # alpha stays linear


if __name__ == "__main__":
    unittest.main()
