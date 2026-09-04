"""Gate/cadence logic for the reseed+replace combined proposal (v9 run).

The combined proposal is hosted on the SEARCH replace move. Three pure gates
decide its behavior each propose: ``_reseed_host_armed`` (is this the armed
host), ``_cold_reseed_fire`` (is this the firing iteration), and
``_cold_reseed_replace_passes`` (how many replace sweeps per firing). All are
exercised here on a lightweight stand-in via the unbound methods -- no move
construction, no GPU, no data.
"""

import os
import unittest
from types import MethodType, SimpleNamespace

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


def _replace(**over):
    """A stand-in for THE search replace move (the combined-proposal host).

    ``_cold_reseed_fire`` calls ``self._reseed_host_armed()``, so bind that
    method onto the stand-in too (the rest are self-contained).
    """
    d = dict(rj_replace=True, name="rj_replace_search")
    d.update(over)
    ns = SimpleNamespace(**d)
    ns._reseed_host_armed = MethodType(GBSpecialBase._reseed_host_armed, ns)
    return ns


class ReseedHostArmedTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("GB_COLD_RESEED_EVERY")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_COLD_RESEED_EVERY", None)
        else:
            os.environ["GB_COLD_RESEED_EVERY"] = self._old

    def test_off_by_default_not_a_host(self):
        os.environ.pop("GB_COLD_RESEED_EVERY", None)
        self.assertFalse(GBSpecialBase._reseed_host_armed(_replace()))

    def test_zero_not_a_host(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "0"
        self.assertFalse(GBSpecialBase._reseed_host_armed(_replace()))

    def test_armed_replace_search_move_is_host(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "7"
        self.assertTrue(GBSpecialBase._reseed_host_armed(_replace()))

    def test_non_replace_or_pe_moves_are_not_hosts(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "7"
        for over in (
            dict(rj_replace=False),          # a birth/removal move
            dict(name="rj_replace_pe"),      # the PE replace move (no "search")
        ):
            self.assertFalse(
                GBSpecialBase._reseed_host_armed(_replace(**over)),
                msg=f"must not host for {over}")


class ColdReseedFireTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("GB_COLD_RESEED_EVERY")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_COLD_RESEED_EVERY", None)
        else:
            os.environ["GB_COLD_RESEED_EVERY"] = self._old

    def test_off_never_fires(self):
        os.environ.pop("GB_COLD_RESEED_EVERY", None)
        m = _replace()
        self.assertFalse(any(GBSpecialBase._cold_reseed_fire(m) for _ in range(20)))

    def test_fires_every_n_on_host(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "3"
        m = _replace()
        fired = [GBSpecialBase._cold_reseed_fire(m) for _ in range(9)]
        self.assertEqual(
            fired,
            [False, False, True, False, False, True, False, False, True],
        )

    def test_never_fires_on_non_host(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "1"
        for over in (dict(rj_replace=False), dict(name="rj_replace_pe")):
            m = _replace(**over)
            self.assertFalse(
                GBSpecialBase._cold_reseed_fire(m),
                msg=f"must not fire for {over}")

    def test_bad_env_value_is_off(self):
        os.environ["GB_COLD_RESEED_EVERY"] = "not-an-int"
        self.assertFalse(GBSpecialBase._cold_reseed_fire(_replace()))


class ReplacePassesTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("GB_COLD_RESEED_REPLACE_PASSES")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_COLD_RESEED_REPLACE_PASSES", None)
        else:
            os.environ["GB_COLD_RESEED_REPLACE_PASSES"] = self._old

    def test_default_is_three(self):
        os.environ.pop("GB_COLD_RESEED_REPLACE_PASSES", None)
        self.assertEqual(GBSpecialBase._cold_reseed_replace_passes(_replace()), 3)

    def test_env_override(self):
        os.environ["GB_COLD_RESEED_REPLACE_PASSES"] = "5"
        self.assertEqual(GBSpecialBase._cold_reseed_replace_passes(_replace()), 5)

    def test_clamped_to_at_least_one(self):
        os.environ["GB_COLD_RESEED_REPLACE_PASSES"] = "0"
        self.assertEqual(GBSpecialBase._cold_reseed_replace_passes(_replace()), 1)

    def test_bad_value_falls_back_to_three(self):
        os.environ["GB_COLD_RESEED_REPLACE_PASSES"] = "xyz"
        self.assertEqual(GBSpecialBase._cold_reseed_replace_passes(_replace()), 3)


if __name__ == "__main__":
    unittest.main()
