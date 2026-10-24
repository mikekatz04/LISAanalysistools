"""``erebor.gb.build_gb_wdm_comp``: the one GB / VGB comp builder.

``GB_LIKELIHOOD=lookup`` (the default) gives a ``GBLookupWDMComputations``
built on the GB lookup table, for the GB block and the VGB block alike;
``chunked`` (the 9mo launcher) the plain ``GBWDMComputations``; anything else
is refused. Both settings blocks read the one env name. The grid's ``t0`` is moved to the data start by the builder.

The lookup table is the recipe table in ``~/.cache/gb_lookup_tables`` (or
``$GB_LOOKUP_TABLE_DIR``), built once by ``resolve_gb_lookup_table`` when it is
missing (a minute or two; the GBGPU lookup tests share it).
"""

import os
import types
import unittest
from unittest import mock

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

NF, DT, NT, EDGE = 180, 20.0, 720, 60


def _general_info(t0):
    wdm = WDMSettings(NF, NT, DT, t0=0.0, min_freq=1e-4, max_freq=2.5e-2,
                      min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                      force_backend="cpu")
    return types.SimpleNamespace(domain_settings=wdm, data_t0=t0,
                                 gpu_orbits=ESAOrbits(force_backend="cpu"),
                                 force_backend="cpu")


def _info(likelihood, t0, lookup_fill=False):
    return types.SimpleNamespace(
        likelihood=likelihood, t0=t0, nt_sub=256, n_pad=32, n_sparse=256, n_cp_sig=48,
        n_cp_orbit=32, use_tdi2=True, lookup_table_path="", lookup_table_dir="",
        lookup_n_nodes=-1, lookup_k_coarse=True, lookup_fill=lookup_fill)


class BuildGBWDMCompTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t0 = int(0.4 * YRSID_SI / DT) * DT

    def _build(self, likelihood, branch="gb", **extra):
        from lisatools.globalfit.stock.erebor.gb import build_gb_wdm_comp

        gi = _general_info(self.t0)
        comp = build_gb_wdm_comp(_info(likelihood, self.t0, **extra), gi, branch=branch)
        return comp, gi

    def test_lookup_builds_the_lookup_comp_on_the_table(self):
        from gbgpu.gblookupcomputations import GBLookupWDMComputations

        comp, gi = self._build("lookup")
        self.assertIsInstance(comp, GBLookupWDMComputations)
        self.assertFalse(comp.lookup_fill)
        self.assertEqual(float(gi.domain_settings.t0), float(self.t0))
        self.assertEqual(float(comp.t_obs_start), float(self.t0))

    def test_the_vgb_branch_gets_the_same_comp(self):
        from gbgpu.gblookupcomputations import GBLookupWDMComputations

        comp, _ = self._build("lookup", branch="vgb")
        self.assertIsInstance(comp, GBLookupWDMComputations)

    def test_chunked_builds_the_plain_chunked_comp(self):
        from gbgpu.gbcomps import GBWDMComputations

        comp, _ = self._build("chunked")
        self.assertIs(type(comp), GBWDMComputations)

    def test_lookup_fill_is_passed_through(self):
        comp, _ = self._build("lookup", lookup_fill=True)
        self.assertTrue(comp.lookup_fill)

    def test_another_value_is_refused(self):
        with self.assertRaises(ValueError):
            self._build("fd")


class LikelihoodFieldTest(unittest.TestCase):
    """The GB and VGB settings blocks read ``GB_LIKELIHOOD``; default lookup."""

    def _blocks(self):
        from lisatools.globalfit.stock.erebor.variants.gb_no_fg import GBNoFgGBSettings
        from lisatools.globalfit.stock.erebor.vgb import VGBSettings

        return GBNoFgGBSettings(), VGBSettings()

    def test_default_is_lookup(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("GB_LIKELIHOOD", None)
            gb, vgb = self._blocks()
        self.assertEqual((gb.likelihood, vgb.likelihood), ("lookup", "lookup"))

    def test_the_env_sets_both_blocks(self):
        with mock.patch.dict(os.environ, {"GB_LIKELIHOOD": "chunked"}):
            gb, vgb = self._blocks()
        self.assertEqual((gb.likelihood, vgb.likelihood), ("chunked", "chunked"))


if __name__ == "__main__":
    unittest.main()
