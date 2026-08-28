"""Wiring tests for the unequal-arm + modulation knobs on ``all_sources``.

Construction-level: nothing here loads data or builds a fit. Covers the
UNEQUAL_ARM / UNEQUAL_ARM_STRIDE / WDM_PSD_METHOD / GALFOR_MODULATION_T0
knobs, their validation, the deferred-spec handoff to
``GeneralSetup._resolve_deferred_noise_model``, and the noise-model identity
those runs persist for resume checks.
"""

import copy
import logging
import os
import pickle
import tempfile
import types
import unittest

import h5py
import numpy as np

from lisatools.globalfit.engine import GeneralSetup
from lisatools.globalfit.stock import erebor
from lisatools.sensitivity import (
    GalForTimeModulation,
    LinkDelayTable,
    UnequalArmInstrumentNoise,
)

L0 = 8.339  # ~equal-arm light travel time, seconds

BRICK_T0 = 1.0e8
BRICK_DT = 2.5
BRICK_N = 2000  # table spans ~5000 s


def _make_noise_brick(path, with_ltts=True):
    with h5py.File(path, "w") as fh:
        fh.create_dataset("noise_estimates_placeholder", data=np.zeros(4))
        if with_ltts:
            grp = fh.create_group("ltts")
            samp = grp.create_group("sampling")
            samp.attrs["t0"] = BRICK_T0
            samp.attrs["dt"] = BRICK_DT
            for i, link in enumerate(["12", "23", "31", "13", "32", "21"]):
                grp.create_dataset(
                    f"ltt_{link}",
                    data=L0 * (1.0 + 1e-3 * i) + np.zeros(BRICK_N),
                )


def _make_modulation_table(path, t0=BRICK_T0, n=64, span=5000.0):
    t = t0 + np.linspace(0.0, span, n)
    cols = [t] + [np.full(n, v) for v in (1.0, 1.0, 1.0, -0.5, -0.5, -0.5)]
    np.savetxt(path, np.stack(cols, axis=1))


def _engine_stub(data_t0=BRICK_T0 + 100.0, tobs=3600.0):
    return types.SimpleNamespace(
        data_t0=data_t0, Tobs=tobs, logger=logging.getLogger("wiring-test")
    )


class UnequalArmKnobWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brick = os.path.join(self.tmp.name, "NOISE_test.h5")
        _make_noise_brick(self.brick)
        self.brick_no_ltts = os.path.join(self.tmp.name, "NOISE_bald.h5")
        _make_noise_brick(self.brick_no_ltts, with_ltts=False)
        self.modulation = os.path.join(self.tmp.name, "modulation.dat")
        _make_modulation_table(self.modulation)

    def tearDown(self):
        self.tmp.cleanup()

    def _fit(self, **kwargs):
        fit = erebor.all_sources(lite=True, **kwargs)
        return fit

    # -- validation ---------------------------------------------------------
    def test_synthetic_mode_rejected(self):
        fit = self._fit(unequal_arm=True)
        fit.general.data_mode = "synthetic"
        with self.assertRaisesRegex(ValueError, "mojito"):
            fit.finalize_general(fit.general)

    def test_missing_ltts_group_rejected(self):
        fit = self._fit(unequal_arm=True)
        fit.general.data_mode = "mojito"
        fit.general.noise_file = self.brick_no_ltts
        with self.assertRaisesRegex(ValueError, "/ltts"):
            fit.finalize_general(fit.general)

    def test_knobs_default_inert(self):
        fit = self._fit()
        self.assertFalse(fit.general.unequal_arm)
        fit.general.data_mode = "synthetic"
        fit.finalize_general(fit.general)
        sik = fit.general.sensitivity_init_kwargs
        self.assertNotIn("galfor_modulation_anchor", sik)
        self.assertIsNone(sik.get("instrument_component_cls"))

    # -- deferred-spec wiring ----------------------------------------------
    def test_wiring_sets_deferred_specs(self):
        fit = self._fit(
            unequal_arm=True,
            unequal_arm_stride=10,
            wdm_psd_method="layer_calibrated",
            galfor_modulation_path=self.modulation,
            galfor_modulation_t0="data",
        )
        fit.general.data_mode = "mojito"
        fit.general.noise_file = self.brick
        fit.finalize_general(fit.general)

        self.assertIs(
            fit.psd.instrument_component_cls, UnequalArmInstrumentNoise
        )
        sik = fit.general.sensitivity_init_kwargs
        comp = sik["instrument_component_kwargs"]
        self.assertEqual(comp["ltts_l1_file"], self.brick)
        self.assertEqual(comp["ltts_stride"], 10)
        self.assertEqual(comp["wdm_psd_method"], "layer_calibrated")
        self.assertEqual(sik["wdm_psd_method"], "layer_calibrated")
        self.assertEqual(sik["galfor_modulation_anchor"], "data_t0")
        self.assertIsInstance(sik["galfor_modulation"], GalForTimeModulation)

    def test_prebuild_fit_still_pickles(self):
        fit = self._fit(
            unequal_arm=True,
            galfor_modulation_path=self.modulation,
            galfor_modulation_t0="data",
        )
        fit.general.data_mode = "mojito"
        fit.general.noise_file = self.brick
        fit.finalize_general(fit.general)
        clone = pickle.loads(pickle.dumps(copy.deepcopy(fit)))
        comp = clone.general.sensitivity_init_kwargs["instrument_component_kwargs"]
        self.assertEqual(comp["ltts_l1_file"], self.brick)

    # -- engine-side late resolution ---------------------------------------
    def test_deferred_resolution_builds_table_at_data_t0(self):
        stub = _engine_stub()
        mod = GalForTimeModulation(self.modulation)  # t0=0 until anchored
        sik = {
            "instrument_component_cls": UnequalArmInstrumentNoise,
            "instrument_component_kwargs": {
                "ltts_l1_file": self.brick,
                "ltts_stride": 10,
                "wdm_psd_method": "layer_calibrated",
            },
            "wdm_psd_method": "layer_calibrated",
            "galfor_modulation": mod,
            "galfor_modulation_anchor": "data_t0",
        }
        out = GeneralSetup._resolve_deferred_noise_model(stub, sik)

        comp = out["instrument_component_kwargs"]
        self.assertNotIn("ltts_l1_file", comp)
        table = comp["ltts"]
        self.assertIsInstance(table, LinkDelayTable)
        self.assertEqual(table.t.size, int(np.ceil(BRICK_N / 10)))
        self.assertEqual(mod.t0, stub.data_t0)
        self.assertNotIn("galfor_modulation_anchor", out)

        identity = stub.noise_model_identity
        self.assertTrue(identity["unequal_arm"])
        self.assertEqual(identity["wdm_psd_method"], "layer_calibrated")
        self.assertEqual(identity["instrument_component"], "UnequalArmInstrumentNoise")
        self.assertTrue(identity["ltts_digest"])
        self.assertTrue(identity["galfor_modulation_digest"])
        self.assertEqual(identity["data_t0"], stub.data_t0)

    def test_deferred_resolution_identity_is_stable(self):
        def run():
            stub = _engine_stub()
            GeneralSetup._resolve_deferred_noise_model(
                stub,
                {
                    "instrument_component_kwargs": {
                        "ltts_l1_file": self.brick,
                        "ltts_stride": 10,
                    }
                },
            )
            return stub.noise_model_identity

        self.assertEqual(run(), run())

    def test_delay_table_coverage_rejected(self):
        stub = _engine_stub(data_t0=BRICK_T0 + 1e6, tobs=3600.0)
        with self.assertRaisesRegex(ValueError, "delay table"):
            GeneralSetup._resolve_deferred_noise_model(
                stub,
                {
                    "instrument_component_kwargs": {
                        "ltts_l1_file": self.brick,
                        "ltts_stride": 10,
                    }
                },
            )

    def test_modulation_coverage_rejected_without_anchor(self):
        # mission-clock table left at t0=0: covers [1e8, ...] of a [0, Tobs]
        # data frame -> must fail loudly at build, not interpolate nonsense
        stub = _engine_stub()
        mod = GalForTimeModulation(self.modulation)
        with self.assertRaisesRegex(ValueError, "modulation table"):
            GeneralSetup._resolve_deferred_noise_model(
                stub, {"galfor_modulation": mod}
            )

    def test_anchor_without_modulation_rejected(self):
        stub = _engine_stub()
        with self.assertRaisesRegex(ValueError, "galfor_modulation_anchor"):
            GeneralSetup._resolve_deferred_noise_model(
                stub, {"galfor_modulation_anchor": "data_t0"}
            )

    def test_default_identity_records_equal_arm(self):
        stub = _engine_stub()
        GeneralSetup._resolve_deferred_noise_model(stub, {})
        identity = stub.noise_model_identity
        self.assertFalse(identity["unequal_arm"])
        self.assertEqual(identity["instrument_component"], "InstrumentNoise")
        self.assertEqual(identity["wdm_psd_method"], "fold")


if __name__ == "__main__":
    unittest.main()
