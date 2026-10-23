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
    GALFOR_MODULATION_TABLE,
    GalForTimeModulation,
    LinkDelayTable,
    UnequalArmInstrumentNoise,
    packaged_galfor_modulation_path,
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

    def test_resolved_kwargs_are_published_for_second_consumers(self):
        """The attribute must hold RESOLVED kwargs, not the deferred spec.

        A second consumer rebuilds a backend from
        ``general_info.sensitivity_init_kwargs`` -- the coarse sidecar in
        run.py does exactly that. If resolution only mutated a local copy, that
        consumer would be handed ``galfor_modulation_anchor`` (no constructor
        takes it) and the unresolved ``ltts_l1_file`` spec. Job 372 died on the
        first of those.
        """
        stub = _engine_stub()
        mod = GalForTimeModulation(self.modulation)
        sik = {
            "instrument_component_cls": UnequalArmInstrumentNoise,
            "instrument_component_kwargs": {
                "ltts_l1_file": self.brick,
                "ltts_stride": 10,
            },
            "galfor_modulation": mod,
            "galfor_modulation_anchor": "data_t0",
        }
        GeneralSetup._resolve_deferred_noise_model(stub, sik)

        published = stub.sensitivity_init_kwargs
        self.assertNotIn("galfor_modulation_anchor", published)
        comp = published["instrument_component_kwargs"]
        self.assertNotIn("ltts_l1_file", comp)
        self.assertNotIn("ltts_stride", comp)
        self.assertIsInstance(comp["ltts"], LinkDelayTable)
        # and it is directly usable as **kwargs by a second backend build
        import inspect

        for key in published:
            self.assertIsInstance(key, str)
            self.assertTrue(key.isidentifier(), f"{key!r} is not a valid kwarg")
        self.assertTrue(inspect.isclass(published["instrument_component_cls"]))

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


#: the mojito data start; the packaged table's first epoch
MOJITO_DATA_T0 = 97729939.827664


class PackagedModulationTableTest(unittest.TestCase):
    """The installed ``modulation_unequal.dat`` and the ``all_sources`` default."""

    def test_the_table_is_the_one_the_live_stores_recorded(self):
        # name + digest are what the resume guard compares; the 9mo / 12mo
        # stores were sampled under scripts/noise/modulation_unequal.dat
        self.assertTrue(os.path.isfile(packaged_galfor_modulation_path()))
        mod = GalForTimeModulation(GALFOR_MODULATION_TABLE)
        self.assertEqual(mod.path, packaged_galfor_modulation_path())
        stub = _engine_stub(data_t0=MOJITO_DATA_T0, tobs=360 * 86400.0)
        GeneralSetup._resolve_deferred_noise_model(
            stub, {"galfor_modulation": mod, "galfor_modulation_anchor": "data_t0"}
        )
        identity = stub.noise_model_identity
        self.assertEqual(identity["galfor_modulation"], "modulation_unequal.dat")
        self.assertEqual(identity["galfor_modulation_digest"], "2fd5af9aaf893724")
        self.assertEqual(identity["galfor_modulation_t0"], MOJITO_DATA_T0)
        t = mod._table()[:, 0]
        self.assertEqual(mod._table().shape, (199, 7))
        self.assertEqual(t[0], MOJITO_DATA_T0)
        self.assertGreater(t[-1] - t[0], 1.99 * 365.25 * 86400.0)

    def test_a_bare_name_not_packaged_is_left_alone(self):
        self.assertEqual(GalForTimeModulation("no_such_table.dat").path, "no_such_table.dat")

    def _finalized(self, data_mode="mojito", drop_galfor=False, **kwargs):
        fit = erebor.all_sources(lite=True, **kwargs)
        if drop_galfor:
            fit.remove_branch("galfor")
        fit.general.data_mode = data_mode
        fit.finalize_general(fit.general)
        return fit.general

    def test_mojito_defaults_to_the_packaged_table_at_the_data_start(self):
        if os.environ.get("GALFOR_MODULATION_PATH"):
            self.skipTest("GALFOR_MODULATION_PATH is set in this environment")
        self.assertEqual(erebor.all_sources(lite=True).general.galfor_modulation_path, "auto")
        gs = self._finalized()
        sik = gs.sensitivity_init_kwargs
        self.assertIsInstance(sik["galfor_modulation"], GalForTimeModulation)
        self.assertEqual(sik["galfor_modulation"].path, packaged_galfor_modulation_path())
        self.assertEqual(sik["galfor_modulation_anchor"], "data_t0")
        self.assertEqual(gs.galfor_modulation_path, packaged_galfor_modulation_path())
        self.assertEqual(gs.galfor_modulation_t0, "data")

    def test_auto_is_stationary_off_mojito_or_without_galfor(self):
        for kw in (dict(data_mode="synthetic"), dict(drop_galfor=True)):
            with self.subTest(**kw):
                gs = self._finalized(galfor_modulation_path="auto", **kw)
                self.assertIsNone(gs.sensitivity_init_kwargs["galfor_modulation"])
                self.assertNotIn("galfor_modulation_anchor", gs.sensitivity_init_kwargs)
                self.assertIsNone(gs.galfor_modulation_path)

    def test_none_turns_it_off_on_mojito(self):
        for off in (None, "none", "None"):
            with self.subTest(off=off):
                gs = self._finalized(galfor_modulation_path=off)
                self.assertIsNone(gs.sensitivity_init_kwargs["galfor_modulation"])
                self.assertNotIn("galfor_modulation_anchor", gs.sensitivity_init_kwargs)

    def test_an_explicit_epoch_wins_over_the_auto_anchor(self):
        gs = self._finalized(galfor_modulation_path="auto", galfor_modulation_t0="123.0")
        mod = gs.sensitivity_init_kwargs["galfor_modulation"]
        self.assertEqual(mod.t0, 123.0)
        self.assertNotIn("galfor_modulation_anchor", gs.sensitivity_init_kwargs)


if __name__ == "__main__":
    unittest.main()
