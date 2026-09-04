"""Tests for the diagnostic coarse->fine resume restamp
(scripts/fstat_proposal/coarse_resume_restamp.py).

The restamp lets a chain sampled under the coarse (delayed_acceptance)
likelihood be resumed under the exact-fine likelihood for the drift test,
by rewriting ONLY the coarse_* keys of the stored noise-model identity so the
engine's resume guard passes. These tests pin: (a) a coarse-only mismatch is
restamped and the engine's own compare then finds no mismatch; (b) a real
data-model mismatch (unequal_arm) is still refused; (c) with the override off
the strict refusal is unchanged.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "scripts" / "fstat_proposal" / "coarse_resume_restamp.py"
_spec = importlib.util.spec_from_file_location("coarse_resume_restamp", _MOD)
crr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crr)

COARSE_STAMP = dict(
    instrument_component="UnequalArmInstrumentNoise",
    unequal_arm=True,
    wdm_psd_method="layer_calibrated",
    galfor_modulation="modulation_unequal.dat",
    data_t0=97729939.827664,
    coarse_mode="delayed_acceptance",
    coarse_Q=8,
    coarse_use_ws=True,
    coarse_fiducial_digest="856ddb07f6b46e39",
)


def _make_store(path, **identity):
    with h5py.File(path, "w") as f:
        sub = f.create_group("global_fit").create_group("noise_model_identity")
        for k, v in identity.items():
            sub.attrs[k] = v


def _engine_mismatch(stored, current):
    """Mirror run.py's resume compare (loops over the CURRENT identity keys)."""
    mism = {}
    for key, value in current.items():
        sv = stored.get(key)
        if isinstance(value, float):
            same = sv is not None and np.isclose(float(sv), value, rtol=0.0, atol=1e-6)
        else:
            same = sv == value
        if not same:
            mism[key] = (sv, value)
    return mism


class RestampTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.store = str(Path(self._td.name) / "store.h5")

    def tearDown(self):
        self._td.cleanup()

    def test_coarse_only_mismatch_is_restamped(self):
        _make_store(self.store, **COARSE_STAMP)
        # deliberate coarse -> fine switch; must NOT raise with the override on
        crr.preflight(
            self.store, want_unequal_arm=True, want_method="layer_calibrated",
            want_mode="off", want_q=1, restamp_enabled=True,
        )
        a = crr.read_identity(self.store)
        self.assertEqual(a["coarse_mode"], "off")
        self.assertEqual(int(a["coarse_Q"]), 1)
        self.assertNotIn("coarse_fiducial_digest", a)  # dropped when Q<=1
        # data-model keys untouched
        self.assertEqual(a["wdm_psd_method"], "layer_calibrated")
        self.assertTrue(bool(a["unequal_arm"]))

    def test_engine_compare_passes_after_restamp(self):
        _make_store(self.store, **COARSE_STAMP)
        crr.preflight(
            self.store, want_unequal_arm=True, want_method="layer_calibrated",
            want_mode="off", want_q=1, restamp_enabled=True,
        )
        # what the engine produces for a coarse-OFF resume (no sidecar -> no
        # coarse_fiducial_digest); every non-coarse key unchanged.
        current = {
            "instrument_component": "UnequalArmInstrumentNoise",
            "unequal_arm": True,
            "wdm_psd_method": "layer_calibrated",
            "galfor_modulation": "modulation_unequal.dat",
            "data_t0": 97729939.827664,
            "coarse_mode": "off",
            "coarse_Q": 1,
            "coarse_use_ws": True,
        }
        stored = crr.read_identity(self.store)
        self.assertEqual(_engine_mismatch(stored, current), {})

    def test_noncoarse_mismatch_is_refused(self):
        # a real data-model change (equal-arm) must never be restamped away
        stamp = {**COARSE_STAMP, "unequal_arm": False}
        _make_store(self.store, **stamp)
        with self.assertRaises(SystemExit):
            crr.preflight(
                self.store, want_unequal_arm=True, want_method="layer_calibrated",
                want_mode="off", want_q=1, restamp_enabled=True,
            )
        # store NOT mutated on refusal
        self.assertEqual(crr.read_identity(self.store)["coarse_mode"],
                         "delayed_acceptance")

    def test_coarse_mismatch_refused_when_override_off(self):
        _make_store(self.store, **COARSE_STAMP)
        with self.assertRaises(SystemExit):
            crr.preflight(
                self.store, want_unequal_arm=True, want_method="layer_calibrated",
                want_mode="off", want_q=1, restamp_enabled=False,
            )
        self.assertEqual(crr.read_identity(self.store)["coarse_mode"],
                         "delayed_acceptance")

    def test_matching_identity_ok_no_restamp(self):
        stamp = {**COARSE_STAMP, "coarse_mode": "off", "coarse_Q": 1}
        del stamp["coarse_fiducial_digest"]
        _make_store(self.store, **stamp)
        crr.preflight(
            self.store, want_unequal_arm=True, want_method="layer_calibrated",
            want_mode="off", want_q=1, restamp_enabled=True,
        )
        self.assertEqual(crr.read_identity(self.store)["coarse_mode"], "off")

    def test_missing_identity_is_refused(self):
        with h5py.File(self.store, "w") as f:
            f.create_group("global_fit")  # no noise_model_identity
        with self.assertRaises(SystemExit):
            crr.preflight(
                self.store, want_unequal_arm=True, want_method="layer_calibrated",
                want_mode="off", want_q=1, restamp_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
