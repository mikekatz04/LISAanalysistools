"""The MBH branch's inner-move stack carries the LISA-frame sky-mode hops.

``source_runtime.make_mbh_inner_moves`` (called by ``prepare_mbh_branch``
when ``inner_moves`` is unset) mixes three ``SkyMove`` proposals in ICRS
(the stock basis) at ``sky_move_{both,lat,long}_weight`` and scales the
``inner_move_kind`` stack to the remainder; the hops read the constellation
phase off the general setup's orbits.
"""

import os
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves import EigenAxisMove, StretchMove

from lisatools.sampling.moves.skymodehop import ConstellationPhase, SkyMove
from lisatools.utils.constants import YRSID_SI


_ORBITS = []


def _orbits():
    """Stock equal-arm orbits on a coarse 2-yr hourly grid (built once; the
    default dense grid over the whole file costs minutes)."""
    from lisatools.detector import EqualArmlengthOrbits

    if not _ORBITS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ORBITS.append(
                EqualArmlengthOrbits(
                    force_backend="cpu", frame="ecliptic", linear_interp_setup=False,
                    t_arr=np.arange(0.0, 2.0 * YRSID_SI, 3600.0),
                )
            )
    return _ORBITS[0]


class MakeMBHInnerMovesTest(unittest.TestCase):
    WF_T0 = 1234.5

    def _settings(self, **env):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("MBH_SKY_MOVE_BOTH_WEIGHT", "MBH_SKY_MOVE_LAT_WEIGHT",
                      "MBH_SKY_MOVE_LONG_WEIGHT", "MBH_INNER_MOVE_KIND"):
                if k not in env:
                    os.environ.pop(k, None)
            return SourceMBHSettings()

    def test_defaults_are_the_requested_proportions(self):
        s = self._settings()
        self.assertEqual((s.sky_move_both_weight, s.sky_move_lat_weight, s.sky_move_long_weight), (0.02, 0.04, 0.02))

    def test_env_knobs(self):
        s = self._settings(MBH_SKY_MOVE_BOTH_WEIGHT="0.1", MBH_SKY_MOVE_LAT_WEIGHT="0", MBH_SKY_MOVE_LONG_WEIGHT="0.05")
        self.assertEqual((s.sky_move_both_weight, s.sky_move_lat_weight, s.sky_move_long_weight), (0.1, 0.0, 0.05))

    def test_stack_composition_and_phase_table(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.globalfit.stock.erebor.transforms import make_mbh_transform_container

        mbh = self._settings()
        mbh.transform = make_mbh_transform_container()
        gsetup = SimpleNamespace(orbits=_orbits())
        gs = SimpleNamespace(mbh_waveform_t0=self.WF_T0)
        moves = sr.make_mbh_inner_moves(mbh, gsetup, gs)
        self.assertEqual(len(moves), 4)
        (base, w_base), *hops = moves
        self.assertIsInstance(base, EigenAxisMove)
        self.assertAlmostEqual(w_base, 0.92)
        self.assertAlmostEqual(sum(w for _, w in moves), 1.0)
        kinds = {m.which: w for m, w in hops}
        self.assertEqual(kinds, {"both": 0.02, "lat": 0.04, "long": 0.02})
        for m, _ in hops:
            self.assertIsInstance(m, SkyMove)
            self.assertEqual(m.coord_frame, "icrs")
            self.assertEqual(m.ind_map, sr.MBH_SKY_MOVE_IND_MAP)
            self.assertIsInstance(m.constellation_phase, ConstellationPhase)
            self.assertEqual(m.constellation_phase.t_offset, self.WF_T0)
            self.assertTrue(m.jacobian)
        # the phase table is the orbits' (equal-arm: alpha = Omega t)
        from bbhx.utils.constants import ConstOmega

        t = 0.3 * YRSID_SI
        alpha = hops[0][0].constellation_phase(t - self.WF_T0)
        self.assertAlmostEqual(float(np.angle(np.exp(1j * (alpha - ConstOmega * t)))), 0.0, places=3)

    def test_index_map_matches_the_stock_basis(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.globalfit.stock.erebor.transforms import make_mbh_transform_container

        basis = make_mbh_transform_container().input_basis
        self.assertEqual(
            {k: basis[i] for k, i in sr.MBH_SKY_MOVE_IND_MAP.items()},
            {"cosinc": "cos_iota", "psi": "psi", "lam": "alpha", "sinbeta": "sin_delta", "t_ref": "t_plunge"},
        )
        mbh = self._settings()
        mbh.transform = SimpleNamespace(input_basis=["a"] * 11)
        with self.assertRaises(ValueError):
            sr.make_mbh_inner_moves(mbh, SimpleNamespace(orbits=_orbits()), SimpleNamespace(mbh_waveform_t0=0.0))

    def test_zero_weights_give_the_bare_stack(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        mbh = self._settings(MBH_SKY_MOVE_BOTH_WEIGHT="0", MBH_SKY_MOVE_LAT_WEIGHT="0",
                             MBH_SKY_MOVE_LONG_WEIGHT="0", MBH_INNER_MOVE_KIND="stretch")
        moves = sr.make_mbh_inner_moves(mbh, SimpleNamespace(), SimpleNamespace())
        self.assertEqual(len(moves), 1)
        self.assertIsInstance(moves[0][0], StretchMove)
        self.assertEqual(moves[0][1], 1.0)

    def test_weights_must_leave_room_for_the_stack(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        mbh = self._settings(MBH_SKY_MOVE_BOTH_WEIGHT="0.5", MBH_SKY_MOVE_LAT_WEIGHT="0.5")
        with self.assertRaises(ValueError):
            sr.make_mbh_inner_moves(mbh, SimpleNamespace(), SimpleNamespace())
        mbh = self._settings(MBH_SKY_MOVE_BOTH_WEIGHT="-0.1")
        with self.assertRaises(ValueError):
            sr.make_mbh_inner_moves(mbh, SimpleNamespace(), SimpleNamespace())

    def test_no_orbits_falls_back_to_the_analytic_phase(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        mbh = self._settings()
        with self.assertLogs(sr.logger, level="WARNING"):
            moves = sr.make_mbh_inner_moves(mbh, SimpleNamespace(), SimpleNamespace())
        self.assertEqual(len(moves), 4)
        self.assertIsNone(moves[1][0].constellation_phase)

    def test_prepare_mbh_branch_wires_the_stack(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        mbh = self._settings()
        mbh.initialize_kwargs = {}
        rows = {0: np.r_[np.zeros(10), 5.0 * 86400.0]}
        gsetup = SimpleNamespace(data_t0=0.0, Tobs=30 * 86400.0, orbits=_orbits())
        gs = SimpleNamespace(n_injections={"MBHB": 1}, data_mode="mojito", mbh_waveform_t0=0.0)
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(sr, "mbh_catalogue_to_sampling_basis", side_effect=lambda r: r):
            out = sr.prepare_mbh_branch(mbh, gsetup, gs)
        self.assertEqual(len(out.inner_moves), 4)
        self.assertEqual([type(m).__name__ for m, _ in out.inner_moves], ["EigenAxisMove"] + ["SkyMove"] * 3)

    def test_settings_pickle_and_deepcopy(self):
        import copy
        import pickle

        s = self._settings()
        s2 = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertEqual(s2.sky_move_lat_weight, 0.04)


if __name__ == "__main__":
    unittest.main()
