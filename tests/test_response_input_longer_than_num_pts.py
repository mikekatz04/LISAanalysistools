"""pyResponseTDI with an input waveform one sample LONGER than ``num_pts``.

``get_projections`` sizes ``y_gw`` by the input length (one row of
``num_inputs_per_source`` per link, the stride the response kernel writes
with) and only asserts ``num_inputs_per_source >= num_pts``. ``get_tdi_delays``
then hands the TDI kernel ``y_gw_length`` as that stride. Before 2026-10-19 it
passed ``num_pts`` instead, which is only right when the two are equal: FEW on
the 7.5 d ``all_sources_lite`` grid returns 129601 samples for ``num_pts`` =
129600, and the binding's length check refused the call ("input_links ...
Should be 777600. It's length is 777606"). This test is that configuration
in miniature; the fix is one line in ``directresponse.get_projections``.
"""

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.directresponse import pyResponseTDI

from tests.test_batched_response import _DT, _GB_ARGS, _ORDER, _T, _T_BUFFER, _GBWave

_LAM, _BETA = 5.22979888, 0.98057429


class TestInputLongerThanNumPts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wave = _GBWave(force_backend="cpu")
        pols = wave(
            _GB_ARGS["A"], _GB_ARGS["f"], _GB_ARGS["fdot"],
            _GB_ARGS["iota"], _GB_ARGS["phi0"], _GB_ARGS["psi"],
            T=_T, dt=_DT,
        )
        cls.pols = np.asarray(pols)
        cls.num_pts = len(cls.pols) - 1  # the data grid is one sample SHORTER than the waveform
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")

    def _tdi(self, pols):
        resp = pyResponseTDI(
            sampling_frequency=1.0 / _DT,
            num_pts=self.num_pts,
            orbits=self.orbits,
            order=_ORDER,
            tdi="1st generation",
            tdi_chan="XYZ",
            force_backend="cpu",
        )
        resp.get_projections(pols, _LAM, _BETA, t_buffer=_T_BUFFER)
        return [np.asarray(ch) for ch in resp.get_tdi_delays()]

    def test_longer_input_runs_and_matches_exact_length_input(self):
        longer = self._tdi(self.pols)              # num_pts + 1 samples in
        exact = self._tdi(self.pols[: self.num_pts])
        for ch_long, ch_exact in zip(longer, exact):
            self.assertEqual(ch_long.shape, (self.num_pts,))
            # The TDI kernel only ever reads EARLIER samples, so the extra
            # trailing input sample changes nothing it computes.
            np.testing.assert_array_equal(ch_long, ch_exact)


if __name__ == "__main__":
    unittest.main()
