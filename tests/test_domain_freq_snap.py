"""Regression: domain settings snap ``min_freq``/``max_freq`` to the physical grid.

A non-grid-aligned ``min_freq`` / ``max_freq`` must be reported by the
``.min_freq`` / ``.max_freq`` properties as the *snapped* grid value
``ind_min * df`` -- the value the C++ ``STFTDomain`` consumes as its float
``f_min`` origin (``cutils/domains.cu::get_freq_index`` reconstructs the bin as
``(int)((f - f_min) / df)``).  The raw user request is preserved separately on
``.min_freq_input`` / ``.max_freq_input`` (which the global-fit engine reads).

Before this fix the property returned the raw input, so an off-grid ``min_freq``
left the C++ STFT carrier index offset from the grid-aligned data by up to one
bin (mismatch ~0.76).  ``FDSettings`` / ``WDMSettings`` index by the integer
``ind_min`` and were never wrong at the C++ level, but are snapped here too so
``.min_freq`` is a consistent, grid-honest value across every domain.
"""

import unittest

import numpy as np

from lisatools.domains import FDSettings, STFTSettings, WDMSettings


class DomainFreqSnapTest(unittest.TestCase):
    """Off-grid ``min_freq``/``max_freq`` round-trip to the grid; raw kept on ``*_input``."""

    def _assert_offgrid_snapped(self, s, raw_min, raw_max, ind_min, ind_max, df):
        # the property returns the snapped grid edge: exactly ind * df ...
        self.assertEqual(s.min_freq, ind_min * df)
        self.assertEqual(s.max_freq, ind_max * df)
        # ... which, for an off-grid request, differs from the raw input ...
        self.assertNotEqual(s.min_freq, raw_min)
        self.assertNotEqual(s.max_freq, raw_max)
        # ... and the raw request is preserved verbatim on *_input.
        self.assertEqual(s.min_freq_input, raw_min)
        self.assertEqual(s.max_freq_input, raw_max)

    def test_fd_offgrid_snaps_to_grid(self):
        df = 0.25  # binary-exact: no float fuzz in ceil()/int()
        raw_min, raw_max = 20.5 * df, 100.5 * df
        s = FDSettings(N=512, df=df, min_freq=raw_min, max_freq=raw_max,
                       force_backend="cpu")
        self.assertEqual(s.ind_min, int(np.ceil(raw_min / df)))  # 21
        self.assertEqual(s.ind_max, int(raw_max / df))           # 100
        self._assert_offgrid_snapped(s, raw_min, raw_max, s.ind_min, s.ind_max, s.df)

    def test_stft_offgrid_snaps_to_grid(self):
        # STFT is the correctness-critical domain: its C++ get_freq_index uses
        # the float f_min origin, so an off-grid f_min mis-indexes the carrier.
        df = 0.25
        raw_min, raw_max = 20.5 * df, 100.5 * df
        s = STFTSettings(t0=0.0, dt=16.0, df=df, NT=8, NF=512,
                         min_freq=raw_min, max_freq=raw_max, force_backend="cpu")
        self.assertEqual(s.ind_min, int(np.ceil(raw_min / df)))
        self.assertEqual(s.ind_max, int(raw_max / df))
        self._assert_offgrid_snapped(s, raw_min, raw_max, s.ind_min, s.ind_max, s.df)

    def test_wdm_offgrid_snaps_to_grid(self):
        Nf, Nt, dt = 128, 128, 16.0
        layer_df = 1.0 / (2 * Nf * dt)  # = 1/4096, binary-exact
        raw_min, raw_max = 20.5 * layer_df, 100.5 * layer_df
        s = WDMSettings(Nf, Nt, dt, t0=0.0, min_freq=raw_min, max_freq=raw_max,
                        force_backend="cpu")
        self.assertEqual(s.ind_min_f, int(np.ceil(raw_min / layer_df)))
        self.assertEqual(s.ind_max_f, int(raw_max / layer_df))
        self._assert_offgrid_snapped(s, raw_min, raw_max,
                                     s.ind_min_f, s.ind_max_f, s.layer_df)

    def test_ongrid_request_is_unchanged(self):
        # a grid-aligned request round-trips unchanged (the snap is idempotent).
        df = 0.25
        on_min, on_max = 20 * df, 100 * df
        fd = FDSettings(N=512, df=df, min_freq=on_min, max_freq=on_max,
                        force_backend="cpu")
        self.assertEqual(fd.min_freq, on_min)
        self.assertEqual(fd.max_freq, on_max)
        st = STFTSettings(t0=0.0, dt=16.0, df=df, NT=8, NF=512,
                          min_freq=on_min, max_freq=on_max, force_backend="cpu")
        self.assertEqual(st.min_freq, on_min)
        self.assertEqual(st.max_freq, on_max)
        layer_df = 1.0 / (2 * 128 * 16.0)
        wd = WDMSettings(128, 128, 16.0, t0=0.0, min_freq=20 * layer_df,
                         max_freq=100 * layer_df, force_backend="cpu")
        self.assertEqual(wd.min_freq, 20 * layer_df)
        self.assertEqual(wd.max_freq, 100 * layer_df)


if __name__ == "__main__":
    unittest.main()
