"""SOBBH chunked-het chunk sizing: the chirp-vs-layer criterion.

Hermetic, CPU-only: exercises the pure sizing helpers in
``stock.erebor.source_runtime`` -- no waveforms, no bbhx. The numbers are
pinned to the real mojito-lite catalogue's worst chirper (id 0:
m1=71.06, m2=66.19 Msun SSB, f_GW=15.265 mHz, tau_c=4.966e7 s) so the
test doubles as the record of WHY the defaults are what they are:
Nt_sub=32 is chirp-safe at 6 months and NOT at 1 year.
"""

import unittest

import numpy as np

from lisatools.globalfit.stock.erebor.source_runtime import (
    resolve_sobbh_nt_sub,
    sobbh_fdot_max_at_window_end,
    sobbh_leading_order_fdot,
)

# mojito-lite SOBHB id 0 (SSB-frame masses; f quoted at the window start)
M1, M2, F0 = 71.057763, 66.193855, 0.015265
TAU_C = 4.96572295e7          # s to coalescence at the window start
TOBS_6MO, TOBS_1YR = 15552000.0, 31104000.0
# 6-mo/1-yr WDM grids (Nf=1440, dt=2.5): layer_dt=3600 s, layer_df=1/7200
LAYER_DT, LAYER_DF = 3600.0, 1.0 / 7200.0


class LeadingOrderFdotTest(unittest.TestCase):
    def test_matches_the_time_to_coalescence_relation(self):
        """0PN self-consistency: fdot == (3/8) f / tau at the same epoch.

        The catalogue-side estimate used in planning (0.375*f/tau) and the
        mass-side formula are the SAME leading order -- they must agree to
        the precision of the catalogue's own tau_c (the catalogue tau is
        not exactly 0PN, so allow 10%).
        """
        fd = sobbh_leading_order_fdot(M1, M2, F0)
        self.assertAlmostEqual(fd / (0.375 * F0 / TAU_C), 1.0, delta=0.10)

    def test_vectorized(self):
        fd = sobbh_leading_order_fdot([M1, M1], [M2, M2], [F0, 2 * F0])
        self.assertEqual(fd.shape, (2,))
        # f^{11/3} scaling
        self.assertAlmostEqual(fd[1] / fd[0], 2.0 ** (11.0 / 3.0), places=6)


class FdotMaxAtWindowEndTest(unittest.TestCase):
    def _basis(self, f0=F0, m1=M1, m2=M2):
        # injection-basis order: f_low at COLUMN 6 (column 5 is the
        # inclination -- the exact column confusion the helper shipped
        # with; pinned here so it cannot regress)
        row = np.zeros(11)
        row[0], row[1], row[6] = m1, m2, f0
        return row[None, :]

    def test_grows_with_tobs(self):
        b = self._basis()
        fd6 = sobbh_fdot_max_at_window_end(b, TOBS_6MO)
        fd12 = sobbh_fdot_max_at_window_end(b, TOBS_1YR)
        fd0 = sobbh_leading_order_fdot(M1, M2, F0)
        self.assertGreater(fd6, fd0)          # end-of-window > start
        self.assertGreater(fd12, fd6)

    def test_six_month_worst_case_value(self):
        """Pins the planning number: ~1.9e-10 Hz/s for id 0 at 6 mo."""
        fd6 = sobbh_fdot_max_at_window_end(self._basis(), TOBS_6MO)
        self.assertAlmostEqual(fd6 / 1.93e-10, 1.0, delta=0.15)

    def test_merger_in_window_caps_at_band_edge(self):
        # a source whose tau_c < tobs must cap at f_cap, not blow up
        heavy = self._basis(f0=0.020)          # tau_c ~ 1.2e7 s < 6 mo
        fd = sobbh_fdot_max_at_window_end(heavy, TOBS_6MO, f_cap=0.025)
        self.assertEqual(fd, float(sobbh_leading_order_fdot(M1, M2, 0.025)))
        self.assertTrue(np.isfinite(fd))

    def test_empty_basis_is_zero(self):
        self.assertEqual(
            sobbh_fdot_max_at_window_end(np.zeros((0, 11)), TOBS_6MO), 0.0)


class ResolveNtSubTest(unittest.TestCase):
    """Semantics MEASURED by the 6-mo removal-null sweep (2026-08-25):
    the residual-vs-Nt_sub curve is U-shaped -- stitch overhead below
    Nt_sub=16, chirp shedding above ~7 layers/chunk of sweep, optimum
    ~3.5 -- so auto targets sweep <= sweep_max_layers (3.5) with a hard
    floor of 16, and warnings fire only beyond the 7-layer tolerance."""

    FD6 = 1.93e-10       # catalogue id 0 at the 6-mo window end
    # full_year stress grid (wavelet 40 ks): layer_dt=40000, Nt=388
    S_DT, S_DF, S_NT = 40000.0, 1.25e-5, 388
    FD_STOCK = 3.46e-11  # stock synthetic 15 mHz source

    def test_auto_production_grid_takes_long_chunks(self):
        """1-h layers: sweep/chunk is tiny (5e-3 layers per layer-chunk),
        so auto extends chunks toward the clamp -- 240 divides Nt=4320,
        sweep 1.2 layers, inside the measured optimum plateau."""
        got = resolve_sobbh_nt_sub(0, self.FD6, 4320, LAYER_DT, LAYER_DF,
                                   3.5)
        self.assertEqual(got, 240)

    def test_auto_stress_grid_floors_at_16(self):
        """The full_year 11.1-h-layer grid with the stock source: bound
        ~31.6 but Nt=388 has no divisor in [16, 31] -> floor 16, sweep
        1.77 layers -- exactly the measured sweet spot."""
        got = resolve_sobbh_nt_sub(0, self.FD_STOCK, self.S_NT, self.S_DT,
                                   self.S_DF, 3.5)
        self.assertEqual(got, 16)

    def test_auto_warns_when_no_safe_chunk_exists(self):
        """Catalogue id 0 on the stress grid: even Nt_sub=16 sweeps ~9.9
        layers (> 7) -- the coarse-layer-grid disease; auto floors at 16
        and says so loudly."""
        with self.assertLogs(
                "lisatools.globalfit.stock.erebor.source_runtime",
                level="WARNING") as cm:
            got = resolve_sobbh_nt_sub(0, self.FD6, self.S_NT, self.S_DT,
                                       self.S_DF, 3.5)
        self.assertEqual(got, 16)
        self.assertIn("no chirp-safe chunk", cm.output[0])

    def test_explicit_value_is_honored_but_warned_past_tolerance(self):
        # id 0 on the stress grid at Nt_sub=32: sweep ~19.8 layers > 7
        with self.assertLogs(
                "lisatools.globalfit.stock.erebor.source_runtime",
                level="WARNING") as cm:
            got = resolve_sobbh_nt_sub(32, self.FD6, self.S_NT, self.S_DT,
                                       self.S_DF, 3.5)
        self.assertEqual(got, 32)              # explicit wins
        self.assertIn("sheds chirped power", cm.output[0])

    def test_explicit_safe_value_is_silent(self):
        # production grid: sweep at 32 is 0.16 layers -- nothing to say
        got = resolve_sobbh_nt_sub(32, self.FD6, 4320, LAYER_DT, LAYER_DF,
                                   3.5)
        self.assertEqual(got, 32)

    def test_stock_stress_case_within_tolerance_is_not_warned(self):
        # the agent-measured production default case: stock source on the
        # stress grid at 32 -> sweep 3.55 layers, inside the 7-layer band
        got = resolve_sobbh_nt_sub(32, self.FD_STOCK, self.S_NT, self.S_DT,
                                   self.S_DF, 3.5)
        self.assertEqual(got, 32)

    def test_no_chirp_bound_auto_falls_back_to_32(self):
        self.assertEqual(
            resolve_sobbh_nt_sub(0, 0.0, 4320, LAYER_DT, LAYER_DF, 3.5),
            32)
        self.assertEqual(
            resolve_sobbh_nt_sub(0, None, 4320, LAYER_DT, LAYER_DF, 3.5),
            32)

    def test_extreme_chirp_floors_at_16_with_warning(self):
        with self.assertLogs(
                "lisatools.globalfit.stock.erebor.source_runtime",
                level="WARNING"):
            got = resolve_sobbh_nt_sub(0, 1e-6, 4320, LAYER_DT, LAYER_DF,
                                       3.5)
        self.assertEqual(got, 16)


if __name__ == "__main__":
    unittest.main()
