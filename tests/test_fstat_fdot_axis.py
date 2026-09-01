"""The F-stat grid's fdot axis: bounds, sizing, and the (Mc, r) conversion.

WHAT IS WRONG TODAY. Both grid paths assemble rows as

    pr[:, 2] = get_fdot(f=pr[:, 1], Mc=mc_node)          # no (1 + r)

so the grid IS the ``r = 0`` manifold. Three consequences:

* it searches only GR-driven chirps; ``r`` is bolted on afterwards as a
  draw the grid never scored;
* the reachable ``fdot`` is ``[fdot_gr(mc_lo), fdot_gr(mc_hi)]``, which is
  strictly POSITIVE -- **negative fdot is unrepresentable** -- yet 40% of
  low-f and 21% of high-f leaves in v7 carry ``fdot < 0``, and the
  spurious +22-bin flagship mode is a ``fdot ~ 0`` solution;
* ``fdot ~ Mc^(5/3)``, so uniform-in-Mc nodes are NON-uniform in fdot
  while ``StackedFStatProposal4D`` hard-assumes uniform axes -- the
  density is implicitly ``Mc^(5/3)``-tilted in the coordinate that
  matters.

THE FIX (user ruling): fit in ``(f0, fdot_total)``, convert after. The Mc
axis becomes a LINEAR fdot axis in Hz/s, per group -- which is what the
existing per-group axis architecture already supports, so
``StackedFStatProposal4D`` itself does not change. What changes is what
axis 1 MEANS, and everything that reads it.

The conversion back is the part that can be silently wrong, so it is
tested the same way the in-model measure was: a round trip, an exactness
pairing between ``rvs`` and ``logpdf``, and negative controls that must
skew.
"""
import unittest

import numpy as np

from lisatools.sampling.gb_observable_basis import (
    FDOT_K,
    fdot_axis_bounds,
    fdot_gr,
    mc_floor_for_fdot,
    n_fdot_nodes,
    r_from_fdot,
)

TOBS = 7.776e6
MC_LO, MC_HI = 0.001, 1.0
RATIO_MAX = 5.0                      # GBSettings.fdot_astro_ratio_max default
F0_FLAG = 20.3803767e-3              # the flagship, Hz
F0_LOW = 6.5e-3                      # probe B's band, Hz


class AxisBoundsTest(unittest.TestCase):
    """The reachable fdot range is set by the PRIOR, not by mc_lo."""

    def test_bounds_are_the_full_prior_range_at_mc_hi(self):
        """``fdot = fdot_gr(f0,Mc)(1+r)`` is extremised at Mc = mc_hi.

        A tempting wrong answer is ``[fdot_gr(mc_lo)(1-M),
        fdot_gr(mc_hi)(1+M)]``: with mc_lo = 0.001 that lower bound is
        1e-5 of the upper, so the axis would be effectively one-sided and
        the negative-fdot defect would survive the redesign intact.
        """
        lo, hi = fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX)
        g = fdot_gr(F0_FLAG, MC_HI)
        self.assertAlmostEqual(lo / g, 1.0 - RATIO_MAX, places=12)
        self.assertAlmostEqual(hi / g, 1.0 + RATIO_MAX, places=12)

    def test_negative_fdot_is_representable(self):
        """THE coverage defect. Non-negotiable."""
        lo, _ = fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX)
        self.assertLess(lo, 0.0)

    def test_every_prior_allowed_source_is_inside_the_axis(self):
        rng = np.random.default_rng(0)
        mc = rng.uniform(MC_LO, MC_HI, 20000)
        r = rng.uniform(-RATIO_MAX, RATIO_MAX, 20000)
        fd = fdot_gr(F0_FLAG, mc) * (1.0 + r)
        lo, hi = fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX)
        self.assertTrue(np.all(fd >= lo) and np.all(fd <= hi))

    def test_a_ratio_max_below_one_keeps_the_axis_positive(self):
        """M < 1 genuinely cannot reach fdot < 0; the bound must say so."""
        lo, _ = fdot_axis_bounds(F0_FLAG, MC_HI, 0.4)
        self.assertGreater(lo, 0.0)


class NodeSizingTest(unittest.TestCase):
    """Coverage went up 10x; the node count must not."""

    def _n(self, f0, **kw):
        lo, hi = fdot_axis_bounds(f0, MC_HI, RATIO_MAX)
        return n_fdot_nodes(lo, hi, TOBS, **kw)

    def test_high_f_costs_fewer_nodes_than_the_rule_it_replaces(self):
        """The aligned criterion is 13.4x coarser, coverage 10x wider.

        Net: the flagship's ~70-96 Mc nodes over a POSITIVE sliver become
        ~50 fdot nodes over the FULL prior range. Strictly better on both
        axes, which is the whole argument for the redesign at 3 months.
        """
        n = self._n(F0_FLAG, aligned=True)
        self.assertGreater(n, 20)
        self.assertLess(n, 70)

    def test_low_f_collapses_to_the_floor(self):
        """Below ~7 mHz fdot is not measurable over 90 d; 3 nodes is right.

        Same criterion, different frequency -- the current rule's own
        behaviour (the band75 study's 3 nodes at 7.5 mHz), preserved.
        """
        self.assertEqual(self._n(F0_LOW, aligned=True), 3)

    def test_unaligned_is_the_more_expensive_leg(self):
        self.assertGreater(self._n(F0_FLAG, aligned=False),
                           self._n(F0_FLAG, aligned=True))

    def test_clamped_into_the_documented_range(self):
        self.assertEqual(n_fdot_nodes(-1e-6, 1e-6, TOBS), 96)
        self.assertEqual(n_fdot_nodes(0.0, 0.0, TOBS), 3)

    def test_eta_scales_the_count(self):
        a = self._n(F0_FLAG, eta=1.0)
        b = self._n(F0_FLAG, eta=4.0)
        self.assertGreater(a, b)


class McConversionTest(unittest.TestCase):
    """``(f0, fdot) -> (Mc, r)``: the fiber choice and its feasible set."""

    def test_the_mc_floor_is_exactly_the_r_in_range_condition(self):
        """A strong internal consistency check.

        ``Mc >= mc_floor`` and ``|r| <= M`` must be THE SAME condition --
        derived independently, they have to agree, and if they ever stop
        agreeing one of the two derivations is wrong.
        """
        rng = np.random.default_rng(3)
        for f0 in (F0_FLAG, F0_LOW, 1.2e-2):
            lo, hi = fdot_axis_bounds(f0, MC_HI, RATIO_MAX)
            fd = rng.uniform(lo, hi, 4000)
            mc = rng.uniform(MC_LO, MC_HI, 4000)
            floor = mc_floor_for_fdot(fd, f0, RATIO_MAX, MC_LO)
            r = r_from_fdot(fd, f0, mc)
            by_floor = mc >= floor * (1.0 - 1e-12)
            by_r = np.abs(r) <= RATIO_MAX * (1.0 + 1e-12)
            n_bad = int((by_floor != by_r).sum())
            self.assertEqual(n_bad, 0, f"{n_bad} disagreements at f0={f0}")

    def test_round_trip_through_mc_and_r(self):
        rng = np.random.default_rng(5)
        lo, hi = fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX)
        fd = rng.uniform(lo, hi, 5000)
        floor = mc_floor_for_fdot(fd, F0_FLAG, RATIO_MAX, MC_LO)
        mc = floor + rng.random(5000) * (MC_HI - floor)
        r = r_from_fdot(fd, F0_FLAG, mc)
        back = fdot_gr(F0_FLAG, mc) * (1.0 + r)
        np.testing.assert_allclose(back, fd, rtol=1e-12)

    def test_the_feasible_interval_is_never_empty_inside_the_axis(self):
        """Every drawable fdot must admit at least one legal Mc.

        An empty interval would make the birth undrawable for that fdot
        with no error at all -- it would just silently never be proposed.
        """
        for f0 in (F0_FLAG, F0_LOW, 1.2e-2):
            lo, hi = fdot_axis_bounds(f0, MC_HI, RATIO_MAX)
            fd = np.linspace(lo, hi, 2001)
            floor = mc_floor_for_fdot(fd, f0, RATIO_MAX, MC_LO)
            self.assertTrue(np.all(floor <= MC_HI * (1.0 + 1e-12)),
                            f"empty feasible Mc interval at f0={f0}")
            self.assertTrue(np.all(floor >= MC_LO * (1.0 - 1e-12)))

    def test_negative_fdot_needs_a_heavier_chirp_mass(self):
        """Physical sanity: a big negative fdot needs a big |1+r|, and
        ``1 + r >= 1 - M``, so ``fdot_gr`` must be large enough to reach
        it -- i.e. Mc is pushed UP, not down."""
        g = fdot_gr(F0_FLAG, MC_HI)
        near_zero = mc_floor_for_fdot(np.array([-0.01 * g]), F0_FLAG,
                                      RATIO_MAX, MC_LO)[0]
        extreme = mc_floor_for_fdot(np.array([-3.9 * g]), F0_FLAG,
                                    RATIO_MAX, MC_LO)[0]
        self.assertLess(near_zero, extreme)
        self.assertGreater(extreme, 0.9 * MC_HI)

    def test_f0_may_be_per_row(self):
        """REGRESSION. Every real caller passes a per-row ``f0``.

        The birth block derives each row's ``f0`` from its own ``f_mid``
        and ``fdot``, so ``f0`` is an ARRAY there. An earlier version took
        ``float(f0_hz)`` and passed every test in this file only because
        they all handed it a scalar -- the gap surfaced immediately on the
        first integration call.
        """
        rng = np.random.default_rng(11)
        fd = rng.uniform(*fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX),
                         size=500)
        f0 = F0_FLAG * (1.0 + rng.normal(0.0, 1e-4, 500))
        got = mc_floor_for_fdot(fd, f0, RATIO_MAX, MC_LO)
        self.assertEqual(got.shape, (500,))
        # must agree row-by-row with the scalar evaluation
        want = np.array([float(mc_floor_for_fdot(np.array([fd[i]]),
                                                 float(f0[i]), RATIO_MAX,
                                                 MC_LO)[0])
                         for i in range(0, 500, 25)])
        np.testing.assert_allclose(got[::25], want, rtol=1e-14)
        # and it must actually VARY with f0, or the array is being ignored
        flat = mc_floor_for_fdot(fd, F0_FLAG, RATIO_MAX, MC_LO)
        self.assertGreater(float(np.abs(got - flat).max()), 0.0)

    def test_fdot_of_exactly_zero_is_handled(self):
        """``r = -1`` exactly, legal for any Mc; must not divide by zero."""
        floor = mc_floor_for_fdot(np.zeros(3), F0_FLAG, RATIO_MAX, MC_LO)
        np.testing.assert_allclose(floor, MC_LO)
        r = r_from_fdot(np.zeros(3), F0_FLAG, np.full(3, 0.5))
        np.testing.assert_allclose(r, -1.0)

    def test_matches_the_installed_fdot_helper(self):
        """The grid rows are built with gbgpu's get_fdot; do not drift."""
        from gbgpu.utils.utility import get_fdot
        mc = np.array([0.05, 0.3, 0.7, 1.0])
        f0 = np.full(4, F0_FLAG)
        np.testing.assert_allclose(fdot_gr(f0, mc), get_fdot(f=f0, Mc=mc),
                                   rtol=1e-14)


class ShearTest(unittest.TestCase):
    """With fdot a first-class axis the shear is EXACTLY unit-determinant.

    That is strictly cleaner than the shear-only patch this replaces: there
    the grid's fdot depended on its own f0 node (through
    ``fdot_gr(f0, Mc)``), which made the determinant ``1 - 2.6e-4`` rather
    than exactly 1. Here ``fdot`` is independent of ``f0``, so
    ``f0 = f_mid - (T/2) fdot`` is a pure shear and contributes nothing to
    the MH factor.
    """

    def test_the_map_is_an_exact_involution(self):
        rng = np.random.default_rng(7)
        f_mid = F0_FLAG + rng.normal(0, 1e-6, 1000)
        fd = rng.uniform(*fdot_axis_bounds(F0_FLAG, MC_HI, RATIO_MAX),
                         size=1000)
        f0 = f_mid - 0.5 * TOBS * fd
        np.testing.assert_allclose(f0 + 0.5 * TOBS * fd, f_mid, rtol=0,
                                   atol=1e-18)

    def test_determinant_is_one_for_any_coefficient(self):
        for c in (0.0, 0.5 * TOBS, 0.41 * TOBS, TOBS, -3.0 * TOBS):
            J = np.array([[1.0, -c], [0.0, 1.0]])
            self.assertAlmostEqual(float(np.linalg.det(J)), 1.0, places=15)


if __name__ == "__main__":
    unittest.main()
