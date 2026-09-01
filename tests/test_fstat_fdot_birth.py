"""The fdot-axis birth block: ``(f0, Mc, r, alpha, sin_delta)``, exact.

WHY IT IS 5-D. The grid draws ``(f_mid, fdot, alpha, sin_delta)`` and the
conversion back needs ``Mc`` AND ``r`` -- ``fdot = fdot_gr(f0,Mc)(1+r)``
cannot be inverted without both. The existing 4-D interface
``(f0, Mc, alpha, sin_delta)`` has no ``r`` in it, so the conversion
cannot live there; the block has to own ``r`` too. Doing so RETIRES
``RatioTightenedBirth``'s blind draw, which is the point: today ``r`` is
drawn by something the grid never scored.

THE TEST THAT MATTERS. ``rvs`` and ``logpdf`` must describe the SAME
distribution. A mismatch does not crash and does not look wrong -- it
silently biases the RJ acceptance ratio in both directions. So the suite
runs the block as an INDEPENDENCE proposal against a flat likelihood: the
chain then targets the prior exactly, and only if ``rvs`` and ``logpdf``
agree. Then each defect is injected deliberately and asserted to skew.

MEASURED SENSITIVITY -- exactly as the tests call it (fresh block, seed 1;
8000 rows x 240 sweeps; KS p vs the prior RESTRICTED to the proposal's
support):

    defect            |        f0        Mc         r   min(f0,Mc,r)
    none              |  8.16e-01  4.72e-01  8.16e-01    4.72e-01
    omit_jac          |  1.65e-04  4.78e-83  7.89e-01    4.78e-83
    flip_jac          |  2.40e-05 3.91e-158  7.61e-01   3.91e-158
    omit_mcwidth      |  7.19e-06  5.36e-26  1.44e-11    5.36e-26
    shear_rvs_only    |  2.76e-30  7.19e-43 1.06e-107   1.06e-107
    mc_full_box_rvs   |  7.34e-03  1.39e-06  2.41e-03    1.39e-06

Controls assert ``min(f0, Mc, r) < 1e-5``: five orders below the correct
block, seven times above the WEAKEST control. ``mc_full_box_rvs`` is that
weakest one (1.4e-6) because half its proposals are infeasible and get
rejected, halving the effective sample -- if the harness is ever changed,
that is the control to re-check first. The bar is set from this table
rather than the table being tuned to hit a round bar.

Never assert on a single marginal. ``Mc`` happens to catch all five here,
but narrowing to one column is exactly the mistake that made the
observable-basis sensitivity table wrong the first time; ``r`` alone would
miss omit_jac and flip_jac entirely (p = 0.79 and 0.76).

Same discipline as ``test_gb_observable_basis_invariance``, and for the
same reason: twice this campaign a control caught a defect in the TEST
rather than in the code. An invariance claim without one proves nothing.
"""
import os
import unittest
from unittest import mock

import numpy as np
from scipy import stats

from lisatools.sampling.fstat_proposal import (
    FdotAxisBirth,
    RatioTightenedBirth,
)
from lisatools.sampling.gb_observable_basis import (
    fdot_axis_bounds,
    fdot_gr,
    mc_floor_for_fdot,
)

TOBS = 7.776e6
MC_LO, MC_HI = 0.05, 1.0          # narrowed vs production so the chain mixes
RATIO_MAX = 5.0
# A REALISTIC peak box. 1 bin = 1/Tobs = 1.286e-4 mHz, and stage-B boxes are
# tens of bins wide -- not the 0.16 mHz an earlier version of this file used.
# That matters for power, not just realism: the shear excursion here is
# 0.0085 mHz, i.e. LARGER than the box, which is exactly the high-frequency
# regime the redesign exists for. Against the old wide box the shear was 5%
# of the box and its control could not fire (measured p = 0.016 even with the
# f_mid tilt below).
F0_LO, F0_HI = 20.3755, 20.3855   # mHz, ~78 bins around the flagship
# The axis reference must be the LOWEST f0 any row can reach, not the peak:
# the reachable |fdot| goes as f0^(11/3), and f0 = f_mid - (T/2) fdot dips
# below the f_mid box. Size at the peak instead and ~0.4% of draws land with
# an EMPTY feasible Mc interval. This mirrors the intersection rule in
# run_stacked_stage_b.
F0_REF = (F0_LO - 0.02) * 1e-3    # Hz, the group's constant reference
F0_REF_MISSIZED = 20.46e-3        # deliberately at the TOP, for the control

NROW, NSTEP = 8000, 240
F0, MC, R, AL, SD = 0, 1, 2, 3, 4
NAMES = ["f0", "Mc", "r", "alpha", "sin_delta"]


#: log-density tilt across the f_mid box. A SHEAR IS A TRANSLATION IN
#: f_mid, and a uniform density is translation-invariant except at its
#: edges -- so against a flat grid the shear control has NO POWER (measured:
#: it reported p = 0.06 while the defect was fully active). The tilt gives
#: it something to bite on. It is applied to f_mid ONLY: the other three
#: axes stay flat so the Mc-floor and Jacobian controls cannot be masked by
#: grid structure in the direction they act on.
FMID_TILT = 6.0


class _FlatGrid:
    """Stand-in 4-D grid: tilted in ``f_mid``, flat in the other three.

    ``p(f_mid) ~ exp(FMID_TILT * (f_mid - lo) / (hi - lo))``, exactly
    normalised, with independent uniforms on ``(fdot, alpha, sin_delta)``.
    See FMID_TILT for why f_mid is not flat and why the rest is.
    """

    param_names = ("f_mid", "fdot", "alpha", "sin_delta")
    ndim = 4

    def __init__(self, seed=0, f0_ref=F0_REF):
        _lo, _hi = fdot_axis_bounds(f0_ref, MC_HI, RATIO_MAX)
        self.lo = np.array([F0_LO, _lo, 0.0, -1.0])
        self.hi = np.array([F0_HI, _hi, 2 * np.pi, 1.0])
        self._rng = np.random.default_rng(seed)
        self._w = self.hi - self.lo
        # flat part: the last three axes
        self._lnV3 = float(np.log(self._w[1:]).sum())
        k = FMID_TILT
        # exact normaliser of exp(k*u) on u in [0,1], u = (f_mid-lo)/w
        self._ln_norm = float(np.log((np.exp(k) - 1.0) / k) + np.log(self._w[0]))

    def rvs(self, size=1):
        n = int(np.prod(size)) if not isinstance(size, int) else int(size)
        out = self.lo + self._rng.random((n, 4)) * self._w
        # inverse-CDF for exp(k*u) on [0,1]
        u = self._rng.random(n)
        k = FMID_TILT
        out[:, 0] = self.lo[0] + self._w[0] * np.log1p(u * (np.exp(k) - 1.0)) / k
        return out

    def logpdf(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        inside = np.all((x >= self.lo) & (x <= self.hi), axis=1)
        u = (x[:, 0] - self.lo[0]) / self._w[0]
        lp = FMID_TILT * u - self._ln_norm - self._lnV3
        return np.where(inside, lp, -np.inf)


def _block(defect=None, seed=0, f0_ref=F0_REF):
    return FdotAxisBirth(_FlatGrid(seed=seed, f0_ref=f0_ref), tobs=TOBS,
                         mc_lo=MC_LO, mc_hi=MC_HI, ratio_max=RATIO_MAX,
                         seed=seed, _defect=defect)


# ---- the prior the chain must reproduce ---------------------------------
# f0 is WIDER than the grid's f_mid box on purpose: f0 = f_mid - (T/2) fdot
# carries rows outside it, by up to 0.0085 mHz at this fdot range. Equating
# the two boxes is a category error -- the grid box is a PEAK box, the prior
# box is the band.
_SHEAR_PAD = 0.02
PLO = np.array([F0_LO - _SHEAR_PAD, MC_LO, -RATIO_MAX, 0.0, -1.0])
PHI = np.array([F0_HI + _SHEAR_PAD, MC_HI, RATIO_MAX, 2 * np.pi, 1.0])


def _prior_logpdf(x):
    inside = np.all((x >= PLO) & (x <= PHI), axis=1)
    return np.where(inside, 0.0, -np.inf)


def _prior_rvs(n, rng):
    return PLO + rng.random((n, 5)) * (PHI - PLO)


def _independence_chain(block, nrow=NROW, nstep=NSTEP, seed=1,
                        return_rate=False):
    """MH with ``block`` as an INDEPENDENCE proposal, flat likelihood.

    Stationary distribution is the prior IF AND ONLY IF ``rvs`` and
    ``logpdf`` describe one distribution:

        lnpdiff = [logpi(y) - logpi(x)] + [logq(x) - logq(y)]
    """
    rng = np.random.default_rng(seed)
    x = _prior_rvs(nrow, rng)
    # start from prior draws the proposal can actually reach
    lq_x = np.asarray(block.logpdf(x))
    keep = np.isfinite(lq_x)
    x, lq_x = x[keep], lq_x[keep]
    lp_x = _prior_logpdf(x)
    n = x.shape[0]
    n_acc = 0
    for _ in range(nstep):
        y = np.asarray(block.rvs(n))
        lq_y = np.asarray(block.logpdf(y))
        lp_y = _prior_logpdf(y)
        with np.errstate(invalid="ignore"):
            d = (lp_y - lp_x) + (lq_x - lq_y)
        acc = np.log(rng.random(n)) < np.where(np.isfinite(d), d, -np.inf)
        n_acc += int(acc.sum())
        x = np.where(acc[:, None], y, x)
        lp_x = np.where(acc, lp_y, lp_x)
        lq_x = np.where(acc, lq_y, lq_x)
    return (x, n_acc / float(n * nstep)) if return_rate else x


def _reference(block, n, seed):
    """Prior draws RESTRICTED to the proposal's support.

    An independence proposal whose support is a strict subset S of the
    prior targets ``pi|S``, not ``pi`` -- and S here is a SHEARED region
    (the f_mid box pushed through ``f0 = f_mid - (T/2) fdot``), not a box.
    Comparing against unrestricted prior draws would report that geometry
    as a defect and mask the real ones. In production the uniform floor in
    ``UniformFloorMixture`` is what restores full support.
    """
    rng = np.random.default_rng(seed)
    out = []
    got = 0
    while got < n:
        c = _prior_rvs(4 * n, rng)
        c = c[np.isfinite(np.asarray(block.logpdf(c)))]
        out.append(c)
        got += c.shape[0]
    return np.concatenate(out)[:n]


def _ks_table(block, seed=1):
    x = _independence_chain(block, seed=seed)
    # Restrict by the CORRECT block, never by the one under test: a defect
    # that shrinks its own support would otherwise be absorbed into the
    # reference and hide itself. Measured -- restricting by the defective
    # block dropped the shear control from skewing to p = 0.078.
    ref = _reference(_block(), x.shape[0], seed + 999)
    return {NAMES[c]: float(stats.ks_2samp(x[:, c], ref[:, c]).pvalue)
            for c in range(5)}


class RoundTripTest(unittest.TestCase):

    def test_rvs_rows_are_inside_the_prior_box(self):
        x = np.asarray(_block().rvs(20000))
        for c, nm in enumerate(NAMES):
            self.assertGreaterEqual(float(x[:, c].min()), PLO[c] - 1e-9, nm)
            self.assertLessEqual(float(x[:, c].max()), PHI[c] + 1e-9, nm)

    def test_every_drawn_row_has_finite_logpdf(self):
        """rvs must never produce a row its own logpdf calls impossible.

        That combination is the classic silent RJ bug: the birth is made,
        priced at -inf, and rejected forever -- so the move looks merely
        inefficient rather than broken.
        """
        b = _block()
        lp = np.asarray(b.logpdf(np.asarray(b.rvs(20000))))
        self.assertTrue(np.all(np.isfinite(lp)),
                        f"{int((~np.isfinite(lp)).sum())} unreachable draws")

    def test_negative_fdot_is_actually_produced(self):
        """THE coverage defect. The current grid CANNOT do this at all."""
        x = np.asarray(_block().rvs(20000))
        fd = fdot_gr(x[:, F0] * 1e-3, x[:, MC]) * (1.0 + x[:, R])
        frac = float((fd < 0).mean())
        self.assertGreater(frac, 0.05, f"only {frac:.3%} negative-fdot draws")

    def test_the_shear_is_applied_to_f0(self):
        """``f0 = f_mid - (T/2) fdot``, checked by EXACT reconstruction.

        Not by correlation: the shear excursion is 0.0085 mHz against a
        0.16 mHz box, so corr(f0, fdot) is about -0.09 here and a
        correlation threshold would be measuring the box aspect ratio
        rather than the code. Recovering f_mid and finding it back inside
        the grid box is exact and cannot be passed by accident.
        """
        b = _block()
        x = np.asarray(b.rvs(20000))
        fd = fdot_gr(x[:, F0] * 1e-3, x[:, MC]) * (1.0 + x[:, R])
        f_mid = x[:, F0] * 1e-3 + 0.5 * TOBS * fd
        self.assertGreaterEqual(float(f_mid.min()) * 1e3, F0_LO - 1e-9)
        self.assertLessEqual(float(f_mid.max()) * 1e3, F0_HI + 1e-9)
        # and the shear must actually MOVE f0 off f_mid
        self.assertGreater(float(np.abs(f_mid * 1e3 - x[:, F0]).max()),
                           0.004)

    def test_prior_rows_outside_the_grid_are_minus_inf_not_nan(self):
        b = _block()
        x = _prior_rvs(500, np.random.default_rng(2))
        x[:, R] = 50.0                     # far outside |r| <= M
        lp = np.asarray(b.logpdf(x))
        self.assertTrue(np.all(lp == -np.inf))
        self.assertFalse(np.any(np.isnan(lp)))


class MixingTest(unittest.TestCase):
    """Without mixing the controls below have no power. Assert it directly.

    This is not hypothetical: the first version of the observable-basis
    suite reused production step scales, never traversed, and reported a
    broken Jacobian as p = 0.028 -- a suite that passes and proves nothing.
    """

    def test_the_chain_actually_accepts_moves(self):
        """Measured 0.172 for the correct block. A chain that never
        accepts reproduces its initial draw whatever the measure says."""
        _, rate = _independence_chain(_block(), seed=1, return_rate=True)
        self.assertGreater(rate, 0.05, f"acceptance {rate:.4f}")

    def test_the_chain_spread_matches_the_target_support(self):
        """Against the TARGET (pi restricted to the proposal's support),
        not against the prior box: the box is 0.05 mHz wide in f0 while
        the support is the sheared band, so a box comparison reports a
        perfectly healthy chain as collapsed."""
        b = _block()
        x = _independence_chain(b)
        ref = _reference(b, x.shape[0], 12345)
        for c in (F0, MC, R):
            ratio = float(x[:, c].std()) / float(ref[:, c].std())
            self.assertGreater(ratio, 0.5, f"{NAMES[c]} under-dispersed")
            self.assertLess(ratio, 2.0, f"{NAMES[c]} over-dispersed")


class ExactnessTest(unittest.TestCase):
    """rvs and logpdf describe one distribution -- with controls."""

    def test_the_prior_is_preserved(self):
        p = _ks_table(_block())
        self.assertGreater(min(p.values()), 1e-3, p)

    def test_control_omitted_jacobian_skews(self):
        """Drop ``+ log fdot_gr`` (the dfdot/dr term)."""
        p = _ks_table(_block(defect="omit_jac"))
        self.assertLess(min(p[k] for k in ("f0", "Mc", "r")), 1e-5, p)

    def test_control_flipped_jacobian_skews(self):
        """A test that only checks the term is PRESENT passes this one."""
        p = _ks_table(_block(defect="flip_jac"))
        self.assertLess(min(p[k] for k in ("f0", "Mc", "r")), 1e-5, p)

    def test_control_omitted_mc_width_skews(self):
        """Drop ``- log(mc_hi - mc_floor)``, the Mc draw's own density.

        The floor depends on ``(fdot, f0)``, so omitting it is not a
        constant -- it is a state-dependent tilt, which is exactly the
        kind of error that hides.
        """
        p = _ks_table(_block(defect="omit_mcwidth"))
        self.assertLess(min(p[k] for k in ("f0", "Mc", "r")), 1e-5, p)

    def test_control_shear_in_rvs_only_skews(self):
        """THE most important control.

        Applying the shear when drawing but not when pricing (or vice
        versa) silently biases the RJ ratio in BOTH directions and raises
        no error anywhere. Nothing else in the suite catches it.
        """
        p = _ks_table(_block(defect="shear_rvs_only"))
        self.assertLess(min(p[k] for k in ("f0", "Mc", "r")), 1e-5, p)

    def test_control_mc_drawn_over_the_full_box_skews(self):
        """rvs ignores the feasible floor; logpdf keeps it.

        The ASYMMETRY is the control. Applying the widened interval to
        both paths leaves the proposal self-consistent and it does not
        skew (measured KS p = 0.005) -- a defect injected into a shared
        helper is not a defect. It has to break rvs against logpdf.
        """
        p = _ks_table(_block(defect="mc_full_box_rvs"))
        self.assertLess(min(p[k] for k in ("f0", "Mc", "r")), 1e-5, p)


class DefectRegistryTest(unittest.TestCase):
    """Every control must be a real code path, not a silently ignored name."""

    def test_an_unknown_defect_name_raises(self):
        with self.assertRaises(ValueError):
            _block(defect="not_a_defect")

    def test_the_default_block_has_no_defect(self):
        self.assertIsNone(_block()._defect)


class ContainerWiringTest(unittest.TestCase):
    """``make_gb_rj_birth_container`` under FSTAT_FDOT_AXIS.

    The block only matters if it is actually REACHED, and reaching it
    means changing the eryn tuple key from 4 columns to 5 -- which also
    has to retire the separate ``U[-M, M]`` ratio column and
    ``RatioTightenedBirth``. Get that wrong and the run silently keeps
    the blind r-draw the redesign exists to remove.
    """

    A_LIMS = [1e-24, 1e-20]
    RT = dict(tobs=TOBS, phase_rad=2 * np.pi, eps=0.1, w_min=0.05)

    def _make(self, armed, **kw):
        from lisatools.sampling.fstat_proposal import (
            make_gb_rj_birth_container)
        kw.setdefault("tobs", TOBS)
        kw.setdefault("mc_lims", [MC_LO, MC_HI])
        env = {"FSTAT_FDOT_AXIS": "1" if armed else "0"}
        with mock.patch.dict(os.environ, env):
            return make_gb_rj_birth_container(
                _FlatGrid(), self.A_LIMS, fdot_astro_ratio_max=RATIO_MAX,
                dist_lims=[0.1, 30.0], ratio_tight=self.RT, **kw)

    @staticmethod
    def _keys(dist):
        base = getattr(dist, "base", dist)
        return set(base.priors_in.keys())

    def test_disarmed_keeps_the_4d_key_and_the_uniform_ratio(self):
        d = self._make(armed=False)
        k = self._keys(d)
        self.assertIn(("f0", "Mc", "alpha", "sin_delta"), k)
        self.assertIn("fdot_astro_ratio", k)
        self.assertIsInstance(d, RatioTightenedBirth)

    def test_armed_uses_the_5d_key_and_retires_the_blind_draw(self):
        d = self._make(armed=True)
        k = self._keys(d)
        self.assertIn(("f0", "Mc", "fdot_astro_ratio", "alpha", "sin_delta"),
                      k)
        self.assertNotIn(("f0", "Mc", "alpha", "sin_delta"), k)
        self.assertNotIn("fdot_astro_ratio", k,
                         "the separate U[-M,M] ratio column must be gone")
        self.assertNotIsInstance(
            d, RatioTightenedBirth,
            "RatioTightenedBirth patches a draw the grid never scored; "
            "under the fdot axis the grid scores it")

    def test_armed_without_tobs_raises_rather_than_defaulting(self):
        """A getattr default of 0.0 silently disables the shear."""
        with self.assertRaises(ValueError):
            self._make(armed=True, tobs=None)

    def test_the_container_round_trips_nine_columns(self):
        for armed in (False, True):
            d = self._make(armed=armed)
            x = np.asarray(d.rvs(size=400))
            self.assertEqual(x.shape, (400, 9), f"armed={armed}")
            lp = np.asarray(d.logpdf(x))
            self.assertEqual(lp.shape, (400,), f"armed={armed}")
            frac = float(np.isfinite(lp).mean())
            self.assertGreater(frac, 0.95,
                               f"armed={armed}: only {frac:.1%} scoreable")


if __name__ == "__main__":
    unittest.main()
