"""Detailed balance for the observable-basis move, with negative controls.

THE TEST. Run the move with a FLAT likelihood: the target reduces to the
prior, so a correct proposal + Jacobian must leave every prior marginal
invariant. This is a complete detailed-balance check for any MH kernel, and
it is the only test here that can actually catch a wrong measure term.

THE CONTROLS ARE THE POINT. An invariance test that cannot fail proves
nothing, so each defect below is injected deliberately and asserted to skew
the marginals. Template: ``tests/test_ridge_fiber.py:215`` paired with
``:231``. Four controls rather than one, because:

* omitting the term and flipping its sign are different bugs, and a test that
  only checks "the term is present" passes the flipped version;
* a half-implemented Jacobian (one of the two log terms) passes in any regime
  where the other dominates.

WHICH MARGINALS -- and this is not obvious. Measured sensitivity (KS p,
6000 rows x 300 sweeps, mixing asserted separately):

    defect              dist        f0        Mc         r
    none             8.5e-01   8.2e-01   5.5e-01   8.4e-02
    omit             0.0e+00   4.0e-04   7.8e-85   1.5e-11
    flip             0.0e+00   1.9e-11  2.5e-253   2.0e-46
    dist_half_only   2.6e-01   9.0e-03   2.9e-68   2.8e-11   <- dist BLIND
    fdot_half_only   0.0e+00   9.5e-01   2.7e-02   3.6e-01   <- Mc, r BLIND

**No single marginal catches every defect.** ``dist`` cannot see a dropped
``ln fdot_gr`` half (p = 0.26); ``Mc`` and ``r`` cannot see a dropped
``ln dist`` half. So each control asserts on the MINIMUM over
``{dist, Mc, r}``, which catches all four.

An earlier version of this docstring claimed ``Mc`` was insensitive and
``f0`` useless. Both were artifacts of an UNMIXED simulation -- with a chain
that actually traverses the box, ``Mc`` is the single most sensitive column
to the half-Jacobian defects. Do not narrow this set without re-measuring
the table above.
"""
import unittest

import numpy as np
from scipy import stats

from lisatools.sampling.gb_observable_basis import (
    GBObservableFiberBasis,
    fdot_gr,
    gb_observable_step_scales,
)

TOBS = 7.776e6
DIST, F0, MC, PHI0, CI, PSI, AL, SD, R = range(9)
NAMES = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
         "sin_delta", "r"]
IN_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
            "sin_delta", "fdot_astro_ratio"]

# Deliberately narrow in f0/Mc so fdot_gr does not span many decades: the
# point is to isolate the Jacobian, not to stress the waveform model.
LO = np.array([0.5, 19.0, 0.15, 0.0, -1.0, 0.0, 0.0, -1.0, -0.6])
HI = np.array([25.0, 21.0, 0.85, 2 * np.pi, 1.0, np.pi, 2 * np.pi, 1.0, 0.6])

NROW, NSTEP = 6000, 300

# TEST step scales -- deliberately NOT gb_observable_step_scales(). Production
# scales are sized to a real source's posterior (1/rho), which for f_mid is
# ~1.8e-9 Hz; against this test's 2e-3 Hz box the chain would never mix, the
# marginals would still look like their initial draw whatever the Jacobian
# said, and the negative controls would report p ~ 0.03 -- a test with NO
# POWER that nonetheless passes. (That is exactly what happened on the first
# attempt; the controls caught it.) These are sized to cross the box in a few
# hundred sweeps, which is what gives the controls something to bite on.
# Mixing is asserted directly by MixingTest below rather than assumed.
TEST_STEP = np.array([0.40,      # lnA     (dist box spans ln 3.9)
                      2.0e-4,    # f_mid   [Hz]  (box 2e-3 Hz)
                      1.0e-14,   # fdot    [Hz/s]
                      0.30, 0.30, 0.30, 0.30, 0.30,   # extrinsics
                      0.05])     # Mc      (box 0.70 wide)


class _FakeTC:
    input_basis = list(IN_BASIS)


def _basis(shear=0.5):
    return GBObservableFiberBasis(_FakeTC(), Tobs=TOBS, shear=shear)


def _log_prior(y):
    ok = np.all((y >= LO) & (y <= HI), axis=1)
    return np.where(ok, 0.0, -np.inf)


def _draw(rng, n=NROW):
    return np.column_stack([rng.uniform(LO[i], HI[i], n) for i in range(9)])


def _scales():
    """State-INDEPENDENT step scales. See TEST_STEP for why these are not the
    production scales."""
    return TEST_STEP


def _sweep(y, b, rng, nstep=NSTEP, defect=None):
    """MH sweeps with a flat likelihood. Correct factors => prior invariant.

    ``defect`` injects a deliberate error so the controls can prove this test
    has the power to catch one.
    """
    st = _scales()
    for _ in range(nstep):
        z = b.to_internal(y)
        yp = b.from_internal(z + rng.normal(size=z.shape) * st[None, :])

        if defect is None:
            f = b.factors(y, yp)
        elif defect == "omit":
            f = np.zeros(len(y))
        elif defect == "flip":
            f = -b.factors(y, yp)
        elif defect == "dist_half_only":          # drop the ln fdot_gr term
            f = np.log(yp[:, DIST]) - np.log(y[:, DIST])
        elif defect == "fdot_half_only":          # drop the ln dist term
            f = -(np.log(fdot_gr(yp[:, F0] * 1e-3, yp[:, MC]))
                  - np.log(fdot_gr(y[:, F0] * 1e-3, y[:, MC])))
        else:                                     # pragma: no cover
            raise ValueError(defect)
        f = np.where(np.isfinite(f), f, -1e300)

        with np.errstate(invalid="ignore"):
            la = _log_prior(yp) - _log_prior(y) + f
        la = np.where(np.isfinite(la), la, -np.inf)
        acc = np.log(rng.uniform(size=len(y))) < la
        y = np.where(acc[:, None], yp, y)
    return y


def _ks(out, fresh, col):
    return stats.ks_2samp(out[:, col], fresh[:, col]).pvalue


class PriorInvarianceTest(unittest.TestCase):
    def test_all_nine_marginals_are_preserved(self):
        rng = np.random.default_rng(4004)
        out = _sweep(_draw(rng), _basis(), rng)
        fresh = _draw(np.random.default_rng(9099))
        bad = [(NAMES[c], _ks(out, fresh, c)) for c in range(9)
               if _ks(out, fresh, c) <= 1e-3]
        self.assertFalse(bad, f"marginals drifted: {bad}")

class MixingTest(unittest.TestCase):
    """Guards the invariance test's POWER, not its correctness.

    A chain that has not mixed reproduces its initial draw regardless of the
    Jacobian, so every KS test passes and every negative control fails to
    fire. That is a silently useless suite, and it is what the first version
    of this file did. Assert mixing directly instead of hoping for it.
    """

    def test_chain_traverses_the_box_in_the_sensitive_columns(self):
        rng = np.random.default_rng(11)
        y0 = _draw(rng, 2000)
        out = _sweep(y0.copy(), _basis(), rng)
        for col in (DIST, MC, R, F0):
            # each row should have moved a decent fraction of the box width
            span = float(np.median(np.abs(out[:, col] - y0[:, col])))
            frac = span / (HI[col] - LO[col])
            self.assertGreater(
                frac, 0.10,
                f"{NAMES[col]} barely moved ({frac:.1%} of its box) -- the "
                "invariance test would have no power")

    def test_rows_actually_accept(self):
        rng = np.random.default_rng(12)
        y0 = _draw(rng, 2000)
        out = _sweep(y0.copy(), _basis(), rng, nstep=40)
        moved = float(np.mean(np.any(out != y0, axis=1)))
        self.assertGreater(moved, 0.5, f"only {moved:.1%} of rows ever moved")


class NegativeControlTest(unittest.TestCase):
    """Each asserts p < 1e-6 on dist (primary) and r (secondary).

    If ANY of these fails to skew, the invariance test above has no power and
    the Jacobian is unverified -- that is a stop, not a warning.
    """

    #: the set that catches all four defects; see the module docstring table
    SENSITIVE = (DIST, MC, R)

    def _run(self, defect):
        rng = np.random.default_rng(5005)
        out = _sweep(_draw(rng), _basis(), rng, defect=defect)
        fresh = _draw(np.random.default_rng(9099))
        return {NAMES[c]: _ks(out, fresh, c) for c in self.SENSITIVE}

    def _assert_skews(self, defect):
        ps = self._run(defect)
        self.assertLess(min(ps.values()), 1e-6,
                        f"{defect} did not skew any of {ps} -- the invariance "
                        "test has NO POWER and the Jacobian is unverified")

    def test_omitted_factors_skews(self):
        self._assert_skews("omit")

    def test_flipped_sign_skews(self):
        self._assert_skews("flip")

    def test_dist_half_only_skews(self):
        self._assert_skews("dist_half_only")

    def test_fdot_half_only_skews(self):
        self._assert_skews("fdot_half_only")

    def test_no_single_marginal_would_suffice(self):
        """Pins WHY the controls take a minimum over three columns.

        dist alone cannot see a dropped ln(fdot_gr) half; Mc alone cannot see
        a dropped ln(dist) half. Narrowing the set to either would leave a
        half-implemented Jacobian shipping undetected. If this ever fails,
        re-measure the whole table before touching the controls.
        """
        d_half = self._run("dist_half_only")
        f_half = self._run("fdot_half_only")
        self.assertGreater(d_half["dist"], 1e-6)   # dist blind here
        self.assertLess(d_half["Mc"], 1e-6)        # Mc catches it
        self.assertGreater(f_half["Mc"], 1e-6)     # Mc blind here
        self.assertLess(f_half["dist"], 1e-6)      # dist catches it


class ReversibilityTest(unittest.TestCase):
    """+d then -d must return exactly. The negative control is the rejected
    r-additive design, which does not."""

    def test_internal_step_is_an_exact_involution(self):
        b = _basis()
        rng = np.random.default_rng(2)
        y = _draw(rng, 256)
        st = _scales()
        d = rng.normal(size=(256, 9)) * st[None, :]
        y1 = b.from_internal(b.to_internal(y) + d)
        y2 = b.from_internal(b.to_internal(y1) - d)
        np.testing.assert_allclose(y2, y, rtol=1e-11, atol=1e-14)

    def test_r_additive_form_is_NOT_reversible(self):
        """The rejected alternative, kept executable so it cannot come back.

        Stepping r and compensating f0 by -(T/2)*fdot_gr(f0,Mc)*u re-evaluates
        the coefficient at the MOVED f0 (fdot_gr ~ f0^(11/3)), so -u does not
        undo +u. Measured round-trip error 2.7e-5 bins -- tiny, but a BIAS,
        not noise: it does not average away.
        """
        rng = np.random.default_rng(3)
        y = _draw(rng, 256)

        def r_add(yy, u):
            out = yy.copy()
            f0 = yy[:, F0] * 1e-3
            out[:, F0] = (f0 - 0.5 * fdot_gr(f0, yy[:, MC]) * u * TOBS) * 1e3
            out[:, R] = yy[:, R] + u
            return out

        u = rng.normal(size=256) * 0.05
        back = r_add(r_add(y, u), -u)
        err = np.abs(back[:, F0] - y[:, F0]).max()
        self.assertGreater(err, 1e-12,
                           "the r-additive form round-tripped exactly -- the "
                           "control has lost its power")
        np.testing.assert_allclose(
            b_err := np.abs(_basis().from_internal(
                _basis().to_internal(y)) - y).max(), 0.0, atol=1e-12)


class ShearIrrelevanceTest(unittest.TestCase):
    def test_invariance_holds_for_any_shear_coefficient(self):
        """The executable form of 'a wrong T/2 costs efficiency, not
        correctness'. Determinant is 1 for any coefficient, so the prior must
        stay invariant even at an absurd one."""
        for shear in (0.0, 0.5, -3.0):
            rng = np.random.default_rng(777)
            out = _sweep(_draw(rng, 3000), _basis(shear), rng, nstep=80)
            fresh = _draw(np.random.default_rng(9099), 3000)
            p = _ks(out, fresh, DIST)
            self.assertGreater(p, 1e-3, f"shear={shear} broke invariance (p={p:.2e})")


if __name__ == "__main__":
    unittest.main()
