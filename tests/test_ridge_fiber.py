"""Tests for the GB (Mc, ratio, distance) ridge-Gibbs fiber move.

Covers the fiber algebra (round trip, bounds), the core correctness claim
(EXACT physical (A, f0, fdot) invariance through the installed production
transform -- the waveform cannot change), the MH correctness of the
fiber-measure term (prior invariance under repeated sweeps, plus a negative
control with the measure term omitted), and an eryn-level smoke test that the
move never touches log_like and keeps log_prior consistent.
"""

import numpy as np
import pytest
from scipy import stats

from eryn.model import Model
from eryn.moves import RidgeGibbsMove
from eryn.priors import ProbDistContainer, UniformDistribution
from eryn.state import State
from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container
from lisatools.sampling.ridge_fiber import McRatioDistFiber, make_gb_ridge_gibbs_move

MC_LIMS = (0.05, 1.0)
DIST_LIMS = (1.0, 30.0)
RATIO_MAX = 5.0
F0_LIMS_MHZ = (2.0, 30.0)

# sampling basis: [dist, f0(mHz), Mc, phi0, cos_iota, psi, alpha, sin_delta, r]
NDIM = 9


def _transform_container():
    return make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True
    )


def _fiber(ratio_max=RATIO_MAX, mc_lims=MC_LIMS, dist_lims=DIST_LIMS):
    return McRatioDistFiber(_transform_container(), mc_lims, dist_lims, ratio_max)


def _priors():
    return ProbDistContainer(
        {
            0: UniformDistribution(*DIST_LIMS),
            1: UniformDistribution(*F0_LIMS_MHZ),
            2: UniformDistribution(*MC_LIMS),
            3: UniformDistribution(0.0, 2.0 * np.pi),
            4: UniformDistribution(-1.0, 1.0),
            5: UniformDistribution(0.0, np.pi),
            6: UniformDistribution(0.0, 2.0 * np.pi),
            7: UniformDistribution(-1.0, 1.0),
            8: UniformDistribution(-RATIO_MAX, RATIO_MAX),
        }
    )


def _random_rows(n, rng):
    """Random 9-column sampling rows inside the prior box."""
    rows = np.empty((n, NDIM))
    rows[:, 0] = rng.uniform(*DIST_LIMS, size=n)
    rows[:, 1] = rng.uniform(*F0_LIMS_MHZ, size=n)
    rows[:, 2] = rng.uniform(*MC_LIMS, size=n)
    rows[:, 3] = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rows[:, 4] = rng.uniform(-1.0, 1.0, size=n)
    rows[:, 5] = rng.uniform(0.0, np.pi, size=n)
    rows[:, 6] = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rows[:, 7] = rng.uniform(-1.0, 1.0, size=n)
    rows[:, 8] = rng.uniform(-RATIO_MAX, RATIO_MAX, size=n)
    return rows


def _propose_in_bounds(fiber, rows, rng):
    """One fiber proposal for every row with a nondegenerate interval.

    Returns (new_rows, valid_mask, u, u_new) with new_rows only for valid rows.
    """
    u, invariants = fiber.to_fiber(rows)
    u_lo, u_hi = fiber.fiber_bounds(invariants)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(u_lo) & np.isfinite(u_hi) & (u_hi > u_lo)
    sub = {key: value[valid] for key, value in invariants.items()}
    u_new = u_lo[valid] + (u_hi[valid] - u_lo[valid]) * rng.uniform(
        size=int(valid.sum())
    )
    new_rows = fiber.from_fiber(u_new, sub, rows[valid])
    return new_rows, valid, u[valid], u_new


def test_basis_validation():
    """8-column bases (no Mc/dist/ratio) must be rejected loudly."""
    tc8 = make_gb_transform_container(use_chirp_mass=False)
    with pytest.raises(ValueError, match="not eligible"):
        McRatioDistFiber(tc8, MC_LIMS, DIST_LIMS, RATIO_MAX)


def test_round_trip():
    """to_fiber -> from_fiber reproduces (Mc, r, d) to 1e-12 for 10^4 rows."""
    rng = np.random.default_rng(1001)
    fiber = _fiber()
    rows = _random_rows(10_000, rng)

    u, invariants = fiber.to_fiber(rows)
    back = fiber.from_fiber(u, invariants, rows)

    np.testing.assert_allclose(back, rows, rtol=1e-12, atol=1e-12)


def test_exact_likelihood_invariance():
    """Physical (A, f0, fdot) via the INSTALLED production transform are
    unchanged by the fiber move to reldiff < 1e-10 (the waveform cannot
    change)."""
    rng = np.random.default_rng(2002)
    fiber = _fiber()
    tc = _transform_container()

    rows = _random_rows(4000, rng)
    # keep 1 + r away from the exact zero (measure-zero fdot = 0 line where
    # a RELATIVE comparison is meaningless)
    rows = rows[np.abs(1.0 + rows[:, 8]) > 1e-3]

    new_rows, valid, _, _ = _propose_in_bounds(fiber, rows, rng)
    assert valid.sum() > 1000  # the fiber intervals must not be empty in bulk

    phys_old = tc.both_transforms(rows[valid])
    phys_new = tc.both_transforms(new_rows)

    # output basis: [A, f0, fdot, fddot, phi0, iota, psi, alpha, delta]
    for name, col in (("A", 0), ("f0", 1), ("fdot", 2)):
        old = phys_old[:, col]
        new = phys_new[:, col]
        reldiff = np.abs(new - old) / np.abs(old)
        assert np.max(reldiff) < 1e-10, f"{name} changed: max reldiff {reldiff.max()}"

    # the untouched columns pass through bit-identically
    np.testing.assert_array_equal(phys_old[:, 3:], phys_new[:, 3:])


def test_bounds_and_empty_intervals():
    """Proposed points respect all three prior boxes; empty intervals are
    skipped, not crashed."""
    rng = np.random.default_rng(3003)
    fiber = _fiber()
    rows = _random_rows(10_000, rng)

    new_rows, valid, _, _ = _propose_in_bounds(fiber, rows, rng)
    assert valid.all()  # in-prior rows always contain their own fiber point

    tol = 1e-12
    mc = new_rows[:, 2]
    d = new_rows[:, 0]
    r = new_rows[:, 8]
    assert np.all(mc >= MC_LIMS[0] * (1 - tol)) and np.all(
        mc <= MC_LIMS[1] * (1 + tol)
    )
    assert np.all(d >= DIST_LIMS[0] * (1 - tol)) and np.all(
        d <= DIST_LIMS[1] * (1 + tol)
    )
    assert np.all(r >= -RATIO_MAX - tol) and np.all(r <= RATIO_MAX + tol)

    # Kf < 0 with a tight ratio box (M < 1): r' < -1 is unreachable -> empty
    fiber_tight = _fiber(ratio_max=0.5)
    bad_rows = _random_rows(16, rng)
    bad_rows[:, 8] = -2.0  # Kf < 0
    u, invariants = fiber_tight.to_fiber(bad_rows)
    u_lo, u_hi = fiber_tight.fiber_bounds(invariants)
    assert np.all(u_hi <= u_lo)

    # Kf < 0 with M > 1: the ratio floor u >= |Kf|/(M - 1) can climb above
    # the Mc/dist ceiling (tight bounds -> empty). r = -100 (out of the
    # M = 1.5 box) gives |Kf|/(M - 1) = 198 u >> u_mc_hi = 1.
    fiber_wide = _fiber(ratio_max=1.5)
    tight = _random_rows(16, rng)
    tight[:, 2] = 0.9
    tight[:, 8] = -100.0  # Kf < 0, far outside the tight ratio box
    u, invariants = fiber_wide.to_fiber(tight)
    u_lo, u_hi = fiber_wide.fiber_bounds(invariants)
    assert np.all(u_hi <= u_lo)

    # move-level: a state full of empty-interval leaves is a no-op, no crash
    priors = _priors()
    move = RidgeGibbsMove(
        branch_name="gb",
        fiber_map=fiber_tight,
        per_leaf_log_prior=priors.logpdf,
    )
    coords = bad_rows.reshape(1, 4, 4, NDIM).copy()
    state = State(
        {"gb": coords},
        log_like=np.zeros((1, 4)),
        log_prior=np.zeros((1, 4)),
    )
    model = _mock_model(priors, np.random.RandomState(7))
    new_state, accepted = move.propose(model, state)
    assert not accepted.any()
    np.testing.assert_array_equal(new_state.branches["gb"].coords, coords)


def _sweep(rows, fiber, log_prior_fn, rng, include_measure=True):
    """One standalone sweep of the exact move mechanics over (n, ndim) rows."""
    new_rows, valid, u, u_new = _propose_in_bounds(fiber, rows, rng)
    dlogp = log_prior_fn(new_rows) - log_prior_fn(rows[valid])
    log_alpha = dlogp.copy()
    if include_measure:
        inv = {}  # measure depends only on u here
        log_alpha += fiber.log_fiber_measure(u_new, inv) - fiber.log_fiber_measure(
            u, inv
        )
    accept = np.log(rng.uniform(size=len(u))) < log_alpha
    out = rows.copy()
    idx = np.where(valid)[0][accept]
    out[idx] = new_rows[accept]
    return out


def test_prior_invariance():
    """200 sweeps of the move mechanics leave the (Mc, r, d) prior marginals
    unchanged (KS p > 1e-3 against fresh prior draws)."""
    rng = np.random.default_rng(4004)
    priors = _priors()
    rows = np.asarray(priors.rvs(size=20_000))

    for _ in range(200):
        rows = _sweep(rows, _fiber(), priors.logpdf, rng)

    fresh = np.asarray(priors.rvs(size=20_000))
    for name, col in (("Mc", 2), ("r", 8), ("dist", 0)):
        p = stats.ks_2samp(rows[:, col], fresh[:, col]).pvalue
        assert p > 1e-3, f"{name} marginal drifted (KS p = {p:.2e})"


def test_prior_invariance_negative_control():
    """Omitting the (2/5) ln(u/u') measure term visibly skews the Mc
    marginal -- confirms the invariance test has the power to catch a wrong
    Jacobian."""
    rng = np.random.default_rng(5005)
    priors = _priors()
    rows = np.asarray(priors.rvs(size=20_000))

    for _ in range(200):
        rows = _sweep(rows, _fiber(), priors.logpdf, rng, include_measure=False)

    fresh = np.asarray(priors.rvs(size=20_000))
    p = stats.ks_2samp(rows[:, 2], fresh[:, 2]).pvalue
    assert p < 1e-6, f"negative control failed to skew Mc (KS p = {p:.2e})"


def _mock_model(priors, random_state):
    """Model namedtuple whose likelihood must never be called."""

    def _never_called(*args, **kwargs):  # pragma: no cover - the assertion
        raise AssertionError("RidgeGibbsMove must make zero likelihood calls.")

    def _compute_log_prior(coords, inds=None):
        c = coords["gb"]
        ntemps, nwalkers, nleaves, _ = c.shape
        out = np.zeros((ntemps, nwalkers))
        alive = inds["gb"]
        it, iw, il = np.where(alive)
        vals = priors.logpdf(c[it, iw, il])
        np.add.at(out, (it, iw), vals)
        return out

    return Model(
        log_like_fn=_never_called,
        compute_log_like_fn=_never_called,
        compute_log_prior_fn=_compute_log_prior,
        temperature_control=None,
        map_fn=map,
        random=random_state,
    )


def test_gf_substate_cold_row_sync():
    """Global-fit wiring: the move samples the MAIN (engine) state while the
    band moves' working ensemble lives on a tempered ``ModuleSubState``.
    After every propose the sub-state's cold row must match the main state
    (``check_cold_row`` is the production invariant that MPI-aborted the
    2026-08-20 v5/v6 runs at gb_search it=1) and the hot rungs must be
    untouched -- the fiber move runs on the cold chain only."""
    from lisatools.globalfit.state import GFState, ModuleSubState

    ntemps_sub, nwalkers, nleaves = 3, 4, 3
    priors = _priors()
    model = _mock_model(priors, np.random.RandomState(43))

    # main engine state: cold chain only (ntemps = 1), one dead leaf
    coords = np.asarray(priors.rvs(size=(1, nwalkers, nleaves)))
    inds = np.ones((1, nwalkers, nleaves), dtype=bool)
    inds[0, 0, 2] = False
    state = GFState(
        {"gb": coords.copy()},
        inds={"gb": inds.copy()},
        log_like=np.zeros((1, nwalkers)),
        log_prior=model.compute_log_prior_fn({"gb": coords}, inds={"gb": inds}),
        is_eryn_state_input=True,
        sub_state_bases={"gb": ModuleSubState},
    )

    # the module's tempered ladder: row 0 mirrors the main state's cold row
    ladder = np.asarray(priors.rvs(size=(ntemps_sub, nwalkers, nleaves)))
    ladder[0] = coords[0]
    ladder_inds = np.broadcast_to(inds[0], (ntemps_sub, nwalkers, nleaves))
    sub = state.sub_states["gb"]
    sub.initialize_tempered(
        ntemps_sub, nwalkers, nleaves, NDIM, coords=ladder, inds=ladder_inds
    )
    hot_before = sub.coords[1:].copy()

    move = make_gb_ridge_gibbs_move(
        priors, _transform_container(), MC_LIMS, DIST_LIMS, RATIO_MAX
    )

    any_accepted = False
    for _ in range(30):
        state, accepted = move.propose(model, state)
        any_accepted |= bool(accepted.any())

        # the state copy must stay a GFState with a live sub-state
        sub = state.sub_states["gb"]
        assert sub is not None and sub.tempered_initialized

        # the production consistency invariant (raises on mismatch)
        sub.check_cold_row(state, "gb")

        # cold-chain-only: hot rungs never move
        np.testing.assert_array_equal(sub.coords[1:], hot_before)

    assert any_accepted  # the move must actually mix
    # accepted fiber jumps must have REACHED the sub-state (not just main)
    assert not np.array_equal(
        sub.coords[0][ladder_inds[0]], coords[0][inds[0]]
    )


def test_eryn_integration_smoke():
    """2 temps x 4 walkers x 3 leaves, 50 propose calls: log_like never
    changes, log_prior stays consistent with a from-scratch recomputation,
    coords stay in bounds, dead leaves stay untouched."""
    ntemps, nwalkers, nleaves = 2, 4, 3
    rng = np.random.default_rng(6006)
    priors = _priors()

    coords = np.asarray(priors.rvs(size=(ntemps, nwalkers, nleaves)))
    inds = np.ones((ntemps, nwalkers, nleaves), dtype=bool)
    inds[0, 0, 2] = False  # one dead leaf: must never be touched
    dead_row = coords[0, 0, 2].copy()

    model = _mock_model(priors, np.random.RandomState(42))
    log_like = np.asarray(model.random.randn(ntemps, nwalkers))
    state = State(
        {"gb": coords},
        inds={"gb": inds},
        log_like=log_like.copy(),
        log_prior=model.compute_log_prior_fn(
            {"gb": coords}, inds={"gb": inds}
        ),
    )

    move = make_gb_ridge_gibbs_move(
        priors, _transform_container(), MC_LIMS, DIST_LIMS, RATIO_MAX,
        leaf_fraction=0.7,
    )

    any_accepted = False
    for _ in range(50):
        state, accepted = move.propose(model, state)
        any_accepted |= bool(accepted.any())

        # the zero-likelihood-call invariant
        np.testing.assert_array_equal(state.log_like, log_like)

        # log_prior consistent with recomputation from scratch
        recomputed = model.compute_log_prior_fn(
            state.branches_coords, inds=state.branches_inds
        )
        assert np.max(np.abs(state.log_prior - recomputed)) < 1e-10

        # coords in bounds on alive leaves; dead leaf untouched
        c = state.branches["gb"].coords
        it, iw, il = np.where(state.branches["gb"].inds)
        alive = c[it, iw, il]
        assert np.all(alive[:, 2] >= MC_LIMS[0]) and np.all(alive[:, 2] <= MC_LIMS[1])
        assert np.all(alive[:, 0] >= DIST_LIMS[0]) and np.all(
            alive[:, 0] <= DIST_LIMS[1]
        )
        assert np.all(np.abs(alive[:, 8]) <= RATIO_MAX)
        np.testing.assert_array_equal(state.branches["gb"].coords[0, 0, 2], dead_row)

    assert any_accepted  # the move must actually mix
    assert move.num_proposals == 50
