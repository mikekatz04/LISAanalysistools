"""Tests for the model-dependent terms of :class:`HyperMove` (diagnostic D1).

The move is exercised with mock priors and a mock analysis-container array, so no
waveform or flow is evaluated; :meth:`HyperMove.compute_expected_resolved_counts` is
replaced by a fixed array. Importing ``lisatools.globalfit`` pulls in CuPy and GBGPU,
so these tests need a CUDA capable node and are skipped elsewhere.

The central check is ``test_matches_previous_grouping``: the refactored
:meth:`HyperMove.compute_model_terms` must reproduce, term by term, the quantity the
move used to build out of ``compute_source_contribution`` and
``compute_number_contribution``, so that the regrouping into intensity + expected
count + stochastic changes only the bookkeeping, never the acceptance ratio.

The reference for that check is :func:`previous_grouping`, which transcribes those two
methods from commit ``861dc05`` -- the last commit before the regrouping, and the point
at which the expression was already correct. Neither method survives in the tree, so the
reference has to be written out by hand; the one thing it must not do is rebuild the old
quantity out of ``compute_model_terms``'s own return values, which is what an earlier
version of this test did and which reduces the assertion to the commutativity of
addition.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
if not cupy.cuda.is_available():
    # importing lisatools.globalfit initialises CUDA and aborts without a device,
    # which pytest cannot turn into a skip -- so bail out before importing it
    pytest.skip("requires a CUDA device", allow_module_level=True)

hypermove = pytest.importorskip("lisatools.globalfit.moves.hypermove")
hyperdiagnostics = pytest.importorskip("lisatools.globalfit.hyperdiagnostics")

HyperMove = hypermove.HyperMove
HyperMoveRecorder = hyperdiagnostics.HyperMoveRecorder
HyperMoveDiagnostics = hyperdiagnostics.HyperMoveDiagnostics


NMODELS = 2
NTEMPS = 3
NWALKERS = 4
NLEAVES = 5
NDIM_RESOLVED = 3
NDIM_STOCHASTIC = 2
N_TOT = (1.0e6, 4.0e6)


class MockContainer:
    """Stand-in for ``ProbDistContainer``: a model dependent log density.

    ``log p(x | m) = -(m + 1) * sum(x ** 2) - |x[0]|``, so that the model dependence is
    genuine, every coordinate of every source matters, and the density is defined for
    any number of columns.
    """

    @staticmethod
    def log_density(x, model):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        model = np.asarray(model, dtype=np.float64)
        return -(model + 1.0) * np.sum(x**2, axis=-1) - np.abs(x[..., 0])

    def logpdf(self, x, model_index, **kwargs):
        return self.log_density(x, model_index)


class MockACS:
    """Stand-in for ``AnalysisContainerArray``: array module and walker count only."""

    xp = np

    def __len__(self):
        return NWALKERS


@pytest.fixture
def move():
    obj = HyperMove.__new__(HyperMove)
    obj.acs = MockACS()
    obj.nmodels = NMODELS
    obj.snr_threshold = 7.0
    obj.N_tot_array = np.asarray(N_TOT)
    obj.ln_N_tot_array = np.log(obj.N_tot_array)
    obj.model_log_prior = np.log([0.25, 0.75])
    obj.resolved_priors = MockContainer()
    obj.stochastic_priors = MockContainer()
    obj.branch_name_map = dict(resolved="gb", stochastic="galfor")
    obj.num_proposals = 0
    obj.recorder = None
    obj.diagnostics_file = None
    obj.diagnostics_overwrite = False

    # the expected counts need a waveform generator and a PSD; fix them instead
    obj._counts = np.asarray(
        [np.linspace(3000.0, 3100.0, NWALKERS), np.linspace(2000.0, 2100.0, NWALKERS)]
    )
    obj.compute_expected_resolved_counts = lambda: obj._counts
    return obj


@pytest.fixture
def coords():
    rng = np.random.default_rng(42)
    resolved = rng.normal(size=(NTEMPS, NWALKERS, NLEAVES, NDIM_RESOLVED))
    inds = rng.random((NTEMPS, NWALKERS, NLEAVES)) < 0.7
    # make sure at least one walker carries no source at all
    inds[0, 0] = False
    stochastic = rng.normal(size=(NTEMPS, NWALKERS, 1, NDIM_STOCHASTIC))
    return resolved, inds, stochastic


def previous_grouping(move, model_coords, resolved, inds, stochastic):
    """The pre-refactor expression, transcribed from commit ``861dc05``.

    This is ``compute_source_contribution + compute_number_contribution`` as they stood
    immediately before the regrouping, written out here because neither method survives in
    the tree. It is deliberately **independent** of :meth:`HyperMove.compute_model_terms`:
    it calls the priors itself, indexes the catalogue sizes from the module-level
    ``N_TOT`` rather than from the move, and reproduces the old flatten/mask/reshape by
    hand. A test that instead rearranged ``compute_model_terms``'s own return values would
    verify only that addition commutes, which is what the previous version of this test
    did.

    Two details of the old code that matter and are preserved exactly: the resolved sum is
    masked by ``resolved_inds`` before the priors are called and scattered back into a
    zero-filled array, and the catalogue size enters as ``num_resolved * log(N_tot)``.
    Both were introduced in ``861dc05``; the version before that (``a141ac0``) had neither,
    and comparing against *it* would fail by :math:`k_1(N_{\\rm tot} - \\ln N_{\\rm tot})`
    per walker. The baseline for a regression test is the immediate predecessor, not the
    oldest ancestor.
    """
    ntemps, nwalkers, nleaves_max, ndim_res = resolved.shape
    ndim_stoc = stochastic.shape[-1]
    model_coords_2d = np.asarray(model_coords, dtype=int)

    # --- compute_source_contribution ---------------------------------------------
    resolved_reshaped = resolved.reshape(-1, ndim_res)
    model_flat_1 = np.broadcast_to(
        model_coords_2d[:, :, np.newaxis], (ntemps, nwalkers, nleaves_max)
    ).flatten()
    inds_flat = inds.flatten()
    resolved_in = resolved_reshaped[inds_flat, :]

    active_pdfs = move.resolved_priors.logpdf(
        resolved_in, model_index=model_flat_1[inds_flat]
    )
    resolved_pdfs = np.zeros(ntemps * nwalkers * nleaves_max, dtype=np.float64)
    resolved_pdfs[inds_flat] = active_pdfs
    logp_resolved = resolved_pdfs.reshape((ntemps, nwalkers, nleaves_max)).sum(axis=-1)

    logp_stochastic = move.stochastic_priors.logpdf(
        stochastic.reshape(-1, ndim_stoc), model_index=model_coords_2d.flatten()
    ).reshape((ntemps, nwalkers))

    source_contribution = logp_resolved + logp_stochastic

    # --- compute_number_contribution ---------------------------------------------
    counts = move.compute_expected_resolved_counts()
    term_1 = counts[model_coords_2d, np.arange(nwalkers)]
    term_2 = np.asarray(N_TOT)[model_coords_2d]
    num_resolved = inds.sum(axis=-1)
    number_contribution = -term_1 + num_resolved * np.log(term_2)

    return source_contribution + number_contribution


def explicit_resolved(resolved, inds):
    """Brute-force reference for the resolved sum, one source at a time."""
    out = np.zeros((NMODELS, NTEMPS, NWALKERS))
    for temp in range(NTEMPS):
        for walker in range(NWALKERS):
            for leaf in range(NLEAVES):
                if not inds[temp, walker, leaf]:
                    continue
                theta = resolved[temp, walker, leaf]
                for model in range(NMODELS):
                    out[model, temp, walker] += float(
                        MockContainer.log_density(theta[None], model)[0]
                    )
    return out


class TestResolvedContribution:
    def test_matches_source_by_source_sum(self, move, coords):
        resolved, inds, _ = coords
        values = move.compute_resolved_contribution(resolved, inds)

        assert values.shape == (NMODELS, NTEMPS, NWALKERS)
        np.testing.assert_allclose(values, explicit_resolved(resolved, inds))

    def test_inactive_leaves_do_not_contribute(self, move, coords):
        resolved, inds, _ = coords
        values = move.compute_resolved_contribution(resolved, inds)
        # walker (0, 0) has no active leaf
        np.testing.assert_allclose(values[:, 0, 0], 0.0)

        # changing the coordinates of an inactive leaf must change nothing
        modified = resolved.copy()
        inactive = np.nonzero(~inds)
        modified[inactive[0][0], inactive[1][0], inactive[2][0]] += 100.0
        np.testing.assert_allclose(
            values, move.compute_resolved_contribution(modified, inds)
        )

    def test_no_active_leaves_at_all(self, move, coords):
        resolved, inds, _ = coords
        values = move.compute_resolved_contribution(resolved, np.zeros_like(inds))
        np.testing.assert_allclose(values, 0.0)


class TestModelTerms:
    def test_grouping_is_consistent(self, move, coords):
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)

        k1 = inds.sum(axis=-1)
        np.testing.assert_array_equal(terms["num_resolved"], k1)

        # intensity = sum_i ln[N_tot(m) p_pop(theta_i|m)]
        expected_intensity = (
            terms["resolved"] + k1[None] * move.ln_N_tot_array[:, None, None]
        )
        np.testing.assert_allclose(terms["intensity"], expected_intensity)

        expected_ell = (
            terms["intensity"]
            - terms["n1_expected"]
            + terms["stochastic"]
            + move.model_log_prior[:, None, None]
        )
        np.testing.assert_allclose(terms["ell"], expected_ell)

    def test_expected_count_has_no_temperature_dependence(self, move, coords):
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)
        for temp in range(1, NTEMPS):
            np.testing.assert_allclose(
                terms["n1_expected"][:, temp], terms["n1_expected"][:, 0]
            )
        np.testing.assert_allclose(terms["n1_expected"][:, 0], move._counts)

    def test_matches_previous_grouping(self, move, coords):
        """The refactor must not change the acceptance ratio.

        Compares ``ell_m`` against :func:`previous_grouping`, an independent transcription
        of the pre-refactor ``compute_source_contribution + compute_number_contribution``.
        Independence is the whole point: the reference calls the priors itself rather than
        rearranging ``compute_model_terms``'s output, so a wrong model index, a dropped
        term or a sign error in the new code makes this fail.
        """
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)

        for model in range(NMODELS):
            # every walker held at this model, which is what ell[model] describes
            model_coords = np.full((NTEMPS, NWALKERS), model)
            old_logp = previous_grouping(
                move, model_coords, resolved, inds, stochastic
            )
            np.testing.assert_allclose(
                old_logp,
                terms["ell"][model] - move.model_log_prior[model],
                err_msg=f"the regrouping changed ell for model {model}",
            )

    def test_matches_previous_grouping_at_mixed_model_indices(self, move, coords):
        """The same, on the path the move actually takes.

        In production every walker carries its own model index and the old code was
        evaluated at that index directly, whereas the new code evaluates all models and
        gathers. This exercises the gather against the old expression, which the
        all-walkers-at-one-model comparison above cannot.
        """
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)

        rng = np.random.default_rng(2718)
        model_coords = rng.integers(0, NMODELS, size=(NTEMPS, NWALKERS))

        old_logp = previous_grouping(move, model_coords, resolved, inds, stochastic)
        gathered = HyperMove.gather_model(terms["ell"], model_coords)
        new_logp = gathered - move.model_log_prior[model_coords]

        np.testing.assert_allclose(old_logp, new_logp)

    def test_the_reference_is_sensitive_to_a_planted_error(self, move, coords):
        """The reference must fail when the new code is wrong, or it proves nothing.

        A regression test that cannot fail is the defect this whole test was rewritten to
        remove, so the sensitivity is asserted rather than assumed: perturbing one term of
        ``ell`` by one nat must break the comparison.
        """
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)
        model_coords = np.zeros((NTEMPS, NWALKERS), dtype=int)

        old_logp = previous_grouping(move, model_coords, resolved, inds, stochastic)
        tampered = terms["ell"][0] - move.model_log_prior[0] + 1.0

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(old_logp, tampered)

    def test_gather_model(self, move, coords):
        resolved, inds, stochastic = coords
        ell = move.compute_model_terms(resolved, inds, stochastic)["ell"]
        rng = np.random.default_rng(7)
        indices = rng.integers(0, NMODELS, size=(NTEMPS, NWALKERS))
        gathered = HyperMove.gather_model(ell, indices)
        for temp in range(NTEMPS):
            for walker in range(NWALKERS):
                assert gathered[temp, walker] == ell[indices[temp, walker], temp, walker]


class TestAcceptance:
    @staticmethod
    def _indices(value):
        return np.full((NTEMPS, NWALKERS), value, dtype=int)

    def test_nan_is_rejected(self, move):
        delta = np.zeros((NTEMPS, NWALKERS))
        delta[1, 2] = np.nan
        delta[0, 0] = 1.0e3  # certain acceptance
        accepted = move.accept_delta(
            delta, np.random.RandomState(0), self._indices(0), self._indices(1)
        )
        assert not accepted[1, 2]
        assert accepted[0, 0]

    def test_neg_inf_is_rejected_and_pos_inf_accepted(self, move):
        delta = np.zeros((NTEMPS, NWALKERS))
        delta[0, 1] = -np.inf
        delta[0, 2] = np.inf
        accepted = move.accept_delta(
            delta, np.random.RandomState(0), self._indices(0), self._indices(1)
        )
        assert not accepted[0, 1]
        assert accepted[0, 2]

    def test_draws_are_unchanged_by_the_guard(self, move):
        """The guard must not alter the random stream of the move."""
        delta = np.full((NTEMPS, NWALKERS), -0.5)
        accepted = move.accept_delta(
            delta, np.random.RandomState(123), self._indices(0), self._indices(1)
        )
        reference = delta > np.log(np.random.RandomState(123).rand(NTEMPS, NWALKERS))
        np.testing.assert_array_equal(accepted, reference)


class TestRecording:
    def test_round_trip_and_rao_blackwell(self, move, coords, tmp_path):
        resolved, inds, stochastic = coords
        terms = move.compute_model_terms(resolved, inds, stochastic)

        filename = tmp_path / "hyper_move_diagnostics.h5"
        move.diagnostics_file = str(filename)

        current = np.zeros((NTEMPS, NWALKERS), dtype=int)
        proposed = np.ones((NTEMPS, NWALKERS), dtype=int)
        accepted = np.zeros((NTEMPS, NWALKERS), dtype=bool)
        for iteration in range(3):
            move.record_diagnostics(
                terms, current, proposed, accepted, iteration=iteration
            )

        diagnostics = HyperMoveDiagnostics(filename)
        assert diagnostics.niterations == 3
        np.testing.assert_allclose(diagnostics.n_tot, np.asarray(N_TOT))
        recorded = diagnostics.term("ell", temperature=0)
        assert recorded.shape == (3, NMODELS, NWALKERS)
        for row in recorded:
            np.testing.assert_allclose(row, terms["ell"][:, 0])

        # ell_m is deterministic here, so the Rao-Blackwell weights are the softmax
        ell = terms["ell"][:, 0]  # (nmodels, nwalkers)
        shifted = ell - ell.max(axis=0)
        weights = np.exp(shifted) / np.exp(shifted).sum(axis=0)
        np.testing.assert_allclose(
            diagnostics.rao_blackwell(), weights.mean(axis=1), rtol=1e-12
        )

        text = diagnostics.summary_text()
        assert "Rao-Blackwell" in text and "intensity" in text


# ----------------------------------------------------------------------
# the stored likelihood after a swap (_dev/prior_tempering.md, section 7.4)
# ----------------------------------------------------------------------


class MockBranch:
    def __init__(self, coords):
        self.coords = coords
        self.inds = None


class MockState:
    """Just the surface ``_restore_stored_likelihood`` touches."""

    def __init__(self, rng):
        self.branches = {
            "psd": MockBranch(rng.normal(size=(NTEMPS, NWALKERS, 1, 3))),
            "galfor": MockBranch(rng.normal(size=(NTEMPS, NWALKERS, 1, 2))),
        }
        self.log_like = rng.normal(size=(NTEMPS, NWALKERS))


class RecordingEvaluator:
    """Returns a value identifying the coordinates it was handed."""

    def __init__(self):
        self.calls = []

    def evaluate(
        self, resolved_coords, resolved_inds, psd_coords, galfor_coords,
        walker_of_config, subtract_resolved=True,
    ):
        assert not subtract_resolved, "restoring the stored value subtracts no templates"
        self.calls.append(
            dict(psd=psd_coords.copy(), galfor=galfor_coords.copy(),
                 walkers=np.asarray(walker_of_config).copy())
        )
        # a fingerprint of the parameters, so a mis-indexed gather is visible
        return psd_coords.sum(axis=-1) + 100.0 * galfor_coords.sum(axis=-1)


def test_only_the_stale_entries_of_the_stored_likelihood_are_recomputed(move):
    """An untouched entry must keep the number the foreground move produced.

    Not a re-derivation of it: this path reaches the likelihood through the analysis
    container while the foreground move reaches it through the sensitivity backend, and
    the two have historically disagreed at the level ``PSDMove``'s own CHECK2 catches.
    Recomputing everything unconditionally would quietly substitute one for the other on
    every iteration, including iterations where no swap was accepted.
    """
    rng = np.random.default_rng(11)
    state = MockState(rng)
    before = state.log_like.copy()
    move.temperature_likelihood = RecordingEvaluator()

    stale = np.zeros((NTEMPS, NWALKERS), dtype=bool)
    stale[1, 2] = True
    stale[2, 0] = True

    move._restore_stored_likelihood(state, stale)

    # untouched entries are bit-for-bit what they were
    untouched = ~stale
    assert np.array_equal(state.log_like[untouched], before[untouched])

    # and the recomputed ones used their own (temperature, walker) coordinates
    for temp, walker in ((1, 2), (2, 0)):
        expected = (
            state.branches["psd"].coords[temp, walker, 0].sum()
            + 100.0 * state.branches["galfor"].coords[temp, walker, 0].sum()
        )
        assert np.isclose(state.log_like[temp, walker], expected)

    # the walker index decides which walker's data a configuration is scored against
    assert np.array_equal(move.temperature_likelihood.calls[0]["walkers"], [2, 0])


def test_nothing_stale_touches_nothing(move):
    rng = np.random.default_rng(12)
    state = MockState(rng)
    before = state.log_like.copy()
    move.temperature_likelihood = RecordingEvaluator()

    move._restore_stored_likelihood(state, np.zeros((NTEMPS, NWALKERS), dtype=bool))

    assert np.array_equal(state.log_like, before)
    assert move.temperature_likelihood.calls == []


# ----------------------------------------------------------------------
# the swap ratio, equation (15) (_dev/prior_tempering.md, section 5)
# ----------------------------------------------------------------------


def swap_inputs(seed=7, n=6):
    rng = np.random.default_rng(seed)
    return dict(
        logl_new_hot=rng.normal(size=n) * 10.0,
        logl_new_cold=rng.normal(size=n) * 10.0,
        logl_hot=rng.normal(size=n) * 10.0,
        logl_cold=rng.normal(size=n) * 10.0,
        u_hot=rng.normal(size=n) * 100.0,
        u_cold=rng.normal(size=n) * 100.0,
    )


def test_the_swap_ratio_collapses_to_the_standard_form_when_nothing_is_left_behind():
    """The sharpest check available on the algebra, and it needs no GPU.

    Equation (15) is unusual only because the instrument-noise branch stays put, so the
    two states are not a permutation and the likelihood does not travel. Hand it the
    likelihoods it *would* see had that branch travelled too -- ``L_hot^new = L_cold``
    and conversely -- and it must reduce to the familiar
    ``dbeta * [A(x_hot) - A(x_cold)]`` with ``A = ln L + ell_M - R``. A sign error or a
    flipped ``dbeta`` breaks this identity and nothing else in the expression does.
    """
    rng = np.random.default_rng(3)
    n = 6
    logl_hot, logl_cold = rng.normal(size=n) * 10.0, rng.normal(size=n) * 10.0
    u_hot, u_cold = rng.normal(size=n) * 100.0, rng.normal(size=n) * 100.0
    beta_hot, beta_cold = 0.1, 0.5

    got = HyperMove.swap_log_ratio_from_terms(
        beta_hot, beta_cold,
        logl_new_hot=logl_cold,   # the exchange now carries the likelihood with it
        logl_new_cold=logl_hot,
        logl_hot=logl_hot, logl_cold=logl_cold,
        u_hot=u_hot, u_cold=u_cold,
    )

    a_hot = logl_hot + u_hot
    a_cold = logl_cold + u_cold
    expected = (beta_cold - beta_hot) * (a_hot - a_cold)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_equal_temperatures_leave_only_the_likelihood_terms():
    """At ``dbeta = 0`` the model term cannot contribute, whatever the barrier."""
    inputs = swap_inputs()
    beta = 0.3
    got = HyperMove.swap_log_ratio_from_terms(beta, beta, **inputs)
    expected = beta * (
        inputs["logl_new_hot"] - inputs["logl_hot"]
        + inputs["logl_new_cold"] - inputs["logl_cold"]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_the_model_term_is_oriented_so_the_cold_chain_gains():
    """``dbeta = beta_cold - beta_hot`` is positive, so a hotter state with the larger
    ``u`` is what the ratio rewards moving down. Getting this backwards would drive the
    model index the wrong way while still producing a plausible acceptance rate."""
    zero = dict(
        logl_new_hot=np.zeros(1), logl_new_cold=np.zeros(1),
        logl_hot=np.zeros(1), logl_cold=np.zeros(1),
    )
    favourable = HyperMove.swap_log_ratio_from_terms(
        0.1, 0.5, u_hot=np.array([5.0]), u_cold=np.array([0.0]), **zero
    )
    unfavourable = HyperMove.swap_log_ratio_from_terms(
        0.1, 0.5, u_hot=np.array([0.0]), u_cold=np.array([5.0]), **zero
    )
    assert favourable[0] > 0.0 > unfavourable[0]
    assert np.isclose(favourable[0], -unfavourable[0])


def test_the_used_ratio_is_the_pure_one(move):
    """``hyper_swap_log_ratio`` must not re-derive the arithmetic it delegates."""
    import inspect

    source = inspect.getsource(HyperMove.hyper_swap_log_ratio)
    assert "swap_log_ratio_from_terms" in source
