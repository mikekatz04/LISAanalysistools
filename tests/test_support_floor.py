"""Tests for :class:`SupportFloor`, R5 method 2 of the population-inference note.

The floor replaces a flow density by the mixture

.. math::
    p_\\epsilon = (1-\\epsilon)\\, p_{\\rm flow} + \\epsilon\\, \\mathcal U(\\cdot \\mid B),

which is a change of the *target* distribution, not of any sampler. That is only
harmless if three things hold, and each has a test here:

1. **The mixture is a proper density.** Both components integrate to one over the broad
   box :math:`B`, and the flow's own box lies inside :math:`B`, so
   :math:`\\int_B p_\\epsilon = 1`. If this failed, every acceptance ratio in the run
   would carry an unnormalised factor that does not cancel.
2. **``rvs`` samples the density ``logpdf`` reports.** These priors are also the
   reversible-jump birth proposal, where the proposal density enters the acceptance
   ratio explicitly; reporting one density and sampling another breaks detailed balance
   for the birth move.
3. **Off is exactly off.** With ``support_floor=None`` the returned log-density must be
   bit-identical to the unfloored flow, so that turning the floor on is the only thing
   that ever changes an existing result.

The mathematics is exercised against a stub flow with an analytically known density, so
these tests need neither a GPU nor a checkpoint. One integration test at the end runs
the real classes and skips without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

network = pytest.importorskip("lisatools.globalfit.priors.network")

SupportFloor = network.SupportFloor
apply_common_support_floor = network.apply_common_support_floor


NDIM = 3


class StubFlow(SupportFloor):
    """A uniform density over its own box, wearing the interface of a flow prior.

    Uniform is the one case where every quantity the floor computes has a closed form,
    so the tests check numbers rather than merely self-consistency.
    """

    def __init__(self, box_min, box_max):
        self.param_min = np.asarray(box_min, dtype=np.float64)
        self.param_max = np.asarray(box_max, dtype=np.float64)
        self.log_density = -float(np.sum(np.log(self.param_max - self.param_min)))

    @property
    def xp(self):
        """A property, as in ``JointPrior``, where it is derived from ``use_cupy``."""
        return np

    def raw_logpdf(self, x):
        """``ln p_flow``: constant inside the box, ``-inf`` outside, as a flow is."""
        x = np.asarray(x, dtype=np.float64)
        inside = np.all((x >= self.param_min) & (x <= self.param_max), axis=-1)
        return np.where(inside, self.log_density, -np.inf)

    def logpdf(self, x):
        return self._apply_support_floor(self.raw_logpdf(x), np.asarray(x))

    def rvs(self, size):
        draws = self.param_min + np.random.random((size, NDIM)) * (
            self.param_max - self.param_min
        )
        return self._draw_support_floor(draws)


class TaperedStubFlow(StubFlow):
    """A stub whose density *decays*, so that it has a tail a uniform stub cannot have.

    ``p(x) proportional to exp(-lam * sum(x))`` on its own box, normalised in closed form.
    This exists because the uniform :class:`StubFlow` cannot exhibit the one regime where
    the floor is not negligible: a uniform density over a sub-box of ``B`` is never below
    ``1/|B|``, so ``eps/|B|`` is always a small correction to it whatever ``eps`` is. A
    real flow is not uniform, and the model comparison lives in its tail -- the resolved
    sources sit at training quantile 0.9988 -- which is exactly where ``p_flow`` falls
    below ``eps/|B|`` and the floor takes over.
    """

    def __init__(self, box_min, box_max, lam=20.0):
        super().__init__(box_min, box_max)
        self.lam = float(lam)
        # integral of exp(-lam x) over [lo, hi] per coordinate
        widths = np.exp(-self.lam * self.param_min) - np.exp(-self.lam * self.param_max)
        self.log_norm = float(np.sum(np.log(widths / self.lam)))

    def raw_logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        inside = np.all((x >= self.param_min) & (x <= self.param_max), axis=-1)
        return np.where(inside, -self.lam * np.sum(x, axis=-1) - self.log_norm, -np.inf)


def test_the_tapered_stub_is_normalised():
    """The fixture has to be right before anything is concluded from it."""
    flow = TaperedStubFlow([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    rng = np.random.default_rng(11)
    points = rng.random((200_000, NDIM))
    assert float(np.mean(np.exp(flow.raw_logpdf(points)))) == pytest.approx(1.0, rel=0.05)


def test_the_floor_is_not_negligible_in_the_tail():
    """The claim "no perturbation where it matters", tested where it actually matters.

    ``test_negligible_where_the_flow_is_supported`` checks a point at the centre of the
    box, where the flow's density is large and the floor is irrelevant. That is the easy
    case. In the tail the flow's density falls below ``eps/|B|`` and the floor dominates
    completely -- the returned value stops being a statement about the population and
    becomes a statement about ``eps``. This test documents that boundary rather than
    leaving it to be discovered in a Bayes factor.
    """
    epsilon = 1e-6
    flow = TaperedStubFlow([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    flow.set_support_floor(epsilon, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    broad_volume = 1.0
    floor_value = epsilon / broad_volume

    mode = np.array([[0.0, 0.0, 0.0]])
    assert flow.logpdf(mode)[0] == pytest.approx(flow.raw_logpdf(mode)[0], abs=1e-5)

    # deep in the tail: exp(-20 * 3) against a floor of 1e-6
    tail = np.array([[1.0, 1.0, 1.0]])
    raw = float(flow.raw_logpdf(tail)[0])
    floored = float(flow.logpdf(tail)[0])
    assert np.exp(raw) < floor_value, "the fixture must reach below the floor"
    assert floored == pytest.approx(np.log((1 - epsilon) * np.exp(raw) + floor_value))
    assert floored - raw > 10.0, (
        "in the tail the floor is worth many nats per source, not a rounding difference"
    )

    # the crossover, stated as a number: where p_flow == eps/|B| the floor adds ln 2
    crossover = float(np.log((1 - epsilon) * floor_value + floor_value) - np.log(floor_value))
    assert crossover == pytest.approx(np.log(2.0), abs=1e-5)


@pytest.fixture
def boxes():
    """Two boxes that overlap only partially, as B9's foreground fits do."""
    return (
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([0.4, 0.4, 0.4]), np.array([0.6, 2.0, 0.6])),
    )


@pytest.fixture
def flows(boxes):
    return [StubFlow(*box) for box in boxes]


def test_off_by_default(flows):
    """A flow with no floor set must be untouched."""
    flow = flows[0]
    assert flow.support_floor_epsilon is None
    x = np.array([[0.5, 0.5, 0.5], [-1.0, 0.5, 0.5]])
    np.testing.assert_array_equal(flow.logpdf(x), flow.raw_logpdf(x))
    assert np.isneginf(flow.logpdf(x)[1])


def test_union_box_covers_every_model(flows):
    apply_common_support_floor(flows, 1e-3)
    for flow in flows:
        assert np.all(flow._floor_min_host <= flow.param_min)
        assert np.all(flow._floor_max_host >= flow.param_max)
    # the union of the two fixtures, computed by hand
    np.testing.assert_allclose(flows[0]._floor_min_host, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(flows[0]._floor_max_host, [1.0, 2.0, 1.0])


def test_broad_box_must_cover_the_flow(flows):
    """A broad box that does not cover the flow would make the mixture improper."""
    with pytest.raises(ValueError, match="must cover"):
        flows[0].set_support_floor(1e-3, [0.5, 0.5, 0.5], [0.6, 0.6, 0.6])


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -1e-3, 2.0])
def test_epsilon_must_be_a_weight(flows, epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        flows[0].set_support_floor(epsilon, [0.0, 0.0, 0.0], [1.0, 2.0, 1.0])


def test_mixture_integrates_to_one(flows):
    """Property 1: the floored density is normalised over the broad box.

    Uniform sampling over the broad box makes the Monte Carlo estimate of
    :math:`\\int_B p_\\epsilon` a plain mean times the box volume.

    The 2% tolerance is what this estimator can deliver, not a slack choice: the
    integrand is two-valued and the high value is confined to 3.2% of the broad box, so
    at 400 000 points the relative scatter is 0.73% and the worst of forty seeds is 2.07%.

    Note what that implies, and see
    :func:`test_the_flow_component_carries_its_one_minus_epsilon_weight` for the remedy: a
    dropped :math:`(1-\\epsilon)` factor shifts this estimate by exactly
    :math:`\\epsilon = 1\\%`, which is 1.4 standard errors and therefore invisible here.
    This test shows the mixture is not grossly wrong; it cannot police the weights.
    """
    epsilon = 1e-2
    apply_common_support_floor(flows, epsilon)
    flow = flows[1]

    rng = np.random.default_rng(20260813)
    lower, upper = flow._floor_min_host, flow._floor_max_host
    volume = float(np.prod(upper - lower))
    points = lower + rng.random((400_000, NDIM)) * (upper - lower)

    estimate = float(np.mean(np.exp(flow.logpdf(points)))) * volume
    assert estimate == pytest.approx(1.0, rel=2e-2)


def test_the_flow_component_carries_its_one_minus_epsilon_weight(flows):
    """The :math:`(1-\\epsilon)` the Monte Carlo test cannot see, pinned exactly.

    Inside the flow's own box the mixture is
    :math:`(1-\\epsilon)p_{\\rm flow} + \\epsilon/|B|`, and for a uniform stub every term
    is known in closed form. Asserting it to 1e-12 catches a dropped or misplaced weight
    outright, which is the defect that makes the mixture improper and which a 2%-tolerance
    integral is 1.4 sigma away from noticing.
    """
    epsilon = 1e-2
    apply_common_support_floor(flows, epsilon)
    flow = flows[1]

    own_volume = float(np.prod(flow.param_max - flow.param_min))
    broad_volume = float(np.prod(flow._floor_max_host - flow._floor_min_host))
    inside = np.array([[0.5, 1.0, 0.5]])
    assert np.isfinite(flow.raw_logpdf(inside)[0])

    expected = np.log((1.0 - epsilon) / own_volume + epsilon / broad_volume)
    assert flow.logpdf(inside)[0] == pytest.approx(expected, rel=1e-12)

    # and the same statement made where it bites hardest: without the (1 - epsilon) the
    # value would be larger by exactly that factor, which this resolves and the integral
    # does not
    without_weight = np.log(1.0 / own_volume + epsilon / broad_volume)
    assert abs(expected - without_weight) > 1e-3


def test_floor_value_where_the_flow_has_no_support(flows):
    """The exact number the note quotes: ``ln(epsilon) - ln|B|`` outside the flow."""
    epsilon = 1e-6
    apply_common_support_floor(flows, epsilon)
    flow = flows[1]

    # inside the broad box, outside this flow's own box
    outside_flow = np.array([[0.05, 0.05, 0.05]])
    assert np.isneginf(flow.raw_logpdf(outside_flow)[0])

    volume = float(np.prod(flow._floor_max_host - flow._floor_min_host))
    expected = np.log(epsilon) - np.log(volume)
    assert flow.logpdf(outside_flow)[0] == pytest.approx(expected, rel=1e-12)


def test_outside_the_broad_box_is_still_minus_infinity(flows):
    """The floor widens the support; it does not abolish the notion of support."""
    apply_common_support_floor(flows, 1e-3)
    far_away = np.array([[-5.0, -5.0, -5.0]])
    for flow in flows:
        assert np.isneginf(flow.logpdf(far_away)[0])


def test_negligible_where_the_flow_is_supported(flows):
    """The floor must not perturb a comparison that was already meaningful."""
    epsilon = 1e-6
    raw = flows[0].raw_logpdf(np.array([[0.5, 0.5, 0.5]]))
    apply_common_support_floor(flows, epsilon)
    floored = flows[0].logpdf(np.array([[0.5, 0.5, 0.5]]))
    assert floored[0] == pytest.approx(raw[0], abs=1e-5)


def test_rvs_matches_logpdf(flows):
    """Property 2: the density reported is the density sampled.

    A large ``epsilon`` makes the broad component visible in a finite sample. The check
    is that the fraction of draws landing outside the flow's own box matches the
    fraction of the mixture mass the broad component puts there, which is what the
    reversible-jump acceptance ratio assumes.
    """
    epsilon = 0.25
    apply_common_support_floor(flows, epsilon)
    flow = flows[1]

    np.random.seed(884422)
    draws = flow.rvs(200_000)

    inside_own = np.all(
        (draws >= flow.param_min) & (draws <= flow.param_max), axis=-1
    )
    own_volume = float(np.prod(flow.param_max - flow.param_min))
    broad_volume = float(np.prod(flow._floor_max_host - flow._floor_min_host))
    # mass inside the own box = flow component + the broad component's share of it
    expected_inside = (1.0 - epsilon) + epsilon * own_volume / broad_volume
    assert np.mean(inside_own) == pytest.approx(expected_inside, abs=5e-3)

    # and every draw is inside the broad box, i.e. has finite reported density
    assert np.all(np.isfinite(flow.logpdf(draws)))


def test_ratio_is_finite_where_it_used_to_be_infinite(flows):
    """The point of the exercise, stated as a test.

    At a state only model 0 can reach, the unfloored log-density ratio is ``+inf``; with
    the floor it is a finite number that a sampler can act on and a diagnostic report.
    """
    x = np.array([[0.05, 0.05, 0.05]])

    unfloored = flows[0].raw_logpdf(x) - flows[1].raw_logpdf(x)
    assert np.isposinf(unfloored[0])

    apply_common_support_floor(flows, 1e-6)
    floored = flows[0].logpdf(x) - flows[1].logpdf(x)
    assert np.isfinite(floored[0])
    # capped by the floor: ln p_0 - (ln epsilon - ln|B|)
    volume = float(np.prod(flows[0]._floor_max_host - flows[0]._floor_min_host))
    assert floored[0] == pytest.approx(
        flows[0].logpdf(x)[0] - (np.log(1e-6) - np.log(volume)), rel=1e-12
    )


def test_survives_a_deepcopy(flows):
    """``build_gb_moves`` deep-copies these priors, then flips ``use_cupy``.

    The floor must therefore be part of the object state, not something applied to one
    copy afterwards: two copies evaluating different densities would make the GB moves
    and the model move target different distributions.
    """
    from copy import deepcopy

    apply_common_support_floor(flows, 1e-4)
    x = np.array([[0.05, 0.05, 0.05]])
    copied = deepcopy(flows[1])
    assert copied.support_floor_epsilon == flows[1].support_floor_epsilon
    assert copied.logpdf(x)[0] == pytest.approx(flows[1].logpdf(x)[0], rel=1e-12)


# ------------------------------------------------------------------ D14: reversibility

# The floor is a change of the target, so a run made with it reports a Bayes factor for
# the floored model. These tests pin the arithmetic that lets that be undone, and --
# more importantly -- pin where it stops working, because that boundary is the finding:
# the floor cannot be corrected away in the regime it was introduced for.


def test_unfloor_inverts_the_floor_exactly():
    """``unfloor(refloor(x)) == x`` wherever the flow is above the floor at all."""
    epsilon, log_uniform = 1e-6, -4.07
    raw = np.array([15.0, 0.0, -10.0, -17.0, -17.9, -25.0])
    floored = network.refloor_log_density(raw, epsilon, log_uniform)
    np.testing.assert_allclose(
        network.unfloor_log_density(floored, epsilon, log_uniform), raw, rtol=1e-9
    )


def test_unfloor_returns_minus_infinity_where_the_floor_carried_the_state():
    """The boundary, which is the point: what the floor invented cannot be un-invented.

    A state the flow gives zero density to is reported at exactly the floor level. There
    is no information in that number about the population, and the inverse says so by
    returning ``-inf`` rather than a plausible-looking finite value. This is the reason
    §10.6 concludes that the floored Bayes factor cannot be "corrected" back into a
    meaningful unfloored one: the correction is exact, and it returns the degeneracy.
    """
    epsilon, log_uniform = 1e-6, -4.07
    at_the_floor = np.log(epsilon) + log_uniform
    assert np.isneginf(
        network.unfloor_log_density(np.array([at_the_floor]), epsilon, log_uniform)[0]
    )
    # and float64 noise a hair below the floor must not become a NaN either
    assert np.isneginf(
        network.unfloor_log_density(
            np.array([at_the_floor - 1e-13]), epsilon, log_uniform
        )[0]
    )


def test_unfloor_agrees_with_the_prior_that_produced_the_value(flows):
    """The inverse is pinned against a real ``SupportFloor``, not only against itself."""
    epsilon = 1e-4
    x = np.array([[0.5, 1.0, 0.5], [0.05, 0.05, 0.05]])
    raw = flows[1].raw_logpdf(x)

    apply_common_support_floor(flows, epsilon)
    recovered = flows[1].unfloor_logpdf(flows[1].logpdf(x))

    # the supported point comes back exactly; the unsupported one was -inf and stays -inf
    assert recovered[0] == pytest.approx(raw[0], rel=1e-9)
    assert np.isneginf(raw[1]) and np.isneginf(recovered[1])


def test_the_floor_reports_its_own_parameters(flows):
    """A recording can only be un-floored if the run wrote down what it used."""
    assert flows[0].support_floor_log_uniform is None
    assert flows[0].support_floor_level is None

    apply_common_support_floor(flows, 1e-6)
    volume = float(np.prod(flows[0]._floor_max_host - flows[0]._floor_min_host))
    assert flows[0].support_floor_log_uniform == pytest.approx(-np.log(volume))
    assert flows[0].support_floor_level == pytest.approx(np.log(1e-6) - np.log(volume))
    # every model of a family shares them, which is what lets one pair of numbers into
    # the recording describe the whole family
    assert flows[1].support_floor_log_uniform == flows[0].support_floor_log_uniform


def test_the_comparison_moves_one_for_one_with_ln_epsilon():
    """The D14 headline, on a fixture where the answer is known by hand.

    At a state model 1 cannot reach, the floored log-density ratio is
    ``ln p_0 - (ln eps - ln|B|)``, so it moves by exactly one nat per e-fold of eps.
    That slope of one is what "the Bayes factor is a restatement of epsilon" means, and
    it is what ``epsilon_scan`` and D14 check 3 measure on the real densities.
    """
    boxes = ((np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
             (np.array([0.4, 0.4, 0.4]), np.array([0.6, 0.6, 0.6])))
    x = np.array([[0.05, 0.05, 0.05]])  # inside model 0's box, outside model 1's

    deltas = []
    epsilons = [1e-4, 1e-5, 1e-6, 1e-7]
    for epsilon in epsilons:
        pair = [StubFlow(*box) for box in boxes]
        apply_common_support_floor(pair, epsilon)
        deltas.append(float(pair[0].logpdf(x)[0] - pair[1].logpdf(x)[0]))

    slope = np.polyfit(np.log(epsilons), deltas, 1)[0]
    assert slope == pytest.approx(-1.0, abs=1e-3), (
        "model 1 is floored and model 0 is not, so the comparison must move by one nat "
        "per e-fold of epsilon"
    )


def test_the_slope_is_zero_where_both_models_are_supported():
    """The other half of the verdict: a floor that changes nothing must measure as zero.

    Both stubs cover the point, so both pick up the same ``(1-eps)`` and the difference
    is independent of eps. A diagnostic that could not tell this case from the one above
    would be useless, since it is the case that licenses quoting the Bayes factor.
    """
    boxes = ((np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
             (np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])))
    x = np.array([[0.5, 0.5, 0.5]])

    epsilons = [1e-4, 1e-5, 1e-6, 1e-7]
    deltas = []
    for epsilon in epsilons:
        pair = [StubFlow(*box) for box in boxes]
        apply_common_support_floor(pair, epsilon)
        deltas.append(float(pair[0].logpdf(x)[0] - pair[1].logpdf(x)[0]))

    slope = np.polyfit(np.log(epsilons), deltas, 1)[0]
    assert slope == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -1e-3])
def test_the_mixture_helpers_refuse_a_weight_that_is_not_one(epsilon):
    """Same guard as ``set_support_floor``: these are public and take epsilon directly."""
    values = np.array([-1.0])
    with pytest.raises(ValueError, match="epsilon"):
        network.unfloor_log_density(values, epsilon, -4.0)
    with pytest.raises(ValueError, match="epsilon"):
        network.refloor_log_density(values, epsilon, -4.0)


# ------------------------------------------------------------------ integration


@pytest.mark.slow
def test_real_galfor_priors_become_finite():
    """The real B9 configuration: with the floor, ln p(Sigma | M_1) stops being -inf."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("requires a CUDA device")

    import h5py
    import yaml
    from pathlib import Path

    data_root = Path("/sps/lisaf/crondeel/pop_inf/data")
    configs = [
        str(data_root / name / "prior_density_galfor.yaml")
        for name in ("medium_int", "medium_weak_int")
    ]
    for config in configs:
        if not Path(config).is_file():
            pytest.skip(f"{config} is not available")

    with open(configs[0]) as stream:
        chain_path = Path(str(yaml.safe_load(stream)["samples"]["path"]).strip())
    if not chain_path.is_file():
        pytest.skip(f"{chain_path} is not available")
    with h5py.File(str(chain_path), "r", locking=False) as handle:
        total = int(handle["mcmc"].attrs["iteration"])
        sigma = np.asarray(handle["mcmc"]["chain"]["galfor"][total - 2, 0]).reshape(
            -1, 5
        )

    bare = network.HyperGalForPrior(configs, use_cupy=False, return_gpu=False)
    floored = network.HyperGalForPrior(
        configs, use_cupy=False, return_gpu=False, support_floor=1e-6
    )

    ones = np.ones(sigma.shape[0], dtype=int)
    assert np.all(np.isneginf(bare.logpdf(sigma, model_index=ones))), (
        "B9 is expected to hold for the unfloored fits; if this fails the fits have "
        "changed and the note needs updating"
    )
    assert np.all(np.isfinite(floored.logpdf(sigma, model_index=ones)))
    # model 0 must be essentially unchanged: the floor acts only in the dead tail
    zeros = np.zeros(sigma.shape[0], dtype=int)
    np.testing.assert_allclose(
        floored.logpdf(sigma, model_index=zeros),
        bare.logpdf(sigma, model_index=zeros),
        atol=1e-5,
    )
