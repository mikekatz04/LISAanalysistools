"""Contract of the population-prior tempering (``_dev/prior_tempering.md``).

These tests pin the arithmetic of
:class:`~lisatools.globalfit.priors.tempering.PopulationTempering`: that switching the
feature off changes nothing, that the coldest temperature is exactly the untempered
target, that each scheme obeys its defining identity, and that the behaviour where a
model has no support is the one section 4.5 of the note specifies rather than whatever
the floating-point arithmetic happens to produce.

They need no GPU, no flow and no data.
"""

from copy import deepcopy

import numpy as np
import pytest

from lisatools.globalfit.priors.tempering import (
    PopulationTempering,
    resolve_scheme,
)


def ladder(ntemps: int = 12) -> np.ndarray:
    """The ladder the settings file uses for the model branch."""
    betas = 1.0 / 1.2 ** np.arange(ntemps)
    betas[-1] = 1e-4
    return betas


# ----------------------------------------------------------------------
# the settings flag
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, scheme",
    [
        (False, "off"),
        (None, "off"),
        (True, "geometric"),
        ("off", "off"),
        ("flat", "flat"),
        ("geometric", "geometric"),
        ("GEOMETRIC", "geometric"),
    ],
)
def test_the_flag_resolves_to_a_scheme(value, scheme):
    assert resolve_scheme(value) == scheme


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError):
        resolve_scheme("annealed")


def test_a_scheme_without_a_ladder_is_refused():
    """A ladder is not optional: without one there is no temperature to weight by."""
    with pytest.raises(ValueError):
        PopulationTempering("geometric")


def test_the_coldest_temperature_must_be_one():
    """Otherwise the cold chain does not sample the posterior."""
    with pytest.raises(ValueError):
        PopulationTempering("flat", np.array([0.9, 0.5, 0.1]))


# ----------------------------------------------------------------------
# off is exactly off
# ----------------------------------------------------------------------


def test_off_is_the_identical_array():
    """The regression guard that makes the flag safe to add.

    Not merely equal: the same object, so no arithmetic whatever is performed on the
    untempered path.
    """
    tempering = PopulationTempering("off")
    values = np.random.default_rng(0).normal(size=(3, 5)) * 10.0
    assert tempering.combine(values, 0.3) is values
    assert not tempering.enabled
    assert not tempering.needs_all_models


def test_only_the_geometric_scheme_needs_every_model():
    """The other schemes keep the single-model fast path, which for a flow is n passes."""
    assert not PopulationTempering("off").needs_all_models
    assert not PopulationTempering("flat", ladder()).needs_all_models
    assert PopulationTempering("geometric", ladder()).needs_all_models


# ----------------------------------------------------------------------
# beta = 1 is the untempered target, exactly
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["flat", "geometric"])
def test_the_coldest_temperature_is_exactly_untempered(scheme):
    """Exact, not merely close.

    ``R + 1*(ln p - R)`` equals ``ln p`` only up to rounding, and the coldest temperature
    is the one results are read from, so the implementation short-circuits it. Without
    that, enabling the scheme would perturb the cold chain's arithmetic at the level of
    machine precision for no reason.
    """
    rng = np.random.default_rng(1)
    tempering = PopulationTempering(scheme, ladder())
    log_p_all = rng.normal(size=(2, 7)) * 5.0
    index = rng.integers(0, 2, size=7)
    selected = np.take_along_axis(log_p_all, index[None, :], axis=0)[0]

    got = tempering.combine(selected, 1.0, log_p_all=log_p_all)
    assert np.array_equal(got, selected)


# ----------------------------------------------------------------------
# each scheme's defining identity
# ----------------------------------------------------------------------


def test_flat_is_the_power_prior():
    tempering = PopulationTempering("flat", ladder())
    selected = np.random.default_rng(2).normal(size=20) * 3.0
    for beta in (0.9, 0.37, 0.0):
        assert np.allclose(tempering.combine(selected, beta), beta * selected)


@pytest.mark.parametrize("beta", [1.0, 0.5, 0.1, 0.0])
def test_geometric_anneals_the_difference_and_leaves_the_sum(beta):
    """Equations (10) and (11) of ``_dev/prior_tempering.md``, for two models.

    The first is the point of the scheme -- the model discrimination scales with the
    temperature. The second is what distinguishes it from the power prior: the part of
    the log-density the two populations agree on, which is what actually constrains the
    parameters, is untouched at every temperature.
    """
    tempering = PopulationTempering("geometric", ladder())
    log_p_all = np.random.default_rng(3).normal(size=(2, 500)) * 4.0

    tempered = np.stack(
        [tempering.combine(log_p_all[m], beta, log_p_all=log_p_all) for m in range(2)]
    )

    assert np.allclose(tempered[0] - tempered[1], beta * (log_p_all[0] - log_p_all[1]))
    assert np.allclose(tempered[0] + tempered[1], log_p_all[0] + log_p_all[1])


@pytest.mark.parametrize("scheme", ["flat", "geometric"])
def test_the_acceptance_ratio_is_beta_times_the_difference(scheme):
    """Equation (14): the model move's ratio at temperature t is beta_t * (ell' - ell).

    Under both schemes the model-independent reference cancels from the difference, which
    is why the schemes share one acceptance ratio and differ only in what the parameters
    see.
    """
    betas = ladder()
    tempering = PopulationTempering(scheme, betas)
    ell = np.random.default_rng(4).normal(size=(2, betas.size, 20)) * 100.0

    tempered = tempering.temper_ell(ell, betas[:, None])

    assert np.allclose(tempered[1] - tempered[0], betas[:, None] * (ell[1] - ell[0]))


# ----------------------------------------------------------------------
# where a model has no support
# ----------------------------------------------------------------------


def test_behaviour_without_common_support_is_the_specified_one():
    """Section 4.5, case by case.

    The third case is the one that matters for the design: where only one model has
    support the reference collapses onto that model, so the annealing switches itself off
    exactly at the points that make a barrier infinite. It is correct, and it is inert.
    """
    tempering = PopulationTempering("geometric", ladder())
    log_p_all = np.array(
        [
            [1.0, 2.0, -np.inf, -np.inf],
            [3.0, -np.inf, -np.inf, 5.0],
        ]
    )
    beta = 0.5
    first = tempering.combine(log_p_all[0], beta, log_p_all=log_p_all)
    second = tempering.combine(log_p_all[1], beta, log_p_all=log_p_all)

    # both models finite: annealed about their mean
    assert np.isclose(first[0], 2.0 + beta * (1.0 - 2.0))
    # only this model finite: the reference is itself, so nothing is annealed
    assert np.isclose(first[1], 2.0)
    # this model absent while another is present: impossible at any beta > 0
    assert np.isneginf(second[1])
    # no model reaches this point
    assert np.isneginf(first[2]) and np.isneginf(second[2])


def test_at_zero_temperature_an_absent_model_gets_the_shared_reference():
    """The prior is the reference there, which every model shares by construction."""
    tempering = PopulationTempering("geometric", ladder())
    log_p_all = np.array([[1.0, 2.0], [3.0, -np.inf]])
    assert np.isclose(tempering.combine(log_p_all[1], 0.0, log_p_all=log_p_all)[1], 2.0)


def test_the_partial_reference_census_is_recorded():
    """The only measurement a live run can make of how weak the annealing really was."""
    tempering = PopulationTempering("geometric", ladder())
    assert tempering.partial_reference_fraction == 0.0

    log_p_all = np.array(
        [
            [1.0, 2.0, -np.inf, -np.inf],
            [3.0, -np.inf, -np.inf, 5.0],
        ]
    )
    for m in range(2):
        tempering.combine(log_p_all[m], 0.5, log_p_all=log_p_all)

    # three of the four points have fewer than two models finite
    assert np.isclose(tempering.partial_reference_fraction, 0.75)


def test_flat_needs_no_common_support():
    """beta * (-inf) stays a clean rejection, so the guard applies to geometric alone."""
    tempering = PopulationTempering("flat", ladder())
    values = np.array([1.0, -np.inf])
    got = tempering.combine(values, 0.5)
    assert np.isclose(got[0], 0.5) and np.isneginf(got[1])


# ----------------------------------------------------------------------
# the guard that decides whether the geometric scheme is worth running
# ----------------------------------------------------------------------


class FakeModel:
    """A density with a training box and, optionally, a support floor over a broad box."""

    def __init__(self, low, high, epsilon=None, box=None):
        self.param_min = np.asarray(low, dtype=float)
        self.param_max = np.asarray(high, dtype=float)
        self.support_floor_epsilon = epsilon
        self.support_floor_box = (
            None
            if box is None
            else (np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float))
        )


def geometric():
    return PopulationTempering("geometric", ladder())


def test_identical_training_boxes_are_shared():
    models = [FakeModel([0.0, 0.0], [1.0, 1.0]) for _ in range(2)]
    assert geometric().check_common_support(models) == "shared"


def test_a_common_floor_over_one_broad_box_is_shared():
    """What ``apply_common_support_floor`` produces: one epsilon, one union box."""
    box = ([-1.0, -1.0], [2.0, 2.0])
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0], epsilon=1e-6, box=box),
        FakeModel([0.2, 0.1], [1.4, 0.9], epsilon=1e-6, box=box),
    ]
    assert geometric().check_common_support(models) == "shared"


def test_the_same_epsilon_over_different_boxes_is_refused():
    """The bug this pins: an equal epsilon says nothing about where the floor acts.

    ``set_support_floor`` takes a box per model, so two models can carry the same
    mixture weight over different regions -- in which case each density is positive
    somewhere the other is not, and the geometric reference averages densities defined
    on different supports.
    """
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0], epsilon=1e-6, box=([-1.0, -1.0], [2.0, 2.0])),
        FakeModel([0.0, 0.0], [1.0, 1.0], epsilon=1e-6, box=([-5.0, -5.0], [9.0, 9.0])),
    ]
    with pytest.raises(ValueError, match="different broad boxes"):
        geometric().check_common_support(models)


def test_different_epsilons_are_refused():
    box = ([-1.0, -1.0], [2.0, 2.0])
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0], epsilon=1e-6, box=box),
        FakeModel([0.0, 0.0], [1.0, 1.0], epsilon=1e-4, box=box),
    ]
    with pytest.raises(ValueError, match="different support floors"):
        geometric().check_common_support(models)


def test_overlapping_but_unequal_boxes_are_graded_partial():
    """Where the population priors of the current configuration sit: warned, not refused.

    The annealing is weaker over the unshared fraction, which is what
    ``partial_reference_fraction`` measures during the run.
    """
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0]),
        FakeModel([0.01, 0.0], [1.0, 0.99]),
    ]
    assert geometric().check_common_support(models) == "partial"


def test_disjoint_boxes_are_refused_unless_overridden():
    """Refusing matters because the scheme is inert exactly where the models differ."""
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0]),
        FakeModel([0.0, 3.0], [1.0, 4.0]),
    ]
    with pytest.raises(ValueError, match="do not overlap"):
        geometric().check_common_support(models)

    permissive = PopulationTempering(
        "geometric", ladder(), allow_partial_support=True
    )
    assert permissive.check_common_support(models) == "disjoint"


@pytest.mark.parametrize("scheme", ["off", "flat"])
def test_only_the_geometric_scheme_is_guarded(scheme):
    """``beta * (-inf)`` stays a clean rejection, so the other schemes need no support."""
    betas = None if scheme == "off" else ladder()
    tempering = PopulationTempering(scheme, betas)
    models = [
        FakeModel([0.0, 0.0], [1.0, 1.0]),
        FakeModel([0.0, 3.0], [1.0, 4.0]),
    ]
    assert tempering.check_common_support(models) == "shared"


# ----------------------------------------------------------------------
# one ladder
# ----------------------------------------------------------------------


def test_the_ladder_survives_a_deep_copy_as_one_object():
    """``build_gb_moves`` deep-copies the priors; a run must still have one ladder."""
    tempering = PopulationTempering("geometric", ladder())
    assert deepcopy(tempering) is tempering
    assert deepcopy({"a": tempering, "b": tempering})["a"] is tempering


def test_beta_for_reads_the_one_ladder():
    betas = ladder()
    tempering = PopulationTempering("geometric", betas)
    indices = np.array([0, 3, 11, 3])
    assert np.array_equal(tempering.beta_for(indices), betas[indices])


# ----------------------------------------------------------------------
# small contracts
# ----------------------------------------------------------------------


def test_the_settings_builder_is_exported():
    """It is what the settings module imports, so it belongs in the public surface."""
    import lisatools.globalfit.priors.tempering as module

    assert "population_tempering_from_settings" in module.__all__
    assert callable(module.population_tempering_from_settings)


def test_flat_at_zero_temperature_warns_about_nothing():
    """``0 * -inf`` is nan, discarded by the mask -- but numpy computes it first.

    Without the errstate guard this prints an invalid-value warning on every call, which
    over a run buries anything worth reading.
    """
    tempering = PopulationTempering("flat", ladder())
    values = np.array([1.0, -np.inf, 3.0])
    with np.errstate(all="raise"):
        got = tempering.combine(values, 0.0)
    assert got[0] == 0.0 and np.isneginf(got[1]) and got[2] == 0.0


def test_combine_accepts_a_plain_list():
    """``log_p_selected.dtype`` is not something every caller can offer."""
    tempering = PopulationTempering("flat", ladder())
    got = tempering.combine(np.asarray([2.0, 4.0]), np.asarray([0.5, 0.5]))
    assert np.allclose(got, [1.0, 2.0])
