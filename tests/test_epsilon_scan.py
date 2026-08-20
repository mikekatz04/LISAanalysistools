"""Tests for D14, the support floor's grip on the Bayes factor.

``SupportFloor`` turns a ``-inf`` log-density into ``ln eps - ln|B|``. That makes the
model move measurable, and it makes the number it reports partly a property of ``eps``.
The question these tests pin is how much of it is: in the regime the floor was
introduced for, the floored density stops depending on the population altogether, so

    d ln B_10 / d ln eps  =  n_floored(M_1) - n_floored(M_0),

and a slope of one means the Bayes factor is a restatement of ``eps``. ``epsilon_scan``
measures that slope from a finished recording without rerunning anything, which is
possible only because the galfor term is one evaluation per state and the mixture
inverts exactly.

Each test builds a synthetic recording with ``HyperMoveRecorder`` directly: no flows, no
catalogues, no move. The arrays are chosen so every expected number is known by hand,
which is the only way an assertion about a slope means anything.

See section 10.6 of ``_dev/why_the_model_index_does_not_jump.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
if not cupy.cuda.is_available():
    # importing lisatools.globalfit initialises CUDA and aborts without a device,
    # which pytest cannot turn into a skip -- so bail out before importing it
    pytest.skip("requires a CUDA device", allow_module_level=True)

hyperdiagnostics = pytest.importorskip("lisatools.globalfit.hyperdiagnostics")
network = pytest.importorskip("lisatools.globalfit.priors.network")

HyperMoveRecorder = hyperdiagnostics.HyperMoveRecorder
HyperMoveDiagnostics = hyperdiagnostics.HyperMoveDiagnostics

NMODELS = 2
NTEMPS = 2
NWALKERS = 4
NROWS = 5

N_TOT = (3.78e7, 1.57e7)
LOG_UNIFORM = -4.105  # ln|B| = 4.105, the measured galfor union box
EPSILON = 1e-6


def build_recording(tmp_path, raw_stochastic, epsilon=EPSILON, with_floor=True):
    """A recording whose stochastic term is ``raw_stochastic`` put through the floor.

    Args:
        raw_stochastic: ``ln p_flow(Sigma | m)``, anything broadcastable to
            ``(nmodels, ntemps, nwalkers)``; a per-model list is the common case.
            ``-inf`` is allowed and is the B9 case.
        epsilon: The floor the "run" was made with.
        with_floor: When ``False`` the recording carries no floor attributes, as a run
            with ``SUPPORT_FLOOR_GALFOR = None`` produces.
    """
    shape = (NMODELS, NTEMPS, NWALKERS)
    raw = np.asarray(raw_stochastic, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None, None]
    raw = np.broadcast_to(raw, shape)
    stochastic = (
        network.refloor_log_density(raw, epsilon, LOG_UNIFORM) if with_floor else raw
    )

    # the model-independent parts are arbitrary but must be identical across models
    # where the test wants the stochastic term to be the only thing separating them
    resolved = np.zeros(shape)
    intensity = np.zeros(shape)
    n1_expected = np.zeros(shape)
    model_log_prior = np.zeros(NMODELS)
    ell = intensity - n1_expected + stochastic + model_log_prior[:, None, None]

    metadata = {}
    if with_floor:
        metadata = dict(
            stochastic_support_floor_epsilon=float(epsilon),
            stochastic_support_floor_log_uniform=float(LOG_UNIFORM),
        )

    filename = tmp_path / "recording.h5"
    recorder = HyperMoveRecorder(
        filename,
        nmodels=NMODELS,
        ntemps=NTEMPS,
        nwalkers=NWALKERS,
        n_tot=N_TOT,
        model_log_prior=model_log_prior,
        metadata=metadata,
        overwrite=True,
    )
    walker_shape = (NTEMPS, NWALKERS)
    for iteration in range(NROWS):
        recorder.record(
            iteration,
            dict(
                resolved=resolved,
                intensity=intensity,
                n1_expected=n1_expected,
                stochastic=np.asarray(stochastic),
                ell=np.asarray(ell),
                model_current=np.zeros(walker_shape, dtype=np.int16),
                model_proposed=np.ones(walker_shape, dtype=np.int16),
                num_resolved=np.full(walker_shape, 4080, dtype=np.int32),
                accepted=np.zeros(walker_shape, dtype=bool),
            ),
        )
    return HyperMoveDiagnostics(filename)


def test_the_recording_describes_its_own_floor(tmp_path):
    """Without the attributes nothing downstream is possible, so check them first."""
    diagnostics = build_recording(tmp_path, [-2.0, -np.inf])
    epsilon, log_uniform = diagnostics.stochastic_support_floor
    assert epsilon == pytest.approx(EPSILON)
    assert log_uniform == pytest.approx(LOG_UNIFORM)


def test_a_run_without_a_floor_says_so_rather_than_guessing(tmp_path):
    """A recording with the floor off must not be silently treated as floored at some eps."""
    diagnostics = build_recording(tmp_path, [-2.0, -6.0], with_floor=False)
    assert diagnostics.stochastic_support_floor is None
    # the unfloored request is a no-op and is allowed; anything else has no ln|B| to use
    np.testing.assert_allclose(
        diagnostics.stochastic_at_epsilon(None), diagnostics.term("stochastic")
    )
    with pytest.raises(ValueError, match="no support-floor attributes"):
        diagnostics.stochastic_at_epsilon(1e-6)


def test_the_floor_margin_separates_the_two_regimes(tmp_path):
    """The census number: how far above ``ln eps - ln|B|`` each model's density sits."""
    diagnostics = build_recording(tmp_path, [-2.0, -np.inf])
    margin = diagnostics.stochastic_floor_margin()
    level = np.log(EPSILON) + LOG_UNIFORM

    # model 0 is far above the floor, model 1 is exactly on it
    assert np.all(margin[:, 0] > 10.0)
    assert margin[:, 0].mean() == pytest.approx(-2.0 - level, abs=1e-5)
    np.testing.assert_allclose(margin[:, 1], 0.0, atol=1e-9)


def test_undoing_the_floor_recovers_the_flow_and_the_degeneracy(tmp_path):
    """Both halves of the correction, in one recording.

    Where the flow had something to say, the raw value comes back exactly. Where the
    floor was carrying the state, it comes back as ``-inf`` -- which is the answer the
    run would have had without a floor, and the reason §10.6 says the correction is
    available but not useful.
    """
    diagnostics = build_recording(tmp_path, [-2.0, -np.inf])
    raw = diagnostics.stochastic_at_epsilon(None)
    np.testing.assert_allclose(raw[:, 0], -2.0, rtol=1e-9)
    assert np.all(np.isneginf(raw[:, 1]))


def test_moving_epsilon_moves_the_bayes_factor_one_for_one(tmp_path):
    """The headline. Model 1 is floored, model 0 is not, so the slope must be one.

    The separation is kept small enough that the Rao-Blackwell posterior stays finite at
    every grid point; with a production-sized separation ``P(M_1)`` underflows to zero
    and the slope is unmeasurable, which is itself a finding rather than a bug.

    Model 0 must also stay clear of the floor at the *largest* epsilon of the grid, or
    both models are floored and the slope goes to zero for the wrong reason: at
    ``eps = 1e-4`` the floor already sits at ``-13.32`` nats.
    """
    # model 1's flow density is far below the floor, so its floored value *is* the floor
    diagnostics = build_recording(tmp_path, [-2.0, -60.0])
    grid = [1e-4, 1e-5, 1e-6, 1e-7]
    scan = diagnostics.epsilon_scan(grid, reweight=False)

    assert scan["slope"] == pytest.approx(1.0, abs=1e-3)
    # and the mechanism, not merely the fit: ell_1 tracks ln eps exactly
    for row, epsilon in zip(scan["rows"], grid):
        expected = (np.log(epsilon) + LOG_UNIFORM) - (-2.0)
        assert row["mean_delta_ell"] == pytest.approx(expected, abs=1e-3)


def test_the_slope_is_zero_when_both_models_are_above_the_floor(tmp_path):
    """The case that licenses quoting a Bayes factor, and it must measure as zero."""
    diagnostics = build_recording(tmp_path, [-2.0, -4.0])
    scan = diagnostics.epsilon_scan([1e-4, 1e-5, 1e-6, 1e-7], reweight=False)

    assert scan["slope"] == pytest.approx(0.0, abs=1e-4)
    for row in scan["rows"]:
        assert row["mean_delta_ell"] == pytest.approx(-2.0, abs=1e-3)


def test_the_text_report_states_the_verdict(tmp_path):
    """The scan is read by a person; the wording must not need interpreting."""
    floored = build_recording(tmp_path, [-2.0, -60.0])
    text = floored.epsilon_scan_text([1e-4, 1e-5, 1e-6, 1e-7], reweight=False)
    assert "FLOOR-DRIVEN" in text
    assert "restatement of epsilon" in text

    inert = build_recording(tmp_path / "inert", [-2.0, -4.0])
    assert "INERT" in inert.epsilon_scan_text([1e-4, 1e-5, 1e-6], reweight=False)


def test_reweighting_to_the_recorded_epsilon_is_a_no_op(tmp_path):
    """The round trip has to be exact, or every reweighted number inherits its error."""
    diagnostics = build_recording(tmp_path, [-2.0, -60.0])
    row = diagnostics.epsilon_scan([EPSILON], reweight=True)["rows"][0]
    assert row["effective_sample_size"] == pytest.approx(
        float(NROWS * NWALKERS), rel=1e-9
    )
    np.testing.assert_allclose(
        row["posterior"], diagnostics.rao_blackwell(), rtol=1e-9
    )


def test_reweighting_away_from_the_floor_collapses_the_sample_size(tmp_path):
    """The caveat on the exact correction, pinned as a number.

    Undoing the floor reweights towards a target that gives the floored states less
    mass, so the states the floor was carrying stop counting. Here half the walkers have
    a model 0 density *below* the floor, so at the recorded epsilon their posterior mass
    is the floor's doing entirely; pushing epsilon down by six orders of magnitude takes
    it away and their importance weights fall by ``e^-7``. The ESS is what makes that
    visible instead of leaving a confident-looking posterior built on the other half.
    """
    # walkers 2 and 3 sit below the floor under model 0 as well, so the floor is the only
    # thing giving them any mass at all
    raw = np.zeros((NMODELS, NTEMPS, NWALKERS))
    raw[0] = np.array([-2.0, -2.0, -25.0, -25.0])
    raw[1] = -np.inf
    diagnostics = build_recording(tmp_path, raw)

    rows = diagnostics.epsilon_scan([EPSILON, 1e-12], reweight=True)["rows"]
    at_recorded, far_away = rows
    total = float(NROWS * NWALKERS)

    assert at_recorded["effective_sample_size"] == pytest.approx(total, rel=1e-9)
    assert far_away["effective_sample_size"] < 0.6 * total


def test_only_the_stochastic_term_is_touched(tmp_path):
    """(T) is counted from the catalogues, so the floor cannot and does not move it.

    That is the inconsistency §10.6 flags rather than a property to rely on, but
    ``ell_at_epsilon`` must reproduce the code's actual behaviour, not the idealised
    mixture, or the scan would be measuring something the run does not do.
    """
    diagnostics = build_recording(tmp_path, [-2.0, -30.0])
    ell = diagnostics.term("ell")
    moved = diagnostics.ell_at_epsilon(1e-9)
    difference = moved - ell
    stochastic_difference = (
        diagnostics.stochastic_at_epsilon(1e-9) - diagnostics.term("stochastic")
    )
    np.testing.assert_allclose(difference, stochastic_difference, rtol=1e-12)
