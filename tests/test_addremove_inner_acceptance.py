"""Inner moves of ResidualAddOneRemoveOneMove keep standard acceptance counters."""
from __future__ import annotations

import numpy as np
import pytest

from eryn.moves import StretchMove

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove


NTEMPS, NWALKERS, NLEAVES, NDIM = 4, 6, 2, 3


def _build_move(inner_moves):
    """Construct the move with inert collaborators.

    ``__init__`` only stores ``acs`` / ``transform_fn`` / ``priors`` and builds
    one TemperatureControl per leaf, so none of them need to be real here.
    """
    return ResidualAddOneRemoveOneMove(
        branch_name="gb",
        coords_shape=(NTEMPS, NWALKERS, NLEAVES, NDIM),
        waveform_gen=lambda *args, **kwargs: None,
        waveform_gen_kwargs={},
        waveform_like_kwargs={},
        acs=None,
        num_repeats=1,
        transform_fn=None,
        priors=None,
        inner_moves=inner_moves,
    )


def test_inner_moves_start_with_initialised_counters():
    inner = StretchMove()
    move = _build_move([inner])

    assert inner.accepted.shape == (NTEMPS, NWALKERS)
    np.testing.assert_array_equal(inner.accepted, 0.0)
    assert inner.num_proposals == 0


def test_sub_moves_exposes_the_inner_moves():
    a, b = StretchMove(), StretchMove()
    move = _build_move([(a, 1.0), (b, 2.0)])
    assert move.sub_moves == [a, b]


def test_recording_accumulates_beyond_a_single_accept_per_walker():
    # the counter array must be float: `bool += bool` saturates at 1 and would
    # cap every walker at a single accept for the whole run
    inner = StretchMove()
    move = _build_move([inner])

    accepted = np.zeros((NTEMPS, NWALKERS), dtype=bool)
    accepted[0, 0] = True
    for _ in range(3):
        move._record_inner_acceptance(inner, accepted)

    assert inner.accepted[0, 0] == 3.0
    assert inner.num_proposals == 3
    assert inner.acceptance_fraction[0, 0] == pytest.approx(1.0)
    assert inner.acceptance_fraction[0, 1] == pytest.approx(0.0)


def test_recording_drops_the_unused_temperature_tail():
    # `accepted` inside propose is sized (ntemps_full, nwalkers); only the
    # temperatures this move uses are ever set
    inner = StretchMove()
    move = _build_move([inner])

    accepted = np.zeros((NTEMPS + 2, NWALKERS), dtype=bool)
    accepted[0, :] = True
    move._record_inner_acceptance(inner, accepted)

    assert inner.accepted.shape == (NTEMPS, NWALKERS)
    np.testing.assert_array_equal(inner.accepted[0], 1.0)


def test_bespoke_history_attribute_is_gone():
    move = _build_move([StretchMove()])
    assert not hasattr(move, "inner_moves_acceptance_fractions")
