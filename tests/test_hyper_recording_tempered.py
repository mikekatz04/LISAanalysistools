"""The recording keeps the tempered terms beside the raw ones, and the reader takes raw.

Section 6.4 of ``_dev/prior_tempering.md``. The Rao-Blackwellised model posterior and the
Bayes factor are statements about the *target*, so they need :math:`\\ell_m` as equation
(1) defines it. Substituting the beta-weighted array returns the model posterior of a
different distribution, and does it silently: the two arrays have the same shape, the
same sign and, at the cold end, the same values. The note asserted for some time that
the reader checked this. It did not, so these tests exist to make the claim true and keep
it true.

``hyperdiagnostics`` depends only on ``h5py`` and ``numpy``, so unlike the rest of the
model-move tests these need no GPU.
"""

import numpy as np
import pytest

from lisatools.globalfit.hyperdiagnostics import HyperMoveRecorder, HyperMoveDiagnostics

NMODELS, NTEMPS, NWALKERS = 2, 4, 3
BETAS = np.array([1.0, 0.5, 0.1, 1e-4])


def make_terms(rng, ell=None, ell_tempered="auto"):
    shape = (NMODELS, NTEMPS, NWALKERS)
    ell = rng.normal(size=shape) * 50.0 if ell is None else ell
    terms = {
        name: rng.normal(size=shape)
        for name in ("resolved", "intensity", "n1_expected", "stochastic")
    }
    terms["ell"] = ell
    if ell_tempered is not None:
        terms["ell_tempered"] = (
            ell * BETAS[None, :, None]
            if isinstance(ell_tempered, str)
            else ell_tempered
        )
    terms["num_resolved"] = np.full((NTEMPS, NWALKERS), 4, dtype=np.int32)
    terms["model_current"] = np.zeros((NTEMPS, NWALKERS), dtype=np.int16)
    terms["model_proposed"] = np.ones((NTEMPS, NWALKERS), dtype=np.int16)
    terms["accepted"] = np.zeros((NTEMPS, NWALKERS), dtype=bool)
    return terms


def recorder(path, scheme="geometric", betas=BETAS, **kwargs):
    metadata = {"population_tempering_scheme": scheme}
    if betas is not None:
        metadata["population_tempering_betas"] = betas
    return HyperMoveRecorder(
        path,
        nmodels=NMODELS,
        ntemps=NTEMPS,
        nwalkers=NWALKERS,
        n_tot=[1.0e6, 4.0e6],
        model_log_prior=np.log([0.5, 0.5]),
        metadata=metadata,
        **kwargs,
    )


def test_both_arrays_are_stored(tmp_path):
    """The acceptance ratio the move actually used must be recoverable afterwards."""
    rng = np.random.default_rng(0)
    path = tmp_path / "d.h5"
    terms = make_terms(rng)
    recorder(path).record(0, terms)

    diagnostics = HyperMoveDiagnostics(path)
    np.testing.assert_allclose(
        diagnostics.term("ell", temperature=2)[0], terms["ell"][:, 2]
    )
    np.testing.assert_allclose(
        diagnostics.term("ell_tempered", temperature=2)[0], terms["ell_tempered"][:, 2]
    )


def test_the_model_posterior_uses_the_raw_terms(tmp_path):
    """The two arrays differ where it matters, and the estimator must follow the raw one.

    Recording both and reading the wrong one would be an easy mistake to make and an
    impossible one to spot in the output, since the result is a well-formed probability
    either way.
    """
    rng = np.random.default_rng(1)
    path = tmp_path / "d.h5"
    terms = make_terms(rng)
    recorder(path).record(0, terms)

    diagnostics = HyperMoveDiagnostics(path)
    raw = diagnostics.raw_ell(temperature=0)
    np.testing.assert_array_equal(raw, diagnostics.term("ell", temperature=0))

    # and the estimator is the softmax of those, not of the tempered ones
    ell = terms["ell"][:, 0]
    shifted = ell - ell.max(axis=0)
    weights = np.exp(shifted) / np.exp(shifted).sum(axis=0)
    np.testing.assert_allclose(
        diagnostics.rao_blackwell(), weights.mean(axis=1), rtol=1e-12
    )


def test_a_recording_whose_raw_terms_are_tempered_is_refused(tmp_path):
    """The failure this guards against: the tempered array written under the raw name.

    Then ``ell`` and ``ell_tempered`` agree at the hottest temperature, where the ladder
    says they cannot, and every number downstream describes the tempered target.
    """
    rng = np.random.default_rng(2)
    path = tmp_path / "d.h5"
    ell = rng.normal(size=(NMODELS, NTEMPS, NWALKERS)) * 50.0
    tempered = ell * BETAS[None, :, None]
    # the mistake: both datasets carry the tempered values
    recorder(path).record(0, make_terms(rng, ell=tempered, ell_tempered=tempered))

    diagnostics = HyperMoveDiagnostics(path)
    with pytest.raises(ValueError, match="written under the raw name"):
        diagnostics.raw_ell()
    with pytest.raises(ValueError, match="written under the raw name"):
        diagnostics.rao_blackwell()


def test_an_untempered_recording_is_not_refused(tmp_path):
    """With the scheme off the two arrays are legitimately identical."""
    rng = np.random.default_rng(3)
    path = tmp_path / "d.h5"
    ell = rng.normal(size=(NMODELS, NTEMPS, NWALKERS)) * 50.0
    recorder(path, scheme="off", betas=None).record(
        0, make_terms(rng, ell=ell, ell_tempered=ell)
    )

    diagnostics = HyperMoveDiagnostics(path)
    np.testing.assert_array_equal(diagnostics.raw_ell()[0], ell[:, 0])


def test_a_recording_without_the_tempered_terms_still_reads(tmp_path):
    """Recordings written before the tempered array existed stay readable.

    The optional term is what buys this; making it required would have stranded every
    existing recording.
    """
    rng = np.random.default_rng(4)
    path = tmp_path / "d.h5"
    terms = make_terms(rng, ell_tempered=None)
    recorder(path, scheme="off", betas=None).record(0, terms)

    diagnostics = HyperMoveDiagnostics(path)
    np.testing.assert_allclose(diagnostics.raw_ell()[0], terms["ell"][:, 0])
    assert diagnostics.rao_blackwell().shape == (NMODELS,)


def test_appending_a_row_without_the_tempered_terms_does_not_raise(tmp_path):
    """A move built without tempering appending to a file that has the dataset."""
    rng = np.random.default_rng(5)
    path = tmp_path / "d.h5"
    rec = recorder(path)
    rec.record(0, make_terms(rng))
    rec.record(1, make_terms(rng, ell_tempered=None))

    diagnostics = HyperMoveDiagnostics(path)
    assert diagnostics.niterations == 2
    # the datasets stay aligned: the row that supplied nothing is nan, not absent,
    # because a dataset growing more slowly than the others would silently stop
    # lining up with them
    assert diagnostics["ell_tempered"].shape == (2, NMODELS, NTEMPS, NWALKERS)
    assert np.isfinite(diagnostics["ell_tempered"][0]).all()
    assert np.isnan(diagnostics["ell_tempered"][1]).all()
    # and the raw terms of both rows survive intact
    assert np.isfinite(diagnostics["ell"]).all()
