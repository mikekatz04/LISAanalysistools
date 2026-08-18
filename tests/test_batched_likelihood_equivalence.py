"""The batched path must produce the SAME numbers as the serial loop.

WHY THE FALLBACK ASSERTION IS THE POINT OF THIS FILE
----------------------------------------------------
``eryn_likelihood_wrap`` falls back to the serial loop when a batched launch
is refused. So "batched == serial" is trivially true for a batched path that
never ran -- which is exactly how a mis-wired call site survived review once
already: ``template_likelihood`` takes ``include_psd_info``, not
``source_only``, and passing the latter raised TypeError inside
``inner_product`` on every call, was swallowed by the fallback, and produced
correct serial numbers under a warning nobody read.

Every test here therefore asserts ``n_batch_fallbacks == 0`` BEFORE comparing
values. Without that guard these tests are vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest


class _BatchableGen:
    """Deterministic stub whose output depends on its parameters.

    Accepts either scalars (one signal, ``(3, N)``) or equal-length arrays
    (a batch, ``(B, 3, N)``) -- which is precisely the contract
    ``supports_batch`` advertises.
    """

    supports_batch = True

    def __init__(self, n_active: int):
        self.n_active = n_active
        self.n_calls = 0
        self.max_rows_in_a_call = 0

    def __call__(self, a, b, **kwargs):
        self.n_calls += 1
        a = np.atleast_1d(np.asarray(a, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        batched = np.ndim(np.asarray(kwargs.get("_probe", a))) > 0 and a.size > 1
        self.max_rows_in_a_call = max(self.max_rows_in_a_call, a.size)

        k = np.arange(self.n_active, dtype=float)
        # (B, N) -> distinct, parameter-dependent, non-degenerate content
        arr = (a[:, None] + 1.0) * np.cos(0.01 * k)[None, :] \
            + 1j * (b[:, None] + 1.0) * np.sin(0.02 * k)[None, :]
        stack = np.repeat(arr[:, None, :], 3, axis=1)          # (B, 3, N)
        if a.size == 1 and not batched:
            return stack[0]
        return stack


def _container(gen_factory):
    from lisatools import detector as lisa
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix

    settings = FDSettings(
        N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2, force_backend="cpu",
    )
    sens = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)
    rng = np.random.default_rng(0)
    data_arr = (rng.normal(size=(3, settings.N_active))
                + 1j * rng.normal(size=(3, settings.N_active))) * 1e-21
    data = FDSignal(data_arr, settings)
    gen = gen_factory(settings.N_active)
    return AnalysisContainer(data, sens, signal_gen=gen), gen


def _params(n):
    rng = np.random.default_rng(42)
    return np.column_stack([rng.uniform(1.0, 2.0, n), rng.uniform(3.0, 4.0, n)])


@pytest.mark.parametrize("B", [2, 4, 8])
def test_batched_matches_serial(B):
    """The headline. One launch, same numbers as the loop."""
    try:
        aca, gen = _container(_BatchableGen)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"container unavailable: {exc}")

    x = _params(B)

    aca.batch_evaluation = False
    serial = aca.eryn_likelihood_wrap(x)

    aca.batch_evaluation = True
    batched = aca.eryn_likelihood_wrap(x)

    assert aca.n_batch_fallbacks == 0, (
        f"the batched path fell back to serial ({aca.last_batch_error!r}), so "
        f"this comparison proves nothing about batching"
    )
    assert np.all(np.isfinite(serial)) and np.all(np.isfinite(batched))
    assert serial.shape == batched.shape == (B,)
    np.testing.assert_allclose(batched, serial, rtol=0, atol=1e-9)


def test_batched_is_actually_one_launch():
    """Not a loop wearing a batch's clothes."""
    try:
        aca, gen = _container(_BatchableGen)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"container unavailable: {exc}")

    aca.batch_evaluation = True
    aca.eryn_likelihood_wrap(_params(6))
    assert aca.n_batch_fallbacks == 0
    assert gen.max_rows_in_a_call == 6, (
        f"expected all 6 rows in one generator call; largest call carried "
        f"{gen.max_rows_in_a_call}"
    )


def test_rows_are_not_interchangeable():
    """Guards against a batch that returns one value broadcast to all rows."""
    try:
        aca, _ = _container(_BatchableGen)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"container unavailable: {exc}")

    aca.batch_evaluation = True
    out = aca.eryn_likelihood_wrap(_params(4))
    assert aca.n_batch_fallbacks == 0
    assert len(np.unique(out)) > 1, (
        "every row returned the same likelihood; the per-source axis is being "
        "collapsed somewhere"
    )


def test_chunking_does_not_change_the_answer():
    """batch_max_size is a memory control, not a numerical one."""
    try:
        aca, _ = _container(_BatchableGen)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"container unavailable: {exc}")

    x = _params(8)
    aca.batch_evaluation = True
    aca.batch_max_size = None
    whole = aca.eryn_likelihood_wrap(x)
    aca.batch_max_size = 3
    chunked = aca.eryn_likelihood_wrap(x)

    assert aca.n_batch_fallbacks == 0
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=0)


def test_miswired_call_site_raises_instead_of_falling_back():
    """A bug in our own call site must not masquerade as a refusal."""
    try:
        aca, _ = _container(_BatchableGen)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"container unavailable: {exc}")

    def _bad_kwarg(*a, **k):
        raise TypeError("inner_product() got an unexpected keyword argument 'x'")

    aca.template_likelihood = _bad_kwarg
    aca.batch_evaluation = True
    with pytest.raises(RuntimeError, match="mis-wired"):
        aca.eryn_likelihood_wrap(_params(4))
    assert aca.n_batch_fallbacks == 0, "a mis-wire must not be counted as a refusal"
