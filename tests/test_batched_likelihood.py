"""The batched likelihood must produce the SAME numbers as the serial loop.

WHY EVERY TEST HERE ASSERTS ``n_batch_fallbacks == 0``
    ``eryn_likelihood_wrap`` falls back to the serial loop when a batch is
    refused. So "batched == serial" is trivially true for a batched path that
    never ran -- which is how a mis-wired call site survived review once:
    ``template_likelihood`` takes ``include_psd_info``, not ``source_only``,
    and passing the latter raised ``TypeError`` inside ``inner_product`` on
    every call, was swallowed by a then-blanket fallback, and produced correct
    serial numbers under a warning nobody read.

    Without the fallback assertion these tests are vacuous.
"""

from __future__ import annotations

import unittest

import numpy as np


class _BatchableGen:
    """Deterministic stub whose output depends on its parameters.

    Accepts scalars (one signal, ``(3, N)``) or equal-length arrays (a batch,
    ``(B, 3, N)``) -- precisely the contract ``supports_batch`` advertises.
    """

    supports_batch = True

    def __init__(self, n_active: int):
        self.n_active = n_active
        self.max_rows_in_a_call = 0
        self.n_calls = 0

    def __call__(self, a, b, **kwargs):
        self.n_calls += 1
        a = np.atleast_1d(np.asarray(a, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        self.max_rows_in_a_call = max(self.max_rows_in_a_call, a.size)
        k = np.arange(self.n_active, dtype=float)
        arr = (a[:, None] + 1.0) * np.cos(0.01 * k)[None, :] \
            + 1j * (b[:, None] + 1.0) * np.sin(0.02 * k)[None, :]
        stack = np.repeat(arr[:, None, :], 3, axis=1)           # (B, 3, N)
        return stack[0] if a.size == 1 else stack


class _AxisSwappedGen(_BatchableGen):
    """Returns ``(nchannels, nrows, N)`` -- the axes the wrong way round."""

    def __call__(self, a, b, **kwargs):
        out = super().__call__(a, b, **kwargs)
        return out if out.ndim == 2 else np.swapaxes(out, 0, 1)


def _make(gen_cls=_BatchableGen, **kwargs):
    from lisatools import detector as lisa
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix

    settings = FDSettings(
        N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2, force_backend="cpu"
    )
    sens = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)
    rng = np.random.default_rng(0)
    data_arr = (rng.normal(size=(3, settings.N_active))
                + 1j * rng.normal(size=(3, settings.N_active))) * 1e-21
    data = FDSignal(data_arr, settings)
    gen = gen_cls(settings.N_active)
    return AnalysisContainer(data, sens, signal_gen=gen, **kwargs), gen


def _params(n):
    rng = np.random.default_rng(42)
    return np.column_stack([rng.uniform(1.0, 2.0, n), rng.uniform(3.0, 4.0, n)])


class BatchedEquivalenceTest(unittest.TestCase):

    def setUp(self):
        try:
            self.aca, self.gen = _make()
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"container unavailable: {exc}")

    def _assert_batched_actually_ran(self):
        self.assertEqual(
            self.aca.n_batch_fallbacks, 0,
            f"the batched path fell back to serial "
            f"({self.aca.last_batch_error!r}), so this comparison proves "
            f"nothing about batching",
        )

    def test_batched_matches_serial(self):
        for B in (2, 4, 8):
            with self.subTest(B=B):
                aca, _ = _make()
                x = _params(B)
                aca.batch_evaluation = False
                serial = aca.eryn_likelihood_wrap(x)
                aca.batch_evaluation = True
                batched = aca.eryn_likelihood_wrap(x)
                self.assertEqual(aca.n_batch_fallbacks, 0)
                self.assertEqual(serial.shape, (B,))
                self.assertEqual(batched.shape, (B,))
                self.assertTrue(np.all(np.isfinite(batched)))
                np.testing.assert_allclose(batched, serial, rtol=0, atol=1e-9)

    def test_public_method_matches_the_wrapper(self):
        x = _params(4)
        direct = self.aca.batched_signal_likelihood(x)
        self._assert_batched_actually_ran()
        viawrap = self.aca.eryn_likelihood_wrap(x)
        np.testing.assert_allclose(direct, viawrap, rtol=0, atol=0)

    def test_batched_is_actually_one_launch(self):
        self.aca.eryn_likelihood_wrap(_params(6))
        self._assert_batched_actually_ran()
        self.assertEqual(
            self.gen.max_rows_in_a_call, 6,
            f"expected all 6 rows in one generator call; largest carried "
            f"{self.gen.max_rows_in_a_call}",
        )

    def test_rows_are_not_interchangeable(self):
        """Guards against one value broadcast across rows."""
        out = self.aca.eryn_likelihood_wrap(_params(4))
        self._assert_batched_actually_ran()
        self.assertGreater(
            len(np.unique(out)), 1,
            "every row returned the same likelihood; the per-source axis is "
            "being collapsed somewhere",
        )

    def test_chunking_does_not_change_the_answer(self):
        x = _params(8)
        self.aca.batch_max_size = None
        whole = self.aca.eryn_likelihood_wrap(x)
        self.aca.batch_max_size = 3
        chunked = self.aca.eryn_likelihood_wrap(x)
        self._assert_batched_actually_ran()
        np.testing.assert_allclose(chunked, whole, rtol=0, atol=0)

    def test_batch_max_size_is_a_constructor_kwarg(self):
        aca, _ = _make(batch_max_size=2, batch_evaluation=True)
        self.assertEqual(aca.batch_max_size, 2)
        self.assertTrue(aca.batch_evaluation)

    def test_axis_swapped_template_is_refused(self):
        """(nchannels, nrows, N) must not pass as (nrows, nchannels, N).

        With 3 walkers against 3 TDI channels a length-only check cannot tell
        the two apart, and the result would be channel mixtures presented as
        per-walker likelihoods: finite, plausible, wrong.
        """
        aca, _ = _make(_AxisSwappedGen)
        with self.assertRaises(ValueError):
            aca.batched_signal_likelihood(_params(4))

    def test_axis_swap_is_undetectable_when_rows_equal_channels(self):
        """Documents a real blind spot rather than pretending it is covered.

        With 3 walkers against 3 TDI channels, ``(nrows, nchannels, N)`` and
        ``(nchannels, nrows, N)`` are the same shape, so no shape check can
        tell them apart. The mitigation is structural, not defensive: use
        ``BatchedDomainSignalGen``, which builds the stack itself and fixes
        the axis order by construction.

        If this test ever starts failing because the swap IS caught, that is
        good news -- delete it.
        """
        aca, _ = _make(_AxisSwappedGen)
        out = aca.batched_signal_likelihood(_params(3))
        self.assertEqual(out.shape, (3,))

    def test_refusal_falls_back_and_is_counted(self):
        """A genuine refusal is not an error: fall back, warn, count."""
        from lisatools.utils.exceptions import BatchNotLaunchable

        def _refuse(*a, **k):
            raise BatchNotLaunchable("merger times spread too wide")

        self.aca.batched_signal_likelihood = _refuse
        with self.assertWarns(RuntimeWarning):
            out = self.aca.eryn_likelihood_wrap(_params(4))
        self.assertEqual(self.aca.n_batch_fallbacks, 1)
        self.assertEqual(out.shape, (4,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_non_refusal_errors_propagate(self):
        """A bug in our own call site must NOT masquerade as a refusal."""
        def _bug(*a, **k):
            raise TypeError("inner_product() got an unexpected keyword 'x'")

        self.aca.batched_signal_likelihood = _bug
        with self.assertRaises(TypeError):
            self.aca.eryn_likelihood_wrap(_params(4))
        self.assertEqual(
            self.aca.n_batch_fallbacks, 0,
            "a mis-wire must not be counted or hidden as a refusal",
        )


class AdapterTest(unittest.TestCase):
    """``BatchedDomainSignalGen`` turns a TD generator into a signal_gen."""

    def test_supports_batch_is_forwarded_not_manufactured(self):
        from lisatools.sources.batching import BatchedDomainSignalGen

        class _No:
            pass

        class _Yes:
            supports_batch = True

        self.assertFalse(BatchedDomainSignalGen(_No()).supports_batch)
        self.assertTrue(BatchedDomainSignalGen(_Yes()).supports_batch)

    def test_stack_rejects_ragged_sources(self):
        from lisatools.sources.batching import BatchedDomainSignalGen

        class _Dom:
            def __init__(self, arr):
                self.arr = arr
                self.settings = object()

        with self.assertRaises(ValueError):
            BatchedDomainSignalGen._stack(
                [_Dom(np.zeros((3, 8))), _Dom(np.zeros((3, 9)))]
            )

    def test_stack_refuses_empty(self):
        from lisatools.sources.batching import BatchedDomainSignalGen

        with self.assertRaises(ValueError):
            BatchedDomainSignalGen._stack([])


class GridAlignedContractTest(unittest.TestCase):
    """Regressions on the grid-aligned generator's own contract."""

    def test_reference_quantities_forwarded_whole(self):
        """``t_min`` must reach phentax, not be replaced by NaN.

        ``get_reference_quantities`` adds ``t_min = -T`` whenever
        ``time_bounded_start`` is set -- the DEFAULT -- and phentax derives the
        start from ``f_min`` only when ``t_min`` is NaN. Filling the
        positionals by hand and passing NaN un-bounded the template in time:
        57,789 valid samples against the stock 525,970 at m1 = 1e7,
        m2 = 8e6 Msun. A physics divergence wearing grid-alignment's clothes.
        """
        import inspect

        from lisatools.sources.bbh.gridaligned import (
            GridAlignedPhenomTHMTDIWaveform,
        )

        src = inspect.getsource(
            GridAlignedPhenomTHMTDIWaveform._aligned_polarizations
        )
        self.assertIn(
            "**ref_kw", src,
            "reference quantities must be forwarded WHOLE to "
            "initial_processing; naming individual keys silently drops t_min",
        )
        self.assertNotIn(
            "jnp.nan", src,
            "initial_processing's t_min positional must not be hardcoded NaN",
        )

    def test_supports_batch_tracks_alignment(self):
        """One decision in one place."""
        from lisatools.sources.bbh.gridaligned import (
            GridAlignedPhenomTHMTDIWaveform,
        )

        self.assertIsInstance(
            GridAlignedPhenomTHMTDIWaveform.__dict__["supports_batch"],
            property,
            "supports_batch must be a property, not a constant that can "
            "disagree with grid_align",
        )

        class _Stub:
            grid_align = True
            supports_batch = GridAlignedPhenomTHMTDIWaveform.supports_batch

        stub = _Stub()
        self.assertTrue(stub.supports_batch)
        stub.grid_align = False
        self.assertFalse(
            stub.supports_batch,
            "with alignment off the generator must stop advertising batching",
        )

    def test_exported_from_package(self):
        from lisatools.sources.bbh import GridAlignedPhenomTHMTDIWaveform  # noqa: F401


if __name__ == "__main__":
    unittest.main()
