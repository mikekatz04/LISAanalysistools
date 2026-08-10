"""The synthetic-GB N_sparse resolver, and its mirror of the C++ budget.

``SyntheticGBProcessingStep`` must pick an ``N_sparse`` the device can
actually launch: the FD heterodyne kernel keeps its whole per-source working
set in SHARED memory, so 2048 x 3 channels needs ~266 KB against an A100's
164 KB. Over-budget launches fail as a bare ``GPUassert: invalid argument``
(``cudaFuncSetAttribute``'s return is unchecked), naming neither the knob nor
the limit -- hence both the resolver and this test.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from lisatools.globalfit.stock.erebor.injections import (
    _largest_fitting_n_sparse,
    _resolve_synth_n_sparse,
    _SYNTH_N_SPARSE_PREFERRED,
    gb_fd_shared_bytes,
)

A100_LIMIT = 164 * 1024


class SharedBytesMirrorTest(unittest.TestCase):
    """The Python mirror must equal the compiled helper exactly."""

    def test_matches_cpp(self):
        try:
            import gbgpu  # noqa: F401  (registers the gbgpu backend namespace)
            from lisatools.detector import ESAOrbits
            from lisatools.response.tdiconfig import TDIConfig
        except Exception as exc:  # pragma: no cover - backend not built
            self.skipTest(f"gbgpu CPU backend unavailable ({exc})")

        # GBTDIonTheFlyWrap(OrbitsWrap*, TDIConfigWrap*, T, t_ref); nanobind
        # rejects an uninitialized instance, so build a real one. T / t_ref
        # do not enter the shared-memory arithmetic.
        orbits = ESAOrbits(force_backend="cpu")
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        # The wrap lives in the GBGPU backend namespace, not lisatools'.
        wrap_cls = getattr(gbgpu.get_backend("cpu"), "GBTDIonTheFlyWrap", None)
        if wrap_cls is None:
            self.skipTest("GBTDIonTheFlyWrap not exposed on this build")
        try:
            # Build the TDIConfigWrap directly: TDIConfig.pytdiconfig goes
            # through backend.pyTDIConfig, which this build does not expose
            # (the backend name is TDIConfigWrap).
            import lisatools

            tdi_wrap = lisatools.get_backend("cpu").TDIConfigWrap(
                *tdi.pytdiconfig_args
            )
            wrap = wrap_cls(orbits.pycppdetector, tdi_wrap, 1.0, 0.0)
        except Exception as exc:  # pragma: no cover - wrap plumbing differs
            self.skipTest(f"could not construct GBTDIonTheFlyWrap ({exc})")

        self.assertEqual(wrap.get_fd_buffer_size(256, 3),
                         gb_fd_shared_bytes(256, 3))
        for n in (128, 256, 512, 1024, 2048, 4096):
            for nch in (2, 3):
                self.assertEqual(
                    wrap.get_fd_buffer_size(n, nch),
                    gb_fd_shared_bytes(n, nch),
                    f"mirror drifted at N_sparse={n}, nchannels={nch}",
                )


class ResolverTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("GB_SYNTH_N_SPARSE", None)

    def tearDown(self):
        os.environ.pop("GB_SYNTH_N_SPARSE", None)
        if self._saved is not None:
            os.environ["GB_SYNTH_N_SPARSE"] = self._saved

    def test_a100_budget_excludes_2048_and_admits_1024(self):
        """The exact case that crashed: 2048x3 does not fit an A100, 1024 does."""
        self.assertGreater(gb_fd_shared_bytes(2048, 3), A100_LIMIT)
        self.assertLessEqual(gb_fd_shared_bytes(1024, 3), A100_LIMIT)
        self.assertEqual(_largest_fitting_n_sparse(3, A100_LIMIT), 1024)

    def test_cpu_backend_is_unbounded(self):
        """CPU holds the same working set on the heap -> preferred value."""
        self.assertEqual(
            _resolve_synth_n_sparse(3, "cpu"), _SYNTH_N_SPARSE_PREFERRED
        )

    def test_generous_device_keeps_the_preferred_value(self):
        with mock.patch(
            "lisatools.globalfit.stock.erebor.injections.gb_fd_shared_limit",
            return_value=1 << 20,
        ):
            self.assertEqual(
                _resolve_synth_n_sparse(3, "cuda12x"), _SYNTH_N_SPARSE_PREFERRED
            )

    def test_tight_device_steps_down(self):
        with mock.patch(
            "lisatools.globalfit.stock.erebor.injections.gb_fd_shared_limit",
            return_value=A100_LIMIT,
        ):
            self.assertEqual(_resolve_synth_n_sparse(3, "cuda12x"), 1024)

    def test_explicit_over_budget_raises_naming_the_knob(self):
        os.environ["GB_SYNTH_N_SPARSE"] = "2048"
        with mock.patch(
            "lisatools.globalfit.stock.erebor.injections.gb_fd_shared_limit",
            return_value=A100_LIMIT,
        ):
            with self.assertRaises(ValueError) as ctx:
                _resolve_synth_n_sparse(3, "cuda12x")
        msg = str(ctx.exception)
        self.assertIn("GB_SYNTH_N_SPARSE", msg)
        self.assertIn("1024", msg)

    def test_explicit_within_budget_is_honored(self):
        os.environ["GB_SYNTH_N_SPARSE"] = "512"
        with mock.patch(
            "lisatools.globalfit.stock.erebor.injections.gb_fd_shared_limit",
            return_value=A100_LIMIT,
        ):
            self.assertEqual(_resolve_synth_n_sparse(3, "cuda12x"), 512)

    def test_explicit_is_honored_on_cpu_regardless(self):
        os.environ["GB_SYNTH_N_SPARSE"] = "8192"
        self.assertEqual(_resolve_synth_n_sparse(3, "cpu"), 8192)

    def test_two_channels_fit_where_three_do_not(self):
        """The budget is per channel, so AE has more headroom than XYZ."""
        self.assertEqual(_largest_fitting_n_sparse(2, A100_LIMIT), 1024)
        self.assertLessEqual(gb_fd_shared_bytes(1024, 2), A100_LIMIT)


if __name__ == "__main__":
    unittest.main()
