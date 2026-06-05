"""Unit tests for ``AnalysisContainerArray._vectorized_dispatch``.

Covers the CPU path of the dispatcher introduced as the multi-GPU
replacement for ``_loop_operation``. The multi-GPU parity test requires
two CUDA devices and lives in a separate manual harness.

Cases:

* No payload, no index (legacy parameterless "apply to every AC" case):
  result shape matches ``acs_shape`` and entries match a direct per-AC
  loop.
* No payload, explicit subset ``index`` (e.g. ``[0]`` or ``[1, 0]``):
  output is ``(len(index),)`` and entries match the per-AC values picked
  by ``index``.
* Property dispatch (``start_freq_ind`` etc.): no payload, properties
  flow through the dispatcher and reshape back to ``acs_shape``.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import cupy as cp  # noqa: F401 -- only needed for the optional GPU branch
    HAVE_CUPY = True
except (ImportError, ModuleNotFoundError):
    HAVE_CUPY = False


def _build_test_acs(num_acs: int = 2):
    """Return ``num_acs`` AnalysisContainers sharing one FD grid and sens_mat.

    The data is just zeros; the inner_product / likelihood values are
    deterministic per-AC (same input → same output), which is all we need
    for the per-AC dispatch contract.
    """
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix
    from lisatools import detector as lisa

    settings = FDSettings(
        N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2,
        force_backend="cpu",
    )
    sens_mat = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)

    acs = []
    for i in range(num_acs):
        arr = np.zeros((3, settings.N_active), dtype=complex)
        # Seed a unique value in each AC's residual so per-AC inner products
        # are distinguishable.
        arr[0, 0] = float(i + 1)
        data = FDSignal(arr, settings)
        acs.append(AnalysisContainer(data, sens_mat))
    return acs, settings, sens_mat


class VectorizedDispatchTest(unittest.TestCase):
    def setUp(self):
        try:
            self.acs, self.settings, self.sens_mat = _build_test_acs(num_acs=3)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")

    def _make_aca(self):
        from lisatools.analysiscontainer import AnalysisContainerArray
        return AnalysisContainerArray(self.acs, gpus=None)

    def test_no_payload_no_index_matches_loop(self):
        aca = self._make_aca()
        ref = np.array([ac.inner_product() for ac in self.acs])
        out = aca.inner_product()
        self.assertEqual(out.shape, (len(self.acs),))
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)

    def test_no_payload_subset_index(self):
        aca = self._make_aca()
        idx = np.array([2, 0])
        out = aca.inner_product(index=idx)
        ref = np.array([self.acs[i].inner_product() for i in idx])
        self.assertEqual(out.shape, idx.shape)
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)

    def test_no_payload_duplicate_index(self):
        """Multiple rows targeting the same AC should each get that AC's value."""
        aca = self._make_aca()
        idx = np.array([1, 1, 0])
        out = aca.inner_product(index=idx)
        ref = np.array([self.acs[i].inner_product() for i in idx])
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)

    def test_property_dispatch_returns_per_ac_values(self):
        """``start_freq_ind`` is a property; dispatcher should forward it."""
        aca = self._make_aca()
        out = aca.start_freq_ind
        # Should match a direct per-AC fetch.
        ref = np.array([ac.start_freq_ind for ac in self.acs])
        self.assertEqual(out.shape, ref.shape)
        np.testing.assert_array_equal(out, ref)

    def test_gather_linear_data_arr_single_shard_returns_buffer(self):
        """``gather_linear_data_arr()`` on a single shard returns the buffer directly."""
        aca = self._make_aca()
        gathered = aca.gather_linear_data_arr()
        # Identity equality: single-shard returns the underlying buffer
        # without a copy.
        self.assertIs(gathered, aca.linear_data_arr[0])


if __name__ == "__main__":
    unittest.main()
