"""Tests for :mod:`lisatools.sensitivity` PSD and sensitivity-matrix builders."""

import numpy as np
import time

import unittest

try:
    import cupy as cp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from lisatools.sensitivity import (
    get_sensitivity,
    AET1SensitivityMatrix,
    XYZ1SensitivityMatrix,
    AET2SensitivityMatrix,
    XYZ2SensitivityMatrix,
    AE1SensitivityMatrix,
    AE2SensitivityMatrix,
    SensitivityMatrixBase,
    SensitivityMatrix,
    CompositeSensitivityMatrix,
    InstrumentNoise,
    GalacticForeground,
    SGWB,
    A2TDISens,
    E2TDISens,
    T2TDISens,
)
from lisatools.stochastic import PowerLawSGWB
from lisatools.domains import FDSettings, STFTSettings
from lisatools.utils.constants import *
from lisatools import detector as lisa

import sys

class SensitivityTest(unittest.TestCase):
    """Sanity checks for :func:`get_sensitivity` and the TDI sensitivity-matrix classes."""

    def _make_fd_settings(self):
        """Uniform FD grid covering the LISA band (the matrix builders require ``FDSettings``)."""
        force_backend = "gpu" if gpu_available else "cpu"
        # df = 1e-4 Hz, N = 10001 covers ~[0, 1] Hz; mask the band of interest.
        return FDSettings(
            N=10001, df=1e-4, min_freq=1e-4, max_freq=1.0,
            force_backend=force_backend,
        )

    def _make_stft_settings(self, NT=8):
        """STFT grid over the same LISA band as ``_make_fd_settings`` with ``NT`` time segments."""
        force_backend = "gpu" if gpu_available else "cpu"
        # Same df / band as the FD helper so the per-frequency PSD matches.
        return STFTSettings(
            t0=0.0, dt=86400.0, df=1e-4, NT=NT, NF=10001,
            min_freq=1e-4, max_freq=1.0,
            force_backend=force_backend,
        )

    def test_get_sen(self):
        """Verify :func:`get_sensitivity` returns finite values for the X1 TDI PSD."""
        xp = cp if gpu_available else np

        Sn = get_sensitivity(self._make_fd_settings(), sens_fn="X1TDISens", model=lisa.sangria)

        self.assertFalse(xp.any(xp.isnan(Sn)))

    def test_get_sen_stft(self):
        """STFT PSD replicates the FD PSD across every time segment.

        With the stationary-noise assumption, :func:`get_sensitivity` on an
        ``STFTSettings`` should return shape ``(NT, NF_active)`` where every row
        equals the FD PSD evaluated on the same active frequency bins.
        """
        xp = cp if gpu_available else np

        stft_settings = self._make_stft_settings(NT=8)
        Sn_stft = get_sensitivity(stft_settings, sens_fn="X1TDISens", model=lisa.sangria)

        # Shape must match basis_shape_active == (NT, NF_active).
        self.assertEqual(Sn_stft.shape, stft_settings.basis_shape_active)
        self.assertFalse(xp.any(xp.isnan(Sn_stft)))

        # The per-frequency reference: get_Sn on the same active frequency bins.
        Sn_ref = get_sensitivity(self._make_fd_settings(), sens_fn="X1TDISens", model=lisa.sangria)
        # FD active band is [1e-4, 1.0] -> identical f_arr to the STFT band.
        xp.testing.assert_array_equal(stft_settings.f_arr, self._make_fd_settings().f_arr)
        for it in range(stft_settings.NT):
            xp.testing.assert_array_equal(Sn_stft[it], Sn_ref)

    def _test_sens_mat(self, sens_mat_class, model):
        """Construct ``sens_mat_class`` over a uniform LISA-band grid for ``model``."""
        settings = self._make_fd_settings()
        Sn = sens_mat_class(settings, model=model)
        # The matrix must be finite on the active band and inv/det must compute.
        self.assertFalse(np.any(np.isnan(np.asarray(Sn.sens_mat))))
        self.assertEqual(Sn.sens_mat.shape[-1], settings.N_active)
        # Reading invC triggers the lazy compute path.
        _ = Sn.invC
        _ = Sn.detC

    def test_sensitivity_matrix_AET1(self):
        """Build an :class:`AET1SensitivityMatrix` for the ``sangria`` noise model."""
        self._test_sens_mat(AET1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AET2(self):
        """Build an :class:`AET2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(AET2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ1(self):
        """Build an :class:`XYZ1SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(XYZ1SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ2(self):
        """Build an :class:`XYZ2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(XYZ2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_AE1(self):
        """Build an :class:`AE1SensitivityMatrix` for the ``sangria`` noise model."""
        self._test_sens_mat(AE1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AE2(self):
        """Build an :class:`AE2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(AE2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ1_stft(self):
        """Build an :class:`XYZ1SensitivityMatrix` on an STFT basis.

        The matrix must carry shape ``(3, 3, NT, NF_active)``, be finite, have
        a computable inverse/determinant, and — because the STFT PSD replicates
        the frequency-domain PSD across time — every time segment must equal the
        FD sensitivity matrix on the same active band.
        """
        NT = 5
        stft = self._make_stft_settings(NT=NT)
        Sn = XYZ1SensitivityMatrix(stft, model=lisa.sangria)

        stft_mat = np.asarray(Sn.sens_mat)
        self.assertEqual(stft_mat.shape, (3, 3, NT, stft.NF_active))
        self.assertFalse(np.any(np.isnan(stft_mat)))
        # Lazy inverse / determinant must compute on the (NT, NF_active) grid.
        self.assertEqual(np.asarray(Sn.invC).shape, (3, 3, NT, stft.NF_active))
        self.assertEqual(np.asarray(Sn.detC).shape, (NT, stft.NF_active))

        # Each STFT time segment equals the FD matrix on the matching band.
        fd_mat = np.asarray(XYZ1SensitivityMatrix(self._make_fd_settings(), model=lisa.sangria).sens_mat)
        for it in range(NT):
            np.testing.assert_array_equal(stft_mat[:, :, it, :], fd_mat)


class SensitivityMatrixArithmeticTest(unittest.TestCase):
    """Add / subtract overloads on :class:`SensitivityMatrixBase` with lazy ``invC`` / ``detC``."""

    def setUp(self):
        # Small FD grid in the LISA band (avoid the f=0 bin where the PSD models
        # return NaN by construction).
        force_backend = "cpu"
        self.N = 256
        self.df = 1e-4
        self.settings = FDSettings(
            N=self.N, df=self.df, min_freq=1e-4, max_freq=2e-2,
            force_backend=force_backend,
        )

    def _make_mat(self):
        """Diagonal AET2 sensitivity matrix on the test settings."""
        return SensitivityMatrix(
            self.settings,
            [A2TDISens, E2TDISens, T2TDISens],
            model=lisa.sangria_v2,
        )

    def test_add_returns_new_instance_with_summed_sens_mat(self):
        """``A + B`` returns a fresh :class:`SensitivityMatrixBase` whose ``sens_mat`` is the elementwise sum."""
        A = self._make_mat()
        B = self._make_mat()
        C = A + B
        self.assertIsInstance(C, SensitivityMatrixBase)
        self.assertIsNot(C, A)
        self.assertIsNot(C, B)
        np.testing.assert_allclose(C.sens_mat, A.sens_mat + B.sens_mat)

    def test_sub_returns_new_instance_with_differenced_sens_mat(self):
        """``A - B`` returns a fresh matrix whose ``sens_mat`` is the elementwise difference."""
        A = self._make_mat()
        B = self._make_mat()
        C = A - B
        np.testing.assert_allclose(C.sens_mat, A.sens_mat - B.sens_mat)

    def test_inv_det_lazy_after_add(self):
        """``__add__`` defers inv/det until first access; reading ``invC`` clears the dirty flag."""
        A = self._make_mat()
        B = self._make_mat()
        C = A + B
        # Fresh from __add__: dirty (no inv/det computed yet).
        self.assertTrue(C._inv_det_dirty)
        # First read triggers compute and clears the flag.
        inv = C.invC
        self.assertFalse(C._inv_det_dirty)
        # Reads after that re-use the cached _invC.
        self.assertIs(C.invC, inv)

    def test_inv_det_correct_after_chained_add(self):
        """``A + B + C`` matches a manual sum, with inv/det computed exactly once at first read."""
        A = self._make_mat()
        B = self._make_mat()
        D = self._make_mat()
        total = A + B + D
        np.testing.assert_allclose(total.sens_mat, A.sens_mat + B.sens_mat + D.sens_mat)
        # Inverse on a diagonal AET PSD is just elementwise reciprocal.
        np.testing.assert_allclose(total.invC, 1.0 / total.sens_mat)
        np.testing.assert_allclose(total.detC, np.prod(total.sens_mat, axis=0))

    def test_dot_add_and_dot_subtract_methods(self):
        """``.add`` / ``.subtract`` are OO equivalents of the operators."""
        A = self._make_mat()
        B = self._make_mat()
        np.testing.assert_allclose((A.add(B)).sens_mat, (A + B).sens_mat)
        np.testing.assert_allclose((A.subtract(B)).sens_mat, (A - B).sens_mat)

    def test_add_array_directly(self):
        """``__add__`` also accepts a raw array of matching shape."""
        A = self._make_mat()
        delta = 1e-42 * np.ones_like(A.sens_mat)
        C = A + delta
        np.testing.assert_allclose(C.sens_mat, A.sens_mat + delta)

    def test_shape_mismatch_raises(self):
        """Combining two matrices with mismatched ``sens_mat`` shapes raises ``ValueError``."""
        A = self._make_mat()
        bad = np.zeros((2,) + A.sens_mat.shape[1:])
        with self.assertRaises(ValueError):
            A + bad

    def test_setitem_marks_dirty(self):
        """In-place ``__setitem__`` flips the dirty flag so the next ``invC`` read recomputes."""
        A = self._make_mat()
        _ = A.invC  # force eager compute
        self.assertFalse(A._inv_det_dirty)
        A[0, 5] = A.sens_mat[0, 5] * 2.0
        self.assertTrue(A._inv_det_dirty)
        # next read recomputes and reflects the update
        np.testing.assert_allclose(A.invC, 1.0 / A.sens_mat)

    def test_explicit_invC_setter_clears_dirty(self):
        """An explicit ``sm.invC = ...`` assignment overrides the lazy recompute path."""
        A = self._make_mat()
        manual = np.full_like(A.sens_mat, 7.0)
        A.invC = manual
        self.assertFalse(A._inv_det_dirty)
        np.testing.assert_array_equal(A.invC, manual)


class CompositeSensitivityMatrixTest(unittest.TestCase):
    """:class:`CompositeSensitivityMatrix` summing instrument + foreground + SGWB via OO addition."""

    def setUp(self):
        self.settings = FDSettings(
            N=512, df=1e-4, min_freq=1e-4, max_freq=3e-2,
            force_backend="cpu",
        )

    def test_composite_sum_matches_component_sum(self):
        """The composite ``sens_mat`` equals the elementwise sum of each component's covariance."""
        instrument = InstrumentNoise(tdi_generation=2, model="sangria_v2")
        foreground = GalacticForeground(
            foreground_params=(1e-44, 5e-4, 1.0, 1.0, 1.0),
            tdi_generation=2,
        )
        sgwb = SGWB(
            sgwb_params=(-12.0, 0.0),
            stochastic_fn=PowerLawSGWB,
            tdi_generation=2,
        )
        composite = CompositeSensitivityMatrix(
            self.settings, [instrument, foreground, sgwb]
        )

        expected = (
            instrument.covariance(self.settings)
            + foreground.covariance(self.settings)
            + sgwb.covariance(self.settings)
        )
        np.testing.assert_allclose(composite.sens_mat, expected)

    def test_composite_inv_det_lazy(self):
        """Just-built composite has dirty inv/det; reads off ``invC`` produce a finite Σ⁻¹."""
        instrument = InstrumentNoise(tdi_generation=2, model="sangria_v2")
        sgwb = SGWB(
            sgwb_params=(-12.0, 0.0),
            stochastic_fn=PowerLawSGWB,
            tdi_generation=2,
        )
        composite = CompositeSensitivityMatrix(self.settings, [instrument, sgwb])
        self.assertTrue(composite._inv_det_dirty)
        invC = composite.invC
        self.assertFalse(composite._inv_det_dirty)
        # Σ Σ⁻¹ = I on the active band.
        eye = np.einsum("ij...,jk...->ik...", composite.sens_mat, invC)
        I = np.broadcast_to(np.eye(3)[..., None], eye.shape)
        np.testing.assert_allclose(eye, I, atol=1e-8)

    def test_composite_update_component_reuses_cache(self):
        """Updating one component recomputes only that contribution; the sum still matches."""
        instrument = InstrumentNoise(tdi_generation=2, model="sangria_v2")
        sgwb = SGWB(
            sgwb_params=(-12.0, 0.0),
            stochastic_fn=PowerLawSGWB,
            tdi_generation=2,
        )
        composite = CompositeSensitivityMatrix(self.settings, [instrument, sgwb])

        # Swap the SGWB amplitude and recompute only that index.
        new_sgwb = SGWB(
            sgwb_params=(-10.0, 0.0),
            stochastic_fn=PowerLawSGWB,
            tdi_generation=2,
        )
        composite.components[1] = new_sgwb
        composite.update_component(1)

        expected = (
            instrument.covariance(self.settings)
            + new_sgwb.covariance(self.settings)
        )
        np.testing.assert_allclose(composite.sens_mat, expected)
