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
from lisatools.domains import FDSettings
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

    def test_get_sen(self):
        """Verify :func:`get_sensitivity` returns finite values for the X1 TDI PSD."""
        xp = cp if gpu_available else np

        Sn = get_sensitivity(self._make_fd_settings(), sens_fn="X1TDISens", model=lisa.sangria)

        self.assertFalse(xp.any(xp.isnan(Sn)))

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


class NoiseCovarianceFastPathTest(unittest.TestCase):
    """The optimizations that make a WDM noise MCMC affordable on CPU.

    Each one must be exactly equivalent to the path it replaces -- these are
    algebraic identities and index bookkeeping, not approximations. The noise
    MCMC rebuilds the covariance once per proposal per walker per temperature,
    so a silent regression here is a ~30x slowdown, not a wrong answer; these
    tests pin the equivalence so the fast paths can't quietly stop being taken.
    """

    def _wdm(self, **kwargs):
        from lisatools.domains import WDMSettings

        opts = dict(
            Nf=128, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3,
            force_backend="cpu",
        )
        opts.update(kwargs)
        return WDMSettings(**opts)

    # -- WDMSettings.fold_shift_map ------------------------------------------

    def test_fold_shift_map_matches_inline_construction(self):
        """The cached map reproduces the layer/bin bookkeeping it replaced."""
        s = self._wdm()
        if s.ind_min_f == 0:
            m1 = np.concatenate(
                [np.arange(s.ind_min_f, s.ind_max_f + 1), np.array([s.Nf])]
            )
        else:
            m1 = np.arange(s.ind_min_f, s.ind_max_f + 1)
        k = s.get_shift_map(np.repeat(m1[:, None], s.Nt, axis=-1))
        neg, over = k < 0, k > int(s.N / 2)
        k = k.copy()
        k[neg] = np.abs(k[neg])
        k[over] = s.N - k[over]

        m1_c, k_c, herm_c, uniq_c = s.fold_shift_map()
        np.testing.assert_array_equal(m1_c, m1)
        np.testing.assert_array_equal(k_c, k)
        np.testing.assert_array_equal(herm_c, neg | over)
        np.testing.assert_array_equal(uniq_c, np.unique(k))

    def test_fold_shift_map_cache_invalidates_on_band_change(self):
        """Reassigning the active band must not serve a stale map."""
        s = self._wdm()
        first = s.fold_shift_map()[1]
        self.assertIs(s.fold_shift_map()[1], first)  # cached on repeat call
        s.max_freq = 4e-3
        self.assertNotEqual(s.fold_shift_map()[1].shape, first.shape)

    def test_fold_reads_a_small_fraction_of_the_rfft_grid(self):
        """The whole point: a narrow active band gathers few rFFT bins."""
        s = self._wdm()
        self.assertLess(s.fold_frequency_indices.size, (s.N // 2 + 1) / 4)

    # -- sparse PSD evaluation in get_sensitivity -----------------------------

    def test_sparse_psd_fold_is_bit_identical_to_full_grid(self):
        """Scoring only the gathered bins gives the same folded PSD column."""
        from lisatools.domains import FDSignal
        from lisatools.sensitivity import X2TDISens

        s = self._wdm()
        model = lisa.LISAModel((15e-12) ** 2, (3e-15) ** 2, lisa.DefaultOrbits(), "ref")

        f_full = np.fft.rfftfreq(s.N, s.data_dt)
        df = float(f_full[1] - f_full[0])
        dense = X2TDISens.get_Sn(f_full, model=model)
        folded_dense = np.real(
            FDSignal(dense, FDSettings(f_full.shape[0], df, force_backend="cpu"))
            .wdmtransform(settings=s, is_psd=True)[0]
        )[:, 0]

        sparse = get_sensitivity(s, sens_fn=X2TDISens, model=model, fill_nans=0.0)[:, 0]
        np.testing.assert_array_equal(np.nan_to_num(folded_dense), sparse)

    # -- get_Sn stochastic short-circuit --------------------------------------

    def test_get_Sn_stochastic_short_circuit(self):
        """Skipping the zero stochastic term changes nothing about the result."""
        from lisatools.sensitivity import X2TDISens
        from lisatools.stochastic import HyperbolicTangentGalacticForeground

        f = np.linspace(1e-4, 1e-2, 500)
        model = lisa.LISAModel((15e-12) ** 2, (3e-15) ** 2, lisa.DefaultOrbits(), "ref")

        base = X2TDISens.get_Sn(f, model=model)
        np.testing.assert_array_equal(
            base,
            X2TDISens.get_Sn(
                f, model=model, stochastic_params=(), stochastic_function=None
            ),
        )
        # instrument off AND no stochastic model -> still an array shaped like f
        empty = X2TDISens.get_Sn(f, model=model, include_instrument=False)
        self.assertEqual(np.shape(empty), f.shape)
        self.assertFalse(np.any(empty))
        # a requested stochastic term is still added
        with_stoch = X2TDISens.get_Sn(
            f,
            model=model,
            stochastic_params=(3.3e-44, 2.1e-3, 1.18, 3014.0, 2958.0),
            stochastic_function=HyperbolicTangentGalacticForeground,
        )
        self.assertTrue(np.all(with_stoch > base))

    # -- analytic 3x3 det / inverse -------------------------------------------

    def test_mat3x3_det_inv_matches_numpy(self):
        """The adjugate matches ``np.linalg`` for symmetric and general stacks."""
        from lisatools.sensitivity import _mat3x3_det_inv

        rng = np.random.default_rng(0)
        sym = rng.normal(size=(3, 3, 12, 7))
        sym = sym + np.einsum("ij...->ji...", sym)
        sym += 6.0 * np.eye(3)[:, :, None, None]
        gen = rng.normal(size=(3, 3, 9, 4)) + 5.0 * np.eye(3)[:, :, None, None]

        for C in (sym, gen):
            det, inv = _mat3x3_det_inv(C, np)
            batched = C.transpose(2, 3, 0, 1)
            np.testing.assert_allclose(det, np.linalg.det(batched), rtol=1e-12)
            np.testing.assert_allclose(
                inv.transpose(2, 3, 0, 1), np.linalg.inv(batched), rtol=1e-10
            )

    def test_invC_inverts_the_covariance(self):
        """End to end through the matrix: Σ Σ⁻¹ = I on the active band."""
        instrument = InstrumentNoise(tdi_generation=2, model="sangria_v2")
        settings = FDSettings(N=512, df=1e-4, min_freq=1e-4, max_freq=3e-2,
                              force_backend="cpu")
        composite = CompositeSensitivityMatrix(settings, [instrument])
        eye = np.einsum("ij...,jk...->ik...", composite.sens_mat, composite.invC)
        np.testing.assert_allclose(
            eye, np.broadcast_to(np.eye(3)[..., None], eye.shape), atol=1e-8
        )

    # -- instrument two-basis cache -------------------------------------------

    def test_basis_cache_reproduces_the_direct_build(self):
        """``Soms_d * B_oms + Sa_a * B_acc`` == rebuilding the model from scratch."""
        from lisatools.sensitivity import CompositeSensitivityBackend

        s = self._wdm()
        cached = CompositeSensitivityBackend(s, tdi_generation=2)
        direct = CompositeSensitivityBackend(
            s, tdi_generation=2, cache_instrument_basis=False
        )
        self.assertIsNotNone(cached._instrument_basis_cache)
        self.assertIsNone(direct._instrument_basis_cache)

        galfor = (3.26651613e-44, 2.09278117e-03, 1.18300266e00, 3014.3, 2957.7)
        for psd in ([15e-12, 3e-15], [9.3e-12, 4.1e-15], [22e-12, 1.7e-15]):
            a = np.asarray(cached("w", np.array(psd), galfor_params=galfor).sens_mat)
            b = np.asarray(direct("w", np.array(psd), galfor_params=galfor).sens_mat)
            np.testing.assert_allclose(
                np.nan_to_num(a), np.nan_to_num(b), rtol=1e-13, atol=0.0
            )

    def test_basis_cache_skipped_for_nonlinear_models(self):
        """A model that overrides ``lisanoises`` must not take the linear path."""
        class WeirdModel(lisa.LISAModel):
            def lisanoises(self, f, unit="relative_frequency"):
                out = super().lisanoises(f, unit=unit)
                # squares the level dependence -> no longer linear
                return type(out)(
                    out.isi_oms_noise ** 2, out.rfi_oms_noise, out.tmi_oms_noise,
                    out.tm_noise, out.rfi_backlink_noise, out.tmi_backlink_noise,
                    out.units,
                )

        stock = InstrumentNoise(
            tdi_generation=2,
            model=lisa.LISAModel(1e-22, 1e-29, lisa.DefaultOrbits(), "m"),
            basis_cache={},
        )
        self.assertTrue(stock._linear_in_noise_levels())

        weird = InstrumentNoise(
            tdi_generation=2,
            model=WeirdModel(1e-22, 1e-29, lisa.DefaultOrbits(), "m"),
            basis_cache={},
        )
        self.assertFalse(weird._linear_in_noise_levels())

        # a name-based (stock string) model has no levels to factor either
        self.assertFalse(
            InstrumentNoise(tdi_generation=2, model="sangria_v2",
                            basis_cache={})._linear_in_noise_levels()
        )

    def test_backend_cache_is_not_pickled(self):
        """Sprint deepcopy/pickle rule: derived cache must not ride along."""
        import copy
        import pickle

        from lisatools.sensitivity import CompositeSensitivityBackend

        s = self._wdm()
        be = CompositeSensitivityBackend(s, tdi_generation=2)
        be("w", np.array([15e-12, 3e-15]))
        self.assertEqual(len(be._instrument_basis_cache), 1)

        rt = pickle.loads(pickle.dumps(copy.deepcopy(be)))
        self.assertEqual(len(rt._instrument_basis_cache), 0)
        # and it still builds the right thing after the round trip
        direct = CompositeSensitivityBackend(
            s, tdi_generation=2, cache_instrument_basis=False
        )
        np.testing.assert_allclose(
            np.nan_to_num(np.asarray(rt("w", np.array([9.3e-12, 4.1e-15])).sens_mat)),
            np.nan_to_num(np.asarray(direct("w", np.array([9.3e-12, 4.1e-15])).sens_mat)),
            rtol=1e-13,
        )
