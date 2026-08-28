import copy
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.coarsewdm import (
    CoarseWDMStatistic,
    coarse_q_scan,
    coarse_wdm_log_likelihood,
    coarse_wdm_log_likelihood_batch,
    coarse_wdm_log_likelihood_batch_frequency_terms,
    compute_qeff,
)
from lisatools.diagnostic import residual_full_source_and_noise_likelihood
from lisatools.domains import CoarseWDMSettings, WDMSignal, WDMSettings
from lisatools.detector import DefaultOrbits, LISAModel
from lisatools.sensitivity import (
    GalForTimeModulation,
    LinkDelayTable,
    SensitivityMatrixBase,
    UnequalArmInstrumentNoise,
)


class CoarseWDMTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.fine = WDMSettings(Nf=8, Nt=10, dt=2.0, force_backend="cpu")
        self.data = WDMSignal(
            self.rng.normal(size=(3,) + self.fine.basis_shape_active), self.fine
        )

    def _sensitivity(self, settings, covariance):
        out = SensitivityMatrixBase(settings)
        out.sens_mat = np.asarray(covariance)
        return out

    def _random_fine_covariance(self):
        shape = self.fine.basis_shape_active
        A = self.rng.normal(size=shape + (3, 3))
        covariance = np.einsum("...ik,...jk->...ij", A, A)
        covariance += 0.5 * np.eye(3)
        return covariance.transpose(2, 3, 0, 1)

    def test_q1_matches_fine_nonstationary_likelihood(self):
        fine_cov = self._random_fine_covariance()
        fine_sens = self._sensitivity(self.fine, fine_cov)
        coarse_settings = CoarseWDMSettings.from_fine(self.fine, 1)
        coarse_sens = self._sensitivity(
            coarse_settings, coarse_settings.cell_mean(fine_cov)
        )
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data,
            coarse_settings,
            fiducial_sens_mat_fine=fine_sens,
            use_ws=True,
        )

        fine_logl = residual_full_source_and_noise_likelihood(self.data, fine_sens)
        coarse_logl = coarse_wdm_log_likelihood(stat, coarse_sens)
        np.testing.assert_allclose(coarse_logl, fine_logl, rtol=1e-12, atol=1e-12)

    def test_batched_likelihood_matches_scalar_matrices(self):
        coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        fine_cov = self._random_fine_covariance()
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data,
            coarse,
            fiducial_sens_mat_fine=self._sensitivity(self.fine, fine_cov),
        )
        covariances = []
        scalar = []
        for scale in (0.7, 1.0, 1.4, 2.1):
            covariance = scale * coarse.cell_mean(fine_cov)
            covariances.append(covariance)
            scalar.append(
                coarse_wdm_log_likelihood(
                    stat, self._sensitivity(coarse, covariance)
                )
            )
        batched = coarse_wdm_log_likelihood_batch(
            stat, np.stack(covariances, axis=0)
        )
        np.testing.assert_allclose(batched, scalar, rtol=2e-15, atol=0.0)

    def test_batched_frequency_terms_support_exact_subband_replacement(self):
        coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        fine_cov = self._random_fine_covariance()
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data,
            coarse,
            fiducial_sens_mat_fine=self._sensitivity(self.fine, fine_cov),
        )
        covariance = np.stack(
            [scale * coarse.cell_mean(fine_cov) for scale in (0.8, 1.3)],
            axis=0,
        )
        full_terms = coarse_wdm_log_likelihood_batch_frequency_terms(
            stat, covariance
        )
        np.testing.assert_allclose(
            full_terms.sum(axis=1),
            coarse_wdm_log_likelihood_batch(stat, covariance),
            rtol=2e-15,
            atol=0.0,
        )

        indices = np.array([0, 2, 5])
        subband_terms = coarse_wdm_log_likelihood_batch_frequency_terms(
            stat,
            covariance[:, :, :, indices, :],
            frequency_indices=indices,
        )
        np.testing.assert_allclose(
            subband_terms,
            full_terms[:, indices],
            rtol=2e-15,
            atol=0.0,
        )

    def test_tabulated_modulation_caches_final_coarse_grid(self):
        coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        t = np.linspace(
            float(coarse.fine_t_arr.min()) - 1.0,
            float(coarse.fine_t_arr.max()) + 1.0,
            31,
        )
        table = np.column_stack(
            [t] + [1.0 + 0.01 * i * t for i in range(1, 7)]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/modulation.dat"
            np.savetxt(path, table)
            modulation = GalForTimeModulation(path)
            with mock.patch(
                "lisatools.sensitivity.np.loadtxt", wraps=np.loadtxt
            ) as loadtxt:
                first = modulation.evaluate(coarse)
                second = modulation.evaluate(coarse)
                self.assertIs(first, second)
                self.assertEqual(loadtxt.call_count, 1)
            expected = coarse.cell_mean(modulation(coarse.fine_t_arr))
            np.testing.assert_allclose(first, expected, rtol=0.0, atol=0.0)

            restored = pickle.loads(pickle.dumps(modulation))
            self.assertIsNone(restored._table_cache)
            self.assertEqual(restored._domain_cache, {})

    def test_stationary_identity_with_ragged_cell(self):
        A = self.rng.normal(size=(self.fine.Nf_active, 3, 3))
        layer_cov = np.einsum("mik,mjk->mij", A, A) + 0.5 * np.eye(3)[None]
        fine_cov = np.repeat(
            layer_cov.transpose(1, 2, 0)[:, :, :, None],
            self.fine.Nt_active,
            axis=-1,
        )
        fine_sens = self._sensitivity(self.fine, fine_cov)
        coarse_settings = CoarseWDMSettings.from_fine(self.fine, 4)
        self.assertEqual(coarse_settings.cell_sizes.tolist(), [4, 4, 2])
        coarse_sens = self._sensitivity(
            coarse_settings, coarse_settings.cell_mean(fine_cov)
        )
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data,
            coarse_settings,
            fiducial_sens_mat_fine=fine_sens,
            use_ws=True,
        )
        np.testing.assert_array_equal(
            stat.Qeff[0], np.asarray(coarse_settings.cell_sizes, dtype=float)
        )

        fine_logl = residual_full_source_and_noise_likelihood(self.data, fine_sens)
        coarse_logl = coarse_wdm_log_likelihood(stat, coarse_sens)
        np.testing.assert_allclose(coarse_logl, fine_logl, rtol=1e-12, atol=1e-12)

    def test_q_scan_reports_exact_stationary_likelihood_differences(self):
        A = self.rng.normal(size=(self.fine.Nf_active, 3, 3))
        layer_cov = np.einsum("mik,mjk->mij", A, A) + np.eye(3)[None]
        fine_cov = np.repeat(
            layer_cov.transpose(1, 2, 0)[:, :, :, None],
            self.fine.Nt_active,
            axis=-1,
        )
        comparison_cov = 1.2 * fine_cov
        rows = coarse_q_scan(
            self.fine,
            self._sensitivity(self.fine, fine_cov),
            [1, 4],
            self.data,
            comparison_sens_mat=self._sensitivity(self.fine, comparison_cov),
        )
        self.assertEqual([row["Q"] for row in rows], [1, 4])
        for row in rows:
            self.assertAlmostEqual(row["qeff_ratio_min"], 1.0)
            self.assertAlmostEqual(row["qeff_ratio_median"], 1.0)
            self.assertAlmostEqual(row["fiducial_logl_gap"], 0.0, places=11)
            self.assertAlmostEqual(row["delta_logl_gap"], 0.0, places=11)

    def test_channelwise_qeff_does_not_hide_opposing_drifts(self):
        fine = WDMSettings(Nf=2, Nt=2, dt=1.0, force_backend="cpu")
        coarse = CoarseWDMSettings.from_fine(fine, 2)
        covariance = np.zeros((3, 3, 2, 2))
        covariance[0, 0, :, :] = np.array([1.0, 3.0])
        covariance[1, 1, :, :] = np.array([3.0, 1.0])
        covariance[2, 2, :, :] = np.array([2.0, 2.0])

        qeff, channels = compute_qeff(
            covariance, coarse, use_ws=True, return_channels=True
        )
        np.testing.assert_allclose(channels[:, 0, 0], [1.6, 1.6, 2.0])
        np.testing.assert_allclose(qeff[:, 0], (1.6 + 1.6 + 2.0) / 3.0)
        self.assertLess(float(qeff[0, 0]), 2.0)

    def test_bartlett_and_pickle_roundtrip(self):
        coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data, coarse, use_ws=False
        )
        np.testing.assert_array_equal(stat.Qeff[0], [4.0, 4.0, 2.0])

        restored_settings = pickle.loads(pickle.dumps(copy.deepcopy(coarse)))
        restored_stat = pickle.loads(pickle.dumps(copy.deepcopy(stat)))
        self.assertEqual(restored_settings, coarse)
        self.assertEqual(restored_stat.settings, coarse)
        np.testing.assert_array_equal(restored_stat.P, stat.P)
        np.testing.assert_array_equal(restored_stat.Qeff, stat.Qeff)

    def test_analysis_container_array_supports_mixed_data_psd_geometry(self):
        A = self.rng.normal(size=(self.fine.Nf_active, 3, 3))
        layer_cov = np.einsum("mik,mjk->mij", A, A) + np.eye(3)[None]
        fine_cov = np.repeat(
            layer_cov.transpose(1, 2, 0)[:, :, :, None],
            self.fine.Nt_active,
            axis=-1,
        )
        coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        fine_sens = self._sensitivity(self.fine, fine_cov)
        coarse_sens = self._sensitivity(coarse, coarse.cell_mean(fine_cov))
        stat = CoarseWDMStatistic.from_wdm_signal(
            self.data, coarse, fiducial_sens_mat_fine=fine_sens
        )
        ac = AnalysisContainer(self.data, coarse_sens, coarse_stats=stat)
        aca = AnalysisContainerArray([ac])

        self.assertEqual(aca.end_shape, self.fine.basis_shape_active)
        self.assertEqual(aca.psd_end_shape, coarse.basis_shape_active)
        self.assertEqual(
            aca.linear_psd_arr[0].size,
            9 * int(np.prod(coarse.basis_shape_active)),
        )
        np.testing.assert_allclose(aca.likelihood()[0], ac.likelihood())

    def test_stationary_unequal_arm_covariance_is_exactly_averaged(self):
        fine = WDMSettings(
            Nf=8, Nt=10, dt=2.0, min_freq=0.01, force_backend="cpu"
        )
        coarse = CoarseWDMSettings.from_fine(fine, 4)
        model = LISAModel(
            (15e-12) ** 2, (3e-15) ** 2, DefaultOrbits(), "coarse_ua"
        )
        ltts = np.array([8.31, 8.34, 8.29, 8.32, 8.35, 8.30])
        cache = {}
        fine_cov = UnequalArmInstrumentNoise(
            ltts, model=model, fill_nans=0.0, basis_cache=cache
        ).covariance(fine)
        coarse_cov = UnequalArmInstrumentNoise(
            ltts, model=model, fill_nans=0.0, basis_cache=cache
        ).covariance(coarse)
        np.testing.assert_allclose(
            coarse_cov, coarse.cell_mean(fine_cov), rtol=1e-12, atol=0.0
        )

        # ACA repacking must retain the imaginary cross-spectra. Historically
        # its default float PSD buffer silently discarded them.
        fine_sens = self._sensitivity(fine, fine_cov)
        complex_coarse_cov = coarse_cov.astype(complex)
        phase = 0.01 * np.sqrt(coarse_cov[0, 0] * coarse_cov[1, 1])
        complex_coarse_cov[0, 1] += 1j * phase
        complex_coarse_cov[1, 0] -= 1j * phase
        coarse_sens = self._sensitivity(coarse, complex_coarse_cov)
        data = WDMSignal(self.rng.normal(size=(3,) + fine.basis_shape_active), fine)
        stat = CoarseWDMStatistic.from_wdm_signal(
            data, coarse, fiducial_sens_mat_fine=fine_sens
        )
        inv_before = coarse_sens.invC.copy()
        aca = AnalysisContainerArray(
            [AnalysisContainer(data, coarse_sens, coarse_stats=stat)]
        )
        self.assertTrue(np.issubdtype(aca.linear_psd_arr[0].dtype, np.complexfloating))
        np.testing.assert_allclose(aca[0].sens_mat.invC, inv_before)

    def test_breathing_unequal_arm_averages_covariance_not_delays(self):
        fine = WDMSettings(
            Nf=8, Nt=10, dt=2.0, min_freq=0.01, force_backend="cpu"
        )
        coarse = CoarseWDMSettings.from_fine(fine, 4)
        model = LISAModel(
            (15e-12) ** 2, (3e-15) ** 2, DefaultOrbits(), "coarse_ua_table"
        )
        t = np.arange(0.0, fine.N * fine.data_dt, fine.data_dt)
        base = np.array([8.31, 8.34, 8.29, 8.32, 8.35, 8.30])
        breathing = 1.0 + 0.015 * np.sin(2.0 * np.pi * t / t[-1])
        table = LinkDelayTable(t, breathing[:, None] * base[None, :])
        cache = {}
        fine_cov = UnequalArmInstrumentNoise(
            table, model=model, fill_nans=0.0, basis_cache=cache
        ).covariance(fine)
        coarse_cov = UnequalArmInstrumentNoise(
            table, model=model, fill_nans=0.0, basis_cache=cache
        ).covariance(coarse)
        np.testing.assert_allclose(
            coarse_cov, coarse.cell_mean(fine_cov), rtol=1e-12, atol=0.0
        )

    def test_layer_constant_uses_coarse_time_cell_centers(self):
        fine = WDMSettings(
            Nf=8, Nt=10, dt=2.0, min_freq=0.01, force_backend="cpu"
        )
        coarse = CoarseWDMSettings.from_fine(fine, 4)
        model = LISAModel(
            (15e-12) ** 2, (3e-15) ** 2, DefaultOrbits(), "coarse_center"
        )
        explicit_ltts = np.empty((fine.Nt, 6))
        base = np.array([8.31, 8.34, 8.29, 8.32, 8.35, 8.30])
        for index in range(fine.Nt):
            explicit_ltts[index] = base * (1.0 + 1e-3 * index)

        component = UnequalArmInstrumentNoise(
            explicit_ltts,
            model=model,
            fill_nans=0.0,
            basis_cache={},
            wdm_psd_method="layer_constant",
        )
        covariance = component.covariance(coarse)
        center_ltts = component._coarse_center_ltts(coarse)
        expected = np.empty_like(covariance)
        for cell in range(coarse.Ncoarse):
            column = component._folded_unit_column(fine, center_ltts[cell])
            expected[..., cell] = (
                model.Soms_d * column[0] + model.Sa_a * column[1]
            )
        np.testing.assert_allclose(covariance, expected, rtol=2e-13, atol=0.0)

        qeff, channels = component.coarse_qeff(coarse)
        sizes = coarse.cell_sizes.astype(float)
        np.testing.assert_allclose(
            channels, np.broadcast_to(sizes, channels.shape)
        )
        np.testing.assert_allclose(qeff, np.broadcast_to(sizes, qeff.shape))

    def test_unequal_arm_streamed_qeff_and_persistent_cache(self):
        """Fine diagonals reproduce WS exactly and survive a fresh backend."""
        fine = WDMSettings(
            Nf=8, Nt=10, dt=2.0, min_freq=0.01, force_backend="cpu"
        )
        coarse = CoarseWDMSettings.from_fine(fine, 4)
        model = LISAModel(
            (15e-12) ** 2, (3e-15) ** 2, DefaultOrbits(), "coarse_cache"
        )
        t = np.arange(0.0, fine.N * fine.data_dt, fine.data_dt)
        base = np.array([8.31, 8.34, 8.29, 8.32, 8.35, 8.30])
        breathing = 1.0 + 0.01 * np.sin(2.0 * np.pi * t / t[-1])
        table = LinkDelayTable(t, breathing[:, None] * base[None, :])
        extra = 1e-43 * self.rng.uniform(
            0.5, 1.5, size=(3,) + fine.basis_shape_active
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            first = UnequalArmInstrumentNoise(
                table,
                model=model,
                fill_nans=0.0,
                basis_cache={},
                coarse_cache_dir=cache_dir,
            )
            coarse_cov = first.covariance(coarse)
            qeff, channels = first.coarse_qeff(coarse, extra_diagonal=extra)

            fine_cov = UnequalArmInstrumentNoise(
                table, model=model, fill_nans=0.0, basis_cache={}
            ).covariance(fine)
            total = fine_cov.copy()
            for channel in range(3):
                total[channel, channel] += extra[channel]
            expected_qeff, expected_channels = compute_qeff(
                total, coarse, return_channels=True
            )
            np.testing.assert_allclose(qeff, expected_qeff, rtol=1e-13, atol=0.0)
            np.testing.assert_allclose(
                channels, expected_channels, rtol=1e-13, atol=0.0
            )

            second = UnequalArmInstrumentNoise(
                table,
                model=model,
                fill_nans=0.0,
                basis_cache={},
                coarse_cache_dir=cache_dir,
            )
            with mock.patch.object(
                second,
                "_build_coarse_basis_data",
                side_effect=AssertionError("persistent cache was not loaded"),
            ):
                np.testing.assert_array_equal(second.covariance(coarse), coarse_cov)


class CoarseKnobValidationTest(unittest.TestCase):
    """Shared coarse-knob validation across noise-only and all_sources (T3)."""

    def test_all_source_q_without_mode_rejected(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.all_sources(lite=True, coarse_Q=4)
        with self.assertRaisesRegex(ValueError, "COARSE_GPU_MODE"):
            validate_coarse_settings(fit.general, all_source=True)

    def test_all_source_mode_without_q_rejected(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.all_sources(lite=True, coarse_gpu_mode="delayed_acceptance")
        with self.assertRaisesRegex(ValueError, "COARSE_Q > 1"):
            validate_coarse_settings(fit.general, all_source=True)

    def test_all_source_valid_combo_passes(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.all_sources(
            lite=True, coarse_Q=4, coarse_gpu_mode="delayed_acceptance"
        )
        validate_coarse_settings(fit.general, all_source=True)
        self.assertEqual(fit.general.coarse_gpu_mode, "delayed_acceptance")

    def test_bad_mode_rejected(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.all_sources(lite=True, coarse_Q=4, coarse_gpu_mode="fastmode")
        with self.assertRaisesRegex(ValueError, "coarse_gpu_mode"):
            validate_coarse_settings(fit.general, all_source=True)

    def test_noise_only_mode_rejected(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.noise_only_lite(coarse_gpu_mode="search_approx")
        with self.assertRaisesRegex(ValueError, "all-source"):
            validate_coarse_settings(fit.general, all_source=False)

    def test_noise_only_legacy_gpu_rejection_kept(self):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.noise import validate_coarse_settings

        fit = erebor.noise_only_lite(coarse_Q=4)
        fit.general.gpus = [0]
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            validate_coarse_settings(fit.general, all_source=False)


class CoarseWDMRuntimeTest(unittest.TestCase):
    """Per-walker batched coarse statistics + runtime container (plan-2 T2).

    An all-source run gives every walker its own residual, so the shared
    single-``CoarseWDMStatistic`` model does not apply: these pin the batched
    per-walker builder bitwise against the existing per-signal reference, the
    per-row-P scoring against the scalar loop, and the runtime's
    pickle/deepcopy hygiene (no statistic arrays on the wire).
    """

    NW = 5

    def setUp(self):
        from lisatools.coarsewdm import (  # deferred: TDD, added by plan-2 T2
            CoarseWDMRuntime,
            build_coarse_P_batch,
        )

        self.CoarseWDMRuntime = CoarseWDMRuntime
        self.build = build_coarse_P_batch
        self.rng = np.random.default_rng(777)
        self.fine = WDMSettings(Nf=8, Nt=10, dt=2.0, force_backend="cpu")
        # Q=4 over Nt_active=10 -> ragged: cells of 4, 4, 2
        self.coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        self.res = self.rng.normal(
            size=(self.NW, 3) + tuple(self.fine.basis_shape_active)
        )

    def _sens(self, settings, covariance):
        out = SensitivityMatrixBase(settings)
        out.sens_mat = np.asarray(covariance)
        return out

    def _fine_cov(self):
        shape = tuple(self.fine.basis_shape_active)
        A = self.rng.normal(size=shape + (3, 3))
        cov = np.einsum("...ik,...jk->...ij", A, A) + 0.5 * np.eye(3)
        return cov.transpose(2, 3, 0, 1)

    def _reference_P(self, res=None):
        from lisatools.coarsewdm import _coarse_sample_covariance

        res = self.res if res is None else res
        return np.stack(
            [
                _coarse_sample_covariance(
                    WDMSignal(res[w], self.fine), self.coarse
                )
                for w in range(res.shape[0])
            ]
        )

    def test_build_matches_reference_bitwise(self):
        np.testing.assert_array_equal(
            np.asarray(self.build(self.res, self.coarse)), self._reference_P()
        )

    def test_build_chunking_invariant(self):
        whole = np.asarray(self.build(self.res, self.coarse))
        tiny = np.asarray(
            self.build(self.res, self.coarse, chunk_bytes=1)
        )
        np.testing.assert_array_equal(whole, tiny)

    def _stat_for(self, P_row, qeff, channels):
        return CoarseWDMStatistic(
            P=P_row, Qeff=qeff, settings=self.coarse, Qeff_channels=channels
        )

    def test_per_row_P_scoring_matches_loop(self):
        P = self._reference_P()
        fine_cov = self._fine_cov()
        qeff, channels = compute_qeff(
            self._sens(self.fine, fine_cov),
            self.coarse,
            use_ws=True,
            return_channels=True,
        )
        scales = (0.7, 0.9, 1.1, 1.5, 2.0)
        cov_rows = np.stack(
            [s * self.coarse.cell_mean(fine_cov) for s in scales]
        )
        template = self._stat_for(np.zeros_like(P[0]), qeff, channels)
        got = coarse_wdm_log_likelihood_batch(
            template, cov_rows, per_row_P=P
        )
        expected = [
            coarse_wdm_log_likelihood(
                self._stat_for(P[w], qeff, channels),
                self._sens(self.coarse, cov_rows[w]),
            )
            for w in range(self.NW)
        ]
        np.testing.assert_allclose(got, expected, rtol=2e-15, atol=0.0)

    def test_per_row_degenerate_cell_is_row_local(self):
        P = self._reference_P()
        fine_cov = self._fine_cov()
        qeff, channels = compute_qeff(
            self._sens(self.fine, fine_cov),
            self.coarse,
            use_ws=True,
            return_channels=True,
        )
        cov_rows = np.stack([self.coarse.cell_mean(fine_cov)] * self.NW)
        template = self._stat_for(np.zeros_like(P[0]), qeff, channels)
        clean = coarse_wdm_log_likelihood_batch(template, cov_rows, per_row_P=P)
        P_bad = P.copy()
        P_bad[2, 0, 0, 3, 1] = np.nan
        dirty = coarse_wdm_log_likelihood_batch(
            template, cov_rows, per_row_P=P_bad
        )
        self.assertNotEqual(clean[2], dirty[2])
        np.testing.assert_array_equal(np.delete(clean, 2), np.delete(dirty, 2))
        self.assertTrue(np.all(np.isfinite(dirty)))

    class _ACStub:
        def __init__(self, arr):
            self.data_res_arr = arr

    def test_runtime_refresh_and_rows(self):
        rt = self.CoarseWDMRuntime(
            coarse_settings=self.coarse,
            use_ws=True,
            mode="delayed_acceptance",
        )
        acs = [self._ACStub(self.res[w].copy()) for w in range(self.NW)]
        rt.refresh_P(acs)
        np.testing.assert_array_equal(
            np.asarray(rt.P_rows(np.arange(self.NW))), self._reference_P()
        )
        # a mutated residual must be re-read on the next refresh
        acs[1].data_res_arr *= 2.0
        rt.refresh_P(acs)
        new_res = self.res.copy()
        new_res[1] *= 2.0
        np.testing.assert_array_equal(
            np.asarray(rt.P_rows(np.arange(self.NW))),
            self._reference_P(new_res),
        )

    def test_runtime_pickle_deepcopy_drops_arrays(self):
        rt = self.CoarseWDMRuntime(
            coarse_settings=self.coarse,
            use_ws=True,
            mode="search_approx",
            fiducial_digest="deadbeef",
        )
        rt.refresh_P([self._ACStub(self.res[w]) for w in range(self.NW)])
        clone = pickle.loads(pickle.dumps(copy.deepcopy(rt)))
        self.assertEqual(clone.mode, "search_approx")
        self.assertEqual(clone.fiducial_digest, "deadbeef")
        self.assertIsNone(clone._P)
        # and the original still has its statistics
        self.assertIsNotNone(rt._P)


class PSDMoveCoarseSidecarTest(unittest.TestCase):
    """``compute_coarse_log_like`` plumbing, driven unbound on a stub (T4/T5).

    The coarse math is pinned by :class:`CoarseWDMRuntimeTest` with real
    arrays; a fake coarse backend here isolates the MOVE plumbing: the prior
    mask, the ``walker_inds`` row mapping, the frozen-branch merge, and the
    per-walker statistic routing. (Same unbound-drive pattern as the repack
    fast-path tests.)
    """

    NW = 4
    NT = 3

    class _FakeBackend:
        def __init__(self, base):
            self.base = base
            self.calls = []

        def covariance_from_params(
            self,
            name,
            psd_params,
            galfor_params=None,
            sgwb_params=None,
            fixed_covariances=None,
        ):
            self.calls.append(
                dict(
                    name=name,
                    psd=None if psd_params is None else np.asarray(psd_params),
                    galfor=None
                    if galfor_params is None
                    else np.asarray(galfor_params),
                    fixed=sorted((fixed_covariances or {}).keys()),
                )
            )
            out = (
                float(psd_params[0]) if psd_params is not None else 1.0
            ) * self.base
            if galfor_params is not None:
                out = out + float(galfor_params[0]) * 0.5 * self.base
            for cov in (fixed_covariances or {}).values():
                out = out + cov
            return out

        def component_covariance(self, branch, params, name="x"):
            return float(params[0]) * 0.5 * self.base

    def setUp(self):
        import types as _types

        from lisatools.coarsewdm import CoarseWDMRuntime
        from lisatools.globalfit.moves.psdmove import PSDMove

        self._types = _types
        self.PSDMove = PSDMove
        self.rng = np.random.default_rng(4242)
        self.fine = WDMSettings(Nf=8, Nt=10, dt=2.0, force_backend="cpu")
        self.coarse = CoarseWDMSettings.from_fine(self.fine, 4)
        self.res = self.rng.normal(
            size=(self.NW, 3) + tuple(self.fine.basis_shape_active)
        )
        self.rt = CoarseWDMRuntime(
            coarse_settings=self.coarse, use_ws=False, mode="delayed_acceptance"
        )
        self.rt.refresh_P([self.res[w] for w in range(self.NW)])
        shape = tuple(self.coarse.basis_shape_active)
        base = np.zeros((3, 3) + shape)
        layer = 1.0 + 0.1 * np.arange(shape[0])[:, None]
        for a in range(3):
            base[a, a] = layer * (1.0 + 0.05 * a)
        self.base = base
        self.backend = self._FakeBackend(base)
        self.rt.coarse_backend = self.backend

    def _stub(self, fixed_noise_coords=None, logp=None):
        PSDMove = self.PSDMove
        stub = self._types.SimpleNamespace(
            coarse_runtime=self.rt,
            coarse_sidecar_active=True,
            psd_transform_fn=None,
            galfor_transform_fn=None,
            sgwb_transform_fn=None,
            NOISE_BRANCHES=PSDMove.NOISE_BRANCHES,
            _fixed_noise_coords=dict(fixed_noise_coords or {}),
            _fixed_component_covariances_coarse={},
        )
        want_logp = (
            np.zeros((self.NT, self.NW)) if logp is None else np.asarray(logp)
        )
        stub.compute_log_prior = lambda coords, **kw: want_logp
        for name in (
            "compute_coarse_log_like",
            "_build_coarse_covariance_batch",
            "_merged_noise_rows",
            "_prepare_fixed_component_covariances_coarse",
        ):
            setattr(stub, name, getattr(PSDMove, name).__get__(stub))
        stub._to_physical = PSDMove._to_physical
        return stub

    def _supps(self):
        from eryn.state import BranchSupplemental

        return BranchSupplemental(
            {"walker_inds": np.tile(np.arange(self.NW), (self.NT, 1))},
            base_shape=(self.NT, self.NW),
            copy=True,
        )

    def _coords(self):
        vals = 1.0 + self.rng.uniform(size=(self.NT, self.NW, 1, 2))
        return {"psd": vals}

    def test_scores_match_direct_batch(self):
        stub = self._stub()
        coords = self._coords()
        logl, _ = stub.compute_coarse_log_like(coords, supps=self._supps())
        self.assertEqual(logl.shape, (self.NT, self.NW))
        for t in range(self.NT):
            for w in range(self.NW):
                cov = self.backend.covariance_from_params(
                    "direct", coords["psd"][t, w, 0]
                )
                want = self.rt.coarse_log_like_batch(cov[None], [w])[0]
                self.assertAlmostEqual(logl[t, w] / want, 1.0, places=13)

    def test_prior_mask_rows_skipped(self):
        logp = np.zeros((self.NT, self.NW))
        logp[1, 2] = -np.inf
        stub = self._stub(logp=logp)
        logl, _ = stub.compute_coarse_log_like(
            self._coords(), supps=self._supps()
        )
        self.assertEqual(logl[1, 2], -1e300)
        self.assertTrue(np.all(logl[logp == 0] > -1e299))

    def test_frozen_branch_coords_merged_per_walker(self):
        frozen = {"galfor": 2.0 + np.arange(self.NW, dtype=float)[:, None]}
        stub = self._stub(fixed_noise_coords=frozen)
        stub.compute_coarse_log_like(self._coords(), supps=self._supps())
        for call in self.backend.calls:
            self.assertIsNotNone(call["galfor"])
            w = int(call["name"].rsplit("_", 1)[1])
            self.assertEqual(float(call["galfor"][0]), 2.0 + w)
            self.assertEqual(call["fixed"], [])

    def test_frozen_branch_fixed_covariance_path(self):
        frozen = {"galfor": 2.0 + np.arange(self.NW, dtype=float)[:, None]}
        stub = self._stub(fixed_noise_coords=frozen)
        stub._prepare_fixed_component_covariances_coarse()
        self.assertEqual(sorted(stub._fixed_component_covariances_coarse), list(range(self.NW)))
        self.backend.calls.clear()
        stub.compute_coarse_log_like(self._coords(), supps=self._supps())
        for call in self.backend.calls:
            self.assertIsNone(call["galfor"])
            self.assertEqual(call["fixed"], ["galfor"])


if __name__ == "__main__":
    unittest.main()
