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


if __name__ == "__main__":
    unittest.main()
