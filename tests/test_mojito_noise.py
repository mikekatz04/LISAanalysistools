"""Tests for the mojito NOISE-brick noise model (proper noise).

Covers the tabulated :class:`~lisatools.sensitivity.MojitoNoiseEstimates`
component, the scalar-parameter fit, the ``psd_params=None`` fixed-PSD backend
path, and the stock variants' noise-file resolution. Everything here is
hermetic — the component is fed a hand-built analytic table instead of a real
brick — except the ``RealBrickTest`` cases, which are skipped unless the
standard mojito-light NOISE brick is present on the machine.
"""

import copy
import os
import pickle
import unittest

import numpy as np

from lisatools.detector import DefaultOrbits, LISAModel
from lisatools.domains import FDSettings, WDMSettings
from lisatools.sensitivity import (
    CompositeSensitivityBackend,
    CompositeSensitivityMatrix,
    InstrumentNoise,
    MojitoNoiseEstimates,
    MojitoNoiseSensitivityMatrix,
    X2TDISens,
    XY2TDISens,
)

SOMS_TRUE = 12e-12
SA_TRUE = 4e-15


def _analytic_table(n_days=8, n_freq=800, drift=0.0):
    """Hand-built ``(times, freqs, cov)`` table from the analytic XYZ2 model.

    ``drift`` scales the whole covariance linearly across days (fractional
    peak-to-peak) so time-dependence tests have signal.
    """
    model = LISAModel(SOMS_TRUE**2, SA_TRUE**2, DefaultOrbits(), "table_truth")
    freqs = np.logspace(-5, 0, n_freq)
    xx = X2TDISens.get_Sn(freqs, model=model)
    xy = XY2TDISens.get_Sn(freqs, model=model)
    cov = np.zeros((n_days, n_freq, 3, 3))
    for i in range(3):
        for j in range(3):
            cov[:, :, i, j] = xx if i == j else xy
    times = np.arange(n_days) * 86400.0
    scale = 1.0 + drift * np.linspace(-0.5, 0.5, n_days)
    cov *= scale[:, None, None, None]
    return times, freqs, cov


class ComponentTest(unittest.TestCase):
    def _comp(self, drift=0.0, **kwargs):
        comp = MojitoNoiseEstimates("/nonexistent/noise_brick.h5", **kwargs)
        comp._tab = _analytic_table(drift=drift)
        return comp

    def test_fd_matches_analytic(self):
        comp = self._comp()
        fd = FDSettings(2049, 1.0 / (4096 * 10.0), force_backend="cpu")
        C = np.asarray(comp.covariance(fd))
        self.assertEqual(C.shape, (3, 3) + tuple(fd.basis_shape_active))
        model = LISAModel(SOMS_TRUE**2, SA_TRUE**2, DefaultOrbits(), "truth")
        f = np.asarray(fd.f_arr)
        band = (f > 1e-4) & (f < 2e-2)
        expected = X2TDISens.get_Sn(f[band], model=model)
        ratio = C[0, 0, band] / expected
        self.assertLess(np.abs(ratio - 1).max(), 5e-3)  # interp error only
        # out-of-table bins (f = 0) filled with fill_value
        self.assertEqual(C[0, 0, 0], 0.0)
        # symmetric
        self.assertTrue(np.allclose(C, np.swapaxes(C, 0, 1)))

    def test_wdm_matches_instrument_noise_fold(self):
        comp = self._comp()
        wdm = WDMSettings(
            Nf=128, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3, force_backend="cpu"
        )
        C = np.asarray(comp.covariance(wdm))
        self.assertEqual(C.shape, (3, 3) + tuple(wdm.basis_shape_active))
        ana = np.asarray(
            InstrumentNoise(
                tdi_generation=2,
                model=LISAModel(SOMS_TRUE**2, SA_TRUE**2, DefaultOrbits(), "truth"),
                fill_nans=0.0,
            ).covariance(wdm)
        )
        mask = ana != 0
        ratio = C[mask] / ana[mask]
        self.assertLess(np.abs(np.median(ratio) - 1), 1e-2)
        self.assertLess(np.abs(ratio - 1).max(), 3e-2)

    def test_time_dependence(self):
        # 20% linear drift across the table; a grid spanning the full table
        # must reproduce it along the wavelet time axis.
        comp = self._comp(drift=0.2)
        wdm = WDMSettings(
            Nf=32, Nt=128, dt=160.0, min_freq=3e-4, max_freq=8e-3, force_backend="cpu"
        )  # Tobs = 32*128*160 s ~ 7.6 d
        C = np.asarray(comp.covariance(wdm))
        layer = C[0, 0, C.shape[2] // 2]
        self.assertGreater(layer[-1] / layer[0], 1.1)
        # stationary mode: constant along time
        comp_st = self._comp(drift=0.2, time_dependent=False)
        C_st = np.asarray(comp_st.covariance(wdm))
        self.assertTrue(np.allclose(C_st[..., 0], C_st[..., -1]))

    def test_fit_scalar_params(self):
        comp = self._comp()
        soms, sa = comp.fit_scalar_params()
        self.assertLess(abs(soms / SOMS_TRUE - 1), 1e-6)
        self.assertLess(abs(sa / SA_TRUE - 1), 1e-6)

    def test_pickle_drops_cache(self):
        comp = self._comp()
        clone = pickle.loads(pickle.dumps(copy.deepcopy(comp)))
        self.assertIsNone(clone._tab)
        self.assertEqual(clone.path, comp.path)

    def test_matrix_wrapper_and_inner_product(self):
        from lisatools.diagnostic import inner_product

        comp = self._comp()
        fd = FDSettings(2049, 1.0 / (4096 * 10.0), force_backend="cpu")
        sens = CompositeSensitivityMatrix(fd, [comp])
        ana = CompositeSensitivityMatrix(
            fd,
            [
                InstrumentNoise(
                    tdi_generation=2,
                    model=LISAModel(
                        SOMS_TRUE**2, SA_TRUE**2, DefaultOrbits(), "truth"
                    ),
                    fill_nans=0.0,
                )
            ],
        )
        rng = np.random.default_rng(7)
        sig = rng.normal(size=(3, 2049)) + 1j * rng.normal(size=(3, 2049))
        sig -= sig.mean(axis=0, keepdims=True)  # GW-like: no T-null power
        f = np.asarray(fd.f_arr)
        sig[:, (f < 3e-4) | (f > 8e-3)] = 0.0
        snr_tab = inner_product(sig, sig, psd=sens, basis_settings=fd).real ** 0.5
        snr_ana = inner_product(sig, sig, psd=ana, basis_settings=fd).real ** 0.5
        self.assertLess(abs(snr_tab / snr_ana - 1), 2e-2)


class BackendNoneParamsTest(unittest.TestCase):
    def setUp(self):
        self.wdm = WDMSettings(
            Nf=64, Nt=32, dt=5.0, min_freq=3e-4, max_freq=8e-3, force_backend="cpu"
        )
        self.fixed = InstrumentNoise(
            tdi_generation=2,
            model=LISAModel(SOMS_TRUE**2, SA_TRUE**2, DefaultOrbits(), "fixed"),
            fill_nans=0.0,
        )

    def test_none_params_uses_extras_only(self):
        backend = CompositeSensitivityBackend(
            self.wdm, tdi_generation=2, extra_components=[self.fixed]
        )
        m_none = backend("walker_0", None)
        m_par = backend("walker_0", [SOMS_TRUE, SA_TRUE])
        self.assertEqual(len(m_none.components), 1)
        self.assertEqual(len(m_par.components), 2)
        a, b = np.asarray(m_par.sens_mat), np.asarray(m_none.sens_mat)
        mask = b != 0
        # params + identical extra component == exactly double
        self.assertTrue(np.allclose(a[mask] / b[mask], 2.0))

    def test_none_params_without_components_raises(self):
        backend = CompositeSensitivityBackend(self.wdm, tdi_generation=2)
        with self.assertRaises(ValueError):
            backend("walker_0", None)


_BRICK_DIR = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/data/INSTRUMENT/L1"
)


def _find_brick():
    if not os.path.isdir(_BRICK_DIR):
        return None
    for name in os.listdir(_BRICK_DIR):
        if name.startswith("NOISE_"):
            return os.path.join(_BRICK_DIR, name)
    return None


@unittest.skipUnless(_find_brick(), "mojito-light NOISE brick not on this machine")
class RealBrickTest(unittest.TestCase):
    """Against the real downloaded brick (skipped where it is absent)."""

    def test_fit_matches_stock_levels(self):
        from lisatools.sensitivity import estimate_noise_params_from_file

        soms, sa = estimate_noise_params_from_file(_find_brick())
        # the mojito-light sim runs near-scird levels
        self.assertLess(abs(soms / 15e-12 - 1), 0.05)
        self.assertLess(abs(sa / 3e-15 - 1), 0.05)

    def test_matrix_from_file(self):
        wdm = WDMSettings(
            Nf=64, Nt=32, dt=5.0, min_freq=3e-4, max_freq=8e-3, force_backend="cpu"
        )
        sens = MojitoNoiseSensitivityMatrix(wdm, _find_brick())
        arr = np.asarray(sens.sens_mat)
        self.assertEqual(arr.shape, (3, 3) + tuple(wdm.basis_shape_active))
        self.assertTrue((arr[0, 0][arr[0, 0] != 0] > 0).all())

    def test_gb_no_fg_reads_noise_file(self):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock("gb_no_fg", nwalkers=2)
        gs = fit.make_general_settings()
        self.assertNotEqual(gs.fixed_psd_params, [15e-12, 3e-15])
        self.assertLess(abs(gs.fixed_psd_params[0] / 15e-12 - 1), 0.05)
        self.assertLess(abs(gs.fixed_psd_params[1] / 3e-15 - 1), 0.05)


class StockFallbackTest(unittest.TestCase):
    """Without a brick, every variant must resolve to the stock levels."""

    def test_gb_no_fg_stock_levels(self):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock(
            "gb_no_fg", nwalkers=2, mojito_data_path="/nonexistent_mojito_dir/"
        )
        gs = fit.make_general_settings()
        self.assertEqual(gs.data_mode, "synthetic")
        self.assertEqual(gs.fixed_psd_params, [15e-12, 3e-15])

    def test_noise_only_synthetic_fallback(self):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock(
            "noise_only", nwalkers=2, mojito_data_path="/nonexistent_mojito_dir/"
        )
        gs = fit.make_general_settings()
        self.assertEqual(gs.data_mode, "synthetic")
        self.assertEqual(list(gs.psd_injection), [15e-12, 3e-15])

    def test_full_year_no_double_count(self):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock(
            "full_year_combined", mojito_data_path="/nonexistent_mojito_dir/"
        )
        gs = fit.make_general_settings()
        # instrument noise lives in extra_components only
        self.assertEqual(
            gs.fixed_psd_kwargs, dict(psd_params=None, galfor_params=None)
        )
        names = [type(c).__name__ for c in gs.sensitivity_init_kwargs["extra_components"]]
        self.assertEqual(names.count("InstrumentNoise"), 1)
        self.assertEqual(gs.noise_soms_d, 15e-12)
        self.assertEqual(gs.noise_sa_a, 3e-15)

    def test_all_sources_synthetic_noise(self):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock(
            "all_sources", mojito_data_path="/nonexistent_mojito_dir/"
        )
        gs = fit.make_general_settings()
        self.assertEqual(gs.add_instrument_noise, "synthetic")
        self.assertEqual(list(gs.psd_injection), [15e-12, 3e-15])


if __name__ == "__main__":
    unittest.main()
