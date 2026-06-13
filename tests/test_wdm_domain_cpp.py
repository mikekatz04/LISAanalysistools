"""Tests for the batched WDM likelihood kernels (2026-06 merge follow-up).

Validates ``WDMDomainWrap.compute_likelihood_terms`` — the WDM counterpart
of the STFT two-pass likelihood kernels in ``cutils/domains.cu`` — against
the Python WDM inner-product path:

* explicit formula checks in all three TDI modes (AET diag, AE diag,
  XYZ full cross-channel), with per-binary template sub-grids placed at
  random offsets inside the active band;
* one genuine end-to-end check through
  :func:`lisatools.diagnostic.inner_product` with :class:`WDMSignal` +
  :class:`SensitivityMatrix` objects;
* the :class:`lisatools.domaincomputation.WDMComputationGroup` Python
  wrapper's physical→integer index conversion and active-band validation.

All comparisons are required to agree to ≤ 1e-12 relative (observed
~1e-15, i.e. floating-point order-of-operations only).
"""

import unittest

import numpy as np

from lisatools.diagnostic import inner_product
from lisatools.domaincomputation import WDMComputationGroup
from lisatools.domains import WDMSettings, WDMSignal
from lisatools.sensitivity import SensitivityMatrix
from lisatools.utils.parallelbase import LISAToolsParallelModule

REL_TOL = 1e-12


class _BackendHolder(LISAToolsParallelModule):
    """Minimal concrete module used to resolve the CPU backend object."""


class TestWDMLikelihoodKernels(unittest.TestCase):
    """C++ WDM batched likelihood terms vs the Python WDM references."""

    @classmethod
    def setUpClass(cls):
        cls.backend = _BackendHolder(force_backend="cpu").backend
        cls.rng = np.random.default_rng(42)

        cls.Nf, cls.Nt, cls.dt = 32, 64, 5.0
        cls.layer_df = 1.0 / (2 * cls.Nf * cls.dt)
        cls.layer_dt = cls.Nf * cls.dt
        cls.ind_min_f, cls.ind_max_f = 3, 20
        cls.ind_min_t, cls.ind_max_t = 4, 50
        cls.Nf_active = cls.ind_max_f - cls.ind_min_f + 1
        cls.Nt_active = cls.ind_max_t - cls.ind_min_t + 1
        cls.num_data, cls.num_noise = 2, 2

        # Per-binary template sub-grids at random offsets in the active band.
        cls.num_binaries = 5
        cls.n_m, cls.n_n = 4, 8
        cls.start_m = cls.rng.integers(
            cls.ind_min_f, cls.ind_max_f - cls.n_m + 2, cls.num_binaries
        )
        cls.start_n = cls.rng.integers(
            cls.ind_min_t, cls.ind_max_t - cls.n_n + 2, cls.num_binaries
        )
        cls.data_index = cls.rng.integers(0, cls.num_data, cls.num_binaries).astype(
            np.int32
        )
        cls.noise_index = cls.rng.integers(0, cls.num_noise, cls.num_binaries).astype(
            np.int32
        )

    def _make_arrays(self, nchannels, cross=False):
        data = self.rng.standard_normal(
            (self.num_data, nchannels, self.Nf_active, self.Nt_active)
        )
        if cross:
            invC = self.rng.uniform(
                0.5,
                2.0,
                (self.num_noise, nchannels, nchannels, self.Nf_active, self.Nt_active),
            )
            invC = 0.5 * (invC + invC.transpose(0, 2, 1, 3, 4))  # symmetric
        else:
            invC = self.rng.uniform(
                0.5, 2.0, (self.num_noise, nchannels, self.Nf_active, self.Nt_active)
            )
        templates = self.rng.standard_normal(
            (self.num_binaries, nchannels, self.n_m, self.n_n)
        )
        return data, invC, templates

    def _make_wrap(self, data, invC, nchannels):
        return self.backend.WDMDomainWrap(
            data.ravel().copy(),
            invC.ravel().copy(),
            self.layer_df,
            self.layer_dt,
            self.Nf,
            self.Nt,
            nchannels,
            self.ind_min_t,
            self.ind_max_t,
            self.ind_min_f,
            self.ind_max_f,
            self.num_data,
            self.num_noise,
        )

    def _run_cpp(self, wrap, templates, tdi_type):
        d_h = np.zeros(self.num_binaries)
        h_h = np.zeros(self.num_binaries)
        wrap.compute_likelihood_terms(
            d_h,
            h_h,
            templates.ravel().copy(),
            self.start_m.astype(np.int32),
            self.start_n.astype(np.int32),
            self.num_binaries,
            self.data_index,
            self.noise_index,
            self.n_m,
            self.n_n,
            self.backend.TDITypeDict[tdi_type],
            False,
        )
        return d_h, h_h

    def _subgrid_slices(self, b):
        sl_m = slice(
            self.start_m[b] - self.ind_min_f,
            self.start_m[b] - self.ind_min_f + self.n_m,
        )
        sl_n = slice(
            self.start_n[b] - self.ind_min_t,
            self.start_n[b] - self.ind_min_t + self.n_n,
        )
        return sl_m, sl_n

    def _check_diag(self, data, invC, templates, d_h, h_h, nchannels):
        """Reference = 4 * sum(d*h*invC) * 0.25 — the diagnostic.inner_product
        WDM convention (differential_component = 0.25)."""
        for b in range(self.num_binaries):
            sl_m, sl_n = self._subgrid_slices(b)
            d_sub = data[self.data_index[b]][:nchannels, sl_m, sl_n]
            w_sub = invC[self.noise_index[b]][:nchannels, sl_m, sl_n]
            dh_ref = 4 * 0.25 * np.sum(d_sub * templates[b] * w_sub)
            hh_ref = 4 * 0.25 * np.sum(templates[b] ** 2 * w_sub)
            self.assertLess(abs(d_h[b] - dh_ref) / abs(dh_ref), REL_TOL)
            self.assertLess(abs(h_h[b] - hh_ref) / abs(hh_ref), REL_TOL)

    def test_aet_diag_vs_formula(self):
        data, invC, templates = self._make_arrays(3)
        wrap = self._make_wrap(data, invC, 3)
        d_h, h_h = self._run_cpp(wrap, templates, "AET")
        self._check_diag(data, invC, templates, d_h, h_h, 3)

    def test_ae_diag_vs_formula(self):
        data, invC, templates = self._make_arrays(2)
        wrap = self._make_wrap(data, invC, 2)
        d_h, h_h = self._run_cpp(wrap, templates, "AE")
        self._check_diag(data, invC, templates, d_h, h_h, 2)

    def test_xyz_cross_vs_formula(self):
        data, invC, templates = self._make_arrays(3, cross=True)
        wrap = self._make_wrap(data, invC, 3)
        d_h, h_h = self._run_cpp(wrap, templates, "XYZ")
        for b in range(self.num_binaries):
            sl_m, sl_n = self._subgrid_slices(b)
            d_sub = data[self.data_index[b]][:, sl_m, sl_n]
            W = invC[self.noise_index[b]][:, :, sl_m, sl_n]
            h_sub = templates[b]
            dh_ref = 4 * 0.25 * np.einsum("imn,ijmn,jmn->", d_sub, W, h_sub)
            hh_ref = 4 * 0.25 * np.einsum("imn,ijmn,jmn->", h_sub, W, h_sub)
            self.assertLess(abs(d_h[b] - dh_ref) / abs(dh_ref), REL_TOL)
            self.assertLess(abs(h_h[b] - hh_ref) / abs(hh_ref), REL_TOL)

    def _make_settings(self):
        # Half-pixel margins so the ceil/int index setters land exactly on
        # the target indices regardless of floating-point rounding.
        return WDMSettings(
            self.Nf,
            self.Nt,
            self.dt,
            min_freq=(self.ind_min_f - 0.5) * self.layer_df,
            max_freq=(self.ind_max_f + 0.5) * self.layer_df,
            min_time=(self.ind_min_t - 0.5) * self.layer_dt,
            max_time=(self.ind_max_t + 0.5) * self.layer_dt,
        )

    def test_vs_diagnostic_inner_product(self):
        """Genuine end-to-end check through the Python WDM likelihood path."""
        data, invC, templates = self._make_arrays(3)
        wrap = self._make_wrap(data, invC, 3)
        d_h, _ = self._run_cpp(wrap, templates, "AET")

        settings = self._make_settings()
        self.assertEqual(settings.ind_min_f, self.ind_min_f)
        self.assertEqual(settings.ind_max_f, self.ind_max_f)
        self.assertEqual(settings.ind_min_t, self.ind_min_t)
        self.assertEqual(settings.ind_max_t, self.ind_max_t)

        for b in range(self.num_binaries):
            sl_m, sl_n = self._subgrid_slices(b)
            sig_d = WDMSignal(data[self.data_index[b]], settings)
            h_full = np.zeros((3, self.Nf_active, self.Nt_active))
            h_full[:, sl_m, sl_n] = templates[b]
            sig_h = WDMSignal(h_full, settings)
            sm = SensitivityMatrix(settings, 1.0 / invC[self.noise_index[b]])
            ref = inner_product(sig_d, sig_h, psd=sm)
            self.assertLess(abs(ref - d_h[b]) / abs(ref), REL_TOL)

    def _make_group(self, data, invC, nchannels):
        """Build a WDMComputationGroup without an AnalysisContainerArray.

        Exercises ``compute_signal_likelihood_terms`` (index conversion,
        validation, C++ dispatch) with the extraction step stubbed out.
        """
        group = WDMComputationGroup.__new__(WDMComputationGroup)
        LISAToolsParallelModule.__init__(group, force_backend="cpu")
        group.tdi_type = "AET"
        group.settings = self._make_settings()
        group.num_channels = nchannels
        group.num_data = self.num_data
        group.num_noise = self.num_noise
        group.data_arr = data.ravel().copy()
        group.invC_arr = invC.ravel().copy()
        group._create_cpp_domain()
        return group

    def test_python_wrapper_index_conversion(self):
        data, invC, templates = self._make_arrays(3)
        group = self._make_group(data, invC, 3)

        start_freqs = self.start_m * self.layer_df
        start_times = self.start_n * self.layer_dt  # t0-relative (t_arr axis)
        d_h, h_h = group.compute_signal_likelihood_terms(
            self.data_index,
            self.noise_index,
            templates,
            start_freqs,
            start_times,
        )
        self.assertEqual(d_h.dtype, np.float64)
        self._check_diag(data, invC, templates, d_h, h_h, 3)

    def test_python_wrapper_band_validation(self):
        data, invC, templates = self._make_arrays(3)
        group = self._make_group(data, invC, 3)

        # frequency below the active band
        bad_freqs = np.full(self.num_binaries, (self.ind_min_f - 1) * self.layer_df)
        with self.assertRaises(ValueError):
            group.compute_signal_likelihood_terms(
                self.data_index,
                self.noise_index,
                templates,
                bad_freqs,
                self.start_n * self.layer_dt,
            )

        # time sub-grid overrunning the active band
        bad_times = np.full(
            self.num_binaries, (self.ind_max_t - self.n_n + 2) * self.layer_dt
        )
        with self.assertRaises(ValueError):
            group.compute_signal_likelihood_terms(
                self.data_index,
                self.noise_index,
                templates,
                self.start_m * self.layer_df,
                bad_times,
            )

        # start_times is required
        with self.assertRaises(ValueError):
            group.compute_signal_likelihood_terms(
                self.data_index,
                self.noise_index,
                templates,
                self.start_m * self.layer_df,
                None,
            )


if __name__ == "__main__":
    unittest.main()
