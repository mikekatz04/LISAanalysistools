"""Tests for the unequal-arm (orbit-informed) instrument-noise component.

Covers :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise` and the six
generated element classes in :mod:`lisatools._unequal_arm_expressions`.

The three properties worth pinning down:

* **Equal-arm limit.** Feeding all six links the same delay must reproduce the
  stock :class:`~lisatools.sensitivity.InstrumentNoise` exactly -- that is what
  makes this a safe drop-in replacement rather than a different model.
* **Agreement with the C++ path.** ``XYZSensitivityMatrix`` in
  ``cutils/PSD.cu`` models the same physics from averaged + differential link
  delays. A compact port of its transfer functions lives here as the reference.
* **Hermiticity and linearity.** The cross-spectra are complex once the arms
  differ, and the covariance stays exactly linear in ``(Soms_d, Sa_a)`` -- the
  property the basis cache relies on.
"""

import copy
import pickle
import unittest
import warnings

import numpy as np

from lisatools import detector as lisa
from lisatools._unequal_arm_fused import unequal_arm_tdi2_unit_covariances
from lisatools.domains import FDSettings, WDMSettings
from lisatools.sensitivity import (
    UNEQUAL_ARM_LINKS,
    InstrumentNoise,
    LinkDelayTable,
    UnequalArmInstrumentNoise,
    UnequalArmXX2TDISens,
)
from lisatools.utils.constants import C_SI, L_SI

L0 = L_SI / C_SI  # equal-arm light travel time, seconds


def _model():
    return lisa.LISAModel((15e-12) ** 2, (3e-15) ** 2, lisa.DefaultOrbits(), "test")


def _unequal_ltts():
    """Realistic delays: ~1% arm-to-arm spread, ~1e-5 relative Sagnac split.

    Ordered as :data:`UNEQUAL_ARM_LINKS` == ``[12, 23, 31, 13, 32, 21]``.
    """
    return np.array(
        [
            L0 * 1.004 + 3.1e-5 * L0,  # 12
            L0 * 0.993 + 0.7e-5 * L0,  # 23
            L0 * 1.002 - 1.9e-5 * L0,  # 31
            L0 * 1.002 + 1.9e-5 * L0,  # 13
            L0 * 0.993 - 0.7e-5 * L0,  # 32
            L0 * 1.004 - 3.1e-5 * L0,  # 21
        ]
    )


# ---------------------------------------------------------------------------
# Reference port of cutils/PSD.cu XYZSensitivityMatrix (unequal-armlength path)
# ---------------------------------------------------------------------------
def _cpp_reference(f, ltts, soms_amp, sa_amp):
    """The C++ covariance elements, from averaged + differential delays.

    Faithful transcription of ``oms_{xx,xy}_unequal_armlength`` /
    ``tm_{xx,xy}_unequal_armlength`` and ``get_noise_tfs``. Note the C++ takes
    the *unsquared* amplitudes and squares them internally, whereas
    ``LISAModel`` stores the squares.
    """
    s = lambda d: np.sin(d * 2 * np.pi * f)  # noqa: E731
    c = lambda d: np.cos(d * 2 * np.pi * f)  # noqa: E731
    c2 = lambda d: np.cos(2.0 * d * 2 * np.pi * f)  # noqa: E731
    dw = lambda d: d * 2 * np.pi * f  # noqa: E731

    def oms_xx(a_ij, a_ik):
        return 8.0 * (s(a_ij) ** 2 + s(a_ik) ** 2) * 4.0 * s(a_ij + a_ik) ** 2

    def oms_xy(a_ij, a_ik, a_jk, d_ij):
        return (
            -8.0
            * (c(a_ij) * s(a_ik) * s(a_jk) * np.exp(-1j * dw(a_ik - a_jk + 0.5 * d_ij)))
            * (4.0 * s(a_ij + a_ik) * s(a_ij + a_jk) * np.exp(-1j * dw(a_ik - a_jk)))
        )

    def tm_xx(a_ij, a_ik):
        return (
            8.0
            * (s(a_ij) ** 2 * (3 + c2(a_ik)) + s(a_ik) ** 2 * (3 + c2(a_ij)))
            * 4.0
            * s(a_ij + a_ik) ** 2
        )

    def tm_xy(a_ij, a_ik, a_jk, d_ij):
        return (
            -32.0
            * (c(a_ij) * s(a_ik) * s(a_jk) * np.exp(-1j * dw(a_ik - a_jk + 0.5 * d_ij)))
            * (4.0 * s(a_ij + a_ik) * s(a_ij + a_jk) * np.exp(-1j * dw(a_ik - a_jk)))
        )

    lt = np.asarray(ltts)
    idx = np.arange(6)
    avg = 0.5 * (lt[idx] + lt[idx[::-1]])
    dlt = lt[idx] - lt[idx[::-1]]

    s_oms = soms_amp**2 * (1.0 + (2.0e-3 / f) ** 4) * (2.0 * np.pi * f / C_SI) ** 2
    s_tm = (
        sa_amp**2
        * (1.0 + (0.4e-3 / f) ** 2)
        * (1.0 + (f / 8e-3) ** 4)
        * (2.0 * np.pi * f) ** -4.0
        * (2.0 * np.pi * f / C_SI) ** 2
    )
    i12, i23, i31, i13, i32, i21 = 0, 1, 2, 3, 4, 5

    def mix(o, t):
        return o * s_oms + t * s_tm

    return {
        "XX": mix(oms_xx(avg[i12], avg[i13]), tm_xx(avg[i12], avg[i13])),
        "YY": mix(oms_xx(avg[i23], avg[i21]), tm_xx(avg[i23], avg[i21])),
        "ZZ": mix(oms_xx(avg[i31], avg[i32]), tm_xx(avg[i31], avg[i32])),
        "XY": mix(
            oms_xy(avg[i12], avg[i13], avg[i23], dlt[i12]),
            tm_xy(avg[i12], avg[i13], avg[i23], dlt[i12]),
        ),
        "YZ": mix(
            oms_xy(avg[i23], avg[i21], avg[i13], dlt[i23]),
            tm_xy(avg[i23], avg[i21], avg[i13], dlt[i23]),
        ),
        # PSD.cu passes delta_d[link_to_index(31)] here; the correct delay
        # difference for the (0, 2) element is delta_d[13] == -delta_d[31].
        # See test_matches_cpp_reference for the size of the discrepancy.
        "XZ": mix(
            oms_xy(avg[i31], avg[i12], avg[i32], dlt[i13]),
            tm_xy(avg[i31], avg[i12], avg[i32], dlt[i13]),
        ),
    }


class UnequalArmNoiseTest(unittest.TestCase):
    """Equal-arm reduction, C++ agreement, Hermiticity, linearity, picklability."""

    def setUp(self):
        self.model = _model()
        self.fd = FDSettings(1024, 2e-4)
        self.ltts = _unequal_ltts()

    # -- equal-arm reduction ------------------------------------------------
    def test_equal_arm_limit_matches_stock_fd(self):
        """All six delays equal => the stock equal-arm covariance, to ~1e-11."""
        ua = UnequalArmInstrumentNoise(
            np.full(6, L0), model=self.model, fill_nans=0.0
        ).covariance(self.fd)
        stock = InstrumentNoise(
            tdi_generation=2, model=self.model, fill_nans=0.0
        ).covariance(self.fd)

        mask = np.isfinite(np.real(stock)) & (np.abs(stock) > 0)
        rel = np.abs(ua - stock)[mask] / np.abs(stock)[mask]
        # The closed forms carry ~300*L0*f of accumulated phase, so the worst
        # bins (near Nyquist) lose a few digits to cancellation.
        self.assertLess(np.median(rel), 1e-13)
        self.assertLess(rel.max(), 1e-9)

    def test_equal_arm_limit_matches_stock_wdm(self):
        """Same reduction in the WDM basis the noise PE actually runs in."""
        wdm = WDMSettings(Nf=32, Nt=32, dt=5.0)
        ua = UnequalArmInstrumentNoise(
            np.full(6, L0), model=self.model, fill_nans=0.0
        ).covariance(wdm)
        stock = InstrumentNoise(
            tdi_generation=2, model=self.model, fill_nans=0.0
        ).covariance(wdm)
        mask = np.isfinite(np.real(stock)) & (np.abs(stock) > 0)
        rel = np.abs(ua - stock)[mask] / np.abs(stock)[mask]
        self.assertLess(rel.max(), 1e-11)
        # the WDM fold keeps only Re[C_ij], so the folded covariance is real
        self.assertFalse(np.iscomplexobj(ua))

    def test_layer_constant_wdm_uses_existing_half_psd_normalization(self):
        """Center evaluation is exactly 0.5*S(f_m), like get_sensitivity."""
        wdm = WDMSettings(Nf=32, Nt=32, dt=5.0, force_backend="cpu")
        ltts = _unequal_ltts()
        got = UnequalArmInstrumentNoise(
            ltts,
            model=self.model,
            fill_nans=0.0,
            basis_cache={},
            wdm_psd_method="layer_constant",
        ).covariance(wdm)

        B_oms, B_acc = unequal_arm_tdi2_unit_covariances(wdm.f_arr, ltts)
        expected_column = 0.5 * np.real(
            self.model.Soms_d * B_oms + self.model.Sa_a * B_acc
        )
        expected_column[np.isnan(expected_column)] = 0.0
        expected = np.repeat(expected_column[..., None], wdm.Nt_active, axis=-1)
        np.testing.assert_allclose(got, expected, rtol=2e-13, atol=0.0)

        # The exact DFT->WDM fold has precisely the same normalization for a
        # locally flat one-sided PSD, rather than merely approaching 1/2.
        flat_fold = wdm.fold_sparse_psd(np.ones(len(wdm.fold_frequency_arr)))
        np.testing.assert_array_equal(flat_fold, np.full(wdm.Nf_active, 0.5))

    # -- agreement with the C++ path ---------------------------------------
    def test_matches_cpp_reference(self):
        """Five of six elements match PSD.cu to ~1e-12; XZ carries a known bug.

        ``get_noise_tfs`` builds the ``(0, 2)`` element with
        ``delta_d[link_to_index(31)]`` where the other two cross terms use the
        delay difference of their own first link. The correct argument is
        ``delta_d[13] == -delta_d[31]``; with that substitution the element
        agrees to machine precision, and as-written it is off by a phase
        ``2 pi f * 2 * (Sagnac split)``.
        """
        f = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
        soms, sa = 15e-12, 3e-15
        ref = _cpp_reference(f, self.ltts, soms, sa)

        from lisatools import _unequal_arm_expressions as ua_expr

        d = {f"d_{ln}": self.ltts[i] for i, ln in enumerate(UNEQUAL_ARM_LINKS)}
        kw = dict(Soms_d=soms**2, Sa_a=sa**2, **d)
        got = {
            "XX": ua_expr.noise_cov_XX(f, **kw),
            "YY": ua_expr.noise_cov_YY(f, **kw),
            "ZZ": ua_expr.noise_cov_ZZ(f, **kw),
            "XY": ua_expr.noise_cov_XY(f, **kw),
            "XZ": ua_expr.noise_cov_XZ(f, **kw),
            "YZ": ua_expr.noise_cov_YZ(f, **kw),
        }
        # 1e-10 is the cancellation floor of the closed forms, not a modelling
        # tolerance: they carry ~300*L0*f of accumulated phase that has to
        # cancel against a common prefactor. Typical agreement is ~1e-14.
        for name, want in ref.items():
            rel = np.abs(got[name] - want) / np.abs(want)
            self.assertLess(rel.max(), 1e-10, f"{name} disagrees with the C++ form")

    # -- structure ----------------------------------------------------------
    def test_hermitian_and_complex_cross_spectra(self):
        """Unequal arms => Hermitian complex covariance, real diagonal."""
        C = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0
        ).covariance(self.fd)
        self.assertTrue(np.iscomplexobj(C))
        np.testing.assert_allclose(C, np.conj(np.transpose(C, (1, 0, 2))), rtol=0, atol=0)
        for i in range(3):
            self.assertEqual(np.abs(np.imag(C[i, i])).max(), 0.0)
        # the whole point: the CSDs pick up an imaginary part the equal-arm
        # model cannot produce
        self.assertGreater(np.abs(np.imag(C[0, 1])).max() / np.abs(C[0, 1]).max(), 1e-3)

    def test_differs_from_equal_arm(self):
        """The correction is large enough to matter for a noise fit."""
        eq = UnequalArmInstrumentNoise(
            np.full(6, L0), model=self.model, fill_nans=0.0
        ).covariance(self.fd)
        ua = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0
        ).covariance(self.fd)
        mask = np.abs(eq[0, 0]) > 0
        frac = np.abs(np.real(ua[0, 0] - eq[0, 0]))[mask] / np.abs(np.real(eq[0, 0]))[mask]
        self.assertGreater(np.median(frac), 1e-3)

    # -- linearity / caching ------------------------------------------------
    def test_linear_in_levels_and_basis_cache(self):
        """Cached two-basis recombination reproduces the direct evaluation."""
        direct = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0, basis_cache=None
        ).covariance(self.fd)
        cached = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0, basis_cache={}
        ).covariance(self.fd)
        np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=0)

    def test_basis_cache_keys_on_ltts(self):
        """Two components sharing a cache must not serve each other's bases."""
        cache = {}
        eq = UnequalArmInstrumentNoise(
            np.full(6, L0), model=self.model, fill_nans=0.0, basis_cache=cache
        ).covariance(self.fd)
        ua = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0, basis_cache=cache
        ).covariance(self.fd)
        self.assertEqual(len(cache), 2)
        ref = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0
        ).covariance(self.fd)
        np.testing.assert_allclose(ua, ref, rtol=1e-12, atol=0)
        self.assertGreater(np.abs(ua - eq).max() / np.abs(eq).max(), 1e-3)

    # -- contract / errors --------------------------------------------------
    def test_tdi1_rejected(self):
        with self.assertRaises(ValueError):
            UnequalArmInstrumentNoise(np.full(6, L0), tdi_generation=1)

    def test_bad_ltts_shape_rejected(self):
        with self.assertRaises(ValueError):
            UnequalArmInstrumentNoise(np.full(5, L0))

    def test_missing_ltts_rejected(self):
        with self.assertRaises(ValueError):
            UnequalArmXX2TDISens.get_Sn(np.array([1e-3]), model=self.model)

    def test_per_epoch_ltts_rejected_in_fd(self):
        """(Nt, 6) needs a time axis; FD must say so rather than silently pick one."""
        comp = UnequalArmInstrumentNoise(
            np.tile(self.ltts, (4, 1)), model=self.model, fill_nans=0.0
        )
        with self.assertRaises(ValueError):
            comp.covariance(self.fd)

    # -- sprint deepcopy / pickle rule --------------------------------------
    def test_deepcopy_pickle_round_trip(self):
        """The component holds only plain arrays, so it survives the settings tree."""
        comp = UnequalArmInstrumentNoise(self.ltts, model=self.model, fill_nans=0.0)
        again = pickle.loads(pickle.dumps(copy.deepcopy(comp)))
        np.testing.assert_array_equal(again.ltts, comp.ltts)
        np.testing.assert_allclose(
            again.covariance(self.fd), comp.covariance(self.fd), rtol=1e-14, atol=0
        )


class LinkDelayTableTest(unittest.TestCase):
    """Per-WDM-slice averaging of a tabulated delay time series."""

    def setUp(self):
        self.model = _model()
        self.wdm = WDMSettings(Nf=32, Nt=16, dt=5.0)
        self.width = float(np.asarray(self.wdm.t_arr)[1] - np.asarray(self.wdm.t_arr)[0])
        # a delay series with a known linear ramp on every link, sampled far
        # finer than the wavelet columns
        self.t = np.arange(0.0, self.wdm.N * self.wdm.data_dt, 5.0)
        ramp = 1.0 + 1e-3 * (self.t / self.t[-1])
        self.ltts = _unequal_ltts()[None, :] * ramp[:, None]
        self.table = LinkDelayTable(self.t, self.ltts, data_t0=0.0)

    def test_slice_average_equals_mean_over_slice(self):
        """Each row is the mean of the samples inside that column's window."""
        got = self.table.slice_averages(self.wdm)
        centres = np.asarray(self.wdm.t_arr, dtype=float)
        self.assertEqual(got.shape, (centres.size, 6))
        for n, c in enumerate(centres):
            sel = (self.t >= c - self.width / 2) & (self.t < c + self.width / 2)
            np.testing.assert_allclose(got[n], self.ltts[sel].mean(axis=0), rtol=1e-12)

    def test_slice_averages_track_the_ramp(self):
        """A rising delay series must give rising per-column delays."""
        got = self.table.slice_averages(self.wdm)
        self.assertTrue(np.all(np.diff(got[:, 0]) > 0))
        self.assertLess(abs(got.mean() / self.table.run_average().mean() - 1), 1e-3)

    def test_coarse_table_interpolates_empty_columns(self):
        """A table coarser than the grid still yields finite delays everywhere."""
        coarse_t = np.array([0.0, self.wdm.N * self.wdm.data_dt])
        coarse = LinkDelayTable(coarse_t, np.tile(_unequal_ltts(), (2, 1)))
        got = coarse.slice_averages(self.wdm)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertEqual(got.shape, (self.wdm.Nt, 6))

    def test_component_uses_slice_averages(self):
        """A table drives the per-column path; a constant table matches (6,)."""
        flat = LinkDelayTable(
            self.t, np.tile(_unequal_ltts(), (self.t.size, 1)), data_t0=0.0
        )
        via_table = UnequalArmInstrumentNoise(
            flat, model=self.model, fill_nans=0.0
        ).covariance(self.wdm)
        via_array = UnequalArmInstrumentNoise(
            _unequal_ltts(), model=self.model, fill_nans=0.0
        ).covariance(self.wdm)
        np.testing.assert_allclose(via_table, via_array, rtol=1e-12, atol=0)

    def test_time_varying_table_changes_columns(self):
        """A breathing table must NOT give the same covariance in every column."""
        C = UnequalArmInstrumentNoise(
            self.table, model=self.model, fill_nans=0.0
        ).covariance(self.wdm)
        col0, coln = C[0, 0, :, 0], C[0, 0, :, -1]
        mask = np.abs(col0) > 0
        self.assertGreater(
            np.median(np.abs(coln - col0)[mask] / np.abs(col0)[mask]), 1e-6
        )

    def test_table_falls_back_to_run_average_without_time_axis(self):
        """FD has no time axis, so the table collapses instead of erroring."""
        fd = FDSettings(256, 1e-3)
        C = UnequalArmInstrumentNoise(
            self.table, model=self.model, fill_nans=0.0
        ).covariance(fd)
        ref = UnequalArmInstrumentNoise(
            self.table.run_average(), model=self.model, fill_nans=0.0
        ).covariance(fd)
        np.testing.assert_allclose(C, ref, rtol=1e-12, atol=0)

    def test_table_deepcopy_pickle_round_trip(self):
        comp = UnequalArmInstrumentNoise(self.table, model=self.model, fill_nans=0.0)
        again = pickle.loads(pickle.dumps(copy.deepcopy(comp)))
        np.testing.assert_allclose(
            again.covariance(self.wdm), comp.covariance(self.wdm), rtol=1e-14, atol=0
        )

    def test_table_basis_cache_distinguishes_tables(self):
        """Two different tables sharing one cache must not collide."""
        cache = {}
        flat = LinkDelayTable(
            self.t, np.tile(_unequal_ltts(), (self.t.size, 1)), data_t0=0.0
        )
        a = UnequalArmInstrumentNoise(
            flat, model=self.model, fill_nans=0.0, basis_cache=cache
        ).covariance(self.wdm)
        b = UnequalArmInstrumentNoise(
            self.table, model=self.model, fill_nans=0.0, basis_cache=cache
        ).covariance(self.wdm)
        self.assertEqual(len(cache), 2)
        self.assertGreater(np.abs(a - b).max() / np.abs(a).max(), 1e-9)

    def test_layer_calibrated_matches_fold(self):
        """The calibrated layer-center path must track the exact fold.

        ``layer_constant`` samples each unit basis at the layer center instead
        of averaging it over the wavelet's frequency response, so it runs
        systematically low. ``layer_calibrated`` divides that error out with a
        single exact fold; both ingredients are parameter-independent, so the
        residual is only the (second-order) delay dependence of the ratio.
        """
        # A production-like band: the calibration is a curvature correction,
        # so it is only meaningful where the TDI transfer varies slowly across
        # one layer (see test_layer_calibration_warns_out_of_band).
        wdm = WDMSettings(Nf=256, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3)
        got = {
            method: UnequalArmInstrumentNoise(
                self.table, model=self.model, fill_nans=0.0,
                wdm_psd_method=method, basis_cache={},
            ).covariance(wdm)
            for method in ("fold", "layer_constant", "layer_calibrated")
        }
        scale = np.abs(got["fold"]).max()
        uncorrected = np.abs(got["layer_constant"] - got["fold"]).max() / scale
        corrected = np.abs(got["layer_calibrated"] - got["fold"]).max() / scale
        # The uncorrected error must be real, or this test proves nothing.
        self.assertGreater(uncorrected, 1e-4)
        self.assertLess(corrected, 1e-5)
        self.assertLess(corrected, uncorrected / 100.0)

    def test_layer_calibration_warns_out_of_band(self):
        """One reference epoch is not enough near Nyquist -- say so.

        Above roughly the first TDI transfer null the transfer turns over
        inside a single WDM layer, the correction stops being delay-independent,
        and calibrating at one epoch no longer helps. That must warn rather
        than quietly return a covariance no better than layer_constant.
        """
        wide = WDMSettings(Nf=256, Nt=64, dt=5.0)      # full band to Nyquist
        with self.assertWarns(RuntimeWarning):
            UnequalArmInstrumentNoise(
                self.table, model=self.model, fill_nans=0.0,
                wdm_psd_method="layer_calibrated", basis_cache={},
            ).covariance(wide)

    def test_layer_calibration_quiet_in_band(self):
        """...and stays quiet where it is valid."""
        wdm = WDMSettings(Nf=256, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            UnequalArmInstrumentNoise(
                self.table, model=self.model, fill_nans=0.0,
                wdm_psd_method="layer_calibrated", basis_cache={},
            ).covariance(wdm)
        # Only ours; numpy raises unrelated invalid-value warnings in here.
        mine = [w for w in caught if "layer_calibrated" in str(w.message)]
        self.assertEqual(mine, [], f"unexpected calibration warning: {mine}")

    def test_layer_calibration_cached_and_shared(self):
        """The calibration fold is paid once, not once per component.

        The backend rebuilds the component per proposal and hands it a shared
        ``basis_cache``; if the calibration did not live there it would cost an
        extra exact fold on every likelihood call.
        """
        cache = {}
        first = UnequalArmInstrumentNoise(
            self.table, model=self.model, fill_nans=0.0,
            wdm_psd_method="layer_calibrated", basis_cache=cache,
        ).covariance(self.wdm)
        n_after_first = len(cache)
        second = UnequalArmInstrumentNoise(
            self.table, model=self.model, fill_nans=0.0,
            wdm_psd_method="layer_calibrated", basis_cache=cache,
        ).covariance(self.wdm)
        self.assertEqual(len(cache), n_after_first)
        np.testing.assert_allclose(second, first, rtol=1e-14, atol=0)

    def test_unknown_wdm_psd_method_rejected(self):
        with self.assertRaises(ValueError):
            UnequalArmInstrumentNoise(
                self.table, model=self.model, wdm_psd_method="layer_calibrated_typo"
            )

    def test_bad_table_shape_rejected(self):
        with self.assertRaises(ValueError):
            LinkDelayTable(np.arange(5.0), np.zeros((4, 6)))
        with self.assertRaises(ValueError):
            LinkDelayTable(np.array([1.0, 0.0]), np.zeros((2, 6)))


def _has_cupy():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class UnequalArmGPUBoundaryTest(unittest.TestCase):
    """Host/device boundary of the fused WDM unit bases (GPU unblock, 2026-08).

    The fused transfer evaluation is NumPy; the fold and the final bases live
    on the settings backend. These tests pin the CPU arithmetic bitwise against
    an in-test naive reference, prove the batched fold path is bit-identical to
    the per-column path, and record the current device in the basis-cache key
    (the multi-GPU cache-poisoning fix; see noise-dev-merge-handoff.md §3.3).
    """

    def setUp(self):
        self.model = _model()
        self.wdm = WDMSettings(Nf=32, Nt=16, dt=5.0, force_backend="cpu")
        self.ltts = _unequal_ltts()
        # (Nt, 6) breathing table: static delays + a small per-epoch wobble
        wob = 1e-4 * L0 * np.sin(
            2 * np.pi * np.arange(self.wdm.Nt)[:, None] / self.wdm.Nt
            + np.arange(6)[None, :]
        )
        self.ltts_t = self.ltts[None, :] + wob

    def _naive_fold_bases(self, wdm, ltts_2d, fill_nans=0.0):
        """Reference: one exact sparse fold per WDM time column."""
        f_active = np.asarray(wdm.fold_frequency_arr, dtype=float)
        cols = []
        for g in range(wdm.ind_min_t, wdm.ind_max_t + 1):
            bases = unequal_arm_tdi2_unit_covariances(f_active, ltts_2d[g])
            stacked = np.stack(bases, axis=0)
            stacked[np.isnan(stacked)] = fill_nans
            cols.append(np.asarray(wdm.fold_sparse_psd(stacked)))
        return np.stack(cols, axis=-1)

    def test_fold_bases_match_naive_reference_time_resolved(self):
        comp = UnequalArmInstrumentNoise(
            self.ltts_t, model=self.model, fill_nans=0.0, basis_cache={}
        )
        B_oms, B_acc = comp._bases(self.wdm)
        ref = self._naive_fold_bases(self.wdm, self.ltts_t)
        np.testing.assert_array_equal(np.asarray(B_oms), ref[0])
        np.testing.assert_array_equal(np.asarray(B_acc), ref[1])

    def test_fold_bases_match_naive_reference_static(self):
        comp = UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0, basis_cache={}
        )
        B_oms, B_acc = comp._bases(self.wdm)
        one = self._naive_fold_bases(
            self.wdm, np.tile(self.ltts, (self.wdm.Nt, 1))
        )
        np.testing.assert_array_equal(np.asarray(B_oms), one[0])
        np.testing.assert_array_equal(np.asarray(B_acc), one[1])

    def test_batched_fold_matches_per_column(self):
        """The chunked batch fold is bitwise the per-column fold on CPU."""
        comp = UnequalArmInstrumentNoise(
            self.ltts_t, model=self.model, fill_nans=0.0, basis_cache={}
        )
        rows = [
            self.ltts_t[g]
            for g in range(self.wdm.ind_min_t, self.wdm.ind_max_t + 1)
        ]
        batched = comp._folded_unit_columns_batched(self.wdm, rows)
        ref = self._naive_fold_bases(self.wdm, self.ltts_t)
        np.testing.assert_array_equal(np.asarray(batched), ref)

    def test_batched_fold_chunking_is_invariant(self):
        comp = UnequalArmInstrumentNoise(
            self.ltts_t, model=self.model, fill_nans=0.0, basis_cache={}
        )
        rows = [
            self.ltts_t[g]
            for g in range(self.wdm.ind_min_t, self.wdm.ind_max_t + 1)
        ]
        whole = comp._folded_unit_columns_batched(self.wdm, rows)
        tiny = comp._folded_unit_columns_batched(
            self.wdm, rows, chunk_bytes=1
        )
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(tiny))

    def test_basis_cache_key_records_device(self):
        """A shared settings object must warm per-device bases (handoff §3.3)."""
        from lisatools.utils.device import current_device

        cache = {}
        UnequalArmInstrumentNoise(
            self.ltts, model=self.model, fill_nans=0.0, basis_cache=cache
        ).covariance(self.wdm)
        (key,) = cache.keys()
        self.assertIn(current_device(self.wdm.xp), key)

    def test_unit_bases_return_settings_backend_arrays(self):
        comp = UnequalArmInstrumentNoise(
            self.ltts_t, model=self.model, fill_nans=0.0, basis_cache={}
        )
        B_oms, B_acc = comp._bases(self.wdm)
        self.assertIsInstance(B_oms, np.ndarray)
        self.assertIsInstance(B_acc, np.ndarray)

    @unittest.skipUnless(_has_cupy(), "needs a CUDA device + cupy")
    def test_wdm_bases_on_gpu_match_cpu(self):
        """GPU bases: device-resident, and equal to CPU to fp round-off."""
        import cupy

        for method in ("fold", "layer_constant", "layer_calibrated"):
            with self.subTest(method=method):
                cpu = UnequalArmInstrumentNoise(
                    self.ltts_t,
                    model=self.model,
                    fill_nans=0.0,
                    basis_cache={},
                    wdm_psd_method=method,
                ).covariance(self.wdm)
                wdm_gpu = WDMSettings(Nf=32, Nt=16, dt=5.0, force_backend="cuda")
                gpu = UnequalArmInstrumentNoise(
                    self.ltts_t,
                    model=self.model,
                    fill_nans=0.0,
                    basis_cache={},
                    wdm_psd_method=method,
                ).covariance(wdm_gpu)
                self.assertIsInstance(gpu, cupy.ndarray)
                np.testing.assert_allclose(
                    cupy.asnumpy(gpu), np.asarray(cpu), rtol=1e-13, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
