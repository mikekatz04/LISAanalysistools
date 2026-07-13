"""Alignment tests: stock per-source waveform classes vs the all_sources defaults.

The stock per-source waveform classes in ``lisatools.sources`` build their
waveforms the *same* way the stock ``all_sources`` global fit builds its default
waveforms (2nd-generation ``XYZ`` TDI everywhere). These tests lock that in.

FAST (always run): assert the sources-side waveform defaults equal the erebor
reference defaults — ``Source{MBH,EMRI,SOBBH}Settings`` plus the
``get_mbh_phenom_wave_gen`` / ``get_emri_response_wrapper`` /
``get_sobbh_tdionfly_gen`` signatures. The check is by signature/constant
introspection (no global-fit build, no heavy waveform construction). Because the
commit that aligned the sources classes could not edit the erebor builders, this
FAST test is *the* guard that keeps the two copies in sync: if either side ever
drifts, it fails.

SLOW (``LAT_SLOW_TESTS=1``): build ``all_sources_lite`` once and assert each
aligned stock class, given only the run essentials, reproduces the branch
``signal_gen`` output to machine precision.
"""

import inspect
import os
import unittest

import numpy as np


def _defaults(fn) -> dict:
    """Resolved default values of a callable's parameters."""
    return {
        k: v.default
        for k, v in inspect.signature(fn).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


# ============================================================
# FAST: default-value alignment (always runs, no builds)
# ============================================================
class FastWaveformDefaultAlignmentTest(unittest.TestCase):
    def test_general_default_tdi_is_gen2_xyz(self):
        """The all_sources run TDI convention is 2nd-generation XYZ."""
        from lisatools.globalfit.stock import erebor

        gs = erebor.all_sources_lite().general
        self.assertEqual(gs.tdi_gen_str, "2nd generation")
        self.assertEqual(gs.tdi_chan, "XYZ")

    def test_emri_defaults_match_all_sources(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceEMRISettings,
        )
        from lisatools.sources.emri.response import get_emri_response_wrapper
        from lisatools.sources.emri.waveform import (
            EMRI_STOCK_RESPONSE_ORDER,
            EMRI_STOCK_TDI_CHAN,
            EMRI_STOCK_TDI_GENERATION,
            EMRITDIWaveform,
        )

        wrap = _defaults(get_emri_response_wrapper)
        # sources-side constants == erebor reference defaults
        self.assertEqual(
            EMRI_STOCK_RESPONSE_ORDER, SourceEMRISettings().response_order
        )
        self.assertEqual(EMRI_STOCK_RESPONSE_ORDER, wrap["order"])
        self.assertEqual(EMRI_STOCK_TDI_CHAN, wrap["tdi_chan"])
        self.assertEqual(EMRI_STOCK_TDI_CHAN, "XYZ")
        self.assertEqual(EMRI_STOCK_TDI_GENERATION, "2nd generation")
        # the aligned class actually uses them (XYZ, gen-2, special frame ==
        # the all_sources get_emri_response_wrapper default)
        cls = _defaults(EMRITDIWaveform.__init__)
        self.assertEqual(cls["order"], EMRI_STOCK_RESPONSE_ORDER)
        self.assertEqual(cls["tdi_chan"], "XYZ")
        self.assertEqual(cls["tdi_gen"], "2nd generation")
        self.assertEqual(cls["special_frame"], wrap["special_frame"])

    def test_mbh_defaults_match_all_sources(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceMBHSettings,
        )
        from lisatools.globalfit.stock.erebor.wrappers import (
            get_mbh_phenom_wave_gen,
        )
        from lisatools.sources.bbh.waveform import (
            MBH_PHENOM_DEFAULT_BUFFER_TIME,
            MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE,
            MBH_PHENOM_DEFAULT_FREQ_MAX,
            MBH_PHENOM_DEFAULT_FREQ_MIN,
            MBH_PHENOM_DEFAULT_RESPONSE_ORDER,
            MBH_PHENOM_DEFAULT_START_FREQ,
            MBH_PHENOM_DEFAULT_TDI_CHANNELS,
            MBH_PHENOM_DEFAULT_TDI_GENERATION,
            MBH_PHENOM_DEFAULT_TOBS,
            MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS,
            PhenomTHMTDIWaveform,
        )

        m = SourceMBHSettings()
        g = _defaults(get_mbh_phenom_wave_gen)
        wk = MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS

        # constants == SourceMBHSettings
        self.assertEqual(tuple(wk["higher_modes"]), tuple(m.higher_modes))
        self.assertEqual(wk["atol"], m.phenom_tol)
        self.assertEqual(wk["rtol"], m.phenom_tol)
        self.assertEqual(MBH_PHENOM_DEFAULT_START_FREQ, m.start_freq)
        self.assertEqual(MBH_PHENOM_DEFAULT_RESPONSE_ORDER, m.response_order)
        self.assertEqual(MBH_PHENOM_DEFAULT_BUFFER_TIME, m.buffer_time)
        self.assertEqual(MBH_PHENOM_DEFAULT_TOBS, m.waveform_duration)

        # constants == get_mbh_phenom_wave_gen signature defaults
        self.assertEqual(MBH_PHENOM_DEFAULT_RESPONSE_ORDER, g["response_order"])
        self.assertEqual(MBH_PHENOM_DEFAULT_BUFFER_TIME, g["buffer_time"])
        self.assertEqual(MBH_PHENOM_DEFAULT_START_FREQ, g["start_freq"])
        self.assertEqual(MBH_PHENOM_DEFAULT_FREQ_MIN, g["min_freq"])
        self.assertEqual(MBH_PHENOM_DEFAULT_FREQ_MAX, g["max_freq"])
        self.assertEqual(MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE, g["fft_batch_size"])
        self.assertEqual(MBH_PHENOM_DEFAULT_TDI_GENERATION, g["tdi_gen_str"])
        self.assertEqual(MBH_PHENOM_DEFAULT_TDI_CHANNELS, g["tdi_chan"])
        self.assertEqual(tuple(wk["higher_modes"]), tuple(g["higher_modes"]))
        self.assertEqual(MBH_PHENOM_DEFAULT_TOBS, g["waveform_duration"])

        # the fixed booleans get_mbh_phenom_wave_gen builds inline
        self.assertTrue(wk["include_negative_modes"])
        self.assertTrue(wk["t_low_fit"])
        self.assertFalse(wk["coarse_grain"])  # pyResponseTDI needs equispaced grid
        self.assertEqual(MBH_PHENOM_DEFAULT_TDI_CHANNELS, "XYZ")

        # the aligned class actually uses the constants
        c = _defaults(PhenomTHMTDIWaveform.__init__)
        self.assertIsNone(c["waveform_kwargs"])  # None -> copy of default in __init__
        self.assertEqual(c["Tobs"], MBH_PHENOM_DEFAULT_TOBS)
        self.assertEqual(c["start_freq"], MBH_PHENOM_DEFAULT_START_FREQ)
        self.assertEqual(c["order"], MBH_PHENOM_DEFAULT_RESPONSE_ORDER)
        self.assertEqual(c["buffer_time"], MBH_PHENOM_DEFAULT_BUFFER_TIME)
        self.assertEqual(c["freq_min"], MBH_PHENOM_DEFAULT_FREQ_MIN)
        self.assertEqual(c["freq_max"], MBH_PHENOM_DEFAULT_FREQ_MAX)
        self.assertEqual(c["fft_batch_size"], MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE)
        self.assertEqual(c["tdi_generation"], "2nd generation")
        self.assertEqual(c["tdi_channels"], "XYZ")

    def test_sobbh_defaults_match_all_sources(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceSOBBHSettings,
        )
        from lisatools.sources.sobbh.response import (
            SOBBH_STOCK_BUFFER_TIME,
            SOBBH_STOCK_N_GRID,
            SOBBH_STOCK_RESPONSE_ORDER,
            build_sobbh_stock_waveform,
            get_sobbh_tdionfly_gen,
        )

        s = SourceSOBBHSettings()
        g = _defaults(get_sobbh_tdionfly_gen)
        b = _defaults(build_sobbh_stock_waveform)

        self.assertTrue(s.use_tdionfly)  # stock SOBBH default = TDI-on-the-fly
        self.assertEqual(SOBBH_STOCK_N_GRID, s.n_grid)
        self.assertEqual(SOBBH_STOCK_BUFFER_TIME, s.buffer_time)
        self.assertEqual(SOBBH_STOCK_RESPONSE_ORDER, s.response_order)
        self.assertEqual(SOBBH_STOCK_N_GRID, g["n_grid"])
        self.assertEqual(SOBBH_STOCK_BUFFER_TIME, g["buffer_time"])
        self.assertEqual(b["n_grid"], SOBBH_STOCK_N_GRID)
        self.assertEqual(b["buffer_time"], SOBBH_STOCK_BUFFER_TIME)

    def test_gb_xyz_is_primary_stock_default(self):
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticGBProcessingStep,
        )
        from lisatools.sources.gb import GBXYZTDIWaveform as GBExported
        from lisatools.sources.gb.waveform import GBXYZTDIWaveform

        self.assertIs(GBExported, GBXYZTDIWaveform)
        c = _defaults(GBXYZTDIWaveform.__init__)
        # XYZ TDI-2 is the primary/aligned default; no new AET default.
        self.assertEqual(c["tdi_chan"], "XYZ")
        self.assertTrue(c["use_tdi2"])
        self.assertEqual(c["oversample"], 1)
        # matches the all_sources GB injection/template convention
        s = _defaults(SyntheticGBProcessingStep.__init__)
        self.assertEqual(s["tdi_chan"], "XYZ")
        self.assertTrue(s["use_tdi2"])
        self.assertEqual(s["oversample"], 1)

    def test_no_new_aet_defaults_in_aligned_classes(self):
        """Every aligned stock waveform defaults to the fit's XYZ convention."""
        from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
        from lisatools.sources.emri.waveform import EMRITDIWaveform
        from lisatools.sources.gb.waveform import GBXYZTDIWaveform

        self.assertEqual(_defaults(EMRITDIWaveform.__init__)["tdi_chan"], "XYZ")
        self.assertEqual(
            _defaults(PhenomTHMTDIWaveform.__init__)["tdi_channels"], "XYZ"
        )
        self.assertEqual(_defaults(GBXYZTDIWaveform.__init__)["tdi_chan"], "XYZ")


# ============================================================
# SLOW: end-to-end numeric equivalence to the branch signal_gen
# ============================================================
@unittest.skipUnless(
    os.environ.get("LAT_SLOW_TESTS") == "1",
    "set LAT_SLOW_TESTS=1 to run the slow all_sources_lite build + equivalence",
)
class SlowWaveformEquivalenceTest(unittest.TestCase):
    curr = None

    @classmethod
    def setUpClass(cls):
        from lisatools.globalfit.stock import erebor

        # Nonexistent mojito path -> synthetic-fallback build (no external data).
        fit = erebor.all_sources_lite(mojito_data_path="/nonexistent/")
        cls.curr = fit.build()

    # -- helpers ------------------------------------------------------------
    def _essentials(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            find_source_cfg,
        )

        return self.curr.general_info, find_source_cfg(self.curr)

    @staticmethod
    def _arr(sig):
        from lisatools.utils.utility import asnumpy

        return np.asarray(asnumpy(sig.arr))

    def _assert_equiv(self, ref, out, name):
        a = self._arr(ref)
        b = self._arr(out)
        self.assertEqual(a.shape, b.shape, f"{name}: shape mismatch")
        scale = float(np.max(np.abs(a))) or 1.0
        max_abs = float(np.max(np.abs(a - b)))
        rel = max_abs / scale
        print(f"[{name}] max_abs_diff={max_abs:.3e} rel_diff={rel:.3e} scale={scale:.3e}")
        self.assertLess(rel, 1e-10, f"{name}: rel diff {rel:.3e} exceeds 1e-10")

    def _row_and_params(self, branch):
        info = self.curr.source_info[branch]
        row = np.asarray(info.injection[0], dtype=float)
        params_in = info.transform.both_transforms(row)
        return info, row, params_in

    # -- per-branch equivalence --------------------------------------------
    def test_mbh_equivalence(self):
        from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

        gi, cfg = self._essentials()
        self.assertFalse(cfg["mbh_use_tdionfly"], "expected the legacy phentax path")
        info, row, params_in = self._row_and_params("mbh")
        ref = info.signal_gen(*row)
        orbits = gi.gpu_orbits if getattr(gi, "gpus", None) is not None else gi.orbits
        aligned = PhenomTHMTDIWaveform(
            waveform_t0=cfg["mbh_waveform_t0"],
            data_td_settings=gi.data_td_settings,
            orbits=orbits,
            output_domain_settings=gi.domain_settings,
            tukey_alpha=gi.window_alpha,
            sampling_frequency=1.0 / gi.dt,
            force_backend=gi.force_backend,
        )
        out = aligned.get_signals_for_residuals(*params_in)
        self._assert_equiv(ref, out, "mbh")

    def test_emri_equivalence(self):
        from lisatools.sources.emri.response import EMRIWaveWrap
        from lisatools.sources.emri.waveform import EMRITDIWaveform
        from lisatools.utils.constants import YRSID_SI

        gi, cfg = self._essentials()
        info, row, params_in = self._row_and_params("emri")
        ref = info.signal_gen(*row)
        out_N = int(round(gi.Tobs / gi.dt))
        emri = EMRITDIWaveform(
            T=out_N * gi.dt / YRSID_SI,
            dt=gi.dt,
            t0=gi.data_t0,
            order=cfg["emri_response_order"],
            tdi_gen=cfg["tdi_gen_str"],
            tdi_chan=cfg["tdi_chan"],
            orbits=gi.orbits,
            force_backend=gi.force_backend,
            special_frame=True,
        )
        wrap = EMRIWaveWrap(
            emri.response,
            gi.data_td_settings,
            gi.domain_settings,
            td_window=None,
            nchannels=cfg["nchannels"],
            offset_int=0,
        )
        out = wrap(*params_in)
        self._assert_equiv(ref, out, "emri")

    def test_sobbh_equivalence(self):
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.sobbh.response import build_sobbh_stock_waveform

        gi, cfg = self._essentials()
        self.assertTrue(cfg["sobbh_use_tdionfly"], "expected the TDI-on-the-fly path")
        info, row, params_in = self._row_and_params("sobbh")
        ref = info.signal_gen(*row)
        tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=gi.force_backend)
        wave = build_sobbh_stock_waveform(
            Tobs=gi.Tobs,
            dt=gi.dt,
            t_start=gi.data_t0,
            td_settings=gi.data_td_settings,
            target_domain=gi.domain_settings,
            tdi_config=tdi_config,
            reference_time=cfg["sobbh_reference_time"],
            orbits=gi.orbits,
            n_grid=cfg["sobbh_n_grid"],
            buffer_time=cfg["sobbh_buffer_time"],
            force_backend=gi.force_backend,
            nchannels=cfg["nchannels"],
        )
        out = wave(*params_in)
        self._assert_equiv(ref, out, "sobbh")

    def test_gb_xyz_matches_generate_global_template(self):
        from gbgpu.gbgpu import GBGPU

        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.globalfit.stock.erebor.injections import GB_INJECTION_PARAMS
        from lisatools.sources.gb.waveform import GBXYZTDIWaveform

        gi = self.curr.general_info
        Tobs, dt = gi.Tobs, gi.dt
        row = np.atleast_2d(np.asarray(GB_INJECTION_PARAMS, dtype=np.float64))[0:1]

        wave = GBXYZTDIWaveform(
            tdi_chan="XYZ", use_tdi2=True, oversample=1, force_backend="cpu"
        )
        out = np.asarray(wave(row, Tobs, dt))

        # Direct generate_global_template call (SyntheticGBProcessingStep convention).
        target_N = int(round(Tobs / dt))
        data_length = target_N // 2 + 1
        gb = GBGPU(force_backend="cpu", orbits=EqualArmlengthOrbits(force_backend="cpu"))
        gb.gpus = None
        template_flat = gb.xp.zeros(3 * data_length, dtype=gb.xp.complex128)
        gb.generate_global_template(
            row,
            gb.xp.zeros(1, dtype=gb.xp.int32),
            template_flat,
            start_freq_ind=0,
            T=Tobs,
            dt=dt,
            tdi_channel_setup="XYZ",
            tdi2=True,
            oversample=1,
            data_length=data_length,
        )
        ref = np.asarray(template_flat.reshape(3, data_length))
        scale = float(np.max(np.abs(ref))) or 1.0
        rel = float(np.max(np.abs(out - ref))) / scale
        print(f"[gb] rel_diff={rel:.3e} scale={scale:.3e}")
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
