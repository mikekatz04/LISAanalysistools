"""Tests for buffer preload limits estimation and STFT/WDM buffer indexing."""

import unittest
import numpy as np

from lisatools.domains import FDSettings, STFTSettings, WDMSettings
from lisatools.globalfit.moves.gbbands import estimate_buffer_preload_limits, SubBandBuffer
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.detector import EqualArmlengthOrbits


class TestBufferPreloadLimits(unittest.TestCase):
    def test_fd_limits(self):
        fd_settings = FDSettings(N=128, df=1e-4, force_backend="cpu")
        p_fd, t_fd = estimate_buffer_preload_limits(
            fd_settings,
            nchannels=3,
            tdi_setup="XYZ",
            max_data_store_size=6000,
            ntemps=4,
            max_budget_bytes=4 * 1024**3,
        )
        self.assertEqual(t_fd, 200)
        self.assertGreater(p_fd, 0)

    def test_stft_limits(self):
        stft_dt = 86400.0
        Tobs = 0.75 * 31558149.7632
        NT = int(np.ceil(Tobs / stft_dt))
        NF = int(stft_dt / 5.0)
        stft_settings = STFTSettings(
            t0=0.0,
            dt=stft_dt,
            df=1.0 / stft_dt,
            NT=NT,
            NF=NF,
            min_freq=0.014,
            max_freq=0.022,
            force_backend="cpu",
        )
        p_stft, t_stft = estimate_buffer_preload_limits(
            stft_settings,
            nchannels=3,
            tdi_setup="XYZ",
            ntemps=4,
            max_budget_bytes=4 * 1024**3,
        )
        self.assertGreaterEqual(t_stft, 1)
        self.assertLessEqual(t_stft, 15)
        self.assertGreaterEqual(p_stft, 1)
        self.assertLessEqual(p_stft, 50)

    def test_fill_buffer_stft_1d_indexing(self):
        stft_dt = 86400.0
        NT = 8
        NF = 128
        settings = STFTSettings(
            t0=0.0,
            dt=stft_dt,
            df=1.0 / stft_dt,
            NT=NT,
            NF=NF,
            min_freq=0.014,
            max_freq=0.022,
            force_backend="cpu",
        )
        orbits = EqualArmlengthOrbits(force_backend="cpu")

        nch = 3
        data_shape = (nch, settings.NT, settings.NF_active)
        sens_shape = (nch, nch, settings.NT, settings.NF_active)

        acs = []
        for i in range(4):
            res_data = np.full(data_shape, i + 1, dtype=np.complex128)
            data_domain = settings.associated_class(res_data, settings)
            sm = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend="cpu")
            sm.sens_mat = np.zeros(sens_shape, dtype=np.complex128)
            sm.invC = np.zeros(sens_shape, dtype=np.complex128)
            for j in range(nch):
                sm.invC[j, j] = 1e40 * (i + 1)
            sm.detC = np.ones((settings.NT, settings.NF_active))
            sm.channel_shape = sens_shape[: -len(settings.basis_shape_active)]
            acs.append(AnalysisContainer(data_domain, sm))

        aca = AnalysisContainerArray(
            acs,
            gpus=None,
            domain_group_kwargs=dict(tdi_type="XYZ", window_alpha=0.0, use_midpoint=False),
        )

        from gbgpu.gbcomps import STFTGBComputations
        gb_stft_comp = STFTGBComputations(
            stft_comps=aca.cpp_splits[0],
            T=NT * stft_dt,
            t_ref=0.0,
            response="tdi_on_the_fly",
        )

        combos = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 2, 1],
            [1, 3, 1],
        ])
        from lisatools.globalfit.moves.gbbands import pack_special_index
        specials = pack_special_index(combos[:, 0], combos[:, 1], combos[:, 2], nwalkers=4)

        class DummyBackend:
            name = "cpu"

        class DummyGB:
            backend = DummyBackend()

        class DummyTransform:
            pass

        buf = SubBandBuffer(
            is_rj=False,
            nwalkers=4,
            gb=DummyGB(),
            band_edges=np.array([0.014, 0.018, 0.022]),
            band_N_vals=np.array([64, 64]),
            unique_band_combos=combos,
            params_interest=np.empty((0, 8)),
            num_bands_now=4,
            nchannels=3,
            data_length=settings.NT * settings.NF_active,
            special_indices_unique=specials,
            transform_fn=DummyTransform(),
            waveform_kwargs={"tdi_channel_setup": "XYZ"},
            df=settings.df,
            sources_now_map=np.empty((0,), dtype=int),
            sources_inject_now_map=np.empty((0,), dtype=int),
            special_band_inds=np.empty((0,), dtype=int),
            force_backend="cpu",
            use_template_arr=True,
            basis_settings=settings,
            gb_stft_comp=gb_stft_comp,
        )

        buf.fill_buffer_residual_and_psd_from_acs(aca)

        for cell_i, (temp, walker, band) in enumerate(combos):
            np.testing.assert_allclose(buf.band_buffer[cell_i], aca.data_shaped[0][walker])
            np.testing.assert_allclose(buf.psd_buffer[cell_i], aca.psd_shaped[0][walker])

    def test_fill_buffer_stft_multi_shard_routing(self):
        """Verify that 1D indexing correctly gathers cross-shard in BandView."""
        from lisatools.analysiscontainer import BandView

        try:
            from test_band_view_multi_shard import _FakeMultiShardACA
        except ImportError:
            from tests.test_band_view_multi_shard import _FakeMultiShardACA

        # 4 walkers, 2 shards (2 walkers per shard), per-band shape: (3 channels, 4 time, 8 freq)
        fake_aca = _FakeMultiShardACA(per_band_shape=(3, 4, 8), num_acs=4, num_shards=2)
        data_view = BandView(fake_aca, kind="data")

        # 1D index mapping selecting walkers across shards: [3, 0, 2, 1]
        walker_inds = np.array([3, 0, 2, 1])
        gathered = data_view[walker_inds]

        self.assertEqual(gathered.shape, (4, 3, 4, 8))
        np.testing.assert_allclose(gathered[0], 4.0)
        np.testing.assert_allclose(gathered[1], 1.0)
        np.testing.assert_allclose(gathered[2], 3.0)
        np.testing.assert_allclose(gathered[3], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
