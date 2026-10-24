"""Store-window tests for the STFT GB band buffer.

A windowed cell holds ``[start_j, start_j + W)`` of the parent active grid instead of the whole
grid, and the kernels place every pixel through the per-cell start. These tests compare a windowed
buffer against a full-grid one built from the same parent and the same sources:

* **T2 window parity** -- ``get_ll``, ``get_swap_ll`` and the windowed slice of the filled template
  are bit-identical between the two.
* **T3 slot refill** -- moving a slot to another band keeps the per-split start array's pointer and
  T2 parity.
* **T4 grid edges** -- bands whose window clips at bin 0 and at ``NF_active - W`` keep T2 parity.
* **T5 tempering score** -- the likelihood DIFFERENCE across a template swap agrees with the
  full-grid one; the absolute values differ by the residual outside the window, which cancels.
* **T7 track check** -- the counter fires when a predicted carrier span leaves the window, and
  stays at zero otherwise.

Both backends run the same bodies. The CUDA cases skip unless a GPU is usable, and everything
skips unless gbgpu with ``STFTGBComputations`` is importable.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

BIG_DT = 21600.0          # STFT segment length
NT_STFT = 8
NF_STFT = 128
DF_STFT = 1.0 / BIG_DT
T0_STFT = 10.0 * 86400.0  # anchored away from the orbit-file edge
IND_MIN, IND_MAX = 40, 100
BAND_COLS = 9             # STFT columns per GB band
N_BANDS = 6
NTEMPS, NWALKERS, NLEAVES_MAX, NDIM = 2, 4, 3, 8
N_SIDE_BINS = 3
INVC_SCALE = 1e41
TOBS = NT_STFT * BIG_DT


def _have_gbgpu_stft() -> bool:
    try:
        from gbgpu.gbcomps import STFTGBComputations  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _cuda_backend_name():
    """Name of a usable CUDA backend, or None."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        import lisatools

        for name in ("cuda12x", "cuda11x", "cuda13x"):
            if lisatools.has_backend(name):
                cupy.zeros(1)  # a device that cannot allocate is not usable
                return name
    except Exception:
        return None
    return None


def _test_gpu_list():
    """Devices the parent ACA and the band buffer are sharded over, from ``GB_TEST_GPUS``.

    Default is the current device alone, so the CUDA cases stay single-shard. Set
    ``GB_TEST_GPUS=0,1`` to shard both sides and reach the cross-device window copy in
    ``SubBandBuffer._copy_stft_windows``, which a single-shard run never takes.
    """
    import cupy

    requested = os.environ.get("GB_TEST_GPUS", "").strip()
    if not requested:
        return [int(cupy.cuda.runtime.getDevice())]
    return [int(part) for part in requested.split(",") if part.strip() != ""]


def f_ms_to_s(x):
    return x * 1e-3


def _band_edges():
    """``N_BANDS`` contiguous BAND_COLS-column bands, starting two columns inside the grid."""
    return (IND_MIN + 2) * DF_STFT + np.arange(N_BANDS + 1) * BAND_COLS * DF_STFT


def build_case(backend: str = "cpu", seed: int = 7) -> dict:
    """Parent ACA with random residual and inverse CSD, a band sorter and the store windows."""
    from gbgpu.gbcomps import STFTGBComputations
    from gbgpu.gbgpu import GBGPU

    from eryn.state import Branch
    from eryn.utils import TransformContainer

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import STFTSettings
    from lisatools.globalfit.moves.gbbandstructure import (
        GBBandStructure, stft_store_window_layout,
    )
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZSensitivityBackend

    rng = np.random.default_rng(seed)
    gpus = None if backend == "cpu" else _test_gpu_list()
    settings = STFTSettings(
        t0=T0_STFT, dt=BIG_DT, df=DF_STFT, NT=NT_STFT, NF=NF_STFT,
        min_freq=IND_MIN * DF_STFT, max_freq=IND_MAX * DF_STFT, force_backend=backend,
    )
    nch, n_active = 3, settings.NF_active
    data_shape = (nch, NT_STFT, n_active)
    sens_shape = (nch, nch, NT_STFT, n_active)

    orbits = EqualArmlengthOrbits(force_backend=backend)
    ac_list = []
    for _ in range(NWALKERS):
        # Random residual and a Hermitian inverse CSD: parity must not depend on either being flat.
        res = (rng.standard_normal(data_shape) + 1j * rng.standard_normal(data_shape)) * 1e-19
        invC = np.zeros(sens_shape, dtype=np.complex128)
        for channel in range(nch):
            invC[channel, channel] = INVC_SCALE * (1.0 + rng.random((NT_STFT, n_active)))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            off = 0.1 * INVC_SCALE * (rng.standard_normal((NT_STFT, n_active))
                                      + 1j * rng.standard_normal((NT_STFT, n_active)))
            invC[a, b], invC[b, a] = off, off.conj()
        sens = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=backend)
        sens.sens_mat = np.zeros(sens_shape, dtype=np.complex128)
        sens.invC = invC
        sens.detC = np.full((NT_STFT, n_active), (1.0 / INVC_SCALE) ** nch)
        sens.channel_shape = sens_shape[: -len(settings.basis_shape_active)]
        ac_list.append(AnalysisContainer(
            settings.associated_class(res.astype(np.complex128), settings), sens))

    aca = AnalysisContainerArray(
        ac_list, gpus=gpus,
        domain_group_kwargs=dict(tdi_type="XYZ", window_alpha=0.0, use_midpoint=False),
    )

    gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend=backend), force_backend=backend)
    gb.gpus = gpus
    comp = STFTGBComputations(
        stft_comps=aca.cpp_splits[0], T=TOBS, t_ref=0.0, orbits=orbits,
        tdi_config=TDIConfig("1st generation", force_backend=backend),
        force_backend=backend, n_side_bins=N_SIDE_BINS, window_factor=1.0,
        freq_from_tdi_phase=False, response="tdi_on_the_fly",
    )

    band_edges = _band_edges()
    structure = GBBandStructure(
        domain="stft", edges_hz=band_edges, tobs_s=TOBS, grid_df_hz=DF_STFT)
    windows = stft_store_window_layout(
        structure, settings, n_side_bins=N_SIDE_BINS,
        epoch_offset_s=comp.t_ref - settings.t0, tier_split_hz=0.0,
    )

    transform = TransformContainer(
        input_basis=["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"],
        output_basis=["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"],
        parameter_transforms={"A": np.exp, "f0": f_ms_to_s, "cos_iota": np.arccos,
                              "sin_beta": np.arcsin},
        fill_dict={"fddot": 0.0},
    )

    # One source per (temp, walker, band) in the interior bands, at the band centre.
    coords = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX, NDIM))
    inds = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX), dtype=bool)
    interior = list(range(1, N_BANDS - 1))
    for walker in range(NWALKERS):
        draws = []
        for leaf, band in enumerate(interior[:NLEAVES_MAX]):
            centre_hz = 0.5 * (band_edges[band] + band_edges[band + 1])
            draws.append([
                np.log(4e-21), centre_hz * 1e3, 1e-14 * rng.standard_normal(),
                rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1), rng.uniform(0, np.pi),
                rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1),
            ])
        for temp in range(NTEMPS):
            coords[temp, walker, :len(draws)] = np.asarray(draws)
            inds[temp, walker, :len(draws)] = True

    return dict(
        backend=backend, settings=settings, aca=aca, gb=gb, comp=comp, windows=windows,
        transform=transform, band_edges=band_edges, band_N_vals=np.full(N_BANDS, 64),
        branch=Branch(coords, inds=inds), waveform_kwargs=dict(dt=5.0, T=TOBS,
                                                               tdi_channel_setup="XYZ"),
    )


def make_sorter(case: dict, windows):
    from lisatools.globalfit.moves.gbbands import BandSorter

    return BandSorter(
        case["branch"], case["band_edges"], case["band_N_vals"],
        force_backend=case["backend"], transform_fn=case["transform"],
        max_data_store_size=512, gb=case["gb"], gb_stft_comp=case["comp"],
        waveform_kwargs=case["waveform_kwargs"], stft_store_windows=windows,
    )


def host(value):
    from lisatools.utils.utility import asnumpy

    return asnumpy(value)


class _StoreWindowCases:
    """Bodies shared by the CPU and CUDA test cases; ``BACKEND`` is set by the subclasses."""

    BACKEND = "cpu"

    def setUp(self):
        self.case = build_case(self.BACKEND)
        self.windows = self.case["windows"]
        self.width = self.windows.buffer_width
        self.assertLess(self.width, int(self.case["settings"].NF_active))

    # ---------------- helpers ----------------

    def buffers(self, specials, **kwargs):
        """A windowed buffer and a full-grid buffer over the same cells."""
        windowed = make_sorter(self.case, self.windows).get_buffer(
            self.case["aca"], specials, **kwargs)
        full = make_sorter(self.case, None).get_buffer(self.case["aca"], specials, **kwargs)
        return windowed, full

    def live_cells(self):
        """Special indices of every cell holding a source, and that sorter."""
        sorter = make_sorter(self.case, self.windows)
        specials = np.unique(host(sorter.special_band_inds[sorter.inds]))
        return sorter, sorter.xp.asarray(specials)

    def assert_parity(self, windowed, full, sorter, source_ids):
        """T2 on one pair of buffers: likelihoods, swap and the windowed template slice."""
        xp = windowed.xp
        params = sorter.coords[source_ids]
        specials = sorter.special_band_inds[source_ids]
        slots = windowed.get_index(specials)
        np.testing.assert_array_equal(host(slots), host(full.get_index(specials)))
        n_vals = sorter.band_N_vals[sorter.band_inds[source_ids]]

        for name, call in (
            ("get_ll", lambda buf: buf.get_ll(params, slots, slots, n_vals)),
            ("d_h", lambda buf: buf.get_ll(params, slots, slots, n_vals) * 0 + buf.d_h_out),
            ("h_h", lambda buf: buf.get_ll(params, slots, slots, n_vals) * 0 + buf.h_h_out),
            ("get_swap_ll", lambda buf: buf.get_swap_ll(
                params, params * xp.asarray(np.where(np.arange(NDIM) == 1, 1.00001, 1.0)),
                slots, n_vals)),
        ):
            np.testing.assert_array_equal(
                host(call(windowed)), host(call(full)),
                err_msg=f"{name} differs between the windowed and the full-grid buffer",
            )

        starts = host(windowed.buffer_start_index)
        for slot in range(windowed.num_bands_now):
            sliced = host(full.band_buffer[slot])[
                ..., starts[slot]:starts[slot] + self.width]
            np.testing.assert_array_equal(
                host(windowed.band_buffer[slot]), sliced,
                err_msg=f"slot {slot} residual differs after the source injection",
            )

    # ---------------- T2 ----------------

    def test_t2_window_parity(self):
        sorter, specials = self.live_cells()
        windowed, full = self.buffers(specials)
        self.assertEqual(
            windowed.band_buffer.shape[-1], self.width,
            msg="the windowed buffer did not take the store-window width",
        )
        ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        self.assert_parity(windowed, full, sorter, ids)

    # ---------------- T3 ----------------

    def test_t3_slot_refill_keeps_pointer(self):
        from lisatools.globalfit.moves.gbbands import pack_special_index

        sorter, _ = self.live_cells()
        live = np.unique(host(sorter.special_band_inds[sorter.inds]))
        # Start from one cell, then refill that slot with a cell of a DIFFERENT band.
        first, second = int(live[0]), int(live[-1])
        self.assertNotEqual(first % int(1e6), second % int(1e6),
                            msg="the two cells must differ in band for this test")

        windowed = make_sorter(self.case, self.windows).get_buffer(
            self.case["aca"], sorter.xp.asarray(np.asarray([first])))
        pointers = [_pointer(arr) for arr in windowed.stft_split_start_inds]
        start_before = int(host(windowed.buffer_start_index)[0])

        refill_sorter = make_sorter(self.case, self.windows)
        refill_sorter.get_buffer(
            self.case["aca"], refill_sorter.xp.asarray(np.asarray([second])),
            inds_fill=refill_sorter.xp.asarray(np.asarray([0])), buffer_obj=windowed,
        )
        self.assertEqual(pointers, [_pointer(arr) for arr in windowed.stft_split_start_inds],
                         msg="the per-split start array was rebound instead of updated in place")
        start_after = int(host(windowed.buffer_start_index)[0])
        self.assertNotEqual(start_before, start_after,
                            msg="the refilled slot kept the old band's window start")
        self.assertEqual(int(host(windowed.stft_split_start_inds[0])[0]), start_after)

        full = make_sorter(self.case, None).get_buffer(
            self.case["aca"], refill_sorter.xp.asarray(np.asarray([second])))
        ids = refill_sorter.xp.arange(refill_sorter.num_sources)[
            refill_sorter.special_band_inds == second]
        self.assert_parity(windowed, full, refill_sorter, ids)

    # ---------------- T4 ----------------

    def test_t4_grid_edge_windows(self):
        from lisatools.globalfit.moves.gbbands import pack_special_index

        sorter, _ = self.live_cells()
        # Band 0 clips at bin 0, the last band clips at NF_active - W.
        edge_specials = np.asarray([
            int(pack_special_index(0, 0, band, NWALKERS)) for band in (0, N_BANDS - 1)
        ])
        starts = self.windows.window_starts(np.asarray([0, N_BANDS - 1]), self.width)
        self.assertEqual(int(starts[0]), 0, msg="band 0 should clip at bin 0")
        self.assertEqual(int(starts[1]), int(self.case["settings"].NF_active) - self.width,
                         msg="the last band should clip at NF_active - W")

        windowed, full = self.buffers(sorter.xp.asarray(edge_specials))
        # Sources placed inside each edge band: their tracks are what the parity must hold for.
        params = []
        for band in (0, N_BANDS - 1):
            centre = 0.5 * (self.case["band_edges"][band] + self.case["band_edges"][band + 1])
            params.append([np.log(4e-21), centre * 1e3, 0.0, 0.3, 0.2, 0.4, 1.0, 0.1])
        params = windowed.xp.asarray(np.asarray(params))
        slots = windowed.get_index(windowed.xp.asarray(edge_specials))
        n_vals = windowed.xp.asarray(np.full(2, 64))
        np.testing.assert_array_equal(
            host(windowed.get_ll(params, slots, slots, n_vals)),
            host(full.get_ll(params, slots, slots, n_vals)),
            err_msg="edge-band cells lost parity with the full grid",
        )

    # ---------------- T5 ----------------

    def test_t5_tempering_swap_difference(self):
        sorter, specials = self.live_cells()
        windowed, full = self.buffers(specials, use_template_arr=True)

        # Swap the templates of the two temperatures of one (walker, band) cell pair.
        combos = host(windowed.unique_band_combos)
        pairs = {}
        for slot, (temp, walker, band) in enumerate(combos):
            pairs.setdefault((int(walker), int(band)), {})[int(temp)] = slot
        slots_a = [v[0] for v in pairs.values() if 0 in v and 1 in v]
        slots_b = [v[1] for v in pairs.values() if 0 in v and 1 in v]
        self.assertGreater(len(slots_a), 0, msg="no temperature pair in the buffer")

        deltas = []
        for buf in (windowed, full):
            xp = buf.xp
            before = host(buf.band_likelihoods(source_only=True))
            buf.swap_template_slots(xp.asarray(np.asarray(slots_a)),
                                    xp.asarray(np.asarray(slots_b)))
            after = host(buf.band_likelihoods(source_only=True))
            deltas.append(after - before)
        scale = np.abs(host(full.band_likelihoods(source_only=True))).max()
        np.testing.assert_allclose(
            deltas[0], deltas[1], rtol=0, atol=1e-12 * scale,
            err_msg="the swap difference of the windowed buffer left the full-grid one",
        )

    # ---------------- T8 ----------------

    def test_t8_source_terms_of_selected_cells(self):
        """Per-cell source terms against a host contraction, and unchanged when cells are selected.

        Each shard contracts its own cells, and the tempering swap asks for two temperature columns
        at a time, so this covers the sharding, the stride selection and the unordered one. The
        residual buffer must also survive the template subtraction.
        """
        sorter, specials = self.live_cells()
        buf = make_sorter(self.case, self.windows).get_buffer(
            self.case["aca"], specials, use_template_arr=True)

        ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        slots = buf.get_index(sorter.special_band_inds[ids])
        buf.add_sources_to_template_buffer(
            sorter.coords[ids], slots, sorter.band_N_vals[sorter.band_inds[ids]])
        band_before = host(buf._materialize(buf.band_buffer))
        self.assertTrue(np.any(host(buf._materialize(buf.template_buffer)) != 0.0),
                        msg="the template buffer stayed empty, so the subtraction is untested")

        all_terms = host(buf.band_likelihoods(source_only=True))

        residual = band_before - host(buf._materialize(buf.template_buffer))
        num_cells, nchannels = residual.shape[0], buf.nchannels
        flat = residual.reshape(num_cells, nchannels, -1)
        inv_csd = host(buf._materialize(buf.psd_buffer)).reshape(
            num_cells, nchannels, nchannels, -1)
        reference = -2.0 * float(buf.settings.differential_component) * np.einsum(
            "bik,bijk,bjk->b", flat.conj(), inv_csd, flat).real
        np.testing.assert_allclose(
            all_terms, reference, rtol=1e-12, atol=0,
            err_msg="the per-cell source terms left a host contraction of the same cells",
        )

        # One block per temperature column, which is the stride the tempering swap rescores.
        columns = np.arange(num_cells).reshape(-1, NTEMPS).T
        np.testing.assert_allclose(
            host(buf.band_likelihoods(source_only=True, cells=columns)), all_terms[columns],
            rtol=1e-12, atol=0, err_msg="selecting temperature columns changed their source terms",
        )
        # Unordered cells, which cannot be read as a stride.
        scattered = np.asarray([num_cells - 1, 0, num_cells // 2])
        np.testing.assert_allclose(
            host(buf.band_likelihoods(source_only=True, cells=scattered)), all_terms[scattered],
            rtol=1e-12, atol=0, err_msg="an unordered cell selection changed its source terms",
        )

        np.testing.assert_array_equal(
            host(buf._materialize(buf.band_buffer)), band_before,
            err_msg="scoring changed the residual buffer, so a view was subtracted in place",
        )

    # ---------------- T7 ----------------

    def test_t7_track_check_counts(self):
        from lisatools.globalfit.moves.gbbands import StoreWindowTrackCounter

        sorter, specials = self.live_cells()
        counter = StoreWindowTrackCounter()
        windowed = make_sorter(self.case, self.windows).get_buffer(
            self.case["aca"], specials, track_counter=counter)
        xp = windowed.xp

        ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        params = sorter.coords[ids].copy()
        slots = windowed.get_index(sorter.special_band_inds[ids])
        n_vals = sorter.band_N_vals[sorter.band_inds[ids]]

        counter.__init__()
        windowed.get_ll(params, slots, slots, n_vals)
        counter._flush()
        self.assertEqual(counter.rows_outside, 0,
                         msg="in-band sources must not be reported outside their window")
        self.assertEqual(counter.rows_evaluated, int(params.shape[0]))

        # fdot far past the prior edge: the carrier leaves the window over the record.
        runaway = params.copy()
        runaway[:, 2] = 1e-9
        counter.__init__()
        windowed.get_ll(runaway, slots, slots, n_vals)
        counter._flush()
        self.assertEqual(counter.rows_outside, int(params.shape[0]),
                         msg="a runaway chirp must be counted for every row")
        self.assertGreater(len(counter._cells_outside), 0)

    # ---------------- Fisher kernel on a real computation group ----------------

    def test_information_matrix_kernel_matches_reference(self):
        bound = 1e-13 if self.BACKEND == "cpu" else 1e-10
        sorter, _specials = self.live_cells()
        ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        physical = self.case["transform"].both_transforms(sorter.coords[ids], xp=sorter.xp)
        walkers = sorter.walker_inds[ids].astype(sorter.xp.int32)
        test_inds = [0, 1, 2, 4, 5, 6, 7, 8]
        comp, aca = self.case["comp"], self.case["aca"]

        from gbgpu.testing import stft_information_matrix_reference

        kernel = host(comp.information_matrix(physical, aca, inds=test_inds, noise_index=walkers))
        reference = host(stft_information_matrix_reference(
            comp, physical, aca, inds=test_inds, noise_index=walkers))

        diagonal = np.abs(np.diagonal(reference, axis1=1, axis2=2))
        self.assertTrue(np.all(diagonal > 0), msg="every sampled direction must carry information")
        norm = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
        worst = float(np.max(np.abs(kernel - reference) / norm))
        self.assertLess(worst, bound, msg=f"kernel Fisher left the reference by {worst:.3e}")
        np.testing.assert_array_equal(kernel, np.swapaxes(kernel, 1, 2))


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class StoreWindowDebugFlowTest(unittest.TestCase):
    """A windowed propose with the debug tools on: every slab tool must run, not skip.

    The debug helpers reshape cell slabs and convert bins to frequencies, so a full-grid geometry
    on a windowed cell shows up as a caught exception and a "skipped" warning rather than a failure.
    """

    def test_debug_tools_run_on_windowed_cells(self):
        import logging
        import os
        import tempfile

        from lisatools.globalfit.moves.gbbandstructure import (
            GBBandStructure, stft_store_window_layout,
        )
        from lisatools.globalfit.moves.gbdebug import GBDebugSettings
        from lisatools.globalfit.moves.gbspecialmove import GBSpecialRJPriorMove

        from .test_gbspecial_flow_stft import build_fixture, N_BANDS

        fx = build_fixture()
        settings = fx["settings"]
        structure = GBBandStructure(
            domain="stft", edges_hz=np.asarray(fx["band_edges"]),
            tobs_s=settings.NT * settings.dt, grid_df_hz=settings.df,
        )
        windows = stft_store_window_layout(
            structure, settings, n_side_bins=fx["gb_stft_comp"].n_side_bins,
            epoch_offset_s=fx["gb_stft_comp"].t_ref - settings.t0, tier_split_hz=0.0,
        )
        self.assertLess(windows.buffer_width, int(settings.NF_active))

        with tempfile.TemporaryDirectory() as plot_dir:
            move = GBSpecialRJPriorMove(
                *fx["move_args"], is_rj_prop=True, name="rj_windowed_debug",
                **{
                    **fx["move_kwargs"],
                    "rj_proposal_distribution": fx["priors"],
                    "stft_store_windows": windows,
                    "debug_settings": GBDebugSettings(
                        enabled=True, plot_dir=plot_dir, plot_walker=0,
                        plot_band=N_BANDS // 2),
                },
            )
            move.temperature_control = fx["temperature_control"]
            move.time = 0

            with self.assertLogs("lisatools.globalfit.moves", level=logging.INFO) as captured:
                new_state, _ = move.propose(fx["model"], fx["state"])

            figures = sorted(os.listdir(plot_dir))

        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        skipped = [line for line in captured.output
                   if "GB_DEBUG" in line and ("skipped" in line or "failed" in line)]
        self.assertEqual(skipped, [], msg="a debug tool could not read the windowed cells")
        self.assertTrue(
            any("store-window track check" in line for line in captured.output),
            msg="the track check did not report once for the propose",
        )
        # The slab tools must have RUN, not merely not failed.
        self.assertTrue(
            any("sub-band SOURCE-ONLY residual ll" in line for line in captured.output),
            msg="the band-null log never ran on the windowed cells",
        )
        self.assertTrue(
            any(name.startswith("gb_debug_seq1_before_removal") for name in figures),
            msg=f"no in-model sequence figure was written; got {figures}",
        )


def _pointer(arr) -> int:
    """Device or host address of an array's first element."""
    data = getattr(arr, "data", None)
    if hasattr(data, "ptr"):
        return int(data.ptr)
    return int(arr.__array_interface__["data"][0])


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class StoreWindowCPUTest(_StoreWindowCases, unittest.TestCase):
    BACKEND = "cpu"


@unittest.skipUnless(_have_gbgpu_stft() and _cuda_backend_name(), "requires a usable CUDA backend")
class StoreWindowCudaTest(_StoreWindowCases, unittest.TestCase):
    BACKEND = _cuda_backend_name() or "cpu"


if __name__ == "__main__":
    unittest.main()
