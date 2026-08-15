"""Flow tests for the reworked GB special moves (CPU, small fixture).

Three levels:

1. **Buffer identities** -- cell loading equality against the parent window
   and the removal-delta identity ``get_removal_ll == ll(r+h) - ll(r)``
   measured through an actual residual round-trip (independent
   ``band_likelihoods`` path vs the engine inner products).
2. **RJ propose smoke** -- ``GBSpecialRJPriorMove.propose`` runs the full
   new flow (without-replacement picks, RJ step, in-model repeats,
   info-matrix Cholesky) on a state with real sources.
3. **In-model propose smoke** -- ``GBSpecialStretchMove.propose`` (no RJ
   distribution) runs twice so the second pass can draw the group stretch.

The fixture is intentionally tiny (2 temps, 4 walkers, ~3.6k FD bins,
Tobs = 0.25 yr) to stay memory-light and runs the production TDI
convention: XYZ with the full (real-part) cross-channel inverse
covariance through the new gb_fd kernel family.

Everything skips unless gbgpu is importable.
"""

from __future__ import annotations

import unittest

import numpy as np


def _have_gbgpu() -> bool:
    try:
        import gbgpu  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


NTEMPS = 2
NWALKERS = 4
NLEAVES_MAX = 4
NDIM = 8
N_PER_BAND = 64
N_BANDS = 6
K_SOURCES = 2  # per (temp, walker)


def f_ms_to_s(x):
    return x * 1e-3


def build_fixture(seed=42, use_fdot_astro=False, use_distance=False):
    """Small FD global-fit fixture with K_SOURCES live GBs per walker.

    ``use_fdot_astro`` builds the 9-column ratio basis (``[A, f0, Mc, phi0,
    cos_iota, psi, alpha, sin_delta, fdot_astro_ratio]``) via the real
    :func:`make_gb_transform_container` factory; ``use_distance`` swaps slot 0
    from lnA to the distance basis (``dist`` kpc, amplitude derived). Either
    exercises the move-propose flow in the same basis the stock
    chirp-mass+astro-prior run walks.
    """
    use_fdot_astro = use_fdot_astro or use_distance
    from gbgpu.gbgpu import GBGPU

    from eryn.moves.tempering import TemperatureControl
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State as ErynState
    from eryn.utils import PeriodicContainer, TransformContainer

    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.datacontainer import DataResidualArray
    from lisatools.domains import FDSettings
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.globalfit.engine import GlobalFitInfo
    from lisatools.globalfit.state import GBState, GFState
    from lisatools.utils.constants import YRSID_SI

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    dt = 10.0
    Tobs = 0.25 * YRSID_SI
    df = 1.0 / Tobs
    f0_lo, f0_hi = 2.9e-3, 3.1e-3
    n_pad = 1024
    data_length = int(np.ceil((f0_hi - f0_lo) / df)) + 2 * n_pad
    f_start = f0_lo - n_pad * df
    fd = f_start + np.arange(data_length) * df

    gb_orbits = EqualArmlengthOrbits(force_backend="cpu")
    gb = GBGPU(orbits=gb_orbits, force_backend="cpu")
    gb.gpus = None

    band_width = (2 * N_PER_BAND + 5) * df
    band_edges = fd[0] + (n_pad // 2) * df + np.arange(N_BANDS + 1) * band_width
    band_N_vals = np.full(N_BANDS, N_PER_BAND)

    # f0 prior INSIDE the interior bands (1 .. N_BANDS-2); the edge bands are
    # excluded from proposals by the move.
    f0_prior_lo = (band_edges[1] + 0.2 * band_width) * 1e3   # mHz
    f0_prior_hi = (band_edges[N_BANDS - 1] - 0.2 * band_width) * 1e3
    ndim = 9 if use_fdot_astro else NDIM
    ratio_max = 5.0
    mc_lims = (0.001, 1.0)
    if use_fdot_astro:
        # Real 9-column ratio-basis factory: col 2 = Mc (modest box keeps
        # fdot_gr*(1+r) inside the fixture band), col 8 = r ~ U[-M, M]
        # (r < -1 gives fdot < 0, exercising the interacting branch).
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container,
        )

        # slot 0: lnA (ratio basis) or distance kpc (distance basis)
        slot0 = (uniform_dist(1.0, 30.0) if use_distance
                 else uniform_dist(np.log(5e-22), np.log(1e-20)))
        priors_in = {
            0: slot0,
            1: uniform_dist(f0_prior_lo, f0_prior_hi),
            2: uniform_dist(0.15, 0.45),           # Mc [Msol]
            3: uniform_dist(0.0, 2 * np.pi),
            4: uniform_dist(-1, 1),
            5: uniform_dist(0.0, np.pi),
            6: uniform_dist(0.0, 2 * np.pi),
            7: uniform_dist(-1, 1),
            8: uniform_dist(-ratio_max, ratio_max),  # fdot_astro_ratio
        }
        transform = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=use_distance,
            mc_lims=mc_lims,
        )
    else:
        priors_in = {
            0: uniform_dist(np.log(5e-22), np.log(1e-20)),
            1: uniform_dist(f0_prior_lo, f0_prior_hi),
            2: uniform_dist(-1e-13, 1e-13),
            3: uniform_dist(0.0, 2 * np.pi),
            4: uniform_dist(-1, 1),
            5: uniform_dist(0.0, np.pi),
            6: uniform_dist(0.0, 2 * np.pi),
            7: uniform_dist(-1, 1),
        }
        transform = TransformContainer(
            input_basis=["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"],
            output_basis=["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"],
            parameter_transforms={
                "A": np.exp,
                "f0": f_ms_to_s,
                "cos_iota": np.arccos,
                "sin_beta": np.arcsin,
            },
            fill_dict={"fddot": 0.0},
        )
    priors = {"gb": ProbDistContainer(priors_in)}
    waveform_kwargs = dict(dt=dt, T=Tobs, tdi_channel_setup="XYZ")

    nchan = 3
    N_full = int(np.ceil(fd[-1] / df)) + 1
    ind_lo = int(np.round(fd[0] / df))
    acs_list = []
    for _ in range(NWALKERS):
        fd_settings = FDSettings(
            N=N_full, df=df, min_freq=float(fd[0]), max_freq=float(fd[-1]),
            force_backend="cpu",
        )
        drs = DataResidualArray(
            np.zeros((nchan, N_full), dtype=np.complex128),
            input_signal_domain=fd_settings,
        )
        drs._f_arr = fd_settings.f_arr
        drs._df = df
        drs._dt = None
        drs._Tobs = None
        drs._fmax = float(fd_settings.f_arr.max())
        acs_list.append(
            AnalysisContainer(drs, XYZ2SensitivityMatrix(drs.settings, model="scirdv1"))
        )
    acs = AnalysisContainerArray(acs_list, gpus=None, complex_psd=True)

    # K_SOURCES live sources per walker, identical across temperatures.
    gb_coords = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX, ndim))
    gb_inds = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX), dtype=bool)
    for w in range(NWALKERS):
        draws = priors["gb"].rvs(size=K_SOURCES)
        for t in range(NTEMPS):
            gb_coords[t, w, :K_SOURCES] = draws
            gb_inds[t, w, :K_SOURCES] = True

    eryn_state = ErynState(
        {"gb": gb_coords},
        inds={"gb": gb_inds},
        log_like=np.zeros((NTEMPS, NWALKERS)),
        log_prior=np.zeros((NTEMPS, NWALKERS)),
    )
    state = GFState(eryn_state, is_eryn_state_input=True, sub_state_bases={"gb": GBState})
    betas = np.array([1.0, 0.5])[:NTEMPS]
    band_temps = np.tile(betas, (N_BANDS, 1))
    state.sub_states["gb"].initialize_band_information(NWALKERS, NTEMPS, band_edges, band_temps)

    model = GlobalFitInfo(
        analysis_container_arr=acs, map_fn=map, random=np.random.RandomState(seed)
    )

    orbits = EqualArmlengthOrbits(force_backend="cpu")
    move_kwargs = dict(
        rj_proposal_distribution=None,
        orbits=orbits,
        tdi_config=TDIConfig("1st generation"),
        t_ref=0.0,
        max_data_store_size=512,
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform,
        run_swaps=False,
        nfriends=NWALKERS,
        force_backend="cpu",
        provide_betas=True,
        num_repeat_proposals=3,
        periodic=PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}),
        debug=False,
    )
    move_args = (gb, priors, ind_lo, acs.data_length, acs, band_edges, band_N_vals, priors)

    tc = TemperatureControl(ndim, NWALKERS, betas=betas)

    return dict(
        gb=gb, priors=priors, transform=transform, acs=acs, model=model,
        state=state, band_edges=band_edges, band_N_vals=band_N_vals,
        move_args=move_args, move_kwargs=move_kwargs, temperature_control=tc,
        waveform_kwargs=waveform_kwargs, df=df, ind_lo=ind_lo,
    )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class BufferIdentityTest(unittest.TestCase):
    def test_load_equality_and_removal_identity(self):
        import numpy as np
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove
        from lisatools.globalfit.moves.gbbands import BandSorter, pack_special_index

        fx = build_fixture()
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_fixture",
            **{**fx["move_kwargs"], "rj_proposal_distribution": fx["priors"]},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0
        move.nwalkers, move.ntemps = NWALKERS, NTEMPS

        sorter = BandSorter(
            fx["state"].branches["gb"],
            move.band_edges,
            move.band_N_vals,
            force_backend="cpu",
            transform_fn=fx["transform"],
            max_data_store_size=512,
            gb=fx["gb"],
            gb_fd_comp=move.gb_fd_comp,
            waveform_kwargs=fx["waveform_kwargs"],
        )

        # Open all bands (units=1): parent central windows hold raw data.
        move.remove_cold_chain_sources_from_residual(
            fx["model"], sorter, units=1, remainder=0
        )
        try:
            # Cells: every (0, w, band) hosting a source + one empty cell.
            live_specials = np.unique(
                np.asarray(sorter.special_band_inds[sorter.inds & (sorter.temp_inds == 0)])
            )
            src_bands = set(np.asarray(sorter.band_inds[sorter.inds]).tolist())
            empty_band = next(b for b in range(1, N_BANDS - 1) if b not in src_bands)
            empty_special = int(pack_special_index(0, 0, empty_band, NWALKERS))
            specials = np.unique(np.concatenate([live_specials, [empty_special]]))

            buffer_obj = sorter.get_buffer(fx["acs"], specials)

            # ---- load equality on the empty cell ----
            slot = int(buffer_obj.get_index(np.asarray([empty_special]))[0])
            start = int(buffer_obj.buffer_start_index[slot]) - int(
                fx["acs"].start_freq_ind[0]
            )
            width = buffer_obj.band_buffer.shape[-1]
            parent_slice = fx["acs"].data_shaped[0][0, :, start:start + width]
            np.testing.assert_allclose(
                np.asarray(buffer_obj.band_buffer[slot]), np.asarray(parent_slice),
                rtol=0, atol=0,
                err_msg="empty-cell buffer load must equal the parent window slice",
            )

            # ---- removal identity on a live source ----
            src = np.where(np.asarray(sorter.inds & (sorter.temp_inds == 0)))[0][0]
            params = sorter.coords[src][None, :]
            special = sorter.special_band_inds[src][None]
            slot_arr = buffer_obj.get_index(special).astype(np.int32)
            N_arr = sorter.band_N_vals[sorter.band_inds[src]][None]

            ll_before = buffer_obj.band_likelihoods(source_only=True)[slot_arr[0]]
            delta_engine = float(
                buffer_obj.get_removal_ll(params, slot_arr, slot_arr, N_arr)[0]
            )
            buffer_obj.remove_sources_from_band_buffer(params, slot_arr, N_arr)
            ll_after = buffer_obj.band_likelihoods(source_only=True)[slot_arr[0]]
            buffer_obj.add_sources_to_band_buffer(params, slot_arr, N_arr)

            delta_direct = float(ll_after - ll_before)
            self.assertLess(
                abs(delta_direct - delta_engine) / max(abs(delta_engine), 1.0), 1e-6,
                msg=f"removal identity: direct {delta_direct} vs engine {delta_engine}",
            )
        finally:
            move.add_cold_chain_sources_to_residual(
                fx["model"], sorter, units=1, remainder=0
            )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ProposeFlowTest(unittest.TestCase):
    USE_FDOT_ASTRO = False
    USE_DISTANCE = False

    def _fixture(self):
        return build_fixture(use_fdot_astro=self.USE_FDOT_ASTRO,
                             use_distance=self.USE_DISTANCE)

    def _assert_basis(self, state):
        gb = np.asarray(state.branches["gb"].coords)
        nine = self.USE_FDOT_ASTRO or self.USE_DISTANCE
        self.assertEqual(gb.shape[-1], 9 if nine else 8)
        if nine:
            inds = np.asarray(state.branches["gb"].inds)
            # fdot_astro_ratio (col 8) stays inside the U[-M, M] prior box
            r = gb[..., 8][inds]
            if r.size:
                self.assertTrue(np.all(np.abs(r) <= 5.0 + 1e-9))
            if self.USE_DISTANCE:
                # slot 0 is a positive distance (kpc), not lnA
                d = gb[..., 0][inds]
                if d.size:
                    self.assertTrue(np.all(d > 0.0))

    def test_rj_propose(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove

        fx = self._fixture()
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_flow",
            **{**fx["move_kwargs"], "rj_proposal_distribution": fx["priors"]},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        new_state, accepted = move.propose(fx["model"], fx["state"])

        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        self._assert_basis(new_state)
        # every eligible source visited exactly once -> RJ proposals recorded
        band_info = new_state.sub_states["gb"].band_info
        self.assertGreater(int(band_info["band_num_proposed_rj"].sum()), 0)
        # in-model repeats also ran (deaths of real sources get rejected,
        # so picked sources stay alive and enter the repeat block)
        self.assertGreater(int(band_info["band_num_proposed"].sum()), 0)

    def test_rj_propose_cap_cell_grid(self):
        """The same RJ flow with the leaf caps on the CAP-CELL grid.

        Divisor 4 (user design 2026-08-15): caps enforced per quarter-band
        cell instead of per sub-band. Exercises the cap-cell occupancy
        census, the per-cell prior gate, the live-cap pick filter, the
        band-saturation budget transitions and the per-cell increment gate
        in one pass.
        """
        import numpy as np
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove
        from lisatools.globalfit.state import make_cap_edges

        K = 4
        fx = self._fixture()
        state = fx["state"]
        bi = state.sub_states["gb"].band_info
        band_edges = np.asarray(bi["band_edges"], dtype=float)
        bi["cap_edges"] = make_cap_edges(band_edges, K)
        bi["num_cap_cells"] = len(bi["cap_edges"]) - 1

        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_cap_cells",
            **{**fx["move_kwargs"],
               "rj_proposal_distribution": fx["priors"],
               "cap_divisor": K,
               "leaf_cap_start": 1,
               "leaf_cap_min_iters": 1},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        new_state, accepted = move.propose(fx["model"], state)

        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        self._assert_basis(new_state)
        nbi = new_state.sub_states["gb"].band_info
        num_bands = len(band_edges) - 1
        cell_cap = np.asarray(nbi["cap_cell_leaf_cap"])
        self.assertEqual(cell_cap.shape, (num_bands * K,))
        # armed at leaf_cap_start (possibly already incremented once)
        self.assertTrue(np.all(cell_cap >= 1))
        # the legacy per-band array stays written as the max over its cells
        np.testing.assert_array_equal(
            np.asarray(nbi["band_leaf_cap"]),
            cell_cap.reshape(num_bands, K).max(axis=1),
        )
        # the per-cell audit trace is populated
        self.assertEqual(
            np.asarray(nbi["cap_cell_cold_ll"]).shape[-1], num_bands * K
        )
        self.assertGreater(int(nbi["band_num_proposed_rj"].sum()), 0)

    def test_cap_grid_mismatch_raises(self):
        import numpy as np
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove

        fx = self._fixture()
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_cap_mismatch",
            **{**fx["move_kwargs"],
               "rj_proposal_distribution": fx["priors"],
               "cap_divisor": 4, "leaf_cap_start": 1},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0
        # the fixture state was built on the divisor-1 cap grid
        with self.assertRaises(ValueError) as ctx:
            move.propose(fx["model"], fx["state"])
        self.assertIn("cap grid mismatch", str(ctx.exception))

    def test_in_model_propose_two_passes(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

        fx = self._fixture()
        move = GBSpecialStretchMove(
            *fx["move_args"], is_rj_prop=False, name="stretch_flow",
            stretch_probability=0.5,
            **fx["move_kwargs"],
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        state_1, _ = move.propose(fx["model"], fx["state"])
        self.assertTrue(np.all(np.isfinite(state_1.log_like)))
        self._assert_basis(state_1)
        # source count must be unchanged by a pure in-model move
        self.assertEqual(
            int(state_1.branches["gb"].inds.sum()),
            int(fx["state"].branches["gb"].inds.sum()),
        )
        # second pass: group stretch now allowed (move.time >= 1)
        state_2, _ = move.propose(fx["model"], state_1)
        self.assertTrue(np.all(np.isfinite(state_2.log_like)))
        self.assertGreater(
            int(state_2.sub_states["gb"].band_info["band_num_proposed"].sum()), 0
        )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ProposeFlowFdotAstroTest(ProposeFlowTest):
    """The same propose flow on the 9-column fdot_astro ratio basis.

    Exercises the info-matrix Cholesky (with the ratio row zeroed) and the
    RJ birth/death with fdot<0 (interacting) templates from ratio < -1.
    """

    USE_FDOT_ASTRO = True


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ProposeFlowDistanceTest(ProposeFlowTest):
    """The same propose flow on the 9-column DISTANCE basis (slot 0 = dist).

    The stock default when the astrophysical prior is on: amplitude is
    derived from the sampled (dist, f0, Mc). Exercises the info-matrix
    Cholesky and RJ birth/death with the derived-amplitude quad transform.
    """

    USE_DISTANCE = True


if __name__ == "__main__":
    unittest.main()
