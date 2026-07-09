"""Flow tests for the GB special moves on the STFT (Fresnel) basis (CPU).

STFT mirror of ``test_gbspecial_flow.py`` — the first end-to-end exercise of
the restored STFT band-likelihood path through the REAL move flow (rather
than the engine/buffer unit tests):

1. **Buffer identities** — whole-grid cell loading equality against the
   parent walker slab (the STFT sub-band buffer is whole-grid parity: every
   (temp, walker, band) cell holds the walker's full active STFT grid) and
   the removal-delta identity ``get_removal_ll == ll(r+h) - ll(r)`` through
   an actual residual round-trip.
2. **RJ propose smoke** — ``GBSpecialRJPriorMove.propose`` runs the full new
   flow (without-replacement picks, RJ step, in-model repeats, interim
   STFT info-matrix Cholesky) on a state with real sources.
3. **In-model propose smoke** — ``GBSpecialStretchMove.propose`` twice so
   the second pass can draw the group stretch.
4. **RJ propose with phase maximization** — the two-quadrature analytic
   phase-max mixin driven through the real RJ flow.

The fixture is intentionally tiny (2 temps, 4 walkers, 8 x 6-h STFT
segments, 61 active frequency columns) and runs the production TDI
convention: XYZ with complex cross-channel inverse covariance through the
``gb_stft_*`` kernel family on CPU.

Everything skips unless gbgpu (with ``STFTGBComputations``) is importable.
"""

from __future__ import annotations

import unittest

import numpy as np


def _have_gbgpu_stft() -> bool:
    try:
        from gbgpu.gbcomps import STFTGBComputations  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


NTEMPS = 2
NWALKERS = 4
NLEAVES_MAX = 4
NDIM = 8
N_BANDS = 6
K_SOURCES = 2  # per (temp, walker)

# STFT grid: 6-h segments, 8 of them (Tobs = 2 d), anchored 10 d into the
# orbit span (away from the orbit-file t0 edge).
BIG_DT = 21600.0
NT_STFT = 8
NF_STFT = 128
DF_STFT = 1.0 / BIG_DT
T0_STFT = 10.0 * 86400.0
IND_MIN, IND_MAX = 40, 100  # active band [40, 100] * DF_STFT
BAND_COLS = 9  # STFT columns per GB band

# Inverse-covariance scale for the unit-diagonal invC. Probe on this grid:
# h_h(A = 1e-21, unit invC) = 3.35e-39, so this scale puts the fixture's
# amplitude prior [5e-22, 1e-20] at per-source SNR ~ 9 - 180.
INVC_SCALE = 1e41


def f_ms_to_s(x):
    return x * 1e-3


def build_fixture(seed=42):
    """Small STFT global-fit fixture with K_SOURCES live GBs per walker."""
    from gbgpu.gbcomps import STFTGBComputations
    from gbgpu.gbgpu import GBGPU

    from eryn.moves.tempering import TemperatureControl
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State as ErynState
    from eryn.utils import PeriodicContainer, TransformContainer

    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.domains import STFTSettings
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.globalfit.engine import GlobalFitInfo
    from lisatools.globalfit.state import GBState, GFState

    rng = np.random.default_rng(seed)  # noqa: F841 (parity with the FD fixture)
    np.random.seed(seed)

    settings = STFTSettings(
        t0=T0_STFT, dt=BIG_DT, df=DF_STFT, NT=NT_STFT, NF=NF_STFT,
        min_freq=IND_MIN * DF_STFT, max_freq=IND_MAX * DF_STFT,
        force_backend="cpu",
    )
    Tobs = NT_STFT * BIG_DT

    # GB bands: N_BANDS contiguous BAND_COLS-column windows, starting two
    # columns inside the active band. f0 prior inside the interior bands
    # (1 .. N_BANDS-2); the edge bands are excluded from proposals by the
    # move.
    band_width = BAND_COLS * DF_STFT
    band_edges = (settings.ind_min + 2) * DF_STFT + np.arange(N_BANDS + 1) * band_width
    assert band_edges[-1] < settings.ind_max * DF_STFT
    band_N_vals = np.full(N_BANDS, 64)

    f0_prior_lo = (band_edges[1] + 0.2 * band_width) * 1e3   # mHz
    f0_prior_hi = (band_edges[N_BANDS - 1] - 0.2 * band_width) * 1e3
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
    priors = {"gb": ProbDistContainer(priors_in)}

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
    waveform_kwargs = dict(dt=5.0, T=Tobs, tdi_channel_setup="XYZ")

    nch = 3
    data_shape = (nch, settings.NT, settings.NF_active)
    sens_shape = (nch, nch, settings.NT, settings.NF_active)
    orbits = EqualArmlengthOrbits()
    ac_list = []
    for _ in range(NWALKERS):
        res_data = np.zeros(data_shape, dtype=np.complex128)
        data_domain = settings.associated_class(res_data, settings)
        # The parent-group rebuild path needs a REAL sensitivity backend
        # (orbits + kwargs) — same construction the engine numeric test and
        # SubBandBuffer._build_band_ac_list use.
        sm = XYZSensitivityBackend(
            orbits=orbits, settings=settings, force_backend="cpu"
        )
        sm.sens_mat = np.zeros(sens_shape, dtype=np.complex128)
        invC = np.zeros(sens_shape, dtype=np.complex128)
        for j in range(nch):
            invC[j, j] = INVC_SCALE
        sm.invC = invC
        # Consistent determinant for the noise likelihood term:
        # det C = S^nch per pixel with C = diag(S, S, S), S = 1/INVC_SCALE.
        sm.detC = np.full(
            (settings.NT, settings.NF_active), (1.0 / INVC_SCALE) ** nch
        )
        sm.channel_shape = sens_shape[: -len(settings.basis_shape_active)]
        ac_list.append(AnalysisContainer(data_domain, sm))

    aca = AnalysisContainerArray(
        ac_list,
        gpus=None,
        domain_group_kwargs=dict(
            tdi_type="XYZ", window_alpha=0.0, use_midpoint=False
        ),
    )

    gb_orbits = EqualArmlengthOrbits(force_backend="cpu")
    gb = GBGPU(orbits=gb_orbits, force_backend="cpu")
    gb.gpus = None

    # Parent-level STFT computations: Fresnel knobs fixed at construction,
    # data/invC read through the group bound to ``stft_comps`` (the engine
    # rebinds per call).
    gb_stft_comp = STFTGBComputations(
        stft_comps=aca.cpp_splits[0],
        T=Tobs,
        t_ref=0.0,
        orbits=orbits,
        tdi_config=TDIConfig("1st generation"),
        force_backend="cpu",
        n_side_bins=3,
        window_factor=1.0,
        freq_from_tdi_phase=False,
    )

    # K_SOURCES live sources per walker, identical across temperatures.
    gb_coords = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX, NDIM))
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
        analysis_container_arr=aca, map_fn=map, random=np.random.RandomState(seed)
    )

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
        gb_stft_comp=gb_stft_comp,
    )
    move_args = (
        gb, priors, int(settings.ind_min), aca.data_length, aca,
        band_edges, band_N_vals, priors,
    )

    tc = TemperatureControl(NDIM, NWALKERS, betas=betas)

    return dict(
        gb=gb, gb_stft_comp=gb_stft_comp, priors=priors, transform=transform,
        acs=aca, model=model, state=state, band_edges=band_edges,
        band_N_vals=band_N_vals, move_args=move_args, move_kwargs=move_kwargs,
        temperature_control=tc, waveform_kwargs=waveform_kwargs,
        settings=settings,
    )


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class STFTBufferIdentityTest(unittest.TestCase):
    def test_load_equality_and_removal_identity(self):
        import numpy as np
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove
        from lisatools.globalfit.moves.gbbands import BandSorter, pack_special_index

        fx = build_fixture()
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_fixture_stft",
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
            gb_stft_comp=move.gb_stft_comp,
            waveform_kwargs=fx["waveform_kwargs"],
        )

        # Open all bands (units=1): parent slabs hold the cold-chain sources.
        move.remove_cold_chain_sources_from_residual(
            fx["model"], sorter, units=1, remainder=0
        )
        try:
            # The parent residual must actually hold the removed sources —
            # every walker has K_SOURCES live ones. Guards the load-equality
            # check below against a silently no-op STFT parent fill (zeros ==
            # zeros would pass it trivially).
            parent_walker_max = np.abs(np.asarray(fx["acs"].data_shaped[0])).max(
                axis=(1, 2, 3)
            )
            self.assertTrue(
                bool((parent_walker_max > 0).all()),
                msg=f"per-walker parent slab maxima: {parent_walker_max}",
            )
            # Cells: every (0, w, band) hosting a source + one empty cell.
            live_specials = np.unique(
                np.asarray(sorter.special_band_inds[sorter.inds & (sorter.temp_inds == 0)])
            )
            src_bands = set(np.asarray(sorter.band_inds[sorter.inds]).tolist())
            empty_band = next(b for b in range(1, N_BANDS - 1) if b not in src_bands)
            empty_special = int(pack_special_index(0, 0, empty_band, NWALKERS))
            specials = np.unique(np.concatenate([live_specials, [empty_special]]))

            buffer_obj = sorter.get_buffer(fx["acs"], specials)

            # ---- load equality on the empty cell (whole-grid slab) ----
            # The STFT band buffer is whole-grid parity: the cell's buffer is
            # the walker's FULL active (nch, NT, NF_active) parent slab.
            slot = int(buffer_obj.get_index(np.asarray([empty_special]))[0])
            parent_slab = fx["acs"].data_shaped[0][0]  # walker 0
            np.testing.assert_allclose(
                np.asarray(buffer_obj.band_buffer[slot]), np.asarray(parent_slab),
                rtol=0, atol=0,
                err_msg="empty-cell buffer load must equal the parent walker slab",
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

        # Round trip: adding the cold-chain sources back must empty the
        # residual again (data is zeros; ~1e-35 float cancellation noise is
        # 17 orders below the ~1e-18 slab signal).
        self.assertLess(
            float(np.abs(np.asarray(fx["acs"].data_shaped[0])).max()), 1e-30
        )


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class STFTProposeFlowTest(unittest.TestCase):
    def test_rj_propose(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove

        fx = build_fixture()
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_flow_stft",
            **{**fx["move_kwargs"], "rj_proposal_distribution": fx["priors"]},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        new_state, accepted = move.propose(fx["model"], fx["state"])

        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        # every eligible source visited exactly once -> RJ proposals recorded
        band_info = new_state.sub_states["gb"].band_info
        self.assertGreater(int(band_info["band_num_proposed_rj"].sum()), 0)
        # in-model repeats also ran (deaths of real sources get rejected,
        # so picked sources stay alive and enter the repeat block)
        self.assertGreater(int(band_info["band_num_proposed"].sum()), 0)

    def test_rj_propose_phase_maximize(self):
        """Two-quadrature analytic phase maximization through the real RJ flow."""
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove

        fx = build_fixture(seed=1234)
        move = GBSpecialRJPriorMove(
            *fx["move_args"], is_rj_prop=True, name="rj_flow_stft_pm",
            phase_maximize=True,
            **{**fx["move_kwargs"], "rj_proposal_distribution": fx["priors"]},
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        new_state, _ = move.propose(fx["model"], fx["state"])

        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        band_info = new_state.sub_states["gb"].band_info
        self.assertGreater(int(band_info["band_num_proposed_rj"].sum()), 0)

    def test_in_model_propose_two_passes(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

        fx = build_fixture()
        move = GBSpecialStretchMove(
            *fx["move_args"], is_rj_prop=False, name="stretch_flow_stft",
            stretch_probability=0.5,
            **fx["move_kwargs"],
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        state_1, _ = move.propose(fx["model"], fx["state"])
        self.assertTrue(np.all(np.isfinite(state_1.log_like)))
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


if __name__ == "__main__":
    unittest.main()
