"""Round-trip tests for the global-fit sub-state / sub-backend storage layer.

Phase-2 form (cold-chain storage rework): every sub-state owns the module's
full tempered ensemble (``chain``/``inds`` + per-branch log_like/log_prior +
delta counters) alongside its module extras (GB band_info, per-leaf
``betas_all``). These tests pin the on-disk schema (dataset names, shapes,
attrs), the save/load round-trip, the delta-counter semantics, the GFState
copy path, and the cold-row sync/check primitives.
"""

import os
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GBHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
    ModuleSubBackend,
    SOBBHHDFBackend,
)
from lisatools.globalfit.state import (
    EMRIState,
    GBState,
    GFState,
    MBHState,
    ModuleSubState,
    SOBBHState,
)

NTEMPS = 3
NWALKERS = 4
NUM_BANDS = 6
BAND_EDGES = np.linspace(1e-3, 7e-3, NUM_BANDS + 1)

BRANCH_SHAPES = {
    # branch: (nleaves_max, ndim)
    "gb": (5, 8),
    "mbh": (2, 11),
    "emri": (2, 12),
    "sobbh": (2, 11),
    "psd": (1, 4),
}

SUB_BACKENDS = {
    "gb": GBHDFBackend,
    "mbh": MBHHDFBackend,
    "emri": EMRIHDFBackend,
    "sobbh": SOBBHHDFBackend,
    "psd": ModuleSubBackend,
}

SUB_STATE_BASES = {
    "gb": GBState,
    "mbh": MBHState,
    "emri": EMRIState,
    "sobbh": SOBBHState,
    "psd": ModuleSubState,
}


def _tempered_schema(branch):
    """The standard tempered datasets for one branch (shapes after step axis)."""
    nleaves, ndim = BRANCH_SHAPES[branch]
    out = {
        "chain": (NTEMPS, NWALKERS, nleaves, ndim),
        "inds": (NTEMPS, NWALKERS, nleaves),
        # per-leaf cold-chain inner products (NaN = dead / not recorded)
        "d_h": (NWALKERS, nleaves),
        "h_h": (NWALKERS, nleaves),
    }
    if branch == "gb":
        # band_info carries the GB tempering record; no per-branch ll/counters
        return out
    if branch in ("mbh", "emri", "sobbh"):
        ll_shape = (nleaves, NTEMPS, NWALKERS)
        counter_shape = (nleaves, NTEMPS)
        swaps_shape = (nleaves, NTEMPS - 1)
    else:
        ll_shape = (NTEMPS, NWALKERS)
        counter_shape = (NTEMPS,)
        swaps_shape = (NTEMPS - 1,)
        # base branches carry the flat module ladder (GB uses band_temps,
        # per-leaf branches use betas_all)
        out["betas"] = (NTEMPS,)
    out.update(
        {
            "log_like": ll_shape,
            "log_prior": ll_shape,
            "in_model_proposed": counter_shape,
            "in_model_accepted": counter_shape,
            "rj_proposed": counter_shape,
            "rj_accepted": counter_shape,
            "swaps_proposed": swaps_shape,
            "swaps_accepted": swaps_shape,
        }
    )
    return out


# The exact per-branch datasets the sub-backends put on disk (shapes after
# the leading step axis).
EXPECTED_SUB_SCHEMA = {
    "gb": {
        "band_edges": (NUM_BANDS + 1,),
        "band_temps": (NUM_BANDS, NTEMPS),
        "band_swaps_proposed": (NUM_BANDS, NTEMPS - 1),
        "band_swaps_accepted": (NUM_BANDS, NTEMPS - 1),
        "band_num_proposed": (NUM_BANDS, NTEMPS),
        "band_num_accepted": (NUM_BANDS, NTEMPS),
        "band_num_proposed_rj": (NUM_BANDS, NTEMPS),
        "band_num_accepted_rj": (NUM_BANDS, NTEMPS),
        "band_num_binaries": (NTEMPS, NWALKERS, NUM_BANDS),
        "band_leaf_cap": (NUM_BANDS,),
        "band_cap_iters": (NUM_BANDS,),
        "band_best_ll": (NUM_BANDS,),
        # per-cold-walker per-band ll, stored every step (leaf-cap audit)
        "band_cold_ll": (NWALKERS, NUM_BANDS),
        **_tempered_schema("gb"),
    },
    "mbh": {
        "betas_all": (BRANCH_SHAPES["mbh"][0], NTEMPS),
        **_tempered_schema("mbh"),
    },
    "emri": {
        "betas_all": (BRANCH_SHAPES["emri"][0], NTEMPS),
        **_tempered_schema("emri"),
    },
    "sobbh": {
        "betas_all": (BRANCH_SHAPES["sobbh"][0], NTEMPS),
        **_tempered_schema("sobbh"),
    },
    "psd": _tempered_schema("psd"),
}

# band_edges is written once with the data (no step axis); everything else
# is a growable per-iteration dataset.
STATIC_DATASETS = {"gb": {"band_edges"}}


def make_state(rng):
    """Build a fully-populated GFState with all sub-states initialized."""
    coords = {
        name: rng.standard_normal((NTEMPS, NWALKERS, nleaves, ndim))
        for name, (nleaves, ndim) in BRANCH_SHAPES.items()
    }
    inds = {
        name: np.ones((NTEMPS, NWALKERS, nleaves), dtype=bool)
        for name, (nleaves, _) in BRANCH_SHAPES.items()
    }
    state = GFState(
        coords,
        inds=inds,
        log_like=rng.standard_normal((NTEMPS, NWALKERS)),
        log_prior=rng.standard_normal((NTEMPS, NWALKERS)),
        betas=np.linspace(1.0, 0.1, NTEMPS),
        random_state=np.random.get_state(),
        sub_state_bases=SUB_STATE_BASES,
    )

    band_temps = np.tile(np.linspace(1.0, 0.1, NTEMPS), (NUM_BANDS, 1))
    state.sub_states["gb"].initialize_band_information(
        NWALKERS, NTEMPS, BAND_EDGES, band_temps
    )
    for name in ("mbh", "emri", "sobbh"):
        nleaves = BRANCH_SHAPES[name][0]
        state.sub_states[name].betas_all = np.tile(
            np.linspace(1.0, 0.05, NTEMPS), (nleaves, 1)
        )
    # mirror the tempered ensembles into every sub-state
    for name, sub in state.sub_states.items():
        sub.pull_from_main(state, name)
    return state


class GFSubStateRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fp = os.path.join(self.tmpdir.name, "roundtrip_test.h5")
        self.rng = np.random.default_rng(1234)

        self.backend = GFHDFBackend(
            self.fp,
            sub_backend=dict(SUB_BACKENDS),
            sub_state_bases=dict(SUB_STATE_BASES),
        )
        ndims = {name: shape[1] for name, shape in BRANCH_SHAPES.items()}
        nleaves_max = {name: shape[0] for name, shape in BRANCH_SHAPES.items()}
        sub_reset_kwargs = {
            name: dict(nleaves_max=shape[0], ndim=shape[1])
            for name, shape in BRANCH_SHAPES.items()
        }
        sub_reset_kwargs["gb"].update(num_bands=NUM_BANDS, band_edges=BAND_EDGES)
        sub_reset_kwargs["mbh"].update(num_mbhs=BRANCH_SHAPES["mbh"][0])
        sub_reset_kwargs["emri"].update(num_emris=BRANCH_SHAPES["emri"][0])
        sub_reset_kwargs["sobbh"].update(num_sobbhs=BRANCH_SHAPES["sobbh"][0])
        self.backend.reset(
            NWALKERS,
            ndims,
            nleaves_max=nleaves_max,
            ntemps=NTEMPS,
            branch_names=list(BRANCH_SHAPES.keys()),
            nbranches=len(BRANCH_SHAPES),
            rj=False,
            moves=None,
            sub_reset_kwargs=sub_reset_kwargs,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_one(self, state):
        accepted = np.ones((NTEMPS, NWALKERS))
        self.backend.save_step(state, accepted)

    def test_on_disk_schema(self):
        """Dataset names/shapes under sub_backend/<branch> match the pinned schema."""
        import h5py

        self.backend.grow(1, None)
        with h5py.File(self.fp, "r") as f:
            sub = f["global_fit"]["sub_backend"]
            self.assertEqual(set(sub.keys()), set(EXPECTED_SUB_SCHEMA.keys()))
            for branch, datasets in EXPECTED_SUB_SCHEMA.items():
                grp = sub[branch]
                self.assertEqual(
                    set(grp.keys()), set(datasets.keys()), f"branch {branch}"
                )
                for dset_name, bare_shape in datasets.items():
                    on_disk = grp[dset_name].shape
                    if dset_name in STATIC_DATASETS.get(branch, ()):
                        self.assertEqual(on_disk, bare_shape, f"{branch}/{dset_name}")
                    else:
                        # growable: leading step axis then the bare shape
                        self.assertEqual(
                            on_disk, (1,) + bare_shape, f"{branch}/{dset_name}"
                        )
                # inds native bool, chain float
                self.assertEqual(grp["inds"].dtype, np.dtype(bool), branch)
                self.assertEqual(grp["chain"].dtype, np.dtype(float), branch)
                # tempered geometry attrs
                nleaves, ndim = BRANCH_SHAPES[branch]
                self.assertEqual(grp.attrs["ntemps"], NTEMPS)
                self.assertEqual(grp.attrs["nwalkers"], NWALKERS)
                self.assertEqual(grp.attrs["nleaves_max"], nleaves)
                self.assertEqual(grp.attrs["ndim"], ndim)
            # legacy count attrs
            self.assertEqual(sub["gb"].attrs["num_bands"], NUM_BANDS)
            self.assertEqual(sub["mbh"].attrs["num_mbhs"], BRANCH_SHAPES["mbh"][0])
            self.assertEqual(sub["emri"].attrs["num_emris"], BRANCH_SHAPES["emri"][0])
            self.assertEqual(
                sub["sobbh"].attrs["num_sobbhs"], BRANCH_SHAPES["sobbh"][0]
            )

    def test_save_and_reload_roundtrip(self):
        """Two saved iterations read back with get_a_sample identical to what went in."""
        self.backend.grow(2, None)

        state0 = make_state(self.rng)
        state0.sub_states["gb"].band_info["band_num_accepted"][:] = 7
        state0.sub_states["psd"].in_model_accepted[:] = 3
        self._save_one(state0)
        # delta semantics: counters zeroed on the live state after the save
        self.assertTrue(
            np.all(state0.sub_states["gb"].band_info["band_num_accepted"] == 0)
        )
        self.assertTrue(np.all(state0.sub_states["psd"].in_model_accepted == 0))

        state1 = make_state(self.rng)
        state1.sub_states["mbh"].betas_all *= 0.5
        self._save_one(state1)

        for it, state_in in ((0, state0), (1, state1)):
            state_out = self.backend.get_a_sample(it)
            for name, (nleaves, ndim) in BRANCH_SHAPES.items():
                np.testing.assert_allclose(
                    state_out.branches[name].coords,
                    state_in.branches[name].coords,
                    err_msg=f"coords mismatch branch {name} it {it}",
                )
                # sub-state tempered ensemble round-trips too
                np.testing.assert_allclose(
                    state_out.sub_states[name].coords,
                    state_in.sub_states[name].coords,
                    err_msg=f"sub-state coords mismatch branch {name} it {it}",
                )
                np.testing.assert_array_equal(
                    state_out.sub_states[name].inds,
                    state_in.sub_states[name].inds,
                    err_msg=f"sub-state inds mismatch branch {name} it {it}",
                )
            for name in ("mbh", "emri", "sobbh"):
                np.testing.assert_allclose(
                    state_out.sub_states[name].betas_all,
                    state_in.sub_states[name].betas_all,
                    err_msg=f"betas_all mismatch branch {name} it {it}",
                )
            # GBHDFBackend keeps the leading step axis on the band_info
            # arrays (stripped later by initialize_band_information's
            # rank-based logic) -- pin that behavior here.
            np.testing.assert_allclose(
                state_out.sub_states["gb"].band_info["band_temps"][0],
                state_in.sub_states["gb"].band_info["band_temps"],
                err_msg=f"band_temps mismatch it {it}",
            )

    def test_reset_kwargs_roundtrip(self):
        """Sub-backend reset_kwargs read back from disk (incl. the EMRI/SOBBH attrs fix).

        Queried per sub-backend rather than through GFHDFBackend.reset_kwargs:
        eryn's HDFBackend.reset_kwargs references an undefined ``self.moves``
        (latent upstream bug), so the merged property cannot be used here.
        """
        sub = self.backend.sub_backend
        gb_kwargs = sub["gb"].reset_kwargs
        self.assertEqual(gb_kwargs["num_bands"], NUM_BANDS)
        np.testing.assert_allclose(gb_kwargs["band_edges"], BAND_EDGES)
        self.assertEqual(sub["mbh"].reset_kwargs["num_mbhs"], BRANCH_SHAPES["mbh"][0])
        self.assertEqual(
            sub["emri"].reset_kwargs["num_emris"], BRANCH_SHAPES["emri"][0]
        )
        self.assertEqual(
            sub["sobbh"].reset_kwargs["num_sobbhs"], BRANCH_SHAPES["sobbh"][0]
        )
        for name, (nleaves, ndim) in BRANCH_SHAPES.items():
            kwargs = sub[name].reset_kwargs
            self.assertEqual(kwargs["ntemps"], NTEMPS, name)
            self.assertEqual(kwargs["nwalkers"], NWALKERS, name)
            self.assertEqual(kwargs["nleaves_max"], nleaves, name)
            self.assertEqual(kwargs["ndim"], ndim, name)

    def test_gfstate_copy_preserves_substates(self):
        """GFState(state, copy=True) deep-copies sub-states incl. tempered blocks."""
        state = make_state(self.rng)
        copied = GFState(state, copy=True)

        self.assertEqual(copied.sub_states["mbh"].num_mbhs, BRANCH_SHAPES["mbh"][0])
        self.assertEqual(copied.sub_states["emri"].num_emris, BRANCH_SHAPES["emri"][0])
        self.assertEqual(
            copied.sub_states["sobbh"].num_sobbhs, BRANCH_SHAPES["sobbh"][0]
        )

        # deep copy: mutating the copy must not touch the original
        copied.sub_states["mbh"].betas_all[:] = -1.0
        self.assertFalse(np.any(state.sub_states["mbh"].betas_all == -1.0))
        copied.sub_states["gb"].band_info["band_temps"][:] = -1.0
        self.assertFalse(
            np.any(state.sub_states["gb"].band_info["band_temps"] == -1.0)
        )
        for name in BRANCH_SHAPES:
            self.assertTrue(copied.sub_states[name].tempered_initialized, name)
            copied.sub_states[name].coords[:] = -99.0
            self.assertFalse(
                np.any(state.sub_states[name].coords == -99.0), name
            )

    def test_cold_row_check_and_sync(self):
        """check_cold_row trips on divergence; sync_cold_row repairs it."""
        state = make_state(self.rng)
        sub = state.sub_states["gb"]

        sub.check_cold_row(state, "gb")  # consistent after pull_from_main

        # a move that updates the sub-state without the main state
        sub.coords[0, 0, 0, 0] += 1.0
        with self.assertRaises(ValueError):
            sub.check_cold_row(state, "gb")

        sub.sync_cold_row(state, "gb")
        sub.check_cold_row(state, "gb")

        # inds divergence trips too
        sub.inds[0, 0, 0] = False
        with self.assertRaises(ValueError):
            sub.check_cold_row(state, "gb")
        sub.sync_cold_row(state, "gb")
        sub.check_cold_row(state, "gb")


if __name__ == "__main__":
    unittest.main()
