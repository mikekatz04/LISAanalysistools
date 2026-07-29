"""Round-trip tests for the global-fit sub-state / sub-backend storage layer.

Written against the storage behavior as of the start of the cold-chain
storage rework (branch ``cold-chain-storage``): these tests pin the on-disk
schema (dataset names, shapes, attrs) and the save/load round-trip semantics
of the per-branch sub-backends, so the re-expression of the concrete
sub-backends on the ``ModuleSubBackend`` base can be verified to be
byte-compatible.
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
    SOBBHHDFBackend,
)
from lisatools.globalfit.state import (
    EMRIState,
    GBState,
    GFState,
    MBHState,
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
    "psd": None,
}

SUB_STATE_BASES = {
    "gb": GBState,
    "mbh": MBHState,
    "emri": EMRIState,
    "sobbh": SOBBHState,
    "psd": None,
}

# The exact per-branch datasets the sub-backends put on disk (shapes after
# the leading step axis). Phase 1 of the storage rework must reproduce this
# schema exactly for existing branches.
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
    },
    "mbh": {"betas_all": (BRANCH_SHAPES["mbh"][0], NTEMPS)},
    "emri": {"betas_all": (BRANCH_SHAPES["emri"][0], NTEMPS)},
    "sobbh": {"betas_all": (BRANCH_SHAPES["sobbh"][0], NTEMPS)},
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
        self.backend.reset(
            NWALKERS,
            ndims,
            nleaves_max=nleaves_max,
            ntemps=NTEMPS,
            branch_names=list(BRANCH_SHAPES.keys()),
            nbranches=len(BRANCH_SHAPES),
            rj=False,
            moves=None,
            sub_reset_kwargs={
                "gb": dict(num_bands=NUM_BANDS, band_edges=BAND_EDGES),
                "mbh": dict(num_mbhs=BRANCH_SHAPES["mbh"][0]),
                "emri": dict(num_emris=BRANCH_SHAPES["emri"][0]),
                "sobbh": dict(num_sobbhs=BRANCH_SHAPES["sobbh"][0]),
            },
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
        self._save_one(state0)
        # delta semantics: counters zeroed on the live state after the save
        self.assertTrue(
            np.all(state0.sub_states["gb"].band_info["band_num_accepted"] == 0)
        )

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
            for name in ("mbh", "emri", "sobbh"):
                np.testing.assert_allclose(
                    state_out.sub_states[name].betas_all,
                    state_in.sub_states[name].betas_all,
                    err_msg=f"betas_all mismatch branch {name} it {it}",
                )
            # GBHDFBackend.get_a_sample keeps the leading step axis on the
            # band_info arrays (stripped later by initialize_band_information's
            # rank-based logic) -- pin that behavior here.
            np.testing.assert_allclose(
                state_out.sub_states["gb"].band_info["band_temps"][0],
                state_in.sub_states["gb"].band_info["band_temps"],
                err_msg=f"band_temps mismatch it {it}",
            )
            self.assertIsNone(state_out.sub_states["psd"])

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

    def test_gfstate_copy_preserves_substates(self):
        """GFState(state, copy=True) deep-copies sub-states and leaf counts."""
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


if __name__ == "__main__":
    unittest.main()
