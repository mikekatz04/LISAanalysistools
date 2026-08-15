"""Tests for scripts/fstat_proposal/migrate_gb_band_edges.py.

Builds a synthetic small global-fit h5 with the GB sub-backend's per-band
arrays (leading step axis, GBState.initialize_band_information shapes) and
checks the migration: band_temps interpolated in frequency per temperature
rung, counters and derived arrays reset, band_leaf_cap back to the -1
re-arm sentinel, the static band_edges swapped, the num_bands attr updated,
and the .bak backup written.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fstat_proposal", "migrate_gb_band_edges.py",
)

NSTEPS, NT, NW = 3, 4, 2
NB_OLD = 6
OLD_EDGES = 1e-3 + 1e-4 * np.arange(NB_OLD + 1)  # uniform 1-layer-like grid
# variable-width layer-aligned new grid over the same span (4 bands)
NEW_EDGES = 1e-3 + 1e-4 * np.array([0.0, 1.0, 2.0, 4.0, 6.0])
NB_NEW = len(NEW_EDGES) - 1


def make_store(path):
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        sub = f.create_group("global_fit").create_group(
            "sub_backend").create_group("gb")
        sub.attrs["num_bands"] = NB_OLD
        sub.attrs["ntemps"] = NT
        sub.attrs["nwalkers"] = NW
        sub.create_dataset("band_edges", data=OLD_EDGES)

        def growable(name, shape, data):
            sub.create_dataset(name, data=data, maxshape=(None,) + shape[1:])

        band_temps = rng.uniform(0.1, 1.0, (NSTEPS, NB_OLD, NT))
        growable("band_temps", band_temps.shape, band_temps)
        for name in ("band_swaps_proposed", "band_swaps_accepted"):
            growable(name, (NSTEPS, NB_OLD, NT - 1),
                     rng.integers(0, 50, (NSTEPS, NB_OLD, NT - 1)))
        for name in ("band_num_proposed", "band_num_accepted",
                     "band_num_proposed_rj", "band_num_accepted_rj"):
            growable(name, (NSTEPS, NB_OLD, NT),
                     rng.integers(0, 50, (NSTEPS, NB_OLD, NT)))
        growable("band_num_binaries", (NSTEPS, NT, NW, NB_OLD),
                 rng.integers(0, 5, (NSTEPS, NT, NW, NB_OLD)))
        growable("band_leaf_cap", (NSTEPS, NB_OLD),
                 rng.integers(1, 9, (NSTEPS, NB_OLD)))
        growable("band_cap_iters", (NSTEPS, NB_OLD),
                 rng.integers(0, 5, (NSTEPS, NB_OLD)))
        growable("band_best_ll", (NSTEPS, NB_OLD),
                 rng.normal(size=(NSTEPS, NB_OLD)))
        growable("band_cold_ll", (NSTEPS, NW, NB_OLD),
                 rng.normal(size=(NSTEPS, NW, NB_OLD)))
        # non-band datasets must pass through untouched
        sub.create_dataset("chain", data=rng.normal(size=(NSTEPS, NT, NW, 5, 9)),
                           maxshape=(None, NT, NW, 5, 9))
    return band_temps


class MigrateGBBandEdgesTest(unittest.TestCase):
    def _run(self, tmp, extra_args=()):
        h5_path = os.path.join(tmp, "store.h5")
        old_band_temps = make_store(h5_path)
        edges_npy = os.path.join(tmp, "new_edges.npy")
        np.save(edges_npy, NEW_EDGES)
        env = dict(os.environ)
        for var in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
            env[f"{var}_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        proc = subprocess.run(
            [sys.executable, SCRIPT, h5_path, "--edges-npy", edges_npy,
             *extra_args],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return h5_path, old_band_temps

    def test_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path, old_bt = self._run(tmp)
            self.assertTrue(os.path.exists(h5_path + ".bak"))

            with h5py.File(h5_path, "r") as f:
                sub = f["global_fit"]["sub_backend"]["gb"]
                self.assertEqual(int(sub.attrs["num_bands"]), NB_NEW)
                np.testing.assert_array_equal(sub["band_edges"][:], NEW_EDGES)

                # band_temps: per-rung frequency interpolation, every step
                bt = sub["band_temps"][:]
                self.assertEqual(bt.shape, (NSTEPS, NB_NEW, NT))
                ctr_old = 0.5 * (OLD_EDGES[:-1] + OLD_EDGES[1:])
                ctr_new = 0.5 * (NEW_EDGES[:-1] + NEW_EDGES[1:])
                for s in range(NSTEPS):
                    for t in range(NT):
                        np.testing.assert_allclose(
                            bt[s, :, t],
                            np.interp(ctr_new, ctr_old, old_bt[s, :, t]),
                        )

                # counters zeroed at the new size
                for name, band_axis in (
                    ("band_swaps_proposed", 1), ("band_swaps_accepted", 1),
                    ("band_num_proposed", 1), ("band_num_accepted", 1),
                    ("band_num_proposed_rj", 1), ("band_num_accepted_rj", 1),
                    ("band_num_binaries", 3),
                ):
                    arr = sub[name][:]
                    self.assertEqual(arr.shape[band_axis], NB_NEW, name)
                    self.assertTrue(np.all(arr == 0), name)

                # progressive caps re-arm sentinel; plateau state reset
                self.assertTrue(np.all(sub["band_leaf_cap"][:] == -1))
                self.assertEqual(sub["band_leaf_cap"].shape, (NSTEPS, NB_NEW))
                self.assertTrue(np.all(sub["band_cap_iters"][:] == 0))
                self.assertTrue(np.all(np.isneginf(sub["band_best_ll"][:])))
                self.assertTrue(np.all(np.isneginf(sub["band_cold_ll"][:])))
                self.assertEqual(sub["band_cold_ll"].shape,
                                 (NSTEPS, NW, NB_NEW))

                # growability preserved (resume appends steps)
                self.assertEqual(sub["band_temps"].maxshape,
                                 (None, NB_NEW, NT))

                # non-band data untouched
                self.assertEqual(sub["chain"].shape, (NSTEPS, NT, NW, 5, 9))

            # backup holds the original
            with h5py.File(h5_path + ".bak", "r") as f:
                sub = f["global_fit"]["sub_backend"]["gb"]
                self.assertEqual(int(sub.attrs["num_bands"]), NB_OLD)
                np.testing.assert_array_equal(sub["band_edges"][:], OLD_EDGES)

    def test_refuses_existing_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = os.path.join(tmp, "store.h5")
            make_store(h5_path)
            open(h5_path + ".bak", "w").close()
            edges_npy = os.path.join(tmp, "new_edges.npy")
            np.save(edges_npy, NEW_EDGES)
            proc = subprocess.run(
                [sys.executable, SCRIPT, h5_path, "--edges-npy", edges_npy],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing to overwrite", proc.stderr + proc.stdout)

    def test_refuses_identical_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = os.path.join(tmp, "store.h5")
            make_store(h5_path)
            edges_npy = os.path.join(tmp, "same_edges.npy")
            np.save(edges_npy, OLD_EDGES)
            proc = subprocess.run(
                [sys.executable, SCRIPT, h5_path, "--edges-npy", edges_npy],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("identical", proc.stderr + proc.stdout)
            self.assertFalse(os.path.exists(h5_path + ".bak"))


if __name__ == "__main__":
    unittest.main()
