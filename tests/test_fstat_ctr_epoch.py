"""Epoch F-stat CENTER table (user ruling 2026-08-15): build the center
distributions ONCE when the F-stat birth distribution is fitted, smear them,
look them up by f0 at propose time.

Light fakes in the style of test_fstat_ctr_fallback.py: no comps, no buffers.
The node sweep is driven by a deterministic stand-in ``call_fstat`` that
returns (N, M) with M = identity, so the Jaranowski-Krol inversion the sweep
shares with production has a closed form (N = [n,0,0,0] -> A_max = 2|n|,
F = n^2/2).
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove
from lisatools.sampling.fstat_gridfit import (
    CENTER_TABLE_BASENAME,
    GRID_BASENAME,
    build_fstat_center_table,
    enumerate_center_nodes,
    run_center_sweep,
)

# M4 = identity in the row-major upper-triangle layout the F-stat returns.
_M_EYE = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0])


def _n_of_f0(f0_mHz):
    """Deterministic first filter overlap as a function of f0 [mHz]."""
    return 10.0 + np.asarray(f0_mHz, dtype=float)


class _FakeFstat:
    """``call_fstat(params_phys) -> (N, M)``, counting its calls/rows."""

    def __init__(self):
        self.calls = []

    def __call__(self, params):
        p = np.asarray(params, dtype=float)
        self.calls.append(int(p.shape[0]))
        N = np.zeros((p.shape[0], 4))
        N[:, 0] = _n_of_f0(p[:, 1] * 1e3)  # params carry f0 in Hz
        return N, np.tile(_M_EYE, (p.shape[0], 1))


def _write_stacked(cache_dir, f0_los, f0_dxs, n_f0, mc_ax, al_ax, sd_ax,
                   grids):
    np.savez(
        os.path.join(cache_dir, GRID_BASENAME.replace(
            ".npz", "_peaks_stacked.npz")),
        logp_grids=grids, f0_los=np.asarray(f0_los, dtype=float),
        f0_dxs=np.asarray(f0_dxs, dtype=float), mc_ax=np.asarray(mc_ax),
        alpha_ax=np.asarray(al_ax), sin_delta_ax=np.asarray(sd_ax),
    )


def _write_comb(cache_dir, f0_nodes, best_alpha=None, best_sd=None):
    kw = {}
    if best_alpha is not None:
        kw = dict(best_alpha=np.asarray(best_alpha),
                  best_sin_delta=np.asarray(best_sd))
    np.savez(
        os.path.join(cache_dir, GRID_BASENAME.replace(".npz", "_comb.npz")),
        f0_nodes_mHz=np.asarray(f0_nodes, dtype=float),
        F_max=np.ones(len(f0_nodes)), **kw)


class _EpochDirCase(unittest.TestCase):
    """One temp epoch dir holding a 2-box grid cache + a 4-node comb cache."""

    #: box 0 nodes 3.000/3.010/3.020 mHz, box 1 nodes 5.000/5.004 ... mHz
    F0_LOS = [3.0, 5.0]
    F0_DXS = [0.01, 0.004]
    N_F0 = 3
    MC_AX = [0.1, 0.4, 0.9]
    AL_AX = [0.0, 1.0]
    SD_AX = [-0.5, 0.5]
    COMB_F0 = [2.0, 4.0, 6.0, 8.0]
    COMB_AL = [0.11, 0.22, 0.33, 0.44]
    COMB_SD = [-0.9, -0.3, 0.3, 0.9]

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fstat_ctr_epoch_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        os.environ["FSTAT_COMB_MC"] = "0.25"
        self.addCleanup(os.environ.pop, "FSTAT_COMB_MC", None)
        # grids: put a UNIQUE argmax in the (Mc, alpha, sin_delta) block of
        # every (box, f0 node) so the enumeration's argmax is checkable.
        shape = (2, self.N_F0, len(self.MC_AX), len(self.AL_AX),
                 len(self.SD_AX))
        g = np.zeros(shape)
        self.argmax_idx = {}
        for k in range(2):
            for i in range(self.N_F0):
                j = ((k + i) % len(self.MC_AX), (k + i) % len(self.AL_AX),
                     (k + i + 1) % len(self.SD_AX))
                g[(k, i) + j] = 100.0 + k + i
                self.argmax_idx[(k, i)] = j
        _write_stacked(self.dir, self.F0_LOS, self.F0_DXS, self.N_F0,
                       self.MC_AX, self.AL_AX, self.SD_AX, g)
        _write_comb(self.dir, self.COMB_F0, self.COMB_AL, self.COMB_SD)

    def peak_f0(self):
        return np.concatenate([
            np.asarray(self.F0_LOS)[k] + np.arange(self.N_F0)
            * np.asarray(self.F0_DXS)[k] for k in range(2)])


class NodeEnumerationTest(_EpochDirCase):
    def test_support_is_peak_nodes_plus_comb_nodes(self):
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0])
        self.assertEqual(nodes["n_peak_nodes"], 2 * self.N_F0)
        self.assertEqual(nodes["n_comb_nodes"], len(self.COMB_F0))
        np.testing.assert_allclose(
            np.sort(nodes["f0_mHz"]),
            np.sort(np.concatenate([self.peak_f0(), self.COMB_F0])))
        # sorted by f0 (the lookup searchsorts on it)
        np.testing.assert_array_equal(nodes["f0_mHz"],
                                      np.sort(nodes["f0_mHz"]))

    def test_peak_nodes_carry_the_grid_argmax(self):
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0])
        for k in range(2):
            for i in range(self.N_F0):
                f0 = self.F0_LOS[k] + i * self.F0_DXS[k]
                w = int(np.argmin(np.abs(nodes["f0_mHz"] - f0)))
                j_mc, j_al, j_sd = self.argmax_idx[(k, i)]
                self.assertAlmostEqual(nodes["mc"][w], self.MC_AX[j_mc])
                self.assertAlmostEqual(nodes["alpha"][w], self.AL_AX[j_al])
                self.assertAlmostEqual(nodes["sin_delta"][w],
                                       self.SD_AX[j_sd])

    def test_comb_nodes_use_scan_best_sky_and_fixed_mc(self):
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0])
        for f0, al, sd in zip(self.COMB_F0, self.COMB_AL, self.COMB_SD):
            w = int(np.argmin(np.abs(nodes["f0_mHz"] - f0)))
            self.assertAlmostEqual(nodes["alpha"][w], al)
            self.assertAlmostEqual(nodes["sin_delta"][w], sd)
            self.assertAlmostEqual(nodes["mc"][w], 0.25)  # FSTAT_COMB_MC

    def test_max_nodes_thins_the_comb_and_keeps_every_peak_node(self):
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0],
                                       max_nodes=8)
        self.assertEqual(nodes["n_peak_nodes"], 2 * self.N_F0)
        self.assertLess(nodes["n_comb_nodes"], len(self.COMB_F0))
        for f0 in self.peak_f0():
            self.assertTrue(np.isclose(nodes["f0_mHz"], f0).any())

    def test_comb_only_epoch(self):
        os.remove(os.path.join(self.dir, GRID_BASENAME.replace(
            ".npz", "_peaks_stacked.npz")))
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0])
        self.assertEqual(nodes["n_peak_nodes"], 0)
        np.testing.assert_allclose(nodes["f0_mHz"], self.COMB_F0)

    def test_no_caches_means_no_support(self):
        empty = tempfile.mkdtemp(prefix="fstat_ctr_empty_")
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        nodes = enumerate_center_nodes(empty)
        self.assertEqual(int(nodes["f0_mHz"].size), 0)


class CenterSweepTest(unittest.TestCase):
    def _nodes(self, f0):
        f0 = np.asarray(f0, dtype=float)
        return dict(f0_mHz=f0, mc=np.full(f0.shape, 0.3),
                    alpha=np.zeros_like(f0), sin_delta=np.zeros_like(f0))

    def test_sweep_matches_the_shared_jk_inversion(self):
        f0 = np.array([1.0, 2.0, 3.0])
        out = run_center_sweep(_FakeFstat(), self._nodes(f0), xp=np)
        n = _n_of_f0(f0)
        np.testing.assert_allclose(out["ln_A_max"], np.log(2.0 * n))
        ln_snr = 0.5 * np.log(np.clip(n ** 2, 1.0, None))
        np.testing.assert_allclose(out["ln_snr"], ln_snr)
        np.testing.assert_allclose(out["sigma_base"], np.exp(-ln_snr))
        # the reference-basis maxima are finite and in range
        self.assertTrue(np.all((out["phi0"] >= 0) & (out["phi0"] < 2 * np.pi)))
        self.assertTrue(np.all((out["iota"] >= 0) & (out["iota"] <= np.pi)))

    def test_batching_covers_every_row_identically(self):
        f0 = np.linspace(1.0, 5.0, 5)
        one = run_center_sweep(_FakeFstat(), self._nodes(f0), xp=np)
        fake = _FakeFstat()
        many = run_center_sweep(fake, self._nodes(f0), xp=np, batch=2)
        self.assertEqual(fake.calls, [2, 2, 1])
        for key in one:
            np.testing.assert_allclose(many[key], one[key], err_msg=key)

    def test_empty_support(self):
        out = run_center_sweep(_FakeFstat(), self._nodes([]), xp=np)
        self.assertEqual(int(out["ln_snr"].size), 0)


class CenterTableCacheTest(_EpochDirCase):
    def test_build_persists_and_reload_never_recomputes(self):
        fake = _FakeFstat()
        tbl = build_fstat_center_table(fake, cache_dir=self.dir, xp=np,
                                       mc_lims=[0.001, 1.0])
        self.assertTrue(os.path.exists(
            os.path.join(self.dir, CENTER_TABLE_BASENAME)))
        n_calls = len(fake.calls)
        self.assertGreater(n_calls, 0)
        # checkpoint-load path: same numbers, no sweep
        again = build_fstat_center_table(fake, cache_dir=self.dir, xp=np,
                                         mc_lims=[0.001, 1.0])
        self.assertEqual(len(fake.calls), n_calls)
        for key in ("f0_mHz", "ln_A_max", "ln_snr", "phi0", "iota", "psi",
                    "sigma_base", "mc", "alpha", "sin_delta"):
            np.testing.assert_allclose(again[key], tbl[key], err_msg=key)

    def test_missing_table_for_an_old_epoch_recomputes(self):
        fake = _FakeFstat()
        tbl = build_fstat_center_table(fake, cache_dir=self.dir, xp=np,
                                       mc_lims=[0.001, 1.0])
        os.remove(os.path.join(self.dir, CENTER_TABLE_BASENAME))
        fake2 = _FakeFstat()
        rebuilt = build_fstat_center_table(fake2, cache_dir=self.dir, xp=np,
                                           mc_lims=[0.001, 1.0])
        self.assertGreater(len(fake2.calls), 0)
        np.testing.assert_allclose(rebuilt["ln_A_max"], tbl["ln_A_max"])

    def test_values_are_the_sweep_of_the_enumerated_support(self):
        tbl = build_fstat_center_table(_FakeFstat(), cache_dir=self.dir,
                                       xp=np, mc_lims=[0.001, 1.0])
        nodes = enumerate_center_nodes(self.dir, mc_lims=[0.001, 1.0])
        np.testing.assert_allclose(tbl["f0_mHz"], nodes["f0_mHz"])
        np.testing.assert_allclose(
            tbl["ln_A_max"], np.log(2.0 * _n_of_f0(nodes["f0_mHz"])))

    def test_no_support_returns_none(self):
        empty = tempfile.mkdtemp(prefix="fstat_ctr_empty_")
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        self.assertIsNone(build_fstat_center_table(
            _FakeFstat(), cache_dir=empty, xp=np))

    def test_missing_table_without_call_fstat_is_none(self):
        self.assertIsNone(build_fstat_center_table(
            None, cache_dir=self.dir, xp=np, mc_lims=[0.001, 1.0]))


# ---------------------------------------------------------------------------
# propose-time lookup (move level)
# ---------------------------------------------------------------------------

def _move(basis="lnA", table=None, mode=None):
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = "test"
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"
    m.transform_fn = SimpleNamespace(input_basis=[basis, "f0_ms", "fdot"])
    m._fstat_ctr_table = table
    if mode is not None:
        os.environ["GB_FSTAT_CTR_MODE"] = mode
    return m


def _table(f0_nodes, ln_A_max=None, ln_snr=None):
    f0_nodes = np.asarray(f0_nodes, dtype=float)
    n = len(f0_nodes)
    ln_A = np.arange(n, dtype=float) if ln_A_max is None else np.asarray(
        ln_A_max, dtype=float)
    ls = (np.log(np.linspace(20.0, 60.0, n)) if ln_snr is None
          else np.asarray(ln_snr, dtype=float))
    return dict(
        f0_mHz=f0_nodes, ln_A_max=ln_A, ln_snr=ls, sigma_base=np.exp(-ls),
        phi0=np.linspace(0.0, 6.0, n), iota=np.linspace(0.1, 3.0, n),
        psi=np.linspace(0.2, 3.0, n),
    )


def _rows(f0_mHz, mc=0.3, ndim=8):
    f0_mHz = np.atleast_1d(np.asarray(f0_mHz, dtype=float))
    p = np.zeros((len(f0_mHz), ndim))
    p[:, 1] = f0_mHz
    p[:, 2] = mc
    return p


class _ModeEnv(unittest.TestCase):
    def setUp(self):
        for k in ("GB_FSTAT_CTR_MODE", "GB_FSTAT_CTR_SMEAR"):
            self.addCleanup(os.environ.pop, k, None)
            os.environ.pop(k, None)


class ModeAndSmearTest(_ModeEnv):
    def test_epoch_is_the_default_mode(self):
        self.assertEqual(_move()._fstat_ctr_mode(), "epoch")

    def test_smear_default_is_mode_dependent(self):
        self.assertAlmostEqual(_move(mode="epoch")._fstat_ctr_smear(), 2.0)
        self.assertAlmostEqual(_move(mode="unit")._fstat_ctr_smear(), 1.5)

    def test_env_override_wins_in_both_modes(self):
        os.environ["GB_FSTAT_CTR_SMEAR"] = "3.25"
        self.assertAlmostEqual(_move(mode="epoch")._fstat_ctr_smear(), 3.25)
        self.assertAlmostEqual(_move(mode="unit")._fstat_ctr_smear(), 3.25)

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            _move(mode="somethingelse")._fstat_ctr_mode()

    def test_unit_mode_never_activates_the_table(self):
        m = _move(table=_table([1.0, 2.0]), mode="unit")
        self.assertIsNone(m._fstat_ctr_table_active())

    def test_epoch_mode_without_a_table_falls_back(self):
        self.assertIsNone(_move(mode="epoch")._fstat_ctr_table_active())

    def test_unit_mode_compute_is_unchanged(self):
        # pre-change behavior: sigma = 1.5 / SNR from the per-row F-stat
        m = _move(mode="unit")
        m._fstat_dist_centers = lambda model, params, ref: (
            1.0 + 0.01 * params[:, 1], 0.1 * params[:, 1],
            0.2 + params[:, 1] * 0, 0.3 + params[:, 1] * 0,
            50.0 + params[:, 1])
        rows = _rows([10.0, 11.0, 12.0])
        phi0, iota, psi, ln_center, sigma, ln_snr = m._fstat_ctr_compute(
            None, rows)
        A = 1.0 + 0.01 * rows[:, 1]
        F = 50.0 + rows[:, 1]
        snr = np.sqrt(np.clip(2.0 * F, 1.0, None))
        np.testing.assert_allclose(ln_center, np.log(A))
        np.testing.assert_allclose(sigma, 1.5 / snr)
        np.testing.assert_allclose(ln_snr, np.log(snr))


class TableLookupTest(_ModeEnv):
    def test_nearest_node_including_the_edges(self):
        t = _table([1.0, 2.0, 4.0])
        m = _move(table=t)
        # below the first node, exact hits, midpoint tie (-> lower node),
        # nearer-upper, above the last node
        q = np.array([0.1, 1.0, 2.0, 3.0, 3.4, 9.0])
        _, _, _, _, _, ln_snr = m._fstat_ctr_table_lookup(_rows(q))
        expect = t["ln_snr"][[0, 0, 1, 1, 2, 2]]
        np.testing.assert_allclose(ln_snr, expect)

    def test_single_node_table(self):
        t = _table([7.0])
        out = _move(table=t)._fstat_ctr_table_lookup(_rows([1.0, 7.0, 90.0]))
        np.testing.assert_allclose(out[5], np.repeat(t["ln_snr"], 3))

    def test_angles_and_center_come_from_the_nearest_node(self):
        t = _table([1.0, 2.0, 4.0])
        m = _move(mode="epoch", table=t)
        phi0, iota, psi, ln_center, sigma, ln_snr = \
            m._fstat_ctr_table_lookup(_rows([3.9]))
        self.assertAlmostEqual(float(phi0[0]), t["phi0"][2])
        self.assertAlmostEqual(float(iota[0]), t["iota"][2])
        self.assertAlmostEqual(float(psi[0]), t["psi"][2])
        # amplitude basis: ln_center is the node's own ln A_max
        self.assertAlmostEqual(float(ln_center[0]), t["ln_A_max"][2])
        # sigma = smear / SNR, with SNR read back from the table exactly
        self.assertAlmostEqual(float(sigma[0]),
                               2.0 * float(np.exp(-t["ln_snr"][2])))
        self.assertAlmostEqual(float(ln_snr[0]), t["ln_snr"][2])

    def test_sigma_tracks_the_smear_knob(self):
        t = _table([1.0, 2.0])
        os.environ["GB_FSTAT_CTR_SMEAR"] = "1.0"
        raw = _move(table=t)._fstat_ctr_table_lookup(_rows([2.0]))[4]
        os.environ["GB_FSTAT_CTR_SMEAR"] = "4.0"
        wide = _move(table=t)._fstat_ctr_table_lookup(_rows([2.0]))[4]
        np.testing.assert_allclose(wide, 4.0 * raw)

    def test_no_fstat_evaluation_at_propose_time(self):
        t = _table([1.0, 2.0, 4.0])
        m = _move(table=t)

        def _boom(*a, **kw):  # pragma: no cover - must never be reached
            raise AssertionError("the epoch table must not evaluate F-stat")

        m._fstat_dist_centers = _boom
        m._fstat_ctr_table_lookup(_rows([1.3, 2.2, 3.9]))


class TableDetailedBalanceTest(_ModeEnv):
    """Birth draw and death reverse density read the SAME table entries."""

    def test_birth_and_death_share_the_table_at_the_same_f0(self):
        t = _table([1.0, 2.0, 4.0])
        m = _move(basis="dist", table=t)
        rows = _rows([1.4, 2.6, 3.95])
        birth = m._fstat_ctr_table_lookup(rows)
        death = m._fstat_ctr_table_lookup(rows)
        for b, d in zip(birth, death):
            np.testing.assert_array_equal(b, d)

    def test_density_symmetry_of_a_drawn_birth(self):
        t = _table([1.0, 2.0, 4.0])
        m = _move(basis="dist", table=t)
        rows = _rows([1.4, 2.6, 3.95])
        _, _, _, ln_center, sigma, ln_snr = m._fstat_ctr_table_lookup(rows)
        alpha = m._snr_trunc_alpha(ln_snr, sigma, 5.0)
        z = m._truncnorm_std_draw(len(rows), alpha)
        v = np.exp(ln_center + sigma * z)          # stored slot-0 value
        fwd = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        # the death side re-derives (ln_center, sigma, alpha) from the table
        # at the SAME f0 -- nothing about the leaf's slot 0 enters the lookup
        rows_d = rows.copy()
        rows_d[:, 0] = v
        _, _, _, lc_d, sig_d, ls_d = m._fstat_ctr_table_lookup(rows_d)
        rev = m._slot0_log_proposal(
            v, lc_d, sig_d, alpha=m._snr_trunc_alpha(ls_d, sig_d, 5.0))
        self.assertTrue(np.all(fwd > -1e299))
        np.testing.assert_array_equal(fwd, rev)

    def test_snr_truncation_boundary_uses_the_table_ln_snr(self):
        # SNR 50 at the node, limit 5 -> alpha = ln(10) / sigma
        t = _table([1.0, 2.0], ln_snr=np.log([50.0, 50.0]))
        os.environ["GB_FSTAT_CTR_SMEAR"] = "1.0"
        m = _move(basis="dist", table=t)
        _, _, _, _, sigma, ln_snr = m._fstat_ctr_table_lookup(_rows([2.0]))
        np.testing.assert_allclose(sigma, 1.0 / 50.0)
        alpha = m._snr_trunc_alpha(ln_snr, sigma, 5.0)
        np.testing.assert_allclose(alpha, np.log(10.0) * 50.0)
        # and the truncated draw respects it
        z = m._truncnorm_std_draw(256, np.full(256, float(alpha[0])))
        self.assertLessEqual(float(np.max(z)), float(alpha[0]) + 1e-9)

    def test_mc_scaling_is_per_row_not_per_node(self):
        # distance basis: ln_center carries amp_from_dist(f0, Mc, 1) / A_max,
        # so two rows at the SAME node but different Mc must NOT share a
        # center (the Mc**(5/3) term spans ~11 e-folds across the prior).
        t = _table([2.0])
        m = _move(basis="dist", table=t)
        c_lo = m._fstat_ctr_table_lookup(_rows([2.0], mc=0.05))[3]
        c_hi = m._fstat_ctr_table_lookup(_rows([2.0], mc=0.9))[3]
        self.assertGreater(abs(float(c_hi[0]) - float(c_lo[0])), 1.0)
        # ... while sigma (F-stat curvature) is the node's
        s_lo = m._fstat_ctr_table_lookup(_rows([2.0], mc=0.05))[4]
        s_hi = m._fstat_ctr_table_lookup(_rows([2.0], mc=0.9))[4]
        np.testing.assert_allclose(s_lo, s_hi)


if __name__ == "__main__":
    unittest.main()
