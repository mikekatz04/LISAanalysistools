"""The two PE F-stat moves (user ruling 2026-10-12).

Mike, 2026-10-12: "I would like to do two fstat moves that will split their
share of the total moves. One fstat move fits the GB-free residual so the
whole galaxy down to SNR 6.5. [...] The second fstat move will fit the
gb-inclusive residual every 150 iterations. These iteration counts are the
overall iterations, so the first time after the overall iterations pass a
number, it should refit the first time that proposal is chosen." Then: "set
the gb-free residual refit (with the full galaxy found) to not refitting
during full pe. It should just fit once the first time it is visiting in full
pe. The residual fstat will take care of low snr adjustments." And: "this
should not be specific to each walker, just specific to the one walker that
was randomly drawn."

  rj_fstat_pe            GB-free residual, root ``shared`` (unchanged), fit
                         ONCE (GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0), its own
                         floor GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE=6.5 (a cut on a
                         loaded epoch's cached peaks, never a refit)
  rj_fstat_pe_residual   live residual, one random cold walker per epoch, root
                         ``shared_pe_residual``, refit every
                         GB_FSTAT_REFIT_EVERY_PE=150 elapsed iterations, the
                         stage's floor

The 9mo resume requirement (2026-10-12): the galaxy move must PICK UP the PE
epoch_0000 already on disk as its one fit and never refit -- see
``GalaxyResumeLoadTest``.

CPU only, bare moves (``__new__``, no ctor), a real epoch on disk built by the
library fit with an analytic scorer (KB-sized). The draw table is in
``test_gf_combine_pe_rj_draw_one``, the composition in
``test_search_stage_recipe_knobs``, the launchers in
``test_submit_scripts_layout``.
"""

import inspect
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit import recipe as R
from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    GBSpecialRJFStatGridMove,
)
from lisatools.sampling import fstat_gridfit as G
from lisatools.sampling import fstat_proposal as FP

#: the knobs this module is about, cleared unless a test sets them
_KNOBS = ("GB_FSTAT_REFIT_EVERY_PE", "GB_FSTAT_REFIT_EVERY_PE_GB_FREE",
          "GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE", "GB_PE_RJ_FSTAT_RESIDUAL_SPLIT",
          "GB_FSTAT_PE_REF", "GB_FSTAT_PE_REF_SEED", "GB_FSTAT_SEARCH_REF",
          "GB_FSTAT_GB_FREE",
          "GB_FSTAT_FIT_PER_MOVE", "GB_FSTAT_FORCE_REFIT", "FSTAT_PEAK_MIN_SNR",
          "FSTAT_PEAK_MIN_F")

BAND_EDGES = np.linspace(6.2e-3, 7.0e-3, 6)   # 5 sub-bands, 3 interior
F0_LIMS = (float(BAND_EDGES[1]), float(BAND_EDGES[-2]))
#: (centre mHz, peak F): two bumps in each interior band, quiet and loud
BUMPS = ((6.40, 12.0), (6.45, 30.0), (6.56, 18.0), (6.62, 45.0),
         (6.72, 22.0), (6.78, 14.0))


def _bump_call_fstat():
    """``params -> (N, M_upper)``: F = sum of Gaussian bumps in f0 (identity M,
    so F = 0.5 |N|^2 exactly), the same shape as test_fstat_gridfit's fake."""
    def call(params):
        p = np.asarray(params.get() if hasattr(params, "get") else params)
        f0_mHz = p[:, 1] * 1e3
        amp = np.zeros(p.shape[0])
        for c, a in BUMPS:
            amp += a * np.exp(-0.5 * ((f0_mHz - c) / 4e-3) ** 2)
        N = np.zeros((p.shape[0], 4))
        N[:, 0] = np.sqrt(2.0 * np.maximum(amp, 0.0))
        M = np.zeros((p.shape[0], 10))
        M[:, 0] = M[:, 4] = M[:, 7] = M[:, 9] = 1.0
        return N, M
    return call


_FIT_ENV = {
    "FSTAT_BATCH": "512", "FSTAT_CKPT_SECS": "0", "FSTAT_F0_SPACING_MHZ": "0.002",
    "FSTAT_COMB_NSKY": "2", "FSTAT_PEAKS_PER_BAND": "2", "FSTAT_N_MC": "2",
    "FSTAT_N_ALPHA": "2", "FSTAT_N_SINDELTA": "2", "FSTAT_PEAK_HALF_MHZ": "0.02",
    "FSTAT_FDOT_AXIS": "0",
}


class _Isolated(unittest.TestCase):
    """Fresh fit dir, empty F-stat registries, the class-level refit clock and
    the process-global peak-floor override saved and restored, env knobs
    cleared."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pe2fstat_")
        self.addCleanup(shutil.rmtree, self.root, True)
        for reg in (gbs._FSTAT_GRID_REGISTRY, gbs._FSTAT_CTR_TABLE_REGISTRY):
            p = mock.patch.dict(reg, {}, clear=True)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in _KNOBS:
            os.environ.pop(k, None)
        saved = [(d, dict(d)) for d in (
            GBSpecialBase._branch_propose_counts,
            GBSpecialBase._branch_iteration_counts,
            GBSpecialBase._branch_iteration_seen,
            GBSpecialRJFStatGridMove._fstat_clock_written)]
        seeded = set(GBSpecialRJFStatGridMove._fstat_clock_seeded)
        for d, _ in saved:
            d.clear()
        GBSpecialRJFStatGridMove._fstat_clock_seeded.clear()
        prev_floor = FP.peak_min_F_override()
        FP._PEAK_MIN_F_OVERRIDE = None     # no stage floor unless a test sets one

        def _restore():
            for d, v in saved:
                d.clear()
                d.update(v)
            GBSpecialRJFStatGridMove._fstat_clock_seeded.clear()
            GBSpecialRJFStatGridMove._fstat_clock_seeded.update(seeded)
            FP._PEAK_MIN_F_OVERRIDE = prev_floor
        self.addCleanup(_restore)


def _bare(fit_dir, name, *, search=False, cfg=None):
    """A bare F-stat grid move configured the way build_gb_moves configures it
    (``pe_fstat_move_config`` for the two PE moves, the flag for a search one)."""
    m = GBSpecialRJFStatGridMove.__new__(GBSpecialRJFStatGridMove)
    m.name = name
    m.branch_name = "gb"
    m._backend_name = "lisatools_cpu"
    m.fstat_fit_dir = fit_dir
    m.fstat_fit_kwargs = dict(mc_lims=[0.01, 1.0], A_lims=[1e-24, 1e-21])
    m.fstat_refit_every = 0
    m.fstat_peak_min_snr = None
    m.df = 1.0 / 7.776e6
    m.band_edges = BAND_EDGES.copy()
    m.is_rj_prop = True
    m.rj_replace = False
    m.rj_removal_only = False
    m.priors = {"gb": "PRIOR"}
    m.gpu_priors = {"gb": "GPU_PRIOR"}
    m.rj_proposal_distribution = None
    m._fstat_epoch = None
    m._fstat_last_fit_hit = -1
    m._fstat_zero_peak_epoch = None
    m.gf_iteration = None
    m.gf_rank = None
    m.num_proposals = 0
    if search:
        m.fstat_search_residual = True
    if cfg is not None:
        m.fstat_refit_every = cfg["fstat_refit_every"]
        m.fstat_peak_min_snr = cfg["fstat_peak_min_snr"]
        m.fstat_pe_residual = cfg["fstat_pe_residual"]
    # heavy neighbours setup() calls but these tests do not exercise
    m._band_shutoff_epoch_sync = lambda: 0
    m._install_ctr_table = lambda k, model=None, branches=None: None
    m._fstat_release_ref_row = lambda: None
    m._free_inmodel_batch_pools = lambda model, why: None
    return m


def _galaxy(fit_dir):
    return _bare(fit_dir, "rj_fstat_pe", cfg=R.pe_fstat_move_config("rj_fstat_pe"))


def _residual(fit_dir):
    return _bare(fit_dir, R.PE_FSTAT_RESIDUAL_MOVE,
                 cfg=R.pe_fstat_move_config(R.PE_FSTAT_RESIDUAL_MOVE))


# ======================================================================
# 1. the knobs and the per-move configuration
# ======================================================================
class ConfigTest(_Isolated):

    def test_defaults_are_the_ruling(self):
        gal = R.pe_fstat_move_config("rj_fstat_pe")
        res = R.pe_fstat_move_config("rj_fstat_pe_residual")
        self.assertEqual(gal, dict(fstat_refit_every=0, fstat_peak_min_snr=6.5,
                                   fstat_pe_residual=False))
        self.assertEqual(res, dict(fstat_refit_every=150, fstat_peak_min_snr=None,
                                   fstat_pe_residual=True))
        # rj_replace_pe shares the galaxy's root, so it shares its config
        self.assertEqual(R.pe_fstat_move_config("rj_replace_pe"), gal)
        self.assertEqual(R.PE_FSTAT_RESIDUAL_MOVE, "rj_fstat_pe_residual")

    def test_knobs_reach_the_config(self):
        os.environ.update(GB_FSTAT_REFIT_EVERY_PE="40",
                          GB_FSTAT_REFIT_EVERY_PE_GB_FREE="500",
                          GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE="7")
        self.assertEqual(R.pe_fstat_move_config("rj_fstat_pe")["fstat_refit_every"], 500)
        self.assertEqual(R.pe_fstat_move_config("rj_fstat_pe")["fstat_peak_min_snr"], 7.0)
        self.assertEqual(
            R.pe_fstat_move_config("rj_fstat_pe_residual")["fstat_refit_every"], 40)
        os.environ["GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE"] = "none"
        self.assertIsNone(R.pe_fstat_move_config("rj_fstat_pe")["fstat_peak_min_snr"])

    def test_bad_values_are_refused(self):
        for knob, bad in (("GB_FSTAT_REFIT_EVERY_PE_GB_FREE", "-1"),
                          ("GB_FSTAT_REFIT_EVERY_PE", "x"),
                          ("GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE", "0"),
                          ("GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE", "abc")):
            with mock.patch.dict(os.environ, {knob: bad}):
                with self.assertRaises(ValueError, msg=(knob, bad)):
                    R.pe_fstat_move_config("rj_fstat_pe")
                    R.pe_fstat_move_config("rj_fstat_pe_residual")

    def test_build_gb_moves_builds_both_from_the_config(self):
        """Structural (a real build_gb_moves needs a built fit): both PE F-stat
        moves are constructed from ``pe_fstat_move_config``, the residual one
        carries the mode flag and is in the PE list right after the galaxy."""
        src = inspect.getsource(R.build_gb_moves)
        self.assertIn('_pe_fstat_gb_free = pe_fstat_move_config("rj_fstat_pe")', src)
        self.assertIn(
            "_pe_fstat_residual = pe_fstat_move_config(PE_FSTAT_RESIDUAL_MOVE)", src)
        i_gal = src.index('name="rj_fstat_pe",')
        self.assertIn("**_pe_fit_kwargs(_pe_fstat_gb_free)", src[i_gal - 400:i_gal])
        i_res = src.index("name=PE_FSTAT_RESIDUAL_MOVE,")
        self.assertIn("**_pe_fit_kwargs(_pe_fstat_residual)", src[i_res - 400:i_res])
        self.assertIn('gb_pe_residual_move.fstat_pe_residual = '
                      '_pe_fstat_residual["fstat_pe_residual"]', src)
        self.assertIn("gb_pe_moves = [gb_pe_prior_move, gb_pe_residual_move]", src)
        # the galaxy move is NOT put in either live-residual mode
        self.assertNotIn("gb_pe_prior_move.fstat_pe_residual", src)
        self.assertNotIn("gb_pe_prior_move.fstat_search_residual", src)
        # the old shared PE cadence variable is gone (it fed BOTH moves)
        self.assertNotIn("_refit_pe", src)
        # the fit kwargs carry the move's floor to the grid class
        self.assertIn('out["fstat_peak_min_snr"] = cfg["fstat_peak_min_snr"]', src)


# ======================================================================
# 2. modes: residual swept, reference walker, epoch root
# ======================================================================
class ModeTest(_Isolated):

    def test_galaxy_is_exactly_todays_rj_fstat_pe(self):
        """GB-free window open, root ``shared`` (where the stores' PE epochs
        live), the PE random walker rule, no per-band floor / skip in the
        fingerprint -- nothing that names or keys its epochs changed."""
        g = _galaxy(self.root)
        self.assertTrue(g._fstat_gb_free_on())
        self.assertEqual(g._fstat_root, os.path.join(self.root, "shared"))
        self.assertEqual(g._fstat_fit_ref_rule(), "random")
        self.assertEqual(gbs.fstat_stage_fingerprint_for(g), "")
        self.assertFalse(g.fstat_pe_residual)
        self.assertFalse(g.fstat_search_residual)
        # the floor is NOT in the fit fingerprint (the fingerprint is epoch /
        # gbfree / wref + the per-band stage terms, never the move's floor)
        fit_src = inspect.getsource(GBSpecialRJFStatGridMove._run_fstat_fit)
        self.assertNotIn("fstat_peak_min_snr", fit_src)
        self.assertIn('f"|epoch={k}|gbfree={int(_gb_free)}|wref={w_global}"', fit_src)

    def test_residual_sweeps_the_live_residual_on_the_pe_rule_in_its_own_root(self):
        r = _residual(self.root)
        self.assertFalse(r._fstat_gb_free_on())             # GBs stay subtracted
        os.environ["GB_FSTAT_GB_FREE"] = "1"                # a stray env cannot reopen it
        self.assertFalse(r._fstat_gb_free_on())
        self.assertEqual(r._fstat_fit_ref_rule(), "random")  # ONE random walker
        self.assertEqual(r._fstat_root, os.path.join(self.root, "shared_pe_residual"))
        # and the search moves are untouched by the PE knob: live residual,
        # the ROTATING walker (2026-10-16; GB_FSTAT_SEARCH_REF), own root
        s = _bare(self.root, "rj_fstat_search", search=True)
        self.assertFalse(s._fstat_gb_free_on())
        self.assertEqual(s._fstat_fit_ref_rule(), "rotate")
        self.assertEqual(s._fstat_root, os.path.join(self.root, "shared_search"))
        os.environ["GB_FSTAT_PE_REF"] = "max"
        self.assertEqual(s._fstat_fit_ref_rule(), "rotate")
        self.assertEqual(r._fstat_fit_ref_rule(), "max")    # follows the PE knob
        os.environ["GB_FSTAT_SEARCH_REF"] = "min"
        self.assertEqual(s._fstat_fit_ref_rule(), "min")    # the search knob
        self.assertEqual(r._fstat_fit_ref_rule(), "max")    # not the PE moves

    def test_three_distinct_roots(self):
        roots = {m._fstat_root for m in (
            _galaxy(self.root), _residual(self.root),
            _bare(self.root, "rj_fstat_search", search=True))}
        self.assertEqual(len(roots), 3)

    def test_one_walker_per_epoch_not_one_fit_per_walker(self):
        self.assertEqual(_residual(self.root).fstat_fit_walkers, 1)

    def test_reference_walker_streams(self):
        """Galaxy: the (seed, epoch) stream bit-for-bit (a resumed store
        re-derives the walker its epochs were fitted to). Residual: the same
        seed with the residual tag -- one random walker per epoch, drawn
        independently of the galaxy's."""
        g, r = _galaxy(self.root), _residual(self.root)
        lls = np.array([-5.0, -2.0, -9.0, -1.0, -3.0, -7.0])
        tag = GBSpecialBase._FSTAT_PE_RESIDUAL_REF_TAG
        gal, res = [], []
        for k in range(40):
            want_g = int(np.random.default_rng([0, k]).integers(6))
            want_r = int(np.random.default_rng([0, k, tag]).integers(6))
            self.assertEqual(g._fstat_fit_ref_from(lls, k), want_g, k)
            self.assertEqual(r._fstat_fit_ref_from(lls, k), want_r, k)
            gal.append(want_g)
            res.append(want_r)
        self.assertNotEqual(gal, res)
        self.assertGreater(len(set(res)), 2)      # random, not a fixed walker
        # the per-propose distance-birth centre (no epoch) keeps the argmin
        self.assertEqual(r._fstat_fit_ref_from(lls), 2)


# ======================================================================
# 3. the refit clock: residual every 150 overall iterations, galaxy never
# ======================================================================
class ClockTest(_Isolated):
    """Simulate full_pe: the move is drawn on a random subset of iterations
    (P = 0.45 * 0.5); the decision runs only when it is drawn, on the
    restart-persistent clock that counts ELAPSED iterations."""

    def _run(self, m, n_iter=1200, p=0.225, seed=3):
        rng = np.random.default_rng(seed)
        draws = np.flatnonzero(rng.uniform(size=n_iter) < p)
        fits, last_fit_it = [], None
        for it in draws:
            m.gf_iteration = int(it)
            action, k = m._fstat_fit_decision()
            if action == "fit":
                fits.append(int(it))
                # what setup() leaves behind after a fit
                m.rj_proposal_distribution = {"gb": "GRID"}
                m._fstat_epoch = k
                m._fstat_last_fit_hit = m._fstat_clock()
        return draws, fits

    def test_residual_refits_at_the_first_draw_at_or_after_150(self):
        r = _residual(self.root)
        draws, fits = self._run(r)
        # first draw fits epoch 0 (nothing on disk), then: the first DRAW whose
        # iteration is >= 150 past the previous fit, never earlier
        want, last = [], None
        for it in draws:
            if last is None or it - last >= 150:
                want.append(int(it))
                last = int(it)
        self.assertEqual(fits, want)
        self.assertGreaterEqual(len(fits), 6)
        gaps = np.diff(fits)
        self.assertTrue(np.all(gaps >= 150), gaps)
        self.assertTrue(np.all(gaps < 150 + 40), gaps)   # first draw after: a few its late

    def test_galaxy_fits_once_and_never_again(self):
        g = _galaxy(self.root)
        draws, fits = self._run(g)
        self.assertEqual(fits, [int(draws[0])])

    def test_the_galaxy_cadence_is_what_stops_it(self):
        """Mutation guard: the SAME harness refits a galaxy move given the
        residual's cadence -- so the never-refit above is the knob, not a
        harness that cannot see refits."""
        g = _galaxy(self.root)
        g.fstat_refit_every = 150
        _, fits = self._run(g)
        self.assertGreater(len(fits), 1)


# ======================================================================
# 4. the 9mo resume: the galaxy LOADS the stored epoch, never refits
# ======================================================================
class _FitCalled(Exception):
    pass


class GalaxyResumeLoadTest(_Isolated):
    """A galaxy-mode move over a fit dir that already holds a complete PE
    ``shared/epoch_0000`` (fitted at a LOWER floor, as the 9mo's at F >= 19.53)
    loads it as its one fit, cuts it to its own floor, and never fits."""

    STORE_FLOOR_F = 10.0     # the epoch's selection floor (stamped)
    MOVE_FLOOR_F = 20.0      # the galaxy's own floor in this fixture

    def setUp(self):
        super().setUp()
        os.environ.update(_FIT_ENV)
        os.environ["FSTAT_PEAK_MIN_F"] = str(self.STORE_FLOOR_F)
        self.fit_dir = os.path.join(self.root, "gb_fstat_fit")
        self.epoch = os.path.join(self.fit_dir, "shared", "epoch_0000")
        _, n_peaks = G.run_fstat_grid_fit(
            _bump_call_fstat(), xp=np, Tobs=7.776e6, band_edges_hz=BAND_EDGES,
            f0_lims_hz=F0_LIMS, mc_lims=[0.01, 1.0], cache_dir=self.epoch,
            fingerprint_extra="|epoch=0|gbfree=1|wref=3")
        with open(os.path.join(self.epoch, "DONE.json"), "w") as f:
            json.dump(dict(epoch=0, walker_ref=3, fitted_here=True,
                           n_peaks=int(n_peaks), clock=117), f)
        os.environ.pop("FSTAT_PEAK_MIN_F")
        with np.load(G.stacked_grid_path(self.epoch)) as z:
            self.peak_F = np.asarray(z["peak_F"], dtype=float)
            self.stamp = float(z["peak_min_F"])
        self.n_all = int(self.peak_F.size)
        self.n_keep = int((self.peak_F >= self.MOVE_FLOOR_F).sum())
        # the stage's declared floor is in force (full_pe: 6.25)
        FP.set_peak_min_F_override(6.25)
        os.environ["GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE"] = repr(
            float(np.sqrt(2.0 * self.MOVE_FLOOR_F)))

    def _move(self):
        m = _galaxy(self.fit_dir)

        def _no_fit(*a, **kw):
            raise _FitCalled("the galaxy move tried to FIT")
        m._run_fstat_fit = _no_fit
        return m

    @staticmethod
    def _boxes(m):
        return sum(c.K for c in FP.iter_stacked_components(
            m.rj_proposal_distribution["gb"]))

    def test_the_fixture_is_a_real_lower_floor_epoch(self):
        self.assertAlmostEqual(self.stamp, self.STORE_FLOOR_F)
        self.assertGreater(self.n_all, self.n_keep)
        self.assertGreater(self.n_keep, 0)

    def test_first_visit_loads_epoch_0000_and_never_fits(self):
        m = self._move()
        self.assertEqual(m._fstat_fit_decision(), ("load", 0))
        m.gf_iteration = 0
        m.setup(None, None)                               # _FitCalled would propagate
        self.assertEqual(m._fstat_epoch, 0)
        self.assertIsNotNone(m.rj_proposal_distribution)
        self.assertEqual(m._fstat_last_fit_hit, 117)      # the stored fit's clock
        # the cut: only the cached boxes at the galaxy's own floor, no refit
        self.assertEqual(self._boxes(m), self.n_keep)
        # nothing new on disk: no epoch_0001, the epoch dir untouched
        self.assertEqual(sorted(os.listdir(os.path.join(self.fit_dir, "shared"))),
                         ["epoch_0000"])
        # and the stage's floor is back for every other reader
        self.assertAlmostEqual(FP.fstat_peak_min_F(), 0.5 * 6.25 ** 2)
        # every later visit keeps it, however far the clock runs
        for it in range(1, 5000, 37):
            m.gf_iteration = it
            m.setup(None, None)
        self.assertEqual(m._fstat_epoch, 0)
        self.assertEqual(sorted(os.listdir(os.path.join(self.fit_dir, "shared"))),
                         ["epoch_0000"])

    def test_mutation_guard_a_cadence_would_refit_it(self):
        """The same fixture with the galaxy given the residual's 150: a refit is
        attempted once the clock passes it -- the harness sees refits, so the
        test above passing means the 0 cadence, not a blind harness."""
        m = self._move()
        m.fstat_refit_every = 150
        m.gf_iteration = 0
        m.setup(None, None)
        with self.assertRaises(_FitCalled):
            for it in range(1, 400):
                m.gf_iteration = it
                m.setup(None, None)

    def test_mutation_guard_a_different_root_would_fit(self):
        """A galaxy move that landed in any root but ``shared`` (e.g. the
        residual's) finds no epoch and fits -- so the load above is the root."""
        m = self._move()
        m.fstat_pe_residual = True
        with self.assertRaises(_FitCalled):
            m.setup(None, None)

    def test_an_incomplete_later_epoch_is_ignored_by_a_fit_once_move(self):
        """The 9mo ran the 150 cadence between 811cf945 and the two-move
        change, so its ``shared/`` may hold a half-fitted ``epoch_0001`` next
        to the complete ``epoch_0000``. A fit-once galaxy move must load the
        complete epoch and leave the half-fitted directory alone, not spend
        hours finishing a sweep it would never refit."""
        partial = os.path.join(self.fit_dir, "shared", "epoch_0001")
        os.makedirs(os.path.join(partial, "fstat_grid_parts"))
        with open(os.path.join(partial, "fstat_grid_parts", "comb_nsky16_r0.progress.npz"), "wb") as f:
            f.write(b"not a complete epoch")
        m = self._move()
        self.assertEqual(m._fstat_fit_decision(), ("load", 0))
        m.gf_iteration = 0
        m.setup(None, None)                               # _FitCalled would propagate
        self.assertEqual(m._fstat_epoch, 0)
        self.assertEqual(self._boxes(m), self.n_keep)
        for it in range(1, 2000, 41):
            m.gf_iteration = it
            m.setup(None, None)
        self.assertEqual(m._fstat_epoch, 0)
        # the partial directory is untouched
        self.assertEqual(os.listdir(os.path.join(partial, "fstat_grid_parts")),
                         ["comb_nsky16_r0.progress.npz"])

    def test_a_move_with_a_cadence_still_resumes_the_incomplete_epoch(self):
        """Mutation guard for the test above: the pre-existing semantics (a
        mid-fit death resumes the same epoch) are unchanged for any move
        with a cadence -- only the fit-once move skips the partial epoch."""
        partial = os.path.join(self.fit_dir, "shared", "epoch_0001")
        os.makedirs(partial)
        m = self._move()
        m.fstat_refit_every = 150
        self.assertEqual(m._fstat_fit_decision(), ("fit", 1))

    def test_without_a_floor_of_its_own_every_box_loads(self):
        m = self._move()
        m.fstat_peak_min_snr = None
        m.setup(None, None)
        self.assertEqual(self._boxes(m), self.n_all)

    def test_a_rank_install_cuts_the_same_boxes(self):
        """The rank-side load (``_setup_from_directive`` -> ``_install``) applies
        the same cut from the same move attribute: head and ranks agree."""
        m = self._move()
        m._install(0, sync_shutoff=False)
        self.assertEqual(self._boxes(m), self.n_keep)
        self.assertEqual(gbs._FSTAT_GRID_REGISTRY[m._epoch_dir(0)][2], self.n_keep)


# ======================================================================
# 5. the per-move floor around a FIT
# ======================================================================
class FitFloorScopeTest(_Isolated):

    def test_scope_sets_and_restores_exactly(self):
        FP.set_peak_min_F_override(6.25)
        before = FP.peak_min_F_override()
        with FP.peak_min_snr_scope(6.5):
            self.assertEqual(FP.fstat_peak_min_F(), 0.5 * 6.5 ** 2)
        self.assertEqual(FP.peak_min_F_override(), before)    # bit-exact
        with FP.peak_min_snr_scope(None):
            self.assertEqual(FP.peak_min_F_override(), before)
        with self.assertRaises(RuntimeError):
            with FP.peak_min_snr_scope(7.0):
                raise RuntimeError("fit died")
        self.assertEqual(FP.peak_min_F_override(), before)

    def _floor_seen_by_fit(self, m):
        seen = []

        def _fit(model, k, branches=None):
            seen.append(FP.fstat_peak_min_F())
            return None, 0
        m._run_fstat_fit = _fit
        m._install = lambda k, stacked=None, n_peaks=None, **kw: None
        m.setup(None, None)
        return seen

    def test_the_galaxy_fits_at_its_own_floor_and_the_residual_at_the_stage_s(self):
        FP.set_peak_min_F_override(6.25)
        self.assertEqual(self._floor_seen_by_fit(_galaxy(self.root)), [0.5 * 6.5 ** 2])
        self.assertAlmostEqual(FP.fstat_peak_min_F(), 0.5 * 6.25 ** 2)
        self.assertEqual(self._floor_seen_by_fit(_residual(self.root)),
                         [0.5 * 6.25 ** 2])


# ======================================================================
# 6. the cut IS the re-selection at the higher floor
# ======================================================================
class CutEqualsReselectionTest(_Isolated):

    def setUp(self):
        super().setUp()
        os.environ.update(_FIT_ENV)
        os.environ["FSTAT_PEAK_MIN_F"] = "10"
        self.d = os.path.join(self.root, "epoch")
        G.run_fstat_grid_fit(_bump_call_fstat(), xp=np, Tobs=7.776e6,
                             band_edges_hz=BAND_EDGES, f0_lims_hz=F0_LIMS,
                             mc_lims=[0.01, 1.0], cache_dir=self.d)

    def test_cut_keeps_exactly_the_peaks_a_reselection_selects(self):
        comb = np.load(G.stacked_grid_path(self.d).replace(
            "_peaks_stacked.npz", "_comb.npz"))
        nodes = np.asarray(comb["f0_nodes_mHz"], dtype=float)
        stacked = np.load(G.stacked_grid_path(self.d))
        for F_hi in (11.0, 15.0, 20.0, 26.0, 40.0):
            resel = G.select_comb_peaks(nodes, np.asarray(comb["F_max"]), BAND_EDGES,
                                        float(nodes[1] - nodes[0]), np, min_F=F_hi)
            cut, n_kept, n_all = G.stacked_cache_at_floor(stacked, F_hi)
            want = sorted(zip(np.round(resel[:, 0], 9), np.round(resel[:, 1], 9)))
            if n_kept == 0:
                self.assertIsNone(cut)
                self.assertEqual(want, [])
                continue
            got = sorted(zip(np.round(np.asarray(cut["peak_f0_mHz"]), 9),
                             np.round(np.asarray(cut["peak_F"]), 9)))
            self.assertEqual(got, want, F_hi)
            if n_kept == n_all:
                self.assertIs(cut, stacked)          # nothing dropped, nothing copied
                continue
            # every per-box array cut consistently, the stamp raised
            K = len(got)
            for key in G._STACKED_PER_BOX_KEYS:
                if key in cut:
                    self.assertEqual(len(cut[key]), K, key)
            grids = (cut["logp_grids"] if "logp_grids" in cut else np.concatenate(
                [cut[f"logp_grids_g{i}"] for i in range(len(cut["group_sizes"]))]))
            self.assertEqual(grids.shape[0], K)
            self.assertAlmostEqual(float(cut["peak_min_F"]), F_hi)
            # and a loader builds from it
            self.assertIsNotNone(FP.stacked_from_cache(cut))

    def test_kept_box_grids_are_the_stored_ones(self):
        stacked = np.load(G.stacked_grid_path(self.d))
        F = np.asarray(stacked["peak_F"], dtype=float)
        F_hi = float(np.median(F))
        cut, n_kept, n_all = G.stacked_cache_at_floor(stacked, F_hi)
        self.assertLess(n_kept, n_all)
        full = (stacked["logp_grids"] if "logp_grids" in stacked.files
                else np.concatenate([stacked[f"logp_grids_g{i}"]
                                     for i in range(len(stacked["group_sizes"]))]))
        sub = (cut["logp_grids"] if "logp_grids" in cut else np.concatenate(
            [cut[f"logp_grids_g{i}"] for i in range(len(cut["group_sizes"]))]))
        np.testing.assert_array_equal(sub, full[F >= F_hi])

    def test_no_box_below_returns_the_cache_itself(self):
        stacked = np.load(G.stacked_grid_path(self.d))
        cut, n_kept, n_all = G.stacked_cache_at_floor(stacked, 1.0)
        self.assertIs(cut, stacked)
        self.assertEqual(n_kept, n_all)

    @staticmethod
    def _grouped(cls=dict):
        """The grouped key layout (three Mc groups), built by hand; group 0's
        boxes are all below F = 10."""
        sizes = np.array([2, 3, 1])
        K = int(sizes.sum())
        F = np.array([5.0, 6.0, 30.0, 4.0, 40.0, 50.0])
        d = cls(group_sizes=sizes, peak_F=F, peak_f0_mHz=np.arange(K) * 0.1 + 6.4,
                f0_los=np.arange(K, dtype=float), f0_dxs=np.ones(K),
                band_idx=np.ones(K, dtype=int), alpha_ax=np.zeros(2),
                sin_delta_ax=np.zeros(2), peak_min_F=np.float64(4.0))
        for gi in range(3):
            d[f"logp_grids_g{gi}"] = np.full((int(sizes[gi]), 2, 2, 2, 2), float(gi))
            d[f"mc_ax_g{gi}"] = np.array([gi, gi + 1.0])
            d[f"alpha_ax_g{gi}"] = np.array([gi, gi + 0.5])
            d[f"sin_delta_ax_g{gi}"] = np.array([-gi, gi])
        return d

    def test_the_cut_reads_no_grid_until_asked_and_one_group_per_read(self):
        """Lazy: on the 9mo every rank cuts a ~32k-box galaxy grid, and holding
        every cut group in host memory at once is what the unmodified
        one-group-at-a-time loader never did."""
        reads = []

        class _Counting(dict):
            def __getitem__(self, k):
                reads.append(k)
                return dict.__getitem__(self, k)

        cut, _, _ = G.stacked_cache_at_floor(self._grouped(_Counting), 10.0)
        self.assertEqual([k for k in reads if k.startswith("logp_grids")], [])
        g0 = cut["logp_grids_g0"]                     # the old group 1
        self.assertEqual([k for k in reads if k.startswith("logp_grids")],
                         ["logp_grids_g1"])
        self.assertEqual(g0.shape[0], 2)
        self.assertTrue(np.all(g0 == 1.0))

    def test_grouped_layout_drops_an_emptied_group_and_renumbers(self):
        """A group whose every box falls below the floor goes, the rest
        renumber from 0."""
        d = self._grouped()
        cut, n_kept, n_all = G.stacked_cache_at_floor(d, 10.0)
        self.assertEqual((n_kept, n_all), (3, 6))
        np.testing.assert_array_equal(cut["group_sizes"], [2, 1])   # group 0 gone
        np.testing.assert_array_equal(cut["mc_ax_g0"], [1.0, 2.0])  # was group 1
        np.testing.assert_array_equal(cut["mc_ax_g1"], [2.0, 3.0])  # was group 2
        self.assertNotIn("logp_grids_g2", cut)
        self.assertEqual(cut["logp_grids_g0"].shape[0], 2)
        np.testing.assert_array_equal(cut["peak_F"], [30.0, 40.0, 50.0])
        np.testing.assert_array_equal(cut["f0_los"], [2.0, 4.0, 5.0])
        np.testing.assert_array_equal(cut["alpha_ax"], cut["alpha_ax_g0"])
        self.assertEqual(float(cut["peak_min_F"]), 10.0)


if __name__ == "__main__":
    unittest.main()
