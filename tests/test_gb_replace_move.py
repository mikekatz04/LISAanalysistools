"""GB F-stat REPLACEMENT move (2026-08-24 exact-MH reinstatement).

Covers the testable-without-a-sampler pieces of
``GBSpecialBase._run_replace_step``:

* the accept-time covering-set occupancy transition
  (``_cap_covering_transition_scatter`` -- the factored 04c78c56
  accounting), against a brute-force covering-set diff at overlap 0 and
  0.25;
* the destination-headroom veto (``_cap_new_entry_veto``): newly-entered
  at-cap cells veto, the mover's own (about to be vacated) cells never do;
* the forward/reverse slot-0 proposal-density symmetry off a toy F-stat
  center table (``_fstat_ctr_table_lookup`` + ``_slot0_log_proposal``):
  the factor correction is exactly antisymmetric under old <-> new, and
  the (truncated) density integrates to 1 -- i.e. the density matches the
  draw.

2026-08-24 candidate-quality fixes (root causes a/b/c of the 0/~500
acceptance forensics):

* knob defaults: replace ships with per-row F-stat extrinsics
  (``GB_REPLACE_CTR_MODE=perrow``), trilinear in-cell intrinsic draws
  (``GB_REPLACE_INCELL=trilinear``) and the slot-0 uniform floor
  (``GB_REPLACE_SLOT0_FLOOR_EPS=0.05``);
* the FLOOR-MIXED slot-0 density (``_slot0_log_proposal_floored``):
  normalization, forward/reverse antisymmetry, the bounded (~log eps)
  reverse bill for a far-off-center incumbent, and eps=0 equivalence
  with the unfloored density;
* the TRILINEAR in-cell mode of ``StackedFStatProposal4D``:
  normalization of ``logpdf`` in both modes, ``rvs``/``logpdf``
  consistency on a single-cell toy grid (per-axis marginal CDF/mean of
  the multilinear law), pointwise ``logpdf`` against a brute-force
  interpolant, and the ``stacked_in_cell_mode`` context (finds the
  stacked component through the production wrapper chain, restores the
  mode on exit);
* the ``fstat_maximized_extrinsics`` psi/phi0 CONVENTION fix: an exact
  round trip through the calibrated GBGPU forward amplitude map (build
  ``a`` from known (A, iota, psi, phi0), invert, recover the angles up
  to the physical (phi0 + pi, psi + pi/2) identity, with A/iota/F
  exact).

Light fakes in the ``test_gb_cap_cell_grid.py`` style (imported from
there): everything under test is pure array arithmetic, no built move,
ACA, or backend needed.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from tests.test_gb_cap_cell_grid import BAND_EDGES, _move, _move_overlap


def _members_sets(m, band, f):
    """Python-set covering-cell membership per row (brute-force oracle)."""
    band = np.asarray(band)
    f = np.asarray(f, dtype=float)
    p, nb, hn = m._cap_cell_members(band, f)
    out = []
    for i in range(len(p)):
        s = {int(p[i])}
        if nb is not None and bool(hn[i]):
            s.add(int(nb[i]))
        out.append(s)
    return out


class CoveringTransitionScatterTest(unittest.TestCase):
    """+1 newly covered / -1 no longer covered, weight 0 elsewhere."""

    def _check_random(self, m, seed, n=96):
        rng = np.random.default_rng(seed)
        be = np.asarray(m.band_edges)
        band = rng.integers(0, m.num_bands, n)
        lo, hi = be[band], be[band + 1]
        f_cur = lo + rng.random(n) * (hi - lo)
        f_new = lo + rng.random(n) * (hi - lo)
        t = np.zeros(n, dtype=np.int64)
        w = np.zeros(n, dtype=np.int64)
        accept = rng.random(n) < 0.5
        counts0 = rng.integers(0, 3, m.num_cap_cells).astype(np.int64)

        expected = counts0.copy()
        cur_sets = _members_sets(m, band, f_cur)
        new_sets = _members_sets(m, band, f_new)
        for i in range(n):
            if not accept[i]:
                continue
            # ntemps = nwalkers = 1 -> the flat index IS the cell index
            for c in new_sets[i] - cur_sets[i]:
                expected[c] += 1
            for c in cur_sets[i] - new_sets[i]:
                expected[c] -= 1

        got = counts0.copy()
        m._cap_covering_transition_scatter(
            got, t, w,
            m._cap_cell_members(band, f_cur),
            m._cap_cell_members(band, f_new),
            accept,
        )
        np.testing.assert_array_equal(got, expected)

    def test_overlap_zero_matches_brute_force(self):
        for seed in (0, 1, 2):
            self._check_random(_move(4), seed)

    def test_overlap_quarter_stagger_matches_brute_force(self):
        # the production v6 layout: K = 4, p = 0.25, staggered seam cells
        for seed in (3, 4, 5):
            self._check_random(_move_overlap(4, 0.25, stagger=True), seed)

    def test_rejected_and_noncrossing_rows_are_noops(self):
        for m in (_move(4), _move_overlap(4, 0.25, stagger=True)):
            n = 8
            band = np.zeros(n, dtype=np.int64)
            f = np.full(n, 5.4e-3)
            t = np.zeros(n, dtype=np.int64)
            w = np.zeros(n, dtype=np.int64)
            counts0 = np.arange(m.num_cap_cells, dtype=np.int64)
            # (a) same position (non-crossing), all accepted -> no change
            got = counts0.copy()
            m._cap_covering_transition_scatter(
                got, t, w,
                m._cap_cell_members(band, f), m._cap_cell_members(band, f),
                np.ones(n, dtype=bool),
            )
            np.testing.assert_array_equal(got, counts0)
            # (b) real crossings, all rejected -> no change
            f_new = np.full(n, 5.9e-3)
            got = counts0.copy()
            m._cap_covering_transition_scatter(
                got, t, w,
                m._cap_cell_members(band, f), m._cap_cell_members(band, f_new),
                np.zeros(n, dtype=bool),
            )
            np.testing.assert_array_equal(got, counts0)


class NewEntryVetoTest(unittest.TestCase):
    """Destination headroom: foreign at-cap cells veto; own cells never."""

    def _veto(self, m, counts, cap, band, f_cur, f_new):
        band = np.asarray(band, dtype=np.int64)
        n = band.size
        t = np.zeros(n, dtype=np.int64)
        w = np.zeros(n, dtype=np.int64)
        return m._cap_new_entry_veto(
            np.asarray(counts, dtype=np.int64), np.asarray(cap, dtype=np.int64),
            t, w,
            m._cap_cell_members(band, np.asarray(f_cur, dtype=float)),
            m._cap_cell_members(band, np.asarray(f_new, dtype=float)),
        )

    def test_partition_semantics_overlap_zero(self):
        m = _move(4)  # band 0 = 5-6 mHz -> cells 0..3, width 0.25 mHz
        counts = np.zeros(m.num_cap_cells, dtype=np.int64)
        cap = np.ones(m.num_cap_cells, dtype=np.int64)
        counts[0] = 1  # the mover itself
        counts[1] = 1  # a neighbour at cap
        cap[2] = -1    # disarmed
        counts[2] = 5
        band = [0, 0, 0, 0]
        f_cur = [5.1e-3] * 4  # cell 0 in every case
        f_new = [
            5.3e-3,   # -> cell 1 (foreign, at cap)      : VETO
            5.15e-3,  # -> cell 0 (own cell, even at cap): pass
            5.6e-3,   # -> cell 2 (disarmed)             : pass
            5.85e-3,  # -> cell 3 (headroom)             : pass
        ]
        got = self._veto(m, counts, cap, band, f_cur, f_new)
        np.testing.assert_array_equal(got, [True, False, False, False])

    def test_own_vacated_cells_are_legal_in_overlap_mode(self):
        # A mover sitting in an overlap zone covers TWO cells; a
        # replacement landing anywhere within its own covering set must be
        # legal even when those cells are at cap ONLY because of the mover.
        m = _move_overlap(4, 0.25, stagger=True)
        band = np.array([0], dtype=np.int64)
        # scan for an f whose covering set has two cells (an overlap zone)
        f_scan = np.linspace(5.01e-3, 5.99e-3, 397)
        sets = _members_sets(m, np.zeros(f_scan.size, dtype=np.int64), f_scan)
        i2 = next(i for i, s in enumerate(sets) if len(s) == 2)
        f_cur = f_scan[i2]
        own = sets[i2]
        counts = np.zeros(m.num_cap_cells, dtype=np.int64)
        cap = np.ones(m.num_cap_cells, dtype=np.int64)
        for c in own:
            counts[c] = 1  # occupied ONLY by the mover; at cap 1
        # (a) a move whose new covering set stays inside `own` passes
        f_inside = [
            f for f, s in zip(f_scan, sets) if s and s.issubset(own)
        ]
        self.assertGreater(len(f_inside), 0)
        got = self._veto(m, counts, cap, band, [f_cur], [f_inside[-1]])
        np.testing.assert_array_equal(got, [False])
        # (b) a move that newly enters a foreign at-cap cell is vetoed
        j = next(
            i for i, s in enumerate(sets) if s - own
        )
        foreign = sorted(sets[j] - own)[0]
        counts[foreign] = 1  # someone else fills it
        got = self._veto(m, counts, cap, band, [f_cur], [f_scan[j]])
        np.testing.assert_array_equal(got, [True])
        # (c) same destination with headroom in the foreign cell passes
        counts[foreign] = 0
        got = self._veto(m, counts, cap, band, [f_cur], [f_scan[j]])
        np.testing.assert_array_equal(got, [False])


class SlotZeroDensityTest(unittest.TestCase):
    """Forward/reverse density symmetry + normalization off a toy table.

    Amplitude basis (the shim has no transform container, so
    ``_gb_use_distance`` is False and slot 0 is lnA): ``ln_center`` is the
    table's ``ln_A_max`` directly and no stock-transform import is
    needed.
    """

    def _shim(self):
        m = _move(4)
        nodes = np.linspace(5.0, 9.0, 41)
        rng = np.random.default_rng(7)
        m._fstat_ctr_table = dict(
            f0_mHz=nodes,
            phi0=rng.uniform(0.0, 2.0 * np.pi, nodes.size),
            iota=rng.uniform(0.0, np.pi, nodes.size),
            psi=rng.uniform(0.0, np.pi, nodes.size),
            ln_A_max=rng.uniform(-51.0, -47.0, nodes.size),
            sigma_base=np.full(nodes.size, 0.05),
            ln_snr=rng.uniform(np.log(50.0), np.log(500.0), nodes.size),
        )
        return m

    @staticmethod
    def _rows(m, rng, n):
        """Rows with slot 0 IN the truncated support (near its own center)."""
        p = np.zeros((n, 8))
        p[:, 1] = rng.uniform(5.0, 9.0, n)     # f0 [mHz]
        p[:, 2] = rng.uniform(0.1, 0.9, n)     # Mc (unused, amp basis)
        (_, _, _, lc, sg, _) = m._fstat_ctr_table_lookup(p)
        p[:, 0] = lc + sg * rng.standard_normal(n)
        return p

    def _corr(self, m, pa, pb, lim):
        """The factor correction _run_replace_step adds for the swap a -> b.

        ``(+log g(a) + log_range) + (-log g(b) - log_range)`` with ``g``
        each side's own (truncated) table lognormal -- the +/- log_range
        pair cancels, so it is omitted here exactly as it cancels there.
        """
        (_, _, _, lc_b, sg_b, ls_b) = m._fstat_ctr_table_lookup(pb)
        (_, _, _, lc_a, sg_a, ls_a) = m._fstat_ctr_table_lookup(pa)
        a_b = m._snr_trunc_alpha(ls_b, sg_b, lim)
        a_a = m._snr_trunc_alpha(ls_a, sg_a, lim)
        bl = m._slot0_log_proposal(pb[:, 0], lc_b, sg_b, alpha=a_b)
        dl = m._slot0_log_proposal(pa[:, 0], lc_a, sg_a, alpha=a_a)
        return dl - bl

    def test_forward_reverse_antisymmetry(self):
        m = self._shim()
        rng = np.random.default_rng(11)
        pa = self._rows(m, rng, 64)
        pb = self._rows(m, rng, 64)
        fwd = self._corr(m, pa, pb, lim=8.0)
        rev = self._corr(m, pb, pa, lim=8.0)
        self.assertTrue(np.all(np.isfinite(fwd)))
        np.testing.assert_allclose(fwd + rev, 0.0, atol=1e-10)

    def test_lookup_is_deterministic_both_directions(self):
        # The reverse density at the CURRENT leaf's parameters uses the
        # same table lookup as the forward draw would at those parameters
        # -- one code path, so the pair is symmetric by construction; this
        # pins that construction.
        m = self._shim()
        rng = np.random.default_rng(13)
        p = self._rows(m, rng, 32)
        first = m._fstat_ctr_table_lookup(p)
        second = m._fstat_ctr_table_lookup(p)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_truncated_density_normalizes(self):
        # The density must match the draw: integrate exp(log g) over lnA
        # for a per-row truncation boundary and recover 1 (amp basis:
        # lower truncation at ln_center - sigma * alpha).
        m = self._shim()
        lc, sg = -49.0, 0.5
        for alpha in (0.7, 3.0, 40.0):
            grid = np.linspace(lc - sg * min(alpha, 9.0), lc + 9.0 * sg,
                               200001)
            logg = m._slot0_log_proposal(
                grid, np.full(grid.size, lc), np.full(grid.size, sg),
                alpha=np.full(grid.size, alpha),
            )
            dens = np.exp(np.clip(logg, -1e290, None))
            integral = np.trapezoid(dens, grid)
            self.assertAlmostEqual(float(integral), 1.0, places=3)

    def test_out_of_support_reverse_density_force_rejects(self):
        # An old leaf whose lnA lies below the truncated support gets
        # reverse density -1e300 -> the assembled factors force-reject the
        # swap (the truncated proposal could never have produced it).
        m = self._shim()
        rng = np.random.default_rng(17)
        pa = self._rows(m, rng, 4)
        pb = self._rows(m, rng, 4)
        pa[:, 0] = -80.0  # far below any center/support
        fwd = self._corr(m, pa, pb, lim=8.0)
        self.assertTrue(np.all(fwd <= -1e299))


class _EnvPatch:
    """Set/unset env vars for one test, restoring on exit."""

    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ReplaceKnobDefaultsTest(unittest.TestCase):
    """The 2026-08-24 candidate-quality fixes ship ON by default."""

    def test_ctr_mode_default_is_perrow(self):
        m = _move(4)
        with _EnvPatch(GB_REPLACE_CTR_MODE=None):
            self.assertEqual(m._replace_ctr_mode(), "perrow")
        with _EnvPatch(GB_REPLACE_CTR_MODE="table"):
            self.assertEqual(m._replace_ctr_mode(), "table")
        with _EnvPatch(GB_REPLACE_CTR_MODE="bogus"):
            with self.assertRaises(ValueError):
                m._replace_ctr_mode()

    def test_perrow_default_never_consults_the_table(self):
        # the replace step's table handle is None unless the knob says
        # "table" -- mirror the exact gating expression it uses.
        m = _move(4)
        m._fstat_ctr_table = {"f0_mHz": np.array([5.0])}  # a table EXISTS
        with _EnvPatch(GB_REPLACE_CTR_MODE=None, GB_FSTAT_CTR_MODE=None):
            tbl = (m._fstat_ctr_table_active()
                   if m._replace_ctr_mode() == "table" else None)
            self.assertIsNone(tbl)
        with _EnvPatch(GB_REPLACE_CTR_MODE="table", GB_FSTAT_CTR_MODE=None):
            tbl = (m._fstat_ctr_table_active()
                   if m._replace_ctr_mode() == "table" else None)
            self.assertIsNotNone(tbl)

    def test_incell_mode_default_is_trilinear(self):
        m = _move(4)
        with _EnvPatch(GB_REPLACE_INCELL=None):
            self.assertEqual(m._replace_incell_mode(), "trilinear")
        with _EnvPatch(GB_REPLACE_INCELL="uniform"):
            self.assertEqual(m._replace_incell_mode(), "uniform")
        with _EnvPatch(GB_REPLACE_INCELL="bogus"):
            with self.assertRaises(ValueError):
                m._replace_incell_mode()

    def test_slot0_floor_eps_default(self):
        m = _move(4)
        with _EnvPatch(GB_REPLACE_SLOT0_FLOOR_EPS=None):
            self.assertAlmostEqual(m._replace_slot0_floor_eps(), 0.05)
        with _EnvPatch(GB_REPLACE_SLOT0_FLOOR_EPS="0"):
            self.assertEqual(m._replace_slot0_floor_eps(), 0.0)


class SlotZeroFloorMixtureTest(unittest.TestCase):
    """Fix 3: the floor-mixed slot-0 density (root cause (c)).

    Amp basis (shim has no transform container): slot 0 is lnA, the
    container's slot-0 prior is U[lo, hi] in lnA. The fake sorter carries
    that prior so ``_slot0_range`` and ``_log_dist_range`` extract
    CONSISTENT bounds/width — as they do from the real birth container.
    """

    LO, HI = -52.0, -40.0

    def _shim(self):
        m = _move(4)
        m._log_dist_range_cache = None
        prior = SimpleNamespace(minimum=self.LO, maximum=self.HI,
                                width=self.HI - self.LO)
        sorter = SimpleNamespace(rj_prop=SimpleNamespace(
            priors=[((0,), prior)]))
        return m, sorter

    def test_floored_density_normalizes(self):
        m, sorter = self._shim()
        lc, sg, eps = -46.0, 0.5, 0.05
        grid = np.linspace(self.LO - 1.0, self.HI + 1.0, 400001)
        ones = np.ones_like(grid)
        for alpha in (0.7, 3.0, 40.0):
            logg = m._slot0_log_proposal_floored(
                grid, lc * ones, sg * ones, alpha * ones, sorter, eps)
            integral = np.trapezoid(np.exp(np.clip(logg, -1e290, None)), grid)
            self.assertAlmostEqual(float(integral), 1.0, places=3)

    def test_eps_zero_is_the_unfloored_density(self):
        m, sorter = self._shim()
        rng = np.random.default_rng(5)
        v = rng.uniform(self.LO, self.HI, 64)
        lc = np.full(64, -46.0)
        sg = np.full(64, 0.5)
        al = np.full(64, 3.0)
        a = m._slot0_log_proposal_floored(v, lc, sg, al, sorter, 0.0)
        b = m._slot0_log_proposal(v, lc, sg, alpha=al)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_bounded_reverse_bill(self):
        # a polished incumbent 8 sigma BELOW its center (outside the
        # truncated support in the amp basis) pays ~log(eps/width), not
        # -1e300: the p10 ~ -125 / truncation force-reject tax is gone.
        m, sorter = self._shim()
        lc, sg, al, eps = np.array([-46.0]), np.array([0.5]), np.array([3.0]), 0.05
        v = lc - 8.0 * sg  # -50, inside [LO, HI]
        bare = m._slot0_log_proposal(v, lc, sg, alpha=al)
        floored = m._slot0_log_proposal_floored(v, lc, sg, al, sorter, eps)
        self.assertLessEqual(float(bare[0]), -1e299)  # old: force-reject
        expected_floor = np.log(eps) - np.log(self.HI - self.LO)
        self.assertAlmostEqual(float(floored[0]), float(expected_floor),
                               places=6)
        self.assertGreater(float(floored[0]), -6.0)  # bounded (~-5.5)

    def test_forward_reverse_antisymmetry(self):
        m, sorter = self._shim()
        rng = np.random.default_rng(23)
        n = 64
        lc_a = rng.uniform(-49.0, -43.0, n)
        lc_b = rng.uniform(-49.0, -43.0, n)
        sg = np.full(n, 0.4)
        al = np.full(n, 5.0)
        va = lc_a + 0.5 * rng.standard_normal(n)
        vb = lc_b + 0.5 * rng.standard_normal(n)
        # swap correction a -> b: (+g(a)) + (-g(b)); reverse negates it
        ga = m._slot0_log_proposal_floored(va, lc_a, sg, al, sorter, 0.05)
        gb = m._slot0_log_proposal_floored(vb, lc_b, sg, al, sorter, 0.05)
        fwd = ga - gb
        rev = gb - ga
        self.assertTrue(np.all(np.isfinite(fwd)))
        np.testing.assert_allclose(fwd + rev, 0.0, atol=1e-12)


class TrilinearStackedProposalTest(unittest.TestCase):
    """Fix 2 (option ii): the trilinear in-cell mode of the stacked grid."""

    @staticmethod
    def _toy(K=2, seed=0, node_shape=(4, 3, 3, 3)):
        from lisatools.sampling.fstat_proposal import StackedFStatProposal4D

        rng = np.random.default_rng(seed)
        grids = rng.normal(0.0, 2.0, (K,) + node_shape)
        f0_los = 5.0 + np.arange(K) * 1.0          # disjoint f0 boxes
        f0_dxs = np.full(K, 0.5 / (node_shape[0] - 1))
        mc_ax = np.linspace(0.1, 1.0, node_shape[1])
        alpha_ax = np.linspace(0.0, 2.0, node_shape[2])
        sd_ax = np.linspace(-1.0, 1.0, node_shape[3])
        return StackedFStatProposal4D(
            grids, f0_los, f0_dxs, mc_ax, alpha_ax, sd_ax, seed=seed + 1)

    def _mesh_mass(self, dist, n0=30, n3=24):
        """Midpoint-rule integral of exp(logpdf) over every box."""
        total = 0.0
        ax3 = [np.linspace(lo, hi, n3 + 1) for lo, hi in
               zip(dist._lo3, dist._hi3)]
        mid3 = [0.5 * (a[:-1] + a[1:]) for a in ax3]
        d3 = [a[1] - a[0] for a in ax3]
        for k in range(dist.K):
            f0e = np.linspace(dist._f0_lo[k], dist._f0_hi[k], n0 + 1)
            f0m = 0.5 * (f0e[:-1] + f0e[1:])
            g = np.meshgrid(f0m, *mid3, indexing="ij")
            pts = np.stack([x.ravel() for x in g], axis=1)
            lp = np.asarray(dist.logpdf(pts))
            vol = (f0e[1] - f0e[0]) * np.prod(d3)
            total += float(np.sum(np.exp(lp)) * vol)
        return total

    def test_logpdf_normalizes_in_both_modes(self):
        dist = self._toy()
        self.assertEqual(dist.in_cell, "uniform")
        m_uni = self._mesh_mass(dist)
        dist.in_cell = "trilinear"
        m_tri = self._mesh_mass(dist)
        dist.in_cell = "uniform"
        self.assertAlmostEqual(m_uni, 1.0, delta=0.02)
        self.assertAlmostEqual(m_tri, 1.0, delta=0.02)

    def test_pointwise_logpdf_matches_bruteforce_interpolant(self):
        dist = self._toy(K=1, seed=3)
        dist.in_cell = "trilinear"
        rng = np.random.default_rng(7)
        n = 256
        pts = np.stack([
            rng.uniform(dist._f0_lo[0], dist._f0_hi[0], n),
            rng.uniform(dist._lo3[0], dist._hi3[0], n),
            rng.uniform(dist._lo3[1], dist._hi3[1], n),
            rng.uniform(dist._lo3[2], dist._hi3[2], n),
        ], axis=1)
        lp = np.asarray(dist.logpdf(pts))
        # brute force: multilinear interp of exp(node grid) / Z
        g = np.exp(dist._chunks[0]["log_node"][0])
        lo = np.array([dist._f0_lo[0], *dist._lo3])
        dx = np.array([dist._f0_dx[0], *dist._dx3])
        expect = np.empty(n)
        for r in range(n):
            u = (pts[r] - lo) / dx
            i = np.minimum(u.astype(int), np.array(g.shape) - 2)
            t = u - i
            val = 0.0
            for c in range(16):
                bits = [(c >> b) & 1 for b in range(4)]
                w = np.prod([t[b] if bits[b] else 1 - t[b] for b in range(4)])
                val += w * g[i[0] + bits[0], i[1] + bits[1],
                             i[2] + bits[2], i[3] + bits[3]]
            expect[r] = np.log(val) - dist._log_norm[0]
        np.testing.assert_allclose(lp, expect, atol=1e-10)
        dist.in_cell = "uniform"

    def test_rvs_matches_the_multilinear_law_single_cell(self):
        # one box, one cell: the in-cell draw IS the multilinear density;
        # check the per-axis marginal mean and CDF(1/2) against the
        # analytic linear-marginal values (endpoints = face means).
        from lisatools.sampling.fstat_proposal import StackedFStatProposal4D

        rng = np.random.default_rng(11)
        grids = rng.uniform(-1.0, 2.5, (1, 2, 2, 2, 2))
        dist = StackedFStatProposal4D(
            grids, [5.0], [1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], seed=13)
        dist.in_cell = "trilinear"
        draws = np.asarray(dist.rvs(size=40000))
        t = draws.copy()
        t[:, 0] -= 5.0  # unit cell in every axis
        w = np.exp(grids[0])
        for ax in range(4):
            a = w.mean(axis=tuple(i for i in range(4) if i != ax))
            lo_face, hi_face = a[0], a[1]
            mean_th = (lo_face + 2 * hi_face) / (3.0 * (lo_face + hi_face))
            cdf_half_th = (0.75 * lo_face + 0.25 * hi_face) / (lo_face + hi_face)
            self.assertAlmostEqual(float(t[:, ax].mean()), mean_th, delta=0.01)
            self.assertAlmostEqual(float((t[:, ax] < 0.5).mean()),
                                   cdf_half_th, delta=0.01)
        dist.in_cell = "uniform"

    def test_uniform_mode_untouched_and_context_restores(self):
        from lisatools.sampling.fstat_proposal import (
            iter_stacked_components,
            stacked_in_cell_mode,
        )

        dist = self._toy(seed=17)
        rng = np.random.default_rng(19)
        pts = np.stack([
            rng.uniform(dist._f0_lo[0], dist._f0_hi[0], 32),
            rng.uniform(dist._lo3[0], dist._hi3[0], 32),
            rng.uniform(dist._lo3[1], dist._hi3[1], 32),
            rng.uniform(dist._lo3[2], dist._hi3[2], 32),
        ], axis=1)
        before = np.asarray(dist.logpdf(pts))
        with stacked_in_cell_mode(dist, "trilinear") as active:
            self.assertTrue(active)
            self.assertEqual(dist.in_cell, "trilinear")
            inside = np.asarray(dist.logpdf(pts))
        self.assertEqual(dist.in_cell, "uniform")
        after = np.asarray(dist.logpdf(pts))
        np.testing.assert_array_equal(before, after)
        self.assertTrue(np.any(np.abs(inside - before) > 1e-6))
        # uniform mode is a no-op context
        with stacked_in_cell_mode(dist, "uniform") as active:
            self.assertFalse(active)
        self.assertEqual(list(iter_stacked_components(dist)), [dist])

    def test_walker_finds_stacked_through_production_chain(self):
        from lisatools.sampling.fstat_proposal import (
            UniformFloorMixture,
            iter_stacked_components,
            make_gb_rj_birth_container,
            stacked_in_cell_mode,
        )

        stacked = self._toy(seed=23)
        mix = UniformFloorMixture(
            stacked, [5.0, 0.1, 0.0, -1.0], [6.5, 1.0, 2.0, 1.0], eps=0.05)
        cont = make_gb_rj_birth_container(
            mix, A_lims=[7e-26, 1e-19], use_cupy=False,
            fdot_astro_ratio_max=5.0, dist_lims=[0.001, 40.0],
            ratio_tight=dict(tobs=7776000.0),
        )
        found = list(iter_stacked_components(cont))
        self.assertEqual(found, [stacked])
        with stacked_in_cell_mode(cont, "trilinear") as active:
            self.assertTrue(active)
            self.assertEqual(stacked.in_cell, "trilinear")
        self.assertEqual(stacked.in_cell, "uniform")


class FstatExtrinsicsConventionTest(unittest.TestCase):
    """The psi/phi0 convention fix in ``fstat_maximized_extrinsics``.

    Forward map (calibrated 2026-08-24 against GBGPU CPU waveforms on a
    noise-free truth residual — the harness recovered the FULL truth
    delta, match 1.0000, only with this map): with ``phi = -phi0`` and
    ``A+ = A (1 + cos^2 iota)/2``, ``Ax = A cos iota``,

        t1 =  A+ c2psi cphi - Ax s2psi sphi
        t2 =  A+ s2psi cphi + Ax c2psi sphi
        t3 = -A+ c2psi sphi - Ax s2psi cphi
        t4 = -A+ s2psi sphi + Ax c2psi cphi
        a  = (t1, -t2, t3, -t4)

    The inversion must recover (A, iota, psi, phi0) from ``a`` up to the
    physical (phi0 + pi, psi + pi/2) identity, with F = a.N/2 exact.
    """

    @staticmethod
    def _forward(A, iota, psi, phi0):
        c = np.cos(iota)
        Ap = A * (1.0 + c * c) / 2.0
        Ax = A * c
        phi = -phi0
        c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
        cf, sf = np.cos(phi), np.sin(phi)
        t1 = Ap * cf * c2 - Ax * sf * s2
        t2 = Ap * cf * s2 + Ax * sf * c2
        t3 = -Ap * sf * c2 - Ax * cf * s2
        t4 = -Ap * sf * s2 + Ax * cf * c2
        return np.stack([t1, -t2, t3, -t4], axis=-1)

    def test_roundtrip_recovers_the_angles(self):
        from lisatools.sampling.fstat_proposal import (
            _TRIU_COLS,
            _TRIU_ROWS,
            fstat_maximized_extrinsics,
        )

        rng = np.random.default_rng(29)
        n = 256
        A = rng.uniform(0.5, 2.0, n)
        iota = np.arccos(rng.uniform(-0.95, 0.95, n))
        psi = rng.uniform(0.0, np.pi, n)
        phi0 = rng.uniform(0.0, 2.0 * np.pi, n)
        a = self._forward(A, iota, psi, phi0)          # (n, 4)
        # well-conditioned SPD M per row
        Q = rng.normal(0.0, 0.3, (n, 4, 4))
        M4 = 3.0 * np.eye(4)[None] + Q @ np.swapaxes(Q, 1, 2)
        N_arr = np.einsum("nij,nj->ni", M4, a)
        M_up = np.stack([M4[:, i, j] for i, j in zip(_TRIU_ROWS, _TRIU_COLS)],
                        axis=1)
        A_r, phi0_r, iota_r, psi_r, F_r = fstat_maximized_extrinsics(
            N_arr, M_up)
        np.testing.assert_allclose(F_r, 0.5 * np.sum(a * N_arr, axis=1),
                                   rtol=1e-8)
        np.testing.assert_allclose(A_r, A, rtol=1e-6)
        np.testing.assert_allclose(np.cos(iota_r), np.cos(iota), atol=1e-6)
        # angles up to the physical (phi0 + pi, psi + pi/2) identity
        dpsi = (np.asarray(psi_r) - psi) % np.pi
        half_flip = np.abs(dpsi - np.pi / 2) < 1e-6
        same_psi = np.minimum(dpsi, np.pi - dpsi) < 1e-6
        self.assertTrue(np.all(half_flip | same_psi))
        dphi = (np.asarray(phi0_r) - phi0) % (2.0 * np.pi)
        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)   # distance to 0
        is_pi = np.abs(dphi - np.pi) < 1e-6
        is_same = dphi < 1e-6
        # psi half-flip must pair with the phi0 + pi shift; unflipped psi
        # must pair with unshifted phi0
        self.assertTrue(np.all(np.where(half_flip, is_pi, is_same)))

    def test_concrete_template_projection_is_lossless(self):
        # d_h of the reconstructed concrete parameters at the OPTIMAL
        # amplitude equals 2F exactly: b(A_r, iota_r, psi_r, phi0_r) == a,
        # so the pinned extrinsics reproduce the ML template bit-for-bit
        # (the property the replace candidates rely on).
        rng = np.random.default_rng(31)
        n = 128
        A = rng.uniform(0.5, 2.0, n)
        iota = np.arccos(rng.uniform(-0.95, 0.95, n))
        psi = rng.uniform(0.0, np.pi, n)
        phi0 = rng.uniform(0.0, 2.0 * np.pi, n)
        a = self._forward(A, iota, psi, phi0)
        from lisatools.sampling.fstat_proposal import (
            _TRIU_COLS,
            _TRIU_ROWS,
            fstat_maximized_extrinsics,
        )
        Q = rng.normal(0.0, 0.3, (n, 4, 4))
        M4 = 3.0 * np.eye(4)[None] + Q @ np.swapaxes(Q, 1, 2)
        N_arr = np.einsum("nij,nj->ni", M4, a)
        M_up = np.stack([M4[:, i, j] for i, j in zip(_TRIU_ROWS, _TRIU_COLS)],
                        axis=1)
        A_r, phi0_r, iota_r, psi_r, _ = fstat_maximized_extrinsics(N_arr, M_up)
        b = self._forward(np.asarray(A_r), np.asarray(iota_r),
                          np.asarray(psi_r), np.asarray(phi0_r))
        np.testing.assert_allclose(b, a, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
