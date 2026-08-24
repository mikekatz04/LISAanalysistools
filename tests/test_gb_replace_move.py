"""GB F-stat REPLACEMENT move (2026-08-24 exact-MH reinstatement).

Covers the three testable-without-a-sampler pieces of
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

Light fakes in the ``test_gb_cap_cell_grid.py`` style (imported from
there): everything under test is pure array arithmetic, no built move,
ACA, or backend needed.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
