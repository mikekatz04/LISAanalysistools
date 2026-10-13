"""Every sub-band owns its covering WDM layer(s) as the valve window.

USER RULING 2026-10-13 ("Give every sub-band its covering layer as the valve
window"). The per-(walker, band) RJ valve judges the band GAIN
``1/2 <d|d> - 1/2 <r|r>`` over a window of whole WDM layers
(``GBSpecialBase._window_bins``). Until this change the window was the layers
whose CENTRE ``m * layer_df`` lies inside the band. The production sub-bands
are 1/8 of a layer wide (9mo: 1232 sub-bands, 17.36 uHz, on 138.9 uHz
layers), so 924 of them held no centre, had an EMPTY window and a gain of
exactly 0 on every walker and row; the judge saw a pair that never improved
and shut it after the patience count whatever its sources did (9mo store,
row 66, the first gb_search_2 row with a shut pair: 40.6 % of all blind
(walker, band) pairs shut -- about every occupied one -- against 7.7 % of the
window-owning pairs).

The WDM tiling (``lisatools.domains.WDMSettings``): layer ``m`` is centred on
``m * layer_df`` and tiles ``[(m - 1/2), (m + 1/2)) * layer_df``. The covering
rule gives a band every layer whose tile it overlaps: a sub-band inside one
tile owns that layer (the tile's 8 sub-bands share its gain), one straddling
a tile edge owns both, and every band that owned a layer before keeps it.

The geometry has three consumers that must cover the same pixels: the
residual term and the data term of the band gain (``_window_residual_lls``
on the band grid) and the in-model sites' per-slot window
(``_slot_band_windows`` -> ``_observed_band_gain``). The cap-cell grid keeps
the centres rule: its resolution test depends on sub-layer cells coming back
empty.

THE TRANSITION is persisted: ``band_info["band_window_rule"]`` (1 = centres,
2 = covering). A record learned under rule 1 is re-BASED once at the first
judge under rule 2: the running max / best / peak take that step's gain under
the new geometry, the streaks, reset counts and shut set stay exactly as
stored and that step is not judged (user ruling 2026-10-13: "a reset mid
stage does not adjust anything about convergence/turn over ... it should
stay in its current state through a reset"); the stamp is saved, and a
resume does not repeat it.
"""

import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

import lisatools.globalfit.moves.gbspecialstretch as gbs
from lisatools.domains import WDMSettings
from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend
from lisatools.globalfit.moves.gbbands import SubBandBuffer
from lisatools.globalfit.state import (
    VALVE_WINDOW_RULE,
    VALVE_WINDOW_RULE_CENTRES,
    VALVE_WINDOW_RULE_COVERING,
    VALVE_WINDOW_RULE_FIELD,
    GBState,
    ensure_search_shutoff_fields,
    ensure_valve_window_rule,
)

from tests.test_gb_search_stage_schedule import (_band_info, _counts,
                                                 _stage_move, _state)
from tests.test_gf_substate_roundtrip import (BAND_EDGES as RT_EDGES,
                                              BRANCH_SHAPES, NTEMPS, NWALKERS,
                                              NUM_BANDS as RT_NB,
                                              _PER_WALKER_FLAGS, make_gb_state)

# ---- the synthetic WDM geometry ------------------------------------------
LDF = 1.0                  # layer_df: frequencies below are in layer units
IND_MIN_F = 2              # first active layer
NFA = 6                    # active layers 2..7
NT = 4                     # time pixels per layer
NC = 2                     # diagonal channels
DIFF = 0.5                 # differential_component -> <a|b> = 2 sum(a b invC)
NORM = 4.0 * DIFF
DIV = 8                    # sub-bands per layer, as GB_SUBBAND_DIVISOR=8
#: 24 sub-bands of 1/8 layer over [3, 6] layer_df, the uniform-mode grid
#: (edges ``k * layer_df / 8``): band 0 starts on layer 3's CENTRE.
EDGES = np.arange(3 * DIV, 6 * DIV + 1) / DIV * LDF
NB = len(EDGES) - 1
#: the covering layer (absolute) of each sub-band: the tile [m-1/2, m+1/2)
COVER = np.array([3] * 4 + [4] * 8 + [5] * 8 + [6] * 4)
#: the sub-bands that owned a layer under the centres rule (a centre on an
#: edge: 3, 4, 4, 5, 5, 6) and that layer
OLD_OWNERS = {0: 3, 7: 4, 8: 4, 15: 5, 16: 5, 23: 6}

TOL = 4.0                  # D/2 with the fixture's leaf_cap_ndim = 8
CONV = 3


class _WDM(WDMSettings):
    """The attributes the window code reads, without building a transform."""

    layer_df = LDF
    ind_min_f = IND_MIN_F
    ind_max_f = IND_MIN_F + NFA - 1
    Nf_active = NFA
    Nt_active = NT
    differential_component = DIFF


def _wdm_basis():
    return object.__new__(_WDM)


def _layers(k0, k1):
    """Absolute layer lists per window, for readable assertions."""
    return [list(range(a + IND_MIN_F, b + IND_MIN_F)) for a, b in zip(k0, k1)]


class _WDMACA:
    """One CPU shard, diagonal noise, WDM-shaped ``(nw, nc, Nf, Nt)`` planes.

    ``input_data_residual_array`` mirrors the production chain
    ``DataResidualArray -> .data_res_arr (DomainBase) -> .arr``.
    """

    gpus = None
    nchannels = NC
    shape_sens = (NC,)

    def __init__(self, data, residual, inv_psd=1.0):
        self.residual = np.asarray(residual, dtype=float)
        nw = self.residual.shape[0]
        self.inv_psd = np.full((nw, NC, NFA, NT), 1.0) * np.asarray(inv_psd)
        self.acs_total_entries = nw
        self.gpu_splits = [np.arange(nw)]
        self.psd_version = 0
        self.input_data_residual_array = SimpleNamespace(
            data_res_arr=SimpleNamespace(arr=np.asarray(data, dtype=float)))

    @property
    def data_shaped(self):
        return [self.residual]

    @property
    def psd_shaped(self):
        return [self.inv_psd]


NOISE = 0.5
SRC = np.sqrt(62.5)        # 1/2 <s|s> over one layer = 1/2 * 2 * 8 * 62.5 = 500


def _source_plane(layers):
    """``(NC, NFA, NT)``: a source of amplitude SRC filling each given layer."""
    s = np.zeros((NC, NFA, NT))
    for m in layers:
        s[:, m - IND_MIN_F, :] = SRC
    return s


def _planes(fit, src_layers):
    """``(data, residual)`` for per-walker fit fractions ``fit[w][layer]``:
    ``d = n + sum s``, ``r_w = n + sum (1 - f) s`` (real WDM coefficients)."""
    nw = len(fit)
    n = np.full((NC, NFA, NT), NOISE)
    d = n + sum(_source_plane([m]) for m in src_layers)
    r = np.repeat(n[None], nw, axis=0)
    for w in range(nw):
        for m in src_layers:
            r[w] += (1.0 - fit[w].get(m, 0.0)) * _source_plane([m])
    return d, r


def _layer_gain(d, r_w, m, inv_psd=1.0):
    """Hand-computed gain of ONE layer: 1/2 <d|d> - 1/2 <r|r> on its pixels."""
    k = m - IND_MIN_F
    return 0.5 * NORM * inv_psd * float(np.sum(d[:, k] ** 2 - r_w[:, k] ** 2))


def _wdm_move(nwalkers, conv_iter=CONV, cap_edges=None):
    """The valve fixture of ``test_gb_search_stage_schedule`` on the WDM
    sub-band grid above."""
    m = _stage_move(shutoff=True, conv_iter=conv_iter, nwalkers=nwalkers)
    m._basis_settings = _wdm_basis()
    m.band_edges = np.asarray(EDGES, dtype=float)
    m.num_bands = NB
    m.cap_divisor = 1
    m.cap_stagger = False
    m.cap_edges = np.asarray(EDGES if cap_edges is None else cap_edges,
                             dtype=float)
    m.num_cap_cells = len(m.cap_edges) - 1
    bi = _band_info(nwalkers=nwalkers, num_bands=NB, shutoff=True)
    bi["band_edges"] = np.asarray(EDGES)
    m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
    m._bind_shutoff_window(bi, np.shape(bi["band_rj_shutoff_w"]))
    return m, bi


def _judge(m, bi, aca, occ):
    """One iteration's judge with NO stash: the valve forms its statistic
    from ``_cap_stats_local`` on the model's ACA (the single-process path)."""
    m.num_proposals += 1
    m._stage_band_gain = m._stage_band_lls = None
    m._update_search_band_shutoff(
        SimpleNamespace(analysis_container_arr=aca), _state(bi), _counts(occ))


def _no_tol_env():
    env = dict(os.environ)
    for k in ("GB_SEARCH_BAND_SHUTOFF_LL_TOL", "GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC",
              "GB_CAP_LL_SOURCE"):
        env.pop(k, None)
    return mock.patch.dict(os.environ, env, clear=True)


def _centres_only():
    """Force the pre-2026-10-13 band-grid geometry: the paired control."""
    orig = gbs.GBSpecialBase._window_bins

    def _old(self, acs, edges, cover=True):
        return orig(self, acs, edges, cover=False)

    return mock.patch.object(gbs.GBSpecialStretchMove, "_window_bins", _old)


# ==========================================================================
# (a) the geometry
# ==========================================================================
class CoveringGeometryTest(unittest.TestCase):
    """``wdm_window_layers``: who owns which layer, on synthetic grids."""

    def test_every_sub_band_owns_its_covering_layer(self):
        k0, k1 = gbs.wdm_window_layers(EDGES, LDF, IND_MIN_F, NFA, cover=True)
        self.assertTrue(np.all(k1 > k0), "a sub-band with no window")
        self.assertEqual(_layers(k0, k1), [[int(c)] for c in COVER])

    def test_the_centres_rule_left_three_in_four_blind(self):
        """The control: the old geometry, verbatim."""
        k0, k1 = gbs.wdm_window_layers(EDGES, LDF, IND_MIN_F, NFA, cover=False)
        owners = {b: lay[0] for b, lay in enumerate(_layers(k0, k1)) if lay}
        self.assertEqual(owners, OLD_OWNERS)
        self.assertEqual(int(np.sum(k1 == k0)), NB - len(OLD_OWNERS))

    def test_centre_touching_sub_bands_keep_exactly_their_own_layer(self):
        """The bands that owned a window keep it: their values are unchanged."""
        n0, n1 = gbs.wdm_window_layers(EDGES, LDF, IND_MIN_F, NFA, cover=True)
        o0, o1 = gbs.wdm_window_layers(EDGES, LDF, IND_MIN_F, NFA, cover=False)
        for b, layer in OLD_OWNERS.items():
            self.assertEqual(_layers(n0[b:b + 1], n1[b:b + 1]), [[layer]], b)
            self.assertEqual((n0[b], n1[b]), (o0[b], o1[b]), b)

    def test_a_band_straddling_a_tile_edge_owns_both_layers(self):
        # [3.3, 3.7] crosses the tile edge 3.5; [3.7, 4.2] lies in tile 4;
        # [4.2, 4.5] ends ON the tile edge 4.5 and gains nothing past it
        edges = np.array([3.3, 3.7, 4.2, 4.5, 4.9])
        k0, k1 = gbs.wdm_window_layers(edges, LDF, IND_MIN_F, NFA, cover=True)
        self.assertEqual(_layers(k0, k1), [[3, 4], [4], [4], [5]])

    def test_a_tile_aligned_band_owns_one_layer_not_three(self):
        edges = np.array([3.5, 4.5, 5.5])
        k0, k1 = gbs.wdm_window_layers(edges, LDF, IND_MIN_F, NFA, cover=True)
        self.assertEqual(_layers(k0, k1), [[4], [5]])

    def test_a_layer_wide_centre_aligned_grid_is_unchanged(self):
        """GB_SUBBAND_DIVISOR=1 (edges on centres): both rules give each band
        its two centre layers, so such a store's valve is not disturbed."""
        edges = np.arange(3, 7) * LDF
        new = gbs.wdm_window_layers(edges, LDF, IND_MIN_F, NFA, cover=True)
        old = gbs.wdm_window_layers(edges, LDF, IND_MIN_F, NFA, cover=False)
        np.testing.assert_array_equal(new[0], old[0])
        np.testing.assert_array_equal(new[1], old[1])
        self.assertEqual(_layers(*new), [[3, 4], [4, 5], [5, 6]])

    def test_windows_clip_to_the_active_band(self):
        # active layers are 2..7: a band below them owns nothing it can read
        edges = np.array([0.2, 0.9, 3.0, 7.4, 9.0])
        k0, k1 = gbs.wdm_window_layers(edges, LDF, IND_MIN_F, NFA, cover=True)
        self.assertTrue(np.all((k0 >= 0) & (k1 <= NFA) & (k1 >= k0)))
        self.assertEqual(_layers(k0, k1), [[], [2, 3], [3, 4, 5, 6, 7], [7]])


class Production9moGridTest(unittest.TestCase):
    """The REAL 9mo grid: the launcher's WDM settings (Nf 1440, Nt 6480,
    dt 2.5 s, MIN_FREQ 2.5e-4) and the uniform divisor-8 band edges
    (``GBSettings.init_band_structure``) over GB_MIN/MAX_FREQ 5.5e-4 /
    2.2e-2 -- 1232 sub-bands, the store's ``band_edges`` exactly."""

    @classmethod
    def setUpClass(cls):
        ws = WDMSettings(1440, 6480, 2.5, min_freq=2.5e-4, max_freq=2.5e-2,
                         force_backend="cpu")
        cls.ldf = float(ws.layer_df)
        cls.imf = int(ws.ind_min_f)
        cls.nfa = int(ws.Nf_active)
        k_lo = int(np.ceil(max(5.5e-4, cls.imf * cls.ldf) / cls.ldf)) * DIV
        k_hi = int(np.floor(min(2.2e-2, ws.ind_max_f * cls.ldf) / cls.ldf)) * DIV
        cls.edges = np.asarray([k * cls.ldf / DIV for k in range(k_lo, k_hi + 1)])

    def _rule(self, cover):
        return gbs.wdm_window_layers(self.edges, self.ldf, self.imf, self.nfa,
                                     cover=cover)

    def test_the_grid_is_the_production_one(self):
        self.assertEqual(self.edges.size - 1, 1232)
        self.assertEqual(self.imf, 2)
        self.assertAlmostEqual(self.ldf, 1.0 / (2 * 1440 * 2.5), places=18)

    def test_before_924_blind_after_none(self):
        o0, o1 = self._rule(False)
        n0, n1 = self._rule(True)
        self.assertEqual(int(np.sum(o1 == o0)), 924)
        self.assertEqual(int(np.sum(o1 - o0 == 1)), 308)
        self.assertEqual(int(np.sum(n1 == n0)), 0)
        self.assertTrue(np.all(n1 - n0 == 1))
        # the 308 window owners (band index 0 or 7 mod 8) keep their layer
        owners = o1 > o0
        self.assertEqual(sorted(set((np.flatnonzero(owners) % 8).tolist())), [0, 7])
        np.testing.assert_array_equal(n0[owners], o0[owners])

    def test_eight_sub_bands_share_each_layer_four_at_the_grid_ends(self):
        n0, _ = self._rule(True)
        _, counts = np.unique(n0, return_counts=True)
        self.assertEqual(counts.tolist(), [4] + [8] * 153 + [4])


class WindowRoutingTest(unittest.TestCase):
    """Which grid gets which rule, through the move's own ``_window_bins``."""

    def _aca(self, nw=1):
        d, r = _planes([{}] * nw, [])
        return _WDMACA(d, r)

    def test_the_band_grid_reads_the_covering_rule(self):
        m, _ = _wdm_move(1)
        k0, k1, nf, dof = m._window_bins(self._aca(), m.band_edges)
        self.assertEqual(_layers(k0, k1), [[int(c)] for c in COVER])
        self.assertEqual(nf, NFA)
        np.testing.assert_array_equal(dof, NC * NT)

    def test_the_cap_grid_keeps_the_centres_rule(self):
        """Sub-layer cap cells must stay unresolvable (source-attributed
        statistic), layer-wide ones resolvable -- the resolution test the
        covering rule would defeat."""
        with _no_tol_env():
            m, _ = _wdm_move(1)          # cap grid = the sub-layer band grid
            self.assertFalse(m._cap_cells_resolvable(self._aca()))
            m2, _ = _wdm_move(1, cap_edges=np.arange(3, 7) * LDF)
            self.assertTrue(m2._cap_cells_resolvable(self._aca()))

    def test_the_fd_branch_is_untouched(self):
        m, _ = _wdm_move(1)
        m._basis_settings = SimpleNamespace(df=1e-4, differential_component=1e-4)
        aca = SimpleNamespace(nchannels=2, start_freq_ind=np.array([50]),
                              data_length=40)
        edges = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
        a = m._window_bins(aca, edges, cover=True)
        b = m._window_bins(aca, edges, cover=False)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a[0], [0, 10, 20, 30])
        np.testing.assert_array_equal(a[1], [10, 20, 30, 40])


# ==========================================================================
# (b) the gain: shared by a layer's sub-bands, one geometry for three terms
# ==========================================================================
class _WDMFakeBuf:
    """A SubBandBuffer stand-in running the REAL slot likelihood code on
    narrow WDM slabs: one slot per (walker, band) cell, a 3-layer slab around
    the band's covering layer cut out of the ACA's residual and inverse PSD
    (``slab_min_f`` = the slab's first ABSOLUTE layer, as the buffer holds
    it)."""

    likelihood = SubBandBuffer.likelihood
    band_likelihoods = SubBandBuffer.likelihood
    _likelihood_slots = SubBandBuffer._likelihood_slots
    nchannels = NC
    tdi_channel_setup = "AE"
    use_template_arr = False
    _psd_shared_mirror = False
    xp = np
    SLAB = 3

    def __init__(self, aca, cells):
        self.settings = SimpleNamespace(differential_component=DIFF)
        self._per_band_data_shape = (NC, self.SLAB, NT)
        origins = np.array([min(max(COVER[b] - IND_MIN_F - 1, 0), NFA - self.SLAB)
                            for _, b in cells])
        self.slab_min_f = origins + IND_MIN_F
        self.band_buffer = np.stack([aca.residual[w, :, o:o + self.SLAB]
                                     for (w, _), o in zip(cells, origins)])
        self.psd_buffer = np.stack([aca.inv_psd[w, :, o:o + self.SLAB]
                                    for (w, _), o in zip(cells, origins)])


class SharedLayerGainTest(unittest.TestCase):
    """A sub-band inside a layer judges THAT layer's gain, like its 7 peers."""

    FIT = [{4: 0.6, 5: 1.0}, {4: 1.0, 5: 0.2}]
    SRC_LAYERS = (4, 5)

    def _setup(self):
        m, _bi = _wdm_move(2)
        d, r = _planes(self.FIT, self.SRC_LAYERS)
        aca = _WDMACA(d, r)
        return m, SimpleNamespace(analysis_container_arr=aca), aca, d, r

    def test_the_gain_of_a_sub_band_is_its_layer_gain(self):
        m, model, _aca, d, r = self._setup()
        gain = m._cap_stats_local(model, None)["band_gain"]
        self.assertEqual(gain.shape, (2, NB))
        for w in range(2):
            for b in range(NB):
                want = _layer_gain(d, r[w], int(COVER[b]))
                self.assertAlmostEqual(gain[w, b], want, places=9,
                                       msg=f"walker {w} band {b}")

    def test_the_eight_sub_bands_of_a_layer_share_one_value(self):
        m, model, _aca, d, r = self._setup()
        gain = m._cap_stats_local(model, None)["band_gain"]
        for layer in (4, 5):
            members = np.flatnonzero(COVER == layer)
            self.assertEqual(members.size, 8)
            for w in range(2):
                np.testing.assert_array_equal(gain[w, members],
                                              gain[w, members[0]])
                self.assertGreater(gain[w, members[0]], 100.0)
        # a layer no template touches: r = d there, so exactly 0
        for b in np.flatnonzero((COVER == 3) | (COVER == 6)):
            np.testing.assert_allclose(gain[:, b], 0.0, rtol=0, atol=1e-9)

    def test_the_residual_and_data_terms_cover_the_same_layer(self):
        m, model, aca, d, r = self._setup()
        band_lls = m._band_residual_lls(aca)
        dd = m._band_data_term(aca)
        for b in range(NB):
            k = int(COVER[b]) - IND_MIN_F
            for w in range(2):
                self.assertAlmostEqual(
                    band_lls[w, b], -0.5 * NORM * float(np.sum(r[w][:, k] ** 2)),
                    places=9)
                self.assertAlmostEqual(
                    dd[w, b], 0.5 * NORM * float(np.sum(d[:, k] ** 2)), places=9)

    def test_the_in_model_observation_equals_the_judge_time_gain(self):
        """The third consumer: the per-slot window, from narrow slabs, for
        EVERY cell -- the formerly blind ones included."""
        m, model, aca, _d, _r = self._setup()
        cells = [(w, b) for w in range(2) for b in range(NB)]
        buf = _WDMFakeBuf(aca, cells)
        w = np.array([c[0] for c in cells])
        b = np.array([c[1] for c in cells])
        obs = m._observed_band_gain(model, buf, np.arange(len(cells)), w, b)
        head = m._cap_stats_local(model, None)["band_gain"]
        self.assertTrue(np.all(np.isfinite(obs)), "a cell was not observed")
        np.testing.assert_allclose(obs, head[w, b], rtol=1e-12, atol=1e-9)

    def test_the_centres_rule_observed_zero_in_the_blind_bands(self):
        """The control: under the old geometry the same cells read 0."""
        m, model, aca, _d, _r = self._setup()
        with _centres_only():
            gain = m._cap_stats_local(model, None)["band_gain"]
        blind = np.array([b not in OLD_OWNERS for b in range(NB)])
        np.testing.assert_array_equal(gain[:, blind], 0.0)
        self.assertTrue(np.all(gain[:, [7, 8, 15, 16]] > 100.0))


# ==========================================================================
# (c) a previously blind pair no longer shuts on the timer
# ==========================================================================
class BlindPairJudgeTest(unittest.TestCase):
    """Band 5 (layer 4's tile, no centre of its own) is occupied on both
    walkers: walker 0 keeps improving its source, walker 1's is fitted.
    Band 13 (layer 5, no centre either) holds a fitted source on both."""

    # walker 0's band-5 gain (layer 4): 101, 274, 407, 499, 543, 559 nats --
    # every step beats the running max by more than the 4-nat tolerance
    FITS = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]

    def _run(self):
        m, bi = _wdm_move(2)
        occ = np.zeros((2, NB), dtype=np.int64)
        occ[:, 5] = 1
        occ[:, 13] = 1
        d, r = _planes([{4: self.FITS[0], 5: 1.0}, {4: 1.0, 5: 1.0}], (4, 5))
        aca = _WDMACA(d, r)
        shut_at = {}
        for t, f in enumerate(self.FITS):
            _d, aca.residual = _planes([{4: f, 5: 1.0}, {4: 1.0, 5: 1.0}],
                                       (4, 5))
            _judge(m, bi, aca, occ)
            for pair in ((0, 5), (1, 5), (0, 13), (1, 13)):
                if bi["band_rj_shutoff_w"][pair] and pair not in shut_at:
                    shut_at[pair] = t
        return bi, shut_at

    def test_an_improving_blind_pair_stays_open_a_converged_one_shuts(self):
        with _no_tol_env():
            bi, shut_at = self._run()
        self.assertNotIn((0, 5), shut_at, "the improving pair shut")
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"][0, 5], 0)
        # converged pairs: an improvement at the first judge (on -inf), then
        # CONV quiet judges
        for pair in ((1, 5), (0, 13), (1, 13)):
            self.assertEqual(shut_at.get(pair), CONV, pair)
        # the judged value is the layer's gain, never the blind 0
        self.assertGreater(float(bi["band_cold_logl_w"][0, 5]), 400.0)

    def test_the_centres_rule_shut_the_improving_pair_on_the_timer(self):
        """The paired control: the same schedule judged on the old geometry
        shuts walker 0's band 5 after the patience count, sources or not."""
        with _no_tol_env(), _centres_only():
            bi, shut_at = self._run()
        self.assertEqual(shut_at.get((0, 5)), CONV)
        np.testing.assert_array_equal(bi["band_cold_logl_w"][:, 5], 0.0)


# ==========================================================================
# (d) the persisted one-time re-learn
# ==========================================================================
class WindowRuleStampTest(unittest.TestCase):
    """``ensure_valve_window_rule``: what an absent stamp means."""

    def _bi(self):
        bi = {"nwalkers": 2, "num_bands": 3}
        ensure_search_shutoff_fields(bi, 3, per_walker=True)
        return bi

    def test_a_fresh_valve_is_stamped_with_the_current_rule(self):
        bi = self._bi()
        self.assertEqual(int(bi[VALVE_WINDOW_RULE_FIELD][0]), VALVE_WINDOW_RULE)
        self.assertEqual(VALVE_WINDOW_RULE, VALVE_WINDOW_RULE_COVERING)
        self.assertEqual(bi[VALVE_WINDOW_RULE_FIELD].dtype, np.dtype(np.int64))
        self.assertEqual(bi[VALVE_WINDOW_RULE_FIELD].shape, (1,))

    def test_a_stamp_less_record_that_learned_is_a_centres_rule_record(self):
        for name, value in (("band_cold_logl_max_w", 12.0),
                            ("band_cold_logl_peak_w", 3.0),
                            ("band_shutoff_best_w", -1.0),
                            ("band_shutoff_streak_w", 2)):
            bi = self._bi()
            bi.pop(VALVE_WINDOW_RULE_FIELD)
            bi[name][1, 2] = value
            self.assertEqual(ensure_valve_window_rule(bi),
                             VALVE_WINDOW_RULE_CENTRES, name)

    def test_a_stamp_less_record_that_learned_nothing_needs_no_relearn(self):
        bi = self._bi()
        bi.pop(VALVE_WINDOW_RULE_FIELD)
        self.assertEqual(ensure_valve_window_rule(bi), VALVE_WINDOW_RULE)

    def test_a_stored_stamp_is_kept(self):
        bi = self._bi()
        bi[VALVE_WINDOW_RULE_FIELD] = np.array([[1]])     # a step-axis row
        bi["band_cold_logl_max_w"][:] = 5.0
        self.assertEqual(ensure_search_shutoff_fields(bi, 3, per_walker=True),
                         "restored")
        np.testing.assert_array_equal(bi[VALVE_WINDOW_RULE_FIELD], [1])


def _learned(bi):
    """A record learned under the centres rule: finite max / best / peak,
    mid-window streaks, reset counts, two shut pairs."""
    bi["band_cold_logl_max_w"][:] = 0.0
    bi["band_cold_logl_max_w"][:, 0] = 900.0
    bi["band_shutoff_best_w"][:] = 0.0
    bi["band_cold_logl_peak_w"][:] = 0.0
    bi["band_shutoff_streak_w"][:] = 2
    bi["band_shutoff_reset_w"][:] = 1
    bi["band_rj_shutoff_w"][0, 1] = True
    bi["band_rj_shutoff_w"][2, 3] = True
    bi.pop(VALVE_WINDOW_RULE_FIELD, None)


class OneTimeRelearnTest(unittest.TestCase):
    """The judge re-learns a centres-rule record once, then judges normally."""

    def _judge_stash(self, m, bi, gain, occ):
        m.num_proposals += 1
        m._stage_band_gain = m._stage_band_lls = np.array(gain, dtype=float)
        m._stage_band_lls_stamp = m.num_proposals
        m._update_search_band_shutoff(None, _state(bi), _counts(occ))

    def _move(self, bi):
        nw, nb = bi["band_rj_shutoff_w"].shape
        m = _stage_move(shutoff=True, conv_iter=CONV, nwalkers=nw)
        m.band_edges = np.linspace(1e-3, 7e-3, nb + 1)
        m.num_bands = nb
        m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
        m._bind_shutoff_window(bi, np.shape(bi["band_rj_shutoff_w"]))
        return m

    def _relearn_lines(self, logs):
        return [l for l in logs.output if "window geometry v2" in l]

    def test_the_first_judge_relearns_once_and_keeps_the_shut_set(self):
        nw, nb = 4, 5
        bi = {"nwalkers": nw, "num_bands": nb}
        ensure_search_shutoff_fields(bi, nb, per_walker=True)
        _learned(bi)
        self.assertEqual(ensure_valve_window_rule(bi), VALVE_WINDOW_RULE_CENTRES)
        m = self._move(bi)
        gain = np.full((nw, nb), 250.0)
        occ = np.ones((nw, nb), dtype=np.int64)
        with _no_tol_env(), self.assertLogs(gbs.logger, level="INFO") as logs:
            self._judge_stash(m, bi, gain, occ)
        self.assertEqual(len(self._relearn_lines(logs)), 1, logs.output)
        self.assertIn("re-basing the per-(walker, band) running max",
                      self._relearn_lines(logs)[0])
        self.assertEqual(int(bi[VALVE_WINDOW_RULE_FIELD][0]), VALVE_WINDOW_RULE)
        # re-based: the stale 900 max (band 0) and the 0 maxima of the blind
        # bands are replaced by this step's gain under the new geometry...
        np.testing.assert_array_equal(bi["band_cold_logl_max_w"], 250.0)
        np.testing.assert_array_equal(bi["band_shutoff_best_w"], 250.0)
        np.testing.assert_array_equal(bi["band_cold_logl_peak_w"], 250.0)
        # ...while NOTHING about convergence moves (user ruling 2026-10-13:
        # "it should stay in its current state through a reset"): the
        # streaks and reset counts are exactly as stored, the step is not
        # judged (band 0, 650 nats below its stale max with a streak of 2,
        # would otherwise have shut here), and the two shut pairs stay shut
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"], 2)
        np.testing.assert_array_equal(bi["band_shutoff_reset_w"], 1)
        self.assertEqual(int(bi["band_rj_shutoff_w"].sum()), 2)
        self.assertFalse(bi["band_rj_shutoff_w"][1:, 0].any())
        # ...and the next judges are ordinary ones against the re-based
        # maxima: no second re-base; an improvement resets the streak
        with _no_tol_env(), self.assertLogs(gbs.logger, level="INFO") as logs2:
            self._judge_stash(m, bi, np.full((nw, nb), 300.0), occ)
        self.assertEqual(self._relearn_lines(logs2), [])
        np.testing.assert_array_equal(bi["band_cold_logl_max_w"], 300.0)
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"][0, 0], 0)
        np.testing.assert_array_equal(bi["band_shutoff_reset_w"][0, 0], 2)

    def test_a_quiet_round_after_the_rebase_continues_the_stored_streak(self):
        """The stored patience count carries through: a pair at streak 2
        that does not improve on the re-based max at the next judge reaches
        CONV and shuts THEN, not at the re-basing step."""
        nw, nb = 4, 5
        bi = {"nwalkers": nw, "num_bands": nb}
        ensure_search_shutoff_fields(bi, nb, per_walker=True)
        _learned(bi)
        m = self._move(bi)
        gain = np.full((nw, nb), 250.0)
        occ = np.ones((nw, nb), dtype=np.int64)
        with _no_tol_env():
            self._judge_stash(m, bi, gain, occ)          # the re-base: no judge
        self.assertEqual(int(bi["band_rj_shutoff_w"].sum()), 2)
        with _no_tol_env():
            self._judge_stash(m, bi, gain, occ)          # quiet: streak 2 -> 3
        self.assertTrue(bi["band_rj_shutoff_w"].all())
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"], CONV)

    def test_without_the_relearn_the_stale_max_would_shut_every_pair(self):
        """The control: judged on the centres-rule max (900 in band 0), a
        band whose gain now reads 250 cannot improve and runs its streak out
        -- and the streak of 2 it carries shuts it at the FIRST judge."""
        nw, nb = 4, 5
        bi = {"nwalkers": nw, "num_bands": nb}
        ensure_search_shutoff_fields(bi, nb, per_walker=True)
        _learned(bi)
        bi[VALVE_WINDOW_RULE_FIELD] = np.full(1, VALVE_WINDOW_RULE, np.int64)
        m = self._move(bi)
        with _no_tol_env():
            self._judge_stash(m, bi, np.full((nw, nb), 250.0),
                              np.ones((nw, nb), dtype=np.int64))
        self.assertTrue(bi["band_rj_shutoff_w"][:, 0].all())


class PersistedRelearnStoreTest(unittest.TestCase):
    """The store round trip: a store written before the stamp existed is
    re-learned at its first judge on a resume, the stamp is saved (earlier
    rows read the centres rule), and the next resume does not repeat it."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.fp = os.path.join(self.tmpdir.name, "window_rule_test.h5")
        self.rng = np.random.default_rng(20261013)

    def _backend(self, gb_reset_kwargs):
        nleaves, ndim = BRANCH_SHAPES["gb"]
        backend = GFHDFBackend(self.fp, sub_backend={"gb": GBHDFBackend},
                               sub_state_bases={"gb": GBState})
        backend.reset(NWALKERS, {"gb": ndim}, nleaves_max={"gb": nleaves},
                      ntemps=NTEMPS, branch_names=["gb"], nbranches=1,
                      rj=False, moves=None,
                      sub_reset_kwargs={"gb": dict(gb_reset_kwargs)})
        return backend

    def _save(self, backend, state):
        backend.grow(1, None)
        backend.save_step(state, np.ones((NTEMPS, NWALKERS)))

    def _resume(self, backend):
        resumed = backend.get_a_sample(backend.iteration - 1)
        resumed.sub_states["gb"].initialize_band_information(
            NWALKERS, NTEMPS, RT_EDGES, np.zeros((RT_NB, NTEMPS)),
            **{flag: True for flag in _PER_WALKER_FLAGS})
        return resumed

    def _move(self):
        m = _stage_move(shutoff=True, conv_iter=CONV, nwalkers=NWALKERS)
        m.band_edges = np.asarray(RT_EDGES, dtype=float)
        m.num_bands = RT_NB
        return m

    def _judge(self, m, state, gain):
        bi = state.sub_states["gb"].band_info
        m.num_proposals += 1
        m._stage_band_gain = m._stage_band_lls = np.array(gain, dtype=float)
        m._stage_band_lls_stamp = m.num_proposals
        m._update_search_band_shutoff(
            None, state, _counts(np.ones((NWALKERS, RT_NB), dtype=np.int64)))
        return bi

    def _pre_stamp_store(self):
        """The 9mo layout after 8e1f5f5a: the valve datasets, no stamp."""
        live = make_gb_state(self.rng, per_walker=True)
        backend = self._backend(live.sub_states["gb"].reset_kwargs)
        with h5py.File(self.fp, "a") as f:
            del f["global_fit/sub_backend/gb"][VALVE_WINDOW_RULE_FIELD]
        for _ in range(2):
            state = make_gb_state(self.rng, per_walker=True)
            _learned(state.sub_states["gb"].band_info)
            self._save(backend, state)
        with h5py.File(self.fp, "r") as f:
            self.assertNotIn(VALVE_WINDOW_RULE_FIELD, f["global_fit/sub_backend/gb"])
        return backend

    def test_a_pre_stamp_store_is_relearned_once_and_the_stamp_persists(self):
        backend = self._pre_stamp_store()
        # resume 1: no stamp + a learned record -> the centres rule
        state = self._resume(backend)
        bi = state.sub_states["gb"].band_info
        self.assertEqual(int(bi[VALVE_WINDOW_RULE_FIELD][0]),
                         VALVE_WINDOW_RULE_CENTRES)
        m = self._move()
        gain = np.full((NWALKERS, RT_NB), 250.0)
        with _no_tol_env(), self.assertLogs(gbs.logger, level="INFO") as logs:
            self._judge(m, state, gain)
        self.assertEqual(
            len([l for l in logs.output if "window geometry v2" in l]), 1)
        np.testing.assert_array_equal(bi["band_cold_logl_max_w"], 250.0)
        # the stored convergence state rides through the re-base untouched
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"], 2)
        self.assertEqual(int(bi["band_rj_shutoff_w"].sum()), 2)
        with self.assertLogs("lisatools.globalfit.hdfbackend", level="INFO") as slog:
            self._save(backend, state)
        created = [l for l in slog.output
                   if f"sub_backend/gb/{VALVE_WINDOW_RULE_FIELD} " in l]
        self.assertEqual(len(created), 1, slog.output)
        with h5py.File(self.fp, "r") as f:
            ds = f["global_fit/sub_backend/gb"][VALVE_WINDOW_RULE_FIELD]
            self.assertEqual(ds.shape, (3, 1))
            self.assertEqual(ds.dtype, np.dtype(np.int64))
            # rows written before the stamp existed read the centres rule
            np.testing.assert_array_equal(ds[:2], VALVE_WINDOW_RULE_CENTRES)
            np.testing.assert_array_equal(ds[2], VALVE_WINDOW_RULE)

        # resume 2: the stamp is restored, nothing is re-learned
        state2 = self._resume(backend)
        bi2 = state2.sub_states["gb"].band_info
        self.assertEqual(int(bi2[VALVE_WINDOW_RULE_FIELD][0]), VALVE_WINDOW_RULE)
        np.testing.assert_array_equal(bi2["band_cold_logl_max_w"], 250.0)
        np.testing.assert_array_equal(bi2["band_shutoff_streak_w"], 2)
        m2 = self._move()
        with _no_tol_env(), self.assertLogs(gbs.logger, level="INFO") as logs2:
            self._judge(m2, state2, np.full((NWALKERS, RT_NB), 300.0))
        self.assertEqual(
            [l for l in logs2.output if "window geometry v2" in l], [])
        # an ordinary judge against the re-based max: an improvement
        np.testing.assert_array_equal(bi2["band_cold_logl_max_w"], 300.0)
        np.testing.assert_array_equal(bi2["band_shutoff_streak_w"], 0)

    def test_a_rewind_onto_a_pre_stamp_row_relearns_again(self):
        """Row 1 was learned under the centres rule; a resume from it (a
        rewound store) must re-learn even though a later row is stamped."""
        backend = self._pre_stamp_store()
        state = self._resume(backend)
        with _no_tol_env():
            self._judge(self._move(), state, np.full((NWALKERS, RT_NB), 250.0))
        self._save(backend, state)
        rewound = backend.get_a_sample(1)
        rewound.sub_states["gb"].initialize_band_information(
            NWALKERS, NTEMPS, RT_EDGES, np.zeros((RT_NB, NTEMPS)),
            **{flag: True for flag in _PER_WALKER_FLAGS})
        self.assertEqual(
            int(rewound.sub_states["gb"].band_info[VALVE_WINDOW_RULE_FIELD][0]),
            VALVE_WINDOW_RULE_CENTRES)

    def test_a_fresh_store_carries_the_stamp_from_reset_and_never_relearns(self):
        live = make_gb_state(self.rng, per_walker=True)
        backend = self._backend(live.sub_states["gb"].reset_kwargs)
        with h5py.File(self.fp, "r") as f:
            self.assertIn(VALVE_WINDOW_RULE_FIELD, f["global_fit/sub_backend/gb"])
        self._save(backend, live)
        state = self._resume(backend)
        self.assertEqual(
            int(state.sub_states["gb"].band_info[VALVE_WINDOW_RULE_FIELD][0]),
            VALVE_WINDOW_RULE)
        with _no_tol_env(), self.assertLogs(gbs.logger, level="INFO") as logs:
            self._judge(self._move(), state, np.full((NWALKERS, RT_NB), 250.0))
        self.assertEqual([l for l in logs.output if "window geometry v2" in l], [])


if __name__ == "__main__":
    unittest.main()
