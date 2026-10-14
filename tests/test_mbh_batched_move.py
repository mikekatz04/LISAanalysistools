"""MBHBatchedLikeMove: batched windowed scoring against per-walker residuals
AND per-walker PSDs equals the base container path; batched expose/fold
restores the residual; chunking, hysteresis, routing, fallback.

The MBH twin of the EMRI pair (``python -m unittest tests.test_wdm_lookup_sum_kernel
tests.test_tdi_dense -v``): the batched-scorer half. Run on a GPU node::

    python -m unittest tests.test_mbh_windowed_signal_gen tests.test_mbh_batched_move \\
        tests.test_mbh_harness_noise -v

GPU == CPU (skips without a GPU backend): ``MBHBatchedGPUParityTest`` -- the same toy
(three walkers, walker 1 on another PSD, the same noise draws) on each backend:
batched fill, per-walker offsets, the batched scorer's logL, <d|h> / <h|h>, the
container path and the fold-back. Every class here is stub-based (no phentax):
seconds on a GPU node, ~a minute on the laptop."""
from __future__ import annotations

import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.utility import asnumpy

NF, NT, DT = 32, 128, 10.0
N = NF * NT
LAYER = NF * DT
NWALKERS, NTEMPS = 3, 2
T0 = 0.0  # data start == waveform_t0 (already on the lattice)
#: A mojito-scale ABSOLUTE data start (s). The stock erebor build's WDM
#: settings carry t0 = 0 (``WDMSettings.make_factory`` ignores times[0]) while
#: the data start and every merger time are absolute.
EPOCH = 9.7e7
# waveform basis: m1 m2 s1z s2z dist phi_ref iota psi alpha delta t_plunge
BASE_ROW = np.array([1.0e6, 5.0e5, 0.1, 0.2, 2.0e3, 0.3, 0.9, 0.4, 1.0, 0.2, 64 * LAYER])
WINDOW = dict(window_before=6 * LAYER, window_after=2 * LAYER, window_pad=2 * LAYER, window_margin=1 * LAYER)
# Scoring/fill window. WINDOW (above) pins the geometry arithmetic but is too
# tight to SCORE with: the WDM basis spreads even a pulse centred on layer 64
# over ~64 +- 8 layers (measured: 0.7% of <h|h> in layers 68-73, outside
# WINDOW's box), and a 2-layer pad leaves 1.1% error in the kept layers. Both
# are fixed numbers of LAYERS, independent of the layer duration, so the box
# and pad here are sized in layers to put the truncation below the 5e-2 bound.
WINDOW_WIDE = dict(window_before=10 * LAYER, window_after=10 * LAYER, window_pad=8 * LAYER, window_margin=1 * LAYER)


def _model_td(row, t0=T0, data_t0=T0):
    """Toy 'MBH': three channels of a Gaussian-enveloped chirp centred at
    t = t0 + t_plunge (t0 = the generator's epoch), amplitude m1/dist,
    inclination in channel 1. Always sampled on the DATA lattice, which
    starts at the ABSOLUTE time ``data_t0``."""
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, alpha, delta, t_plunge = [float(v) for v in row]
    t = np.arange(N) * DT + data_t0
    tc = t0 + t_plunge
    # Strain-scale amplitude (deviates from the plan's 3e-3, deliberately): at
    # 3e-3 against the ~1e-43 PSD lnL was ~1e42, where atol=5e-2 is below
    # float64 eps. At 1.5e-22 (noise 1e-23) SNR ~ 30, so the 5e-2 bound tests
    # the move, not the toy's units.
    env = np.exp(-0.5 * ((t - tc) / (1.5 * LAYER)) ** 2)
    amp = 1.5e-22 * (m1 / 1.0e6) * (2.0e3 / dist)
    ph = 2 * np.pi * (4e-3 * (t - tc) + 0.5 * 2e-6 * (t - tc) ** 2) + phi_ref
    x = amp * env * np.cos(ph)
    return np.stack([x, (0.5 + 0.1 * inc) * x, 0.25 * np.cos(psi) * x])


class _FastGen:
    """compute_tdi_channels contract on the full lattice; the adapter clips.
    Times are ABSOLUTE: the lattice starts at the data start ``data_t0``."""

    supports_batch = True
    t_plunge_snap = 0.0

    def __init__(self, data_t0=T0, gpu=False):
        self.n_calls = 0
        self.fail_with = None
        #: ``callable(rows) -> bool``: when set, ``fail_with`` is raised only for
        #: a launch whose (B, 11) row block satisfies it (a "bad row" trigger)
        self.fail_if = None
        self.data_t0 = float(data_t0)
        self.waveform_t0 = float(data_t0)   # snapped epoch == data start
        self.gpu = bool(gpu)                # output on the GPU, as the real generator's

    def compute_tdi_channels(self, *cols, **kwargs):
        self.n_calls += 1
        rows = np.stack([np.atleast_1d(np.asarray(c, dtype=float)) for c in cols], axis=1)
        if self.fail_with is not None and (self.fail_if is None or self.fail_if(rows)):
            raise self.fail_with
        t = np.arange(N) * DT + self.data_t0
        ch = np.stack([_model_td(r, self.waveform_t0, self.data_t0) for r in rows])
        if np.ndim(cols[0]) == 0:
            return self._out(t, ch[0])
        return self._out(np.broadcast_to(t, (rows.shape[0], N)).copy(), ch)

    def _out(self, t, ch):
        if not self.gpu:
            return t, ch
        import cupy as cp

        return cp.asarray(t), cp.asarray(ch)


def _slow_gen(*params, apply_transform=False, leaf_inds=None, **kwargs):
    """The containers' installed (slow, full-grid) generator: same model, on
    the STOCK (possibly off-lattice) epoch the rows' t_plunge is relative to.
    Like the stock generator it places on the ABSOLUTE data lattice and
    transforms onto the containers' own settings (whatever their ``t0``), on
    their backend (``_slow_gen.backend``, set by :func:`_build`)."""
    return TDSignal(
        _slow_gen.wdm.xp.asarray(_model_td(params, _slow_gen.t0_stock, _slow_gen.data_t0)),
        TDSettings(N, DT, t0=_slow_gen.data_t0, force_backend=_slow_gen.backend),
    ).transform(_slow_gen.wdm)


_slow_gen.backend = "cpu"


def _toy_wdm(t0, backend="cpu"):
    return WDMSettings(NF, NT, DT, t0=t0, min_freq=2e-3, max_freq=2e-2, force_backend=backend)


def _build(t0_stock=None, data_t0=T0, container_t0=None, adapter_t0=None, t0_abs=None, backend="cpu"):
    """``t0_stock != data_t0``: the stock epoch sits OFF the data lattice. The
    fast generator is built on the snapped epoch ``data_t0`` and, as in the
    Task 5 wiring, the adapter itself carries ``waveform_t0 = data_t0``
    (snapped) and ``t_plunge_snap = data_t0 - t0_stock``.

    ``data_t0`` is the ABSOLUTE data start (the toy generators' times are
    absolute). ``container_t0`` is the ``t0`` the CONTAINERS' data settings
    carry (default ``data_t0``); ``adapter_t0`` the ``t0`` of the settings the
    adapter is built on (default: the containers' settings object itself);
    ``t0_abs`` is passed to the adapter when given. A stock erebor mojito
    build has ``container_t0 = 0`` and the adapter settings' ``t0`` equal to
    0 or ``data_t0`` depending on build order.

    ``backend`` (default ``"cpu"``): every settings object, the containers'
    data and PSDs, the stub generators' output and the container array
    (``gpus=[0]`` on a GPU backend, as the benchmark builds it) live there; the
    noise draws are the same on every backend."""
    from lisatools.sources.batching import MBHWindowedWDMSignalGen

    t0_stock = data_t0 if t0_stock is None else float(t0_stock)
    wdm = _toy_wdm(data_t0 if container_t0 is None else container_t0, backend)
    _slow_gen.wdm = wdm
    _slow_gen.t0_stock = float(t0_stock)
    _slow_gen.data_t0 = float(data_t0)
    _slow_gen.backend = backend
    rng = np.random.default_rng(7)
    acs_list = []
    models = ["scirdv1", "mrdv1", "scirdv1"]      # walker 1 has a DIFFERENT PSD
    for w in range(NWALKERS):
        noise = 1e-23 * rng.normal(size=(3, wdm.Nf_active, NT))
        ac = AnalysisContainer(WDMSignal(wdm.xp.asarray(noise), wdm), XYZ2SensitivityMatrix(wdm, model=models[w]))
        ac.signal_gen = {"mbh": _slow_gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list, gpus=None if backend == "cpu" else [0])
    fast = _FastGen(data_t0, gpu=backend != "cpu")
    ad_wdm = wdm if adapter_t0 is None else _toy_wdm(adapter_t0, backend)
    kw = {} if t0_abs is None else dict(t0_abs=t0_abs)
    adapter = MBHWindowedWDMSignalGen(fast, ad_wdm, nchannels=3, tukey_alpha=0.0, **kw)
    if t0_stock != data_t0:
        adapter.waveform_t0 = float(data_t0)
        adapter.t_plunge_snap = float(data_t0) - float(t0_stock)
    return acs, adapter, fast, wdm


def _build_move(acs, adapter, batch_max_size=2, **window):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist
    from lisatools.globalfit.moves import MBHBatchedLikeMove

    betas = 1 / 1.2 ** np.arange(NTEMPS)
    priors = {"mbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
    kw = dict(WINDOW_WIDE)
    kw.update(window)
    move = MBHBatchedLikeMove(
        "mbh", (NTEMPS, NWALKERS, 1, 11), None, {}, {}, acs, 1, None, priors,
        [(StretchMove(), 1.0)], betas_all=np.tile(betas, (1, 1)),
        batched_gen=adapter, batch_max_size=batch_max_size, name="mbh batched test", **kw,
    )
    move._current_leaf = 0
    return move


def _cold_rows(seed=3):
    rng = np.random.default_rng(seed)
    rows = np.tile(BASE_ROW, (NWALKERS, 1))
    rows[:, 4] *= rng.uniform(0.8, 1.2, NWALKERS)          # dist
    rows[:, 10] += rng.uniform(-0.2, 0.2, NWALKERS) * LAYER  # t_plunge
    return rows


def _proposal_rows(n, seed=11):
    rng = np.random.default_rng(seed)
    rows = np.tile(BASE_ROW, (n, 1))
    rows[:, 4] *= rng.uniform(0.7, 1.3, n)
    rows[:, 10] += rng.uniform(-0.3, 0.3, n) * LAYER
    rows[:, 5] += rng.uniform(-1, 1, n)
    return rows


class WindowLayersTest(unittest.TestCase):
    def test_geometry_is_a_function_of_durations_only(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = WDMSettings(NF, NT, DT, t0=T0, force_backend="cpu")
        a = mbh_window_layers(wdm, T0 + 64 * LAYER, **WINDOW)
        b = mbh_window_layers(wdm, T0 + 70 * LAYER, **WINDOW)
        self.assertEqual(a["Nt_keep"], b["Nt_keep"])
        self.assertEqual(a["n_pad"], 2)
        self.assertEqual(b["n_start"] - a["n_start"], 6)
        self.assertEqual((a["Nt_keep"] + 2 * a["n_pad"]) % 2, 0)
        # kept box covers [t - before - margin, t + after + margin]
        self.assertLessEqual(a["n_start"] * LAYER, 64 * LAYER - 7 * LAYER)
        self.assertGreaterEqual((a["n_start"] + a["Nt_keep"]) * LAYER, 64 * LAYER + 3 * LAYER)

    def test_window_clamps_at_grid_edges(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = WDMSettings(NF, NT, DT, t0=T0, force_backend="cpu")
        lo = mbh_window_layers(wdm, T0 + 1 * LAYER, **WINDOW)
        # the segment starts AT the grid edge: no low pad, the box reaches layer 0
        self.assertEqual((lo["n_start"], lo["n_pad_lo"]), (0, 0))
        self.assertEqual(lo["n_pad_hi"], 2 * lo["n_pad"])
        hi = mbh_window_layers(wdm, T0 + 127 * LAYER, **WINDOW)
        self.assertEqual(hi["n_start"] + hi["Nt_keep"], NT)
        self.assertEqual((hi["n_pad_hi"], hi["n_pad_lo"]), (0, 2 * hi["n_pad"]))
        # a window longer than the data no longer raises (2026-09-30): it is
        # clamped to the whole grid (WindowClampShortSpanTest)
        big = mbh_window_layers(wdm, T0 + 64 * LAYER, window_before=100 * LAYER, window_after=100 * LAYER, window_pad=LAYER, window_margin=0.0)
        self.assertEqual((big["n_start"], big["Nt_keep"], big["n_pad_lo"], big["n_pad_hi"]), (0, NT, 0, 0))

    def test_absolute_layer0_time_overrides_settings_t0(self):
        """Stock erebor build: the settings carry t0 = 0 while the merger time
        is absolute. ``t0_abs`` (the data start) places the window; reading
        ``wdm.t0`` clamps every window to the grid end."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = WDMSettings(NF, NT, DT, t0=0.0, force_backend="cpu")
        ref = mbh_window_layers(wdm, 64 * LAYER, **WINDOW)
        got = mbh_window_layers(wdm, EPOCH + 64 * LAYER, t0_abs=EPOCH, **WINDOW)
        self.assertEqual(got, ref)
        self.assertEqual(ref["n_start"], 64 - 7)
        stale = mbh_window_layers(wdm, EPOCH + 64 * LAYER, **WINDOW)
        self.assertEqual(stale["n_start"] + stale["Nt_keep"], NT)   # the defect: clamped


class WindowEdgeProductionGeometryTest(unittest.TestCase):
    """Production durations (90 d before, 10 d after, 4 d pad, 1 d margin) on
    a 120-day grid of 12-h layers -- the mojito check grid's layer, with
    Nf = 32 so construction is cheap. Before the edge fix the kept box was
    clamped ``n_pad`` layers (4 d) inside each grid edge, so a merger in the
    last 4 d of the data sat OUTSIDE the box (never scored, never filled)."""

    L = 43200.0
    T0D = 1.0e7
    PROD = dict(window_before=90 * 86400.0, window_after=10 * 86400.0, window_pad=4 * 86400.0, window_margin=86400.0)

    def _wdm(self, **kw):
        return WDMSettings(32, 240, self.L / 32, t0=self.T0D, force_backend="cpu", **kw)

    def _adapter_accepts(self, wdm, g):
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        ad = MBHWindowedWDMSignalGen(object(), wdm, nchannels=3)
        ad.set_window(g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"])
        a = ad.geometry
        self.assertEqual(a["s0"] % 2, 0, g)
        self.assertGreaterEqual(a["s0"], 0)
        self.assertLessEqual(a["s0"] + a["Nt_seg"], int(wdm.Nt))
        return a

    def _covers(self, wdm, g, t_merge):
        rel = (t_merge - float(wdm.t0)) / float(wdm.layer_dt)
        return g["n_start"] <= rel < g["n_start"] + g["Nt_keep"]

    def test_merger_one_day_before_data_end_is_kept(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm()
        t_m = self.T0D + 240 * self.L - 86400.0
        g = mbh_window_layers(wdm, t_m, **self.PROD)
        self.assertTrue(self._covers(wdm, g, t_m), g)
        self.assertEqual(g["n_start"] + g["Nt_keep"], 240)
        self.assertEqual(g["n_pad_hi"], 0)
        self.assertEqual(g["n_pad_lo"], 2 * g["n_pad"])
        self._adapter_accepts(wdm, g)

    def test_merger_one_day_after_data_start_is_kept(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm()
        t_m = self.T0D + 86400.0
        g = mbh_window_layers(wdm, t_m, **self.PROD)
        self.assertTrue(self._covers(wdm, g, t_m), g)
        self.assertEqual((g["n_start"], g["n_pad_lo"]), (0, 0))
        self.assertEqual(g["n_pad_hi"], 2 * g["n_pad"])
        self._adapter_accepts(wdm, g)

    def test_sweep_parity_constant_segment_and_coverage(self):
        """Every merger time across the grid: the adapter accepts the geometry,
        the segment starts on an even layer, its length never changes except
        by the adapter's +2 odd-start parity growth, and the box covers the
        merger (the box can only reach [0, Nt), so every merger inside the
        data is covered)."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm()
        lengths = set()
        for day in np.arange(0.25, 120.0, 1.0):
            t_m = self.T0D + day * 86400.0
            g = mbh_window_layers(wdm, t_m, **self.PROD)
            self.assertTrue(self._covers(wdm, g, t_m), (day, g))
            self.assertEqual(g["n_pad_lo"] + g["n_pad_hi"], 2 * g["n_pad"])
            a = self._adapter_accepts(wdm, g)
            lengths.add(a["Nt_seg"])
        n = g["Nt_keep"] + 2 * g["n_pad"]
        self.assertTrue(lengths <= {n, n + 2}, lengths)

    def test_active_time_box_is_respected(self):
        """A narrowed active time box: the KEPT box stays inside it (the
        container can only slice there) while the segment may use the grid
        layers beyond it as pad."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm(min_time=9.5 * self.L, max_time=229.5 * self.L)
        lo_a, hi_a = int(wdm.ind_min_t), int(wdm.ind_max_t) + 1
        self.assertGreater(lo_a, 0)
        self.assertLess(hi_a, 240)
        t_m = self.T0D + (hi_a * self.L) - 86400.0
        g = mbh_window_layers(wdm, t_m, **self.PROD)
        self.assertEqual(g["n_start"] + g["Nt_keep"], hi_a)
        self.assertTrue(self._covers(wdm, g, t_m))
        self.assertEqual(g["n_pad_hi"], min(g["n_pad"], 240 - hi_a))
        self._adapter_accepts(wdm, g)
        t_m = self.T0D + lo_a * self.L + 86400.0
        g = mbh_window_layers(wdm, t_m, **self.PROD)
        self.assertEqual(g["n_start"], lo_a)
        self.assertEqual(g["n_pad_lo"], min(g["n_pad"], lo_a))
        self._adapter_accepts(wdm, g)


class WindowClampShortSpanTest(unittest.TestCase):
    """Data spans SHORTER than the window (2026-09-30: batched is the default,
    so a 3-month run and the lite smokes must build). The configured window
    is clamped -- kept box to the active box, segment to the grid, pads
    shrinking to what fits (possibly 0) -- logged once; only a geometry with
    no valid segment raises."""

    L = WindowEdgeProductionGeometryTest.L
    T0D = WindowEdgeProductionGeometryTest.T0D
    PROD = WindowEdgeProductionGeometryTest.PROD
    _adapter_accepts = WindowEdgeProductionGeometryTest._adapter_accepts
    _covers = WindowEdgeProductionGeometryTest._covers
    LOGGER = "lisatools.globalfit.moves.mbhbatchedmove"

    def setUp(self):
        import lisatools.globalfit.moves.mbhbatchedmove as mod

        mod._CLAMP_LOGGED.clear()
        self.addCleanup(mod._CLAMP_LOGGED.clear)

    def _wdm(self, nt, **kw):
        return WDMSettings(32, nt, self.L / 32, t0=self.T0D, force_backend="cpu", **kw)

    def test_three_month_grid_clamps_to_the_whole_active_box_and_grid(self):
        """90-day grid (180 x 12-h layers), 2-layer edge crop: the 102-d box
        cannot fit, so the kept box is the whole active box [2, 178) and the
        segment the whole grid (pads 2 / 2) for EVERY merger time -- logged
        once, never again."""
        import logging

        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm(180, min_time=2 * self.L, max_time=178.5 * self.L)
        self.assertEqual((int(wdm.ind_min_t), int(wdm.ind_max_t) + 1), (2, 179))
        act_lo, act_hi = 2, 179
        with self.assertLogs(self.LOGGER, logging.INFO) as cm:
            first = mbh_window_layers(wdm, self.T0D + 45 * 86400.0, **self.PROD)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("window clamped to the data span", cm.output[0])
        self.assertIn("(the whole active box)", cm.output[0])
        self.assertIn("(the whole grid)", cm.output[0])
        with self.assertNoLogs(self.LOGGER, logging.INFO):
            for day in np.arange(0.25, 90.0, 1.0):
                t_m = self.T0D + day * 86400.0
                g = mbh_window_layers(wdm, t_m, **self.PROD)
                self.assertEqual(g, first)                      # merger-independent
                if act_lo * self.L <= day * 86400.0 < act_hi * self.L:
                    self.assertTrue(self._covers(wdm, g, t_m), (day, g))
        self.assertEqual((first["n_start"], first["Nt_keep"]), (act_lo, act_hi - act_lo))
        self.assertEqual((first["n_pad_lo"], first["n_pad_hi"]), (2, 1))
        self.assertEqual(first["n_pad"], 8)                     # the CONFIGURED pad
        a = self._adapter_accepts(wdm, first)
        self.assertEqual((a["s0"], a["Nt_seg"]), (0, 180))      # the whole grid

    def test_segment_only_clamp_shrinks_the_pads(self):
        """105-day grid: the kept box fits (206 layers) but box + 2 x 8 pad
        layers does not -- the segment becomes the whole grid and the pads
        shrink to the 4 spare layers, split by where the box sits."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm(210)
        for day in (2.0, 92.0, 93.5, 104.0):
            t_m = self.T0D + day * 86400.0
            g = mbh_window_layers(wdm, t_m, **self.PROD)
            self.assertEqual(g["Nt_keep"], 206)
            self.assertEqual(g["n_pad_lo"] + g["n_pad_hi"], 210 - 206)
            self.assertGreaterEqual(min(g["n_pad_lo"], g["n_pad_hi"]), 0)
            self.assertTrue(self._covers(wdm, g, t_m), (day, g))
            a = self._adapter_accepts(wdm, g)
            self.assertEqual((a["s0"], a["Nt_seg"]), (0, 210))

    def test_odd_clamped_box_gets_the_spare_layer_as_pad(self):
        """Active box of 195 layers inside a 240-layer grid: the kept box is
        the (odd) active box and the segment 195 + 16 + 1 = 212 layers (even)
        with the spare layer as high pad; the adapter's even-start rule still
        holds at an even and at an odd segment start."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        for lo, s0_expect in ((20, 12), (21, 13)):
            wdm = self._wdm(240, min_time=lo * self.L, max_time=(lo + 194.5) * self.L)
            self.assertEqual((int(wdm.ind_min_t), int(wdm.ind_max_t) + 1), (lo, lo + 195))
            g = mbh_window_layers(wdm, self.T0D + 60 * 86400.0, **self.PROD)
            self.assertEqual((g["n_start"], g["Nt_keep"]), (lo, 195))
            self.assertEqual((g["n_pad_lo"], g["n_pad_hi"]), (8, 9))
            self.assertEqual(g["n_start"] - g["n_pad_lo"], s0_expect)
            a = self._adapter_accepts(wdm, g)
            self.assertEqual(a["Nt_seg"] % 2, 0)

    def test_a_fitting_window_is_not_clamped_or_logged(self):
        import logging

        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = self._wdm(240)
        with self.assertNoLogs(self.LOGGER, logging.INFO):
            g = mbh_window_layers(wdm, self.T0D + 60 * 86400.0, **self.PROD)
        self.assertEqual(g["n_pad_lo"] + g["n_pad_hi"], 2 * g["n_pad"])
        self.assertEqual(g["Nt_keep"], 206)

    def test_impossible_geometry_still_raises(self):
        """No valid segment: an empty active box, or a whole-grid segment on an
        odd-length grid (not a valid WDM grid; duck-typed stubs)."""
        from types import SimpleNamespace

        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        empty = SimpleNamespace(layer_dt=self.L, t0=0.0, Nt=180, ind_min_t=10, ind_max_t=9)
        with self.assertRaisesRegex(ValueError, "cannot be placed"):
            mbh_window_layers(empty, 50 * self.L, **self.PROD)
        odd = SimpleNamespace(layer_dt=self.L, t0=0.0, Nt=181, ind_min_t=0, ind_max_t=180)
        with self.assertRaisesRegex(ValueError, "cannot be placed"):
            mbh_window_layers(odd, 50 * self.L, **self.PROD)
        # the same odd grid is fine when the clamped segment stays inside it
        odd_wide = SimpleNamespace(layer_dt=self.L, t0=0.0, Nt=241, ind_min_t=20, ind_max_t=214)
        g = mbh_window_layers(odd_wide, 60 * 86400.0, **self.PROD)
        self.assertEqual((g["Nt_keep"], g["n_pad_lo"] + g["n_pad_hi"]), (195, 17))


#: A window far longer than the toy grid (128 layers): clamped to the whole
#: grid with zero pads -- the stock full-grid transform.
WINDOW_HUGE = dict(window_before=200 * LAYER, window_after=10 * LAYER, window_pad=8 * LAYER, window_margin=1 * LAYER)


class MBHBatchedWholeGridClampParityTest(unittest.TestCase):
    """The batched move on a CLAMPED (whole-grid, zero-pad) window equals the
    container path: a whole-grid segment IS the stock full-grid transform, so
    the parity is at float rounding, far inside the windowed 5e-2 bound."""

    def test_whole_grid_window_matches_container_path(self):
        import lisatools.globalfit.moves.mbhbatchedmove as mod

        mod._CLAMP_LOGGED.clear()
        self.addCleanup(mod._CLAMP_LOGGED.clear)
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter, **WINDOW_HUGE)
        before = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)          # expose (batched fill)
        move.setup_likelihood_here(cold)
        g = move._leaf_windows[0]
        self.assertEqual((g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"]), (0, NT, 0, 0))
        self.assertEqual((adapter.geometry["s0"], adapter.geometry["Nt_seg"]), (0, NT))
        rows = _proposal_rows(6)
        idx = np.array([0, 1, 2, 1, 0, 1])
        fast_ll = move.compute_like(rows, idx)
        slow_ll = move.compute_acs_like(rows, idx)
        d = float(np.abs(fast_ll - slow_ll).max())
        # template-level: windowed (whole-grid segment) vs the stock full-grid transform
        tmpl = np.asarray(move._on_container_box(move._generate(adapter, rows[:2])).arr)
        ref = np.stack([np.asarray(_slow_gen(*r).arr) for r in rows[:2]])
        rel = float(np.abs(tmpl - ref).max() / np.abs(ref).max())
        print(f"[mbh batched whole-grid clamp] max |fast-slow| = {d:.3e} on lnL ~ "
              f"{np.abs(slow_ll).max():.3e}; template max rel diff {rel:.3e}")
        self.assertLess(rel, 1e-12)
        np.testing.assert_allclose(fast_ll, slow_ll, rtol=0, atol=1e-6)
        self.assertEqual(move.n_batch_fallbacks, 0)
        move.add_back_in_cold_chain_sources(cold)     # fold-back restores the residual
        for ac, b in zip(acs.acs.flatten(), before):
            np.testing.assert_allclose(ac.data.arr, b, rtol=0, atol=1e-12 * np.abs(b).max())


class MBHBatchedParityTest(unittest.TestCase):
    def setUp(self):
        self.acs, self.adapter, self.fast, self.wdm = _build()
        self.move = _build_move(self.acs, self.adapter)
        self.cold = _cold_rows()
        self.move.remove_cold_chain_sources(self.cold)     # expose (sets the window)
        self.move.setup_likelihood_here(self.cold)

    def _rows(self, n, seed=11):
        return _proposal_rows(n, seed)

    def test_batched_matches_container_path_per_walker_psd(self):
        rows = self._rows(6)
        idx = np.array([0, 1, 2, 1, 0, 1])
        fast = self.move.compute_like(rows, idx)
        slow = self.move.compute_acs_like(rows, idx)
        print(f"[mbh batched parity] max |fast-slow| = {np.abs(fast - slow).max():.3e} on lnL ~ {np.abs(slow).max():.3e}")
        np.testing.assert_allclose(fast, slow, rtol=0, atol=5e-2)
        self.assertEqual(self.move.n_batch_fallbacks, 0)
        self.assertTrue(np.all(np.isfinite(self.move._last_d_h)))
        # walker 1's PSD differs: scoring the same row against walker 0 vs 1 must differ
        same = np.tile(rows[:1], (2, 1))
        vals = self.move.compute_like(same, np.array([0, 1]))
        self.assertNotAlmostEqual(vals[0] - self.move._exposed_offset[0], vals[1] - self.move._exposed_offset[1], places=3)

    def test_chunk_size_does_not_change_results(self):
        rows = self._rows(5)
        idx = np.array([0, 1, 2, 0, 1])
        a = self.move.compute_like(rows, idx)
        big = _build_move(self.acs, self.adapter, batch_max_size=16)
        big._exposed_offset = self.move._exposed_offset
        big._leaf_windows = self.move._leaf_windows
        b = big.compute_like(rows, idx)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_inner_products_batched_per_walker_not_per_row(self):
        """One chunk of 4 rows over 2 walkers: inner_product is called once
        per (walker, <r|h> / <h|h>) with a batched template, never per row
        (each unbatched call ``.item()``s -- a device->host sync on GPU)."""
        from unittest import mock

        import lisatools.globalfit.moves.mbhbatchedmove as mod

        big = _build_move(self.acs, self.adapter, batch_max_size=4)
        big._exposed_offset = self.move._exposed_offset
        big._leaf_windows = self.move._leaf_windows
        rows = self._rows(4)
        idx = np.array([0, 1, 0, 1])
        with mock.patch.object(mod, "inner_product", wraps=mod.inner_product) as spy:
            out = big.compute_like(rows, idx)
        self.assertLessEqual(spy.call_count, 2 * len(np.unique(idx)))
        np.testing.assert_allclose(out, self.move.compute_like(rows, idx), rtol=1e-12, atol=0)

    def test_non_finite_row_gets_sentinel(self):
        rows = self._rows(2)
        rows[1, 0] = np.nan
        out = self.move.compute_like(rows, np.array([0, 1]))
        self.assertTrue(np.isfinite(out[0]))
        self.assertEqual(out[1], -1e300)

    def test_compute_like_requires_armed_offset(self):
        move = _build_move(self.acs, self.adapter)
        with self.assertRaises(RuntimeError):
            move.compute_like(self._rows(1), np.array([0]))

    def test_verify_prev_logl_passes_at_default_tolerance(self):
        rows = self._rows(NWALKERS * NTEMPS)
        idx = np.tile(np.arange(NWALKERS), NTEMPS)
        prev = self.move.compute_like(rows, idx).reshape(NTEMPS, NWALKERS)
        with self.assertNoLogs("lisatools.globalfit.moves.mbhbatchedmove", level="WARNING"):
            self.move._verify_prev_logl(prev, rows, idx, 0)   # must not raise/warn

    def test_verify_prev_logl_gates_on_the_cold_rung_only(self):
        """Hot-rung disagreements never warn/raise; a cold-rung one does, and
        its message carries the hot-rung max|diff| as information."""
        import os

        rows = self._rows(NWALKERS * NTEMPS)
        idx = np.tile(np.arange(NWALKERS), NTEMPS)
        prev = self.move.compute_like(rows, idx).reshape(NTEMPS, NWALKERS)
        hot_bad = prev.copy()
        hot_bad[1:] += 7.0
        self.move.check_ll_mode = "strict"
        with self.assertNoLogs("lisatools.globalfit.moves.mbhbatchedmove", level="WARNING"):
            self.move._verify_prev_logl(hot_bad, rows, idx, 0)   # must not raise
        both_bad = hot_bad.copy()
        both_bad[0, 1] += 3.0
        with self.assertRaises(ValueError) as cm:
            self.move._verify_prev_logl(both_bad, rows, idx, 0)
        msg = str(cm.exception)
        self.assertIn("COLD", msg)
        self.assertRegex(msg, r"max\|diff\|=3\.0\d*e\+00")
        self.assertRegex(msg, r"hot rungs, not gating: max\|diff\|=7\.0\d*e\+00")

    def test_check_cadence_defaults_to_every_10th_visit(self):
        import os
        from unittest import mock

        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

        env = {k: v for k, v in os.environ.items() if k != "MBH_CHECK_LL_EVERY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_build_move(self.acs, self.adapter).check_ll_every, 10)
        with mock.patch.dict(os.environ, {"MBH_CHECK_LL_EVERY": "3"}):
            self.assertEqual(_build_move(self.acs, self.adapter).check_ll_every, 3)
        # the base path keeps checking every visit
        self.assertEqual(ResidualAddOneRemoveOneMove._check_ll_every_default, "1")

    def test_telemetry_split_outside_box_and_leaf_label(self):
        import logging

        rows = self._rows(3)
        rows[2, 10] += 30 * LAYER            # merger far past the kept box
        self.move.compute_like(rows, np.array([0, 1, 2]))
        st = self.move._stats
        self.assertEqual(st["outside_box"], 1)
        self.assertGreater(st["gen_s"], 0.0)
        self.assertGreater(st["score_s"], 0.0)
        self.move._current_leaf = 1          # flush happens at the NEXT leaf's setup
        try:
            with self.assertLogs("lisatools.globalfit.moves.mbhbatchedmove", logging.INFO) as cm:
                self.move._flush_stats()
        finally:
            self.move._current_leaf = 0
        msg = "\n".join(cm.output)
        self.assertIn("leaf 0:", msg)
        self.assertIn("outside_box=1", msg)
        self.assertIn("outside_data=0", msg)
        self.assertIn("generate=", msg)
        self.assertIn("score=", msg)
        self.assertIsNone(self.move._stats["leaf"])

    def test_telemetry_merger_past_the_data_end(self):
        """A row merging after the data end is truncated by the DATA, not by
        the box, when the box is clamped at the data end: it counts only in
        ``outside_data``. With the box stopping short of the data end the same
        row's in-data inspiral IS cut by the box: ``outside_box`` too."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        rows = self._rows(3)
        rows[:, 10] = np.array([130.0, 110.0, 90.0]) * LAYER   # data ends at layer 128
        geom = mbh_window_layers(self.wdm, T0 + 127 * LAYER, **WINDOW_WIDE)
        self.assertEqual(geom["n_start"] + geom["Nt_keep"], NT)   # clamped at the data end
        self.assertGreater(geom["n_start"], 90)
        geom["t_ref"] = T0 + 127 * LAYER
        saved = self.move._leaf_windows[0]
        self.move._leaf_windows[0] = geom
        try:
            self.move._stats = self.move._new_stats()
            out = self.move.compute_like(rows, np.array([0, 1, 2]))
            st = self.move._stats
            # only the layer-90 row (in the data, before the box) is cut by the box
            self.assertEqual((st["outside_box"], st["outside_data"]), (1, 1))
            self.assertTrue(np.all(np.isfinite(out)))
        finally:
            self.move._leaf_windows[0] = saved
        self.move._stats = self.move._new_stats()
        self.move.compute_like(rows[:1], np.array([0]))           # box around layer 64
        st = self.move._stats
        self.assertEqual((st["outside_box"], st["outside_data"]), (1, 1))

    def test_batch_refusal_falls_back_to_container_path(self):
        from lisatools.utils.exceptions import BatchNotLaunchable

        rows = self._rows(3)
        idx = np.array([0, 1, 2])
        slow = self.move.compute_acs_like(rows, idx)
        self.fast.fail_with = BatchNotLaunchable("test refusal")
        out = self.move.compute_like(rows, idx)
        np.testing.assert_allclose(out, slow, rtol=0, atol=0)
        self.assertEqual(self.move.n_batch_fallbacks, 2)   # one per chunk of 2
        self.fast.fail_with = None

    # -- JAX runtime failures of the waveform model (job 760, 2026-10-14) ----
    # phentax raised ``_EquinoxRuntimeError: The maximum number of steps was
    # reached in the nonlinear solver`` (wrapped as ``jax.errors.JaxRuntimeError:
    # INTERNAL: CpuCallback error``) for ONE hot-rung row of MBHB 10, which took
    # the whole vmapped chunk, then rank 1, then the 1-year run down. The move
    # must isolate the row, score it at the floor and carry on.

    def test_generator_runtime_failure_isolates_the_row(self):
        rows = self._rows(5)
        idx = np.array([0, 1, 2, 0, 1])
        baseline = self.move.compute_like(rows, idx)
        bad = rows.copy()
        bad[3, 0] = 7.7e9                                   # the trigger (m1 absurd)
        self.fast.fail_if = lambda r: bool(np.any(r[:, 0] > 5e9))
        self.fast.fail_with = _FakeXlaRuntimeError(
            "INTERNAL: CpuCallback error calling callback: Traceback (most recent call last):\n"
            "  File \"equinox/_errors.py\", line 88, in raises\n"
            "_EquinoxRuntimeError: The maximum number of steps was reached in the "
            "nonlinear solver. The problem may not be solveable."
        )
        try:
            self.move._stats = self.move._new_stats()
            with self.assertLogs("lisatools.globalfit.moves.mbhbatchedmove", level="WARNING") as cm:
                out = self.move.compute_like(bad, idx)
        finally:
            self.fast.fail_if = None
            self.fast.fail_with = None
        keep = [0, 1, 2, 4]
        self.assertEqual(out[3], -1e300)                    # the bad row: rejected
        np.testing.assert_allclose(out[keep], baseline[keep], rtol=0, atol=1e-9)
        self.assertTrue(np.isnan(self.move._last_d_h[3]))
        self.assertTrue(np.all(np.isfinite(self.move._last_d_h[keep])))
        self.assertEqual(self.move.n_generate_failures, 1)
        self.assertEqual(self.move.n_batch_fallbacks, 0)    # not a refusal
        self.assertEqual(self.move._stats["gen_failed"], 1)
        self.assertIsInstance(self.move.last_generate_error, _FakeXlaRuntimeError)
        text = "\n".join(cm.output)
        self.assertIn("waveform generation FAILED at runtime", text)
        self.assertIn("7.70000000e+09", text)               # the row's parameters are logged
        self.assertIn("maximum number of steps", text)      # the model's reason, not the wrapper's
        with self.assertLogs("lisatools.globalfit.moves.mbhbatchedmove", level="INFO") as cm2:
            self.move._flush_stats()
        self.assertIn("gen_failed=1", "\n".join(cm2.output))
        # the state is clean afterwards: the same rows score as before
        np.testing.assert_allclose(self.move.compute_like(rows, idx), baseline, rtol=0, atol=1e-9)

    def test_generator_failure_log_is_capped_per_leaf(self):
        from lisatools.globalfit.moves import mbhbatchedmove as mod

        n = mod._GEN_FAIL_LOG_CAP + 3
        rows = self._rows(n)
        rows[:, 0] = 7.7e9                                  # every row is bad
        idx = np.zeros(n, dtype=int)
        self.fast.fail_if = lambda r: bool(np.any(r[:, 0] > 5e9))
        self.fast.fail_with = _FakeXlaRuntimeError("INTERNAL: CpuCallback error: boom")
        try:
            with self.assertLogs("lisatools.globalfit.moves.mbhbatchedmove", level="WARNING") as cm:
                out = self.move.compute_like(rows, idx)
        finally:
            self.fast.fail_if = None
            self.fast.fail_with = None
        self.assertTrue(np.all(out == -1e300))
        self.assertEqual(self.move.n_generate_failures, n)
        with_params = [m for m in cm.output if "waveform-basis params" in m]
        self.assertEqual(len(with_params), mod._GEN_FAIL_LOG_CAP)
        self.assertTrue(any("counted only" in m for m in cm.output))

    def test_plain_runtime_error_in_the_generator_still_raises(self):
        """The isolation is for JAX / equinox RUNTIME failures only: a Python
        bug in the generator must not be silently scored at the floor."""
        rows = self._rows(2)
        self.fast.fail_with = RuntimeError("a lisatools bug, not a solver")
        try:
            with self.assertRaises(RuntimeError):
                self.move.compute_like(rows, np.array([0, 1]))
        finally:
            self.fast.fail_with = None
        self.assertEqual(self.move.n_generate_failures, 0)

    def test_jax_runtime_failure_classification(self):
        from lisatools.globalfit.moves.mbhbatchedmove import _is_jax_runtime_failure

        self.assertTrue(_is_jax_runtime_failure(_FakeXlaRuntimeError("x")))
        self.assertTrue(_is_jax_runtime_failure(_FakeEquinoxRuntimeError("x")))
        self.assertTrue(_is_jax_runtime_failure(_FakeJaxRuntimeError("x")))
        self.assertFalse(_is_jax_runtime_failure(RuntimeError("x")))
        self.assertFalse(_is_jax_runtime_failure(_FakeJaxValueError("x")))   # not a RuntimeError
        self.assertFalse(_is_jax_runtime_failure(TypeError("x")))
        try:
            import jax
            real = jax.errors.JaxRuntimeError("INTERNAL: CpuCallback error calling callback")
        except Exception as exc:  # noqa: BLE001 -- jax absent or the class not constructible
            self.skipTest(f"real jax.errors.JaxRuntimeError unavailable: {exc!r}")
        self.assertTrue(_is_jax_runtime_failure(real))


class _FakeXlaRuntimeError(RuntimeError):
    """Stands in for ``jaxlib.xla_extension.XlaRuntimeError`` (== ``jax.errors.JaxRuntimeError``)."""
    __module__ = "jaxlib.xla_extension"


class _FakeJaxRuntimeError(RuntimeError):
    __module__ = "jax._src.errors"


class _FakeEquinoxRuntimeError(RuntimeError):
    __module__ = "equinox._errors"


class _FakeJaxValueError(ValueError):
    __module__ = "jax._src.errors"


class MBHBatchedSnapParityTest(unittest.TestCase):
    """Stock epoch OFF the data lattice by 0.3 dt: the batched path (snapped
    generator + adapter-carried snap) must match the container path (stock
    epoch), and flipping the snap's sign must break it (pins the sign)."""

    def test_nonzero_snap_parity_and_sign(self):
        snap = 0.3 * DT
        acs, adapter, fast, wdm = _build(t0_stock=T0 - snap)
        move = _build_move(acs, adapter)
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        move.setup_likelihood_here(cold)
        rows = np.tile(BASE_ROW, (4, 1))
        rows[:, 4] *= np.array([0.8, 1.0, 1.1, 1.25])
        rows[:, 10] += np.array([-0.2, 0.1, 0.25, -0.05]) * LAYER
        idx = np.array([0, 1, 2, 1])
        fast_ll = move.compute_like(rows, idx)
        slow_ll = move.compute_acs_like(rows, idx)
        d = np.abs(fast_ll - slow_ll).max()
        print(f"[mbh batched snap parity] snap={snap} s: max |fast-slow| = {d:.3e}")
        np.testing.assert_allclose(fast_ll, slow_ll, rtol=0, atol=5e-2)
        adapter.t_plunge_snap = -snap          # negative control: wrong sign
        flipped = move.compute_like(rows, idx)
        d_flip = np.abs(flipped - slow_ll).max()
        print(f"[mbh batched snap parity] sign flipped: max |fast-slow| = {d_flip:.3e}")
        self.assertGreater(d_flip, 1.0)


class MBHBatchedAbsoluteEpochTest(unittest.TestCase):
    """The stock erebor t0 layout (mojito mode, 2026-09-30 defect).

    ``WDMSettings.make_factory`` ignores ``times[0]``, so the run's WDM
    settings -- and the containers' data, poured with them -- carry t0 = 0,
    while the data start ``data_t0`` and every merger time are ABSOLUTE
    (~9.7e7 s). Building a GB comp later sets ``domain_settings.t0 =
    data_t0`` IN PLACE, so the adapter's (device-local) settings carry 0 or
    ``data_t0`` depending on build order. The window geometry must come from
    the adapter's explicit ``t0_abs`` (= data_t0), and the template labels
    from the CONTAINERS' own settings; the numbers must equal a run on a
    plain relative (t0 = 0, data start 0) grid."""

    ROWS_IDX = np.array([0, 1, 2, 1, 0, 1])

    def _run(self, dense_fill=False, **build_kw):
        import os
        from unittest import mock

        acs, adapter, fast, wdm = _build(**build_kw)
        move = _build_move(acs, adapter)
        before = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        cold = _cold_rows()
        env = {"MBH_BATCHED_FILL": "0" if dense_fill else "1"}
        with mock.patch.dict(os.environ, env):
            move.remove_cold_chain_sources(cold)      # expose
            exposed = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
            move.setup_likelihood_here(cold)
            rows = _proposal_rows(len(self.ROWS_IDX))
            fast_ll = move.compute_like(rows, self.ROWS_IDX)
            slow_ll = move.compute_acs_like(rows, self.ROWS_IDX)
            out = dict(
                fast=fast_ll, slow=slow_ll, geom=dict(move._leaf_windows[0]),
                seg=dict(adapter.geometry), stats=dict(move._stats),
                fallbacks=move.n_batch_fallbacks, before=before, exposed=exposed,
            )
            move.add_back_in_cold_chain_sources(cold)  # fold-back
        out["after"] = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        return out

    def _reference(self, dense_fill=False):
        """Same toy on a plain relative grid: data start 0, every t0 = 0."""
        return self._run(dense_fill=dense_fill)

    def _assert_matches_reference(self, got, ref, atol):
        d = max(np.abs(got["fast"] - ref["fast"]).max(), np.abs(got["slow"] - ref["slow"]).max())
        print(f"[mbh batched abs epoch] max |lnL(absolute epoch) - lnL(relative grid)| = {d:.3e}")
        np.testing.assert_allclose(got["fast"], ref["fast"], rtol=0, atol=atol)
        np.testing.assert_allclose(got["slow"], ref["slow"], rtol=0, atol=atol)
        for g, r in zip(got["exposed"], ref["exposed"]):
            np.testing.assert_allclose(g, r, rtol=0, atol=1e-6 * np.abs(r).max())

    def _assert_round_trip(self, got):
        for b, e, a in zip(got["before"], got["exposed"], got["after"]):
            self.assertGreater(np.abs(e - b).max(), 1e-3 * np.abs(b).max())   # expose added h
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.abs(b).max())

    def test_settings_t0_zero_places_the_window_by_t0_abs(self):
        """(a) containers AND adapter settings t0 = 0, absolute times,
        adapter t0_abs = data start: the window lands on the merger's layers
        (not clamped to the grid end), the segment is placed at the ABSOLUTE
        time of its first layer, batched == container path, the telemetry
        sees no row outside the box, and expose/fold restores the residual."""
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        ref = self._reference()
        got = self._run(data_t0=EPOCH, container_t0=0.0, t0_abs=EPOCH)
        t_med = float(np.median(_cold_rows()[:, 10]))
        want = mbh_window_layers(_toy_wdm(0.0), t_med, **WINDOW_WIDE)
        self.assertEqual(got["geom"]["n_start"], want["n_start"])
        self.assertLess(got["geom"]["n_start"] + got["geom"]["Nt_keep"], NT)   # not clamped
        self.assertEqual(got["geom"]["n_start"], ref["geom"]["n_start"])
        self.assertEqual(got["seg"]["t_seg"], EPOCH + got["seg"]["s0"] * LAYER)
        d = np.abs(got["fast"] - got["slow"]).max()
        print(f"[mbh batched abs epoch, settings t0=0] max |fast-slow| = {d:.3e}")
        np.testing.assert_allclose(got["fast"], got["slow"], rtol=0, atol=5e-2)
        self.assertEqual(got["fallbacks"], 0)
        self.assertEqual((got["stats"]["outside_box"], got["stats"]["outside_data"]), (0, 0))
        # float rounding of the toy's absolute times at ~1e8 s only (measured
        # value printed above)
        self._assert_matches_reference(got, ref, atol=1e-3)
        self._assert_round_trip(got)

    def test_adapter_settings_t0_zero_containers_at_data_start_scoring(self):
        """(b) mutation order reversed: containers carry t0 = data start, the
        adapter's settings t0 = 0. Scoring (dense fill, so only the scorer's
        re-label is exercised) must not raise and must give the same numbers."""
        ref = self._reference(dense_fill=True)
        got = self._run(dense_fill=True, data_t0=EPOCH, container_t0=EPOCH, adapter_t0=0.0, t0_abs=EPOCH)
        np.testing.assert_allclose(got["fast"], got["slow"], rtol=0, atol=5e-2)
        self.assertEqual(got["fallbacks"], 0)
        self._assert_matches_reference(got, ref, atol=1e-3)

    def test_adapter_settings_t0_zero_containers_at_data_start_fill(self):
        """(b) the same layout through the batched expose/fold: the fill's
        re-label must not raise, must add exactly the reference template and
        must restore the residual."""
        ref = self._reference()
        got = self._run(data_t0=EPOCH, container_t0=EPOCH, adapter_t0=0.0, t0_abs=EPOCH)
        self._assert_matches_reference(got, ref, atol=1e-3)
        self._assert_round_trip(got)

    def test_relabel_forgives_only_t0(self):
        """The re-label borrows the containers' t0 and nothing else: a
        template on a different wavelet grid still raises."""
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        other = WDMSettings(NF, NT + 2, DT, t0=T0, min_freq=2e-3, max_freq=2e-2, force_backend="cpu")
        box = other.get_slice((slice(0, int(other.Nf_active)), slice(10, 20)))
        tmpl = WDMSignal(np.zeros((3, int(box.Nf_active), 10)), box)
        with self.assertRaisesRegex(ValueError, "share the wavelet grid"):
            move._on_container_box(tmpl)


class MBHBatchedFillTest(unittest.TestCase):
    def test_expose_then_fold_restores_residual(self):
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        before = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        # scale-free "changed" test: np.allclose's default atol=1e-8 calls any
        # strain-scale (1e-22) edit "unchanged"
        changed = [np.abs(ac.data.arr - b).max() > 1e-3 * np.abs(b).max() for ac, b in zip(acs.acs.flatten(), before)]
        self.assertTrue(all(changed))
        move.add_back_in_cold_chain_sources(cold)
        for ac, b in zip(acs.acs.flatten(), before):
            np.testing.assert_allclose(ac.data.arr, b, rtol=0, atol=1e-12 * np.abs(b).max())
        self.assertEqual(fast.n_calls, 4)   # 3 walkers at batch 2 -> 2 chunks per pass

    def test_window_hysteresis(self):
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        first = dict(move._leaf_windows[0])
        moved = cold.copy()
        moved[:, 10] += 0.5 * LAYER              # inside the 1-layer margin
        move.add_back_in_cold_chain_sources(moved)
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"])
        far = cold.copy()
        far[:, 10] += 5 * LAYER                  # outside the margin
        move.add_back_in_cold_chain_sources(far)  # fold never rebuilds
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"])
        move.remove_cold_chain_sources(far)       # expose rebuilds
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"] + 5)

    def test_dense_fill_knob(self):
        import os

        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        os.environ["MBH_BATCHED_FILL"] = "0"
        try:
            move.remove_cold_chain_sources(_cold_rows())
        finally:
            del os.environ["MBH_BATCHED_FILL"]
        self.assertEqual(fast.n_calls, 0)


class MBHBatchedDevicePinningTest(unittest.TestCase):
    """Multi-GPU: phentax runs in JAX, which cupy's device context does not
    move. Every shard's rows must be generated (scoring AND fill) under BOTH
    the cupy device context and the JAX default-device context of THAT
    shard's device."""

    def test_each_shard_generates_under_its_cupy_and_jax_device(self):
        import contextlib
        from unittest import mock

        import lisatools.globalfit.moves.mbhbatchedmove as mod
        from lisatools.analysiscontainer import AnalysisContainerArray

        try:
            from tests._multishard import RecordingXp
        except ImportError:  # pragma: no cover - run from tests/
            from _multishard import RecordingXp

        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter, batch_max_size=2)
        rx = RecordingXp()
        jax_stack = [None]

        @contextlib.contextmanager
        def fake_jax(device):
            jax_stack.append(device)
            try:
                yield
            finally:
                jax_stack.pop()

        owner = {0: 1, 1: 2, 2: 1}     # walker -> device (walkers 0, 2 on gpu 1)

        def fake_split(idx):
            idx = np.asarray(idx).reshape(-1)
            out = []
            for dev in (1, 2):
                pos = np.where([owner[int(w)] == dev for w in idx])[0]
                if pos.size:
                    out.append((dev, pos))
            return out

        calls = []
        real = fast.compute_tdi_channels

        def recording(*cols, **kw):
            # column 1 (m2) is unused by the toy model: it tags the walker
            calls.append((rx.current_device, jax_stack[-1], [int(round(v)) for v in np.atleast_1d(cols[1])]))
            return real(*cols, **kw)

        fast.compute_tdi_channels = recording
        cold = _cold_rows()
        cold[:, 1] = np.arange(NWALKERS)
        rows = np.tile(BASE_ROW, (6, 1))
        idx = np.array([0, 1, 2, 1, 0, 2])
        rows[:, 1] = idx
        with mock.patch.object(AnalysisContainerArray, "xp", new=property(lambda self: rx)), \
                mock.patch.object(mod, "jax_device_context", fake_jax), \
                mock.patch.object(move, "_split_rows", fake_split):
            move.remove_cold_chain_sources(cold)          # fill path
            move.setup_likelihood_here(cold)
            n_fill = len(calls)
            move.compute_like(rows, idx)                  # scoring path
        self.assertGreater(n_fill, 0)
        self.assertGreater(len(calls), n_fill)
        seen = set()
        for cuda_dev, jax_dev, walkers in calls:
            expect = {owner[w] for w in walkers}
            self.assertEqual(len(expect), 1, calls)
            self.assertEqual(cuda_dev, expect.pop(), calls)
            self.assertEqual(jax_dev, cuda_dev, calls)
            seen.add(cuda_dev)
        self.assertEqual(seen, {1, 2})


class MBHBatchedRoutingTest(unittest.TestCase):
    def test_split_rows_by_shard(self):
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Holder:
            linear_data_arr = [object(), object()]
            gpus = [0, 1]
            gpu_splits = [np.array([0, 2]), np.array([1])]
            split_map = np.array([0, 1, 0])
            acs_total_entries = 3

        groups = MBHBatchedLikeMove._split_rows_static(_Holder(), np.array([2, 1, 0, 1]))
        self.assertEqual([g[0] for g in groups], [0, 1])
        np.testing.assert_array_equal(groups[0][1], [0, 2])
        np.testing.assert_array_equal(groups[1][1], [1, 3])

    def test_epoch_and_snap_read_through_the_adapter(self):
        """MBHWindowedWDMSignalGen does not forward waveform_t0/t_plunge_snap;
        the move must read them off the wrapped generator, never default the
        snap to 0 (that would shift every template by the snap)."""
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Inner:
            waveform_t0 = 1000.0
            t_plunge_snap = 7.0

        class _Adapter:
            wave_gen = _Inner()

            def __init__(self):
                self.seen = None

            def __call__(self, *cols, **kw):
                self.seen = np.stack(cols, axis=1)
                return "tmpl"

        move = MBHBatchedLikeMove.__new__(MBHBatchedLikeMove)
        move.batched_gen = _Adapter()
        move._gen_kwargs = {}
        self.assertEqual(move._stock_waveform_t0(), 993.0)
        rows = np.tile(BASE_ROW, (2, 1))
        move._generate(move.batched_gen, rows)
        np.testing.assert_array_equal(move.batched_gen.seen[:, 10], rows[:, 10] - 7.0)

    def test_adapter_carried_epoch_and_snap(self):
        """Task 5 layout: plain attributes on the adapter; the inner generator
        is built on the snapped epoch and carries NO snap."""
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Inner:
            waveform_t0 = 1000.0

        class _Adapter:
            wave_gen = _Inner()

            def __call__(self, *cols, **kw):
                self.seen = np.stack(cols, axis=1)
                return "tmpl"

        adapter = _Adapter()
        adapter.waveform_t0 = 1000.0
        adapter.t_plunge_snap = 7.0
        move = MBHBatchedLikeMove.__new__(MBHBatchedLikeMove)
        move.batched_gen = adapter
        move._gen_kwargs = {}
        self.assertEqual(move._stock_waveform_t0(), 993.0)
        rows = np.tile(BASE_ROW, (2, 1))
        move._generate(adapter, rows)
        np.testing.assert_array_equal(adapter.seen[:, 10], rows[:, 10] - 7.0)

    def test_missing_snap_raises(self):
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Inner:
            waveform_t0 = 1000.0

        class _Adapter:
            wave_gen = _Inner()

            def __call__(self, *cols, **kw):
                return "tmpl"

        move = MBHBatchedLikeMove.__new__(MBHBatchedLikeMove)
        move.batched_gen = _Adapter()
        move._gen_kwargs = {}
        with self.assertRaisesRegex(AttributeError, "t_plunge_snap"):
            move._stock_waveform_t0()
        with self.assertRaisesRegex(AttributeError, "t_plunge_snap"):
            move._generate(move.batched_gen, np.tile(BASE_ROW, (1, 1)))

    def test_layer0_time_is_read_from_the_adapter_never_defaulted(self):
        """``t0_abs`` is REQUIRED from the adapter (cached once): an adapter
        without it raises rather than falling back to the containers'
        settings t0 (0 in a stock erebor build)."""
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Adapter:
            wave_gen = object()

        move = MBHBatchedLikeMove.__new__(MBHBatchedLikeMove)
        move.batched_gen = _Adapter()
        with self.assertRaisesRegex(AttributeError, "t0_abs"):
            move._grid_t0_abs()
        move.batched_gen.t0_abs = 9.7e7
        self.assertEqual(move._grid_t0_abs(), 9.7e7)
        move.batched_gen.t0_abs = 1.0
        self.assertEqual(move._grid_t0_abs(), 9.7e7)   # a run constant: cached

    def test_ctor_guards(self):
        from unittest import mock

        import lisatools.globalfit.moves.mbhbatchedmove as mod

        acs, adapter, fast, wdm = _build()
        with self.assertRaises(ValueError):
            _build_move(acs, None)
        # a window longer than the data BUILDS (clamped; 2026-09-30) ...
        move = _build_move(acs, adapter, window_before=200 * LAYER)
        self.assertEqual(move.window_before, 200 * LAYER)
        # ... and the geometry is still validated at BUILD, not at the first
        # leaf: an impossible geometry (mbh_window_layers raises) fails here
        with mock.patch.object(mod, "mbh_window_layers", side_effect=ValueError("no geometry")):
            with self.assertRaisesRegex(ValueError, "no geometry"):
                _build_move(acs, adapter)
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, uniform_dist
        from lisatools.globalfit.moves import MBHBatchedLikeMove

        priors = {"mbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
        with self.assertRaises(ValueError):
            MBHBatchedLikeMove("mbh", (NTEMPS, NWALKERS, 1, 11), None, {}, {}, acs, 1, None, priors,
                               [(StretchMove(), 1.0)], batched_gen=adapter, dcga=object(), **WINDOW)


def _gpu_backend_name():
    """The concrete GPU backend (``"cuda12x"`` / ``"cuda13x"``), or None without one."""
    import lisatools

    try:
        return lisatools.get_backend("gpu").name.split("_")[-1]
    except Exception:
        return None


class MBHBatchedGPUParityTest(unittest.TestCase):
    """GPU build of the batched scorer == its CPU build (skips without a GPU backend).

    The same toy on each backend -- three walkers, walker 1 on another PSD model, the
    same noise draws, the stub generator's output on that backend, the container array
    on ``gpus=[0]`` as the benchmark builds it -- through the full choreography:
    batched expose (fill), the per-walker offsets (``acs.likelihood()``),
    ``compute_like`` (the batched scorer) with its <d|h> / <h|h>, the container path,
    and the fold-back.

    Tolerances: the arithmetic is the same float64 sequence; only reduction order
    (cupy's vs numpy's sums over ~7e3 pixels), cuFFT vs pocketfft and the PSD model's
    transcendental ulps differ, each ~1e-16..1e-15 relative to the magnitudes summed.
    So logL / offsets agree to 1e-9 of max(1, |logL|) (the offsets carry the
    noise-normalisation term, |logL| ~ 1e5-1e6 here: the bound is ~1e-4..1e-3 nats,
    five orders under the 0.5-nat check tolerance and far under the O(1-100) nats
    between rows), <d|h> / <h|h> and the residual arrays to 1e-9 / 1e-11 of their
    maxima. A defect this targets -- a template scored against the wrong walker's PSD
    or residual slice, a mis-labelled box, a lost row -- moves logL by O(1) or more
    (MBHBatchedParityTest pins the walker-0 vs walker-1 PSD difference). Cluster GPU,
    2026-10-02: max |logL gpu - cpu| = 2.3e-10 on |logL| ~ 2e5 (1e-15 relative), so the logL
    bound is 1e-12 relative (~2e-7 nats here, ~900x margin)."""

    LL_RTOL, IP_RTOL, ARR_RTOL = 1e-12, 1e-9, 1e-11

    def _run(self, backend):
        acs, adapter, fast, wdm = _build(backend=backend)
        move = _build_move(acs, adapter)
        host = lambda a: np.real(np.asarray(asnumpy(a)))  # noqa: E731
        before = [host(ac.data.arr).copy() for ac in acs.acs.flatten()]
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)                 # batched fill (expose)
        exposed = [host(ac.data.arr).copy() for ac in acs.acs.flatten()]
        move.setup_likelihood_here(cold)
        rows = _proposal_rows(6)
        idx = np.array([0, 1, 2, 1, 0, 1])
        out = dict(
            fast=host(move.compute_like(rows, idx)).reshape(-1),
            d_h=host(move._last_d_h).copy(), h_h=host(move._last_h_h).copy(),
            offset=host(move._exposed_offset).copy(),
        )
        out.update(
            slow=host(move.compute_acs_like(rows, idx)).reshape(-1),
            fallbacks=int(move.n_batch_fallbacks), before=before, exposed=exposed,
        )
        move.add_back_in_cold_chain_sources(cold)            # fold-back
        out["after"] = [host(ac.data.arr).copy() for ac in acs.acs.flatten()]
        return out

    def test_gpu_equals_cpu(self):
        gpu = _gpu_backend_name()
        if gpu is None:
            self.skipTest("no GPU backend")
        c = self._run("cpu")
        g = self._run(gpu)
        tol = self.LL_RTOL * max(1.0, float(np.abs(c["fast"]).max()))
        d = float(np.abs(g["fast"] - c["fast"]).max())
        print(f"[mbh batched GPU vs CPU] max |logL gpu - cpu| = {d:.3e} on |logL| ~ "
              f"{np.abs(c['fast']).max():.3e} (bound {tol:.1e}); rows span {np.ptp(c['fast']):.3e}")
        self.assertGreater(np.ptp(c["fast"]), 1e3 * tol)     # the rows are distinguishable
        np.testing.assert_allclose(g["fast"], c["fast"], rtol=0, atol=tol)
        np.testing.assert_allclose(g["offset"], c["offset"], rtol=0, atol=tol)
        for key in ("d_h", "h_h"):
            np.testing.assert_allclose(g[key], c[key], rtol=0,
                                       atol=self.IP_RTOL * float(np.abs(c[key]).max()), err_msg=key)
        for key in ("exposed", "after"):
            for w, (a, b) in enumerate(zip(g[key], c[key])):
                np.testing.assert_allclose(a, b, rtol=0, atol=self.ARR_RTOL * float(np.abs(b).max()),
                                           err_msg=f"{key} walker {w}")
        # on the GPU itself: batched == container path (the CPU tests' windowing bound)
        np.testing.assert_allclose(g["fast"], g["slow"], rtol=0, atol=5e-2)
        self.assertEqual(g["fallbacks"], 0)
        for a, b in zip(g["after"], g["before"]):           # fold-back restores the residual
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.abs(b).max())


if __name__ == "__main__":
    unittest.main()
