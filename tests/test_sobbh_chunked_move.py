"""SOBBHChunkedLikeMove: fast-vs-slow parity through the move's own seams.

CPU-only, small WDM geometry. The slow side is the REAL stock path
(``get_sobbh_tdionfly_gen`` + ``SOBBHTDIonFlyWaveWrap`` + the container
likelihood); the fast side is the move's ``compute_like`` (chunked-heterodyne
``SOBBHWDMComputations.get_ll_wdm`` + the per-walker exposed-residual
offset). Run at a NONZERO reference epoch (t0 = t_ref = 0.5 yr): the slow
python PN evaluates at ``times - reference_time`` while the chunked C++
evaluates at ``t - t_ref`` only since the bbhx t_ref fix — so this parity
also pins that fix (and the phi0 -> phi_c passthrough).
"""

from __future__ import annotations

import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI

# small WDM toy: Tobs = Nf*Nt*dt ~ 7.6 days, slow-chirp source stays in one
# layer, dense per-call cost small enough for a laptop unittest
DT = 10.0
NF, NT = 256, 256
NOBS = NF * NT
T_START = int(0.5 * YRSID_SI / DT) * DT  # nonzero epoch (the point!)
F_LOW = 6.0e-3
NWALKERS = 3
NTEMPS = 2

#: stock SOBBH waveform basis (what compute_like receives):
#: (m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)
REF_STOCK = np.array(
    [60.0, 55.0, 0.1, 0.2, 0.4, np.arccos(0.3), F_LOW, 3.1,
     np.arcsin(0.2), 0.7, 1.1]
)


def _build_toy():
    """(acs, wrap_adapter, comp, wdm, td_set) on the shared toy geometry."""
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import TDSettings, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.response import get_sobbh_tdionfly_gen
    from lisatools.globalfit.stock.erebor.wrappers import SOBBHTDIonFlyWaveWrap

    from bbhx.sobbhcomps import SOBBHWDMComputations

    backend = "cpu"
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    td_set = TDSettings(NOBS, DT, force_backend=backend)
    # band floor at 2 mHz: keeps the scirdv1 XYZ2 inverse covariance finite
    # on every active layer, so the container noise term is well-defined
    wdm = WDMSettings(
        NF, NT, DT, t0=T_START, min_freq=2e-3, max_freq=1.2e-2,
        is_complex=False, force_backend=backend,
    )

    gen = get_sobbh_tdionfly_gen(
        Tobs=NOBS * DT,
        dt=DT,
        t_start=T_START,
        tdi_config=tdi_config,
        reference_time=float(T_START),
        orbits=orbits,
        n_grid=1024,
        buffer_time=5000.0,
        force_backend=backend,
    )
    t_arr = np.arange(NOBS) * DT + T_START
    wrap = SOBBHTDIonFlyWaveWrap(gen, t_arr, td_set, wdm, td_window=None, nchannels=3)

    def sobbh_gen(*params, apply_transform=False, leaf_inds=None, **kwargs):
        # engine-convention adapter: waveform-basis params, transform done
        return wrap(*params)

    inj = wrap(*REF_STOCK)
    acs_list = []
    for _ in range(NWALKERS):
        ac = AnalysisContainer(
            WDMSignal(np.array(np.asarray(inj.arr), copy=True), wdm),
            XYZ2SensitivityMatrix(wdm, model="scirdv1"),
        )
        ac.signal_gen = {"sobbh": sobbh_gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list)

    comp = SOBBHWDMComputations(
        wdm, t_ref=float(T_START), Nt_sub=64, n_pad=8, N_sparse=64,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend,
        d_d=0.0, tdi_type="XYZ",
    )
    return acs, sobbh_gen, comp, wdm, td_set


def _build_move(acs, comp, m_band_half_width=2):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import SOBBHChunkedLikeMove

    betas = 1 / 1.2 ** np.arange(NTEMPS)
    betas_all = np.tile(betas, (1, 1))
    priors = {
        "sobbh": ProbDistContainer(
            {i: uniform_dist(-1e10, 1e10) for i in range(11)}
        )
    }
    move = SOBBHChunkedLikeMove(
        "sobbh",
        (NTEMPS, NWALKERS, 1, 11),
        None,  # waveform_gen: containers carry the installed generator
        {},
        {},
        acs,
        1,
        None,
        priors,
        [(StretchMove(), 1.0)],
        betas_all=betas_all,
        chunked_comp=comp,
        m_band_half_width=m_band_half_width,
        name="sobbh chunked test",
    )
    return move


class SOBBHChunkedShimTest(unittest.TestCase):
    def test_shim_mapping(self):
        from lisatools.globalfit.moves import SOBBHChunkedLikeMove

        row = np.arange(11, dtype=float) + 1.0
        out = SOBBHChunkedLikeMove.to_chunked_basis(row)
        self.assertEqual(out.shape, (1, 11))
        # (m1, m2, s1, s2, dist*1e9, f_low, phi_c, inc, psi, lam, beta)
        expected = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0e9, 7.0, 11.0, 6.0, 10.0, 8.0, 9.0]
        )
        np.testing.assert_allclose(out[0], expected)
        # 2D passthrough + input untouched
        rows = np.tile(row, (3, 1))
        out2 = SOBBHChunkedLikeMove.to_chunked_basis(rows)
        np.testing.assert_allclose(out2, np.tile(expected, (3, 1)))
        np.testing.assert_allclose(rows[0], row)


class SOBBHChunkedParityTest(unittest.TestCase):
    """Fast-vs-slow through the move seams on the shared toy."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.acs, cls.gen, cls.comp, cls.wdm, cls.td_set = _build_toy()
        except Exception as exc:  # missing compiled deps etc.
            raise unittest.SkipTest(f"toy setup unavailable: {exc}")
        cls.move = _build_move(cls.acs, cls.comp)

    def _rows(self, n_pert=8, seed=5):
        rng = np.random.default_rng(seed)
        rows = np.tile(REF_STOCK, (n_pert + 1, 1))
        layer_df = float(self.wdm.layer_df)
        rows[1:, 6] += rng.uniform(-0.2, 0.2, n_pert) * layer_df  # f_low
        rows[1:, 10] = rng.uniform(0, 2 * np.pi, n_pert)  # phi0
        rows[1:, 5] = np.arccos(rng.uniform(-0.9, 0.9, n_pert))  # inc
        rows[1:, 4] *= rng.uniform(0.7, 1.5, n_pert)  # dist
        idx = np.arange(rows.shape[0], dtype=np.int32) % NWALKERS
        return rows, idx

    def test_fast_vs_slow_parity_through_move(self):
        move = self.move
        # arm the per-leaf offset exactly as propose() would post-expose
        move.setup_likelihood_here(None)
        rows, idx = self._rows()

        fast = move.compute_like(rows, idx)
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)

        self.assertTrue(np.all(np.isfinite(fast)))
        diff = np.abs(fast - slow)
        self.assertLess(
            float(diff.max()), move.check_ll_tol,
            msg=f"fast vs slow lnL diff {diff.max():.3e} exceeds tol "
                f"{move.check_ll_tol} (median {np.median(diff):.3e})",
        )
        # the built-in cross-check must agree with itself (no warning raise)
        move.check_ll_mode = "strict"
        move._verify_prev_logl(fast.reshape(1, -1), rows, idx, leaf=0)

    def test_expose_invariant_via_offset(self):
        # with the data holding exactly the injection, scoring the injection
        # against it must reproduce acs.likelihood() at the injection-free
        # point: fast(inj) = offset + (d_h - h_h/2) ~ lnL of (d - h) residual
        move = self.move
        move.setup_likelihood_here(None)
        fast = move.compute_like(
            np.tile(REF_STOCK, (NWALKERS, 1)),
            np.arange(NWALKERS, dtype=np.int32),
        )
        slow = move.compute_acs_like(
            np.tile(REF_STOCK, (NWALKERS, 1)),
            np.arange(NWALKERS, dtype=np.int32),
            **move.waveform_like_kwargs,
        )
        np.testing.assert_allclose(fast, slow, atol=move.check_ll_tol)

    def test_out_of_band_sentinel(self):
        move = self.move
        move.setup_likelihood_here(None)
        bad = REF_STOCK.copy()
        bad[6] = 0.5  # f_low far above the WDM band
        out = move.compute_like(bad.reshape(1, -1), np.zeros(1, dtype=np.int32))
        self.assertEqual(float(out[0]), -1e300)
        nanrow = REF_STOCK.copy()
        nanrow[0] = np.nan
        out2 = move.compute_like(nanrow.reshape(1, -1), np.zeros(1, dtype=np.int32))
        self.assertEqual(float(out2[0]), -1e300)

    def test_record_leaf_inner_products_from_kernel(self):
        from types import SimpleNamespace

        move = self.move
        move.setup_likelihood_here(None)
        sub = SimpleNamespace(
            d_h=np.full((NWALKERS, 1), np.nan),
            h_h=np.full((NWALKERS, 1), np.nan),
        )
        state = SimpleNamespace(sub_states={"sobbh": sub})
        move._record_leaf_inner_products(
            state, np.tile(REF_STOCK, (NWALKERS, 1)), 0
        )
        self.assertTrue(np.all(np.isfinite(sub.d_h)))
        self.assertTrue(np.all(np.isfinite(sub.h_h)))
        self.assertTrue(np.all(sub.h_h > 0))

    def test_compute_like_requires_armed_offset(self):
        move = _build_move(self.acs, self.comp)
        with self.assertRaises(RuntimeError):
            move.compute_like(
                REF_STOCK.reshape(1, -1), np.zeros(1, dtype=np.int32)
            )

    def test_ctor_guards(self):
        from lisatools.globalfit.moves import SOBBHChunkedLikeMove

        with self.assertRaises(ValueError):
            _ = SOBBHChunkedLikeMove("sobbh", (1, 1, 1, 11), None, {}, {},
                                     self.acs, 1, None, {}, [],
                                     chunked_comp=None)

    def test_ladder_length_matches_ntemps_knob(self):
        # P1.0 pin: SOBBHSetup sizes its default ladder from the ntemps knob
        from lisatools.globalfit.stock.erebor.sobbh import (
            SOBBHSettings, SOBBHSetup)

        s = SOBBHSettings(Tobs=31536000.0, dt=10.0)
        setup = SOBBHSetup(s)
        self.assertEqual(len(setup.betas), s.ntemps)

    def test_chunked_is_the_stock_default(self):
        # A/B-gated default flip (2026-07-30); SOBBH_LIKELIHOOD=full is the
        # escape hatch
        import os

        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceSOBBHSettings)

        assert "SOBBH_LIKELIHOOD" not in os.environ
        self.assertEqual(SourceSOBBHSettings().likelihood, "chunked")


class SOBBHChunkedRoutingTest(unittest.TestCase):
    """Multi-shard (multi-GPU walker-shard) routing in compute_like.

    Uses the shared FakeMultiShardACA + a stub comp: verifies rows are
    partitioned by owning split, scored against THAT split's buffer with
    INTRA-shard indices under the owning device context, and reassembled
    in global row order (d_h/h_h included)."""

    def test_multi_shard_routing(self):
        try:
            from tests._multishard import FakeMultiShardACA
        except ImportError:
            from _multishard import FakeMultiShardACA
        from types import SimpleNamespace

        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, uniform_dist

        from lisatools.globalfit.moves import SOBBHChunkedLikeMove

        nwalkers, ntemps = 6, 2
        acs = FakeMultiShardACA((3, 8), nwalkers, 2, layout="blocked",
                                dtype=float)

        class _StubComp:
            d_d = 0.0
            wdm_settings = SimpleNamespace(
                ind_min_f=0, ind_max_f=999, layer_df=1e-3
            )

            def __init__(self):
                self.calls = []

            def get_ll_wdm(self, params, holder, data_index=None,
                           noise_index=None, m_band_half_width=1):
                fp = float(np.real(np.asarray(holder.linear_data_arr[0])[0]))
                intra = np.asarray(data_index, dtype=int)
                self.calls.append((fp, intra.copy()))
                self.d_h_out = fp * 10.0 + intra.astype(float)
                self.h_h_out = 2.0 * self.d_h_out
                return self.d_h_out - 0.5 * self.h_h_out + fp

        comp = _StubComp()
        betas = 1 / 1.2 ** np.arange(ntemps)
        move = SOBBHChunkedLikeMove(
            "sobbh", (ntemps, nwalkers, 1, 11), None, {}, {}, acs, 1, None,
            {"sobbh": ProbDistContainer(
                {i: uniform_dist(-1e10, 1e10) for i in range(11)})},
            [(StretchMove(), 1.0)],
            betas_all=np.tile(betas, (1, 1)),
            chunked_comp=comp,
            name="sobbh routing test",
        )
        move._exposed_offset = np.zeros(nwalkers)

        rows = np.tile(REF_STOCK, (nwalkers, 1))
        idx = np.arange(nwalkers, dtype=np.int32)
        out = move.compute_like(rows, idx)

        # blocked layout: walkers 0-2 -> shard 0 (buffer fingerprint 1.0,
        # first owned row 0), walkers 3-5 -> shard 1 (fingerprint 4.0)
        self.assertEqual(len(comp.calls), 2)
        fps = sorted(c[0] for c in comp.calls)
        self.assertEqual(fps, [1.0, 4.0])
        for fp, intra in comp.calls:
            np.testing.assert_array_equal(intra, np.arange(3))
        expected = np.array([
            (1.0 if w < 3 else 4.0) for w in range(nwalkers)
        ])
        np.testing.assert_allclose(out, expected)
        # d_h/h_h reassembled in global row order with intra offsets
        np.testing.assert_allclose(
            move._last_d_h,
            [10.0, 11.0, 12.0, 40.0, 41.0, 42.0],
        )
        # each split ran under its owning device context
        self.assertEqual(sorted(set(acs.xp.device_log)), [0, 1])


if __name__ == "__main__":
    unittest.main()
