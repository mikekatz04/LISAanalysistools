"""MBH proposals in the LISA frame (``MBH_PROPOSAL_FRAME``, user ruling 2026-10-19).

Mike, 2026-10-19: "keep the same sampling/waveform parameters on the MBHB,
but when we do the sampling moves, can we move in the LISA reference frame
... Only the move that produces the new coordinates should be moved in the
LISA frame." The add/remove move conjugates its inner moves with the frame
map ``F`` (``LISAFrameMap``: ICRS ``alpha / sin_delta / psi / t_plunge`` ->
``lam_L / sinbeta_L / psi_L / t_L``): ``y = F(x)``, the inner move proposes
``y'``, ``x' = F^-1(y')``, and the Metropolis factor gains ``ln|J_F(x)| -
ln|J_F(x')|``.

Pinned here (all CPU):

* the frame map: the refined inverse is exact (the legacy SkyMove inverse
  is not), the central-difference Jacobian against the analytic
  ``1 - (R/c) alpha'(t) cos(zeta) cos(beta_L) sin(lam_L - alpha)`` and a
  coarser stencil, the domain, the box widths, pickling (``FrameMapTest``);
* SkyMove bit-identical after the refactor (``GoldenStreamTest``, golden
  written at 34cd57e7) and a sky hop proposes the same ``x'`` in or out of
  the conjugation (``SkyHopFrameTest``);
* detailed balance of the conjugated proposal on a curved toy with a
  STRONG Jacobian (``|J| = e^u``): the chain's moments match the target, and
  the paired negative controls -- no Jacobian, the wrong sign -- miss it
  (``DetailedBalanceTest``);
* the information matrix in the frame by ``y``-perturbations equals
  ``J^-T F_x J^-1`` of the sampling-basis matrix where the latter is exact
  (``FrameInfoMatrixTest``);
* the adaptation, the cross-walker partners and the sidecar work in the
  frame (``AdaptationAndJumpInFrameTest``, ``SidecarFrameGuardTest``);
* the REAL ``MBHBatchedLikeMove`` with the frame on, single process (eigen,
  and the full stock-like stack with the adaptation / tempering /
  controller), and the 4-rank walker-block fan-out equals single process
  (``RealMoveFrameTest``, ``WalkerBlockFrameTest``);
* knob-off bit-identity of the MBH stream (``GoldenStreamTest``, written by
  :func:`write_golden` against the code BEFORE the proposal frame existed,
  34cd57e7);
* the knob, the refusal for bases without frame columns, the stock wiring
  (``KnobTest``);
* the curved-posterior harness: 4 walkers x 8 rungs, IAT along ``t`` and the
  sky, frame off vs on (``CurvedPosteriorHarnessTest``; numbers printed with
  ``PROPOSAL_FRAME_HARNESS_VERBOSE=1``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    from tests import test_mbh_batched_fanout as H
except ImportError:  # pragma: no cover - run from tests/
    import test_mbh_batched_fanout as H

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GOLDEN = os.path.join(DATA, "mbh_proposal_frame_golden.npz")
SKY_GOLDEN = os.path.join(DATA, "skymove_frame_golden.npz")
N_GOLDEN = 3

#: the stock MBH sampling basis' frame columns (source_runtime.MBH_SKY_MOVE_IND_MAP)
IND_MAP = dict(cosinc=6, psi=7, lam=8, sinbeta=9, t_ref=10)
#: the stock MBH periodic columns: phi_ref 2 pi, psi pi, alpha 2 pi
PERIODIC = {"mbh": {5: 2 * np.pi, 7: np.pi, 8: 2 * np.pi}}

#: every MBH knob the add/remove move reads at use time
_MBH_KNOBS = tuple(f"MBH_{k}" for k in (
    "EIGEN_AXIS_ADAPT", "EIGEN_ADAPT_VISITS", "EIGEN_ADAPT_MIN_VISITS", "EIGEN_ADAPT_POOL",
    "EIGEN_TEMPER_SIGMAS", "EIGEN_ADAPT_MAX_FACTOR", "EIGEN_AXIS_ACC_LOG", "EIGEN_AXIS_ACC_TUNE",
    "EIGEN_AXIS_ACC_MIN", "EIGEN_AXIS_ACC_MAX", "EIGEN_AXIS_ACC_FACTOR", "EIGEN_INFO",
    "EIGEN_GRAM_TARGET", "EIGEN_GRAM_EPS_REL", "COLD_TO_HOT_EVERY", "LADDER_REANCHOR_NATS",
    "LADDER_TOP_SNR", "PROPOSAL_FRAME", "DE_JUMP_MIX", "DE_JUMP_MODE_FRAC"))

#: (single process?, nwalkers, env, stack, move kwargs)
GOLDEN_SCENARIOS = {
    "single_eigen": (True, 4, {}, "eigen",
                     dict(scope="walker_max", permute_every=1, adaptive=True)),
    "single_stack": (True, 4, {"MBH_EIGEN_AXIS_ADAPT": "1", "MBH_EIGEN_ADAPT_MIN_VISITS": "1",
                               "MBH_EIGEN_AXIS_ACC_TUNE": "1", "MBH_EIGEN_TEMPER_SIGMAS": "1"},
                     "stack", dict(scope="walker_max", permute_every=1, adaptive=True)),
    "block4_per_walker": (False, 4, {}, "eigen",
                          dict(scope="per_walker", permute_every=0, adaptive=False)),
}


def _env(extra=None, **kw):
    env = {k: v for k, v in os.environ.items() if k not in H._CLEARED_ENV and k not in _MBH_KNOBS}
    env.update(MBH_CHECK_LL_EVERY="1", MBH_CHECK_LL="1", EIGEN_TABLES_PERSIST="0")
    env.update(extra or {})
    env.update(kw)
    return mock.patch.dict(os.environ, env, clear=True)


def _sky_phase():
    """A NON-linear constellation phase table (equation-of-centre-like wobble)
    so the per-row pinning is exercised."""
    from bbhx.utils.constants import ConstOmega

    from lisatools.sampling.moves.skymodehop import ConstellationPhase

    t = np.linspace(-5e5, 4e7, 600)
    return ConstellationPhase(t, ConstOmega * t + 0.7 + 0.02 * np.sin(ConstOmega * t + 0.3))


def _stack(kind):
    """``"eigen"``: the plain eryn move. ``"stack"``: eigen + cross-walker jump +
    the three ICRS sky hops (the stock MBH composition, toy weights)."""
    from eryn.moves import EigenAxisMove

    from lisatools.sampling.moves.skymodehop import SkyMove
    from lisatools.sampling.moves.walkerdiffjump import WalkerDifferenceJumpMove

    eig = EigenAxisMove(mode="axis")
    if kind == "eigen":
        return [(eig, 1.0)]
    phase = _sky_phase()
    return [
        (eig, 0.6),
        (WalkerDifferenceJumpMove(eig, mode_frac=0.5), 0.1),
        (SkyMove(ind_map=dict(IND_MAP), which="both", coord_frame="icrs",
                 constellation_phase=phase), 0.1),
        (SkyMove(ind_map=dict(IND_MAP), which="lat", coord_frame="icrs",
                 constellation_phase=phase), 0.1),
        (SkyMove(ind_map=dict(IND_MAP), which="long", coord_frame="icrs",
                 constellation_phase=phase), 0.1),
    ]


def make_move(acs, nwalkers, *, stack="eigen", scope, permute_every, adaptive, refresh=1,
              prior_bounds=None, **extra):
    """test_mbh_batched_fanout's real ``MBHBatchedLikeMove`` with a chosen stack,
    the stock periodic columns and (optionally) other prior bounds."""
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.utils import PeriodicContainer

    from lisatools.globalfit.moves import MBHBatchedLikeMove
    from lisatools.sources.batching import MBHWindowedWDMSignalGen

    wdm = acs.acs.flatten()[0].data.settings
    adapter = MBHWindowedWDMSignalGen(H._FastGen(0.0), wdm, nchannels=3, tukey_alpha=0.0)
    bounds = H.PRIOR_BOUNDS if prior_bounds is None else prior_bounds
    priors = {"mbh": ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(bounds)})}
    move = MBHBatchedLikeMove(
        "mbh", (H.NTEMPS, nwalkers, H.NLEAVES, H.NDIM), None, {}, {}, acs, H.NUM_REPEATS,
        H._Identity(), priors, _stack(stack) if isinstance(stack, str) else stack,
        betas_all=np.tile(H.BETAS, (H.NLEAVES, 1)), permute_every=permute_every,
        batched_gen=adapter, batch_max_size=H.BATCH, name=H.MOVE_NAME,
        eigen_refresh_every=refresh, eigen_table_scope=scope, **H.WINDOW_WIDE, **extra,
    )
    move.periodic = PeriodicContainer(PERIODIC)
    move.gf_move_name = H.MOVE_NAME
    move.fanout_branches = ["mbh"]
    for tc in move.temperature_controls:
        tc.adaptive = bool(adaptive)
    H._install_spies(move)
    return move


def _record_inner(move, qs):
    """Wrap every inner move's get_proposal to record ``(q, factors)``."""
    for m in {id(m): m for m in move.moves}.values():
        orig = m.get_proposal

        def rec(*a, _orig=orig, **k):
            q, f = _orig(*a, **k)
            qs.append((np.array(q["mbh"], dtype=float, copy=True),
                       np.array(f, dtype=float, copy=True)))
            return q, f

        m.get_proposal = rec


def golden_stream(name, n_proposes=N_GOLDEN):
    """One scenario's proposal stream + per-propose end states, as arrays."""
    from lisatools.globalfit.engine import GlobalFitInfo

    single, nw, env, stack, kw = GOLDEN_SCENARIOS[name]
    out = {}
    with _env(env):
        H._wdm()
        for w in range(4):
            H._residual(w)
        if single:
            walkers = list(range(nw))
            acs = H._aca(walkers)
            move = make_move(acs, nw, stack=stack, **kw)
            qs = []
            _record_inner(move, qs)
            model = GlobalFitInfo(acs, map, np.random.RandomState(H.SEED))
            np.random.seed(H.SEED)
            states, accepted = H._run_proposes(move, model, H._state(walkers), n_proposes)
            out["q"] = np.stack([q for q, _ in qs])
            out["factors"] = np.stack([f for _, f in qs])
            for leaf, (axes, sig) in sorted((getattr(move, "_eigen_tables", None) or {}).items()):
                out[f"axes_{leaf}"] = np.asarray(axes, dtype=float)
                out[f"sigmas_{leaf}"] = np.asarray(sig, dtype=float)
        else:
            res = _world(4, nw, stack=stack, n_proposes=n_proposes, **kw)
            states, accepted = res[0]["states"], res[0]["accepted"]
        for k, s in enumerate(states):
            sub = s.sub_states["mbh"]
            out[f"coords_{k}"] = np.asarray(sub.coords, dtype=float)
            out[f"log_like_{k}"] = np.asarray(sub.log_like, dtype=float)
            out[f"in_model_accepted_{k}"] = np.asarray(sub.in_model_accepted)
            out[f"swaps_accepted_{k}"] = np.asarray(sub.swaps_accepted)
            out[f"betas_all_{k}"] = np.asarray(sub.betas_all, dtype=float)
            out[f"accepted_{k}"] = np.asarray(accepted[k])
    return out


def _world(n_compute, nwalkers, *, stack="eigen", n_proposes=N_GOLDEN, move_hook=None, **kw):
    """The 4-rank walker-block fan-out (FakeWorld) of the real move, row-local draws."""
    from lisatools.globalfit.communication.fakecomm import FakeWorld
    from lisatools.globalfit.communication.fanout import (
        LIKELIHOOD_OP, ComputeService, WalkerFanout)
    from lisatools.globalfit.communication.ranks import RankRole, build_layout
    from lisatools.globalfit.engine import GlobalFitInfo
    from lisatools.globalfit.moves import MBHBatchedLikeMove
    from lisatools.globalfit.recipe import _body_nwalkers
    from lisatools.utils.utility import asnumpy

    world = FakeWorld(n_compute + 1, timeout=600.0)

    def fn(rank, comm):
        layout = build_layout(comm, nwalkers, [], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return dict(role="saver", layout=layout)
        w0, w1 = layout.block_of(rank)
        acs = H._aca(range(w0, w1))
        width = _body_nwalkers(SimpleNamespace(rank_layout=layout), acs, MBHBatchedLikeMove)
        move = make_move(acs, width, stack=stack, **kw)
        if move_hook is not None:
            move_hook(move)
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move.install_walker_fanout(SimpleNamespace(fanout=fo, rank=rank))
        model = GlobalFitInfo(acs, map, H._RowLocalRandom())
        fcomm.barrier()
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage(H.STAGE, "pe")
            try:
                states, accepted = H._run_proposes(move, model, H._state(range(nwalkers)), n_proposes)
            finally:
                fo.stop()
            return dict(role="head", layout=layout, states=states, accepted=accepted, move=move)
        ComputeService(
            fcomm, layout, rank, registry={(H.STAGE, H.MOVE_NAME): move}, model=model,
            builtins={LIKELIHOOD_OP: lambda payload, clock, m: np.asarray(
                asnumpy(m.analysis_container_arr.likelihood(complex=False)))},
        ).serve()
        return dict(role="compute", layout=layout, move=move)

    with H._row_local_numpy():
        return world.run(fn)


def sky_golden():
    """SkyMove's frame maps and proposals (ICRS + SSB ecliptic, phase table +
    constant offset), as arrays."""
    from lisatools.sampling.moves.skymodehop import SkyMove

    rng = np.random.default_rng(77)
    n = 48
    x = rng.uniform(0, 1, (n, 11))
    x[:, 6] = rng.uniform(-0.95, 0.95, n)
    x[:, 7] = rng.uniform(0, np.pi, n)
    x[:, 8] = rng.uniform(0, 2 * np.pi, n)
    x[:, 9] = rng.uniform(-0.98, 0.98, n)
    x[:, 10] = rng.uniform(1e5, 3e7, n)
    out = {"x": x}
    phase = _sky_phase()
    for frame in ("icrs", "ssb_ecliptic"):
        for tag, kw in (("table", dict(constellation_phase=phase)), ("t0", dict(frame_t0=0.37))):
            for which in ("both", "lat", "long"):
                mv = SkyMove(ind_map=dict(IND_MAP), which=which, coord_frame=frame, **kw)
                key = f"{frame}_{tag}_{which}"
                y = mv._to_lisa_frame(x)
                out[f"{key}/fwd"] = y
                out[f"{key}/inv"] = mv._from_lisa_frame(y)
                out[f"{key}/ldj"] = mv._log_abs_det_jacobian(x)
                out[f"{key}/ldj_inv"] = mv._log_abs_det_jacobian(y, fn=mv._from_lisa_frame)
                q, f = mv.get_proposal({"mbh": x.reshape(2, 24, 1, 11)},
                                       np.random.RandomState(5))
                out[f"{key}/q"] = q["mbh"]
                out[f"{key}/factors"] = f
    return out


def write_golden():
    """Regenerate BOTH golden files (run against the PRE-change code only)."""
    flat = {}
    for name in GOLDEN_SCENARIOS:
        for key, arr in golden_stream(name).items():
            flat[f"{name}/{key}"] = arr
    os.makedirs(DATA, exist_ok=True)
    np.savez(GOLDEN, **flat)
    np.savez(SKY_GOLDEN, **sky_golden())
    return GOLDEN, SKY_GOLDEN


class GoldenStreamTest(unittest.TestCase):
    """Knob off: the MBH proposal stream is bit-identical to the pre-frame code."""

    def test_the_knob_off_stream_is_bit_identical_to_head(self):
        ref = np.load(GOLDEN)
        names = sorted({k.split("/", 1)[0] for k in ref.files})
        self.assertEqual(names, sorted(GOLDEN_SCENARIOS))
        for name in names:
            got = golden_stream(name)
            want = {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(name + "/")}
            with self.subTest(scenario=name):
                self.assertEqual(sorted(got), sorted(want))
                for key in want:
                    np.testing.assert_array_equal(got[key], want[key], err_msg=f"{name}/{key}")

    def test_skymove_is_bit_identical_after_the_frame_map_refactor(self):
        ref = np.load(SKY_GOLDEN)
        got = sky_golden()
        self.assertEqual(sorted(got), sorted(ref.files))
        for key in ref.files:
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)



# ---------------------------------------------------------------------------
# shared helpers (post-golden: these use the proposal-frame API)
# ---------------------------------------------------------------------------

LOGGER = "lisatools.globalfit.moves.addremovemove"
#: the stock MBH prior box of the toy's frame columns: sin_delta in (-1, 1)
FRAME_BOUNDS = [b if i != 9 else (-1.0 + 1e-6, 1.0 - 1e-6) for i, b in enumerate(H.PRIOR_BOUNDS)]
_PHASE = {}


def _esa_phase():
    """The constellation phase of the packaged ESA trailing orbits (a
    realistic, non-linear ``alpha(t)``), cached."""
    if "esa" not in _PHASE:
        import warnings

        from lisatools.detector import ESAOrbits
        from lisatools.sampling.moves.skymodehop import ConstellationPhase
        from lisatools.utils.constants import YRSID_SI

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orb = ESAOrbits(force_backend="cpu", linear_interp_setup=False,
                            t_arr=np.arange(0.0, 2.0 * YRSID_SI, 3600.0))
        _PHASE["esa"] = ConstellationPhase.from_orbits(orb)
    return _PHASE["esa"]


def _fmap(phase="toy", **kw):
    from lisatools.sampling.moves.skymodehop import LISAFrameMap

    cp = {"toy": _sky_phase, "esa": _esa_phase}[phase]() if phase else None
    return LISAFrameMap(dict(IND_MAP), coord_frame="icrs", constellation_phase=cp, **kw)


def _mbh_rows(n, rng, t_range=(1e6, 3e7)):
    x = np.zeros((n, 11))
    x[:, :7] = rng.normal(size=(n, 7))
    x[:, 7] = rng.uniform(0, np.pi, n)
    x[:, 8] = rng.uniform(0, 2 * np.pi, n)
    x[:, 9] = rng.uniform(-0.98, 0.98, n)
    x[:, 10] = rng.uniform(*t_range, n)
    return x


def _wrap_cols(d):
    d = np.array(d, dtype=float, copy=True)
    d[..., 8] = (d[..., 8] + np.pi) % (2 * np.pi) - np.pi
    d[..., 7] = (d[..., 7] + np.pi / 2) % np.pi - np.pi / 2
    return d


# ---- the curved toy posterior in the stock MBH basis (the harness target) ----
T_MID = 2.0e7
MU_Y = np.array([1.0, 0.3, 1.2, T_MID])            # lam_L, sinbeta_L, psi_L, t_L
SIG_Y = np.array([0.15, 0.1, 0.1, 0.5])
CURVED_BOX = [(-10, 10)] * 7 + [(0, np.pi), (0, 2 * np.pi), (-1 + 1e-6, 1 - 1e-6),
                                (T_MID - 2e4, T_MID + 2e4)]
CURVED_FMAP = None


def _curved_fmap():
    global CURVED_FMAP
    if CURVED_FMAP is None:
        from lisatools.sampling.moves.skymodehop import LISAFrameMap

        CURVED_FMAP = LISAFrameMap(dict(IND_MAP), coord_frame="icrs", frame_t0=0.13)
    return CURVED_FMAP


def curved_lnl(x):
    """A Gaussian in the LISA frame: in the ICRS basis t_plunge is a curved
    function of the sky (+-75 s across the sky width, ~5 s of curvature =
    10 sigma_tL)."""
    fm = _curved_fmap()
    x = np.asarray(x, dtype=float)
    sh = x.shape[:-1]
    x2 = x.reshape(-1, 11)
    y = fm.forward(x2)
    d = y[:, fm.columns] - MU_Y
    d[:, 0] = (d[:, 0] + np.pi) % (2 * np.pi) - np.pi
    d[:, 2] = (d[:, 2] + np.pi / 2) % np.pi - np.pi / 2
    return (-0.5 * np.sum((d / SIG_Y) ** 2, axis=1)
            - 0.5 * np.sum(x2[:, :7] ** 2, axis=1)).reshape(sh)


def curved_target(rng, n):
    """Exact draws of the curved target: y ~ N(MU_Y, SIG_Y), x = F^-1(y), with
    importance weights 1 / |J_F(x)| (1 +- 5e-5 here)."""
    fm = _curved_fmap()
    y = np.zeros((n, 11))
    y[:, :7] = rng.normal(size=(n, 7))
    y[:, fm.columns] = MU_Y + SIG_Y * rng.normal(size=(n, 4))
    y[:, 8] %= 2 * np.pi
    y[:, 7] %= np.pi
    x = fm.inverse(y)
    return x, np.exp(-fm.log_abs_det_jacobian(x))


def stub_move(frame, *, ntemps=8, nwalkers=4, lnl=curved_lnl, box=CURVED_BOX, fmap=None,
              moves=None, adapt=False, temper=False, tune=False, refresh=5, eps_rel=1e-4,
              store=None):
    """The add/remove move without data (``__new__``) on the stock MBH basis:
    the REAL table refresh (the ll route at the max-lnL cold walker against
    ``lnl``), adaptation, tempering, controller and proposal frame."""
    from eryn.moves import EigenAxisMove
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.utils import PeriodicContainer

    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    m = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    m.branch_name = "mbh"
    m.ndim, m.ntemps, m.nwalkers = 11, ntemps, nwalkers
    m.moves = moves if moves is not None else [EigenAxisMove(mode="axis")]
    m.priors = {"mbh": ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(box)})}
    m.eigen_refresh_every = refresh
    m.eigen_eps_rel = eps_rel
    m.eigen_table_scope = None
    m.eigen_info = "ll"
    m.eigen_axis_adapt = adapt
    m.eigen_adapt_visits = 50
    m.eigen_adapt_min_visits = 10
    m.eigen_adapt_max_factor = None
    m.eigen_temper_sigmas = temper
    m.eigen_axis_acc_tune = tune
    m.eigen_store_path = store
    m._to_phys = lambda x: x
    m.compute_like = lambda X, data_index=None: lnl(X)
    m._periodic = PeriodicContainer(PERIODIC)
    for mv in m.moves:
        mv.periodic = m._periodic
    m.proposal_frame = frame
    m.proposal_frame_map = _curved_fmap() if fmap is None else fmap
    return m


def mh_sweep(move, work, ll, lp, betas, rng, repeats, trace=None):
    """``repeats`` proposals of ``move.moves[0]`` the way ``propose_local``
    makes them (the per-split install, the frame gate, the conjugated
    proposal, the outside-row prior, the tempered accept), then adjacent-rung
    swaps top down."""
    name = move.branch_name
    nt, nw, _, nd = work.coords.shape
    inner = move.moves[0]
    pri = move.priors[name]
    S1 = np.ones((nt, nw), dtype=bool)
    nacc = 0
    for _ in range(repeats):
        sets = [work.coords[:nt][S1][:, 0].reshape(nt, -1, 1, nd)]
        move._install_eigen_split_table(inner, 0, S1)
        fm = move._frame_for_move(inner)
        outside = None
        if fm is None:
            q, f = inner.get_proposal({name: sets[0]}, rng)
            new = q[name].reshape(nt, nw, nd)
        else:
            new, f, _, _, outside = move._frame_proposal(fm, inner, sets, 0, rng)
        lpn = pri.logpdf(new.reshape(-1, nd))
        if outside is not None and np.any(outside):
            lpn = np.where(outside, -np.inf, lpn)
        lpn = lpn.reshape(nt, nw)
        lln = np.full((nt, nw), -np.inf)
        fin = np.isfinite(lpn)
        if np.any(fin):
            lln[fin] = move.compute_like(new[fin])
        keep = (np.asarray(f) + betas[:, None] * (lln - ll) + (lpn - lp)
                > np.log(rng.random((nt, nw))))
        work.coords[keep, 0] = new[keep]
        ll = np.where(keep, lln, ll)
        lp = np.where(keep, lpn, lp)
        nacc += int(keep[0].sum())
        for k in range(nt - 1, 0, -1):
            sw = np.log(rng.random(nw)) < (betas[k - 1] - betas[k]) * (ll[k] - ll[k - 1])
            for w in np.flatnonzero(sw):
                work.coords[[k - 1, k], w] = work.coords[[k, k - 1], w]
                ll[[k - 1, k], w] = ll[[k, k - 1], w]
                lp[[k - 1, k], w] = lp[[k, k - 1], w]
        if trace is not None:
            trace.append(np.array(work.coords[0, :, 0, :], copy=True))
    return ll, lp, nacc


def tau(x):
    """Integrated autocorrelation time (FFT acf, Sokal window c = 5)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    win = np.arange(n) < 5.0 * taus
    m = int(np.argmin(win)) if np.any(~win) else n - 1
    return float(taus[m])


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _MBH_KNOBS
           and not k.startswith(("EMRI_PROPOSAL", "SOBBH_PROPOSAL"))}
    env["EIGEN_TABLES_PERSIST"] = "0"
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


# ---------------------------------------------------------------------------
# the frame map
# ---------------------------------------------------------------------------

class FrameMapTest(unittest.TestCase):
    def test_the_refined_inverse_is_exact_where_the_legacy_one_is_not(self):
        fm = _fmap("esa")
        x = _mbh_rows(400, np.random.default_rng(1))
        y = fm.forward(x)
        legacy = _wrap_cols(fm.inverse_approx(y) - x)
        refined = _wrap_cols(fm.inverse(y) - x)
        # the legacy inverse (SkyMove's) misses by ~1e-3 s / ~5e-6 rad on the
        # ESA phase: above 1e-3 of the narrowest MBH widths (0.17 s, 3e-3 rad)
        self.assertGreater(np.abs(legacy[:, 10]).max(), 1e-4)
        self.assertGreater(np.abs(legacy[:, [7, 8, 9]]).max(), 1e-6)
        self.assertLess(np.abs(refined[:, [7, 8, 9]]).max(), 1e-12)
        self.assertLess(np.abs(refined[:, 10]).max(), 1e-8)
        np.testing.assert_array_equal(refined[:, :7], 0.0)       # untouched columns
        # and F(F^-1(y)) = y
        np.testing.assert_allclose(_wrap_cols(fm.forward(fm.inverse(y)) - y), 0.0, atol=1e-8)

    def test_the_jacobian_is_one_plus_O_omega_R_over_c_and_matches_the_analytic_form(self):
        from bbhx.utils.constants import C_SI, ConstOmega
        from bbhx.utils.transform import OrbitR

        from lisatools.sampling.moves.skymodehop import (FRAME_FD_STEPS, LISAFrameMap,
                                                         fd_log_abs_det_jacobian)

        class Phase:      # a smooth phase with a known derivative
            def __call__(self, t):
                return ConstOmega * np.asarray(t) + 0.4 + 0.01 * np.sin(ConstOmega * np.asarray(t))

            def deriv(self, t):
                return ConstOmega * (1 + 0.01 * np.cos(ConstOmega * np.asarray(t)))

        ph = Phase()
        fm = LISAFrameMap(dict(IND_MAP), coord_frame="icrs", constellation_phase=ph)
        x = _mbh_rows(200, np.random.default_rng(3))
        ldj = fm.log_abs_det_jacobian(x)
        self.assertLess(np.abs(ldj).max(), 1e-4)
        self.assertGreater(np.abs(ldj).max(), 1e-5)
        # ln|det J| = ln|1 - (R/c) alpha'(t) cos(zeta) cos(beta_L) sin(lam_L - alpha)|
        # (the sky rotation is area preserving in (lam, sin beta), the
        # polarization block is unit triangular; only dt_L/dt at fixed LISA
        # sky is off 1)
        y = fm.forward(x)
        an = np.log(np.abs(1 - OrbitR / C_SI * ph.deriv(x[:, 10]) * np.cos(np.pi / 3)
                           * np.cos(np.arcsin(y[:, 9])) * np.sin(y[:, 8] - ph(x[:, 10]))))
        self.assertLess(np.abs(ldj - an).max(), 2e-8)
        coarse = fd_log_abs_det_jacobian(x, fm.forward, fm.columns,
                                         tuple(3 * s for s in FRAME_FD_STEPS))
        self.assertLess(np.abs(ldj - coarse).max(), 2e-8)
        # the full (ndim, ndim) Jacobian embeds the block, identity elsewhere
        J = fm.jacobian(x[:5])
        np.testing.assert_array_equal(J[:, :7, :7], np.broadcast_to(np.eye(7), (5, 7, 7)))
        np.testing.assert_array_equal(J[:, :7, 7:], 0.0)
        np.testing.assert_allclose(np.log(np.abs(np.linalg.det(J))), ldj[:5], atol=1e-12)

    def test_the_domain_the_box_and_pickling(self):
        import pickle

        fm = _fmap("toy")
        y = fm.forward(_mbh_rows(6, np.random.default_rng(4)))
        y[1, 9] = 1.02
        y[2, 9] = np.nan
        x, ok = fm.inverse(y, return_ok=True)
        np.testing.assert_array_equal(ok, [True, False, False, True, True, True])
        w = fm.proposal_widths(np.arange(1.0, 12.0))
        np.testing.assert_allclose(w[[8, 9, 7]], [2 * np.pi, 2.0, np.pi])
        np.testing.assert_allclose(w[[0, 6, 10]], [1.0, 7.0, 11.0])
        fm2 = pickle.loads(pickle.dumps(fm))
        z = _mbh_rows(4, np.random.default_rng(5))
        np.testing.assert_array_equal(fm2.forward(z), fm.forward(z))
        self.assertEqual(fm.name, "lisa")
        self.assertEqual(fm.time_column, 10)

    def test_skymove_converts_with_the_frame_map(self):
        from lisatools.sampling.moves.skymodehop import LISAFrameMap, SkyMove

        ph = _sky_phase()
        mv = SkyMove(ind_map=dict(IND_MAP), which="lat", coord_frame="icrs", constellation_phase=ph)
        self.assertIsInstance(mv.frame_map, LISAFrameMap)
        self.assertTrue(mv.conjugates_own_frame)
        self.assertFalse(SkyMove(which="lat").conjugates_own_frame)
        x = _mbh_rows(8, np.random.default_rng(6))
        np.testing.assert_array_equal(mv._to_lisa_frame(x), _fmap("toy").forward(x))
        np.testing.assert_array_equal(mv._from_lisa_frame(x), _fmap("toy").inverse_approx(x))


# ---------------------------------------------------------------------------
# detailed balance of the conjugated proposal (a strong-Jacobian toy frame)
# ---------------------------------------------------------------------------

class _ToyCurvedFrame:
    """y = (u, (t - A cos u) e^{k u}): curved in x, |J| = e^{k u} (strong)."""

    name = "lisa"

    def __init__(self, A=3.0, k=1.0):
        self.A, self.k = float(A), float(k)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        y = x.copy()
        y[:, 1] = (x[:, 1] - self.A * np.cos(x[:, 0])) * np.exp(self.k * x[:, 0])
        return y

    def inverse(self, y, refine=True, return_ok=False):
        y = np.asarray(y, dtype=float)
        x = y.copy()
        x[:, 1] = y[:, 1] * np.exp(-self.k * y[:, 0]) + self.A * np.cos(y[:, 0])
        ok = np.all(np.isfinite(y), axis=1)
        return (x, ok) if return_ok else x

    def log_abs_det_jacobian(self, x):
        return self.k * np.asarray(x, dtype=float)[:, 0]


SIG_U, SIG_T, TOY_A = 0.5, 0.05, 3.0


def _toy_lnl(x):
    x = np.asarray(x, dtype=float)
    return -0.5 * (x[..., 0] / SIG_U) ** 2 - 0.5 * ((x[..., 1] - TOY_A * np.cos(x[..., 0])) / SIG_T) ** 2


class DetailedBalanceTest(unittest.TestCase):
    """The conjugated eigen move samples pi(x) exactly; without the Jacobian
    factor, or with its sign flipped, it does not (paired negative controls).

    Target: u ~ N(0, 0.5), t | u ~ N(3 cos u, 0.05) -- a thin curved ridge in
    x. The frame ``y = (u, (t - 3 cos u) e^u)`` straightens it with ``|J| =
    e^u``: without the factor the chain samples ``pi(x) e^u`` (E[u] = +0.25),
    with the wrong sign ``pi(x) e^{2u}`` (E[u] = +0.5).
    """

    N_REPEATS = 15000

    def _run(self, patch=None, seed=11):
        from eryn.moves import EigenAxisMove
        from eryn.prior import ProbDistContainer, uniform_dist

        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

        m = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
        m.branch_name = "mbh"
        m.moves = [EigenAxisMove(mode="axis")]
        m.priors = {"mbh": ProbDistContainer({0: uniform_dist(-4, 4), 1: uniform_dist(-20, 20)})}
        m.compute_like = lambda X, data_index=None: _toy_lnl(X)
        m.proposal_frame = "lisa"
        m.proposal_frame_map = _ToyCurvedFrame(TOY_A, 1.0)
        m.moves[0].set_axes("mbh", np.eye(2), np.array([0.6, 0.12]))
        rng = np.random.default_rng(seed)
        nw = 4
        u = rng.normal(size=nw) * SIG_U
        x0 = np.stack([u, TOY_A * np.cos(u) + SIG_T * rng.normal(size=nw)], axis=1)
        work = SimpleNamespace(coords=x0[None, :, None, :].copy())
        ll = _toy_lnl(work.coords[:, :, 0, :])
        lp = np.zeros((1, nw))
        trace = []
        with (patch if patch is not None else contextlib.nullcontext()):
            mh_sweep(m, work, ll, lp, np.ones(1), rng, self.N_REPEATS, trace)
        return np.array(trace).reshape(-1, 2)

    def _expected(self):
        e_u, e_cos = 0.0, np.exp(-SIG_U ** 2 / 2)
        var_cos = 0.5 * (1 + np.exp(-2 * SIG_U ** 2)) - np.exp(-SIG_U ** 2)
        return e_u, TOY_A * e_cos, SIG_U, np.sqrt(SIG_T ** 2 + TOY_A ** 2 * var_cos)

    def test_the_conjugated_chain_samples_the_target(self):
        tr = self._run()
        e_u, e_t, s_u, s_t = self._expected()
        self.assertLess(abs(tr[:, 0].mean() - e_u), 0.04, tr[:, 0].mean())
        self.assertLess(abs(tr[:, 1].mean() - e_t), 0.04, tr[:, 1].mean())
        self.assertLess(abs(tr[:, 0].std() / s_u - 1), 0.06)
        self.assertLess(abs(tr[:, 1].std() / s_t - 1), 0.08)

    def test_the_paired_negative_controls_miss_the_target(self):
        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

        e_u = self._expected()[0]
        no_jac = mock.patch.object(ResidualAddOneRemoveOneMove, "_frame_log_det",
                                   lambda self, fmap, rows: np.zeros(len(rows)))
        tr = self._run(no_jac)
        self.assertGreater(tr[:, 0].mean() - e_u, 0.15, tr[:, 0].mean())     # ~ +0.25
        flipped = mock.patch.object(
            ResidualAddOneRemoveOneMove, "_frame_log_det",
            lambda self, fmap, rows: -2.0 * np.asarray(fmap.log_abs_det_jacobian(rows)))
        tr = self._run(flipped)
        self.assertGreater(tr[:, 0].mean() - e_u, 0.35, tr[:, 0].mean())     # ~ +0.5


class OffTheSphereTest(unittest.TestCase):
    def test_a_proposal_off_the_sphere_is_rejected_outright(self):
        """A y' with |sinbeta_L| > 1 has no preimage: factor -inf, x' = x."""
        fm = _curved_fmap()

        class Push:
            periodic = None

            def get_proposal(self, s, random):
                q = {k: np.array(v, copy=True) for k, v in s.items()}
                q["mbh"][0, :2, 0, 9] = 1.5                        # rows 0, 1 off the sphere
                q["mbh"][..., 10] += 0.1
                return q, np.zeros(q["mbh"].shape[:2])

        m = stub_move("lisa", ntemps=2, nwalkers=4)
        x, _ = curved_target(np.random.default_rng(2), 8)
        S = x.reshape(2, 4, 1, 11)
        xn, f, y_old, y_new, out = m._frame_proposal(fm, Push(), [S], 0, None)
        np.testing.assert_array_equal(out, [True, True] + [False] * 6)
        self.assertTrue(np.all(np.isneginf(f[0, :2])))
        self.assertTrue(np.all(np.isfinite(f.ravel()[2:])))
        np.testing.assert_array_equal(xn[0, :2], S[0, :2, 0])
        self.assertTrue(np.all(xn.reshape(-1, 11)[2:, 10] != S.reshape(-1, 11)[2:, 10]))
        np.testing.assert_allclose(y_old.reshape(-1, 11), fm.forward(S.reshape(-1, 11)))


# ---------------------------------------------------------------------------
# the information matrix in the frame
# ---------------------------------------------------------------------------

def _corr_norm_diff(a, b):
    d = np.sqrt(np.outer(np.diag(a), np.diag(a)))
    d = np.where(d > 0, d, 1.0)
    return float(np.max(np.abs(a - b) / d))


class FrameInfoMatrixTest(unittest.TestCase):
    def test_the_y_space_gram_equals_the_transformed_sampling_gram(self):
        """The REAL MBH move's Gram route on the toy: by perturbations in the
        frame (production) vs ``J^-T F_x J^-1`` (exact here: the toy's
        templates do not depend on the ICRS sky, so F_x has exact zeros there
        and the transform involves no cancellation)."""
        from lisatools.globalfit.moves import eigen_refresh

        with _env():
            H._wdm()
            acs = H._aca([0, 1, 2, 3])
            fm = _fmap("toy")
            move = make_move(acs, 4, scope="walker_max", permute_every=0, adaptive=False,
                             prior_bounds=FRAME_BOUNDS, proposal_frame="lisa",
                             proposal_frame_map=fm)
            move._current_leaf = 0
            x0 = H._cold(1, 0)
            widths = eigen_refresh.prior_box_widths(move.priors["mbh"], 11)
            move.setup_likelihood_here(H._tempered_coords([0, 1, 2, 3])[0, :, 0])
            f_x, s_x = move._gram_info(x0, 1, widths, return_steps=True)
            f_y, s_y = move._gram_info(x0, 1, widths, return_steps=True, frame=fm)
        lo, hi = eigen_refresh.prior_box_bounds(move.priors["mbh"], 11)
        xc = eigen_refresh.nudge_inside(x0, s_x, lo, hi)
        ref = eigen_refresh.transform_information(f_x, fm.jacobian(xc[None])[0])
        # the time and the frame's sky are coupled in y (t = t_L + R/c (...))
        self.assertGreater(abs(f_y[10, 8]), 1e-3 * np.sqrt(f_y[10, 10] * f_y[8, 8]))
        self.assertLess(_corr_norm_diff(f_y, ref), 2e-2)

    def test_the_y_space_ll_matrix_is_exact_on_the_curved_target(self):
        """On the curved target (a Gaussian in y) the ll route by y-perturbations
        recovers diag(1 / SIG_Y^2) to the finite-difference accuracy."""
        from lisatools.globalfit.moves import eigen_refresh
        from lisatools.info_matrix_ll import information_matrix_from_ll

        fm = _curved_fmap()
        y0 = np.zeros(11)
        y0[fm.columns] = MU_Y
        widths = fm.proposal_widths(eigen_refresh.prior_box_widths(
            stub_move("lisa").priors["mbh"], 11))
        info = np.asarray(information_matrix_from_ll(
            lambda Y: curved_lnl(fm.inverse(np.atleast_2d(Y))), y0[None, :],
            xp=np, param_eps=1e-4 * widths))[0]
        diag = np.ones(11)
        diag[fm.columns] = 1 / SIG_Y ** 2
        want = np.diag(diag)
        self.assertLess(_corr_norm_diff(info, want), 1e-3)


    def test_why_y_perturbations_the_ll_hessian_is_not_a_tensor(self):
        """Off the maximum the likelihood Hessian carries the gradient times
        the curvature of F, so J^-T H_x J^-1 is NOT the y-space Hessian: on the
        curved target, ~1 sigma off the mode, the transformed ll matrix is off
        by O(1) while the y-perturbed one is exact."""
        from lisatools.globalfit.moves import eigen_refresh
        from lisatools.info_matrix_ll import information_matrix_from_ll

        fm = _curved_fmap()
        wx = eigen_refresh.prior_box_widths(stub_move("lisa").priors["mbh"], 11)
        wy = fm.proposal_widths(wx)
        diag = np.ones(11)
        diag[fm.columns] = 1 / SIG_Y ** 2
        exact = np.diag(diag)
        x0 = curved_target(np.random.default_rng(5), 4)[0][3]
        y0 = fm.forward(x0[None])[0]
        h_x = np.asarray(information_matrix_from_ll(
            lambda X: curved_lnl(np.atleast_2d(X)), x0[None], xp=np, param_eps=1e-5 * wx))[0]
        h_y = np.asarray(information_matrix_from_ll(
            lambda Y: curved_lnl(fm.inverse(np.atleast_2d(Y))), y0[None], xp=np,
            param_eps=1e-5 * wy))[0]
        transformed = eigen_refresh.transform_information(h_x, fm.jacobian(x0[None])[0])
        self.assertLess(_corr_norm_diff(h_y, exact), 1e-5)
        self.assertGreater(_corr_norm_diff(transformed, exact), 0.3)

    def test_the_refresh_builds_the_gram_table_in_the_frame(self):
        """``MBH_EIGEN_INFO=gram`` with the frame on: the refresh's table is the
        frame's Gram (``_gram_info(..., frame=F)``) at the max-lnL cold walker,
        whitened with the frame's box, and the expansion point is recorded in x."""
        from lisatools.globalfit.moves import eigen_refresh

        with _env(MBH_EIGEN_INFO="gram"):
            H._wdm()
            for w in range(4):
                H._residual(w)
            acs = H._aca([0, 1, 2, 3])
            fm = _fmap("toy")
            move = make_move(acs, 4, scope="walker_max", permute_every=0, adaptive=False,
                             prior_bounds=FRAME_BOUNDS, proposal_frame="lisa",
                             proposal_frame_map=fm)
            move._current_leaf = 0
            coords = H._tempered_coords([0, 1, 2, 3])
            move.setup_likelihood_here(coords[0, :, 0])
            work = SimpleNamespace(coords=coords)
            with self.assertLogs(LOGGER, level="INFO") as logs:
                axes, sig = move._build_eigen_table(0, work)
            best, cold = move._eigen_best_walker(0, work)
            widths = eigen_refresh.prior_box_widths(move.priors["mbh"], 11)
            info = move._gram_info(cold[best], best, widths, frame=fm)
        want_axes, want_sig = eigen_refresh._table_from_info(info, fm.proposal_widths(widths))
        np.testing.assert_allclose(sig, want_sig, rtol=1e-10)
        np.testing.assert_allclose(np.abs(axes), np.abs(want_axes), atol=1e-10)
        x0 = move._eigen_x0[0]
        self.assertLess(abs(x0[10] - cold[best][10]), 1e-6)      # x, not t_L
        self.assertTrue(any("proposals in the LISA frame (t_L sigma" in ln for ln in logs.output))

    def test_the_ll_tables_are_built_in_the_frame(self):
        """The ll route by y-perturbations, both scopes, on the curved target:
        the table's Gaussian on the frame columns is diag(SIG_Y^2)."""
        rng = np.random.default_rng(21)
        x, _ = curved_target(rng, 8)
        fm = _curved_fmap()
        cols = fm.columns
        for scope in ("walker_max", "per_walker"):
            with _clean_env(), self.subTest(scope=scope):
                m = stub_move("lisa", ntemps=2, nwalkers=4)
                m.eigen_table_scope = scope
                m.eigen_eps_rel = 1e-5
                work = SimpleNamespace(coords=x.reshape(2, 4, 1, 11))
                axes, sig = m._build_eigen_table(0, work)
                C = eigen_refresh_cov(axes, sig)
                C = C.reshape(-1, 11, 11)
                for Ck in C:
                    np.testing.assert_allclose(np.sqrt(np.diag(Ck)[cols]), SIG_Y, rtol=0.05)


def eigen_refresh_cov(axes, sig):
    from lisatools.globalfit.moves import eigen_refresh

    return eigen_refresh.table_covariance(axes, sig)


# ---------------------------------------------------------------------------
# a sky hop proposes the same x' in or out of the conjugation
# ---------------------------------------------------------------------------

class SkyHopFrameTest(unittest.TestCase):
    """The ICRS hops stay OUTSIDE the conjugation (they convert frames
    themselves). The alternative -- a LISA-frame hop handed y -- proposes the
    same x' up to the legacy inverse's round trip, with the same factor."""

    def test_same_x_prime_either_way(self):
        from lisatools.sampling.moves.skymodehop import SkyMove

        fm = _fmap("esa")
        m = stub_move("lisa", fmap=fm)
        x = _mbh_rows(32, np.random.default_rng(8), t_range=(2e6, 2.5e7)).reshape(2, 16, 1, 11)
        for which in ("both", "lat", "long"):
            icrs = SkyMove(ind_map=dict(IND_MAP), which=which, coord_frame="icrs",
                           constellation_phase=fm.constellation_phase)
            lisa = SkyMove(ind_map=dict(IND_MAP), which=which, coord_frame="lisa")
            self.assertIsNone(m._frame_for_move(icrs))
            self.assertIs(m._frame_for_move(lisa), fm)
            q, f = icrs.get_proposal({"mbh": x}, np.random.RandomState(4))
            x2, f2, _, _, out = m._frame_proposal(fm, lisa, [x], 0, np.random.RandomState(4))
            self.assertFalse(np.any(out))
            d = _wrap_cols(q["mbh"].reshape(2, 16, 11) - x2)
            with self.subTest(which=which):
                self.assertLess(np.abs(d[..., [7, 8, 9]]).max(), 2e-5)
                self.assertLess(np.abs(d[..., 10]).max(), 3e-3)
                np.testing.assert_array_equal(d[..., :7], 0.0)
                np.testing.assert_allclose(f, f2, atol=1e-7)
                if which != "lat":     # the reflection keeps cos(beta_L), lam_L: |f| ~ 1e-9
                    self.assertGreater(np.abs(f).max(), 1e-7)


# ---------------------------------------------------------------------------
# the adaptation, the cross-walker partners and the sidecar in the frame
# ---------------------------------------------------------------------------

class AdaptationAndJumpInFrameTest(unittest.TestCase):
    def test_the_adapted_scale_is_measured_in_the_frame(self):
        """Samples of the curved target: in y the t_L axis is 0.5 s wide, in x
        t_plunge spreads ~40 s. With the frame on, the adaptation measures the
        y widths on the y axes."""
        from lisatools.globalfit.moves import eigen_refresh

        with _clean_env():
            m = stub_move("lisa", ntemps=2, nwalkers=4, adapt=True)
            m.eigen_adapt_min_visits = 1
            m.eigen_adapt_max_factor = 0          # no factor cap: the raw widths
            rng = np.random.default_rng(9)
            betas = np.array([1.0, 1.0])
            for _ in range(20):
                x, _ = curved_target(rng, 8)
                m._note_eigen_adapt_sample(0, SimpleNamespace(coords=x.reshape(2, 4, 1, 11)), betas)
            fm = _curved_fmap()
            axes = np.eye(11)
            floor = np.full(11, 1e-3)
            _, sig = m._eigen_adapt_table(0, (axes, floor))
        self.assertLess(abs(sig[10] / SIG_Y[3] - 1), 0.3, sig)          # t_L: 0.5 s
        self.assertLess(abs(sig[9] / SIG_Y[1] - 1), 0.3, sig)           # sinbeta_L: 0.1
        x, _ = curved_target(np.random.default_rng(10), 4000)
        self.assertGreater(x[:, 10].std(), 30.0)                        # t_plunge: ~43 s

    def test_the_cross_walker_partners_are_frame_rows(self):
        from eryn.moves import EigenAxisMove

        from lisatools.sampling.moves.walkerdiffjump import WalkerDifferenceJumpMove

        eig = EigenAxisMove(mode="axis")
        m = stub_move("lisa", ntemps=2, nwalkers=4, moves=[eig, WalkerDifferenceJumpMove(eig)])
        m._walker_offset = lambda: 0
        pool, _ = curved_target(np.random.default_rng(12), 8)
        pool = pool.reshape(2, 4, 11)
        with _clean_env():
            m._set_de_pool({0: pool}, 0)
            active, P = m._de_leaf_spec(0)
            off = stub_move("sampling", ntemps=2, nwalkers=4, moves=[eig, WalkerDifferenceJumpMove(eig)])
            off._walker_offset = lambda: 0
            off._set_de_pool({0: pool}, 0)
            _, P_off = off._de_leaf_spec(0)
        np.testing.assert_array_equal(active, [False, True, False, True])
        want = np.unique(_curved_fmap().forward(pool[:, [0, 2]].reshape(-1, 11)), axis=0)
        np.testing.assert_array_equal(np.sort(P, axis=0), np.sort(want, axis=0))
        np.testing.assert_array_equal(np.sort(P_off, axis=0),
                                      np.sort(np.unique(pool[:, [0, 2]].reshape(-1, 11), axis=0), axis=0))


class SidecarFrameGuardTest(unittest.TestCase):
    def test_entries_carry_the_frame_and_never_cross_it(self):
        import tempfile

        from lisatools.globalfit.moves import eigen_table_persist as P

        g = dict(scope="walker_max", ndim=3, ntemps=2, nwalkers=4)
        off = P.make_entry(np.eye(3), np.ones(3), 1, **g)
        self.assertNotIn("frame", off)                         # knob-off entry unchanged
        on = P.make_entry(np.eye(3), np.ones(3), 1, frame="lisa", **g)
        self.assertEqual(on["frame"], "lisa")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s_eigen_tables.pkl")
            P.save_entry(path, "mbh", 0, on)
            P.save_entry(path, "mbh", 1, off)
            self.assertIsNotNone(P.load_entry(path, "mbh", 0, frame="lisa", **g))
            self.assertIsNone(P.load_entry(path, "mbh", 0, **g))
            self.assertIsNotNone(P.load_entry(path, "mbh", 1, **g))
            self.assertIsNone(P.load_entry(path, "mbh", 1, frame="lisa", **g))

    def test_the_move_rebuilds_instead_of_adopting_across_frames(self):
        import tempfile

        rng = np.random.default_rng(13)
        x, _ = curved_target(rng, 8)
        with tempfile.TemporaryDirectory() as d, _clean_env(EIGEN_TABLES_PERSIST="1"):
            store = os.path.join(d, "run.h5")
            on = stub_move("lisa", ntemps=2, store=store)
            work = SimpleNamespace(coords=x.reshape(2, 4, 1, 11))
            on.refresh_inner_move_tables(0, work)
            built = on._eigen_tables[0]
            fresh_on = stub_move("lisa", ntemps=2, store=store)
            fresh_on._eigen_tables, fresh_on._eigen_visit_count = {}, {}
            self.assertTrue(fresh_on._adopt_persisted_eigen_table(0))
            np.testing.assert_array_equal(fresh_on._eigen_tables[0][1], built[1])
            fresh_off = stub_move("sampling", ntemps=2, store=store)
            fresh_off._eigen_tables, fresh_off._eigen_visit_count = {}, {}
            self.assertFalse(fresh_off._adopt_persisted_eigen_table(0))


# ---------------------------------------------------------------------------
# the real MBH move with the frame on
# ---------------------------------------------------------------------------

def _frame_move_kw(**extra):
    kw = dict(prior_bounds=FRAME_BOUNDS, proposal_frame="lisa", proposal_frame_map=_fmap("toy"))
    kw.update(extra)
    return kw


class RealMoveFrameTest(unittest.TestCase):
    def _single(self, stack, env=None, n=3, frame=True, spy=None):
        from lisatools.globalfit.engine import GlobalFitInfo

        with _env(env or {}):
            H._wdm()
            for w in range(4):
                H._residual(w)
            walkers = [0, 1, 2, 3]
            acs = H._aca(walkers)
            kw = _frame_move_kw() if frame else dict(prior_bounds=FRAME_BOUNDS)
            move = make_move(acs, 4, stack=stack, scope="walker_max", permute_every=1,
                             adaptive=True, **kw)
            if spy is not None:
                spy(move)
            cap = H._WarnCapture()
            lat = logging.getLogger("lisatools")
            lat.addHandler(cap)
            try:
                with self.assertLogs(LOGGER, level="INFO") as logs:
                    model = GlobalFitInfo(acs, map, np.random.RandomState(H.SEED))
                    np.random.seed(H.SEED)
                    states, accepted = H._run_proposes(move, model, H._state(walkers), n)
            finally:
                lat.removeHandler(cap)
        # assertLogs keeps the move's own records from propagating: check both
        lines = cap.lines + [ln for ln in logs.output if ln.startswith(("WARNING", "ERROR"))]
        bad = [ln for ln in lines if any(b in ln for b in H._BAD_WARNINGS)]
        self.assertEqual(bad, [], "\n".join(lines))
        return move, states, logs.output

    def test_the_eigen_move_proposes_in_the_lisa_frame(self):
        seen = []
        conj = []

        def spy(move):
            inner = move.moves[0]
            orig = inner.get_proposal

            def rec(s, *a, **k):
                seen.append(np.array(s["mbh"], copy=True))
                return orig(s, *a, **k)

            inner.get_proposal = rec
            fp = move._frame_proposal

            def frame_proposal(*a, **k):
                conj.append(1)
                return fp(*a, **k)

            move._frame_proposal = frame_proposal

        move, states, logs = self._single("eigen", spy=spy)
        # every inner proposal went through the conjugation
        self.assertEqual(len(conj), len(seen))
        self.assertGreater(len(conj), 0)
        fm = move.proposal_frame_map
        y = np.concatenate([a.reshape(-1, 11) for a in seen])
        x = fm.inverse(y)
        # the inner move saw LISA-frame rows: t_L != t_plunge by up to R/c
        self.assertGreater(np.abs(x[:, 10] - y[:, 10]).max(), 10.0)
        np.testing.assert_allclose(_wrap_cols(fm.forward(x) - y), 0.0, atol=1e-8)
        last = states[-1].sub_states["mbh"]
        self.assertGreater(int(last.in_model_accepted.sum()), 0)
        self.assertLess(int(last.in_model_accepted.sum()), int(last.in_model_proposed.sum()))
        lo = np.array([b[0] for b in FRAME_BOUNDS])
        hi = np.array([b[1] for b in FRAME_BOUNDS])
        c = last.coords
        self.assertTrue(np.all((c >= lo) & (c <= hi)))
        lines = [ln for ln in logs if "proposals in the LISA frame (t_L sigma" in ln]
        self.assertGreaterEqual(len(lines), 2)                      # once per leaf refresh
        self.assertIn("s in the sampling basis)", lines[0])
        self.assertIn("[MBH_EIGEN] leaf", lines[0])

    def test_the_stock_like_stack_with_adaptation_tempering_and_the_controller(self):
        seen_sky = []

        def spy(move):
            for mv in move.moves:
                if type(mv).__name__ == "SkyMove":
                    orig = mv.get_proposal

                    def rec(s, *a, _orig=orig, **k):
                        seen_sky.append(np.array(s["mbh"], copy=True))
                        return _orig(s, *a, **k)

                    mv.get_proposal = rec

        env = {"MBH_EIGEN_AXIS_ADAPT": "1", "MBH_EIGEN_ADAPT_MIN_VISITS": "1",
               "MBH_EIGEN_AXIS_ACC_TUNE": "1", "MBH_EIGEN_TEMPER_SIGMAS": "1",
               "MBH_EIGEN_AXIS_ACC_LOG": "1"}
        move, states, logs = self._single("stack", env=env, n=4, spy=spy)
        self.assertTrue(seen_sky)
        # the hops see SAMPLING-basis rows (they convert frames themselves):
        # their t column is a stored t_plunge of the chain, inside its prior
        t = np.concatenate([a[..., 10].ravel() for a in seen_sky])
        self.assertTrue(np.all((t >= FRAME_BOUNDS[10][0]) & (t <= FRAME_BOUNDS[10][1])))
        self.assertTrue(any("proposals in the LISA frame" in ln for ln in logs))
        last = states[-1].sub_states["mbh"]
        self.assertGreater(int(last.in_model_accepted.sum()), 0)

    def test_frame_on_changes_the_stream_and_off_does_not(self):
        _, on, _ = self._single("eigen", n=1)
        _, off, _ = self._single("eigen", n=1, frame=False)
        self.assertFalse(np.array_equal(on[-1].sub_states["mbh"].coords,
                                        off[-1].sub_states["mbh"].coords))


class WalkerBlockFrameTest(H._FanoutCase):
    """The 4-rank walker-block fan-out (one walker per rank, the v9 layout)
    with the frame on equals the single-process run under row-local draws."""

    def test_four_ranks_one_walker_each_with_the_frame_on(self):
        def factory(acs, nwalkers, *, scope, permute_every, adaptive, refresh=1):
            return make_move(acs, nwalkers, scope=scope, permute_every=permute_every,
                             adaptive=adaptive, refresh=refresh, **_frame_move_kw())

        kw = dict(scope="per_walker", adaptive=False, refresh=10, rng=H._RowLocalRandom,
                  permute_every=0)
        with mock.patch.object(H, "_make_move", factory), H._row_local_numpy():
            ref = self._single(4, **kw)
            out = self._world(4, 4, **kw)
        head = out[0]
        self.assertEqual(head["snap"]["move"]._proposal_frame(), "lisa")
        self._assert_states_match(head, ref)
        self._assert_residuals_match(out, ref)
        self._assert_rank_health(out, ref)
        self._assert_no_bad_warnings()


# ---------------------------------------------------------------------------
# the knob
# ---------------------------------------------------------------------------

class KnobTest(unittest.TestCase):
    def test_the_settings_field_default_and_env(self):
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings

        with _clean_env():
            self.assertEqual(MBHSettings().proposal_frame, "sampling")
            self.assertIsNone(MBHSettings().proposal_frame_map)
        with _clean_env(MBH_PROPOSAL_FRAME="lisa"):
            self.assertEqual(MBHSettings().proposal_frame, "lisa")

    def test_the_move_resolves_kwarg_then_env_then_sampling(self):
        m = stub_move(None)
        with _clean_env():
            self.assertEqual(m._proposal_frame(), "sampling")
            self.assertIsNone(m._frame_map())
        with _clean_env(MBH_PROPOSAL_FRAME="LISA"):
            self.assertEqual(m._proposal_frame(), "lisa")
            self.assertIs(m._frame_map(), m.proposal_frame_map)
            m.proposal_frame = "sampling"
            self.assertIsNone(m._frame_map())
        with _clean_env(MBH_PROPOSAL_FRAME="galactic"):
            m.proposal_frame = None
            with self.assertRaises(ValueError):
                m._proposal_frame()

    def test_a_basis_without_frame_columns_is_refused(self):
        from lisatools.globalfit.stock.erebor.source_runtime import check_proposal_frame

        emri = stub_move(None)
        emri.branch_name = "emri"
        emri.proposal_frame_map = None
        with _clean_env(EMRI_PROPOSAL_FRAME="lisa"):
            with self.assertRaises(ValueError) as cm:
                check_proposal_frame(emri)
        self.assertIn("EMRI_PROPOSAL_FRAME='lisa' refused", str(cm.exception))
        with _clean_env():
            check_proposal_frame(emri)                         # sampling: fine

    def test_the_stock_wiring_builds_and_stamps_the_map(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.globalfit.stock.erebor.transforms import make_mbh_transform_container
        from lisatools.sampling.moves.skymodehop import LISAFrameMap, SkyMove

        self.assertIn("proposal_frame", sr.SOURCE_MOVE_SETTINGS_ATTRS)
        self.assertIn("proposal_frame_map", sr.SOURCE_MOVE_SETTINGS_ATTRS)
        ph = _sky_phase()
        mbh = SimpleNamespace(
            transform=make_mbh_transform_container(), proposal_frame="lisa",
            inner_moves=[(SkyMove(ind_map=dict(IND_MAP), which="lat", coord_frame="icrs",
                                  constellation_phase=ph), 0.1)])
        fm = sr.make_mbh_proposal_frame_map(mbh, SimpleNamespace(orbits=None), SimpleNamespace())
        self.assertIsInstance(fm, LISAFrameMap)
        self.assertIs(fm.constellation_phase, ph)                 # the hops' phase
        self.assertEqual(fm.columns, [8, 9, 7, 10])
        move = stub_move(None)
        move.proposal_frame = None
        move.proposal_frame_map = None
        sr.stamp_source_move_settings(move, SimpleNamespace(proposal_frame="lisa",
                                                            proposal_frame_map=fm))
        self.assertEqual(move._proposal_frame(), "lisa")
        self.assertIs(move._frame_map(), fm)
        w = move._frame_widths(np.ones(11))
        np.testing.assert_allclose(w[[8, 9, 7, 10, 0]], [2 * np.pi, 2.0, np.pi, 1.0, 1.0])

    def test_prepare_mbh_branch_builds_the_map_with_the_hops_phase(self):
        import copy
        import pickle

        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sampling.moves.skymodehop import LISAFrameMap

        try:
            from tests.test_mbh_sky_move_wiring import _orbits
        except ImportError:  # pragma: no cover
            from test_mbh_sky_move_wiring import _orbits

        with _clean_env(MBH_PROPOSAL_FRAME="lisa"):
            mbh = sr.SourceMBHSettings()
        mbh.initialize_kwargs = {}
        rows = {0: np.r_[np.zeros(10), 5.0 * 86400.0]}
        gsetup = SimpleNamespace(data_t0=0.0, Tobs=30 * 86400.0, orbits=_orbits())
        gs = SimpleNamespace(n_injections={"MBHB": 1}, data_mode="mojito", mbh_waveform_t0=0.0)
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(sr, "mbh_catalogue_to_sampling_basis", side_effect=lambda r: r):
            out = sr.prepare_mbh_branch(mbh, gsetup, gs)
        self.assertEqual(out.proposal_frame, "lisa")
        self.assertIsInstance(out.proposal_frame_map, LISAFrameMap)
        self.assertIs(out.proposal_frame_map.constellation_phase,
                      out.inner_moves[1][0].constellation_phase)
        s2 = pickle.loads(pickle.dumps(copy.deepcopy(out)))
        x = _mbh_rows(3, np.random.default_rng(0))
        np.testing.assert_array_equal(s2.proposal_frame_map.forward(x), out.proposal_frame_map.forward(x))

    def test_the_stock_build_refuses_an_impossible_frame_before_the_run(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        emri = stub_move(None)
        emri.branch_name = "emri"
        emri.proposal_frame_map = None
        curr = SimpleNamespace(source_info={"emri": SimpleNamespace()})
        with _clean_env(EMRI_PROPOSAL_FRAME="lisa"), \
                mock.patch.object(sr, "build_emri_move_runtime", return_value=emri):
            with self.assertRaises(ValueError):
                sr.build_source_moves(curr, None, None, None, {})
        with _clean_env(), mock.patch.object(sr, "build_emri_move_runtime", return_value=emri):
            self.assertIs(sr.build_source_moves(curr, None, None, None, {})["emri_pe"], emri)


# ---------------------------------------------------------------------------
# the harness: the curved posterior, frame off vs on
# ---------------------------------------------------------------------------

def curved_run(frame, seed, visits, repeats=25, ntemps=8, tmax=10.0, **stub_kw):
    rng = np.random.default_rng(seed)
    betas = 1.0 / np.geomspace(1.0, tmax, ntemps)
    m = stub_move(frame, ntemps=ntemps, **stub_kw)
    x0, _ = curved_target(rng, ntemps * 4)
    work = SimpleNamespace(coords=x0.reshape(ntemps, 4, 1, 11).copy())
    ll = curved_lnl(work.coords[:, :, 0, :])
    lp = m.priors["mbh"].logpdf(work.coords.reshape(-1, 11)).reshape(ntemps, 4)
    trace = []
    nacc = 0
    for _ in range(visits):
        m.refresh_inner_move_tables(0, work, betas=betas)
        ll, lp, a = mh_sweep(m, work, ll, lp, betas, rng, repeats, trace)
        nacc += a
        m._note_eigen_adapt_sample(0, work, betas)
    return np.array(trace), nacc / (visits * repeats * 4)


class CurvedPosteriorHarnessTest(unittest.TestCase):
    """4 walkers x 8 rungs (T_max 10), 25 repeats per visit, the table
    refreshed every 5 visits by the ll route at the max-lnL cold walker."""

    VISITS = 100

    def test_the_frame_mixes_the_curved_posterior_faster_and_samples_it(self):
        res = {}
        xt, wt = curved_target(np.random.default_rng(0), 200000)
        with _clean_env():
            for frame in ("sampling", "lisa"):
                tr, acc = curved_run(frame, 1, self.VISITS)
                tr = tr[tr.shape[0] // 4:]
                res[frame] = dict(
                    acc=acc,
                    tau_t=float(np.mean([tau(tr[:, w, 10]) for w in range(4)])),
                    tau_a=float(np.mean([tau(tr[:, w, 8]) for w in range(4)])),
                    tau_s=float(np.mean([tau(tr[:, w, 9]) for w in range(4)])),
                    t_mean=float(tr[..., 10].mean() - T_MID), t_std=float(tr[..., 10].std()),
                    a_mean=float(tr[..., 8].mean()), a_std=float(tr[..., 8].std()))
        tgt = dict(t_mean=float(np.average(xt[:, 10], weights=wt) - T_MID), t_std=float(xt[:, 10].std()),
                   a_mean=float(np.average(xt[:, 8], weights=wt)), a_std=float(xt[:, 8].std()))
        if os.environ.get("PROPOSAL_FRAME_HARNESS_VERBOSE"):
            print(f"\ntarget: t-T {tgt['t_mean']:.2f} +- {tgt['t_std']:.2f} s, alpha "
                  f"{tgt['a_mean']:.4f} +- {tgt['a_std']:.4f}")
            for frame, r in res.items():
                print(f"{frame:9s}: cold acc {r['acc']:.2f}; IAT (repeats) t {r['tau_t']:.0f}, "
                      f"alpha {r['tau_a']:.0f}, sin_delta {r['tau_s']:.0f}; t-T {r['t_mean']:.2f} +- "
                      f"{r['t_std']:.2f} s, alpha {r['a_mean']:.4f} +- {r['a_std']:.4f}")
        on, off = res["lisa"], res["sampling"]
        self.assertLess(on["tau_t"], 0.5 * off["tau_t"], res)
        self.assertLess(on["tau_s"], 0.5 * off["tau_s"], res)
        # the frame-on chain samples the target (moments within ~4 standard errors)
        n_eff = 4 * (0.75 * self.VISITS * 25) / max(on["tau_t"], on["tau_a"])
        self.assertLess(abs(on["t_mean"] - tgt["t_mean"]), 4 * tgt["t_std"] / np.sqrt(n_eff), (on, tgt))
        self.assertLess(abs(on["a_mean"] - tgt["a_mean"]), 4 * tgt["a_std"] / np.sqrt(n_eff), (on, tgt))
        self.assertLess(abs(on["t_std"] / tgt["t_std"] - 1), 0.2, (on, tgt))
        self.assertLess(abs(on["a_std"] / tgt["a_std"] - 1), 0.2, (on, tgt))


if __name__ == "__main__":
    unittest.main()
