"""The eigen inner move's larger-jump share (user ruling 2026-10-13).

``EigenAxisJumpMixMove`` (``lisatools.sampling.moves.eigenjumpmix``) gives a
fraction ``jump_mix`` of the eigen proposals a step ``jump_mix_factor`` x the
table width; ``{BRANCH}_EIGEN_JUMP_MIX`` / ``{BRANCH}_EIGEN_JUMP_FACTOR``
(code defaults 0 / 3) build it for MBH / EMRI / SOBBH. Pinned here:

* mix 0 reproduces the eryn ``EigenAxisMove`` draws bit for bit (and the stock
  stack at mix 0 IS the plain eryn move);
* the base step of the mixture path mirrors eryn's line for line (mix > 0
  with factor 1 equals eryn's proposal);
* mix 1 with factor 3 triples every step; mix 0.1 widens 10 % of the rows;
* the knob reaches the PRODUCTION inner moves (prepare_emri_branch /
  prepare_sobbh_branch / make_mbh_inner_moves -> the branch Setup -> the
  move) and the move's construction log line names it -- the exact line the
  launchers quote.
"""

from __future__ import annotations

import copy
import os
import pickle
import re
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves import EigenAxisMove
from eryn.utils import PeriodicContainer

from lisatools.sampling.moves.eigenjumpmix import EigenAxisJumpMixMove

NT, NW, NL, ND = 4, 6, 2, 5
HERE = os.path.dirname(os.path.abspath(__file__))
NINE_MO_V9 = os.path.join(os.path.dirname(HERE), "scripts", "fstat_proposal", "submit_gf_9mo_v9.sh")
_KNOBS = tuple(f"{b}_{k}" for b in ("EMRI", "MBH", "SOBBH")
               for k in ("EIGEN_JUMP_MIX", "EIGEN_JUMP_FACTOR", "INNER_MOVE_KIND",
                         "SKY_MOVE_BOTH_WEIGHT", "SKY_MOVE_LAT_WEIGHT", "SKY_MOVE_LONG_WEIGHT"))


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _tables(kind, rng):
    q, _ = np.linalg.qr(rng.normal(size=(ND, ND)))
    s = rng.uniform(0.1, 2.0, size=ND)
    if kind == "shared":
        return q, s
    if kind == "per_leaf":
        return np.stack([q, q.T]), np.stack([s, s[::-1]])
    axes = np.broadcast_to(q, (NT, NW, NL, ND, ND)).copy()
    sig = rng.uniform(0.1, 2.0, size=(NT, NW, NL, ND))
    return axes, sig


def _coords(rng):
    return {"b": rng.normal(size=(NT, NW, NL, ND))}


def _pair(kind="shared", mode="axis", periodic=None, **mix):
    rng = np.random.default_rng(11)
    axes, sig = _tables(kind, rng)
    plain = EigenAxisMove(mode=mode, periodic=periodic)
    mixed = EigenAxisJumpMixMove(mode=mode, periodic=periodic, **mix)
    plain.set_axes("b", axes, sig)
    mixed.set_axes("b", axes, sig)
    return plain, mixed, _coords(rng)


def _draw(move, coords, rng, inds=None):
    kw = {} if inds is None else {"branches_inds": {"b": inds}}
    q, f = move.get_proposal({k: v.copy() for k, v in coords.items()}, rng, **kw)
    return q["b"], f


def _rngs(seed=7):
    return (lambda: np.random.RandomState(seed)), (lambda: np.random.default_rng(seed))


class MixZeroIsTheErynMoveTest(unittest.TestCase):
    def test_bit_identical_draws_and_stream(self):
        periodic = PeriodicContainer({"b": {1: 2 * np.pi}})
        inds = np.ones((NT, NW, NL), dtype=bool)
        inds[1, 2, 0] = False
        for kind in ("shared", "per_leaf", "full"):
            for mode in ("axis", "full"):
                for per in (None, periodic):
                    for make in _rngs():
                        with self.subTest(kind=kind, mode=mode, periodic=per is not None,
                                          rng=type(make()).__name__):
                            plain, mixed, x = _pair(kind, mode, per, jump_mix=0.0)
                            r1, r2 = make(), make()
                            for _ in range(3):
                                qa, fa = _draw(plain, x, r1, inds)
                                qb, fb = _draw(mixed, x, r2, inds)
                                np.testing.assert_array_equal(qa, qb)
                                np.testing.assert_array_equal(fa, fb)
                            # the stream is consumed identically
                            self.assertEqual(float(r1.uniform()), float(r2.uniform()))

    def test_the_stock_stack_at_mix_zero_is_the_plain_eryn_move(self):
        from lisatools.globalfit.stock.erebor.common import make_default_inner_moves

        (move, w), = make_default_inner_moves("eigen")
        self.assertIs(type(move), EigenAxisMove)
        (move, w), = make_default_inner_moves("eigen", jump_mix=0.0, jump_factor=3.0)
        self.assertIs(type(move), EigenAxisMove)
        (move, w), = make_default_inner_moves("eigen", jump_mix=0.1, jump_factor=3.0)
        self.assertIs(type(move), EigenAxisJumpMixMove)
        self.assertEqual((move.jump_mix, move.jump_mix_factor, move.mode, w), (0.1, 3.0, "axis", 1.0))
        (move, w), = make_default_inner_moves("stretch", jump_mix=0.5)   # n/a to the stretch
        self.assertNotIsInstance(move, EigenAxisMove)


class MixturePathTest(unittest.TestCase):
    def test_the_base_step_mirrors_eryn(self):
        """mix > 0 with factor 1: every row's step is eryn's (stream order kept)."""
        periodic = PeriodicContainer({"b": {0: 1.0}})
        for kind in ("shared", "per_leaf", "full"):
            for mode in ("axis", "full"):
                for make in _rngs(3):
                    with self.subTest(kind=kind, mode=mode, rng=type(make()).__name__):
                        plain, mixed, x = _pair(kind, mode, periodic, jump_mix=0.5, jump_mix_factor=1.0)
                        qa, _ = _draw(plain, x, make())
                        qb, fb = _draw(mixed, x, make())
                        np.testing.assert_array_equal(qa, qb)
                        np.testing.assert_array_equal(fb, np.zeros((NT, NW)))

    def test_mix_one_factor_three_triples_every_step(self):
        for mode in ("axis", "full"):
            with self.subTest(mode=mode):
                plain, mixed, x = _pair("full", mode, jump_mix=1.0, jump_mix_factor=3.0)
                qa, _ = _draw(plain, x, np.random.RandomState(5))
                qb, _ = _draw(mixed, x, np.random.RandomState(5))
                np.testing.assert_allclose(qb - x["b"], 3.0 * (qa - x["b"]), rtol=1e-12, atol=1e-14)
                np.testing.assert_array_equal(mixed.last_jump_scale["b"], 3.0)

    def test_mix_one_step_scale_statistically(self):
        """The 1-D axis step has std = 3 x the table sigma at mix 1, factor 3."""
        rng = np.random.default_rng(0)
        move = EigenAxisJumpMixMove(mode="axis", jump_mix=1.0, jump_mix_factor=3.0)
        move.set_axes("b", np.eye(ND), np.full(ND, 0.5))
        x = {"b": np.zeros((50, 400, 1, ND))}
        q, _ = move.get_proposal(x, rng)
        step = np.abs(q["b"]).sum(axis=-1).ravel()      # one axis moves per row
        signed = q["b"].sum(axis=-1).ravel()
        self.assertAlmostEqual(np.std(signed) / 0.5, 3.0, delta=0.05)
        self.assertGreater(step.mean(), 0)

    def test_mix_tenth_widens_a_tenth_of_the_rows(self):
        rng = np.random.default_rng(1)
        move = EigenAxisJumpMixMove(mode="axis", jump_mix=0.1, jump_mix_factor=3.0)
        plain = EigenAxisMove(mode="axis")
        for m in (move, plain):
            m.set_axes("b", np.eye(ND), np.full(ND, 0.5))
        x = {"b": np.zeros((100, 1000, 2, ND))}
        n = x["b"][..., 0].size
        qm, _ = move.get_proposal(x, np.random.RandomState(9))
        qp, _ = plain.get_proposal(x, np.random.RandomState(9))
        ratio = (qm["b"].sum(-1) / qp["b"].sum(-1)).ravel()
        wide = np.isclose(ratio, 3.0)
        self.assertTrue(np.all(wide | np.isclose(ratio, 1.0)))
        frac = wide.mean()
        sd = np.sqrt(0.1 * 0.9 / n)
        self.assertLess(abs(frac - 0.1), 5 * sd, (frac, sd))
        np.testing.assert_array_equal(np.sort(np.unique(move.last_jump_scale["b"])), [1.0, 3.0])

    def test_validation_and_pickle(self):
        for bad in (dict(jump_mix=-0.1), dict(jump_mix=1.5), dict(jump_mix_factor=0.0)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                EigenAxisJumpMixMove(mode="axis", **bad)
        m = EigenAxisJumpMixMove(mode="axis", jump_mix=0.1, jump_mix_factor=3.0)
        m2 = pickle.loads(pickle.dumps(copy.deepcopy(m)))
        self.assertEqual((m2.jump_mix, m2.jump_mix_factor), (0.1, 3.0))
        self.assertEqual(m2.describe(), "jump mix 0.10 x 3.0")


# ---------------------------------------------------------------------------
# production wiring
# ---------------------------------------------------------------------------

def _emri_prepared():
    from lisatools.globalfit.stock.erebor import source_runtime as sr

    emri = sr.SourceEMRISettings()
    emri.initialize_kwargs = {}
    rows = {}
    for k in (0, 3):
        r = np.zeros(14)
        r[5] = 1.0          # xI0 prograde, Phi_theta0 0
        rows[k] = r
    gs = SimpleNamespace(n_injections={"EMRI": 2}, data_mode="mojito")
    tc = SimpleNamespace(both_inverse_transforms=lambda x: x)
    with mock.patch.object(sr, "source_catalogue", return_value=rows), \
            mock.patch.object(sr, "emri_catalogue_to_waveform_basis", side_effect=lambda r: r), \
            mock.patch.object(sr, "make_emri_transform_container", return_value=tc):
        return sr.prepare_emri_branch(emri, SimpleNamespace(force_backend="cpu"), gs)


def _sobbh_prepared():
    from lisatools.globalfit.stock.erebor import injections as inj
    from lisatools.globalfit.stock.erebor import source_runtime as sr
    from lisatools.globalfit.stock.erebor import transforms as trf

    sobbh = sr.SourceSOBBHSettings()
    sobbh.initialize_kwargs = {}
    sobbh.chirp_fdot_max = 0.0
    rows = {k: np.full(11, float(k)) for k in (5, 1)}
    gs = SimpleNamespace(n_injections={"SOBHB": 2}, data_mode="mojito")
    tc = SimpleNamespace(both_inverse_transforms=lambda x: x)
    with mock.patch.object(sr, "source_catalogue", return_value=rows), \
            mock.patch.object(inj, "sobbh_catalogue_to_waveform_basis", side_effect=lambda r: r), \
            mock.patch.object(trf, "make_sobbh_transform_container", return_value=tc):
        return sr.prepare_sobbh_branch(sobbh, SimpleNamespace(force_backend="cpu"), gs)


def _addremove_move(branch, inner_moves):
    """The real add/remove move constructor (its log line is the proof)."""
    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    return ResidualAddOneRemoveOneMove(
        branch, (2, 2, 1, 12 if branch == "emri" else 11), lambda *a, **k: None, {}, {},
        None, 2, None, None, inner_moves,
    )


class ProductionWiringTest(unittest.TestCase):
    LOGGER = "lisatools.globalfit.moves.addremovemove"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_emri_knob_reaches_the_production_move(self):
        """EMRI_EIGEN_JUMP_MIX=0.1 -> prepare_emri_branch -> EMRISetup -> the move."""
        from lisatools.globalfit.stock.erebor.emri import EMRISetup

        with _clean_env(EMRI_EIGEN_JUMP_MIX="0.1"):
            emri = _emri_prepared()
        (move, w), = emri.inner_moves
        self.assertIsInstance(move, EigenAxisJumpMixMove)
        self.assertEqual((move.jump_mix, move.jump_mix_factor, w), (0.1, 3.0, 1.0))
        emri.Tobs, emri.dt, emri.log_dir = 31536000.0, 10.0, self.tmp
        with _clean_env(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup = EMRISetup(emri)
        self.assertIs(setup.inner_moves[0][0], move)
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            ar = _addremove_move("emri", setup.inner_moves)
        self.assertIn(
            "[EMRI] inner-move stack: EigenAxisJumpMixMove 1.000 (jump mix 0.10 x 3.0)",
            "\n".join(cm.output))
        self.assertIs(ar.moves[0], move)

    def test_emri_default_is_the_plain_move(self):
        with _clean_env():
            emri = _emri_prepared()
        self.assertIs(type(emri.inner_moves[0][0]), EigenAxisMove)
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            _addremove_move("emri", emri.inner_moves)
        self.assertIn("[EMRI] inner-move stack: EigenAxisMove 1.000\n", "\n".join(cm.output) + "\n")

    def test_sobbh_knob_reaches_the_inner_moves(self):
        with _clean_env(SOBBH_EIGEN_JUMP_MIX="0.1", SOBBH_EIGEN_JUMP_FACTOR="3.0"):
            sobbh = _sobbh_prepared()
        (move, _), = sobbh.inner_moves
        self.assertIsInstance(move, EigenAxisJumpMixMove)
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            _addremove_move("sobbh", sobbh.inner_moves)
        self.assertIn("[SOBBH] inner-move stack: EigenAxisJumpMixMove 1.000 (jump mix 0.10 x 3.0)",
                      "\n".join(cm.output))

    def test_mbh_mix_rides_the_eigen_part_only(self):
        from lisatools.globalfit.moves.addremovemove import describe_inner_moves
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with _clean_env(MBH_EIGEN_JUMP_MIX="0.1"):
            mbh = sr.SourceMBHSettings()
        with self.assertLogs(sr.logger, level="WARNING"):   # no orbits: analytic phase
            moves = sr.make_mbh_inner_moves(mbh, SimpleNamespace(), SimpleNamespace())
        text = describe_inner_moves([m for m, _ in moves], [w for _, w in moves])
        self.assertEqual(
            text,
            "EigenAxisJumpMixMove 0.920 (jump mix 0.10 x 3.0) | SkyMove(both, icrs) 0.020 | "
            "SkyMove(lat, icrs) 0.040 | SkyMove(long, icrs) 0.020",
        )

    def test_the_launcher_quotes_the_line_the_move_prints(self):
        """The 9mo launcher's comment names the exact construction log line."""
        from lisatools.globalfit.moves.addremovemove import describe_inner_moves

        src = open(NINE_MO_V9).read()
        quoted = re.search(r"(?m)^#\s+(\[EMRI\] inner-move stack: .*)$", src).group(1)
        mix = float(re.search(r"(?m)^export EMRI_EIGEN_JUMP_MIX=\$\{EMRI_EIGEN_JUMP_MIX:-([0-9.]+)\}$", src).group(1))
        fac = float(re.search(r"(?m)^export EMRI_EIGEN_JUMP_FACTOR=\$\{EMRI_EIGEN_JUMP_FACTOR:-([0-9.]+)\}$", src).group(1))
        from lisatools.globalfit.stock.erebor.common import make_default_inner_moves

        stack = make_default_inner_moves("eigen", jump_mix=mix, jump_factor=fac)
        self.assertEqual(
            quoted,
            "[EMRI] inner-move stack: " + describe_inner_moves([m for m, _ in stack], [1.0]),
        )


if __name__ == "__main__":
    unittest.main()
