"""Default inner-move stack selection for the single-source PE branches.

SOBBH / MBH / EMRI default their addremove inner move to the eryn
eigen-axis proposal (``inner_move_kind = "eigen"``), with the legacy
stretch reachable per branch via ``{BRANCH}_INNER_MOVE_KIND=stretch`` (env)
or the explicit Settings field. ``make_default_inner_moves`` /
``resolve_inner_moves`` in ``stock/erebor/common.py`` are the single
implementation every fallback site calls.
"""

import os
import types
import unittest
from unittest import mock

from eryn.moves import EigenAxisMove, StretchMove

from lisatools.globalfit.stock.erebor.common import (
    make_default_inner_moves,
    resolve_inner_moves,
)


class MakeDefaultInnerMovesTest(unittest.TestCase):
    def test_eigen_stack(self):
        stack = make_default_inner_moves("eigen")
        self.assertEqual(len(stack), 1)
        move, weight = stack[0]
        self.assertIsInstance(move, EigenAxisMove)
        self.assertEqual(move.mode, "axis")
        self.assertEqual(weight, 1.0)

    def test_stretch_stack(self):
        stack = make_default_inner_moves("stretch")
        move, weight = stack[0]
        self.assertIsInstance(move, StretchMove)
        self.assertEqual(weight, 1.0)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            make_default_inner_moves("nope")

    def test_case_and_whitespace_insensitive(self):
        move, _ = make_default_inner_moves(" Eigen ")[0]
        self.assertIsInstance(move, EigenAxisMove)


class ResolveInnerMovesTest(unittest.TestCase):
    def test_fills_none_from_kind(self):
        ns = types.SimpleNamespace(inner_moves=None, inner_move_kind="stretch")
        out = resolve_inner_moves(ns)
        self.assertIsInstance(out[0][0], StretchMove)
        self.assertIs(ns.inner_moves, out)

    def test_explicit_inner_moves_win(self):
        sentinel = [("custom", 2.0)]
        ns = types.SimpleNamespace(
            inner_moves=sentinel, inner_move_kind="eigen"
        )
        self.assertIs(resolve_inner_moves(ns), sentinel)

    def test_missing_kind_defaults_to_eigen(self):
        ns = types.SimpleNamespace(inner_moves=None)
        out = resolve_inner_moves(ns)
        self.assertIsInstance(out[0][0], EigenAxisMove)


class SettingsFieldTest(unittest.TestCase):
    """The three branch Settings expose the knob with the eigen default."""

    def _cases(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        return [
            (SOBBHSettings, "SOBBH_INNER_MOVE_KIND"),
            (MBHSettings, "MBH_INNER_MOVE_KIND"),
            (EMRISettings, "EMRI_INNER_MOVE_KIND"),
        ]

    def test_default_is_eigen(self):
        for cls, env in self._cases():
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env, None)
                self.assertEqual(cls().inner_move_kind, "eigen", cls.__name__)

    def test_env_escape_to_stretch(self):
        for cls, env in self._cases():
            with mock.patch.dict(os.environ, {env: "stretch"}):
                self.assertEqual(
                    cls().inner_move_kind, "stretch", cls.__name__
                )

    def test_explicit_kwarg_beats_env(self):
        for cls, env in self._cases():
            with mock.patch.dict(os.environ, {env: "stretch"}):
                self.assertEqual(
                    cls(inner_move_kind="eigen").inner_move_kind,
                    "eigen",
                    cls.__name__,
                )


class EigenRefreshFieldTest(unittest.TestCase):
    """Per-branch refresh cadence: cheap SOBBH refreshes often; the
    per-row-dense MBH/EMRI information matrices refresh sparsely."""

    def _cases(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        return [
            (SOBBHSettings, "SOBBH_EIGEN_REFRESH", 10),
            (MBHSettings, "MBH_EIGEN_REFRESH", 100),
            (EMRISettings, "EMRI_EIGEN_REFRESH", 100),
        ]

    def test_defaults(self):
        for cls, env, default in self._cases():
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env, None)
                self.assertEqual(
                    cls().eigen_refresh_every, default, cls.__name__
                )

    def test_env_override(self):
        for cls, env, _ in self._cases():
            with mock.patch.dict(os.environ, {env: "37"}):
                self.assertEqual(
                    cls().eigen_refresh_every, 37, cls.__name__
                )


class EigenScopeFieldTest(unittest.TestCase):
    """SOBBH builds per-(temp, walker) tables (cheap batched scorer);
    MBH/EMRI build one table at the max-lnL cold walker."""

    def _cases(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        return [
            (SOBBHSettings, "SOBBH_EIGEN_SCOPE", "per_walker"),
            (MBHSettings, "MBH_EIGEN_SCOPE", "walker_max"),
            (EMRISettings, "EMRI_EIGEN_SCOPE", "walker_max"),
        ]

    def test_defaults(self):
        for cls, env, default in self._cases():
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env, None)
                self.assertEqual(
                    cls().eigen_table_scope, default, cls.__name__
                )

    def test_env_override(self):
        for cls, env, _ in self._cases():
            with mock.patch.dict(os.environ, {env: "walker_max"}):
                self.assertEqual(
                    cls().eigen_table_scope, "walker_max", cls.__name__
                )


if __name__ == "__main__":
    unittest.main()
