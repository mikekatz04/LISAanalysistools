"""PE-mode GB REPLACE move (``rj_replace_pe``) — install + staging knobs.

USER DIRECTIVE 2026-08-28: "we need a PE replace that also uses the same
machinery as fstat pe". The search replace (``rj_replace``) already draws
full 9-column candidates from the F-stat birth container and scores exact
add-deltas both sides; the PE install is the same move with PE-flavored
kwargs, and it MUST resolve the exact-detailed-balance path:

* it must NOT carry the recipe's ``replace_search_stage`` stamp, and its
  name must not trip the ``"search" in name`` scoping idiom — otherwise
  :meth:`GBSpecialBase._replace_fstat_max` auto-arms the search-only
  maximize-then-pretend draw (DB deliberately broken there);
* "same machinery as fstat pe" = the EPOCH CENTER TABLE for the
  extrinsic centers, exactly what ``_rj_birth_perrow`` hands the pe-named
  F-stat moves. The replace's own center knob
  (:meth:`GBSpecialBase._replace_ctr_mode`) is therefore made
  STAGE-AWARE: unstamped / search-stamped moves keep ``"perrow"``
  bit-identically, a ``replace_pe_stage``-stamped move resolves
  ``"table"``, and an explicitly set ``GB_REPLACE_CTR_MODE`` still wins
  either way (the project-wide env-knob convention).

Move-level resolution is pinned here with the lightweight-stub harness of
``test_gb_replace_fstat_max.py``; the full-flow numerics ride on
``_debug_verify_replace_step`` (GB_DEBUG) in production, as for every
other replace knob. The recipe install site itself needs a built fit, so
only its two load-bearing structural invariants are asserted (the PE
construction exists; the search stamp is written exactly once).
"""

import dataclasses
import inspect
import os
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

#: The PE replace move's stock name (recipe.py + gb_no_fg recipe wiring).
PE_REPLACE_NAME = "rj_replace_pe"


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


class _SearchReplace:
    """The production SEARCH install: plain name + the stage stamp."""

    name = "rj_replace"
    replace_search_stage = True
    rj_fstat_dist_birth = True


class _PEReplace:
    """The PE install: pe-suffixed name + the PE stage stamp, no search stamp."""

    name = PE_REPLACE_NAME
    replace_pe_stage = True
    rj_fstat_dist_birth = True


class _UnstampedReplace:
    """A replace move carrying neither stamp (the pre-staging default)."""

    name = "rj_replace"
    rj_fstat_dist_birth = True


class ReplaceCtrModeStagingTest(unittest.TestCase):
    """``_replace_ctr_mode`` resolution: search perrow / PE table / env wins."""

    def _mode(self, stub):
        return GBSpecialBase._replace_ctr_mode(stub)

    def test_unstamped_default_is_perrow(self):
        # BIT-IDENTITY GUARD: every move that existed before the PE
        # install resolves exactly what it resolved before.
        with _EnvPatch(GB_REPLACE_CTR_MODE=None):
            self.assertEqual(self._mode(_UnstampedReplace()), "perrow")

    def test_search_stamp_default_is_perrow(self):
        with _EnvPatch(GB_REPLACE_CTR_MODE=None):
            self.assertEqual(self._mode(_SearchReplace()), "perrow")

    def test_pe_stamp_default_is_table(self):
        # "same machinery as fstat pe" -> the epoch center table.
        with _EnvPatch(GB_REPLACE_CTR_MODE=None):
            self.assertEqual(self._mode(_PEReplace()), "table")

    def test_explicit_env_wins_over_the_stamp(self):
        with _EnvPatch(GB_REPLACE_CTR_MODE="perrow"):
            self.assertEqual(self._mode(_PEReplace()), "perrow")
        with _EnvPatch(GB_REPLACE_CTR_MODE="table"):
            self.assertEqual(self._mode(_SearchReplace()), "table")
            self.assertEqual(self._mode(_UnstampedReplace()), "table")

    def test_auto_is_the_explicit_spelling_of_the_default(self):
        with _EnvPatch(GB_REPLACE_CTR_MODE="auto"):
            self.assertEqual(self._mode(_SearchReplace()), "perrow")
            self.assertEqual(self._mode(_PEReplace()), "table")
            self.assertEqual(self._mode(_UnstampedReplace()), "perrow")

    def test_bogus_env_still_raises(self):
        with _EnvPatch(GB_REPLACE_CTR_MODE="bogus"):
            with self.assertRaises(ValueError):
                self._mode(_PEReplace())

    def test_case_and_whitespace_tolerated(self):
        with _EnvPatch(GB_REPLACE_CTR_MODE="  TABLE "):
            self.assertEqual(self._mode(_UnstampedReplace()), "table")


class PEReplaceCarriesNoSearchIdiomTest(unittest.TestCase):
    """The PE install must resolve every search-scoped idiom to its PE side."""

    def test_name_trips_no_search_scoping(self):
        self.assertNotIn("search", PE_REPLACE_NAME.lower())
        # gb_no_fg splits stock names on this suffix (``include_search`` /
        # ``pe_names``): the PE replace must land in the PE list.
        self.assertFalse(PE_REPLACE_NAME.endswith("_search"))

    def test_fstat_max_is_off_for_the_pe_install(self):
        # _replace_fstat_max auto-arms on "search" in the name OR the
        # search stamp; the PE install has neither -> exact-DB draw path.
        with _EnvPatch(GB_REPLACE_FSTAT_MAX=None):
            self.assertFalse(GBSpecialBase._replace_fstat_max(_PEReplace()))
            self.assertTrue(GBSpecialBase._replace_fstat_max(_SearchReplace()))

    def test_pe_stamp_does_not_leak_into_fstat_max(self):
        # A PE stamp is not a search stamp under any spelling of the knob.
        for env in (None, "auto"):
            with _EnvPatch(GB_REPLACE_FSTAT_MAX=env):
                self.assertFalse(
                    GBSpecialBase._replace_fstat_max(_PEReplace()))

    def test_rj_birth_ctr_mode_matches_the_pe_fstat_moves(self):
        # _rj_birth_perrow is the pe-vs-search center split for the fstat
        # BIRTH moves (per-row in search, epoch table in PE). The replace
        # step never reads it (it reads _replace_ctr_mode), but the PE
        # replace's NAME must resolve to the same PE side as rj_fstat_pe
        # so no idiom keyed on the name can hand it search behavior.
        class _PEFstat:
            name = "rj_fstat_pe"

        class _SearchFstat:
            name = "rj_fstat_search"

        with _EnvPatch(GB_RJ_BIRTH_CTR_MODE=None):
            self.assertTrue(GBSpecialBase._rj_birth_perrow(_SearchFstat()))
            self.assertFalse(GBSpecialBase._rj_birth_perrow(_PEFstat()))
            self.assertFalse(GBSpecialBase._rj_birth_perrow(_PEReplace()))


class ReplaceTableGatingTest(unittest.TestCase):
    """The exact ``_tbl`` expression ``_run_replace_step`` evaluates."""

    def _tbl(self, m):
        return (m._fstat_ctr_table_active()
                if m._replace_ctr_mode() == "table" else None)

    def _move_with_table(self, **stamps):
        from tests.test_gb_cap_cell_grid import _move

        m = _move(4)
        m.name = "rj_replace"
        m.rj_fstat_dist_birth = True
        m._fstat_ctr_table = {"f0_mHz": np.array([5.0])}
        for k, v in stamps.items():
            setattr(m, k, v)
        return m

    def test_search_stamp_never_consults_the_table(self):
        m = self._move_with_table(replace_search_stage=True)
        with _EnvPatch(GB_REPLACE_CTR_MODE=None, GB_FSTAT_CTR_MODE=None):
            self.assertIsNone(self._tbl(m))

    def test_pe_stamp_consumes_the_table(self):
        m = self._move_with_table(
            replace_pe_stage=True, name=PE_REPLACE_NAME)
        with _EnvPatch(GB_REPLACE_CTR_MODE=None, GB_FSTAT_CTR_MODE=None):
            self.assertIsNotNone(self._tbl(m))

    def test_pe_stamp_degrades_gracefully_without_a_table(self):
        # No epoch table (prior-container install / empty support): the
        # step falls back to the per-row F-stat, exactly as under
        # GB_FSTAT_CTR_MODE=unit.
        m = self._move_with_table(
            replace_pe_stage=True, name=PE_REPLACE_NAME)
        m._fstat_ctr_table = None
        with _EnvPatch(GB_REPLACE_CTR_MODE=None, GB_FSTAT_CTR_MODE=None):
            self.assertIsNone(self._tbl(m))


class SettingsKnobTest(unittest.TestCase):
    """``GBSettings.pe_rj_replace``: env-seeded, default ON."""

    def _field(self, name):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        for f in dataclasses.fields(GBSettings):
            if f.name == name:
                return f
        raise AssertionError(f"{name} field missing on GBSettings")

    def test_default_on(self):
        with _EnvPatch(GB_PE_RJ_REPLACE=None):
            self.assertIs(self._field("pe_rj_replace").default_factory(), True)

    def test_env_off(self):
        with _EnvPatch(GB_PE_RJ_REPLACE="0"):
            self.assertIs(self._field("pe_rj_replace").default_factory(), False)
        with _EnvPatch(GB_PE_RJ_REPLACE="1"):
            self.assertIs(self._field("pe_rj_replace").default_factory(), True)

    def test_sibling_search_knob_untouched(self):
        with _EnvPatch(GB_SEARCH_RJ_REPLACE=None):
            self.assertIs(
                self._field("search_rj_replace").default_factory(), True)


class RecipeInstallStructureTest(unittest.TestCase):
    """The two load-bearing invariants of the install site.

    A full recipe-level test would need a built fit (10-26 GB); these are
    the assertions that actually protect the directive.
    """

    def _src(self):
        from lisatools.globalfit.recipe import build_gb_moves

        return inspect.getsource(build_gb_moves)

    def test_pe_replace_is_constructed(self):
        src = self._src()
        self.assertIn(f'name="{PE_REPLACE_NAME}"', src)
        self.assertIn("pe_rj_replace", src)

    def test_search_stage_stamp_is_written_exactly_once(self):
        # THE guard for the directive: only the SEARCH install may stamp
        # replace_search_stage. A second occurrence means the PE install
        # armed the maximize-then-pretend path.
        self.assertEqual(
            self._src().count("replace_search_stage = True"), 1)

    def test_pe_stage_stamp_is_written_exactly_once(self):
        self.assertEqual(self._src().count("replace_pe_stage = True"), 1)


if __name__ == "__main__":
    unittest.main()
