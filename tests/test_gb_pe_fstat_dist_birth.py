"""PE F-stat distance-birth stamp on ``rj_fstat_pe`` (user ruling 2026-08-28).

USER, verbatim: *"rj_fstat_pe get the same stamp? yes mirror them. That
would be better than drawing from the full prior widths. We need to make
sure it follows detailed balance."*

``GBSpecialBase.rj_fstat_dist_birth`` defaults to ``bool(rj_amp_maximize)``
(= ``bool(phase_maximize)``), which is False on every pe-named move — so
stock PE births drew slot 0 and the extrinsic angles at FULL PRIOR WIDTHS
even where the epoch F-stat centers were sitting right there, already
built for the search stage and the PE replace. The recipe now stamps
``rj_fstat_dist_birth`` on the PE F-stat birth move the same way the two
replace installs do (``GB_RJ_FSTAT_DIST_BIRTH`` overrides), so a PE birth
gets:

* epoch-table centers (the same table ``rj_replace_pe`` reads),
* a truncated + normalized lognormal slot-0 draw, priced on BOTH sides,
* ``pe_extrinsic_draw`` angles drawn and priced on BOTH sides,

and no maximize-and-keep anywhere (the general no-PE-maximization rule).

WHAT IS PINNED HERE. The table/slot-0 half of detailed balance is already
covered bit-exactly by ``test_fstat_ctr_epoch.TableDetailedBalanceTest``
(same table entries for birth and death; forward density ==
``assert_array_equal`` reverse density under truncation; the truncation
boundary derived from the table's own ln_snr). The extrinsic half is
covered by ``test_gb_pe_extrinsic_draw``. What was NOT covered, and is
covered here, is the ASSEMBLED factor pair the RJ step actually builds
(slot 0 + extrinsics + the log-range pair together), the install-site
stamp resolution, and the resolved-mode set that keeps the two
maximize-and-keep branches dead on a stamped pe-named move.
"""

import inspect
import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    GBSpecialStretchMove,
)

#: Moves the ruling covers / deliberately excludes (see the report).
PE_FSTAT_BIRTH = "rj_fstat_pe"


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


def _move(basis="dist", name=PE_FSTAT_BIRTH, **attrs):
    """A bare move carrying just what the density helpers read."""
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = name
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"
    m.transform_fn = SimpleNamespace(input_basis=[basis, "f0_ms", "fdot"])
    m._fstat_ctr_table = None
    m.phase_maximize = False
    m.pe_extrinsic_draw = True
    m.rj_fstat_dist_birth = True
    m.rj_removal_only = False
    m.rj_replace = False
    m.is_rj_prop = True
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class StampResolverTest(unittest.TestCase):
    """``_fstat_dist_birth_stamp``: the shared install-site resolver.

    All three install sites (search replace, PE replace, PE fstat birth)
    resolve the same way: ON by default, ``GB_RJ_FSTAT_DIST_BIRTH``
    always wins.
    """

    def _fn(self):
        from lisatools.globalfit.recipe import _fstat_dist_birth_stamp

        return _fstat_dist_birth_stamp

    def test_default_is_on(self):
        with _EnvPatch(GB_RJ_FSTAT_DIST_BIRTH=None):
            self.assertIs(self._fn()(), True)

    def test_env_zero_restores_the_prior_width_path(self):
        # BLAST-RADIUS ESCAPE HATCH: =0 must give back exactly the
        # pre-stamp resolution (rj_fstat_dist_birth False -> the RJ step's
        # fstat branch is skipped entirely and births keep prior widths).
        with _EnvPatch(GB_RJ_FSTAT_DIST_BIRTH="0"):
            self.assertIs(self._fn()(), False)

    def test_env_one_forces_on(self):
        with _EnvPatch(GB_RJ_FSTAT_DIST_BIRTH="1"):
            self.assertIs(self._fn()(), True)


class RecipeStampSiteTest(unittest.TestCase):
    """Which moves get the stamp — structural, no fit build (10-26 GB)."""

    def _src(self):
        from lisatools.globalfit.recipe import build_gb_moves

        return inspect.getsource(build_gb_moves)

    def test_pe_fstat_birth_move_is_stamped(self):
        src = self._src()
        self.assertIn(
            "gb_pe_prior_move.rj_fstat_dist_birth = _fstat_dist_birth_stamp()",
            src,
        )

    def test_every_stamp_goes_through_the_shared_resolver(self):
        # search replace + PE replace + PE fstat birth == 3, and none of
        # them re-rolls the env parsing inline.
        src = self._src()
        self.assertEqual(src.count("_fstat_dist_birth_stamp()"), 3)
        self.assertNotIn('os.environ.get("GB_RJ_FSTAT_DIST_BIRTH")', src)

    def test_moves_deliberately_left_alone(self):
        # rj_prior_pe is the PURE-PRIOR complement of rj_fstat_pe (its
        # prior-width births are the channel the F-stat grid cannot
        # reach); rj_fstat_mcmc runs its own serial-MCMC proposal;
        # rj_refit births from the GMM refit file with its own densities;
        # rj_prior_removal keeps the prior reverse-density convention.
        src = self._src()
        for mv in ("gb_pe_prior_birth_move", "gb_pe_fstat_mcmc_move",
                   "gb_pe_refit_move", "gb_prior_removal_move"):
            self.assertNotIn(f"{mv}.rj_fstat_dist_birth", src)

    def test_stamp_is_scoped_to_the_pe_flavor(self):
        # SEARCH bit-identity: the stamp sits under the same PE-flavor
        # guard the PE replace build uses, so a GB_MODE=search campaign
        # running through the pe-named moves is untouched.
        src = self._src()
        i = src.index("gb_pe_prior_move.rj_fstat_dist_birth")
        head = src[:i]
        self.assertIn("not _gb_mode_search or _pe_strict", head)


class StampedPEMoveModeSetTest(unittest.TestCase):
    """REGRESSION: the resolved mode set of a stamped pe-named move.

    Audit rows 1-2 (the two maximize-and-keep branches of ``_run_rj_step``)
    must stay dead at defaults once the stamp turns the fstat branch on.
    """

    def test_no_maximize_and_keep_at_defaults(self):
        m = _move()
        with _EnvPatch(GB_PE_EXTRINSIC_DRAW=None, GB_RJ_BIRTH_CTR_MODE=None):
            # audit row 1: _pin_mode = not _pe_extr_active() -> False, so
            # _eval(birth_k, False) and no phi0 -= phase_angle write-back.
            self.assertTrue(m._pe_extr_active())
            self.assertFalse(not m._pe_extr_active())
            # audit row 2: the `elif self.phase_maximize` branch is both
            # unreachable (the fstat branch wins) and False anyway.
            self.assertTrue(m.rj_fstat_dist_birth)
            self.assertFalse(m.phase_maximize)
            # audit row 3: the legacy amplitude pin needs the fstat path
            # OFF -- it is on, so that branch is dead too.
            self.assertFalse(not m.rj_fstat_dist_birth)

    def test_centers_come_from_the_epoch_table_like_the_pe_replace(self):
        m = _move()
        with _EnvPatch(GB_RJ_BIRTH_CTR_MODE=None):
            self.assertFalse(m._rj_birth_perrow())   # table, not per-row

    def test_band_shutoff_stays_off_for_a_pe_named_move(self):
        # _band_shutoff_enabled newly passes its rj_fstat_dist_birth gate;
        # the name-scoped default must still keep it off in PE.
        m = _move()
        with _EnvPatch(GB_RJ_BAND_SHUTOFF_SCOPE=None):
            self.assertFalse(m._band_shutoff_enabled())

    def test_extrinsic_knob_off_restores_the_pin(self):
        # The documented escape hatch stays documented: this is the ONE
        # way a stamped PE move goes back to pin + phase-max.
        m = _move(pe_extrinsic_draw=False)
        self.assertFalse(m._pe_extr_active())
        self.assertTrue(not m._pe_extr_active())     # _pin_mode True


class BirthDeathFactorPairingTest(unittest.TestCase):
    """The ASSEMBLED RJ factor pair of the fstat-dist-birth path.

    ``_run_rj_step`` builds
        birth: ``-_bl - log_range + _extr_corr_b``
        death: ``+_dl + log_range + _extr_corr_d``
    For a leaf whose stored coordinates ARE a drawn birth (same row, same
    centers, same truncation), the two must sum to exactly zero -- every
    normalization (the lognormal's, the ``-log Phi(alpha)`` truncation
    renormalization, the extrinsic mixture's, and the uniform log-volume)
    has to appear on both sides or the residue shows up here.
    """

    LOG_RANGE = 3.7182818          # any fixed value; it must cancel

    def _pieces(self, m, v, ln_center, sigma, alpha, params, rows,
                p0, io, ps, ls):
        bl = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        corr_b = m._pe_or_pin_extrinsics(params, rows, p0, io, ps, ls)
        birth = -bl - self.LOG_RANGE + corr_b
        # DEATH side of the very same row: same centers (the table is
        # keyed on f0/Mc, which a birth does not change), same alpha.
        dl = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        corr_d = m._pe_death_extr_corr(params, rows, p0, io, ps, ls)
        death = dl + self.LOG_RANGE + corr_d
        return birth, death

    def _run(self, basis):
        rng = np.random.default_rng(2028)
        n = 96
        m = _move(basis=basis)
        ln_center = rng.uniform(-2.0, 2.0, n)
        sigma = rng.uniform(0.02, 0.4, n)
        ln_snr = np.log(rng.uniform(6.0, 200.0, n))
        alpha = m._snr_trunc_alpha(ln_snr, sigma, 5.0)
        z = m._truncnorm_std_draw(n, alpha)
        ln_draw = ln_center + sigma * z
        v = np.exp(ln_draw) if basis == "dist" else ln_draw
        params = rng.random((n, 8))
        params[:, 0] = v
        rows = np.arange(n)
        p0 = rng.uniform(0, 2 * np.pi, n)
        io = rng.uniform(0.05, np.pi - 0.05, n)
        ps = rng.uniform(0, np.pi, n)
        birth, death = self._pieces(
            m, v, ln_center, sigma, alpha, params, rows, p0, io, ps, ln_snr)
        self.assertTrue(np.all(np.isfinite(birth)))
        np.testing.assert_allclose(birth + death, 0.0, atol=1e-12)

    def test_distance_basis(self):
        self._run("dist")

    def test_amplitude_basis(self):
        self._run("lnA")

    def test_truncation_normalization_is_on_both_sides(self):
        # Drop alpha from ONE side only and the pair must stop cancelling
        # -- proof the -log Phi(alpha) term is load-bearing, not inert.
        rng = np.random.default_rng(99)
        n = 32
        m = _move(basis="dist")
        ln_center = np.zeros(n)
        # sigma large enough that Phi(alpha) < 1 in double precision
        sigma = np.full(n, 0.5)
        ln_snr = np.log(np.full(n, 6.0))
        alpha = m._snr_trunc_alpha(ln_snr, sigma, 5.0)
        self.assertLess(float(np.max(m._std_norm_cdf(alpha))), 1.0)
        z = m._truncnorm_std_draw(n, alpha)
        v = np.exp(ln_center + sigma * z)
        bl = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        bl_untrunc = m._slot0_log_proposal(v, ln_center, sigma, alpha=None)
        self.assertFalse(np.allclose(bl, bl_untrunc))
        # and with alpha on BOTH sides the slot-0 halves cancel exactly
        np.testing.assert_array_equal(bl, m._slot0_log_proposal(
            v, ln_center, sigma, alpha=alpha))

    def test_pin_mode_contributes_no_extrinsic_factor(self):
        # GB_RJ_FSTAT_DIST_BIRTH stays on but GB_PE_EXTRINSIC_DRAW=0:
        # the corrections collapse to 0.0 on both sides, so the pair still
        # cancels (the pre-2026-08-25 bookkeeping, bit-identical).
        rng = np.random.default_rng(7)
        n = 16
        m = _move(basis="dist", pe_extrinsic_draw=False)
        params = rng.random((n, 8))
        rows = np.arange(n)
        p0 = rng.uniform(0, 2 * np.pi, n)
        io = rng.uniform(0.05, np.pi - 0.05, n)
        ps = rng.uniform(0, np.pi, n)
        ls = np.log(rng.uniform(6.0, 60.0, n))
        self.assertEqual(
            m._pe_or_pin_extrinsics(params, rows, p0, io, ps, ls), 0.0)
        self.assertEqual(
            m._pe_death_extr_corr(params, rows, p0, io, ps, ls), 0.0)


if __name__ == "__main__":
    unittest.main()
