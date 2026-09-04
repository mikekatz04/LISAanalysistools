"""Unit tests for the cold->hottest-rung search reseed on ModuleSubState.

The reseed (user request 2026-09-04, the v9 run) literally copies the cold
rung's per-walker state into the highest-temperature rung, walker-permuted, as
a search-only seeding device. These tests pin the exact copy semantics without
instantiating the GB move or a GPU: they operate on a bare ModuleSubState whose
tempered block is allocated by ``initialize_tempered``.
"""

import unittest

import numpy as np

from lisatools.globalfit.state import ModuleSubState


def _fill(sub):
    """Distinctive, invertible values so a mis-indexed copy is visible."""
    nt, nw, nl, nd = sub.ntemps, sub.nwalkers, sub.nleaves_max, sub.ndim
    t = np.arange(nt)[:, None, None, None]
    w = np.arange(nw)[None, :, None, None]
    l = np.arange(nl)[None, None, :, None]
    d = np.arange(nd)[None, None, None, :]
    sub.coords[:] = 1000 * t + 100 * w + 10 * l + d
    # inds: a per-(t,w) parity pattern over leaves
    tt = np.arange(nt)[:, None, None]
    ww = np.arange(nw)[None, :, None]
    ll = np.arange(nl)[None, None, :]
    sub.inds[:] = ((tt + ww + ll) % 2).astype(bool)
    ll2 = np.arange(nt)[:, None] * 10 + np.arange(nw)[None, :]
    sub.log_like[:] = ll2.astype(float)
    sub.log_prior[:] = -ll2.astype(float)
    sub.betas[:] = np.linspace(1.0, 1e-4, nt)
    sub.in_model_proposed[:] = np.arange(nt) + 1
    sub.in_model_accepted[:] = np.arange(nt)
    sub.d_h[:] = 7.0
    sub.h_h[:] = 9.0


class ReseedColdIntoHottestTest(unittest.TestCase):
    def _make(self, ntemps=4, nwalkers=3, nleaves_max=2, ndim=2):
        sub = ModuleSubState()
        sub.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim)
        _fill(sub)
        return sub

    def test_hottest_rung_becomes_permuted_cold(self):
        sub = self._make()
        perm = np.array([2, 0, 1])
        cold0 = sub.coords[0].copy()
        ll0 = sub.log_like[0].copy()
        lp0 = sub.log_prior[0].copy()
        inds0 = sub.inds[0].copy()

        sub.reseed_cold_into_hottest(perm=perm)

        hot = sub.ntemps - 1
        np.testing.assert_array_equal(sub.coords[hot], cold0[perm])
        np.testing.assert_array_equal(sub.log_like[hot], ll0[perm])
        np.testing.assert_array_equal(sub.log_prior[hot], lp0[perm])
        np.testing.assert_array_equal(sub.inds[hot], inds0[perm])

    def test_cold_and_middle_rungs_unchanged(self):
        sub = self._make()
        before = {k: getattr(sub, k).copy()
                  for k in ("coords", "inds", "log_like", "log_prior")}
        perm = np.array([1, 2, 0])
        sub.reseed_cold_into_hottest(perm=perm)
        hot = sub.ntemps - 1
        for k, arr in before.items():
            for t in range(sub.ntemps):
                if t == hot:
                    continue
                np.testing.assert_array_equal(
                    getattr(sub, k)[t], arr[t],
                    err_msg=f"rung {t} of {k} changed but must not")

    def test_cold_independent_after_reseed(self):
        # copy semantics: mutating the hot rung must not bleed into cold
        sub = self._make()
        sub.reseed_cold_into_hottest(perm=np.array([0, 1, 2]))
        hot = sub.ntemps - 1
        cold_before = sub.coords[0].copy()
        sub.coords[hot] += 123.0
        np.testing.assert_array_equal(sub.coords[0], cold_before)

    def test_ladder_and_counters_untouched(self):
        sub = self._make()
        betas0 = sub.betas.copy()
        imp0 = sub.in_model_proposed.copy()
        ima0 = sub.in_model_accepted.copy()
        dh0 = sub.d_h.copy()
        hh0 = sub.h_h.copy()
        sub.reseed_cold_into_hottest(perm=np.array([2, 1, 0]))
        np.testing.assert_array_equal(sub.betas, betas0)
        np.testing.assert_array_equal(sub.in_model_proposed, imp0)
        np.testing.assert_array_equal(sub.in_model_accepted, ima0)
        np.testing.assert_array_equal(sub.d_h, dh0)
        np.testing.assert_array_equal(sub.h_h, hh0)

    def test_identity_permutation_clones_cold(self):
        sub = self._make()
        cold0 = sub.coords[0].copy()
        sub.reseed_cold_into_hottest(perm=np.arange(sub.nwalkers))
        np.testing.assert_array_equal(sub.coords[sub.ntemps - 1], cold0)

    def test_rng_draws_a_valid_permutation(self):
        sub = self._make()
        cold0 = sub.coords[0].copy()
        sub.reseed_cold_into_hottest(rng=np.random.default_rng(0))
        hot = sub.ntemps - 1
        # every hot-slot walker must equal SOME cold walker (a permutation),
        # and each cold walker used exactly once
        matches = [
            int(np.where((cold0 == sub.coords[hot][h]).all(axis=(-2, -1)))[0][0])
            for h in range(sub.nwalkers)
        ]
        self.assertEqual(sorted(matches), list(range(sub.nwalkers)))

    def test_single_rung_ladder_is_noop(self):
        sub = self._make(ntemps=1)
        c0 = sub.coords.copy()
        sub.reseed_cold_into_hottest(perm=np.array([0, 1, 2]))
        np.testing.assert_array_equal(sub.coords, c0)


if __name__ == "__main__":
    unittest.main()
