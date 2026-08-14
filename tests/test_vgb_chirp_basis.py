"""VGB chirp-mass basis (VGB_CHIRP_MASS_BASIS): transform math + init spread.

The 2026-08-14 ruling makes ``[dist, phi0, cos_iota, psi, Mc,
fdot_astro_ratio]`` the opt-in VGB sampled basis (6-mo run only; default
stays the legacy 5-dim distance basis so live production stores resume).
These tests pin (a) the derived-fdot math against the GB 9-column formula,
(b) the additive ratio-init exception that prevents the zero-truth /
multiplicative-init / pure-stretch collapse, and (c) the flag-off legacy
shapes staying bit-identical.
"""

import copy
import os
import pickle
import unittest

import numpy as np

from lisatools.globalfit.run import seed_injection_coords
from lisatools.globalfit.stock.erebor.transforms import (
    gb_amp_from_dist,
    make_gb_transform_container,
)
from lisatools.globalfit.stock.erebor.vgb import (
    VGB_FIXED_BASIS_CHIRP,
    VGB_FIXED_BASIS_DIST,
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
    VGBSettings,
    vgb_fixed_basis,
    vgb_sampled_basis,
)

from .test_stock_globalfit import _EnvGuard


class VGBChirpTransformTest(unittest.TestCase):
    def test_fdot_matches_gb_formula(self):
        """Per-leaf-fill chirp basis reproduces fdot_gr(f0, Mc)*(1+r)."""
        from gbgpu.utils.utility import get_fdot

        # two leaves with distinct fixed (f0 [mHz], alpha, sin_delta)
        fixed = np.array([[2.6, 1.2, 0.3], [4.1, 2.0, -0.5]])
        fill_list = [
            {name: fixed[leaf, j] for j, name in enumerate(VGB_FIXED_BASIS_CHIRP)}
            for leaf in range(fixed.shape[0])
        ]
        tc = make_gb_transform_container(
            use_chirp_mass=True,
            use_fdot_astro=True,
            use_distance=True,
            input_basis=list(VGB_SAMPLED_BASIS_CHIRP),
            fill_dict=fill_list,
            mc_lims=(0.001, 1.0),
        )
        # sampled rows: [dist (kpc), phi0, cos_iota, psi, Mc, ratio]
        rows = np.array([
            [1.5, 0.7, 0.2, 0.9, 0.45, 0.0],
            [8.0, 2.1, -0.4, 1.4, 0.30, -1.7],  # interacting: fdot < 0
        ])
        leaf_inds = np.array([0, 1])
        out = tc.both_transforms(rows, leaf_inds=leaf_inds)
        # output basis: [A, f0, fdot, fddot, phi0, cos_iota, psi, alpha, sin_delta]
        f0_hz = fixed[:, 0] * 1e-3
        mc = rows[:, VGB_SAMPLED_BASIS_CHIRP.index("Mc")]
        ratio = rows[:, VGB_SAMPLED_BASIS_CHIRP.index("fdot_astro_ratio")]
        expected_fdot = get_fdot(f=f0_hz, Mc=mc) * (1.0 + ratio)
        np.testing.assert_allclose(out[:, 2], expected_fdot, rtol=1e-13)
        np.testing.assert_array_equal(out[:, 3], 0.0)  # fddot exactly 0
        np.testing.assert_allclose(out[:, 1], f0_hz, rtol=1e-15)
        # amplitude derived from the sampled (dist, Mc) + per-leaf f0
        np.testing.assert_allclose(
            out[:, 0], gb_amp_from_dist(f0_hz, mc, rows[:, 0]), rtol=1e-13
        )
        # sign check: the interacting row must come out with fdot < 0
        self.assertLess(out[1, 2], 0.0)

    def test_physical_round_trip(self):
        """forward(inverse(forward(x))) == forward(x) (mirror convention).

        The inverse's (Mc, r) split is a CONVENTION (Mc from |fdot|), so
        the coordinate round trip is not the identity; the physical
        (A, f0, fdot, fddot) must be reproduced exactly either way.
        """
        fixed = np.array([[2.6, 1.2, 0.3]])
        fill_list = [
            {name: fixed[0, j] for j, name in enumerate(VGB_FIXED_BASIS_CHIRP)}
        ]
        tc = make_gb_transform_container(
            use_chirp_mass=True,
            use_fdot_astro=True,
            use_distance=True,
            input_basis=list(VGB_SAMPLED_BASIS_CHIRP),
            fill_dict=fill_list,
            mc_lims=(0.001, 1.0),
        )
        rows = np.array([[1.5, 0.7, 0.2, 0.9, 0.45, 0.25]])
        leaf_inds = np.array([0])
        out = tc.both_transforms(rows, leaf_inds=leaf_inds)
        back = np.atleast_2d(tc.both_inverse_transforms(out))
        self.assertEqual(back.shape, rows.shape)  # input-basis width (6)
        out2 = tc.both_transforms(back, leaf_inds=leaf_inds)
        np.testing.assert_allclose(out2, out, rtol=1e-12, atol=1e-300)


class VGBInitSpreadTest(unittest.TestCase):
    def test_additive_ratio_jitter_prevents_collapse(self):
        """Both Mc and ratio columns get nonzero walker spread at init.

        The ratio truth is exactly 0 (catalogue fdots are exactly
        fdot_GR(f0, Mc)), so multiplicative init would freeze that
        dimension for the pure stretch move; the additive exception must
        give it spread while Mc's nonzero truth spreads multiplicatively.
        """
        np.random.seed(7)
        sampled = list(VGB_SAMPLED_BASIS_CHIRP)
        nleaves = 3
        inj = np.zeros((nleaves, len(sampled)))
        inj[:, sampled.index("dist")] = [1.5, 3.0, 8.0]
        inj[:, sampled.index("phi0")] = 0.7
        inj[:, sampled.index("cos_iota")] = 0.2
        inj[:, sampled.index("psi")] = 0.9
        inj[:, sampled.index("Mc")] = [0.45, 0.30, 0.65]
        inj[:, sampled.index("fdot_astro_ratio")] = 0.0  # exact zero truth
        factor = 1e-5
        widths = {sampled.index("fdot_astro_ratio"): 0.02 * 5.0}
        coords = seed_injection_coords(inj, factor, 2, 16, additive_start_widths=widths)
        self.assertEqual(coords.shape, (2, 16, nleaves, len(sampled)))
        mc_std = coords[..., sampled.index("Mc")].std(axis=1)
        r_std = coords[..., sampled.index("fdot_astro_ratio")].std(axis=1)
        self.assertTrue((mc_std > 0).all(), "Mc walker spread collapsed")
        self.assertTrue((r_std > 0).all(), "ratio walker spread collapsed")
        # the additive width actually applies: spread ~ factor * width
        self.assertGreater(r_std.mean(), factor * widths[5] * 0.1)
        # truth-null convention: factor = 0 -> exact injection everywhere
        exact = seed_injection_coords(inj, 0.0, 1, 4, additive_start_widths=widths)
        np.testing.assert_array_equal(exact, np.broadcast_to(inj, exact.shape))


class VGBSettingsBasisTest(unittest.TestCase):
    def test_flag_off_legacy_shapes(self):
        """Default = legacy 5-dim distance basis, bit-unchanged constants."""
        with _EnvGuard(VGB_CHIRP_MASS_BASIS=None, VGB_SAMPLE_DISTANCE=None):
            s = VGBSettings()
            self.assertFalse(s.chirp_mass_basis)
            self.assertEqual(s.ndim, 5)
            self.assertIsNone(s.additive_start_widths)
            self.assertEqual(
                vgb_sampled_basis(s),
                ["dist", "phi0", "cos_iota", "psi", "fdot_astro_ratio"],
            )
            self.assertEqual(
                vgb_fixed_basis(s), ["f0", "alpha", "sin_delta", "Mc"]
            )
            pickle.loads(pickle.dumps(copy.deepcopy(s)))

    def test_flag_on_chirp_shapes(self):
        with _EnvGuard(VGB_CHIRP_MASS_BASIS="1", VGB_SAMPLE_DISTANCE=None):
            s = VGBSettings()
            self.assertTrue(s.chirp_mass_basis)
            self.assertEqual(s.ndim, 6)
            self.assertEqual(
                vgb_sampled_basis(s),
                ["dist", "phi0", "cos_iota", "psi", "Mc", "fdot_astro_ratio"],
            )
            self.assertEqual(vgb_fixed_basis(s), ["f0", "alpha", "sin_delta"])
            pickle.loads(pickle.dumps(copy.deepcopy(s)))

    def test_constants_pinned(self):
        # legacy constants are what live 5-dim production stores resume on
        self.assertEqual(
            VGB_SAMPLED_BASIS_DIST,
            ["dist", "phi0", "cos_iota", "psi", "fdot_astro_ratio"],
        )
        self.assertEqual(VGB_FIXED_BASIS_DIST, ["f0", "alpha", "sin_delta", "Mc"])
        self.assertEqual(
            VGB_SAMPLED_BASIS_CHIRP,
            ["dist", "phi0", "cos_iota", "psi", "Mc", "fdot_astro_ratio"],
        )
        self.assertEqual(VGB_FIXED_BASIS_CHIRP, ["f0", "alpha", "sin_delta"])


if __name__ == "__main__":
    unittest.main()
