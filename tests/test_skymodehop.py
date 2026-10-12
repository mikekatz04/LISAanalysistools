"""Sky-mode hop move (``lisatools.sampling.moves.skymodehop``).

* LISA-frame hops (legacy behaviour) are unchanged;
* the SSB-ecliptic and ICRS paths are exact round trips and agree with
  each other through the analytic ICRS <-> ecliptic rotation;
* the frame Jacobian carried in the proposal factors is small, consistent
  between the forward and inverse maps, and antisymmetric under the inverse
  hop;
* :class:`ConstellationPhase` reproduces the equal-arm orbit phase, rotates
  ICRS-frame positions, extrapolates linearly and pickles;
* PRODUCTION-PATH check: every one of the 8 hopped modes reproduces the
  LISA Analysis Tools time-domain TDI response of a 3 mHz burst (overlap >
  0.98, no residual time shift) on the stock equal-arm orbits, and a move
  whose time transform is disabled is caught by the same check.
"""

import pickle
import unittest
import warnings

import numpy as np

from lisatools.sampling.moves.skymodehop import ConstellationPhase, SkyMove
from lisatools.utils.constants import YRSID_SI

IND_MAP_ICRS = dict(cosinc=6, psi=7, lam=8, sinbeta=9, t_ref=10)  # the stock MBH basis
IND_MAP_LEGACY = dict(cosinc=6, lam=7, sinbeta=8, psi=9, t_ref=10)
NDIM = 11

_ORBITS = {}


def _orbits(frame="ecliptic", t_lo=0.0, t_hi=2.0 * YRSID_SI, dt=3600.0):
    """Stock equal-arm orbits on a COARSE configured grid (the default dense
    50-s grid over the 5-yr file costs minutes to build; the hourly grid is
    linear-interpolation exact to ~1 m here). Cached per grid."""
    from lisatools.detector import EqualArmlengthOrbits

    key = (frame, t_lo, t_hi, dt)
    if key not in _ORBITS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ORBITS[key] = EqualArmlengthOrbits(
                force_backend="cpu", frame=frame, linear_interp_setup=False,
                t_arr=np.arange(t_lo, t_hi, dt),
            )
    return _ORBITS[key]


class _FixedRandom:
    """``randint`` stub: ``hi == 2`` draws the lat flags, ``hi == 4`` the k's."""

    def __init__(self, lat_flags=None, ks=None):
        self.lat_flags = lat_flags
        self.ks = ks

    def randint(self, lo, hi, size=None):
        if hi == 2:
            return np.resize(np.asarray(self.lat_flags, dtype=int), size)
        if hi == 4:
            return np.resize(np.asarray(self.ks, dtype=int), size)
        raise AssertionError(hi)


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


def _random_coords(n, ind_map, rng, t_range=(0.1 * YRSID_SI, 1.5 * YRSID_SI)):
    c = rng.uniform(0, 1, (n, NDIM))
    c[:, ind_map["cosinc"]] = rng.uniform(-0.95, 0.95, n)
    c[:, ind_map["lam"]] = rng.uniform(0, 2 * np.pi, n)
    c[:, ind_map["sinbeta"]] = rng.uniform(-0.95, 0.95, n)
    c[:, ind_map["psi"]] = rng.uniform(0, np.pi, n)
    c[:, ind_map["t_ref"]] = rng.uniform(*t_range, n)
    return c


def _assert_same_point(test, a, b, ind_map, atol=1e-8, t_atol=1e-3):
    test.assertLessEqual(np.abs(_wrap(a[:, ind_map["lam"]] - b[:, ind_map["lam"]], 2 * np.pi)).max(), atol)
    test.assertLessEqual(np.abs(a[:, ind_map["sinbeta"]] - b[:, ind_map["sinbeta"]]).max(), atol)
    test.assertLessEqual(np.abs(_wrap(a[:, ind_map["psi"]] - b[:, ind_map["psi"]], np.pi)).max(), atol)
    test.assertLessEqual(np.abs(a[:, ind_map["cosinc"]] - b[:, ind_map["cosinc"]]).max(), atol)
    test.assertLessEqual(np.abs(a[:, ind_map["t_ref"]] - b[:, ind_map["t_ref"]]).max(), t_atol)
    others = [i for i in range(NDIM) if i not in ind_map.values()]
    np.testing.assert_allclose(a[:, others], b[:, others])


# ---------------------------------------------------------------------------
class LisaFrameHopTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.coords = _random_coords(16, IND_MAP_LEGACY, self.rng)

    def test_lat_flips_and_reflects(self):
        move = SkyMove(which="lat")
        out = move.lat_transform(self.coords, None)
        im = IND_MAP_LEGACY
        np.testing.assert_allclose(out[:, im["sinbeta"]], -self.coords[:, im["sinbeta"]])
        np.testing.assert_allclose(out[:, im["cosinc"]], -self.coords[:, im["cosinc"]])
        np.testing.assert_allclose(
            _wrap(out[:, im["psi"]] - (np.pi - self.coords[:, im["psi"]]), np.pi), 0, atol=1e-12
        )
        self.assertFalse(np.shares_memory(out, self.coords))
        # involution
        _assert_same_point(self, move.lat_transform(out, None), self.coords, im, atol=1e-12, t_atol=0.0)

    def test_long_rotates_by_k_pi_over_2(self):
        move = SkyMove(which="long")
        ks = np.arange(16) % 4
        out = move.long_transform(self.coords, _FixedRandom(ks=ks))
        im = IND_MAP_LEGACY
        np.testing.assert_allclose(
            _wrap(out[:, im["lam"]] - self.coords[:, im["lam"]] - ks * np.pi / 2, 2 * np.pi), 0, atol=1e-12
        )
        np.testing.assert_allclose(
            _wrap(out[:, im["psi"]] - self.coords[:, im["psi"]] - ks * np.pi / 2, np.pi), 0, atol=1e-12
        )
        self.assertTrue(np.all(out[:, im["lam"]] >= 0) and np.all(out[:, im["lam"]] < 2 * np.pi))
        self.assertTrue(np.all(out[:, im["psi"]] >= 0) and np.all(out[:, im["psi"]] < np.pi))

    def test_both_does_not_mutate_its_input(self):
        move = SkyMove(which="both")
        before = self.coords.copy()
        move.both_transform(self.coords, _FixedRandom(lat_flags=[1, 0], ks=[1, 2, 3, 0]))
        np.testing.assert_array_equal(self.coords, before)

    def test_get_proposal_shapes_inds_and_zero_factors(self):
        move = SkyMove(which="both")
        coords = self.coords.reshape(2, 4, 2, NDIM)
        inds = np.ones((2, 4, 2), dtype=bool)
        inds[0, 1, 1] = False
        q, factors = move.get_proposal({"mbh": coords}, _FixedRandom(lat_flags=[1], ks=[1]), branches_inds={"mbh": inds})
        self.assertEqual(q["mbh"].shape, coords.shape)
        self.assertEqual(factors.shape, (2, 4))
        np.testing.assert_array_equal(factors, 0.0)
        np.testing.assert_array_equal(q["mbh"][0, 1, 1], coords[0, 1, 1])  # inactive leaf untouched
        self.assertFalse(np.allclose(q["mbh"][0, 1, 0], coords[0, 1, 0]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            SkyMove(which="nope")
        with self.assertRaises(ValueError):
            SkyMove(coord_frame="galactic")
        with self.assertRaises(ValueError):
            SkyMove(coord_frame="icrs", ind_map=dict(cosinc=6, lam=7, sinbeta=8, psi=9))
        with self.assertRaises(ValueError):
            SkyMove(coord_frame="icrs", constellation_phase=3.0)


# ---------------------------------------------------------------------------
class FrameRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orbits = _orbits()
        cls.phase = ConstellationPhase.from_orbits(cls.orbits, t_offset=0.0)
        cls.rng = np.random.default_rng(2)

    def _moves(self, frame, which="both"):
        # analytic phase and the orbit table
        return [
            SkyMove(ind_map=IND_MAP_ICRS, which=which, coord_frame=frame),
            SkyMove(ind_map=IND_MAP_ICRS, which=which, coord_frame=frame, constellation_phase=self.phase),
        ]

    def test_frame_conversion_round_trips(self):
        coords = _random_coords(64, IND_MAP_ICRS, self.rng)
        for frame in ("ssb_ecliptic", "icrs"):
            for move in self._moves(frame):
                with self.subTest(frame=frame, table=move.constellation_phase is not None):
                    back = move._from_lisa_frame(move._to_lisa_frame(coords))
                    _assert_same_point(self, back, coords, IND_MAP_ICRS, atol=1e-9, t_atol=1e-4)

    def test_lat_hop_is_an_involution_in_every_frame(self):
        coords = _random_coords(32, IND_MAP_ICRS, self.rng).reshape(2, 16, 1, NDIM)
        for frame in ("lisa", "ssb_ecliptic", "icrs"):
            for move in self._moves(frame, "lat"):
                q, _ = move.get_proposal({"mbh": coords}, _FixedRandom())
                q2, _ = move.get_proposal(q, _FixedRandom())
                _assert_same_point(
                    self, q2["mbh"].reshape(-1, NDIM), coords.reshape(-1, NDIM), IND_MAP_ICRS, atol=1e-8, t_atol=1e-3
                )

    def test_hop_moves_the_ssb_time_but_not_the_lisa_time(self):
        coords = _random_coords(32, IND_MAP_ICRS, self.rng)
        move = SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="ssb_ecliptic", constellation_phase=self.phase)
        lisa_before = move._to_lisa_frame(coords)
        hopped = move._from_lisa_frame(move.lat_transform(lisa_before, None))
        lisa_after = move._to_lisa_frame(hopped)
        t = IND_MAP_ICRS["t_ref"]
        self.assertLess(np.abs(lisa_after[:, t] - lisa_before[:, t]).max(), 1e-3)
        # the SSB time DOES move (by up to ~2 R/c)
        dt = np.abs(hopped[:, t] - coords[:, t])
        self.assertGreater(dt.max(), 100.0)
        self.assertLess(dt.max(), 2 * 500.0 + 1.0)

    def test_icrs_path_equals_the_ecliptic_path_conjugated(self):
        from lisatools.sources.utils import ecliptic_to_icrs, icrs_to_ecliptic

        im = IND_MAP_ICRS
        coords_ecl = _random_coords(32, im, self.rng)
        coords_icrs = coords_ecl.copy()
        ra, dec, psi_i = ecliptic_to_icrs(
            coords_ecl[:, im["lam"]], np.arcsin(coords_ecl[:, im["sinbeta"]]), coords_ecl[:, im["psi"]]
        )
        coords_icrs[:, im["lam"]], coords_icrs[:, im["sinbeta"]], coords_icrs[:, im["psi"]] = ra, np.sin(dec), psi_i
        rnd = dict(lat_flags=np.arange(32) % 2, ks=np.arange(32) % 4)
        m_ecl = SkyMove(ind_map=im, which="both", coord_frame="ssb_ecliptic", constellation_phase=self.phase)
        m_icrs = SkyMove(ind_map=im, which="both", coord_frame="icrs", constellation_phase=self.phase)
        q_ecl, f_ecl = m_ecl.get_proposal({"mbh": coords_ecl.reshape(1, 32, 1, NDIM)}, _FixedRandom(**rnd))
        q_icrs, f_icrs = m_icrs.get_proposal({"mbh": coords_icrs.reshape(1, 32, 1, NDIM)}, _FixedRandom(**rnd))
        out_icrs = q_icrs["mbh"].reshape(32, NDIM)
        lam, beta, psi = icrs_to_ecliptic(out_icrs[:, im["lam"]], np.arcsin(out_icrs[:, im["sinbeta"]]), out_icrs[:, im["psi"]])
        back = out_icrs.copy()
        back[:, im["lam"]], back[:, im["sinbeta"]], back[:, im["psi"]] = lam, np.sin(beta), psi
        _assert_same_point(self, back, q_ecl["mbh"].reshape(32, NDIM), im, atol=1e-8, t_atol=1e-3)
        # the ICRS<->ecliptic rotation is a rigid sky rotation: same Jacobian factors
        np.testing.assert_allclose(f_icrs, f_ecl, atol=1e-6)


# ---------------------------------------------------------------------------
class JacobianTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.coords = _random_coords(64, IND_MAP_ICRS, self.rng)

    def test_small_and_consistent_between_forward_and_inverse(self):
        for frame in ("ssb_ecliptic", "icrs"):
            move = SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame=frame, frame_t0=0.21)
            fwd = move._log_abs_det_jacobian(self.coords)
            self.assertLess(np.abs(fwd).max(), 2e-3)  # 1 + O(Omega R / c)
            self.assertGreater(np.abs(fwd).max(), 1e-6)  # ... but not identically zero
            y = move._to_lisa_frame(self.coords)
            inv = move._log_abs_det_jacobian(y, fn=move._from_lisa_frame)
            self.assertLess(np.abs(fwd + inv).max(), 1e-6)

    def test_factors_are_antisymmetric_under_the_inverse_hop(self):
        move = SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="icrs")
        coords = self.coords.reshape(2, 32, 1, NDIM)
        q, f1 = move.get_proposal({"mbh": coords}, _FixedRandom())
        _, f2 = move.get_proposal(q, _FixedRandom())
        self.assertEqual(f1.shape, (2, 32))
        self.assertGreater(np.abs(f1).max(), 0.0)
        np.testing.assert_allclose(f1 + f2, 0.0, atol=1e-6)

    def test_jacobian_off_and_lisa_frame_give_zero(self):
        coords = self.coords.reshape(2, 32, 1, NDIM)
        for move in (
            SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="icrs", jacobian=False),
            SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="lisa"),
        ):
            _, f = move.get_proposal({"mbh": coords}, _FixedRandom())
            np.testing.assert_array_equal(f, 0.0)

    def test_factors_sum_over_active_leaves(self):
        move = SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="icrs")
        coords = self.coords.reshape(1, 16, 4, NDIM)
        inds = np.ones((1, 16, 4), dtype=bool)
        inds[0, :, 2:] = False
        _, f = move.get_proposal({"mbh": coords}, _FixedRandom(), branches_inds={"mbh": inds})
        per_leaf = [
            move.get_proposal({"mbh": coords[:, :, j : j + 1]}, _FixedRandom())[1] for j in range(2)
        ]
        np.testing.assert_allclose(f, per_leaf[0] + per_leaf[1], atol=1e-12)


# ---------------------------------------------------------------------------
class ConstellationPhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orbits = _orbits()
        cls.orbits_icrs_label = _orbits(frame="icrs")

    def test_equal_arm_phase_is_omega_t(self):
        from bbhx.utils.constants import ConstOmega

        ph = ConstellationPhase.from_orbits(self.orbits)
        t = np.linspace(1e4, 1.9 * YRSID_SI, 50)
        self.assertLess(np.abs(_wrap(ph(t) - ConstOmega * t, 2 * np.pi)).max(), 1e-3)
        self.assertTrue(np.all(np.diff(ph(t)) > 0))  # unwrapped

    def test_icrs_frame_positions_are_rotated(self):
        from lisatools.sampling.moves.skymodehop import _icrs_to_ecliptic_cartesian
        from lisatools.utils.constants import EPS_RAD

        # the same (ecliptic) file labelled ICRS: the builder applies R_x(eps)
        a = ConstellationPhase.from_orbits(self.orbits)
        b = ConstellationPhase.from_orbits(self.orbits_icrs_label)
        t = np.array([0.3, 0.6]) * YRSID_SI
        self.assertGreater(np.abs(_wrap(a(t) - b(t), 2 * np.pi)).max(), 1e-3)
        v = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(_icrs_to_ecliptic_cartesian(v), [[0.0, np.cos(EPS_RAD), -np.sin(EPS_RAD)]])

    def test_offset_extrapolation_and_pickle(self):
        ph = ConstellationPhase(np.array([0.0, 10.0, 20.0]), np.array([0.0, 1.0, 2.0]), t_offset=5.0)
        np.testing.assert_allclose(ph(np.array([-15.0, 0.0, 25.0])), [-1.0, 0.5, 3.0])
        ph2 = pickle.loads(pickle.dumps(ph))
        np.testing.assert_allclose(ph2(np.array([-15.0, 25.0])), ph(np.array([-15.0, 25.0])))
        with self.assertRaises(ValueError):
            ConstellationPhase(np.array([0.0, 0.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
class ResponseSymmetryTest(unittest.TestCase):
    """Production path: a 3 mHz SSB-anchored burst through pyResponseTDI on
    the equal-arm orbits reproduces itself at every hopped mode."""

    F0 = 3e-3
    DT = 5.0
    SIG = 3600.0
    NPTS = int(0.5 * 86400 / 5.0)
    T_BUFFER = 20000.0

    @classmethod
    def setUpClass(cls):
        from lisatools.response.directresponse import pyResponseTDI

        rng = np.random.default_rng(3)
        cls.params = dict(
            lam=rng.uniform(0, 2 * np.pi), beta=np.arcsin(rng.uniform(-0.9, 0.9)),
            psi=rng.uniform(0, np.pi), inc=np.arccos(rng.uniform(-0.9, 0.9)),
            phi0=rng.uniform(0, 2 * np.pi), t_ref=0.37 * YRSID_SI,
        )
        cls.t0 = cls.params["t_ref"] - 0.25 * 86400
        # a 10-min grid over +-3 d around the burst (the response reads
        # positions / unit vectors / light-travel times off this grid)
        cls.orbits = _orbits(t_lo=cls.t0 - 3 * 86400, t_hi=cls.t0 + 3 * 86400, dt=600.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.resp = pyResponseTDI(
                1.0 / cls.DT, cls.NPTS, orbits=cls.orbits, order=25, tdi="1st generation",
                tdi_chan="AET", force_backend="cpu",
            )
        cls.phase = ConstellationPhase.from_orbits(cls.orbits, step=3600.0)

    def _tdi(self, lam, beta, inc, psi, t_ref):
        t_abs = self.t0 + np.arange(self.NPTS) * self.DT
        x = t_abs - t_ref
        env = np.exp(-0.5 * (x / self.SIG) ** 2)
        ph = 2 * np.pi * self.F0 * x + self.params["phi0"]
        hSp = 0.5 * (1 + np.cos(inc) ** 2) * env * np.cos(ph)
        hSc = np.cos(inc) * env * np.sin(ph)
        h = (hSp * np.cos(2 * psi) - hSc * np.sin(2 * psi)) + 1j * (hSp * np.sin(2 * psi) + hSc * np.cos(2 * psi))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.resp.get_projections(h, lam, beta, t0=self.t0, t_buffer=self.T_BUFFER)
            out = self.resp.get_tdi_delays()
        k = self.resp.tdi_start_ind
        return np.asarray([np.asarray(o)[k:-k] for o in out[:2]])

    @staticmethod
    def _overlap_and_shift(a, b, dt):
        nf = 4 * a.shape[1]
        cc = np.fft.irfft(np.sum(np.fft.rfft(a, n=nf, axis=1).conj() * np.fft.rfft(b, n=nf, axis=1), axis=0), n=nf)
        j = int(np.argmax(cc))
        tau = (j if j < nf // 2 else j - nf) * dt
        norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
        return np.sum(a * b) / norm, tau, cc[j] / norm

    def _row(self):
        p = self.params
        c = np.zeros((1, NDIM))
        im = IND_MAP_ICRS
        c[0, im["cosinc"]], c[0, im["lam"]], c[0, im["sinbeta"]] = np.cos(p["inc"]), p["lam"], np.sin(p["beta"])
        c[0, im["psi"]], c[0, im["t_ref"]] = p["psi"], p["t_ref"]
        return c

    def _hop(self, move, lat, k):
        q, _ = move.get_proposal({"mbh": self._row().reshape(1, 1, 1, NDIM)}, _FixedRandom(lat_flags=[int(lat)], ks=[k]))
        r = q["mbh"].reshape(NDIM)
        im = IND_MAP_ICRS
        return r[im["lam"]], np.arcsin(r[im["sinbeta"]]), np.arccos(r[im["cosinc"]]), r[im["psi"]], r[im["t_ref"]]

    def test_all_eight_modes_reproduce_the_response(self):
        p = self.params
        h0 = self._tdi(p["lam"], p["beta"], p["inc"], p["psi"], p["t_ref"])
        move = SkyMove(ind_map=IND_MAP_ICRS, which="both", coord_frame="ssb_ecliptic", constellation_phase=self.phase)
        for lat in (False, True):
            for k in range(4):
                with self.subTest(lat=lat, k=k):
                    h1 = self._tdi(*self._hop(move, lat, k))
                    ov, tau, ovmax = self._overlap_and_shift(h0, h1, self.DT)
                    self.assertGreater(ov, 0.98)
                    # no residual time shift (the cross-correlation peak sits
                    # within one sample; a wrong t_ref hop lands hundreds of
                    # seconds away -- see the mutation test below)
                    self.assertLessEqual(abs(tau), self.DT)
                    self.assertGreater(ovmax, 0.98)

    def test_a_move_without_the_time_transform_is_caught(self):
        p = self.params
        h0 = self._tdi(p["lam"], p["beta"], p["inc"], p["psi"], p["t_ref"])

        class _NoTimeShift(SkyMove):
            def _from_lisa_frame(self, coords):
                out = super()._from_lisa_frame(coords)
                out[:, self.ind_map["t_ref"]] = coords[:, self.ind_map["t_ref"]] + self._dt_hold
                return out

        move = _NoTimeShift(ind_map=IND_MAP_ICRS, which="lat", coord_frame="ssb_ecliptic", constellation_phase=self.phase, jacobian=False)
        # keep the ORIGINAL SSB time: hold = t_ref - t_L of the input row
        row = self._row()
        move._dt_hold = p["t_ref"] - move._to_lisa_frame(row)[0, IND_MAP_ICRS["t_ref"]]
        lam, beta, inc, psi, t_ref = self._hop(move, True, 0)
        self.assertAlmostEqual(t_ref, p["t_ref"], places=6)
        h1 = self._tdi(lam, beta, inc, psi, t_ref)
        ov, tau, _ = self._overlap_and_shift(h0, h1, self.DT)
        # the correctly-hopped SSB time differs from the original by hundreds of seconds
        correct = SkyMove(ind_map=IND_MAP_ICRS, which="lat", coord_frame="ssb_ecliptic", constellation_phase=self.phase)
        dt_hop = self._hop(correct, True, 0)[4] - p["t_ref"]
        self.assertGreater(abs(dt_hop), 50.0)
        self.assertLess(abs(tau + dt_hop), 2 * self.DT + 1e-9)
        self.assertLess(ov, 0.98)


if __name__ == "__main__":
    unittest.main()
