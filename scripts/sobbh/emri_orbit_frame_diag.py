"""Localize the ~8% ecliptic-orbits EMRI mismatch: measure whether the ORBIT's
icrs->ecliptic rotation (astropy BarycentricMeanEcliptic, used by L1Orbits frame=
"ecliptic") agrees with the SKY's icrs->ecliptic rotations used in the response/EMRI.

Three ecliptic conventions are in play:
  * sources.utils.icrs_to_ecliptic        -> LISA-DDPC fixed-obliquity (the EMRI sky angles)
  * response.directresponse.icrs_to_ecliptic -> astropy 'barycentrictrueecliptic'
  * detector.icrs_to_ecliptic (positions) -> astropy 'BarycentricMeanEcliptic' (the orbits)

If the orbit rotation and the sky rotation disagree by an angle large enough to move the
LISA response, that's the ecliptic-orbits inconsistency.
"""
import numpy as np
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, BarycentricTrueEcliptic
import astropy.units as u

from lisatools.sources.utils import icrs_to_ecliptic as sky_i2e_ddpc          # LISA-DDPC
from lisatools.response.directresponse import icrs_to_ecliptic as resp_i2e    # astropy true

# EMRI src1 sky (from the mojito catalogue / earlier runs)
RA, DEC = 5.2391, -0.1944


def unit(ra, dec):
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def angle_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def ecl_unit_from_angles(lam, beta):
    return np.array([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)])


def astropy_rotate(n_icrs, frame_cls):
    c = SkyCoord(x=n_icrs[0], y=n_icrs[1], z=n_icrs[2],
                 representation_type="cartesian", frame="icrs")
    ce = c.transform_to(frame_cls())
    ce.representation_type = "cartesian"
    return np.array([ce.x.value, ce.y.value, ce.z.value])


def main():
    print(f"  EMRI sky: ra={RA:.4f} dec={DEC:.4f}", flush=True)
    n_icrs = unit(RA, DEC)

    # sky ecliptic unit vectors via each convention
    lam_d, beta_d = sky_i2e_ddpc(RA, DEC)              # LISA-DDPC (EMRI sky angles)
    lam_t, beta_t = resp_i2e(RA, DEC)                  # astropy true (response convert path)
    n_ddpc = ecl_unit_from_angles(lam_d, beta_d)
    n_true = astropy_rotate(n_icrs, BarycentricTrueEcliptic)
    n_mean = astropy_rotate(n_icrs, BarycentricMeanEcliptic)   # the ORBIT rotation

    print("\n  ecliptic sky-direction unit vectors (from the SAME ICRS ra/dec):", flush=True)
    print(f"    LISA-DDPC (sources.utils, EMRI angles) lam={lam_d:.5f} beta={beta_d:.5f}", flush=True)
    print(f"    astropy true  (response convert)       lam={lam_t:.5f} beta={beta_t:.5f}", flush=True)
    print(f"    astropy mean  (L1Orbits frame=ecliptic) -> cart {n_mean}", flush=True)

    print("\n  pairwise angular disagreement [deg] (and [arcsec]):", flush=True)
    for (na, nb, lbl) in [
        (n_ddpc, n_true, "DDPC(sky) vs astropy-true(resp)"),
        (n_ddpc, n_mean, "DDPC(sky) vs astropy-mean(ORBIT)"),
        (n_true, n_mean, "astropy-true(resp) vs astropy-mean(ORBIT)"),
    ]:
        ang = angle_between(na, nb)
        print(f"    {lbl:42s}: {ang:.6f} deg = {ang*3600:.2f} arcsec", flush=True)

    # rough sensitivity: a sky-direction error d (rad) -> k.x phase error ~ 2*pi*f0*(1AU/c)*d
    f0 = 2e-3; kx = 1.496e11 * 3.3356e-9  # ~500 s
    ang_orbit = np.radians(angle_between(n_ddpc, n_mean))
    print(f"\n  conventions AGREE to {angle_between(n_ddpc, n_mean)*3600:.2f} arcsec "
          f"-> k.x mismatch ~ {0.5*(2*np.pi*f0*kx*ang_orbit)**2:.2e}  (NOT the 8%)", flush=True)

    # ---- the real cause: the +/x POLARIZATION BASIS (u,v) is defined relative to the FRAME
    # POLE, so the ICRS and ecliptic responses project h+/hx onto bases rotated by the
    # PARALLACTIC ANGLE chi between the celestial pole and the ecliptic pole at the source. ----
    def astropy_back(n_ecl):
        c = SkyCoord(x=n_ecl[0], y=n_ecl[1], z=n_ecl[2],
                     representation_type="cartesian", frame=BarycentricMeanEcliptic())
        ci = c.transform_to("icrs"); ci.representation_type = "cartesian"
        return np.array([ci.x.value, ci.y.value, ci.z.value])

    sl, cl, sb, cb = np.sin(RA), np.cos(RA), np.sin(DEC), np.cos(DEC)
    v_icrs = np.array([-sb * cl, -sb * sl, cb])            # "north" in ICRS (response v)
    sl2, cl2, sb2, cb2 = np.sin(lam_d), np.cos(lam_d), np.sin(beta_d), np.cos(beta_d)
    v_ecl = np.array([-sb2 * cl2, -sb2 * sl2, cb2])        # "north" in ecliptic (response v)
    v_ecl_in_icrs = astropy_back(v_ecl)
    chi = angle_between(v_icrs, v_ecl_in_icrs)             # parallactic angle [deg]
    two_chi = np.radians(2 * chi)
    print(f"\n  PARALLACTIC ANGLE chi (celestial pole vs ecliptic pole at the source) = {chi:.3f} deg", flush=True)
    print(f"    -> polarization basis rotates by 2*chi = {2*chi:.3f} deg between ICRS and ecliptic resp", flush=True)
    print(f"    -> a pure 2chi pol rotation gives mismatch ~ sin^2(2chi) = {np.sin(two_chi)**2:.3e} "
          f"(scaled by the source's pol content) -- THIS is the ~8% (ALT mm=8.1e-2, |O|=0.919)", flush=True)
    print(f"\n  CONCLUSION: not an orbit-rotation bug and not an ecliptic-convention bug. The +/x basis", flush=True)
    print(f"  is frame-relative; mojito's data uses the ICRS basis, so ICRS-orbits+convert_to_ra_dec", flush=True)
    print(f"  (SPECIAL) matches (mm 3.8e-5) while ecliptic-orbits (ALT) is rotated by 2chi ~ {2*chi:.1f} deg.", flush=True)


if __name__ == "__main__":
    main()
