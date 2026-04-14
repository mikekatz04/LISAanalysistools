"""test for the conversions between sky coordinates"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from lisatools.sources.utils import icrs_to_ecliptic, ecliptic_to_icrs

ra = 3.47
dec = 0.05
psi_icrs = 0.09

def astropy_icrs_to_ecliptic(ra, dec):
        c = SkyCoord(ra=ra*u.radian, dec=dec*u.radian, frame='icrs')
        ecliptic = c.barycentricmeanecliptic
        return ecliptic.lon.radian, ecliptic.lat.radian

def astropy_ecliptic_to_icrs(lon, lat):
        c = SkyCoord(lon=lon*u.radian, lat=lat*u.radian, frame='barycentricmeanecliptic')
        icrs = c.icrs
        return icrs.ra.radian, icrs.dec.radian

lam, beta = astropy_icrs_to_ecliptic(ra, dec)

class TestSky:

    def test_icrs_to_ecliptic(self):
        lon, lat = icrs_to_ecliptic(ra, dec)
        expected_lon, expected_lat = astropy_icrs_to_ecliptic(ra, dec)
        assert np.isclose(lon, expected_lon), f"LON: lisatools: {lon}, astropy: {expected_lon}"
        assert np.isclose(lat, expected_lat), f"LAT: lisatools: {lat}, astropy: {expected_lat}"

    def test_ecliptic_to_icrs(self):
        ra_converted, dec_converted = ecliptic_to_icrs(lam, beta)
        expected_ra, expected_dec = astropy_ecliptic_to_icrs(lam, beta)
        assert np.isclose(ra_converted, expected_ra), f"RA: lisatools: {ra_converted}, astropy: {expected_ra}"
        assert np.isclose(dec_converted, expected_dec), f"DEC: lisatools: {dec_converted}, astropy: {expected_dec}"

    def test_round_trip(self):
        lon, lat = icrs_to_ecliptic(ra, dec)
        ra_converted, dec_converted = ecliptic_to_icrs(lon, lat)
        assert np.isclose(ra, ra_converted)
        assert np.isclose(dec, dec_converted)

    def test_astropy_round_trip(self):
        lon, lat = astropy_icrs_to_ecliptic(ra, dec)
        ra_converted, dec_converted = astropy_ecliptic_to_icrs(lon, lat)
        assert np.isclose(ra, ra_converted)
        assert np.isclose(dec, dec_converted)

    def test_round_trip_with_psi(self):
        # Test round trip with psi included
        lon, lat, psi_ecl = icrs_to_ecliptic(ra, dec, psi_icrs)
        ra_converted, dec_converted, psi_converted = ecliptic_to_icrs(lon, lat, psi_ecl)
        assert np.isclose(ra, ra_converted)
        assert np.isclose(dec, dec_converted)
        assert np.isclose(psi_icrs, psi_converted)
