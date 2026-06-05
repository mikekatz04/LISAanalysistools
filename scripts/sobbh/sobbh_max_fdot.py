#!/usr/bin/env python
"""Probe maximal fdot of an SOBBH TDI signal over a 1-year observation.

Builds an SOBBHTDIonTheFly waveform, retrieves the TDTDIOutput splines
(amp, tdi_phase, phase_ref), and uses CubicSplineInterpolant's analytic
derivative= keyword to evaluate

    f      = (1/(2 pi)) d(phase_total)/dt
    fdot   = (1/(2 pi)) d^2(phase_total)/dt^2

on a dense grid. The "carrier" is phase_ref (sky/Doppler delay applied at
SC1); per-channel phase_total = phase_ref + tdi_phase[channel] adds the
TDI delay residuals. Reports the maxima of both.
"""

import numpy as np
import matplotlib.pyplot as plt

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import SOBBHTDIonTheFly


def probe(m1, m2, s1=0.0, s2=0.0, distance=1.0e9, f_low=2.0e-3, phi_c=0.0,
          inc=0.5, psi=1.0, lam=4.1, beta=0.1,
          Tobs_years=1.0, dt=10.0, N_samples=16384, N_probe=4096):
    backend = "cpu"
    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig('2nd generation')

    Tobs = Tobs_years * YRSID_SI
    t_start = 0.0
    t_ref = t_start

    t_tdi = np.linspace(t_start, t_start + Tobs, N_samples)

    gen = SOBBHTDIonTheFly(
        t_tdi, Tobs, t_ref,
        1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )

    arrs = [np.full(1, v) for v in (m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)]
    out = gen(*arrs, convert_to_ra_dec=False, return_spline=True)

    # Avoid the spline boundaries so finite-element derivatives are clean
    edge = 1e-3 * Tobs
    t_probe = np.linspace(t_start + edge, t_start + Tobs - edge, N_probe)
    t_probe_2d = np.atleast_2d(t_probe)
    t_probe_3d = np.tile(t_probe, (1, 3, 1))

    phase_ref = out.phase_ref_spl(t_probe_2d, derivative=0)[0]
    dphase_ref = out.phase_ref_spl(t_probe_2d, derivative=1)[0]
    d2phase_ref = out.phase_ref_spl(t_probe_2d, derivative=2)[0]

    f_ref = dphase_ref / (2.0 * np.pi)
    fdot_ref = d2phase_ref / (2.0 * np.pi)

    # Total phase per channel = phase_ref + tdi_phase[chan]
    tdi_dphase = out.tdi_phase_spl(t_probe_3d, derivative=1)[0]   # (3, N_probe)
    tdi_d2phase = out.tdi_phase_spl(t_probe_3d, derivative=2)[0]
    f_chan = (dphase_ref[None, :] + tdi_dphase) / (2.0 * np.pi)
    fdot_chan = (d2phase_ref[None, :] + tdi_d2phase) / (2.0 * np.pi)

    return t_probe, f_ref, fdot_ref, f_chan, fdot_chan


def report(label, t_probe, f_ref, fdot_ref, f_chan, fdot_chan):
    print(f"=== {label} ===")
    print(f"  f_ref  range  [Hz]   : {f_ref.min():.6e} .. {f_ref.max():.6e}")
    print(f"  fdot_ref range [Hz/s]: {fdot_ref.min():.6e} .. {fdot_ref.max():.6e}")
    i_max = int(np.argmax(np.abs(fdot_ref)))
    print(f"  |fdot_ref| max       : {abs(fdot_ref[i_max]):.6e}  at t = {t_probe[i_max]:.3e} s "
          f"(f = {f_ref[i_max]:.6e} Hz)")
    for c, name in enumerate(("X", "Y", "Z")):
        ic = int(np.argmax(np.abs(fdot_chan[c])))
        print(f"  channel {name}: |fdot| max = {abs(fdot_chan[c, ic]):.6e}  at t = {t_probe[ic]:.3e} s "
              f"(f = {f_chan[c, ic]:.6e} Hz)")
    print()


if __name__ == "__main__":
    # 1) repo-default SOBBH (30 + 20 Msun, f_low = 2 mHz) -- deep inspiral over 1 yr
    res = probe(m1=30.0, m2=20.0, f_low=2.0e-3, Tobs_years=1.0)
    report("30 + 20 Msun, f_low = 2 mHz, T=1yr", *res)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    t_probe, f_ref, fdot_ref, f_chan, fdot_chan = res
    ax[0].plot(t_probe / YRSID_SI, f_ref, label="phase_ref")
    for c, name in enumerate(("X", "Y", "Z")):
        ax[0].plot(t_probe / YRSID_SI, f_chan[c], ls="--", lw=0.6, label=name)
    ax[0].set_ylabel("f [Hz]")
    ax[0].legend()
    ax[1].plot(t_probe / YRSID_SI, fdot_ref, label="phase_ref")
    for c, name in enumerate(("X", "Y", "Z")):
        ax[1].plot(t_probe / YRSID_SI, fdot_chan[c], ls="--", lw=0.6, label=name)
    ax[1].set_ylabel("fdot [Hz/s]")
    ax[1].set_xlabel("t [yr]")
    ax[1].legend()
    fig.suptitle("SOBBH 30+20 Msun, f_low=2 mHz, 1 yr")
    fig.tight_layout()
    fig.savefig("sobbh_max_fdot.png", dpi=120)
    print("Saved sobbh_max_fdot.png")

    # 2) sanity: heavier system / higher f_low -> stronger chirp over the year
    res2 = probe(m1=50.0, m2=40.0, f_low=5.0e-3, Tobs_years=1.0)
    report("50 + 40 Msun, f_low = 5 mHz, T=1yr", *res2)
