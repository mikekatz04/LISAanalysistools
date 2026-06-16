"""Convention check for the time-domain SOBBH TDI-on-the-fly source feed.

Verifies that the amp/phase representation consumed by SOBBHTDIonFly
(SOBBHWaveform.compute_amp_phase -> A, 2*Phi) reproduces, through the
on-the-fly response's get_hp_hc formula, EXACTLY the hp/hx that the legacy
pyResponse path gets from SOBBHWaveform.__call__.

This isolates the SOURCE convention (amplitude factors, phase sign, psi
rotation) from the orbit/TDI machinery. Pure Python, no orbits needed.

get_hp_hc (lat_tdi_on_the_fly.cu):
    hSp = -cos(phase) * amp * (1 + cosi^2)
    hSc = -sin(phase) * 2 * amp * cosi
    hp  = hSp*cos2psi - hSc*sin2psi
    hc  = hSp*sin2psi + hSc*cos2psi
with amp = A, phase = 2*Phi + pi, cosi = cos(inc).
"""
import numpy as np
from lisatools.sources.sobbh.waveform import SOBBHWaveform

# --- a representative SOBBH (mojito-ish) -----------------------------------
M1, M2 = 35.0, 30.0          # Msun
S1, S2 = 0.1, -0.05          # aligned spins
DIST = 2.0                   # Gpc
INC = 0.9                    # rad
F_LOW = 8e-3                 # Hz
LAM, BETA = 1.3, -0.4        # ecliptic (unused by the source feed)
PSI = 0.7                    # rad
PHI0 = 1.1                   # reference-epoch orbital phase
REF = 1.0e7                  # absolute reference epoch (sec)

DT = 5.0
TOBS = 2.0e5                 # short window, fully pre-merger
T0 = REF + 1.0e4             # window starts after the reference epoch


def get_hp_hc_formula(amp, gw_phase, inc, psi):
    """Replicate lat_tdi_on_the_fly.cu get_hp_hc with phase = 2*Phi + pi."""
    phase = gw_phase + np.pi
    cosi = np.cos(inc)
    cos2psi, sin2psi = np.cos(2.0 * psi), np.sin(2.0 * psi)
    hSp = -np.cos(phase) * amp * (1.0 + cosi * cosi)
    hSc = -np.sin(phase) * 2.0 * amp * cosi
    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi
    return hp, hc


def main():
    wf = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=T0, reference_time=REF, pad_zeros=False)

    # Legacy path: complex h = hp_rot + 1j hx_rot on the internal grid.
    h_legacy = np.asarray(
        wf(M1, M2, S1, S2, DIST, INC, F_LOW, LAM, BETA, PSI, PHI0)
    )
    hp_legacy, hx_legacy = h_legacy.real, h_legacy.imag

    # On-the-fly feed: A, 2*Phi on the SAME grid, then get_hp_hc formula.
    abs_t, gw_amp, gw_phase = wf.compute_amp_phase(
        M1, M2, S1, S2, DIST, F_LOW, PHI0
    )
    hp_fly, hx_fly = get_hp_hc_formula(gw_amp, gw_phase, INC, PSI)

    n = min(len(hp_legacy), len(hp_fly))
    hp_legacy, hx_legacy = hp_legacy[:n], hx_legacy[:n]
    hp_fly, hx_fly = hp_fly[:n], hx_fly[:n]

    scale = np.max(np.abs(hp_legacy)) + np.max(np.abs(hx_legacy))
    err_p = np.max(np.abs(hp_legacy - hp_fly)) / scale
    err_x = np.max(np.abs(hx_legacy - hx_fly)) / scale

    print(f"  n samples compared : {n}")
    print(f"  amp range          : [{gw_amp.min():.3e}, {gw_amp.max():.3e}]")
    print(f"  legacy |hp| max     : {np.max(np.abs(hp_legacy)):.6e}")
    print(f"  rel err hp          : {err_p:.3e}")
    print(f"  rel err hx          : {err_x:.3e}")
    # ~1e-13 residual is pure FP order-of-ops: the on-the-fly encodes the sign
    # as cos(2*Phi + pi) = -cos(2*Phi), and the +pi on a ~1e4-rad phase rounds
    # at the phase ULP (~1e-12 rad). A convention error would be O(1).
    ok = err_p < 1e-12 and err_x < 1e-12
    print("\n  RESULT:", "PASS (source convention matches legacy)" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
