"""End-to-end smoke test of the time-domain SOBBH TDI-on-the-fly class.

Builds the SOBBH TDI two ways on the SAME EqualArmlengthOrbits and checks they
agree:

  A = legacy pyResponse  : ResponseWrapper(SOBBHWaveform)         (Lagrange interp)
  B = TDI-on-the-fly      : SOBBHTDIonFly(SOBBHWaveform, ...)     (analytic delays)

Both share the identical 3.5PN core (waveform_generate_h_plus_cross vs
waveform_generate_amp_phase) -- only the response differs. EqualArmlengthOrbits
is in the ecliptic frame, so the on-the-fly sky (ra, dec) slots take the same
ecliptic (lam, beta) the legacy uses. Reports a windowed normalized overlap on
the common interior.
"""
import numpy as np
from lisatools.utils.constants import YRSID_SI
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from bbhx.sobbhtdionfly import SOBBHTDIonFly

# --- a representative SOBBH -------------------------------------------------
M1, M2 = 35.0, 30.0
S1, S2 = 0.1, -0.05
DIST = 2.0                   # Gpc
INC = 0.9
F_LOW = 8e-3
LAM, BETA = 1.3, -0.4        # ecliptic sky
PSI = 0.7
PHI0 = 1.1
T_SHIFT = 0.0

DT = 5.0
TOBS = 4.0e5                 # sec
REF = 0.0                    # reference epoch == window start (synthetic)
T0 = 0.0
NCH = 3
TDI_GEN = "2nd generation"
ORDER = 40
T_BUFFER = 3.0e4


def windowed_overlap(a, b, alpha=0.1):
    """Normalized broadband overlap of two (3, N) TD signals on the interior."""
    from scipy.signal.windows import tukey
    n = a.shape[1]
    w = tukey(n, alpha)
    num = den_a = den_b = 0.0
    for ch in range(3):
        fa = np.fft.rfft(a[ch] * w)
        fb = np.fft.rfft(b[ch] * w)
        num += np.real(np.vdot(fa, fb))
        den_a += np.real(np.vdot(fa, fa))
        den_b += np.real(np.vdot(fb, fb))
    return num / np.sqrt(den_a * den_b)


def main():
    tdi_config = TDIConfig(TDI_GEN, force_backend="cpu")
    orbits = EqualArmlengthOrbits(force_backend="cpu")

    # --- A: legacy pyResponse ---------------------------------------------
    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=T0, reference_time=REF)
    legacy = ResponseWrapper(
        sobbh_gen,
        Tobs=TOBS / YRSID_SI,
        dt=DT,
        index_lambda=7,
        index_beta=8,
        t0=T0,
        t_buffer=T_BUFFER,
        # flip_hx=False is the mojito-matching convention (2026-06-14 fix);
        # it reads (hp_rot, +hx_rot) -- exactly what get_hp_hc produces from
        # the on-the-fly amp/phase feed, so the two responses should agree.
        flip_hx=False,
        is_ecliptic_latitude=True,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(force_backend="cpu"),
        tdi=tdi_config,
        tdi_chan="XYZ",
        order=ORDER,
        force_backend="cpu",
    )
    # output-basis order: m1,m2,s1,s2,dist,inc,f_low,lam,beta,psi,phi0,t_shift
    params = [M1, M2, S1, S2, DIST, INC, F_LOW, LAM, BETA, PSI, PHI0, T_SHIFT]
    A = np.asarray(legacy(*params, convert_to_ra_dec=False))[:NCH]
    print(f"  A (legacy)  shape={A.shape}  |X|max={np.max(np.abs(A[0])):.3e}")

    # --- B: TDI-on-the-fly -------------------------------------------------
    fly = SOBBHTDIonFly(
        SOBBHWaveform(Tobs=TOBS, dt=DT, t0=T0, reference_time=REF),
        orbits,
        tdi_config,
        DT,
        TOBS,
        t0=T0,
        force_backend="cpu",
    )
    grid_t = np.arange(A.shape[1]) * DT + T0
    B = np.asarray(
        fly(M1, M2, S1, S2, DIST, F_LOW, PHI0, INC, LAM, BETA, PSI,
            t_shift=T_SHIFT, upsample_t_arr=grid_t, combine=True)
    )[:NCH]
    print(f"  B (on-fly)  shape={B.shape}  |X|max={np.max(np.abs(B[0])):.3e}")
    print(f"  finite A/B: {np.isfinite(A).all()} / {np.isfinite(B).all()}")

    # Common interior: both nonzero (legacy zeros its t_buffer edges).
    nz = (np.abs(A).sum(0) > 0) & (np.abs(B).sum(0) > 0)
    i0, i1 = np.argmax(nz), len(nz) - np.argmax(nz[::-1])
    pad = int(T_BUFFER / DT)
    i0, i1 = i0 + pad, i1 - pad
    Ai, Bi = A[:, i0:i1], B[:, i0:i1]
    print(f"  interior samples: {Ai.shape[1]}  (idx {i0}..{i1})")

    ov = windowed_overlap(Ai, Bi)
    print(f"\n  broadband normalized overlap (interior): {ov:.6f}")
    print(f"  mismatch 1-O                            : {1 - ov:.3e}")

    # --- diagnostics: phase-max + time-max overlap ------------------------
    from scipy.signal.windows import tukey
    n = Ai.shape[1]
    w = tukey(n, 0.1)
    cross = np.zeros(n // 2 + 1, dtype=complex)
    da = db = 0.0
    for ch in range(3):
        fa = np.fft.rfft(Ai[ch] * w)
        fb = np.fft.rfft(Bi[ch] * w)
        cross += np.conj(fa) * fb
        da += np.real(np.vdot(fa, fa))
        db += np.real(np.vdot(fb, fb))
    norm = np.sqrt(da * db)
    phase_max = np.abs(np.sum(cross)) / norm
    # time-max via inverse FFT of the cross-spectrum (full-band correlation)
    corr = np.fft.irfft(cross, n=n)
    lag = int(np.argmax(np.abs(corr)))
    if lag > n // 2:
        lag -= n
    time_phase_max = np.max(np.abs(corr)) / norm
    print(f"  phase-max |O| (no shift)               : {phase_max:.6f}")
    print(f"  time+phase-max |O|                     : {time_phase_max:.6f}  (lag={lag} samp = {lag*DT:.1f} s)")
    print("\n  RESULT:", "PASS (on-the-fly runs + agrees with legacy)"
          if (np.isfinite(B).all() and ov > 0.99) else "CHECK (low overlap / non-finite)")


if __name__ == "__main__":
    main()
