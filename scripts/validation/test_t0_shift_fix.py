"""Verify the legacy-response t0_shift_to_data fix.

t0_shift_to_data is the sub-sample offset that aligns the response output onto the DATA
grid. A shift of delta must therefore come out as a PURE TIME DELAY of the TDI output:
the output sampled at {t0 + delta + i*dt} instead of {t0 + i*dt}. In the frequency domain
that is FB(f) = FA(f) * exp(+2 pi i f delta) -> |FB/FA| ~ 1 and a recovered delay ~ delta.

  * BEFORE the fix: t0_shift_to_data was baked into t0_arr (the waveform reference). The
    kernel index `delay - t0_arr` then cancelled the shift, so the waveform was sampled at
    the UNSHIFTED retarded time while the output was labelled on the shifted grid ->
    recovered delay ~ 0 (the output did NOT move onto the data grid).
  * AFTER the fix: the shift lives on the eval grid (t_arr), the projection kernel indexes
    the waveform relative to the unshifted t0, and the TDI kernel references the projection
    start (t0_arr + t_arr[0]) -> recovered delay ~ delta.

Isolated setup: EqualArmlengthOrbits (analytic) + a monochromatic h+/hx, so the only thing
that changes between the two runs is t0_shift_to_data.
"""
import numpy as np
from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI

DT = 10.0; N = 16384; TOBS_S = N * DT
F0 = 2e-3          # carrier
DELTA = 3.7        # sub-sample data-grid shift (< dt)
T_BUF = 30000.0


class SineGen:
    """h = h+ - i hx = exp(-i 2 pi F0 t); length matches the response grid."""
    def __call__(self, *args, T=None, dt=None, **kwargs):
        n = int(round(T * YRSID_SI / dt))
        t = np.arange(n) * dt
        return np.exp(-1j * 2 * np.pi * F0 * t)


def run(shift):
    wl = ResponseWrapper(SineGen(), Tobs=TOBS_S / YRSID_SI, dt=DT,
        index_lambda=0, index_beta=1, t0=0.0, t0_shift_to_data=shift, t_buffer=T_BUF,
        flip_hx=False, remove_sky_coords=False, is_ecliptic_latitude=True,
        tdi=TDIConfig("2nd generation", force_backend="cpu"), tdi_chan="XYZ",
        order=25, remove_garbage="zero", orbits=EqualArmlengthOrbits(), force_backend="cpu")
    out = np.atleast_2d(np.asarray(wl(1.0, 0.5)))   # lam=1.0, beta=0.5 (gen ignores them)
    return out


def recover_delay(A, B):
    """Recover the time delay between B and A from the FFT phase slope (power-weighted)."""
    nb = int(T_BUF / DT) + 200
    sl = slice(nb, N - nb)
    f = np.fft.rfftfreq(N, d=DT)
    deltas, mags = [], []
    for ci in range(A.shape[0]):
        a = np.zeros(N); a[sl] = A[ci, sl]
        b = np.zeros(N); b[sl] = B[ci, sl]
        FA = np.fft.rfft(a); FB = np.fft.rfft(b)
        m = np.abs(FA) > 0.05 * np.abs(FA).max()
        ratio = FB[m] / FA[m]
        # delta from phase: angle = 2 pi f delta
        d = np.angle(ratio) / (2 * np.pi * f[m])
        w = np.abs(FA[m])
        deltas.append(float(np.sum(d * w) / np.sum(w)))
        mags.append(float(np.median(np.abs(ratio))))
    return deltas, mags


def main():
    print(f"  DT={DT}  N={N}  F0={F0}  true DELTA={DELTA} s  (< dt={DT})", flush=True)
    A = run(0.0)
    B = run(DELTA)
    deltas, mags = recover_delay(A, B)
    print(f"  per-channel recovered delay [s] (target {DELTA}): "
          f"{[f'{d:.3f}' for d in deltas]}", flush=True)
    print(f"  per-channel |FB/FA| (target 1.0):                 "
          f"{[f'{m:.4f}' for m in mags]}", flush=True)
    dmean = float(np.mean(deltas))
    print(f"\n  mean recovered delay = {dmean:.3f} s   target = {DELTA:.3f} s   "
          f"err = {dmean - DELTA:+.3f} s  ({100*(dmean-DELTA)/DELTA:+.1f}%)", flush=True)
    ok = abs(dmean - DELTA) < 0.05 * DELTA
    print(f"  VERDICT: {'PASS -- shift comes out as a clean data-grid delay' if ok else 'FAIL -- shift not recovered (bug: delay ~ 0)'}", flush=True)


if __name__ == "__main__":
    main()
