"""PURE lisatools likelihood of pre-built TDI templates vs mojito data.

Source-agnostic. Deliberately decoupled from waveform/response creation: this
process imports ONLY the lisatools likelihood machinery (domains /
analysiscontainer / sensitivity) and consumes the pre-built time-domain TDI
arrays a *_likelihood_compare.py builder wrote, e.g.

    /tmp/sobbh_ll_arrays_src0.npz   (SOBBH)
    /tmp/mbh_ll_arrays_id0.npz      (MBH)

each holding D (mojito data), A (legacy pyResponse), B (TDI-on-the-fly), dt,
and the band edges/labels the builder used.

It asserts that NO waveform/response generator module is in scope, then computes
logL = -1/2 <d-h|d-h>, SNRs and mm with AnalysisContainer -- proving the quoted
logL values are pure lisaanalysistools computations, independent of how the
templates were generated.

Usage:
    python likelihood_pure.py /tmp/sobbh_ll_arrays_src0.npz
    python likelihood_pure.py /tmp/mbh_ll_arrays_id0.npz
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from scipy.signal.windows import tukey

# --- ONLY lisatools likelihood machinery; no generators ---
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

# Prove the likelihood is computed with NO waveform/response generation in scope.
_FORBIDDEN = [
    "bbhx", "bbhx.sobbhtdionfly", "bbhx.mbhtdionfly", "phentax",
    "lisatools.response.tdionfly", "lisatools.response.directresponse",
    "lisatools.sources.sobbh.waveform", "lisatools.sources.bbh.waveform",
]
_leaked = [m for m in _FORBIDDEN if m in sys.modules]
assert not _leaked, f"waveform/response modules leaked into the pure likelihood: {_leaked}"
print(f"[pure] waveform/response generators in scope: NONE  (checked {len(_FORBIDDEN)})", flush=True)

TUKEY_ALPHA = 0.1
SENS = ["scirdv1", "mrdv1"]


def banner(s): print("\n" + "=" * 92 + f"\n {s}\n" + "=" * 92, flush=True)


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sobbh_ll_arrays_src0.npz"
    if not os.path.exists(npz):
        raise SystemExit(f"missing {npz}; run the matching *_likelihood_compare.py builder first.")
    z = np.load(npz, allow_pickle=True)
    D, A, B = np.asarray(z["D"]), np.asarray(z["A"]), np.asarray(z["B"])
    dt = float(z["dt"]); N_WIN = D.shape[1]
    bands = list(zip([str(x) for x in z["band_label"]],
                     [float(x) for x in z["band_lo"]], [float(x) for x in z["band_hi"]]))
    print(f"[pure] {npz}\n[pure] D/A/B {D.shape}  dt={dt}  bands={[b[0] for b in bands]}", flush=True)

    win = tukey(N_WIN, TUKEY_ALPHA)
    td = TDSettings(N_WIN, dt, t0=0.0, force_backend="cpu")

    for name, T in [("A = legacy pyResponse", A), ("B = TDI-on-the-fly", B)]:
        banner(name + "   (pure lisatools likelihood vs mojito data)")
        print(f"  {'sens':>8} {'band':>17} {'<d|d>':>11} {'<d|h>':>11} {'<h|h>':>11} "
              f"{'SNRopt':>7} {'SNRdet':>8} {'logL':>12} {'mm':>10}", flush=True)
        for sens in SENS:
            for tag, flo, fhi in bands:
                fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * dt),
                                min_freq=flo, max_freq=fhi, force_backend="cpu")
                d = TDSignal(D, td).transform(fd, window=win)
                t = TDSignal(T, td).transform(fd, window=win)
                ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=sens))
                dd = ac.inner_product().real
                opt, det = ac.template_snr(t)
                dh = float(np.real(ac.template_inner_product(t, normalize=False, complex=False)))
                logL = float(np.real(ac.template_likelihood(t)))
                O = ac.template_inner_product(t, normalize=True, complex=True)
                print(f"  {sens:>8} {tag:>17} {dd:11.3e} {dh:11.3e} {float(opt) ** 2:11.3e} "
                      f"{float(opt):7.1f} {float(det):+8.2f} {logL:12.4e} {1-abs(O):10.3e}", flush=True)
    print("\n[pure] DONE.", flush=True)


if __name__ == "__main__":
    main()
