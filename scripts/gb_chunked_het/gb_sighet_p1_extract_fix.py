"""P0/P1: reproduce + fix the extraction slip at the NODE level (Python).

Key fact: the extract is an invertible per-sample reparametrization, so the
raw complex envelope at the generator's internal nodes is EXACTLY
reconstructable from the stored spline node arrays:
    M_i = amp_i * exp(1j*(tdi_phase_i + phase_ref_i))
(the flip/pjump bookings cancel in the product). The slip corrupts the
NODE SEQUENCES (a missed sign booking rectifies the amplitude -> kinks at
every subsequent pattern zero + pi-steps in phase), which the separate
amp/phase cubic splines then interpolate wrongly BETWEEN nodes for months.

P0 (reproduce): leave-one-out node test — spline the STORED (amp, tdi_phase)
on even nodes, predict odd nodes, compare the complex product against the
exact node values. Slip-class refs show LOO spikes; clean refs don't.

P1 (fix): rebuild the representation from reconstructed M via complex
products: dtheta_i = angle(M_i conj(M_{i-1})); at amplitude-minimum
intervals with |rotation| ~ pi, book a SIGN FLIP (signed smooth amplitude,
pi removed from the phase step); 2*pi windings stay in the continuous
cumulative phase (smooth, spline-representable). Same LOO test on the
corrected (amp_signed, theta - phase_ref) pair.

PASS = corrected LOO error uniform and low for ALL refs incl ref02/05.
Ambiguous minima (aliased winding, |M| tiny AND |dtheta|~pi/2..3pi/2
unresolvable) are counted -> that residual population is what the P3
fixed-denser-N host re-batch is for.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_p1_extract_fix.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations


def loo_err(t, amp, ph, pref, M_exact):
    """Leave-one-out: spline (amp, ph) on even nodes, predict odd; return
    max |z_pred - z_exact| / max|z| per channel (complex-product metric)."""
    ev, od = np.arange(0, len(t), 2), np.arange(1, len(t) - 1, 2)
    out = []
    for c in range(amp.shape[0]):
        fa = CubicSpline(t[ev], amp[c, ev])
        fp = CubicSpline(t[ev], ph[c, ev])
        z_pred = fa(t[od]) * np.exp(1j * (fp(t[od]) + pref[od]))
        z_ex = M_exact[c, od]
        out.append(float(np.max(np.abs(z_pred - z_ex))
                         / max(np.max(np.abs(M_exact[c])), 1e-300)))
    return out


def p1_correct(t, M):
    """Signed-amplitude + continuous-phase rebuild from node products."""
    nch, N = M.shape
    amp_s = np.abs(M).copy()
    theta = np.zeros_like(amp_s)
    n_flip = np.zeros(nch, int)
    n_ambig = np.zeros(nch, int)
    for c in range(nch):
        z = M[c]
        a = np.abs(z)
        dth = np.angle(z[1:] * np.conj(z[:-1]))
        sign = np.ones(N)
        th = np.zeros(N)
        th[0] = np.angle(z[0])
        is_min = np.zeros(N, bool)
        is_min[1:-1] = (a[1:-1] < a[:-2]) & (a[1:-1] < a[2:])
        s_cur = 1.0
        for i in range(1, N):
            d = dth[i - 1]
            near_min = is_min[i] or is_min[i - 1]
            if near_min and abs(d) > np.pi / 2:
                k = int(np.round(d / np.pi))
                if abs(d - k * np.pi) < np.pi / 3 and k % 2 != 0:
                    s_cur = -s_cur          # true sign crossing
                    d = d - k * np.pi
                    n_flip[c] += 1
                elif abs(d) > np.pi / 2:
                    n_ambig[c] += 1         # aliased/unresolved winding
            th[i] = th[i - 1] + d
            sign[i] = s_cur
        amp_s[c] = sign * a
        theta[c] = th
    return amp_s, theta, n_flip, n_ambig


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, 12288
    t_start = int(0.5 * YRSID_SI / dt) * dt
    edge, tk = 330, 307
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4,
                          max_freq=2e-2, min_time=edge * Nf * dt,
                          max_time=(Nt - edge) * Nf * dt,
                          force_backend=backend)
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend, d_d=0.0,
        tdi_type="XYZ", tukey_alpha=2.0 * tk / Nt)
    chunked.convert_to_ra_dec = False
    sighet = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=0, nt_layer=64,
        m_active_half_width=2)
    g = sighet._g
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    nfa = ihi - ilo
    nta = np.zeros(Nt, bool)
    nta[wdm_set.active_slice_t] = True
    nta = int(nta.sum())
    invC = np.zeros((3, 3, nfa, nta))
    for c in range(3):
        invC[c, c] = 1.0
    holder = proto._FullGridWDMHolder(np.zeros((3, nfa, nta)), invC)
    zeros = np.zeros(1, dtype=np.int32)
    dummy = np.array([1e-22, 3e-3, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.3])
    sighet.setup_in_model(holder, dummy[None, :], zeros)
    gen = sighet._keep_alive["gb_gen"]

    rngr = np.random.default_rng(19)
    refs = []
    for _ in range(8):
        refs.append(np.array([
            10 ** rngr.uniform(-22.5, -21.0),
            rngr.uniform(1.5e-3, 1.5e-2),
            rngr.uniform(0.0, 3e-16),
            0.0,
            rngr.uniform(0, 2 * np.pi),
            np.arccos(rngr.uniform(-1, 1)),
            rngr.uniform(0, np.pi),
            rngr.uniform(0, 2 * np.pi),
            np.arcsin(rngr.uniform(-1, 1)),
        ]))

    print("ref  min_env | P0 LOO stored (X/Y/Z)          | "
          "P1 LOO corrected (X/Y/Z)        | flips  ambig")
    results = []
    for rrr, ref in enumerate(refs):
        spl = proto.build_spl(gen, ref)

        def arr(x):
            x = np.asarray(x)
            return x[0] if x.ndim == 3 else x

        a_n = arr(spl.tdi_amp)                    # (3, N)
        ph_n = arr(spl.tdi_phase)                 # (3, N)
        pref_n = np.asarray(spl.phase_ref)
        pref_n = pref_n[0] if pref_n.ndim == 2 else pref_n   # (N,)
        t_n = np.asarray(spl.x)
        t_n = t_n[0] if t_n.ndim == 2 else t_n

        M = a_n * np.exp(1j * (ph_n + pref_n[None, :]))
        a_abs = np.abs(M)
        min_env = float((a_abs / a_abs.max(axis=1, keepdims=True)).min())

        e0 = loo_err(t_n, a_n, ph_n, pref_n, M)
        amp_c, theta_c, n_flip, n_ambig = p1_correct(t_n, M)
        e1 = loo_err(t_n, amp_c, theta_c - pref_n[None, :], pref_n, M)
        print(f"{rrr:3d}  {min_env:.4f} | "
              + " ".join(f"{x:9.2e}" for x in e0) + " | "
              + " ".join(f"{x:9.2e}" for x in e1)
              + f" | {n_flip.sum():3d}  {n_ambig.sum():3d}")
        results.append((rrr, min_env, e0, e1,
                        int(n_flip.sum()), int(n_ambig.sum())))

    np.savez("./ratio_proto_out/p1_extract_fix.npz",
             results=np.array([(r, m, *a, *b, f, g_) for
                               r, m, a, b, f, g_ in results]))
    worst0 = max(max(r[2]) for r in results)
    worst1 = max(max(r[3]) for r in results)
    print(f"\n[P0/P1] worst LOO stored {worst0:.3e} -> corrected "
          f"{worst1:.3e}; ambiguous minima total "
          f"{sum(r[5] for r in results)} (P3 re-batch population)")


if __name__ == "__main__":
    main()
