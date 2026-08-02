"""Decisive cutoff probe for the C++ producer X-truncation at ref02.

(1) ref02 at the standard t0 AND at t0 - 30 d: if the cutoff sits at a
fixed ABSOLUTE time (~t0+205 d = 1.0646 yr) it is an orbit-file /
spline-domain clamp; if it stays at the same RELATIVE day it is an
index/geometry bound. (2) ref05 fingerprint (channel + cutoff fraction).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations


def run_case(label, ref, t0):
    dt, Nf, Nt = 10.0, 256, 12288
    edge, tk = 330, 307
    orb = ESAOrbits(force_backend="cpu")
    ws = WDMSettings(Nf, Nt, dt, t0=t0, min_freq=1e-4, max_freq=2e-2,
                     min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
                     force_backend="cpu")
    ch = GBWDMComputations(ws, t_ref=t0, Nt_sub=128, n_pad=16, N_sparse=256,
                           N_cp_sig=0, N_cp_orbit=0, orbits=orb,
                           tdi_config="2nd generation", force_backend="cpu",
                           d_d=0.0, tdi_type="XYZ", tukey_alpha=2.0 * tk / Nt)
    ch.convert_to_ra_dec = False
    sh = GBSignalHetComputations.for_band_engine(
        ch, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2)
    g = sh._g
    ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
    href = np.zeros((3, Nf, Nt))
    ch.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
    h_act = np.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
    nfa, nta = h_act.shape[1], h_act.shape[2]
    invC = np.zeros((3, 3, nfa, nta))
    for c in range(3):
        invC[c, c] = 1.0
    holder = proto._FullGridWDMHolder(h_act, invC)
    z0 = np.zeros(1, dtype=np.int32)
    sh.setup_in_model(holder, ref[None, :], z0)
    m_act = proto.m_active_for(ref[1], g)
    ml = np.asarray(m_act) - g["ind_min_f"]
    _, c1d = proto.kernel_c1_full(sh, ref)
    ic = len(ml) // 2
    print(f"\n[{label}] t0={t0/86400:.1f} d abs, f0={ref[1]*1e3:.4f} mHz")
    for c, chn in enumerate("XYZ"):
        w = c1d[c, ml[ic], :].real
        t = h_act[c, ml[ic], :]
        segs = np.array_split(np.arange(nta), 16)
        ratios = [float(np.dot(w[s], w[s]))
                  / max(float(np.dot(t[s], t[s])), 1e-300) for s in segs]
        cut = next((i for i in range(len(ratios))
                    if all(r < 0.5 for r in ratios[i:])), None)
        if cut is None:
            cf = "none"
        else:
            rel_d = cut / 16.0 * 364.0
            cf = (f"seg {cut} = day {rel_d:.0f} rel = "
                  f"{(t0/86400.0 + rel_d):.0f} d abs")
        print(f"  {chn}: cutoff {cf} | " +
              " ".join(f"{r:.2f}" for r in ratios))


def main():
    rng = np.random.default_rng(19)
    refs = []
    for _ in range(6):
        refs.append(np.array([
            10 ** rng.uniform(-22.5, -21.0),
            rng.uniform(1.5e-3, 1.5e-2),
            rng.uniform(0.0, 3e-16),
            0.0,
            rng.uniform(0, 2 * np.pi),
            np.arccos(rng.uniform(-1, 1)),
            rng.uniform(0, np.pi),
            rng.uniform(0, 2 * np.pi),
            np.arcsin(rng.uniform(-1, 1)),
        ]))
    dt = 10.0
    t0_std = int(0.5 * YRSID_SI / dt) * dt
    run_case("ref02 std t0", refs[2], t0_std)
    run_case("ref02 t0-30d", refs[2], t0_std - 30 * 86400.0)
    run_case("ref05 std t0", refs[5], t0_std)


if __name__ == "__main__":
    main()
