"""Probe the ref02 stash defect (many-ref seed 19, 3rd draw).

Established (smoke_diag): at ref02 (f0=7.466 mHz, iota=2.700, min_env=0.002,
both-pattern-null sweep) the r==1 SELF-CHECK fails at 15,556 raw lnL --
h_h from the A/B stash fold is +22,510 on a true 33,233 (+68%), with the
reference folded against itself.  Candidate-independent, n_cp-independent.

Questions this probe answers:
  1. Is the PRODUCTION v2 engine's own h_h at the reference also inflated
     (stash bug in production) or only my mirror's reading of the stash?
  2. Does the direct dense contraction of the SAME producer's coefficients
     (kernel_c1_full(ref).c1_dense) match the chunked engine? (isolates
     stash CONTRACTION vs coefficient BUILD)
  3. WHERE does the excess live -- time profile per sparse pixel of
     [B0 + B0nc](summed c,d,m) vs the direct dense window sums, against the
     envelope null positions; per-layer split.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_ref02_probe.py
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
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine


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
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2)
    g = sighet._g
    N = g["n_sparse_fd"]
    eng_s = WDMBandLikelihoodEngine(sighet, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")
    eng_c = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")

    # many-ref draws, seed 19: refs 00-02 (00/01 = clean controls)
    rngr = np.random.default_rng(19)
    refs = []
    for _ in range(3):
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
    ref = refs[2]
    del ref

    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    invC = np.ascontiguousarray(
        np.asarray(XYZ2SensitivityMatrix(wdm_set, model="scirdv1").invC),
        dtype=np.float64)
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})

    for rrr, ref in enumerate(refs):
        print(f"\n[ref{rrr:02d}] f0={ref[1]*1e3:.4f} mHz fdot={ref[2]:.3e} "
              f"iota={ref[5]:.4f} psi={ref[6]:.4f} lam={ref[7]:.4f} "
              f"beta={ref[8]:+.4f}")
        href = np.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
        h_act = np.ascontiguousarray(
            href[:, ilo:ihi, wdm_set.active_slice_t])
        nta = h_act.shape[-1]
        holder = proto._FullGridWDMHolder(h_act, invC)
        eng_c.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        hh_chunk = float(eng_c.h_h_out[0])
        sighet.clear_in_model()
        sighet.setup_in_model(holder, ref[None, :], zeros)
        eng_s.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        hh_v2 = float(eng_s.h_h_out[0])
        dh_v2 = float(eng_s.d_h_out[0])
        print(f"  [eng ] chunked h_h={hh_chunk:.2f}; v2 h_h={hh_v2:.2f} "
              f"d_h={dh_v2:.2f} (raw dLL err at ref: "
              f"{abs((dh_v2 - 0.5 * hh_v2) - 0.5 * hh_chunk):.3f})")

        m_act = proto.m_active_for(ref[1], g)
        ml = np.asarray(m_act) - g["ind_min_f"]
        _, c1_dense = proto.kernel_c1_full(sighet, ref)
        w = c1_dense[:, ml, :].real
        t = h_act[:, ml, :]
        iw = invC[:, :, ml, :]

        d2 = float(np.einsum("cmj,cmj->", w - t, w - t))
        t2 = float(np.einsum("cmj,cmj->", t, t))
        w2 = float(np.einsum("cmj,cmj->", w, w))
        print(f"  [id  ] UNWEIGHTED |p|^2/|t|^2={w2/t2:.6f}  "
              f"|diff|^2/|t|^2={d2/t2:.3e}")
        for c, ch in enumerate("XYZ"):
            tc = float(np.einsum("mj,mj->", t[c], t[c]))
            dc = float(np.einsum("mj,mj->", w[c] - t[c], w[c] - t[c]))
            print(f"  [id  ]   {ch}: |t_c|^2 share {tc/t2:.3f}  "
                  f"per-ch diff/|t_c|^2 {dc/max(tc,1e-300):.3e}")
        cm_t = t.sum(axis=0) / np.sqrt(3.0)
        cm_p = w.sum(axis=0) / np.sqrt(3.0)
        cmt2 = float(np.einsum("mj,mj->", cm_t, cm_t))
        cmd2 = float(np.einsum("mj,mj->", cm_p - cm_t, cm_p - cm_t))
        print(f"  [cm  ] common-mode: |cm_t|^2/|t|^2={cmt2/t2:.3e}  "
              f"|cm_p-cm_t|^2/|t|^2={cmd2/t2:.3e}  "
              f"(spurious-cm / true-cm = {cmd2/max(cmt2,1e-300):.2f})")
        seg = np.array_split(np.arange(nta), 8)
        fr = [float(np.einsum("cmj,cmj->", (w - t)[:, :, s],
                              (w - t)[:, :, s])) / max(d2, 1e-300)
              for s in seg]
        print("  [prof] |diff|^2 fraction per year-eighth: "
              + " ".join(f"{x:.2f}" for x in fr))
        hh_p = float(np.einsum("cmj,cdmj,dmj->", w, iw, w))
        hh_t = float(np.einsum("cmj,cdmj,dmj->", t, iw, t))
        dh_p = float(np.einsum("cmj,cdmj,dmj->", w, iw, t))
        print(f"  [wt  ] WEIGHTED  h_h p={hh_p:.2f} t={hh_t:.2f} "
              f"overlap={dh_p:.2f}  raw dLL err at ref = "
              f"{abs((dh_p - 0.5 * hh_p) - 0.5 * hh_t):.3f}")
        C3 = iw[:, :, len(ml) // 2, nta // 2]
        ev, evec = np.linalg.eigh(C3)
        d2w_top = float(ev[-1] * np.einsum(
            "mj,mj->",
            np.einsum("c,cmj->mj", evec[:, -1], w - t),
            np.einsum("c,cmj->mj", evec[:, -1], w - t)))
        d2w = float(np.einsum("cmj,cdmj,dmj->", w - t, iw, w - t))
        print(f"  [wt  ] top-evec {np.round(evec[:, -1], 3)} "
              f"(ev ratio {ev[-1]/ev[0]:.1f}); diff_w along top: "
              f"{d2w_top/max(d2w,1e-300):.3f} of |diff|^2_w={d2w:.1f}")


if __name__ == "__main__":
    main()
