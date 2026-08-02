"""Does the PRODUCTION chunked config (N_cp_sig=48 control-point spline)
inherit the ref02-class unwrap slip? The scaffold's chunked (N_cp_sig=0,
dense eval) was clean at ref02; the cluster search runs N_cp_sig=48.
Compare chunked-48 get_ll at ref02 vs the dense-truth grid built by the
same dense path used everywhere as truth.
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

    def make(ncp_sig, ncp_orb):
        c = GBWDMComputations(
            wdm_set, t_ref=t_start, Nt_sub=256, n_pad=16, N_sparse=256,
            N_cp_sig=ncp_sig, N_cp_orbit=ncp_orb, orbits=orbits,
            tdi_config="2nd generation", force_backend=backend, d_d=0.0,
            tdi_type="XYZ", tukey_alpha=2.0 * tk / Nt)
        c.convert_to_ra_dec = False
        return c

    dense = make(0, 0)
    prod = make(48, 32)

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

    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    invC = np.ascontiguousarray(
        np.asarray(XYZ2SensitivityMatrix(wdm_set, model="scirdv1").invC),
        dtype=np.float64)
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})
    eng_d = WDMBandLikelihoodEngine(dense, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")
    eng_p = WDMBandLikelihoodEngine(prod, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")

    for rrr, ref in enumerate(refs):
        href = np.zeros((3, Nf, Nt))
        dense.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
        h_act = np.ascontiguousarray(
            href[:, ilo:ihi, wdm_set.active_slice_t])
        holder = proto._FullGridWDMHolder(h_act, invC)
        eng_d.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        hh_d = float(eng_d.h_h_out[0])
        dh_d = float(eng_d.d_h_out[0])
        eng_p.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        hh_p = float(eng_p.h_h_out[0])
        dh_p = float(eng_p.d_h_out[0])
        err = abs((dh_p - 0.5 * hh_p) - (dh_d - 0.5 * hh_d))
        print(f"[ref{rrr:02d}] beta={ref[8]:+.4f} "
              f"dense-chunked d_h={dh_d:.2f} h_h={hh_d:.2f} | "
              f"PROD chunked (ncp48) d_h={dh_p:.2f} h_h={hh_p:.2f} | "
              f"raw dLL diff {err:.3f}")


if __name__ == "__main__":
    main()
