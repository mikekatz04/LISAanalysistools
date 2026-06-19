#!/usr/bin/env python
"""Validate the C++ moving-window SOBBH signal-het get_ll on a CHIRPING source.

The C++ kernel's active-m-band array was enlarged (SOBBH_SIGHET_MAX_M) and the
frontend auto-sizes m_active_half_width from the reference's carrier sweep, so
the reference-guided band now covers the chirp. This checks that
SOBBHSignalHetComputations.get_ll (the C++ kernel) recovers the dense lisatools
logL for a year-long chirping SOBBH -- the regime where the old fixed 5-layer
band failed by ~7.5 logL.

Run::  F_LOW_HZ=0.012 python sobbh_signal_het_moving_cpp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from sobbhsignalhetcomputations import SOBBHSignalHetComputations  # noqa: E402


def main():
    backend = "cpu"
    dt = 10.0
    Nf = int(os.environ.get("NF", "1460"))
    Nt = int(os.environ.get("NT", "2560"))
    Nobs = Nf * Nt
    EC = 20
    f_low = float(os.environ.get("F_LOW_HZ", "0.012"))
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nobs * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                           tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                           force_backend=backend)

    def td(p):
        sp = gen(*np.asarray(p, float).reshape(11, 1), convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)
    wdm = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
                      min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                      is_complex=False, force_backend=backend)
    layer_df = wdm.layer_df

    ref = np.array([60.0, 55.0, 0.1, 0.2, 4.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = td(ref)
    data_r = TDSignal(td_inj, settings=td_set).transform(wdm, window=window)
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_r), sens)
    analysis.signal_gen = lambda *p: TDSignal(td(np.asarray(p, float).reshape(11)),
                                              settings=td_set).transform(wdm, window=window)
    snr = float(analysis.snr())
    print(f"[inj] Tobs={Tobs/YRSID_SI:.2f}yr f_low={f_low*1e3:.3f}mHz SNR={snr:.1f}", flush=True)

    n_sparse_fd = int(os.environ.get("N_SPARSE_FD", "1024"))
    sighet = SOBBHSignalHetComputations(
        td_inj, ref, Nf=Nf, Nt=Nt, dt=dt, t0=t_start, t_ref=t_start,
        orbits=orbits, tdi_config=tdi_config, min_freq=1e-4, max_freq=35e-3,
        n_sparse_fd=n_sparse_fd, force_backend=backend)

    print(f"\n   {'pert':>12s} {'logL_dense':>13s} {'logL_sighet':>13s} {'|diff|':>11s}",
          flush=True)
    worst0 = None
    for label, delta in [("zero", 0.0), ("df0+0.05lf", 0.05 * layer_df),
                         ("df0+0.2lf", 0.2 * layer_df), ("df0+1lf", 1.0 * layer_df)]:
        p = ref.copy(); p[5] = f_low + delta
        ll_d = float(analysis.calculate_signal_likelihood(*p, source_only=True))
        ll_s = float(np.asarray(sighet.get_ll(p)).reshape(()))
        diff = abs(ll_s - ll_d)
        if delta == 0.0:
            worst0 = diff
        print(f"   {label:>12s} {ll_d:+13.4e} {ll_s:+13.4e} {diff:11.3e}", flush=True)

    ok = worst0 is not None and worst0 < 1e-2
    print(f"\n[summary] |sighet-dense| at injection = {worst0:.3e}", flush=True)
    print("PASS" if ok else "CHECK",
          "(C++ moving-window sig-het recovers dense logL for a chirping SOBBH)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
