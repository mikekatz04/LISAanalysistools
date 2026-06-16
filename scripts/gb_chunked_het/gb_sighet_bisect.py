"""Bisection debug for GBSignalHetComputations on the global-fit setup.

build_pack (the verified signal-het reference) uses ESAOrbits + a synthetic
injection + the FULL [1e-4,35e-3] band + Nt=2560. The mojito global-fit setup
differs on THREE axes at once -- L1Orbits, real data, and a NARROW per-source
band -- and the sighet get_ll collapses (d_h~0, h_h~0 => logL~-0.5 d_d).

This routes a CONTROLLED synthetic injection (cand=ref => logL must be ~0)
through the class and flips ONE axis at a time so we can see which breaks it:

  A  Nt=2560  full band   ESAOrbits   (== build_pack regime: must give logL~0)
  B  Nt=512   full band   ESAOrbits   (Nt regime: half_Nt < half_NS)
  C  Nt=2560  narrow band ESAOrbits   (per-source active-layer band)
  D  Nt=512   narrow band ESAOrbits   (both global-fit changes, synthetic data)

Each config builds the data WDM from the SAME synthetic GB so logL@inj must be
~0; whichever config first departs from 0 localizes the bug.
"""
import os, sys, gc
import numpy as np

import gbgpu  # noqa: F401  registers backend
from lisatools.detector import ESAOrbits
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbsignalhetcomputations import GBSignalHetComputations  # noqa: E402

DT = 10.0
NF = 1460
BACKEND = "cpu"
BAND_LAYERS = 15  # +/- layers for the "narrow" band


def make_synth(Nt, t_start, orbits, tdi_config, amp=6e-23):
    """Synthetic GB injection TD (3, Nf*Nt) + its 9-vector params at f0=14.22mHz."""
    Nobs = NF * Nt
    Tobs = Nt * NF * DT
    t_arr = np.arange(Nobs) * DT + t_start
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gen = GBTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / DT, 1,
                        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                        force_backend=BACKEND)
    p9 = np.array([amp, 14.22e-3, 1e-16, 0.0, 1.4, np.pi / 3.0, 0.7, 2.1, 0.5])
    sp = gen(*p9.reshape(9, 1), convert_to_ra_dec=False, return_spline=True)
    td = np.asarray(sp.eval_tdi(t_arr))
    td = (td[0] if td.ndim == 3 else td)[:3]
    return td, p9, Tobs


def run_config(tag, Nt, narrow, t_start):
    print(f"\n{'='*78}\n  CONFIG {tag}: Nt={Nt}  band={'narrow' if narrow else 'full'}  "
          f"half_Nt={Nt//2}  half_NS={1024//2}\n{'='*78}", flush=True)
    orbits = ESAOrbits(force_backend=BACKEND)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)
    td, p9, Tobs = make_synth(Nt, t_start, orbits, tdi_config)

    layer_df = 1.0 / (2.0 * NF * DT)
    f0 = p9[1]
    m_floor = int(np.floor(f0 / layer_df))
    if narrow:
        lof = (m_floor - BAND_LAYERS) * layer_df
        hif = (m_floor + BAND_LAYERS + 1) * layer_df
    else:
        lof, hif = 1e-4, 35e-3

    sighet = GBSignalHetComputations(
        td, p9, Nf=NF, Nt=Nt, dt=DT, t0=t_start, t_ref=t_start,
        orbits=orbits, tdi_config=tdi_config, min_freq=lof, max_freq=hif,
        tukey_alpha=0.0, force_backend=BACKEND)
    g = sighet._g
    print(f"  ind_min_f={g['ind_min_f']}  Nf_active={g['Nf_active']}  "
          f"Nt_active={g['Nt_active']}  stride={g['stride']}  "
          f"N_sparse_t={g['N_sparse_t']}  d_d={sighet.d_d:.6e}", flush=True)

    ll = float(np.asarray(sighet.get_ll(p9))[0])
    d_h = float(sighet.last_d_h[0]); h_h = float(sighet.last_h_h[0])
    dd = sighet.d_d
    print(f"  -> logL={ll:+.6e}   d_h={d_h:+.6e} (d_d={dd:.6e}, d_h/d_d={d_h/dd:+.4f})"
          f"   h_h={h_h:+.6e} (h_h/d_d={h_h/dd:+.4f})", flush=True)
    print(f"     EXPECT @inj: logL~0, d_h/d_d~1, h_h/d_d~1", flush=True)
    del sighet, td, orbits; gc.collect()
    return ll


def main():
    t_start = int(0.5 * YRSID_SI / DT) * DT
    only = os.environ.get("ONLY", "")
    configs = [("A", 2560, False), ("B", 512, False),
               ("C", 2560, True), ("D", 512, True)]
    for tag, Nt, narrow in configs:
        if only and tag not in only:
            continue
        try:
            run_config(tag, Nt, narrow, t_start)
        except Exception as e:
            import traceback
            print(f"  CONFIG {tag} RAISED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
