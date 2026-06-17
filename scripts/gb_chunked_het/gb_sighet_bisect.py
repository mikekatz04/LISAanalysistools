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


def build_sighet(Nt, narrow, t_start, orbits, tdi_config, td, p9, *,
                 n_sparse_fd=1024, nt_layer=64, m_half=2, tukey_alpha=0.0):
    layer_df = 1.0 / (2.0 * NF * DT)
    f0 = p9[1]
    m_floor = int(np.floor(f0 / layer_df))
    if narrow:
        lof = (m_floor - BAND_LAYERS) * layer_df
        hif = (m_floor + BAND_LAYERS + 1) * layer_df
    else:
        lof, hif = 1e-4, 35e-3
    return GBSignalHetComputations(
        td, p9, Nf=NF, Nt=Nt, dt=DT, t0=t_start, t_ref=t_start,
        orbits=orbits, tdi_config=tdi_config, min_freq=lof, max_freq=hif,
        n_sparse_fd=n_sparse_fd, nt_layer=nt_layer, m_active_half_width=m_half,
        tukey_alpha=tukey_alpha, force_backend=BACKEND)


def eval_floor(sighet, p9):
    """Return (logL, d_h/d_d, h_h/d_d, eff_mm) at cand=ref. eff_mm = -2 logL/d_d
    is the effective full-band mismatch the sig-het floor corresponds to."""
    ll = float(np.asarray(sighet.get_ll(p9))[0])
    d_h = float(sighet.last_d_h[0]); h_h = float(sighet.last_h_h[0])
    dd = sighet.d_d
    return ll, d_h / dd, h_h / dd, -2.0 * ll / dd, dd


def run_config(tag, Nt, narrow, t_start):
    print(f"\n{'='*78}\n  CONFIG {tag}: Nt={Nt}  band={'narrow' if narrow else 'full'}  "
          f"half_Nt={Nt//2}  half_NS={1024//2}\n{'='*78}", flush=True)
    orbits = ESAOrbits(force_backend=BACKEND)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)
    td, p9, Tobs = make_synth(Nt, t_start, orbits, tdi_config)
    sighet = build_sighet(Nt, narrow, t_start, orbits, tdi_config, td, p9)
    g = sighet._g
    print(f"  ind_min_f={g['ind_min_f']}  Nf_active={g['Nf_active']}  "
          f"Nt_active={g['Nt_active']}  stride={g['stride']}  "
          f"N_sparse_t={g['N_sparse_t']}  d_d={sighet.d_d:.6e}", flush=True)
    ll, dh_r, hh_r, eff_mm, dd = eval_floor(sighet, p9)
    print(f"  -> logL={ll:+.6e}   d_h/d_d={dh_r:+.4f}  h_h/d_d={hh_r:+.4f}  "
          f"eff_mm={eff_mm:.3e}", flush=True)
    del sighet, td, orbits; gc.collect()
    return ll


def run_sweep(Nt, t_start):
    """Dig into the Nt=512 floor: sweep n_sparse_fd, nt_layer, m_half on the
    narrow-band synthetic source and report eff_mm (the floor as a mismatch)."""
    print(f"\n{'#'*78}\n  FLOOR SWEEP  Nt={Nt}  half_Nt={Nt//2}  narrow band\n{'#'*78}", flush=True)
    orbits = ESAOrbits(force_backend=BACKEND)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)
    td, p9, Tobs = make_synth(Nt, t_start, orbits, tdi_config)
    base = dict(n_sparse_fd=1024, nt_layer=64, m_half=2)
    cells = []
    for nsfd in [256, 512, 1024, 2048]:
        cells.append(("n_sparse_fd", nsfd, {**base, "n_sparse_fd": nsfd}))
    for ntl in [32, 64, 128, 256]:
        cells.append(("nt_layer", ntl, {**base, "nt_layer": ntl}))
    for mh in [2, 3, 4, 6]:
        cells.append(("m_half", mh, {**base, "m_half": mh}))
    print(f"  {'knob':>14} {'val':>6} {'half_NS':>8} {'stride':>7} | "
          f"{'logL':>12} {'d_h/d_d':>9} {'h_h/d_d':>9} {'eff_mm':>10}", flush=True)
    for knob, val, kw in cells:
        try:
            sh = build_sighet(Nt, True, t_start, orbits, tdi_config, td, p9, **kw)
            ll, dh_r, hh_r, eff_mm, dd = eval_floor(sh, p9)
            print(f"  {knob:>14} {val:>6} {kw['n_sparse_fd']//2:>8} {sh._g['stride']:>7} | "
                  f"{ll:>+12.4e} {dh_r:>9.4f} {hh_r:>9.4f} {eff_mm:>10.3e}", flush=True)
            del sh; gc.collect()
        except Exception as e:
            print(f"  {knob:>14} {val:>6}  RAISED {type(e).__name__}: {e}", flush=True)
    del td, orbits; gc.collect()


def main():
    t_start = int(0.5 * YRSID_SI / DT) * DT
    only = os.environ.get("ONLY", "")
    if os.environ.get("SWEEP", "0") == "1":
        for Nt in [int(x) for x in os.environ.get("SWEEP_NT", "512,2560").split(",")]:
            run_sweep(Nt, t_start)
        print("\nDONE.", flush=True)
        return
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
