"""Three-way GB waveform arbitration: A (spline TDI-on-the-fly) vs B (chunked
WDM fill_global) vs C (plain hp/hc through the installed direct response).

Context: at ref02 (seed-19 draw 3, f0=7.4656 mHz, near-ecliptic) paths A and B
disagree by ~40% of X-channel power over days ~170-364; at ref00 they agree at
~1e-9.  TRUTH ruling: full time-domain generation at dt steps.  Path C is the
fastlisaresponse-heritage GBWave (lifted from LATW tutorial 05) fed through the
installed ResponseWrapper/pyResponseTDI with TDIConfig("2nd generation") +
ESAOrbits.  Calibrate C against A on ref00 (gate: mid-year rel L2 < 1e-3),
then judge ref02.

Scaffold: grid/sources copied verbatim from gb_sighet_ref02_probe.py; path A
spline build via proto.build_spl + the TDIOutput class's own eval_tdi; WDM
transform of C via the TDSignal(...).transform pattern from
gb_test_script_td_wave.py.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_tdtruth_threeway.py
Out: ./ratio_proto_out_fix/tdtruth_threeway.npz + tdtruth_threeway.png
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto  # build_spl + scaffold conventions

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings, TDSettings, TDSignal
from lisatools.utils.constants import YRSID_SI
from lisatools.response.directresponse import ResponseWrapper
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ratio_proto_out_fix")
DAY = 86400.0


# ---------------------------------------------------------------------------
# Path C waveform: GBWave lifted from LATW tutorials/05_ResponseAndTDI.ipynb
# (the fastlisaresponse-heritage generator), with three flagged adjustments to
# match the path-A kernel (GBTDIonTheFly::ucb_phase/ucb_amplitude,
# GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu lines 78-100):
#   1. signature takes the 9-param order minus sky: (amp, f0, fdot, fddot,
#      phi0, iota, psi) -- fddot is a free parameter (refs use 0), NOT the
#      tutorial's 11/3*fdot^2/f astro rule;
#   2. amplitude evolution A*(1 + 2/3*(fdot/f0)*t) as in ucb_amplitude;
#   3. fixed sample count n instead of arange(0, T*YRSID_SI, dt) (the C++
#      response wrap REQUIRES waveform length == num_pts; the garbage margins
#      own the edge handling, and remove_garbage='zero' blanks them).
# The time grid is RELATIVE (t=0 at the first sample); with t0 = t_ref =
# t_start this equals t_abs - t_ref, exactly the kernel's t_diff.
# phi0_sign=-1.0 reproduces the tutorial/kernel "- phi0"; +1.0 is the sweep
# alternative.
# ---------------------------------------------------------------------------
class GBWave:
    def __init__(self, n, pad=0, phi0_sign=-1.0):
        self.n, self.pad, self.phi0_sign = int(n), int(pad), float(phi0_sign)

    def __call__(self, amp, f0, fdot, fddot, phi0, iota, psi, T=1.0, dt=10.0):
        t = np.arange(self.n + self.pad) * dt
        cos2psi, sin2psi, cosi = np.cos(2 * psi), np.sin(2 * psi), np.cos(iota)
        phase = (2 * np.pi * (f0 * t + 0.5 * fdot * t ** 2
                              + fddot * t ** 3 / 6.0)
                 + self.phi0_sign * phi0)
        amp_t = amp * (1.0 + (2.0 / 3.0) * (fdot / f0) * t)
        hSp = -np.cos(phase) * amp_t * (1.0 + cosi * cosi)
        hSc = -np.sin(phase) * 2.0 * amp_t * cosi
        hp = hSp * cos2psi - hSc * sin2psi
        hx = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hx


def run_path_C(ref, *, n, dt, t_start, orbits, phi0_sign, flip_hx):
    """Plain GB hp/hc through the installed direct response.  Output on the
    grid t_start + arange(n)*dt with the first/last tdi_start_ind samples
    zeroed (remove_garbage='zero')."""
    wrap = ResponseWrapper(
        GBWave(n, phi0_sign=phi0_sign),
        n * dt / YRSID_SI, dt, 7, 8,       # gen, Tobs(yr), dt, idx_lam, idx_beta
        t0=t_start,                        # absolute start (orbits frame time)
        flip_hx=flip_hx,
        remove_sky_coords=True,
        is_ecliptic_latitude=True,
        remove_garbage="zero",             # keep grid alignment with path A
        n_overide=n,
        orbits=orbits,
        force_backend="cpu",
        order=25,
        tdi="2nd generation",
        tdi_chan="XYZ",
    )
    X, Y, Z = wrap(*ref)                   # sky consumed in orbits frame
    return np.asarray([X, Y, Z]), wrap.response_model.tdi_start_ind


def rel_l2(diff, refr):
    return float(np.sum(diff * diff) / max(np.sum(refr * refr), 1e-300))


def window_ratios(a, b, t_rel, windows):
    """Per-channel |a-b|^2 / |a|^2 over each (lo, hi) window of t_rel."""
    out = {}
    for name, (lo, hi) in windows.items():
        m = (t_rel >= lo) & (t_rel < hi)
        out[name] = [rel_l2(a[c][..., m] - b[c][..., m], a[c][..., m])
                     for c in range(3)]
    return out


def main():
    t_wall = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- scaffold: copied verbatim from gb_sighet_ref02_probe.py -----------
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
    gen = sighet._keep_alive["gb_gen"]     # THE path-A generator

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
    assert abs(refs[0][1] * 1e3 - 13.9992) < 5e-4, refs[0][1]
    assert abs(refs[2][1] * 1e3 - 7.4656) < 5e-4, refs[2][1]
    for i, r in enumerate(refs):
        print(f"[ref{i:02d}] f0={r[1]*1e3:.4f} mHz fdot={r[2]:.3e} "
              f"iota={r[5]:.4f} psi={r[6]:.4f} lam={r[7]:.4f} "
              f"beta={r[8]:+.4f}", flush=True)

    Nobs = Nf * Nt
    t_arr = np.arange(Nobs) * dt + t_start           # absolute grid
    t_rel = t_arr - t_start
    td_set = TDSettings(Nobs, dt, t0=t_start, force_backend=backend)
    tukey = signal.windows.tukey(Nobs, alpha=2.0 * tk / Nt)  # matches WDM edge
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    tsl = wdm_set.active_slice_t
    # WDM active time-layer centers, seconds relative to t_start
    lay_t = (np.arange(Nt) + 0.5) * Nf * dt
    lay_t_act = lay_t[tsl]
    act_shape = (3, ihi - ilo, len(lay_t_act))

    def as_active(arr):
        """Normalize to the ONE wdm_set active-region convention.
        fill_global_wdm fills the full (3, Nf, Nt) grid -> slice it once with
        [:, ind_min_f:ind_max_f+1, active_slice_t]; TDSignal.transform already
        returns the active region -> pass through (never slice twice)."""
        arr = np.asarray(arr)
        if arr.shape == (3, Nf, Nt):
            arr = arr[:, ilo:ihi, tsl]
        assert arr.shape == act_shape, (arr.shape, act_shape)
        return arr

    def paths_AB(ref):
        # A: spline TD via the installed generator + TDIOutput's own eval_tdi
        spl = proto.build_spl(gen, ref)
        td_A = np.asarray(spl.eval_tdi(t_arr))[0]        # (3, Nobs)
        # B: chunked fill_global_wdm real-coefficient grid (full -> active)
        href = np.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
        return td_A, as_active(href)

    def wdm_of(td):
        # exact TDSignal->transform pattern from gb_test_script_td_wave.py
        return as_active(
            TDSignal(td, settings=td_set).transform(wdm_set, window=tukey).arr)

    # ================= CALIBRATION GATE on ref00 ==========================
    print("\n=== CALIBRATION (ref00) ===", flush=True)
    ref0 = refs[0]
    td_A0, href_B0 = paths_AB(ref0)

    mid = (t_rel >= 60 * DAY) & (t_rel < 300 * DAY)      # mid-year, no edges
    combos = [(-1.0, False), (1.0, False), (-1.0, True), (1.0, True)]
    cal = []
    td_C0 = garb = None
    win_combo = None
    for phi0_sign, flip_hx in combos:
        t0c = time.time()
        td_C, garb = run_path_C(ref0, n=Nobs, dt=dt, t_start=t_start,
                                orbits=orbits, phi0_sign=phi0_sign,
                                flip_hx=flip_hx)
        mm = [rel_l2(td_C[c][mid] - td_A0[c][mid], td_C[c][mid])
              for c in range(3)]
        cal.append((phi0_sign, flip_hx, mm))
        print(f"  combo phi0_sign={phi0_sign:+.0f} flip_hx={flip_hx}: "
              f"mid-year relL2 X={mm[0]:.3e} Y={mm[1]:.3e} Z={mm[2]:.3e} "
              f"({time.time()-t0c:.0f}s)", flush=True)
        if max(mm) < 1e-3:
            win_combo = (phi0_sign, flip_hx)
            td_C0 = td_C
            break
    if win_combo is None:
        raise RuntimeError(f"calibration gate FAILED for all combos: {cal}")
    print(f"  WINNER: phi0_sign={win_combo[0]:+.0f} flip_hx={win_combo[1]}",
          flush=True)

    # WDM side of the gate: C transformed to WDM vs B, active region
    wdm_C0 = wdm_of(td_C0)                 # already active-region normalized
    wdm_gate0 = [rel_l2(wdm_C0[c] - href_B0[c], wdm_C0[c]) for c in range(3)]
    print(f"  WDM gate C-vs-B (active region): X={wdm_gate0[0]:.3e} "
          f"Y={wdm_gate0[1]:.3e} Z={wdm_gate0[2]:.3e}", flush=True)
    # full-window TD C-vs-A on ref00 (excluding response garbage edges)
    ok = slice(garb, Nobs - garb)
    td_gate0 = [rel_l2(td_C0[c][ok] - td_A0[c][ok], td_C0[c][ok])
                for c in range(3)]
    print(f"  TD  gate C-vs-A (full, no edges): X={td_gate0[0]:.3e} "
          f"Y={td_gate0[1]:.3e} Z={td_gate0[2]:.3e}", flush=True)

    # ================= VERDICT on ref02 ===================================
    print("\n=== VERDICT (ref02) ===", flush=True)
    ref2 = refs[2]
    td_A2, href_B2 = paths_AB(ref2)
    td_C2, garb = run_path_C(ref2, n=Nobs, dt=dt, t_start=t_start,
                             orbits=orbits, phi0_sign=win_combo[0],
                             flip_hx=win_combo[1])
    wdm_C2 = wdm_of(td_C2)
    wdm_A2 = wdm_of(td_A2)

    windows_td = {"d0-170": (garb * dt, 170 * DAY),
                  "d170-364": (170 * DAY, (Nobs - garb) * dt)}
    windows_wdm = {"d0-170": (0.0, 170 * DAY),
                   "d170-364": (170 * DAY, 365 * DAY)}

    r_CA_td = window_ratios(td_C2, td_A2, t_rel, windows_td)
    B_act = href_B2                        # all three already active-region
    C_act = wdm_C2
    A_act = wdm_A2
    r_CB_wdm = window_ratios(C_act, B_act, lay_t_act, windows_wdm)
    r_CA_wdm = window_ratios(C_act, A_act, lay_t_act, windows_wdm)
    r_AB_wdm = window_ratios(A_act, B_act, lay_t_act, windows_wdm)

    def show(tag, r):
        for w, v in r.items():
            print(f"  {tag} [{w}]: X={v[0]:.3e} Y={v[1]:.3e} Z={v[2]:.3e}",
                  flush=True)

    show("TD  |C-A|^2/|C|^2", r_CA_td)
    show("WDM |C-B|^2/|C|^2", r_CB_wdm)
    show("WDM |C-A|^2/|C|^2", r_CA_wdm)
    show("WDM |A-B|^2/|A|^2", r_AB_wdm)

    # verdict logic on the X channel, discrepancy window
    xa = r_CA_wdm["d170-364"][0]
    xb = r_CB_wdm["d170-364"][0]
    xab = r_AB_wdm["d170-364"][0]
    if min(xa, xb) < 0.1 * max(xa, xb) and min(xa, xb) < 1e-2:
        winner = "A" if xa < xb else "B"
        loser = "B" if winner == "A" else "A"
        verdict = (f"path {winner} MATCHES C (X, d170-364: "
                   f"{min(xa, xb):.3e}); path {loser} deviates at "
                   f"{max(xa, xb):.3e} (A-vs-B: {xab:.3e})")
    else:
        verdict = (f"NO clean match: C-A={xa:.3e}, C-B={xb:.3e}, "
                   f"A-B={xab:.3e} (X, d170-364) -- reporting pairwise only")
    print(f"\nVERDICT: {verdict}", flush=True)

    # ================= outputs ============================================
    np.savez(
        os.path.join(OUT_DIR, "tdtruth_threeway.npz"),
        refs=np.asarray(refs), win_combo=np.asarray(win_combo, dtype=object),
        cal=np.asarray(cal, dtype=object),
        td_gate0=td_gate0, wdm_gate0=wdm_gate0,
        r_CA_td_w1=r_CA_td["d0-170"], r_CA_td_w2=r_CA_td["d170-364"],
        r_CB_wdm_w1=r_CB_wdm["d0-170"], r_CB_wdm_w2=r_CB_wdm["d170-364"],
        r_CA_wdm_w1=r_CA_wdm["d0-170"], r_CA_wdm_w2=r_CA_wdm["d170-364"],
        r_AB_wdm_w1=r_AB_wdm["d0-170"], r_AB_wdm_w2=r_AB_wdm["d170-364"],
        verdict=verdict,
        # decimated ref02 X-channel traces for offline inspection
        t_days=t_rel[::64] / DAY, X_A2=td_A2[0][::64], X_C2=td_C2[0][::64],
        garb=garb)

    # --- figure: residual trace + per-window bar summary ------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_LINE, C_A, C_B = "#4269d0", "#efb118", "#ff725c"  # blue / orange / red
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
    fig.suptitle("ref02 three-way X channel: C = TD truth via direct response",
                 fontsize=11)

    dsl = slice(garb, Nobs - garb, 16)
    ax1.plot(t_rel[dsl] / DAY, (td_C2[0] - td_A2[0])[dsl], lw=0.5,
             color=C_LINE)
    ax1.axvline(170, color="0.6", lw=1, ls="--")
    ax1.text(171, 0.9, "day 170", transform=ax1.get_xaxis_transform(),
             fontsize=8, color="0.4")
    ax1.set_xlabel("days since t_start")
    ax1.set_ylabel("X_C - X_A")
    ax1.set_title("time-domain residual, path C minus path A", fontsize=9)
    ax1.grid(alpha=0.2)

    labels = ["C-A d0-170", "C-A d170-364", "C-B d0-170", "C-B d170-364"]
    vals = [r_CA_wdm["d0-170"][0], r_CA_wdm["d170-364"][0],
            r_CB_wdm["d0-170"][0], r_CB_wdm["d170-364"][0]]
    cols = [C_A, C_A, C_B, C_B]
    bars = ax2.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2e}",
                 ha="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"X: $|{\rm diff}|^2 / |X_C|^2$ (WDM)")
    ax2.set_title("per-window X-power mismatch vs path C (WDM grid)",
                  fontsize=9)
    ax2.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tdtruth_threeway.png"), dpi=150)
    print(f"\nsaved {OUT_DIR}/tdtruth_threeway.npz + .png "
          f"({time.time()-t_wall:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
