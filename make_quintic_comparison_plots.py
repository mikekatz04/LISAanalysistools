"""Compare the Lagrange vs quintic-spline PROJECTION interpolation in the LISA
direct response, on one GB signal. Produces:
  * quintic_vs_lagrange_projections.png  -- 6 arm projections: overlay + (quin-lag)
  * quintic_vs_lagrange_tdi_xyz.png       -- X/Y/Z TDI channels: overlay + (quin-lag)

Run with the project venv, e.g.:
  /Users/alessandrosantini/reps/globalift/erebor_org_setup/.venv/bin/python make_quintic_comparison_plots.py
"""

import sys
sys.modules["scienceplots"] = None  # pre-existing env workaround (caught ImportError)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.directresponse import pyResponseTDI
from lisatools.response.parallelbase import FastLISAResponseParallelModule

YRSID_SI = 31558149.763545603
DT = 10.0
T = 0.03                     # years (~95k samples; cheap on CPU)
T_BUFFER = 10000.0
LAM, BETA = 5.22979888, 0.98057429
GB = dict(A=1.084702251e-22, f=2.35962078e-3, fdot=1.47197271e-17,
          iota=1.11820901, phi0=4.91128699, psi=2.3290324)


class GBWave(FastLISAResponseParallelModule):
    @property
    def xp(self):
        return self.backend.xp

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        c2psi, s2psi, ci = np.cos(2 * psi), np.sin(2 * psi), np.cos(iota)
        fddot = 11.0 / 3.0 * fdot ** 2 / f
        ph = 2 * np.pi * (f * t + 0.5 * fdot * t ** 2 + (1 / 6) * fddot * t ** 3) - phi0
        hSp = -self.xp.cos(ph) * A * (1.0 + ci ** 2)
        hSc = -self.xp.sin(ph) * 2.0 * A * ci
        return (hSp * c2psi - hSc * s2psi) + 1j * (hSp * s2psi + hSc * c2psi)


def run(use_spline, h, orbits):
    r = pyResponseTDI(
        sampling_frequency=1.0 / DT, num_pts=len(h), orbits=orbits, order=25,
        tdi="1st generation", tdi_chan="XYZ", use_spline=use_spline, force_backend="cpu",
    )
    r.get_projections(h, LAM, BETA, t_buffer=T_BUFFER)
    proj = np.asarray(r.y_gw).copy()                 # (6, num_pts)
    xyz = [np.asarray(c).copy() for c in r.get_tdi_delays()]   # [X, Y, Z]
    return r, proj, xyz


def overlay_and_diff(axL, axR, t, a_lag, a_quin, zoom, valid, ylabel):
    # left: overlay on a zoom window; right: (quintic - lagrange) over valid window
    axL.plot(t[zoom], a_lag[zoom], lw=1.1, label="lagrange")
    axL.plot(t[zoom], a_quin[zoom], lw=1.1, ls="--", label="quintic")
    axL.set_ylabel(ylabel)
    axL.grid(alpha=0.3)
    d = a_quin[valid] - a_lag[valid]
    axR.plot(t[valid], d, lw=0.8, color="C3")
    scale = np.abs(a_lag[valid]).max()
    axR.set_title(f"quintic - lagrange  (max |Δ| = {np.abs(d).max():.2e},"
                  f" max |Δ|/max|y| = {np.abs(d).max() / scale:.2e})", fontsize=9)
    axR.grid(alpha=0.3)


def main():
    gen = GBWave(force_backend="cpu")
    h = np.asarray(gen(**GB, T=T, dt=DT))
    orbits = EqualArmlengthOrbits()
    orbits.configure(linear_interp_setup=True)

    rl, proj_l, xyz_l = run(False, h, orbits)
    rq, proj_q, xyz_q = run(True, h, orbits)

    n = rl.num_pts
    t = np.arange(n) * DT
    ps = rl.projections_start_ind
    ts = rl.tdi_start_ind
    proj_valid = slice(ps + 50, n - ps - 50)
    tdi_valid = slice(ts, n - ts)
    mid = n // 2
    zoom = slice(mid - 1000, mid + 1000)        # ~2000 samples (~47 GB periods)

    links = list(rl.response_orbits.LINKS)

    # ---- projections: 6 links x (overlay | diff) ----
    fig, axes = plt.subplots(6, 2, figsize=(15, 17), constrained_layout=True)
    fig.suptitle("LISA arm projections y_sr -- Lagrange vs quintic-spline interpolation\n"
                 f"GB f={GB['f']*1e3:.3f} mHz, dt={DT:.0f}s, T={T} yr  "
                 f"(left: overlay, zoom {t[zoom][0]/1e3:.0f}-{t[zoom][-1]/1e3:.0f} ks; right: difference, full valid span)",
                 fontsize=12)
    for i in range(6):
        overlay_and_diff(axes[i, 0], axes[i, 1], t, proj_l[i], proj_q[i],
                         zoom, proj_valid, ylabel=f"link {links[i]}")
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.savefig("quintic_vs_lagrange_projections.png", dpi=130)
    plt.close(fig)

    # ---- TDI XYZ: 3 channels x (overlay | diff) ----
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("TDI channels X/Y/Z -- Lagrange vs quintic-spline PROJECTION interpolation\n"
                 "(TDI delay interpolation itself stays Lagrange in both cases)",
                 fontsize=12)
    for i, name in enumerate("XYZ"):
        overlay_and_diff(axes[i, 0], axes[i, 1], t, xyz_l[i], xyz_q[i],
                         zoom, tdi_valid, ylabel=f"TDI {name}")
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.savefig("quintic_vs_lagrange_tdi_xyz.png", dpi=130)
    plt.close(fig)

    # ---- summary numbers ----
    print("Saved: quintic_vs_lagrange_projections.png, quintic_vs_lagrange_tdi_xyz.png")
    hmax = np.abs(h).max()
    for i in range(6):
        d = np.abs(proj_q[i][proj_valid] - proj_l[i][proj_valid]).max()
        sc = np.abs(proj_l[i][proj_valid]).max()
        print(f"  projection link {links[i]}: max|Δ|={d:.3e}  Δ/max|y|={d/sc:.2e}  Δ/|h|max={d/hmax:.2e}")
    for i, name in enumerate("XYZ"):
        d = np.abs(xyz_q[i][tdi_valid] - xyz_l[i][tdi_valid]).max()
        sc = np.abs(xyz_l[i][tdi_valid]).max()
        print(f"  TDI {name}: max|Δ|={d:.3e}  Δ/max|y|={d/sc:.2e}")


if __name__ == "__main__":
    main()
