"""
Density study for in-kernel orbit-spline caching.

Goal: pick the smallest N_cp (control points per chunk) such that a cubic
spline through orbit values at N_cp uniformly-spaced times within a chunk
reproduces the LISATDI-on-the-fly orbit lookups (get_light_travel_times,
get_pos) to within target tolerance.

Why this matters: the C++ chunked-heterodyne kernel calls orbit lookups
~32-64 times per sparse-grid sample per binary. With N_sparse ~ 256 and
num_bin ~ thousands, every saved orbit eval is large. Caching a cubic
spline in shared memory (built once per chunk, eval'd cheaply per delay)
beats raw global-mem linear interp -- IF N_cp is small enough that the
spline coefs fit in shared (~ N_cp * 4 doubles per scalar series).

Target tolerance: 1e-3 m on x (positions, ~AU scale), 1e-12 s on L (light
travel times, ~16 s scale). That keeps orbit-induced phase error well
below 1e-7 rad even at 25 mHz.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from lisatools.utils.constants import YRSID_SI


def study(orbits_cls, label):
    print(f"\n== {label} ==")
    orbits = orbits_cls()
    # Build internal interp table with a generous span.
    # Match the chunked pipeline's setup: dt=10, span > 1 yr.
    dt = 10.0
    t_arr = np.arange(0.0, 1.05 * YRSID_SI, dt)
    try:
        orbits.configure(t_arr=t_arr, dt=dt, linear_interp_setup=True)
    except TypeError:
        orbits.configure(t_arr=t_arr)

    LINKS = orbits.LINKS
    SC = orbits.SC

    # Chunk widths to test (seconds). Half-day, day, week, month scales.
    chunk_widths_s = [
        0.5 * 86400,
        1.0 * 86400,
        7.0 * 86400,
        30.0 * 86400,
    ]

    # Pad chunk starts so we're not at t=0 edge.
    t0 = 0.25 * YRSID_SI

    N_cp_list = [4, 6, 8, 12, 16, 24, 32]

    for T_chunk in chunk_widths_s:
        # Dense ground-truth grid inside chunk.
        N_dense = 2048
        t_dense = np.linspace(0.0, T_chunk, N_dense)
        t_dense_abs = t0 + t_dense

        L_truth = {}
        for link in LINKS:
            L_truth[int(link)] = orbits.get_light_travel_times(
                t_dense_abs.copy(), int(link)
            )

        x_truth = {}
        for sc in SC:
            x_truth[int(sc)] = orbits.get_pos(t_dense_abs.copy(), int(sc))

        print(f"  T_chunk = {T_chunk/86400:5.2f} d")
        for N_cp in N_cp_list:
            t_cp = np.linspace(0.0, T_chunk, N_cp)
            t_cp_abs = t0 + t_cp

            max_L_err = 0.0
            for link in LINKS:
                L_cp = orbits.get_light_travel_times(t_cp_abs.copy(), int(link))
                cs = CubicSpline(t_cp, L_cp)
                err = np.max(np.abs(cs(t_dense) - L_truth[int(link)]))
                if err > max_L_err:
                    max_L_err = err

            max_x_err = 0.0
            for sc in SC:
                x_cp = orbits.get_pos(t_cp_abs.copy(), int(sc))
                xt = x_truth[int(sc)]
                # x has shape (N, 3) or (3, N) depending on backend
                if x_cp.shape[0] != t_cp.size:
                    x_cp = x_cp.T
                    xt = xt.T if xt.shape[0] != t_dense.size else xt
                for d in range(3):
                    cs = CubicSpline(t_cp, x_cp[:, d])
                    err = np.max(np.abs(cs(t_dense) - xt[:, d]))
                    if err > max_x_err:
                        max_x_err = err

            print(
                f"    N_cp={N_cp:3d}  maxLerr={max_L_err:.3e} s   maxXerr={max_x_err:.3e} m"
            )


if __name__ == "__main__":
    study(EqualArmlengthOrbits, "EqualArmlengthOrbits")
    try:
        study(ESAOrbits, "ESAOrbits")
    except Exception as e:
        print(f"ESAOrbits skipped: {e}")
