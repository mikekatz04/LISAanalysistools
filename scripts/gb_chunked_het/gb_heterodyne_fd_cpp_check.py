#!/usr/bin/env python
"""
Cross-check: C++ heterodyne FD GB TDI vs the Python prototype in
gb_heterodyne_fd.py.

The Python prototype has already been compared against the dense
lisatools TD->rfft reference (TD mismatch ~1e-6 at 256 pts/yr), so a tight
match between this C++ call and the Python prototype is sufficient to
certify the kernel.

Run:
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_heterodyne_fd_cpp_check.py
"""

from __future__ import annotations

import numpy as np
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly, GBFDTDIonTheFly


BACKEND = "cpu"


def python_prototype_fd(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
                        t_start, Tobs, t_ref, N_sparse, gb_kwargs, dt_factor):
    """Heterodyne FD via the explicit Python prototype: GBTDIonTheFly on
    sparse grid + manual carrier removal + numpy fft."""
    dt_sparse = Tobs / N_sparse
    t_sparse = t_start + np.arange(N_sparse) * dt_sparse

    gb = GBTDIonTheFly(t_sparse, Tobs, t_ref, dt_factor, 1, **gb_kwargs)
    out = gb(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
             convert_to_ra_dec=False, return_spline=False)
    tdi_amp   = np.asarray(out.tdi_amp[0])      # (3, N)
    tdi_phase = np.asarray(out.tdi_phase[0])    # (3, N)
    phi_ref   = np.asarray(out.phase_ref[0])    # (N,)

    df = 1.0 / Tobs
    k_f0 = int(round(float(f0[0]) / df))
    f0_grid = k_f0 * df

    tau = t_sparse - t_start
    carrier = 2.0 * np.pi * f0_grid * tau
    slow = tdi_amp * np.exp(+1j * (tdi_phase + phi_ref[None, :] - carrier[None, :]))
    S = np.fft.fft(slow, axis=-1) * dt_sparse
    X_het = 0.5 * S
    return X_het, k_f0, f0_grid


def main():
    dt = 15.0
    Tobs_nom = 1.0 * YRSID_SI
    N_dense = int(round(Tobs_nom / dt))
    Tobs = N_dense * dt

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start

    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=BACKEND)
    gb_kwargs = dict(tdi_config=tdi_config, orbits=orbits,
                     tdi_chan="XYZ", force_backend=BACKEND)

    # GB params (same as gb_heterodyne_fd.py defaults)
    amp = np.array([8.0e-23])
    f0 = np.array([20.0e-3])
    fdot = np.array([1.0e-14])
    fddot = np.array([0.0])
    phi0 = np.array([2.09802430298])
    inc = np.array([0.23984234])
    psi = np.array([1.234019814])
    lam = np.array([4.09808143])
    beta = np.array([0.090])

    # ---- dense reference for the end-to-end TD reconstruction check -----
    print("Building dense lisatools reference rfft ...")
    import time as _t
    t0 = _t.time()
    t_dense = t_start + np.arange(N_dense) * dt
    gb_dense = GBTDIonTheFly(t_dense, Tobs, t_ref, 1.0 / dt, 1, **gb_kwargs)
    out_d = gb_dense(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
                     convert_to_ra_dec=False, return_spline=False)
    x_dense = np.stack([np.asarray(out_d.X[0]).real,
                        np.asarray(out_d.Y[0]).real,
                        np.asarray(out_d.Z[0]).real], axis=0)
    print(f"  dense TDI: {_t.time()-t0:.2f}s")

    df = 1.0 / Tobs

    for N_sparse in (256, 512, 1024):
        # Python prototype
        X_py, k_py, f0g_py = python_prototype_fd(
            amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
            t_start, Tobs, t_ref, N_sparse, gb_kwargs, 1.0 / dt
        )

        # C++ kernel via the new wrapper
        fd_gen = GBFDTDIonTheFly(
            T=Tobs, t_ref=t_ref, N_sparse=N_sparse,
            num_sub=1, nchannels=3,
            tdi_config=tdi_config, orbits=orbits,
            tdi_chan="XYZ", force_backend=BACKEND,
        )
        tcpp = _t.time()
        X_cpp, k_cpp, f0g_cpp = fd_gen(
            amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
            t_start=t_start, convert_to_ra_dec=False
        )
        tcpp = _t.time() - tcpp
        X_cpp = np.asarray(X_cpp[0])  # (3, N_sparse)

        # Compare
        diff = X_cpp - X_py
        peak = np.max(np.abs(X_py))
        rel = np.max(np.abs(diff)) / (peak + 1e-300)
        print(f"N_sparse={N_sparse:5d}  k_f0_py={k_py}  k_f0_cpp={int(k_cpp[0])}  "
              f"f0_grid agree={abs(f0g_py-float(f0g_cpp[0]))<1e-15}")
        print(f"           max |X_cpp - X_py| = {np.max(np.abs(diff)):.3e}")
        print(f"           peak |X_py|        = {peak:.3e}")
        print(f"           relative max err   = {rel:.3e}")
        for c, name in enumerate("XYZ"):
            num = np.sum(X_cpp[c] * np.conj(X_py[c]))
            da = np.sqrt(np.sum(np.abs(X_cpp[c]) ** 2))
            db = np.sqrt(np.sum(np.abs(X_py[c]) ** 2))
            ov = np.real(num) / (da * db + 1e-300)
            print(f"           overlap[{name}] = {ov:.15f}  mismatch = {1-ov:.3e}")

        # ---- end-to-end TD check via the C++ output --------------------
        n_rfft = N_dense // 2 + 1
        X_full = np.zeros((3, n_rfft), dtype=complex)
        m = np.fft.fftfreq(N_sparse, d=1.0 / N_sparse).astype(int)
        kbins = int(k_cpp[0]) + m
        mask = (kbins >= 0) & (kbins < n_rfft)
        for c in range(3):
            X_full[c, kbins[mask]] = X_cpp[c, mask]
        x_recon = np.fft.irfft(X_full, n=N_dense, axis=-1) / dt
        i0 = int(0.05 * N_dense); i1 = N_dense - i0
        mm = []
        for c in range(3):
            num = np.sum(x_recon[c, i0:i1] * x_dense[c, i0:i1])
            da = np.sqrt(np.sum(x_recon[c, i0:i1] ** 2))
            db = np.sqrt(np.sum(x_dense[c, i0:i1] ** 2))
            mm.append(1.0 - num / (da * db + 1e-300))
        print(f"           TD mismatch vs dense rfft (cpp_path): "
              f"X={mm[0]:.3e} Y={mm[1]:.3e} Z={mm[2]:.3e}")
        print(f"           kernel time = {tcpp*1e3:.1f} ms")


if __name__ == "__main__":
    main()
