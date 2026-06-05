#!/usr/bin/env python
"""Compare GBTDIonTheFly numerical derivatives (mirroring the C-side
``fast_wdm_inner`` in lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu)
against spline derivatives evaluated from a TDTDIOutput.

The C kernel evaluates the GB TDI carrier at ``tn ± deriv_delta_t`` (and at
``tn``), reconstructs the per-channel TDI phase
``tdi_phase = -arg(M_raw * exp(i * phase_ref))`` from raw outputs, applies a
``±pi`` wrap of ``tdi_phase_{up,down} - tdi_phase_mid``, and forms a 3-point
central difference for the per-channel frequency
``f[i] = residual_frequency + tdi_frequency`` where
    residual_frequency = d(phase_ref)/dt / (2 pi)
    tdi_frequency      = d(tdi_phase)/dt / (2 pi).

This script:
  (1) Builds a sparse-grid TDTDIOutput via GBTDIonTheFly(..., return_spline=True)
      so we have ``tdi_phase_spl`` and ``phase_ref_spl``.
  (2) For each test time it calls GBTDIonTheFly directly at the 5-point stencil
      ``[t-2h, t-h, t, t+h, t+2h]`` and forms both 3-point and 5-point central
      differences for the first AND second derivatives.
  (3) Compares against the spline first derivative (truth-ish, since the spline
      is fit to dense GB-TDI samples). The cubic-spline second derivative is
      piecewise linear and is included only as a reference curve.
  (4) Stabilises the second derivative with Richardson extrapolation
      ``fdot_R = (4 * fdot[h/2] - fdot[h]) / 3``.

Env knobs:
  N_SPARSE    sparse-grid length for the spline fit          (default 4096)
  N_TEST      number of test anchors                         (default 30)
  H_LIST      comma-separated stencil half-widths in seconds (default 50,100,200,500,1000,2000)
"""

import os
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

backend = "cpu"
xp = np

# ----------------------------------------------------------------------
# Setup (mirrors gb_lookup_table_test_script.py / compare_inputs_current.py)
# ----------------------------------------------------------------------
orbits = ESAOrbits(force_backend=backend)
dt = 10.0
Nt = 256 * 10
Nf = 1460
wavelet_duration = Nf * dt
Tobs = Nt * wavelet_duration
Nobs = Nf * Nt
tdi_config = TDIConfig("2nd generation")
t_start = int(0.5 * YRSID_SI / dt) * dt
t_ref = t_start
t_arr_full = np.arange(Nobs) * dt + t_start

amp_v   = np.full(1, 8.0e-22)
f0_v    = np.full(1, 3.0e-3)
fdot_v  = np.full(1, 1.0e-16)
fddot_v = np.full(1, 0.0)
phi0_v  = np.full(1, 2.09802430298)
inc_v   = np.full(1, 0.23984234)
psi_v   = np.full(1, 1.234019814)
lam_v   = np.full(1, 4.09808143)
beta_v  = np.full(1, 1.1)
params  = np.array([amp_v, f0_v, fdot_v, fddot_v, phi0_v, inc_v, psi_v, lam_v, beta_v]).T

gb_tdi_kwargs = dict(
    tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
    force_backend=backend,
)

# ----------------------------------------------------------------------
# Sparse-grid TDTDIOutput (provides the spline derivatives we compare to)
# ----------------------------------------------------------------------
N_sparse = int(os.environ.get("N_SPARSE", 4096))
t_sparse = np.linspace(t_arr_full[0], t_arr_full[-1], N_sparse)
gb_gen_spline = GBTDIonTheFly(
    t_sparse, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs
)
wave_spline = gb_gen_spline(*params.T, convert_to_ra_dec=False, return_spline=True)

# Test anchors well inside the observation window
N_test = int(os.environ.get("N_TEST", 30))
t_lo = t_arr_full[0] + 0.1 * Tobs
t_hi = t_arr_full[0] + 0.9 * Tobs
t_test = np.linspace(t_lo, t_hi, N_test)

# ----------------------------------------------------------------------
# Raw GBTDIonTheFly evaluations at arbitrary times.
# Returns (tdi_phase[3, N], phase_ref[N], tdi_amp[3, N]) for num_sub=1.
# ----------------------------------------------------------------------
def raw_values_at(times):
    t_1d = np.asarray(times, dtype=np.float64)
    gen = GBTDIonTheFly(t_1d, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)
    out = gen(*params.T, convert_to_ra_dec=False, return_spline=False)
    return (
        np.asarray(out.tdi_phase[0]),   # (3, N)
        np.asarray(out.phase_ref[0]),   # (N,)
        np.asarray(out.tdi_amp[0]),     # (3, N)
    )


def wrap_pi(d):
    """Wrap a phase difference into (-pi, pi].  Matches the C-side wrap of
    ``tdi_phase_{up,down} - tdi_phase_mid``."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def numerical_deriv_at(t_anchor, h):
    """Numerical first/second derivative of (tdi_phase + phase_ref)/(2pi) at
    each anchor t_anchor[k], using the 5-point stencil [-2h, -h, 0, +h, +2h]
    around each anchor.  Returns (f1_3, f1_5, fdot_3, fdot_5), each of shape
    (3, N_anchor), in Hz and Hz/s respectively.  The 3-point form is the one
    fast_wdm_inner uses; the 5-point form is the more accurate stencil we
    extend to here for the second derivative."""
    N = len(t_anchor)
    offsets = np.array([-2.0 * h, -h, 0.0, h, 2.0 * h])
    t_stencil = (t_anchor[:, None] + offsets[None, :]).reshape(-1)
    tdi_phase_5, phase_ref_5, _ = raw_values_at(t_stencil)
    tdi_phase_5 = tdi_phase_5.reshape(3, N, 5)
    phase_ref_5 = phase_ref_5.reshape(N, 5)

    # tdi_phase differences are wrapped to (-pi, pi] vs. the central anchor
    # (mirrors the C-side ``dphi_{up,down} = wrap(tdi_phase_{up,down} -
    # tdi_phase_mid)`` correction).  phase_ref is the continuous carrier --
    # it is built as ``-phi0 + 2 pi (f0 t + ...)`` in GBTDIonTheFly::ucb_phase
    # and is *not* wrapped by the C kernel, so we use raw differences here.
    mid_tdi = tdi_phase_5[..., 2:3]
    dphi_tdi = wrap_pi(tdi_phase_5 - mid_tdi)    # (3, N, 5)
    mid_ref = phase_ref_5[..., 2:3]
    dphi_ref = phase_ref_5 - mid_ref             # (N, 5)

    inv_2pi = 1.0 / (2.0 * np.pi)

    # 3-point central diff: phi'(0) = (phi[+1] - phi[-1]) / (2h)
    f1_tdi_3 = (dphi_tdi[..., 3] - dphi_tdi[..., 1]) / (2.0 * h) * inv_2pi
    f1_ref_3 = (dphi_ref[..., 3] - dphi_ref[..., 1]) / (2.0 * h) * inv_2pi
    f1_3 = f1_tdi_3 + f1_ref_3[None, :]

    # 5-point central diff: phi'(0) = (-phi[+2] + 8 phi[+1] - 8 phi[-1] + phi[-2]) / (12h)
    def cfd5_first(d):
        return (-d[..., 4] + 8.0 * d[..., 3] - 8.0 * d[..., 1] + d[..., 0]) / (12.0 * h)
    f1_tdi_5 = cfd5_first(dphi_tdi) * inv_2pi
    f1_ref_5 = cfd5_first(dphi_ref) * inv_2pi
    f1_5 = f1_tdi_5 + f1_ref_5[None, :]

    # 3-point central diff: phi''(0) = (phi[+1] - 2 phi[0] + phi[-1]) / h^2
    def cfd3_second(d):
        return (d[..., 3] - 2.0 * d[..., 2] + d[..., 1]) / (h * h)
    fdot_tdi_3 = cfd3_second(dphi_tdi) * inv_2pi
    fdot_ref_3 = cfd3_second(dphi_ref) * inv_2pi
    fdot_3 = fdot_tdi_3 + fdot_ref_3[None, :]

    # 5-point central diff:
    #  phi''(0) = (-phi[-2] + 16 phi[-1] - 30 phi[0] + 16 phi[+1] - phi[+2]) / (12 h^2)
    def cfd5_second(d):
        return (-d[..., 0] + 16.0 * d[..., 1] - 30.0 * d[..., 2]
                + 16.0 * d[..., 3] - d[..., 4]) / (12.0 * h * h)
    fdot_tdi_5 = cfd5_second(dphi_tdi) * inv_2pi
    fdot_ref_5 = cfd5_second(dphi_ref) * inv_2pi
    fdot_5 = fdot_tdi_5 + fdot_ref_5[None, :]

    return f1_3, f1_5, fdot_3, fdot_5


# ----------------------------------------------------------------------
# Spline derivatives from TDTDIOutput
# ----------------------------------------------------------------------
t_3 = np.tile(t_test, (1, 3, 1))           # (1, 3, N_test) for tdi_phase_spl
t_1 = t_test[None, :]                       # (1, N_test) for phase_ref_spl
inv_2pi = 1.0 / (2.0 * np.pi)
f1_spl_tdi = wave_spline.tdi_phase_spl(t_3, derivative=1)[0] * inv_2pi
f1_spl_ref = wave_spline.phase_ref_spl(t_1, derivative=1)[0] * inv_2pi
f1_spl = f1_spl_tdi + f1_spl_ref[None, :]

# The cubic-spline second derivative is piecewise linear (and discontinuous
# at the knots).  Useful as a sanity check, not as ground truth.
fdot_spl_tdi = wave_spline.tdi_phase_spl(t_3, derivative=2)[0] * inv_2pi
fdot_spl_ref = wave_spline.phase_ref_spl(t_1, derivative=2)[0] * inv_2pi
fdot_spl = fdot_spl_tdi + fdot_spl_ref[None, :]

# ----------------------------------------------------------------------
# Sweep h and report
# ----------------------------------------------------------------------
h_list = [float(v) for v in os.environ.get("H_LIST", "50,100,200,500,1000,2000").split(",")]

print(f"[setup] backend={backend}  N_sparse={N_sparse}  N_test={N_test}")
print(f"[setup] f0={f0_v[0]:.4e} Hz  fdot={fdot_v[0]:.4e} Hz/s  Tobs={Tobs:.3e} s")
print(f"[setup] t_test spans [{t_test[0]:.3e}, {t_test[-1]:.3e}] s")

results = {h: numerical_deriv_at(t_test, h) for h in h_list}

def banner(msg):
    print()
    print("=" * 112)
    print(msg)
    print("=" * 112)

def header_for(label):
    return f"{'h [s]':>8s}  " + "  ".join(
        f"{label}_chan{c}_3pt   {label}_chan{c}_5pt" for c in range(3)
    )

banner("FIRST DERIVATIVE: max |f_num - f_spl| (Hz)  -- spline.derivative=1 used as reference")
print(header_for("f"))
for h in h_list:
    f1_3, f1_5, _, _ = results[h]
    row = f"{h:8.1f}  "
    for c in range(3):
        e3 = np.abs(f1_3[c] - f1_spl[c]).max()
        e5 = np.abs(f1_5[c] - f1_spl[c]).max()
        row += f"  {e3:11.3e}    {e5:11.3e} "
    print(row)

banner("SECOND DERIVATIVE: max |fdot_num - fdot_spl| (Hz/s) -- spline.derivative=2 only a rough comparator")
print(header_for("d"))
for h in h_list:
    _, _, fdot_3, fdot_5 = results[h]
    row = f"{h:8.1f}  "
    for c in range(3):
        e3 = np.abs(fdot_3[c] - fdot_spl[c]).max()
        e5 = np.abs(fdot_5[c] - fdot_spl[c]).max()
        row += f"  {e3:11.3e}    {e5:11.3e} "
    print(row)

banner("RICHARDSON 2nd derivative: fdot_R = (4 fdot[h/2] - fdot[h]) / 3 "
       "(self-consistency)")
# For each pair (h_big, h_small=h_big/2) report max | fdot_R - fdot[h_small] |
# (rate of convergence) and max | fdot_R - fdot_spl | (vs spline).
print(f"{'h_big->h_small':>16s}  " + "  ".join(
    f"chan{c}: |R-h/2|     |R-spl|   " for c in range(3)
))
pairs = []
sorted_hs = sorted(set(h_list), reverse=True)
for i, h in enumerate(sorted_hs[:-1]):
    h_small = h * 0.5
    if h_small not in results:
        results[h_small] = numerical_deriv_at(t_test, h_small)
    pairs.append((h, h_small))

for h_big, h_small in pairs:
    _, _, fdot3_big, _   = results[h_big]
    _, _, fdot3_small, _ = results[h_small]
    fdot_R = (4.0 * fdot3_small - fdot3_big) / 3.0
    row = f"  {h_big:5.1f}->{h_small:6.2f}  "
    for c in range(3):
        e_small = np.abs(fdot_R[c] - fdot3_small[c]).max()
        e_spl   = np.abs(fdot_R[c] - fdot_spl[c]).max()
        row += f" {e_small:9.3e}  {e_spl:9.3e}  "
    print(row)

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
h_plot = float(os.environ.get("H_PLOT", 500.0))
if h_plot not in results:
    results[h_plot] = numerical_deriv_at(t_test, h_plot)
f1_3, f1_5, fdot_3, fdot_5 = results[h_plot]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
chan_label = ["X", "Y", "Z"]
chan_plot = 0  # X channel
ax = axes[0, 0]
ax.set_title(f"first derivative, channel {chan_label[chan_plot]}, h={h_plot}s")
ax.plot(t_test, f1_spl[chan_plot], 'k-', label='spline (deriv=1)')
ax.plot(t_test, f1_3[chan_plot], 'ro', label='num 3pt CFD', mfc='none')
ax.plot(t_test, f1_5[chan_plot], 'b+', label='num 5pt CFD')
ax.set_ylabel("f [Hz]")
ax.legend(loc='best', fontsize=9)

ax = axes[0, 1]
ax.set_title(f"first derivative residual (num - spline), channel {chan_label[chan_plot]}")
for h in sorted(set(h_list)):
    f1_3_h, _, _, _ = results[h]
    ax.plot(t_test, f1_3_h[chan_plot] - f1_spl[chan_plot], '.-', label=f"3pt h={h:g}")
ax.axhline(0.0, color='k', lw=0.5)
ax.set_ylabel("Δf [Hz]")
ax.legend(loc='best', fontsize=8)

ax = axes[1, 0]
ax.set_title(f"second derivative, channel {chan_label[chan_plot]}, h={h_plot}s")
ax.plot(t_test, fdot_spl[chan_plot], 'k-', label='spline (deriv=2)')
ax.plot(t_test, fdot_3[chan_plot], 'ro', label='num 3pt CFD', mfc='none')
ax.plot(t_test, fdot_5[chan_plot], 'b+', label='num 5pt CFD')
# add a richardson curve for the smallest available pair
if pairs:
    h_big, h_small = pairs[-1]
    _, _, fdot3_big, _   = results[h_big]
    _, _, fdot3_small, _ = results[h_small]
    fdot_R = (4.0 * fdot3_small - fdot3_big) / 3.0
    ax.plot(t_test, fdot_R[chan_plot], 'gs',
            label=f"Richardson(h={h_big:g},{h_small:g})", mfc='none')
ax.set_xlabel("t [s]")
ax.set_ylabel("fdot [Hz/s]")
ax.legend(loc='best', fontsize=9)

ax = axes[1, 1]
ax.set_title("second derivative scan vs h (channel X, |fdot_3pt - fdot_R[h_min]|)")
# Use the finest Richardson estimate as a reference
if pairs:
    h_big, h_small = pairs[-1]
    _, _, fdot3_big, _   = results[h_big]
    _, _, fdot3_small, _ = results[h_small]
    fdot_R_ref = (4.0 * fdot3_small - fdot3_big) / 3.0
    hs_axis = []
    err_3 = []
    err_5 = []
    for h in sorted(set(h_list)):
        _, _, fdot_3_h, fdot_5_h = results[h]
        hs_axis.append(h)
        err_3.append(np.abs(fdot_3_h[chan_plot] - fdot_R_ref[chan_plot]).max())
        err_5.append(np.abs(fdot_5_h[chan_plot] - fdot_R_ref[chan_plot]).max())
    ax.loglog(hs_axis, err_3, 'r-o', label='3pt CFD')
    ax.loglog(hs_axis, err_5, 'b-+', label='5pt CFD')
    ax.set_xlabel("h [s]")
    ax.set_ylabel("max |fdot - Richardson| [Hz/s]")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

fig.tight_layout()
out_png = "gb_tdi_numerical_derivative.png"
fig.savefig(out_png, dpi=140)
print(f"\n[plot] {out_png}")
