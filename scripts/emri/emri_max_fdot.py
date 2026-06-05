#!/usr/bin/env python
"""Probe maximal fdot of an EMRI TDI signal over a 1-year observation.

Mirrors sobbh_max_fdot.py: builds an EMRITDIonFly waveform, retrieves the
TDTDIOutput splines (amp, tdi_phase, phase_ref) — note that for EMRIs the
``num_sub`` axis indexes mode triplets (m, k, n), not separate sources —
and uses CubicSplineInterpolant's analytic derivative= keyword to evaluate

    f      = (1/(2 pi)) d(phase_total)/dt
    fdot   = (1/(2 pi)) d^2(phase_total)/dt^2

on a dense grid. The "carrier" per mode is phase_ref (sky/Doppler delay
applied at SC1); per-channel phase_total = phase_ref + tdi_phase[channel]
adds the TDI delay residuals. Reports the maxima per mode and the global
worst-case across modes/channels.
"""

import sys
import types

# emritdionfly.py at module-import time pulls in a few side-imports that
# aren't part of EMRITDIonFly itself (preprocessing.L1ProcessingStep,
# phentax.waveform.IMRPhenomTHM, scienceplots). Stub them so we can use
# EMRITDIonFly without those extras installed.
for _stub_name in ("preprocessing", "phentax", "phentax.waveform", "scienceplots"):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = types.ModuleType(_stub_name)
sys.modules["preprocessing"].L1ProcessingStep = object
sys.modules["phentax.waveform"].IMRPhenomTHM = object

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Make plt.style.use("science",...) a no-op since scienceplots is stubbed.
_orig_style_use = plt.style.use
def _safe_style_use(*args, **kwargs):
    try:
        return _orig_style_use(*args, **kwargs)
    except Exception:
        return None
plt.style.use = _safe_style_use

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.domains import WDMSettings
from fastlisaresponse.tdiconfig import TDIConfig

from few.waveform import FastKerrEccentricEquatorialFlux

from emritdionfly import EMRITDIonFly


def probe_emri(
    m1=1e6, m2=1e1, a=0.99, p0=6.1, e0=0.3, xI0=+1.0,
    dist=2.0,
    qS=None, phiS=None, qK=None, phiK=None,
    Phi_phi0=1.0803123123, Phi_theta0=1.9823423423, Phi_r0=4.32094823423,
    Tobs_years=1.0, dt=10.0, N_probe=4096,
):
    backend = "cpu"
    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig('2nd generation')

    # Match the WDM rounding used by emri_test_script_td_wave.py so the
    # window is an integer number of WDM pixels.
    _Tobs = Tobs_years * YRSID_SI
    (Nf, Nt, wavelet_duration) = WDMSettings.adjust_to_even_bins(
        0.5 * 24 * 3600.0, 0.75 * 24 * 3600.0, dt, _Tobs)
    Tobs = Nt * wavelet_duration

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start
    t_buffer = 3e4

    # EMRI test-script defaults if the caller didn't override sky/spin angles.
    lam = 0.2538922432234
    beta_eclip = -0.418762312
    if qS is None: qS = np.pi / 2 - beta_eclip
    if phiS is None: phiS = lam
    if qK is None: qK = 0.2340980298542
    if phiK is None: phiK = 4.098234232

    N_pts = 16384
    inspiral_kwargs_tof = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "upsample": True,
        "fix_t": True,
        "new_t": np.linspace(t_start + t_buffer, t_start + Tobs - t_buffer, N_pts),
    }
    sum_kwargs = {"pad_output": True}
    mode_selector_kwargs = {"mode_selection_threshold": 1e-2}

    wave_generator_kerr = FastKerrEccentricEquatorialFlux(
        force_backend=backend,
        inspiral_kwargs=inspiral_kwargs_tof,
        sum_kwargs=sum_kwargs,
        mode_selector_kwargs=mode_selector_kwargs,
    )

    # EMRITDIonFly does ``t_arr_in = self.t0 + Kerr_wave.t_arr``, but
    # Kerr_wave.t_arr already encodes ``t_start`` via ``new_t`` in
    # ``inspiral_kwargs_tof``. Pass t0=0 here so the start offset isn't
    # double-counted.
    emri_gen = EMRITDIonFly(
        wave_generator_kerr, orbits, tdi_config, dt, Tobs, 0.0,
    )

    out = emri_gen(
        m1, m2, a, p0, e0, xI0, dist,
        qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
    )

    # out is a TDTDIOutput. x has shape (num_modes, N_pts).
    num_modes = out.num_bin
    t_grid = np.asarray(out.x)
    t_start_arr = t_grid[:, 0]
    t_end_arr = t_grid[:, -1]
    t_lo = float(np.max(t_start_arr))
    t_hi = float(np.min(t_end_arr))
    edge = 1e-3 * (t_hi - t_lo)
    t_probe = np.linspace(t_lo + edge, t_hi - edge, N_probe)

    # phase_ref_spl expects shape (num_modes, N_probe).
    t_probe_pr = np.tile(t_probe, (num_modes, 1))
    dphase_ref = out.phase_ref_spl(t_probe_pr, derivative=1)
    d2phase_ref = out.phase_ref_spl(t_probe_pr, derivative=2)
    f_ref = dphase_ref / (2.0 * np.pi)
    fdot_ref = d2phase_ref / (2.0 * np.pi)

    # tdi_phase_spl expects shape (num_modes, 3, N_probe).
    t_probe_3d = np.repeat(t_probe_pr[:, None, :], 3, axis=1)
    tdi_dphase = out.tdi_phase_spl(t_probe_3d, derivative=1)
    tdi_d2phase = out.tdi_phase_spl(t_probe_3d, derivative=2)
    f_chan = (dphase_ref[:, None, :] + tdi_dphase) / (2.0 * np.pi)
    fdot_chan = (d2phase_ref[:, None, :] + tdi_d2phase) / (2.0 * np.pi)

    return {
        "t_probe": t_probe,
        "f_ref": f_ref, "fdot_ref": fdot_ref,           # (num_modes, N_probe)
        "f_chan": f_chan, "fdot_chan": fdot_chan,        # (num_modes, 3, N_probe)
        "num_modes": num_modes,
        "Tobs": Tobs,
    }


def report(label, res):
    t_probe = res["t_probe"]
    f_ref = res["f_ref"]
    fdot_ref = res["fdot_ref"]
    f_chan = res["f_chan"]
    fdot_chan = res["fdot_chan"]
    num_modes = res["num_modes"]

    abs_fdot_ref = np.abs(fdot_ref)
    abs_fdot_chan = np.abs(fdot_chan)

    print(f"=== {label} (num_modes = {num_modes}) ===")
    print(f"  Tobs                          : {res['Tobs']:.3e} s "
          f"({res['Tobs'] / YRSID_SI:.3f} yr)")
    print(f"  f_ref range  [Hz]            : {f_ref.min():.6e} .. {f_ref.max():.6e}")
    print(f"  fdot_ref range [Hz/s]         : {fdot_ref.min():.6e} .. {fdot_ref.max():.6e}")

    # Global worst mode for the carrier
    flat = abs_fdot_ref.reshape(-1)
    imax = int(np.argmax(flat))
    mode_imax, t_imax = np.unravel_index(imax, abs_fdot_ref.shape)
    print(f"  |fdot_ref| max (carrier)      : {flat[imax]:.6e}  "
          f"mode={mode_imax}/{num_modes}, t={t_probe[t_imax]:.3e} s, "
          f"f={f_ref[mode_imax, t_imax]:.6e} Hz")

    # Per-channel worst mode (carrier + tdi_phase derivative)
    for c, name in enumerate(("X", "Y", "Z")):
        flat = abs_fdot_chan[:, c].reshape(-1)
        ic = int(np.argmax(flat))
        m_ic, t_ic = np.unravel_index(ic, abs_fdot_chan[:, c].shape)
        print(f"  channel {name}: |fdot| max          : {flat[ic]:.6e}  "
              f"mode={m_ic}/{num_modes}, t={t_probe[t_ic]:.3e} s, "
              f"f={f_chan[m_ic, c, t_ic]:.6e} Hz")
    print()


if __name__ == "__main__":
    # Canonical EMRI test source (matches emri_test_script_td_wave.py)
    res = probe_emri(
        m1=1e6, m2=1e1, a=0.99, p0=6.1, e0=0.3, xI0=+1.0,
        dist=2.0, Tobs_years=1.0,
    )
    report("M=1e6 Msun, mu=10 Msun, a=0.99, p0=6.1, e0=0.3, T=1yr", res)

    # Per-mode summary plot: max-over-time |fdot| vs mode index
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    per_mode_max_carrier = np.max(np.abs(res["fdot_ref"]), axis=-1)
    per_mode_max_chan = np.max(np.abs(res["fdot_chan"]), axis=(-1, -2))
    ax[0].semilogy(per_mode_max_carrier, ".", label="carrier (phase_ref)")
    ax[0].semilogy(per_mode_max_chan, "x", label="carrier + tdi_phase")
    ax[0].set_xlabel("mode index")
    ax[0].set_ylabel("max |fdot| [Hz/s]")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    # Time series of fdot for the worst-carrier mode
    worst_mode = int(np.argmax(per_mode_max_carrier))
    ax[1].plot(res["t_probe"] / YRSID_SI, res["fdot_ref"][worst_mode],
               label=f"carrier, mode {worst_mode}")
    for c, name in enumerate(("X", "Y", "Z")):
        ax[1].plot(res["t_probe"] / YRSID_SI, res["fdot_chan"][worst_mode, c],
                   ls="--", lw=0.6, label=f"{name}, mode {worst_mode}")
    ax[1].set_xlabel("t [yr]")
    ax[1].set_ylabel("fdot [Hz/s]")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.suptitle("EMRI M=1e6, mu=10, a=0.99, p0=6.1, e0=0.3, T=1yr")
    fig.tight_layout()
    fig.savefig("emri_max_fdot.png", dpi=120)
    print("Saved emri_max_fdot.png")
