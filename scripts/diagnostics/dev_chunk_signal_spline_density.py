"""
Density study for the within-(chunk, binary) signal spline.

Question: inside one chunk we currently evaluate gb->get_tdi at N_sparse
times. That's N_sparse * (orbit-eval cost). The de-carriered amp / phase
/ phi_ref are all *slow* functions of t over the chunk (modulated only
by LISA orbit on ~1yr scale). Can we get away with much fewer than
N_sparse get_tdi calls, fit cubic splines through (amp, dphi, phi_ref)
at sparse control points, and eval the spline at the full N_sparse grid?

If yes -> per (chunk, binary) cost drops from N_sparse get_tdi calls
(~256 with 32-64 orbit lookups each = ~8-16k orbit lookups) to N_cp_sig
get_tdi calls + cheap shared-mem spline eval at the rest.

What we test here:
  1. evaluate (amp, dphi=phase-2pi*f0_grid*t, phi_ref) at N_sparse
     ground truth (the existing chunked path).
  2. sub-sample to N_cp_sig (small) control points; fit CubicSpline.
  3. eval spline at the N_sparse grid; reconstruct phase=dphi+carrier.
  4. compute the slow signal s(tau) both ways and report:
       - relative amp error
       - absolute phase error
       - 1 - <s_full|s_spline>/sqrt(<full|full><spline|spline>)
         (mismatch on the slow signal)

Tolerance target: mismatch < 1e-9 (matches what the full chunked path
already hits), which works out to amp_relerr <~ 1e-5 and phase_err <~
1e-5 rad on the modulated channels.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

from check_shortened_wdm import CachedHeterodyneGenerator
from fastlisaresponse.tdionfly import GBTDIonTheFly, SOBBHTDIonTheFly
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI, MSUN_SI
from fastlisaresponse.tdiconfig import TDIConfig


def slow_signal(tdi_amp, tdi_phase, phi_ref, f0_grid, t_off):
    carrier = 2.0 * np.pi * f0_grid * t_off
    return tdi_amp * np.exp(
        +1j * (tdi_phase + phi_ref[None, :] - carrier[None, :])
    )


def slow_mismatch(a, b):
    # nchannels-summed inner product
    aa = np.real(np.sum(np.conj(a) * a))
    bb = np.real(np.sum(np.conj(b) * b))
    ab = np.real(np.sum(np.conj(a) * b))
    return 1.0 - ab / np.sqrt(aa * bb)


def run_one(params, gen, chunk_t_start, N_cp_list):
    """Compare full-N_sparse vs N_cp_sig spline reconstruction."""
    N = gen.N_sparse
    t_off = gen.t_offsets  # (N,)
    f0 = float(params[gen.f0_param_index])
    T_window = gen.T_window
    df = 1.0 / T_window
    f0_grid = round(f0 / df) * df

    # -- ground truth: evaluate at all N times --
    nch = gen.nchannels
    tdi_amp_full = np.zeros(N * nch)
    tdi_phase_full = np.zeros(N * nch)
    phi_ref_full = np.zeros(N)
    tdi_channels = np.zeros(N * nch, dtype=complex)
    t_full = chunk_t_start + t_off
    gen.gb.t_arr = gen.gb.xp.atleast_2d(t_full).copy()
    gen._wave_gen.run_wave_tdi_wrap(
        tdi_channels, tdi_amp_full, tdi_phase_full, phi_ref_full,
        params, gen.gb.t_arr.flatten().copy(),
        N, 1, gen.n_params, nch,
    )
    tdi_amp_full = tdi_amp_full.reshape(nch, N)
    tdi_phase_full = tdi_phase_full.reshape(nch, N)
    s_full = slow_signal(tdi_amp_full, tdi_phase_full, phi_ref_full, f0_grid, t_off)

    # phi_ref is FAST (carrier-bearing). We assume the kernel will be
    # modified to return a heterodyned phi_ref directly; here we simulate
    # that by subtracting the snapped carrier from the dense ground truth
    # and sub-sampling at control points (fair comparison of spline
    # accuracy independent of phase-unwrap-at-sparse-sampling pathology).
    carrier_full = 2.0 * np.pi * f0_grid * t_off
    dphi_ref_full = phi_ref_full - carrier_full

    out = {}
    for N_cp_sig in N_cp_list:
        i_cp = np.linspace(0, N - 1, N_cp_sig).astype(int)

        # Spline fits use SUB-SAMPLED ground truth at the cp indices.
        # (This is what we'd get from a kernel that already unwrapped at
        # the full dense N_sparse and then handed back the heterodyned
        # outputs at a sub-set of times.)
        tdi_amp_spl = np.empty_like(tdi_amp_full)
        tdi_phase_spl = np.empty_like(tdi_phase_full)
        for c in range(nch):
            cs_a = CubicSpline(t_off[i_cp], tdi_amp_full[c, i_cp])
            cs_p = CubicSpline(t_off[i_cp], tdi_phase_full[c, i_cp])
            tdi_amp_spl[c] = cs_a(t_off)
            tdi_phase_spl[c] = cs_p(t_off)
        cs_r = CubicSpline(t_off[i_cp], dphi_ref_full[i_cp])
        dphi_ref_spl = cs_r(t_off)
        phi_ref_spl = dphi_ref_spl + carrier_full

        s_spl = slow_signal(tdi_amp_spl, tdi_phase_spl, phi_ref_spl, f0_grid, t_off)

        amp_rel = np.max(
            np.abs(tdi_amp_spl - tdi_amp_full) / (np.abs(tdi_amp_full) + 1e-30)
        )
        tdi_phase_err = np.max(np.abs(tdi_phase_spl - tdi_phase_full))
        dphi_ref_err = np.max(np.abs(dphi_ref_spl - dphi_ref_full))
        mm = slow_mismatch(s_full, s_spl)

        out[N_cp_sig] = dict(
            amp_rel=amp_rel,
            tdi_phase_err=tdi_phase_err,
            dphi_ref_err=dphi_ref_err,
            mm=mm,
        )

    return out


def _make_gen(Nf, Nt_sub, dt, N_sparse, T_obs, orbits, tdi_config,
              source_class, n_params, f0_index, t_ref_source):
    layer_dt = Nf * dt / 2.0
    T_chunk = Nt_sub * layer_dt
    gen = CachedHeterodyneGenerator(
        T_window=T_chunk, t_ref_source=t_ref_source,
        N_sparse=N_sparse, dt=dt, nchannels=3,
        gb_kwargs=dict(orbits=orbits, tdi_config=tdi_config),
        source_class=source_class, n_params=n_params,
        f0_param_index=f0_index,
    )
    return gen, T_chunk


def _gb_cases(rng):
    """Draw a few GB sources spanning f0 in [1.5, 22] mHz."""
    cases = []
    for f0_mHz in (1.5, 5.0, 12.0, 22.0):
        for _ in range(2):
            amp = 10.0 ** rng.uniform(-23, -21)
            inc = np.arccos(rng.uniform(-1.0, 1.0))
            phi0 = rng.uniform(0, 2 * np.pi)
            psi = rng.uniform(0, np.pi)
            lam = rng.uniform(0, 2 * np.pi)
            beta = np.arcsin(rng.uniform(-1.0, 1.0))
            params = np.array(
                [amp, f0_mHz * 1e-3, 0.0, 0.0, phi0, inc, psi, lam, beta],
                dtype=float,
            )
            cases.append((f0_mHz, params))
    return cases


def _sobbh_cases(rng):
    """SOBBH params: [m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta]."""
    cases = []
    for f0_mHz in (3.0, 10.0, 20.0):
        for _ in range(2):
            m1 = rng.uniform(20.0, 60.0)
            m2 = rng.uniform(10.0, m1)
            s1 = rng.uniform(-0.5, 0.5)
            s2 = rng.uniform(-0.5, 0.5)
            distance = rng.uniform(500.0, 5000.0)  # Mpc
            phi_c = rng.uniform(0, 2 * np.pi)
            inc = np.arccos(rng.uniform(-1.0, 1.0))
            psi = rng.uniform(0, np.pi)
            lam = rng.uniform(0, 2 * np.pi)
            beta = np.arcsin(rng.uniform(-1.0, 1.0))
            params = np.array(
                [m1, m2, s1, s2, distance, f0_mHz * 1e-3, phi_c, inc, psi, lam, beta],
                dtype=float,
            )
            cases.append((f0_mHz, params))
    return cases


def study(label, Nf, Nt_sub, dt, N_sparse, T_obs, orbits, tdi_config,
          source_class, n_params, f0_index, cases, N_cp_list,
          chunk_start_fracs=(0.05, 0.25, 0.5, 0.75)):
    t_ref_source = 0.25 * T_obs
    gen, T_chunk = _make_gen(
        Nf, Nt_sub, dt, N_sparse, T_obs, orbits, tdi_config,
        source_class, n_params, f0_index, t_ref_source,
    )
    chunk_starts = [f * T_obs for f in chunk_start_fracs]
    worst = {N_cp: dict(amp_rel=0.0, tdi_phase_err=0.0, dphi_ref_err=0.0, mm=0.0) for N_cp in N_cp_list}
    for f0_mHz, params in cases:
        for cs_start in chunk_starts:
            try:
                out = run_one(params, gen, cs_start, N_cp_list)
            except Exception as e:
                print(f"  skip {label} f0={f0_mHz} cs={cs_start:.2e}: {e}")
                continue
            for N_cp, r in out.items():
                for k, v in r.items():
                    if v > worst[N_cp][k]:
                        worst[N_cp][k] = v

    print(
        f"\n== {label}: T_chunk={T_chunk/86400:.2f} d, N_sparse={N_sparse}, "
        f"{len(cases)} sources, {len(chunk_starts)} starts =="
    )
    print(
        f"{'N_cp':>5}  {'amp_rel':>12}  {'tdi_phase_err':>14}  "
        f"{'dphi_ref_err':>14}  {'mm (slow)':>12}"
    )
    for N_cp in N_cp_list:
        r = worst[N_cp]
        print(
            f"{N_cp:>5d}  {r['amp_rel']:>12.3e}  {r['tdi_phase_err']:>14.3e}  "
            f"{r['dphi_ref_err']:>14.3e}  {r['mm']:>12.3e}"
        )


def main():
    dt = 10.0
    N_sparse = 256
    N_cp_list = [4, 6, 8, 12, 16, 24, 32, 48, 64]
    rng = np.random.default_rng(20260523)

    # Two regimes: (i) Nf=256 / Nt_sub=256 -> T_chunk ~ 3.8 d (existing setup),
    # (ii) Nf=4320 / Nt_sub=128 -> T_chunk ~ 32 d (half-day wavelets, 1 yr Tobs).
    configs = [
        dict(label="Nf=256 Nt_sub=256", Nf=256, Nt_sub=256, T_obs=0.5 * YRSID_SI),
        dict(label="Nf=4320 Nt_sub=128", Nf=4320, Nt_sub=128, T_obs=1.0 * YRSID_SI),
    ]

    gb_cases = _gb_cases(rng)
    sobbh_cases = _sobbh_cases(rng)

    for cfg in configs:
        # orbits configured to span Tobs.
        orbits = EqualArmlengthOrbits()
        t_arr = np.arange(0.0, cfg["T_obs"] + dt, dt)
        try:
            orbits.configure(t_arr=t_arr, dt=dt, linear_interp_setup=True)
        except TypeError:
            orbits.configure(t_arr=t_arr)
        tdi_config = TDIConfig("2nd generation")

        # GBs
        study(
            label=f"GB {cfg['label']}",
            Nf=cfg["Nf"], Nt_sub=cfg["Nt_sub"], dt=dt,
            N_sparse=N_sparse, T_obs=cfg["T_obs"],
            orbits=orbits, tdi_config=tdi_config,
            source_class=GBTDIonTheFly, n_params=9, f0_index=1,
            cases=gb_cases, N_cp_list=N_cp_list,
        )
        # SOBBHs
        try:
            study(
                label=f"SOBBH {cfg['label']}",
                Nf=cfg["Nf"], Nt_sub=cfg["Nt_sub"], dt=dt,
                N_sparse=N_sparse, T_obs=cfg["T_obs"],
                orbits=orbits, tdi_config=tdi_config,
                source_class=SOBBHTDIonTheFly, n_params=11, f0_index=5,
                cases=sobbh_cases, N_cp_list=N_cp_list,
            )
        except Exception as e:
            print(f"  SOBBH skipped at {cfg['label']}: {e}")


if __name__ == "__main__":
    main()
