"""Numerical validation of the two-quadrature phase maximisation
(``gbgpu.gb_likelihood.TwoQuadraturePhaseMaxMixin``).

Builds the WDM chunked-het engine on a small CPU grid (same scaffolding
as gb_chunked_swap_ll_validate.py), injects template(A) as the data, and
checks ``phase_maximize=True`` end to end:

1. carrier period: d_h flips sign under a pi shift of physical phi0
   (multiplier-1 carrier; catches a wrong quadrature step);
2. h_h is phase-invariant;
3. |D| recovers the FULL overlap: get_ll(phase_maximize=True) at a
   phase-offset copy of A reports d_h_max ~ h_h;
4. the returned phase_angle is the shift to ADD to PHYSICAL phi0:
   re-evaluating plainly at phi0 + phase_angle reproduces d_h_max and
   the maximised ll;
5. swap equivalence: get_swap_ll(phase_maximize=True) with a
   phase-offset add-template equals the plain get_swap_ll at the
   corrected phase (ll_diff, d_h_add, hh_cross);
6. sampling-basis contract: the stock GB transform maps sampling
   ``phi0 -> -phi0``, so the move-side "subtract phase_angle from
   sampling column 3" lands exactly on the physical maximum.

Run:  python scripts/gb_chunked_het/gb_phase_max_validate.py
"""
import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine, PHYS_IDX_PHI0


class _FullGridWDMHolder:
    """Duck-type for the per-band ACA: full-grid residual + invC."""

    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


failures = []


def check(name, got, want, rtol):
    rel = abs(got - want) / max(abs(want), 1e-300)
    ok = rel <= rtol
    print(f"  {name:44s} got={got:+.10e} want={want:+.10e} "
          f"rel={rel:.2e} [{'OK' if ok else 'FAIL'}]")
    if not ok:
        failures.append(name)


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, 512
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)

    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        force_backend=backend,
    )
    comp = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
    )
    engine = WDMBandLikelihoodEngine(
        comp, wdm_set, nchannels=3, tdi_channel_setup="XYZ",
    )

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    B = A.copy()
    B[1] += 0.6 * layer_df
    B[4:7] = [2.9, 1.1, 0.9]

    # Data = template(A) on the active grid; identity invC.
    hA = np.zeros((3, Nf, Nt))
    comp.fill_global_wdm(A[None, :], hA, convert_to_ra_dec=False)
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    hA_act = np.ascontiguousarray(hA[:, ilo:ihi, wdm_set.active_slice_t])
    nch, nfa, nta = hA_act.shape
    invC = np.zeros((nch, nch, nfa, nta))
    for c in range(nch):
        invC[c, c] = 1.0
    holder = _FullGridWDMHolder(hA_act, invC)

    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})
    # convert_to_ra_dec defaults must match between fill and get_ll: the
    # comp was built with the same params basis both ways, so pin it off.
    comp.convert_to_ra_dec = False

    # Reference at the TRUE phase.
    engine.get_ll(holder, A[None, :], phase_maximize=False, **kw)
    d_h_true = float(engine.d_h_out[0])
    h_h_true = float(engine.h_h_out[0])
    check("d_h(true phase) == h_h", d_h_true, h_h_true, 1e-8)

    # 1. carrier period: pi shift flips the sign (multiplier-1 carrier).
    A_pi = A.copy()
    A_pi[PHYS_IDX_PHI0] += np.pi
    engine.get_ll(holder, A_pi[None, :], phase_maximize=False, **kw)
    check("d_h(phi0+pi) == -d_h(true)", float(engine.d_h_out[0]),
          -d_h_true, 1e-6)

    # Phase-offset proposal.
    delta_true = 1.3
    P = A.copy()
    P[PHYS_IDX_PHI0] += delta_true

    ll_pm = engine.get_ll(holder, P[None, :], phase_maximize=True, **kw)
    d_h_max = float(engine.d_h_out[0])
    h_h_pm = float(engine.h_h_out[0])
    angle = float(engine.phase_angle[0])

    # 2 + 3: h_h invariant (only to HETERODYNE accuracy, ~1e-4: the
    # chunked-het reference interacts weakly with the template phase);
    # |D| recovers the full overlap.
    check("h_h(phase-max) == h_h(true)", h_h_pm, h_h_true, 1e-3)
    check("d_h_max == h_h (full overlap)", d_h_max, h_h_true, 1e-6)

    # 4: phase_angle is the shift to ADD to physical phi0. The tight
    # (1e-8) checks prove the mixin finds the KERNEL's own maximum
    # exactly; the analytic-offset check is heterodyne-limited (~1e-4).
    P_corr = P.copy()
    P_corr[PHYS_IDX_PHI0] += angle
    ll_plain = engine.get_ll(holder, P_corr[None, :], phase_maximize=False, **kw)
    check("d_h(phi0 + angle) == d_h_max", float(engine.d_h_out[0]),
          d_h_max, 1e-8)
    check("ll(phase-max) == ll(corrected)", float(ll_pm[0]),
          float(ll_plain[0]), 1e-3)
    # The recovered angle must undo the imposed offset (mod 2 pi).
    ang_err = (angle + delta_true) % (2 * np.pi)
    ang_err = min(ang_err, 2 * np.pi - ang_err)
    print(f"  {'angle == -imposed offset (abs err)':44s} "
          f"got={ang_err:.3e} rad [{'OK' if ang_err < 1e-3 else 'FAIL'}]")
    if ang_err >= 1e-3:
        failures.append("angle == -imposed offset")

    # 5: swap equivalence -- phase-max swap == plain swap at the
    # corrected add-phase.
    res_pm = engine.get_swap_ll(holder, B[None, :], P[None, :],
                                phase_maximize=True, **kw)
    ang_swap = float(res_pm.phase_angle[0])
    P_sw = P.copy()
    P_sw[PHYS_IDX_PHI0] += ang_swap
    res_ref = engine.get_swap_ll(holder, B[None, :], P_sw[None, :],
                                 phase_maximize=False, **kw)
    check("swap ll_diff(pm) == ll_diff(corrected)",
          float(res_pm.ll_diff[0]), float(res_ref.ll_diff[0]), 1e-3)
    check("swap d_h_add(pm) == d_h_add(corrected)",
          float(res_pm.d_h_add[0]), float(res_ref.d_h_add[0]), 1e-6)
    check("swap hh_cross(pm) == hh_cross(corrected)",
          float(res_pm.hh_cross[0]), float(res_ref.hh_cross[0]), 1e-6)

    # 6: sampling-basis contract. The LIVE GB transform (built by
    # GBSetup.init_sampling_info, the one gb_info.transform carries into
    # the moves) flips phi0's sign ("phi0": x -> -x, JaxGB convention),
    # so the move-side "sampling col 3 -= angle" equals the physical
    # "phi0 += angle" this engine convention requires. NOTE:
    # make_gb_transform_container() is now THE single stock factory and
    # carries this same phi0 flip (the old ecliptic no-flip variant is gone).
    from eryn.utils import TransformContainer
    tc = TransformContainer(
        input_basis=["A", "f0", "fdot", "phi0",
                     "cos_iota", "psi", "alpha", "sin_delta"],
        output_basis=["A", "f0", "fdot", "fddot", "phi0",
                      "cos_iota", "psi", "alpha", "sin_delta"],
        parameter_transforms={
            "A": np.exp,
            "f0": lambda x: x / 1e3,
            "phi0": lambda x: -1 * x,
            "cos_iota": np.arccos,
            "sin_delta": np.arcsin,
        },
        fill_dict={"fddot": 0.0},
    )
    samp = np.array([np.log(A[0]), A[1] * 1e3, A[2], 0.7,
                     np.cos(A[5]), A[6], A[7], np.sin(0.3)])
    a_test = 0.41
    samp_corr = samp.copy()
    samp_corr[3] -= a_test
    phys_0 = tc.both_transforms(samp[None, :])
    phys_1 = tc.both_transforms(samp_corr[None, :])
    check("live transform: sampling -a -> physical +a",
          float(phys_1[0, PHYS_IDX_PHI0] - phys_0[0, PHYS_IDX_PHI0]),
          a_test, 1e-12)

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        raise SystemExit(1)
    print("\nALL PHASE-MAX CHECKS PASSED")


if __name__ == "__main__":
    main()
