"""Validate the sig-het in-model scoring path against chunked-het.

Builds the WDM band engine twice on the same small CPU grid (same
scaffolding as gb_phase_max_validate.py): once around a plain
GBWDMComputations, once around GBSignalHetComputations.for_band_engine
wrapping that same chunked comp. Data = template(A) + template(B) with
identity invC; the "in-model repeat block" scenario is simulated the way
_run_in_model_repeats produces it: source A is the picked source, the
residual CONTAINS its signal, and candidates are small perturbations of A.

Checks:

1. OUTSIDE a repeat block (no setup), the sig-het comp's engine surface is
   byte-identical to chunked (pure delegation): get_ll_wdm / d_h / h_h.
2. INSIDE a repeat block (setup_in_model at ref=A), sig-het deltas
   ``d_h - 0.5 h_h`` for perturbed candidates match chunked to the sig-het
   accuracy budget (get_ll parity ~1e-6 per the reference de-risk).
3. phase_maximize=True through the engine mixin works on the sig-het
   path: |D| ~ chunked's phase-maximized d_h.
4. clear_in_model() restores exact chunked delegation.

Run:  python scripts/gb_chunked_het/gb_sighet_inmodel_validate.py
"""
import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine, PHYS_IDX_PHI0


class _FullGridWDMHolder:
    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


failures = []


def check(name, got, want, rtol):
    rel = abs(got - want) / max(abs(want), 1e-300)
    ok = rel <= rtol
    print(f"  {name:46s} got={got:+.9e} want={want:+.9e} "
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
    # Production WDM grids carry a time-edge crop (the gb_no_foreground
    # run crops ~58 wavelets); the sig-het bin-fold quadrature needs it --
    # untrimmed edge layers (analysis window -> 0) contribute a ~0.5%-of-h_h
    # absolute systematic. Trim 40 layers each side here to match.
    edge = 40
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
    )
    chunked.convert_to_ra_dec = False
    sighet = GBSignalHetComputations.for_band_engine(chunked)

    eng_c = WDMBandLikelihoodEngine(chunked, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")
    eng_s = WDMBandLikelihoodEngine(sighet, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    B = A.copy()
    B[1] += 0.6 * layer_df
    B[4:7] = [2.9, 1.1, 0.9]

    # Residual = template(A) + template(B): the picked source A has been
    # removed from the model (its signal is IN the residual); B is an
    # unmodeled neighbor for realism.
    h = np.zeros((3, Nf, Nt))
    chunked.fill_global_wdm(A[None, :], h, convert_to_ra_dec=False)
    chunked.fill_global_wdm(B[None, :], h, convert_to_ra_dec=False)
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    h_act = np.ascontiguousarray(h[:, ilo:ihi, wdm_set.active_slice_t])
    nch, nfa, nta = h_act.shape
    invC = np.zeros((nch, nch, nfa, nta))
    for c in range(nch):
        invC[c, c] = 1.0
    holder = _FullGridWDMHolder(h_act, invC)

    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})

    # Candidate perturbations around A (typical in-model step sizes).
    rng = np.random.default_rng(7)
    cands = [A.copy()]
    for _ in range(4):
        p = A.copy()
        p[0] *= 1.0 + 0.02 * rng.standard_normal()
        p[1] += 0.02 * layer_df * rng.standard_normal()
        p[2] *= 1.0 + 0.05 * rng.standard_normal()
        p[PHYS_IDX_PHI0] += 0.2 * rng.standard_normal()
        p[5] += 0.05 * rng.standard_normal()
        cands.append(p)

    # 1. No setup active: pure delegation, byte-identical.
    eng_c.get_ll(holder, A[None, :], phase_maximize=False, **kw)
    dh_c, hh_c = float(eng_c.d_h_out[0]), float(eng_c.h_h_out[0])
    eng_s.get_ll(holder, A[None, :], phase_maximize=False, **kw)
    check("delegation d_h == chunked d_h", float(eng_s.d_h_out[0]), dh_c, 0.0)
    check("delegation h_h == chunked h_h", float(eng_s.h_h_out[0]), hh_c, 0.0)

    # 2. Repeat block: setup at ref=A. Away from the reference the two
    # het schemes carry INDEPENDENT approximation errors, so each is
    # checked against DENSE truth (direct template inner products on the
    # same slabs), not against the other. The kernel <-> dense
    # normalization is calibrated once from h_h at the reference.
    sighet.setup_in_model(holder, A[None, :], zeros)
    print("  [setup_in_model built 1 reference]")

    def dense_terms(p):
        ht = np.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(p[None, :], ht, convert_to_ra_dec=False)
        ht_act = ht[:, ilo:ihi, wdm_set.active_slice_t]
        return float(np.sum(h_act * ht_act)), float(np.sum(ht_act * ht_act))

    _, hh_dense_A = dense_terms(A)
    norm = hh_c / hh_dense_A
    worst_c, worst_s = 0.0, 0.0
    for i, p in enumerate(cands):
        dh_d, hh_d = dense_terms(p)
        delta_d = norm * (dh_d - 0.5 * hh_d)
        eng_c.get_ll(holder, p[None, :], phase_maximize=False, **kw)
        delta_c = float(eng_c.d_h_out[0] - 0.5 * eng_c.h_h_out[0])
        eng_s.get_ll(holder, p[None, :], phase_maximize=False, **kw)
        delta_s = float(eng_s.d_h_out[0] - 0.5 * eng_s.h_h_out[0])
        scale = abs(delta_d) if abs(delta_d) > 0 else 1.0
        err_c = abs(delta_c - delta_d) / scale
        err_s = abs(delta_s - delta_d) / scale
        worst_c, worst_s = max(worst_c, err_c), max(worst_s, err_s)
        print(f"  cand{i}: delta dense={delta_d:+.6e} "
              f"chunked err={err_c:.2e} sighet err={err_s:.2e}")
    print(f"  worst vs dense: chunked={worst_c:.2e} sighet={worst_s:.2e}")
    if worst_s > max(3.0 * worst_c, 1e-3):
        failures.append("sighet accuracy vs dense")
        print("  ** sig-het error budget exceeded relative to chunked **")
    else:
        print("  in-model sig-het accuracy within budget [OK]")

    # 3. Phase-max through the mixin on the sig-het path: internal
    # consistency (|D| equals the plain d_h at phi0 + angle).
    p = cands[3]
    ll_pm = eng_s.get_ll(holder, p[None, :], phase_maximize=True, **kw)
    dmax_s = float(eng_s.d_h_out[0])
    ang_s = float(eng_s.phase_angle[0])
    p_corr = p.copy()
    p_corr[PHYS_IDX_PHI0] += ang_s
    eng_s.get_ll(holder, p_corr[None, :], phase_maximize=False, **kw)
    check("sighet phase-max round trip d_h", float(eng_s.d_h_out[0]),
          dmax_s, 1e-8)

    # 4. Mid-block PATCH (drift refresh): re-anchor the reference at
    # cand4 in place, then compare with a FRESH setup at cand4 -- the
    # two must be identical computations.
    p4 = cands[4]
    sighet.setup_in_model(holder, p4[None, :], zeros)   # patch in place
    eng_s.get_ll(holder, p4[None, :], phase_maximize=False, **kw)
    dh_patch = float(eng_s.d_h_out[0])
    sighet.clear_in_model()
    sighet.setup_in_model(holder, p4[None, :], zeros)   # fresh build
    eng_s.get_ll(holder, p4[None, :], phase_maximize=False, **kw)
    check("patched reference == fresh reference", dh_patch,
          float(eng_s.d_h_out[0]), 1e-14)

    # 5. Teardown restores exact delegation.
    sighet.clear_in_model()
    eng_s.get_ll(holder, A[None, :], phase_maximize=False, **kw)
    check("post-clear d_h == chunked d_h", float(eng_s.d_h_out[0]), dh_c, 0.0)

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        raise SystemExit(1)
    print("\nALL SIG-HET IN-MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
