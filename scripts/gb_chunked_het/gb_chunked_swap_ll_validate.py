"""Numerical validation of GBWDMComputations.get_swap_ll_wdm (chunked-het WDM).

The sampler no longer scores proposals with swap_ll (conditional-Gibbs
get_ll path), but the proposal Fisher (``information_matrix`` ->
``_compute_proposal_cholesky``) still consumes swap's add-remove cross
term ``<h(pa)|h(pb)>`` -- the only kernel that computes a pairwise
template overlap. This script checks that path numerically on a small
CPU grid (no data files needed):

1. degenerate swap: ``swap(A, A)`` collapses -- d_h_a == d_h_r,
   aa == rr == ar;
2. consistency with get_ll: swap's d_h_a / aa match get_ll(A)'s
   d_h_out / h_h_out (identical kernels inside, so to roundoff);
3. symmetry: ``swap(A, B).ar == swap(B, A).ar``;
4. cross-term physics: the normalized overlap ar / sqrt(aa * rr)
   matches the dense-template overlap computed from fill_global
   buffers with the same diagonal invC weighting (conventions cancel
   in the ratio; agreement is limited only by the heterodyne
   approximation, so ~1e-6-level, not roundoff).

Both layer-grouping modes (the canonical grouped path the Fisher uses,
and the wide-band ungrouped path) are exercised.

Run:  python scripts/gb_chunked_het/gb_chunked_swap_ll_validate.py
"""
import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations


class _FullGridWDMHolder:
    """Duck-type for the ``wdm_holder`` argument: full-grid data + invC."""

    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


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
    print(f"[setup] Nf={Nf} Nt={Nt} n_chunks={comp.n_chunks} "
          f"layer_df={layer_df:.3e} Hz")

    # Two sources in the same 5-layer band (nonzero cross term): B sits
    # 0.6 layers above A with different phase/orientation.
    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    B = A.copy()
    B[1] += 0.6 * layer_df
    B[4:7] = [2.9, 1.1, 0.9]

    # Data = template(A) on the full grid; invC = ones (identity checks
    # only need both sides weighted identically).
    hA = np.zeros((3, Nf, Nt))
    hB = np.zeros((3, Nf, Nt))
    comp.fill_global_wdm(A[None, :], hA, convert_to_ra_dec=False)
    comp.fill_global_wdm(B[None, :], hB, convert_to_ra_dec=False)

    # The kernels consume the ACTIVE-band layout (nchannels, Nf_active,
    # Nt_active); slice the filled full-grid templates down for the data
    # holder and for the dense-overlap comparison.
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    hA_act = np.ascontiguousarray(hA[:, ilo:ihi, wdm_set.active_slice_t])
    hB_act = np.ascontiguousarray(hB[:, ilo:ihi, wdm_set.active_slice_t])
    # tdi_type="XYZ" consumes the full cross-channel invC slab
    # (nchannels, nchannels, Nf_active, Nt_active): identity across
    # channels keeps the weighting trivial for the identity checks.
    nch, nfa, nta = hA_act.shape
    invC = np.zeros((nch, nch, nfa, nta))
    for c in range(nch):
        invC[c, c] = 1.0
    holder = _FullGridWDMHolder(hA_act, invC)

    failures = []

    def check(name, got, want, rtol):
        rel = abs(got - want) / max(abs(want), 1e-300)
        ok = rel <= rtol
        print(f"  {name:34s} got={got:+.12e} want={want:+.12e} "
              f"rel={rel:.2e} [{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(name)

    for use_groups in (True, False):
        tag = "grouped" if use_groups else "ungrouped"
        print(f"[{tag}]")
        kw = dict(convert_to_ra_dec=False, use_layer_groups=use_groups)

        comp.get_ll_wdm(A[None, :], holder, **kw)
        dhA, hhA = float(comp.d_h_out[0]), float(comp.h_h_out[0])
        comp.get_ll_wdm(B[None, :], holder, **kw)
        dhB, hhB = float(comp.d_h_out[0]), float(comp.h_h_out[0])

        # 1 + 2: degenerate swap collapses onto get_ll.
        out = comp.get_swap_ll_wdm(A[None, :], A[None, :], holder, **kw)
        _, _, d_h_a, d_h_r, aa, rr, ar = [np.asarray(v) for v in out]
        check("swap(A,A) d_h_a == d_h_r", float(d_h_a[0]), float(d_h_r[0]), 1e-12)
        check("swap(A,A) aa == rr", float(aa[0]), float(rr[0]), 1e-12)
        check("swap(A,A) ar == aa", float(ar[0]), float(aa[0]), 1e-12)
        check("swap(A,A) d_h_a == get_ll d_h", float(d_h_a[0]), dhA, 1e-12)
        check("swap(A,A) aa == get_ll h_h", float(aa[0]), hhA, 1e-12)

        # 3: cross swap pieces + symmetry.
        outAB = comp.get_swap_ll_wdm(A[None, :], B[None, :], holder, **kw)
        outBA = comp.get_swap_ll_wdm(B[None, :], A[None, :], holder, **kw)
        aa_AB, rr_AB, ar_AB = (float(np.asarray(outAB[i])[0]) for i in (4, 5, 6))
        ar_BA = float(np.asarray(outBA[6])[0])
        check("swap(A,B) aa == get_ll(A) h_h", aa_AB, hhA, 1e-12)
        check("swap(A,B) rr == get_ll(B) h_h", rr_AB, hhB, 1e-12)
        check("swap(A,B) ar == swap(B,A) ar", ar_AB, ar_BA, 1e-10)

        # 4: normalized overlap vs dense templates (heterodyne-limited).
        ov_kernel = ar_AB / np.sqrt(aa_AB * rr_AB)
        ov_dense = float(np.sum(hA_act * hB_act) /
                         np.sqrt(np.sum(hA_act * hA_act) * np.sum(hB_act * hB_act)))
        check("overlap(kernel) == overlap(dense)", ov_kernel, ov_dense, 1e-4)

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        raise SystemExit(1)
    print("\nALL SWAP_LL CHECKS PASSED")


if __name__ == "__main__":
    main()
