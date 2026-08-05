"""How many independent numbers are really in the F-stat M matrix?

The four Cornish & Crowder basis filters are not four independent waveforms.
The kernel builds them at fixed ``(A, iota, psi, phi0) = (2, pi/2,
{0, pi/4, 0, pi/4}, {0, pi, 3pi/2, pi/2})``. With ``iota = pi/2`` the cross
polarization vanishes, and the four reduce to the real and imaginary parts of
TWO complex waveforms that share one carrier -- one per ``psi``::

    u(t) = F_psi0(t)    * exp(i Phi(t))     A_0 = Re u,  A_2 = Im u
    v(t) = F_psi45(t)   * exp(i Phi(t))     A_1 = Re v,  A_3 = Im v

NOTE the pairing: (0, 2) and (1, 3) share a waveform -- NOT (0, 1) and (2, 3).
Getting that wrong makes the structure look ~30x worse than it is (measured:
1.4e-2 with the wrong pairing, 4.3e-4 with the right one).

Bilinearity gives ``<Re U|Re V> = 1/2 Re[<U|V> + Int(U V)]`` etc., where
``Int(U V)`` (no conjugate) carries ``exp(2 i Phi)`` and averages away over a
long observation. What survives::

          [ A   C   0  -D ]
    M  =  [ C   B   D   0 ]      A = 1/2 <u|u>     C = 1/2 Re<u|v>
          [ 0   D   A   C ]      B = 1/2 <v|v>     D = 1/2 Im<u|v>
          [-D   0   C   B ]

FOUR independent numbers, not ten. This script MEASURES that against the
existing chunked-het kernel rather than assuming it -- it reports each
structural relation separately and the resulting error in F.

Measured on 400 candidates, 6-7 mHz, synthetic (relative to sqrt(M00*M11)):

    bilinear |M02|,|M13| -> 0    median 4.3e-4   p90 1.3e-3
    |M00-M22|  (A)               median 6.6e-4   p90 2.5e-3
    |M11-M33|  (B)               median 6.7e-4   p90 2.5e-3
    |M01-M23|  (C)               median 1.0e-4   p90 3.5e-4
    |M03+M12|  (D antisymmetry)  median 8.8e-5   p90 4.5e-4
    F from 4 numbers vs full 4x4 median 3.2e-4   p90 1.4e-3   max 1.9e-2

Implication for a v5 F-stat kernel: 4 accumulators for N + 4 for M = 8 per
candidate, not 4 + 10 = 14. Six fewer shared-memory accumulators, which is
the scarce resource in the v5 arena.

Run (CPU, synthetic, a few minutes)::

    python scripts/fstat_proposal/fstat_bilinear_structure.py --n 400

or against real data::

    MOJITO_DATA_PATH=/path/to/data python ... --data-mode mojito
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# M is returned as the flattened UPPER TRIANGLE, row-major:
#   [M00, M01, M02, M03, M11, M12, M13, M22, M23, M33]
_IU = np.triu_indices(4)


def unpack_M(M_flat):
    """(n, 10) upper-triangle -> (n, 4, 4) symmetric."""
    M_flat = np.atleast_2d(np.asarray(M_flat))
    n = M_flat.shape[0]
    M = np.zeros((n, 4, 4))
    M[:, _IU[0], _IU[1]] = M_flat
    M = M + np.transpose(M, (0, 2, 1))
    # undo the double-count on the diagonal
    idx = np.arange(4)
    M[:, idx, idx] *= 0.5
    return M


def fstat_full(N, M):
    """F = 1/2 N^T M^-1 N with the full 4x4 solve."""
    return 0.5 * np.einsum("ni,nij,nj->n", N, np.linalg.pinv(M), N)


def abcd_from_M(M):
    """Read the four independent numbers off M (averaging equal entries)."""
    A = 0.5 * (M[:, 0, 0] + M[:, 2, 2])
    B = 0.5 * (M[:, 1, 1] + M[:, 3, 3])
    C = 0.5 * (M[:, 0, 1] + M[:, 2, 3])
    D = 0.5 * (M[:, 1, 2] - M[:, 0, 3])
    return A, B, C, D


def M_from_abcd(A, B, C, D):
    """Rebuild the structured M from (A, B, C, D)."""
    n = A.shape[0]
    Z = np.zeros(n)
    return np.stack([
        np.stack([A, C, Z, -D], axis=1),
        np.stack([C, B, D, Z], axis=1),
        np.stack([Z, D, A, C], axis=1),
        np.stack([-D, Z, C, B], axis=1),
    ], axis=1)


def fstat_block(N, M):
    """F using ONLY the 4 structured numbers (A, B, C, D)."""
    A, B, C, D = abcd_from_M(M)
    return fstat_full(N, M_from_abcd(A, B, C, D))


def structure_report(N, M):
    n = M.shape[0]
    diag = np.sqrt(np.abs(M[:, 0, 0] * M[:, 1, 1])) + 1e-300

    # Pairs (0,2) and (1,3) are (Re, Im) of the SAME complex waveform, so
    # their cross terms are the pure bilinear Int(UU) -- these are the ones
    # the carrier averages away.
    zer = np.stack([M[:, 0, 2], M[:, 1, 3]], axis=1)
    zer_rel = np.abs(zer).max(axis=1) / diag

    # Entries the structure claims are equal / antisymmetric.
    eqA = np.abs(M[:, 0, 0] - M[:, 2, 2]) / diag
    eqB = np.abs(M[:, 1, 1] - M[:, 3, 3]) / diag
    eqC = np.abs(M[:, 0, 1] - M[:, 2, 3]) / diag
    eqD = np.abs(M[:, 0, 3] + M[:, 1, 2]) / diag

    def q(x):
        return (np.median(x), np.percentile(x, 90), np.max(x))

    print(f"\n=== M structure over {n} candidates "
          f"(relative to sqrt(M00*M11)) ===")
    print(f"  bilinear |M02|,|M13|  -> 0        median {q(zer_rel)[0]:.3e}  "
          f"p90 {q(zer_rel)[1]:.3e}  max {q(zer_rel)[2]:.3e}")
    print(f"  |M00-M22|  (A)                    median {q(eqA)[0]:.3e}  "
          f"p90 {q(eqA)[1]:.3e}  max {q(eqA)[2]:.3e}")
    print(f"  |M11-M33|  (B)                    median {q(eqB)[0]:.3e}  "
          f"p90 {q(eqB)[1]:.3e}  max {q(eqB)[2]:.3e}")
    print(f"  |M01-M23|  (C)                    median {q(eqC)[0]:.3e}  "
          f"p90 {q(eqC)[1]:.3e}  max {q(eqC)[2]:.3e}")
    print(f"  |M03+M12|  (D antisym)            median {q(eqD)[0]:.3e}  "
          f"p90 {q(eqD)[1]:.3e}  max {q(eqD)[2]:.3e}")

    Ff, Fb = fstat_full(N, M), fstat_block(N, M)
    # Gate on CONDITIONING, not on |F|: with synthetic data most random
    # candidates sit off-source and have F ~ 0, which is still a valid
    # test of the algebra.
    cond = np.linalg.cond(M)
    good = np.isfinite(Ff) & np.isfinite(Fb) & (cond < 1e10)
    scale = np.maximum(np.abs(Ff), np.percentile(np.abs(Ff), 90) * 1e-3)
    rel = np.abs(Fb[good] - Ff[good]) / scale[good]
    print(f"\n=== F: block form vs full 4x4 ({good.sum()} usable) ===")
    print(f"  |dF|/F   median {np.median(rel):.3e}  p90 "
          f"{np.percentile(rel, 90):.3e}  max {np.max(rel):.3e}")
    loud = good & (Ff > 12.5)                     # F >= SNR 5, the peak floor
    if loud.sum():
        rl = np.abs(Fb[loud] - Ff[loud]) / scale[loud]
        print(f"  loud only (F > 12.5, n={loud.sum()}): median "
              f"{np.median(rl):.3e}  max {np.max(rl):.3e}")
    return float(np.median(rel)) if good.sum() else np.nan


def build_fit(args):
    """Minimal stock fit whose gb_wdm_comp + acs can score F-stat."""
    from lisatools.globalfit.stock import erebor

    os.environ.setdefault("GB_MODE", "pe")
    fit = erebor.gb_no_fg(nwalkers=2, ntemps=1)
    fit.general.data_mode = args.data_mode
    fit.general.file_store_dir = args.out
    fit.gb.min_freq, fit.gb.max_freq = args.f_lo, args.f_hi
    fit.general.min_freq, fit.general.max_freq = args.f_lo, args.f_hi
    os.makedirs(args.out, exist_ok=True)
    curr = fit.build()
    return fit, curr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="candidates to score")
    ap.add_argument("--data-mode", default="synthetic")
    ap.add_argument("--f-lo", type=float, default=6.0e-3)
    ap.add_argument("--f-hi", type=float, default=7.0e-3)
    ap.add_argument("--out", default="./gf_fstat_bilinear/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    from mpi4py import MPI
    from eryn.state import BranchSupplemental
    from lisatools.globalfit.run import GlobalFit

    print("[build] stock gb_no_fg ...", flush=True)
    fit, curr = build_fit(args)
    gf = GlobalFit(curr, MPI.COMM_WORLD)
    priors = {}
    for nm in curr.branch_names:
        priors.update(curr.source_info[nm].priors)
    state = gf.load_info(priors)
    nt, nw = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nw), (nt, 1))},
        base_shape=(nt, nw), copy=True)
    acs = gf.setup_acs(state)

    # The chunked-het comp is built by the variant's setup_function during
    # recipe materialization, NOT by fit.build() -- mirror run.py:1309-1316.
    recipe = curr.source_metadata["recipe"]
    recipe._init_runtime()
    print("[build] materializing recipe (builds gb_wdm_comp) ...", flush=True)
    curr.settings_dict.setup_function(
        recipe, gf.engine_info, curr, acs, priors, state)

    gb_info = curr.source_info["gb"]
    comp = getattr(gb_info.gb_wdm_comp, "chunked", gb_info.gb_wdm_comp)
    if comp is None:
        raise RuntimeError("gb_wdm_comp is still None after recipe setup")
    xp = comp.xp
    print(f"[build] comp={type(comp).__name__} backend={comp.backend.name}")

    # Candidates spread over the band and the sky; the F-stat basis only
    # depends on (f0, fdot, lam, beta), extrinsics are fixed inside.
    rng = np.random.default_rng(args.seed)
    n = args.n
    band = np.asarray(fit.gb.f0_lims if fit.gb.f0_lims else (args.f_lo, args.f_hi))
    f0 = rng.uniform(band[0], band[1], n)
    p = np.zeros((n, 9))
    p[:, 0] = 1e-22                       # amp (irrelevant to the basis)
    p[:, 1] = f0
    p[:, 2] = 1e-16
    p[:, 4] = rng.uniform(0, 2 * np.pi, n)
    p[:, 5] = np.arccos(rng.uniform(-1, 1, n))
    p[:, 6] = rng.uniform(0, np.pi, n)
    p[:, 7] = rng.uniform(0, 2 * np.pi, n)
    p[:, 8] = np.arcsin(rng.uniform(-1, 1, n))

    print(f"[fstat] scoring {n} candidates ...", flush=True)
    N, M_flat = comp.get_fstat_ll_wdm(
        xp.asarray(p), acs,
        data_index=xp.zeros(n, dtype=xp.int32),
        noise_index=xp.zeros(n, dtype=xp.int32),
        convert_to_ra_dec=False,
    )
    N = np.asarray(N.get() if hasattr(N, "get") else N)
    M_flat = np.asarray(M_flat.get() if hasattr(M_flat, "get") else M_flat)
    M = unpack_M(M_flat)

    med = structure_report(N, M)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.sqrt(np.abs(M[:, 0, 0] * M[:, 1, 1])) + 1e-300
        rows = [("bilinear M02,M13 -> 0",
                 np.abs(np.stack([M[:, 0, 2], M[:, 1, 3]], 1)).max(1) / d),
                ("|M00-M22|  (A)", np.abs(M[:, 0, 0] - M[:, 2, 2]) / d),
                ("|M11-M33|  (B)", np.abs(M[:, 1, 1] - M[:, 3, 3]) / d),
                ("|M01-M23|  (C)", np.abs(M[:, 0, 1] - M[:, 2, 3]) / d),
                ("|M03+M12|  (D)", np.abs(M[:, 0, 3] + M[:, 1, 2]) / d)]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        bins = np.logspace(-8, 0, 60)
        for lab, v in rows:
            ax[0].hist(np.clip(v, 1e-8, None), bins=bins, histtype="step",
                       lw=1.8, label=lab)
        ax[0].set_xscale("log"); ax[0].set_xlabel("relative residual")
        ax[0].set_ylabel("candidates"); ax[0].legend(frameon=False, fontsize=8)
        ax[0].set_title("M has only 4 independent numbers")
        Ff2, Fb2 = fstat_full(N, M), fstat_block(N, M)
        m = np.isfinite(Ff2) & np.isfinite(Fb2)
        ax[1].loglog(np.abs(Ff2[m]) + 1e-12, np.abs(Fb2[m] - Ff2[m]) + 1e-18,
                     ".", ms=3, alpha=.5)
        ax[1].set_xlabel("F (full 4x4)"); ax[1].set_ylabel("|F_struct - F_full|")
        ax[1].set_title("structured F vs full F")
        for a in ax: a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fstat_bilinear_structure.png"), dpi=140)
        print(f"[plot] {args.out}/fstat_bilinear_structure.png")
    except Exception as exc:
        print(f"[plot] skipped ({exc})")

    print("\n=== implication for a v5 F-stat kernel ===")
    if np.isfinite(med) and med < 1e-3:
        print("  The 2x2 block form holds. A v5 F-stat variant needs")
        print("    4 accumulators for N  +  4 for M   =  8   (not 4 + 10 = 14)")
        print("  and a 2x2 solve instead of 4x4. Half the shared-memory")
        print("  accumulator traffic of the naive port.")
    else:
        print("  Block form does NOT hold at this accuracy -- the kernel")
        print("  needs the full 10-entry M. Check whether the off-block")
        print("  terms track observation time (they should fall as the")
        print("  carrier accumulates cycles).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
