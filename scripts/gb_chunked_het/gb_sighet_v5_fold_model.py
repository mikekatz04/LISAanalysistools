"""Sig-het V5 fold-stage model: prove the v5 algebra IS v4's, without a build.

This is the cheap gate. ``gb_sighet_v5_parity.py`` is the real one -- it runs
the compiled kernels -- but it needs a rebuilt GBGPU wheel and a scaffold. This
script needs neither, so it is the thing to run first after touching the v5
reconstruction, and the thing that localises a failure when the compiled gate
goes red.

Scope is deliberately narrow. Everything up to and including ``r_pix`` (the
node stage, the fixed-knot resample, the banded cardinal application) is v4
VERBATIM in v5 -- it is spliced out of the v4 scorer at build time, not
retyped -- so this model starts where v5 actually diverges: the |c0| row
floor, the mask, the centred difference and the two fold branches.

v4 (transcribed from ``gb_signal_het_v4_score_one_source``):

    per (c, im) row:  max_mag = max_b |c0_row[b]|          (serial scan)
                      floor   = max(1e-12 * max_mag, 1e-300)
                      r_row[b]  = |c0_row[b]| > floor ? r_pix[c][b] : 0
                      dr_row[b] = centred difference of r_row
    fold: d_h += A0*r + A1*dr ; h_h += B0*<rr> + B1*<cross> (+ nc terms)

``r_sparse`` and ``dr_sparse`` are materialised in shared memory: together
2*nch*M*N*16 = 480 B per sparse-time point at nch=3, M=5, on top of r_pix's
48 B. That is 73% of v4's 143.4 KB footprint and it is why only one block is
resident per SM.

v5 materialises neither:

  * ``max_mag`` and the mask are functions of (data_idx, c, m_local) ONLY --
    ``c0_sparse_all`` does not depend on the candidate -- so they move out of
    the scorer into ``setup_in_model`` as a bit-packed mask over the b axis
    (``GBSignalHetComputations._c0_row_mask_bits``);
  * the scorer rebuilds r and dr in registers from ``r_pix`` plus one bit.

Both arms below run the SAME thread-strided accumulation the kernel uses, so
a mismatch means the reconstruction is wrong, not that a sum reassociated.

Run:
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_v5_fold_model.py
Env:
    V5_MODEL_SEED (20260804)
"""
import os
import sys

import numpy as np

FLOOR_EPS = 1e-12
TINY = 1e-300


# ---------------------------------------------------------------------------
# v4: the materialised producer, transcribed statement for statement
# ---------------------------------------------------------------------------
def v4_producer(c0_rows, r_pix, row_channel, stride):
    """Exact transcription of v4's ``one thread per (c, im) row`` loop."""
    n_rows, N = c0_rows.shape
    r_sparse = np.zeros((n_rows, N), dtype=np.complex128)
    dr_sparse = np.zeros((n_rows, N), dtype=np.complex128)

    for row in range(n_rows):
        c = row_channel[row]
        c0_row = c0_rows[row]

        max_mag = 0.0                       # kernel starts at 0.0 and uses `>`
        for b in range(N):
            mag = abs(c0_row[b])
            if mag > max_mag:
                max_mag = mag
        floor_th_a = FLOOR_EPS * max_mag
        floor_th = floor_th_a if floor_th_a > TINY else TINY

        r_row = r_sparse[row]
        for b in range(N):
            r_row[b] = r_pix[c][b] if abs(c0_row[b]) > floor_th else 0.0

        Dn = float(stride)
        dr_row = dr_sparse[row]
        for b in range(N):
            d = 0.0 + 0.0j
            if N >= 3:
                if b == 0:
                    d = (r_row[1] - r_row[0]) / Dn
                elif b == N - 1:
                    d = (r_row[b] - r_row[b - 1]) / Dn
                else:
                    d = (r_row[b + 1] - r_row[b - 1]) / (2.0 * Dn)
            elif N == 2:
                d = (r_row[1] - r_row[0]) / Dn
            dr_row[b] = d

    return r_sparse, dr_sparse


# ---------------------------------------------------------------------------
# v5: the setup_in_model precompute + the scorer-side reconstruction
# ---------------------------------------------------------------------------
def v5_pack_mask(c0_full):
    """Mirror of ``GBSignalHetComputations._c0_row_mask_bits``.

    ``c0_full`` (n, nch, Nf_active, N). Returns uint64
    (n, nch, Nf_active, ceil(N/64)); bit b%64 of word b//64 is the keep flag.
    Kept independent of the shipping implementation on purpose: this file is
    the specification, and a divergence between the two is exactly the kind of
    bug it exists to catch.
    """
    mag = np.abs(c0_full)
    floor_a = FLOOR_EPS * mag.max(axis=-1)
    floor_th = np.where(floor_a > TINY, floor_a, TINY)
    keep = mag > floor_th[..., None]

    N = keep.shape[-1]
    nwords = (N + 63) // 64
    pad = nwords * 64 - N
    if pad:
        keep = np.concatenate(
            [keep, np.zeros(keep.shape[:-1] + (pad,), dtype=bool)], axis=-1)
    k = keep.reshape(keep.shape[:-1] + (nwords, 64)).astype(np.uint64)
    return (k << np.arange(64, dtype=np.uint64)).sum(axis=-1, dtype=np.uint64)


def _bit(words_row, b):
    """The scorer's mask test: one word read plus a shift."""
    return bool((words_row[b >> 6] >> np.uint64(b & 63)) & np.uint64(1))


def v5_r_dr(words_row, r_pix_c, b, N, stride):
    """Mirror of ``gb_sighet_v5_r_dr``: r and dr with no per-pixel scratch.

    The branch structure is v4's centred difference statement for statement --
    same operands, same order, same divisions -- which is what makes the fold
    bit-identical rather than merely equal to round-off.
    """
    def r_at(bb):
        return r_pix_c[bb] if _bit(words_row, bb) else 0.0 + 0.0j

    r = r_at(b)
    Dn = float(stride)
    d = 0.0 + 0.0j
    if N >= 3:
        if b == 0:
            d = (r_at(1) - r) / Dn
        elif b == N - 1:
            d = (r - r_at(b - 1)) / Dn
        else:
            d = (r_at(b + 1) - r_at(b - 1)) / (2.0 * Dn)
    elif N == 2:
        d = (r_at(1) - r_at(0)) / Dn
    return r, d


# ---------------------------------------------------------------------------
# the folds -- identical index mapping and accumulation order in both arms
# ---------------------------------------------------------------------------
def fold(get_r_dr, coefs, nch, M, N, m_local_of_im, data_idx, Nf_active,
         tdi_type, project_real, nthreads):
    """Thread-strided fold mirroring the kernel.

    ``get_r_dr(row, b)`` supplies (r, dr) and is the ONLY thing that differs
    between the arms. Per-thread partials are summed in thread order, matching
    the cub block reduce.
    """
    A0, A1, B0, B1, B0nc, B1nc = coefs

    def coef1(c, im, b):
        return ((data_idx * nch + c) * Nf_active + m_local_of_im[im]) * N + b

    def coef2(c, c2, im, b):
        return (((data_idx * nch + c) * nch + c2) * Nf_active
                + m_local_of_im[im]) * N + b

    n_dh = nch * M * N
    dh_parts = np.zeros(nthreads, dtype=np.complex128)
    for t in range(nthreads):
        acc = 0.0 + 0.0j
        for idx in range(t, n_dh, nthreads):
            c, im, b = idx // (M * N), (idx // N) % M, idx % N
            r, dr = get_r_dr(c * M + im, b)
            ci = coef1(c, im, b)
            acc += A0[ci] * r + A1[ci] * dr
        dh_parts[t] = acc

    hh_parts = np.zeros(nthreads, dtype=np.complex128)
    if tdi_type == 0:
        n_hh = nch * nch * M * N
        for t in range(nthreads):
            acc = 0.0 + 0.0j
            for idx in range(t, n_hh, nthreads):
                c = idx // (nch * M * N)
                c2 = (idx // (M * N)) % nch
                im, b = (idx // N) % M, idx % N
                r_c, dr_c = get_r_dr(c * M + im, b)
                r_c2, dr_c2 = get_r_dr(c2 * M + im, b)
                ci = coef2(c, c2, im, b)
                acc += (B0[ci] * (np.conj(r_c) * r_c2)
                        + B1[ci] * (np.conj(r_c) * dr_c2
                                    + np.conj(dr_c) * r_c2))
                if project_real:
                    acc += (B0nc[ci] * (r_c * r_c2)
                            + B1nc[ci] * (r_c * dr_c2 + dr_c * r_c2))
            hh_parts[t] = acc
    else:
        n_hh = nch * M * N
        for t in range(nthreads):
            acc = 0.0 + 0.0j
            for idx in range(t, n_hh, nthreads):
                c, im, b = idx // (M * N), (idx // N) % M, idx % N
                r, dr = get_r_dr(c * M + im, b)
                ci = coef1(c, im, b)
                acc += (B0[ci] * (np.conj(r) * r).real
                        + B1[ci] * (np.conj(r) * dr + np.conj(dr) * r))
                if project_real:
                    acc += B0nc[ci] * (r * r) + B1nc[ci] * (r * dr + dr * r)
            hh_parts[t] = acc

    return dh_parts.sum().real, hh_parts.sum().real


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def make_case(rng, nch, m_half, N, Nf_active, zero_frac, dyn_decades):
    """c0 spanning many decades with exact zeros, so BOTH the FLOOR_EPS mask
    and the 1e-300 guard fire and whole rows can mask out entirely."""
    M = 2 * m_half + 1
    n_data, data_idx = 2, 1

    shape = (n_data, nch, Nf_active, N)
    scale = 10.0 ** rng.uniform(-dyn_decades, dyn_decades, size=shape)
    c0 = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale
    c0[rng.random(shape) < zero_frac] = 0.0
    c0[data_idx, 0, 0, :] = 0.0          # an all-zero row -> the 1e-300 branch

    r_pix = rng.standard_normal((nch, N)) + 1j * rng.standard_normal((nch, N))

    n1 = n_data * nch * Nf_active * N
    n2 = n_data * nch * nch * Nf_active * N
    def cx(n):
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)
    coefs = (cx(n1), cx(n1), cx(n2), cx(n2), cx(n2), cx(n2))

    m_local_of_im = np.clip(rng.integers(0, Nf_active, size=M),
                            0, Nf_active - 1)
    return dict(nch=nch, M=M, N=N, Nf_active=Nf_active, data_idx=data_idx,
                c0=c0, r_pix=r_pix, coefs=coefs, m_local_of_im=m_local_of_im)


def run_case(case, stride, tdi_type, project_real, nthreads):
    nch, M, N = case["nch"], case["M"], case["N"]
    Nf_active, data_idx = case["Nf_active"], case["data_idx"]
    c0, r_pix, m_local_of_im = case["c0"], case["r_pix"], case["m_local_of_im"]

    n_rows = nch * M
    row_channel = np.array([row // M for row in range(n_rows)], dtype=int)
    c0_rows = np.stack([c0[data_idx, row // M, m_local_of_im[row % M], :]
                        for row in range(n_rows)])

    r_s, dr_s = v4_producer(c0_rows, r_pix, row_channel, stride)
    words = v5_pack_mask(c0)
    row_words = np.stack([words[data_idx, row // M, m_local_of_im[row % M], :]
                          for row in range(n_rows)])

    args = (case["coefs"], nch, M, N, m_local_of_im, data_idx, Nf_active,
            tdi_type, project_real, nthreads)
    return (fold(lambda row, b: (r_s[row, b], dr_s[row, b]), *args),
            fold(lambda row, b: v5_r_dr(row_words[row],
                                        r_pix[row_channel[row]],
                                        b, N, stride), *args))


def main():
    seed = int(os.environ.get("V5_MODEL_SEED", "20260804"))
    rng = np.random.default_rng(seed)
    n_fail = n_case = 0
    worst = 0.0

    print(f"[cfg] seed={seed}  sweeping nch x m_half x N_sparse_t x stride "
          f"x tdi_type x project_real")
    for nch in (1, 3):
        for m_half in (0, 2, 4):
            for N in (2, 3, 8, 65, 127, 204):
                for stride in (1, 7):
                    for tdi_type in (0, 1):
                        for project_real in (0, 1):
                            case = make_case(rng, nch, m_half, N,
                                             Nf_active=max(3, m_half + 2),
                                             zero_frac=0.05, dyn_decades=7.0)
                            (d4, h4), (d5, h5) = run_case(
                                case, stride, tdi_type, project_real,
                                nthreads=1 if N < 8 else 128)
                            n_case += 1
                            if (d4 != d5) or (h4 != h5):
                                n_fail += 1
                                den = max(abs(d4), abs(h4), TINY)
                                worst = max(worst,
                                            (abs(d4 - d5) + abs(h4 - h5)) / den)
                                print(f"  MISMATCH nch={nch} m_half={m_half} "
                                      f"N={N} stride={stride} tdi={tdi_type} "
                                      f"pr={project_real}\n"
                                      f"    d_h {d4!r} vs {d5!r}\n"
                                      f"    h_h {h4!r} vs {h5!r}")

    verdict = "BIT-IDENTICAL" if n_fail == 0 else f"FAIL (worst rel {worst:.3e})"
    print(f"\n{n_case} cases, {n_fail} mismatches -> {verdict}")
    print("\nGATE: " + ("PASS" if n_fail == 0 else "FAIL"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
