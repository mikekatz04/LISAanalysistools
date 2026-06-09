// signal_het_views.hpp
//
// POD view structs for the LAT-side signal-heterodyne C++ kernels. These
// match the plan in /Users/mkatz/.claude/plans/yes-find-and-read-sprightly-garden.md
// and are designed to be byte-equivalent on CPU and GPU (no inheritance,
// no virtual functions). When the plan's LAT cutils carve-out lands, these
// move under `LISAanalysistools/src/lisatools/cutils/`.

#pragma once

#include <complex>
#include <cstddef>

#ifdef __CUDACC__
  #include <cuda/std/complex>
  using cmplx_t = cuda::std::complex<double>;
#else
  using cmplx_t = std::complex<double>;
#endif

namespace signal_het {

// Describes the polyphase fold + iFFT operation for one binary
// across `m_active_layers` active m-layers.
//
// All buffers are device-resident (or host-resident for the CPU mirror).
// Lifetime: caller-managed; the view does not own the memory.
struct PolyphaseView {
    int            Nt;                  // window-support width (= full WDM Nt)
    int            Nt_layer;            // polyphase iFFT length (= output strided n count)
    int            stride;              // = Nt / Nt_layer (must be exact divisor)
    int            Nf;                  // total WDM m-layers (Nf, for layer-centre indexing)
    int            N;                   // = Nf * Nt (full FD-bin space)
    int            ind_min_t;           // start of active n in the Nt grid
    int            Nt_active;           // active n-count
    int            m_active_layers;     // typically 5 for GB (m_floor +- 2)
    int            nchannels;           // typically 3 (X, Y, Z)
    double         data_dt;             // TD sample step
    // Per-layer phitilde window evaluated on the Nt FD bins around the layer centre:
    // window[i] for i in [0, Nt). Same window for every layer (layer-centre-relative).
    const double  *window;              // (Nt,) real
};

// FD source view: the candidate's absolute h(f) at the FD bins for each active layer.
// For Stage 1 testing this is supplied as a precomputed buffer (Python rfft).
// For Stage 2 GBAbsoluteFD it is filled in-kernel by the source class.
struct AbsoluteFDView {
    const cmplx_t *h_layers;            // (nchannels, m_active_layers, Nt) complex
    const int     *m_layers;            // (m_active_layers,) global m indices
    int            m_active_layers;
    int            Nt;
    int            nchannels;
};

// Heterodyne reference view: precomputed at construction.
// c0_sparse[c, m_active_local, n_sparse] -- lisatools complex WDM at x_inj,
// restricted to the active m-band and sampled at sparse-n bin centres.
struct HeterodyneRefView {
    const cmplx_t *c0_sparse;           // (nchannels, m_active_layers, Nt_layer) complex
    int            m_active_layers;
    int            Nt_layer;            // == sparse n count == polyphase output length
    int            nchannels;
};

// Sparse-grid descriptor: positions of the sparse n bin centres in the
// LOCAL active n-index space [0, Nt_active).
struct SparseGridView {
    const int    *n_sparse_local;       // (Nt_layer,)
    int           Nt_layer;
    int           stride;
};

// Bin-folded coefficients view: precomputed at construction from
// (c0, data, invC) on the FULL Nf_active band x Nt_active.
//
// A_*[c', m_full_local, b] -- (nchannels, Nf_active, Nt_layer) complex
//   (used in get_ll: <d|h> = sum_{c',m,b} A0 r + A1 dr)
// B_*[c, c', m_full_local, b] -- (nchannels, nchannels, Nf_active, Nt_layer) complex (XYZ)
//   (used in get_ll: <h|h> = sum_{c,c',m,b} B0 r.conj() r + B1 cross-r-dr)
struct BinFoldView {
    const cmplx_t *A0;                  // (nchannels, Nf_active, Nt_layer)
    const cmplx_t *A1;                  // same shape
    const cmplx_t *B0;                  // (nchannels, nchannels, Nf_active, Nt_layer) for XYZ
    const cmplx_t *B1;                  // same shape as B0
    int            nchannels;
    int            Nf_active;           // FULL active-m count (= ind_max_f - ind_min_f + 1)
    int            Nt_layer;            // sparse n count
    int            ind_min_f;           // global-m offset
    int            tdi_type;            // 0=XYZ (use B0/B1), 1=AE/AET (B0/B1 diagonal)
};

// Carrier de-rotation predictor: analytic phase = 2*pi * Df0 * t + pi * Dfdot * t^2
// where t = (ind_min_t + n) * Nf * dt. Df0 and Dfdot are taken from
// (candidate - reference) params.
struct DerotateView {
    double Df0;                         // f0_cand - f0_ref
    double Dfdot;                       // fdot_cand - fdot_ref
    double layer_dt;                    // = Nf * data_dt
    int    ind_min_t;
};

// Linear-interpolation descriptor mapping dense n in [0, Nt_active) to a
// pair of sparse-bin indices + weight. Computed once at construction from
// SparseGridView.
struct InterpView {
    const int    *bracket_lo;           // (Nt_active,)
    const double *offset;               // (Nt_active,) = (n - n_sparse[bracket_lo]) / stride
    int           Nt_active;
};

}  // namespace signal_het
