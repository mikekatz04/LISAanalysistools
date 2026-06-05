"""
Python reference implementation of the spline-WDM kernel pipeline.

Mirrors the planned C++/CUDA design in
  lisa-on-gpu/src/fastlisaresponse/cutils/WDMSplineHelpers.hh
exactly enough that the C++ port is a mechanical translation.  Validated
against ``lisatools.domains.FDSignal.wdmtransform`` (see
``test_fd_spline_wdm.py`` in the repo root).

The pipeline, per binary:

  1. Synthesize the TD waveform on the spline grid (callback supplied by
     the caller) into a length-N transient buffer.
  2. Take the full DFT of that TD via a 2-stage row-column decomposition:
     ``Nt`` FFTs of size ``Nf`` then ``Nf`` FFTs of size ``Nt``.  Each
     stage uses **Bluestein** so general-even ``Nf``, ``Nt`` work.
  3. For each WDM frequency layer ``m`` in ``[m_min, m_max]``: gather the
     Nt-bin slice from the FD, multiply by the precomputed Cmm window,
     run an IFFT of size ``Nt_layer`` (full or per-layer narrow), apply
     the ``(m+n)``-parity sign/real-imag selector on the *global* n
     grid, and feed it to one of three "finishers":

       * ``fill_global``  -- atomic-add into a sparse template buffer
       * ``get_ll``       -- accumulate <d|h> and <h|h> against a
         WDMDomain-like object (data + per-pixel diag noise)
       * ``swap_ll``      -- the five RJMCMC swap inner products

The narrow-window path is per-(binary, layer): ``narrow_widths[bin, m]``
== 0 means use the full transform; > 0 (and even) uses an
``Nt_narrow``-sized layer IFFT centered on ``narrow_centers[bin, m]``.
The (m+n)-parity uses the **global** n index, never the local IFFT
index, so the kernel handles either ``(m + n_global_start) % 2``
correctly -- this is the "evens/odds" subtlety called out in the spec.

Designed so the C++ port is a 1:1 translation: each Python loop
corresponds to a CUDA block-cooperative loop (THREAD_START /
BLOCK_INCR), and shared-memory allocations correspond to the
shared-memory carve-up at the top of each kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.special import betainc

# ---------------------------------------------------------------------------
# Bluestein FFT for general-even N -- exact N-point DFT/IDFT.
# ---------------------------------------------------------------------------


def _next_pow2_geq(n: int) -> int:
    m = 1
    while m < n:
        m <<= 1
    return m


def bluestein_pad_length(N: int) -> int:
    """Internal radix-2 length for an N-point Bluestein FFT."""
    return _next_pow2_geq(2 * N - 1)


@dataclass
class BluesteinPlan:
    """Precomputed chirp arrays for one N.  ``N`` is the *user-visible*
    DFT length; ``M = bluestein_pad_length(N)`` is the internal radix-2
    convolution length.  The output of ``fft(plan, x)`` is the exact
    N-point DFT of ``x`` -- the internal M is invisible to callers."""

    N: int
    M: int
    chirp_n: np.ndarray  # (N,) cmplx, exp(-i pi k^2 / N)
    chirp_M_fft: np.ndarray  # (M,) cmplx, FFT of zero-padded b_k

    @classmethod
    def build(cls, N: int) -> "BluesteinPlan":
        if N <= 1:
            raise ValueError("Bluestein requires N >= 2")
        M = bluestein_pad_length(N)
        k = np.arange(N)
        # Forward chirp: exp(-i pi k^2 / N)
        phase = -np.pi * (k.astype(np.float64) ** 2) / float(N)
        chirp_n = np.exp(1j * phase)
        # Build b: length M, b[0] = 1, b[k] = b[M-k] = exp(+i pi k^2 / N) for k=1..N-1
        b = np.zeros(M, dtype=np.complex128)
        b[0] = 1.0
        k_pos = np.arange(1, N)
        b_vals = np.exp(1j * np.pi * (k_pos.astype(np.float64) ** 2) / float(N))
        b[1:N] = b_vals
        b[M - (N - 1) : M] = b_vals[::-1]
        chirp_M_fft = np.fft.fft(b)
        return cls(N=N, M=M, chirp_n=chirp_n, chirp_M_fft=chirp_M_fft)


def bluestein_fft(plan: BluesteinPlan, x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Exact N-point DFT (or IDFT) of ``x`` (length N) via Bluestein.

    Output length == N regardless of plan.M.  The plan's power-of-2 M is
    invisible to callers -- you get the full N-point spectrum back.
    """
    N = plan.N
    if x.shape[-1] != N:
        raise ValueError(f"x.shape[-1] must be {N}, got {x.shape[-1]}")
    cn = plan.chirp_n.conj() if inverse else plan.chirp_n
    cM = plan.chirp_M_fft.conj() if inverse else plan.chirp_M_fft

    # Pad: a[k] = x[k] * cn[k] for k in [0,N), zero elsewhere
    pad = np.zeros(x.shape[:-1] + (plan.M,), dtype=np.complex128)
    pad[..., :N] = x * cn
    # FFT of pad
    A = np.fft.fft(pad, axis=-1)
    # Pointwise multiply
    B = A * cM
    # IFFT (numpy's IFFT carries 1/M internally)
    C = np.fft.ifft(B, axis=-1)
    # Output: y[k] = C[k] * cn[k], scaled by 1/N for the inverse
    y = C[..., :N] * cn
    if inverse:
        y = y / float(N)
    return y


# ---------------------------------------------------------------------------
# 2-stage row-column FFT of size N = Nf * Nt.
# ---------------------------------------------------------------------------
#
# We choose the index layout n = a*Nt + b, a in [0, Nf), b in [0, Nt), so
# that the input matrix is x[a, b] with row stride Nt.  The DFT then
# splits as:
#
#   X[k_f + Nf * k_t] = sum_b [ FFT_Nf(x[:, b])[k_f] * W[k_f, b] ] * FFT_Nt[k_t]
#
# where W[k_f, b] = exp(-i 2*pi * k_f * b / N).  The 2-stage routine
# returns the FD as a 2D matrix indexed by (k_f, k_t).
#


@dataclass
class TwoStageFFTPlan:
    Nf: int
    Nt: int
    plan_Nf: BluesteinPlan
    plan_Nt: BluesteinPlan

    @classmethod
    def build(cls, Nf: int, Nt: int) -> "TwoStageFFTPlan":
        if Nf % 2 != 0 or Nt % 2 != 0:
            raise ValueError("Nf and Nt must both be even")
        return cls(Nf=Nf, Nt=Nt, plan_Nf=BluesteinPlan.build(Nf), plan_Nt=BluesteinPlan.build(Nt))


def two_stage_fft(plan: TwoStageFFTPlan, td_real: np.ndarray) -> np.ndarray:
    """Return the (Nf, Nt) FD matrix indexed by (k_f, k_t).

    td_real is real-valued, shape (..., N) with N = Nf * Nt.  Output is
    cmplx, shape (..., Nf, Nt).  The output index ``X[k_f + Nf*k_t]`` is
    the FFT of the input at that (single) frequency bin.
    """
    Nf, Nt = plan.Nf, plan.Nt
    N = Nf * Nt
    if td_real.shape[-1] != N:
        raise ValueError(f"input length must be {N}, got {td_real.shape[-1]}")
    # Reshape to (..., Nf, Nt) with n = a*Nt + b
    x = td_real.reshape(td_real.shape[:-1] + (Nf, Nt)).astype(np.complex128)

    # Stage A: Nt FFTs of size Nf along axis -2.
    # bluestein_fft operates on axis -1, so transpose temporarily.
    # x has shape (..., Nf, Nt); we want to FFT the "Nf" axis for each (..., :, b).
    # Move that axis to last:
    xT = np.moveaxis(x, -2, -1)  # (..., Nt, Nf)
    S = bluestein_fft(plan.plan_Nf, xT)  # (..., Nt, Nf)  -- FFT along last axis
    S = np.moveaxis(S, -1, -2)  # (..., Nf, Nt)

    # Twiddle: S'[a, b] = S[a, b] * exp(-i 2*pi * a * b / N)
    a = np.arange(Nf)[:, None]
    b = np.arange(Nt)[None, :]
    W = np.exp(-1j * 2.0 * np.pi * (a * b).astype(np.float64) / float(N))
    S = S * W  # broadcast

    # Stage B: Nf FFTs of size Nt along axis -1.
    FD = bluestein_fft(plan.plan_Nt, S)
    return FD  # shape (..., Nf, Nt) indexed by (k_f, k_t)


def two_stage_fft_to_dense(fd_2d: np.ndarray) -> np.ndarray:
    """Convert the (Nf, Nt) FD matrix to a dense (N,) FFT array.

    X[k_f + Nf * k_t] = fd_2d[..., k_f, k_t].  Useful for cross-checking
    against numpy.fft.fft.
    """
    Nf, Nt = fd_2d.shape[-2], fd_2d.shape[-1]
    # Rebuild dense X by index mapping
    X = np.empty(fd_2d.shape[:-2] + (Nf * Nt,), dtype=np.complex128)
    for kt in range(Nt):
        X[..., kt * Nf : (kt + 1) * Nf] = fd_2d[..., :, kt]
    return X


# ---------------------------------------------------------------------------
# WDM analysis window.  Matches WDMSettings.phitilde + setup_window in
# lisatools/domains.py, parametrized by the layer IFFT length Nt_layer.
# ---------------------------------------------------------------------------


def _phitilde(omega: np.ndarray, dOmega: float) -> np.ndarray:
    insDOM = 1.0 / np.sqrt(dOmega)
    A = dOmega / 4.0
    B = dOmega - 2.0 * A
    abs_om = np.abs(omega)
    z = np.zeros_like(omega)
    inner = abs_om < A
    z[inner] = insDOM
    band = (abs_om >= A) & (abs_om <= A + B)
    if np.any(band):
        x = (abs_om[band] - A) / B
        y = betainc(4, 4, x)
        z[band] = insDOM * np.cos(y * np.pi / 2.0)
    return z


def build_wdm_window(Nt_layer: int, dOmega_s: Optional[float] = None) -> np.ndarray:
    """Build the WDM analysis window for layer IFFT length Nt_layer.

    For the full transform with Nt_layer = Nt, pass ``dOmega_s = pi/Nf``
    to match the standard WDM normalization.  For a narrow window of
    length Nt_narrow, you typically pass the same dOmega_s as the full
    transform (the narrower IFFT then samples that window at coarser
    bin spacing, giving a basis function with wider time support).
    """
    if dOmega_s is None:
        dOmega_s = 2.0 * np.pi / float(Nt_layer)
    omega = 2.0 * np.pi / float(Nt_layer) * (np.arange(Nt_layer) - Nt_layer // 2)
    return _phitilde(omega, dOmega_s)


# ---------------------------------------------------------------------------
# Per-layer WDM extraction (the algorithmic core).
# ---------------------------------------------------------------------------


def extract_layer_wdm(
    fd_full: np.ndarray,
    window: np.ndarray,
    m: int,
    Nf: int,
    Nt: int,
    Nt_layer: int,
    n_global_start: int,
    kappa: float,
    data_dt: float,
    layer_ifft_plan: Optional[BluesteinPlan] = None,
) -> np.ndarray:
    """Compute ``w_mn[i]`` for ``i in [0, Nt_layer)`` of layer ``m``.

    Args
    ----
    fd_full     : (N,) cmplx -- dense FD of the binary's TDI channel.
                  N == Nf * Nt.
    window      : (Nt_layer,) double -- precomputed Cmm window for this
                  layer IFFT length.
    m           : layer index in [0, Nf] (inclusive of the Nyquist Nf).
    Nf, Nt      : full WDM dimensions.
    Nt_layer    : the layer-IFFT length used here (full = Nt, narrow < Nt).
                  Must be even.
    n_global_start : global n-index of pixel i=0.  For the full transform
                  this is 0; for a narrow window centered at n_c with
                  width Nt_narrow it is n_c - Nt_narrow/2.
    kappa       : 2 * sqrt(pi * data_dt) / Nf (matches domains.py:823)
    data_dt     : the underlying TD sample step.
    layer_ifft_plan : Bluestein plan for the layer IFFT.  Built on the
                  fly if None (cache externally for performance).

    Returns
    -------
    w_mn[i] for i in [0, Nt_layer).  Edge layers (m==0 or m==Nf) zero out
    the pixels where (m + n_global) is odd -- caller is responsible for
    DC/Nyquist merging.
    """
    if Nt_layer % 2 != 0:
        raise ValueError("Nt_layer must be even")
    N = Nf * Nt
    if layer_ifft_plan is None:
        layer_ifft_plan = BluesteinPlan.build(Nt_layer)

    # Bin centre for this layer in the full FD
    centre = m * (Nt_layer // 2)
    # Gather Nt_layer bins around the centre, with Hermitian wraparound for k<0 or k>N/2
    i_local = np.arange(Nt_layer)
    kk = centre + (i_local - Nt_layer // 2)
    k_use = kk.copy()
    conj_mask = np.zeros(Nt_layer, dtype=bool)
    neg = kk < 0
    over = kk > (N // 2)
    k_use[neg] = -kk[neg]
    k_use[over] = N - kk[over]
    conj_mask = neg | over
    # Clip to range (out-of-range bins get zero).  Convention: fd_full is the
    # raw numpy.fft.fft(td) (no dt scaling) -- the driver supplies it that
    # way after two_stage_fft.  If a caller passes lisatools-style FDSignal
    # arrays (which carry an extra factor of dt), they should divide by dt
    # first.
    in_range = (k_use >= 0) & (k_use < N)
    vals = np.zeros(Nt_layer, dtype=np.complex128)
    vals[in_range] = fd_full[k_use[in_range]]
    vals[conj_mask] = np.conj(vals[conj_mask])
    # Window
    vals = vals * window
    # Inverse FFT (general even N)
    ifft_out = bluestein_fft(layer_ifft_plan, vals, inverse=True)
    # (m+n)-parity on the global grid
    n_global = n_global_start + i_local
    parity = (m + n_global) & 1
    phase_p = ((m + 1) * n_global) & 1
    sign = np.where(phase_p, -1.0, +1.0)
    val = np.where(parity, ifft_out.imag, ifft_out.real)
    w_mn = kappa * sign * val
    # Edge-layer zero (caller handles DC/Nyquist merging separately)
    if m == 0 or m == Nf:
        edge_zero = parity.astype(bool)
        w_mn = np.where(edge_zero, 0.0, w_mn)
    return w_mn


# ---------------------------------------------------------------------------
# Top-level per-binary driver: full TD->FD->WDM pipeline with optional
# narrow-window per layer.
# ---------------------------------------------------------------------------


@dataclass
class WDMSplineKernelPlan:
    """Container holding all precomputed scaffolding used by one call."""

    Nf: int
    Nt: int
    data_dt: float
    m_min: int
    m_max: int
    n_min: int
    n_max: int

    big_fft_plan: TwoStageFFTPlan
    window_full: np.ndarray  # (Nt,)
    layer_full_plan: BluesteinPlan

    narrow_widths: list[int] = field(default_factory=list)
    narrow_windows: dict[int, np.ndarray] = field(default_factory=dict)
    narrow_plans: dict[int, BluesteinPlan] = field(default_factory=dict)

    @property
    def Nf_active(self) -> int:
        return self.m_max - self.m_min + 1

    @property
    def Nt_active(self) -> int:
        return self.n_max - self.n_min + 1

    @property
    def kappa(self) -> float:
        return 2.0 * np.sqrt(np.pi * self.data_dt) / float(self.Nf)

    @classmethod
    def build(
        cls,
        Nf: int,
        Nt: int,
        data_dt: float,
        m_min: int,
        m_max: int,
        n_min: int,
        n_max: int,
        narrow_widths: Optional[list[int]] = None,
    ) -> "WDMSplineKernelPlan":
        big = TwoStageFFTPlan.build(Nf, Nt)
        # Match domains.py:1597-1602: dOmega_s = pi/Nf, omega step = 2*pi/N over Nt samples.
        dOmega_s = np.pi / float(Nf)
        omega_full = 2.0 * np.pi / float(Nf * Nt) * (np.arange(Nt) - Nt // 2)
        window_full = _phitilde(omega_full, dOmega_s)
        layer_full_plan = BluesteinPlan.build(Nt)

        narrow_widths = list(narrow_widths or [])
        narrow_windows = {}
        narrow_plans = {}
        for w in narrow_widths:
            if w <= 0 or w % 2 != 0:
                raise ValueError(f"narrow width {w} must be a positive even int")
            # Narrow window: same physical analysis filter, sampled at the
            # narrower grid.  We use the *same* dOmega_s but evaluate at
            # the narrower omega grid spaced 2*pi/(Nf * w) so the
            # underlying wavelet is consistent with the global WDM.
            omega = 2.0 * np.pi / float(Nf * w) * (np.arange(w) - w // 2)
            narrow_windows[w] = _phitilde(omega, dOmega_s)
            narrow_plans[w] = BluesteinPlan.build(w)
        return cls(
            Nf=Nf,
            Nt=Nt,
            data_dt=data_dt,
            m_min=m_min,
            m_max=m_max,
            n_min=n_min,
            n_max=n_max,
            big_fft_plan=big,
            window_full=window_full,
            layer_full_plan=layer_full_plan,
            narrow_widths=narrow_widths,
            narrow_windows=narrow_windows,
            narrow_plans=narrow_plans,
        )


def synth_and_transform_one_binary(
    plan: WDMSplineKernelPlan,
    td_synthesizer: Callable[[np.ndarray], np.ndarray],
    nchannels: int = 1,
    narrow_widths_row: Optional[np.ndarray] = None,
    narrow_centers_row: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the WDM coefficients for one binary, all channels, all layers.

    Args
    ----
    plan : ``WDMSplineKernelPlan`` describing the grid.
    td_synthesizer : callable ``f(t_arr) -> td[c, n]`` returning the real
        TD waveform on the supplied time grid for all ``nchannels``
        channels.  Shape ``(nchannels, N)``.  This is the "in-kernel TD
        synth" hook -- in the C++ port it becomes a per-sample call into
        the spline + LISA response, never persisted.
    nchannels : number of TDI channels to compute.
    narrow_widths_row : (Nf_active,) int or None -- per-layer Nt_narrow
        (0 means use full transform).
    narrow_centers_row : (Nf_active,) int or None -- per-layer global
        n-index of the narrow-window centre.

    Returns
    -------
    w_mn : ``(nchannels, Nf_active, Nt_active)`` ndarray.  Pixels outside
        a narrow window for narrow layers are left at zero.
    """
    Nf, Nt = plan.Nf, plan.Nt
    N = Nf * Nt
    # Step 1: synthesize TD on the global sample grid
    t_arr = np.arange(N) * plan.data_dt
    td = td_synthesizer(t_arr)  # (nchannels, N), real
    if td.ndim == 1:
        td = td[None, :]
    if td.shape[0] < nchannels:
        raise ValueError(f"td_synthesizer returned {td.shape[0]} channels, need {nchannels}")
    # Step 2: per-channel 2-stage FFT
    fd_dense = np.empty((nchannels, N), dtype=np.complex128)
    for c in range(nchannels):
        fd_2d = two_stage_fft(plan.big_fft_plan, td[c])
        fd_dense[c] = two_stage_fft_to_dense(fd_2d)

    # Step 3: per-layer extraction.  Layer index m runs over [0, Nf]
    # (inclusive), with the m==Nf "layer" feeding the DC-merge for m==0
    # exactly as in domains.py.
    tmp_w_mn = np.zeros((nchannels, Nf + 1, Nt), dtype=np.float64)
    kappa = plan.kappa
    for m in range(Nf + 1):
        # Narrow row applies only to layers within the active range.
        Nt_layer = Nt
        n_global_start = 0
        window = plan.window_full
        layer_plan = plan.layer_full_plan
        active_in_layer_range = plan.m_min <= m <= plan.m_max
        if (
            narrow_widths_row is not None
            and active_in_layer_range
            and 0 <= (m - plan.m_min) < len(narrow_widths_row)
            and narrow_widths_row[m - plan.m_min] > 0
        ):
            w = int(narrow_widths_row[m - plan.m_min])
            if w not in plan.narrow_windows:
                raise ValueError(
                    f"narrow width {w} for layer {m} not in plan.narrow_widths {plan.narrow_widths}"
                )
            Nt_layer = w
            n_c = int(narrow_centers_row[m - plan.m_min])
            n_global_start = n_c - w // 2
            window = plan.narrow_windows[w]
            layer_plan = plan.narrow_plans[w]
        for c in range(nchannels):
            w_layer = extract_layer_wdm(
                fd_dense[c],
                window,
                m=m,
                Nf=Nf,
                Nt=Nt,
                Nt_layer=Nt_layer,
                n_global_start=n_global_start,
                kappa=kappa,
                data_dt=plan.data_dt,
                layer_ifft_plan=layer_plan,
            )
            if Nt_layer == Nt:
                tmp_w_mn[c, m, :] = w_layer
            else:
                # Narrow: scatter into the global pixel range
                for i in range(Nt_layer):
                    n_global = n_global_start + i
                    if 0 <= n_global < Nt:
                        tmp_w_mn[c, m, n_global] = w_layer[i]

    # DC/Nyquist merge (domains.py:830-832):
    # final[:, 0, 0::2] = tmp[:, 0, 0::2] / sqrt(2)
    # final[:, 0, 1::2] = tmp[:, Nf, 0::2] / sqrt(2)
    # final[:, 1:Nf]    = tmp[:, 1:Nf]
    w_mn_full = np.zeros((nchannels, Nf, Nt), dtype=np.float64)
    w_mn_full[:, 1:, :] = tmp_w_mn[:, 1:-1, :]
    w_mn_full[:, 0, 0::2] = tmp_w_mn[:, 0, 0::2] / np.sqrt(2.0)
    w_mn_full[:, 0, 1::2] = tmp_w_mn[:, Nf, 0::2] / np.sqrt(2.0)

    # Crop to active slice
    return w_mn_full[:, plan.m_min : plan.m_max + 1, plan.n_min : plan.n_max + 1]


# ---------------------------------------------------------------------------
# Three "finisher" wrappers: fill_global, get_ll, swap_ll.
# ---------------------------------------------------------------------------


def fill_global_wdm(
    plan: WDMSplineKernelPlan,
    td_synthesizers: list[Callable[[np.ndarray], np.ndarray]],
    template_fill: np.ndarray,  # (num_data, nchannels, Nf_active, Nt_active)
    data_index_all: np.ndarray,
    factors_all: np.ndarray,
    nchannels: int = 1,
    narrow_widths: Optional[np.ndarray] = None,  # (num_bin, Nf_active)
    narrow_centers: Optional[np.ndarray] = None,
) -> None:
    """Atomic-add each binary's WDM template into ``template_fill``."""
    num_bin = len(td_synthesizers)
    for bin_i in range(num_bin):
        w_mn = synth_and_transform_one_binary(
            plan,
            td_synthesizers[bin_i],
            nchannels=nchannels,
            narrow_widths_row=narrow_widths[bin_i] if narrow_widths is not None else None,
            narrow_centers_row=narrow_centers[bin_i] if narrow_centers is not None else None,
        )
        di = int(data_index_all[bin_i])
        fac = float(factors_all[bin_i])
        template_fill[di] += fac * w_mn


def get_ll_wdm(
    plan: WDMSplineKernelPlan,
    td_synthesizers: list[Callable[[np.ndarray], np.ndarray]],
    wdm_data: np.ndarray,  # (num_data, nchannels, Nf_active, Nt_active)
    wdm_invC: np.ndarray,  # (num_noise, nchannels, Nf_active, Nt_active) diagonal
    data_index_all: np.ndarray,
    noise_index_all: np.ndarray,
    nchannels: int = 1,
    narrow_widths: Optional[np.ndarray] = None,
    narrow_centers: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute <d|h>, <h|h> per binary.  Returns ``(d_h_out, h_h_out)``.

    Uses the diagonal-noise inner product
        (a|b) = 4 * sum_pixels a * b * invC
    (matches WDMDomain::add_ip_contrib for AET/AE).  For cross-channel
    XYZ noise, see the C++ port.
    """
    num_bin = len(td_synthesizers)
    d_h_out = np.zeros(num_bin, dtype=np.float64)
    h_h_out = np.zeros(num_bin, dtype=np.float64)
    for bin_i in range(num_bin):
        w_mn = synth_and_transform_one_binary(
            plan,
            td_synthesizers[bin_i],
            nchannels=nchannels,
            narrow_widths_row=narrow_widths[bin_i] if narrow_widths is not None else None,
            narrow_centers_row=narrow_centers[bin_i] if narrow_centers is not None else None,
        )
        di = int(data_index_all[bin_i])
        ni = int(noise_index_all[bin_i])
        d_h_out[bin_i] = 4.0 * np.sum(wdm_data[di] * w_mn * wdm_invC[ni])
        h_h_out[bin_i] = 4.0 * np.sum(w_mn * w_mn * wdm_invC[ni])
    return d_h_out, h_h_out


def swap_ll_wdm(
    plan: WDMSplineKernelPlan,
    td_synth_add: list[Callable[[np.ndarray], np.ndarray]],
    td_synth_remove: list[Callable[[np.ndarray], np.ndarray]],
    wdm_data: np.ndarray,
    wdm_invC: np.ndarray,
    data_index_all: np.ndarray,
    noise_index_all: np.ndarray,
    nchannels: int = 1,
    narrow_widths_add: Optional[np.ndarray] = None,
    narrow_centers_add: Optional[np.ndarray] = None,
    narrow_widths_remove: Optional[np.ndarray] = None,
    narrow_centers_remove: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Five RJMCMC swap inner products per binary."""
    num_bin = len(td_synth_add)
    d_h_add = np.zeros(num_bin)
    d_h_rem = np.zeros(num_bin)
    add_add = np.zeros(num_bin)
    rem_rem = np.zeros(num_bin)
    add_rem = np.zeros(num_bin)
    for bin_i in range(num_bin):
        w_a = synth_and_transform_one_binary(
            plan,
            td_synth_add[bin_i],
            nchannels=nchannels,
            narrow_widths_row=narrow_widths_add[bin_i] if narrow_widths_add is not None else None,
            narrow_centers_row=narrow_centers_add[bin_i] if narrow_centers_add is not None else None,
        )
        w_r = synth_and_transform_one_binary(
            plan,
            td_synth_remove[bin_i],
            nchannels=nchannels,
            narrow_widths_row=(
                narrow_widths_remove[bin_i] if narrow_widths_remove is not None else None
            ),
            narrow_centers_row=(
                narrow_centers_remove[bin_i] if narrow_centers_remove is not None else None
            ),
        )
        di = int(data_index_all[bin_i])
        ni = int(noise_index_all[bin_i])
        invC = wdm_invC[ni]
        d = wdm_data[di]
        d_h_add[bin_i] = 4.0 * np.sum(d * w_a * invC)
        d_h_rem[bin_i] = 4.0 * np.sum(d * w_r * invC)
        add_add[bin_i] = 4.0 * np.sum(w_a * w_a * invC)
        rem_rem[bin_i] = 4.0 * np.sum(w_r * w_r * invC)
        add_rem[bin_i] = 4.0 * np.sum(w_a * w_r * invC)
    return d_h_add, d_h_rem, add_add, rem_rem, add_rem
