"""
CuPy-based PSD likelihood computation.

This module provides the same functionality as `psd_likelihood_numba.py` but uses
CuPy's vectorized operations and vendor-optimized libraries (cuBLAS, cuSOLVER)
instead of custom CUDA kernels. This approach:

- Leverages highly-optimized vendor libraries for linear algebra
- Simplifies code maintenance (no manual kernel optimization)
- Often faster for small matrices (3×3) due to better memory access patterns
- More Pythonic and easier to debug

Trade-offs:
- Less fine-grained control over GPU execution
- May use more temporary memory for intermediate arrays
- Batch operations require careful broadcasting

Additionally includes optimized Numba CUDA kernels for maximum performance.
"""

import math
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    from numba import cuda, float64, complex128
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    cuda = None


CLIGHT = 299_792_458.0
LISA_L = 2.5e9
LISA_LT = LISA_L / CLIGHT

# Thread block size for likelihood kernels
NUM_THREADS_LIKE = 512


# =============================================================================
# Numba CUDA Device Functions
# =============================================================================

if HAS_NUMBA:
    @cuda.jit(device=True, inline=True)
    def _lisanoises_device(f, Soms_d_in, Sa_a_in):
        """LISA noise model - returns (Sa_nu, Soms_nu) in relative frequency units."""
        # Acceleration noise
        Sa_a = Sa_a_in * (1.0 + (0.4e-3 / f)**2) * (1.0 + (f / 8e-3)**4)
        Sa_d = Sa_a * (2.0 * math.pi * f)**(-4.0)
        Sa_nu = Sa_d * (2.0 * math.pi * f / CLIGHT)**2
        
        # OMS noise
        Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f)**4)
        Soms_nu = Soms_d * (2.0 * math.pi * f / CLIGHT)**2
        
        return Sa_nu, Soms_nu
    
    @cuda.jit(device=True, inline=True)
    def _sgal_device(fr, amp, alpha, sl1, kn, sl2):
        """Galactic confusion noise model."""
        return amp * math.exp(-(fr**alpha) * sl1) * fr**(-7.0/3.0) * 0.5 * (1.0 + math.tanh(-(fr - kn) * sl2))
    
    @cuda.jit(device=True, inline=True)
    def _wdconfusion_xx_device(f, Amp, alpha, sl1, kn, sl2, tdi2=False):
        """White dwarf confusion noise PSD."""
        x = 2.0 * math.pi * f * LISA_L / CLIGHT
        sinx = math.sin(x)
        t = 4.0 * x * x * sinx * sinx

        if tdi2:    
            sin2x = math.sin(2.0 * x)
            t *= 4.0 * sin2x * sin2x
        
        Sgal = _sgal_device(f, Amp, alpha, sl1, kn, sl2)
        return t * Sgal
    
    @cuda.jit(device=True, inline=True)
    def _wdconfusion_xy_device(f, Amp, alpha, sl1, kn, sl2, tdi2=False):
        """White dwarf confusion noise CSD."""
        x = 2.0 * math.pi * f * LISA_L / CLIGHT
        sinx = math.sin(x)
        t = - 0.5 * 4.0 * x * x * sinx * sinx

        if tdi2:    
            sin2x = math.sin(2.0 * x)
            t *= 4.0 * sin2x * sin2x
        
        Sgal = _sgal_device(f, Amp, alpha, sl1, kn, sl2)
        return t * Sgal

    @cuda.jit(device=True, inline=True)
    def _noisepsd_xx_device(f, tm_noise, isi_oms_noise, Amp, alpha, sl1, kn, sl2, tdi2=False):
        """Diagonal noise PSD for XYZ (TDI 1.5)."""
        x = 2.0 * math.pi * f * LISA_L / CLIGHT
        sinx = math.sin(x)
        cos2x = math.cos(2.0 * x)
        
        Cxx = 4.0 * sinx * sinx
        if tdi2:
            sin2x = math.sin(2.0 * x)
            Cxx *= 4.0 * sin2x * sin2x
        
        isi_rfi_readout_transfer = 4.0 * Cxx
        tm_transfer = 4.0 * Cxx * (3.0 + cos2x)
        
        total_noise = tm_transfer * tm_noise + isi_rfi_readout_transfer * isi_oms_noise
        sgal = _wdconfusion_xx_device(f, Amp, alpha, sl1, kn, sl2, tdi2)
        return total_noise + sgal

    @cuda.jit(device=True, inline=True)
    def _noisecsd_xy_device(f, tm_noise, isi_oms_noise, Amp, alpha, sl1, kn, sl2, tdi2=False):
        """Off-diagonal CSD for XYZ (TDI 1.5)."""
        x = 2.0 * math.pi * f * LISA_L / CLIGHT
        sinx = math.sin(x)
        sin2x = math.sin(2.0 * x)
        
        Cxy = -4.0 * sinx * sin2x
        if tdi2:    
            Cxy *= 4.0 * sin2x * sin2x
        
        isi_rfi_readout_transfer = Cxy
        tm_transfer = 4.0 * Cxy
        
        total_noise = tm_transfer * tm_noise + isi_rfi_readout_transfer * isi_oms_noise
        sgal = _wdconfusion_xy_device(f, Amp, alpha, sl1, kn, sl2, tdi2)
        return total_noise + sgal

    @cuda.jit(device=True, inline=True)
    def _inv_logdet_3x3_symmetric(diag, off):
        """Analytically invert 3x3 matrix with equal diag and equal off-diag using Cholesky.
        
        Matrix form:
            [diag, off, off]
            [off, diag, off]
            [off, off, diag]
        
        Returns: (c00, c01, c11, logdet) where c01=c02=c12 due to symmetry.
        """
        # Cholesky decomposition: C = L @ L^T
        l11 = math.sqrt(diag)
        l21 = off / l11
        l31 = off / l11
        l22 = math.sqrt(diag - l21 * l21)
        l32 = (off - l21 * l31) / l22
        l33 = math.sqrt(diag - l31 * l31 - l32 * l32)
        
        # Inverse of L (lower triangular)
        w11 = 1.0 / l11
        w21 = -l21 / (l11 * l22)
        w31 = (-l31 * l22 + l21 * l32) / (l11 * l22 * l33)
        w22 = 1.0 / l22
        w32 = -l32 / (l22 * l33)
        w33 = 1.0 / l33
        
        # C^{-1} = L^{-T} @ L^{-1} (only compute unique elements)
        c00 = w11 * w11 + w21 * w21 + w31 * w31
        c01 = w21 * w22 + w31 * w32
        c02 = w31 * w33
        c11 = w22 * w22 + w32 * w32
        c12 = w32 * w33
        c22 = w33 * w33
        
        # log(det(C)) = 2 * sum(log(diag(L)))
        logdet = 2.0 * (math.log(l11) + math.log(l22) + math.log(l33))
        
        return c00, c01, c02, c11, c12, c22, logdet

    @cuda.jit(device=True, inline=True)
    def _quadratic_form_3x3(d_x_re, d_x_im, d_y_re, d_y_im, d_z_re, d_z_im,
                            c00, c01, c02, c11, c12, c22):
        """Compute d^H @ C^{-1} @ d for complex d and real symmetric C^{-1}.
        
        C^{-1} has structure: [[c00,c01,c02],[c01,c11,c12],[c02,c12,c22]]
        """
        # C^{-1} @ d (real matrix times complex vector)
        inv_d_x_re = c00 * d_x_re + c01 * d_y_re + c02 * d_z_re
        inv_d_x_im = c00 * d_x_im + c01 * d_y_im + c02 * d_z_im
        inv_d_y_re = c01 * d_x_re + c11 * d_y_re + c12 * d_z_re
        inv_d_y_im = c01 * d_x_im + c11 * d_y_im + c12 * d_z_im
        inv_d_z_re = c02 * d_x_re + c12 * d_y_re + c22 * d_z_re
        inv_d_z_im = c02 * d_x_im + c12 * d_y_im + c22 * d_z_im
        
        # d^H @ (C^{-1} @ d) = conj(d) . (C^{-1} @ d)
        # Real part only (imaginary cancels for Hermitian form)
        quad = (d_x_re * inv_d_x_re + d_x_im * inv_d_x_im +
                d_y_re * inv_d_y_re + d_y_im * inv_d_y_im +
                d_z_re * inv_d_z_re + d_z_im * inv_d_z_im)
        
        return quad

    # =========================================================================
    # Main CUDA Kernel for XYZ PSD Likelihood
    # =========================================================================
    
    @cuda.jit
    def _psd_likelihood_xyz_kernel_fused(
        like_contrib,      # output: (num_psds, num_blocks)
        f_arr,             # input: (data_length,)
        data,              # input: complex, linearized (num_streams * 3 * data_length,)
        data_index_all,    # input: (num_psds,) int32
        Soms_d_in_all,     # input: (num_psds,)
        Sa_a_in_all,       # input: (num_psds,)
        Amp_all,           # input: (num_psds,)
        alpha_all,         # input: (num_psds,)
        sl1_all,           # input: (num_psds,)
        kn_all,            # input: (num_psds,)
        sl2_all,           # input: (num_psds,)
        df,                # scalar
        data_length,       # scalar
        num_psds,          # scalar
        num_blocks,         # scalar
        tdi2=False          # boolean
    ):
        """Fused kernel: compute noise, invert, quadratic form, reduce."""
        
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        psd_i = cuda.blockIdx.y
        
        if psd_i >= num_psds:
            return
        
        # Shared memory for block reduction
        like_vals = cuda.shared.array(shape=NUM_THREADS_LIKE, dtype=float64)
        logdet_vals = cuda.shared.array(shape=NUM_THREADS_LIKE, dtype=float64)
        
        # Initialize shared memory
        like_vals[tid] = 0.0
        logdet_vals[tid] = 0.0
        cuda.syncthreads()
        
        # Load PSD parameters (constant across frequencies for this PSD)
        data_index = data_index_all[psd_i]
        Soms_d_sq = Soms_d_in_all[psd_i] * Soms_d_in_all[psd_i]
        Sa_a_sq = Sa_a_in_all[psd_i] * Sa_a_in_all[psd_i]
        # Galactic foreground parameters (unused for now)
        Amp = Amp_all[psd_i]
        alpha = alpha_all[psd_i]
        sl1 = sl1_all[psd_i]
        kn = kn_all[psd_i]
        sl2 = sl2_all[psd_i]
        
        # Stride loop over frequency bins
        stride = cuda.blockDim.x * cuda.gridDim.x
        idx = bid * cuda.blockDim.x + tid
        
        while idx < data_length:
            f = f_arr[idx]
            if f == 0.0:
                f = df  # Avoid division by zero
            
            # Compute noise PSDs
            tm_noise, isi_oms_noise = _lisanoises_device(f, Soms_d_sq, Sa_a_sq)
            
            # Compute covariance elements
            diag = _noisepsd_xx_device(f, tm_noise, isi_oms_noise, Amp, alpha, sl1, kn, sl2, tdi2)
            off = _noisecsd_xy_device(f, tm_noise, isi_oms_noise, Amp, alpha, sl1, kn, sl2, tdi2)
            
            # Invert 3x3 covariance analytically
            c00, c01, c02, c11, c12, c22, logdet = _inv_logdet_3x3_symmetric(diag, off)
            
            # Load data (linearized: channel c of stream i at index (3*i + c)*data_length + freq_idx)
            base_x = (data_index * 3 + 0) * data_length + idx
            base_y = (data_index * 3 + 1) * data_length + idx
            base_z = (data_index * 3 + 2) * data_length + idx
            
            d_x = data[base_x]
            d_y = data[base_y]
            d_z = data[base_z]
            
            # Compute quadratic form d^H C^{-1} d
            quad = _quadratic_form_3x3(
                d_x.real, d_x.imag,
                d_y.real, d_y.imag,
                d_z.real, d_z.imag,
                c00, c01, c02, c11, c12, c22
            )
            
            # Accumulate
            like_vals[tid] += 4.0 * df * quad
            logdet_vals[tid] += logdet
            
            idx += stride
        
        cuda.syncthreads()
        
        # Parallel reduction in shared memory
        s = NUM_THREADS_LIKE // 2
        while s > 0:
            if tid < s:
                like_vals[tid] += like_vals[tid + s]
                logdet_vals[tid] += logdet_vals[tid + s]
            cuda.syncthreads()
            s //= 2
        
        # Write block result
        if tid == 0:
            like_contrib[psd_i * num_blocks + bid] = -0.5 * like_vals[0] - logdet_vals[0]

    @cuda.jit
    def _reduce_blocks_kernel(
        like_final,        # output: (num_psds,)
        like_contrib,      # input: (num_psds * num_blocks,)
        num_blocks,        # scalar
        num_psds           # scalar
    ):
        """Final reduction across blocks for each PSD."""
        
        tid = cuda.threadIdx.x
        psd_i = cuda.blockIdx.y
        
        if psd_i >= num_psds:
            return
        
        like_vals = cuda.shared.array(shape=NUM_THREADS_LIKE, dtype=float64)
        like_vals[tid] = 0.0
        cuda.syncthreads()
        
        # Each thread sums a portion of blocks
        i = tid
        while i < num_blocks:
            like_vals[tid] += like_contrib[psd_i * num_blocks + i]
            i += cuda.blockDim.x
        cuda.syncthreads()
        
        # Parallel reduction
        s = NUM_THREADS_LIKE // 2
        while s > 0:
            if tid < s:
                like_vals[tid] += like_vals[tid + s]
            cuda.syncthreads()
            s //= 2
        
        if tid == 0:
            like_final[psd_i] = like_vals[0]


# =============================================================================
# Python wrapper for Numba kernel
# =============================================================================

def psd_likelihood_xyz_numba_fused(
    f_arr,
    data,
    data_index_all,
    Soms_d_in_all,
    Sa_a_in_all,
    Amp_all,           
    alpha_all,        
    sl1_all,          
    kn_all,           
    sl2_all,          
    df,
    data_length,
    tdi2=False
):
    """Optimized PSD likelihood using fused Numba CUDA kernel.
    
    This version fuses noise computation, matrix inversion, quadratic form,
    and reduction into a single kernel launch, minimizing memory traffic.
    
    Args:
        f_arr: frequency array, CuPy or NumPy, shape (data_length,)
        data: complex data, CuPy array, linearized (num_streams * 3 * data_length,)
        data_index_all: stream indices, shape (num_psds,)
        Soms_d_in_all, Sa_a_in_all: noise parameters, shape (num_psds,)
        Amp_all, alpha_all, sl1_all, kn_all, sl2_all: galactic foreground (unused)
        df: frequency resolution
        data_length: number of frequency bins, int
        tdi2: whether to use TDI 2.0 noise model
        
    Returns:
        CuPy array of log-likelihoods, shape (num_psds,)
    """
    if not HAS_NUMBA:
        raise RuntimeError("Numba CUDA is required for this function")
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for this function")
    
    # Ensure arrays are on GPU with correct dtype
    f_arr_d = cp.asarray(f_arr, dtype=cp.float64)
    data_d = cp.asarray(data, dtype=cp.complex128)
    data_index_d = cp.asarray(data_index_all, dtype=cp.int32)
    Soms_d_d = cp.asarray(Soms_d_in_all, dtype=cp.float64)
    Sa_a_d = cp.asarray(Sa_a_in_all, dtype=cp.float64)
    
    num_psds = len(data_index_all)
    num_blocks = int(math.ceil(data_length / NUM_THREADS_LIKE))
    
    # Allocate output
    like_contrib_d = cp.zeros((num_psds, num_blocks), dtype=cp.float64)
    like_final_d = cp.zeros(num_psds, dtype=cp.float64)
    
    # Launch main kernel
    grid = (num_blocks, num_psds)
    _psd_likelihood_xyz_kernel_fused[grid, NUM_THREADS_LIKE](
        cuda.as_cuda_array(like_contrib_d.ravel()),
        cuda.as_cuda_array(f_arr_d),
        cuda.as_cuda_array(data_d),
        cuda.as_cuda_array(data_index_d),
        cuda.as_cuda_array(Soms_d_d),
        cuda.as_cuda_array(Sa_a_d),
        cuda.as_cuda_array(Amp_all),
        cuda.as_cuda_array(alpha_all),
        cuda.as_cuda_array(sl1_all),
        cuda.as_cuda_array(kn_all),
        cuda.as_cuda_array(sl2_all),
        float(df),
        int(data_length),
        int(num_psds),
        int(num_blocks),
        tdi2
    )
    
    # Final reduction
    grid_reduce = (1, num_psds)
    _reduce_blocks_kernel[grid_reduce, NUM_THREADS_LIKE](
        cuda.as_cuda_array(like_final_d),
        cuda.as_cuda_array(like_contrib_d.ravel()),
        int(num_blocks),
        int(num_psds)
    )
    
    return like_final_d


def _ensure_cupy_array(arr, dtype=None):
    """Convert input to CuPy array if CuPy is available, else raise."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required but not installed")
    if dtype is None:
        return cp.asarray(arr)
    return cp.asarray(arr, dtype=dtype)


def lisanoises(f, Soms_d_in, Sa_a_in, return_relative_frequency=True):
    """LISA noise model (vectorized).
    
    Args:
        f: frequency array, shape (N,) or broadcastable
        Soms_d_in, Sa_a_in: scalar or array parameters
        return_relative_frequency: if True, return relative frequency noise
        
    Returns:
        Sa, Soms: noise PSDs, same shape as f
    """
    xp = cp if HAS_CUPY else np
    
    Sa_a = Sa_a_in * (1.0 + (0.4e-3 / f)**2) * (1.0 + (f / 8e-3)**4)
    Sa_d = Sa_a * (2.0 * xp.pi * f)**(-4.0)
    Sa_nu = Sa_d * (2.0 * xp.pi * f / CLIGHT)**2
    
    Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f)**4)
    Soms_nu = Soms_d * (2.0 * xp.pi * f / CLIGHT)**2
    
    if return_relative_frequency:
        return Sa_nu, Soms_nu
    return Sa_d, Soms_d


def extended_lisanoises(f, Soms_d_in, Sa_a_in, return_relative_frequency=True):
    """Extended LISA noise model with additional noise sources (all zeros for now).
    
    Returns:
        tm, isi_oms, rfi_oms, tmi_oms, rfi_backlink, tmi_backlink
    """
    Sa, Soms = lisanoises(f, Soms_d_in, Sa_a_in, return_relative_frequency)
    
    xp = cp if HAS_CUPY else np
    zeros = xp.zeros_like(f)
    
    return Sa, Soms, zeros, zeros, zeros, zeros


def sgal(fr, amp, alpha, sl1, kn, sl2):
    """Galactic confusion noise model (vectorized)."""
    xp = cp if HAS_CUPY else np
    return amp * xp.exp(-(fr**alpha) * sl1) * fr**(-7.0/3.0) * 0.5 * (1.0 + xp.tanh(-(fr - kn) * sl2))


def noisepsd_xx(f, tm_noise, isi_oms_noise, rfi_oms_noise, tmi_oms_noise, 
                rfi_backlink_noise, tmi_backlink_noise, tdi2=False):
    """Instrument noise PSD for XYZ (TDI 2.0), equal arms, uncorrelated sources."""
    xp = cp if HAS_CUPY else np
    
    x = 2.0 * xp.pi * f * LISA_L / CLIGHT
    sinx = xp.sin(x)
    sin2x = xp.sin(2.0 * x)
    cos2x = xp.cos(2.0 * x)
    
    Cxx = 4.0 * sinx**2 #* sin2x**2
    if tdi2:
        Cxx *= 4.0 * sin2x**2
    
    isi_rfi_readout_transfer = 4.0 * Cxx
    tmi_readout_transfer = Cxx * (3.0 + cos2x)
    tm_transfer = 4.0 * Cxx * (3.0 + cos2x)
    rfi_backlink_transfer = 4.0 * Cxx
    tmi_backlink_transfer = Cxx * (3.0 + cos2x)
    
    total_noise = (
        tm_transfer * tm_noise +
        isi_rfi_readout_transfer * (isi_oms_noise + rfi_oms_noise) +
        tmi_readout_transfer * tmi_oms_noise +
        rfi_backlink_transfer * rfi_backlink_noise +
        tmi_backlink_transfer * tmi_backlink_noise
    )
    
    return total_noise


def noisecsd_xy(f, tm_noise, isi_oms_noise, rfi_oms_noise, tmi_oms_noise,
                rfi_backlink_noise, tmi_backlink_noise, tdi2=False):
    """Cross-spectral density between X and Y channels."""
    xp = cp if HAS_CUPY else np
    
    x = 2.0 * xp.pi * f * LISA_L / CLIGHT
    sinx = xp.sin(x)
    sin2x = xp.sin(2.0 * x)
    
    Cxy = -4.0 * sinx * sin2x
    if tdi2:
        Cxy *= 4.0 * sin2x**2
    
    isi_rfi_readout_transfer = Cxy
    tmi_readout_transfer = Cxy
    tm_transfer = 4.0 * Cxy
    rfi_backlink_transfer = Cxy
    tmi_backlink_transfer = Cxy
    
    total_noise = (
        tm_transfer * tm_noise +
        isi_rfi_readout_transfer * (isi_oms_noise + rfi_oms_noise) +
        tmi_readout_transfer * tmi_oms_noise +
        rfi_backlink_transfer * rfi_backlink_noise +
        tmi_backlink_transfer * tmi_backlink_noise
    )
    
    return total_noise


def build_xyz_covariance(f_arr, Soms_d_all, Sa_a_all, Amp_all, alpha_all, 
                         sl1_all, kn_all, sl2_all):
    """Build 3×3 noise covariance matrices for XYZ channels.
    
    Args:
        f_arr: frequency array, shape (Nfreq,)
        *_all: parameter arrays, shape (num_psds,)
        
    Returns:
        cov: covariance matrices, shape (num_psds, Nfreq, 3, 3)
    """
    xp = cp if HAS_CUPY else np
    
    f_arr = _ensure_cupy_array(f_arr, dtype=xp.float64)
    Soms_d_all = _ensure_cupy_array(Soms_d_all, dtype=xp.float64)
    Sa_a_all = _ensure_cupy_array(Sa_a_all, dtype=xp.float64)
    Amp_all = _ensure_cupy_array(Amp_all, dtype=xp.float64)
    alpha_all = _ensure_cupy_array(alpha_all, dtype=xp.float64)
    sl1_all = _ensure_cupy_array(sl1_all, dtype=xp.float64)
    kn_all = _ensure_cupy_array(kn_all, dtype=xp.float64)
    sl2_all = _ensure_cupy_array(sl2_all, dtype=xp.float64)
    
    num_psds = len(Soms_d_all)
    Nfreq = len(f_arr)
    
    # Broadcast: (num_psds, 1) and (Nfreq,) -> (num_psds, Nfreq)
    f = f_arr[None, :]  # (1, Nfreq)
    Soms_d = Soms_d_all[:, None]  # (num_psds, 1)
    Sa_a = Sa_a_all[:, None]
    
    # Compute extended noise (broadcasts to num_psds × Nfreq)
    tm, isi, rfi, tmi, rfi_bl, tmi_bl = extended_lisanoises(
        f, Soms_d**2, Sa_a**2, return_relative_frequency=True
    )
    
    # Diagonal elements (all equal for XYZ)
    diag = noisepsd_xx(f, tm, isi, rfi, tmi, rfi_bl, tmi_bl)  # (num_psds, Nfreq)
    
    # Off-diagonal elements (all equal)
    off = noisecsd_xy(f, tm, isi, rfi, tmi, rfi_bl, tmi_bl)  # (num_psds, Nfreq)
    
    # Build covariance matrix: shape (num_psds, Nfreq, 3, 3)
    cov = xp.zeros((num_psds, Nfreq, 3, 3), dtype=xp.float64)
    
    # Set diagonal
    cov[:, :, 0, 0] = diag
    cov[:, :, 1, 1] = diag
    cov[:, :, 2, 2] = diag
    
    # Set off-diagonals (symmetric matrix)
    cov[:, :, 0, 1] = off
    cov[:, :, 0, 2] = off
    cov[:, :, 1, 0] = off
    cov[:, :, 1, 2] = off
    cov[:, :, 2, 0] = off
    cov[:, :, 2, 1] = off
    
    return cov


def invert_covariance_batch(cov):
    """Invert batched 3×3 covariance matrices using Cholesky decomposition.
    
    Args:
        cov: covariance array, shape (..., 3, 3), symmetric positive definite
        
    Returns:
        inv_cov: inverse matrices, shape (..., 3, 3)
        logdet: log determinants, shape (...)
    """
    xp = cp if HAS_CUPY else np
    
    original_shape = cov.shape
    flat_cov = cov.reshape(-1, 3, 3)
    
    # Cholesky: C = L L^T
    L = xp.linalg.cholesky(flat_cov)
    
    # Log determinant: log(det(C)) = 2 * sum(log(diag(L)))
    logdet = 2.0 * xp.log(xp.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
    
    # Inverse via triangular solves: C^{-1} = L^{-T} L^{-1}
    # More stable than explicit inverse
    I = xp.eye(3, dtype=xp.float64)
    # Solve L Y = I for Y, then solve L^T X = Y for X = C^{-1}
    Iall = xp.tile(I[None, :, :], (flat_cov.shape[0], 1, 1))  # (N, 3, 3)
    Y = xp.linalg.solve(L, Iall)  # broadcasts (N, 3, 3) and (1, 3, 3)
    inv_cov = xp.linalg.solve(xp.swapaxes(L, -2, -1), Y)
    
    # Reshape back
    inv_cov = inv_cov.reshape(original_shape)
    logdet = logdet.reshape(original_shape[:-2])
    
    return inv_cov, logdet

def psd_likelihood_xyz_cupy(
    f_arr,
    data,
    data_index_all,
    Soms_d_in_all,
    Sa_a_in_all,
    Amp_all,
    alpha_all,
    sl1_all,
    kn_all,
    sl2_all,
    df,
    data_length
):
    """Compute PSD log-likelihood for XYZ channels using CuPy.
    
    Likelihood: -0.5 * sum_f [ 4*df * d^H C^{-1} d + log(det(C)) ]
    
    Args:
        f_arr: frequency array, shape (Nfreq,)
        data: complex data, shape (num_streams * 3, Nfreq) where channels are [X, Y, Z] triples
        data_index_all: stream indices, shape (num_psds,)
        *_in_all: noise parameters, shape (num_psds,)
        df: frequency bin width
        data_length: length of data segments 
        
    Returns:
        log_likes: NumPy array, shape (num_psds,)
    """
    xp = cp if HAS_CUPY else np
    
    # Move to GPU
    f_arr = _ensure_cupy_array(f_arr, dtype=xp.float64)
    data = _ensure_cupy_array(data, dtype=xp.complex128)
    data_index_all = _ensure_cupy_array(data_index_all, dtype=xp.int32)
    
    # Build covariance matrices
    cov = build_xyz_covariance(
        f_arr, Soms_d_in_all, Sa_a_in_all, 
        Amp_all, alpha_all, sl1_all, kn_all, sl2_all
    )  # (num_psds, Nfreq, 3, 3)
    
    # breakpoint()

    # Invert covariance and get log determinants
    inv_cov, logdet = invert_covariance_batch(cov)  # (num_psds, Nfreq, 3, 3), (num_psds, Nfreq)
    
    num_psds = len(data_index_all)
    
    # Extract data for each PSD
    # data shape: (num_streams * 3, Nfreq)
    # For stream i: X at 3*i, Y at 3*i+1, Z at 3*i+2
    data_indices = data_index_all
    # X_indices = 3 * data_indices
    # Y_indices = 3 * data_indices + 1
    # Z_indices = 3 * data_indices + 2

    # [(data_index * 2 + 0) * data_length + i];
    #breakpoint()

    # data is linearized, need to reshape while picking correct indices

    X_indices = ((3 * data_indices * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)
    Y_indices = (((3 * data_indices + 1) * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)
    Z_indices = (((3 * data_indices + 2) * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)


    d_X = data[X_indices].reshape(-1, data_length)  # (num_psds, Nfreq)
    d_Y = data[Y_indices].reshape(-1, data_length)  # (num_psds, Nfreq)
    d_Z = data[Z_indices].reshape(-1, data_length)  # (num_psds, Nfreq) 

    xp.get_default_memory_pool().free_all_blocks()
    
    # Stack into data vector: shape (num_psds, Nfreq, 3)
    d_vec = xp.stack([d_X, d_Y, d_Z], axis=-1)  # (num_psds, Nfreq, 3)
    
    # Compute quadratic form: d^H C^{-1} d using einsum
    # d_vec: (num_psds, Nfreq, 3)
    # inv_cov: (num_psds, Nfreq, 3, 3)
    # Result: (num_psds, Nfreq)
    
    # First: C^{-1} @ d  (matrix-vector multiply per batch)
    # inv_cov @ d_vec: shape (num_psds, Nfreq, 3)
    inv_cov_d = xp.einsum('...ij,...j->...i', inv_cov, d_vec)  # (num_psds, Nfreq, 3)
    
    # Second: d^H @ (C^{-1} @ d)  (complex conjugate dot product)
    # For real covariance, this is just real part of d* @ inv_cov @ d
    quad = xp.einsum('...i,...i->...', d_vec.conj(), inv_cov_d).real  # (num_psds, Nfreq)
    
    # Sum over frequencies
    inner_sum = (4.0 * df * quad).sum(axis=1)  # (num_psds,)
    logdet_sum = logdet.sum(axis=1)  # (num_psds,)
    
    # Total likelihood
    log_like = -0.5 * inner_sum - logdet_sum
    xp.get_default_memory_pool().free_all_blocks()
    # Return as NumPy (transfer to host)
    return log_like

# def invert_3x3_analytical_batch(cov):
#     """Analytically invert batched 3×3 symmetric matrices.
    
#     Args:
#         cov: shape (..., 3, 3)
#     Returns:
#         inv_cov: shape (..., 3, 3)
#         logdet: shape (...)
#     """
#     xp = cp
    
#     # Extract elements (much faster than indexing later)
#     c00, c01, c02 = cov[..., 0, 0], cov[..., 0, 1], cov[..., 0, 2]
#     c11, c12 = cov[..., 1, 1], cov[..., 1, 2]
#     c22 = cov[..., 2, 2]
    
#     # Cofactors for first row
#     cof00 = c11 * c22 - c12 * c12
#     cof01 = c12 * c02 - c01 * c22
#     cof02 = c01 * c12 - c11 * c02
    
#     # Determinant
#     det = c00 * cof00 + c01 * cof01 + c02 * cof02
    
#     # Log determinant
#     logdet = xp.log(xp.abs(det))
    
#     # Inverse
#     inv_det = 1.0 / det
    
#     inv_cov = xp.empty_like(cov)
#     inv_cov[..., 0, 0] = cof00 * inv_det
#     inv_cov[..., 0, 1] = cof01 * inv_det
#     inv_cov[..., 0, 2] = cof02 * inv_det
    
#     inv_cov[..., 1, 0] = cof01 * inv_det
#     inv_cov[..., 1, 1] = (c00 * c22 - c02 * c02) * inv_det
#     inv_cov[..., 1, 2] = (c02 * c01 - c00 * c12) * inv_det
    
#     inv_cov[..., 2, 0] = cof02 * inv_det
#     inv_cov[..., 2, 1] = inv_cov[..., 1, 2]  # Symmetric
#     inv_cov[..., 2, 2] = (c00 * c11 - c01 * c01) * inv_det
    
#     return inv_cov, logdet

def psd_likelihood_xyz_cupy_optimized(
    f_arr,
    data,
    data_index_all,
    Soms_d_in_all,
    Sa_a_in_all,
    Amp_all,
    alpha_all,
    sl1_all,
    kn_all,
    sl2_all,
    df,
    data_length
):
    """Compute PSD log-likelihood for XYZ channels using CuPy.
    
    Likelihood: -0.5 * sum_f [ 4*df * d^H C^{-1} d + log(det(C)) ]
    
    Args:
        f_arr: frequency array, shape (Nfreq,)
        data: complex data, shape (num_streams * 3, Nfreq) where channels are [X, Y, Z] triples
        data_index_all: stream indices, shape (num_psds,)
        *_in_all: noise parameters, shape (num_psds,)
        df: frequency bin width
        data_length: length of data segments 
        
    Returns:
        log_likes: CuPy array, shape (num_psds,)
    """
    xp = cp if HAS_CUPY else np
    
    # Move to GPU
    f_arr = _ensure_cupy_array(f_arr, dtype=xp.float64)
    data = _ensure_cupy_array(data, dtype=xp.complex128)
    data_index_all = _ensure_cupy_array(data_index_all, dtype=xp.int32)
    
    # Build covariance matrices
    cov = build_xyz_covariance(
        f_arr, Soms_d_in_all, Sa_a_in_all, 
        Amp_all, alpha_all, sl1_all, kn_all, sl2_all
    )  # (num_psds, Nfreq, 3, 3)
    
    # breakpoint()

    # Invert covariance and get log determinants
    #inv_cov, logdet = invert_3x3_analytical_batch(cov)  # (num_psds, Nfreq, 3, 3), (num_psds, Nfreq)
    logdet = xp.linalg.slogdet(cov)[1]

    
    num_psds = len(data_index_all)
    
    # Extract data for each PSD
    # data shape: (num_streams * 3, Nfreq)
    # For stream i: X at 3*i, Y at 3*i+1, Z at 3*i+2
    data_indices = data_index_all
    # X_indices = 3 * data_indices
    # Y_indices = 3 * data_indices + 1
    # Z_indices = 3 * data_indices + 2

    # [(data_index * 2 + 0) * data_length + i];
    #breakpoint()

    # data is linearized, need to reshape while picking correct indices

    X_indices = ((3 * data_indices * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)
    Y_indices = (((3 * data_indices + 1) * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)
    Z_indices = (((3 * data_indices + 2) * data_length)[:, None] + xp.arange(data_length)[None, :]).reshape(-1)


    d_X = data[X_indices].reshape(-1, data_length)  # (num_psds, Nfreq)
    d_Y = data[Y_indices].reshape(-1, data_length)  # (num_psds, Nfreq)
    d_Z = data[Z_indices].reshape(-1, data_length)  # (num_psds, Nfreq) 

    # xp.get_default_memory_pool().free_all_blocks()
    
    # Stack into data vector: shape (num_psds, Nfreq, 3)
    d_vec = xp.stack([d_X, d_Y, d_Z], axis=-1)  # (num_psds, Nfreq, 3)
    
    # Compute quadratic form: d^H C^{-1} d using einsum
    # d_vec: (num_psds, Nfreq, 3)
    # inv_cov: (num_psds, Nfreq, 3, 3)
    # Result: (num_psds, Nfreq)
    
    # First: C^{-1} @ d  (matrix-vector multiply per batch)
    # inv_cov @ d_vec: shape (num_psds, Nfreq, 3)
    # inv_cov_d = xp.einsum('...ij,...j->...i', inv_cov, d_vec)  # (num_psds, Nfreq, 3)
    inv_cov_d = xp.linalg.solve(cov, d_vec[..., None]).squeeze(-1)
    
    # Second: d^H @ (C^{-1} @ d)  (complex conjugate dot product)
    # For real covariance, this is just real part of d* @ inv_cov @ d
    quad = xp.einsum('...i,...i->...', d_vec.conj(), inv_cov_d).real  # (num_psds, Nfreq)
    
    # Sum over frequencies
    inner_sum = (4.0 * df * quad).sum(axis=1)  # (num_psds,)
    logdet_sum = logdet.sum(axis=1)  # (num_psds,)
    
    # Total likelihood
    log_like = -0.5 * inner_sum - logdet_sum
    # xp.get_default_memory_pool().free_all_blocks()
    
    return log_like


# =============================================================================
# Optimized entry point - uses Numba kernel when available
# =============================================================================

def psd_likelihood_numba(
    f_arr,
    data,
    data_index_all,
    Soms_d_in_all,
    Sa_a_in_all,
    Amp_all,
    alpha_all,
    sl1_all,
    kn_all,
    sl2_all,
    df,
    data_length,
    tdi2=False
):
    """Optimized PSD log-likelihood for XYZ channels.
    
    Uses fused Numba CUDA kernel when available (fastest), otherwise falls back
    to CuPy vectorized implementation.
    
    The Numba kernel fuses:
    1. Noise PSD computation
    2. 3×3 covariance matrix construction
    3. Analytical Cholesky-based inversion
    4. Quadratic form computation
    5. Parallel reduction
    
    into a single kernel, minimizing GPU memory traffic.
    
    Args:
        f_arr: frequency array, shape (data_length,)
        data: complex data, linearized (num_streams * 3 * data_length,)
        data_index_all: stream indices, shape (num_psds,)
        Soms_d_in_all, Sa_a_in_all: noise parameters, shape (num_psds,)
        Amp_all, alpha_all, sl1_all, kn_all, sl2_all: galactic foreground params
        df: frequency resolution
        data_length: number of frequency bins
        tdi2: whether to use TDI 2.0 noise model
        
    Returns:
        CuPy array of log-likelihoods, shape (num_psds,)
    """
    if HAS_NUMBA:
        # Use fast fused Numba kernel
        return psd_likelihood_xyz_numba_fused(
            f_arr, data, data_index_all,
            Soms_d_in_all, Sa_a_in_all,
            Amp_all, alpha_all, sl1_all, kn_all, sl2_all,
            df, data_length, tdi2
        )
    else:
        # Fallback to CuPy implementation
        return psd_likelihood_xyz_cupy(
            f_arr, data, data_index_all,
            Soms_d_in_all, Sa_a_in_all,
            Amp_all, alpha_all, sl1_all, kn_all, sl2_all,
            df, data_length
        )