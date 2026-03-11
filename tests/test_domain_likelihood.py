"""Tests comparing C++ domain likelihood kernels against Python inner_product().

Each test constructs raw data/invC/template arrays, computes (d|h) and (h|h)
via the C++ DomainComputationGroup, and checks against the equivalent
NumPy computation that mirrors diagnostic.inner_product's logic:

    <a|b> = 4 * df * sum_over_channels_and_freqs( conj(a) * invC * b )
"""

import numpy as np
import pytest

from lisatools.domaincomputation import FDComputationGroup, STFTComputationGroup
from lisatools.domains import FDSettings, STFTSettings


def _python_inner_product_diag(sig1, sig2, invC, df):
    """Python reference: diagonal (AET) inner product.

    sig1, sig2: (num_channels, num_freqs) complex
    invC: (num_channels, num_freqs) complex (real-valued in practice)

    Returns complex scalar <sig1|sig2> = 4 * df * sum(conj(sig1) * invC * sig2).
    """
    return 4.0 * df * np.sum(np.conj(sig1) * invC * sig2)


def _python_inner_product_cross(sig1, sig2, invC, df):
    """Python reference: full-matrix (XYZ) inner product.

    sig1, sig2: (num_channels, num_freqs) complex
    invC: (num_channels, num_channels, num_freqs) complex

    Returns complex scalar <sig1|sig2> = 4 * df * sum_{i,j,f} conj(sig1[i,f]) * invC[i,j,f] * sig2[j,f].
    """
    num_channels = sig1.shape[0]
    result = 0.0 + 0.0j
    for i in range(num_channels):
        for j in range(num_channels):
            result += np.sum(np.conj(sig1[i]) * invC[i, j] * sig2[j])
    return 4.0 * df * result


def _python_inner_product_diag_stft(sig1, sig2, invC, df):
    """Python reference: diagonal STFT inner product.

    sig1, sig2: (num_channels, num_times, num_freqs) complex
    invC: (num_channels, num_times, num_freqs) complex

    Returns complex scalar.
    """
    return 4.0 * df * np.sum(np.conj(sig1) * invC * sig2)


# =====================================================================
# Test 1: FD + AET diagonal — single binary
# =====================================================================


class TestFDAETDiagonal:
    def test_single_binary(self):
        rng = np.random.default_rng(42)
        num_channels = 2
        num_freqs = 1024
        df = 1.0 / 3600.0
        f_min = df  # start at df to avoid DC
        f_max = f_min + (num_freqs - 1) * df

        # data: (1, num_channels, num_freqs) — 1 data instance
        data = (rng.standard_normal((1, num_channels, num_freqs))
                + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        # invC: (1, num_channels, num_freqs) — 1 noise instance, diagonal (AET)
        invC = np.abs(rng.standard_normal((1, num_channels, num_freqs))) + 0.1
        invC = invC.astype(np.complex128)

        # template: (1, num_channels, num_freqs) — 1 binary
        template = (rng.standard_normal((1, num_channels, num_freqs))
                    + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        settings = FDSettings(N=num_freqs, df=df, min_freq=f_min, max_freq=f_max, force_backend="cpu")

        dcg = FDComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=1,
            num_noise=1,
            num_channels=num_channels,
            settings=settings,
            tdi_type="AET",
            force_backend="cpu",
        )

        start_freqs = np.array([f_min])
        data_index = np.array([0], dtype=np.int32)
        noise_index = np.array([0], dtype=np.int32)

        d_h_cpp, h_h_cpp = dcg.compute_likelihood_terms(
            template, start_freqs, data_index, noise_index
        )

        # Python reference
        d_h_py = _python_inner_product_diag(data[0], template[0], invC[0], df)
        h_h_py = _python_inner_product_diag(template[0], template[0], invC[0], df)

        np.testing.assert_allclose(d_h_cpp[0], d_h_py, rtol=1e-10)
        np.testing.assert_allclose(h_h_cpp[0], h_h_py, rtol=1e-10)


# =====================================================================
# Test 2: FD + XYZ full covariance — single binary
# =====================================================================


class TestFDXYZCross:
    def test_single_binary(self):
        rng = np.random.default_rng(123)
        num_channels = 3
        num_freqs = 512
        df = 1.0 / 7200.0
        f_min = df
        f_max = f_min + (num_freqs - 1) * df

        # data: (1, 3, num_freqs)
        data = (rng.standard_normal((1, num_channels, num_freqs))
                + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        # invC: (1, 3, 3, num_freqs) — full cross-channel
        # Build Hermitian positive-definite per freq bin
        invC = np.zeros((1, num_channels, num_channels, num_freqs), dtype=np.complex128)
        for f_idx in range(num_freqs):
            A = rng.standard_normal((num_channels, num_channels)) + 1j * rng.standard_normal((num_channels, num_channels))
            invC[0, :, :, f_idx] = A @ A.conj().T + 3.0 * np.eye(num_channels)

        # template: (1, 3, num_freqs)
        template = (rng.standard_normal((1, num_channels, num_freqs))
                    + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        settings = FDSettings(N=num_freqs, df=df, min_freq=f_min, max_freq=f_max, force_backend="cpu")

        dcg = FDComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=1,
            num_noise=1,
            num_channels=num_channels,
            settings=settings,
            tdi_type="XYZ",
            force_backend="cpu",
        )

        start_freqs = np.array([f_min])
        data_index = np.array([0], dtype=np.int32)
        noise_index = np.array([0], dtype=np.int32)

        d_h_cpp, h_h_cpp = dcg.compute_likelihood_terms(
            template, start_freqs, data_index, noise_index
        )

        d_h_py = _python_inner_product_cross(data[0], template[0], invC[0], df)
        h_h_py = _python_inner_product_cross(template[0], template[0], invC[0], df)

        np.testing.assert_allclose(d_h_cpp[0], d_h_py, rtol=1e-10)
        np.testing.assert_allclose(h_h_cpp[0], h_h_py, rtol=1e-10)


# =====================================================================
# Test 3: STFT + AET — single binary with sub-grid offset
# =====================================================================


class TestSTFTAET:
    def test_single_binary(self):
        rng = np.random.default_rng(77)
        num_channels = 2
        num_times = 8
        num_freqs = 256
        dt = 3600.0
        df = 1.0 / 3600.0
        t0 = 0.0
        f_min = df
        f_max = f_min + (num_freqs - 1) * df

        # data: (1, num_channels, num_times, num_freqs)
        data = (rng.standard_normal((1, num_channels, num_times, num_freqs))
                + 1j * rng.standard_normal((1, num_channels, num_times, num_freqs)))

        # invC: (1, num_channels, num_times, num_freqs)
        invC = np.abs(rng.standard_normal((1, num_channels, num_times, num_freqs))) + 0.1
        invC = invC.astype(np.complex128)

        # Template occupies a sub-grid: starts at t_idx=2, f_idx=10
        n_t_template = 4
        n_f_template = 100
        start_t_idx = 2
        start_f_idx = 10
        start_time = t0 + start_t_idx * dt
        start_freq = f_min + start_f_idx * df

        # template: (1, num_channels, n_t_template, n_f_template)
        template = (rng.standard_normal((1, num_channels, n_t_template, n_f_template))
                    + 1j * rng.standard_normal((1, num_channels, n_t_template, n_f_template)))

        settings = STFTSettings(
            t0=t0, dt=dt, df=df,
            NT=num_times, NF=num_freqs,
            min_freq=f_min, max_freq=f_max,
            force_backend="cpu",
        )

        dcg = STFTComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=1,
            num_noise=1,
            num_channels=num_channels,
            settings=settings,
            tdi_type="AET",
            force_backend="cpu",
        )

        start_freqs = np.array([start_freq])
        start_times = np.array([start_time])
        data_index = np.array([0], dtype=np.int32)
        noise_index = np.array([0], dtype=np.int32)

        d_h_cpp, h_h_cpp = dcg.compute_likelihood_terms(
            template, start_times, start_freqs, data_index, noise_index,
        )

        # Python reference: extract the sub-grid from data and invC
        data_sub = data[0, :, start_t_idx:start_t_idx + n_t_template,
                        start_f_idx:start_f_idx + n_f_template]
        invC_sub = invC[0, :, start_t_idx:start_t_idx + n_t_template,
                        start_f_idx:start_f_idx + n_f_template]
        d_h_py = _python_inner_product_diag_stft(data_sub, template[0], invC_sub, df)
        h_h_py = _python_inner_product_diag_stft(template[0], template[0], invC_sub, df)

        np.testing.assert_allclose(d_h_cpp[0], d_h_py, rtol=1e-10)
        np.testing.assert_allclose(h_h_cpp[0], h_h_py, rtol=1e-10)


# =====================================================================
# Test 4: Multi-binary batching — FD + AET
# =====================================================================


class TestMultiBinaryBatch:
    def test_five_binaries(self):
        rng = np.random.default_rng(999)
        num_channels = 2
        num_freqs = 256
        df = 1.0 / 3600.0
        f_min = df
        f_max = f_min + (num_freqs - 1) * df
        num_data = 3
        num_noise = 2
        num_binaries = 5

        # data: (num_data, num_channels, num_freqs)
        data = (rng.standard_normal((num_data, num_channels, num_freqs))
                + 1j * rng.standard_normal((num_data, num_channels, num_freqs)))

        # invC: (num_noise, num_channels, num_freqs) — diagonal AET
        invC = np.abs(rng.standard_normal((num_noise, num_channels, num_freqs))) + 0.1
        invC = invC.astype(np.complex128)

        # Each binary uses a different data/noise index and sub-frequency range
        n_f_templates = [50, 60, 70, 80, 90]
        start_f_indices = [0, 10, 20, 5, 15]
        data_indices = [0, 1, 2, 0, 1]
        noise_indices = [0, 1, 0, 1, 0]

        # Use the max template size and pad shorter ones with zeros
        max_nf = max(n_f_templates)
        templates = np.zeros((num_binaries, num_channels, max_nf), dtype=np.complex128)
        for b in range(num_binaries):
            nf = n_f_templates[b]
            templates[b, :, :nf] = (
                rng.standard_normal((num_channels, nf))
                + 1j * rng.standard_normal((num_channels, nf))
            )

        settings = FDSettings(N=num_freqs, df=df, min_freq=f_min, max_freq=f_max, force_backend="cpu")

        dcg = FDComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=num_data,
            num_noise=num_noise,
            num_channels=num_channels,
            settings=settings,
            tdi_type="AET",
            force_backend="cpu",
        )

        start_freqs = np.array([f_min + si * df for si in start_f_indices])
        data_index = np.array(data_indices, dtype=np.int32)
        noise_index = np.array(noise_indices, dtype=np.int32)

        d_h_cpp, h_h_cpp = dcg.compute_likelihood_terms(
            templates, start_freqs, data_index, noise_index
        )
        print("C++ d_h:", d_h_cpp)
        print("C++ h_h:", h_h_cpp)

        # Python reference: one binary at a time
        for b in range(num_binaries):
            nf = n_f_templates[b]
            sf = start_f_indices[b]
            di = data_indices[b]
            ni = noise_indices[b]

            d_sub = data[di, :, sf:sf + max_nf]
            c_sub = invC[ni, :, sf:sf + max_nf]
            h_sub = templates[b]

            d_h_py = _python_inner_product_diag(d_sub, h_sub, c_sub, df)
            h_h_py = _python_inner_product_diag(h_sub, h_sub, c_sub, df)

            print(f"Binary {b}: Python d_h = {d_h_py}, h_h = {h_h_py}")

            np.testing.assert_allclose(d_h_cpp[b], d_h_py, rtol=1e-10,
                                       err_msg=f"d_h mismatch for binary {b}")
            np.testing.assert_allclose(h_h_cpp[b], h_h_py, rtol=1e-10,
                                       err_msg=f"h_h mismatch for binary {b}")


# =====================================================================
# Test 5: Full likelihood comparison — FD + AET
# =====================================================================


class TestFullLikelihood:
    def test_likelihood_value(self):
        """Single container, single binary — d_d indexing is trivial."""
        rng = np.random.default_rng(2024)
        num_channels = 2
        num_freqs = 512
        df = 1.0 / 3600.0
        f_min = df
        f_max = f_min + (num_freqs - 1) * df

        data = (rng.standard_normal((1, num_channels, num_freqs))
                + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        invC = np.abs(rng.standard_normal((1, num_channels, num_freqs))) + 0.1
        invC = invC.astype(np.complex128)

        template = (rng.standard_normal((1, num_channels, num_freqs))
                    + 1j * rng.standard_normal((1, num_channels, num_freqs)))

        settings = FDSettings(N=num_freqs, df=df, min_freq=f_min, max_freq=f_max, force_backend="cpu")

        dcg = FDComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=1,
            num_noise=1,
            num_channels=num_channels,
            settings=settings,
            tdi_type="AET",
            force_backend="cpu",
        )

        # Manually set d_d (shape (num_data,) = (1,))
        d_d_py = _python_inner_product_diag(data[0], data[0], invC[0], df)
        dcg.d_d = np.array([d_d_py.real])

        start_freqs = np.array([f_min])
        data_index = np.array([0], dtype=np.int32)
        noise_index = np.array([0], dtype=np.int32)

        like_cpp = dcg.compute_likelihood(
            template, start_freqs, data_index, noise_index
        )

        # Python reference likelihood
        d_h_py = _python_inner_product_diag(data[0], template[0], invC[0], df)
        h_h_py = _python_inner_product_diag(template[0], template[0], invC[0], df)
        like_py = -0.5 * (d_d_py + h_h_py - 2.0 * d_h_py).real

        np.testing.assert_allclose(like_cpp[0], like_py, rtol=1e-10)


# =====================================================================
# Test 6: Multi-container d_d selection — FD + AET
# =====================================================================


class TestMultiContainerDdSelection:
    """Multiple containers, multiple binaries with repeated data_index.

    Verifies that each binary selects the correct d_d value from the
    per-container array, even when len(data_index) >> num_data.
    """

    def test_per_binary_d_d(self):
        rng = np.random.default_rng(7777)
        num_channels = 2
        num_freqs = 256
        df = 1.0 / 3600.0
        f_min = df
        f_max = f_min + (num_freqs - 1) * df
        num_data = 3
        num_noise = 3

        # 3 distinct data containers and noise instances
        data = (rng.standard_normal((num_data, num_channels, num_freqs))
                + 1j * rng.standard_normal((num_data, num_channels, num_freqs)))
        invC = np.abs(rng.standard_normal((num_noise, num_channels, num_freqs))) + 0.1
        invC = invC.astype(np.complex128)

        # 8 binaries — more than num_data, with repeated data_index values
        # Simulates ntemps * nwalkers scenario
        num_binaries = 8
        data_indices = [0, 1, 2, 0, 1, 2, 0, 1]
        noise_indices = [0, 0, 1, 1, 2, 2, 0, 1]

        templates = (rng.standard_normal((num_binaries, num_channels, num_freqs))
                     + 1j * rng.standard_normal((num_binaries, num_channels, num_freqs)))

        settings = FDSettings(N=num_freqs, df=df, min_freq=f_min, max_freq=f_max, force_backend="cpu")

        dcg = FDComputationGroup(
            data_arr=data.ravel(),
            invC_arr=invC.ravel(),
            num_data=num_data,
            num_noise=num_noise,
            num_channels=num_channels,
            settings=settings,
            tdi_type="AET",
            force_backend="cpu",
        )

        # Compute d_d per container (shape (num_data,))
        d_d = np.zeros(num_data, dtype=np.float64)
        for i in range(num_data):
            d_d[i] = _python_inner_product_diag(data[i], data[i], invC[i], df).real
        dcg.d_d = d_d

        start_freqs = np.full(num_binaries, f_min)
        data_index = np.array(data_indices, dtype=np.int32)
        noise_index = np.array(noise_indices, dtype=np.int32)

        like_cpp = dcg.compute_likelihood(
            templates, start_freqs, data_index, noise_index
        )

        # Python reference: compute per-binary likelihood
        for b in range(num_binaries):
            di = data_indices[b]
            ni = noise_indices[b]
            d_h_b = _python_inner_product_diag(data[di], templates[b], invC[ni], df)
            h_h_b = _python_inner_product_diag(templates[b], templates[b], invC[ni], df)
            like_py = -0.5 * (d_d[di] + h_h_b - 2.0 * d_h_b).real

            np.testing.assert_allclose(
                like_cpp[b], like_py, rtol=1e-10,
                err_msg=f"Likelihood mismatch for binary {b} (data_index={di})"
            )
