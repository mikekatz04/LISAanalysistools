"""Validate the FD time-averaged transfer-function covariance.

Oracle: the averaging feature builds C from epoch-averaged transfer functions and
multiplies by the noise amplitudes; that must equal the numpy mean of the full
per-epoch covariance C(f;L(t)) (for Amp=0, exactly, since C is linear in the TFs
and the amplitudes S are epoch-independent). CPU-only; no GPU required.

The averaged backend decimates the orbit's native LTT time grid internally, so the
oracle pulls the exact epochs the feature used straight from get_averaged_ltts().
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU backend
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

from lisatools.detector import L1Orbits
from lisatools.domains import FDSettings
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gf_dev"))
from ltt_averaging_diagnostic import to_tt, to_aet_diag  # validated AET projection

NOISE_FILE = ("/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
              "data/INSTRUMENT/L1/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
Soms_d, Sa_a, Tobs = 15e-12, 3e-15, 0.9 * YRSID_SI

requires_data = pytest.mark.skipif(not os.path.exists(NOISE_FILE),
                                   reason="mojito noise file unavailable")


def _to_np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _backend(average):
    orbits = L1Orbits(NOISE_FILE, force_backend="cpu", frame="icrs")
    orbits.configure(linear_interp_setup=True)
    # df chosen so the [min_freq, max_freq] band fits within N (active_slice end_idx
    # is NOT clamped to N), keeping total_terms == len(f_arr).
    fd = FDSettings(N=4096, df=5e-6, min_freq=1e-4, max_freq=2e-2, force_backend="cpu")
    bk = XYZSensitivityBackend(orbits=orbits, settings=fd, tdi_generation=2,
                               force_backend="cpu", average_transfer_functions=average)
    return orbits, bk


def _per_epoch_mean_cov(backend, f_arr, avg, delta):
    """Reference: build C(f;L(t)) per epoch from the (averaged-pair) LTTs the feature
    used, then mean over epochs."""
    N = avg.shape[0]
    nf = len(f_arr)
    # the wrap stores these LTT pointers NON-OWNED -> keep the arrays alive in named
    # locals (an inline temporary is freed before get_noise_covariance_wrap runs).
    avg_flat = np.ascontiguousarray(avg.flatten())
    delta_flat = np.ascontiguousarray(delta.flatten())
    wrap = backend.backend.SensitivityMatrixWrap(
        avg_flat, delta_flat, N, float(backend.orbits.armlength), 2, False, 1.0)
    c = [np.empty(N * nf, dtype=np.float64 if r else np.complex128)
         for r in (True, False, False, True, False, True)]
    wrap.get_noise_covariance_wrap(
        np.ascontiguousarray(f_arr), np.arange(N, dtype=np.int32),
        float(Soms_d), float(Sa_a), 0.0, 0.0, 0.0, 0.0, 0.0,
        np.zeros(nf), np.zeros(nf), c[0], c[1], c[2], c[3], c[4], c[5], nf, N)
    r = lambda a: a.reshape(N, nf)
    C = np.empty((3, 3, N, nf), dtype=np.complex128)
    C[0, 0] = r(c[0]); C[1, 1] = r(c[3]); C[2, 2] = r(c[5])
    C[0, 1] = r(c[1]); C[1, 0] = np.conj(r(c[1]))
    C[0, 2] = r(c[2]); C[2, 0] = np.conj(r(c[2]))
    C[1, 2] = r(c[4]); C[2, 1] = np.conj(r(c[4]))
    return C.mean(axis=2)


@requires_data
def test_averaged_backend_matches_per_epoch_mean():
    orbits, bk = _backend(True)
    assert bk._averaging_active
    # the exact (N, 6) averaged-pair LTTs the feature averaged the TFs over
    favg, fdelta = bk.get_averaged_ltts()
    favg = _to_np(favg); fdelta = _to_np(fdelta)
    f_arr = np.asarray(bk.f_arr)
    C_backend = np.asarray(bk.compute_sensitivity_matrix(bk.f_arr, Soms_d, Sa_a))
    C_ref = _per_epoch_mean_cov(bk, f_arr, favg, fdelta)
    assert np.allclose(C_backend, C_ref, rtol=1e-9, atol=0.0)


@requires_data
def test_averaging_shifts_TT_not_AA():
    _, bk_avg = _backend(True)
    _, bk_one = _backend(False)   # single median-LTT epoch, i.e. C(E[L])
    f = np.asarray(bk_avg.f_arr)
    Cavg = np.asarray(bk_avg.compute_sensitivity_matrix(bk_avg.f_arr, Soms_d, Sa_a))
    Cone = np.asarray(bk_one.compute_sensitivity_matrix(bk_one.f_arr, Soms_d, Sa_a))
    lowf = f < 5e-4
    assert to_tt(Cavg)[lowf].mean() > 1.05 * to_tt(Cone)[lowf].mean()        # TT null raised
    assert np.allclose(to_aet_diag(Cavg)[0], to_aet_diag(Cone)[0], rtol=0.02)  # AA ~unchanged
