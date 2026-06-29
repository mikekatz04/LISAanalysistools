"""Tests for the ACA-owned C++ likelihood backend + forwarder.

Covers the additive integration in :class:`AnalysisContainerArray`:

* the lazily-built, owned :class:`DomainComputationGroupArray`
  (:attr:`cpp_likelihood_backend` / :attr:`domain_computation_group`,
  configured via :attr:`domain_group_kwargs`);
* the :meth:`AnalysisContainerArray.cpp_template_likelihood` forwarder
  that drives the multi-split propagation through the owned backend; and
* :meth:`refresh_cpp_dd` for the cached ``(d|d)`` term.

The forwarder is pure plumbing over the DCGA primitives
(``unpack_indices`` / ``unpack_coords`` / ``place_on_device`` /
``cpp_signal_likelihood``), all already covered by
``test_multi_gpu_placement.py``. Here we exercise the *forwarder* code path
itself, two ways:

* **WDM** (:class:`TestWDMForwarderRealKernel`) drives the **real C++ WDM
  likelihood kernel** (``WDMDomainWrap``) — the lazily-built real
  ``WDMComputationGroup`` runs on CPU. This validates the forwarder feeding a
  genuine kernel end-to-end, including the lazy backend build.
* **FD** (:class:`TestFDForwarderRealKernel`) drives the **real C++ FD
  likelihood kernel** (``FDDomainForStftWrap``) — the lazily-built real
  ``FDComputationGroup`` runs on CPU and matches the full-covariance reference
  to machine precision. Covers the complex-dtype + ``start_times=None`` path.

* **STFT** (:class:`TestSTFTForwarderRealKernel`) drives the **real C++ STFT
  kernel** (``STFTDomainWrap``) over a 4-D time-frequency template. The STFT
  inner product is the FD full-cov product summed over time bins (``4 df`` per
  ``(t,f)`` pixel), so the same ``_cross_inner`` reference applies.

Together WDM (4-index, real), FD (3-index, complex) and STFT (4-index, complex)
exercise the shape and dtype generality of the forwarder.

We drive the *real* ``AnalysisContainerArray`` methods (borrowed onto a
lightweight ACS host that carries exactly the attributes the backend reads),
so the code under test is the production forwarder, not a re-implementation.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from lisatools.analysiscontainer import AnalysisContainerArray as _ACA
from lisatools.domains import FDSettings, STFTSettings, WDMSettings


# ---------------------------------------------------------------------------
# Re-instantiable stub sensitivity / orbits / AC (mirrors test_multi_gpu_placement).
# build_cpp_objects rebuilds orbits + sensitivity per split via the
# ``args`` / ``kwargs`` reconstruction protocol; the stubs satisfy exactly
# that, with no C++ sensitivity objects touched. The per-AC ``inner_product``
# returns the precomputed (d|d) so the cached term is deterministic.
# ---------------------------------------------------------------------------


class _StubOrbits:
    def __init__(self):
        self.args = ()
        self.kwargs = {}


class _StubSensMat:
    def __init__(self, orbits=None):
        self.orbits = orbits if orbits is not None else _StubOrbits()
        self.kwargs = {"orbits": self.orbits}


class _StubAC:
    def __init__(self, d_d_value: float):
        self._d_d = float(d_d_value)
        self.sens_mat = _StubSensMat()

    def inner_product(self, **_):
        return self._d_d


class _ACSHost:
    """A stub ``AnalysisContainerArray`` that *borrows* the real forwarder.

    Carries every attribute the owned ``DomainComputationGroupArray`` reads
    (``settings`` / ``gpus`` / ``xp`` / ``acs_total_entries`` /
    ``gpu_splits`` / ``split_map`` / ``linear_data_arr`` / ``linear_psd_arr``
    / ``nchannels`` / ``acs``), plus the ``xp`` / ``data_dtype`` /
    ``run_threaded`` the forwarder reads, and the lazy-state attributes. The
    forwarder + accessors are the genuine ``AnalysisContainerArray``
    implementations, assigned here so the test exercises production code.
    """

    # --- borrowed from AnalysisContainerArray, under test ---
    cpp_template_likelihood = _ACA.cpp_template_likelihood
    refresh_cpp_dd = _ACA.refresh_cpp_dd
    unpack_indices = _ACA.unpack_indices
    unpack_coords = _ACA.unpack_coords
    place_on_device = _ACA.place_on_device
    _loop_operation = _ACA._loop_operation
    device_context = _ACA.device_context
    free_gpu_memory = _ACA.free_gpu_memory
    _to_host = _ACA._to_host
    synchronize = _ACA.synchronize
    compute_d_d_terms = _ACA.compute_d_d_terms
    compute_noise_terms = _ACA.compute_noise_terms
    _compute_group_likelihood = _ACA._compute_group_likelihood
    cpp_signal_likelihood = _ACA.cpp_signal_likelihood
    cpp_psd_likelihood = _ACA.cpp_psd_likelihood
    _cpp_strategy_class = _ACA._cpp_strategy_class
    _build_cpp_splits = _ACA._build_cpp_splits
    _ensure_cpp_splits = _ACA._ensure_cpp_splits
    cpp_split = _ACA.cpp_split
    # borrowed properties
    cpp_likelihood_backend = _ACA.cpp_likelihood_backend
    domain_computation_group = _ACA.domain_computation_group
    domain_group_kwargs = _ACA.domain_group_kwargs
    num_splits = _ACA.num_splits
    ac_to_split = _ACA.ac_to_split
    cpp_splits = _ACA.cpp_splits
    thread_pool = _ACA.thread_pool

    xp = np
    run_threaded = False

    def __init__(
        self,
        settings,
        data,
        invC,
        d_d_values,
        nchannels,
        num_splits,
        data_dtype,
        domain_group_kwargs=None,
    ):
        num_acs = data.shape[0]
        if num_acs % num_splits:
            raise ValueError("num_acs must be divisible by num_splits.")

        self.settings = settings
        self.gpus = None
        self.nchannels = nchannels
        self.acs_total_entries = int(num_acs)
        self.data_dtype = data_dtype

        split_num = int(np.ceil(num_acs / num_splits))
        split_inds = np.arange(split_num, num_acs, split_num)
        self.gpu_splits = np.split(np.arange(num_acs), split_inds)
        assert len(self.gpu_splits) == num_splits
        self.split_map = np.zeros(num_acs, dtype=int)
        for split_id, entries in enumerate(self.gpu_splits):
            self.split_map[entries] = split_id

        self.linear_data_arr = [
            np.concatenate([data[i].ravel() for i in entries])
            for entries in self.gpu_splits
        ]
        self.linear_psd_arr = [
            np.concatenate([invC[i].ravel() for i in entries])
            for entries in self.gpu_splits
        ]
        self.acs = np.asarray([_StubAC(v) for v in d_d_values], dtype=object)

        # Routing table the borrowed unpack_indices reads (ac_to_split = split_map).
        self.ac_to_intra = np.empty(num_acs, dtype=np.int32)
        for split_id, entries in enumerate(self.gpu_splits):
            self.ac_to_intra[entries] = np.arange(len(entries), dtype=np.int32)

        # Lazy-state attributes the borrowed accessors read/write.
        self._domain_group_kwargs = dict(domain_group_kwargs or {})
        self._cpp_splits = None
        self._cpp_likelihood_backend = None
        self._thread_pool = None


# ---------------------------------------------------------------------------
# Full-covariance XYZ FD inner-product reference (from test_multi_gpu_placement).
# ---------------------------------------------------------------------------


def _cross_inner(a, b, invC, df):
    """``<a|b> = 4 df sum_{i,j,f} conj(a[i,f]) invC[i,j,f] b[j,f]`` (XYZ FD)."""
    nch = a.shape[0]
    acc = 0.0 + 0.0j
    for i in range(nch):
        for j in range(nch):
            acc += np.sum(np.conj(a[i]) * invC[i, j] * b[j])
    return 4.0 * df * acc


# ---------------------------------------------------------------------------
# WDM: real C++ kernel through the forwarder (lazy backend build).
# ---------------------------------------------------------------------------


class TestWDMForwarderRealKernel(unittest.TestCase):
    """``AnalysisContainerArray.cpp_template_likelihood`` against the real
    WDM kernel, exercising the lazy backend build + multi-split propagation."""

    Nf, Nt, dt = 32, 64, 5.0
    IMN_F, IMX_F, IMN_T, IMX_T = 3, 20, 4, 50
    NCH = 3
    N_M, N_N = 4, 8

    @classmethod
    def setUpClass(cls):
        cls.layer_df = 1.0 / (2 * cls.Nf * cls.dt)
        cls.layer_dt = cls.Nf * cls.dt
        cls.Nf_a = cls.IMX_F - cls.IMN_F + 1
        cls.Nt_a = cls.IMX_T - cls.IMN_T + 1

    def _settings(self):
        # Half-pixel margins so the index setters land exactly on target.
        return WDMSettings(
            self.Nf, self.Nt, self.dt,
            min_freq=(self.IMN_F - 0.5) * self.layer_df,
            max_freq=(self.IMX_F + 0.5) * self.layer_df,
            min_time=(self.IMN_T - 0.5) * self.layer_dt,
            max_time=(self.IMX_T + 0.5) * self.layer_dt,
        )

    def _make_data(self, num_acs, seed):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((num_acs, self.NCH, self.Nf_a, self.Nt_a))
        invC = rng.uniform(0.5, 2.0, (num_acs, self.NCH, self.Nf_a, self.Nt_a))
        d_d = np.array([4 * 0.25 * np.sum(data[i] ** 2 * invC[i]) for i in range(num_acs)])
        return data, invC, d_d

    def _make_host(self, num_acs, num_splits, seed=0):
        data, invC, d_d = self._make_data(num_acs, seed)
        host = _ACSHost(
            self._settings(), data, invC, d_d,
            nchannels=self.NCH, num_splits=num_splits, data_dtype=float,
            domain_group_kwargs={"tdi_type": "AET"},
        )
        host._raw = (data, invC, d_d)  # for references
        return host

    def _make_batch(self, num_acs, nb, seed=7):
        rng = np.random.default_rng(seed)
        data_index = np.tile(np.arange(num_acs), int(np.ceil(nb / num_acs)))[:nb].astype(np.int32)
        start_m = rng.integers(self.IMN_F, self.IMX_F - self.N_M + 2, nb)
        start_n = rng.integers(self.IMN_T, self.IMX_T - self.N_N + 2, nb)
        templ = rng.standard_normal((nb, self.NCH, self.N_M, self.N_N))
        start_freqs = (start_m * self.layer_df).astype(np.float64)
        start_times = (start_n * self.layer_dt).astype(np.float64)
        return data_index, templ, start_freqs, start_times, start_m, start_n

    def _reference(self, host, data_index, templ, start_m, start_n):
        data, invC, d_d = host._raw
        nb = data_index.shape[0]
        out = np.zeros(nb)
        for b in range(nb):
            sl_m = slice(start_m[b] - self.IMN_F, start_m[b] - self.IMN_F + self.N_M)
            sl_n = slice(start_n[b] - self.IMN_T, start_n[b] - self.IMN_T + self.N_N)
            d_sub = data[data_index[b]][:, sl_m, sl_n]
            w_sub = invC[data_index[b]][:, sl_m, sl_n]
            h = templ[b]
            d_h = 4 * 0.25 * np.sum(d_sub * h * w_sub)
            h_h = 4 * 0.25 * np.sum(h ** 2 * w_sub)
            out[b] = -0.5 * (d_d[data_index[b]] + h_h - 2 * d_h)
        return out

    def test_lazy_build_then_matches_reference(self):
        host = self._make_host(num_acs=4, num_splits=2)
        self.assertIsNone(host._cpp_splits)
        data_index, templ, sf, st, sm, sn = self._make_batch(4, nb=6)

        out = host.cpp_template_likelihood(data_index, templ, sf, st)

        # Lazy build happened and produced the real WDM strategy per split.
        self.assertIsNotNone(host._cpp_splits)
        self.assertEqual(
            type(host.cpp_splits[0]).__name__,
            "WDMComputationGroup",
        )
        ref = self._reference(host, data_index, templ, sm, sn)
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)

    def test_parity_vs_hand_driven_coordinator(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st, _, _ = self._make_batch(4, nb=6)

        host._ensure_cpp_splits()  # build the per-split strategies once
        pos, di, ni = host.unpack_indices(data_index, None)
        coords = host.unpack_coords(pos, (templ.astype(float), sf, st), keep_tuple=True)
        di, ni, coords = host.place_on_device((di, ni, coords))
        ref = host.cpp_signal_likelihood(pos, di, ni, coords)

        out = host.cpp_template_likelihood(data_index, templ, sf, st)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_split_layout_invariance(self):
        h1 = self._make_host(num_acs=4, num_splits=1, seed=123)
        h2 = self._make_host(num_acs=4, num_splits=2, seed=123)
        data_index, templ, sf, st, _, _ = self._make_batch(4, nb=8)
        o1 = h1.cpp_template_likelihood(data_index, templ, sf, st)
        o2 = h2.cpp_template_likelihood(data_index, templ, sf, st)
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)

    def test_run_threaded_matches_serial(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st, _, _ = self._make_batch(4, nb=6)
        serial = host.cpp_template_likelihood(data_index, templ, sf, st, run_threaded=False)
        threaded = host.cpp_template_likelihood(data_index, templ, sf, st, run_threaded=True)
        np.testing.assert_array_equal(serial, threaded)

    def test_empty_split_no_crash(self):
        host = self._make_host(num_acs=4, num_splits=2)
        # All binaries resolve to AC 0 -> split 1 is empty.
        _, templ, sf, st, sm, sn = self._make_batch(4, nb=6)
        data_index = np.zeros(6, dtype=np.int32)
        out = host.cpp_template_likelihood(data_index, templ, sf, st)
        ref = self._reference(host, data_index, templ, sm, sn)
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)

    def test_noise_index_defaults_to_data_index(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st, _, _ = self._make_batch(4, nb=6)
        a = host.cpp_template_likelihood(data_index, templ, sf, st, noise_index=None)
        b = host.cpp_template_likelihood(
            data_index, templ, sf, st, noise_index=data_index.copy()
        )
        np.testing.assert_array_equal(a, b)

    def test_dtype_coercion(self):
        """A float32 template is coerced to data_dtype (float64) and matches."""
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st, sm, sn = self._make_batch(4, nb=6)
        out = host.cpp_template_likelihood(
            data_index, templ.astype(np.float32), sf.astype(np.float32), st
        )
        ref = self._reference(host, data_index, templ, sm, sn)
        # float32 templates lose precision, so a looser tolerance.
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_stale_dd_refresh(self):
        """(d|d) is cached at build time; refresh recomputes it after mutation."""
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st, sm, sn = self._make_batch(4, nb=6)
        host.cpp_template_likelihood(data_index, templ, sf, st)  # build backend

        # Mutate the cached (d|d) source on the stub ACs, leaving the kernel's
        # (d|h)/(h|h) inputs untouched: the forwarder result must shift by
        # exactly -0.5 * delta(d_d) per binary only after a refresh.
        delta = 10.0
        for ac in host.acs:
            ac._d_d += delta
        before = host.cpp_template_likelihood(data_index, templ, sf, st)
        host.refresh_cpp_dd()
        after = host.cpp_template_likelihood(data_index, templ, sf, st)
        np.testing.assert_allclose(after - before, -0.5 * delta, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# FD: stub group through the forwarder (complex dtype, start_times=None).
# ---------------------------------------------------------------------------


class TestFDForwarderRealKernel(unittest.TestCase):
    """``cpp_template_likelihood`` forwarder over the complex FD path,
    driving the real C++ ``FDComputationGroup`` (``FDDomainForStftWrap``)."""

    NCH = 3
    NFREQ = 64
    DF = 1e-3

    def _make_host(self, num_acs, num_splits, seed=1):
        rng = np.random.default_rng(seed)
        data = (
            rng.standard_normal((num_acs, self.NCH, self.NFREQ))
            + 1j * rng.standard_normal((num_acs, self.NCH, self.NFREQ))
        )
        invC = np.zeros((num_acs, self.NCH, self.NCH, self.NFREQ), dtype=np.complex128)
        for ac in range(num_acs):
            for f in range(self.NFREQ):
                A = rng.standard_normal((self.NCH, self.NCH)) + 1j * rng.standard_normal(
                    (self.NCH, self.NCH)
                )
                invC[ac, :, :, f] = A @ A.conj().T + 3.0 * np.eye(self.NCH)
        d_d = np.array(
            [_cross_inner(data[i], data[i], invC[i], self.DF).real for i in range(num_acs)]
        )
        settings = FDSettings(
            N=self.NFREQ + 1, df=self.DF, min_freq=self.DF,
            max_freq=self.NFREQ * self.DF, force_backend="cpu",
        )
        host = _ACSHost(
            settings, data, invC, d_d,
            nchannels=self.NCH, num_splits=num_splits, data_dtype=complex,
        )
        # Drives the real FDComputationGroup: the borrowed _cpp_strategy_class
        # dispatches FDSettings -> FDComputationGroup, and the stub
        # orbits/sensitivity satisfy build_cpp_objects (the signal inner
        # products don't use orbits). The real FDDomainForStftWrap kernel
        # matches _cross_inner to machine precision.
        host._raw = (data, invC, d_d)
        return host

    def _batch(self, num_acs, nb, seed=7):
        rng = np.random.default_rng(seed)
        data_index = np.tile(np.arange(num_acs), int(np.ceil(nb / num_acs)))[:nb].astype(np.int32)
        templ = (
            rng.standard_normal((nb, self.NCH, self.NFREQ))
            + 1j * rng.standard_normal((nb, self.NCH, self.NFREQ))
        )
        start_freqs = np.full(nb, self.DF, dtype=np.float64)
        return data_index, templ, start_freqs

    def _reference(self, host, data_index, templ):
        data, invC, d_d = host._raw
        out = np.zeros(data_index.shape[0])
        for b in range(data_index.shape[0]):
            d = data[data_index[b]]
            ic = invC[data_index[b]]
            h = templ[b]
            d_h = _cross_inner(d, h, ic, self.DF)
            h_h = _cross_inner(h, h, ic, self.DF)
            out[b] = -0.5 * (d_d[data_index[b]] + h_h - 2 * d_h).real
        return out

    def test_matches_reference_no_start_times(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf = self._batch(4, nb=6)
        out = host.cpp_template_likelihood(data_index, templ, sf, start_times=None)
        ref = self._reference(host, data_index, templ)
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_parity_vs_hand_driven_coordinator(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf = self._batch(4, nb=6)
        host._ensure_cpp_splits()
        pos, di, ni = host.unpack_indices(data_index, None)
        coords = host.unpack_coords(pos, (templ.astype(complex), sf), keep_tuple=True)
        di, ni, coords = host.place_on_device((di, ni, coords))
        ref = host.cpp_signal_likelihood(pos, di, ni, coords)
        out = host.cpp_template_likelihood(data_index, templ, sf, start_times=None)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_split_layout_invariance(self):
        h1 = self._make_host(num_acs=4, num_splits=1, seed=99)
        h2 = self._make_host(num_acs=4, num_splits=2, seed=99)
        data_index, templ, sf = self._batch(4, nb=8)
        o1 = h1.cpp_template_likelihood(data_index, templ, sf)
        o2 = h2.cpp_template_likelihood(data_index, templ, sf)
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)

    def test_run_threaded_matches_serial(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf = self._batch(4, nb=6)
        s = host.cpp_template_likelihood(data_index, templ, sf, run_threaded=False)
        t = host.cpp_template_likelihood(data_index, templ, sf, run_threaded=True)
        np.testing.assert_array_equal(s, t)

    def test_dtype_coercion_complex64(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf = self._batch(4, nb=6)
        out = host.cpp_template_likelihood(data_index, templ.astype(np.complex64), sf)
        ref = self._reference(host, data_index, templ)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# STFT: real C++ time-frequency kernel through the forwarder.
# ---------------------------------------------------------------------------


class TestSTFTForwarderRealKernel(unittest.TestCase):
    """``cpp_template_likelihood`` forwarder over the time-frequency STFT path,
    driving the real C++ ``STFTComputationGroup`` (``STFTDomainWrap``). The STFT
    inner product is the FD full-cov product summed over time bins (``4 df`` per
    ``(t, f)`` pixel), so the same ``_cross_inner`` reference applies with
    ``(nch, NT, NF)`` arrays. ``start_times``/``start_freqs`` are chosen to put
    the template sub-grid at index ``(0, 0)`` (full-grid templates)."""

    NCH = 3
    NT = 4
    NFREQ = 8  # == NF_active given min_freq=DF, max_freq=NFREQ*DF, NF=NFREQ+1
    DT = 10.0
    DF = 1e-3

    def _make_host(self, num_acs, num_splits, seed=1):
        rng = np.random.default_rng(seed)
        shp = (num_acs, self.NCH, self.NT, self.NFREQ)
        data = rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
        invC = np.zeros(
            (num_acs, self.NCH, self.NCH, self.NT, self.NFREQ), dtype=np.complex128
        )
        for ac in range(num_acs):
            for t in range(self.NT):
                for f in range(self.NFREQ):
                    A = rng.standard_normal((self.NCH, self.NCH)) + 1j * rng.standard_normal(
                        (self.NCH, self.NCH)
                    )
                    invC[ac, :, :, t, f] = A @ A.conj().T + 3.0 * np.eye(self.NCH)
        d_d = np.array(
            [_cross_inner(data[i], data[i], invC[i], self.DF).real for i in range(num_acs)]
        )
        settings = STFTSettings(
            t0=0.0, dt=self.DT, df=self.DF, NT=self.NT, NF=self.NFREQ + 1,
            min_freq=self.DF, max_freq=self.NFREQ * self.DF, force_backend="cpu",
        )
        assert settings.NF_active == self.NFREQ
        host = _ACSHost(
            settings, data, invC, d_d,
            nchannels=self.NCH, num_splits=num_splits, data_dtype=complex,
        )
        # Real STFTComputationGroup (STFTDomainWrap); stub orbits/sensitivity
        # satisfy build_cpp_objects (signal inner products don't use orbits).
        host._raw = (data, invC, d_d)
        return host

    def _batch(self, num_acs, nb, seed=7):
        rng = np.random.default_rng(seed)
        data_index = np.tile(np.arange(num_acs), int(np.ceil(nb / num_acs)))[:nb].astype(np.int32)
        shp = (nb, self.NCH, self.NT, self.NFREQ)
        templ = rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
        start_freqs = np.full(nb, self.DF, dtype=np.float64)  # -> freq index 0
        start_times = np.full(nb, 0.0, dtype=np.float64)      # -> time index 0
        return data_index, templ, start_freqs, start_times

    def _reference(self, host, data_index, templ):
        data, invC, d_d = host._raw
        out = np.zeros(data_index.shape[0])
        for b in range(data_index.shape[0]):
            d = data[data_index[b]]
            ic = invC[data_index[b]]
            h = templ[b]
            d_h = _cross_inner(d, h, ic, self.DF)
            h_h = _cross_inner(h, h, ic, self.DF)
            out[b] = -0.5 * (d_d[data_index[b]] + h_h - 2 * d_h).real
        return out

    def test_matches_reference(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st = self._batch(4, nb=6)
        out = host.cpp_template_likelihood(data_index, templ, sf, start_times=st)
        ref = self._reference(host, data_index, templ)
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_parity_vs_hand_driven_coordinator(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st = self._batch(4, nb=6)
        host._ensure_cpp_splits()
        pos, di, ni = host.unpack_indices(data_index, None)
        coords = host.unpack_coords(pos, (templ.astype(complex), sf, st), keep_tuple=True)
        di, ni, coords = host.place_on_device((di, ni, coords))
        ref = host.cpp_signal_likelihood(pos, di, ni, coords)
        out = host.cpp_template_likelihood(data_index, templ, sf, start_times=st)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_split_layout_invariance(self):
        h1 = self._make_host(num_acs=4, num_splits=1, seed=99)
        h2 = self._make_host(num_acs=4, num_splits=2, seed=99)
        data_index, templ, sf, st = self._batch(4, nb=8)
        o1 = h1.cpp_template_likelihood(data_index, templ, sf, start_times=st)
        o2 = h2.cpp_template_likelihood(data_index, templ, sf, start_times=st)
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)

    def test_run_threaded_matches_serial(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st = self._batch(4, nb=6)
        s = host.cpp_template_likelihood(data_index, templ, sf, start_times=st, run_threaded=False)
        t = host.cpp_template_likelihood(data_index, templ, sf, start_times=st, run_threaded=True)
        np.testing.assert_array_equal(s, t)

    def test_dtype_coercion_complex64(self):
        host = self._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st = self._batch(4, nb=6)
        out = host.cpp_template_likelihood(
            data_index, templ.astype(np.complex64), sf, start_times=st
        )
        ref = self._reference(host, data_index, templ)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# State semantics of the lazy accessor + kwargs setter.
# ---------------------------------------------------------------------------


class TestForwarderStateSemantics(unittest.TestCase):
    NCH, NFREQ, DF = 3, 16, 1e-3

    def _host(self, num_splits=2):
        rng = np.random.default_rng(3)
        num_acs = 4
        data = (
            rng.standard_normal((num_acs, self.NCH, self.NFREQ))
            + 1j * rng.standard_normal((num_acs, self.NCH, self.NFREQ))
        )
        invC = np.zeros((num_acs, self.NCH, self.NCH, self.NFREQ), dtype=np.complex128)
        for ac in range(num_acs):
            for f in range(self.NFREQ):
                A = rng.standard_normal((self.NCH, self.NCH)) + 1j * rng.standard_normal(
                    (self.NCH, self.NCH)
                )
                invC[ac, :, :, f] = A @ A.conj().T + 3.0 * np.eye(self.NCH)
        d_d = np.zeros(num_acs)
        settings = FDSettings(
            N=self.NFREQ + 1, df=self.DF, min_freq=self.DF,
            max_freq=self.NFREQ * self.DF, force_backend="cpu",
        )
        return _ACSHost(
            settings, data, invC, d_d, nchannels=self.NCH,
            num_splits=num_splits, data_dtype=complex,
            domain_group_kwargs={"tdi_type": "XYZ"},
        )

    def test_lazy_none_until_accessed(self):
        host = self._host()
        self.assertIsNone(host._cpp_splits)
        self.assertIsNone(host._cpp_likelihood_backend)
        first = host.cpp_likelihood_backend  # builds strategies + the shim
        self.assertIsNotNone(host._cpp_splits)
        self.assertIs(first, host.cpp_likelihood_backend)  # cached, same object
        self.assertIs(first, host.domain_computation_group)  # alias

    def test_domain_group_kwargs_setter_invalidates_cache(self):
        host = self._host()
        host.cpp_likelihood_backend  # build strategies + shim
        self.assertIsNotNone(host._cpp_likelihood_backend)
        self.assertIsNotNone(host._cpp_splits)
        host.domain_group_kwargs = {"tdi_type": "AET"}
        # Setter must reset both the strategy list and the shim cache.
        self.assertIsNone(host._cpp_likelihood_backend)
        self.assertIsNone(host._cpp_splits)
        self.assertEqual(host.domain_group_kwargs, {"tdi_type": "AET"})

    def test_refresh_noop_when_not_built(self):
        host = self._host()
        self.assertIsNone(host._cpp_splits)
        host.refresh_cpp_dd()  # must not raise
        self.assertIsNone(host._cpp_splits)

    def test_direct_construction_emits_deprecation_warning(self):
        """Constructing the deprecated alias directly warns; the ACA's own
        ``cpp_likelihood_backend`` compat handle (``_internal=True``) does not.
        The thin alias still forwards state to the ACA."""
        from lisatools.domaincomputation import DomainComputationGroupArray

        host = self._host()
        with self.assertWarns(DeprecationWarning):
            shim = DomainComputationGroupArray(host)
        self.assertIs(shim.acs, host)
        self.assertIs(shim.computation_groups, host.cpp_splits)
        self.assertEqual(shim.num_splits, host.num_splits)
        self.assertIs(shim.xp, host.xp)

        # The internal compat handle must stay quiet.
        host._cpp_likelihood_backend = None
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            host.cpp_likelihood_backend  # must not raise (no warning)


if __name__ == "__main__":
    unittest.main()
