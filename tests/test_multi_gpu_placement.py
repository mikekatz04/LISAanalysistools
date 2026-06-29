"""Placement + routing tests for the :class:`AnalysisContainerArray` C++
likelihood coordinator (absorbed from the former
:class:`DomainComputationGroupArray`).

These tests exercise the multi-split routing logic introduced by the
multi-GPU refactor *without* requiring any real GPUs. We drive the real
ACA coordinator methods (borrowed onto a lightweight ``_StubACS`` host)
that read exactly the attributes the coordinator and each
``DomainKernelStrategy.extract_from_acs`` / ``build_cpp_objects`` path
need. (The strategy base is imported here under its back-compat alias
``BaseDomainComputationGroup`` so the tests also cover that the alias
still resolves.)

Key properties verified:
    * ``ac_to_split`` / ``ac_to_intra`` routing tables match
      ``gpu_splits``.
    * ``unpack_indices`` partitions flat ``(data_index, noise_index)``
      batches by split into ``(positions, data_intra, noise_intra)``.
    * ``compute_d_d_terms`` populates each group's per-split ``d_d``
      vector in intra-split order (and runs automatically at
      coordinator construction).
    * ``cpp_signal_likelihood`` reproduces the per-binary reference
      and is invariant to the split layout (single split vs. two
      splits), including empty splits.
    * ``run_threaded=True`` (ThreadPoolExecutor dispatch) matches the
      serial path.
    * ``unpack_coords`` slices flat coordinate arrays per split and
      returns ``()`` for empty splits.
    * ``_loop_operation`` supports callables with per-split args/kwargs
      and short-circuits empty splits via ``positions_per_split``.
    * ``place_on_device`` round-trips arrays / tuples per split (CPU
      no-op device placement).

All work runs on the CPU backend with ``gpus=None``; the multi-split
dimension is controlled entirely via ``gpu_splits``. The C++-backed
``FDComputationGroup`` is swapped for a NumPy stub group (the real
likelihood kernels are exercised elsewhere); everything above the
group's ``compute_signal_likelihood_terms`` — extraction, routing,
scatter-back — is the real library code.
"""

from __future__ import annotations

import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainerArray as _ACA
from lisatools.domaincomputation import BaseDomainComputationGroup
from lisatools.domains import FDSettings


def _cross_inner(a, b, invC, df):
    """Python reference for the full (XYZ) FD inner product.

    ``<a|b> = 4 * df * sum_{i,j,f} conj(a[i,f]) * invC[i,j,f] * b[j,f]``.
    This mirrors the XYZ path inside ``domains.cu`` — the default
    ``tdi_type`` for :class:`BaseDomainComputationGroup` is ``"XYZ"``.
    """
    num_channels = a.shape[0]
    acc = 0.0 + 0.0j
    for i in range(num_channels):
        for j in range(num_channels):
            acc += np.sum(np.conj(a[i]) * invC[i, j] * b[j])
    return 4.0 * df * acc


class _StubOrbits:
    """Re-instantiable orbits stand-in.

    ``BaseDomainComputationGroup.build_cpp_objects`` rebuilds the
    orbits per split via ``orbits.__class__(*orbits.args, **orbits.kwargs)``
    so each device gets its own copy; the stub only needs the
    ``args`` / ``kwargs`` reconstruction protocol.
    """

    def __init__(self):
        self.args = ()
        self.kwargs = {}


class _StubSensMat:
    """Re-instantiable sensitivity-backend stand-in.

    ``build_cpp_objects`` reads ``sens_mat.orbits`` (with ``args`` /
    ``kwargs``) and ``sens_mat.kwargs``, then re-instantiates
    ``sens_mat.__class__(**kwargs)`` with the rebuilt orbits swapped
    in. No C++ objects are touched on this path.
    """

    def __init__(self, orbits=None):
        self.orbits = orbits if orbits is not None else _StubOrbits()
        self.kwargs = {"orbits": self.orbits}


class _StubAC:
    """Fake ``AnalysisContainer`` exposing ``inner_product`` + ``sens_mat``.

    ``BaseDomainComputationGroup.compute_d_d_term`` loops over the
    split ACs and writes ``ac.inner_product()`` into its ``d_d`` array,
    and ``build_cpp_objects`` reads ``sens_mat`` off the split's first
    AC. Returning a precomputed scalar lets the test assert the full
    coordinator-level ``compute_d_d_terms`` wiring without needing to
    build a real ``DataResidualArray`` + ``SensitivityMatrix`` stack.
    """

    def __init__(self, d_d_value: float):
        self._d_d = float(d_d_value)
        self.sens_mat = _StubSensMat()

    def inner_product(self, **_):
        return self._d_d


class _StubACS:
    """Minimal ``AnalysisContainerArray`` host for the routing tests.

    Builds synthetic split data and **borrows the real, now-ACA-resident
    coordinator methods** (``unpack_indices`` / ``unpack_coords`` /
    ``place_on_device`` / ``_loop_operation`` / ``compute_*`` / ...), so the
    routing/dispatch code under test is the genuine library implementation.
    Only the per-split *strategy* is stubbed: ``_cpp_strategy_class`` returns
    the NumPy ``_StubFDComputationGroup`` (the merged real FDComputationGroup
    needs an unreconciled FD binding).

    CPU-only (``gpus=None``) so every split's ``device`` is ``None`` and the
    ``force_backend='cpu'`` assertion in ``extract_from_acs`` holds.
    """

    # --- borrowed from AnalysisContainerArray (the absorbed DCGA coordinator) ---
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
    _build_cpp_splits = _ACA._build_cpp_splits
    _ensure_cpp_splits = _ACA._ensure_cpp_splits
    cpp_split = _ACA.cpp_split
    # borrowed properties
    num_splits = _ACA.num_splits
    ac_to_split = _ACA.ac_to_split
    cpp_splits = _ACA.cpp_splits
    thread_pool = _ACA.thread_pool
    domain_group_kwargs = _ACA.domain_group_kwargs

    run_threaded = False

    @property
    def computation_groups(self):
        """Back-compat alias used by these routing tests (== ``cpp_splits``)."""
        return self.cpp_splits

    def _cpp_strategy_class(self):
        return _StubFDComputationGroup

    def __init__(
        self,
        num_acs: int,
        num_splits: int,
        num_channels: int,
        num_freqs: int,
        df: float,
        seed: int = 0,
    ):
        if num_acs % num_splits != 0:
            raise ValueError(
                "Pick num_acs divisible by num_splits to keep the intra-split "
                "layout deterministic across tests."
            )

        rng = np.random.default_rng(seed)

        self.nchannels = num_channels
        self.acs_total_entries = int(num_acs)
        self.gpus = None
        self.xp = np
        self.df = df
        self.num_freqs = num_freqs

        # Active band = [df, num_freqs * df] on an (num_freqs + 1)-bin
        # rFFT grid, so N_active == num_freqs and the linear buffers
        # below line up with the settings the groups receive.
        f_min = df
        f_max = num_freqs * df
        self.settings = FDSettings(
            N=num_freqs + 1,
            df=df,
            min_freq=f_min,
            max_freq=f_max,
            force_backend="cpu",
        )
        assert self.settings.N_active == num_freqs

        split_num = int(np.ceil(num_acs / num_splits))
        split_inds = np.arange(split_num, num_acs, split_num)
        self.gpu_splits = np.split(np.arange(num_acs), split_inds)
        assert len(self.gpu_splits) == num_splits

        self.split_map = np.zeros(num_acs, dtype=int)
        for split_id, entries in enumerate(self.gpu_splits):
            self.split_map[entries] = split_id

        self.data = (
            rng.standard_normal((num_acs, num_channels, num_freqs))
            + 1j * rng.standard_normal((num_acs, num_channels, num_freqs))
        )
        # Hermitian positive-definite cross-channel invC per freq bin.
        # Shape ``(num_acs, nchannels, nchannels, nfreqs)`` — matches
        # the XYZ path the default ``tdi_type`` dispatches to.
        self.invC = np.zeros(
            (num_acs, num_channels, num_channels, num_freqs), dtype=np.complex128
        )
        for ac_id in range(num_acs):
            for f_idx in range(num_freqs):
                A = rng.standard_normal((num_channels, num_channels)) \
                    + 1j * rng.standard_normal((num_channels, num_channels))
                self.invC[ac_id, :, :, f_idx] = (
                    A @ A.conj().T + 3.0 * np.eye(num_channels)
                )

        self.linear_data_arr = []
        self.linear_psd_arr = []
        for entries in self.gpu_splits:
            self.linear_data_arr.append(
                np.concatenate([self.data[i].ravel() for i in entries])
            )
            self.linear_psd_arr.append(
                np.concatenate([self.invC[i].ravel() for i in entries])
            )

        d_d_vals = np.array(
            [
                _cross_inner(self.data[i], self.data[i], self.invC[i], df).real
                for i in range(num_acs)
            ]
        )
        self.d_d_reference = d_d_vals
        self.acs = np.asarray([_StubAC(v) for v in d_d_vals], dtype=object)

        # Routing table for the borrowed coordinator (ac_to_split = split_map).
        self.ac_to_intra = np.empty(num_acs, dtype=np.int32)
        for split_id, ids in enumerate(self.gpu_splits):
            self.ac_to_intra[ids] = np.arange(len(ids), dtype=np.int32)

        # Lazy-state attributes the borrowed coordinator reads/writes, then
        # build the per-split strategies eagerly (mirrors DCGA's auto-build at
        # construction, incl. the (d|d) snapshot the tests assert on).
        self._domain_group_kwargs = {}
        self._cpp_splits = None
        self._cpp_likelihood_backend = None
        self._thread_pool = None
        self._build_cpp_splits()


class _StubFDComputationGroup(BaseDomainComputationGroup):
    """NumPy stand-in for ``FDComputationGroup``.

    The base-class machinery (``extract_from_acs``,
    ``build_cpp_objects``, ``compute_d_d_term``,
    ``cpp_signal_likelihood``) runs unmodified — only the
    C++-kernel-backed ``compute_signal_likelihood_terms`` is replaced
    with the Python XYZ reference so the tests run on a CPU-only
    install (the merged ``FDComputationGroup`` requires the not-yet-
    reconciled STFT-era ``FDDomainWrap`` binding).
    """

    def compute_signal_likelihood_terms(
        self,
        data_index,
        noise_index,
        template_vals,
        start_freqs,
        start_times=None,
        **kwargs,
    ):
        num_binaries, nchan, nfreq = template_vals.shape
        data = self.data_arr.reshape(self.num_data, nchan, nfreq)
        invC = self.invC_arr.reshape(self.num_noise, nchan, nchan, nfreq)
        df = self.settings.df

        d_h = np.zeros(num_binaries, dtype=np.complex128)
        h_h = np.zeros(num_binaries, dtype=np.complex128)
        for b in range(num_binaries):
            d = data[int(data_index[b])]
            ic = invC[int(noise_index[b])]
            h = template_vals[b]
            d_h[b] = _cross_inner(d, h, ic, df)
            h_h[b] = _cross_inner(h, h, ic, df)
        return d_h, h_h


def _python_log_likelihood(data, invC, template, d_d, df):
    """Per-binary log-likelihood reference (XYZ full-cov, FD)."""
    d_h = _cross_inner(data, template, invC, df)
    h_h = _cross_inner(template, template, invC, df)
    return -0.5 * (d_d + h_h - 2.0 * d_h).real


def _route_batch(coord, data_index, noise_index, *flat_arrays):
    """Build the per-split routing + ``likelihood_args`` from flat arrays.

    Callers who have flat ``(template, start_freqs[, start_times])``
    tensors use this to reach the merged ``cpp_signal_likelihood``
    signature, which takes the ``unpack_indices`` output plus a
    ``list[tuple]`` of per-split likelihood args.
    """
    positions, data_intra, noise_intra = coord.unpack_indices(data_index, noise_index)
    likelihood_args = [tuple(a[pos] for a in flat_arrays) for pos in positions]
    return positions, data_intra, noise_intra, likelihood_args


# ---------------------------------------------------------------------------
# Routing tables
# ---------------------------------------------------------------------------


class TestRoutingTables(unittest.TestCase):
    """``ac_to_split`` / ``ac_to_intra`` must agree with ``gpu_splits``."""

    def test_single_split(self):
        acs = _StubACS(num_acs=3, num_splits=1, num_channels=3, num_freqs=32, df=1e-3)
        coord = acs

        np.testing.assert_array_equal(coord.ac_to_split, np.zeros(3, dtype=int))
        np.testing.assert_array_equal(coord.ac_to_intra, np.arange(3))
        assert coord.num_splits == 1

    def test_multi_split_layout(self):
        # 4 ACs split evenly across 2 splits -> split_map = [0,0,1,1],
        # intra indices reset inside each split.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=32, df=1e-3)
        coord = acs

        np.testing.assert_array_equal(coord.ac_to_split, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(coord.ac_to_intra, np.array([0, 1, 0, 1]))
        assert coord.num_splits == 2
        assert len(coord.computation_groups) == 2
        for group in coord.computation_groups:
            assert group.device is None  # CPU path
            assert group.num_data == 2


# ---------------------------------------------------------------------------
# unpack_indices
# ---------------------------------------------------------------------------


class TestUnpackIndices(unittest.TestCase):
    """Tempered-sampler-style flat inputs must partition into the right split."""

    def _tempered_index(self, nwalkers: int, ntemps: int) -> np.ndarray:
        # ``np.tile(np.arange(nwalkers), ntemps)`` reproduces the
        # [0,1,2,0,1,2,...] layout the move class builds before calling
        # the likelihood.
        return np.tile(np.arange(nwalkers), ntemps)

    def test_flat_routing(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        nwalkers, ntemps = 4, 3
        data_index = self._tempered_index(nwalkers, ntemps)
        noise_index = data_index.copy()

        positions, data_intra, noise_intra = coord.unpack_indices(
            data_index, noise_index
        )
        assert len(positions) == 2
        assert len(data_intra) == 2
        assert len(noise_intra) == 2

        # Split 0 holds AC ids {0, 1}; split 1 holds {2, 3}.
        expected_positions = [
            np.where(np.isin(data_index, [0, 1]))[0],
            np.where(np.isin(data_index, [2, 3]))[0],
        ]
        for pos, di, ni, expected in zip(
            positions, data_intra, noise_intra, expected_positions
        ):
            np.testing.assert_array_equal(pos, expected)
            expected_intra = np.array(
                [0 if data_index[p] in (0, 2) else 1 for p in pos],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(di, expected_intra)
            np.testing.assert_array_equal(ni, expected_intra)

    def test_empty_split(self):
        """A split with no matching binaries yields zero-length entries."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        # All binaries resolve to AC 0 -> split 0; split 1 must be empty.
        data_index = np.zeros(5, dtype=int)
        noise_index = data_index.copy()

        positions, data_intra, noise_intra = coord.unpack_indices(
            data_index, noise_index
        )
        assert len(positions[0]) == 5
        assert len(positions[1]) == 0
        assert len(data_intra[1]) == 0
        assert len(noise_intra[1]) == 0


# ---------------------------------------------------------------------------
# compute_d_d_terms
# ---------------------------------------------------------------------------


class TestComputeDdTerms(unittest.TestCase):
    def test_populates_per_split_d_d_at_init(self):
        # The merged coordinator computes (d|d) automatically inside
        # ``initialize_computation_groups`` — no explicit call needed.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        # Each group holds exactly the d_d values of its own ACs, in
        # intra-split order.
        for split_id, group in enumerate(coord.computation_groups):
            expected = acs.d_d_reference[acs.gpu_splits[split_id]]
            np.testing.assert_allclose(group.d_d, expected, rtol=1e-12)

        # An explicit re-run must reproduce the same values.
        coord.compute_d_d_terms()
        for split_id, group in enumerate(coord.computation_groups):
            expected = acs.d_d_reference[acs.gpu_splits[split_id]]
            np.testing.assert_allclose(group.d_d, expected, rtol=1e-12)

    def test_out_returns_per_split_list(self):
        # The merged API returns a list with one (num_data,) array per
        # split (previously a flat concatenated array). Concatenating
        # in split order recovers the global AC order because
        # ``gpu_splits`` are contiguous blocks.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        d_d_all = coord.compute_d_d_terms(out=True)
        assert isinstance(d_d_all, list)
        assert len(d_d_all) == coord.num_splits
        for split_id, d_d_split in enumerate(d_d_all):
            assert d_d_split.shape == (len(acs.gpu_splits[split_id]),)
        np.testing.assert_allclose(
            np.concatenate(d_d_all), acs.d_d_reference, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# cpp_signal_likelihood
# ---------------------------------------------------------------------------


class TestComputeSignalLikelihoodPlacement(unittest.TestCase):
    """End-to-end: per-binary log-likelihood routed across splits."""

    def _build_batch(self, acs, nwalkers, ntemps, seed=7):
        """Tempered flat batch mimicking the move's data_index pattern."""
        rng = np.random.default_rng(seed)
        data_index = np.tile(np.arange(nwalkers), ntemps).astype(np.int32)
        noise_index = data_index.copy()
        batch = data_index.shape[0]
        template = (
            rng.standard_normal((batch, acs.nchannels, acs.num_freqs))
            + 1j * rng.standard_normal((batch, acs.nchannels, acs.num_freqs))
        )
        start_freqs = np.full(batch, acs.df, dtype=np.float64)
        return data_index, noise_index, template, start_freqs

    def _reference_likes(self, acs, data_index, noise_index, template):
        return np.array(
            [
                _python_log_likelihood(
                    acs.data[int(data_index[b])],
                    acs.invC[int(noise_index[b])],
                    template[b],
                    acs.d_d_reference[int(data_index[b])],
                    acs.df,
                )
                for b in range(data_index.shape[0])
            ]
        )

    def test_single_split_matches_reference(self):
        acs = _StubACS(num_acs=4, num_splits=1, num_channels=3, num_freqs=64, df=1e-3)
        coord = acs

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs, nwalkers=4, ntemps=3
        )
        positions, data_intra, noise_intra, likelihood_args = _route_batch(
            coord, data_index, noise_index, template, start_freqs
        )
        likes = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args
        )
        expected = self._reference_likes(acs, data_index, noise_index, template)

        np.testing.assert_allclose(likes, expected, rtol=1e-10)

    def test_multi_split_matches_reference_and_preserves_order(self):
        # Two splits over 4 ACs -> split_map = [0,0,1,1]. A tempered
        # flat batch puts some binaries on each split; the return must
        # still be in the original flat order.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = acs

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs, nwalkers=4, ntemps=3
        )
        positions, data_intra, noise_intra, likelihood_args = _route_batch(
            coord, data_index, noise_index, template, start_freqs
        )
        likes = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args
        )
        expected = self._reference_likes(acs, data_index, noise_index, template)

        np.testing.assert_allclose(likes, expected, rtol=1e-10)

    def test_empty_split_no_crash(self):
        """All binaries on split 0; split 1 must be skipped, not crash."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = acs

        rng = np.random.default_rng(11)
        # AC 0 -> split 0. Every binary lives on split 0.
        data_index = np.zeros(6, dtype=np.int32)
        noise_index = data_index.copy()
        template = (
            rng.standard_normal((6, acs.nchannels, acs.num_freqs))
            + 1j * rng.standard_normal((6, acs.nchannels, acs.num_freqs))
        )
        start_freqs = np.full(6, acs.df, dtype=np.float64)

        positions, data_intra, noise_intra, likelihood_args = _route_batch(
            coord, data_index, noise_index, template, start_freqs
        )
        # Split 1 must carry empty-length args tuples — never invoked
        # thanks to the ``positions_per_split`` short-circuit.
        assert all(len(a) == 0 for a in likelihood_args[1])

        likes = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args
        )
        expected = np.array(
            [
                _python_log_likelihood(
                    acs.data[int(data_index[b])],
                    acs.invC[int(noise_index[b])],
                    template[b],
                    acs.d_d_reference[int(data_index[b])],
                    acs.df,
                )
                for b in range(data_index.shape[0])
            ]
        )
        np.testing.assert_allclose(likes, expected, rtol=1e-10)

    def test_split_layout_invariance(self):
        """Same batch, same seed, 1-split vs 2-splits must yield bit-equal likes."""
        kwargs = dict(num_acs=4, num_channels=3, num_freqs=64, df=1e-3, seed=1234)
        acs1 = _StubACS(num_splits=1, **kwargs)
        acs2 = _StubACS(num_splits=2, **kwargs)

        coord1 = acs1
        coord2 = acs2

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs1, nwalkers=4, ntemps=3
        )

        route1 = _route_batch(coord1, data_index, noise_index, template, start_freqs)
        route2 = _route_batch(coord2, data_index, noise_index, template, start_freqs)
        likes1 = coord1.cpp_signal_likelihood(*route1[:3], route1[3])
        likes2 = coord2.cpp_signal_likelihood(*route2[:3], route2[3])
        np.testing.assert_allclose(likes1, likes2, rtol=1e-12)

    def test_run_threaded_matches_serial(self):
        """``run_threaded=True`` (ThreadPoolExecutor dispatch) is now a
        supported mode — it must reproduce the serial path exactly.

        (Replaces the pre-merge ``mode='threaded'`` NotImplementedError
        guard: the merged library implements threaded dispatch.)
        """
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=32, df=1e-3)
        coord = acs

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs, nwalkers=4, ntemps=2
        )
        positions, data_intra, noise_intra, likelihood_args = _route_batch(
            coord, data_index, noise_index, template, start_freqs
        )

        likes_serial = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args, run_threaded=False
        )
        likes_threaded = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args, run_threaded=True
        )
        np.testing.assert_array_equal(likes_serial, likes_threaded)


# ---------------------------------------------------------------------------
# _loop_operation callable path
# ---------------------------------------------------------------------------


class TestLoopOperationCallable(unittest.TestCase):
    """Bound-method dispatch is covered by compute_d_d_terms /
    cpp_signal_likelihood; here we verify the generic callable path
    used for external per-device work (waveform generation, etc.)."""

    def test_callable_invoked_per_group_with_args(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        calls = []

        def op(x, *, tag):
            calls.append((x, tag))
            return x * 10

        outputs = coord._loop_operation(
            op,
            operation_args_per_split=[(1,), (2,)],
            operation_kwargs=[{"tag": "a"}, {"tag": "b"}],
        )

        assert outputs == [10, 20]
        assert calls == [(1, "a"), (2, "b")]

    def test_callable_defaults_empty_args_kwargs(self):
        acs = _StubACS(num_acs=6, num_splits=3, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        counter = {"n": 0}

        def op():
            counter["n"] += 1
            return counter["n"]

        outs = coord._loop_operation(op)
        assert outs == [1, 2, 3]

    def test_args_length_mismatch_raises(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        with self.assertRaisesRegex(ValueError, "must match the number of splits"):
            coord._loop_operation(lambda x: x, operation_args_per_split=[(1,)])

    def test_positions_short_circuit_skips_empty_split(self):
        """Empty splits yield ``None`` and never invoke the operation."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        calls = []

        def wf(coords_s):
            calls.append(coords_s.shape[0])
            return np.ones((coords_s.shape[0], acs.nchannels, acs.num_freqs))

        data_index = np.zeros(3, dtype=np.int32)
        coords = np.ones((3, 2), dtype=np.float64)

        positions, *_ = coord.unpack_indices(data_index, data_index)
        args_per_split = coord.unpack_coords(positions, coords, keep_tuple=True)
        per_split = coord._loop_operation(
            wf,
            operation_args_per_split=args_per_split,
            positions_per_split=positions,
        )

        assert per_split[0] is not None
        assert per_split[0].shape == (3, acs.nchannels, acs.num_freqs)
        assert per_split[1] is None
        # Exactly one invocation — the short-circuit skipped split 1.
        assert calls == [3]


# ---------------------------------------------------------------------------
# unpack_coords
# ---------------------------------------------------------------------------


class TestUnpackCoords(unittest.TestCase):
    """Per-split coordinate slicing (the merged replacement for the old
    ``_generate_waveform`` input plumbing — DCGA no longer owns waveform
    generation; callers compose ``unpack_indices`` + ``unpack_coords``
    + ``_loop_operation``)."""

    def test_single_array_sliced_per_split(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        data_index = np.array([0, 3, 1, 2, 0, 2], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64) + 0.5,
             np.arange(data_index.shape[0], dtype=np.float64) * 0.1]
        )

        positions, *_ = coord.unpack_indices(data_index, data_index)
        args_per_group = coord.unpack_coords(positions, coords)

        assert len(args_per_group) == coord.num_splits
        for split_id, positions_s in enumerate(positions):
            out = args_per_group[split_id]
            assert isinstance(out, np.ndarray)
            np.testing.assert_array_equal(out, coords[positions_s])

    def test_tuple_coords_per_split(self):
        """Multi-branch coordinate tuples stay tuples per split."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        data_index = np.array([0, 3, 2, 1, 0], dtype=np.int32)
        c0 = np.arange(data_index.shape[0], dtype=np.float64)
        c1 = np.arange(data_index.shape[0], dtype=np.float64) * 2.0

        positions, *_ = coord.unpack_indices(data_index, data_index)
        args_per_group = coord.unpack_coords(positions, (c0, c1))

        for split_id, positions_s in enumerate(positions):
            out = args_per_group[split_id]
            assert isinstance(out, tuple)
            assert len(out) == 2
            np.testing.assert_array_equal(out[0], c0[positions_s])
            np.testing.assert_array_equal(out[1], c1[positions_s])

    def test_empty_split_yields_empty_tuple(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        data_index = np.zeros(3, dtype=np.int32)  # all on split 0
        coords = np.ones((3, 2), dtype=np.float64)

        positions, *_ = coord.unpack_indices(data_index, data_index)
        args_per_group = coord.unpack_coords(positions, coords)

        assert isinstance(args_per_group[0], np.ndarray)
        assert args_per_group[1] == ()

    def test_keep_tuple_wraps_single_array(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        data_index = np.array([0, 2], dtype=np.int32)
        coords = np.arange(2, dtype=np.float64)

        positions, *_ = coord.unpack_indices(data_index, data_index)
        args_per_group = coord.unpack_coords(positions, coords, keep_tuple=True)

        for split_id, positions_s in enumerate(positions):
            out = args_per_group[split_id]
            assert isinstance(out, tuple)
            assert len(out) == 1
            np.testing.assert_array_equal(out[0], coords[positions_s])


# ---------------------------------------------------------------------------
# place_on_device
# ---------------------------------------------------------------------------


class TestPlaceOnDevice(unittest.TestCase):
    def test_cpu_round_trip_arrays_and_tuples(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = acs

        arrays_per_split = [np.arange(3, dtype=np.float64), np.arange(2.0)]
        tuples_per_split = [(np.ones(2), np.zeros(2)), ()]

        (arrays_out, tuples_out) = coord.place_on_device(
            (arrays_per_split, tuples_per_split)
        )

        assert len(arrays_out) == 2 and len(tuples_out) == 2
        for orig, placed in zip(arrays_per_split, arrays_out):
            np.testing.assert_array_equal(placed, orig)
            assert placed is not orig  # copy=True
        assert tuples_out[1] == ()
        for orig, placed in zip(tuples_per_split[0], tuples_out[0]):
            np.testing.assert_array_equal(placed, orig)


# ---------------------------------------------------------------------------
# Composed coords -> waveform -> likelihood pipeline
# ---------------------------------------------------------------------------


class TestComposedLikelihoodFromCoords(unittest.TestCase):
    """The pre-merge ``compute_likelihood_from_coords`` convenience was
    removed; callers now compose ``unpack_indices`` + ``unpack_coords``
    + ``_loop_operation`` + ``cpp_signal_likelihood``. Verify the
    composition end to end against the per-binary reference."""

    def test_matches_per_binary_reference(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = acs

        rng = np.random.default_rng(42)
        data_index = np.array([0, 1, 2, 3, 0, 2], dtype=np.int32)
        coords = rng.standard_normal((data_index.shape[0], 2))

        nchan, nfreq = acs.nchannels, acs.num_freqs

        def wf(c0, c1):
            k = c0.shape[0]
            tmpl = (c0[:, None, None] + 1j * c1[:, None, None]) * np.ones(
                (k, nchan, nfreq), dtype=np.complex128
            )
            return tmpl, np.full(k, acs.df)

        positions, data_intra, noise_intra = coord.unpack_indices(
            data_index, data_index
        )
        coord_args = coord.unpack_coords(
            positions, (coords[:, 0], coords[:, 1])
        )
        wf_out_per_split = coord._loop_operation(
            wf,
            operation_args_per_split=coord_args,
            positions_per_split=positions,
        )
        likelihood_args = [
            out if out is not None else () for out in wf_out_per_split
        ]
        likes = coord.cpp_signal_likelihood(
            positions, data_intra, noise_intra, likelihood_args
        )

        # Per-binary python reference on the flat batch.
        flat_vals, _ = wf(*coords.T)
        expected = np.array(
            [
                _python_log_likelihood(
                    acs.data[int(data_index[b])],
                    acs.invC[int(data_index[b])],
                    flat_vals[b],
                    acs.d_d_reference[int(data_index[b])],
                    acs.df,
                )
                for b in range(data_index.shape[0])
            ]
        )
        np.testing.assert_allclose(likes, expected, rtol=1e-10)
