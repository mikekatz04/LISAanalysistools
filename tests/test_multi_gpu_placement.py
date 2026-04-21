"""Placement + routing tests for :class:`DomainComputationGroupArray`.

These tests exercise the multi-split routing logic introduced by the
multi-GPU refactor *without* requiring any real GPUs. We drive the
coordinator with a lightweight ``AnalysisContainerArray`` shim that
exposes exactly the attributes the coordinator and each
``BaseDomainComputationGroup.extract_from_acs`` path read.

Key properties verified:
    * ``ac_to_split`` / ``ac_to_intra`` routing tables match
      ``gpu_splits``.
    * ``_unpack_indices`` partitions flat ``(data_index, noise_index)``
      batches by split into ``(positions, data_intra, noise_intra)``.
    * ``compute_d_d_terms`` populates each group's per-split ``d_d``
      vector in intra-split order.
    * ``compute_likelihood`` reproduces the per-binary reference and is
      invariant to the split layout (single split vs. two splits).
    * ``mode='threaded'`` is reserved and raises ``NotImplementedError``.
    * ``_generate_waveform`` keeps per-split outputs opaque — FD tuples,
      STFT tuples, or any other waveform return shape.
    * ``compute_likelihood_from_coords`` composes generation + scatter.
    * ``generate_signals_for_residuals`` routes
      ``get_signals_for_residuals`` per-split without centralizing.

All work runs on the CPU backend with ``gpus=None``; the multi-split
dimension is controlled entirely via ``gpu_splits``. This is sufficient
to exercise the routing/scatter-back code paths — the actual C++
likelihood kernels are already covered by
``tests/test_domain_likelihood.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lisatools.domaincomputation import DomainComputationGroupArray
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


class _StubAC:
    """Fake ``AnalysisContainer`` exposing only ``inner_product``.

    ``BaseDomainComputationGroup.compute_d_d_term`` loops over the
    split ACs and writes ``ac.inner_product()`` into its ``d_d`` array.
    Returning a precomputed scalar lets the test assert the full
    coordinator-level ``compute_d_d_terms`` wiring without needing to
    build a real ``DataResidualArray`` + ``SensitivityMatrix`` stack.
    """

    def __init__(self, d_d_value: float):
        self._d_d = float(d_d_value)

    def inner_product(self, **_):
        return self._d_d


class _StubACS:
    """Minimal ``AnalysisContainerArray`` shim for routing tests.

    The coordinator reads ``settings``, ``gpus``, ``xp``,
    ``acs_total_entries``, ``gpu_splits``, ``split_map``,
    ``linear_data_arr``, ``linear_psd_arr``, ``nchannels``, and the
    object array ``acs``. We provide all of those and nothing else.

    The shim is intentionally CPU-only (``gpus=None``) so that every
    split's ``device_id`` is ``None`` and the
    ``force_backend='cpu'`` assertion in ``extract_from_acs`` holds
    regardless of split count.
    """

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

        f_min = df
        f_max = f_min + (num_freqs - 1) * df
        self.settings = FDSettings(
            N=num_freqs,
            df=df,
            min_freq=f_min,
            max_freq=f_max,
            force_backend="cpu",
        )

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


def _python_log_likelihood(data, invC, template, d_d, df):
    """Per-binary log-likelihood reference (XYZ full-cov, FD)."""
    d_h = _cross_inner(data, template, invC, df)
    h_h = _cross_inner(template, template, invC, df)
    return -0.5 * (d_d + h_h - 2.0 * d_h).real


# ---------------------------------------------------------------------------
# Routing tables
# ---------------------------------------------------------------------------


class TestRoutingTables:
    """``ac_to_split`` / ``ac_to_intra`` must agree with ``gpu_splits``."""

    def test_single_split(self):
        acs = _StubACS(num_acs=3, num_splits=1, num_channels=3, num_freqs=32, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        np.testing.assert_array_equal(coord.ac_to_split, np.zeros(3, dtype=int))
        np.testing.assert_array_equal(coord.ac_to_intra, np.arange(3))
        assert coord.num_splits == 1

    def test_multi_split_layout(self):
        # 4 ACs split evenly across 2 splits -> split_map = [0,0,1,1],
        # intra indices reset inside each split.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=32, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        np.testing.assert_array_equal(coord.ac_to_split, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(coord.ac_to_intra, np.array([0, 1, 0, 1]))
        assert coord.num_splits == 2
        assert len(coord.computation_groups) == 2
        for group in coord.computation_groups:
            assert group.device_id is None  # CPU path
            assert group.num_data == 2


# ---------------------------------------------------------------------------
# _unpack_indices
# ---------------------------------------------------------------------------


def _split_flat_args(coord, data_index, noise_index, *flat_arrays):
    """Build the per-split ``likelihood_args`` list from flat arrays.

    Callers who already have flat ``(template, start_freqs[, start_times])``
    tensors use this to reach the new ``compute_likelihood`` signature, which
    takes ``list[tuple]`` per split rather than flat inputs.
    """
    positions, *_ = coord._unpack_indices(data_index, noise_index)
    return [tuple(a[pos] for a in flat_arrays) for pos in positions]


class TestUnpackIndices:
    """Tempered-sampler-style flat inputs must partition into the right split."""

    def _tempered_index(self, nwalkers: int, ntemps: int) -> np.ndarray:
        # ``np.tile(np.arange(nwalkers), ntemps)`` reproduces the
        # [0,1,2,0,1,2,...] layout the move class builds before calling
        # the likelihood.
        return np.tile(np.arange(nwalkers), ntemps)

    def test_flat_routing(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        nwalkers, ntemps = 4, 3
        data_index = self._tempered_index(nwalkers, ntemps)
        noise_index = data_index.copy()

        positions, data_intra, noise_intra = coord._unpack_indices(
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
        coord = DomainComputationGroupArray(acs)

        # All binaries resolve to AC 0 -> split 0; split 1 must be empty.
        data_index = np.zeros(5, dtype=int)
        noise_index = data_index.copy()

        positions, data_intra, noise_intra = coord._unpack_indices(
            data_index, noise_index
        )
        assert len(positions[0]) == 5
        assert len(positions[1]) == 0
        assert len(data_intra[1]) == 0
        assert len(noise_intra[1]) == 0


# ---------------------------------------------------------------------------
# compute_d_d_terms
# ---------------------------------------------------------------------------


class TestComputeDdTerms:
    def test_populates_per_split_d_d(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        coord.compute_d_d_terms()

        # Each group holds exactly the d_d values of its own ACs, in
        # intra-split order.
        for split_id, group in enumerate(coord.computation_groups):
            expected = acs.d_d_reference[acs.gpu_splits[split_id]]
            np.testing.assert_allclose(group.d_d, expected, rtol=1e-12)

    def test_out_returns_concatenated(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        d_d_all = coord.compute_d_d_terms(out=True)
        np.testing.assert_allclose(d_d_all, acs.d_d_reference, rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_likelihood
# ---------------------------------------------------------------------------


class TestComputeLikelihoodPlacement:
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
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs, nwalkers=4, ntemps=3
        )
        likelihood_args = _split_flat_args(
            coord, data_index, noise_index, template, start_freqs
        )
        likes = coord.compute_likelihood(data_index, noise_index, likelihood_args)
        expected = self._reference_likes(acs, data_index, noise_index, template)

        np.testing.assert_allclose(likes, expected, rtol=1e-10)

    def test_multi_split_matches_reference_and_preserves_order(self):
        # Two splits over 4 ACs -> split_map = [0,0,1,1]. A tempered
        # flat batch puts some binaries on each split; the return must
        # still be in the original flat order.
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs, nwalkers=4, ntemps=3
        )
        likelihood_args = _split_flat_args(
            coord, data_index, noise_index, template, start_freqs
        )
        likes = coord.compute_likelihood(data_index, noise_index, likelihood_args)
        expected = self._reference_likes(acs, data_index, noise_index, template)

        np.testing.assert_allclose(likes, expected, rtol=1e-10)

    def test_empty_split_no_crash(self):
        """All binaries on split 0; split 1 must be skipped, not crash."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        rng = np.random.default_rng(11)
        # AC 0 -> split 0. Every binary lives on split 0.
        data_index = np.zeros(6, dtype=np.int32)
        noise_index = data_index.copy()
        template = (
            rng.standard_normal((6, acs.nchannels, acs.num_freqs))
            + 1j * rng.standard_normal((6, acs.nchannels, acs.num_freqs))
        )
        start_freqs = np.full(6, acs.df, dtype=np.float64)

        likelihood_args = _split_flat_args(
            coord, data_index, noise_index, template, start_freqs
        )
        # Split 1 must carry empty-length args tuples — never invoked.
        assert all(len(a) == 0 for a in likelihood_args[1])

        likes = coord.compute_likelihood(data_index, noise_index, likelihood_args)
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

        coord1 = DomainComputationGroupArray(acs1)
        coord2 = DomainComputationGroupArray(acs2)
        coord1.compute_d_d_terms()
        coord2.compute_d_d_terms()

        data_index, noise_index, template, start_freqs = self._build_batch(
            acs1, nwalkers=4, ntemps=3
        )

        args1 = _split_flat_args(coord1, data_index, noise_index, template, start_freqs)
        args2 = _split_flat_args(coord2, data_index, noise_index, template, start_freqs)
        likes1 = coord1.compute_likelihood(data_index, noise_index, args1)
        likes2 = coord2.compute_likelihood(data_index, noise_index, args2)
        np.testing.assert_allclose(likes1, likes2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Threaded mode guard
# ---------------------------------------------------------------------------


class TestThreadedModeGuard:
    def _args_for(self, acs, coord, data_index):
        template = np.zeros(
            (data_index.shape[0], acs.nchannels, acs.num_freqs),
            dtype=np.complex128,
        )
        start_freqs = np.full(data_index.shape[0], acs.df)
        return _split_flat_args(
            coord, data_index, data_index, template, start_freqs
        )

    def test_threaded_raises(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index = np.array([0, 1], dtype=np.int32)
        likelihood_args = self._args_for(acs, coord, data_index)

        with pytest.raises(NotImplementedError, match="threaded"):
            coord.compute_likelihood(
                data_index, data_index, likelihood_args, mode="threaded"
            )

    def test_unknown_mode_raises(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index = np.array([0, 1], dtype=np.int32)
        likelihood_args = self._args_for(acs, coord, data_index)

        with pytest.raises(ValueError, match="Unknown mode"):
            coord.compute_likelihood(
                data_index, data_index, likelihood_args, mode="nope"
            )


# ---------------------------------------------------------------------------
# _loop_operation callable path
# ---------------------------------------------------------------------------


class TestLoopOperationCallable:
    """String-form dispatch is covered by compute_d_d_terms/compute_likelihood;
    here we verify the callable-form path added for external per-device work
    (waveform generation, etc.)."""

    def test_callable_invoked_per_group_with_args(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        calls = []

        def op(x, *, tag):
            calls.append((x, tag))
            return x * 10

        outputs = coord._loop_operation(
            op, args_per_group=[(1,), (2,)], kwargs_per_group=[{"tag": "a"}, {"tag": "b"}]
        )

        assert outputs == [10, 20]
        assert calls == [(1, "a"), (2, "b")]

    def test_callable_defaults_empty_args_kwargs(self):
        acs = _StubACS(num_acs=6, num_splits=3, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        counter = {"n": 0}

        def op():
            counter["n"] += 1
            return counter["n"]

        outs = coord._loop_operation(op)
        assert outs == [1, 2, 3]


# ---------------------------------------------------------------------------
# _generate_waveform
# ---------------------------------------------------------------------------


class TestGenerateWaveform:
    """Opaque per-split dispatcher: DCGA does not inspect the return shape."""

    def _fd_stub(self, nchan, nfreq, df):
        def wf(*cols, amp=1.0):
            coords = np.stack(cols, axis=-1)
            k = coords.shape[0]
            base = coords[:, 0:1, None].astype(np.complex128)  # (k, 1, 1)
            vals = amp * np.broadcast_to(base, (k, nchan, nfreq)).copy()
            sfreqs = np.full(k, df)
            return vals, sfreqs
        return wf

    def test_fd_tuple_per_split(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        wf = self._fd_stub(acs.nchannels, acs.num_freqs, acs.df)

        data_index = np.array([0, 3, 1, 2, 0, 2], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64) + 0.5,
             np.arange(data_index.shape[0], dtype=np.float64) * 0.1]
        )

        positions, *_ = coord._unpack_indices(data_index, data_index)
        per_split = coord._generate_waveform(wf, coords, positions, {"amp": 2.0})

        assert len(per_split) == coord.num_splits
        for split_id, positions_s in enumerate(positions):
            out = per_split[split_id]
            assert out is not None
            vals, sfreqs = out
            expected_vals, expected_sfreqs = wf(
                *coords[positions_s].T, amp=2.0
            )
            np.testing.assert_allclose(vals, expected_vals)
            np.testing.assert_allclose(sfreqs, expected_sfreqs)

    def test_stft_tuple_arity_transparent(self):
        """STFT returns 3-tuples; DCGA must not inspect arity."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        nchan, nfreq = acs.nchannels, acs.num_freqs

        def wf(*cols):
            coords = np.stack(cols, axis=-1)
            k = coords.shape[0]
            vals = np.zeros((k, nchan, 2, nfreq), dtype=np.complex128)
            sfreqs = np.full(k, acs.df)
            stimes = np.full(k, 42.0)
            return vals, sfreqs, stimes

        data_index = np.array([0, 3, 2, 1, 0], dtype=np.int32)
        coords = np.arange(data_index.shape[0] * 3, dtype=np.float64).reshape(-1, 3)

        positions, *_ = coord._unpack_indices(data_index, data_index)
        per_split = coord._generate_waveform(wf, coords, positions, None)

        for split_id, positions_s in enumerate(positions):
            k = len(positions_s)
            if k == 0:
                assert per_split[split_id] is None
                continue
            vals, sfreqs, stimes = per_split[split_id]
            assert vals.shape == (k, nchan, 2, nfreq)
            np.testing.assert_array_equal(stimes, np.full(k, 42.0))

    def test_empty_split_returns_none_and_short_circuits(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        calls = []

        def wf(*cols):
            coords = np.stack(cols, axis=-1)
            k = coords.shape[0]
            calls.append(k)
            return (
                np.zeros((k, acs.nchannels, acs.num_freqs), dtype=np.complex128),
                np.full(k, acs.df),
            )

        data_index = np.zeros(3, dtype=np.int32)
        coords = np.ones((3, 2), dtype=np.float64)

        positions, *_ = coord._unpack_indices(data_index, data_index)
        per_split = coord._generate_waveform(wf, coords, positions, None)

        assert per_split[0] is not None
        assert per_split[1] is None
        # Exactly one invocation — the short-circuit wrapper skipped split 1.
        assert calls == [3]


# ---------------------------------------------------------------------------
# compute_likelihood_from_coords
# ---------------------------------------------------------------------------


class TestComputeLikelihoodFromCoords:
    def test_matches_per_binary_reference(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=64, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

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

        likes_auto = coord.compute_likelihood_from_coords(wf, coords, data_index)

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
        np.testing.assert_allclose(likes_auto, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# generate_signals_for_residuals
# ---------------------------------------------------------------------------


class TestGenerateSignalsForResiduals:
    """DCGA routes ``get_signals_for_residuals`` per-split without centralizing."""

    def test_routes_method_per_split_and_returns_positions(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        # Stub waveform object with a `get_signals_for_residuals` method;
        # DCGA must call the method (not __call__).
        called_via = []

        class Stub:
            def __call__(self_stub, *cols, **_):
                called_via.append("__call__")
                return ("via_call", np.stack(cols, axis=-1))

            def get_signals_for_residuals(self_stub, *cols, **_):
                called_via.append("get_signals_for_residuals")
                # Return a marker per split so the caller can verify
                # per-split dispatch.
                return ("signals_for_split", np.stack(cols, axis=-1))

        stub = Stub()
        data_index = np.array([0, 3, 1, 2, 0], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64),
             np.zeros(data_index.shape[0], dtype=np.float64)]
        )

        signals_per_split, positions_per_split = coord.generate_signals_for_residuals(
            stub, coords, data_index
        )

        # DCGA must route through get_signals_for_residuals on each
        # non-empty split — never through __call__.
        assert set(called_via) == {"get_signals_for_residuals"}
        assert called_via.count("get_signals_for_residuals") == coord.num_splits

        # Positions partition the flat batch.
        all_positions = np.concatenate(positions_per_split)
        np.testing.assert_array_equal(
            np.sort(all_positions), np.arange(data_index.shape[0])
        )

        # Each non-empty split's signals encode that split's coord slice.
        for split_id, positions in enumerate(positions_per_split):
            tag, encoded = signals_per_split[split_id]
            assert tag == "signals_for_split"
            np.testing.assert_array_equal(encoded, coords[positions])


# ---------------------------------------------------------------------------
# warm_jax_compile via (coords, data_index)
# ---------------------------------------------------------------------------


