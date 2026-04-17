"""Placement + routing tests for :class:`DomainComputationGroupArray`.

These tests exercise the multi-split routing logic introduced by the
multi-GPU refactor *without* requiring any real GPUs. We drive the
coordinator with a lightweight ``AnalysisContainerArray`` shim that
exposes exactly the attributes the coordinator and each
``BaseDomainComputationGroup.extract_from_acs`` path read.

Key properties verified:
    * ``ac_to_split`` / ``ac_to_intra`` routing tables match
      ``gpu_splits``.
    * ``_unpack_input_args`` handles tempered-sampler flat inputs and
      the pre-split ``dict`` form identically.
    * ``compute_d_d_terms`` populates each group's per-split ``d_d``
      vector in intra-split order.
    * ``compute_likelihood`` reproduces the per-binary reference and is
      invariant to the split layout (single split vs. two splits).
    * ``mode='threaded'`` is reserved and raises ``NotImplementedError``.

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
# _unpack_input_args
# ---------------------------------------------------------------------------


class TestUnpackInputArgs:
    """Tempered-sampler-style flat inputs must route to the right split."""

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
        batch = data_index.shape[0]

        template = np.zeros((batch, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        start_freqs = np.full(batch, acs.df)

        per_group = coord._unpack_input_args(
            data_index, noise_index, template, start_freqs
        )
        assert len(per_group) == 2

        # Split 0 holds AC ids {0, 1}; split 1 holds {2, 3}.
        expected_positions = [
            np.where(np.isin(data_index, [0, 1]))[0],
            np.where(np.isin(data_index, [2, 3]))[0],
        ]
        for routing, expected in zip(per_group, expected_positions):
            np.testing.assert_array_equal(routing["positions"], expected)
            # intra indices are [0,1] within each split (matching split order).
            expected_intra = np.array(
                [0 if data_index[p] in (0, 2) else 1 for p in routing["positions"]],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(routing["intra_data_index"], expected_intra)
            np.testing.assert_array_equal(routing["intra_noise_index"], expected_intra)
            # Template / start_freqs slices are aligned with positions.
            np.testing.assert_array_equal(
                routing["template_vals"], template[routing["positions"]]
            )
            np.testing.assert_array_equal(
                routing["start_freqs"], start_freqs[routing["positions"]]
            )
            assert routing["start_times"] is None

    def test_dict_input_passthrough(self):
        """Pre-split dict form bypasses per-call slicing."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        data_index = np.array([0, 1, 2, 3])
        noise_index = data_index.copy()

        # Callers that generated templates on each target device pass a
        # dict keyed by split id; _unpack_input_args must return those
        # exact objects (no copy / re-slice).
        t0 = np.ones((2, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        t1 = 2.0 * np.ones((2, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        template_dict = {0: t0, 1: t1}
        freqs_dict = {0: np.array([acs.df, acs.df]), 1: np.array([acs.df, acs.df])}

        per_group = coord._unpack_input_args(
            data_index, noise_index, template_dict, freqs_dict
        )

        assert per_group[0]["template_vals"] is t0
        assert per_group[1]["template_vals"] is t1
        assert per_group[0]["start_freqs"] is freqs_dict[0]
        assert per_group[1]["start_freqs"] is freqs_dict[1]

    def test_empty_split(self):
        """A split with no matching binaries produces an empty routing record."""
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        # All binaries resolve to AC 0 -> split 0; split 1 must be empty.
        data_index = np.zeros(5, dtype=int)
        noise_index = data_index.copy()
        template = np.zeros((5, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        start_freqs = np.full(5, acs.df)

        per_group = coord._unpack_input_args(
            data_index, noise_index, template, start_freqs
        )
        assert len(per_group[0]["positions"]) == 5
        assert len(per_group[1]["positions"]) == 0
        # The empty-split slices are ``None`` so the caller can skip
        # without materializing zero-length device arrays.
        assert per_group[1]["template_vals"] is None
        assert per_group[1]["start_freqs"] is None


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
        likes = coord.compute_likelihood(
            data_index, noise_index, template, start_freqs
        )
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
        likes = coord.compute_likelihood(
            data_index, noise_index, template, start_freqs
        )
        expected = self._reference_likes(acs, data_index, noise_index, template)

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

        likes1 = coord1.compute_likelihood(
            data_index, noise_index, template, start_freqs
        )
        likes2 = coord2.compute_likelihood(
            data_index, noise_index, template, start_freqs
        )
        np.testing.assert_allclose(likes1, likes2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Threaded mode guard
# ---------------------------------------------------------------------------


class TestThreadedModeGuard:
    def test_threaded_raises(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index = np.array([0, 1], dtype=np.int32)
        template = np.zeros((2, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        start_freqs = np.full(2, acs.df)

        with pytest.raises(NotImplementedError, match="threaded"):
            coord.compute_likelihood(
                data_index, data_index, template, start_freqs, mode="threaded"
            )

    def test_unknown_mode_raises(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord.compute_d_d_terms()

        data_index = np.array([0, 1], dtype=np.int32)
        template = np.zeros((2, acs.nchannels, acs.num_freqs), dtype=np.complex128)
        start_freqs = np.full(2, acs.df)

        with pytest.raises(ValueError, match="Unknown mode"):
            coord.compute_likelihood(
                data_index, data_index, template, start_freqs, mode="nope"
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
# _generate_per_split
# ---------------------------------------------------------------------------


class TestGeneratePerSplit:
    def _fd_waveform_stub(self, nchan, nfreq, df, amp_capture=None):
        """Return a deterministic FD stub waveform_gen.

        Encodes the coord's first column into the template so the test
        can verify scatter per-split.
        """
        def wf(*cols, amp=1.0):
            coords = np.stack(cols, axis=-1)
            k = coords.shape[0]
            base = coords[:, 0:1, None].astype(np.complex128)
            vals = amp * np.broadcast_to(base[..., None], (k, nchan, nfreq)).copy()
            sfreqs = np.full(k, df)
            if amp_capture is not None:
                amp_capture.append(amp)
            return vals, sfreqs

        return wf

    def test_fd_partitioning_and_positions(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        wf = self._fd_waveform_stub(acs.nchannels, acs.num_freqs, acs.df)

        data_index = np.array([0, 3, 1, 2, 0, 2], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64) + 0.5,
             np.arange(data_index.shape[0], dtype=np.float64) * 0.1]
        )

        per_split = coord._generate_per_split(wf, coords, data_index, {"amp": 2.0})

        split_of_each = acs.split_map[data_index]
        assert per_split["start_times"] is None
        for split_id in range(coord.num_splits):
            expected_positions = np.where(split_of_each == split_id)[0]
            np.testing.assert_array_equal(
                per_split["positions"][split_id], expected_positions
            )
            expected_vals, expected_sfreqs = wf(
                *coords[expected_positions].T, amp=2.0
            )
            np.testing.assert_allclose(
                per_split["templates"][split_id], expected_vals
            )
            np.testing.assert_allclose(
                per_split["start_freqs"][split_id], expected_sfreqs
            )

    def test_stft_domain_unpacks_start_times(self, monkeypatch):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        # Only the unpacking branch depends on domain_type; override the
        # property to exercise the STFT path without an STFT kernel.
        monkeypatch.setattr(
            type(coord), "domain_type", property(lambda self: "STFT")
        )

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

        per_split = coord._generate_per_split(wf, coords, data_index, None)

        assert per_split["start_times"] is not None
        for split_id, positions in per_split["positions"].items():
            k = len(positions)
            assert per_split["templates"][split_id].shape == (k, nchan, 2, nfreq)
            np.testing.assert_array_equal(
                per_split["start_times"][split_id], np.full(k, 42.0)
            )

    def test_empty_split_skipped(self):
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

        # Every binary lives on split 0; split 1 has no data.
        data_index = np.zeros(3, dtype=np.int32)
        coords = np.ones((3, 2), dtype=np.float64)

        per_split = coord._generate_per_split(wf, coords, data_index, None)

        assert set(per_split["positions"].keys()) == {0}
        assert set(per_split["templates"].keys()) == {0}
        # Exactly one invocation — the short-circuit wrapper skips split 1.
        assert calls == [3]


# ---------------------------------------------------------------------------
# compute_likelihood_from_coords
# ---------------------------------------------------------------------------


class TestComputeLikelihoodFromCoords:
    def test_matches_direct_compute_likelihood(self):
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

        flat_vals, flat_sfreqs = wf(*coords.T)
        likes_manual = coord.compute_likelihood(
            data_index, data_index, flat_vals, flat_sfreqs
        )
        np.testing.assert_allclose(likes_auto, likes_manual, rtol=1e-12)


# ---------------------------------------------------------------------------
# generate_flat_templates_from_coords
# ---------------------------------------------------------------------------


class TestGenerateFlatTemplates:
    def test_flat_order_preserved_across_splits(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        nchan, nfreq = acs.nchannels, acs.num_freqs

        def wf(c0, c1):
            k = c0.shape[0]
            # Encode coord[:, 0] into the template so scatter is verifiable.
            base = c0[:, None, None].astype(np.complex128)
            tmpl = np.broadcast_to(base[..., None], (k, nchan, nfreq)).copy()
            return tmpl, c0.astype(np.float64) + 0.5

        data_index = np.array([0, 3, 1, 2, 0], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64),
             np.zeros(data_index.shape[0], dtype=np.float64)]
        )

        flat_vals, flat_sfreqs, flat_stimes = (
            coord.generate_flat_templates_from_coords(wf, coords, data_index)
        )

        assert flat_vals.shape == (data_index.shape[0], nchan, nfreq)
        assert flat_stimes is None
        for i in range(data_index.shape[0]):
            assert flat_vals[i, 0, 0] == coords[i, 0] + 0j
            assert flat_sfreqs[i] == coords[i, 0] + 0.5


# ---------------------------------------------------------------------------
# warm_jax_compile via (coords, data_index)
# ---------------------------------------------------------------------------


class TestWarmJaxCompileFromCoords:
    def test_derives_per_split_from_flat_batch(self):
        acs = _StubACS(num_acs=4, num_splits=2, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)

        calls = []

        def wf(*cols, **_):
            calls.append(np.stack(cols, axis=-1))
            return None

        # Split 0 owns AC ids {0, 1}; split 1 owns {2, 3}.
        # data_index pattern: positions of split-0 entries are [0, 2, 4];
        # positions of split-1 entries are [1, 3]. With sample_size=2 we
        # expect coords[[0, 2]] for split 0 and coords[[1, 3]] for split 1.
        data_index = np.array([0, 3, 1, 2, 0], dtype=np.int32)
        coords = np.column_stack(
            [np.arange(data_index.shape[0], dtype=np.float64),
             np.arange(data_index.shape[0], dtype=np.float64) + 10.0]
        )

        coord.warm_jax_compile(
            wf, coords=coords, data_index=data_index, sample_size_per_split=2
        )

        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0], coords[[0, 2]])
        np.testing.assert_array_equal(calls[1], coords[[1, 3]])
        assert coord._warm_compile_done is True

    def test_already_warmed_is_noop(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        coord._warm_compile_done = True

        calls = []
        coord.warm_jax_compile(
            lambda *a, **k: calls.append(1),
            sample_coords_per_split=[np.zeros((1, 2))],
        )
        assert calls == []

    def test_both_entry_forms_rejected(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        with pytest.raises(ValueError, match="either"):
            coord.warm_jax_compile(
                lambda *a, **k: None,
                sample_coords_per_split=[np.zeros((1, 2))],
                coords=np.zeros((1, 2)),
                data_index=np.zeros(1, dtype=np.int32),
            )

    def test_no_entry_form_rejected(self):
        acs = _StubACS(num_acs=2, num_splits=1, num_channels=3, num_freqs=16, df=1e-3)
        coord = DomainComputationGroupArray(acs)
        with pytest.raises(ValueError, match="sample_coords_per_split"):
            coord.warm_jax_compile(lambda *a, **k: None)
