"""Multi-shard behavior of the unified ResidualAddOneRemoveOneMove.

Covers the shard-aware ``compute_acs_like`` (rows grouped by owning split
and scored INSIDE that device context), the DCGA plumbing of the unified
move (one structure: ``dcga=`` selects the per-device replica path), the
deprecated ``MultiGPUResidualAddRemoveMove`` shim, and
``recipe.get_shared_dcga`` caching.
"""

from __future__ import annotations

import unittest
import warnings
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


class _FakeData:
    """Residual-array stand-in recording the sign + device of each apply."""

    def __init__(self, xp):
        self._xp = xp
        self.applied = []  # list of (sign, device)

    def add_signal(self, sig, sign):
        self.applied.append((sign, self._xp.current_device))


class _FakeAC:
    """Container stand-in recording the device its ops ran under."""

    def __init__(self, row, xp):
        self.row = row
        self._xp = xp
        self.seen_devices = []       # calculate_signal_likelihood
        self.built_devices = []      # build_template
        self.data = _FakeData(xp)

    def calculate_signal_likelihood(self, params_dict, **kwargs):
        self.seen_devices.append(self._xp.current_device)
        return float(self.row)

    def build_template(self, params_dict, **kwargs):
        self.built_devices.append(self._xp.current_device)
        return ("sig", self.row)


class _FakeIndexableACA(FakeMultiShardACA):
    """Fake ACA running the REAL ACA entry points the move delegates to.

    The move no longer carries its own sharded loops (2026-07-28 ACA
    delegation): ``compute_acs_like`` -> ``calculate_signal_likelihood`` and
    ``_apply_cold_chain_sources`` -> ``apply_signal_from_params``. Binding
    the real methods onto the fake means these tests exercise the REAL
    dispatcher / fused build+apply code against the recording xp/device
    machinery — same assertions, deeper coverage.
    """

    from lisatools.analysiscontainer import AnalysisContainerArray as _RealACA

    calculate_signal_likelihood = _RealACA.calculate_signal_likelihood
    _vectorized_dispatch = _RealACA._vectorized_dispatch
    apply_signal_from_params = _RealACA.apply_signal_from_params
    _payload_leading = staticmethod(_RealACA._payload_leading)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acs = [_FakeAC(i, self.xp) for i in range(self.acs_total_entries)]

    @property
    def num_acs(self):
        return self.acs_total_entries

    @property
    def acs(self):
        return SimpleNamespace(flatten=lambda: self._acs)

    def __getitem__(self, i):
        return self._acs[int(i)]

    def _signal_on_device(self, sig, gpu):
        # Real ACA migrates the array; the fake just records nothing and
        # returns it (device residency is exercised via the context stack).
        return sig


class ComputeAcsLikeShardTest(unittest.TestCase):
    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.globalfit.moves.addremovemove import (
                ResidualAddOneRemoveOneMove)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"addremovemove not available: {exc}")
        self.MoveCls = ResidualAddOneRemoveOneMove
        self.acs = _FakeIndexableACA((3, 4), self.NUM_ACS, self.NUM_SHARDS,
                                     layout="blocked")
        # minimal duck-typed move "self" for the unbound method call
        self.fake_move = SimpleNamespace(
            acs=self.acs,
            branch_name="emri",
            _resolve_signal_gen_override=lambda ac: None,
            _branch_waveform_kwargs=lambda: {},
        )

    def test_rows_grouped_by_split_and_device(self):
        data_index = np.array([5, 0, 3, 2, 1])
        coords = np.arange(len(data_index) * 4, dtype=float
                           ).reshape(len(data_index), 4)
        ll = self.MoveCls.compute_acs_like(
            self.fake_move, coords, data_index)
        # scoring returned in the caller's row order
        np.testing.assert_array_equal(ll, data_index.astype(float))
        # each AC was scored exactly once, inside its owning device context
        for i in data_index:
            ac = self.acs[int(i)]
            self.assertEqual(ac.seen_devices,
                             [int(self.acs.gpu_map[int(i)])])

    def test_threaded_dispatch_scores_each_ac_on_its_device(self):
        # run_threaded=True -> the per-split runner fans shards out across the
        # thread pool; each AC must still be scored exactly once and INSIDE its
        # own owning device context (RecordingXp tracks device thread-locally).
        acs = _FakeIndexableACA((3, 4), self.NUM_ACS, self.NUM_SHARDS,
                                layout="blocked", run_threaded=True)
        fake_move = SimpleNamespace(
            acs=acs,
            branch_name="emri",
            _resolve_signal_gen_override=lambda ac: None,
            _branch_waveform_kwargs=lambda: {},
        )
        data_index = np.arange(self.NUM_ACS)
        coords = np.zeros((self.NUM_ACS, 4))
        ll = self.MoveCls.compute_acs_like(fake_move, coords, data_index)
        np.testing.assert_array_equal(ll, data_index.astype(float))
        for i in range(self.NUM_ACS):
            ac = acs[i]
            self.assertEqual(ac.seen_devices, [int(acs.gpu_map[i])])
        # both shards were actually entered (real fan-out, not all-on-primary)
        self.assertEqual(set(acs.xp.device_log), {0, 1})

    def test_domain_error_scores_floor(self):
        from lisatools.utils.exceptions import WaveformDomainError

        class _RaisingAC(_FakeAC):
            def calculate_signal_likelihood(self, params_dict, **kwargs):
                raise WaveformDomainError("outside domain")

        self.acs._acs[2] = _RaisingAC(2, self.acs.xp)
        data_index = np.array([2, 4])
        coords = np.zeros((2, 4))
        ll = self.MoveCls.compute_acs_like(self.fake_move, coords, data_index)
        self.assertEqual(ll[0], -1e300)
        self.assertEqual(ll[1], 4.0)


class ApplyColdChainShardTest(unittest.TestCase):
    """`_apply_cold_chain_sources` builds + applies each walker's template
    INSIDE its owning device context (the serial per-walker loop that
    previously made source moves identical on 1 vs 2 GPU)."""

    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.globalfit.moves.addremovemove import (
                ResidualAddOneRemoveOneMove)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"addremovemove not available: {exc}")
        self.MoveCls = ResidualAddOneRemoveOneMove

    def _run(self, run_threaded):
        acs = _FakeIndexableACA((3, 4), self.NUM_ACS, self.NUM_SHARDS,
                                layout="blocked", run_threaded=run_threaded)
        fake_move = SimpleNamespace(
            acs=acs,
            branch_name="emri",
            _resolve_signal_gen_override=lambda ac: None,
            _branch_waveform_kwargs=lambda: {},
        )
        coords = np.zeros((self.NUM_ACS, 4))
        self.MoveCls._apply_cold_chain_sources(fake_move, coords, sign=-1)
        return acs

    def test_serial_builds_and_applies_on_owning_device(self):
        acs = self._run(run_threaded=False)
        for i in range(self.NUM_ACS):
            ac = acs[i]
            dev = int(acs.gpu_map[i])
            self.assertEqual(ac.built_devices, [dev])       # built on its device
            self.assertEqual(ac.data.applied, [(-1, dev)])  # applied there, sign -1

    def test_threaded_builds_and_applies_on_owning_device(self):
        acs = self._run(run_threaded=True)
        for i in range(self.NUM_ACS):
            ac = acs[i]
            dev = int(acs.gpu_map[i])
            self.assertEqual(ac.built_devices, [dev])
            self.assertEqual(ac.data.applied, [(-1, dev)])
        self.assertEqual(set(acs.xp.device_log), {0, 1})    # both shards entered


class _StubComputationGroup:
    def __init__(self, orbits="orbits"):
        self.orbits = orbits
        self.sensitivity_backend = SimpleNamespace(use_splines=False)


class _StubDCGA:
    """Recording DomainComputationGroupArray stand-in."""

    def __init__(self, acs, num_splits=2):
        self.acs = acs
        self.gpus = list(range(num_splits))
        self.num_splits = num_splits
        self.computation_groups = [
            _StubComputationGroup() for _ in range(num_splits)
        ]
        self.calls = []

    def device_context(self, device):
        return nullcontext()

    def compute_d_d_terms(self):
        self.calls.append("compute_d_d_terms")

    def free_gpu_memory(self):
        self.calls.append("free_gpu_memory")

    @property
    def xp(self):
        return np


class _KwargsGen:
    """Waveform generator object exposing the ``.kwargs`` replica seed."""

    def __init__(self, tag="main", orbits="orbits"):
        self.kwargs = {"tag": tag, "orbits": orbits}
        self.tag = tag

    def gen_method(self, *params, **kwargs):
        return ("wave", self.tag, params)


class UnifiedMoveDcgaTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.globalfit.moves.addremovemove import (
                MultiGPUResidualAddRemoveMove, ResidualAddOneRemoveOneMove)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"addremovemove not available: {exc}")
        self.MoveCls = ResidualAddOneRemoveOneMove
        self.ShimCls = MultiGPUResidualAddRemoveMove
        self.acs = _FakeIndexableACA((3, 4), 4, 2, layout="blocked")
        self.dcga = _StubDCGA(self.acs, num_splits=2)
        self.ctor_common = dict(
            waveform_gen_kwargs={},
            waveform_like_kwargs={},
            num_repeats=1,
            transform_fn=None,
            priors={},
            inner_moves=[],
        )

    def _coords_shape(self):
        return (2, 4, 1, 4)  # ntemps, nwalkers, nleaves_max, ndim

    def test_dcga_ctor_builds_replicas(self):
        gen = _KwargsGen()
        move = self.MoveCls(
            "emri", self._coords_shape(), gen,
            acs=None, dcga=self.dcga, waveform_gen_method="gen_method",
            **self.ctor_common,
        )
        self.assertIs(move.dcga, self.dcga)
        self.assertIs(move.acs, self.acs)          # acs from dcga
        self.assertEqual(len(move.waveform_generators), 2)
        # each replica seeded with the split's orbits
        for wg in move.waveform_generators:
            self.assertIsInstance(wg, _KwargsGen)
        # bound-callable compatibility retained
        self.assertEqual(move.waveform_gen("x")[0], "wave")

    def test_plain_ctor_has_no_dcga(self):
        move = self.MoveCls(
            "emri", self._coords_shape(), lambda *p, **k: None,
            acs=self.acs,
            **self.ctor_common,
        )
        self.assertIsNone(move.dcga)
        # setup_likelihood_here is a no-op without a DCGA
        move.setup_likelihood_here(None)
        self.assertEqual(self.dcga.calls, [])

    def test_setup_likelihood_here_dispatch(self):
        gen = _KwargsGen()
        move = self.MoveCls(
            "emri", self._coords_shape(), gen,
            acs=None, dcga=self.dcga, waveform_gen_method="gen_method",
            **self.ctor_common,
        )
        move.setup_likelihood_here(None)
        self.assertEqual(self.dcga.calls, ["compute_d_d_terms"])
        move.free_gpu_memory()
        self.assertEqual(self.dcga.calls,
                         ["compute_d_d_terms", "free_gpu_memory"])

    def test_deprecated_shim_maps_signature(self):
        gen = _KwargsGen()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            move = self.ShimCls(
                self.dcga, gen, "emri", self._coords_shape(),
                "gen_method", {}, {}, 1, None, {}, [],
            )
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertIsInstance(move, self.MoveCls)
        self.assertIs(move.dcga, self.dcga)
        self.assertEqual(len(move.waveform_generators), 2)


class GetSharedDcgaTest(unittest.TestCase):
    def setUp(self):
        try:
            import lisatools.domaincomputation as dc
            from lisatools.globalfit.recipe import get_shared_dcga
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"recipe not available: {exc}")
        self.dc = dc
        self.get_shared_dcga = get_shared_dcga

    def test_none_on_cpu_and_single_gpu(self):
        acs = SimpleNamespace(gpus=None)
        self.assertIsNone(self.get_shared_dcga(acs))
        acs = SimpleNamespace(gpus=[0])
        self.assertIsNone(self.get_shared_dcga(acs))

    def test_built_once_and_cached(self):
        built = []

        class _FakeDCGA:
            def __init__(self, acs):
                built.append(acs)
                self.num_splits = len(acs.gpus)

        real = self.dc.DomainComputationGroupArray
        self.dc.DomainComputationGroupArray = _FakeDCGA
        try:
            acs = SimpleNamespace(gpus=[0, 1])
            d1 = self.get_shared_dcga(acs)
            d2 = self.get_shared_dcga(acs)
        finally:
            self.dc.DomainComputationGroupArray = real
        self.assertIs(d1, d2)
        self.assertEqual(len(built), 1)
        self.assertIs(acs._shared_dcga, d1)

    def test_fallback_when_sens_mat_lacks_orbits(self):
        # A multi-GPU ACA whose sensitivity matrix carries no ``orbits``
        # (the stock CompositeSensitivityBackend case) must NOT build a DCGA
        # -- the per-device C++ replicas would assert. get_shared_dcga
        # returns None so the moves take the plain shard-aware path.
        built = []

        class _FakeDCGA:
            def __init__(self, acs):
                built.append(acs)

        class _NoOrbitsSens:  # no ``orbits`` / ``kwargs`` attributes
            pass

        class _Inner:
            def __init__(self, sens):
                self._sens = sens

            def flatten(self):
                return [SimpleNamespace(sens_mat=self._sens)]

        real = self.dc.DomainComputationGroupArray
        self.dc.DomainComputationGroupArray = _FakeDCGA
        try:
            acs = SimpleNamespace(gpus=[0, 1], acs=_Inner(_NoOrbitsSens()))
            result = self.get_shared_dcga(acs)
        finally:
            self.dc.DomainComputationGroupArray = real
        self.assertIsNone(result)
        self.assertEqual(len(built), 0)  # never constructed
        self.assertIsNone(getattr(acs, "_shared_dcga", None))


if __name__ == "__main__":
    unittest.main()
