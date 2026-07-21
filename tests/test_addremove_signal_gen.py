"""Unit tests for the signal_gen paths used by ``ResidualAddOneRemoveOneMove``.

The add/remove move no longer generates waveforms itself: every operation on
its :class:`AnalysisContainerArray` runs through the containers' ``signal_gen``
machinery, **defaulting to the generator the engine already installed on each
container** under the move's branch name and falling back to the move's own
:class:`MoveSignalGen` when the container doesn't carry that model name.
Because the move's choreography transforms coordinates once up front, the
containers' generators are always called with ``apply_transform=False`` — the
transform must be applied exactly once on every path.

Covered:

* ``build_template`` per-call ``signal_gen`` swap + ``apply_transform``
  injection (installed generator restored afterwards, ``None`` injects
  nothing).
* ``_apply_cold_chain_sources``: defaults to the installed branch generator;
  registers the move's own generator when the branch is missing (durably when
  safe, per-call otherwise, never clobbering); +1/-1 round trip.
* ``compute_acs_like``: installed-generator default matches a direct
  ``template_likelihood`` reference; the custom ``signal_gen`` heterodyne
  hook still scores with the given waveform-basis callable.
* Engine-convention transform handling: ``MoveSignalGen`` and the stock
  ``SourceSignalGen`` apply the branch transform by default and skip it with
  ``apply_transform=False``.
"""

from __future__ import annotations

import unittest

import numpy as np


class RecordingTransform:
    """Minimal transform stand-in: doubles param 0 and counts calls."""

    def __init__(self):
        self.calls = 0

    def both_transforms(self, params, leaf_inds=None):
        self.calls += 1
        out = np.array(params, dtype=float)
        out[..., 0] = 2.0 * out[..., 0]
        return out


def _toy_setup():
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix
    from lisatools import detector as lisa

    settings = FDSettings(N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2, force_backend="cpu")
    sens_mat = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)

    def wave_gen(amp, ind, scale=1.0):
        """The move's waveform-basis generator (old ``waveform_gen`` role)."""
        arr = np.zeros((3, settings.N_active), dtype=complex)
        arr[0, int(ind)] = amp * scale * 1e-20
        arr[1, int(ind)] = 0.5 * amp * scale * 1e-20
        return FDSignal(arr, settings)

    def installed_gen(amp, ind, apply_transform=True, scale=1.0):
        """Engine-convention installed generator, detectably different."""
        if apply_transform:
            amp = 2.0 * amp  # stand-in for the branch transform
        arr = np.zeros((3, settings.N_active), dtype=complex)
        arr[2, int(ind)] = amp * scale * 1e-20  # different channel than wave_gen
        return FDSignal(arr, settings)

    def make_acs(num, rng, signal_gen=None):
        acs = []
        for _ in range(num):
            arr = (
                rng.standard_normal((3, settings.N_active))
                + 1j * rng.standard_normal((3, settings.N_active))
            ) * 1e-20
            kwargs = {"signal_gen": signal_gen} if signal_gen is not None else {}
            acs.append(AnalysisContainer(FDSignal(arr, settings), sens_mat, **kwargs))
        return AnalysisContainerArray(acs, gpus=None)

    return settings, sens_mat, wave_gen, installed_gen, make_acs


def _make_move(acs, waveform_gen, waveform_gen_kwargs, branch="toy"):
    """Move instance with only the attributes the tested methods touch."""
    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    move = object.__new__(ResidualAddOneRemoveOneMove)
    move.acs = acs
    move.waveform_gen = waveform_gen
    move.waveform_gen_kwargs = waveform_gen_kwargs
    move.branch_name = branch
    move.transform_fn = RecordingTransform()
    return move


class ToySetupMixin:
    def setUp(self):
        try:
            (
                self.settings,
                self.sens_mat,
                self.wave_gen,
                self.installed_gen,
                self.make_acs,
            ) = _toy_setup()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")
        self.nw = 3
        rng = np.random.default_rng(7)
        # waveform-basis rows, one per walker (what the move's choreography
        # feeds after its single up-front transform)
        self.coords_in = np.column_stack(
            [rng.uniform(1.0, 3.0, self.nw), rng.integers(3, 40, self.nw).astype(float)]
        )


class BuildTemplateSignalGenSwapTest(ToySetupMixin, unittest.TestCase):
    def _bare_ac(self, **kwargs):
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.domains import FDSignal

        data = FDSignal(np.zeros((3, self.settings.N_active), complex), self.settings)
        return AnalysisContainer(data, self.sens_mat, **kwargs)

    def test_swap_on_bare_container_and_restore(self):
        ac = self._bare_ac()
        params = np.array([2.0, 5.0])
        out = ac.build_template(params, waveform_kwargs={"scale": 2.0}, signal_gen=self.wave_gen)
        np.testing.assert_array_equal(out.arr, self.wave_gen(*params, scale=2.0).arr)
        # the per-call generator must NOT stay installed
        with self.assertRaises(ValueError):
            ac.signal_gen

    def test_swap_restores_installed_dict(self):
        ac = self._bare_ac(signal_gen={"toy": self.installed_gen})
        params = np.array([2.0, 5.0])
        out = ac.build_template(params, signal_gen=self.wave_gen)  # single-model this call
        np.testing.assert_array_equal(out.arr, self.wave_gen(*params).arr)
        self.assertTrue(ac.is_multi_model)
        self.assertEqual(list(ac.signal_gen), ["toy"])

    def test_apply_transform_injection(self):
        ac = self._bare_ac(signal_gen={"toy": self.installed_gen})
        params = np.array([2.0, 5.0])
        # apply_transform=False reaches the generator (no internal transform)
        out = ac.build_template({"toy": params}, apply_transform=False)
        np.testing.assert_array_equal(
            out.arr, self.installed_gen(*params, apply_transform=False).arr
        )
        # default None injects nothing -> generator's own default (transform on)
        out = ac.build_template({"toy": params})
        np.testing.assert_array_equal(out.arr, self.installed_gen(*params).arr)
        self.assertFalse(
            np.array_equal(
                self.installed_gen(*params).arr,
                self.installed_gen(*params, apply_transform=False).arr,
            )
        )
        # None-injection also means plain generators (no kwarg) keep working
        out = ac.build_template(params, signal_gen=self.wave_gen)
        np.testing.assert_array_equal(out.arr, self.wave_gen(*params).arr)


class ApplyColdChainSourcesTest(ToySetupMixin, unittest.TestCase):
    def test_defaults_to_installed_generator(self):
        acs = self.make_acs(self.nw, np.random.default_rng(11), {"toy": self.installed_gen})
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        before = [acs[i].data.arr.copy() for i in range(self.nw)]

        move._apply_cold_chain_sources(self.coords_in, sign=+1)
        for i in range(self.nw):
            expect = before[i] + self.installed_gen(
                *self.coords_in[i], apply_transform=False, scale=1.5
            ).arr
            np.testing.assert_array_equal(acs[i].data.arr, expect)
        # installed entry untouched; the move transformed nothing here
        for i in range(self.nw):
            self.assertIs(acs[i].signal_gen["toy"], self.installed_gen)
        self.assertEqual(move.transform_fn.calls, 0)

        move._apply_cold_chain_sources(self.coords_in, sign=-1)
        for i in range(self.nw):
            np.testing.assert_allclose(acs[i].data.arr, before[i], rtol=1e-12)

    def test_missing_branch_registers_own_generator(self):
        from lisatools.globalfit.moves.addremovemove import MoveSignalGen

        acs = self.make_acs(self.nw, np.random.default_rng(11))  # no signal_gen at all
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        before = [acs[i].data.arr.copy() for i in range(self.nw)]

        move._apply_cold_chain_sources(self.coords_in, sign=+1)
        for i in range(self.nw):
            # own generator == the old inline path: wave_gen on the
            # already-transformed rows, untransformed again
            expect = before[i] + self.wave_gen(*self.coords_in[i], scale=1.5).arr
            np.testing.assert_array_equal(acs[i].data.arr, expect)
            # durably added under the branch name, engine-convention adapter
            own = acs[i].signal_gen["toy"]
            self.assertIsInstance(own, MoveSignalGen)
        self.assertEqual(move.transform_fn.calls, 0)

        # the registered adapter is engine-usable: raw params in, transform on
        raw = np.array([2.0, 5.0])
        out = acs[0].signal_gen["toy"](*raw)
        transformed = move.transform_fn.both_transforms(raw.copy())
        np.testing.assert_array_equal(out.arr, self.wave_gen(*transformed).arr)

    def test_dict_without_branch_gains_key_and_keeps_others(self):
        other = self.installed_gen
        acs = self.make_acs(self.nw, np.random.default_rng(11), {"other": other})
        move = _make_move(acs, self.wave_gen, {})
        move._apply_cold_chain_sources(self.coords_in, sign=+1)
        for i in range(self.nw):
            self.assertIn("toy", acs[i].signal_gen)
            self.assertIs(acs[i].signal_gen["other"], other)

    def test_incompatible_installed_generator_not_clobbered(self):
        # installed under the branch name but no apply_transform kwarg ->
        # the move must use its own generator per call and leave it alone
        def plain_gen(amp, ind, scale=1.0):
            return self.wave_gen(amp, ind, scale=scale)

        acs = self.make_acs(self.nw, np.random.default_rng(11), {"toy": plain_gen})
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        before = [acs[i].data.arr.copy() for i in range(self.nw)]
        move._apply_cold_chain_sources(self.coords_in, sign=+1)
        for i in range(self.nw):
            expect = before[i] + self.wave_gen(*self.coords_in[i], scale=1.5).arr
            np.testing.assert_array_equal(acs[i].data.arr, expect)
            self.assertIs(acs[i].signal_gen["toy"], plain_gen)


class ComputeAcsLikeTest(ToySetupMixin, unittest.TestCase):
    def _reference_ll(self, acs, gen_call, data_index, **kwargs):
        source_only = kwargs.pop("source_only", False)
        ll = np.empty(len(data_index), dtype=float)
        for i, di in enumerate(data_index):
            ll[i] = acs[int(di)].template_likelihood(
                gen_call(self.coords_in[i]),
                include_psd_info=not source_only,
                **kwargs,
            )
        return ll

    def test_defaults_to_installed_generator(self):
        acs = self.make_acs(self.nw, np.random.default_rng(11), {"toy": self.installed_gen})
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        data_index = np.array([0, 2, 1], dtype=np.int32)
        for extra in ({}, {"source_only": True}, {"phase_maximize": True}):
            got = move.compute_acs_like(self.coords_in, data_index, **dict(extra))
            want = self._reference_ll(
                acs,
                lambda row: self.installed_gen(*row, apply_transform=False, scale=1.5),
                data_index,
                **dict(extra),
            )
            np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
        self.assertEqual(move.transform_fn.calls, 0)

    def test_custom_signal_gen_hook(self):
        # heterodyne-style override: a different waveform-basis callable
        acs = self.make_acs(self.nw, np.random.default_rng(11), {"toy": self.installed_gen})
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        data_index = np.array([0, 1, 2], dtype=np.int32)

        got = move.compute_acs_like(self.coords_in, data_index, self.wave_gen)
        want = self._reference_ll(
            acs, lambda row: self.wave_gen(*row, scale=1.5), data_index
        )
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
        # override is per-call only: the installed generator stays put
        for i in range(self.nw):
            self.assertIs(acs[i].signal_gen["toy"], self.installed_gen)

    def test_single_row_scalar_index(self):
        acs = self.make_acs(self.nw, np.random.default_rng(11), {"toy": self.installed_gen})
        move = _make_move(acs, self.wave_gen, {"scale": 1.5})
        got = move.compute_acs_like(self.coords_in[0], np.int32(1))
        want = self._reference_ll(
            acs,
            lambda row: self.installed_gen(*row, apply_transform=False, scale=1.5),
            np.array([1]),
        )
        np.testing.assert_allclose(got, want, rtol=1e-13)


class EngineConventionGeneratorTest(unittest.TestCase):
    def test_stock_source_signal_gen_transform_toggle(self):
        try:
            from lisatools.globalfit.stock.erebor.source_runtime import SourceSignalGen
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"stock deps not installed: {exc}")

        transform = RecordingTransform()
        gen = SourceSignalGen("not-a-branch", transform, None, None)
        # apply_transform=False must skip the internal transform entirely
        with self.assertRaises(ValueError):
            gen(1.0, 2.0, apply_transform=False)
        self.assertEqual(transform.calls, 0)
        # default applies it (exactly once)
        with self.assertRaises(ValueError):
            gen(1.0, 2.0)
        self.assertEqual(transform.calls, 1)


if __name__ == "__main__":
    unittest.main()
