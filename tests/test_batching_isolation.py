"""Batching is opt-in; the other sources must never enter the batched path.

The batched likelihood exists for one generator family (grid-aligned phentax
MBHB). Every other source -- GB, VGB, SOBBH, EMRI -- shares the WDM transform,
the inner products and the TDI delay kernel with it, so what has to be pinned
is that they never ENTER the batched path, whatever happens to their code
later.

Isolation is DEFAULT-DENY AT THE READ SITE, not inheritance. The sources do
not share a base class::

    GBXYZTDIWaveform         (no base)
    SOBBHWaveform            (no base)
    GBAETWaveform            AETTDIWaveform
    EMRITDIWaveform          AETTDIWaveform
    PhenomTHMTDIWaveform     TDPyResponseWaveformBase, PhenomTHMWaveformBase

so ``supports_batch = False`` on any one base could not reach them all.
``getattr(gen, "supports_batch", False)`` does -- including classes with no
base at all, and anything third-party.
"""

from __future__ import annotations

import unittest

import numpy as np


def _supports_batch(obj) -> bool:
    """Exactly the read the container performs."""
    return bool(getattr(obj, "supports_batch", False))


class _Tripwire(BaseException):
    """Deliberately NOT an ``Exception``.

    The production fallback catches a narrow refusal type now, but it caught
    ``Exception`` once, and a tripwire raised as ``AssertionError`` was
    swallowed into a serial result -- deleting the gate entirely left the test
    green. Measured. ``BaseException`` puts the tripwire outside anything the
    fallback can catch, whatever it catches next.
    """


class _PlainGen:
    """A generator with no ``supports_batch`` attribute at all."""

    n_active = 256

    def __init__(self):
        self.calls = []

    def __call__(self, *params, **kwargs):
        self.calls.append(params)
        return np.zeros((3, self.n_active), dtype=complex)


def _container(gen):
    """Minimal CPU container, built the way the rest of tests/ builds one."""
    from lisatools import detector as lisa
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix

    settings = FDSettings(
        N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2, force_backend="cpu"
    )
    sens = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)
    data = FDSignal(np.zeros((3, settings.N_active), dtype=complex), settings)
    return AnalysisContainer(data, sens, signal_gen=gen)


class SourceCapabilityTest(unittest.TestCase):
    """No source but the grid-aligned MBHB one may claim batch support."""

    SOURCES = (
        ("lisatools.sources.gb.waveform", "GBXYZTDIWaveform"),
        ("lisatools.sources.gb.waveform", "GBAETWaveform"),
        ("lisatools.sources.emri.waveform", "EMRITDIWaveform"),
        ("lisatools.sources.sobbh.waveform", "SOBBHWaveform"),
        ("lisatools.sources.bbh.waveform", "BBHSNRWaveform"),
    )

    def test_sources_do_not_advertise_batching(self):
        """Checked on the CLASS: the realistic breakage is a future shared
        base introducing the attribute."""
        checked = 0
        for mod, name in self.SOURCES:
            with self.subTest(source=name):
                try:
                    cls = getattr(__import__(mod, fromlist=[name]), name)
                except Exception as exc:  # pragma: no cover - env dependent
                    self.skipTest(f"{name} unavailable: {exc}")
                    continue
                checked += 1
                self.assertFalse(
                    _supports_batch(cls),
                    f"{name} now advertises supports_batch=True. Batching was "
                    f"designed for generators that can guarantee a shared "
                    f"sub-sample alignment; if {name} genuinely can, add a "
                    f"batched-vs-serial test for it before opting in.",
                )
        self.assertGreater(checked, 0, "no source classes were actually checked")

    def test_base_classes_declare_the_default(self):
        from lisatools.sources.waveformbase import AETTDIWaveform, TDWaveformBase

        self.assertIs(TDWaveformBase.supports_batch, False)
        self.assertIs(AETTDIWaveform.supports_batch, False)

    def test_base_class_docstrings_survive(self):
        """The attribute must not be inserted ahead of the class docstring."""
        from lisatools.sources.waveformbase import AETTDIWaveform, TDWaveformBase

        self.assertTrue(TDWaveformBase.__doc__, "TDWaveformBase lost its docstring")
        self.assertTrue(AETTDIWaveform.__doc__, "AETTDIWaveform lost its docstring")

    def test_wrappers_do_not_forward_the_capability(self):
        """A blanket ``__getattr__`` must not manufacture the capability.

        ``_EMRISpecialFrameWrap`` forwards attribute lookups to the generator
        it wraps. Verified that it reported True whenever the wrapped object
        did -- inert only because nothing below sets the flag today, and this
        work is what adds a batch API to ``pyResponseTDI``.
        """
        from lisatools.sources.emri.response import _EMRISpecialFrameWrap

        class _OptedIn:
            supports_batch = True

        wrapper = _EMRISpecialFrameWrap.__new__(_EMRISpecialFrameWrap)
        wrapper._wave_gen = _OptedIn()
        self.assertFalse(
            _supports_batch(wrapper),
            "the EMRI frame wrapper forwarded supports_batch from the wrapped "
            "generator; a default-deny capability must not be reachable by "
            "blanket attribute forwarding",
        )


class ContainerGateTest(unittest.TestCase):
    """The container refuses to batch anything that has not opted in."""

    def setUp(self):
        self.gen = _PlainGen()
        try:
            self.aca = _container(self.gen)
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"container unavailable: {exc}")

    def test_unknown_generator_defaults_to_serial(self):
        """No attribute at all => excluded, quietly. Not an error."""
        self.assertFalse(_supports_batch(self.aca._signal_gen))
        self.assertTrue(
            self.aca.batch_evaluation,
            "batch_evaluation defaults True; isolation must come from the "
            "capability read, not from the flag being off",
        )

    def test_batched_branch_not_taken_for_non_batching_generator(self):
        """Asserts three things, because any one alone can pass vacuously."""
        def _tripwire(*a, **k):
            raise _Tripwire("batched path ran for a non-batching generator")

        self.aca.batched_signal_likelihood = _tripwire
        self.aca.eryn_likelihood_wrap(np.zeros((4, 2)))

        self.assertEqual(
            self.aca.n_batch_fallbacks, 0,
            f"a non-batching generator must not even attempt a batched launch; "
            f"last error {self.aca.last_batch_error!r}",
        )
        self.assertEqual(
            len(self.gen.calls), 4,
            "expected one generator call per row from the serial loop",
        )

    def test_container_level_kwargs_decline_the_batch(self):
        """Kwargs the batched builder cannot express must fall to serial."""
        from lisatools.analysiscontainer import _CONTAINER_LEVEL_KWARGS

        self.assertIn("apply_transform", _CONTAINER_LEVEL_KWARGS)
        self.assertIn("transform_fn", _CONTAINER_LEVEL_KWARGS)
        self.assertIn("signal_gen", _CONTAINER_LEVEL_KWARGS)


class SharedWindowGuardTest(unittest.TestCase):
    """The batched response must refuse rows that do not share a window start.

    ``_apply_response`` applies ONE crop (``start_ind = start_inds.max()``)
    and one fixed leading-buffer zeroing to every row of a batch. Rows whose
    valid data begins elsewhere are cropped at the wrong sample, and the
    likelihood comes out finite, plausible and WRONG -- measured at 7.6e5 nats
    on a 4-walker block, with the error growing as the walker cloud widens.
    Worse, a row's value then depends on which other rows shared its batch,
    which breaks detailed balance.

    The pre-existing alignment check does not cover this: it compares
    ``t0_shift_to_data``, which grid-aligned generation drives to exactly zero
    BY CONSTRUCTION, so it is inert for the one generator that can batch.

    Both the crop and the zeroing are pre-existing on dev; this branch is what
    makes them reachable for heterogeneous walkers, so the guard belongs here.
    """

    def test_guard_is_present_and_typed(self):
        import inspect

        from lisatools.sources.waveformbase import TDPyResponseWaveformBase

        # The guard lives on the pyResponseTDI subclass -- the legacy response
        # path, which is the one MBHB batching actually uses.
        src = inspect.getsource(TDPyResponseWaveformBase._apply_response)
        self.assertIn(
            "BatchNotLaunchable", src,
            "the shared-window guard must raise BatchNotLaunchable so callers "
            "fall back to per-row evaluation instead of dying",
        )
        self.assertIn(
            "start_inds.min()", src,
            "the guard must compare the SPREAD of start_inds; comparing only "
            "the max cannot detect rows that start elsewhere",
        )

    def test_refusal_type_is_catchable_as_a_refusal(self):
        from lisatools.utils.exceptions import (
            BatchNotLaunchable,
            LISAToolsException,
        )

        self.assertTrue(issubclass(BatchNotLaunchable, LISAToolsException))
        self.assertFalse(
            issubclass(BatchNotLaunchable, (TypeError, AttributeError)),
            "a refusal must be distinguishable from a wiring bug",
        )


class OptionalDependencyTest(unittest.TestCase):
    """Importing lisatools.sources.bbh must not require jax.

    ``gridaligned`` imports ``jax`` from ``waveform``, and
    ``sources/bbh/__init__.py`` imports ``gridaligned`` unconditionally, so
    leaving the name unbound when the optional import fails turned jax into a
    HARD dependency of the whole bbh package -- locking out anyone who only
    wants the pure-bbhx ``BBHSNRWaveform``.
    """

    def test_jax_name_is_always_bound(self):
        from lisatools.sources.bbh import waveform as wf

        self.assertTrue(
            hasattr(wf, "jax"),
            "waveform.py must bind `jax` on both branches of its try/except",
        )

    def test_bbh_package_exports_survive(self):
        from lisatools.sources.bbh import (  # noqa: F401
            BBHSNRWaveform,
            GridAlignedPhenomTHMTDIWaveform,
            PhenomTHMTDIWaveform,
        )
