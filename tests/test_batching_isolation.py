"""Batching is opt-in, and the other source classes must stay out of it.

The batched likelihood path exists for one generator family (grid-aligned
phentax MBHB). Every other source -- GB, VGB, SOBBH, EMRI -- shares the WDM
transform, the inner products and the TDI delay kernel with it, so the thing
that has to be pinned is that they never ENTER the batched path, whatever
happens to their own code later.

Isolation here is default-deny at the READ site, not inheritance. The sources
do not share a base class:

    GBXYZTDIWaveform         (no base)
    SOBBHWaveform            (no base)
    GBAETWaveform            AETTDIWaveform
    EMRITDIWaveform          AETTDIWaveform
    PhenomTHMTDIWaveform     TDPyResponseWaveformBase, PhenomTHMWaveformBase

so a ``supports_batch = False`` on any one base would not reach them all.
``getattr(gen, "supports_batch", False)`` does, including for classes with no
base at all and for anything third-party.
"""

from __future__ import annotations

import numpy as np
import pytest


def _supports_batch(obj) -> bool:
    """Exactly the read the container performs."""
    return bool(getattr(obj, "supports_batch", False))


# --------------------------------------------------------------------------
# 1. no other source advertises the capability
# --------------------------------------------------------------------------
def _source_classes():
    """Import lazily and skip individually; a missing optional dep for one
    source must not hide the check for the others."""
    out = []
    for mod, name in (
        ("lisatools.sources.gb.waveform", "GBXYZTDIWaveform"),
        ("lisatools.sources.gb.waveform", "GBAETWaveform"),
        ("lisatools.sources.emri.waveform", "EMRITDIWaveform"),
        ("lisatools.sources.sobbh.waveform", "SOBBHWaveform"),
        ("lisatools.sources.bbh.waveform", "BBHSNRWaveform"),
    ):
        try:
            out.append((name, getattr(__import__(mod, fromlist=[name]), name)))
        except Exception as exc:  # pragma: no cover - env dependent
            out.append((name, exc))
    return out


@pytest.mark.parametrize("name,cls", _source_classes(), ids=lambda v: getattr(v, "__name__", str(v)[:20]))
def test_sources_do_not_advertise_batching(name, cls):
    """No source but the grid-aligned MBHB one may claim batch support.

    Checked on the CLASS, so it fires even if a future shared base introduces
    the attribute -- which is the realistic way this would break.
    """
    if isinstance(cls, Exception):
        pytest.skip(f"{name} unavailable in this environment: {cls}")
    assert _supports_batch(cls) is False, (
        f"{name} now advertises supports_batch=True. Batching was designed "
        f"for generators that can guarantee a shared sub-sample alignment; if "
        f"{name} genuinely can, add a golden-value test for it before opting "
        f"in, because pyResponseTDI will otherwise silently share one "
        f"evaluation grid across sources that need different ones."
    )


def test_base_classes_declare_the_default():
    """Both ABCs declare it False -- documentation, not the mechanism."""
    from lisatools.sources.waveformbase import AETTDIWaveform, TDWaveformBase

    assert TDWaveformBase.supports_batch is False
    assert AETTDIWaveform.supports_batch is False


def test_base_class_docstrings_survive():
    """The attribute must not be inserted ahead of the class docstring."""
    from lisatools.sources.waveformbase import AETTDIWaveform, TDWaveformBase

    assert TDWaveformBase.__doc__, "TDWaveformBase lost its docstring"
    assert AETTDIWaveform.__doc__, "AETTDIWaveform lost its docstring"


# --------------------------------------------------------------------------
# 2/3. the container refuses to batch anything that has not opted in
# --------------------------------------------------------------------------
class _PlainGen:
    """A generator with no ``supports_batch`` attribute at all."""

    def __init__(self):
        self.calls = []

    n_active = 256

    def __call__(self, *params, **kwargs):
        self.calls.append(params)
        return np.zeros((3, self.n_active), dtype=complex)


class _OptedInGen(_PlainGen):
    supports_batch = True


def _container(gen):
    """Minimal CPU container, built the way the other tests here build one
    (``FDSettings`` + ``FDSignal``); ``DataResidualArray`` is deprecated and
    refuses a bare array without ``input_signal_domain``."""
    from lisatools import detector as lisa
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix

    settings = FDSettings(
        N=256, df=1e-4, min_freq=1e-4, max_freq=2e-2, force_backend="cpu",
    )
    sens_mat = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)
    data = FDSignal(np.zeros((3, settings.N_active), dtype=complex), settings)
    return AnalysisContainer(data, sens_mat, signal_gen=gen), settings


def test_unknown_generator_defaults_to_serial():
    """No attribute at all => serial, quietly. Not an error."""
    gen = _PlainGen()
    try:
        aca, _settings = _container(gen)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"container unavailable: {exc}")

    assert _supports_batch(aca._signal_gen) is False
    assert aca.batch_evaluation is True, (
        "batch_evaluation defaults True; isolation must come from the "
        "capability read, not from the flag being off"
    )


class _Tripwire(BaseException):
    """Deliberately NOT an Exception.

    The production fallback in ``eryn_likelihood_wrap`` catches ``Exception``,
    so a tripwire raised as ``AssertionError`` is swallowed, converted into a
    serial result, and the test passes whether or not the gate works. That was
    measured: deleting the gate entirely left this test green. Inheriting from
    ``BaseException`` puts the tripwire outside what the fallback can catch.
    """


def test_batched_branch_not_taken_for_non_batching_generator(monkeypatch):
    """A 2D block against a non-batching generator must take the serial loop.

    Asserts three independent things, because any one alone can pass
    vacuously: the batched worker never ran, nothing was recorded as a
    fallback (which would mean it ran and failed), and the generator really
    was called once per row.
    """
    gen = _PlainGen()
    try:
        aca, _settings = _container(gen)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"container unavailable: {exc}")

    def _tripwire(*a, **k):
        raise _Tripwire(
            "the batched path ran for a generator that never declared "
            "supports_batch"
        )

    monkeypatch.setattr(aca, "_batched_likelihood", _tripwire)
    x = np.zeros((4, 2))

    # _Tripwire is a BaseException, so it propagates out of the production
    # `except Exception` and fails this test loudly if the gate is loosened.
    aca.eryn_likelihood_wrap(x)

    assert aca.n_batch_fallbacks == 0, (
        "a non-batching generator must not even attempt a batched launch; "
        f"n_batch_fallbacks = {aca.n_batch_fallbacks}, "
        f"last error {aca.last_batch_error!r}"
    )
    assert len(gen.calls) == 4, (
        f"expected one generator call per row from the serial loop; got "
        f"{len(gen.calls)}"
    )


def test_shape_mismatch_is_refused_not_returned():
    """A generator that opts in but returns the wrong count must not silently
    misalign likelihoods with walkers."""
    gen = _OptedInGen()
    try:
        aca, _settings = _container(gen)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"container unavailable: {exc}")

    monkeypatch_val = np.zeros(2)  # wrong length for a 4-row batch
    aca.template_likelihood = lambda *a, **k: monkeypatch_val
    aca._build_batched_template = lambda *a, **k: object()

    with pytest.raises(ValueError, match="expected"):
        aca._batched_likelihood(np.zeros((4, 2)))


# --------------------------------------------------------------------------
# grid-aligned generation must not diverge from the stock path except in
# the grid it evaluates on
# --------------------------------------------------------------------------
def test_grid_aligned_forwards_reference_quantities_whole():
    """``t_min`` must reach phentax, not be replaced by NaN.

    ``get_reference_quantities`` adds ``t_min = -T`` whenever
    ``time_bounded_start`` is set, which is the DEFAULT, and phentax derives
    the waveform start from ``f_min`` ONLY when ``t_min`` is NaN. An earlier
    version of ``_aligned_polarizations`` filled ``initial_processing``'s
    positionals by hand and passed NaN there, which un-bounded the template in
    time: 57,789 valid samples against the stock 525,970 at m1 = 1e7,
    m2 = 8e6 Msun. That is a physics divergence wearing grid-alignment's
    clothes, so it is pinned at the call boundary.
    """
    import inspect

    src = inspect.getsource(_aligned_polarizations_source())
    assert "**ref_kw" in src, (
        "reference quantities must be forwarded WHOLE to initial_processing; "
        "naming individual keys silently drops t_min"
    )
    assert "jnp.nan" not in src, (
        "initial_processing's t_min positional must not be hardcoded to NaN"
    )


def _aligned_polarizations_source():
    from lisatools.sources.bbh.waveform import GridAlignedPhenomTHMTDIWaveform

    return GridAlignedPhenomTHMTDIWaveform._aligned_polarizations


def test_grid_aligned_supports_batch_tracks_alignment():
    """One decision in one place: the flag must follow ``grid_align``."""
    from lisatools.sources.bbh.waveform import GridAlignedPhenomTHMTDIWaveform

    assert isinstance(
        GridAlignedPhenomTHMTDIWaveform.__dict__["supports_batch"], property
    ), "supports_batch must be a property, not a class-level constant that can disagree with grid_align"

    class _Stub:
        grid_align = True
        supports_batch = GridAlignedPhenomTHMTDIWaveform.supports_batch

    s = _Stub()
    assert s.supports_batch is True
    s.grid_align = False
    assert s.supports_batch is False, (
        "with alignment off the generator must stop advertising batching; "
        "otherwise every launch is refused by pyResponseTDI"
    )
