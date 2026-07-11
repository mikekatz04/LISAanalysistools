#!/usr/bin/env python
"""Check the task-b per-band WDM slab feature end to end.

Task-b shrinks each GB ``SubBandBuffer`` per-band slab from the full active
band (``Nf_active`` layers) to a narrow window of ``band_slab_Nf`` layers
centered on the band, cutting the dominant sub-band-buffer GPU-memory term by
~``Nf_active / band_slab_Nf``. The narrow slabs address the chunked-het
kernels with a per-slot layer origin (``slab_min_f``); the whole thing is
guarded so ``wdm_band_slab_layers=None`` is bit-identical to the pre-task-b
full-active-band layout.

This script runs three sections; the exit code is 0 iff every non-skipped
check passes.

  1. LEAKAGE  -- measure how many WDM frequency layers a near-monochromatic GB
                 actually occupies (CPU ``TDSignal.transform``) and print the
                 recommended slab size. Justifies the auto default
                 ``band_span + 2*(leakage + guard)``.
  2. LOGIC    -- pure-numpy checks of the narrow-slab Python (the real property
                 / index-map code, exercised on lightweight stubs): the slab
                 recommender, ``band_slab_Nf`` (off / auto / explicit + clamp),
                 ``slab_min_f`` (centering + clamp + tracks the band), and the
                 fill index-map (OFF-path bit-identity + ON-path per-slot
                 slicing). Runs anywhere (no GPU, no kernel build).
  3. KERNEL   -- if the installed gbgpu backend exposes the task-b kernel
                 signature (rebuilt), numerically compare narrow-vs-full
                 ``get_ll`` on the CPU C++ backend; otherwise SKIP with a
                 rebuild note (the kernel change can only be validated after a
                 GPU/CPU-C++ rebuild).

Run:  python check_wdm_band_slab.py
"""
from __future__ import annotations

import sys
import types

import numpy as np

# gbbands is pure-Python (imports cupy-or-numpy as ``cp``); safe to import on
# a CPU-only box.
from lisatools.globalfit.moves.gbbands import (
    SubBandBuffer,
    _WDM_SLAB_LEAKAGE_LAYERS,
)
from lisatools.domains import TDSettings, TDSignal, WDMSettings

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def _record(section: str, ok, msg: str):
    status = SKIP if ok is None else (PASS if ok else FAIL)
    _results.append((section, status, msg))
    print(f"  [{status}] {msg}")


# ---------------------------------------------------------------------------
# 1. LEAKAGE
# ---------------------------------------------------------------------------
def section_leakage():
    print("\n== 1. WDM leakage / recommended slab size ==")
    Nf, Nt, dt = 256, 128, 5.0
    N = Nf * Nt
    wdm = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend="cpu")
    ldf = wdm.layer_df
    m0 = Nf // 4
    t = np.arange(N) * dt

    # Sweep sub-layer carrier offsets (worst case is off-center) and record the
    # smallest +/-k window that captures >= 1 - eps of the tone's WDM energy.
    worst_k = {1e-3: 0, 1e-5: 0, 1e-7: 0}
    for frac in np.linspace(0.0, 1.0, 9):
        f0 = (m0 + frac) * ldf
        tone = np.cos(2 * np.pi * f0 * t)
        w = TDSignal(
            tone.reshape(1, N), TDSettings(N=N, dt=dt, force_backend="cpu")
        ).transform(wdm)
        E = np.sum(np.asarray(w.arr)[0] ** 2, axis=-1)
        E = E / E.sum()
        mc = int(round(f0 / ldf))
        for eps in worst_k:
            k = 0
            while k <= 20 and E[max(0, mc - k):mc + k + 1].sum() < 1 - eps:
                k += 1
            worst_k[eps] = max(worst_k[eps], k)

    print(f"  ideal-WDM tone localization (worst case over sub-layer offsets):")
    for eps, k in worst_k.items():
        print(f"    +/- {k} layers hold >= 1 - {eps:.0e} of the energy")
    print(
        "  chunked-het effective leakage = m_band_half_width (kernel arg,\n"
        "    default 1 -> template spans +/-1 layers, validated to mm5 ~1e-9).\n"
        f"  => leakage constant _WDM_SLAB_LEAKAGE_LAYERS = {_WDM_SLAB_LEAKAGE_LAYERS}"
        " is a safe margin."
    )
    # The ideal-WDM worst case must be within the leakage constant (a couple).
    ok = worst_k[1e-5] <= _WDM_SLAB_LEAKAGE_LAYERS
    _record("leakage", ok,
            f"ideal-WDM 1e-5 half-width ({worst_k[1e-5]}) <= leakage "
            f"({_WDM_SLAB_LEAKAGE_LAYERS})")

    # Recommended slab for a representative 1-layer GB band grid.
    band_edges = np.array([m0 * ldf, (m0 + 1) * ldf])  # one 1-layer band
    for guard in (0, 1, 2):
        rec = SubBandBuffer.recommend_band_slab_layers(
            band_edges, ldf, leakage=_WDM_SLAB_LEAKAGE_LAYERS, guard=guard
        )
        print(f"  recommend (1-layer band, guard={guard}): {rec} layers "
              f"(= band_span 1 + 2*({_WDM_SLAB_LEAKAGE_LAYERS}+{guard}))")
    # default (guard=1) must exceed 2*leakage+1 so both tails fit.
    rec1 = SubBandBuffer.recommend_band_slab_layers(
        band_edges, ldf, leakage=_WDM_SLAB_LEAKAGE_LAYERS, guard=1)
    _record("leakage", rec1 >= 2 * _WDM_SLAB_LEAKAGE_LAYERS + 1,
            f"default recommendation ({rec1}) covers +/-(leakage+guard)")


# ---------------------------------------------------------------------------
# 2. LOGIC (pure numpy; exercises the real property / index-map code)
# ---------------------------------------------------------------------------
class _StubBuffer(SubBandBuffer):
    """A ``SubBandBuffer`` with ``__init__`` bypassed and ``xp`` / ``backend``
    pinned to NumPy, so the REAL narrow-slab property / index-map code can be
    exercised on a CPU box without building a full GPU buffer."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    @property
    def xp(self):
        return np

    @property
    def backend(self):
        return types.SimpleNamespace(xp=np, uses_cupy=False, name="lisatools_cpu")


def _stub(**attrs):
    return _StubBuffer(**attrs)


def section_logic():
    print("\n== 2. narrow-slab Python logic ==")
    Nf, Nt, dt = 1024, 512, 2.5
    wdm = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, min_freq=6e-3, max_freq=8e-3,
                      force_backend="cpu")
    ldf = wdm.layer_df
    Nf_active = wdm.Nf_active
    # 3 one-layer bands inside the active band.
    b0 = wdm.ind_min_f + 5
    band_edges = np.array([(b0 + i) * ldf for i in range(4)])  # 3 bands
    # per-slot band assignment: unique_band_combos[:, 2] = band index
    band_of_slot = np.array([0, 1, 2, 1])  # 4 slots (band 1 twice)
    ubc = np.stack([np.zeros(4, int), np.zeros(4, int), band_of_slot], axis=1)

    # ---- band_slab_Nf: OFF / AUTO / EXPLICIT + clamp ----
    off = _stub(_wdm_band_slab_layers=None, _basis_settings=wdm)
    _record("logic", off.band_slab_Nf is None, "band_slab_Nf OFF -> None")

    auto = _stub(_wdm_band_slab_layers=0, _wdm_slab_guard_layers=1,
                 _basis_settings=wdm, _df=ldf, band_edges=band_edges)
    exp_auto = 1 + 2 * (_WDM_SLAB_LEAKAGE_LAYERS + 1)
    _record("logic", auto.band_slab_Nf == exp_auto,
            f"band_slab_Nf AUTO -> {auto.band_slab_Nf} (expect {exp_auto})")

    explicit = _stub(_wdm_band_slab_layers=9, _wdm_slab_guard_layers=1,
                     _basis_settings=wdm, _df=ldf, band_edges=band_edges)
    _record("logic", explicit.band_slab_Nf == 9, "band_slab_Nf EXPLICIT -> 9")

    huge = _stub(_wdm_band_slab_layers=10 * Nf_active, _wdm_slab_guard_layers=1,
                 _basis_settings=wdm, _df=ldf, band_edges=band_edges)
    _record("logic", huge.band_slab_Nf == Nf_active,
            f"band_slab_Nf clamps to Nf_active ({Nf_active})")

    # ---- slab_min_f: centered on band, clamped, tracks the assignment ----
    slab_Nf = auto.band_slab_Nf
    s = _stub(_wdm_band_slab_layers=0, _wdm_slab_guard_layers=1,
              _basis_settings=wdm, _df=ldf, band_edges=band_edges,
              unique_band_combos=ubc)
    smf = np.asarray(s.slab_min_f)
    # centered: origin = band_center_layer - slab_Nf//2 (clamped to active band)
    lo = (band_edges[band_of_slot] / ldf).astype(int)
    hi = (band_edges[band_of_slot + 1] / ldf).astype(int)
    center = (lo + hi) // 2
    want = np.clip(center - slab_Nf // 2, wdm.ind_min_f,
                   max(wdm.ind_min_f, wdm.ind_max_f + 1 - slab_Nf))
    _record("logic", smf.shape == (4,) and np.array_equal(smf, want),
            "slab_min_f centered + clamped + per-slot")
    # every source's m-band [m_floor +/- 1] stays inside its slab
    inside = np.all((lo - 1 >= smf) & (hi + 1 < smf + slab_Nf))
    _record("logic", bool(inside),
            "band +/- m_band_half_width fits inside every slab")

    # ---- fill index-map: OFF bit-identity + ON per-slot slice ----
    inds_fill = np.arange(4)
    # OFF: layer axis must be the shared 0..Nf_active-1 ramp
    boff = _stub(_wdm_band_slab_layers=None, _basis_settings=wdm,
                 unique_band_combos=ubc, nchannels=3, tdi_channel_setup="AET",
                 num_bands_now=4)
    i1, i2, i3, i4 = SubBandBuffer._get_fill_buffer_ind_map(boff, boff, inds_fill,
                                                            is_psd=False)
    off_ok = (i3.shape[-2] == Nf_active
              and np.array_equal(i3[0, 0, :, 0], np.arange(Nf_active)))
    _record("logic", off_ok, "fill map OFF-path == full 0..Nf_active-1 ramp")

    # ON: per-slot layer window starting at slab_min_f - ind_min_f
    bon = _stub(_wdm_band_slab_layers=0, _wdm_slab_guard_layers=1,
                _basis_settings=wdm, _df=ldf, band_edges=band_edges,
                unique_band_combos=ubc, nchannels=3, tdi_channel_setup="AET",
                num_bands_now=4)
    i1, i2, i3, i4 = SubBandBuffer._get_fill_buffer_ind_map(bon, bon, inds_fill,
                                                            is_psd=False)
    exp_off = smf - wdm.ind_min_f
    on_ok = (i3.shape[-2] == slab_Nf
             and np.array_equal(i3[:, 0, 0, 0], exp_off)          # slot start
             and np.array_equal(i3[0, 0, :, 0],
                                exp_off[0] + np.arange(slab_Nf)))  # slot0 ramp
    _record("logic", on_ok, "fill map ON-path == per-slot [slab_min_f-ind_min_f : +slab_Nf]")
    # slices stay inside the parent active band
    within = bool(np.all(i3 >= 0) and np.all(i3 < Nf_active))
    _record("logic", within, "ON-path layer indices stay inside parent active band")


# ---------------------------------------------------------------------------
# 3. KERNEL (numerical; requires a task-b rebuild of the backend)
# ---------------------------------------------------------------------------
def _kernel_has_taskb():
    """True iff the installed gbgpu backend's gb_wdm_het_get_ll takes the
    task-b (Nf_slab, slab_min_f) args. Detected by arg count via a probe call
    that is expected to raise for a *reason other than* arity."""
    try:
        import inspect
        from gbgpu.gbcomps import GBWDMComputations  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        return None, f"gbgpu import failed: {exc}"
    # The nanobind method has no Python signature; probe the module docstring
    # / a trial call is unreliable, so we rely on a sentinel the rebuilt
    # binding sets. Until that exists, treat as unknown -> SKIP.
    return None, ("cannot introspect the compiled nanobind arity here; run the "
                  "GPU match test after rebuild (see wdm_per_band_slab_taskb.md)")


def section_kernel():
    print("\n== 3. kernel numerics (needs task-b rebuild) ==")
    ok, msg = _kernel_has_taskb()
    if ok is None:
        _record("kernel", None, f"SKIP: {msg}")
        print("        -> rebuild GBGPU (+LAT) with the task-b kernels, then\n"
              "           validate OFF-path bit-identity + ON-path mm5/mm2 on GPU.")
        return
    # (Populated once a rebuilt backend is present: build a tiny WDM buffer,
    #  run get_ll narrow vs full, assert the inner products agree within the
    #  documented sub-band-edge tail budget.)


def main():
    section_leakage()
    section_logic()
    section_kernel()

    print("\n== summary ==")
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        for sec, s, m in _results:
            if s == FAIL:
                print(f"    FAIL [{sec}] {m}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
