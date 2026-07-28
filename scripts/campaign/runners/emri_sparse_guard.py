#!/usr/bin/env python
"""t2-lite-emri domain-guard check.

Exercises the EMRI waveform-domain guard THROUGH THE STOCK STRUCTURE: build a
stock EMRI-only ``full_year_combined`` fit and call the installed EMRI
``signal_gen`` (SourceSignalGen -> stock EMRI transform + FEW wave wrap) on an
out-of-domain param. Driving p0 toward/below the Kerr separatrix yields a
<3-point FEW trajectory -> ``WaveformDomainError``. That exception is precisely
what the sampler catches (addremovemove.py: ``except WaveformDomainError`` ->
ll = -1e300), so a raised WaveformDomainError == the -1e300 floor path firing;
the guard must raise it (catchable) with the PROCESS STILL ALIVE, not hard-exit
/ segfault. A valid param (the injection) must still produce a finite template
through the same generator, so an always-raising guard can't false-pass.

Emits for the ledger:
  [RESULT] guard_ll_floor=1 process_survived=1  (on success)

No hand-rolled waveform: uses the installed SourceSignalGen + stock EMRI
transform only.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# EMRI-only, matching the gate's _ONLY_EMRI selection.
os.environ.setdefault("MBHB_IDS", "")
os.environ.setdefault("EMRI_IDS", "1")
os.environ.setdefault("SOBHB_IDS", "")
os.environ.setdefault("EMRI_DEBUG", "0")

import numpy as np

P0_COL = 3  # EMRI sampling-basis col for p0 (introspected: injection p0=4.862)
A_COL = 2   # spin a
E_COL = 4   # eccentricity e0
# Fallback p0 probes (LOW-first so the first is below the separatrix and fails
# FAST -- a p0 well above the separatrix would compute a full slow waveform).
P0_PROBES_FALLBACK = (1.0, 1.5, 0.5, 2.0, 0.1)


def _separatrix_probes(a, e0):
    """p0 values right at/just inside the separatrix -> a <3-point trajectory
    (the documented guard trigger). Uses FEW's separatrix helper if available;
    the '+epsilon' plunge-immediately cases fail fast."""
    try:
        from few.utils.utility import get_separatrix
        p_sep = float(get_separatrix(abs(float(a)), float(e0), 1.0))
        return [p_sep + d for d in (0.02, 0.1, -0.1, -0.5)], p_sep
    except Exception:
        return [], None


def _probe(signal_gen, coords_row, WaveformDomainError):
    """Call the stock EMRI signal_gen on one sampling-basis param row.

    Returns (status, finite) where status is 'ok' (template built),
    'domain_error' (WaveformDomainError -> the sampler's -1e300 floor path),
    or 'other:<Exc>' (unexpected). ``finite`` is the template-finiteness for
    the 'ok' case else None.
    """
    try:
        # Stock convention (AnalysisContainer._call_single_model): a 1D param
        # row is unpacked as SEPARATE positional args -- fn(*tuple(row)) -- so
        # params_arr is 1D (leading shape ()); the per-leaf-fill leaf_inds must
        # then be a matching 0-d scalar (leaf 0), not a (1,) vector.
        tmpl = signal_gen(*np.asarray(coords_row, dtype=float),
                          leaf_inds=np.array(0))
    except WaveformDomainError:
        return "domain_error", None
    except Exception as exc:  # any OTHER exception is not the guard path
        return f"other:{type(exc).__name__}:{str(exc)[:90]}", None
    arr = np.asarray(getattr(tmpl, "arr", tmpl))
    return "ok", bool(np.all(np.isfinite(arr.view(float) if np.iscomplexobj(arr) else arr)))


def main() -> int:
    from lisatools.globalfit.stock import erebor
    from lisatools.utils.exceptions import WaveformDomainError

    fit = erebor.full_year_combined(lite=True)
    fit.build()
    em = fit.source_info["emri"]
    signal_gen = em.signal_gen
    inj = np.asarray(em.injection, dtype=float).ravel()
    print(f"[RESULT] emri_guard build=ok p0_inj={inj[P0_COL]:.4f}", flush=True)

    # (1) Sanity: the VALID injection builds a finite template through the
    # same generator (so an always-raising guard can't false-pass).
    vstatus, vfin = _probe(signal_gen, inj, WaveformDomainError)
    valid_ok = (vstatus == "ok") and bool(vfin)
    print(f"[RESULT] emri_guard valid_status={vstatus} valid_finite={int(valid_ok)}",
          flush=True)

    # (2) Guard: separatrix-edge p0 (documented <3-point trigger) first, then
    # low fallbacks. First WaveformDomainError proves the guard path.
    sep_probes, p_sep = _separatrix_probes(inj[A_COL], inj[E_COL])
    print(f"[RESULT] emri_guard p_sep={p_sep if p_sep is not None else 'n/a'}", flush=True)
    probes = [p for p in sep_probes if p > 0] + list(P0_PROBES_FALLBACK)
    guard_fired = 0
    fired_p0 = None
    for p0 in probes:
        bad = inj.copy()
        bad[P0_COL] = float(p0)
        status, _ = _probe(signal_gen, bad, WaveformDomainError)
        floored = status == "domain_error"
        print(f"[RESULT] emri_guard probe p0={p0:.3f} status={status} "
              f"floored={int(floored)}", flush=True)
        if floored:
            guard_fired = 1
            fired_p0 = float(p0)
            break

    # We reached here -> the process SURVIVED every out-of-domain waveform call
    # (a hard FEW exit/segfault would have killed us before this line).
    process_survived = 1
    print(f"[RESULT] emri_guard guard_ll_floor={guard_fired} "
          f"process_survived={process_survived} "
          f"fired_p0={fired_p0 if fired_p0 is not None else 'none'} "
          f"valid_finite={int(valid_ok)}", flush=True)
    # Non-zero exit if the guard never fired or the valid case wasn't finite,
    # so campaign.py marks the check failed rather than silently green.
    return 0 if (guard_fired and valid_ok) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # any UNCAUGHT error here is itself a guard failure
        import traceback
        traceback.print_exc()
        print(f"[RESULT] emri_guard guard_ll_floor=0 process_survived=1 "
              f"error={type(exc).__name__}", flush=True)
        sys.exit(1)
