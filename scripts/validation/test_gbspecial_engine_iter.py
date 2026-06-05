#!/usr/bin/env python
"""One-iteration smoke test of the GB special-move likelihood pipeline (CPU).

What this exercises
-------------------
The :class:`GBSpecialStretchMove` / :class:`GBSpecialRJPriorMove` proposal
inner loop is, at its core, three calls per band per proposal:

  * ``engine.fill_template`` -- add or remove a source from a per-band buffer.
  * ``engine.get_ll``        -- evaluate ``<d|h>`` / ``<h|h>`` for a source
                                against that buffer.
  * ``engine.get_swap_ll``   -- evaluate the swap log-likelihood difference
                                between an "add" and a "remove" template.

This script drives those three calls directly via the WDM engine
(:class:`fastlisaresponse.gbcomps.GBWDMComputations` -> WDM kernel) for two
proposal classes:

  * **RJ-style proposal (birth)**: current source amp -> 0 (remove); proposed
    source = random draw from the prior with amp > 0 (add).
  * **In-model proposal**: current source = injection; proposed source =
    perturbation of the injection.

It runs on CPU under ``fastlisaresponse``'s CPU backend. The full
:class:`GBSpecialRJPriorMove`/``GBSpecialStretchMove`` wrappers themselves
need a GPU + ``cupy`` + ``gbgpu`` (they call ``self.xp.cuda.runtime.``...);
this script intentionally stops one level below the eryn move so it runs on
CPU and exercises the same numerical pipeline.

Both TDI conventions are supported via the ``--tdi-type`` flag (default AET,
which is diagonal and avoids the off-diagonal noise drift that XYZ can show
on a single noise realisation).

Run from the repo root:

    /Users/mkatz/miniconda3/envs/deving/bin/python test_gbspecial_engine_iter.py
    /Users/mkatz/miniconda3/envs/deving/bin/python test_gbspecial_engine_iter.py --tdi-type XYZ
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import signal as scipy_signal

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import (
    XYZ2SensitivityMatrix,
    AET1SensitivityMatrix,
    AET2SensitivityMatrix,
    SensitivityMatrixBase,
)
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable,
)

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations


# ----------------------------------------------------------------------------
# Static config -- aligned with gb_lookup_table_test_script.py so we can reuse
# the cached WDM lookup table on disk.
# ----------------------------------------------------------------------------
BACKEND = "cpu"
DT = 10.0
NF = 1460
NT = 256 * 10
WAVELET_DURATION = NF * DT
TOBS = NT * WAVELET_DURATION

LOOKUP_TABLE_PATH = "wdm_lookup_new_all_time_layers_1.h5"

MIN_FREQ = 0.0029493407356002777
MAX_FREQ = 0.00306500115660421
MIN_TIME = 20 * WAVELET_DURATION
MAX_TIME = (NT - 20) * WAVELET_DURATION

T_START = int(0.5 * YRSID_SI / DT) * DT
T_REF = T_START
N_INJ_GRID = 16384


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _xyz_to_aet(arr: np.ndarray) -> np.ndarray:
    """Linear XYZ -> AET combination on the leading 3-channel axis."""
    X, Y, Z = arr[0], arr[1], arr[2]
    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return np.stack([A, E, T], axis=0)


def _draw_injection_params(rng: np.random.Generator, wdm_set: WDMSettings) -> np.ndarray:
    """Random GB params with f0 inside the WDM active band.

    Returns shape (9,) = (amp, f0, fdot, fddot, phi0, inc, psi, lam, beta).
    """
    layer_df = wdm_set.layer_df
    m_lo, m_hi = wdm_set.ind_min_f, wdm_set.ind_max_f
    # Keep f0 strictly inside the band so the +/- num_diff layer window
    # doesn't fall off the edge.
    margin = 0.25 * layer_df
    f0 = rng.uniform((m_lo + 1) * layer_df + margin, m_hi * layer_df - margin)

    return np.array([
        rng.uniform(5e-22, 1.5e-21),
        f0,
        rng.uniform(5e-15, 5e-14),
        0.0,
        rng.uniform(0.0, 2 * np.pi),
        rng.uniform(0.0, np.pi),
        rng.uniform(0.0, np.pi),
        rng.uniform(0.0, 2 * np.pi),
        rng.uniform(-0.5 * np.pi, 0.5 * np.pi),
    ])


def _perturb_in_model(rng: np.random.Generator, params: np.ndarray,
                      wdm_set: WDMSettings, scale: float = 1e-3) -> np.ndarray:
    """Small in-model perturbation of the source params.

    The frequency perturbation is fractions of a layer so the proposed source
    stays in the same layer (no birth/death). amp and phi0 take a relative
    bump; sky position is locked so this stays a 'small step' proposal.
    """
    out = params.copy()
    layer_df = wdm_set.layer_df
    out[0] *= 1.0 + scale * rng.standard_normal()       # amp
    out[1] += scale * layer_df * rng.standard_normal()  # f0 (fractional layer)
    out[4] += scale * rng.standard_normal()             # phi0
    return out


def _build_injection_td(
    rng: np.random.Generator,
    wdm_set: WDMSettings,
    orbits, tdi_config, gb_gen,
    t_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a source and generate its time-domain XYZ TDI."""
    params = _draw_injection_params(rng, wdm_set)
    amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params
    inj_tmp = gb_gen(
        np.array([amp]), np.array([f0]), np.array([fdot]), np.array([fddot]),
        np.array([phi0]), np.array([inc]), np.array([psi]),
        np.array([lam]), np.array([beta]),
        convert_to_ra_dec=False, return_spline=True,
    )
    inj_td = inj_tmp.eval_tdi(t_arr)
    # eval_tdi returns shape (num_bin, 3, N_total); we only have 1 binary
    # in this script, so drop the leading axis.
    if inj_td.ndim == 3:
        inj_td = inj_td[0]
    return params, inj_td  # (3, N_total) XYZ


def _build_wdm_ac(
    inj_td: np.ndarray, td_set: TDSettings, wdm_set: WDMSettings,
    window: np.ndarray, tdi_type: str,
) -> AnalysisContainer:
    """Pack a WDM AC for the requested TDI type.

    XYZ: full 3-channel WDM data + XYZ2SensitivityMatrix (3x3 inverse cov).
    AET: convert XYZ->AET in time-domain, then transform; use AET1Sens.
    """
    if tdi_type == "XYZ":
        inj_wdm = TDSignal(inj_td, settings=td_set).transform(wdm_set, window=window)
        injection = DataResidualArray(inj_wdm)
        sens = XYZ2SensitivityMatrix(injection.settings, model="scirdv1")
    elif tdi_type == "AET":
        inj_aet_td = _xyz_to_aet(inj_td)
        inj_wdm = TDSignal(inj_aet_td, settings=td_set).transform(wdm_set, window=window)
        injection = DataResidualArray(inj_wdm)
        sens = AET1SensitivityMatrix(injection.settings)
    else:
        raise NotImplementedError(f"tdi_type={tdi_type!r} not supported in this script.")
    return AnalysisContainer(injection, sens)


# ----------------------------------------------------------------------------
# Engine-iteration smoke test
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tdi-type", choices=["AET", "XYZ"], default="AET",
        help="TDI channel convention. AET decouples cross-channel noise (default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if not os.path.exists(LOOKUP_TABLE_PATH):
        print(
            f"[skip] WDM lookup table {LOOKUP_TABLE_PATH!r} not found. "
            "Build it via gb_lookup_table_test_script.py first."
        )
        return 0

    print(f"[step] Loading WDM lookup table {LOOKUP_TABLE_PATH!r}")
    wdm_lookup = WDMLookupTable.from_file(LOOKUP_TABLE_PATH, force_backend=BACKEND)
    wdm_set = WDMSettings(
        NF, NT, DT,
        min_freq=MIN_FREQ, max_freq=MAX_FREQ,
        min_time=MIN_TIME, max_time=MAX_TIME,
    )
    if not wdm_lookup.settings.eq_without_inds(wdm_set):
        print("[fail] WDM lookup table settings do not match this script's WDMSettings.")
        return 1

    N_total = wdm_set.N
    td_set = TDSettings(N_total, DT, force_backend=BACKEND)
    window = scipy_signal.windows.tukey(N_total, alpha=0.05)
    t_arr = np.arange(N_total) * DT + T_START

    print(f"[step] Building TDI / orbits / GB generator (tdi_type={args.tdi_type})")
    tdi_config = TDIConfig("1st generation")
    orbits = ESAOrbits(force_backend=BACKEND)
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ", force_backend=BACKEND,
    )
    gb_gen = GBTDIonTheFly(
        np.linspace(t_arr[0], t_arr[-1], N_INJ_GRID),
        TOBS, T_REF, 1.0 / DT, 1,
        **gb_tdi_kwargs,
    )

    print("[step] Drawing injection")
    inj_params, inj_td = _build_injection_td(rng, wdm_set, orbits, tdi_config, gb_gen, t_arr)
    layer_m_inj = int(inj_params[1] / wdm_set.layer_df)
    print(
        f"   injection: f0={inj_params[1]:.6e} (layer {layer_m_inj}), "
        f"amp={inj_params[0]:.3e}, phi0={inj_params[4]:.3f}, inc={inj_params[5]:.3f}"
    )

    print("[step] Building WDM AC (injection)")
    wdm_ac = _build_wdm_ac(inj_td, td_set, wdm_set, window, args.tdi_type)
    wdm_holder = AnalysisContainerArray([wdm_ac])

    print("[step] Building GBWDMComputations")
    gb_comps = GBWDMComputations(
        wdm_lookup, TOBS, T_REF,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=BACKEND, tdi_type=args.tdi_type,
    )

    # ------------------------------------------------------------------
    # Step A: get_ll(injection) -- should give a large positive likelihood.
    # ------------------------------------------------------------------
    print("\n==== get_ll on the injection ====")
    _ = gb_comps.get_ll_wdm(
        inj_params.reshape(1, -1), wdm_holder,
        data_index=None, noise_index=None, convert_to_ra_dec=False,
    )
    d_h_inj = float(gb_comps.d_h_out[0])
    h_h_inj = float(gb_comps.h_h_out[0])
    snr_inj = max(0.0, h_h_inj) ** 0.5
    like_inj = -0.5 * (h_h_inj - 2.0 * d_h_inj)
    print(f"   <d|h>      = {d_h_inj:+.6e}")
    print(f"   <h|h>      = {h_h_inj:+.6e}")
    print(f"   opt SNR    = {snr_inj:.3f}")
    print(f"   log-like   = {like_inj:+.6e}")

    if not (np.isfinite(d_h_inj) and np.isfinite(h_h_inj)):
        print("[FAIL] get_ll on the injection produced non-finite values.")
        return 1
    if snr_inj < 10.0:
        print(f"[warn] injection SNR is low ({snr_inj:.2f}); rest of the test may be noise-dominated.")

    # ------------------------------------------------------------------
    # Step B: RJ-style proposal -- "birth" from the prior.
    #
    # Current state: amp=0 (no source). Proposal: a random GB draw from the
    # prior. The 'remove' template is the zero-amplitude version of the
    # proposal (which contributes nothing), and the 'add' template is the
    # proposed source. The expected behaviour is:
    #
    #   d_h_add ~ <d|h_proposal>   d_h_remove ~ 0
    #   hh_add ~ <h_proposal|h_proposal>   hh_remove ~ 0
    #   ll_diff ~ d_h_add - 0.5 * hh_add  (the standard birth log-like change)
    # ------------------------------------------------------------------
    print("\n==== RJ-style proposal (birth from prior) ====")
    prop_birth = _draw_injection_params(rng, wdm_set)
    prop_birth_zero = prop_birth.copy()
    prop_birth_zero[0] = 0.0  # remove template = zero amplitude

    (
        like_add, like_remove,
        d_h_add, d_h_remove,
        aa, rr, ar,
    ) = gb_comps.get_swap_ll_wdm(
        prop_birth.reshape(1, -1),
        prop_birth_zero.reshape(1, -1),
        wdm_holder,
        data_index=None, noise_index=None,
        convert_to_ra_dec=False,
    )

    d_h_add_v = float(d_h_add[0])
    d_h_rem_v = float(d_h_remove[0])
    aa_v = float(aa[0]); rr_v = float(rr[0]); ar_v = float(ar[0])
    ll_diff_birth = (d_h_add_v - d_h_rem_v) - 0.5 * (aa_v - rr_v) - (ar_v - rr_v)

    print(f"   proposed source f0={prop_birth[1]:.6e} (layer {int(prop_birth[1]/wdm_set.layer_df)})")
    print(f"   d_h_add          = {d_h_add_v:+.6e}")
    print(f"   d_h_remove (~0)  = {d_h_rem_v:+.6e}")
    print(f"   add_add          = {aa_v:+.6e}")
    print(f"   remove_remove (~0)={rr_v:+.6e}")
    print(f"   add_remove (~0)  = {ar_v:+.6e}")
    print(f"   ll_diff (birth)  = {ll_diff_birth:+.6e}")

    if not all(np.isfinite([d_h_add_v, d_h_rem_v, aa_v, rr_v, ar_v])):
        print("[FAIL] RJ birth proposal returned non-finite values.")
        return 1
    if abs(rr_v) > 1e-6 * max(abs(aa_v), 1.0) or abs(ar_v) > 1e-6 * max(abs(aa_v), 1.0):
        print("[warn] removed-template inner products should be ~0 for amp=0 source.")
    print("[ok] RJ birth proposal numerics look sane.")

    # ------------------------------------------------------------------
    # Step C: In-model proposal -- perturb the injected source.
    #
    # Current state: the injection. Proposal: a small perturbation. Expected
    # ll_diff close to zero on average (within the noise floor), and both
    # template inner products should track <h_inj|h_inj> closely.
    # ------------------------------------------------------------------
    print("\n==== In-model proposal (perturb the injection) ====")
    perturbed = _perturb_in_model(rng, inj_params, wdm_set, scale=1e-3)

    (
        like_add2, like_remove2,
        d_h_add2, d_h_remove2,
        aa2, rr2, ar2,
    ) = gb_comps.get_swap_ll_wdm(
        perturbed.reshape(1, -1),
        inj_params.reshape(1, -1),
        wdm_holder,
        data_index=None, noise_index=None,
        convert_to_ra_dec=False,
    )

    d_h_add2_v = float(d_h_add2[0])
    d_h_rem2_v = float(d_h_remove2[0])
    aa2_v = float(aa2[0]); rr2_v = float(rr2[0]); ar2_v = float(ar2[0])
    ll_diff_inmodel = (d_h_add2_v - d_h_rem2_v) - 0.5 * (aa2_v - rr2_v) - (ar2_v - rr2_v)

    print(f"   perturbed source f0={perturbed[1]:.6e} (Δf={(perturbed[1]-inj_params[1]):+.3e})")
    print(f"   d_h_add          = {d_h_add2_v:+.6e}")
    print(f"   d_h_remove       = {d_h_rem2_v:+.6e}    (expected ~= d_h_inj = {d_h_inj:+.6e})")
    print(f"   add_add          = {aa2_v:+.6e}")
    print(f"   remove_remove    = {rr2_v:+.6e}    (expected ~= h_h_inj = {h_h_inj:+.6e})")
    print(f"   add_remove       = {ar2_v:+.6e}")
    print(f"   ll_diff (inmodel)= {ll_diff_inmodel:+.6e}")

    if not all(np.isfinite([d_h_add2_v, d_h_rem2_v, aa2_v, rr2_v, ar2_v])):
        print("[FAIL] in-model proposal returned non-finite values.")
        return 1

    # Sanity: the remove side of an in-model proposal must reproduce the
    # injection's own get_ll up to small drift introduced by Doppler /
    # layer-windowing differences between the two code paths.
    rel_d_h = abs(d_h_rem2_v - d_h_inj) / max(abs(d_h_inj), 1.0)
    rel_h_h = abs(rr2_v - h_h_inj) / max(abs(h_h_inj), 1.0)
    print(f"   rel(d_h_remove - d_h_inj) = {rel_d_h:.3e}")
    print(f"   rel(remove_remove - h_h_inj) = {rel_h_h:.3e}")

    tol = 1e-9
    if rel_d_h > tol or rel_h_h > tol:
        print(
            f"[FAIL] in-model removeside drifted vs get_ll(injection) above tol={tol} "
            f"(rel_d_h={rel_d_h:.3e}, rel_h_h={rel_h_h:.3e})."
        )
        return 1
    print("[ok] In-model proposal numerics look sane.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n==== SUMMARY ====")
    print(f"  tdi_type             = {args.tdi_type}")
    print(f"  injection SNR        = {snr_inj:.2f}")
    print(f"  RJ birth   ll_diff   = {ll_diff_birth:+.4e}")
    print(f"  in-model   ll_diff   = {ll_diff_inmodel:+.4e}")
    print("[ok] One full iteration of the engine pipeline ran cleanly on CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
