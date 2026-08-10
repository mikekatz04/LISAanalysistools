#!/usr/bin/env python
"""Staged combined run: noise+foreground search -> GB search -> GB PE, with VGBs.

The run this builds toward: mojito data, ``T_obs`` = 3 months, one composition
carrying **gb + vgb + psd + galfor**, driven by a THREE-stage recipe rather
than the single combined PE stage ``all_sources`` ships with.

    1. ``noise_search``  kind="search"  psd_search + galfor_search
       Both are ``PSDMove(max_logl_mode=True)``; ``run_move_max_likelihood``
       loops internally until the cold-chain max lnL plateaus, and
       ``SearchRecipeStep`` is done on its first check because the criterion
       lives INSIDE the move. So this stage converges the noise model before
       a single GB is subtracted.

    2. ``gb_search``     kind="rj"      psd_pe + galfor_pe + GB search moves
       ``RJRecipeStep`` watches the cold-chain leaf count on ``plateau_branch
       ="gb"`` and advances when it plateaus -- "GB PE when the leaves of
       search converge". Noise keeps sampling underneath (PE mode) so the GB
       search sees a live PSD rather than a frozen one.

    3. ``gb_pe``         kind="pe"      everything, VGBs included
       ``PERecipeStep`` never stops on its own.

Composition is ``all_sources`` MINUS mbh/emri/sobbh -- it already carries vgb,
and ``remove_branch`` drops the moves that sample a removed branch (stock
``bd02f94``), so nothing dangles.

WHY NOT gb_no_fg: it loads ``source_types = ("GB",)`` with a FIXED PSD, i.e.
there is no instrument-noise realization in the data at all. Sampling a PSD
against it would be fitting noise that is not there. This script defaults
``source_types`` to NOISE + GB + VGB so every sampled branch has something to
fit.

Run (one GPU)::

    MOJITO_DATA_PATH=/path/to/L1 USE_GPU=1 GPU_BACKEND=cuda12x GPUS=0 \
    NITER=100 NWALKERS=16 \
    python scripts/fstat_proposal/run_combined_staged.py

Smoke mode (COMBINED_SMOKE=1) shrinks every axis and turns the GB/VGB debug
verifications on -- see the knob table at the bottom of this docstring.

Key env knobs
-------------
    COMBINED_SMOKE=1     small band / few iterations / debug on
    SOURCE_TYPES         comma list (default "NOISE,GB,VGB")
    NITER, NWALKERS      sampler shape
    GB_NTEMPS, VGB_NTEMPS, PSD_NTEMPS     per-branch ladders
    GB_DEBUG=1, VGB_DEBUG=1               residual round-trip verification
    GB_DEBUG_PLOT_BAND   ONE band index -- unset means EVERY band, which at
                         ~1150 cells renders thousands of figures per proposal
    STAGE_SKIP_NOISE=1   start at stage 2 (noise already converged)
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("combined_staged")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True")


def _apply_smoke_defaults() -> None:
    """Shrink every cost axis and arm the debug verifications.

    ``setdefault`` throughout: an explicitly-set knob always wins, so the
    same script serves the smoke and the production run and the diff between
    them is exactly this function.
    """
    smoke = {
        # --- shape ---
        "NITER": "3",
        "NWALKERS": "8",
        "GB_NTEMPS": "6",
        "VGB_NTEMPS": "4",
        "PSD_NTEMPS": "4",
        # --- band: ~2 sub-bands instead of 13 ---
        "GB_MIN_FREQ": "7.0e-3",
        "GB_MAX_FREQ": "7.3e-3",
        "GB_NUM_REPEAT_PROPOSALS": "20",
        "GB_N_SUBBANDS": "256",
        # --- F-stat fit: every stage still RUNS, each does less work ---
        "FSTAT_N_PER_AXIS": "8",
        "FSTAT_COMB_NSKY_MAX": "32",
        "FSTAT_COMB_NSKY_MIN": "8",
        "FSTAT_PEAKS_PER_BAND": "5",
        "FSTAT_CKPT_SECS": "15",   # so checkpointing actually fires
        # --- verification ---
        "GB_DEBUG": "1",
        "VGB_DEBUG": "1",
        # ONE band: unset renders a WDM tile per source per block.
        "GB_DEBUG_PLOT_BAND": "1",
        "GB_DEBUG_PLOT_WALKER": "0",
    }
    for k, v in smoke.items():
        os.environ.setdefault(k, v)


def build_fit():
    from lisatools.globalfit.recipe import Move, Recipe, Stage
    from lisatools.globalfit.stock import erebor

    nwalkers = int(os.environ.get("NWALKERS", "16"))
    fit = erebor.all_sources(nwalkers=nwalkers)

    # Every sampled branch needs a stream: NOISE for psd/galfor, GB, VGB.
    src = os.environ.get("SOURCE_TYPES", "NOISE,GB,VGB")
    fit.general.source_types = tuple(
        s.strip().upper() for s in src.split(",") if s.strip()
    )

    for branch in ("mbh", "emri", "sobbh"):
        fit.remove_branch(branch)

    noise_search = [Move("psd_search", branch="psd"),
                    Move("galfor_search", branch="galfor")]
    noise_pe = [Move("psd_pe", branch="psd"),
                Move("galfor_pe", branch="galfor")]

    stages = []
    if not _env_flag("STAGE_SKIP_NOISE"):
        stages.append(Stage(
            name="noise_search", kind="search", moves=noise_search,
            combine_kwargs=dict(share_temperature_control=False),
        ))
    stages += [
        Stage(
            name="gb_search", kind="rj",
            # Noise keeps sampling in PE mode underneath the GB search, so
            # the search sees a live PSD. Per-stage move-name uniqueness
            # (recipe.py ebd8612) is what lets psd_pe/galfor_pe recur below.
            moves=noise_pe + [Move("rj_prior_search", branch="gb"),
                              Move("in_model", branch="gb")],
            step_kwargs=dict(
                plateau_branch="gb",
                convergence_iter=int(os.environ.get("GB_PLATEAU_ITERS", "5")),
            ),
            combine_kwargs=dict(share_temperature_control=False),
        ),
        Stage(
            name="gb_pe", kind="pe",
            moves=noise_pe + [Move("in_model", branch="gb"),
                              Move("rj_prior", branch="gb"),
                              Move("vgb_pe", branch="vgb")],
            combine_kwargs=dict(share_temperature_control=False),
        ),
    ]
    fit.recipe = Recipe(stages)
    return fit


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if _env_flag("COMBINED_SMOKE"):
        _apply_smoke_defaults()
        print("[combined] SMOKE mode: shrunk axes, GB_DEBUG + VGB_DEBUG on.",
              flush=True)

    fit = build_fit()
    print(f"[combined] branches: {list(fit.branches)}", flush=True)
    for st in fit.recipe.stages:
        print(f"[combined] stage {st.name:>13s} ({st.kind:>6s}): "
              f"{[m.name for m in st.moves]}", flush=True)
    print(f"[combined] source_types: {fit.general.source_types}", flush=True)

    if _env_flag("COMBINED_DRY_RUN"):
        print("[combined] COMBINED_DRY_RUN=1 -- composition only, not built.",
              flush=True)
        return 0

    print("[combined] fit.build() ...", flush=True)
    fit.build()
    print("[combined] running", flush=True)
    fit.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
