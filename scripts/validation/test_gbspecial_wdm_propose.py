#!/usr/bin/env python
"""End-to-end WDM propose() run of the reworked GB special move.

Extends ``scripts/gb_chunked_het/smoke_gb_no_foreground_settings.py`` one
stage further: after the settings construction (Sangria load, WDM domain,
fixed PSD, chunked-het comps, ``build_gb_moves``), it actually calls
``propose()`` on the ``rj_prior`` GB move twice — exercising, on the WDM
path with real data: parent open/close fills, sub-band buffer loading,
without-replacement source picks, RJ birth/death deltas through
``get_ll_wdm``, in-model repeats with the WDM Fisher, band-temperature
swaps, and the final state write-back.

Run with tiny knobs (3-day Tobs, 2 walkers, 2 temps, 3 GB layers):

    /Users/mkatz/miniconda3/envs/deving/bin/python \
        scripts/validation/test_gbspecial_wdm_propose.py
"""

import os
import sys
import traceback

# ----- tiny env knobs (must be set BEFORE importing the settings module) ----
os.environ.setdefault("TOBS_TARGET", str(3 * 86400.0))
os.environ.setdefault("NWALKERS", "2")
os.environ.setdefault("NTEMPS", "2")
os.environ.setdefault("CHUNKED_NT_SUB", "64")
os.environ.setdefault("CHUNKED_N_PAD", "8")
os.environ.setdefault("CHUNKED_N_SPARSE", "64")
os.environ.setdefault("CHUNKED_N_CP_SIG", "16")
os.environ.setdefault("CHUNKED_N_CP_ORBIT", "16")
# keep the repeat count tiny for the smoke
os.environ.setdefault("GB_NUM_REPEAT_PROPOSALS", "3")

import numpy as np

SETTINGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "global_fit_input")
)
if SETTINGS_DIR not in sys.path:
    sys.path.insert(0, SETTINGS_DIR)


class _StubComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1


def main() -> int:
    import gb_no_foreground_global_fit_settings as S

    if not os.path.exists(S.LDC_SOURCE_FILE):
        print(f"BLOCKED: Sangria data file not found: {S.LDC_SOURCE_FILE}")
        return 2

    print("[stage] settings construction")
    curr = S.get_global_fit_settings()

    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.engine import GlobalFitInfo

    priors = {}
    for name in curr.engine_info.branch_names:
        si = curr.source_info.get(name)
        if si is None:
            continue
        priors.update(si.priors)

    gf = GlobalFit(curr, _StubComm())
    state = gf.load_info(priors)
    gb_branch = state.branches["gb"]
    n_on = min(2, gb_branch.inds.shape[-1])
    gb_branch.inds[:, :, :n_on] = True

    print("[stage] setup_acs(rebuild_residuals=True)")
    acs = gf.setup_acs(state, rebuild_residuals=True)

    print("[stage] setup_recipe (builds gb_wdm_comp + rj_prior move)")
    from lisatools.globalfit.recipe import Recipe

    recipe = Recipe()
    S.setup_recipe(recipe, curr.engine_info, curr, acs, priors, state)

    # dig the GB move out of the recipe: entries are dicts with the PE step
    # under "adjust" -> GFCombineMove -> [move]
    move = None
    for entry in recipe.recipe:
        step = entry["adjust"] if isinstance(entry, dict) else entry
        moves = getattr(step, "moves", None)
        if moves:
            combine = moves[0]
            inner = getattr(combine, "moves", [combine])
            move = inner[0]
            break
    assert move is not None, "no GB move found in the recipe"

    # The engine's sampler step normally wires move.periodic from the
    # sampler. This settings file defines no periodicity (latent gap: the
    # sampler-side wiring would also hand the move None), so wire the GB
    # sampling-basis periodic dims here: phi0 (3) 2pi, psi (5) pi, lam (6) 2pi.
    if move.periodic is None:
        from eryn.utils import PeriodicContainer

        move.periodic = PeriodicContainer(
            {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}
        )

    if int(os.environ["GB_NUM_REPEAT_PROPOSALS"]) > 0:
        move.num_repeat_proposals = int(os.environ["GB_NUM_REPEAT_PROPOSALS"])
    print(f"  move: {move.name}  is_rj={move.is_rj_prop}  "
          f"repeats={move.num_repeat_proposals}  n_subbands={move.num_band_preload}")

    model = GlobalFitInfo(
        analysis_container_arr=acs, map_fn=map, random=np.random.RandomState(7)
    )

    for it in range(2):
        print(f"[stage] propose() #{it + 1}")
        state, _ = move.propose(model, state)
        bi = state.sub_states["gb"].band_info
        live = int(state.branches["gb"].inds.sum())
        print(
            f"  log_like finite={bool(np.all(np.isfinite(state.log_like)))}  "
            f"live={live}  "
            f"rj prop/acc={int(bi['band_num_proposed_rj'].sum())}/"
            f"{int(bi['band_num_accepted_rj'].sum())}  "
            f"in-model prop/acc={int(bi['band_num_proposed'].sum())}/"
            f"{int(bi['band_num_accepted'].sum())}  "
            f"swaps={int(bi['band_swaps_proposed'].sum())}/"
            f"{int(bi['band_swaps_accepted'].sum())}"
        )
        if not np.all(np.isfinite(state.log_like)):
            print("[FAIL] non-finite log-likelihood after propose")
            return 1
        if int(bi["band_num_proposed_rj"].sum()) == 0:
            print("[FAIL] no RJ proposals recorded")
            return 1

    print("[ok] WDM end-to-end propose GREEN")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
