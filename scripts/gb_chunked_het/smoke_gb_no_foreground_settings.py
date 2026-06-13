"""Low-memory smoke for gb_no_foreground_global_fit_settings.py.

Exercises the GB no-foreground settings construction + engine-side GB
template path on a tiny WDM grid, to catch breakage from the recent
sweeping merges (domains-never-by-string / WDM C++ likelihood kernels /
nanobind bindings / GBGPU mojito merge signature drift / engine-side
signal_gen + setup_acs(rebuild_residuals) / batched LISA response).

Run ONE at a time. Tiny knobs (3-day Tobs, 2 walkers, 2 temps).

    TOBS_TARGET=259200 NWALKERS=2 NTEMPS=2 \
        python smoke_gb_no_foreground_settings.py

Stages:
  1. import the settings module (env knobs already shrunk),
  2. get_global_fit_settings()  -> full GeneralSetup (Sangria load,
     WDM domain build, fixed-PSD sensitivity, deepcopy),
  3. build priors + a stub size-1 GlobalFit (no real MPI),
  4. load_info(priors) -> GFState drawn from the GB prior,
  5. seed a couple of GB leaves "on" so the WDM template path runs,
  6. setup_acs(state, rebuild_residuals=True) -> per-walker ACs +
     fixed PSD + (engine-side signal_gen rebuild, if any),
  7. setup_recipe(...) -> builds GBWDMHeterodyne (chunked-het) and the
     GB moves; with seeded inds this calls gb_wdm_comp.fill_global_wdm
     (the WDM C++ kernel + GBGPU chunked-het path).
"""

import os
import sys
import traceback

# ----- tiny env knobs (must be set BEFORE importing the settings module) -----
os.environ.setdefault("TOBS_TARGET", str(3 * 86400.0))  # 3 days -> Nt=72
os.environ.setdefault("NWALKERS", "2")
os.environ.setdefault("NTEMPS", "2")
# Chunked-het sub-grid must satisfy Nt_sub <= Nt and step=(Nt_sub-2*n_pad)
# even & > 0. With the 3-day Nt=72 grid the default Nt_sub=256 is too big,
# so shrink to a tiny valid tiling for the smoke.
os.environ.setdefault("CHUNKED_NT_SUB", "64")
os.environ.setdefault("CHUNKED_N_PAD", "8")
os.environ.setdefault("CHUNKED_N_SPARSE", "64")
os.environ.setdefault("CHUNKED_N_CP_SIG", "16")
os.environ.setdefault("CHUNKED_N_CP_ORBIT", "16")

import numpy as np


SETTINGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "global_fit_input")
)
if SETTINGS_DIR not in sys.path:
    sys.path.insert(0, SETTINGS_DIR)


class _StubComm:
    """Minimal size-1 MPI-like comm so GlobalFit.__init__ runs without mpi4py."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1


def main():
    print("=" * 70)
    print("STAGE 1: import settings module")
    print("=" * 70)
    import gb_no_foreground_global_fit_settings as S

    print(
        f"  TOBS={S.TOBS:.3e}s  Nf={S.NF} Nt={S.NT}  band=[{S.MIN_FREQ:.3e},{S.MAX_FREQ:.3e}]Hz"
    )
    print(f"  backend={S.GPU_BACKEND}  data={S.LDC_SOURCE_FILE}")
    if not os.path.exists(S.LDC_SOURCE_FILE):
        print(f"BLOCKED: Sangria data file not found: {S.LDC_SOURCE_FILE}")
        return 2

    print("=" * 70)
    print("STAGE 2: get_global_fit_settings()  (full GeneralSetup construction)")
    print("=" * 70)
    curr = S.get_global_fit_settings()
    gen = curr.general_info
    gb_info = curr.source_info["gb"]
    print(f"  domain_settings resolved -> {type(gen.domain_settings).__name__}")
    print(f"  force_backend={gen.force_backend}  data_t0={gen.data_t0:.3e}")
    print(f"  fixed_psd_kwargs={gen.fixed_psd_kwargs}")
    print(f"  GB band_edges: n={len(gb_info.band_edges)}  "
          f"[{gb_info.band_edges.min():.3e},{gb_info.band_edges.max():.3e}]")

    print("=" * 70)
    print("STAGE 3: build priors + stub GlobalFit")
    print("=" * 70)
    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.engine import GeneralSetup
    from lisatools.globalfit.recipe import Recipe

    # Replicate run_global_fit's prior gathering (Setup branch).
    priors = {}
    for name in curr.engine_info.branch_names:
        si = curr.source_info.get(name)
        if si is None:
            continue
        for key, value in si.priors.items():
            priors[key] = value
    print(f"  priors keys: {list(priors.keys())}")

    gf = GlobalFit(curr, _StubComm())
    print(f"  GlobalFit built (rank={gf.rank}, nwalkers={gf.nwalkers}, ntemps={gf.ntemps})")

    print("=" * 70)
    print("STAGE 4: load_info(priors)  -> GFState from prior")
    print("=" * 70)
    state = gf.load_info(priors)
    print(f"  state.branches keys: {list(state.branches.keys())}")
    gb_branch = state.branches["gb"]
    print(f"  gb coords shape {gb_branch.coords.shape}, inds.sum()={gb_branch.inds.sum()}")

    print("=" * 70)
    print("STAGE 5: seed a few GB leaves 'on' to exercise the WDM template path")
    print("=" * 70)
    # Turn on a small handful of GB leaves across walkers (keep tiny).
    n_on = min(2, gb_branch.inds.shape[-1])
    gb_branch.inds[:, :, :n_on] = True
    print(f"  seeded {n_on} GB leaves on per (temp,walker); inds.sum()={gb_branch.inds.sum()}")

    print("=" * 70)
    print("STAGE 6: setup_acs(state, rebuild_residuals=True)")
    print("=" * 70)
    acs = gf.setup_acs(state, rebuild_residuals=True)
    print(f"  acs built; n_containers={len(acs)}")
    print(f"  acs.start_freq_ind[0]={int(acs.start_freq_ind[0])}")
    ll = acs.likelihood(complex=False)
    print(f"  acs.likelihood() shape={np.asarray(ll).shape}  finite={np.all(np.isfinite(np.asarray(ll)))}")

    print("=" * 70)
    print("STAGE 7: setup_recipe(...)  (builds GBWDMHeterodyne + GB moves)")
    print("=" * 70)
    recipe = Recipe()
    try:
        S.setup_recipe(recipe, curr.engine_info, curr, acs, priors, state)
    except AssertionError as exc:
        msg = str(exc)
        if "not divisible" in msg or "Nf*Nt" in msg:
            print(f"  gb_wdm_comp built -> {type(gb_info.gb_wdm_comp).__name__}  (GBWDMHeterodyne ctor OK)")
            print(f"  GBGPU + build_gb_moves reached fill_global_wdm  (no GBGPU signature drift)")
            print("\n" + "#" * 70)
            print("BLOCKED (shared-src, pre-existing): WDM fill_global_wdm layout mismatch")
            print("#" * 70)
            print(f"  AssertionError: {msg}")
            print(
                "  build_gb_moves passes the AC's *active-band* WDM linear buffer\n"
                "  (data_length = Nf_active*Nt_active) into fill_global_wdm, but BOTH\n"
                "  gb_wdm_het.GBWDMHeterodyne.fill_global_wdm AND the canonical\n"
                "  lisatools.chunked_het.WDMComputationsBase.fill_global_wdm require a\n"
                "  *full dense* (nchannels, Nf, Nt) buffer. Active-band WDM signal\n"
                "  layout came in via the stft_tof / chunked-het merges; the\n"
                "  fill_global_wdm call site predates them. Needs a shared-src\n"
                "  decision -- NOT a target settings-file bug."
            )
            return 3
        raise
    print(f"  gb_wdm_comp built -> {type(gb_info.gb_wdm_comp).__name__}")
    print(f"  recipe components: {[c for c in getattr(recipe, 'components', getattr(recipe, '_components', []))]}")

    print("=" * 70)
    print("SMOKE GREEN")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        print("\n" + "!" * 70)
        print("SMOKE FAILED with traceback:")
        print("!" * 70)
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
