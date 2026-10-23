"""Warm starts of a child run from its parent run's store: GB birth mixtures, sources, noise pin.

User ruling 2026-09-14: "All the warmstart code should be part of the
globalfit/lisatools package" -- the former ``scripts/gb/warmstart_*.py``
stage scripts and ``lisatools.sampling.warmstart_proposal`` now live here
(the old locations remain as thin shims).

THE GB WARM START (the ``rj_warm_search`` / ``rj_warm_pe`` RJ birth moves,
armed by ``GB_WARM_START_COMPONENTS``):

* :mod:`.fit_from_store` -- stage 1: cluster the parent store's cold-chain
  GB leaf table (the last ``--last-k`` stored rows x every walker) into
  per-cluster Gaussian mixtures in the observable basis (CLI).
* :mod:`.match_referee` -- stage 2: judge the clusters with real waveform
  matches (within-group coherence, merge test) (CLI).
* :mod:`.referee_apply` -- stage 2.5: apply the verdicts (merge split
  artifacts, flag blends) and write the REFEREED npz (CLI).
* :mod:`.proposal` -- :class:`WarmStartComponents`, the RJ birth
  distribution (``rvs`` / ``logpdf``) that loads the refereed npz.
* :mod:`.build` -- :func:`ensure_warm_start_components`, the check-then-build
  orchestrator (fit -> referee -> apply, in-process, MPI-safe lock) that
  ``recipe.build_gb_moves`` calls; it builds only when the npz is missing.
* :mod:`.basis` -- the one observable-basis map adapter all of the above
  share.
* :mod:`.self_refit` -- the SELF refit (``GB_WARM_SELF=1`` on the 9mo / 12mo
  launchers): the head refits the mixture from the run's OWN last
  ``GB_WARM_SELF_LAST_K`` stored rows at a warm move's entry, and
  :class:`~.self_refit.WarmSelfMixer` draws each propose from the original,
  the full self refit or the filtered self refit.
* :mod:`.para_search` -- the GB parallel residual search (``rj_para_search_pe``,
  ``GB_PARA_SEARCH``; a documented option, OFF in production) and its CLI.

OTHER WARM STARTS:

* :mod:`.sources` -- the MBH / EMRI / SOBBH SOURCE warm start: on a fresh
  store, leaves whose seed-run SNR is above ``SOURCE_WARM_START_SNR_MIN``
  start at the seed store's last cold-chain ladder (``run.py``'s fresh-start
  path; dry-run CLI).
* :mod:`.noise_pin` -- the launcher CLI that reads the seed store's maxlogL
  cold walker and prints ``PSD_START_PARAMS`` / ``GALFOR_START_PARAMS``
  (it uses :func:`.opt_snr.best_logl_noise`; the rest of :mod:`.opt_snr`, an
  optimal-SNR gate for the fitted clusters, is not in the build pipeline).

STORED ROWS. Every ``*_LAST_K`` / ``*_EVERY_*`` count here is in STORED ROWS
of the store, not iterations: a numbered search stage stores 3 rows per
cycle under ``GB_SEARCH_LEGS=1`` (one per leg: after ``in_model``,
``in_model_fstat``, ``in_model_removal``), ``full_pe`` one row per
iteration. The launchers fit the parent's last 300 stored rows
(``GB_WARM_START_LAST_K=300``, the parent's ``full_pe``) and refit the
child from its last 50 (``GB_WARM_SELF_LAST_K=50``).

Example:
    The by-hand build of the parent warm start (what
    :func:`ensure_warm_start_components` runs in-process when the npz is
    missing; the 9mo launcher's values):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.fit_from_store \\
            --store gf_prod_6mo_testing.h5 --last-k 300 --tobs 15552000 \\
            --out warmstart/gf_prod_6mo_v9_4gpu_refereed_fit.npz
        python -m lisatools.globalfit.warmstart.match_referee \\
            --npz warmstart/gf_prod_6mo_v9_4gpu_refereed_fit.npz \\
            --store gf_prod_6mo_testing.h5
        python -m lisatools.globalfit.warmstart.referee_apply \\
            --fit warmstart/gf_prod_6mo_v9_4gpu_refereed_fit.npz \\
            --referee warmstart/gf_prod_6mo_v9_4gpu_refereed_fit_referee.npz \\
            --out warmstart/gf_prod_6mo_v9_4gpu_refereed.npz

See Also:
    ``docs/global-fit-navigation.md`` section 2.8 (the warm-start knob
    tables) and ``docs/gb-para-search.md``.
"""

from .build import ensure_warm_start_components
from .proposal import WarmStartComponents

__all__ = ["WarmStartComponents", "ensure_warm_start_components"]
