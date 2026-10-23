"""GB parallel residual search: one GB per sub-band per cold walker -> a birth GMM.

User ruling (Mike, 2026-10-16), the design this module implements: "build the
paraensemble setup to sample 1 GB per sub-band per walker (cold-chain only).
You can operate directly on the residuals, no need to put them in multiple
buffers. ... run the parallel-tempered fixed-dimensional MCMC on each cold
(walker, band) until they all converge per sub-band in max logL. Then determine
if any source has been found at all above optimal SNR of 5 or 7 (probably 5),
turn each sub-band's surviving source into a GMM, combine them into a large
GMM, and then we slot that in and sample from it. There could be multiple
sources per sub-band at the low snrs but try to examine that."

WHY. On the 9mo run (job 773, rows 871-1008) ``full_pe`` adds ~0.4 resolved GBs
per row per walker and the RJ cold birth acceptance is ~1e-3 per proposal: the
prior and the galaxy-wide F-stat grid do not know what a walker's OWN residual
still holds. This search finds, per walker and per sub-band, the next source
that residual contains, and pools the finds of every walker into one birth
container (``docs/gb-para-search.md``).

THE PIECES (one module, no move machinery):

* :class:`ParaSearchConfig` -- the knobs (``GBSettings.para_search_*``, env
  ``GB_PARA_SEARCH_*``).
* :class:`BandBoxPrior` -- the per-group f0 / third-column box, the
  ``prior_transform_fn`` contract of Eryn's ``ParaEnsembleSampler`` (the old
  ``PriorTransformFn`` of ``gbspecialstretch.py``, rewritten without its
  ``breakpoint()`` and the undefined ``xp``).
* :class:`ResidualScorer` -- the F-statistic and the full likelihood of
  arbitrary GB rows against ONE residual row of a WDM holder: the shared-
  reference sig-het F-stat (``fstat_gridfit.build_sighet_call_fstat``) or the
  exact chunked one (``get_fstat_ll_wdm``), and ``get_ll_wdm``'s
  ``d_h_out`` / ``h_h_out`` (the comps are built with ``d_d = 0``).
* :class:`ParaSearchBuilder` -- stage 1 (F-stat PT-MCMC on every sub-band of a
  band group, extrinsics maximised, Gibbs on f0 / Mc / sky, per-band
  convergence of the cold-rung running max switches groups off), the cluster
  split (2+ sources per band), gate A (``sqrt(2 F) >= snr_min``), stage 2 (full
  likelihood, every column sampled) and gate B (opt SNR >= snr_min and
  det >= det_frac x opt).
* :func:`merge_finds` -- the head's merge across walkers and band edges.
* :func:`write_epoch` -- the npz in the ``WarmStartComponents`` OBSERVABLE
  schema (exactly what ``fit_from_store.run`` writes) + ``DONE.json`` LAST.
* :class:`ParaSearchHolder` -- the move's RJ container (poll / adopt /
  directive; ``rvs`` / ``logpdf`` of the adopted epoch).
* the CLI (``python -m lisatools.globalfit.warmstart.para_search``) -- the
  offline search on a store snapshot's residual.

Reproducibility: Eryn's ``ParaEnsembleSampler`` and ``ProbDistContainer`` draw
from the GLOBAL numpy / cupy random state. Every search runs inside
:func:`seeded_global_rng`, which seeds it and restores the caller's state on
exit, so a search is reproducible from its seed and the run's own global
streams are untouched.

WHERE IT SITS. A documented stock option, OFF by default and OFF in the
9mo / 12mo production runs (``GB_PARA_SEARCH`` unset -> ``gb.para_search``
False -> no move object exists). With ``GB_PARA_SEARCH=1``
``recipe.build_gb_moves`` builds the ``rj_para_search_pe`` move
(``gbspecialstretch.GBParaSearchRJMove``: its RJ container is a
:class:`ParaSearchHolder`, its scorer a :class:`ResidualScorer` subclass)
and :func:`compose_para_search` slots it after ``rj_prior_pe`` in every PE
stage carrying that move (it proposes nothing outside the PE stages and in
``replica_pe``, :func:`para_search_stage_ok`). Every ``every_pe`` PE
iterations (:class:`ParaSearchClock`)
each compute rank searches its own walker's residual
(:meth:`ParaSearchBuilder.run`), the head merges the finds
(:func:`merge_finds`) and writes an epoch under
``<run_dir>/warmstart/parasearch/epoch_NNNN/`` (:func:`write_epoch`, the
epoch protocol of :mod:`.self_refit`), and the move's RJ births draw from the
adopted epoch. Units: the sampling basis has f0 in mHz; band edges, the
scorer and the finds' ``f0_hz`` are in Hz.

KNOBS (:class:`ParaSearchConfig`; ``GBSettings.para_search_*`` fields with
the same defaults, env ``GB_PARA_SEARCH_<FIELD>``): ``EVERY_PE`` 100,
``NTEMPS`` 8, ``NWALKERS`` 16 (even, >= 4), ``SNR_MIN`` 5.0, ``DET_FRAC``
0.7, ``CONVERGE_TOL`` 0.5, ``CONVERGE_WINDOW`` 50, ``MIN_STEPS`` 100,
``MAX_STEPS`` 600, ``STAGE2_STEPS`` 100, ``SPLIT_BINS`` 4.0,
``CLUSTER_RUNGS`` 2, ``SCORER`` ``auto``, ``GROUP_BANDS`` 0, ``INIT``
``comb``, ``INIT_NSKY_MIN`` 8, ``INIT_NSKY_MAX`` 64, ``STAGE2_NTEMPS`` 4,
``MAX_FINDS`` 512. The sig-het scorer also reads
``FSTAT_SIGHET_REF_SPACING_HZ`` (reference spacing override),
``FSTAT_SIGHET_REF_BLOCK`` (512) and ``FSTAT_BATCH``.

Example:
    The offline CLI on one walker of a store snapshot (needs
    ``<store>_residual_snapshot.npz`` beside the store; source the run's
    launcher exports first so the stock GB settings resolve):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.para_search \\
            --store gf_prod_9mo_testing.h5 --walker 0 --bands 400:480 \\
            --out para_cli_w0 --backend cuda12x --gpu 0 --merge-epoch

See Also:
    ``docs/gb-para-search.md`` (the design, the walls, the CLI runbook).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import socket
import time
import typing
import warnings

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BandBoxPrior",
    "EPOCH_NPZ",
    "FINDS_NPZ",
    "PARA_SEARCH_MOVE",
    "ParaSearchClock",
    "band_restricted_priors_in",
    "compose_para_search",
    "empty_finds",
    "load_finds_npz",
    "ParaSearchBuilder",
    "ParaSearchConfig",
    "ParaSearchHolder",
    "ResidualScorer",
    "band_groups",
    "build_search_prior",
    "default_search_priors_in",
    "greedy_peaks",
    "passes_gate_a",
    "passes_gate_b",
    "merge_finds",
    "para_search_epoch_root",
    "para_search_stage_ok",
    "seeded_global_rng",
    "write_epoch",
]

#: the move's stage / recipe name
PARA_SEARCH_MOVE = "rj_para_search_pe"
#: the merged birth container (WarmStartComponents observable schema)
EPOCH_NPZ = "parasearch.npz"
#: every walker's raw finds (diagnostics; small)
FINDS_NPZ = "finds.npz"
#: the build clock journal under the epoch root
CLOCK_JSON = "clock.json"
#: GB sampling basis this module works in (the warm-start fitter's columns)
SAMPLING_COLUMNS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                    "sin_delta", "fdot_astro_ratio"]
#: stage 1 samples f0, the third column and the sky; the F-stat maximises the
#: four extrinsics and is flat along the (Mc, ratio) split, so the ratio is
#: HELD AT 0 (one target for the whole ensemble: the stretch move is only valid
#: when every walker samples the same distribution, which a per-walker ratio
#: drawn from the prior -- the old move's choice -- breaks).
STAGE1_GIBBS_COLS = (1, 2, 6, 7)
F0_COL, THIRD_COL, RATIO_COL = 1, 2, 8
#: GB periodic columns of the sampling basis (phi0, psi, alpha)
GB_PERIODIC = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}
#: cap on (band, cluster) finds per band and walker (stage-2 cost bound)
MAX_FINDS_PER_BAND = 4
#: stage-2 seeds per find whose extrinsic maximisers are computed exactly
N_STAGE2_SEEDS = 16
#: stage-2 cold samples re-scored per find for the median opt / det SNR
N_RESCORE = 32
#: the GMM fit (fit_from_store.fit_cluster_gmms): resample size, component
#: cap, members per allowed component
GMM_SAMPLES = 2048
GMM_MAX_COMP = 6
GMM_MIN_MEMBERS = 25
#: members of one find kept for the merged union (equal weight per walker)
UNION_PER_FIND = 800


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
def _env(name, default, cast):
    """Env ``name`` cast by ``cast``; unset or empty -> ``default``.

    For ``cast=bool`` the values ``0`` / ``false`` / ``False`` / ``no`` /
    ``off`` are False and anything else True.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if cast is bool:
        return raw not in ("0", "false", "False", "no", "off", "")
    return cast(raw)


@dataclasses.dataclass
class ParaSearchConfig:
    """The search knobs (``GBSettings.para_search_*`` / ``GB_PARA_SEARCH_*``).

    ``ntemps`` x ``nwalkers`` chains per sub-band (one GB each), the per-band
    convergence rule (stop when the cold-rung running max has not improved by
    more than ``converge_tol`` over ``converge_window`` steps, after at least
    ``min_steps`` and at most ``max_steps``), the detection gate
    (``snr_min``, ``det_frac``), the stage-2 length, the cluster split radius
    in frequency bins, the warm rungs pooled with the cold one for the split,
    the stage-1 scorer and the band-group size. Validated (and normalized) at
    construction by :meth:`validate`.

    Attributes:
        every_pe (int): PE iterations between two searches. Default 100.
        ntemps (int): Stage-1 rungs per sub-band. Default 8.
        nwalkers (int): Walkers per rung (even, >= 4). Default 16.
        snr_min (float): Detection floor of both gates. Default 5.0.
        det_frac (float): Gate B: ``det >= det_frac x opt``. Default 0.7.
        converge_tol (float): Stage-1 per-band convergence: the cold-rung
            running max gains at most this over ``converge_window`` steps.
            Default 0.5.
        converge_window (int): Default 50.
        min_steps (int): Default 100.
        max_steps (int): Default 600.
        stage2_steps (int): Full-likelihood steps per find (>= 2). Default
            100.
        split_bins (float): Cluster split / merge radius in ``1/Tobs`` bins.
            Default 4.0.
        cluster_rungs (int): Rungs pooled for the split (clipped to
            ``[1, ntemps]``). Default 2.
        scorer (str): ``auto`` / ``sighet`` / ``chunked``. Default ``auto``.
        group_bands (int): Sub-bands per sampler run (0 = by the sig-het
            reference block). Default 0.
        init (str): ``comb`` (seed from an F-stat comb) or ``prior``. Default
            ``comb``.
        init_nsky_min (int): Default 8.
        init_nsky_max (int): Default 64.
        stage2_ntemps (int): Stage-2 rungs (``<= 0`` = ``ntemps``). Default 4.
        max_finds (int): Stage-1 finds kept for stage 2 per walker (``<= 0``
            = no cap). Default 512.
        seed (int or None): Search seed (``None`` = fresh). Not an env knob.

    Example:
        >>> from lisatools.globalfit.warmstart.para_search import ParaSearchConfig
        >>> cfg = ParaSearchConfig(nwalkers=8, snr_min=7.0)
        >>> cfg.stage2_ntemps, cfg.cluster_rungs
        (4, 2)
    """

    every_pe: int = 100
    ntemps: int = 8
    nwalkers: int = 16
    snr_min: float = 5.0
    det_frac: float = 0.7
    converge_tol: float = 0.5
    converge_window: int = 50
    min_steps: int = 100
    max_steps: int = 600
    stage2_steps: int = 100
    split_bins: float = 4.0
    cluster_rungs: int = 2
    scorer: str = "auto"
    group_bands: int = 0
    init: str = "comb"
    init_nsky_min: int = 8
    init_nsky_max: int = 64
    stage2_ntemps: int = 4
    max_finds: int = 512
    seed: typing.Optional[int] = None

    #: (field, env name, cast) -- env knob = the capitalised attribute name
    _ENV = (
        ("every_pe", "GB_PARA_SEARCH_EVERY_PE", int),
        ("ntemps", "GB_PARA_SEARCH_NTEMPS", int),
        ("nwalkers", "GB_PARA_SEARCH_NWALKERS", int),
        ("snr_min", "GB_PARA_SEARCH_SNR_MIN", float),
        ("det_frac", "GB_PARA_SEARCH_DET_FRAC", float),
        ("converge_tol", "GB_PARA_SEARCH_CONVERGE_TOL", float),
        ("converge_window", "GB_PARA_SEARCH_CONVERGE_WINDOW", int),
        ("min_steps", "GB_PARA_SEARCH_MIN_STEPS", int),
        ("max_steps", "GB_PARA_SEARCH_MAX_STEPS", int),
        ("stage2_steps", "GB_PARA_SEARCH_STAGE2_STEPS", int),
        ("split_bins", "GB_PARA_SEARCH_SPLIT_BINS", float),
        ("cluster_rungs", "GB_PARA_SEARCH_CLUSTER_RUNGS", int),
        ("scorer", "GB_PARA_SEARCH_SCORER", str),
        ("group_bands", "GB_PARA_SEARCH_GROUP_BANDS", int),
        ("init", "GB_PARA_SEARCH_INIT", str),
        ("init_nsky_min", "GB_PARA_SEARCH_INIT_NSKY_MIN", int),
        ("init_nsky_max", "GB_PARA_SEARCH_INIT_NSKY_MAX", int),
        ("stage2_ntemps", "GB_PARA_SEARCH_STAGE2_NTEMPS", int),
        ("max_finds", "GB_PARA_SEARCH_MAX_FINDS", int),
    )

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check the knobs; normalize ``scorer``, ``init``, ``cluster_rungs``, ``stage2_ntemps``.

        Raises:
            ValueError: On an odd or < 4 ``nwalkers``, ``ntemps < 1``, a bad
                step / window / split / ``det_frac`` / ``init_nsky_*`` value,
                or an unknown ``scorer`` / ``init``.
        """
        if int(self.nwalkers) < 4 or int(self.nwalkers) % 2:
            raise ValueError(
                f"GB_PARA_SEARCH_NWALKERS={self.nwalkers}: needs an even "
                "number >= 4 (the stretch move splits the walkers in half).")
        if int(self.ntemps) < 1:
            raise ValueError(f"GB_PARA_SEARCH_NTEMPS={self.ntemps} must be >= 1.")
        if not 1 <= int(self.min_steps) <= int(self.max_steps):
            raise ValueError(
                f"GB_PARA_SEARCH_MIN_STEPS={self.min_steps} / MAX_STEPS="
                f"{self.max_steps}: need 1 <= min <= max.")
        if int(self.converge_window) < 1:
            raise ValueError("GB_PARA_SEARCH_CONVERGE_WINDOW must be >= 1.")
        if int(self.stage2_steps) < 2:
            raise ValueError("GB_PARA_SEARCH_STAGE2_STEPS must be >= 2.")
        if not float(self.split_bins) > 0:
            raise ValueError("GB_PARA_SEARCH_SPLIT_BINS must be > 0.")
        if not 0.0 <= float(self.det_frac) <= 1.0:
            raise ValueError("GB_PARA_SEARCH_DET_FRAC must lie in [0, 1].")
        if str(self.scorer).lower() not in ("auto", "sighet", "chunked"):
            raise ValueError(
                f"GB_PARA_SEARCH_SCORER={self.scorer!r}: 'auto', 'sighet' or "
                "'chunked'.")
        self.scorer = str(self.scorer).lower()
        if str(self.init).lower() not in ("comb", "prior"):
            raise ValueError(
                f"GB_PARA_SEARCH_INIT={self.init!r}: 'comb' or 'prior'.")
        self.init = str(self.init).lower()
        if not 1 <= int(self.init_nsky_min) <= int(self.init_nsky_max):
            raise ValueError("GB_PARA_SEARCH_INIT_NSKY_MIN/_MAX: need 1 <= min <= max.")
        self.cluster_rungs = max(1, min(int(self.cluster_rungs), int(self.ntemps)))
        if int(self.stage2_ntemps) <= 0:
            self.stage2_ntemps = int(self.ntemps)
        self.stage2_ntemps = int(self.stage2_ntemps)
        self.max_finds = int(self.max_finds)

    @classmethod
    def from_env(cls, **overrides) -> "ParaSearchConfig":
        """Defaults <- ``GB_PARA_SEARCH_*`` env <- explicit ``overrides``."""
        kw = {}
        for field, env, cast in cls._ENV:
            default = getattr(cls, field)
            kw[field] = _env(env, default, cast)
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    @classmethod
    def from_settings(cls, gb_info, **overrides) -> "ParaSearchConfig":
        """From a ``GBSettings``-like object's ``para_search_*`` fields (the
        fields already resolved their env defaults); a missing field falls
        back to the env / hard default."""
        kw = {}
        for field, env, cast in cls._ENV:
            val = getattr(gb_info, f"para_search_{field}", None)
            kw[field] = _env(env, getattr(cls, field), cast) if val is None else cast(val)
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    def as_dict(self) -> dict:
        """Every field as a plain dict (the CLI report's knob line)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def para_search_stage_ok(stage_kind, stage_name=None) -> bool:
    """PE stages only in v1 (search stages: the move proposes nothing).

    Returns:
        bool: True for ``stage_kind == "pe"`` unless the stage is
        ``replica_pe``.
    """
    return str(stage_kind or "") == "pe" and str(stage_name or "") != "replica_pe"


def compose_para_search(recipe, *, anchor="rj_prior_pe") -> list:
    """Put ``rj_para_search_pe`` right after ``anchor`` in every PE stage.

    Called by ``setup_gb_moves`` when ``gb.para_search`` is on (the stage
    lists are composed by the driver, which knows nothing of this move).
    Idempotent; returns the stage names it was added to (a warning when
    none carries the anchor).

    Args:
        recipe: The stock ``Recipe`` (``stages``, ``add_move``,
            ``move_names``).
        anchor (str, optional): The move it goes after. Default
            ``"rj_prior_pe"``.

    Returns:
        list of str: The stages it was added to in this call.
    """
    added = []
    for st in list(getattr(recipe, "stages", [])):
        names = [m.name for m in st.moves]
        if (getattr(st, "kind", None) == "pe" and anchor in names
                and PARA_SEARCH_MOVE not in names):
            recipe.add_move(PARA_SEARCH_MOVE, stage=st.name, after=anchor,
                            branch="gb")
            added.append(st.name)
    if not added and PARA_SEARCH_MOVE not in recipe.move_names():
        logger.warning("gb.para_search is on but no PE stage carries %s; %s is "
                       "not composed into the recipe.", anchor, PARA_SEARCH_MOVE)
    elif added:
        logger.info("[GB_PARA_SEARCH] %s composed after %s in stage(s) %s",
                    PARA_SEARCH_MOVE, anchor, added)
    return added


def para_search_epoch_root(store_path: str) -> str:
    """``<run_dir>/warmstart/parasearch`` (the run dir is the store's directory)."""
    return os.path.join(os.path.dirname(os.path.abspath(str(store_path))),
                        "warmstart", "parasearch")


# ---------------------------------------------------------------------------
# global RNG scope
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def seeded_global_rng(seed, xp=np):
    """Seed the GLOBAL numpy (and cupy) random state; restore it on exit.

    ``ParaEnsembleSampler`` (``np.random.shuffle``, ``xp.random.rand``) and
    ``ProbDistContainer.rvs`` (``xp.random.uniform``) use the global streams;
    without this a search would both be unreproducible and shift every later
    draw of the run that uses them. ``seed=None`` draws a fresh seed (still
    restored afterwards).

    Args:
        seed (int or None): The seed (taken mod 2**32).
        xp (module, optional): ``numpy`` (default) or ``cupy``; cupy's global
            state is seeded and restored too.

    Yields:
        int: The seed used.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    seed = int(seed) % (2 ** 32)
    np_state = np.random.get_state()
    cp_state = None
    np.random.seed(seed)
    if xp is not np:
        cp_state = xp.random.get_random_state()
        xp.random.set_random_state(xp.random.RandomState(seed))
    try:
        yield seed
    finally:
        np.random.set_state(np_state)
        if cp_state is not None:
            xp.random.set_random_state(cp_state)


def _host(a):
    """cupy-or-numpy -> numpy."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------
class BandBoxPrior:
    """Per-group f0 / third-column box for Eryn's ``ParaEnsembleSampler``.

    The prior container carries UNIT-CUBE uniforms on the f0 (1, mHz) and the
    third (2, Mc or fdot) columns; this maps them to each group's own box
    ``[f_lo, f_hi] x [third_lo, third_hi]`` and adds the flat log density of
    the box. ``groups_running`` is the sampler's array of running group
    INDICES (``None`` = all).

    Args:
        f_lo_mhz, f_hi_mhz (array-like): ``(n_groups,)`` f0 box [mHz].
        third_lo, third_hi (array-like): ``(n_groups,)`` third-column box.
        xp (module, optional): ``numpy`` (default) or ``cupy``.

    Raises:
        ValueError: When a box has ``hi <= lo``.
    """

    def __init__(self, f_lo_mhz, f_hi_mhz, third_lo, third_hi, xp=np):
        self.xp_name = "cupy" if xp is not np else "numpy"
        self.f_lo = xp.asarray(np.asarray(f_lo_mhz, dtype=float))
        self.f_hi = xp.asarray(np.asarray(f_hi_mhz, dtype=float))
        self.t_lo = xp.asarray(np.asarray(third_lo, dtype=float))
        self.t_hi = xp.asarray(np.asarray(third_hi, dtype=float))
        if np.any(_host(self.f_hi) <= _host(self.f_lo)) or np.any(
                _host(self.t_hi) <= _host(self.t_lo)):
            raise ValueError("BandBoxPrior: every box needs hi > lo.")

    @property
    def xp(self):
        """``cupy`` or ``numpy`` (from the stored name)."""
        if self.xp_name == "cupy":
            import cupy as cp

            return cp
        return np

    def _sel(self, groups_running):
        """The group indices to use (all when ``None``)."""
        if groups_running is None:
            return self.xp.arange(len(self.f_lo))
        return groups_running

    def adjust_logp(self, logp, groups_running):
        """Add each group's flat box log density to ``logp`` ``(groups, nt, nw)``, in place."""
        g = self._sel(groups_running)
        lp = (-self.xp.log(self.f_hi[g] - self.f_lo[g])
              - self.xp.log(self.t_hi[g] - self.t_lo[g]))
        logp[:] += lp[:, None, None]
        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        """Map the f0 / third columns of ``coords`` ``(groups, nt, nw, 9)`` to the unit cube."""
        g = self._sel(groups_running)
        lo, hi = self.f_lo[g][:, None, None], self.f_hi[g][:, None, None]
        coords[:, :, :, F0_COL] = (coords[:, :, :, F0_COL] - lo) / (hi - lo)
        lo, hi = self.t_lo[g][:, None, None], self.t_hi[g][:, None, None]
        coords[:, :, :, THIRD_COL] = (coords[:, :, :, THIRD_COL] - lo) / (hi - lo)

    def transform_from_prior_basis(self, coords, groups_running):
        """Map unit-cube f0 / third columns back onto each group's box, in place."""
        g = self._sel(groups_running)
        lo, hi = self.f_lo[g][:, None, None], self.f_hi[g][:, None, None]
        coords[:, :, :, F0_COL] = coords[:, :, :, F0_COL] * (hi - lo) + lo
        lo, hi = self.t_lo[g][:, None, None], self.t_hi[g][:, None, None]
        coords[:, :, :, THIRD_COL] = coords[:, :, :, THIRD_COL] * (hi - lo) + lo


def default_search_priors_in(*, dist_lims, ratio_max, use_cupy=False):
    """The search prior for a standalone search (the CLI): uniform boxes, f0 / Mc unit-cube.

    f0 and Mc are unit-cube placeholders (mapped per group by
    :class:`BandBoxPrior`); the other columns are uniform over their full
    ranges.

    Args:
        dist_lims (tuple): ``(lo, hi)`` distance [kpc].
        ratio_max (float): ``fdot_astro_ratio`` half-width.
        use_cupy (bool, optional): Default ``False``.

    Returns:
        dict: ``priors_in`` keyed by the 9 sampling column names.
    """
    from eryn.prior import UniformDistribution

    def u(a, b):
        return UniformDistribution(float(a), float(b), use_cupy=use_cupy)

    return {
        "dist": u(*dist_lims), "f0": u(0.0, 1.0), "Mc": u(0.0, 1.0),
        "phi0": u(0.0, 2.0 * np.pi), "cos_iota": u(-1.0, 1.0),
        "psi": u(0.0, np.pi), "alpha": u(0.0, 2.0 * np.pi),
        "sin_delta": u(-1.0, 1.0),
        "fdot_astro_ratio": u(-float(ratio_max), float(ratio_max)),
    }


def band_restricted_priors_in(priors_in_global: dict, use_cupy=False) -> dict:
    """The run's ``priors_in`` with the f0 + third columns swapped for unit-cube
    uniforms (the old move's ``_band_restricted_priors_in``, free-standing).

    Keeps the run's column layout (string keys = insertion order): the joint
    ``("f0", "Mc")`` prior (or the single ``f0`` / ``fdot`` / ``Mc`` keys) is
    replaced IN PLACE by two unit-cube singles; every other prior -- the
    ``fdot_astro_ratio`` box, a joint galaxy (dist, sky) prior -- is carried
    through unchanged. Used by ``GBParaSearchRJMove`` on the run's GB prior.

    Args:
        priors_in_global (dict): The run's GB ``priors_in`` (deep-copied).
        use_cupy (bool, optional): Default ``False``.

    Returns:
        dict: The search ``priors_in`` (insertion order kept).
    """
    from copy import deepcopy

    from eryn.prior import UniformDistribution

    src = deepcopy(priors_in_global)
    out = {}
    for key, dist in src.items():
        if key in ("f0", "fdot", "Mc", ("f0", "Mc"), ("f0", "fdot")):
            if "f0" not in out:
                out["f0"] = UniformDistribution(0.0, 1.0, use_cupy=use_cupy)
                out["Mc"] = UniformDistribution(0.0, 1.0, use_cupy=use_cupy)
        else:
            out[key] = dist
    return out


def build_search_prior(priors_in: dict, *, input_basis=SAMPLING_COLUMNS,
                       use_cupy=False):
    """``ProbDistContainer`` over the 9 sampling columns, keys reordered by NAME (``input_basis``).

    The reorder matters because a joint ``("dist", "alpha", "sin_delta")``
    prior grabbed consecutive columns by insertion order.

    Args:
        priors_in (dict): :func:`band_restricted_priors_in` or
            :func:`default_search_priors_in` output.
        input_basis (list of str, optional): Column order. Default
            :data:`SAMPLING_COLUMNS`.
        use_cupy (bool, optional): Default ``False``.

    Returns:
        eryn.prior.ProbDistContainer: The prior (``return_gpu=True``).
    """
    from eryn.prior import ProbDistContainer

    cont = ProbDistContainer(priors_in, return_gpu=True, use_cupy=use_cupy)
    if list(cont.key_order) != list(input_basis):
        cont.reset_key_order(list(input_basis))
    return cont


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------
class ResidualScorer:
    """F-stat and full likelihood of GB rows against ONE residual row.

    ``chunked`` is the chunked-heterodyne comp (``GBWDMComputations``; the
    sig-het wrapper's ``.chunked`` delegate), ``holder`` anything exposing
    ``linear_data_arr`` / ``linear_psd_arr`` (an ``AnalysisContainerArray``,
    or a one-row holder), ``row`` / ``noise_row`` its residual / noise rows.
    ``sighet`` (a ``GBSignalHetComputations`` built with ``v4_knots > 0``)
    enables the shared-reference F-stat; ``mode`` = ``"sighet"`` /
    ``"chunked"`` / ``"auto"`` (sighet when available).

    Rows are PHYSICAL ``(A, f0 [Hz], fdot, fddot, phi0, iota, psi, alpha,
    delta)``. The full likelihood's comps carry ``d_d = 0``, so
    ``d_h_out`` / ``h_h_out`` are ``<r|h>`` / ``<h|h>`` of the residual.

    Args:
        chunked (GBWDMComputations): The chunked comp.
        holder: The WDM holder (``linear_data_arr`` / ``linear_psd_arr``).
        row: Residual row. Default ``0``.
        noise_row (int, optional): Noise row. Default ``None`` = ``row``.
        sighet (GBSignalHetComputations, optional): Enables the shared-
            reference F-stat when built with ``v4_knots > 0``.
        tobs: Observation time [s].
        mode: ``auto`` (default) / ``sighet`` / ``chunked``.
        batch: Full-likelihood batch. Default ``None`` = the ``FSTAT_BATCH``
            knob.

    Attributes:
        mode (str): The resolved scorer (``sighet`` or ``chunked``).
        n_fstat (int): Exact F-stat rows scored.
        n_full (int): Full-likelihood rows scored.

    Raises:
        ValueError: ``mode="sighet"`` without a usable sig-het comp.
    """

    def __init__(self, chunked, holder, *, row: int = 0, noise_row=None,
                 sighet=None, tobs: float, mode: str = "auto",
                 batch: typing.Optional[int] = None):
        self.chunked = chunked
        self.holder = holder
        self.row = int(row)
        self.noise_row = self.row if noise_row is None else int(noise_row)
        self.tobs = float(tobs)
        mode = str(mode or "auto").lower()
        has_sig = (sighet is not None
                   and hasattr(sighet, "setup_fstat_references")
                   and int(getattr(sighet, "_g", {}).get("v4_knots", 0) or 0) > 0)
        if mode == "sighet" and not has_sig:
            raise ValueError(
                "GB_PARA_SEARCH_SCORER=sighet needs a GBSignalHetComputations "
                "comp built with v4_knots > 0 (GB_SIGHET_INMODEL=1).")
        self.mode = ("sighet" if (mode in ("auto", "sighet") and has_sig)
                     else "chunked")
        self.sighet = sighet if self.mode == "sighet" else None
        if batch is None:
            from lisatools.sampling.fstat_proposal import fstat_knob

            batch = int(fstat_knob("FSTAT_BATCH", int))
        self.batch = max(1, int(batch))
        self.n_fstat = 0
        self.n_full = 0

    @property
    def xp(self):
        """The chunked comp's array module."""
        return self.chunked.xp

    def _di(self, n, row):
        """``(n,)`` int32 index array filled with ``row``."""
        return self.xp.full(int(n), int(row), dtype=self.xp.int32)

    # -- F-statistic ------------------------------------------------------
    def margin_hz(self) -> float:
        """Reference spacing of the sig-het scorer (Hz); 0 for chunked.

        ``FSTAT_SIGHET_REF_SPACING_HZ`` overrides the spacing derived from
        ``v4_knots`` and Tobs.
        """
        if self.mode != "sighet":
            return 0.0
        from lisatools.sampling.fstat_gridfit import sighet_fstat_ref_margin_hz

        env_sp = os.environ.get("FSTAT_SIGHET_REF_SPACING_HZ", "").strip()
        if env_sp:
            return float(env_sp)
        return sighet_fstat_ref_margin_hz(
            int(self.sighet._g.get("v4_knots", 0)), self.tobs)

    def ref_block(self) -> int:
        """References per resident sig-het block (``FSTAT_SIGHET_REF_BLOCK``, default 512)."""
        return max(1, int(os.environ.get("FSTAT_SIGHET_REF_BLOCK", "512")))

    def fstat_call(self, f0_lims_hz):
        """``call(P) -> (N, M)`` for rows with f0 inside ``f0_lims_hz``."""
        if self.mode == "sighet":
            return self._sighet_call(f0_lims_hz)
        return self.fstat_exact

    def _sighet_call(self, f0_lims_hz):
        """The shared-reference sig-het F-stat call for ``f0_lims_hz`` (references stashed)."""
        from lisatools.sampling.fstat_gridfit import build_sighet_call_fstat

        return build_sighet_call_fstat(
            self.sighet, self.holder, xp=self.xp, Tobs=self.tobs,
            f0_lims_hz=(float(f0_lims_hz[0]), float(f0_lims_hz[-1])),
            data_index=self.row, noise_index=self.noise_row)

    def release(self):
        """Drop the sig-het reference stash (one block is ~GB on a device)."""
        if self.sighet is not None and hasattr(self.sighet, "clear_fstat_references"):
            try:
                self.sighet.clear_fstat_references()
            except Exception:                       # noqa: BLE001
                pass

    def fstat_exact(self, P):
        """Exact chunked F-stat ``(N, M)`` (any rows)."""
        n = int(P.shape[0])
        self.n_fstat += n
        return self.chunked.get_fstat_ll_wdm(
            P, self.holder, data_index=self._di(n, self.row),
            noise_index=self._di(n, self.noise_row))

    # -- full likelihood ----------------------------------------------------
    def full(self, P):
        """``(d_h, h_h)`` of rows ``P`` against the residual (xp arrays)."""
        xp = self.xp
        n = int(P.shape[0])
        self.n_full += n
        d_h = xp.empty(n)
        h_h = xp.empty(n)
        for s in range(0, n, self.batch):
            e = min(s + self.batch, n)
            self.chunked.get_ll_wdm(
                P[s:e], self.holder, data_index=self._di(e - s, self.row),
                noise_index=self._di(e - s, self.noise_row))
            d_h[s:e] = xp.asarray(self.chunked.d_h_out).real
            h_h[s:e] = xp.asarray(self.chunked.h_h_out).real
        return d_h, h_h


# ---------------------------------------------------------------------------
# likelihood callables handed to the sampler
# ---------------------------------------------------------------------------
class _FStatLnL:
    """``F = 0.5 N^T M^-1 N`` of sampling rows (the extrinsic-maximised lnL)."""

    def __init__(self, call, transform_fn, xp, batch):
        self.call, self.transform_fn, self.batch = call, transform_fn, int(batch)
        self._xp = xp
        self.n_evals = 0
        self.n_calls = 0

    def __call__(self, x):
        from lisatools.sampling.fstat_proposal import compute_fstat

        xp = self._xp
        phys = self.transform_fn.both_transforms(x, xp=xp)
        n = int(phys.shape[0])
        self.n_evals += n
        self.n_calls += 1
        out = xp.empty(n)
        for s in range(0, n, self.batch):
            e = min(s + self.batch, n)
            N, M = self.call(phys[s:e])
            out[s:e] = compute_fstat(xp.asarray(N), xp.asarray(M))
        return xp.where(xp.isfinite(out), out, -1e300)


class _FullLnL:
    """``<r|h> - <h|h>/2`` of sampling rows (stage 2)."""

    def __init__(self, scorer, transform_fn, xp):
        self.scorer, self.transform_fn, self._xp = scorer, transform_fn, xp

    def __call__(self, x):
        xp = self._xp
        phys = self.transform_fn.both_transforms(x, xp=xp)
        d_h, h_h = self.scorer.full(phys)
        ll = d_h - 0.5 * h_h
        return xp.where(xp.isfinite(ll), ll, -1e300)


# ---------------------------------------------------------------------------
# per-group convergence + sample ring (the sampler's stopping_fn)
# ---------------------------------------------------------------------------
class _Tracker:
    """Per-group running max of the cold rung, switch-off, sample ring.

    Called every step as ``stopping_fn(i, state, sampler)``. A running group
    stops once ``steps >= min_steps`` and its cold-rung running max has not
    improved by more than ``tol`` over the last ``window`` steps, or at
    ``max_steps``; stopping sets ``state.groups_running[g] = False`` (the
    sampler's next ``propose`` copies that mask, so the batch shrinks).
    The ring keeps the last ``keep`` steps of rungs ``0..n_rungs-1`` of each
    group as it ran (a stopped group's ring is frozen at its last steps).
    """

    def __init__(self, ngroups, ntemps, nwalkers, ndim, *, window, tol,
                 min_steps, max_steps, keep, n_rungs):
        self.ng = int(ngroups)
        self.window, self.tol = int(window), float(tol)
        self.min_steps, self.max_steps = int(min_steps), int(max_steps)
        self.keep = max(1, int(keep))
        self.nk = max(1, min(int(n_rungs), int(ntemps)))
        self.runmax = np.full(self.ng, -np.inf)
        self.hist = np.full((self.max_steps + 1, self.ng), -np.inf)
        self.ring = np.zeros((self.keep, self.ng, self.nk, int(nwalkers), int(ndim)))
        self.ring_ll = np.full((self.keep, self.ng, self.nk, int(nwalkers)), -np.inf)
        self.n_written = np.zeros(self.ng, dtype=np.int64)
        self.steps = np.zeros(self.ng, dtype=np.int64)
        self.converged = np.zeros(self.ng, dtype=bool)
        self.best_ll = np.full(self.ng, -np.inf)
        self.best_x = np.zeros((self.ng, int(ndim)))
        self.running = np.ones(self.ng, dtype=bool)

    def seed(self, state):
        """Initialize the running max and the best sample from the start state."""
        ll = _host(state.log_like)[:, 0]
        self.runmax = ll.max(axis=-1)
        self.hist[0] = self.runmax
        self._note_best(state, np.arange(self.ng))

    def _note_best(self, state, run):
        """Keep each running group's best cold-rung sample and lnL."""
        if run.size == 0:
            return
        ll = _host(state.log_like[:, 0])[run]                 # (r, nw)
        x = _host(state.branches["gb"].coords[:, 0])[run]     # (r, nw, ndim)
        j = ll.argmax(axis=-1)
        v = ll[np.arange(run.size), j]
        better = v > self.best_ll[run]
        self.best_ll[run[better]] = v[better]
        self.best_x[run[better]] = x[np.arange(run.size), j][better]

    def __call__(self, i, state, sampler):
        s = int(i) + 1                                   # steps completed
        run = np.nonzero(self.running)[0]
        if run.size:
            ll_all = _host(state.log_like)               # (ng, nt, nw)
            cold = ll_all[run, 0].max(axis=-1)
            self.runmax[run] = np.maximum(self.runmax[run], cold)
            slot = (s - 1) % self.keep
            coords = _host(state.branches["gb"].coords[:, : self.nk])
            self.ring[slot, run] = coords[run]
            self.ring_ll[slot, run] = ll_all[run, : self.nk]
            self.n_written[run] += 1
            self._note_best(state, run)
        self.hist[min(s, self.max_steps)] = self.runmax
        stop = []
        for g in run:
            if s >= self.max_steps:
                stop.append((g, False))
            elif s >= self.min_steps and s >= self.window:
                gain = self.runmax[g] - self.hist[s - self.window, g]
                if gain <= self.tol:
                    stop.append((g, True))
        if stop:
            gr = state.groups_running
            for g, conv in stop:
                self.running[g] = False
                self.steps[g] = s
                self.converged[g] = conv
                gr[int(g)] = False
        return not self.running.any()

    def pool(self, g):
        """``(rows (n, ndim), ll (n,))`` of group ``g``'s kept samples."""
        n = int(min(self.n_written[g], self.keep))
        if n == 0:
            return np.zeros((0, self.ring.shape[-1])), np.zeros(0)
        slots = np.arange(self.keep) if n >= self.keep else np.arange(n)
        x = self.ring[slots, g].reshape(-1, self.ring.shape[-1])
        ll = self.ring_ll[slots, g].reshape(-1)
        return x, ll

    def pool_cold(self, g):
        """``(n, ndim)`` cold-rung kept samples of group ``g``."""
        n = int(min(self.n_written[g], self.keep))
        slots = np.arange(self.keep) if n >= self.keep else np.arange(n)
        return self.ring[slots, g, 0].reshape(-1, self.ring.shape[-1])


# ---------------------------------------------------------------------------
# the detection gates
# ---------------------------------------------------------------------------
def passes_gate_a(F_best, cfg) -> bool:
    """Stage 1 gate: the cluster's best extrinsic-maximised lnL ``F`` reaches the SNR floor.

    The test is ``sqrt(2 F) >= snr_min`` (F = SNR^2 / 2 at the F-stat
    maximum of a noise-free source); ``cfg`` is a :class:`ParaSearchConfig`.
    Returns a bool.
    """
    return float(F_best) >= 0.5 * float(cfg.snr_min) ** 2


def passes_gate_b(opt, det, cfg) -> bool:
    """Stage 2: the best sample's optimal SNR ``sqrt(<h|h>)`` reaches
    ``snr_min`` AND its detection SNR ``<r|h> / sqrt(<h|h>)`` is consistent
    with it (``det >= det_frac x opt``). At an exact amplitude maximum
    ``det == opt``, so the second clause catches a best sample whose template
    the residual does not actually support (an edge-pinned or mis-scored
    row), not a weak source. ``cfg`` is a :class:`ParaSearchConfig`; returns
    a bool.
    """
    return (float(opt) >= float(cfg.snr_min)
            and float(det) >= float(cfg.det_frac) * float(opt))


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def greedy_peaks(f0, score, radius, *, min_score=-np.inf,
                 max_peaks=MAX_FINDS_PER_BAND):
    """Greedy mode picking on one axis: ``(labels (n,), centres [idx])``.

    Samples are visited in descending ``score``; an unassigned sample becomes
    a new centre and takes every unassigned sample within ``radius`` of it
    (in ``f0`` units). Centres are therefore local maxima of the pooled
    samples, and two sources further apart than ``radius`` split. Stops at
    ``max_peaks`` centres or at the first centre below ``min_score``
    (every later one would be lower). Unassigned samples keep label -1.
    """
    f0 = np.asarray(f0, dtype=float)
    score = np.asarray(score, dtype=float)
    labels = np.full(f0.shape[0], -1, dtype=np.int64)
    centres = []
    for idx in np.argsort(-score, kind="stable"):
        if labels[idx] >= 0:
            continue
        if len(centres) >= int(max_peaks) or score[idx] < min_score:
            break
        m = (labels < 0) & (np.abs(f0 - f0[idx]) <= radius)
        labels[m] = len(centres)
        centres.append(int(idx))
    return labels, centres


def band_groups(band_edges_hz, bands, *, spacing_hz=0.0, ref_block=512,
                group_bands=0):
    """Contiguous runs of ``bands`` searched by one sampler each.

    ``group_bands > 0``: at most that many bands per group. Otherwise, with a
    sig-het ``spacing_hz``, as many bands as keep the group's reference comb
    (``(f_hi - f_lo) / spacing + 3`` references, ``build_sighet_call_fstat``'s
    layout) inside ONE resident block of ``ref_block``; with no spacing
    (chunked scorer) every band in one group. A gap in ``bands`` always
    starts a new group.

    Args:
        band_edges_hz (array-like): ``(n_bands + 1,)`` sub-band edges [Hz].
        bands (iterable of int): Band indices to group (sorted here).
        spacing_hz (float, optional): Sig-het reference spacing [Hz]
            (:meth:`ResidualScorer.margin_hz`). Default ``0.0``.
        ref_block (int, optional): References per block. Default ``512``.
        group_bands (int, optional): Fixed group size cap. Default ``0``.

    Returns:
        list of np.ndarray: int64 band-index arrays, one per group.
    """
    edges = np.asarray(band_edges_hz, dtype=float)
    bands = np.asarray(sorted(int(b) for b in bands), dtype=np.int64)
    groups, cur = [], []
    for b in bands:
        if cur and (b != cur[-1] + 1):
            groups.append(cur)
            cur = []
        if cur:
            lo = edges[cur[0]]
            hi = edges[b + 1]
            too_many = group_bands > 0 and len(cur) >= int(group_bands)
            too_wide = (group_bands <= 0 and spacing_hz > 0
                        and (hi - lo) / spacing_hz + 3 > int(ref_block))
            if too_many or too_wide:
                groups.append(cur)
                cur = []
        cur.append(int(b))
    if cur:
        groups.append(cur)
    return [np.asarray(g, dtype=np.int64) for g in groups]


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
class ParaSearchBuilder:
    """Search ONE residual row: every sub-band, one GB each.

    Args:
        scorer: :class:`ResidualScorer` (or anything with its interface).
        band_edges_hz: the run's sub-band edges (Hz).
        transform_fn: the run's GB ``TransformContainer`` (9-column sampling
            basis ``dist, f0 [mHz], Mc, phi0, cos_iota, psi, alpha, sin_delta,
            fdot_astro_ratio``).
        prior: ``ProbDistContainer`` over the 9 columns with UNIT-CUBE f0 / Mc
            (:func:`build_search_prior` of :func:`band_restricted_priors_in` or
            :func:`default_search_priors_in`).
        mc_lims: the third column's box (the old ``_third_col_search_bounds``
            for the chirp-mass basis: the flat ``m_chirp_lims``).
        ratio_max: half-width of the ``fdot_astro_ratio`` prior.
        cfg: :class:`ParaSearchConfig`.
        gpu: CUDA device index for the sampler (``None`` = CPU).
        periodic (dict, optional): Eryn ``periodic`` map; default
            ``{"gb": GB_PERIODIC}`` (phi0, psi, alpha).
        f0_lims_hz: the global f0 prior range (stage-2 windows are clipped
            to it); default the outer band edges.
        tag: log prefix (``"rank r walker w"``).

    Raises:
        ValueError: When ``transform_fn``'s ``input_basis`` is not the
            9-column sampling basis.

    Example:
        .. code-block:: python

            builder = ParaSearchBuilder(
                scorer, band_edges_hz=edges, transform_fn=transform,
                prior=prior, mc_lims=(0.001, 1.0), ratio_max=1.0,
                cfg=ParaSearchConfig.from_env(), gpu=0, tag="walker 0")
            finds = builder.run(bands=np.arange(400, 480), walker=0)
    """

    def __init__(self, scorer, *, band_edges_hz, transform_fn, prior, mc_lims,
                 ratio_max, cfg: ParaSearchConfig, gpu=None, periodic=None,
                 f0_lims_hz=None, tag=""):
        self.scorer = scorer
        self.edges = np.asarray(_host(band_edges_hz), dtype=float)
        self.transform_fn = transform_fn
        ib = list(getattr(transform_fn, "input_basis", None) or [])
        if ib and ib != SAMPLING_COLUMNS:
            raise ValueError(
                f"the GB parallel search needs the 9-column sampling basis "
                f"{SAMPLING_COLUMNS}; the run samples {ib}.")
        self.prior = prior
        self.mc_lims = (float(mc_lims[0]), float(mc_lims[-1]))
        self.ratio_max = float(ratio_max)
        self.cfg = cfg
        self.gpu = gpu
        self.periodic = {"gb": dict(GB_PERIODIC)} if periodic is None else periodic
        self.tobs = float(scorer.tobs)
        self.df_hz = 1.0 / self.tobs
        lims = (self.edges[0], self.edges[-1]) if f0_lims_hz is None else f0_lims_hz
        self.f0_lims_mhz = (float(lims[0]) * 1e3, float(lims[-1]) * 1e3)
        self.tag = str(tag)

    @property
    def xp(self):
        """The scorer's array module."""
        return self.scorer.xp

    def default_bands(self):
        """The interior sub-bands (the outer buffer band at each end excluded)."""
        nb = len(self.edges) - 1
        return np.arange(1, nb - 1) if nb >= 3 else np.arange(nb)

    # ---- the sampler -------------------------------------------------------
    def _run_sampler(self, *, lnl, box, gibbs, start, tracker, max_steps):
        """Run Eryn's ``ParaEnsembleSampler`` from ``start`` until ``tracker`` stops it.

        ``start`` is ``(ngroups, nt, nw, 9)``; tempering to ``Tmax = inf``;
        raises ``RuntimeError`` when a start point is outside the prior.
        Returns the sampler.
        """
        from eryn.paraensemble import ParaEnsembleSampler
        from eryn.state import ParaState

        xp = self.xp
        ng, nt, nw, ndim = start.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sampler = ParaEnsembleSampler(
                ndim, nw, ng, lnl, {"gb": self.prior},
                tempering_kwargs=dict(ntemps=nt, Tmax=np.inf),
                gpu=self.gpu, periodic=self.periodic,
                prior_transform_fn=box, name="gb",
                gibbs_sampling_setup=gibbs,
                stopping_fn=tracker, stopping_iterations=1)
        state = ParaState({"gb": xp.asarray(start)},
                          groups_running=xp.ones(ng, dtype=bool))
        state.log_prior = sampler.compute_log_prior(
            state.branches_coords, groups_running=xp.arange(ng))
        if not bool(xp.all(xp.isfinite(state.log_prior))):
            raise RuntimeError(f"{self.tag}: a start point lies outside the search prior.")
        state.log_like = sampler.compute_log_like(
            state.branches_coords, groups_running=xp.arange(ng),
            logp=state.log_prior)
        tracker.seed(state)
        sampler.run_mcmc(state, int(max_steps), burn=0, store=False)
        return sampler

    # ---- stage 1 -----------------------------------------------------------
    def _stage1_group(self, bands):
        """Stage 1 on one band group: F-stat PT-MCMC, Gibbs on f0 / Mc / sky, ratio held at 0.

        Returns the group's :class:`_Tracker` (with ``comb`` and
        ``n_evals`` attached); the scorer's references are released.
        """
        cfg, xp = self.cfg, self.xp
        ng = len(bands)
        f_lo = self.edges[bands] * 1e3
        f_hi = self.edges[bands + 1] * 1e3
        box = BandBoxPrior(f_lo, f_hi, np.full(ng, self.mc_lims[0]),
                           np.full(ng, self.mc_lims[1]), xp=xp)
        call = self.scorer.fstat_call((self.edges[bands[0]], self.edges[bands[-1] + 1]))
        lnl = _FStatLnL(call, self.transform_fn, xp, self.scorer.batch)
        gibbs = np.zeros(9, dtype=bool)
        gibbs[list(STAGE1_GIBBS_COLS)] = True
        start = self.prior.rvs(size=(ng, cfg.ntemps, cfg.nwalkers))
        start = xp.asarray(start)
        box.transform_from_prior_basis(start, None)
        start[..., RATIO_COL] = 0.0
        comb = None
        try:
            if cfg.init == "comb":
                comb = self._comb(bands, lnl)
                start = self._comb_start(start, comb, bands)
            tracker = _Tracker(ng, cfg.ntemps, cfg.nwalkers, 9,
                               window=cfg.converge_window, tol=cfg.converge_tol,
                               min_steps=cfg.min_steps, max_steps=cfg.max_steps,
                               keep=cfg.converge_window, n_rungs=cfg.cluster_rungs)
            self._run_sampler(lnl=lnl, box=box, gibbs=gibbs, start=start,
                              tracker=tracker, max_steps=cfg.max_steps)
        finally:
            self.scorer.release()
        tracker.comb = comb
        tracker.n_evals = int(lnl.n_evals)
        return tracker

    # ---- the comb that seeds stage 1 -----------------------------------------
    def _comb(self, bands, lnl):
        """Per band: F on a 1-bin f0 comb x ``n_sky`` sky points (a golden-
        ratio spread, randomly rotated per node; ``n_sky`` = the F-stat fit's
        Doppler law ``sky_nodes_required`` clipped to
        [``init_nsky_min``, ``init_nsky_max``]) with Mc drawn from its box.
        Returns ``[(rows (n, 9), F (n,))]`` per band, host arrays.

        Why: a 1-bin-wide F-stat peak in a sub-band hundreds of bins wide
        (2 get_N / Tobs) is found by blind PT exploration only slowly -- the
        hot rung samples the prior a few walkers at a time. Starting the cold
        and warm rungs at the best comb cells and letting the PT refine from
        there costs one extra pass of ~bins x n_sky F-stat rows per band.
        """
        from lisatools.sampling.fstat_gridfit import sky_nodes_required

        cfg, xp = self.cfg, self.xp
        dfm = self.df_hz * 1e3
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        base = self.prior.rvs(size=1)
        base = np.asarray(_host(base))[0].copy()
        base[RATIO_COL] = 0.0
        out = []
        rows_all = []
        for b in bands:
            lo, hi = self.edges[b] * 1e3, self.edges[b + 1] * 1e3
            nb = max(1, int(round((hi - lo) / dfm)))
            f0 = lo + (np.arange(nb) + 0.5) * (hi - lo) / nb
            ns = int(np.clip(sky_nodes_required(0.5 * (lo + hi), self.tobs),
                             int(cfg.init_nsky_min), int(cfg.init_nsky_max)))
            ks = np.arange(ns)
            sd = -1.0 + 2.0 * (ks + 0.5) / ns
            al = (2.0 * np.pi * ks * 0.6180339887)
            rot = rng.uniform(0.0, 2.0 * np.pi, size=nb)
            r = np.tile(base, (nb * ns, 1))
            r[:, F0_COL] = np.repeat(f0, ns)
            r[:, 6] = (np.tile(al, nb) + np.repeat(rot, ns)) % (2.0 * np.pi)
            r[:, 7] = np.tile(sd, nb)
            r[:, THIRD_COL] = rng.uniform(self.mc_lims[0], self.mc_lims[1], size=nb * ns)
            rows_all.append(r)
        rows = np.concatenate(rows_all, axis=0)
        F = np.asarray(_host(lnl(xp.asarray(rows))), dtype=float)
        off = 0
        for r in rows_all:
            n = r.shape[0]
            out.append((r, F[off:off + n]))
            off += n
        return out

    def _comb_start(self, start, comb, bands):
        """Cold + warm rungs from the comb's best cells, hottest from the prior.

        Rung ``k < ntemps - 1`` draws its walkers from the band's best
        ``4 x nwalkers`` comb cells with probability ``~ exp(beta_k (F -
        F_max))`` at the sampler's initial ladder (``make_ladder``), plus a
        +-0.3-bin f0 jitter; the hottest rung keeps its prior draw.
        """
        from eryn.moves.tempering import make_ladder

        cfg, xp = self.cfg, self.xp
        nt, nw = cfg.ntemps, cfg.nwalkers
        st = np.array(_host(start), dtype=float, copy=True)
        betas = (np.asarray(make_ladder(9, ntemps=nt, Tmax=np.inf), dtype=float)
                 if nt > 1 else np.ones(1))
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        dfm = self.df_hz * 1e3
        for g, (rows, F) in enumerate(comb):
            top = np.argsort(-F, kind="stable")[: max(4 * nw, 1)]
            Ft = F[top] - F[top].max()
            for k in range(max(1, nt - 1) if nt > 1 else 1):
                w = np.exp(float(betas[k]) * Ft)
                w = w / w.sum()
                pick = top[rng.choice(top.size, size=nw, p=w)]
                x = rows[pick].copy()
                lo, hi = self.edges[bands[g]] * 1e3, self.edges[bands[g] + 1] * 1e3
                x[:, F0_COL] = np.clip(x[:, F0_COL] + rng.uniform(-0.3, 0.3, nw) * dfm,
                                       lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
                keep_cols = list(STAGE1_GIBBS_COLS)
                st[g, k][:, keep_cols] = x[:, keep_cols]
        return xp.asarray(st)

    def _clusters(self, tracker, g, band):
        """Gate-A clusters of group ``g`` (band ``band``): list of dicts.

        The pool is the kept cold + warm rung samples and the band's comb
        cells (each with its own, beta-independent F)."""
        x, ll = tracker.pool(g)
        comb = getattr(tracker, "comb", None)
        if comb is not None:
            cr, cF = comb[g]
            keep = np.argsort(-cF, kind="stable")[: 4 * int(self.cfg.nwalkers)]
            x = np.concatenate([x, cr[keep]], axis=0)
            ll = np.concatenate([ll, cF[keep]])
        if x.shape[0] == 0:
            return [], 0
        radius = float(self.cfg.split_bins) * self.df_hz * 1e3       # mHz
        F_gate = 0.5 * float(self.cfg.snr_min) ** 2
        labels, centres = greedy_peaks(x[:, F0_COL], ll, radius,
                                       min_score=F_gate,
                                       max_peaks=MAX_FINDS_PER_BAND)
        out = []
        for k, c in enumerate(centres):
            if not passes_gate_a(ll[c], self.cfg):
                break
            m = labels == k
            out.append(dict(band=int(band), F=float(ll[c]),
                            centre=x[c].copy(), members=x[m], members_F=ll[m]))
        return out, len(centres)

    # ---- stage 2 -----------------------------------------------------------
    def _stage2_seed(self, find, rng):
        """``(ntemps, nwalkers, 9)`` start for one find: its pooled members with
        the extrinsics set at the exact F-stat maximisers, a spread in the
        (Mc, ratio) split and a small extrinsic jitter."""
        from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist
        from lisatools.sampling.fstat_proposal import fstat_maximized_extrinsics

        cfg, xp = self.cfg, self.xp
        mem, mF = find["members"], find["members_F"]
        order = np.argsort(-mF, kind="stable")
        # distinct members, best first
        _, uniq = np.unique(np.round(mem[order][:, [F0_COL, THIRD_COL, 6, 7]], 12),
                            axis=0, return_index=True)
        pick = order[np.sort(uniq)][:N_STAGE2_SEEDS]
        seeds = mem[pick].copy()
        phys = _host(self.transform_fn.both_transforms(xp.asarray(seeds), xp=xp))
        N, M = self.scorer.fstat_exact(xp.asarray(phys))
        A, phi0, iota, psi, _F = (np.asarray(_host(v)) for v in
                                  fstat_maximized_extrinsics(xp.asarray(N), xp.asarray(M)))
        # THE MAXIMISER -> TEMPLATE MAPPING (measured 2026-10-16 on the chunked
        # WDM comp, noise-free injections, every |cos iota|; see
        # docs/gb-para-search.md section 9): the chunked template that
        # reproduces lnL = F exactly has PHYSICAL phi0 = -phi0_max,
        # iota = pi - iota_max, psi = psi_max. In the sampling basis (the
        # transform negates phi0) that is phi0 = phi0_max, cos_iota =
        # -cos(iota_max), psi = psi_max. (cos_iota = +cos(iota_max), the
        # production birth pin's choice, is exact only edge-on: lnL -152
        # instead of +113 for an SNR-15 source at cos iota = -0.48.) The
        # distance is A propto 1/d at the seed's own (f0, Mc), written
        # directly rather than through the inverse transform, whose (Mc,
        # ratio) mirror would re-derive Mc from |fdot|.
        samp = seeds.copy()
        good = np.isfinite(A) & (A > 0)
        f0_hz = seeds[:, F0_COL] * 1e-3
        samp[:, 0] = np.where(good, np.asarray(gb_amp_from_dist(
            f0_hz, seeds[:, THIRD_COL], 1.0)) / np.where(good, A, 1.0), seeds[:, 0])
        samp[:, 3] = np.where(good, phi0 % (2.0 * np.pi), seeds[:, 3])
        samp[:, 4] = np.where(good, -np.cos(iota), seeds[:, 4])
        samp[:, 5] = np.where(good, psi % np.pi, seeds[:, 5])
        nt, nw = cfg.stage2_ntemps, cfg.nwalkers
        idx = rng.integers(0, samp.shape[0], size=(nt, nw))
        out = samp[idx].copy()                                  # (nt, nw, 9)
        lo_d, hi_d = self._prior_box(0)
        r_half = min(0.3, 0.3 * self.ratio_max)
        r = rng.uniform(-r_half, r_half, size=(nt, nw))
        mc = out[..., THIRD_COL]
        mc_new = mc * (1.0 / (1.0 + r)) ** 0.6
        ok = (mc_new > self.mc_lims[0]) & (mc_new < self.mc_lims[1])
        r = np.where(ok, r, 0.0)
        mc_new = np.where(ok, mc_new, mc)
        out[..., THIRD_COL] = mc_new
        out[..., RATIO_COL] = r
        # A propto Mc^(5/3) / d: keep the amplitude when Mc moves
        out[..., 0] = out[..., 0] * (mc_new / mc) ** (5.0 / 3.0)
        out[..., 0] *= np.exp(rng.normal(0.0, 0.05, size=(nt, nw)))
        out[..., 3] = (out[..., 3] + rng.normal(0.0, 0.1, size=(nt, nw))) % (2 * np.pi)
        out[..., 4] = np.clip(out[..., 4] + rng.normal(0.0, 0.03, size=(nt, nw)),
                              -0.999, 0.999)
        out[..., 5] = (out[..., 5] + rng.normal(0.0, 0.05, size=(nt, nw))) % np.pi
        if lo_d is not None:
            out[..., 0] = np.clip(out[..., 0], lo_d * 1.0001, hi_d * 0.9999)
        return out

    def _prior_box(self, col):
        """``(lo, hi)`` of a single-column uniform prior, else ``(None, None)``."""
        for inds, dist in getattr(self.prior, "priors", []):
            if len(inds) == 1 and int(inds[0]) == int(col) and hasattr(dist, "minimum"):
                return float(dist.minimum), float(dist.maximum)
        return None, None

    def _logprior(self, coords, box):
        """The sampler's ``compute_log_prior`` for ``(ng, nt, nw, 9)`` coords."""
        xp = self.xp
        c = xp.array(xp.asarray(coords), copy=True)
        box.transform_to_prior_basis(c, None)
        lp = self.prior.logpdf(c.reshape(-1, c.shape[-1])).reshape(c.shape[:-1])
        box.adjust_logp(lp, None)
        return np.asarray(_host(lp))

    def _stage2_boxes(self, finds):
        """``(BandBoxPrior, f_lo, f_hi)``: a ``+-split_bins/Tobs`` f0 box [mHz] per find."""
        radius = float(self.cfg.split_bins) * self.df_hz * 1e3
        f_c = np.array([f["centre"][F0_COL] for f in finds])
        f_lo = np.maximum(f_c - radius, self.f0_lims_mhz[0])
        f_hi = np.minimum(f_c + radius, self.f0_lims_mhz[1])
        n = len(finds)
        box = BandBoxPrior(f_lo, f_hi, np.full(n, self.mc_lims[0]),
                           np.full(n, self.mc_lims[1]), xp=self.xp)
        return box, f_lo, f_hi

    def _stage2(self, finds, rng):
        """Stage 2 over every find; returns ``(tracker, kept finds)``.

        A start row outside the run's prior (a weak source whose F-stat
        amplitude needs a distance beyond ``dist_lims`` at the seed's Mc, a
        galaxy-model sky prior's zero, ...) is replaced by an in-prior row of
        the same find; a find with none is dropped (``n_prior_dropped``).
        """
        cfg, xp = self.cfg, self.xp
        box, f_lo, f_hi = self._stage2_boxes(finds)
        start = np.stack([self._stage2_seed(f, rng) for f in finds])
        start[..., F0_COL] = np.clip(start[..., F0_COL], f_lo[:, None, None],
                                     f_hi[:, None, None])
        lp = self._logprior(start, box)
        keep = []
        for g in range(len(finds)):
            ok = np.isfinite(lp[g]).ravel()
            if not ok.any():
                continue
            if not ok.all():
                flat = start[g].reshape(-1, start.shape[-1])
                good = np.nonzero(ok)[0]
                bad = np.nonzero(~ok)[0]
                flat[bad] = flat[rng.choice(good, size=bad.size)]
                start[g] = flat.reshape(start[g].shape)
            keep.append(g)
        self._n_prior_dropped = len(finds) - len(keep)
        if not keep:
            return None, []
        if len(keep) < len(finds):
            finds = [finds[g] for g in keep]
            start = start[keep]
            box, f_lo, f_hi = self._stage2_boxes(finds)
        n = len(finds)
        lnl = _FullLnL(self.scorer, self.transform_fn, xp)
        n_keep = max(1, int(cfg.stage2_steps) // 2)
        tracker = _Tracker(n, cfg.stage2_ntemps, cfg.nwalkers, 9,
                           window=cfg.stage2_steps, tol=-np.inf,
                           min_steps=cfg.stage2_steps, max_steps=cfg.stage2_steps,
                           keep=n_keep, n_rungs=1)
        self._run_sampler(lnl=lnl, box=box, gibbs=None, start=start,
                          tracker=tracker, max_steps=cfg.stage2_steps)
        return tracker, finds

    def _rescore(self, rows):
        """``(opt, det, phys)``: ``sqrt(<h|h>)``, ``<r|h>/sqrt(<h|h>)`` and the physical rows."""
        xp = self.xp
        phys = self.transform_fn.both_transforms(xp.asarray(rows), xp=xp)
        d_h, h_h = self.scorer.full(phys)
        d_h, h_h = np.asarray(_host(d_h)), np.asarray(_host(h_h))
        opt = np.sqrt(np.maximum(h_h, 0.0))
        det = np.where(opt > 0, d_h / np.where(opt > 0, opt, 1.0), 0.0)
        return opt, det, phys

    # ---- the whole search ----------------------------------------------------
    def run(self, bands=None, walker=None, seed=None) -> dict:
        """Search ``bands`` (default: the interior ones) of this residual row.

        Stage 1 per band group, gate A, the ``max_finds`` cap (loudest
        kept), stage 2, gate B, then the within-walker de-duplication; all
        inside :func:`seeded_global_rng`.

        Args:
            bands (array-like, optional): Band indices. Default ``None`` =
                :meth:`default_bands`.
            walker (int, optional): The walker label of the record.
            seed (int, optional): Default ``cfg.seed`` (``None`` = fresh).

        Returns:
            dict: The walker's finds (host numpy; see :func:`empty_finds`):
            per-band ``steps`` / ``converged`` / ``band_finds`` /
            ``band_peaks``, per-find scalars (``f0_hz`` [Hz], ``fdot``,
            ``F1``, ``lnl2``, ``opt_snr``, ``det_snr``, medians, ``edge``),
            ``best`` ``(n_finds, 9)`` and the stage-2 cold ``samples``
            (sampling basis) with ``offsets``, counters, walls and the seed.
        """
        cfg = self.cfg
        bands = self.default_bands() if bands is None else np.asarray(bands, dtype=np.int64)
        seed = cfg.seed if seed is None else seed
        out = empty_finds(walker)
        out["bands"] = np.asarray(bands, dtype=np.int64)
        out["steps"] = np.zeros(len(bands), dtype=np.int64)
        out["converged"] = np.zeros(len(bands), dtype=bool)
        out["band_finds"] = np.zeros(len(bands), dtype=np.int64)
        out["band_peaks"] = np.zeros(len(bands), dtype=np.int64)
        if len(bands) == 0:
            return out
        spacing = self.scorer.margin_hz()
        groups = band_groups(self.edges, bands, spacing_hz=spacing,
                             ref_block=self.scorer.ref_block(),
                             group_bands=int(cfg.group_bands))
        pos = {int(b): i for i, b in enumerate(bands)}
        stage1_finds = []
        t0 = time.perf_counter()
        n_fstat = 0
        n_full0 = int(getattr(self.scorer, "n_full", 0))
        with seeded_global_rng(seed, self.xp) as used_seed:
            rng = np.random.default_rng(used_seed)
            for gi, gb in enumerate(groups):
                tr = self._stage1_group(gb)
                n_fstat += int(getattr(tr, "n_evals", 0))
                for j, b in enumerate(gb):
                    i = pos[int(b)]
                    out["steps"][i] = tr.steps[j]
                    out["converged"][i] = tr.converged[j]
                    cl, n_peaks = self._clusters(tr, j, int(b))
                    out["band_peaks"][i] = n_peaks
                    stage1_finds.extend(cl)
                if len(groups) > 1 and (gi + 1) % max(1, len(groups) // 10) == 0:
                    logger.info("[GB_PARA_SEARCH] %s: stage 1 group %d/%d done "
                                "(%.1f s so far)", self.tag, gi + 1, len(groups),
                                time.perf_counter() - t0)
            t1 = time.perf_counter()
            out["walls"]["fstat"] = t1 - t0
            out["n_fstat_evals"] = int(n_fstat)
            out["n_gate_a"] = len(stage1_finds)
            if 0 < int(cfg.max_finds) < len(stage1_finds):
                # stage-2 cost bound (noise peaks clear SNR 5 in most
                # sub-bands of a long run, docs/gb-para-search.md 5): keep
                # the loudest stage-1 finds
                stage1_finds.sort(key=lambda d: -d["F"])
                out["n_capped"] = len(stage1_finds) - int(cfg.max_finds)
                logger.info("[GB_PARA_SEARCH] %s: %d gate-A finds, stage 2 keeps "
                            "the loudest %d (GB_PARA_SEARCH_MAX_FINDS)", self.tag,
                            len(stage1_finds), int(cfg.max_finds))
                stage1_finds = stage1_finds[: int(cfg.max_finds)]
            finds = []
            if stage1_finds:
                tr2, stage1_finds = self._stage2(stage1_finds, rng)
                out["n_prior_dropped"] = int(getattr(self, "_n_prior_dropped", 0))
                for k, f in enumerate(stage1_finds):
                    best = tr2.best_x[k][None, :]
                    opt_b, det_b, phys_b = self._rescore(best)
                    samples = tr2.pool_cold(k)
                    sub = samples[rng.choice(samples.shape[0],
                                             size=min(N_RESCORE, samples.shape[0]),
                                             replace=False)]
                    opt_s, det_s, _ = self._rescore(sub)
                    if not passes_gate_b(opt_b[0], det_b[0], cfg):
                        out["n_gate_b_fail"] += 1
                        continue
                    lo = self.edges[f["band"]] * 1e3
                    hi = self.edges[f["band"] + 1] * 1e3
                    f0b = float(best[0, F0_COL])
                    dfm = self.df_hz * 1e3
                    edge = bool((f0b - lo < dfm and f["band"] > 0)
                                or (hi - f0b < dfm and f["band"] < len(self.edges) - 2))
                    finds.append(dict(
                        band=f["band"], f0_hz=f0b * 1e-3,
                        fdot=float(np.asarray(phys_b)[0, 2]), F1=f["F"],
                        lnl2=float(tr2.best_ll[k]), opt_snr=float(opt_b[0]),
                        det_snr=float(det_b[0]),
                        opt_med=float(np.median(opt_s)),
                        det_med=float(np.median(det_s)), edge=edge,
                        samples=samples, best=best[0]))
            out["walls"]["full"] = time.perf_counter() - t1
            out["n_full_evals"] = int(getattr(self.scorer, "n_full", 0)) - n_full0
        finds, n_dup = _dedup_walker_finds(finds, self.cfg.split_bins * self.df_hz)
        out["n_dedup"] = n_dup
        _pack_finds(out, finds)
        for f in finds:
            out["band_finds"][pos[int(f["band"])]] += 1
        out["seed"] = int(used_seed)
        self._log_summary(out)
        return out

    def _log_summary(self, out):
        """One ``[GB_PARA_SEARCH]`` summary line of a walker's search."""
        st = out["steps"]
        nb = len(st)
        two = int((out["band_finds"] >= 2).sum())
        logger.info(
            "[GB_PARA_SEARCH] %s: %d bands searched, %d converged in %s steps "
            "(min/median/max; the rest hit GB_PARA_SEARCH_MAX_STEPS=%d), "
            "%d finds (>= SNR %g; gate A passed %d, capped %d, outside the "
            "prior %d, gate B failed %d, %d within-walker duplicate(s) "
            "dropped), %d band(s) with 2+ finds, "
            "wall %.1f s (fstat stage %.1f s [%s, %d evals], full stage %.1f s "
            "[%d evals]; GMM on the head)",
            self.tag, nb, int(out["converged"].sum()),
            ("%d/%d/%d" % (st.min(), int(np.median(st)), st.max())) if nb else "-",
            int(self.cfg.max_steps), int(out["n_finds"]), self.cfg.snr_min,
            int(out["n_gate_a"]), int(out["n_capped"]), int(out["n_prior_dropped"]),
            int(out["n_gate_b_fail"]), int(out["n_dedup"]),
            two, out["walls"]["fstat"] + out["walls"]["full"],
            out["walls"]["fstat"], self.scorer.mode, int(out["n_fstat_evals"]),
            out["walls"]["full"], int(out["n_full_evals"]))


# ---------------------------------------------------------------------------
# finds containers
# ---------------------------------------------------------------------------
_FIND_SCALARS = ("band", "f0_hz", "fdot", "F1", "lnl2", "opt_snr", "det_snr",
                 "opt_med", "det_med", "edge")


def empty_finds(walker=None) -> dict:
    """The per-walker finds record (host numpy, picklable, small), empty.

    Args:
        walker (int, optional): The walker label (``None`` -> -1).

    Returns:
        dict: The record :meth:`ParaSearchBuilder.run` fills: ``walker``,
        per-band arrays, the counters (``n_finds``, ``n_gate_a``,
        ``n_gate_b_fail``, ``n_dedup``, ``n_capped``, ``n_prior_dropped``,
        ``n_fstat_evals``, ``n_full_evals``), ``seed``, ``walls``, the
        per-find scalar arrays, ``best`` ``(0, 9)``, ``offsets`` ``[0]`` and
        ``samples`` ``(0, 9)``.
    """
    return dict(
        walker=(-1 if walker is None else int(walker)),
        bands=np.zeros(0, dtype=np.int64), steps=np.zeros(0, dtype=np.int64),
        converged=np.zeros(0, dtype=bool), band_finds=np.zeros(0, dtype=np.int64),
        band_peaks=np.zeros(0, dtype=np.int64),
        n_finds=0, n_gate_a=0, n_gate_b_fail=0, n_dedup=0, n_capped=0,
        n_prior_dropped=0,
        n_fstat_evals=0, n_full_evals=0, seed=-1,
        walls=dict(fstat=0.0, full=0.0),
        **{k: np.zeros(0) for k in _FIND_SCALARS},
        best=np.zeros((0, 9)), offsets=np.zeros(1, dtype=np.int64),
        samples=np.zeros((0, 9)),
    )


def _pack_finds(out, finds):
    """Write find dicts into the record ``out`` (scalars, ``best``, ``samples`` + ``offsets``)."""
    out["n_finds"] = len(finds)
    for k in _FIND_SCALARS:
        dt = (np.int64 if k == "band" else bool if k == "edge" else float)
        out[k] = np.asarray([f[k] for f in finds], dtype=dt)
    out["best"] = (np.stack([f["best"] for f in finds]) if finds
                   else np.zeros((0, 9)))
    n = [int(f["samples"].shape[0]) for f in finds]
    out["offsets"] = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    out["samples"] = (np.concatenate([f["samples"] for f in finds], axis=0)
                      if finds else np.zeros((0, 9)))


def _unpack_finds(rec) -> list:
    """Inverse of :func:`_pack_finds`: a record -> one dict per find (with its ``walker``)."""
    out = []
    off = np.asarray(rec["offsets"])
    for i in range(int(rec["n_finds"])):
        d = {k: (rec[k][i].item() if hasattr(rec[k][i], "item") else rec[k][i])
             for k in _FIND_SCALARS}
        d["samples"] = np.asarray(rec["samples"])[off[i]:off[i + 1]]
        d["best"] = np.asarray(rec["best"])[i]
        d["walker"] = int(rec.get("walker", -1))
        out.append(d)
    return out


def _dedup_walker_finds(finds, radius_hz):
    """One walker's finds whose best f0 lie within ``radius_hz`` describe one
    source (a sidelobe, or the same source seen from a neighbouring band):
    keep the highest stage-2 lnL."""
    if len(finds) < 2:
        return finds, 0
    order = sorted(range(len(finds)), key=lambda i: -finds[i]["lnl2"])
    kept = []
    for i in order:
        if any(abs(finds[i]["f0_hz"] - finds[j]["f0_hz"]) <= radius_hz for j in kept):
            continue
        kept.append(i)
    kept.sort(key=lambda i: finds[i]["f0_hz"])
    return [finds[i] for i in kept], len(finds) - len(kept)


# ---------------------------------------------------------------------------
# the head: merge, GMM, epoch
# ---------------------------------------------------------------------------
def merge_finds(walker_finds, *, tobs, radius_bins, seed=0):
    """Merge every walker's finds into physical sources.

    Two finds are the SAME source when their best f0 differ by at most
    ``radius_bins / Tobs`` AND their fdot by at most ``radius_bins / Tobs**2``
    (single linkage, so a chain of finds spread over a few bins merges).
    Within a merged source, finds pinned at a band edge (the neighbouring
    band's view of a source just outside it) are dropped whenever an
    unpinned find exists. The source's samples are the UNION of its finds'
    stage-2 samples, each find subsampled to the same count (at most
    :data:`UNION_PER_FIND`) so every walker that found it weighs the same:
    the birth proposal then covers the spread between the walkers'
    residuals, which is what proposing to each walker what the others found
    needs (the per-source GMM is a min-BIC mixture, so a bimodal union is
    represented, not averaged). Returns a list of dicts sorted by f0.

    Args:
        walker_finds (list of dict): Every walker's :meth:`ParaSearchBuilder.run`
            record.
        tobs (float): Observation time [s].
        radius_bins (float): The merge radius in ``1/Tobs`` bins (the
            config's ``split_bins``).
        seed (int, optional): Subsampling seed. Default ``0``.

    Returns:
        list of dict: One per source: ``samples`` (the union, sampling
        basis), ``walkers``, ``n_finds``, ``n_pinned_dropped``, ``f0_hz``,
        ``fdot``, ``band``, ``opt_snr``, ``det_snr`` (of its loudest find)
        and ``f0_span_hz``.
    """
    allf = []
    for rec in walker_finds:
        allf.extend(_unpack_finds(rec))
    if not allf:
        return []
    allf.sort(key=lambda d: d["f0_hz"])
    r_f = float(radius_bins) / float(tobs)
    r_fd = float(radius_bins) / float(tobs) ** 2
    n = len(allf)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if allf[j]["f0_hz"] - allf[i]["f0_hz"] > r_f:
                break
            if abs(allf[j]["fdot"] - allf[i]["fdot"]) <= max(r_fd, 1e-300):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rng = np.random.default_rng(int(seed))
    out = []
    for members in groups.values():
        fs = [allf[i] for i in members]
        unpinned = [f for f in fs if not f["edge"]]
        n_pinned_dropped = 0
        if unpinned and len(unpinned) < len(fs):
            n_pinned_dropped = len(fs) - len(unpinned)
            fs = unpinned
        per = min(UNION_PER_FIND, min(int(f["samples"].shape[0]) for f in fs))
        per = max(per, 1)
        rows = []
        for f in fs:
            s = f["samples"]
            take = rng.choice(s.shape[0], size=per, replace=s.shape[0] < per)
            rows.append(s[take])
        top = max(fs, key=lambda f: f["opt_snr"])
        out.append(dict(
            samples=np.concatenate(rows, axis=0),
            walkers=sorted({int(f["walker"]) for f in fs}),
            n_finds=len(fs), n_pinned_dropped=n_pinned_dropped,
            f0_hz=top["f0_hz"], fdot=top["fdot"], band=int(top["band"]),
            opt_snr=float(top["opt_snr"]), det_snr=float(top["det_snr"]),
            f0_span_hz=(min(f["f0_hz"] for f in fs), max(f["f0_hz"] for f in fs)),
        ))
    out.sort(key=lambda d: d["f0_hz"])
    return out


def _write_json_atomic(path, payload):
    """JSON to ``path`` via ``<path>.tmp`` + fsync + ``os.replace``."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(o):
    """``json`` fallback: numpy scalars / arrays to Python, anything else to ``str``."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _quantiles(v):
    """``{min, p10, p50, p90, max, n}`` of ``v`` (``{}`` when empty)."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {}
    q = np.quantile(v, [0.0, 0.1, 0.5, 0.9, 1.0])
    return dict(min=q[0], p10=q[1], p50=q[2], p90=q[3], max=q[4], n=int(v.size))


def write_epoch(out_dir, sources, walker_finds, *, tobs, transform_fn=None,
                manifest_extra=None, gmm_gpu=None, seed=7):
    """The epoch: ``parasearch.npz`` (WarmStartComponents OBSERVABLE schema),
    ``finds.npz`` (every walker's raw finds) and ``DONE.json`` written LAST.

    The npz is laid out exactly as ``fit_from_store.run(basis="observable")``
    writes it -- per-cluster ``p``, ``mult``, ``n_members``, ``island_id``,
    ``f0_window_edges`` and the packed ``gmm_*`` arrays -- so
    ``WarmStartComponents.from_npz`` loads it unchanged. One cluster per
    merged source, ``p = 1`` each (equal birth weight per source; an SNR
    weighting is a later knob, docs/gb-para-search.md). Returns the manifest
    dict (``None`` when there are no sources: no npz, the manifest still
    records the build and says ``n_sources = 0``, and the epoch is NOT
    complete, so no move adopts it).

    The sources' samples are mapped to the observable basis at ``tobs`` and
    fitted with :func:`.fit_from_store.fit_cluster_gmms` (``GMM_SAMPLES`` /
    ``GMM_MAX_COMP`` / ``GMM_MIN_MEMBERS``); each cluster's
    ``f0_window_edges`` is its finds' f0 span +- 2 / Tobs [Hz]. Extra
    per-source columns: ``src_f0_hz``, ``src_opt_snr``, ``src_det_snr``,
    ``src_band``, ``src_n_walkers``.

    Args:
        out_dir (str): The epoch directory (created).
        sources (list of dict): :func:`merge_finds` output.
        walker_finds (list of dict): Every walker's record (``finds.npz``
            and the manifest's per-walker statistics).
        tobs (float): Observation time [s].
        transform_fn (optional): The run's GB transform container, checked
            against the fitter's columns. Default ``None``.
        manifest_extra (dict, optional): Merged into the manifest first
            (``epoch``, ``stage``, ...).
        gmm_gpu (int, optional): CUDA device for the GMM fit; ``None`` = CPU.
        seed (int, optional): GMM seed. Default ``7``.

    Returns:
        dict or None: The manifest, or ``None`` with no sources.
    """
    from lisatools.sampling.fstat_proposal import pack_gmm_components

    from . import basis as wb
    from .fit_from_store import (CIRCULAR_COLS, OBSERVABLE_BOUNDED_COLS,
                                 build_observable_map, fit_cluster_gmms,
                                 to_observable)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    walls = {}
    _write_finds_npz(os.path.join(out_dir, FINDS_NPZ), walker_finds)
    manifest = dict(manifest_extra or {})
    manifest.update(
        epoch=int(manifest.get("epoch", -1)),
        n_walkers=len(walker_finds),
        walkers=[int(r["walker"]) for r in walker_finds],
        finds_per_walker={str(int(r["walker"])): int(r["n_finds"]) for r in walker_finds},
        bands_2plus_finds={str(int(r["walker"])): int((np.asarray(r["band_finds"]) >= 2).sum())
                           for r in walker_finds},
        steps_per_walker={str(int(r["walker"])): (
            [int(np.min(r["steps"])), int(np.median(r["steps"])), int(np.max(r["steps"]))]
            if len(r["steps"]) else []) for r in walker_finds},
        converged_frac={str(int(r["walker"])): (
            float(np.mean(r["converged"])) if len(r["converged"]) else 0.0)
            for r in walker_finds},
        walls_per_walker={str(int(r["walker"])): dict(r["walls"]) for r in walker_finds},
        n_finds=int(sum(int(r["n_finds"]) for r in walker_finds)),
        n_sources=len(sources),
        opt_snr=_quantiles([s["opt_snr"] for s in sources]),
        finds_per_band={str(b): c for b, c in zip(*np.unique(
            [s["band"] for s in sources], return_counts=True))} if sources else {},
        tobs=float(tobs), built_on=socket.gethostname(),
        finds_npz=FINDS_NPZ,
    )
    if not sources:
        manifest.update(npz=None, n_components=0, walls=walls)
        _write_json_atomic(os.path.join(out_dir, "DONE.json"), manifest)
        return None
    obs_map = build_observable_map(tobs, transform_fn)
    rows_obs = [to_observable(s["samples"], obs_map) for s in sources]
    tg = time.perf_counter()
    comps = fit_cluster_gmms(rows_obs, n_samples=GMM_SAMPLES,
                             max_comp=GMM_MAX_COMP, min_members=GMM_MIN_MEMBERS,
                             seed=int(seed), gpu=gmm_gpu)
    walls["gmm"] = time.perf_counter() - tg
    packed = pack_gmm_components(comps)
    n = len(sources)
    df = 1.0 / float(tobs)
    f0e = np.array([[s["f0_span_hz"][0] - 2 * df, s["f0_span_hz"][1] + 2 * df]
                    for s in sources])
    meta = dict(
        basis="observable", column_names=wb.OBSERVABLE_COLUMN_NAMES,
        f0_units="Hz", circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        bounded_cols={str(k): list(v) for k, v in OBSERVABLE_BOUNDED_COLS.items()},
        map_params=wb.map_params_from_map(obs_map), tobs=float(tobs),
        df_mhz=df * 1e3, pipeline="GB parallel residual search v1 (para_search)",
        gmm=dict(n_samples=GMM_SAMPLES, max_comp=GMM_MAX_COMP,
                 min_members=GMM_MIN_MEMBERS,
                 n_components=int(sum(len(w) for w in comps[0])),
                 components_basis="physical"),
        epoch=int(manifest["epoch"]),
    )
    payload = dict(
        p=np.ones(n), mult=np.ones(n),
        n_members=np.array([s["samples"].shape[0] for s in sources], dtype=np.int64),
        island_id=np.arange(n, dtype=np.int64), f0_window_edges=f0e,
        src_f0_hz=np.array([s["f0_hz"] for s in sources]),
        src_opt_snr=np.array([s["opt_snr"] for s in sources]),
        src_det_snr=np.array([s["det_snr"] for s in sources]),
        src_band=np.array([s["band"] for s in sources], dtype=np.int64),
        src_n_walkers=np.array([len(s["walkers"]) for s in sources], dtype=np.int64),
        **packed)
    tw = time.perf_counter()
    np.savez_compressed(os.path.join(out_dir, EPOCH_NPZ), meta=json.dumps(meta),
                        **payload)
    walls["write"] = time.perf_counter() - tw
    walls["head_total"] = time.perf_counter() - t0
    manifest.update(npz=EPOCH_NPZ, n_components=int(meta["gmm"]["n_components"]),
                    walls=walls,
                    gmm_device=("cpu" if gmm_gpu is None else f"gpu{int(gmm_gpu)}"))
    _write_json_atomic(os.path.join(out_dir, "DONE.json"), manifest)
    return manifest


def _write_finds_npz(path, walker_finds):
    """Every walker's record into one npz, keys ``w<walker>_<field>`` plus ``walkers``."""
    out = {}
    for r in walker_finds:
        w = int(r["walker"])
        for k, v in r.items():
            if k == "walls":
                out[f"w{w}_walls"] = json.dumps(v)
            elif isinstance(v, np.ndarray):
                out[f"w{w}_{k}"] = v
            else:
                out[f"w{w}_{k}"] = np.asarray(v)
    np.savez_compressed(path, walkers=np.array([int(r["walker"]) for r in walker_finds]),
                        **out)


def load_finds_npz(path) -> list:
    """Inverse of the epoch's ``finds.npz`` (per-walker records)."""
    out = []
    with np.load(path, allow_pickle=False) as d:
        for w in np.asarray(d["walkers"]).tolist():
            rec = {}
            pre = f"w{int(w)}_"
            for k in d.files:
                if k.startswith(pre):
                    key = k[len(pre):]
                    v = d[k]
                    rec[key] = json.loads(str(v)) if key == "walls" else (
                        v.item() if v.ndim == 0 else v)
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# the move's container
# ---------------------------------------------------------------------------
class ParaSearchHolder:
    """The RJ birth law of ``rj_para_search_pe``: the adopted epoch's mixture.

    Duck-typed like ``WarmStartComponents`` (``rvs`` / ``logpdf`` /
    ``n_components``), delegating to the epoch adopted for the CURRENT
    propose. Only the head polls the epoch root (:meth:`poll`); a compute
    rank adopts exactly the epoch the head shipped (:meth:`apply_directive`,
    authoritative: an epoch-less directive empties it). With no epoch the
    move proposes nothing (the head returns before any command). Seeds are
    ``warm_seed(build_seed, tag ^ (epoch + 1))``, per rank and epoch, the
    warm-start self refit's rule. Pickle / deepcopy safe (numpy containers).

    Args:
        epoch_root (str): :func:`para_search_epoch_root` of the store.
        load_kwargs (dict): :meth:`.proposal.WarmStartComponents.from_npz`
            options (``new_tobs``, ``use_cupy``, ``floor_box``, ...).
        build_seed (int, optional): The run's per-rank build seed.
        tag (int, optional): Seed domain tag. Default ``0``.
        name (str, optional): Move name for logs. Default
            ``"rj_para_search_pe"``.

    Attributes:
        container (WarmStartComponents or None): The adopted epoch's mixture.
        epoch (int or None): The adopted epoch.
        tallies (dict): Cumulative propose / birth / death counts.
    """

    is_para_search_holder = True
    ndim = 9

    def __init__(self, *, epoch_root, load_kwargs, build_seed=None, tag=0,
                 name="rj_para_search_pe"):
        self.epoch_root = str(epoch_root)
        self.load_kwargs = dict(load_kwargs or {})
        self.build_seed = build_seed
        self.tag = int(tag)
        self.name = str(name)
        self.container = None
        self.epoch: typing.Optional[int] = None
        self.epoch_info: dict = {}
        self._bad: set = set()
        self.tallies = dict(proposes=0, births=0, viable=0, birth_acc=0,
                            viable_cold=0, birth_acc_cold=0, deaths=0, death_acc=0)

    # -- container surface ------------------------------------------------
    @property
    def active(self):
        """The adopted container; raises ``RuntimeError`` when none is adopted."""
        if self.container is None:
            raise RuntimeError(
                f"{self.name}: no para-search epoch is adopted; the move must "
                "not propose before one is (the head skips such a propose).")
        return self.container

    @property
    def n_components(self) -> int:
        """The adopted container's component count (0 with none)."""
        return 0 if self.container is None else int(self.container.n_components)

    @property
    def use_cupy(self) -> bool:
        """``load_kwargs["use_cupy"]``."""
        return bool(self.load_kwargs.get("use_cupy", False))

    def rvs(self, size=1):
        """Birth draws from the adopted container."""
        return self.active.rvs(size=size)

    def logpdf(self, x):
        """Log density of the adopted container."""
        return self.active.logpdf(x)

    # -- adoption ---------------------------------------------------------
    def epoch_seed(self, k: int):
        """The container seed of epoch ``k``: ``warm_seed(build_seed, tag ^ (k + 1))``."""
        from .self_refit import warm_seed

        return warm_seed(self.build_seed, self.tag ^ (int(k) + 1))

    def _adopt(self, k, manifest, rank) -> bool:
        """Load epoch ``k``'s npz; False (and one warning per epoch) on failure."""
        from .proposal import WarmStartComponents
        from .self_refit import read_manifest

        if manifest is None:
            manifest = read_manifest(self.epoch_root, k)
        if manifest is None:
            if k not in self._bad:
                self._bad.add(k)
                logger.warning("[GB_PARA_SEARCH] rank %s (%s): epoch %d is not "
                               "complete; keeping epoch %s.", rank, self.name,
                               k, self.epoch)
            return False
        try:
            cont = WarmStartComponents.from_npz(
                manifest["npz_path"], seed=self.epoch_seed(k), **self.load_kwargs)
        except Exception as exc:                      # noqa: BLE001
            if k not in self._bad:
                self._bad.add(k)
                logger.warning("[GB_PARA_SEARCH] rank %s (%s): epoch %d failed "
                               "to load (%s: %s); keeping epoch %s.", rank,
                               self.name, k, type(exc).__name__, exc, self.epoch)
            return False
        self.container = cont
        self.epoch = int(k)
        self.epoch_info = {kk: manifest.get(kk) for kk in (
            "n_sources", "n_finds", "n_components", "built_at_clock", "stage",
            "n_walkers")}
        logger.info("[GB_PARA_SEARCH] rank %s adopted epoch %d into %s: %s "
                    "sources (%s finds over %s walkers), %d components.", rank,
                    k, self.name, manifest.get("n_sources"),
                    manifest.get("n_finds"), manifest.get("n_walkers"),
                    cont.n_components)
        return True

    def poll(self, rank=None) -> None:
        """HEAD: adopt the newest COMPLETE epoch if newer than the held one."""
        from .self_refit import latest_complete_epoch

        found = latest_complete_epoch(self.epoch_root, newer_than=self.epoch)
        if found is None:
            return
        k, man = found
        if k not in self._bad:
            self._adopt(k, man, rank)

    def directive(self) -> dict:
        """HEAD: ``{"epoch": the adopted epoch or None}`` for the compute ranks."""
        return {"epoch": self.epoch}

    def apply_directive(self, directive, rank=None) -> None:
        """COMPUTE RANK: hold exactly the head's epoch (``None`` empties).

        Raises:
            RuntimeError: When the head's epoch cannot be loaded here.
        """
        k = (directive or {}).get("epoch")
        if k is None:
            self.container, self.epoch = None, None
            return
        if int(k) != self.epoch and int(k) not in self._bad:
            self._adopt(int(k), None, rank)
        if self.epoch != int(k):
            raise RuntimeError(
                f"[GB_PARA_SEARCH] rank {rank} ({self.name}) cannot load the "
                f"head's epoch {k} from {self.epoch_root} (holds {self.epoch}); "
                "a propose must draw and price from the head's container.")

    def note_census(self, sp, *, move_name=None, rank=None) -> None:
        """Fold one propose's RJ census into the tallies and log the cumulative line."""
        t = self.tallies
        t["proposes"] += 1
        for kk in ("births", "viable", "birth_acc", "viable_cold",
                   "birth_acc_cold", "deaths", "death_acc"):
            t[kk] += int((sp or {}).get(kk, 0) or 0)
        logger.info(
            "[GB_PARA_SEARCH %s] %sthis propose used epoch %s (%d components) | "
            "cumulative proposes %d births viable %d acc %d (%.2e) cold %d/%d "
            "(%.2e) deaths acc %d/%d", move_name or self.name,
            "" if rank is None else f"r{rank} ", self.epoch, self.n_components,
            t["proposes"], t["viable"], t["birth_acc"],
            t["birth_acc"] / max(t["viable"], 1), t["birth_acc_cold"],
            t["viable_cold"], t["birth_acc_cold"] / max(t["viable_cold"], 1),
            t["death_acc"], t["deaths"])


class ParaSearchClock:
    """The build clock: PE iterations seen by the move, journaled to
    ``<root>/clock.json`` so the cadence survives a restart.

    ``tick(gf_iteration)`` adds the iterations elapsed since the previous
    tick (``gf_iteration`` is the stage combine's per-process counter: it
    restarts at 0 in every job, so only its DIFFERENCES are used; the first
    tick of a process adds 0).

    Args:
        root (str): The epoch root (the journal is ``<root>/clock.json``;
            an unreadable journal starts at 0).

    Attributes:
        value (int): PE iterations counted so far.
    """

    def __init__(self, root):
        self.root = str(root)
        self.path = os.path.join(self.root, CLOCK_JSON)
        self.value = 0
        self._last = None
        try:
            with open(self.path) as fh:
                self.value = int(json.load(fh).get("iterations", 0))
        except (OSError, ValueError, TypeError):
            self.value = 0

    def tick(self, gf_iteration) -> int:
        """Advance by the iterations since the last tick (``None`` adds 1); journal; return it."""
        if gf_iteration is None:
            self.value += 1
        else:
            it = int(gf_iteration)
            if self._last is not None and it > self._last:
                self.value += it - self._last
            self._last = it
        try:
            os.makedirs(self.root, exist_ok=True)
            _write_json_atomic(self.path, {"iterations": int(self.value)})
        except OSError:
            pass
        return self.value


# ---------------------------------------------------------------------------
# offline CLI: search a store snapshot's residual
# ---------------------------------------------------------------------------
def empirical_inverse_covariance(res, *, time_blocks=1, layer_smooth=1):
    """``(3, 3, Nf, Nt)`` inverse of the empirical XYZ covariance of ``res``.

    The run's noise model is not reconstructed offline: the covariance is the
    residual's own per-(layer, time block) ``E[r r^T]`` (averaged over
    ``2 * layer_smooth + 1`` neighbouring layers), i.e. the noise PLUS the
    unresolved signal the walker's model still holds -- the same thing the
    run's noise + galactic-foreground fit describes. A loud unfound source
    inflates its own layers' estimate slightly (its SNRs read low).

    Args:
        res (array-like): ``(3, Nf, Nt)`` XYZ WDM residual (active box).
        time_blocks (int, optional): Equal time blocks per layer. Default
            ``1``.
        layer_smooth (int, optional): Neighbouring layers averaged on each
            side. Default ``1``.

    Returns:
        np.ndarray: ``(3, 3, Nf, Nt)`` inverse covariance (a ``1e-9``
        relative diagonal load before inversion), constant within each time
        block.
    """
    res = np.asarray(res, dtype=float)
    nch, nf, nt = res.shape
    edges = np.linspace(0, nt, int(max(1, time_blocks)) + 1).astype(int)
    C = np.zeros((nf, len(edges) - 1, nch, nch))
    for b in range(len(edges) - 1):
        r = res[:, :, edges[b]:edges[b + 1]]
        C[:, b] = np.einsum("cft,dft->fcd", r, r) / max(1, r.shape[-1])
    if layer_smooth > 0:
        Cs = np.zeros_like(C)
        for f in range(nf):
            lo, hi = max(0, f - layer_smooth), min(nf, f + layer_smooth + 1)
            Cs[f] = C[lo:hi].mean(axis=0)
        C = Cs
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / nch
    C = C + 1e-9 * np.maximum(tr, 1e-300) * np.eye(nch)
    iC = np.linalg.inv(C)                                     # (nf, nb, 3, 3)
    out = np.empty((nch, nch, nf, nt))
    for b in range(len(edges) - 1):
        out[:, :, :, edges[b]:edges[b + 1]] = iC[:, b].transpose(1, 2, 0)[..., None]
    return out


class _OneRowHolder:
    """Minimal WDM holder: one residual row + its inverse covariance, on
    the comp's array module (device arrays for a CUDA comp)."""

    def __init__(self, res, invC, xp=np):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(res[None])).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invC[None])).ravel()]

    def __len__(self):
        return 1


def _cli_world(args):
    """The CLI's world: ``(scorer, band_edges, transform, prior, mc_lims, ratio_max, tobs, store)``.

    Reads the store's WDM ``domain_settings``, ``noise_model_identity``
    (``data_t0`` = the WDM t0 unless ``--t0``), ``sub_backend/gb/band_edges``
    and the walker's residual from the residual snapshot; the noise is
    :func:`empirical_inverse_covariance` of that residual. The orbits are
    :class:`~lisatools.detector.ESAOrbits` of ``--orbit-file``, else the
    packaged default ESA file -- not the run's L1 orbits (see the frame note
    of :class:`~lisatools.detector.Orbits`). Exits (``SystemExit``) on a
    non-WDM store, a walker missing from the snapshot or a residual of the
    wrong shape.
    """
    import h5py

    from lisatools.globalfit.hdfbackend import GFHDFBackend
    from lisatools.globalfit.residual_snapshot import snapshot_path

    store = os.path.abspath(args.store)
    be = GFHDFBackend(store)
    wdm = be.read_domain_settings(force_backend=args.backend)
    if wdm is None:
        raise SystemExit(f"{store}: no global_fit/domain_settings (not a WDM store?)")
    ident = be.read_noise_model_identity() or {}
    with h5py.File(store, "r") as f:
        band_edges = np.asarray(f["global_fit/sub_backend/gb/band_edges"][()], dtype=float)
    snap = args.residual or snapshot_path(store)
    with np.load(snap) as d:
        walkers = (list(np.asarray(d["walkers"]).tolist()) if "walkers" in d.files
                   else [int(d["walker"])])
        if int(args.walker) not in walkers:
            raise SystemExit(f"{snap}: walker {args.walker} not in {walkers}")
        key = "residuals" if "residuals" in d.files else "residual"
        res = np.asarray(d[key][walkers.index(int(args.walker))]
                         if key == "residuals" else d[key], dtype=float)
    nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
    nta = int(wdm.Nt_active)
    if res.shape != (3, nfa, nta):
        raise SystemExit(f"{snap}: residual shape {res.shape} != the store's active "
                         f"WDM box {(3, nfa, nta)}")
    data_t0 = ident.get("data_t0")
    if args.t0 is not None:
        wdm.t0 = float(args.t0)
    elif data_t0 is not None:
        wdm.t0 = float(data_t0)
    tb = int(args.noise_time_blocks) if int(args.noise_time_blocks) > 0 \
        else max(1, nta // 512)
    invC = empirical_inverse_covariance(res, time_blocks=tb,
                                        layer_smooth=args.noise_layer_smooth)
    from gbgpu.gbcomps import GBWDMComputations

    from lisatools.detector import ESAOrbits

    orbits = ESAOrbits(args.orbit_file, force_backend=args.backend) if args.orbit_file \
        else ESAOrbits(force_backend=args.backend)
    if args.orbit_dt > 0:
        T = float(wdm.Tobs)
        margin = 64 * int(wdm.Nf) * float(wdm.data_dt)
        t_arr = np.arange(-margin, T + margin + args.orbit_dt, args.orbit_dt) + float(wdm.t0)
        orbits._configure(t_arr=t_arr)
    chunked = GBWDMComputations(
        wdm, t_ref=float(args.t_ref if args.t_ref is not None else wdm.t0),
        Nt_sub=args.nt_sub, n_pad=args.n_pad, N_sparse=args.n_sparse,
        N_cp_sig=args.n_cp_sig, N_cp_orbit=args.n_cp_orbit, orbits=orbits,
        tdi_config=f"{args.tdi_gen}{'nd' if args.tdi_gen == 2 else 'st'} generation",
        force_backend=args.backend, d_d=0.0, tdi_type="XYZ")
    chunked.convert_to_ra_dec = False
    holder = _OneRowHolder(res, invC, xp=chunked.xp)
    sighet = None
    if args.scorer in ("auto", "sighet"):
        try:
            from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

            sighet = GBSignalHetComputations.for_band_engine(
                chunked, v3_n_nodes=args.sighet_v3_nodes, v4_knots=args.sighet_v4_knots,
                v4_band=16, v5=1)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("sig-het F-stat unavailable (%s); using the chunked scorer", exc)
            sighet = None
    tobs = float(wdm.Tobs)
    scorer = ResidualScorer(chunked, holder, row=0, sighet=sighet, tobs=tobs,
                            mode=args.scorer)
    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

    mc_lims = (args.mc_min, args.mc_max)
    transform = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                            use_distance=True, mc_lims=mc_lims)
    uc = chunked.xp is not np
    prior = build_search_prior(default_search_priors_in(
        dist_lims=(args.dist_min, args.dist_max), ratio_max=args.ratio_max,
        use_cupy=uc), use_cupy=uc)
    return scorer, band_edges, transform, prior, mc_lims, args.ratio_max, tobs, store


def _held_report(store, walker, finds, tobs, radius_bins):
    """How many finds the walker's own last cold row already holds (should be ~0).

    ~0 because the residual has the walker's leaves subtracted. A find is
    "held" when a leaf of the store's last row lies within
    ``radius_bins / tobs`` of its f0. Returns ``None`` when the store cannot
    be read.
    """
    try:
        import h5py

        with h5py.File(store, "r") as f:
            g = f["global_fit"]
            it = int(g.attrs["iteration"]) - 1
            if it < 0 or "chain" not in g or "gb" not in g["chain"]:
                return None
            x = g["chain/gb"][it, 0, 0, int(walker)]
            ind = g["inds/gb"][it, 0, 0, int(walker)]
        f0 = np.asarray(x[ind, 1], dtype=float) * 1e-3
    except Exception:                                    # noqa: BLE001
        return None
    r = float(radius_bins) / float(tobs)
    held = [bool(np.any(np.abs(f0 - v) <= r)) for v in finds["f0_hz"]]
    return int(np.sum(held))


def _report_text(finds, *, walker, store, held, cfg, edges):
    """The CLI's ``report_w<walker>.txt``: knobs, per-band table, per-find table."""
    lines = [f"GB parallel residual search: walker {walker} of {store}",
             f"knobs: {json.dumps(cfg.as_dict(), sort_keys=True)}",
             f"bands searched {len(finds['bands'])}, converged "
             f"{int(np.sum(finds['converged']))}, steps min/median/max "
             f"{int(np.min(finds['steps'])) if len(finds['steps']) else 0}/"
             f"{int(np.median(finds['steps'])) if len(finds['steps']) else 0}/"
             f"{int(np.max(finds['steps'])) if len(finds['steps']) else 0}",
             f"finds {int(finds['n_finds'])} (gate A {finds['n_gate_a']}, gate B "
             f"failed {finds['n_gate_b_fail']}, duplicates {finds['n_dedup']}); "
             f"bands with 2+ finds {int(np.sum(np.asarray(finds['band_finds']) >= 2))}",
             f"walls: fstat {finds['walls']['fstat']:.1f} s, full {finds['walls']['full']:.1f} s",
             f"already held by the walker's last cold row: {held}", "",
             "band  f_lo[mHz]   f_hi[mHz]   steps conv  finds",]
    for b, s, c, nf in zip(finds["bands"], finds["steps"], finds["converged"],
                           finds["band_finds"]):
        if nf or not c:
            lines.append(f"{int(b):5d} {edges[b] * 1e3:10.6f} {edges[b + 1] * 1e3:10.6f}"
                         f" {int(s):6d} {'y' if c else 'n':>4} {int(nf):5d}")
    lines += ["", "find  band   f0[mHz]       fdot[Hz/s]   F1      opt    det    "
              "opt_med det_med edge"]
    for i in range(int(finds["n_finds"])):
        lines.append(
            f"{i:4d} {int(finds['band'][i]):5d} {finds['f0_hz'][i] * 1e3:12.7f} "
            f"{finds['fdot'][i]:12.4e} {finds['F1'][i]:7.1f} {finds['opt_snr'][i]:6.2f} "
            f"{finds['det_snr'][i]:6.2f} {finds['opt_med'][i]:6.2f} "
            f"{finds['det_med'][i]:6.2f} {'E' if finds['edge'][i] else '-'}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """The offline CLI: search one walker of a store snapshot's residual.

    Writes ``<out>/report_w<walker>.txt`` (also printed) and
    ``<out>/finds_w<walker>.npz``; with ``--merge-epoch`` also a one-walker
    ``<out>/epoch/parasearch.npz`` + ``DONE.json`` (:func:`write_epoch`).
    The search knobs are the ``GB_PARA_SEARCH_*`` env
    (:meth:`ParaSearchConfig.from_env`; ``--scorer`` and ``--seed``
    override); the comp sizing (``--nt-sub``, ``--n-pad``, ``--n-sparse``,
    ``--n-cp-sig``, ``--n-cp-orbit``, ``--sighet-v4-knots``,
    ``--sighet-v3-nodes``), ``--t-ref`` and the prior boxes (``--mc-min`` /
    ``--mc-max`` / ``--dist-min`` / ``--dist-max`` / ``--ratio-max``)
    default to the env-resolved stock all-sources GB settings. Other
    options: ``--store``, ``--walker``, ``--bands a:b``, ``--out``,
    ``--residual``, ``--backend`` (default ``cpu``), ``--gpu``, ``--t0``,
    ``--orbit-file``, ``--orbit-dt``, ``--tdi-gen`` (default 2),
    ``--noise-time-blocks``, ``--noise-layer-smooth``.

    Args:
        argv (list of str, optional): Arguments; ``None`` = ``sys.argv[1:]``.

    Returns:
        dict: The walker's finds record (:meth:`ParaSearchBuilder.run`).
    """
    ap = argparse.ArgumentParser(
        description="GB parallel residual search on a store snapshot's residual "
                    "(docs/gb-para-search.md, section 8).")
    ap.add_argument("--store", required=True, help="the run's h5 store (live or *_extract.h5)")
    ap.add_argument("--walker", type=int, required=True, help="cold walker index")
    ap.add_argument("--bands", default=None,
                    help="band range a:b (half open; default the interior bands)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--residual", default=None,
                    help="residual snapshot npz (default <store>_residual_snapshot.npz)")
    ap.add_argument("--backend", default="cpu", help="cpu / cuda12x / ...")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device for the sampler")
    ap.add_argument("--scorer", default=None, help="auto / sighet / chunked")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--merge-epoch", action="store_true",
                    help="also fit the GMM and write a one-walker epoch npz")
    # comp / basis settings (defaults = the stock GB settings, env-resolved)
    ap.add_argument("--t0", type=float, default=None,
                    help="WDM t0 (default the store's noise_model_identity data_t0)")
    ap.add_argument("--t-ref", type=float, default=None,
                    help="GB t_ref (default the stock GB t0, else the WDM t0)")
    ap.add_argument("--orbit-file", default=None)
    ap.add_argument("--orbit-dt", type=float, default=0.0,
                    help=">0: configure the orbits on a coarse window (laptop)")
    ap.add_argument("--tdi-gen", type=int, default=2)
    ap.add_argument("--noise-time-blocks", type=int, default=0,
                    help="empirical noise time blocks per layer (0 = Nt_active // 512)")
    ap.add_argument("--noise-layer-smooth", type=int, default=1)
    args, _rest = ap.parse_known_args(argv)

    try:
        from lisatools.globalfit.stock.erebor.variants.all_sources import AllSourcesGBSettings

        gbs = AllSourcesGBSettings()
    except Exception:                                        # noqa: BLE001
        gbs = None

    def _g(name, default):
        """The stock GB setting ``name``, else ``default``."""
        return getattr(gbs, name, default) if gbs is not None else default

    for name, default, attr in (("nt_sub", 256, "nt_sub"), ("n_pad", 32, "n_pad"),
                                ("n_sparse", 256, "n_sparse"), ("n_cp_sig", 48, "n_cp_sig"),
                                ("n_cp_orbit", 32, "n_cp_orbit")):
        ap.add_argument(f"--{name.replace('_', '-')}", type=int, default=int(_g(attr, default)))
    mcl = list(_g("m_chirp_lims", None) or [0.001, 1.0])
    dl = list(_g("dist_lims", None) or [0.001, 40.0])
    ap.add_argument("--mc-min", type=float, default=float(mcl[0]))
    ap.add_argument("--mc-max", type=float, default=float(mcl[-1]))
    ap.add_argument("--dist-min", type=float, default=float(dl[0]))
    ap.add_argument("--dist-max", type=float, default=float(dl[-1]))
    ap.add_argument("--ratio-max", type=float,
                    default=float(_g("fdot_astro_ratio_max", None) or 1.0))
    ap.add_argument("--sighet-v4-knots", type=int, default=int(_g("sighet_v4_knots", 128)))
    ap.add_argument("--sighet-v3-nodes", type=int, default=int(_g("sighet_v3_n_nodes", 64)))
    args = ap.parse_args(argv)
    if args.t_ref is None and gbs is not None and getattr(gbs, "t0", None) is not None:
        args.t_ref = float(gbs.t0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ParaSearchConfig.from_env(scorer=args.scorer, seed=args.seed)
    args.scorer = cfg.scorer
    t0 = time.perf_counter()
    scorer, edges, transform, prior, mc_lims, ratio_max, tobs, store = _cli_world(args)
    builder = ParaSearchBuilder(scorer, band_edges_hz=edges, transform_fn=transform,
                                prior=prior, mc_lims=mc_lims, ratio_max=ratio_max,
                                cfg=cfg, tag=f"cli walker {args.walker}",
                                gpu=(args.gpu if args.gpu is not None
                                     else (0 if scorer.xp is not np else None)))
    if args.bands:
        a, b = (int(v) for v in args.bands.split(":"))
        bands = np.arange(a, b)
    else:
        bands = None
    finds = builder.run(bands=bands, walker=args.walker)
    os.makedirs(args.out, exist_ok=True)
    held = _held_report(store, args.walker, finds, tobs, cfg.split_bins)
    _write_finds_npz(os.path.join(args.out, f"finds_w{args.walker}.npz"), [finds])
    txt = _report_text(finds, walker=args.walker, store=store, held=held, cfg=cfg,
                       edges=edges)
    with open(os.path.join(args.out, f"report_w{args.walker}.txt"), "w") as fh:
        fh.write(txt)
    print(txt)
    if args.merge_epoch:
        sources = merge_finds([finds], tobs=tobs, radius_bins=cfg.split_bins)
        man = write_epoch(os.path.join(args.out, "epoch"), sources, [finds], tobs=tobs,
                          transform_fn=transform,
                          manifest_extra=dict(epoch=0, stage="offline", cli=True))
        print(f"epoch npz: {None if man is None else os.path.join(args.out, 'epoch', EPOCH_NPZ)}"
              f" ({0 if man is None else man['n_components']} components)")
    print(f"total wall {time.perf_counter() - t0:.1f} s")
    return finds


if __name__ == "__main__":
    main()
