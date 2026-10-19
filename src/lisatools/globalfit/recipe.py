"""Recipe orchestration for sequencing global-fit sampling stages.

This module is the single home for the *installable* recipe machinery: the
:class:`Recipe` engine, the generic recipe-step base classes
(:class:`SearchRecipeStep` / :class:`PERecipeStep` / :class:`RJRecipeStep`), the
per-source move-builder hierarchy (:class:`SourceMoveBuilder` and subclasses),
and the injection / catalogue helpers. Settings files under ``global_fit_input/``
*compose* these into a concrete recipe inside their ``setup_recipe`` — they do not
re-implement the machinery. (Folded in from the retired ``recipe_steps.py``.)
"""

from __future__ import annotations

import os
import time
import logging
import typing
from copy import deepcopy
import dataclasses
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

# DataResidualArray is now a deprecation shim; we pass DomainBase children
# (or raw arrays via the AnalysisContainer/template APIs) directly.
from lisatools.domains import FDSettings, STFTSettings, WDMSettings
from lisatools.response.tdiconfig import TDIConfig

from bbhx.utils.transform import SSB_to_LISA
from gbgpu.gbgpu import GBGPU
from gbgpu.gbcomps import STFTGBComputations
from gbgpu.gb_likelihood import make_band_likelihood_engine
from eryn.moves.tempering import TemperatureControl, make_ladder
from eryn.prior import ProbDistContainer

from ..sources.utils import icrs_to_ecliptic, evolve_galactic_binary
from ..utils.utility import asnumpy
from .moves import (
    PSDMove,
    ResidualAddOneRemoveOneMove,
    # GBSpecialRJPriorMove,
    # GBSpecialRJSerialSearchMCMC,
    # GBSpecialRJRefitMove,
)
from .moves.gbbandstructure import stft_store_window_layout
from .moves.gbspecialmove import (
    GBSpecialRJPriorMove, 
    GBSpecialRJSerialSearchMCMC,
    GBSpecialRJRefitMove,
    GBSpecialBase   
)
# from .moves.gbspecialstretch import GBSpecialBase

# Type-only imports. These live under TYPE_CHECKING because ``run`` imports this
# module (``run.py`` -> ``from .recipe import Recipe``); importing ``.run`` /
# ``.engine`` / ``.state`` / ``.stock.erebor`` / ``..analysiscontainer`` at module
# scope here would create an import cycle. ``from __future__ import annotations``
# turns every annotation below into a string, so these names are only needed by
# type checkers, never at runtime.
if typing.TYPE_CHECKING:
    from .engine import Setup, GlobalFitEngine
    from .run import CurrentInfoGlobalFit
    from .state import GFState
    from .stock.erebor import GBSetup, GeneralSetup
    from ..analysiscontainer import AnalysisContainerArray

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664
MOJITO_AVERAGE_ARMLENGTH = 2493162305.42235

class Recipe:
    """Ordered sequence of :class:`RecipeStep` instances driving the sampler.

    A ``Recipe`` is iterated by the global-fit driver. At each call it asks the
    current step's stopping function whether to advance, and on advance it
    invokes the next step's ``setup_run`` to reconfigure the sampler.
    """

    def __init__(self):
        self.recipe = []
        self.backend_added = False
        self._current_iter = 0
        self._current_recipe_step = None
        self._has_setup_first_step = False

    @property
    def backend(self):
        """Backend object that records recipe-step completion."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend
        self.backend_added = True

    def add_recipe_component(self, adjust_fn, name=None):
        """Append a recipe step.

        Args:
            adjust_fn: A :class:`RecipeStep` (or compatible object) implementing
                ``setup_run`` and ``stopping_function``.
            name: Optional human-readable name. If ``None``, a default name is
                assigned based on the current recipe length.
        """
        if name is None:
            name = f"recipe step {len(self.recipe) + 1}"
        self.recipe.append({"name": name, "adjust": adjust_fn, "status": False})

    def to_file(self):
        """Return a dict mapping recipe-step names to their completion status."""
        _tmp = {recipe_step["name"]: recipe_step["status"] for recipe_step in self.recipe}
        return _tmp

    def __next__(self):
        """Advance the internal cursor past any completed steps."""
        while self._current_iter < len(self.recipe):
            # False means it is not finished
            if self.recipe[self._current_iter]["status"]:
                self._current_iter += 1

            else:
                break

        if self._current_iter < len(self.recipe):
            self._current_recipe_step = self.recipe[self._current_iter]

    def setup_first_recipe_step(self, iteration, last_sample, sampler):
        """Configure the sampler for the very first (incomplete) recipe step.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The :class:`GlobalFitEngine` (or eryn-compatible) sampler.

        Raises:
            ValueError: If the recipe has already been completed.
        """
        assert not self._has_setup_first_step
        # move to next recipe step
        next(self)
        if self._current_iter >= len(self.recipe):
            raise ValueError("Recipe is already finished.")

        self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
        self._has_setup_first_step = True

    @property
    def current_recipe_step(self):
        """The active recipe-step record (a dict of ``name``/``adjust``/``status``)."""
        return self._current_recipe_step

    def __call__(self, iteration, last_sample, sampler):
        """Evaluate the current step's stopping criterion and advance if met.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The active sampler.

        Returns:
            ``True`` if the entire recipe has finished, ``False`` otherwise.
        """
        stop_here = self._current_recipe_step["adjust"].stopping_function(
            iteration, last_sample, sampler
        )
        if stop_here:
            self.backend.completed_recipe_step(self._current_recipe_step["name"])
            self._current_recipe_step["status"] = True
            next(self)

            if self._current_iter >= len(self.recipe):
                return True
            self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)

        return False


class RecipeStep:
    """Abstract base for a single stage in a :class:`Recipe`.

    Each subclass must define a ``setup_run`` method that configures the
    sampler when the step becomes active and a ``stopping_function`` that
    decides when to advance to the next step.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``. Defaults to uniform.
    """

    def __init__(self, moves=None, weights=None):
        if moves is not None:
            self.moves = moves
            if weights is not None:
                self.weights = weights

    def __repr__(self):
        return f"RecipeStep with moves: {self.moves} and weights: {self.weights}"

    @property
    def moves(self):
        """List of MCMC moves used by this step."""
        if not hasattr(self, "_moves"):
            raise ValueError("Must add moves for this recipe step.")
        return self._moves

    @moves.setter
    def moves(self, moves):
        self._moves = moves

    @property
    def weights(self):
        """List of weights corresponding to :attr:`moves`. Uniform by default."""
        if not hasattr(self, "_weights"):
            self._weights = [1.0 / len(self.moves) for _ in self.moves]
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def setup_run(self, iteration, last_sample, sampler):
        """Configure ``sampler`` for the start of this recipe step."""
        raise NotImplementedError

    def stopping_function(self, iteration, last_sample, sampler):
        """Return ``True`` when this recipe step should be considered done."""
        raise NotImplementedError

class BaseRecipeStep(RecipeStep):
    """Default :class:`RecipeStep` that simply assigns moves to the sampler.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``.
    """

    def __init__(self, moves=None, weights=None):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        """Install :attr:`moves`/:attr:`weights` on the sampler.

        Each move that lacks an explicit periodicity setting inherits the
        sampler's periodicity.
        """
        for move in self.moves:
            if sampler.periodic is not None and move.periodic is None:
                logger.debug(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic

        sampler.moves = self.moves
        sampler.weights = self.weights


# ======================================================================
# Installable recipe steps, injection helpers, and move builders
# (folded in from the retired recipe_steps.py).
# ======================================================================


class SearchRecipeStep(BaseRecipeStep):
    """Recipe step that completes immediately (one-shot search/initialization).

    Used when the stopping criterion is embedded inside the move itself
    rather than at the recipe level.
    """

    def stopping_function(self, *args, **kwargs):
        """Always stop after one call."""
        return True


class PERecipeStep(BaseRecipeStep):
    """Recipe step that runs indefinitely (ongoing parameter estimation)."""

    def stopping_function(self, *args, **kwargs):
        """Never stop on its own — relies on outer stopping logic."""
        return False


class IterationCountRecipeStep(BaseRecipeStep):
    """Recipe step that runs for N iterations before moving to the next step."""

    def __init__(self, moves: list, weights: typing.Optional[list] = None, num_iters: int = 100):
        super().__init__(moves, weights)
        self.num_iters = num_iters

    def setup_run(self, iteration, last_sample, sampler):
        super().setup_run(iteration, last_sample, sampler)

        self.start_iteration = iteration

    def stopping_function(self, iteration, last_sample, sampler):
        """Stop after num_iters iterations."""
        stop = (iteration - self.start_iteration) >= self.num_iters
        if stop:
            logger.info(f"IterationCountRecipeStep stopping after {self.num_iters} iterations.")

        return stop


class RJRecipeStep(BaseRecipeStep):
    """Reversible-jump recipe step that stops once GB leaf count plateaus.

    Args:
        convergence_iter: Window length used to compare older vs newer
            cold-chain max leaf counts.
        thin_by: Forwarded thinning factor applied to the sampler.
    """

    def __init__(
        self,
        *args,
        convergence_iter: int = 5,
        thin_by: int = 1,
        plateau_branch: str = "gb",
        convergence_fn: typing.Callable | None = None,
        **kwargs
    ):
        BaseRecipeStep.__init__(self, *args, **kwargs)
        self.convergence_iter = convergence_iter
        self.thin_by = thin_by
        # Branch whose cold-chain leaf count is monitored for the plateau test.
        # Defaults to ``"gb"`` (the historical hardcoded value); set it per recipe
        # in a settings file to reuse this step for another RJ branch.
        self.plateau_branch = plateau_branch
        # Optional full override of the stopping criterion (``(i, sample, sampler)
        # -> bool``); lets a settings file define convergence per recipe without
        # editing this class.
        self.convergence_fn = convergence_fn

    def stopping_function(
        self,
        i,
        sample,
        sampler: GlobalFitEngine
    ) -> bool:
        """Stop when the cold chain stops growing in number of leaves."""

        if self.convergence_fn is not None:
            return self.convergence_fn(i, sample, sampler)

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration

        assert isinstance(current_iter, (int, np.integer))

        stop = False
        if current_iter > self.convergence_iter:
            #? Actual convergence should be related to the same number of sources above SNR XX for Y itterations
            nleaves_cc = sampler.backend.get_nleaves(
                branch_names=[self.plateau_branch], temp_index=0
            )[self.plateau_branch]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[:-self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter:].max()

            if nleaves_cc_max_old >= nleaves_cc_max_new:
                stop = True

            else:
                stop = False

            
            dur = (time.perf_counter() - self.st) / 3600.0  # hours
            print(f"Previous nleaves: {nleaves_cc_max_old} --> new nleaves: {nleaves_cc_max_new}")
            print(f"TIME SINCE START: {dur} hours")

        return stop
        
    def setup_run(
        self,
        iteration,
        last_sample,
        sampler: GlobalFitEngine
    ):
        """Configure the sampler for this RJ recipe step (moves, weights, thinning)."""
        # TODO: maybe make this the default setup
        sampler.moves = self.moves
        sampler.weights = self.weights
        sampler.yield_step = self.thin_by
        sampler.checkpoint_step = self.thin_by
        # sampler.override_thin_by = self.thin_by --> # TODO check this one
        
        for move in self.moves: 
            if sampler.periodic is not None and move.periodic is None:
                print(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic
            if sampler.temperature_control is not None and move.temperature_control is None:
                print(f"Setting temperature control of move {move} to {sampler.temperature_control}")
                move.temperature_control = sampler.temperature_control
            
            # TODO: do we also need to set these? I think the current settings setup has ntemps covered, not sure about temp_cntrl            
            # move.ntemps = sampler.ntemps 
            

def scatter_around_injection(
    state: GFState,
    branch_name: str,
    injection_params: np.ndarray,
    spread: float | np.ndarray,
    reverse_transform: typing.Callable | None = None,
    betas: np.ndarray | None = None,
    priors: ProbDistContainer | None = None,
    max_resample_tries: int = 50,
):
    """
    Initialize branch coordinates by scattering walkers around injection parameters.

    For each leaf, draws coordinates from a multivariate Gaussian centered on
    the (transformed) injection parameters.  Higher-temperature chains receive
    proportionally wider scatter when ``betas`` is provided.  Initialized
    leaves are marked as active (``inds = True``).

    When ``priors`` is supplied, any draw that lies outside the prior support
    (``logpdf == -inf``) is rejected and redrawn.  This is essential for
    sampling bases that contain ``arcsin``/``arccos`` transforms (e.g. MBH
    ``sin_beta``, ``cos_iota``) where an out-of-support initial coordinate
    silently produces NaN once the transform pipeline runs, eventually
    surfacing as a CUDA illegal-memory-access in downstream kernels.

    The function modifies ``state`` in-place, so it can be called from
    ``setup_recipe`` (before MCMC) or from a ``RecipeStep.setup_run``
    (between recipe phases).

    Parameters
    ----------
    state : GFState
        Sampler state to modify in-place.
    branch_name : str
        Name of the branch to initialize (e.g. ``"mbh"``, ``"emri"``).
    injection_params : ndarray
        True source parameters in the **physical** (waveform) basis.
        Shape ``(ndim_phys,)`` for a single leaf, or
        ``(nleaves, ndim_phys)`` for multiple leaves.
    spread : float, ndarray
        Controls the width of the Gaussian scatter (in sampling basis).

        * *scalar* -- isotropic standard deviation for every parameter.
        * *1-D array* ``(ndim,)`` -- per-parameter standard deviations.
        * *2-D array* ``(ndim, ndim)`` -- full covariance matrix
          (shared across leaves).
        * *3-D array* ``(nleaves, ndim, ndim)`` -- per-leaf covariance
          matrices.
    reverse_transform : callable, optional
        Converts a single parameter vector from physical basis to
        sampling basis: ``(ndim_phys,) -> (ndim_sampling,)``.
        If *None*, ``injection_params`` are assumed to already be in
        the sampling basis.
    betas : ndarray of shape ``(ntemps,)``, optional
        Inverse-temperature ladder.  When provided the covariance for
        temperature index *t* is scaled by ``1 / betas[t]`` so that
        hotter chains start with a wider scatter.
    priors : ProbDistContainer, optional
        Prior container for ``branch_name``.  When given, walker draws
        outside the prior support are rejected and redrawn (up to
        ``max_resample_tries`` per walker).  Without it, this routine can
        seed walkers that lie outside arcsin/arccos domains, producing
        NaNs once the transform pipeline runs.
    max_resample_tries : int, optional
        Hard cap on resampling attempts per walker before raising.  Only
        used when ``priors`` is supplied.  Default 50 — for any reasonable
        scatter and prior, this is wildly more than needed; the cap exists
        only to surface pathological configs (e.g. the entire scatter
        landing outside the prior) instead of looping forever.
    """
    # TODO: make this better
    coords = state.branches_coords[branch_name]
    ntemps, nwalkers, nleaves_max, ndim = coords.shape

    injection_params = np.atleast_2d(np.asarray(injection_params, dtype=float))
    # Physical → sampling basis
    if reverse_transform is not None:
        injection_sampling = np.array([reverse_transform(p) for p in injection_params])
    else:
        injection_sampling = injection_params

    nleaves_init = injection_sampling.shape[0]
    assert (
        nleaves_init <= nleaves_max
    ), f"More injection leaves ({nleaves_init}) than nleaves_max ({nleaves_max})"
    assert (
        injection_sampling.shape[-1] == ndim
    ), f"Injection ndim ({injection_sampling.shape[-1]}) != branch ndim ({ndim})"

    # Build covariance matrix/matrices
    spread = np.asarray(spread, dtype=float)
    if spread.ndim == 0:
        cov = spread.item() ** 2 * np.eye(ndim)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 1:
        cov = np.diag(spread**2)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 2:
        covs = np.tile(spread, (nleaves_init, 1, 1))
    elif spread.ndim == 3:
        assert spread.shape == (nleaves_init, ndim, ndim)
        covs = spread
    else:
        raise ValueError(f"spread must be scalar, 1-D, 2-D, or 3-D; got shape {spread.shape}")

    if betas is not None:
        logger.info(f"Scaling initial covariance by betas: {betas}")

    leaf_prior = priors[branch_name] if priors is not None else None

    for leaf in range(nleaves_init):
        center = injection_sampling[leaf]
        leaf_cov = covs[leaf]
        for t in range(ntemps):
            if betas is not None:
                scaled_cov = leaf_cov / max(betas[t], 1e-10)
            else:
                scaled_cov = leaf_cov

            draws = np.random.multivariate_normal(center, scaled_cov, size=nwalkers)

            if leaf_prior is not None:
                bad = ~np.isfinite(leaf_prior.logpdf(draws))
                tries = 0
                while bad.any():
                    if tries >= max_resample_tries:
                        n_bad = int(bad.sum())
                        raise RuntimeError(
                            f"scatter_around_injection: leaf={leaf} temp={t}: "
                            f"{n_bad}/{nwalkers} walkers still outside prior support "
                            f"after {max_resample_tries} resample passes. "
                            f"Injection sampling-basis params = {center.tolist()}. "
                            f"Likely the injection sits on / outside a prior edge, or "
                            f"the scatter is too wide for the prior range."
                            f"Last resampled points (showing up to 10): {draws[bad][:10].tolist()}"
                        )
                    redraws = np.random.multivariate_normal(
                        center, scaled_cov, size=int(bad.sum())
                    )
                    draws[bad] = redraws
                    bad = ~np.isfinite(leaf_prior.logpdf(draws))
                    tries += 1

            coords[t, :, leaf] = draws

        state.branches_inds[branch_name][:, :, leaf] = True


def mbh_catalogue_to_sampling_basis(
    catalogue_entry: dict,
    trim_duration: float = 0.0,
    to_lisa_frame: typing.Optional[callable] = None,
) -> np.ndarray:
    """Convert a single Mojito MBHB catalogue entry to MBH sampling basis.

    The sampling basis is:
    ``[logM, q, s1z, s2z, dist, phi_ref, cos_iota, psi, lam, sin_beta, t_plunge]``

    By default sky / polarization / time parameters are returned raw in
    **ICRS** (ra, sin_dec, psi_icrs; time SSB) — the sprint-wide run frame.
    If ``to_lisa_frame`` is provided (a partial of
    :func:`lisatools.sources.utils.to_lisa_frame` with ``orbits`` bound),
    they are instead converted to the LISA frame consistently with the
    orbits' own frame (``orbits.frame``), matching the sampling-side
    inverse transform used by the MBH transform container.

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for one MBHB source, as
        stored by ``L1DataLoader.catalogue['MBHB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(11,)`` in the MBH sampling basis
        (SSB ecliptic frame for sky/time parameters).
    """
    m1 = float(catalogue_entry["PrimaryMassSSBFrame"])
    m2 = float(catalogue_entry["SecondaryMassSSBFrame"])

    # Ensure m1 >= m2
    if m2 > m1:
        m1, m2 = m2, m1

    logM = np.log(m1 + m2)
    q = m2 / m1
    Q = m1 / m2
    logq = np.log(q)

    s1z = float(catalogue_entry["PrimarySpinCompZ"])
    s2z = float(catalogue_entry["SecondarySpinCompZ"])
    dist = float(catalogue_entry["LuminosityDistance"]) / 1e3  # Mpc -> Gpc
    phi_ref = float(catalogue_entry["PhaseReferenceSourceFrame"]) % (2 * np.pi)
    cos_iota = np.cos(float(catalogue_entry["InclinationAngle"]))

    # Sky coordinates: catalogue is ICRS
    ra = float(catalogue_entry["RightAscension"]) % (2 * np.pi)
    dec = float(catalogue_entry["Declination"])
    sin_dec = np.sin(dec)
    psi_icrs = float(catalogue_entry["PolarisationAngle"]) % np.pi  # ensure polarization is within [0, pi]
    t_ssb = float(catalogue_entry["TimeCoalescencePhenomTPHMSSBFrame"])

    if to_lisa_frame is not None:
        # ``to_lisa_frame`` interprets its input angles in the frame the orbit
        # positions are expressed in, and the sampling-side inverse
        # (``from_lisa_frame`` -> alpha, delta, psi via the MBH transform
        # container) hands its output directly to the ICRS-frame waveform
        # basis.  The input angles must therefore be given in the orbits' own
        # frame: no ecliptic detour for ICRS orbits.
        orbits = getattr(to_lisa_frame, "keywords", {}).get("orbits")
        frame = getattr(orbits, "frame", "icrs")
        if frame == "icrs":
            lam, beta, psi = ra, dec, psi_icrs
        elif frame == "ecliptic":
            lam, beta, psi = icrs_to_ecliptic(ra, dec, psi_icrs)
        else:
            raise ValueError(
                f"Unsupported orbits frame {frame!r} for LISA-frame conversion."
            )
        t_L, lam_L, beta_L, psi_L = to_lisa_frame(t_ssb, lam, beta, psi)
        return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_L % np.pi, lam_L % (2 * np.pi), np.sin(beta_L), t_L])

    # ICRS sampling basis (stft_tof + 2026-06 run-frame directive): sky and
    # polarization are kept in ICRS (ra, sin_dec, psi_icrs); time stays SSB.
    # Erebor's stock MBH transform (MBHSetup.init_sampling_info /
    # make_mbh_transform_container) uses the same direct-ICRS basis.
    return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_icrs, ra, sin_dec, t_ssb])


def emri_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Convert a single Mojito EMRI catalogue entry to EMRI sampling basis.

    The sampling basis is:
    ``["logm1", "m2", "a", "p0", "e0", "dist", "cosqK", "phiK", "Phi_phi0", "Phi_r0", "alpha", "sin_delta",]``

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for one EMRI source, as
        stored by ``L1DataLoader.catalogue['EMRI'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(12,)`` in the EMRI sampling basis
        (ICRS for sky parameters).
    """
    m1 = float(catalogue_entry['PrimaryMassSSBFrame'])
    m2 = float(catalogue_entry['SecondaryMassSSBFrame'])
    a = float(catalogue_entry['PrimarySpinParameter'])
    p0 = float(catalogue_entry['SemiLatusRectum'])
    e0 = float(catalogue_entry['Eccentricity'])
    xI0= float(np.cos(catalogue_entry['InclinationAngle']))
    dist = float(catalogue_entry['LuminosityDistance'] * 1e-3) # convert to Gpc
    ra = float(catalogue_entry['RightAscension']) % (2 * np.pi)
    dec = float(catalogue_entry['Declination'])

    qK = float(catalogue_entry['PolarAnglePrimarySpin'])
    phiK = float(catalogue_entry['AzimuthalAnglePrimarySpin']) % (2 * np.pi)

    Phi_phi0 = float(catalogue_entry['AzimuthalPhase']) % (2 * np.pi)
    Phi_r0 = float(catalogue_entry['RadialPhase']) % (2 * np.pi)

    return np.array([np.log(m1), m2, a * xI0, p0, e0, dist, np.cos(qK), phiK, Phi_phi0, Phi_r0, ra, np.sin(dec)])


def gb_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Converts the (V)GB catalogue entries to the sampling basis. 
    The index 0 in f0 and phi0 refer to the frequency and phase at the start of the data.

    The sampling basis is:
    ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for all (V)GBs, as
        stored by ``L1DataLoader.catalogue['(V)GB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(8,)`` in the (V)GB sampling basis
        (ICRS or LISA frame for sky/time parameters).
    """
    amp = np.array(catalogue_entry["Amplitude"])
    logA = np.log(amp)

    # VALIDATED mojito GB convention (scripts/gb/gb_mojito_match.py +
    # gb_mojito_mcmc_three_ways.py, mm ~ 1e-8 vs band-passed data):
    # catalogue params are consumed AT the catalogue reference epoch
    # (TimeReferenceSSBFrame == MOJITO_REFERENCE_TIME) with NO trim
    # evolution -- the GB kernels' ``t_ref`` is that same epoch. The
    # PHYSICAL phase is phi0 = +TrueAnomaly; the sampling basis stores
    # -phi0 because the transform container flips the sign
    # (``phi0: x -> -x``, JaxGB convention).
    del trim_duration  # accepted for signature compat; anchor is REF
    f_init = np.array(catalogue_entry["GW22FrequencySSBFrame"])
    fdot = np.array(catalogue_entry["GW22FrequencyDerivativeSourceFrame"])
    phi_init = (np.array(catalogue_entry["TrueAnomaly"])) % (2 * np.pi)

    f0_mHz = f_init * 1e3
    cos_iota = np.cos(np.array(catalogue_entry["InclinationAngle"]))# % (np.pi)

    ra = np.array(catalogue_entry["RightAscension"]) # alpha
    dec = np.array(catalogue_entry["Declination"]) # delta
    psi_icrs = np.array(catalogue_entry["PolarisationAngle"]) % np.pi  # ensure polarization is within [0, pi]
    # lam_ecl, beta_ecl, psi_ecl= icrs_to_ecliptic(ra, dec, psi_icrs)

    alpha = ra % (2 * np.pi)
    sin_delta = np.sin(dec)

    return np.array([logA, f0_mHz, fdot, phi_init, cos_iota, psi_icrs, alpha, sin_delta]).T


def setup_state_for_injection(curr: CurrentInfoGlobalFit, state: GFState, source_type: str, branch_name: str, spread: float | np.ndarray  = 1e-5, subset_inds = None, priors: ProbDistContainer | None = None):
    """Initialize 'branch_name' walkers from catalogue injection parameters"""

    catalogue = getattr(curr.general_info, "catalogue", {})
    catalogue = catalogue.get(source_type, {})
    if catalogue:
        injection_params_list = []
        for source_id in sorted(catalogue.keys()):
            entry = catalogue[source_id]
            
            func_name = f"{branch_name}_catalogue_to_sampling_basis"
            conversion_func = globals().get(func_name)

            assert conversion_func and callable(conversion_func), f"catalogue_to_sampling_basis function for {branch_name} was not found."
            assert curr.general_info.preprocess_kwargs

            trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME # curr.general_info.data_processor.original_t0
            sampling_params = conversion_func(entry, trim_duration=trim_duration)

            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)
        
        ndim = state.branches_coords[branch_name].shape[-1]
        if injection_params.ndim == 3:
            injection_params = injection_params.reshape(-1, ndim)

        if subset_inds is not None:
            injection_params = injection_params[subset_inds, :]
        
        # Store injection truths for diagnostic plots
        try:
            setattr(curr.source_info[branch_name], "injection", injection_params)
        except AttributeError:
            logger.warning(f"No injection data is saved for {branch_name}.")
        
        scatter_around_injection(
            state, branch_name, injection_params, spread, betas=getattr(curr.source_info[branch_name], "betas"), priors=priors
        )


def select_gb_injection_subset_by_snr(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    gb_info,
    gb_wdm_comp,
    snr_threshold: float = 3.0,
    source_type: str = "GB",
    branch_name: str = "gb",
    f0_lims: typing.Optional[typing.Sequence[float]] = None,
) -> np.ndarray:
    """Select in-band GB catalogue sources by optimal SNR for true-point start.

    Enumerates ``curr.general_info.catalogue[source_type]`` in the SAME sorted
    order as :func:`setup_state_for_injection`, converts each to the sampling
    basis (via ``<branch_name>_catalogue_to_sampling_basis``), masks to the GB
    band (``gb_info.f0_lims``), and computes the optimal SNR
    ``sqrt(<h|h>)`` with the WDM likelihood object ``gb_wdm_comp``
    (``get_ll_wdm`` stashes ``h_h_out``; ``<h|h>`` needs only the PSD in
    ``acs``, so it is data-independent). Returns the array of catalogue-row
    indices whose optimal SNR exceeds ``snr_threshold`` — suitable for the
    ``subset_inds`` argument of :func:`setup_state_for_injection`.
    """
    catalogue = getattr(curr.general_info, "catalogue", {}) or {}
    catalogue = catalogue.get(source_type, {})
    if not catalogue:
        logger.warning(
            f"No '{source_type}' catalogue found; GB SNR-cut injection skipped."
        )
        return np.array([], dtype=int)

    keys = sorted(catalogue.keys())
    conversion_func = globals().get(f"{branch_name}_catalogue_to_sampling_basis")
    assert conversion_func is not None and callable(conversion_func), (
        f"catalogue_to_sampling_basis function for '{branch_name}' was not found."
    )

    trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME
    sampling = np.array(
        [conversion_func(catalogue[k], trim_duration=trim_duration) for k in keys]
    )
    ndim = int(np.asarray(sampling).shape[-1])
    # GB/VGB catalogue entries are array-valued over the whole galaxy, so the
    # per-entry conversion returns (N_src, ndim); flatten to (N_total, ndim)
    # exactly as setup_state_for_injection does so subset_inds line up.
    if sampling.ndim == 3:
        sampling = sampling.reshape(-1, ndim)

    # In-band on f0 (sampling basis stores f0 in mHz at index 1).
    f0_hz = np.asarray(sampling[:, 1], dtype=float) * 1e-3
    if f0_lims is None:
        f0_lims = gb_info.f0_lims
    f0_lo, f0_hi = float(f0_lims[0]), float(f0_lims[1])
    in_band = (f0_hz >= f0_lo) & (f0_hz <= f0_hi)
    n_in = int(in_band.sum())
    if n_in == 0:
        logger.info(
            f"GB SNR-cut injection: no '{source_type}' sources in band "
            f"[{f0_lo:.6e}, {f0_hi:.6e}] Hz; state stays at prior draws."
        )
        return np.array([], dtype=int)

    # Optimal SNR via the band likelihood (consistent with the fit's likelihood).
    # ``<h|h>`` needs only the (shared, fixed) PSD, so a single walker slab
    # suffices: pass walker-0's AnalysisContainer.  ``get_ll_wdm`` wraps a lone
    # AnalysisContainer into a 1-element ACA under the hood, so the data path is
    # identical to the multi-walker band-buffer case.
    #
    # Build the injection params on ``gb_wdm_comp``'s OWN backend, not the
    # module-level ``cp``: the comp is constructed with the run's
    # ``force_backend`` (often "cpu" even on a GPU node), so a hardcoded
    # ``cp.asarray`` would hand a cupy array to a numpy ``self.xp.atleast_2d``
    # inside ``get_ll_wdm`` and raise the implicit-conversion TypeError.
    #
    # ``gb_catalogue_to_sampling_basis`` returns FDOT-basis rows by contract
    # (slot 2 = fdot, negatives preserved for interacting DWDs), so the
    # physical rows must come from the fdot-basis transform -- the RUN's
    # transform would read slot 2 as Mc on chirp-mass runs (fdot ~1e-15 as
    # Mc -> template fdot ~ 0; fdot < 0 -> (neg)^(5/3) = NaN template).
    from .stock.erebor.transforms import make_gb_transform_container

    _tc_fdot = make_gb_transform_container(use_chirp_mass=False)
    _xp = gb_wdm_comp.xp
    params_phys = _tc_fdot.both_transforms(
        _xp.asarray(sampling[in_band]), xp=_xp
    )
    di = _xp.zeros(params_phys.shape[0], dtype=_xp.int32)
    if hasattr(gb_wdm_comp, "get_ll_wdm"):
        gb_wdm_comp.get_ll_wdm(params_phys, acs[0], data_index=di, noise_index=di)
    else:
        # STFT comp (STFTGBComputations): reads data / PSD through the parent
        # STFTComputationGroup bound to ``stft_comps`` -- walker-0 rows via
        # data_index=0; ``<h|h>`` is data-independent here too.
        gb_wdm_comp.get_ll_stft(params_phys, data_index=di, noise_index=di)

    h_h_np = asnumpy(gb_wdm_comp.h_h_out).real
    d_h_np = asnumpy(getattr(gb_wdm_comp, "d_h_out", np.zeros_like(h_h_np))).real
    params_np = asnumpy(params_phys)

    # Optional single-template information_matrix validation (GB_INFO_VALIDATE=1).
    # Since h scales linearly with amplitude A, dh/dA = h/A, so the Fisher
    # diagonal Gamma_AA satisfies Gamma_AA * A^2 == <h|h>. Cross-check the new
    # WDM information_matrix (inds=[0] -> only the amplitude derivative, cheap)
    # against the h_h just computed by get_ll_wdm.
    if bool(int(os.environ.get("GB_INFO_VALIDATE", "0"))) and params_np.shape[0]:
        try:
            _fish = asnumpy(gb_wdm_comp.information_matrix(
                params_phys, acs[0], inds=[0]))[:, 0, 0]
            _pred = _fish * params_np[:, 0] ** 2
            _rel = np.abs(_pred - h_h_np) / np.abs(np.where(h_h_np != 0, h_h_np, 1.0))
            _fin = np.isfinite(_rel)
            logger.warning(
                "GB info-matrix validation: Gamma_AA*A^2 vs <h|h> "
                "median reldiff=%.3e max=%.3e over %d sources (expect <~1e-3).",
                float(np.median(_rel[_fin])) if _fin.any() else float("nan"),
                float(np.nanmax(_rel[_fin])) if _fin.any() else float("nan"),
                int(_fin.sum()),
            )
        except Exception as e:  # noqa: BLE001 -- validation must never break setup
            logger.warning("GB info-matrix validation failed: %s: %s",
                           type(e).__name__, e)

    # Show the physical parameters EXACTLY as they enter the C kernel (post
    # both_transforms, which is what get_ll_wdm flattens into params_in). Dump
    # per-column min/max + values so units/magnitudes can be sanity-checked
    # (GB physical convention: amp, f0[Hz], fdot, fddot, phi0, iota, psi, lam, beta).
    _sampling_in = np.asarray(sampling[in_band])
    _cols = ["amp", "f0[Hz]", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta"]
    _lines = [
        f"    [{j}] {(_cols[j] if j < len(_cols) else 'col'+str(j)):8s}: "
        f"min={params_np[:, j].min():+.6e} max={params_np[:, j].max():+.6e} "
        f"vals={np.array2string(params_np[:, j], precision=6, max_line_width=240)}"
        for j in range(params_np.shape[1])
    ]
    logger.warning(
        "GB SNR-cut params ENTERING C kernel (post-transform), "
        f"{params_np.shape[0]} sources x {params_np.shape[1]} cols:\n"
        + "\n".join(_lines)
        + f"\n    sampling[in_band][0] (pre-transform, 8-col)={np.array2string(_sampling_in[0], precision=6)}"
        + f"\n    params_phys[0] (post-transform, {params_np.shape[1]}-col)={np.array2string(params_np[0], precision=6)}"
    )

    # Diagnostics: a NaN optimal SNR means either bad physical params fed to the
    # kernel or a non-finite <h|h> out of the kernel (e.g. inf invC from a zero
    # PSD cell, or a NaN WDM template). Report which so the fix lands at the
    # right layer.
    param_bad = ~np.isfinite(params_np).all(axis=1)
    hh_bad = ~np.isfinite(h_h_np)
    dh_bad = ~np.isfinite(d_h_np)
    if int(param_bad.sum()) or int(hh_bad.sum()):
        bad_cols = np.where(~np.isfinite(params_np).all(axis=0))[0].tolist()
        logger.warning(
            f"GB SNR-cut diagnostics: {int(param_bad.sum())}/{n_in} sources have "
            f"non-finite physical params (bad param columns={bad_cols}); "
            f"{int(hh_bad.sum())}/{n_in} non-finite <h|h>; "
            f"{int(dh_bad.sum())}/{n_in} non-finite <d|h>."
        )
        # Probe the exact arrays the kernel reads (invC = linear_psd_arr, data =
        # linear_data_arr on the under-the-hood 1-element ACA wrap) to localise
        # the NaN to the PSD/invC vs the template path.
        try:
            _holder = gb_wdm_comp._as_wdm_holder(acs[0])
            _psd = asnumpy(_holder.linear_psd_arr[0])
            _dat = asnumpy(_holder.linear_data_arr[0])
            logger.warning(
                f"GB SNR-cut invC probe: linear_psd_arr[nan={int(np.isnan(_psd).sum())}, "
                f"inf={int(np.isinf(_psd).sum())}, min={np.nanmin(_psd):.3e}, "
                f"max={np.nanmax(_psd):.3e}]; linear_data_arr[nan={int(np.isnan(_dat).sum())}, "
                f"inf={int(np.isinf(_dat).sum())}]."
            )
            # Generate the WDM template directly to isolate template-generation
            # from the inner product: if THIS is NaN, the on-the-fly WDM template
            # (orbit / chunk / t_ref path) is the culprit, not <.|.>.
            _templ = _holder.xp.zeros_like(_holder.linear_data_arr[0])
            gb_wdm_comp.fill_global_wdm(params_phys, _templ, data_index=di)
            _tn = asnumpy(_templ)
            _tabs = np.abs(_tn)
            logger.warning(
                f"GB SNR-cut template probe: fill_global_wdm out "
                f"[nan={int(np.isnan(_tn).sum())}, inf={int(np.isinf(_tn).sum())}, "
                f"nonzero={int((_tn != 0).sum())}/{_tn.size}, "
                f"absmax={(np.nanmax(_tabs) if np.isfinite(_tabs).any() else float('nan')):.3e}]."
            )
        except Exception as e:  # noqa: BLE001 -- diagnostics must never crash the run
            logger.warning(f"GB SNR-cut invC probe failed: {type(e).__name__}: {e}")

    opt_snr = np.sqrt(np.clip(h_h_np, 0.0, None))
    keep = np.isfinite(opt_snr) & (opt_snr > snr_threshold)
    subset_inds = np.where(in_band)[0][keep]
    finite_snr = opt_snr[np.isfinite(opt_snr)]
    logger.info(
        f"GB SNR-cut injection: {n_in} in-band, {int(keep.sum())} with optimal "
        f"SNR > {snr_threshold} (max finite SNR in band = "
        f"{float(finite_snr.max()) if finite_snr.size else 0.0:.2f}); injecting "
        f"{subset_inds.size} true-point leaves."
    )
    return subset_inds


def subtract_gb_neighbors_from_data(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    gb_info,
    gb_wdm_comp,
    *,
    exclude_f0_lims: typing.Sequence[float],
    window_hz: float,
    source_type: str = "GB",
    branch_name: str = "gb",
) -> int:
    """Subtract KNOWN neighbor-band GB templates from every walker's data.

    For focused single-band runs: catalogue sources whose ``f0`` falls
    OUTSIDE ``exclude_f0_lims`` (the sampled band) but within ``window_hz``
    of it are treated as known signals and subtracted from the residual of
    every cold-chain walker (``fill_global_wdm`` with ``factors = -1``), so
    their frequency spread does not bias the in-band fit. Returns the number
    of subtracted sources.

    Pair with ``select_gb_injection_subset_by_snr(..., f0_lims=
    exclude_f0_lims)`` so injected leaves and subtracted neighbors are
    disjoint (a source must never be modeled AND pre-subtracted).
    """
    catalogue = getattr(curr.general_info, "catalogue", {}) or {}
    catalogue = catalogue.get(source_type, {})
    if not catalogue:
        logger.warning(
            f"No '{source_type}' catalogue found; neighbor subtraction skipped."
        )
        return 0

    keys = sorted(catalogue.keys())
    conversion_func = globals().get(f"{branch_name}_catalogue_to_sampling_basis")
    trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME
    sampling = np.array(
        [conversion_func(catalogue[k], trim_duration=trim_duration) for k in keys]
    )
    if sampling.ndim == 3:
        sampling = sampling.reshape(-1, sampling.shape[-1])

    f0_hz = np.asarray(sampling[:, 1], dtype=float) * 1e-3
    lo, hi = float(exclude_f0_lims[0]), float(exclude_f0_lims[1])
    mask = (((f0_hz >= lo - window_hz) & (f0_hz < lo))
            | ((f0_hz > hi) & (f0_hz <= hi + window_hz)))
    n_sub = int(mask.sum())
    if n_sub == 0:
        logger.info("Neighbor subtraction: no catalogue sources in the "
                    f"window around [{lo:.6e}, {hi:.6e}] Hz.")
        return 0

    # FDOT-basis physical rows (same contract note as in
    # select_gb_injection_subset_by_snr: the run transform would misread
    # slot 2 as Mc on chirp-mass runs and NaN the fdot < 0 sources).
    from .stock.erebor.transforms import make_gb_transform_container

    _tc_fdot = make_gb_transform_container(use_chirp_mass=False)
    xp = gb_wdm_comp.xp
    params_phys = _tc_fdot.both_transforms(
        xp.asarray(sampling[mask]), xp=xp)
    nwalkers = int(curr.general_info.nwalkers)
    params_tiled = xp.tile(xp.asarray(params_phys), (nwalkers, 1))
    data_index = xp.repeat(
        xp.arange(nwalkers, dtype=xp.int32), n_sub).astype(xp.int32)
    factors = -xp.ones(params_tiled.shape[0], dtype=xp.float64)
    gb_wdm_comp.fill_global_wdm(
        params_tiled, acs.gather_linear_data_arr(),
        data_index=data_index, factors=factors,
    )
    logger.info(
        "Neighbor subtraction: subtracted %d known catalogue sources "
        "(window %.3e Hz around [%.6e, %.6e] Hz) from %d walkers.",
        n_sub, window_hz, lo, hi, nwalkers,
    )
    return n_sub


def subtract_initial_signal(
    acs: AnalysisContainerArray,
    state: GFState,
    wave_gen: typing.Callable,
    source_name: str,
    source_info: Setup,
):
    """Subtract pre-injected source templates from the residual buffers in ``acs``.

    Used at run start when a recipe seeds branches with known signal
    parameters (e.g. catalog injections); the corresponding template is
    removed from the residual so the sampler sees only the noise + other
    sources.

    Args:
        acs: Shared :class:`AnalysisContainerArray` whose residuals are
            modified in place.
        state: Current sampler state.
        wave_gen: Waveform generator for ``source_name``.
        source_name: Branch name (e.g. ``"mbh"``, ``"emri"``).
        source_info: Per-source :class:`Setup` providing transforms /
            waveform kwargs.
    """
    xp = acs.xp
    if np.any(inds := state.branches_inds[source_name][0]):
        logger.info(f"Subtracting initial signals for {source_name}")
        counter = 0
        for leaf in range(inds.shape[-1]):
            if inds[0, leaf]:
                assert np.all(inds[:, leaf])
                inj_coords = state.branches_coords[source_name][0, :, leaf]
                inj_coords_in = xp.asarray(source_info.transform.both_transforms(inj_coords))

                # logger.debug(f"CUDA device here: {cp.cuda.runtime.getDevice()}")  # Debugging line to check current CUDA device

                # C-order columns are non-contiguous (stride = ndim*8), so ascontiguousarray
                # is forced to allocate a fresh, pool-aligned buffer for each parameter —
                # avoiding the misalignment that arises with F-order when nwalkers is odd.
                signals_in = wave_gen(*[xp.ascontiguousarray(col) for col in inj_coords_in.T], **source_info.waveform_kwargs)
                for w in range(len(signals_in)):
                    ll_here = acs.acs[w].template_likelihood(template=signals_in[w], include_psd_info=False)
                    logger.debug(f"Initial log-likelihood contribution from walker {w}, leaf {leaf}: {ll_here}")
                acs.add_signal_to_residual(signals_in)
                counter += 1

                # Release the per-leaf template allocation before the next
                # leaf's wave_gen call. Without it the pool grows by one
                # full (nwalkers, nchannels, basis) template per injected
                # leaf, which OOMs a many-leaf seed on a single device.
                acs.free_gpu_memory()

                # if acs.gpus is not None:
                #     acs.synchronize()  # Ensure GPU computations are complete before logging
                #     # acs.xp.get_default_memory_pool().free_all_blocks()
                #     cp.cuda.runtime.setDevice(main_device)  # Switch back to main device after subtraction
                #     logger.debug(f"Switched back to main CUDA device {main_device} after subtraction.")
                    
        logger.debug(f"Subtracted {counter} initial signals for {source_name}")
    else:
        logger.info(f"No initial signals for {source_name}")

    #breakpoint()

def build_psd_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    *,
    num_repeats: int = 60,
    permute_every: int = 50,
    Tmax: float = 1e6,
) -> tuple[PSDMove, PSDMove]:
    """Build PSD search and PE moves.

    Both moves share the same ``acs``, ``priors``, and
    ``TemperatureControl`` instance, so updates to ``acs`` (e.g. signal
    subtraction by another branch) are visible to both moves at runtime.

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["psd"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    num_repeats : int, optional
        Number of internal PSD move repeats. Default 60.
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.

    Returns
    -------
    psd_search_move : PSDMove
    psd_pe_move : PSDMove
    """
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    psd_info = curr.source_info["psd"]
    galfor_info = curr.source_info.get("galfor", None)

    effective_ndim = engine_info.ndims["psd"] if galfor_info is None else engine_info.ndims["galfor"] + engine_info.ndims["psd"] 
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=num_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        galfor_transform_fn=galfor_info.transform if galfor_info is not None else None,
        sensitivity_backend=general_info.sensitivity_backend,
        temperature_control=temperature_control,
        use_gpu=True,
    )

    psd_search_move = PSDMove(
        acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = PSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    return psd_search_move, psd_pe_move


def build_mbh_moves_phenom(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    permute_every: int = 20,
    wave_gen: typing.Callable = None,
    subtract_initial: bool = True,
    ) -> tuple[typing.Callable, ResidualAddOneRemoveOneMove]:
    """Build MBH PE move using ``PhenomTHMTDIWaveform`` + ``ResidualAddOneRemoveOneMove``.

    Sets ``state.sub_states['mbh'].betas_all`` as a side effect.

    Parameters
    ----------
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["mbh"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    state :
        Current sampler state; ``sub_states["mbh"].betas_all`` is set here.
    wave_gen :
        Optional pre-built ``PhenomTHMTDIWaveform`` instance. When the
        settings file already constructed (and cached) the generator —
        e.g. to register ``source_info['mbh'].signal_gen`` for the
        engine-side residual rebuild — pass it here so the move shares
        the same instance. Default ``None`` builds a fresh one from
        ``mbh_info.initialize_kwargs``.
    subtract_initial : bool
        If ``True`` (default), subtract the state's current MBH templates
        from the residuals here (legacy recipe-side path). Settings files
        that register ``source_info['mbh'].signal_gen`` must pass
        ``False`` — the engine already subtracts during
        ``setup_acs(rebuild_residuals=True)`` and doing it twice corrupts
        the residuals.

    Returns
    -------
    wave_gen : PhenomTHMTDIWaveform
    mbh_pe_move : ResidualAddOneRemoveOneMove
    """
    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    mbh_info = curr.source_info["mbh"]

    if wave_gen is None:
        wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
    # Legacy pre-``signal_gen`` residual subtraction. When the settings file
    # registers ``source_info['mbh'].signal_gen`` the engine's
    # ``setup_acs(rebuild_residuals=True)`` already subtracted — pass
    # ``subtract_initial=False`` there (doing it twice corrupts the residuals).
    if subtract_initial:
        subtract_initial_signal(acs, state, wave_gen.get_signals_for_residuals, "mbh", mbh_info)

    # The move construction (make_ladder -> betas_all -> coords_shape ->
    # ResidualAddOneRemoveOneMove, plus the ``mbh_info.betas`` /
    # ``state.sub_states['mbh'].betas_all`` side effects) is the shared
    # single-source machinery in :class:`MBHMoveBuilder`.
    _, mbh_pe_moves = MBHMoveBuilder(
        wave_gen=wave_gen.get_signals_for_residuals, permute_every=permute_every
    ).build(None, curr, acs, priors, state)

    return wave_gen, mbh_pe_moves[0]


@dataclass
class GBWaveformDict(typing.TypedDict):
    dt: float
    T: float
    use_c_implementation: bool
    start_freq_ind: int
    tdi_channel_setup: str
    tdi2: bool
    window: None | str
    window_alpha: float


def build_gb_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    *,
    Tmax: float = 1e6,
    search_move_names: typing.Optional[typing.Sequence[str]] = None,
    pe_move_names: typing.Optional[typing.Sequence[str]] = None,
    include_search: typing.Optional[bool] = None,
    include_refit: bool = True,
    **kwargs,
) -> typing.Tuple[typing.List[GBSpecialBase], typing.List[GBSpecialBase]]:
    """Build GB search and PE moves.

    Both moves share the same ``acs``, ``priors``, and
    ``TemperatureControl`` instance, so updates to ``acs`` (e.g. signal
    subtraction by another branch) are visible to both moves at runtime.

    The GB move classes, per-move flags, and list ordering are the GB
    *reference recipe* and stay here; a settings file steers which of them it
    actually installs through the keyword-only design knobs below (this
    replaces the post-hoc ``[m for m in gb_pe_moves if "prior" in m.name]``
    filtering that GB settings files used to duplicate).

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["gb"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.
    search_move_names : sequence of str, optional
        Explicit names of search moves to build (e.g. ``["rj_prior_search"]``).
        If ``None`` and ``include_search`` is not ``False``, defaults to
        ``["rj_fstat_mcmc_search", "rj_prior_search"]`` for FD, or ``[]`` for STFT/WDM.
    pe_move_names : sequence of str, optional
        Explicit names of PE moves to build (e.g. ``["rj_prior"]``). If ``None``,
        defaults to ``["rj_prior", "rj_fstat_mcmc"]`` (plus refit if available) for FD,
        or ``["rj_prior"]`` for STFT/WDM.
    include_search : bool, optional
        Deprecated in favor of ``search_move_names``. When ``False``, return an empty
        search-move list. Default ``None``.
    include_refit : bool, optional
        When ``False``, never build the GMM-refit moves even if the refit
        file is present. Default ``True`` (refit is still gated on the file
        existing).

    Returns
    -------
    gb_search_moves : List[GBSpecialBase]
    gb_pe_moves : List[GBSpecialBase]
    """
    gb_info: GBSetup = curr.source_info["gb"]
    general_info: GeneralSetup = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    
    gb_betas = gb_info.betas
    gpus: list[int] = general_info.gpus

    domain_settings = general_info.domain_settings

    #* Setting up gbgpu on the correct backend and (if any) gpu(s).
    #* CPU path keeps numpy and avoids the cupy-only setDevice call.
    from gbgpu.gbgpu import GBGPU
    import gbgpu

    gb_force_backend = general_info.force_backend
    _gb_backend = gbgpu.get_backend(gb_force_backend)
    if gpus is not None:
        _gb_backend.set_cuda_device(gpus[0])
        
    gb = GBGPU(**gb_info.initialize_kwargs)
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])
        gb.gpus = gpus
    else:
        gb.gpus = None

    logger.debug(f"GBGPU initialized at t0 = {gb_info.initialize_kwargs['t0']}")
    logger.debug(f"GBGPU initialized with gpus: {gb.gpus} and backend: {gb.backend}")
    
    #* Ensure gb computations frames are initialized. (currently STFT only)
    if isinstance(gb_info.gb_stft_comp, type) and issubclass(gb_info.gb_stft_comp, STFTGBComputations):
        assert gb_info.gb_stft_kwargs is not None
        
        if acs.domain_group_kwargs is None or acs.domain_group_kwargs == dict():
            domain_kwargs = dict(
                tdi_type="XYZ",
                window_alpha=general_info.window_alpha,
                use_midpoint=gb_info.gb_stft_kwargs["use_midpoint"],
                linear_envelope=gb_info.gb_stft_kwargs["linear_envelope"]
            )
            acs.domain_group_kwargs = domain_kwargs
            acs._ensure_cpp_splits()
        
        tdi_config = TDIConfig(
            tdi='2nd generation', 
            force_backend=general_info.force_backend
        )

        gb_stft_comp = STFTGBComputations(
            stft_comps=acs.cpp_splits[0],
            T=gb_info.Tobs,
            t_ref=MOJITO_REFERENCE_TIME,
            orbits=general_info.gpu_orbits,
            tdi_config=tdi_config,
            force_backend=general_info.force_backend,
            n_side_bins=gb_info.gb_stft_kwargs["n_side_bins"],
            freq_from_tdi_phase=gb_info.gb_stft_kwargs["freq_from_tdi_phase"],
            window_alpha=general_info.window_alpha,
            use_midpoint=gb_info.gb_stft_kwargs["use_midpoint"],
            linear_envelope=gb_info.gb_stft_kwargs["linear_envelope"],
            # * Optional keys: the response and the DFFR settings; without them the comp's defaults hold.
            **{key: gb_info.gb_stft_kwargs[key]
               for key in ("response", "first_order_delays", "chirp_rate_floor")
               if key in gb_info.gb_stft_kwargs},
        )
        gb_info.gb_stft_comp = gb_stft_comp

    #* STFT store windows are sized here, where the comp's stencil, its t_ref and data_t0 are all known.
    gb_info.stft_store_windows = None
    band_structure = getattr(gb_info, "band_structure", None)
    if isinstance(acs.settings, STFTSettings) and band_structure is not None:
        if (float(band_structure.grid_df_hz) != float(acs.settings.df)
                or not np.isclose(float(asnumpy(band_structure.edges_hz)[0]), float(acs.settings.min_freq))):
            raise ValueError(
                "the GB band structure was built on a different STFT grid than the analysis container; "
                "its store windows would index the wrong bins."
            )
        gb_info.stft_store_windows = stft_store_window_layout(
            band_structure,
            acs.settings,
            n_side_bins=gb_info.gb_stft_comp.n_side_bins,
            epoch_offset_s=gb_info.gb_stft_comp.t_ref - curr.general_info.data_t0,
        )
        logger.info(
            "STFT store windows: tier widths %s bins, buffer width %d of NF_active=%d",
            gb_info.stft_store_windows.tier_width,
            gb_info.stft_store_windows.buffer_width,
            int(acs.settings.NF_active),
        )

    #* Make sure that priors are evaluated on gpus (when available).
    use_gpu_priors = gpus is not None
    gpu_priors = {}
    for key, prior_container in priors.items():
        if "gb" in key:
            gpu_priors_in = deepcopy(prior_container.priors_in)
            
            # Update cupy status for each prior object
            for item in gpu_priors_in.values():
                item.use_cupy = use_gpu_priors
            
            gpu_prior = ProbDistContainer(gpu_priors_in, use_cupy=use_gpu_priors)
            
            # Safely preserve CPU key order if defined and different
            cpu_key_order = list(getattr(prior_container, "key_order", []))
            if cpu_key_order and list(gpu_prior.key_order) != cpu_key_order:
                gpu_prior.reset_key_order(cpu_key_order)
                
            gpu_priors[key] = gpu_prior
            
    
    nleaves_max_gb = state.branches["gb"].shape[-2]
    
    #* Get band information
    band_edges = gb_info.band_edges
    band_N_vals = gb_info.band_N_vals

    #* This checks if the initialization has any gbs in it (when injecting gbs) and adjusts acs accordingly.
    #* Skipped when a GB ``signal_gen`` is registered — the engine's
    #* setup_acs(rebuild_residuals=True) already subtracted the state's GB
    #* templates. No GB signal_gen exists today, so this stays active.
    if getattr(gb_info, "signal_gen", None) is None and state.branches["gb"].inds[0].sum() > 0:

        coords_out_gb = state.branches["gb"].coords[0,
            state.branches["gb"].inds[0]
        ]
        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)

        check = priors["gb"].logpdf(coords_out_gb)
        if np.any(np.isinf(check)):

            # check which prior is inf
            inf_indices = np.where(np.isinf(check))[0]
            inf_coords = coords_out_gb[inf_indices]
            logger.error(f"Found {len(inf_indices)} coordinates with inf logpdf under GB priors. Example inf coordinates: {inf_coords[:5]}") 

            logger.info("Prior bounds for GB parameters:")
            for param_name, prior in priors["gb"].priors_in.items():
                logger.info(f"  {param_name}: [{prior.min_val},{prior.max_val}]")
            breakpoint()
            raise ValueError("Starting priors are inf. If injecting, try reducing spread.")

        coords_in_in = gb_info.transform.both_transforms(coords_out_gb)

        walker_vals = np.tile(
            np.arange(nwalkers), (nleaves_max_gb, 1)
        ).transpose((1, 0))[state.branches["gb"].inds[0]]

        data_index_1 = walker_vals  # ((band_inds % 2) + 0) * nwalkers + walker_vals

        # Build index/factor arrays on the ACA's backend
        _xp = acs.xp
        data_index = _xp.asarray(data_index_1).astype(_xp.int32)
        factors = -_xp.ones_like(data_index, dtype=_xp.float64) # -1 for subtraction


        logger.debug("Generating global GB template")
        if gpus is not None:
            gb.gpus = gpus

        template_in = deepcopy(acs.linear_data_arr)
        if isinstance(domain_settings, FDSettings):
            #* TODO: add test to make sure the generator matches the general information.
            band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1
            N_vals = band_N_vals[band_inds]
            
            num_per_gpu_walker = (
                len(acs.gpu_splits[0]) if (acs.gpus is not None and len(acs.gpus) > 1) else None
            )
            gb.generate_global_template(
                coords_in_in,
                data_index,
                acs.linear_data_arr,
                data_length=acs.data_length,
                factors=factors,
                data_splits=acs.gpu_map,
                num_per_gpu=num_per_gpu_walker,
                N=N_vals,
                **gb_info.waveform_kwargs,
            )
        elif isinstance(domain_settings, WDMSettings):
            if gb_info.gb_wdm_comp is None:
                raise ValueError(
                    "WDM-domain GB initialization requires "
                    "gb_info.gb_wdm_comp; build a GBWDMComputations in the "
                    "settings file and pass it via GBSettings.gb_wdm_comp."
                )
            num_bin = coords_in_in.shape[0]
            xp = gb_info.gb_wdm_comp.xp
            factors_arr = xp.asarray(factors).astype(xp.float64)

            gb_info.gb_wdm_comp.fill_global_wdm(
                coords_in_in,
                acs.gather_linear_data_arr(),
                data_index=xp.asarray(data_index),
                factors=factors_arr,
            )
        elif isinstance(domain_settings, STFTSettings):
            assert gb_info.gb_stft_comp is not None, (
                "STFT-domain GB initialization requires "
                "gb_info.gb_stft_comp; build an STFTGBComputations in the "
                "settings file and pass it via GBSettings.gb_stft_comp."
            )
            
            xp = gb_info.gb_stft_comp.xp

            make_band_likelihood_engine(
                domain_settings,
                gb_stft_comp=gb_info.gb_stft_comp,
                nchannels=acs.nchannels,
                tdi_channel_setup=gb_info.waveform_kwargs["tdi_channel_setup"],
            ).fill_template(
                acs,
                xp.asarray(coords_in_in),
                xp.asarray(data_index).astype(xp.int32),
                None,
                factor=-1,
                waveform_kwargs=gb_info.waveform_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Domain settings {type(domain_settings).__name__} are not "
                f"supported for GB initialization."
            )

        max_diff_templates = 0.0
        for shard, (before, after) in enumerate(zip(template_in, acs.linear_data_arr)):
            # Both operands live on this shard's device; subtracting them anywhere else is
            # a cross-device op.
            with acs.device_context(None if gpus is None else int(gpus[shard])):
                max_diff_templates = max(
                    max_diff_templates, float(_xp.abs(before - after).max())
                )
        del template_in
        logger.debug(
            f"Global GB template generated with max template in/out diff = "
            f"{max_diff_templates:5e}"
        )
        if max_diff_templates == 0.0:
            raise RuntimeError(
                "The initial GB subtraction left the residual unchanged on every shard. "
                "The template was written somewhere other than acs.linear_data_arr."
            )

    _post_sub = acs[0].data_res_arr.data_res_arr
    if hasattr(_post_sub, "plot"):
        _post_sub.plot(channel=0, filename=curr.general_info.artifacts_file_dir + "data_post_subtraction.png")

    #* Check if we need to adjust the band temps, and adjust if required
    adjust_temps = False
    state_band_info = getattr(state, "band_info", None)
    if state_band_info is not None:
        band_info_check = deepcopy(state_band_info)
        adjust_temps = True
        #    del state.band_info

    band_temps = np.tile(np.asarray(gb_betas), (len(band_edges) - 1, 1))
    state.sub_states["gb"].initialize_band_information(nwalkers, ntemps, band_edges, band_temps)
    state.sub_states["gb"].band_info["band_temps"][:] = band_temps
    if adjust_temps:
        state.sub_states["gb"].band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

    # TODO Check if the block below is needed... I.e., do we need band_inds in brach supplemental?
    # band_inds_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)
    # N_vals_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)

    # if state.branches["gb"].inds.sum() > 0:
    #     f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
    #     band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
    #     N_vals_in[state.branches["gb"].inds] = band_N_vals[band_inds_in[state.branches["gb"].inds]]

    # branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
    # state.branches["gb"].branch_supplemental = BranchSupplemental(
    #     {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    # )

    #* Assembling args and kwargs
    #* ``fd`` is no longer a positional — the move derives it from
    #* ``acs.settings`` so the same call works for FDSettings and
    #* WDMSettings (and any future domain).
    start_freq_ind_val = (
        int(acs.start_freq_ind[0])
        if (acs.start_freq_ind is not None and acs.start_freq_ind[0] is not None)
        else None
    )
    gb_move_args = (
        gb,
        priors,
        start_freq_ind_val,
        acs.end_shape[0],
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
    )

    effective_ndim = engine_info.ndims["gb"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )
    
    # * GB move options come from the settings file through gb_info; None gives each class's defaults.
    debug_settings = getattr(gb_info, "debug_settings", None)
    if debug_settings is not None and debug_settings.plot_dir is None:
        debug_settings = dataclasses.replace(
            debug_settings,
            plot_dir=os.path.join(curr.general_info.artifacts_file_dir, "gb_debug", ""),
        )
    leaf_cap_settings = getattr(gb_info, "leaf_cap_settings", None)
    sighet_settings = getattr(gb_info, "sighet_settings", None)

    gb_move_kwargs = dict(
        waveform_kwargs=gb_info.waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.force_backend,
        nfriends=nwalkers,
        temperature_control=temperature_control,
        num_repeat_proposals=gb_info.num_repeat_proposals,
        search_kwargs=gb_info.search_kwargs,
        stretch_probability=gb_info.stretch_probability,
        gb_wdm_comp=getattr(gb_info, "gb_wdm_comp", None),
        gb_fd_comp=getattr(gb_info, "gb_fd_comp", None),
        gb_stft_comp=getattr(gb_info, "gb_stft_comp", None),
        orbits=getattr(gb_info, "orbits", None),
        tdi_config=getattr(gb_info, "tdi_config", None),
        t_ref=float(getattr(gb_info, "t0", 0.0) or 0.0),
        debug_settings=debug_settings,
        leaf_cap_settings=leaf_cap_settings,
        sighet_settings=sighet_settings,
        num_band_preload=getattr(gb_info, "num_band_preload", None),
        num_bands_preload_temp=getattr(gb_info, "num_bands_preload_temp", None),
        ntemps=ntemps,
        stft_store_windows=gb_info.stft_store_windows,
    )

    _rj_phase_max = bool(int(os.environ.get("GB_RJ_PHASE_MAXIMIZE", "0")))

    # Check aliases in kwargs if any
    if "pe_search_names" in kwargs and pe_move_names is None:
        pe_move_names = kwargs.pop("pe_search_names")
    if "search_moves_names" in kwargs and search_move_names is None:
        search_move_names = kwargs.pop("search_moves_names")

    # Resolve search_move_names
    if search_move_names is None:
        if include_search is False:
            search_move_names = []
        elif include_search is True:
            if isinstance(domain_settings, FDSettings):
                search_move_names = ["rj_fstat_mcmc_search", "rj_prior_search"]
            else:
                search_move_names = ["rj_prior_search"]
        else:
            # search_move_names is None and include_search is None
            if isinstance(domain_settings, FDSettings):
                search_move_names = ["rj_fstat_mcmc_search", "rj_prior_search"]
            else:
                search_move_names = []
    else:
        search_move_names = list(search_move_names)

    # Refit file check
    _refit_fp = getattr(general_info, "main_file_path", None)
    _refit_available = include_refit and isinstance(_refit_fp, str) and os.path.exists(_refit_fp)

    # Resolve pe_move_names
    if pe_move_names is None:
        if isinstance(domain_settings, FDSettings):
            pe_move_names = (
                ["rj_prior", "rj_refit", "rj_fstat_mcmc"]
                if _refit_available
                else ["rj_prior", "rj_fstat_mcmc"]
            )
        else:
            pe_move_names = ["rj_prior"]
    else:
        pe_move_names = list(pe_move_names)

    # Validate domain compatibility
    fd_only_moves = {"rj_fstat_mcmc_search", "rj_fstat_mcmc", "rj_refit_search", "rj_refit"}
    if not isinstance(domain_settings, FDSettings):
        for name in search_move_names + pe_move_names:
            if name in fd_only_moves:
                raise NotImplementedError(
                    f"GB move '{name}' requires FD basis, but domain is {type(domain_settings).__name__}."
                )

    # Validate refit availability if explicitly requested
    for name in search_move_names + pe_move_names:
        if name in ("rj_refit", "rj_refit_search") and not _refit_available:
            raise FileNotFoundError(
                f"GB refit move '{name}' was requested, but refit file '{_refit_fp}' "
                f"does not exist or include_refit is False."
            )

    def _create_move(name: str) -> GBSpecialBase:
        if name == "rj_prior_search":
            move = GBSpecialRJPriorMove(
                *gb_move_args,
                rj_proposal_distribution=gpu_priors,
                name="rj_prior_search",
                use_prior_removal=True,
                phase_maximize=_rj_phase_max,
                ranks_needed=0,
                run_swaps=True,
                **gb_move_kwargs,
            )
        elif name == "rj_fstat_mcmc_search":
            move = GBSpecialRJSerialSearchMCMC(
                *gb_move_args,
                rj_proposal_distribution=None,
                is_rj_prop=True,
                run_swaps=False,
                name="rj_fstat_mcmc_search",
                phase_maximize=True,
                ranks_needed=0,
                **{**gb_move_kwargs, "leaf_cap_update": False},
            )
        elif name == "rj_refit_search":
            move = GBSpecialRJRefitMove(
                *gb_move_args,
                rj_proposal_distribution=None,
                is_rj_prop=True,
                run_swaps=False,
                name="rj_refit_search",
                fp=_refit_fp,
                phase_maximize=True,
                ranks_needed=0,
                **{**gb_move_kwargs, "leaf_cap_update": False},
            )
        elif name == "rj_prior":
            move = GBSpecialRJPriorMove(
                *gb_move_args,
                rj_proposal_distribution=gpu_priors,
                name="rj_prior",
                use_prior_removal=False,
                phase_maximize=_rj_phase_max,
                ranks_needed=0,
                run_swaps=True,
                **gb_move_kwargs,
            )
        elif name == "rj_fstat_mcmc":
            move = GBSpecialRJSerialSearchMCMC(
                *gb_move_args,
                rj_proposal_distribution=None,
                run_swaps=True,
                name="rj_fstat_mcmc",
                phase_maximize=False,
                ranks_needed=0,
                **{**gb_move_kwargs, "leaf_cap_update": False},
            )
        elif name == "rj_refit":
            move = GBSpecialRJRefitMove(
                *gb_move_args,
                rj_proposal_distribution=None,
                run_swaps=True,
                name="rj_refit",
                fp=_refit_fp,
                phase_maximize=False,
                ranks_needed=0,
                **{**gb_move_kwargs, "leaf_cap_update": False},
            )
        else:
            raise ValueError(f"Unknown GB move name '{name}'.")

        move.accepted = np.zeros((ntemps, nwalkers))
        return move

    gb_search_moves = [_create_move(name) for name in search_move_names]
    gb_pe_moves = [_create_move(name) for name in pe_move_names]

    return gb_search_moves, gb_pe_moves

# ======================================================================
# Source move-builder hierarchy
#
# ``SourceMoveBuilder`` is the installable base class; per-source subclasses
# carry the design knobs. Settings files construct and ``build()`` these inside
# ``setup_recipe`` instead of hand-rolling the move construction. GB / PSD keep
# their richer function form (``build_gb_moves`` / ``build_psd_moves``) and the
# builder classes wrap them for a uniform ``(search_moves, pe_moves)`` return.
# ======================================================================


class SourceMoveBuilder:
    """Base class for building the recipe move(s) of one source branch.

    A builder is a light factory: construction carries the per-recipe *design
    knobs* (which move variants, ordering, thresholds), while :meth:`build`
    consumes the runtime context (``curr`` / ``acs`` / ``priors`` / ``state``)
    and returns ``(search_moves, pe_moves)`` — either list may be empty.
    Settings files construct and call a builder inside ``setup_recipe``; the
    repeatable machinery lives here in :mod:`recipe`.
    """

    #: Branch this builder targets; set on the subclass or via the constructor.
    branch_name: typing.Optional[str] = None

    def __init__(self, *, branch_name: typing.Optional[str] = None):
        if branch_name is not None:
            self.branch_name = branch_name
        assert self.branch_name is not None, "SourceMoveBuilder needs a branch_name"

    def build(self, engine_info, curr, acs, priors, state):
        """Return ``(search_moves, pe_moves)`` for this branch."""
        raise NotImplementedError


class SingleSourcePEBuilder(SourceMoveBuilder):
    """Build a :class:`ResidualAddOneRemoveOneMove` PE move for one branch.

    MBH / EMRI / SOBBH share the identical ``make_ladder`` -> ``betas_all`` ->
    ``coords_shape`` -> ``ResidualAddOneRemoveOneMove`` construction; this is
    that shared core. Subclasses set :attr:`branch_name` (and, where the source
    differs, :attr:`like_kwargs_from_waveform_kwargs`). Any constructor argument
    left ``None`` falls back to the matching field on
    ``curr.source_info[branch_name]``.
    """

    #: EMRI passes ``waveform_kwargs`` as the likelihood kwargs; MBH/SOBBH pass
    #: an empty dict. Encoded as a class flag so the per-source default lives on
    #: the subclass while the construction stays shared.
    like_kwargs_from_waveform_kwargs: bool = False

    def __init__(
        self,
        *,
        branch_name: typing.Optional[str] = None,
        wave_gen: typing.Callable,
        waveform_gen_kwargs: typing.Optional[dict] = None,
        waveform_like_kwargs: typing.Optional[dict] = None,
        num_repeats: typing.Optional[int] = None,
        inner_moves: typing.Optional[list] = None,
        transform=None,
        betas: typing.Optional[np.ndarray] = None,
        Tmax: float = np.inf,
        permute_every: int = 20,
        move_name: typing.Optional[str] = None,
        **move_kwargs,
    ):
        super().__init__(branch_name=branch_name)
        self.wave_gen = wave_gen
        self.waveform_gen_kwargs = waveform_gen_kwargs
        self.waveform_like_kwargs = waveform_like_kwargs
        self.num_repeats = num_repeats
        self.inner_moves = inner_moves
        self.transform = transform
        self.betas = betas
        self.Tmax = Tmax
        self.permute_every = permute_every
        self.move_name = move_name
        self.move_kwargs = move_kwargs

    def build(self, engine_info, curr, acs, priors, state):
        info = curr.source_info[self.branch_name]
        gi = curr.general_info
        ntemps, nwalkers = gi.ntemps, gi.nwalkers

        betas = (
            self.betas if self.betas is not None else make_ladder(info.ndim, ntemps=ntemps)
        )
        # Side effects (parity with the old build_mbh_moves_phenom): stash the
        # ladder on the source info and the tiled ladder on the sub-state.
        info.betas = betas
        betas_all = np.tile(betas, (info.nleaves_max, 1))
        state.sub_states[self.branch_name].betas_all = betas_all
        logger.debug(f"{self.branch_name} betas: {betas}")

        coords_shape = (ntemps, nwalkers, info.nleaves_max, info.ndim)

        wf_gen_kw = (
            self.waveform_gen_kwargs
            if self.waveform_gen_kwargs is not None
            else info.waveform_kwargs
        ).copy()
        if self.waveform_like_kwargs is not None:
            wf_like_kw = self.waveform_like_kwargs.copy()
        elif self.like_kwargs_from_waveform_kwargs:
            wf_like_kw = info.waveform_kwargs.copy()
        else:
            wf_like_kw = dict()

        move = ResidualAddOneRemoveOneMove(
            self.branch_name,
            coords_shape,
            self.wave_gen,
            wf_gen_kw,
            wf_like_kw,
            acs,
            self.num_repeats if self.num_repeats is not None else info.num_prop_repeats,
            self.transform if self.transform is not None else info.transform,
            priors,
            self.inner_moves if self.inner_moves is not None else info.inner_moves,
            Tmax=self.Tmax,
            betas_all=betas_all,
            permute_every=self.permute_every,
            name=self.move_name,
            **self.move_kwargs,
        )
        move.accepted = np.zeros((ntemps, nwalkers))
        return [], [move]


class MBHMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"mbh"`` branch."""

    branch_name = "mbh"


class EMRIMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"emri"`` branch."""

    branch_name = "emri"
    like_kwargs_from_waveform_kwargs = True


class SOBBHMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"sobbh"`` branch.

    Like EMRI, SOBBH passes ``waveform_kwargs`` as the likelihood kwargs (this
    matches the EMRI/SOBBH inline builders it replaces).
    """

    branch_name = "sobbh"
    like_kwargs_from_waveform_kwargs = True


class GBMoveBuilder(SourceMoveBuilder):
    """Build the GB search + PE move lists.

    Wraps :func:`build_gb_moves` (which owns the GB machinery + the GB reference
    move recipe) in the :class:`SourceMoveBuilder` interface. Design knobs are
    forwarded so a settings file can steer which moves it installs.
    """

    branch_name = "gb"

    def __init__(
        self,
        *,
        Tmax: float = 1e6,
        search_move_names: typing.Optional[typing.Sequence[str]] = None,
        pe_move_names: typing.Optional[typing.Sequence[str]] = None,
        include_search: typing.Optional[bool] = None,
        include_refit: bool = True,
        **kwargs,
    ):
        super().__init__(branch_name="gb")
        self.Tmax = Tmax
        self.search_move_names = search_move_names
        self.pe_move_names = pe_move_names
        self.include_search = include_search
        self.include_refit = include_refit
        self.extra_kwargs = kwargs

    def build(self, engine_info, curr, acs, priors, state):
        return build_gb_moves(
            engine_info,
            curr,
            acs,
            priors,
            state,
            Tmax=self.Tmax,
            search_move_names=self.search_move_names,
            pe_move_names=self.pe_move_names,
            include_search=self.include_search,
            include_refit=self.include_refit,
            **self.extra_kwargs,
        )


class PSDMoveBuilder(SourceMoveBuilder):
    """Build the PSD search + PE moves (wraps :func:`build_psd_moves`)."""

    branch_name = "psd"

    def __init__(self, *, num_repeats: int = 60, permute_every: int = 50, Tmax: float = 1e6):
        super().__init__(branch_name="psd")
        self.num_repeats = num_repeats
        self.permute_every = permute_every
        self.Tmax = Tmax

    def build(self, engine_info, curr, acs, priors, state):
        search_move, pe_move = build_psd_moves(
            engine_info,
            curr,
            acs,
            priors,
            num_repeats=self.num_repeats,
            permute_every=self.permute_every,
            Tmax=self.Tmax,
        )
        return [search_move], [pe_move]
