"""Helper utilities for LISA sampling: GB grouping, state restoration, updates."""

import os
from multiprocessing.sharedctypes import Value

import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    pass

from eryn.state import BranchSupplemental, State
from eryn.utils.transform import TransformContainer
from eryn.utils.utility import groups_from_inds

from ..utils.utility import asnumpy


# DetermineGBGroups / GetLastGBState were removed with the legacy
# MultiGPUDataHolder they consumed (parallel-resources plan P4); the
# stock path restores GB state through the model AnalysisContainerArray.


class HeterodynedUpdate:
    """Periodic update that re-centers the heterodyne reference for MBH likelihoods.

    Designed to be passed to ``eryn`` as an ``update_fn``. On each call it
    finds the highest-likelihood walker in the current state, calls
    ``init_heterodyne_info`` on the underlying MBH template model with that
    point as the new reference, optionally zeros the model's ``d_d`` term,
    and recomputes log-prior, log-likelihood, and blobs for the existing
    samples so they are consistent with the new heterodyne expansion.

    Args:
        update_kwargs: Keyword arguments forwarded to
            ``template_model.init_heterodyne_info``.
        set_d_d_zero: If ``True``, set ``template_model.reference_d_d = 0``
            after the update.
    """

    def __init__(self, update_kwargs, set_d_d_zero=False):
        self.update_kwargs = update_kwargs
        self.set_d_d_zero = set_d_d_zero

    def __call__(self, it, sample_state, sampler, **kwargs):
        """Re-center the heterodyne and refresh likelihoods on the current state."""

        samples = sample_state.branches_coords["mbh"].reshape(-1, sampler.ndims[0])
        lp_max = sample_state.log_like.argmax()
        best = samples[lp_max]

        lp = sample_state.log_like.flatten()
        sorted = np.argsort(lp)
        inds_best = sorted[-1000:]
        inds_worst = sorted[:1000]

        best_full = sampler.log_like_fn.f.parameter_transforms["mbh"].both_transforms(
            best, copy=True
        )

        sampler.log_like_fn.f.template_model.init_heterodyne_info(best_full, **self.update_kwargs)

        if self.set_d_d_zero:
            sampler.log_like_fn.f.template_model.reference_d_d = 0.0

        # TODO: make this a general update function in Eryn (?)
        # samples[inds_worst] = samples[inds_best].copy()
        samples = samples.reshape(sampler.ntemps, sampler.nwalkers, 1, sampler.ndims[0])
        logp = sampler.compute_log_prior({"mbh": samples})
        logL, blobs = sampler.compute_log_like({"mbh": samples}, logp=logp)

        sample_state.branches["mbh"].coords = samples
        sample_state.log_like = logL
        sample_state.blobs = blobs

        # sampler.backend.save_step(sample_state, np.full_like(lp, True))


def get_psd_transform_container(
    Soms_fill: float = None,
    Sa_fill: float = None,
    n_knots: int = 5,
    freq_min: float = None,
    freq_max: float = None,
) -> TransformContainer:
    """Prepare a :class:`eryn.utils.transform.TransformContainer` for PSD sampling.

    Args:
        Soms_fill: Optical metrology noise level used to fill PSD knots.
        Sa_fill: Test-mass acceleration noise level used to fill PSD knots.
        n_knots: Number of spline knots used to parameterize the PSD.
        freq_min: Minimum frequency (Hz) of the PSD spline.
        freq_max: Maximum frequency (Hz) of the PSD spline.

    Returns:
        Configured ``TransformContainer`` for PSD parameter sampling.
    """
    # TODO/DOCS: function body is currently empty; this docstring describes
    # the intended interface but no transforms are actually constructed yet.
