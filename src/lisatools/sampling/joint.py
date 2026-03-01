"""
Joint GW + EM parameter estimation.

Combines a LISA gravitational-wave likelihood with an electromagnetic
(lcurve_rs) light-curve likelihood through a shared parameter mapping,
enabling simultaneous multi-messenger fitting of verification binaries.

Example
-------
>>> from lisatools.sampling.joint import JointFitter, ParameterMapping
>>> mapping = ParameterMapping.verification_binary(...)
>>> jf = JointFitter(em_model, em_data, gw_log_likelihood, mapping)
>>> result = jf.run_eryn(nwalkers=32, nsteps=5000, burn=1000, ntemps=8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# ParameterMapping
# ---------------------------------------------------------------------------

@dataclass
class ParameterMapping:
    """Map a joint parameter vector to EM (lcurve) and GW (LISA) sub-models.

    Attributes
    ----------
    joint_names : list[str]
        Ordered parameter names in the joint vector.
    em_param_names : list[str]
        lcurve parameter names for the EM subset.
    gw_param_names : list[str]
        Labels for the GW parameter subset.
    em_indices : list[int]
        Indices into the joint vector that feed the EM model.
    gw_indices : list[int]
        Indices into the joint vector that feed the GW model.
    em_transforms : dict
        ``{joint_idx: callable}`` — unit conversions from joint to EM-native.
    gw_transforms : dict
        ``{joint_idx: callable}`` — unit conversions from joint to GW-native.
    """

    joint_names: list
    em_param_names: list
    gw_param_names: list
    em_indices: list
    gw_indices: list
    em_transforms: dict = field(default_factory=dict)
    gw_transforms: dict = field(default_factory=dict)

    def joint_to_em(self, theta) -> np.ndarray:
        """Extract and transform EM parameters from the joint vector."""
        out = np.empty(len(self.em_indices))
        for i, idx in enumerate(self.em_indices):
            val = theta[idx]
            if idx in self.em_transforms:
                val = self.em_transforms[idx](val)
            out[i] = val
        return out

    def joint_to_gw(self, theta) -> np.ndarray:
        """Extract and transform GW parameters from the joint vector."""
        out = np.empty(len(self.gw_indices))
        for i, idx in enumerate(self.gw_indices):
            val = theta[idx]
            if idx in self.gw_transforms:
                val = self.gw_transforms[idx](val)
            out[i] = val
        return out

    @staticmethod
    def verification_binary(
        em_param_names: list,
        gw_param_names: list,
        shared_period_idx: int = 0,
    ) -> "ParameterMapping":
        """Factory for a standard verification binary mapping.

        Assumes the first parameter is the orbital period (seconds) shared
        between EM and GW, with standard transforms:

        - EM:  period used directly
        - GW:  ``f_GW = 2 / period``

        Any EM-only or GW-only parameters follow in order.

        Parameters
        ----------
        em_param_names : list[str]
            Names of EM parameters (period first, then EM-only).
        gw_param_names : list[str]
            Names of GW parameters (frequency first, then GW-only).
        shared_period_idx : int
            Index of the shared orbital period in the joint vector (default 0).
        """
        # Joint vector layout: [period, *em_only, *gw_only]
        n_em = len(em_param_names)
        n_gw = len(gw_param_names)

        # Period is shared (index 0 in joint by default)
        em_only_count = n_em - 1  # minus the shared period
        gw_only_count = n_gw - 1  # minus the shared frequency

        joint_names = [em_param_names[0]]  # period
        em_indices = [shared_period_idx]
        gw_indices = [shared_period_idx]

        # EM-only params
        for i, name in enumerate(em_param_names[1:]):
            idx = 1 + i
            joint_names.append(name)
            em_indices.append(idx)

        # GW-only params
        for i, name in enumerate(gw_param_names[1:]):
            idx = 1 + em_only_count + i
            joint_names.append(name)
            gw_indices.append(idx)

        # Transforms: GW frequency = 2 / period
        gw_transforms = {
            shared_period_idx: lambda p: 2.0 / p,
        }

        return ParameterMapping(
            joint_names=joint_names,
            em_param_names=em_param_names,
            gw_param_names=gw_param_names,
            em_indices=em_indices,
            gw_indices=gw_indices,
            em_transforms={},
            gw_transforms=gw_transforms,
        )


# ---------------------------------------------------------------------------
# JointFitter
# ---------------------------------------------------------------------------

class JointFitter:
    """Joint GW + EM parameter estimation.

    Parameters
    ----------
    em_model : lcurve_rs.Model
        EM light-curve model (a copy is made internally).
    em_data : str
        Path to the lcurve data file for EM fitting.
    gw_log_likelihood : callable
        GW log-likelihood: ``theta_gw → float`` (or vectorized).
    mapping : ParameterMapping
        Defines the parameter split between EM and GW.
    priors : dict or None
        ``{name: distribution_with_logpdf_rvs}`` over joint parameter names.
        If ``None``, you must provide priors via eryn's ProbDistContainer directly.
    em_scale : bool
        Whether to auto-scale the EM model to data (default True).
    gw_vectorized : bool
        Whether ``gw_log_likelihood`` accepts batched input (default False).
    """

    def __init__(
        self,
        em_model,
        em_data: str,
        gw_log_likelihood: Callable,
        mapping: ParameterMapping,
        priors: Optional[dict] = None,
        em_scale: bool = True,
        gw_vectorized: bool = False,
    ):
        # Lazy import to avoid hard dep at module level
        try:
            import lcurve_rs as _lcurve_rs
        except ImportError:
            _lcurve_rs = None

        self._em_model = em_model.copy() if hasattr(em_model, "copy") else em_model
        self._em_data = em_data
        self._em_scale = em_scale
        self._gw_ll = gw_log_likelihood
        self._gw_vectorized = gw_vectorized
        self.mapping = mapping
        self.priors = priors

        self.joint_names = list(mapping.joint_names)
        self.ndim = len(self.joint_names)

    # Large finite negative value used instead of -inf to avoid NaN
    # in eryn's internal move calculations (e.g. -inf * 0 = NaN).
    _BAD_LL = -1e100

    # -- Scalar likelihood --

    def log_likelihood_em(self, theta_joint) -> float:
        """EM log-likelihood for a single joint parameter vector."""
        theta_em = self.mapping.joint_to_em(theta_joint)
        try:
            self._em_model.set_params(self.mapping.em_param_names, list(theta_em))
            result = self._em_model.light_curve(
                data=self._em_data, scale=self._em_scale
            )
            ll = -0.5 * result.chisq
            if not math.isfinite(ll):
                return self._BAD_LL
            return ll
        except Exception:
            return self._BAD_LL

    def log_likelihood_gw(self, theta_joint) -> float:
        """GW log-likelihood for a single joint parameter vector."""
        theta_gw = self.mapping.joint_to_gw(theta_joint)
        try:
            ll = float(self._gw_ll(theta_gw))
            if not math.isfinite(ll):
                return self._BAD_LL
            return ll
        except Exception:
            return self._BAD_LL

    def log_likelihood(self, theta_joint) -> float:
        """Joint log-likelihood = EM + GW."""
        ll_em = self.log_likelihood_em(theta_joint)
        if ll_em <= self._BAD_LL:
            return self._BAD_LL
        ll_gw = self.log_likelihood_gw(theta_joint)
        if ll_gw <= self._BAD_LL:
            return self._BAD_LL
        return ll_em + ll_gw

    # -- Batch (vectorized) likelihood --

    def log_likelihood_batch(self, theta_batch: np.ndarray) -> np.ndarray:
        """Vectorised joint log-likelihood for (N, ndim) array.

        Uses lcurve_rs ``chisq_batch`` for EM (parallel Rayon evaluation).
        Falls back to per-row for GW unless ``gw_vectorized=True``.
        """
        theta_batch = np.atleast_2d(theta_batch)
        n = theta_batch.shape[0]

        # EM batch via chisq_batch
        em_params = np.array(
            [self.mapping.joint_to_em(theta_batch[i]) for i in range(n)]
        )
        em_params = np.ascontiguousarray(em_params, dtype=np.float64)
        try:
            chisq = self._em_model.chisq_batch(
                self._em_data,
                self.mapping.em_param_names,
                em_params,
                self._em_scale,
            )
            ll_em = -0.5 * np.asarray(chisq, dtype=np.float64)
            ll_em = np.where(np.isfinite(ll_em), ll_em, self._BAD_LL)
        except Exception:
            ll_em = np.full(n, self._BAD_LL)

        # GW
        if self._gw_vectorized:
            gw_params = np.array(
                [self.mapping.joint_to_gw(theta_batch[i]) for i in range(n)]
            )
            try:
                ll_gw = np.asarray(self._gw_ll(gw_params), dtype=np.float64)
                ll_gw = np.where(np.isfinite(ll_gw), ll_gw, self._BAD_LL)
            except Exception:
                ll_gw = np.full(n, self._BAD_LL)
        else:
            ll_gw = np.empty(n)
            for i in range(n):
                ll_gw[i] = self.log_likelihood_gw(theta_batch[i])

        result = ll_em + ll_gw
        result = np.where(np.isfinite(result), result, self._BAD_LL)
        return result

    # -- Sampler runners --

    def run_eryn(
        self,
        nwalkers: int = 0,
        nsteps: int = 5000,
        burn: int = 1000,
        ntemps: int = 8,
        vectorize: bool = True,
        thin_by: int = 1,
        **kwargs,
    ):
        """Run eryn parallel-tempered MCMC on the joint model.

        Parameters
        ----------
        nwalkers : int
            Number of walkers (default: ``max(32, 2*ndim + 2)``).
        nsteps : int
            Steps after burn-in (default 5000).
        burn : int
            Burn-in steps (default 1000).
        ntemps : int
            Number of temperatures (default 8).
        vectorize : bool
            Batch evaluation (default True).
        thin_by : int
            Thinning factor (default 1).

        Returns
        -------
        FitResult
            From ``lcurve_rs.fitting.FitResult``.
        """
        from eryn.ensemble import EnsembleSampler
        from eryn.prior import ProbDistContainer
        from lcurve_rs.fitting import FitResult

        if nwalkers <= 0:
            nwalkers = max(32, 2 * self.ndim + 2)

        # Build eryn priors
        if isinstance(self.priors, ProbDistContainer):
            eryn_priors = self.priors
        elif isinstance(self.priors, dict):
            eryn_priors = ProbDistContainer(self.priors)
        else:
            raise ValueError(
                "priors must be a dict or eryn ProbDistContainer"
            )

        if vectorize:
            log_fn = self.log_likelihood_batch
        else:
            log_fn = self.log_likelihood

        tempering_kwargs = kwargs.pop("tempering_kwargs", {})
        if ntemps > 1:
            tempering_kwargs.setdefault("ntemps", ntemps)

        actual_ntemps = tempering_kwargs.get("ntemps", ntemps) if ntemps > 1 else 1
        coords = eryn_priors.rvs(size=(actual_ntemps, nwalkers, 1))

        # Ensure all initial walkers have reasonable log-likelihoods
        # (eryn rejects NaN and +/-inf; we also avoid extreme negatives)
        max_resample = 200
        for t in range(actual_ntemps):
            for attempt in range(max_resample):
                flat = coords[t, :, 0, :]  # (nwalkers, ndim)
                ll = self.log_likelihood_batch(flat)
                bad = (ll <= self._BAD_LL) | ~np.isfinite(ll)
                if not bad.any():
                    break
                new = eryn_priors.rvs(size=(int(bad.sum()), 1))
                coords[t, bad, :, :] = new

        sampler = EnsembleSampler(
            nwalkers,
            self.ndim,
            log_fn,
            eryn_priors,
            tempering_kwargs=tempering_kwargs,
            vectorize=vectorize,
            **kwargs,
        )
        sampler.run_mcmc(coords, nsteps, burn=burn, progress=True, thin_by=thin_by)

        chain = sampler.get_chain(temp_index=0)["model_0"]
        chain = chain.reshape(-1, self.ndim)

        logl = sampler.get_log_like(temp_index=0)
        logl = logl.reshape(-1)

        return FitResult(
            backend="eryn",
            param_names=list(self.joint_names),
            samples=chain,
            log_likelihood=logl,
            raw_result=sampler,
        )

    def run_emcee(
        self,
        nwalkers: int = 0,
        nsteps: int = 5000,
        burn: int = 1000,
        vectorize: bool = True,
        **kwargs,
    ):
        """Run emcee MCMC on the joint model.

        Parameters
        ----------
        nwalkers : int
            Number of walkers (default: ``max(32, 2*ndim + 2)``).
        nsteps : int
            Total steps (default 5000).
        burn : int
            Burn-in steps (default 1000).
        vectorize : bool
            Batch evaluation (default True).

        Returns
        -------
        FitResult
        """
        import emcee
        from lcurve_rs.fitting import FitResult

        if nwalkers <= 0:
            nwalkers = max(32, 2 * self.ndim + 2)

        if self.priors is None:
            raise ValueError("emcee requires priors for initialization")

        # Build log_prior from the priors dict
        from eryn.prior import ProbDistContainer

        if isinstance(self.priors, ProbDistContainer):
            prior_container = self.priors
        else:
            prior_container = ProbDistContainer(self.priors)

        def log_prior(theta):
            return prior_container.logpdf(np.atleast_2d(theta)).item()

        def log_probability(theta):
            lp = log_prior(theta)
            if not math.isfinite(lp):
                return -math.inf
            return lp + self.log_likelihood(theta)

        def log_probability_batch(theta_batch):
            theta_batch = np.atleast_2d(theta_batch)
            n = theta_batch.shape[0]
            lp = np.array([log_prior(theta_batch[i]) for i in range(n)])
            result = np.full(n, -np.inf)
            finite_mask = np.isfinite(lp)
            if finite_mask.any():
                ll = self.log_likelihood_batch(theta_batch[finite_mask])
                result[finite_mask] = lp[finite_mask] + ll
            return result

        # Init walkers from priors
        p0 = prior_container.rvs(size=(nwalkers,))

        if vectorize:
            log_fn = log_probability_batch
        else:
            log_fn = log_probability

        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, log_fn, vectorize=vectorize,
        )
        sampler.run_mcmc(p0, nsteps, **kwargs)

        chain = sampler.get_chain(discard=burn, flat=True)
        logl = sampler.get_log_prob(discard=burn, flat=True)

        return FitResult(
            backend="emcee",
            param_names=list(self.joint_names),
            samples=chain,
            log_likelihood=logl,
            raw_result=sampler,
        )
