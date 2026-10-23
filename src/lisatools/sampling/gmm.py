"""Vectorized Gaussian-mixture fitting (EM) over many independent sample groups.

A batched fork of scikit-learn's ``GaussianMixture`` (BSD-3-Clause; see the
author lines below): every array carries a leading ``n_groups`` axis, so one
EM run fits an independent mixture to each group (one GB band, one leaf, one
cluster) on numpy or cupy. Only ``covariance_type="full"`` with the
``"random_from_data"`` initialization works (``"random"`` is coded but fails
on an invalid normalization axis); the other scikit-learn options raise
``NotImplementedError``. The commented-out blocks are the unported
scikit-learn code, kept for reference.

Entry points:

- :class:`GaussianMixtureModel`: the batched EM fitter.
- :class:`GMMFit`: maps each group's samples into the :math:`[-1, 1]^d` cube of
  its own min/max box, fits a :class:`GaussianMixtureModel` there, and
  evaluates / draws in physical coordinates (adding the map's log-Jacobian).
- :func:`vec_fit_gmm_min_bic`: per-group sweep of the component count that
  keeps the minimum-BIC fit; returns the component lists or a
  :class:`~lisatools.sampling.prior.FullGaussianMixtureModel`.
- :func:`fit_gb_gmm_rj_container`: GB samples -> fitted mixture -> the 8/9-column
  eryn RJ proposal container.

Callers. Run path (9mo / 12mo): the ``GB_WARM_SELF`` self-refit
(:mod:`lisatools.globalfit.warmstart.self_refit`) runs the ``fit_from_store``
step, whose :func:`~lisatools.globalfit.warmstart.fit_from_store.fit_cluster_gmms`
calls :func:`vec_fit_gmm_min_bic` with ``return_components=True`` (so
:class:`GMMFit` and :class:`GaussianMixtureModel` run there). Off the
production path: the GB GMM refit move
(:class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJRefitMove`,
not built with ``include_refit=False``) and :func:`fit_gb_gmm_rj_container`,
whose only caller
:class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJSerialSearchMCMC`
is built but never staged in the production stage lists, and the offline
F-statistic GMM layer :func:`~lisatools.sampling.fstat_proposal.fit_gmm_to_stacked`
(scripts only).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import typing
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time

import numpy as np
from scipy.special import logsumexp as logsumexp_cpu

try:
    import cupy as cp

    # cp.cuda.runtime.setDevice(7)
    from cupyx.scipy.special import logsumexp as logsumexp_gpu
except (ModuleNotFoundError, ImportError) as e:
    pass

# from .. import cluster
# from ..base import BaseEstimator, DensityMixin, _fit_context
# from ..cluster import kmeans_plusplus
# from ..exceptions import ConvergenceWarning
# from ..utils import check_random_state
# from ..utils._param_validation import Interval, StrOptions
# from ..utils.validation import check_is_fitted, validate_data
from lisatools.sampling.prior import FullGaussianMixtureModel
from lisatools.utils.utility import asnumpy, searchsorted2d_vec

import logging

logger = logging.getLogger(__name__)


# TODO check: keep/remove/document -- no callers (its scikit-learn users are commented out)
def _check_shape(param, param_shape, name, xp=None):
    """Validate the shape of the input parameter ``param``.

    Args:
        param (array-like): Parameter to check.
        param_shape (tuple): Expected shape.
        name (str): Parameter name for the error message.
        xp (module, optional): Array module; default numpy.

    Raises:
        ValueError: The shape differs from ``param_shape``.
    """
    if xp is None:
        xp = np

    param = xp.array(param)
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s"
            % (name, param_shape, param.shape)
        )


# class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
#     """Base class for mixture models.

#     This abstract class specifies an interface for all mixture classes and
#     provides basic common methods for mixture models.
#     """

#     _parameter_constraints: dict = {
#         "n_components": [Interval(Integral, 1, None, closed="left")],
#         "tol": [Interval(Real, 0.0, None, closed="left")],
#         "reg_covar": [Interval(Real, 0.0, None, closed="left")],
#         "max_iter": [Interval(Integral, 0, None, closed="left")],
#         "n_init": [Interval(Integral, 1, None, closed="left")],
#         "init_params": [
#             StrOptions({"kmeans", "random", "random_from_data", "k-means++"})
#         ],
#         "random_state": ["random_state"],
#         "warm_start": ["boolean"],
#         "verbose": ["verbose"],
#         "verbose_interval": [Interval(Integral, 1, None, closed="left")],
#     }

#     def __init__(
#         self,
#         n_components,
#         tol,
#         reg_covar,
#         max_iter,
#         n_init,
#         init_params,
#         random_state,
#         warm_start,
#         verbose,
#         verbose_interval,
#     ):
#         self.n_components = n_components
#         self.tol = tol
#         self.reg_covar = reg_covar
#         self.max_iter = max_iter
#         self.n_init = n_init
#         self.init_params = init_params
#         self.random_state = random_state
#         self.warm_start = warm_start
#         self.verbose = verbose
#         self.verbose_interval = verbose_interval

#     @abstractmethod
#     def _check_parameters(self, X):
#         """Check initial parameters of the derived class.

#         Parameters
#         ----------
#         X : array-like of shape  (n_samples, n_features)
#         """
#         pass


#     @abstractmethod
#     def _initialize(self, X, resp):
#         """Initialize the model parameters of the derived class.

#         Parameters
#         ----------
#         X : array-like of shape  (n_samples, n_features)

#         resp : array-like of shape (n_samples, n_components)
#         """
#         pass


#     @abstractmethod
#     def _m_step(self, X, log_resp):
#         """M step.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)

#         log_resp : array-like of shape (n_samples, n_components)
#             Logarithm of the posterior probabilities (or responsibilities) of
#             the point of each sample in X.
#         """
#         pass

#     @abstractmethod
#     def _get_parameters(self):
#         pass

#     @abstractmethod
#     def _set_parameters(self, params):
#         pass

#     def predict(self, X):
#         """Predict the labels for the data samples in X using trained model.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             List of n_features-dimensional data points. Each row
#             corresponds to a single data point.

#         Returns
#         -------
#         labels : array, shape (n_samples,)
#             Component labels.
#         """
#         check_is_fitted(self)
#         X = validate_data(self, X, reset=False)
#         return self._estimate_weighted_log_prob(X).argmax(axis=1)


#     @abstractmethod
#     def _estimate_log_weights(self):
#         """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

#         Returns
#         -------
#         log_weight : array, shape (n_components, )
#         """
#         pass

#     @abstractmethod
#     def _estimate_log_prob(self, X):
#         """Estimate the log-probabilities log P(X | Z).

#         Compute the log-probabilities per each component for each sample.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)

#         Returns
#         -------
#         log_prob : array, shape (n_samples, n_component)
#         """
#         pass


# """Gaussian Mixture Model."""

# # Authors: The scikit-learn developers
# # SPDX-License-Identifier: BSD-3-Clause

# import numpy as np
# from scipy import linalg

# from ..utils import check_array
# from ..utils._param_validation import StrOptions
# from ..utils.extmath import row_norms
# from ._base import BaseMixture, _check_shape

# ###############################################################################
# # Gaussian mixture shape checkers used by the GaussianMixture class


# def _check_weights(weights, n_components):
#     """Check the user provided 'weights'.

#     Parameters
#     ----------
#     weights : array-like of shape (n_components,)
#         The proportions of components of each mixture.

#     n_components : int
#         Number of components.

#     Returns
#     -------
#     weights : array, shape (n_components,)
#     """
#     weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
#     _check_shape(weights, (n_components,), "weights")

#     # check range
#     if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
#         raise ValueError(
#             "The parameter 'weights' should be in the range "
#             "[0, 1], but got max value %.5f, min value %.5f"
#             % (np.min(weights), np.max(weights))
#         )

#     # check normalization
#     atol = 1e-6 if weights.dtype == np.float32 else 1e-8
#     if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0, atol=atol):
#         raise ValueError(
#             "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
#             % np.sum(weights)
#         )
#     return weights


# def _check_means(means, n_components, n_features):
#     """Validate the provided 'means'.

#     Parameters
#     ----------
#     means : array-like of shape (n_components, n_features)
#         The centers of the current components.

#     n_components : int
#         Number of components.

#     n_features : int
#         Number of features.

#     Returns
#     -------
#     means : array, (n_components, n_features)
#     """
#     means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
#     _check_shape(means, (n_components, n_features), "means")
#     return means


# def _check_precision_positivity(precision, covariance_type):
#     """Check a precision vector is positive-definite."""
#     if np.any(np.less_equal(precision, 0.0)):
#         raise ValueError("'%s precision' should be positive" % covariance_type)


# def _check_precision_matrix(precision, covariance_type):
#     """Check a precision matrix is symmetric and positive-definite."""
#     if not (
#         np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
#     ):
#         raise ValueError(
#             "'%s precision' should be symmetric, positive-definite" % covariance_type
#         )


# def _check_precisions_full(precisions, covariance_type):
#     """Check the precision matrices are symmetric and positive-definite."""
#     for prec in precisions:
#         _check_precision_matrix(prec, covariance_type)


# def _check_precisions(precisions, covariance_type, n_components, n_features):
#     """Validate user provided precisions.

#     Parameters
#     ----------
#     precisions : array-like
#         'full' : shape of (n_components, n_features, n_features)
#         'tied' : shape of (n_features, n_features)
#         'diag' : shape of (n_components, n_features)
#         'spherical' : shape of (n_components,)

#     covariance_type : str

#     n_components : int
#         Number of components.

#     n_features : int
#         Number of features.

#     Returns
#     -------
#     precisions : array
#     """
#     precisions = check_array(
#         precisions,
#         dtype=[np.float64, np.float32],
#         ensure_2d=False,
#         allow_nd=covariance_type == "full",
#     )

#     precisions_shape = {
#         "full": (n_components, n_features, n_features),
#         "tied": (n_features, n_features),
#         "diag": (n_components, n_features),
#         "spherical": (n_components,),
#     }
#     _check_shape(
#         precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
#     )

#     _check_precisions = {
#         "full": _check_precisions_full,
#         "tied": _check_precision_matrix,
#         "diag": _check_precision_positivity,
#         "spherical": _check_precision_positivity,
#     }
#     _check_precisions[covariance_type](precisions, covariance_type)
#     return precisions


# ###############################################################################
# # Gaussian mixture parameters estimators (used by the M-Step)


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar, xp=None):
    """Estimate the full covariance matrices of every group's components.

    Args:
        resp (xp.ndarray): Responsibilities, ``(n_groups, n_samples, n_components)``.
        X (xp.ndarray): Data, ``(n_groups, n_samples, n_features)``.
        nk (xp.ndarray): Effective counts per component, ``(n_groups, n_components)``.
        means (xp.ndarray): Component means, ``(n_groups, n_components, n_features)``.
        reg_covar (float): Added to every covariance diagonal.
        xp (module, optional): Array module; default numpy.

    Returns:
        xp.ndarray: Covariances, ``(n_groups, n_components, n_features, n_features)``.
    """
    if xp is None:
        xp = np

    n_groups, n_components, n_features = means.shape
    covariances = xp.empty((n_groups, n_components, n_features, n_features), dtype=X.dtype)
    tmp0 = xp.repeat(xp.arange(n_groups), n_features)
    tmp2 = xp.tile(xp.arange(n_features), (n_groups,))
    tmp3 = tmp2.copy()
    for k in range(n_components):
        tmp1 = xp.full_like(tmp0, k)
        inds_tmp = (tmp0, tmp1, tmp2, tmp3)
        diff = X - means[:, k][:, None, :]
        covariances[:, k] = (
            xp.einsum("ijk,ijl->ikl", (resp[:, :, k][:, :, None] * diff), diff)
            / nk[:, k][:, None, None]
        )
        covariances[inds_tmp] += reg_covar
    return covariances


# def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
#     """Estimate the tied covariance matrix.

#     Parameters
#     ----------
#     resp : array-like of shape (n_samples, n_components)

#     X : array-like of shape (n_samples, n_features)

#     nk : array-like of shape (n_components,)

#     means : array-like of shape (n_components, n_features)

#     reg_covar : float

#     Returns
#     -------
#     covariance : array, shape (n_features, n_features)
#         The tied covariance matrix of the components.
#     """
#     avg_X2 = np.dot(X.T, X)
#     avg_means2 = np.dot(nk * means.T, means)
#     covariance = avg_X2 - avg_means2
#     covariance /= nk.sum()
#     covariance.flat[:: len(covariance) + 1] += reg_covar
#     return covariance


# def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
#     """Estimate the diagonal covariance vectors.

#     Parameters
#     ----------
#     responsibilities : array-like of shape (n_samples, n_components)

#     X : array-like of shape (n_samples, n_features)

#     nk : array-like of shape (n_components,)

#     means : array-like of shape (n_components, n_features)

#     reg_covar : float

#     Returns
#     -------
#     covariances : array, shape (n_components, n_features)
#         The covariance vector of the current components.
#     """
#     avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
#     avg_means2 = means**2
#     return avg_X2 - avg_means2 + reg_covar


# def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
#     """Estimate the spherical variance values.

#     Parameters
#     ----------
#     responsibilities : array-like of shape (n_samples, n_components)

#     X : array-like of shape (n_samples, n_features)

#     nk : array-like of shape (n_components,)

#     means : array-like of shape (n_components, n_features)

#     reg_covar : float

#     Returns
#     -------
#     variances : array, shape (n_components,)
#         The variance values of each components.
#     """
#     return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type, xp=None):
    """Estimate the Gaussian distribution parameters (the M step) per group.

    Args:
        X (xp.ndarray): Data, ``(n_groups, n_samples, n_features)``.
        resp (xp.ndarray): Responsibilities, ``(n_groups, n_samples, n_components)``.
        reg_covar (float): Regularization added to the covariance diagonals.
        covariance_type (str): Only ``"full"`` is implemented (any other value
            raises ``KeyError``).
        xp (module, optional): Array module; default numpy.

    Returns:
        tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(nk, means, covariances)``:
        effective counts ``(n_groups, n_components)`` (unnormalized, plus
        ``10 eps``), means ``(n_groups, n_components, n_features)`` and
        covariances ``(n_groups, n_components, n_features, n_features)``.
    """
    if xp is None:
        xp = np

    nk = resp.sum(axis=1) + 10 * np.finfo(resp.dtype).eps
    # means = np.dot(resp.T, X) / nk[:, np.newaxis]
    means = xp.einsum("lij,lik->ljk", resp, X) / nk[:, :, xp.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        # "tied": _estimate_gaussian_covariances_tied,
        # "diag": _estimate_gaussian_covariances_diag,
        # "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar, xp=xp)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type, xp=None):
    """Compute the Cholesky factors of the precision matrices.

    For each component, ``L = cholesky(cov)`` and the returned factor is
    ``(L^{-1})^T``, so ``prec_chol prec_chol^T`` is the precision matrix.

    Args:
        covariances (xp.ndarray): ``(n_groups, n_components, n_features, n_features)``.
        covariance_type (str): ``"full"``; anything else raises
            ``NotImplementedError``.
        xp (module, optional): Array module; default numpy.

    Returns:
        xp.ndarray: Precision Cholesky factors, same shape as ``covariances``.

    Raises:
        ValueError: A covariance is not positive definite (collapsed or
            singleton component).
        NotImplementedError: ``covariance_type`` is not ``"full"``.
    """
    if xp is None:
        xp = np

    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "increase reg_covar, or scale the input data."
    )
    dtype = covariances.dtype
    if dtype == np.float32:
        estimate_precision_error_message += (
            " The numerical accuracy can also be improved by passing float64"
            " data instead of float32."
        )

    if covariance_type == "full":
        n_groups, n_components, n_features, _ = covariances.shape
        eye = xp.zeros((n_groups, n_features, n_features))
        for i in range(n_features):
            eye[:, i, i] = 1.0

        precisions_chol = xp.empty((n_groups, n_components, n_features, n_features), dtype=dtype)
        for k in range(n_components):
            covariance = covariances[:, k]
            try:
                cov_chol = xp.linalg.cholesky(covariance)  # , lower=True)
            except xp.linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            # precisions_chol[k] = linalg.solve_triangular(
            #     cov_chol, np.eye(n_features, dtype=dtype), lower=True
            # ).T
            precisions_chol[:, k] = xp.linalg.solve(cov_chol, eye).transpose(0, 2, 1)

    elif covariance_type == "tied":
        raise NotImplementedError
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features, dtype=dtype), lower=True
        ).transpose()
    else:
        raise NotImplementedError
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


# def _flipudlr(array):
#     """Reverse the rows and columns of an array."""
#     return np.flipud(np.fliplr(array))


# def _compute_precision_cholesky_from_precisions(precisions, covariance_type):
#     r"""Compute the Cholesky decomposition of precisions using precisions themselves.

#     As implemented in :func:`_compute_precision_cholesky`, the `precisions_cholesky_` is
#     an upper-triangular matrix for each Gaussian component, which can be expressed as
#     the $UU^T$ factorization of the precision matrix for each Gaussian component, where
#     $U$ is an upper-triangular matrix.

#     In order to use the Cholesky decomposition to get $UU^T$, the precision matrix
#     $\Lambda$ needs to be permutated such that its rows and columns are reversed, which
#     can be done by applying a similarity transformation with an exchange matrix $J$,
#     where the 1 elements reside on the anti-diagonal and all other elements are 0. In
#     particular, the Cholesky decomposition of the transformed precision matrix is
#     $J\Lambda J=LL^T$, where $L$ is a lower-triangular matrix. Because $\Lambda=UU^T$
#     and $J=J^{-1}=J^T$, the `precisions_cholesky_` for each Gaussian component can be
#     expressed as $JLJ$.

#     Refer to #26415 for details.

#     Parameters
#     ----------
#     precisions : array-like
#         The precision matrix of the current components.
#         The shape depends on the covariance_type.

#     covariance_type : {'full', 'tied', 'diag', 'spherical'}
#         The type of precision matrices.

#     Returns
#     -------
#     precisions_cholesky : array-like
#         The cholesky decomposition of sample precisions of the current
#         components. The shape depends on the covariance_type.
#     """
#     if covariance_type == "full":
#         precisions_cholesky = np.array(
#             [
#                 _flipudlr(linalg.cholesky(_flipudlr(precision), lower=True))
#                 for precision in precisions
#             ]
#         )
#     elif covariance_type == "tied":
#         precisions_cholesky = _flipudlr(
#             linalg.cholesky(_flipudlr(precisions), lower=True)
#         )
#     else:
#         precisions_cholesky = np.sqrt(precisions)
#     return precisions_cholesky


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features, xp=None):
    """Compute the log-determinant of triangular Cholesky factors.

    Sums the logs of the diagonal entries.

    Args:
        matrix_chol (xp.ndarray): Triangular factors,
            ``(n_groups, n_components, n_features, n_features)``.
        covariance_type (str): ``"full"``; anything else raises
            ``NotImplementedError``.
        n_features (int): Number of features.
        xp (module, optional): Array module; default numpy.

    Returns:
        xp.ndarray: ``(n_groups, n_components)`` log-determinants (for a
        precision Cholesky factor: half the log-determinant of the precision).
    """
    if xp is None:
        xp = np

    if covariance_type == "full":
        n_groups, n_components, _, _ = matrix_chol.shape
        log_det_chol = xp.sum(
            xp.log(matrix_chol.reshape(n_groups, n_components, -1)[:, :, :: n_features + 1]),
            axis=-1,
        )

        # log_det_chol = np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), axis=-1)

    elif covariance_type == "tied":
        raise NotImplementedError
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        raise NotImplementedError
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        raise NotImplementedError
        log_det_chol = n_features * np.log(matrix_chol)

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type, xp=None):
    """Estimate the per-component log Gaussian density of every sample.

    Args:
        X (xp.ndarray): Data, ``(n_groups, n_samples, n_features)``.
        means (xp.ndarray): ``(n_groups, n_components, n_features)``.
        precisions_chol (xp.ndarray): Precision Cholesky factors,
            ``(n_groups, n_components, n_features, n_features)``.
        covariance_type (str): ``"full"``; anything else raises
            ``NotImplementedError``.
        xp (module, optional): Array module; default numpy.

    Returns:
        xp.ndarray: ``(n_groups, n_samples, n_components)`` log densities
        (unweighted).
    """
    if xp is None:
        xp = np

    n_groups, n_samples, n_features = X.shape
    n_groups, n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features, xp=xp)

    if covariance_type == "full":
        log_prob = xp.empty((n_groups, n_samples, n_components), dtype=X.dtype)
        # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        for k in range(n_components):
            mu = means[:, k]
            prec_chol = precisions_chol[:, k]
            # y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            y = (
                xp.einsum("ijk,ikl->ijl", X, prec_chol)
                - xp.einsum("ik,ikl->il", mu, prec_chol)[:, None, :]
            )
            log_prob[:, :, k] = xp.sum(xp.square(y), axis=-1)

    elif covariance_type == "tied":
        raise NotImplementedError
        log_prob = np.empty((n_samples, n_components), dtype=X.dtype)
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        raise NotImplementedError
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    elif covariance_type == "spherical":
        raise NotImplementedError
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * xp.log(2 * np.pi).astype(X.dtype) + log_prob) + log_det[:, None, :]


def draw_multinomial_vec(n_samples, weights, random_state, xp=None):
    """Vectorized multinomial draw across mixture groups.

    For each row of ``weights`` (one mixture / group), draw ``n_samples`` from
    the corresponding categorical distribution and return the per-component
    counts.

    Args:
        n_samples (int): Number of categorical draws per group.
        weights (xp.ndarray): Array of shape ``(n_groups, n_components)`` of
            per-group mixture weights (renormalized per row here).
        random_state (object): Object exposing ``multinomial(n, pvals)``
            (numpy / cupy ``random`` or a ``RandomState``); called once per
            group.
        xp (module, optional): Array module (numpy or cupy). Defaults to numpy.

    Returns:
        xp.ndarray: Integer counts of shape ``(n_groups, n_components)``; each
        row sums to ``n_samples``.
    """
    if xp is None:
        xp = np
        
    n_groups, n_components = weights.shape 
    counts = xp.zeros((n_groups, n_components), dtype=int)
    
    # Normalize weights per group to guard against any floating point drift
    weights_norm = weights / weights.sum(axis=-1, keepdims=True)
    
    for g in range(n_groups):
        counts[g] = xp.array(
            random_state.multinomial(n_samples, weights_norm[g].tolist())
        )
    
    return counts
    # n_groups, n_components = weights.shape
    # # weights_tmp = xp.cumsum(weights, axis=-1)
    # cumulative_weights = xp.concatenate(
    #     [xp.zeros((n_groups, 1)), xp.cumsum(weights, axis=-1)], axis=1
    # )
    # draw = random_state.rand(n_groups, n_samples)
    # extra_kwargs = {}
    # try:
    #     extra_kwargs["gpu"] = xp.cuda.runtime.getDevice()
    # except AttributeError:
    #     pass

    # component = (
    #     searchsorted2d_vec(cumulative_weights, draw, side="right", xp=xp, **extra_kwargs) - 1
    # ).reshape(draw.shape)

    # # stride must be n_components, not n_components + 1
    # component_tmp = n_components * xp.repeat(xp.arange(n_groups), n_samples).reshape(
    #     n_groups, n_samples
    # )
    # counts = xp.zeros((n_groups, n_components), dtype=int)
    # uni, uni_counts = xp.unique((component + component_tmp).flatten(), return_counts=True)
    # group = uni // n_components  # matching stride
    # comp = uni % n_components    # matching stride
    # counts[group, comp] = uni_counts
    # return counts
    # # OLD
    # cumulative_weights = xp.concatenate(
    #     [xp.zeros((n_groups, 1)), xp.cumsum(weights, axis=-1)], axis=1
    # )
    # draw = random_state.rand(n_groups, n_samples)
    # extra_kwargs = {}
    # try:
    #     extra_kwargs["gpu"] = xp.cuda.runtime.getDevice()
    # except AttributeError:
    #     pass

    # component = (
    #     searchsorted2d_vec(cumulative_weights, draw, side="right", xp=xp, **extra_kwargs) - 1
    # ).reshape(draw.shape)

    # component_tmp = (n_components + 1) * xp.repeat(xp.arange(n_groups), n_samples).reshape(
    #     n_groups, n_samples
    # )
    # counts = xp.zeros((n_groups, n_components), dtype=int)
    # uni, uni_counts = xp.unique((component + component_tmp).flatten(), return_counts=True)
    # group = uni // (n_components + 1)
    # comp = uni % (n_components + 1)
    # counts[group, comp] = uni_counts
    # return counts


class GaussianMixtureModel:
    """Batched Gaussian mixture model fitted by EM, one mixture per group.

    A fork of scikit-learn's ``GaussianMixture`` with a leading ``n_groups``
    axis on every array: :meth:`fit` takes ``X`` of shape
    ``(n_groups, n_samples, n_features)`` and runs EM on all groups at once,
    freezing a group once its lower-bound change drops below ``tol``. Runs on
    numpy, or on cupy when ``gpu`` is a device index. In the global fit it is
    constructed only through :class:`GMMFit` (``init_params="random_from_data"``,
    ``max_iter=1000``).

    Args:
        n_components (int, optional): Components per group. Default ``1``.
        covariance_type (str, optional): Only ``"full"`` is implemented; the
            scikit-learn ``"tied"`` / ``"diag"`` / ``"spherical"`` raise
            ``NotImplementedError`` (or ``KeyError``) during :meth:`fit`.
            Default ``"full"``.
        tol (float, optional): Convergence threshold on the per-group change
            of the lower bound (the mean log-likelihood). Default ``1e-3``.
        reg_covar (float, optional): Non-negative value added to every
            covariance diagonal. Default ``1e-6``.
        max_iter (int, optional): Maximum EM iterations. ``0`` raises
            ``NotImplementedError``. Default ``100``.
        n_init (int, optional): Number of initializations (see
            :meth:`fit_predict` for which trial is kept). Default ``1``.
        init_params (str, optional): ``"random_from_data"`` (initial means are
            ``n_components`` data points, the same sample indices in every
            group) is the working option. ``"random"`` (random
            responsibilities) currently fails: its normalization sums over an
            invalid axis. ``"kmeans"`` and ``"k-means++"`` raise
            ``NotImplementedError``, so the default ``"kmeans"`` must be
            overridden.
        weights_init: Not implemented; anything but ``None`` raises
            ``NotImplementedError`` in :meth:`fit`.
        means_init: Not implemented; as ``weights_init``.
        precisions_init: Not implemented; as ``weights_init``.
        random_state (int, RandomState or None, optional): ``None`` uses the
            array module's global ``random`` (unseeded); an ``int`` is turned
            into ``xp.random.RandomState(seed)``; anything else is used as is.
            Drives the initialization and :meth:`sample`. Default ``None``.
        warm_start (bool, optional): Kept from scikit-learn: with ``True`` a
            second :meth:`fit` skips the initialization, but
            :meth:`fit_predict` re-zeroes the parameter arrays before that
            point, so a warm-started fit runs EM from all-zero parameters and
            does not work. Default ``False``.
        verbose (int, optional): ``1`` logs each initialization and every
            ``verbose_interval`` iterations; ``>= 2`` adds timings but raises
            ``TypeError`` at the end of an initialization (it formats the
            per-group lower-bound array as a scalar). Default ``0``.
        verbose_interval (int, optional): Iterations between verbose log
            lines. Default ``10``.
        gpu (int, optional): CUDA device index; when set, computes with cupy
            and calls ``cupy.cuda.runtime.setDevice(gpu)`` at construction.
            ``None`` (default) computes with numpy.

    Attributes:
        weights_ (xp.ndarray): ``(n_groups, n_components)`` weights.
        means_ (xp.ndarray): ``(n_groups, n_components, n_features)``.
        covariances_ (xp.ndarray): ``(n_groups, n_components, n_features, n_features)``.
        precisions_cholesky_ (xp.ndarray): Precision Cholesky factors, same
            shape as ``covariances_``.
        precisions_ (xp.ndarray): Precision matrices, same shape.
        inv_covs_ (xp.ndarray): ``inv(covariances_)`` (set at the end of
            :meth:`fit`).
        det_covs_ (xp.ndarray): ``det(covariances_)``, ``(n_groups, n_components)``.
        converged_ (xp.ndarray): ``(n_groups,)`` bool, EM converged per group.
        n_iter_ (xp.ndarray): ``(n_groups,)`` EM iterations used per group.
        lower_bound_ (xp.ndarray): ``(n_groups,)`` final lower bound.
        lower_bounds_ (list): One entry per EM iteration of the kept
            initialization; every entry is the same lower-bound array object,
            so all of them show its final values.

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.gmm import GaussianMixtureModel
        >>> X = np.random.default_rng(0).normal(size=(3, 500, 2))
        >>> gmm = GaussianMixtureModel(
        ...     n_components=2, init_params="random_from_data", random_state=0
        ... )
        >>> _ = gmm.fit(X)
        >>> gmm.weights_.shape, gmm.means_.shape, gmm.logpdf(X).shape
        ((3, 2), (3, 2, 2), (3, 500))
    """

    # _parameter_constraints: dict = {
    #     **BaseMixture._parameter_constraints,
    #     "covariance_type": [StrOptions({"full", "tied", "diag", "spherical"})],
    #     "weights_init": ["array-like", None],
    #     "means_init": ["array-like", None],
    #     "precisions_init": ["array-like", None],
    # }

    @property
    def xp(self):
        """Array module (``numpy`` or ``cupy``) used for the computations."""
        if self.use_gpu:
            return cp
        else:
            return np

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        gpu=None,
    ):
        self.gpu = gpu
        if gpu is not None:
            self.use_gpu = True
            self.xp.cuda.runtime.setDevice(gpu)

        else:
            self.use_gpu = False

        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        # ``None`` -> the array module itself (numpy/cupy ``random``), the
        # historical unseeded behaviour. An INT is normalised to that
        # module's ``RandomState``, which carries every method this class
        # calls (choice / multinomial / rand / uniform / randn /
        # standard_normal / multivariate_normal) while making a refit of the
        # same data reproducible -- ``init_params="random_from_data"`` draws
        # fresh starting points on every call, so an unseeded refit of one
        # store gave different component counts AND different components
        # (2026-09-18). Anything else is passed through untouched, so a
        # caller supplying its own generator still works.
        if random_state is None:
            random_state = self.xp.random
        elif isinstance(random_state, (int, np.integer)):
            random_state = self.xp.random.RandomState(int(random_state))

        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    @property
    def xp(self):
        """Array module (``numpy`` or ``cupy``) used for the computations."""
        if self.use_gpu:
            return cp
        else:
            return np

    @property
    def logsumexp(self):
        """callable: ``logsumexp`` matching the array module (scipy or cupyx.scipy)."""
        if self.use_gpu:
            return logsumexp_gpu
        else:
            return logsumexp_cpu

    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        """Log the end of one initialization (how many groups converged) when verbose."""
        converged_msg = (
            f"converged {init_has_converged.sum()} out of {init_has_converged.shape[0]}."
        )
        if self.verbose == 1:
            logger.info(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            logger.info(
                f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound" f" {lb:.5f}."
            )

    def _check_parameters(self, X):
        """Reject user-provided initial parameters (not implemented in the batched fork).

        Raises:
            NotImplementedError: Any of ``weights_init`` / ``means_init`` /
                ``precisions_init`` is set.
        """
        _, _, n_features = X.shape

        if self.weights_init is not None:
            raise NotImplementedError
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            raise NotImplementedError
            self.means_init = _check_means(self.means_init, self.n_components, n_features)

        if self.precisions_init is not None:

            raise NotImplementedError
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Log the EM iteration every ``verbose_interval`` iterations when verbose."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                logger.info("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                logger.info(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _initialize_parameters(self, X, random_state):
        """Run initialization, skipping it if all init arrays were user-provided."""
        # If all the initial parameters are all provided, then there is no need to run
        # the initialization.
        compute_resp = (
            self.weights_init is None or self.means_init is None or self.precisions_init is None
        )
        if compute_resp:
            # super()._initialize_parameters(X, random_state)
            self._initialize_parameters_inner(X, random_state)
        else:
            self._initialize(X, None)

    def _initialize_parameters_inner(self, X, random_state):
        """Initialize the responsibilities by ``init_params`` and set the parameters.

        Args:
            X (xp.ndarray): Data, ``(n_groups, n_samples, n_features)``.
            random_state (object): Random generator (``choice`` / ``uniform``)
                for the chosen initialization.

        Raises:
            NotImplementedError: ``init_params`` is ``"kmeans"`` or
                ``"k-means++"``.
        """
        n_groups, n_samples, _ = X.shape

        if self.init_params == "kmeans":
            raise NotImplementedError
            resp = self.xp.zeros((n_samples, self.n_components), dtype=X.dtype)
            label = (
                cluster.KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state)
                .fit(X)
                .labels_
            )
            resp[self.xp.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = self.xp.asarray(
                random_state.uniform(size=(n_groups, n_samples, self.n_components)),
                dtype=X.dtype,
            )
            # resp /= resp.sum(axis=1)[:, self.xp.newaxis]
            resp /= resp.sum(axis=-11)[:, :, self.xp.newaxis]
        elif self.init_params == "random_from_data":
            resp = self.xp.zeros((n_groups, n_samples, self.n_components), dtype=X.dtype)
            indices = random_state.choice(n_samples, size=self.n_components, replace=False)
            tmp2 = self.xp.tile(self.xp.arange(self.n_components), (n_groups, 1)).flatten()
            tmp1 = self.xp.tile(indices, (n_groups, 1)).flatten()
            tmp0 = self.xp.repeat(self.xp.arange(n_groups), indices.shape[0])
            resp[(tmp0, tmp1, tmp2)] = 1
        elif self.init_params == "k-means++":
            raise NotImplementedError
            resp = self.xp.zeros((n_samples, self.n_components), dtype=X.dtype)
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            resp[indices, self.xp.arange(self.n_components)] = 1
        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Set the initial weights, means, covariances and precision factors.

        Args:
            X (xp.ndarray): Data, ``(n_groups, n_samples, n_features)``.
            resp (xp.ndarray or None): Initial responsibilities,
                ``(n_groups, n_samples, n_components)``.
        """
        n_groups, n_samples, _ = X.shape
        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, self.reg_covar, self.covariance_type, xp=self.xp
            )
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type, xp=self.xp
            )
        else:
            self.precisions_cholesky_ = _compute_precision_cholesky_from_precisions(
                self.precisions_init, self.covariance_type
            )

    def score_samples(self, X):
        """Compute the mixture log-density of each sample under its group's mixture.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``; group ``g`` of
                ``X`` is scored with mixture ``g``.

        Returns:
            xp.ndarray: ``(n_groups, n_samples)`` log-densities.
        """
        # check_is_fitted(self)
        # X = validate_data(self, X, reset=False)

        return self.logsumexp(self._estimate_weighted_log_prob(X), axis=-1)

    def score(self, X, y=None):
        """Compute the per-group mean log-density of ``X``.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            y: Ignored (scikit-learn API).

        Returns:
            xp.ndarray: ``(n_groups,)`` mean of :meth:`score_samples` over the
            samples.
        """
        return self.score_samples(X).mean(axis=-1)

    def logpdf(self, X):
        """Alias for :meth:`score_samples`."""
        return self.score_samples(X)

    # TODO check: keep/remove/document -- no callers (scikit-learn API kept from the fork)
    def predict_proba(self, X):
        """Return the component responsibilities of each sample.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.

        Returns:
            xp.ndarray: ``(n_groups, n_samples, n_components)`` posterior
            component probabilities.
        """
        # check_is_fitted(self)
        # X = validate_data(self, X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return self.xp.exp(log_resp)

    def _estimate_log_prob_resp(self, X, converged=None):
        """Estimate log probabilities and responsibilities for each sample.

        Only the groups with ``converged == False`` are evaluated.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            converged (xp.ndarray, optional): ``(n_groups,)`` bool; default all
                ``False``.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(log_prob_norm, log_resp)``:
            ``log p(x)`` of shape ``(n_active, n_samples)`` and the log
            responsibilities ``(n_active, n_samples, n_components)``.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        weighted_log_prob = self._estimate_weighted_log_prob(X, converged=converged)
        log_prob_norm = self.logsumexp(weighted_log_prob, axis=-1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, :, self.xp.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X, converged=None):
        """Estimate the weighted log-probabilities, ``log P(X | Z) + log weights``.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            converged (xp.ndarray, optional): ``(n_groups,)`` bool; converged
                groups are skipped. Default all ``False``.

        Returns:
            xp.ndarray: ``(n_active, n_samples, n_components)``.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        return (
            self._estimate_log_prob(X, converged=converged)
            + self._estimate_log_weights(converged=converged)[:, None, :]
        )

    def sample(self, n_samples=1, flat=False):
        """Generate random samples from the fitted mixtures.

        Component counts come from :func:`draw_multinomial_vec`; the draws are
        ``mean + L z`` with ``L = cholesky(cov)``, written contiguously per
        component (so samples are grouped by component, not shuffled).

        Args:
            n_samples (int, optional): Samples per group. Default ``1``.
            flat (bool, optional): If ``True``, treat all groups' components
                as ONE mixture (weights renormalized over the whole set) and
                draw ``n_samples`` from it (output group axis of length 1).
                Default ``False``.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(X, counts, comp)``:
            samples ``(n_groups, n_samples, n_features)`` (``n_groups = 1``
            when ``flat``), per-component counts ``(n_groups, n_components)``
            and the component index of each sample ``(n_groups, n_samples)``
            (flat index ``group * n_components + k`` when ``flat``).

        Raises:
            ValueError: ``n_samples < 1``.
        """
        # check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        if not flat:
            weights = self.weights_
            means = self.means_
            covariances = self.covariances_
            n_components = self.n_components

        else:
            weights = self.weights_.reshape(1, -1) / self.weights_.sum()
            means = self.means_.reshape((1, -1) + self.means_.shape[-1:])
            covariances = self.covariances_.reshape((1, -1) + self.covariances_.shape[-2:])
            n_components = self.weights_.shape[0] * self.n_components
            # new_precisions_cholesky = self.gmm.precisions_cholesky_.reshape((1, -1) + self.gmm.precisions_cholesky_.shape[-2:])

        n_groups, _, n_features = means.shape
        rng = self.random_state

        n_samples_comp = draw_multinomial_vec(n_samples, weights, rng, xp=self.xp)

        # n_samples_comp = rng.multinomial(n_samples, self.weights_)
        assert (n_samples_comp.sum(axis=-1) == n_samples).all(), \
        f"draw_multinomial_vec row sums {n_samples_comp.sum(axis=-1)} != {n_samples}"

        X = self.xp.zeros((n_groups, n_samples, n_features))
        samples_so_far = self.xp.zeros((n_groups, n_samples), dtype=bool)
        comp_out = self.xp.zeros((n_groups, n_samples), dtype=int)
        chol_decomp = self.xp.linalg.cholesky(covariances)
        
        if self.covariance_type == "full":
            fill_pos = self.xp.zeros(n_groups, dtype=int)  # next unfilled index per group

            for k in range(n_components):
                n_samp_k = n_samples_comp[:, k]          # shape (n_groups,)
                n_samp_k_max = n_samp_k.max().item()
                if n_samp_k_max == 0:
                    continue

                z = rng.randn(n_groups, n_samp_k_max, n_features)
                _samples = means[:, k][:, None, :] + self.xp.einsum(
                    "ijk,imk->imj", chol_decomp[:, k], z
                )  # shape (n_groups, n_samp_k_max, n_features)

                for g in range(n_groups):
                    n_g = n_samp_k[g].item()
                    if n_g == 0:
                        continue
                    start = fill_pos[g].item()
                    X[g, start:start + n_g] = _samples[g, :n_g]
                    comp_out[g, start:start + n_g] = k
                    fill_pos[g] += n_g

            assert (fill_pos == n_samples).all(), \
                f"Not all slots filled: fill_pos={fill_pos}"
            # for k in range(n_components):
            #     n_samp_k = n_samples_comp[:, k]
            #     n_samp_k_max = n_samp_k.max().item()
            #     if n_samp_k_max == 0:
            #         continue
            #     z = rng.randn(n_groups, n_samp_k_max, n_features)
            #     _samples = means[:, k][:, None, :] + self.xp.einsum(
            #         "ijk,imk->imj", chol_decomp[:, k], z
            #     )
            #     try:
            #         repeat_arg = list(n_samp_k.get())
            #     except AttributeError:
            #         repeat_arg = list(n_samp_k)

            #     tmp0 = self.xp.repeat(self.xp.arange(n_groups), repeat_arg)
            #     # _tmp = self.xp.tile(self.xp.arange(n_samp_k_max), (n_groups, 1))
            #     # tmp1 = _tmp[_tmp < n_samp_k[:, None]]

            #     # tmp_fill = samples_so_far.argmin(axis=-1)[tmp0] + tmp1

            #     # X[tmp0, tmp_fill] = _samples[tmp0, tmp1]
                
            #     # assert self.xp.all(~samples_so_far[tmp0, tmp_fill])
            #     # samples_so_far[tmp0, tmp_fill] = True
            #     # comp_out[tmp0, tmp_fill] = k
                
            #     tmp1_per_group = [self.xp.arange(n_samp_k[g].item()) for g in range(n_groups)]
            #     tmp1_local = self.xp.concatenate(tmp1_per_group)
                
            #     _tmp = self.xp.tile(self.xp.arange(n_samp_k_max), (n_groups, 1))
            #     tmp1_sample_idx = _tmp[_tmp < n_samp_k[:, None]]
                
            #     fill_offsets = samples_so_far.argmin(axis=-1)[tmp0] 
            #     tmp_fill = fill_offsets + tmp1_local  
                
            #     X[tmp0, tmp_fill] = _samples[tmp0, tmp1_sample_idx]
            #     assert self.xp.all(~samples_so_far[tmp0, tmp_fill])
            #     samples_so_far[tmp0, tmp_fill] = True
            #     comp_out[tmp0, tmp_fill] = k
                
            # try:
            #     assert self.xp.all(samples_so_far)
            # except:
            #     breakpoint()
            # X = self.xp.vstack(
            #     [
            #         rng.multivariate_normal(mean, covariance, int(sample))
            #         for (mean, covariance, sample) in zip(
            #             self.means_, self.covariances_, n_samples_comp
            #         )
            #     ]
            # )
        elif self.covariance_type == "tied":
            X = self.xp.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = self.xp.vstack(
                [
                    mean + rng.standard_normal(size=(sample, n_features)) * self.xp.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )

        # y = self.xp.concatenate(
        #     [self.xp.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        # )

        return (X, n_samples_comp, comp_out)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm (see :meth:`fit_predict`).

        The method fits the model ``n_init`` times and keeps the parameters
        of the last trial in which any group's lower bound improved (see
        :meth:`fit_predict`). Within each trial, EM iterates up to
        ``max_iter`` times, per group, until the lower-bound change is below
        ``tol``; if some group never converges a ``UserWarning`` is issued.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            y: Ignored (scikit-learn API).

        Returns:
            GaussianMixtureModel: ``self``, fitted.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y)
        return self

    def _print_verbose_msg_init_beg(self, n_init):
        """Log the start of initialization ``n_init`` when verbose (and start the timers)."""
        if self.verbose == 1:
            logger.info("Initialization %d" % n_init)
        elif self.verbose >= 2:
            logger.info("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None):
        """Fit the batched mixtures by EM and return the final responsibility argmax.

        Allocates the per-group parameter arrays, then for each of ``n_init``
        initializations runs EM, freezing each group once its lower-bound
        change is below ``tol``. A trial's parameters are kept when ANY group's
        lower bound improved on the best so far (all groups' parameters are
        then taken from that trial). Ends with :meth:`_set_parameters`
        (``store_all=True``, filling ``inv_covs_`` / ``det_covs_``) and a final
        E step. Issues a ``UserWarning`` if some group did not converge.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            y: Ignored (scikit-learn API).

        Returns:
            xp.ndarray: ``argmax`` of the final log responsibilities over axis
            1 (the SAMPLE axis of the ``(n_groups, n_samples, n_components)``
            array, so ``(n_groups, n_components)``; not per-sample labels).
            :meth:`fit` discards it.

        Raises:
            ValueError: ``n_samples < n_components``.
            NotImplementedError: An unsupported option (see the class
                docstring).
        """
        # X = validate_data(self, X, dtype=[self.xp.float64, self.xp.float32], ensure_min_samples=2)
        if X.shape[1] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[1]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        best_lower_bounds = []

        # random_state = check_random_state(self.random_state)
        random_state = self.random_state

        n_groups, n_samples, n_features = X.shape
        self.converged_ = self.xp.full(n_groups, False)
        self.weights_ = self.xp.zeros((n_groups, self.n_components))
        self.means_ = self.xp.zeros((n_groups, self.n_components, n_features))
        self.covariances_ = self.xp.zeros((n_groups, self.n_components, n_features, n_features))
        self.precisions_cholesky_ = self.xp.zeros(
            (n_groups, self.n_components, n_features, n_features)
        )
        max_lower_bound = self.xp.full(n_groups, -self.xp.inf)
        best_n_iter = self.xp.zeros(n_groups, dtype=int)
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound_tmp = -self.xp.inf if do_init else self.lower_bound_
            lower_bound = self.xp.full(n_groups, lower_bound_tmp)
            current_lower_bounds = []

            if self.max_iter == 0:
                raise NotImplementedError
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = self.xp.full(n_groups, False)
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound.copy()

                    log_prob_norm, log_resp = self._e_step(X, converged=converged)

                    self._m_step(X, log_resp, converged=converged)

                    lower_bound[~converged] = self._compute_lower_bound(log_resp, log_prob_norm)
                    current_lower_bounds.append(lower_bound)

                    change = lower_bound[~converged] - prev_lower_bound[~converged]
                    self._print_verbose_msg_iter_end(n_iter, change)

                    conv = self.xp.abs(change) < self.tol
                    change_converge_status = self.xp.arange(len(converged))[~converged][conv]
                    best_n_iter[~converged] = n_iter

                    converged[change_converge_status] = True
                    if self.xp.all(converged):
                        break

                self._print_verbose_msg_init_end(lower_bound, converged)

                adjust = (lower_bound > max_lower_bound) | (max_lower_bound == -self.xp.inf)
                if self.xp.any(adjust):
                    inds_adjust = self.xp.arange(converged.shape[0])[adjust]

                    # if lower_bound > max_lower_bound or max_lower_bound == -self.xp.inf:
                    max_lower_bound[inds_adjust] = lower_bound[inds_adjust]
                    best_params = self._get_parameters()
                    best_lower_bounds = current_lower_bounds
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.xp.all(self.converged_) and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                # ConvergenceWarning,
            )

        self._set_parameters(best_params, store_all=True)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.lower_bounds_ = best_lower_bounds

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X, converged=None):
        """Run the E step on the non-converged groups.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            converged (xp.ndarray, optional): ``(n_groups,)`` bool; default all
                ``False``.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(mean_log_prob, log_resp)``: the
            per-group mean of ``log p(x)``, ``(n_active,)``, and the log
            responsibilities ``(n_active, n_samples, n_components)``.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, converged=converged)
        return self.xp.mean(log_prob_norm, axis=-1), log_resp

    def _m_step(self, X, log_resp, converged=None):
        """Run the M step in place on the non-converged groups.

        Updates ``weights_`` (normalized per group), ``means_``,
        ``covariances_`` and ``precisions_cholesky_`` of the active groups.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.
            log_resp (xp.ndarray): Log responsibilities of the active groups,
                ``(n_active, n_samples, n_components)``.
            converged (xp.ndarray, optional): ``(n_groups,)`` bool; default all
                ``False``.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        (
            self.weights_[~converged],
            self.means_[~converged],
            self.covariances_[~converged],
        ) = _estimate_gaussian_parameters(
            X[~converged],
            self.xp.exp(log_resp),
            self.reg_covar,
            self.covariance_type,
            xp=self.xp,
        )
        self.weights_[~converged] /= self.weights_[~converged].sum(axis=-1)[:, None]
        self.precisions_cholesky_[~converged] = _compute_precision_cholesky(
            self.covariances_[~converged], self.covariance_type, xp=self.xp
        )

    def _estimate_log_prob(self, X, converged=None):
        """Per-component log-probabilities for non-converged groups only."""
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        return _estimate_log_gaussian_prob(
            X[~converged],
            self.means_[~converged],
            self.precisions_cholesky_[~converged],
            self.covariance_type,
            xp=self.xp,
        )

    def _estimate_log_weights(self, converged=None):
        """Log of mixture weights for non-converged groups only."""
        if converged is None:
            converged = self.xp.full(self.weights_.shape[0], False)

        return self.xp.log(self.weights_[~converged])

    def _compute_lower_bound(self, _, log_prob_norm):
        """Return the EM lower bound: the per-group mean ``log p(x)`` from :meth:`_e_step`."""
        return log_prob_norm

    def _estimate_weighted_log_prob_for_flat(self, X):
        """Return each group's weighted log-density of ``X`` with flat-mixture weights.

        The weights are renormalized over ALL groups' components, so the
        log-sum-exp over groups of the result (as :meth:`GMMFit.logpdf` does
        with ``flat=True``, after adding each group's log-Jacobian) is the
        log-density of the one concatenated mixture. Must be called after
        :meth:`fit`.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``, each group
                holding the same points mapped into that group's cube.

        Returns:
            xp.ndarray: ``(n_groups, n_samples)``.
        """
        # must be converged already
        logpdf_gaussian = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type, xp=self.xp
        )

        weights = self.weights_ / self.weights_.sum()
        log_weights = self.xp.log(weights)
        logpdf_all_components = logpdf_gaussian + log_weights[:, None, :]
        logpdf = self.logsumexp(logpdf_all_components, axis=-1)
        return logpdf

    def _get_parameters(self):
        """Snapshot of ``(weights_, means_, covariances_, precisions_cholesky_)``."""
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params, store_all=False):
        """Set ``(weights_, means_, covariances_, precisions_cholesky_)`` and derived arrays.

        Recomputes ``precisions_``; with ``store_all`` also ``inv_covs_`` and
        ``det_covs_`` (used by :func:`vec_fit_gmm_min_bic`).

        Args:
            params (tuple): As returned by :meth:`_get_parameters`.
            store_all (bool, optional): Also store the covariance inverses and
                determinants. Default ``False``.
        """
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        n_groups, _, n_features = self.means_.shape

        dtype = self.precisions_cholesky_.dtype
        if self.covariance_type == "full":
            self.precisions_ = self.xp.empty_like(self.precisions_cholesky_)
            # for k, prec_chol in enumerate(self.precisions_cholesky_):
            for k in range(self.precisions_cholesky_.shape[1]):
                prec_chol = self.precisions_cholesky_[:, k]
                self.precisions_[:, k] = self.xp.einsum("ijk,ilk->ijl", prec_chol, prec_chol)
                # self.precisions_[k] = self.xp.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            raise NotImplementedError
            self.precisions_ = self.xp.dot(self.precisions_cholesky_, self.precisions_cholesky_.T)
        else:
            raise NotImplementedError
            self.precisions_ = self.precisions_cholesky_**2

        if store_all:
            self.inv_covs_ = self.xp.linalg.inv(self.covariances_)
            self.det_covs_ = self.xp.linalg.det(self.covariances_)

    def _n_parameters(self):
        """Return the number of free parameters of ONE group's mixture (used by :meth:`bic`)."""
        n_groups, _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Return the Bayesian information criterion of each group's mixture on ``X``.

        ``BIC = -2 n mean_log_p + n_params ln n`` with ``n = n_samples``.

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.

        Returns:
            xp.ndarray: ``(n_groups,)`` BIC values (lower is better).
        """
        return -2 * self.score(X) * X.shape[1] + self._n_parameters() * np.log(X.shape[1])

    # TODO check: keep/remove/document -- no callers (cross-check helper; only commented-out uses)
    def general_way_logpdf(self, X, flat=True):
        """Reference log-PDF computed directly from :math:`(x-\\mu)^T \\Sigma^{-1} (x-\\mu)`.

        Provided as a cross-check of :meth:`score_samples`; uses the cached
        ``inv_covs_`` and ``det_covs_`` attributes set when ``store_all`` is
        used in :meth:`_set_parameters`.

        Args:
            X: Array of shape ``(n_groups, n_samples, n_features)``.
            flat: If ``True``, normalize the weights across the full
                concatenated mixture rather than per group.

        Returns:
            Array of log-PDF values of shape ``(n_groups, n_samples)``.
        """

        n_groups, n_components, n_features, _ = self.inv_covs_.shape

        x_minus_mu = X[:, :, None, :] - self.means_[:, None, :, :]
        _tmp1 = self.xp.einsum("ijkl,iklm->ijkm", x_minus_mu, self.inv_covs_)
        kernel = self.xp.einsum("ijkl,ijkl->ijk", _tmp1, x_minus_mu)

        if flat:
            weights = self.weights_ / self.weights_.sum()
        else:
            weights = self.weights_

        _logpdf = (
            self.xp.log(weights)[:, None, :]
            - (n_features / 2.0) * self.xp.log(2 * np.pi)
            - (1.0 / 2.0) * self.xp.log(self.det_covs_)[:, None, :]
            - (1.0 / 2.0) * kernel
        )
        logpdf = self.logsumexp(_logpdf, axis=-1)
        return logpdf

    # TODO check: keep/remove/document -- no callers; raises NotImplementedError
    def aic(self, X):
        """Akaike information criterion (not implemented in the batched fork).

        Args:
            X (xp.ndarray): ``(n_groups, n_samples, n_features)``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()


class GMMFit:
    """Fit a vectorized GMM to per-group samples mapped onto the unit cube.

    For each group the per-feature ``min`` / ``max`` of the input samples is
    used to map the data linearly into :math:`[-1, 1]^d` before fitting a
    :class:`GaussianMixtureModel`. The Jacobian of this mapping is precomputed
    so that :meth:`logpdf` returns log-densities in the original coordinates.
    The fit runs at construction, with ``covariance_type="full"``,
    ``tol=1e-3``, ``reg_covar=1e-6``, ``max_iter=1000``, ``n_init=1`` and
    ``init_params="random_from_data"``.

    Args:
        samples_in (xp.ndarray): Array of shape ``(n_groups, n_samples, n_features)``
            to be fit; numpy when ``gpu`` is ``None``, cupy otherwise. Every
            feature must have a nonzero range in every group. Required despite
            the ``None`` default.
        n_components (int, optional): Number of mixture components per group.
            Default ``30``.
        gpu (int, optional): GPU device index to use; ``None`` (default)
            selects CPU.
        random_state (int, optional): Seed for the EM initialization; ``None``
            (default) is unseeded.

    Raises:
        AssertionError: ``samples_in`` is ``None`` or not an array of the
            selected array module.

    Attributes:
        gmm (GaussianMixtureModel): The fitted mixture (cube coordinates).
        sample_mins (xp.ndarray): ``(n_groups, n_features)`` box lower bounds.
        sample_maxs (xp.ndarray): ``(n_groups, n_features)`` box upper bounds.
        log_det_J (xp.ndarray): ``(n_groups,)`` log-Jacobian of the cube map.
        n_groups (int): Number of groups.
        ndim (int): Number of features.

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.gmm import GMMFit
        >>> samples = np.random.default_rng(1).normal(size=(2, 400, 3))
        >>> fit = GMMFit(samples, n_components=2, random_state=1)
        >>> fit.logpdf(samples).shape, fit.rvs(10).shape
        ((2, 400), (2, 10, 3))
    """

    @property
    def xp(self):
        """Array module (``numpy`` or ``cupy``) used for the computations."""
        if self.use_gpu:
            return cp
        else:
            return np

    def __init__(self, samples_in=None, n_components=30, gpu=None,
                 random_state=None):

        self.gpu = gpu
        if gpu is not None:
            self.use_gpu = True
            self.xp.cuda.runtime.setDevice(gpu)

        else:
            self.use_gpu = False

        self.fitted = False
        run = True
        min_bic = self.xp.inf

        self.gmm_init_kwargs = dict(
            n_components=n_components,
            covariance_type="full",
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=1000,
            n_init=1,
            init_params="random_from_data",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            # Seedable (2026-09-18). ``None`` keeps the historical unseeded
            # behaviour; a value makes a refit of the same data reproducible,
            # which it was not -- ``init_params="random_from_data"`` draws
            # fresh starting points every call, so the same store refitted
            # twice gave different component counts AND different components.
            random_state=random_state,
            warm_start=False,
            verbose=0,
            verbose_interval=10,
            gpu=self.gpu,
        )
        self.gmm = GaussianMixtureModel(**self.gmm_init_kwargs)

        assert samples_in is not None

        assert isinstance(samples_in, self.xp.ndarray)
        self.sample_mins = sample_mins = samples_in.min(axis=1)
        self.sample_maxs = sample_maxs = samples_in.max(axis=1)
        self.n_groups = samples_in.shape[0]
        samples = self.transform_to_gmm_basis(samples_in)
        self.ndim = samples.shape[-1]

        self.gmm.fit(samples)
        self.log_det_J = self.ndim * self.xp.log(2) - self.xp.sum(
            self.xp.log(self.sample_maxs - self.sample_mins), axis=-1
        )
        self.fitted = True

    def transform_to_gmm_basis(self, samples):
        """Map physical samples into each group's :math:`[-1, 1]` cube.

        Args:
            samples (xp.ndarray): ``(n_groups, n, n_features)``, or
                ``(n, n_features)`` (then mapped with group 0's box only).

        Returns:
            xp.ndarray: Same shape as ``samples``, in cube coordinates.
        """
        squeeze = samples.ndim == 2
        if squeeze:
            samples = samples[None, :]

        maxs = self.sample_maxs[:, None, :]
        mins = self.sample_mins[:, None, :]

        tmp = ((samples - mins) / (maxs - mins)) * 2 - 1
        if squeeze:
            return tmp[0]
        else:
            return tmp

    def transform_from_gmm_basis(self, samples, inds_component=None):
        """Map :math:`[-1, 1]` unit-cube samples back into physical units.

        Args:
            samples (xp.ndarray): Samples in the unit-cube basis,
                ``(n_groups, n, n_features)``, or ``(n, n_features)`` (then,
                without ``inds_component``, mapped with group 0's box only).
            inds_component (xp.ndarray, optional): 1D array of length ``n``
                selecting which group's ``(mins, maxs)`` to use per sample
                (used by the flat mixture path).

        Returns:
            xp.ndarray: Same shape as ``samples``, in physical coordinates.
        """
        squeeze = samples.ndim == 2
        if squeeze:
            samples = samples[None, :]
        if inds_component is None:
            maxs = self.sample_maxs[:, None, :]
            mins = self.sample_mins[:, None, :]

        else:
            assert inds_component.ndim == 1
            assert inds_component.shape[0] == samples.shape[1]
            assert inds_component.min() >= 0
            assert inds_component.max() <= self.sample_maxs.shape[0]
            maxs = self.sample_maxs[inds_component][None, :]
            mins = self.sample_mins[inds_component][None, :]

        tmp = (samples + 1.0) / 2.0 * (maxs - mins) + mins
        if squeeze:
            return tmp[0]
        else:
            return tmp

    def rvs(self, size=(1,), flat=False):
        """Draw samples from the fitted mixture in physical coordinates.

        Args:
            size (int or tuple, optional): Number of draws (``prod(size)``; per
                group unless ``flat`` is set). Default ``(1,)``.
            flat (bool, optional): If ``True``, sample from the one mixture of
                all groups' components (weights renormalized over all of them)
                rather than per group. Default ``False``.

        Returns:
            xp.ndarray: Samples in physical coordinates,
            ``(n_groups, prod(size), n_features)``, or ``(prod(size), n_features)``
            when ``flat``.
        """
        assert self.fitted
        if isinstance(size, int):
            size = (size,)

        _samples, _comp_counts, _comps = self.gmm.sample(np.prod(size), flat=flat)
        if not flat:
            inds_component = None
            squeeze = False
        else:
            # combined GMM
            assert _comps.shape[0] == 1 and _samples.shape[0] == 1
            inds_component = _comps[0] // self.gmm.n_components
            squeeze = True

        samples = self.transform_from_gmm_basis(_samples, inds_component=inds_component)
        if squeeze:
            return samples[0]
        return samples

    def logpdf(self, x, flat=False):
        """Evaluate the log-PDF of the fitted mixture in physical coordinates.

        Args:
            x (xp.ndarray): Sample locations, ``(n_groups, n, n_features)``
                (group ``g`` scored with mixture ``g``); ``(n, n_features)``
                when ``flat``.
            flat (bool, optional): If ``True``, treat the union of
                group-mixtures as a single distribution and return one
                log-density per row of ``x``. Default ``False``.

        Returns:
            xp.ndarray: ``(n_groups, n)`` log-PDF values (``(n,)`` when
            ``flat``), including the log-Jacobian of the unit-cube mapping.
        """

        if not flat:
            x_in = self.transform_to_gmm_basis(x)
            _logpdf = self.gmm.logpdf(x_in)

        else:
            assert x.ndim == 2
            _x = self.xp.tile(x, (self.sample_maxs.shape[0], 1, 1))
            x_in = self.transform_to_gmm_basis(_x)
            _logpdf = self.gmm._estimate_weighted_log_prob_for_flat(x_in)
            # tmp = self.gmm.general_way_logpdf(x_in, flat=flat)

        # np.save("tmp", x_in)
        # np.save("logpdf", _logpdf)
        # print(x_in, _logpdf)
        _logpdf += self.log_det_J[
            :,
            None,
        ]
        # tmp += self.log_det_J[:, None, ]
        if flat:
            logpdf = self.gmm.logsumexp(_logpdf, axis=0)
            # tmp2 = self.gmm.logsumexp(tmp, axis=0)

        else:
            logpdf = _logpdf

        return logpdf

    def bic(self, x, flat=False):
        """Bayesian Information Criterion of the fitted mixture at ``x``.

        Args:
            x (xp.ndarray): Physical sample locations,
                ``(n_groups, n, n_features)`` (``(n, n_features)`` when ``flat``).
            flat (bool, optional): If ``True``, tile ``x`` over every group
                before mapping it into each cube. Default ``False``.

        Returns:
            xp.ndarray: ``(n_groups,)`` BIC values (cube coordinates; see
            :meth:`GaussianMixtureModel.bic`).
        """

        if not flat:
            x_in = self.transform_to_gmm_basis(x)

        else:
            assert x.ndim == 2
            _x = self.xp.tile(x, (self.sample_maxs.shape[0], 1, 1))
            x_in = self.transform_to_gmm_basis(_x)

        bic = self.gmm.bic(x_in)
        return bic

    # def combine_into_single_gmm(self):

    #     new_weights = self.gmm.weights_.reshape(1, -1) / self.gmm.weights_.sum()
    #     new_means = self.gmm.means_.reshape((1, -1) + self.gmm.means_.shape[-1:])
    #     new_covariances = self.gmm.covariances_.reshape((1, -1) + self.gmm.covariances_.shape[-2:])
    #     new_precisions_cholesky = self.gmm.precisions_cholesky_.reshape((1, -1) + self.gmm.precisions_cholesky_.shape[-2:])

    #     init_info = dict(
    #         weights_=new_weights,
    #         means_=new_means,
    #         covariances_=new_covariances,
    #         precisions_cholesky_=new_precisions_cholesky,
    #         sample_mins=self.sample_mins,
    #         sample_maxs=self.sample_maxs,
    #         log_det_J=self.log_det_J
    #     )
    #     new_gmm = GMMFit(n_components=new_weights.shape[-1], init_info=init_info, use_gpu=self.use_gpu)
    #     return new_gmm


def vec_fit_gmm_min_bic(
    samples,
    min_comp=1,
    max_comp=30,
    n_samp_bic_test=5000,
    gpu=None,
    verbose=False,
    return_components=False,
    random_state=None,
):
    """Fit per-group GMMs by sweeping component count and selecting min BIC.

    For each group, fits :class:`GMMFit` with increasing ``n_components`` and
    keeps the parameters at which the BIC (on the fitted samples) reaches its
    minimum. A group is declared converged once its BIC has come out above
    its running minimum twice (counted cumulatively), after which it is
    excluded from further fits; the sweep stops when every group has
    converged or at ``max_comp``.

    Args:
        samples (xp.ndarray): Array of shape ``(n_groups, n_samples, n_features)``
            (moved to the selected array module).
        min_comp (int, optional): Smallest number of components to try.
            Default ``1``.
        max_comp (int, optional): Largest number of components to try.
            Default ``30``.
        n_samp_bic_test (int, optional): UNUSED since 2026-09-18, kept so
            existing callers keep working. The BIC is now evaluated on the
            FITTED DATA. It used to be evaluated on ``n_samp_bic_test``
            synthetic draws from each fit, which is an entropy estimate rather
            than a goodness-of-fit: it decreases monotonically with the
            component count, so nothing was ever selected and every group
            returned ``max_comp``.
        gpu (int, optional): GPU device index to use (cupy, and
            ``cupy.cuda.runtime.setDevice(gpu)``); ``None`` (default) selects CPU.
        verbose (bool, optional): If ``True``, log convergence progress per
            component count. Default ``False``.
        return_components (bool, optional): If ``True``, return the raw lists
            ``[weights, means, covs, invcovs, dets, mins, maxs]`` instead of
            a :class:`~lisatools.sampling.prior.FullGaussianMixtureModel`.
            Default ``False``.
        random_state (int, optional): Seed for the underlying EM. ``None``
            (default) keeps the historical unseeded behaviour; a value makes a
            refit of the same samples reproducible.

    Returns:
        list or FullGaussianMixtureModel: With ``return_components``, seven
        lists of length ``n_groups``: per-group weights ``(K_g,)``, means
        ``(K_g, n_features)``, covariances, inverse covariances and
        determinants, all in that group's :math:`[-1, 1]` cube coordinates,
        plus the group's physical box ``mins`` / ``maxs`` ``(n_features,)``.
        Otherwise a :class:`~lisatools.sampling.prior.FullGaussianMixtureModel`
        built from them (``use_cupy`` when ``gpu`` is set).

    Example:
        .. code-block:: python

            from lisatools.sampling.gmm import vec_fit_gmm_min_bic

            # samples: (n_bands, n_samples, 6) GB posterior draws, physical columns
            full_gmm = vec_fit_gmm_min_bic(samples, min_comp=1, max_comp=30, gpu=0)
            logp = full_gmm.logpdf(points)  # points: (n, 6)
    """

    if gpu is not None:
        use_gpu = True
        xp = cp
        xp.cuda.runtime.setDevice(gpu)
    else:
        use_gpu = False
        xp = np

    n_groups, n_samples, ndim = samples.shape
    samples = xp.asarray(samples)

    converged = xp.zeros(n_groups, dtype=bool)
    counts_above_min_bic = xp.zeros(n_groups, dtype=int)
    min_bic = xp.full(n_groups, +np.inf)
    minimum_bic_i = xp.zeros(samples.shape[0], dtype=int)
    weights = []
    means = []
    covs = []
    inv_covs = []
    dets = []
    maxs = []
    mins = []
    final_gmm_info = [None for _ in range(n_groups)]
    for comp_i in range(min_comp, max_comp + 1):
        if xp.all(converged):
            break

        inds_here = xp.arange(n_groups)[~converged]
        samples_here = samples[inds_here]
        gmm = GMMFit(samples_here, n_components=comp_i, gpu=gpu,
                     random_state=random_state)
        # SCORE THE CRITERION ON THE DATA (2026-09-18 fix). This previously
        # read
        #     samp = gmm.rvs(n_samp_bic_test); bic = gmm.bic(samp)
        # i.e. it scored the model against its OWN synthetic draws. That is
        # an entropy estimate, not a goodness-of-fit: it falls monotonically
        # as components are added, so the criterion had no minimum, the
        # "risen twice past the running minimum" retirement below could never
        # fire, and EVERY group ran to ``max_comp`` and returned the cap.
        # Measured on a clean unimodal 9-D Gaussian: 30556 at K=1 decreasing
        # to 22255 at K=7, with and without resampling. ``GMMFit.bic`` takes
        # PHYSICAL coordinates and maps them to the unit cube itself, so the
        # fitted input is exactly what it wants.
        bic = gmm.bic(samples_here)

        above_min_bic = bic > min_bic[inds_here]
        below_min_bic = bic < min_bic[inds_here]

        counts_above_min_bic[inds_here] += (above_min_bic).astype(int)
        # print(i, (bic < min_bic).sum() / bic.shape[0])
        update_main_with_min_bic = inds_here[below_min_bic]
        update_here_with_min_bic = xp.arange(bic.shape[0])[below_min_bic]
        minimum_bic_i[update_main_with_min_bic] = comp_i
        min_bic[update_main_with_min_bic] = bic[below_min_bic]

        for main_update_i, here_update_i in zip(update_main_with_min_bic, update_here_with_min_bic):
            main_update_i = main_update_i.item()
            here_update_i = here_update_i.item()
            _weights = gmm.gmm.weights_[here_update_i]
            _means = gmm.gmm.means_[here_update_i]
            _covs = gmm.gmm.covariances_[here_update_i]
            _inv_covs = gmm.gmm.inv_covs_[here_update_i]
            _dets = gmm.gmm.det_covs_[here_update_i]
            _mins = gmm.sample_mins[here_update_i]
            _maxs = gmm.sample_maxs[here_update_i]

            final_gmm_info[main_update_i] = [
                _weights,
                _means,
                _covs,
                _inv_covs,
                _dets,
                _mins,
                _maxs,
            ]

        converged = counts_above_min_bic >= 2
        converged_here = converged[inds_here]

        del gmm
        if use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            logger.info(f"{comp_i} components: {converged.sum()} converged out of {converged.shape[0]}.")

    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()

    for tmp, n_comp in zip(final_gmm_info, minimum_bic_i):
        assert tmp[0].shape[0] == n_comp

    weights = [tmp[0] for tmp in final_gmm_info]
    means = [tmp[1] for tmp in final_gmm_info]
    covs = [tmp[2] for tmp in final_gmm_info]
    invcovs = [tmp[3] for tmp in final_gmm_info]
    dets = [tmp[4] for tmp in final_gmm_info]
    mins = [tmp[5] for tmp in final_gmm_info]
    maxs = [tmp[6] for tmp in final_gmm_info]

    if return_components:
        return [weights, means, covs, invcovs, dets, mins, maxs]

    # import pickle
    # with open("gmm_tmp_2.pickle", "wb") as fp:
    #     pickle.dump((weights, means, covs, invcovs, dets, mins, maxs), fp, pickle.HIGHEST_PROTOCOL)

    full_gmm = FullGaussianMixtureModel(
        weights, means, covs, invcovs, dets, mins, maxs, use_cupy=use_gpu
    )
    return full_gmm


def fit_gb_gmm_rj_container(
    samples,
    use_chirp_mass: bool = False,
    use_cupy: bool = True,
    gpu=None,
    min_comp: int = 1,
    max_comp: int = 30,
    n_samp_bic_test: int = 5000,
    verbose: bool = False,
    fdot_astro_ratio_max: typing.Optional[float] = None,
    use_distance: bool = False,
):
    """Per-group GB samples -> batched GMM -> 8/9-column RJ birth container.

    The standalone "samples -> batched GMM container" entry point: the
    guard checks + :func:`vec_fit_gmm_min_bic` fit + eryn container assembly
    that the GB serial-search move used to drive inline through its
    ``ParaEnsembleSampler`` chains. Any (n_groups, n_samples, 6) sample
    block works -- per-band search chains, per-leaf PE chains, grid draws.

    The only caller is
    :class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJSerialSearchMCMC`
    (the ``rj_fstat_mcmc_search`` / ``rj_fstat_mcmc`` moves), which the 9mo /
    12mo recipes build but never stage, so the production runs do not call it.

    Args:
        samples (array-like): ``(n_groups, n_samples, 6)`` in the NON-PHASE columns of
            the GB sampling basis -- ``(lnA, f0 [mHz], fdot|Mc, cos_iota,
            alpha, sin_delta)``, i.e. columns ``[0, 1, 2, 4, 6, 7]`` of the
            8-column sampler layout (column 0 is the distance in kpc in the
            distance basis).
        use_chirp_mass: Column 2's meaning: ``False`` -> legacy ``fdot``
            basis, ``True`` -> chirp-mass basis (``Mc`` slot), matching the
            run's ``GBSettings.use_chirp_mass``. Default ``False``.
        use_cupy: Array module of the returned container (match the run
            backend). Default ``True``.
        gpu (int, optional): GPU device index for the fit (``None`` -> CPU).
        min_comp: Forwarded to :func:`vec_fit_gmm_min_bic`. Default ``1``.
        max_comp: Forwarded. Default ``30``.
        n_samp_bic_test: Forwarded (unused there). Default ``5000``.
        verbose: Forwarded. Default ``False``.
        fdot_astro_ratio_max: When not ``None``, append a 9th
            ``fdot_astro_ratio`` column drawn from ``U[-M, M]`` (the
            fdot_astro ratio basis; the GMM still fits only the 6 intrinsic
            columns, births draw the ratio from its prior). ``None`` ->
            8-column container.
        use_distance: Column 0 is the luminosity distance (``"dist"`` key)
            instead of ``"A"``. Default ``False``.

    Returns:
        eryn.priors.ProbDistContainer: Container whose ``rvs``/``logpdf``
        columns follow the sampler layout ``[A|dist, f0, fdot|Mc, phi0,
        cos_iota, psi, alpha, sin_delta(, fdot_astro_ratio)]``, with ``phi0``
        uniform on ``[0, 2 pi)`` and ``psi`` on ``[0, pi)``
        (``reset_key_order`` re-maps them -- a bare ``key_order = [...]``
        assignment would NOT).

    Raises:
        ValueError: On a wrong sample shape, non-finite samples or degenerate
            (zero-range) features -- the failure modes the caller should
            log-and-skip.

    Example:
        .. code-block:: python

            from lisatools.sampling.gmm import fit_gb_gmm_rj_container

            # samples: (n_bands, n_samples, 6) non-phase GB columns
            rj_dist = fit_gb_gmm_rj_container(samples, use_chirp_mass=True, gpu=0)
            births = rj_dist.rvs(size=100)  # (100, 8), sampler column order
    """
    from eryn.priors import ProbDistContainer, UniformDistribution

    xp_here = cp if gpu is not None else np
    samples = xp_here.asarray(samples)
    if samples.ndim != 3 or samples.shape[-1] != 6:
        raise ValueError(
            f"expected (n_groups, n_samples, 6) samples; got {samples.shape}"
        )
    n_nan = int(xp_here.isnan(samples).sum())
    n_inf = int(xp_here.isinf(samples).sum())
    if n_nan or n_inf:
        raise ValueError(
            f"samples contain NaN/Inf before GMM fitting (NaN: {n_nan}, "
            f"Inf: {n_inf})"
        )
    ranges = samples.max(axis=1) - samples.min(axis=1)
    if bool((ranges == 0).any()):
        bad_groups, bad_feats = xp_here.where(ranges == 0)
        raise ValueError(
            f"degenerate (zero-range) features in groups {bad_groups} for "
            f"features {bad_feats}; the [-1, 1] GMM basis map would produce "
            "NaN"
        )

    full_gmm = vec_fit_gmm_min_bic(
        samples,
        min_comp=min_comp,
        max_comp=max_comp,
        n_samp_bic_test=n_samp_bic_test,
        gpu=gpu,
        verbose=verbose,
    )

    third = "Mc" if use_chirp_mass else "fdot"
    # slot 0 is the luminosity distance (kpc) in the distance basis, else lnA;
    # the 6-D GMM fits whatever column 0 holds either way.
    first = "dist" if use_distance else "A"
    priors_in = {
        (first, "f0", third, "cos_iota", "alpha", "sin_delta"): full_gmm,
        "phi0": UniformDistribution(0.0, 2.0 * np.pi),
        "psi": UniformDistribution(0.0, np.pi),
    }
    key_order = [first, "f0", third, "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    if fdot_astro_ratio_max is not None:
        # 9th column: births draw the fdot_astro ratio from its U[-M, M]
        # prior (the GMM did not fit it -- degenerate with Mc under F-stat).
        M = float(fdot_astro_ratio_max)
        priors_in["fdot_astro_ratio"] = UniformDistribution(-M, M)
        key_order.append("fdot_astro_ratio")
    rj_dist = ProbDistContainer(priors_in, use_cupy=use_cupy)
    rj_dist.reset_key_order(key_order)
    return rj_dist


if __name__ == "__main__":
    samples_tmp = np.load("samples_examples.npy")[:, :, :, np.array([0, 1, 2, 4, 6, 7])]  # [:1]
    samples_tmp = samples_tmp.reshape(samples_tmp.shape[0], -1, samples_tmp.shape[-1])
    full_gmm = vec_fit_gmm_min_bic(samples_tmp, use_gpu=True, verbose=True)
    samp_gen = full_gmm.rvs(size=(int(1e6),))
    samp_logpdf = full_gmm.logpdf(samp_gen)
    import matplotlib.pyplot as plt

    plt.scatter(asnumpy(samp_gen[:, 1]), asnumpy(samp_gen[:, 0]))
    plt.savefig("check0.png")
    breakpoint()

    # sample_mins = samples_tmp.min(axis=1)
    # sample_maxs = samples_tmp.max(axis=1)
    # samples2 = ((samples_tmp - sample_mins[:, None, :]) / (sample_maxs[:, None, :] - sample_mins[:, None, :])) * 2.0 - 1.0

    # # gmm = GaussianMixtureModel(
    # #     n_components=25,
    # #     covariance_type="full",
    # #     tol=1e-3,
    # #     reg_covar=1e-6,
    # #     max_iter=100,
    # #     n_init=1,
    # #     init_params="random_from_data",
    # #     weights_init=None,
    # #     means_init=None,
    # #     precisions_init=None,
    # #     random_state=None,
    # #     warm_start=False,
    # #     verbose=0,
    # #     verbose_interval=10,
    # # )
    # check2 = full_gmm.logpdf(out2)
    # breakpoint()
    # print(f"max diff: {np.abs(check2 - tmp2).max()}")
    # breakpoint()
    # from sklearn.mixture import GaussianMixture

    # gmm2 = GaussianMixture(
    #     n_components=30,
    #     covariance_type="full",
    #     tol=1e-3,
    #     reg_covar=1e-6,
    #     max_iter=1000,
    #     n_init=1,
    #     init_params="random_from_data",
    #     weights_init=None,
    #     means_init=None,
    #     precisions_init=None,
    #     random_state=None,
    #     warm_start=False,
    #     verbose=0,
    #     verbose_interval=10,
    # )

    # gmm2.fit(samples2[0])
    # # gmm.fit(samples)
    # samp = gmm2.sample(1)
    # samp_pdf = gmm2.score_samples(samp[0])
    # breakpoint()

    # check1 = gmm.logpdf(samp[0])
    # check2 = logsumexp_cpu(gmm2.predict_proba(samp[0][0]), axis=-1)
    # breakpoint()
