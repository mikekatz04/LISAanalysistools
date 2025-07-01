"""Base class for mixture models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

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

from lisatools.utils.utility import searchsorted2d_vec
  
# from .. import cluster
# from ..base import BaseEstimator, DensityMixin, _fit_context
# from ..cluster import kmeans_plusplus
# from ..exceptions import ConvergenceWarning
# from ..utils import check_random_state
# from ..utils._param_validation import Interval, StrOptions
# from ..utils.validation import check_is_fitted, validate_data
from lisatools.sampling.prior import FullGaussianMixtureModel

def _check_shape(param, param_shape, name, xp=None):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : str
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
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
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
        covariances[:, k] = xp.einsum("ijk,ijl->ikl", (resp[:, :, k][:, :, None] * diff), diff) / nk[:, k][:, None, None]
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
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
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
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
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
                cov_chol = xp.linalg.cholesky(covariance)  #, lower=True)
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
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if xp is None:
        xp = np

    if covariance_type == "full":
        n_groups, n_components, _, _ = matrix_chol.shape
        log_det_chol = xp.sum(xp.log(matrix_chol.reshape(n_groups, n_components, -1)[:, :, :: n_features + 1]), axis=-1)

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
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
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
            y = xp.einsum("ijk,ikl->ijl", X, prec_chol) - xp.einsum("ik,ikl->il", mu, prec_chol)[:, None, :]
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
    if xp is None:
        xp = np

    n_groups, n_components = weights.shape
    weights_tmp = xp.cumsum(weights, axis=-1)
    cumulative_weights = xp.concatenate([xp.zeros((n_groups, 1)), xp.cumsum(weights, axis=-1)], axis=1)
    draw = random_state.rand(n_groups, n_samples)
    extra_kwargs = {}
    try:
        extra_kwargs["gpu"] = xp.cuda.runtime.getDevice()
    except AttributeError:
        pass

    component = (
        searchsorted2d_vec(cumulative_weights, draw, side="right", xp=xp, **extra_kwargs) - 1
    ).reshape(draw.shape)

    component_tmp = (n_components + 1) * xp.repeat(xp.arange(n_groups), n_samples).reshape(n_groups, n_samples)
    counts = xp.zeros((n_groups, n_components), dtype=int)
    uni, uni_counts = xp.unique((component + component_tmp).flatten(), return_counts=True)
    group = uni // (n_components + 1)
    comp = uni % (n_components + 1)
    counts[group, comp] = uni_counts
    return counts


class GaussianMixtureModel:
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each component has its own diagonal covariance matrix.
        - 'spherical': each component has its own single variance.

        For an example of using `covariance_type`, refer to
        :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py`.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

        For an example of using covariances, refer to
        :ref:`sphx_glr_auto_examples_mixture_plot_gmm_covariances.py`.

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence of the best fit of EM was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    lower_bounds_ : array-like of shape (`n_iter_`,)
        The list of lower bound values on the log-likelihood from each
        iteration of the best fit of EM.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])

    For a comparison of Gaussian Mixture with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
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
        self.reg_covar=reg_covar
        self.max_iter=max_iter
        self.n_init=n_init
        self.init_params=init_params
        if random_state is None:
            random_state = self.xp.random

        self.random_state=random_state
        self.warm_start=warm_start
        self.verbose=verbose
        self.verbose_interval=verbose_interval

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        

    @property
    def xp(self):
        if self.use_gpu:
            return cp
        else:
            return np
    
    @property
    def logsumexp(self):
        if self.use_gpu:
            return logsumexp_gpu
        else:
            return logsumexp_cpu

    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        """Print verbose message on the end of iteration."""
        converged_msg = f"converged {init_has_converged.sum()} out of {init_has_converged.shape[0]}."
        if self.verbose == 1:
            print(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            print(
                f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound"
                f" {lb:.5f}."
            )

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, _, n_features = X.shape

        if self.weights_init is not None:
            raise NotImplementedError
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            raise NotImplementedError
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:

            raise NotImplementedError
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _initialize_parameters(self, X, random_state):
        # If all the initial parameters are all provided, then there is no need to run
        # the initialization.
        compute_resp = (
            self.weights_init is None
            or self.means_init is None
            or self.precisions_init is None
        )
        if compute_resp:
            # super()._initialize_parameters(X, random_state)
            self._initialize_parameters_inner(X, random_state)
        else:
            self._initialize(X, None)
    
    def _initialize_parameters_inner(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_groups, n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_groups, n_samples, _ = X.shape

        if self.init_params == "kmeans":
            raise NotImplementedError
            resp = self.xp.zeros((n_samples, self.n_components), dtype=X.dtype)
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[self.xp.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = self.xp.asarray(
                random_state.uniform(size=(n_groups, n_samples, self.n_components)), dtype=X.dtype
            )
            # resp /= resp.sum(axis=1)[:, self.xp.newaxis]
            resp /= resp.sum(axis=-11)[:, :, self.xp.newaxis]
        elif self.init_params == "random_from_data":
            resp = self.xp.zeros((n_groups, n_samples, self.n_components), dtype=X.dtype)
            indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
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
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
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
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        # check_is_fitted(self)
        # X = validate_data(self, X, reset=False)

        return self.logsumexp(self._estimate_weighted_log_prob(X), axis=-1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        return self.score_samples(X).mean(axis=-1)

    def logpdf(self, X):
        return self.score_samples(X)

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        # check_is_fitted(self)
        # X = validate_data(self, X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return self.xp.exp(log_resp)


    def _estimate_log_prob_resp(self, X, converged=None):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
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
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        return self._estimate_log_prob(X, converged=converged) + self._estimate_log_weights(converged=converged)[:, None, :]

    def sample(self, n_samples=1, flat=False):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
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
        n_features = means.shape[-1]
        X = self.xp.zeros((n_groups, n_samples, n_features))
        samples_so_far = self.xp.zeros((n_groups, n_samples), dtype=bool)
        comp_out = self.xp.zeros((n_groups, n_samples), dtype=int)
        chol_decomp = self.xp.linalg.cholesky(covariances)
        if self.covariance_type == "full":
            for k in range(n_components):
                n_samp_k = n_samples_comp[:, k]
                n_samp_k_max = n_samp_k.max().item()
                if n_samp_k_max == 0:
                    continue
                z = rng.randn(n_groups, n_samp_k_max, n_features)
                _samples = means[:, k][:, None, :] + self.xp.einsum("ijk,imk->imj", chol_decomp[:, k], z)
                try:
                    repeat_arg = list(n_samp_k.get())
                except AttributeError:
                    repeat_arg = list(n_samp_k)

                tmp0 = self.xp.repeat(self.xp.arange(n_groups), repeat_arg)
                _tmp = self.xp.tile(self.xp.arange(n_samp_k_max), (n_groups, 1))
                tmp1 = _tmp[_tmp < n_samp_k[:, None]]

                tmp_fill = samples_so_far.argmin(axis=-1)[tmp0] + tmp1

                X[tmp0, tmp_fill] = _samples[tmp0, tmp1]
                assert self.xp.all(~samples_so_far[tmp0, tmp_fill])
                samples_so_far[tmp0, tmp_fill] = True
                comp_out[tmp0, tmp_fill] = k

            assert self.xp.all(samples_so_far)
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
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * self.xp.sqrt(covariance)
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
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y)
        return self

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
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
        self.precisions_cholesky_ = self.xp.zeros((n_groups, self.n_components, n_features, n_features))
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
                ConvergenceWarning,
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
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, converged=converged)
        return self.xp.mean(log_prob_norm, axis=-1), log_resp

    def _m_step(self, X, log_resp, converged=None):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        self.weights_[~converged], self.means_[~converged], self.covariances_[~converged] = _estimate_gaussian_parameters(
            X[~converged], self.xp.exp(log_resp), self.reg_covar, self.covariance_type, xp=self.xp
        )
        self.weights_[~converged] /= self.weights_[~converged].sum(axis=-1)[:, None]
        self.precisions_cholesky_[~converged] = _compute_precision_cholesky(
            self.covariances_[~converged], self.covariance_type, xp=self.xp
        )

    def _estimate_log_prob(self, X, converged=None):
        if converged is None:
            converged = self.xp.full(X.shape[0], False)

        return _estimate_log_gaussian_prob(
            X[~converged], self.means_[~converged], self.precisions_cholesky_[~converged], self.covariance_type, xp=self.xp
        )

    def _estimate_log_weights(self, converged=None):
        if converged is None:
            converged = self.xp.full(self.weights_.shape[0], False)

        return self.xp.log(self.weights_[~converged])

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _estimate_weighted_log_prob_for_flat(self, X):
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
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params, store_all=False):
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
            self.precisions_ = self.xp.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            raise NotImplementedError
            self.precisions_ = self.precisions_cholesky_**2

        if store_all:
            self.inv_covs_ = self.xp.linalg.inv(self.covariances_)
            self.det_covs_ = self.xp.linalg.det(self.covariances_)

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
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
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        For an example of GMM selection using `bic` information criterion,
        refer to :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py`.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[1] + self._n_parameters() * np.log(
            X.shape[1]
        )

    def general_way_logpdf(self, X, flat=True):
        
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
            -(n_features / 2.) * self.xp.log(2 * np.pi)
            - (1./2.) * self.xp.log(self.det_covs_)[:, None, :]
            - (1./2.) * kernel
        )
        logpdf = self.logsumexp(_logpdf, axis=-1)
        return logpdf

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        raise NotImplementedError
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()



class GMMFit:
    
    @property
    def xp(self):
        if self.use_gpu:
            return cp
        else:
            return np

    def __init__(self, samples_in=None, n_components=30, gpu=None):

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
            random_state=None,
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
        self.log_det_J = (self.ndim * self.xp.log(2) - self.xp.sum(self.xp.log(self.sample_maxs - self.sample_mins), axis=-1))
        self.fitted = True
        
    def transform_to_gmm_basis(self, samples):
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
        
        tmp = (samples + 1.) / 2. * (maxs - mins) + mins
        if squeeze:
            return tmp[0]
        else:
            return tmp

    def rvs(self, size=(1,), flat=False):
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
        _logpdf += self.log_det_J[:, None, ]
        # tmp += self.log_det_J[:, None, ]
        if flat:
            logpdf = self.gmm.logsumexp(_logpdf, axis=0)
            # tmp2 = self.gmm.logsumexp(tmp, axis=0)

        else:
            logpdf = _logpdf

        return logpdf

    def bic(self, x, flat=False):
        
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


def vec_fit_gmm_min_bic(samples, min_comp=1, max_comp=30, n_samp_bic_test=5000, gpu=None, verbose=False):

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
        gmm = GMMFit(samples_here, n_components=comp_i, gpu=gpu)
        samp = gmm.rvs(n_samp_bic_test)
        bic = gmm.bic(samp)

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
                _maxs
            ]
        
        converged = (counts_above_min_bic >= 2)
        converged_here = converged[inds_here]
       
        del gmm
        if use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"{comp_i} components: {converged.sum()} converged out of {converged.shape[0]}.")

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

    # import pickle
    # with open("gmm_tmp_2.pickle", "wb") as fp:
    #     pickle.dump((weights, means, covs, invcovs, dets, mins, maxs), fp, pickle.HIGHEST_PROTOCOL)

    full_gmm = FullGaussianMixtureModel(weights, means, covs, invcovs, dets, mins, maxs, use_cupy=use_gpu)
    return full_gmm

    
if __name__ == "__main__":
    samples_tmp = np.load("samples_examples.npy")[:, :, :, np.array([0, 1, 2, 4, 6, 7])]  # [:1]
    samples_tmp = samples_tmp.reshape(samples_tmp.shape[0], -1, samples_tmp.shape[-1])
    full_gmm = vec_fit_gmm_min_bic(samples_tmp, use_gpu=True, verbose=True)
    samp_gen = full_gmm.rvs(size=(int(1e6),))
    samp_logpdf = full_gmm.logpdf(samp_gen)
    import matplotlib.pyplot as plt
    plt.scatter(samp_gen[:, 1].get(), samp_gen[:, 0].get())
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
