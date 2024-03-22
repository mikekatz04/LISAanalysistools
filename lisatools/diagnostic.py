import warnings
from types import ModuleType, NoneType
from typing import Optional, Any, Tuple, List

from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from eryn.utils import TransformContainer

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    pass

from .sensitivity import get_sensitivity, SensitivityMatrix
from .datacontainer import DataResidualArray
from .utils.utility import get_array_module


def inner_product(
    sig1: np.ndarray | list | DataResidualArray,
    sig2: np.ndarray | list | DataResidualArray,
    dt: Optional[float] = None,
    df: Optional[float] = None,
    f_arr: Optional[float] = None,
    psd: Optional[str | NoneType | np.ndarray | SensitivityMatrix] = "LISASens",
    psd_args: Optional[tuple] = (),
    psd_kwargs: Optional[dict] = {},
    normalize: Optional[bool | str] = False,
    complex: Optional[bool] = False,
) -> float | complex:
    """Compute the inner product between two signals weighted by a psd.

    The inner product between time series :math:`a(t)` and :math:`b(t)` is

    .. math::

        \langle a | b \\rangle = 2\int_{f_\\text{min}}^{f_\\text{max}} \\frac{\\tilde{a}(f)^*\\tilde{b}(f) + \\tilde{a}(f)\\tilde{b}(f)^*}{S_n(f)} df\ \ ,

    where :math:`\\tilde{a}(f)` is the Fourier transform of :math:`a(t)` and :math:`S_n(f)` is the one-sided Power Spectral Density of the noise.

    The inner product can be left complex using the ``complex`` kwarg.

    **GPU Capability**: Pass CuPy arrays rather than NumPy arrays.

    Args:
        sig1: First signal to use for the inner product.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        sig2: Second signal to use for the inner product.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        dt: Time step in seconds. If provided, assumes time-domain signals.
        df: Constant frequency spacing. This will assume a frequency domain signal with constant frequency spacing.
        f_arr: Array of specific frequencies at which the signal is given.
        psd: Indicator of what psd to use. If a ``str``, this will be passed as the ``sens_fn`` kwarg to :func:`get_sensitivity`.
            If ``None``, it will be an array of ones. Or, you can pass a 1D ``np.ndarray`` of psd values that must be the same length
            as the frequency domain signals.
        psd_args: Arguments to pass to the psd function if ``type(psd) == str``.
        psd_kwargs: Keyword arguments to pass to the psd function if ``type(psd) == str``.
        normalize: Normalize the inner product. If ``True``, it will normalize the square root of the product of individual signal inner products.
            You can also pass ``"sig1"`` or ``"sig2"`` to normalize with respect to one signal.
        complex: If ``True``, return the complex value of the inner product rather than just its real-valued part.

    Returns:
        Inner product value.

    """
    # initial input checks and setup
    sig1 = DataResidualArray(sig1, dt=dt, f_arr=f_arr, df=df)
    sig2 = DataResidualArray(sig2, dt=dt, f_arr=f_arr, df=df)

    if sig1.nchannels != sig2.nchannels:
        raise ValueError(
            f"Signal 1 has {sig1.nchannels} channels. Signal 2 has {sig2.nchannels} channels. Must be the same."
        )

    xp = get_array_module(sig1[0])

    # checks
    for i in range(sig1.nchannels):
        if not type(sig1[0]) == type(sig1[i]) and type(sig1[0]) == type(sig2[i]):
            raise ValueError(
                "Array in sig1, index 0 sets array module. Not all arrays match that module type (Numpy or Cupy)"
            )

    if sig1.data_length != sig2.data_length:
        raise ValueError(
            "The two signals are two different lengths. Must be the same length."
        )

    freqs = sig1.f_arr

    # get psd weighting
    if not isinstance(psd, SensitivityMatrix):
        psd = SensitivityMatrix(freqs, [psd], *psd_args, **psd_kwargs)

    operational_sets = []

    if psd.ndim == 3:
        assert psd.shape[0] == psd.shape[1] == sig1.shape[0] == sig2.shape[0]

        for i in range(psd.shape[0]):
            for j in range(i, psd.shape[1]):
                factor = 1.0 if i == j else 2.0
                operational_sets.append(
                    dict(factor=factor, sig1_ind=i, sig2_ind=j, psd_ind=(i, j))
                )

    elif psd.ndim == 2 and psd.shape[0] > 1:
        assert psd.shape[0] == sig1.shape[0] == sig2.shape[0]
        for i in range(psd.shape[0]):
            operational_sets.append(dict(factor=1.0, sig1_ind=i, sig2_ind=i, psd_ind=i))

    elif psd.ndim == 2 and psd.shape[0] == 1:
        for i in range(sig1.shape[0]):
            operational_sets.append(dict(factor=1.0, sig1_ind=i, sig2_ind=i, psd_ind=0))

    else:
        raise ValueError("# TODO")

    if complex:
        func = lambda x: x
    else:
        func = xp.real

    # initialize
    out = 0.0
    x = freqs

    # account for hp and hx if included in time domain signal
    for op_set in operational_sets:
        factor = op_set["factor"]

        temp1 = sig1[op_set["sig1_ind"]]
        temp2 = sig2[op_set["sig2_ind"]]
        psd_tmp = psd[op_set["psd_ind"]]

        ind_start = 1 if np.isnan(psd_tmp[0]) else 0

        y = (
            func(temp1[ind_start:].conj() * temp2[ind_start:]) / psd_tmp[ind_start:]
        )  # assumes right summation rule
        # df is sunk into trapz
        tmp_out = factor * 4 * xp.trapz(y, x=x[ind_start:])
        out += tmp_out

    # normalize the inner produce
    normalization_value = 1.0
    if normalize is True:
        norm1 = inner_product(
            sig1,
            sig1,
            psd=psd,
            normalize=False,
        )
        norm2 = inner_product(
            sig2,
            sig2,
            psd=psd,
            normalize=False,
        )

        normalization_value = np.sqrt(norm1 * norm2)

    elif isinstance(normalize, str):
        if normalize == "sig1":
            sig_to_normalize = sig1

        elif normalize == "sig2":
            sig_to_normalize = sig2

        else:
            raise ValueError(
                "If normalizing with respect to sig1 or sig2, normalize kwarg must either be 'sig1' or 'sig2'."
            )

        normalization_value = inner_product(
            sig_to_normalize,
            sig_to_normalize,
            psd=psd,
            normalize=False,
        )

    elif normalize is not False:
        raise ValueError("Normalize must be True, False, 'sig1', or 'sig2'.")

    out /= normalization_value
    return out


def residual_source_likelihood_term(
    data_res_arr: DataResidualArray, **kwargs: dict
) -> float | complex:
    """Calculate the source term in the Likelihood for a data residual (d - h).

    The source term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_\\text{src} = -\\frac{1}{2}\\langle \\vec{d} - \\vec{h} | \\vec{d} - \\vec{h}\\rangle.

    Args:
        data_res_arr: Data residual.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Source term Likelihood value.

    """
    kwargs["normalize"] = False
    ip_val = inner_product(data_res_arr, data_res_arr, **kwargs)
    return -1 / 2.0 * ip_val


def noise_likelihood_term(psd: SensitivityMatrix) -> float:
    """Calculate the noise term in the Likelihood.

    The noise term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_n = -\\sum \\log{\\vec{S}_n}.

    Args:
        psd: Sensitivity information.

    Returns:
        Noise term Likelihood value.

    """
    fix = np.isnan(psd[:]) | np.isinf(psd[:])
    assert np.sum(fix) == np.prod(psd.shape[:-1]) or np.sum(fix) == 0
    nl_val = -1.0 / 2.0 * np.sum(np.log(psd[~fix]))
    return nl_val


def residual_full_source_and_noise_likelihood(
    data_res_arr: DataResidualArray,
    psd: str | NoneType | np.ndarray | SensitivityMatrix,
    **kwargs: dict,
) -> float | complex:
    """Calculate the full Likelihood including noise and source terms.

    The noise term is calculated with :func:`noise_likelihood_term`.

    The source term is calcualted with :func:`residual_source_likelihood_term`.

    Args:
        data_res_arr: Data residual.
        psd: Sensitivity information.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
       Full Likelihood value.

    """
    if not isinstance(psd, SensitivityMatrix):
        # TODO: maybe adjust so it can take a list just like Sensitivity matrix
        psd = SensitivityMatrix(data_res_arr.f_arr, [psd], **kwargs)

    # remove key
    for key in "psd", "psd_args", "psd_kwargs":
        if key in kwargs:
            kwargs.pop(key)

    rslt = residual_source_likelihood_term(data_res_arr, psd=psd, **kwargs)

    nlt = noise_likelihood_term(psd)
    return nlt + rslt


def data_signal_source_likelihood_term(
    data_arr: DataResidualArray, sig_arr: DataResidualArray, **kwargs: dict
) -> float | complex:
    """Calculate the source term in the Likelihood for separate signal and data.

    The source term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_\\text{src} = -\\frac{1}{2}\\left(\\langle \\vec{d} | \\vec{d}\\rangle + \\langle \\vec{h} | \\vec{h}\\rangle - 2\\langle \\vec{d} | \\vec{h}\\rangle \\right)\ \ .

    Args:
        data_arr: Data.
        sig_arr: Signal.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Source term Likelihood value.

    """
    kwargs["normalize"] = False
    d_h = inner_product(data_arr, sig_arr, **kwargs)
    h_h = inner_product(sig_arr, sig_arr, **kwargs)
    d_d = inner_product(data_arr, data_arr, **kwargs)
    return -1 / 2.0 * (d_d + h_h - 2 * d_h)


def data_signal_full_source_and_noise_likelihood(
    data_arr: DataResidualArray,
    sig_arr: DataResidualArray,
    psd: str | NoneType | np.ndarray | SensitivityMatrix,
    **kwargs: dict,
) -> float | complex:
    """Calculate the full Likelihood including noise and source terms.

    Here, the signal is treated separate from the data.

    The noise term is calculated with :func:`noise_likelihood_term`.

    The source term is calcualted with :func:`data_signal_source_likelihood_term`.

    Args:
        data_arr: Data.
        sig_arr: Signal.
        psd: Sensitivity information.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
       Full Likelihood value.

    """
    if not isinstance(psd, SensitivityMatrix):
        # TODO: maybe adjust so it can take a list just like Sensitivity matrix
        psd = SensitivityMatrix(data_arr.f_arr, [psd], **kwargs)

    # remove key
    for key in "psd", "psd_args", "psd_kwargs":
        if key in kwargs:
            kwargs.pop(key)

    rslt = data_signal_source_likelihood_term(data_arr, sig_arr, psd=psd, **kwargs)

    nlt = noise_likelihood_term(psd)

    return nlt + rslt


def snr(
    sig1: np.ndarray | list | DataResidualArray,
    *args: Any,
    data: Optional[np.ndarray | list | DataResidualArray] = None,
    **kwargs: Any,
) -> float:
    """Compute the snr between two signals weighted by a psd.

    The signal-to-noise ratio of a signal is :math:`\\sqrt{\\langle a|a\\rangle}`.

    This will be the optimal SNR if ``data==None``. If a data array is given, it will be the observed
    SNR: :math:`\\langle d|a\\rangle/\\sqrt{\\langle a|a\\rangle}`.

    **GPU Capability**: Pass CuPy arrays rather than NumPy arrays.

    Args:
        sig1: Signal to use as the templatefor the SNR.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        *args: Arguments to pass to :func:`inner_product`.
        data: Data becomes the ``sig2`` argument to :func:`inner_product`.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Optimal or detected SNR value (depending on ``data`` kwarg).

    """

    # get optimal SNR
    opt_snr = np.sqrt(inner_product(sig1, sig1, *args, **kwargs).real)
    if data is None:
        return opt_snr

    else:
        # if inputed data, calculate detected SNR
        det_snr = inner_product(sig1, data, *args, **kwargs).real / opt_snr
        return det_snr


def h_var_p_eps(
    step: float,
    waveform_model: callable,
    params: ArrayLike,
    index: int,
    parameter_transforms: Optional[TransformContainer] = None,
    waveform_args: Optional[tuple] = (),
    waveform_kwargs: Optional[dict] = {},
) -> np.ndarray:  # TODO: check this
    """Calculate the waveform with a perturbation step of the variable V[i]

    Args:
        waveform_model: Callable function to the waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters that are over derivatives (not in fill dict of parameter transforms)
        step: Absolute step size for variable of interest.
        index: Index to parameter of interest.
        parameter_transforms: `TransformContainer <https://mikekatz04.github.io/Eryn/html/user/utils.html#eryn.utils.TransformContainer>`_ object to transform from the derivative parameter basis
            to the waveform parameter basis. This class can also fill in fixed parameters where the derivatives are not being taken.
        waveform_args: args (beyond parameters) for the waveform generator.
        waveform_kwargs: kwargs for the waveform generation.

    Returns:
        Perturbation to the waveform in the given parameter. Will always be 2D array with shape ``(num channels, data length)``

    """
    params_p_eps = params.copy()
    params_p_eps[index] += step

    if parameter_transforms is not None:
        # transform
        params_p_eps = parameter_transforms.transform_base_parameters(params_p_eps)

    args_in = tuple(params_p_eps) + tuple(waveform_args)
    dh = waveform_model(*args_in, **waveform_kwargs)

    # adjust output based on waveform model output
    # needs to be 2D array
    if (isinstance(dh, np.ndarray) or isinstance(dh, cp.ndarray)) and dh.ndim == 1:
        xp = get_array_module(dh)
        dh = xp.atleast_2d(dh)

    elif isinstance(dh, list):
        xp = get_array_module(dh[0])
        dh = xp.asarray(dh)

    return dh


def dh_dlambda(
    eps: float,
    *args: tuple,
    more_accurate: Optional[bool] = True,
    **kwargs: dict,
) -> np.ndarray:
    """Derivative of the waveform

    Calculate the derivative of the waveform with precision of order (step^4)
    with respect to the variable V in the i direction.

    Args:
        eps: Absolute **derivative** step size for variable of interest.
        *args: Arguments passed to :func:`h_var_p_eps`.
        more_accurate: If ``True``, run a more accurate derivate requiring 2x more waveform generations.
        **kwargs: Keyword arguments passed to :func:`h_var_p_eps`.

    Returns:
        Numerical derivative of the waveform with respect to a varibale of interest. Will be 2D array
        with shape ``(num channels, data length)``.

    """
    if more_accurate:
        # Derivative of the Waveform
        # up
        h_I_up_2eps = h_var_p_eps(2 * eps, *args, **kwargs)
        h_I_up_eps = h_var_p_eps(eps, *args, **kwargs)
        # down
        h_I_down_2eps = h_var_p_eps(-2 * eps, *args, **kwargs)
        h_I_down_eps = h_var_p_eps(-eps, *args, **kwargs)

        # make sure they are all the same length
        ind_max = np.min(
            [
                h_I_up_2eps.shape[1],
                h_I_up_eps.shape[1],
                h_I_down_2eps.shape[1],
                h_I_down_eps.shape[1],
            ]
        )

        # error scales as eps^4
        dh_I = (
            -h_I_up_2eps[:, :ind_max]
            + h_I_down_2eps[:, :ind_max]
            + 8 * (h_I_up_eps[:, :ind_max] - h_I_down_eps[:, :ind_max])
        ) / (12 * eps)

    else:
        # Derivative of the Waveform
        # up
        h_I_up_eps = h_var_p_eps(eps, *args, **kwargs)
        # down
        h_I_down_eps = h_var_p_eps(-eps, *args, **kwargs)

        ind_max = np.min([h_I_up_eps.shape[1], h_I_down_eps.shape[1]])

        # TODO: check what error scales as.
        dh_I = (h_I_up_eps[:, :ind_max] - h_I_down_eps[:, :ind_max]) / (2 * eps)
        # Time thta it takes for one variable: approx 5 minutes

    return dh_I


def info_matrix(
    eps: float | np.ndarray,
    waveform_model: callable,
    params: ArrayLike,
    deriv_inds: Optional[ArrayLike] = None,
    inner_product_kwargs: Optional[dict] = {},
    return_derivs: Optional[bool] = False,
    **kwargs: dict,
) -> np.ndarray | Tuple[np.ndarray, list]:
    """Calculate Information Matrix.

    This calculates the information matrix for a given waveform model at a given set of parameters.
    The inverse of the information matrix gives the covariance matrix.

    This is also referred to as the Fisher information matrix, but @MichaelKatz has chosen to leave out the name because of `this <https://www.newstatesman.com/long-reads/2020/07/ra-fisher-and-science-hatred>`_.

    The info matrix is given by:

    .. math::

        M_{ij} = \\langle h_i | h_j \\rangle \\text{ with } h_i = \\frac{\\partial h}{\\partial \\lambda_i}.

    Args:
        eps: Absolute **derivative** step size for variable of interest. Can be provided as a ``float`` value that applies
            to all variables or an array, one for each parameter being evaluated in the information matrix.
        waveform_model: Callable function to the waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters.
        deriv_inds: Subset of parameters of interest for which to calculate the information matrix, by index.
            If ``None``, it will be ``np.arange(len(params))``.
        inner_product_kwargs: Keyword arguments for the inner product function.
        return_derivs: If ``True``, also returns computed numerical derivatives.
        **kwargs: Keyword arguments passed to :func:`dh_dlambda`.

    Returns:
        If ``not return_derivs``, this will be the information matrix as a numpy array. If ``return_derivs is True``,
        it will be a tuple with the first entry as the information matrix and the second entry as the partial derivatives.

    """

    # setup initial information
    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_info_params = len(deriv_inds)

    if isinstance(eps, float):
        eps = np.full_like(params, eps)

    # collect derivatives over the variables of interest
    dh = []
    for i, eps_i in zip(deriv_inds, eps):
        # derivative up
        temp = dh_dlambda(eps_i, waveform_model, params, i, **kwargs)
        dh.append(temp)

    # calculate the components of the symmetric matrix
    info = np.zeros((num_info_params, num_info_params))
    for i in range(num_info_params):
        for j in range(i, num_info_params):
            info[i][j] = inner_product(
                [dh[i][k] for k in range(len(dh[i]))],
                [dh[j][k] for k in range(len(dh[i]))],
                **inner_product_kwargs,
            )
            info[j][i] = info[i][j]

    if return_derivs:
        return info, dh
    else:
        return info


def covariance(
    *info_mat_args: tuple,
    info_mat: Optional[np.ndarray] = None,
    diagonalize: Optional[bool] = False,
    return_info_mat: Optional[bool] = False,
    precision: Optional[bool] = False,
    **info_mat_kwargs: dict,
) -> np.ndarray | list:
    """Calculate covariance matrix for a set of EMRI parameters, computing the information matrix if not supplied.

    Args:
        *info_mat_args: Set of arguments to pass to :func:`info_matrix`. Not required if ``info_mat`` is not ``None``.
        info_mat: Pre-computed information matrix. If supplied, this matrix will be inverted.
        diagonalize: If ``True``, diagonalizes the covariance matrix.
        return_info_mat: If ``True``, also returns the computed information matrix.
        precision: If ``True``, uses 500-dps precision to compute the information matrix inverse (requires `mpmath <https://mpmath.org>`_). This is typically a good idea as the information matrix can be highly ill-conditioned.
        **info_mat_kwargs: Keyword arguments to pass to :func:`info_matrix`.

    Returns:
        Covariance matrix. If ``return_info_mat is True``. A list will be returned with the covariance as the first
        entry and the information matrix as the second entry. If ``return_derivs is True`` (keyword argument to :func:`info_matrix`),
        then another entry will be added to the list for the derivatives.

    """

    if info_mat is None:
        info_mat = info_matrix(*info_mat_args, **info_mat_kwargs)

    # parse output properly
    if "return_derivs" in info_mat_kwargs and info_mat_kwargs["return_derivs"]:
        return_derivs = True
        info_mat, dh = info_mat

    else:
        return_derivs = False

    # attempt to import and setup mpmath if precision required
    if precision:
        try:
            import mpmath as mp

            mp.mp.dps = 500
        except ModuleNotFoundError:
            print("mpmath module not installed. Defaulting to low precision...")
            precision = False

    if precision:
        hp_info_mat = mp.matrix(info_mat.tolist())
        U, S, V = mp.svd_r(hp_info_mat)  # singular value decomposition
        temp = mp.diag([val ** (-1) for val in S])  # get S**-1
        temp2 = V.T * temp * U.T  # construct pseudo-inverse
        cov = np.array(temp2.tolist(), dtype=np.float64)
    else:
        cov = np.linalg.pinv(info_mat)

    if diagonalize:
        # get eigeninformation
        eig_vals, eig_vecs = get_eigeninfo(cov, high_precision=precision)

        # diagonal cov now
        cov = np.dot(np.dot(eig_vecs.T, cov), eig_vecs)

    # just requesting covariance
    if True not in [return_info_mat, return_derivs]:
        return cov

    # if desiring more information, create a list capable of variable size
    returns = [cov]

    # add information matrix
    if return_info_mat:
        returns.append(info_mat)

    # add derivatives
    if return_derivs:
        returns.append(dh)

    return returns


def plot_covariance_corner(
    params: np.ndarray,
    cov: np.ndarray,
    nsamp: Optional[int] = 25000,
    fig: Optional[plt.Figure] = None,
    **kwargs: dict,
) -> plt.Figure:
    """Construct a corner plot for a given covariance matrix.

    The `corner <https://corner.readthedocs.io/en/latest/>`_ module is required for this.

    Args:
        params: The set of parameters used for the event (the mean vector of the covariance matrix).
        cov: Covariance matrix from which to construct the corner plot.
        nsamp: Number of samples to draw from the multivariate distribution.
        fig: Matplotlib :class:`plt.Figure` object. Use this if passing an existing corner plot figure.
        **kwargs: Keyword arguments for the corner plot - see the module documentation for more info.

    Returns:
        The corner plot figure.

    """

    # TODO: add capability for ChainConsumer?
    try:
        import corner
    except ModuleNotFoundError:
        raise ValueError(
            "Attempting to plot using the corner module, but it is not installed."
        )

    # generate fake samples from the covariance distribution
    samp = np.random.multivariate_normal(params, cov, size=nsamp)

    # make corner plot
    fig = corner.corner(samp, **kwargs)
    return fig


def plot_covariance_contour(
    params: np.ndarray,
    cov: np.ndarray,
    horizontal_index: int,
    vertical_index: int,
    nsamp: Optional[int] = 25000,
    ax: Optional[plt.Axes] = None,
    **kwargs: dict,
) -> plt.Axes | Tuple[plt.Figure, plt.Axes]:
    """Construct a contour plot for a given covariance matrix on a single axis object.

    The `corner <https://corner.readthedocs.io/en/latest/>`_ module is required for this.

    Args:
        params: The set of parameters used for the event (the mean vector of the covariance matrix).
        cov: Covariance matrix from which to construct the corner plot.
        horizontal_index: Parameter index to plot along the horizontal axis of the contour plot.
        vertical_index: Parameter index to plot along the vertical axis of the contour plot.
        nsamp: Number of samples to draw from the multivariate distribution.
        fig: Matplotlib :class:`plt.Figure` object. Use this if passing an existing corner plot figure.
        **kwargs: Keyword arguments for the corner plot - see the module documentation for more info.

    Returns:
        If ``ax`` is provided, the return will be that ax object. If it is not provided, then a
        Matplotlib Figure and Axes obejct is created and returned as a tuple: ``(plt.Figure, plt.Axes)``.

    """

    # TODO: add capability for ChainConsumer?
    try:
        import corner
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Attempting to plot using the corner module, but it is not installed."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    # generate fake samples from the covariance distribution
    samp = np.random.multivariate_normal(params, cov, size=nsamp)

    x = samp[:, horizontal_index]
    y = samp[:, vertical_index]

    # make corner plot
    corner.hist2d(x, y, ax=ax, **kwargs)

    if fig is None:
        return ax
    else:
        return (fig, ax)


def get_eigeninfo(
    arr: np.ndarray, high_precision: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs eigenvalue decomposition and returns the eigenvalues and right-eigenvectors for the supplied fisher/covariance matrix.

    Args:
        arr: Input matrix for which to perform eigenvalue decomposition.
        high_precision: If ``True``, use 500-dps precision to ensure accurate eigenvalue decomposition
            (requires `mpmath <https://mpmath.org>`_ to be installed). Defaults to False.

    Returns:
        Tuple containing Eigenvalues and right-Eigenvectors for the supplied array, constructed such that evects[:,k] corresponds to evals[k].


    """

    if high_precision:
        try:
            import mpmath as mp

            mp.mp.dps = 500
        except ModuleNotFoundError:
            print("mpmath is not installed - using low-precision eigen decomposition.")
            high_precision = False

    if high_precision:
        hp_arr = mp.matrix(arr.tolist())
        # get eigenvectors
        E, EL, ER = mp.eig(hp_arr, left=True, right=True)

        # convert back
        evals = np.array(E, dtype=np.float64)
        evects = np.array(ER.tolist(), dtype=np.float64)

        return evals, evects

    else:
        evals, evects = np.linalg.eig(arr)
        return evals, evects


def cutler_vallisneri_bias(
    waveform_model_true: callable,
    waveform_model_approx: callable,
    params: np.ndarray,
    eps: float | np.ndarray,
    input_diagnostics: Optional[dict] = None,
    info_mat: Optional[np.ndarray] = None,
    deriv_inds: Optional[ArrayLike] = None,
    return_derivs: Optional[bool] = False,
    return_cov: Optional[bool] = False,
    parameter_transforms: Optional[TransformContainer] = None,
    waveform_true_args: Optional[tuple] = (),
    waveform_true_kwargs: Optional[dict] = {},
    waveform_approx_args: Optional[tuple] = (),
    waveform_approx_kwargs: Optional[dict] = {},
    inner_product_kwargs: Optional[dict] = {},
) -> list:
    """Calculate the Cutler-Vallisneri bias.

    # TODO: add basic math

    Args:
        waveform_model_true: Callable function to the **true** waveform generator with signature ``(*params, **waveform_kwargs)``.
        waveform_model_approx: Callable function to the **approximate** waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters.
        eps: Absolute **derivative** step size. See :func:`info_matrix`.
        input_diagnostics: Dictionary including the diagnostic information if it is precomputed. Dictionary must include
            keys ``"cov"`` (covariance matrix, output of :func:`covariance`), ``"h_true"`` (the **true** waveform),
            and ``"dh"`` (derivatives of the waveforms, list of outputs from :func:`dh_dlambda`).
        info_mat: Pre-computed information matrix. If supplied, this matrix will be inverted to find the covariance.
        deriv_inds: Subset of parameters of interest. See :func:`info_matrix`.
        return_derivs: If ``True``, also returns computed numerical derivatives.
        return_cov: If ``True``, also returns computed covariance matrix.
        parameter_transforms: `TransformContainer <https://mikekatz04.github.io/Eryn/html/user/utils.html#eryn.utils.TransformContainer>`_ object. See :func:`info_matrix`.
        waveform_true_args: Arguments for the **true** waveform generator.
        waveform_true_kwargs: Keyword arguments for the **true** waveform generator.
        waveform_approx_args: Arguments for the **approximate** waveform generator.
        waveform_approx_kwargs: Keyword arguments for the **approximate** waveform generator.
        inner_product_kwargs: Keyword arguments for the inner product function.

    Returns:
        List of return information. By default, it is ``[systematic error, bias]``.
        If ``return_derivs`` or ``return_cov`` are ``True``, they will be added to the list with derivs added before covs.

    """

    if deriv_inds is None:
        deriv_inds = np.arange(len(params))

    if info_mat is not None and input_diagnostics is not None:
        warnings.warn(
            "Provided info_mat and input_diagnostics kwargs. Ignoring info_mat."
        )

    # adjust parameters to waveform basis
    params_in = parameter_transforms.transform_base_parameters(params.copy())

    if input_diagnostics is None:
        # get true waveform
        h_true = waveform_model_true(
            *(tuple(params_in) + tuple(waveform_true_args)), **waveform_true_kwargs
        )

        # get covariance info and waveform derivatives
        cov, dh = covariance(
            eps,
            waveform_model_true,
            params,
            return_derivs=True,
            deriv_inds=deriv_inds,
            info_mat=info_mat,
            parameter_transforms=parameter_transforms,
            waveform_args=waveform_true_args,
            waveform_kwargs=waveform_true_kwargs,
            inner_product_kwargs=inner_product_kwargs,
        )

    else:
        # pre-computed info
        cov = input_diagnostics["cov"]
        h_true = input_diagnostics["h_true"]
        dh = input_diagnostics["dh"]

    # get approximate waveform
    h_approx = waveform_model_approx(
        *(tuple(params_in) + tuple(waveform_approx_args)), **waveform_approx_kwargs
    )

    # adjust/check waveform outputs
    if isinstance(h_true, np.ndarray) and h_true.ndim == 1:
        h_true = [h_true]
    elif isinstance(h_true, np.ndarray) and h_true.ndim == 2:
        h_true = list(h_true)

    if isinstance(h_approx, np.ndarray) and h_approx.ndim == 1:
        h_approx = [h_approx]
    elif isinstance(h_approx, np.ndarray) and h_approx.ndim == 2:
        h_approx = list(h_approx)

    assert len(h_approx) == len(h_true)
    assert np.all(
        np.asarray([len(h_approx[i]) == len(h_true[i]) for i in range(len(h_true))])
    )

    # difference in the waveforms
    diff = [h_true[i] - h_approx[i] for i in range(len(h_approx))]

    # systematic err
    syst_vec = np.array(
        [
            inner_product(
                dh[k],
                diff,
                **inner_product_kwargs,
            )
            for k in range(len(deriv_inds))
        ]
    )

    bias = np.dot(cov, syst_vec)

    # return list
    returns = [syst_vec, bias]

    # add anything requested
    if return_cov:
        returns.append(cov)

    if return_derivs:
        returns.append(dh)

    return returns


def scale_to_snr(
    target_snr: float,
    sig: np.ndarray | list,
    *snr_args: tuple,
    return_orig_snr: Optional[bool] = False,
    **snr_kwargs: dict,
) -> np.ndarray | list | Tuple[np.ndarray | list, float]:
    """Calculate the SNR and scale a signal.

    Args:
        target_snr: Desired SNR value for the injected signal.
        sig: Signal to adjust. A copy will be made.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        *snr_args: Arguments to pass to :func:`snr`.
        return_orig_snr: If ``True``, return the original SNR in addition to the adjusted data.
        **snr_kwargs: Keyword arguments to pass to :func:`snr`.

    Returns:
        Returns the copied input signal adjusted to the target SNR. If ``return_orig_snr is True``, the original
        SNR is added as the second entry of a tuple with the adjusted signal (as the first entry in the tuple).

    """
    # get the snr and adjustment factor
    snr_out = snr(sig, *snr_args, **snr_kwargs)
    factor = target_snr / snr_out

    # any changes back to the original signal type
    back_single = False
    back_2d_array = False

    if isinstance(sig, list) is False and sig.ndim == 1:
        sig = [sig]
        back_single = True

    elif isinstance(sig, list) is False and sig.ndim == 2:
        sig = list(sig)
        back_2d_array = True

    # adjust
    out = [sig_i * factor for sig_i in sig]

    # adjust type back to the input type
    if back_2d_array:
        out = np.asarray(out)

    if back_single:
        out = out[0]

    if return_orig_snr:
        return (out, snr_out)

    return out
