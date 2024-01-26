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

from lisatools.sensitivity import get_sensitivity


def get_array_module(arr: np.ndarray | cp.ndarray) -> ModuleType:
    """Return array library of an array (np/cp).

    Args:
        arr: Numpy or Cupy array.

    """
    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, cp.ndarray):
        return cp
    else:
        raise ValueError("arr must be a numpy or cupy array.")


def inner_product(
    sig1: np.ndarray | list,
    sig2: np.ndarray | list,
    dt: Optional[float] = None,
    df: Optional[float] = None,
    f_arr: Optional[float] = None,
    PSD: Optional[str | NoneType | np.ndarray] = "lisasens",
    PSD_args: Optional[tuple] = (),
    PSD_kwargs: Optional[dict] = {},
    normalize: Optional[bool | str] = False,
    complex: Optional[bool] = False,
) -> float | complex:
    """Compute the inner product between two signals weighted by a PSD.

    The inner product between time series :math:`a(t)` and :math:`b(t)` is

    .. math::

        \langle a | b \\rangle = 2\int_{f_\\text{min}}^{f_\\text{max}} \\frac{\\tilde{a}(f)^*\\tilde{b}(f)}{S_n(f)} df\ \ ,

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
        PSD: Indicator of what PSD to use. If a ``str``, this will be passed as the ``sens_fn`` kwarg to :func:`get_sensitivity`.
            If ``None``, it will be an array of ones. Or, you can pass a 1D ``np.ndarray`` of PSD values that must be the same length
            as the frequency domain signals.
        PSD_args: Arguments to pass to the PSD function if ``type(PSD) == str``.
        PSD_kwargs: Keyword arguments to pass to the PSD function if ``type(PSD) == str``.
        normalize: Normalize the inner product. If ``True``, it will normalize the square root of the product of individual signal inner products.
            You can also pass ``"sig1"`` or ``"sig2"`` to normalize with respect to one signal.
        complex: If ``True``, return the complex value of the inner product rather than just its real-valued part.

    Returns:
        Inner product value.

    """
    # initial input checks and setup
    if df is None and dt is None and f_arr is None:
        raise ValueError("Must provide either df, dt or f_arr keyword arguments.")

    if not isinstance(sig1, list) and sig1.ndim == 1:
        sig1 = [sig1]
    elif not isinstance(sig1, list) and sig1.ndim == 2:
        sig1 = list(sig1)

    if not isinstance(sig2, list) and sig2.ndim == 1:
        sig2 = [sig2]
    elif not isinstance(sig2, list) and sig2.ndim == 2:
        sig2 = list(sig2)

    if len(sig1) != len(sig2):
        raise ValueError(
            "Signal 1 has {} channels. Signal 2 has {} channels. Must be equal.".format(
                len(sig1), len(sig2)
            )
        )

    xp = get_array_module(sig1[0])

    # checks
    for i in range(len(sig1)):
        if not type(sig1[0]) == type(sig1[i]) and type(sig1[0]) == type(sig2[i]):
            raise ValueError(
                "Array in sig1, index 0 sets array module. Not all arrays match that module type (Numpy or Cupy)"
            )

    if dt is not None:
        # setup time-domain signals for inner product
        if len(sig1[0]) != len(sig2[0]):
            warnings.warn(
                "The two signals are two different lengths in the time domain. Zero padding smaller array."
            )

            length = len(sig1[0]) if len(sig1[0]) > len(sig2[0]) else len(sig2[0])

            sig1 = [xp.pad(sig, (0, length - len(sig1[0]))) for sig in sig1]
            sig2 = [xp.pad(sig, (0, length - len(sig2[0]))) for sig in sig2]

        length = len(sig1[0])

        freqs = xp.fft.rfftfreq(length, dt)

        ft_sig1 = [xp.fft.rfft(sig) * dt for sig in sig1]
        ft_sig2 = [xp.fft.rfft(sig) * dt for sig in sig2]

    else:
        # setup frequency domain info
        ft_sig1 = sig1
        ft_sig2 = sig2

        if df is not None:
            freqs = xp.arange(len(sig1[0])) * df

        else:
            freqs = f_arr

    # get PSD weighting
    if isinstance(PSD, str):
        PSD_arr = get_sensitivity(freqs, sens_fn=PSD, *PSD_args, **PSD_kwargs)

    elif isinstance(PSD, xp.ndarray):
        assert PSD.ndim == 1
        PSD_arr = PSD

    elif PSD is None:
        PSD_arr = xp.full_like(freqs, 1.0)

    else:
        raise ValueError(
            "PSD must be a string giving the sens_fn or a predetermimed array or None if noise weighting is included in a signal."
        )

    out = 0.0
    x = freqs

    # fix nan if initial freq is zero
    if xp.isnan(PSD_arr[0]):
        # set it to the neighboring value
        PSD_arr[0] = PSD_arr[1]

    # account for hp and hx if included in time domain signal
    for temp1, temp2 in zip(ft_sig1, ft_sig2):
        if complex:
            func = lambda x: x
        else:
            func = xp.real
        y = func(temp1.conj() * temp2) / PSD_arr  # assumes right summation rule

        out += 4 * xp.trapz(y, x=x)

    # normalize the inner produce
    normalization_value = 1.0
    if normalize is True:
        norm1 = inner_product(
            sig1,
            sig1,
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD=PSD,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
            normalize=False,
        )
        norm2 = inner_product(
            sig2,
            sig2,
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD=PSD,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
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
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD=PSD,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
            normalize=False,
        )

    elif normalize is not False:
        raise ValueError("Normalize must be True, False, 'sig1', or 'sig2'.")

    out /= normalization_value
    return out


def snr(
    sig1: np.ndarray | list,
    *args: Any,
    data: Optional[np.ndarray | list] = None,
    **kwargs: Any,
) -> float:
    """Compute the snr between two signals weighted by a PSD.

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
        eig_vals, eig_vecs = get_eigens(cov, high_precision=precision)

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


def mismatch_criterion(
    waveform_model,
    params,
    deriv_inds=None,
    inner_product_kwargs={},
    waveform_kwargs={},
    parameter_transforms=None,
    fish=None,
    eps=None,
    eigens=None,
    return_fish=False,
    fisher_kwargs={},
):
    """
    return the mismatch criterion of Vallisneri abs(log r)for a zero noise signal approximation
    and the overlap(h_true, h(0 + delta))/(1- 0.5 *delta gamma delta)
    this is a good check for the fisher matrix approximation

    Args:
        waveform_model (function): Function used to compute GW waveform.
        params (ndarray): Set of GW parameters for the waveform.
        deriv_inds (ndarray): Subset of the GW parameters for which the fisher matrix will be used, specified by indices.
            Defaults to all parameters.
        inner_product_kwargs (dict): Keyword arguments for the inner_product function (optional).
        waveform_kwargs (dict): Keyword arguments for the waveform_model function (optional).
        parameter_transforms (TransformContainer): Parameter transformations to be applied, specified by index (optional).
        fish (ndarray): Pre-computed fisher matrix, as output by fisher(*fisher_args). Must match supplied *fisher_args.
        eps (ndarray or int): Numerical derivative step sizes to use in the fisher matrix calculation, if required.
            If int, the same step size is used for all parameters (not recommended).
        eigens (tuple): tuple of pre-computed eigenvalue and right-eigenvector arrays corresponding
            to the provided fisher matrix.
        return_fish (bool): If True, returns Fisher matrix (defaults to False).
        fisher_kwargs (dict): Further keyword arguments to be passed to the fisher matrix generation function (optional).

    Returns:
        tuple containing

            mismatch (double): Mismatch between perturbed and ML waveforms.
            ratio (double): Computed value of |ln(r)| for this instance of parameter perturbation.
            fish (ndarray): Fisher matrix, if return_fish is set to True.
    """

    # TODO tidy up handling of all of these kwargs!

    params_true = params.copy()

    params_true = parameter_transforms.transform_base_parameters(params_true)

    h_true = waveform_model(*params_true, **waveform_kwargs)

    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_fish_params = len(deriv_inds)

    if fish is None:
        if eps is None:
            print(
                "No numerical derivative step-sizes (eps) supplied for Fisher matrix generation."
            )
            raise Exception

        elif isinstance(eps, float):
            eps = np.full_like(params, eps)

        fish = fisher(
            waveform_model=waveform_model,
            params=params,
            eps=eps,
            deriv_inds=deriv_inds,
            parameter_transforms=parameter_transforms,
            waveform_kwargs=waveform_kwargs,
            inner_product_kwargs=inner_product_kwargs,
            **fisher_kwargs,
        )

    try:
        fish = (
            fish.get()
        )  # This works both for use_gpu=True and for passing in a numpy Fisher matrix
    except AttributeError:
        pass

    if eigens is None:
        w, v = np.linalg.eig(fish)
    else:
        w, v = eigens

    d = num_fish_params
    vec_delta = np.zeros(d)
    u = np.random.normal(0, 1, d)  # an array of d normally distributed random variables
    norm = np.sum(u**2) ** (0.5)
    # r = (np.random.uniform(0, 1)) ** (1.0 / d)  - i dont know why this is here if we are sampling from sphere surface
    x = u / norm  # r * u / norm
    # MISMATCH vector
    for l in range(0, d):
        vec_delta += x[l] * v[:, l] / np.sqrt(w[l])

    # signal perturbed
    var_p_eps = params.copy()
    for i, ind in enumerate(deriv_inds):  # for only the considered variables
        var_p_eps[ind] = var_p_eps[ind].copy() + vec_delta[i]

    var_p_eps = parameter_transforms.transform_base_parameters(var_p_eps)

    # properly treat boundary conditions for polar angles
    for val in [7, 9]:
        var_p_eps[val] = var_p_eps[val] % (2 * np.pi)  # wrap the angle to [0, 2pi]
        if var_p_eps[val] > np.pi:
            var_p_eps[val] = (
                2 * np.pi - var_p_eps[val]
            )  # reflect polar angle on boundary
            # if val == 7:  # if qK, both phiS and phiK must be flipped which cancels out. Still need to flip for qS --- not sure about this, actually
            var_p_eps[val + 1] += np.pi  # flip azimuthal angle

    h_delta = waveform_model(*var_p_eps, **waveform_kwargs)

    # get the derivative
    prod = np.sum(
        [
            [
                fish[j, k] * vec_delta[j] * vec_delta[k]
                for j in range(0, num_fish_params)
            ]
            for k in range(0, num_fish_params)
        ]
    )

    # X = window_zero(h_delta,var_p_eps[7])

    # Y = h_true
    over = inner_product(
        [h_delta.real, h_delta.imag],
        [h_true.real, h_true.imag],
        **inner_product_kwargs,
        normalize=True,
    )
    ratio = over / (
        1
        - 0.5
        * prod
        / inner_product(
            [h_true.real, h_true.imag],
            [h_true.real, h_true.imag],
            **inner_product_kwargs,
            normalize=False,
        )
    )
    mism = (1.0 - over) / 2.0

    if not return_fish:
        return mism, ratio
    elif return_fish:
        return mism, ratio, fish


def get_eigens(arr, high_precision=False):
    """Performs eigenvalue decomposition and returns the eigenvalues and right-eigenvectors for the supplied fisher/covariance matrix.

    Args:
        arr (ndarray): Input matrix for which to perform eigenvalue decomposition.
        high_precision (bool): If True, use 500-dps precision to ensure accurate eigenvalue decomposition
            (requires mpmath to be installed). Defaults to False.
    Returns:
        tuple containing

            evals (ndarray): Eigenvalues for the supplied array.
            evects (ndarray): Right-eigenvectors for the supplied array, constructed such that evects[:,k] corresponds
                to the evals[k].
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
        E, EL, ER = mp.eig(hp_arr, left=True, right=True)

        evals = np.array(E, dtype=np.float64)
        evects = np.array(ER.tolist(), dtype=np.float64)

        return evals, evects

    else:
        evals, evects = np.linalg.eig(arr)

        return evals, evects


def vallisneri_criterion_cdf(
    *mismatch_args,
    num_samples=100,
    return_cdf=True,
    return_ratios=False,
    fish=None,
    precision=False,
    fisher_kwargs={},
    **mismatch_kwargs,
):
    """Generates CDF of 1-sigma isoprobability contour mismatches abs(log(r)) as in Vallisneri (2008) and reads off the 90th percentile value.

    Args:
        *mismatch_args: Arguments to pass to the mismatch_criterion() function call.
        num_samples (int, optional): number of parameter samples to generate (defaults to 100).
        return_cdf (bool): If True, returns the CDF quantiles and values. Defaults to True.
        return_ratios (bool): If True, returns the set of drawn |ln(r)| samples. Defaults to False.
        fish (ndarray): Fisher matrix corresponding to the input event parameters (optional - will be generated if not provided).
        precision (bool): If True, use 500-dps precision to compute eigenvalues/eigenvectors (requires mpmath). Defaults to False.
        fisher_kwargs (dict): Keyword arguments to pass to the fisher matrix generation function (optional).

    Returns:
        tuple containing

            r_at_90 (double): 90th percentile value of the maximum-mismatch CDF.
            quantiles (ndarray): Cumulative distribution function quantiles.
            cdf (ndarray): Cumulative distribution function values.
            ratios (ndarray): Set of drawn |ln(r)| samples.
    """

    ratios = np.zeros(num_samples)

    if fish is None:
        fish = fisher(*mismatch_args, **fisher_kwargs)

    # handle CuPy use in Fisher matrix generation
    try:
        fish = fish.get()
    except AttributeError:
        pass

    w, v = get_eigens(fish, high_precision=precision)

    eigens = (w, v)

    j = 0
    while j < num_samples:
        try:
            mism, ratio = mismatch_criterion(
                *mismatch_args, fish=fish, eigens=eigens, **mismatch_kwargs
            )
            ratios[j] = abs(np.log(ratio))
            j += 1
        except Exception as err:
            print("Error generating CDF sample: ", err)
            raise

    quantiles, counts = np.unique(ratios, return_counts=True)
    cdf = np.cumsum(counts).astype(np.double) / ratios.size

    r_at_90 = np.interp(0.9, cdf, quantiles)

    out = (r_at_90,)
    if return_cdf:
        out += (
            quantiles,
            cdf,
        )
    if return_ratios:
        out += (ratios,)

    return out


def cutler_vallisneri_bias(
    waveform_model_true,
    waveform_model_approx,
    params,
    eps,
    in_diagnostics=None,
    fish=None,
    deriv_inds=None,
    return_fisher=False,
    return_derivs=False,
    return_cov=False,
    parameter_transforms=None,
    waveform_true_kwargs={},
    waveform_approx_kwargs={},
    inner_product_kwargs={},
):
    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_fish_params = len(deriv_inds)

    if isinstance(eps, float):
        eps = np.full_like(params, eps)

    params_true = params.copy()
    params_true = parameter_transforms.transform_base_parameters(params_true)

    if in_diagnostics is None:
        h_true = waveform_model_true(*params_true, **waveform_true_kwargs)

        cov, fish, dh = covariance(
            waveform_model_true,
            params,
            eps,
            return_fisher=True,
            return_derivs=True,
            deriv_inds=deriv_inds,
            parameter_transforms=parameter_transforms,
            waveform_kwargs=waveform_true_kwargs,
            inner_product_kwargs=inner_product_kwargs,
        )

    else:
        cov = in_diagnostics["cov"]
        h_true = in_diagnostics["h_true"]
        dh = in_diagnostics["dh"]

    h_approx = waveform_model_approx(*params_true, **waveform_approx_kwargs)

    diff = h_true - h_approx

    syst_vec = np.array(
        [
            inner_product(
                [dh[k, :].real, dh[k, :].imag],
                [diff.real, diff.imag],
                **inner_product_kwargs,
            )
            for k in range(num_fish_params)
        ]
    )

    bias = np.dot(cov, syst_vec)

    if True not in [return_fisher, return_cov, return_derivs]:
        return syst_vec, bias

    returns = [syst_vec, bias]
    if return_fisher:
        returns.append(fish)

    if return_cov:
        returns.append(cov)

    if return_derivs:
        returns.append(dh)

    return returns


def scale_snr(target_snr, sig, *snr_args, return_orig_snr=False, **snr_kwargs):
    snr_out = snr(sig, *snr_args, **snr_kwargs)

    factor = target_snr / snr_out

    if isinstance(sig, list) is False:
        sig = [sig]

    out = [sig_i * factor for sig_i in sig]
    if return_orig_snr:
        return (out, snr_out)

    return out
