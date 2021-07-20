import warnings

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    pass

from lisatools.sensitivity import get_sensitivity


def inner_product(
    sig1,
    sig2,
    dt=None,
    df=None,
    f_arr=None,
    PSD="lisasens",
    PSD_args=(),
    PSD_kwargs={},
    normalize=False,
    use_gpu=False,
    complex=False,
):
    if use_gpu:
        xp = cp

    else:
        xp = np

    if df is None and dt is None and f_arr is None:
        raise ValueError("Must provide either df, dt or f_arr keyword arguments.")

    if isinstance(sig1, list) is False:
        sig1 = [sig1]

    if isinstance(sig2, list) is False:
        sig2 = [sig2]

    if len(sig1) != len(sig2):
        raise ValueError(
            "Signal 1 has {} channels. Signal 2 has {} channels. Must be equal.".format(
                len(sig1), len(sig2)
            )
        )

    if dt is not None:

        if len(sig1[0]) != len(sig2[0]):
            warnings.warn(
                "The two signals are two different lengths in the time domain. Zero padding smaller array."
            )

            length = len(sig1[0]) if len(sig1[0]) > len(sig2[0]) else len(sig2[0])

            sig1 = [xp.pad(sig, (0, length - len(sig1[0]))) for sig in sig1]
            sig2 = [xp.pad(sig, (0, length - len(sig2[0]))) for sig in sig2]

        length = len(sig1[0])

        freqs = xp.fft.rfftfreq(length, dt)[1:]

        ft_sig1 = [
            xp.fft.rfft(sig)[1:] * dt for sig in sig1
        ]  # remove DC / dt factor helps to cast to proper dimensionality
        ft_sig2 = [xp.fft.rfft(sig)[1:] * dt for sig in sig2]  # remove DC

    else:
        ft_sig1 = sig1
        ft_sig2 = sig2

        if df is not None:
            freqs = (xp.arange(len(sig1[0])) + 1) * df  # ignores DC component using + 1

        else:
            freqs = f_arr

    # get PSD weighting
    if isinstance(PSD, str):
        PSD_arr = get_sensitivity(freqs, sens_fn=PSD, *PSD_args, **PSD_kwargs)

    elif isinstance(PSD, xp.ndarray):
        PSD_arr = PSD

    elif PSD is None:
        PSD_arr = 1.0

    else:
        raise ValueError(
            "PSD must be a string giving the sens_fn or a predetermimed array or None if noise weighting is included in a signal."
        )

    out = 0.0
    # assumes right summation rule
    x_vals = xp.zeros(len(PSD_arr))

    x_vals[1:] = xp.diff(freqs)
    x_vals[0] = x_vals[1]

    # account for hp and hx if included in time domain signal
    for temp1, temp2 in zip(ft_sig1, ft_sig2):
        if complex:
            func = lambda x: x
        else:
            func = xp.real
        y = func(temp1.conj() * temp2) / PSD_arr  # assumes right summation rule

        out += 4 * xp.sum(x_vals * y)

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
            use_gpu=use_gpu,
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
            use_gpu=use_gpu,
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


def snr(sig1, *args, data=None, use_gpu=False, **kwargs):
    if use_gpu:
        xp = cp

    else:
        xp = np

    if data is None:
        sig2 = sig1

    else:
        sig2 = data

    return xp.sqrt(inner_product(sig1, sig2, *args, use_gpu=use_gpu, **kwargs))


def h_var_p_eps(
    waveform_model, params, step, i, parameter_transforms=None, waveform_kwargs={}
):
    """
    Calculate the waveform with a perturbation step of the variable V[i]
    """
    params_p_eps = params.copy()
    params_p_eps[i] += step

    if parameter_transforms:
        # transform
        params_p_eps = parameter_transforms.transform_base_parameters(params_p_eps)

    dh = waveform_model(*params_p_eps, **waveform_kwargs)

    return dh


def dh_dlambda(
    waveform_model,
    params,
    eps,
    i,
    parameter_transforms=None,
    waveform_kwargs={},
    accuracy=True,
):
    """
    Calculate the derivative of the waveform with precision of order (step^4)
    with respect to the variable V in the i direction
    """
    if accuracy:
        # Derivative of the Waveform
        # up
        h_I_up_2eps = h_var_p_eps(
            waveform_model,
            params,
            2 * eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )
        h_I_up_eps = h_var_p_eps(
            waveform_model,
            params,
            eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )
        # down
        h_I_down_2eps = h_var_p_eps(
            waveform_model,
            params,
            -2 * eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )
        h_I_down_eps = h_var_p_eps(
            waveform_model,
            params,
            -eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )

        ind_max = np.min(
            [len(h_I_up_2eps), len(h_I_up_eps), len(h_I_down_2eps), len(h_I_down_eps)]
        )
        # print([len(h_I_up_2eps), len(h_I_up_eps), len(h_I_down_2eps), len(h_I_down_eps)])

        # error scales as eps^4
        dh_I = (
            -h_I_up_2eps[:ind_max]
            + h_I_down_2eps[:ind_max]
            + 8 * (h_I_up_eps[:ind_max] - h_I_down_eps[:ind_max])
        ) / (12 * eps)
        # Time thta it takes for one variable: approx 5 minutes
    else:
        # Derivative of the Waveform
        # up
        h_I_up_eps = h_var_p_eps(
            waveform_model,
            params,
            eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )
        # down
        h_I_down_eps = h_var_p_eps(
            waveform_model,
            params,
            -eps,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
        )

        ind_max = np.min([len(h_I_up_eps), len(h_I_down_eps)])
        # print([len(h_I_up_2eps), len(h_I_up_eps), len(h_I_down_2eps), len(h_I_down_eps)])

        # error scales as eps^4
        dh_I = (h_I_up_eps[:ind_max] - h_I_down_eps[:ind_max]) / (2 * eps)
        # Time thta it takes for one variable: approx 5 minutes

    return dh_I


def fisher(
    waveform_model,
    params,
    eps,
    use_gpu=False,
    deriv_inds=None,
    parameter_transforms=None,
    waveform_kwargs={},
    inner_product_kwargs={},
    return_derivs=False,
    accuracy=True,
):
    """

    Args:
        waveform_model (function): Function used to compute GW waveform.
        params (ndarray): Set of GW waveform parameters.
        eps (ndarray): Numerical derivative step-sizes (must be same size as the params array).
        use_gpu (bool): If True, uses the GPU. Defaults to False.
        deriv_inds (ndarray): Subset of event parameters for which to calculate the Fisher matrix, by index.
            Defaults to all parameters.
        parameter_transforms (TransformContainer): Transformations to be applied to parameters, by index (optional).
        waveform_kwargs (dict): Keyword arguments for waveform_model (optional).
        inner_product_kwargs (dict): Keyword arguments for the inner product function (optional).
        return_derivs (bool): If True, also returns computed numerical derivatives. Defaults to False.
        accuracy (bool): If True, uses five-point stencil to compute numerical derivative as opposed to the 3-point stencil.
            Defaults to True.

    Returns:
        tuple containing

            fish (ndarray): Fisher Matrix for the GW waveform parameters supplied.
            dh (ndarray): Waveform derivatives computed as part of the Fisher matrix calculation,
                if return_derivs is set to True.
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    # is for inner product

    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_fish_params = len(deriv_inds)

    if isinstance(eps, float):
        eps = np.full_like(params, eps)

    dh = []
    for i, eps_i in zip(deriv_inds, eps):
        # derivative up
        temp = dh_dlambda(
            waveform_model,
            params,
            eps_i,
            i,
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=parameter_transforms,
            accuracy=accuracy,
        )
        dh.append(temp)

        # A = window_zero(A,dt)
        # B = window_zero(B,dt)

    dh = xp.asarray(dh)

    fish = np.zeros((num_fish_params, num_fish_params))
    for i in range(num_fish_params):
        for j in range(i, num_fish_params):
            fish[i][j] = inner_product(
                [dh[i].real, dh[i].imag],
                [dh[j].real, dh[j].imag],
                use_gpu=use_gpu,
                **inner_product_kwargs
            )
            fish[j][i] = fish[i][j]

    if return_derivs:
        return fish, dh
    else:
        return fish


def covariance(
    *fisher_args,
    fish=None,
    diagonalize=False,
    return_fisher=False,
    precision=False,
    **fisher_kwargs
):
    """Calculate covariance matrix for a set of EMRI parameters, computing the fisher matrix if not supplied.

    Args:
        *fisher_args: Set of arguments to pass to fisher(). Not required if fish is not None.
        fish (ndarray): Pre-computed fisher matrix (optional). If supplied, this matrix will be inverted.
        diagonalize (bool): If True, diagonalizes the covariance matrix. Defaults to False.
        return_fisher (bool): If True, also returns the computed fisher matrix.
        precision (bool): If True, uses 500-dps precision to compute the fisher matrix inverse (requires mpmath). This is typically a good idea as the Fisher matrix can be highly ill-conditioned. Defaults to False.
        **fisher_kwargs: Keyword arguments to pass to fisher().
    """

    if fish is None:
        fish = fisher(*fisher_args, **fisher_kwargs)

    if "return_derivs" in fisher_kwargs:
        return_derivs = True
        fish, dh = fish

    else:
        return_derivs = False

    if precision:
        try:
            import mpmath as mp

            mp.mp.dps = 500
        except ModuleNotFoundError:
            print("mpmath module not installed. Defaulting to low precision...")
            precision = False
    if precision:
        hp_fish = mp.matrix(fish.tolist())
        U, S, V = mp.svd_r(hp_fish)  # singular value decomposition
        temp = mp.diag([val ** (-1) for val in S])  # get S**-1
        temp2 = V.T * temp * U.T  # construct pseudo-inverse
        cov = np.array(temp2.tolist(), dtype=np.float64)
    else:
        cov = np.linalg.pinv(fish)

    if diagonalize:
        eig_vals, eig_vecs = get_eigens(cov, high_precision=precision)

        cov = np.dot(np.dot(eig_vecs.T, cov), eig_vecs)

    if True not in [return_fisher, return_derivs]:
        return cov

    returns = [cov]

    if return_fisher:
        returns.append(fish)

    if return_derivs:
        returns.append(dh)

    return returns


def plot_corner(
    params,
    cov,
    nsamp=25000,
    filename="corner.png",
    savefig_kwargs={},
    corner_plot_kwargs={},
):
    """Construct a corner plot for a given covariance matrix (requires the corner module).

    Args:
        params (ndarray): The set of parameters used for the event (the mean vector of the covariance matrix).
        cov (ndarray): Covariance matrix from which to construct the corner plot.
        nsamp (int): Number of samples to draw from the multivariate distribution (defaults to 25000).
        filename (str): Filename for the output figure to be saved under. Defaults to 'corner.png' in the cwd.
        savefig_kwargs (dict): Dictionary of keyword arguments to pass to matplotlib.pyplot.savefig(), such as dpi. (optional)
        corner_plot_kwargs (dict): Keyword arguments for the corner plot - see the module documentation for more info.
    Returns:
        figure (figure object): The corner plot figure.

    """

    import corner

    if corner_plot_kwargs == {}:
        print("No corner plot kwargs supplied - using defaults.")
        corner_plot_kwargs = dict(show_titles=True, title_fmt=".2e",)

    samp = np.random.multivariate_normal(params, cov, size=nsamp)
    fig = corner.corner(samp, **corner_plot_kwargs)
    fig.savefig(filename, **savefig_kwargs)

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
            **fisher_kwargs
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
    norm = np.sum(u ** 2) ** (0.5)
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
        normalize=True
    )
    ratio = over / (
        1
        - 0.5
        * prod
        / inner_product(
            [h_true.real, h_true.imag],
            [h_true.real, h_true.imag],
            **inner_product_kwargs,
            normalize=False
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
    **mismatch_kwargs
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
                **inner_product_kwargs
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
