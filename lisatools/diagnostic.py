import warnings

import numpy as np

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError):
    import numpy as xp

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
):

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
        y = xp.real(temp1.conj() * temp2) / PSD_arr  # assumes right summation rule

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


def snr(sig1, *args, data=None, **kwargs):
    if data is None:
        sig2 = sig1

    else:
        sig2 = data

    return xp.sqrt(inner_product(sig1, sig2, *args, **kwargs))


def h_var_p_eps(
    waveform_model, params, step, i, parameter_transforms={}, waveform_kwargs={}
):
    """
    Calculate the waveform with a perturbation step of the variable V[i]
    """
    params_p_eps = params.copy()
    params_p_eps[i] += step

    if parameter_transforms != {}:
        for ind, transform_fn in parameter_transforms.items():
            params_p_eps[ind] = transform_fn(params_p_eps[ind])

    dh = waveform_model(*params_p_eps, **waveform_kwargs)

    return dh


def dh_dlambda(
    waveform_model, params, eps, i, parameter_transforms={}, waveform_kwargs={}
):
    """
    Calculate the derivative of the waveform with precision of order (step^4)
    with respect to the variable V in the i direction
    """
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

    return dh_I


def fisher(
    waveform_model,
    params,
    eps,
    deriv_inds=None,
    parameter_transforms={},
    waveform_kwargs={},
    inner_product_kwargs={},
    return_derivs=False,
):
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
                **inner_product_kwargs
            )
            fish[j][i] = fish[i][j]

    if return_derivs:
        return fish, dh
    else:
        return fish


def covariance(*fisher_args, diagonalize=False, return_fisher=False, **fisher_kwargs):
    fish = fisher(*fisher_args, **fisher_kwargs)

    if "return_derivs" in fisher_kwargs:
        return_derivs = True
        fish, dh = fish

    else:
        return_derivs = False

    cov = np.linalg.pinv(fish)

    if diagonalize:
        eig_vals, eig_vecs = np.linalg.eig(cov)

        cov = np.dot(np.dot(eig_vecs.T, cov), eig_vecs)

    if True not in [return_fisher, return_derivs]:
        return cov

    returns = [cov]

    if return_fisher:
        returns.append(fish)

    if return_derivs:
        returns.append(dh)

    return returns


def mismatch_criterion(
    waveform_model,
    params,
    eps,
    deriv_inds=None,
    parameter_transforms={},
    waveform_kwargs={},
    inner_product_kwargs={},
    fish=None,
):
    """
    return the mismatch criterion of Vallisneri abs(log r)for a zero noise signal approximation
    and the overlap(h_true, h(0 + delta))/(1- 0.5 *delta gamma delta)
    this is a good check for the fisher matrix approximation
    """

    params_true = params.copy()

    for ind, transform_fn in parameter_transforms.items():
        params_true[ind] = transform_fn(params_true[ind])

    h_true = waveform_model(*params_true, **waveform_kwargs)

    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_fish_params = len(deriv_inds)

    if isinstance(eps, float):
        eps = np.full_like(params, eps)

    if fish is None:
        fish = fisher(
            waveform_model,
            params,
            eps,
            deriv_inds=deriv_inds,
            parameter_transforms=parameter_transforms,
            waveform_kwargs=waveform_kwargs,
            inner_product_kwargs=inner_product_kwargs,
        )

    w, v = np.linalg.eig(fish)
    d = num_fish_params
    vec_delta = np.zeros_like(eps)
    u = np.random.normal(0, 1, d)  # an array of d normally distributed random variables
    norm = np.sum(u ** 2) ** (0.5)
    r = (np.random.uniform(0, 1)) ** (1.0 / d)
    x = r * u / norm
    # MISMATCH vector
    for l in range(0, d):
        vec_delta = x[l] * v[:, l] / np.sqrt(w[l])

    # signal perturbed
    var_p_eps = params.copy()
    for i, ind in enumerate(deriv_inds):  # for only the considered variables
        var_p_eps[ind] = var_p_eps[ind].copy() + vec_delta[i]

    for ind, transform_fn in parameter_transforms.items():
        var_p_eps[ind] = transform_fn(var_p_eps[ind])

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
            [h_delta.real, h_delta.imag],
            [h_true.real, h_true.imag],
            **inner_product_kwargs,
            normalize=False
        )
    )
    mism = (1.0 - over) / 2.0
    return mism, ratio


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
    parameter_transforms={},
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
    for ind, transform_fn in parameter_transforms.items():
        params_true[ind] = transform_fn(params_true[ind])

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
                x,
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
