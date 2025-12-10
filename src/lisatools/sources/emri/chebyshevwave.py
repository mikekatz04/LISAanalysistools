from __future__ import annotations
import sys
import os
import typing

import numpy.typing as npt
#import few
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import pi
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
#rem I need to remind myself which of these I use
from scipy.integrate import cumulative_simpson as cumulative_simpson_scipy
#from emri_singleharmonic_funcs import *

def alpha_to_tf(alpha,npts,fmin,fmax):
#This function takes in the Chebyshev coeffs \vec \alpha (alpha_0
# thru \alpha4) and calculates t(f) on a 1-d grid where the f values are
#evenly spaced in lnf.  npts is the number of pts in the grid.     
#The "zero of time" convention here is that
#t=0 when f = fmid, where fmid is the geometric mean of fmin and fmax.   
#As an intermediate step, it calculates t(y), where
# y is basically ln(f), but squeezed and shifted so it goes from y= -1 and y=1.
    
#   from emri_singleharmonic_funcs import chebyfit   #probably dont need this
#Msun = 4.925490949197807e-06  #mass of sun in sec  #prob don't need this either
    fmid = np.sqrt(fmin*fmax)
    kappa = 0.5*(np.log(fmax/fmin))
    y = np.linspace(-1.e0, 1.e0, num=npts, endpoint=True)
    nterms = np.size(alpha)
    basis_funcs = np.zeros([npts,nterms])  #these are the Chebyshev polys (of 2nd kind)
    y0 = (y + 2.0)**0.0
    basis_funcs[:,0] = y0
    y1 = 2*y
    basis_funcs[:,1] = y1
    y2 = y1**2.0 -1.0
    basis_funcs[:,2] = y2
    y3 = y1*(y2-1.0)
    basis_funcs[:,3] = y3
    y4 = y1*y3 - y2
    basis_funcs[:,4] = y4
    dw = 2.0/(npts-1)
    midpt = (npts-1)//2
    #print('midpt =',midpt)
    input1 = np.matmul(basis_funcs,alpha) + kappa*y
    input1.shape = (npts,1)
    pre_integrand =np.zeros([npts,1])
    pre_integrand = np.exp(input1) #using nterm chebyshev fit to y
#We call this the pre-integrand since it was stripped from a code     
    exp_kappa_y = np.exp(kappa*y)
    f = fmid*exp_kappa_y
    exp_kappa_y.shape = (npts,1)
    integrand = np.zeros([npts,1])
    integrand[:,0] = dw*cumulative_simpson_scipy(pre_integrand[:,0], initial=0.e0)
    integrand.shape = (npts,1)
    const1 = integrand[midpt,0]
    integrand = integrand - np.ones([npts,1])*const1
    #integrand = integrand*exp_kappa_y*2.0*pi*kappa*kappa
    t = integrand*kappa
    #Psi  = np.zeros([npts,1])
    #Psi[:,0] = dw*cumulative_simpson_scipy(integrand[:,0], initial=0.e0)
    #Psi.shape = (npts,1)
    #const2 = Psi[midpt,0]
    #Psi = Psi - const2*np.ones([npts,1])
    return f,t



def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        0 or None are the only values accepted. Default is None, which means
        `res` has one element less than `y` along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum, numpy.cumprod
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    y = np.asarray(y)
    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if initial != 0:
            raise ValueError("`initial` must be `None` or `0`.")
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    return res


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None):
    r"""
    Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.
    The integral of the samples at every point is calculated by assuming a 
    quadratic relationship between each point and the two adjacent points.

    Parameters
    ----------
    y : array_like
        Values to integrate. Requires at least one point along `axis`. If two or fewer
        points are provided along `axis`, Simpson's integration is not possible and the
        result is calculated with `cumulative_trapezoid`.
    x : array_like, optional
        The coordinate to integrate along. Must have the same shape as `y` or
        must be 1D with the same length as `y` along `axis`. `x` must also be
        strictly increasing along `axis`.
        If `x` is None (default), integration is performed using spacing `dx`
        between consecutive elements in `y`.
    dx : scalar or array_like, optional
        Spacing between elements of `y`. Only used if `x` is None. Can either 
        be a float, or an array with the same shape as `y`, but of length one along
        `axis`. Default is 1.0.
    axis : int, optional
        Specifies the axis to integrate along. Default is -1 (last axis).
    initial : scalar or array_like, optional
        If given, insert this value at the beginning of the returned result,
        and add it to the rest of the result. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration. Can either be a float, or an array with
        the same shape as `y`, but of length one along `axis`.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum
    cumulative_trapezoid : cumulative integration using the composite 
        trapezoidal rule
    simpson : integrator for sampled data using the Composite Simpson's Rule

    Notes
    -----

    .. versionadded:: 1.12.0

    The composite Simpson's 1/3 method can be used to approximate the definite 
    integral of a sampled input function :math:`y(x)` [1]_. The method assumes 
    a quadratic relationship over the interval containing any three consecutive
    sampled points.

    Consider three consecutive points: 
    :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`.

    Assuming a quadratic relationship over the three points, the integral over
    the subinterval between :math:`x_1` and :math:`x_2` is given by formula
    (8) of [2]_:
    
    .. math::
        \int_{x_1}^{x_2} y(x) dx\ &= \frac{x_2-x_1}{6}\left[\
        \left\{3-\frac{x_2-x_1}{x_3-x_1}\right\} y_1 + \
        \left\{3 + \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} + \
        \frac{x_2-x_1}{x_3-x_1}\right\} y_2\\
        - \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} y_3\right]

    The integral between :math:`x_2` and :math:`x_3` is given by swapping
    appearances of :math:`x_1` and :math:`x_3`. The integral is estimated
    separately for each subinterval and then cumulatively summed to obtain
    the final result.
    
    For samples that are equally spaced, the result is exact if the function
    is a polynomial of order three or less [1]_ and the number of subintervals
    is even. Otherwise, the integral is exact for polynomials of order two or
    less. 

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Simpson's_rule
    .. [2] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
            MS Excel and Irregularly-spaced Data. Journal of Mathematical
            Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x**2
    >>> y_int = integrate.cumulative_simpson(y, x=x, initial=0)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y_int, 'ro', x, x**3/3 - (x[0])**3/3, 'b-')
    >>> ax.grid()
    >>> plt.show()

    The output of `cumulative_simpson` is similar to that of iteratively
    calling `simpson` with successively higher upper limits of integration, but
    not identical.

    >>> def cumulative_simpson_reference(y, x):
    ...     return np.asarray([integrate.simpson(y[:i], x=x[:i])
    ...                        for i in range(2, len(y) + 1)])
    >>>
    >>> rng = np.random.default_rng(354673834679465)
    >>> x, y = rng.random(size=(2, 10))
    >>> x.sort()
    >>>
    >>> res = integrate.cumulative_simpson(y, x=x)
    >>> ref = cumulative_simpson_reference(y, x)
    >>> equal = np.abs(res - ref) < 1e-15
    >>> equal  # not equal when `simpson` has even number of subintervals
    array([False,  True, False,  True, False,  True, False,  True,  True])

    This is expected: because `cumulative_simpson` has access to more
    information than `simpson`, it can typically produce more accurate
    estimates of the underlying integral over subintervals.

    """
    assert y.dtype == float

    # validate `axis` and standardize to work along the last axis
    original_y = y
    original_shape = y.shape
    try:
        y = np.swapaxes(y, axis, -1)
    except IndexError as e:
        message = f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
        raise ValueError(message) from e
    if y.shape[-1] < 3:
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        res = np.swapaxes(res, axis, -1)

    elif x is not None:
        x = _ensure_float_array(x)
        message = ("If given, shape of `x` must be the same as `y` or 1-D with "
                   "the same length as `y` along `axis`.")
        if not (x.shape == original_shape
                or (x.ndim == 1 and len(x) == original_shape[axis])):
            raise ValueError(message)

        x = np.broadcast_to(x, y.shape) if x.ndim == 1 else np.swapaxes(x, axis, -1)
        dx = np.diff(x, axis=-1)
        if np.any(dx <= 0):
            raise ValueError("Input x must be strictly increasing.")
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_unequal_intervals
        )

    else:
        dx = _ensure_float_array(dx)
        final_dx_shape = tupleset(original_shape, axis, original_shape[axis] - 1)
        alt_input_dx_shape = tupleset(original_shape, axis, 1)
        message = ("If provided, `dx` must either be a scalar or have the same "
                   "shape as `y` but with only 1 point along `axis`.")
        if not (dx.ndim == 0 or dx.shape == alt_input_dx_shape):
            raise ValueError(message)
        dx = np.broadcast_to(dx, final_dx_shape)
        dx = np.swapaxes(dx, axis, -1)
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_equal_intervals
        )

    if initial is not None:
        initial = _ensure_float_array(initial)
        alt_initial_input_shape = tupleset(original_shape, axis, 1)
        message = ("If provided, `initial` must either be a scalar or have the "
                   "same shape as `y` but with only 1 point along `axis`.")
        if not (initial.ndim == 0 or initial.shape == alt_initial_input_shape):
            raise ValueError(message)
        initial = np.broadcast_to(initial, alt_initial_input_shape)
        initial = np.swapaxes(initial, axis, -1)

        res += initial
        res = np.concatenate((initial, res), axis=-1)

    res = np.swapaxes(res, -1, axis)
    return res



def _cumulatively_sum_simpson_integrals(
    y: np.ndarray, 
    dx: np.ndarray, 
    integration_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Calculate cumulative sum of Simpson integrals.
    Takes as input the integration function to be used. 
    The integration_func is assumed to return the cumulative sum using
    composite Simpson's rule. Assumes the axis of summation is -1.
    """
    sub_integrals_h1 = integration_func(y, dx)
    sub_integrals_h2 = integration_func(y[..., ::-1], dx[..., ::-1])[..., ::-1]
    
    shape = list(sub_integrals_h1.shape)
    shape[-1] += 1
    sub_integrals = np.empty(shape)
    sub_integrals[..., :-1:2] = sub_integrals_h1[..., ::2]
    sub_integrals[..., 1::2] = sub_integrals_h2[..., ::2]
    # Integral over last subinterval can only be calculated from 
    # formula for h2
    sub_integrals[..., -1] = sub_integrals_h2[..., -1]
    res = np.cumsum(sub_integrals, axis=-1)
    return res


def _cumulative_simpson_equal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming equal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    # Calculate integral over the subintervals (eqn (10) of Reference [2])
    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)


def _cumulative_simpson_unequal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming unequal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    x21 = dx[..., :-1]
    x32 = dx[..., 1:]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    x31 = x21 + x32
    x21_x31 = x21/x31
    x21_x32 = x21/x32
    x21x21_x31x32 = x21_x31 * x21_x32

    # Calculate integral over the subintervals (eqn (8) of Reference [2])
    coeff1 = 3 - x21_x31
    coeff2 = 3 + x21x21_x31x32 + x21_x31
    coeff3 = -x21x21_x31x32

    return x21/6 * (coeff1*f1 + coeff2*f2 + coeff3*f3)


def _ensure_float_array(arr: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float, copy=False)
    return arr


class ChebyshevWave:
    
    def __init__(self, npts: int,fmin: float,fmax: float):

        self.npts, self.fmin, self.fmax = npts,fmin,fmax
        
        self.fmid = np.sqrt(self.fmin*self.fmax)
        self.kappa = 0.5*(np.log(self.fmax/self.fmin))
        self.y = np.linspace(-1.e0, 1.e0, num=self.npts, endpoint=True)
        self.kappa_y = self.kappa * self.y
        self.exp_kappa_y = np.exp(self.kappa_y)
        self.dw = 2.0/(self.npts-1)
        self.midpt = (self.npts-1)//2

    def __call__(self, *alpha_in: np.ndarray | list) -> typing.Tuple[np.ndarray, np.ndarray]:
        
        alpha = np.asarray(alpha_in).T
        squeeze = (alpha.ndim == 1)
        alpha = np.atleast_2d(alpha)
        
        assert alpha.ndim < 3

        nsource, nterms = alpha.shape
        basis_funcs = np.zeros([nsource, self.npts, nterms])  #these are the Chebyshev polys (of 2nd kind)
        y0 = (self.y + 2.0)**0.0
        basis_funcs[:, :,0] = y0[None, :]
        y1 = 2*self.y
        basis_funcs[:, :,1] = y1[None, :]
        y2 = y1**2.0 -1.0
        basis_funcs[:, :,2] = y2[None, :]
        y3 = y1*(y2-1.0)
        basis_funcs[:, :,3] = y3[None, :]
        y4 = y1*y3 - y2
        basis_funcs[:, :,4] = y4[None, :]

        #print('midpt =',midpt)
        # faster than einsum?
        input1 = (np.einsum("...ij,...j->...i", basis_funcs, alpha) + self.kappa_y)

        pre_integrand = np.exp(input1) #using nterm chebyshev fit to y
    #We call this the pre-integrand since it was stripped from a code     
        f = self.fmid*self.exp_kappa_y
        
        # test_pre_integrand = np.tile(pre_integrand, (9, 1))
        
        # _integrand = self.dw*cumulative_simpson_scipy(test_pre_integrand, initial=0.e0, axis=-1)
        integrand = self.dw*cumulative_simpson(pre_integrand, initial=0.e0, axis=-1)
        const1 = integrand[:, self.midpt]
        integrand -= const1[:, None]
        #integrand = integrand*exp_kappa_y*2.0*pi*kappa*kappa
        t = integrand*self.kappa
        if squeeze:
            return t[0], f
        return t, f
    

from fastlisaresponse.tdionfly import FDTDIonTheFly
from gpubackendtools.interpolate import CubicSplineInterpolant


class ChebyshevXYZ(ChebyshevWave):
    
    def __init__(self, fs, *args, **kwargs):
        ChebyshevWave.__init__(self, *args, **kwargs)
        # Need to initialize FDTDIonTheFly each time through
        self.fd_gen = FDTDIonTheFly
        self.fs = fs

    def __call__(self, amp0, inc, psi, lam, beta, t_start, *alpha, return_spline: bool =False, **cheb_kwargs):
        
        t_intrinsic, _f = ChebyshevWave.__call__(self, *alpha, **cheb_kwargs)
        
        squeeze = (t_intrinsic.ndim == 1)
        t_intrinsic = np.atleast_2d(t_intrinsic)
        
        amp0 = np.atleast_1d(amp0)
        inc = np.atleast_1d(inc)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        t_start = np.atleast_1d(t_start)

        num_bin = inc.shape[0]
        f = np.tile(_f, (num_bin, 1))

        assert t_start.ndim == 1 and t_start.shape[0] == inc.shape[0] and t_start.shape[0] == t_intrinsic.shape[0]
        t = t_intrinsic - (t_intrinsic[:, 0] - t_start)[:, None]

        t_max = t.max(axis=-1)
        buffer = 500.0
        t_tdi = np.linspace(t_start + buffer, t_max - buffer, self.npts, axis=-1)
        # print("NEED TO FIX t so that it fits orbits window")
        assert t.ndim == 2 and f.ndim == 2
        f_of_t = CubicSplineInterpolant(t, f)
        A_of_t = CubicSplineInterpolant(t, np.ones_like(t))

        nsources, npts = t.shape

        CUBIC_SPLINE_LINEAR_SPACING = 1
        CUBIC_SPLINE_LOG10_SPACING = 2
        CUBIC_SPLINE_GENERAL_SPACING = 3

        phase_ref = (2 * np.pi * f * t).flatten().copy()
        
        # 11 is nparams / will not affect spline
        # needs to be 4? need to check that
        fd_wave_gen = self.fd_gen(t_tdi, A_of_t, f_of_t, phase_ref, self.fs, nsources, n_params=4, spline_type=CUBIC_SPLINE_GENERAL_SPACING)
        wave_output = fd_wave_gen(inc, psi, lam, beta, return_spline=return_spline)
        
        # will reset splines if splines are desired
        # TODO: remove this for speed?
        if np.any(amp0 != 1.0):
            wave_output.tdi_amp = wave_output.tdi_amp * amp0[:, None, None]
        return wave_output

if __name__ == "__main__":

    alpha = [13.0,-0.80,3.e-1,-1.e-3,2.e-4] #just any old alpha
    npts = 32001  #first do on fine grid

    fmin = 0.003 #3mHz
    fmax = 0.008
    f,t = alpha_to_tf(alpha,npts,fmin,fmax)
    cheb_wave = ChebyshevWave(npts, fmin, fmax)
    t1, f1 = cheb_wave(alpha)
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(t.squeeze(), f.squeeze())

    # ax1.plot(t1, f1, "--")
    # ax2.plot(t1.squeeze() - t.squeeze())
    
    # print(t[(npts-1)//2]) #should be zero
    # plt.show()
    # plt.close()
    # breakpoint()

    cheb_xyz = ChebyshevXYZ(npts, fmin, fmax)

    temp_xyz = cheb_xyz(alpha)
    