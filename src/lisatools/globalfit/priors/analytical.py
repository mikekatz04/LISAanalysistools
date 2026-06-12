"""Module containing analytical priors, i.e. priors that can be evaluated at any point in parameter space, 
and for which we can draw samples from. This includes uniform, Gaussian, and log-uniform priors, as well as 
any other prior for which the user can provide a logpdf and rvs method. These priors are implemented as 
subclasses of the base.py Prior class, which defines the interface for all priors. Most inspiration for
these classes comes form BILBY, but they have been adapted to fit the needs of the global fit, and to be 
more flexible in terms of the parameters they can be applied to and cpu/gpu usage. 
"""


# TODO: Add proper documentation to these classes, and make sure they are consistent with the base Prior class.

from __future__ import annotations

from typing import Any, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

from lisatools.utils.typing import NDArrayLike
from .base import Prior, UniformDistribution

ANGLE_SAFE = 1e-9  # To avoid numerical issues with arccos and arcsin at the boundaries

class DeltaFunction(Prior):
    """Dirac delta function prior. Always returns the peak value."""

    def __init__(
        self,
        peak: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=peak,
            maximum=peak,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.peak = peak

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.full(size, self.peak, dtype=self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr == self.peak, self.xp.inf, -self.xp.inf)
        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr >= self.peak, 1.0, 0.0)
        return self._to_device(out)


class PowerLaw(UniformDistribution):
    """
    Generalized Power Law distribution.
    
    Sampler space: Uniform in u = x^(alpha + 1).
    Physical space: x distributed as p(x) ~ x^alpha in [minimum, maximum].
    """
    def __init__(
        self,
        alpha: float,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if alpha == -1:
            raise ValueError("For alpha=-1, use the LogUniform class instead.")
        if minimum < 0 and (alpha + 1) % 1 != 0:
            raise ValueError("Fractional powers of negative numbers are undefined.")

        self.alpha = alpha
        self._power = alpha + 1.0
        self._inv_power = 1.0 / self._power
        self._log_abs_power = self.xp.log(self.xp.abs(self._power))

        # The bounds in u-space can flip if the power is negative.
        # e.g., x in [1, 10] with alpha=-2 -> u in [1, 0.1]. Uniform needs min < max.
        u_bound_1 = minimum ** self._power
        u_bound_2 = maximum ** self._power

        super().__init__(
            minimum=min(u_bound_1, u_bound_2),
            maximum=max(u_bound_1, u_bound_2),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike) -> NDArrayLike:
        """Physical x -> Sampling u = x^(alpha+1)"""
        return x_phys ** self._power

    def _inverse(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Sampling u -> Physical x = u^(1/(alpha+1))"""
        return u_samp ** self._inv_power

    def _jacobian(self, x_phys: NDArrayLike) -> NDArrayLike:
        """
        u = x^(alpha+1)  ->  du/dx = (alpha+1) * x^alpha
        log |du/dx| = ln|alpha+1| + alpha * ln(x)
        """
        # We use xp.where to safely mask negative or zero values before logging
        x_phys = self.xp.asarray(x_phys)
        valid_x = self.xp.where(x_phys > 0, x_phys, 1.0)
        out = self._log_abs_power + self.alpha * self.xp.log(valid_x)
        return self._to_device(out)


class LogUniform(UniformDistribution):
    """
    Log-Uniform distribution.
    
    Sampler space: Uniform in u = ln(x) between [ln(minimum), ln(maximum)].
    Physical space: x distributed as p(x) ~ 1/x between [minimum, maximum].
    """
    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("LogUniform minimum value must be strictly positive.")

        # Initialize the Uniform base class in the log-space
        super().__init__(
            minimum=float(np.log(minimum)),
            maximum=float(np.log(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        # Attach transforms (Physical x <-> Sampling u)
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike) -> NDArrayLike:
        """Physical x -> Sampling u = ln(x)"""
        x_phys = self.xp.asarray(x_phys)
        out = self.xp.log(x_phys)
        return self._to_device(out)

    def _inverse(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Sampling u -> Physical x = exp(u)"""
        u_samp = self.xp.asarray(u_samp)
        out = self.xp.exp(u_samp)
        return self._to_device(out)

    def _jacobian(self, x_phys: NDArrayLike) -> NDArrayLike:
        """
        log | du / dx | = log(1 / x) = -log(x)
        """
        x_phys = self.xp.asarray(x_phys)
        out = -self.xp.log(x_phys)
        return self._to_device(out)


class CosineUniform(UniformDistribution):
    """
    Uniform in Cosine. (Matches Bilby's 'Sine' Prior).
    
    Used for inclinations (iota) or colatitudes.
    Sampler space: Uniform in u = cos(iota) in [-1, 1].
    Physical space: iota distributed as p(iota) ~ sin(iota) in [0, pi].
    """
    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        # Initialize Uniform in u = cos(iota) between [-1, 1]
        super().__init__(
            minimum=-1.0+ANGLE_SAFE,
            maximum=1.0-ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, iota_phys: NDArrayLike) -> NDArrayLike:
        """Physical iota -> Sampling u = cos(iota)"""
        iota_phys = self.xp.asarray(iota_phys)
        out = self.xp.cos(iota_phys)
        return self._to_device(out)

    def _inverse(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Sampling u -> Physical iota = arccos(u)"""
        u_samp = self.xp.asarray(u_samp)
        out = self.xp.arccos(u_samp)
        return self._to_device(out)

    def _jacobian(self, iota_phys: NDArrayLike) -> NDArrayLike:
        """
        log | du / diota | = log( |-sin(iota)| ) = log(sin(iota))
        Valid because iota is in [0, pi], so sin(iota) is positive.
        """
        iota_phys = self.xp.asarray(iota_phys)
        out = self.xp.log(self.xp.sin(iota_phys))
        return self._to_device(out)


class SineUniform(UniformDistribution):
    """
    Uniform in Sine. (Matches Bilby's 'Cosine' Prior).
    
    Used for latitudes, declinations, or elevations (beta).
    Sampler space: Uniform in u = sin(beta) in [-1, 1].
    Physical space: beta distributed as p(beta) ~ cos(beta) in [-pi/2, pi/2].
    """
    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        # Initialize Uniform in u = sin(beta) between [-1, 1]
        super().__init__(
            minimum=-1.0+ANGLE_SAFE,
            maximum=1.0-ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, beta_phys: NDArrayLike) -> NDArrayLike:
        """Physical beta -> Sampling u = sin(beta)"""
        beta_phys = self.xp.asarray(beta_phys)
        out = self.xp.sin(beta_phys)
        return self._to_device(out)

    def _inverse(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Sampling u -> Physical beta = arcsin(u)"""
        u_samp = self.xp.asarray(u_samp)
        out = self.xp.arcsin(u_samp)
        return self._to_device(out)

    def _jacobian(self, beta_phys: NDArrayLike) -> NDArrayLike:
        """
        log | du / dbeta | = log( |cos(beta)| ) = log(cos(beta))
        Valid because beta is in [-pi/2, pi/2], so cos(beta) is positive.
        """
        beta_phys = self.xp.asarray(beta_phys)
        out = self.xp.log(self.xp.cos(beta_phys))
        return self._to_device(out)
    

class Log10Uniform(UniformDistribution):
    """
    Log10-Uniform distribution.
    
    Sampler space: Uniform in u = log10(x) between [log10(minimum), log10(maximum)].
    Physical space: x distributed as p(x) ~ 1/x in [minimum, maximum].
    """
    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("Log10Uniform minimum value must be strictly positive.")

        super().__init__(
            minimum=float(np.log10(minimum)),
            maximum=float(np.log10(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        # Precompute constants to avoid recalculating during sampling
        self._log_log10 = self.xp.log(self.xp.log(10.0))

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike) -> NDArrayLike:
        """Physical x -> Sampling u = log10(x)"""
        x_phys = self.xp.asarray(x_phys)
        out = self.xp.log10(x_phys)
        return self._to_device(out)

    def _inverse(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Sampling u -> Physical x = 10^u"""
        u_samp = self.xp.asarray(u_samp)
        out = self.xp.power(10.0, u_samp)
        return self._to_device(out)

    def _jacobian(self, x_phys: NDArrayLike) -> NDArrayLike:
        """
        u = ln(x) / ln(10)  ->  du/dx = 1 / (x * ln(10))
        log |du/dx| = -ln(x) - ln(ln(10))
        """
        x_phys = self.xp.asarray(x_phys)
        out = -self.xp.log(x_phys) - self._log_log10
        return self._to_device(out)
    
    
class UniformInVolume(PowerLaw):
    """
    Euclidean Volume prior, commonly used for luminosity distance (d_L).
    
    Sampler space: Uniform in u = d_L^3.
    Physical space: d_L distributed as p(d_L) ~ d_L^2.
    """
    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum < 0:
            raise ValueError("Distance cannot be negative.")
            
        super().__init__(
            alpha=2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )


class InverseUniform(PowerLaw):
    """
    Uniform in Inverse space.
    
    Sampler space: Uniform in u = 1/x.
    Physical space: x distributed as p(x) ~ 1/x^2.
    """
    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("InverseUniform minimum value must be strictly positive.")
            
        super().__init__(
            alpha=-2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        
class Gaussian(Prior):
    """Gaussian prior with mean mu and standard deviation sigma."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.mu = mu
        self.sigma = sigma
        self._log_norm = 0.5 * self.xp.log(2.0 * self.xp.pi * self.sigma**2)

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = -0.5 * ((x_arr - self.mu) / self.sigma) ** 2 - self._log_norm
        return self._to_device(out)


class Normal(Gaussian):
    """Synonym for the Gaussian distribution."""
    pass


class LogNormal(Prior):
    """Log-normal prior with parameters mu and sigma."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if sigma <= 0.0:
            raise ValueError("Standard deviation sigma must be positive.")
        self.mu = mu
        self.sigma = sigma
        self._log_sqrt_2pi = self.xp.log(self.xp.sqrt(2.0 * self.xp.pi))

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr > self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 1.0)
        log_x = self.xp.log(valid_x)
        
        log_prob = (
            -0.5 * ((log_x - self.mu) / self.sigma) ** 2
            - self.xp.log(valid_x * self.sigma)
            - self._log_sqrt_2pi
        )
        out[mask] = log_prob[mask]
        
        return self._to_device(out)


class LogGaussian(LogNormal):
    """Synonym of LogNormal prior."""
    pass


class Exponential(Prior):
    """Exponential prior with scale mu (mean)."""

    def __init__(
        self,
        mu: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if mu <= 0.0:
            raise ValueError("Exponential mean 'mu' must be positive.")
        self.mu = mu
        self._log_mu = self.xp.log(self.mu)

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.exponential(scale=self.mu, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr >= self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = -(valid_x / self.mu) - self._log_mu
        
        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.zeros_like(x_arr, dtype=self.xp.float64)
        mask = x_arr >= self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = 1.0 - self.xp.exp(-valid_x / self.mu)
        
        return self._to_device(out)
    
    

class ResolvabilityPrior(Prior):
    """
    Error-function (Normal CDF) based prior representing the probability 
    of resolving a source given its Signal-to-Noise Ratio (SNR).
    
    p(resolved | rho) = 0.5 * (1 + erf((rho - rho_th) / (sqrt(2) * sigma)))
    """

    def __init__(
        self,
        rho_threshold: float = 7.0,
        sigma: float = 1.0,
        name: str | None = None,
        latex_label: str | None = r"p_{\rm res}",
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            name_phys=name,
            latex_label=latex_label,
            minimum=0.0,
            maximum=np.inf,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )
        self.rho_threshold = rho_threshold
        self.sigma = sigma

        # Check for cupyx if GPU is requested
        if self.use_cupy:
            try:
                import cupyx.scipy.special  # noqa: F401
            except ImportError:
                raise ImportError(
                    "ResolvabilityPrior requires 'cupyx' for stable erf/log_ndtr "
                    "evaluation when use_cupy=True."
                )

    def _log_ndtr(self, x: NDArrayLike) -> NDArrayLike:
        """
        Dynamically dispatch the log of the Normal CDF.
        This is mathematically equivalent to log(0.5 * (1 + erf(x / sqrt(2))))
        but is numerically stable for large negative values.
        """
        if self.use_cupy:
            from cupyx.scipy.special import log_ndtr
            return log_ndtr(x)
        else:
            from scipy.special import log_ndtr
            return log_ndtr(x)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """
        Calculates the log probability of resolution.
        Args:
            x: Flattened array of SNRs.
        """
        x_arr = self.xp.asarray(x)
        
        # Calculate z = (rho - rho_th) / sigma
        # Note: log_ndtr divides by sqrt(2) internally!
        z = (x_arr - self.rho_threshold) / self.sigma
        
        log_prob = self._log_ndtr(z)
        
        return self._to_device(log_prob)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """
        Resolvability evaluates SNR probabilities. We cannot draw SNRs 
        directly from it without a waveform and noise realization.
        """
        raise NotImplementedError(
            "ResolvabilityPrior evaluates SNR probabilities. You cannot "
            "draw SNRs directly from it without a waveform and noise realization."
        )