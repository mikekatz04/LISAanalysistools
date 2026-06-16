from __future__ import annotations

from typing import Tuple, Union, Sequence
import numpy as np
from numpy.typing import ArrayLike

from ...utils.typing import NDArrayLike
from .base import Prior


class DiscreteUniform(Prior):
    """
    A uniform discrete prior over a range of integers [minimum, maximum].
    
    This is useful for model selection (RJ-MCMC) or selecting discrete states.
    """

    def __init__(
        self,
        minimum: int,
        maximum: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError(f"DiscreteUniform requires integer bounds. Got min: {minimum}, max: {maximum}")
        if minimum >= maximum:
            raise ValueError(f"maximum ({maximum}) must be strictly greater than minimum ({minimum}).")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            unit=unit,
            minimum=float(minimum), # Store as float for base class compatibility
            maximum=float(maximum),
            check_range_nonzero=True,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.min_int = minimum
        self.max_int = maximum
        
        # Total number of discrete states (inclusive of bounds)
        self.n_states = self.max_int - self.min_int + 1
        
        # log(1 / N) = -log(N)
        self._logpdf_val = -np.log(self.n_states)

    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
            
        # randint is exclusive of the upper bound, so we add 1
        samples = self.xp.random.randint(self.min_int, self.max_int + 1, size=size)
        
        # Samplers usually expect float arrays even for discrete parameters
        samples = samples.astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        
        # A valid discrete state must be within bounds AND be an exact integer
        mask = (x_arr >= self.min_int) & (x_arr <= self.max_int) & (self.xp.round(x_arr) == x_arr)
        
        out[mask] = self._logpdf_val
        return self._to_device(out)

    # Alias logpmf to logpdf. Some samplers (like Eryn's RJ-MCMC) sometimes prefer 
    # to call logpmf for discrete distributions. This makes it bulletproof.
    logpmf = logpdf

    def cdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """
        The CDF of a discrete uniform distribution is a step function.
        """
        x_arr = self.xp.asarray(x)
        
        # Count how many discrete states are <= x
        states_below = self.xp.floor(x_arr) - self.min_int + 1
        
        # Clip between 0 and total number of states
        states_below = self.xp.clip(states_below, 0, self.n_states)
        
        out = states_below / self.n_states
        return self._to_device(out)


class Categorical(DiscreteUniform):
    """
    A convenience prior for sampling categorically among N possibilities.
    Samples uniformly from the integers [0, 1, ..., n_categories - 1].
    """

    def __init__(
        self,
        n_categories: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(f"Categorical prior requires n_categories >= 2. Got {n_categories}.")

        super().__init__(
            minimum=0,
            maximum=n_categories - 1,
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )
      


class Poisson(Prior):
    """
    Poisson prior distribution for integer event counts.
    
    Probability mass function: p(k) = (lam^k * e^-lam) / k!
    """

    def __init__(
        self,
        lam: float,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if lam <= 0:
            raise ValueError(f"Poisson rate parameter 'lam' must be strictly positive. Got {lam}.")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            minimum=0.0,
            maximum=np.inf,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.lam = lam
        self._log_lam = np.log(self.lam)

        # Check for CuPyX if using GPU
        if self.use_cupy:
            try:
                import cupyx.scipy.special  # noqa: F401
            except ImportError:
                raise ImportError(
                    "The Poisson prior requires 'cupyx' for the gammaln function "
                    "when use_cupy=True. Please install cupy/cupyx."
                )

    def _gammaln(self, x: NDArrayLike) -> NDArrayLike:
        """Dynamically dispatches the log-gamma function based on device."""
        if self.use_cupy:
            from cupyx.scipy.special import gammaln
            return gammaln(x)
        else:
            from scipy.special import gammaln
            return gammaln(x)

    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
            
        # Draw samples and cast to float to play nice with samplers
        samples = self.xp.random.poisson(lam=self.lam, size=size).astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        
        # Must be non-negative and an exact integer
        mask = (x_arr >= 0) & (self.xp.round(x_arr) == x_arr)
        
        valid_x = self.xp.where(mask, x_arr, 0.0)
        
        # ln p(k) = k * ln(lam) - lam - ln(k!)
        # where ln(k!) = gammaln(k + 1)
        log_prob = valid_x * self._log_lam - self.lam - self._gammaln(valid_x + 1.0)
        
        out[mask] = log_prob[mask]
        return self._to_device(out)
    
    

class HyperPoisson(Prior):
    """
    Conditional Poisson prior for the number of resolved sources N,
    given a discrete model index M_i.
    
    p(N | M_i) = Poisson(lam_i)
    """

    def __init__(
        self,
        lams: Sequence[float],
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            minimum=0.0,
            maximum=np.inf,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.lams = np.array(lams, dtype=np.float64)
        self.n_models = len(self.lams)
        
        if self.use_cupy:
            self.lams = self.xp.asarray(self.lams)

        self._log_lams = self.xp.log(self.lams)

        if self.use_cupy:
            try:
                import cupyx.scipy.special  # noqa: F401
            except ImportError:
                raise ImportError("HyperPoisson requires 'cupyx' when use_cupy=True.")

    def _gammaln(self, x: NDArrayLike) -> NDArrayLike:
        """Helper to safely route to the correct scipy/cupyx gammaln implementation."""
        if self.use_cupy:
            from cupyx.scipy.special import gammaln
            return gammaln(x)
        else:
            from scipy.special import gammaln
            return gammaln(x)

    def logpdf(self, x: ArrayLike, model_index: ArrayLike, **kwargs) -> NDArrayLike:
        """
        Evaluate the log probability of N given M_i.
        
        Args:
            x: Array of event counts N. 
            model_index: Array of model integers.
        """
        x_arr = self.xp.asarray(x, dtype=self.xp.float64)
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
  
        if mod_idx.shape != x_arr.shape:
            if mod_idx.size == x_arr.size:
                mod_idx = mod_idx.reshape(x_arr.shape)
            else:
                mod_idx = self.xp.broadcast_to(mod_idx, x_arr.shape)

        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        
        mask = (x_arr >= 0) & (self.xp.round(x_arr) == x_arr) & (mod_idx >= 0) & (mod_idx < self.n_models)
        
        valid_x = x_arr[mask]
        valid_mod = mod_idx[mask]
        
        if valid_x.size == 0:
            return self._to_device(out)
        
        lam_active = self.lams[valid_mod]
        log_lam_active = self._log_lams[valid_mod]
        
        # ln p(N|M) = N * ln(lam_M) - lam_M - ln(N!)
        log_prob = valid_x * log_lam_active - lam_active - self._gammaln(valid_x + 1.0)
        
        out[mask] = log_prob
        
        return self._to_device(out)

    # Alias logpmf to logpdf because Eryn's ProbDistContainer checks for logpmf 
    # when dealing with discrete/integer counting variables.
    logpmf = logpdf

    def rvs(self, size: int | Tuple[int, ...] = (1,), model_index: ArrayLike = 0, **kwargs) -> NDArrayLike:
        """
        Draw samples condition on a given model index (or array of model indices).
        """
        if isinstance(size, int):
            size = (size,)
            
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.shape != size:
            if mod_idx.size == np.prod(size):
                mod_idx = mod_idx.reshape(size)
            else:
                mod_idx = self.xp.broadcast_to(mod_idx, size)
                
        lam_active = self.lams[mod_idx]
        
        samples = self.xp.random.poisson(lam_active).astype(self.xp.float64)
        
        return self._to_device(samples)