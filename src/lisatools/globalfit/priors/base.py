"""
The base Prior infrastructure for LISA analysis tools.

Provides a functional, stateless base Prior class supporting GPU acceleration,
automatic conditional dependency tracking via method signatures, and 
seamless transformations between physical and sampling parameter spaces.
"""

from __future__ import annotations

import inspect
import warnings
from copy import deepcopy

from typing import Callable, Union, Dict, List, Tuple, Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from lisatools.utils.typing import NDArrayLike




class PriorException(Exception):
    """ General base class for all prior exceptions """
    
    
class Prior(object):  
    _default_latex_labels = {}

    def __init__(
        self,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        minimum: float = -np.inf,
        maximum: float = np.inf,
        boundary: str | None = None,
        check_range_nonzero: bool = True,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform: Callable[..., NDArrayLike] | None = None,
        inverse_transform: Callable[..., NDArrayLike] | None = None,
        log_jacobian: Callable[..., NDArrayLike] | None = None,
    ):
        """Main prior constructor

        Args:
            name (str, optional): Name associated with prior.
            name_phys (str, optional): Name of the associated physical parameter, if different from name.
                defaults to name if not provided.
            latex_label (str, optional): Latex label associated with prior, used for plotting.
            unit (str, optional): If given, a Latex string describing the units of the parameter.
            minimum (float, optional): Minimum of the domain, default=-np.inf
            maximum (float, optional): Maximum of the domain, default=np.inf
            boundary (str, optional): Type of boundary conditions for the prior (e.g., 'periodic', 'reflective').
            check_range_nonzero (boolean, optional): If True, checks that the prior range is non-zero
            use_cupy (boolean, optional): If True, uses CuPy for GPU acceleration
            return_gpu (boolean, optional): If True, returns GPU arrays when use_cupy is True
            forward_transform (callable, optional): A function that transforms from the physical to 
                the sampling parameter space.
            inverse_transform (callable, optional): A function that transforms from the sampling to 
                the physical parameter space.
            log_jacobian (callable, optional): A function that calculates the log Jacobian of the 
                transformation between physical and sampling parameter spaces. If not provided, it is 
                assumed to be zero, i.e., that the transformation is volume-preserving.
                
        Raises:
            ValueError: If check_range_nonzero is True and maximum <= minimum.
        """
        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                f"maximum {maximum} <= minimum {minimum} "
                f"for {type(self).__name__} prior on {name}"
            )

        self.name = name
        self.name_phys = name_phys if name_phys is not None else name
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu
        self.latex_label = latex_label
        self.boundary = boundary
        if self.boundary not in [None, "periodic", "reflective"]:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary}' for prior '{self.name}'. "
                f"Supported values are None, 'periodic', or 'reflective'."
            )

        # Coordinate transformation attributes (Physical <-> Sampling)
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.log_jacobian = log_jacobian

        # Parse dependencies automatically from the class signatures
        self._infer_dependencies()
        
    def _infer_dependencies(self):
        """Automatically infers the required dependencies for the prior from the method signatures of 
        logpdf and rvs. This allows for automatic handling of conditional priors, where the prior on 
        one parameter may depend on the value of another parameter. By inspecting the method signatures 
        of logpdf and rvs, we can determine which parameters are required for evaluating the prior, and 
        ensure that these are provided when the prior is evaluated. This makes it easier to implement 
        complex priors that may have dependencies on other parameters, without having to manually specify 
        these dependencies.
        """
        deps = set()

        # Reserved keywords that are not conditional physical parameters
        reserved_kwargs = {"self", "x", "size", "rng", "random_state", "kwargs", "args"}

        for method_name in ("logpdf", "rvs"):
            # Check if the subclass implemented the method
            if method_name not in self.__class__.__dict__:
                continue

            method = getattr(self, method_name)
            sig = inspect.signature(method)

            for name, param in sig.parameters.items():
                if name in reserved_kwargs:
                    continue
                # Ignore *args and **kwargs
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                deps.add(name)

        # Sort for deterministic ordering
        self.required_variables: tuple[str, ...] = tuple(sorted(deps))

    @property
    def xp(self) -> Any:
        """Dynamically return the array module (numpy or cupy)."""
        if self.use_cupy:
            import cupy as cp
            return cp
        return np

    def _to_device(self, array: NDArrayLike) -> NDArrayLike:
        """Safely move arrays back to CPU if requested."""
        if self.use_cupy and not self.return_gpu:
            if hasattr(array, "get"):
                return getattr(array, "get")()
            return np.asarray(array) # future proofing for JAX
        return array
    
    def rvs(
        self, 
        size: int | Tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draws a random variable(s) from the prior distribution.

        Args:
            size (int | Tuple[int, ...], optional): Size of the sample. Defaults to (1,).
            **kwargs: Additional keyword arguments to pass to the rvs method, which may include
            random_state or other distribution-specific parameters.

        Raises:
            ValueError: If size is not an integer or a tuple of integers.

        Returns:
            NDArrayLike: A random sample drawn from the prior distribution, with the specified size.
        """
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        
        if isinstance(size, int):
            size = (size,)
        
        raise NotImplementedError("rvs method is implemented in subclass")

    def rvs_physical(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draw a random sample in the physical parameter space. This uses the rvs method to draw a 
        sample in the sampling parameter space, and then applies the inverse transform to get a sample
        in the physical parameter space.
        
        Args:
            size (int | Tuple[int, ...], optional): Size of the sample. Defaults to (1,).
            **kwargs: Additional keyword arguments to pass to the rvs method.
            
        Returns:
            NDArrayLike: A random sample drawn from the prior distribution in the physical parameter 
                space, with the specified size.
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform is None:
            return self._to_device(samples)
        
        output = self.inverse_transform(samples, **kwargs)
        return self._to_device(output)
    
    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """The log probability density of the prior distribution at the given value(s).

        Args:
            x (ArrayLike): The value(s) for which to calculate the log probability density.
            **kwargs: Additional keyword arguments to pass to the logpdf method.

        Returns:
            NDArrayLike: The log probability density of the given value(s).
        """
        raise NotImplementedError("logpdf method is implemented in subclass")
    
    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Log probability density evaluated in the physical parameter space. This uses the logpdf
        method to evaluate the log probability density in the sampling parameter space, and then 
        applies the log Jacobian of the transformation to get the log probability density in the 
        physical parameter space.
        
        Args:
            x_phys (ArrayLike): The value(s) in the physical parameter space for which to calculate 
                the log probability density.
            **kwargs: Additional keyword arguments to pass to the logpdf method, as well as to the 
                log_jacobian method if a transformation is defined.
                
        Returns:
            NDArrayLike: The log probability density of the given value(s) in the physical parameter space.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = logp_samp + log_J
        return self._to_device(output)
    
    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Calculates the probability density of the prior distribution at the given value(s). This 
        is a generic method that uses the logpdf method to calculate the log probability density, and 
        then exponentiates it to get the probability density. This can be overwritten in subclasses if
        a more efficient implementation is available for a specific prior.

        Args:
            x (ArrayLike): The value(s) for which to calculate the probability density.
            **kwargs: Additional keyword arguments to pass to the logpdf method.

        Returns:
            NDArrayLike: The probability density of the given value(s).
        """
        output = self.xp.exp(self.logpdf(x, **kwargs))
        return self._to_device(output)
    
    def pdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Probability density evaluated in the physical parameter space. This uses the pdf method to 
        evaluate the probability density in the sampling parameter space, and then applies the Jacobian
        of the transformation to get the probability density in the physical parameter space.
        
        Args:
            x_phys (ArrayLike): The value(s) in the physical parameter space for which to calculate 
                the probability density.
            **kwargs: Additional keyword arguments to pass to the logpdf method, as well as to the 
                log_jacobian method if a transformation is defined.

        Returns:
            NDArrayLike: The probability density of the given value(s) in the physical parameter space.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        n_points: int = 1000,
        **kwargs
    ) -> NDArrayLike:
        """Generic method to calculate CDF, can be overwritten in subclass"""
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with"
                "infinite support")
        
        x_arr = self.xp.asarray(x)
        grid, dx = self.xp.linspace(self.minimum, self.maximum, n_points, retstep=True)

        pdf_grid = self.pdf(grid, **kwargs) 
        cdf_grid = self.xp.zeros_like(pdf_grid)
        cdf_grid[1:] = self.xp.cumsum(pdf_grid[:-1] + pdf_grid[1:]) * dx
        cdf_grid /= cdf_grid[-1]  # Normalize to ensure CDF goes to 1 at maximum
        
        output = self.xp.interp(x_arr, grid, cdf_grid, left=0.0, right=1.0)        

        if isinstance(x, (int, float)) and not self.return_gpu:
            output = float(output)
        return self._to_device(output)

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        fallback_samples: int = 10_000
    ) -> NDArray:
        if self.minimum and self.maximum is not None:
            return (x >= self.minimum) & (x <= self.maximum)
        else:
            warnings.warn(
                "Prior range not defined, using samples to estimate range for is_in_prior_range \
                method. This may result in an inaccurate estimate, or could be slow for complex priors.",
                stacklevel=3
            )
            samples = self.rvs(size=fallback_samples)
            minimum, maximum = samples.min(), samples.max()
            return (x >= minimum) & (x <= maximum)

    @property
    def latex_label(self) -> str:
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: str | None = None):
        if latex_label is not None:
            self.__latex_label = latex_label
        elif self.name in self._default_latex_labels:
            self.__latex_label = self._default_latex_labels[self.name]
        else:
            self.__latex_label = str(self.name)

    @property
    def unit(self) -> str | None:
        return self.__unit

    @unit.setter
    def unit(self, unit: str | None = None):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):
        """ If a unit is specified, returns a string of the latex label and unit """
        if self.unit is not None:
            return f"{self.latex_label} [{self.unit}]"
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    @property
    def width(self):
        return self.maximum - self.minimum

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the prior that can be evaluated 
        to reconstruct the object. Perfect for metadata serialization.
        """
        class_name = self.__class__.__name__
        
        # We dynamically fetch the initialization arguments
        import inspect
        sig = inspect.signature(self.__init__)
        
        args = []
        for key in sig.parameters.keys():
            if key in ["self", "kwargs", "args"]:
                continue
            if hasattr(self, key):
                val = getattr(self, key)
                # Format strings and floats properly
                if isinstance(val, str):
                    args.append(f"{key}='{val}'")
                elif isinstance(val, float):
                    args.append(f"{key}={val}")
                elif isinstance(val, tuple) or isinstance(val, list):
                    args.append(f"{key}={val}")
                
        args_str = ", ".join(args)
        return f"{class_name}({args_str})"
    
    def copy(self) -> Prior:
        return deepcopy(self)


class UniformDistribution(Prior):
    """
    Standard Uniform prior. Does not depend on conditional variables.
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
        super().__init__(
            name=name,
            minimum=minimum,
            maximum=maximum,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.pdf_val = 1.0 / self.width 
        self.logpdf_val = np.log(self.pdf_val)

        if self.use_cupy:
            try:
                import cupy as cp  # noqa: F401
            except ImportError:
                raise ValueError("use_cupy is True, but CuPy is not installed.")

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        samples = self.xp.random.uniform(self.minimum, self.maximum, size=size)
        return self._to_device(samples)

    def pdf(self, x: ArrayLike) -> NDArrayLike:
        out = self.pdf_val * ((x >= self.minimum) & (x <= self.maximum))
        return self._to_device(out)
    

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)

        mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
        out[mask] = self.logpdf_val

        return self._to_device(out)


class HierarchicalUniformDistribution(Prior):
    """
    Conditional Uniform prior.
    Depends on a variable `y` dictating its upper bound: p(x | y) = U(min_val, y)
    """

    def __init__(
        self,
        minimum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=minimum,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

    def rvs(self, y: ArrayLike, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        y_arr = self.xp.asarray(y)
        samples = self.xp.random.uniform(self.minimum, y_arr, size=size)
        return self._to_device(samples)

    def pdf(self, x: ArrayLike, y: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        y_arr = self.xp.asarray(y)

        out = (y_arr - self.minimum) * ((x_arr >= self.minimum) & (x_arr <= y_arr))
        return self._to_device(out)

    def logpdf(self, x: ArrayLike, y: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        y_arr = self.xp.asarray(y)

        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        mask = (x_arr >= self.minimum) & (x_arr <= y_arr)

        valid_y = self.xp.where(mask, y_arr, self.minimum + 1.0)
        out[mask] = -self.xp.log(valid_y - self.minimum)

        return self._to_device(out)