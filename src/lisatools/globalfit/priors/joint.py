from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, Sequence, Callable, Tuple
import numpy as np

from numpy.typing import ArrayLike
from ...utils.typing import NDArrayLike


class JointPrior(object):
    _default_latex_labels = {}

    def __init__(
        self,
        names: Sequence[str],
        names_phys: Sequence[str] | None = None,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        boundaries: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform_nd: Callable[..., NDArrayLike] | None = None,
        inverse_transform_nd: Callable[..., NDArrayLike] | None = None,
        log_jacobian_nd: Callable[..., NDArrayLike] | None = None,
    ):
        """Main joint prior constructor for N-dimensional parameter spaces.

        Args:
            names (Sequence[str]): Names associated with the parameters in this joint prior.
            names_phys (Sequence[str], optional): Names of the associated physical parameters, 
                if different from names. Defaults to names if not provided.
            latex_label (Sequence[str | None], optional): Latex labels associated with 
                the parameters, used for plotting.
            unit (Sequence[str | None], optional): If given, Latex strings describing 
                the units of the parameters.
            boundaries (Sequence[str | None], optional): If given, specifies the type of
                boundary conditions for each parameter (e.g., 'periodic', 'reflective').
            use_cupy (boolean, optional): If True, uses CuPy for GPU acceleration.
            return_gpu (boolean, optional): If True, returns GPU arrays when use_cupy is True.
            forward_transform_nd (callable, optional): A function that transforms from the 
                N-dimensional physical to the sampling parameter space.
            inverse_transform_nd (callable, optional): A function that transforms from the 
                N-dimensional sampling to the physical parameter space.
            log_jacobian_nd (callable, optional): A function that calculates the log-determinant 
                of the Jacobian matrix of the transformation between physical and sampling parameter 
                spaces. If not provided, it is assumed to be zero (volume-preserving).
                
        Raises:
            ValueError: If names sequence is empty.
        """
        if not names:
            raise ValueError("Joint priors require a sequence of parameter names.")

        self.names = tuple(names)
        
        self.num_vars = len(self.names)
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

        # Validate and set physical names
        if names_phys is not None:
            if len(names_phys) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} names_phys, got {len(names_phys)}")
            self.names_phys = tuple(
                name_phys if name_phys is not None else name for name, name_phys in zip(self.names, names_phys)
            )
        else:
            self.names_phys = self.names
            
        # Validate and set boundaries
        if boundaries is not None:
                if len(boundaries) != self.num_vars:
                    raise ValueError(f"Expected {self.num_vars} boundaries, got {len(boundaries)}")
                for boundary in boundaries:
                    if boundary not in [None, "periodic", "reflective"]:
                        raise ValueError(
                            f"Invalid boundary condition '{boundary}' for joint prior parameter. "
                            f"Supported values are None, 'periodic', or 'reflective'."
                        )
                self.boundaries = tuple(boundaries)
        else:
            self.boundaries = tuple(None for _ in range(self.num_vars))

        # Trigger the setters to validate lengths and apply defaults
        self.latex_label = latex_labels
        self.unit = units

        # N-dimensional coordinate transformation attributes (Physical <-> Sampling)
        self.forward_transform_nd = forward_transform_nd
        self.inverse_transform_nd = inverse_transform_nd
        self.log_jacobian_nd = log_jacobian_nd

        # Parse dependencies automatically from the class signatures
        self._infer_dependencies()

    def _infer_dependencies(self):
        """Automatically infers the required dependencies for the joint prior from the 
        method signatures of logpdf and rvs. 
        
        For a JointPrior, the core dependencies are always the parameter names themselves, 
        but inspecting the signatures allows for automatic handling of hierarchical 
        dependencies where the N-dimensional joint prior may additionally depend on an 
        external parameter.
        """
        deps = set(self.names)

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
            return np.asarray(array)  # future proofing for JAX
        return array

    def rvs(
        self, 
        size: int | Tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draws random variable(s) from the joint prior distribution.

        Args:
            size (int | Tuple[int, ...], optional): Size of the sample. Defaults to (1,).
            **kwargs: Additional keyword arguments to pass to the rvs method, which may include
                random_state or other distribution-specific parameters.

        Raises:
            ValueError: If size is not an integer or a tuple of integers.

        Returns:
            NDArrayLike: A random sample drawn from the prior distribution, of shape 
                (*size, self.num_vars).
        """
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        
        if isinstance(size, int):
            size = (size,)
        
        raise NotImplementedError("rvs method is implemented in subclass")

    def rvs_physical(
        self, 
        size: int | Tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draw a random sample in the physical parameter space. This uses the rvs method to draw a 
        sample in the sampling parameter space, and then applies the N-dimensional inverse transform 
        to get a sample in the physical parameter space.
        
        Args:
            size (int | Tuple[int, ...], optional): Size of the sample. Defaults to (1,).
            **kwargs: Additional keyword arguments to pass to the rvs method.
            
        Returns:
            NDArrayLike: A random sample drawn from the joint prior distribution in the physical 
                parameter space, of shape (*size, self.num_vars).
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform_nd is None:
            return self._to_device(samples)
        
        output = self.inverse_transform_nd(samples, **kwargs)
        return self._to_device(output)

    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """The log probability density of the joint prior distribution at the given value(s).

        Args:
            x (ArrayLike): The N-dimensional value(s) for which to calculate the log probability 
                density. Expected shape is (..., self.num_vars).
            **kwargs: Additional keyword arguments to pass to the logpdf method.

        Returns:
            NDArrayLike: The log probability density of the given value(s), of shape (...).
        """
        raise NotImplementedError("logpdf method is implemented in subclass")

    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Log probability density evaluated in the physical parameter space. This uses the logpdf
        method to evaluate the log probability density in the sampling parameter space, and then 
        applies the log-determinant of the Jacobian to get the density in the physical space.
        
        Args:
            x_phys (ArrayLike): The value(s) in the physical parameter space for which to calculate 
                the log probability density. Expected shape is (..., self.num_vars).
            **kwargs: Additional keyword arguments to pass to the logpdf method, as well as to the 
                log_jacobian_nd method if a transformation is defined.
                
        Returns:
            NDArrayLike: The log probability density in the physical parameter space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = logp_samp + log_det_J
        return self._to_device(output)

    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Calculates the probability density of the joint prior distribution at the given value(s).

        Args:
            x (ArrayLike): The N-dimensional value(s) for which to calculate the probability density.
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
        """Probability density evaluated in the physical parameter space.
        
        Args:
            x_phys (ArrayLike): The N-dimensional value(s) in the physical parameter space.
            **kwargs: Additional keyword arguments to pass to the logpdf and log_jacobian methods.

        Returns:
            NDArrayLike: The probability density of the given value(s) in the physical parameter space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_det_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """CDF numerical integration is undefined generally for N-dimensional JointPriors."""
        raise NotImplementedError(
            "Generic CDF numerical integration is mathematically undefined "
            "for N-dimensional JointPriors."
        )

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Boundary checking for N-dimensional bounds."""
        raise NotImplementedError(
            "Boundary checking must be implemented per JointPrior subclass using "
            "an N-dimensional bounding box or geometric constraint."
        )

    @property
    def latex_label(self) -> tuple[str, ...]:
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: Sequence[str | None] | None = None):
        if latex_label is not None:
            if len(latex_label) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} latex_label, got {len(latex_label)}")
            
            labels = []
            for name, label in zip(self.names, latex_label):
                if label is not None:
                    labels.append(label)
                elif name in self._default_latex_labels:
                    labels.append(self._default_latex_labels[name])
                else:
                    labels.append(str(name))
            self.__latex_label = tuple(labels)
        else:
            self.__latex_label = tuple(
                self._default_latex_labels.get(name, str(name)) for name in self.names
            )

    @property
    def unit(self) -> tuple[str | None, ...]:
        return self.__unit

    @unit.setter
    def unit(self, unit: Sequence[str | None] | None = None):
        if unit is not None:
            if len(unit) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} units, got {len(unit)}")
            self.__unit = tuple(unit)
        else:
            self.__unit = tuple(None for _ in self.names)

    @property
    def latex_label_with_unit(self) -> tuple[str, ...]:
        """ If a unit is specified, returns a tuple of strings of the latex labels and units """
        return tuple(
            f"{label} [{unit}]" if unit is not None else label
            for label, unit in zip(self.latex_label, self.unit)
        )
        
    # @property
    # def __default_latex_label(self):
    #     if self.name in self._default_latex_labels.keys():
    #         label = self._default_latex_labels[self.name]
    #     else:
    #         label = self.name
    #     return label
    
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
    
    def copy(self) -> JointPrior:
        return deepcopy(self)
    
    
    
class MultivariateGaussian(JointPrior):
    """
    N-dimensional Multivariate Gaussian Prior.
    """

    def __init__(
        self,
        names: Sequence[str],
        mu: ArrayLike,
        cov: ArrayLike,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        super().__init__(
            names=names, 
            latex_labels=latex_labels,
            units=units,
            use_cupy=use_cupy, 
            return_gpu=return_gpu
        )

        self.mu = self.xp.asarray(mu, dtype=self.xp.float64)
        self.cov = self.xp.asarray(cov, dtype=self.xp.float64)

        if self.mu.shape != (self.num_vars,):
            raise ValueError(f"'mu' must have shape ({self.num_vars},).")
        if self.cov.shape != (self.num_vars, self.num_vars):
            raise ValueError(f"'cov' must have shape ({self.num_vars}, {self.num_vars}).")

        if not self.xp.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix must be symmetric.")

        # Precompute precision and normalization on CPU
        try:
            self.prec = self.xp.linalg.inv(self.cov)
        except self.xp.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular and cannot be inverted.")

        sign, logdet = self.xp.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite.")
            
        self._log_norm = -0.5 * (self.num_vars * self.xp.log(2.0 * self.xp.pi) + logdet)
        self.cholesky_lower = self.xp.linalg.cholesky(self.cov)

        # Push to GPU if requested
        if self.use_cupy:
            self.mu = self.xp.asarray(self.mu)
            self.prec = self.xp.asarray(self.prec)
            self.cholesky_lower = self.xp.asarray(self.cholesky_lower)

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        z = self.xp.random.normal(0.0, 1.0, size=(*size, self.num_vars))
        samples = self.mu + self.xp.einsum("ij,...j->...i", self.cholesky_lower, z)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != self.num_vars:
            raise ValueError(
                f"Expected last dimension of x to be {self.num_vars}, got {x_arr.shape[-1]}"
            )

        diff = x_arr - self.mu
        quadratic_term = self.xp.einsum("...i,ij,...j->...", diff, self.prec, diff)
        out = -0.5 * quadratic_term + self._log_norm
        return self._to_device(out)
    
    
    
class MojitoF0FdotPrior(JointPrior):
    """
    Joint prior for f0 and fdot.
    
    f0 is distributed uniformly between [f0_min, f0_max].
    fdot is distributed uniformly between dynamic bounds dictated by f0.
    """

    def __init__(
        self,
        f0_min: float = 1e-4,
        f0_max: float = 2.1e-2,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            names=("f0", "fdot"),
            latex_labels=(r"f_0", r"\dot{f}"),
            units=("Hz", "Hz/s"),
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )
        self.f0_min = f0_min
        self.f0_max = f0_max

    def _get_fdot_bounds(self, f0: NDArrayLike) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        Vectorized calculation of the dynamic fdot bounds.
        """
        f0_safe = self.xp.where(f0 > 0, f0, 1.0)
        
        min_val = -2e-20 * (f0_safe / 4e-4) ** (16 / 3)
        max_val = 3e-21 * (f0_safe / 1e-4) ** (11 / 3)
        
        return min_val, max_val

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        f0_samples = self.xp.random.uniform(self.f0_min, self.f0_max, size=size)
        fdot_min, fdot_max = self._get_fdot_bounds(f0_samples)
        
        fdot_samples = self.xp.random.uniform(fdot_min, fdot_max, size=size)
        samples = self.xp.stack([f0_samples, fdot_samples], axis=-1)
        
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != 2:
            raise ValueError(f"Expected last dimension of x to be 2, got {x_arr.shape[-1]}")

        f0 = x_arr[..., 0]
        fdot = x_arr[..., 1]
        
        out = self.xp.full_like(f0, -self.xp.inf, dtype=self.xp.float64)
        
        mask_f0 = (f0 >= self.f0_min) & (f0 <= self.f0_max)
        
        fdot_min, fdot_max = self._get_fdot_bounds(f0)
        mask_fdot = (fdot >= fdot_min) & (fdot <= fdot_max)
        
        # Overall valid mask
        valid = mask_f0 & mask_fdot
        
        # log p = log p(f0) - log(fdot_max - fdot_min)
        valid_fdot_diff = self.xp.where(valid, fdot_max - fdot_min, 1.0)
        f0_log_norm = -self.xp.log(self.f0_max - self.f0_min)
        out[valid] = f0_log_norm - self.xp.log(valid_fdot_diff[valid])
        
        return self._to_device(out)
    
    
    
def f_mHz_to_Hz_and_fdot(f0_mHz: NDArrayLike, fdot: NDArrayLike) -> Tuple[NDArrayLike, NDArrayLike]:
    """Utility function to convert f0 from mHz to Hz and fdot from mHz/yr to Hz/s."""
    f0_Hz = f0_mHz * 1e-3
    return f0_Hz, fdot


class MojitoF0mHzFdotPrior(JointPrior):
    """
    Joint prior for f0 in mHz and fdot.
    
    f0 is distributed uniformly between [f0_min, f0_max].
    fdot is distributed uniformly between dynamic bounds dictated by f0.
    """

    def __init__(
        self,
        f0_min_mHz: float = 0.1,
        f0_max_mHz: float = 21.0,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            names=("f0", "fdot"),
            names_phys=("f0", "fdot"), # Maps to the physical Hz basis
            latex_labels=(r"f_0", r"\dot{f}"),
            units=("mHz", "Hz/s"),
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=f_mHz_to_Hz_and_fdot,
            **kwargs
        )
        self.f0_min = f0_min_mHz
        self.f0_max = f0_max_mHz

    def _get_fdot_bounds(self, f0_mHz: NDArrayLike) -> tuple[NDArrayLike, NDArrayLike]:
        """Calculates dynamic bounds for fdot. The formula requires f0 in Hz."""
        f0_Hz = f0_mHz * 1e-3
        
        min_val = -2e-20 * (f0_Hz / 4e-4) ** (16 / 3)
        max_val = 3e-21 * (f0_Hz / 1e-4) ** (11 / 3)
        return min_val, max_val

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        # 1. Sample f0 in mHz
        f0_mHz_samples = self.xp.random.uniform(self.f0_min, self.f0_max, size=size)
        
        # 2. Get bounds and sample fdot
        fdot_min, fdot_max = self._get_fdot_bounds(f0_mHz_samples)
        fdot_samples = self.xp.random.uniform(fdot_min, fdot_max, size=size)
        
        samples = self.xp.stack([f0_mHz_samples, fdot_samples], axis=-1)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != 2:
            raise ValueError("Expected last dimension of x to be 2.")

        f0_mHz = x_arr[..., 0]
        fdot = x_arr[..., 1]
        
        out = self.xp.full_like(f0_mHz, -self.xp.inf, dtype=self.xp.float64)
        
        mask_f0 = (f0_mHz >= self.f0_min) & (f0_mHz <= self.f0_max)
        fdot_min, fdot_max = self._get_fdot_bounds(f0_mHz)
        mask_fdot = (fdot >= fdot_min) & (fdot <= fdot_max)
        
        valid = mask_f0 & mask_fdot
        valid_fdot_diff = self.xp.where(valid, fdot_max - fdot_min, 1.0)
        
        # p(f0, fdot) = p(f0) * p(fdot|f0)
        f0_log_norm = -self.xp.log(self.f0_max - self.f0_min)
        out[valid] = f0_log_norm - self.xp.log(valid_fdot_diff[valid])
        
        return self._to_device(out)
    
    
    
