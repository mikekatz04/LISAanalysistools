import warnings
import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp
    _CUPY_AVAILABLE = False

ArrayType = np.ndarray | cp.ndarray

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer


class BaseSourcePrior:
    """Unified infrastructure for LISA source priors and transformations.

    This class wraps Eryn's ProbDistContainer and TransformContainer, managing
    the inputs, checking for inconsistencies, and exposing an easy-to-use API
    per source (e.g., MBHB, GB).

    Args:
        source_name (str): Identifier for the source (e.g., 'mbhb', 'gb').
        sampling_params (List[str]): Parameters in the sampling basis.
        param_priors (Dict[Union[str, Tuple[str, ...]], Union[Callable, Any]]):
            Dictionary mapping sampling params to either initialized general distributions
            (objects with .logpdf) or callable prior functions (like uniform_dist).
        param_prior_inputs (Optional[Union[Dict, List[List[Any]]]]): Inputs for
            the callables in `param_priors`. Dict mapping is preferred for safety.
            May be omitted only if all priors are already initialized (have .logpdf).
        physical_params (Optional[List[str]]): Parameters in the physical basis.
            Defaults to `sampling_params` if not provided.
        fill_dict (Optional[Dict[str, float]]): Fixed parameters to inject during transforms.
        key_map (Optional[Dict[str, str]]): Mapping from sampling to physical param names
            for cases where they differ and no TransformContainer is provided.
        param_transforms (Optional[Union[Dict, TransformContainer]]): Callables or
            an already initialized TransformContainer mapping sampling to physical.
        periodic (Optional[Dict[str, float]]): Periodic boundaries for parameters.
        use_cupy (bool): Whether to use CuPy for GPU acceleration.
        return_gpu (bool): Whether the priors should return arrays on the GPU.
        verbose (bool): If True, enables sanity check warnings.
    """
    def __init__(
        self,
        source_name: str,
        sampling_params: List[str],
        param_priors: Dict[str | Tuple[str, ...], Callable | Any] | ProbDistContainer,
        param_prior_inputs: Optional[Dict[str | Tuple[str, ...], List[Any]] | List[List[Any]]] = None,
        physical_params: Optional[List[str]] = None,
        fill_dict: Optional[Dict[str, float]] = None,
        key_map: Optional[Dict[str, str]] = None,
        param_transforms: Optional[Dict[str | Tuple[str, ...], Callable] | TransformContainer] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False,
    ):
        self.source_name = source_name
        self.sampling_params = sampling_params
        self.physical_params = physical_params if physical_params is not None else sampling_params.copy()

        self.fill_dict = fill_dict or {}
        self.key_map = key_map or {}
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu
        self.verbose = verbose

        self._xp = cp if self.use_cupy else np

        # verbose controls the broader set of optional checks below.
        self._check_fill_dict()

        if self.verbose:
            self._run_checks(param_transforms)

        # Build transform container
        if param_transforms is None:
            self.transform_container = None
        elif isinstance(param_transforms, dict):
            self.transform_container = TransformContainer(
                input_basis=self.sampling_params,
                output_basis=self.physical_params,
                parameter_transforms=param_transforms,
                fill_dict=self.fill_dict,
                key_map=self.key_map,
            )
        else:
            assert isinstance(param_transforms, TransformContainer)
            self.transform_container = param_transforms

        # Build prior container
        if isinstance(param_priors, ProbDistContainer):
            param_priors.use_cupy = self.use_cupy
            param_priors.return_gpu = self.return_gpu
            self.priors = {self.source_name: param_priors}
        else:
            if param_prior_inputs is None:
                all_initialized = all(
                    hasattr(p, "logpdf") for p in param_priors.values()
                )
                if not all_initialized:
                    raise ValueError(
                        "`param_prior_inputs` must be provided when any prior "
                        "is a callable that requires arguments."
                    )
                param_prior_inputs = {}

            initialized_priors = self._initialize_priors(param_priors, param_prior_inputs)

            self.priors = {
                self.source_name: ProbDistContainer(
                    initialized_priors,
                    use_cupy=self.use_cupy,
                    return_gpu=self.return_gpu,
                )
            }

    @staticmethod
    def _normalize_prior_inputs(
        param_priors: Dict,
        param_prior_inputs: Dict | List,
    ) -> Dict:
        """Normalise param_prior_inputs to a dict keyed by param name.

        Accepts either a dict (used as-is) or a positional list (zipped against
        the param_priors key order). The loop body then does a uniform
        `.get(param, [])` instead of inline type-branching.
        """
        if isinstance(param_prior_inputs, dict):
            return param_prior_inputs

        keys = list(param_priors.keys())
        if len(param_prior_inputs) > len(keys):
            raise ValueError(
                f"`param_prior_inputs` list has {len(param_prior_inputs)} entries "
                f"but `param_priors` has only {len(keys)} keys."
            )
        return {keys[i]: args for i, args in enumerate(param_prior_inputs)}

    def _initialize_priors(
        self,
        param_priors: Dict[str | Tuple[str, ...], Callable | Any],
        param_prior_inputs: Dict[str | Tuple[str, ...], List[Any]] | List[List[Any]],
    ) -> Dict[str | Tuple[str, ...], Any]:
        """Instantiate callable priors and wrap every distribution in _GPUPriorWrapper."""

        inputs = self._normalize_prior_inputs(param_priors, param_prior_inputs)

        priors_out = {}
        for param, prior_func in param_priors.items():
            obj = prior_func

            if not hasattr(prior_func, "logpdf"):
                if not callable(prior_func):
                    raise TypeError(
                        f"Prior for '{param}' must be an initialized distribution "
                        f"(with .logpdf) or a callable. Got {type(prior_func)}."
                    )

                args = inputs.get(param, [])

                sig = inspect.signature(prior_func)
                kwargs: Dict[str, Any] = {}
                if "use_cupy" in sig.parameters:
                    kwargs["use_cupy"] = self.use_cupy
                if "return_gpu" in sig.parameters:
                    kwargs["return_gpu"] = self.return_gpu

                obj = prior_func(*args, **kwargs)

            if not isinstance(obj, _GPUPriorWrapper):
                obj = _GPUPriorWrapper(obj)

            priors_out[param] = obj

        return priors_out

    def logpdf(self, x: ArrayType, *args, keys: Optional[List[str]] = None, **kwargs) -> ArrayType:
        """Log probability of the sample."""
        return self.priors[self.source_name].logpdf(x, *args, keys=keys, **kwargs)

    def pdf(self, x: ArrayType, *args, keys: Optional[List[str]] = None, **kwargs) -> ArrayType:
        """Probability density of the sample."""
        if hasattr(self.priors[self.source_name], "pdf"):
            return getattr(self.priors[self.source_name], "pdf")(x, *args, keys=keys, **kwargs)
        else:
            log_prob = self.logpdf(x, *args, keys=keys, **kwargs)
            if self.return_gpu:
                return cp.exp(cp.asarray(log_prob))
            log_prob_cpu = getattr(log_prob, "get")() if (_CUPY_AVAILABLE and isinstance(log_prob, cp.ndarray)) else log_prob
            return np.exp(log_prob_cpu)

    def rvs(self, *args, size: int | Tuple[int, ...] = 1, keys: Optional[List[str]] = None, **kwargs) -> ArrayType:
        """Sample from the prior."""
        return self.priors[self.source_name].rvs(*args, size=size, keys=keys, **kwargs)

    def _check_fill_dict(self) -> None:
        """Always-on fill_dict consistency check (not verbose-gated).

        fill_dict keys must appear in physical_params because TransformContainer
        calls output_basis.index(key) for each one; a mismatch causes a hard
        IndexError there. Warn early and clearly instead.
        """
        if self.fill_dict:
            missing = [k for k in self.fill_dict if k not in self.physical_params]
            if missing:
                warnings.warn(
                    f"[{self.source_name}] `fill_dict` keys {missing} are not in "
                    "`physical_params` (output basis). This will cause an error in "
                    "TransformContainer.",
                    stacklevel=3,
                )

    def _run_checks(self, param_transforms: Any) -> None:
        """Optional verbose checks for common design-intent mismatches."""
        if set(self.sampling_params) != set(self.physical_params) and param_transforms is None:
            warnings.warn(
                f"[{self.source_name}] `physical_params` and `sampling_params` differ "
                "but `param_transforms` is None. Transformations were expected.",
                stacklevel=3,
            )
            
        # if self.transform_container is not None:
        #     missing_keys = set(self.sampling_params) - set(self.transform_container.input_basis)
        #     if missing_keys:
        #         warnings.warn(
        #             f"[{self.source_name}] The following sampling parameters are missing "
        #             f"in the transform container's input_basis: {missing_keys}",
        #             stacklevel=3,
        #         )
        
        # if self.key_map:
        #     unmapped = set(self.sampling_params) - set(self.key_map.keys())
        #     if unmapped:
        #         warnings.warn(
        #             f"[{self.source_name}] The following sampling parameters are missing "
        #             f"in `key_map`: {unmapped}. This may cause issues if no "
        #             "TransformContainer is provided.",
        #             stacklevel=3,
        #         )



class UniformSourcePrior(BaseSourcePrior):
    """Convenience class for sources parameterized entirely by uniform priors.
    
    This class wraps `BaseSourcePrior`. Instead of passing initialized prior 
    distribution objects, the user provides a dictionary of bounds `(min, max)`.
    The class automatically maps these to Eryn's `uniform_dist` callables.

    Args:
        source_name (str): Identifier for the source (e.g., 'gb', 'mbhb').
        sampling_params (List[str]): List of parameter names exactly as they will be sampled by Eryn.
        param_limits (Dict[str | Tuple[str, ...], Tuple[float, float]]): Dictionary mapping sampling 
            parameter names to their (min, max) boundaries. These boundaries MUST correspond to the 
            SAMPLING basis, not the physical basis.
        physical_params (Optional[List[str]]): List of parameter names in the physical evaluation basis. 
            If None, assumes the physical basis is identical to the sampling basis.
        fill_dict (Optional[Dict[str, float]]): Dictionary mapping physical parameters to fixed constant 
            values. Used to inject parameters that are not sampled.
            Dictionary mapping physical parameters to fixed constant values. Used to inject parameters that 
            are not sampled.
        param_transforms (Optional[Dict[str | Tuple[str, ...], Callable] | TransformContainer], default=None):
            Mapping of sampling parameters to callable functions (e.g., `np.exp`) that transform them into 
            the physical basis. Alternatively, a fully initialized `TransformContainer`.
        key_map (Optional[Dict[str, str]], default=None): Dictionary bridging naming mismatches between 
            `sampling_params` and `physical_params` during transformations (e.g., `{"logA": "A"}`).
        use_cupy (bool, default=False): Whether to instantiate distributions with CuPy acceleration.
        return_gpu (bool, default=False): Whether Eryn should return outputs explicitly on the GPU memory space.
        verbose (bool, default=False): If True, issues warnings for potentially misaligned bases or missing parameters.

    Notes:
        If a parameter requires a non-uniform prior (e.g., Gaussian or Normalizing Flow), 
        do not use this class. Instead, initialize `BaseSourcePrior` directly and 
        pass the instantiated distributions.
    """
    def __init__(
        self,
        source_name: str,
        sampling_params: List[str],
        param_limits: Dict[str | Tuple[str, ...], Tuple[float, float]],
        physical_params: Optional[List[str]] = None,
        fill_dict: Optional[Dict[str, float]] = None,
        param_transforms: Optional[Dict[str | Tuple[str, ...], Callable] | TransformContainer] = None,
        key_map: Optional[Dict[str, str]] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False,
    ):
        # Map boundaries to uniform_dist callables
        param_priors = {k: uniform_dist for k in param_limits.keys()}
        param_prior_inputs = {k: list(v) for k, v in param_limits.items()}

        super().__init__(
            source_name=source_name,
            sampling_params=sampling_params,
            param_priors=param_priors,
            param_prior_inputs=param_prior_inputs,
            physical_params=physical_params,
            fill_dict=fill_dict,
            param_transforms=param_transforms,
            key_map=key_map,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            verbose=verbose
        )



class _GPUPriorWrapper:
    """Wraps prior distributions to handle CPU/GPU placement transparently.

    Intercepts calls to logpdf, logpmf, pdf, and rvs. On first call per method,
    probes whether the underlying distribution accepts GPU arrays. If it raises,
    falls back to CPU input. Subsequent calls skip the probe and use the cached
    result. Output placement is left entirely to ProbDistContainer.

    Per-method `_needs_cpu` flags prevent cross-method state contamination (e.g.
    a distribution that accepts GPU arrays in rvs but not in logpdf).
    """
    def __init__(self, prior_obj: Any):
        self._prior_obj = prior_obj

        # Per-method flag: None = unknown, True = needs CPU input, False = GPU ok
        self._needs_cpu: Dict[str, Optional[bool]] = {
            "logpdf": None,
            "logpmf": None,
            "pdf":    None,
            "rvs":    None,
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate any attribute not defined on the wrapper to the inner object."""
        return getattr(self._prior_obj, name)

    def _call_with_cpu_fallback(self, func_name: str, *args, **kwargs) -> Any:
        """Unified probe-and-retry runner for all intercepted methods.

        On the first call for a given method name the distribution is called
        directly. If it raises a device-placement error the inputs are moved to
        CPU and the call is retried. The result of the probe is cached in
        `_needs_cpu[func_name]` so subsequent calls skip the try/except.

        Output is returned as-is; device placement is ProbDistContainer's job.
        """
        func = getattr(self._prior_obj, func_name)
        needs_cpu = self._needs_cpu[func_name]

        def _to_cpu(a: Any) -> Any:
            return a.get() if (_CUPY_AVAILABLE and isinstance(a, cp.ndarray)) else a

        if needs_cpu is True:
            args = tuple(_to_cpu(a) for a in args)
            return func(*args, **kwargs)

        if needs_cpu is False:
            return func(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
            self._needs_cpu[func_name] = False
            return result
        except (TypeError, AttributeError, ValueError, RuntimeError, NotImplementedError):
            self._needs_cpu[func_name] = True
            args = tuple(_to_cpu(a) for a in args)
            return func(*args, **kwargs)

    def logpdf(self, x: ArrayType, *args, **kwargs) -> ArrayType:
        if not hasattr(self._prior_obj, "logpdf"):
            raise AttributeError(f"{type(self._prior_obj).__name__} has no logpdf method.")
        return self._call_with_cpu_fallback("logpdf", x, *args, **kwargs)

    def logpmf(self, x: ArrayType, *args, **kwargs) -> ArrayType:
        if not hasattr(self._prior_obj, "logpmf"):
            raise AttributeError(f"{type(self._prior_obj).__name__} has no logpmf method.")
        return self._call_with_cpu_fallback("logpmf", x, *args, **kwargs)

    def pdf(self, x: ArrayType, *args, **kwargs) -> ArrayType:
        if not hasattr(self._prior_obj, "pdf"):
            raise AttributeError(f"{type(self._prior_obj).__name__} has no pdf method.")
        return self._call_with_cpu_fallback("pdf", x, *args, **kwargs)

    def rvs(self, size: int | Tuple[int, ...] = 1, *args, **kwargs) -> ArrayType:
        if not hasattr(self._prior_obj, "rvs"):
            raise AttributeError(f"{type(self._prior_obj).__name__} has no rvs method.")
        return self._call_with_cpu_fallback("rvs", *args, size=size, **kwargs)
    
    
    