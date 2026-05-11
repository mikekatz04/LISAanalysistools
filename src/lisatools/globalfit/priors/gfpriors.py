import warnings
import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp
        
ArrayType = np.ndarray| cp.ndarray
    
from eryn.prior import ProbDistContainer
from eryn.utils import TransformContainer


class BaseSourcePrior:
    """Unified infrastructure for LISA source priors and transformations.
    
    This class wraps Eryn's ProbDistContainer and TransformContainer, managing 
    the inputs, checking for inconsistencies, and exposing an easy-to-use API 
    per source (e.g., MBHB, GB).

    Args:
        source_name (str): Identifier for the source (e.g., 'mbh', 'gb').
        sampling_params (List[str]): Parameters in the sampling basis.
        param_priors (Dict[Union[str, Tuple[str, ...]], Union[Callable, Any]]): 
            Dictionary mapping sampling params to either initialized Eryn priors 
            (objects with .logpdf) or callable prior functions (like uniform_dist).
        param_prior_inputs (Optional[Union[Dict, List[List[Any]]]]): Inputs for 
            the callables in `param_priors`. Dict mapping is preferred for safety.
        physical_params (Optional[List[str]]): Parameters in the physical basis. 
            Defaults to `sampling_params` if not provided.
        fill_dict (Optional[Dict[str, float]]): Fixed parameters to inject during transforms.
        param_transforms (Optional[Union[Dict, TransformContainer]]): Callables or 
            an already initialized TransformContainer mapping sampling to physical.
        periodic (Optional[Dict[str, float]]): Periodic boundaries for parameters.
        use_cupy (bool): Whether to use CuPy for GPU acceleration.
        return_gpu (bool): Whether Eryn should return arrays on the GPU.
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
        param_transforms: Optional[Dict[str | Tuple[str, ...], Callable] | TransformContainer] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False
    ):
        self.source_name = source_name
        self.sampling_params = sampling_params
        self.physical_params = physical_params if physical_params is not None else sampling_params.copy()
        
        self.fill_dict = fill_dict or {}
        
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu
        self.verbose = verbose
        
        self._xp = cp if self.use_cupy else np

        if self.verbose:
            self._run_sanity_checks(param_transforms)

        # set transform container
        if param_transforms is None:
            self.transform_container = None
            
        elif isinstance(param_transforms, dict):
            self.transform_container = TransformContainer(
                input_basis=self.sampling_params,
                output_basis=self.physical_params,
                parameter_transforms=param_transforms,
                fill_dict=self.fill_dict
            )
        else:
            assert isinstance(param_transforms, TransformContainer)
            self.transform_container = param_transforms

        # initialize priors
        if isinstance(param_priors, ProbDistContainer):
            param_priors.use_cupy = self.use_cupy
            param_priors.return_gpu = self.return_gpu
            self.priors = {self.source_name: param_priors}
        else:
            assert param_prior_inputs, "No param_prior_inputs provided"
            initialized_priors = self._initialize_priors(param_priors, param_prior_inputs)
            
            self.priors = {
                self.source_name: ProbDistContainer(
                    initialized_priors, 
                    use_cupy=self.use_cupy, 
                    return_gpu=self.return_gpu
                )
            }

    def _initialize_priors(
        self, 
        param_priors: Dict[str | Tuple[str, ...], Callable | Any], 
        param_prior_inputs: Dict[str | Tuple[str, ...], List[Any]] | List[List[Any]]
    ) -> Dict[str | Tuple[str, ...], Callable]:
        """Parses the prior dict to instantiate and optionally GPU-wrap distributions."""
        priors_out = {}
        
        for i, (param, prior_func) in enumerate(param_priors.items()):
            obj = prior_func
            
            if not hasattr(prior_func, "logpdf") and callable(prior_func):
                args = []
                if isinstance(param_prior_inputs, dict):
                    args = param_prior_inputs.get(param, [])
                elif isinstance(param_prior_inputs, list):
                    try:
                        args = param_prior_inputs[i]
                    except IndexError:
                        raise ValueError(f"Missing inputs in `param_prior_inputs` list for index {i} (key: {param}).")
                
                sig = inspect.signature(prior_func)
                kwargs = {}
                if "use_cupy" in sig.parameters:
                    kwargs["use_cupy"] = self.use_cupy
                if "return_gpu" in sig.parameters:
                    kwargs["return_gpu"] = self.return_gpu
                
                obj = prior_func(*args, **kwargs)
                
            elif not hasattr(prior_func, "logpdf"): # TODO make this a general distribution class
                raise TypeError(
                    f"Prior for {param} must be an initialized distribution (with .logpdf) "
                    f"or a callable function. Got {type(prior_func)}."
                )

            if not hasattr(obj, "use_cupy") or not hasattr(obj, "return_gpu"):
                obj = _GPUPriorWrapper(obj)
            else:
                try:
                    setattr(obj, "use_cupy", self.use_cupy)
                    setattr(obj, "return_gpu", self.return_gpu)
                except Exception:
                    pass
                
            priors_out[param] = obj
                
        return priors_out

    def logpdf(self, x: ArrayType, keys: Optional[List[str]] = None) -> ArrayType:
        """Get the log probability of the sample. Routes to ProbDistContainer."""
        return self.priors[self.source_name].logpdf(x, keys=keys)

    def pdf(self, x: ArrayType, keys: Optional[List[str]] = None) -> ArrayType:
        """Get the probability density of the sample."""
        log_prob = self.logpdf(x, keys=keys)
        
        if self.return_gpu:
            return cp.exp(cp.asarray(log_prob))
        else:
            log_prob_cpu = getattr(log_prob, "get")() if hasattr(log_prob, "get") else log_prob
            return np.exp(log_prob_cpu)

    def rvs(self, size: int | Tuple[int, ...] = 1, keys: Optional[List[str]] = None) -> ArrayType:
        """Sample from the prior distribution. Routes to Eryn."""
        return self.priors[self.source_name].rvs(size=size, keys=keys)
    
    def _run_sanity_checks(self, param_transforms):
        """Runs non-breaking checks to ensure code design intents are met."""
        
        # Check if bases differ but no transform is supplied
        if set(self.sampling_params) != set(self.physical_params) and param_transforms is None:
            warnings.warn(
                f"[{self.source_name}] `physical_params` and `sampling_params` differ, "
                "but `param_transforms` is None. Expected transformations."
            )

        # Check if fill_dict introduces parameters not in the output basis
        if self.fill_dict:
            missing_in_output = [k for k in self.fill_dict.keys() if k not in self.physical_params]
            if missing_in_output:
                warnings.warn(
                    f"[{self.source_name}] `fill_dict` keys {missing_in_output} "
                    "are not in `physical_params` (output basis)."
                )
    
    
class _GPUPriorWrapper:
    """Wraps prior distributions to avoid any CPU vs GPU conflicts.
    
    Dynamically deduces if the underlying object requires CPU or GPU arrays
    using a single shared state flag. It intercepts memory placement issues 
    and seamlessly outputs exactly what Eryn dictates via use_cupy/return_gpu.
    """
    def __init__(self, prior_obj: Any):
        self._prior_obj = prior_obj
        
        self.use_cupy = None
        self.return_gpu = None
        
        # CPU Flag: None = unknown, True = needs CPU, False = GPU supported
        self._needs_cpu = None

        if hasattr(self._prior_obj, "rvs"):
            self.rvs = self._rvs_wrapped
            
        if hasattr(self._prior_obj, "logpdf"):
            self.logpdf = self._logpdf_wrapped
            
        if hasattr(self._prior_obj, "logpmf"):
            self.logpmf = self._logpmf_wrapped
            
        if hasattr(self._prior_obj, "pdf"):
            self.pdf = self._pdf_wrapped

    def __getattr__(self, name: str):
        return getattr(self._prior_obj, name)

    def _format_output(self, res: Any) -> ArrayType:
        """Ensures the output strictly matches what Eryn expects."""
        if self.return_gpu:
            return cp.asarray(res)
        return res.get() if hasattr(res, "get") else res

    def _evaluate(self, func_name: str, x: ArrayType, *args, **kwargs):
        """Unified runner for pdf, logpdf, and logpmf that checks memory placement."""
        func = getattr(self._prior_obj, func_name)
        
        if self._needs_cpu is False:
            res = func(x, *args, **kwargs)
            
        elif self._needs_cpu is True:
            x_in = getattr(x, "get")() if hasattr(x, "get") else x
            res = func(x_in, *args, **kwargs)
        
        # First attempt
        else:
            try:
                res = func(x, *args, **kwargs)
                self._needs_cpu = False
            except (TypeError, AttributeError, ValueError):
                self._needs_cpu = True
                x_in = getattr(x, "get")() if hasattr(x, "get") else x
                res = func(x_in, *args, **kwargs)
                
        return self._format_output(res)

    def _logpdf_wrapped(self, x: ArrayType, *args, **kwargs):
        return self._evaluate("logpdf", x, *args, **kwargs)
        
    def _logpmf_wrapped(self, x: ArrayType, *args, **kwargs):
        return self._evaluate("logpmf", x, *args, **kwargs)

    def _pdf_wrapped(self, x: ArrayType, *args, **kwargs):
        return self._evaluate("pdf", x, *args, **kwargs)

    def _rvs_wrapped(self, size: int | Tuple[int, ...] = 1, *args, **kwargs):
        """Draws samples. Does not take 'x', so it won't crash on input type."""
        res = self._prior_obj.rvs(size=size, *args, **kwargs)
        
        if self._needs_cpu is None:
            self._needs_cpu = not hasattr(res, "get")
            
        return self._format_output(res)