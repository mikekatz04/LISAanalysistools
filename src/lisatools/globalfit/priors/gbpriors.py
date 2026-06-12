from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from .gfpriors import BaseSourcePrior


class GalacticBinaryPrior(BaseSourcePrior):
    """Galactic Binary specific prior infrastructure.
    
    If left as None, the initialisation defaults to a standard 8-parameter 
    sampling basis mapped to a 9-parameter physical basis (injecting \ddot{f} = 0).
    """
    
    def __init__(
        self,
        param_priors: Dict[str | Tuple[str, ...], Callable[..., Any] | Any] | ProbDistContainer,
        param_prior_inputs: Optional[Dict[str | Tuple[str, ...], List[Any]] | List[List[Any]]] = None,
        source_name: Optional[str] = None,
        sampling_params: Optional[List[str]] = None,
        physical_params: Optional[List[str]] = None,
        fill_dict: Optional[Dict[str, float]] = None,
        param_transforms: Optional[Dict[str | Tuple[str, ...], Callable[..., Any]] | TransformContainer] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False
    ):  
        
        source_name = source_name if source_name is not None else "gb"
        
        if sampling_params is None:
            sampling_params = [
                r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", 
                r"$\cos\iota$", r"$\psi$", r"$\alpha$", r"$\sin\delta$"
            ]
        
        if physical_params is None:
            physical_params = [
                r"$A$", r"$f_0$", r"$\dot{f}$", r"$\ddot{f}$", 
                r"$\phi_0$", r"$\iota$", r"$\psi$", r"$\alpha$", r"$\delta$"
            ]
            
        if fill_dict is None:
            fill_dict = {r"$\ddot{f}$": 0.0}
            
        if param_transforms is None:
            param_transforms = {
                r"$\log A$": np.exp,
                r"$\cos\iota$": np.arccos,
                r"$\sin\delta$": np.arcsin,
            }
            
        super().__init__(
            source_name=source_name,
            sampling_params=sampling_params,
            param_priors=param_priors,
            param_prior_inputs=param_prior_inputs,
            physical_params=physical_params,
            fill_dict=fill_dict,
            param_transforms=param_transforms,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            verbose=verbose
        )


def f_Hz_to_mHz(f_Hz):
    """Convert frequency from Hz to mHz."""
    return f_Hz * 1e3

def get_fdot_mojito(f, sign="+"):
    """Get fdot of GW according to predetermined lisa priors for mojito    

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        sign (str): Either positive or negative fdot bound.

    Returns:
        xp.ndarray: fdot.

    Raises:
        ValueError: Inputs are incorrect.
    """
    assert f is not None
    
    if sign == "+":
        fdot = 3e-21 * (f / 1e-4) ** (11/3)
    elif sign == "-":
        fdot = -2e-20 * (f / 4e-4) ** (16/3)
    else:
        raise ValueError("sign must be either positive '+' or negative '-'.")

    return fdot


