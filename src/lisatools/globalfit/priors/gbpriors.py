"""Galactic-binary prior helpers.

The one function on the production path is :func:`get_fdot_mojito`: the
frequency-dependent ``fdot`` envelopes that set ``GBSettings.fdot_lims`` in
:meth:`~lisatools.globalfit.stock.erebor.gb.GBSetup.init_band_structure` (and
the VGB equivalent), and the legacy-``fdot``-basis third-column search bounds of
:class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJSerialSearchMCMC`.
:class:`GalacticBinaryPrior` (built on
:class:`~lisatools.globalfit.priors.gfpriors.BaseSourcePrior`) and
:func:`f_Hz_to_mHz` have no callers; the stock GB priors are built directly in
``GBSetup.init_sampling_info``.
"""

from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from .gfpriors import BaseSourcePrior


# TODO check: keep/remove/document -- no caller anywhere; stock GB priors are built in GBSetup
class GalacticBinaryPrior(BaseSourcePrior):
    """Galactic Binary specific prior infrastructure.

    If left as None, the initialisation defaults to a standard 8-parameter
    sampling basis mapped to a 9-parameter physical basis (injecting \\ddot{f} = 0).

    Args:
        chirp_mass: When True, the default sampling basis carries **chirp mass
            ``Mc`` (solar masses)** in place of ``fdot``, and the default
            ``param_transforms`` include a multi-parameter
            ``(f0, Mc) -> (f0, fdot)`` map computed via
            :func:`gbgpu.utils.utility.get_fdot` -- so the physical basis still
            receives ``fdot`` while the sampler operates in chirp mass. Off by
            default for backward compatibility. (The stock erebor GB branch
            does not use this class.)
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
        chirp_mass: bool = False,
        key_map: Optional[Dict[str, str]] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False
    ):

        source_name = source_name if source_name is not None else "gb"

        # Third-slot label in the sampling basis: fdot (legacy) or Mc (new).
        third = "Mc" if chirp_mass else "fdot"

        if sampling_params is None:
            sampling_params = [
                "A", "f0", third, "phi0",
                "cos_iota", "psi", "alpha", "sin_delta"
            ]

        if physical_params is None:
            # Physical basis is always the 9-param waveform basis with
            # fdot at slot 2 and fddot at slot 3. In chirp-mass mode the
            # sampling name "Mc" is mapped to the physical name "fdot" via
            # ``key_map`` (see below), and a multi-parameter transform
            # converts Mc -> fdot after fill_values() places the Mc value
            # at the fdot slot.
            physical_params = [
                "A", "f0", "fdot", "fddot",
                "phi0", "cos_iota", "psi", "alpha", "sin_delta"
            ]

        if fill_dict is None:
            fill_dict = {"fddot": 0.0}

        if param_transforms is None:
            if chirp_mass:
                from ..stock.erebor.transforms import mchirp_to_fdot_pair

                param_transforms = {
                    "A": np.exp,
                    ("f0", "Mc"): mchirp_to_fdot_pair,
                    "cos_iota": np.arccos,
                    "sin_delta": np.arcsin,
                }
            else:
                param_transforms = {
                    "A": np.exp,
                    "cos_iota": np.arccos,
                    "sin_delta": np.arcsin,
                }

        if key_map is None and chirp_mass:
            key_map = {"Mc": "fdot"}

        super().__init__(
            source_name=source_name,
            sampling_params=sampling_params,
            param_priors=param_priors,
            param_prior_inputs=param_prior_inputs,
            physical_params=physical_params,
            fill_dict=fill_dict,
            key_map=key_map,
            param_transforms=param_transforms,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            verbose=verbose
        )


# TODO check: keep/remove/document -- no caller anywhere
def f_Hz_to_mHz(f_Hz):
    """Convert frequency from Hz to mHz."""
    return f_Hz * 1e3

def get_fdot_mojito(f, sign="+"):
    r"""Upper or lower ``fdot`` envelope of the mojito GB prior at frequency ``f``.

    .. math::

        \dot f_+ = 3\times10^{-21}\,(f / 10^{-4}\,{\rm Hz})^{11/3},
        \qquad
        \dot f_- = -2\times10^{-20}\,(f / 4\times10^{-4}\,{\rm Hz})^{16/3}
        \quad [{\rm Hz\,s^{-1}}].

    The stock GB and VGB setups evaluate both envelopes at the top of the
    ``f0`` prior range (``f0_lims[1]``) to set ``fdot_lims``.

    Args:
        f (float or xp.ndarray): GW frequency [Hz].
        sign (str): ``"+"`` (default) for the upper envelope, ``"-"`` for
            the lower one.

    Returns:
        float or xp.ndarray: ``fdot`` [Hz/s], same shape and array module as
        ``f``.

    Raises:
        ValueError: ``sign`` is neither ``"+"`` nor ``"-"``.
        AssertionError: ``f`` is ``None``.

    Example:
        >>> from lisatools.globalfit.priors.gbpriors import get_fdot_mojito
        >>> get_fdot_mojito(1e-4)
        3e-21
        >>> get_fdot_mojito(4e-4, sign="-")
        -2e-20
    """
    assert f is not None
    
    if sign == "+":
        fdot = 3e-21 * (f / 1e-4) ** (11/3)
    elif sign == "-":
        fdot = -2e-20 * (f / 4e-4) ** (16/3)
    else:
        raise ValueError("sign must be either positive '+' or negative '-'.")

    return fdot


