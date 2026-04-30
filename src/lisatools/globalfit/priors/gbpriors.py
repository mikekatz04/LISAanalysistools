import numpy as np


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



