from abc import ABC, abstractmethod
from typing import Any, List
from dataclasses import dataclass


@dataclass
class LISAModelSettings:
    """Required LISA model settings:

    TODO: rename these

    Args:
        Soms_d: OMS displacement noise.
        Sa_a: Acceleration noise.
        name: Name of model.

    """

    Soms_d: float
    Sa_a: float
    name: str


class LISAModel(LISAModelSettings, ABC):
    """Model for the LISA Constellation"""

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out


scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, "scirdv1")
proposal = LISAModel((10.0e-12) ** 2, (3.0e-15) ** 2, "proposal")
mrdv1 = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, "mrdv1")
sangria = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, "sangria")

__stock_list_models__ = [scirdv1, proposal, mrdv1, sangria]
__stock_list_models_name__ = [tmp.name for tmp in __stock_list_models__]


def get_available_default_lisa_models() -> List[LISAModel]:
    """Get list of default LISA models

    Returns:
        List of LISA models.

    """
    return __stock_list_models__


def get_default_lisa_model_from_str(model: str) -> LISAModel:
    """Return a LISA model from a ``str`` input.

    Args:
        model: Model indicated with a ``str``.

    Returns:
        LISA model associated to that ``str``.

    """
    if model not in __stock_list_models_name__:
        raise ValueError(
            "Requested string model is not available. See lisatools.detector documentation."
        )
    return globals()[model]


def check_lisa_model(model: Any) -> LISAModel:
    """Check input LISA model.

    Args:
        model: LISA model to check.

    Returns:
        LISA Model checked. Adjusted from ``str`` if ``str`` input.

    """
    if isinstance(model, str):
        model = get_default_lisa_model_from_str(model)

    if not isinstance(model, LISAModel):
        raise ValueError("model argument not given correctly.")

    return model
