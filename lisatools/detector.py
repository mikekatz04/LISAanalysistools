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

    """

    Soms_d: float
    Sa_a: float


class LISAModel(LISAModelSettings, ABC):
    """Model for the LISA Constellation"""

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out


scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2)
proposal = LISAModel((10.0e-12) ** 2, (3.0e-15) ** 2)
mrdv1 = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2)
sangria = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2)

__stock_list_models__ = ["scirdv1", "proposal", "mrdv1", "sangria"]


def get_available_default_lisa_models() -> List[LISAModel]:
    """Get list of default LISA models

    Returns:
        List of LISA models.

    """
    return __stock_list_models__


def get_default_lisa_model_from_str(model: str) -> LISAModel:
    if model not in __stock_list_models__:
        raise ValueError(
            "Requested string model is not available. See lisatools.detector documentation."
        )
    return globals()[model]


def check_lisa_model(model: Any) -> LISAModel:
    if isinstance(model, str):
        model = get_default_lisa_model_from_str(model)

    if not isinstance(model, LISAModel):
        raise ValueError("model argument not given correctly.")

    return model
