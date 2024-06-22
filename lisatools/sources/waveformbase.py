from abc import ABC
from typing import Union, Tuple


class AETTDIWaveform(ABC):
    # @classmethod
    # @property
    # def domain_variables(self) -> dict:
    #     breakpoint()
    #     return {"dt": self.dt, "f_arr": self.f_arr, "df": self.df}

    @property
    def dt(self) -> float:
        return None

    @property
    def f_arr(self) -> float:
        return None

    @property
    def df(self) -> float:
        return None


class SNRWaveform(ABC):
    # @classmethod
    # @property
    # def domain_variables(self) -> dict:
    #     breakpoint()
    #     return {"dt": self.dt, "f_arr": self.f_arr, "df": self.df}

    @property
    def dt(self) -> float:
        return None

    @property
    def f_arr(self) -> float:
        return None

    @property
    def df(self) -> float:
        return None
