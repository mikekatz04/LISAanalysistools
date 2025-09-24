from copy import deepcopy
import numpy as np
import cupy as cp
from ..detector import sangria, EqualArmlengthOrbits
from ..sensitivity import get_sensitivity, SensitivityMatrix


class BasicResidualacsLikelihood:
    def __init__(self, acs):
        self.acs = acs

    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.acs.likelihood()
        overall_inds = supps["overal_inds"]
        breakpoint()
        return ll_temp[overall_inds]


def new_sens_mat(name, psd_params, galfor_params, f_arr):
    orbits = EqualArmlengthOrbits()
    A_params = psd_params[:2]
    E_params = psd_params[2:]

    # amp, fk, alpha, s1, s2 = galfor_params
    # lisa_model_A = LISAModel(A_Soms_d**2, A_Sa_a**2, orbits, f"{name}_A_channel")
    # lisa_model_E = LISAModel(E_Soms_d**2, E_Sa_a**2, orbits, f"{name}_E_channel")
    # psd_kwargs = [
    #     dict(
    #         model=lisa_model_A,
    #         stochastic_params=galfor_params,
    #         stochastic_function=HyperbolicTangentGalacticForeground
    #     ),
    #     dict(
    #         model=lisa_model_E,
    #         stochastic_params=galfor_params,
    #         stochastic_function=HyperbolicTangentGalacticForeground
    #     ),
    # ]
    tmp_lisa_model_A = deepcopy(sangria)
    tmp_lisa_model_E = deepcopy(sangria)

    # TODO: use PSD generating function from general setup
    tmp_lisa_model_A.Soms_d = A_params[0] ** 2
    tmp_lisa_model_A.Sa_a = A_params[1] ** 2
    A_tmp1 = get_sensitivity(f_arr, model=tmp_lisa_model_A, stochastic_params=galfor_params, stochastic_function="HyperbolicTangentGalacticForeground",  sens_fn="A1TDISens")
    A_tmp1[0] = A_tmp1[1]

    tmp_lisa_model_E.Soms_d = E_params[0] ** 2
    tmp_lisa_model_E.Sa_a = E_params[1] ** 2
    E_tmp1 = get_sensitivity(f_arr, model=tmp_lisa_model_E, stochastic_params=galfor_params, stochastic_function="HyperbolicTangentGalacticForeground",  sens_fn="A1TDISens")
    E_tmp1[0] = E_tmp1[1]
          
    sens_AE = SensitivityMatrix(f_arr.copy(), np.asarray([A_tmp1, E_tmp1]))
    return sens_AE


from dataclasses import dataclass


class AllSetupInfoTransfer:
    def __init__(self, setup_1, setup_2):
        # TODO: cleanup
        for key in ["name", "in_model_moves", "rj_moves"]:
            if key != "name":
                tmp1 = getattr(setup_1, key)
                tmp2 = getattr(setup_2, key)
            if isinstance(setup_1, AllSetupInfoTransfer) and isinstance(setup_2, AllSetupInfoTransfer):
                if key == "name":
                    self.setup_names = setup_1.setup_names + setup_2.setup_names
                    continue
                
                if isinstance(tmp1, list):
                    assert isinstance(tmp2, list)
                    tmp_out = tmp1 + tmp2

                setattr(self, key,  tmp_out)
                
            elif isinstance(setup_1, SetupInfoTransfer) and isinstance(setup_2, SetupInfoTransfer):
                if key == "name":
                    self.setup_names = [setup_1.name, setup_2.name]
                    continue
                if isinstance(tmp1, list):
                    assert isinstance(tmp2, list)
                    tmp_out = tmp1 + tmp2
                    
                setattr(self, key,  tmp_out)

            else:
                if not isinstance(setup_2, SetupInfoTransfer):
                    # switch so all setup is in position 1
                    tmp = setup_1
                    setup_1 = setup_2
                    setup_2 = tmp

                if key == "name":
                    self.setup_names = setup_1.setup_names + [setup_2.name]
                    continue

                if isinstance(tmp1, list):
                    assert isinstance(tmp2, list)
                    tmp_out = tmp1 + tmp2
                    
                setattr(self, key,  tmp_out)

    def __add__(self, setup_2):
        return AllSetupInfoTransfer(self, setup_2)
    
    @property
    def setup_names(self):
        return self._setup_names
    
    @setup_names.setter
    def setup_names(self, setup_names):
        assert isinstance(setup_names, list)
        self._setup_names = setup_names

    @property
    def rj_moves(self):
        return self._rj_moves
    
    @rj_moves.setter
    def rj_moves(self, rj_moves):
        assert isinstance(rj_moves, list)
        self._rj_moves = rj_moves

    @property
    def in_model_moves(self):
        return self._in_model_moves
    
    @in_model_moves.setter
    def in_model_moves(self, in_model_moves):
        assert isinstance(in_model_moves, list)
        self._in_model_moves = in_model_moves

    def unwrap_moves(self, moves):
        for i in range(len(moves)):
            if isinstance(moves[i], tuple):
                assert isinstance(moves[i][1], float)
            else:
                moves[i] = (moves[i], 1.0)
        return moves
        
    @property
    def rj_moves_input(self) -> list:
        if self.rj_moves == []:
            return None

        return self.unwrap_moves(self.rj_moves)

    @property
    def in_model_moves_input(self) -> list:
        if self.in_model_moves == []:
            return None

        return self.unwrap_moves(self.in_model_moves)

    


# TODO: this was not working?
# @dataclass
class SetupInfoTransfer:
    # name = None
    # in_model_moves = []
    # rj_moves = []

    def __init__(self, 
        name = None,
        in_model_moves = [],
        rj_moves = [],
    ):
        self.name = name
        self.in_model_moves = in_model_moves
        self.rj_moves = rj_moves

    def __add__(self, setup_2):
        return AllSetupInfoTransfer(self, setup_2)

