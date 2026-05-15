"""Misc utilities used by the global-fit setup code."""

import typing as tp
from copy import deepcopy

import cupy as cp
import numpy as np

from ..detector import EqualArmlengthOrbits, LISAModel, Orbits, sangria
from ..sensitivity import SensitivityMatrix, get_sensitivity


class BasicResidualacsLikelihood:
    """Minimal residual-driven likelihood that delegates to an
    :class:`AnalysisContainerArray`.

    Args:
        acs: The analysis-container array whose ``likelihood()`` method is
            invoked on each call.
    """

    def __init__(self, acs):
        self.acs = acs

    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.acs.likelihood()
        overall_inds = supps["overal_inds"]
        breakpoint()
        return ll_temp[overall_inds]


class NewSensitivityMatrix:
    """Builder that returns a :class:`SensitivityMatrix` for given PSD parameters.

    Args:
        orbits: Detector orbits.
        noise_model: Base :class:`LISAModel`; this is deep-copied per call so
            ``Soms_d`` / ``Sa_a`` can be overridden without mutating the
            original.
        sens_fns: Either a single sensitivity-function callable (returns a
            full matrix in one call) or a list of names that name individual
            channel sensitivities to be assembled.
    """

    def __init__(
        self,
        orbits: Orbits,
        noise_model: LISAModel,
        sens_fns: tp.Union[SensitivityMatrix, tp.List[str]],
    ):

        self.orbits = orbits
        self.noise_model = noise_model
        self.sens_fns = sens_fns

    def __call__(self, name, psd_params, f_arr, galfor_params=None):
        """Return a sensitivity matrix on ``f_arr`` for the supplied parameters.

        Args:
            name: Identifier (currently unused; reserved for future logging).
            psd_params: Flat PSD parameter array. With a list-style
                ``sens_fns`` the array is split into pairs
                ``(Soms_d, Sa_a)`` per channel.
            f_arr: Frequency array on which the sensitivity is evaluated.
            galfor_params: Optional galactic-foreground parameters; if
                ``None``, no stochastic foreground is added.

        Returns:
            Either the result of the user-provided sensitivity function or a
            :class:`SensitivityMatrix` built from the per-channel pieces.
        """

        if galfor_params is None:
            galfor_params = ()
            stochastic_function = None
        else:
            stochastic_function = "HyperbolicTangentGalacticForeground"

        if not isinstance(self.sens_fns, list):
            # this is XYZ
            tmp_lisa_model = deepcopy(self.noise_model)

            tmp_lisa_model.Soms_d = psd_params[0] ** 2
            tmp_lisa_model.Sa_a = psd_params[1] ** 2

            return self.sens_fns(
                f_arr,
                model=tmp_lisa_model,
                stochastic_params=galfor_params,
                stochastic_function=stochastic_function,
            )

        else:
            sens_list = []
            for i, sens_fn in enumerate(self.sens_fns):
                tmp_lisa_model = deepcopy(self.noise_model)

                params_here = psd_params[i * 2 : (i + 1) * 2]

                tmp_lisa_model.Soms_d = params_here[0] ** 2
                tmp_lisa_model.Sa_a = params_here[1] ** 2

                tmp_sens = get_sensitivity(
                    f_arr,
                    model=tmp_lisa_model,
                    stochastic_params=galfor_params,
                    stochastic_function=stochastic_function,
                    sens_fn=sens_fn,
                )
                tmp_sens[0] = tmp_sens[1]  # why?
                sens_list.append(tmp_sens)

            sens_arr = np.asarray(sens_list)

            return SensitivityMatrix(f_arr.copy(), sens_arr)


def new_sens_mat(name, psd_params, f_arr, galfor_params=None):
    """Build a 2-channel ``A1``/``E1`` :class:`SensitivityMatrix` for given PSD/galfor params.

    Args:
        name: Unused identifier.
        psd_params: 4-element array ``[A_Soms_d, A_Sa_a, E_Soms_d, E_Sa_a]``.
        f_arr: Frequencies at which to evaluate the sensitivity.
        galfor_params: Optional galactic-foreground parameters.

    Returns:
        :class:`SensitivityMatrix` of shape ``(2, len(f_arr))``.
    """
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
    if galfor_params is None:
        galfor_params = ()
        stochastic_function = None
    else:
        stochastic_function = "HyperbolicTangentGalacticForeground"

    A_tmp1 = get_sensitivity(
        f_arr,
        model=tmp_lisa_model_A,
        stochastic_params=galfor_params,
        stochastic_function=stochastic_function,
        sens_fn="A1TDISens",
    )
    A_tmp1[0] = A_tmp1[1]

    tmp_lisa_model_E.Soms_d = E_params[0] ** 2
    tmp_lisa_model_E.Sa_a = E_params[1] ** 2
    E_tmp1 = get_sensitivity(
        f_arr,
        model=tmp_lisa_model_E,
        stochastic_params=galfor_params,
        stochastic_function=stochastic_function,
        sens_fn="E1TDISens",
    )
    E_tmp1[0] = E_tmp1[1]

    sens_AE = SensitivityMatrix(f_arr.copy(), np.asarray([A_tmp1, E_tmp1]))
    return sens_AE


from dataclasses import dataclass


class AllSetupInfoTransfer:
    """Aggregate of two :class:`SetupInfoTransfer` instances.

    Combining two setups via ``+`` (or by direct construction) yields an
    :class:`AllSetupInfoTransfer` whose ``in_model_moves`` and ``rj_moves``
    are the concatenations of the inputs and whose ``setup_names`` lists all
    contributing setup names.
    """

    def __init__(self, setup_1, setup_2):
        # TODO: cleanup
        for key in ["name", "in_model_moves", "rj_moves"]:
            if key != "name":
                tmp1 = getattr(setup_1, key)
                tmp2 = getattr(setup_2, key)
            if isinstance(setup_1, AllSetupInfoTransfer) and isinstance(
                setup_2, AllSetupInfoTransfer
            ):
                if key == "name":
                    self.setup_names = setup_1.setup_names + setup_2.setup_names
                    continue

                if isinstance(tmp1, list):
                    assert isinstance(tmp2, list)
                    tmp_out = tmp1 + tmp2

                setattr(self, key, tmp_out)

            elif isinstance(setup_1, SetupInfoTransfer) and isinstance(setup_2, SetupInfoTransfer):
                if key == "name":
                    self.setup_names = [setup_1.name, setup_2.name]
                    continue
                if isinstance(tmp1, list):
                    assert isinstance(tmp2, list)
                    tmp_out = tmp1 + tmp2

                setattr(self, key, tmp_out)

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

                setattr(self, key, tmp_out)

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
        """Normalize a moves list so each entry is a ``(move, weight)`` tuple."""
        for i in range(len(moves)):
            if isinstance(moves[i], tuple):
                assert isinstance(moves[i][1], float)
            else:
                moves[i] = (moves[i], 1.0)
        return moves

    @property
    def rj_moves_input(self) -> list:
        """Reversible-jump moves as a list of ``(move, weight)`` tuples (or ``None``)."""
        if self.rj_moves == []:
            return None

        return self.unwrap_moves(self.rj_moves)

    @property
    def in_model_moves_input(self) -> list:
        """In-model moves as a list of ``(move, weight)`` tuples (or ``None``)."""
        if self.in_model_moves == []:
            return None

        return self.unwrap_moves(self.in_model_moves)


# TODO: this was not working?
# @dataclass
class SetupInfoTransfer:
    """Container for a single source's MCMC moves and identifying name.

    Args:
        name: Identifier for this setup (e.g. ``"gb"``).
        in_model_moves: List of in-model moves (each either a move object or
            a ``(move, weight)`` tuple).
        rj_moves: List of reversible-jump moves in the same form.
    """
    # name = None
    # in_model_moves = []
    # rj_moves = []

    def __init__(
        self,
        name=None,
        in_model_moves=[],
        rj_moves=[],
    ):
        self.name = name
        self.in_model_moves = in_model_moves
        self.rj_moves = rj_moves

    def __add__(self, setup_2):
        return AllSetupInfoTransfer(self, setup_2)
