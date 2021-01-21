import numpy as np
from astropy import units
from scipy import constants as ct

from .constants import *


def modpi(phase):
    # from sylvain
    return phase - np.floor(phase / np.pi) * np.pi


def mod2pi(phase):
    # from sylvain
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


# Compute Solar System Barycenter time tSSB from retarded time at the center of the LISA constellation tL */
# NOTE: depends on the sky position given in SSB parameters */
def tSSBfromLframe(tL, lambdaSSB, betaSSB, t0):
    ConstPhi0 = ConstOmega * t0
    OrbitR = 1.4959787066e11  # AU_SI
    C_SI = 299792458.0
    phase = ConstOmega * tL + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return (
        tL
        + RoC * np.cos(betaSSB) * np.cos(phase)
        - 1.0 / 2 * ConstOmega * pow(RoC * np.cos(betaSSB), 2) * np.sin(2.0 * phase)
    )


# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0):
    ConstPhi0 = ConstOmega * t0
    OrbitR = 1.4959787066e11  # AU_SI
    C_SI = 299792458.0
    phase = ConstOmega * tSSB + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return tSSB - RoC * np.cos(betaSSB) * np.cos(phase)


class LISA_to_SSB:
    def __init__(self, t0):
        self.t0 = t0 * YRSID_SI

    def __call__(self, tL, lambdaL, betaL, psiL):
        """
            # from Sylvan
            int ConvertLframeParamsToSSBframe(
              double* tSSB,
              double* lambdaSSB,
              double* betaSSB,
              double* psiSSB,
              const tL,
              const lambdaL,
              const betaL,
              const psiL,
              const LISAconstellation *variant)
            {
        """
        ConstPhi0 = ConstOmega * (self.t0)
        coszeta = np.cos(np.pi / 3.0)
        sinzeta = np.sin(np.pi / 3.0)
        coslambdaL = np.cos(lambdaL)
        sinlambdaL = np.sin(lambdaL)
        cosbetaL = np.cos(betaL)
        sinbetaL = np.sin(betaL)
        cospsiL = np.cos(psiL)
        sinpsiL = np.sin(psiL)
        lambdaSSB_approx = 0.0
        betaSSB_approx = 0.0
        # Initially, approximate alpha using tL instead of tSSB - then iterate */
        tSSB_approx = tL
        for k in range(3):
            alpha = ConstOmega * tSSB_approx + ConstPhi0
            cosalpha = np.cos(alpha)
            sinalpha = np.sin(alpha)
            lambdaSSB_approx = np.arctan2(
                cosalpha * cosalpha * cosbetaL * sinlambdaL
                - sinalpha * sinbetaL * sinzeta
                + cosbetaL * coszeta * sinalpha * sinalpha * sinlambdaL
                - cosalpha * cosbetaL * coslambdaL * sinalpha
                + cosalpha * cosbetaL * coszeta * coslambdaL * sinalpha,
                cosbetaL * coslambdaL * sinalpha * sinalpha
                - cosalpha * sinbetaL * sinzeta
                + cosalpha * cosalpha * cosbetaL * coszeta * coslambdaL
                - cosalpha * cosbetaL * sinalpha * sinlambdaL
                + cosalpha * cosbetaL * coszeta * sinalpha * sinlambdaL,
            )
            betaSSB_approx = np.arcsin(
                coszeta * sinbetaL
                + cosalpha * cosbetaL * coslambdaL * sinzeta
                + cosbetaL * sinalpha * sinzeta * sinlambdaL
            )
            tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, self.t0)

        lambdaSSB_approx = lambdaSSB_approx % (2 * np.pi)
        #  /* Polarization */
        psiSSB = modpi(
            psiL
            + np.arctan2(
                cosalpha * sinzeta * sinlambdaL - coslambdaL * sinalpha * sinzeta,
                cosbetaL * coszeta
                - cosalpha * coslambdaL * sinbetaL * sinzeta
                - sinalpha * sinbetaL * sinzeta * sinlambdaL,
            )
        )

        return (tSSB_approx, lambdaSSB_approx, betaSSB_approx, psiSSB)


# Convert SSB-frame params to L-frame params  from sylvain marsat / john baker
# NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
class SSB_to_LISA:
    def __init__(self, t0):
        self.t0 = t0 * YRSID_SI

    def __call__(self, tSSB, lambdaSSB, betaSSB, psiSSB):

        ConstPhi0 = ConstOmega * (self.t0)
        alpha = 0.0
        cosalpha = 0
        sinalpha = 0.0
        coslambda = 0
        sinlambda = 0.0
        cosbeta = 0.0
        sinbeta = 0.0
        cospsi = 0.0
        sinpsi = 0.0
        coszeta = np.cos(np.pi / 3.0)
        sinzeta = np.sin(np.pi / 3.0)
        coslambda = np.cos(lambdaSSB)
        sinlambda = np.sin(lambdaSSB)
        cosbeta = np.cos(betaSSB)
        sinbeta = np.sin(betaSSB)
        cospsi = np.cos(psiSSB)
        sinpsi = np.sin(psiSSB)
        alpha = ConstOmega * tSSB + ConstPhi0
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, self.t0)
        lambdaL = np.arctan2(
            cosalpha * cosalpha * cosbeta * sinlambda
            + sinalpha * sinbeta * sinzeta
            + cosbeta * coszeta * sinalpha * sinalpha * sinlambda
            - cosalpha * cosbeta * coslambda * sinalpha
            + cosalpha * cosbeta * coszeta * coslambda * sinalpha,
            cosalpha * sinbeta * sinzeta
            + cosbeta * coslambda * sinalpha * sinalpha
            + cosalpha * cosalpha * cosbeta * coszeta * coslambda
            - cosalpha * cosbeta * sinalpha * sinlambda
            + cosalpha * cosbeta * coszeta * sinalpha * sinlambda,
        )
        betaL = np.arcsin(
            coszeta * sinbeta
            - cosalpha * cosbeta * coslambda * sinzeta
            - cosbeta * sinalpha * sinzeta * sinlambda
        )
        psiL = modpi(
            psiSSB
            + np.arctan2(
                coslambda * sinalpha * sinzeta - cosalpha * sinzeta * sinlambda,
                cosbeta * coszeta
                + cosalpha * coslambda * sinbeta * sinzeta
                + sinalpha * sinbeta * sinzeta * sinlambda,
            )
        )

        return (tL, lambdaL, betaL, psiL)


class TransformContainer:
    def __init__(self, parameter_transforms):
        self.base_transforms = {}

        inplace_transforms_temp = parameter_transforms.get("inplace", {})
        base_transforms_temp = parameter_transforms.get("base", {})

        self.base_transforms = {"single_param": {}, "mult_param": {}}
        self.inplace_transforms = {"single_param": {}, "mult_param": {}}

        for trans, trans_temp in zip(
            [self.base_transforms, self.inplace_transforms],
            [base_transforms_temp, inplace_transforms_temp],
        ):
            for key, item in trans_temp.items():
                if isinstance(key, int):
                    trans["single_param"][key] = item
                elif isinstance(key, tuple):
                    trans["mult_param"][key] = item
                else:
                    raise ValueError(
                        "Parameter transform keys must be int or tuple of ints. {} is neither.".format(
                            key
                        )
                    )

    def transform_inplace_parameters(self, params, inds_map=None):
        # inplace transforms
        if inds_map is None:
            inds_map = np.arange(params.shape[1])
        try:
            inds_list = inds_map.tolist()
        except AttributeError:
            pass

        try:
            inds_map = {ind1: inds_list.index(ind1) for ind1 in inds_map}
        except ValueError:
            raise ValueError(
                "inplace transformations can only be done on indices that are being tested."
            )

        params = params.T
        for ind_temp, trans_fn in self.inplace_transforms["single_param"].items():
            try:
                ind = inds_map[ind_temp]
            except KeyError:
                raise KeyError(
                    "Parameters from inplace tansform are not being sampled."
                )
            params[ind] = trans_fn(params[ind])

        for inds_temp, trans_fn in self.inplace_transforms["mult_param"].items():
            try:
                inds = [inds_map[ind_temp_i] for ind_temp_i in inds_temp]
            except KeyError:
                raise KeyError(
                    "Parameters from inplace tansform are not being sampled."
                )
            temp = trans_fn(*[params[i] for i in inds])
            for j, i in enumerate(inds):
                params[i] = temp[j]

    def transform_base_parameters(self, params):
        params_temp = params.copy().T
        # regular transforms to waveform domain
        for ind, trans_fn in self.base_transforms["single_param"].items():
            params_temp[ind] = trans_fn(params_temp[ind])

        for inds, trans_fn in self.base_transforms["mult_param"].items():
            temp = trans_fn(*[params_temp[i] for i in inds])
            for j, i in enumerate(inds):
                params_temp[i] = temp[j]

        return params_temp
