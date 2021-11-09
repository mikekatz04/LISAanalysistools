import numpy as np
from astropy import units
from scipy import constants as ct

from .constants import *


def mT_q(mT, q):
    return (mT / (1 + q), mT * q / (1 + q))


def transfer_tref(tRef_sampling_frame_orig, tRef_sampling_frame_keep):
    return (tRef_sampling_frame_orig, tRef_sampling_frame_orig)


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


def mbh_sky_mode_transform(
    coords, ind_map=None, kind="both", inplace=False, cos_i=False
):

    if ind_map is None:
        ind_map = dict(inc=7, lam=8, beta=9, psi=10)

    elif isinstance(ind_map, dict) is False:
        raise ValueError("If providing the ind_map kwarg, it must be a dict.")

    if kind not in ["both", "lat", "long"]:
        raise ValueError(
            "The kwarg 'kind' must be lat for latitudinal transformation, long for longitudinal transformation, or both for both."
        )

    elif kind == "both":
        factor = 8

    elif kind == "long":
        factor = 4

    elif kind == "lat":
        factor = 2

    if inplace:
        if (coords.shape[0] % factor) != 0:
            raise ValueError(
                "If performing an inplace transformation, the coords provided must have a first dimension size divisible by {} for a '{}' transformation.".format(
                    factor, kind
                )
            )

    else:
        coords = np.tile(coords, (factor, 1))

    if kind == "both" or kind == "lat":

        # inclination
        if cos_i:
            coords[1::2, ind_map["inc"]] *= -1

        else:
            coords[1::2, ind_map["inc"]] = np.pi - coords[1::2, ind_map["inc"]]

        # beta
        coords[1::2, ind_map["beta"]] *= -1

        # psi
        coords[1::2, ind_map["psi"]] = np.pi - coords[1::2, ind_map["psi"]]

    if kind == "long":
        for i in range(1, 4):
            coords[i::4, ind_map["lam"]] = (
                coords[i::4, ind_map["lam"]] + np.pi / 2 * i
            ) % (2 * np.pi)

            coords[i::4, ind_map["psi"]] = (
                coords[i::4, ind_map["psi"]] + np.pi / 2 * i
            ) % (np.pi)
    if kind == "both":
        num = coords.shape[0]
        for i in range(1, 4):
            for j in range(2):
                coords[2 * i + j :: 8, ind_map["lam"]] = (
                    coords[2 * i + j :: 8, ind_map["lam"]] + np.pi / 2 * i
                ) % (2 * np.pi)

                coords[2 * i + j :: 8, ind_map["psi"]] = (
                    coords[2 * i + j :: 8, ind_map["psi"]] + np.pi / 2 * i
                ) % (np.pi)

    return coords
