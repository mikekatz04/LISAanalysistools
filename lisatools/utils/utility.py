import numpy as np
from scipy.stats import uniform
from lisatools.sensitivity import get_sensitivity


def generate_noise_fd(freqs, df, **sensitivity_kwargs):
    norm = 0.5 * (1.0 / df) ** 0.5
    psd = get_sensitivity(freqs, **sensitivity_kwargs)
    noise_to_add = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs))
        + 1j * np.random.normal(0, norm, len(freqs))
    )

    return noise_to_add


def uniform_dist(min, max):
    if min > max:
        temp = min
        min = max
        max = temp

    mean = (max + min) / 2.0
    sig = max - min
    dist = uniform(min, sig)
    return dist


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
