"""Stock Erebor transform atoms.

Named, picklable transform functions and the per-source
:class:`~eryn.utils.TransformContainer` builders (sampling basis <->
waveform basis) shared by every Erebor-family variant.
"""

from __future__ import annotations

import numpy as np

# frame helpers kept for settings files; the stock MBH basis is direct ICRS
from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA, mT_q
from eryn.utils import TransformContainer

from lisatools.utils.constants import PC_SI

def f_ms_to_s(x):
    """Convert frequencies from mHz to Hz (named for pickling support)."""
    return x * 1e-3


def f_s_to_ms(x):
    """Convert frequencies from Hz to mHz (inverse of :func:`f_ms_to_s`)."""
    return x * 1e3


def mchirp_to_fdot_pair(f0_Hz, Mc):
    """``(f0[Hz], Mc[Msol]) -> (f0[Hz], fdot[Hz/s])`` via the monochromatic-GB relation.

    Named at module scope so the containing :class:`TransformContainer` stays
    picklable. Consumers pair this with the single-param ``f0 -> f_ms_to_s``
    transform so the sampling basis carries ``f0`` in mHz linearly and Mc in
    solar masses, while the physical basis receives ``f0`` in Hz and fdot in
    Hz/s (with fddot = 0 filled elsewhere).
    """
    from gbgpu.utils.utility import get_fdot

    fdot = get_fdot(f=f0_Hz, Mc=Mc)
    return f0_Hz, fdot


def fdot_to_mchirp_pair(f0_Hz, fdot):
    """Inverse of :func:`mchirp_to_fdot_pair`: ``(f0[Hz], fdot) -> (f0[Hz], Mc[Msol])``."""
    from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

    Mc = get_chirp_mass_from_f_fdot(f0_Hz, fdot)
    return f0_Hz, Mc


def ten_to_the_x(x):
    """Return ``10 ** x`` (named 1-arg transform for pickling support).

    Used as a per-parameter :class:`~eryn.utils.TransformContainer` transform for
    PSD / foreground amplitudes sampled in ``log10`` space.
    """
    return 10.0 ** x


def make_gb_transform_container():
    """Build the stock GB :class:`TransformContainer` (forward + inverse).

    Sampling basis: ``A (ln), f0 (mHz), fdot, phi0, cos_iota, psi, lam,
    sin_beta``; output basis adds the filled ``fddot``. The inverse
    transforms enable ``both_inverse_transforms`` (full basis -> sampling
    basis).
    """
    input_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

    gb_transform_fn_in = {
        "A": np.exp,
        "f0": f_ms_to_s,
        "cos_iota": np.arccos,
        "sin_beta": np.arcsin,
    }

    gb_inverse_transform_fn_in = {
        "A": np.log,
        "f0": f_s_to_ms,
        "cos_iota": np.cos,
        "sin_beta": np.sin,
    }

    output_basis = [
        "A",
        "f0",
        "fdot",
        "fddot",
        "phi0",
        "cos_iota",
        "psi",
        "lam",
        "sin_beta",
    ]
    gb_fill_dict = {"fddot": 0.0}

    # gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=gb_transform_fn_in,
        fill_dict=gb_fill_dict,
        inverse_parameter_transforms=gb_inverse_transform_fn_in,
    )


def mbh_dist_trans(x):
    """Transform an MBH ``ln total mass`` parameter (named for pickling support)."""
    return x * PC_SI * 1e9  # Gpc


def gpc_to_mpc(x):
    """
    Transform from Gpc to Mpc, for distance prior.
    """
    return x * 1e3

def mT_Q(M, Q):
    """
    Transform from total mass and mass ratio m1/m2 to m1 and m2.
    """
    m2 = M / (1 + Q)
    m1 = Q * m2
    assert np.all(m1 >= m2), "m1 should be the larger mass"
    return m1, m2

def mpc_to_gpc(x):
    """
    Transform from Mpc to Gpc (inverse of :func:`gpc_to_mpc`).
    """
    return x * 1e-3


def m1_m2_to_mT_q(m1, m2):
    """Convert m1, m2 to total mass and mass ratio (inverse of ``mT_q``)."""
    return (m1 + m2, m2 / m1)


def m1_m2_to_mT_Q(m1, m2):
    """Convert m1, m2 to total mass and ``Q = m1/m2 >= 1`` (inverse of :func:`mT_Q`)."""
    return (m1 + m2, m1 / m2)


def make_mbh_transform_container():
    """Build the stock MBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (ICRS — the 2026-06 run frame, matching
    :func:`lisatools.globalfit.recipe.mbh_catalogue_to_sampling_basis`):
    ``logM, Q (= m1/m2 >= 1), s1z, s2z, dist (Gpc), phi_ref, cos_iota,
    psi (ICRS), alpha (RA), sin_delta, t_plunge (SSB)``. The output basis
    holds ``m1, m2 (via mT_Q), dist (Mpc), iota, delta`` with sky /
    polarization passed through unchanged in ICRS — no frame
    conversion; the orbits must be loaded with ``frame='icrs'`` so the
    response consumes ``(alpha, delta)`` directly. The inverse
    transforms enable ``both_inverse_transforms`` (full basis ->
    sampling basis).
    """
    input_basis = [
        "logM",
        "Q",
        "s1z",
        "s2z",
        "dist",
        "phi_ref",
        "cos_iota",
        "psi",
        "alpha",
        "sin_delta",
        "t_plunge",
    ]

    output_basis = list(input_basis)

    mbh_transform_fn_in = {
        "logM": np.exp,
        "dist": gpc_to_mpc,
        "cos_iota": np.arccos,
        "sin_delta": np.arcsin,
        ("logM", "Q"): mT_Q,
    }

    # keyed identically to mbh_transform_fn_in; same insertion order so
    # the multi-parameter inverses unwind in reverse of the forward order
    mbh_inverse_transform_fn_in = {
        "logM": np.log,
        "dist": mpc_to_gpc,
        "cos_iota": np.cos,
        "sin_delta": np.sin,
        ("logM", "Q"): m1_m2_to_mT_Q,
    }

    # for transforms
    # mbh_fill_dict = {"f_ref": 0.0}
    mbh_fill_dict = {}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=mbh_transform_fn_in,
        fill_dict=mbh_fill_dict,
        inverse_parameter_transforms=mbh_inverse_transform_fn_in,
    )


def make_emri_transform_container(fill_values):
    """Build the stock EMRI :class:`TransformContainer` (forward + inverse).

    Sampling basis (12 params): ``logm1, m2, a, p0, e0, dist, qS (cos),
    phiS, qK (cos), phiK, Phi_phi0, Phi_r0``; the output basis adds the
    filled ``xI0`` and ``Phi_theta0``. The inverse transforms enable
    ``both_inverse_transforms`` (full basis -> sampling basis).

    Args:
        fill_values: Length-2 sequence with the fill values for ``xI0``
            (inclination) and ``Phi_theta0``.
    """
    input_basis = [
        "logm1",
        "m2",
        "a",
        "p0",
        "e0",
        "dist",
        "cosqS",
        "phiS",
        "cosqK",
        "phiK",
        "Phi_phi0",
        "Phi_r0",
    ]

    output_basis = [
        "m1",
        "m2",
        "a",
        "p0",
        "e0",
        "xI0",
        "dist",
        "qS",
        "phiS",
        "qK",
        "phiK",
        "Phi_phi0",
        "Phi_theta0",
        "Phi_r0",
    ]

    # for transforms

    emri_fill_dict = {
        "xI0": fill_values[0],  # inclination
        "Phi_theta0": fill_values[1],  # Phi_theta
    }

    emri_transform_fn_in = {
        output_basis[0]: np.exp,  # M
        output_basis[7]: np.arccos,  # qS
        output_basis[9]: np.arccos,  # qK
    }

    emri_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # M
        output_basis[7]: np.cos,  # qS
        output_basis[9]: np.cos,  # qK
    }
    key_map = {
        "logm1": "m1",
        "cosqK": "qK",
        "cosqS": "qS",
    }

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=emri_transform_fn_in,
        fill_dict=emri_fill_dict,
        inverse_parameter_transforms=emri_inverse_transform_fn_in,
        key_map=key_map
    )


def cosqS_to_beta(x):
    """Transform ``cosqS`` to ecliptic latitude ``beta = pi/2 - arccos(cosqS)``."""
    return np.pi / 2.0 - np.arccos(x)


def beta_to_cosqS(x):
    """Transform ecliptic latitude ``beta`` to ``cosqS = sin(beta)`` (inverse of :func:`cosqS_to_beta`)."""
    return np.sin(x)


def make_sobbh_transform_container():
    """Build the stock SOBBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (11 params): ``logm1, logm2, s1, s2, dist, cosinc,
    f_low, phiS, cosqS, psi, phi0``; output basis: ``m1, m2, s1, s2,
    dist, inc, f_low, lam, beta, psi, phi0``. The inverse transforms
    enable ``both_inverse_transforms`` (full basis -> sampling basis).
    """
    input_basis = [
        "logm1",
        "logm2",
        "s1",
        "s2",
        "dist",
        "cosinc",
        "f_low",
        "phiS",
        "cosqS",
        "psi",
        "phi0",
    ]

    output_basis = [
        "m1",
        "m2",
        "s1",
        "s2",
        "dist",
        "inc",
        "f_low",
        "lam",
        "beta",
        "psi",
        "phi0",
    ]

    sobbh_transform_fn_in = {
        output_basis[0]: np.exp,  # m1
        output_basis[1]: np.exp,  # m2
        output_basis[5]: np.arccos,  # inc <- cosinc
        output_basis[8]: cosqS_to_beta,  # beta <- cosqS
    }

    sobbh_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # logm1 <- m1
        output_basis[1]: np.log,  # logm2 <- m2
        output_basis[5]: np.cos,  # cosinc <- inc
        output_basis[8]: beta_to_cosqS,  # cosqS <- beta
    }

    # Every input-basis name not already present in output_basis
    # needs a key_map entry pointing to its renamed counterpart.
    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=sobbh_transform_fn_in,
        key_map={
            "logm1": "m1",
            "logm2": "m2",
            "cosinc": "inc",
            "cosqS": "beta",
            "phiS": "lam",
        },
        inverse_parameter_transforms=sobbh_inverse_transform_fn_in,
    )
