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

def negate(x):
    """Return ``-x`` (named 1-arg transform for pickling support; it is its
    own functional inverse -- e.g. the GB phi0 sign flip)."""
    return -x


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


class McFdotAstroRatioTriple:
    """``(f0[Hz], Mc[Msol], r) -> (f0, fdot_gr*(1+r), 0.0)`` (fdot_astro basis).

    ``r = fdot_astro / fdot_gr`` is the sampled dimensionless ratio: the
    physical ``fdot = fdot_gr(f0, Mc) * (1 + r)`` (astrophysical processes
    add linearly to the GW chirp; ``r < -1`` gives the interacting
    ``fdot < 0`` systems), and the fddot slot -- which carried the sampled
    ``r`` after ``fill_values`` -- is overwritten with exactly 0.

    Measure note (why the ratio basis): the total fdot is bounded to
    ``(1 +/- M) * fdot_gr`` by construction under an ``r ~ U[-M, M]``
    prior, so fdot_astro can never dominate unphysically. The induced
    physical prior at fixed ``(f0, Mc)`` is uniform in fdot with density
    ``1 / (2 M fdot_gr)`` -- along a measured-fdot ridge the Mc marginal
    therefore carries a ``1/fdot_gr(Mc) ~ Mc^(-5/3)`` tilt relative to the
    f0-Mc prior. Any proposal whose density is defined in PHYSICAL fdot
    space must multiply by the Jacobian ``fdot_gr`` when converted to this
    sampling basis; everything in the stock stack defines densities in the
    sampling basis directly, so no such factor appears today.

    Module-level class so the containing :class:`TransformContainer`
    pickles.
    """

    def __call__(self, f0_Hz, Mc, ratio):
        from gbgpu.utils.utility import get_fdot

        return f0_Hz, get_fdot(f=f0_Hz, Mc=Mc) * (1.0 + ratio), f0_Hz * 0.0


class McFdotAstroRatioTripleInverse:
    """Mirror-convention split ``(f0[Hz], fdot, fddot) -> (f0, Mc, r)``.

    The ``(Mc, r)`` split of a physical fdot is likelihood-degenerate (the
    waveform sees only the total), so the inverse picks the convention
    ``Mc = Mc_GW(f0, |fdot|)`` (exact GW inversion of the fdot MAGNITUDE,
    clipped into ``mc_lims``) with the remainder in the ratio: in-box this
    lands every source at exactly ``r = 0`` (fdot > 0) or ``r = -2``
    (fdot < 0) -- both well inside any sensible ``U[-M, M]`` prior box --
    and reproduces the physical fdot exactly for either sign (no NaNs).
    """

    _TINY_FDOT = 1e-30  # inverts to an Mc far below any box floor -> clip

    def __init__(self, mc_lims=(0.001, 1.0)):
        self.mc_lims = tuple(mc_lims)

    def __call__(self, f0_Hz, fdot, fddot):
        from gbgpu.utils.utility import get_chirp_mass_from_f_fdot, get_fdot

        with np.errstate(invalid="ignore"):
            Mc = get_chirp_mass_from_f_fdot(
                f0_Hz, np.maximum(np.abs(fdot), self._TINY_FDOT)
            )
        Mc = np.clip(Mc, *self.mc_lims)
        return f0_Hz, Mc, fdot / get_fdot(f=f0_Hz, Mc=Mc) - 1.0


def ten_to_the_x(x):
    """Return ``10 ** x`` (named 1-arg transform for pickling support).

    Used as a per-parameter :class:`~eryn.utils.TransformContainer` transform for
    PSD / foreground amplitudes sampled in ``log10`` space.
    """
    return 10.0 ** x


def make_gb_transform_container(
    *,
    use_chirp_mass: bool = False,
    use_fdot_astro: bool = False,
    input_basis: list | None = None,
    fill_dict: dict | list | None = None,
    mc_lims: tuple = (0.001, 1.0),
) -> TransformContainer:
    """THE stock (V)GB :class:`TransformContainer` factory (forward + inverse).

    Single source of the GB transform convention — the ICRS run frame and,
    in particular, the **phi0 sign**: sampling ``phi0`` is NEGATED to the
    physical basis ("match JaxGB convention"). The physical/waveform phase
    is ``phi0 = +TrueAnomaly`` (validated GB<->mojito convention); the
    sampling coordinate therefore stores ``-TrueAnomaly``. Do not encode
    this sign anywhere else — catalogue conversions must route through this
    container's ``both_inverse_transforms``.

    Full sampling basis: ``A (ln), f0 (mHz), Mc|fdot, phi0, cos_iota, psi,
    alpha, sin_delta`` (ICRS sky), plus a 9th ``fdot_astro_ratio`` column
    appended when ``use_fdot_astro``; the 9-name output basis adds
    ``fddot`` (filled 0, or transform-zeroed in the fdot_astro mode).
    Output columns keep the sampling-style names but hold the PHYSICAL
    values after the registered transforms run (``A``: amplitude, ``f0``:
    Hz, ``cos_iota``: iota, ``sin_delta``: delta).

    Args:
        use_chirp_mass: Slot 2 carries ``Mc`` (Msol) instead of ``fdot``;
            the ``(f0, Mc) -> (f0, fdot)`` pair transform + ``key_map``
            recover the physical fdot (the GBSetup chirp-mass convention).
        use_fdot_astro: Requires ``use_chirp_mass=True``. APPENDS a 9th
            sampled column ``fdot_astro_ratio`` (``r = fdot_astro /
            fdot_gr``, dimensionless) so columns 0-7 keep their meaning;
            the physical ``fdot = fdot_gr(f0, Mc) * (1 + r)`` via
            :class:`McFdotAstroRatioTriple` and ``fddot`` comes out exactly
            0 from the transform itself. ``key_map`` routes ``r`` to the
            fddot slot and ``fill_dict`` defaults to ``{}`` in this mode:
            an EMPTY dict keeps the sampled->output reorder scatter, while
            ``None`` would skip it, and nothing may fill the fddot slot the
            sampled ``r`` occupies.
        input_basis: Subset (in order) of the full sampling basis actually
            sampled. Default: all 8 (9 with ``use_fdot_astro``). The 5D VGB
            basis passes ``["A", "fdot", "phi0", "cos_iota", "psi"]`` with
            the fixed ``f0``/``alpha``/``sin_delta`` supplied through
            ``fill_dict``.
        fill_dict: Fill spec for the non-sampled output parameters. Default
            ``{"fddot": 0.0}`` (``{}`` with ``use_fdot_astro``). May be a
            LIST of per-leaf dicts (Eryn per-leaf fills; transform calls
            then need ``leaf_inds``) for fixed-source branches, e.g.
            ``{"fddot": 0.0, "f0": f0_i, "alpha": alpha_i, "sin_delta":
            sin_delta_i}`` per VGB leaf. Fill values are in SAMPLING units
            (f0 in mHz, ``sin_delta`` — not delta): the registered
            transforms run on the filled columns exactly like on sampled
            ones, so the physical basis comes out in Hz / delta.
        mc_lims: ``[Mc_min, Mc_max]`` box used by the fdot_astro inverse
            (mirror convention clips ``Mc_GW(f0, |fdot|)`` into it).
            Ignored unless ``use_fdot_astro``.
    """
    if use_fdot_astro and not use_chirp_mass:
        raise ValueError(
            "use_fdot_astro=True requires use_chirp_mass=True (the ratio "
            "multiplies fdot_gr(f0, Mc))."
        )
    full_sampling_basis = [
        "A", "f0", "Mc" if use_chirp_mass else "fdot", "phi0",
        "cos_iota", "psi", "alpha", "sin_delta",
    ]
    if use_fdot_astro:
        full_sampling_basis.append("fdot_astro_ratio")
    if input_basis is None:
        input_basis = full_sampling_basis
    else:
        unknown = [key for key in input_basis if key not in full_sampling_basis]
        if unknown:
            raise ValueError(
                f"input_basis entries {unknown} are not in the GB sampling "
                f"basis {full_sampling_basis}."
            )
        input_basis = list(input_basis)

    output_basis = [
        "A", "f0", "fdot", "fddot", "phi0",
        "cos_iota", "psi", "alpha", "sin_delta",
    ]

    gb_transform_fn_in = {
        "A": np.exp,
        "f0": f_ms_to_s,
        "phi0": negate,  # flip sign of phi0 to match JaxGB convention.
        "cos_iota": np.arccos,
        "sin_delta": np.arcsin,
    }
    gb_inverse_transform_fn_in = {
        "A": np.log,
        "f0": f_s_to_ms,
        "phi0": negate,  # its own inverse
        "cos_iota": np.cos,
        "sin_delta": np.sin,
    }
    gb_key_map = None
    if use_fdot_astro:
        # After fill_values, Mc lives at the fdot slot and the sampled
        # ratio r at the fddot slot (key_map below). The triple transform
        # then computes (f0[Hz], fdot_gr*(1+r), 0.0) in output-index space;
        # the single-param f0: f_ms_to_s runs first so it sees Hz.
        gb_transform_fn_in[("f0", "Mc", "fdot_astro_ratio")] = (
            McFdotAstroRatioTriple()
        )
        gb_inverse_transform_fn_in[("f0", "Mc", "fdot_astro_ratio")] = (
            McFdotAstroRatioTripleInverse(mc_lims)
        )
        gb_key_map = {"Mc": "fdot", "fdot_astro_ratio": "fddot"}
        if fill_dict is None:
            # {} KEEPS the reorder scatter (None would skip fill_values
            # entirely); the fddot slot carries the sampled r, so no fill
            # may write it -- the transform zeroes it after use.
            fill_dict = {}
    elif use_chirp_mass:
        # After fill_values, the Mc value lives at the fdot slot (key_map
        # "Mc"->"fdot"). The multi-parameter transform then computes
        # (f0[Hz], fdot) from (f0[Hz], Mc); the single-param f0: f_ms_to_s
        # runs first so both f0 slots see Hz.
        gb_transform_fn_in[("f0", "Mc")] = mchirp_to_fdot_pair
        gb_inverse_transform_fn_in[("f0", "Mc")] = fdot_to_mchirp_pair
        gb_key_map = {"Mc": "fdot"}

    if fill_dict is None:
        fill_dict = {"fddot": 0.0}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=gb_transform_fn_in,
        fill_dict=fill_dict,
        key_map=gb_key_map,
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
        fill_values: EITHER a length-2 sequence ``[xI0, Phi_theta0]`` filled
            identically for every leaf, OR an ``(nleaves, 2)`` array (or
            list of per-leaf pairs) — one fill per leaf (Eryn per-leaf
            ``fill_dict``: ``xI0`` is the intrinsic prograde/retrograde flag
            fixed per EMRI source, so it can differ per leaf; transform
            calls then require ``leaf_inds``).
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

    _fv = np.asarray(fill_values, dtype=float)
    if _fv.ndim == 2:
        # per-leaf fills (Eryn per-leaf fill_dict) — one dict per leaf
        emri_fill_dict = [
            {"xI0": row[0], "Phi_theta0": row[1]} for row in _fv
        ]
    else:
        emri_fill_dict = {
            "xI0": _fv[0],  # inclination (prograde/retrograde flag)
            "Phi_theta0": _fv[1],  # Phi_theta
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
