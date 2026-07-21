"""Stock Erebor VGB (verification galactic binary) branch.

Verification binaries are KNOWN sources: their frequency and sky location
are fixed per leaf, so the branch samples only the 5 remaining parameters
``[lnA, fdot, phi0, cos_iota, psi]``. The branch is fixed-dimensional
(``nleaves_min == nleaves_max``, NO RJ) and leaf ``i`` is the same physical
source at every walker and temperature.

The fixed per-leaf ``f0``/``alpha``/``sin_delta`` live in the transform
container's per-leaf ``fill_dict`` (Eryn per-leaf fills, selected by
``leaf_inds`` at transform time) — built through
:func:`~.transforms.make_gb_transform_container`, the single source of the
GB basis conventions (phi0 sign included). Values are stored in SAMPLING
units (f0 in mHz, ``sin_delta``); the container's registered transforms
convert the filled columns to physical (Hz, delta) exactly like sampled
ones.

The move is :class:`~lisatools.globalfit.moves.VGBSpecialStretchMove`
(plain same-leaf stretch; no group-stretch friends, no info-matrix
proposal, no phase maximization) built by
:func:`lisatools.globalfit.recipe.build_vgb_moves`.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from ...engine import GeneralSetup, Settings
from ...recipe import MOJITO_REFERENCE_TIME, gb_catalogue_to_sampling_basis
from ..base import env_default
from .common import tdi_generation_info
from .gb import GBSettings, GBSetup
from .transforms import make_gb_transform_container

logger = logging.getLogger(__name__)

#: Sampled VGB parameter names (subset, in order, of the full GB sampling
#: basis emitted by :func:`make_gb_transform_container`).
VGB_SAMPLED_BASIS = ["A", "fdot", "phi0", "cos_iota", "psi"]

#: Fixed per-leaf parameter names (per-leaf fill keys, besides ``fddot``).
VGB_FIXED_BASIS = ["f0", "alpha", "sin_delta"]


@dataclasses.dataclass
class VGBSettings(GBSettings):
    """Settings for the VGB branch: 5 sampled params, fixed f0/sky per leaf.

    Inherits the GB knobs (band structure, chunked-het kernel sizes flow in
    through the variant); the sampling-facing fields below override the GB
    defaults. ``fixed_params`` / ``injection`` / ``nleaves_max`` are resolved
    from the mojito VGB catalogue in :func:`prepare_vgb_branch`.
    """

    ndim: int = 5
    # resolved to the catalogue source count at prepare time
    nleaves_min: typing.Optional[int] = None
    nleaves_max: typing.Optional[int] = None
    # fdot is sampled directly (no chirp-mass basis for known binaries)
    use_chirp_mass: bool = False
    use_astrophysical_f0_mc_prior: bool = False
    # ONE red-blue stretch sweep per iteration. VGBSpecialStretchMove.in_model_proposal
    # does a Goodman-Weare red-blue sweep against the BLOCK-START ensemble
    # (each mover's complement is the opposite walker-parity of its
    # (temp, leaf)). One sweep against the block-start complement is exact;
    # the gbspecialstretch repeat block, however, reuses that SAME block-start
    # complement for every repeat, so with >1 repeat the ensemble drifts
    # ahead of the stale complement (a pure-stretch ratchet). Keep this at 1
    # until the base repeat block refreshes the complement per repeat.
    num_repeat_proposals: int = dataclasses.field(
        default_factory=env_default("VGB_NUM_REPEAT_PROPOSALS", 1, int)
    )
    # chunked-het kernel sizes (WDM likelihood): shared CHUNKED_* env knobs,
    # same validated defaults as the GB branch.
    nt_sub: int = dataclasses.field(default_factory=env_default("CHUNKED_NT_SUB", 256, int))
    n_pad: int = dataclasses.field(default_factory=env_default("CHUNKED_N_PAD", 32, int))
    n_sparse: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_SPARSE", 256, int)
    )
    n_cp_sig: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_SIG", 48, int)
    )
    n_cp_orbit: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_ORBIT", 32, int)
    )
    # (nleaves, 3) per-leaf fixed [f0 (mHz), alpha, sin_delta] in SAMPLING
    # units, ordered like VGB_FIXED_BASIS; feeds the per-leaf fill list.
    fixed_params: typing.Optional[typing.Any] = None
    # (nleaves, 5) sampling-basis truth rows (seeds the fixed-leaf start).
    injection: typing.Optional[typing.Any] = None


class VGBSetup(GBSetup):
    """:class:`Setup` for verification galactic binaries.

    Reuses the GB band structure / waveform kwargs / state-backend init;
    only the sampling info differs: a 5-name input basis with the fixed
    per-leaf parameters as Eryn per-leaf fills, 5D priors/periodic keyed
    ``"vgb"``.
    """

    def init_band_structure(self):
        # GBSetup's band init derives fdot_lims from the top band edge; the
        # VGB fdot prior is catalogue-derived in prepare_vgb_branch, so an
        # explicitly-set range survives.
        _fdot_lims_in = list(self.fdot_lims) if self.fdot_lims else None
        super().init_band_structure()
        if _fdot_lims_in is not None:
            self.fdot_lims = _fdot_lims_in

    def init_sampling_info(self):
        if self.fixed_params is None:
            raise ValueError(
                "VGBSetup needs per-leaf fixed_params (nleaves, 3) "
                "[f0 (mHz), alpha, sin_delta]; run prepare_vgb_branch first "
                "(mojito VGB catalogue required)."
            )
        fixed = np.asarray(self.fixed_params, dtype=float)
        n_leaves = fixed.shape[0]
        assert fixed.shape == (n_leaves, len(VGB_FIXED_BASIS))
        if self.nleaves_max is not None:
            assert int(self.nleaves_max) == n_leaves, (
                f"fixed_params rows ({n_leaves}) != nleaves_max "
                f"({self.nleaves_max})"
            )

        if self.transform is None:
            # One per-leaf fill dict per source, keys shared across leaves
            # (asserted by Eryn). Values in SAMPLING units — the factory's
            # registered transforms convert the filled columns.
            fill_list = [
                {
                    "fddot": 0.0,
                    **{
                        name: fixed[leaf, j]
                        for j, name in enumerate(VGB_FIXED_BASIS)
                    },
                }
                for leaf in range(n_leaves)
            ]
            self.transform = make_gb_transform_container(
                use_chirp_mass=False,
                input_basis=list(VGB_SAMPLED_BASIS),
                fill_dict=fill_list,
            )

        if self.periodic is None:
            self.periodic = {"vgb": {"phi0": 2 * np.pi, "psi": np.pi}}

        if self.priors is None:
            from eryn.prior import ProbDistContainer, uniform_dist

            # Global uniforms over the 5 sampled params. fdot_lims must
            # cover every catalogue fdot (checked below with margin).
            cat_fdot = None
            if self.injection is not None:
                inj = np.asarray(self.injection, dtype=float)
                cat_fdot = inj[:, VGB_SAMPLED_BASIS.index("fdot")]
                if not (
                    (cat_fdot > self.fdot_lims[0]).all()
                    and (cat_fdot < self.fdot_lims[1]).all()
                ):
                    raise ValueError(
                        "VGB fdot prior "
                        f"[{self.fdot_lims[0]:.3e}, {self.fdot_lims[1]:.3e}] "
                        "does not cover the catalogue fdot range "
                        f"[{cat_fdot.min():.3e}, {cat_fdot.max():.3e}]; "
                        "widen VGBSettings.fdot_lims."
                    )
            priors_vgb = {
                0: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                1: uniform_dist(self.fdot_lims[0], self.fdot_lims[1]),
                2: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                # cos is DECREASING on [0, pi]: sort defensively.
                3: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                4: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
            }
            self.priors = {"vgb": ProbDistContainer(priors_vgb)}

        # Shared tail (betas / waveform kwargs / group_proposal_kwargs):
        # transform/priors/periodic are set above, so GBSetup's blocks for
        # those are skipped and only the shared pieces run.
        super().init_sampling_info()


def prepare_vgb_branch(vgb: VGBSettings, general_setup: GeneralSetup, *, data_mode: str):
    """Resolve the VGB branch from the mojito VGB catalogue.

    Builds the sampling-basis rows through the single GB factory convention
    (``gb_catalogue_to_sampling_basis`` -> container inverse) and splits
    them BY NAME into the 5 sampled columns (``injection``) and the 3 fixed
    per-leaf columns (``fixed_params``); sets the fixed-dimensional leaf
    count and the band structure bounds (one guard band each side — the
    band machinery never proposes in the first/last band).
    """
    if data_mode != "mojito":
        raise ValueError(
            "The VGB branch needs the mojito VGB catalogue "
            f"(data_mode='mojito'); got data_mode={data_mode!r}."
        )
    catalogue = (getattr(general_setup, "catalogue", None) or {}).get("VGB", {})
    if not catalogue:
        raise ValueError(
            "No VGB catalogue on the general setup: include 'VGB' in the "
            "run's source types so the L1 loader populates catalogue['VGB']."
        )

    # (nleaves, 8) sampling-basis rows; leaf order = sorted catalogue keys
    # then file order within a key (the mojito loader stores (V)GB catalogue
    # entries as whole arrays under one id) — deterministic across restarts,
    # so the per-leaf fills rebuild identically at every start.
    rows = np.array(
        [
            gb_catalogue_to_sampling_basis(catalogue[k])
            for k in sorted(catalogue.keys())
        ]
    )
    if rows.ndim == 3:
        rows = rows.reshape(-1, rows.shape[-1])

    # Column split derived from the factory's basis names, never literals.
    full_basis = list(make_gb_transform_container(use_chirp_mass=False).input_basis)
    sampled_idx = [full_basis.index(name) for name in VGB_SAMPLED_BASIS]
    fixed_idx = [full_basis.index(name) for name in VGB_FIXED_BASIS]

    vgb.injection = rows[:, sampled_idx]
    vgb.fixed_params = rows[:, fixed_idx]
    n = rows.shape[0]
    vgb.nleaves_min = vgb.nleaves_max = n

    if vgb.t0 in (None, 0.0):
        # phase/frequency reference epoch = the mojito catalogue epoch
        vgb.t0 = MOJITO_REFERENCE_TIME

    # Band structure bounds from the fixed f0 table, one guard band per
    # side: run_proposal never proposes in the first/last band, and
    # GBSetup.init_band_structure's f0_lims are the interior edges.
    f0_hz = rows[:, full_basis.index("f0")] * 1e-3
    from lisatools.domains import WDMSettings

    domain_settings = general_setup.domain_settings
    if isinstance(domain_settings, WDMSettings):
        guard = 2.0 * float(domain_settings.layer_df)
    else:
        # FD bands are ~(2 N + buffer) * df wide; a conservative guard.
        from gbgpu.utils.utility import get_N

        df = 1.0 / general_setup.Tobs
        n_max = get_N(
            1e-30, float(f0_hz.max()), general_setup.Tobs, oversample=vgb.oversample
        ).item()
        guard = 3.0 * (2 * n_max + vgb.extra_buffer) * df
    vgb.start_freq = max(float(f0_hz.min()) - guard, general_setup.min_freq)
    vgb.end_freq = min(float(f0_hz.max()) + guard, general_setup.max_freq)
    if not (
        vgb.start_freq < float(f0_hz.min()) and vgb.end_freq > float(f0_hz.max())
    ):
        raise ValueError(
            "VGB band guard does not fit inside the data band "
            f"[{general_setup.min_freq:.6e}, {general_setup.max_freq:.6e}] Hz: "
            f"VGB f0 span [{f0_hz.min():.6e}, {f0_hz.max():.6e}] Hz needs a "
            f"{guard:.3e} Hz guard each side. Widen the general band."
        )

    # Prior ranges left empty resolve to wide defaults covering the
    # catalogue (checked hard in init_sampling_info for fdot).
    if not vgb.A_lims:
        amps = np.exp(rows[:, full_basis.index("A")])
        vgb.A_lims = [float(amps.min()) / 100.0, float(amps.max()) * 100.0]
    if not vgb.fdot_lims:
        cat_fdot = rows[:, full_basis.index("fdot")]
        span = float(np.abs(cat_fdot).max())
        vgb.fdot_lims = [-10.0 * span, 10.0 * span]
    if not vgb.phi0_lims:
        vgb.phi0_lims = [0.0, 2 * np.pi]
    if not vgb.iota_lims:
        _eps = 1e-5
        vgb.iota_lims = [0.0 + _eps, np.pi - _eps]
    if not vgb.psi_lims:
        vgb.psi_lims = [0.0, np.pi]

    vgb.tdi_setup = general_setup.tdi_chan
    vgb.use_tdi2 = tdi_generation_info(general_setup.tdi_chan)[0] == 2
    vgb.initialize_kwargs = dict(force_backend=general_setup.force_backend)
    if vgb.betas is None:
        betas = 1.0 / 1.2 ** np.arange(general_setup.ntemps)
        betas[-1] = 1e-4
        vgb.betas = betas
    vgb.gb_wdm_comp = None

    logger.info(
        "VGB branch: %d catalogue sources, f0 in [%.6e, %.6e] Hz, band "
        "[%.6e, %.6e] Hz.",
        n, f0_hz.min(), f0_hz.max(), vgb.start_freq, vgb.end_freq,
    )
    return vgb
