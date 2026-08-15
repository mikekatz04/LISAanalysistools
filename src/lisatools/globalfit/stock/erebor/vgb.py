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
import os
import typing

import numpy as np

from ...engine import GeneralSetup, Settings
from ...recipe import MOJITO_REFERENCE_TIME, gb_catalogue_to_sampling_basis
from ..base import env_default, env_is_set, env_resolve


def _vgb_sighet_inmodel_default() -> bool:
    """Per-branch override first (2026-08-12 smoke-2 finding): a set
    VGB_SIGHET_INMODEL wins; the shared GB_SIGHET_INMODEL (which flips gb
    and vgb together) is the fallback. Before this, VGB_SIGHET_INMODEL was
    SILENTLY IGNORED -- the exact rule-0 failure mode -- so the chunked-het
    vs sig-het VGB discriminator never actually ran. Module-level named
    function: the pre-build fit must pickle/deepcopy."""
    if env_is_set("VGB_SIGHET_INMODEL"):
        return env_resolve("VGB_SIGHET_INMODEL", True, bool)
    return env_resolve("GB_SIGHET_INMODEL", True, bool)
from .common import tdi_generation_info
from .gb import GBSettings, GBSetup
from .transforms import make_gb_transform_container

logger = logging.getLogger(__name__)

#: Sampled VGB parameter names (subset, in order, of the full GB sampling
#: basis emitted by :func:`make_gb_transform_container`).
VGB_SAMPLED_BASIS = ["A", "fdot", "phi0", "cos_iota", "psi"]

#: Fixed per-leaf parameter names (per-leaf fill keys, besides ``fddot``).
VGB_FIXED_BASIS = ["f0", "alpha", "sin_delta"]

#: Distance-basis variants (``VGBSettings.sample_distance``): slot 0 samples
#: the luminosity DISTANCE (kpc) with the amplitude DERIVED from the
#: per-leaf ``(f0, Mc)`` through the factory's astro package, and the
#: ``fdot_astro_ratio`` column carries the frequency-derivative freedom
#: (``fdot = fdot_gr(f0, Mc) * (1 + r)``) so mass-transfer systems stay
#: reachable. Mc joins the per-leaf fills (known binaries).
VGB_SAMPLED_BASIS_DIST = ["dist", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
VGB_FIXED_BASIS_DIST = ["f0", "alpha", "sin_delta", "Mc"]

#: OPT-IN chirp-mass distance basis (``VGBSettings.chirp_mass_basis``;
#: 2026-08-14 ruling, planned for the 6-month run ONLY): Mc moves from the
#: per-leaf fills to the SAMPLED side. The nonzero catalogue Mc truth gives
#: the multiplicative walker init real spread, which the pure-stretch move
#: needs on every dimension; the still-exactly-zero ratio truth gets the
#: additive init exception (see ``VGBSettings.ratio_init_width``).
VGB_SAMPLED_BASIS_CHIRP = ["dist", "phi0", "cos_iota", "psi", "Mc", "fdot_astro_ratio"]
VGB_FIXED_BASIS_CHIRP = ["f0", "alpha", "sin_delta"]


def vgb_sampled_basis(settings) -> list:
    """The active VGB sampled-parameter names (single source of the choice)."""
    if not getattr(settings, "sample_distance", False):
        return list(VGB_SAMPLED_BASIS)
    if getattr(settings, "chirp_mass_basis", False):
        return list(VGB_SAMPLED_BASIS_CHIRP)
    return list(VGB_SAMPLED_BASIS_DIST)


def vgb_fixed_basis(settings) -> list:
    """The active VGB per-leaf fixed-parameter names."""
    if not getattr(settings, "sample_distance", False):
        return list(VGB_FIXED_BASIS)
    if getattr(settings, "chirp_mass_basis", False):
        return list(VGB_FIXED_BASIS_CHIRP)
    return list(VGB_FIXED_BASIS_DIST)


def _vgb_ndim_default() -> int:
    """5 for the legacy bases, 6 under VGB_CHIRP_MASS_BASIS=1 (module-level
    named function: the pre-build fit must pickle/deepcopy)."""
    return (
        len(VGB_SAMPLED_BASIS_CHIRP)
        if env_resolve("VGB_CHIRP_MASS_BASIS", False, bool)
        else len(VGB_SAMPLED_BASIS_DIST)
    )


@dataclasses.dataclass
class VGBSettings(GBSettings):
    """Settings for the VGB branch: fixed f0/sky per leaf, the rest sampled.

    Inherits the GB knobs (band structure, chunked-het kernel sizes flow in
    through the variant); the sampling-facing fields below override the GB
    defaults. ``fixed_params`` / ``injection`` / ``nleaves_max`` are resolved
    from the mojito VGB catalogue in :func:`prepare_vgb_branch`.
    """

    # 5 for both legacy bases; 6 when chirp_mass_basis is on. The default
    # tracks the env knob so a pre-build settings object is self-consistent;
    # prepare_vgb_branch re-resolves it from the active basis either way.
    ndim: int = dataclasses.field(default_factory=_vgb_ndim_default)
    # VGB's own tempering ladder size (overrides the GB default)
    ntemps: int = dataclasses.field(default_factory=env_default("VGB_NTEMPS", 12, int))
    # resolved to the catalogue source count at prepare time
    nleaves_min: typing.Optional[int] = None
    nleaves_max: typing.Optional[int] = None
    # fdot is sampled directly (no chirp-mass basis for known binaries)
    use_chirp_mass: bool = False
    use_astrophysical_f0_mc_prior: bool = False
    # Distance sampling basis (default ON): slot 0 = luminosity distance
    # (kpc) with A DERIVED from the per-leaf (f0, Mc) catalogue values; the
    # fdot_astro_ratio column keeps fdot free (mass transfer reachable).
    # Named ``sample_distance`` because the inherited ``use_distance`` is a
    # read-only property of the GB astro-package switches. Prior boxes:
    # inherited ``dist_lims`` (kpc) and ``fdot_astro_ratio_max``
    # (U[-max, max] on r). VGB_SAMPLE_DISTANCE=0 reverts to the (lnA, fdot)
    # basis.
    sample_distance: bool = dataclasses.field(
        default_factory=env_default("VGB_SAMPLE_DISTANCE", True, bool)
    )
    # OPT-IN 6-dim chirp-mass basis (VGB_SAMPLED_BASIS_CHIRP; 2026-08-14
    # ruling, planned for the 6-MONTH run only): Mc moves from the per-leaf
    # fills to the sampled side, fdot stays DERIVED as
    # ``fdot_gr(f0, Mc) * (1 + r)`` exactly like the GB 9-column basis.
    # Default False keeps the legacy 5-dim distance basis BIT-UNCHANGED so
    # the live 3-mo/23-mo production runs resume their existing stores;
    # resuming a store written with the other basis fails loudly (ndim
    # guard in run.py) and points at
    # scripts/fstat_proposal/migrate_vgb_chirp_basis.py. Requires
    # sample_distance.
    chirp_mass_basis: bool = dataclasses.field(
        default_factory=env_default("VGB_CHIRP_MASS_BASIS", False, bool)
    )
    # Chirp-basis only: fdot_astro_ratio walker-init width as a FRACTION of
    # the ratio prior half-width (fdot_astro_ratio_max). The catalogue ratio
    # truth is exactly 0, so the sprint-wide MULTIPLICATIVE start scatter
    # would give the column zero ensemble spread -- and the affine-invariant
    # stretch move can never create spread it does not have (verified in
    # production: walker std 4.4e-19). The ratio column therefore gets
    # ADDITIVE jitter ``ratio_0 = VGB_START_FACTOR * ratio_init_width *
    # fdot_astro_ratio_max * randn`` (documented exception to the
    # multiplicative ruling for exactly-zero truths; START_FACTOR=0 still
    # gives the exact truth-null start).
    ratio_init_width: float = dataclasses.field(
        default_factory=env_default("VGB_RATIO_INIT_WIDTH", 0.02, float)
    )
    # {sampled-column index: additive width} consumed by run.py's generic
    # injection seeding (``x = truth + factor * width * randn`` for these
    # columns instead of the multiplicative form); resolved from
    # ratio_init_width in prepare_vgb_branch (chirp basis only).
    additive_start_widths: typing.Optional[typing.Dict[int, float]] = None
    # Red-blue stretch sweeps per iteration. VGBSpecialStretchMove runs each
    # repeat as eryn's sequential red-blue split (even-parity walkers move
    # against the current odd half, then odd against the UPDATED even half,
    # complement synced before each half; sequential_parity_repeats=True in
    # gbspecialstretch). Every half-sweep is an invariant kernel, so this
    # count is a COST knob, not a bias knob -- raise it freely (e.g. to
    # amortize the sig-het per-block reference build across many repeats,
    # mirroring the GB branch's default of 100).
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
    # Signal-heterodyne in-model likelihood: same three knobs (and the same
    # GB_SIGHET_* / SIGHET_* env names) as the GB branch, so the two branches
    # switch engines identically. Duplicated here rather than inherited
    # because the GB copies live on gb_no_fg's own GBNoFgGBSettings subclass,
    # not on the shared GBSettings -- matching how nt_sub/n_sparse above are
    # duplicated per branch.
    #
    # VGB is fixed-dimensional (no RJ), so toggling this is the clean
    # apples-to-apples chunked-het vs sig-het comparison: identical sources,
    # moves and iteration count, only the in-model likelihood differs.
    sighet_inmodel: bool = dataclasses.field(
        # Default ON (2026-08-12 user ruling; mirrors the GB branch).
        # Resolution: VGB_SIGHET_INMODEL (per-branch, wins when set) >
        # GB_SIGHET_INMODEL (shared, flips both branches) > True.
        default_factory=_vgb_sighet_inmodel_default
    )
    sighet_nt_layer: int = dataclasses.field(
        default_factory=env_default("SIGHET_NT_LAYER", 64, int)
    )
    sighet_n_sparse_fd: int = dataclasses.field(
        default_factory=env_default("SIGHET_N_SPARSE_FD", 1024, int)
    )
    # Heterodyne-ratio magnitude clip; 0 = disabled (the kernel's FLOOR_EPS
    # guards the divide-by-small). A positive value silently saturates any
    # candidate whose amplitude ratio vs the fixed reference exceeds it —
    # diagnostic only, never for sampling runs.
    sighet_max_r: float = dataclasses.field(
        default_factory=env_default("SIGHET_MAX_R", 0.0, float)
    )
    # Control-point count for the sig-het spline waveform build (shared
    # machinery with chunked-het's N_cp_sig splines). -1 = AUTO (one node
    # per ~4 days of Tobs, clamped [32, 256]); 0 = direct per-point
    # evaluation (legacy); >1 = explicit count.
    sighet_n_cp: int = dataclasses.field(
        default_factory=env_default("SIGHET_N_CP", -1, int)
    )
    # Signal-het V3: ratio-spline candidate build (r modeled directly from
    # n raw evals per candidate; no FFT/polyphase/division). 0 = off (v2
    # exact build); >0 = fixed node count; -1 = ADAPTIVE from the batch's
    # predicted displacement (prototype-calibrated policy, clip [8, 64]).
    sighet_v3_nodes: int = dataclasses.field(
        default_factory=env_default("SIGHET_V3_NODES", 64, int)
    )
    # Signal-het V4: the fitted ratio is resampled onto ``sighet_v4_knots``
    # FIXED, candidate-independent knots as linear complex values before the
    # pixel-time evaluation -- the representation that makes the fold a fixed
    # linear operator.  0 = off (v3/v2 path).  128 is the measured-converged
    # value; 64 is lossy, 256 buys nothing.  Requires sighet_v3_nodes > 0
    # (v4 reuses the v3 node fit).
    sighet_v4_knots: int = dataclasses.field(
        default_factory=env_default("SIGHET_V4_KNOTS", 128, int)
    )
    # V4 evaluation mode: 0 = cooperative fixed-knot spline solve (PCR on
    # GPU); >0 = precomputed cardinal weights with this half-band -- no
    # solve, no block syncs, ~18 KB less shared memory.  Banded and PCR agree
    # to 1e-11 relative at half-band 16 and banded is never slower.
    sighet_v4_band: int = dataclasses.field(
        default_factory=env_default("SIGHET_V4_BAND", 16, int)
    )
    # V5 kernel (scratch-eliminated v4; bit-identical, 5x occupancy).
    # Default ON per the 1fd83b1 ruling ("both blocks") -- the field was
    # MISSING here until 2026-08-12 smoke 2 showed the VGB comp silently
    # building v4-only (scorer line "v5=0"). Shared env, like the GB block.
    sighet_v5: int = dataclasses.field(
        default_factory=env_default("SIGHET_V5", 1, int)
    )
    # (nleaves, len(fixed basis)) per-leaf fixed values in SAMPLING units
    # (f0 in mHz), ordered like the active fixed basis (vgb_fixed_basis);
    # feeds the per-leaf fill list.
    fixed_params: typing.Optional[typing.Any] = None
    # (nleaves, ndim) sampling-basis truth rows (seeds the fixed-leaf start).
    injection: typing.Optional[typing.Any] = None
    # VGB band separations DEFAULT to the same per-WDM-layer edges as the GB
    # branch (GBSetup.init_band_structure) for ease; this knob coarsens them
    # to L layers per band. Only bands that actually contain VGB leaves do
    # any work (the band sorter holds real leaves only).
    band_layers: int = dataclasses.field(
        default_factory=env_default("VGB_BAND_LAYERS", 1, int)
    )
    # Band-edge mode overrides: VGB reads its OWN env knobs (VGB_BAND_*)
    # so flipping the GB branch to variable-width bands
    # (GB_BAND_EDGES_MODE=get_n) does NOT silently move the VGB band grid
    # (which has no RJ surface and whose stored per-band arrays would
    # otherwise also need migrating on resume). Set VGB_BAND_EDGES_MODE
    # explicitly to opt VGB in.
    band_edges_mode: str = dataclasses.field(
        default_factory=env_default("VGB_BAND_EDGES_MODE", "uniform", str)
    )
    band_target_count: int = dataclasses.field(
        default_factory=env_default("VGB_BAND_TARGET_COUNT", 0, int)
    )
    # DEPRECATED like the GB field (2026-08-15 width ruling: get_n bands
    # are sized by 2*get_N(f_max_band)/Tobs); 1.0 = the legacy default,
    # ignored without a warning; any other value is warned about and
    # ignored by the builder.
    band_min_layers: float = dataclasses.field(
        default_factory=env_default("VGB_BAND_MIN_LAYERS", 1.0, float)
    )


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

        # Band separations default to the SAME per-WDM-layer edges as the GB
        # branch (what super() just built). VGB_BAND_LAYERS > 1 coarsens the
        # separation to L layers per band; the interior-band bookkeeping
        # (f0_lims, num_sub_bands, band_N_vals) is recomputed to match.
        L = int(getattr(self, "band_layers", 1) or 1)
        if L > 1:
            edges = np.asarray(self.band_edges)
            keep = edges[::L]
            if keep[-1] != edges[-1]:
                keep = np.append(keep, edges[-1])
            # per-band N: max over the merged fine bands (unused on WDM,
            # conservative on FD)
            n_vals = np.asarray(self.band_N_vals)
            grp = np.arange(len(n_vals)) // L
            self.band_N_vals = np.maximum.reduceat(n_vals, np.unique(grp, return_index=True)[1])[: len(keep) - 1]
            self.band_edges = keep
            self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]
            self.num_sub_bands = len(self.band_edges) - 1
            self.logger.info(
                "VGB band separations coarsened to %d WDM layers/band "
                "(VGB_BAND_LAYERS): %d sub-bands.", L, self.num_sub_bands,
            )

    def init_sampling_info(self):
        if self.fixed_params is None:
            raise ValueError(
                "VGBSetup needs per-leaf fixed_params (nleaves, 3) "
                "[f0 (mHz), alpha, sin_delta]; run prepare_vgb_branch first "
                "(mojito VGB catalogue required)."
            )
        fixed = np.asarray(self.fixed_params, dtype=float)
        n_leaves = fixed.shape[0]
        _fixed_basis = vgb_fixed_basis(self)
        _sampled_basis = vgb_sampled_basis(self)
        assert fixed.shape == (n_leaves, len(_fixed_basis))
        assert int(self.ndim) == len(_sampled_basis), (
            f"VGB ndim ({self.ndim}) != sampled basis width "
            f"({len(_sampled_basis)}: {_sampled_basis}); ndim is 6 only "
            "with VGB_CHIRP_MASS_BASIS=1 (prepare_vgb_branch resolves it)."
        )
        if self.nleaves_max is not None:
            assert int(self.nleaves_max) == n_leaves, (
                f"fixed_params rows ({n_leaves}) != nleaves_max "
                f"({self.nleaves_max})"
            )

        if self.transform is None:
            # One per-leaf fill dict per source, keys shared across leaves
            # (asserted by Eryn). Values in SAMPLING units — the factory's
            # registered transforms convert the filled columns. In the
            # distance basis the astro quad transform EMITS fddot (exactly
            # 0), so the fills must not touch that slot.
            fill_list = [
                {
                    **({} if self.sample_distance else {"fddot": 0.0}),
                    **{
                        name: fixed[leaf, j]
                        for j, name in enumerate(_fixed_basis)
                    },
                }
                for leaf in range(n_leaves)
            ]
            if self.sample_distance:
                if self.chirp_mass_basis:
                    # Mc is SAMPLED: same mc box convention as GBSetup
                    # (transform inverse clip + the uniform Mc prior below):
                    # m_chirp_lims when set, else (0.001, 1.0).
                    _mc_lims = (tuple(self.m_chirp_lims) if self.m_chirp_lims
                                else (0.001, 1.0))
                else:
                    _mc = fixed[:, _fixed_basis.index("Mc")]
                    _mc_lims = (0.5 * float(_mc.min()), 2.0 * float(_mc.max()))
                self.transform = make_gb_transform_container(
                    use_chirp_mass=True,
                    use_fdot_astro=True,
                    use_distance=True,
                    input_basis=list(_sampled_basis),
                    fill_dict=fill_list,
                    mc_lims=_mc_lims,
                )
            else:
                self.transform = make_gb_transform_container(
                    use_chirp_mass=False,
                    input_basis=list(_sampled_basis),
                    fill_dict=fill_list,
                )

        if self.periodic is None:
            self.periodic = {"vgb": {"phi0": 2 * np.pi, "psi": np.pi}}

        if self.priors is None:
            from eryn.prior import ProbDistContainer, uniform_dist

            if self.sample_distance:
                # Global uniforms over [dist, phi0, cos_iota, psi, (Mc,) r].
                # The r box must cover every catalogue ratio (r_cat = 0
                # exactly for GW-driven catalogues), the dist box every
                # catalogue distance, and (chirp basis) the Mc box every
                # catalogue chirp mass.
                _rmax = float(self.fdot_astro_ratio_max)
                # Mirrors the GB chirp-mass slot (gb.py use_chirp_mass
                # branch): uniform over m_chirp_lims, defaulted to the same
                # (0.001, 1.0) box the GB transform factory uses.
                _mc_box = (tuple(self.m_chirp_lims) if self.m_chirp_lims
                           else (0.001, 1.0))
                if self.injection is not None:
                    inj = np.asarray(self.injection, dtype=float)
                    _d = inj[:, _sampled_basis.index("dist")]
                    _r = inj[:, _sampled_basis.index(
                        "fdot_astro_ratio")]
                    if not ((_d > self.dist_lims[0]).all()
                            and (_d < self.dist_lims[1]).all()):
                        raise ValueError(
                            f"VGB dist prior {self.dist_lims} kpc does not "
                            "cover the catalogue distances "
                            f"[{_d.min():.3e}, {_d.max():.3e}] kpc."
                        )
                    if not (np.abs(_r) < _rmax).all():
                        raise ValueError(
                            f"VGB fdot_astro_ratio prior [-{_rmax}, {_rmax}]"
                            " does not cover the catalogue ratios "
                            f"[{_r.min():.3f}, {_r.max():.3f}]; raise "
                            "VGBSettings.fdot_astro_ratio_max."
                        )
                    if self.chirp_mass_basis:
                        _mc = inj[:, _sampled_basis.index("Mc")]
                        if not ((_mc > _mc_box[0]).all()
                                and (_mc < _mc_box[1]).all()):
                            raise ValueError(
                                f"VGB Mc prior {list(_mc_box)} Msol does "
                                "not cover the catalogue chirp masses "
                                f"[{_mc.min():.3f}, {_mc.max():.3f}]; set "
                                "VGBSettings.m_chirp_lims."
                            )
                # Distance prior MATCHES the GB distance-basis setup
                # (gb.py, use_distance branch): the placeholder there is a
                # 3-D (dist, alpha, sin_delta) joint of independent
                # uniforms with dist ~ U(dist_lims) LINEAR in kpc. VGB sky
                # is per-leaf fixed, so the samplable piece is exactly that
                # dist marginal — same box, same linear uniform. When the
                # real Galaxy 3-D distribution replaces the placeholder
                # (GBSettings.sky_dist_distribution), VGB needs the
                # PER-LEAF conditional p(dist | alpha_i, sin_delta_i),
                # which the shared prior container cannot express yet —
                # same documented follow-up as the GB birth container's
                # independent-U(dist) draw. Fail loudly rather than
                # silently ignoring the joint.
                if self.sky_dist_distribution is not None:
                    raise NotImplementedError(
                        "VGB distance basis with a joint "
                        "sky_dist_distribution needs per-leaf conditional "
                        "p(dist | sky) priors (not yet supported); unset "
                        "it or use VGB_SAMPLE_DISTANCE=0."
                    )
                priors_vgb = {
                    0: uniform_dist(self.dist_lims[0], self.dist_lims[1]),
                    1: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: sort defensively.
                    2: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    3: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                }
                if self.chirp_mass_basis:
                    # Mc ~ U(m_chirp_lims | (0.001, 1.0)) — same form/box
                    # as the GB chirp-mass slot (gb.py use_chirp_mass).
                    priors_vgb[4] = uniform_dist(_mc_box[0], _mc_box[1])
                # r ~ U[-M, M] in the SAMPLING basis (no in-sampler
                # Jacobian): induced physical prior at fixed (f0, Mc)
                # is uniform in fdot — same convention as GB.
                priors_vgb[_sampled_basis.index("fdot_astro_ratio")] = (
                    uniform_dist(-_rmax, _rmax)
                )
            else:
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
                            f"[{self.fdot_lims[0]:.3e}, "
                            f"{self.fdot_lims[1]:.3e}] "
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


def load_vgb_catalogue_file(mojito_data_path: str) -> dict:
    """Read the (small) VGB catalogue file directly into the L1-loader
    layout ``{"vgb": {column: array}}``.

    The catalogue lives in ``catalogues/`` separately from the (large) L1
    data bricks, so it is available without any data transfer -- this is
    what lets ``data_mode="synthetic"`` build a catalogue-faithful VGB run
    fully in-process.
    """
    import h5py

    path = os.path.join(
        mojito_data_path, "catalogues", "vgb_cat_mojito_lite_processed.hdf5"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VGB catalogue file not found at {path!r} (needed for the "
            "synthetic VGB mode)."
        )
    with h5py.File(path, "r") as f:
        B = f["Binaries"]
        entry = {k: np.asarray(B[k][:]) for k in B.keys()}
    return {"vgb": entry}


def prepare_vgb_branch(vgb: VGBSettings, general_setup: GeneralSetup, *,
                       data_mode: str, synthetic_t_start: float | None = None):
    """Resolve the VGB branch from the mojito VGB catalogue.

    Builds the sampling-basis rows through the single GB factory convention
    (``gb_catalogue_to_sampling_basis`` -> container inverse) and splits
    them BY NAME into the sampled columns (``injection``) and the fixed
    per-leaf columns (``fixed_params``) of the active basis
    (:func:`vgb_sampled_basis` / :func:`vgb_fixed_basis`; ``ndim`` is
    re-resolved to match); sets the fixed-dimensional leaf count and the
    band structure bounds (one guard band each side — the band machinery
    never proposes in the first/last band).

    ``data_mode="synthetic"``: the catalogue is read straight from the
    (small) catalogue file — no L1 data needed — and the phase/frequency
    reference epoch is the synthetic stream start, matching the in-process
    injection built by the variant's synthetic processor.
    """
    if data_mode not in ("mojito", "synthetic"):
        raise ValueError(
            "The VGB branch needs the mojito VGB catalogue "
            f"(data_mode='mojito' or 'synthetic'); got data_mode={data_mode!r}."
        )
    if vgb.chirp_mass_basis and not vgb.sample_distance:
        # never silently ignore the knob (rule-0 failure mode)
        raise ValueError(
            "VGB_CHIRP_MASS_BASIS=1 requires the distance basis "
            "(VGB_SAMPLE_DISTANCE=1); the legacy (lnA, fdot) basis has no "
            "chirp-mass variant."
        )
    catalogue = (getattr(general_setup, "catalogue", None) or {}).get("VGB", {})
    if not catalogue and data_mode == "synthetic":
        catalogue = load_vgb_catalogue_file(general_setup.mojito_data_path)
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
    n = rows.shape[0]

    if vgb.sample_distance:
        # Distance basis: dist/Mc come straight from the catalogue columns
        # (same sorted-key concatenation order as ``rows``), the ratio from
        # the catalogue fdot vs the GW-driven fdot_gr(f0, Mc), and
        # phi0/cos_iota/psi from the standard sampling rows (the phi0 sign
        # convention stays routed through the container). With
        # ``chirp_mass_basis`` Mc lands on the SAMPLED side instead of the
        # per-leaf fills.
        from .transforms import McDistFdotAstroQuad, gb_amp_from_dist

        def _cat_col(name):
            return np.concatenate([
                np.atleast_1d(np.asarray(catalogue[k][name], dtype=float))
                for k in sorted(catalogue.keys())
            ])

        # Catalogue LuminosityDistance is in Mpc; the gbgpu amplitude
        # convention (gb_amp_from_dist) takes kpc. Verified against the
        # catalogue Amplitude column: the ratio is exactly 1000, and after
        # conversion the (f0, Mc, d) -> A relation reproduces Amplitude
        # (hard-checked below, so a future catalogue convention change
        # fails loudly instead of shifting every injection).
        d_kpc = _cat_col("LuminosityDistance") * 1e3
        mc = _cat_col("ChirpMassSSBFrame")
        f0_hz_rows = rows[:, full_basis.index("f0")] * 1e-3
        a_phys = np.exp(rows[:, full_basis.index("A")])
        fdot_phys = rows[:, full_basis.index("fdot")]
        _, _, fdot_gr, _ = McDistFdotAstroQuad()(
            d_kpc, f0_hz_rows, mc, np.zeros_like(d_kpc))
        ratio = fdot_phys / fdot_gr - 1.0
        rel = np.abs(gb_amp_from_dist(f0_hz_rows, mc, d_kpc) / a_phys - 1.0)
        if rel.max() > 1e-3:
            raise ValueError(
                "VGB catalogue (f0, Mc, dist) does not reproduce the "
                f"catalogue Amplitude (max rel {rel.max():.3e}); check the "
                "LuminosityDistance units / amplitude convention."
            )
        _sampled_basis = vgb_sampled_basis(vgb)
        inj = np.empty((n, len(_sampled_basis)))
        inj[:, _sampled_basis.index("dist")] = d_kpc
        for name in ("phi0", "cos_iota", "psi"):
            inj[:, _sampled_basis.index(name)] = (
                rows[:, full_basis.index(name)])
        inj[:, _sampled_basis.index("fdot_astro_ratio")] = ratio
        if vgb.chirp_mass_basis:
            inj[:, _sampled_basis.index("Mc")] = mc
            vgb.fixed_params = rows[:, fixed_idx]
            # Zero-truth ratio + multiplicative init + pure stretch =
            # collapsed dimension (zero spread the stretch can never grow).
            # Additive-jitter exception: width = ratio_init_width fraction
            # of the prior half-width, still scaled by VGB_START_FACTOR so
            # 0 keeps the exact truth-null start.
            vgb.additive_start_widths = {
                _sampled_basis.index("fdot_astro_ratio"):
                    float(vgb.ratio_init_width) * float(vgb.fdot_astro_ratio_max)
            }
        else:
            vgb.fixed_params = np.column_stack([rows[:, fixed_idx], mc])
        vgb.injection = inj
    else:
        vgb.injection = rows[:, sampled_idx]
        vgb.fixed_params = rows[:, fixed_idx]
    vgb.ndim = len(vgb_sampled_basis(vgb))
    vgb.nleaves_min = vgb.nleaves_max = n

    if vgb.t0 in (None, 0.0):
        # phase/frequency reference epoch: the mojito catalogue epoch, or
        # the synthetic stream start (the synthetic processor injects the
        # same rows at that epoch, so truth-null holds identically).
        if data_mode == "synthetic":
            vgb.t0 = float(synthetic_t_start if synthetic_t_start is not None
                           else 10_000.0)
        else:
            vgb.t0 = MOJITO_REFERENCE_TIME

    # Band structure bounds from the fixed f0 table, one guard band per
    # side: run_proposal never proposes in the first/last band, and
    # GBSetup.init_band_structure's f0_lims are the interior edges.
    f0_hz = rows[:, full_basis.index("f0")] * 1e-3
    from lisatools.domains import WDMSettings

    domain_settings = general_setup.domain_settings
    if isinstance(domain_settings, WDMSettings):
        # one guard BAND each side; a band is band_layers WDM layers wide
        _L = int(getattr(vgb, "band_layers", 1) or 1)
        guard = 2.0 * _L * float(domain_settings.layer_df)
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
        # BUGFIX (2026-08-15, root-caused from the frozen-VGB monitor
        # panel): this used general_setup.ntemps (1 in cold-chain-only
        # runs) instead of the branch's own ladder, and unconditionally
        # clobbered the last rung -- betas collapsed to [1e-4], turning
        # the VGB likelihood OFF (every "posterior" was the SNR-gated
        # prior). Mirror gb.py: resolve from the branch's own ntemps and
        # never clobber a single-rung ladder.
        #
        # NOTE (resume): this ladder is the CONFIGURED one. A resumed store
        # carries its own rung count in sub_backend/vgb/band_temps and that
        # one WINS -- build_vgb_moves reconciles (warning names both counts)
        # and overwrites ``vgb.betas`` with the stored ladder. To actually
        # change the rung count of a live store, re-rung it first with
        # scripts/fstat_proposal/fix_vgb_band_temps.py <store.h5> <k>.
        _nt = int(getattr(vgb, "ntemps", None) or general_setup.ntemps)
        betas = 1.0 / 1.2 ** np.arange(_nt)
        if _nt > 1:
            betas[-1] = 1e-4
        vgb.betas = betas
    vgb.gb_wdm_comp = None

    logger.info(
        "VGB branch: %d catalogue sources, f0 in [%.6e, %.6e] Hz, band "
        "[%.6e, %.6e] Hz.",
        n, f0_hz.min(), f0_hz.max(), vgb.start_freq, vgb.end_freq,
    )
    return vgb
