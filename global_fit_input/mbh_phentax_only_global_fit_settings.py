"""Global-fit settings: MBH-only smoke test (CPU + WDM + self-generated
injection) wired around the ``phentax`` IMRPhenomTHM JAX waveform model.

Structurally a 1:1 mirror of ``emri_only_global_fit_settings.py``:

* ``IMRPhenomTHMWaveform`` plays the role of the EMRI waveform-with-
  ``ResponseWrapper``-compatible-``__call__`` adapter (analogous to
  :class:`lisatools.sources.sobbh.SOBBHWaveform`).
* ``get_mbh_phentax_response_wrapper`` caches one
  :class:`ResponseWrapper` so the synthetic-injection data loader and
  the template move share a single waveform instance.
* ``MBHWaveWrap`` projects the TD output to the run's domain
  (FD / STFT / WDM).
* ``SyntheticMBHProcessingStep`` generates the injection in-process.
* The recipe wires a :class:`ResidualAddOneRemoveOneMove` for MBH PE.

Smoke-test choices:

* CPU-only (no cupy / no GPU paths).
* WDM grid ``Nf=720, Nt=2160, dt=5`` so ``Tobs = Nf*Nt*dt = 90 d ≈ 3 mo``.
* ``nwalkers=4`` and ``ntemps=2`` for fast turnaround.
* MBH injection generated in-process — no external h5 paths.
* Single cached ``ResponseWrapper`` reused between the synthetic-
  injection data loader and the template move so the (slow) phentax
  generator setup runs once.
"""

import gc
import logging
import shutil
from typing import Optional

import numpy as np

# CPU-only smoke test. Map ``cp`` to numpy so the existing call sites
# (``cp.cuda.runtime.setDevice`` etc.) can still be guarded by
# ``gpu_available``.
import numpy as cp

GPU_BACKEND = "cpu"
gpu_available = False

logger = logging.getLogger(__name__)

from bbhx.utils.transform import mT_q  # paired (logM, q) -> (m1, m2)

from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils.transform import TransformContainer

from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.domains import (
    DomainBaseArray,
    FDSettings,
    TDSettings,
    TDSignal,
    WDMSettings,
)
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.moves import ResidualAddOneRemoveOneMove
from lisatools.globalfit.preprocessing import BaseProcessingStep
from lisatools.globalfit.recipe import RecipeStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import MBHSettings, MBHSetup
from lisatools.utils.constants import YRSID_SI

# ``phentax`` is the JAX implementation of IMRPhenomT(HM). Imported lazily
# inside :class:`IMRPhenomTHMWaveform` so the module-level import path
# stays cheap if a user only wants to inspect the settings.
import jax.numpy as jnp


# ============================================================
# *** WDM grid ***
# ============================================================
NF = 720
NT = 2160
DT = 5.0
TOBS = NF * NT * DT  # 7,776,000 s ≈ 90 d ≈ 3 mo
T_START = 0.0


DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(NT - 20) * 3600.0,
)


# ============================================================
# *** Self-generated MBH injection ***
# ============================================================
# phentax's :class:`IMRPhenomTHM` consumes 8 intrinsic + 3 extrinsic
# parameters in the order used below by :class:`IMRPhenomTHMWaveform`.
# We choose an 11-parameter waveform basis to match the existing
# ``MBHSetup`` sampling philosophy (intrinsic masses + spins + sky
# direction + polarization + merger time within the observation):
#
#   waveform basis (call-signature order — what the waveform sees):
#     0 m1       (M_sun)
#     1 m2       (M_sun)
#     2 s1z      (dimensionless aligned spin)
#     3 s2z      (dimensionless aligned spin)
#     4 dist     (Gpc — converted to Mpc inside)
#     5 phi_ref  (rad)
#     6 inc      (rad)
#     7 psi      (rad)
#     8 lam      (ecliptic longitude, rad) — consumed by ResponseWrapper
#     9 beta     (ecliptic latitude, rad)  — consumed by ResponseWrapper
#    10 t_plunge (s, merger time in the observation window)
#
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        1.0e6,          # m1 (M_sun)
        5.0e5,          # m2 (M_sun)
        0.5,            # s1z
        0.3,            # s2z
        10.0,           # dist (Gpc)
        1.0,            # phi_ref
        np.pi / 3.0,    # inclination
        0.5,            # psi
        1.5,            # lam (ecliptic longitude)
        0.3,            # beta (ecliptic latitude)
        0.5 * TOBS,     # t_plunge — merger placed mid-observation
    ]
)


def mbh_full_to_sampling(params_full: np.ndarray) -> np.ndarray:
    """Convert an 11-param waveform-basis vector to the 11-param sampling basis.

    Sampling basis re-parameterises the intrinsic + extrinsic ranges to be
    flat-prior friendly:

      ``logM = log(m1 + m2)`` (total mass)
      ``q = m2 / m1`` (mass ratio, 0 < q < 1 when m1 >= m2 — we don't
        enforce ordering here; pass the same convention as the prior
        bounds use)
      ``cos_iota = cos(inclination)``
      ``sin_beta = sin(beta)``
      All other params pass through unchanged.
    """
    # TODO: add an inversion tool to the transform container
    p = np.asarray(params_full, dtype=float).copy()
    m1, m2 = float(p[0]), float(p[1])
    p[0] = np.log(m1 + m2)
    p[1] = m2 / m1
    p[6] = np.cos(p[6])      # cos_iota
    p[9] = np.sin(p[9])      # sin_beta
    return p


# ---- Cached MBH generator + wrappers (shared across data path + moves) ----

_WAVE_GEN_CACHE = {}


class IMRPhenomTHMWaveform:
    """LISA-side IMRPhenomTHM (phentax) waveform generator.

    Mirrors the call-signature contract that
    ``fastlisaresponse.ResponseWrapper`` expects (and that
    :class:`lisatools.sources.sobbh.SOBBHWaveform` implements for SOBBHs):

    * ``__init__(Tobs, dt, t0, ...)`` builds the internal uniform time
      grid + caches a single :class:`phentax.IMRPhenomTHM` instance.
    * ``__call__(m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta,
      t_plunge, **kwargs) -> complex Array`` evaluates the polarisation-
      rotated waveform on the internal time grid and returns
      ``hp + 1j*hx``.

    ``lam`` and ``beta`` are accepted in the call signature so
    ``ResponseWrapper`` can read them off (via ``index_lambda`` /
    ``index_beta``) and apply the LISA sky projection; the waveform
    itself does not use them.

    The merger is placed at ``t = t_plunge`` within the observation
    window. phentax produces its waveform on a grid centred at the
    merger (``t = 0``), so we ask it for ``t_min = -t_plunge`` and
    ``T = Tobs``; the returned ``num_steps = ceil(Tobs/dt)`` samples
    then align one-to-one with our target grid because the spacing
    is ``delta_t = dt`` on both.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t0: float = 0.0,
        higher_modes: Optional[object] = "all",
        include_negative_modes: bool = True,
        coarse_grain: bool = False,
        force_backend: str = "cpu",
        pad_zeros: bool = True,
    ):
        # Lazy-import phentax so the module-level import-time cost is paid
        # only when the smoke test is actually constructed.
        from phentax.waveform import IMRPhenomTHM

        self.Tobs = Tobs
        self.dt = dt
        self.t0 = t0
        self.force_backend = force_backend
        self.pad_zeros = pad_zeros
        N = int(round(Tobs / dt))
        self._N = N
        self._times_np = np.arange(N) * dt + t0

        self._phentax = IMRPhenomTHM(
            higher_modes=higher_modes,
            include_negative_modes=include_negative_modes,
            coarse_grain=coarse_grain,
            T=Tobs,
        )

    @property
    def N(self) -> int:
        return self._N

    @property
    def times(self) -> np.ndarray:
        return self._times_np

    def __call__(
        self,
        m1,
        m2,
        s1z,
        s2z,
        dist,
        phi_ref,
        inc,
        psi,
        lam,
        beta,
        t_plunge,
        **kwargs,
    ):
        """Evaluate the polarisation-rotated waveform on the internal time grid.

        Returns a complex array ``h = hp + 1j*hx`` so the call signature
        matches what ``fastlisaresponse.ResponseWrapper`` expects (it
        reads ``h.real`` / ``h.imag`` to get the two polarisations).
        Set ``flip_hx=True`` on ``ResponseWrapper`` to get the standard
        ``h.real - 1j*h.imag`` convention.

        ``dist`` is in Gpc; converted to Mpc inside (phentax convention).
        ``lam`` / ``beta`` are accepted but unused here — they're
        forwarded through ``ResponseWrapper`` for sky-projection.
        ``t_plunge`` is the merger time relative to the start of the
        observation; phentax internally uses merger == 0.
        """
        del lam, beta  # consumed by ResponseWrapper, not the source waveform
        # ResponseWrapper injects ``T`` and ``dt`` into kwargs; ignore.
        kwargs.pop("T", None)
        kwargs.pop("dt", None)
        kwargs.pop("convert_to_ra_dec", None)

        dist_mpc = float(dist) * 1.0e3  # Gpc -> Mpc (phentax convention)
        t_plunge_f = float(t_plunge)

        # phentax compute_polarizations returns (times, mask, h_plus, h_cross)
        # with merger at t=0. Setting ``t_min = -t_plunge`` and ``T = Tobs``
        # makes the returned ``ceil(T/dt)`` samples align with our target
        # ``[0, Tobs)`` grid sample-for-sample.
        _, mask_phen, hp_phen, hx_phen = self._phentax.compute_polarizations(
            m1=float(m1),
            m2=float(m2),
            chi1z=float(s1z),
            chi2z=float(s2z),
            distance=dist_mpc,
            phi_ref=float(phi_ref),
            inclination=float(inc),
            psi=float(psi),
            delta_t=float(self.dt),
            t_min=-t_plunge_f,
            t_ref=-t_plunge_f,
            T=float(self.Tobs),
        )

        # phentax always returns a leading batch axis (shape (1, N) for a
        # single-source call); squeeze it so the wrapped ``ResponseWrapper``
        # (which uses ``len(h)`` to size the projection) sees a 1D array.
        hp = np.asarray(hp_phen, dtype=np.float64).reshape(-1)
        hx = np.asarray(hx_phen, dtype=np.float64).reshape(-1)
        mask = np.asarray(mask_phen, dtype=bool).reshape(-1)

        # Zero out invalid samples flagged by phentax's adaptive/uniform grid.
        hp = np.where(mask, hp, 0.0)
        hx = np.where(mask, hx, 0.0)

        # Pad with zeros (or clip) to exactly N so ``ResponseWrapper`` and
        # the WDM transform downstream see the right shape.
        if hp.shape[-1] < self._N:
            if self.pad_zeros:
                pad = self._N - hp.shape[-1]
                hp = np.pad(hp, (0, pad), mode="constant")
                hx = np.pad(hx, (0, pad), mode="constant")
            else:
                raise ValueError(
                    f"phentax produced {hp.shape[-1]} samples but N={self._N}; "
                    "either set pad_zeros=True or increase the input grid."
                )
        elif hp.shape[-1] > self._N:
            hp = hp[: self._N]
            hx = hx[: self._N]

        return hp + 1j * hx


def get_mbh_phentax_response_wrapper(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    tdi_config: TDIConfig,
    tdi_chan: str = "XYZ",
    role: str = "template",
    order: int = 40,
    t_buffer: float = 3e4,
    higher_modes: Optional[object] = "all",
    orbits: Optional[Orbits] = None,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`ResponseWrapper` around
    :class:`IMRPhenomTHMWaveform`.

    Mirrors :func:`get_emri_response_wrapper`: a single generator is
    reused between the synthetic-injection data loader and the template
    move so the slow phentax JIT compilation runs once per
    ``(Tobs, dt, t_start, tdi_chan, order, force_backend, higher_modes)``
    key.

    Note: this mirrors the EMRI smoke-test convention where ``role`` is
    intentionally **not** part of the cache key, so injection and
    template share the same generator instance.
    """
    key = (
        Tobs, dt, t_start, tdi_chan, order, force_backend,
        str(higher_modes),
    )
    if key in _WAVE_GEN_CACHE:
        return _WAVE_GEN_CACHE[key]

    waveform_gen = IMRPhenomTHMWaveform(
        Tobs=Tobs,
        dt=dt,
        t0=t_start,
        higher_modes=higher_modes,
        force_backend=force_backend,
    )

    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        # Indices match the call signature of IMRPhenomTHMWaveform.__call__:
        #   (m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta, t_plunge)
        #              0   1   2    3    4     5    6    7    8     9     10
        "index_lambda": 8,
        "index_beta": 9,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "is_ecliptic_latitude": True,
        "t_buffer": t_buffer,
    }

    if orbits is None:
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        
    # ResponseWrapper.get_t0_shift_to_data(, dt: float, t_start: float) -> float:
    wave_gen = ResponseWrapper(
        waveform_gen,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class MBHWaveWrap:
    """Adapter that runs the cached ResponseWrapper and projects to the
    run's domain.

    Mirrors :class:`EMRIWaveWrap` from
    ``emri_only_global_fit_settings.py`` — the call output is a
    :class:`DomainBase` subclass (FDSignal / WDMSignal / ...) so the
    global-fit move and ACA dispatch land on the right kernels.
    """

    def __init__(
        self,
        wave_gen,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def __call__(self, *params, **kwargs):
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # ``convert_to_ra_dec=False`` keeps the sampler basis aligned
        # with ``[lam, beta]`` (ecliptic) rather than (ra, dec).
        call_kwargs.setdefault("convert_to_ra_dec", False)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        breakpoint()
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


class SyntheticMBHProcessingStep(BaseProcessingStep):
    """Generate the MBH injection in-process via the shared ResponseWrapper.

    Hands ``(times, data, fs)`` to :class:`BaseProcessingStep` — same
    interface as :class:`SangriaProcessingStep`. No external h5 path
    needed.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params_full_basis: np.ndarray,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        higher_modes: Optional[object] = "all",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend="cpu")
        wave_gen = get_mbh_phentax_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            higher_modes=higher_modes,
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis, convert_to_ra_dec=False)
        )
        td_signal = np.atleast_2d(td_signal)[:nchannels]

        # ``ResponseWrapper`` can produce a couple fewer samples than
        # ``Tobs/dt`` (Lagrange buffer / remove_garbage). Pad/clip to
        # exactly ``Tobs/dt`` so downstream WDM ``N = Nf*Nt`` matches.
        target_N = int(round(Tobs / dt))
        if td_signal.shape[-1] < target_N:
            pad = target_N - td_signal.shape[-1]
            td_signal = np.pad(td_signal, ((0, 0), (0, pad)), mode="constant")
        elif td_signal.shape[-1] > target_N:
            td_signal = td_signal[:, :target_N]

        N = td_signal.shape[-1]
        times = np.arange(N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


################
#  RECIPE STEPS
################


class MBHPERecipeStep(RecipeStep):
    """PE-only step (runs indefinitely; outer ``run_mcmc`` count caps it)."""

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        return False


################
#  RECIPE
################


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    print("[setup_recipe] entered", flush=True)
    mbh_info = curr.source_info["mbh"]
    general_info = curr.general_info
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    domain_settings = general_info.domain_settings
    print(
        f"[setup_recipe] nwalkers={nwalkers} ntemps={ntemps} "
        f"domain={type(domain_settings).__name__} nchannels={acs.nchannels}",
        flush=True,
    )

    # Pull the cached ResponseWrapper (built once at data-load time) and
    # wrap it for the template path. ``MBHWaveWrap`` returns a
    # ``DomainBase`` so ACA add/remove_signal_to_residual and the MBH
    # move's get_waveform_here use the right kernels.
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    template_wave_gen = get_mbh_phentax_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="template",
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend="cpu",
    )
    wave_gen = MBHWaveWrap(
        template_wave_gen,
        td_settings,
        domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Pre-inject starting templates into each walker's residual.
    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        print(
            f"[setup_recipe] pre-injecting MBH starts; nleaves_max={mbh_inds.shape[-1]}",
            flush=True,
        )
        for leaf in range(mbh_inds.shape[-1]):
            if not mbh_inds[0, leaf]:
                continue
            assert np.all(mbh_inds[:, leaf])
            inj_coords = state.branches_coords["mbh"][0, :, leaf]
            inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
            n_starts = inj_coords.shape[0]
            print(
                f"[setup_recipe] generating + applying {n_starts} starting waveforms (leaf {leaf}) one at a time...",
                flush=True,
            )
            # One MBH waveform at a time: generate, add to that walker's
            # residual, drop the reference, force a gc cycle. Keeps peak
            # RAM at one waveform worth instead of ``n_starts``
            # simultaneously.
            for i in range(n_starts):
                print(
                    f"[setup_recipe]   walker {i}/{n_starts}: generating...",
                    flush=True,
                )
                sig = wave_gen(*inj_coords_in[i], **mbh_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()
    print("[setup_recipe] pre-injection done; building MBH PE move", flush=True)

    betas_all = np.tile(
        make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1)
    )
    state.sub_states["mbh"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_pe_move = ResidualAddOneRemoveOneMove(
        "mbh",
        coords_shape,
        wave_gen,
        mbh_info.waveform_kwargs.copy(),
        mbh_info.waveform_kwargs.copy(),
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        mbh_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(
        MBHPERecipeStep(moves=[mbh_pe_move]), name="mbh pe"
    )


##########################
#  SETTINGS
##########################


def _build_mbh_phentax_transform() -> TransformContainer:
    """Build the sampling-basis -> waveform-basis transform for phentax MBH.

    Mirrors the convention used by the legacy ``MBHSetup.init_sampling_info``
    in lisatools: identical input + output basis names, and the
    ``parameter_transforms`` operate *in-place* on those positions. So
    after the transform fires, the value at position ``logM`` is in fact
    ``m1`` (via ``(logM, q) -> mT_q``), the value at ``q`` is ``m2``, etc.
    ``IMRPhenomTHMWaveform.__call__`` then receives them positionally in
    the order: ``(m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta,
    t_plunge)``.

    Unlike the legacy MBH transform we do *not* apply ``LISA_to_SSB``:
    phentax + :class:`ResponseWrapper` consume ecliptic-frame
    ``(lam, beta)`` directly (``is_ecliptic_latitude=True``). We also
    drop the ``gpc_to_mpc`` step — :class:`IMRPhenomTHMWaveform`
    converts ``dist`` from Gpc -> Mpc inside its own ``__call__``.
    """
    basis = [
        "logM", "q", "s1z", "s2z", "dist", "phi_ref",
        "cos_iota", "psi", "lam", "sin_beta", "t_plunge",
    ]
    return TransformContainer(
        input_basis=basis,
        output_basis=basis,
        parameter_transforms={
            "logM": np.exp,
            ("logM", "q"): mT_q,           # (M_total, q) -> (m1, m2)
            "cos_iota": np.arccos,         # cos_iota -> inc
            "sin_beta": np.arcsin,         # sin_beta -> beta
        },
        fill_dict={},
    )


def get_mbh_phentax_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    """Build the MBH :class:`MBHSetup` for the CPU phentax smoke test."""
    # ``initialize_kwargs`` is consumed by ``MBHSetup`` only to track
    # metadata for the run; the active waveform path is the cached
    # generator in ``setup_recipe``.
    initialize_kwargs_mbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        mbh_waveform="phentax.IMRPhenomTHM",
        mbh_waveform_kwargs=dict(higher_modes="all", force_backend="cpu"),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend="cpu",
            remove_garbage="zero",
        ),
    )

    waveform_kwargs_pe = dict()

    delta_prior = 1e-2
    # TODO: adjust this to regular transform container indexing
    breakpoint()
    injection_sampling = mbh_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)

    # Tight box around the injection in the sampling basis. Angles
    # (phi_ref, psi, lam) use additive deltas to avoid issues around
    # 0; cos_iota / sin_beta use absolute deltas clamped to [-1, 1].
    logM_inj = injection_sampling[0]
    q_inj = injection_sampling[1]
    s1z_inj = injection_sampling[2]
    s2z_inj = injection_sampling[3]
    dist_inj = injection_sampling[4]
    phi_ref_inj = injection_sampling[5]
    cos_iota_inj = injection_sampling[6]
    psi_inj = injection_sampling[7]
    lam_inj = injection_sampling[8]
    sin_beta_inj = injection_sampling[9]
    t_plunge_inj = injection_sampling[10]

    priors_mbh = {
        "logM":     uniform_dist((1 - delta_prior) * logM_inj,
                                 (1 + delta_prior) * logM_inj),
        "q":        uniform_dist(max(0.01, (1 - delta_prior) * q_inj),
                                 min(0.999, (1 + delta_prior) * q_inj)),
        "s1z":      uniform_dist(max(-0.99, s1z_inj - delta_prior),
                                 min(0.99, s1z_inj + delta_prior)),
        "s2z":      uniform_dist(max(-0.99, s2z_inj - delta_prior),
                                 min(0.99, s2z_inj + delta_prior)),
        "dist":     uniform_dist((1 - delta_prior) * dist_inj,
                                 (1 + delta_prior) * dist_inj),
        "phi_ref":  uniform_dist(phi_ref_inj - delta_prior * np.pi,
                                 phi_ref_inj + delta_prior * np.pi),
        "cos_iota": uniform_dist(max(-1.0 + 1e-6, cos_iota_inj - delta_prior),
                                 min(1.0 - 1e-6, cos_iota_inj + delta_prior)),
        "psi":      uniform_dist(psi_inj - delta_prior * np.pi,
                                 psi_inj + delta_prior * np.pi),
        "lam":      uniform_dist(lam_inj - delta_prior * np.pi,
                                 lam_inj + delta_prior * np.pi),
        "sin_beta": uniform_dist(max(-1.0 + 1e-6, sin_beta_inj - delta_prior),
                                 min(1.0 - 1e-6, sin_beta_inj + delta_prior)),
        "t_plunge": uniform_dist(t_plunge_inj - 60.0,    # +/- 1 min around injection
                                 t_plunge_inj + 60.0),
    }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    inner_moves = [(StretchMove(), 1.0)]

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_mbh,
        waveform_kwargs=waveform_kwargs_pe,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11,
        transform=_build_mbh_phentax_transform(),
        priors=priors,
        # Angles where periodicity matters in the sampling basis:
        periodic={"mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}},
        log_dir=general_set.file_store_dir,
    )

    return MBHSetup(mbh_settings)


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "mbh_phentax_only_smoke_test"
    file_store_dir = "./gf_output/"

    nwalkers = 4
    ntemps = 2

    domain_settings = DOMAIN_CHOICE

    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t_start=T_START,
        injection_params_full_basis=INJECTION_PARAMS_FULL_BASIS,
        tdi_chan="XYZ",
        nchannels=3,
        higher_modes="all",
    )

    # CompositeSensitivityBackend is the default; only ``tdi_generation``
    # is consumed.
    sensitivity_init_kwargs = dict(tdi_generation=2)

    # Smoke test: no Tukey taper.
    window_taper_duration = 0.0

    # Skip the engine's default highpass + 200 h edge-trim + Tobs trim.
    # The synthesised signal already covers exactly ``Tobs = Nf*Nt*dt``;
    # the default ``Tobs`` trim uses ``T=(N-1)*dt`` which would lose one
    # sample and break the WDM ``Nf*Nt`` shape.
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        main_file_key="testing",
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type="tukey",
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=None,
        data_processor=SyntheticMBHProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    return GeneralSetup(general_settings)


def get_global_fit_settings(copy_settings_file=False):
    general_setup = get_general_erebor_settings()
    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    rank_info = RankInfo(head_rank=1, main_rank=0)

    mbh_setup = get_mbh_phentax_erebor_settings(general_setup)

    global_settings = GlobalFitSettings(
        source_info={"mbh": mbh_setup},
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    return CurrentInfoGlobalFit(global_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("MBH (phentax) smoke-test settings constructed OK")
