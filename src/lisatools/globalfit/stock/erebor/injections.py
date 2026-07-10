"""Synthetic injections and data processors for the Erebor family.

Stock injection parameter tables, per-class injection-row makers, synthetic
in-process data processors (single-source and combined), and the
annual-modulated galactic-foreground machinery. Fully parameterized — the
annual-modulation callables are picklable classes rather than functions
reading module globals — so stock variants and settings files bind their
own configuration explicitly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from lisatools.domains import FDSettings, TDSettings, place_td_signal_on_grid
from lisatools.globalfit.preprocessing import BaseProcessingStep, L1ProcessingStep
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    generate_correlated_instrument_noise_td,
)
from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground
from lisatools.utils.constants import YRSID_SI

from .transforms import (
    make_emri_transform_container,
    make_mbh_transform_container,
    make_sobbh_transform_container,
)
from .wrappers import (
    get_emri_response_wrapper,
    get_mbh_phenom_wave_gen,
    get_sobbh_response_wrapper,
)

# ============================================================
# *** EMRI injection (synthetic, in-process) ***
# ============================================================
# FEW's ``FastKerrEccentricEquatorialFlux`` consumes 14 params in this order:
# M, mu, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0.
# EMRISetup's transform fills xI0 (idx 5) and Phi_theta0 (idx 12) so the
# sampling basis has 12 params with ``M -> logM, qS -> cos qS, qK -> cos qK``.
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        1.0e6,      # M
        1.0e1,      # mu
        0.5,        # a
        10.0,       # p0
        0.3,        # e0
        1.0,        # xI0 — fill_value[0]
        1.0,        # dist (Gpc)
        np.pi / 3,  # qS
        1.0,        # phiS
        np.pi / 4,  # qK
        2.0,        # phiK
        0.0,        # Phi_phi0
        0.0,        # Phi_theta0 — fill_value[1]
        0.0,        # Phi_r0
    ]
)
SAMPLE_FILL_INDICES = [5, 12]


def emri_full_to_sampling(params_full):
    """Convert a 14-param waveform-basis vector to the 12-param sampling basis."""
    p = np.asarray(params_full, dtype=float)
    transform = make_emri_transform_container(p[SAMPLE_FILL_INDICES])
    return transform.both_inverse_transforms(p)


# ============================================================
# *** SOBBH injection (synthetic, in-process) ***
# ============================================================
# Output-basis order consumed by :class:`SOBBHWaveform.__call__` (11 params):
#   m1 (M_sun), m2 (M_sun), s1, s2, dist (Gpc), inc (rad), f_low (Hz),
#   lam (ecliptic longitude, rad), beta (ecliptic latitude, rad),
#   psi (polarization, rad), phi0 (coalescence-phase offset, rad).
#
# Sampling basis (consumed by ``SOBBHSetup.transform``, 11 params):
#   logm1, logm2, s1, s2, dist, cosinc, f_low, phiS, cosqS, psi, phi0
# with cosqS -> beta = pi/2 - arccos(cosqS) and phiS -> lam.
SOBBH_INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        40.0,        # m1 (M_sun)
        30.0,        # m2 (M_sun)
        0.3,         # s1
        -0.2,        # s2
        1.0,         # dist (Gpc)
        np.pi / 3,   # inc
        1.5e-2,      # f_low (Hz)
        1.0,         # lam   (phiS in sampling basis)
        np.pi / 4,   # beta  (cosqS = cos(pi/2 - beta) in sampling basis)
        0.3,         # psi
        0.0,         # phi0
    ]
)


def sobbh_full_to_sampling(params_full):
    """Convert an 11-param SOBBH waveform-basis vector to the 11-param sampling basis."""
    transform = make_sobbh_transform_container()
    return transform.both_inverse_transforms(np.asarray(params_full, dtype=float))



# ============================================================
# *** Synthetic in-process data processors ***
# ============================================================
#
# Each one matches the SangriaProcessingStep / BaseProcessingStep
# interface: it generates the source's signal in-process and hands
# ``(times, data, fs)`` to :class:`BaseProcessingStep`. The engine's
# downstream code (window / domain transform / sensitivity) doesn't
# know or care that the data is synthetic.
#
# To inject multiple branches simultaneously, see
# :class:`SyntheticCombinedProcessingStep` below — it forwards the call
# to a list of single-branch processors and adds their TD outputs.


class SyntheticEMRIProcessingStep(BaseProcessingStep):
    """Generate the EMRI injection in-process via the shared ResponseWrapper.

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
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
        wave_gen = get_emri_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis)
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

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


class SyntheticSOBBHProcessingStep(BaseProcessingStep):
    """Generate the SOBBH injection in-process via the shared ResponseWrapper.

    Mirrors :class:`SyntheticEMRIProcessingStep` — pads/clips to exactly
    ``Tobs/dt`` so downstream WDM ``N = Nf*Nt`` matches.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params_full_basis: np.ndarray,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
        wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis)
        )
        td_signal = np.atleast_2d(td_signal)[:nchannels]

        target_N = int(round(Tobs / dt))
        if td_signal.shape[-1] < target_N:
            pad = target_N - td_signal.shape[-1]
            td_signal = np.pad(td_signal, ((0, 0), (0, pad)), mode="constant")
        elif td_signal.shape[-1] > target_N:
            td_signal = td_signal[:, :target_N]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


# ----------------------------------------------------------------------
# Synthetic GB injection (matches the SyntheticEMRIProcessingStep pattern).
# ----------------------------------------------------------------------
#
# Param layout for ``GBGPU.generate_global_template``:
#   shape ``(num_sources, 9)``: [A, f0, fdot, fddot, phi0, iota, psi, lam, beta]
#
# Two example sources in band — mid-mHz, modest fdot. Edit / extend.
GB_INJECTION_PARAMS = np.array(
    [
        [1.0e-22, 2.0e-3,  1.0e-16, 0.0, 0.0, 0.6, 0.4, 1.5, 0.3],
        [3.0e-23, 8.5e-3,  5.0e-15, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6],
    ]
)


class SyntheticGBProcessingStep(BaseProcessingStep):
    """In-process synthetic GB injection.

    Mirrors :class:`SyntheticEMRIProcessingStep`: fills a FD buffer with
    :meth:`GBGPU.generate_global_template`, IRFFTs back to TD, and hands
    ``(times, data, fs)`` to :class:`BaseProcessingStep`.

    Args:
        Tobs: Observation time in seconds.
        dt: Time-domain sample step in seconds.
        t_start: Initial time of the first sample.
        injection_params: ``(num_sources, 9)`` array of GB parameters in
            the order expected by ``GBGPU.generate_global_template``:
            ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``.
        tdi_chan: ``"XYZ"`` (3 channels) or ``"AE"`` (2 channels).
        nchannels: Number of leading channels to keep.
        use_tdi2: Use TDI 2nd generation kernel. Defaults to ``True``.
        oversample: GB-internal oversampling factor (1 is plenty for an
            injection).
        force_backend: ``"cpu"`` / ``"cuda12x"`` etc. — forwarded to GBGPU.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params: np.ndarray = GB_INJECTION_PARAMS,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        use_tdi2: bool = True,
        oversample: int = 1,
        force_backend: str = "cpu",
        orbits=None,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        from gbgpu.gbgpu import GBGPU

        from lisatools.detector import EqualArmlengthOrbits

        params = np.atleast_2d(injection_params).astype(np.float64)
        assert params.shape[1] == 9, (
            f"injection_params must have shape (num_sources, 9); got {params.shape}"
        )
        num_sources = params.shape[0]

        nchannels_full = 3 if tdi_chan == "XYZ" else 2
        if nchannels > nchannels_full:
            raise ValueError(
                f"nchannels={nchannels} exceeds tdi_chan={tdi_chan}'s {nchannels_full} channels"
            )

        target_N = int(round(Tobs / dt))
        data_length = target_N // 2 + 1

        # GBGPU requires an orbits object (it carries the phase-reference
        # t0); default to the stock analytic orbits, consistent with the
        # ecliptic lam/beta injection basis.
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=force_backend)
        gb = GBGPU(force_backend=force_backend, orbits=orbits)
        # ``GBGPU.generate_global_template`` reads ``self.gpus`` directly,
        # which is only set elsewhere (e.g. in the move setup). For a
        # standalone injection we set it explicitly.
        gb.gpus = None if force_backend == "cpu" else [0]
        xp = gb.xp

        # NOTE: ``generate_global_template`` internally ``.flatten()``s any
        # 2D/3D buffer it receives, which returns a copy — the kernel
        # writes never propagate back to our caller. The ndim==1 path does
        # not copy, so we pass a 1D buffer of length
        # ``num_templates * nchannels_full * data_length`` and reshape after.
        template_flat = xp.zeros(
            nchannels_full * data_length, dtype=xp.complex128
        )
        group_index = xp.zeros(num_sources, dtype=xp.int32)

        gb.generate_global_template(
            params,
            group_index,
            template_flat,
            start_freq_ind=0,
            T=Tobs,
            dt=dt,
            tdi_channel_setup=tdi_chan,
            tdi2=use_tdi2,
            oversample=oversample,
            data_length=data_length,
        )

        # FD -> TD. GBGPU stores tilde-h(f) in the LDC convention
        # (FD = dt * rfft(td)), so the inverse is td = irfft(FD)/dt.
        fd_signal = template_flat.reshape(nchannels_full, data_length)
        if hasattr(fd_signal, "get"):
            fd_signal = fd_signal.get()
        td_signal = np.fft.irfft(fd_signal, n=target_N, axis=-1) / dt
        td_signal = td_signal[:nchannels]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params = params
        self.tdi_chan = tdi_chan


class SyntheticCombinedProcessingStep(BaseProcessingStep):
    """Combine several synthetic processors into one residual.

    Each entry in ``processor_specs`` is a ``(class, init_kwargs)`` pair;
    each child generates its own TD signal at the same ``(Tobs, dt, t_start)``
    grid, and the combined output is the sum of all child TDs.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        processor_specs: list,  # list[(ProcessorClass, kwargs)]
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        if not processor_specs:
            raise ValueError("processor_specs must contain at least one (class, kwargs) entry")

        target_N = int(round(Tobs / dt))
        combined = np.zeros((nchannels, target_N), dtype=np.float64)
        for cls, kwargs in processor_specs:
            child_kwargs = dict(kwargs)
            child_kwargs.setdefault("Tobs", Tobs)
            child_kwargs.setdefault("dt", dt)
            child_kwargs.setdefault("t_start", t_start)
            child_kwargs.setdefault("nchannels", nchannels)
            child = cls(**child_kwargs, verbose=False, do_plots=False)
            combined[: child.data.shape[0]] += child.data[:, :target_N]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None


# ============================================================
# *** MBH phentax single-injection baseline ***
# ============================================================
# ``(m1, m2, s1z, s2z, dist [Gpc], phi_ref, inc, psi, lam, beta)`` — the
# waveform-basis row WITHOUT the run-dependent ``t_plunge`` (append one via
# :func:`mbh_phentax_injection_full_basis`).
MBH_PHENTAX_BASE_PARAMS = np.array(
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
    ]
)


def mbh_phentax_injection_full_basis(t_plunge: float) -> np.ndarray:
    """The stock MBH phentax injection row with ``t_plunge`` appended."""
    return np.concatenate([MBH_PHENTAX_BASE_PARAMS, [float(t_plunge)]])


def sobbh_catalogue_to_waveform_basis(row) -> np.ndarray:
    """Mojito SOBHB catalogue row -> 11-param waveform basis.

    ICRS run frame: sky + polarization are read raw (RA in the lam slot,
    Dec in the beta slot, psi ICRS); the orbits must be loaded with
    ``frame='icrs'`` to match. ``phi0`` is the catalogue ``TrueAnomaly``.
    """
    return np.array(
        [
            row["PrimaryMassSSBFrame"],
            row["SecondaryMassSSBFrame"],
            row["PrimarySpinCompZ"],
            row["SecondarySpinCompZ"],
            row["LuminosityDistance"] / 1e3,  # Mpc -> Gpc
            row["InclinationAngle"],
            row["GW22FrequencySSBFrame"],
            row["RightAscension"] % (2 * np.pi),
            row["Declination"],
            row["PolarisationAngle"] % np.pi,
            row["TrueAnomaly"],
        ]
    )


# ============================================================
# *** Annual modulation for the galactic-foreground noise ***
# ============================================================


class AnnualAmplitudeEnvelope:
    """Picklable per-sample amplitude envelope ``1 + A*cos(2*pi*t/yr + phi0)``."""

    def __init__(self, amp: float = 0.10, phase0: float = 0.0):
        self.amp = amp
        self.phase0 = phase0

    def __call__(self, t_arr: np.ndarray) -> np.ndarray:
        return 1.0 + self.amp * np.cos(
            2.0 * np.pi * np.asarray(t_arr) / YRSID_SI + self.phase0
        )


class AnnualCovarianceModulation:
    """``(nch, nch, Nt)`` modulation matrix for :class:`GalacticForeground`.

    Uses the standard isotropic-foreground per-element pattern (diag = 1,
    off-diag = -1/2) with an overall annual amplitude envelope applied
    uniformly across elements. The envelope is the square of the TD
    amplitude envelope because the modulation acts on the covariance
    (power), not the realization amplitude.
    """

    def __init__(self, amp: float = 0.10, phase0: float = 0.0):
        self.envelope = AnnualAmplitudeEnvelope(amp, phase0)

    def __call__(self, t_arr):
        t_arr = np.asarray(t_arr)
        base = np.array([
            [ 1.0, -0.5, -0.5],
            [-0.5,  1.0, -0.5],
            [-0.5, -0.5,  1.0],
        ])
        env = self.envelope(t_arr) ** 2  # power-domain envelope
        return base[:, :, None] * env[None, None, :]


class AnnualModulatedGalacticForeground(GalacticForeground):
    """:class:`GalacticForeground` with an annual modulation pre-bound.

    ``foreground_params=None`` uses the ``FittedHyperbolicTangent`` default
    tabulated values; a placeholder ``(tobs,)`` param vector satisfies the
    ``SeparableComponent`` contract (the fitted class reads ``Tobs`` at call
    time), so ``tobs`` is required in that case.
    """

    def __init__(
        self,
        foreground_params: Optional[Sequence[float]] = None,
        tdi_generation: int = 2,
        tobs: Optional[float] = None,
        annual_amp: float = 0.10,
        annual_phase0: float = 0.0,
    ):
        if foreground_params is None:
            if tobs is None:
                raise ValueError(
                    "AnnualModulatedGalacticForeground needs tobs when "
                    "foreground_params is None."
                )
            foreground_params = (tobs,)
        super().__init__(
            foreground_params=foreground_params,
            modulation=AnnualCovarianceModulation(annual_amp, annual_phase0),
            tdi_generation=tdi_generation,
            stochastic_fn=FittedHyperbolicTangentGalacticForeground,
        )


# ============================================================
# *** Synthetic foreground TD generators ***
# ============================================================


def _generate_foreground_fd_only_covariance(
    Nf_rfft: int,
    df: float,
    Tobs: float,
    foreground_params: Optional[Sequence[float]],
    tdi_generation: int,
) -> np.ndarray:
    """Build a 3x3xNf_rfft FD covariance for the galactic foreground only
    (no instrument), using the same path as the model-side
    :class:`GalacticForeground` so the synthetic realization matches the
    likelihood model.
    """
    fd_settings = FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    fg_params = (
        foreground_params
        if foreground_params is not None
        else (Tobs,)
    )
    fg_component = GalacticForeground(
        foreground_params=fg_params,
        modulation=None,  # stationary base — the annual envelope acts in TD
        tdi_generation=tdi_generation,
        stochastic_fn=FittedHyperbolicTangentGalacticForeground,
    )
    sens = CompositeSensitivityMatrix(fd_settings, [fg_component])
    return np.asarray(sens.sens_mat)  # (3, 3, Nf_rfft)


def generate_modulated_foreground_td(
    N: int,
    dt: float,
    Tobs: float,
    foreground_params: Optional[Sequence[float]],
    tdi_generation: int,
    seed: int,
    annual_amp: float = 0.10,
    annual_phase0: float = 0.0,
) -> np.ndarray:
    """Sample a 3-channel TD foreground realization with the annual envelope.

    Workflow: build the stationary 3x3 FD covariance, per-frequency
    Cholesky, draw complex Gaussian, IRFFT to TD, then multiply each
    channel by ``sqrt(envelope(t))`` so the realization is non-stationary
    in the same way the model covariance is.
    """
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    cov = _generate_foreground_fd_only_covariance(
        Nf_rfft, df, Tobs, foreground_params, tdi_generation
    )

    rng = np.random.default_rng(seed)
    norm = 0.5 * (1.0 / df) ** 0.5
    z = rng.normal(0, norm, (3, Nf_rfft)) + 1j * rng.normal(0, norm, (3, Nf_rfft))

    n_fd = np.zeros_like(z)
    eye = np.eye(3) * 1e-60
    for k in range(Nf_rfft):
        C_k = cov[..., k] + eye
        try:
            L = np.linalg.cholesky(C_k)
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(cov[..., k]), 0.0)
            n_fd[:, k] = np.sqrt(diag) * z[:, k]
            continue
        n_fd[:, k] = L @ z[:, k]

    n_td = (np.fft.irfft(n_fd, n=N, axis=-1) / dt).astype(np.float64)

    # Apply per-sample annual envelope (amplitude domain).
    t_arr = np.arange(N) * dt  # mojito processor adds its own t0 separately
    env = np.sqrt(AnnualAmplitudeEnvelope(annual_amp, annual_phase0)(t_arr))
    n_td *= env[None, :]
    return n_td


# ============================================================
# *** Per-class synthetic injection-row makers ***
# ============================================================


def make_emri_injections(n: int, base: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``(n, 14)`` EMRI waveform-basis injection vectors.

    Intrinsic params shared (so a tight per-leaf prior is feasible);
    sky / phase / distance vary per source (fixed seed).
    """
    base = (INJECTION_PARAMS_FULL_BASIS if base is None else np.asarray(base)).copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        row = base.copy()
        row[6]  = 1.0 + 2.0 * rng.uniform()           # dist (Gpc)
        row[7]  = rng.uniform(0.05, np.pi - 0.05)     # qS
        row[8]  = rng.uniform(0.0, 2.0 * np.pi)       # phiS
        row[9]  = rng.uniform(0.05, np.pi - 0.05)     # qK
        row[10] = rng.uniform(0.0, 2.0 * np.pi)       # phiK
        row[11] = rng.uniform(0.0, 2.0 * np.pi)       # Phi_phi0
        rows.append(row)
    return np.stack(rows, axis=0)


def make_sobbh_injections(n: int, base: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``(n, 11)`` SOBBH waveform-basis injection vectors (fixed seed)."""
    base = (
        SOBBH_INJECTION_PARAMS_FULL_BASIS if base is None else np.asarray(base)
    ).copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    rng = np.random.default_rng(22)
    rows = []
    for i in range(n):
        row = base.copy()
        row[4]  = 1.0 + 1.0 * rng.uniform()           # dist
        row[5]  = rng.uniform(0.05, np.pi - 0.05)     # inc
        row[7]  = rng.uniform(0.0, 2.0 * np.pi)       # lam
        row[8]  = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)  # beta
        row[9]  = rng.uniform(0.0, np.pi)             # psi
        row[10] = rng.uniform(0.0, 2.0 * np.pi)       # phi0
        rows.append(row)
    return np.stack(rows, axis=0)


def make_mbh_injections(
    n: int, tobs: float, base: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return ``(n, 11)`` MBH sampling-basis injection vectors (fixed seed).

    Basis: ``(logM, Q, s1z, s2z, dist [Gpc], phi_ref, cos_iota, psi,
    alpha, sin_delta, t_plunge)``. Masses and spins come from the shared
    phentax baseline; sky / phase / distance / merger time vary per source
    with merger times spread across the interior of the observation.
    """
    base = (MBH_PHENTAX_BASE_PARAMS if base is None else np.asarray(base)).copy()
    if n == 0:
        return np.zeros((0, 11), dtype=float)
    m1, m2 = float(base[0]), float(base[1])
    rng = np.random.default_rng(33)
    rows = []
    for i in range(n):
        rows.append(np.array([
            np.log(m1 + m2),                               # logM
            m1 / m2,                                       # Q (>= 1)
            base[2],                                       # s1z
            base[3],                                       # s2z
            10.0 + 15.0 * rng.uniform(),                   # dist (Gpc)
            rng.uniform(0.0, 2.0 * np.pi),                 # phi_ref
            np.cos(rng.uniform(0.1, np.pi - 0.1)),         # cos_iota
            rng.uniform(0.0, np.pi),                       # psi
            rng.uniform(0.0, 2.0 * np.pi),                 # alpha
            np.sin(rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)),  # sin_delta
            (0.2 + 0.6 * (i + 1) / (n + 1)) * tobs,        # t_plunge
        ]))
    return np.stack(rows, axis=0)


# ============================================================
# *** Multi-source synthetic streams + data processors ***
# ============================================================


def build_synthetic_source_streams(
    Tobs: float,
    dt: float,
    t_start: float,
    target_N: int,
    nchannels: int,
    force_backend: str,
    emri_injections: np.ndarray,
    sobbh_injections: np.ndarray,
    mbh_injections: np.ndarray,
    tdi_chan: str = "XYZ",
    tdi_gen_str: str = "2nd generation",
    sobbh_reference_time: Optional[float] = None,
    mbh_wave_gen=None,
    mbh_transform=None,
) -> tuple:
    """Build per-class TD signal sums via the cached response wrappers.

    ``sobbh_reference_time`` is the epoch ``f_low`` is defined at (``None``
    -> the window start; mojito mode passes ``MOJITO_REFERENCE_TIME``).
    ``mbh_wave_gen`` (a ``PhenomTHMTDIWaveform``) and ``mbh_transform`` are
    required only when ``mbh_injections`` is non-empty.
    """
    tdi_config = TDIConfig(tdi_gen_str, force_backend=force_backend)
    zero = np.zeros((nchannels, target_N), dtype=np.float64)
    # Stock full-grid placement target (right-pad / clip onto the data grid).
    grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")

    emri_td = zero.copy()
    if emri_injections.shape[0] > 0:
        emri_wave_gen = get_emri_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=tdi_chan,
            role="injection", force_backend=force_backend,
        )
        for ii, params in enumerate(emri_injections):
            print(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [start]")
            sig = np.asarray(emri_wave_gen(*params))
            emri_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            print(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [end]")

    sobbh_td = zero.copy()
    if sobbh_injections.shape[0] > 0:
        sobbh_wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=tdi_chan,
            role="injection", force_backend=force_backend,
            reference_time=sobbh_reference_time,
        )
        for ii, params in enumerate(sobbh_injections):
            print(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [start]")
            sig = np.asarray(sobbh_wave_gen(*params))
            sobbh_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            print(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [end]")

    mbh_td = zero.copy()
    if mbh_injections.shape[0] > 0:
        # PhenomTHMTDIWaveform path: legacy pyResponseTDI response on the
        # dense phentax grid, then stock full-grid placement. Same generator
        # class as the PE templates so the engine-side residual rebuild
        # nulls the injections exactly.
        if mbh_wave_gen is None or mbh_transform is None:
            raise ValueError(
                "MBH injections need mbh_wave_gen and mbh_transform "
                "(see get_mbh_phenom_wave_gen / make_mbh_transform_container)."
            )
        for ii, params in enumerate(mbh_injections):
            print(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [start]")
            params_in = mbh_transform.both_transforms(
                np.asarray(params, dtype=float)
            )
            times, channels = mbh_wave_gen.compute_tdi_channels(*params_in)
            mbh_td += np.asarray(
                place_td_signal_on_grid(channels, grid, times=times).arr
            )[:nchannels]
            print(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [end]")
    return emri_td, sobbh_td, mbh_td


class L1ProcessingStepWithSyntheticNoise(L1ProcessingStep):
    """Mojito L1 source loader + synthetic FD noise + modulated foreground.

    Loads MBHB / EMRI / SOBHB source TD signals from a mojito L1 folder
    (no NOISE, no GB / VGB), then optionally adds:
      1. Synthetic FD-correlated instrument noise drawn from the
         ``(noise_soms_d, noise_sa_a)`` covariance.
      2. A galactic-foreground TD realization with the annual amplitude
         envelope applied per-sample.

    The mojito catalog populated by the base loader on ``self.catalogue``
    is preserved for downstream factories.
    """

    def __init__(
        self,
        L1_folder: str,
        source_ids: dict,
        orbits_class=None,
        orbits_kwargs: Optional[dict] = None,
        verbose: bool = True,
        do_plots: bool = False,
        Tobs: float = None,
        window_start_offset: float = 0.0,
        add_instrument_noise: bool = False,
        noise_soms_d: float = 15e-12,
        noise_sa_a: float = 3e-15,
        noise_seed: int = 12345,
        noise_model_name: str = "full_year_noise_model",
        add_galactic_foreground: bool = False,
        foreground_params: Optional[Sequence[float]] = None,
        foreground_seed: int = 67890,
        annual_amp: float = 0.10,
        annual_phase0: float = 0.0,
        tdi_generation: int = 2,
    ):
        # Drop any class whose source_ids list is empty (mojito's
        # L1DataLoader raises on missing IDs). GB / VGB load the whole
        # galaxy file (ids just gate inclusion). Instrument noise + galactic
        # foreground come from the synthetic generators, not from mojito.
        source_types = [
            t for t in ["GB", "VGB", "MBHB", "EMRI", "SOBHB"] if source_ids.get(t)
        ]
        if orbits_class is None:
            from lisatools.detector import L1Orbits
            orbits_class = L1Orbits

        super().__init__(
            L1_folder=L1_folder,
            source_types=source_types,
            source_ids=source_ids,
            orbits_class=orbits_class,
            orbits_kwargs=orbits_kwargs,
            verbose=verbose,
            do_plots=do_plots,
            Tobs=Tobs,
            window_start_offset=window_start_offset,
        )
        # super().__init__ already called load_data() and stored .data,
        # .times, .fs, .orbits, .catalogue.

        # Add synthetic FD instrument noise + modulated foreground.
        # Stock full-grid placement (right-pad / clip onto the data grid).
        N = int(round(self.T / self.dt))
        nch = self.data.shape[0]
        grid = TDSettings(
            N=N, dt=self.dt, t0=float(self.times[0]), force_backend="cpu"
        )
        combined = np.asarray(
            place_td_signal_on_grid(self.data[:nch], grid).arr
        )
        if add_instrument_noise:
            noise_td = generate_correlated_instrument_noise_td(
                N=N, dt=self.dt,
                Soms_d=noise_soms_d, Sa_a=noise_sa_a,
                tdi_generation=tdi_generation, seed=noise_seed,
                model_name=noise_model_name,
            )
            combined = combined + np.asarray(
                place_td_signal_on_grid(noise_td[:nch], grid).arr
            )
        if add_galactic_foreground:
            fg_td = generate_modulated_foreground_td(
                N=N, dt=self.dt, Tobs=self.T,
                foreground_params=foreground_params,
                tdi_generation=tdi_generation, seed=foreground_seed,
                annual_amp=annual_amp, annual_phase0=annual_phase0,
            )
            combined = combined + np.asarray(
                place_td_signal_on_grid(fg_td[:nch], grid).arr
            )
        self.data = combined


class SyntheticDataProcessor(BaseProcessingStep):
    """All-synthetic processor used for testing without a mojito L1 folder.

    Builds per-class source TD signals via the cached response wrappers
    and sums the same instrument-noise + modulated-foreground TD streams
    as :class:`L1ProcessingStepWithSyntheticNoise`. Exposes a
    ``catalogue`` dict in the same shape so downstream factories can
    read leaf counts uniformly.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        emri_injection_params_full_basis: np.ndarray,
        sobbh_injection_params_full_basis: np.ndarray,
        mbh_injection_params_sampling_basis: np.ndarray,
        source_ids: Optional[dict] = None,
        nchannels: int = 3,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
        tdi_chan: str = "XYZ",
        tdi_gen_str: str = "2nd generation",
        sobbh_reference_time: Optional[float] = None,
        mbh_phenom_kwargs: Optional[dict] = None,
        add_instrument_noise: bool = False,
        noise_soms_d: float = 15e-12,
        noise_sa_a: float = 3e-15,
        noise_seed: int = 12345,
        noise_model_name: str = "full_year_noise_model",
        add_galactic_foreground: bool = False,
        foreground_params: Optional[Sequence[float]] = None,
        foreground_seed: int = 67890,
        annual_amp: float = 0.10,
        annual_phase0: float = 0.0,
        tdi_generation: int = 2,
    ):
        target_N = int(round(Tobs / dt))
        emri = np.atleast_2d(emri_injection_params_full_basis)
        sobbh = np.atleast_2d(sobbh_injection_params_full_basis)
        # MBH rows arrive in the sampling basis; build_synthetic_source_streams
        # applies the stock MBH transform per row.
        mbh = np.atleast_2d(mbh_injection_params_sampling_basis)

        mbh_wave_gen = None
        mbh_transform = None
        if mbh.shape[0] > 0:
            grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")
            mbh_wave_gen = get_mbh_phenom_wave_gen(
                data_td_settings=grid,
                waveform_t0=t_start,
                dt=dt,
                orbits=None,  # stock analytic orbits (synthetic mode)
                output_domain_settings=None,  # TD output only here
                force_backend=force_backend,
                data_span=Tobs,
                tdi_gen_str=tdi_gen_str,
                tdi_chan=tdi_chan,
                **(mbh_phenom_kwargs or {}),
            )
            mbh_transform = make_mbh_transform_container()

        emri_td, sobbh_td, mbh_td = build_synthetic_source_streams(
            Tobs=Tobs, dt=dt, t_start=t_start, target_N=target_N,
            nchannels=nchannels, force_backend=force_backend,
            emri_injections=emri,
            sobbh_injections=sobbh,
            mbh_injections=mbh,
            tdi_chan=tdi_chan,
            tdi_gen_str=tdi_gen_str,
            sobbh_reference_time=sobbh_reference_time,
            mbh_wave_gen=mbh_wave_gen,
            mbh_transform=mbh_transform,
        )

        combined = emri_td + sobbh_td + mbh_td
        if add_instrument_noise:
            noise_td = generate_correlated_instrument_noise_td(
                N=target_N, dt=dt,
                Soms_d=noise_soms_d, Sa_a=noise_sa_a,
                tdi_generation=tdi_generation, seed=noise_seed,
                model_name=noise_model_name,
            )[:nchannels]
            combined = combined + noise_td
        if add_galactic_foreground:
            fg_td = generate_modulated_foreground_td(
                N=target_N, dt=dt, Tobs=Tobs,
                foreground_params=foreground_params,
                tdi_generation=tdi_generation, seed=foreground_seed,
                annual_amp=annual_amp, annual_phase0=annual_phase0,
            )[:nchannels]
            combined = combined + fg_td
        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots,
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        # Synthetic-mode catalogue mirrors the mojito shape so downstream
        # factories can use a uniform `len(general_info.catalogue[<CLASS>])`.
        if source_ids is None:
            source_ids = {
                "MBHB": list(range(mbh.shape[0])),
                "EMRI": list(range(emri.shape[0])),
                "SOBHB": list(range(sobbh.shape[0])),
            }
        self.catalogue = {
            "MBHB": {sid: {} for sid in source_ids.get("MBHB", [])},
            "EMRI": {sid: {} for sid in source_ids.get("EMRI", [])},
            "SOBHB": {sid: {} for sid in source_ids.get("SOBHB", [])},
        }
        # Stash truths so plot/diagnostic code can find them.
        self.emri_injection_params_full_basis = emri
        self.sobbh_injection_params_full_basis = sobbh
        self.mbh_injection_params_sampling_basis = mbh
