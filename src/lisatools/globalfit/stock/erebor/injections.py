"""Synthetic injections and data processors for the Erebor family.

Stock injection parameter tables, per-class injection-row makers, synthetic
in-process data processors (single-source and combined), and the
annual-modulated galactic-foreground machinery. Fully parameterized — the
annual-modulation callables are picklable classes rather than functions
reading module globals — so stock variants and settings files bind their
own configuration explicitly.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from lisatools.domains import (
    FDSettings,
    TDSettings,
    TDSignal,
    WDMSettings,
    WDMSignal,
    place_td_signal_on_grid,
)
from lisatools.globalfit.preprocessing import BaseProcessingStep, L1ProcessingStep
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sensitivity import (
    CompositeSensitivityBackend,
    CompositeSensitivityMatrix,
    GalacticForeground,
    generate_correlated_instrument_noise_td,
)
from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import asnumpy, get_array_module

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

import logging

logger = logging.getLogger(__name__)

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

        # asnumpy: GPU-backend generators return device arrays (data
        # production follows the run backend); streams accumulate on host.
        td_signal = np.atleast_2d(
            asnumpy(wave_gen(*injection_params_full_basis))
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

        # asnumpy: GPU-backend generators return device arrays (data
        # production follows the run backend); streams accumulate on host.
        td_signal = np.atleast_2d(
            asnumpy(wave_gen(*injection_params_full_basis))
        )[:nchannels]

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

    Mirrors :class:`SyntheticEMRIProcessingStep`: generates the summed TD
    stream with :class:`~lisatools.response.tdionfly.GBTDIonTheFly` and
    hands ``(times, data, fs)`` to :class:`BaseProcessingStep`.

    ``GBTDIonTheFly`` is the SAME generator the analysis side templates
    with, so the synthetic stream and the template agree by construction
    and no phase-convention fixup is needed. It also requires none of the
    legacy ``SharedMemoryGBGPU.cu`` kernels, so this works against a GBGPU
    built with ``-DGBGPU_WITH_SHAREDMEM=OFF``.

    Args:
        Tobs: Observation time in seconds.
        dt: Time-domain sample step in seconds.
        t_start: Initial time of the first sample. Also the phase
            reference ``t_ref``, matching what the GB branch uses in
            synthetic mode.
        injection_params: ``(num_sources, 9)`` array of GB parameters,
            ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]`` — the
            ``GBTDIonTheFly`` call order.
        tdi_chan: ``"XYZ"`` (3 channels) or ``"AE"`` (2 channels).
        nchannels: Number of leading channels to keep.
        use_tdi2: Use TDI 2nd generation. Defaults to ``True``.
        oversample: Unused; kept for backwards compatibility with the
            legacy FD-kernel signature. The on-the-fly accuracy knob is
            ``GB_SYNTH_N_INJ`` (sparse amp/phase nodes, default 16384);
            ``GB_SYNTH_CHUNK`` caps how many sources are batched at once.
        force_backend: ``"cpu"`` / ``"cuda12x"`` etc.
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
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdionfly import GBTDIonTheFly

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

        # The generator needs an orbits object (it carries the
        # phase-reference t0); default to the stock analytic orbits,
        # consistent with the ecliptic lam/beta injection basis.
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=force_backend)

        # Generated by ``GBTDIonTheFly`` -- the SAME generator the analysis
        # side templates with -- directly in the time domain.
        #
        # This replaced the legacy ``GBGPU.generate_global_template`` FD
        # kernel (+ IRFFT) in 2026-08. Three things fall out of that:
        #
        #  * No phase-convention fixup. The legacy fastGB kernel carries
        #    e^{+i*phi0} and needed ``flip_ref_phase=True`` to match the
        #    analysis side's LDC/JaxGB e^{-i*phi0}; getting that wrong left
        #    every source dephased by 2*phi0 (measured 2026-07-23 as
        #    data-template overlap cos(2*phi0)). on-the-fly already IS the
        #    analysis convention -- ``eval_tdi`` evaluates
        #    Re[A e^{-i(phi + phi_ref)}] -- so there is nothing to flip.
        #  * SHOULD close TODO(synthetic-amp-evolution): the ~0.7%-weighted
        #    residual between the synthetic stream and the on-the-fly
        #    template (vgb 55-source initial ll ~ -82 vs ~0 on mojito) was
        #    attributed to generating the data with a DIFFERENT kernel than
        #    the one that analyses it (suspected amplitude evolution /
        #    windowing). Same kernel both sides removes that by
        #    construction. NOT YET MEASURED -- verify with a synthetic
        #    truth-null read (expect ~0) before relying on it.
        #  * Drops the last data-path dependency on the legacy
        #    SharedMemoryGBGPU.cu family, so a GBGPU built with
        #    -DGBGPU_WITH_SHAREDMEM=OFF can generate synthetic GB data.
        #
        # t_ref = t_start matches what the analysis branch uses in
        # synthetic mode (gb_no_fg.py: ``gb.t0 = ... else synthetic_t_start``).
        tdi_config = TDIConfig(
            "2nd generation" if use_tdi2 else "1st generation",
            force_backend=force_backend,
        )
        times = np.arange(target_N) * dt + t_start

        # on-the-fly evaluates amp/phase on a SPARSE grid and splines onto
        # the dense one; N_INJ is that sparse count (the accuracy knob).
        n_inj = int(os.environ.get("GB_SYNTH_N_INJ", "16384"))
        # Sources are batched, but a batch holds (chunk, nchannels, N)
        # doubles -- at Tobs/dt = O(1e6) that is ~25 MB per source per
        # channel, so cap the batch and accumulate.
        chunk = max(1, int(os.environ.get("GB_SYNTH_CHUNK", "16")))

        # on-the-fly ALWAYS emits 3 XYZ channels (TDIConfig.nchannels is 3
        # unconditionally; ``tdi_chan`` does not switch the kernel output,
        # unlike the legacy FD kernel's ``tdi_channel_setup``). So build in
        # XYZ and rotate afterwards when AE/AET was asked for.
        td_signal = np.zeros((3, target_N), dtype=np.float64)
        for lo in range(0, num_sources, chunk):
            sub = params[lo:lo + chunk]
            gb_gen = GBTDIonTheFly(
                np.linspace(times[0], times[-1], n_inj),
                Tobs, t_start, 1.0 / dt, sub.shape[0],
                tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                force_backend=force_backend,
            )
            # params columns are [A, f0, fdot, fddot, phi0, iota, psi,
            # lam, beta] -- exactly GBTDIonTheFly's call signature.
            # convert_to_ra_dec=False: the injection basis is ecliptic
            # lam/beta, matching the orbits above.
            spline = gb_gen(
                *sub.T, convert_to_ra_dec=False, return_spline=True
            )
            td_sub = spline.eval_tdi(gb_gen.xp.asarray(times))
            if hasattr(td_sub, "get"):
                td_sub = td_sub.get()
            td_signal += np.asarray(td_sub).sum(axis=0)

        if tdi_chan != "XYZ":
            from lisatools.utils.utility import AET

            A, E, T = AET(td_signal[0], td_signal[1], td_signal[2])
            td_signal = np.asarray([A, E, T])

        td_signal = td_signal[:nchannels]
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
        xp = get_array_module(t_arr)
        return 1.0 + self.amp * xp.cos(
            2.0 * np.pi * xp.asarray(t_arr) / YRSID_SI + self.phase0
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
        xp = get_array_module(t_arr)
        t_arr = xp.asarray(t_arr)
        base = xp.array([
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
        import warnings

        warnings.warn(
            "AnnualModulatedGalacticForeground is deprecated; use the general "
            "modulation framework directly: GalacticForeground(foreground_params, "
            "modulation=AnnualCovarianceModulation(annual_amp, annual_phase0), "
            "stochastic_fn=FittedHyperbolicTangentGalacticForeground) — or "
            "GalForTimeModulation(path) for a tabulated modulation.",
            DeprecationWarning,
            stacklevel=2,
        )
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
#
# Two modes, selected by ``general.synthetic_injections``:
#
# * ``mode="stock"`` (default) — the fixed stock tables above, extrinsics
#   varied with per-class FIXED seeds (11/22/33): byte-identical to the
#   historical behaviour, independent of ``seed``.
# * ``mode="prior"`` — every varied parameter is drawn with a seeded RNG
#   from (an interior region of) the source class's sampling priors.
#   Extrinsics use their full prior ranges (isotropic sky/orientation);
#   intrinsics are drawn from a documented interior region chosen so every
#   draw yields a valid, laptop-fast waveform (e.g. EMRI ``p0``/``e0``
#   combinations that FEW can integrate). The same ``(mode, seed, n)``
#   always reproduces the same rows — the data processor and the branch
#   preparation call these independently and MUST agree.

_PRIOR_DRAW_CLASS_OFFSET = {"EMRI": 1, "SOBHB": 2, "MBHB": 3, "GB": 4}
_DEFAULT_INJECTION_SEED = 1234


def _prior_rng(seed: Optional[int], cls: str) -> np.random.Generator:
    """Deterministic per-class RNG: one user seed, decorrelated across classes."""
    base = _DEFAULT_INJECTION_SEED if seed is None else int(seed)
    return np.random.default_rng([base, _PRIOR_DRAW_CLASS_OFFSET[cls]])


def make_emri_injections(
    n: int,
    base: Optional[np.ndarray] = None,
    *,
    mode: str = "stock",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return ``(n, 14)`` EMRI waveform-basis injection vectors.

    ``mode="stock"``: intrinsic params shared from the stock baseline (so a
    tight per-leaf prior is feasible); sky / phase / distance vary per
    source (fixed seed). ``mode="prior"``: intrinsics drawn from an interior
    region of the wide priors (M 5e5-2e6, mu 5-30, a 0-0.9, p0 9-13,
    e0 0.1-0.5 — guaranteed-valid FEW trajectories); extrinsics isotropic.
    """
    base = (INJECTION_PARAMS_FULL_BASIS if base is None else np.asarray(base)).copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    if mode == "prior":
        rng = _prior_rng(seed, "EMRI")
        rows = []
        for _ in range(n):
            row = base.copy()
            row[0]  = np.exp(rng.uniform(np.log(5.0e5), np.log(2.0e6)))  # M
            row[1]  = rng.uniform(5.0, 30.0)                             # mu
            row[2]  = rng.uniform(0.0, 0.9)                              # a
            row[3]  = rng.uniform(9.0, 13.0)                             # p0
            row[4]  = rng.uniform(0.1, 0.5)                              # e0
            row[6]  = rng.uniform(0.5, 3.0)                              # dist (Gpc)
            row[7]  = np.arccos(rng.uniform(-0.99, 0.99))                # qS
            row[8]  = rng.uniform(0.0, 2.0 * np.pi)                      # phiS
            row[9]  = np.arccos(rng.uniform(-0.99, 0.99))                # qK
            row[10] = rng.uniform(0.0, 2.0 * np.pi)                      # phiK
            row[11] = rng.uniform(0.0, 2.0 * np.pi)                      # Phi_phi0
            row[13] = rng.uniform(0.0, 2.0 * np.pi)                      # Phi_r0
            rows.append(row)
        return np.stack(rows, axis=0)
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


def make_sobbh_injections(
    n: int,
    base: Optional[np.ndarray] = None,
    *,
    mode: str = "stock",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return ``(n, 11)`` SOBBH waveform-basis injection vectors.

    ``mode="stock"``: the stock baseline with fixed-seed extrinsic scatter.
    ``mode="prior"``: masses/spins/``f_low`` from an interior region of the
    wide priors (m1 25-55, m2 15-m1, |s| <= 0.9, f_low 6-18 mHz — in-band,
    laptop-fast); extrinsics isotropic.
    """
    base = (
        SOBBH_INJECTION_PARAMS_FULL_BASIS if base is None else np.asarray(base)
    ).copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    if mode == "prior":
        rng = _prior_rng(seed, "SOBHB")
        rows = []
        for _ in range(n):
            row = base.copy()
            row[0]  = rng.uniform(25.0, 55.0)                            # m1
            row[1]  = rng.uniform(15.0, row[0])                          # m2 <= m1
            row[2]  = rng.uniform(-0.9, 0.9)                             # s1
            row[3]  = rng.uniform(-0.9, 0.9)                             # s2
            row[4]  = rng.uniform(0.5, 2.0)                              # dist (Gpc)
            row[5]  = np.arccos(rng.uniform(-0.99, 0.99))                # inc
            row[6]  = rng.uniform(6.0e-3, 1.8e-2)                        # f_low
            row[7]  = rng.uniform(0.0, 2.0 * np.pi)                      # lam
            row[8]  = np.arcsin(rng.uniform(-0.99, 0.99))                # beta
            row[9]  = rng.uniform(0.0, np.pi)                            # psi
            row[10] = rng.uniform(0.0, 2.0 * np.pi)                      # phi0
            rows.append(row)
        return np.stack(rows, axis=0)
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
    n: int,
    tobs: float,
    base: Optional[np.ndarray] = None,
    *,
    mode: str = "stock",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return ``(n, 11)`` MBH sampling-basis injection vectors.

    Basis: ``(logM, Q, s1z, s2z, dist [Gpc], phi_ref, cos_iota, psi,
    alpha, sin_delta, t_plunge)``. ``mode="stock"``: masses and spins from
    the shared phentax baseline; sky / phase / distance / merger time vary
    per source (fixed seed) with merger times spread across the interior of
    the observation. ``mode="prior"``: total mass log-uniform 8e5-4e6,
    Q log-uniform 1-8, |s| <= 0.9, dist 10-50 Gpc, isotropic sky/orientation,
    ``t_plunge`` uniform over the interior (0.25-0.85) of the observation.
    """
    base = (MBH_PHENTAX_BASE_PARAMS if base is None else np.asarray(base)).copy()
    if n == 0:
        return np.zeros((0, 11), dtype=float)
    if mode == "prior":
        rng = _prior_rng(seed, "MBHB")
        rows = []
        for _ in range(n):
            rows.append(np.array([
                rng.uniform(np.log(8.0e5), np.log(4.0e6)),           # logM
                np.exp(rng.uniform(0.0, np.log(8.0))),               # Q (>= 1)
                rng.uniform(-0.9, 0.9),                              # s1z
                rng.uniform(-0.9, 0.9),                              # s2z
                rng.uniform(10.0, 50.0),                             # dist (Gpc)
                rng.uniform(0.0, 2.0 * np.pi),                       # phi_ref
                rng.uniform(-0.99, 0.99),                            # cos_iota
                rng.uniform(0.0, np.pi),                             # psi
                rng.uniform(0.0, 2.0 * np.pi),                       # alpha
                rng.uniform(-0.99, 0.99),                            # sin_delta
                rng.uniform(0.25, 0.85) * tobs,                      # t_plunge
            ]))
        return np.stack(rows, axis=0)
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


def make_gb_injections(
    n: int,
    *,
    mode: str = "stock",
    seed: Optional[int] = None,
    band: tuple = (1.0e-3, 1.0e-2),
) -> np.ndarray:
    """Return ``(n, 9)`` GB rows ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``.

    ``mode="stock"``: rows cycled from :data:`GB_INJECTION_PARAMS` (small
    fixed-seed ``f0`` jitter beyond the table length keeps rows distinct).
    ``mode="prior"``: amplitude log-uniform 3e-23-3e-22, ``f0`` uniform in
    ``band``, ``fdot`` log-uniform 1e-17-1e-14, isotropic orientation/sky.
    """
    if n == 0:
        return np.zeros((0, 9), dtype=float)
    if mode == "prior":
        rng = _prior_rng(seed, "GB")
        f_lo, f_hi = float(band[0]), float(band[1])
        rows = []
        for _ in range(n):
            rows.append(np.array([
                np.exp(rng.uniform(np.log(3.0e-23), np.log(3.0e-22))),  # A
                rng.uniform(f_lo, f_hi),                                # f0
                np.exp(rng.uniform(np.log(1.0e-17), np.log(1.0e-14))),  # fdot
                0.0,                                                    # fddot
                rng.uniform(0.0, 2.0 * np.pi),                          # phi0
                np.arccos(rng.uniform(-0.99, 0.99)),                    # iota
                rng.uniform(0.0, np.pi),                                # psi
                rng.uniform(0.0, 2.0 * np.pi),                          # lam
                np.arcsin(rng.uniform(-0.99, 0.99)),                    # beta
            ]))
        return np.stack(rows, axis=0)
    table = np.atleast_2d(GB_INJECTION_PARAMS)
    rows = [table[i % len(table)].copy() for i in range(n)]
    if n > len(table):
        jitter_rng = np.random.default_rng(44)
        for i in range(len(table), n):
            rows[i][1] *= 1.0 + 1.0e-3 * jitter_rng.uniform(1.0, 5.0)
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
            logger.info(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [start]")
            # asnumpy: GPU-backend generator output -> host accumulation.
            sig = asnumpy(emri_wave_gen(*params))
            emri_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            logger.info(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [end]")

    sobbh_td = zero.copy()
    if sobbh_injections.shape[0] > 0:
        # TDI-on-the-fly injection generator (2026-07-30): the SAME response
        # family the PE templates use (analytic delays), ~70-200x faster than
        # the legacy pyResponseTDI path — whose Lagrange-interp output also
        # carried a ~3.4e3 s effective time-anchor offset vs the on-the-fly
        # output on the synthetic grid, corrupting truth-start residuals.
        from lisatools.sources.sobbh.response import get_sobbh_tdionfly_gen

        from .wrappers import SOBBHTDIonFlyWaveWrap

        sobbh_gen = get_sobbh_tdionfly_gen(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config,
            reference_time=sobbh_reference_time,
            force_backend=force_backend,
        )
        t_arr_sobbh = np.arange(target_N) * dt + t_start
        sobbh_wrap = SOBBHTDIonFlyWaveWrap(
            sobbh_gen, t_arr_sobbh, grid, None, nchannels=nchannels
        )
        for ii, params in enumerate(sobbh_injections):
            logger.info(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [start]")
            # asnumpy: GPU-backend generator output -> host accumulation.
            sig = asnumpy(sobbh_wrap.raw_td(*params))
            sobbh_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            logger.info(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [end]")

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
            logger.info(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [start]")
            params_in = mbh_transform.both_transforms(
                np.asarray(params, dtype=float)
            )
            times, channels = mbh_wave_gen.compute_tdi_channels(*params_in)
            # asnumpy: GPU-backend generator output -> host grid placement.
            mbh_td += np.asarray(
                place_td_signal_on_grid(
                    asnumpy(channels), grid, times=asnumpy(times)
                ).arr
            )[:nchannels]
            logger.info(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [end]")
    return emri_td, sobbh_td, mbh_td


class L1ProcessingStepWithSyntheticNoise(L1ProcessingStep):
    """Mojito L1 source loader + instrument noise + modulated foreground.

    Loads MBHB / EMRI / SOBHB source TD signals from a mojito L1 folder
    (no GB / VGB unless requested), then optionally adds:
      1. Instrument noise — either the **real mojito NOISE brick**
         (``add_instrument_noise="mojito"``: the loader sums the
         ``INSTRUMENT/L1`` TD noise stream, which shares the sources' exact
         time grid) or a synthetic FD-correlated draw from the
         ``(noise_soms_d, noise_sa_a)`` covariance
         (``add_instrument_noise=True`` / ``"synthetic"``).
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
        add_instrument_noise: "bool | str" = False,
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
        if add_instrument_noise not in (False, True, "synthetic", "mojito"):
            raise ValueError(
                f"add_instrument_noise={add_instrument_noise!r} not recognised; "
                "use False, True, 'synthetic', or 'mojito'."
            )
        # Drop any class whose source_ids list is empty (mojito's
        # L1DataLoader raises on missing IDs). GB / VGB load the whole
        # galaxy file (ids just gate inclusion). "mojito" instrument noise
        # rides the base loader ("NOISE" reads INSTRUMENT/L1 and sums it
        # into the data on the shared time grid); the synthetic instrument
        # noise + galactic foreground come from the synthetic generators.
        source_types = [
            t for t in ["GB", "VGB", "MBHB", "EMRI", "SOBHB"] if source_ids.get(t)
        ]
        if add_instrument_noise == "mojito":
            source_types = ["NOISE"] + source_types
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
        if add_instrument_noise in (True, "synthetic"):
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
        add_instrument_noise: "bool | str" = False,
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
        if add_instrument_noise == "mojito":
            raise ValueError(
                "add_instrument_noise='mojito' (the real NOISE brick) is a "
                "mojito-data option; the all-synthetic processor only draws "
                "synthetic noise."
            )
        if add_instrument_noise in (True, "synthetic"):
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


class SyntheticNoiseProcessingStep(BaseProcessingStep):
    """Canonical mojito-shaped synthetic *noise* processor (no mojito data).

    A proper :class:`~lisatools.globalfit.preprocessing.BaseProcessingStep`
    subclass — it flows through the normal ``process`` / ``pour`` pipeline and
    exposes ``td_signal`` / ``catalogue`` / ``orbits`` in the same shape a
    mojito :class:`~lisatools.globalfit.preprocessing.L1ProcessingStep` would,
    so downstream engine code reads it uniformly — but it needs **no mojito
    files**. There are no GW *sources*: the data are a self-consistent noise
    realization Cholesky-drawn from the *injected* composite covariance
    (instrument PSD + optional modulated galactic foreground + optional SGWB),
    the very covariance the noise fit recovers.

    The draw lives directly in the analysis (WDM) basis so the full per-element
    time-modulation and SGWB structure are captured exactly (a TD generator
    cannot reproduce a time-varying ``(3, 3, Nt)`` covariance). ``process``
    establishes the time grid with a zero TD stream; the noise is materialised
    in ``pour``. Swap this ``data_processor_class`` for a Sangria / mojito / CD1L
    loader to run the same noise fit on real data.

    Args:
        Tobs: Observation span (seconds); the time grid is ``N = round(Tobs/dt)``.
        dt: Underlying TD sample spacing (seconds).
        psd_injection: ``(Soms_d, Sa_a)`` instrument noise levels (sqrt units).
        galfor_injection: Optional galactic-foreground parameters; ``None`` omits
            the foreground component.
        sgwb_injection: Optional SGWB spectral-template parameters; ``None``
            (the default, and always for ``noise_only``) omits the SGWB
            component.
        galfor_modulation: Per-element foreground time modulation forwarded to
            :class:`~lisatools.sensitivity.CompositeSensitivityBackend`
            (``None`` = stationary; a callable such as
            :class:`~lisatools.sensitivity.GalForTimeModulation`, or an array).
        sgwb_stochastic_fn: SGWB spectral template (class or stock name).
        tdi_generation: 1 or 2.
        seed: RNG seed for the noise draw (reproducible data).
        t_start: Start time of the stream (seconds).
        nchannels: Number of TDI channels (3 for XYZ).
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        psd_injection: Sequence[float],
        galfor_injection: Optional[Sequence[float]] = None,
        sgwb_injection: Optional[Sequence[float]] = None,
        galfor_modulation: Optional[object] = None,
        sgwb_stochastic_fn: str = "PowerLawSGWB",
        tdi_generation: int = 2,
        seed: int = 0,
        t_start: float = 0.0,
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        self.Tobs = float(Tobs)
        self.psd_injection = np.asarray(psd_injection, dtype=float)
        self.galfor_injection = (
            None if galfor_injection is None else np.asarray(galfor_injection, dtype=float)
        )
        self.sgwb_injection = (
            None if sgwb_injection is None else np.asarray(sgwb_injection, dtype=float)
        )
        self.galfor_modulation = galfor_modulation
        self.sgwb_stochastic_fn = sgwb_stochastic_fn
        self.tdi_generation = int(tdi_generation)
        self.seed = int(seed)

        # Zero TD placeholder grid — establishes the (mojito-shaped) time base;
        # the actual noise is drawn in the analysis domain in ``pour``.
        N = int(round(self.Tobs / dt))
        times = np.arange(N) * dt + t_start
        data = np.zeros((nchannels, N), dtype=float)
        BaseProcessingStep.__init__(
            self, times, data, 1.0 / dt, verbose=verbose, do_plots=do_plots
        )

        # ``pour`` returns orbits=None so the engine keeps the settings' orbits;
        # ``catalogue`` mirrors the mojito shape but carries only the noise
        # injection truths (no source-class keys -> zero source leaves).
        self.orbits = None
        self.catalogue = dict(
            psd_injection=self.psd_injection,
            galfor_injection=self.galfor_injection,
            sgwb_injection=self.sgwb_injection,
            seed=self.seed,
        )

    def pour(self, settings, window=None, return_orbits=False):
        """Draw the composite-noise realization on the active WDM grid."""
        if not isinstance(settings, WDMSettings):
            raise NotImplementedError(
                "SyntheticNoiseProcessingStep pours into WDM domains only; got "
                f"{type(settings).__name__}. (The composite draw is done in the "
                "analysis basis; extend here for FD/STFT if needed.)"
            )
        backend = CompositeSensitivityBackend(
            settings,
            tdi_generation=self.tdi_generation,
            galfor_modulation=self.galfor_modulation,
            sgwb_stochastic_fn=self.sgwb_stochastic_fn,
        )
        sensmat = backend(
            "injection",
            self.psd_injection,
            galfor_params=self.galfor_injection,
            sgwb_params=self.sgwb_injection,
        )
        # sens_mat is on the active grid (nch, nch, Nf_active, Nt_active) -- the
        # same shape TDSignal.transform produces for poured data. Keep the whole
        # draw on the analysis backend (settings.xp: numpy on CPU, cupy on GPU).
        xp = settings.xp
        C = xp.asarray(sensmat.sens_mat)
        nch, _, nf, nt = C.shape
        Cp = C.transpose(2, 3, 0, 1).reshape(-1, nch, nch)  # (Npix, nch, nch)
        # Clean non-finite band-edge pixels (they are dropped by the likelihood's
        # detC mask anyway) and add a tiny diagonal jitter so every pixel is
        # positive-definite for the batched Cholesky. The jitter is far below the
        # model's precision and negligible on real (finite) pixels.
        Cp = xp.where(xp.isfinite(Cp), Cp, 0.0)
        diag = xp.einsum("...ii->...i", Cp)
        eps = 1e-12 * xp.maximum(diag.max(axis=1, keepdims=True), 1e-300)
        Cp = Cp + eps[..., None] * xp.eye(nch)
        L = xp.linalg.cholesky(Cp)
        # Host numpy RNG on CPU keeps the draw reproducible + bit-identical; on
        # GPU a cupy RNG keeps the draw on-device (a different but valid
        # realization for a given seed -- CPU is the reproducible reference).
        rng = (np if xp is np else xp).random.default_rng(self.seed)
        z = rng.standard_normal((Cp.shape[0], nch, 1))
        draw = (L @ z)[:, :, 0].reshape(nf, nt, nch).transpose(2, 0, 1)  # (nch, nf, nt)

        data_signal = WDMSignal(draw, settings)
        if return_orbits:
            return data_signal, self.orbits
        return data_signal


class UserDataProcessor(BaseProcessingStep):
    """Bring-your-own-data processor: wrap user arrays as a run's data source.

    The simplest way to run any stock variant on your own data — hand it the
    time-domain channels (and optionally your own orbits) and swap it in; an
    explicit ``data_processor_class`` always wins over the variant default::

        fit = erebor.blank(
            data_processor_class=UserDataProcessor,
            processor_init_kwargs=dict(data=my_xyz, dt=5.0, orbits=my_orbits),
        )

    The inherited ``process``/``pour`` pipeline handles conditioning and the
    transform into the run's analysis domain, exactly as for loaded data.

    Args:
        data: Time-domain channel data, shape ``(nchannels, N)`` (a 1D array
            is treated as one channel).
        dt: Sample spacing (seconds).
        t_start: Start time of the stream (seconds).
        orbits: Optional orbits instance; ``None`` keeps the run's configured
            orbits.
        catalogue: Optional injection-truth metadata dict (mojito-shaped),
            recorded alongside the run.
    """

    def __init__(
        self,
        data: np.ndarray,
        dt: float,
        t_start: float = 0.0,
        orbits=None,
        catalogue: Optional[dict] = None,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        data = np.atleast_2d(np.asarray(data, dtype=float))
        times = np.arange(data.shape[-1]) * dt + t_start
        BaseProcessingStep.__init__(
            self, times, data, 1.0 / dt, verbose=verbose, do_plots=do_plots
        )
        # ``pour(return_orbits=True)`` hands these to the engine; orbits=None
        # keeps the settings' orbits.
        self.orbits = orbits
        self.catalogue = dict(catalogue) if catalogue else {}


class ZeroDataProcessingStep(BaseProcessingStep):
    """All-zero data streams: the truly blank canvas.

    The residual starts at exactly zero, so with a fixed PSD and a source-only
    likelihood the null log-like is exactly 0 — inject your own signal (or
    don't) and everything you see in the residual is yours. Default data
    source of the ``blank`` stock variant (``include_noise=True`` swaps in the
    synthetic noise realization instead).

    Args:
        Tobs: Observation span (seconds); the time grid is ``N = round(Tobs/dt)``.
        dt: Sample spacing (seconds).
        t_start: Start time of the stream (seconds).
        nchannels: Number of TDI channels (3 for XYZ).
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float = 0.0,
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        N = int(round(Tobs / dt))
        times = np.arange(N) * dt + t_start
        BaseProcessingStep.__init__(
            self, times, np.zeros((nchannels, N)), 1.0 / dt, verbose=verbose, do_plots=do_plots
        )
        # pour(return_orbits=True) hands these to the engine; orbits=None keeps
        # the settings' orbits (the base pour transforms the zero TD stream).
        self.orbits = None
        self.catalogue = {}
