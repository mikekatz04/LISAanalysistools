"""Likelihood surfaces of four galactic-binary waveforms against a time-domain truth.

    python likelihood_surfaces.py --device gpu --truth DIR/truth --output DIR
        [--sources NAME ...] [--days 365] [--check-only] [--no-figures] [--no-summary]
    python likelihood_surfaces.py --output DIR --summarize-only

The data are the noiseless X2, Y2 and Z2 that ``td_truth.py`` made for the eight validation sources of ``stft_accuracy.py`` (lisagwresponse and
pytdi, a year on the ESA orbits that LAT ships), scaled to an optimal SNR of 100. SNR² is the STFT
(d|d) with the periodic Tukey window and 60 side bins. Each waveform is evaluated in its own
domain, against the transform of the same data, always on X, Y and Z with the full 3 x 3
cross-spectral density (CSD) of ``XYZSensitivityBackend``:

* STFT: both responses of ``STFTGBComputations`` (TDI on the fly and DFFR), through
  ``get_ll_stft``, on the STFT of the data with the periodic Tukey window, as in ``stft_accuracy.py``; one CSD
  per segment;
* FD: ``GBFDComputations`` (``GBTDIonTheFly`` sampled on N points over the year, then Fourier
  transformed) and ``gbgpu.gbgpu.GBGPU`` (FastGB), on dt rfft of the data with the window of the
  global fit's FD path (LAT's symmetric Tukey with a taper of 1e4 s), with their own windows off,
  since neither resolves that taper. Both go through one product in this script, with one CSD
  whose transfer functions are averaged over the orbit.

In each domain ln L(theta) = -(d - h(theta) | d - h(theta)) / 2, so an exact waveform gives 0 at
the truth. A surface scans one of the eight parameters of the global fit (the nine of
``GBTDIonTheFly`` without fddot), all others at the truth, over +-4 sigma, with sigma = 1/sqrt(F_ii)
from DFFR's information matrix at the truth; a peak farther than 3 sigma from the centre of its
scan gets up to three more scans centred on it. Parameters refer to t_ref = 0, the start of the
orbit file; the FD codes take them at the data start, carried there per point. FastGB runs with
``flip_ref_phase=True``, since its phase is +phi0 + 2 pi f0 xi + ... where LAT's is
-phi0 + 2 pi f0 tau + ....

Before the scans, a convention check at the truth gives each waveform's overlap, amplitude ratio
and phase of (d|h); FastGB's compiled path against its Python path; ``get_ll_fd``, which keeps only
the real part of the inverse CSD, against the script's product; both FD codes at 4 and 16 times
their N and against the data without a window; the FD window against no window; and the weight of
T = (X + Y + Z) / sqrt(3) in both domains, with the other FD CSD of the backend (at the
orbit-averaged arm lengths) for comparison.

Every run writes NAME.json, NAME.npz and NAME.png per source and then, unless ``--no-summary``,
the combined summary.json, report.txt and summary figures of every source in the output folder;
``--summarize-only`` writes only those. There is no threshold; the results are recorded in
``docs/response/dffr/results.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stft_accuracy import (  # noqa: E402
    AXIS,
    DEFAULT_N_SIDE_BINS,
    DOPPLER_FRACTION,
    GRID,
    INK,
    INK_SECONDARY,
    MUTED,
    RESPONSE_COLORS,
    SURFACE,
    TDI_GENERATION,
    TruthInjection,
    build_comp,
    inner,
    make_source_set,
    source_settings,
    window_samples,
)
from stft_fixtures import (  # noqa: E402
    BAND_MIN_FREQ,
    PRODUCTION_WINDOW_ALPHA,
    SAMPLING_INTERVAL,
    SEGMENT_DURATION,
    array_module,
    load_orbits,
    run_metadata,
    to_host,
)
from td_injection import PARAMETER_NAMES  # noqa: E402

DEFAULT_DAYS = 365
SNR_TARGET = 100.0
#: Indices into PARAMETER_NAMES of the eight scanned parameters: all but fddot.
SCAN_INDS = (0, 1, 2, 4, 5, 6, 7, 8)
#: Half width of a scan in sigma, and points per scan (spacing 0.1 sigma, the truth on the grid).
SCAN_WIDTH = 4.0
SCAN_POINTS = 81
#: Half width in sigma, about the largest grid value, of the parabola that locates a peak.
PEAK_FIT_HALF_WIDTH = 1.0
#: A peak farther than RESCAN_THRESHOLD sigma from the centre of its scan gets another scan
#: centred on it, at most RESCAN_LIMIT times.
RESCAN_THRESHOLD = 3.0
RESCAN_LIMIT = 3
#: Ranges of the bounded angles: iota in [0, pi], beta in [-pi/2, pi/2].
BOUNDS = {5: (0.0, np.pi), 8: (-0.5 * np.pi, 0.5 * np.pi)}
#: The largest sample count of FastGB's compiled kernels; beyond it, its Python path runs.
FASTGB_COMPILED_MAX = 2048
# * FD sample counts: four times the smallest power of two, at least 2048, that holds the
# * Doppler-shifted frequency track from the carrier at the data start with a margin. At the base
# * length the FD heterodyne's own sampling error reached 2.2 in ln L at the truth; at four times
# * it, at most 0.1 (author's choice of 2026-10-21, docs/response/dffr/results.md).
FD_BASE_LENGTH = 2048
FD_LENGTH_FACTOR = 4
FD_MARGIN_BINS = 64
# ? Taper of the global fit's FD window: window_taper_duration = 1 / start_freq of the production
# ? settings (mojito_input/galfor_psd_multigpu_stft_settings.py), as engine.py applies it to FD.
FD_TAPER_DURATION = 1.0 / BAND_MIN_FREQ
#: Template samples per call of the FD codes: 256 templates at N = 8192, 8 at N = 262144, where
#: the Python path of FastGB holds about 20 arrays of (batch, N) complex values at once.
FD_BATCH_SAMPLES = 1 << 21
#: Multiples of N at which the convention check also evaluates both FD codes at the truth, on
#: the same band, to separate their sampling error from the data window.
FD_RESOLUTION_FACTORS = (1, 4, 16)

WAVEFORMS = ("dffr", "tdi_on_the_fly", "fd_tdi_on_the_fly", "fastgb")
WAVEFORM_LABELS = {
    "dffr": "DFFR (STFT)",
    "tdi_on_the_fly": "TDI on the fly (STFT)",
    "fd_tdi_on_the_fly": "TDI on the fly (FD)",
    "fastgb": "FastGB (FD)",
}
# * Categorical slots 1 to 4 in fixed order; the STFT responses keep stft_accuracy.py's colours.
WAVEFORM_COLORS = {
    "dffr": RESPONSE_COLORS["dffr"],
    "tdi_on_the_fly": RESPONSE_COLORS["tdi_on_the_fly"],
    "fd_tdi_on_the_fly": "#1baf7a",
    "fastgb": "#eda100",
}
#: The FD heterodyne's own likelihood kernel, which keeps only Re(C^-1): evaluated at the truth.
KERNEL_VARIANT = "fd_tdi_on_the_fly_kernel"
PARAMETER_LABELS = {
    "amp": "A",
    "f0": "f0 [Hz]",
    "fdot": "fdot [Hz/s]",
    "phi0": "phi0 [rad]",
    "iota": "iota [rad]",
    "psi": "psi [rad]",
    "lam": "lam [rad]",
    "beta": "beta [rad]",
}


def log(message):
    print(f"[surfaces] {time.strftime('%H:%M:%S')} {message}", flush=True)


def at_epoch(params, shift):
    """Parameters referred to t_ref + shift instead of t_ref, in the GB model of gb_physics.hh.

    The phase -phi0 + 2 pi (f0 tau + fdot tau^2 / 2 + fddot tau^3 / 6) carries over exactly; the
    amplitude A (1 + 2/3 fdot/f0 tau) is carried to first order, as the model has no other form.
    """
    out = np.array(params, dtype=np.float64, copy=True)
    amp, f0, fdot, fddot, phi0 = (out[..., k].copy() for k in range(5))
    cycles = f0 * shift + 0.5 * fdot * shift * shift + fddot * shift * shift * shift / 6.0
    out[..., 0] = amp * (1.0 + (2.0 / 3.0) * fdot / f0 * shift)
    out[..., 1] = f0 + fdot * shift + 0.5 * fddot * shift * shift
    out[..., 2] = fdot + fddot * shift
    # * Whole cycles dropped before the multiplication by 2 pi, which keeps the phase exact.
    out[..., 4] = np.mod(phi0 - 2.0 * np.pi * (cycles - np.floor(cycles)), 2.0 * np.pi)
    return out


def fd_length(params, data_start, num_samples, t_ref, df):
    """Sample count of both FD codes and the carrier bin at the data start."""
    times = np.linspace(data_start, data_start + num_samples * SAMPLING_INTERVAL, 1001) - t_ref
    f0, fdot, fddot = params[1:4]
    track = f0 + fdot * times + 0.5 * fddot * times * times
    carrier = at_epoch(params, data_start - t_ref)[1]
    carrier_bin = int(round(carrier / df))
    doppler_bins = int(np.ceil(DOPPLER_FRACTION * track.max() / df))
    reach = np.abs(track / df - carrier_bin).max() + doppler_bins + FD_MARGIN_BINS
    base = max(FD_BASE_LENGTH, 1 << int(np.ceil(np.log2(2.0 * reach))))
    return FD_LENGTH_FACTOR * base, carrier_bin


def fd_inner(first, second, inverse_covariance, df):
    """Complex 4 df sum conj(first_i) C^-1_ij second_j of (..., 3, N) arrays over the band."""
    return 4.0 * df * np.einsum("...if,ijf,...jf->...", np.conj(first), inverse_covariance, second)


def release_device_memory(xp):
    """Return the blocks that CuPy's memory pool keeps to the device, between the large phases.

    Processes run side by side on one GPU (``run_dffr.sh`` part ``surfaces``); each would
    otherwise keep the peak of its data phase, 2 to 4.5 GB, for the rest of the run.
    """
    if hasattr(xp, "get_default_memory_pool"):
        xp.get_default_memory_pool().free_all_blocks()


def without_t(data):
    """X, Y, Z with the combination T = (X + Y + Z) / sqrt(3) removed (channel axis 0)."""
    return data - np.mean(data, axis=0, keepdims=True)


def pair_metrics(model, reference, product):
    """Mismatch, amplitude ratio, phase and residual of ``model`` against ``reference``."""
    model_model = product(model, model).real
    reference_reference = product(reference, reference).real
    cross = product(reference, model)
    difference = model - reference
    return dict(
        mismatch=float(1.0 - cross.real / np.sqrt(model_model * reference_reference)),
        amplitude_ratio=float(np.sqrt(model_model / reference_reference)),
        phase=float(np.angle(cross)),
        residual=float(product(difference, difference).real / reference_reference),
    )


def scan_row(params, sigma, row, centre):
    """Points of one scan: (SCAN_POINTS, 9) parameters, offsets in sigma, and validity."""
    index = SCAN_INDS[row]
    offsets = centre + np.linspace(-SCAN_WIDTH, SCAN_WIDTH, SCAN_POINTS)
    points = np.repeat(params[None, :], SCAN_POINTS, axis=0)
    values = params[index] + offsets * sigma[row]
    valid = np.ones(SCAN_POINTS, dtype=bool)
    if index in BOUNDS:
        low, high = BOUNDS[index]
        valid = (values >= low) & (values <= high)
        # * Out-of-range points are evaluated at the bound and reported as missing.
        values = np.clip(values, low, high)
    points[:, index] = values
    return points, offsets, valid


def peak_metrics(offsets, lnl):
    """Peak position and width (in sigma) of a parabola fitted near the largest grid value."""
    finite = np.isfinite(lnl)
    if not finite.any():
        return dict(peak=float("nan"), width=float("nan"), lnl_peak=float("nan"))
    best = int(np.nanargmax(np.where(finite, lnl, -np.inf)))
    near = finite & (np.abs(offsets - offsets[best]) <= PEAK_FIT_HALF_WIDTH + 1e-9)
    curvature, slope, constant = np.polyfit(offsets[near], lnl[near], 2)
    if curvature >= 0.0:
        return dict(peak=float(offsets[best]), width=float("nan"), lnl_peak=float(lnl[best]))
    return dict(
        peak=float(-slope / (2.0 * curvature)),
        width=float(np.sqrt(-0.5 / curvature)),
        lnl_peak=float(constant - slope * slope / (4.0 * curvature)),
    )


class SourceSurfaces:
    """Data, waveforms and likelihoods of one source in both domains."""

    def __init__(self, name, params_file, injection, orbits, cpu_orbits, t_ref, days, device):
        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.domains import FDSettings, FDSignal, TDSettings, TDSignal
        from lisatools.sensitivity import XYZSensitivityBackend
        from lisatools.utils.utility import windowfun

        self.name, self.device, self.t_ref = name, device, t_ref
        self.xp = array_module(device)
        self.timings = {}
        started = time.time()
        num_samples = injection.num_samples
        self.data_start = injection.data_start
        series = injection.series(name, params_file, t_ref)

        # ---- STFT: domain, noise and the scale to SNR 100 ----
        settings = source_settings(
            params_file, self.data_start, days, t_ref, DEFAULT_N_SIDE_BINS, device
        )
        time_settings = TDSettings(
            num_samples, SAMPLING_INTERVAL, self.data_start, force_backend=device
        )
        segment_window = self.xp.asarray(
            window_samples(
                "periodic", settings.get_nperseg(SAMPLING_INTERVAL), PRODUCTION_WINDOW_ALPHA
            )
        )
        sensitivity = XYZSensitivityBackend(
            orbits=orbits, settings=settings, tdi_generation=2, force_backend=device
        )
        sensitivity.set_sensitivity_matrix()
        self.stft_inverse_covariance = to_host(sensitivity.invC)
        unscaled = TDSignal(series, settings=time_settings).stft(
            window=segment_window, settings=settings
        )
        unscaled_d_d = inner(
            to_host(unscaled.arr), to_host(unscaled.arr), self.stft_inverse_covariance, settings.df
        ).real
        self.scale = SNR_TARGET / np.sqrt(unscaled_d_d)
        self.params = np.array(params_file, dtype=np.float64)
        self.params[0] *= self.scale
        series = series * self.scale
        del unscaled

        stft_data = TDSignal(series, settings=time_settings).stft(
            window=segment_window, settings=settings
        )
        self.container_array = AnalysisContainerArray(
            [AnalysisContainer(stft_data, sensitivity)],
            gpus=None if device == "cpu" else [0],
            domain_group_kwargs=dict(
                tdi_type="XYZ",
                window_alpha=PRODUCTION_WINDOW_ALPHA,
                use_midpoint=True,
                linear_envelope=True,
            ),
        )
        self.group = self.container_array.cpp_splits[0]
        self.group.compute_d_d_term()
        self.stft_d_d = float(to_host(self.group.d_d)[0])
        self.stft_data = to_host(stft_data.arr)
        self.stft_df = settings.df
        self.stft_band = (int(settings.ind_min), int(settings.NF_active))
        self.comps = {
            response: build_comp(
                self.group,
                orbits,
                t_ref,
                days,
                DEFAULT_N_SIDE_BINS,
                PRODUCTION_WINDOW_ALPHA,
                response,
                device,
            )
            for response in ("dffr", "tdi_on_the_fly")
        }

        # ---- FD: band, window and data ----
        df = 1.0 / (num_samples * SAMPLING_INTERVAL)
        self.fd_df = df
        self.fd_length, self.fd_centre = fd_length(
            self.params, self.data_start, num_samples, t_ref, df
        )
        self.fd_first = self.fd_centre - self.fd_length // 2
        num_rfft = num_samples // 2 + 1
        # * Half-bin edges, so that the band holds exactly the N bins of the FD codes.
        band = dict(
            min_freq=(self.fd_first - 0.5) * df,
            max_freq=(self.fd_first + self.fd_length - 0.5) * df,
        )
        fd_settings = FDSettings(N=num_rfft, df=df, force_backend=device, **band)
        self.fd_window_alpha = FD_TAPER_DURATION / (num_samples * SAMPLING_INTERVAL)
        fd_window, _ = windowfun("tukey", num_samples, alpha=self.fd_window_alpha)
        fd_window = self.xp.asarray(fd_window)[None, :]
        signal = TDSignal(series, settings=time_settings)
        self.fd_data = to_host(signal.fft(settings=fd_settings, window=fd_window).arr)
        if self.fd_data.shape != (3, self.fd_length) or int(fd_settings.ind_min) != self.fd_first:
            raise RuntimeError(f"{name}: the FD band is not the band of the FD codes")
        # * The same year without a window, which both FD codes model: for the convention check.
        self.fd_data_rect = to_host(signal.fft(settings=fd_settings).arr)
        # * Unweighted share of the window's change that falls inside the band, over all bins.
        full_settings = FDSettings(N=num_rfft, df=df, force_backend=device)
        full_rect = signal.fft(settings=full_settings).arr
        change = signal.fft(settings=full_settings, window=fd_window).arr - full_rect
        total_change = float(self.xp.sum(self.xp.abs(change) ** 2))
        in_band = change[:, self.fd_first : self.fd_first + self.fd_length]
        self.fd_window_change = dict(
            change_over_rect_all_bins=total_change
            / float(self.xp.sum(self.xp.abs(full_rect) ** 2)),
            share_in_band=float(self.xp.sum(self.xp.abs(in_band) ** 2)) / total_change,
        )
        del series, signal, full_rect, change, in_band
        release_device_memory(self.xp)

        # ---- FD: the CSD with transfer functions averaged over the orbit (author's choice) ----
        cpu_settings = FDSettings(N=num_rfft, df=df, force_backend="cpu", **band)
        cpu_sensitivity = XYZSensitivityBackend(
            orbits=cpu_orbits,
            settings=cpu_settings,
            tdi_generation=2,
            force_backend="cpu",
            average_transfer_functions=True,
        )
        cpu_sensitivity.set_sensitivity_matrix()
        self.fd_inverse_covariance = np.asarray(cpu_sensitivity.invC)
        conjugate_transpose = np.conj(np.transpose(self.fd_inverse_covariance, (1, 0, 2)))
        self.fd_hermitian_error = float(
            np.abs(self.fd_inverse_covariance - conjugate_transpose).max()
            / np.abs(self.fd_inverse_covariance).max()
        )
        self.fd_d_d = float(self.fd_product(self.fd_data, self.fd_data).real)
        self.fd_d_d_real = float(
            fd_inner(self.fd_data, self.fd_data, self.fd_inverse_covariance.real, df).real
        )
        self.fd_holder = AnalysisContainerArray(
            [AnalysisContainer(FDSignal(self.fd_data.copy(), cpu_settings), cpu_sensitivity)],
            gpus=None,
        )
        self.fd_window_only = pair_metrics(self.fd_data, self.fd_data_rect, self.fd_product)

        # ---- noise diagnostics: the weight of T, and the backend's other FD CSD ----
        def stft_product(first, second):
            return inner(first, second, self.stft_inverse_covariance, self.stft_df)

        arm_lengths = XYZSensitivityBackend(
            orbits=cpu_orbits, settings=cpu_settings, tdi_generation=2, force_backend="cpu"
        )
        arm_lengths.set_sensitivity_matrix()
        arm_lengths_inverse = np.asarray(arm_lengths.invC)

        def arm_lengths_product(first, second):
            return fd_inner(first, second, arm_lengths_inverse, df)

        arm_lengths_d_d = arm_lengths_product(self.fd_data, self.fd_data).real
        self.noise_diagnostics = dict(
            stft_share_without_t=float(
                stft_product(without_t(self.stft_data), without_t(self.stft_data)).real
                / self.stft_d_d
            ),
            fd_share_without_t=float(
                self.fd_product(without_t(self.fd_data), without_t(self.fd_data)).real / self.fd_d_d
            ),
            fd_snr_arm_lengths_averaged=float(np.sqrt(arm_lengths_d_d)),
            fd_share_without_t_arm_lengths_averaged=float(
                arm_lengths_product(without_t(self.fd_data), without_t(self.fd_data)).real
                / arm_lengths_d_d
            ),
        )
        self.fd_shift = self.data_start - t_ref
        self.fd_duration = num_samples * SAMPLING_INTERVAL
        self.timings["data"] = round(time.time() - started, 1)

        from gbgpu.gbcomps import GBFDComputations
        from gbgpu.gbgpu import GBGPU

        from lisatools.response.tdiconfig import TDIConfig

        # ! The heterodyne requires t_start = t_ref: its parameters refer to the data start.
        self.fd_comp = GBFDComputations(
            cpu_settings,
            self.data_start,
            N_sparse=self.fd_length,
            orbits=cpu_orbits,
            tdi_config=TDIConfig(TDI_GENERATION, force_backend="cpu"),
            force_backend="cpu",
            tdi_type="XYZ",
            nchannels=3,
            tukey_alpha=0.0,
        )
        # ? FastGB's phase is +phi0 + 2 pi f0 xi + ... from its t0 on, LAT's -phi0 + 2 pi f0 tau +
        # ? ...; flip_ref_phase negates phi0, and the convention check confirms the mapping.
        self.fastgb = GBGPU(
            orbits=orbits, force_backend=device, t0=self.data_start, flip_ref_phase=True
        )

    def fd_product(self, first, second):
        return fd_inner(first, second, self.fd_inverse_covariance, self.fd_df)

    # ---- likelihood terms: complex (d|h) and real (h|h) per point ----

    def stft_terms(self, response, points):
        comp = self.comps[response]
        num = points.shape[0]
        comp.get_ll_stft(
            self.xp.asarray(points),
            data_index=self.xp.zeros(num, dtype=self.xp.int32),
            noise_index=self.xp.zeros(num, dtype=self.xp.int32),
        )
        return to_host(comp.d_h_out).astype(complex), to_host(comp.h_h_out).real

    def fd_tdi_templates(self, points, length=None):
        """Templates of GBFDComputations on the FD band, (num, 3, N), sampled on ``length``."""
        length = self.fd_length if length is None else length
        mapped = at_epoch(points, self.fd_shift)
        templates = np.zeros((mapped.shape[0], 3, length), dtype=np.complex128)
        self.fd_comp.fill_global(
            mapped,
            templates,
            data_index=np.arange(mapped.shape[0], dtype=np.int32),
            template_start_inds=np.full(
                mapped.shape[0], self.fd_centre - length // 2, dtype=np.int32
            ),
            N_sparse=length,
        )
        middle = length // 2 - self.fd_length // 2
        return templates[:, :, middle : middle + self.fd_length]

    def run_fastgb(self, points, length, compiled):
        """FastGB templates (num, 3, length) and their first bins, on the parameters ``points``."""
        mapped = at_epoch(points, self.fd_shift)
        # ! run_wave takes NumPy arrays (np.atleast_1d) and moves them to its backend itself.
        self.fastgb.run_wave(
            *mapped.T,
            N=length,
            T=self.fd_duration,
            dt=SAMPLING_INTERVAL,
            tdi2=True,
            tdi_channel_setup="XYZ",
            use_c_implementation=compiled,
        )
        return to_host(self.fastgb.XYZf), to_host(self.fastgb.start_inds)

    def fastgb_templates(self, points, length=None):
        """FastGB templates on the FD band, (num, 3, N), sampled on ``length``."""
        length = self.fd_length if length is None else length
        templates, starts = self.run_fastgb(points, length, length <= FASTGB_COMPILED_MAX)
        if np.any(starts != self.fd_centre - length // 2):
            raise RuntimeError(f"{self.name}: FastGB templates left the FD band")
        middle = length // 2 - self.fd_length // 2
        return templates[:, :, middle : middle + self.fd_length]

    def fd_terms(self, waveform, points):
        build = self.fd_tdi_templates if waveform == "fd_tdi_on_the_fly" else self.fastgb_templates
        batch = max(1, FD_BATCH_SAMPLES // self.fd_length)
        d_h, h_h = [], []
        for first in range(0, points.shape[0], batch):
            templates = build(points[first : first + batch])
            d_h.append(self.fd_product(self.fd_data[None], templates))
            h_h.append(self.fd_product(templates, templates).real)
        return np.concatenate(d_h), np.concatenate(h_h)

    def kernel_terms(self, points):
        """get_ll_fd of GBFDComputations, which contracts with Re(C^-1) only."""
        self.fd_comp.get_ll_fd(at_epoch(points, self.fd_shift), self.fd_holder)
        return np.asarray(self.fd_comp.d_h_out).astype(complex), np.asarray(self.fd_comp.h_h_out)

    def terms(self, waveform, points):
        started = time.time()
        if waveform in ("dffr", "tdi_on_the_fly"):
            result = self.stft_terms(waveform, points)
        elif waveform == KERNEL_VARIANT:
            result = self.kernel_terms(points)
        else:
            result = self.fd_terms(waveform, points)
        self.timings[waveform] = round(self.timings.get(waveform, 0.0) + time.time() - started, 1)
        return result

    def d_d(self, waveform):
        if waveform in ("dffr", "tdi_on_the_fly"):
            return self.stft_d_d
        return self.fd_d_d_real if waveform == KERNEL_VARIANT else self.fd_d_d

    def log_likelihood(self, waveform, points):
        d_h, h_h = self.terms(waveform, points)
        return d_h.real - 0.5 * h_h - 0.5 * self.d_d(waveform)

    def sigma(self):
        """Conditional widths 1/sqrt(F_ii) and the Fisher matrix of DFFR at the truth."""
        fisher = to_host(
            self.comps["dffr"].information_matrix(
                self.xp.asarray(self.params[None]), inds=list(SCAN_INDS)
            )
        )[0]
        return 1.0 / np.sqrt(np.diag(fisher)), fisher

    def convention_check(self):
        """Every waveform at the truth, the parities of the FD codes and their resolution."""
        record = {}
        truth = self.params[None]
        for waveform in WAVEFORMS + (KERNEL_VARIANT,):
            d_h, h_h = self.terms(waveform, truth)
            d_d = self.d_d(waveform)
            record[waveform] = dict(
                overlap=float(d_h[0].real / np.sqrt(d_d * h_h[0])),
                amplitude_ratio=float(np.sqrt(h_h[0] / d_d)),
                phase=float(np.angle(d_h[0])),
                lnl=float(d_h[0].real - 0.5 * h_h[0] - 0.5 * d_d),
                d_d=d_d,
            )
        fastgb = self.fastgb_templates(truth)[0]
        fd_tdi = self.fd_tdi_templates(truth)[0]
        record["fd_templates"] = dict(fastgb=fastgb, fd_tdi_on_the_fly=fd_tdi)
        # * FastGB's Python path against its compiled kernel, at the largest compiled length.
        compiled, _ = self.run_fastgb(truth, FASTGB_COMPILED_MAX, True)
        python_path, _ = self.run_fastgb(truth, FASTGB_COMPILED_MAX, False)
        record["fastgb_python_vs_compiled"] = float(
            np.abs(python_path - compiled).max() / np.abs(compiled).max()
        )
        # * The kernel's product with Re(C^-1) against the script's products.
        _, kernel_h_h = self.kernel_terms(truth)
        own_real = fd_inner(fd_tdi, fd_tdi, self.fd_inverse_covariance.real, self.fd_df).real
        own_full = self.fd_product(fd_tdi, fd_tdi).real
        record["kernel_h_h_vs_script_real"] = float(abs(kernel_h_h[0] - own_real) / own_real)
        record["kernel_h_h_vs_script_full"] = float((kernel_h_h[0] - own_full) / own_full)
        # * Both FD codes on finer grids and against the data without a window, at the truth.
        rect_d_d = float(self.fd_product(self.fd_data_rect, self.fd_data_rect).real)
        data_sets = dict(tapered=(self.fd_data, self.fd_d_d), rect=(self.fd_data_rect, rect_d_d))
        resolution = {}
        for waveform, build in (
            ("fd_tdi_on_the_fly", self.fd_tdi_templates),
            ("fastgb", self.fastgb_templates),
        ):
            resolution[waveform] = {}
            for factor in FD_RESOLUTION_FACTORS:
                template = build(truth, length=factor * self.fd_length)[0]
                h_h = float(self.fd_product(template, template).real)
                entry = {}
                for label, (data, d_d) in data_sets.items():
                    d_h = complex(self.fd_product(data, template))
                    entry[label] = dict(
                        overlap=float(d_h.real / np.sqrt(d_d * h_h)),
                        amplitude_ratio=float(np.sqrt(h_h / d_d)),
                        phase=float(np.angle(d_h)),
                        lnl=float(d_h.real - 0.5 * h_h - 0.5 * d_d),
                    )
                resolution[waveform][str(factor * self.fd_length)] = entry
        record["fd_resolution"] = resolution
        release_device_memory(self.xp)
        return record


def scan_waveform(source, waveform, sigma, arrays):
    """Scans of every parameter for one waveform, with rescans of far peaks; returns metrics."""
    rows = [scan_row(source.params, sigma, row, 0.0) for row in range(len(SCAN_INDS))]
    flat = np.concatenate([points for points, _, _ in rows])
    lnl = source.log_likelihood(waveform, flat).reshape(len(SCAN_INDS), SCAN_POINTS)
    metrics = {}
    for row, (_, offsets, valid) in enumerate(rows):
        lnl[row, ~valid] = np.nan
        name = PARAMETER_NAMES[SCAN_INDS[row]]
        entry = peak_metrics(offsets, lnl[row])
        lnl_truth = float(lnl[row, SCAN_POINTS // 2])
        centre, rescans = 0.0, 0
        while abs(entry["peak"] - centre) > RESCAN_THRESHOLD and rescans < RESCAN_LIMIT:
            # * Another scan centred on the latest peak, for this waveform and parameter.
            centre, rescans = round(entry["peak"], 1), rescans + 1
            points, extra_offsets, extra_valid = scan_row(source.params, sigma, row, centre)
            extra = source.log_likelihood(waveform, points)
            extra[~extra_valid] = np.nan
            arrays[f"rescan{rescans}_offsets_{waveform}_{name}"] = extra_offsets
            arrays[f"rescan{rescans}_lnl_{waveform}_{name}"] = extra
            entry = peak_metrics(extra_offsets, extra)
        entry.update(
            lnl_truth=lnl_truth,
            rescans=rescans,
            peak_inside_scan=bool(abs(entry["peak"] - centre) <= SCAN_WIDTH),
        )
        metrics[name] = entry
    arrays[f"lnl_{waveform}"] = lnl
    release_device_memory(source.xp)
    if waveform == WAVEFORMS[0]:
        arrays["offsets"] = np.stack([offsets for _, offsets, _ in rows])
    return metrics


def evaluate_source(
    name, params_file, injection, orbits, cpu_orbits, t_ref, days, device, check_only
):
    """Convention check, scans and metrics of one source; returns (record, arrays)."""
    started = time.time()
    source = SourceSurfaces(name, params_file, injection, orbits, cpu_orbits, t_ref, days, device)
    sigma, fisher = source.sigma()
    check = source.convention_check()
    fd_templates = check.pop("fd_templates")
    record = dict(
        source=name,
        metadata=run_metadata(device),
        truth=injection.records[name],
        params=dict(zip(PARAMETER_NAMES, map(float, source.params))),
        amplitude_scale=float(source.scale),
        snr_stft=float(np.sqrt(source.stft_d_d)),
        snr_fd=float(np.sqrt(source.fd_d_d)),
        stft_band=dict(first_bin=source.stft_band[0], num_bins=source.stft_band[1]),
        fd_band=dict(first_bin=int(source.fd_first), num_bins=int(source.fd_length)),
        fd_window_alpha=float(source.fd_window_alpha),
        fd_hermitian_error=source.fd_hermitian_error,
        fd_d_d_real_over_full=float(source.fd_d_d_real / source.fd_d_d),
        fd_window_only=source.fd_window_only,
        fd_window_change=source.fd_window_change,
        noise_diagnostics=source.noise_diagnostics,
        sigma=dict(zip((PARAMETER_NAMES[index] for index in SCAN_INDS), map(float, sigma))),
        convention_check=check,
    )
    arrays = dict(
        params=source.params,
        sigma=sigma,
        fisher=fisher,
        stft_data=source.stft_data,
        fd_data=source.fd_data,
        fd_data_rect=source.fd_data_rect,
        fd_template_fastgb=fd_templates["fastgb"],
        fd_template_fd_tdi_on_the_fly=fd_templates["fd_tdi_on_the_fly"],
    )
    for line in truth_lines(record) + resolution_lines(check["fd_resolution"]):
        log(f"{name} {line}")
    for line in diagnostic_lines(record):
        log(f"{name} {line}")
    if not check_only:
        record["metrics"] = {
            waveform: scan_waveform(source, waveform, sigma, arrays) for waveform in WAVEFORMS
        }
        within = np.abs(arrays["offsets"]) <= 2.0 + 1e-9
        for waveform in WAVEFORMS[1:]:
            difference = np.abs(arrays[f"lnl_{waveform}"] - arrays["lnl_dffr"])
            for row, index in enumerate(SCAN_INDS):
                record["metrics"][waveform][PARAMETER_NAMES[index]][
                    "max_difference_to_dffr_2sigma"
                ] = float(np.nanmax(np.where(within[row], difference[row], np.nan)))
    record["timings"] = dict(source.timings, total=round(time.time() - started, 1))
    return record, arrays


def style_axes(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(True, which="major", color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)


def plot_source(record, arrays, path):
    """ln L minus each curve's own peak along every scan, rescans included, with the parabola."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), facecolor=SURFACE)
    offsets = arrays["offsets"]
    # * Drawn back to front: the STFT curves and the Fisher parabola nearly coincide, so DFFR lies
    # * above the FD curves, the STFT TDI on the fly is dashed over it, and the parabola is last.
    order = ("fastgb", "fd_tdi_on_the_fly", "dffr", "tdi_on_the_fly")
    styles = {"tdi_on_the_fly": "--"}
    for row, (axis, index) in enumerate(zip(axes.flat, SCAN_INDS)):
        name = PARAMETER_NAMES[index]
        low, high = -SCAN_WIDTH, SCAN_WIDTH
        for waveform in order:
            entry = record["metrics"][waveform][name]
            curves = [(offsets[row], arrays[f"lnl_{waveform}"][row])] + [
                (
                    arrays[f"rescan{k}_offsets_{waveform}_{name}"],
                    arrays[f"rescan{k}_lnl_{waveform}_{name}"],
                )
                for k in range(1, entry["rescans"] + 1)
            ]
            for number, (x_values, lnl) in enumerate(curves):
                axis.plot(
                    x_values,
                    lnl - entry["lnl_peak"],
                    color=WAVEFORM_COLORS[waveform],
                    linewidth=1.0,
                    linestyle=styles.get(waveform, "-"),
                    label=WAVEFORM_LABELS[waveform] if number == 0 else None,
                )
                low, high = min(low, x_values.min()), max(high, x_values.max())
        reference = np.linspace(low, high, 401)
        axis.plot(
            reference,
            -0.5 * reference * reference,
            color=INK,
            linewidth=0.8,
            linestyle=":",
            label="Fisher parabola of DFFR at the truth",
        )
        axis.set_ylim(-9.0, 0.6)
        axis.set_xlim(low, high)
        axis.set_title(
            f"{PARAMETER_LABELS[name]}: truth {record['params'][name]:.6g}, "
            f"sigma {record['sigma'][name]:.3g}",
            fontsize=9,
            color=INK,
        )
        axis.set_xlabel("(theta - theta_true) / sigma", fontsize=8, color=INK_SECONDARY)
        if row % 4 == 0:
            axis.set_ylabel("ln L minus its peak", fontsize=8, color=INK_SECONDARY)
        style_axes(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    params = record["params"]
    figure.suptitle(
        f"{record['source']}: f0 = {params['f0']:.4g} Hz, fdot = {params['fdot']:.3g} Hz/s, "
        f"lam = {params['lam']}, beta = {params['beta']}, SNR {record['snr_stft']:.1f} (STFT), "
        f"{record['snr_fd']:.1f} (FD)",
        fontsize=11,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_summary(records, path, metric, ylabel, reference_value):
    """One panel per source: the metric of every waveform along each parameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, 3.8 * rows), facecolor=SURFACE, sharey=True, squeeze=False
    )
    names = [PARAMETER_NAMES[index] for index in SCAN_INDS]
    positions = np.arange(len(names))
    shifts = np.linspace(-0.24, 0.24, len(WAVEFORMS))
    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)
    for axis, record in zip(axes.flat, records):
        axis.axhline(reference_value, color=AXIS, linewidth=1.0)
        for shift, waveform in zip(shifts, WAVEFORMS):
            values = [record["metrics"][waveform][name][metric] for name in names]
            axis.plot(
                positions + shift,
                values,
                linestyle="none",
                marker="o",
                markersize=5.5,
                markerfacecolor=WAVEFORM_COLORS[waveform],
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=WAVEFORM_LABELS[waveform],
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([PARAMETER_LABELS[name].split(" ")[0] for name in names])
        axis.set_title(record["source"], fontsize=9, color=INK)
        style_axes(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_levels(records, path):
    """-ln L at the truth of every waveform, per source: each waveform's own error there."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    positions = np.arange(len(records))
    shifts = np.linspace(-0.24, 0.24, len(WAVEFORMS))
    for shift, waveform in zip(shifts, WAVEFORMS):
        values = [-record["convention_check"][waveform]["lnl"] for record in records]
        axis.plot(
            positions + shift,
            values,
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=WAVEFORM_COLORS[waveform],
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=WAVEFORM_LABELS[waveform],
        )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels([record["source"] for record in records], rotation=30, ha="right")
    axis.set_ylabel("-ln L at the true parameters", fontsize=8, color=INK_SECONDARY)
    style_axes(axis)
    axis.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def truth_lines(record):
    """Every waveform at the truth and the parities of the FD codes."""
    check = record["convention_check"]
    lines = [
        f"at the truth {waveform:24s} overlap {check[waveform]['overlap']:.9f} amplitude ratio "
        f"{check[waveform]['amplitude_ratio']:.6f} phase {check[waveform]['phase']:+.2e} ln L "
        f"{check[waveform]['lnl']:.3e}"
        for waveform in WAVEFORMS + (KERNEL_VARIANT,)
    ]
    for key in (
        "fastgb_python_vs_compiled",
        "kernel_h_h_vs_script_real",
        "kernel_h_h_vs_script_full",
    ):
        lines.append(f"{key}: {check[key]:.2e}")
    return lines


def resolution_lines(resolution):
    """The FD codes at the truth against both FD data sets, per sample count."""
    lines = []
    for waveform, per_length in resolution.items():
        for length, entry in per_length.items():
            lines.append(
                f"FD resolution {waveform:18s} N {int(length):8d}: "
                + ", ".join(
                    f"{label} data: mismatch {1.0 - values['overlap']:.2e}, amplitude ratio "
                    f"{values['amplitude_ratio']:.6f}, phase {values['phase']:+.2e}, ln L "
                    f"{values['lnl']:.3e}"
                    for label, values in entry.items()
                )
            )
    return lines


def diagnostic_lines(record):
    """The FD window against no window, and the weight of T in both domains."""
    window, change = record["fd_window_only"], record["fd_window_change"]
    noise = record["noise_diagnostics"]
    return [
        f"FD window against no window, in band with the noise weights: mismatch "
        f"{window['mismatch']:.2e}, amplitude ratio {window['amplitude_ratio']:.8f}, phase "
        f"{window['phase']:+.2e}, residual {window['residual']:.2e}; unweighted over all bins: "
        f"change {change['change_over_rect_all_bins']:.2e} of the power, "
        f"{change['share_in_band']:.2e} of it in band",
        f"(d|d) without T over (d|d): STFT {noise['stft_share_without_t']:.4f}, FD "
        f"{noise['fd_share_without_t']:.4f}; FD with the CSD at the orbit-averaged arm lengths: "
        f"SNR {noise['fd_snr_arm_lengths_averaged']:.2f}, without T "
        f"{noise['fd_share_without_t_arm_lengths_averaged']:.4f}",
    ]


def report_lines(records):
    lines = []
    for record in records:
        lines += [
            "",
            f"{record['source']}: amplitude {record['params']['amp']:.6e} (scale "
            f"{record['amplitude_scale']:.6f}), SNR {record['snr_stft']:.3f} STFT, "
            f"{record['snr_fd']:.3f} FD; FD band {record['fd_band']['num_bins']} bins from bin "
            f"{record['fd_band']['first_bin']}; FD C^-1 Hermitian to "
            f"{record['fd_hermitian_error']:.1e}, (d|d) with Re(C^-1) / full "
            f"{record['fd_d_d_real_over_full']:.6f}",
        ]
        lines += [
            f"  {line}"
            for line in truth_lines(record)
            + resolution_lines(record["convention_check"]["fd_resolution"])
            + diagnostic_lines(record)
        ]
        if "metrics" not in record:
            continue
        lines.append(
            f"  {'parameter':10s} {'sigma':>10s} "
            + " ".join(f"{waveform:>42s}" for waveform in WAVEFORMS)
        )
        lines.append(
            f"  {'':10s} {'':>10s} "
            + " ".join(
                f"{'offset':>10s} {'width':>9s} {'lnL true':>10s} {'lnL peak':>10s}"
                for _ in WAVEFORMS
            )
        )
        for index in SCAN_INDS:
            name = PARAMETER_NAMES[index]
            cells = []
            for waveform in WAVEFORMS:
                entry = record["metrics"][waveform][name]
                mark = "*" * entry["rescans"] if entry["peak_inside_scan"] else "!"
                cells.append(
                    f"{entry['peak']:+9.3f}{mark:1s}{entry['width']:9.4f} "
                    f"{entry['lnl_truth']:10.3e} {entry['lnl_peak']:10.3e}"
                )
            lines.append(f"  {name:10s} {record['sigma'][name]:10.3e} " + " ".join(cells))
    return lines


def write_combined(output, names, no_figures):
    """summary.json, report.txt and the summary figures of every source in ``output``."""
    records = [
        json.loads((output / f"{name}.json").read_text())
        for name in names
        if (output / f"{name}.json").exists()
    ]
    if not records:
        return []
    (output / "summary.json").write_text(json.dumps(dict(sources=records), indent=1))
    with_metrics = [record for record in records if "metrics" in record]
    if not no_figures:
        if with_metrics:
            plot_summary(
                with_metrics,
                output / "summary_offsets.png",
                "peak",
                "peak offset (theta_peak - theta_true) / sigma",
                0.0,
            )
            plot_summary(
                with_metrics,
                output / "summary_widths.png",
                "width",
                "width of the scan / sigma (Fisher, DFFR)",
                1.0,
            )
        plot_levels(records, output / "summary_levels.png")
    lines = [
        f"Likelihood surfaces, SNR {SNR_TARGET} (STFT, periodic window), {DEFAULT_N_SIDE_BINS} "
        f"side bins, scans of +-{SCAN_WIDTH} sigma in {SCAN_POINTS} points; offset and width in "
        "sigma of DFFR's Fisher at the truth; * = one rescan centred on the latest peak, "
        "! = peak still outside the last scan",
    ] + report_lines(records)
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    arguments = parser.parse_args()

    chosen = make_source_set("validation")
    all_names = list(chosen.sources)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.summarize_only:
        print("\n".join(write_combined(arguments.output, all_names, arguments.no_figures)))
        return
    if arguments.truth is None:
        raise SystemExit("--truth is needed unless --summarize-only is given")
    names = arguments.sources or all_names
    unknown = sorted(set(names) - set(all_names))
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; choose from {all_names}")
    num_samples = arguments.days * int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
    # * Times count from the start of the ESA orbit file, where t_ref sits, as in stft_accuracy.py.
    t_ref = 0.0
    data_start = t_ref + chosen.start_offset
    orbits, orbits_start = load_orbits("esa", arguments.device)
    cpu_orbits, _ = load_orbits("esa", "cpu")
    if orbits_start != t_ref:
        raise SystemExit("LAT's ESA orbits no longer count times from the start of their file")
    injection = TruthInjection(arguments.truth, data_start, num_samples, arguments.device)
    log(f"device {arguments.device}, {arguments.days} days, sources {names}")

    for name in names:
        record, arrays = evaluate_source(
            name,
            chosen.sources[name],
            injection,
            orbits,
            cpu_orbits,
            t_ref,
            arguments.days,
            arguments.device,
            arguments.check_only,
        )
        np.savez_compressed(arguments.output / f"{name}.npz", **arrays)
        (arguments.output / f"{name}.json").write_text(json.dumps(record, indent=1))
        if "metrics" in record and not arguments.no_figures:
            plot_source(record, arrays, arguments.output / f"{name}.png")
        log(f"{name} done ({record['timings']})")
    if not arguments.no_summary:
        print("\n".join(write_combined(arguments.output, all_names, arguments.no_figures)))
    log("done")


if __name__ == "__main__":
    main()
