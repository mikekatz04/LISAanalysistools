"""Mismatch, SNR and likelihood surfaces of six galactic-binary waveforms on Mojito-light data.

    python mojito_dffr_validation.py --device gpu --output DIR [--mojito DIR] [--sources NAME ...]
        [--parts mismatch snr surfaces] [--no-figures] [--no-summary]
    python mojito_dffr_validation.py --output DIR --summarize-only

Step 6 of the DFFR integration, validation item 4. The data are the Mojito-light L1 time series of
the galactic binaries (the white-dwarf population of ``catalogues/wdwd_cat_mojito_lite_processed``)
and of the instrument noise, read and preprocessed as the global fit does with the settings of
``mojito_input/gb_high_freq_global_fit_settings.py``: ``L1ProcessingStep`` with the L1 orbits of
the noise file (ICRS, Mojito mean arm length), a high-pass at 1e-5 Hz and a low-pass at 0.1 Hz
(order-2 Butterworth, zero phase), downsampling to 5 s (Kaiser window, beta 31) and 200 h trimmed
from each end, then the first 1.92 years, the longest span the trim leaves (author's choice for
item 4; the settings file takes 9 months). Both domains use the 701 whole one-day segments of that
span. The processed streams of the galaxy, of the noise and of their sum are cached in DIR/data
with the chosen sources; delete that folder to build them again.

Sources: the NUM_SOURCES binaries of highest GW22 frequency, at the catalogue's amplitudes, with
the catalogue's parameters at MOJITO_REFERENCE_TIME mapped as the global fit maps them
(``gb_catalogue_to_sampling_basis`` and the stock GB transform, whose phi0 sign flip is commented
out: phi0 = +TrueAnomaly; sky angles and polarisation in ICRS). A catalogue binary whose template
band reaches a source's STFT band is a neighbour of that source; a neighbour inside the source's
side bins makes the source "overlapping".

The four waveforms of item 3 (``likelihood_surfaces.py``) and the two of jaxgb, each in its own
domain, on X, Y and Z with the full complex 3 x 3 CSD:

* STFT: DFFR and TDI on the fly in ``STFTGBComputations``, t_ref = MOJITO_REFERENCE_TIME as in the
  global fit, 60 side bins, the periodic Tukey window of taper 0.1157 on the data;
* FD: ``GBFDComputations``, FastGB, JaxGB and JaxGBaccurate on dt rfft of the data with LAT's
  symmetric Tukey of taper 1e4 s, their parameters carried to the data start (frequency and phase;
  the amplitude too for the codes whose model evolves it). FastGB runs with
  ``flip_ref_phase=True``, the conventions of item 3. ``GBFDComputations``, FastGB and JaxGB give
  the transform of the data span without a window, which no FD code can taper. JaxGBaccurate gives
  the transform of a span twice as long, [t0 - T/2, t0 + 3T/2) under its own window, which is flat
  over the data; its template goes back to the time domain, where the data span takes the FD data
  window, as the data did. Both jaxgb classes take lisaorbits orbits: ``InterpolatedOrbits`` on the
  spacecraft positions and velocities of the L1 file, whose light travel times this script compares
  with those stored in the file.

The noise weights are those of ``XYZSensitivityBackend`` at its default levels, with the options of
the settings file (TDI 2.0, transfer-function dips masked within 2 %, transfer functions averaged
over the orbit in the FD) and, as its ``normalize_window`` asks, the CSD scaled by the mean square
of the data window. Every product runs on three data sets:

* data: galaxy plus noise, the data of the global fit;
* gb: the galaxy alone;
* isolated: the galaxy alone without the source's neighbours, whose templates leave it at their
  catalogue parameters: DFFR's in the STFT; in the FD those of ``GBFDComputations`` on 16 times its
  sample count, with the FD data window applied to them. In both domains the templates first take
  the gain of the preprocessing filters at their frequencies, as the neighbours in the data did.

Parts (``--parts``, all by default):

* mismatch: every waveform at the truth against "isolated" and "gb": the mismatch
  1 - Re(d|h) / sqrt((d|d)(h|h)), the amplitude ratio sqrt((h|h)/(d|d)), the phase of (d|h), which
  checks the conventions, and ln L = -(d - h|d - h)/2; the FD codes on 4 and 16 times their sample
  count; the overlap of every neighbour's template with the source's template; the gain of the
  preprocessing at the source, whose inverse is the amplitude ratio a waveform without filters
  shows;
* snr: the optimal SNR sqrt((h|h)) of every waveform, the SNR of "isolated" and "gb" in the band,
  the matched-filter SNR Re(d|h) / sqrt((h|h)) on "data", and the noise level (n|n) / (6 N) of the
  noise alone over the N pixels (STFT) or bins (FD) of the band, 1 when the CSD describes it;
* surfaces: the scans of item 3 (8 parameters, +-4 sigma in 81 points, sigma = 1/sqrt(F_ii) of
  DFFR's information matrix at the truth, up to 3 rescans of a far peak of "data") on all three
  data sets.

Every run writes NAME.json, NAME.npz and NAME.png per source and then, unless ``--no-summary``,
summary.json, report.txt and the summary figures; ``--summarize-only`` writes only those and
draws the figures again. There is no threshold; the results are recorded in
``docs/response/dffr/tests.md``. Besides the Erebor stack the script needs jaxgb (PyPI, 0.2.2
measured) and the lisaorbits it brings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from likelihood_surfaces import (  # noqa: E402
    FASTGB_COMPILED_MAX,
    FD_BATCH_SAMPLES,
    FD_RESOLUTION_FACTORS,
    FD_TAPER_DURATION,
    PARAMETER_LABELS,
    RESCAN_LIMIT,
    RESCAN_THRESHOLD,
    SCAN_INDS,
    SCAN_POINTS,
    SCAN_WIDTH,
)
from likelihood_surfaces import WAVEFORM_COLORS as ITEM3_COLORS  # noqa: E402
from likelihood_surfaces import WAVEFORM_LABELS as ITEM3_LABELS  # noqa: E402
from likelihood_surfaces import WAVEFORMS as ITEM3_WAVEFORMS  # noqa: E402
from likelihood_surfaces import (  # noqa: E402
    at_epoch,
    fd_inner,
    fd_length,
    peak_metrics,
    release_device_memory,
    scan_row,
    style_axes,
)
from stft_accuracy import (  # noqa: E402
    AXIS,
    DEFAULT_N_SIDE_BINS,
    DOPPLER_FRACTION,
    INK,
    INK_SECONDARY,
    SURFACE,
    TDI_GENERATION,
    build_comp,
    inner,
    source_settings,
    window_samples,
)
from stft_fixtures import (  # noqa: E402
    PRODUCTION_WINDOW_ALPHA,
    SAMPLING_INTERVAL,
    SEGMENT_DURATION,
    array_module,
    load_orbits,
    run_metadata,
    to_host,
)
from td_injection import PARAMETER_NAMES  # noqa: E402

DEFAULT_MOJITO = Path("/project/lisa_nlddpc/Data/mojito_light")
NUM_SOURCES = 5
# * The preprocessing of mojito_input/gb_high_freq_global_fit_settings.py, with 1.92 years in
# * place of its 9 months: the longest span that 200 h trimmed at each end leave of 731 days.
OBSERVATION_YEARS = 1.92
PREPROCESS_KWARGS = dict(
    highpass_kwargs=dict(cutoff=1e-5, order=2, zero_phase=True),
    lowpass_kwargs=dict(cutoff=1e-1, order=2, zero_phase=True),
    trim_kwargs=dict(duration=200 * 3600, is_percent=False, trimming_type="from_each_end"),
    downsample_kwargs=dict(target_fs=1.0 / SAMPLING_INTERVAL, window=("kaiser", 31.0)),
)
#: sensitivity_init_kwargs of the same settings file.
SENSITIVITY_KWARGS = dict(tdi_generation=2, mask_percentage=0.02, average_transfer_functions=True)
#: Streams cached by the preparation: galaxy plus noise, galaxy, noise.
STREAMS = ("data", "gb", "noise")
DATA_SETS = ("data", "gb", "isolated")
DATA_SET_LABELS = {
    "data": "galaxy + noise",
    "gb": "galaxy",
    "isolated": "galaxy without neighbours",
}
#: Catalogue fields that gb_catalogue_to_sampling_basis reads.
CATALOGUE_FIELDS = (
    "Amplitude",
    "GW22FrequencySSBFrame",
    "GW22FrequencyDerivativeSourceFrame",
    "TrueAnomaly",
    "InclinationAngle",
    "PolarisationAngle",
    "RightAscension",
    "Declination",
)
#: Catalogue fields recorded per source and neighbour, besides CATALOGUE_FIELDS.
RECORDED_FIELDS = ("ChirpMassSSBFrame", "Eccentricity", "TimeReferenceSSBFrame")
# * A neighbour leaves the FD data through GBFDComputations on 16 times its own sample count,
# * where item 3 measured its sampling error at about 1e-3 in ln L at SNR 100.
NEIGHBOUR_FD_FACTOR = FD_RESOLUTION_FACTORS[-1]
#: Tolerance of the DFFR template that fill_global_stft writes against get_ll_stft's (h|h).
WRITE_PARITY_TOLERANCE = 1e-10
WAVEFORMS = ITEM3_WAVEFORMS + ("jaxgb", "jaxgb_accurate")
WAVEFORM_LABELS = {**ITEM3_LABELS, "jaxgb": "JaxGB (FD)", "jaxgb_accurate": "JaxGBaccurate (FD)"}
# * Categorical slots 5 and 6 after the four of item 3, in the fixed order of the palette.
WAVEFORM_COLORS = {**ITEM3_COLORS, "jaxgb": "#e87ba4", "jaxgb_accurate": "#008300"}
FD_WAVEFORMS = ("fd_tdi_on_the_fly", "fastgb", "jaxgb", "jaxgb_accurate")
#: Link order of lisaorbits, which jaxgb follows.
JAXGB_LINKS = (12, 23, 31, 13, 32, 21)
#: JaxGBaccurate's extension of the span on both sides, in units of the span (its default).
JAXGB_EXTENSION = 0.5


def log(message):
    print(f"[mojito] {time.strftime('%H:%M:%S')} {message}", flush=True)


def reference_time():
    # * Imported here: the recipe module pulls in the whole global fit.
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME

    return float(MOJITO_REFERENCE_TIME)


def catalogue_params(catalogue, indices):
    """(len(indices), 9) parameters of GBTDIonTheFly at MOJITO_REFERENCE_TIME, the global fit's."""
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis

    entry = {key: np.asarray(catalogue[key])[indices] for key in CATALOGUE_FIELDS}
    sampling = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    log_amp, f0_mhz, fdot, phi0, cos_iota, psi, alpha, sin_delta = sampling.T
    # ? The stock GB transform (GBSetup in globalfit/stock/erebor.py) has its phi0 sign flip
    # ? commented out, so the waveform takes phi0 = +TrueAnomaly.
    return np.stack(
        [
            np.exp(log_amp),
            f0_mhz * 1e-3,
            fdot,
            np.zeros_like(fdot),
            phi0,
            np.arccos(cos_iota),
            psi,
            alpha,
            np.arcsin(sin_delta),
        ],
        axis=-1,
    )


def catalogue_entry(catalogue, index):
    """The recorded catalogue fields of one binary, JSON-ready."""
    entry = {
        key: float(np.asarray(catalogue[key])[index]) for key in CATALOGUE_FIELDS + RECORDED_FIELDS
    }
    identifier = np.asarray(catalogue["ID"])[index]
    entry["ID"] = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
    entry["index"] = int(index)
    return entry


def track_limits(frequency, fdot, t_ref, data_start, duration):
    """Lowest and highest frequency of linear tracks over the data, element-wise."""
    first = frequency + fdot * (data_start - t_ref)
    last = frequency + fdot * (data_start + duration - t_ref)
    return np.minimum(first, last), np.maximum(first, last)


def orbit_file_check(gb_file, noise_file):
    """Largest differences between the orbits stored in the galaxy and the noise files."""
    import h5py

    result = {}
    with h5py.File(gb_file, "r") as galaxy, h5py.File(noise_file, "r") as noise:
        for group, step in (("orbits", 1), ("ltts", 1000)):
            difference = 0.0
            for key in galaxy[group]:
                if not hasattr(galaxy[group][key], "shape"):
                    continue
                # * The light travel times hold 25 million samples each: every 1000th is compared.
                first = galaxy[group][key][::step]
                second = noise[group][key][::step]
                difference = max(difference, float(np.max(np.abs(first - second))))
            result[f"{group}_max_difference"] = difference
    return result


def preprocessing_response(frequencies, raw_interval):
    """Amplitude gain of the preprocessing on a tone, at each frequency [Hz].

    Each Butterworth filter runs forward and backward (zero phase), so it scales a tone by |H|^2.
    The anti-aliasing filter of resample_poly is a symmetric FIR whose delay resample_poly removes,
    so it scales a tone by its zero-phase response.
    """
    import scipy.signal

    frequencies = np.asarray(frequencies, dtype=np.float64)
    rate = 1.0 / raw_interval
    gain = np.ones_like(frequencies)
    for key, btype in (("highpass_kwargs", "highpass"), ("lowpass_kwargs", "lowpass")):
        settings = PREPROCESS_KWARGS[key]
        if not settings["zero_phase"]:
            raise ValueError("the gain holds for zero-phase filters only")
        sos = scipy.signal.butter(
            settings["order"], settings["cutoff"], btype=btype, fs=rate, output="sos"
        )
        _, response = scipy.signal.sosfreqz(sos, worN=frequencies, fs=rate)
        gain *= np.abs(response) ** 2
    # * The filter design of scipy.signal.resample_poly for up = 1: half length 10 down, cutoff
    # * 1/down of the Nyquist frequency.
    down = int(round(rate * SAMPLING_INTERVAL))
    half_length = 10 * down
    taps = scipy.signal.firwin(
        2 * half_length + 1, 1.0 / down, window=PREPROCESS_KWARGS["downsample_kwargs"]["window"]
    )
    _, response = scipy.signal.freqz(taps, worN=frequencies, fs=rate)
    gain *= np.real(response * np.exp(2j * np.pi * frequencies / rate * half_length))
    return gain


def jaxgb_orbits(orbit_file):
    """lisaorbits' InterpolatedOrbits on the spacecraft positions and velocities of an L1 file.

    JaxGBaccurate evaluates the orbits half a span beyond each end of the data, where the light
    travel times stored in the file end; the positions cover 4.3 years.
    """
    import h5py
    from lisaorbits import InterpolatedOrbits

    with h5py.File(orbit_file, "r") as file:
        sampling = file["orbits/sampling"].attrs
        times = sampling["t0"] + sampling["dt"] * np.arange(sampling["size"])
        positions = np.stack([file[f"orbits/sc_position_{sc}"][:] for sc in (1, 2, 3)], axis=1)
        velocities = np.stack([file[f"orbits/sc_velocity_{sc}"][:] for sc in (1, 2, 3)], axis=1)
    return InterpolatedOrbits(times, positions, spacecraft_velocities=velocities)


def jaxgb_orbit_check(orbits, orbit_file):
    """Largest difference [s] between the light travel times of ``orbits`` and those of the file."""
    import h5py

    with h5py.File(orbit_file, "r") as file:
        sampling = file["ltts/sampling"].attrs
        # * Every 400 000th sample, about every 12 days of the 2.5 s grid.
        step = 400000
        times = sampling["t0"] + sampling["dt"] * np.arange(0, sampling["size"], step)
        stored = np.stack([file[f"ltts/ltt_{link}"][::step] for link in JAXGB_LINKS], axis=1)
    return float(np.max(np.abs(orbits.compute_ltt(times, JAXGB_LINKS) - stored)))


def prepare_data(mojito, folder):
    """Preprocess the streams as the global fit does, pick the sources and cache both in folder."""
    from lisatools.detector import L1Orbits
    from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
    from lisatools.globalfit.recipe import MOJITO_AVERAGE_ARMLENGTH
    from lisatools.utils.constants import YRSID_SI

    started = time.time()
    folder.mkdir(parents=True, exist_ok=True)
    t_ref = reference_time()
    loader = L1ProcessingStep(
        L1_folder=str(mojito),
        source_types=["gb", "noise"],
        source_ids={},
        verbose=False,
        do_plots=False,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend="cpu", frame="icrs", armlength=MOJITO_AVERAGE_ARMLENGTH),
        store_individual_timeseries=True,
    )
    log(f"loaded the streams and the catalogue ({time.time() - started:.0f} s)")
    raw_interval = float(loader.dt)
    loader.process(**PREPROCESS_KWARGS, Tobs=OBSERVATION_YEARS * YRSID_SI)
    log(f"preprocessed ({time.time() - started:.0f} s)")
    times = np.asarray(loader.times)
    sample_interval = float(times[1] - times[0])
    if abs(sample_interval - SAMPLING_INTERVAL) > 1e-9:
        raise RuntimeError(f"the preprocessed data are sampled every {sample_interval} s")
    samples_per_segment = int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
    num_segments = len(times) // samples_per_segment
    num_samples = num_segments * samples_per_segment
    series = loader.individual_timeseries
    streams = dict(data=series["COMBINED"], gb=series["GB_0"], noise=series["NOISE"])
    for key, stream in streams.items():
        np.save(folder / f"{key}.npy", np.ascontiguousarray(stream[:, :num_samples]))
    # * The filters are linear, so the processed sum must equal the sum of the processed parts.
    sum_check = float(
        np.max(np.abs(streams["data"] - streams["gb"] - streams["noise"]))
        / np.max(np.abs(streams["data"]))
    )
    data_start = float(times[0])
    duration = num_samples * SAMPLING_INTERVAL
    gb_file = find_file(str(mojito / "data" / "GB" / "L1"), "GB", 0)
    noise_file = find_file(str(mojito / "data" / "INSTRUMENT" / "L1"), "NOISE", 0)
    layout = dict(
        data_start=data_start,
        t_ref=t_ref,
        num_segments=num_segments,
        num_samples=num_samples,
        sample_interval=SAMPLING_INTERVAL,
        raw_sample_interval=raw_interval,
        observation_years=OBSERVATION_YEARS,
        preprocess_kwargs=PREPROCESS_KWARGS,
        processed_samples=int(len(times)),
        sum_check=sum_check,
        gb_file=gb_file,
        noise_file=noise_file,
        orbit_file=noise_file,
        orbit_file_check=orbit_file_check(gb_file, noise_file),
    )
    series.clear()
    del streams

    catalogue = loader.catalogue["GB"][0]
    frequency = np.asarray(catalogue["GW22FrequencySSBFrame"], dtype=np.float64)
    fdot = np.asarray(catalogue["GW22FrequencyDerivativeSourceFrame"], dtype=np.float64)
    amplitude = np.asarray(catalogue["Amplitude"], dtype=np.float64)
    if np.any(np.asarray(catalogue["TimeReferenceSSBFrame"]) != t_ref):
        raise RuntimeError("the catalogue holds parameters at another time than t_ref")
    low, high = track_limits(frequency, fdot, t_ref, data_start, duration)
    stft_df = 1.0 / SEGMENT_DURATION
    # * A binary's template spans its track with the side bins and its Doppler shift.
    reach = (DEFAULT_N_SIDE_BINS + np.ceil(DOPPLER_FRACTION * high / stft_df)) * stft_df
    ranked = np.argsort(frequency)[::-1][:NUM_SOURCES]
    names = {
        int(index): f"gb{rank}_{frequency[index] * 1e3:.3f}mHz" for rank, index in enumerate(ranked)
    }
    params = catalogue_params(catalogue, ranked)
    records = []
    for rank, index in enumerate(ranked):
        settings = source_settings(
            params[rank], data_start, num_segments, t_ref, DEFAULT_N_SIDE_BINS, "cpu"
        )
        band_low = (settings.ind_min - 0.5) * stft_df
        band_high = (settings.ind_min + settings.NF_active - 0.5) * stft_df
        reaching = (high + reach >= band_low) & (low - reach <= band_high)
        reaching[index] = False
        neighbour_inds = np.flatnonzero(reaching)
        neighbour_params = catalogue_params(catalogue, neighbour_inds)
        middle = data_start + 0.5 * duration - t_ref
        carrier = frequency[index] + fdot[index] * middle
        neighbours = []
        for position, other in enumerate(neighbour_inds):
            gap = (frequency[other] + fdot[other] * middle - carrier) / stft_df
            neighbours.append(
                dict(
                    catalogue=catalogue_entry(catalogue, other),
                    params=neighbour_params[position].tolist(),
                    gap_bins=float(gap),
                    amplitude_ratio=float(amplitude[other] / amplitude[index]),
                    selected_as=names.get(int(other)),
                )
            )
        records.append(
            dict(
                name=names[int(index)],
                rank=rank,
                catalogue=catalogue_entry(catalogue, index),
                params=params[rank].tolist(),
                stft_band=dict(first_bin=int(settings.ind_min), num_bins=int(settings.NF_active)),
                neighbours=neighbours,
                # * A neighbour whose carrier lies within the side bins sits inside the template.
                overlapping=bool(
                    any(abs(entry["gap_bins"]) <= DEFAULT_N_SIDE_BINS for entry in neighbours)
                ),
            )
        )
    layout["timings"] = dict(prepare=round(time.time() - started, 1))
    (folder / "layout.json").write_text(json.dumps(layout, indent=1))
    (folder / "sources.json").write_text(json.dumps(dict(sources=records), indent=1))
    log(f"chose {[record['name'] for record in records]} ({time.time() - started:.0f} s)")


class MojitoSource:
    """Data sets, neighbours, waveforms and likelihoods of one source in both domains."""

    def __init__(self, record, streams, fd_window, orbits, cpu_orbits, lisa_orbits, layout, device):
        from gbgpu.gbcomps import GBFDComputations
        from gbgpu.gbgpu import GBGPU

        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.domains import FDSettings, STFTSignal, TDSettings, TDSignal
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sensitivity import XYZSensitivityBackend

        self.name, self.device = record["name"], device
        self.xp = array_module(device)
        self.timings = {}
        started = time.time()
        self.t_ref = layout["t_ref"]
        self.params = np.asarray(record["params"], dtype=np.float64)
        neighbours = record["neighbours"]
        neighbour_params = np.array([entry["params"] for entry in neighbours]).reshape(-1, 9)
        data_start, num_segments = layout["data_start"], layout["num_segments"]
        samples_per_segment = int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
        num_samples = layout["num_samples"]
        raw_interval = layout["raw_sample_interval"]
        self.num_samples, self.fd_window, self.lisa_orbits = num_samples, fd_window, lisa_orbits
        self.jaxgb_models = {}
        middle = data_start + 0.5 * num_samples * SAMPLING_INTERVAL - self.t_ref
        self.preprocessing_gain = float(
            preprocessing_response([self.params[1] + self.params[2] * middle], raw_interval)[0]
        )
        time_settings = TDSettings(num_samples, SAMPLING_INTERVAL, data_start, force_backend=device)
        signals = {key: TDSignal(streams[key], settings=time_settings) for key in STREAMS}

        # ---- STFT: domain, noise weights and the transforms of the streams ----
        settings = source_settings(
            self.params, data_start, num_segments, self.t_ref, DEFAULT_N_SIDE_BINS, device
        )
        self.stft_df = settings.df
        self.stft_band = (int(settings.ind_min), int(settings.NF_active))
        segment_window = self.xp.asarray(
            window_samples("periodic", samples_per_segment, PRODUCTION_WINDOW_ALPHA)
        )

        def stft_sensitivity():
            sensitivity = XYZSensitivityBackend(
                orbits=orbits,
                settings=settings,
                force_backend=device,
                window_values=segment_window,
                **SENSITIVITY_KWARGS,
            )
            sensitivity.set_sensitivity_matrix()
            return sensitivity

        def stft_group(signals_of_sets):
            container_array = AnalysisContainerArray(
                [AnalysisContainer(signal, stft_sensitivity()) for signal in signals_of_sets],
                gpus=None if device == "cpu" else [0],
                domain_group_kwargs=dict(
                    tdi_type="XYZ",
                    window_alpha=PRODUCTION_WINDOW_ALPHA,
                    use_midpoint=True,
                    linear_envelope=True,
                ),
            )
            group = container_array.cpp_splits[0]
            group.compute_d_d_term()
            return container_array, group

        stft = {key: signals[key].stft(window=segment_window, settings=settings) for key in STREAMS}
        self.stft_inverse_covariance = to_host(stft_sensitivity().invC)

        def stft_product(first, second):
            return inner(first, second, self.stft_inverse_covariance, self.stft_df)

        # * DFFR writes the templates of the source and of its neighbours on the grid of a first
        # * group; the neighbours' templates then leave the galaxy stream.
        _, scratch_group = stft_group([stft["gb"]])
        writer = build_comp(
            scratch_group,
            orbits,
            self.t_ref,
            num_segments,
            DEFAULT_N_SIDE_BINS,
            PRODUCTION_WINDOW_ALPHA,
            "dffr",
            device,
        )
        written = self.xp.zeros(
            (len(neighbours) + 1, 3, settings.NT, settings.NF_active), dtype=self.xp.complex128
        )
        writer.fill_global_stft(
            self.xp.asarray(np.concatenate([self.params[None], neighbour_params])),
            written,
            data_index=self.xp.arange(len(neighbours) + 1, dtype=self.xp.int32),
        )
        writer.get_ll_stft(self.xp.asarray(self.params[None]))
        source_template = to_host(written[0])
        source_power = float(stft_product(source_template, source_template).real)
        comp_d_h = complex(to_host(writer.d_h_out)[0])
        own_d_h = stft_product(to_host(stft["gb"].arr), source_template)
        # * Which side of (d|h) the comp conjugates is recorded, not assumed.
        same = abs(own_d_h - comp_d_h) / abs(comp_d_h)
        conjugate = abs(own_d_h - comp_d_h.conjugate()) / abs(comp_d_h)
        self.write_parity = dict(
            h_h=abs(source_power / float(to_host(writer.h_h_out)[0].real) - 1.0),
            d_h=min(same, conjugate),
            d_h_conjugated=bool(conjugate < same),
        )
        # ! The isolated data rest on these templates, so their scale must be that of get_ll_stft.
        if max(self.write_parity["h_h"], self.write_parity["d_h"]) > WRITE_PARITY_TOLERANCE:
            raise RuntimeError(f"{self.name}: fill_global_stft disagrees with get_ll_stft")
        # * The neighbours in the data passed the preprocessing filters, so their templates take
        # * its gain before they leave the galaxy stream.
        stft_gain = preprocessing_response(
            (settings.ind_min + np.arange(settings.NF_active)) * self.stft_df, raw_interval
        )
        isolated = stft["gb"].arr - self.xp.sum(written[1:], axis=0) * self.xp.asarray(stft_gain)
        stft["isolated"] = STFTSignal(isolated, settings)
        self.container_array, self.group = stft_group([stft[key] for key in DATA_SETS])
        self.stft_d_d = dict(zip(DATA_SETS, map(float, to_host(self.group.d_d).real)))
        noise = to_host(stft["noise"].arr)
        pixels = int(np.count_nonzero(self.stft_inverse_covariance[0, 0]))
        self.stft_noise_level = float(stft_product(noise, noise).real) / (6.0 * pixels)
        self.neighbours = []
        for position, entry in enumerate(neighbours):
            template = to_host(written[position + 1])
            power = float(stft_product(template, template).real)
            cross = stft_product(template, source_template)
            self.neighbours.append(
                dict(
                    index=entry["catalogue"]["index"],
                    selected_as=entry["selected_as"],
                    gap_bins=entry["gap_bins"],
                    amplitude_ratio=entry["amplitude_ratio"],
                    stft=dict(
                        power_in_band=power,
                        share_of_gb=power / self.stft_d_d["gb"],
                        cross_term=float(cross.real),
                        overlap=(
                            float(cross.real / np.sqrt(power * source_power))
                            if power > 0.0
                            else 0.0
                        ),
                    ),
                )
            )
        self.comps = {
            response: build_comp(
                self.group,
                orbits,
                self.t_ref,
                num_segments,
                DEFAULT_N_SIDE_BINS,
                PRODUCTION_WINDOW_ALPHA,
                response,
                device,
            )
            for response in ("dffr", "tdi_on_the_fly")
        }
        del stft, written, scratch_group, writer
        release_device_memory(self.xp)

        # ---- FD: band, the FD codes and the transforms of the streams ----
        df = 1.0 / (num_samples * SAMPLING_INTERVAL)
        self.fd_df = df
        self.fd_length, self.fd_centre = fd_length(
            self.params, data_start, num_samples, self.t_ref, df
        )
        self.fd_first = self.fd_centre - self.fd_length // 2
        num_rfft = num_samples // 2 + 1
        # * Half-bin edges, so that the band holds exactly the N bins of the FD codes.
        band = dict(
            min_freq=(self.fd_first - 0.5) * df,
            max_freq=(self.fd_first + self.fd_length - 0.5) * df,
        )
        fd_settings = FDSettings(N=num_rfft, df=df, force_backend=device, **band)
        fd = {
            key: to_host(signals[key].fft(settings=fd_settings, window=fd_window).arr)
            for key in STREAMS
        }
        if fd["data"].shape != (3, self.fd_length):
            raise RuntimeError(f"{self.name}: the FD band is not the band of the FD codes")
        cpu_settings = FDSettings(N=num_rfft, df=df, force_backend="cpu", **band)
        cpu_sensitivity = XYZSensitivityBackend(
            orbits=cpu_orbits,
            settings=cpu_settings,
            force_backend="cpu",
            window_values=to_host(fd_window[0]),
            **SENSITIVITY_KWARGS,
        )
        cpu_sensitivity.set_sensitivity_matrix()
        self.fd_inverse_covariance = np.asarray(cpu_sensitivity.invC)
        self.data_start = data_start
        self.fd_shift = data_start - self.t_ref
        self.fd_duration = num_samples * SAMPLING_INTERVAL
        # ! The heterodyne requires t_start = t_ref: its parameters refer to the data start.
        self.fd_comp = GBFDComputations(
            cpu_settings,
            data_start,
            N_sparse=self.fd_length,
            orbits=cpu_orbits,
            tdi_config=TDIConfig(TDI_GENERATION, force_backend="cpu"),
            force_backend="cpu",
            tdi_type="XYZ",
            nchannels=3,
            tukey_alpha=0.0,
        )
        # ? FastGB's phase is +phi0 + 2 pi f0 xi + ... from its t0 on, LAT's -phi0 + 2 pi f0 tau +
        # ? ...; flip_ref_phase negates phi0, and the check at the truth confirms the mapping.
        self.fastgb = GBGPU(orbits=orbits, force_backend=device, t0=data_start, flip_ref_phase=True)

        # * The neighbours whose own template band reaches this band leave the FD galaxy stream,
        # * with the gain of the preprocessing, as in the STFT.
        source_fd = self.fd_tdi_templates(self.params[None])[0]
        source_fd_power = float(self.fd_product(source_fd, source_fd).real)
        fd_gain = preprocessing_response(
            (self.fd_first + np.arange(self.fd_length)) * df, raw_interval
        )
        removed = np.zeros_like(fd["gb"])
        for position, params in enumerate(neighbour_params):
            own_length, own_centre = fd_length(params, data_start, num_samples, self.t_ref, df)
            gap = abs(own_centre - self.fd_centre)
            if gap + self.fd_length // 2 > NEIGHBOUR_FD_FACTOR * own_length // 2:
                self.neighbours[position]["fd"] = None
                continue
            template = fd_gain * self.windowed_fd_template(
                params, NEIGHBOUR_FD_FACTOR * own_length, own_centre, num_samples
            )
            removed += template
            power = float(self.fd_product(template, template).real)
            cross = self.fd_product(template, source_fd)
            self.neighbours[position]["fd"] = dict(
                gap_bins=int(gap),
                power_in_band=power,
                cross_term=float(cross.real),
                overlap=(
                    float(cross.real / np.sqrt(power * source_fd_power)) if power > 0.0 else 0.0
                ),
            )
        fd["isolated"] = fd["gb"] - removed
        self.fd_data = {key: fd[key] for key in DATA_SETS}
        self.fd_d_d = {key: float(self.fd_product(fd[key], fd[key]).real) for key in DATA_SETS}
        for entry in self.neighbours:
            if entry["fd"] is not None:
                entry["fd"]["share_of_gb"] = entry["fd"]["power_in_band"] / self.fd_d_d["gb"]
        bins = int(np.count_nonzero(self.fd_inverse_covariance[0, 0]))
        self.fd_noise_level = float(self.fd_product(fd["noise"], fd["noise"]).real) / (6.0 * bins)
        del fd, signals
        release_device_memory(self.xp)
        self.timings["data"] = round(time.time() - started, 1)

    def fd_product(self, first, second):
        return fd_inner(first, second, self.fd_inverse_covariance, self.fd_df)

    def windowed_template(self, spectrum, first, span_samples, offset):
        """A template with the FD data window, on this FD band.

        ``spectrum`` (3, N) holds dt rfft bins ``first`` to ``first + N`` of a span of
        ``span_samples`` samples; back in the time domain the data span, from sample ``offset``
        on, takes the window, as the data did.
        """
        xp = self.xp
        length = spectrum.shape[-1]
        if first < 0 or first + length > span_samples // 2 + 1:
            raise RuntimeError(f"{self.name}: an FD template leaves the rfft grid of its span")
        full = xp.zeros((3, span_samples // 2 + 1), dtype=xp.complex128)
        full[:, first : first + length] = xp.asarray(spectrum)
        # * dt rfft is the convention of the data and the FD codes, so the inverse divides by dt.
        series = xp.fft.irfft(full / SAMPLING_INTERVAL, n=span_samples, axis=-1)
        series = series[:, offset : offset + self.num_samples]
        windowed = SAMPLING_INTERVAL * xp.fft.rfft(series * self.fd_window, axis=-1)
        return to_host(windowed[:, self.fd_first : self.fd_first + self.fd_length])

    def windowed_fd_template(self, params, length, centre, num_samples):
        """A binary's GBFDComputations template, on ``length`` samples about its own carrier bin
        ``centre``, with the FD data window, on this FD band."""
        first = centre - length // 2
        template = np.zeros((1, 3, length), dtype=np.complex128)
        self.fd_comp.fill_global(
            at_epoch(params[None], self.fd_shift),
            template,
            data_index=np.zeros(1, dtype=np.int32),
            template_start_inds=np.array([first], dtype=np.int32),
            N_sparse=length,
        )
        return self.windowed_template(template[0], first, num_samples, 0)

    # ---- likelihood terms: complex (d|h) per data set and real (h|h) per point ----

    def stft_terms(self, response, points):
        comp = self.comps[response]
        num = points.shape[0]
        d_h = {}
        for index, key in enumerate(DATA_SETS):
            selector = self.xp.full(num, index, dtype=self.xp.int32)
            comp.get_ll_stft(self.xp.asarray(points), data_index=selector, noise_index=selector)
            d_h[key] = to_host(comp.d_h_out).astype(complex)
        return d_h, to_host(comp.h_h_out).real

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

    def fastgb_templates(self, points, length=None):
        """FastGB templates on the FD band, (num, 3, N), sampled on ``length``."""
        length = self.fd_length if length is None else length
        mapped = at_epoch(points, self.fd_shift)
        # ! run_wave takes NumPy arrays (np.atleast_1d) and moves them to its backend itself.
        self.fastgb.run_wave(
            *mapped.T,
            N=length,
            T=self.fd_duration,
            dt=SAMPLING_INTERVAL,
            tdi2=True,
            tdi_channel_setup="XYZ",
            use_c_implementation=length <= FASTGB_COMPILED_MAX,
        )
        if np.any(to_host(self.fastgb.start_inds) != self.fd_centre - length // 2):
            raise RuntimeError(f"{self.name}: FastGB templates left the FD band")
        middle = length // 2 - self.fd_length // 2
        return to_host(self.fastgb.XYZf)[:, :, middle : middle + self.fd_length]

    def jaxgb_params(self, points):
        """jaxgb's (num, 8) parameters at the data start: f0, fdot, A, ra, dec, psi, iota, phi0.

        Frequency and phase are carried from t_ref as for the other FD codes; the amplitude is not,
        since jaxgb has no amplitude evolution (as ``GBObject.to_jaxgb_array``). The sky angles and
        the polarisation are those of the ICRS frame of the orbits.
        """
        mapped = at_epoch(points, self.fd_shift)
        return np.stack(
            [
                mapped[:, 1],
                mapped[:, 2],
                points[:, 0],
                points[:, 7],
                points[:, 8],
                points[:, 6],
                points[:, 5],
                mapped[:, 4],
            ],
            axis=-1,
        )

    def jaxgb_model(self, accurate, size):
        """JaxGB or JaxGBaccurate over the data span, with ``size`` points of slow response."""
        key = (accurate, size)
        if key not in self.jaxgb_models:
            from jaxgb.jaxgb import JaxGB, JaxGBaccurate

            settings = dict(orbits=self.lisa_orbits, t_obs=self.fd_duration, t0=self.data_start)
            self.jaxgb_models[key] = (
                JaxGBaccurate(extension=JAXGB_EXTENSION, n=size, **settings)
                if accurate
                else JaxGB(n=size, **settings)
            )
        return self.jaxgb_models[key]

    def jaxgb_tdi(self, model, points):
        """X2, Y2, Z2 of a jaxgb model, (num, 3, n), and the first bin of every template."""
        import jax.numpy as jnp

        params = jnp.asarray(self.jaxgb_params(points))
        channels = model.get_tdi(params, tdi_generation=2, tdi_combination="XYZ")
        spectra = np.stack([np.asarray(channel) for channel in channels], axis=1)
        return spectra, np.asarray(model.get_kmin(params[:, 0]))

    def jaxgb_templates(self, points, length=None):
        """JaxGB templates on the FD band, (num, 3, N), sampled on ``length``."""
        length = self.fd_length if length is None else length
        spectra, starts = self.jaxgb_tdi(self.jaxgb_model(False, length), points)
        if np.any(starts != self.fd_centre - length // 2):
            raise RuntimeError(f"{self.name}: JaxGB templates left the FD band")
        middle = length // 2 - self.fd_length // 2
        return spectra[:, :, middle : middle + self.fd_length]

    def jaxgb_accurate_templates(self, points, length=None):
        """JaxGBaccurate templates with the FD data window, on the FD band, (num, 3, N).

        Its transform covers [t0 - e T, t0 + (1 + e) T) with e its extension, on bins of
        1 / ((1 + 2e) T), and takes that span's start as its time origin; ``length`` sets its
        bandwidth as the sample count does for the other FD codes.
        """
        length = self.fd_length if length is None else length
        model = self.jaxgb_model(True, int(round((1 + 2 * JAXGB_EXTENSION) * length)))
        span_samples = int(round(model.t_ext / SAMPLING_INTERVAL))
        offset = int(round(model.extension * model.t_obs / SAMPLING_INTERVAL))
        spectra, starts = self.jaxgb_tdi(model, points)
        templates = np.empty((points.shape[0], 3, self.fd_length), dtype=np.complex128)
        for index in range(points.shape[0]):
            templates[index] = self.windowed_template(
                spectra[index], int(starts[index]), span_samples, offset
            )
        return templates

    def fd_builder(self, waveform):
        return dict(
            fd_tdi_on_the_fly=self.fd_tdi_templates,
            fastgb=self.fastgb_templates,
            jaxgb=self.jaxgb_templates,
            jaxgb_accurate=self.jaxgb_accurate_templates,
        )[waveform]

    def fd_terms(self, waveform, points):
        build = self.fd_builder(waveform)
        batch = max(1, FD_BATCH_SAMPLES // self.fd_length)
        d_h, h_h = {key: [] for key in DATA_SETS}, []
        for first in range(0, points.shape[0], batch):
            templates = build(points[first : first + batch])
            for key in DATA_SETS:
                d_h[key].append(self.fd_product(self.fd_data[key][None], templates))
            h_h.append(self.fd_product(templates, templates).real)
        return {key: np.concatenate(values) for key, values in d_h.items()}, np.concatenate(h_h)

    def terms(self, waveform, points):
        started = time.time()
        if waveform in ("dffr", "tdi_on_the_fly"):
            result = self.stft_terms(waveform, points)
        else:
            result = self.fd_terms(waveform, points)
        self.timings[waveform] = round(self.timings.get(waveform, 0.0) + time.time() - started, 1)
        return result

    def d_d(self, waveform, key):
        return (self.stft_d_d if waveform in ("dffr", "tdi_on_the_fly") else self.fd_d_d)[key]

    def log_likelihoods(self, waveform, points):
        d_h, h_h = self.terms(waveform, points)
        return {key: d_h[key].real - 0.5 * h_h - 0.5 * self.d_d(waveform, key) for key in DATA_SETS}

    def sigma(self):
        """Conditional widths 1/sqrt(F_ii) and the Fisher matrix of DFFR at the truth."""
        fisher = to_host(
            self.comps["dffr"].information_matrix(
                self.xp.asarray(self.params[None]), inds=list(SCAN_INDS)
            )
        )[0]
        return 1.0 / np.sqrt(np.diag(fisher)), fisher

    def truth_check(self):
        """Every waveform at the truth: mismatch, amplitude, phase, ln L and SNRs."""
        record = {}
        for waveform in WAVEFORMS:
            d_h, h_h = self.terms(waveform, self.params[None])
            h_h = float(h_h[0])
            entry = dict(
                optimal_snr=float(np.sqrt(h_h)),
                matched_filter_snr=float(d_h["data"][0].real / np.sqrt(h_h)),
            )
            for key in ("isolated", "gb"):
                d_d = self.d_d(waveform, key)
                value = complex(d_h[key][0])
                entry[key] = dict(
                    mismatch=float(1.0 - value.real / np.sqrt(d_d * h_h)),
                    amplitude_ratio=float(np.sqrt(h_h / d_d)),
                    phase=float(np.angle(value)),
                    lnl=float(value.real - 0.5 * h_h - 0.5 * d_d),
                )
            record[waveform] = entry
        return record

    def fd_resolution(self):
        """The FD codes at the truth on finer samplings, cut to the band, against "isolated"."""
        data, d_d = self.fd_data["isolated"], self.fd_d_d["isolated"]
        result = {}
        for waveform in FD_WAVEFORMS:
            result[waveform] = {}
            for factor in FD_RESOLUTION_FACTORS:
                template = self.fd_builder(waveform)(
                    self.params[None], length=factor * self.fd_length
                )[0]
                h_h = float(self.fd_product(template, template).real)
                d_h = complex(self.fd_product(data, template))
                result[waveform][str(factor * self.fd_length)] = dict(
                    mismatch=float(1.0 - d_h.real / np.sqrt(d_d * h_h)),
                    amplitude_ratio=float(np.sqrt(h_h / d_d)),
                    phase=float(np.angle(d_h)),
                    lnl=float(d_h.real - 0.5 * h_h - 0.5 * d_d),
                )
        release_device_memory(self.xp)
        return result


def scan_waveform(source, waveform, sigma, arrays):
    """Scans of every parameter for one waveform on the three data sets; returns their metrics.

    A far peak of "data" gets rescans centred on it, which all data sets share; each data set's
    peak comes from the scan that holds its largest value.
    """
    rows = [scan_row(source.params, sigma, row, 0.0) for row in range(len(SCAN_INDS))]
    flat = np.concatenate([points for points, _, _ in rows])
    lnl = {
        key: values.reshape(len(SCAN_INDS), SCAN_POINTS)
        for key, values in source.log_likelihoods(waveform, flat).items()
    }
    metrics = {key: {} for key in DATA_SETS}
    for row, (_, offsets, valid) in enumerate(rows):
        name = PARAMETER_NAMES[SCAN_INDS[row]]
        for key in DATA_SETS:
            lnl[key][row, ~valid] = np.nan
        curves = {key: [(offsets, lnl[key][row])] for key in DATA_SETS}
        entry = peak_metrics(offsets, lnl["data"][row])
        centre, rescans = 0.0, 0
        while abs(entry["peak"] - centre) > RESCAN_THRESHOLD and rescans < RESCAN_LIMIT:
            centre, rescans = round(entry["peak"], 1), rescans + 1
            points, extra_offsets, extra_valid = scan_row(source.params, sigma, row, centre)
            extra = source.log_likelihoods(waveform, points)
            arrays[f"rescan{rescans}_offsets_{waveform}_{name}"] = extra_offsets
            for key in DATA_SETS:
                extra[key][~extra_valid] = np.nan
                arrays[f"rescan{rescans}_lnl_{key}_{waveform}_{name}"] = extra[key]
                curves[key].append((extra_offsets, extra[key]))
            entry = peak_metrics(extra_offsets, extra["data"])
        for key in DATA_SETS:
            best_offsets, best_lnl = max(
                curves[key],
                key=lambda curve: (np.nanmax(curve[1]) if np.isfinite(curve[1]).any() else -np.inf),
            )
            result = peak_metrics(best_offsets, best_lnl)
            result.update(
                lnl_truth=float(lnl[key][row, SCAN_POINTS // 2]),
                rescans=rescans,
                peak_inside_scan=bool(best_offsets.min() <= result["peak"] <= best_offsets.max()),
            )
            metrics[key][name] = result
    for key in DATA_SETS:
        arrays[f"lnl_{key}_{waveform}"] = lnl[key]
    release_device_memory(source.xp)
    if waveform == WAVEFORMS[0]:
        arrays["offsets"] = np.stack([offsets for _, offsets, _ in rows])
    return metrics


def evaluate_source(record, streams, fd_window, all_orbits, layout, device, parts):
    """The requested parts for one source; returns (record, arrays).

    ``all_orbits`` holds LAT's L1 orbits on the device and on the CPU, lisaorbits' orbits for
    jaxgb, and the check of the latter against the light travel times of the file.
    """
    started = time.time()
    orbits, cpu_orbits, lisa_orbits, lisa_orbits_check = all_orbits
    source = MojitoSource(
        record, streams, fd_window, orbits, cpu_orbits, lisa_orbits, layout, device
    )
    sigma, fisher = source.sigma()
    result = dict(
        source=record["name"],
        metadata=run_metadata(device, mojito_l1=layout["orbit_file"]),
        rank=record["rank"],
        catalogue=record["catalogue"],
        overlapping=record["overlapping"],
        params=dict(zip(PARAMETER_NAMES, record["params"])),
        sigma=dict(zip((PARAMETER_NAMES[index] for index in SCAN_INDS), map(float, sigma))),
        stft_band=dict(first_bin=source.stft_band[0], num_bins=source.stft_band[1]),
        fd_band=dict(first_bin=int(source.fd_first), num_bins=int(source.fd_length)),
        write_parity=source.write_parity,
        preprocessing_gain=source.preprocessing_gain,
        lisa_orbits_ltt_max_difference=lisa_orbits_check,
        neighbours=source.neighbours,
        d_d=dict(stft=source.stft_d_d, fd=source.fd_d_d),
    )
    arrays = dict(params=source.params, sigma=sigma, fisher=fisher)
    if "mismatch" in parts or "snr" in parts:
        check = source.truth_check()
        if "mismatch" in parts:
            result["mismatch"] = {
                waveform: {key: check[waveform][key] for key in ("isolated", "gb")}
                for waveform in WAVEFORMS
            }
            result["fd_resolution"] = source.fd_resolution()
        if "snr" in parts:
            result["snr"] = dict(
                optimal={waveform: check[waveform]["optimal_snr"] for waveform in WAVEFORMS},
                matched_filter={
                    waveform: check[waveform]["matched_filter_snr"] for waveform in WAVEFORMS
                },
                data={
                    domain: {key: float(np.sqrt(values[key])) for key in ("isolated", "gb")}
                    for domain, values in (("stft", source.stft_d_d), ("fd", source.fd_d_d))
                },
                noise_level=dict(stft=source.stft_noise_level, fd=source.fd_noise_level),
            )
        for line in truth_lines(result):
            log(f"{record['name']} {line}")
    if "surfaces" in parts:
        result["metrics"] = {
            waveform: scan_waveform(source, waveform, sigma, arrays) for waveform in WAVEFORMS
        }
        within = np.abs(arrays["offsets"]) <= 2.0 + 1e-9
        for waveform in WAVEFORMS[1:]:
            for key in DATA_SETS:
                difference = np.abs(arrays[f"lnl_{key}_{waveform}"] - arrays[f"lnl_{key}_dffr"])
                for row, index in enumerate(SCAN_INDS):
                    result["metrics"][waveform][key][PARAMETER_NAMES[index]][
                        "max_difference_to_dffr_2sigma"
                    ] = float(np.nanmax(np.where(within[row], difference[row], np.nan)))
    result["timings"] = dict(source.timings, total=round(time.time() - started, 1))
    return result, arrays


def plot_source(record, arrays, path):
    """ln L minus each curve's own peak along every scan, for "data" and "isolated"."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), facecolor=SURFACE)
    offsets = arrays["offsets"]
    # * Drawn back to front; "data" solid, "isolated" dashed. The STFT curves coincide, so TDI on
    # * the fly lies wide under a thin DFFR line and both stay visible.
    order = ("jaxgb", "jaxgb_accurate", "fastgb", "fd_tdi_on_the_fly", "tdi_on_the_fly", "dffr")
    styles = {"data": "-", "isolated": "--"}
    widths = {"tdi_on_the_fly": 2.6}
    for row, (axis, index) in enumerate(zip(axes.flat, SCAN_INDS)):
        name = PARAMETER_NAMES[index]
        low, high = -SCAN_WIDTH, SCAN_WIDTH
        for waveform in order:
            metrics = record["metrics"][waveform]
            rescans = metrics["data"][name]["rescans"]
            for key, style in styles.items():
                curves = [(offsets[row], arrays[f"lnl_{key}_{waveform}"][row])] + [
                    (
                        arrays[f"rescan{k}_offsets_{waveform}_{name}"],
                        arrays[f"rescan{k}_lnl_{key}_{waveform}_{name}"],
                    )
                    for k in range(1, rescans + 1)
                ]
                for x_values, lnl in curves:
                    axis.plot(
                        x_values,
                        lnl - metrics[key][name]["lnl_peak"],
                        color=WAVEFORM_COLORS[waveform],
                        linewidth=widths.get(waveform, 1.0 if key == "data" else 0.8),
                        linestyle=style,
                    )
                    low, high = min(low, x_values.min()), max(high, x_values.max())
        reference = np.linspace(low, high, 401)
        axis.plot(reference, -0.5 * reference * reference, color=INK, linewidth=0.8, linestyle=":")
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
    # * Colour names the waveform and line style the data set, which keeps the legend to two rows.
    handles = [
        Line2D(
            [],
            [],
            color=WAVEFORM_COLORS[waveform],
            linewidth=widths.get(waveform, 1.0),
            label=WAVEFORM_LABELS[waveform],
        )
        for waveform in WAVEFORMS
    ] + [
        Line2D(
            [], [], color=INK_SECONDARY, linewidth=1.0, linestyle=style, label=DATA_SET_LABELS[key]
        )
        for key, style in styles.items()
    ]
    handles.append(
        Line2D([], [], color=INK, linewidth=0.8, linestyle=":", label="Fisher parabola of DFFR")
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    params = record["params"]
    snr = record.get("snr", {}).get("optimal", {}).get("dffr")
    figure.suptitle(
        f"{record['source']} ({'overlapping' if record['overlapping'] else 'isolated'}): "
        f"f0 = {params['f0']:.6g} Hz, fdot = {params['fdot']:.3g} Hz/s"
        + (f", SNR {snr:.1f} (DFFR)" if snr is not None else ""),
        fontsize=11,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_summary(records, path, key, metric, ylabel, reference_value):
    """One panel per source: the metric of every waveform along each parameter, on data set key."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(records)
    figure, axes = plt.subplots(
        1, columns, figsize=(3.6 * columns, 3.8), facecolor=SURFACE, sharey=True, squeeze=False
    )
    names = [PARAMETER_NAMES[index] for index in SCAN_INDS]
    positions = np.arange(len(names))
    shifts = np.linspace(-0.24, 0.24, len(WAVEFORMS))
    for axis, record in zip(axes.flat, records):
        axis.axhline(reference_value, color=AXIS, linewidth=1.0)
        for shift, waveform in zip(shifts, WAVEFORMS):
            values = [record["metrics"][waveform][key][name][metric] for name in names]
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
        flag = " (overlapping)" if record["overlapping"] else ""
        axis.set_title(f"{record['source']}{flag}", fontsize=9, color=INK)
        style_axes(axis)
    axes[0, 0].set_ylabel(f"{ylabel}\n{DATA_SET_LABELS[key]}", fontsize=8, color=INK_SECONDARY)
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
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_mismatch(records, path):
    """Mismatch at the truth of every waveform against "isolated", per source."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    positions = np.arange(len(records))
    shifts = np.linspace(-0.24, 0.24, len(WAVEFORMS))
    for shift, waveform in zip(shifts, WAVEFORMS):
        values = [record["mismatch"][waveform]["isolated"]["mismatch"] for record in records]
        axis.plot(
            positions + shift,
            np.abs(values),
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
    axis.set_ylabel(
        f"|mismatch| at the truth, {DATA_SET_LABELS['isolated']}", fontsize=8, color=INK_SECONDARY
    )
    style_axes(axis)
    axis.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def neighbour_lines(record):
    lines = []
    for entry in record["neighbours"]:
        stft, fd = entry["stft"], entry.get("fd")
        selected = f" = {entry['selected_as']}" if entry["selected_as"] else ""
        text = (
            f"neighbour {entry['index']}{selected}: {entry['gap_bins']:+.1f} STFT bins, A ratio "
            f"{entry['amplitude_ratio']:.3f}; STFT power in band {stft['power_in_band']:.3e} "
            f"({stft['share_of_gb']:.2e} of gb), cross term {stft['cross_term']:+.3e}, overlap "
            f"{stft['overlap']:+.2e}"
        )
        if fd is not None:
            text += (
                f"; FD {fd['gap_bins']} bins away, power in band {fd['power_in_band']:.3e} "
                f"({fd['share_of_gb']:.2e} of gb), cross term {fd['cross_term']:+.3e}, overlap "
                f"{fd['overlap']:+.2e}"
            )
        else:
            text += "; FD: its template band does not reach the band"
        lines.append(text)
    return lines


def truth_lines(record):
    gain = record["preprocessing_gain"]
    lines = [
        f"preprocessing gain at the source {gain:.7f}: amplitude ratio {1.0 / gain:.7f} for a "
        f"waveform without the filters; jaxgb orbits against the file's light travel times "
        f"{record['lisa_orbits_ltt_max_difference']:.2e} s"
    ]
    if "snr" in record:
        snr = record["snr"]
        lines.append(
            "SNR of the data in band: "
            + ", ".join(
                f"{domain} {DATA_SET_LABELS[key]} {value:.3f}"
                for domain, values in snr["data"].items()
                for key, value in values.items()
            )
            + f"; noise level (n|n)/(6N): STFT {snr['noise_level']['stft']:.4f}, FD "
            f"{snr['noise_level']['fd']:.4f}"
        )
        for waveform in WAVEFORMS:
            lines.append(
                f"{waveform:18s} optimal SNR {snr['optimal'][waveform]:10.4f}, matched-filter "
                f"SNR on data {snr['matched_filter'][waveform]:10.4f}"
            )
    if "mismatch" in record:
        for waveform in WAVEFORMS:
            lines.append(
                f"{waveform:18s} "
                + "; ".join(
                    f"{DATA_SET_LABELS[key]}: mismatch {values['mismatch']:+.3e}, amplitude ratio "
                    f"{values['amplitude_ratio']:.6f}, phase {values['phase']:+.2e}, ln L "
                    f"{values['lnl']:.3e}"
                    for key, values in record["mismatch"][waveform].items()
                )
            )
        for waveform, per_length in record["fd_resolution"].items():
            for length, values in per_length.items():
                lines.append(
                    f"FD resolution {waveform:18s} N {int(length):7d}: mismatch "
                    f"{values['mismatch']:+.3e}, amplitude ratio {values['amplitude_ratio']:.6f}, "
                    f"phase {values['phase']:+.2e}, ln L {values['lnl']:.3e}"
                )
    return lines


def report_lines(records):
    lines = []
    for record in records:
        entry, parity = record["catalogue"], record["write_parity"]
        stft_band, fd_band = record["stft_band"], record["fd_band"]
        lines += [
            "",
            f"{record['source']} ({'overlapping' if record['overlapping'] else 'isolated'}): "
            f"catalogue index {entry['index']} ({entry['ID']}), f "
            f"{entry['GW22FrequencySSBFrame']:.9e} Hz, fdot "
            f"{entry['GW22FrequencyDerivativeSourceFrame']:.4e} Hz/s, A {entry['Amplitude']:.4e}, "
            f"chirp mass {entry['ChirpMassSSBFrame']:.3f}",
            f"  STFT band {stft_band['num_bins']} bins from bin {stft_band['first_bin']}, FD band "
            f"{fd_band['num_bins']} bins from bin {fd_band['first_bin']}; DFFR template written "
            f"by fill_global_stft against get_ll_stft: (h|h) {parity['h_h']:.1e}, (d|h) "
            f"{parity['d_h']:.1e} (comp conjugated: {parity['d_h_conjugated']})",
        ]
        lines += [f"  {line}" for line in neighbour_lines(record) + truth_lines(record)]
        if "metrics" not in record:
            continue
        for key in DATA_SETS:
            lines.append(f"  surfaces on {DATA_SET_LABELS[key]}")
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
                    values = record["metrics"][waveform][key][name]
                    mark = "*" * values["rescans"] if values["peak_inside_scan"] else "!"
                    cells.append(
                        f"{values['peak']:+9.3f}{mark:1s}{values['width']:9.4f} "
                        f"{values['lnl_truth']:10.3e} {values['lnl_peak']:10.3e}"
                    )
                lines.append(f"  {name:10s} {record['sigma'][name]:10.3e} " + " ".join(cells))
    return lines


def write_combined(output, names, no_figures):
    """summary.json, report.txt and the figures of every source in ``output``.

    The figure of each source with surfaces is drawn again from its NAME.json and NAME.npz.
    """
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
        for record in with_metrics:
            with np.load(output / f"{record['source']}.npz") as arrays:
                plot_source(record, arrays, output / f"{record['source']}.png")
        for key in ("data", "isolated"):
            if with_metrics:
                plot_summary(
                    with_metrics,
                    output / f"summary_offsets_{key}.png",
                    key,
                    "peak",
                    "peak offset / sigma",
                    0.0,
                )
                plot_summary(
                    with_metrics,
                    output / f"summary_widths_{key}.png",
                    key,
                    "width",
                    "width / sigma (DFFR Fisher)",
                    1.0,
                )
        with_mismatch = [record for record in records if "mismatch" in record]
        if with_mismatch:
            plot_mismatch(with_mismatch, output / "summary_mismatch.png")
    lines = [
        f"Mojito-light validation: {OBSERVATION_YEARS} years after the global fit's preprocessing,"
        f" STFT with the periodic window and {DEFAULT_N_SIDE_BINS} side bins; offset and width in"
        " sigma of DFFR's Fisher at the truth; * = one rescan centred on the latest peak of the "
        "data, ! = peak outside its scan",
    ] + report_lines(records)
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mojito", type=Path, default=DEFAULT_MOJITO)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("mismatch", "snr", "surfaces"),
        default=["mismatch", "snr", "surfaces"],
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    data_folder = arguments.output / "data"
    if arguments.summarize_only:
        sources = json.loads((data_folder / "sources.json").read_text())["sources"]
        names = [record["name"] for record in sources]
        print("\n".join(write_combined(arguments.output, names, arguments.no_figures)))
        return
    if not (data_folder / "sources.json").exists():
        prepare_data(arguments.mojito, data_folder)
    layout = json.loads((data_folder / "layout.json").read_text())
    sources = json.loads((data_folder / "sources.json").read_text())["sources"]
    all_names = [record["name"] for record in sources]
    names = arguments.sources or all_names
    unknown = sorted(set(names) - set(all_names))
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; choose from {all_names}")

    import jax

    from lisatools.utils.utility import windowfun

    # ! jaxgb computes in double precision only with this flag.
    jax.config.update("jax_enable_x64", True)
    xp = array_module(arguments.device)
    streams = {key: xp.asarray(np.load(data_folder / f"{key}.npy")) for key in STREAMS}
    num_samples = layout["num_samples"]
    fd_window, _ = windowfun(
        "tukey", num_samples, alpha=FD_TAPER_DURATION / (num_samples * SAMPLING_INTERVAL)
    )
    fd_window = xp.asarray(fd_window)[None, :]
    orbits, _ = load_orbits("l1", arguments.device, mojito_l1=layout["orbit_file"])
    cpu_orbits, _ = load_orbits("l1", "cpu", mojito_l1=layout["orbit_file"])
    lisa_orbits = jaxgb_orbits(layout["orbit_file"])
    all_orbits = (
        orbits,
        cpu_orbits,
        lisa_orbits,
        jaxgb_orbit_check(lisa_orbits, layout["orbit_file"]),
    )
    log(f"device {arguments.device}, parts {arguments.parts}, sources {names}")

    for record in sources:
        if record["name"] not in names:
            continue
        result, arrays = evaluate_source(
            record, streams, fd_window, all_orbits, layout, arguments.device, arguments.parts
        )
        name = record["name"]
        np.savez_compressed(arguments.output / f"{name}.npz", **arrays)
        (arguments.output / f"{name}.json").write_text(json.dumps(result, indent=1))
        if "metrics" in result and not arguments.no_figures:
            plot_source(result, arrays, arguments.output / f"{name}.png")
        log(f"{name} done ({result['timings']})")
    if not arguments.no_summary:
        print("\n".join(write_combined(arguments.output, all_names, arguments.no_figures)))
    log("done")


if __name__ == "__main__":
    main()
