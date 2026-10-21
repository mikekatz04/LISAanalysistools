"""Both responses against the STFT of a time-domain injection, and the window convention.

    python stft_accuracy.py --write-sources DIR/sources.json [--days 365]
    DFFR_REPOSITORY/.venv/bin/python td_truth.py --sources DIR/sources.json --output DIR/truth
    python stft_accuracy.py --device gpu --output DIR [--truth DIR/truth] [--days 365]
        [--n-side-bins 60] [--sources NAME ...] [--check-only] [--no-figures]

Injection: X, Y and Z of TDI 2nd generation every 5 s from 10 days after the start of the ESA
orbit file that LAT ships. With ``--truth``, the LISA Consortium chain of ``td_truth.py``
(lisagwresponse and pytdi, in the environment of the dffr repository), which takes the sources
and the grid from ``--write-sources``; without it, LAT's full time-domain response
(``td_injection.TimeDomainTDI``, ``pyResponseTDI``). Data: ``TDSignal.stft`` of the injection per
one-day segment, twice: with LAT's Tukey window (``windowfun``, symmetric), as
``globalfit/engine.py`` builds the data, and with the periodic Tukey window that both responses
integrate. Templates: ``STFTGBComputations`` with each response and the production settings (XYZ,
taper 0.1157, segment midpoints, linear envelope, ``freq_from_tdi_phase``) but 60 side bins
(``--n-side-bins``; production takes 10), on one domain per source that spans its frequency
track over the data with the side bins and 20 more bins to spare on each side, cut where the
production band starts (0.1 mHz). The
inverse covariance is the XYZ instrument noise of ``XYZSensitivityBackend`` (TDI 2nd generation,
its default noise levels); every metric is normalised, so the noise level only weights pixels.

Source sets (``--source-set``): ``validation``, the eight sources of the validation items 1 and 3
(0.5 and 20 mHz, the two fdot bounds of the global fit's GB prior at each, two sky positions),
on the grid above; ``step0``, the eight cases of the step 0 truth (3, 20 and 40 mHz, and 20 mHz
with fdot 1e-13 Hz/s, at the same two sky positions) on its grid: taper 0.116, data from 5 days
after the start of the orbit file, and the flat metrics over every bin within 3e-4 Hz of the
carrier instead of the template's pixels.

Metrics per source, response and data window, with d the data and h the template:

* over the pixels that the template fills: the flat mismatch 1 - |<d, h>| / (|d| |h|) and, per
  segment and channel, max |h - d| / max |d|: the metrics of the standalone package's
  time-domain comparison, so they compare with step 0;
* with the noise weights of the likelihood, over the whole domain: the mismatch
  1 - Re(d|h) / sqrt((d|d)(h|h)) from ``get_ll_stft`` and the group's (d|d), the amplitude ratio
  sqrt((h|h)/(d|d)), the phase of (d|h) and the residual (d-h|d-h)/(d|d);
* the share of (d|d) outside the pixels that the template fills.

The two data transforms against each other give the window convention alone. Before the
sources, the script compares the injection with ``GBTDIonTheFly`` in the time domain, which shows
whether the two codes share their conventions (sign of the cross polarisation, polarisation
angle, phase offset, strain factor, time origin). There is no threshold; the results are
recorded in ``docs/response/dffr/tests.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stft_fixtures import (  # noqa: E402
    BAND_MIN_FREQ,
    DATA_START_OFFSET,
    PRODUCTION_WINDOW_ALPHA,
    RESPONSES,
    SAMPLING_INTERVAL,
    SEGMENT_DURATION,
    array_module,
    load_orbits,
    run_metadata,
    to_host,
)
from td_injection import PARAMETER_NAMES, TimeDomainTDI  # noqa: E402

TDI_GENERATION = "2nd generation"
DEFAULT_DAYS = 365
# * Side bins of the templates: under the production taper, a tone keeps 4e-8 to 1e-7 of its power
# * beyond 60 bins, against 1.2e-3 to 1.4e-3 beyond the 10 bins of production.
DEFAULT_N_SIDE_BINS = 60
#: Bins of each domain beyond the template band, on both sides of the frequency track.
BAND_PADDING = 20
# * LISA's orbital speed over the speed of light, about 1e-4, bounds the Doppler shift.
DOPPLER_FRACTION = 1.0e-4
#: Low and high frequency of the validation sources [Hz].
FREQUENCIES = {"0.5mHz": 0.5e-3, "20mHz": 20.0e-3}
#: Largest and smallest fdot at the source's frequency: the bounds of the global fit's GB prior.
FDOT_BOUNDS = {"fdotmax": "+", "fdotmin": "-"}
#: Sky positions (lam, beta) [rad] and the other angles of the standalone package's truth runs.
SKY_POSITIONS = ((2.1, -0.35), (5.0, 0.9))
AMPLITUDE, INITIAL_PHASE, INCLINATION, POLARIZATION = 1.0e-22, 1.2, 0.9, 0.7
#: Data windows: LAT's Tukey (symmetric), as the global fit builds the data, and the periodic
#: Tukey that both responses integrate.
WINDOWS = ("lat", "periodic")
#: The eight cases of the step 0 truth (dffr's scripts/compare_time_domain.py): carrier and fdot,
#: each at the two sky positions above, carriers moved to the nearest bin as there.
STEP0_CASES = {
    "3mHz": (3.0e-3, 3.0e-16),
    "20mHz": (2.0e-2, 3.0e-16),
    "20mHz-chirping": (2.0e-2, 1.0e-13),
    "40mHz": (4.0e-2, 3.0e-16),
}
#: Grid of the step 0 truth: Tukey taper, first segment in days after the start of the orbit file,
#: and the half width of its band around the carrier [Hz], where its metrics were taken.
STEP0_WINDOW_ALPHA = 0.116
STEP0_START_DAY = 5
STEP0_HALF_BAND = 3.0e-4
# * Convention check: two days from day 100 of the data (earlier on shorter data), GBTDIonTheFly
# * spline knots every 300 s.
CHECK_START_DAY = 100
CHECK_DAYS = 2
CHECK_KNOT_SPACING = 300.0

RESPONSE_LABELS = {"tdi_on_the_fly": "TDI on the fly", "dffr": "DFFR"}
WINDOW_LABELS = {"lat": "LAT window (symmetric)", "periodic": "periodic window"}
# * Figure colours: categorical slots 1 and 2 for the responses, ink for the data and the
# * window-only curves, which are references rather than series.
RESPONSE_COLORS = {"dffr": "#2a78d6", "tdi_on_the_fly": "#eb6834"}
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)


def log(message):
    print(f"[accuracy] {time.strftime('%H:%M:%S')} {message}", flush=True)


def validation_sources():
    """The eight sources: two frequencies, the two fdot bounds at each, two sky positions."""
    from lisatools.globalfit.priors.gbpriors import get_fdot_mojito

    sources = {}
    for frequency_name, frequency in FREQUENCIES.items():
        for bound_name, sign in FDOT_BOUNDS.items():
            fdot = float(get_fdot_mojito(frequency, sign))
            for sky_index, (lam, beta) in enumerate(SKY_POSITIONS):
                sources[f"{frequency_name}_{bound_name}_sky{sky_index}"] = np.array(
                    [
                        AMPLITUDE,
                        frequency,
                        fdot,
                        0.0,
                        INITIAL_PHASE,
                        INCLINATION,
                        POLARIZATION,
                        lam,
                        beta,
                    ]
                )
    return sources


def step0_sources():
    """The eight cases of the step 0 truth, with its amplitude and angles."""
    sources = {}
    for case, (frequency, fdot) in STEP0_CASES.items():
        on_bin = round(frequency * SEGMENT_DURATION) / SEGMENT_DURATION
        for sky_index, (lam, beta) in enumerate(SKY_POSITIONS):
            sources[f"{case}_sky{sky_index}"] = np.array(
                [AMPLITUDE, on_bin, fdot, 0.0, INITIAL_PHASE, INCLINATION, POLARIZATION, lam, beta]
            )
    return sources


@dataclass(frozen=True)
class SourceSet:
    """The sources of a validation item and the grid they are compared on."""

    sources: dict
    #: Tukey taper of the data windows and of the responses.
    window_alpha: float
    #: First data sample after the start of the orbit file [s].
    start_offset: float
    #: Half width [Hz] of the band, about each carrier, of the flat metrics; None takes the
    #: pixels that the template fills.
    half_band: float | None


def make_source_set(name):
    """``"validation"``: the sources of items 1 and 3; ``"step0"``: the cases of the step 0 truth."""
    if name == "validation":
        return SourceSet(validation_sources(), PRODUCTION_WINDOW_ALPHA, DATA_START_OFFSET, None)
    return SourceSet(
        step0_sources(), STEP0_WINDOW_ALPHA, STEP0_START_DAY * SEGMENT_DURATION, STEP0_HALF_BAND
    )


def window_samples(window, num_samples, window_alpha):
    """One segment of the data window ``"lat"`` or ``"periodic"``."""
    if window == "lat":
        from lisatools.utils.utility import windowfun

        samples, _ = windowfun("tukey", num_samples, alpha=window_alpha)
        return np.asarray(samples, dtype=np.float64)
    return scipy.signal.windows.tukey(num_samples, window_alpha, sym=False)


def esa_orbits_md5():
    """MD5 of the ESA orbit file that LAT ships, which the truth must have used as well."""
    import lisatools

    path = Path(lisatools.__file__).resolve().parent / "orbit_files" / "esa-trailing-orbits.h5"
    return hashlib.md5(path.read_bytes()).hexdigest()


class ResponseInjection:
    """X, Y, Z of LAT's full time-domain response (pyResponseTDI) on the data grid."""

    def __init__(self, orbits, data_start, num_samples, device):
        self.data_start, self.num_samples = data_start, num_samples
        self.label = "pyResponseTDI"
        self.records = {}
        self._tdi = TimeDomainTDI(
            orbits, TDI_GENERATION, data_start, num_samples, SAMPLING_INTERVAL, force_backend=device
        )

    def series(self, name, params, t_ref):
        return self._tdi(params, t_ref)


class TruthInjection:
    """X2, Y2, Z2 of the LISA Consortium chain, from the files of ``td_truth.py``."""

    def __init__(self, directory, data_start, num_samples, device):
        self.directory = Path(directory)
        self.data_start, self.num_samples = data_start, num_samples
        self.xp = array_module(device)
        self.records = {}
        self.label = None

    def series(self, name, params, t_ref):
        record = json.loads((self.directory / f"{name}.json").read_text())
        # ! The truth must hold this source on this grid, or every metric compares unlike things.
        if not np.array_equal(np.asarray(record["params"]), np.asarray(params)):
            raise ValueError(f"{name}: the truth was made for other parameters")
        if record["data_start"] != self.data_start or record["t_ref"] != t_ref:
            raise ValueError(f"{name}: the truth starts or is referenced at another time")
        if (
            record["sample_interval"] != SAMPLING_INTERVAL
            or record["num_samples"] < self.num_samples
        ):
            raise ValueError(f"{name}: the truth is sampled otherwise or is too short")
        if record["orbits_md5"] != esa_orbits_md5():
            raise ValueError(f"{name}: the truth used another orbit file")
        chain = record["truth_chain"]
        label = "lisagwresponse, " + ("lisainstrument, " if chain == "instrument" else "") + "pytdi"
        if self.label not in (None, label):
            raise ValueError("the truth files come from different chains")
        self.label = label
        self.records[name] = record
        series = np.load(self.directory / f"{name}.npz")["series"][:, : self.num_samples]
        return self.xp.asarray(series)


def check_conventions(device, orbits, t_ref, sources, injection):
    """The injection against GBTDIonTheFly in the time domain, the first source of each carrier.

    Per channel: the least-squares scale of the GBTDIonTheFly series onto the injection, and the
    largest difference relative to the injection's peak, before and after that scale.
    """
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.response.tdionfly import GBTDIonTheFly

    xp = array_module(device)
    samples_per_day = int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
    num_samples = CHECK_DAYS * samples_per_day
    start_day = min(CHECK_START_DAY, injection.num_samples // samples_per_day - CHECK_DAYS)
    if start_day < 0:
        raise ValueError("the convention check needs more days of data")
    first = start_day * samples_per_day
    start = injection.data_start + first * SAMPLING_INTERVAL
    times = start + xp.arange(num_samples) * SAMPLING_INTERVAL
    # * Knots one hour beyond both ends, so the spline is not evaluated at its edges.
    knots = xp.arange(
        start - 3600.0, start + CHECK_DAYS * SEGMENT_DURATION + 3600.0 + 1.0, CHECK_KNOT_SPACING
    )
    tdi_config = TDIConfig(TDI_GENERATION, force_backend=device)

    first_of_carrier = {}
    for key, values in sources.items():
        first_of_carrier.setdefault(float(values[1]), key)
    records = []
    for name in first_of_carrier.values():
        params = sources[name]
        full = to_host(injection.series(name, params, t_ref)[:, first : first + num_samples])
        generator = GBTDIonTheFly(
            knots,
            CHECK_DAYS * SEGMENT_DURATION,
            t_ref,
            1.0 / SAMPLING_INTERVAL,
            1,
            tdi_config=tdi_config,
            orbits=orbits,
            tdi_chan="XYZ",
            force_backend=device,
        )
        output = generator(*params, convert_to_ra_dec=False, return_spline=True)
        on_the_fly = to_host(output.eval_tdi(times))[0]
        channels = []
        for channel in range(3):
            reference, other = full[channel], on_the_fly[channel]
            peak = np.abs(reference).max()
            scale = float(np.dot(other, reference) / np.dot(other, other))
            channels.append(
                dict(
                    channel="XYZ"[channel],
                    scale=scale,
                    difference=float(np.abs(reference - other).max() / peak),
                    difference_after_scale=float(np.abs(reference - scale * other).max() / peak),
                    peak=float(peak),
                )
            )
        records.append(
            dict(source=name, start=start, start_day=start_day, days=CHECK_DAYS, channels=channels)
        )
        for entry in channels:
            log(
                f"convention check {name} {entry['channel']}: scale {entry['scale']:.12f}, "
                f"difference {entry['difference']:.2e}, after the scale "
                f"{entry['difference_after_scale']:.2e}"
            )
    return records


def source_settings(params, data_start, num_segments, t_ref, n_side_bins, device):
    """STFT settings of one source: its frequency track over the data, padded on both sides."""
    from lisatools.domains import STFTSettings

    df = 1.0 / SEGMENT_DURATION
    samples_per_segment = int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
    t_diff = np.linspace(data_start, data_start + num_segments * SEGMENT_DURATION, 1001) - t_ref
    f0, fdot, fddot = params[1:4]
    track = f0 + fdot * t_diff + 0.5 * fddot * t_diff * t_diff
    reach = n_side_bins + BAND_PADDING + int(np.ceil(DOPPLER_FRACTION * track.max() / df))
    # ! A low-frequency source's band is cut where the production band starts (0.1 mHz), as the
    # ! global fit cuts it; the data there also hold leakage of the negative-frequency image.
    first_bin = max(int(np.floor(track.min() / df)) - reach, int(np.ceil(BAND_MIN_FREQ / df)))
    last_bin = int(np.ceil(track.max() / df)) + reach
    # * Half-bin edges, so that the band holds exactly the bins first_bin to last_bin.
    return STFTSettings(
        t0=data_start,
        dt=SEGMENT_DURATION,
        df=df,
        NT=num_segments,
        NF=samples_per_segment // 2 + 1,
        min_freq=(first_bin - 0.5) * df,
        max_freq=(last_bin + 0.5) * df,
        force_backend=device,
    )


def build_group(settings, data, orbits, window_alpha, device):
    """Computation group over the two data transforms, both with the XYZ instrument noise."""
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.sensitivity import XYZSensitivityBackend

    containers = []
    for window in WINDOWS:
        sensitivity = XYZSensitivityBackend(
            orbits=orbits, settings=settings, tdi_generation=2, force_backend=device
        )
        sensitivity.set_sensitivity_matrix()
        containers.append(AnalysisContainer(data[window], sensitivity))
    container_array = AnalysisContainerArray(
        containers,
        gpus=None if device == "cpu" else [0],
        domain_group_kwargs=dict(
            tdi_type="XYZ",
            window_alpha=window_alpha,
            use_midpoint=True,
            linear_envelope=True,
        ),
    )
    group = container_array.cpp_splits[0]
    group.compute_d_d_term()
    return container_array, group, to_host(sensitivity.invC)


def build_comp(group, orbits, t_ref, num_segments, n_side_bins, window_alpha, response, device):
    from gbgpu.gbcomps import STFTGBComputations

    return STFTGBComputations(
        stft_comps=group,
        T=num_segments * SEGMENT_DURATION,
        t_ref=t_ref,
        orbits=orbits,
        tdi_config=TDI_GENERATION,
        force_backend=device,
        n_side_bins=n_side_bins,
        window_factor=1.0,
        freq_from_tdi_phase=True,
        window_alpha=window_alpha,
        use_midpoint=True,
        linear_envelope=True,
        response=response,
    )


def inner(first, second, inverse_covariance, df):
    """Complex noise-weighted product 4 df sum conj(first_i) C^-1_ij second_j of (3, NT, NF) arrays.

    The pixel sum of ``lisatools.diagnostic.inner_product``, kept complex.
    """
    return complex(
        4.0 * df * np.einsum("itf,ijtf,jtf->", np.conj(first), inverse_covariance, second)
    )


def flat_metrics(model, reference, mask):
    """Flat mismatch with a free phase and per-segment errors over the pixels of ``mask``.

    Returns the summary, the errors per channel and segment (3, NT) and the mismatch per segment.
    """
    model = np.where(mask, model, 0.0)
    reference = np.where(mask, reference, 0.0)
    norm = np.linalg.norm
    overlap = abs(np.vdot(reference, model))
    mismatch = 1.0 - overlap / (norm(reference) * norm(model))
    segment_errors = np.abs(model - reference).max(axis=-1) / np.abs(reference).max(axis=-1)
    segment_overlaps = np.abs(np.einsum("ctf,ctf->t", np.conj(reference), model))
    segment_mismatch = 1.0 - segment_overlaps / (
        norm(reference, axis=(0, 2)) * norm(model, axis=(0, 2))
    )
    summary = dict(
        mismatch=float(mismatch),
        median_segment_error=float(np.median(segment_errors)),
        max_segment_error=float(segment_errors.max()),
        median_segment_mismatch=float(np.median(segment_mismatch)),
        max_segment_mismatch=float(segment_mismatch.max()),
    )
    return summary, segment_errors, segment_mismatch


def noise_metrics(model, reference, inverse_covariance, df):
    """Mismatch, amplitude ratio, phase and residual of ``model`` against ``reference``."""
    model_model = inner(model, model, inverse_covariance, df).real
    reference_reference = inner(reference, reference, inverse_covariance, df).real
    cross = inner(reference, model, inverse_covariance, df)
    difference = model - reference
    residual = inner(difference, difference, inverse_covariance, df).real
    return dict(
        mismatch=float(1.0 - cross.real / np.sqrt(model_model * reference_reference)),
        amplitude_ratio=float(np.sqrt(model_model / reference_reference)),
        phase=float(np.angle(cross)),
        residual=float(residual / reference_reference),
    )


def evaluate_source(
    name, params, injection, orbits, t_ref, num_segments, n_side_bins, source_set, device
):
    """Data, templates and every metric of one source; returns (record, arrays)."""
    from lisatools.domains import TDSettings, TDSignal

    xp = array_module(device)
    started = time.time()
    series = injection.series(name, params, t_ref)
    settings = source_settings(
        params, injection.data_start, num_segments, t_ref, n_side_bins, device
    )
    signal = TDSignal(
        series,
        settings=TDSettings(
            injection.num_samples, SAMPLING_INTERVAL, injection.data_start, force_backend=device
        ),
    )
    samples_per_segment = settings.get_nperseg(SAMPLING_INTERVAL)
    data = {
        window: signal.stft(
            window=xp.asarray(window_samples(window, samples_per_segment, source_set.window_alpha)),
            settings=settings,
        )
        for window in WINDOWS
    }
    container_array, group, inverse_covariance = build_group(
        settings, data, orbits, source_set.window_alpha, device
    )
    data_host = {window: to_host(data[window].arr) for window in WINDOWS}
    group_d_d = to_host(group.d_d)
    df = settings.df
    shape = data_host["lat"].shape

    record = dict(
        source=name,
        params=dict(zip(PARAMETER_NAMES, map(float, params))),
        band_first_bin=int(settings.ind_min),
        band_num_bins=int(settings.NF_active),
        num_segments=num_segments,
        n_side_bins=n_side_bins,
        window_alpha=source_set.window_alpha,
        responses={},
    )
    band = None
    if source_set.half_band is not None:
        # * The band of the step 0 grid: every bin within half_band of the carrier, all segments.
        first = int(np.ceil((params[1] - source_set.half_band) / df - 1e-9)) - settings.ind_min
        last = int(np.floor((params[1] + source_set.half_band) / df + 1e-9)) - settings.ind_min
        band = np.zeros(shape, dtype=bool)
        band[:, :, first : last + 1] = True
        record["metric_band_bins"] = last - first + 1
    arrays = {f"data_{window}": data_host[window] for window in WINDOWS}
    checks = []
    for window_index, window in enumerate(WINDOWS):
        own = inner(data_host[window], data_host[window], inverse_covariance, df).real
        checks.append(abs(own - group_d_d[window_index]) / group_d_d[window_index])

    for response in RESPONSES:
        comp = build_comp(
            group,
            orbits,
            t_ref,
            num_segments,
            n_side_bins,
            source_set.window_alpha,
            response,
            device,
        )
        templates = xp.zeros((1,) + shape, dtype=xp.complex128)
        comp.fill_global_stft(params[None], templates, data_index=xp.zeros(1, dtype=xp.int32))
        template = to_host(templates)[0]
        comp.get_ll_stft(
            np.stack([params, params]),
            data_index=xp.arange(len(WINDOWS), dtype=xp.int32),
            noise_index=xp.arange(len(WINDOWS), dtype=xp.int32),
        )
        d_h = to_host(comp.d_h_out)
        h_h = to_host(comp.h_h_out).real
        # * The pixels the template fills: its band of 2 n_side_bins + 1 bins in each segment.
        filled = np.broadcast_to((np.abs(template) > 0.0).any(axis=0), shape)
        mask = filled if band is None else band
        if band is not None and not filled[band].all():
            raise RuntimeError(f"{name}: the template does not fill the band of the metrics")
        arrays[f"template_{response}"] = template
        arrays[f"mask_{response}"] = filled[0]
        own_h_h = inner(template, template, inverse_covariance, df).real
        checks.append(abs(own_h_h - h_h[0]) / h_h[0])

        per_window = {}
        for window_index, window in enumerate(WINDOWS):
            reference = data_host[window]
            flat, segment_errors, segment_mismatch = flat_metrics(template, reference, mask)
            noise = noise_metrics(template, reference, inverse_covariance, df)
            own_d_h = inner(reference, template, inverse_covariance, df)
            checks.append(abs(own_d_h.real - d_h[window_index].real) / abs(d_h[window_index]))
            outside = np.where(filled, 0.0, reference)
            share_outside = inner(outside, outside, inverse_covariance, df).real / (
                group_d_d[window_index]
            )
            per_window[window] = dict(
                flat=flat,
                noise=noise,
                comp=dict(
                    mismatch=float(
                        1.0
                        - d_h[window_index].real
                        / np.sqrt(group_d_d[window_index] * h_h[window_index])
                    ),
                    d_h=[float(d_h[window_index].real), float(d_h[window_index].imag)],
                    h_h=float(h_h[window_index]),
                    d_d=float(group_d_d[window_index]),
                ),
                share_outside_band=float(share_outside),
            )
            arrays[f"segment_errors_{response}_{window}"] = segment_errors
            arrays[f"segment_mismatch_{response}_{window}"] = segment_mismatch
        record["responses"][response] = per_window

    # * The window convention alone: the LAT-window data against the periodic-window data.
    everywhere = np.ones(shape, dtype=bool)
    flat, segment_errors, segment_mismatch = flat_metrics(
        data_host["lat"], data_host["periodic"], everywhere
    )
    record["window_only"] = dict(
        flat=flat,
        noise=noise_metrics(data_host["lat"], data_host["periodic"], inverse_covariance, df),
    )
    arrays["segment_errors_window_only"] = segment_errors
    arrays["segment_mismatch_window_only"] = segment_mismatch
    record["inner_product_check"] = float(max(checks))
    record["seconds"] = round(time.time() - started, 1)
    del container_array
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
    """Per-segment error and mismatch over the year, and the pixels of the worst segment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharey="row", facecolor=SURFACE)
    days = np.arange(record["num_segments"])
    worst_errors = arrays["segment_errors_dffr_lat"]
    worst_channel, worst_segment = np.unravel_index(np.argmax(worst_errors), worst_errors.shape)

    for column, window in enumerate(WINDOWS):
        error_axis, mismatch_axis, pixel_axis = axes[:, column]
        for response in RESPONSES:
            color = RESPONSE_COLORS[response]
            label = RESPONSE_LABELS[response]
            error_axis.plot(
                days,
                arrays[f"segment_errors_{response}_{window}"].max(axis=0),
                color=color,
                linewidth=1.0,
                label=label,
            )
            mismatch_axis.plot(
                days,
                arrays[f"segment_mismatch_{response}_{window}"],
                color=color,
                linewidth=1.0,
                label=label,
            )
        data = arrays[f"data_{window}"][worst_channel, worst_segment]
        peak = np.abs(data).max()
        offsets = np.arange(data.size) - int(np.argmax(np.abs(data)))
        pixel_axis.plot(offsets, np.abs(data) / peak, color=INK, linewidth=1.0, label="|d|")
        for response in RESPONSES:
            template = arrays[f"template_{response}"][worst_channel, worst_segment]
            inside = arrays[f"mask_{response}"][worst_segment]
            pixel_axis.plot(
                offsets[inside],
                np.abs(template - data)[inside] / peak,
                color=RESPONSE_COLORS[response],
                linewidth=1.0,
                marker="o",
                markersize=2.5,
                label=f"|h - d|, {RESPONSE_LABELS[response]}",
            )
        if window == "lat":
            error_axis.plot(
                days,
                arrays["segment_errors_window_only"].max(axis=0),
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=":",
                label="LAT minus periodic data",
            )
            mismatch_axis.plot(
                days,
                arrays["segment_mismatch_window_only"],
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=":",
                label="LAT minus periodic data",
            )
            difference = (
                arrays["data_lat"][worst_channel, worst_segment]
                - arrays["data_periodic"][worst_channel, worst_segment]
            )
            pixel_axis.plot(
                offsets,
                np.abs(difference) / peak,
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=":",
                label="|LAT minus periodic data|",
            )
        error_axis.set_title(f"Data with the {WINDOW_LABELS[window]}", fontsize=10, color=INK)
        error_axis.set_ylabel("max |h - d| / max |d| per segment", fontsize=8, color=INK_SECONDARY)
        mismatch_axis.set_ylabel("flat mismatch per segment", fontsize=8, color=INK_SECONDARY)
        mismatch_axis.set_xlabel("days after the data start", fontsize=8, color=INK_SECONDARY)
        pixel_axis.set_ylabel(
            f"relative to the peak, day {worst_segment}, {'XYZ'[worst_channel]}",
            fontsize=8,
            color=INK_SECONDARY,
        )
        pixel_axis.set_xlabel("bins from the carrier", fontsize=8, color=INK_SECONDARY)
        reach = record["n_side_bins"] + BAND_PADDING
        pixel_axis.set_xlim(-reach, reach)
        for axis in (error_axis, mismatch_axis, pixel_axis):
            axis.set_yscale("log")
            style_axes(axis)
            axis.legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY)
    params = record["params"]
    figure.suptitle(
        f"{record['source']}: f0 = {params['f0']:.4g} Hz, fdot = {params['fdot']:.3g} Hz/s, "
        f"lam = {params['lam']}, beta = {params['beta']}",
        fontsize=11,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_summary(records, path):
    """In-band mismatch, median segment error and noise-weighted mismatch of every source."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    positions = np.arange(len(records))
    panels = (
        ("flat", "mismatch", "flat mismatch over the template's pixels"),
        ("flat", "median_segment_error", "median of max |h - d| / max |d| per segment"),
        ("noise", "mismatch", "mismatch with the noise weights, whole domain"),
    )
    # * Half the share of (d|d) outside the template band: the mismatch that the band alone leaves.
    half_outside = [
        0.5 * record["responses"]["dffr"]["lat"]["share_outside_band"] for record in records
    ]
    axes[2].plot(
        positions,
        half_outside,
        linestyle="none",
        marker="_",
        markersize=18,
        markeredgewidth=1.5,
        color=INK,
        label="half the share of (d|d) outside the template band",
    )
    for axis, (group, metric, title) in zip(axes, panels):
        for offset, response in zip((-0.15, 0.15), RESPONSES):
            color = RESPONSE_COLORS[response]
            for window, face in (("lat", color), ("periodic", "none")):
                values = [
                    record["responses"][response][window][group][metric] for record in records
                ]
                axis.plot(
                    positions + offset,
                    values,
                    linestyle="none",
                    marker="o",
                    markersize=7,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    label=f"{RESPONSE_LABELS[response]}, {WINDOW_LABELS[window]}",
                )
        window_values = [record["window_only"][group][metric] for record in records]
        axis.plot(
            positions,
            window_values,
            linestyle="none",
            marker="D",
            markersize=6,
            markerfacecolor=INK_SECONDARY,
            markeredgecolor=INK_SECONDARY,
            label="LAT against periodic data",
        )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels([record["source"] for record in records], rotation=35, ha="right")
        axis.set_title(title, fontsize=10, color=INK)
        style_axes(axis)
    # * One legend below the panels: the responses and windows of every panel, then the band line.
    handles = axes[0].get_lines() + axes[2].get_lines()[:1]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def report_lines(records, checks, injection_label):
    lines = [
        f"Injection: {injection_label}",
        f"Convention check, the injection against GBTDIonTheFly (time domain, {CHECK_DAYS} days):",
    ]
    for entry in checks:
        for channel in entry["channels"]:
            lines.append(
                f"  {entry['source']:22s} day {entry['start_day']} {channel['channel']}: "
                f"scale {channel['scale']:.12f}, "
                f"difference {channel['difference']:.2e}, after the scale "
                f"{channel['difference_after_scale']:.2e}"
            )
    header = (
        f"{'source':22s} {'response':15s} {'window':9s} {'flat mm':>9s} {'med seg':>9s} "
        f"{'max seg':>9s} {'noise mm':>9s} {'amp-1':>9s} {'phase':>9s} {'residual':>9s} "
        f"{'outside':>9s}"
    )
    lines += ["", header]
    for record in records:
        for response in RESPONSES:
            for window in WINDOWS:
                entry = record["responses"][response][window]
                flat, noise = entry["flat"], entry["noise"]
                lines.append(
                    f"{record['source']:22s} {response:15s} {window:9s} {flat['mismatch']:9.2e} "
                    f"{flat['median_segment_error']:9.2e} {flat['max_segment_error']:9.2e} "
                    f"{noise['mismatch']:9.2e} {noise['amplitude_ratio'] - 1:9.2e} "
                    f"{noise['phase']:9.2e} {noise['residual']:9.2e} "
                    f"{entry['share_outside_band']:9.2e}"
                )
        flat, noise = record["window_only"]["flat"], record["window_only"]["noise"]
        lines.append(
            f"{record['source']:22s} {'window only':15s} {'':9s} {flat['mismatch']:9.2e} "
            f"{flat['median_segment_error']:9.2e} {flat['max_segment_error']:9.2e} "
            f"{noise['mismatch']:9.2e} {noise['amplitude_ratio'] - 1:9.2e} "
            f"{noise['phase']:9.2e} {noise['residual']:9.2e}"
        )
        lines.append(
            f"{'':22s} inner products against the comp and group: {record['inner_product_check']:.1e}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--source-set", choices=("validation", "step0"), default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-sources", type=Path, default=None, metavar="JSON")
    parser.add_argument("--truth", type=Path, default=None, metavar="DIR")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--n-side-bins", type=int, default=DEFAULT_N_SIDE_BINS)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()

    chosen = make_source_set(arguments.source_set)
    sources = chosen.sources
    names = arguments.sources or list(sources)
    unknown = sorted(set(names) - set(sources))
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; choose from {list(sources)}")
    num_samples = arguments.days * int(round(SEGMENT_DURATION / SAMPLING_INTERVAL))
    # * Times count from the start of the ESA orbit file, as LAT's orbit classes count them, and
    # * t_ref sits at that start, as the ESA production fixtures set it.
    t_ref = 0.0
    data_start = t_ref + chosen.start_offset
    if arguments.write_sources is not None:
        arguments.write_sources.parent.mkdir(parents=True, exist_ok=True)
        grid = dict(
            source_set=arguments.source_set,
            data_start=data_start,
            t_ref=t_ref,
            num_samples=num_samples,
            sample_interval=SAMPLING_INTERVAL,
            orbits_md5=esa_orbits_md5(),
            parameter_names=list(PARAMETER_NAMES),
            sources={name: sources[name].tolist() for name in names},
        )
        arguments.write_sources.write_text(json.dumps(grid, indent=1))
        log(f"wrote {len(names)} sources to {arguments.write_sources}")
        return
    if arguments.output is None:
        raise SystemExit("--output is needed unless --write-sources is given")
    arguments.output.mkdir(parents=True, exist_ok=True)
    orbits, orbits_start = load_orbits("esa", arguments.device)
    if orbits_start != t_ref:
        raise SystemExit("LAT's ESA orbits no longer count times from the start of their file")
    if arguments.truth is not None:
        injection = TruthInjection(arguments.truth, data_start, num_samples, arguments.device)
    else:
        injection = ResponseInjection(orbits, data_start, num_samples, arguments.device)
    log(
        f"device {arguments.device}, source set {arguments.source_set}, {arguments.days} days, "
        f"{arguments.n_side_bins} side bins, sources {names}"
    )

    checks = check_conventions(
        arguments.device, orbits, t_ref, {name: sources[name] for name in names}, injection
    )
    summary = dict(
        metadata=run_metadata(arguments.device),
        source_set=arguments.source_set,
        convention_check=checks,
        sources=[],
    )
    if not arguments.check_only:
        for name in names:
            record, arrays = evaluate_source(
                name,
                sources[name],
                injection,
                orbits,
                t_ref,
                arguments.days,
                arguments.n_side_bins,
                chosen,
                arguments.device,
            )
            summary["sources"].append(record)
            np.savez_compressed(arguments.output / f"{name}.npz", **arrays)
            if not arguments.no_figures:
                plot_source(record, arrays, arguments.output / f"{name}.png")
            dffr = record["responses"]["dffr"]
            existing = record["responses"]["tdi_on_the_fly"]
            log(
                f"{name} ({record['seconds']} s): noise mismatch DFFR "
                f"{dffr['lat']['noise']['mismatch']:.2e} / {dffr['periodic']['noise']['mismatch']:.2e}"
                f", existing {existing['lat']['noise']['mismatch']:.2e} / "
                f"{existing['periodic']['noise']['mismatch']:.2e} (LAT / periodic window); window "
                f"alone {record['window_only']['noise']['mismatch']:.2e}"
            )
        if summary["sources"] and not arguments.no_figures:
            plot_summary(summary["sources"], arguments.output / "summary.png")
    summary["injection"] = dict(label=injection.label, records=injection.records)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [
        f"Source set: {arguments.source_set}, Tukey taper {chosen.window_alpha}, side bins of the "
        f"templates: {arguments.n_side_bins}"
    ] + report_lines(summary["sources"], checks, injection.label)
    (arguments.output / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    log("done")


if __name__ == "__main__":
    main()
