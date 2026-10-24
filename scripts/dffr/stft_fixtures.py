"""Fixed fixtures of the STFT galactic-binary likelihood, for the DFFR integration regressions.

Builds ``gbgpu.gbcomps.STFTGBComputations`` on deterministic grids, data, inverse covariances and
source batches, on the CPU or the GPU backend. ``stft_golden_outputs.py`` and
``stft_benchmarks.py`` evaluate the kernels on them. What each fixture covers, and why, is
described in ``docs/response/dffr/results.md``.
"""

import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

import gbgpu
import lisatools
from gbgpu.gbcomps import STFTGBComputations
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits, L1Orbits
from lisatools.domains import STFTSettings
from lisatools.sensitivity import XYZSensitivityBackend

NUM_CHANNELS = 3

# * Seeds of scripts/validation/stft_column_policy_oracle.py, so its arrays can be cross-checked.
ORACLE_SEED = 20260710
PRODUCTION_SEED = 20261017

# * Production STFT settings of mojito_input/galfor_psd_multigpu_stft_settings.py.
SEGMENT_DURATION = 86400.0
SAMPLING_INTERVAL = 5.0
BAND_MIN_FREQ = 1.0e-4
BAND_MAX_FREQ = 2.9e-2
# ? Taper of 1 / min_freq seconds per segment, as globalfit/engine.py sets window_alpha.
PRODUCTION_WINDOW_ALPHA = (1.0 / BAND_MIN_FREQ) / SEGMENT_DURATION
PRODUCTION_N_SIDE_BINS = 10
# * Data start after the 200 h the production preprocessing trims from the file start.
DATA_START_OFFSET = 10 * 86400.0

# * Two store windows of a band buffer, as active-bin offsets into the band (about 3 and 15 mHz).
CELL_WINDOW_WIDTH = 64
CELL_WINDOW_STARTS = (250, 1287)

ORACLE_CONFIGS = {
    "rect": dict(window_alpha=0.0, use_midpoint=False, freq_from_tdi_phase=True),
    "tukey_mid": dict(window_alpha=0.3, use_midpoint=True, freq_from_tdi_phase=True),
    "astro": dict(window_alpha=0.0, use_midpoint=False, freq_from_tdi_phase=False),
}

PRODUCTION_CONFIGS = {
    "l1_tdi2_envelope": dict(
        orbits="l1", tdi_generation="2nd generation", linear_envelope=True, tdi_type="XYZ"
    ),
    "esa_tdi2_envelope": dict(
        orbits="esa", tdi_generation="2nd generation", linear_envelope=True, tdi_type="XYZ"
    ),
    "esa_tdi1_const": dict(
        orbits="esa", tdi_generation="1st generation", linear_envelope=False, tdi_type="XYZ"
    ),
    "esa_aet_diag": dict(
        orbits="esa", tdi_generation="2nd generation", linear_envelope=True, tdi_type="AET"
    ),
}

ALL_CONFIGS = tuple(ORACLE_CONFIGS) + tuple(PRODUCTION_CONFIGS)
#: Responses of STFTGBComputations; the first is the default.
RESPONSES = ("tdi_on_the_fly", "dffr")
#: Configurations the DFFR response runs on: the production ones with XYZ channels. The oracle
#: configurations reproduce the column policy of the TDI-on-the-fly response.
DFFR_CONFIGS = tuple(
    name for name, config in PRODUCTION_CONFIGS.items() if config["tdi_type"] == "XYZ"
)


@dataclass
class Fixture:
    """A comp bound to its domain, with the per-source cell indices and window starts.

    ``cells`` gives the data and noise row of each source; ``window_starts`` is ``None`` for the
    full-grid oracle domains. ``keepalive`` holds every buffer the C++ wraps point into.
    """

    name: str
    comp: STFTGBComputations
    params: np.ndarray
    cells: np.ndarray
    window_starts: object
    num_templates: int
    template_shape: tuple
    description: dict
    keepalive: list = field(default_factory=list)


class ShimGroup:
    """Stand-in for ``STFTComputationGroup`` with only what ``STFTGBComputations`` reads.

    For the DFFR response it also carries the group's ``linear_envelope`` and a kinematics table
    built by the caller, which ``cpp_kinematics`` returns.
    """

    def __init__(
        self, cpp_fresnel, cpp_domain, num_noise, buffers, linear_envelope=None, kinematics=None
    ):
        self.cpp_fresnel = cpp_fresnel
        self.cpp_domain = cpp_domain
        self.d_d = None
        self.num_noise = num_noise
        # ! The domain wrap holds raw pointers into these buffers.
        self._keepalive = buffers
        # * Set only when given, so the TDI-on-the-fly response sees the shim it saw before.
        if linear_envelope is not None:
            self.linear_envelope = linear_envelope
        self._kinematics = kinematics

    def cpp_kinematics(self, cpp_orbits):
        if self._kinematics is None:
            raise TypeError("this shim holds no kinematics table")
        return self._kinematics


def array_module(backend):
    return lisatools.get_backend(backend).xp


def to_host(array):
    """NumPy copy of a NumPy or CuPy array."""
    if hasattr(array, "get"):
        return array.get()
    return np.array(array, copy=True)


def synchronize(backend):
    if backend != "cpu":
        array_module(backend).cuda.runtime.deviceSynchronize()


def draw_cell_arrays(rng, num_segments, num_freqs, diagonal):
    """Seeded data and inverse covariance of one cell, in the draw order of the oracle script.

    The inverse covariance is Hermitian across channels (full 3 x 3 for XYZ) or one positive
    value per channel (diagonal layout).
    """
    shape = (NUM_CHANNELS, num_segments, num_freqs)
    data = 1e-20 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    if diagonal:
        invc = (1e40 * (1.0 + 0.1 * rng.random(shape))).astype(np.complex128)
        return np.ascontiguousarray(data), np.ascontiguousarray(invc)

    pixel_shape = (num_segments, num_freqs)
    invc = np.zeros((NUM_CHANNELS, NUM_CHANNELS) + pixel_shape, np.complex128)
    for row in range(NUM_CHANNELS):
        invc[row, row] = 1e40 * (1.0 + 0.1 * rng.random(pixel_shape))
        for col in range(row + 1, NUM_CHANNELS):
            off_diagonal = 1e39 * (
                rng.random(pixel_shape) - 0.5 + 1j * (rng.random(pixel_shape) - 0.5)
            )
            invc[row, col] = off_diagonal
            invc[col, row] = np.conj(off_diagonal)
    return np.ascontiguousarray(data), np.ascontiguousarray(invc)


def draw_angles(rng, num):
    """Phase, inclination, polarisation and sky angles, in the oracle's draw order."""
    phi0 = rng.uniform(0, 2 * np.pi, num)
    iota = rng.uniform(0.1, np.pi - 0.1, num)
    psi = rng.uniform(0, np.pi, num)
    lam = rng.uniform(0, 2 * np.pi, num)
    beta = rng.uniform(-1.2, 1.2, num)
    return phi0, iota, psi, lam, beta


def build_oracle_fixture(name, backend):
    """One configuration of scripts/validation/stft_column_policy_oracle.py on ``backend``.

    Same grid, seeds, data, inverse covariance and sources as the oracle, so the arrays it
    captures on the CPU must equal these.
    """
    config = ORACLE_CONFIGS[name]
    segment_duration = 21600.0
    num_segments = 16
    settings = STFTSettings(
        t0=10.0 * 86400.0,
        dt=segment_duration,
        df=1.0 / segment_duration,
        NT=num_segments,
        NF=128,
        min_freq=40 / segment_duration,
        max_freq=100 / segment_duration,
        force_backend=backend,
    )
    num_freqs = settings.NF_active
    xp = array_module(backend)
    lat_backend = lisatools.get_backend(backend)

    rng = np.random.default_rng(ORACLE_SEED)
    data_host, invc_host = draw_cell_arrays(rng, num_segments, num_freqs, diagonal=False)
    data = xp.ascontiguousarray(xp.asarray(data_host[None]))
    invc = xp.ascontiguousarray(xp.asarray(invc_host[None]))
    domain = lat_backend.STFTDomainWrap(
        num_segments,
        num_freqs,
        NUM_CHANNELS,
        settings.t0,
        settings.min_freq,
        settings.max_freq,
        settings.dt,
        settings.df,
        data.reshape(-1),
        invc.reshape(-1),
        1,
        1,
        lat_backend.TDITypeDict["XYZ"],
    )
    fresnel = lat_backend.STFTFresnelWrap(
        num_segments,
        num_freqs,
        NUM_CHANNELS,
        settings.t0,
        settings.min_freq,
        settings.max_freq,
        settings.dt,
        settings.df,
        window_alpha=config["window_alpha"],
        use_midpoint=config["use_midpoint"],
    )
    shim = ShimGroup(fresnel, domain, num_noise=1, buffers=(data, invc))
    comp = STFTGBComputations(
        stft_comps=shim,
        T=num_segments * segment_duration,
        t_ref=0.0,
        orbits=EqualArmlengthOrbits(force_backend=backend),
        tdi_config="2nd generation",
        force_backend=backend,
        n_side_bins=4,
        window_factor=1.0,
        freq_from_tdi_phase=config["freq_from_tdi_phase"],
        response=RESPONSES[0],
    )

    num_sources = 24
    param_rng = np.random.default_rng(ORACLE_SEED + 1)
    df = settings.df
    first_bin, last_bin = settings.ind_min, settings.ind_max
    f0 = (first_bin + 6 + (last_bin - first_bin - 12) * param_rng.random(num_sources)) * df
    # * Carriers next to both band edges, so the side-bin clamps run.
    f0[0] = (first_bin + 1) * df
    f0[1] = (last_bin - 1) * df
    amp = 10 ** param_rng.uniform(-21.5, -20.5, num_sources)
    fdot = param_rng.uniform(-1e-15, 1e-15, num_sources)
    phi0, iota, psi, lam, beta = draw_angles(param_rng, num_sources)
    params = np.stack([amp, f0, fdot, np.zeros(num_sources), phi0, iota, psi, lam, beta], axis=-1)

    description = dict(
        kind="oracle",
        orbits="equal_arm",
        tdi="2nd generation",
        tdi_type="XYZ",
        segment_duration=segment_duration,
        num_segments=num_segments,
        num_freqs=num_freqs,
        n_side_bins=4,
        linear_envelope=False,
        domain="shim",
        **config,
    )
    return Fixture(
        name=name,
        comp=comp,
        params=params,
        cells=np.zeros(num_sources, dtype=np.int32),
        window_starts=None,
        num_templates=2,
        template_shape=(NUM_CHANNELS, num_segments, num_freqs),
        description=description,
        keepalive=[shim, data, invc],
    )


def load_orbits(kind, backend, mojito_l1=None):
    """Orbits and the time of their first sample."""
    if kind == "esa":
        orbits = ESAOrbits(force_backend=backend)
        return orbits, float(orbits.t0)
    if kind == "l1":
        if mojito_l1 is None:
            raise ValueError("the l1 fixtures need --mojito-l1 (a Mojito L1 .h5 file)")
        # * Imported here: the recipe module pulls in the whole global fit.
        from lisatools.globalfit.recipe import MOJITO_AVERAGE_ARMLENGTH

        orbits = L1Orbits(
            mojito_l1, armlength=MOJITO_AVERAGE_ARMLENGTH, force_backend=backend, frame="icrs"
        )
        return orbits, max(float(orbits.sc_t0), float(orbits.ltt_t0))
    raise ValueError(f"unknown orbits kind {kind!r}")


def production_time_reference(kind, orbits_start):
    if kind == "l1":
        from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME

        return float(MOJITO_REFERENCE_TIME)
    return orbits_start


def production_settings(data_start, num_segments, backend):
    """Band settings of production and the cell settings of a store window inside it.

    The cell settings follow ``Buffer._build_per_band_basis_settings`` in gbbands.py: the
    band's ``min_freq`` and ``CELL_WINDOW_WIDTH`` active bins.
    """
    band_settings = STFTSettings(
        t0=data_start,
        dt=SEGMENT_DURATION,
        df=1.0 / SEGMENT_DURATION,
        NT=num_segments,
        NF=int(SEGMENT_DURATION / SAMPLING_INTERVAL) // 2 + 1,
        min_freq=BAND_MIN_FREQ,
        max_freq=BAND_MAX_FREQ,
        force_backend=backend,
    )
    df = band_settings.df
    cell_settings = STFTSettings(
        *band_settings.args,
        min_freq=max(band_settings.ind_min - 0.5, 0.0) * df,
        max_freq=(band_settings.ind_min + CELL_WINDOW_WIDTH - 0.5) * df,
        force_backend=backend,
    )
    if (
        cell_settings.NF_active != CELL_WINDOW_WIDTH
        or cell_settings.min_freq != band_settings.min_freq
    ):
        raise RuntimeError("cell settings do not match the store-window layout of gbbands.py")
    last_start = band_settings.NF_active - CELL_WINDOW_WIDTH
    if max(CELL_WINDOW_STARTS) > last_start:
        raise RuntimeError("a store window reaches past the band")
    return band_settings, cell_settings


def build_group(cell_settings, backend, tdi_type, linear_envelope, rng):
    """A real ``STFTComputationGroup`` over one cell per store window, as the band buffer builds it.

    The group's sensitivity backend gets equal-arm orbits because the GB kernels never read
    them; they use the comp's own orbits.
    """
    xp = array_module(backend)
    diagonal = tdi_type != "XYZ"
    num_segments, num_freqs = cell_settings.NT, cell_settings.NF_active
    sensitivity_orbits = EqualArmlengthOrbits(force_backend=backend)

    analysis_containers = []
    for _ in CELL_WINDOW_STARTS:
        data_host, invc_host = draw_cell_arrays(rng, num_segments, num_freqs, diagonal)
        data_domain = cell_settings.associated_class(xp.asarray(data_host), cell_settings)
        sensitivity = XYZSensitivityBackend(
            orbits=sensitivity_orbits, settings=cell_settings, force_backend=backend
        )
        sensitivity.sens_mat = xp.zeros(invc_host.shape, dtype=np.complex128)
        sensitivity.invC = xp.asarray(invc_host)
        sensitivity.channel_shape = invc_host.shape[: -len(cell_settings.basis_shape_active)]
        analysis_containers.append(AnalysisContainer(data_domain, sensitivity))

    container_array = AnalysisContainerArray(
        analysis_containers,
        gpus=None if backend == "cpu" else [0],
        domain_group_kwargs=dict(
            tdi_type=tdi_type,
            window_alpha=PRODUCTION_WINDOW_ALPHA,
            use_midpoint=True,
            linear_envelope=linear_envelope,
        ),
    )
    return container_array, container_array.cpp_splits[0]


def make_production_params(rng, band_settings, num_sources):
    """Seeded sources spread over the two store windows, alternating between them.

    Each window gets carriers next to both of its edges; every third source carries the
    general-relativity ``fddot = 11/3 fdot^2 / f0`` so the second derivative enters the phase.
    """
    df = band_settings.df
    cells = (np.arange(num_sources) % len(CELL_WINDOW_STARTS)).astype(np.int32)
    first_bin = band_settings.ind_min + np.asarray(CELL_WINDOW_STARTS)[cells]
    f0 = (first_bin + 6 + (CELL_WINDOW_WIDTH - 12) * rng.random(num_sources)) * df
    for cell in range(len(CELL_WINDOW_STARTS)):
        members = np.flatnonzero(cells == cell)
        if members.size >= 2:
            f0[members[0]] = (first_bin[members[0]] + 1) * df
            f0[members[1]] = (first_bin[members[1]] + CELL_WINDOW_WIDTH - 2) * df
    amp = 10 ** rng.uniform(-22.5, -21.5, num_sources)
    fdot = rng.uniform(-1e-16, 1e-16, num_sources)
    fddot = np.where(np.arange(num_sources) % 3 == 2, 11.0 / 3.0 * fdot**2 / f0, 0.0)
    phi0, iota, psi, lam, beta = draw_angles(rng, num_sources)
    params = np.stack([amp, f0, fdot, fddot, phi0, iota, psi, lam, beta], axis=-1)
    return params, cells


def production_sources(band_settings, num_sources=24, tdi_type="XYZ"):
    """The sources and the generator state of ``build_production_fixture``, without its group.

    The group draws the data and inverse covariance of its store windows first, so the generator
    makes the same draws before the sources.
    """
    rng = np.random.default_rng(PRODUCTION_SEED)
    for _ in CELL_WINDOW_STARTS:
        draw_cell_arrays(rng, band_settings.NT, CELL_WINDOW_WIDTH, tdi_type != "XYZ")
    params, cells = make_production_params(rng, band_settings, num_sources)
    return params, cells, rng


def build_production_fixture(
    name,
    backend,
    mojito_l1=None,
    num_segments=32,
    num_sources=24,
    orbits=None,
    response=RESPONSES[0],
):
    """One production-like configuration on a real computation group with two store windows.

    ``orbits`` may pass an already loaded ``(orbits, start)`` pair, so a large orbit file is read
    once per process. ``response`` selects the response of the comp; DFFR builds its kinematics
    table through the group, as in a global-fit run.
    """
    config = PRODUCTION_CONFIGS[name]
    if orbits is None:
        orbits = load_orbits(config["orbits"], backend, mojito_l1)
    orbits_object, orbits_start = orbits
    data_start = orbits_start + DATA_START_OFFSET
    t_ref = production_time_reference(config["orbits"], orbits_start)
    band_settings, cell_settings = production_settings(data_start, num_segments, backend)

    rng = np.random.default_rng(PRODUCTION_SEED)
    container_array, group = build_group(
        cell_settings, backend, config["tdi_type"], config["linear_envelope"], rng
    )
    comp = STFTGBComputations(
        stft_comps=group,
        T=num_segments * SEGMENT_DURATION,
        t_ref=t_ref,
        orbits=orbits_object,
        tdi_config=config["tdi_generation"],
        force_backend=backend,
        n_side_bins=PRODUCTION_N_SIDE_BINS,
        window_factor=1.0,
        freq_from_tdi_phase=True,
        window_alpha=PRODUCTION_WINDOW_ALPHA,
        use_midpoint=True,
        linear_envelope=config["linear_envelope"],
        response=response,
        # * The value of every capture of the effort; the comp defaults to False since the review.
        first_order_delays=True,
    )
    params, cells = make_production_params(rng, band_settings, num_sources)
    xp = array_module(backend)
    window_starts = xp.asarray(np.asarray(CELL_WINDOW_STARTS, dtype=np.int32))

    description = dict(
        kind="production",
        orbits=config["orbits"],
        tdi=config["tdi_generation"],
        tdi_type=config["tdi_type"],
        linear_envelope=config["linear_envelope"],
        segment_duration=SEGMENT_DURATION,
        num_segments=num_segments,
        window_alpha=PRODUCTION_WINDOW_ALPHA,
        use_midpoint=True,
        freq_from_tdi_phase=True,
        n_side_bins=PRODUCTION_N_SIDE_BINS,
        band_ind_min=int(band_settings.ind_min),
        band_num_freqs=int(band_settings.NF_active),
        cell_window_width=CELL_WINDOW_WIDTH,
        cell_window_starts=list(CELL_WINDOW_STARTS),
        data_start=data_start,
        t_ref=t_ref,
        orbits_start=orbits_start,
        domain="STFTComputationGroup",
        response=response,
    )
    return Fixture(
        name=name,
        comp=comp,
        params=params,
        cells=cells,
        window_starts=window_starts,
        num_templates=len(CELL_WINDOW_STARTS),
        template_shape=(NUM_CHANNELS, num_segments, CELL_WINDOW_WIDTH),
        description=description,
        keepalive=[container_array, group],
    )


def build_fixture(name, backend, mojito_l1=None, orbits_cache=None, response=RESPONSES[0]):
    """Build any configuration by name; ``orbits_cache`` maps an orbits kind to a loaded pair."""
    if response != RESPONSES[0] and name not in DFFR_CONFIGS:
        raise ValueError(f"{name} is not a configuration of the DFFR response: {DFFR_CONFIGS}")
    if name in ORACLE_CONFIGS:
        return build_oracle_fixture(name, backend)
    kind = PRODUCTION_CONFIGS[name]["orbits"]
    orbits = None
    if orbits_cache is not None:
        if kind not in orbits_cache:
            orbits_cache[kind] = load_orbits(kind, backend, mojito_l1)
        orbits = orbits_cache[kind]
    return build_production_fixture(
        name, backend, mojito_l1=mojito_l1, orbits=orbits, response=response
    )


def _git_state(repo_root):
    def run(*args):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return dict(
        commit=run("rev-parse", "HEAD"),
        branch=run("symbolic-ref", "--short", "-q", "HEAD"),
        tracked_changes=bool(run("status", "--porcelain", "--untracked-files=no")),
    )


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_metadata(backend, mojito_l1=None):
    """Where and on what code a capture or benchmark ran."""
    versions = {}
    for package in ("lisaanalysistools", "gbgpu", "gpubackendtools", "numpy", "scipy", "cupy"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    info = dict(
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(),
        cpu_model=_cpu_model(),
        backend=backend,
        lat=_git_state(Path(lisatools.__file__).resolve().parents[2]),
        gbgpu=_git_state(Path(gbgpu.__file__).resolve().parents[2]),
        versions=versions,
        environment={
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
            )
        },
    )
    if backend != "cpu":
        cupy = array_module(backend)
        info["gpu_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    if mojito_l1 is not None:
        stat = os.stat(mojito_l1)
        info["mojito_l1"] = dict(
            name=os.path.basename(mojito_l1),
            size_bytes=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        )
    return info
