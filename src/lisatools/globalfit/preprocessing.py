"""Data in: load, condition and pour the global-fit input data.

The data layer of the global fit. A *data processor* class (a
:class:`BaseProcessingStep` subclass) is set as
``GeneralSettings.data_processor_class`` with its constructor kwargs in
``processor_init_kwargs``; :meth:`~lisatools.globalfit.engine.GeneralSetup.init_data_information`
constructs it (which loads the data), calls :meth:`BaseProcessingStep.process`
with ``preprocess_kwargs`` (merged over the engine defaults: 2e-5 Hz high-pass,
200 h edge trim, ``Tobs`` trim) and then :meth:`BaseProcessingStep.pour` onto
the analysis domain. ``general.data_mode`` (env ``DATA_MODE``) picks the
processor in the stock fits:

- ``"mojito"`` (default): :class:`L1ProcessingStep` / :class:`L1DataLoader`
  on a mojito L1 brick cache (``general.mojito_data_path``). The production
  9mo/12mo runs (``erebor.all_sources``) use the subclass
  :class:`~lisatools.globalfit.stock.erebor.injections.L1ProcessingStepWithSyntheticNoise`
  with ``source_types=["COMBINED", "GB", "VGB", "MBHB", "EMRI", "SOBHB"]``:
  the data is the pre-summed ``COMBINED`` stream (:func:`find_combined_file`),
  the other classes contribute catalogue rows only, the loader keeps ``Tobs``
  seconds at load, and ``all_sources`` disables the engine's filter / trim
  steps, so :meth:`BaseProcessingStep.process` only builds ``td_signal``.
- ``"sangria"``: :class:`SangriaProcessingStep` on an LDC Sangria HDF5 file.
- ``"synthetic"``: in-process streams; the stock fits use the processors of
  :mod:`lisatools.globalfit.stock.erebor.injections`. The generic
  :class:`SyntheticDataProcessor`, :class:`SyntheticSourceProcessingStep`
  and :func:`build_synthetic_source_streams` here are not used by the stock
  fits; the first and the last share their names with different objects of
  that module.

Mojito L1 layout read here (``L1_folder``)::

    <L1_folder>/data/COMBINED/L1/<one>.h5        pre-summed stream (MOJITO_COMBINED_FILE)
    <L1_folder>/data/INSTRUMENT/L1/NOISE_..._source0_....h5
    <L1_folder>/data/GALFOR/L1/<one>.h5          foreground stream (MOJITO_GALFOR_FILE)
    <L1_folder>/data/<TYPE>/L1/<TYPE>_..._source<id>_....h5   per-class bricks
    <L1_folder>/catalogues/<catalogue>.hdf5     group "Binaries" (see L1DataLoader.catalogues_map)

Each brick's ``tdis.xyz_doppler`` (dimensionless TDI X, Y, Z) and
``tdis.time_sampling`` (times in s) are read; the orbits come from the first
brick read, as ``orbits_class(file_path, **orbits_kwargs)`` (the stock fits
pass :class:`~lisatools.detector.L1Orbits` with ``frame="icrs"``, the run
frame). Data arrays are ``(n_channels, n_times)`` NumPy arrays.

Credits for most of the signal processing functions to Ollie Burke and Martina
Muratore for the repository:
https://github.com/OllieBurke/mojito-noise-sprint/blob/main/Mojito_Sprint/
"""

import dataclasses
import logging
import os
from typing import Callable, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mojito import MojitoL1File
from MojitoProcessor import SignalProcessor as MPSignalProcessor
from tqdm import tqdm

from ..domains import DomainBase
from ..detector import L1Orbits, Orbits
from ..domains import (
    DomainBase,
    DomainSettingsBase,
    FDSettings,
    TDSettings,
    TDSignal,
    place_td_signal_on_grid,
)
from ..utils.utility import get_array_module

logger = logging.getLogger(__name__)

level = logging.INFO
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # * Prevent duplicate messages*
    logger.propagate = False

#: ``COMBINED`` is mojito's pre-summed L1 stream (``data/COMBINED/L1``) --
#: instrument noise plus every source class in one brick. It is a DATA source,
#: not a source CLASS: it carries no catalogue, so listing it alongside the
#: per-class types means "take the data from here, but still read those
#: classes' catalogues" (see ``load_data``). Listing it with ``NOISE`` is an
#: error -- the noise is already in it.
ALLOWED_SOURCES = ["NOISE", "COMBINED", "GALFOR", "GB", "VGB", "MBHB", "EMRI", "SOBHB"]


def find_file(folder: str, source_type: str, source_id: int) -> str:
    """Find the Mojito L1 brick of one source in ``folder``.

    Matches on the ``{source_type}_`` prefix and the ``source{source_id}_``
    infix only, so the rest of the name (duration, cadence, timestamp, e.g.
    ``NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5``) does not need to
    be known. The first match in ``os.listdir`` order is returned. Used for
    the per-class bricks and the ``NOISE`` brick by :meth:`L1DataLoader.load_data`,
    and by the stock noise-file resolver, the monitor and the warm-start
    referee.

    Args:
        folder: Folder holding the bricks (e.g. ``<L1_folder>/data/MBHB/L1``).
        source_type: Source type prefix (``"NOISE"``, ``"GB"``, ``"MBHB"``,
            ...).
        source_id: Source index in the brick name.

    Returns:
        str: Path of the matching file.

    Raises:
        FileNotFoundError: If no matching file is found.

    Example:
        .. code-block:: python

            from lisatools.globalfit.preprocessing import find_file

            path = find_file("/data/mojito_light/data/MBHB/L1", "MBHB", 5)
    """
    for filename in os.listdir(folder):
        if filename.startswith(f"{source_type}_") and f"source{source_id}_" in filename:
            return os.path.join(folder, filename)

    raise FileNotFoundError(
        f"No Mojito L1 file found for source type '{source_type}' and source ID '{source_id}' in folder '{folder}'."
    )


def find_stream_file(folder: str, label: str, env_var: str) -> str:
    """The single stream brick in ``folder``, whatever it is called.

    Shared resolver for the whole-stream source types (``COMBINED``,
    ``GALFOR``), which -- unlike the per-class bricks -- have no stable
    ``{TYPE}_..._source{id}_`` naming and so cannot go through
    :func:`find_file`. The mojito-light GALFOR brick, for instance, is
    ``GALFOR_731d_2.5s_L1.h5``: right prefix, but no ``source0_`` infix.
    Since these folders hold exactly one data file, resolve by folder.

    ``env_var`` names an environment override with an explicit path
    (absolute, or a basename inside ``folder``) for when one does hold more.
    Used for ``GALFOR`` (``MOJITO_GALFOR_FILE``) by
    :meth:`L1DataLoader.load_data`; not reached by the production runs, which
    read ``COMBINED`` through :func:`find_combined_file`.

    Args:
        folder: Folder holding the stream (e.g. ``<L1_folder>/data/GALFOR/L1``).
        label: Stream name, used in error messages and as the preferred
            ``{label}_`` file prefix when several ``.h5`` files exist.
        env_var: Name of the override environment variable.

    Returns:
        str: Path of the stream file: the override if set, else the single
        ``.h5`` file of ``folder`` (hidden files ignored), else the single
        one named ``{label}_...``.

    Raises:
        FileNotFoundError: The override path does not exist, ``folder`` is not
            a directory, or it holds no ``.h5`` file.
        ValueError: Several ``.h5`` files and not exactly one ``{label}_``
            match.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        path = override if os.path.isabs(override) else os.path.join(folder, override)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{env_var}={override!r} does not exist (looked at {path})."
            )
        return path

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"No {label} data folder at {folder!r}. The {label} stream is "
            f"expected under <mojito_data_path>/{label}/L1."
        )
    cands = sorted(
        f for f in os.listdir(folder) if f.endswith(".h5") and not f.startswith(".")
    )
    if not cands:
        raise FileNotFoundError(f"No .h5 file in {folder!r}.")
    if len(cands) > 1:
        # Prefer an explicitly-named one, else refuse to guess.
        named = [f for f in cands if f.upper().startswith(f"{label}_")]
        if len(named) == 1:
            return os.path.join(folder, named[0])
        raise ValueError(
            f"{len(cands)} .h5 files in {folder!r} ({cands}); set "
            f"{env_var} to pick one."
        )
    return os.path.join(folder, cands[0])


def find_combined_file(folder: str) -> str:
    """The combined L1 brick in ``folder``, whatever it is called.

    Unlike the per-class bricks, the combined stream has no stable
    ``{TYPE}_..._source{id}_`` naming: the mojito-light release ships e.g.
    ``mojito_light_731d_2.5s_L1_0_0_<stamp>.h5``, which matches neither the
    ``COMBINED_`` prefix nor the ``source0_`` infix :func:`find_file`
    requires. Since ``data/COMBINED/L1`` holds exactly one data file, resolve
    it by folder rather than by name.

    ``MOJITO_COMBINED_FILE`` overrides with an explicit path (absolute, or a
    basename inside ``folder``) when a folder does hold more than one.

    The production data source: :meth:`L1DataLoader.load_data` calls it on
    ``<L1_folder>/data/COMBINED/L1`` when ``COMBINED`` is in ``source_types``
    (the 9mo/12mo runs); the monitor and the warm-start referee use it too.

    Args:
        folder: The combined-stream folder.

    Returns:
        str: Path of the combined brick: the override if set, else the single
        ``.h5`` file (hidden files ignored), else the single ``COMBINED_...``
        one.

    Raises:
        FileNotFoundError: The override path does not exist, ``folder`` is not
            a directory, or it holds no ``.h5`` file.
        ValueError: Several ``.h5`` files and not exactly one ``COMBINED_``
            match.
    """
    override = os.environ.get("MOJITO_COMBINED_FILE", "").strip()
    if override:
        path = override if os.path.isabs(override) else os.path.join(folder, override)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MOJITO_COMBINED_FILE={override!r} does not exist (looked at {path})."
            )
        return path

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"No combined-data folder at {folder!r}. The combined stream is "
            "expected under <mojito_data_path>/COMBINED/L1."
        )
    cands = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".h5") and not f.startswith(".")
    )
    if not cands:
        raise FileNotFoundError(f"No .h5 file in {folder!r}.")
    if len(cands) > 1:
        # Prefer an explicitly-named one, else refuse to guess.
        named = [f for f in cands if f.upper().startswith("COMBINED_")]
        if len(named) == 1:
            return os.path.join(folder, named[0])
        raise ValueError(
            f"{len(cands)} .h5 files in {folder!r} ({cands}); set "
            "MOJITO_COMBINED_FILE to pick one."
        )
    return os.path.join(folder, cands[0])


def normalize_source_ids(d: dict) -> dict:
    """Coerce a per-class source-id mapping to lists.

    Each value is normalized so ``None -> []``, a scalar ``v -> [v]``, and a
    list/tuple is returned as a ``list``. Used by
    :func:`~lisatools.globalfit.stock.erebor.source_runtime.default_source_ids`
    to build a uniform ``{source_type: [ids...]}`` mapping from user-edited
    literals / environment overrides (``MBHB_IDS``, ``EMRI_IDS``,
    ``SOBHB_IDS``).

    Args:
        d: ``{source_type: ids}`` with each value ``None``, a scalar, a list
            or a tuple.

    Returns:
        dict: New mapping with the same keys and ``list`` values.

    Example:
        >>> from lisatools.globalfit.preprocessing import normalize_source_ids
        >>> normalize_source_ids({"MBHB": 3, "EMRI": None, "SOBHB": (1, 2)})
        {'MBHB': [3], 'EMRI': [], 'SOBHB': [1, 2]}
    """
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = []
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = [v]
    return out


class L1DataLoader:
    """Mojito L1 brick loader: data stream, orbits and catalogue rows.

    Base of :class:`L1ProcessingStep` (it does no I/O at construction; the
    subclass calls :meth:`load_data`). The streams and catalogues it reads are
    laid out as in the module docstring. Types and their roles:

    - ``COMBINED``: the pre-summed L1 stream (noise + every source class). It
      sets the data, times and orbits; every other listed class then
      contributes CATALOGUE rows only (their signal is already in it).
      Cannot be combined with ``NOISE`` or ``GALFOR``. The production
      setting.
    - ``NOISE``: the instrument-noise brick (``data/INSTRUMENT/L1``).
    - ``GALFOR``: the galactic-confusion stream, summed into the data.
    - ``GB``, ``VGB``: one brick each (``source0``); the catalogue entry holds
      the whole catalogue columns.
    - ``MBHB``, ``EMRI``, ``SOBHB``: one brick and one catalogue row per id in
      ``source_ids``; the bricks are summed.

    Args:
        L1_folder: Path to the mojito L1 brick cache (as downloaded from the
            brick market): it holds the ``data`` and ``catalogues``
            subfolders.
        source_types: Source types to load, case-insensitive; each must be
            in :data:`ALLOWED_SOURCES`.
        source_ids: ``{type: [ids]}`` for the per-class types (keys are
            upper-cased). Required for ``MBHB`` / ``EMRI`` / ``SOBHB``;
            ignored for ``GB`` / ``VGB`` (always id 0). Default ``None``.
        orbits_class: Orbits class constructed on the first brick read as
            ``orbits_class(file_path, **orbits_kwargs)``. Default
            :class:`~lisatools.detector.L1Orbits`.
        orbits_kwargs: Keyword arguments of ``orbits_class`` (the stock fits
            pass ``force_backend`` and ``frame="icrs"``). Default ``None``.
        verbose: Log progress at INFO. Default ``True``.
        store_individual_timeseries: Keep every stream read (and the NOISE
            brick's noise-covariance estimate) in
            :attr:`individual_timeseries`. Default ``False``.
        allow_missing_bricks: HYBRID fill: a missing per-class brick (any
            of ``GB``, ``VGB``, ``MBHB``, ``EMRI``, ``SOBHB``; the erebor
            fill synthesizes MBHB / EMRI / SOBHB) is recorded in
            :attr:`missing_source_bricks` (its catalogue row is still
            loaded) instead of raising; the caller must synthesize those
            streams. Default ``False``.

    Raises:
        ValueError: A source type outside :data:`ALLOWED_SOURCES`.

    Attributes:
        data_folder (str): ``<L1_folder>/data``.
        catalogues_folder (str): ``<L1_folder>/catalogues``.
        catalogue (dict): ``{type: {id: {column: value}}}`` filled by
            :meth:`load_data`.
        source_types (list[str]): Upper-cased types still to load
            (:meth:`load_data` removes ``COMBINED`` / ``NOISE`` / ``GALFOR``
            from it as it reads them).
        source_ids (dict | None): Upper-cased ``source_ids``.
        missing_source_bricks (list[tuple[str, int]]): ``(type, id)`` of the
            bricks skipped under ``allow_missing_bricks``.
        orbits_class (type): Constructor argument.
        orbits_kwargs (dict | None): Constructor argument.
        verbose (bool): Constructor argument.
        store_individual_timeseries (bool): Constructor argument.
        allow_missing_bricks (bool): Constructor argument.
    """

    def __init__(
        self,
        L1_folder: str,
        source_types: list,
        source_ids: dict = None,
        orbits_class: Orbits = L1Orbits,
        orbits_kwargs: dict = None,
        verbose: bool = True,
        store_individual_timeseries: bool = False,  # whether to store individual timeseries for each source type and ID
        allow_missing_bricks: bool = False,
    ):
        self.data_folder = os.path.join(L1_folder, "data")
        self.catalogues_folder = os.path.join(L1_folder, "catalogues")

        self.catalogue = {}
        # HYBRID support (2026-09-14): when ``allow_missing_bricks`` is on, a
        # missing per-source L1 brick is RECORDED here (its catalogue
        # parameters are still loaded) instead of raising -- the caller is
        # then responsible for synthesizing the missing sources' streams
        # (see erebor's ``L1ProcessingStepWithSyntheticNoise
        # (synthesize_missing=True)``). Default off: a missing file stays a
        # loud failure.
        self.allow_missing_bricks = bool(allow_missing_bricks)
        self.missing_source_bricks: list = []

        source_types = [st.upper() for st in source_types]

        for source_type in source_types:
            if source_type not in ALLOWED_SOURCES:
                raise ValueError(
                    f"Source type '{source_type}' is not allowed. Allowed types are: {ALLOWED_SOURCES}"
                )

        self.source_types = source_types

        # uppercase the keys of source_ids
        if source_ids is not None:
            source_ids = {k.upper(): v for k, v in source_ids.items()}
        self.source_ids = source_ids

        self.orbits_class = orbits_class
        self.orbits_kwargs = orbits_kwargs
        self.verbose = verbose
        self.store_individual_timeseries = store_individual_timeseries
        if self.verbose:
            logger.info(f"L1DataLoader initialized with data folder: {self.data_folder}")
            logger.info(f"Source types to load: {self.source_types}")
            logger.info(f"Source IDs: {self.source_ids}")

            logger.info(f"Orbits class: {self.orbits_class.__name__}")
            if self.orbits_kwargs:
                logger.info(f"Orbits kwargs: {self.orbits_kwargs}")

    @property
    def catalogues_map(self) -> dict:
        """Catalogue file name (under ``<L1_folder>/catalogues``) per source type.

        The mojito-light processed catalogues; each holds a ``Binaries``
        group of per-parameter datasets.

        Returns:
            dict: ``{"GB": ..., "VGB": ..., "MBHB": ..., "EMRI": ...,
            "SOBHB": ...}`` file names.
        """
        return {
            "GB": "wdwd_cat_mojito_lite_processed.hdf5",
            "VGB": "vgb_cat_mojito_lite_processed.hdf5",
            "MBHB": "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5",
            "EMRI": "emri_cat_mojito_lite_processed_MT.hdf5",
            "SOBHB": "sobhb_cat_mojito_lite_processed_MT.hdf5",
        }

    def load_single_binary(self, group: h5py.Group, binary_id: int, source_type: str) -> dict:
        """Read one source's catalogue parameters from a ``Binaries`` group.

        Args:
            group: The catalogue's ``Binaries`` group (one dataset per
                parameter).
            binary_id: Row index of the source (the catalogue id).
            source_type: Upper-case type. For ``"GB"`` / ``"VGB"`` the WHOLE
                column of every dataset is returned (``binary_id`` is
                ignored).

        Returns:
            dict: ``{dataset name: value}``, with the catalogue's own names
            and units (a scalar per parameter, or the full column for GB /
            VGB).
        """
        params = {}
        for key in group.keys():
            try:
                if source_type in ["VGB", "GB"]:
                    ds = group[key][:]
                else:
                    ds = group[key][binary_id]

                if ds.dtype.kind == "O":  # object type (e.g. variable-length strings)
                    val = ds
                    if isinstance(val, bytes):
                        val = val.decode()
                else:
                    val = ds[()]
                params[key] = val
            except Exception as e:
                raise e
        return params

    def _open(self, file_path: str) -> MojitoL1File:
        """Open a brick as a context manager; subclasses return a ``MojitoL1File``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("_open method should be implemented in subclasses.")

    @property
    def individual_timeseries(self) -> dict | None:
        """Per-stream copies kept when ``store_individual_timeseries`` is set.

        Keys: ``"COMBINED_DATA"``, ``"NOISE"``, ``"GALFOR"`` and
        ``"{TYPE}_{id}"`` streams of shape ``(3, n_times)``; from the NOISE
        brick also ``"PSD_MATRIX"`` (its ``noise_estimates.xyz`` divided by
        ``laser_frequency**2``), ``"PSD_FREQUENCIES"`` and ``"PSD_TIMES"``.
        :meth:`L1ProcessingStep.process` adds ``"TIMES"`` and ``"COMBINED"``
        (the processed data).

        Returns:
            dict | None: The mapping, or ``None`` when nothing was stored.
        """
        if not hasattr(self, "_individual_timeseries"):
            return None
        return self._individual_timeseries

    def load_data(self) -> tuple:
        """Read the data stream(s), the orbits and the catalogue rows.

        Order: ``COMBINED`` (if listed), then ``NOISE``, then ``GALFOR``,
        then the per-class types in ``source_types`` order. The first brick
        read sets the time grid and builds the orbits (eagerly configured
        for COMBINED / NOISE / GALFOR, lazily otherwise); every later brick
        must be on the same grid. Under ``COMBINED`` the per-class bricks are
        not opened (their catalogue rows are still read).

        Side effects: fills :attr:`catalogue`, :attr:`missing_source_bricks`
        and (with ``store_individual_timeseries``) the per-stream store, and
        REMOVES ``COMBINED`` / ``NOISE`` / ``GALFOR`` from
        :attr:`source_types`, so a second call does not read them again.

        Returns:
            tuple: ``(tdi_times, tdi_fs, xyz, orbits)``: ``tdi_times``
            ``np.ndarray`` ``(n_times,)`` [s]; ``tdi_fs`` float [Hz]; ``xyz``
            ``np.ndarray`` ``(3, n_times)`` dimensionless TDI X, Y, Z (summed
            over the streams read); ``orbits`` the ``orbits_class`` instance.

        Raises:
            ValueError: ``COMBINED`` listed with ``NOISE`` or ``GALFOR``
                (double counting), or a per-class type has an empty id list.
            AttributeError: ``source_ids`` is ``None`` and an ``MBHB`` /
                ``EMRI`` / ``SOBHB`` type is listed.
            FileNotFoundError: A brick or catalogue file is missing (a
                per-class brick only without ``allow_missing_bricks``).
            RuntimeError: No brick could be loaded at all.
            AssertionError: Bricks on different time grids.
        """
        xyz = None
        orbits: Orbits = None

        if self.store_individual_timeseries:
            _individual_timeseries = (
                {}
            )  # to store individual timeseries for each source type and ID, if needed for debugging or further analysis.

        import time as _t

        # COMBINED: mojito's pre-summed stream. Same role as NOISE below --
        # it establishes ``xyz`` and the orbits -- but it ALREADY contains
        # every source class, so the per-class loop must not add their
        # waveforms on top. It still runs, to read their CATALOGUES: those
        # feed VGB seeding, nleaves sizing and the F-stat overlays, and are
        # metadata rather than data.
        combined_base = "COMBINED" in self.source_types
        if combined_base:
            if "NOISE" in self.source_types:
                raise ValueError(
                    "source_types lists both COMBINED and NOISE: the combined "
                    "stream already contains the instrument noise, so adding "
                    "the NOISE brick would double-count it. Drop NOISE."
                )
            subfolder = os.path.join(self.data_folder, "COMBINED", "L1")
            file_path = find_combined_file(subfolder)

            _t0 = _t.perf_counter()
            orbits = self.orbits_class(file_path, **(self.orbits_kwargs or {}))
            orbits._ensure_configured()
            logger.debug(
                f"[load_data] COMBINED orbits took {_t.perf_counter()-_t0:.2f}s"
            )
            logger.info("Initialized orbits from COMBINED file.")

            with self._open(file_path) as f:
                _t0 = _t.perf_counter()
                xyz = f.tdis.xyz_doppler[:]
                logger.debug(
                    f"[load_data] COMBINED f.tdis.xyz_doppler[:] took "
                    f"{_t.perf_counter()-_t0:.2f}s shape={xyz.shape}"
                )
                tdi_dt = f.tdis.time_sampling.dt
                tdi_fs = f.tdis.time_sampling.fs
                tdi_times = f.tdis.time_sampling.t()

                if self.store_individual_timeseries:
                    _individual_timeseries["COMBINED_DATA"] = xyz.T.copy()

            self.source_types.remove("COMBINED")

            if self.verbose:
                logger.info(f"Loaded COMBINED data from {file_path}")
                logger.info(
                    "data, times and orbits initialized from COMBINED file; "
                    "remaining source types contribute CATALOGUES ONLY "
                    "(their signal is already in the combined stream)."
                )
                logger.info(f"TDI time step: {tdi_dt} seconds")
                logger.info(f"TDI sampling frequency: {tdi_fs} Hz")

        if "NOISE" in self.source_types:
            subfolder = os.path.join(self.data_folder, "INSTRUMENT", "L1")
            file_path = find_file(subfolder, "NOISE", 00)

            if orbits is None:
                _t0 = _t.perf_counter()
                orbits = self.orbits_class(file_path, **(self.orbits_kwargs or {}))
                logger.debug(f"[load_data] L1Orbits(...) (incl. _setup) took {_t.perf_counter()-_t0:.2f}s")
                _t0 = _t.perf_counter()
                # Eagerly build the interpolation grid here (data-load time is
                # the right place to pay the one-time cost); equivalent to the
                # lazy first-use configuration.
                orbits._ensure_configured()
                logger.debug(f"[load_data] orbits._ensure_configured() took {_t.perf_counter()-_t0:.2f}s")
                logger.info(f"Initialized orbits from NOISE file.")

            with self._open(file_path) as f:
                _t0 = _t.perf_counter()
                xyz = f.tdis.xyz_doppler[:]
                logger.debug(f"[load_data] NOISE f.tdis.xyz_doppler[:] took {_t.perf_counter()-_t0:.2f}s shape={xyz.shape}")

                tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                tdi_times = f.tdis.time_sampling.t()

                if self.store_individual_timeseries:
                    noise_covariance = f.noise_estimates.xyz[:] / (f.laser_frequency**2)
                    noise_frequencies = f.noise_estimates.freq_sampling.f()
                    noise_times = f.noise_estimates.time_sampling.t()

                    _individual_timeseries["NOISE"] = (
                        xyz.T.copy()
                    )  # store the noise timeseries separately if needed
                    _individual_timeseries["PSD_MATRIX"] = noise_covariance
                    _individual_timeseries["PSD_FREQUENCIES"] = noise_frequencies
                    _individual_timeseries["PSD_TIMES"] = noise_times

            self.source_types.remove("NOISE")

            if self.verbose:
                logger.info(f"Loaded NOISE data from {file_path}")
                logger.info(f"data, times and orbits initialized from NOISE file.")
                logger.info(f"TDI time step: {tdi_dt} seconds")
                logger.info(f"TDI sampling frequency: {tdi_fs} Hz")

        # GALFOR: mojito's galactic-confusion foreground stream
        # (``data/GALFOR/L1``). Handled HERE, next to NOISE, and NOT in the
        # per-class loop below, because it is a STOCHASTIC STREAM, not a
        # source class: it has no catalogue of binaries, no source ids, and
        # its brick carries no ``source{id}_`` infix to resolve by. Like
        # NOISE it either establishes ``xyz`` (and the orbits) or is SUMMED
        # into it -- so NOISE + GALFOR is instrument noise plus the
        # foreground, on the same grid.
        #
        # WHAT IT ACTUALLY CONTAINS (read the brick's own provenance before
        # interpreting a fit against it): the mojito-light GALFOR brick is
        # derived_from the GB brick with "GalacticStochastic resolvable
        # binaries subtracted" -- it is the REAL unresolved-GB confusion
        # residual, not a draw from the analytic tanh foreground the galfor
        # branch fits. So a galfor fit against it is a MODEL-ADEQUACY test,
        # and the branch's ``galfor_injection`` is a reference curve, not a
        # truth to recover. The instrument ``psd_injection`` IS still truth.
        if "GALFOR" in self.source_types:
            if combined_base:
                raise ValueError(
                    "source_types lists both COMBINED and GALFOR: the "
                    "combined stream already contains the galactic "
                    "foreground, so adding the GALFOR brick would "
                    "double-count it. Drop GALFOR."
                )
            subfolder = os.path.join(self.data_folder, "GALFOR", "L1")
            file_path = find_stream_file(subfolder, "GALFOR", "MOJITO_GALFOR_FILE")

            if orbits is None:
                orbits = self.orbits_class(file_path, **(self.orbits_kwargs or {}))
                orbits._ensure_configured()
                logger.info("Initialized orbits from GALFOR file.")

            with self._open(file_path) as f:
                _t0 = _t.perf_counter()
                _xyz = f.tdis.xyz_doppler[:]
                logger.debug(
                    f"[load_data] GALFOR f.tdis.xyz_doppler[:] took "
                    f"{_t.perf_counter()-_t0:.2f}s shape={_xyz.shape}"
                )
                _tdi_dt = f.tdis.time_sampling.dt
                _tdi_fs = f.tdis.time_sampling.fs
                _tdi_times = f.tdis.time_sampling.t()

                if xyz is None:
                    xyz, tdi_dt, tdi_fs, tdi_times = _xyz, _tdi_dt, _tdi_fs, _tdi_times
                else:
                    # Same assertions the per-class loop makes: a GALFOR brick
                    # on a different grid than the NOISE brick would sum into
                    # nonsense silently.
                    assert tdi_dt == _tdi_dt, "Time steps do not match between files."
                    assert (
                        tdi_fs == _tdi_fs
                    ), "Sampling frequencies do not match between files."
                    assert (
                        tdi_times == _tdi_times
                    ).all(), "Time arrays do not match between files."
                    xyz = xyz + _xyz

                if self.store_individual_timeseries:
                    _individual_timeseries["GALFOR"] = _xyz.T.copy()

            self.source_types.remove("GALFOR")

            if self.verbose:
                logger.info(f"Loaded GALFOR data from {file_path}")
                logger.info(
                    "galactic-foreground stream summed into the data "
                    "(unresolved-GB confusion residual)."
                )

        for source_type in self.source_types:

            subfolder = os.path.join(self.data_folder, source_type, "L1")

            if source_type in ["GB", "VGB"]:
                ids = [0]  # only one file for GB/VGB
            else:
                ids = self.source_ids.get(source_type, [])

            if not isinstance(ids, list):
                ids = [ids]
            if len(ids) == 0:
                raise ValueError(f"No source IDs provided for source type '{source_type}'.")

            catalogue_path = os.path.join(
                self.catalogues_folder, self.catalogues_map.get(source_type)
            )

            self.catalogue[source_type] = {}

            if self.verbose:
                logger.info(f"Loading data for source type '{source_type}'")

            with h5py.File(catalogue_path, "r") as catalogue_file:
                binary_params = catalogue_file["Binaries"]
                for source_id in tqdm(
                    ids, desc=f"Loading {source_type} sources", disable=not self.verbose
                ):
                    # Catalogue parameters load UNCONDITIONALLY -- for a
                    # missing brick under allow_missing_bricks they are what
                    # the synthetic fill builds the source from.
                    self.catalogue[source_type][source_id] = self.load_single_binary(
                        binary_params, source_id, source_type
                    )
                    if self.verbose:
                        logger.info(
                            f"Loaded catalogue parameters for {source_type} source ID {source_id} from catalogue."
                        )

                    if combined_base:
                        # Catalogue read; the waveform is ALREADY in the
                        # combined stream, so loading the brick would
                        # double-count it (and cost a multi-GB read).
                        continue

                    # Resolve the per-class brick ONLY when its waveform is
                    # actually needed: under COMBINED the signal is already in
                    # the data, so requiring the file to exist would make the
                    # combined stream depend on bricks it does not read.
                    try:
                        file_path = find_file(subfolder, source_type, source_id)
                    except FileNotFoundError:
                        if not self.allow_missing_bricks:
                            raise
                        self.missing_source_bricks.append(
                            (source_type, source_id)
                        )
                        logger.warning(
                            f"[HYBRID] {source_type} source {source_id}: no "
                            "L1 brick found -- catalogue parameters kept; "
                            "the caller must synthesize this source's "
                            "stream (allow_missing_bricks=True)."
                        )
                        continue

                    with self._open(file_path) as f:
                        _t0 = _t.perf_counter()
                        _xyz = f.tdis.xyz_doppler[:]
                        logger.debug(f"[load_data] {source_type} source {source_id} f.tdis.xyz_doppler[:] took {_t.perf_counter()-_t0:.2f}s shape={_xyz.shape}")
                        _tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                        _tdi_times = f.tdis.time_sampling.t()
                        _tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                        if xyz is None:
                            xyz = _xyz
                            tdi_dt = _tdi_dt
                            tdi_times = _tdi_times
                            tdi_fs = _tdi_fs

                            if orbits is None:
                                orbits: Orbits = self.orbits_class(
                                    file_path, **(self.orbits_kwargs or {})
                                )  # configures lazily on first use
                                logger.info(f"Initialized orbits from {source_type} file.")

                        else:
                            xyz += _xyz
                            assert tdi_dt == _tdi_dt, "Time steps do not match between files."
                            assert tdi_fs == _tdi_fs, "Sampling frequencies do not match between files."
                            assert (
                                tdi_times == _tdi_times
                            ).all(), "Time arrays do not match between files."

                        if self.store_individual_timeseries:
                            _individual_timeseries[f"{source_type}_{source_id}"] = (
                                _xyz.T.copy()
                            )  # store individual timeseries for this source

        if xyz is None:
            raise RuntimeError(
                "No L1 brick could be loaded at all"
                + (f" (missing: {self.missing_source_bricks})"
                   if self.missing_source_bricks else "")
                + "; the hybrid fill still needs at least one real brick "
                "for the time grid and orbits -- use data_mode='synthetic' "
                "instead."
            )

        xyz = xyz.T  # Transpose to have shape (n_channels, n_times)
        assert (
            xyz.shape[1] == tdi_times.shape[0]
        ), "Data time dimension does not match time array length."

        assert orbits is not None, "Orbits were not initialized from any file."

        if self.store_individual_timeseries:
            self._individual_timeseries = (
                _individual_timeseries  # store the individual timeseries for potential further use
            )

        return tdi_times, tdi_fs, xyz, orbits

    # TODO check: keep/remove/document -- debug dump; no caller in src/scripts/tests/LATW
    def dump_individual_timeseries(self, file_path: str, delete_after_dump: bool = False):
        """
        Dump the individual timeseries for each source type and ID to a .h5 file for debugging or further analysis.

        Args:
            file_path (str): The path to the output .h5 file.
            delete_after_dump (bool, optional): Whether to delete the individual timeseries after dumping them. Defaults to False.
        """
        with h5py.File(file_path, "w") as f:
            for key, ts_data in self.individual_timeseries.items():
                f.create_dataset(key, data=ts_data)
        if self.verbose:
            logger.info(f"Dumped individual timeseries to {file_path}")

        if delete_after_dump:
            self.individual_timeseries.clear()

    # TODO check: keep/remove/document -- debug dump; no caller; np.string_ is gone in NumPy 2
    def dump_catalogue(self, file_path: str):
        """
        Dump the loaded catalogue parameters to a .h5 file for debugging or further analysis.

        Args:
            file_path (str): The path to the output .h5 file.
        """
        with h5py.File(file_path, "w") as f:
            for source_type, sources in self.catalogue.items():
                grp = f.create_group(source_type)
                for source_id, params in sources.items():
                    subgrp = grp.create_group(f"source_{source_id}")
                    for param_key, param_value in params.items():
                        if isinstance(param_value, str):
                            param_value = np.string_(param_value)  # convert to bytes for h5py
                        subgrp.create_dataset(param_key, data=param_value)
        if self.verbose:
            logger.info(f"Dumped catalogue parameters to {file_path}")


class SangriaDataLoader:
    """Loader for an LDC Sangria HDF5 file (``data_mode="sangria"``).

    Reads the structured ``obs/tdi`` dataset (fields ``t``, ``X``, ``Y``,
    ``Z``) and, for removals, the per-component ``sky/<component>/tdi``
    datasets of the same layout. Base of :class:`SangriaProcessingStep`;
    no I/O at construction.

    Args:
        data_input_path: Path of the Sangria HDF5 file
            (``general.ldc_source_file`` in the stock fits).
        remove_from_data: Components to drop. A ``sky`` component name
            (e.g. ``"vgb"``, ``"mbhb"``) is subtracted from ``obs/tdi``;
            ``"noise"`` switches to summing the ``sky`` components not
            listed (a noise-free stream). Default ``None`` (keep all).

    Attributes:
        data_input_path (str): Constructor argument.
        remove_from_data (list | None): Constructor argument (``None``
            becomes ``[]`` in :meth:`load_data`).
    """

    def __init__(
        self,
        data_input_path: str,
        remove_from_data: Optional[list] = None,
    ):
        self.data_input_path = data_input_path
        self.remove_from_data = remove_from_data

    def _open(self, file_path: str) -> h5py.File:
        """Open the Sangria file read-only."""
        return h5py.File(file_path, "r")

    def load_data(
        self,
    ):
        """Read the Sangria TDI stream with the requested components removed.

        Returns:
            tuple: ``(times, fs, data_xyz, None)``: ``times`` ``np.ndarray``
            ``(n_times,)`` [s]; ``fs`` float [Hz] from the first time step;
            ``data_xyz`` ``np.ndarray`` ``(3, n_times)`` TDI X, Y, Z; no
            orbits (``None``).

        Raises:
            AssertionError: ``remove_from_data`` is not a list.
        """
        if self.remove_from_data is None:
            self.remove_from_data = []

        assert isinstance(self.remove_from_data, list)

        # TODO: Generalize input.
        with self._open(self.data_input_path) as f:
            if "noise" not in self.remove_from_data:
                tXYZ = f["obs"]["tdi"][:]

                # remove sources
                for source in self.remove_from_data:  # , "dgb", "igb"]:  # "vgb" ,
                    if source == "noise":
                        continue
                    logger.info(f"Removing {source} from data injection.")
                    change_arr = f["sky"][source]["tdi"][:]
                    for change in ["X", "Y", "Z"]:
                        tXYZ[change] -= change_arr[change]

            else:
                keys = list(f["sky"])
                logger.info("Initial keys in data injection: %s", keys)
                tmp_keys = keys.copy()
                for key in tmp_keys:
                    logger.debug(key)
                    if key in self.remove_from_data:
                        keys.remove(key)
                        logger.info(f"Removing {key} from data injection.")
                tXYZ = f["sky"][keys[0]]["tdi"][:]
                for key in keys[1:]:
                    tXYZ["X"] += f["sky"][key]["tdi"][:]["X"]
                    tXYZ["Y"] += f["sky"][key]["tdi"][:]["Y"]
                    tXYZ["Z"] += f["sky"][key]["tdi"][:]["Z"]

        times, X, Y, Z = (
            tXYZ["t"].squeeze(),
            tXYZ["X"].squeeze(),
            tXYZ["Y"].squeeze(),
            tXYZ["Z"].squeeze(),
        )
        data_xyz = np.vstack([X, Y, Z])  # shape (n_channels, n_times)
        dt = times[1] - times[0]
        fs = 1.0 / dt

        return times, fs, data_xyz, None  # no orbits available


# todo: add catalogue loading functionality for signal parameters.


class SignalProcessor:
    """Time-domain conditioning of multi-channel data. Credits: Ollie Burke.

    Holds the stream and provides the filter / downsample / trim steps that
    :meth:`BaseProcessingStep.process` chains. Filtering and resampling
    delegate to ``MojitoProcessor.SignalProcessor`` with channels named
    ``X``, ``Y``, ``Z``. The filter methods return the filtered array without
    storing it; :meth:`downsample` and :meth:`trim` update the instance.

    Args:
        times: Time samples [s], shape ``(n_times,)``, uniformly spaced.
        data: Channel data, shape ``(n_channels, n_times)``.
        fs: Sampling frequency [Hz].
        verbose: Log each step at INFO. Default ``True``.
        do_plots: Save a before/after PNG of the X channel at each filter or
            trim step (folder from the ``plot_folder`` keyword of the step).
            Default ``False``.

    Attributes:
        times (np.ndarray): Current time samples [s].
        data (np.ndarray): Current ``(n_channels, n_times)`` data.
        dt (float): Sample spacing ``times[1] - times[0]`` [s].
        N (int): Number of samples.
        T (float): Duration [s]: ``N * dt`` at construction, then
            ``times[-1] - times[0]`` after a trim or downsample.
        fs (float): Sampling frequency [Hz].
        verbose (bool): Constructor argument.
        do_plots (bool): Constructor argument.
    """

    _CHANNEL_NAMES = ["X", "Y", "Z"]

    def __init__(
        self,
        times: np.ndarray,
        data: np.ndarray,
        fs: float,
        verbose: bool = True,
        do_plots: bool = False,
    ):

        self.times = times
        self.dt = times[1] - times[0]
        self.N = times.shape[0]
        self.T = self.N * self.dt # total duration. Takes into account that times[0] may not be zero.
        self.data = data
        self.fs = fs
        self.verbose = verbose
        self.do_plots = do_plots

        self._original_t0 = float(times[0])  # store original t0 for reference in case of trimming

    @property
    def original_t0(self) -> float:
        """Original start time of the data before any trimming."""
        return self._original_t0

    def _to_mp_dict(self):
        """ndarray (n_ch, N) -> {'X': arr, 'Y': arr, 'Z': arr}"""
        return {ch: self.data[i] for i, ch in enumerate(self._CHANNEL_NAMES[: self.data.shape[0]])}

    def _from_mp_dict(self, mp_data):
        """{'X': arr, 'Y': arr, 'Z': arr} -> ndarray (n_ch, N)"""
        return np.vstack([mp_data[ch] for ch in self._CHANNEL_NAMES if ch in mp_data])

    def _filter_via_mp(
        self, *, low=None, high=None, order=2, filter_type="butterworth", zero_phase=True
    ):
        """Filter a copy of :attr:`data` with ``MojitoProcessor`` (``low``/``high`` cutoffs in Hz).

        Returns:
            np.ndarray: Filtered ``(n_channels, n_times)`` array (``data`` is
            not modified).
        """
        mp = MPSignalProcessor(self._to_mp_dict(), fs=self.fs)
        mp.filter(low=low, high=high, order=order, filter_type=filter_type, zero_phase=zero_phase)
        return self._from_mp_dict(mp.data)

    def _plot_filter(self, filtered, btype, order, filter_type, **kwargs):
        """Plot original vs filtered X channel."""
        decimate_factor = kwargs.get("decimate_factor", 1000)
        plt.figure(figsize=(10, 4))
        plt.plot(
            self.data[0, ::decimate_factor],
            label="Original X channel",
            c="gray",
            alpha=0.5,
        )
        plt.plot(filtered[0, ::decimate_factor], label="Filtered X channel", c="blue")
        plt.xlabel("Sample Index")
        plt.ylabel("Strain")
        plt.title(f"Signal {btype.capitalize()} Filtering ({filter_type}, order={order})")
        plt.legend()
        plt.savefig(os.path.join(kwargs.get("plot_folder", "."), f"{btype}_filter_signal.png"))
        plt.close()

    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Apply a band-pass filter to all channels.

        Args:
            low: Lower cutoff frequency [Hz].
            high: Upper cutoff frequency [Hz].
            order: Filter order. Default ``2``.
            filter_type: ``"butterworth"``, ``"chebyshev1"``, ``"chebyshev2"``
                or ``"bessel"``. Default ``"butterworth"``.
            zero_phase: Zero-phase (forward-backward) filtering if ``True``,
                else single pass. Default ``True``.
            **kwargs: Plot options when ``do_plots`` (``plot_folder``,
                ``decimate_factor``); others are ignored.

        Returns:
            np.ndarray: Filtered ``(n_channels, n_times)`` data (:attr:`data`
            is not modified).
        """
        filtered = self._filter_via_mp(
            low=low, high=high, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "bandpass", order, filter_type, **kwargs)
        return filtered

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Apply a low-pass filter to all channels.

        Args:
            cutoff: Cutoff frequency [Hz].
            order: Filter order. Default ``2``.
            filter_type: Filter family (see :meth:`bandpass_filter`). Default
                ``"butterworth"``.
            zero_phase: Zero-phase filtering if ``True``. Default ``True``.
            **kwargs: Plot options when ``do_plots`` (``plot_folder``,
                ``decimate_factor``); others are ignored.

        Returns:
            np.ndarray: Filtered ``(n_channels, n_times)`` data (:attr:`data`
            is not modified).
        """
        filtered = self._filter_via_mp(
            high=cutoff, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "lowpass", order, filter_type, **kwargs)
        return filtered

    def highpass_filter(
        self,
        cutoff: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Apply a high-pass filter to all channels.

        The engine's default ``highpass_kwargs`` (cutoff 2e-5 Hz, order 2,
        zero phase) run this for the data modes that keep the engine
        defaults (``sangria``); the production mojito runs skip it.

        Args:
            cutoff: Cutoff frequency [Hz].
            order: Filter order. Default ``2``.
            filter_type: Filter family (see :meth:`bandpass_filter`). Default
                ``"butterworth"``.
            zero_phase: Zero-phase filtering if ``True``. Default ``True``.
            **kwargs: Plot options when ``do_plots`` (``plot_folder``,
                ``decimate_factor``); others are ignored.

        Returns:
            np.ndarray: Filtered ``(n_channels, n_times)`` data (:attr:`data`
            is not modified).
        """
        filtered = self._filter_via_mp(
            low=cutoff, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "highpass", order, filter_type, **kwargs)
        return filtered

    def _update_params(self):
        """Update internal parameters after data modification."""
        self.N = self.data.shape[1]
        self.T = self.times[-1] - self.times[0]

        if self.verbose:
            logger.info(f"Updated parameters: N={self.N}, T={self.T}")

    def downsample(self, target_fs, window=("kaiser", 31.0), padtype="line"):
        """Downsample all channels to ``target_fs`` via polyphase filtering.

        Delegates to ``MojitoProcessor.SignalProcessor.downsample``. Updates
        :attr:`data`, :attr:`fs`, :attr:`dt`, :attr:`times` (rebuilt from
        ``times[0]``) and :attr:`N` / :attr:`T`. A ``target_fs`` equal to the
        current rate is a no-op.

        Args:
            target_fs (float): Target sampling frequency [Hz].
            window (tuple): Anti-aliasing window. Default
                ``("kaiser", 31.0)``.
            padtype (str): Padding type of the resampling. Default
                ``"line"``.

        Returns:
            np.ndarray: The downsampled ``(n_channels, n_times)`` data.

        Raises:
            ValueError: ``target_fs`` above the current rate.
        """
        if target_fs == self.fs:
            if self.verbose:
                logger.info(
                    f"Target sampling frequency is the same as current fs ({self.fs} Hz). No downsampling applied."
                )
            return self.data
        elif target_fs > self.fs:
            raise ValueError(
                f"Target sampling frequency ({target_fs} Hz) must be less than current fs ({self.fs} Hz) for downsampling."
            )

        mp = MPSignalProcessor(self._to_mp_dict(), fs=self.fs)
        mp.downsample(target_fs=target_fs, window=window, padtype=padtype)
        self.data = self._from_mp_dict(mp.data)
        self.fs = mp.fs
        self.dt = 1.0 / self.fs
        self.times = self.times[0] + np.arange(self.data.shape[1]) * self.dt
        self._update_params()
        return self.data

    def trim(
        self,
        duration: float,
        is_percent: bool = False,
        trimming_type: str = "from_each_end",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trim the data in time and update the instance.

        ``duration`` is converted to ``int(duration / dt)`` samples. Modes:

        - ``"from_each_end"``: drop that many samples at both ends (the
          engine default trims 200 h this way after filtering).
        - ``"discard_from_start"``: drop them at the start.
        - ``"keep_from_start"``: keep only the first that many samples (the
          ``Tobs`` cut of :meth:`BaseProcessingStep.process`).

        Args:
            duration: Seconds, or a fraction of :attr:`T` in ``[0, 1]`` when
                ``is_percent``.
            is_percent: Treat ``duration`` as a fraction of the duration.
                Default ``False``.
            trimming_type: One of the modes above. Default
                ``"from_each_end"``.
            **kwargs: Plot options when ``do_plots`` (``plot_folder``,
                ``filename``, ``decimate_factor``); others are ignored.

        Returns:
            tuple[np.ndarray, np.ndarray]: The trimmed ``times`` and ``data``
            (also stored on the instance).

        Raises:
            ValueError: The trim removes (or keeps) at least the whole data.
            AssertionError: Unknown ``trimming_type``, or ``is_percent`` with
                ``duration`` outside ``[0, 1]``.
        """

        assert trimming_type in [
            "from_each_end",
            "keep_from_start",
            "discard_from_start",
        ]  # todo find proper name and implement

        if is_percent:
            assert 0 <= duration <= 1, "Percentage duration must be between 0 and 1."
            duration = duration * self.T

        trim_samples = int(duration / self.dt)

        if trimming_type == "from_each_end":
            if 2 * trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {2*duration}s from {self.T}s data. "
                    f"Total trim exceeds data length."
                )
            if self.verbose:
                logger.info(
                    f"Trimming {duration}s ({trim_samples} samples) from each end of the data."
                )

            trimmed_data = self.data[:, trim_samples:-trim_samples]
            trimmed_times = self.times[trim_samples:-trim_samples]

        elif trimming_type == "discard_from_start":
            if trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {duration}s from {self.T}s data. " f"Trim exceeds data length."
                )
            if self.verbose:
                logger.info(
                    f"Trimming {duration}s ({trim_samples} samples) from the start of the data."
                )

            trimmed_data = self.data[:, trim_samples:]
            trimmed_times = self.times[trim_samples:]

        else:  # keep_from_start
            if trim_samples >= self.N:
                raise ValueError(
                    f"Cannot keep {duration}s from {self.T}s data. "
                    f"Keep duration exceeds data length."
                )
            if self.verbose:
                logger.info(f"Keeping first {duration}s ({trim_samples} samples) of the data.")

            trimmed_data = self.data[:, :trim_samples]
            trimmed_times = self.times[:trim_samples]

        if self.do_plots:
            decimate_factor = kwargs.get(
                "decimate_factor", 1000
            )  # Decimate for plotting if too many points
            plt.figure(figsize=(10, 4))
            plt.plot(
                self.times[::decimate_factor],
                self.data[0, ::decimate_factor],
                label="Original X channel",
                c="gray",
                alpha=0.5,
            )
            plt.plot(
                trimmed_times[::decimate_factor],
                trimmed_data[0, ::decimate_factor],
                label="Trimmed X channel",
                c="blue",
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Strain")
            plt.title("Signal Trimming")
            plt.legend()
            plt.savefig(
                os.path.join(
                    kwargs.get("plot_folder", "."),
                    kwargs.get("filename", "trim_signal.png"),
                )
            )
            plt.close()

        # Update internal state
        self.data = trimmed_data
        self.times = trimmed_times
        self._update_params()

        return trimmed_times, trimmed_data


class BaseProcessingStep(SignalProcessor):
    """:class:`SignalProcessor` extended with :meth:`process` and :meth:`pour` helpers.

    The interface every global-fit data processor implements
    (``GeneralSettings.data_processor_class``). Subclasses load the data in
    their constructor and hand ``(times, data, fs)`` to this base; they also
    set ``orbits`` (``None`` when the source has none). The engine then calls
    :meth:`process` (filters / downsample / trims, then builds
    :attr:`td_signal`) and :meth:`pour` (projection onto the analysis
    domain), and the run calls :meth:`release_transients` once sampling
    starts.

    Args:
        *args: Forwarded to :class:`SignalProcessor` (``times``, ``data``,
            ``fs``, ``verbose``, ``do_plots``).
        **kwargs: Forwarded to :class:`SignalProcessor`.

    Attributes:
        td_signal (lisatools.domains.TDSignal): The processed stream, set by
            :meth:`process` (absent before it runs).
    """

    def __init__(self, *args, **kwargs):
        SignalProcessor.__init__(self, *args, **kwargs)

    def __repr__(self):
        return (
            f"ProcessingStep with {self.N} samples, "
            f"dt={self.dt:.3e}s, T={self.T:.3e}s, "
            f"fs={self.fs:.3e}Hz, "
            f"and channels: {self.data.shape[0]}"
        )

    def release_transients(self) -> None:
        """Drop large intermediate arrays kept only for data production.

        Called once by :meth:`~lisatools.globalfit.run.GlobalFit.setup_acs`
        at the build->sampling transition, after the produced data/residuals
        have been installed
        into the shared :class:`AnalysisContainerArray`. Subclasses that
        cache big device (or host) arrays during production — waveform
        buffers, full-rate streams already poured into the ACA — should
        override this to ``del``/None them so the following per-device
        memory-pool sweep can actually return the memory. The default is a
        no-op; settings-relevant attributes (e.g. ``td_signal.settings``)
        must survive.
        """

    def process(
        self,
        highpass_kwargs: dict = None,
        lowpass_kwargs: dict = None,
        bandpass_kwargs: dict = None,
        downsample_kwargs: dict = None,
        trim_kwargs: dict = None,
        Tobs: float = None,
        **kwargs,
    ) -> tuple:
        """Apply the configured conditioning steps, then build :attr:`td_signal`.

        Steps run in this order, each only when its kwargs are not ``None``:
        high-pass, low-pass, band-pass (each result replaces :attr:`data`),
        :meth:`downsample`, :meth:`trim`. If a filter ran but ``trim_kwargs``
        is ``None``, 100 h are trimmed from each end (with a warning). Then
        ``Tobs`` keeps the first ``Tobs`` seconds (``keep_from_start``). A
        ``Tobs`` longer than the data is clamped to :attr:`T` with a warning
        that says no trim is applied, but the trim still runs on
        ``int(T / dt)`` samples: with ``T = N * dt`` (no earlier trim or
        downsample) that is normally ``N``, which raises ``ValueError``;
        after a trim (``T = times[-1] - times[0]``) it drops the last sample.
        Finally :attr:`td_signal` is built on
        ``TDSettings(t0=times[0], dt, N)`` with the backend of :attr:`data`.

        Called by :meth:`~lisatools.globalfit.engine.GeneralSetup.init_data_information`
        with ``preprocess_kwargs``. The engine defaults are a 2e-5 Hz
        high-pass, a 200 h trim from each end and ``Tobs`` = the run's
        ``Tobs``; ``all_sources`` in mojito / synthetic mode sets all three
        to ``None``, so in the production runs only :attr:`td_signal` is
        built.

        Args:
            highpass_kwargs (dict, optional): Keyword arguments of
                :meth:`highpass_filter`.
            lowpass_kwargs (dict, optional): Keyword arguments of
                :meth:`lowpass_filter`.
            bandpass_kwargs (dict, optional): Keyword arguments of
                :meth:`bandpass_filter`.
            downsample_kwargs (dict, optional): Keyword arguments of
                :meth:`downsample`, e.g. ``{"target_fs": 1.0}``. Applied after
                filtering and before trimming.
            trim_kwargs (dict, optional): Keyword arguments of :meth:`trim`.
            Tobs (float, optional): Keep only the first ``Tobs`` seconds.
            **kwargs: Plot options forwarded to the filter and trim steps
                (``plot_folder``, ``decimate_factor``); unknown keys (such as
                the stock ``normalize``) are ignored.

        Returns:
            tuple[np.ndarray, np.ndarray]: The processed :attr:`times` [s]
            and :attr:`data` ``(n_channels, n_times)``.
        """
        filtered = False

        if highpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying highpass filter...")
            self.data = self.highpass_filter(**highpass_kwargs, **kwargs)
            filtered = True

        if lowpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying lowpass filter...")
            self.data = self.lowpass_filter(**lowpass_kwargs, **kwargs)
            filtered = True

        if bandpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying bandpass filter...")
            self.data = self.bandpass_filter(**bandpass_kwargs, **kwargs)
            filtered = True

        if downsample_kwargs is not None:
            if self.verbose:
                logger.info("Downsampling data...")
            self.downsample(**downsample_kwargs)

        if trim_kwargs is not None:
            if self.verbose:
                logger.info("Trimming data...")
            _, _ = self.trim(**trim_kwargs, **kwargs)

        elif trim_kwargs is None and filtered:
            logger.warning(
                "Data was filtered but no trim_kwargs provided. Applying default trimming of 100 hours from each end to avoid edge effects."
            )
            trim_duration = 100 * 3600  # 100 hours in seconds
            _, _ = self.trim(
                duration=trim_duration,
                is_percent=False,
                trimming_type="from_each_end",
                **kwargs,
            )

        if Tobs is not None:
            if Tobs > self.T:
                logger.warning(
                    f"Requested observation time Tobs={Tobs}s exceeds total data duration T={self.T}s. No trimming applied."
                )
                Tobs = self.T

            _, _ = self.trim(
                duration=Tobs,
                is_percent=False,
                trimming_type="keep_from_start",
                filename="science_data.png",
                **kwargs,
            )

        xp = get_array_module(self.data)
        
        settings = TDSettings(
            t0=self.times[0],
            dt=self.dt,
            N=self.N,
            force_backend="cpu" if xp is np else "gpu",
            # xp = get_array_module(self.data)
        )
        self.td_signal = TDSignal(arr=self.data, settings=settings)

        return self.times, self.data

    def pour(
        self,
        settings: DomainSettingsBase,
        window: Optional[np.ndarray | str] = None,
        return_orbits: bool = False,
    ) -> "DomainBase | tuple[DomainBase, Orbits]":
        """Project the processed :attr:`td_signal` onto the analysis domain.

        If ``settings`` equals ``td_signal.settings`` the TD signal is
        returned as is. Otherwise the TD stream is first cut from its start
        to the target length (``round(1 / (df * dt))`` samples for
        :class:`~lisatools.domains.FDSettings`, ``settings.N`` otherwise) and
        transformed with ``window``. Called by the engine with the resolved
        domain (WDM in the production runs) and the analysis window
        (``general.window_type``, default Tukey) on the TD grid. Requires
        :meth:`process` to have run.

        Args:
            settings: Target domain settings (WDM, FD, STFT or TD).
            window: Window applied in the transform (an array on the TD grid,
                or a name the transform accepts). Default ``None``.
            return_orbits: Also return :attr:`orbits`. Default ``False``.

        Returns:
            DomainBase | tuple[DomainBase, Orbits]: The data as a
            :class:`~lisatools.domains.DomainBase` child (``WDMSignal``,
            ``FDSignal``, ``STFTSignal`` or ``TDSignal``), plus the orbits
            when ``return_orbits``.

        Raises:
            AssertionError: The target needs more samples than the data has.
        """

        # The TD signal is already a DomainBase; ``.transform(settings, window)``
        # produces the matching DomainBase child in the target domain.
        #
        # TODO (2026-07-07, backend auto-conversion follow-up): transform()
        # now auto-transfers a CPU-loaded signal onto the target settings'
        # backend (DomainBase._coerce_transform_backend), which fixes the
        # GPU-settings-vs-CPU-data mismatch on this pour() path. Two related
        # boundaries still need an audit pass:
        #   1. the equality short-circuit below can hand back the
        #      CPU-resident td_signal WITHOUT any transfer when the target
        #      settings compare equal but live on another backend — if a
        #      NumPy/CuPy mix fires just downstream of pour(), apply
        #      ``data_signal = data_signal.with_backend(settings.backend)``;
        #   2. other CPU-side ingest points (noise-knot arrays into
        #      XYZSensitivityBackend.set_sensitivity_matrix, hand-built
        #      window arrays in settings files) are not auto-converted —
        #      DomainBase.with_backend / xp.asarray at the boundary is the
        #      pattern to use, one transfer per array, when they fire.
        if self.td_signal.settings == settings:
            data_signal = self.td_signal
        else:
            # TODO: fix this to be chopped based on Tobs originally
            # Target TD length for the chop: FDSettings.N counts rfft BINS,
            # not TD samples -- derive the TD length from df there. Other
            # domains carry the TD sample count on ``settings.N``.
            if isinstance(settings, FDSettings):
                target_td_n = round(1 / (settings.df * self.td_signal.dt))
            else:
                target_td_n = settings.N
            if self.td_signal.N != target_td_n:
                assert target_td_n < self.td_signal.N
                _td_sig = TDSignal(
                    self.td_signal[:, :target_td_n],
                    TDSettings(target_td_n, self.td_signal.dt),
                )
            else:
                _td_sig = self.td_signal
            data_signal = _td_sig.transform(settings, window=window)

        if return_orbits:
            return data_signal, self.orbits
        return data_signal


class L1ProcessingStep(L1DataLoader, BaseProcessingStep):
    """Load and condition a mojito L1 brick cache (the ``data_mode="mojito"`` processor).

    Construction does the I/O: :meth:`L1DataLoader.load_data`, then the
    optional ``Tobs`` window, then :class:`BaseProcessingStep` on the result.
    The stock fits' default processor; the production runs use its subclass
    :class:`~lisatools.globalfit.stock.erebor.injections.L1ProcessingStepWithSyntheticNoise`.

    Args:
        L1_folder: Mojito brick-cache root (see :class:`L1DataLoader`).
        source_types: Types to load (see :class:`L1DataLoader`).
        source_ids: ``{type: [ids]}`` (see :class:`L1DataLoader`). Default
            ``None``.
        orbits_class: Orbits class. Default :class:`~lisatools.detector.L1Orbits`.
        orbits_kwargs: Orbits keyword arguments. Default ``None``.
        verbose: Log progress. Default ``True``.
        store_individual_timeseries: Keep every stream (see
            :attr:`L1DataLoader.individual_timeseries`). Default ``False``.
        do_plots: Diagnostic plots of the conditioning steps. Default
            ``False``.
        Tobs: Window length [s] kept at load: the samples whose time from
            the first sample lies in ``[window_start_offset,
            window_start_offset + Tobs)``. ``None`` (default) keeps the whole
            file.
        window_start_offset: Start of the kept window [s from the first
            sample]. Default ``0.0``.
        allow_missing_bricks: See :class:`L1DataLoader`. Default ``False``.

    Attributes:
        orbits (Orbits): Orbits built from the first brick read.

    Example:
        .. code-block:: python

            from lisatools.detector import L1Orbits
            from lisatools.globalfit.preprocessing import L1ProcessingStep

            step = L1ProcessingStep(
                L1_folder="/data/mojito_light",
                source_types=["COMBINED", "MBHB"],
                source_ids={"MBHB": [0, 5]},
                orbits_class=L1Orbits,
                orbits_kwargs=dict(force_backend="cpu", frame="icrs"),
                Tobs=31104000.0,
            )
            times, data = step.process()  # no filters: builds step.td_signal
            catalogue_rows = step.catalogue["MBHB"]
    """

    def __init__(
        self,
        L1_folder: str,
        source_types: list,
        source_ids: dict = None,
        orbits_class: Orbits = L1Orbits,
        orbits_kwargs: dict = None,
        verbose: bool = True,
        store_individual_timeseries: bool = False,
        do_plots: bool = False,
        Tobs: float = None,
        window_start_offset: float = 0.0,
        allow_missing_bricks: bool = False,
    ):
        L1DataLoader.__init__(
            self,
            L1_folder=L1_folder,
            source_types=source_types,
            source_ids=source_ids,
            orbits_class=orbits_class,
            orbits_kwargs=orbits_kwargs,
            verbose=verbose,
            store_individual_timeseries=store_individual_timeseries,
            allow_missing_bricks=allow_missing_bricks,
        )

        times, fs, data_xyz, orbits = self.load_data()
        # TODO: add this to other processors?
        # ``window_start_offset`` (seconds from the file start) lets a caller
        # chop a sub-window that does NOT begin at the file start -- e.g. a
        # snippet centered on an MBH merger mid-mission. Default 0.0 keeps the
        # legacy behavior (first ``Tobs`` from the file start) byte-for-byte.
        if Tobs is not None:
            rel = times[:] - times[0]
            offset = float(window_start_offset)
            keep = (rel >= offset) & (rel < offset + Tobs)
            times = times[keep]
            data_xyz = data_xyz[:, keep]

        BaseProcessingStep.__init__(self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots)

        self.orbits = orbits

    def _open(self, file_path: str) -> MojitoL1File:
        """Open a brick as a ``mojito.MojitoL1File``."""
        return MojitoL1File(file_path)

    def process(self, *args, **kwargs):
        """Apply identical processing to the main data and all individual timeseries.

        With ``store_individual_timeseries`` each stored stream (except the
        ``PSD_*`` entries) is processed by a throwaway
        :class:`BaseProcessingStep` with the same arguments and replaced;
        ``"TIMES"`` and ``"COMBINED"`` (the processed data) are then added.

        Args:
            *args: Forwarded to :meth:`BaseProcessingStep.process`.
            **kwargs: Forwarded to :meth:`BaseProcessingStep.process`.

        Returns:
            tuple[np.ndarray, np.ndarray]: The processed times [s] and data
            ``(n_channels, n_times)``.
        """
        if hasattr(self, "individual_timeseries") and self.individual_timeseries is not None:
            for key, ts_data in self.individual_timeseries.items():
                if "PSD" in key:  # skip processing for PSD data
                    continue
                temp_processor = BaseProcessingStep(
                    times=self.times.copy(),
                    data=ts_data.copy(),
                    fs=self.fs,
                    verbose=False,  # Suppress logs for the individual processing
                    do_plots=False,
                )
                _, ts_data = temp_processor.process(*args, **kwargs)
                self.individual_timeseries[key] = ts_data

        processed_times, processed_data = super().process(*args, **kwargs)

        if hasattr(self, "individual_timeseries") and self.individual_timeseries is not None:
            self.individual_timeseries["TIMES"] = (
                processed_times  # store the processed times as well
            )
            self.individual_timeseries["COMBINED"] = processed_data

        return processed_times, processed_data


class SangriaProcessingStep(SangriaDataLoader, BaseProcessingStep):
    """Load and condition an LDC Sangria file (the ``data_mode="sangria"`` processor).

    Construction reads the file (:meth:`SangriaDataLoader.load_data`) and
    initializes :class:`BaseProcessingStep`. Used by ``erebor.all_sources``
    with ``data_mode="sangria"``; that mode keeps the engine's default
    high-pass + edge trim + ``Tobs`` cut.

    Args:
        data_input_path: Sangria HDF5 path.
        remove_from_data: Components to drop (see :class:`SangriaDataLoader`).
            Default ``None``.
        verbose: Log progress. Default ``True``.
        do_plots: Diagnostic plots of the conditioning steps. Default
            ``False``.

    Attributes:
        orbits (None): Sangria files carry no orbits.
        individual_timeseries (dict): Always empty.
    """

    def __init__(
        self,
        data_input_path: str,
        remove_from_data: Optional[list] = None,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        SangriaDataLoader.__init__(
            self, data_input_path=data_input_path, remove_from_data=remove_from_data
        )

        times, fs, data_xyz, _ = self.load_data()

        BaseProcessingStep.__init__(self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots)

        self.orbits = None  # no orbital information available in Sangria files
        self.individual_timeseries = (
            {}
        )  # to store individual timeseries for each source type and ID, if needed for debugging or further analysis


# ============================================================
# Synthetic-data / injection processing
# ============================================================
@dataclasses.dataclass
class SyntheticSourceSpec:
    """One source class to inject onto a synthetic data grid.

    Input of :func:`build_synthetic_source_streams` (this module's generic
    version; the stock ``data_mode="synthetic"`` path uses the different,
    same-named helpers of :mod:`lisatools.globalfit.stock.erebor.injections`).

    Args:
        label: Identifier for the returned per-source stream (e.g. ``"EMRI"``).
        wave_gen: Callable producing a per-source signal. When
            ``times_from_gen`` is ``False`` it returns a ``(nch, n)`` (or ``(n,)``)
            array on the data grid; when ``True`` it returns ``(times, channels)``.
        injections: ``(num_sources, ndim)`` array of parameter rows (a single 1-D
            row is broadcast to ``(1, ndim)``).
        param_transform: Optional per-row transform applied before calling
            ``wave_gen`` (e.g. an MBH sampling->waveform basis transform).
        times_from_gen: If ``True``, ``wave_gen`` returns ``(times, channels)`` and
            the channels are placed on the grid at their native ``times``.
        call_kwargs: Optional extra keyword arguments forwarded to ``wave_gen``
            (e.g. ``{"convert_to_ra_dec": False}`` for the EMRI wrapper).
    """

    label: str
    wave_gen: Callable
    injections: np.ndarray
    param_transform: Optional[Callable] = None
    times_from_gen: bool = False
    call_kwargs: Optional[dict] = None


def build_synthetic_source_streams(
    grid: TDSettings,
    *,
    sources: Sequence[SyntheticSourceSpec],
    nchannels: int,
    progress: bool = False,
) -> "dict[str, np.ndarray]":
    """Sum per-source TD signal streams onto ``grid``.

    Each :class:`SyntheticSourceSpec` is evaluated over its injection rows and
    the resulting signals are placed onto the shared ``grid`` via
    :func:`~lisatools.domains.place_td_signal_on_grid`. Returns a mapping
    ``{label: (nchannels, grid.N)}`` (one summed stream per source class).
    Not used by the stock fits, which call the same-named function of
    :mod:`lisatools.globalfit.stock.erebor.injections` (a different
    signature).

    Args:
        grid: Target time grid (``N``, ``dt``, ``t0``).
        sources: Source classes to evaluate.
        nchannels: Channels kept from each placed signal (the first
            ``nchannels`` rows).
        progress: Log one line per injected source. Default ``False``.

    Returns:
        dict[str, np.ndarray]: ``{spec.label: (nchannels, grid.N)}`` float64
        streams; a spec with no injection rows gives zeros.
    """
    N = grid.N
    out: "dict[str, np.ndarray]" = {}
    for spec in sources:
        inj = np.atleast_2d(spec.injections)
        stream = np.zeros((nchannels, N), dtype=np.float64)
        call_kwargs = spec.call_kwargs or {}
        if inj.shape[0] > 0 and inj.size > 0:
            for ii, row in enumerate(inj):
                if progress:
                    logger.info(
                        "Injecting %s signal %d of %d", spec.label, ii + 1, inj.shape[0]
                    )
                if spec.param_transform is not None:
                    params = spec.param_transform(np.asarray(row, dtype=float))
                else:
                    params = row
                if spec.times_from_gen:
                    times, channels = spec.wave_gen(*params, **call_kwargs)
                    placed = place_td_signal_on_grid(
                        np.atleast_2d(channels), grid, times=times
                    ).arr
                else:
                    sig = np.asarray(spec.wave_gen(*params, **call_kwargs))
                    placed = place_td_signal_on_grid(np.atleast_2d(sig), grid).arr
                stream += np.asarray(placed)[:nchannels]
        out[spec.label] = stream
    return out


class SyntheticSourceProcessingStep(BaseProcessingStep):
    """Generate a single source class's injection in-process.

    Builds the injected TD stream from a source-specific response wrapper and
    hands ``(times, data, fs)`` to :class:`BaseProcessingStep` — the same
    interface as :class:`SangriaProcessingStep`, with no external file needed.

    Supply the wrapper either directly (``wave_gen``) for runtime use, or via a
    ``wave_gen_factory`` for the global-fit settings path: the response wrappers
    hold unpicklable C++ objects, so when this step is registered as
    ``GeneralSettings.data_processor_class`` (whose ``processor_init_kwargs`` are
    deep-copied), pass a **module-level** ``wave_gen_factory`` function and let it
    build the wrapper lazily inside ``__init__``. The factory is called with
    the keywords ``Tobs``, ``dt``, ``t_start`` and ``tdi_chan``.

    Not used by the stock fits (their per-source synthetic steps live in
    :mod:`lisatools.globalfit.stock.erebor.injections`).

    Args:
        Tobs: Duration [s]; the grid has ``round(Tobs / dt)`` samples.
        dt: Sample spacing [s].
        t_start: Grid start time [s].
        wave_gen: Per-source signal callable (see
            :class:`SyntheticSourceSpec`). Default ``None``.
        wave_gen_factory: Builds ``wave_gen`` when it is ``None``. Default
            ``None``.
        injections: ``(num_sources, ndim)`` parameter rows.
        param_transform: Per-row transform applied before ``wave_gen``.
            Default ``None``.
        times_from_gen: ``wave_gen`` returns ``(times, channels)``. Default
            ``False``.
        call_kwargs: Extra keywords of ``wave_gen``. Default ``None``.
        tdi_chan: Channel label, recorded on :attr:`tdi_chan`. Default
            ``"XYZ"``.
        nchannels: Number of channels. Default ``3``.
        verbose: Log progress. Default ``True``.
        do_plots: Diagnostic plots. Default ``False``.

    Raises:
        ValueError: Neither ``wave_gen`` nor ``wave_gen_factory`` is given.

    Attributes:
        orbits (None): No orbits.
        tdi_chan (str): Channel label.
        injections (np.ndarray): ``(num_sources, ndim)`` injection rows.
    """

    def __init__(
        self,
        *,
        Tobs: float,
        dt: float,
        t_start: float,
        wave_gen: Optional[Callable] = None,
        wave_gen_factory: Optional[Callable] = None,
        injections: np.ndarray,
        param_transform: Optional[Callable] = None,
        times_from_gen: bool = False,
        call_kwargs: Optional[dict] = None,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        if wave_gen is None:
            if wave_gen_factory is None:
                raise ValueError(
                    "SyntheticSourceProcessingStep needs either wave_gen or "
                    "wave_gen_factory."
                )
            wave_gen = wave_gen_factory(
                Tobs=Tobs, dt=dt, t_start=t_start, tdi_chan=tdi_chan
            )
        target_N = int(round(Tobs / dt))
        grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")
        spec = SyntheticSourceSpec(
            label=tdi_chan,
            wave_gen=wave_gen,
            injections=injections,
            param_transform=param_transform,
            times_from_gen=times_from_gen,
            call_kwargs=call_kwargs,
        )
        streams = build_synthetic_source_streams(
            grid, sources=[spec], nchannels=nchannels, progress=verbose
        )
        combined = next(iter(streams.values()))

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        self.injections = np.atleast_2d(injections)


class SyntheticDataProcessor(BaseProcessingStep):
    """All-synthetic processor: injected sources + optional noise + foreground.

    Sums the per-class source streams (:func:`build_synthetic_source_streams`)
    onto a data grid, optionally adding correlated instrument noise
    (:func:`~lisatools.sensitivity.generate_correlated_instrument_noise_td`) and
    a galactic-foreground TD realization
    (:func:`~lisatools.sensitivity.generate_foreground_td`), all placed on the
    same grid. An optional ``base_data`` (e.g. a Sangria sky-only slice or a
    mojito L1 stream) is added underneath. Exposes ``catalogue`` (if supplied)
    so downstream factories can read leaf counts uniformly.

    This generic version is not what the stock fits run: their
    ``data_mode="synthetic"`` path uses the same-named
    :class:`lisatools.globalfit.stock.erebor.injections.SyntheticDataProcessor`
    (different arguments).

    Args:
        Tobs: Duration [s]; the grid has ``N = round(Tobs / dt)`` samples.
        dt: Sample spacing [s].
        t_start: Grid start time [s].
        source_specs: :class:`SyntheticSourceSpec` list to inject.
        nchannels: Number of TDI channels.
        add_instrument_noise: ``None`` or kwargs for
            :func:`~lisatools.sensitivity.generate_correlated_instrument_noise_td`
            (``Soms_d``, ``Sa_a``, ``tdi_generation``, ``seed``, ...).
        add_foreground: ``None`` or kwargs for
            :func:`~lisatools.sensitivity.generate_foreground_td`
            (``foreground_params``, ``tdi_generation``, ``seed``, ``envelope`` ...;
            ``Tobs`` defaults to this processor's ``Tobs``).
        base_data: Optional ``(nch, n)`` TD stream added underneath the injections.
        catalogue: Optional catalogue dict stored on ``self.catalogue``.
        tdi_chan: TDI channel label recorded on ``self.tdi_chan``.
        verbose: Log progress. Default ``True``.
        do_plots: Diagnostic plots. Default ``False``.

    Attributes:
        orbits (None): No orbits.
        tdi_chan (str): Channel label.
        catalogue (dict): Only when ``catalogue`` was given.
    """

    def __init__(
        self,
        *,
        Tobs: float,
        dt: float,
        t_start: float,
        source_specs: Sequence[SyntheticSourceSpec],
        nchannels: int = 3,
        add_instrument_noise: Optional[dict] = None,
        add_foreground: Optional[dict] = None,
        base_data: Optional[np.ndarray] = None,
        catalogue: Optional[dict] = None,
        tdi_chan: str = "XYZ",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        # Local import: sensitivity is a heavier module and only needed when
        # synthetic noise / foreground are requested (no import cycle —
        # sensitivity does not import globalfit).
        from ..sensitivity import (
            generate_correlated_instrument_noise_td,
            generate_foreground_td,
        )

        target_N = int(round(Tobs / dt))
        grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")

        combined = np.zeros((nchannels, target_N), dtype=np.float64)
        if base_data is not None:
            combined = combined + np.asarray(
                place_td_signal_on_grid(np.atleast_2d(base_data), grid).arr
            )[:nchannels]

        streams = build_synthetic_source_streams(
            grid, sources=source_specs, nchannels=nchannels, progress=verbose
        )
        for stream in streams.values():
            combined = combined + stream

        if add_instrument_noise is not None:
            noise_td = generate_correlated_instrument_noise_td(
                target_N, dt, **add_instrument_noise
            )[:nchannels]
            combined = combined + noise_td

        if add_foreground is not None:
            fg_kwargs = dict(add_foreground)
            fg_kwargs.setdefault("Tobs", Tobs)
            fg_td = generate_foreground_td(target_N, dt, **fg_kwargs)[:nchannels]
            combined = combined + fg_td

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        if catalogue is not None:
            self.catalogue = catalogue
