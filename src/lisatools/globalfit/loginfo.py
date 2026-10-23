"""Logging setup and the run-settings dump for the global fit.

Three entry points, all called at run start:

* :func:`init_logger`: a named logger with its own file handler and a
  managed console handler (stdout when verbose, stderr WARNING otherwise).
  Called by :class:`~lisatools.globalfit.engine.GeneralSetup`, by
  :class:`~lisatools.globalfit.run.GlobalFit` (one ``global_fit.log`` per
  rank, ``global_fit.rank<N>.log`` off the head) and by every per-branch
  stock ``Setup`` (with ``shared_console=False``).
* :func:`setup_root_file_handler`: one combined file handler on the
  ``lisatools`` logger (``globalfit_run.log`` on the head,
  ``globalfit_run.rank<N>.log`` on the other ranks), attached by
  :class:`~lisatools.globalfit.run.GlobalFit`.
* :func:`dump_settings`: writes ``run_settings.log`` into the artifacts
  directory (head rank only). The monitor reads this file back (for
  example the ``mojito_data_path`` the job actually used).

The console verbosity follows the run-level ``verbose`` setting
(``VERBOSE`` in the launchers).
"""

import logging
import os
import sys

import numpy as np

# from global_fit_input.global_fit_settings import get_global_fit_settings


def init_logger(
    filename=None,
    level=logging.DEBUG,
    name="GlobalFit",
    log_dir=None,
    console=False,
    shared_console=True,
):
    """Initialize a named logger with a file handler and a managed console handler.

    Sets ``name``'s level to ``level`` and ``propagate=False``, adds a
    ``FileHandler`` for ``filename`` (only once per absolute path, so a
    repeated call does not duplicate lines), and installs the console handler
    through :func:`_set_console_handler`. The shared ``lisatools`` logger's
    level is set to ``level`` on EVERY call; with ``shared_console`` it also
    gets ``propagate=False`` and the managed console handler.

    Args:
        filename (str, optional): Log file name. If provided with log_dir, the
            file is placed inside log_dir. ``None`` adds no file handler.
        level (int): Logging level (default: ``logging.DEBUG``).
        name (str): Logger name (default: ``"GlobalFit"``).
        log_dir (str, optional): Directory to place the log file in. When
            provided alongside filename, the log file is written to
            ``os.path.join(log_dir, filename)`` instead of the CWD.
        console (bool): When True, also stream records to stdout at ``level`` (the
            run-level ``verbose`` knob). Default False: log statements are
            still captured in the log files, but the console only sees
            warnings and errors.
        shared_console (bool): When True (default) this call also (re)configures the
            console handler of the shared ``lisatools`` logger -- the
            run-level owners (``GeneralSetup``, ``GlobalFit``) do that with
            the run's ``verbose``. Per-branch Setups pass ``False``: they own
            only their named logger. (Before 2026-10-13 every branch Setup
            re-ran this with ``console=False`` and so DOWNGRADED the shared
            console to WARNING part-way through ``fit.build()``, silencing
            every build-time INFO line from ``lisatools.*`` after the first
            branch Setup -- e.g. the MBH inner-move stack report.)

    Returns:
        logging.Logger: The configured logger ``name``.

    Example:
        >>> import logging
        >>> from lisatools.globalfit.loginfo import init_logger
        >>> log = init_logger(level=logging.INFO, name="my_fit", shared_console=False)
        >>> log.name
        'my_fit'
    """
    lisatools_logger = logging.getLogger("lisatools")
    lisatools_logger.setLevel(level)
    if shared_console:
        # Propagated lisatools.* records: a named stdout handler when verbose, a
        # stderr WARNING handler when quiet (installed/removed per call, so the
        # knob can flip within one process). propagate=False because the run owns
        # the lisatools console output: with a root handler also installed (e.g. a
        # run script's logging.basicConfig), propagation prints every line TWICE
        # when verbose and leaks the full DEBUG stream when quiet (root handlers
        # do not re-check ancestor logger levels).
        lisatools_logger.propagate = False
        _set_console_handler(lisatools_logger, console, level, quiet_stream=sys.stderr)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = _formatter()
    if filename:
        if log_dir:
            filename = os.path.join(log_dir, filename)
        filename = os.path.abspath(filename)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == filename
            for h in logger.handlers
        ):
            rfhandler = logging.FileHandler(filename)
            rfhandler.setFormatter(formatter)
            logger.addHandler(rfhandler)
    # propagate=False: this logger needs its own console handler so that
    # warnings/errors still surface when quiet.
    _set_console_handler(logger, console, level, quiet_stream=sys.stderr)
    return logger


#: Name tagging the (single, swappable) console handler this module manages.
_CONSOLE_HANDLER_NAME = "lisatools-console"


def _formatter():
    """The shared ``"time - logger - level - message"`` record format."""
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _set_console_handler(logger, console, level, quiet_stream):
    """Install/replace/remove ``logger``'s managed console handler.

    ``console=True``: a stdout handler at ``level``. ``console=False``: a
    stderr WARNING handler when ``quiet_stream`` is given, else no console
    handler at all. Re-running with a different ``console`` swaps the handler
    (the verbose knob can differ between fits in one process/notebook).

    Args:
        logger (logging.Logger): Logger whose handler named
            :data:`_CONSOLE_HANDLER_NAME` is replaced.
        console (bool): Stream to stdout at ``level`` when True (and
            ``level`` is truthy).
        level (int): Level of the stdout handler.
        quiet_stream (io.TextIOBase, optional): Stream for the WARNING-level
            handler used when ``console`` is False; ``None`` removes the
            console handler entirely.
    """
    for h in list(logger.handlers):
        if h.get_name() == _CONSOLE_HANDLER_NAME:
            logger.removeHandler(h)
    if console and level:
        shandler = logging.StreamHandler(sys.stdout)
        shandler.setLevel(level)
    elif quiet_stream is not None:
        shandler = logging.StreamHandler(quiet_stream)
        shandler.setLevel(logging.WARNING)
    else:
        return
    shandler.set_name(_CONSOLE_HANDLER_NAME)
    shandler.setFormatter(_formatter())
    logger.addHandler(shandler)


def setup_root_file_handler(log_dir, level=logging.DEBUG, filename="globalfit_run.log"):
    """Attach a FileHandler to the 'lisatools' root logger.

    Since all lisatools sub-loggers propagate to 'lisatools', this single
    handler captures all log output into one combined file inside log_dir.
    A second call for the same file path is a no-op (no duplicate handler).

    Args:
        log_dir (str): Directory where the combined log file will be written.
        level (int): Logging level for the file handler (default:
            ``logging.DEBUG``).
        filename (str): Log file name (default: ``"globalfit_run.log"``); a
            non-head rank in a multi-rank run gets its own per-rank file name
            (``globalfit_run.rank<N>.log``, chosen by
            :class:`~lisatools.globalfit.run.GlobalFit`).
    """
    root_logger = logging.getLogger("lisatools")
    filepath = os.path.join(log_dir, filename)
    # Avoid duplicate handlers if called more than once
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(filepath):
            return
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)


def dump_settings(settings, artifacts_dir):
    """Write a human-readable settings summary to ``<artifacts_dir>/run_settings.log``.

    Three sections: GENERAL SETTINGS (every dataclass field of
    ``settings.general_info``), SOURCE SETTINGS (every field of each entry of
    ``settings.source_info``) and RANK INFO (``settings.rank_info``). Simple
    values (scalars, strings, ``None``, lists/tuples under 50 entries, dicts
    under 20 keys, arrays under 50 elements) are written with ``repr``;
    anything larger is written as ``<TypeName>``. The file is overwritten.
    Called once on the head rank by :class:`~lisatools.globalfit.run.GlobalFit`;
    the monitor parses values (e.g. ``mojito_data_path``) back out of it.

    Args:
        settings (GlobalFitSettings): The run's
            :class:`~lisatools.globalfit.engine.GlobalFitSettings` (the
            ``settings_dict`` of the built fit), or any object with
            ``general_info``, ``source_info`` (a dict) and ``rank_info``.
        artifacts_dir (str): Directory where ``run_settings.log`` will be
            written (must exist).

    Example:
        .. code-block:: python

            from lisatools.globalfit.loginfo import dump_settings

            curr = fit.build()
            dump_settings(curr.settings_dict, curr.general_info.artifacts_file_dir)
    """
    import dataclasses

    filepath = os.path.join(artifacts_dir, "run_settings.log")

    def _is_simple(value):
        """Return True if the value should be printed in full."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        if isinstance(value, (list, tuple)):
            return len(value) < 50
        if isinstance(value, dict):
            return len(value) < 20
        if isinstance(value, np.ndarray):
            return value.size < 50
        return False

    def _format_value(value):
        """``repr(value)`` when simple, else ``<TypeName>``."""
        if _is_simple(value):
            return repr(value)
        return f"<{type(value).__name__}>"

    lines = []
    lines.append("=" * 72)
    lines.append("GLOBAL FIT RUN SETTINGS")
    lines.append("=" * 72)

    # General info
    general = settings.general_info
    lines.append("")
    lines.append("-" * 40)
    lines.append("GENERAL SETTINGS")
    lines.append("-" * 40)
    if dataclasses.is_dataclass(general):
        for field in dataclasses.fields(general):
            value = getattr(general, field.name, None)
            lines.append(f"  {field.name}: {_format_value(value)}")
    else:
        lines.append(f"  {_format_value(general)}")

    # Source info
    lines.append("")
    lines.append("-" * 40)
    lines.append("SOURCE SETTINGS")
    lines.append("-" * 40)
    if hasattr(settings, "source_info") and isinstance(settings.source_info, dict):
        for source_name, source_setup in settings.source_info.items():
            lines.append(f"")
            lines.append(f"  [{source_name}]")
            if dataclasses.is_dataclass(source_setup):
                for field in dataclasses.fields(source_setup):
                    value = getattr(source_setup, field.name, None)
                    lines.append(f"    {field.name}: {_format_value(value)}")
            else:
                lines.append(f"    {_format_value(source_setup)}")

    # Rank info
    lines.append("")
    lines.append("-" * 40)
    lines.append("RANK INFO")
    lines.append("-" * 40)
    rank_info = settings.rank_info
    if dataclasses.is_dataclass(rank_info):
        for field in dataclasses.fields(rank_info):
            value = getattr(rank_info, field.name, None)
            lines.append(f"  {field.name}: {_format_value(value)}")
    else:
        lines.append(f"  {_format_value(rank_info)}")

    lines.append("")
    lines.append("=" * 72)

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
