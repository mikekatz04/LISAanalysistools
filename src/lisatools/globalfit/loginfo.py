import logging
import os
import sys

import numpy as np

# from global_fit_input.global_fit_settings import get_global_fit_settings


def init_logger(filename=None, level=logging.DEBUG, name="GlobalFit", log_dir=None, console=False):
    """Initialize a logger.

    Args:
        filename: Log file name. If provided with log_dir, the file is placed inside log_dir.
        level: Logging level (default: DEBUG).
        name: Logger name (default: "GlobalFit").
        log_dir: Directory to place the log file in. When provided alongside filename,
            the log file is written to os.path.join(log_dir, filename) instead of CWD.
        console: When True, also stream records to stdout at ``level`` (the
            run-level ``verbose`` knob). Default False: log statements are
            still captured in the log files, but the console only sees
            warnings and errors.
    """
    lisatools_logger = logging.getLogger("lisatools")
    lisatools_logger.setLevel(level)
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
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _set_console_handler(logger, console, level, quiet_stream):
    """Install/replace/remove ``logger``'s managed console handler.

    ``console=True``: a stdout handler at ``level``. ``console=False``: a
    stderr WARNING handler when ``quiet_stream`` is given, else no console
    handler at all. Re-running with a different ``console`` swaps the handler
    (the verbose knob can differ between fits in one process/notebook).
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


def setup_root_file_handler(log_dir, level=logging.DEBUG):
    """Attach a FileHandler to the 'lisatools' root logger.

    Since all lisatools sub-loggers propagate to 'lisatools', this single
    handler captures all log output into one combined file inside log_dir.

    Args:
        log_dir: Directory where the combined log file will be written.
        level: Logging level for the file handler (default: DEBUG).
    """
    root_logger = logging.getLogger("lisatools")
    filepath = os.path.join(log_dir, "globalfit_run.log")
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
    """Write a human-readable settings summary to a text file.

    Args:
        settings: GlobalFitSettings object (the original settings_dict).
        artifacts_dir: Directory where run_settings.log will be written.
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
