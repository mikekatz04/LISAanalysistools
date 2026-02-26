import logging
import sys

# from global_fit_input.global_fit_settings import get_global_fit_settings


def init_logger(filename=None, level=logging.DEBUG, name="GlobalFit"):
    """Initialize a logger."""
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("lisatools").setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if len(logger.handlers) < 2:
        formatter = logging.Formatter("%(asctime)s - %(name)s - " "%(levelname)s - %(message)s")
        if filename:
            rfhandler = logging.FileHandler(filename)
            logger.addHandler(rfhandler)
            rfhandler.setFormatter(formatter)
        if level:
            shandler = logging.StreamHandler(sys.stdout)
            shandler.setLevel(level)
            shandler.setFormatter(formatter)
            logger.addHandler(shandler)
    return logger
