import logging
import sys

# from global_fit_input.global_fit_settings import get_global_fit_settings


def init_logger(filename=None, level=logging.DEBUG, name="GlobalFit"):
    """Initialize a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) < 2:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - " "%(levelname)s - %(message)s"
        )
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
