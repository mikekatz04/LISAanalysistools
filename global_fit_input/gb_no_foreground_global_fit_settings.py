"""Legacy stub — this run variant now lives in the installed stock package.

The GB-only / no-foreground global fit is ``lisatools.globalfit.stock.erebor``
option ``"gb_no_fg"``::

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(nwalkers=4)      # adjust knobs / recipe / branches
    fit.build()                            # heavy stage, on command
    fit.run()                              # or scripts/run_global.py --stock gb_no_fg

This stub only keeps the old file path working (``run_global.py -sfp``, the
gb_chunked_het smoke script): it constructs the stock variant with the
default (env-honoring) configuration. Do not add settings here — adjust the
stock class (see the sprint rule: no new settings files).
"""

from lisatools.globalfit.stock.erebor import GBNoForegroundGlobalFit
from lisatools.globalfit.stock.erebor.common import resolve_compute
from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (  # noqa: F401
    GBNoFgGBSettings,
    GBNoFgGeneralSettings,
    setup_recipe,
)

_fit = GBNoForegroundGlobalFit.from_env()

# --- legacy module surface (consumed by the smoke script / older tooling) ---
NF, NT, WAVELET_DURATION, TOBS = _fit.wdm_grid
DT = _fit.general.dt
TOBS_TARGET = _fit.general.tobs_target
MIN_FREQ = _fit.general.min_freq
MAX_FREQ = _fit.general.max_freq
NWALKERS = _fit.general.nwalkers
NTEMPS = _fit.general.ntemps
FIXED_PSD_PARAMS = list(_fit.general.fixed_psd_params)
MOJITO_DATA_PATH = _fit.general.mojito_data_path
LDC_SOURCE_FILE = _fit.general.ldc_source_file
FILE_STORE_DIR = _fit.general.file_store_dir
BASE_FILE_NAME = _fit.general.base_file_name
GB_DEBUG = _fit.debug
GB_MODE = _fit.gb.mode

_gpus, _backend = resolve_compute(
    _fit.general.use_gpu, _fit.general.gpu_backend, _fit.general.gpus
)
GPU_BACKEND = _backend if _gpus is not None else "cpu"
gpu_available = _gpus is not None


def _pre_init_cuda_gpus():
    # Literal ``gpus`` assignment kept for run_global.py's _pre_init_cuda
    # AST scan (it sets the CUDA device before any cupy import).
    gpus = [0]
    return gpus


def get_global_fit_settings(copy_settings_file=False):
    """Build the default stock ``gb_no_fg`` fit (env knobs honored)."""
    return GBNoForegroundGlobalFit.from_env().build()


if __name__ == "__main__":
    print(_fit.describe())
