"""Legacy stub — this run variant now lives in the installed stock package.

The full-year multi-leaf MBH+EMRI+SOBBH fit is
``lisatools.globalfit.stock.erebor`` option ``"full_year_combined"``::

    from lisatools.globalfit.stock import erebor

    fit = erebor.full_year_combined()
    fit.general.mojito_source_ids = {"MBHB": [0], "EMRI": [1], "SOBHB": []}
    fit.mbh.use_tdionfly = True        # waveform paths are per-branch knobs
    fit.build()
    fit.run()   # or scripts/run_global.py --stock full_year_combined

This stub keeps the old file path working (``run_global.py -sfp``, the
mojito signal-gen test, the scripts/sobbh + scripts/mbh debug harnesses):
it constructs the stock variant with the default (env-honoring)
configuration. Do not add settings here — adjust the stock class (see the
sprint rule: no new settings files).
"""

from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME  # noqa: F401
from lisatools.globalfit.stock.erebor import FullYearCombinedGlobalFit
from lisatools.globalfit.stock.erebor.variants.full_year_combined import (  # noqa: F401
    FullYearGeneralSettings,
    FullYearMBHSettings,
    FullYearSOBBHSettings,
    FullYearSignalGen,
    _get_emri_wave_wrap as _stock_get_emri_wave_wrap,
    _get_mbh_tdionfly_wave_wrap as _stock_get_mbh_tdionfly_wave_wrap,
    _get_sobbh_wave_wrap as _stock_get_sobbh_wave_wrap,
    _signal_cfg,
    setup_recipe,
)

_fit = FullYearCombinedGlobalFit.from_env()
_CFG = _signal_cfg(_fit.general, _fit)

# --- legacy module surface (consumed by tests / debug scripts) --------------
NF, NT, WAVELET_DURATION, TOBS = _fit.wdm_grid
DT = _fit.general.dt
TOBS_TARGET = _fit.general.tobs_target
MIN_FREQ = _fit.general.min_freq
MAX_FREQ = _fit.general.max_freq
NCHANNELS = _fit.general.nchannels
TDI_CHAN = _fit.general.tdi_chan
MOJITO_DATA_PATH = _fit.general.mojito_data_path
MOJITO_SOURCE_IDS = _fit.general.mojito_source_ids
DATA_PROCESSOR = _fit.general.data_mode
CHOP_WINDOW = _fit.general.chop_window
ACTIVE_SOURCE = _fit.general.active_source
USE_TDIONFLY = _CFG["sobbh_use_tdionfly"]
MBH_WAVEFORM_T0 = _fit.general.mbh_waveform_t0
N_MBH_INJECTIONS = _fit.general.n_injections["MBHB"]
N_EMRI_INJECTIONS = _fit.general.n_injections["EMRI"]
N_SOBBH_INJECTIONS = _fit.general.n_injections["SOBHB"]
GPU_BACKEND = "cpu" if _fit.general.gpus is None else _fit.general.gpu_backend


def _pre_init_cuda_gpus():
    # Literal ``gpus`` assignment kept for run_global.py's _pre_init_cuda
    # AST scan (it sets the CUDA device before any cupy import).
    gpus = [0]
    return gpus


# Legacy wave-wrap getters (single-argument signatures; consumed by
# tests/test_global_fit_signal_gen_mojito.py's FD cross-checks).
def _get_emri_wave_wrap(general_info, nchannels: int = NCHANNELS):
    return _stock_get_emri_wave_wrap(general_info, _CFG)


def _get_sobbh_wave_wrap(general_info, nchannels: int = NCHANNELS):
    return _stock_get_sobbh_wave_wrap(general_info, _CFG)


def _get_mbh_tdionfly_wave_wrap(general_info, nchannels: int = NCHANNELS):
    return _stock_get_mbh_tdionfly_wave_wrap(general_info, _CFG)


def get_global_fit_settings(copy_settings_file: bool = False):
    """Build the default stock ``full_year_combined`` fit (env knobs honored)."""
    return FullYearCombinedGlobalFit.from_env().build()


if __name__ == "__main__":
    print(_fit.describe())
