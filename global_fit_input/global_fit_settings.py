"""Legacy stub — this run variant now lives in the installed stock package.

The all-branch global fit is ``lisatools.globalfit.stock.erebor`` option
``"all_sources"``::

    from lisatools.globalfit.stock import erebor

    fit = erebor.all_sources(nwalkers=4)   # adjust knobs / branches / recipe
    fit.build()                            # heavy stage, on command
    fit.run()                              # or scripts/run_global.py --stock all_sources

This stub keeps the old file path working (``run_global.py -sfp``) and
re-exports the shared wrapper / injection names the sibling settings files
import from here. Do not add settings here — adjust the stock class (see
the sprint rule: no new settings files).

NOTE: the legacy version of this file was broken on two counts
(``data_processor=`` kwarg and pre-ICRS ``lam_lims``/``beta_lims`` GB
kwargs); the stock variant fixes both.
"""

from lisatools.globalfit.stock.erebor import AllSourcesGlobalFit
from lisatools.globalfit.stock.erebor.gb import GBSetup
from lisatools.globalfit.stock.erebor.noise import GalForSetup, PSDSetup

# --- shared wrapper / injection surface (imported by sibling files) ---------
from lisatools.globalfit.stock.erebor.wrappers import (  # noqa: F401
    EMRI_INSPIRAL_KWARGS,
    EMRI_MODE_SELECTOR_KWARGS,
    EMRI_SUM_KWARGS,
    EMRIWaveWrap,
    MBH_TDIONFLY_COARSE_SCALE,
    MBH_TDIONFLY_HIGHER_MODES,
    MBH_TDIONFLY_TOL,
    MBHTDIonFlyWaveWrap,
    SOBBHTDIonFlyWaveWrap,
    SOBBHWaveWrap,
    get_emri_response_wrapper,
    get_mbh_tdionfly_gen,
    get_sobbh_response_wrapper,
    get_sobbh_tdionfly_gen,
)
from lisatools.globalfit.stock.erebor.injections import (  # noqa: F401
    GB_INJECTION_PARAMS,
    INJECTION_PARAMS_FULL_BASIS,
    SAMPLE_FILL_INDICES,
    SOBBH_INJECTION_PARAMS_FULL_BASIS,
    SyntheticCombinedProcessingStep,
    SyntheticEMRIProcessingStep,
    SyntheticGBProcessingStep,
    SyntheticSOBBHProcessingStep,
    emri_full_to_sampling,
    sobbh_full_to_sampling,
)
from lisatools.globalfit.stock.erebor.variants.all_sources import (  # noqa: F401
    _build_residual_pe_move,
    _legacy_wave_wrap,
    setup_recipe,
)

_fit = AllSourcesGlobalFit.from_env()

# --- legacy module surface ---------------------------------------------------
NF, NT, WAVELET_DURATION, TOBS = _fit.wdm_grid
DT = _fit.general.dt
T_START = _fit.general.t_start
MIN_FREQ = _fit.general.min_freq
MAX_FREQ = _fit.general.max_freq
GPU_BACKEND = "cpu" if _fit.general.gpus is None else _fit.general.gpu_backend


def _pre_init_cuda_gpus():
    # Literal ``gpus`` assignment kept for run_global.py's _pre_init_cuda
    # AST scan (it sets the CUDA device before any cupy import).
    gpus = [0]
    return gpus


# Legacy per-branch factories (consumed by sibling settings files).
def get_gb_erebor_settings(general_set) -> GBSetup:
    return GBSetup(_fit.prepare_branch_settings("gb", general_set))


def get_psd_erebor_settings(general_set) -> PSDSetup:
    return PSDSetup(_fit.prepare_branch_settings("psd", general_set))


def get_galfor_erebor_settings(general_set) -> GalForSetup:
    return GalForSetup(_fit.prepare_branch_settings("galfor", general_set))


# Legacy move builders (consumed by sibling settings files).
def _build_emri_move(curr, acs, priors, state):
    wave_gen = _legacy_wave_wrap("emri", curr, acs)
    return _build_residual_pe_move("emri", curr, acs, priors, state, wave_gen)


def _build_sobbh_move(curr, acs, priors, state):
    wave_gen = _legacy_wave_wrap("sobbh", curr, acs)
    return _build_residual_pe_move("sobbh", curr, acs, priors, state, wave_gen)


def get_global_fit_settings(copy_settings_file=False):
    """Build the default stock ``all_sources`` fit (env knobs honored)."""
    return AllSourcesGlobalFit.from_env().build()


if __name__ == "__main__":
    print(_fit.describe())
