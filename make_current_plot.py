
from lisatools.globalfit.run import CurrentInfoGlobalFit
from global_fit_input.global_fit_settings import get_global_fit_settings
from lisatools.globalfit.plot import RunResultsProduction

if __name__ == "__main__":
    settings = get_global_fit_settings()
    curr = CurrentInfoGlobalFit(settings)
    res = RunResultsProduction(None, None, add_mbhs=False, add_gbs=False)
    res.build_plots(curr)