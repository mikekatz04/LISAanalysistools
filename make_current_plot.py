
from lisatools.globalfit.run import CurrentInfoGlobalFit
from global_fit_input.global_fit_settings import get_global_fit_settings
from lisatools.globalfit.plot import make_current_plot

if __name__ == "__main__":
    settings = get_global_fit_settings()
    curr = CurrentInfoGlobalFit(settings)
    make_current_plot(curr, save_file="check1.png", add_mbhs=True, add_gbs=False)