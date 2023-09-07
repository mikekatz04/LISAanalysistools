import numpy as np
import matplotlib.pyplot as plt


def make_current_plot(current_info, save_file=None, add_mbhs=False, add_gbs=False, **kwargs):

    generated_info_0 = current_info.get_data_psd(only_max_ll=True, include_gbs=False, include_mbhs=False, **kwargs)
    generated_info_1 = current_info.get_data_psd(only_max_ll=True, **kwargs)

    plt.loglog(current_info.general_info["fd"], 2 * current_info.general_info["df"] * np.abs(generated_info_0["data"][0]) ** 2)
    
    if add_gbs:
        generated_info_gb = current_info.get_data_psd(only_max_ll=True, include_gbs=False, include_mbhs=True, **kwargs)
        plt.loglog(current_info.general_info["fd"], 2 * current_info.general_info["df"] * np.abs(generated_info_1["data"][0] - generated_info_gb["data"][0]) ** 2)

    if add_mbhs:
        generated_info_mbh = current_info.get_data_psd(only_max_ll=True, include_gbs=True, include_mbhs=False, **kwargs)
        plt.loglog(current_info.general_info["fd"], 2 * current_info.general_info["df"] * np.abs(generated_info_1["data"][0] - generated_info_mbh["data"][0]) ** 2)

    plt.loglog(current_info.general_info["fd"], 2 * current_info.general_info["df"] * np.abs(generated_info_1["data"][0]) ** 2)

    plt.loglog(current_info.general_info["fd"], generated_info_0["psd"][0])

    if save_file is not None:
        plt.savefig(save_file)

