import numpy as np
import matplotlib.pyplot as plt
import corner

from bbhx.utils.transform import LISA_to_SSB

def produce_mbh_plots(mbh_reader, num_leaves, discard=0, save_file=None, fig=None):

    plt.close()
    fig2, ax2 = plt.subplots(1, 1)
    samples_list = []
    for i in range(num_leaves):
        mbh_samp = mbh_reader.get_chain(discard=discard)["mbh"][:, 0, :, i].reshape(-1, 11)
        if i == 0: 
            print("num mbh samples:", mbh_samp.shape[0])
        # best_fit = np.abs(mbh_samp[:, -1].mean() - mbhbs_in[:, -1]).argmin()
        
        # truth = mbhbs_in[best_fit].copy()
        mbh_samp[:, 0] = np.exp(mbh_samp[:, 0])
        # truth[4] /= 1e3
        # truth[6] = np.cos(truth[6])
        # truth[8] = np.sin(truth[8])
        # truth[5] = truth[5] % (2 * np.pi)
        # truth[7] = truth[7] % (2 * np.pi)
        # truth[9] = truth[9] % (np.pi)

        inds_keep = np.array([0, 1, 2, 3, 4, 7, 8, 10])
        mbh_samp_in = mbh_samp[:, inds_keep]
        # truths_in = truth[inds_keep]
        labels = [r"$M$", r"$q$", r"$a_1$", r"$a_2$", r"$d_L$", r"$\phi_0$", r"$\cos{\iota}$", r"$\lambda$", r"$\sin{\beta}$", r"$\psi$", r"$t_c$"]
        labels2 = [labels[w] for w in inds_keep]
        fig = corner.corner(mbh_samp_in, label_kwargs=dict(fontsize=16), plot_datapoints=False, smooth=0.6, levels=1 - np.exp(-0.5 * np.array([1, 2, 3])**2), labels=labels2)  # , truths=truths_in
        
        # tc mT plot
        samples_list.append([mbh_samp[:, 10], mbh_samp[:, 0]])
        if save_file is not None:
            save_file_tmp = save_file[:-4] + f"_mbh_posterior_{i+1}.png"
            fig.savefig(save_file_tmp)
            
        del fig
        plt.close()

    plt.close()
    for i, tmp in enumerate(samples_list):
        ax2.scatter(tmp[0], tmp[1], color=f"C{i % 10}", s=8)
        
    save_file_tmp = save_file[:-4] + "_mT_vs_tc.png"
    fig2.savefig(save_file_tmp)
    plt.close()

def produce_sky_plot(current_info, save_file=None, fig=None):
    
    ll_gb_ind_max = current_info.gb_info["cc_ll"].argmax()
    gb_lam, gb_sinbeta = current_info.gb_info["cc_params"][ll_gb_ind_max, :, np.array([6, 7])][:, current_info.gb_info["cc_inds"][ll_gb_ind_max, :]]

    gb_lam_rad = gb_lam  - np.pi #  * 180. / np.pi - 180.0
    gb_beta_rad = np.arcsin(gb_sinbeta) #  * 180 / np.pi

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.scatter(gb_lam_rad, gb_beta_rad, s=2, alpha=0.3)

    ll_mbh_ind_max = current_info.mbh_info["cc_ll"].argmax()
    
    mbh_lam_L, mbh_sinbeta_L, mbh_psi_L, mbh_tc_L = current_info.mbh_info["cc_params"][ll_mbh_ind_max, :, np.array([7, 8, 9, 10])]

    mbh_tc_SSB, mbh_lam_SSB, mbh_beta_SSB, mbh_psi_SSB = LISA_to_SSB(mbh_tc_L, mbh_lam_L, np.arcsin(mbh_sinbeta_L), mbh_psi_L)
    mbh_lam_rad = (mbh_lam_SSB % (2 * np.pi)) - np.pi   # * 180. / np.pi - 180.0

    mbh_beta_rad = mbh_beta_SSB  #  * 180 / np.pi - 90.0

    ax.scatter(mbh_lam_rad, mbh_beta_rad, marker="x", color="C1", s=20)
    
    if save_file is not None:
        save_file_tmp = save_file[:-4] + "_sky_map.png"
        fig.savefig(save_file_tmp)

    plt.close()


def produce_gbs_plots(gb_reader, discard=0, save_file=None, fig=None):
    plt.close()
    nl = gb_reader.get_nleaves()
    ll = gb_reader.get_log_like()

    for i in range(nl["gb"].shape[1]):
        plt.plot(nl["gb"][:, i, 0], color=f"C{i % 10}", label="t = " + str(i + 1))
    plt.ylabel("Number of Binaries (max, mean, min)")
    plt.xlabel("Sampler Iteration (thinned)")
    plt.legend(loc="upper left")
    save_file_tmp = save_file[:-4] + "_gb_nleaves_over_time.png"
    plt.savefig(save_file_tmp)
    plt.close()
    plt.hist(nl["gb"][-500:, 0].flatten(), bins=30)
    plt.xlabel("Number of Binaries (last 500 iterations)")
    save_file_tmp = save_file[:-4] + "_gb_nleaves_hist.png"
    plt.savefig(save_file_tmp)
    plt.close()
    plt.hist(nl["gb"][-500:, 0].flatten(), bins=30)
    plt.xlabel("Number of Binaries (last 500 iterations)")
    save_file_tmp = save_file[:-4] + "_gb_nleaves_hist.png"
    plt.savefig(save_file_tmp)
    plt.close()
    plt.plot(ll[:, 0].mean(axis=-1), color="C1")
    plt.plot(ll[:, 0].max(axis=-1), color="C0", ls="--")
    plt.plot(ll[:, 0].min(axis=-1), color="C0", ls="--")
    plt.ylabel("logL (max, mean, min)")
    plt.xlabel("Sampler Iteration (thinned)")
    save_file_tmp = save_file[:-4] + "_gb_ll_over_time.png"
    plt.savefig(save_file_tmp)
    plt.close()
    


def produce_psd_plots(psd_reader, discard=0, save_file=None, fig=None):
    if fig is not None:
        plt.close()

    samples = np.concatenate([psd_reader.get_chain(discard=discard)["galfor"][:, 0].reshape(-1, 5), psd_reader.get_chain(discard=discard)["psd"][:, 0].reshape(-1, 4)], axis=-1)
    fig = corner.corner(samples, plot_datapoints=False, smooth=0.6, levels=1 - np.exp(-0.5 * np.array([1, 2, 3])**2), labels=["GB amplitude", "GB par1", "GB par2", "GB par3", "GB par4", "A_oms", "A_acc", "E_oms", "E_acc"])
    save_file_tmp = save_file[:-4] + "_psd_corner.png"
    fig.savefig(save_file_tmp)
    plt.close()


def make_current_plot(current_info, save_file=None, add_mbhs=False, add_gbs=False, **kwargs):
    generated_info_0 = current_info.get_data_psd(only_max_ll=True, include_gbs=False, include_mbhs=False, **kwargs)
    generated_info_1 = current_info.get_data_psd(only_max_ll=True, **kwargs)

    plt.loglog(current_info.general_info["fd"], 2 * current_info.general_info["df"] * np.abs(generated_info_0["data"][0]) ** 2)
    
    print("nleaves:", current_info.gb_info["reader"].get_nleaves()["gb"][-1].mean(axis=-1))
        
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
    plt.close()

class RunResultsProduction:
    def __init__(self, comm, head_rank, add_mbhs=False, add_gbs=False, **kwargs):
        self.comm = comm
        self.head_rank = head_rank
        self.add_mbhs = add_mbhs
        self.add_gbs = add_gbs
        self.kwargs = kwargs

    def run_results_production(**kwargs):

        self.comm.send({"ready": True}, dest=self.head_rank, tag=7979)

        run = True
        while run:

            current_info = self.comm.recv(source=self.head_rank, tag=7878)

            if isinstance(current_info, str):
                if current_info == "end":
                    run = False
                    continue

            self.build_plots(current_info)

            self.comm.send({"ready": True}, dest=self.head_rank, tag=7979)

    def build_plots(self, current_info):
        base_save_file = current_info.general_info["file_information"]["plot_base"]
        if base_save_file[-4:] == ".png":
            base_save_file = base_save_file[:-4]
        current_save_file = base_save_file + "_current_full_plot.png"
        make_current_plot(current_info, save_file=current_save_file, add_mbhs=self.add_mbhs, add_gbs=self.add_gbs, **self.kwargs)
        
        mbh_save_file = base_save_file + f"_mbh_posterior_leaf.png"
        produce_mbh_plots(current_info.mbh_info["reader"], current_info.mbh_info["cc_params"].shape[1], discard=0, save_file=mbh_save_file, fig=None)
        
        gb_save_file = base_save_file + f"_gb_posterior.png"
        produce_gbs_plots(current_info.gb_info["reader"], discard=0, save_file=gb_save_file, fig=None)

        psd_save_file = base_save_file + f"_psd_posterior.png"
        produce_psd_plots(current_info.psd_info["reader"], save_file=psd_save_file, discard=0, fig=None)

        skymap_file = base_save_file + "_sky_map.png"
        produce_sky_plot(current_info, save_file=skymap_file)