import numpy as np

from lisatools.sampling.utility import ModifiedHDFBackend
import corner.corner


class PlotContainer:
    def __init__(
        self,
        fp,
        source,
        include_titles=True,
        print_diagnostics=False,
        thin_chain_by_ac=True,
        corner_kwargs={},
        test_inds=None,
        sub_keys={},
        parameter_transforms=None,
    ):

        if parameter_transforms is not None:
            raise NotImplementedError

        self.fp = fp
        self.name = fp[:-3]  # remove .h5
        self.reader = ModifiedHDFBackend(fp)
        self.thin_chain_by_ac = thin_chain_by_ac
        self.print_diagnostics = print_diagnostics
        self.include_titles = include_titles
        self.test_inds = self.reader.get_attr("test_inds")

        if source not in ["emri", "gb", "mbh"]:
            raise NotImplementedError

        self.parameter_transforms = parameter_transforms
        self.corner_kwargs = corner_kwargs

        getattr(self, "_load_" + source)(test_inds=self.test_inds, sub_keys=sub_keys)

        self.injection = self.reader.get_attr("test_injection_values")

        default_corner_kwargs = dict(
            levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)),
            bins=25,
            plot_density=False,
            plot_datapoints=False,
            smooth=0.4,
            contour_kwargs={"colors": "blue"},
            hist_kwargs={"density": True},
            truths=self.injection,
        )

        for key, default in default_corner_kwargs.items():
            self.corner_kwargs[key] = self.corner_kwargs.get(key, default)

    def _load_mbh(self, test_inds=None, sub_keys={}):

        if test_inds is None:
            self.test_inds = test_inds = np.arange(11)

        if "labels" not in self.corner_kwargs:
            keys_default = [
                r"ln$M_T$",
                r"$q$",
                r"$a_1$",
                r"$a_2$",
                r"$d_L$",
                r"$\phi_0$",
                r"cos$\iota$",
                r"$\lambda$",
                r"sin$\beta$",
                r"$\psi$",
                r"$\t_c$",
            ]

            self.corner_kwargs["labels"] = [keys_default[i] for i in test_inds]

            for ind, label in sub_keys.items():
                self.corner_kwargs["labels"][ind] = label

    def _load_gb(self, test_inds=None, sub_keys={}):

        if test_inds is None:
            self.test_inds = test_inds = np.arange(8)

        if "labels" not in self.corner_kwargs:
            keys_default = [
                r"$A$",
                r"$f_0$ (mHz)",
                r"$\dot{f}_0$",
                r"$\ddot{f}_0$",
                r"$\phi_0$",
                r"cos$\iota$",
                r"$\psi$",
                r"$\lambda$",
                r"sin$\beta$",
            ]

            self.corner_kwargs["labels"] = [keys_default[i] for i in test_inds]

            for ind, label in sub_keys.items():
                self.corner_kwargs["labels"][ind] = label

    def _load_emri(self, test_inds=None, sub_keys={}):

        if test_inds is None:
            self.test_inds = test_inds = np.arange(6)

        if "labels" not in self.corner_kwargs:
            keys_default = [
                r"ln$M$",
                r"$\mu$",
                r"$a$",
                r"$p_0$",
                r"$e_0$",
                r"$Y_0$",
                r"$d_L$",
                r"$\theta_S$",
                r"$\phi_S$",
                r"$\theta_K$",
                r"$\phi_K$",
                r"$\Phi_{\varphi, 0}$",
                r"$\Phi_{\theta, 0}$",
                r"$\Phi_{r, 0}$",
            ]

            self.corner_kwargs["labels"] = [keys_default[i] for i in test_inds]

            for ind, label in sub_keys.items():
                self.corner_kwargs["labels"][ind] = label

    def generate_corner(self, burn=None, thin=None, **corner_kwargs):

        samples = self.reader.get_chain()

        tau = self.reader.get_autocorr_time(tol=0)
        if burn is None or thin is None:
            if burn is None:
                burn = int(2 * np.max(tau))

            if thin is None:
                thin = int(0.5 * np.min(tau))

        if thin == 0:
            thin = 1

        ntemps, nwalkers, ndim = self.reader.get_dims()
        ntemps_target = self.reader.get_attr("ntemps_target")
        original_samples_shape = samples.shape
        samples = samples[burn::thin, :ntemps_target].reshape(-1, ndim)
        plot_sample_shape = samples.shape

        corner_kwargs = {**corner_kwargs, **self.corner_kwargs}

        if self.print_diagnostics:
            print("burn-in: {0}".format(burn))
            print("thin: {0}".format(thin))
            print("flat chain shape: {0}".format(samples.shape))

        fig = corner.corner(samples, **corner_kwargs,)

        if self.include_titles:
            title_str = self.name + "\n"
            if ntemps > 1:
                title_str += "ntemps: {}\n".format(ntemps)
                title_str += "ntemps target: {}\n".format(ntemps_target)
                title_str += "nwalkers per temp: {}\n".format(nwalkers)

            else:
                title_str += "nwalkers: {}\n".format(nwalkers)

            title_str += "Sampler output shape: {}\n".format(original_samples_shape)
            title_str += "Plotted samples shape: {}\n".format(plot_sample_shape)
            title_str += "Burnin Samples: {} / Thin: {}\n".format(burn, thin)

            title_str += "Autocorrelation: {}".format(tau)

            fig.suptitle(title_str, fontsize=16)

        fig.savefig(self.name + "_corner.pdf")


if __name__ == "__main__":
    plot = PlotContainer("../GPU4GW/MBH_for_corner.h5", "mbh")

    plot.generate_corner()
