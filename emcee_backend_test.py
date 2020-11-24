import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt


reader = emcee.backends.HDFBackend("test_full_2yr.h5")

tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
# print("flat log prior shape: {0}".format(log_prior_samples.shape))

truth = [np.log(1e6), 1e2, 10.601813660750054, 0.2]

levels = 1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)
corner.corner(
    samples,
    truths=truth,
    plot_density=False,
    plot_datapoints=False,
    levels=levels,
    labels=[r"ln$M$", r"$\mu$", r"$p_0$", r"$e_0$"],
)

plt.savefig("test_runs_2yr.pdf")
plt.show()
