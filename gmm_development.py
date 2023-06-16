from lisatools.sampling.prior import FullGaussianMixtureModel
from eryn.prior import ProbDistContainer, uniform_dist
import numpy as np
import cupy as xp


if __name__ == "__main__":
    import pickle
    with open("gmm_info.pickle", "rb") as fp:
        gmm_info = pickle.load(fp)

    xp.cuda.runtime.setDevice(7)


    gmm_all = FullGaussianMixtureModel(*gmm_info, use_cupy=True)

    probs_in = {
        (0, 1, 2, 4, 6, 7): gmm_all,
        3: uniform_dist(0.0, 2 * np.pi, use_cupy=True),
        5: uniform_dist(0.0, np.pi, use_cupy=True)
    }
    gen_dist = ProbDistContainer(probs_in, use_cupy=True)
    points = gen_dist.rvs(size=(10000,))
    logpdf = gen_dist.logpdf(points)
    breakpoint()
    out = gmm_all.rvs(size=(10000,))
    check = gmm_all.logpdf(out)

    import matplotlib.pyplot as plt
    plt.scatter(out[:, 1], out[:, 0])
    plt.savefig("gmm_check.png")
    
    breakpoint()
