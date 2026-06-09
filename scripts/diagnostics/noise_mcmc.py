"""Eryn MCMC over the LISA WDM noise / confusion-foreground / SGWB parameters.

Injects a noise realization drawn from a known truth covariance (instrument +
galactic foreground + SGWB) on a WDM grid, then samples the 9 parameters back
with eryn and checks recovery.

Sampled parameters (units chosen so values are O(1) where convenient):
  0 soms_d       OMS displacement noise amplitude        [1e-12 m]
  1 sa_a         acceleration noise amplitude            [1e-15 m/s^2]
  2 fg_log10amp  galactic foreground log10(amplitude)
  3 fg_fk        foreground knee frequency               [1e-3 Hz]
  4 fg_alpha     foreground power-law index
  5 fg_s1        foreground slope below knee
  6 fg_s2        foreground slope above knee
  7 sgwb_log10A  SGWB log10(amplitude) at f_ref
  8 sgwb_alpha   SGWB power-law index

Likelihood (WDM real domain): see noise_mcmc_validate.py — the determinant term
needs the 1/2 that noise_likelihood_term omits, so
    logL = -0.5*<d|d>_C + 0.5*noise_likelihood_term(C).
"""
import dataclasses
import numpy as np
from scipy.interpolate import interp1d

from lisatools.sensitivity import (
    InstrumentNoise, GalacticForeground, SGWB, CompositeSensitivityMatrix,
)
from lisatools.datacontainer import DataResidualArray
from lisatools.diagnostic import inner_product, noise_likelihood_term
from lisatools.detector import sangria
from lisatools import domains

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State

# ----------------------------------------------------------------------------
# config (reduced "demo" grid by default; bump to 1536*4 / 338*24//4 for full res)
# ----------------------------------------------------------------------------
NF, NT, DT = 1536, 338, 5
INJECT_SEED = 0
NWALKERS, NBURN, NSTEPS = 24, 40, 160
INIT_SEED = 1234

PARAM_NAMES = ["soms_d", "sa_a", "fg_log10amp", "fg_fk", "fg_alpha",
               "fg_s1", "fg_s2", "sgwb_log10A", "sgwb_alpha"]
# SGWB indices: at the truth amplitude (log10_A=-20.45) the SGWB is ~12 orders
# below the noise in-band, so these two params are expected to be unconstrained
# (posterior = prior). Their walkers are spread across the prior accordingly.
SGWB_IDX = [7, 8]

# truth in *sampled* units
TRUTH = np.array([
    7.9,                          # soms_d  [1e-12 m]    (sangria (7.9e-12)**2)
    2.4,                          # sa_a    [1e-15 m/s^2](sangria (2.4e-15)**2)
    np.log10(3.26651613e-44),     # fg_log10amp
    2.09278117,                   # fg_fk   [1e-3 Hz]
    1.18300266,                   # fg_alpha
    3.01430978e03,                # fg_s1
    2.95774596e03,                # fg_s2
    -20.45,                       # sgwb_log10A
    2.0 / 3.0,                    # sgwb_alpha
])

# uniform prior bounds (lo, hi), bracketing truth
PRIOR_BOUNDS = np.array([
    [4.0, 14.0],      # soms_d
    [1.0, 5.0],       # sa_a
    [-45.0, -42.0],   # fg_log10amp
    [0.8, 4.0],       # fg_fk
    [0.3, 2.5],       # fg_alpha
    [500.0, 6000.0],  # fg_s1
    [500.0, 6000.0],  # fg_s2
    [-23.0, -19.0],   # sgwb_log10A
    [-1.5, 3.0],      # sgwb_alpha
])

# gaussian jitter for the initial walker ball (tight, near truth — short demo)
INIT_SIGMA = np.array([0.3, 0.1, 0.05, 0.05, 0.03, 80.0, 80.0, 0.1, 0.05])


def build_problem():
    settings = domains.WDMSettings(
        Nf=NF, Nt=NT, dt=DT, min_freq=3e-4, max_freq=8e-3, force_backend="cpu"
    )
    gm = np.loadtxt("./modulation.dat")
    mod = np.array([[gm[:, 1], gm[:, 4], gm[:, 5]],
                    [gm[:, 4], gm[:, 2], gm[:, 6]],
                    [gm[:, 5], gm[:, 6], gm[:, 3]]])
    modulation = interp1d(gm[:, 0], mod)(settings.t_arr)

    components = [
        InstrumentNoise(model="sangria"),
        GalacticForeground(foreground_params=[1.0, 1.0, 1.0, 1.0, 1.0], modulation=modulation),
        SGWB(sgwb_params=[-20.45, 2.0 / 3.0], stochastic_fn="PowerLawSGWB", modulation=None),
    ]
    sensmat = CompositeSensitivityMatrix(settings, components)
    return settings, components, sensmat


def set_params(components, sensmat, x):
    """Push a sampled parameter vector into the composite covariance and rebuild."""
    soms_d, sa_a, fg_la, fg_fk, fg_al, fg_s1, fg_s2, sg_lA, sg_al = x
    components[0].model = dataclasses.replace(
        sangria, Soms_d=(soms_d * 1e-12) ** 2, Sa_a=(sa_a * 1e-15) ** 2, name="mcmc"
    )
    components[1].foreground_params = (10.0 ** fg_la, fg_fk * 1e-3, fg_al, fg_s1, fg_s2)
    components[2].sgwb_params = (sg_lA, sg_al)
    sensmat.rebuild()


class WDMNoiseLogLike:
    """Callable eryn log-likelihood; mutates a shared composite matrix in place
    (safe because eryn calls this sequentially per walker with vectorize=False)."""

    def __init__(self, settings, components, sensmat, data):
        self.components = components
        self.sensmat = sensmat
        self.data = data

    def __call__(self, x, *args):
        try:
            set_params(self.components, self.sensmat, np.asarray(x))
            detC = self.sensmat.detC
            if (not np.all(np.isfinite(detC))) or np.any(detC <= 0.0):
                return -1e300  # not a valid (positive-definite) covariance
            dd = inner_product(self.data, self.data, psd=self.sensmat)
            nlt = noise_likelihood_term(self.sensmat)
            ll = 0.5 * nlt - 0.5 * dd
            return float(ll) if np.isfinite(ll) else -1e300
        except Exception:
            return -1e300


def inject_data(settings, components, sensmat):
    set_params(components, sensmat, TRUTH)
    C = sensmat.sens_mat
    nf, nt = C.shape[2], C.shape[3]
    Cp = C.transpose(2, 3, 0, 1).reshape(-1, 3, 3)
    L = np.linalg.cholesky(Cp)
    rng = np.random.default_rng(INJECT_SEED)
    z = rng.standard_normal((Cp.shape[0], 3, 1))
    data_inj = (L @ z)[:, :, 0].reshape(nf, nt, 3).transpose(2, 0, 1).real
    return DataResidualArray(data_inj, input_signal_domain=settings), nf * nt


def main():
    ndim = len(PARAM_NAMES)
    settings, components, sensmat = build_problem()
    data, npix = inject_data(settings, components, sensmat)
    print(f"grid: Nf={NF} Nt={NT}  active pixels={npix}  ndim={ndim}")

    log_like = WDMNoiseLogLike(settings, components, sensmat, data)

    # sanity: truth should beat perturbed points
    ll_truth = log_like(TRUTH)
    print(f"logL(truth)               = {ll_truth:.1f}")
    for i, name in enumerate(PARAM_NAMES):
        xp = TRUTH.copy()
        xp[i] += 3.0 * INIT_SIGMA[i]
        print(f"  logL(truth, {name:>11}+3sig) = {log_like(xp):.1f}  (dlogL={log_like(xp)-ll_truth:+.1f})")

    priors = {"model_0": ProbDistContainer(
        {i: uniform_dist(PRIOR_BOUNDS[i, 0], PRIOR_BOUNDS[i, 1]) for i in range(ndim)}
    )}

    # initialize a tight walker ball around truth for the constrained params; the
    # unconstrained SGWB params are spread across their prior so the flat posterior
    # is visible even in a short chain.
    rng = np.random.default_rng(INIT_SEED)
    p0 = TRUTH[None, :] + INIT_SIGMA[None, :] * rng.standard_normal((NWALKERS, ndim))
    for i in SGWB_IDX:
        p0[:, i] = rng.uniform(PRIOR_BOUNDS[i, 0], PRIOR_BOUNDS[i, 1], size=NWALKERS)
    p0 = np.clip(p0, PRIOR_BOUNDS[:, 0] + 1e-6, PRIOR_BOUNDS[:, 1] - 1e-6)
    in_bounds = np.all((p0 > PRIOR_BOUNDS[:, 0]) & (p0 < PRIOR_BOUNDS[:, 1]), axis=1)
    assert in_bounds.all(), "init walker outside prior"

    sampler = EnsembleSampler(NWALKERS, ndim, log_like, priors, vectorize=False)
    state = State({"model_0": p0[None, :, None, :]})

    print(f"\nrunning eryn: {NWALKERS} walkers, burn={NBURN}, steps={NSTEPS} ...")
    sampler.run_mcmc(state, NSTEPS, burn=NBURN, progress=True)

    chain = sampler.get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, ndim)  # (nsteps*nwalkers, ndim)
    np.save("noise_mcmc_chain.npy", chain)

    print("\n=== recovery (post-burn) ===")
    print(f"{'param':>12} {'truth':>12} {'median':>12} {'16%':>12} {'84%':>12} {'(med-tru)/sig':>14} {'note':>14}")
    prior_w = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
    for i, name in enumerate(PARAM_NAMES):
        med = np.median(chain[:, i])
        lo, hi = np.percentile(chain[:, i], [16, 84])
        sig = 0.5 * (hi - lo)
        z = (med - TRUTH[i]) / sig if sig > 0 else np.nan
        # flag params whose 68% interval spans a large fraction of the prior
        note = "unconstrained" if (hi - lo) > 0.4 * prior_w[i] else ""
        print(f"{name:>12} {TRUTH[i]:12.4g} {med:12.4g} {lo:12.4g} {hi:12.4g} {z:14.2f} {note:>14}")

    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = corner.corner(chain, labels=PARAM_NAMES, truths=list(TRUTH))
        fig.savefig("noise_mcmc_corner.png", dpi=120)
        print("\nsaved noise_mcmc_corner.png and noise_mcmc_chain.npy")
    except Exception as e:
        print(f"\nsaved noise_mcmc_chain.npy (corner plot skipped: {e})")


if __name__ == "__main__":
    main()
