# %% [markdown]
# # Scattered-field → distribution (`rvs` / `logpdf`) for LISA F-stat proposals
#
# Given a **cloud of parameter points** `θ_i ∈ ℝ^d` (`d = 3` or `4`), each with a
# **scalar statistic** `s_i = s(θ_i)` (think: the GB **F-statistic** from
# `get_fstat_ll_wdm`), build a distribution object with `rvs(size)` and
# `logpdf(x)` that plugs into the `eryn` / `lisatools` `priors.py` machinery and
# runs on GPU.
#
# Target density (with an inverse-temperature knob `β`):
# $$ p(θ) \propto \exp\!\big(β\,s(θ)\big) $$
#
# **We control where the points go**, so this is *design-of-experiments +
# surrogate modeling*, not density-estimation-from-a-fixed-sample. That makes
# **field interpolation** (Family A) and **analytic surrogates** (Family C) the
# workhorses; KDE (Family D) is the fixed-cloud fallback.
#
# All code is **`xp`-agnostic**: it uses `numpy` here and transplants to `cupy`
# by flipping `USE_GPU = True`. See the plan in `PLAN_scattered_field_distributions.md`.

# %%
import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False

USE_GPU = False  # flip to True on a CUDA box (cupy installed) — everything below is xp-generic
xp = cp if (USE_GPU and HAS_CUPY) else np


def to_np(a):
    """Bring an xp array back to host numpy (for plotting / printing)."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def logsumexp(a, axis=None, keepdims=False):
    """xp-generic, numerically stable log-sum-exp."""
    m = xp.max(a, axis=axis, keepdims=True)
    m = xp.where(xp.isfinite(m), m, xp.zeros_like(m))
    out = xp.log(xp.sum(xp.exp(a - m), axis=axis, keepdims=True)) + m
    if not keepdims:
        out = xp.squeeze(out, axis=axis)
    return out


def weighted_choice(weights, size, rng):
    """xp-generic categorical draw ∝ weights → integer indices of shape `size`."""
    cdf = xp.cumsum(weights)
    cdf = cdf / cdf[-1]
    u = xp.asarray(rng.random(size))
    return xp.searchsorted(cdf, u.ravel(), side="right").reshape(u.shape)


rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A toy 4D "F-stat-like" field with a *known* target
#
# We fake `s(θ)` as the log-pdf of a Gaussian mixture, so `exp(s)` is analytic and
# we can validate every method against ground truth (`β = 1` ⇒ target *is* the
# mixture). The mixture has a **dominant narrow peak** (the true source), a
# **curved sky ridge** (two offset elongated blobs — a fake sky degeneracy), and a
# **broad background**. In production `s` comes from `F = ½ Nᵀ M⁻¹ N`; the methods
# only ever see evaluations `s_i`, never the mixture.

# %%
NDIM = 4
BOUNDS = xp.array([[0.0, 1.0]] * NDIM)  # unit box (stand-in for f0, fdot, lam, beta)


class GaussianMixtureTruth:
    """Ground-truth field: s(θ) = log Σ_k a_k N(θ; μ_k, Σ_k)."""

    def __init__(self, weights, means, covs):
        self.w = xp.asarray(weights) / xp.sum(xp.asarray(weights))
        self.means = xp.asarray(means)
        self.covs = xp.asarray(covs)
        self.K, self.d = self.means.shape
        self.chol = xp.linalg.cholesky(self.covs)
        _, self.logdet = xp.linalg.slogdet(self.covs)

    def _log_comp(self, x):
        # x: (n, d) -> (n, K) per-component log N
        diff = x[:, None, :] - self.means[None, :, :]  # (n,K,d)
        sol = xp.linalg.solve(self.covs[None, :, :, :], diff[..., None])[..., 0]  # (n,K,d)
        maha = xp.sum(diff * sol, axis=-1)  # (n,K)
        return -0.5 * (self.d * np.log(2 * np.pi) + self.logdet[None, :] + maha)

    def logpdf(self, x):
        return logsumexp(xp.log(self.w)[None, :] + self._log_comp(x), axis=1)

    def s(self, x):
        """The 'F-statistic' value the tools would return at θ = x."""
        return self.logpdf(x)

    def rvs(self, n):
        comp = weighted_choice(self.w, (n,), rng)
        z = xp.asarray(rng.standard_normal((n, self.d)))
        return self.means[comp] + xp.einsum("nij,nj->ni", self.chol[comp], z)


def _iso(c):
    return (c**2) * xp.eye(NDIM)


_means = xp.array(
    [
        [0.50, 0.50, 0.50, 0.50],  # dominant peak
        [0.30, 0.35, 0.62, 0.40],  # sky ridge lobe A
        [0.38, 0.45, 0.55, 0.52],  # sky ridge lobe B
        [0.55, 0.55, 0.55, 0.55],  # broad background
    ]
)
_covs = xp.stack(
    [
        _iso(0.06),
        xp.diag(xp.array([0.03, 0.12, 0.05, 0.08])) ** 1,  # elongated (fake sky ridge)
        xp.diag(xp.array([0.05, 0.10, 0.07, 0.06])) ** 1,
        _iso(0.25),
    ]
)
_weights = xp.array([0.45, 0.20, 0.20, 0.15])
truth = GaussianMixtureTruth(_weights, _means, _covs)

BETA = 1.0  # inverse-temperature; 1.0 => target is exactly `truth`
print("truth: K =", truth.K, " ndim =", truth.d)

# %% [markdown]
# ## 2. Design of experiments — *we choose where to evaluate `s`*
#
# Two designs:
# * a **tensor grid** (feeds Family A), and
# * a **uniform random cloud** with importance weights `w_i ∝ exp(β s_i)`
#   (feeds Families C & D — the "fixed cloud" style).
#
# In production each `θ_i` batch is passed to `get_fstat_ll_wdm(design, wdm_holder)`
# → `N, M` → `F = ½ Nᵀ M⁻¹ N`, all on the GPU in one batched launch.

# %%
# --- grid design (Family A) ---
N_PER_AXIS = 24  # 24^4 = 331,776 nodes
axes = [xp.linspace(BOUNDS[j, 0], BOUNDS[j, 1], N_PER_AXIS) for j in range(NDIM)]
mesh = xp.meshgrid(*axes, indexing="ij")
grid_pts = xp.stack([m.ravel() for m in mesh], axis=1)  # (N^d, d)
s_grid = truth.s(grid_pts)  # <-- the F-stat evaluations on the grid
g_grid = (BETA * s_grid).reshape([N_PER_AXIS] * NDIM)  # log-target field on the grid

# --- random cloud design (Families C, D) ---
N_CLOUD = 15000
cloud_pts = xp.asarray(rng.random((N_CLOUD, NDIM)))
s_cloud = truth.s(cloud_pts)
log_w_cloud = BETA * s_cloud  # importance log-weight (uniform design ⇒ q const)
w_cloud = xp.exp(log_w_cloud - xp.max(log_w_cloud))
print("grid nodes:", grid_pts.shape[0], " cloud points:", cloud_pts.shape[0])

# %% [markdown]
# ## Family A — Structured grid + inverse-CDF  (**recommended baseline**)
#
# Represent `p` as a normalized histogram-density on the grid (each node = a cell
# center of width `dx`). `rvs` = categorical over cells `∝ exp(g)` + uniform
# in-cell jitter (this is the flattened form of exact **conditional inverse-CDF**
# sampling — `cumsum` + `searchsorted`). `logpdf` = the cell's `g − log Z`.
# Pure `xp`; no fitting, no native kernel. `rvs` and `logpdf` are *exactly*
# consistent for this piecewise-constant density.

# %%
class GridInverseCDF:
    def __init__(self, axes, g):
        self.axes = axes
        self.g = g  # (N,)*d log-target on grid nodes
        self.ndim = len(axes)
        self.dx = xp.array([ax[1] - ax[0] for ax in axes])
        self.lo = xp.array([ax[0] for ax in axes])
        self.n = xp.array([ax.shape[0] for ax in axes])
        self.cell_vol = xp.prod(self.dx)
        # normalization (rectangle rule): Z = Σ exp(g) * cell_vol
        self.logZ = logsumexp(self.g.ravel()) + xp.log(self.cell_vol)
        # flattened cdf for sampling
        flat = self.g.ravel()
        p = xp.exp(flat - xp.max(flat))
        self._cdf = xp.cumsum(p)
        self._cdf = self._cdf / self._cdf[-1]

    def _nearest_index(self, x):
        # (n,d) physical -> (n,d) integer node indices (nearest cell center)
        idx = xp.round((x - self.lo[None, :]) / self.dx[None, :]).astype(xp.int64)
        inside = xp.all((idx >= 0) & (idx < self.n[None, :]), axis=1)
        idx = xp.clip(idx, 0, (self.n - 1)[None, :])
        return idx, inside

    def logpdf(self, x):
        x = xp.atleast_2d(x)
        idx, inside = self._nearest_index(x)
        g_here = self.g[tuple(idx[:, j] for j in range(self.ndim))]
        out = g_here - self.logZ
        return xp.where(inside, out, -xp.inf)

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        u = xp.asarray(rng.random(size))
        flat_idx = xp.searchsorted(self._cdf, u.ravel(), side="right")
        flat_idx = xp.clip(flat_idx, 0, self._cdf.shape[0] - 1)
        multi = xp.unravel_index(flat_idx, tuple(int(v) for v in self.n))  # tuple of (M,)
        centers = xp.stack([self.axes[j][multi[j]] for j in range(self.ndim)], axis=1)
        jitter = (xp.asarray(rng.random(centers.shape)) - 0.5) * self.dx[None, :]
        pts = xp.clip(centers + jitter, self.lo[None, :], self.lo[None, :] + (self.n - 1)[None, :] * self.dx[None, :])
        return pts.reshape(size + (self.ndim,))


gridA = GridInverseCDF(axes, g_grid)
print("Family A built. log Z =", float(to_np(gridA.logZ)))

# %% [markdown]
# ## Family C — Gaussian-mixture surrogate  (**analytic; mirrors `FullGaussianMixtureModel`**)
#
# Fit a `K`-component GMM to `exp(β s)` by **weighted EM** on the cloud (weights
# `∝ exp(β s_i)`). `logpdf` = `logsumexp` over components; `rvs` = pick component
# `∝ weight`, Cholesky draw. In LAT this is *already built*:
# `vec_fit_gmm_min_bic` (`gmm.py:2134`, batched GPU EM + BIC) →
# `FullGaussianMixtureModel` (`prior.py:578`, native `compute_logpdf` kernel).
# The only new production code is the F-stat→resample→fit glue.

# %%
class GMMDistribution:
    def __init__(self, weights, means, covs):
        self.w = xp.asarray(weights) / xp.sum(xp.asarray(weights))
        self.means = xp.asarray(means)
        self.covs = xp.asarray(covs)
        self.K, self.ndim = self.means.shape
        self.chol = xp.linalg.cholesky(self.covs)
        _, self.logdet = xp.linalg.slogdet(self.covs)

    def _log_comp(self, x):
        diff = x[:, None, :] - self.means[None, :, :]
        sol = xp.linalg.solve(self.covs[None, :, :, :], diff[..., None])[..., 0]
        maha = xp.sum(diff * sol, axis=-1)
        return -0.5 * (self.ndim * np.log(2 * np.pi) + self.logdet[None, :] + maha)

    def logpdf(self, x):
        x = xp.atleast_2d(x)
        return logsumexp(xp.log(self.w)[None, :] + self._log_comp(x), axis=1)

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        comp = weighted_choice(self.w, (n,), rng)
        z = xp.asarray(rng.standard_normal((n, self.ndim)))
        pts = self.means[comp] + xp.einsum("nij,nj->ni", self.chol[comp], z)
        return pts.reshape(size + (self.ndim,))

    @classmethod
    def fit(cls, points, sample_weights, K, n_iter=150, reg=1e-5):
        n, d = points.shape
        w = sample_weights / xp.sum(sample_weights)
        # init: K weighted-random points as means; global weighted cov
        mu = points[weighted_choice(w, (K,), rng)]
        mean_all = xp.sum(w[:, None] * points, axis=0)
        diff_all = points - mean_all[None, :]
        cov_all = (w[:, None] * diff_all).T @ diff_all + reg * xp.eye(d)
        cov = xp.repeat(cov_all[None, :, :], K, axis=0)
        mix = xp.ones(K) / K
        for _ in range(n_iter):
            # E-step
            diff = points[:, None, :] - mu[None, :, :]  # (n,K,d)
            sol = xp.linalg.solve(cov[None, :, :, :], diff[..., None])[..., 0]
            maha = xp.sum(diff * sol, axis=-1)
            _, logdet = xp.linalg.slogdet(cov)
            log_comp = -0.5 * (d * np.log(2 * np.pi) + logdet[None, :] + maha)
            log_r = xp.log(mix)[None, :] + log_comp
            log_r = log_r - logsumexp(log_r, axis=1, keepdims=True)
            r = xp.exp(log_r) * w[:, None]  # weighted responsibilities (n,K)
            # M-step
            Nk = xp.sum(r, axis=0) + 1e-300
            mix = Nk / xp.sum(Nk)
            mu = (r.T @ points) / Nk[:, None]
            for k in range(K):
                dk = points - mu[k][None, :]
                cov[k] = (r[:, k][:, None] * dk).T @ dk / Nk[k] + reg * xp.eye(d)
        return cls(mix, mu, cov)


gmmC = GMMDistribution.fit(cloud_pts, w_cloud, K=8, n_iter=120)
print("Family C fitted:", gmmC.K, "components; mixture weights =", np.round(to_np(gmmC.w), 3))

# %% [markdown]
# ## Family D — Weighted KDE  (*fixed-cloud fallback*)
#
# A Gaussian kernel per cloud point, weighted by `w_i ∝ exp(β s_i)`, with a
# global full-covariance bandwidth (Scott's rule × weighted covariance).
# `logpdf` is a dense `logsumexp` over kernels (chunked); `rvs` picks a point
# `∝ w_i` and adds kernel noise. Simple and exact for the KDE density, but
# `O(N²)` `logpdf` and bandwidth-sensitive — the fallback when a grid design
# is impossible.

# %%
class WeightedKDE:
    def __init__(self, points, weights, bw_factor=None):
        self.points = xp.asarray(points)
        self.w = xp.asarray(weights) / xp.sum(xp.asarray(weights))
        self.n, self.ndim = self.points.shape
        mean = xp.sum(self.w[:, None] * self.points, axis=0)
        diff = self.points - mean[None, :]
        cov = (self.w[:, None] * diff).T @ diff
        neff = 1.0 / xp.sum(self.w**2)
        if bw_factor is None:
            bw_factor = float(neff) ** (-1.0 / (self.ndim + 4))  # Scott
        self.H = (bw_factor**2) * cov + 1e-6 * xp.eye(self.ndim)
        self.cholH = xp.linalg.cholesky(self.H)
        _, self.logdetH = xp.linalg.slogdet(self.H)
        self.Hinv = xp.linalg.inv(self.H)
        self.log_w = xp.log(self.w)

    def logpdf(self, x, chunk=512):
        x = xp.atleast_2d(x)
        norm = -0.5 * (self.ndim * np.log(2 * np.pi) + self.logdetH)
        out = xp.empty(x.shape[0])
        for i in range(0, x.shape[0], chunk):
            xi = x[i : i + chunk]
            diff = xi[:, None, :] - self.points[None, :, :]  # (c, n, d)
            maha = xp.einsum("cnd,de,cne->cn", diff, self.Hinv, diff)
            out[i : i + chunk] = logsumexp(self.log_w[None, :] + norm - 0.5 * maha, axis=1)
        return out

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        idx = weighted_choice(self.w, (n,), rng)
        z = xp.asarray(rng.standard_normal((n, self.ndim)))
        pts = self.points[idx] + (z @ self.cholH.T)
        return pts.reshape(size + (self.ndim,))


kdeD = WeightedKDE(cloud_pts, w_cloud)
print("Family D built. bandwidth diag(H) =", np.round(to_np(xp.diag(kdeD.H)), 5))

# %% [markdown]
# ## 3. Validation
#
# For each method:
# 1. **`logpdf` vs truth** — on independent test points, method `logpdf` should
#    equal `β·s − log Z_true` (both normalized). Scatter + RMS.
# 2. **`rvs` marginals vs truth** — 1D histograms vs the analytic mixture marginal.
# 3. **Normalization check** — `∫ exp(logpdf) dθ ≈ 1` via a fine grid.

# %%
# uniform MC points reused for both the true-Z estimate and per-method
# normalization checks (random points don't alias sharp peaks the way a coarse
# grid quadrature does).
box_vol = float(to_np(xp.prod(BOUNDS[:, 1] - BOUNDS[:, 0])))
norm_pts = xp.asarray(rng.random((40000, NDIM)))
logt_norm = BETA * truth.s(norm_pts)
logZ_true = float(to_np(logsumexp(logt_norm) - np.log(norm_pts.shape[0]) + np.log(box_vol)))
print("log Z_true (MC) =", round(logZ_true, 4))

test_pts = truth.rvs(3000)  # test where the mass is
test_pts = xp.clip(test_pts, BOUNDS[:, 0][None, :] + 1e-6, BOUNDS[:, 1][None, :] - 1e-6)
true_logpdf = BETA * truth.s(test_pts) - logZ_true

methods = {"A: grid+invCDF": gridA, "C: GMM": gmmC, "D: KDE": kdeD}


def shape_rms(a, b):
    """RMS of (a-b) after removing the constant offset -> pure shape error."""
    a, b = to_np(a), to_np(b)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    d = d - d.mean()
    return float(np.sqrt(np.mean(d**2)))


print("\nlogpdf shape-RMS vs truth (nats, offset removed, lower=better):")
for name, dist in methods.items():
    lp = dist.logpdf(test_pts)
    print(f"  {name:16s}: {shape_rms(lp, true_logpdf):.3f}")

# normalization: ∫ exp(logpdf) dθ ≈ box_vol * E_uniform[exp(logpdf)]
print("\n∫ exp(logpdf) dθ  (should be ≈ 1):")
for name, dist in methods.items():
    integ = float(to_np(xp.mean(xp.exp(dist.logpdf(norm_pts))) * box_vol))
    print(f"  {name:16s}: {integ:.3f}")

# %% [markdown]
# ### Plots — `logpdf` scatter and `rvs` marginals

# %%
try:
    import matplotlib

    matplotlib.use("Agg")  # remove for interactive; keeps this runnable headless
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception as e:  # pragma: no cover - environment dependent
    HAS_MPL = False
    print("matplotlib unavailable, skipping plots:", e)


def true_marginal_pdf(j, xs):
    # mixture marginal in dim j = Σ_k w_k N(xs; μ_kj, Σ_kjj)
    xs = to_np(xs)
    w = to_np(truth.w)
    mu = to_np(truth.means)[:, j]
    var = to_np(truth.covs)[:, j, j]
    p = np.zeros_like(xs)
    for k in range(truth.K):
        p += w[k] * np.exp(-0.5 * (xs - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k])
    return p


def plot_validation():
    fig, axarr = plt.subplots(len(methods), NDIM, figsize=(4 * NDIM, 3 * len(methods)))
    for r, (name, dist) in enumerate(methods.items()):
        samp = to_np(dist.rvs(20000))
        for j in range(NDIM):
            ax = axarr[r, j]
            ax.hist(samp[:, j], bins=60, range=(0, 1), density=True, alpha=0.6, label="rvs")
            xs = np.linspace(0, 1, 300)
            ax.plot(xs, true_marginal_pdf(j, xs), "k-", lw=1.5, label="truth")
            if j == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(f"param {j}")
            if r == 0 and j == NDIM - 1:
                ax.legend(fontsize=7)
    fig.suptitle("rvs marginals vs truth (beta = %.2f)" % BETA)
    fig.tight_layout()
    fig.savefig("validation_marginals.png", dpi=110)

    fig2, ax2 = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5))
    for c, (name, dist) in enumerate(methods.items()):
        lp = to_np(dist.logpdf(test_pts))
        tl = to_np(true_logpdf)
        m = np.isfinite(lp)
        ax2[c].scatter(tl[m], lp[m], s=3, alpha=0.3)
        lim = [min(tl[m].min(), lp[m].min()), max(tl[m].max(), lp[m].max())]
        ax2[c].plot(lim, lim, "k--", lw=1)
        ax2[c].set_title(name)
        ax2[c].set_xlabel("true logpdf")
        ax2[c].set_ylabel("method logpdf")
    fig2.tight_layout()
    fig2.savefig("validation_logpdf.png", dpi=110)
    print("saved validation_marginals.png, validation_logpdf.png")


if HAS_MPL:
    plot_validation()

# %% [markdown]
# ## 4. The `priors.py` interface + a mock `ProbDistContainer`
#
# `eryn` distributions are duck-typed: `rvs(size) → size+(ndim,)` and
# `logpdf(x) → (n,)` with `x` shape `(n, ndim)`. Our joint distribution registers
# under a **tuple key** so the container maps it to several columns at once. Below
# is a minimal container mirroring `eryn.prior.ProbDistContainer` semantics.

# %%
class MockProbDistContainer:
    """Minimal stand-in for eryn.prior.ProbDistContainer (independent blocks)."""

    def __init__(self, priors_in):
        self.priors = [(xp.asarray(np.atleast_1d(k)), d) for k, d in priors_in.items()]
        self.ndim = int(max(int(xp.max(inds)) for inds, _ in self.priors) + 1)

    def logpdf(self, x):
        x = xp.atleast_2d(x)
        out = xp.zeros(x.shape[0])
        for inds, dist in self.priors:
            out = out + xp.squeeze(dist.logpdf(x[:, inds]))
        return out

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        out = xp.zeros(size + (self.ndim,))
        for inds, dist in self.priors:
            draw = dist.rvs(size=size)
            if len(inds) == 1:
                out[..., inds] = draw[..., None] if draw.shape[-1:] != (1,) else draw
            else:
                out[..., inds] = draw
        return out


# joint 4D F-stat proposal on columns (0,1,2,3); (in a real run add other 1D priors)
container = MockProbDistContainer({(0, 1, 2, 3): gridA})
draws = container.rvs(size=(1000,))
lp = container.logpdf(draws)
print("container.rvs shape:", tuple(draws.shape), " logpdf shape:", tuple(lp.shape))
print("finite logpdf fraction:", float(to_np(xp.mean(xp.isfinite(lp).astype(xp.float64)))))

# %% [markdown]
# ## 5. Temperature knob + GPU notes
#
# * **`β` (temperature).** Rebuild with `BETA = 0.5` to broaden the proposal
#   (safer MCMC acceptance) or `BETA = 2.0` to sharpen. Only `g_grid` /
#   `log_w_cloud` change; the classes are `β`-agnostic.
# * **GPU.** Flip `USE_GPU = True` (with `cupy`): every class is already `xp`-generic.
#   Family A is pure `cupy` (`cumsum`/`searchsorted`/`unravel_index`). Family C
#   maps onto the *existing* GPU `FullGaussianMixtureModel` + `vec_fit_gmm_min_bic`
#   — prefer those in production (native `compute_logpdf` kernel with per-box
#   spatial culling). Family D's dense `logpdf` is the one place a custom CUDA
#   kernel (tiled kernel sum, or KMeans kernel reduction) would earn its keep.
# * **Real F-stat.** Replace `truth.s(θ)` with:
#   `N, M = get_fstat_ll_wdm(θ, wdm_holder); F = 0.5 * einsum('bi,bij,bj->b', N, solve(M4x4, N))`
#   where `M4x4` is the 10→(4,4) symmetric un-pack. `θ = (f0, fdot, lam, beta)`.

# %% [markdown]
# ## 7. Narrow 4D peaks? **Measure first** (diagnostic), then branch
#
# Don't assume the regime — *check it*. Near a peak the amplitude-maximized
# statistic is quadratic, `2F ≈ 2F_peak − (θ−θ_peak)ᵀ Γ (θ−θ_peak)`, so a few
# **batched** F-stat evals around a candidate give the curvature `Γ` (the Fisher
# metric) by finite difference — and `Γ` *is* the peak shape. From it we read the
# per-axis width `σ_i`, the **resolution elements** `R_i = L_i/σ_i` (how many grid
# nodes/axis a global grid would need), and the **metric template count** (search
# difficulty). Point `s_fn` at the real F-stat to settle the question for your data.

# %%
def _pd_project(A, floor=1e-8):
    """Nearest positive-definite matrix (eigenvalue clip)."""
    w, V = xp.linalg.eigh((A + A.T) / 2)
    w = xp.clip(w, floor, None)
    return (V * w) @ V.T


def fd_grad_hess(s_fn, theta, h=2e-3):
    """Gradient + Hessian of s at theta via central finite differences.

    Builds the whole stencil and evaluates s_fn ONCE (mirrors batching the
    stencil into get_fstat_ll_wdm)."""
    theta = xp.asarray(theta, dtype=float)
    d = theta.shape[0]
    eye = xp.eye(d) * h
    pts, key = [theta], {}
    key["0"] = 0

    def add(k, dv):
        key[k] = len(pts)
        pts.append(theta + dv)

    for i in range(d):
        add(("+", i), eye[i])
        add(("-", i), -eye[i])
    for i in range(d):
        for j in range(i + 1, d):
            add(("++", i, j), eye[i] + eye[j])
            add(("+-", i, j), eye[i] - eye[j])
            add(("-+", i, j), -eye[i] + eye[j])
            add(("--", i, j), -eye[i] - eye[j])
    S = s_fn(xp.stack(pts))
    s0 = S[key["0"]]
    g = xp.zeros(d)
    H = xp.zeros((d, d))
    for i in range(d):
        sp, sm = S[key[("+", i)]], S[key[("-", i)]]
        g[i] = (sp - sm) / (2 * h)
        H[i, i] = (sp - 2 * s0 + sm) / (h * h)
    for i in range(d):
        for j in range(i + 1, d):
            hij = (S[key[("++", i, j)]] - S[key[("+-", i, j)]] - S[key[("-+", i, j)]] + S[key[("--", i, j)]]) / (4 * h * h)
            H[i, j] = hij
            H[j, i] = hij
    return float(to_np(s0)), g, H


def refine_peak(s_fn, theta0, n_steps=8, h=2e-3, max_step=0.05):
    """Newton ascent to the local max of s (θ ← θ + (−H)⁻¹∇), F-stat evals only."""
    theta = xp.asarray(theta0, dtype=float)
    for _ in range(n_steps):
        _, g, H = fd_grad_hess(s_fn, theta, h)
        step = xp.linalg.solve(_pd_project(-H), g)
        nrm = float(to_np(xp.sqrt(xp.sum(step**2))))
        if nrm > max_step:
            step = step * (max_step / nrm)
        theta = xp.clip(theta + step, BOUNDS[:, 0], BOUNDS[:, 1])
    return theta


def characterize_peak(s_fn, theta_peak, beta, h=2e-3):
    """Local Laplace shape of exp(β s): mean, cov=(β·(−H))⁻¹, per-axis σ."""
    s0, _, H = fd_grad_hess(s_fn, theta_peak, h)
    negH = _pd_project(-H)
    cov = xp.linalg.inv(beta * negH)
    cov = (cov + cov.T) / 2
    sigma = xp.sqrt(xp.clip(xp.diag(cov), 1e-30, None))
    return dict(mu=theta_peak, cov=cov, sigma=sigma, s_peak=s0, negH=negH)


def peak_width_report(s_fn, beta, n_scan=60000, top=60, n_report=3, min_sep=0.08, node_per_sigma=3.0):
    """Coarse-scan → refine → measure width. Prints the narrow-vs-not verdict."""
    L = to_np(BOUNDS[:, 1] - BOUNDS[:, 0])
    scan = xp.asarray(rng.random((n_scan, NDIM)))
    order = to_np(xp.argsort(s_fn(scan)))[::-1][:top]
    peaks = []
    for idx in order:
        th = refine_peak(s_fn, scan[int(idx)])
        thn = to_np(th)
        if all(np.sqrt(np.sum((thn - to_np(p["mu"])) ** 2)) > min_sep for p in peaks):
            peaks.append(characterize_peak(s_fn, th, beta))
    peaks.sort(key=lambda p: -p["s_peak"])
    print(f"found {len(peaks)} distinct peak(s); reporting top {min(n_report, len(peaks))}")
    Ntot = []
    for k, p in enumerate(peaks[:n_report]):
        sig = to_np(p["sigma"])
        R = L / sig
        Ni = node_per_sigma * R
        nt = float(np.prod(Ni))
        Ntot.append(nt)
        cov = to_np(p["cov"])
        ev = np.linalg.eigvalsh(cov)
        cond = ev[-1] / max(ev[0], 1e-30)  # >>1 => correlated ridge (whitening helps)
        _, logdet = np.linalg.slogdet(to_np(beta * p["negH"]))
        ntempl = np.exp(0.5 * logdet) * float(np.prod(L))
        print(f"  peak {k}: mu={np.round(to_np(p['mu']), 3)}  s_peak={p['s_peak']:.2f}")
        print(f"    sigma/axis    = {np.round(sig, 4)}")
        print(f"    R=range/sigma = {np.round(R, 1)}   (grid nodes/axis needed ~ {np.round(Ni, 0)})")
        print(f"    -> global-grid N_total ~ {nt:.2e}   | metric templates ~ {ntempl:.2e}   | cond(cov)={cond:.1f}")
    verdict = "GLOBAL GRID FEASIBLE" if (Ntot and max(Ntot) < 1e7) else "NARROW -> peak-find + local surrogate"
    print("  VERDICT:", verdict, "\n")
    return peaks


print("=== BROAD toy (section 1) ===")
_ = peak_width_report(truth.s, BETA, n_report=2)

# %% [markdown]
# ### A narrow-peak toy — same box, tiny σ (and a correlated ridge)

# %%
def narrow_cov(s, rho01=0.0, rho23=0.0):
    C = xp.zeros((NDIM, NDIM))
    for i in range(NDIM):
        C[i, i] = s[i] ** 2
    C[0, 1] = C[1, 0] = rho01 * s[0] * s[1]
    C[2, 3] = C[3, 2] = rho23 * s[2] * s[3]
    return C


narrow_truth = GaussianMixtureTruth(
    xp.array([0.6, 0.4]),
    xp.array([[0.50, 0.50, 0.50, 0.50], [0.30, 0.62, 0.35, 0.68]]),
    xp.stack(
        [
            narrow_cov(xp.array([0.020, 0.020, 0.05, 0.05]), rho01=0.7),  # tight, f0-fdot ridge
            narrow_cov(xp.array([0.020, 0.030, 0.06, 0.04]), rho23=0.8),  # tight, sky ridge
        ]
    ),
)
print("=== NARROW toy ===")
narrow_peaks = peak_width_report(narrow_truth.s, BETA, n_report=2)

# %% [markdown]
# ### The narrow-peak methods
#
# **(1) Laplace-Gaussian-per-peak → GMM** (the primary path): the diagnostic
# already gave `(μ_k, Σ_k)`; weight by the Laplace evidence and feed `GMMDistribution`
# (production: `FullGaussianMixtureModel`). **(2) Whitened local grid** (Family A
# applied *locally* in `u = Lᵀ(θ−μ)`) for non-Gaussian tails — same node budget
# that fails globally now resolves the peak. We compare both to a *global* 24⁴
# grid, which fails.

# %%
# --- (0) global grid on the narrow field: reuse the section-2 nodes/axes ---
g_narrow = (BETA * narrow_truth.s(grid_pts)).reshape([N_PER_AXIS] * NDIM)
grid_narrow = GridInverseCDF(axes, g_narrow)


# --- (1) Laplace mixture from the diagnostic peaks ---
def laplace_mixture(peaks, beta):
    means = xp.stack([p["mu"] for p in peaks])
    covs = xp.stack([p["cov"] for p in peaks])
    logw = xp.array([beta * p["s_peak"] + 0.5 * xp.linalg.slogdet(p["cov"])[1] for p in peaks])
    w = xp.exp(logw - xp.max(logw))
    return GMMDistribution(w / xp.sum(w), means, covs), logw


laplaceC, peak_logw = laplace_mixture(narrow_peaks, BETA)


# --- (2) whitened local grid per peak + a mixture over them ---
class WhitenedLocalGrid:
    def __init__(self, s_fn, mu, cov, beta, half_width=5.0, n=18):
        self.mu = xp.asarray(mu)
        self.L = xp.linalg.cholesky(cov)  # cov = L Lᵀ ;  θ = μ + L u
        self.logdetL = xp.linalg.slogdet(self.L)[1]
        u_axes = [xp.linspace(-half_width, half_width, n) for _ in range(NDIM)]
        U = xp.stack([m.ravel() for m in xp.meshgrid(*u_axes, indexing="ij")], axis=1)
        theta = self.mu[None, :] + U @ self.L.T
        g = (beta * s_fn(theta)).reshape([n] * NDIM)
        self.grid_u = GridInverseCDF(u_axes, g)

    def logpdf(self, x):
        x = xp.atleast_2d(x)
        u = xp.linalg.solve(self.L, (x - self.mu[None, :]).T).T
        return self.grid_u.logpdf(u) - self.logdetL  # Jacobian |det L|

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        u = self.grid_u.rvs(size)
        return self.mu + u @ self.L.T


class LocalMixture:
    def __init__(self, comps, logw):
        self.comps = comps
        w = xp.exp(logw - xp.max(logw))
        self.w = w / xp.sum(w)
        self.logw = xp.log(self.w)

    def logpdf(self, x):
        lps = xp.stack([c.logpdf(x) for c in self.comps], axis=0)
        return logsumexp(self.logw[:, None] + lps, axis=0)

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        comp = weighted_choice(self.w, (n,), rng)
        out = xp.zeros((n, NDIM))
        for k, c in enumerate(self.comps):
            m = comp == k
            cnt = int(to_np(xp.sum(m)))
            if cnt > 0:
                out[m] = c.rvs(cnt)
        return out.reshape(size + (NDIM,))


whitenedD = LocalMixture(
    [WhitenedLocalGrid(narrow_truth.s, p["mu"], p["cov"], BETA) for p in narrow_peaks],
    peak_logw,
)

# %% [markdown]
# ### Validation on the narrow toy — global grid **fails**, local surrogates **win**

# %%
def log_box_prob(tr, n=200000):
    s = tr.rvs(n)
    inb = xp.all((s >= BOUNDS[:, 0][None, :]) & (s <= BOUNDS[:, 1][None, :]), axis=1)
    return float(np.log(max(float(to_np(xp.mean(inb.astype(xp.float64)))), 1e-12)))


logZ_n = log_box_prob(narrow_truth)
test_n = xp.clip(narrow_truth.rvs(3000), BOUNDS[:, 0][None, :] + 1e-6, BOUNDS[:, 1][None, :] - 1e-6)
true_lp_n = BETA * narrow_truth.s(test_n) - logZ_n

# importance points from the (normalized-over-ℝ⁴) truth cover the narrow peaks
q_s = narrow_truth.rvs(40000)
s_q = narrow_truth.s(q_s)

narrow_methods = {
    "global grid 24^4": grid_narrow,
    "Laplace->GMM": laplaceC,
    "whitened local grid": whitenedD,
}
print("logpdf shape-RMS vs truth (nats, lower=better):")
for name, d in narrow_methods.items():
    print(f"  {name:22s}: {shape_rms(d.logpdf(test_n), true_lp_n):.3f}")
print("\n∫ p dθ via importance sampling (should be ≈ 1):")
for name, d in narrow_methods.items():
    integ = float(to_np(xp.mean(xp.exp(d.logpdf(q_s) - s_q))))
    print(f"  {name:22s}: {integ:.3f}")

# %% [markdown]
# **Reading the result.** The diagnostic's `R = range/σ` and `N_total` tell you,
# from a few F-stat evals, whether a global grid is feasible *before* you build
# anything. If narrow: `Laplace→GMM` (→ `FullGaussianMixtureModel`) is the
# analytic primary; `whitened local grid` adds tail accuracy at the same node
# budget; both crush the global 24⁴ grid, which cannot resolve a σ≈0.02 peak with
# dx≈0.043. Remember to **temper/inflate** (`β<1`, or `Σ←c·Σ`) for MCMC acceptance.
#
# **On the real F-stat**, run exactly this:
# ```python
# def s_fn(theta):                       # theta: (n,4) = (f0, fdot, lam, beta)
#     N, M = comp.get_fstat_ll_wdm(theta, wdm_holder)      # (n,4), (n,10)
#     M4 = unpack_upper_triangle(M)                        # (n,4,4) symmetric
#     return 0.5 * xp.einsum("bi,bij->bj", N, xp.linalg.inv(M4)).__mul__(N).sum(-1)
# peak_width_report(s_fn, beta=1.0)      # -> narrow-or-not verdict for YOUR data
# ```
