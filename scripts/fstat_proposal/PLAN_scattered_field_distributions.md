# Plan — 3D/4D distributions from a scatter-cloud with a scalar statistic

**Goal.** Turn a *cloud of parameter points* `θ_i ∈ ℝ^d` (`d ∈ {3, 4}`), each
carrying a *scalar statistic* `s_i = s(θ_i)` (an F-statistic / log-likelihood /
detection statistic), into a probability **distribution object** that supports

* `rvs(size)` — draw samples, and
* `logpdf(x)` — normalized log density,

and that drops into the `eryn` / `lisatools` `priors.py` machinery (see
§4) and runs on **GPU** in the LAT backend structure.

Pipeline:

```
prior/design draw  ─►  F-stat computation  ─►  distribution(rvs, logpdf)  ─►  eryn proposal/prior
   (we choose θ_i)      (get_fstat_ll_wdm)       (this plan)                   (ProbDistContainer)
```

---

## 0. The one fact that reframes everything: **we control point placement**

There are two very different problems hiding under "cloud of points → density":

| | **(I) Fixed weighted cloud** | **(II) We choose where to evaluate** ← *our case* |
|---|---|---|
| What `s_i` is | an *importance weight* on a draw we could not choose | a *field value* we can query anywhere, cheaply, in huge batches |
| Right tool | density **estimation** from samples (KDE, fit-a-GMM) | **design of experiments + surrogate modeling** of the field `s(θ)` |
| `rvs` quality | limited by where the samples happened to land | limited only by how well we *design* the evaluation set |

Because `get_fstat_ll_wdm` lets us evaluate `s` at **any** batch of `θ` on the
GPU (§5), we are firmly in column (II). That promotes *field-interpolation*
methods (grid + inverse-CDF, sparse/adaptive grids, GP/flow surrogates) to
first class — they give **exact, cheap sampling** of a controllable-accuracy
surrogate — and demotes sample-density-estimation (KDE) to a fallback.

Throughout, the target density is

```
p(θ) ∝ exp( β · s(θ) )       (masked to the design box)
```

with an **inverse-temperature `β`** knob: `β = 1` → likelihood-shaped proposal,
`β < 1` → broadened/tempered proposal (safer for MCMC acceptance), `β > 1` →
peak-sharpened. Everything below carries `β`.

---

## 1. Method menu

Six families, ordered roughly from "simplest / most GPU-native" to "most
flexible / most engineering". Each is scored for our setting.

### A. Structured grid + conditional inverse-CDF  — **recommended baseline**

* **Design.** Tensor-product grid over the box, optionally with per-axis
  coordinate transforms (log in `fdot`, `sin(beta)` for uniform sky-area, etc.).
  Evaluate `s` on the grid → store `g = β·s` (an `N^d` log-density array).
* **`logpdf(x)`.** Multilinear (or multicubic) interpolation of the stored `g`,
  minus `log Z`. `Z` is a one-time grid quadrature of `exp(g)`. O(1) per query,
  pure gather — trivially GPU.
* **`rvs(size)`.** *Exact* conditional inverse-CDF: marginalize `exp(g)` over
  axes `1..d-1` → CDF in `θ_0` → invert with `searchsorted`; then draw
  `θ_1 | θ_0`, `θ_2 | θ_0,θ_1`, … Every step is `cumsum` + `searchsorted` +
  gather. Produces exact draws from the piecewise-multilinear density.
* **GPU / cost.** Ideal — no fitting, no training, no native kernel needed
  (all `cupy` primitives). Storage `N^d`: `d=3` fine to `N≈128`; `d=4`
  comfortable at `N≈48–64` (`64^4` f64 = 134 MB). Coordinate transforms and
  adaptive design (B) buy effective resolution.
* **Pros.** Exact `rvs`, cheap normalized `logpdf`, dead-simple, robust, no
  hyperparameters. Directly consumes designed F-stat evaluations.
* **Cons.** Curse of dimensionality caps resolution at `d=4`; a *narrow*
  isolated peak needs the grid to resolve it → pair with (B) or transforms.
* **Status in tree.** Net-new, but small (pure `cupy`). No ND interpolator
  exists today (only batched-1D `CubicSplineInterpolant`), and none is needed —
  multilinear + `cumsum`/`searchsorted` suffice.

### B. Adaptive / sparse-grid design — *extension of A for peaky d=4*

* **Sparse (Smolyak) grids**: points scale `~ N (log N)^(d-1)` instead of `N^d`;
  excellent for *smooth* fields.
* **Adaptive mesh refinement / active learning**: start coarse, refine cells
  where `exp(g)` mass or `|∇g|`/curvature is large. This is exactly where "we
  control points" pays off — iterate `design ← current surrogate`.
* `rvs`/`logpdf` via a final piecewise (KD-tree of cells) representation with
  per-cell inverse-CDF, or by importance-resampling onto a dense patch around
  the peak.
* **Pros.** Beats the curse of dim; concentrates F-stat calls on the peak.
  **Cons.** More machinery; `rvs` on irregular partitions is fiddlier.

### C. Gaussian-mixture surrogate — **recommended analytic path; ~90% already built**

* **Fit** a `K`-component GMM to the target `exp(β·s)`, two ways:
  * *(c1) resample-then-fit:* importance-resample the designed points with
    weights `w_i ∝ exp(β·s_i)/q(θ_i)` (`q` = design density; uniform grid → `q`
    const → `w_i ∝ exp(β·s_i)`), then fit an unweighted GMM to the resample.
  * *(c2) weighted EM:* run weighted EM directly on the designed points, no
    resample.
* **`logpdf`** = `logsumexp` over components (analytic, cheap, GPU).
  **`rvs`** = pick component `∝ weight`, draw MVN (analytic, cheap, GPU).
* **Reuse (important).** The distribution object already exists and is in
  production:
  * `lisatools.sampling.prior.FullGaussianMixtureModel` (`prior.py:578`) —
    GPU `rvs` + native C/CUDA `compute_logpdf` kernel with per-box spatial
    culling; maps each sub-domain into `[-1,1]^d`; already the GB RJ-birth
    proposal.
  * `lisatools.sampling.gmm.vec_fit_gmm_min_bic` (`gmm.py:2134`) — batched
    GPU EM over `(n_groups, n_samples, n_features)` with per-group min-BIC
    component selection → returns a `FullGaussianMixtureModel`.
  * `GaussianMixtureModel` (`gmm.py:841`) / `GMMFit` (`gmm.py:~1900`) — the
    vectorized GPU EM + `[-1,1]^d` boxing under the hood.
  So Family C's **new work is only the fitter** (F-stat evals → resampled
  points → `vec_fit_gmm_min_bic`), not the distribution class. This is the
  lowest-effort path to a working eryn proposal.
* **Pros.** Compact, analytic, differentiable, multimodal via `K`, GPU-cheap,
  *already wired into eryn*. **Cons.** Assumes locally-Gaussian blobs; curved
  degeneracies / heavy tails need many components; EM needs `K` + init (BIC
  sweep handles `K`).

### D. Weighted KDE — *fixed-cloud fallback; good for irregular designs*

* Kernel per point weighted by `w_i ∝ exp(β·s_i)`:
  `logpdf(θ) = logsumexp_i[ log w_i + logK_H(θ − θ_i) ] − log Σw`.
  `rvs` = pick point `∝ w_i`, add kernel noise.
* Bandwidth `H`: Scott/Silverman global, full-covariance from weighted sample
  cov, or per-point adaptive (k-NN balloon).
* **GPU / cost.** `logpdf` is dense `O(N_query × N_kernel)` — fine for moderate
  `N` (tile it), or reduce kernels via KMeans to `K` representatives (→ becomes
  a fixed-covariance GMM). `rvs` is trivial and exact for the KDE density.
* **Pros.** Trivial to build, non-parametric, no mode assumption. **Cons.**
  `O(N²)` `logpdf`, bandwidth bias, boundary bias, worse variance in 4D. Best
  only when you *cannot* design a grid.
* **Status in tree.** Absent (only seaborn plotting + a commented-out
  `KDEMove`). Net-new.

### E. Normalizing flow / GP-emulator surrogate — *flexible endgame*

* **(e1) Normalizing flow.** Train on resampled points *or* by
  **density-regression** against the known `g_i` (we have log-density *values*,
  so we can fit `log q_flow(θ_i) ≈ g_i − log Z` — a supervised target, richer
  than plain MLE-on-samples). Exact `logpdf` + `rvs` after training; captures
  curved/multimodal geometry; scales past `d=4`. A `FlowDist` stub already
  exists (commented, `prior.py:817`) — it targeted an external `lisaflow`.
* **(e2) GP / neural surrogate of `s(θ)`** itself (regression on the design),
  then sample `exp(β·ŝ)` by rejection / importance-resampling a coarse grid.
  The GP posterior variance *drives adaptive design* (closes the loop with B).
  Needs sparse/inducing-point GP for many points in 4D on GPU.
* **Pros.** Most flexible; best for hard geometries / `d ≥ 4`. **Cons.**
  Heaviest engineering; training + tuning; `rvs` for (e2) not closed-form.

---

## 2. Comparison at a glance

| Family | Build | `logpdf` cost | `rvs` | GPU fit | `d=4` scaling | Handles curved/multimodal | New code |
|---|---|---|---|---|---|---|---|
| **A** grid + inv-CDF | none (just evals) | O(1) gather | **exact** | native `cupy` | `N^d` mem; ok `N≤64` | modest peaks yes; sharp needs B | small, pure `cupy` |
| **B** adaptive/sparse | iterative design | O(log) tree | exact-ish | good | **beats curse** | yes | medium |
| **C** GMM | EM (BIC sweep) | `logsumexp K` | **exact** | **exists** | great | multimodal yes; curved ~ needs K | **fitter only** |
| **D** KDE | trivial | O(N·Nq) | exact | ok (tile) | poor-ish | via kernels | small |
| **E** flow / GP | training | O(1) | exact (flow) | native | **great** | **yes** | large |

---

## 3. Recommended path (phased)

> **First measure the peak width (§7).** If the F-stat peaks are *narrow* in 4D,
> the *global* forms of A/C/D degrade and the path below shifts to **peak-find +
> local surrogate**. Run the §7 diagnostic before committing to a global grid.

1. **Phase 1 — Grid + inverse-CDF (A).** The robust "just works" default for
   `d=3` and smooth `d=4`. Exact `rvs`, cheap normalized `logpdf`, no fitting,
   pure `cupy`. Validate against the toy (notebook) and then against a real
   F-stat slice.
2. **Phase 2 — GMM (C) reusing `FullGaussianMixtureModel`.** Write only the
   F-stat→resample→`vec_fit_gmm_min_bic` fitter. Gives a compact analytic
   proposal that plugs straight into the existing eryn/GB machinery.
3. **Phase 3 — Adaptive design (B).** Layer active refinement on top of A/C for
   narrow 4D peaks; reuse the same F-stat evaluator in the refinement loop.
4. **Phase 4 (optional) — Flow (E).** Revive the `FlowDist` stub for hard
   geometries / higher-d, trained by density-regression on the F-stat values.

Use **A** and **C** as the two production workhorses; **D** only when a grid
design is impossible; **B/E** as accuracy/scale escalations.

---

## 4. Interface / architecture (matching `priors.py`)

`eryn`'s distributions are **duck-typed** — no base class; an object just needs
`rvs(size)` and `logpdf(x)` (`Eryn/src/eryn/prior.py`). `ProbDistContainer`
aggregates a dict `{key: dist}`:

* a **tuple key** `(0,1,2,3)` maps *one joint distribution* to those columns —
  this is how a multivariate density registers (exactly our case);
* `logpdf(x)` receives `x` of shape `(n, ndim)` and the joint dist gets the
  slice `x[:, inds]` of shape `(n, len(inds))`, returning `(n,)`;
* `rvs(size)` for a joint dist must return `size + (len(inds),)`;
* every child honors `use_cupy` / `return_gpu` (container sets `return_gpu=True`
  and does the final `.get()`).

Proposed classes (all `LISAToolsParallelModule` subclasses → `self.backend`,
`self.xp`, `force_backend`; **no backend kwargs on methods** per sprint rule;
**no module stored as attribute** per deepcopy/pickle rule — expose `xp` as a
property):

```
ScatteredFieldDistribution(LISAToolsParallelModule)          # abstract
  ├─ GridInverseCDFDistribution        (Family A)
  ├─ FullGaussianMixtureModel  (REUSE) (Family C)  + FStatGMMFitter
  ├─ WeightedKDEDistribution           (Family D)
  └─ FlowDistribution                  (Family E, later)

  common:
    __init__(points, values, *, beta=1.0, bounds=..., transforms=..., use_cupy=..., return_gpu=...)
    classmethod from_field(evaluator_fn, bounds, design=..., beta=1.0)   # calls the F-stat evaluator
    rvs(size=1)  -> size + (ndim,)
    logpdf(x)    -> (n,)           # x is (n, ndim)
    to_prob_dist_container(param_indices)  -> {tuple(param_indices): self}
```

Register into a sampler prior with e.g.
`ProbDistContainer({(idx_f0, idx_fdot, idx_lam, idx_beta): dist, ...other params...})`.

---

## 5. Computing the statistic from the **F-statistic** (`gb_wdm` / `gb_fd`)

This is the concrete driver. The scalar `s` is the GB **F-statistic**, already
available on GPU in the WDM chunked-het infrastructure:

* **`lisatools.chunked_het.get_fstat_ll_wdm(params, wdm_holder, ...)`**
  (`chunked_het.py:952`) builds the 4 Cornish–Crowder basis filters per binary
  at its `(f0, fdot, lam, beta)` and returns, **batched over many binaries**:
  * `N` — shape `(num_bin, 4)`, the `<d | A_i>`;
  * `M` — shape `(num_bin, 10)`, the upper-triangle `<A_i | A_j>`.
* Form the statistic on the Python side:
  `F = ½ · Nᵀ M⁻¹ N`  (batched `xp.linalg.solve` / `inv`).
* Backed by the CUDA kernel `wdm_het_get_fstat_ll_kernel`
  (`cutils/lat_chunked_het_kernels.hh:2296`), so thousands of design points cost
  one batched launch.

Consequences that shape the distribution:

* **The 4 extrinsic amplitude params `(A, ι, ψ, φ0)` are analytically
  maximized** inside the F-stat. So `F` is a function of the **4 intrinsic
  params `(f0, fdot, lam, beta)` only** — *this is the 4D cloud*. (A 3D variant
  fixes/marginalizes `fdot`, e.g. `(f0, lam, beta)`.)
* The resulting proposal is over **intrinsics only**; extrinsics are drawn
  separately, or reconstructed from the ML amplitudes implied by `(N, M)`.
* `exp(F)` carries the F-stat's implicit **flat-in-amplitude-basis** prior —
  document this so the proposal convention is explicit. Use `β<1` if `exp(F)`
  is too peaked for healthy MCMC acceptance.

End-to-end for the recommended path (A):

```
design (f0,fdot,lam,beta) on a grid/adaptive set
   └─► get_fstat_ll_wdm(design, wdm_holder)  → N, M     (GPU, batched)
        └─► F = 0.5 * Nᵀ M⁻¹ N                          (GPU)
             └─► GridInverseCDFDistribution(design_grid, F, beta=T)
                  └─► rvs / logpdf  ─►  ProbDistContainer{(i_f0,i_fdot,i_lam,i_beta): dist}
```

The same `F` array feeds Family C instead (resample ∝ `exp(βF)` →
`vec_fit_gmm_min_bic` → `FullGaussianMixtureModel`) with zero new distribution
code.

---

## 6. Notebook

`illustrate_scattered_field_distributions.ipynb` (same folder) implements, in
**`xp`-agnostic** form (numpy now, cupy drop-in) so it transplants onto GPU:

* a toy 4D "F-stat-like" field (dominant narrow peak + a curved sky ridge +
  broad background) with a **known** target so every method is validated;
* controllable **design of experiments** (uniform grid vs adaptive/importance);
* **Family A** (grid + conditional inverse-CDF), **Family C** (hand-rolled
  weighted-EM GMM mirroring `FullGaussianMixtureModel`), **Family D**
  (weighted KDE);
* validation of each: `logpdf` vs the true field (up to a constant),
  `rvs` marginals vs truth, and a numerical normalization check;
* a `priors.py`-shaped wrapper + a mock `ProbDistContainer` round-trip;
* notes on the `xp`→`cupy` swap and where a native kernel would earn its keep.

---

## 7. Narrow 4D peaks? **Measure first**, then branch

Don't assume the regime — *check it*, cheaply, from the F-stat's own local
curvature. Near a peak the amplitude-maximized statistic is quadratic:

```
2F(θ) ≈ 2F_peak − (θ − θ_peak)ᵀ Γ (θ − θ_peak),   Γ = intrinsic Fisher metric
```

so a handful of extra (batched) F-stat evals around a candidate gives you `Γ`
by finite difference — and `Γ` *is* the peak shape.

### 7.1 The diagnostic (run this before choosing a method)

For each candidate peak (from a coarse scan / your search bank):

1. **Refine** to the local max with a few Newton steps (`θ ← θ − H⁻¹∇`), all
   from batched F-stat evals.
2. **Curvature** `H = ∇²(β·s)` by finite difference → local covariance
   `Σ = (−H)⁻¹`; per-axis width `σ_i = √Σ_ii`; correlation matrix from `Σ`.
3. **Resolution elements** per axis `R_i = L_i / σ_i` (`L_i` = prior range).
   Global grid needs `N_i ≈ c·R_i` nodes/axis (`c ≈ 3`), so
   `N_total ≈ ∏ c·R_i`.
4. **Search difficulty (template count)** `≈ √det(β·(−H)) · Vol_box` — the
   number of ~1-mismatch cells a blind search must cover.
5. **Modes**: count distinct refined peaks (dedup by separation ≫ σ).

**Branch rule:**

| Diagnostic says | Regime | Use |
|---|---|---|
| `N_total ≲ 10⁷` (e.g. `R_i ≲ 30`/axis) | *not narrow* | **global grid A** (or global GMM C) — done |
| `N_total ≫ 10⁷` (large `R_i`, esp. in `f0`/`fdot`) | **narrow** | **peak-find + local surrogate** (7.2) |
| many distinct peaks | multimodal | mixture over local models (7.2.4) |

The notebook implements this as `peak_width_report(s_fn, bounds, β)` and runs it
on a broad *and* a narrow toy so you can see the numbers for each; point it at
the real F-stat (`s_fn = θ ↦ ½NᵀM⁻¹N` from `get_fstat_ll_wdm`) to settle the
question for your data.

### 7.2 If it *is* narrow — peak-find + local surrogate

Global dense grids fail (`N⁴ ≈ (L/σ)⁴` nodes, mostly empty; a coarse search grid
steps *over* the peak) and global KDE / resample-GMM fail (importance weights
`∝ exp(β·s)` collapse the effective sample size). Instead exploit that the F-stat
hands you each peak's location + curvature:

1. **Laplace (Fisher-Gaussian) per peak — recommended.** From the diagnostic you
   already have `(μ_k, Σ_k)`. Place a Gaussian per peak, weight by the Laplace
   evidence `w_k ∝ exp(β·F_peak,k)·det(Σ_k)^{1/2}`, sum into a GMM → feed the
   existing `FullGaussianMixtureModel`. A narrow peak *is* a tight Gaussian, so
   this is near-exact, analytic, GPU-cheap. **This is the primary path.**
2. **Whitened local grid (Family A applied *locally*) for non-Gaussian tails.**
   Build a small grid per peak in whitened coords `u = Lᵀ(θ−μ)` (`Σ⁻¹=LLᵀ`):
   the peak is isotropic and O(1)-wide in `u`, so `~20⁴` over `±5σ` resolves it
   with the *same* node budget that failed globally. `rvs`/`logpdf` via inverse-
   CDF in `u` + the constant linear Jacobian `−log|det L|`.
3. **Tempering / inflation matters more.** A narrow proposal that is slightly
   mis-located or too tight has ~0 acceptance. Use `β<1` and/or inflate
   `Σ ← c·Σ` (`c≈2–4`, or the `2.38²/d` rule). Better too wide than missing.
4. **Mixture + background floor.** `p = Σ_k w_k·(local_k) + ε·prior` keeps
   `logpdf` finite everywhere (no `−inf` where the sampler wanders).
5. **Adaptive refinement (Family B)** closes the loop when peaks are hard to
   find: fit → draw → batch-evaluate F-stat → add high-F points / new peaks →
   refit.

**Net for the narrow regime:** drop *global* A; use **Laplace-Gaussian-per-peak
→ `FullGaussianMixtureModel`**, add **whitened local grids** only where tails
are non-Gaussian, peaks found by a **Fisher-metric-spaced scan + adaptive
refinement**. All the curvature you need is a few batched F-stat evals per
candidate.
