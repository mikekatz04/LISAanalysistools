# F-statistic proposal — step-by-step walkthrough (2026-07-15)

> **The polished tutorial + final validation record lives at
> [`docs/tutorials/fstat-proposal.md`](../../docs/tutorials/fstat-proposal.md)**
> (committed figures under `docs/tutorials/img/`). This file is the raw
> session walkthrough. Rendered artifact:
> https://claude.ai/code/artifact/6067fd55-a952-4687-8574-e5b19d643d5a

Test configuration for the real-data runs: `erebor.gb_no_fg` in mojito mode —
**GB-injection-only data (no noise realization)**, whitened by the **tabulated
empirical PSD** from the mojito NOISE brick
(`MojitoNoiseEstimates`, extras-only fixed-PSD path:
`fixed_psd_kwargs=dict(psd_params=None, galfor_params=None)` +
`sensitivity_init_kwargs["extra_components"]`).

## Pipeline

```
WDM residual + empirical PSD
  └─► get_fstat_ll_wdm(θ batch) → N (n,4), M (n,10)          [chunked_het.py]
       └─► F = ½ NᵀM⁻¹N                                       [compute_fstat]
            └─► g = β·F on a GridSpec box                     [FStatProposal4D]
                 └─► corner-averaged cell CDF (log Z)
                      └─► rvs(size) / logpdf(x)               [eryn duck-typed]
                           └─► ProbDistContainer{("f0","Mc","alpha","sin_delta"): prop}
```

## Steps and their verification status

### 1. Score any point with the F-statistic — VERIFIED

`lisatools/chunked_het.py::get_fstat_ll_wdm` builds the 4 Cornish–Crowder
basis filters per candidate at its (f0, fdot, sky) and returns `N` (data–filter
inner products) and `M` (filter Gram, upper triangle);
`lisatools/sampling/fstat_proposal.py::compute_fstat` forms `F = ½NᵀM⁻¹N` —
the likelihood analytically maximized over the 4 extrinsics (A, ι, ψ, φ0).

Checked (`tests/test_fstat_proposal.py`): identity-Gram exact; random SPD Gram
vs direct inverse to 1e-6; fully singular Gram returns finite/−inf (ridge
solve), never raises.

### 2. Chirp-mass sampling basis — VERIFIED (2 bugs fixed)

Sampling axes `(f0 [mHz], Mc [M☉], α, sin δ)` match the new stock GB basis
(`GBSettings.use_chirp_mass=True` default); grid points map to physical rows
via `fdot = get_fdot(f0, Mc)`.

Checked: forward transform exact on 1000 prior draws; full inverse round-trip
(`both_inverse_transforms`) to 4e-15; priors finite; container pickles.

Bugs found & fixed en route:
- **gbgpu `get_fdot` mutated the caller's `Mc` array in place**
  (`Mc *= MSUN_SI` — converted your chirp masses to kg). Now `Mc = Mc * MSUN_SI`.
- gb.py's phi0 sign flip used local lambdas → built transform container was
  unpicklable. Replaced with named `transforms.negate`.

### 3. Grid sweep → density — VERIFIED

`p(θ) ∝ exp(β·F)` piecewise-constant on cells; β tempers/sharpens. Cell
weights are the **corner-averaged (trapezoid)** target over the 16 surrounding
nodes — chosen because lower-corner cells biased every sample mean by exactly
+dx/2 (measured, then removed):

| axis      | injected | mean (lower-corner) | mean (trapezoid) |
|-----------|----------|---------------------|------------------|
| f0 [mHz]  | 20.3804  | 20.4232 (+dx/2)     | 20.3803          |
| Mc [M☉]   | 0.5192   | 0.5233  (+dx/2)     | 0.5200           |
| α [rad]   | 4.0617   | 4.0736  (+dx/2)     | 4.0608           |
| sin δ     | −0.7864  | −0.7830 (+dx/2)     | −0.7863          |

### 4. Resample: `rvs()` — VERIFIED

Exact inverse-CDF draws (cumsum + searchsorted + in-cell jitter). Mock
validation (known 4-D Gaussian encoded into (N, M) at the real highest-f GB's
parameters): sample moments to 0.1σ / 10%; grid marginals and rvs histograms
overlap on every axis. Plot: `../../docs/tutorials/img/mock_zoom.png`
(wide-box variant regenerable via
`plot_fstat_proposal_mock_highest_gb.py`).

Note on the wide plot's look: it is a **completed** mock result, not a pending
one. The f0 marginal is blocky because the mock's σ_f0 (0.75 mHz) ≈ one grid
cell (0.77 mHz) — you are seeing the resolution limit of a coarse "locate"
grid; and the sin δ marginal legitimately rides the box edge (clipped at −1 /
+0.16 around the source). A half-cell `imshow` extent bug (white gap strip at
panel edges) was found here and fixed in both plot scripts.

### 5. Score back: `logpdf()` — VERIFIED

Reads the same corner-averaged cell weight rvs samples from, minus log Z; −∞
outside the box. Checked: MC ∫exp(logpdf)dθ = 1 within 5%; mean −logpdf of
draws matches the cell-distribution entropy to 2%; eryn shapes
(`rvs(size)→size+(4,)`, `logpdf((n,4))→(n,)`); pickled copies keep the fitted
grid (kernel/data handles dropped per sprint rule).

### 6. Find real sources per sub-band — IN PROGRESS (design pivoted)

Runner: `scripts/fstat_proposal/plot_fstat_proposal_mojito.py`
(`FSTAT_TARGET=highest|band75`).

**The coarse→zoom first attempt is the wrong tool here — measured, not
assumed.** With 90 d of data the F-stat f0 peaks are ~1/T ≈ 1.3e-4 mHz wide;
a 16-node f0 grid over a 12-layer band has cells ~0.09 mHz — the grid almost
surely never lands within a correlation length of any source (the plan §7
narrow-peak regime). The coarse stage-1 sweep is kept as the negative control.

**Pivot (what "close to the posterior" needs):**

1. Over a 90-day stretch, `fdot·T ≈ 1e-8 Hz ≪ 1/T` → **Mc is nearly
   unmeasurable per band**; hold it fixed during the scan.
2. **Dense-in-f0 comb scan**: f0 nodes every ~1/(2T) across the sub-band
   (≈2100 nodes for a 0.139 mHz band interior) × a handful of sky points
   (the f0–sky Doppler ridge smears peaks by ±f0·v/c ≈ ±2e-3 mHz) — this
   should reproduce the catalogue comb: ~15 tight peaks in [7.500, 7.639] mHz,
   one isolated peak at 20.3804 mHz.
3. **Local 4D proposal per top peak** (dense f0 box ± Doppler width × sky ×
   Mc) → the posterior-shaped object; Laplace/Fisher-Gaussian per peak
   (`chunked_het.information_matrix`, ~40 kernel evals per peak) is the
   documented next escalation (plan §7: Laplace → `FullGaussianMixtureModel`).

Kernel cost note: `get_fstat_ll_wdm` ≈ 170 ms/eval single-threaded on this
laptop (90 d, Nt=2160) — grids are budgeted around that; on GPU the same sweep
is one batched launch.

Targets:
- `highest`: ID 7725228, f0=20.380377 mHz — isolated (nearest neighbor
  0.71 mHz away).
- `band75`: stock band [7.36, 7.78] mHz (interior [7.500, 7.639]); loudest
  in-band source ID 1229636, f0=7.580260 mHz, A=9.07e-23.

### Known caveats

- **Interacting catalogue Mc trap**: the mojito DWD catalogue is an
  interacting population. Highest-f GB: mass-based Mc = 0.5192 M☉, but the
  injected (f0, fdot) implies **Mc_eff = 0.4658 M☉** — the grid maps Mc→fdot
  via the GW-only relation, so the F-stat must peak at Mc_eff. Plots mark
  both. (For the 7.5 mHz band sources the two agree to ~1e-4 — GW-driven.)
- **Missing module from the other machine**: the "progress on fstat" commit
  referenced `lisatools.sampling.fstat_proposal` (reconstructed here) and
  `lisatools.sampling.f0_mchirp_prior` (`F0McGMMSampling.from_heatmap`, fit in
  `heatmap_GMMs.ipynb`) — the latter is still only on the other machine.
  Only reachable behind `GB_F0MC_GMM_PRIOR=1` (default off); push it or the
  flag ImportErrors.
- Per-band Mc flatness means the Mc axis of a per-band proposal is
  prior-dominated — expected, not a bug.

### Repro

```sh
# hermetic unit tests (11)
python -m unittest tests.test_fstat_proposal

# mock validation corner plots
python scripts/fstat_proposal/plot_fstat_proposal_mock_highest_gb.py

# real mojito run (empirical PSD, no added noise)
OMP_NUM_THREADS=1 FSTAT_TARGET=band75 \
  python scripts/fstat_proposal/plot_fstat_proposal_mojito.py
```
