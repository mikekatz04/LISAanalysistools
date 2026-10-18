# EMRI sampling audit and adjustments (2026-10-18)

User ruling (Mike, 2026-10-18): "over the next few hours, do a fully audited and
adjusted EMRI setup to get this to be better. Please take a look." On the
stretch move: "we do not have enough walkers for the stretch move, so we need
to improve what we have."

Data: the 9mo flagship, job 773, store rows 840-1007 (`full_pe`, the first job
with the pooled / tempered / factor-capped eigen adaptation of 2026-10-15),
its log `slurm_stdout_773.log` and the eigen-table sidecar
`gf_prod_9mo_testing_eigen_tables.pkl` (head = walker 0's table only: the
sidecar has a single writer). 4 walkers, one per compute rank, 8 EMRI leaves,
8 rungs per leaf (make_ladder, T_max 6), 25 repeats per visit, one visit per
leaf per row. Everything below was measured on CPU from that extract; nothing
was run on a GPU.

## The finding that changes the picture: the posteriors are combs

The "stuck along the ridge" description of 2026-10-14/15 is wrong in one
important way. The loud EMRIs do not sit on a continuous ridge they fail to
traverse. Their posteriors are a COMB of discrete islands along the intrinsic
degeneracy. Histogramming every (row, rung, walker) sample of rows 840-1007 in
p0 gives disjoint islands with empty gaps between them:

| EMRI 3 island (p0) | samples | held by | a | logm1 | e0 | max lnL - cold |
|---|---|---|---|---|---|---|
| 12.3305-12.3308 | 1344 | walker 3 (all rungs) | 0.25944 | 14.2714 | 0.29992 | +9.5 |
| 12.3309-12.3311 | 2520 | walkers 0, 1 | 0.25951 | 14.2713 | 0.29993 | +14.3 |
| 12.3328-12.3330 | 504 | walker 2 | 0.25996 | 14.2711 | 0.30000 | +8.2 |
| 12.3345-12.3348 | 840 | walker 2 | 0.26036 | 14.2708 | 0.30005 | +7.4 |
| 12.3371 | 168 | walker 0 (rungs 2-7) | 0.26092 | 14.2705 | 0.30013 | -2.2 |

("max lnL - cold" is the best sample's lnL minus the same row's cold rung,
on its own walker.) The islands sit within ~10 nats of each other; the
spacing is ~1.8e-3 in p0 with a, logm1, m2, e0 and the phases shifting
together, the signature of a cycle-count degeneracy. Walker 0's hot rungs do
not roam the ridge: of its 1344 rung samples none lies between p0 12.3311
and 12.3369. They hold two islands and swap states between them. EMRI 0 shows
the same structure: seven islands in p0 13.5708-13.5797, each with its own
(e0, Phi_r0) (e0 0.005-0.029, Phi_r0 2.1-4.0), held by different walkers.

History (all rows of the extract): until row ~120, on the old ladder, walker
0's cold chain sat in the 12.3371 island. Its ladder has kept a copy of that
island ever since. Walker 1 lost its copy at row ~400, the T_max 6 ladder
reset. Since then no walker has discovered any island it did not already
hold. In `full_pe` nothing exchanges states between walkers:

* the walker-permuting tempering swap needs two walkers on one body, and the
  compute rank's move holds one (`nwalkers 1` in the sidecar), so
  `EMRI_PERMUTE_EVERY=10` is inert under the 4-rank layout;
* the cold -> hot rotation runs in search stages only (no `cold->hot` line in
  job 773).

A walker can therefore only find another island through its own ladder, and
no single-axis Gaussian step, at any width, connects two islands.

## Audit table

| # | Item | Finding | Evidence (job 773, rows 840-1007) | Decision | Knob |
|---|---|---|---|---|---|
| 1 | Ladder (8 rungs, make_ladder T_max 6) | right | swap acceptance per pair 0.60-0.70, flat over the pairs on every leaf (leaf 3: 0.66 0.66 0.67 0.66 0.68 0.69 0.69); round trip ~150 swap sweeps = ~6 visits; hot-rung deficits on the tempered-Gaussian expectation (ndim/2)(T-1); ladder drift over the window <= 0.0095 in beta | keep | `EMRI_NTEMPS=8`, `EMRI_LADDER_TMAX=6` |
| 2 | Eigen table (Gram at walker_max, refresh 10 visits) | directions right locally, but no axis carries the inter-island vector; the table difference between walkers does not explain walker 0 | the full-space PC1 of the chain is the sky / phase degeneracy (qK, qS, Phi_phi0), \|cos\| 0.81-0.99 with axis 0; the intrinsic PC1 = the inter-island vector (whitened a +0.96, e0 +0.18, logm1 -0.15, m2 -0.11) has \|cos\| 0.52 / 0.63 / 0.55 with the intrinsic axes 7 / 8 / 9, and the island spacing is 60-800x their Fisher sigmas; Gram build 1.2 s median (0.98-3.8) | keep the table; add the cross-walker jump (B3) | `EMRI_EIGEN_INFO=gram`, `EMRI_EIGEN_REFRESH=10` |
| 3 | Adaptation (pooled, x100 cap) | wrong on the narrow intrinsic axes: it measures the BETWEEN-island offsets, and those axes are dead | capped axes (x14-100) are always the intrinsic combinations (logm1 / p0 / m2 / a / e0, axes 7-11), never a phase or the sky; the cold within-walker width / adapted step on them: EMRI 3 axes 8-11 0.17 / 0.09 / 0.09 / 0.09, EMRI 0 axes 9-11 0.04 / 0.21 / 0.22, EMRI 7 axes 10-11 0.33; the between-walker offset there is 7.8-11 adapted steps on EMRI 3 (39-1113 Fisher sigmas) | add the acceptance controller (B2); keep the cap | `EMRI_EIGEN_AXIS_ACC_TUNE` |
| 4 | Jump mix (0.1 x 3) | cannot cross islands (one axis, 3 sigma); on the dead axes its draws are 15-75 local sigmas | per-axis acceptance was not logged before B1 | keep; read the new `jump-mix draws` field | `EMRI_EIGEN_JUMP_MIX` |
| 5 | Repeats (25 / visit, 1 visit / row) | right | within-island lag-1 autocorrelation of the cold chain ~0 at the row cadence on every leaf; cold acceptance 0.20-0.42 per leaf; a visit costs ~200 rows | keep | `EMRI_NUM_PROP_REPEATS=25` |
| 6 | Lost-rung guard (50 nats) | re-anchors legitimate top-rung exploration | 27 firings, all rungs 6-7, deficits 72-100 nats: the posterior's own tail (rung-7 q99 70-85, medians 24-30 against the expected 30); leaf 2 14, 1 4, 5 4, 3 2, 4 2, 7 1, on leaves whose lost deficit SNR^2/2 is 82 / 205 / 118 / 1262 / 318 / 161 | raise to 100 in the launchers (B4) | `EMRI_LADDER_REANCHOR_NATS=100` |
| 7 | Periodic parameters | right | the recipe stamps the sampler's `PeriodicContainer` (phiS, phiK, Phi_phi0, Phi_r0 = columns 7, 9, 10, 11) on every rank's moves (`prepare_compute` runs `setup_run`); the eigen move wraps its proposal; the adaptation wraps deviations before projecting; the axis caps use the 2 pi box | keep; the new jump wraps its differences (tested) | none |
| 8 | xI0 fill + prior domain gate | inexact for a retrograde leaf in principle, inert in practice | mixed population [-1, 1]; leaves 1, 2, 5 are certainly prograde (retrograde invalid there); at every leaf's posterior both gates pass (prograde u 0.08-0.90, retrograde u 0.74-0.88 where valid), so the gate's u <= 1 edge is >= 0.1 in u away (p0 shifts of order 1, ~1000 posterior widths) | keep; the per-leaf gate stays the TODO in `EMRIKerrDomainPrior` | none |
| 9 | Likelihood path (direct WDM) | right; cost per visit is per row, not per call | 4412 `[EMRI_DIRECT]` lines, 0 fallbacks, 0 rows refused; ~200-209 rows / 27 chunks per visit, 0.019-0.026 s/row (leaf 3, plunge in the window: 0.067 s/row, 14 s / visit); B2 / B3 add no rows | keep | none |
| 10a | Cross-walker exchange in PE | missing | see "combs" above: the walker-permuting swap cannot fire at one walker per rank, the rotation is search-only | add the cross-walker jump (B3) | `EMRI_DE_JUMP_MIX=0.2` |
| 10b | Per-axis acceptance | invisible | nothing logged the eigen move's acceptance by axis | add the accounting (B1) | `EMRI_EIGEN_AXIS_ACC_LOG=1` |
| 10c | Adaptation in cold units on a comb | under-estimates when the hot rungs sit in islands (not tempered-wide) | EMRI 5 axes 9-11: the walkers' own cold spread is 1.7-2.6x the adapted step (x x sqrt(beta) shrinks hot-rung deviations that are not wider) | open: the controller may not widen past the adapted width (by design); raising its ceiling to the factor cap is an experiment | none |

## The items in detail

### 1. Ladder

Swap acceptance per adjacent pair, rows 840-1007 (deltas of the cumulative
counters, 67,200 swaps per pair per leaf):

| leaf (SNR) | pairs (0,1) .. (6,7) | round trip |
|---|---|---|
| 0 (21.6) | 0.62 0.64 0.67 0.68 0.68 0.70 0.70 | 5.8 visits |
| 1 (20.3) | 0.68 0.67 0.66 0.66 0.65 0.65 0.66 | 5.9 |
| 2 (12.8) | 0.65 0.63 0.63 0.63 0.62 0.60 0.61 | 6.3 |
| 3 (50.2) | 0.66 0.66 0.67 0.66 0.68 0.69 0.69 | 5.8 |
| 4 (25.2) | 0.67 0.67 0.67 0.66 0.67 0.66 0.65 | 5.9 |
| 5 (15.4) | 0.66 0.65 0.65 0.64 0.65 0.67 0.66 | 6.0 |
| 6 (34.3) | 0.64 0.64 0.67 0.66 0.67 0.67 0.66 | 6.0 |
| 7 (17.9) | 0.67 0.67 0.67 0.66 0.66 0.66 0.69 | 5.9 |

Round trip = 2 N^2 / a for a nearest-neighbour walk over N = 7 pairs at
acceptance a, in swap sweeps (25 per visit). This is an upper bound, since
eryn's top-down sweep can carry a state several rungs in one sweep. Rungs 1-2
are not too close: their pair accepts like every other. 10-12 rungs at the
same T_max would push the per-pair acceptance up to ~0.75, but the round trip
grows as N^2 / a (11 pairs: ~13 visits), and every rung costs 25 rows per
visit. A higher T_max does not help either. At T = 6 the top rung of the
faintest leaf (2, SNR 12.8) is already partly prior-dominated: its rung-7
deficit median is 39.8 against the expected 30, and its lost level is 82. On
the loud leaves the hot rungs already hold more than one island but cannot
build a bridge. Tempering is not the bottleneck. **No change** (B5).

### 2. The eigen table

The head's tables (walker 0) were built at its cold point (EMRI 3: x0 p0
12.33101, its island). In the prior-box-whitened basis the axes are
orthonormal (|A^T A - I| <= 9e-16). Axes are sorted by curvature: 0-6 are
the sky / distance / phase combinations, 7-11 the intrinsic ones, with 11
(p0 + 0.29 logm1) the best measured. The chain's dominant spread (the
full-space PC1, whitened) is the sky / phase degeneracy, and axis 0 carries
it (|cos| 0.99 for walker 2's cold rung, 0.95 for walker 0's rung 7). The
intrinsic PC1 is the inter-island vector itself (the island medians and the
PC1 agree to 2 decimals). No axis carries it. It splits over axes 7 / 8 / 9
(|cos| 0.52 / 0.63 / 0.55), and the island spacing (4.7e-4 whitened) is
60-800x those axes' Fisher sigmas.

Why do walker 0's rungs "span the ridge" while walkers 1 and 3 do not, from
the same cold point? Not because of the tables. Each rank builds its own
walker_max table at its own walker's cold point. Walkers 0 and 1 share an
island, and walker 3 sits 3e-4 away in p0, so the local information matrices
are the same up to each walker's noise model. Only the head's table is
persisted, so the other ranks' tables cannot be read back. Their adaptation
lines in the log are near-identical, though (median x7.1-12.8, the same
axis 11 at the cap). The difference is history: walker 0's ladder kept an
island its cold chain had held at rows 0-119 (above).

The refresh cadence (10 visits, 1.2 s per Gram build) is fine: the local
Fisher is the same on every island, and a frozen table is valid MH. **Keep.**

### 3. The step-scale adaptation

Per leaf, the axes at x >= 10 and the walkers' own cold spread along them
(median over walkers) in units of the adapted step:

| leaf | widened axes (factor) | composition (whitened, top 2) | own width / adapted step |
|---|---|---|---|
| 0 | 8 (x40), 9 (x100), 10 (x96), 11 (x100) | a; logm1-p0; m2-p0; p0-logm1 | 0.65, 0.04, 0.21, 0.22 |
| 1 | 10 (x35), 11 (x14) | logm1-m2; p0-logm1 | 0.39, 0.60 |
| 2 | 10 (x21), 11 (x34) | logm1-p0; p0-logm1 | 1.20, 1.14 |
| 3 | 8 (x14), 9 (x50), 10 (x69), 11 (x100) | a-e0; m2-a; logm1-p0; p0-logm1 | 0.17, 0.09, 0.09, 0.09 |
| 4 | 9 (x15), 10 (x40), 11 (x38) | m2-logm1; m2-logm1; p0-logm1 | 0.48, 0.74, 0.62 |
| 5 | 7 (x36), 10 (x15) | e0; logm1 | 1.12, 2.63 |
| 6 | 10 (x34), 11 (x36) | logm1-p0; p0-logm1 | 0.92, 0.82 |
| 7 | 8 (x100), 9 (x78), 10 (x100), 11 (x67) | m2-logm1; e0-m2; logm1-p0; p0-logm1 | 1.29, 0.64, 0.33, 0.33 |

Every capped axis is an intrinsic combination. None is a phase or the sky.
The pooled estimator centres each rung on its median over the UNION of
walkers. With 4 walkers on 3-4 islands, the MAD then measures the offset
between islands as it projects onto each narrow axis. EMRI 3's between-walker
offset is 7.8-11 adapted steps on axes 7-11. A step 5-25 local sigmas long is
never accepted, so 4 of EMRI 3's 12 axes (3 of EMRI 0's) waste their draws.
The x100 cap is not the problem; it just stops the loss at 100. The fix is to
read the acceptance (B1) and pull a dead axis back (B2). The pooled
adaptation stays, because on EMRI 5 / 6 and the phase axes it is right.

### 4. Jump mix (10 % of the draws at 3x)

It cannot jump islands: the hop is a specific 12-D vector, not a 3-sigma
excursion along one axis. On the dead axes a 3x draw is 15-75 local sigmas.
Its acceptance was invisible before B1, which now logs it separately (`jump-mix
draws cold a/b hot c/d`) and keeps it out of the per-axis counts. **Keep**
(cheap, and it helps on the healthy axes); read the field on the next run.

### 5. Repeats and visits

25 repeats per visit, one visit per row. The cold chain's lag-1
autocorrelation along its leading intrinsic PC is ~0 at the row cadence on
every leaf, so within an island 25 repeats decorrelate. More repeats would not
cross islands; fewer would save ~4 s per leaf visit. **Keep.**

### 6. The lost-rung guard

Hot-rung deficits `lnL[cold] - lnL[k]`, end of visit, 672 samples per leaf and
rung:

| leaf | lost at SNR^2/2 | rung 7: median / q95 / q99 / max (expected 30) | firings |
|---|---|---|---|
| 0 | 230 | 23.9 / 47.3 / 57.5 / 70.1 | 0 |
| 1 | 205 | 30.2 / 59.0 / 76.2 / 101.2 | 4 |
| 2 | 82 | 39.8 / 73.4 / 84.6 / 93.5 | 14 |
| 3 | 1262 | 25.2 / 52.0 / 71.5 / 99.7 | 2 |
| 4 | 318 | 28.9 / 58.0 / 71.9 / 84.3 | 2 |
| 5 | 118 | 30.4 / 61.4 / 74.3 / 99.0 | 4 |
| 6 | 586 | 27.1 / 57.0 / 69.5 / 77.6 | 0 |
| 7 | 161 | 27.6 / 50.5 / 69.2 / 87.5 | 1 |

The guard fires at `50 + (ndim/2)(1/beta - 1)` = 80 nats at rung 7. Every
firing was a rung 6-7 state 72-100 nats down. That is the tail of the top
rung's own tempered distribution, where the loud leaves keep the far islands.
It is not a lost rung: losing the source costs 118-1262 nats on those leaves.
**Raised to 100** (130 at the top rung; B4). It still catches a lost rung of
every EMRI above SNR 16. The two faintest (SNR 12.8 / 15.4, lost at 82 / 118
nats) are no longer caught, and at T = 6 their top rungs are prior-dominated
whatever the guard does.

### 7. Periodic parameters

The EMRI periodic spec is name-keyed (`phiS`, `phiK`, `Phi_phi0`, `Phi_r0`).
`run.py` resolves it to columns 7 / 9 / 10 / 11 and wraps it as a
`PeriodicContainer`, on the head (`prepare_main`) and on every compute rank
(`prepare_compute` runs each step's `setup_run`). The container reaches the
eigen move through the combine / add-remove setters. The eigen move wraps its
proposal into [0, 2 pi). The adaptation wraps deviations to [-pi, pi) before
the axis-basis projection. Axis caps use the 2 pi box (a phase step is capped
at one period). No defect found. The new jump wraps its partner differences
to [-P/2, P/2) before scaling and wraps the result (B6 tests). The Gram
stencil's `nudge_inside` also nudges a phase sitting within one step of 0 or
2 pi, which is harmless: it moves the expansion point by one 1e-3 template
step.

### 8. Prograde / retrograde

The population mixes xI0 = -1 and +1, and the shared prior gate uses the
prograde grid. At every leaf's posterior both gates pass where the leaf can
be either (0, 3, 4, 6, 7). Leaves 1, 2, 5 fail the retrograde gate, so they
are prograde. Prograde u at the posteriors is 0.08-0.90 and retrograde u
0.74-0.88. The prograde gate's u <= 1 edge would wrongly cut a retrograde
leaf's upper p0 range, but it sits >= 0.1 in u (order 1 in p0, ~1000
posterior widths) from any chain. Inert in practice. The per-leaf gate stays
the TODO in `EMRIKerrDomainPrior`. Which of 0, 3, 4, 6, 7 are retrograde
needs the catalogue (not in the extract).

### 9. The likelihood path

All 4412 `[EMRI_DIRECT]` visit lines report 0 fallbacks and 0 rows refused by
FEW. A visit is ~200-209 rows in 27 chunks. Generation dominates: 0.019-0.026
s/row, and for leaf 3 (plunge inside the window) 0.067 s/row, 14 s per visit.
Cost is per row, so neither the controller nor the jump adds anything: the
jump replaces the rows of the eigen draw, and its partner pool rides in the
payload the pooled adaptation already ships (shared arrays, ~0 extra bytes).

## The adjustments as built

All code defaults are OFF or 0, so stock runs are bit-identical. That is
pinned twice: the golden stream of `tests/test_eigen_pool_temper.py`, and
`tests/test_emri_sampling_audit.py::TelemetryGoldenTest`, where B1 ON is
also bit-identical. The v9 launchers turn the blocks on (B7).

### B1: per-axis acceptance accounting (`{BRANCH}_EIGEN_AXIS_ACC_LOG`)

`ResidualAddOneRemoveOneMove._note_inner_outcome`, called by `propose_local`
after each split's MH test, tallies proposals and acceptances per table axis
in two bands, the cold rung and the hot rungs. It counts only eigen draws at
the table width: the jump mixture's wide draws and the cross-walker jumps are
tallied apart. The picked axis comes from the LAT
`EigenAxisJumpMixMove.last_pick` (recorded since this change). For the plain
eryn move it is derived from the displacement (`c = A^-1 dy`, periodic
columns wrapped first). It draws no random numbers. Once per table refresh
and rank:

```
[EMRI_EIGEN] leaf 3 (walker 1): axis acceptance over the last 10 visits, cold 0.45 0.41 ... 0.00 (n 14-25 per axis), hot 0.52 ... ; jump-mix draws cold 3/22 hot 30/170
```

The adaptation line also gains `, cold acceptance of the widened axes 8:0.00
9:0.04 ...`.

### B2: the acceptance controller (`{BRANCH}_EIGEN_AXIS_ACC_TUNE`, `_ACC_MIN` 0.1, `_ACC_MAX` 0.5, `_ACC_FACTOR` 2)

`_eigen_axis_tune_table` runs at each refresh, after the adaptation. It keeps
one multiplier per axis per leaf on the table's sigma. The window's cold
proposals along an axis accumulate until there are at least 8. Then the
multiplier is divided by the factor if cold acceptance is below 0.1, and
multiplied by it if above 0.5. The installed sigma is `max(Fisher, min(m x
sigma, ceiling))`, where the ceiling is the adapted sigma, or the box cap
with the adaptation off. The multiplier changes only at a leaf-visit
boundary, so every repeat sweep runs a frozen symmetric kernel, the same
status as the table refresh. Tempered rungs scale the tuned cold sigma. The
multiplier rides in the sidecar entry (`axis_tune`). On adoption it is
divided back out of the carried scale, so a restart does not apply it twice
(tested). Log:

```
[EMRI_EIGEN] leaf 3 (walker 1): acceptance-tuned step: shrunk axes 8,9,10,11 (cold acc 0.00-0.06), widened none (band 0.10-0.50, factor 2); step now x0.50..x1.00 of the adapted sigma
```

### B3: the cross-walker difference jump (`{BRANCH}_DE_JUMP_MIX`, `_DE_JUMP_MODE_FRAC`)

`lisatools.sampling.moves.walkerdiffjump.WalkerDifferenceJumpMove` sits in the
inner-move stack at weight `DE_JUMP_MIX`; the eigen move keeps the rest and is
the jump's fallback. For a row of an ACTIVE walker it proposes `x + gamma (x_a
- x_b)`. The pair is an ordered pair of distinct states drawn uniformly from
the partner pool, the start-of-propose rung ensembles (all rungs) of the
OTHER walker group. `gamma` is 1 with probability `MODE_FRAC` (the island
jump), otherwise `2.38 / sqrt(2 d) x N(1, 0.1)` (0.486 for d = 12). Periodic
differences are wrapped. Rows of inactive walkers take the eigen proposal.

The schedule: walkers split by the parity of their global index, {0, 2} and
{1, 3}. In the k-th distinct global iteration the move proposes in (the
`gf_iteration` stamp; every propose when unstamped), the walkers with `(w + k)
% 2 == 1` jump and the others run their ordinary stack. When the move
proposes every iteration this is `gf_iteration` parity. Under an even source
cadence in a gb_search stage it still alternates, where the bare parity
would pin one group. Like `gf_iteration` it restarts with each job. It never
reads the chain.

The partner pool is taken in `fanout_prepare_state`, before the cold -> hot
rotation edits a hot rung. In single process and the flat replica layout the
body reads it directly. Under the walker-block fan-out the head ships it as
`de_pool` / `de_parity` in the propose payload, and every rank, the head's
own body included, installs it. Arrays are shared with the pooled
adaptation's `eigen_pool`.

Exactness (also in the module docstring). The proposal is symmetric: the
pool is fixed and ordered pairs are uniform, so `d` and `-d` are equally
likely, and gamma does not depend on the point. The MH ratio is the tempered
posterior ratio alone. Within one propose the jumping group A conditions on
group B's start-of-propose states, while B never reads A. The joint kernel
is `K_A(x_A -> x_A' | x_B) K_B(x_B -> x_B')`, `K_A` is reversible for
`pi(x_A)` at every fixed `x_B`, and `K_B` is invariant for `pi(x_B)`.
Integrating `pi(x_A) pi(x_B) K_A K_B` gives `pi(x_A') pi(x_B')`. Over the 25
repeats of a visit, B evolves while A keeps B's start states, and the same
integral holds.

The ensemble limitation, measured. With ONE rung, a configuration with every
walker in the same island can never be entered: the last walker in the
other island only sees same-island partners. The chain then samples the
product target restricted to the reachable configurations. For a 0.7 / 0.3
two-mode target and 4 walkers, the expected right-mode fraction is (2.8 - 4 x
0.7^4) / (1 - 0.7^4 - 0.3^4) / 4 = 0.612. The test measures 0.62-0.63. With a
ladder the hot rungs straddle the islands and the cold fraction is the
target's 0.70; measured 0.700-0.705 on 4 rungs. Production has 8 rungs.
Breaking the one-group rule collapses the ensemble onto one mode (fraction
1.0), which is the paired negative control. All in
`tests/test_emri_sampling_audit.py::JumpExactnessTest`.

Log, once per refresh and rank:

```
[EMRI_DE] leaf 3 (walker 1): cross-walker jumps over the last 10 visits (active on 5 of 10): mode cold 2/24 hot 15/170, scaled cold 9/25 hot 80/175
[EMRI] inner-move stack: EigenAxisJumpMixMove 0.800 (jump mix 0.10 x 3.0) | WalkerDifferenceJumpMove 0.200 (cross-walker, mode frac 0.50)
```

### B4: the lost-rung guard

`EMRI_LADDER_REANCHOR_NATS` 50 -> 100 in the launchers; see item 6.

### B5: the ladder

Unchanged; see item 1.

### B6: defects

No plain defect was found in the periodic handling, the caps or their units.
The new code's own periodic paths are tested: difference wrap, derived pick
across 2 pi. Two inert settings are recorded for the run log:
`EMRI_PERMUTE_EVERY=10` never fires at one walker per rank, and the
retrograde prior gate (item 8).

### B7: launchers

The 6mo v9 source (`scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh`) and the
flagship template (`scripts/flagship/launcher.sh.in`) carry the block, and the
9mo / 1yr v9 are re-derived from the 6mo source. The flagship scripts are
rebuilt (`build.py`, `--check` clean). The 3mo v9 twin carries no EMRI export
and is unchanged. Exported values:

| knob | value | code default |
|---|---|---|
| `EMRI_EIGEN_AXIS_ACC_LOG`, `MBH_EIGEN_AXIS_ACC_LOG`, `SOBBH_EIGEN_AXIS_ACC_LOG` | 1 | 0 |
| `EMRI_EIGEN_AXIS_ACC_TUNE` | 1 | 0 |
| `EMRI_EIGEN_AXIS_ACC_MIN` / `_MAX` / `_FACTOR` | 0.1 / 0.5 / 2 | same |
| `EMRI_DE_JUMP_MIX` | 0.2 | 0 |
| `EMRI_DE_JUMP_MODE_FRAC` | 0.5 | 0.5 |
| `EMRI_LADDER_REANCHOR_NATS` | 100 (was 50) | 200 |

MBH and SOBBH get only the telemetry. Their controller and jump knobs exist
under the same names and stay off. Pins: `tests/test_submit_scripts_layout.py
::EigenJumpMixLauncherTest::test_the_emri_audit_block_is_on_in_every_v9_production_launcher`,
`test_the_ladder_caps_are_main_mode` (100), and the flagship line in
`tests/test_flagship_launchers.py`.

## The harness: a curved ridge with a secondary maximum

`tests/test_emri_sampling_audit.py::CombRidgeHarnessTest` (print with
`EMRI_AUDIT_HARNESS_VERBOSE=1`) has 4 walkers x 8 rungs (T_max 6), 25 repeats
per visit, refresh every 5 visits and 200 visits. Each walker is a one-walker
"rank" behind the real head / payload / rank hooks and the move's own
per-repeat seams. The target, in the table's axes: island A at u = 0 and
island B at (u0, u4) = (30, 40). They are joined by a curved bridge u0 =
15 (1 - cos(pi u4 / 40)) whose floor is 45 nats below the peaks (7.5 at the
top rung). The islands are 1 wide on axis 0 and 2 on axis 4, and the table
is the information matrix at A (sigma 1). Walkers 0 and 1 start in B, 2 and
3 in A: the 9mo situation. Seed 5:

| configuration | cold island switches per walker, visit to visit (of 199) | fraction in B (target 0.5) | cold sigma axis 0 / 4 (widths 1 / 2) | cold acceptance axis 0 / 4 |
|---|---|---|---|---|
| today (adapt + pool + temper + jump mix) | 0 0 0 0 | 1 1 0 0 | 14.1 / 19.2 | 0.06 / 0.15 |
| + B2 | 0 0 0 0 | 1 1 0 0 | 3.6 / 4.8-9.7 | 0.32 / 0.38 |
| + B2 + B3 (0.2) | 74 70 98 74 | 0.56 0.50 0.41 0.39 | 3.8-7.7 / 9.7-19.3 | 0.20 / 0.14 |
| + B3 alone | 80 91 70 98 | 0.51 0.46 0.60 0.44 | 15.7 / 20.2 | 0.05 / 0.11 |

Today the pooled adaptation sizes axes 0 / 4 to the island offset, and those
axes are dead (cold acceptance 0.06 / 0.15). B2 restores them (0.32 / 0.38)
but cannot cross. B3 moves every walker between the islands from the first
visits, and each walker spends 39-60 % in B. With the jump on, the walkers
straddle both islands, and the union widths stay large because the pool
measures the offset. B2 is what keeps those axes alive then.

## Tests

`tests/test_emri_sampling_audit.py` (new, 35 tests, CPU) covers:

* the jump's formula, pair law, gamma law, periodic wrap, single-draw-call
  stream, validation and stack label;
* the parity schedule, single-process groups and partners, block-head
  shipping and rank mapping, array sharing with the pooled sample, the off
  switch, one walker;
* exactness against the restricted target, with a ladder, and the two
  negative controls;
* the tallies: recorded = derived picks on non-orthogonal axes, the periodic
  crossing, jump-mix rows apart, a too-wide axis showing up, the window log
  and its reset, off = silent, the widened-axis note;
* the controller: the widened-but-rejected axis is shrunk back, plus floor,
  ceiling, carry-over, off, the sidecar round trip without double
  application, and inconsistent knobs;
* the real `EMRIDirectLikeMove`: the 4-rank block fan-out equals single
  process row for row with the jump in the stack, and a single-process run
  with every knob on runs and logs;
* the telemetry golden, the harness, and the knobs and stock stamping.

Mutation checks (21): each mutation broke a named test, and every source was
restored byte-identical by md5. The list is in the 2026-10-18 report. One
mutation is EQUIVALENT: deriving the pick with `A^T` instead of `A^-1` picks
the same axis for a one-axis step on unit axes. The periodic-wrap mutation
of the same function is caught.

## Unverified

* No GPU run. The new code is host-side (the add/remove move's coordinates
  and tables are host numpy, and the jump runs in numpy), but the first
  cluster launch is its first real-data run.
* No real-data mixing number. The harness numbers are synthetic. On the next
  tar, read the `[EMRI_DE]` lines (mode-jump acceptance cold / hot), the
  per-axis lines, and the island histogram (`p0` per walker per rung).
* The harness asserts that the jump moves walkers between islands. Whether
  the real islands are close enough in shape for gamma = 1 jumps to be
  accepted is measured by the `[EMRI_DE]` mode fields.
* The controller's ceiling is the adapted width (per the ruling). EMRI 5's
  walkers spread 1.7-2.6x past it on axes 9-11 (item 10c); widening past
  it is not implemented.

## What to look at on the next launch

* `[EMRI] inner-move stack: ... | WalkerDifferenceJumpMove 0.200 (cross-walker,
  mode frac 0.50)`: one line per rank. Its absence means `EMRI_DE_JUMP_MIX`
  did not reach the branch.
* `[EMRI_DE] leaf k (walker w): ... (active on 5 of 10)`: about half the
  visits active, and mode-jump acceptance cold > 0 on EMRI 0 / 3 within a
  few refreshes.
* `[EMRI_EIGEN] ... acceptance-tuned step: shrunk axes 8,9,10,11` on EMRI 3
  in the first windows, then quiet.
* The island histogram per walker per rung. Success: every walker's cold
  chain visits more than one island.
* Rollback: `EMRI_DE_JUMP_MIX=0 EMRI_EIGEN_AXIS_ACC_TUNE=0
  EMRI_LADDER_REANCHOR_NATS=50` on the launch line.

## SNR-keyed per-leaf ladders (`{BRANCH}_LADDER_TOP_SNR`, 2026-10-18, default off)

User ruling (Mike, 2026-10-18). The uniform EMRI ladder moved to T_max 64,
then to 625 ("let's go until it is at SNR ~2"), and the rung count to 11
("we want swaps ideally at 25 %": `make_ladder(12, Tmax=625)` is 11 rungs
at ratio 1.90). The aim is that the LOUDEST leaf (EMRI 3, SNR 50) sees an
effective SNR ~2 at the top rung. The valleys between the islands of item 1
scale with SNR^2, so a hot rung can hop them only once the valley is down to
a few nats. A uniform T_max is right for one SNR only: at 625 the SNR 12-15
leaves have prior-like top rungs and the SNR 20-35 leaves run hotter than
they need. Mike: "While that runs, you can add an option (default off) to
set the ladder by SNR." This section describes that option. The launchers
do not set it.

### The rule

For leaf `k` of an MBH / EMRI / SOBBH branch:

```
T_k = clip((SNR_k / TOP_SNR)^2, {BRANCH}_LADDER_TMAX_MIN, {BRANCH}_LADDER_TMAX)
betas_all[k] = make_ladder(ndim, ntemps, Tmax=T_k)
```

The top rung then sees the leaf at effective SNR `SNR_k / sqrt(T_k) =
TOP_SNR`, unless the clip bites. With the knob on, the existing
`{BRANCH}_LADDER_TMAX` is the CEILING; unset, the ceiling is the top of the
uncapped `make_ladder(ndim, ntemps)`. The floor `{BRANCH}_LADDER_TMAX_MIN`
defaults to 4. `{BRANCH}_LADDER_TOP_SNR` is a number (0 = off); the words
on / true / yes mean 2.0, the code default. At top SNR 2, floor 4 and
ceiling 625, on the 9mo EMRI leaves:

| leaf | SNR | (SNR / 2)^2 | T_max | ratio over 11 rungs |
|---|---|---|---|---|
| 3 | 50.2 | 630 | 625 (ceiling) | 1.90 |
| 6 | 34.3 | 294 | 294 | 1.77 |
| 4 | 25.2 | 159 | 159 | 1.66 |
| 0 | 21.6 | 117 | 117 | 1.61 |
| 1 | 20.3 | 103 | 103 | 1.59 |
| 7 | 17.9 | 80 | 80 | 1.55 |
| 5 | 15.4 | 59 | 59 | 1.50 |
| 2 | 12.8 | 41 | 41 | 1.45 |
| a prior-only leaf | < 4 | < 4 | 4 (floor) | 1.15 |

The rung COUNT stays per branch: the store fixes the `ntemps` axis of
`betas_all`. The spacing therefore varies per leaf. The ceiling's leaf gets
make_ladder's ~25 % swap spacing at the launcher count. A leaf below the
ceiling gets a DENSER ladder: a smaller ratio, more swaps accepted, no
extra cost. Every log line names the leaf's ratio.

`SNR_k` is the leaf's optimal SNR `sqrt(<h|h>)` at the max-lnL cold walker,
the same point the `walker_max` eigen table is built at. It is read from
the sub-state's per-walker, per-leaf inner products: `d_h` / `h_h`, written
by the move at the end of every leaf visit, stored per row under
`sub_backend/<branch>`, and on by default on the fast EMRI / MBH / SOBBH
moves through `{BRANCH}_RECORD_DH`. The walker is chosen on the cold rung of
`log_like[leaf]`. Only when that walker's `<h|h>` is unusable does the code
fall back to the detection SNR at the maximum, `sqrt(<d|h>)`, because
`<d|h> = <h|h>` there. The helpers are pure functions in
`src/lisatools/sampling/snrladder.py`.

### When a ladder is built

* **Fresh store** (`recipe.resolve_snr_keyed_ladders`, from
  `SingleSourcePEBuilder.build`). A leaf with a recorded SNR on the
  sub-state gets its ladder at build. A dead leaf (no source on any cold
  walker) gets the floor. An alive leaf with no record yet, which is the
  normal fresh case, keeps the configured ladder and is PENDING: the move
  builds its ladder at the first propose that finds a record, which is
  propose 2. That is the construction, deferred, so it happens once per leaf
  in any stage.
* **`{BRANCH}_LADDER_RESET_ON_RESUME=1`**. The stored ladders are replaced
  as today (the old -> new reset lines still print), but by the SNR-keyed
  ladders, built from the SNRs of the store's last row (`d_h` / `h_h` /
  `log_like` are loaded with the state).
* **Plain resume**. The stored ladders stand. Each leaf's refresh baseline
  becomes its first measured SNR.
* **Refresh** (`{BRANCH}_LADDER_TOP_SNR_REFRESH` = N, default 0 = never).
  Every N proposes of the move (every alive leaf is visited once per
  propose), a leaf whose SNR has moved more than 20 % since its ladder was
  built gets its ladder rebuilt. Examples are a copy's SNR jump or a lost
  source. If the new T_max equals the current top (both at the clip), only
  the baseline moves.

All of this runs in `ResidualAddOneRemoveOneMove._maybe_update_snr_ladders`,
the last step of `fanout_prepare_state`. It runs on the process that owns
the whole ensemble: single process, the flat head, or the block head before
it slices. A ladder therefore changes only BETWEEN proposes. Inside a
propose the swaps, the lost-rung guard and the tempered eigen steps all use
the visit's ladder, exactly as before. The new row is written to
`sub.betas_all[leaf]` of the state being proposed from, and from there:

* the visit reads it;
* the walker-block slices carry it by value;
* the head's pooled adaptation (`fanout_merge_extra`) starts from it;
* the saver persists it with the row (`PerLeafLadderState.storage_arrays`).

The rung STATES are not moved. A rung that the new temperatures orphan is
re-anchored by the lost-rung guard at its next visit, which is what the
reset path does today. The guard's expected tempering drop `(ndim/2)(1/beta_k
- 1)` already uses the visited leaf's own row: `propose_local` loads
`temperature_controls[leaf].betas` from `betas_all[leaf]` and passes it to
`_reanchor_lost_rungs`. This is pinned in
`GuardPerLeafLadderTest`. At T_max 625 the top-rung margin is
`6 x 624 = 3744` nats (EMRI, ndim 12), so a loud leaf's top rung is
practically never re-anchored.

**A refresh is an adaptive change.** While it happens the kernel is not
detailed-balance exact: the ladder depends on the chain's recent SNR. The
refresh therefore runs only in the search stages (`gf_stage_kind` search /
rj). It runs in PE only with `{BRANCH}_LADDER_TOP_SNR_REFRESH_PE=1`,
because of Mike's rule: no adaptive tricks in PE unless ruled. It never
runs when the stage is unknown. The deferred construction is not a refresh:
it fires once per leaf, at the leaf's first SNR.

### Migration: `rerung_store_branches.py --top-snr`

```
python scripts/fstat_proposal/rerung_store_branches.py STORE \
    --set emri=auto --ntemps-from-tmax --top-snr 2 --ladder-tmax 625      # dry run
... --apply
```

`--top-snr S` (2 when given without a value) writes the per-leaf
SNR-keyed ladders, with SNRs from the store's last row. A leaf with no
record there takes an earlier row; a leaf never recorded keeps the ladder
the tool would otherwise write. The ladders go on every stored row (the
tool's ladder rule), and the backup keeps the history. The dry run prints
the per-leaf table: row, walker, SNR, kind, (SNR/top)^2, T_max, clip and
ratio. `--set B=N` at the store's own count rewrites `betas_all` only.
`--ntemps-from-tmax` sets the count to `make_ladder(ndim,
Tmax=--ladder-tmax)`'s own (EMRI at 625: 11). `--ladder-tmax-min` is the
floor. Launch afterwards with the printed knobs: a plain resume keeps the
migrated ladders.

### Knobs (code defaults; none is set in a launcher)

| knob | settings field | default | meaning |
|---|---|---|---|
| `{EMRI,MBH,SOBBH}_LADDER_TOP_SNR` | `ladder_top_snr` | 0 (off) | the top rung's target effective SNR; on / true / yes = 2.0 |
| `{EMRI,MBH,SOBBH}_LADDER_TMAX` | `ladder_tmax` | unset | with the knob on, the CEILING of a leaf's T_max (unset: the uncapped make_ladder top) |
| `{EMRI,MBH,SOBBH}_LADDER_TMAX_MIN` | `ladder_tmax_min` | 4 | the floor (prior-only leaves sit here) |
| `{EMRI,MBH,SOBBH}_LADDER_TOP_SNR_REFRESH` | `ladder_top_snr_refresh` | 0 | refresh cadence in proposes (20 % SNR change triggers) |
| `{EMRI,MBH,SOBBH}_LADDER_TOP_SNR_REFRESH_PE` | `ladder_top_snr_refresh_pe` | off | allow the refresh in PE stages |

Logs (`[EMRI_LADDER]`, also `[MBH_LADDER]` / `[SOBBH_LADDER]`):

```
[EMRI_LADDER] EMRI_LADDER_TOP_SNR=2: per-leaf ladders from each leaf's SNR (T_max = clip((SNR / 2)^2, 4, 625), 11 rungs, EMRI_LADDER_RESET_ON_RESUME=1)
[EMRI_LADDER] leaf 3: SNR 50.2 -> T_max 630.0 (top SNR 2.0, ceiling 625 -> 625.0), ladder [1 ... 0.0016], ratio 1.90 over 11 rungs (EMRI_LADDER_RESET_ON_RESUME=1; optimal SNR at walker 1)
[EMRI_LADDER] leaf 2: SNR 12.8 -> T_max 41.0 (top SNR 2.0, within [4, 625]), ladder [1 ... 0.0244], ratio 1.45 over 11 rungs (...)
[EMRI_LADDER] leaf 1: no SNR recorded yet -> the configured ladder (T_max 625.0) until its first measurement (fresh store)
[EMRI_LADDER] leaf 1: SNR 20.3 -> T_max 103.0 (...) (construction, deferred to the leaf's first recorded SNR; optimal SNR at walker 0)
[EMRI_LADDER] leaf 5: SNR 15.4 -> T_max 59.3 (...) (refresh: SNR 21.0 at the last build, -27 %, stage kind search; ...)
[EMRI_LADDER] leaf 3: refresh baseline SNR 50.2 (the stored ladder is kept, T_max 625.0)
```

### Turning it on (for the coordinator; not in any launcher)

Once, on the relaunch that switches to it (rebuilds every leaf from the
store's SNRs):

```
export EMRI_LADDER_TOP_SNR=2          # the top rung at effective SNR ~2 per leaf
export EMRI_LADDER_TMAX=625           # already the launcher value: now the ceiling
export EMRI_LADDER_TMAX_MIN=4         # code default
export EMRI_LADDER_RESET_ON_RESUME=1  # ONE launch only
# optional, search stages only (adaptive):
export EMRI_LADDER_TOP_SNR_REFRESH=10
```

On later relaunches, keep `EMRI_LADDER_TOP_SNR=2` (and the refresh, if
used) and drop `RESET_ON_RESUME`. To get there without a reset, migrate the
store with `rerung_store_branches.py --top-snr` instead. Rollback:
`EMRI_LADDER_TOP_SNR=0` plus `EMRI_LADDER_RESET_ON_RESUME=1` once, which
restores the uniform ladder.

### Tests

`tests/test_snr_keyed_ladder.py` (CPU):

* the formula, the clip, the prior-only case and the knob casts;
* the per-leaf SNR and its walker;
* one ladder per leaf with the right rung count and spacing;
* construction through the real `SingleSourcePEBuilder`: fresh, pending,
  dead leaf, reset-on-resume, plain resume, rung-count mismatch, knob off,
  env, kwargs, refusal;
* the move's update: off, the deferred construction, the 20 % refresh,
  the PE gate, the cadence, the clipped no-op, the baseline, the knobs;
* the real `EMRIDirectLikeMove`: the deferred construction single process,
  with the 4-rank walker-block fan-out equal to it, `betas_all` persisted
  through `GFHDFBackend`, and the guard reading each leaf's own row;
* rerung `--top-snr`: dry-run table, apply, `--ntemps-from-tmax`, the
  never-recorded leaf, refusals, unchanged without the flag;
* settings and env;
* the knob at 0 equal to the golden stream (single process and the 4-rank
  fan-out). The unset case is `tests/test_eigen_pool_temper.py::GoldenStreamTest`.

### Unverified

* No GPU and no real-data run. The code is host-side (`betas_all` is host
  numpy) and reads only the inner products the fast moves already record.
* Whether per-leaf T_max mixes the islands better than the uniform 625 has
  not been measured. The quiet leaves' denser ladders are the expected
  gain; read the per-leaf swap acceptance (it should be flat over the
  pairs) and the island histogram on the first tar.
* The SNR the ladder is built from is one cold walker's. A walker on
  another island has a slightly different `<h|h>`, which is immaterial for
  a temperature.
