# 6-month run prep — TODO walkthrough (2026-08-23, rulings 2026-08-24)

Working document for standing up **6mo_v1** (config lineage: 3mo_v6).
Three workstreams: **(A)** before-launch assessment TODOs, **(B)** the
in-between-run distribution fitting (warm-start posterior→proposal),
**(C)** the MBHB/SOBHB/EMRI source-merge campaign. Each item carries a
status gate; the run launches when every BLOCKER row is green.

**User rulings (2026-08-24):**
1. A1 approach approved — `submit_gf_6mo_v1.sh` now EXISTS (copied from
   `submit_gf_3mo_v6.sh`, only Tobs-dependent lines re-derived).
2. A2 (single-walker reference) SUPERSEDED: do not build the
   union/consensus machinery now. **The F-stat refit reference = the
   GB-FREE residual** — data minus all *other* branches' templates, with
   the *current* fitted sensitivity (PSD+galfor updated as GBs are fit
   out). No GB templates in the reference ⇒ no per-walker peak deletion.
3. A3 decided: **the sources go IN.** Plan and run the merge campaign
   (workstream C) — single-GPU, multi-GPU, likelihoods, everything, at
   6 months.

---

## Where the last few weeks landed (context, 1 page)

Four production runs since 2026-08-14, all on the 2-GPU cluster node:

| Run | State (batch-3 snapshot, 2026-08-23 11:00) | Verdict |
|---|---|---|
| 3-mo **v4** | FINISHED at it=374 (the reference run) | baseline; extract h5 coords valid rows 372–374 only |
| 3-mo **v5** | it=756, full_pe; **crossed v4-final ~it490**, now +2452 w/ 178 fewer leaves | healthy; save_step 146–184 s at a 4.07 GB h5 now ~half of light iterations; flagship fdot ratio slid to 0.32x |
| 3-mo **v6** | it=291, resumed cleanly after the it=173 OOM — **fix HOLDING** (1449 stagings, all ≤1024) | **best run**: +6061 over v4-final with 27.9% fewer leaves at 2.42 min/it (lead shrank ~750 over last 45 its — watch); flagship 0.787x |
| 1-yr **v5** | it=74, 28–32 min/it; **inds/gb extract now FULL (batch-2 open item CLOSED)** | cleanest science (neg-fdot >10 mHz 10.6%); flagship up to 0.731x |

New cluster-side issues (batch-3): **gpu_util telemetry dead** (0-byte
CSVs on every launch since ~08-23 07:30 — fix the sampler wrapper before
6-mo, we're blind on memory margins); recurring mid-write kills → store
self-heal rewinds (machinery works, cause looks like preemption during
save); save cost growing with h5 size.

Key facts the 6-mo plan inherits:

* **v6 (sub-band shrinkage: `GB_SUBBAND_DIVISOR=8 / GB_BAND_UNIT_STRIDE=9 /
  GB_CAP_DIVISOR=4` + `VGB_BAND_LAYERS=8` + temper-skip) is the winning
  configuration** — more lnL per leaf, fastest wall clock. The trio is
  Tobs-invariant (band_geom study, 2026-08-23) and the cap grid reproduces
  the live grid to 1 ulp, so it ports to 6-mo directly.
* Every OOM class seen so far has a landed fix: helper-rank pool release
  (e85a9b42), staging cap `GB_INMODEL_SETUP_BATCH` (3b209c46, 4e89d919),
  infomat mempool free, temper-skip-empty. The 6-mo script must carry all
  of them from day one.
* v5's full_pe transition showed a dead-flat ~20-iteration stall at the
  gb_search→full_pe handoff (it390–410) — unexplained, watch item A9.
* Save cost is no longer negligible: v6's h5 write alternates 0.1 s /
  ~110–130 s and is ~half of a light iteration — re-opens the mpiexec
  saver-rank decision at 6-mo sizes (A5).

---

## Workstream A — before-launch assessment TODOs

Ranked; **[B] = launch blocker**, [H] = should land, [W] = in-run watch.

### A1 [B] Re-base `submit_gf_6mo.sh` onto the v6 config — DONE 2026-08-24
**Status: `scripts/fstat_proposal/submit_gf_6mo_v1.sh` written** (literal
v6 copy; deltas: TOBS_TARGET=15552000, SIGHET_NT_LAYER=120,
GB_N_SUBBANDS=4096/GPU, GB_INMODEL_SETUP_BATCH=512 (stash law linear in
N_sparse_t), EDGE_CROP_WAVELETS=60 armed (fresh store),
VGB_CHIRP_MASS_BASIS=1 + VGB_NTEMPS=8 (no migration on fresh store),
GB_NLEAVES_MAX=15000, gf6mo_v1 naming incl. the slurm-log fix; warm-start
/ source-branch / shave-stack blocks staged as comments). The old
`submit_gf_6mo.sh` (08-14) is superseded — delete or leave as history.
Original rationale kept below for the record.
`scripts/fstat_proposal/submit_gf_6mo.sh` exists (383602ef, 2026-08-14) but
predates the entire v5/v6 era — a knob diff against `submit_gf_3mo_v6.sh`
shows it is missing ~40 exports, including every OOM fix
(`GB_INMODEL_SETUP_BATCH`, `GB_INFOMAT_MEMPOOL_FREE`), the v6 geometry trio,
the ridge move (`GB_RIDGE_GIBBS=1`), birth-ratio/SNR-trunc knobs, direct
batch, router device-resident, sig-het guards, and the tuned
`GB_JUMP_FACTOR=1.2` (the old script pins the long-obsolete 0.2).
**Action**: copy `submit_gf_3mo_v6.sh` and re-derive only the
Tobs-dependent lines:
* `TOBS_TARGET=15552000` (Nf 1440 × Nt 4320 × dt 2.5 — exact snap, same
  layer_df; comment block in the old 6-mo script is still correct),
* `SIGHET_NT_LAYER=120` (36-h stride, same density as validated 3-mo 60),
* `SIGHET_N_CP=256` — but see A6 (256 ceiling caps Tobs ≥ 6 mo; re-verify),
* `EDGE_CROP_WAVELETS=60` (constant-layer ruling; 2.8% at 6 mo),
* `GB_N_SUBBANDS` per-GPU sizing at the ~2 MB/slot 6-mo cost (8192/GPU at
  3-mo → likely 4096/GPU at 6-mo; check the honest sizing INFO line),
* `GB_INMODEL_SETUP_BATCH`: 1024 at 3-mo, 512 at 1-yr — pick 6-mo value
  (768–1024) and keep the escalation ladder comment,
* `VGB_CHIRP_MASS_BASIS=1` (the 6-mo debut, already designed + migration
  script exists; the v6 3-mo script runs 0),
* keep `GB_RJ_BAND_SHUTOFF_*` and decide scope (A8),
* fresh `STORE_DIR=./gf_prod_6mo/`, job name, 24 h wall.

### A2 [B] F-stat refit → GB-FREE reference (RULED 2026-08-24)
Supersedes the union/consensus design (not built now). **The refit fits
the F-stat grid against the GB-free residual: data minus all OTHER
branches' templates (VGB/MBH/SOBBH/EMRI at the reference walker), GBs
NOT subtracted, with the CURRENT fitted sensitivity** — so the grid's
peaks track the updated PSD/foreground as GBs are fit out, and no peak a
walker still needs can be deleted by another walker's subtraction. This
is the same construction as the v4 clean restart's "GB-free F-stat grid"
(2830 peaks), extended from epoch 0 to every refit.
**STATUS (verified in code 2026-08-24): already implemented and default
ON** — `GB_FSTAT_GB_FREE=1`, `_gb_free_residual`
(gbspecialstretch.py ~:11313; commits 85b59671 + 203c35be crash fix +
942aea11 cache fingerprint). The sweep restores the reference walker's
cold GBs into the residual, so it sees data minus MBH/VGB/noise/
foreground at their current fitted state with no GB subtraction; the
live noise_index supplies the current sensitivity.
**The real open item: refits never fire in production — ROOT-CAUSED
2026-08-24.** The decision table is correct and unit-tested; its clock
starves: the cadence counts per-instance, in-process `propose()` calls,
and (a) full_pe draws ONE of 6 moves per iteration (`random_choice=1`)
so `rj_fstat_pe` needs ~300 iterations in a single launch to reach 50
hits; (b) `rj_fstat_search` only lives ~11 iterations per launch
(gb_search's stopping rule) before handing off; (c) nothing persists
`num_proposals`/`_fstat_last_fit_hit` across launches — every restart
zeroes both and re-syncs on load; (d) the search and pe instances don't
pool hits despite sharing the grid. **Fix sketch**: count on a run-level
clock (backend iteration or the shared `_branch_propose_counts`) and
persist the last-fit value in the epoch DONE.json.
**Second finding**: at a real refit, `_install_ctr_table` runs AFTER the
GB-free window closes — the epoch center table would be swept against
the GB-subtracted residual while the peaks came from the GB-free one,
mis-centering exactly the loud recovered peaks the design protects. Move
the center-table sweep inside the same GB-free window when fixing the
cadence. (Also: the n_live>0 restore/re-subtract round-trip has never
run at production scale.)
Trade-off accepted: the grid keeps peaks for already-recovered loud GBs
(births there are MH-rejected / at-cap).

### A3 [B] Branch scope — RULED 2026-08-24: sources go IN
MBHB/SOBHB/EMRI are in scope for the 6-mo run, gated on the source-merge
campaign → **workstream C** below. Physics expectations set by the user:
most SOBHBs undetectable at 6 mo; EMRI detectability unknown (census);
MBH selection = only systems that merge within the observation window.

### A4 [B] 3-mo correctness gate = warm-start input gate
The 3-mo posterior seeds the warm start (workstream B), so it must pass
assessment first: recovery vs full-band truth sets (3-mo det = 1064,
overlap-verified — NEVER f0-proximity dedup), faint-tail fraction,
PSD/galfor posteriors vs injected, VGB truth-null lnL, flagship
fdot ratios (v6 0.90x truth), drift floor, cell-ll warnings.
Choose the donor run: **v6 (best lnL/leaf) after its resume reaches
stable full_pe**, else v5 (deepest PE, it=473). Note v5 it=472 coord slab
is unwritten — use 471.

### A5 [H] mpiexec saver-rank decision (re-opened)
3-mo-era answer was "not justified" (~2%), but v6 now measures the h5
write at ~half of light iterations, and 6-mo state is ~2× bigger.
(a) Read [SAVE]/save_step numbers from the current v5/v6 logs;
(b) if adopting: verify rank-gated build (saver/spare ranks must not
allocate GPU memory in `fit.build()` — helper-pool release e85a9b42
already parks less, but build itself is unverified), `--ntasks=3`,
handoff cost. The commented block at the bottom of the 6-mo script is
the template.

### A6 [H] Sig-het accuracy re-verification at 6-mo geometry
`SIGHET_N_CP=256` is the shared-arena ceiling; at 6-mo that is a coarser
effective control-point spacing than the 0.35-day criterion validated at
3-mo. Run `gb_sighet_bfold_gpu_probe.py` (or the anchor-check path) on
the 6-mo grid before trusting low-f h_h; the durable fix (amp/phase
redesign) is not a 6-mo deliverable. Tiered lens applies
(allowed(T) ~ max(0.1, T/100)).

### A7 [H] Startup-to-target-stage latency
~8 min per restart re-walking setup + earlier stages at 3-mo; grows with
Tobs and branch count and throttles every debug cycle. Wants: stage
fast-forward on resume (skip stages whose exit criterion already passed
in the h5) + lazy per-branch builds. If not landed, accept and budget
restart cadence.

### A8 [H] High-f barren-band shutoff scope reassessment
Currently `GB_RJ_BAND_SHUTOFF_SCOPE=search`. Decide for 6-mo whether
pe/full_pe should also stop birthing in barren >10 mHz bands or keep full
trans-D sampling there. (SNR-gate assessment 2026-08-23: keep
`GB_SNR_REJ_DETECTED` OFF — that ruling stands.)

### A9 [W] gb_pe stage watches (from v5, first run to reach full_pe)
* the dead-flat gb_search→full_pe stall (it390–410): understand mechanism
  or at least recognize it live;
* PE temper-cell occupancy ~53% → temper-skip yields little in PE;
* `GB_TEMPER_EVERY_PROPOSES=3` (PE cadence) never A/B'd;
* rj_fstat_pe wall bimodality.

### A10 [W] In-run checks carried from the trackers
* VGB-vs-GB cost crossover (user expectation: vgb_pe flat, GB grows;
  vgb_pe > ~15% of iteration at maturity = something wrong);
* GPU0 headroom watch (1-yr margin was 1.1 GB; 6-mo should be roomier
  than 1-yr but verify first save + first full_pe entry);
* interlock/pgrep `pytho[n]` pattern in any watcher scripts; caffeinate
  during laptop batch processing;
* slurm log naming (still `*v4*.log` legacy in some scripts — fix name in
  the new script).

---

## Workstream B — in-between-run distribution fitting (warm start)

Design is DONE and committed: `docs/warm-start-gb-proposal.md` +
validated prototype `scripts/gb/proto_warmstart_cluster.py` (e5889a3b).
Pipeline: f0 density-valley segmentation → within-island robust-whitened
single-linkage → per-cluster Gaussian + inclusion probability p.
Prototype: 10/10 synthetic sources incl. p=0.10 and a 2.5/Tobs blend;
0.65 µs/point mixture eval; 13 s full-table f0 pass at 6e7 rows.

### B-TODOs (build order)

1. **[B] Real-data extractor**: previous-run h5 → the (rows × 9,
   sample_id) leaf table (cold chain, last-K iteration window, all
   walkers). Read `global_fit/chain/gb` + `inds` directly (h5py or
   `GFHDFBackend`); no waveforms, no curr_info needed. Input
   availability resolved below (B-data): unzip the local full stores.
2. **[B] v1 refinements** flagged in the design note: satellite-fragment
   merge pass, circular phi0/psi in the fitted component, covariance
   eigenvalue floors.
3. **[B] Components → proposal object**: npz schema (means, covs, p,
   mult, n) + an f0-windowed Gaussian-mixture proposal with exact
   rvs/logpdf (same plumbing shape as `StackedFStatProposal4D`;
   `fit_gb_gmm_rj_container`/`FullGaussianMixtureModel` are the
   ready-made container layer — reuse, don't rewrite), wired via the
   `rj_birth_distribution` hook; mixture weight vs the F-stat proposal
   (`GB_WARM_START_COMPONENTS` / `GB_WARM_START_WEIGHT` — the commented
   block already sits in the 6-mo script). Clustering stage stays a
   swappable strategy (user ruling).
4. **[B] Cross-Tobs policy** (design note): propose at previous-run
   widths (wider = safer for MH), f0 windows re-checked against the new
   1/Tobs. No Fisher re-scaling in v1.
5. **[H] Leaf seeding decision**: proposal-only vs also seeding initial
   leaves from high-p components (`past_file_for_start` exists for
   direct state restart but is cross-Tobs untested). Design session
   pending with user.
6. **[H] PSD / galfor / VGB warm starts**: low-dim, direct
   samples/GMM — cheap; decide per-branch (VGB chirp-basis migration
   interacts: a 3-mo VGB state carried forward needs
   `migrate_vgb_chirp_basis.py`).
7. **[W] Validation**: p_hat sanity vs run-level completeness numbers;
   cross-run dedup uses overlap/match statistic, never f0 proximity.

### B-existing code — reuse map (2026-08-23 sweep)

More exists than the tracker's "MISSING: builder" line suggests, but each
piece has a catch:

| Piece | What it does | Catch |
|---|---|---|
| `scripts/gb/proto_warmstart_cluster.py` | the validated 3-stage pipeline | synthetic input only; no h5 reader; CPU numpy/scipy |
| `globalfit/gathergalaxy.py` `gather_gb_samples` | production sample-grouping by waveform **overlap** (reads `chain/gb` cold last-K via `GFHDFBackend`) | **requires cupy/GPU** + a live curr_info/ACS (regenerates FD waveforms + needs sens_mat); exact path O(N·g²) crawls at large samples_keep |
| `globalfit/postprocessing.py` `SubmissionWriter` / `buildcatalog.py` | end-of-run clustering → submission/catalog (knobs: samples_keep=5, overlap_lim 0.7) | in-process at run end, no CLI; GPU required; `build_mbh_catalog` has a guaranteed NameError (commented-out import) |
| `sampling/gmm.py` `vec_fit_gmm_min_bic` / `fit_gb_gmm_rj_container` | batched per-group GMM fit → 8/9-col eryn RJ birth container with rvs/logpdf | **CPU-capable** (`gpu=None`); this is the ready-made stage-3/densify + container layer |
| `moves/gbspecialstretch.py` `GBSpecialRJRefitMove` | the full h5 → gather → GMM → RJ-proposal chain, already a move | **raises NotImplementedError on WDM runs** (FD-basis only — gather builds FD waveforms); production runs are WDM, so unusable as-is; fp is wired to the *current* store |

So the v1 between-run builder = prototype pipeline (density-valley +
single-linkage, **no waveforms → no GPU, no WDM problem**) reading the
previous store's chain, with `fit_gb_gmm_rj_container` /
`FullGaussianMixtureModel` as the densify + proposal-container backend.
The overlap-based `gather_gb_samples` machinery is the *upgrade* path
(match-statistic referee), not the v1 dependency — dodging both its GPU
requirement and its WDM gap.

### B-data — inputs on hand (investigation result)

* **The cluster-level stores are already local** (LAT repo root, zipped):
  `gf_prod_3mo_v5.zip` (665M → store h5 329M, chain through it≈473),
  `gf_prod_3mo_v6.zip` (658M → 181M, through it=173),
  `gf_prod_1yr_v5.zip` (231M → 132M), plus each run's
  `gb_fstat_fit/shared/epoch_0000/` grid caches. **No cluster pull
  needed to build and validate the fitter** — unzip locally.
* The snapshot **tars are NOT sufficient** for fitting: the extractor
  (`gf_store_extract.py`) keeps only the last 3 iterations of
  `chain/gb`/`inds` (~3 × 24 walkers × ~800 leaves) — 4 orders of
  magnitude below the ~1e7–1e8 rows the design targets. Fine for smoke
  tests of the reader, not for the real fit.
* Required slabs: `global_fit/chain/gb[it, temp0, walker, leaf, ndim]` +
  matching `inds`, last-K window (K sized from tau_int 4–8 stored iters;
  ~all stored PE iterations is affordable). v5 it=472 coord slab is
  unwritten — use ≤471.
* **CPU/GPU verdict**: v1 fit runs on this laptop, CPU-pinned
  (`OMP_NUM_THREADS=1`, VECLIB pinned, ≤50% budget — the 6e7-row f0 pass
  measured 13 s; linkage runs on ≤1500-row subsamples). GPUs enter only
  for the deferred match-statistic referee or an overlap-based upgrade —
  in that case run it on the cluster next to the live store instead of
  pulling data.
* Freshness caveat: local zips are the 2026-08-22/23 batch. Build and
  validate against these; refit on the donor's final store right before
  launch (fit is minutes, by design).

---

## Workstream C — MBHB/SOBHB/EMRI source-merge campaign (ruled IN, 2026-08-24)

Rerun of the testing campaign (`scripts/campaign/`: gates.py DAG,
ledger.json, campaign.py CLI) for the three source branches, **retargeted
to Tobs = 6 mo** (the original end target was all_sources 2 yr — that
stays the eventual goal; this pass gates the 6-mo run). Hard rule stands:
every test runs THROUGH the stock structure (`erebor.<name>()` → build →
sample), minimum granularity = "the fit runs with this component
exercised in-sampler". Ledger state today: T1 green (t1-gt-mbh yellow),
T2-lite mostly green (gb/mbh yellow), **all T3-gpu / T4-heavy / T5-mg
gates pending** — exactly the single-GPU/multi-GPU/likelihood coverage
this ruling asks for.

### C0 [B] Unblock the known per-source work items (inside the gates)
* **EMRI**: commit the interpolate.cu C++ throw hardening in FEW (exists
  UNCOMMITTED), pull + FULL rebuild on the cluster GPU box — the cusparse
  hard `exit(-1)` bypasses the Python domain guard and is a
  process-killer in a week-long run. Domain-validity guard itself is in.
* **SOBBH**: close the chunked moving-window chirp build (the open
  sig-het item) — or pin the 6-mo SOBBH path to its validated
  TDI-on-the-fly/chunked configuration and defer the moving window.
* **MBH**: t1-gt-mbh / t2-lite-mbh yellow cleanup (lite needs
  CHOP_WINDOW=1 — uncommitted fixes from the campaign-lite session);
  T-channel + start_freq truncation are understood, not blockers.

### C1 [B] 6-mo detectability censuses (sets each branch's catalogue cut)
* **MBH**: catalogue cut **t_merge ≤ TOBS_TARGET** (user ruling — only
  systems that merge in-window). **Computed 2026-08-24** from the mojito
  lite catalogue (20 MBHBs): exactly **4 merge within 6 mo — ids
  2, 5, 16, 18** (t_c = 173.3 / 104.7 / 111.4 / 92.0 days).
* **SOBBH**: 6 in catalogue — small enough to run all; the probe below
  doubles as the census (expectation: mostly sub-threshold).
* **EMRI**: 8 in catalogue (ids 0–7) — probe runs all, census for free.

### C-PROBE [READY] `submit_gf_6mo_sources_probe.sh` (user request 2026-08-24)
Written: `scripts/fstat_proposal/submit_gf_6mo_sources_probe.sh` — the
stock `full_year_combined` variant (MBH+EMRI+SOBBH only, no GB/VGB, no
psd/galfor branches, **fixed sensitivity fitted to the mojito NOISE
brick = injection PSD**, source-only likelihood) re-gridded to
TOBS_TARGET=15552000, EDGE_CROP_WAVELETS=60 (same domain as 6mo_v1),
NWALKERS=24, 30 iterations, **GPUS=0,1** with [GF_MOVE_TIMING]+SYNC
armed throughout. Sources: MBHB_IDS=2,5,16,18 / EMRI_IDS=0–7 /
SOBHB_IDS=0–5. Config dry-constructed locally and verified (3 branches,
one full_pe stage: mbh_pe/emri_pe/sobbh_pe).
**Confusion-foreground data added (user request)**: the self-generated
`GALFOR_731d_2.5s_L1.h5` (subtract_resolvable_tdi.py output — GB L1
brick with resolvables regenerated and subtracted) is summed into the
data via a shadow mojito folder (all classes symlinked, `data/GB/L1`
holds only the confusion file under a `GB_*source0_*` name) + a
class-API driver adding `GB:[0]` to source_ids. **File is NOT on the
laptop** (verified) — the script auto-searches the cluster cache and
exits loudly if absent (`GALFOR_FILE` env overrides). The variant's
fixed sensitivity already carries the modeled confusion component
(InstrumentNoise + annually-modulated GalacticForeground in
extra_components), and `ADD_GALACTIC_FOREGROUND` is deliberately NOT
set (it would add a synthetic draw on top of the real one). Shadow
mechanics + find_file resolution + noise-brick discovery all verified
locally with a dummy file. Covers C1 (census), first
half of C3/C4 (single-vehicle 2-GPU exercise + per-move walls). KNOWN
RISK, acceptable here: EMRI cusparse hard-exit hardening still
uncommitted in FEW (C0) — a bare exit(-1) in this probe is that item
firing. Also verify the gpu_util CSV is non-empty (telemetry dead on
current launches).

### C2 [B] Likelihood ground truth at 6 mo (t1 rerun at this Tobs)
Truth-null residuals per source through the stock path at
TOBS_TARGET=15552000; criteria = 2× the mojito null baselines (MBHB
worst 1.5e-3 @ SNR 1009, SOBHB 1.4e-6, EMRI 5.7e-4). Catches anything
Tobs-dependent (window/taper/start_freq) the 3-mo and full-year passes
couldn't.

### C3 [B] Single-GPU in-sampler gates (t3-gpu-{mbh,sobbh,emri} @ 6 mo)
Per-branch runs via the ids envs (empty EMRI_IDS/MBHB_IDS/SOBHB_IDS
drops a branch; NB SOBHB= ids spelling vs SOBBH branch prefix), a few
iterations each, on one cluster GPU. Requires C0-EMRI first.

### C4 [B] Multi-GPU gates (t5-mg-correctness/scaling @ 6 mo)
2-GPU parity vs the C3 runs (the all_sources hardening — jax
default_device + cupy ctx + per-device orbits TOGETHER — landed but is
unproven at this Tobs/branch mix), then scaling readout.

### C5 [B] Recipe fold-in + combined smoke (t4-heavy @ 6 mo)
Staged search order for the new branches (full_year pattern: MBH search
first), revisit the `remove_branch` lines in `run_combined_staged.py`,
then a short 7-branch combined smoke on the 6-mo grid (fresh throwaway
store): sizing lines, stage handoffs, no OOM, [GF_MOVE_TIMING] per-branch
walls. Startup-latency item A7 bites here — budget restart cadence.

### C6 [H] Warm start for the new branches
MBH/SOBBH/EMRI warm-start from direct samples of previous posteriors
(design note scope: clustering is GBs-only) — full_year_combined
posteriors are the donors where applicable; else cold start (searches
exist).

**Execution split** (campaign convention): laptop CPU tiers autonomous;
cluster gates via `campaign.py batch N` checklists the user runs and
hands logs back for ingest; dashboard republished at checkpoints.

---

## Launch sequence (the walk-through, updated 2026-08-24)

1. ~~A3/branch-scope + donor rulings~~ DONE: sources IN, campaign C gates
   them; donor = v6 (else v5 ≤ it471).
2. Cluster side (user): keep v6/v5/1-yr running; **fix the dead
   gpu_util sampler** (0-byte CSVs since ~08-23 07:30) — the 6-mo memory
   questions need it.
3. ~~A1 script~~ DONE: `submit_gf_6mo_v1.sh` written (4-branch as-is;
   source + warm-start blocks staged as comments).
4. A2': implement the GB-free refit reference (+ current sensitivity) in
   the fit-in-move path; verify on a 3-mo store refit before 6-mo.
5. B1–B3: extractor → refined clustering → proposal object; validate
   locally against the unzipped v5/v6 full stores (CPU-pinned, ≤50%
   budget); refit on the donor's final store just before launch.
6. C0–C2 in parallel with 4–5 (laptop-feasible parts; EMRI rebuild is
   cluster-side user action).
7. C3–C5 on the cluster (campaign batch checklists).
8. A6 sig-het n_cp probe on the 6-mo grid; A5 saver decision from the
   latest [SAVE]/save_step numbers (v5: 146–184 s at a 4 GB h5 — the
   saver rank is likely justified now; the v6-lineage script already
   runs `mpiexec -n 3`).
9. Dry-run: short 6-mo smoke (fresh throwaway store) with warm-start
   armed — sizing lines, proposal fires, no staging OOM at full width.
10. Launch 6mo_v1; A9/A10 watches on the first snapshots.
