# 6-mo readiness campaign (2026-08-25)

The test campaign that gates the 6mo_v1 launch. Follows the standing
campaign rules (`scripts/campaign/`): every test runs THROUGH the stock
structure; a gate is green only with parsed evidence. Execution split:
laptop CPU gates run autonomously; cluster gates are batch checklists the
user runs and hands logs back for ingest.

**Scope ruling (user, 2026-08-25):** the F-stat proposal is being worked
and tested BY THE USER right now, and the origin data for refit testing
is inaccessible — so the campaign focuses AWAY from the fstat/refit
track; that track (F below) builds up slowly and gates nothing else.

**Standing precondition P0 [cluster, user]:** fix the gpu_util sampler
(0-byte CSVs on every launch since 08-23 ~07:30). Gates S1/D5 produce
their memory evidence only with it working; they can RUN without it.

---

## Track S — source branches (MBH / EMRI / SOBBH at 6 mo)

**S1 — sources timing probe [cluster, READY NOW]**
`git pull && sbatch scripts/fstat_proposal/submit_gf_6mo_sources_probe.sh`
(pushed; foreground OFF). PASS: 30 iterations complete; `[GF_MOVE_TIMING]`
per-move walls for mbh_pe / emri_pe / sobbh_pe captured; both GPUs show
activity; no cross-device errors. Precondition: cluster cache has MBHB
L1 bricks for ids 2 and 5 (loader raises loudly if not).
Evidence: slurm log + gpu_util CSV.

**S2 — EMRI cusparse hardening [cluster, user + build]**
Commit the interpolate.cu C++ throw hardening (UNCOMMITTED in FEW),
pull + FULL rebuild on the GPU box. PASS: the known crash case
(m<3 at p_sep + 2.001*DELTAPMIN) returns ll = −1e300 through the domain
guard instead of killing the process. This is the one hard blocker for
any long run containing EMRIs — if S1 dies with a bare exit(-1) in an
EMRI likelihood, that is THIS gate failing, by design.

**S3 — SOBBH chunked likelihood at 6 mo — GREEN 2026-08-25 (evidence in)**
Measured by the removal-null run (`scripts/sobbh/sobbh_removal_null_6mo.py`,
pushed d33bb888): removal via the production chunked fill is **GO** —
residual 5.5–6.7e-4 of ⟨h|h⟩ per source at the 32/8 defaults, truth
null −3.9e-4, and the Nt_sub curve is U-shaped with its optimum exactly
at the default (κ*≈3.5 layers/chunk; tolerance ~7 with fill ≥ 4;
collapse ~14; Nt_sub=8 pays 13× stitch overhead). Scoring: **m=1 is
NO-GO on the full_year grid** (11.1-h layers — sheds 15% of ⟨h|h⟩, bias
0.076·SNR²; m=2 recovers 99.9%) and expected GO on the production 1-h
grid. Shipped: chirp-safe auto-sizer + criterion warnings
(`SOBBH_NT_SUB=0`, `SOBBH_SWEEP_MAX_LAYERS=3.5`), the prepared-settings
cfg fix, `SOBBH_M_BAND_HALF_WIDTH=3` in the probe (user ruling: converged width — not a bottleneck). The sig-het
moving-window build stays DEFERRED (not needed for 6 mo). Residual
follow-up: watch the `[sobbh_chunked]` resolver line in the S1 log.

**S4 — detectability census readout [laptop, after S1]**
From S1's store: per-source posterior width vs prior for all 8 EMRIs and
6 SOBHBs (expectation: most SOBHBs prior-like), MBH recovery sanity for
ids 2/5/16/18. PASS: census table appended to this doc; it fixes the
final EMRI_IDS/SOBHB_IDS for the production run.

**S5 — multi-GPU parity [cluster]**
Re-run S1's config short (5 its) on ONE GPU; compare cold lnL trajectory
vs the 2-GPU run. PASS: agreement within the known threaded-noise floor
(runs are not bit-deterministic — compare magnitudes, not bits); no
device-placement errors. This closes the t5-mg-correctness slice these
branches need.

**S6 — 7-branch combined smoke [cluster, after S2 + W3]**
Fold mbh/emri/sobbh back into the staged recipe (revisit the
`remove_branch` lines + stage order: MBH search first, full_year
pattern) and run a few iterations on the 6-mo grid in a THROWAWAY fresh
store. PASS: all stage handoffs, honest sizing lines, no OOM, per-branch
`[GF_MOVE_TIMING]`. This is the t4-heavy slice and the last gate before
the production script gains the source blocks.

## Track W — warm start (the between-run GMM pipeline)

**W1 — final-store fit at depth [cluster or zip pull]**
Run `scripts/gb/warmstart_fit_from_store.py --last-k 50` against the
FULL final v5 store (it=1005; the laptop extracts carry only ~5
iterations of coords — keep-window). Minutes, CPU. PASS: accounting
closes (Σ p·mult ≈ leaves/walker − drops); p-structure stable vs the
5-it laptop fit; components written to a durable npz.

**W2 — referee the production npz [laptop or cluster, ~45 s]**
`scripts/gb/warmstart_match_referee.py` on W1's npz. PASS: ≥95% of
refereed groups sinc-coherent (ratio ≥ 0.5); merge candidates ≤ 2%;
blend tail flagged and counted. Its output is the arming evidence for
`GB_WARM_START_COMPONENTS`.

**W3 — rj_warm_search in-sampler smoke [laptop CPU]**
A lite GB run (gb_no_fg_lite / synthetic) with `GB_WARM_START_COMPONENTS`
pointed at a small npz. PASS: the move appears in the gb_search cycle
IMMEDIATELY BEFORE rj_fstat_search (timing lines), warm births are
proposed and some accepted, death factors finite (the floor guard), and
with the knob unset the run is bit-identical to baseline. (The wiring
unit tests already pass; this is the sampler-level exercise.)

**W4 — warm-start A/B probe [cluster, after W1–W3]**
Short GB search on the 3-mo grid WITH the W1 components armed (fresh
store or rewound copy): `[GB_ACCEPT]` split per move — warm-birth
acceptance vs fstat-birth acceptance on recovered sources; drift +
cell-ll clean. PASS: warm births accept at a healthy rate (they carry
full posteriors incl. phases — expect ≥ fstat's) and nothing regresses.
This is the go/no-go for arming warm start in 6mo_v1.

## Track D — the 6-mo domain, noise, and tooling

**D1 — build smoke [cluster, cheap]**
`sbatch submit_gf_6mo_v1.sh` on a fresh store; let it reach noise_search,
save once, scancel. PASS: WDM grid resolves 1440×4320; the sig-het
edge-exclusion guard passes (predicted taper+margin 30 ≤ crop 60); GB
sizing INFO lines match the ~2 MB/slot law at GB_N_SUBBANDS=4096; no
build errors. (The store can be kept — D3 resumes it.)

**D2 — sig-het n_cp re-verification [cluster GPU] (prep item A6)**
`gb_sighet_bfold_gpu_probe.py` on the 6-mo grid at SIGHET_N_CP=256.
PASS: scored-anchor |log hh ratio| at the ~1e-4 level seen at 3 mo.
FAIL → the low-f h_h trust question reopens (amp/phase redesign
discussion) before GBs run at 6 mo.

**D3 — noise stages at 6 mo [cluster, resumes D1]**
Run noise_search → noise_vgb to completion (~2–3 h fresh fit — the 3-mo
graft does NOT apply, the grid changed). PASS: psd/galfor track the
injection (galfor's known prior-pull caveat noted, not failed), VGB
8-rung ladder built on the chirp basis (ndim 6), stage walls recorded
(the Tobs scaling-law datapoint).

**D4 — tooling at 6 mo [laptop]**
`build_truth.py` at TOBS_TARGET=15552000 (stamped band, explicit
--iteration and --kappa-out per the known traps) + `gf_monitor_gen.py`
against a D1/D3 snapshot. PASS: monitor page renders with 6-mo stamps;
truth npz written to a durable path. Done before the first real
snapshot arrives, not after.

**D5 — memory envelope [cluster, needs P0]**
During D3 and S6: nvidia-smi peaks vs the 96 GB cards with the 6-mo
knobs (4096/GPU, SETUP_BATCH 512). PASS: comfortable headroom (>10 GB)
through a save and a full-width in-model block.

## Track F — F-stat proposal + refit (user-owned / slow build; gates nothing)

- **F1 (user, in progress):** F-stat proposal testing — outside this
  campaign's scope by ruling.
- **F2 (slow build):** refit-cadence + GB-free e2e. The origin data for
  the intended refit test is inaccessible, so build up in steps as data
  allows: unit layer DONE (51/51 incl. restart-budget survival);
  next when possible = a lite CPU run with GB_FSTAT_REFIT_EVERY set low
  on synthetic data (watch clock.json appear + epoch_0001 fit + the
  GB-FREE "restoring N>0 signals" line); finally observed refits on the
  next real run's logs. Nothing else waits on this.

---

## Suggested order

1. **Now:** S1 (probe is pushed), P0 (telemetry), W1 (final-store fit —
   cluster CPU, minutes), D4 (laptop, parallel).
2. **Next:** S2 (EMRI rebuild), W2+W3, D1.
3. **Then:** D2, D3, S3–S5, W4.
4. **Last before launch:** S6 (7-branch smoke) → arm warm start + source
   blocks in `submit_gf_6mo_v1.sh` → launch per the sequence in
   `docs/6mo-run-prep.md`.

Open user rulings the campaign will surface: mult policy (W2's blend
tail), SOBBH path (S3), tukey-law choice for ≥1-yr scripts (not a 6-mo
blocker), final EMRI/SOBHB id cuts (S4).
