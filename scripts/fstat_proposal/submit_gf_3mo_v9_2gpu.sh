#!/bin/bash
# ============================================================================
# PRODUCTION global fit -- 3mo_v9_2gpu. Derived from
# submit_gf_6mo_v9_4gpu.sh, per
# docs/superpowers/specs/2026-09-26-3mo-test-run-handoff.md: COPY THE 6MO
# FILE AND EDIT IT. Do NOT derive from submit_gf_3mo_v9.sh -- it has drifted
# from the 6mo script by ten knobs, several of them correctness-relevant (the
# phase-maximization ruling, the SNR floors, the unequal-arm PSD reference),
# and ThreeMonthTwinTest has been failing on exactly that for days.
#
# Design, and the provenance of every number below:
#   docs/superpowers/specs/2026-09-26-3mo-v9-2gpu-design.md
#
# Everything here is the 6mo v9 file verbatim except the changes listed now.
# Diff the two files and the only hunks should be these.
#
#   3MO-1. Tobs = 90 d and ONLY its derived settings: TOBS_TARGET=7776000
#          (Nf 1440 x Nt 2160 x dt 2.5), SIGHET_NT_LAYER and
#          EDGE_CROP_WAVELETS UNSET (the 3-month code defaults 60/stride-36h
#          and 20 ARE the validated values -- see their blocks),
#          GB_NLEAVES_MAX=10000, GB_N_SUBBANDS=32768/GPU,
#          GB_RJ_INMODEL_CHUNK=65536, gf_prod_3mo naming.
#          ⚠ The four sig-het STAGING knobs (GB_INMODEL_SETUP_BATCH,
#          GB_SIGHET_FOLD_MAX_BYTES, both *_MEMPOOL_FREE) deliberately do
#          NOT revert to the v8 3mo arm's aggressive sizing (0 / 8 GiB /
#          0 / 0). They keep the 6mo post-OOM values, and that is what makes
#          32768 slots safe here -- see the GB_N_SUBBANDS block.
#   3MO-2. NO WARM START, NO gb_search_seed. A 3-month run is the SOURCE of
#          warm-start components, not a consumer. GB_WARM_START_COMPONENTS is
#          explicitly EMPTY and GB_SEARCH_SEED_ITERS=0, so rj_warm_search /
#          rj_warm_pe are absent from every stage and the seed stage is gone.
#          The seed STORE goes with them: this run reads no previous run's h5.
#   3MO-3. THE NOISE EVOLVES THROUGHOUT (revised 2026-09-26 off the first
#          3-month run). noise_search opens the recipe fitting psd AND
#          galfor jointly, and the noise then KEEPS SAMPLING in all three
#          GB search stages -- the leading joint rider plus a noise
#          convergence after every in-model convergence proposal. galfor
#          STARTS at the 90-day add-back fixed point inflated by amp x1.25
#          / fk x1.10 but is free from iteration 0. Knobs
#          GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1 + STAGE_SKIP_NOISE_VGB=1.
#          ⚠ THE FIRST ATTEMPT FROZE psd+galfor through gb_search_1/2 and
#          failed two ways at once: the four walkers' psds DIVERGED (four
#          independent prior draws, each maximized to its own optimum, then
#          frozen apart, each walker searching against its own floor), and
#          whatever stage 0 converged to was contaminated by the entire
#          unsubtracted GB galaxy. Freezing was the defect, not the value.
#   3MO-4. DATA = COMBINED, BRANCHES = gb + vgb + psd + galfor ONLY.
#          SOURCE_TYPES=COMBINED,GB,VGB with MBHB_IDS / EMRI_IDS / SOBHB_IDS
#          UNSET (not set-empty -- that breaks `import erebor`) and
#          STAGE_SKIP_SOURCE_SEARCH absent (it RAISES with nothing armed).
#   3MO-5. 2 GPUs ON 1 NODE with NWALKERS=4 -- the first production shape
#          with NWALKERS > NGPUS (two walkers per compute rank).
#   3MO-6. MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM dropped -> defaults to 1.
#          The 6mo pin at 0 exists only to keep a v8-lineage store resumable;
#          this store is fresh, so it takes the correct unequal-arm reference.
#
# The recipe this composes to:
#
#   noise_search  psd_pe + galfor_pe (vgb frozen at exact truth)
#   gb_search_1   phase_max on,  opt_snr 8, F-stat peak 8     noise SAMPLED
#   gb_search_2   phase_max off, opt_snr 5, F-stat peak 6.25  noise SAMPLED
#   gb_search_3   phase_max off, opt_snr 5, F-stat peak 6.25  noise SAMPLED
#   full_pe
#
# Every search stage runs the leading joint psd+galfor+vgb rider AND a noise
# convergence after each in-model slot; the three still differ in phase
# maximization, the opt-SNR boundary and the F-stat peak floor.
#
# ⚠ FRESH STORE. STORE_DIR below is a new path; the first submission starts
# from scratch and re-submitting this same script RESUMES it. For a genuinely
# fresh restart, move that directory aside first.
#
# ---- inherited from 6mo_v8 -------------------------------------------------
# 6mo_v8: 6-month Tobs, EXACT COPY of 3mo_v8 with
# exactly three families of changes (user spec 2026-09-02):
#
#   1. WARM START (the between-run refit proposal): rj_warm_search runs
#      IMMEDIATELY BEFORE rj_fstat_search in gb_search -- built exactly
#      like fstat_search with the distribution switched out for the
#      3mo_v8-posterior mixture (GB_WARM_START_COMPONENTS npz from the
#      fit -> referee -> apply pipeline; phase-max follows the same
#      GB_RJ_PHASE_MAXIMIZE=1 this file already exports).
#   2. Tobs = 6 months + ONLY its derived settings (same derivation the
#      audited 6mo_v1 script proved): TOBS_TARGET=15552000
#      (Nf 1440 x Nt 4320 x dt 2.5), SIGHET_NT_LAYER=120 (36-h stride
#      parity), GB_N_SUBBANDS=4096/GPU (~2 MB/slot = the 3-mo byte
#      budget), GB_INMODEL_SETUP_BATCH=2048 (stash law LINEAR in
#      N_sparse_t -> half the 3-mo 4096), EDGE_CROP_WAVELETS=60
#      (REQUIRED at 6 mo: sig-het taper 22+8=30 > the default crop 20 --
#      the build guard would raise), GB_NLEAVES_MAX=15000, 6mo naming.
#   3. MBHB + EMRI + SOBHB branches (campaign S6): id-armed via
#      MBHB_IDS/EMRI_IDS/SOBHB_IDS; a source_search stage runs FIRST
#      (sources converge + subtract before the noise stages fit the
#      PSD), then their PE moves ride gb_search + full_pe
#      (sobbh -> mbh -> emri banking order). Knob values lifted from the
#      probe's latest rulings (submit_gf_6mo_sources_probe.sh).
#
#   4. (2026-09-11, user ruling) 10-WALKER REBASE: this run builds off the
#      VALIDATED 10-walker 3mo arm (submit_gf_3mo_v8_10walkers.sh, jobs
#      465/473) rather than the 24-walker base. NWALKERS=10 plus the whole
#      09-08..11 correctness/perf stack that arm carries: PSD shared
#      MIRROR, column-atomic staging + GB_TEMPER_CELL_ORDER=band (the
#      vertical-swap fix set b3f5acbd -- REQUIRED with GB_TEMPER_VERTICAL=1
#      below), windowed sig-het stash + one-block SETUP_BATCH,
#      adaptive gb_search noise rider, PE draw-one (the 09-11 refresh
#      threshold and fused accept kernel were both REVERTED on that arm
#      and are reverted here too -- see items 6). Tobs-scaled here:
#      GB_N_SUBBANDS 16384/GPU and
#      GB_RJ_INMODEL_CHUNK 32768 (byte-parity with the 3mo twin's
#      32768/65536 at half the per-slot/-cell cost -- 6mo slots are ~2x
#      3mo bytes).
#
#   5. (2026-09-14, user ruling) GB OBSERVABLE + EIGENBASIS: this run is
#      the 6mo GB eigenbasis test -- GB_INMODEL_OBSERVABLE_EIGEN=full
#      combines the observable-basis step with the whitened information-
#      matrix eigen table (see the block by GB_INMODEL_OBSERVABLE_SHEAR),
#      and the warm-start refit proposal (change 1) rides along, rebuilt
#      from the MOST RECENT 3mo run (the 10-walker science arm).
#
#   6. (2026-09-14, user ruling) THE 6MO TESTING CAMPAIGN, ON THE
#      COMBINED DATA: "This is going to be our 6mo testing campaign until
#      full running." SOURCE_TYPES=COMBINED,... -- mojito's pre-summed L1
#      stream is the data (see the COMBINED block by the id lists) -- and
#      every post-rebase improvement from the 3mo 10-walker science arm
#      is ported: per-stage RJ flip fractions (search 0.5 / PE 0.1),
#      GB_TEMPER_SKIP_SHUTOFF_BANDS=1, BACKUP_ITER=10, repeats 100/50,
#      GALFOR_ALPHA_MAX=20, and the 09-11 reverts (accept kernel 0,
#      trace 0, ortho premise check 0, anchor audit retired, sig-het
#      refresh threshold back to 0).
#
# V8-PARITY NOTES (deliberate divergences from the old 6mo_v1 draft --
# the exact-copy rule wins for every non-Tobs knob): GB_USE_GALAXY_PRIOR
# stays 1 (6mo_v1 had it 0, an 08-24 test-only ruling). Flip it ONLY
# with a fresh store.
#
# *** VGB_CHIRP_MASS_BASIS IS NOW 1 (user ruling 2026-09-16) -- the
# v8-parity exact-copy pin at 0 is SUPERSEDED. See the VGB RELAUNCH
# BLOCK below for the ruling, the validation numbers, and the LOUD
# resume caveat (vgb chain ndim 5 -> 6: fresh store or migration, never
# mid-store). ***
#
# Staged recipe: source_search -> noise_search -> noise_vgb_search ->
# gb_search -> full_pe
# via scripts/fstat_proposal/run_combined_staged.py  (LAT dev >= the
# staged-sources + warm-phase-max commit; run the wiring tests
# tests/test_staged_sources_wiring.py before first launch)
#
# RESUME: re-submitting this script resumes automatically -- the h5 backend
# restores the last saved iteration and completed stage statuses. Keep the
# same FILE_STORE_DIR/BASE_FILE_NAME and env between submissions. For a
# fresh start, move/delete ${STORE_DIR} first.
#
# ############################################################################
# ## ⚠⚠ 6MO_V8 LAUNCH GATES (do not launch before ALL are green):          ##
# ##                                                                        ##
# ## >>> 3MO NOTE: THE FIRST THREE GATES DO NOT APPLY TO THIS RUN. <<<      ##
# ## No source branch is armed (3MO-4), so the S6 smoke has nothing to      ##
# ## smoke; and there is NO warm start (3MO-2), so the W4 A/B and the       ##
# ## whole GB_WARM_START_COMPONENTS gate below are moot -- the npz is not   ##
# ## built, not read, and not preflighted. Kept as inherited history so     ##
# ## the 6mo lineage stays readable; do not act on them here.               ##
# ##                                                                        ##
# ##   * S6 smoke (docs/6mo-campaign.md): a few staged iterations with     ##
# ##     the source branches armed, THROWAWAY store -- first cluster       ##
# ##     exposure of source_search + sources-in-gb_search.                 ##
# ##   * W4 warm-start A/B: rj_warm_search acceptance healthy on a short   ##
# ##     3-mo rewind before arming here.                                   ##
# ##   * GB_WARM_START_COMPONENTS: the REFEREED npz from the FULL FINAL    ##
# ##     store of the MOST RECENT 3mo run -- the 10-WALKER science arm     ##
# ##     gf_prod_3mo_v8_10walkers (2026-09-14 ruling; never the            ##
# ##     make_snapshots tars). AUTO-BUILT at recipe build when missing     ##
# ##     (fit -> referee -> apply from GB_WARM_START_SOURCE_STORE, run     ##
# ##     IN-PROCESS via imports; watch the [WARMSTART-BUILD] lines) --     ##
# ##     the 09-14 first launch died on the old hard preflight refusal,    ##
# ##     hence the automation. The npz lives INSIDE ${STORE_DIR}           ##
# ##     (warmstart/ subdir) so it travels with the run.                   ##
# ##   * GB OBSERVABLE+EIGEN (2026-09-14): first 6mo exposure of           ##
# ##     GB_INMODEL_OBSERVABLE_EIGEN=full -- read the 3mo probe            ##
# ##     (submit_gf_3mo_v8_10w_eigenaxis_probe.sh) first; empty knob =     ##
# ##     bit-identical fallback. Tests: tests.test_gb_observable_eigen.    ##
# ##   * EIGEN INNER MOVES (2026-09-08): first cluster exposure of the     ##
# ##     EigenAxisMove defaults pinned below (sobbh per-walker tables,     ##
# ##     mbh/emri max-lnL-walker tables). Requires Eryn dev >= the        ##
# ##     eryn.moves.eigenaxis commit PULLED TOGETHER WITH LAT (LAT dev     ##
# ##     imports it). Watch the "[eigen_refresh]" WARNING lines; the       ##
# ##     stretch escape is {BRANCH}_INNER_MOVE_KIND=stretch.               ##
# ##   * git pull BOTH repos + ./install.sh (native headers) + the tests:  ##
# ##       python -m unittest tests.test_staged_sources_wiring \          ##
# ##           tests.test_eigen_refresh tests.test_inner_move_kind \      ##
# ##           tests.test_vgb_eigen_inmodel tests.test_vgb_ridge_gibbs \  ##
# ##           tests.test_vgb_observable_basis tests.test_ridge_fiber      ##
# ## Store gf_prod_6mo_v8/ is NEW -- nothing to migrate; the RELAUNCH      ##
# ## block below is 3mo_v8 history only.                                   ##
# ##                                                                        ##
# ## ⚠ 2026-09-16 VGB=GB PARITY: VGB_CHIRP_MASS_BASIS 0 -> 1 (vgb ndim     ##
# ## 5 -> 6) + the new vgb_ridge_gibbs move. If gf_prod_6mo_v8/ has        ##
# ## ALREADY been launched on the 5-dim basis, this is NOT a plain resume: ##
# ## run.py refuses it at backend construction naming                       ##
# ## VGB_CHIRP_MASS_BASIS. TWO ways forward:                                ##
# ##                                                                        ##
# ##  (a) KEEP THE RUN, REBORN VGBs (user ruling 2026-09-16, PREFERRED --  ##
# ##      "keep the hdf backend except for the VGBs (and just start them   ##
# ##      at injection again)"):                                            ##
# ##        python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \     ##
# ##            <store.h5> <catalogue_dir>                                  ##
# ##      DEFAULT mode = --restart-at-injection. Everything except the vgb ##
# ##      branch survives byte-for-byte -- GB chains + band state, noise,  ##
# ##      galfor, mbh/emri/sobbh, log_like, the ITERATION COUNTER -- so    ##
# ##      the GB search keeps every iteration of progress. The vgb branch  ##
# ##      is discarded and reseeded at catalogue truth in the 6 columns.   ##
# ##      It also quarantines the stale 5-column sidecars (the midit       ##
# ##      checkpoint and the running backup copy, either of which would    ##
# ##      otherwise silently undo the migration) into                       ##
# ##      pre_chirp_migration/ next to the store.                           ##
# ##                                                                        ##
# ##  (b) FRESH STORE -- also fine, but pays back the whole GB search.     ##
# ##      The eigen sidecar and the warm-start npz SURVIVE either way      ##
# ##      (they are gb/sobbh/mbh/emri artifacts, not vgb), and the vgb     ##
# ##      chain restarts at exact truth in both routes under               ##
# ##      VGB_START_FACTOR=0.                                               ##
# ############################################################################
# ############################################################################
# ## ⚠ RELAUNCH REQUIRED (2026-08-29): CAP GRID 1 -> 2 + STAGGER.          ##
# ##                                                                        ##
# ## GB_CAP_DIVISOR=1 -> 2 and GB_CAP_STAGGER=0 -> 1 CHANGE cap_edges, and  ##
# ## cap_edges is one of GBState.static_names, so the resume guard          ##
# ## (GBState.initialize_band_information) REFUSES a plain resume with      ##
# ## "leaf-cap grid mismatch". A rewind + migration is required; rewinding  ##
# ## FIRST is what makes the migration exact rather than approximate.       ##
# ##                                                                        ##
# ##   S=/shared/data/global_fit_output/gf_prod_3mo_v8/gf_prod_3mo_testing.h5
# ##                                                                        ##
# ##   scancel <jobid>            # stop the live job FIRST                 ##
# ##   cd <repo> && git pull                                                ##
# ##                                                                        ##
# ##   # 1. rewind GB to zero leaves (dry run prints, then --apply)         ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \                ##
# ##       "$S" gb_search --rewind-to-empty gb                              ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \                ##
# ##       "$S" gb_search --rewind-to-empty gb --apply                      ##
# ##                                                                        ##
# ##   # 2. rewrite the cap grid in place. REFUSES if a .bak exists.        ##
# ##   [ -e "$S.bak" ] && mv "$S.bak" "$S.bak.pre-stagger"                  ##
# ##   python scripts/fstat_proposal/migrate_gb_cap_grid.py \               ##
# ##       "$S" --cap-divisor 2 --stagger --dry-run                         ##
# ##   python scripts/fstat_proposal/migrate_gb_cap_grid.py \               ##
# ##       "$S" --cap-divisor 2 --stagger                                   ##
# ##                                                                        ##
# ##   # 3. the rewind invalidates all THREE sidecars -- the backup copy    ##
# ##   #    sits at a LATER iteration and a truncation restore would        ##
# ##   #    silently UNDO the rewind; the fstat cache was fitted against a  ##
# ##   #    residual that removing every GB leaf just invalidated.          ##
# ##   python scripts/fstat_proposal/compact_gf_store.py "$S" \             ##
# ##       --reset-backup --reset-midit --reset-fstat --apply               ##
# ##                                                                        ##
# ##   # 4. resubmit                                                        ##
# ##   sbatch scripts/fstat_proposal/submit_gf_3mo_v8.sh                    ##
# ##                                                                        ##
# ## A FRESH STORE ALSO WORKS and needs no migration, but pays ~1-2 h to    ##
# ## refit the noise + VGB stages. NOTE v8's noise stages are the expensive ##
# ## part (unequal-arm + layer_calibrated), so the rewind is worth more     ##
# ## here than in v7.                                                       ##
# ##                                                                        ##
# ## WHY REWIND BEFORE MIGRATING. migrate_gb_cap_grid inherits per-cell     ##
# ## state from the OWNING band (cell c -> band c // K), exact for interior ##
# ## cells but approximate for the one straddling cell per seam -- and with ##
# ## live GB leaves, sources within half a cell of a seam silently change   ##
# ## cells. Rewound to zero leaves there is no occupancy to reattribute.    ##
# ## The ramp state follows: band_leaf_cap / cap_cell_leaf_cap /            ##
# ## cap_cell_iters are PER-ITERATION datasets (only band_edges and         ##
# ## cap_edges are static), so the rewind returns them to the pre-search    ##
# ## handover values and the migration broadcasts THOSE to the children.    ##
# ############################################################################
#
# ############################################################################
# ## V5 (2026-08-20) -- THE STAGGERED-GRID RUN.                             ##
# ##                                                                        ##
# ## One structural change on top of everything v4 learned:                 ##
# ##                                                                        ##
# ## * STAGGERED CAP-CELL GRID (GB_CAP_STAGGER=1, user design 2026-08-20).  ##
# ##   Every interior cap edge shifts by half a cell, so NO cap edge        ##
# ##   coincides with a sub-band edge: the leaf-cap seams and the band      ##
# ##   seams (serial-within-band scheduling, F-stat fit interior, band      ##
# ##   shutoff) share no equivalent boundary, and no source can sit on      ##
# ##   both at once. Cells at band seams straddle them; storage sizes,      ##
# ##   index arithmetic and the monitor are unchanged (LAT tests           ##
# ##   tests/test_cap_stagger.py pin arithmetic == searchsorted exactly).   ##
# ##                                                                        ##
# ## Also explicitly pinned (both are code defaults since 8d926f27, run    ##
# ## here for the first time in a full 3-month production):                 ##
# ##   * BIRTH FIX (1274a66c): RJ births draw fdot_astro_ratio | (f0, Mc)  ##
# ##     tight around the F-stat grid fdot instead of U[-5,5] -- the       ##
# ##     high-f mosaic root cause.                                          ##
# ##   * RIDGE-GIBBS (8d926f27 / Eryn 6ed5a8b): zero-likelihood resample   ##
# ##     along the exact Mc^(5/3)(1+r)=const ridge -- unfreezes the        ##
# ##     (Mc, r, dist) marginals. REQUIRES the GFRidgeGibbsMove sub-state   ##
# ##     cold-row write-back (2026-08-21 fix): the plain eryn move updated  ##
# ##     ONLY the main engine state, so the first-launch runs MPI-aborted   ##
# ##     at gb_search it=2 on the coords-mismatch consistency check.        ##
# ##                                                                        ##
# ############################################################################
# ## V6 (2026-08-20) -- THE SUB-BAND SHRINKAGE RUN.                         ##
# ##                                                                        ##
# ## ONE variable against v5 (user ruling): the SUB-BAND size. Everything  ##
# ## else -- staggered cap grid, birth fix, ridge-Gibbs, cap drift gate,   ##
# ## every other knob -- is byte-identical to submit_gf_3mo_v5.sh.         ##
# ##                                                                        ##
# ## * GB_SUBBAND_DIVISOR=8: uniform bands of layer/8 = 135 FD bins        ##
# ##   (1232 bands vs v5's 154). The band is a SCHEDULING unit, not a      ##
# ##   containment unit (user ruling: waveforms already extend past band   ##
# ##   edges; the slabs carry max(leakage, FD-support) margins) -- the     ##
# ##   dense-band 30-40-source serial chains become ~4-5 sources/band,    ##
# ##   and per-band tempering ladders live at 1/8-layer granularity.       ##
# ## * GB_BAND_UNIT_STRIDE=9: same-unit gap = 8 x layer/8 = exactly ONE    ##
# ##   LAYER -- the separation production's stride-2-on-1-layer grid has   ##
# ##   always run with (the conservative FD-support envelope was already   ##
# ##   violated there by design; [GB_ORTHO_LL], default ON, remains the    ##
# ##   operative independence monitor). 1232/9 ~ 137 concurrent bands per  ##
# ##   unit, MORE than v5's 77 -- concurrency is preserved, the serial     ##
# ##   chains shrink 8x, the pass runs 9 units instead of 2.               ##
# ## * GB_CAP_DIVISOR=4 (v5: 32): K scales so the CAP-CELL GRID IS         ##
# ##   BIT-IDENTICAL to v5's -- (layer/8)/4 = layer/32 cells, and the      ##
# ##   staggered edge set lands on the same layer*(n+0.5)/32 points. The   ##
# ##   cap variable is fully controlled; only the band grid moves.         ##
# ##                                                                        ##
# ## START: FRESH STORE ONLY -- the BAND grid changes, which no rewind or  ##
# ## migration handles (three migration attempts failed on band-grid       ##
# ## cascades; the resume guard refuses). Noise stages rebuild (~1-2 h):   ##
# ##   rm -rf gf_prod_3mo_v6                                                ##
# ##   git pull && sbatch scripts/fstat_proposal/submit_gf_3mo_v6.sh        ##
# ############################################################################
# ## START (user ruling 2026-08-20): REWOUND v4 COPY + CAP-GRID MIGRATION. ##
# ## The staggered edges differ from the v4 store's, and the resume guard  ##
# ## refuses a mismatched cap_edges array -- so after the rewind, rewrite  ##
# ## the (empty-GB) cap grid in place with the migrate script:             ##
# ##                                                                        ##
# ##   git pull                                                             ##
# ##   cp -r gf_prod_3mo_v4 gf_prod_3mo_v5                                  ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \                ##
# ##       gf_prod_3mo_v5/gf_prod_3mo_testing.h5 gb_search \                ##
# ##       --rewind-to-empty gb --apply                                     ##
# ##   python scripts/fstat_proposal/migrate_gb_cap_grid.py \               ##
# ##       gf_prod_3mo_v5/gf_prod_3mo_testing.h5 \                          ##
# ##       --cap-divisor 32 --stagger                                       ##
# ##   # verify readable, then STAMP THE BACKUP so self-heal can never      ##
# ##   # resurrect the pre-rewind, pre-migration state:                     ##
# ##   python -c "import h5py; h5py.File(                                   ##
# ##       'gf_prod_3mo_v5/gf_prod_3mo_testing.h5','r').close()"            ##
# ##   cp gf_prod_3mo_v5/gf_prod_3mo_testing.h5 \                           ##
# ##      gf_prod_3mo_v5/gf_prod_3mo_testing_running_backup_copy.h5         ##
# ##   sbatch scripts/fstat_proposal/submit_gf_3mo_v5.sh                    ##
# ##                                                                        ##
# ## (A genuinely fresh STORE_DIR also works -- the noise stages then       ##
# ## rebuild from scratch and no migration is needed.)                      ##
# ############################################################################
# ############################################################################
# ## V4 (2026-08-18) -- THE IN-MODEL CORRECTNESS RUN.                       ##
# ##                                                                        ##
# ## v3 could not refine f0: the GB ensemble's between-walker scatter was   ##
# ## 0.798 Fourier bins against 0.051 within-walker. Root-caused to the     ##
# ## in-model proposal covariance, broken two independent ways, both        ##
# ## silent (LAT 09687e4b):                                                 ##
# ##   * the fdot CONDITIONING scale 1e-16 was resolved by matching         ##
# ##     ("fdot","Mc"), so on the distance/chirp-mass basis it landed on    ##
# ##     the Mc column (natural scale O(0.1-1)). That drove the Mc          ##
# ##     eigenvalue under the eigen-floor and the Mc step came out at       ##
# ##     1.27e-15 x the true posterior width. Mc never moved.               ##
# ##   * the physical->sampling Jacobian was DIAGONAL on a map that is not  ##
# ##     separable, so Mc lost its amplitude term and fdot_astro_ratio got  ##
# ##     EXACTLY zero curvature (which is why its proposal row had to be    ##
# ##     zeroed). Now the exact congruence J^T Gamma J.                     ##
# ## Verified against the real GB likelihood: all 9 columns 1.000 (was      ##
# ## 8.5e-40 for Mc, 0 for the ratio). CONFIRMED IN PRODUCTION: in-model    ##
# ## infomat cold acceptance 0.31-0.38 at GB_JUMP_FACTOR=1.2, against 0.95  ##
# ## under the broken covariance. Do NOT retune the jump factor.            ##
# ##                                                                        ##
# ## Also in v4:                                                            ##
# ##   * LAT 71c0bbd1 -- the multi-shard router returned None, so on >= 2   ##
# ##     GPUs sighet_active was ALWAYS False and the anchor check, the ll   ##
# ##     audit, the reference refresh and the trust gate were all silently  ##
# ##     dead. Every diagnostic below exists because of this fix.           ##
# ##   * GB group stretch OFF (c3725f7d): measured cold 2/472 = 0.0042 on   ##
# ##     v3 while the VGB stretch scored 0.4485 on the same run.            ##
# ##   * SIGHET_NT_LAYER stays at the v3 default (36 h). 270 was tried and  ##
# ##     REVERTED: it OOMs (the sig-het stash goes as cells x N_sparse_t)   ##
# ##     and the accuracy gain did not reproduce once the config echo made  ##
# ##     the measurement attributable. See the block at SIGHET_NT_LAYER.    ##
# ##   * F-stat peak weighting flattens to w ~ sqrt(SNR) after the first    ##
# ##     refit (62cd814e), plus a fix for alpha being silently dropped on   ##
# ##     the stage-B reload path.                                          ##
# ##                                                                        ##
# ## START: this expects a COPY of the v3 store rewound to zero GB leaves,  ##
# ## so the fitted noise, the VGB ladder and the F-stat epoch cache are     ##
# ## inherited and only the GB search re-runs:                              ##
# ##   cp -r gf_prod_3mo_v3 gf_prod_3mo_v4                                  ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \                ##
# ##       gf_prod_3mo_v4/gf_prod_3mo_testing.h5 gb_search \                ##
# ##       --rewind-to-empty gb --apply                                     ##
# ## For a genuinely fresh start instead, just point STORE_DIR at an empty  ##
# ## dir -- every piece of state then rebuilds from this file.              ##
# ##                                                                        ##
# ## Deploy:  git pull && sbatch scripts/fstat_proposal/submit_gf_3mo_v4.sh ##
# ############################################################################
# ============================================================================

# ############################################################################
# ## V8 (2026-08-27) -- THE UNEQUAL-ARM NOISE RUN.                          ##
# ##                                                                        ##
# ## v8 = v7 with the noise-dev merge wired in. The GB/VGB/F-stat side is   ##
# ## UNCHANGED from v7; the whole diff is the noise model:                  ##
# ##                                                                        ##
# ## * UNEQUAL-ARM INSTRUMENT NOISE (UNEQUAL_ARM=1): the equal-arm          ##
# ##   InstrumentNoise is swapped for UnequalArmInstrumentNoise -- six      ##
# ##   independent link light travel times read from the mojito NOISE       ##
# ##   brick's /ltts group (LinkDelayTable, stride 200), averaged per WDM   ##
# ##   time column, anchored at the run's data_t0 by the engine. The data   ##
# ##   was always generated with breathing unequal arms; the model now      ##
# ##   matches it. Complex Hermitian cross-spectra; WDM keeps Re[C_ij].     ##
# ## * WDM_PSD_METHOD=layer_calibrated: one exact fold pins a per-layer     ##
# ##   correction to the ~200x cheaper layer-center evaluation (worst-case  ##
# ##   basis error 9.7e-3 -> 1.2e-6 on the 2-yr grid). It self-checks its   ##
# ##   validity near Nyquist and warns; at MAX_FREQ=2.5e-2 vs a 0.2 Hz      ##
# ##   Nyquist we are at 12.5% -- comfortably safe.                         ##
# ## * TABULATED FOREGROUND MODULATION (GALFOR_MODULATION_PATH =            ##
# ##   scripts/noise/modulation_unequal.dat, GALFOR_MODULATION_T0=data):    ##
# ##   per-element time modulation from the GLASS anisotropy fit, on the    ##
# ##   absolute mission clock, anchored at data_t0 after processing.        ##
# ## * GALFOR PRIOR CEILINGS IN BAND (merge ea7790f3): fk/f_1/f_2 <= 1e-2   ##
# ##   Hz. The old slope-unit plateau cost the 2-yr fit its posterior       ##
# ##   (whitening 0.983 -> 1.0002 with f_1 in band).                        ##
# ##                                                                        ##
# ## FRESH STORE (MANDATORY): a resume across a noise-model change is a     ##
# ##   different likelihood on identical shapes. The run itself now         ##
# ##   persists a noise-model identity and REFUSES such a resume; this      ##
# ##   script also preflights it before eating a slurm allocation.          ##
# ## * COARSE NOISE LIKELIHOOD, delayed acceptance (Q=8, WS). EXACT: the   ##
# ##   fine likelihood stays the sampled target in every stage. Measured    ##
# ##   ~25% off the noise block (psd_pe 5.87 -> 2.32 s; galfor_pe 5.12 ->   ##
# ##   5.92 s, i.e. galfor REGRESSED -- its band-limited fast path is not   ##
# ##   wired into the all-source sidecar yet; Robbie to revisit).           ##
# ##                                                                        ##
# ## VALIDATED ON THE CLUSTER (probe jobs 369 exact-fine / 376 coarse):     ##
# ##   * delay table: 126,233 epochs @ stride 200 over [9.77e7, 1.61e8] s,  ##
# ##     anchored at data_t0=9.772994e7, digest f1f3f00ea5d9cf13;           ##
# ##   * modulation: 199 epochs covering [0, 6.31e7] s of the data frame;   ##
# ##   * layer_calibrated drift 6.320e-07 -- THREE ORDERS below the 1e-4    ##
# ##     tolerance, correction spanning [0.99996, 1.08653] over 3240/3240   ##
# ##     entries (so it is doing real work AND is comfortably in regime);   ##
# ##   * both GPUs active; noise-window peaks 26.3 / 7.8 GB.                ##
# ##                                                                        ##
# ## WATCH ON FIRST LAUNCH:                                                 ##
# ##   * "[unequal-arm] link-delay table ..." line: stride/epochs/digest;   ##
# ##   * "[galfor-modulation] ... anchored at data_t0" line;                ##
# ##   * "coarse WDM sidecar runtime (all-source, mode=...)" -- ONE line,   ##
# ##     confirming Q, Ncoarse, weighting and the fiducial digest;          ##
# ##   * "[COARSE_AUDIT ...]" per propose: stage-2 acceptance and the       ##
# ##     |dlogl| spread ARE the surrogate-accuracy metric (0 / 100% = an    ##
# ##     exact surrogate). If |dlogl| is large or stage-2 acceptance is     ##
# ##     poor, lower COARSE_Q -- accuracy of the SAMPLED chain is not at    ##
# ##     risk either way (delayed acceptance is exact), only efficiency;    ##
# ##   * the layer_calibrated validity warning must NOT fire (it did not    ##
# ##     in 369/376; the only occurrences were inside the probe's own       ##
# ##     unit-test block, on an unrestricted toy grid);                     ##
# ##   * one-time basis build cost (~106 MiB/device at the 6-mo grid;       ##
# ##     smaller here) before the first noise iteration.                    ##
# ##                                                                        ##
# ## OPEN PHYSICS WATCH -- GALFOR RAILING. Both probes drove the foreground ##
# ##   toward prior edges while GB is still EMPTY (the galaxy is entirely   ##
# ##   unsubtracted, so galfor absorbs it): 369 had f_1 climbing to 1.4e-6  ##
# ##   below its 1e-2 ceiling; 376 had amp pinned at its 1e-41 ceiling and  ##
# ##   f_2 at its 1e-5 floor. This is NOT settled as a modelling fault --   ##
# ##   it may simply be the search doing its job against an unsubtracted    ##
# ##   galaxy. RE-CHECK once gb_search populates leaves: if galfor is still ##
# ##   railed after GB sources are subtracted, the ceilings are binding and ##
# ##   the 2-yr plateau lesson is repeating at a new location.              ##
# ## Whitening test (scripts/noise/whitening_test.py) on the first          ##
# ## snapshot is the acceptance metric for the noise side.                  ##
# ############################################################################

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf3mo_v9_2gpu     # job name
#SBATCH --partition=gpu-80-ondemand   # DEFAULT partition (2-GPU flow); the
                                  # NGPUS self-dispatch below overrides it
#SBATCH --gres=gpu:2              # DEFAULT 2 GPUs (GPUS below are LOCAL indices)
#SBATCH --nodes=1                 # DEFAULT 1 node; NGPUS=4 -> 2 nodes (below)
#SBATCH --ntasks=3                # DEFAULT/fallback only (legacy layout: main
                                  # + stopped spare + SAVER rank, mpiexec -n 3);
                                  # the NGPUS self-dispatch below computes the
                                  # real rank count from GPUS_PER_RANK/
                                  # RANKS_PER_GPU/GF_LEGACY_RANK_LAYOUT and
                                  # passes --ntasks explicitly on `exec sbatch`
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=/shared/data/global_fit_output/gf3mo_v9_2gpu_%j.log     # combined stdout+stderr (captures [MAXLOGL]/[BENCH])
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- GPU-count self-dispatch + rank-layout knobs (2026-09-16: rank count
# ---- now derives from the GPU count) ---------------------------------
# NGPUS=2 -> gpu-80-ondemand, 1 node x 2 GPUs (the default here);
# NGPUS=4 -> gpu-80-ondemand, 2 NODES x 2 GPUs each -- the real 4-GPU
# allocation shape (there is no single 4-GPU node; see the design spec's
# "Context" section). #SBATCH lines are static comments, so neither the
# partition/node/task count can follow an env var through a plain
# `sbatch <script>`. Instead, run this script DIRECTLY to pick the GPU
# count and it submits itself with the matching flags:
#
#     NGPUS=2 ./submit_gf_3mo_v9_2gpu.sh # THE INTENDED INVOCATION
#                                        # (gpu-80-ondemand, 1 node x gpu:2)
#     NGPUS=4 ./submit_gf_3mo_v9_2gpu.sh # 2 NODES x 2 GPUs, on-demand
#     sbatch  ./submit_gf_3mo_v9_2gpu.sh # legacy flow: header defaults above
#                                        # (2 GPUs, gpu-80-ondemand, --ntasks=3)
#
# ⚠ THIS RUN IS THE FIRST PRODUCTION SHAPE WITH NWALKERS > NGPUS. At NGPUS=2
# the walker-block layout gives N_COMPUTE = 1 node x 2 GPUs x RANKS_PER_GPU 1
# / GPUS_PER_RANK 1 = 2 compute ranks plus a saver (--ntasks=3), and
# NWALKERS=4 divides cleanly at TWO walkers per rank. Consequences to expect
# rather than discover: per-rank rows DOUBLE against the 6mo 4-GPU shape
# (1 walker/rank there), total resident sub-band slots halve (GB_N_SUBBANDS is
# PER GPU, so per-GPU memory is unchanged and only the pass count moves).
# BOTH GPU counts run ON-DEMAND here, so a silent preemption is no longer
# the expected way for this run to die; MIDIT_CHECKPOINT stays on anyway.
# GF_LAYOUT_DRY_RUN=1 prints every rank's placement and stops before the
# build allocates anything.
#
# GPUS_PER_RANK (empty = AUTO) / RANKS_PER_GPU (default 1) size the compute
# rank count: N_COMPUTE = NGPUS * RANKS_PER_GPU / GPUS_PER_RANK.
#
# LAYOUT DEFAULT (user ruling 2026-09-16): the walker-block layout
# (GF_LEGACY_RANK_LAYOUT=0) is the default at EVERY NGPUS. At NGPUS=2 that is
# head + 1 compute rank + saver (--ntasks=3, one GPU each, NWALKERS/2 walkers
# per rank); the WP7 transport gates (Steps 0-2, 4) passed on the cluster and
# the 6mo campaign run itself is the statistical read (Step 3 was skipped by
# ruling). GF_LEGACY_RANK_LAYOUT=1 still selects today's single-compute-rank
# layout for a 1-node job (rollback knob). NGPUS=4 cannot use the legacy
# layout (it cannot span nodes), so it forces GF_LEGACY_RANK_LAYOUT=0
# regardless of any pre-set value.
#
# Inside the job, the GPU list further below derives from what slurm
# ACTUALLY granted (SLURM_GPUS_ON_NODE), so a manual
# `sbatch --partition=gpu-80-ondemand --gres=gpu:2 --nodes=2 <script>` also
# works.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  NGPUS=${NGPUS:-2}
  GPUS_PER_RANK=${GPUS_PER_RANK:-}
  RANKS_PER_GPU=${RANKS_PER_GPU:-1}
  _k=${GPUS_PER_RANK:-1}
  case "${NGPUS}" in
    # ON-DEMAND AT EVERY GPU COUNT (user ruling 2026-09-26). The 6mo script
    # sends NGPUS=2 to spot because a 1-node job is cheap to lose and the
    # midit checkpoint recovers it. THIS RUN HAS ALREADY PAID THAT BILL:
    # job 633 ran 43 minutes of gb_search_1 on gpu-80-spot and was killed
    # with NO traceback, no abort marker and no further saved iteration --
    # the log simply stops mid-move, and the relaunch (job 636) had to pick
    # up from stored iteration 1 on ondemand anyway.
    # PARTITION=<name> overrides either row without editing this file.
    2) _NGPU_PART=${PARTITION:-gpu-80-ondemand}; _NODES=1; _GRES=gpu:2 ;;
    # NGPUS=4 -> ON-DEMAND (user ruling 2026-09-18). The 4-GPU shape is
    # 2 nodes x 2 GPUs, and on spot a preemption of EITHER node kills the
    # whole MPI world -- twice the exposure of the 1-node flow for the
    # same work. NGPUS=2 stays on spot: it is a single node and the
    # midit-checkpoint path recovers it cheaply.
    4) _NGPU_PART=${PARTITION:-gpu-80-ondemand}; _NODES=2; _GRES=gpu:2 ;;
    *) echo "[SUBMIT] NGPUS=${NGPUS} unsupported (2 or 4)."; exit 2 ;;
  esac
  # NODES=<n> spreads the NGPUS GPUs over n nodes (gres = NGPUS/n per node).
  # One-walker replica gates (2026-09-16 ruling: test ACROSS nodes):
  # NGPUS=2 NODES=2 -> 2 nodes x gpu:1, one replica per node;
  # NGPUS=4 NODES=4 -> 4 nodes x gpu:1. Unset = the NGPUS table above.
  NODES=${NODES:-}
  if [ -n "${NODES}" ]; then
    if [ "${NODES}" -lt 1 ] || [ $(( NGPUS % NODES )) -ne 0 ]; then
      echo "[SUBMIT] NODES=${NODES} must be >= 1 and divide NGPUS=${NGPUS}."; exit 2
    fi
    _NODES=${NODES}; _GRES=gpu:$(( NGPUS / NODES ))
  fi
  if [ "${_NODES}" -gt 1 ]; then
    GF_LEGACY_RANK_LAYOUT=0    # the legacy single-compute-rank layout cannot span nodes
  else
    GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
  fi
  export GF_LEGACY_RANK_LAYOUT
  N_COMPUTE=$(( NGPUS * RANKS_PER_GPU / _k ))
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
    NTASKS=3
  else
    NTASKS=$(( N_COMPUTE + 1 ))
  fi
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
    echo "[SUBMIT] GF_LEGACY_RANK_LAYOUT=1: TODAY's roles (one sampling rank"
    echo "[SUBMIT]   drives all local GPUs; rank 1 stopped spare; rank 2"
    echo "[SUBMIT]   saver) -- set GF_LEGACY_RANK_LAYOUT=0 for the"
    echo "[SUBMIT]   walker-block layout after the WP7 gates."
  else
    echo "[SUBMIT] GF_LEGACY_RANK_LAYOUT=0: walker-block layout: ${N_COMPUTE}"
    echo "[SUBMIT]   compute ranks + 1 saver."
  fi
  echo "[SUBMIT] NGPUS=${NGPUS} -> sbatch --partition=${_NGPU_PART} --gres=${_GRES} --nodes=${_NODES} --ntasks=${NTASKS} (N_COMPUTE=${N_COMPUTE} compute ranks + 1 saver)"
  _DIST_FLAG=""
  if [ "${_NODES}" -gt 1 ]; then
    _DIST_FLAG="--distribution=cyclic"
  fi
  exec sbatch --partition="${_NGPU_PART}" --gres="${_GRES}" --nodes="${_NODES}" \
       --ntasks="${NTASKS}" ${_DIST_FLAG} \
       --export=ALL,NGPUS="${NGPUS}",GPUS_PER_RANK="${GPUS_PER_RANK}",RANKS_PER_GPU="${RANKS_PER_GPU}",GF_LEGACY_RANK_LAYOUT="${GF_LEGACY_RANK_LAYOUT}" \
       "$0" "$@"
fi

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

# FRESH RUN (2026-08-15, user ruling: "we want to totally restart. This is a
# fresh run now."). A NEW store dir so the previous run's h5/logs/fstat cache
# stay intact for comparison and nothing can silently resume. BASE_FILE_NAME
# stays gf_prod_3mo so every analysis tool (monitor generator, digests) works
# unchanged -- they take the DIRECTORY as their argument.
# env-overridable (2026-09-16) so a COPY of a live run's folder can be continued
# under new code/layout without touching the original (STORE_DIR=<copy> ...).
STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_3mo_v9_2gpu/}

# ---- GPU telemetry ---------------------------------------------------------
# Background nvidia-smi sampler: one CSV row per GPU into the run store
# (timestamped per job, so resubmits/resumes add new files rather than
# clobbering). Killed automatically when the job exits. Columns:
# timestamp, index, name, util.gpu [%], util.mem [%], mem.used [MiB],
# mem.total [MiB], power [W], temp [C].
#
# INTERVAL 30 -> 5 s (2026-08-15): 30 s was tuned for 55-minute iterations.
# Post-speedup an iteration is ~5 min and individual moves take SECONDS, so
# 30 s gave ~10 samples/iteration -- far too coarse to attribute utilization
# to a move. At 5 s a 5-min iteration yields ~60 samples/GPU. Cost is
# negligible (a few hundred KB/hour; nvidia-smi polling is cheap) and the
# monitor's gpu_util panel reads it unchanged.
mkdir -p ${STORE_DIR}
GPU_SAMPLE_SEC=${GPU_SAMPLE_SEC:-5}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l ${GPU_SAMPLE_SEC} > "${GPU_LOG}" &
GPU_SMI_PID=$!
# PER-PROCESS GPU memory sampler. --query-gpu reports only DEVICE totals, so
# it cannot say WHICH process holds the memory. Under `mpiexec -n 3` that is
# exactly the open question: run_combined_staged.py builds on EVERY rank
# before roles resolve, so the saver/spare ranks may be holding GPU
# allocations they never use. Columns: timestamp, gpu_uuid, pid,
# process_name, used_memory [MiB] -- one row per process per GPU. Sampled
# 6x slower than the utilization stream (this changes only when a process
# allocates).
GPU_PROC_LOG=${STORE_DIR}/gpu_procs_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits -l $((GPU_SAMPLE_SEC * 6)) > "${GPU_PROC_LOG}" &
GPU_PROC_PID=$!

# ---- slurm stdout MIRROR into the store dir --------------------------------
# Ported from submit_gf_6mo_sources_probe.sh (noise-merge readout item: the
# multi-GPU probe zip carried NO slurm log, so [GF_TIMING]/[MAXLOGL]/
# [PROBE]/[SMOKE] never travelled). Everything that matters for the timing
# readout goes to STDOUT -- i.e. the --output file above, which lives
# OUTSIDE ${STORE_DIR} -- and the pulls are zips OF ${STORE_DIR}, so that
# file has been missing from every pull. Mirror it in every 30 s: zipping
# the store then captures it automatically, and because this is a copy loop
# rather than an EXIT trap it survives a spot preemption (SIGKILL runs no
# traps). Also note slurm stdout only FLUSHES at job end, so the mirror is
# the only way to see these lines while the job is still running.
SLURM_LOG=/shared/data/global_fit_output/gf3mo_v9_2gpu_${SLURM_JOB_ID:-manual}.log
LOG_MIRROR_PID=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ( while true; do
      cp -f "${SLURM_LOG}" "${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log" 2>/dev/null || true
      sleep 30
    done ) &
  LOG_MIRROR_PID=$!
  echo "[LOGMIRROR] ${SLURM_LOG} -> ${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log every 30 s"
fi

trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} ${LOG_MIRROR_PID:-} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) -----------------------------------
export OMP_NUM_THREADS=1

# ---- HDF5 FILE LOCKING (2026-08-15 hang forensics) --------------------------
# Job 210 hung at STARTUP for 11 h: allocation alive, CUDA contexts created
# (10 GB held), 0% GPU on both devices, and the MAIN rank never logged a
# single line -- while a healthy start logs within ~37 s. The first thing the
# main rank does after build() is load_info() -> backend.get_last_sample(),
# i.e. OPEN AND READ the store h5. The previous job had been writing that same
# file 3.5 min earlier, and h5py BLOCKS INDEFINITELY (it does not error) when
# it cannot take the lock -- which is the classic failure on NFS/Lustre, and
# is made likelier here by the saver rank holding the file from a second
# process. Disabling HDF5's own locking is the standard remedy for shared
# filesystems; this run is the only writer, so the lock buys nothing.
export HDF5_USE_FILE_LOCKING=FALSE

# ---- console verbosity ------------------------------------------------------
# WHERE THE DETAIL ALREADY LIVES: the file handler is pinned at DEBUG
# UNCONDITIONALLY, so every per-iteration line ("Number of active leaves
# before proposal", "Current number of active sources in cold chain",
# the [GB_TIMING]/[GB_ACCEPT]/[FSTAT_CTR] records, ...) is ALWAYS written to
#   ${STORE_DIR}/${BASE_FILE_NAME}_artifacts/globalfit_run.log
# regardless of this knob. VERBOSE only MIRRORS them to stdout (the sbatch
# .log). Setting it in the submitting shell is unreliable (it depends on
# sbatch --export propagation), so pin it here.
# Costs no compute: the messages are formatted either way; this only adds a
# second handler writing the same text.
export VERBOSE=1
# PROGRESS defaults to "follow VERBOSE", which would start a tqdm bar. In a
# non-tty sbatch log tqdm emits a line per update and buries the real
# output, so pin it off; VERBOSE stays purely about the log lines.
export PROGRESS=0

# ---- run plumbing ----------------------------------------------------------
export MOJITO_DATA_PATH=/shared/data/mojito_cache
# The SAME directory, named the way the monitor page asks for it: the
# level holding catalogues/ and data/. The page needed two knobs for
# this one path and the second was never exported, so a page built on
# the cluster silently lost its residual-spectrum and
# data/template/residual panels. Sampler never reads it.
export MOJITO_INFO_PATH=/shared/data/mojito_cache
# ---- MID-RUN MONITOR (user request 2026-09-27) -----------------------------
# The saver rank (walkers=[0,0), no sampling work) builds the page and the
# snapshot tar in the quiet gap after a save, so the artifact tracks the run
# instead of waiting for a manual pull. Outputs land BESIDE the run folder:
#   <STORE_DIR>/<run>_monitor.html   and   <run>_snapshot.tar.gz
#
# ⚠ THE HOOK CAN DISABLE ITSELF, BY DESIGN. It measures every build against
# the run's own save cadence and, if a build ever causes the saver to DROP a
# state -- which would lose a stored iteration permanently, since the saver
# coalesces to the newest payload when its queue backs up -- it prints a
# banner and stays off for the rest of the run. The run is never affected.
# If that fires, raise GF_MONITOR_ITER or set GF_MONITOR_SNAPSHOT=0.
# 3MO IS TAR-ONLY (2026-09-28). The hook is back ON; the PAGE is off.
#
# ROOT CAUSE, now KNOWN and fixed by a4: the page renders IN-PROCESS
# (build_monitor defaults to in_process=True, the 09-26 "no
# subprocesses!" ruling), and build_monitor_in_process closed the
# generator's module-level HDF5 handles only from the namespace
# runpy.run_path RETURNS -- which is nothing when the generator
# raises. So a FAILED page left the live store open read-only inside
# the saver, and the next save_step died with
#   OSError: Unable to synchronously open file
#           (file is already open for read-only)
# The 6mo never hit it because its page never failed (job 662: 5
# builds, 0 failures); the 3mo page was erroring standalone.
#
# WHY THE PAGE STAYS OFF HERE ANYWAY, with the fix landed: the
# in-process path has no timeout knob wired yet, so a page that hangs
# rather than raises still blocks the saver. It goes back on once that
# is wired AND one clean stage has run on the 6mo, which keeps it on.
#
# WHAT STILL RUNS: the tar. That path is the EXTRACT alone, which is
# tested to close its handles, so it does not carry the failure above.
# Build the page offline from the tar whenever you want it:
#   python -m lisatools.globalfit.monitor.from_tar SNAP.tar.gz OUT.html
#
# FULL tar, not --short: Mike's morning workflow downloads it.
# MATCH PANELS ON (user request 2026-09-30: "get back to including the phase
# maximized computations of match"). Restores the nine match-criterion panels
# and the Parameter Recovery section. Of those, the F4 three-way split and the
# zoomable-plot marker classes use the REAL phase-maximised, noise-weighted
# overlap (GF_MONITOR_MATCH_MM, default 0.8); the per-iteration
# completeness / purity curves still key off the 2-df-bin f0 proxy so they can
# run on every stored row. Overridable; =0 restores the 2026-08-19 default.
export GF_MONITOR_MATCH_STATS=${GF_MONITOR_MATCH_STATS:-1}
export GF_MONITOR_AFTER_SAVE=1
# PAGE OFF, TAR ON (GF_MONITOR_PAGE, a789d6bd). The 6MO LAUNCHER KEEPS
# THE PAGE -- do not copy this line across.
export GF_MONITOR_PAGE=0
# EVERY THIRD save (1 -> 3, 2026-09-28). ⚠ The old comment's premise
# -- "~2 h/iteration against a page build of order minutes" -- stopped
# being true: on job 663 the page+tar build took 443 s while the
# iteration itself fell to ~500 s once the level-3 mask started
# skipping shut pairs, and it delayed a save. At 1-in-3 the artifacts
# keep flowing at under a quarter of the writer's time, which is the
# watchdog's own warn fraction. Raise it further if iterations get
# faster again; the watchdog is still the backstop.
export GF_MONITOR_ITER=3
# The tar as well as the page -- it is what actually gets downloaded.
# Set to 0 if the watchdog starts complaining; the page alone is much
# cheaper than the multi-GB archive.
export GF_MONITOR_SNAPSHOT=1
# Hard ceiling on one build. Well above the observed cost; exists so a
# pathological build cannot sit on the saver indefinitely.
export GF_MONITOR_TIMEOUT=1800
# Requires MOJITO_INFO_PATH, exported above.

export USE_GPU=1
export GPU_BACKEND=cuda13x
# GPU list follows what slurm actually granted (self-dispatch block above);
# NGPUS is the pre-submit intent, SLURM_GPUS_ON_NODE the in-job truth.
# NOTE GB_N_SUBBANDS below is PER GPU, so total sub-band residency scales
# with the count automatically; nvidia-smi telemetry samples every device.
_NGPUS_EFF=${SLURM_GPUS_ON_NODE:-${NGPUS:-2}}
# printf join, NOT `seq -s,` (BSD seq leaves a trailing separator, which
# the driver's GPUS parser would choke on -- caught in the local smoke).
GPUS=$(printf ",%d" $(seq 0 $((_NGPUS_EFF - 1)))); GPUS=${GPUS:1}
export GPUS
echo "[GPUS] ${_NGPUS_EFF} GPUs -> GPUS=${GPUS} (partition ${SLURM_JOB_PARTITION:-n/a})"

# ---- rank-layout knobs, re-derived in-job -----------------------------
# The self-dispatch block above already exports these via `--export=ALL,...`
# when the script is run directly; a manual `sbatch --gres=... <script>`
# skips that block entirely (SLURM_JOB_ID is already set at job start), so
# re-apply the identical defaults here rather than depending on an unset
# var under `set -u`.
GPUS_PER_RANK=${GPUS_PER_RANK:-}
RANKS_PER_GPU=${RANKS_PER_GPU:-1}
# Re-exported here for symmetry with GF_LEGACY_RANK_LAYOUT below: all three
# knobs are read from the environment by build_layout, and a manual
# `sbatch <script>` never ran the --export=ALL,... dispatch line.
export GPUS_PER_RANK RANKS_PER_GPU
_k=${GPUS_PER_RANK:-1}
if [ "${NGPUS:-2}" = "4" ]; then
  GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
else
  # walker-block layout by default at NGPUS=2 too (user ruling 2026-09-16;
  # GF_LEGACY_RANK_LAYOUT=1 = the rollback knob for a 1-node job)
  GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
fi
# The legacy layout is per-NODE (one compute rank owns that node's whole GPU
# pool; every other non-saver rank is a stopped spare), so it cannot span an
# allocation: on 2 nodes every rank of node B would idle. The self-dispatch
# block forces GF_LEGACY_RANK_LAYOUT=0 at NGPUS=4, but a manual
# `sbatch --nodes=2 --ntasks=5 <script>` skips that block entirely -- so
# force it here too, off the GRANTED allocation, and say so loudly.
if [ "${SLURM_NNODES:-1}" -gt 1 ] && [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
  echo "[SUBMIT] SLURM_NNODES=${SLURM_NNODES} > 1: FORCING GF_LEGACY_RANK_LAYOUT=0"
  echo "[SUBMIT]   (the legacy single-compute-rank layout is per-node and cannot"
  echo "[SUBMIT]    span an allocation -- every rank on the other node(s) would"
  echo "[SUBMIT]    build, idle and waste the node)."
  GF_LEGACY_RANK_LAYOUT=0
fi
export GF_LEGACY_RANK_LAYOUT
if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
  N_COMPUTE_EFF=1
else
  N_COMPUTE_EFF=$(( ${SLURM_NNODES:-1} * _NGPUS_EFF * RANKS_PER_GPU / _k ))
  if [ "${N_COMPUTE_EFF}" -lt 1 ]; then
    echo "[SUBMIT] N_COMPUTE=${N_COMPUTE_EFF} from NNODES=${SLURM_NNODES:-1} x"
    echo "[SUBMIT]   GPUS=${_NGPUS_EFF} x RANKS_PER_GPU=${RANKS_PER_GPU} /"
    echo "[SUBMIT]   GPUS_PER_RANK=${_k}: inconsistent knobs (GPUS_PER_RANK larger"
    echo "[SUBMIT]   than the granted pool?). Leaving NWALKERS alone; build_layout"
    echo "[SUBMIT]   will raise with the exact reason."
  fi
fi

# ---- output ----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_3mo

# ---- v8 noise model (the whole v8-vs-v7 diff) -------------------------------
export UNEQUAL_ARM=1
export UNEQUAL_ARM_STRIDE=200
export WDM_PSD_METHOD=layer_calibrated
export GALFOR_MODULATION_PATH="$PWD/scripts/noise/modulation_unequal.dat"
export GALFOR_MODULATION_T0=data
# PIN THE RESUMABLE NOISE IDENTITY OF THIS STORE (2026-09-23). 36e2319e made
# the NOISE-brick scalar fit use the brick's own link delays, which moves the
# fitted (Soms_d, Sa_a) by 0.26% / 0.59%:
#     equal arms (this store)  1.496182116469e-11  2.982411739286e-15
#     unequal arms (new code)  1.500004011496e-11  3.000107254658e-15
# Those numbers ARE general.psd_injection, and the coarse delayed-acceptance
# fiducial digest is a SHA-256 over their raw float64 bytes, so the new fit
# changes the digest and the resume guard refuses the store:
#     "stored noise-model identity differs from the configured one:
#      {'coarse_fiducial_digest': ('856ddb07f6b46e39', 'b5047f761a4400d5')}"
# =0 restores the equal-arm fit BIT-IDENTICALLY (verified against 36e2319e^
# on the mojito-light brick), so gf_prod_6mo_v8_4gpu resumes unchanged.
# DELETE THIS LINE for a fresh store -- the arm model is the better answer,
# it just is not the answer this chain was started with.
# NOT the noise model: UNEQUAL_ARM=1 above still selects
# UnequalArmInstrumentNoise for the likelihood. This is only the REFERENCE
# fit that produces general.psd_injection.
# MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM IS DELETED HERE (user ruling
# 2026-09-26, 3MO-6). The 6mo script pins it to 0 for ONE reason: to keep a
# v8-lineage store resumable across 36e2319e, which made the NOISE-brick
# scalar fit use the brick's own link delays and moved the fitted
# (Soms_d, Sa_a) by 0.26% / 0.59%:
#     equal arms    1.496182116469e-11  2.982411739286e-15
#     unequal arms  1.500004011496e-11  3.000107254658e-15
# Those numbers ARE general.psd_injection -- the "truth" line every monitor
# and whitening test compares against -- and the PSD-bias forensics concluded
# the estimator was fitting EQUAL-arm while the run ran UNEQUAL. This store is
# FRESH, so the better answer is available and the default (=1, unequal) is
# taken. Safe because COARSE_Q=1 below means there is no coarse fiducial
# digest to invalidate. Re-pin =0 ONLY to resume a store born equal-arm.
echo "[V8-NOISE] UNEQUAL_ARM=${UNEQUAL_ARM} stride=${UNEQUAL_ARM_STRIDE} wdm_psd_method=${WDM_PSD_METHOD} psd_reference_fit_unequal_arm=${MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM:-1 (default: UNEQUAL)}"
echo "[V8-NOISE] modulation=${GALFOR_MODULATION_PATH} t0=${GALFOR_MODULATION_T0}"

# ---- coarse noise likelihood (pinned, not inherited) ------------------------
# all_sources DEFAULTS these on now, but a submit script states its own
# configuration: a reader must be able to tell which noise likelihood the run
# used without cross-referencing the variant.
#
# delayed_acceptance is EXACT -- stage 1 screens PSD/galfor proposals on the
# Q-fold time-coarsened surrogate, stage 2 corrects with the exact fine/coarse
# ratio -- so the SAMPLED target is the fine likelihood in every stage,
# whatever the surrogate's quality. (search_approx is faster but approximate;
# in probe job 376 the search it drove railed galfor against two prior edges.
# The noise block is a small share of wall clock, so that trade is not worth
# taking here -- accuracy ruling 2026-08-28.)
#
# Q=8 measured on the 3-mo grid: Nt_active 2121 -> Ncoarse 266. WS weighting
# (the Welch-Satterthwaite effective dof, frozen at the injection fiducial)
# was exercised on GPU in job 376 -- the unequal-arm coarse basis path works
# there. To fall back: COARSE_GPU_MODE=off restores the exact-fine likelihood
# (job 369's configuration) and COARSE_USE_WS=0 swaps WS for Bartlett.
# ---------------------------------------------------------------------------
# V9-2 (2026-09-24, user ruling): NOISE MODEL BACK FROM "COARSE" TO NORMAL.
# v8 ran mode=delayed_acceptance with the Q=8 time-coarsened surrogate as
# stage 1. v9 runs the plain EXACT-FINE likelihood -- the configuration the
# comment above names as the fallback, and the one probe job 369 ran.
#
# This is NOT a bias change in either direction: delayed acceptance is exact
# (stage 2 corrects with the fine/coarse ratio), so v8 and v9 sample the SAME
# target. What changes is cost and the number of moving parts -- no surrogate
# basis, no WS dof, no fiducial digest, no stage-2 acceptance to audit, and
# the [COARSE_AUDIT] lines simply stop appearing. Expect the noise block to
# get SLOWER; it is a small share of the iteration (28.5 s of 852 s measured
# at v8 3mo), which is what makes the simplification affordable.
#
# ⚠ THE COARSE MODE IS PART OF ``noise_model_identity``. A v8 store CANNOT be
# resumed under this line -- the [V8-NOISE] preflight below and run.py both
# refuse the mismatch by design. STORE_DIR is a fresh v9 path for this reason.
#
# ⚠ COARSE_Q MUST BE 1 HERE, AND "8 + mode=off" WAS NOT A NO-OP.
# JOB 618 (2026-09-25) aborted all 5 ranks at build:
#
#   ValueError: coarse_Q > 1 in an all-source run requires an explicit
#   COARSE_GPU_MODE ('search_approx' for optimization stages,
#   'delayed_acceptance' for production PE): the coarse statistic changes
#   the PSD/GALFOR transition kernel, so it is never an implicit speed knob
#   here.   [stock/erebor/noise.py::validate_coarse_settings]
#
# This comment used to claim COARSE_Q could stay at 8 "so the stamped
# identity and the preflight's wanted values are read from the same env",
# because "with mode=off they select nothing". THAT WAS WRONG, and the
# validator exists to catch precisely it: run.py:1607 reads coarse_Q on its
# own -- `Q = int(getattr(general_info, "coarse_Q", 1) or 1)` -- and builds
# the coarse basis for ANY Q > 1 without consulting the mode at all. So the
# pair (Q=8, mode=off) is not "exact fine with a documentary 8"; it is two
# subsystems disagreeing, which is why it is refused rather than resolved.
#
# Q=1 IS the exact-fine configuration V9-2 asked for -- there is nothing to
# coarsen -- and it is what "the general psd computations for galfor and
# psd, not the coarse version" (user, 2026-09-25) means in this file.
# ⚠ all_sources.py defaults COARSE_Q to 8, so this export must be explicit.
# To go back to the v8 configuration: COARSE_Q=8 AND
# COARSE_GPU_MODE=delayed_acceptance together (and a store created under
# that pair -- the mode is part of noise_model_identity).
export COARSE_Q=1
export COARSE_GPU_MODE=off
export COARSE_USE_WS=1
export COARSE_FIDUCIAL=injection
# Galfor slope-index cap (2026-09-04 diagnostic, KEPT on the 3mo science
# arm; ported): alpha railed at the stock 5.0 cap (~60% of samples at the
# edge). Widened [1e-3, 20] so the slope can explore; revert = drop the
# line. The prior is rebuilt from code each run.
# GALFOR SAMPLING BASIS (user ruling 2026-09-18). amp, fk, f_1, f_2 move to
# log10 over the SAME physical support (galfor_prior_dict + the 10**x
# transform container); alpha is O(1) and stays linear. The foreground model
# still receives linear parameters.
#
# WHY. galfor's linear prior box spans ~42 decades -- amp (1e-47, 1e-41)
# against alpha (1e-3, 20) -- and the eigen proposal's step is scaled to that
# box: the softest whitened eigenvalue puts a ~1-prior-width step on amp,
# i.e. ~190x the parameter's own value, which takes amp negative and the
# prior rejects every draw. That is the measured "galfor does not move".
#
# The two cheaper fixes were tried and REFUTED, in this order:
#   * per-parameter whitening of the info matrix -- ALREADY implemented in
#     eigen_refresh._tables_from_info_batch (diag(w) I diag(w)); rescaling
#     cannot make a likelihood quadratic in a parameter it is wildly
#     non-linear in;
#   * building the eigen tables on the COLD ROW only (512105bd) -- shipped,
#     confirmed live in the 6mo run by its "1 sources x 5 dims" infomat
#     signature, and galfor STILL came back non-positive on 23 of 28 builds,
#     ~29% of them catastrophic (worst lambda/lambda_max -9.7e78).
#
# NEEDS A FRESH STORE: this changes what the stored numbers MEAN, not their
# shape. A resume across the flip is refused by noise_model_identity as of
# 0044b7cc (galfor_log_sampling is stamped and compared) -- pull at least
# that commit before running this, or the refusal is not there to catch you.
export GALFOR_LOG_SAMPLING=${GALFOR_LOG_SAMPLING:-1}
export GALFOR_ALPHA_MAX=20.0
echo "[V8-NOISE] coarse: Q=${COARSE_Q} mode=${COARSE_GPU_MODE} \
use_ws=${COARSE_USE_WS} fiducial=${COARSE_FIDUCIAL}"

# ---- sampler shape ---------------------------------------------------------
# 10 -> 4 (user ruling 2026-09-18, for the fresh walker-block store). One
# walker per compute rank at NGPUS=4. Chosen for SEARCH SPEED: at fixed GPU
# count the per-rank walker block is what grows, and ~364 s of a ~390 s
# gb_search iteration scales with rows (the three GB RJ moves ~260 s,
# mbh/emri amortized ~62, vgb ~30, noise ~11). Only SOBBH does not -- its
# scorer bills a flat 1.73 s per CALL whatever the batch shape. Doubling to
# 8 walkers therefore roughly doubles the iteration while buying nothing in
# search throughput: births come from the F-stat grid and the warm-start
# components, both independent of walker count, so births per HOUR is a wash
# and iterations per hour is what drives the cap ramp and the tempering
# cadence.
# NOTE the walker count LOCKS for the life of the store (the resume refuses
# any change -- state.py's walker-count mismatch), and it carries into
# full_pe, where 4 is a thin posterior ensemble. Deliberate: this store's job
# is the search.
export NWALKERS=${NWALKERS:-4}
                                   # GB rungs stay GB_NTEMPS=24 -- walkers and
                                   # temps are independent axes. NEVER change
                                   # NWALKERS on a resume: the store carries the
                                   # walker axis. Under the walker-block layout
                                   # NWALKERS must divide by N_COMPUTE, and
                                   # N_COMPUTE = NNODES * GPUS_PER_NODE *
                                   # RANKS_PER_GPU / GPUS_PER_RANK. At the
                                   # intended NGPUS=4 shape that is
                                   # 2 * 2 * 1 / 1 = 4, so NWALKERS=4 divides
                                   # cleanly at 1 walker/rank and NOTHING is
                                   # rounded. (The older note here claimed
                                   # "4 GPUs need 5 compute ranks with
                                   # RANKS_PER_GPU=2" -- that is wrong and was
                                   # left over from the NWALKERS=10 plan:
                                   # RANKS_PER_GPU=2 gives N_COMPUTE=8, not 5,
                                   # and 5 is not reachable from 4 GPUs at all.)
                                   # ⚠ A NON-MULTIPLE IS ROUNDED UP, not
                                   # refused, and the store then LOCKS to the
                                   # rounded value: at NGPUS=4, 2 and 3 both
                                   # become 4, 5 and 6 become 8, 10 becomes 12.
                                   # So test at the value you intend to launch
                                   # -- a resume refuses any change.
                                   # Noise-block floor 2*ndim (galfor ndim 5
                                   # -> 10) still satisfied.
                                   # The dispatch's --export=ALL carries the
                                   # shell's value, so `NWALKERS=1 ./submit_gf_6mo_v9_4gpu.sh`
                                   # reaches the one-walker branch below.
if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then
  echo "[SUBMIT] NWALKERS=1 on N_COMPUTE=${N_COMPUTE_EFF}: one-walker replica mode (every compute rank holds the walker; GB/VGB by band range, addremove/PSD by likelihood rows)"
elif [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 0 ] \
     && [ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then
  echo "[SUBMIT] NWALKERS=${NWALKERS} is not a multiple of N_COMPUTE=${N_COMPUTE_EFF}; using NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF )) (user decision at the first 4-GPU launch)"
  export NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF ))
fi
export NUM_ITERATIONS=2000         # total engine iterations (resume-safe; NITER was a dead name)

# ---- band + domain ---------------------------------------------------------
# EXPLICIT Tobs (2026-08-13): sbatch propagates the submitting shell's env,
# and a stale TOBS_TARGET export (a 3-day one was found live in the shell)
# would silently re-grid this run. Pin the 90-d production value.
export TOBS_TARGET=7776000         # 90 d; grid resolves Nf 1440 x Nt 2160 x dt 2.5
# ---- SIGHET_NT_LAYER: DELIBERATELY NOT EXPORTED AT 3 MONTHS ---------------
# The 6mo script pins 120 because 120 divides Nt=4320 and holds the validated
# 36-h sparse stride. At Nt=2160 the CODE DEFAULT does the same job: 64 snaps
# to 60, and 2160/60 = stride 36 time pixels x 3600 s = 36 h. Same constant
# temporal density, taken from the default rather than pinned -- which is what
# every 3-month production run has done and what ThreeMonthTwinTest asserts.
# CONFIRM IT FROM THE LOG, do not assume; nothing else echoes it:
#   grep "sig-het engine resolved" <store>/<base>_artifacts/globalfit_run.log
#   want: nt_layer=60 (stride 36) ... sparse spacing 36.0 h
# ⚠ DO NOT set this to chase accuracy. It is not a mid-store knob, and it
# MULTIPLIES with GB_N_SUBBANDS in the sig-het stash byte product (the v4@270
# OOM: 265 x 8192 = 2.2e6). The 2026-08-18 A/B that motivated a finer grid did
# not reproduce once the config echo made it attributable, and the bench-off
# showed a CPU run at STRIDE 1 still carried the tail -- the real lever on
# sig-het accuracy is NODE PLACEMENT (SIGHET_N_CP, pinned at 256 below).
# export SIGHET_NT_LAYER=120       # <- the 6mo value; do NOT carry it across
# ⚠ MIN_FREQ 4e-4 -> 2.5e-4 (2026-09-06). This is the LAYER-1 FIX, and the
# value is set by the VGBs, not by taste.
#
# WHY IT MOVED AT ALL. all_sources pinned min_freq as a PLAIN default 1e-4 that
# shadowed the env-backed field, so THIS EXPORT NEVER TOOK EFFECT (fixed in
# 7ad3b0bd). The run therefore analysed from WDM layer 1 -- support
# 0.069-0.208 mHz, sharing an edge with DC, where the instrument model diverges
# and instrument_fill_nans=0.0 zeroes the covariance. Measured on this store
# (kappa probe, job 459): det(C)=0 there, q = w^T C^-1 w / 3 = 150.3 against an
# expected 1.0, i.e. 43% of the fit's ENTIRE chi^2 budget. That alone produced
# the instrument-PSD bias: the exact ML alpha = sqrt(mean(q/3)) = 1.389 vs the
# 1.3874 observed (0.21%).
#
# WHY 2.5e-4 AND NOT 4e-4. Only layer 1 is pathological; layer 2 measured
# q/3 = 1.0203, as clean as anything in the band. Both choices cure the bias
# identically (alpha 1.0490 at layer 2 vs 1.0493 at layer 3) -- but the VGB
# catalogue's four lowest sources sit at 0.3117/0.3364/0.3365/0.3392 mHz, and
# layer 3's support starts at 0.34722 mHz, so MIN_FREQ=4e-4 would DROP FOUR
# VGBs. 2.5e-4 lands on layer 2 (support from 0.20833 mHz), keeps all 55 VGBs,
# and retains 93.8% of the S_tm information against 87.9% at layer 3.
#
# Valid range for layer 2: 1.3889e-4 < MIN_FREQ <= 2.7778e-4. 2.5e-4 sits
# comfortably inside (2.7778e-4 is the exact boundary and can tip to layer 3 on
# floating-point rounding -- do not use it).
#
# RESUME-SAFE: band_edges/cap_edges are stored in absolute Hz and start at
# layer 4 (5.5556e-4), entirely above the new floor, so they recompute
# bit-identically and the leaf-cap guard passes.
export MIN_FREQ=2.5e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- GB knobs (everything else rides the flipped defaults: sig-het in-model,
#      fstat-fit-in-move + sig-het fstat, D/2 leaf-cap gate w/ min-iters 5,
#      at-cap RJ skip, cell-lifecycle ll credit, GB_MODE=search +
#      GB_PE_MOVES_STRICT=1 + GB_SEARCH_PRIOR_REMOVAL=1 seeded by the script) --
export GB_NLEAVES_MAX=10000        # 3 mo: what every 3-month run has used (6 mo runs 15000 for its deeper resolved confusion)
# FULL parity-unit residency (grouped RJ->in-model scheduling, 2026-08-13):
# one unit = 77 bands x 24 temps x 24 walkers = 44,352 cells; the scheduler
# clamps n_slots to min(GB_N_SUBBANDS, cells), so 50000 means every cell is
# resident (zero mid-unit refills) and the grouped in-model flush runs at
# full grid width. Buffer cost ~255 KB/slot at slab-5 => ~11.3 GB (user
# budget: 10-20 GB is fine). Back off to 2560 (652 MB) if the pool OOMs.
# 16384 (user Q 2026-08-14): slots now amortize ROUNDS, not just memory --
# rounds/unit ~ waves x per-wave depth, so 4x residency ~ 3-4x fewer
# 2.4s host round-trips. ~4.2 GB buffer; post-fix profile at 4096 was
# flat 42-45/31 GB on 96 GB cards. If the unit-open lines stay flat,
# full residency (50000 -> 44,352 slots, ~11.3 GB) is the next step.
# ---- OOM 2026-09-16 (dev0, 3.68 GB En fold at 88.9 GB allocated, run
# died overnight in setup_in_model/bin_fold_real): 16384 violated the
# sig-het stash law documented at the SIGHET_NT_LAYER block below -- the
# stash goes as CELLS x N_sparse_t and the knobs MULTIPLY. The mirror
# halved SLAB bytes (the rationale for 16384) but not the fold stash:
# 6mo N_sparse_t 118 x 16384 = 1.93e6, right at the 2.2e6 product that
# OOM'd v4@270 (and 4x the calibrated 6-mo-safe 118 x 4096 = 4.8e5).
# Residency fills toward capacity as rj/tempering matures, so early
# telemetry (35-45 GB) looks fine and the wall arrives after the first
# long uninterrupted stretch. 8192 -> product 0.97e6, under the 23-mo
# 1.1e6 OK precedent; back off to 4096 (the calibrated-safe value) if
# dev0 max memory trends past ~70 GB in gpu_util_*.csv again.
export GB_N_SUBBANDS=32768  # PER GPU; total = x n_gpus. 3mo slabs are ~0.25
                            # MB/slot against 6mo's ~0.5 (PSD shared mirror),
                            # and this is the value the 3mo 10-walker arm ran
                            # healthy through iteration 210.
                            # ⚠ IT IS SAFE ONLY BECAUSE THE STAGING KNOBS
                            # BELOW KEEP THE 6MO POST-OOM VALUES. The sig-het
                            # stash goes as CELLS x N_sparse_t and the two
                            # MULTIPLY: 60 x 32768 = 1.97e6, above the 23-mo
                            # 1.1e6 precedent. What bounds the actual
                            # transient is the fold chunker, which sizes each
                            # chunk to FILL GB_SIGHET_FOLD_MAX_BYTES -- so the
                            # 1 GiB cap plus GB_INMODEL_SETUP_BATCH=2048 holds
                            # it bounded whatever the slot count. Back off to
                            # 8192 (the 6mo value) if dev0 max memory trends
                            # past ~70 GB in gpu_util_*.csv; NEVER reach for
                            # SIGHET_NT_LAYER.
# PSD SHARED MIRROR (386f25ce, validated jobs 469/470: 3mo dev0 68->44 GB,
# 1yr 85->61 GB with capacity DOUBLED; parity 11k+ IDENTICAL). Parity gate
# disarmed: vgb_pe rebuilds its buffer every unit so the gate never retires
# (+25 s/unit at 1yr); the gate already banked its evidence on 469/470.
export GB_PSD_SHARED_MIRROR=1
export GB_PSD_MIRROR_PARITY_PROPOSES=0

### SOME OTHER GB options
# V9-11: 5.0 -> 8.0. The opt-SNR boundary is now a PER-STAGE PROFILE value
# (V9_SEARCH_STAGE_PROFILES in run_combined_staged.py): gb_search_1 sets 8.0,
# gb_search_2 and gb_search_3 set 5.0, applied to every GB band move at stage
# entry and logged as [V9-STAGE ...] opt_snr_rej_samp_limit lines.
# This export is the value the moves are CONSTRUCTED with, i.e. what is in
# force before the first stage profile applies, so it is set to stage 1's --
# if the profile ever failed to apply, the run would hold stage 1's floor
# rather than silently running stage 1 at stage 2's.
export GB_OPT_SNR_LIMIT_SEARCH=8.0
# 10 -> 5 (user ruling 2026-09-27). Consumed by
# run_combined_staged.py::source_pe (NOT by src/ -- a grep of the
# package alone says this knob is dead, and it is not).
# Measured on 6mo job 650: sobbh_pe 144.0 s + mbh_pe 95.7 s +
# emri_pe 71.1 s = 310.8 s on the iterations they run, ~4.9% of a
# 6377 s iteration. At every-10 that averaged 31 s/iteration; at
# every-5 it averages 62 s. full_pe is untouched -- all three run
# every iteration there.
export GB_SEARCH_SOURCE_EVERY=5

# RJ pick thinning. UNSET as of 2026-08-28 -- the value now lives in code
# (_SEARCH_RJ_FLIP_DEFAULT / _PE_RJ_FLIP_DEFAULT in recipe.py, both 0.2),
# so behavior is UNCHANGED from the 0.2 this line used to export.
#
# Removed rather than kept because {BRANCH}_RJ_FLIP_FRACTION is a GLOBAL
# override: one exported value lands on every RJ move in every stage, so
# it silently collapses any future search/PE split. The five search-named
# RJ moves are now passed the default explicitly at their construction
# sites (LAT bba4219d) -- before that they only reached 0.2 BECAUSE of
# this export, and would have fallen through to a hard-coded 1.0 without
# it. The old comment here also said "0.3 random subset" while exporting
# 0.2; that contradiction is gone with the line.
# In-model repeats are unaffected either way -- they cover ALL alive
# sources; the flip gate is rj-only by construction.
# export GB_RJ_FLIP_FRACTION=0.2   # <- re-export ONLY to force ALL stages
# PER-STAGE flip knobs (2026-09-11 user ruling, values corrected 09-14 --
# ported from the 3mo 10-walker arm): rj_fstat_search never received the
# flip default (recipe wiring miss, fixed 09-14), so the search ran at 1.0
# in jobs 474/479; the intended "half the birth attempts" against what
# actually ran is 0.5 (0.1 would be a 10x cut of the search feed). PE
# moves DID carry 0.2 before, so 0.1 there is the intended halving.
# V9-5 (user ruling 2026-09-24): 1.0 for the SEARCH parts -- every RJ
# candidate row gets a proposal instead of half of them. This reverses the
# 09-11 halving; the search feed is what the three-stage structure is built
# to consume, and stage 1 in particular is meant to be greedy.
export GB_SEARCH_RJ_FLIP_FRACTION=1.0
export GB_PE_RJ_FLIP_FRACTION=0.1
# In-model info-matrix jump scale: 0.005 default measured 95% cold
# acceptance; 0.2 -> 0.61; 0.4 -> 0.60 (job 196). Job 197 flipped the
# story: with the EXACT per-block SIGHET info matrices live, cold
# acceptance at 0.6 ROSE to 0.71-0.80 (all-T 0.83-0.92, n=27k) -- the
# better-adapted covariance makes the old scale far too timid. 1.2 is
# the next notch (recommended range 1.0-1.5); tune against the
# [GB_ACCEPT] per-proposal-type line toward the 0.15-0.4 target.
export GB_JUMP_FACTOR=1.2
# ---- TIMERS: ALL ARMED (user directive 2026-08-15, fresh run: "make sure
#      all the timers are armed so we can study the differences in detail")
# Always-on, FREE (no knob, no sync): the [GB_TIMING <move>] per-propose
# span breakdown, the [FSTAT_CTR] center census, [GB_ACCEPT] +
# [GB_ACCEPT rj-split] per-proposal-type rates, [GB_CELL_LL] credit
# checks, [SAVE] write times, the buffer-lifecycle lines, and the
# nvidia-smi CSV sampler above.
#
# GF_MOVE_TIMING: per-move wall_s + host RSS + GPU-pool MB for EVERY move
# in the stage. This is the one that finally makes psd_pe / galfor_pe
# visible -- they emit no [GB_TIMING] of their own, which is why the
# ~43 s/iteration noise block was untimed "dark matter" until now. Cheap:
# one timestamp + two memory reads per move.
export GF_MOVE_TIMING=1
# GF_MOVE_TIMING_SYNC / GB_PROP_TIMING_SYNC: make every mark carry EXACTLY
# its own kernel time instead of leaking into the next mark (cupy is async,
# so unsynced spans attribute launch time, not execution time). This is
# what makes a detailed before/after comparison trustworthy.
# COST: ~8-10 extra device syncs per get_ll; measured +2.5% wall on the
# GB propose, and it serializes some concurrent shard work.
# >>> TURNED OFF 2026-08-28 (user ruling). The tempering audit found the
# >>> sync cost is much larger than the +2.5% headline in the TEMPERING
# >>> path specifically: `temper_swap_score` opens and closes INSIDE the
# >>> rung loop, so with sync on it adds roughly 27,000 full device syncs
# >>> per move -- a real part of the ~88 s that `run_tempering` could not
# >>> account for. Leaving it on means the instrumentation is measuring
# >>> itself. Detailed per-stage attribution is already banked from
# >>> snapshots 11-13; steady-state numbers now matter more.
# >>> Set BOTH back to 1 for 2-3 iterations if a fresh detailed
# >>> attribution is ever needed again -- the run resumes cleanly.
export GF_MOVE_TIMING_SYNC=0
# GB_PROP_TIMING_SYNC uses the := form to MATCH submit_gf_3mo_v7.sh:467,
# whose comment states the reason: a plain `export VAR=0` CLOBBERS an
# sbatch-time override, so `GB_PROP_TIMING_SYNC=all sbatch ...` would
# silently do nothing -- exactly the re-arm the comment above tells you to
# perform. Resolved value is still 0, so run behaviour is unchanged; this
# only removes a v7/v8 asymmetry and restores the documented escape hatch.
: "${GB_PROP_TIMING_SYNC:=0}"
export GB_PROP_TIMING_SYNC
#
# ---- rj_fstat_centers INTERIOR PROBE (2026-08-29) --------------------
# WHAT IS ALREADY KNOWN (do not re-derive, and do not re-open it as a
# "missing time" question -- that was an arithmetic error):
#   v7 snapshot: total=1955.1s run_proposal=1810.6s
#   rj_fstat_centers=1334.874s (68% of the propose).
#   The "[FSTAT_CTR ...] unit precompute: ... in 149.75s" line is emitted
#   ONCE PER BAND UNIT and the move runs NINE units per propose. The nine
#   lines sum to ~1339 s vs the reported 1334.874 s -- the bucket closes
#   to 0.2%. Cross-checked twice: the other _run_rj_step marks bound the
#   per-round centre chain at <= 3.077 s, and the nine unit row counts sum
#   to exactly picked_sources = 4,546,846.
#   => ~99.8% of the stage is _precompute_fstat_centers -> _fstat_ctr_compute,
#      4.55 M rows/propose at ~0.293 ms/row, ~97% of them dead birth slots,
#      "0 at-cap excluded".
#
# WHAT IS NOT KNOWN, AND WHAT THIS PROBE IS FOR: what that 0.293 ms/row is
# made of. Basis-filter/scorer kernel? Jaranowski-Krol inversion? Host
# staging of 4.55 M rows? The answer decides the attack -- a cheaper
# kernel vs fewer rows. rj_fstat_centers now decomposes (all NESTED, so
# `tracked` is unchanged):
#   unit-open precompute : fstat_ctr_select / fstat_ctr_solve /
#                          fstat_ctr_census / fstat_ctr_audit /
#                          fstat_ctr_pack
#   the scorer itself    : fstat_nm_transform / fstat_nm_lanes |
#                          fstat_nm_routed / fstat_nm_invert /
#                          fstat_ctr_map / fstat_nm_h2d /
#                          fstat_nm_lane_score / fstat_nm_lane_build /
#                          fstat_ctr_miss_fallback
#   per pick round       : rj_ctr_keep_gate / rj_ctr_birth_lookup /
#                          rj_ctr_birth_draw / rj_ctr_death_lookup /
#                          rj_ctr_death_dens
# plus counters fstat_ctr_units / fstat_ctr_rows / fstat_nm_rows /
# rj_ctr_route_{table,cache,direct} / fstat_ctr_audit_rows.
#
# ***READ THE SPLIT FROM A SYNC-ON PROPOSE, NOT A PRODUCTION ONE.*** The
# TOTAL is solid either way (it is confirmed above), but the SHARES inside
# it are not: with GB_PROP_TIMING_SYNC=0 the spans do not synchronize, so
# a sub-stage that contains a sync point is billed for kernels launched by
# earlier ones -- the fill_indmap_data precedent (598 s measured, 45 s
# real). Splitting a 1,331 s interior is exactly where that bites.
#
#     GB_PROP_TIMING_SYNC=all   # 'all' drains EVERY run device;
#                               # '1' drains only the current one, which
#                               # is not enough with GB_FSTAT_NM_MULTIDEV
#                               # fanning the scorer across both GPUs.
#
# Relaunch this script with that exported (and NUM_ITERATIONS small, or
# just kill it after a couple of [GB_TIMING] lines) -- the run resumes
# cleanly from the checkpoint, exactly as the 2026-08-28 note above says.
# Then read fstat_nm_transform / fstat_nm_lanes (with its fstat_nm_h2d +
# fstat_nm_lane_score) / fstat_nm_invert / fstat_ctr_map as shares of
# fstat_ctr_solve. Leave it at 0 for production.
# ---- 2026-08-15 perf batch (ALL code defaults; pinned for the run
#      record; each knob independently revertible) ---------------------
# De-synced in-model repeat loop (device-resident accept chain) rides
# the pull with no knob (bit-identical decisions, tested).
export GB_ROUTER_DEVICE_RESIDENT=1 # params/outputs never host-stage in the
                                   # shard router; =0 restores host staging
export GB_RJ_SNR_TRUNC_DIST=1      # birth distance draw truncated at the
                                   # analytic SNR-5 boundary; truncated
                                   # density in the factors (DB-exact);
                                   # =0 restores the plain lognormal
# Per-class in-model repeats (user ruling 2026-08-27, final:
# 250/25 after a same-day 200/100 hold): concentrate polish AT BIRTH
# (a newborn lands at grid resolution and must climb its peak before
# removal judges it) and buy iteration RATE with the survivor budget
# (highf endgame: survivor polish saturates — stuck walkers took
# 100/round for ~150 rounds without moving; transport fixes those —
# while survivor cost rides all three RJ moves and scales with alive
# count). Faster rows also tick the iteration-clocked cap patience
# faster and give more permuted + vertical swap rounds per hour.
# NOTE these env pins beat the PE mode default as well — both phases
# run 250/25.
# 100 (2026-09-11 user ruling, was 250 -- ported from the 3mo 10-walker
# arm): with the vertical ladder working, hot-rung newborns descend over
# the following iterations and are polished as survivors at every rung
# they visit; the newborn class had become 83% of the 1yr repeat-rows
# (3,347 rows @250) and ~50% at 3mo.
export GB_INMODEL_REPEATS_NEWBORN=100
# SURVIVOR 25 -> 100 (user ruling 2026-08-29, aligned with v7), restoring the
# value the high-f probe ran (200/100). In-model f0 drift is the ONLY mechanism
# that moves a source across a sub-band edge -- there is no merge operator, RJ
# is serial-within-band, and tempering swaps never cross bands -- and the
# measured crossing rate shows how underpowered it is at 25: median NN-matched
# displacement 0.046 bins/iteration, only 1.64% of matched pairs cross a band
# edge, 0.59% clear a 12-bin gap in one step. Trades iteration RATE for the
# ability to close edge-split pairs.
# ⚠ The per-class split applies on the DIRECT-batch path only; the grouped
# scheduler takes ONE budget for the whole pool from _SURVIVOR, so with
# GB_RJ_GROUPED_INMODEL=1 this raises the effective budget for newborns too.
# 50 (2026-09-11 user ruling, was 100 -- ported from the 3mo 10-walker
# arm): vertical sweeps run once per repeat, so this also halves the
# vertical mixing per block -- judge by progress per wall-clock hour with
# the [GB_CELL_LL] / sig-het cold audit as guards.
export GB_INMODEL_REPEATS_SURVIVOR=50

# ===========================================================================
# V9-1: PER-SOURCE IN-MODEL CONVERGENCE (2026-09-24 user design).
# ===========================================================================
# The two budgets above stop being what a newborn gets and become the SCALE
# the ceiling is derived from. Each ROW -- one (temp, walker, band) cell --
# now runs until its OWN cell lnL stops climbing, then freezes; a band leaves
# when its cold rungs are done; a finished band is swapped out for a queued
# one. SURVIVOR (mature) rows are untouched by default and keep the fixed 50.
#
# Full design + the measurements behind every default:
#   docs/superpowers/specs/2026-09-24-gb-inmodel-convergence-design.md
#
# THE RULE, per row: G = running sum of ACCEPTED delta_ll (== the cell's
# whole-model lnL up to a constant); retire once
#   max(G) has not improved by DLL over the last ITERS repeats.
# It is deliberately NOT the leaf cap's D/2 form: that gate's clock is a
# whole sampler iteration, and on a per-REPEAT clock the same code becomes a
# per-step test that retires a row mid-climb (measured: 17.3 lnL left on the
# table). And it is the GAIN, not `L_free + ll_ref`, because ll_ref is a
# sig-het value whose absolute offset is 1-17 lnL cold and is re-anchored
# mid-block every GB_SIGHET_REFRESH_EVERY repeats.
#
# ⚠ READ (thresh, ITERS) AS A RATE: a row is retired below DLL/ITERS lnL per
# repeat = 0.04 here. ⚠ AND NOTE THE FLOOR: a row that converged instantly
# still pays ITERS + 1 = 101 repeats, ABOVE v8's fixed 100 -- so this is a
# SPEND, not a saving, and the [GB_IMCONV] "work =" ratio will print above
# 1.0. That is the intended trade (user ruling: run to convergence, not to a
# fixed 100). ITERS=50 is the setting that makes it pay for itself instead.
export GB_INMODEL_CONVERGE=on
# ⚠⚠ FIRST JOB OF THE CAMPAIGN SHOULD ARGUABLY BE `observe`. It runs the
# ENTIRE rule under the unchanged fixed budget -- not one proposal differs --
# and logs the repeat/gain distribution the rule WOULD have produced. It is
# the paired negative control, and it answers the one question that decides
# whether arming is worth anything: if every row would retire at the floor
# having gained ~nothing, the rule is a constant budget of ITERS and
# GB_INMODEL_REPEATS_NEWBORN=100 already says that for free. Flip this line
# to `observe`, read the [GB_IMCONV ... calibration] line, then flip back.
#   export GB_INMODEL_CONVERGE=observe

# W, the patience window, in UNTHINNED in-model repeats (user ruling
# 2026-09-24, raised from 50; 2026-09-25 raised again 100 -> 250 "just to
# be safe" for births). 100 tracked climbs out to ~300 repeats where 50
# gave up at 76.
#
# MEASURED AT W=100 (job 620): repeats p10/p50/p90/max = 102/260/400/401,
# only 9.7% retiring at the floor and 94% converging on evidence -- so 100
# was NOT truncating. 250 buys margin on the tail rather than fixing a
# known cut-off.
#
# ⚠ THE FLOOR MOVES WITH IT. A row cannot retire before W+1, so the floor
# goes 101 -> 251, and the ceiling below HAS to move with it or the window
# never gets to speak -- see the note there.
export GB_INMODEL_CONVERGE_ITERS=250
# ---- SURVIVOR WINDOW (user ruling 2026-09-25, off job 628) -------------
# 250 stays for BIRTHS; survivors get 100. The window is a FLOOR on block
# length -- a row cannot converge before seen >= window -- so every block
# runs at least that many repeats whatever the physics says. Measured over
# 628's three complete search cycles:
#
#   survivors  median 260, p90 320-375, MAX 420-480, ~50% sat exactly at
#              the 255 floor
#   births     median 275-285, p90 435-500, max 1200-1865  <- these DO
#              use the depth, so 250 stays for them
#
# and the work is not evenly split: survivors are 96.3% of all converging
# in-model work, births 0.9%. Survivors' mean of 276-306 against a floor of
# 255 means ~90% of the in-model bill was the FLOOR, not convergence.
# Expected saving ~54% of survivor repeats -> ~44% of the iteration.
#
# ⚠ (window, thresh) IS A RATE: thresh/window lnL per repeat. Shortening
# the window at a fixed dll does NOT let survivors stop sooner at the same
# bar, it LOWERS the bar -- 250->100 at dll 4.0 would be 2.5x looser. The
# survivor dll is therefore DERIVED (4.0 * 100/250 = 1.6) so the enforced
# rate is IDENTICAL for both classes and only the earliest permitted exit
# moves. tests/test_inmodel_converge.py pins that, with a control showing
# what the unscaled version would have done.
# 0 = off (both classes share GB_INMODEL_CONVERGE_ITERS, pre-2026-09-25).
# Watch: [GB_IMCONV ... armed] now prints both windows, and the survivor
# block length should collapse toward ~105 from ~280.
# 100 -> 50 in lockstep with the 6mo script (user ruling 2026-09-26,
# measured there over 1,956,023 mature rows: p10 pinned at exactly the
# 101 floor, median gain 0.08-0.23 lnL, i.e. survivors arrive converged
# and the window is a fixed budget they serve out). The derived dll
# follows (4.0 * 50/250 = 0.8), so the enforced rate is unchanged.
# 50 -> 25 (user ruling 2026-09-27). Measured on 6mo job 650 over
# 52,951 survivor blocks: 40.2% of rows terminate AT the floor, i.e.
# they arrived already converged and we paid 51 repeats to prove it.
# The floor is window+1, so 25 proves the same thing in 26. Block
# medians run p50 60 / p75 76, so rows that DO need refinement keep
# room. ⚠ the threshold does NOT scale with this any more -- it stays
# at GB_INMODEL_CONVERGE_DLL=4.0, the same D/2 bar the band valve and
# the leaf cap use (GB_INMODEL_CONVERGE_DLL_SCALE_WITH_WINDOW=1
# restores the old rate-preserving rescale).
export GB_INMODEL_CONVERGE_ITERS_SURVIVOR=25
# The improvement threshold: D/2 = 0.5 * GB_LEAF_CAP_NDIM, the lnL a
# genuinely new D-parameter source has to buy. Same number as the leaf cap
# gate, but see the warning above -- same threshold, different clock.
export GB_INMODEL_CONVERGE_DLL=4.0
# Per-row ceiling. 0 would mean 4x the class budget = 400, which is now
# WRONG twice over: with W=250 the floor alone is 251, and 400 caps rows
# that are still climbing hard.
#
# THE EVIDENCE FOR GOING BIG (job 620 calibration, per rank):
#   lnL gained per row p10/p50/p90 = -12.80 / 21.57 / 222.53, max 1323.1
#   and on another rank max 4988.0
#   at the 400-repeat ceiling: 3.4-4.2% of rows
# The p90 of 222 and those maxima are rows still buying hundreds to
# thousands of lnL when the cap cut them off. Capping a row that is still
# adding logL is the one thing this knob should never do (user ruling
# 2026-09-25: "make the ceiling much larger than 1000 honestly if it is
# still adding logL").
#
# 5000 -> 20000 (user, 2026-09-25). At this size the ceiling stops being
# a budget and becomes a TRIPWIRE: the measured need is p50 260 / p90 400
# repeats, so a row reaching 20000 has run ~50x the p90 without its
# accumulated accepted delta_ll ever going W=250 repeats without a new
# best. A new loud [GB_IMCONV] warning fires whenever ANY row hits it --
# and says not to raise it again, because at that point the suspect is the
# THRESHOLD (DLL) or a row that is not converging at all (a hot rung that
# slipped the ladder gate, a source oscillating between modes).
#
# ⚠ WHY THIS IS NOT A COST BLOWUP. The ceiling is a BACKSTOP, not the
# primary control. Two other things stop a block first:
#   * the convergence rule itself -- 94% of gated rows retired on evidence
#     at W=100, not on the cap;
#   * GB_INMODEL_CONVERGE_STOP_FRAC=0.5 ends the whole block once half the
#     GATED rows have converged, and un-retired rows are released with it.
# So the ceiling only binds inside a block that is already running long,
# and raising it buys exactly the rows that are still paying. If the
# [GB_IMCONV] ceiling fraction climbs anyway, the thing to move is the
# THRESHOLD (GB_INMODEL_CONVERGE_DLL), not this.
export GB_INMODEL_CONVERGE_MAX=20000
# LADDER GATE: only the COLDEST half of the 24 rungs is tested, frozen, and
# allowed to hold a block open; hotter rungs keep sampling (they are the
# transport that feeds the cold rungs) and are reported "released".
# ⚠ Not cosmetic. A hot chain accepts freely, so its gain RANDOM-WALKS and
# the running max of a random walk keeps creeping -- over a 100-repeat window
# that clears D/2, so without the gate the hot rungs would be permanent
# stragglers holding every finished cold source hostage. 1.0 = every rung
# must converge (the pre-gate behaviour).
export GB_INMODEL_CONVERGE_GATE_FRAC=0.5
# COLD-CHAIN-ONLY COLUMN RETIREMENT (user ruling 2026-09-27: "with the
# vertical swaps working well, we should pull the whole group off the
# current running block when its cold chain converges").
#
# ABSOLUTE rung count, overrides GATE_FRAC above. =1 means the T0 row is
# the only one with a vote on when a (walker, band) column may leave the
# in-model block; the whole column goes together and the hot rows are
# recorded as RELEASED (nothing concluded) rather than converged.
#
# NOT the fraction: the exact ratio 1/24 does give one rung, but this file
# exports a typed decimal and the answer flips on the fourth place --
# ceil(24 * 0.04) is 1, ceil(24 * 0.0417) is 2.
# 1 -> 4 (user ruling 2026-09-27, refining the 09-27 cold-only
# ruling): "make sure that all 4 of the coldest 4 chains have
# either no picked source or its source has converged to a max
# logL." The coldest FOUR rungs gate the column, not just T0.
#
# Paired with the two-phase block: until those four are done,
# EVERY row runs to its own convergence; once they are, that
# column's remaining rungs get a fixed tail --
# GB_INMODEL_POST_GATE_NEWBORN=100 steps for births,
# GB_INMODEL_POST_GATE_SURVIVOR=0 for survivors.
export GB_INMODEL_CONVERGE_GATE_RUNGS=4
# SWAP TRIGGER: end a block once this fraction of its (walker, band) COLUMNS
# has fully retired, then swap the finished bands out for queued ones.
# ⚠ 1.0 SILENTLY DISABLES THE REFILL -- at 1.0 every column finishes together,
# nothing carries, and the loop degenerates into the fixed chunk loop.
export GB_INMODEL_CONVERGE_STOP_FRAC=0.5
# Retire finished bands and give their buffer slots to queued ones.
# ⚠ COSTS ONE BLOCK SETUP PER GENERATION, measured at ~73 in-model repeats
# (cholesky + sig-het reference + buffer refill). This is the one part of V9-1
# that can plausibly be net-negative on wall clock. A/B it: the [GB_IMCONV]
# line prints the generation count and the work ratio, and 0 turns the loop
# off while keeping the freeze.
#
# ⚠⚠ THE REFILL ONLY DOES ANYTHING IF THE POOL EXCEEDS THE CHUNK WIDTH.
# It swaps finished bands out for QUEUED ones, so with no queue it runs one
# generation and is a pure no-op. Check before crediting or blaming it:
# v8's own line "<N> survivors polished in <K> in-model chunk(s)" -- if K is
# 1 per class, there was never a queue and REFILL=1 changes nothing.
# ⚠ And note the setup accounting at THIS configuration: with
# GB_INMODEL_SETUP_BATCH=2048 a chunk is already split into ~
# (rows / 2040) sub-blocks, each paying its own setup. So a generation costs
# that many setups, not one, and the refill's marginal cost is one setup per
# 2040 CARRIED rows. Price it from the [GB_IMCONV] generation count, not from
# a single 73-repeat figure.
export GB_INMODEL_CONVERGE_REFILL=1
# BOTH CLASSES. User ruling 2026-09-25, final form: "What I want is RJ
# moves to have their sources run to convergence (in the search) whether
# they are birthed or survived."
#
# A pooled row is MATURE when it was alive at pick and survived the RJ
# step -- it was judged for death and KEPT. Under "newborn" those rows took
# the flat GB_INMODEL_REPEATS_SURVIVOR budget (50) while newborns
# converged, so a source that had just proved it was worth keeping got the
# LEAST polish in the pool. Both classes now run the window (250) /
# ceiling (20000) rule and stop on their own evidence.
#
# ⚠ TWO SCOPES, BOTH WANTED -- this is the distinction I got wrong twice.
#   * THIS knob is the per-ROW rule inside an RJ move's own pool. It is
#     what the ruling above is about.
#   * The pure in-model slots (in_model / in_model_fstat /
#     in_model_replace) that follow each RJ move converge per SUB-BAND via
#     the GROUP rule ([GB_IMGROUP]). That is the "let the in model moves
#     mix them up" half, it is a different scope, and it is untouched --
#     _run_in_model_repeats is not passed a converge state on that path.
#   Arming "mature" here therefore does NOT change the in_model* budgets.
#
# ⚠ COST -- the largest single increase in this batch. Newborns are a
# minority of the pool; mature rows are the bulk once the search has found
# most of the catalogue. Bounded by the same two mechanisms as newborns:
# the stop is evidence-based, and GB_INMODEL_CONVERGE_STOP_FRAC=0.5 ends a
# block once half the gated rows are done. Watch [GB_IMCONV ... mature]:
# a high "at floor" fraction means mature rows plateau immediately and the
# convergence is buying them nothing.
export GB_INMODEL_CONVERGE_CLASSES=newborn,mature

# ===========================================================================
# V9-6: IN-GROUP CONVERGENCE FOR THE PURELY IN-MODEL PROPOSALS
# ===========================================================================
# THIS IS A DIFFERENT SCOPE FROM V9-1 ABOVE. Three convergence rules are now
# live and they share vocabulary, so, explicitly:
#
#   unit      statistic                 clock              lifetime
#   row       accepted delta_ll         one in-model REPEAT  one repeat block
#             (V9-1, newborns only, inside the RJ moves)
#   (w,band)  sub-band cold-chain logL  one in-model PASS    ONE proposal   <- V9-6
#   (w,band)  sub-band cold-chain logL  one sampler ITER     one recipe stage
#             (the per-walker band shutoff, other session)
#
# V9-6: the PURE in-model move repeats its whole pass until every
# (walker, band) sub-band's cold-chain logL has converged, shutting each
# sub-band off as it converges, and ends when all are shut. Pure in-model
# moves only -- armed on an RJ move it logs a warning and does nothing,
# because on an RJ pass a flat sub-band logL means the BIRTHS stopped
# paying, which is the stage valve's decision, not this one's.
# ⚠ THE GROUP RULE IS DEAD WITHOUT THIS PAIR. The move it lives on is the
# PURE in-model move, which is built only when search_in_model is set and
# is otherwise absent from the stage entirely -- so with GB_SEARCH_IN_MODEL
# unset the knobs below are exported, ignored, and nothing logs [GB_IMGROUP]
# at all. That is exactly the silent no-op the preflight exists to catch.
export GB_SEARCH_IN_MODEL=1
# Repeats per source per PASS of that move (the group rule then decides how
# many passes). 25 per the user's structure.
#
# ⚠ DO NOT RAISE THIS EXPECTING A SPEEDUP. It was briefly set to 100 on
# 2026-09-25 and reverted the same day, because the group rule's window is
# GB_INMODEL_GROUP_ITERS *PASSES* and one pass is this many repeats per
# source -- so the EFFECTIVE window is 3 x this, against a FIXED
# GB_INMODEL_GROUP_DLL of 4.0:
#
#     25  -> 4.0 over  75 repeats = 0.053 lnL/repeat
#    100  -> 4.0 over 300 repeats = 0.013 lnL/repeat   (4x STRICTER)
#
# A fatter pass posts a proportionally LARGER per-pass gain, so the 3-pass
# total stays above 4.0 for MORE passes, not fewer. Raising this is 4x the
# work per pass AND a tighter bar. (Unlike the per-ROW rule in V9-1, this
# convergence is per SUB-BAND, so there is no per-source averaging to
# absorb it.) If fatter passes are ever wanted, GB_INMODEL_GROUP_DLL must
# scale with them -- 50 pairs with dll 8.0 -- to hold the rate.
export GB_NUM_REPEAT_PROPOSALS=25
export GB_INMODEL_GROUP=1
# W in PASSES (not repeats): one pass is already GB_NUM_REPEAT_PROPOSALS
# repeats per source, so the per-pass gain is a far coarser quantity than
# the per-repeat one and needs nothing like the 100-repeat window V9-1 uses.
# 3 -> 2 (2026-09-26, carried from submit_gf_6mo_v9_4gpu.sh f7f84a89 -- the
# window is in PASSES and dll stays at D/2, so this tightens the rate the
# group must sustain). NOT a 3mo-specific knob: the twin test caught it
# drifting and the 3-month run takes the 6-month value by the derive rule.
# 2 -> 3 (user ruling 2026-10-01, "for now"; carried from the 6mo script:
# one more pass of history before a sub-band can retire while the noise
# is moving under the galfor ratchet).
export GB_INMODEL_GROUP_ITERS=3
# D/2 again, and FLAT per sub-band (user ruling): "When a source is birthed,
# it is per-source. During the special in-model only proposals it is
# per-sub-band. I want flat D/2. This will focus more resources on the
# sub-bands with more sources." The occupancy scaling is the INTENT -- a
# dense band posts a larger per-pass gain, clears the flat bar for longer,
# and so earns more passes. GB_INMODEL_GROUP_SCALE=per_source is the
# alternative; both numbers are logged every pass.
export GB_INMODEL_GROUP_DLL=4.0
export GB_INMODEL_GROUP_SCALE=flat
# ⚠ COST KNOB, NOT A SAFETY NET. This group is unbounded by construction:
# one pass is a full sweep of every source at 25 repeats each, so 50 passes
# is up to 1250 repeats/source for the slowest sub-band. Watch the
# [GB_IMGROUP] "group done" line's pass distribution and tighten if the p90
# is at the ceiling.
#
# 20 -> 50 (user, 2026-09-25: "this may need to be higher"). Two reasons:
#
#  * SCALE=flat above makes crowded bands structurally slower to shut --
#    that is the INTENT -- so the ceiling truncates exactly the bands the
#    design is trying to favour. At 20 passes a band gets 500
#    repeats/source, and the row-level rule measured on job 620 needs a
#    MEDIAN of 260 and a p90 of 400 repeats for a SINGLE newborn to
#    converge. 20 was ~1.25x the p90 for one source, for a bar the whole
#    band has to clear in aggregate.
#  * raising it is nearly free. A shut sub-band leaves the eligible set
#    (``_group_shutoff_wb`` is the next pass's filter), so late passes only
#    touch what is still open -- with ~61 columns, one stubborn band costs
#    ~1/61 of a pass. The expensive case is MANY bands still open, and that
#    trips the ceiling WARNING, which names the remedy.
#
# ⚠ 50 is reasoned, not measured: the group rule was inert on the fan-out
# path until 2026-09-25, so no run has ever produced a [GB_IMGROUP] pass
# distribution. The first one settles this number.
export GB_INMODEL_GROUP_MAX_PASSES=1000
echo "[V9-IMGROUP] group=${GB_INMODEL_GROUP} iters=${GB_INMODEL_GROUP_ITERS} \
dll=${GB_INMODEL_GROUP_DLL} scale=${GB_INMODEL_GROUP_SCALE} \
max_passes=${GB_INMODEL_GROUP_MAX_PASSES}"
# ---- V9 FEATURE PREFLIGHT --------------------------------------------------
# ⚠ AN UNRECOGNIZED ENV VAR IS SILENTLY IGNORED (the stock-settings rule in
# CLAUDE.md). Every V9-1/V9-6 knob above is new, so running this script
# against an installed lisatools that predates them yields a run that looks
# completely normal, logs nothing unusual, and quietly behaves like v8 with
# caps off. That is the worst possible failure: a wasted allocation that
# produces plausible output. Refuse instead.
#
# ⚠ AND A PRESENCE CHECK IS NOT ENOUGH. The 2026-09-24 audit found the group
# knobs DEAD while every symbol below existed: the resolver built the env
# name ``GB_INMODEL_CONVERGE_GROUP*`` while this script, every docstring and
# every log line said ``GB_INMODEL_GROUP*``. ``hasattr`` cannot see that. So
# the second half of this preflight RESOLVES the knobs from the environment
# this script has actually exported and refuses if a value does not come
# back -- the only check that covers a name mismatch.
python - <<'PYEOF' || exit 2
import sys
from lisatools.globalfit.moves import gbspecialstretch as g

missing = [n for n in ("_InModelConvergeState", "_InModelGroupState",
                       "_converge_stage_allows", "_converge_gate_mask",
                       "_resolve_converge_knob")
           if not hasattr(g, n)]
if missing:
    print("[V9-PREFLIGHT] REFUSING: the installed lisatools has no "
          f"in-model convergence support (missing {missing}). The "
          "GB_INMODEL_CONVERGE / GB_INMODEL_GROUP knobs this script exports "
          "would be SILENTLY IGNORED and the run would quietly be v8. "
          "Install dev at or after the gb-inmodel-converge merge.")
    sys.exit(2)

# RESOLVE, do not merely probe. Each entry is (knob suffix, family, cast,
# a default this script's exported value must DIFFER from).
# The sentinels must be values the cast ACCEPTS (so an unset knob comes back
# as the sentinel and is reported, rather than raising out of the cast and
# hiding which knob was missing) and must differ from what this script sets.
_checks = [
    ("", "INMODEL_GROUP", g._converge_cast_refill, False),
    ("iters", "INMODEL_GROUP", g._converge_cast_window, 999999),
    ("dll", "INMODEL_GROUP", float, -12345.0),
    ("scale", "INMODEL_GROUP", g._converge_cast_scale, "per_source"),
    ("max_passes", "INMODEL_GROUP", int, 999999),
    ("", "INMODEL_CONVERGE", g._converge_cast_mode, "off"),
]
bad = []
for knob, family, cast, sentinel in _checks:
    try:
        got = g._resolve_converge_knob("gb", knob, None, sentinel, cast,
                                       family=family)
    except TypeError:
        bad.append(f"{family}_{knob or '<master>'} (resolver has no 'family' "
                   f"parameter -- this lisatools predates the fix)")
        continue
    except Exception as exc:
        bad.append(f"GB_{family}{'_' + knob.upper() if knob else ''} "
                   f"did not resolve: {exc!r}")
        continue
    if got == sentinel:
        bad.append(f"GB_{family}{'_' + knob.upper() if knob else ''} "
                   f"resolved to the sentinel {sentinel!r}, i.e. the "
                   f"exported value was NOT SEEN")
if bad:
    print("[V9-PREFLIGHT] REFUSING: exported knobs are not reaching the "
          "code:\n  - " + "\n  - ".join(bad))
    sys.exit(2)

# The v9 recipe restructure and its per-stage profile.
from lisatools.globalfit import recipe as _r
from lisatools.sampling import fstat_proposal as _fp
_need = [("recipe.SearchStageProfileStep", hasattr(_r, "SearchStageProfileStep")),
         ("recipe.force_fstat_refit", hasattr(_r, "force_fstat_refit")),
         ("recipe.band_shutoff_w_armed", hasattr(_r, "band_shutoff_w_armed")),
         ("fstat_proposal.set_peak_min_F_override",
          hasattr(_fp, "set_peak_min_F_override")),
         ("GBSpecialRJFStatGridMove.arm_fstat_refit",
          hasattr(g.GBSpecialRJFStatGridMove, "arm_fstat_refit")),
         ("Stage kind 'gb_search'", "gb_search" in _r.Stage._KINDS)]
_absent = [n for n, ok in _need if not ok]
if _absent:
    print("[V9-PREFLIGHT] REFUSING: the installed lisatools has no v9 "
          f"three-stage search support (missing {_absent}). STAGE_V9_SEARCH "
          "would fail at composition, or worse, compose a single legacy "
          "gb_search stage with none of the per-stage profiles applied.")
    sys.exit(2)
# The other two knobs that are NEW on 2026-09-24 and SILENTLY IGNORED on an
# older install. Neither goes through _resolve_converge_knob, so the sentinel
# pass above cannot see them; a plain source probe is enough, because both
# are read with a literal os.environ.get at a known site.
import inspect
# NOTE the fancy-tempering name is built from an f-string
# (f"{BRANCH}_RUN_FANCY_TEMPERING"), so the literal is not in the source --
# probe the SUFFIX, which is what the f-string contains.
_src_checks = [
    ("_RUN_FANCY_TEMPERING", inspect.getsource(g),
     "the permuted swaps would stay ON and the ladder would keep adapting "
     "from them -- v9 expects them OFF with the ladder adapting from the "
     "VERTICAL swaps instead"),
    ("GB_REPLACE_WARM_PASS", inspect.getsource(_r),
     "rj_replace would run SINGLE-pass (F-stat only); the warm pass the "
     "2026-09-24 ruling asked for would silently not happen"),
]
_stale = [f"{n}: not read anywhere in the installed module -- {why}"
          for n, src, why in _src_checks if n not in src]
if _stale:
    print("[V9-PREFLIGHT] REFUSING: knobs this script exports are not "
          "implemented in the installed lisatools:\n  - "
          + "\n  - ".join(_stale))
    sys.exit(2)

print("[V9-PREFLIGHT] in-model convergence + group knobs RESOLVE from the "
      "environment; v9 stage-profile support present; "
      "GB_RUN_FANCY_TEMPERING and GB_REPLACE_WARM_PASS are implemented.")
PYEOF
# WHAT TO WATCH (grep '[GB_IMGROUP]'):
#   * "pass N: cold dlnL +X over M occupied sub-band(s); shut A (+B this
#     pass) / open C (D occupied); mean gain/occupied band ..., per source
#     ..." -- one line PER PASS. The per-source column is what tells you
#     whether flat-vs-per_source would have behaved differently.
#   * "group CONVERGED after N pass(es)" or "group hit the N-pass CEILING
#     with M sub-band(s) still open" -- the ceiling line is a WARNING and
#     means the group is being truncated, not finishing.
#   * "group done: N pass(es) x R repeats/source; shut-off pass
#     p10/p50/p90/max ... over OCCUPIED pairs; K pair(s) never shut.
#     Compare against a fixed 5-pass block: Fx the passes." -- the cost
#     line. F > 1 means this is costing more than the 5-pass block it
#     replaced.
# ===========================================================================
echo "[V9-IMCONV] mode=${GB_INMODEL_CONVERGE} iters=${GB_INMODEL_CONVERGE_ITERS} \
dll=${GB_INMODEL_CONVERGE_DLL} ceiling=${GB_INMODEL_CONVERGE_MAX:-0}(0=4x) \
gate_frac=${GB_INMODEL_CONVERGE_GATE_FRAC} \
stop_frac=${GB_INMODEL_CONVERGE_STOP_FRAC} \
refill=${GB_INMODEL_CONVERGE_REFILL} classes=${GB_INMODEL_CONVERGE_CLASSES}"
# WHAT TO WATCH, in priority order (all at INFO, one set per class per
# in-model phase; grep '[GB_IMCONV]'):
#   1. "... calibration: ... AT FLOOR (<=101) N (X%); lnL gained per row
#      p10/p50/p90 ... rows gaining < dll(4.00) = M (Y%)"
#      THE line. X near 100% with Y near 100% = the rule never fired on
#      evidence and is just a constant budget of ITERS -- recalibrate DLL
#      against a measured delta_ll distribution, do not just ship it.
#      A broad repeat spread with p50 gain of order D/2 and up = working.
#   2. "... work = N.NNx a fixed 100-repeat budget, plus K extra block
#      setup(s) from the refill" -- the cost. Multiply K by ~73 repeats to
#      price the refill against the repeats saved.
#   3. "... by rung: GATED ... | RELEASED ..." -- the two populations are
#      judged by different rules; a pooled number hides which one is paying.
#   4. "... armed (on): window ... gate = coldest 12/24 rung(s) ..." -- one
#      line per move at first use, and it NAMES every move that armed. The
#      mode refuses PE-stage moves (optional stopping is search-only) and
#      says so once; confirm rj_fstat_search/rj_prior_removal are in and
#      nothing named *_pe is.
# ===========================================================================

# VERTICAL TEMPERING ON (2026-08-26 user ruling: "this is crucial").
# Per-repeat vertical band-temperature swaps inside the in-model loop
# (same walker, adjacent rungs) -- built+tested 2026-08-18 (e6ed71e2,
# 30 tests) but never promoted: the flag sat default-off through the
# v6/overlap/replace campaigns. It is ADDITIVE to the permuted swaps
# and never drives ladder adaptation. Directly attacks the measured
# transport bottleneck (correct fdot living at rung 4 while cold holds
# the mosaic): exchange cadence per in-model REPEAT (~100x/block) vs
# once-per-iteration permuted swaps. Known approximation: a mid-block
# vertical swap exchanges occupancy without updating the drift-gate
# census (self-corrects next block). =0 reverts.
# ===========================================================================
# V9-7: PERMUTED ("FANCY") BAND-TEMPERATURE SWAPS OFF (user ruling
# 2026-09-24: "make sure the run_tempering is entirely turned off because
# we have no fancy swaps").
# ===========================================================================
# ⚠ THIS IS THE ONLY RELIABLE OFF SWITCH, and the obvious candidates are
# both traps:
#   * GB_TEMPER_EVERY_PROPOSES only THROTTLES. In _temper_cadence_fire,
#     `n <= 1` returns True unconditionally, so 0 means "fire on EVERY
#     eligible propose" -- the exact opposite of off.
#   * per-move run_swaps= is set from several different expressions in the
#     recipe (`_temper_all_moves or not _temper_on_removal`, hard False on
#     some moves), so there is no single recipe value to flip.
# GB_RUN_FANCY_TEMPERING=0 forces run_swaps False for every GB move at once, at
# the one choke point in the constructor, and logs once that it did.
export GB_RUN_FANCY_TEMPERING=0
# ⚠ AND NOTE WHAT THIS DOES **NOT** TOUCH: the per-repeat VERTICAL rung
# swaps below. They live inside the in-model loop, are ADDITIVE to the
# permuted swaps rather than part of them, and are the transport the
# in-model convergence work depends on -- killing the fancy swaps must not
# silently take them too. They stay ON.
# ===========================================================================
export GB_TEMPER_VERTICAL=1

# ---- ALL-RUNGS VERTICAL SWAP (user ruling 2026-09-26: turn it on) --------
# Swap EVERY rung of a (walker, band) column, not just the rungs that
# happen to hold a picked source. "Even if it does not have a source in
# that band it has a total likelihood ... and should get swapped
# accordingly." A cold cell whose whole model is net-harmful against the
# bare slab can now be evicted upward -- the route recorded as missing in
# project_sole_occupant_swap_gap.
#
# Inactive rungs are NOT made resident: their total is computed once per
# block and swapped against as a number ("you do not need to load
# information into the buffer for good for bands without an active
# source"). Buffer residency is unchanged.
#
# ⚠ TURNED ON AHEAD OF ITS OWN GATE. The design spec asks for a clean
# [GB_CELL_LL] sampled-vs-realized line on a real run before this stops
# being opt-in; it has unit tests and five mutation controls but no
# production hours. WATCH [GB_CELL_LL] on the first snapshot -- both of
# this path's failure modes (a mis-credited cell ledger, a half-applied
# relabel) are invisible in aggregate counts and show up ONLY there.
# Also watch [GB_VERT] for "N pair(s) dropped as unpriceable" (an
# occupied-but-unpicked rung, whose total nothing measured) and for the
# per-rung acceptances staying BELOW ~1.0 -- an empty<->empty pair would
# be accepted unconditionally and collapse the band's ladder, which is
# why those pairs are dropped.
# GB_TEMPER_ALL_RUNGS=0 restores the picked-row-only sweep exactly.
export GB_TEMPER_ALL_RUNGS=1

# PERMUTED-SWAP CADENCE 3 -> 1 (user ruling 2026-08-26): fire the
# permuted band swaps after EVERY GB propose -- 3x/iteration in search
# (was once, on the third move), and every PE iteration (was every ~2-3:
# the measured PE transport drought). Pairs with vertical: permuted
# swaps move whole band contents between rungs, vertical pumps
# per-repeat during polish -- the full transport stack. Probe cost
# ~+40 s/it (tempering block x3); production ~+3%. =3 reverts.
export GB_TEMPER_EVERY_PROPOSES=1
# ---- 10-WALKER-ARM TEMPERING STACK (ported 2026-09-11) --------------------
# COLUMN STAGING (b3f5acbd): every rung of a (walker, band) column staged
# together, pools ordered (band, walker, temp). REQUIRED with
# GB_TEMPER_VERTICAL=1 above -- jobs 465-469 ran order=count and the
# vertical sweep starved/biased (the leaf-shedding root cause pair); job
# 473 on order=band + the whole-cell L_with ratio is the validated config.
export GB_TEMPER_CELL_ORDER=band
# RUNNING-BACKUP CADENCE (2026-09-14, ported from the 3mo 10-walker arm):
# job 479 spent 60-80 s of every 5.4 min PE iteration in [SAVE] save_step
# -- the sampler's blocking handoff waiting for the saver rank to finish
# copying + fsyncing the 14 GB store into *_running_backup_copy.h5 after
# EVERY save. Every 10th save instead; the 600 s mid-iteration checkpoint
# still bounds a torn-store loss to that interval. The 6mo store will be
# larger still, so this matters MORE here.
export BACKUP_ITER=10
export GB_TEMPER_PRELOAD_CELLS=4800
# One occupancy census per unit (assert-guarded exact) + drop inert rows.
export GB_TEMPER_CENSUS_HOIST=1
export GB_TEMPER_COMPACT_ROWS=1
# Adaptive gb_search noise rider (379ae2e1): keep taking MaxLogL rounds
# while a round improves by > tol, stop at the first flat one.
export GB_SEARCH_NOISE_CHECKS=1
export GB_SEARCH_NOISE_ITERS_PER_STEP=100
# PLATEAU TOLERANCE 5 -> 20 lnL (user ruling 2026-09-18). MAXLOGL_TOL is what
# counts as "this round improved"; JointMaxLogLSearch keeps taking rounds while
# a round beats it and stops at the first flat one.
#
# Measured on the 6-month 4-GPU run: the rider took ~11.5 rounds per gb_search
# iteration and ran INTO its 10-round ceiling (GB_SEARCH_NOISE_ITERS_PER_STEP=0
# resolves to MAXLOGL_ITERS_PER_STEP, default 10), costing 101.4 s of a 320 s
# steady-state iteration -- 32%, the single largest line item. The [MAXLOGL]
# trace says where that goes: ROUND 1 carries the real re-tracking (IMPROVED
# jumps of hundreds of lnL), rounds 2+ add ~5 lnL each -- exactly AT the old
# tol, so the plateau rule could never fire and the loop ground to the cap
# chasing the tol-level wobble of an ensemble already near the mode.
#
# 20 keeps round 1 whenever the GB residual genuinely moved and cuts the
# wobble. Against a cold logL of ~1.04e8 gaining ~2e4 per iteration, a 20 lnL
# floor is far below anything that matters.
#
# SCOPE, deliberately noted: MAXLOGL_TOL is GLOBAL to JointMaxLogLSearch, so it
# also loosens the standalone noise_search / noise_vgb_search stages, which
# will now declare plateau sooner. Those stages only need the noise roughly
# converged before the next stage samples it in PE mode (see the tol default's
# own comment), so this is judged acceptable -- but if the searches come out
# under-converged, the rider needs its own tol knob rather than reverting this.
export MAXLOGL_TOL=20
# PE-only exclusive RJ draw (b9aae51f).
export GB_PE_RJ_DRAW_ONE=1
export GB_PE_RJ_FSTAT_FRACTION=0.5
# Per-block EXACT info matrices through the sig-het fast route
# (~2.4 ms/src vs ~29-46 chunked). The data_index misindex is FIXED and
# multi-GPU slots now route by the BUFFER's slot shards. First
# shakedown: set SIGHET_INFOMAT_VALIDATE=1 for one propose to log the
# fast-vs-chunked reldiff (expect ~1e-4 near-peak, larger off-peak =
# observed-vs-Fisher, fine for a proposal), then remove.
# ---- V4 sig-het block (2026-08-18) ------------------------------------
# SPARSE TIME GRID. Every production script has been running ~35 h sparse
# spacing -- 3-mo default 64 (snaps to 60, stride 36), 6-mo 120 (stride 36),
# 23-mo 525 (stride 32) -- the "constant temporal density" prescription from
# the accuracy studies. Measured on the high-f probe 2026-08-18, the compiled
# BANDED v5 kernel wants finer than that: the delta-vs-delta likelihood error
# eps/T on the loud block ran 0.131 at 36 h against 0.072 at 8 h, and the
# trust-gate rejection fell from ~15% to 9.0%. The prescription was
# calibrated with gb_sighet_tier_assess.py, which builds its engine with NO
# v3_n_nodes / v4_knots / v4_band / v5 (i.e. the v2 path) and never calls a
# compiled v3/v4/v5 kernel at all -- so it cannot speak to this engine.
# 270 is an EXACT divisor of Nt=2160 (stride 8 -> 8.0 h), so it lands
# cleanly instead of snapping. If it exceeds the device shared budget the
# fstat scorer FAILS AT SETUP naming the largest value that fits (~2 min,
# not a wasted run) -- it is never silently coarsened.
#
# ---- REVERTED 2026-08-18: 270 OOMs, and the accuracy case did not hold ----
# MEMORY. The sig-het stash is (n, nch, nch, Nf_active, N_sparse_t)
# complex128 in _expand_B -- 4x _expand_B + 4x _expand_A per setup -- so it
# goes as CELLS x N_sparse_t and the two knobs MULTIPLY. 270 raised
# N_sparse_t 60 -> 265 while GB_N_SUBBANDS stayed at the 3-month 8192:
#   3-mo default  60 x 8192 = 4.9e5  OK (= v3)   |  6-mo  118 x 4096 = 4.8e5 OK
#   23-mo        525 x 2048 = 1.1e6  OK          |  v4@270 265 x 8192 = 2.2e6 OOM
# It died on a 14.4 GB request at 91.5 GB allocated on a 99.9 GB card --
# 2x the 23-month run's product on the SHORTEST baseline. The confined
# probes could not surface this: n scales with cells and they ran 128
# against production's 16384.
#
# ACCURACY. The 0.131@36h vs 0.072@8h A/B above predates GBGPU b412089, so
# NEITHER run logged its resolved config -- and 0.131 is also the nodes=64
# value from the separate 32/64/128 sweep (0.103/0.131/0.093), i.e. the two
# arms are not a clean pair. With the echo finally live, four temper arms
# CONFIRMED at nt_layer=270 measured eps/T 0.087-0.102 at small
# displacement -- squarely inside that 0.093-0.131 spread, not 0.072. The
# gain did not reproduce.
#
# And it should not have been expected to. project_sighet_v4_plan.md,
# A100 bench-off 2026-08-03: at 3 months a CPU run at STRIDE 1 (maximum
# possible resolution) still gave 2.90 at T=100 vs 3.70 at the 16 h grid --
# "that tail is resolution-independent ... the known deep-null fit tail
# (remedy = SNR-aware / null-densified nodes)". Same note, longer
# baselines: "a 1.6x finer sparse grid changes NOTHING, and where it
# changes anything it is worse." The real lever is NODE PLACEMENT.
#
# The "production default is TOO COARSE" line in that note (which motivated
# 270) argues for matching the BENCHMARK's 16 h configuration; it was
# written against a SHARED-memory ceiling and never costed the global-memory
# stash above. It is not a measured 3-month accuracy deficit.
#
# TO RE-TEST: 270 needs GB_N_SUBBANDS=2048 to fit (265 x 2048 = 5.4e5), at
# a throughput cost -- fewer resident cells means more sequential passes.
# export SIGHET_NT_LAYER=270
#
# Confirm from the log, do not assume -- nothing else echoes these:
#   grep "sig-het engine resolved" <store>/gf_prod_3mo_artifacts/globalfit_run.log
# want: nt_layer=270 (stride 8) ... sparse spacing 8.0 h   (GBGPU b412089)
#
# MULTI-DEVICE F-STAT FAN-OUT. Validated 2026-08-13 (a86c52af): the =check
# gate ran on 2xH100, full comb+stageB in 122.3 s / 224 peaks with ZERO
# diverging batches vs the pinned scorer; the 2026-08-12 divergence was
# closed by the drift-campaign replica fixes. The 23-month script has used
# it since. Without it every refit runs the serialized pinned scorer. Real
# lane overlap needs a GBGPU wheel at/after 4381300 (GIL release); older
# wheels stay CORRECT but serialized, so this is safe either way.
export FSTAT_SIGHET_MULTIDEV=1
#
# SIG-HET REFERENCE REFRESH (user ruling 2026-08-18). The trust gate measures
# drift from ``ref_track`` -- the parameters the sig-het reference was built
# at -- so REFRESHING THE REFERENCE RESETS THE BUDGET. With no refresh a
# source spends all 200 repeats accumulating against one fixed expansion
# point, which is exactly why the drift audit pinned at max 0.47-0.50 of a
# 0.5 budget in EVERY block of every probe.
#
# BOTH knobs are required. ``REFRESH_EVERY`` is only the cadence at which
# FARNESS IS CHECKED; the refresh fires only for sources with
# ``drift > sighet_refresh_dphase``, which defaults to 0.5 -- the trust gate
# itself. The gate stops the drift one step before the refresh would notice,
# so the phase arm can NEVER trigger at the defaults: two knobs that must
# differ ship equal. DPHASE=0 makes it "refresh anything that moved" (a
# source that never accepted a move still has an exact reference).
#
# The counter is REPEATS, not iterations: newborn blocks (200) get 8
# refreshes, survivor blocks (25) get none -- the `move_i + 1 < n_rep` guard
# -- which is right, they barely drift.
#
# Cost, measured on the high-f probe with this exact configuration:
# inmodel_sighet_refresh 0.25 s typical / 6.26 s worst against
# inmodel_repeats 5.2-15.1 s and an 85-98 s propose = 0.3% typical, 6.4%
# worst. In-model is only ~5% of a propose (rj_step is 87%).
#
# WHY THIS RATHER THAN WIDENING THE GATE: refreshing re-linearizes, so it
# buys mixing while PRESERVING accuracy; widening buys the same mixing by
# SPENDING accuracy. Same reason GB_SIGHET_TRUST_PHASE_C stays at 0 here.
# >>> 25 -> 50 (2026-09-25): inmodel_sighet_refresh was the single largest
# >>> line item inside the in-model polish in job 628 (167-186 s per RJ
# >>> move). Halving the cadence halves that. The reference goes staler
# >>> between rebuilds, so the trade is mixing against accuracy -- and both
# >>> sides are instrumented: watch [GB_TRUST] (was rejecting 4-13% of
# >>> candidates) and the end-of-block 'll AUDIT vs exact' COLD median (was
# >>> 0.026-0.073, max 0.14-0.73). If either degrades materially, revert.
export GB_SIGHET_REFRESH_EVERY=25
# 50 -> 25 (2026-09-28, user ruling). Paired with
# GB_TEMPER_VERTICAL_AT_REFIT below: the vertical sweep now runs ONLY on
# repeats that re-anchored the references, so the swap and the refit
# land in the same repeat. A cold row that swaps up and down across
# 25-50 proposals against references anchored somewhere else can lose
# proper dll tracking -- ll_ref is an add-delta against the reference it
# was built on, so the further a row has walked since, the more the swap
# ratio compares two cells through different expansion points.
# Expected cost signature: inmodel_sighet_refresh roughly DOUBLES
# (twice as many refit ticks) while inmodel_vertical_swap COLLAPSES
# (~25x fewer sweeps). If vertical_swap does not fall, the gate is not
# taking -- check [GB_VERT] '(N sweep(s))'.
export GB_TEMPER_VERTICAL_AT_REFIT=1
# DPHASE 0 -> 0.1 rad (2026-09-11, ported from the 10w arm 379ae2e1):
# refresh only sources drifted past 0.1 rad since their reference build.
# Error bound at 0.1 rad = 2.4e-7 x SNR^2/2 = 0.004 lnL at SNR 184
# (Experiment A); at DPHASE=0 job 473 refreshed 89% of sources every
# check for ~20 s/it of pure rebuild. The 2026-08-18 "ALL RUNGS REFRESH"
# ruling below was written against the 0.1 DEFAULT's rung gating -- this
# knob is a drift threshold, not a rung filter, so hot rungs still
# refresh once they drift.
# REVERTED to 0 (2026-09-11 evening, ported from the 3mo 10-walker arm):
# 0.1 rad refreshed 81% of sources instead of 89% at 3mo and 94% instead
# of 97% at 1yr -- no measurable wall-clock saving -- while the 1yr cold
# audit max rose 15 -> 22 in the same relaunch (cause not isolated).
# Accuracy is primary: refresh every check.
export GB_SIGHET_REFRESH_DPHASE=0
# ALL RUNGS REFRESH (user ruling 2026-08-18). The default 0.1 keeps a stale
# reference on everything hotter, justified in the code as "the ll error is
# beta-suppressed". That reasoning covers the WITHIN-rung accept test, where
# the error enters as beta*eps -- but NOT the tempering swap, where it enters
# as (beta_i - beta_j)*eps. On a geometric ladder from 1.0 to 1e-4 over 24
# rungs the adjacent ratio is 0.687, so beta_i - beta_j = 0.313*beta_i, and a
# stale-reference error of 1e3 lnL (the tail the probes measured off the cold
# chain) contributes ~31 at beta=0.1 and ~3 at beta=0.01. Swaps at those
# rungs would be decided by reference staleness rather than by the data.
# Only around beta ~ 1e-4 does it genuinely vanish (~0.03).
#
# Cost: this refreshes every rung instead of the ~top third, so roughly 3x
# the measured refresh time -- ~0.9% of a propose typically, ~17% in the
# heaviest propose observed. Against rj_step at 87% of the propose that is
# ~12% wall clock worst case, and it buys swap ratios that mean something.
export GB_SIGHET_REFRESH_MIN_BETA=0
#
# SNR-SCALED TRUST GATE (measured 2026-08-18, high-f probe A/B). The uniform
# 0.5 rad gate is the wrong SHAPE: the tiered spec places gates at a constant
# TRUE-lnL displacement T, but a fixed phase offset sits at
# T = 0.5*(dphase*SNR/3.456)**2 -- T~0.7 at SNR 8 against a design point of
# T~1000, while being ~9 sigma for a loud source. It strangled exactly the
# faint population the completeness deficit lives in.
# C_phase = 3.456*sqrt(2*T_gate); 49 -> T=100. Clipped BELOW by
# sighet_trust_dphase, so this can never tighten the gate for anyone.
#
# A/B result, same nt_layer, only the gate differing:
#   C_phase=0   gate=[0.5..0.5] rad    -> [GB_TRUST] 13.8-23.8% rejected,
#                                          infomat cold acceptance 0.323
#   C_phase=49  gate=[0.81..9.81] rad  -> [GB_TRUST] 2.7-3.6% rejected,
#                                          infomat cold acceptance 0.404
# The gate stops being an active constraint (5-6x fewer kills) and becomes
# the rare safety net it was meant to be, and the per-walker acceptance line
# goes from many nan walkers (no in-model proposals reaching them at all) to
# full coverage. Cost: the chain travels further from its reference, so the
# end-of-block DELTA-vs-DELTA on the loud block ran 8.46 against 3.65-7.06
# elsewhere -- ~+20%, at the high end of the observed range but inside it.
# Worth it: the accumulated error affects bookkeeping and swaps; the
# rejection rate affects whether the chain moves at all.
#
# Also re-couples the refresh: with the gate up to ~10 rad, drift CAN now
# exceed the stock refresh trigger, so the two knobs stop being mutually
# exclusive (moot here -- REFRESH_DPHASE=0 above -- but it matters elsewhere).
export GB_SIGHET_TRUST_PHASE_C=49
#
# RUNG-COVERAGE AUDIT -- ARMED FOR THE FIRST FEW ITERATIONS, THEN REMOVE.
# Every probe today ran the degenerate 2-rung ladder, so we have NO data on
# 22 of this run's 24 rungs. The probes' delta-vs-delta line showed a tail of
# 1e3-1e4 lnL error OFF the cold chain (cold maxima stayed ~1-17). At
# beta=1e-4 that is suppressed; on this run's geometric ladder the middle
# rungs sit at beta ~ 0.01-0.1, where 1e3 becomes beta*eps ~ 10-100 and would
# corrupt the tempering swap ratio (beta_i - beta_j)(L_i - L_j).
# These two make DELTA-vs-DELTA report across ALL 24 rungs. Watch the "all"
# median: if it stays ~1 with isolated maxima, unset both and carry on; if it
# climbs with rung count, stop and investigate before spending days on it.
# Cost: one extra exact batched call per in-model block (measured 0.053 s
# against inmodel_repeats ~4-5 s).
# SIG-HET ANCHOR AUDIT RETIRED (2026-09-04 gate PASSED on the 3mo arm:
# all-rung delta-vs-delta p50 0.054, cold max 5.48, well inside tolerance
# over the whole run; ~18 s/iter while armed). Re-arm (=1) if a sig-het
# accuracy question reopens. Ported from the 3mo 10-walker arm.
export GB_SIGHET_ANCHOR_CHECK=0
export GB_SIGHET_DRIFT_CHECK=1
# TIER SCAN RETIRED FOR THE CLEAN RESTART (2026-08-19). It has NO iteration
# cap (the "first-few-iterations" note above it was wrong): it ran 13 extra
# scoring passes -- half of them chunked-exact -- on EVERY in-model block,
# ~0.32 s/propose, for the whole run. Its 72-block dissect record is
# captured and analyzed (see GB_SIGHET_DISSECT below); the clean production
# reading must not carry its overhead. The anchor check above stays: one
# cheap exact call per block, logging |dll@anchor| -- the corrupted-refs
# rate stays on the record for the post-fix comparison.
export GB_SIGHET_TIER_SCAN=""

# ============================================================================
# SIG-HET DISSECTION + IN-RUN ENGINE SWEEP (2026-08-19, LAT 749af2e1).
# Motivated by the 13 anchor checks in this run's own log: ll_het ~ -8e3 vs
# ll_exact ~ +3e2 AT THE EXPANSION POINT (r=1, frozen residual) for the SAME
# recurring sources (band 10 @ 1.9727 mHz x6, band 17 @ 2.9603 x3, band 5 @
# 1.3181 x2) -- corrupted references, not accuracy noise, and nothing a
# resolution knob can touch. Both riders live inside the tier scan above.
#
# DISSECT: one npz per in-model block (first 32) -- the anchor through BOTH
# engines with the d_h/h_h split (data-side vs template-side attribution),
# null-depth/masked-row stats from the engine's own c0 stash, and full
# per-source identity. ~1 extra batched call per block.
#
# SWEEP: at the first 2 blocks, every arm below is rebuilt around the SAME
# underlying chunked comp, re-anchored on the SAME frozen residual and
# scored on the SAME 512-source subset (worst anchor offenders + random
# fill -- the band-10 population is guaranteed in-sample) against ONE shared
# exact side. Arms differ by exactly one thing: the engine config. Base
# (production) config is auto-prepended as the control. A failed arm logs
# and the loop continues; the production engine is restored no matter what,
# and the run then continues normally. Budget: ~10 arms x (make_reference
# on 512 refs + 7 batched scores) -- minutes, twice.
#
# Read the verdict locally once "[GB_SWEEP] wrote" appears (the dumps ride
# home in the store-dir zip):
#   python scripts/gb_chunked_het/gb_sighet_dissect_report.py \
#       <unzipped>/gf_prod_3mo_v4/dissect
# Anchor |dll| CANNOT move under a resolution-only knob: flat across the nt
# arms + moving under node arms = deep-null node fit; flat across ALL arms
# = the reference build itself (then the dissect d_h/h_h split names which
# half). v5=0 differing from base = v5-specific; v5=2 is the flat-carve
# control arm.
# Sig-het reference-build taper, PINNED (2026-08-19 Tukey rulings: equal
# alphas across chunked/sig-het/TD->FD; error-created edges REMOVED by the
# [min_time, max_time] crop). 0.01 -> taper 11 layers + 8 margin = 19 <=
# crop 20: the build-time edge-exclusion guard passes with ONE layer to
# spare. The old inherited 0.05 tapered 54 layers against the 20-layer
# crop -- the flat ~1% h_h bias that was the dissect's high-f 0.984,
# fixed in GBGPU 88b278d / LAT 7e9c4c65+454d04bd. Verify on resume:
#   grep "sig-het engine resolved" ...  (tukey_alpha=0.01)
# and the anchor/AUDIT high-f values should move 0.984 -> ~1.000.
export SIGHET_TUKEY_ALPHA=0.01
# THE LOW-F h_h CORRUPTION FIX (2026-08-19, root-caused on the laptop from
# this run's own dissect captures). Mechanism: the make_reference spline
# reconstruction (n_cp_build control points) matches each channel to ~0.1%
# but its per-channel errors are INCOHERENT across X/Y/Z, so the GW
# template's X+Y+Z null cancellation (true null power ~1e-10 of total) is
# broken at ~1e-5 -- and the near-singular low-f XYZ invC amplifies
# exactly that direction (null eigenvalue 54-7500x the differential ones).
# Result: anchor h_h inflated up to 27x (d_h CLEAN -- the measured v4
# signature), worst for edge-on sources at low f. The AUTO n_cp law
# (4-day spacing -> 32 nodes at 3 months) was set by a phase criterion
# that never saw the null direction. Verified on the production grid
# (gb_sighet_bfold_gpu_probe.py): 32 -> 256 nodes takes the scored anchor
# from max |log hh ratio| 0.31 to 1.5e-4 at +2.6% setup cost. 256 = the
# shared-arena ceiling; pinned explicitly (the GBGPU AUTO default now
# also resolves here, this is the belt to that suspenders).
export SIGHET_N_CP=256
# UNIFORM EDGE EXCLUSION (user ruling 2026-08-19): bringing the domain
# [min_time, max_time] in removes edge-created error in EVERY likelihood
# at once (all WDMSettings inherit the one domain; min/max_freq still vary
# per source). NOTE (capture-replay verdict, same day): the crop is NOT
# the cure for the low-f h_h inflation -- the fresh reference build is
# EXACT on this very crop-20 domain; the inflation is live-stash
# corruption (see the dissect block below). The crop remains the POLICY
# knob (taper must be subsumed; constant-layers scaling, pinned in the
# 6-mo/23-mo scripts) -- flip on a FRESH STORE_DIR only: changing the
# crop changes Nt_active and resume/rewind compatibility across a
# domain-shape change is UNVERIFIED.
# ---- EDGE_CROP_WAVELETS: DELIBERATELY NOT EXPORTED AT 3 MONTHS ------------
# The 6mo script MUST pin 60: the sig-het taper is Tobs-scaled --
# ceil(alpha/2 x Nt) = 22 layers at Nt=4320 -- so taper+margin 22+8 = 30
# EXCEEDS the default crop 20 and the build-time edge-exclusion guard raises
# without it. At Nt=2160 the taper is ceil(0.005 x 2160) = 11 layers, so
# 11+8 = 19 <= the default 20 and the guard passes with ONE layer to spare.
# Taking the default is what every 3-month run has done and what
# ThreeMonthTwinTest asserts.
# ⚠ FRESH-STORE-ONLY either way: it changes Nt_active, and resume across a
# domain-shape change is UNVERIFIED. This store is fresh.
# export EDGE_CROP_WAVELETS=60     # <- the 6mo value; do NOT carry it across

# DISSECT + RAW CAPTURE RETIRED (2026-08-19, mission accomplished). The
# 72-block dissect + 25 raw-slab captures were pulled and replayed locally:
# a FRESH setup_in_model from the captured params/slabs scores EXACT
# (sig-het == direct pixel sum == production's exact side), while the LIVE
# in-run stash scored the same sources 10-35x inflated in h_h with
# wrong-signed d_h. Verdict: the reference BUILD and both engines are
# exonerated; the corruption is in the LIVE slot->reference stash lifecycle
# (prime suspect: multi-GPU router/replica sync -- the [1b] shard-swap
# anomaly), amplified at low f by the near-singular XYZ invC (null-space
# eigenvalue 54x the differential ones). Anchor errors reach the cold chain
# (median |dll| 5.5, low-f 23) but sampler-facing DELTAS track exact
# proportionally (multiplicative distortion): clean sources in spec,
# ~6% corrupted-anchor sources 0.3-2 lnL at 1-2 rad displacement. Same
# condition v3 sampled under -- newly measured, not newly introduced.
# Re-arm all three (TIER_SCAN hosts them) only to re-verify after the
# stash-lifecycle fix lands. Analysis: scratchpad replay_raw.py + the
# dissect report; capture data archived off-cluster.
export GB_SIGHET_DISSECT=""
export GB_SIGHET_DISSECT_RAW=0
# SWEEP RETIRED (2026-08-19, after 8 swept blocks): every arm answered.
# nt_layer=270 differs by 6e-8 (round-off -- resolution DEAD); m_half by
# 1e-4/1e-10 (m-window irrelevant); v5=0/v5=2 bitwise (the v4/v5
# bit-identity holding); c_Nt_sub/c_N_cp_sig bitwise (never enter the
# reference build); n_sparse_fd=2048 device-clamped; n_sparse_fd=512
# negligible; c_N_sparse=512 anomalous (uniform 2x -- suspected
# delegate-rebuild side effect in the sweep harness, NOT an engine
# result). The flat ~1% component was root-caused OFF-engine (tukey
# semantics, fixed); the remaining low-f inflation is being chased by the
# CUDA probes + the dissect below, which stays ON.
export GB_SIGHET_SWEEP=""
export GB_SIGHET_SWEEP_F0="1.9727e-3,2.9603e-3,1.3181e-3,1.6357e-3,2.2107e-3,4.2389e-3,5.1677e-3,1.2269e-2,2.0381e-2"
# 4 qualifying blocks: the low-f targets cluster in nearby bands (one or
# two units) while the high-f pair lives in different units entirely.
export GB_SIGHET_SWEEP_BLOCKS=4
# defaults, pinned for the run record:
export GB_SIGHET_DISSECT_MAX=32
export GB_SIGHET_SWEEP_MAX_SRC=512

export SIGHET_INFOMAT=1
export GB_INFOMAT_PER_BLOCK=1
# Countable-only F-stat center precompute + lookup-miss fallback rides
# the pull (no knob beyond the existing GB_RJ_FSTAT_CTR_HOIST=1); the
# new [FSTAT_CTR] census line diagnoses the job-195 5x centers blowup.
# Bilinearity bookkeeping monitor (code default ON, user ruling: ~1.5 s
# per propose = negligible): per-unit [GB_ORTHO_LL] line compares the
# realized cold parent-residual delta against the summed per-buffer
# deltas; WARNs above GB_ORTHO_LL_TOL (0.05). (Stale note removed 2026-09-02: the [GB_ORTHO]
# premise check is ON below -- GB_ORTHO_CHECK=1 -- and had its first
# GPU validation in the r2 probes: 332/332 measured, max boundary
# overlap 2.95e-04, premise sound.)
export GB_ORTHO_LL_CHECK=1
# Cap-cell grid (user design 2026-08-15): leaf caps on a band/8 grid at
# the confusion scale; scheduling unchanged. RESUME REQUIRES migration
# step 3 in the header checklist. WATCH first propose: leaf growth +
# memory (the band-total throttle is gone -- a band can now reach
# K*cap); GB_CAP_DIVISOR=1 reverts instantly.
# V3: 32, not 8. A 17.36 uHz cell (K=8) is 135 FD bins -- far wider than
# the resolution here -- so 47 cells hold 3-5 separable detectable sources
# against a cap of 2 and 170 of 694 detectable sources (24.5%) are
# structurally unrepresentable. K=32 gives a 4.34 uHz / 34-bin cell and
# drops that to 2.6%, with 456 of 566 occupied cells holding exactly ONE
# source. 32 is also the FLOOR: it is the last divisor whose cell still
# spans this run's own observed duplicate-parking distance of ~1 Doppler
# width (15.5 FD bins at 20 mHz); at K=64 the cell is 1.1 Doppler widths
# there and parked duplicates escape into the neighbouring cell.
# V6: SUB-BAND SHRINKAGE (the run's ONE variable; see the V6 header).
# Uniform layer/8 bands: 135 FD bins each, 1232 bands. The startup
# separation diagnostic logs the conservative-envelope verdict; the
# operative independence monitor is [GB_ORTHO_LL] (default ON) exactly
# as on the production 1-layer grid. GB_SUBBAND_DIVISOR=1 reverts.
export GB_SUBBAND_DIVISOR=8
# Stride 9 = same-unit gap of exactly ONE LAYER, mimicking production's
# stride-2-on-1-layer separation. ~137 concurrent bands/unit (v5: 77);
# 9 units per pass instead of 2.
export GB_BAND_UNIT_STRIDE=9
# PER-WALKER BAND-CLASS ROTATION (aligned with v7 2026-08-29). The stride and
# class membership are UNCHANGED -- band b stays in class b % 9 for every
# walker and band_edges stays one global array, so band b means the same Hz
# everywhere. Only the ORDER changes: each walker gets its own random START
# class and its own +/-1 cycle DIRECTION, then visits the classes in order
# from there. Every walker still covers all 9 classes per sweep (gcd(1,9)=1),
# so it is a rotation, never a permutation -- a walker's concurrently-open
# bands stay one residue class apart and the orthogonality argument is
# untouched (it is a per-walker property: cells of different walkers write to
# disjoint parent rows).
#
# APPLIES IN BOTH SEARCH AND PE (user ruling), so the detailed-balance safety
# is load-bearing rather than a search-stage convenience: both draws are
# UNIFORM and STATE-INDEPENDENT, drawn from model.random so they stay
# seed-reproducible, and run_proposal asserts the per-walker partition every
# propose and refuses to sample if it breaks. NEVER weaken the draw into a
# heuristic ("which walker looks stuck", by logL, by occupancy) -- that
# silently converts a DB-safe change into a DB-breaking one.
#
# ARMED AHEAD OF v7 EVIDENCE (user decision 2026-08-29: "assume it will work
# and it will help or be neutral at worst"). v7 is running these now; if its
# logs show a problem, set both to 0 -- knob-OFF is bit-identical to the
# single global start by construction, not by coincidence. Grep [GB_UNIT_SCAN]
# for the schedule actually used.
export GB_BAND_UNIT_START_PER_WALKER=1
export GB_BAND_UNIT_DIR_PER_WALKER=1
# VGB DE-COUPLED from the fine grid (2026-08-22 timing autopsy): the VGB
# branch inherits GB_SUBBAND_DIVISOR through GBSetup.init_band_structure,
# so v6 silently ran the ~30-source VGB move on 1232 narrow bands --
# vgb_pe 34 s/propose vs v5's 8.5 (run_tempering 25 s; fill_slots 55,920
# vs 4,800). VGB_BAND_LAYERS=8 merges 8 fine bands back to the 1-layer
# separations, restoring v5's VGB geometry (VGB has no RJ surface; its
# per-band arrays migrate on the resume that picks this up). Recovers
# ~4.3 min of the 18.3-min iteration.
export VGB_BAND_LAYERS=8
# STAGING BATCH CAP (2026-08-23, full_pe OOM autopsy): v6 died at it=173
# on a cupy OOM (3.09 GB _expand_B request, gpu0 93.06/93.6 GB) during
# rj_prior_pe's sig-het setup -- this script never carried the cap the
# v5/1-yr scripts got, and full_pe's picked pools (779 leaves/walker,
# 1230/1232 bands occupied) finally outgrew the card.
# MEMORY-FOR-SPEED (user ruling 2026-08-27, snapshot-2 telemetry: GPU
# peaks 45.3/18.2 GB of 99.9 -- 50-68 GB idle in gb_search): batch cap
# raised 1024->2048 (halves the sig-het staging setups, the ~2.3 s
# spikes per sub-block) and BOTH mempool sweeps disabled (stop paying
# free/realloc cycles the headroom does not require).
# *** REVERT AT FULL_PE HANDOFF if memory approaches the card: the v6
# OOM above happened at full_pe occupancy (1230/1232 bands); on any
# resume into full_pe set GB_INMODEL_SETUP_BATCH=1024 and both
# *_MEMPOOL_FREE=1 unless telemetry shows margin. ***
# 4096 (2026-08-27, batch-width autopsy): the in-model REPEAT TRAINS
# run at exactly this width (log: "repeats x 1024/2048 sources" tracks
# the cap), and each step's wall is launch-bound (~dozens of kernel
# launches over ~14 ms of physics at width 2048) -- doubling the width
# halves the number of 100/250-step trains and with them the
# launch-train overhead. Same full_pe revert rule as above.
# ONE-BLOCK STAGING (2026-09-11, ported from the 10w arm). The old
# "SETUP_BATCH=2048, do NOT raise" sizing predates the WINDOWED sig-het
# stash (faadf82: stash bytes go per-source window, not per-band extent)
# and the PSD mirror; the 10w arm runs SETUP_BATCH=0 (whole pool, column-
# atomic per b3f5acbd) with the fold cap below. SAFE ONLY on b3f5acbd+
# code: one-block staging is what raised vertical-pair co-residency, and
# the fixed whole-cell swap ratio is what makes that correct.
export GB_SIGHET_INMODEL_WINDOWED=1
# ######################################################################### #
# ## OOM 2026-09-18 -- THE REVERT ABOVE, APPLIED EARLY.                  ## #
# ##                                                                     ## #
# ## The run died in gb_search (NOT full_pe) at ~4288 leaves:            ## #
# ##   setup_in_model -> gbsignalhetcomputations.py:1074 bin_fold_real   ## #
# ##   -> signal_het.py:119  En = c0[...,:,None,:,:]*iC*c0[...,None,:,:] ## #
# ##   OutOfMemoryError: 2,782,742,528 bytes (89,586,791,936 allocated)  ## #
# ##                                                                     ## #
# ## gbsignalhetcomputations.py:1047 sizes each fold chunk to FILL the   ## #
# ## byte cap, so an 8 GiB cap builds an ~8 GiB transient by design:     ## #
# ##   per_src = 2*nch*nch*W*Nt_active*16 (Ec+En) + nch*W*Nt_active*16   ## #
# ##   chunk   = _SIGHET_FOLD_MAX_BYTES // per_src                       ## #
# ## The CODE default is 1 GiB (1<<30); 8 GiB was this file's override,  ## #
# ## and one-block staging (SETUP_BATCH=0) sat on top of it. That is     ## #
# ## ~8 GiB of transient against a card already 89.6 GB resident.        ## #
# ##                                                                     ## #
# ## The block above says to revert AT THE FULL_PE HANDOFF. The 6-month  ## #
# ## run reaches the danger zone EARLIER than that -- in gb_search, as   ## #
# ## the leaf count climbs -- so the revert is now the default here.     ## #
# ## This file's own header (lines ~15-17) already calls the calibrated  ## #
# ## 6-month sizing GB_N_SUBBANDS=4096/GPU and SETUP_BATCH=2048; the     ## #
# ## live values had drifted to 8192 and 0.                              ## #
# ##                                                                     ## #
# ## All four are TRANSIENT/scheduling knobs: no stored number changes,  ## #
# ## so a mid-store resume is safe. Cost is more chunks per unit (launch ## #
# ## overhead), not accuracy. Every one is env-overridable, so the old   ## #
# ## aggressive sizing is one export away if telemetry shows margin.     ## #
# ##                                                                     ## #
# ## STILL TIGHT? Next lever is GB_N_SUBBANDS 8192 -> 4096 (halves the   ## #
# ## slab AND the active-slot count; the header's calibrated value).     ## #
# ## NEVER reach for SIGHET_NT_LAYER: not a mid-store knob, and it       ## #
# ## MULTIPLIES with GB_N_SUBBANDS in the sig-het byte product.          ## #
# ######################################################################### #
export GB_INMODEL_SETUP_BATCH=${GB_INMODEL_SETUP_BATCH:-2048}
export GB_SIGHET_FOLD_MAX_BYTES=${GB_SIGHET_FOLD_MAX_BYTES:-1073741824}
export GB_RJ_INMODEL_CHUNK=65536  # 3 mo: the 3-month value (6 mo halves it to 32768 for byte parity -- its cells are ~2x the bytes); floored to ntemps multiples by the column-atomic staging
export GB_INFOMAT_MEMPOOL_FREE=${GB_INFOMAT_MEMPOOL_FREE:-1}
export GB_INMODEL_BATCH_MEMPOOL_FREE=${GB_INMODEL_BATCH_MEMPOOL_FREE:-1}
# ######################################################################### #
# ## SEAM-STRADDLING CAP CELLS (divisor 2 + stagger, 2026-08-29).        ## #
# ## ⚠ DO NOT "FIX" THE CAP GRID BACK INTO ALIGNMENT WITH THE SUB-BANDS. ## #
# ######################################################################### #
# This REPLACES the 2026-08-26 aligned-cells ruling (divisor 1, stagger 0)
# on direct measurement. Read this before touching GB_CAP_DIVISOR or
# GB_CAP_STAGGER -- alignment looks tidier and is exactly wrong.
#
# THE FINDING. The 3-month v7 run has a persistent bimodality at the
# flagship 20.380377 mHz: leaves split across the band 1141/1142 seam
# (+12.19 bins from the flagship) with an EMPTY GAP between the two
# populations. The dedicated high-f probe SOLVED this on the same band,
# and the probe's BAND grid is bit-identical to v7's -- so the band grid
# is not the discriminator. THE CAP GRID IS. Cap-cell membership of the
# actual flagship leaves at v7 row 5, under both geometries:
#
#   v7    (divisor=1, stagger=0):
#      cell 1141  [-122.8, +12.2] bins   below-seam 24   above-seam  0
#      cell 1142  [ +12.2, +147.2] bins  below-seam  0   above-seam 22
#
#   probe (divisor=2, stagger=1):
#      cell 2284  [ -21.6,  +45.9] bins  below-seam 24   above-seam 22
#                                        ^^^ SPANS THE SEAM
#
# WHY IT MATTERS. Under the aligned grid each band carries an INDEPENDENT
# cap, so the two modes never compete and eight leaves can sit across the
# seam forever. Under the staggered grid BOTH modes fall in ONE cell and
# compete for ONE cap, which gives the RJ death move direct pressure to
# kill the weaker side. This compounds with the block-Gibbs scan: bands
# 1141/1142 are residue classes 7 and 8 mod GB_BAND_UNIT_STRIDE=9 and are
# therefore NEVER co-open, so no single move can even see both modes at
# once. The straddling cell is what reaches across that.
#
# make_cap_edges' own docstring states the intent: stagger exists so "no
# source can sit on a band seam and a cap seam simultaneously", and the
# cell at index b*K "physically STRADDLES the band-(b-1)/b boundary".
#
# GEOMETRY AT K=2. Sub-bands are 135 FD bins (GB_SUBBAND_DIVISOR=8), so
# cells are 67.5 bins. Staggered edges land at the 1/4 and 3/4 points of
# each band, so every band owns one INTERIOR centre cell and SHARES two
# straddling cells with its neighbours. Every cap-cell boundary is
# mid-band; NO cap edge sits on a band seam. 2464 cells, not 1232.
#
# PINNED BY tests/test_cap_stagger.py::SeamStraddlingCellTest, which
# asserts that at (2, 1) two leaves either side of a band seam map to the
# SAME cell index and at (1, 0) they do not. If you change this, that
# test fails and tells you why.
#
# ⚠ CAP EDGES CHANGE -> the resume guard
# (GBState.initialize_band_information) refuses a v7/v8-lineage store. See
# the relaunch block at the top of this file: rewind GB to empty, then
# migrate_gb_cap_grid.py --cap-divisor 2 --stagger.
export GB_CAP_DIVISOR=1
# 2 -> 1 (user design 2026-08-29). WITH GB_CAP_STAGGER=1 this is the
# MIDPOINT-TO-MIDPOINT grid: 1232 cap cells for 1232 sub-bands, each
# interior cell running from the midpoint of one sub-band to the midpoint
# of the next, so the seam sits at the CENTRE of a cell and the two sides
# of a seam compete for ONE cap. Verified against the real 1232-band grid:
# interior cap edges == band midpoints exactly; widths half / 1.7361e-5 Hz
# (= one full sub-band) / 1.5x at the two ends.
#
# WHY NOT 2 (what the 2026-08-29 restart ran): divisor 2 makes cells HALF
# a sub-band, which (a) halves the straddle reach to +/-0.25 sub-band, so
# a seam-straddling pair further apart falls out of the shared cell,
# (b) doubles the cells, halving per-cell occupancy and delaying at-cap
# birth-row exclusion -- measured +39% F-stat candidate rows at matched
# iteration -- and (c) gives each band TWO owned cells, which is what let
# a birth into an already-full straddling cell slip past the old
# band-saturation gate (4 of 24 walkers held 2 leaves in a cap-1 cell at
# rows 5-6 while no sub-band ever held more than one).
#
# REBUILD: not required for THIS config (the CUDA cell lookup is reached
# only via the fused in-model path, and GB_INMODEL_ACCEPT_KERNEL=0 here),
# but rebuild on relaunch anyway so python and kernel stay in step -- the
# disagreement would be silent if anyone ever set that knob to 1.
# ############################################################################
# ## V7 (2026-08-24) = V6 + OVERLAPPING CAP CELLS, nothing else.            ##
# ## GB_CAP_OVERLAP_FRAC=0.25 widens each cap cell's enforcement span on   ##
# ## the SAME staggered edge grid (width 45 bins = 11.25 shared | 22.5     ##
# ## core | 11.25 shared) so any two leaves within 11.25 bins share a      ##
# ## covering cap-1 cell -- the anti-split-source experiment, now at        ##
# ## production scale (motivation: the flagship's persistent -5.7-bin      ##
# ## cap-edge mode; see the confined highf probes). Edges unchanged ->     ##
# ## resume guard passes on a rewound v6-lineage store.                    ##
# ## ALSO ON (user, 2026-08-24): rj_replace (GB_SEARCH_RJ_REPLACE=1,     ##
# ##   the code default made explicit) -- exact-MH F-stat replacement      ##
# ##   after rj_fstat_search in the search cycle. NOTE: the confined       ##
# ##   probe measured ~0 cold acceptance for this move (forensics in      ##
# ##   flight); at worst it costs wall time, it cannot bias (exact MH).   ##
# ##   Resume-safe: the store guard asserts stage NAMES/order only.       ##
# ## EXCLUDED for now: the core-dominant divisor-2 geometry (probe-only). ##
# ## START (v6 pattern): cp -r gf_prod_3mo_v6 gf_prod_3mo_v7 &&           ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \              ##
# ##     gf_prod_3mo_v7/gf_prod_3mo_testing.h5 gb_search \                ##
# ##     --rewind-to-empty gb --apply                                      ##
# ## (inherits fitted noise + VGB ladder + fstat epoch cache; only the GB  ##
# ## search reruns under overlap enforcement). Fresh dir works too.        ##
# ############################################################################
# CAP CELLS EXACTLY == SUB-BANDS (user ruling, aligned with v7 2026-08-29):
# 0.25 -> 0. With GB_CAP_DIVISOR=1 and GB_CAP_STAGGER=0 the cap edges are
# already bit-identical to the band edges; dropping the overlap removes the
# widened ENFORCEMENT SPAN so a cap cell is exactly its sub-band, with no lip.
#
# WHY: the overlap makes "at cap" an OR over covering cells, so a leaf near a
# band edge is charged against BOTH neighbours' budgets. Harmless when the cap
# is slack -- but measured on the v7 store the cap BINDS: at the flagship bands
# 1141/1142, max-over-walkers occupancy reached the cap in 55 of 104
# (band, iteration) pairs = 53%. With a binding cap the overlap obstructs the
# cross-edge in-model movement the headroom below is meant to allow: a leaf
# could be vetoed because EITHER side was full, not because its destination
# was. Removing the lip makes each leaf count only where it actually is.
#
# Resume-safe, no migration: GBState.static_names is only
# ("band_edges", "cap_edges"), and make_cap_edge_extensions (state.py:107) is
# computed at runtime rather than persisted, so this touches no stored array.
#
# ⚠ STAYS 0 UNDER THE STAGGERED GRID (user ruling 2026-08-29). The probe
# whose geometry we are adopting ran overlap=0.25 as WELL, so this is
# deliberately NOT an exact probe reproduction. Reasoning:
#
#  * Stagger and overlap are THE SAME MECHANISM from opposite sides: both
#    charge a leaf to a cell it does not sit in, creating competition for
#    one cap. Whether that is good depends on WHERE.
#  * At a BAND SEAM competition is what you want, because non-competition
#    there compounds with the block-Gibbs trap (1141/1142 are residues 7
#    and 8 mod 9, never co-open, so no single move sees both modes).
#    STAGGER FIXES EXACTLY THAT CASE, and it is the case with direct
#    evidence -- see the divisor block above.
#  * At a CAP-CELL BOUNDARY MID-BAND competition is worth much less: those
#    two leaves ARE in the same band, so they are co-open, RJ can propose
#    to both and in-model moves freely. Non-competition there is an
#    inefficiency, not a trap. That is the ONLY extra case overlap covers
#    once staggered -- and it buys it by tightening effective capacity
#    everywhere, against a cap measured as BINDING in 55 of 104
#    (band, iteration) pairs = 53% at the flagship bands.
#  * ATTRIBUTION: one change at a time. Stagger+overlap together and we
#    cannot tell which mattered.
#  * COMPOUNDING: at divisor 2, stagger AND overlap gives 67.5-bin cells
#    widened to 90 with 22.5-bin shared zones -- every leaf near any
#    boundary counted twice, layered on an already-binding cap.
#
# IF STAGGER ALONE DOES NOT CLEAR THE BIMODALITY, restoring
# GB_CAP_OVERLAP_FRAC=0.25 is the obvious next step, and we will then know
# which of the two was load-bearing.
export GB_CAP_OVERLAP_FRAC=0
# NO-OP AT GB_CAP_DIVISOR=2 -- RETAINED DELIBERATELY (verified 2026-08-29).
# The drift gate's short-circuit is
#   (cap_divisor == 1 AND overlap <= 0 AND NOT edge_leak) -> return None
# so at divisor 2 the FIRST clause is already False and the gate is armed
# regardless of this knob. Grep confirms it: `cap_drift_gate_edge_leak` has
# exactly two references in the tree -- its ctor read and that one clause --
# so it can have no other effect here.
#
# KEPT SET ANYWAY because it is only inert while divisor > 1: if anyone
# reverts to GB_CAP_DIVISOR=1, this is what stops cap enforcement on
# cross-edge in-model moves from silently vanishing (the 2026-08-20
# "29 leaves into a cap-1 cell" mode, at the seams).
export GB_CAP_DRIFT_GATE_EDGE_LEAK=0   # V9-4: cap cells off
# N/4 IN-MODEL BAND WINDOW ACTUALLY MEANS N/4 (bug fix, aligned with v7).
# The window was BUILT as band_N_vals * layer_df / 4 Hz (gbbands.py:3397-3401)
# while every consumer divides by the move's df = 1/Tobs -- so it was too wide
# by layer_df*Tobs = Nt/2 = 1080x. Measured: N=256 intended +/-64 bins, actual
# +/-69,120 bins = +/-512 sub-bands, WIDER THAN THE WHOLE 3-21 mHz BAND. The
# per-step leash (|df0| <= N/4 bins) was always unit-correct and is untouched.
# =0 restores the old unbounded window.
export GB_BAND_WINDOW_STRICT=1
# CAP GATE READS THE DESTINATION CELL FROM THE CANDIDATE f0 (bug fix, v7-aligned).
# At cap_divisor == 1 _cap_cell_index returned band_inds and never read f0, so
# current cell == new cell for every row and THE VETO COULD NOT FIRE -- the
# cap+2 destination rule was a tautology. =0 is the escape hatch.
#
# STILL CORRECT AT GB_CAP_DIVISOR=2 + STAGGER=1 (verified 2026-08-29), and the
# earlier "the clip folds an out-of-band proposal back into the leaf's own
# boundary cell" reading DOES NOT APPLY: that clip lives in the NESTED branch
# only. The staggered branch clips solely to the GLOBAL cell range, so sub may
# run past K-1 (or below 0) and the leaf lands in the neighbouring STRADDLING
# cell -- exactly what the geometry wants. Verified against
# searchsorted(cap_edges, f) over +/- 0.9 band widths.
#
# On our UNIFORM band grid the b*K terms cancel algebraically, so the staggered
# lookup gives the same cell for ANY handed-in band index and this knob is a
# no-op -- EXCEPT exactly on a cap edge, where the cancellation is exact in
# real arithmetic but not in floating point and the two band references can
# round the tie to different sides (measured: 2 of 37 sample points, both on
# edges; resolve_band matched searchsorted at both). Keep it ON: one
# searchsorted, removes the tie ambiguity, and load-bearing on a ragged get_n
# grid where the per-band steps differ.
export GB_CAP_DEST_BAND=1
# rj_replace DISABLED (aligned with v7 2026-08-29). Not earning its ~580 s/row
# (cold acceptance 0.033-0.046%, delta-ll flat at ~103 mean across 55 calls),
# and its lnL ACCOUNTING IS BROKEN: 365 of 905 [GB_ORTHO_LL rj_replace] lines
# (40.3%) breach GB_ORTHO_LL_TOL=0.05, max 6.971e+03 -- four orders worse than
# any other move -- with a drift ledger claiming +9,693 against -117,009
# realized. Chain state stays correct (drift is repaired from the residual) but
# the per-cell lnL the MH ratio prices against can be wrong by thousands of
# nats. Re-enable only after the accounting is fixed and re-audited.
# V9-11 RE-ENABLED (user ruling 2026-09-24, reversing the 08-29 disable):
# "clean up the replace move to modernize ... and add it to each GB search
# 1-3 before the prior move." The accounting complaint above was raised
# against the PRE-2026-08-24 move; that redesign replaced the unattainable
# phase-max credit with rotation-on-accept (the scored rows ARE the final
# rows) and made both sides exact add-deltas. The user's position is that the
# original problems may have been the old GPU setup.
#
# ⚠ IT IS BEING WATCHED, NOT TRUSTED. Per pass the move now prints
#   [GB_REPLACE rj_replace] pass i/N: candidate source = WARM|FSTAT container.
#   [GB_REPLACE rj_replace] pass i/N (src) done: max |dlnL| ... = X
# and WARNS above 1.0e3 -- the recorded retirement signature was a
# propose-level drift of 1.5-1.9e3, three orders above every other move.
# Also watch [GB_ACCEPT replace-split] (per-pass acceptance, gated counts,
# cold-accepted dll mean/max) and [GB_ORTHO_LL rj_replace] against
# GB_ORTHO_LL_TOL. If either fires, set GB_SEARCH_RJ_REPLACE=0 and relaunch;
# nothing else in the cycle depends on it.
# >>> 1 -> 0 (user ruling 2026-09-25, off job 628). NOT because the
# >>> watchdogs above fired -- max |dlnL| stayed 298-661, under the 1.0e3
# >>> warn and well under the 1.5-1.9e3 retirement signature. It is being
# >>> switched off for YIELD: six consecutive zero-yield proposes, 8 swaps
# >>> accepted out of ~707,000 proposals across 3 iterations x 2 passes,
# >>> and ZERO at the cold chain -- dead at every temperature, not just
# >>> cold. It was simultaneously the most expensive move in iteration 3
# >>> (1610 s = 19.4% of the cycle), ~98% of which is in-model polish on
# >>> its survivors that in_model_replace then provides anyway.
# >>> in_model_replace STAYS: it is a plain in-model pass and does not
# >>> depend on the RJ move. Reinstate for gb_search_3, where sources are
# >>> mis-seated rather than missing and a replacement has something to
# >>> beat.
export GB_SEARCH_RJ_REPLACE=0
# in_model_replace retires WITH it (2026-09-26, carried from
# submit_gf_6mo_v9_4gpu.sh 0e0067de). It was the polish slot for
# rj_replace's survivors; with that move off it is simply a THIRD pure
# in-model pass over rows in_model and in_model_fstat already polished.
# The driver default is already 0 -- stated here so the run script says
# what it runs. =1 restores it, alongside rj_replace, not on its own.
export GB_SEARCH_IN_MODEL_REPLACE=0
# TWO INTERNAL PASSES per propose() (user ruling 2026-09-24): pass 1 draws
# replacement candidates from the warm-start mixture (the SAME container
# rj_warm_search births from), pass 2 from the F-stat grid; each pass prices
# both sides of its MH ratio against its own container, so the two never mix.
# =0 leaves replace single-pass (F-stat only), the pre-2026-09-24 behaviour.
#
# 1 -> 0 AT 3 MONTHS. There is no warm-start mixture for pass 1 to draw from
# (GB_WARM_START_COMPONENTS is empty -- 3MO-2), so the pass would be a knob
# that resolves and reaches nothing. Moot in practice, because
# GB_SEARCH_RJ_REPLACE=0 above keeps rj_replace out of the cycle entirely --
# but if replace is ever reinstated for gb_search_3 here it must run F-stat
# only, and this is the line that makes that true.
export GB_REPLACE_WARM_PASS=0
# PE replace OFF too (user ruling 2026-09-02). It was default-ON and would
# have been the only replace flavor left; in both r2 probes it registered
# but never exercised a single proposal, and with centering off its
# center-table inputs are gone -- installed-but-inert machinery in a
# production run is surface area with no upside. =1 restores it.
export GB_PE_RJ_REPLACE=0
# ORTHOGONALITY PREMISE MONITOR (v7-aligned). NOT GB_ORTHO_LL_CHECK (the lnL
# bookkeeping reconcile, already on). This measures what the band decomposition
# RESTS on: normalized |<h_i|h_j>| between concurrently-open adjacent-band cold
# sources. 8 pairs per unit at unit close, diagnostic only, never mutates state.
# DISARMED (2026-09-04, ported from the 3mo 10-walker arm): this premise
# check DIES on the 2-GPU path (get_swap_ll shards index a numpy
# data_index with a cupy keep_idx -> guarded TypeError skip, ~166/snapshot
# of pure log spam, never runs). The companion GB_ORTHO_LL_CHECK
# (credited-vs-direct) still runs and stays on. Re-arm =1 only after the
# gb_likelihood.py:907 cupy/numpy fix lands.
export GB_ORTHO_CHECK=0
# STAGGER ON (2026-08-29) -- the half-cell shift is the whole point of
# this configuration; see the GB_CAP_DIVISOR block above for the measured
# cell-membership numbers. It is meaningless without divisor > 1 (the move
# forces it off at K == 1), and the pair (2, 1) is what the high-f probe
# ran when it solved the flagship bimodality.
export GB_CAP_STAGGER=1
# Per-cell cap CEILING + entry headroom + stage hold (probe-validated
# 2026-08-26 set):
#   GB_CAP_CELL_MAX=20    -- belt on the cap updater, sized to the
#                            v6-REALIZED envelope (v6 ran NO ceiling --
#                            the knob postdates it -- and its log shows
#                            34-bin cells reaching cap 5; 4 such cells
#                            per 135-bin sub-band -> up to ~20). The
#                            probes pinned 5, but that was a
#                            SINGLE-SOURCE band; a 135-bin confusion
#                            cell legitimately holds 3-5+ separable
#                            detectable sources (v3 analysis above) and
#                            must be able to ramp past 5. NOTE the
#                            regime tension: the ratchet is
#                            STALL-driven (patience without D/2
#                            improvement WHILE occupied at cap
#                            increments), so loud isolated cells lean
#                            on the drift gate / entry veto / removal /
#                            SNR-8 floor rather than this ceiling; if
#                            flagship-style stacking reappears, a
#                            frequency-dependent ceiling is the next
#                            lever, not a global squeeze.
#   GB_CAP_INMODEL_HEADROOM=2 -- in-model/replace f0 moves may enter a
#                            foreign at-cap cell up to cap+2 (peak
#                            handover across an edge); births still
#                            respect the hard cap.
#   GB_SEARCH_CAP_QUIESCENT=1 -- the nleaves plateau cannot end
#                            gb_search while any engaged cap cell is
#                            still mid-ramp (occupied at cap, below
#                            ceiling): the stage holds for the ramp.
# ---- BOTH RE-EXAMINED AT (divisor=2, stagger=1), 2026-08-29 ----
# GB_CAP_CELL_MAX=20 KEPT. Cells now govern HALF a sub-band (67.5 bins,
#   2464 cells), so 20 is loose -- but it is a BELT against the ratchet,
#   not a design target, and it is what the probe effectively ran: the
#   probe set no GB_CAP_CELL_MAX at all, so its ceiling fell back to
#   nleaves_max, which the probe pinned at 20. Same effective per-cell
#   ceiling, reached two different ways. Lowering it now would be a second
#   simultaneous change with no evidence behind it.
# GB_CAP_INMODEL_HEADROOM=2 KEPT, but note its SCOPE SHRANK. The veto only
#   fires on a cell the row is NEWLY entering (_cap_new_entry_veto:
#   `_foreign = _memb & (_cell != c_p) & ...`). With both flagship modes
#   now inside ONE cell, a cross-seam in-model move has new cell == cur
#   cell, is not foreign, and is NOT POLICED AT ALL -- correctly, since the
#   occupancy census does not change when a leaf moves within a cell. What
#   the headroom still governs are the two MID-BAND cap boundaries per band
#   (the 1/4 and 3/4 points), where cap + 2 keeps peak handover possible
#   against a cap measured as binding 53% of the time at the flagship
#   bands. 2 remains right: it is an ADDITIVE allowance on destination
#   occupancy, and the transient double-occupancy it must exceed did not
#   change when the cell halved.
# RAISED 20 -> 50 (user ruling 2026-09-19): "we cannot be up against that
# edge". This is a BELT against a runaway ratchet, not a design target, and
# 20 was only ever inherited from the probe's nleaves_max. Truth-set source
# density per cap cell, scaling SNR by sqrt(Tobs) off the 3-month set,
# detection at SNR 7:
#     Tobs    p99   p99.9   max   cells >=20   cells >=50
#     3 mo      6       8     8        0            0
#     6 mo     11      12    14        0            0
#     1 yr     17      20    21        4            0
#     2 yr     28      34    35       92            0
# So 20 already BINDS at 1 year (four cells exceed it) and badly at 2, while
# 50 is clear of the densest cell at every Tobs through 2 years. It costs
# nothing to raise: the knob gates the increment condition and sizes no
# allocation, and on the live 6mo run the highest cap reached is 17 with
# nothing at the ceiling, so this changes no cell's behaviour today.
export GB_CAP_CELL_MAX=50
#
# ── SUPERSEDED 2026-08-30. The block above reasons about the divisor-2
# geometry (two mid-band cap boundaries per band, "when the cell halved").
# Under GB_CAP_DIVISOR=1 + STAGGER there is ONE cap boundary per band, at
# its midpoint, and the straddling cell is centred on the SEAM.
#
# 2 -> 0: ENFORCE CAP-CELL OCCUPANCY FOR IN-MODEL (user ruling). Sub-band
# boundaries stay PERMEABLE -- the N/4 bin leak in _run_in_model_repeats
# (``new_bin < lo_s - n4_s``) is untouched, and GB_BAND_WINDOW_STRICT
# governs the RJ window, not this. A source may still walk across a
# sub-band edge; what it may no longer do is walk into a cap cell that is
# already at capacity.
#
# MEASURED, first hour of the 2026-08-30 run (row 2, the first GB propose):
# 555 cap cells held two leaves, one either side of a sub-band seam. Those
# doubles CANNOT have been born:
#   - the cap armed 08:28:41, before the grid fit and all births;
#   - only bands c-1 and c can reach cell c, and 555/555 doubles had
#     |delta band| = 1 with 0/555 sharing a residue mod GB_BAND_UNIT_STRIDE
#     -- so the two bands NEVER co-open and two births into one cell inside
#     a single unit is geometrically impossible;
#   - a later unit's birth is blocked: the at-cap mask is rebuilt at every
#     unit open and applied, exclusions growing 213,239 -> 1,462,467.
# By elimination the second leaf WANDERED IN, crossing the cap edge at the
# sub-band midpoint into a cell already at cap -- which headroom 2
# explicitly permits. Occupancy topped out at exactly 2, never 3: a gate
# holding at cap + headroom, not a gate that was missing.
#
# So the birth gate and the staggered grid are working, and this one knob
# manufactured every seam double. 1 does NOT help (cap + 1 = 2 is the
# observed state); only 0 closes it.
#
# ACCEPTED RISK: at 0 a source cannot enter an occupied cell at all, so a
# seam pair resolves by one leaf DYING rather than by the two merging.
# Deaths stay proposable in over-cap cells and a leaf still moves freely
# within its own cell, which is the intended mechanism: the cap forces one
# source per straddling cell, and tempering handles the mid-band pairs.
export GB_CAP_INMODEL_HEADROOM=0
# ===========================================================================
# V9-4: CAP CELLS OFF (user ruling 2026-09-24 -- "So no caps. Keep all those
# options available but turn them off.").
# ===========================================================================
# The progressive per-band/per-cap-cell leaf cap is ARMED ONLY by
# GB_LEAF_CAP_START (recipe.py: `int(os.environ["GB_LEAF_CAP_START"]) if
# os.environ.get("GB_LEAF_CAP_START") else None`), so an EMPTY value is the
# off switch and every knob below stays present and documented, inert,
# ready to be re-armed by setting a start value again.
#
# WHY OFF. The cap existed to stop a band swallowing leaves before the
# search could judge them, and it brought its own failure modes with it --
# the ghost-increment ratchet, the cap-quiescence and cap-headroom vetoes
# that had to be bolted onto stage convergence to stop a cap blockade
# reading as convergence, the half-band cap_edges shift, the at-cap birth
# skip. The v9 structure attacks the same problem from the other end: run
# in-model to convergence per sub-band and let the per-(walker, band)
# shutoff decide when a band is done. The bet is that this makes the cap
# unnecessary; if it is wrong, the symptom is bands running away in leaf
# count and the fix is to set GB_LEAF_CAP_START again.
export GB_LEAF_CAP_START=
# The cap's companion knobs (drift gate, edge leak, the two
# stage-convergence vetoes) are turned off AT THEIR OWN EXPORT LINES below,
# not here -- bash resolves exports in order, so a duplicate here would be
# silently overridden by the later one. Search V9-4 to find them all.
# ===========================================================================
export GB_SEARCH_CAP_QUIESCENT=0   # V9-4: cap off -> nothing to hold for

# ===========================================================================
# V9-9: PER-(WALKER, BAND) RJ SHUTOFF VALVE -- the STAGE convergence signal.
# ===========================================================================
# A (walker, band) pair freezes when its sub-band cold-chain logL converges
# within the current recipe step, and the valve RELEASES when the next step
# begins. A search stage ends when EVERY OCCUPIED pair has shut off (user's
# words) -- empty pairs can never shut off by construction, so "all pairs"
# is unreachable and the occupied-only reading is the operative one.
export GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1
# 5 -> 3 (user ruling 2026-09-25): "let's make sure it's not too many
# iterations. 2 or 3 is fine." This is how many consecutive iterations an
# occupied (walker, band) pair must post a converged sub-band logL before
# it shuts off; the STAGE ends when every occupied pair has. A shorter
# window ends each search stage sooner, which is what we want now that the
# per-leaf cost is climbing (7.4 -> 70.6 -> 101.9 s/leaf over 628's three
# iterations) -- time is better spent in the next stage than re-proving a
# band that already stopped paying.
export GB_SEARCH_BAND_SHUTOFF_CONV_ITER=3
# LEVEL-3 VALVE TOLERANCE. The bar a (walker, band) must clear to keep
# its valve OPEN. Default without this export is 0.5 * leaf_cap_ndim =
# 4.0, which is the LEVEL-1/2 per-row dll -- far too fine for a
# whole-band plateau test, and why 154 pairs sat chronic.
#
# 20, set DIRECTLY by Mike on 2026-09-28. ⚠ This is a JUDGEMENT, not a
# formula: the earlier 32 came from "one floor-SNR source",
# 0.5 * SNR^2 = 0.5 * 8^2, and 20 is not that expression for any floor
# in play (the rule would give 32 at floor 8 and 12.5 at floor 5 in
# gb_search_2/3). Do not "restore" it to a derived value on the
# assumption the rule still holds -- it was superseded.
#
# A CONSTANT rather than derived from the stage profile, deliberately:
# three different SNR attributes are in scope here
# (opt_snr_rej_samp_limit, opt_snr_limit_search_coarse, _fine) and
# picking the wrong one would silently mis-set a gate threshold.
# Change it by hand per stage.
# Read from env AT JUDGE TIME and NOT part of the band_shutoff_w_step
# stamp, so the persisted streak/max survive this change.
export GB_SEARCH_BAND_SHUTOFF_LL_TOL=20
# A PURE IN-MODEL PASS AFTER rj_prior_removal (2026-09-28, Mike).
# rj_prior_removal is the ONLY RJ move with no in-model partner, so the
# SAVED STATE -- and the residual the next iteration's births are drawn
# against -- was the post-death state with no group-rule pass over it.
# The surviving twin of a killed double is exactly the cell that needs
# readjusting after the death.
# ⚠ NOT A DOUBLES CURE: deaths are still JUDGED on the unadjusted
# residual. This is correctness of the saved state and better births.
# Same move class as in_model / in_model_fstat, so it inherits the
# level-2 group rule, the level-3 shut mask and the census. Cost is the
# same class too: ~1400 s at a fully open population, under 200 s once
# most pairs are shut. GB_SEARCH_DROP_MOVES can name it.
export GB_SEARCH_INMODEL_AFTER_REMOVAL=1
# GB_SEARCH_DROP_MOVES EXISTS BUT IS DELIBERATELY UNSET.
# It omits named pure in-model slots from every search stage. Dropping
# in_model_fstat was proposed on 2026-09-28 on cost grounds (it and
# in_model are the same class -- one polish per RJ move -- and cost
# 1392 s + 1248 s of a 5564 s job-662 iteration, 47%) and MIKE
# REJECTED IT: in_model_fstat stays, so the stage compositions are
# unchanged. Unset is a documented no-op; the knob and its guard are
# kept because the guard is worth having -- rj_fstat_search can never
# be dropped (designated updater, and the only move that runs the
# level-3 judge and the stage-end check), and the knob refuses it with
# a SystemExit rather than leaving a stage unable to end.
# ⚠ OFF, deliberately: this is the PER-BAND stage schedule (SNR limits and
# phase maximization moving independently per band). In v9 those move
# together across RECIPE stages instead, so the per-band schedule must stay
# out of the way. The two are independent flags.
export GB_SEARCH_STAGE_PER_WALKER=0
# RJRecipeStep plateau window (user ruling 2026-09-19). The code default is
# 5, and at FOUR walkers that is too tight. The rule is a running-maximum
# ratchet (recipe.py::RJRecipeStep._stop_fn): max of the cold leaf count
# over the stage so far, against its max over the last `convergence_iter`
# iterations -- so ONE noisy dip ends the search stage. Both 4-walker
# production runs died that way while gb_search was still gaining leaves:
#
#   3 mo: ...771->778, 772->782, 778->782 x4, 782->777 STOP
#   6 mo: ...1674->1683, 1679->1683 x3,      1683->1681 STOP  <-- THIS RUN
#
# The 6 mo stopped on a TWO-leaf dip while its own walker-to-walker spread
# was 51 ([1638 1689 1656 1657]), after +110 leaves over the preceding 30
# checks, with GB_SEARCH_CAP_QUIESCENT armed but firing zero times (caps
# were quiescent, so nothing held the stage open).
#
# 20 gives the ratchet a window several times the observed per-check growth
# (0.5-3.7 leaves) without disarming the test: a genuinely converged stage
# still trips it, just later. Note the stage-scoped guard means the first
# check cannot fire until 2*20 = 41 iterations into the stage, which is
# exactly what a post-rewind resume wants.
export GB_PLATEAU_ITERS=20
# ---- THE TWO v4-POSTMORTEM FIXES (code defaults since 8d926f27; pinned
#      so the store's provenance is unambiguous) ----
# Birth fix (1274a66c): births draw fdot_astro_ratio | (f0, Mc) from the
# tight mixture around the F-stat grid fdot instead of U[-5, 5].
export GB_FSTAT_BIRTH_RATIO_TIGHT=1
# Ridge-Gibbs (8d926f27 / Eryn 6ed5a8b): zero-likelihood-call resample
# along the exact Mc^(5/3)(1+r)=const ridge; unfreezes (Mc, r, dist).
export GB_RIDGE_GIBBS=1
# CAP DRIFT GATE (2026-08-20, root-caused on the high-f probe: births
# respect the per-cell cap but in-model repeats walked 29 leaves into a
# cap-1 cell -- the 2026-08-15 TODO made real). In-model proposals whose
# f0 lands in a FOREIGN at-cap cell are vetoed; within-cell moves and
# drains of over-full cells stay allowed. This is the same mechanism
# that let production mosaics stack leaves past their cell caps. Watch
# [GB_CAPGATE] veto counts; GB_CAP_DRIFT_GATE=0 reverts.
export GB_CAP_DRIFT_GATE=0             # V9-4: cap cells off
# Leaf-cap PATIENCE: consecutive iterations without a sufficient (D/2)
# lnL improvement before a cap CELL advances. Code default is now 3
# (2026-08-16, was 5): caps live on the band/8 cap-cell grid, so 1,232
# cells must each climb from 1, and at ~6 min/iteration the ramp -- not
# the wall -- is what limits how fast the model can fill. Pinned here for
# the run record; raise it if caps ever outrun the likelihood evidence.
# GHOST-INCREMENT GUARD -- ON in v3, and ONLY meaningful with K=32.
# An EMPTY cap cell can never improve its max ll by D/2, so under the bare
# counter it accrued patience on a fixed clock and promoted itself
# alongside cells doing real work. In v2 that ran caps from 1 to a median
# of 14 by iteration 60: the model held 591 sources against a permitted
# 15,619 (3.8% of the allowance) with 0.3% of occupied cells at cap -- the
# cap tightest when the model was empty and needed no protection, absent
# once it was fullest. The guard starts a cell's patience clock only after
# it has improved at least once, mirroring `changed_once` in the PSD
# max-logL search. It must NOT ship at K=8: freezing empty cells at cap 1
# with a 135-bin cell re-imposes the 24.5% exclusion above.
export GB_LEAF_CAP_REQUIRE_IMPROVEMENT=1
# 4 (user ruling 2026-08-28, was 5): the v7 cap-gap analysis showed real
# 2->3 increments arrive at median gap 9 with ZERO at the floor of 5 --
# qualification-bound, but E[step] scales with the threshold, so 5->4 buys
# ramp depth with a wide margin before the floor binds. 3 remains the
# aggressive option if the ramp still lags. (Historical: 5-not-3 was
# probe-validated 2026-08-26 with the one-shot engagement latch +
# occupied-only patience, c251b267.)
# ⚠ NOTE for the v8-vs-v7 comparison: this is the ONE knob that makes the
# v8 GB configuration differ from v7's. The cluster checklist's gate 4
# assumed "same GB config by construction, the noise marginals are the new
# content" -- with this change the GB side moved too, so a v8-vs-v7 cap or
# leaf-count difference is NOT purely a noise effect. Set back to 5 if you
# want the noise comparison fully isolated.
# 4 -> 3 (user ruling 2026-09-16): the v4/v5 3-mo campaigns ran 3; 4 was
# the noise-merge-era compromise kept for v7 isolation, which no longer
# applies. At 6-mo wall (~33 min/it) each increment cost ~2.2 h at 4 —
# measured in production: caps raised on exactly the 4th consecutive
# at-cap iteration (rows 13->16/17). Plateau/occupancy gates, staggered
# cap grid, at-cap RJ skip, D/2 and GB_CAP_LL_CHECK all still stand.
# 2 was considered and deferred: a 2-it stagnation window is weak at 10
# walkers — revisit with the next snapshot's cap trajectory if 3 lags.
# STAYS AT 3 (user ruling 2026-09-18, after considering 8 and 5).
# RECORDED COUNTER-EVIDENCE, so the next person does not re-derive it: the
# gate is MAX over cold walkers -- a band's cap holds while the BEST walker
# keeps improving it by D/2, and increments once none has for this many
# CONSECUTIVE iterations. With W walkers the max gets W independent chances
# per iteration to reset that clock, so the same iteration count is weaker
# evidence of a plateau at small W and the cap ratchets FASTER the fewer
# walkers run. Every previous value (5, 4, 3) was tuned at 10 or 24 walkers;
# this store runs 4. Measured on the 4-walker run, iterations 35 -> 68:
# summed cap 1232 -> 2373 (+34.6 slots/it) against cold leaves 420 -> 1066
# (+19.6/it), i.e. 1307 cap slots of unused headroom, 202 bands at cap >= 4
# with a max of 7 while the median band still held ONE source -- and the
# flagship 20.38 mHz source carried THREE fragment leaves per walker.
# Matching the ~30 walker-iterations that 3 bought at 10 walkers would mean
# 8 here. The knob is NOT walker-aware, so raising it is the operator's
# call; the fresh store's fixed warm start is the other half of that
# fragmentation story and is being addressed separately.
export GB_LEAF_CAP_MIN_ITERS=3
export GB_CAP_LL_CHECK=1
# Grouped RJ scheduling: accumulate inds=True picks across RJ rounds
# (1 proposal per cell per round), then ONE full-width in-model block.
# Code default since 2026-08-13; pinned for the run record. =0 restores
# the per-round RJ->in-model interleave.
export GB_RJ_GROUPED_INMODEL=1
# ---- 2026-08-14 rj stack (ALL code defaults; pinned for the run record;
#      each =0 reverts that piece independently) --------------------------
# ⚠ 1 -> 0 (user ruling 2026-09-25). THE SCHEDULE IS THE POINT.
#
# =1 is the rigid-batch path: it runs ALL RJ rounds -- one source per cell
# per round, ~12 rounds per propose measured on job 621 (360k birth
# proposals over ~29.6k cells) -- and only then does ONE end-of-unit
# in-model phase. So a cell could accept a dozen births with nothing
# polished in between, which is how the 19.66822 mHz source ended up with
# 2-3 leaves in 33 of 96 (rung, walker) cells, the extras at 0.76-0.99x
# amplitude because they were never polished at all.
#
# =0 is the staged scheduler, which is the schedule the user asked for:
#   RJ round -> survivors pool, their cells FREEZE
#            -> flush: in-model TO CONVERGENCE on the pool
#            -> cells UNFREEZE, completed cells retire and new bands stage
#            -> next RJ round
# ("RJ 1 round -> in model converge -> RJ 1 round -> in model converge",
#  and "after each converge, you sub out bands that are wholly complete
#  for new bands" -- the latter is _advance_and_refill, called immediately
#  after every flush.) PE gets the same schedule WITHOUT the convergence
#  for free: _converge_state_for returns None on a PE-stage move.
#
# ⚠ COST, unmeasured on this hardware: the rigid path exists because "the
# fixed, repeating fill -> rounds -> accept shape per batch is what CUDA
# graph capture needs". The staged scheduler's shape changes every round,
# so graph capture is off. Watch the [GB_TIMING] rj_step / inmodel_repeats
# split against the 621 baseline; =1 reverts.
export GB_RJ_DIRECT_BATCH=0        # staged scheduler: RJ round -> converge
                                   # -> refill -> RJ round; =1 = rigid batches
# LIVE-CAP PICK OFF (user ruling 2026-08-27, snapshot-2 timing autopsy):
# under the ALIGNED divisor-1 grid, "saturated across all K cap cells"
# is "all 1" -- every occupied band trips it at cap 1, so the live-cap
# regime staged 879,846 dead at-cap birth slots (all temps) through the
# sig-het in-model staging at ~2.3 s per 1024-slot sub-block ~= 220 s
# per band unit -- THE 14x rj_fstat_search blowup vs v6 (whose
# divisor-4 grid needed all 4 cells saturated, i.e. almost never).
# =0 restores the 2026-08-12 unit-open exclusion: at-cap cells' dead
# slots never stage; a cell freed by a death births next UNIT instead
# of same-unit (immaterial at our removal acceptance).
export GB_RJ_LIVE_CAP_PICK=0
export GB_BUFFER_FIXED_CAPACITY=1  # ONE capacity buffer; smaller units
                                   # resize-rebind instead of drop+rebuild
export GB_RJ_FSTAT_CTR_HOIST=1     # F-stat distance centers batched once per
                                   # unit (was 735 s/propose per-round)
# EPOCH CENTERS (user ruling 2026-08-15: compute the center
# distributions ONCE when the fstat distribution is built in setup(),
# smear for inaccuracy, done): per-epoch table over the proposal's
# drawable support; propose-time = nearest-node lookup (centers chain
# 109-953 s -> ~0). Smear defaults 2.0 in epoch mode (covers <=100-
# propose staleness + node mismatch); =unit restores the per-unit hoist.
export GB_FSTAT_CTR_MODE=epoch
# (GB_FSTAT_CTR_SMEAR is unset ON PURPOSE: a 1.5 pin would override the
# epoch-mode 2.0 default, which is what covers the <=100-propose table
# staleness. The smeared sigma feeds BOTH the draw and the densities,
# so detailed balance is exact at any smear.)
#
# ############################################################################
# ## v7 CHANGE-SET FOLDED IN (2026-08-28). The v8 script was cut from the   ##
# ## PRE-change-set v7, so these were missing. Every one was validated in   ##
# ## v7 production (jobs 352-368, snapshots 8-12) unless noted.             ##
# ############################################################################
# FUSED TWO-QUADRATURE PHASE MAX (GBGPU c49fcb1 / LAT 9704c4a8). One kernel
# call returns both quadratures instead of two evaluations at phi0 and
# phi0+pi/2. GPU-VALIDATED in production job 352: 0 errors over 86k log
# lines, TEMPER_CHECK 297/0, COLD audit medians at baseline. =0 is the
# no-rebuild rollback to the legacy two-call path (bit-identical algorithm;
# the fused path is epsilon-better, NOT bit-identical).
export GB_PHASE_MAX_FUSED=1
# F-STAT 4->2 BASIS-FILTER FOLD (GBGPU 8245ced / LAT ae8cdb87+07bbac97).
# The chunked-WDM F-stat kernel generated four independent waveforms, but
# the four basis filters are 2 polarization directions x 2 phase
# quadratures (iota=pi/2 throughout, psi in {0, pi/4}, phi0 in quadrature
# pairs) -- and the quadrature half is a constant unit-modulus rotation
# {+1,-1,+i,-i} of the complex heterodyned representation, which the
# d_h_im_out machinery already produces. Two generations span the
# identical 4-space (rank 4, condition 1.000), so this is EXACT: no
# physics loss, no sampling-semantics change.
# VALIDATED: sign pinned CPU-side against a deliberately CONJUGATED build
# that failed the gate, then the parity suite passed in full ON GPU.
# The gate is on SIGNED (N, M) and must never be reduced to F -- a
# conjugated sign is N->DN, M->DMD with D=diag(1,1,-1,-1), which leaves F
# EXACTLY invariant (tests/test_fstat_filter_fold.py pins this).
# EXPECT rj_fstat_centers ~830 s -> ~400-450 s, i.e. ~18-20% off the GB
# iteration. The OFF baseline is already banked (snapshot 13, 830.2 s).
# =0 restores the unfolded 4-generation path bit-for-bit.
# REQUIRES A GBGPU REBUILD -- LAT does not compile that header. A stale
# .so is a LOUD TypeError at the first F-stat call, NOT a silent no-op
# (corrected 2026-08-28; the earlier note here said the opposite).
# _FSTAT_FOLD_KERNELS in gbcomps.py is a hard constant rather than a probe
# of the compiled binding, and the trailing fold arg is passed
# unconditionally -- fold ON or OFF -- so a binding that predates the fold
# cannot accept it and raises immediately. rj_fstat_centers ~830 -> ~400-450
# s is still the check that the fold is doing work, but a bad build fails
# first and visibly.
export GB_FSTAT_FOLD=1
# REPLACE PHASE-MAX + ROTATION-ON-ACCEPT. "auto" (not =1): ON for the
# search replace exactly as =1 was, OFF for any PE-stamped replace. A hard
# =1 would force maximization onto PE, which the 2026-08-28 general rule
# forbids ("no maximizing over parameters during PE" -- PE samples a
# posterior; maximize-and-keep biases it). Live since job 352: cold replace
# acceptance ~3x (0.0002-3 -> ~0.0010) at Delta-ll up to ~700.
export GB_REPLACE_PHASE_MAX=auto
# PER-ROW F-STAT CENTERS THROUGH THE UNIT-OPEN CACHE (LAT 86ed9353).
# ⚠ MEASURED A WASH in snapshot 12: rj_fstat_centers 725-743 s vs a
# 713-799 s pre-fix band. There was no recomputation to dedupe -- the
# precompute row count ~= the picked row count, at an identical 0.667
# ms/row. Kept ON because it is the code default and costs nothing either
# way; it is NOT a speed lever. The real centers levers are the
# multi-device lane rebalance and row-count reduction (see the scoping
# notes). Telltale: [FSTAT_CTR] says "perrow (unit-cache)".
export GB_FSTAT_PERROW_UNIT_CACHE=1
# DEFERRED CELL-LABEL RELABELS (LAT 9fa32109; code default flipped ON in
# bcdde159 per the user's "only cells change labels" design invariant).
# Rung-pair/vertical-swap relabels accumulate in a slot+pos composition
# table and flush once per tempering chunk / repeat block. Pinned here for
# the run record. Tripwires: [GB_TEMPER_CHECK] must stay 100% MATCH with
# "unit label checks passed" (340/340 in snapshot 12).
export GB_CELL_LABEL_DEFERRED=1
# FUSED IN-MODEL GATE/ACCEPT KERNEL (LAT 0f0fc73a + the 07634536 nvcc
# guard fix). ~160 CuPy launches per repeat-step -> 3 backend calls.
# ⚠ OFF FOR v8 PENDING THE v7 A/B (user ruling 2026-08-28: "if the
# multi-GPU picture is not clearly better, keep it off for v8"). Expected
# size is small -- ~450k launches removed => single-digit seconds, 0.2-1%
# of a ~1130 s propose; the real hope was multi-GPU overlap, since the
# kernel leaves ONE data-dependent host sync where the python chain had
# many. FLIP TO 1 only if the v7 run's `inmodel_gate` mark plus
# gpu_util_*.csv show a clear multi-GPU improvement. Requires ./install.sh
# to have built the binary; without it the loaders degrade to the python
# chain with a one-line warning (safe, just not faster).
# >>> FLIPPED TO 1 (2026-09-11, ported from the 10w arm de903711): first
# >>> production arming was the 3mo relaunch after job 473; est. -30..-40
# >>> s/it. VERIFY on this run: in-model acceptance rates + [GB_CELL_LL]
# >>> + sig-het audit unchanged vs the python chain. =0 reverts.
# BACK TO 0 (2026-09-11 user ruling, ported from the 3mo 10-walker arm):
# it fuses only the inmodel_gate + inmodel_accept launches, ~18 s/it at
# 250/100 repeats and ~9 s at 100/50 -- not worth a first production
# exposure. Turn on only after tests/test_gb_inmodel_accept_kernel.py has
# been run on a cluster GPU node. (With GB_INMODEL_TRACE armed it stood
# down anyway -- job 474 logged "standing down".)
export GB_INMODEL_ACCEPT_KERNEL=0
# ---- THE v8 EXPERIMENT: OBSERVABLE-BASIS IN-MODEL PROPOSAL ----------
# Pinned EXPLICITLY even though it is now the code default, so this run
# does not silently change meaning if the default is ever revisited, and
# so the v7/v8 knob diff names the thing under test.
#
# WHAT IT CHANGES. In-model steps are drawn in
#   z = [lnA, f_mid, fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc]
# and mapped back; the SAMPLING basis is untouched (f0 stays anchored at
# t_ref, so bands, cap cells, storage and the catalogue conversion are all
# unaffected). Two measured defects motivate it, both invisible in the
# sampling basis:
#   1. (dist, Mc, r) -> (A, fdot) is 3->2, so Mc and r both drive fdot and
#      a joint draw can move both and leave fdot_total put.
#   2. f0 is the frequency at the START of the data while the data
#      constrains the frequency at the MIDDLE. On the real flagship Fisher
#      the legacy joint draw walks an f0-fdot ridge of slope -0.898 T
#      against the geometry's -T/2; the excess is 0.170 bins of spurious
#      f_mid motion per fdot step, ~14 sigma at rho = 46.
#
# WHAT TO WATCH, in [GB_ACCEPT] / [GB_OBS_BASIS] lines:
#   * "in-model by proposal type -- obs_basis:" replaces "infomat:"
#   * cold acceptance 0.67 (v7) should fall toward 0.23-0.44
#   * "[GB_OBS_BASIS ...] in-model motion" gives mean |dln_fdot| and
#     |df_mid| in bins, proposed and accepted -- the direct read of
#     whether fdot is finally moving
#   * the flagship's fdot/truth should walk off 1.35
# FALSIFIABLE PREDICTION: the shear goes as fdot*T^2, so this must be
# essentially NEUTRAL below ~7 mHz (0.04 bins) and large above ~15 mHz
# (3.1 bins). A material low-f change means a BUG, not a win.
#
# =legacy reverts to the v7 proposal bit-identically (same seed), which is
# what makes the comparison readable. Keep that path until v8 is read.
export GB_INMODEL_PROPOSAL=observable
# Fiber (Mc) component of the composite step. 0.0 = the 8-observable step
# only; gb_ridge_gibbs already supplies fiber mixing on the main state for
# free, so this stays an INDEPENDENT A/B rather than a coupled one.
export GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0
# Overall step multiplier for this path. Deliberately its own knob and NOT
# the legacy jump_factor, which was tuned against the eigen-floored draw.
# RAISED 1.0 -> 2.0 (2026-09-19). Under GB_INMODEL_OBSERVABLE_EIGEN=axis
# the GB in-model step came out TOO SMALL: measured on the 6mo run,
# obs_basis acceptance cold 0.71 / all-rung 0.78, with
#   mean |df_mid| proposed 0.0184 bins vs 0.0032 accepted
# against the well-tuned 10-walker diagonal arm's proposed 0.038 /
# accepted 0.021 at acceptance 0.219. A one-axis-per-repeat draw is
# effectively 1-D, where the optimal acceptance is 0.44 (not 0.234), so
# 0.71 says the step wants roughly doubling -- which is this knob, and
# deliberately NOT GB_JUMP_FACTOR (that one was tuned against the
# eigen-floored draw; see the block above).
# 2.0 -> 1.5 (2026-09-28, Mike). The 09-19 doubling to 2.0 was too
# far, and the evidence is per-axis: under
# GB_INMODEL_OBSERVABLE_EIGEN=axis ONE eigen axis is drawn per repeat,
# so an axis's accepted/proposed motion ratio IS its own acceptance.
# Job 663 measured mean |df_mid| accepted/proposed at 0.01-0.34 and
# |dln_fdot| at 0.04-0.10 against a POOLED cold acceptance of
# 0.55-0.79 -- i.e. f_mid was accepting ~7% of its own draws and fdot
# 3-7%, while the other seven axes ran ~80% to make up the pooled
# number. The doubling itself made accepted f motion SMALLER
# (0.0032 -> 0.0020 bins), which is the signature of a step already
# past its optimum on that axis.
#
# ⚠ ONE GLOBAL KNOB CANNOT FIX BOTH HALVES: 1.5 helps f_mid and fdot
# and costs the seven axes that were happy at 2.0. It is an interim
# compromise until the per-axis / per-source scaling lands (that work
# is on the gb-inmodel-axis-adapt branch, deliberately NOT on dev).
# The per-axis census added alongside this is what will show whether
# 1.5 moved the right axes.
export GB_INMODEL_OBSERVABLE_JUMP=1.5
# Mc step as a FRACTION of the m_chirp prior box (only reachable when the
# fiber weight above is non-zero).
export GB_INMODEL_OBSERVABLE_MC_STEP=0.05
# Shear coefficient as a fraction of Tobs; 0.5 => f_mid. The shear has
# determinant 1 for ANY coefficient (verified for 0, T/2, 0.41T, T, -3T),
# so a wrong value here costs acceptance and never correctness.
export GB_INMODEL_OBSERVABLE_SHEAR=0.5
# ---- COMBINED WITH THE EIGENBASIS (user ruling 2026-09-14: "GB in model
# should always be observed basis. We should combine that with the
# eigenbasis.") ------------------------------------------------------------
# The step STAYS in observable z with the same log-Jacobian factors, but
# instead of independent per-coordinate steps it draws along the eigen
# table of each source's own information matrix congruenced into z (exact
# chain rule through the transform -- no extra waveform calls) and
# whitened by the analytic step scales above. "full" = the JOINT
# correlated draw over all axes per repeat, the whitened shear-free modern
# version of the legacy full-covariance infomat draw (the old proposal was
# never diagonal -- user preference 2026-09-14); "axis" = one uniformly
# picked eigen-axis per repeat is the one-knob alternative; empty/0
# reverts to the independent per-coordinate draw BIT-IDENTICALLY, RNG
# stream included. Rows without a table yet (fresh births before their
# first infomat visit) silently take the diagonal draw -- by design.
# GB_INMODEL_OBSERVABLE_EIGEN_SMAX (default 10.0) caps the whitened
# sigmas. Watch the [GB_TIMING] span "infomat_obs_eigen" (the per-block
# Gamma_z stash) and the obs_basis cold acceptance vs the 3mo arm's ~0.2.
# THIS RUN IS THE TEST (user ruling 2026-09-14: "Let's just go to the
# full 6mo run test") -- the 3mo probe script
# (submit_gf_3mo_v8_10w_eigenaxis_probe.sh) remains available as an
# ISOLATION arm if the combined draw needs to be separated from the 6mo
# changes. Launch with GB_INMODEL_OBSERVABLE_EIGEN= (explicitly empty)
# to fall back to the diagonal draw bit-identically if it reads badly.
# DEFAULT CHANGED full -> axis (user ruling 2026-09-19). "full" is a JOINT
# draw over every eigen axis per repeat, so ONE ill-conditioned direction
# contaminates EVERY proposal: an observed-information eigenvalue projected
# onto the PSD cone leaves lambda ~ 0, its whitened sigma (~1/sqrt(lambda))
# rails at GB_INMODEL_OBSERVABLE_EIGEN_SMAX = 10, and the step overshoots
# along an axis the matrix could not measure. Measured cost in full_pe:
#     GB obs_basis cold 0.0139 (full)  vs  0.0864 (10-walker arm, no eigen)
#     mean |df_mid| proposed 0.218 bins vs 0.026 accepted -- 8.5x too wide
#     (the 10-walker arm: proposed 0.038, accepted 0.021, ratio 1.8)
# "axis" draws ONE whitened eigen-axis per repeat, so a bad direction costs
# only the 1-in-ndim repeats that pick it instead of all of them. That is
# exactly the mode EigenAxisMove(mode="axis") uses for mbh / emri / sobbh,
# and those three are healthy (accepted fraction 0.57 / 0.16 / 0.26).
# Escapes, both bit-identical to what they name:
#   GB_INMODEL_OBSERVABLE_EIGEN=      -> the diagonal observable draw
#   GB_INMODEL_OBSERVABLE_EIGEN=full  -> the 2026-09-14 joint draw
export GB_INMODEL_OBSERVABLE_EIGEN=${GB_INMODEL_OBSERVABLE_EIGEN-axis}
echo "[GB-OBS-EIGEN] GB_INMODEL_OBSERVABLE_EIGEN='${GB_INMODEL_OBSERVABLE_EIGEN}' (empty = diagonal draw)"
# ---- AND THE F-STAT GRID IN THE SAME BASIS -------------------------
# fdot becomes a FIRST-CLASS grid axis instead of the r = 0 manifold the
# grid searches today. Measured in v7: 39.6% of low-f and 10.5% of high-f
# alive leaves carry fdot < 0, which the current grid CANNOT represent --
# every one of them arrived from the blind post-hoc r-draw or in-model
# drift, never from the search. The new axis also costs FEWER nodes
# (20.38 mHz: 71 -> 53, and v7's top group used exactly 71) while covering
# 10x the range, because the aligned coherence width is 13.4x coarser.
#
# Complementary to the in-model move above -- the grid places births on
# the ridge, the move refines them along it -- so shipping one alone can
# read as "no effect". ATTRIBUTION comes from disjoint observables, the
# sharpest being the fraction of births with fdot < 0: the old grid cannot
# produce one at all, so any non-zero count is unambiguously this change.
#
# WARNING: an *_peaks_stacked.npz fitted in the Mc basis is REFUSED on
# load (by design -- axis 2 is a chirp mass in one basis and Hz/s in the
# other). v8 starts fresh, so it refits; a resume of an older store would
# need its epoch caches cleared.
export FSTAT_FDOT_AXIS=1
export FSTAT_FDOT_RATIO_MAX=5.0

# PEAK-SELECTION FLOOR 8.0 -> 6.25 (user ruling 2026-09-23: "err on the side
# of grabbing things not missing them. Even if there is excess stuff that is
# fine.")
#
# NOTE ON THE RUN IN FLIGHT: this knob was never exported before, so the
# current 6mo run has been selecting at the stock 8.0. That is what the
# measurement below is against.
#
# WHY THE STOCK 8.0 IS TOO STRICT *ON THIS RUN*. The 8.0 default is a
# false-alarm argument (fstat_proposal.py): under the null 2F ~ chi2_4, so
# F = 32 gives ~4e-5 expected false peaks on a full-band production comb.
# That null assumes the ASSUMED PSD equals the true one. Here it does not --
# galfor absorbs the galaxy the search has not resolved, so the assumed floor
# is k = 1.85-2.00x the truth over 3-4 mHz. The realized null is then
# 2F ~ chi2_4 / k: the whole distribution SHRINKS, the cut becomes orders of
# magnitude stricter than designed, and the cost is paid entirely in missed
# sources. Measured 2026-09-23 (galfor_6mo_figs/): 8.0 is an effective
# TRUE-SNR threshold of 11.3 at 3.5 mHz, hiding 693 findable binaries, 506 of
# them in 3-5 mHz.
#
# WHY 6.25 AND NOT LOWER. Expected false peaks against this run's own k(f),
# with the comb size back-calibrated from the ruling's own 4e-5 number:
#
#     cut 8.00 -> 6e-5 false,     0 of 693 recovered   (the run in flight)
#     cut 7.00 -> 0.06 false,   319 recovered
#     cut 6.50 -> 1.2  false,   478 recovered
#     cut 6.25 -> 5.3  false,   551 recovered          <-- HERE
#     cut 6.00 -> 21.7 false,   628 recovered
#     cut 5.66 -> 134  false,   692 recovered
#
# 6.25 buys 73 more sources than 6.50 for ~4 more expected false peaks. The
# marginal rate collapses just below it -- 6.50->6.40 is 36 sources per false
# peak, 6.25->6.20 is 7, 6.10->6.00 is 3.7 -- so 6.25 sits at the last point
# where the exchange is still clearly favourable. 6.0 and below is the regime
# the 2026-08-17 SNR-8 ruling was written to escape (a peak list that was
# ~98% noise put 2-7 leaves on one source).
#
# THIS IS A COMPENSATION, NOT A FIX, AND IT IS RUN-SPECIFIC. It trades a
# known-inflated floor for a looser cut; do NOT port it to a run whose noise
# model is well calibrated, where it would simply loosen the false-alarm rate
# for nothing. The principled replacement is a per-band self-calibrated
# threshold (8/sqrt(k_hat), k_hat measured from the F-stat grid's own median)
# which recovers ALL 693 at the 4e-5 design rate -- see
# galfor_6mo_figs/FSTAT_SELFCAL_PROPOSAL.md. Revert to 8.0 when that lands,
# or when GB has subtracted enough that galfor is no longer inflated.
#
# CAVEAT the table cannot show: the chi2_4 null only holds where the residual
# is Gaussian. At 1-3 mHz the confusion forest is not, so the real false rate
# there is worse than tabulated at ANY global cut. Watch the per-band peak
# counts and the birth multiplicity per source in the 1-3 mHz bands.
#
# ############################################################################
# ## ⚠⚠ THIS KNOB IS A NO-OP ON A RESUME UNLESS YOU CLEAR THE STAGE-B CACHE ##
# ##                                                                        ##
# ## fstat_gridfit's stage-B load path is "load and return, nothing         ##
# ## recomputed": the peak LIST was selected at whatever threshold was in   ##
# ## force when it was FITTED, and nothing on that path re-reads the knob.  ##
# ## So a resume keeps proposing from the old SNR-8 list while this script  ##
# ## and the run log both say 6.25.                                         ##
# ##                                                                        ##
# ## THE CHEAP MIGRATION -- delete ONLY the stacked file, KEEP the comb:    ##
# ##                                                                        ##
# ##     rm <fit_dir>/fstat_grid_peaks_stacked.npz                          ##
# ##     # keep  <fit_dir>/fstat_grid_comb.npz                              ##
# ##                                                                        ##
# ## The comb stores F_max for EVERY node, so peaks re-select at the new    ##
# ## threshold (cheap, deterministic) and only stage B reruns. Deleting the ##
# ## whole epoch directory also works but pays for the comb again -- and    ##
# ## the comb is the expensive half (monitor_6mo.html was still reporting   ##
# ## "the grid fit is still running" at the Sep-18 snapshot).               ##
# ##                                                                        ##
# ## As of 2026-09-23 the stage-B cache STAMPS peak_min_F and the loader    ##
# ## REFUSES a mismatch (fstat_gridfit._check_cached_peak_threshold). But   ##
# ## caches written BEFORE that stamp -- including this run's -- carry no   ##
# ## stamp and only WARN. Watch for "carries no peak_min_F stamp" in the    ##
# ## log on the first resume; that line means you must clear it by hand.    ##
# ############################################################################
# V9-11: 6.25 -> 8.0. Per-stage again: gb_search_1 selects peaks at SNR 8,
# gb_search_2/3 at 6.25. ⚠ THIS ONE CANNOT BE MOVED BY AN ENV VAR MID-RUN --
# the floor is consumed when the grid is FITTED and stamped into the stage-B
# cache, whose loader REFUSES a mismatch. So the stage entry installs an
# in-code override (fstat_proposal.set_peak_min_F_override) AND forces a
# fresh-epoch refit; the old epoch stays on disk, valid and unread, and
# NOTHING IS DELETED. Watch for:
#   [V9-STAGE gb_search_2] F-stat peak floor -> SNR 6.250 (F 19.531)
#   [V9-STAGE rj_fstat_search] FORCED F-stat refit: opening epoch N
# This export is the value epoch 0 is fitted at, i.e. stage 1's.
export FSTAT_PEAK_MIN_SNR=8.0
# F-STAT CENTERING OFF (probe verdict 2026-09-02, 4-arm A/B, readout
# artifact 2f5d673c): centered births are a stacking engine in BOTH test
# bands (A_ctr multiplicity 4.33/walker on a 1-source band, 23% neg-fdot
# junk; B_ctr cold logL max FELL 1,478 nats after it 65) while OFF is
# clean in both -- and best logL was identical-within-1-nat, so the junk
# bought nothing. Maximizing all F-stat quantities finds local maxima too
# easily. This also retires v7's rj_fstat_centers cost (54%/iteration).
# ⚠ THE ENV ALWAYS WINS over the install-site stamps
# (_fstat_dist_birth_stamp): =0 ALSO reverts the 08-28 "mirror them" PE
# ruling, so rj_fstat_pe births go back to full prior widths
# (bit-identically the pre-stamp path). That is DELIBERATE here: the
# clean probe arms ran exactly this env, so this is the validated
# configuration; a PE-only re-arm needs its own scoped knob first.
export GB_RJ_FSTAT_DIST_BIRTH=0
# RJ PHASE MAXIMIZATION (user ruling 2026-09-02, r2 probe verdict). With
# centering off, birth maximization would otherwise be GONE ENTIRELY: in
# v7-style search the maximizers came VIA the centers path, and
# GB_RJ_PHASE_MAXIMIZE defaults 0. The r2 probes measured the difference
# directly -- with phase-maximized birth scoring the flagship was ON
# TARGET at the FIRST stored iteration (-0.23 bins / fdot 0.92x,
# multiplicity 1.00 all run, 0% negative-fdot leaves) where the
# unmaximized r1 arm needed ~10 iterations; low-f stayed neutral and all
# 979 live detailed-balance traces still matched (phase max touches
# scoring only, never the observable-basis factors).
# V9-11: 0 -> 1. Phase maximization is now a PER-STAGE PROFILE value:
# gb_search_1 ON (hunting -- take the credit, and the r2 probes measured the
# flagship ON TARGET at the first stored iteration with it), gb_search_2 and
# gb_search_3 OFF. The profile sets it on the RJ moves only; the pure
# in-model moves stay at their deliberate False (in-model scoring is at the
# ACTUAL phase). As with the opt-SNR floor, this export is stage 1's value so
# a profile that failed to apply would hold stage 1's configuration rather
# than run stage 1 unmaximized.
export GB_RJ_PHASE_MAXIMIZE=1     # birth lands on target at iteration 0
# Amp max DEFAULTS TO FOLLOW phase_maximize -- it would silently arm with
# the line above. User ruling: phase max only, no amp max for now.
export GB_RJ_AMP_MAXIMIZE=0
# LIVE DETAILED-BALANCE TRACE (audit follow-up 2026-09-02). Recomputes the
# observable-basis log-Jacobian INDEPENDENTLY for one traced source per
# repeat and prints "-> match" / "*** MISMATCH ***". Negligible cost, and
# this run is the FIRST MULTI-GPU exposure of the factors path -- every
# probe ran gres=gpu:1 -- so the one failure class the probes could not
# see (a device-routing fault in factors) is exactly what this catches.
# 979/979 matched in the r2 probes; any MISMATCH line here is a stop
# signal. =0 disarms.
# 0 (2026-09-11, was 1 -- ported from the 3mo 10-walker arm): the
# per-repeat MH trace is a DEBUG knob (DEBUG lines per repeat, host
# syncs) and, while armed, the fused accept kernel stands down. Re-arm
# only for a diagnostic run.
export GB_INMODEL_TRACE=0
# DIFF DISCIPLINE: v7 exported GB_CAP_DIAG=1 and it costs time. Leaving it
# on in v7 and off in v8 would make v8 look faster for reasons unrelated
# to the proposal, so it is pinned ON here -- and the cap census is wanted
# regardless. (GB_FSTAT_CTR_AUDIT stays absent: measured at 0.260 s of a
# ~2000 s propose, 0.013%, so it carries no timing weight, and its verdict
# is already in -- see below.)
export GB_CAP_DIAG=1
# NOT PINNED, ON BY DEFAULT -- recorded so the run log is interpretable:
#  * GB_REPLACE_FSTAT_MAX resolves "auto" = ON for the search replace via
#    the recipe's replace_search_stage stamp (the move is named plain
#    "rj_replace", so the name idiom alone would MISS it). Search replace
#    candidates are the full JKS maximizer: slot 0 pinned AT the per-row
#    F-stat center, then priced through the UNCHANGED RJ densities as if
#    drawn (maximize-then-pretend). Telltale: one [GB_REPLACE_FSTAT_MAX]
#    line. =0 restores the exact-DB draw bit-identically.
#  * GB_PE_RJ_REPLACE: now pinned OFF above (user ruling 2026-09-02) --
#    the default-ON rj_replace_pe registered but never proposed in either
#    r2 probe, so it shipped as inert machinery.
#  * GB_RJ_FSTAT_DIST_BIRTH: the rj_fstat_pe stamp WOULD arm epoch-table
#    centers for PE births (user ruling 2026-08-28 "yes mirror them"),
#    but the explicit =0 export above overrides every stamp -- probe
#    verdict 2026-09-02, see that block. PE births draw at full prior
#    widths in v8.
# GB_FSTAT_CTR_AUDIT is DELIBERATELY ABSENT: it is a v7-only diagnostic
# (table-vs-per-row center deltas). Re-arm it here only if v7 never
# produced a completed propose to read it from.
#
# ITS VERDICT IS ALREADY IN (v7, 2026-08-29, 18 unit lines): the epoch
# table CANNOT serve birth centers. Medians dphi0 1.21-1.29 rad, dpsi
# 0.57-0.63 rad, dcos_iota 0.22-0.29, dln_center 0.81 -> 1.21 (factor ~3
# in amplitude) -- with a TINY f0 node gap (~1.1e-8 Hz ~ 0.08/Tobs), so
# the gap is not resolution: the table is solved against the epoch-0000
# residual and the F-stat maximum is data-dependent, drifting as sources
# are subtracted. That is why v8 does not re-run the audit and does not
# use the table. The per-row solve stands; the two levers below are what
# is left.
#
# ---- F-STAT COST LEVERS (both bit-identical at their defaults) ------
# rj_fstat_centers measured at 61% of a v7 iteration, ~100% inside
# fstat_nm_lane_score. GB_FSTAT_CTR_BATCH is the serial python chunk
# (4096 default => only ~2048 rows per device per launch, ~1,409 launches
# per propose); raising it is pure re-chunking, results unchanged.
# ⚠ GPU0 peaked at 90,698 MiB of 95,830 in v7 -- step 4096 -> 8192 first.
# GB_FSTAT_NM_LANE_WEIGHTS reweights the CONTIGUOUS row lanes across
# devices (v7: GPU0 39.1% util vs GPU1 72.1%, yet split 50/50, so every
# join waited on the busier card). Integer weights in holder.gpus order,
# e.g. "3,2" = 60/40; unset/blank/malformed => the exact equal split.
# Both are overridable from the submitting shell without editing this
# file, e.g.  GB_FSTAT_CTR_BATCH=8192 sbatch <this script>
: "${GB_FSTAT_CTR_BATCH:=4096}"
export GB_FSTAT_CTR_BATCH
: "${GB_FSTAT_NM_LANE_WEIGHTS:=}"
export GB_FSTAT_NM_LANE_WEIGHTS
export GB_TEMPER_ON_REMOVAL=1      # band swaps run inside rj_prior_removal
# rj_prior_removal proposes prior BIRTHS as well as deaths in this run
# (what the 2026-09-21 hand edit did from stored iteration 300 on). The
# code default is 1 = the original deaths-only pruning move.
# 0 -> 1 (2026-09-26, carried from submit_gf_6mo_v9_4gpu.sh 938684fd:
# "pruning back on"). rj_prior_removal returns to the deaths-only pruning
# move instead of also proposing prior births. Carried for the same reason
# as the knob above -- it is a v9 tuning ruling, not a Tobs property.
# ⚠ AND IT CANNOT BE MADE TO, cheaply. rj_removal_only IS read at
# propose time (gbspecialstretch :7295 and four more), so flipping it on
# stage entry would work -- but SearchStageProfileStep._apply_profile
# writes its profile to EVERY GB move in the stage tree, and this
# attribute is not safe to broadcast: on rj_fstat_search, True makes
# apply_inds true (alive rows only => NO births at all, silently
# gutting the search), and rj_removal_only with rj_replace RAISES
# (:4383). A per-stage form therefore needs move-SCOPED profile
# entries, which _apply_profile has no concept of -- real plumbing with
# a loud failure mode, not a table row like GB_SEARCH_3_OPT_SNR.
#
# OVERRIDABLE FROM THE COMMAND LINE (2026-09-29, Mike: "let's also
# adjust the gb search 3 stage prior removal to be not just removal
# only"). The hard-coded value meant an exported 0 could not reach the
# run at all. Default unchanged; the stage-3 test passes 0, and because
# the store resumes INSIDE stage 3 the global scope is not a problem
# for it.
export GB_SEARCH_PRIOR_REMOVAL_ONLY=${GB_SEARCH_PRIOR_REMOVAL_ONLY:-1}
echo "[GB-PRIOR-REMOVAL] GB_SEARCH_PRIOR_REMOVAL_ONLY=${GB_SEARCH_PRIOR_REMOVAL_ONLY} (1 = deaths only; 0 = prior births AND deaths, ALL search stages)"
# ---- the galfor RATCHET (user design 2026-09-30) ----------------------------
# GALFOR_RATCHET=1 on the launch line: gb_search_3 carries ONE gated noise
# proposal at the head of the iteration instead of the rider plus the four
# interleaved slots. Stage-local schedule: iteration 0 of a cycle FORCES the
# galfor coordinates down (DLOG10_* in the sampled log10 basis, every rung and
# walker, published through the noise move's own accept path, hard F-stat
# refit armed, then one GB in-model pass), the next HOLD-1 iterations hold the
# noise fixed, then RELEASE iterations run the ordinary joint noise search
# (each followed by the same in-model pass). CYCLES nudges, then permanent
# release. The release is the measurement: a region of the curve that comes
# back was honest, one that stays down had been absorbing resolvable sources.
# STEP SIZE: one nudge takes the 3-5 mHz total noise from 2-5x the add-back
# estimate to ~0.75-1.5x, where the F-stat peak floor (SNR 6.25) admits true
# SNR-8 sources across the band (lisatools.globalfit.noise_ratchet docstring).
# CYCLES=2 (user ruling 2026-09-30, "do at least 2 cycles for now"). ⚠ The
# second nudge applies the same delta to whatever the first release left: on
# a fit that stayed down it lands at 0.2-0.6x the estimate below 4.5 mHz, so
# read its release with the junk indicators (birth truth-partner fraction,
# power share per band) before going further.
# Overridable here, default OFF = today's composition byte-identical.
export GALFOR_RATCHET=${GALFOR_RATCHET:-0}
export GALFOR_RATCHET_HOLD=${GALFOR_RATCHET_HOLD:-3}
export GALFOR_RATCHET_RELEASE=${GALFOR_RATCHET_RELEASE:-2}
export GALFOR_RATCHET_CYCLES=${GALFOR_RATCHET_CYCLES:-2}
export GALFOR_RATCHET_DLOG10_AMP=${GALFOR_RATCHET_DLOG10_AMP:--0.05}
export GALFOR_RATCHET_DLOG10_FK=${GALFOR_RATCHET_DLOG10_FK:--0.10}
export GALFOR_RATCHET_DLOG10_F2=${GALFOR_RATCHET_DLOG10_F2:--0.15}
# ---- search LEGS (user design 2026-09-30) ----------------------------------
# GB_SEARCH_LEGS=1: the numbered search stages store one row per LEG of the
# cycle -- after in_model, after in_model_fstat, after in_model_removal --
# instead of one per full cycle. The resume position is the NAME of the
# in-model move the last row was saved after (stored per row), so a gated
# move that skipped a cycle cannot mis-align a relaunch; cadences
# (GB_SEARCH_3_WARM_EVERY) and the galfor ratchet then count CYCLES. With a
# row per leg the mid-iteration checkpoint only guards part of one leg;
# MIDIT_CHECKPOINT=0 is reasonable for a legged search. Off = today.
export GB_SEARCH_LEGS=${GB_SEARCH_LEGS:-0}
echo "[GALFOR-RATCHET] GALFOR_RATCHET=${GALFOR_RATCHET} hold=${GALFOR_RATCHET_HOLD} release=${GALFOR_RATCHET_RELEASE} cycles=${GALFOR_RATCHET_CYCLES} dlog10 amp/fk/f2=${GALFOR_RATCHET_DLOG10_AMP}/${GALFOR_RATCHET_DLOG10_FK}/${GALFOR_RATCHET_DLOG10_F2} (0 = off: rider + 4 interleaved noise slots as before)"
# High-f barren-band birth shutoff (search scope): bands above FMIN with
# AFTER consecutive zero-birth-accept proposes stop proposing births
# (deaths + in-model continue; [GB_BAND_SHUTOFF] log line per band).
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
# PATIENCE 50 -> 5 (user ruling 2026-08-28). 5 is the CODE DEFAULT; this
# pin is what disabled the valve, and the audit behind the ruling found
# the valve has NEVER FIRED IN PRODUCTION -- zero [GB_BAND_SHUTOFF] lines
# in the whole v7 run log, because AFTER=50 needs 50 consecutive barren
# iterations of the designated move and that run had only reached ~38. So
# this line is both the behaviour change and the first real exercise of
# the machinery.
#
# The 2026-08-16 replay that raised it to 50 still stands on its facts:
# running the EXACT rule over iterations 5-60 at AFTER=5 switches off 74
# bands above 10 mHz, and 9 of them contain a detectable catalogue source
# the run subsequently FOUND -- SNR 45.7 (band 142, 20.278-20.417 mHz,
# silenced at iteration 18), 35.6 (band 90, iter 14), 34.3 (band 72, iter
# 19), 32.7, 26.9, 24.0, 19.8, 15.0, 12.5 -- with observed
# time-to-first-source of 14-21 iterations. What changed is that replay's
# closing clause, "shutoff is PERMANENT for the process, so there is no
# recovery". It is no longer permanent: the shut-off set and the
# occupancy streaks are now REVIVED on every new F-stat epoch (a refit
# brings a new proposal grid AND an updated noise/foreground profile, so
# a band that was unreachable may now be reachable) and, failing that,
# after GB_RJ_BAND_SHUTOFF_RESET_ITERS iterations. With
# GB_FSTAT_REFIT_EVERY=40 below (was 50 when this note was written), a
# band silenced at iteration 5 is open again by ~40 and then has to
# re-earn its shutoff over a fresh 5-window,
# so those 9 bands get repeated chances instead of one. The cost of the
# short clock is now a DELAY on a genuinely barren-looking band, not a
# permanent loss. Revivals log as [GB_BAND_REVIVE <move>].
# 5 -> 2 (user ruling 2026-09-27). The 5 was sized to outlast the
# progressive leaf-CAP ramp -- a band could be empty because its cap
# had not ramped yet. This schedule has no caps, so an empty band is
# empty on the evidence and two consecutive zero-occupancy iterations
# settle it.
export GB_RJ_BAND_SHUTOFF_ITERS=2
export GB_RJ_BAND_SHUTOFF_SCOPE=search
# Backstop revival (new 2026-08-28): iterations with NO new F-stat epoch
# after which the shut-off set is cleared anyway; 0 disables the trigger.
# Even with no refit the noise model keeps evolving, so a long stretch
# should re-open the question on its own. 100 = 2x the refit cadence
# below, so it only bites if refitting stalls or is turned off.
# 100 -> 10 (user ruling 2026-09-27). This is now a periodic CHECK,
# not an unconditional revival: the shut-off set is reconsidered every
# N iterations and reopened ONLY IF THE NOISE HAS MOVED (see
# GB_RJ_BAND_SHUTOFF_NOISE_TOL, default 1e-3). The check is one
# fingerprint compare, so it can run ten times as often. With the 6mo
# noise PINNED it will never revive, which is the intent.
export GB_RJ_BAND_SHUTOFF_RESET_ITERS=10
# USER RULING 2026-08-28: a shut-off band is frozen "for RJ and fancy
# swaps until it resets". The RJ half is enforced in run_proposal; the
# swap half is this knob (default OFF in code) and it had never been
# exported -- shut-off bands kept being built, scored and swapped in the
# horizontal tempering. ON since 2026-09-11 on the 3mo arm; ported here.
# Safe because shutoff revives on every F-stat epoch (REFIT_EVERY=50) and
# after RESET_ITERS. 6MO TODO (memory note): MEASURE the saving -- read
# temper_cells_filled + run_tempering span vs the unskipped baseline.
export GB_TEMPER_SKIP_SHUTOFF_BANDS=1
# 100 -> 50 (2026-08-18): the refit re-derives the peaks against the LIVE
# residual and the UPDATED foreground/PSD, which is the whole point of
# refitting -- and the foreground converges well inside 20 iterations, so a
# 100-propose cadence spends most of the run on a grid fitted to a
# foreground that no longer exists. ~6% overhead at a 17.7-min fit.
# The peak weighting also flattens to w ~ sqrt(SNR) from epoch 1 onward
# (FSTAT_PEAK_WEIGHT_ALPHA_LATE, default 0.25), so this cadence is also
# when that takes effect.
# 50 -> 30 (2026-09-21, user ruling), alongside the SEARCH live-residual
# refit. The clock counts global-fit ITERATIONS in every stage (2026-09-18
# fix), so this is literally "refit every 30 iterations". The search refit
# no longer restores the walker's GBs before sweeping, so its peak list
# now SHRINKS as sources are found -- each refit should get cheaper, which
# is what pays for the faster cadence. WATCH on the first snapshot:
#   grep '\[peaks\]' globalfit_run.log
# Epochs 0/8/9 were 13513 / 19413 / 19586 peaks, all GB-free. Under the
# change the count should start falling, and stage-B wall time with it
# (epoch 8 was 4054 s).
# 40 -> 10 (user ruling 2026-09-25, SEARCH stages only -- the PE cadence
# below is untouched). The grid goes stale as the search subtracts what it
# finds: every accepted birth changes the residual the peaks were fitted
# against, so a 40-iteration-old grid is aiming births at peaks that have
# already been claimed. That is the same staleness the [GB_ACCEPT rj-split]
# birth-staleness ratio measures (s = d_h/h_h << 1 for a claimed peak).
#
# ⚠ COST: 4x the refits. What is supposed to pay for it is that each refit
# gets CHEAPER as sources are found -- the peak list shrinks. Epochs 0/8/9
# were 13513 / 19413 / 19586 peaks, all GB-free, and epoch 8's stage B was
# 4054 s. If the peak count is NOT falling by the third epoch this cadence
# is 4x a fixed 4000 s bill; check with
#   grep '\[peaks\]' globalfit_run.log
# before committing a long allocation. GB_FSTAT_REFIT_EVERY=40 reverts.
# >>> 10 -> 1 (user ruling 2026-09-25, off job 628): refit EVERY search
# >>> iteration, "because everything is happening so quickly with the
# >>> leaves". Job 628 added 113 then 81 leaves/walker per iteration while
# >>> rj_fstat_search's cold yield fell 135 -> 96 -> 54, i.e. the grid was
# >>> proposing against a residual that had moved substantially since it was
# >>> fitted. 628 never refit once in 7h42m -- it reloaded epoch_0000 every
# >>> iteration.
# >>> ⚠ COST, and it is not small: a full fit was 12.0 min stage A + ~18 min
# >>> stage B + 33 s centres = ~30 min. On an 8307 s iteration that is +22%
# >>> BEFORE the peak list starts shrinking. Dropping rj_replace below pays
# >>> back ~1610 s of it. Watch '[peaks]' and '[stageA]' -- if the peak count
# >>> is not falling by the third epoch this is a fixed ~1800 s/iteration
# >>> bill and should go back to 5 or 10.
export GB_FSTAT_REFIT_EVERY=2      # SEARCH stages: every 2nd iteration
# 1 -> 2 (2026-09-28, Mike: "let's do fstat refit every two
# iterations"; an intermediate 10 was staged and superseded). Measured on job 662
# iteration 5: the refit cost 1129 s of a 5564 s iteration (20%, epochs
# 11-14 at 1127-1134 s each) and bought ~16 cold births per iteration.
# Cold births are exhausted at this point in the stage (rj_warm 25 of
# 253k proposals, rj_fstat 80 of 236k), so a stale peak list costs
# nothing measurable while the refit costs a fifth of the wall clock.
# Revert to 1 if cold birth acceptance climbs back.
# PE-STAGE CADENCE (user ruling 2026-09-24): 250 elapsed iterations.
# Separate from the search value because full_pe is a RANDOM_CHOICE stage --
# the GB move is drawn roughly one iteration in N, so a search-tuned cadence
# there refits far more often in wall-clock terms than the number suggests.
# ⚠ THE CLOCK'S UNITS CHANGED WITH THIS: it counts ELAPSED iterations now,
# not "iterations this branch was proposed in". In a sequential stage
# (gb_search) the two coincide exactly, so 40 above is unchanged in meaning;
# in full_pe the old clock ticked ~1/N as fast and the knob silently meant N
# times more iterations than it said. 150 since 2026-10-12 (Mike: "PE fstat
# fit should be based on a randomly chosen walker residual. And we should do a
# refit on that every 150 iterations"), as in the 6mo / 9mo / 1yr scripts.
# Since 2026-10-12 it is the cadence of rj_fstat_pe_residual (the LIVE-residual
# PE move); rj_fstat_pe (the GB-free galaxy grid) fits once -- see below.
export GB_FSTAT_REFIT_EVERY_PE=150
# TWO PE F-STAT MOVES (user ruling 2026-10-12), as in the 6mo / 9mo / 1yr
# scripts (the full block and Mike's words are in submit_gf_6mo_v9_4gpu.sh):
# "two fstat moves that will split their share of the total moves" --
# rj_fstat_pe fits the GB-free residual ("the whole galaxy down to SNR 6.5")
# ONCE and never refits during full_pe ("The residual fstat will take care of
# low snr adjustments"); rj_fstat_pe_residual fits the live residual of one
# randomly drawn cold walker every GB_FSTAT_REFIT_EVERY_PE overall iterations.
export GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0
export GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE=6.5
export GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=${GB_PE_RJ_FSTAT_RESIDUAL_SPLIT:-0.5}
export FSTAT_PEAKS_PER_BAND=200    # per-sub-band peak cap (code default; explicit)
# STAGE-B STACK CHUNKING -- fixes the 2026-09-21 epoch-9 OOM.
# StackedFStatProposal4D.__init__ corner-averages the ENTIRE K-box 4-D grid
# in ONE allocation; unset, k_chunk = K and nothing is chunked. Epoch 9
# (19586 peaks, the largest of the run) asked for 1,708,159,488 bytes on a
# 91.5 GB resident baseline and died. This feeds mem_budget_mb -> k_chunk:
# SAME grid, SAME result, pure chunking, one-time cost, then cached to
# <epoch>/..._peaks_stacked.npz so resumes skip it. 512 rather than the
# 1yr run's proven 1500 because this baseline is tighter. Lower it further
# if it still OOMs; also consider the MEMPOOL_FREE=1 levers.
export FSTAT_GRID_MEM_MB=512
# ---------------------------------------------------------------------------
# STAGE-B SKY GRID: f0-ADAPTIVE (2026-09-24, new this run)
# ---------------------------------------------------------------------------
# WHAT WAS WRONG. Stage A (the comb) has always sized its sky grid from the
# Doppler ridge, n ~ (f0 Tobs v/c)^2, adaptive over 16..512 nodes -- the log
# line is "[comb] ... sky ADAPTIVE [16..512] over 6 level(s)". Stage B, which
# REFINES stage A's peaks, was pinned at a fixed 8 x 8 = 64 full-sky nodes at
# every frequency. So above ~6 mHz the refinement was 4-8x COARSER on the sky
# than the scan that produced its own input, and stage B is the component the
# birth proposal actually draws from.
#
# Worse, of those 64 nodes only 44 were distinct directions:
# linspace(0, 2*pi, 8) repeats the same longitude at both ends, and every
# longitude collapses to one point at sin_delta = +-1. 31% of the sweep
# re-derived values it already had, and the poles were over-weighted by the
# collapse. Stage B now builds alpha half-open and picks the best rectangular
# factorization of its node budget (measured: near-square, 9x7 / 13x10 /
# 18x14 / 26x20), so 95th-percentile miss-to-nearest-node goes 0.399 ->
# 0.368 rad at the same 64 nodes, and 0.187 rad at 252.
#
# WHY IT MATTERS HERE. The 7.444 and 7.599 mHz misses sit exactly in the
# 5-11 mHz range where this run's peak boxes are dense and the old grid was
# 2-4x too coarse. Those groups are the ones that gain most below.
#
# NO NODE CAP (user ruling 2026-09-24): FSTAT_STAGEB_NSKY_MAX stays 0. The
# limit is a per-group BYTE budget instead, which is the right control
# because what breaks is memory and memory goes as
# n_boxes * n_f0 * n_Mc * n_sky -- not as n_sky. Uncapped-with-no-budget on
# the snapshot-19 peak list asks for 45.8 GB of grid with a single 27 GB
# group at 15-20 mHz (438 boxes but n_Mc = 96), and this run already died
# once on a 1.7 GB allocation against a 91.5 GB resident baseline, which is
# what FSTAT_GRID_MEM_MB above exists to work around.
#
# THE BUDGET SELF-ADJUSTS, which is why it beats a node cap: a group's grid
# shrinks as its peak count falls, so as the search subtracts the high-f
# sources the freed bytes are spent on sky resolution there automatically,
# with no knob change. Epoch 0 is the expensive one.
#
# 2.0 GB holds the largest group at 1.82 GB -- within the 1.71 GB this run
# has already survived -- for 1.80x today's stage-B node count. Projected
# per-group sky at epoch 0 (from the snapshot-19 group census):
#     0.57- 5.14 mHz  11630 boxes  nMc  3   sky  64 ->  63   1.39 GB
#     5.14- 6.11 mHz   2447 boxes  nMc  3   sky  64 -> 130   0.60 GB
#     6.11- 7.27 mHz   1547 boxes  nMc  6   sky  64 -> 130   0.76 GB
#     7.28- 7.57 mHz    393 boxes  nMc  6   sky  64 -> 252   0.38 GB
#     7.57- 9.24 mHz   1754 boxes  nMc 12   sky  64 -> 130   1.73 GB
#     9.32-10.29 mHz    435 boxes  nMc 24   sky  64 -> 252   1.66 GB
#    10.29-11.25 mHz    432 boxes  nMc 24   sky  64 -> 252   1.65 GB
#    11.25-13.17 mHz    514 boxes  nMc 43   sky  64 -> 130   1.82 GB
#    15.36-20.38 mHz    438 boxes  nMc 96   sky  64 ->  63   1.67 GB
# Above 13 mHz the sky stays put because n_Mc = 96 eats the budget there.
# Those sources are few and loud; they are the ones the shrinking peak list
# will free up first.
#
# ⚠ THIS FORCES A REFIT. The stage-B cache stores the grid, so an epoch
# fitted at 8x8 cannot be reused under a different sky. Start from a clean
# gb_fstat_fit/ or the loader reuses the old 64-node grids silently.
#
# WATCH on the first snapshot -- one line per group, and the step-downs:
#   grep '\[stageB\] group' globalfit_run.log
#   grep 'stepping the sky down' globalfit_run.log
#   grep 'VARIES BY GROUP' globalfit_run.log
# ESCAPES, in order of bluntness:
#   FSTAT_STAGEB_GROUP_MAX_GB=1.7  hold every group at today's largest
#   FSTAT_STAGEB_NSKY_MAX=128      hard node cap on top of the budget
#   FSTAT_STAGEB_SKY_ADAPT=0       back to the fixed 8x8 exactly
#   FSTAT_N_ALPHA / FSTAT_N_SINDELTA   setting EITHER also pins the sky
export FSTAT_STAGEB_SKY_ADAPT=1
export FSTAT_STAGEB_NSKY_MIN=64      # never coarser than the old fixed 8x8
export FSTAT_STAGEB_NSKY_MAX=0       # 0 = no node cap (user ruling)
export FSTAT_STAGEB_GROUP_MAX_GB=2.0
# BIRTH-DRAW ALLOCATION (2026-08-16). Peak boxes are weighted w ~ F**alpha,
# and the F-statistic goes like SNR^2 -- so the historical alpha=1 hands an
# SNR-10 source 9x FEWER birth attempts than an SNR-30 one, exactly
# backwards. Measured at iteration 15: 80.3% of the birth mass landed on cap
# cells whose source was ALREADY found (median peak F 125.5) against 7.8% on
# cells holding an unfound detectable source (median F 26.6), while 146 of
# 332 cells with a detectable source held no leaf at all. alpha=0.5 makes
# draws ~ SNR instead of SNR^2 -- still preferring real signal, without
# starving the faint tail by the square. Predicted redistribution: unfound
# cells 7.8% -> ~20% of the birth mass (2.6x).
# NO REFIT NEEDED: the weights are applied when the birth proposal is built
# FROM the cached stage-B grids and are not persisted in them, so a restart
# picks this up against the existing epoch cache.
# alpha=1 restores the previous behaviour bit-identically.
export FSTAT_PEAK_WEIGHT_ALPHA=0.5
# HIERARCHICAL BIRTH DRAW: pick a STRATUM uniformly, then draw within it
# with w ~ F**alpha. K = strata PER SUB-BAND.
#
# K 32 -> 1 (user ruling 2026-08-29): "the fstat proposal should probably
# just group by sub-band". ONE STRATUM PER SUB-BAND. Only one RJ source is
# proposed per sub-band per round (serial-within-band scheduling), so the
# sub-band is the unit a birth actually competes in -- "those are the
# effective rj limits as well". Sub-dividing it over-served the sparse
# parts of a sub-band relative to the dense ones, for a partition that
# corresponds to nothing physical: at K=32 a stratum was 1.7361e-5/32 =
# ~4.2 FD bins, far narrower than a source's ~+-16-bin support, and the
# recorded epoch-0 check (median 3 peaks per occupied cell, max 21) means
# the draw was close to flat-per-peak. At K=1 the anti-starvation motive
# is still met -- mass is equalised across the 1232 occupied SUB-BANDS --
# while the F**alpha preference now does real work among the many peaks
# inside each one, which is what it was for.
#
# NOT A COST KNOB. rj_fstat_centers (61% of an iteration) is a per-row
# solve and does not know this exists; K only sets the birth PROPOSAL
# density.
#
# The strata are derived from band_edges (2026-08-29). They used to be
# np.linspace(be[0], be[-1], nb*K+1) -- identical on this uniform band
# grid, but equal to the sub-bands only by luck and to the STAGGERED cap
# cells never, despite the old comment above calling them cap cells. The
# env default also no longer tracks GB_CAP_DIVISOR: a cap knob must not
# silently re-stratify the RJ birth proposal.
#
# Set to 0 for the historical global w ~ F**alpha mixture.
# Implemented as flat composite weights, w_j = (1/N_occupied) *
# F_j**alpha / sum_stratum(F**alpha), so StackedFStatProposal4D's rvs AND
# logpdf stay mutually exact by construction (both read self.weights) --
# the RJ acceptance ratio is never at the mercy of two implementations.
export FSTAT_PEAK_WEIGHT_CELLS=1
# Slab 5 (user ruling): measured-safe (+-1 layer holds >=1-1e-7 of tone
# energy; 5 = 2x that need) and ~30%% smaller band buffers than the AUTO 7.
# Smoke 2 exonerated the slab as the VGB [GB_CELL_LL] growth cause (growth
# persisted at slab 7).
export GB_WDM_BAND_SLAB_LAYERS=5
# 3-D Milky Way (dist, alpha, sin_delta) joint prior (user ruling: the
# proper density, not the flat placeholder). Chirp-mass basis + the
# astrophysical f0-Mc GMM prior are already the code defaults. NOTE:
# this knob was NOT exercised in the smokes (they ran the uniform
# placeholder); detailed balance holds either way -- births still draw
# dist from the birth container and the prior enters through logp.
export GB_USE_GALAXY_PRIOR=1

# ---- NOISE (psd + galfor) internal repeats: 50 -> 10 (user ruling
#      2026-08-15) ---------------------------------------------------------
# Each PSDMove.propose runs num_prop_repeats internal MCMC repeats, and each
# repeat scores the whole (ntemps x nwalkers) ladder -- one batched build per
# distinct walker. At 50 repeats that is ~660 batched covariance
# build+score calls per move propose, ~1320 per iteration across psd_pe +
# galfor_pe, which is what made the noise block ~43-44 s/iteration. The
# noise model is only 4 (psd) + 5 (galfor) parameters and it converges long
# before 50 repeats, so 10 buys back ~5x of that block for very little
# mixing. WATCH on the first snapshot: the psd/galfor acceptance +
# parameter traces (artifact panels) and whether the noise still tracks the
# injection -- if the chains look under-mixed, 20 is the next notch.
# Combines with today's de-sync work (gated debug guard, sync-free
# sanitization, same-device repack), which cuts the cost of EACH call.
export PSD_NUM_PROP_REPEATS=10
export GALFOR_NUM_PROP_REPEATS=10

# ---- VGB in-model scorer: sig-het ON (user ruling 2026-09-17), same engine
#      and the SAME knobs as the GB branch by construction (VGBSettings reads
#      the shared SIGHET_* / GB_SIGHET_* env names: windowed refs, N_CP,
#      trust gate thresholds, n_sparse_fd). Was pinned to 0 ("accuracy at
#      the loudest-VGB SNRs unverified -- [GB_CELL_LL] growth in smoke 1"),
#      which routed the per-propose VGB information matrices through the
#      CHUNKED engine at 30-45 ms/source: vgb_pe 130 s per call, the
#      noise_vgb_search stage 131 s/iteration vs 24 s on the pre-basis-change
#      legacy run. WATCH on the first snapshot: [GB_CELL_LL vgb_pe]
#      |sampled-actual| must stay ~1e-7 (growth = the sig-het error at loud
#      VGBs), [GB_TRUST] rejection fraction for vgb_pe (the trust gate is
#      what protects the chain), the ll-drift rebuild count, VGB acceptance.
#      VGB_SIGHET_INMODEL=0 restores the exact chunked scorer.
export VGB_SIGHET_INMODEL=${VGB_SIGHET_INMODEL:-1}
# VGB RELAUNCH BLOCK (2026-08-15, user rulings). The VGB likelihood was
# OFF all run (betas=[1e-4] bug) and 36/55 leaves were frozen by the GB
# SNR gate -- both fixed in code (76cd3237); pre-fix VGB samples are
# prior-only. Migration 1 in the header checklist is REQUIRED for the
# VGB_NTEMPS=8 ladder below.
# PARAMETERIZATION: the 6-dim CHIRP basis
# [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio] -- Mc SAMPLED, only
# f0 / alpha / sin_delta left as per-leaf fills.
#
# USER RULING 2026-09-16, superseding BOTH the 08-15 "revert to the old
# VGB parameterization" ruling and this script's v8-parity exact-copy
# pin at 0: "The VGBs should now (with stretch removed) have the exact
# same mechanics as the GBs except f0 and sky fixed" -- Mc AND
# fdot_astro_ratio both sampled -- and "really mirror the GBs as much as
# possible ... VGBs get the ridge-gibbs fiber move too".
#
# WHAT THIS FIXES: on the old 5-dim basis fdot_astro_ratio was a
# COLLAPSED dimension (truth exactly 0 x multiplicative init = zero
# spread, and the affine-invariant stretch cannot create spread it never
# had). The chirp basis un-collapses it: the nonzero catalogue Mc gives
# the multiplicative init real spread, and the zero-truth ratio column
# gets the documented ADDITIVE init exception
# (VGB_RATIO_INIT_WIDTH x GB_FDOT_ASTRO_RATIO_MAX, still scaled by
# VGB_START_FACTOR -- so START_FACTOR=0 below still starts at exact
# truth).
#
# STANDALONE VALIDATION (HM Cnc smoke, 2026-09-16):
#   * installed VGBObservableBasis log_jacobian EXACT vs analytic;
#   * observable + eigen=full with fdot-weight 0 freezes Mc BY
#     CONSTRUCTION (the fiber is exactly flat to the info matrix);
#   * obs + eigen fw=0 + RIDGE-GIBBS was the BEST arm, cold acceptance
#     0.593 -- the ridge move is what unfreezes Mc, so the two knobs
#     ship together;
#   * McRatioDistFiber works VERBATIM on the VGB layout: it resolves
#     dist / Mc / fdot_astro_ratio by NAME and never reads f0.
# The vgb_ridge_gibbs move registers AUTOMATICALLY off this flag --
# recipe.build_vgb_moves gates on the basis carrying those three column
# names, and run_combined_staged.py requests it in gb_search + full_pe
# exactly where gb_ridge_gibbs rides. GB_RIDGE_GIBBS=0 kills both.
#
# *** LOUD CAVEAT -- NOT RESUME-COMPATIBLE. ***
# This takes the vgb chain from ndim 5 to ndim 6. An existing store
# CANNOT be resumed across the flip: run.py refuses it at backend
# construction (before any chain load) with a message naming
# VGB_CHIRP_MASS_BASIS. So this requires EITHER a FRESH store, OR a
# vgb-branch migration with
#   python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \
#       <store.h5> <catalogue_dir>
# run BEFORE the first launch. NEVER flip this mid-store.
# LIVE (user ruling 2026-09-18). Staged off since 2026-09-16; flipped on
# after the 4-GPU run measured VGB in-model acceptance at 0.001-0.004 per
# rung under the 5-column basis, with NO vgb_ridge_gibbs registered (the
# ridge gate reads the column names, and the 5-column basis does not carry
# them). The HM Cnc standalone exonerated the map three ways and ranked
# the arms: 6-col observable+eigen with the RIDGE 0.534 acceptance
# (tau 66.6, the best arm) against 0.1596 without it. This run has been
# getting neither half.
#
# *** BEFORE THE NEXT LAUNCH ON AN EXISTING STORE, MIGRATE IT. ***
#   python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \
#       ${STORE_DIR}/gf_prod_6mo_testing.h5 <catalogue_dir>
# Without that the launch REFUSES to resume (run.py's ndim guard fires at
# backend construction, before any chain load, naming this knob) -- a
# clean stop, not a corrupt run. Set 0 to roll back to the 5-column store.
export VGB_CHIRP_MASS_BASIS=${VGB_CHIRP_MASS_BASIS:-1}
# 8-rung ladder (user ruling 2026-08-15). Resume derives the rung count
# from the STORED band_temps shape, so the migration above MUST be run
# with the matching "8" argument (it recreates every rung-dimensioned
# vgb dataset: temps ladder, counters zeroed, 7 swap pairs).
export VGB_NTEMPS=8
# VGB IN-MODEL PROPOSAL = OBSERVABLE + EIGEN=FULL. SUPERSEDES the 09-15
# morning "=eigen" arming (user ruling, same day, later: "we should be
# sampling the VGBs just like the GBs now (in terms of the basis/proposal
# type not RJ, still fixed model. f0 should be filled. sky location should
# be filled. Everything else is just like the GB setup. We should be
# sampling still in the observed basis for VGBs (just without f0, sky
# coords)"). Both are the CODE defaults now; pinned here anyway because a
# proposal swap is exactly the change a runbook must state.
#
# WHAT IT IS: the SAME composite step the GB branch runs, with the reduced
# map swapped in -- a symmetric draw in z = [lnA, fdot, phi0, cos_iota,
# psi] (f0 / alpha / sin_delta / Mc pinned per leaf; no f_mid because a
# pinned f0 has no shear, no Mc fiber because (dist, r) -> (A, fdot) is
# 2->2), whitened by the per-block information-matrix eigenbasis in z
# (=full: one joint correlated step per repeat, all coordinates moving).
# factors are the map's log-Jacobian difference.
#
# WHY IT SUPERSEDES =eigen -- the r-column verdict. The =eigen draw was in
# the SAMPLING basis, where the ONLY physical quantity fdot_astro_ratio
# drives is fdot = fdot_gr(f0, Mc)(1+r); with Mc a per-leaf FILL, Mc
# occupies the physical fdot output slot through the container key_map, so
# fdot sits OUTSIDE test_inds and r's only scored target is the fddot slot
# -- which McDistFdotAstroQuad emits as exactly f0*0.0. Measured on the
# real stock container: J[:, :, r] == 0 EXACTLY, info_y rank-deficient,
# and the r eigen step set by the prior box (U[-5, 5]) instead of by
# curvature. That is a blind jump in the one coordinate a known-f0 /
# known-sky branch exists to measure, and every VGB run before this one
# had it (commit 2bb484e2 recorded the suspicion; it is now confirmed and
# fixed -- _infomat_phys_inds asks the engine for the live fdot slot).
#
# REQUIRED by the exact-truth flow above: VGB_START_FACTOR=0 gives a
# zero-spread ensemble that a stretch draw could never move; the
# observable step comes from the map + step scales, not the spread.
# COST (accepted, unchanged): VGB_SIGHET_INMODEL=0 above routes the
# matrices through the CHUNKED engine at ~29-46 ms/instance x (55 leaves
# x 8 rungs x 10 walkers ~ 4.4k instances) ~ 2-3.5 min of factor builds
# per vgb propose; validating VGB_SIGHET_INMODEL=1 at the loudest-VGB
# SNRs drops that ~12-19x (~11-17 s) and remains the standing
# optimization.
# FIRST CLUSTER EXPOSURE of the vgb observable path -- watch the first
# vgb_pe [GF_TIMING] wall, any "observable-basis proposal unavailable"
# or "falling back to the stretch proposal" warnings, and the vgb fdot
# spread actually opening up. Escapes: VGB_INMODEL_PROPOSAL=eigen (the
# sampling-basis one-axis draw, now with a live r column) and =stretch
# (the bit-identical legacy).
export VGB_INMODEL_PROPOSAL=${VGB_INMODEL_PROPOSAL:-observable}
# OVERRIDABLE (2026-09-19). Measured, 4 walkers, full_pe:
#   VGB  obs_basis  cold 0/47 (0.0000)  all 0/376 (0.0000)   <- EIGEN=full
#   (the 10-walker arm ran VGB_INMODEL_PROPOSAL=stretch and got
#    cold 182/470 = 0.387, all 1512/3760 = 0.402)
# The VGB in-model move is frozen: 47 cold proposals a propose, zero
# accepted. Same failure as the GB twin (GB_INMODEL_OBSERVABLE_EIGEN=full
# -> obs_basis cold 0.0139 against the 10-walker arm's 0.0864) and worse,
# because VGB fixes f0 and sky, so the information matrix keeps only the
# degenerate directions and a larger fraction of its eigenspectrum is
# ill-conditioned. A "full" draw is JOINT over every axis, so one railed
# direction (PSD-projected eigenvalue -> whitened sigma at
# *_OBSERVABLE_EIGEN_SMAX, default 10) contaminates EVERY proposal --
# unlike "axis", which spends only 1-in-ndim repeats on a bad direction
# and is what the healthy mbh/emri/sobbh EigenAxisMove(mode="axis") uses.
# Empty = the diagonal observable draw, bit-identical fallback.
# NOTE stretch (the 10-walker arm's setting) is NOT reachable at 4 walkers
# on 4 ranks: its red/blue pairing is within the local walker block, which
# needs B >= 2 and even. That is why this run moved VGB to observable.
# DEFAULT CHANGED full -> axis (user ruling 2026-09-19), same reasoning as
# the GB knob above and more urgent: VGB measured cold 0/47 = 0.0000 under
# "full" -- frozen, not slow. VGB fixes f0 and sky, so its information
# matrix keeps only the degenerate directions and a larger share of its
# eigenspectrum is ill-conditioned, which is why the joint draw tips from
# bad to zero here. Escape: VGB_INMODEL_OBSERVABLE_EIGEN= (diagonal draw).
export VGB_INMODEL_OBSERVABLE_EIGEN=${VGB_INMODEL_OBSERVABLE_EIGEN-axis}
# VGB's own step scale, split from GB's 2026-09-19. One shared knob used to
# scale BOTH branches; in the same propose GB was over-accepting at 0.71
# while VGB sat healthy at 0.26-0.43, so GB's correction is the last thing
# VGB wants. 1.0 = unchanged (it is also the code default).
export VGB_INMODEL_OBSERVABLE_JUMP=${VGB_INMODEL_OBSERVABLE_JUMP:-1.0}
echo "[VGB-OBS-EIGEN] VGB_INMODEL_OBSERVABLE_EIGEN='${VGB_INMODEL_OBSERVABLE_EIGEN}' (empty = diagonal draw)"
# GB rung count. 24 is already the code default (stock/erebor/gb.py
# env_default("GB_NTEMPS", 24)) -- pinned here anyway because the rung count
# is the one knob whose failure mode is completely silent: resume derives it
# from the STORED band_temps shape, so a store built at the wrong count runs
# the wrong ladder forever while the script still says 24, and the only hint
# is a single build_gb_moves warning buried in a 200k-line log. Both confined
# probes ran a degenerate [1.0, 1e-4] ladder for days on exactly that. The
# LADDER PREFLIGHT below turns the silent case into a refusal to start.
export GB_NTEMPS=24
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1

# ============================================================================
# 3MO-2 -- NO WARM START.
#
# A 3-month run is the SOURCE of warm-start components, not a consumer: there
# is no shorter run to mine. GB_WARM_START_COMPONENTS is therefore EXPLICITLY
# EMPTY, which is the documented off switch and the only supported one --
# run_combined_staged.py's warm() and warm_pe() key on this variable being
# non-empty, so empty removes rj_warm_search from every search stage and
# rj_warm_pe from full_pe, and build_gb_moves never constructs either stock
# move.
#
# ⚠ EMPTY, NOT DELETED. The 6mo line is `${GB_WARM_START_COMPONENTS-<path>}`,
# and that form substitutes its default when the variable is UNSET -- so
# removing this line would silently re-arm the warm start against a path
# inside this store and then auto-BUILD a mixture from a store that does not
# exist.
#
# WHAT GOES WITH IT. The seed store had exactly two jobs: it supplied the
# warm-start mixture and the psd/galfor start pin. The mixture is gone with
# the line below, and the psd pin is gone by design -- SEARCHING the psd IS
# this run's first stage (3MO-3) -- so GF_SEED_STORE,
# GB_WARM_START_SOURCE_STORE, GB_WARM_START_SOURCE_TOBS, GB_WARM_START_LAST_K,
# GB_WARM_START_FLOOR_EPS, GB_WARM_START_CIRC_IMAGES and both of their
# preflight gates ([V9-SEED] and [WARMSTART]) are DELETED rather than left
# inert. This run reads no previous run's store at all.
# ============================================================================
export GB_WARM_START_COMPONENTS=

# ============================================================================
# 3MO-3 -- THE NOISE EVOLVES THROUGHOUT: A noise_search FIRST STAGE, AND A
#          NOISE CONVERGENCE AFTER EVERY IN-MODEL SLOT IN ALL THREE GB
#          SEARCH STAGES.
# (user ruling 2026-09-26, off the first 3-month run; full design +
#  provenance in docs/superpowers/specs/2026-09-26-3mo-v9-2gpu-design.md)
#
#   noise_search     psd_pe + galfor_pe, one joint max-logL criterion,
#                    against a residual that STILL CONTAINS THE GBs (vgb and
#                    every other source are subtracted at truth)
#   gb_search_1/2/3  the leading joint psd+galfor+vgb rider PLUS a noise
#                    convergence after EVERY in-model convergence proposal
#   full_pe          psd + galfor PE as always
#
# ⚠ THIS REPLACES THE FIRST ATTEMPT, AND THE REASON IS MEASURED, NOT
# THEORETICAL. The first configuration fitted the psd ALONE in stage 0
# (STAGE_NOISE_PSD_ONLY=1) and then FROZE psd+galfor for the whole of
# gb_search_1 and gb_search_2. Two things went wrong at once:
#
#   1. THE WALKERS' PSDs DIVERGED. psd has no start pin here (searching it is
#      the point), so the four walkers begin at four independent PRIOR DRAWS,
#      and a max-logL search converges each to its own optimum. Freezing then
#      made that permanent: four walkers each searching for GBs against their
#      own noise floor, with no mechanism left to reconcile them.
#   2. WHATEVER STAGE 0 CONVERGES TO IS GALAXY-CONTAMINATED. Nothing has
#      subtracted a GB yet, so the entire galaxy is in the residual and a
#      2-parameter psd has nowhere to put it except Soms_d / Sa_a. That was
#      called out as "bounded" in the design; the run says it is not.
#
# Freezing is the defect in both, not the value. Sampling the noise in every
# stage lets it track back DOWN as the GB model fills in, and lets the walkers
# pull back together instead of each carrying its own frozen error.
#
# ⚠ COST, and it is not small: this puts FIVE JointMaxLogLSearch slots (the
# leading rider + four interleaved) into stages 1 and 2 as well as stage 3.
# One such rider measured ~93 s per GB iteration at 3mo (job 473, ~15% of the
# iteration) at NOISE_SEARCH_CHECKS=5. GB_SEARCH_NOISE_CHECKS=1 -- the code
# default, inherited -- is what keeps each near ~2 rounds rather than ~6.
# Watch the [GF_TIMING] share of the noise moves on the first snapshot.
#
# ⚠ ONE NEW KNOB. An unrecognized env var is SILENTLY IGNORED (the
# stock-settings rule in CLAUDE.md), so the preflight below checks that the
# driver actually reads it.
#   GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1  every v9 GB search stage samples the
#                           noise, not just gb_search_3. DEFAULT OFF -- the
#                           6mo v9 run shares this driver and keeps its
#                           fixed-noise stages 1/2 unchanged.
#   STAGE_SKIP_NOISE_VGB=1  drops the noise_vgb_search stage (user ruling
#                           2026-09-26). With VGB_START_FACTOR=0 the VGBs
#                           start at EXACT truth, so a max-logL burn-in has
#                           nothing to find; they begin sampling in
#                           gb_search_1, where vgb_pe rides every stage.
#
# STAGE_NOISE_PSD_ONLY IS DELIBERATELY NOT SET. The knob still exists in the
# driver (default off, tested) and freezes galfor through the noise stage --
# which is exactly what this run just learned not to do. Do not re-arm it here
# without re-reading the two failures above.
# ============================================================================
export GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1
export STAGE_SKIP_NOISE_VGB=1
# Plateau length for the STANDALONE noise stage -- the code default, pinned
# here because this stage sets where the GB search starts from and should not
# be inherited silently. MAXLOGL_TOL above is GLOBAL to JointMaxLogLSearch and
# STAYS AT 20 (user ruling 2026-09-26): against a cold logL of ~1e8 a 20 lnL
# floor is far below anything that matters, and five consecutive flat rounds
# is what makes this a plateau test rather than a wobble detector.
export NOISE_SEARCH_CHECKS=5
# PER-WALKER PLATEAU (user ruling 2026-09-26). Pinned explicitly even though
# it is now the code default, because this run's whole first-attempt failure
# was the criterion it replaces. JointMaxLogLSearch used to test
# state.log_like[0].MAX() -- the BEST walker -- so the stage advanced as soon
# as the luckiest walker stopped climbing. Measured on job 636 row 0: plateau
# declared at best=52494067.95 while the other three cold walkers sat 71,050 /
# 128,080 / 1,600 lnL below it. Now every walker carries its own baseline and
# flat counter, the LAGGARD decides, and a walker that has converged is frozen
# so the remaining rounds go to the ones still climbing.
# ⚠ NOTE WHAT IT DOES *NOT* FIX: the walkers still START at four independent
# prior draws (PSD_START_PARAMS is unset). What makes that recoverable here is
# that noise_search runs with ZERO GB leaves, so all four walkers see the SAME
# residual and therefore the same likelihood surface -- run each to its own
# plateau and they land on the same optimum instead of 128k lnL apart. If a
# snapshot still shows them spread, pin PSD_START_PARAMS (see above).
# MAXLOGL_PER_WALKER=0 restores the old best-walker rule bit-identically;
# MAXLOGL_FREEZE_CONVERGED=0 keeps per-walker but stops freezing.
export MAXLOGL_PER_WALKER=1
# ⚠ FREEZE OFF (2026-09-26, after it killed the first relaunch).
# Shutting a converged walker off means restoring its rows, and a
# branch with a tempered ModuleSubState is stored TWICE -- the main
# engine state and state.sub_states[name]. The first version restored
# only the main one and rank 1 died on the next propose with
#   ValueError: [psd] cold-chain coords mismatch between the main
#   state and its sub-state (1 of 2 alive leaves differ)
# The restore now covers sub-states too, but that path has never run
# on a real tempered state, so it stays off here. The PER-WALKER
# plateau above is the substantive fix and is unaffected by this.
export MAXLOGL_FREEZE_CONVERGED=0
#
# ⚠ PSD_START_PARAMS MUST STAY UNSET. run_combined_staged.py SKIPS the noise
# stages when EVERY sampled noise branch is pinned, so setting it would delete
# this run's first stage. A galfor-only pin leaves _noise_pinned False, which
# is exactly the configuration wanted here. STAGE_FORCE_NOISE_SEARCH=1 is the
# escape if a psd start value is ever wanted as well.
# ⚠ AND NOTE WHAT IT WOULD ALSO FIX: an explicit PSD_START_PARAMS is the one
# thing that would start all four walkers at the SAME psd instead of four
# prior draws. It is left unset because the noise now keeps sampling, which
# addresses the divergence from the other end -- but if the walkers still
# spread, pin it (PHYSICAL/linear Soms_d, Sa_a) together with
# STAGE_FORCE_NOISE_SEARCH=1 so the stage still runs.
#
# ---- PREFLIGHT: the knobs above must actually be CONSUMED ------------------
# A preflight proves the NAME resolves, not that any code path reads it. Six
# separate defects this month had that shape. Both knobs are read by the
# DRIVER (not the package), with literal names, so a grep of the driver source
# is the honest check -- and it is what catches a stale checkout.
_DRIVER=scripts/fstat_proposal/run_combined_staged.py
for _k in GB_SEARCH_SAMPLE_NOISE_ALL_STAGES STAGE_SKIP_NOISE_VGB; do
  if ! grep -q "${_k}" "${_DRIVER}"; then
    echo "[3MO-PREFLIGHT] REFUSING: ${_k} is not read anywhere in"
    echo "[3MO-PREFLIGHT]   ${_DRIVER}"
    echo "[3MO-PREFLIGHT] The exported value would be SILENTLY IGNORED and"
    echo "[3MO-PREFLIGHT] this run would FREEZE the noise through gb_search_1"
    echo "[3MO-PREFLIGHT] and gb_search_2 -- plausible output, and exactly the"
    echo "[3MO-PREFLIGHT] failure the first 3-month attempt hit."
    echo "[3MO-PREFLIGHT] git pull to at or after the all-stage-noise commit."
    exit 2
  fi
done
unset _DRIVER _k
#
# ---- THE FOREGROUND START POINT -------------------------------------------
# ⚠ IT IS A START, NOT A HELD REFERENCE, as of the 2026-09-26 revision: the
# galfor branch samples from iteration 0. The inflation below therefore costs
# far less than it did in the first attempt -- it sets where the chain BEGINS
# rather than a floor the whole search is judged against -- but it is kept,
# because with an empty GB model the foreground climbs to absorb the galaxy
# anyway, and starting nearer that point shortens the burn-in.
# PHYSICAL / LINEAR, order (amp, fk, alpha, f_1, f_2). run.py converts into
# whatever basis this run samples in (galfor log10 on amp/fk/f_1/f_2, alpha
# LINEAR), which is why the interchange format is physical -- this file
# exports GALFOR_LOG_SAMPLING=1 above.
#
# BASE = the 90-day ADD-BACK FIXED POINT, galfor_6mo_figs/
# addback_fixedpoint_90d.json, history[-1].prior_box_refit, amplitude
# modulation-corrected by <M_XX> = 0.8080513:
#
#     amp 1.436180605904e-44   fk 2.533915614978e-03   alpha 5.0
#     f_1 1.0e-02              f_2 1.405721657329e-03
#
# The GALFOR brick has a TWO-YEAR resolvable set subtracted (8420 binaries at
# SNR>7 / Tobs=1.999 yr), so the binaries that no longer clear SNR 7 at 90 d
# are regenerated and added back -- and that is ITERATED to a fixed point,
# because the added power demotes more binaries. Converged at iteration 6
# (n_drop 5227 -> 6792, delta +7, rms 0.0755, fmax 3e-3).
# ⚠ fit_galfor's DEFAULT fmax 8e-3 RAILS at every Tobs -- three sessions
# confirmed that independently; 3e-3 is the working value.
# ⚠ The 90 d window sits on a DIM stretch of the annual sweep (<M_XX> < 1,
# unlike 180 d's 1.1031), so the modulation correction RAISES the amplitude
# here. This run puts M(t) back on top (GALFOR_MODULATION_PATH +
# GALFOR_MODULATION_T0=data, exported above), so using the raw 90 d Welch fit
# 1.160508e-44 would understate the foreground by 19%.
#
# INFLATION = amp x1.25, fk x1.10 (user ruling 2026-09-26: start CONSERVATIVE,
# so the reference encapsulates other galaxy models). alpha and f_2 untouched.
#
#     amp 1.436180605904e-44 x 1.25 -> 1.795225757380e-44
#     fk  2.533915614978e-03 x 1.10 -> 2.787307176476e-03   (2.53 -> 2.79 mHz)
#
# WHY A KNEE TERM AT ALL, AND WHY 1.10. Read as the factor every SNR threshold
# is effectively multiplied by, sqrt(S_total(inflated)/S_total(fit)) on the A
# channel:
#
#       f        amp x1.25   +fk x1.06   +fk x1.10   +fk x1.20
#       1 mHz      1.10        1.11        1.12        1.13
#       2 mHz      1.11        1.15        1.17        1.21
#       3.5 mHz    1.10        1.19        1.25        1.40
#       5 mHz      1.04        1.09        1.13        1.24
#       6 mHz      1.01        1.02        1.03        1.06
#       8 mHz      1.00        1.00        1.00        1.00
#
#  * AMPLITUDE inflation lifts the 0.6-2 mHz plateau, which is the part of the
#    fit that is solidly measured. The KNEE term acts almost entirely in
#    2.5-5.5 mHz, which is where the confusion edge actually moves between
#    galaxy models -- so it buys conservatism where the uncertainty is.
#  * x1.06 is the SELF-CONSISTENT partner of x1.25: binary number density goes
#    as dN/df ~ f^(-11/3), so scaling the galaxy by G moves the confusion edge
#    as G^(3/11) = 1.063 at G = 1.25. x1.10 is deliberately just beyond it,
#    because galaxy models differ in SHAPE and not only in normalization.
#  * x1.20 IS THE LINE NOT TO CROSS. It reproduces k ~ 2.0 at 3.5 mHz, which
#    is the measured 6mo pathology (k = 1.85-2.00 over 3-4 mHz) that made the
#    stock SNR-8 F-stat floor an effective TRUE-SNR threshold of 11.3 and hid
#    693 findable binaries, 506 of them in 3-5 mHz.
#
# THE COST, stated plainly: stage 1's SNR-8 F-stat floor acts like ~10 at
# 3.5 mHz and ~9 below 2 mHz; stage 2's 6.25 acts like ~7.8 and ~7.0. That
# cuts against the standing "err toward grabbing, not missing" search policy
# and is accepted deliberately -- it is bounded, it is confined to the
# confusion band, and gb_search_3 RELEASES the foreground, so it shapes three
# stages and then stops mattering. FALLBACK if the 3-5 mHz yield reads badly:
# fk x1.06, i.e. 2.685950551877e-03, with no other change.
#
# ⚠ f_1 IS NUDGED OFF THE PRIOR RAIL, 1.0e-2 -> 9.9e-3. The v9 launch audit
# flagged the exact-boundary value. It is physically irrelevant here: with
# fk = 2.79 mHz and f_2 = 1.41 mHz the tanh has already cut the spectrum by
# ~1e-3 before the exponential starts to bite, so the in-band curve moves by
# well under a percent. f_1 is NOT a measurement at 3 months -- the
# unconstrained fit wants ~0.365 Hz (i.e. exp(-(f/f_1)^alpha) == 1 and the
# tanh doing all the cutting), it sat at 0.14 .. 0.45 Hz across all six
# iterations at flat rms, and refitting inside the prior box changed the rms
# by nothing at all (0.07547 constrained vs 0.07547 free). That is what an
# unidentified parameter looks like. Expect the branch to wander along the
# exp-vs-tanh degeneracy once gb_search_3 releases it; not a pathology.
# ⚠ alpha = 5.0 is INTERIOR only because GALFOR_ALPHA_MAX=20.0 is exported
# above. Do not drop that line.
#
# ⚠ WHAT STAGE 0 IS ACTUALLY FITTING. The residual still contains the WHOLE
# galaxy -- nothing has subtracted a GB yet -- while this reference models
# only the UNRESOLVED part, so the 2-parameter psd will sit high absorbing the
# resolvable set. That is inherent to the design and is bounded by the psd's
# rigidity (two levels with fixed spectral shapes, not a free spectrum); the
# inflation absorbs part of it rather than leaving all of it to Sa_a.
# gb_search_3 is what fixes it properly.
export GALFOR_START_PARAMS=${GALFOR_START_PARAMS:-1.795225757380e-44,2.787307176476e-03,5.0,9.9e-03,1.405721657329e-03}
echo "[3MO-NOISE] stage 0 = joint psd+galfor max-logL search; the noise then"
echo "[3MO-NOISE]   KEEPS SAMPLING in every GB search stage (all_stages=${GB_SEARCH_SAMPLE_NOISE_ALL_STAGES} skip_vgb_stage=${STAGE_SKIP_NOISE_VGB} checks=${NOISE_SEARCH_CHECKS} tol=${MAXLOGL_TOL})"
echo "[3MO-NOISE] galfor STARTS at the inflated 3mo reference (amp x1.25, fk x1.10), free from iteration 0:"
echo "[3MO-NOISE]   GALFOR_START_PARAMS=${GALFOR_START_PARAMS}"
echo "[3MO-NOISE] PSD_START_PARAMS=${PSD_START_PARAMS:-<unset: SEARCHED from a prior draw in stage 0>}"
echo "[3MO-WARM] GB_WARM_START_COMPONENTS='${GB_WARM_START_COMPONENTS}' (empty = NO warm start; rj_warm_search / rj_warm_pe absent from every stage)"

# ============================================================================
# V9-11 -- THE THREE-STAGE GB SEARCH (user spec 2026-09-24).
#
# One gb_search stage becomes gb_search_1 / _2 / _3, each running the cycle
#
#   rj_warm_search -> in_model -> rj_fstat_search -> in_model_fstat
#     -> rj_replace -> in_model_replace -> rj_prior_removal
#
# The cycle ENDS on the removal judge (user amendment 2026-09-24): every
# birth and every swap gets a full in-model refinement pass before it is
# judged for death, so nothing is ever deleted at coordinates it has not
# been given a chance to walk away from. Each in-model slot is named for
# the RJ move it polishes; watch [GF_TIMING] move=in_model* for their cost.
#
# and differing only in noise (fixed / fixed / sampled), phase max (on / off /
# off), opt SNR (8 / 5 / 5) and F-stat peak floor (8 / 6.25 / 6.25). The three
# knob values live in V9_SEARCH_STAGE_PROFILES in run_combined_staged.py, are
# applied to the shared move objects at stage entry, and every mutation
# prints a [V9-STAGE] line. =0 restores the single legacy stage.
export STAGE_V9_SEARCH=1
# A stage ENDS when the cold-chain nleaves plateau AND every OCCUPIED
# (walker, band) pair has shut off (GB_SEARCH_BAND_SHUTOFF_PER_WALKER, armed
# below). Empty pairs can never shut off -- "all pairs" is unreachable -- so
# the plateau gate is composed IN rather than replaced, and covers them.
# Watch: "[V9-STAGE gb_search_1] nleaves plateau reached but N occupied
# (walker, band) pair(s) have NOT shut off -- holding the stage open."
#
# ---- gb_search_seed: DISABLED AT 3 MONTHS (user ruling 2026-09-26) --------
# The seed stage exists to birth HARD off the warm-start posterior for a fixed
# 5 iterations before any F-stat grid is fitted. With no warm start (3MO-2)
# its move list collapses to in_model on an empty model, so it is switched off
# entirely and gb_search_1 leads the recipe. 0 is the documented off switch
# and restores the pre-2026-09-26 recipe exactly.
export GB_SEARCH_SEED_ITERS=0
# GB_SEARCH_3_WARM_EVERY IS DELETED WITH IT. It cadences rj_warm_search inside
# gb_search_3, and there is no warm move to cadence -- leaving it exported
# would be a knob that resolves and reaches nothing, which is the recurring
# defect shape this codebase has hit six times this month.

# ---- MID-ITERATION CHECKPOINTS (user ask 2026-09-24: "make sure there are
# checkpoints between all the GB moves for resume") ------------------------
# The HOOKS already sit at EVERY sub-move boundary: GFCombineMove writes one
# after each move it proposes, on both its sequential and weighted-cycle
# paths, and the v9 search stages take the sequential one. So across a
# search iteration there is a coherent resume point after each of
#   rj_warm_search / in_model / rj_fstat_search / in_model_fstat /
#   rj_replace / in_model_replace / rj_prior_removal
# plus one BETWEEN the two internal passes of rj_replace (that move is a
# single sub-move running two complete sweeps, so the combine's own boundary
# would have left the second pass unprotected).
#
# ⚠ WHICH hooks actually WRITE is the throttle's decision, not the hook's.
# MIDIT_CHECKPOINT_MIN_INTERVAL is the minimum seconds between writes, so at
# the 600 s default a ~2500-4000 s v9 iteration checkpoints at roughly every
# other boundary, and the exposure on a preemption is bounded by the LONGEST
# gap between two writes -- which for v9 is a whole in-model slot, since one
# of those can run up to GB_INMODEL_GROUP_MAX_PASSES x
# GB_NUM_REPEAT_PROPOSALS per source. Lower this if the [MIDIT_CKPT] write
# cost (it logs MB and seconds) turns out to be small against that exposure;
# 0 writes at every boundary. Both pinned explicitly here rather than left
# to the code defaults, because on a SPOT partition this is the knob that
# decides how much a preemption costs.
# Overridable from the command line since 2026-10-01 (carried from the 6mo
# script, where the reasoning lives): `MIDIT_CHECKPOINT=0 ./submit...` for a
# legged + ratcheted search; default 1 otherwise.
export MIDIT_CHECKPOINT=${MIDIT_CHECKPOINT:-1}
export MIDIT_CHECKPOINT_MIN_INTERVAL=${MIDIT_CHECKPOINT_MIN_INTERVAL:-600}
if [ "${MIDIT_CHECKPOINT}" = "1" ]; then
  echo "[V9-CKPT] mid-iteration checkpoints ON, min interval ${MIDIT_CHECKPOINT_MIN_INTERVAL}s"
else
  echo "[V9-CKPT] mid-iteration checkpoints OFF (MIDIT_CHECKPOINT=${MIDIT_CHECKPOINT}); under GB_SEARCH_LEGS a row lands after every leg instead"
fi
echo "[V9-CKPT] watch: [MIDIT_CKPT] wrote ... at stored iteration N (boundary '...')"

# ============================================================================
# 3MO-4 -- DATA + SOURCE BRANCHES (user ruling 2026-09-26).
#
# COMBINED = mojito's PRE-SUMMED L1 stream (${MOJITO_DATA_PATH}/data/
# COMBINED/L1/): it establishes the data AND the orbits, and it already
# contains EVERYTHING -- instrument noise, the full GB galaxy (resolved +
# confusion), VGBs, and ALL sources of every class. The per-class entries
# after it contribute CATALOGUES ONLY (VGB seeding, nleaves sizing, F-stat
# overlays); their L1 bricks are NOT read, so per-id brick presence stops
# mattering. The loader REFUSES COMBINED+NOISE and COMBINED+GALFOR
# (double-count guards) -- do not add them. The NOISE brick must still EXIST
# on disk: UNEQUAL_ARM=1 reads its /ltts delay table and the psd start
# estimates read its noise_estimates, both resolved by PATH (NOISE_FILE /
# data/INSTRUMENT/L1/NOISE_*), independent of this list.
#
# ⚠ MBHB_IDS / EMRI_IDS / SOBHB_IDS ARE UNSET, AND UNSET IS NOT SET-EMPTY.
# Two readers disagree about what "empty" means:
#   * source_runtime.default_source_ids seeds the SETTINGS and tests
#     `os.environ.get(f"{cls}_IDS") is not None`, so a set-but-empty value
#     REPLACES the class default {"MBHB": [], "EMRI": [1], "SOBHB": []} with
#     three empty lists -- and `import lisatools.globalfit.stock.erebor` then
#     FAILS OUTRIGHT from an unrelated variant's constructor
#     ("mojito_source_ids must inject at least 1 source total");
#   * run_combined_staged._source_ids_from_env ARMS the branches and reads
#     os.environ.get(env, ""), to which unset and set-empty are identical.
# So there are no export lines here at all. Do NOT "tidy" them back in as
# empties -- the run dies before it builds anything.
#
# ⚠ STAGE_SKIP_SOURCE_SEARCH IS ALSO ABSENT, deliberately. With no armed
# source branch run_combined_staged RAISES "STAGE_SKIP_SOURCE_SEARCH=1 but no
# source branches are armed -- there is no source_search stage to skip."
#
# ⚠ MODELING CONSEQUENCE. The MBH, EMRI and SOBBH signals ARE in the COMBINED
# data and nothing samples them, so they sit in the residual as unmodeled
# content. That was CHECKED at this Tobs rather than assumed (2026-09-26
# forensics; numbers and method in the design spec):
#
#   * MBHB id18 merges at 91.97 d -- 1.97 d AFTER this window closes -- and
#     contributes NOTHING in band. The reason is its mass, not its timing:
#     SSB chirp mass 6.855e6 Msun (M_tot 1.722e7), so f_ISCO = 2.55e-4 Hz and
#     the source merges AT the analysis floor. By day 90 its (2,2) has reached
#     only 8.8e-5 Hz and the 2nd harmonic 1.8e-4 Hz, both under MIN_FREQ.
#     Measured directly off the brick, in-band SNR over [0, 90 d] is 1e-8 of
#     the same quantity over [0, 120 d].
#   * MBHB id5 (Mc 3.05e5, t_c 104.74 d) is the one that DOES reach the band:
#     it crosses 2.5e-4 Hz at t = 82.9 d and gets to 2.9e-4 Hz by the window
#     edge, depositing SNR <= 63 -- ALL of it in 0.25-0.29 mHz, i.e. BELOW
#     GB_MIN_FREQ=5.5e-4, where its GB-band SNR is exactly 0. THE GB SEARCH IS
#     UNTOUCHED; the exposure is to the psd fit, whose lowest layers carry
#     disproportionate weight (the MIN_FREQ forensics measured layer 1 alone
#     at 43% of the fit's entire chi^2 budget). ACCEPTED for now (user ruling
#     2026-09-26). Escapes if it ever matters: MBHB_IDS=5 TOGETHER WITH
#     MBH_MERGER_TIME_BUFFER=20 (prepare_mbh_branch keeps an MBH only if
#     t_merge < observation_end + buffer, and the default buffer is 2 days, so
#     104.74 d against a 90 d window is dropped), or MIN_FREQ=4.2e-4 -- which
#     costs the four lowest VGBs and is exactly what moving MIN_FREQ to
#     2.5e-4 existed to avoid.
#   * EMRIs / SOBHBs are ~4-5 each at this baseline (short-window census: EMRI
#     id1 2.47 at 1 mo -> 5.67 at 4 mo; SOBHB id0/id1 2.31/2.68 -> 6.96/6.73).
#     Note SOBHBs live near ~10 mHz, i.e. INSIDE the GB band -- watch for GB
#     leaves tracking a chirping source there.
#
# Explicitly-set SOURCE_TYPES always wins over this line.
# ============================================================================
export SOURCE_TYPES=${SOURCE_TYPES-COMBINED,GB,VGB}
echo "[DATA] SOURCE_TYPES=${SOURCE_TYPES} (COMBINED = pre-summed stream; GB/VGB after it are catalogue-only; NO source branch is armed)"
# Ladders / repeats / swap cadence / start scatter: the probe's latest
# rulings (submit_gf_6mo_sources_probe.sh, 2026-08-26..28 -- see its
# comment blocks for the measured cost arithmetic).
# ######################################################################### #
# ## INERT AT 3 MONTHS -- NO SOURCE BRANCH IS ARMED.                     ## #
# ##                                                                     ## #
# ## Everything from here down to the EIGEN INNER MOVE block configures   ## #
# ## the mbh / emri / sobbh branches, and MBHB_IDS / EMRI_IDS /           ## #
# ## SOBHB_IDS are UNSET (see the 3MO-4 block above), so none of it is    ## #
# ## read by anything. It is KEPT rather than deleted for two reasons:    ## #
# ## it holds the 3mo-vs-6mo diff down to the knobs that actually         ## #
# ## differ, and it keeps the values correct for a future 3-month run     ## #
# ## that does arm them. Arming a branch here is ONE id list away.        ## #
# ######################################################################### #
export MBH_NTEMPS=2
export EMRI_NTEMPS=2
# EMRI ladder cap, as in the 6mo / 9mo / 1yr scripts (user ruling 2026-10-13:
# "use max temperature for an snr 100 source for the EMRIs", T_max = 100).
export EMRI_LADDER_TMAX=${EMRI_LADDER_TMAX:-100}
# 12 -> 8 (user ruling 2026-09-16, with repeats 25 -> 20: sobbh cost
# trim now that the single-call fix landed). NOTE: SOBHB's ladder lives
# in the stored PerLeafLadderState (betas_all per leaf), so on a RESUME
# the stored 12-rung ladder wins and this value is inert -- 8 rungs
# take effect at the next FRESH branch init (the deliberate VGB-chirp
# restart). The persisted eigen sidecar's sobbh tables are
# per-(temp,walker) shaped and will rebuild once on the shape change.
# REVERTED 8 -> 12 (2026-09-16 snapshot 7 decode): the resume-safety
# claim was WRONG -- on resume the LIVE ladder followed the config (8
# betas) while the store/sidecar stayed 12-shaped, triggering mass
# eigen-table rebuilds every launch (231 s leaf windows, 772 rows/call,
# 80/80 non-positive infomats, mbh/emri walls 3-5x, iteration ~80 min).
# Set 8 ONLY at the deliberate fresh restart (with the VGB chirp
# migration), never on a resumed 12-rung store.
# (2026-09-17) Overridable, and the STORED rung count now WINS on resume
# for mbh/emri/sobbh (recipe.resume_ladder_wins, same rule as gb/vgb): a
# store born at 8 rungs (d3c0d6ee era, 09-16 12:52 -> 21:55) resumed under
# 12 built a 12-rung move against an 8-rung state and died in the SOBBH
# per-walker eigen sweep ("cannot reshape array of size 88 into shape
# (12,newaxis)"). The knob is reported-and-ignored on such a resume; a
# fresh store honours it. BACK TO 8 (user ruling 2026-09-17 evening): the
# gf_prod_6mo_v8_4gpu store IS an 8-rung store, and a resumed 12-rung
# store (the _mr copy, the original run) keeps its 12 via the store-wins
# rule with a WARNING, so 8 here is safe for every store.
export SOBBH_NTEMPS=${SOBBH_NTEMPS:-8}
# ONE information matrix per leaf at the max-lnL cold walker (like MBH/EMRI)
# instead of one per (temperature, walker) (user ruling 2026-09-16): the
# per-walker stash is keyed by the walker axis, so every resume under a
# different walker-block width (2 -> 4 GPUs) discarded it and rebuilt 60-120
# matrices per leaf (~2.6 min/leaf/rank); walker_max tables are
# layout-independent and persist across resumes. Watch the per-rung SOBBH
# acceptance -- hot rungs now propose with the best walker's curvature.
# THE ONLY SOBBH_EIGEN_SCOPE EXPORT IN THIS FILE: a second
# `export SOBBH_EIGEN_SCOPE=per_walker` further down (the 2026-09-08
# TABLE SCOPE block) silently overrode this line until 2026-09-17, so the
# ruling never took effect. Escape: SOBBH_EIGEN_SCOPE=per_walker in the env.
export SOBBH_EIGEN_SCOPE=${SOBBH_EIGEN_SCOPE:-walker_max}
export MBH_NUM_PROP_REPEATS=2
export EMRI_NUM_PROP_REPEATS=2
# 25 -> 20 (user ruling 2026-09-16) -> 10 (user ruling 2026-09-18).
# THE lever on the dominant per-iteration cost. [SOBBH_LL_TIMING] on the
# 4-GPU run measured the chunked-het scorer at a FLAT 1.73 s per CALL,
# independent of how many rows the call carries (windows of 138 and 404
# rows both cost 1.72 s/call; 100% of it in the kernel-launch span at
# ~605 ms per WDM layer group, ~2.8 groups/call). Calls come from repeats,
# NOT from walkers or rungs, so the walker-block width does not touch this
# cost and repeats are the only knob that moves it: 22 calls/leaf (20
# repeats + 2) x 6 leaves x 1.73 s = 228 s, plus 12 residual fills at
# 1.70 s = 248 s, i.e. 43% of a 9.2-min iteration. At 10 repeats that is
# ~130 s and the iteration drops to ~7.2 min. The underlying defect is the
# 605 ms group launch: the in-code reference for this configuration is
# 2.78 ms/row (sobbhspecialmove.py, job-373 note), ~58x away.
export SOBBH_NUM_PROP_REPEATS=${SOBBH_NUM_PROP_REPEATS:-10}
export MBH_PERMUTE_EVERY=10
export EMRI_PERMUTE_EVERY=10
export SOBBH_PERMUTE_EVERY=10
# EXACT-TRUTH STARTS (user ruling 2026-09-14 late): "start all the emri, mbh,
# vgb, sobhb at the true points exactly ... they switched to the eigen
# proposal, we do not [need] a spread like with the stretch proposal."
# *_START_FACTOR=0 is exact truth (the run.py seeders are x*(1+f*randn), no
# floor; VGB's additive ratio jitter also scales by its factor, so 0 = exact
# truth-null there too).
#
# ⚠ STAGE_SKIP_SOURCE_SEARCH IS NOT EXPORTED HERE -- see the 3MO-4 block
# above. At 6 months it skips the source_search stage; at 3 months no source
# branch is armed and the flag RAISES at composition.
#
# VGB_START_FACTOR is the one that still does work here. The VGBs ARE sampled,
# and starting them at exact truth is what makes "all sources removed except
# the GBs" true for the stage-0 psd search: their templates enter the residual
# at setup_acs from the first pass onward, they sit subtracted and frozen
# through noise_search (there is no noise_vgb_search -- 3MO-3), and they begin
# PE-sampling in gb_search_1, where vgb_pe rides the fixed-noise stages.
# The other three are inert (their branches are unarmed) and are kept for the
# same reason as the block above.
export MBH_START_FACTOR=0.0
export EMRI_START_FACTOR=0.0
export SOBBH_START_FACTOR=0.0
export VGB_START_FACTOR=0.0
# SOBBH chunked scoring width: converged value (S3 ruling; the 11-h-layer
# stress result -- cheap insurance on this 1-h-layer production grid too).
export SOBBH_M_BAND_HALF_WIDTH=3
# Thin the slow-path A/B re-score (per-row TDI-on-the-fly x all rungs);
# it still fails loudly past tolerance when it fires.
export SOBBH_CHECK_LL_EVERY=30

# ---- EIGEN INNER MOVE (2026-09-05/08 work; code defaults, pinned) ----------
# The addremove branches' in-model proposal is the eryn EigenAxisMove: a
# one-axis jump along eigenvectors of each source's own information matrix
# (per-axis curvature widths, prior-capped, factors=0 -- tables frozen
# between refreshes). Replaces the stretch as the DEFAULT inner move; the
# laptop A/B on the real chunked SOBBH kernel measured acceptance 0.544 vs
# stretch 0.156 at equal per-step cost. {BRANCH}_INNER_MOVE_KIND=stretch
# is the per-branch escape back to the legacy stretch, exactly.
# GUARDED (2026-09-16) for the scoring-regression A/B: production
# measures 61 ms/row vs the kernel's 2.78 ms/row job-373 reference.
# One-window discriminator: `SOBBH_INNER_MOVE_KIND=stretch sbatch <this>`
# -- leaf time collapsing toward ~15 s indicts the eigen inner-move path;
# staying ~190 s indicts the scoring seam. Default stays eigen (it beat
# stretch 0.544 vs 0.156 acceptance in the laptop A/B).
export SOBBH_INNER_MOVE_KIND=${SOBBH_INNER_MOVE_KIND:-eigen}
export MBH_INNER_MOVE_KIND=eigen
export EMRI_INNER_MOVE_KIND=eigen
# TABLE SCOPE. SOBBH is set ABOVE (SOBBH_EIGEN_SCOPE, walker_max per the
# 2026-09-16 ruling; the 2026-09-08 per-walker ruling it superseded cost
# 12 rungs x 10 walkers = 120 points x ~245 rows ~ 29k batched likelihood
# rows ~ 1.4 min per LEAF REFRESH at 2.78 ms/row, and the per-walker stash
# is discarded on every walker-block change). SOBBH_EIGEN_REFRESH stretches
# the refresh cadence either way. Do NOT re-add an export here: a second
# export silently overrode the ruling above until 2026-09-17.
export SOBBH_EIGEN_REFRESH=10
# MBH/EMRI: ONE table per leaf, built at the max-lnL COLD walker (their
# likelihood rows are per-row dense, ~1.4 / ~1.0 s). A refresh is
# (nwalkers selection + ~245 build) rows ~ 6.4 min/leaf (MBH) /
# ~4.6 min/leaf (EMRI) -- paid on the FIRST visit per leaf (~26 min MBH,
# ~37 min EMRI, one-time) and then every 100th visit. Escape:
# {BRANCH}_INNER_MOVE_KIND=stretch above.
export MBH_EIGEN_SCOPE=walker_max
export EMRI_EIGEN_SCOPE=walker_max
export MBH_EIGEN_REFRESH=100
export EMRI_EIGEN_REFRESH=100
# WATCH ON FIRST LAUNCH: any "[eigen_refresh] ... fallback" WARNING means
# a table build failed and that leaf is sampling on identity/1%-width
# tables (correct but slow -- MH corrects the shape); a steady stream of
# them means arm the stretch escape and file the traceback.

# EMRI MODE-SELECTION THRESHOLD (user ruling 2026-09-14). FEW's kwarg is
# ``mode_selection_threshold``; ``eps`` is its FEW 1.x name and what we still
# call the knob. The EFFECTIVE value was 1e-5 (the generator default), NOT the
# 1e-2 pinned in lisatools/sources/emri/response.py: that pin is a mode
# SELECTOR CONSTRUCTOR value, and FEW's _generate_waveform hands its own
# call-time default (1e-5, few/waveform/base.py:143) to the selector
# unconditionally, so the constructor value never applied. EMRI_EPS is wired
# at CALL time, through the EMRI wave wrap's runtime_kwargs -- the ONE wrap
# both the engine-side generator (template + residual) and the move-side
# generator resolve through, so it is on every EMRI waveform the run builds.
# It is deliberately NOT stamped into the branch waveform_kwargs: for EMRI
# those double as the move's LIKELIHOOD kwargs, and inner_product has no such
# parameter -- doing so crashed the 6mo run on 2026-09-15 (fixed same day).
# 1e-5 -> 1e-3 LOOSENS the cut: FEW keeps modes until the cumulative mode
# SNR^2 is within (1-threshold)^2 of the total (few/utils/modeselector.py:426),
# so fewer harmonics are kept = direct speedup on the dominant EMRI cost
# (EMRI likelihood rows measure ~1 s each at 6mo). FEW documents the trade as
# "Increasing this value removes modes from consideration and can have a
# considerable affect on the speed of the waveform, albeit at the cost of some
# accuracy (usually an acceptable loss)" (few/utils/modeselector.py class
# docstring), and its tutorial measures the two bracketing endpoints: 455
# modes at 1e-5 vs 25 modes at 1e-2, mismatch 1.5e-3 between them
# (examples/FastEMRIWaveforms_tutorial.ipynb, stored output). 1e-3 sits inside
# that bracket. The tiered-accuracy policy tolerates it, and the campaign IS
# the test.
# RESUME-SAFE: a template/proposal accuracy knob only -- no data, grid, or
# store-layout change, and nothing stamps waveform_kwargs into a store
# identity (the only semantic resume guard is the noise-model identity in
# globalfit/hdfbackend.py). Corollary: a resumed store will NOT flag the
# change, so it has to be recorded with the run.
# Unset or empty reverts to the generator default (1e-5).
export EMRI_EPS=1e-3

# ============================================================================
# FRESH-RUN GUARD (2026-08-15). This submission starts a NEW run in a NEW
# store dir: every piece of state -- the VGB beta ladder, the GB cap-cell
# grid, the band grid, the F-stat epoch cache -- is built from the config
# above, so NO migration scripts apply (they operate on an existing h5 and
# would fail the job here). Resubmitting this same script later RESUMES this
# new store normally; only the very first submission is a fresh start.
# Refuse to start fresh on top of an existing store rather than silently
# resuming one and reporting it as a fresh run.
# ============================================================================
if [ -e "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" ]; then
  echo "[FRESH] ${STORE_DIR} already holds a run -- RESUMING it."
  echo "[FRESH] For a genuinely fresh start, point STORE_DIR at a new dir"
  echo "        or move this one aside first."
else
  echo "[FRESH] no store at ${STORE_DIR} -- starting a NEW run from scratch."
  echo "[FRESH] stages run from the top (noise_search [psd + galfor] ->"
  echo "        gb_search_1 -> gb_search_2 -> gb_search_3 -> full_pe, with"
  echo "        the noise SAMPLED in all three search stages); the F-stat"
  echo "        grid + epoch center table are fitted fresh against this"
  echo "        run's own residual."
fi

# ============================================================================
# OPTIONAL LAUNCH SHORTCUT: graft v3's finished noise_search (2026-08-18)
# ============================================================================
# v4 changes NOTHING on the noise side -- the whole config diff against v3 is
# GB / sig-het / F-stat knobs -- so refitting the PSD and galactic foreground
# from scratch just reproduces a result v3 already has, at ~1.5 h.
#
# But noise_vgb_search MUST re-run: the VGB ladder moved to eryn's
# make_ladder, and a resumed store's stored ladder WINS over the configured
# one. (Measured on the temper probes: arms prepped from an older base kept
# the old 1/1.2**i ladder, while freshly-built arms got make_ladder.)
#
# So: let v4 author its own store -- every grid, shape and ladder correct by
# construction -- and move only the fitted numbers in.
#
#   1. sbatch this script against the fresh STORE_DIR. Let it reach
#      noise_search and SAVE ONE iteration, then scancel. That iteration is
#      throwaway; it exists so the datasets are allocated with >= 1 row.
#   2. python scripts/fstat_proposal/graft_noise_state.py \
#          <v3_store>/gf_prod_3mo_testing.h5 \
#          ${STORE_DIR}/${BASE_FILE_NAME}_testing.h5          # dry run
#      ... then the same command with --apply.
#   3. sbatch this script again. It resumes from the grafted row, sees
#      noise_search complete, and starts noise_vgb_search on the NEW ladder.
#
# The graft tool finds v3's handover row itself (VGB is frozen for the whole
# of noise_search and starts moving on the first noise_vgb iteration), gates
# on both stores having zero GB leaves, and refuses to touch sub_backend/vgb
# -- which is where the ladder lives, and the entire point of the exercise.
# Do NOT rewind a COPY of the v3 store instead: that carries v3's grids and
# rung counts into v4 and needs a migration per array, which is how the three
# earlier band-grid migrations failed.

# LATER REFITS: GB_FSTAT_REFIT_EVERY=40 ITERATIONS (~3.5% overhead at a
# 17.7-min fit). ⚠ THE UNITS CHANGED 2026-09-18 and this note predated it:
# the clock counts distinct global-fit ITERATIONS THIS BRANCH WAS PROPOSED
# IN, not proposal hits. In a SEQUENTIAL stage (gb_search) the GB move
# fires every iteration, so 40 really is 40. In a random_choice stage one
# wrapped move is drawn per step, so the clock ticks at that move's own
# rate and 40 means ~40/(draw probability) iterations -- read the cadence
# per stage, not globally. To force an extra
# refit mid-run, stop the job and archive the epoch dir, then resubmit:
#   mv ${STORE_DIR}/gb_fstat_fit/shared/epoch_* /tmp/  &&  sbatch ...
# (_latest_epoch() then returns None and the fit-in-move rebuilds.)

# DEDICATED SAVER RANK (armed 2026-08-15, user directive -- the [SAVE]
# math flipped: the ~60 s sync write was 2% of a 55-min iteration but
# is 6-10% of the post-mega-batch 10-17 min iterations). np>=3: rank 0
# samples, the HIGHEST rank becomes the async results/saver rank, the
# middle spare is stopped at startup (run.py GlobalFit role logic).
# FIRST-LAUNCH CHECK (known caveat): run_combined_staged.py builds on
# EVERY rank before roles resolve -- watch nvidia-smi for saver/spare
# device allocations; if the extra ranks hold GPU memory, drop back to
# the plain single-process line below until the rank-gated build lands.
# ============================================================================
# LADDER PREFLIGHT. Only fires on a RESUME (a fresh submission has no store
# yet and skips it). Resume derives the GB rung count from the stored
# band_temps shape, NOT from GB_NTEMPS above -- refuse to start rather than
# run a silently-wrong ladder for days.
# ============================================================================
if [ -e "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" ]; then
  python - "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" "${GB_NTEMPS}" <<'PYEOF' || exit 2
import sys, h5py
store, want = sys.argv[1], int(sys.argv[2])
with h5py.File(store, "r") as f:
    bt = f["global_fit"]["sub_backend"]["gb"].get("band_temps")
    if bt is None:
        print("[LADDER] no gb band_temps; nothing to check.")
        raise SystemExit(0)
    have = int(bt.shape[-1])
print(f"[LADDER] stored gb rungs = {have}, GB_NTEMPS = {want}")
if have != want:
    print(f"[LADDER] REFUSING TO START: the store would run {have} rungs, not "
          f"{want}. Resume takes the STORED count. Re-rung it first:\n"
          f"  python scripts/fstat_proposal/reset_recipe_stage.py {store} "
          f"gb_search --rewind-to-empty gb --apply\n"
          f"  python scripts/fstat_proposal/rerunge_gb_ladder.py {store} gb "
          f"{want} --apply")
    raise SystemExit(2)
print("[LADDER] OK.")
PYEOF
fi

# ============================================================================
# V8 NOISE PREFLIGHT. Fail here, not 20 minutes into a slurm allocation:
#   * the modulation table must exist;
#   * the NOISE brick must exist and carry /ltts;
#   * an existing store must have been sampled under THIS noise identity
#     (the run.py resume guard is authoritative; this is the cheap copy).
# ============================================================================
python - "${GALFOR_MODULATION_PATH}" "${MOJITO_DATA_PATH}" "${NOISE_FILE:-}"   "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" "${WDM_PSD_METHOD}" <<'PYEOF' || exit 2
import glob, os, sys
import h5py
mod, mojito, noise_file, store, method = sys.argv[1:6]
if not os.path.isfile(mod):
    print(f"[V8-NOISE] REFUSING: modulation table {mod!r} not found.")
    raise SystemExit(2)
if not noise_file:
    hits = sorted(glob.glob(os.path.join(mojito, "data", "INSTRUMENT", "L1", "NOISE_*")))
    if not hits:
        print(f"[V8-NOISE] REFUSING: no NOISE_* brick under {mojito!r} and NOISE_FILE unset.")
        raise SystemExit(2)
    noise_file = hits[0]
with h5py.File(noise_file, "r") as f:
    if "ltts" not in f:
        print(f"[V8-NOISE] REFUSING: {noise_file!r} has no /ltts group.")
        raise SystemExit(2)
    n = f["ltts"]["ltt_12"].shape[0]
print(f"[V8-NOISE] delay table OK: {noise_file} (/ltts, {n} samples/link)")
if os.path.exists(store):
    with h5py.File(store, "r") as f:
        grp = f.get("global_fit", {})
        ident = grp.get("noise_model_identity") if hasattr(grp, "get") else None
        if ident is None:
            print(f"[V8-NOISE] REFUSING: {store!r} predates noise-model identity "
                  "records -- it cannot have been sampled under the v8 noise "
                  "model. Use a fresh STORE_DIR.")
            raise SystemExit(2)
        a = dict(ident.attrs)
        # The coarse mode/Q are PART of the noise identity: they change the
        # PSD/galfor transition kernel on identical array shapes, so a resume
        # across them is refused by run.py. Check them here too, or the
        # mismatch only surfaces minutes into the allocation.
        want_mode = os.environ.get("COARSE_GPU_MODE", "delayed_acceptance")
        want_q = int(os.environ.get("COARSE_Q", "8"))
        mismatches = {}
        if not bool(a.get("unequal_arm")):
            mismatches["unequal_arm"] = (a.get("unequal_arm"), True)
        if str(a.get("wdm_psd_method", "")) != method:
            mismatches["wdm_psd_method"] = (a.get("wdm_psd_method"), method)
        if str(a.get("coarse_mode", "")) != want_mode:
            mismatches["coarse_mode"] = (a.get("coarse_mode"), want_mode)
        if int(a.get("coarse_Q", 1)) != want_q:
            mismatches["coarse_Q"] = (a.get("coarse_Q"), want_q)
        if mismatches:
            print(f"[V8-NOISE] REFUSING: stored noise identity does not match this "
                  f"config (stored, wanted): {mismatches}. Full stored identity: {a}. "
                  "Use a fresh STORE_DIR.")
            raise SystemExit(2)
        print(f"[V8-NOISE] resume identity OK: {a}")
PYEOF

# ============================================================================
# SOURCES PREFLIGHT (6mo_v8). Hard-check the data before taking a slurm
# allocation (user ruling 2026-09-02: "script hard-checks brick presence
# at launch"). COMBINED-aware (2026-09-14): when SOURCE_TYPES carries
# COMBINED, the per-class L1 bricks are never read -- the checks become
# (a) the COMBINED file resolves exactly the way the loader's
# find_combined_file will resolve it (single .h5 / COMBINED_ prefix /
# MOJITO_COMBINED_FILE override), and (b) NOISE/GALFOR are not also
# listed (the loader refuses the double-count at build; fail it here in
# seconds instead). Without COMBINED, the old per-class brick check.
# ============================================================================
python - "${MOJITO_DATA_PATH}" "${MBHB_IDS:-}" "${EMRI_IDS:-}" "${SOBHB_IDS:-}" "${SOURCE_TYPES:-}" <<'PYEOF' || exit 2
import glob, os, sys
mojito, mbhb, emri, sobhb, src_types = sys.argv[1:6]
types = [s.strip().upper() for s in src_types.split(",") if s.strip()]
armed = {"MBHB": mbhb, "EMRI": emri, "SOBHB": sobhb}
bad = False
if "COMBINED" in types:
    for clash in ("NOISE", "GALFOR"):
        if clash in types:
            print(f"[SOURCES] REFUSING: SOURCE_TYPES lists COMBINED and "
                  f"{clash} -- the combined stream already contains it "
                  "(the loader refuses this at build).")
            bad = True
    folder = os.path.join(mojito, "data", "COMBINED", "L1")
    override = os.environ.get("MOJITO_COMBINED_FILE", "").strip()
    if override:
        p = override if os.path.isabs(override) else os.path.join(folder, override)
        if not os.path.exists(p):
            print(f"[SOURCES] REFUSING: MOJITO_COMBINED_FILE={override!r} "
                  f"does not exist (looked at {p}).")
            bad = True
        else:
            print(f"[SOURCES] COMBINED file (override): {p}")
    elif not os.path.isdir(folder):
        print(f"[SOURCES] REFUSING: no combined-data folder at {folder!r}.")
        bad = True
    else:
        cands = sorted(f for f in os.listdir(folder)
                       if f.endswith(".h5") and not f.startswith("."))
        named = [f for f in cands if f.upper().startswith("COMBINED_")]
        if not cands:
            print(f"[SOURCES] REFUSING: no .h5 file in {folder!r}.")
            bad = True
        elif len(cands) > 1 and len(named) != 1:
            print(f"[SOURCES] REFUSING: {len(cands)} .h5 files in "
                  f"{folder!r} ({cands}); set MOJITO_COMBINED_FILE.")
            bad = True
        else:
            pick = named[0] if len(cands) > 1 else cands[0]
            print(f"[SOURCES] COMBINED file: {os.path.join(folder, pick)}")
    for cls, ids in armed.items():
        state = f"ids {ids}" if ids.strip() else "not armed"
        print(f"[SOURCES] {cls}: catalogue-only under COMBINED ({state}).")
else:
    for cls, ids in armed.items():
        if not ids.strip():
            print(f"[SOURCES] {cls}: not armed (empty id list).")
            continue
        d = os.path.join(mojito, "data", cls, "L1")
        hits = sorted(glob.glob(os.path.join(d, "*")))
        if not hits:
            print(f"[SOURCES] REFUSING: {cls} armed (ids {ids}) but no bricks "
                  f"under {d!r}.")
            bad = True
        else:
            print(f"[SOURCES] {cls}: {len(hits)} brick file(s) under {d}; "
                  f"ids {ids} (per-id resolution is the loader's).")
if bad:
    raise SystemExit(2)
print("[SOURCES] preflight OK.")
PYEOF

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  # Multi-node launch = Intel MPI's OWN launcher (hydra) bootstrapped from the
  # SLURM allocation, NOT `srun --mpi=...` (WP7 Step 0, 2026-09-16): on this
  # cluster `srun` without PMI gives every rank a size-1 world, and
  # `srun --mpi=pmix` bootstraps but Intel MPI's OFI business-card exchange
  # then fails ("Missing hostname ... in business card"). `-ppn 1` places
  # consecutive ranks round-robin over the hosts (A,B,A,B,...) = the cyclic
  # placement the walker-block layout wants: head + saver on node A, the
  # compute ranks spread over both nodes. UCX finds no cross-node transport
  # here (only self/sysv/posix/cma), so the fabric is pinned to libfabric's
  # tcp provider -- correctness first; a faster provider is a WP7 Step 4
  # measurement, not a launch requirement.
  # SLURM_NTASKS is always set inside a job step; the :-3 default matches the
  # mpiexec branch below so an odd/manual allocation still launches the
  # head+compute+saver shape instead of dying on an unset var under `set -u`.
  export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp
  # Makes `-ppn 1` above actually bind: with the SLURM bootstrap, hydra
  # otherwise honours the scheduler's PER-NODE TASK COUNTS over -ppn. With
  # placement respect OFF the round robin is A,B,A,B,A -- two compute ranks
  # per 2-GPU node, which is what the walker-block layout wants.
  #
  # ⚠ NOT REQUIRED ON THIS SCRIPT'S NORMAL PATH, and I first recorded it as a
  # launch blocker in error (corrected 2026-09-24 by the rank-layout session,
  # with hardware corroboration). The self-dispatch block above passes
  # --distribution=cyclic whenever _NODES > 1, and cyclic already lands 5
  # tasks over 2 nodes as A:{0,2,4} B:{1,3} -- node A takes compute 0,
  # compute 2 and the SAVER, i.e. two compute ranks against a 2-GPU pool.
  # Pin or no pin, the placement is identical. The discriminator: v8's own
  # 1yr 4-GPU script never carried the pin and has launched fine.
  #
  # WHERE IT IS GENUINELY REQUIRED: any path that SKIPS the dispatch block
  # and therefore gets no --distribution=cyclic -- a manual
  # `sbatch --nodes=2 --ntasks=5 <script>` (the script's own comments flag
  # that case), and the INTERACTIVE allocation, where --ntasks-per-node=3
  # forces three tasks onto node A and -ppn 1 has to override it. That is
  # the 2026-09-18 failure the one-walker-replicas note describes.
  #
  # Kept as free insurance: identical placement under cyclic, correct
  # placement without it.
  export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
  mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 python scripts/fstat_proposal/run_combined_staged.py
else
  mpiexec -n "${SLURM_NTASKS:-3}" python scripts/fstat_proposal/run_combined_staged.py
fi
# python scripts/fstat_proposal/run_combined_staged.py   # single-process fallback
