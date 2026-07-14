# Lessons

## 2026-07-08 — multi-GPU sync points
Proposed removing the unconditional `synchronize()` before the per-chunk D2H in
`AnalysisContainerArray._compute_group_likelihood`; user corrected: keep it.
**Pattern**: host-consumption barriers (sync before copying per-chunk results
to CPU) are load-bearing and shared across moves (PSD/MBH/EMRI). Optimize by
reducing barrier *frequency* (bigger batches, fewer chunks), never by removing
the barrier itself. Also: `.get()` syncing "should be enough" arguments ignore
off-stream work and shared-code blast radius — weigh risk vs measured gain.

## 2026-07-08 — venv mutations
User intercepted my rebuild and ran it themselves. Don't run installs/rebuilds
into the shared erebor_org_setup/.venv unprompted — especially with a live
production run attached to it. Prepare the exact command, hand it to the user
(or ask), unless explicitly told to run it.
