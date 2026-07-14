# Flow-proposal offline harness (2026-07-14)

Measures ConditionalFlowMove proposal quality against the run's own
likelihood, offline, without touching a live run. Produced the numbers in
`tasks/todo_flow_mbh_proposal.md`.

- `splice_test.py` — the acceptance-ceiling measurement: rebuilds residuals
  from a backend snapshot, scores walker-swap / lagged / flow-draw candidates
  per leaf through the moves' own `compute_like`.
- `train_offline_flows.py` — trains candidate flow configs (train_noise,
  buffer window, periodic_in_cholesky, ...) on backend-derived per-leaf
  buffers (CUDA_VISIBLE_DEVICES pins the GPU).
- `score_offline_flows.py` — scores every candidate checkpoint + the live one
  in a single residual pass; reports the exact independence-MH statistic
  lnpdiff = [ll(y)-logq(y)] - [ll(x)-logq(x)] and implied acceptance.

Usage notes (all machine-specific, edit the constants at the top):
- Scripts expect a scratch dir with `main_backend.h5` (a COPY of the run's
  backend; never read the live file directly), the live flow checkpoints, and
  a patched copy of the settings module (`splice_settings.py`: point
  `head_dir` at scratch, set `gpus=[<free gpu>]`, `do_plots=False`) so the
  data build cannot collide with a live run.
- Validation gate: recomputed walker log-likes must match the stored
  `mcmc/log_like` (median offset ~0) before any candidate numbers are
  trusted.
