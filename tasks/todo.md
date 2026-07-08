# Task: add `get_vel(t, sc)` to the Orbits classes

Mirror `get_pos` so spacecraft velocities can be evaluated at arbitrary times.
Decision (with user): **Python-only evaluator** (no C++/ABI/rebuild) + **JAX parity**.
Rationale: the velocity grid is already computed and stored during `configure()`
(`self.v` on `self.sc_t`); we only need an evaluator that interpolates it the same
linear way the C++ backend interpolates positions.

## Plan
- [ ] Base `Orbits.get_vel(t, sc)` in `src/lisatools/detector.py`, right after `get_pos`.
      - Same signature / 3 input modes / squeeze / `(...,3)` output as `get_pos`.
      - Core: `xp.interp` of `self.v` over `self.sc_t`, per-spacecraft mask loop
        (handles per-element `sc`), wrapped in `xp.asarray` (base `_v` is numpy even on GPU).
      - Validate all `sc` in `self.SC` (C++ path delegates this to the backend).
      - `_check_configured()` guard; works even when `make_cpp=False`.
      - Doc the one seam vs C++: out-of-range clamps (xp.interp) instead of zeroing.
      - Breadcrumb TODO for later C++ `get_vel_wrap` swap.
- [ ] JAX `JAXL1Orbits.get_vel(t, sc)`, right after its `get_pos`.
      - Reuse grid-generic `interpolate_pos` on `self.v` (DRY, no new jitted fn).
- [ ] Tests in `tests/test_detector.py`:
      - Extend `test_orbits` with a `get_vel` finite check.
      - New `test_get_vel`: node exactness, midpoint linearity, input-mode shapes.
- [ ] Run the suite; iterate.
- [ ] Commit, push, draft PR (branch off feat-lisa-frame).

## Notes
- Covers Orbits / EqualArmlength / ESA / L1Orbits via the base method (none override get_pos).
- No changes to C++, OrbitsView, ABI version, or build.

## Review
Implemented as planned; pure-Python + JAX, no native/ABI/build changes.

- `src/lisatools/detector.py`
  - `Orbits.get_vel(t, sc)` after `get_pos`: same 3 input modes / squeeze /
    `(...,3)` output; core is a per-spacecraft `xp.interp` of `self.v` over
    `self.sc_t`, `xp.asarray`-wrapped (base `_v` is numpy even on GPU).
    Validates all `sc in self.SC` (raises instead of silent-zeroing);
    `_check_configured()` guard; carries a `TODO(get_vel)` breadcrumb for the
    future C++ `get_vel_wrap` swap. Covers Orbits/EqualArmlength/ESA/L1Orbits.
  - `JAXL1Orbits.get_vel(t, sc)` after its `get_pos`: reuses the grid-generic
    `interpolate_pos` on `self.v` (no new jitted fn).
- `tests/test_detector.py`
  - `test_orbits`: added a `get_vel` finite check.
  - `test_get_vel`: node exactness (interp at a node == node value), midpoint
    linearity (== average of adjacent nodes), input-mode shapes, out-of-range
    `sc` raises.

Verification (worktree src shadowed onto the editable install; pure-Python so
no rebuild):
- `python -m unittest tests.test_detector` -> 2 passed.
- JAX: `jax_here=True`, `JAXL1Orbits.get_vel` bound, `interpolate_pos`
  node-exactness + midpoint-linearity on a synthetic velocity grid: OK.

One documented seam vs the C++ position path: out-of-range times clamp to the
grid edges (`xp.interp`) instead of returning zeros. Left `get_pos`'s
copy-paste docstring bug untouched (out of scope).
