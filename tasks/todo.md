# Task: generalize `to_lisa_frame` / `from_lisa_frame` to batched sources

Crash: `np.dot((1,3),(1,3)) not aligned` at `sources/utils.py:862` from
`mbh_catalogue_to_sampling_basis`. Root cause: the HEAD version was scalar-only
(`get_pos(float, ...)` squeezes to `(3,)`, so `np.dot(vec3, vec3)` was a plain
inner product). The batching change routes inputs through `np.atleast_1d`, so
every vector becomes `(N,3)` and `np.dot` turns into a matrix multiply.
Latent batch bugs beyond the crash (correct only coincidentally at N=1):
- axis-less `np.linalg.norm` on `(N,3)` normalizations (also in
  `_get_orbital_quantities`),
- `(N,) * (N,3)` broadcasts: `p = cp*u + sp*v`, and
  `k = -cb*cl*x_lisa - ...` in `from_lisa_frame`.

## Plan
- [ ] Add tiny helpers next to `_get_orbital_quantities`: `_dot(a,b)` =
      row-wise dot over last axis; `_unit(v)` = axis-aware normalization.
- [ ] `_get_orbital_quantities`: norms -> `axis=-1, keepdims=True`.
- [ ] `to_lisa_frame`: `k = np.stack([...], axis=-1)`; all `np.dot` -> `_dot`;
      `_unit` for u/u_lisa; `cp[..., None]`/`sp[..., None]` in p.
- [ ] `from_lisa_frame`: `(cb*cl)[..., None] * x_lisa` etc.; all `np.dot` ->
      `_dot`; `_unit`; `cp[..., None]`.
- [ ] Verify (scratchpad script):
      - scalar path == HEAD implementation (loaded from `git show HEAD:`),
      - batched (N=8) == HEAD looped source-by-source,
      - round trip `from(to(x)) ~= x` for scalar + batch.

## Review
(to fill after verification)
