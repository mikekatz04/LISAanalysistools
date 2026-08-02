"""Measurement-only audit of the GB proposal-Fisher reuse structure.

``GBSpecialBase._compute_proposal_cholesky`` is the top compute lever in a GB
search (an information matrix per source per iteration, ~17 waveform evals,
measured at 56-60% of a GPU search iteration and 87% of this repo's CPU smoke
propose).  Any scheme for computing it less often -- a shared cache, a
per-slot cache, clustering -- is a bet about *reuse structure*.  This module
measures that structure instead of assuming it.

**The axis is covariance error, not parameter distance.**  An earlier version
of this audit scored reuse by the Mahalanobis distance between two parameter
points in the proposal covariance's own metric.  That is the wrong
coordinate, and measurably so: the GB Fisher is extremely anisotropic (at
high SNR ``sigma_f0`` is a tiny fraction of a frequency bin), so two walkers
sitting on the same physical source are hundreds of "sigma" apart while their
proposal covariances agree to a fraction of a percent.  Mahalanobis distance
measures how much the POSTERIOR moved; reuse only cares how much the FISHER
moved.  So the primary quantity here is

    cov_reldiff = ||C_a - C_b||_F / ||C_b||_F

with everything else scored as a *predictor* of it.

What is measured, all broken down by temperature rung:

1. **Achievable reuse (oracle clustering).**  Greedy leader clustering whose
   membership test is ``cov_reldiff <= eps`` -- i.e. the number of Fisher
   computations a *perfect* cache would issue at a given accuracy budget.  No
   implementation can beat this, so if it does not show reuse, no cache does.
2. **Predictor quality.**  For sampled pairs: ``cov_reldiff`` alongside the
   candidate cheap predictors (max ``|dln x|`` over the positive-definite
   columns, max ``|dx|`` over the angle-like columns, Mahalanobis distance,
   and whether the shipped fixed-grid key matches).  Offline this gives the
   tolerance for whichever predictor separates best.
3. **Temporal reuse.**  Per ``(temp, walker, leaf)`` slot: the covariance
   error had the previous iteration's factor been reused, plus how often the
   slot identity survives at all (GB runs with ``preserve_leaf_identity=False``
   and RJ birth/death, so leaf indices churn).
4. **The shipped fixed-grid cache**, scored on the same run: hit rate AND the
   covariance error of each hit.

The audit NEVER changes the sampler's numbers: it is fed the direct
(uncached) factors and only reads them.  Enable with ``GB_FISHER_AUDIT=1``.
Knobs: ``GB_FISHER_AUDIT_OUT=<path.npz>`` (raw rows for offline analysis),
``GB_FISHER_AUDIT_EVERY`` (summary cadence in calls, default 25),
``GB_FISHER_AUDIT_MAX`` (per-call sample cap, default 256).
"""

from __future__ import annotations

import atexit
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Covariance-error budgets the achievable-reuse curve is evaluated at.
DEFAULT_EPS = (0.02, 0.05, 0.1, 0.25, 0.5)

# Sampling-basis columns that are positive with wide dynamic range, scored as
# relative (log) differences.  Everything else -- the already-logarithmic
# amplitude column, angles, cos/sin sky coordinates, the fdot_astro ratio --
# is scored as an absolute difference.
LOG_BASIS_NAMES = ("dist", "f0", "Mc", "fdot")


def _cov_from_chol(B: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Proposal covariance ``C = B B^T`` restricted to the valid parameters.

    ``B`` is the eigen square root built by ``_compute_proposal_cholesky_direct``
    (rows = parameters, columns = eigen-directions), so the restriction keeps
    ALL eigen-columns and only drops the requested parameter ROWS -- e.g. the
    ``fdot_astro_ratio`` row the direct method deliberately zeroes, which
    would otherwise make ``C`` singular.
    """
    Bv = B[:, valid, :]
    return Bv @ np.swapaxes(Bv, -1, -2)


def _maha(C_ref: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """``sqrt(dx^T C_ref^{-1} dx)`` batched, ridge-guarded.

    Kept as a candidate predictor only (see the module docstring): it is a
    poor proxy for covariance change, and the audit reports it next to the
    covariance error so that claim stays falsifiable rather than assumed.
    """
    n, k, _ = C_ref.shape
    tr = np.einsum("nii->n", C_ref) / max(k, 1)
    ridge = 1e-10 * np.maximum(tr, 1e-300)
    C = C_ref + ridge[:, None, None] * np.eye(k)[None]
    z = np.linalg.solve(C, dx[..., None])[..., 0]
    return np.sqrt(np.maximum(np.einsum("ni,ni->n", dx, z), 0.0))


def _cov_reldiff(C_a: np.ndarray, C_b: np.ndarray) -> np.ndarray:
    """Frobenius relative difference ``||C_a - C_b|| / ||C_b||`` batched.

    Compared on the COVARIANCE, not on ``B``: ``B`` is defined only up to the
    sign/ordering of its eigen-columns, so two nearby parameter points can
    produce elementwise-different ``B`` with identical covariance.
    """
    num = np.linalg.norm(C_a - C_b, axis=(-2, -1))
    den = np.maximum(np.linalg.norm(C_b, axis=(-2, -1)), 1e-300)
    return num / den


class FisherProposalAudit:
    """Accumulates the reuse statistics described in the module docstring."""

    def __init__(self, name, ndim, skip_cols=(), basis_names=None,
                 eps_grid=DEFAULT_EPS):
        self.name = name
        self.ndim = int(ndim)
        valid = np.ones(self.ndim, dtype=bool)
        for c in skip_cols:
            if c is not None:
                valid[int(c)] = False
        self.valid = valid
        self.eps_grid = tuple(float(e) for e in eps_grid)
        names = list(basis_names) if basis_names is not None else []
        self.log_cols = np.array(
            [i for i, nm in enumerate(names) if nm in LOG_BASIS_NAMES],
            dtype=int,
        )
        self.lin_cols = np.array(
            [i for i in range(self.ndim)
             if i not in set(self.log_cols.tolist()) and valid[i]],
            dtype=int,
        )
        self.max_per_call = int(os.environ.get("GB_FISHER_AUDIT_MAX", "256"))
        self.every = int(os.environ.get("GB_FISHER_AUDIT_EVERY", "25"))
        self.out_path = os.environ.get("GB_FISHER_AUDIT_OUT", "") or None

        self.n_calls = 0
        self.n_lookups = 0
        # per-slot memory: slot key -> (coords, covariance, call index)
        self._slot: dict[int, tuple] = {}
        self._slot_repeats = 0
        # shipped fixed-grid shadow cache: key -> covariance of the first
        # occupant, so every hit can be scored for accuracy
        self._grid: dict = {}
        self._grid_hits = 0
        self._grid_tot = 0
        self._grid_err: list = []
        # temporal rows: (temp, cov_reldiff, dlog, dang, d_maha, gap)
        self._temporal: list = []
        # pair rows: (temp, cov_reldiff, dlog, dang, d_maha, grid_same)
        self._pairs: list = []
        # PERSISTENT oracle cache, one leader set per accuracy budget. This
        # simulates a real cache: leaders survive across calls and across
        # iterations, exactly as cached factors would, so the reported
        # computes/source is a cache MISS rate and not a within-batch
        # clustering statistic (real batches hold only 1-2 sources per rung,
        # so a per-call clustering measurement understates reuse badly).
        self._oracle = {e: [] for e in self.eps_grid}      # eps -> [C, ...]
        self._oracle_lookups = 0
        self._oracle_hits = {e: 0 for e in self.eps_grid}
        self._oracle_by_temp: dict = {}
        self.oracle_max_leaders = int(
            os.environ.get("GB_FISHER_AUDIT_LEADERS", "3000"))
        self._oracle_capped = False
        # direct sensitivity probe: (col, delta) -> list of cov_reldiff
        self._probe: dict = {}
        self.n_probed = 0
        # conditioning diagnostics: (cond(C), n eigenvalues pinned at the top)
        self._cond: list = []

        # The measurement is worthless if the process exits before it is
        # printed -- a sampler run has no natural "end of audit" hook.
        atexit.register(self._at_exit)

    def _at_exit(self):
        if self.n_calls:
            logger.info("[GB_FISHER_AUDIT %s] FINAL\n%s", self.name, self.summary())
            path = self.dump()
            if path:
                logger.info("[GB_FISHER_AUDIT %s] raw rows -> %s", self.name, path)

    # ------------------------------------------------------------------
    def _scalar_dists(self, xa, xb, dx):
        """``(dlog, dang)`` between parameter blocks, given periodic ``dx``."""
        if self.log_cols.size:
            la = np.abs(xa[:, self.log_cols])
            lb = np.abs(xb[:, self.log_cols])
            dlog = np.abs(np.log(np.maximum(la, 1e-300)
                                 / np.maximum(lb, 1e-300))).max(axis=1)
        else:
            dlog = np.zeros(len(xa))
        dang = (np.abs(dx[:, self.lin_cols]).max(axis=1)
                if self.lin_cols.size else np.zeros(len(xa)))
        return dlog, dang

    # ------------------------------------------------------------------
    def record(self, coords, chol, temp_inds, walker_inds, leaf_inds,
               grid_keys=None, periodic_distance=None):
        """Ingest one ``_compute_proposal_cholesky`` batch (host arrays)."""
        coords = np.asarray(coords, dtype=np.float64)
        chol = np.asarray(chol, dtype=np.float64)
        n = coords.shape[0]
        if n == 0:
            return
        self.n_calls += 1
        self.n_lookups += n
        C = _cov_from_chol(chol, self.valid)

        t_i = np.asarray(temp_inds).astype(np.int64)
        w_i = np.asarray(walker_inds).astype(np.int64)
        l_i = np.asarray(leaf_inds).astype(np.int64)
        slots = (t_i * 1_000_000 + w_i) * 1_000_000 + l_i
        self._pdist = periodic_distance

        self._record_conditioning(C)
        if grid_keys is not None:
            self._record_grid(grid_keys, C)
        self._record_temporal(slots, coords, C, t_i)
        self._record_oracle(C, t_i)
        self._record_pairs(coords, C, t_i, grid_keys)

        if self.every > 0 and self.n_calls % self.every == 0:
            logger.info("%s\n%s", f"[GB_FISHER_AUDIT {self.name}]", self.summary())

    def _dx(self, xa, xb):
        """``xb - xa`` with the branch's angular periods folded in."""
        if self._pdist is not None:
            return np.asarray(self._pdist(xa, xb))
        return xb - xa

    # ------------------------------------------------------------------
    def _record_conditioning(self, C):
        """Detect eigen-floor domination of the proposal covariance.

        ``_compute_proposal_cholesky_direct`` clamps the information-matrix
        spectrum at ``floor = 1e-10 * max|eval|`` before inverting, so a
        clamped direction enters the proposal with width
        ``1/sqrt(1e-10 * lam_max)`` -- 1e5 times the best-constrained
        direction, contributing ~1e10 times more to ``||C||_F``.  Whenever a
        direction is clamped, BOTH the proposal jump and any covariance-based
        reuse metric are dominated by a numerically arbitrary eigenvector.
        ``cond(C) ~ 1e10`` is therefore a clean detector that the clamp is
        active, and the count of near-degenerate top eigenvalues says how
        many directions it swallowed.
        """
        ev = np.linalg.eigvalsh(C)
        ev = np.maximum(ev, 1e-300)
        cond = ev[:, -1] / ev[:, 0]
        n_top = (ev >= 0.99 * ev[:, -1:]).sum(axis=1)
        self._cond.append(np.stack([cond, n_top.astype(float)], axis=1))

    def _record_grid(self, grid_keys, C):
        """Hit rate AND per-hit covariance error of the shipped grid cache."""
        for i, k in enumerate(grid_keys):
            prev = self._grid.get(k)
            self._grid_tot += 1
            if prev is None:
                self._grid[k] = C[i]
            else:
                self._grid_hits += 1
                self._grid_err.append(
                    float(_cov_reldiff(prev[None], C[i][None])[0])
                )

    def _record_temporal(self, slots, coords, C, t_i):
        seen = {int(s): i for i, s in enumerate(slots)}
        prev_keys = [s for s in seen if s in self._slot]
        if prev_keys:
            self._slot_repeats += len(prev_keys)
            ci = np.asarray([seen[s] for s in prev_keys])
            x_prev = np.stack([self._slot[s][0] for s in prev_keys])
            C_prev = np.stack([self._slot[s][1] for s in prev_keys])
            gaps = np.asarray([self.n_calls - self._slot[s][2]
                               for s in prev_keys], dtype=float)
            dx = self._dx(x_prev, coords[ci])
            dlog, dang = self._scalar_dists(coords[ci], x_prev, dx)
            self._temporal.append(np.stack([
                t_i[ci].astype(float),
                _cov_reldiff(C_prev, C[ci]),
                dlog, dang,
                _maha(C_prev, dx[:, self.valid]),
                gaps,
            ], axis=1))
        for s, i in seen.items():
            self._slot[int(s)] = (coords[i], C[i], self.n_calls)

    def _record_oracle(self, C, t_i):
        """Feed every lookup through the persistent ideal cache."""
        for i in range(C.shape[0]):
            self._oracle_lookups += 1
            rec = self._oracle_by_temp.setdefault(
                int(t_i[i]), {e: [0, 0] for e in self.eps_grid})
            for eps in self.eps_grid:
                leaders = self._oracle[eps]
                hit = False
                if leaders:
                    L = np.asarray(leaders)
                    hit = bool(
                        _cov_reldiff(L, np.repeat(C[i][None], len(L), 0)).min()
                        <= eps
                    )
                if hit:
                    self._oracle_hits[eps] += 1
                    rec[eps][0] += 1
                elif len(leaders) < self.oracle_max_leaders:
                    leaders.append(C[i])
                else:
                    self._oracle_capped = True
                rec[eps][1] += 1

    def _record_pairs(self, coords, C, t_i, grid_keys):
        """Predictor table on sampled within-call pairs, per temperature."""
        for t in np.unique(t_i):
            sel = np.where(t_i == t)[0]
            if sel.size > self.max_per_call:
                sel = sel[np.linspace(0, sel.size - 1,
                                      self.max_per_call).astype(int)]
            if sel.size < 2:
                continue
            ii, jj = np.triu_indices(sel.size, k=1)
            if ii.size > 400:
                pick = np.linspace(0, ii.size - 1, 400).astype(int)
                ii, jj = ii[pick], jj[pick]
            a, b = sel[ii], sel[jj]
            dx = self._dx(coords[a], coords[b])
            dlog, dang = self._scalar_dists(coords[a], coords[b], dx)
            err = _cov_reldiff(C[a], C[b])
            d_m = _maha(C[a], dx[:, self.valid])
            same = (np.asarray([grid_keys[x] for x in a])
                    == np.asarray([grid_keys[x] for x in b])
                    ).astype(float) if grid_keys is not None else np.zeros(len(a))
            self._pairs.append(np.stack(
                [np.full(len(a), float(t)), err, dlog, dang, d_m, same], axis=1))

    # ------------------------------------------------------------------
    def record_probe(self, col, delta, errs):
        """One column of the direct sensitivity scan.

        ``errs`` are ``cov_reldiff`` values between the proposal covariance at
        ``x`` and at ``x`` perturbed by ``delta`` in column ``col`` (relative
        for the log-scaled columns, absolute otherwise).  This is the
        measurement that SETS a cache key's cell widths: the admissible cell
        for a column is the perturbation at which its covariance error reaches
        the accuracy budget.  Columns whose error stays flat are columns the
        key should not contain at all.
        """
        self._probe.setdefault((int(col), float(delta)), []).extend(
            float(e) for e in errs
        )

    @staticmethod
    def _pct(a, qs=(50, 90, 99)):
        if len(a) == 0:
            return "n/a"
        p = np.percentile(a, qs)
        return " ".join(f"p{q}={v:.3g}" for q, v in zip(qs, p))

    def summary(self) -> str:
        """Compact human-readable report of everything accumulated so far."""
        L = [f"calls={self.n_calls} lookups={self.n_lookups} "
             f"ndim_valid={int(self.valid.sum())} "
             f"log_cols={self.log_cols.tolist()} lin_cols={self.lin_cols.tolist()}"]

        # --- 1. achievable reuse (persistent ideal cache) -----------------
        if self._oracle_lookups:
            n = self._oracle_lookups
            L.append(f"ACHIEVABLE reuse -- persistent IDEAL cache keyed on the "
                     f"covariance itself ({n} lookups)"
                     + (" [LEADER CAP HIT: optimistic]" if self._oracle_capped
                        else ""))
            L.append("  " + "  ".join(
                f"eps{eps:g}: hit {100.0 * self._oracle_hits[eps] / n:5.1f}% "
                f"({len(self._oracle[eps])} entries)"
                for eps in self.eps_grid))
            for t, rec in sorted(self._oracle_by_temp.items()):
                L.append(f"  T{t}: " + " ".join(
                    f"eps{eps:g}:{100.0 * rec[eps][0] / max(rec[eps][1], 1):.0f}%"
                    for eps in self.eps_grid))

        # --- 1b. conditioning / eigen-floor domination --------------------
        if self._cond:
            cd = np.concatenate(self._cond, axis=0)
            frac = 100.0 * (cd[:, 0] > 1e9).mean()
            L.append(f"CONDITIONING of the proposal covariance "
                     f"(n={len(cd)}): cond {self._pct(cd[:, 0])}")
            L.append(f"  eigen-floor ACTIVE (cond > 1e9) on {frac:.1f}% of "
                     f"sources; near-degenerate top eigenvalues "
                     f"{self._pct(cd[:, 1], (50, 90))}")
            if frac > 20.0:
                L.append("  -> ||C|| and the proposal jump are dominated by a "
                         "clamped, numerically arbitrary direction; covariance "
                         "reuse metrics measure THAT, not parameter dependence")

        # --- 2. temporal --------------------------------------------------
        seen_slots = len(self._slot)
        L.append(f"slot identity: {self._slot_repeats} revisits over "
                 f"{seen_slots} distinct (temp,walker,leaf) slots")
        if self._temporal:
            tmp = np.concatenate(self._temporal, axis=0)
            L.append(f"TEMPORAL reuse (n={len(tmp)}): cov err if the previous "
                     f"factor were reused: {self._pct(tmp[:, 1])}")
            for eps in self.eps_grid:
                L.append(f"  eps{eps:<5g} -> temporal hit "
                         f"{100.0 * (tmp[:, 1] <= eps).mean():5.1f}%")
            L.append(f"  dlog {self._pct(tmp[:, 2])} | dang {self._pct(tmp[:, 3])}"
                     f" | d_maha {self._pct(tmp[:, 4])}")
            for t in np.unique(tmp[:, 0]).astype(int):
                sub = tmp[tmp[:, 0] == t]
                L.append(f"  T{t}: cov err {self._pct(sub[:, 1], (50, 90))} "
                         + " ".join(f"eps{e:g}:{100.0 * (sub[:, 1] <= e).mean():.0f}%"
                                    for e in self.eps_grid))

        # --- 3. predictor quality ----------------------------------------
        if self._pairs:
            pr = np.concatenate(self._pairs, axis=0)
            L.append(f"PREDICTORS over {len(pr)} pairs "
                     f"(cov err {self._pct(pr[:, 1])}):")
            for label, col, cuts in (
                ("dlog", 2, (0.01, 0.05, 0.2)),
                ("dang", 3, (0.05, 0.2, 0.5)),
                ("d_maha", 4, (1.0, 10.0, 100.0)),
            ):
                for c in cuts:
                    m = pr[:, col] <= c
                    if m.sum() < 3:
                        continue
                    L.append(f"  {label} <= {c:<5g} ({100.0 * m.mean():4.1f}% of "
                             f"pairs): cov err {self._pct(pr[m, 1], (50, 90, 99))}")
            m = pr[:, 5] > 0
            if m.sum() >= 3:
                L.append(f"  same fixed-grid cell ({100.0 * m.mean():4.1f}% of "
                         f"pairs): cov err {self._pct(pr[m, 1], (50, 90, 99))}")

        # --- 3b. direct per-column sensitivity ---------------------------
        if self._probe:
            L.append(f"SENSITIVITY probe ({self.n_probed} sources): median cov "
                     f"err per column per perturbation "
                     f"[* = relative, else absolute]")
            cols = sorted({c for c, _ in self._probe})
            for c in cols:
                deltas = sorted(d for cc, d in self._probe if cc == c)
                kind = "*" if c in self.log_cols.tolist() else " "
                parts = " ".join(
                    f"{d:g}{kind}:{np.median(self._probe[(c, d)]):.3g}"
                    for d in deltas
                )
                L.append(f"  col{c}: {parts}")

        # --- 4. the shipped grid cache -----------------------------------
        if self._grid_tot:
            L.append(f"SHIPPED fixed-grid cache: hit rate "
                     f"{100.0 * self._grid_hits / self._grid_tot:.1f}% "
                     f"({self._grid_hits}/{self._grid_tot}, "
                     f"{len(self._grid)} cells)")
            if self._grid_err:
                ge = np.asarray(self._grid_err)
                L.append(f"  cov err ON HITS: {self._pct(ge)} "
                         f"(fraction over 25%: {100.0 * (ge > 0.25).mean():.1f}%)")
        return "\n".join(L)

    def dump(self):
        """Write the raw rows to ``GB_FISHER_AUDIT_OUT`` (if set)."""
        if not self.out_path:
            return None

        def _cat(rows, width):
            return (np.concatenate(rows, axis=0) if rows
                    else np.zeros((0, width)))

        np.savez_compressed(
            self.out_path,
            temporal=_cat(self._temporal, 6),
            pairs=_cat(self._pairs, 6),
            oracle=np.asarray(
                [[eps, self._oracle_hits[eps], self._oracle_lookups,
                  len(self._oracle[eps])] for eps in self.eps_grid], dtype=float),
            grid_err=np.asarray(self._grid_err, dtype=float),
            grid=np.asarray([self._grid_hits, self._grid_tot, len(self._grid)],
                            dtype=float),
            eps_grid=np.asarray(self.eps_grid),
            probe=np.asarray(
                [[c, d, e] for (c, d), errs in sorted(self._probe.items())
                 for e in errs], dtype=float
            ) if self._probe else np.zeros((0, 3)),
        )
        return self.out_path
