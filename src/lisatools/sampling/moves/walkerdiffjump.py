"""Cross-walker difference-vector jump for the add/remove source branches.

User ruling 2026-10-18 (Mike, the EMRI audit): "we do not have enough
walkers for the stretch move, so we need to improve what we have." The run
has 4 walkers, one per compute rank, each with its own 8-rung ladder per
leaf. On the 9mo (job 773, rows 840-1007) the loud EMRIs' posteriors are a
COMB of discrete islands along their intrinsic degeneracy (EMRI 3: p0 12.3307
/ 12.3310 / 12.3329 / 12.3346 / 12.3371 with a, logm1, m2, e0 sliding along,
each island within ~10 nats of the others); every walker's ladder holds one
or two islands and no single-axis Gaussian step, at any width, crosses from
one to the next. The inter-island vectors are, however, already present in
the ensemble: they are the differences between walkers. This move proposes

    x' = x + gamma (x_a - x_b)

for a row ``x`` of an ACTIVE walker, with ``x_a``, ``x_b`` an ordered pair of
distinct states (any rungs) drawn uniformly from a FIXED partner pool, the
start-of-propose rung ensembles of the OTHER walker group, and ``gamma`` = 1
with probability ``mode_frac`` (the mode jump: a point near ``x_b``'s island
lands near ``x_a``'s) and otherwise ``2.38 / sqrt(2 d) x N(1, gamma_sigma)``
(the differential-evolution random-walk scale). Periodic columns of the
difference are wrapped to ``[-P/2, P/2)`` and the proposal is wrapped back
into ``[0, P)``. Rows of inactive walkers take the ``fallback`` move's
proposal (the eigen move of the stack), so a walker outside the active group
runs its ordinary stack.

Exactness (the argument the add/remove move's scheduling relies on):

* The pool is fixed for the whole propose and the pair is drawn uniformly
  over ordered pairs, so ``(a, b)`` and ``(b, a)`` are equally likely and the
  difference ``d`` has a symmetric law; ``gamma``'s law does not depend on
  the point. Hence ``q(x -> x') = q(x' -> x)`` (with the periodic wrap: a
  translation on the torus), the proposal factor is 0 and the Metropolis
  ratio is the tempered posterior ratio alone.
* The pool must not depend on the state being updated. The add/remove move
  splits the walkers into two groups by the parity of their global index and
  alternates the roles by global iteration: in one propose the active group
  A draws its partners from group B's START-of-propose states while B runs
  its ordinary stack, which never reads A. The joint kernel of the propose
  is then ``K(x_A, x_B -> x_A', x_B') = K_A(x_A -> x_A' | x_B) K_B(x_B ->
  x_B')`` with ``K_A`` reversible for ``pi(x_A)`` at every fixed ``x_B`` and
  ``K_B`` invariant for ``pi(x_B)``; integrating
  ``pi(x_A) pi(x_B) K_A K_B`` over ``(x_A, x_B)`` gives
  ``pi(x_A') pi(x_B')``, so the product target (each walker its own
  conditional posterior) is invariant. Swapping the roles at the next
  iteration composes two invariant kernels.
* The parity schedule depends only on the iteration stamps, never on the
  chain, so the alternation itself introduces no state dependence.

``mode_frac`` and the stack weight are the branch knobs ``{BRANCH}_DE_JUMP_MODE_FRAC``
(default 0.5) and ``{BRANCH}_DE_JUMP_MIX`` (default 0 = not in the stack);
see :func:`~lisatools.globalfit.stock.erebor.common.make_default_inner_moves`,
which builds this move with the stack's eigen move as its ``fallback`` and
gives the eigen move the remaining ``1 - mix`` weight. The 9mo / 12mo
launchers set ``EMRI_DE_JUMP_MIX=0.2``, ``MBH_DE_JUMP_MIX=0.1``,
``SOBBH_DE_JUMP_MIX=0.2`` and ``EMRI_DE_JUMP_MODE_FRAC=0.5`` (the MBH and
SOBBH mode fractions stay at the 0.5 code default; ``gamma_sigma`` is never
set and stays 0.1). The walker-group split, the partner pool and the
per-split installation (:meth:`WalkerDifferenceJumpMove.set_partners`) live
in :class:`~lisatools.globalfit.moves.addremovemove.ResidualAddOneRemoveOneMove`.
"""

from __future__ import annotations

import numpy as np
from eryn.moves.mh import MHMove

__all__ = ["WalkerDifferenceJumpMove", "DE_GAMMA_SCALE"]

#: the differential-evolution random-walk scale 2.38 / sqrt(2 d) (ter Braak 2006)
DE_GAMMA_SCALE = 2.38

#: ``last_kind`` codes per proposed row
KIND_FALLBACK, KIND_MODE, KIND_SCALED = 0, 1, 2


def _xp_of(arr):
    """Return ``cupy`` when ``arr`` is a ``cupy.ndarray`` (and cupy imports), else ``numpy``."""
    try:
        import cupy as cp  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return np
    return cp if isinstance(arr, cp.ndarray) else np


class WalkerDifferenceJumpMove(MHMove):
    """Symmetric cross-walker difference jump; inactive rows take ``fallback``.

    The add/remove move installs, per leaf visit, the active rows and the
    partner pool with :meth:`set_partners` (:meth:`clear_partners` = every row
    takes the fallback). After each :meth:`get_proposal` the per-row
    bookkeeping is in ``last_kind[name]`` (0 fallback, 1 mode jump, 2 scaled
    jump) and ``last_pick[name]`` (the fallback's picked eigen axis, -1 for a
    jump row, -2 when the fallback does not report one). Setting
    :attr:`periodic` also hands the periodic container to the fallback when
    the fallback has none.

    Args:
        fallback (eryn.moves.Move): The move whose proposal the rows of
            INACTIVE walkers take (the stack's eigen move,
            :class:`~eryn.moves.EigenAxisMove` or
            :class:`~lisatools.sampling.moves.eigenjumpmix.EigenAxisJumpMixMove`;
            its tables are installed by the add/remove move as for any eigen
            inner move). Required; it must have a ``get_proposal`` method.
        mode_frac (float, optional): Probability, per active row, of
            ``gamma = 1`` (the mode jump); in ``[0, 1]``. Default ``0.5``.
        gamma_sigma (float, optional): Relative spread of the scaled
            ``gamma``, ``2.38 / sqrt(2 d) x N(1, gamma_sigma)``; finite and
            ``>= 0``. Default ``0.1``.
        **kwargs: Forwarded to :class:`eryn.moves.MHMove` (``periodic``, ...).

    Raises:
        ValueError: ``fallback`` is ``None`` or has no callable
            ``get_proposal``, ``mode_frac`` is outside ``[0, 1]``, or
            ``gamma_sigma`` is negative or not finite.

    Attributes:
        fallback (eryn.moves.Move): The fallback move.
        mode_frac (float): The mode-jump probability.
        gamma_sigma (float): The relative spread of the scaled ``gamma``.
        last_kind (dict): ``{branch: np.ndarray}`` of per-row kind codes
            (``KIND_FALLBACK`` 0, ``KIND_MODE`` 1, ``KIND_SCALED`` 2), in the row
            order of the alive leaves.
        last_pick (dict): ``{branch: np.ndarray}`` of per-row eigen axes (see
            above).

    Example:
        >>> import numpy as np
        >>> from eryn.moves import EigenAxisMove
        >>> from lisatools.sampling.moves.walkerdiffjump import WalkerDifferenceJumpMove
        >>> move = WalkerDifferenceJumpMove(EigenAxisMove(mode="axis"), mode_frac=0.5)
        >>> move.describe()
        'mode frac 0.50, fallback EigenAxisMove'
        >>> active = np.array([[True, False]])
        >>> move.set_partners("emri", active, np.random.rand(4, 3))
        >>> move.partner_spec("emri")[1].shape
        (4, 3)
    """

    def __init__(self, fallback, mode_frac=0.5, gamma_sigma=0.1, **kwargs):
        if fallback is None or not callable(getattr(fallback, "get_proposal", None)):
            raise ValueError("WalkerDifferenceJumpMove needs a fallback move with get_proposal().")
        mode_frac = float(mode_frac)
        gamma_sigma = float(gamma_sigma)
        if not (0.0 <= mode_frac <= 1.0):
            raise ValueError(f"mode_frac must be in [0, 1], got {mode_frac}.")
        if not (gamma_sigma >= 0.0 and np.isfinite(gamma_sigma)):
            raise ValueError(f"gamma_sigma must be finite and >= 0, got {gamma_sigma}.")
        self.fallback = fallback
        self.mode_frac = mode_frac
        self.gamma_sigma = gamma_sigma
        self._partners = {}
        self.last_kind = {}
        self.last_pick = {}
        super().__init__(**kwargs)

    # ---- periodicity reaches the fallback too ---------------------------------
    @property
    def periodic(self):
        """eryn.utils.PeriodicContainer or None: Periodic container (passed on to the fallback)."""
        return getattr(self, "_periodic", None)

    @periodic.setter
    def periodic(self, periodic):
        self._periodic = periodic
        fb = getattr(self, "fallback", None)
        if periodic is not None and fb is not None and getattr(fb, "periodic", None) is None:
            fb.periodic = periodic

    def _periodic_cols(self, name):
        """Return the periodic column indices and periods of branch ``name``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(cols, periods)``, int and float
            arrays of equal length; both empty when the branch has no periodic
            columns or no periodic container is set.
        """
        per = self.periodic
        inds = getattr(per, "inds_periodic", None)
        periods = getattr(per, "periods", None)
        if isinstance(inds, dict) and isinstance(periods, dict) and name in inds:
            return (np.asarray(inds[name], dtype=int).ravel(),
                    np.asarray(periods[name], dtype=float).ravel())
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    # ---- the partner pool -------------------------------------------------------
    def set_partners(self, name, active, partners):
        """Install branch ``name``'s jump spec for the coming proposals.

        Both arrays are copied to the host; the spec stays frozen until the
        next :meth:`set_partners` or :meth:`clear_partners` for ``name``.

        Args:
            name (str): Branch name.
            active (array-like): Boolean ``(ntemps, nwalkers)`` over the
                (temperature, walker) rows of the coordinates the next
                :meth:`get_proposal` sees (``True`` = every alive leaf of that
                row jumps).
            partners (array-like): ``(P, ndim)`` partner states, ``P >= 2``, in
                the same coordinates as the proposed rows.

        Raises:
            ValueError: ``partners`` is not 2-D or has fewer than two rows.
        """
        partners = np.array(partners, dtype=float, copy=True)
        if partners.ndim != 2 or partners.shape[0] < 2:
            raise ValueError(f"partners must be (P >= 2, ndim), got shape {partners.shape}.")
        self._partners[name] = (np.asarray(active, dtype=bool).copy(), partners)

    def clear_partners(self, name=None):
        """Remove the jump spec, so every row takes the fallback.

        Args:
            name (str, optional): Branch name; ``None`` (default) clears every
                branch.
        """
        if name is None:
            self._partners = {}
        else:
            self._partners.pop(name, None)

    def partner_spec(self, name):
        """Return the installed spec of branch ``name``.

        Args:
            name (str): Branch name.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: ``(active, partners)`` as
            stored by :meth:`set_partners`, or ``None`` when none is installed.
        """
        return self._partners.get(name)

    # ---- the proposal ---------------------------------------------------------
    def _draws(self, random, n, npart, ndim):
        """Draw the partner pair and the jump scale of ``n`` rows.

        Args:
            random (numpy.random.Generator or numpy.random.RandomState): Host
                random stream.
            n (int): Number of jumping rows (may be 0; the calls are made anyway).
            npart (int): Size ``P`` of the partner pool.
            ndim (int): Parameters per row (sets ``2.38 / sqrt(2 ndim)``).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ``(i, j, gamma, mode)``, each of shape ``(n,)``: an ordered pair of
            distinct partner indices drawn uniformly, the jump scale, and the
            boolean mode-jump flag (``gamma == 1``).
        """
        if hasattr(random, "integers"):          # numpy Generator
            i = np.asarray(random.integers(npart, size=n))
            j = np.asarray(random.integers(npart - 1, size=n))
            u = np.asarray(random.random(n))
            z = np.asarray(random.standard_normal(n))
        else:                                    # legacy RandomState
            i = np.asarray(random.randint(0, npart, n))
            j = np.asarray(random.randint(0, npart - 1, n))
            u = np.asarray(random.rand(n))
            z = np.asarray(random.randn(n))
        j = j + (j >= i)                         # uniform over the npart - 1 others
        mode = u < self.mode_frac
        g0 = DE_GAMMA_SCALE / np.sqrt(2.0 * max(int(ndim), 1))
        gamma = np.where(mode, 1.0, g0 * (1.0 + self.gamma_sigma * z))
        return i, j, gamma, mode

    def get_proposal(self, branches_coords, random, branches_inds=None,
                     supps=None, branch_supps=None, **kwargs):
        """Return the fallback's proposal, with the active rows replaced by jumps.

        See the module docstring for the proposal and the exactness argument.
        The fallback always runs first on every row; then, per branch with an
        installed spec, every alive leaf of an active ``(temperature, walker)``
        row is moved from its CURRENT point (``branches_coords``, not the
        fallback's proposal) by ``gamma (x_a - x_b)``. Fills
        :attr:`last_kind` and :attr:`last_pick`.

        Args:
            branches_coords (dict): ``{branch: xp.ndarray}`` of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (numpy.random.Generator or numpy.random.RandomState): Host
                random stream.
            branches_inds (dict, optional): ``{branch: bool array}`` of shape
                ``(ntemps, nwalkers, nleaves_max)``; ``None`` means every leaf.
            supps (optional): Forwarded to the fallback.
            branch_supps (dict, optional): Forwarded to the fallback.
            **kwargs: Forwarded to the fallback's ``get_proposal``.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: the proposed coordinates
            (the fallback's array module) and the ``(ntemps, nwalkers)`` factors,
            the fallback's on its rows and 0 on the jump rows (the jump is
            symmetric).

        Raises:
            ValueError: An installed ``active`` mask does not match
                ``(ntemps, nwalkers)``, or the partners' ``ndim`` differs from
                the branch's.
        """
        q, factors = self.fallback.get_proposal(
            branches_coords, random, branches_inds=branches_inds, supps=supps,
            branch_supps=branch_supps, **kwargs)
        factors = np.array(factors, dtype=float, copy=True)
        fb_pick = dict(getattr(self.fallback, "last_pick", None) or {})
        self.last_kind = {}
        self.last_pick = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = np.asarray(branches_inds[name], dtype=bool)
            t_i, w_i, l_i = np.nonzero(inds)
            nrows = t_i.size
            kind = np.zeros(nrows, dtype=int)
            pick = np.asarray(fb_pick.get(name, np.full(nrows, -2)), dtype=int).ravel()
            if pick.size != nrows:
                pick = np.full(nrows, -2, dtype=int)
            spec = self._partners.get(name)
            rows = np.zeros(0, dtype=int)
            npart = 2
            if spec is not None and nrows:
                active, partners = spec
                if active.shape != (ntemps, nwalkers):
                    raise ValueError(
                        f"active mask {active.shape} does not match the proposal rows "
                        f"{(ntemps, nwalkers)} of branch {name!r}.")
                if partners.shape[1] != ndim:
                    raise ValueError(f"partners ndim {partners.shape[1]} != {ndim}.")
                rows = np.flatnonzero(active[t_i, w_i])
                npart = partners.shape[0]
            # ONE draw call per branch and proposal whether or not a row jumps
            # here, so a body holding only inactive walkers makes the same
            # calls as one holding active ones (a per-call random stream then
            # lines up across walker-block layouts)
            i, j, gamma, mode = self._draws(random, rows.size, npart, ndim)
            if rows.size:
                d = partners[i] - partners[j]
                cols, per = self._periodic_cols(name)
                if cols.size:
                    d[:, cols] = np.mod(d[:, cols] + 0.5 * per, per) - 0.5 * per
                xp = _xp_of(q[name])
                sel = (t_i[rows], w_i[rows], l_i[rows])
                x = np.asarray(coords[sel] if xp is np else coords[sel].get(), dtype=float)
                new = x + gamma[:, None] * d
                if cols.size:
                    new[:, cols] = np.mod(new[:, cols], per)
                q[name][sel] = new if xp is np else xp.asarray(new)
                factors[t_i[rows], w_i[rows]] = 0.0
                kind[rows] = np.where(mode, KIND_MODE, KIND_SCALED)
                pick[rows] = -1
            self.last_kind[name] = kind
            self.last_pick[name] = pick
        return q, factors

    def describe(self) -> str:
        """Return the inner-move stack log suffix.

        Returns:
            str: e.g. ``"mode frac 0.50, fallback EigenAxisJumpMixMove"``.
        """
        return f"mode frac {self.mode_frac:.2f}, fallback {type(self.fallback).__name__}"
