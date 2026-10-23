"""Search LEGS: one stored row per leg of the search cycle.

User design 2026-09-30. A search stage's moves run in a fixed order; with
legs on, ONE sampler iteration runs from the cursor up to and including the
next LEG-ENDER (every ``in_model*`` move by default) and eryn then saves a
row. So the store holds the state after every in-model polish -- after
``in_model``, after ``in_model_fstat``, after ``in_model_removal`` -- instead
of one row per full cycle, and a preemption loses at most one leg.

The resume position is a NAME, never a count: the state carries the name of
the leg-ender it was saved after, the saver writes it into the row
(``global_fit/saved_after``) and stamps the stage's ordered move list on the
recipe group. At startup the step reads the last row's name, finds it in
the live list and sets the cursor to the move after it. A gated move
(``Move(every=N)``, the source cadence, replace on or off) that skipped a
cycle can therefore never mis-align the resume, because only leg-enders are
ever saved after and leg-enders are never gated. Cadences and the galfor
ratchet count CYCLES (cursor wraps), not rows.

Composition: ``GB_SEARCH_LEGS=1`` (launcher) puts ``leg_ends="auto"`` on the
numbered search stages' combine and ``legs=True`` on their recipe step. Off
(default) is today's one row per cycle, byte-identical.

Where the pieces live. The driver (``scripts/fstat_proposal/run_combined_staged.py``)
passes ``combine_kwargs=dict(leg_ends="auto")`` and ``step_kwargs=dict(legs=True)``
to ``Stage(kind="gb_search")``; :class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove`
resolves the names with :func:`leg_ends_from_names` and keeps a :class:`LegCursor`
as ``gf_legs`` (one leg per propose); on stage entry
:meth:`~lisatools.globalfit.recipe.SearchStageProfileStep._legs_enter` positions
that cursor from the store (``saved_after``, ``saved_after_history``,
``stage_move_order`` on the
:class:`~lisatools.globalfit.hdfbackend.GFHDFBackend`). The 9mo / 12mo
production runs export ``GB_SEARCH_LEGS=1``: ``gb_search_1`` and
``gb_search_2`` store one row per leg; ``gb_search_seed`` runs with
``legs=False`` (one row per cycle).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = ["LegCursor", "leg_ends_from_names", "DEFAULT_LEG_END_PREFIX"]

#: Every move whose name starts with this ends a leg, by default.
DEFAULT_LEG_END_PREFIX = "in_model"


def leg_ends_from_names(order: Sequence[str], ends=None) -> List[str]:
    """The leg-enders of a stage, in list order.

    ``ends`` None or ``"auto"``: every name starting with ``in_model``.
    An explicit iterable is validated against ``order`` (a misspelled or
    absent leg-ender would silently merge two legs, which is exactly the
    failure this refuses).

    Args:
        order: The stage's move names, in cycle order.
        ends (str | Iterable[str], optional): ``None`` / ``"auto"`` for every
            name starting with :data:`DEFAULT_LEG_END_PREFIX`, or an explicit
            list of leg-ender names.

    Returns:
        list[str]: The leg-enders, in the order they appear in ``order``.

    Raises:
        ValueError: ``"auto"`` with no ``in_model*`` move in ``order``; an
            explicit name not in ``order``; an empty explicit list.

    Example:
        >>> from lisatools.globalfit.legs import leg_ends_from_names
        >>> order = ["rj_warm_search", "in_model", "rj_fstat_search",
        ...          "in_model_fstat", "mbh_pe"]
        >>> leg_ends_from_names(order)
        ['in_model', 'in_model_fstat']
        >>> leg_ends_from_names(order, ["in_model_fstat"])
        ['in_model_fstat']
    """
    order = list(order)
    if ends is None or ends == "auto":
        out = [n for n in order if str(n).startswith(DEFAULT_LEG_END_PREFIX)]
        if not out:
            raise ValueError(
                f"no {DEFAULT_LEG_END_PREFIX}* move in {order}: a stage without "
                "an in-model move has no legs to end (drop leg_ends for it).")
        return out
    ends = [str(e) for e in ends]
    missing = [e for e in ends if e not in order]
    if missing:
        raise ValueError(f"leg_ends {missing} are not in the move order {order}.")
    if not ends:
        raise ValueError("leg_ends is empty.")
    return [n for n in order if n in set(ends)]


class LegCursor:
    """A cursor over an ordered move list, advancing one leg per propose.

    A leg runs from the cursor up to and including the next leg-ender; the
    LAST leg also runs the tail after the last leg-ender. Owned by the stage
    combine (:class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove`,
    attribute ``gf_legs``), which calls :meth:`plan` / :meth:`advance` once
    per propose; positioned on resume by
    :meth:`~lisatools.globalfit.recipe.SearchStageProfileStep._legs_enter`
    through :meth:`set_after` and :meth:`cycles_from_history`.

    Args:
        order: The stage's move names, in cycle order.
        ends: The leg-ender names, already resolved (for example by
            :func:`leg_ends_from_names`). Validated against ``order``. Pass a
            list: the string ``"auto"`` is NOT resolved here (it would be
            split into characters and rejected).

    Attributes:
        order: the stage's move names in order.
        ends: the leg-ender names (a subset of ``order``, in order).
        cursor: index into ``order`` of the next move to run.
        cycles: completed cycles (cursor wraps) since the stage started.

    Example:
        >>> from lisatools.globalfit.legs import LegCursor, leg_ends_from_names
        >>> order = ["rj_warm_search", "in_model", "rj_fstat_search",
        ...          "in_model_fstat", "mbh_pe"]
        >>> cur = LegCursor(order, leg_ends_from_names(order))
        >>> cur.plan()
        ([0, 1], 'in_model', False)
        >>> cur.advance()
        >>> cur.plan()
        ([2, 3, 4], 'in_model_fstat', True)
        >>> cur.advance()
        >>> cur.cycles
        1
    """

    def __init__(self, order: Sequence[str], ends: Iterable[str]):
        self.order = [str(n) for n in order]
        self.ends = leg_ends_from_names(self.order, list(ends))
        self._end_idx = [self.order.index(e) for e in self.ends]
        self.cursor = 0
        self.cycles = 0

    @property
    def nlegs(self) -> int:
        """Number of legs per cycle (= number of leg-enders)."""
        return len(self.ends)

    @property
    def leg_index(self) -> int:
        """Which leg (0-based) the cursor is in."""
        return sum(1 for i in self._end_idx if i < self.cursor)

    def plan(self) -> Tuple[List[int], str, bool]:
        """``(indices, end_name, is_last_leg)`` for the leg starting at the cursor.

        The LAST leg runs through the tail of the list (the moves after the
        last leg-ender -- ridge Gibbs, the source moves -- belong to it), so
        the row saved after it holds the state after the whole cycle. A
        cursor at or past the end of ``order`` is first wrapped to 0.

        Returns:
            tuple[list[int], str, bool]: Indices into ``order`` to run in this
            propose, the name the row will be saved after, and whether this
            is the last leg of the cycle.
        """
        n = len(self.order)
        if self.cursor >= n:
            self.cursor = 0
        later = [i for i in self._end_idx if i >= self.cursor]
        if not later:
            # cursor sits past the last leg-ender: finish the tail as the
            # last leg (only reachable by a hand-set cursor)
            return list(range(self.cursor, n)), self.ends[-1], True
        end_i = later[0]
        last = end_i == self._end_idx[-1]
        stop = n if last else end_i + 1
        return list(range(self.cursor, stop)), self.order[end_i], last

    def advance(self) -> None:
        """Move past the leg just run; wrapping counts a completed cycle."""
        idx, _, last = self.plan()
        self.cursor = idx[-1] + 1
        if last or self.cursor >= len(self.order):
            self.cursor = 0
            self.cycles += 1

    def set_after(self, name: Optional[str]) -> bool:
        """Position the cursor after move ``name`` (a resume).

        ``name`` is normally a leg-ender, but a move that ended its leg
        EARLY (see :meth:`end_early_at`; the gated noise head when it changed
        the noise) is a valid saved-after name too. Returns False and starts
        at the head when ``name`` is None or not in this list -- an old
        store, or a changed composition. A row saved after the LAST
        leg-ender also resumes at the head (cursor 0; that cycle is
        complete). Does not change :attr:`cycles`.

        Args:
            name: The ``saved_after`` name of the last stored row of this
                stage.

        Returns:
            bool: True when ``name`` was found and the cursor positioned
            after it.
        """
        if name is None or name not in self.order:
            self.cursor = 0
            return False
        if name == self.ends[-1]:
            self.cursor = 0
            return True
        self.cursor = (self.order.index(name) + 1) % len(self.order)
        return True

    # TODO check: keep/remove/document -- only via gf_leg_end_now, set only by NoiseRatchetGate (GALFOR_RATCHET=0 in 9mo/12mo)
    def end_early_at(self, index: int) -> None:
        """End the current leg right after move ``index`` (before its static
        leg-ender): the row is saved after THAT move and the rest of the leg
        runs in the next propose. Only a move past the last static leg-ender
        (the tail) wraps the cycle.

        Called by :class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove`
        when a move in the leg sets ``gf_leg_end_now`` on itself (the gated
        noise head of the galfor ratchet).

        Args:
            index: Index into :attr:`order` of the move the row is saved after.
        """
        index = int(index)
        self.cursor = index + 1
        if index >= self._end_idx[-1] or self.cursor >= len(self.order):
            self.cursor = 0
            self.cycles += 1

    def cycles_from_history(self, names: Iterable[Optional[str]]) -> int:
        """Completed cycles = rows saved after the LAST leg-ender.

        Args:
            names: The ``saved_after`` names of this stage's stored rows
                (``GFHDFBackend.saved_after_history``).

        Returns:
            int: How many of them equal the last leg-ender.
        """
        last = self.ends[-1]
        return sum(1 for n in names if n == last)

    def __repr__(self):
        return (f"LegCursor(cursor={self.cursor} -> "
                f"{self.order[self.cursor] if self.cursor < len(self.order) else '<end>'!r}, "
                f"cycles={self.cycles}, ends={self.ends})")
