"""Scatter independent scoring rows over compute ranks; gather in row order.

Two routing modes, picked by whether the caller names each row's walker.

**Unrouted** (``walkers=None``) -- every compute rank can serve every row,
which is true exactly when they all hold the same walkers (one-walker
replica mode). Rows split into contiguous chunks in compute-rank order via
:func:`numpy.array_split` sizes. This is the original behaviour and is kept
byte-for-byte.

**Walker-routed** (``walkers=`` global indices) -- a row scoring walker *w*
may only be served by a rank whose block CONTAINS *w*; any other rank's
``AnalysisContainerArray`` simply has no such row. Rows are grouped by
walker, each walker's rows split contiguously over that walker's group, and
every shipped row's ``data_index`` rewritten from the GLOBAL walker index to
the BLOCK-LOCAL row index the serving rank will use. Replies are scattered
back to their original positions, so the caller always sees row order.

:meth:`RowFanout.replay` broadcasts a state-mutating step (expose a source,
publish a noise model) so every replica applies it, head included. With one
compute rank both are direct local calls -- no comm, no pickling.

Rank side: ``ComputeService`` routes op ``x`` to ``move.gf_serve("x", ...)``;
the payload is ``{"rows": {...}}`` for ``run`` and the raw payload for
``replay``. A ``run`` body must return EVERY key for an empty chunk too
(zero-length arrays) so the concatenation stays well-defined.

Where it runs: :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin`
installs a :class:`RowFanout` on a move only in replica mode
(``layout.replica_mode``, ``R > 1``): one-walker replica mode, or
``GF_GPU_ROUTING=1`` with more compute ranks than walker blocks. The 9mo/12mo
jobs at ``NGPUS=4`` (``R == 1``) do not use it. The scoring seams in
``moves/addremovemove.py`` and ``moves/psdmove.py`` call
:meth:`RowFanout.run` / :meth:`RowFanout.replay`.
"""

import numpy as np

__all__ = ["RowFanout"]


class RowFanout:
    """Scatter scoring rows of one move over the compute ranks (head side).

    Args:
        fanout (WalkerFanout or None): The head's fan-out; ``None`` makes
            every call local.
        move: The owning move; its ``gf_move_name`` is the registry name the
            commands are sent under.

    Attributes:
        fanout (WalkerFanout or None): The fan-out.
        move: The owning move.
    """

    def __init__(self, fanout, move):
        self.fanout = fanout
        self.move = move

    @property
    def active(self) -> bool:
        """``True`` when there is a multi-rank fan-out (otherwise calls are local)."""
        return self.fanout is not None and not self.fanout.single

    @property
    def move_name(self):
        """The owning move's ``gf_move_name`` (``None`` if it has none)."""
        return getattr(self.move, "gf_move_name", None)

    @staticmethod
    def _bounds(n, n_chunks):
        """Chunk boundaries of :func:`numpy.array_split` sizes, length ``n_chunks + 1``."""
        sizes = [len(c) for c in np.array_split(np.arange(int(n)), int(n_chunks))]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def _plan(self, n, walkers):
        """``{rank: (row positions, local data_index per position)}``.

        Walker-routed: for each walker, its rows are split contiguously over
        the ranks of ITS block, so a rank only ever receives rows it can
        actually score, and the row's ACA index is translated on the way.
        """
        layout = self.fanout.layout
        walkers = np.asarray(walkers).reshape(-1).astype(int)
        if walkers.shape[0] != int(n):
            raise ValueError(
                f"RowFanout.run: walkers has {walkers.shape[0]} entries for {n} rows")
        plan = {}
        for w in np.unique(walkers):
            pos = np.nonzero(walkers == int(w))[0]
            ranks, local = layout.owners_of(int(w))
            bounds = self._bounds(pos.size, len(ranks))
            for i, r in enumerate(ranks):
                s, e = int(bounds[i]), int(bounds[i + 1])
                if e <= s:
                    continue
                prev_pos, prev_loc = plan.get(r, (None, None))
                take = pos[s:e]
                loc = np.full(take.size, int(local), dtype=int)
                plan[r] = ((take, loc) if prev_pos is None else
                           (np.concatenate([prev_pos, take]),
                            np.concatenate([prev_loc, loc])))
        return plan

    def run(self, op, rows, *, local_body, walkers=None):
        """``{name: (N, ...)}`` -> ``{name: (N, ...)}``, rows evaluated over compute ranks.

        ``walkers``: optional GLOBAL walker index per row. Given, the split
        is walker-routed (see the module docstring); omitted, it is the
        original contiguous split over every compute rank.

        At one walker block the contiguous split is used even when
        ``walkers`` is given (every rank holds every walker).

        Args:
            op (str): The ``gf_serve`` op the ranks run.
            rows (dict[str, array-like]): Row arrays, all with the same
                leading length ``N``. A ``"data_index"`` entry is rewritten
                to block-local rows on the walker-routed path.
            local_body (Callable[[dict], dict]): ``fn(rows) -> {name: array}``
                run on the head's chunk (and directly when inactive).
            walkers (array-like, optional): Global walker index per row,
                shape ``(N,)``.

        Returns:
            dict[str, np.ndarray]: Every key of the body's result,
            concatenated / scattered back to row order, leading length ``N``.

        Raises:
            ValueError: When ``rows`` is empty, the arrays disagree on
                length, or ``walkers`` has the wrong length.
        """
        rows = {k: np.asarray(v) for k, v in rows.items()}
        if not rows:
            raise ValueError("RowFanout.run: no row arrays given")
        lengths = {k: int(v.shape[0]) for k, v in rows.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"RowFanout.run: row arrays disagree on length: {lengths}")
        n = next(iter(lengths.values()))
        if not self.active:
            return local_body(rows)
        layout = self.fanout.layout
        # Routing is only NEEDED when blocks differ; at one block every rank
        # holds every walker and the cheaper contiguous split is equivalent.
        if walkers is None or int(getattr(layout, "n_blocks", 1)) <= 1:
            return self._run_contiguous(op, rows, n, local_body)
        return self._run_routed(op, rows, n, local_body, walkers)

    # -- the original path, unchanged ------------------------------------
    def _run_contiguous(self, op, rows, n, local_body):
        """Contiguous split over every compute rank, concatenated in compute-rank order."""
        layout = self.fanout.layout
        bounds = self._bounds(n, layout.n_compute)

        def chunk(rank, _w0, _w1):
            i = layout.fanout_rank(rank)
            s, e = int(bounds[i]), int(bounds[i + 1])
            return {"rows": {k: v[s:e] for k, v in rows.items()}}

        def body(payload, _model):
            return local_body(payload["rows"])

        def merge(results):
            keys = list(results[layout.head_rank].keys())
            return {
                k: np.concatenate(
                    [np.asarray(results[r][k]) for r in layout.compute_ranks], axis=0
                )
                for k in keys
            }

        return self.fanout.run(
            op, move=self.move_name, per_rank_payload=chunk, local_body=body, merge=merge
        )

    # -- walker-routed ----------------------------------------------------
    def _run_routed(self, op, rows, n, local_body, walkers):
        """Walker-routed split (:meth:`_plan`), results scattered back to row order."""
        layout = self.fanout.layout
        plan = self._plan(n, walkers)

        def chunk(rank, _w0, _w1):
            pos, loc = plan.get(rank, (np.zeros(0, int), np.zeros(0, int)))
            out = {k: v[pos] for k, v in rows.items()}
            if "data_index" in out:
                # GLOBAL walker -> BLOCK-LOCAL ACA row. Shipping the global
                # index would read past the end of a narrower block's ACA.
                out["data_index"] = loc.astype(
                    np.asarray(rows["data_index"]).dtype, copy=False)
            return {"rows": out}

        def body(payload, _model):
            return local_body(payload["rows"])

        def merge(results):
            head = results[layout.head_rank]
            out = {}
            for k in list(head.keys()):
                sample = np.asarray(head[k])
                shape = (n,) + tuple(sample.shape[1:])
                buf = np.empty(shape, dtype=sample.dtype)
                for r in layout.compute_ranks:
                    pos, _loc = plan.get(r, (np.zeros(0, int), None))
                    if pos.size:
                        buf[pos] = np.asarray(results[r][k])
                out[k] = buf
            return out

        return self.fanout.run(
            op, move=self.move_name, per_rank_payload=chunk, local_body=body, merge=merge
        )

    def replay(self, op, payload, *, local_body):
        """Apply ``payload`` on every compute rank (the head through ``local_body``).

        Args:
            op (str): The ``gf_serve`` op the other ranks run with the same
                payload.
            payload: Sent unchanged to every rank.
            local_body (Callable[[object], None]): Applies it on the head.
        """
        if not self.active:
            local_body(payload)
            return
        self.fanout.run(
            op,
            move=self.move_name,
            per_rank_payload=lambda _rank, _w0, _w1: payload,
            local_body=lambda p, _model: local_body(p),
            merge=lambda _results: None,
        )
