"""Head-directed fan-out over walker blocks + the compute-rank service loop.

Sampling phase: the head calls :meth:`WalkerFanout.run` inside a proposal;
each computation rank receives one command, runs the same body on its own
walker block and replies; the head merges. With ONE compute rank ``run`` is
a direct call (no communicator, no pickling, no copies). Setup phase:
:meth:`WalkerFanout.allgather_walker_vector` is a symmetric collective every
compute rank calls at the same program point.

Who runs what: ``GlobalFit`` builds a :class:`WalkerFanout` on every compute
rank (the head drives :meth:`WalkerFanout.run`; the others use it only for
the setup-phase collectives) and a :class:`ComputeService` on every compute
rank other than the head, all on the fan-out communicator (head + compute
ranks only,
:meth:`~lisatools.globalfit.communication.ranks.WalkerBlockLayout.make_fanout_comm`;
the saver is not a member). The non-head compute ranks sit in
:meth:`ComputeService.serve` for the whole sampling phase. Moves reach the
fan-out through :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin`
(rank side: ``move.gf_serve``).

Command / reply schemas (host numpy, pickled by mpi4py ``send``/``recv``)::

    cmd   = {"seq", "op", "move", "clock", "payload", "shared"}
    reply = {"seq", "rank", "ok", "result", "wall_s",
             "error": None | {"type", "msg", "traceback"}}

Command fields:

- ``seq`` (int): the head's fan-out counter, incremented once per
  multi-rank :meth:`WalkerFanout.run`. Every reply must echo it; a
  mismatch raises on the head.
- ``op`` (str): the operation. Builtins answered by :class:`ComputeService`
  itself: :data:`STOP_OP` (leave the loop, no reply), :data:`PING_OP`
  (layout digest), and the ones ``GlobalFit.prepare_compute`` installs:
  :data:`LIKELIHOOD_OP`, :data:`RESIDUAL_HASH_OP` and
  ``residual_snapshot.RESIDUAL_SNAPSHOT_OP``. Any other op is routed to
  ``move.gf_serve(op, payload, clock, model)`` of the move named below.
- ``move`` (str or None): the move name; together with ``clock["stage"]``
  it is the service registry key.
- ``clock`` (dict): a copy of :attr:`WalkerFanout.clock` with keys
  ``iteration``, ``stage``, ``stage_kind``, ``move``, ``call_index`` (per
  ``(move, op)`` call count), ``seq`` and ``seed_base`` (the run's seed
  base, so rank-side bodies can derive reproducible streams).
- ``payload``: the per-rank part, ``per_rank_payload(rank, w0, w1)``.
- ``shared``: the same object for every rank (``None`` by default).

Reply fields: ``seq`` (echo), ``rank`` (the replying rank, run-communicator
numbering), ``ok`` (bool), ``result`` (the body's return value, ``None`` on
failure), ``wall_s`` (seconds spent in :meth:`ComputeService.handle`) and
``error`` (``None``, or ``{"type", "msg", "traceback"}`` of the remote
exception; the head raises :class:`RemoteWorkerError` with it).

Transport: the head sends one command to every worker's fan-out rank
(``isend``), waits for the sends, runs its own block, then receives one
reply from each worker in compute-rank order (``recv``). Workers receive
from the head's fan-out rank and send the reply back. Each multi-rank ``run`` logs one
``[FANOUT] op=... move=... head_s=... max_rank_s=... wait_s=...`` INFO line
on this module's logger (the head's ``globalfit_run.log``).
"""

from __future__ import annotations

import collections
import hashlib
import logging
import time
import traceback

import numpy as np

# Module logger: propagates to the ``lisatools`` file handler, i.e. the head's
# ``globalfit_run.log`` -- the file the diagnostics (``summarize_fanout``) and
# the WP7 runbook read. The ``GlobalFit`` logger handed in at construction
# writes ``global_fit.log`` with propagation OFF, which is where the
# ``[FANOUT]`` load-balance line hid until 2026-09-16 (cluster Step 4).
_LOG = logging.getLogger(__name__)

from ...utils.utility import asnumpy

STOP_OP = "stop"
PING_OP = "ping"
#: builtin every ComputeService installs: the rank's current per-walker
#: likelihood over its block (see WalkerFanout.gather_likelihood).
LIKELIHOOD_OP = "likelihood"
#: builtin: sha1 of the rank's residual buffers (replica agreement check)
RESIDUAL_HASH_OP = "residual_hash"


class RemoteWorkerError(RuntimeError):
    """A computation rank's command failed; carries the remote traceback text.

    Raised on the head by :meth:`WalkerFanout.run` for the first failing
    reply (every failing reply's traceback is logged first).

    Args:
        rank (int): The failing rank.
        op (str): The operation that failed.
        move (str or None): The move name of the command.
        error (dict or None): The reply's ``error`` dict
            (``{"type", "msg", "traceback"}``).

    Attributes:
        rank (int): The failing rank.
        op (str): The operation.
        move (str or None): The move name.
        error (dict): A copy of the reply's ``error`` dict.
        remote_traceback (str): The remote traceback text (``""`` if absent).
    """

    def __init__(self, rank, op, move, error):
        self.rank = int(rank)
        self.op = op
        self.move = move
        self.error = dict(error or {})
        self.remote_traceback = self.error.get("traceback", "")
        super().__init__(
            f"rank {self.rank} failed op={op!r} move={move!r}: "
            f"{self.error.get('type')}: {self.error.get('msg')}\n{self.remote_traceback}"
        )


def concat_blocks(results, layout):
    """Concatenate per-BLOCK 1-D results in walker order.

    ONE representative per walker block (``replica_index == 0``), because
    the R ranks of a block hold the same walkers: concatenating all of them
    would repeat every walker R times, and taking the head alone would drop
    every block but its own. Both of the pre-2026-09-23 branches are special
    cases of this -- at ``R == 1`` the leads ARE the compute ranks, and at
    ``n_blocks == 1`` the only lead is the head.

    Args:
        results (dict): ``{rank: 1-D array-like}`` from a fan-out merge.
        layout (WalkerBlockLayout): The run layout (its ``block_leads``).

    Returns:
        np.ndarray: The leads' results concatenated in block order, shape
        ``(nwalkers,)`` for one value per walker.
    """
    leads = getattr(layout, "block_leads", None)
    if leads is None:  # pragma: no cover - stub layouts in older tests
        leads = (layout.head_rank,) if getattr(layout, "replica_mode", False) \
            else layout.compute_ranks
    return np.concatenate([np.atleast_1d(np.asarray(results[r])) for r in leads])


def _reply(seq, rank, ok, result, wall_s, error=None):
    """Build one reply dict (schema in the module docstring)."""
    return {"seq": seq, "rank": rank, "ok": ok, "result": result, "wall_s": wall_s, "error": error}


def _sha1_16(data: bytes) -> str:
    """First 16 hex digits of the sha1 of ``data``."""
    return hashlib.sha1(data).hexdigest()[:16]


def _array_bytes(arr) -> bytes:
    """Host bytes for one array (device arrays pulled via ``asnumpy``)."""
    return np.ascontiguousarray(asnumpy(arr)).tobytes()


def residual_hash(acs) -> str:
    """16-hex sha1 of the ACA's residual AND noise-model buffers.

    Hashes ``linear_data_arr`` (the residual) *and* ``linear_psd_arr`` (the
    noise covariance) when present, so a PSD-only drift between replicas is
    caught too, not just a residual drift; falls back to the likelihood
    vector when neither buffer is available.

    Used in replica mode (``R > 1``: one-walker replica mode, or
    ``GF_GPU_ROUTING=1`` with more compute ranks than walker blocks): by the
    GB replica sync and by the :data:`RESIDUAL_HASH_OP` builtin.

    Args:
        acs: The rank's ``AnalysisContainerArray``.

    Returns:
        str: 16 hex digits.
    """
    arrs = (
        list(getattr(acs, "linear_data_arr", []) or [])
        + list(getattr(acs, "linear_psd_arr", []) or [])
    )
    if not arrs:
        arrs = [acs.likelihood(complex=False)]
    return _sha1_16(b"".join(_array_bytes(a) for a in arrs))


def replicas_agree(residual_hashes, layout=None) -> bool:
    """Do the replicas of every walker block hold the same residual?

    PER BLOCK: under ``n_compute = n_blocks x R`` different blocks hold
    different walkers, so their hashes differ by construction and a
    whole-world comparison would read ``False`` on every line. Without a
    layout (or at one block) every rank holds the same walker and the
    whole map must agree -- the original one-walker replica rule.

    Args:
        residual_hashes (dict or None): ``{rank: hash}`` from
            :meth:`WalkerFanout.gather_residual_hashes`.
        layout (WalkerBlockLayout, optional): Enables the per-block
            comparison.

    Returns:
        bool: ``True`` when every block's hashes agree (or no hashes).
    """
    hashes = {int(r): h for r, h in (residual_hashes or {}).items()}
    if not hashes:
        return True
    n_blocks = int(getattr(layout, "n_blocks", 1) or 1) if layout is not None else 1
    if n_blocks <= 1 or not hasattr(layout, "ranks_in_block"):
        return len(set(hashes.values())) == 1
    for b in range(n_blocks):
        block = {hashes[int(r)] for r in layout.ranks_in_block(b) if int(r) in hashes}
        if len(block) > 1:
            return False
    return True


def fanout_digest_line(iteration, state, residual_hashes=None, layout=None) -> str:
    """``[FANOUT_DIGEST] it=<n> log_like=<sha1> coords=<sha1> inds=<sha1>``.

    Cluster-gate tool (``docs/multirank-cluster-gates.md`` Step 1): the head
    logs this once per iteration under ``GF_FANOUT_DIGEST=1`` so three
    transport-parity layouts (shared-GPU / multi-GPU-one-node / multi-node)
    can be diffed line-for-line for bit-identical merged state. Host numpy
    only; ``coords``/``inds`` each hash the byte-concatenation of every
    branch's array in branch-name-sorted order (branches differ in shape, so
    concatenating raw bytes -- not ``np.concatenate`` -- is what makes a
    single hash possible across all of them), independent of dict order.

    ``residual_hashes`` (one-walker replica mode): ``{world_rank: hash}`` from
    :meth:`WalkerFanout.gather_residual_hashes`; when given, the line gains
    `` residual=r<rank>:<hash>,... replicas_agree=<bool>`` so replica drift is
    visible on the same line as the merged-state digest. ``layout`` (the
    run's ``WalkerBlockLayout``) makes ``replicas_agree`` a PER-BLOCK
    comparison (:func:`replicas_agree`); without it the whole map is
    compared, which is right at one walker block.

    ``GF_FANOUT_DIGEST`` is off by default and the 9mo/12mo launchers leave
    it unset; it is listed in ``docs/global-fit-launch.md``.

    Args:
        iteration (int): The iteration whose moves just ran.
        state: The merged head state (``log_like``, ``branches_coords``,
            ``branches_inds``).
        residual_hashes (dict, optional): ``{rank: hash}``; adds the
            ``residual=`` and ``replicas_agree=`` fields.
        layout (WalkerBlockLayout, optional): Forwarded to
            :func:`replicas_agree`.

    Returns:
        str: The log line.
    """
    log_like_hash = _sha1_16(_array_bytes(state.log_like))
    coords = state.branches_coords
    coords_hash = _sha1_16(b"".join(_array_bytes(coords[name]) for name in sorted(coords)))
    inds = state.branches_inds
    inds_hash = _sha1_16(b"".join(_array_bytes(inds[name]) for name in sorted(inds)))
    line = (
        f"[FANOUT_DIGEST] it={int(iteration)} log_like={log_like_hash} "
        f"coords={coords_hash} inds={inds_hash}"
    )
    if residual_hashes:
        items = ",".join(f"r{int(r)}:{h}" for r, h in sorted(residual_hashes.items()))
        agree = replicas_agree(residual_hashes, layout)
        line += f" residual={items} replicas_agree={agree}"
    return line


class WalkerFanout:
    """Head-side fan-out (usable on every compute rank for the collectives).

    Built by ``GlobalFit`` on every compute rank (multi-rank, or a single
    rank with the test-only ``GB_PROPOSE_ORCHESTRATE=1``) with the fan-out
    communicator and the two sub-communicators built beside it. :meth:`run` is head-only; the
    other compute ranks answer it from :class:`ComputeService`.
    :meth:`allgather_walker_vector` is a collective that every compute rank
    calls.

    Args:
        comm: The fan-out communicator
            (:meth:`~lisatools.globalfit.communication.ranks.WalkerBlockLayout.make_fanout_comm`),
            never the world or run communicator; its size must equal
            ``layout.n_compute`` unless the layout is single. ``None`` is a
            test-only placeholder.
        layout (WalkerBlockLayout): The run layout.
        rank (int): This process's rank in the run communicator.
        model (optional): The eryn model handed to each local body.
        logger (logging.Logger, optional): Receives remote failure
            tracebacks.
        group_comm (optional): The block's replica communicator
            (:meth:`~lisatools.globalfit.communication.ranks.WalkerBlockLayout.make_group_comm`).
        reps_comm (optional): One-lead-per-block communicator
            (:meth:`~lisatools.globalfit.communication.ranks.WalkerBlockLayout.make_reps_comm`).

    Raises:
        ValueError: When ``comm`` is not the fan-out communicator (size
            mismatch).

    Attributes:
        head (int): The head rank.
        is_head (bool): Whether this process is the head.
        single (bool): ``layout.is_single()``; :meth:`run` is then a direct
            call.
        seq (int): The fan-out counter (see the module docstring).
        last_wait_s (float): Seconds the last :meth:`run` waited on its
            sends.
        clock (dict): The clock shipped with every command: ``iteration``,
            ``stage``, ``stage_kind``, ``move``, ``call_index``, ``seq``,
            ``seed_base`` (set by ``GlobalFit`` after the state broadcast).
    """

    def __init__(self, comm, layout, rank, *, model=None, logger=None,
                 group_comm=None, reps_comm=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.model = model
        self.logger = logger
        self.head = layout.head_rank
        self.is_head = self.rank == self.head
        self.single = layout.is_single()
        #: the R ranks sharing this rank's walker block (``None`` when single
        #: or when the caller did not build one). Work shared only by a
        #: block's replicas -- GB's per-unit delta ledger above all -- goes
        #: here, NOT on ``self.comm``: on the fan-out comm it would be
        #: O(n_compute) instead of O(R) AND would put blocks with nothing to
        #: say to each other into unit lockstep.
        #:
        #: PASSED IN, never built here. ``run.py`` builds both beside
        #: ``make_fanout_comm`` in ``GlobalFit.__init__``, which every rank
        #: reaches exactly once. ``WalkerFanout`` itself is built on every
        #: compute rank by ``GlobalFit._make_fanout`` (from ``prepare_main``
        #: on the head, from ``prepare_compute`` on the workers, which also
        #: build their ``ComputeService``). A ``Split`` taken here would turn
        #: every construction into a collective over the fan-out comm, which
        #: holds only while both setup paths call ``_make_fanout`` at the
        #: same point (today right after the state broadcast); tests also
        #: build it on one side only, with a ``None`` comm.
        self.group_comm = group_comm
        #: one representative per block (replica 0); a null comm off-lead.
        self.reps_comm = reps_comm
        if not self.single and comm is not None:
            # `comm` must be the compute-only fan-out comm (layout.make_fanout_comm),
            # never the world comm; `None` is a test-only placeholder that skips this
            # check entirely (see tests/test_fanout_fakecomm.py::test_run_on_a_worker_raises).
            size = int(comm.Get_size())
            if size != layout.n_compute:
                raise ValueError(
                    f"WalkerFanout comm size {size} != layout.n_compute {layout.n_compute}; "
                    "pass the compute-only fan-out comm (layout.make_fanout_comm(comm)), "
                    "not the world/full comm"
                )
        self.seq = 0
        self.last_wait_s = 0.0
        self._stopped = False
        self.clock = {
            "iteration": 0,
            "stage": None,
            "stage_kind": None,
            "move": None,
            "call_index": 0,
            "seq": 0,
            "seed_base": None,
        }
        self._call_index = collections.Counter()

    # -- clock -----------------------------------------------------------
    def enter_stage(self, name, kind):
        """Set ``clock["stage"]`` / ``clock["stage_kind"]`` (recipe step start)."""
        self.clock["stage"] = name
        self.clock["stage_kind"] = kind

    def note_iteration(self, iteration):
        """Set ``clock["iteration"]`` (called by the recipe every iteration)."""
        self.clock["iteration"] = int(iteration)

    # -- commands --------------------------------------------------------
    def gather_likelihood(self, acs):
        """Current per-walker likelihood over ALL walkers (head-only, sampling phase).

        The head scores its own block from ``acs`` (its B-row ACA); every worker
        answers through the ``LIKELIHOOD_OP`` builtin its ComputeService installs.
        Any head-side "all walkers" read of the residual during sampling must go
        through here -- the head's ACA holds only its block. With one compute
        rank this is a direct call.

        Args:
            acs: The head's ``AnalysisContainerArray`` (its block's rows).

        Returns:
            np.ndarray: Real log-likelihood per walker, shape ``(nwalkers,)``,
            in walker order (:func:`concat_blocks`).
        """
        return self.run(
            LIKELIHOOD_OP,
            move=None,
            per_rank_payload=lambda rank, w0, w1: None,
            local_body=lambda payload, model: np.asarray(
                asnumpy(acs.likelihood(complex=False))
            ),
            merge=lambda results: concat_blocks(results, self.layout),
        )

    def gather_residual_hashes(self, acs):
        """``{world_rank: residual_hash}`` over the compute ranks (head-only, sampling phase).

        One-walker replica mode watch: every rank hashes its OWN residual
        buffers (``acs`` for the head, the ``RESIDUAL_HASH_OP`` builtin for
        each worker) so drift between replicas shows up as disagreeing
        hashes -- unlike :meth:`gather_likelihood` / :func:`concat_blocks`,
        this is per-rank, never merged down to the head's single block.

        Called only by the recipe's ``[FANOUT_DIGEST]`` hook
        (``GF_FANOUT_DIGEST=1``) when the layout is in replica mode.

        Args:
            acs: The head's ``AnalysisContainerArray``.

        Returns:
            dict[int, str]: ``{rank: residual_hash}`` for every compute rank.
        """
        return self.run(
            RESIDUAL_HASH_OP,
            move=None,
            per_rank_payload=lambda rank, w0, w1: None,
            local_body=lambda payload, model: residual_hash(acs),
            merge=lambda results: {int(r): str(h) for r, h in results.items()},
        )

    def run(self, op, *, move=None, per_rank_payload, local_body, merge, shared=None):
        """Run one command on every compute rank's walker block and merge (head-only).

        Sends ``{"seq", "op", "move", "clock", "payload", "shared"}`` to each
        worker (payload from ``per_rank_payload``), waits for the sends,
        runs ``local_body`` on the head's own block, receives every reply,
        checks the sequence numbers and failures, logs the ``[FANOUT]``
        line, and returns ``merge({rank: result})``. With a single layout it
        calls ``merge({head: local_body(...)})`` directly. If the head's own
        body raises, the replies of the workers already sent a command are
        drained before re-raising, so the channel stays clean.

        Args:
            op (str): The operation (a builtin or a ``gf_serve`` op).
            move (str, optional): The move name (registry key). Default
                ``None`` for builtins.
            per_rank_payload (Callable[[int, int, int], object]):
                ``fn(rank, w0, w1)`` returning that rank's payload; also
                called for the head.
            local_body (Callable[[object, object], object]):
                ``fn(payload, model)`` run on the head.
            merge (Callable[[dict], object]): Combines ``{rank: result}``
                (head included).
            shared (optional): One object sent unchanged to every worker.

        Returns:
            object: The value returned by ``merge``.

        Raises:
            RuntimeError: When called off the head, or on a reply sequence
                mismatch.
            RemoteWorkerError: When any worker reports a failure.
        """
        if not self.is_head:
            raise RuntimeError("WalkerFanout.run is head-only; compute ranks serve commands")
        key = (move, op)
        self._call_index[key] += 1
        clock = dict(self.clock, move=move, call_index=self._call_index[key], seq=self.seq)
        w0, w1 = self.layout.block_of(self.head)
        if self.single:
            self.clock.update(clock)
            return merge({self.head: local_body(per_rank_payload(self.head, w0, w1), self.model)})

        self.seq += 1
        clock["seq"] = self.seq
        self.clock.update(clock)
        requests = []
        sent_workers = []
        wait_s = 0.0
        try:
            for r in self.layout.worker_ranks:
                rw0, rw1 = self.layout.block_of(r)
                cmd = {
                    "seq": self.seq,
                    "op": op,
                    "move": move,
                    "clock": clock,
                    "payload": per_rank_payload(r, rw0, rw1),
                    "shared": shared,
                }
                req = self.comm.isend(cmd, dest=self.layout.fanout_rank(r))
                requests.append(req)
                sent_workers.append(r)
            # Wait IMMEDIATELY: under real MPI, large payloads run the rendezvous
            # protocol, which only makes progress when the sender calls wait/test.
            # Every worker is already parked in recv, so this cannot block on them;
            # without this, the transfer stalls until the head reaches this point
            # anyway, but AFTER local_body -- serializing head and worker compute.
            t_wait0 = time.perf_counter()
            for req in requests:
                req.wait()
            wait_s = time.perf_counter() - t_wait0
            self.last_wait_s = wait_s
            t0 = time.perf_counter()
            local = local_body(per_rank_payload(self.head, w0, w1), self.model)
            head_wall = time.perf_counter() - t0
        except BaseException:
            # Drain: a worker that was sent a command WILL reply. If we don't
            # receive it here, it sits in the channel and corrupts the next
            # run()'s bookkeeping (or blocks stop()'s send). A worker that never
            # received a command must NOT be recv'ed from (it never sent a reply).
            for req in requests:
                try:
                    req.wait()
                except Exception:  # noqa: BLE001 - best-effort drain, never masks the raise
                    pass
            for r in sent_workers:
                try:
                    self.comm.recv(source=self.layout.fanout_rank(r))
                except Exception:  # noqa: BLE001 - best-effort drain, never masks the raise
                    pass
            raise
        replies = {self.head: _reply(self.seq, self.head, True, local, head_wall)}
        for r in self.layout.worker_ranks:
            replies[r] = self.comm.recv(source=self.layout.fanout_rank(r))
        # seq check FIRST: a stale reply (left over from an earlier, aborted run())
        # must be reported as a sequence error, not misread as a fresh remote failure.
        bad_seq = [rep["rank"] for rep in replies.values() if rep["seq"] != self.seq]
        if bad_seq:
            raise RuntimeError(f"fan-out sequence mismatch from ranks {bad_seq} (op={op!r})")
        failures = [rep for rep in replies.values() if not rep["ok"]]
        if failures:
            if self.logger is not None:
                for rep in failures:
                    self.logger.error(
                        "rank %d failed op=%r move=%r:\n%s",
                        rep["rank"],
                        op,
                        move,
                        (rep.get("error") or {}).get("traceback", ""),
                    )
            first = failures[0]
            raise RemoteWorkerError(first["rank"], op, move, first.get("error"))
        # INFO on the module logger (not DEBUG on ``self.logger``): one line per
        # fan-out op, the load-balance signal the cluster gates diff.
        worst = max(rep["wall_s"] for rep in replies.values())
        _LOG.info(
            "[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f wait_s=%.3f",
            op,
            move,
            head_wall,
            worst,
            wait_s,
        )
        return merge({r: rep["result"] for r, rep in replies.items()})

    def ping(self):
        """Every compute rank's layout digest, keyed by rank; raises on mismatch.

        Sent once by the head after setup, before sampling.

        Returns:
            dict[int, str]: ``{rank: WalkerBlockLayout.digest()}``.

        Raises:
            RuntimeError: When the digests differ (ranks resolved different
                layouts).
        """
        digests = self.run(
            PING_OP,
            per_rank_payload=lambda r, w0, w1: None,
            local_body=lambda payload, model: self.layout.digest(),
            merge=lambda results: results,
        )
        if len(set(digests.values())) != 1:
            raise RuntimeError(f"rank layouts disagree: {digests}")
        return digests

    def stop(self):
        """Send :data:`STOP_OP` to every worker once (head-only; no reply expected).

        Releases the compute ranks from :meth:`ComputeService.serve`. A no-op
        when single, off the head, or already stopped.
        """
        if self.single or not self.is_head or self._stopped:
            return
        self._stopped = True
        for r in self.layout.worker_ranks:
            self.comm.send(
                {
                    "seq": self.seq,
                    "op": STOP_OP,
                    "move": None,
                    "clock": dict(self.clock),
                    "payload": None,
                    "shared": None,
                },
                dest=self.layout.fanout_rank(r),
            )

    # -- setup-phase collective ------------------------------------------
    def allgather_walker_vector(self, local_1d):
        """Concatenate the BLOCKS' 1-D vectors in walker order (collective).

        Stays a single allgather on the fan-out comm -- every compute rank
        must reach it, so a split over the representatives communicator
        would need the non-leads to sit out a collective they are already
        inside. Instead every rank allgathers and then keeps one part per
        block, which is cheap (the parts are one float per walker) and keeps
        the result identical on every rank by construction.

        At ``R == 1`` the kept parts are every part, in compute-rank order --
        byte-for-byte the pre-2026-09-23 concatenation. At ``n_blocks == 1``
        exactly one part survives, the head's, which is what replica mode
        has always returned.

        Args:
            local_1d (array-like): This rank's per-walker values over its
                block, shape ``(block,)``.

        Returns:
            np.ndarray: The values of all walkers in walker order, shape
            ``(nwalkers,)``; ``local_1d`` itself when single.
        """
        local = np.asarray(local_1d)
        if self.single:
            return local
        parts = self.comm.allgather(local)  # fan-out comm ranks == compute-rank order
        layout = self.layout
        leads = getattr(layout, "block_leads", None)
        if leads is None:  # pragma: no cover - stub layouts in older tests
            if getattr(layout, "replica_mode", False):
                return np.asarray(parts[layout.fanout_rank(self.head)])
            return np.concatenate([np.asarray(p) for p in parts])
        return np.concatenate(
            [np.atleast_1d(np.asarray(parts[layout.fanout_rank(r)])) for r in leads])


class ComputeService:
    """Compute-rank command loop: ``recv`` is the clock, ``{"op": "stop"}`` exits.

    Built by ``GlobalFit.prepare_compute`` on every compute rank other than
    the head. :meth:`serve` blocks on the head's commands until
    :data:`STOP_OP`; each command is answered by :meth:`handle` with one
    reply (schemas in the module docstring).

    Args:
        comm: The fan-out communicator.
        layout (WalkerBlockLayout): The run layout.
        rank (int): This process's rank in the run communicator.
        registry (dict): Moves by ``(stage_name, move_name)`` with bare
            ``move_name`` keys as fallback (``run._serve_registry``).
        model (optional): The eryn model handed to every body.
        builtins (dict, optional): Extra ``{op: fn(payload, clock, model)}``
            answered before the registry; :data:`PING_OP` is always
            installed (layout digest). ``GlobalFit`` adds
            :data:`LIKELIHOOD_OP`, :data:`RESIDUAL_HASH_OP` and
            ``RESIDUAL_SNAPSHOT_OP``.
        logger (logging.Logger, optional): Receives the traceback of a
            failing command.
    """

    def __init__(self, comm, layout, rank, *, registry, model=None, builtins=None, logger=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.registry = dict(registry)
        self.model = model
        self.logger = logger
        self.builtins = {PING_OP: lambda payload, clock, model: self.layout.digest()}
        self.builtins.update(builtins or {})
        self._head_local = layout.fanout_rank(layout.head_rank)

    def handle(self, cmd):
        """Execute one command and build its reply; never raises.

        A builtin op is called as ``fn(payload, clock, model)``. Any other op
        looks up the move by ``(clock["stage"], move)`` (then by the bare
        name), sets ``move.gf_stage_kind`` from the clock, and calls
        ``move.gf_serve(op, payload, clock, model)``. An exception becomes an
        ``ok=False`` reply carrying its type, message and traceback.

        Args:
            cmd (dict): The command (see the module docstring).

        Returns:
            dict: The reply.
        """
        t0 = time.perf_counter()
        seq = op = move_name = None
        try:
            seq = cmd.get("seq")
            op = cmd.get("op")
            move_name = cmd.get("move")
            clock = cmd.get("clock") or {}
            if op in self.builtins:
                result = self.builtins[op](cmd.get("payload"), clock, self.model)
            else:
                # Registry keys are ``(stage_name, move_name)`` (run.py
                # _serve_registry: names recur across stages) with a bare-name
                # fallback for stage-less registries (tests, hand-built).
                move = self.registry.get((clock.get("stage"), move_name))
                if move is None:
                    move = self.registry[move_name]
                if "stage_kind" in clock:
                    move.gf_stage_kind = clock["stage_kind"]
                result = move.gf_serve(op, cmd.get("payload"), clock, self.model)
            return _reply(seq, self.rank, True, result, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - reported to the head, never swallowed
            error = {
                "type": type(exc).__name__,
                "msg": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                if self.logger is not None:
                    self.logger.exception(
                        "rank %d: command %r for move %r failed", self.rank, op, move_name
                    )
            except Exception:  # noqa: BLE001 - a failing logger must never mask the reply
                pass
            return _reply(seq, self.rank, False, None, time.perf_counter() - t0, error)

    def serve(self):
        """Serve head commands until :data:`STOP_OP`.

        Receives from the head's fan-out rank, answers each command with
        :meth:`handle`, and sends the reply back. If the reply cannot be
        sent (e.g. it does not pickle), a ``SendError`` failure reply is
        sent instead so the head does not hang.

        Returns:
            int: Number of commands served (STOP excluded).
        """
        served = 0
        while True:
            cmd = self.comm.recv(source=self._head_local)
            if cmd.get("op") == STOP_OP:
                return served
            reply = self.handle(cmd)
            try:
                self.comm.send(reply, dest=self._head_local)
            except Exception as exc:  # noqa: BLE001 - a bad reply must not escape the loop
                error = {
                    "type": "SendError",
                    "msg": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                degraded = _reply(reply["seq"], self.rank, False, None, reply["wall_s"], error)
                self.comm.send(degraded, dest=self._head_local)
            served += 1
