"""Whole-``propose`` fan-out over walker blocks (addremove family, PSD family).

Head: ``propose`` slices the FULL state into per-rank blocks, every compute rank
(the head included) runs the UNCHANGED body ``propose_local`` on its block
against its own ACA, the head merges the blocks back, SUM-reduces the per-ladder
swap tallies and adapts each ladder ONCE. With one compute rank ``propose`` is a
direct call of ``propose_local`` -- bit-identical to the pre-port code.

Semantics that change only when several compute ranks exist (design spec,
"Semantics that change only when n_compute > 1"):

* Eryn ladders adapt once per propose from the pooled swap ratio
  (sum accepted / sum proposed over ranks and repeats); ``tc.time`` advances once
  per propose instead of once per repeat. Ranks never adapt.
* The mid-iteration checkpoint granularity drops from per leaf to per propose.
* Eigen / info-matrix tables are rank-local proposal shapes; the sidecar is
  written by the head only.
* Stretch complements (PSD RedBlue splits) are the rank's local block.
* Cross-rank tempering swaps: none (within-rank); see the spec's WP8.
* Sub-state delta counters of the fanned-out branches equal the SUM over ranks
  of what the body wrote (the head's copy is zeroed first when the body assigns
  them, ``fanout_assigns_counters``).
* PSD search mode (``PSDMove.run_move_max_likelihood``) plateaus on the RANK's
  walker block maximum, so search termination is per block; pooling it would
  need an in-body collective across ranks (deferred).
* The pooled PSD ladder adaptation consumes ``PSDMove.run_move``'s explicit
  ``temperature_swaps`` tallies (the fancy, walker-permuting swaps included),
  where single mode adapts from the identity swaps ``temper_comps`` makes
  inside eryn's ``RedBlueMove.propose``. Both are valid acceptance ratios; the
  swap POPULATION feeding the ladder differs.
* ``move.temperature_control.swaps_accepted`` -- what eryn stores to HDF -- is
  the HEAD block's count, while the sub-state swap counters are pooled over
  ranks. Diagnostic only: nothing reads the control's copy back.

One-walker replica mode (``layout.replica_mode``): ``propose`` runs
``propose_local`` on the head with the full one-walker state; the moves'
scoring seams scatter rows through ``RowFanout``; ranks serve ``serve_<op>``
methods; none of the semantics above change because the body IS the
single-process body.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..communication.walkerslice import merge_state, slice_state
from ..state import GFState

logger = logging.getLogger(__name__)

__all__ = ["PROPOSE_OP", "WalkerFanoutMixin", "pooled_ladder_step"]

PROPOSE_OP = "propose"


def pooled_ladder_step(tc, betas0, acc, prop):
    """One head-side ladder adaptation from pooled swap tallies; advances ``tc.time``.

    Returns the new ladder (``betas0`` when the control is not configured
    adaptive, has one rung, is past ``stop_adaptation``, or ANY rung has zero
    proposals -- stricter than Eryn, which would divide by zero). Mirrors
    ``TemperatureControl.adapt_temps`` (Eryn tempering.py ~867-897) with
    ``ratios = acc / prop`` supplied by the caller; ``tc.time`` advances once
    per call regardless (the spec's "once per propose"), where Eryn advances
    it only while adapting -- unobservable, a non-adaptive control never reads
    ``time``. Single-sampler ladders only (LAT never builds ``nsamplers > 1``).
    """
    if int(getattr(tc, "nsamplers", 1)) != 1:
        raise NotImplementedError("pooled_ladder_step: nsamplers > 1 ladders are not supported")
    betas0 = np.array(betas0, dtype=float, copy=True)
    acc = np.asarray(acc, dtype=float).ravel()
    prop = np.asarray(prop, dtype=float).ravel()
    adaptive = bool(getattr(tc, "gf_configured_adaptive", getattr(tc, "adaptive", False)))
    stop = int(getattr(tc, "stop_adaptation", -1))
    running = stop < 0 or int(tc.time) < stop
    new = betas0
    if adaptive and int(tc.ntemps) > 1 and acc.size and np.all(prop > 0) and running:
        new = betas0 + tc._get_ladder_adjustment(int(tc.time), betas0.copy(), acc / prop)
    tc.time += 1
    return new


class WalkerFanoutMixin:
    """First in the MRO of a move whose whole ``propose`` runs per walker block.

    Subclass contract: rename the body to ``propose_local(model, state)``; the
    recipe stamps ``fanout_branches`` and calls ``install_walker_fanout(curr)``
    after construction; override the hooks below.
    """

    fanout = None  # the run's WalkerFanout (None single-process)
    gf_rank = None
    #: EVERY sub-state the body reads or writes (shipped in the slice, merged
    #: back). A sub-state the body initializes but does not list is refused at
    #: the reply boundary (``gf_serve`` raises) -- never silently dropped.
    fanout_branches = None
    fanout_assigns_counters = False  # body ASSIGNS sub-state delta counters -> zero, then sum
    _fanout_body = False  # True while propose_local runs as a rank body
    gf_clock = None  # the command clock of the propose being served (rank side)
    #: Head-set knobs the rank's body must follow PER PROPOSE. A recipe step
    #: applies its PE declarations (``apply_inmodel_repeats`` -> ``num_repeats``)
    #: to the HEAD's move objects only -- ``note_recipe_step`` runs on the
    #: recipe's announce path, which the compute ranks never execute -- while
    #: each rank holds its own move object built from the env at construction.
    #: Without this, full_pe's 25 repeats would run on the head's walker block
    #: and the construction-time 10 (psd/galfor) or 2 (MBH/EMRI) everywhere
    #: else. Shipped in the propose payload, installed around ``propose_local``
    #: and restored after it (:meth:`fanout_install_live`). Subclasses extend
    #: the tuple for any other knob a step rewrites after construction.
    fanout_live_attrs = ("num_repeats",)

    # ---- one-walker replica mode (row scatter) ---------------------------
    row_fanout = None  # RowFanout when the layout is in replica mode
    likelihood_fanout = True  # {PREFIX}_LIKELIHOOD_FANOUT: 0 = the head scores every row itself

    def fanout_knob_prefix(self) -> str:
        """Env prefix of this move's knobs (rule 0: the branch name, capitalized)."""
        return str(getattr(self, "branch_name", "gf")).upper()

    def rows_active(self) -> bool:
        """True when scoring rows scatter over the replicas (replica mode, knob on)."""
        return self.row_fanout is not None and bool(self.likelihood_fanout)

    def ensemble_likelihood(self, acs):
        """Per-walker lnL over the WHOLE ensemble, not just this rank's block.

        WHICH READ IS CORRECT DEPENDS ON WHICH BODY IS RUNNING:

        * BLOCK mode (``R == 1``) -- every rank runs its OWN block's
          ``propose_local`` against its own ACA and writes into a sliced
          state, so the plain local read is already the right width. Routing
          it would be wrong (and a collective from a rank that is not the
          head would hang).
        * FLAT mode (``R > 1``, i.e. ``row_fanout`` installed) -- the head
          runs the FULL-N body while its ACA holds only its own block, so a
          local read returns B rows for an N-row state. It must be gathered.

        Gated on ``row_fanout is not None`` (the layout), never on the
        ``{PREFIX}_LIKELIHOOD_FANOUT`` knob: that knob chooses who SCORES
        rows, and turning it off must not silently narrow an ensemble-wide
        read back to one block.

        ``gather_likelihood`` is a head-driven fan-out command, so the
        workers answer it from their ``ComputeService`` loop and its merge
        reduces over one representative per block.
        """
        if self.row_fanout is None or not self.fanout_active:
            return acs.likelihood()
        return self.fanout.gather_likelihood(acs)

    # ---- subclass hooks --------------------------------------------------
    def fanout_temperature_controls(self):
        """Every ``TemperatureControl`` this move adapts (ranks never adapt them)."""
        raise NotImplementedError

    def fanout_payload_extra(self):
        """Head -> rank clock values (ladders, propose counters). Host numpy only.

        Return a FRESH dict and treat it as immutable: it is evaluated once per
        propose, shared by reference with the head's own body and pickled to
        the workers.
        """
        return {}

    def fanout_apply_extra(self, extra):
        """Rank side: install the shipped clock values before the body runs."""

    def fanout_live_payload(self) -> dict:
        """Head: ``{name: value}`` of :attr:`fanout_live_attrs` this move carries."""
        return {k: getattr(self, k) for k in (self.fanout_live_attrs or ())
                if hasattr(self, k)}

    def fanout_install_live(self, live) -> dict:
        """Rank: install the head's live knobs; return what they replaced.

        Only names in :attr:`fanout_live_attrs` that this move already carries
        are written. On the head the shipped values are its own (a no-op).
        """
        allowed = set(self.fanout_live_attrs or ())
        saved = {}
        for k, v in dict(live or {}).items():
            if k in allowed and hasattr(self, k):
                saved[k] = getattr(self, k)
                setattr(self, k, v)
        return saved

    def fanout_restore_live(self, saved) -> None:
        for k, v in dict(saved or {}).items():
            setattr(self, k, v)

    def fanout_reply_extra(self, part):
        """Rank -> head: swap tallies and anything the body left rank-local."""
        return {}

    def _passive_propose_reply(self, payload):
        """A no-op propose reply from a rank whose block another rank proposes.

        Shaped exactly like ``gf_serve``'s real reply so the head's merge
        needs no special case, but carrying the slice UNCHANGED and zero
        accepted counts. Nothing of it reaches the state: the merge takes
        one representative per block and this rank is not it. It exists so
        the fan-out's reply set stays uniform -- every compute rank answers
        every command -- rather than the head having to address a subset,
        which ``WalkerFanout.run`` has no way to do.
        """
        part = payload["state"]
        coords = getattr(part, "branches_coords", None) or {}
        if coords:
            shape = np.shape(next(iter(coords.values())))
            accepted = np.zeros(shape[:2], dtype=bool)
        else:
            accepted = np.zeros((1, 1), dtype=bool)
        for name in list(getattr(part, "sub_states", None) or {}):
            if name not in set(self.fanout_branches or []):
                part.sub_states[name] = None
        return {"state": part, "accepted": accepted, "extra": {}}

    def fanout_merge_extra(self, replies, new_state):
        """Head: pooled ladder adaptation etc. ``replies`` = {rank: extra}."""

    # ---- block path under replicas: publish to the passive replicas -----
    def fanout_replica_sync_payload(self, new_state, w0, w1):
        """Head: what a PASSIVE replica of block ``[w0, w1)`` needs after a propose.

        Only reached on the BLOCK path with several ranks per block (a
        family with ``fanout_flat_body = False`` under the ``GF_GPU_ROUTING``
        layout): the block's lead proposed and changed its own containers,
        and the group-mates hold the same walkers on containers that still
        carry the pre-propose state. Host numpy only (it is pickled).
        ``install_walker_fanout`` refuses that layout for a family that does
        not implement this and :meth:`fanout_apply_replica_sync`.
        """
        raise NotImplementedError

    def fanout_apply_replica_sync(self, payload):
        """Rank: install :meth:`fanout_replica_sync_payload`'s products locally."""
        raise NotImplementedError

    def serve_replica_sync(self, payload, clock, model):
        """Fan-out op ``replica_sync``: a passive replica applies the payload;
        every other rank (the leads, the head) gets ``None`` and does nothing."""
        if payload is None:
            return None
        self.fanout_apply_replica_sync(payload)
        return None

    def _fanout_sync_replicas(self, new_state, passive):
        """Head: one ``replica_sync`` command AFTER the propose merge.

        A SECOND command, never one nested inside the propose command (the
        2026-09-23 nested-fan-out lesson): every compute rank answers it, the
        leads with nothing to do. ``passive`` is the list of non-lead ranks.
        """
        fanout = self.fanout
        layout = fanout.layout
        passive = set(int(r) for r in passive)

        def payload(rank, w0, w1):
            if int(rank) not in passive:
                return None
            return self.fanout_replica_sync_payload(new_state, int(w0), int(w1))

        def body(p, _model):
            if p is None:
                return None
            return self.serve_replica_sync(p, fanout.clock, _model)

        fanout.run(
            "replica_sync",
            move=getattr(self, "gf_move_name", None),
            per_rank_payload=payload,
            local_body=body,
            merge=lambda _replies: None,
        )
        logger.debug(
            "[FANOUT] %s: published the merged propose to %d passive "
            "replica(s) %s (%d rank(s) per block)",
            getattr(self, "gf_move_name", type(self).__name__), len(passive),
            sorted(passive), int(getattr(layout, "ranks_per_block", 1)))

    # ---- wiring ------------------------------------------------------------
    @property
    def fanout_active(self):
        return self.fanout is not None and not self.fanout.single

    def install_walker_fanout(self, curr):
        """Bind the run's fan-out (``curr.fanout``) and apply the multi-rank rules."""
        self.fanout = getattr(curr, "fanout", None)
        self.gf_rank = getattr(curr, "rank", None)
        if not self.fanout_active:
            return
        # Refuse BEFORE anything is mutated, so a rejected move leaves no trace.
        if self.fanout_active and type(self).propose is not WalkerFanoutMixin.propose:
            raise TypeError(
                f"{type(self).__name__} overrides propose(); under several compute ranks the "
                "body must live in propose_local() so every rank runs it (the mixin owns propose)"
            )
        layout = self.fanout.layout
        if getattr(layout, "replica_mode", False) and not self._flat_body_ok(layout):
            # REPLICAS OVER SEVERAL BLOCKS for a family that is not
            # flat-capable (PSD/galfor): the BLOCK path, with only each
            # block's lead proposing. The body then runs block-local with
            # NO row scatter -- a ``RowFanout`` here would issue a second
            # fan-out command from inside the propose command (``run`` is
            # head-only; on a worker lead it raises, on the head it nests)
            # -- and the head publishes the merged result to the passive
            # replicas afterwards (:meth:`_fanout_sync_replicas`). Refused
            # HERE, at build time, when the family cannot do that publish:
            # a replica left on the pre-propose noise model would score
            # every later move against a stale likelihood, silently.
            if (type(self).fanout_apply_replica_sync
                    is WalkerFanoutMixin.fanout_apply_replica_sync
                    or type(self).fanout_replica_sync_payload
                    is WalkerFanoutMixin.fanout_replica_sync_payload):
                raise TypeError(
                    f"{type(self).__name__} takes the BLOCK path under "
                    f"replicas over {int(getattr(layout, 'n_blocks', 1))} "
                    "walker blocks (fanout_flat_body=False) but implements no "
                    "replica sync (fanout_replica_sync_payload / "
                    "fanout_apply_replica_sync), so its passive replicas would "
                    "keep a stale state. Implement both, or make the family "
                    "flat-capable."
                )
        elif getattr(layout, "replica_mode", False):
            from ..communication.rowfanout import RowFanout

            self.row_fanout = RowFanout(self.fanout, self)
            env = os.environ.get(f"{self.fanout_knob_prefix()}_LIKELIHOOD_FANOUT")
            if env is not None:
                self.likelihood_fanout = env.strip() not in ("0", "false", "False", "")
            if not self.likelihood_fanout and int(getattr(layout, "n_blocks", 1)) > 1:
                # "Score every row on the head" is only possible when the
                # head's container holds every walker (one block). With
                # several blocks it holds its own block alone, so a full-N
                # body scoring locally would index rows that do not exist
                # there: the knob is refused (kept ON), loudly, not obeyed.
                logger.warning(
                    "[FANOUT] %s_LIKELIHOOD_FANOUT=0 ignored: with %d walker "
                    "blocks the head holds only its own block's rows, so every "
                    "scoring row must go to the ranks that hold its walker.",
                    self.fanout_knob_prefix(), int(getattr(layout, "n_blocks", 1)))
                self.likelihood_fanout = True
            if not self.fanout.is_head and hasattr(self, "eigen_store_path"):
                self.eigen_store_path = None  # single-writer sidecar (head only)
            # the head runs the single-process body: its ladders adapt inside it
            return
        for tc in self.fanout_temperature_controls():
            if tc is None:
                continue
            if not hasattr(tc, "gf_configured_adaptive"):  # shared controls: stamp once
                tc.gf_configured_adaptive = bool(getattr(tc, "adaptive", False))
            tc.adaptive = False  # ranks never adapt; the head adapts once per propose
        if not self.fanout.is_head and hasattr(self, "eigen_store_path"):
            # Single-WRITER sidecar: the head owns the file. Compute ranks keep
            # the path READ-ONLY (2026-09-17): a walker-independent table
            # (``walker_max`` scope -- MBH, EMRI, and SOBBH since the same-day
            # ruling) is valid for every rank, and without it every resume
            # made rank 1 rebuild all its information matrices while the head
            # adopted its sidecar (6mo continuation, job 541: mbh_pe head 190 s
            # vs rank 1324 s; emri_pe 251 s vs 1670 s -- a 7x block imbalance).
            # Per-(temp, walker) stashes are NOT adopted off-head: the head's
            # stash covers the head's block, not this rank's walkers.
            self.eigen_store_readonly = True

    #: Can this family's body run at the FULL ensemble width on a rank whose
    #: ACA holds only its block? True for addremove, whose every evaluation
    #: funnels through one scoring seam. FALSE for PSD/galfor, which derive
    #: their walker count and device map from the LOCAL ACA
    #: (``len(self.acs.flatten())``, ``self.acs.gpu_map[...]``) -- those are
    #: WRONG under a full-N body, not merely slow, so that family keeps the
    #: block path until those reads come from the state instead.
    fanout_flat_body = True

    def _flat_body_ok(self, layout) -> bool:
        """May THIS layout run the full-ensemble body on this rank?

        ``fanout_flat_body`` is about whether the body can cope with a state
        WIDER than the rank's own ACA. At ``n_blocks == 1`` there is no such
        gap -- the single block IS the whole ensemble, so the local ACA holds
        every walker and even a body that sizes itself from
        ``len(self.acs.flatten())`` is correct.

        That case is not hypothetical: it is one-walker replica mode, which
        has ALWAYS run psd/galfor through the flat path. Refusing it here
        pushed that family onto the block path, where ``propose_local`` ran
        inside a ``propose`` command and its ``_replay_noise_begin`` issued a
        second fan-out command from within the first -- a nested fan-out,
        which surfaces as ``fan-out sequence mismatch``. Caught by
        ``tests/test_multirank_noise_smoke.py::test_one_walker_two_replicas``.

        So the opt-out only bites where it must: several blocks, where the
        head's ACA really is narrower than the state.
        """
        if self.fanout_flat_body:
            return True
        return int(getattr(layout, "n_blocks", 1)) <= 1

    # ---- propose -------------------------------------------------------------
    def fanout_prepare_state(self, model, state):
        """Hook: edit the FULL-ensemble state before any slicing (no-op by default).

        Called once per propose at the top of :meth:`propose`, on whichever
        process owns the whole ensemble: the single-process body, the flat
        (replica) head, or the block head right before ``fanout_propose``
        slices the state per rank. It is the one place a cross-walker edit
        of a tempered sub-state (one walker's rung written from another
        walker's) can be made without a replay, since every rank only ever
        sees its own walker block. The source moves' search-stage cold ->
        hot rotation lives here (``ResidualAddOneRemoveOneMove``).
        """
        return None

    def propose(self, model, state):
        self.fanout_prepare_state(model, state)
        if not self.fanout_active:
            return self.propose_local(model, state)
        layout = self.fanout.layout
        if getattr(layout, "replica_mode", False) and self._flat_body_ok(layout):
            # FLAT: the head runs the unchanged full-ensemble body; the body's
            # scoring seams scatter rows over the replicas (RowFanout). At one
            # walker block this is the original one-walker replica path.
            return self.propose_local(model, state)
        # BLOCK: every walker block runs its own body. With replicas present
        # only the block LEAD runs it (see ``fanout_propose``) -- the
        # group-mates hold the same walkers, so letting them all propose
        # would draw R different proposals for one block and the merge would
        # keep whichever replied last.
        return self.fanout_propose(model, state)

    def fanout_propose(self, model, state):
        """Head: slice the ensemble, run every block, merge the replies back.

        ``local_body`` closes over ``propose``'s own ``model``, NOT the one
        ``WalkerFanout.run`` hands it (``fanout.model``, bound once at setup
        and never refreshed): the object eryn passes ``propose`` is the only
        one guaranteed to be this iteration's. Same fix GB's ``_fanout_cmd``
        carries; the workers keep their ``ComputeService`` model, which IS
        their live one.
        """
        fanout = self.fanout
        layout = fanout.layout
        branches = list(self.fanout_branches or [])
        extra = dict(self.fanout_payload_extra() or {})
        _live = self.fanout_live_payload()
        if _live:
            # the head's live knobs ride under a reserved key; gf_serve pops
            # it before the subclass sees ``extra``
            extra["_live_attrs"] = _live

        # the head's ACA must be its own walker block, not the ensemble: the
        # body scores this block's rows against it (spec, ACA-width rule)
        acs = getattr(model, "analysis_container_arr", None)
        entries = getattr(acs, "acs_total_entries", None)
        if entries is not None:
            w0, w1 = layout.block_of(fanout.rank)
            if int(entries) != w1 - w0:
                raise RuntimeError(
                    f"{type(self).__name__}: the head's AnalysisContainerArray "
                    f"carries {int(entries)} walker rows but its block is "
                    f"[{w0}, {w1}) ({w1 - w0} walkers). The per-rank ACA must "
                    "be built at the block width, not the ensemble width."
                )

        # With replicas, only ONE rank per block proposes; the rest are
        # passive. They hold the same walkers, so R bodies would draw R
        # different proposals for one block (the per-rank seeds differ) and
        # the merge below would silently keep whichever replied last.
        def _is_lead(rank):
            if not hasattr(layout, "is_block_lead"):
                return True
            return bool(layout.is_block_lead(rank))

        _passive = [r for r in layout.compute_ranks if not _is_lead(r)]
        if _passive:
            logger.info(
                "[FANOUT] %s: block path with %d rank(s) per block; "
                "leads %s propose, %s stay passive (this family's body is "
                "not flat-capable)",
                getattr(self, "gf_move_name", type(self).__name__),
                int(getattr(layout, "ranks_per_block", 1)),
                list(layout.block_leads), _passive,
            )

        def payload(rank, w0, w1):
            return {"state": slice_state(state, w0, w1, sub_states=branches),
                    "extra": extra, "passive": not _is_lead(rank)}

        def body(p, _model_local):
            if p.get("passive"):
                # Return the slice UNCHANGED plus a zero accepted count: the
                # merge writes the lead's result over this block anyway, and
                # a neutral reply keeps the reply shape uniform.
                return self._passive_propose_reply(p)
            return self.gf_serve(PROPOSE_OP, p, fanout.clock, model)

        def merge(replies):
            new_state = GFState(state, copy=True)
            if self.fanout_assigns_counters:
                for name in branches:
                    sub = (getattr(new_state, "sub_states", None) or {}).get(name)
                    if sub is None or not getattr(sub, "tempered_initialized", False):
                        continue
                    for cname in sub.delta_counter_names:
                        arr = getattr(sub, cname, None)
                        if arr is not None:
                            arr[...] = 0
            # ONE REPRESENTATIVE PER BLOCK. ``block_of`` is no longer a
            # partition: at R > 1 the group-mates return the same walker
            # slice, so merging every compute rank would write each block R
            # times (and concatenate accepted R times, giving an array R x
            # too wide). At R == 1 the leads ARE the compute ranks and this
            # is the original loop.
            _reps = (list(layout.block_leads) if hasattr(layout, "block_leads")
                     else list(layout.compute_ranks))
            for rank in _reps:
                w0, w1 = layout.block_of(rank)
                merge_state(new_state, replies[rank]["state"], w0, w1)
            accepted = np.concatenate(
                [np.asarray(replies[r]["accepted"]) for r in _reps], axis=1
            )
            self.fanout_merge_extra({r: replies[r]["extra"] for r in _reps},
                                    new_state)
            return new_state, accepted

        result = fanout.run(
            PROPOSE_OP,
            move=getattr(self, "gf_move_name", None),
            per_rank_payload=payload,
            local_body=body,
            merge=merge,
        )
        if _passive:
            # the passive replicas still hold the pre-propose state on their
            # containers: publish the merged result to them (a second
            # command, after the first has fully returned)
            self._fanout_sync_replicas(result[0], _passive)
        return result

    def gf_serve(self, op, payload, clock, model):
        if op != PROPOSE_OP:
            handler = None if str(op).startswith("_") else getattr(self, f"serve_{op}", None)
            if handler is None:
                raise ValueError(f"{type(self).__name__} serves no fan-out command {op!r}")
            self.gf_clock = dict(clock or {})
            return handler(payload, clock, model)
        self.gf_clock = dict(clock or {})  # WP8 seeds its synced swap RNG from this
        if payload.get("passive"):
            # A block's non-lead replica (``fanout_propose``): its lead
            # proposes for the block. Honoured HERE as well as in the head's
            # own local body -- a WORKER replica reaches the body through
            # this method, and before 2026-10-15 it ran the full body anyway
            # (its own proposals, its own publish onto its containers, the
            # merge then discarding the result).
            return self._passive_propose_reply(payload)
        _extra = dict(payload.get("extra") or {})
        _saved_live = self.fanout_install_live(_extra.pop("_live_attrs", None))
        self.fanout_apply_extra(_extra)
        self._fanout_body = True
        try:
            part, accepted = self.propose_local(model, payload["state"])
        finally:
            self._fanout_body = False
            self.fanout_restore_live(_saved_live)
        # `propose_local` commonly re-wraps its input as ``GFState(state, copy=True)``;
        # that constructor instantiates a BARE (non-None, not tempered_initialized)
        # sub-state for every branch in ``sub_state_bases``, even ones ``slice_state``
        # deliberately left ``None`` (branches outside ``fanout_branches``). Restore
        # that invariant on the reply so the head's ``merge_state`` only ever touches
        # the branches this move actually fans out -- a bare sub-state reaching
        # ``merge_walkers`` on an already-initialized full branch raises there. A
        # sub-state the body actually INITIALIZED outside ``fanout_branches`` is a
        # contract violation (its columns would be lost): refuse loudly.
        fanned = set(self.fanout_branches or [])
        for name in list(getattr(part, "sub_states", None) or {}):
            if name in fanned:
                continue
            sub = part.sub_states[name]
            if sub is not None and getattr(sub, "tempered_initialized", False):
                raise RuntimeError(
                    f"{type(self).__name__}: the body initialized sub-state {name!r}, which "
                    f"is not in fanout_branches={sorted(fanned)}; list every sub-state the "
                    "body reads or writes."
                )
            part.sub_states[name] = None
        return {
            "state": part,
            "accepted": np.asarray(accepted),
            "extra": self.fanout_reply_extra(part),
        }
