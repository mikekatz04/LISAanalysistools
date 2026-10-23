"""Rank roles, walker-block layout, per-rank device pinning and seeds.

Every rank of the RUN communicator calls :func:`prepare_rank` once, before
``fit.build()``. It resolves one :class:`WalkerBlockLayout` that is identical
on every rank (:func:`build_layout`, one collective ``allgather``), pins the
rank's GPU(s) (:func:`select_rank_device`, before any CUDA initialisation)
and stamps the result on the fit. :class:`~lisatools.globalfit.run.GlobalFit`
then reads the layout to decide what each rank does.

Roles (:class:`RankRole`, resolved by :func:`resolve_roles`):

- HEAD (``main_rank``, rank 0 by default): sequences the recipe, owns the
  full host :class:`~lisatools.globalfit.state.GFState`, AND computes walker
  block 0 like any other compute rank.
- COMPUTE: one device list and one walker block each; runs the
  :class:`~lisatools.globalfit.communication.fanout.ComputeService` loop.
- SAVER: the highest non-head rank at communicator size >= 3, which runs the
  HDF5 save loop. Below 3 ranks the role is aliased to the head (synchronous
  saves), except in the size-2 fallback of :func:`build_layout`, where the
  non-head rank saves and the head computes every walker.
- SPARE: only under ``GF_LEGACY_RANK_LAYOUT=1``.

The MONITOR rank is not a role here. With ``GF_MONITOR_RANK=1`` the driver
splits the highest WORLD rank off first
(:func:`lisatools.globalfit.monitor.rank.split_monitor_rank`) and passes the
remaining run communicator to :func:`prepare_rank`. The 9mo/12mo launchers
run ``NGPUS=4`` (2 nodes x 2 GPUs) with ``NWALKERS=4``: world ranks 0-5, run
ranks 0 (head, walker 0), 1-3 (compute, one walker and one GPU each),
4 (saver), and world rank 5 is the monitor. That shape factorizes as
``(n_blocks, R, block) = (4, 1, 1)`` (:func:`factorize_layout`).

The mapping between ranks and GPUs is general in both directions:
``gpus_per_rank`` (a rank owns several devices, sharded in-process; env
``GPUS_PER_RANK``) and ``ranks_per_gpu`` (several ranks share one device;
env ``RANKS_PER_GPU``). At most one may exceed 1. The values are read from
the fit's general block (:class:`~lisatools.globalfit.stock.erebor.fit.EreborGeneralSettings`).

Environment knobs read here:

- ``GF_LEGACY_RANK_LAYOUT=1`` (:data:`LEGACY_ENV`): the pre-port roles (one
  compute rank owning the whole pool; the other non-saver ranks are SPARE
  and stop at startup). Launchers export ``0``.
- ``GF_GPU_ROUTING=1`` (:data:`GPU_ROUTING_ENV`): the unified GPU-count
  routing. It is OPT-IN; the 9mo/12mo jobs at ``NGPUS=4`` leave it unset.
- ``GF_ONE_WALKER_REPLICAS`` (:data:`ONE_WALKER_ENV`): ``0`` refuses any
  layout with several ranks per walker block.
- ``GF_LAYOUT_DRY_RUN=1``: :func:`layout_dry_run` prints the layout and
  tells the driver to exit before the build.

``mpi4py`` is imported lazily and only for real communicators, so the
in-process :mod:`~lisatools.globalfit.communication.fakecomm` stand-in runs
the same code.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import os
import sys
import threading
import traceback
import typing
import warnings

import numpy as np

LEGACY_ENV = "GF_LEGACY_RANK_LAYOUT"

#: parsed like the likelihood-fanout knob: any value other than
#: ``0``/``false``/``False``/empty enables replica mode (the default, when
#: unset); ``0``/``false``/``False``/empty refuses a one-walker run on
#: several compute ranks (today's error).
ONE_WALKER_ENV = "GF_ONE_WALKER_REPLICAS"

#: ★ The unified GPU-count routing (``n_compute = n_blocks * R`` at ANY
#: walker count: GPUs > walkers, GPUs == walkers, GPUs < walkers) is
#: **OPT-IN**, off unless this is set to ``1``/``true``/``yes``/``on``.
#:
#: TODO(gpu-routing): flip this default to ON once the cluster gates G3-G6
#: of ``docs/gpu-routing-test-campaign.md`` have passed at the production
#: shapes, then delete the knob and the ``gpu_routing`` plumbing entirely
#: (``_factorize``/``factorize_layout``/``build_layout`` keyword, the
#: ``WalkerBlockLayout.gpu_routing`` field, and the ``layout.gpu_routing``
#: check in ``GBSpecialBase._replica_band_weights``'s call site). The
#: unified rule is a strict superset of the legacy one, so the flip is a
#: deletion, not a rewrite.
#:
#: OFF reproduces the pre-2026-09-23 rule EXACTLY, and that is the whole
#: point of the knob: ``nwalkers`` must divide the compute-rank count
#: (``R == 1``, the walker-block layout) with the single ``nwalkers == 1``
#: carve-out that is one-walker replica mode. Any other shape raises the
#: error it always raised, now naming this knob as the way forward.
#:
#: It also pins the one behaviour that differs at a shape the legacy rule
#: DOES accept: in one-walker replica mode the GB band split stays the
#: legacy equal-BAND-COUNT one instead of the source-weighted one, so a
#: one-walker run on this branch is bit-identical to ``dev``.
GPU_ROUTING_ENV = "GF_GPU_ROUTING"


def gpu_routing_enabled(env=None) -> bool:
    """Report whether the unified GPU-count routing is switched on (default: no).

    Deliberately the OPPOSITE polarity to :data:`ONE_WALKER_ENV`, which is
    an opt-OUT of a shipped feature. This is an opt-IN to a new one, so an
    unset variable, an empty one and a typo all mean "legacy".

    Args:
        env (Mapping[str, str], optional): Environment to read. Default
            ``None`` reads :data:`os.environ`.

    Returns:
        bool: ``True`` only when ``GF_GPU_ROUTING`` is ``1``, ``true``,
        ``yes`` or ``on`` (case-insensitive, surrounding spaces ignored).

    Example:
        >>> from lisatools.globalfit.communication.ranks import gpu_routing_enabled
        >>> gpu_routing_enabled({"GF_GPU_ROUTING": "1"})
        True
        >>> gpu_routing_enabled({})
        False
    """
    raw = (os.environ if env is None else env).get(GPU_ROUTING_ENV, "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class RankRole(enum.Enum):
    """The role a rank of the run communicator plays (see the module docstring).

    Attributes:
        HEAD: Sequences the recipe, owns the full host state, computes
            walker block 0.
        COMPUTE: Computes one walker block and serves head commands.
        SAVER: Runs the HDF5 save loop (communicator size >= 3, or the
            non-head rank in the size-2 fallback of :func:`build_layout`).
        SPARE: Built then stopped at startup; ``GF_LEGACY_RANK_LAYOUT=1``
            only.
    """

    HEAD = "head"
    COMPUTE = "compute"
    SAVER = "saver"
    SPARE = "spare"  # legacy layout only


@dataclasses.dataclass(frozen=True)
class RankPlacement:
    """Where one rank sits: role, node, devices and walker block.

    Built by :func:`build_layout` (one per rank of the run communicator) and
    stored in :attr:`WalkerBlockLayout.placements`.

    Attributes:
        rank: Rank in the run communicator.
        role: The rank's :class:`RankRole`.
        node: Processor name of the rank's node.
        local_index: The rank's index among the ranks of its node (from a
            shared-memory ``Split_type``).
        w0: First walker of the rank's block (``0`` for SAVER / SPARE).
        w1: One past the last walker of the block (``0`` for SAVER / SPARE).
    """

    rank: int
    role: RankRole
    node: str
    local_index: int
    #: the rank's device list in the per-node pool's numbering (empty on CPU);
    #: several ranks list the same device when ranks_per_gpu > 1. SAVER /
    #: SPARE ranks list one device they build on and then release.
    devices: tuple
    #: this rank's index among the ranks sharing its device
    device_slot: int
    w0: int
    w1: int
    #: which walker BLOCK this rank computes (``compute.index(r) // R``)
    block_index: int = 0
    #: this rank's position among the R ranks sharing that block (lead = 0)
    replica_index: int = 0


@dataclasses.dataclass(frozen=True)
class WalkerBlockLayout:
    """The resolved rank layout of one run: roles, devices and walker blocks.

    Returned by :func:`build_layout` (identical on every rank) and stored as
    ``fit.rank_layout`` by :func:`prepare_rank`. The walkers are split into
    ``n_blocks`` equal contiguous blocks of ``block`` walkers; each block is
    computed by ``R = ranks_per_block`` compute ranks
    (``n_compute == n_blocks * R``). ``R == 1`` is the walker-block layout of
    the 9mo/12mo runs; ``R > 1`` is replica mode (:attr:`replica_mode`).

    The communicator helpers :meth:`make_fanout_comm`,
    :meth:`make_group_comm` and :meth:`make_reps_comm` are collectives; see
    each one for which ranks must call it.

    Attributes:
        size: Size of the run communicator.
        head_rank: The HEAD rank.
        saver_rank: The SAVER rank. Equals ``head_rank`` below 3 ranks,
            except in the size-2 fallback, where the non-head rank saves and
            the head computes everything.
        compute_ranks: Ranks that compute a walker block, head included, in
            ascending order. The position of a rank in this tuple is its
            fan-out rank (:meth:`fanout_rank`).
        nwalkers: Total walker count.
        block: Walkers per block.
        placements: ``{rank: RankPlacement}`` for every rank of the run
            communicator.
        ranks_per_gpu: Compute ranks sharing one device.
        legacy: ``True`` under ``GF_LEGACY_RANK_LAYOUT=1``.
    """

    size: int
    head_rank: int
    saver_rank: int
    compute_ranks: tuple
    nwalkers: int
    block: int
    placements: dict
    #: resolved devices per compute rank when uniform across nodes; ``None``
    #: when AUTO resolved to different per-node values (mixed node shapes)
    gpus_per_rank: typing.Optional[int] = 1
    #: True when ``gpus_per_rank`` was requested as AUTO (``None`` input)
    gpus_per_rank_auto: bool = False
    ranks_per_gpu: int = 1
    legacy: bool = False
    #: human-readable notes about non-default choices (the size-2 fallback, a
    #: gcd-1 AUTO resolution); each was also emitted as a ``UserWarning``
    notes: tuple = ()
    #: walker blocks; ``n_compute == n_blocks * ranks_per_block`` always
    n_blocks: int = 1
    #: ``R``: compute ranks sharing one walker block. 1 = today's walker-block
    #: layout; ``n_compute`` at ``nwalkers == 1`` = today's replica mode.
    ranks_per_block: int = 1
    #: True when ``ranks_per_block`` was requested as AUTO (``None`` input)
    ranks_per_block_auto: bool = True
    #: resolved value of :data:`GPU_ROUTING_ENV`, carried on the layout so a
    #: move can consult it without re-reading the environment (and so it
    #: lands in :meth:`describe`, hence the digest -- a rank that set the
    #: knob differently from its peers then fails the agreement check
    #: instead of silently routing differently).
    #: TODO(gpu-routing): delete with the knob; see :data:`GPU_ROUTING_ENV`.
    gpu_routing: bool = False

    @property
    def n_compute(self) -> int:
        """Number of compute ranks (head included)."""
        return len(self.compute_ranks)

    @property
    def replica_mode(self) -> bool:
        """Several ranks share a walker block, so their work must be dispersed.

        DERIVED, not a field (2026-09-23). It used to mean the single
        carve-out ``nwalkers == 1``; under the unified factorization it is
        simply ``R > 1``, which reproduces that case exactly (one block,
        ``R == n_compute``) and also covers GPUs > walkers at any walker
        count.
        """
        return int(self.ranks_per_block) > 1

    @property
    def worker_ranks(self) -> tuple:
        """Compute ranks other than the head (the ranks a fan-out sends to)."""
        return tuple(r for r in self.compute_ranks if r != self.head_rank)

    def is_single(self) -> bool:
        """Return ``True`` when one rank computes every walker (no fan-out)."""
        return self.n_compute == 1

    @property
    def n_replicas(self) -> int:
        """R. At ``R == 1`` this is 1 and at one walker it is ``n_compute``,
        reproducing both of the pre-2026-09-23 values."""
        return int(self.ranks_per_block)

    def replica_index(self, rank) -> int:
        """Position among the ranks sharing this rank's block (lead = 0).

        GROUP-RELATIVE. At ``n_blocks == 1`` it equals ``fanout_rank``, which
        is what one-walker replica mode has always used; at ``R == 1`` it is
        0 for every rank, as the walker-block layout has always used.
        """
        return int(self.placements[int(rank)].replica_index)

    def block_index(self, rank) -> int:
        """Which walker block this rank computes (``fanout_rank // R``)."""
        return int(self.placements[int(rank)].block_index)

    def ranks_in_block(self, b) -> tuple:
        """The R compute ranks sharing walker block ``b``, lead first."""
        b, R = int(b), int(self.ranks_per_block)
        if not (0 <= b < int(self.n_blocks)):
            raise ValueError(f"block {b} is outside [0, {int(self.n_blocks)})")
        return tuple(self.compute_ranks[b * R:(b + 1) * R])

    def block_lead(self, b) -> int:
        """Replica 0 of block ``b`` -- its representative in head-side merges."""
        return int(self.ranks_in_block(b)[0])

    @property
    def block_leads(self) -> tuple:
        """One representative rank per block, in block order (head first)."""
        return tuple(self.block_lead(b) for b in range(int(self.n_blocks)))

    def is_block_lead(self, rank) -> bool:
        """Return ``True`` when ``rank`` is replica 0 of its block."""
        return int(self.replica_index(rank)) == 0

    def role_of(self, rank) -> RankRole:
        """Return the :class:`RankRole` of ``rank``."""
        return self.placements[int(rank)].role

    def block_of(self, rank) -> tuple:
        """Return ``(w0, w1)``, the walker block ``[w0, w1)`` of ``rank``.

        ``(0, 0)`` for the SAVER and SPARE ranks.
        """
        p = self.placements[int(rank)]
        return p.w0, p.w1

    def owner_of(self, w) -> tuple:
        """Global walker ``w`` -> ``(owning block's LEAD rank, local row index)``.

        The inverse of :meth:`block_of`. Needed wherever a head-side GLOBAL
        walker index has to become an ACA ROW index: rows are per-rank, so a
        global index is out of range on every rank but its owners (the
        multi-rank F-stat fit's reference walker is exactly that case).

        At ``R > 1`` a walker has R owners, all holding the same rows, so
        this returns the block's LEAD as the deterministic representative --
        which is what a collective root must be. Use :meth:`owners_of` when
        the whole group is wanted. At ``R == 1`` the lead IS the sole owner,
        so this is unchanged from the walker-block layout.
        """
        w = int(w)
        if not (0 <= w < int(self.nwalkers)):
            raise ValueError(f"walker {w} is outside [0, {int(self.nwalkers)})")
        rank = self.block_lead(w // int(self.block))
        w0, _w1 = self.block_of(rank)
        return int(rank), w - int(w0)

    def owners_of(self, w) -> tuple:
        """Global walker ``w`` -> ``(every rank holding it, local row index)``."""
        w = int(w)
        if not (0 <= w < int(self.nwalkers)):
            raise ValueError(f"walker {w} is outside [0, {int(self.nwalkers)})")
        b = w // int(self.block)
        ranks = self.ranks_in_block(b)
        w0, _w1 = self.block_of(ranks[0])
        return ranks, w - int(w0)

    def local_gpus(self, rank):
        """Return the pool device ids of ``rank`` as a list, or ``None`` on CPU."""
        devices = self.placements[int(rank)].devices
        return list(devices) if devices else None

    def fanout_rank(self, rank) -> int:
        """This rank's index in the fan-out communicator (== compute-rank order)."""
        return self.compute_ranks.index(int(rank))

    # TODO check: keep/remove/document -- no caller outside tests/test_rank_layout.py
    def ranks_on_device(self, node, device) -> tuple:
        """Return the HEAD/COMPUTE ranks on ``node`` whose device list holds ``device``."""
        return tuple(
            r
            for r, p in sorted(self.placements.items())
            if p.node == str(node)
            and int(device) in p.devices
            and p.role in (RankRole.HEAD, RankRole.COMPUTE)
        )

    def make_fanout_comm(self, comm):
        """Split ``comm`` into the head+compute (fan-out) communicator.

        COLLECTIVE on ``comm`` (the run communicator): every rank, the saver
        included, must call it. Compute ranks keep their run-rank order, so
        a rank's rank in the new communicator is :meth:`fanout_rank`. The
        saver (and any SPARE rank) receives the null communicator.
        ``GlobalFit`` calls it only when the layout is not single.

        Args:
            comm: The run communicator (mpi4py ``Comm`` or
                :class:`~lisatools.globalfit.communication.fakecomm.FakeComm`).

        Returns:
            Comm: The fan-out communicator, or the null communicator off the
            compute ranks.
        """
        rank = int(comm.Get_rank())
        color = 0 if rank in self.compute_ranks else _undefined(comm)
        return comm.Split(color, key=rank)

    def make_group_comm(self, comm):
        """Split the FAN-OUT comm by walker block: size R, lead = group rank 0.

        The communicator for work shared by the replicas of one block --
        notably GB's per-unit delta-ledger allgather, which must be O(R) and
        must NOT put blocks with nothing to say to each other into unit
        lockstep. Non-compute ranks get the null comm.

        COLLECTIVE ON THE FAN-OUT COMM. Every rank OF THAT COMM -- i.e.
        every compute rank -- must call this the same number of times,
        including at ``R == 1`` where the group is a single rank. The saver
        is not a member and must not call it at all; callers skip it
        entirely when ``is_single()``.
        """
        rank = int(comm.Get_rank())
        if rank not in self.compute_ranks:
            return comm.Split(_undefined(comm), key=rank)
        return comm.Split(self.block_index(rank), key=self.replica_index(rank))

    def make_reps_comm(self, comm):
        """Split the FAN-OUT comm to one REPRESENTATIVE per block (replica 0).

        The communicator for anything that is per-walker rather than
        per-rank: the head-side likelihood gather, the state concatenation,
        the residual-hash digest. Reducing over every compute rank would
        count each walker R times; reducing over the leads counts it once.
        Non-lead and non-compute ranks get the null comm.

        COLLECTIVE, with the same rule as :meth:`make_group_comm`.
        """
        rank = int(comm.Get_rank())
        lead = rank in self.compute_ranks and self.is_block_lead(rank)
        color = 0 if lead else _undefined(comm)
        key = self.block_index(rank) if lead else rank
        return comm.Split(color, key=key)

    def describe(self) -> str:
        """Return the multi-line layout table logged at startup.

        One header line (sizes, block shape, ``gpus_per_rank`` /
        ``ranks_per_gpu``, routing, ``LEGACY`` / ``REPLICAS`` flags), one line
        per rank (role, node, devices, walker block, block / replica index)
        and one line per note. Identical on every rank by construction.
        """
        if self.gpus_per_rank_auto:
            gpk = "AUTO" if self.gpus_per_rank is None else f"AUTO->{self.gpus_per_rank}"
        else:
            gpk = str(self.gpus_per_rank)
        rpb = ("AUTO->%d" % self.ranks_per_block) if self.ranks_per_block_auto \
            else str(self.ranks_per_block)
        head = (
            f"walker-block layout: size={self.size} n_compute={self.n_compute} "
            f"nwalkers={self.nwalkers} n_blocks={self.n_blocks} block={self.block} "
            f"ranks_per_block={rpb} "
            f"gpus_per_rank={gpk} ranks_per_gpu={self.ranks_per_gpu} "
            f"gpu_routing={'on' if self.gpu_routing else 'OFF(legacy)'}"
            f"{' LEGACY' if self.legacy else ''}"
            f"{' REPLICAS' if self.replica_mode else ''}"
        )
        lines = [head]
        for r in range(self.size):
            p = self.placements[r]
            lines.append(
                f"  r{r:<3d} {p.role.value:<7s} node={p.node} local={p.local_index} "
                f"devices={list(p.devices)} slot={p.device_slot} walkers=[{p.w0},{p.w1}) "
                f"block={p.block_index} replica={p.replica_index}"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def digest(self) -> str:
        """Return the first 16 hex digits of the sha256 of :meth:`describe`.

        Answered by every rank to the ``ping`` command
        (:meth:`~lisatools.globalfit.communication.fanout.WalkerFanout.ping`),
        which raises when the ranks disagree.
        """
        return hashlib.sha256(self.describe().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# communicator shims (mpi4py only when the comm is real)
# --------------------------------------------------------------------------


def _comm_type_shared(comm):
    """``COMM_TYPE_SHARED`` from ``comm`` (fake comms) or from mpi4py."""
    value = getattr(comm, "COMM_TYPE_SHARED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.COMM_TYPE_SHARED


def _undefined(comm):
    """The ``UNDEFINED`` split colour from ``comm`` (fake comms) or from mpi4py."""
    value = getattr(comm, "UNDEFINED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.UNDEFINED


def _proc_name(comm) -> str:
    """The node name: ``comm.Get_processor_name()`` or mpi4py's module function."""
    fn = getattr(comm, "Get_processor_name", None)
    if fn is not None:
        return str(fn())
    from mpi4py import MPI

    return str(MPI.Get_processor_name())


# --------------------------------------------------------------------------
# roles + layout
# --------------------------------------------------------------------------


def resolve_roles(size, main_rank=0):
    """Assign head, saver and compute ranks for a communicator size.

    The saver is the highest non-head rank at ``size >= 3`` (aliased to the
    head below that); every other rank computes, the head included. No
    spares (the legacy layout and the size-2 fallback are applied later, in
    :func:`build_layout`).

    Args:
        size (int): Communicator size, >= 1.
        main_rank (int): The head rank. Default ``0``.

    Returns:
        tuple[int, int, tuple[int, ...]]: ``(head, saver, compute_ranks)``.

    Raises:
        ValueError: When ``size < 1`` or ``main_rank`` is not a rank of the
            communicator.

    Example:
        >>> from lisatools.globalfit.communication.ranks import resolve_roles
        >>> resolve_roles(5)
        (0, 4, (0, 1, 2, 3))
        >>> resolve_roles(2)
        (0, 0, (0, 1))
    """
    size = int(size)
    if size < 1:
        raise ValueError("communicator size must be >= 1")
    head = int(main_rank)
    if not (0 <= head < size):
        raise ValueError(f"main_rank {head} is not a rank of a size-{size} communicator")
    others = [r for r in range(size) if r != head]
    saver = others[-1] if size >= 3 else head
    compute = tuple(r for r in range(size) if r == head or r != saver)
    return head, saver, compute


def factorize_layout(nwalkers, n_compute, ranks_per_block=None, *, gpu_routing=True):
    """``(n_blocks, ranks_per_block, block)`` for a run shape -- THE public rule.

    The single source of truth for "how does this many walkers spread over
    this many compute ranks", exported so nothing has to MIRROR the
    arithmetic. Mirrored copies are the failure mode this exists to prevent:
    a planner or submit script that re-derives the rule keeps answering the
    old question after the rule changes, and reports a layout the engine
    will not build.

    Pure and side-effect free -- no communicator, no environment, no
    warnings -- so a planner, a dry run or a test can call it as cheaply as
    the launcher does. :func:`build_layout` uses it and adds the placement,
    the device pinning and the advisory warnings on top.

    With routing on and ``ranks_per_block=None`` (AUTO) the rule maximizes
    walker blocks: ``n_blocks = gcd(nwalkers, n_compute)``,
    ``R = n_compute // n_blocks``. With routing off only two shapes are
    accepted: ``nwalkers % n_compute == 0`` (``R = 1``) and ``nwalkers == 1``
    (``R = n_compute``).

    Args:
        nwalkers (int): Total walker count.
        n_compute (int): Number of compute ranks (head included).
        ranks_per_block (int, optional): ``R``, compute ranks per walker
            block. Default ``None`` is AUTO. Must divide ``n_compute``; a
            value other than 1 needs routing on.
        gpu_routing (bool): The resolved :data:`GPU_ROUTING_ENV` switch.
            Default ``True`` here, unlike at the launcher: this function
            answers "what IS the rule", so a planner or a dry run can price
            a shape the current deployment has not opted into yet.
            :func:`build_layout` defaults it from the environment instead,
            which is what decides whether a run actually starts.

    Returns:
        tuple[int, int, int]: ``(n_blocks, ranks_per_block, block)``, with
        ``block`` the walkers per block.

    Raises:
        ValueError: On a shape the engine would refuse, with the same
            message the launcher would print.

    Example:
        >>> from lisatools.globalfit.communication import factorize_layout
        >>> factorize_layout(4, 4)      # today's production run
        (4, 1, 1)
        >>> factorize_layout(4, 16)     # GPUs > walkers
        (4, 4, 1)
        >>> factorize_layout(24, 8)     # PE, blocks wider than one walker
        (8, 1, 3)
        >>> factorize_layout(1, 4)      # one-walker replica mode
        (1, 4, 1)
    """
    return _factorize(nwalkers, n_compute, ranks_per_block, [], gpu_routing=gpu_routing)


def _legacy_factorize(nwalkers, n_compute, ranks_per_block):
    """The pre-2026-09-23 rule, verbatim, for ``GF_GPU_ROUTING`` unset.

    Two shapes and no others: ``nwalkers % n_compute == 0`` (walker blocks,
    ``R = 1``) and ``nwalkers == 1`` (replica mode, ``R = n_compute``).
    Everything else raised, and still raises -- with the message extended to
    name the knob that lifts the restriction.

    TODO(gpu-routing): delete this function when the default flips; see
    :data:`GPU_ROUTING_ENV`.
    """
    _how = (
        f"Set {GPU_ROUTING_ENV}=1 to enable the unified GPU-count routing, "
        f"which spreads {nwalkers} walker(s) over {n_compute} compute ranks "
        f"as {math.gcd(nwalkers, n_compute)} block(s) of "
        f"{nwalkers // math.gcd(nwalkers, n_compute)} with "
        f"{n_compute // math.gcd(nwalkers, n_compute)} rank(s) each."
    )
    if ranks_per_block is not None and int(ranks_per_block) != 1:
        raise ValueError(
            f"RANKS_PER_BLOCK={ranks_per_block} needs the unified GPU-count "
            f"routing, which is off. {_how}"
        )
    if nwalkers == 1:
        return 1, int(n_compute), 1  # the one-walker replica carve-out
    if n_compute and nwalkers % n_compute:
        raise ValueError(
            f"nwalkers={nwalkers} is not divisible by the {n_compute} compute "
            f"ranks, so the walker-block layout cannot split it evenly. "
            f"{_how} Otherwise pick nwalkers as a multiple of {n_compute}, or "
            f"run fewer ranks."
        )
    return int(n_compute), 1, int(nwalkers // n_compute)


def _factorize(nwalkers, n_compute, ranks_per_block, notes, *, gpu_routing=True):
    """``n_compute = n_blocks * R``; returns ``(n_blocks, R, block)``.

    ONE factorization covers every (nwalkers, n_compute) pair, so there is
    no divisibility error any more and no ``nwalkers == 1`` carve-out.

    AUTO (``ranks_per_block is None``) MAXIMIZES BLOCKS:
    ``n_blocks = gcd(nwalkers, n_compute)``. Blocks are the cheap axis -- GB
    is sublinear in block width (~``B^0.65``), SOBBH is flat in rows, and the
    GB sub-band buffer saturates above a block width of ~2.5 -- while each
    replica costs a full ACA, inverse-PSD plane and whole-grid ``BandSorter``.
    So a GPU is worth more as another walker than as another replica, and
    AUTO spends it that way unless told otherwise.

    Both of today's regimes fall out unchanged: ``nwalkers % n_compute == 0``
    gives ``gcd == n_compute`` -> ``R = 1``, the walker-block layout; one
    walker gives ``gcd == 1`` -> ``R = n_compute``, the replica layout.

    ``gpu_routing=False`` (the deployed default; see :data:`GPU_ROUTING_ENV`)
    hands off to :func:`_legacy_factorize`, which accepts ONLY those two
    regimes. Because AUTO reproduces both of them exactly, the gate never
    changes the answer on a shape the legacy rule accepted -- it only
    restores the error on the shapes it refused.

    PURE, and the ``gpu_routing`` switch keeps it that way: the value is a
    PARAMETER, never an ``os.environ`` read, so a planner may poll the rule
    either way without the environment deciding for it.
    """
    nwalkers, n_compute = int(nwalkers), int(n_compute)
    if not gpu_routing:
        return _legacy_factorize(nwalkers, n_compute, ranks_per_block)
    if ranks_per_block is None:
        n_blocks = math.gcd(nwalkers, n_compute)
        R = n_compute // n_blocks
        if n_blocks == 1 and nwalkers > 1 and n_compute > 1:
            # Legal, but it replicates the WHOLE ensemble on every rank and
            # buys no walker parallelism at all -- almost always a typo in
            # NWALKERS or the rank count rather than an intent.
            # RECORDED, NOT WARNED. ``_factorize`` is the pure rule behind
            # the public :func:`factorize_layout`, which a planner or dry run
            # may call in a loop; emitting a ``UserWarning`` from here would
            # make the rule un-poll-able and put the advice in the caller's
            # stack instead of the launcher's. ``build_layout`` warns for
            # every note this adds -- that is the thing actually launching.
            notes.append(
                f"nwalkers={nwalkers} and {n_compute} compute ranks share no "
                f"common factor (gcd=1), so AUTO resolved to ONE block of "
                f"{nwalkers} walkers replicated on all {n_compute} ranks. That "
                f"gives no walker parallelism; pick nwalkers as a multiple of a "
                f"divisor of {n_compute}, or set RANKS_PER_BLOCK explicitly."
            )
    else:
        R = int(ranks_per_block)
        if R < 1:
            raise ValueError("ranks_per_block must be >= 1")
        if n_compute % R:
            divisors = [d for d in range(1, n_compute + 1) if n_compute % d == 0]
            raise ValueError(
                f"RANKS_PER_BLOCK={R} does not divide the compute-rank count "
                f"{n_compute}; legal values are {divisors}."
            )
        n_blocks = n_compute // R
    if nwalkers % n_blocks:
        _auto = math.gcd(nwalkers, n_compute)
        raise ValueError(
            f"nwalkers={nwalkers} does not divide into {n_blocks} equal walker "
            f"block(s) (RANKS_PER_BLOCK={ranks_per_block} over {n_compute} "
            f"compute ranks). Pick nwalkers as a multiple of {n_blocks}, or "
            f"leave RANKS_PER_BLOCK unset -- AUTO would use "
            f"{_auto} block(s) of {nwalkers // _auto}."
        )
    return int(n_blocks), int(R), int(nwalkers // n_blocks)


def devices_beyond_walkers_message(rank, devices, w0, w1, n_compute, nwalkers,
                                   gpu_routing=False) -> str:
    """The refusal for a rank that would drive more GPUs than it has walkers.

    Shared by :func:`build_layout` (the layout-time check) and
    ``GlobalFit.setup_acs`` (the build-time backstop for a path that never
    resolved a layout), so the two say the same thing.

    Args:
        rank (int): The offending rank.
        devices (Sequence[int]): Its device list.
        w0 (int): First walker of its block.
        w1 (int): One past the last walker of its block.
        n_compute (int): Number of compute ranks.
        nwalkers (int): Total walker count.
        gpu_routing (bool): Whether ``GF_GPU_ROUTING`` is on (changes the
            advice). Default ``False``.

    Returns:
        str: The error message naming both ways out (one GPU per rank with
        the replica routing, or enough walkers per rank).
    """
    n_dev, n_w = len(devices), int(w1) - int(w0)
    routing = ("it is already on, so this is GPUS_PER_RANK>1 on a replicated "
               "block" if gpu_routing else "it is OFF on this run")
    return (
        f"rank {rank} would drive {n_dev} GPUs {list(devices)} but holds only "
        f"{n_w} walker(s) [{w0}, {w1}) of {nwalkers}. A rank's GPUs shard its "
        f"WALKER rows -- GPUS_PER_RANK packs walkers onto one rank, it never "
        f"splits a walker -- so {n_dev - n_w} device(s) would hold no walker "
        f"and the run would die at the first GB gather ('Per-AC element size "
        f"mismatch across shards'). Two routes: (1) split every walker's work "
        f"over several GPUs: ONE GPU per rank plus the replica routing -- "
        f"unset GPUS_PER_RANK and set {GPU_ROUTING_ENV}=1 ({routing}; "
        f"n_compute = n_blocks x R, R ranks per walker block, GB/VGB split by "
        f"band range, MBH/EMRI/SOBBH likelihood rows by walker); or (2) keep "
        f"{n_dev} GPUs per rank and give every rank at least {n_dev} "
        f"walkers: NWALKERS a multiple of {max(1, int(n_compute)) * n_dev} "
        f"(= {n_compute} compute rank(s) x {n_dev} GPUs). A store's walker "
        f"count is fixed for its life, so (2) needs a fresh or rescaled store."
    )


def _refuse_devices_beyond_walkers(placements, n_compute, nwalkers, gpu_routing):
    """Raise when any computing rank owns more devices than walker rows.

    Job 764 (9mo, 2026-10-15): ``NGPUS=8 GPUS_PER_RANK=2`` on the 4-walker
    store gave 4 compute ranks with devices ``[0, 1]`` and ONE walker each.
    ``AnalysisContainerArray`` shards a rank's rows over its devices by
    contiguous ``np.array_split``, so the second device got an EMPTY shard
    and ``recipe.build_gb_moves`` died in ``_gather_per_gpu_to_single``
    ("Per-AC element size mismatch across shards: [3415857, 0]") two
    minutes into the build. Every rank computes this from the same
    allgathered table, so every rank raises together -- before any device is
    touched -- instead of one of them aborting the world mid-build.
    """
    for r, p in sorted(placements.items()):
        if p.role not in (RankRole.HEAD, RankRole.COMPUTE):
            continue
        if len(p.devices) > 1 and len(p.devices) > int(p.w1) - int(p.w0):
            raise ValueError(devices_beyond_walkers_message(
                r, p.devices, p.w0, p.w1, n_compute, nwalkers,
                gpu_routing=gpu_routing))


def build_layout(
    comm,
    nwalkers,
    gpu_pool,
    *,
    gpus_per_rank=None,
    ranks_per_gpu=1,
    ranks_per_block=None,
    main_rank=0,
    legacy=None,
    gpu_routing=None,
):
    """Resolve the identical layout on every rank (one collective ``allgather``).

    ``gpu_pool`` is the PER-NODE device list (the ``GPUS`` setting). Compute
    ranks on a node are ordered by world rank and assigned blocked:
    ``gpus_per_rank = k > 1`` -> devices ``pool[i*k:(i+1)*k]``;
    ``ranks_per_gpu = m > 1`` -> device ``pool[i // m]``, slot ``i % m``.

    ``gpus_per_rank=None`` (the default) is AUTO, resolved per node: a lone
    compute rank on a node (with ``ranks_per_gpu=1``) owns the whole
    per-node pool (``k = len(pool)``, today's in-process ``-n 1`` multi-GPU
    run); several compute ranks on a node instead get one device each
    (``k = 1``). An explicit int pins ``k`` on every node.

    ``gpu_routing=None`` (the default) resolves :data:`GPU_ROUTING_ENV`,
    which is OFF unless set -- so an existing runbook keeps the exact layout
    it had. Pass ``True``/``False`` to pin it (tests, dry runs).

    Special cases: a size-2 launch on a pool that cannot host two compute
    ranks makes rank 1 a dedicated saver and the head computes every walker
    (recorded as a note and warned). Any advice the factorization records
    (:func:`_factorize`) is re-emitted here as a ``UserWarning``.

    Args:
        comm: The run communicator (mpi4py or fake). COLLECTIVE: every rank
            must call this with the same arguments.
        nwalkers (int): Total walker count.
        gpu_pool (Sequence[int] or None): Per-node device ids; empty or
            ``None`` for a CPU run.
        gpus_per_rank (int, optional): ``k``; ``None`` is AUTO (see above).
        ranks_per_gpu (int): ``m``. Default ``1``.
        ranks_per_block (int, optional): ``R``; ``None`` is AUTO. See
            :func:`factorize_layout`.
        main_rank (int): The head rank. Default ``0``.
        legacy (bool, optional): Legacy roles. Default ``None`` reads
            ``GF_LEGACY_RANK_LAYOUT`` (``"1"`` enables).
        gpu_routing (bool, optional): Default ``None`` reads
            ``GF_GPU_ROUTING`` via :func:`gpu_routing_enabled`.

    Returns:
        WalkerBlockLayout: The resolved layout (same on every rank).

    Raises:
        ValueError: On ``gpus_per_rank`` / ``ranks_per_gpu`` < 1 or both > 1,
            a shape :func:`factorize_layout` refuses, ``R > 1`` with
            ``GF_ONE_WALKER_REPLICAS=0``, more compute ranks on a node than
            the pool supports, or a rank owning more GPUs than walkers.

    Example:
        Two-node layout in one process with the fake communicator:

        .. code-block:: python

            from lisatools.globalfit.communication import FakeWorld, build_layout

            world = FakeWorld(5, nodes=[0, 0, 1, 1, 1])
            layouts = world.run(lambda r, comm: build_layout(comm, 4, [0, 1]))
            print(layouts[0].describe())
    """
    if legacy is None:
        legacy = os.environ.get(LEGACY_ENV, "0") == "1"
    if gpu_routing is None:
        gpu_routing = gpu_routing_enabled()
    gpu_routing = bool(gpu_routing)
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    head, saver, compute = resolve_roles(size, main_rank)
    m = int(ranks_per_gpu)
    k_explicit = None if gpus_per_rank is None else int(gpus_per_rank)
    if m < 1 or (k_explicit is not None and k_explicit < 1):
        raise ValueError("gpus_per_rank and ranks_per_gpu must both be >= 1")
    if k_explicit is not None and k_explicit > 1 and m > 1:
        raise ValueError("at most one of gpus_per_rank / ranks_per_gpu may exceed 1")
    pool = [int(g) for g in (gpu_pool or [])]
    if legacy:
        compute = (head,)
    notes = []
    if size == 2 and not legacy and pool and len(pool) * m // (k_explicit or 1) < 2:
        # A `-n 2` launch on a pool that cannot host two compute ranks: instead
        # of the over-subscription error, rank 1 becomes the dedicated saver
        # (user ruling 2026-09-15). The head then computes every walker exactly
        # as a single-rank run does.
        other = [r for r in range(size) if r != head][0]
        saver = other
        compute = (head,)
        note = (
            f"size-2 launch on a per-node GPU pool {pool} that supports only "
            f"{len(pool) * m // (k_explicit or 1)} compute rank(s): rank {other} runs as the "
            "dedicated saver and the head computes all walkers. To use two "
            "compute ranks on this pool set RANKS_PER_GPU=2; for synchronous "
            "saves with no saver rank launch with -n 1."
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=2)
    nwalkers = int(nwalkers)
    n_compute = len(compute)
    _n_notes = len(notes)
    n_blocks, R, block = _factorize(
        nwalkers, n_compute, ranks_per_block, notes, gpu_routing=gpu_routing
    )
    # The rule RECORDS its advice; the launcher is what surfaces it (see the
    # comment in ``_factorize``). Anything it appended is worth a warning
    # here, where the stack points at the code actually starting a run.
    for _note in notes[_n_notes:]:
        warnings.warn(_note, UserWarning, stacklevel=2)
    if R > 1:
        _one_walker_env = os.environ.get(ONE_WALKER_ENV, "1")
        if _one_walker_env.strip() in ("0", "false", "False", ""):
            raise ValueError(
                f"nwalkers={nwalkers} on {n_compute} compute ranks resolves to "
                f"{n_blocks} walker block(s) shared by {R} ranks each, which "
                f"{ONE_WALKER_ENV}=0 disables. Unset it, set RANKS_PER_BLOCK=1 "
                f"(needs nwalkers divisible by {n_compute}), or run fewer ranks."
            )

    if size == 1 or not hasattr(comm, "Split_type"):
        table = [(_proc_name(comm), 0)]
    else:
        node_comm = comm.Split_type(_comm_type_shared(comm), key=rank)
        table = comm.allgather((_proc_name(comm), int(node_comm.Get_rank())))
        free = getattr(node_comm, "Free", None)
        if free is not None:
            free()

    by_node = {}
    for r in range(size):
        by_node.setdefault(str(table[r][0]), []).append(r)

    placements = {}
    resolved_ks = []
    for node, ranks in by_node.items():
        comp_here = [r for r in ranks if r in compute]
        n_comp_here = len(comp_here)
        if k_explicit is not None:
            k = k_explicit
        elif pool and n_comp_here == 1 and m == 1:
            k = len(pool)  # AUTO: a lone compute rank drives the whole pool (today's -n 1)
        else:
            k = 1
        if comp_here:
            resolved_ks.append(k)
        if pool and not legacy:
            capacity = len(pool) * m // k
            if len(comp_here) > capacity:
                raise ValueError(
                    f"node {node}: {len(comp_here)} compute ranks but the per-node GPU pool "
                    f"{pool} supports at most {capacity} "
                    f"(gpus_per_rank={k}, ranks_per_gpu={m})"
                )
        for r in ranks:
            local_index = int(table[r][1])
            if r == head:
                role = RankRole.HEAD
            elif r == saver and saver != head:
                role = RankRole.SAVER
            elif r in compute:
                role = RankRole.COMPUTE
            else:
                role = RankRole.SPARE
            if role in (RankRole.SAVER, RankRole.SPARE):
                # builds on a device like everyone else, then releases it
                devices = (pool[local_index % len(pool)],) if pool else ()
                placements[r] = RankPlacement(r, role, node, local_index, devices, 0, 0, 0)
                continue
            i = comp_here.index(r)
            if not pool:
                devices, slot = (), 0
            elif legacy:
                devices, slot = tuple(pool), 0
            elif k > 1:
                devices, slot = tuple(pool[i * k : (i + 1) * k]), 0
            else:
                devices, slot = (pool[i // m],), i % m
            # ONE general expression (2026-09-23). It reproduces both former
            # branches exactly: at R == 1 ``bi`` is the compute-rank index and
            # the blocks tile the ensemble; at n_blocks == 1 every rank gets
            # ``(0, block)``, which is ``(0, 1)`` at one walker.
            bi, ri = divmod(compute.index(r), R)
            w0, w1 = bi * block, (bi + 1) * block
            placements[r] = RankPlacement(
                r, role, node, local_index, devices, slot, w0, w1,
                block_index=bi, replica_index=ri,
            )
    _refuse_devices_beyond_walkers(placements, len(compute), nwalkers, gpu_routing)
    if k_explicit is not None:
        resolved_k = k_explicit
    elif resolved_ks and len(set(resolved_ks)) == 1:
        resolved_k = resolved_ks[0]
    else:
        resolved_k = None
    return WalkerBlockLayout(
        size=size,
        head_rank=head,
        saver_rank=saver,
        compute_ranks=tuple(compute),
        nwalkers=nwalkers,
        block=block,
        placements=placements,
        gpus_per_rank=resolved_k,
        gpus_per_rank_auto=k_explicit is None,
        ranks_per_gpu=m,
        legacy=bool(legacy),
        notes=tuple(notes),
        n_blocks=n_blocks,
        ranks_per_block=R,
        ranks_per_block_auto=ranks_per_block is None,
        gpu_routing=gpu_routing,
    )


def derive_rank_seed(base_seed, layout, rank) -> int:
    """Derive the deterministic per-compute-rank seed of a run.

    ``SeedSequence([base_seed, 0x5AFE]).spawn(n_compute)[i]`` with ``i`` the
    rank's fan-out rank, reduced to one ``uint32``. Used by
    ``GlobalFit._seed_rank_streams`` to seed the global numpy / cupy streams
    of each compute rank (multi-rank runs only).

    Args:
        base_seed (int): The run's seed base (``GlobalFit._resolve_seed_base``).
        layout (WalkerBlockLayout): The run layout.
        rank (int): A compute rank (head included).

    Returns:
        int: The rank's seed, ``0 <= seed < 2**32``.
    """
    sequence = np.random.SeedSequence([int(base_seed), 0x5AFE])
    children = sequence.spawn(layout.n_compute)
    child = children[layout.fanout_rank(rank)]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def rank_build_seed(fit):
    """Per-rank seed for objects built during ``fit.build()``; ``None`` for entropy.

    ``run.py::_seed_rank_streams`` seeds the GLOBAL numpy/cupy streams per
    rank, but it runs long after the build: priors and proposal objects
    constructed inside ``fit.build()`` own private ``Generator``s that were
    already drawn from OS entropy by then. Seeding those from the bare
    ``general.random_seed`` would hand EVERY rank the identical stream, so
    the walker blocks' prior draws and RJ birth candidates would correlate.
    This is the build-time analogue of :func:`derive_rank_seed`.

    DOMAIN SEPARATION. The run seed also feeds ``np.random.seed()`` /
    ``cupy.random.seed()`` on this rank (``_seed_rank_streams``), so returning
    it (or ``derive_rank_seed``'s value) unchanged would hand the build-time
    generators the very integer the global streams run on -- independent only
    by the accident of different bit-mixing. The build base is therefore
    ``SeedSequence([run_seed, 0xB01D])`` first, the way ``derive_rank_seed``
    tags its own domain with ``0x5AFE``; every case below derives from it, so
    a build seed is never the bare run seed and never equals
    ``derive_rank_seed(run_seed, layout, rank)``.

    * ``general.random_seed is None`` -> ``None`` (entropy, exactly today);
    * a layout is resolved (``prepare_rank`` ran) -> the rank's sub-seed;
    * no layout (single process, ``fit.sample()``) -> the build base itself.

    A rank with no walker block (the saver) has nothing to sample, so it
    takes the head's sub-seed rather than an index error.

    Called by
    :meth:`~lisatools.globalfit.stock.base.StockGlobalFit.prepare_branch_settings`
    to fill ``build_seed`` on the branch blocks that declare it (GB, VGB).

    Args:
        fit: The stock fit (reads ``fit.general.random_seed``,
            ``fit.rank_layout`` and ``fit.rank``).

    Returns:
        int or None: The build seed, or ``None`` when ``random_seed`` is
        unset.
    """
    seed = getattr(getattr(fit, "general", None), "random_seed", None)
    if seed is None:
        return None
    base = int(np.random.SeedSequence(
        [int(seed), 0xB01D]).generate_state(1, dtype=np.uint32)[0])
    layout = getattr(fit, "rank_layout", None)
    if layout is None:
        return base
    rank = getattr(fit, "rank", None)
    rank = layout.head_rank if rank is None else int(rank)
    if rank not in layout.compute_ranks:
        rank = layout.head_rank
    return derive_rank_seed(base, layout, rank)


# --------------------------------------------------------------------------
# device pinning (must run BEFORE any CUDA initialisation on the rank)
# --------------------------------------------------------------------------

_CUDART_NAMES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.dylib",
)


def _load_cudart():
    """Load the CUDA runtime library via ctypes; ``None`` when none is found."""
    import ctypes

    for name in _CUDART_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cudart_device_count():
    """``cudaGetDeviceCount`` via ctypes, or ``None`` when no runtime is loadable."""
    import ctypes

    lib = _load_cudart()
    if lib is None:
        return None
    count = ctypes.c_int(0)
    if lib.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return None
    return int(count.value)


def _cudart_set_device(device):
    """``cudaSetDevice(device)`` via ctypes; raises ``RuntimeError`` on failure."""
    lib = _load_cudart()
    if lib is None:
        raise RuntimeError("cudaSetDevice fallback requested but no CUDA runtime is loadable")
    if lib.cudaSetDevice(int(device)) != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed")


def select_rank_device(
    layout,
    rank,
    *,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
    logger=None,
):
    """Pin this process to its devices. Returns ``(gpus, mode)``.

    ``gpus`` is what ``general_info.gpus`` must become on this rank
    (``None`` on CPU). Modes: ``"cpu"``; ``"legacy"`` (pool untouched);
    ``"visible"`` (``CUDA_VISIBLE_DEVICES`` narrowed, the rank sees its
    devices as ``0..k-1``); ``"setdevice"`` (the runtime was already
    initialised, e.g. by a CUDA-aware MPI, so the env is restored and
    ``cudaSetDevice`` pins the first device; the rank keeps the pool ids).

    When ``CUDA_VISIBLE_DEVICES`` is already set (e.g. narrowed by Slurm),
    the pool ids index the currently visible list; when unset they are
    physical ids. Must run before any CUDA initialisation on the rank.

    Args:
        layout (WalkerBlockLayout): The run layout.
        rank (int): This process's rank in the run communicator.
        environ (MutableMapping[str, str], optional): Environment to edit.
            Default ``None`` is :data:`os.environ`.
        device_count_fn (Callable[[], int or None], optional): Device count
            seen by the CUDA runtime. Default: ``cudaGetDeviceCount`` via
            ctypes (``None`` when no runtime loads).
        set_device_fn (Callable[[int], None], optional): Device pinning for
            the ``"setdevice"`` fallback. Default: ``cudaSetDevice`` via
            ctypes.
        logger (logging.Logger, optional): Receives a warning on the
            ``"setdevice"`` fallback.

    Returns:
        tuple[list[int] or None, str]: ``(gpus, mode)`` as above.

    Raises:
        ValueError: When the pool ids do not fit the inherited
            ``CUDA_VISIBLE_DEVICES`` list.
    """
    environ = os.environ if environ is None else environ
    device_count_fn = _cudart_device_count if device_count_fn is None else device_count_fn
    set_device_fn = _cudart_set_device if set_device_fn is None else set_device_fn
    placement = layout.placements[int(rank)]
    if not placement.devices:
        return None, "cpu"
    if layout.legacy:
        return list(placement.devices), "legacy"

    previous = environ.get("CUDA_VISIBLE_DEVICES")
    if previous is None:
        # the env var is genuinely unset: the pool ids are physical device ids
        physical = [str(d) for d in placement.devices]
    else:
        # SET (possibly to "", meaning explicitly NO devices visible): pool ids
        # index the CURRENTLY visible set (Slurm may already have narrowed it,
        # e.g. via --gpus-per-task/--gpu-bind, down to as little as one device)
        visible = [v.strip() for v in previous.split(",") if v.strip()]
        try:
            physical = [visible[d] for d in placement.devices]
        except IndexError:
            raise ValueError(
                f"rank {rank}: device pool ids {list(placement.devices)} index the "
                f"inherited CUDA_VISIBLE_DEVICES={previous!r} (visible devices={visible!r}), "
                "which does not have that many entries. This usually means the launcher "
                "already bound this task to its own device(s) (Slurm "
                "--gpus-per-task/--gpu-bind), so the layout's pool ids double-count that "
                "binding. Fix by either launching without per-task GPU binding "
                "(--gpu-bind=none) so every rank sees the full node pool, or by setting "
                "GPUS to the per-task pool this rank should see (matching what "
                "CUDA_VISIBLE_DEVICES already narrowed it to)."
            ) from None
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)

    count = device_count_fn()
    if count is None or count == len(placement.devices):
        return list(range(len(placement.devices))), "visible"

    # the runtime saw the pool before we narrowed the env: fall back to cudaSetDevice
    if previous is None:
        environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environ["CUDA_VISIBLE_DEVICES"] = previous
    set_device_fn(placement.devices[0])
    environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if logger is not None:
        logger.warning(
            "rank %d: CUDA runtime already initialised (%d devices visible); pinned device %d "
            "with cudaSetDevice instead of CUDA_VISIBLE_DEVICES",
            rank,
            count,
            placement.devices[0],
        )
    return list(placement.devices), "setdevice"


def prepare_rank(
    fit,
    comm,
    *,
    logger=None,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
):
    """THE driver hook: resolve the layout and pin this rank's device BEFORE ``fit.build()``.

    Idempotent (returns the stored layout on a second call). Reads the
    pre-build settings ``fit.general.nwalkers`` / ``.gpus`` (the per-node
    pool) / ``.gpus_per_rank`` / ``.ranks_per_gpu`` / ``.ranks_per_block``
    and ``fit.main_rank``; writes the rank-local
    ``fit.general.gpus``, ``fit.rank``, ``fit.rank_layout`` and
    ``fit.rank_device_mode``. ``gpus_per_rank`` is passed through unchanged
    (``None`` stays ``None``, AUTO) — it is never coerced to 1.

    ``fit.rank`` is stamped here (``GlobalFit.__init__`` sets it too, but
    that is AFTER the build) so build-time helpers — :func:`rank_build_seed`
    — can tell which block this process owns.

    The production driver (``scripts/fstat_proposal/run_combined_staged.py``)
    calls it on the run communicator returned by
    :func:`~lisatools.globalfit.monitor.rank.split_monitor_rank`, then
    :func:`layout_dry_run`, then ``fit.build()``.

    Args:
        fit: The pre-build stock fit
            (:class:`~lisatools.globalfit.stock.base.StockGlobalFit`).
        comm: The run communicator. COLLECTIVE: every rank calls it.
        logger (logging.Logger, optional): Receives the layout table and the
            rank's device mode at INFO.
        environ (MutableMapping[str, str], optional): Forwarded to
            :func:`select_rank_device`.
        device_count_fn (Callable, optional): Forwarded to
            :func:`select_rank_device`.
        set_device_fn (Callable, optional): Forwarded to
            :func:`select_rank_device`.

    Returns:
        WalkerBlockLayout: The resolved layout.

    Raises:
        RuntimeError: When called on a built fit in a multi-rank run (the
            build has already allocated on a device).
        ValueError: Propagated from :func:`build_layout` /
            :func:`select_rank_device`.

    Example:
        .. code-block:: python

            from mpi4py import MPI
            from lisatools.globalfit.stock import erebor
            from lisatools.globalfit.communication import prepare_rank

            fit = erebor.all_sources(nwalkers=4)
            layout = prepare_rank(fit, MPI.COMM_WORLD)
            fit.build()
            fit.run(MPI.COMM_WORLD)
    """
    layout = getattr(fit, "rank_layout", None)
    if layout is not None:
        return layout
    if int(comm.Get_size()) > 1 and bool(getattr(fit, "built", False)):
        raise RuntimeError(
            "prepare_rank must run BEFORE fit.build(): the build allocates on the device"
        )
    general = fit.general
    pool = list(general.gpus) if getattr(general, "gpus", None) else []
    layout = build_layout(
        comm,
        int(general.nwalkers),
        pool,
        gpus_per_rank=getattr(general, "gpus_per_rank", None),
        ranks_per_gpu=int(getattr(general, "ranks_per_gpu", 1) or 1),
        ranks_per_block=getattr(general, "ranks_per_block", None),
        main_rank=int(getattr(fit, "main_rank", 0) or 0),
    )
    # capture BEFORE select_rank_device narrows/restores it in place
    env = os.environ if environ is None else environ
    inherited_cvd = env.get("CUDA_VISIBLE_DEVICES", "<unset>")
    gpus, mode = select_rank_device(
        layout,
        int(comm.Get_rank()),
        environ=environ,
        device_count_fn=device_count_fn,
        set_device_fn=set_device_fn,
        logger=logger,
    )
    if pool:
        general.gpus = gpus
    fit.rank = int(comm.Get_rank())
    fit.rank_layout = layout
    fit.rank_device_mode = mode
    if logger is not None:
        logger.info(
            "%s\nrank %d device mode: %s (inherited CUDA_VISIBLE_DEVICES=%s)",
            layout.describe(),
            comm.Get_rank(),
            mode,
            inherited_cvd,
        )
    return layout


def layout_dry_run(layout, comm, *, environ=None, out=print) -> bool:
    """Preflight: print this rank's layout view and stop, iff ``GF_LAYOUT_DRY_RUN=1``.

    Drivers call this right after ``prepare_rank`` and BEFORE ``fit.build()``:
    ``if layout_dry_run(layout, MPI.COMM_WORLD): sys.exit(0)``. Returns True
    (armed, caller must stop) or False (unset, no-op -- single-process
    behaviour is unchanged).

    ``layout.describe()`` is the same full table on every rank (resolved by
    one collective allgather in ``build_layout``). Under real MPI each rank
    is its own process calling this once, so "every rank prints" falls out
    of SPMD execution; the head's copy prints unprefixed, and non-head ranks
    prefix theirs with ``rank_tag`` since this runs before ``GlobalFit``
    exists and installs ``prefix_stdout`` for the run proper. With
    ``comm=None`` (no rank to distinguish) the text is printed unprefixed.

    Args:
        layout (WalkerBlockLayout): The resolved layout.
        comm: The run communicator, or ``None``.
        environ (Mapping[str, str], optional): Environment to read. Default
            ``None`` is :data:`os.environ`.
        out (Callable[[str], None]): Printer. Default :func:`print`.

    Returns:
        bool: ``True`` when ``GF_LAYOUT_DRY_RUN=1`` (the table was printed and
        the caller must stop), else ``False``.
    """
    env = os.environ if environ is None else environ
    if env.get("GF_LAYOUT_DRY_RUN", "0") != "1":
        return False
    text = layout.describe()
    if comm is not None:
        rank = int(comm.Get_rank())
        if rank != layout.head_rank:
            tag = rank_tag(layout, rank)
            text = "\n".join(f"[{tag}] {line}" for line in text.splitlines())
    out(text)
    return True


# --------------------------------------------------------------------------
# failure + logging helpers
# --------------------------------------------------------------------------


def install_mpi_abort_on_error(comm):
    """Route any rank's uncaught exception (main thread or threads) through ``comm.Abort``.

    Under ``mpiexec`` a crashed rank otherwise leaves the survivors blocked in
    their receive loops for the whole allocation. Returns ``comm`` when it
    installed the hooks, ``None`` for a single-process run (normal Python
    exception behaviour is kept there).

    Replaces :data:`sys.excepthook` (and :data:`threading.excepthook` where
    available) with a hook that prints an ``[MPI-ABORT]`` banner and the
    traceback to stderr, then calls ``comm.Abort(1)`` (``os._exit(1)`` if
    that fails). The production driver installs it on ``MPI.COMM_WORLD``
    (so the monitor rank is aborted too); ``GlobalFit`` also installs it on
    the run communicator.

    Args:
        comm: The communicator to abort, or ``None``.

    Returns:
        Comm or None: The ``comm`` passed in when the hooks were installed
        (size >= 2), else ``None``.
    """
    if comm is None or int(comm.Get_size()) < 2:
        return None
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    def _hook(exc_type, exc, tb):
        try:
            print(
                f"\n[MPI-ABORT] rank {rank} of {size} raised {exc_type.__name__}: {exc}\n"
                "[MPI-ABORT] aborting ALL ranks so the job fails fast instead of hanging.",
                file=sys.stderr,
                flush=True,
            )
            if exc_type is not KeyboardInterrupt:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - never mask the abort
            pass
        finally:
            try:
                comm.Abort(1)
            except Exception:  # noqa: BLE001
                os._exit(1)

    sys.excepthook = _hook
    if hasattr(threading, "excepthook"):

        def _thread_hook(args):
            _hook(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = _thread_hook
    return comm


def rank_tag(layout, rank) -> str:
    """Return the short log tag of a rank.

    ``r<rank>/head``, ``r<rank>/saver``, ``r<rank>/spare``, or
    ``r<rank>/c<fanout rank>`` for a compute rank. Used as the stdout prefix
    of the non-head ranks (:func:`prefix_stdout`).

    Args:
        layout (WalkerBlockLayout): The run layout.
        rank (int): A rank of the run communicator.

    Returns:
        str: The tag.
    """
    role = layout.role_of(rank)
    if role == RankRole.HEAD:
        return f"r{int(rank)}/head"
    if role == RankRole.SAVER:
        return f"r{int(rank)}/saver"
    if role == RankRole.SPARE:
        return f"r{int(rank)}/spare"
    return f"r{int(rank)}/c{layout.fanout_rank(rank)}"


class _PrefixedStream:
    """Line-prefixing wrapper for a text stream (worker-rank stdout)."""

    def __init__(self, stream, prefix):
        self._stream = stream
        self._prefix = prefix
        self._at_line_start = True

    def write(self, text):
        out = []
        for chunk in str(text).splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._prefix)
            out.append(chunk)
            self._at_line_start = chunk.endswith("\n")
        self._stream.write("".join(out))

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._stream, name)


def prefix_stdout(tag):
    """Prefix every stdout line of this process with ``"[tag] "`` (idempotent).

    Wraps :data:`sys.stdout` in a line-prefixing stream once; a second call
    keeps the first wrapper (and its tag).

    Args:
        tag (str): The prefix text, e.g. from :func:`rank_tag`.

    Returns:
        TextIO: The (wrapped) :data:`sys.stdout`.
    """
    if not isinstance(sys.stdout, _PrefixedStream):
        sys.stdout = _PrefixedStream(sys.stdout, f"[{tag}] ")
    return sys.stdout
