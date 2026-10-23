"""Multi-rank communication layer of the global fit.

One MPI rank per GPU (general in both directions via ``gpus_per_rank`` /
``ranks_per_gpu``), equal static walker blocks, a head rank that sequences
the recipe and also computes block 0, computation ranks that serve
head-directed commands, and the unchanged saver rank. See
``docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md``.

Modules and entry points:

- :mod:`~lisatools.globalfit.communication.ranks`: roles and layout
  (:func:`prepare_rank`, the driver hook run before ``fit.build()``;
  :func:`build_layout`, :func:`factorize_layout`, :class:`WalkerBlockLayout`),
  device pinning, rank seeds, abort-on-error and stdout prefixes.
- :mod:`~lisatools.globalfit.communication.fanout`: the head-side
  :class:`WalkerFanout` and the compute-rank :class:`ComputeService` loop,
  with the command / reply message schemas.
- :mod:`~lisatools.globalfit.communication.walkerslice`: cutting a walker
  block out of a ``GFState`` (:func:`slice_state`) and writing it back
  (:func:`merge_state`).
- :mod:`~lisatools.globalfit.communication.rowfanout`: row scatter for the
  replica layouts (``R > 1``).
- :mod:`~lisatools.globalfit.communication.fakecomm`: :class:`FakeWorld`,
  an in-process stand-in for the mpi4py subset, for tests.

In the 9mo/12mo jobs (``NGPUS=4``, ``NWALKERS=4``, monitor rank on) the world
has 6 ranks: the driver splits off world rank 5 as the monitor
(:mod:`lisatools.globalfit.monitor.rank`); on the 5-rank run communicator
rank 0 is the head (walker 0), ranks 1-3 compute walkers 1-3 (one GPU each)
and rank 4 is the saver.

``mpi4py`` is imported lazily and only for real communicators; the
:mod:`fakecomm` simulator runs the same code in one process for tests.
"""

from .fakecomm import FakeAbort, FakeComm, FakeCommNull, FakeWorld
from .fanout import (
    LIKELIHOOD_OP,
    PING_OP,
    STOP_OP,
    ComputeService,
    RemoteWorkerError,
    WalkerFanout,
    concat_blocks,
)
from .ranks import (
    GPU_ROUTING_ENV,
    LEGACY_ENV,
    RankPlacement,
    RankRole,
    WalkerBlockLayout,
    build_layout,
    derive_rank_seed,
    factorize_layout,
    gpu_routing_enabled,
    install_mpi_abort_on_error,
    prefix_stdout,
    prepare_rank,
    rank_tag,
    resolve_roles,
    select_rank_device,
)
from .walkerslice import WALKER_INDS_KEY, merge_state, slice_state

__all__ = [
    "WALKER_INDS_KEY",
    "merge_state",
    "slice_state",
    "FakeAbort",
    "FakeComm",
    "FakeCommNull",
    "FakeWorld",
    "LIKELIHOOD_OP",
    "PING_OP",
    "STOP_OP",
    "ComputeService",
    "RemoteWorkerError",
    "WalkerFanout",
    "concat_blocks",
    "GPU_ROUTING_ENV",
    "LEGACY_ENV",
    "RankPlacement",
    "RankRole",
    "WalkerBlockLayout",
    "build_layout",
    "derive_rank_seed",
    "factorize_layout",
    "gpu_routing_enabled",
    "install_mpi_abort_on_error",
    "prefix_stdout",
    "prepare_rank",
    "rank_tag",
    "resolve_roles",
    "select_rank_device",
]
