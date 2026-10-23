"""Shared utilities for LISA Analysis Tools.

Submodules (none is imported here; import them directly):

- :mod:`~lisatools.utils.constants`: physical constants from ``lisaconstants`` plus legacy
  LISA literals.
- :mod:`~lisatools.utils.utility`: array-module lookup (``get_array_module``, ``asnumpy``),
  windows, ``AET``, band-grouping helpers.
- :mod:`~lisatools.utils.device`: multi-GPU device-pinning helpers.
- :mod:`~lisatools.utils.parallelbase`: ``LISAToolsParallelModule``, the backend-dispatch
  base class.
- :mod:`~lisatools.utils.exceptions`: package exceptions.
- :mod:`~lisatools.utils.citations`: citation registry (``lisatools_citations`` CLI).
- :mod:`~lisatools.utils.stagetimer`: opt-in device-synced stage timing.
- :mod:`~lisatools.utils.lisaconstants_header`: generator of ``lisaconstants_values.h``.
- :mod:`~lisatools.utils.typing`: array type aliases.
"""
