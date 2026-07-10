"""LISA-response dispatch base — backward-compatibility shim.

The LISA-response classes now inherit :class:`LISAToolsParallelModule`
directly (the single lisatools backend-dispatch base). This module keeps the
historical names importable:

* ``FastLISAResponseParallelModule`` — legacy name for the response base.
* ``LISAToolsResponseParallelModule`` — post-Phase-3L.7k descriptive name.

Both are now aliases of :class:`lisatools.utils.parallelbase.LISAToolsParallelModule`,
which carries the ``"lisatools"`` backend prefix and the
``GPU_RECOMMENDED_WITH_JAX`` helper. Existing consumers (including GB / SOBBH
subclasses that override ``_BACKEND_PREFIX``) keep working unchanged; new code
should inherit ``LISAToolsParallelModule`` directly.
"""

from ..utils.parallelbase import LISAToolsParallelModule

# Backward-compatible aliases (same class object).
FastLISAResponseParallelModule = LISAToolsParallelModule
LISAToolsResponseParallelModule = LISAToolsParallelModule

__all__ = [
    "LISAToolsParallelModule",
    "FastLISAResponseParallelModule",
    "LISAToolsResponseParallelModule",
]
