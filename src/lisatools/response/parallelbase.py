"""LISA-response dispatch base.

Phase 3L.7k (2026-06-04): the prefix used to resolve `force_backend`
strings changed from ``fastlisaresponse`` to ``lisatools`` because the
fastlisaresponse_<flavor> backend family is being retired in favor of
the LAT ``LISAToolsBackend`` (now carrying all of the LISA-response
Wraps it used to hand off through fastlisaresponse). The class name
stays the same for compat -- a future cleanup will rename it to
``LISAToolsResponseParallelModule`` alongside the broader
post-reorg Python sweep.
"""

from gpubackendtools import ParallelModuleBase


class FastLISAResponseParallelModule(ParallelModuleBase):
    # Phase 3L.7k (2026-06-04): class-level override that lets GB / SOBBH
    # subclasses re-prefix to ``gbgpu`` or ``bbhx`` without rewriting
    # __init__. The default ``lisatools`` covers FDSpline / TDSpline /
    # generic-response consumers; GBTDIonTheFly / GBFDTDIonTheFly /
    # SOBBHTDIonTheFly etc. override it to their package's own prefix.
    _BACKEND_PREFIX = "lisatools"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)

    @staticmethod
    def GPU_RECOMMENDED_WITH_JAX() -> list[str]:
        """Same as GPU_RECOMMENDED() but with the JAX backend appended.

        The JAX backend is a host-side pure-Python path (no compiled
        wheel needed); we list it after the GPU options so the default
        "first available" pick stays GPU when both are present.

        Returns bare platform tags (``cuda13x`` / ``cpu`` / ...); the
        consuming :meth:`supported_backends` prepends the
        ``lisatools_`` namespace to match the names registered in
        ``Globals().backends_manager``.
        """
        return ["cuda13x", "cuda12x", "cuda11x", "cpu", "jax"]


# Convenience alias matching the post-Phase-3L.7k naming. Existing
# consumers keep using `FastLISAResponseParallelModule`; new code can
# use the more descriptive form.
LISAToolsResponseParallelModule = FastLISAResponseParallelModule
