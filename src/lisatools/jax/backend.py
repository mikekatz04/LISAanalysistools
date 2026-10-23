"""``LISAToolsJaxBackend`` -- the JAX entry into lisatools' backend registry.

Slots into :class:`lisatools.cutils.LISAToolsBackendMethods` so that
:class:`lisatools.detector.Orbits` (and any other class subclassing
:class:`gpubackendtools.ParallelModuleBase`) can pick the JAX path via
``force_backend='jax'``.

There is no compiled plugin module for this backend; the
implementations live in :mod:`lisatools.jax` as pure JAX code. We
subclass :class:`Backend` directly (not :class:`CpuBackend`) so the
``_check_module_installed`` importlib check is skipped; ``import jax``
is what we verify instead.

The methods filled in are ``OrbitsWrap`` / ``Orbits``
(:class:`~lisatools.jax.orbits.OrbitsWrapJAX`), ``check_orbits`` (a no-op
stub) and ``xp`` (``jax.numpy``); every other
:class:`~lisatools.cutils.LISAToolsBackendMethods` slot keeps its default.
:func:`_unimplemented` builds stand-in classes that raise
``NotImplementedError`` on construction; it is not used at present (the
``SensitivityMatrixWrap`` entry is commented out). Constructing the backend
calls :func:`_check_jax`, which turns on ``jax_enable_x64``.
"""
from __future__ import annotations

from gpubackendtools.exceptions import MissingDependency
from gpubackendtools.gpubackendtools import Backend

from ..cutils import LISAToolsBackend, LISAToolsBackendMethods


def _check_jax():
    """Import jax, enable 64-bit mode (``jax_enable_x64``) and return ``jax.numpy``.

    Raises:
        gpubackendtools.exceptions.MissingDependency: jax is not installed.
    """
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as e:
        raise MissingDependency(
            "'jax' backend requires jax (with jaxlib)",
            pip_deps=["jax", "jaxlib"],
            conda_deps=["jax"],
        ) from e
    from jax import config
    config.update("jax_enable_x64", True)
    return jnp


# TODO check: keep/remove/document -- no callers (its use in _jax_methods_loader is commented out)
def _unimplemented(name):
    """Return a stand-in class named ``name`` whose constructor raises ``NotImplementedError``."""
    class _NotImpl:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"{name} is not implemented in the lisatools JAX backend yet. "
                "Use force_backend='cpu' (or one of the CUDA backends) for "
                "this path until a JAX implementation lands."
            )
    _NotImpl.__name__ = name
    return _NotImpl


def _check_orbits_jax(*args, **kwargs):
    """Stub for the C++ ``check_orbits`` validation routine.

    The C++ routine validates that arm lengths, light-travel times and
    positions are self-consistent. For the JAX path we silently no-op
    -- the orbit data was already validated when the host-side
    ``Orbits.configure(linear_interp_setup=True)`` ran on numpy / cupy
    inputs upstream.
    """
    return None


def _jax_methods_loader() -> LISAToolsBackendMethods:
    """Build the JAX backend's method table (``OrbitsWrap``, ``Orbits``, ``check_orbits``, ``xp``).

    Returns:
        LISAToolsBackendMethods: The table; slots not listed keep their defaults.
    """
    jnp = _check_jax()
    from .orbits import OrbitsWrapJAX

    return LISAToolsBackendMethods(
        OrbitsWrap=OrbitsWrapJAX,
        Orbits=OrbitsWrapJAX,                          # JAX side: same class
        check_orbits=_check_orbits_jax,
        # SensitivityMatrixWrap=_unimplemented("SensitivityMatrixWrap"),  # XYZBackend disabled
        xp=jnp,
    )


class LISAToolsJaxBackend(Backend, LISAToolsBackend):
    """JAX backend object plugged into ``Globals().backends_manager``.

    Registered as ``"lisatools_jax"`` by ``lisatools/__init__`` when the
    :mod:`lisatools.jax` import succeeds; resolve it with
    ``lisatools.get_backend("jax")`` or pick it per object with
    ``force_backend="jax"``. ``xp`` is ``jax.numpy`` and ``OrbitsWrap`` is
    :class:`~lisatools.jax.orbits.OrbitsWrapJAX`. There is no compiled module,
    so :meth:`_check_module_installed` is a no-op.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Raises:
        gpubackendtools.exceptions.MissingDependency: jax is not installed.

    Example:
        .. code-block:: python

            import lisatools
            backend = lisatools.get_backend("jax")
            backend.name   # "lisatools_jax"
    """

    _backend_name: str = "lisatools_backend_jax"
    _name = "lisatools_jax"

    def __init__(self, *args, **kwargs):
        methods = _jax_methods_loader()
        Backend.__init__(
            self,
            name=self._name,
            methods=methods,
            features=Backend.Feature.NUMPY,
        )
        LISAToolsBackend.__init__(self, methods)

    @staticmethod
    def _check_module_installed(*args, **kwargs):
        """No-op: there is no compiled module to check (jax is checked at construction)."""
        return None
