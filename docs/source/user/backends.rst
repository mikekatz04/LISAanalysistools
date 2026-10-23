Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~

Importing :mod:`lisatools` registers the compute backends with the ``gpubackendtools``
registry: ``lisatools_cpu``, ``lisatools_cuda11x``, ``lisatools_cuda12x``,
``lisatools_cuda13x`` and, when ``import jax`` succeeds, ``lisatools_jax``
(:doc:`jax`). A compute class picks its backend once, at construction, with
``force_backend=`` (``"cpu"``, ``"cuda12x"``, ``"cuda13x"``, ``"jax"``, ...); methods never
take a backend argument. The 9-month and 12-month global-fit runs use ``cuda13x``.

Name rules (see :func:`lisatools.get_backend` and :func:`lisatools.has_backend`):

* ``get_backend`` accepts ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"jax"``,
  and the aliases ``"cuda"`` / ``"gpu"`` (the first loadable CUDA backend).
* ``has_backend`` accepts the same concrete names but **not** ``"cuda"`` or ``"gpu"``: those
  raise ``ValueError``.
* :func:`lisatools.get_first_backend` does not work as written (see its Warning).

.. code-block:: python

    import lisatools

    for name in ("cpu", "cuda11x", "cuda12x", "cuda13x"):
        print(name, lisatools.has_backend(name))

    be = lisatools.get_backend("cpu")
    be.xp            # numpy (cupy for a CUDA backend)

Backend lookup
--------------------------

.. autofunction:: lisatools.get_backend

.. autofunction:: lisatools.has_backend

.. autofunction:: lisatools.get_first_backend

Headers for downstream builds
------------------------------

.. autofunction:: lisatools.get_include

.. autofunction:: lisatools.get_cmake_module_path

Backend classes (``lisatools.cutils``)
---------------------------------------

.. automodule:: lisatools.cutils
    :members:
    :show-inheritance:

Parallel-module base
--------------------------

Every compute class derives from this base, which resolves ``force_backend`` and exposes
``self.backend``.

.. automodule:: lisatools.utils.parallelbase
    :members:
    :show-inheritance:
