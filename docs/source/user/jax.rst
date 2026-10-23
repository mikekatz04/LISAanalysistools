JAX Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~

The pure-JAX backend (``lisatools_jax``, registered when ``import jax`` succeeds) and the JAX
LISA-response and WDM implementations that ``gbgpu.jax`` and ``bbhx.jax`` build on. Per the
backend implementation hierarchy (``docs/conventions.md``) the JAX path may differ internally
from the C++/CUDA one but must match it at the inner-product level. The 9-month and 12-month
runs compute on ``cuda13x``; from this package they use only
:class:`~lisatools.jax.jaxbase.JaxBase`, the mixin of the phentax MBH generator.

Most of these modules import ``jax`` at module level, so this page renders fully only when the
documentation is built in an environment with ``jax`` installed.

.. automodule:: lisatools.jax

Backend and orbits
--------------------

.. automodule:: lisatools.jax.backend
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.orbits
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.jaxbase
    :members:
    :show-inheritance:

Response (``lisatools.jax.response``)
--------------------------------------

.. automodule:: lisatools.jax.response.base
    :members:
    :show-inheritance:
    :exclude-members: n_params, param_names, inc_index

.. automodule:: lisatools.jax.response.projection
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.response.tdi_config
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.response.amp_phase_extract
    :members:
    :show-inheritance:

WDM (``lisatools.jax.wdm``)
-----------------------------

.. automodule:: lisatools.jax.wdm.wdm_settings
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.wdm.wdm_domain
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.wdm.wavelet_lookup
    :members:
    :show-inheritance:

.. automodule:: lisatools.jax.wdm.fast_inner
    :members:
    :show-inheritance:
