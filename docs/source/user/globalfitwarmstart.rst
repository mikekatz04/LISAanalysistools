Global Fit Warm Starts
~~~~~~~~~~~~~~~~~~~~~~~~~~

Warm starts of a child run from its parent run's store: the GB RJ birth mixture (fit, referee,
apply, and the run's own self refit), the MBH / EMRI / SOBBH source leaves, and the noise pin.
Several modules are also command-line tools (``python -m lisatools.globalfit.warmstart.<module>
--help``).

.. automodule:: lisatools.globalfit.warmstart

GB birth mixture
+++++++++++++++++

.. automodule:: lisatools.globalfit.warmstart.proposal
    :members:
    :show-inheritance:
    :exclude-members: attach_transform

.. automodule:: lisatools.globalfit.warmstart.build
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.warmstart.basis
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.warmstart.fit_from_store
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.warmstart.match_referee
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.warmstart.referee_apply
    :members:
    :show-inheritance:

Self refit
+++++++++++

The run refits its warm-start mixture from its own store (``GB_WARM_SELF=1`` in the
production launchers).

.. automodule:: lisatools.globalfit.warmstart.self_refit
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: unregister_head_builder

Source leaves and noise pin
++++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.warmstart.sources
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.warmstart.noise_pin
    :members:
    :show-inheritance:
    :ignore-module-all:

.. automodule:: lisatools.globalfit.warmstart.opt_snr
    :members:
    :show-inheritance:
    :exclude-members: boxed_means, component_opt_snr, gate, main

GB parallel residual search
++++++++++++++++++++++++++++

The ``rj_para_search_pe`` birth move and its CLI (``GB_PARA_SEARCH``; a documented option, off
in production; see ``docs/gb-para-search.md``).

.. automodule:: lisatools.globalfit.warmstart.para_search
    :members:
    :show-inheritance:
    :ignore-module-all:
