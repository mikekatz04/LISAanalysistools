Global Fit Internals
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run infrastructure below the recipe: the MPI rank layout and walker-block fan-out, the
mid-iteration checkpoint, the residual snapshot for the monitor, the host-memory telemetry,
the galactic-foreground readout and the GB prior components.

Rank layout and communication
++++++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.communication

.. automodule:: lisatools.globalfit.communication.ranks
    :members:
    :show-inheritance:
    :exclude-members: ranks_on_device

.. automodule:: lisatools.globalfit.communication.fanout
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.communication.walkerslice
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.communication.rowfanout
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.communication.fakecomm
    :members:
    :show-inheritance:

Mid-iteration checkpoint
++++++++++++++++++++++++++

On by code default (``MIDIT_CHECKPOINT``); both production launchers export
``MIDIT_CHECKPOINT=0``.

.. automodule:: lisatools.globalfit.midit_checkpoint
    :members:
    :show-inheritance:

Residual snapshot
++++++++++++++++++

.. automodule:: lisatools.globalfit.residual_snapshot
    :members:
    :show-inheritance:

Host-memory telemetry
++++++++++++++++++++++

.. automodule:: lisatools.globalfit.memwatch
    :members:
    :show-inheritance:
    :ignore-module-all:

Galactic-foreground readout (``lisatools.globalfit.noise_ratchet``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The galfor ratchet itself was dropped from production on 2026-10-03 (``GALFOR_RATCHET=0``;
:func:`~lisatools.globalfit.noise_ratchet.ratchet_from_env` then returns ``None``). The module
still provides the foreground-curve readout and the ``STOP_RULES`` table the GB search stages
validate against; the ratchet-only gate and schedule classes are not listed here.

.. automodule:: lisatools.globalfit.noise_ratchet
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: RatchetSchedule, NoiseRatchetGate, min_gain_from_env, stop_rule_from_env, min_drop_from_env, drop_band_from_env, galfor_band_drop, nudge_delta_from_env, is_noise_ratchet_gate, reset_maxlogl_search

Priors (``lisatools.globalfit.priors``)
++++++++++++++++++++++++++++++++++++++++

The GB galaxy sky / distance prior (``GB_USE_GALAXY_PRIOR``) and the mojito ``fdot`` helper.
The unused ``priors.base`` and ``priors.gfpriors`` modules are listed in :doc:`legacy`.

.. automodule:: lisatools.globalfit.priors.galaxy_sky_dist
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.priors.galaxy_prior_3d
    :members:
    :show-inheritance:
    :exclude-members: build_prior

.. automodule:: lisatools.globalfit.priors.gbpriors
    :members:
    :show-inheritance:
    :exclude-members: GalacticBinaryPrior, f_Hz_to_mHz
