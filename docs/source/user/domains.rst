Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every data, residual and template array travels as a signal object that knows its basis:
time domain (:class:`~lisatools.domains.TDSettings` / :class:`~lisatools.domains.TDSignal`),
frequency domain (:class:`~lisatools.domains.FDSettings` / :class:`~lisatools.domains.FDSignal`),
short-time FFT (:class:`~lisatools.domains.STFTSettings` / :class:`~lisatools.domains.STFTSignal`)
and the WDM wavelet grid (:class:`~lisatools.domains.WDMSettings` /
:class:`~lisatools.domains.WDMSignal`). Domains are never passed by string: code dispatches
on the settings class, and a run configuration takes a settings instance or a
``(times, dt, force_backend)`` factory (``FDSettings.make_factory`` /
``STFTSettings.make_factory`` / ``WDMSettings.make_factory``). The 9-month and 12-month
global-fit runs use the WDM domain.

.. code-block:: python

    from lisatools.domains import FDSettings, TDSettings

    td = TDSettings(N=4096, dt=5.0, t0=0.0)
    fd = FDSettings(N=2049, df=1.0 / (4096 * 5.0))

Settings and signals (``lisatools.domains``)
---------------------------------------------

.. automodule:: lisatools.domains
    :members:
    :show-inheritance:
    :exclude-members: get_fd_window_for_wdm, wavelet

Per-domain computation groups (``lisatools.domaincomputation``)
----------------------------------------------------------------

Batched ``(d|h)`` / ``(h|h)`` likelihood groups over the C++ domain wraps, one per GPU split of
an :class:`~lisatools.analysiscontainer.AnalysisContainerArray`. The global fit builds them only
on multi-GPU runs whose sensitivity matrix carries orbits, so the stock runs'
:class:`~lisatools.sensitivity.CompositeSensitivityBackend` does not use them. Importing this
module requires ``jax``.

.. automodule:: lisatools.domaincomputation
    :members:
    :show-inheritance:

Coarse WDM noise surrogate (``lisatools.coarsewdm``)
-----------------------------------------------------

A time-coarsened WDM statistic for fast noise-likelihood evaluation. It is off in the
9-month and 12-month runs: the launchers set ``COARSE_Q=1``, and with ``Q <= 1`` the run
never builds :class:`~lisatools.coarsewdm.CoarseWDMRuntime` (the ``all_sources`` code
default is ``COARSE_Q=8``).

.. automodule:: lisatools.coarsewdm
    :members:
    :show-inheritance:
