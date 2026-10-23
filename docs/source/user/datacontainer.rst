Data / Residual / Signal Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~lisatools.datacontainer.DataResidualArray` is the data / residual array wrapper of the
frequency-domain workflow. :class:`~lisatools.analysiscontainer.AnalysisContainer` accepts one
but stores its data as a :class:`~lisatools.domains.DomainBase` signal (:doc:`domains`); the
WDM global fit passes domain signals directly.

.. autoclass:: lisatools.datacontainer.DataResidualArray
    :members:
    :show-inheritance:

Analysis Container
~~~~~~~~~~~~~~~~~~~~~~~~~

The central likelihood object (data + sensitivity matrix + optional signal generator), and its
many-walker, multi-GPU array form that every global-fit move scores against.

.. automodule:: lisatools.analysiscontainer
    :members:
    :show-inheritance:
