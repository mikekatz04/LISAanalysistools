GW Source Tools
~~~~~~~~~~~~~~~~~~~~

.. automodule:: lisatools.sources

Waveform Base Classes
+++++++++++++++++++++++++

Base objects for the waveform tools in ``lisatools.sources``: the time-domain generators with
the LISA response step (:class:`~lisatools.sources.waveformbase.TDPyResponseWaveformBase`,
:class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase`) and the simple
:class:`~lisatools.sources.waveformbase.SNRWaveform` / :class:`~lisatools.sources.waveformbase.AETTDIWaveform`
contracts.

.. automodule:: lisatools.sources.waveformbase
    :members:
    :show-inheritance:

Batched signal generators
+++++++++++++++++++++++++++

.. automodule:: lisatools.sources.batching
    :members:
    :show-inheritance:

MBHs (``lisatools.sources.bbh``)
++++++++++++++++++++++++++++++++++

``MBH_LIKELIHOOD=batched`` (the 9-month and 12-month runs) scores MBH proposals with the
windowed grid-aligned generator; :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform`
builds the residual templates.

.. automodule:: lisatools.sources.bbh.waveform
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.bbh.gridaligned
    :members:
    :show-inheritance:

EMRIs (``lisatools.sources.emri``)
++++++++++++++++++++++++++++++++++++

``EMRI_LIKELIHOOD=direct`` (production) builds EMRI templates directly in the WDM domain
(:class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM` behind
:class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`); the legacy FEW +
``ResponseWrapper`` path (:mod:`lisatools.sources.emri.response`) is the cross-check and
fallback. These modules import FEW (``few``).

.. automodule:: lisatools.sources.emri.wdm_direct
    :members:
    :show-inheritance:
    :exclude-members: plunge_chunk_wdm

.. automodule:: lisatools.sources.emri.direct_signal_gen
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.emri.emritdionfly
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.emri.response
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.emri.waveform
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.emri.domain
    :members:
    :show-inheritance:

SOBBHs (``lisatools.sources.sobbh``)
++++++++++++++++++++++++++++++++++++++

``SOBBH_LIKELIHOOD=lookup`` (production) scores and fills SOBBH templates with the WDM lookup
scorer (:class:`~lisatools.sources.sobbh.wdm_direct.SOBBHLookupComputations` behind
:class:`~lisatools.sources.sobbh.wdm_direct.SOBBHLookupRouter`), which calls the free PN
functions of :mod:`lisatools.sources.sobbh.waveform`.

.. automodule:: lisatools.sources.sobbh.wdm_direct
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.sobbh.waveform
    :members:
    :show-inheritance:

.. automodule:: lisatools.sources.sobbh.response
    :members:
    :show-inheritance:

GBs (``lisatools.sources.gb``)
++++++++++++++++++++++++++++++++

Stand-alone GB waveforms on :class:`gbgpu.gbgpu.GBGPU`. The global fit's GB likelihood runs in
GBGPU itself (signal-heterodyne / WDM lookup engines), not through these classes.
:class:`~lisatools.sources.gb.waveform.GBAETWaveform` is the legacy AET wrapper.

.. automodule:: lisatools.sources.gb.waveform
    :members:
    :show-inheritance:

Sky frames and BBH helpers (``lisatools.sources.utils``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ICRS / ecliptic conversions (fixed obliquity ``EPS_RAD``) used on the global-fit run path,
and the BBH parameter helpers.

.. automodule:: lisatools.sources.utils
    :members:
    :show-inheritance:
    :exclude-members: CalculationController, BBHCalculationController, GBCalculationController, EMRICalculationController, evolve_galactic_binary, psi_rotation_icrs_to_ecliptic, WaveWrapCache

Calculation Controllers (not functional)
++++++++++++++++++++++++++++++++++++++++++

Simplified wrappers for quick SNR and covariance studies, re-exported by
:mod:`lisatools.sources`.

.. warning::

    These controllers predate the domain-based :func:`~lisatools.diagnostic.inner_product`
    API. They pass ``dt=`` / ``f_arr=`` / ``df=`` through :func:`~lisatools.diagnostic.snr`
    and :func:`~lisatools.diagnostic.covariance`, which the current ``inner_product``
    signature does not accept, so ``get_snr`` / ``get_cov`` raise ``TypeError``. Use
    :class:`~lisatools.analysiscontainer.AnalysisContainer` (``snr``,
    ``template_inner_product``) for SNR work.

.. autoclass:: lisatools.sources.utils.CalculationController
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sources.utils.BBHCalculationController
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sources.utils.GBCalculationController
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sources.utils.EMRICalculationController
    :members:
    :show-inheritance:


Default Response Settings for GW tools
++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: lisatools.sources.defaultresponse.DefaultResponseKwargs
    :members:
    :show-inheritance: