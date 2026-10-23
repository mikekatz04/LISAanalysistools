Legacy and Unused Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules are still in the package but are not used by the stock global fits or the
9-month / 12-month production runs. They are listed here, not rendered as API pages, because
their members are dead, broken or superseded. Each module docstring gives the details. The
legacy global-fit moves (``mbhspecialmove``, ``multigpumove``, ``gb_likelihood``) are documented
at the end of :doc:`globalfitmoves`.

.. list-table::
    :header-rows: 1
    :widths: 32 42 26

    * - Module
      - Status
      - Use instead
    * - ``lisatools.globalfit.plot``
      - Plots of the retired pre-stock ``current_info`` layout; nothing calls them and they do
        not run against the current API. Still imported at load time by
        :mod:`lisatools.globalfit.hdfbackend`, which pulls in ``corner`` and ``matplotlib``.
      - :mod:`lisatools.globalfit.diagnosticplot`; eryn's ``PlotContainer`` in the run
    * - ``lisatools.globalfit.mbhsearch``
      - Standalone Sangria-era MBH search. Its only importer, ``scripts/mbh/dev_mbh_gf_search.py``,
        fails at import. Importing the module seeds the global NumPy RNG and raises
        ``ValueError`` without CuPy.
      - the catalogue / warm-start seeding of the stock MBH branch
    * - ``lisatools.globalfit.generatefuncs``
      - Frequency-domain A/E template helpers of the settings-file era; no importer under
        ``src/``; several bodies reference undefined names.
      - ``Settings.signal_gen`` with :meth:`lisatools.globalfit.run.GlobalFit.setup_acs`
    * - ``lisatools.globalfit.buildcatalog``
      - Per-source catalog builders for the pre-stock layout; never imported.
      - :class:`lisatools.globalfit.postprocessing.SubmissionWriter`
    * - ``lisatools.globalfit.priors.base``
      - Work-in-progress 1-D prior base; not wired into anything.
      - ``eryn`` priors
    * - ``lisatools.globalfit.priors.gfpriors``
      - Per-source prior + transform wrapper; never instantiated.
      - the stock ``*Setup`` priors
    * - ``lisatools.sampling.likelihood``
      - ``Likelihood`` / ``GlobalLikelihood`` of the ``use_gpu`` era; no callers.
      - :meth:`lisatools.analysiscontainer.AnalysisContainer.eryn_likelihood_function`
    * - ``lisatools.sampling.utility``
      - ``HeterodynedUpdate`` and a PSD-transform stub; no live callers.
      - --
    * - ``lisatools.sampling.warmstart_proposal``
      - Back-compat shim (re-exports).
      - :mod:`lisatools.globalfit.warmstart.proposal`
    * - ``lisatools.utils.jaxbase``
      - Deprecation shim (warns on import).
      - :mod:`lisatools.jax.jaxbase`
    * - ``lisatools.mojito_detector``
      - Module body commented out; importing it still needs ``jax`` and ``cudakima``.
      - :class:`lisatools.detector.L1Orbits`
    * - ``lisatools.cutils.psd_likelihood_utils``
      - CuPy / Numba equal-arm PSD likelihood; imported nowhere.
      - :mod:`lisatools.sensitivity` and the backend kernels
    * - ``lisatools.sources.emri.chebyshevwave``
      - Development Chebyshev EMRI model; only two ``scripts/emri`` scripts use it.
      - :mod:`lisatools.sources.emri.wdm_direct`

Private modules (not part of the public API): ``lisatools._unequal_arm_expressions`` (the
generated unequal-arm TDI 2.0 covariance expressions) and ``lisatools._unequal_arm_fused`` (the
fused unit covariances), both used by the unequal-arm classes of :doc:`sensitivity`;
``lisatools._version`` and ``lisatools._editable`` (packaging markers).

``lisatools.utils.exceptions`` also defines file-manager and configuration exception classes
copied from ``few.utils.exceptions``; nothing in ``lisatools`` raises them, so :doc:`utility`
lists only the live ones.
