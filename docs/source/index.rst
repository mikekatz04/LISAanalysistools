.. include:: readme.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 4
   :caption: Documentation:

   user/sensitivity
   user/detector
   user/response
   user/stochastic
   user/datacontainer
   user/diagnostic
   user/sources
   user/utility

.. toctree::
   :maxdepth: 2
   :caption: Workshop tutorials:

   latw/00_SetupAndAtlas
   latw/01_GlobalFitQuickstart
   latw/02_StockGlobalFitsInDepth
   latw/03_StockGlobalFitGallery
   latw/04_Foundations
   latw/05_ResponseAndTDI
   latw/06_SourceWaveforms
   latw/07_ErynSmallToLarge
   latw/08_BackendsAndDevWorkflow

Workshop exercises
==================

The informational tutorials above are rendered here from the
`LISA Analysis Tools Workshop (LATW) <https://github.com/lisa-analysis-tools/LATW>`_
at its pinned ``dev`` commit. The workshop also ships a set of hands-on
**exercises** that are *not* reproduced in these docs -- work through them in
the LATW repository itself:

* Student notebooks: `tutorials/further/ <https://github.com/lisa-analysis-tools/LATW/tree/dev/tutorials/further>`_
* Worked answers: `tutorials/further/answers/ <https://github.com/lisa-analysis-tools/LATW/tree/dev/tutorials/further/answers>`_

.. toctree::
   :maxdepth: 1
   :caption: Developer guides:

   devguides/conventions
   devguides/architecture-map
   devguides/codebase-map
   devguides/global-fit-launch
   devguides/stock-stages-and-moves
   devguides/multigpu-cluster-validation
