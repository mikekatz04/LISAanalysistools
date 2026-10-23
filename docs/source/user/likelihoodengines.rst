Heterodyne and Lookup Likelihood Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fast WDM-domain likelihoods of the global fit do not transform a dense template for every
proposal. The source-agnostic pieces live here; the source classes (GBGPU for GB, BBHx for
SOBBH, :mod:`lisatools.sources` for EMRI and SOBBH lookup) build on them:

* chunked heterodyne (:mod:`lisatools.chunked_het`, :mod:`lisatools.wdm_het`): the template
  is built chunk by chunk on overlapping ``Nt_sub``-pixel windows and scored against the data
  layer group by layer group;
* signal heterodyne (:mod:`lisatools.signal_het`): a reference template plus the proposed /
  reference ratio on a sparse time grid; this backs the GB in-model scorer in the 9-month and
  12-month runs;
* WDM lookup tables (:mod:`lisatools.wdm_lookup_eval`, :mod:`lisatools.wdm_lookup_store`,
  :class:`lisatools.domains.WDMLookupTable`): the WDM response of a unit linear chirp,
  tabulated once per grid and interpolated per pixel (EMRI direct-to-WDM, SOBBH lookup and the
  GB lookup engine);
* the information matrix from second differences of the likelihood
  (:mod:`lisatools.info_matrix_ll`), used for the eigen proposal tables.

Chunked heterodyne (``lisatools.chunked_het``)
-----------------------------------------------

.. automodule:: lisatools.chunked_het
    :members:
    :show-inheritance:
    :exclude-members: get_swap_ll_grad_wdm

Chunk geometry and windows (``lisatools.wdm_het``)
---------------------------------------------------

.. automodule:: lisatools.wdm_het
    :members:
    :show-inheritance:
    :ignore-module-all:

Signal heterodyne (``lisatools.signal_het``)
---------------------------------------------

.. automodule:: lisatools.signal_het
    :members:
    :show-inheritance:

WDM lookup evaluation (``lisatools.wdm_lookup_eval``)
------------------------------------------------------

.. automodule:: lisatools.wdm_lookup_eval
    :members:
    :show-inheritance:

WDM lookup table store (``lisatools.wdm_lookup_store``)
--------------------------------------------------------

.. automodule:: lisatools.wdm_lookup_store
    :members:
    :show-inheritance:
    :ignore-module-all:

Information matrix from the likelihood (``lisatools.info_matrix_ll``)
----------------------------------------------------------------------

.. automodule:: lisatools.info_matrix_ll
    :members:
    :show-inheritance:
