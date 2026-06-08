LISA Response
~~~~~~~~~~~~~~~~~~~~~~~~~~

LISA arm-projection and TDI machinery. **Absorbed from**
``fastlisaresponse`` **at Phase 3B/3C/3E** (2026-06-02) as part of the
LISA sprint 2026 reorganization. Old ``fastlisaresponse.*`` imports
have been retired; use the ``lisatools.response`` paths below.

Direct Response (``lisatools.response.directresponse``)
========================================================

Fast Response Function
-----------------------

.. autoclass:: lisatools.response.directresponse.pyResponseTDI
    :members:
    :show-inheritance:
    :inherited-members:

Response Function Wrapper
--------------------------

.. autoclass:: lisatools.response.directresponse.ResponseWrapper
    :members:
    :show-inheritance:
    :inherited-members:

Coordinate Transforms
----------------------

.. autofunction:: lisatools.response.directresponse.ecliptic_to_icrs

.. autofunction:: lisatools.response.directresponse.icrs_to_ecliptic

TDI Configuration (``lisatools.response.tdiconfig``)
=====================================================

.. autoclass:: lisatools.response.tdiconfig.TDIConfig
    :members:
    :show-inheritance:
    :inherited-members:

Parallel Module Base (``lisatools.response.parallelbase``)
===========================================================

.. autoclass:: lisatools.response.parallelbase.FastLISAResponseParallelModule
    :members:
    :show-inheritance:

TDI-on-the-Fly (``lisatools.response.tdionfly``)
=================================================

Generic Base Classes
---------------------

.. autoclass:: lisatools.response.tdionfly.TDIonTheFly
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: lisatools.response.tdionfly.TDTDIonTheFly
    :members:
    :show-inheritance:

.. autoclass:: lisatools.response.tdionfly.FDTDIonTheFly
    :members:
    :show-inheritance:

TDI Output Containers
---------------------

.. autoclass:: lisatools.response.tdionfly.TDIOutput
    :members:
    :show-inheritance:

.. autoclass:: lisatools.response.tdionfly.TDTDIOutput
    :members:
    :show-inheritance:

.. autoclass:: lisatools.response.tdionfly.FDTDIOutput
    :members:
    :show-inheritance:

Source-class Subclasses
------------------------

These subclasses route to source-specific backends (``gbgpu.*`` /
``bbhx.*``). The end-user typically instantiates them from the
matching source-class frontend (``GBWDMComputations``,
``SOBBHWDMComputations``, ...) rather than directly.

.. autoclass:: lisatools.response.tdionfly.GBTDIonTheFly
    :members:
    :show-inheritance:

.. autoclass:: lisatools.response.tdionfly.SOBBHTDIonTheFly
    :members:
    :show-inheritance:

.. autoclass:: lisatools.response.tdionfly.GBFDTDIonTheFly
    :members:
    :show-inheritance:
