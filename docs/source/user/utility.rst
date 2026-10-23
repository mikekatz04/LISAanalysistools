Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

Array helpers (``lisatools.utils.utility``)
--------------------------------------------

.. automodule:: lisatools.utils.utility
    :members:
    :show-inheritance:

Device pinning (``lisatools.utils.device``)
--------------------------------------------

.. automodule:: lisatools.utils.device
    :members:
    :show-inheritance:

Constants (``lisatools.utils.constants``)
------------------------------------------

.. automodule:: lisatools.utils.constants

The C/C++/CUDA side reads the same values from the generated ``lisaconstants_values.h``:

.. automodule:: lisatools.utils.lisaconstants_header
    :members:
    :show-inheritance:

Exceptions (``lisatools.utils.exceptions``)
--------------------------------------------

.. automodule:: lisatools.utils.exceptions

.. autoexception:: lisatools.utils.exceptions.LISAToolsException
    :show-inheritance:

.. autoexception:: lisatools.utils.exceptions.WaveformDomainError
    :show-inheritance:

.. autoexception:: lisatools.utils.exceptions.BatchNotLaunchable
    :show-inheritance:

Stage timing (``lisatools.utils.stagetimer``)
----------------------------------------------

.. automodule:: lisatools.utils.stagetimer
    :members:
    :show-inheritance:

Type aliases (``lisatools.utils.typing``)
------------------------------------------

.. automodule:: lisatools.utils.typing

Citations
-------------

The ``lisatools_citations`` command prints the BibTeX references of LISA Analysis Tools
modules:

.. code-block:: bash

    lisatools_citations                      # every reference
    lisatools_citations lisatools.detector   # the references of one module

.. automodule:: lisatools.cmd.citations
    :members:
    :show-inheritance:

.. automodule:: lisatools.utils.citations
    :members:
    :show-inheritance:


