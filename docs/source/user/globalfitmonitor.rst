Global Fit Monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~

The monitor turns a run directory (the ``*testing*.h5`` store, the logs, the run settings) into
a self-contained HTML status page and a snapshot tarball, both written beside the run folder.
In the 9-month and 12-month jobs a dedicated monitor rank (``GF_MONITOR_RANK=1``) builds them
every ``GF_MONITOR_ITER`` saves in a memory-guarded child process; offline, two commands do the
same:

.. code-block:: bash

    # a live run directory -> page (and, with --snapshot, the tar)
    python -m lisatools.globalfit.monitor RUN_DIR --snapshot

    # a downloaded snapshot tar -> page
    python -m lisatools.globalfit.monitor.from_tar SNAPSHOT.tar.gz

The page generator itself (``_generator.py``) is a script that is executed, never imported, so
it has no API entry here; the other underscore modules (``_short``, ``_logs``, ``_add_info``,
``_store_extract``, ``_identity``, ``_memprobe``, ``_png``) are private helpers of the package. See
``docs/global-fit-deep-dives.md`` for how a run is examined.

Page builders (``lisatools.globalfit.monitor``)
------------------------------------------------

.. automodule:: lisatools.globalfit.monitor
    :members:
    :show-inheritance:

.. autofunction:: lisatools.globalfit.monitor.build_monitor_in_process

.. autofunction:: lisatools.globalfit.monitor.describe_run

.. autofunction:: lisatools.globalfit.monitor.check_truth

.. autofunction:: lisatools.globalfit.monitor.build_truth_set

Command-line entry points
--------------------------

.. automodule:: lisatools.globalfit.monitor.__main__
    :members:

.. automodule:: lisatools.globalfit.monitor.from_tar
    :members:

In-run hooks
--------------

.. automodule:: lisatools.globalfit.monitor.rank
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.monitor.buildguard
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.monitor.hooks
    :members:
    :show-inheritance:

Snapshot and truth set
------------------------

.. automodule:: lisatools.globalfit.monitor.snapshot
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.monitor.build_truth
    :members:
    :show-inheritance:
