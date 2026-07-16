Global Fit Moves
~~~~~~~~~~~~~~~~~~~~

A global-fit module is **any object with**
``propose(model, state) -> (new_state, accepted)``. It receives the model
(whose ``analysis_container_arr`` holds the per-walker residuals) and the
sampler state, may edit the residuals or state in place so the next module sees
the update, and returns the (possibly changed) state plus an ``accepted``
mask. It need not be an MCMC proposal at all — a diagnostics dumper, a residual
surgeon, or a bridge to an external code are all legal modules.

Base and combination
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.globalfitmove
    :members:
    :show-inheritance:

Add / remove — the single-source PE choreography
++++++++++++++++++++++++++++++++++++++++++++++++++++

Remove the source from the residual → sample it → put it back. Shared by MBH,
EMRI and SOBBH. This is a **fixed-leaf** move: a leaf must be alive uniformly
across walkers and temperatures, and it reads a per-leaf ``betas_all`` ladder
off the branch's State.

.. automodule:: lisatools.globalfit.moves.addremovemove
    :members:
    :show-inheritance:

Galactic binaries
+++++++++++++++++++++++

The GB reversible-jump machinery: the band tree, the special stretch move, and
the fast per-band likelihood.

.. automodule:: lisatools.globalfit.moves.gbspecialstretch
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.gbbands
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.gb_likelihood
    :members:
    :show-inheritance:

Massive black holes
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.mbhspecialmove
    :members:
    :show-inheritance:

Noise / PSD
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.psdmove
    :members:
    :show-inheritance:

Multi-GPU support
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.multigpumove
    :members:
    :show-inheritance:
