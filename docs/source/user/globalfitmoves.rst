Global Fit Moves
~~~~~~~~~~~~~~~~~~~~

A global-fit module is **any object with**
``propose(model, state) -> (new_state, accepted)``. It receives the model
(whose ``analysis_container_arr`` holds the per-walker residuals) and the
sampler state, may edit the residuals or state in place so the next module sees
the update, and returns the (possibly changed) state plus an ``accepted``
mask. It need not be an MCMC proposal at all — a diagnostics dumper, a residual
surgeon, or a bridge to an external code are all legal modules.

.. automodule:: lisatools.globalfit.moves

Base and combination
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.globalfitmove
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.functionmove
    :members:
    :show-inheritance:

Add / remove — the single-source PE choreography
++++++++++++++++++++++++++++++++++++++++++++++++++++

Expose the source in the residual → sample it → fold it back. Shared by MBH,
EMRI and SOBBH. This is a **fixed-leaf** move: every visited leaf must be alive
on every walker and rung (asserted), and each leaf runs on its own ladder, the
``betas_all[leaf]`` row of the branch's sub-state. The production moves are the
fast-likelihood subclasses ``MBHBatchedLikeMove`` (``mbh_pe``),
``EMRIDirectLikeMove`` (``emri_pe``) and ``SOBBHChunkedLikeMove``
(``sobbh_pe``).

.. automodule:: lisatools.globalfit.moves.addremovemove
    :members:
    :show-inheritance:
    :exclude-members: get_waveform_here, replace_residuals, MultiGPUResidualAddRemoveMove

.. automodule:: lisatools.globalfit.moves.mbhbatchedmove
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.emridirectmove
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.sobbhspecialmove
    :members:
    :show-inheritance:

Eigen proposal tables
++++++++++++++++++++++

The per-leaf eigen tables of the add/remove and noise PE moves, and their sidecar persistence
(``<store>_eigen_tables.pkl``).

.. automodule:: lisatools.globalfit.moves.eigen_refresh
    :members:
    :show-inheritance:
    :exclude-members: transform_information, eigen_table_from_waveform

.. automodule:: lisatools.globalfit.moves.eigen_table_persist
    :members:
    :show-inheritance:

Galactic binaries
+++++++++++++++++++++++

The GB reversible-jump machinery: the special stretch move family (in-model, RJ prior /
warm-start, F-statistic, refit), the band buffers and scheduler, and the in-model proposal
factor cache.

.. automodule:: lisatools.globalfit.moves.gbspecialstretch
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: fstat_fit_plan, fstat_plan_is_collective, cap_diag_birth_violations, gb_fiber_tangent, gb_lnfdot_gradient, gb_shear_ridge_axis, gb_ridge_axis, eigen_axis_set, apply_shutoff_floor, observe, find_friends, fstat_fit_walkers, get_param_limits

.. automodule:: lisatools.globalfit.moves.gbbands
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: get_swap_ll, get_ll_grad, hessian, get_add_ll_exact, get_removal_ll, remove_sources_from_template_buffer, adjust_sources_in_band_buffer, unique_N, build_friend_index, build_friend_table, index_friends, draw_friends

.. automodule:: lisatools.globalfit.moves.gb_chol_cache
    :members:
    :show-inheritance:

Noise / PSD
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.moves.psdmove
    :members:
    :show-inheritance:
    :exclude-members: forced_noise_step, clip_shift_to_prior, MultiGPUPSDMove

Walker-block fan-out
+++++++++++++++++++++++

The whole-``propose`` fan-out over the compute ranks' walker blocks, mixed into the add/remove
and PSD moves (see :doc:`globalfit_internals` for the rank layout).

.. automodule:: lisatools.globalfit.moves.walkerfanout
    :members:
    :show-inheritance:

Legacy moves (not used by the stock fits)
++++++++++++++++++++++++++++++++++++++++++

No recipe builder constructs these. :mod:`~lisatools.globalfit.moves.mbhspecialmove` holds the
pre-batched MBH moves (the MBH branch runs ``MBHBatchedLikeMove`` or the base
``ResidualAddOneRemoveOneMove``), :mod:`~lisatools.globalfit.moves.multigpumove` a DCGA holder
that nothing subclasses, and :mod:`~lisatools.globalfit.moves.gb_likelihood` a deprecation
shim that re-exports ``gbgpu.gb_likelihood`` (importing it warns).

.. automodule:: lisatools.globalfit.moves.mbhspecialmove
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.multigpumove
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.moves.gb_likelihood
