"""Eigen-axis inner move with a larger-jump share (user ruling 2026-10-13).

Why this exists: on the 9mo v9 production run (2026-10-13) the cold walkers of
the loud EMRIs sat on separate islands, secondary maxima 23-38 nats below the
best walker, with no bridge between them. The eryn
:class:`~eryn.moves.EigenAxisMove` steps along one eigen axis by a 1-sigma
Gaussian width, which cannot cross such a gap. This subclass gives a fixed
fraction ``jump_mix`` of the proposals a step ``jump_mix_factor`` times wider
(e.g. 10 % at 3 sigma); the rest are the eryn draw unchanged.

Detailed balance: each row picks its scale with a probability that does not
depend on the current point, so the proposal is a fixed mixture of two
symmetric kernels along the same frozen axis table -- symmetric itself, and
the Metropolis factor stays exactly zero (as for the parent move).

``jump_mix == 0`` delegates to the eryn method untouched: the same draws, in
the same order, from the same random stream (bit-identical). With
``jump_mix > 0`` the base step is drawn exactly as eryn draws it (same
``draw_axis_step`` / joint-draw calls, same stream order), then ONE uniform
per row picks the rows that take the wider step.

Knobs (per branch, the attribute rule): ``{BRANCH}_EIGEN_JUMP_MIX`` (default
0 = off) and ``{BRANCH}_EIGEN_JUMP_FACTOR`` (default 3), the settings fields
``eigen_jump_mix`` / ``eigen_jump_factor`` of the MBH / EMRI / SOBBH branch
blocks; built by ``stock.erebor.common.make_default_inner_moves``.
"""

from __future__ import annotations

import numpy as np
from eryn.moves import EigenAxisMove
from eryn.moves.eigenaxis import _array_module, draw_axis_step

__all__ = ["EigenAxisJumpMixMove"]


class EigenAxisJumpMixMove(EigenAxisMove):
    """:class:`~eryn.moves.EigenAxisMove` with a ``jump_mix`` share of wider steps.

    Args:
        *args: Forwarded to :class:`~eryn.moves.EigenAxisMove`.
        jump_mix (float): Probability, per proposed row, of the wider step;
            in ``[0, 1]``. ``0`` is the parent move exactly.
        jump_mix_factor (float): Step multiplier of the wider draws (on top
            of the parent's own ``jump_factor``); ``> 0``.
        **kwargs: Forwarded to :class:`~eryn.moves.EigenAxisMove`
            (``mode``, ``jump_factor``, ``periodic``, ...).
    """

    def __init__(self, *args, jump_mix=0.0, jump_mix_factor=3.0, **kwargs):
        jump_mix = float(jump_mix)
        jump_mix_factor = float(jump_mix_factor)
        if not (0.0 <= jump_mix <= 1.0):
            raise ValueError(f"jump_mix must be in [0, 1], got {jump_mix}.")
        if not jump_mix_factor > 0.0:
            raise ValueError(f"jump_mix_factor must be > 0, got {jump_mix_factor}.")
        super().__init__(*args, **kwargs)
        self.jump_mix = jump_mix
        self.jump_mix_factor = jump_mix_factor

    def _draw_jump_scale(self, n, random):
        """Per-row step multiplier: ``jump_mix_factor`` w.p. ``jump_mix``, else 1.

        One host uniform per row, drawn AFTER the base step (so the base step
        consumes the stream exactly as the parent does).
        """
        if hasattr(random, "integers"):          # numpy Generator
            u = np.asarray(random.random(n))
        else:                                    # legacy RandomState
            u = np.asarray(random.rand(n))
        return np.where(u < self.jump_mix, self.jump_mix_factor, 1.0)

    def get_proposal(self, branches_coords, random, branches_inds=None,
                     supps=None, branch_supps=None, **kwargs):
        """The parent's proposal; a ``jump_mix`` share of rows steps wider.

        See :meth:`eryn.moves.EigenAxisMove.get_proposal` for the arguments
        and the return value. The base step below mirrors that method line
        for line (``tests/test_eigen_jump_mix.py`` pins the parity).
        """
        if self.jump_mix <= 0.0:
            return super().get_proposal(
                branches_coords, random, branches_inds=branches_inds,
                supps=supps, branch_supps=branch_supps, **kwargs,
            )

        q = {}
        self.last_jump_scale = {}
        for name, coords in branches_coords.items():
            xp = _array_module(coords)
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            inds_here = _array_module(inds).where(inds == True)  # noqa: E712
            nrows = len(inds_here[0])

            axes_rows, sigmas_rows = self._rows_for(
                name, inds_here, nrows, ndim, branch_supps, xp
            )

            if self.mode == "axis":
                dy, _ = draw_axis_step(
                    axes_rows, sigmas_rows, random,
                    jump_factor=self.jump_factor,
                )
            else:
                if hasattr(random, "standard_normal"):
                    z = np.asarray(random.standard_normal((nrows, ndim)))
                else:
                    z = np.asarray(random.randn(nrows, ndim))
                z_x = z if xp is np else xp.asarray(z)
                dy = self.jump_factor * xp.einsum(
                    "nik,nk->ni", axes_rows * sigmas_rows[:, None, :], z_x
                )

            scale = self._draw_jump_scale(nrows, random)
            self.last_jump_scale[name] = scale
            scale_x = scale if xp is np else xp.asarray(scale)
            dy = dy * scale_x[:, None]

            new_coords = coords.copy()
            new_coords[inds_here] = coords[inds_here] + dy
            q[name] = new_coords

        # handle periodic parameters (as the parent does)
        if self.periodic is not None:
            wrapped = self.periodic.wrap(
                {
                    name: tmp.reshape((-1,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )
            q = {
                name: wrapped[name].reshape(q[name].shape)
                for name in q
            }

        return q, np.zeros((ntemps, nwalkers))

    def describe(self) -> str:
        """``"jump mix 0.10 x 3.0"`` (the inner-move stack log suffix)."""
        if self.jump_mix <= 0.0:
            return "jump mix off"
        return f"jump mix {self.jump_mix:.2f} x {self.jump_mix_factor:.1f}"
