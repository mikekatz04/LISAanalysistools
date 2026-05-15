"""Recipe orchestration for sequencing global-fit sampling stages."""

import logging
logger = logging.getLogger(__name__)

class Recipe:
    """Ordered sequence of :class:`RecipeStep` instances driving the sampler.

    A ``Recipe`` is iterated by the global-fit driver. At each call it asks the
    current step's stopping function whether to advance, and on advance it
    invokes the next step's ``setup_run`` to reconfigure the sampler.
    """

    def __init__(self):
        self.recipe = []
        self.backend_added = False
        self._current_iter = 0
        self._current_recipe_step = None
        self._has_setup_first_step = False

    @property
    def backend(self):
        """Backend object that records recipe-step completion."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend
        self.backend_added = True

    def add_recipe_component(self, adjust_fn, name=None):
        """Append a recipe step.

        Args:
            adjust_fn: A :class:`RecipeStep` (or compatible object) implementing
                ``setup_run`` and ``stopping_function``.
            name: Optional human-readable name. If ``None``, a default name is
                assigned based on the current recipe length.
        """
        if name is None:
            name = f"recipe step {len(self.recipe) + 1}"
        self.recipe.append({"name": name, "adjust": adjust_fn, "status": False})

    def to_file(self):
        """Return a dict mapping recipe-step names to their completion status."""
        _tmp = {recipe_step["name"]: recipe_step["status"] for recipe_step in self.recipe}
        return _tmp

    def __next__(self):
        """Advance the internal cursor past any completed steps."""
        while self._current_iter < len(self.recipe):
            # False means it is not finished
            if self.recipe[self._current_iter]["status"]:
                self._current_iter += 1

            else:
                break

        if self._current_iter < len(self.recipe):
            self._current_recipe_step = self.recipe[self._current_iter]

    def setup_first_recipe_step(self, iteration, last_sample, sampler):
        """Configure the sampler for the very first (incomplete) recipe step.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The :class:`GlobalFitEngine` (or eryn-compatible) sampler.

        Raises:
            ValueError: If the recipe has already been completed.
        """
        assert not self._has_setup_first_step
        # move to next recipe step
        next(self)
        if self._current_iter >= len(self.recipe):
            raise ValueError("Recipe is already finished.")

        self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
        self._has_setup_first_step = True

    @property
    def current_recipe_step(self):
        """The active recipe-step record (a dict of ``name``/``adjust``/``status``)."""
        return self._current_recipe_step

    def __call__(self, iteration, last_sample, sampler):
        """Evaluate the current step's stopping criterion and advance if met.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The active sampler.

        Returns:
            ``True`` if the entire recipe has finished, ``False`` otherwise.
        """
        stop_here = self._current_recipe_step["adjust"].stopping_function(
            iteration, last_sample, sampler
        )
        if stop_here:
            self.backend.completed_recipe_step(self._current_recipe_step["name"])
            self._current_recipe_step["status"] = True
            next(self)

            if self._current_iter >= len(self.recipe):
                return True
            self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)

        return False


class RecipeStep:
    """Abstract base for a single stage in a :class:`Recipe`.

    Each subclass must define a ``setup_run`` method that configures the
    sampler when the step becomes active and a ``stopping_function`` that
    decides when to advance to the next step.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``. Defaults to uniform.
    """

    def __init__(self, moves=None, weights=None):
        if moves is not None:
            self.moves = moves
            if weights is not None:
                self.weights = weights

    def __repr__(self):
        return f"RecipeStep with moves: {self.moves} and weights: {self.weights}"

    @property
    def moves(self):
        """List of MCMC moves used by this step."""
        if not hasattr(self, "_moves"):
            raise ValueError("Must add moves for this recipe step.")
        return self._moves

    @moves.setter
    def moves(self, moves):
        self._moves = moves

    @property
    def weights(self):
        """List of weights corresponding to :attr:`moves`. Uniform by default."""
        if not hasattr(self, "_weights"):
            self._weights = [1.0 / len(self.moves) for _ in self.moves]
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def setup_run(self, iteration, last_sample, sampler):
        """Configure ``sampler`` for the start of this recipe step."""
        raise NotImplementedError

    def stopping_function(self, iteration, last_sample, sampler):
        """Return ``True`` when this recipe step should be considered done."""
        raise NotImplementedError

class BaseRecipeStep(RecipeStep):
    """Default :class:`RecipeStep` that simply assigns moves to the sampler.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``.
    """

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        """Install :attr:`moves`/:attr:`weights` on the sampler.

        Each move that lacks an explicit periodicity setting inherits the
        sampler's periodicity.
        """
        for move in self.moves:
            if sampler.periodic is not None and move.periodic is None:
                logger.debug(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic

        sampler.moves = self.moves
        sampler.weights = self.weights
