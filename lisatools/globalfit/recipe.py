
class Recipe:
    def __init__(self):
        self.recipe = []
        self.backend_added = False
        self._current_iter = 0
        self._current_recipe_step = None
        self._has_setup_first_step = False

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend
        self.backend_added = True

    def add_recipe_component(self, adjust_fn, name=None):
        if name is None:
            name = f"recipe step {len(self.recipe) + 1}"
        self.recipe.append({"name": name, "adjust": adjust_fn, "status": False})
    
    def to_file(self):
        _tmp = {recipe_step["name"]: recipe_step["status"] for recipe_step in self.recipe}
        return _tmp

    def __next__(self):
        while self._current_iter < len(self.recipe):
            # False means it is not finished
            if self.recipe[self._current_iter]["status"]:
                self._current_iter += 1

            else:
                break

        if self._current_iter < len(self.recipe):
            self._current_recipe_step = self.recipe[self._current_iter]

    def setup_first_recipe_step(self, iteration, last_sample, sampler):
        assert not self._has_setup_first_step
        # move to next recipe step
        next(self)
        if self._current_iter >= len(self.recipe):
            raise ValueError("Recipe is already finished.")
            
        self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
        self._has_setup_first_step = True

    @property
    def current_recipe_step(self):
        return self._current_recipe_step

    def __call__(self, iteration, last_sample, sampler):
        stop_here = self._current_recipe_step["adjust"].stopping_function(iteration, last_sample, sampler)
        if stop_here:
            self.backend.completed_recipe_step(self._current_recipe_step["name"])
            self._current_recipe_step["status"] = True
            next(self)
            
        if self._current_iter >= len(self.recipe):
            return True
        else:
            self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
            return False


class RecipeStep:

    @property
    def moves(self):
        if not hasattr(self, "_moves"):
            raise ValueError("Must add moves for this recipe step.")
        return self._moves
    
    @moves.setter
    def moves(self, moves):
        self._moves = moves

    @property
    def weights(self):
        if not hasattr(self, "_weights"):
            self._weights = [1.0 / len(self.moves)  for _ in self.moves] 
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def setup_run(self, iteration, last_sample, sampler):
        raise NotImplementedError

    def stopping_function(self, iteration, last_sample, sampler):
        raise NotImplementedError
