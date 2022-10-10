import numpy as np

from eryn.moves import Move

class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_prob.shape)
        try:
            self.temperature_control.swaps_accepted = np.zeros(len(self.temperature_control.betas) - 1, dtype=int)
        except AttributeError:
            pass

        return state, accepted