import numpy as np

from emcee.moves.mh import MHMove


import numpy as np

from emcee.state import State

__all__ = ["ModifiedMHMove"]


class ModifiedMHMove(MHMove):
    r"""A general Metropolis-Hastings proposal
    Concrete implementations can be made by providing a ``proposal_function``
    argument that implements the proposal as described below.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be
    used.
    Args:
        proposal_function: The proposal function. It should take 2 arguments: a
            numpy-compatible random number generator and a ``(K, ndim)`` list
            of coordinate vectors. This function should return the proposed
            position and the log-ratio of the proposal probabilities
            (:math:`\ln q(x;\,x^\prime) - \ln q(x^\prime;\,x)` where
            :math:`x^\prime` is the proposed coordinate).
        ndim (Optional[int]): If this proposal is only valid for a specific
            dimension of parameter space, set that here.
    """

    def __init__(self, nwalkers, proposal_function=None, ndim=None, time_for_af=100):
        self.ndim = ndim
        if proposal_function is not None:
            self.get_proposal = proposal_function

        self.time = 0
        self.time_for_af = time_for_af

        self.num_accepted = np.zeros((self.time_for_af, nwalkers), dtype=int)

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance
        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.
        """
        # Check to make sure that the dimensions match.
        nwalkers, ndim = state.coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")

        # Get the move-specific proposal.
        q, factors = self.get_proposal(state.coords, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        ind = self.time % self.time_for_af

        self.num_accepted[ind] = accepted.astype(int)

        self.time += 1
        denom = self.time if self.time < self.time_for_af else self.time_for_af
        self.af = self.num_accepted.sum(axis=0) / denom

        return state, accepted


class MultiSourceFisherProposal(ModifiedMHMove):
    def __init__(
        self,
        nsystems,
        ndim,
        nwalkers,
        cov,
        desired_acceptance=0.25,
        adaptive=True,
        adapt_time=1,
        factor=1.0,
    ):

        ModifiedMHMove.__init__(self, nwalkers, ndim=ndim)
        self.nsystems, self.ndim, self.nwalkers, self.cov = (
            nsystems,
            ndim,
            nwalkers,
            cov,
        )
        self.per_system_ndim = int(self.ndim / self.nsystems)
        self.factors = np.full(nwalkers, factor)
        self.desired_acceptance = desired_acceptance
        self.adaptive = adaptive
        self.adapt_time = adapt_time

    def get_proposal(self, coords, random):
        mean = np.zeros(self.per_system_ndim)
        old_coords = coords.copy()

        # if self.time > 10 and self.adaptive and not (self.time % self.adapt_time) and np.all(self.af > 0.0):

        #    self.factors = np.exp(self.af - self.desired_acceptance) * self.factors

        if self.time % 25 == 0 and self.time > 0:
            print(
                self.factors,
                self.af,
                self.time > 10
                and self.adaptive
                and not (self.time % self.adapt_time)
                and np.all(self.af > 0.0),
            )

        new_change_in_coords = np.array(
            [
                np.array(
                    [
                        random.multivariate_normal(mean, cov_i * factor_i)
                        for cov_i in self.cov
                    ]
                ).flatten()
                for factor_i in self.factors
            ]
        )

        new_coords = old_coords + new_change_in_coords

        return (new_coords, np.zeros(self.nwalkers))
