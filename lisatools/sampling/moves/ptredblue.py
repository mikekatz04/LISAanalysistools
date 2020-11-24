import emcee

from emcee.state import State
from emcee.moves import Move
import numpy as np


class PTRedBlueMove(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.
    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.
        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.
        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.
    """

    def __init__(
        self,
        betas,
        nwalkers,
        ndim,
        nsplits=2,
        randomize_split=True,
        live_dangerously=False,
    ):

        self.betas = betas

        self.ntemps = len(self.betas)
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance
        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.
        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        for split in range(self.nsplits):
            S1 = inds == split

            # Get the two halves of the ensemble.
            sets = [state.coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = sets[:split] + sets[split + 1 :]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp) in enumerate(zip(all_inds[S1], factors, new_log_probs)):

                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1)

        logp = state.blobs[:, 1]

        x, logP, logl, logp = self._temperature_swaps(
            state.coords.copy(),
            state.log_prob.copy(),
            state.blobs[:, 0].copy(),
            state.blobs[:, 1].copy(),
        )

        state = State(
            x.reshape(-1, self.ndim),
            log_prob=logP.flatten(),
            blobs=np.array([logl.flatten(), logp.flatten()]).T,
        )
        return state, accepted

    def _temperature_swaps(self, x, logP, logl, logp):
        """
        Perform parallel-tempering temperature swaps on the state in ``x`` with associated ``logP`` and ``logl``.
        """

        x = x.reshape(self.ntemps, self.nwalkers, -1)
        logP = logP.reshape(self.ntemps, self.nwalkers)
        logl = logl.reshape(self.ntemps, self.nwalkers)
        logp = logp.reshape(self.ntemps, self.nwalkers)
        ntemps, nwalkers = self.ntemps, self.nwalkers
        self.swaps_accepted = np.empty(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            iperm = np.random.permutation(nwalkers)
            i1perm = np.random.permutation(nwalkers)

            raccept = np.log(np.random.uniform(size=nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            # How many swaps were accepted?
            sel = paccept > raccept
            self.swaps_accepted[i - 1] = np.sum(sel)

            x_temp = np.copy(x[i, iperm[sel], :])
            logl_temp = np.copy(logl[i, iperm[sel]])
            logp_temp = np.copy(logp[i, iperm[sel]])
            logP_temp = np.copy(logP[i, iperm[sel]])

            x[i, iperm[sel], :] = x[i - 1, i1perm[sel], :]
            logl[i, iperm[sel]] = logl[i - 1, i1perm[sel]]
            logp[i, iperm[sel]] = logp[i - 1, i1perm[sel]]
            logP[i, iperm[sel]] = (
                logP[i - 1, i1perm[sel]] - dbeta * logl[i - 1, i1perm[sel]]
            )

            x[i - 1, i1perm[sel], :] = x_temp
            logl[i - 1, i1perm[sel]] = logl_temp
            logp[i - 1, i1perm[sel]] = logp_temp
            logP[i - 1, i1perm[sel]] = logP_temp + dbeta * logl_temp

        return x, logP, logl, logp


class PTStretchMove(PTRedBlueMove):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.
    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)
    """

    def __init__(self, *args, a=2.0, **kwargs):
        self.a = a
        super(PTStretchMove, self).__init__(*args, **kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a
        factors = (ndim - 1.0) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))
        return c[rint] - (c[rint] - s) * zz[:, None], factors
