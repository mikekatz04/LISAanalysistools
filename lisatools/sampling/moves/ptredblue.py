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
        adaptive=True,
        nsplits=2,
        randomize_split=True,
        live_dangerously=False,
        adaptation_lag=10000,
        adaptation_time=100,
    ):

        self.betas = betas

        self.ntemps = len(self.betas)
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

        self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)
        self.time = 0

        self.adaptive = adaptive
        self.adaptation_time, self.adaptation_lag = adaptation_time, adaptation_lag

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
        ntemps, nwalkers, ndim = self.ntemps, self.nwalkers, self.ndim
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers * ntemps, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        for split in range(self.nsplits):
            S1 = inds == split

            nwalkers_here = np.sum(S1)

            # need to update 2nd based on updated coords from 1st
            coords = state.coords.reshape(ntemps, nwalkers, ndim)

            # Get the move-specific proposal.
            q = np.zeros((ntemps, nwalkers_here, ndim))
            factors = np.zeros((ntemps, nwalkers_here))
            for t in range(ntemps):
                # Get the two halves of the ensemble.
                sets = [coords[t, inds == j] for j in range(self.nsplits)]
                s = sets[split]
                c = sets[:split] + sets[split + 1 :]

                q_temp, factors_temp = self.get_proposal(s, c, model.random)
                q[t] = q_temp
                factors[t] = factors_temp

            q = q.reshape(-1, ndim)
            factors = factors.flatten()

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            logl = new_log_probs.reshape(ntemps, nwalkers_here)
            logp = new_blobs.reshape(ntemps, nwalkers_here)
            logP = self._tempered_likelihood(logl, self.betas) + logp

            prev_logl = state.log_prob.reshape(ntemps, nwalkers)[:, all_inds[S1]]
            prev_logp = state.blobs.reshape(ntemps, nwalkers)[:, all_inds[S1]]
            prev_logP = self._tempered_likelihood(prev_logl, self.betas) + prev_logp

            logP = logP.flatten()
            prev_logP = prev_logP.flatten()

            inds_temp = (
                np.tile(all_inds[S1], (ntemps, 1))
                + (nwalkers * np.arange(ntemps)[:, np.newaxis])
            ).flatten()

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp, olp) in enumerate(
                zip(inds_temp, factors, logP, prev_logP)
            ):

                lnpdiff = f + nlp - olp
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            S1_temp = np.tile(S1, (ntemps, 1)).flatten()
            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1_temp)

        logl = state.log_prob.reshape(ntemps, nwalkers)
        logp = state.blobs.reshape(ntemps, nwalkers)
        logP = self._tempered_likelihood(logl, self.betas) + logp

        x, logP, logl, logp = self._temperature_swaps(
            state.coords.reshape(ntemps, nwalkers, -1),
            logP.copy(),
            logl.copy(),
            logp.copy(),
        )

        ratios = self.swaps_accepted / self.swaps_proposed

        if self.adaptive and self.ntemps > 1:
            dbetas = self._get_ladder_adjustment(self.time, self.betas, ratios)
            self.betas += dbetas

        state = State(
            x.reshape(-1, self.ndim), log_prob=logl.flatten(), blobs=logp.flatten(),
        )

        self.time += 1
        return state, accepted

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.
        Here, we find a singularity that demands more careful attention; we allow the likelihood to dominate the
        temperature, since wandering outside the likelihood support causes a discontinuity.
        """

        if betas is None:
            betas = self.betas

        with np.errstate(invalid="ignore"):
            loglT = logl * betas[:, None]
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _temperature_swaps(self, x, logP, logl, logp):
        """
        Perform parallel-tempering temperature swaps on the state in ``x`` with associated ``logP`` and ``logl``.
        """

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

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
        """
        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0


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
