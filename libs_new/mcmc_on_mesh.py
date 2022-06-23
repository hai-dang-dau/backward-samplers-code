from abc import ABC, abstractmethod
import typing as tp

from particles import hilbert

from libs_new.mesh import Mesh, DiscreteGaussianDistribution, NeighborMesh
from particles.kalman import MeanAndCov
from functools import partial
from libs_new.utils_math import ProposalProtocol, DistributionLike, Proposaler, CategoricalDistribution, \
    get_gaussian_posterior, MHKernel, MHProposalv2, LabelledCategoricalDistribution, DiscreteUniform, \
    hilbert_sort_alternative, SingletonDist, MHCompositeKernel
import numpy as np
from libs_new.utils import sum_of_each_column, do_sth_to_each_row_of, zip_with_assert, inverse_permutation, \
    cached_function
from libs_new.smoothing_mcmc_backwards import MCMCStartKit, MCMCKernelsCreator
from libs_new.smoothing_generic_backwards import TargetLogDensityCalculator
from libs_new.smoothing_generic_backwards import simple_mean as _simple_mean

_Ttm1 = tp.TypeVar('_Ttm1')  # Type of particle at time t - 1
_Tt = tp.TypeVar('_Tt')  # Type of particle at time t
_CostInfo = tp.TypeVar('_CostInfo')
_Tp = tp.TypeVar('_Tp')  # Generic type of particle

class DiscreteProposalProducer(ABC):
    """
    Create a proposal to simulate an expected_dist discretized over an approximating_mesh.
    """
    # tested
    def __call__(self, approximating_mesh: Mesh, expected_dist: MeanAndCov) -> ProposalProtocol[int]:
        return Proposaler(partial(self._call, approximating_mesh=approximating_mesh, expected_dist=expected_dist))

    @abstractmethod
    def _call(self, i: int, approximating_mesh: Mesh, expected_dist: MeanAndCov) -> DistributionLike:
        ...

class IndependentDiscreteGaussianProposal(DiscreteProposalProducer):
    # tested
    def _call(self, i: int, approximating_mesh: Mesh, expected_dist: MeanAndCov) -> DistributionLike:
        return DiscreteGaussianDistribution(mesh=approximating_mesh, means=expected_dist.mean, variances=np.diag(expected_dist.cov))

class RandomWalkDiscreteProposal(DiscreteProposalProducer):
    # tested
    def __init__(self, coef: float = 2.38**2):
        self.coef = coef

    def _call(self, i: int, approximating_mesh: Mesh, expected_dist: MeanAndCov) -> DistributionLike:
        return DiscreteGaussianDistribution(mesh=approximating_mesh, means=approximating_mesh.ref_arr[i], variances=self.coef / len(expected_dist.cov) * np.diag(expected_dist.cov))

class Decorrelator:
    # tested
    """
    Transform a correlated distribution to an uncorrelated one.
    """
    def __init__(self, Sigma):
        """
        :param Sigma: The correlation matrix of the initial distribution
        """
        try:
            _, v = np.linalg.eigh(Sigma)
            R = v.T
        except np.linalg.LinAlgError:
            R = np.identity(len(Sigma))
        self.R = R

    def transform_samples(self, x: tp.Sequence[tp.Sequence[float]]) -> tp.Sequence[tp.Sequence[float]]:
        return x @ self.R.T

    def transform_dist(self, dist: MeanAndCov) -> MeanAndCov:
        return MeanAndCov(mean=self.R @ dist.mean, cov=self.R @ dist.cov @ self.R.T)

class BackwardDistGaussianizer(ABC):
    @abstractmethod
    def __call__(self, X_tm1: tp.Sequence[_Ttm1], filtering_dist_tm1: CategoricalDistribution, smoothed_Xt: tp.Sequence[_Tt]) -> tp.Sequence[MeanAndCov]:
        ...

class ForwardGaussianDynamicsBDG(BackwardDistGaussianizer):
    # tested.
    def __call__(self, X_tm1: tp.Sequence[_Ttm1], filtering_dist_tm1: CategoricalDistribution,
                 smoothed_Xt: tp.Sequence[_Tt]) -> tp.Sequence[MeanAndCov]:
        prior_mean = sum_of_each_column(do_sth_to_each_row_of(X_tm1, filtering_dist_tm1.W, '*'))
        prior_cov = np.cov(X_tm1, rowvar=False, aweights=filtering_dist_tm1.W)
        posterior_getter = get_gaussian_posterior(prior=MeanAndCov(mean=prior_mean, cov=prior_cov), transform=self.transform, noise_cov=self.noise_var)
        return [posterior_getter(obs) for obs in smoothed_Xt]

    def __init__(self, transform: tp.Sequence[tp.Sequence[float]], noise_var: tp.Sequence[tp.Sequence[float]]):
        self.transform = transform
        self.noise_var = noise_var

class DiscretizedContinuousMCMC(MCMCKernelsCreator):
    # tested
    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1],
                 smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        target_dists = [self.target_log_density_calc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t) for x_t in smoothed_X_t]
        approximate_continuous_targets = self.bd_gaussianizer(X_tm1=X_tm1, filtering_dist_tm1=filtering_dist_tm1, smoothed_Xt=smoothed_X_t)
        typical_var_matrix = _simple_mean([ad.cov for ad in approximate_continuous_targets])
        transformer = Decorrelator(Sigma=typical_var_matrix)
        mesh = Mesh(transformer.transform_samples(X_tm1))

        new_continuous_dists = [transformer.transform_dist(ocd) for ocd in approximate_continuous_targets]
        proposals = [self.dpp(approximating_mesh=mesh, expected_dist=cd) for cd in new_continuous_dists]
        kernels = [MHKernel(tld, p) for tld, p in zip_with_assert(target_dists, proposals)]
        # continuous_starting_points = [multivariate_normal.rvs(mean=mean, cov=cov) for mean, cov in new_continuous_dists]  # slow
        continuous_starting_points = [np.random.normal(loc=mean, scale=np.sqrt(np.diag(cov))) for mean, cov in new_continuous_dists]
        discrete_starting_points = [mesh.get_cell_containing(x).content[0] for x in continuous_starting_points]
        return [MCMCStartKit(k, sp) for k, sp in zip(kernels, discrete_starting_points)]

    def __init__(self, target_log_density_calculator: TargetLogDensityCalculator, backward_dist_gaussianizer: BackwardDistGaussianizer, discrete_proposal_producer: DiscreteProposalProducer):
        self.target_log_density_calc = target_log_density_calculator
        self.bd_gaussianizer = backward_dist_gaussianizer
        self.dpp = discrete_proposal_producer

def neighbor_mcmc_proposal_ptcl(x: int, nmesh: NeighborMesh) -> MHProposalv2:
    xstar = nmesh.rvs_markov(x)
    return MHProposalv2(x_star=xstar, log_proposal_attractiveness=0)

class NeighborMCMCKreator(MCMCKernelsCreator):
    # tested
    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1],
                 smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        nmesh = NeighborMesh(arr=X_tm1, max_size=self.nmesh_maxsize)
        proposal = partial(neighbor_mcmc_proposal_ptcl, nmesh=nmesh)
        res = []
        for x_t in smoothed_X_t:
            target_log_density = self.tldc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t)
            kernel = MHKernel(log_density=target_log_density, proposal=proposal)
            res.append(MCMCStartKit(kernel=kernel, starting_point=0))
        return res

    def __init__(self, tldc: TargetLogDensityCalculator, nmesh_maxsize: int):
        self.tldc = tldc
        self.nmesh_maxsize = nmesh_maxsize

def _rank_set(r:int, N: int, max_rss: int) -> tp.Tuple[int, int]:
    # tested
    """
    :param r: an integer (in pratice, it will be the Hilbert rank of a certain particle)
    :param N: the number of particles
    :param max_rss: max size of each rank set
    :return: the rank set associated with particle whose rank is r
    """
    assert r < N
    low = r - r % max_rss
    high = min(N, low + max_rss)
    return int(low), int(high)

def _hilbert_proposal(i: int, lw: CategoricalDistribution, ordered: tp.Sequence[int], rank: tp.Sequence[int], max_rss: int) -> LabelledCategoricalDistribution:
    # tested but replaced by _HilbertProposal below.
    """
    Calculate the proposal distribution for the Hilbert MCMC Algorithm for smoothing
    :param i: index of the current particle
    :param lw: weights of the particle
    :param ordered: list of particle indices, ordered by the Hilbert curve
    :param rank: rank of particles by the Hilbert curve. Should be the inverse mapping of `ordered`
    :param max_rss: max size of each rank set
    :return: proposal distribution q(i*|i)
    """
    rank_i = rank[i]
    N = len(lw)
    rankset = _rank_set(rank_i, N, max_rss)
    proposable_idx = ordered[rankset[0]:rankset[1]]
    proposable_lw = lw.lw[proposable_idx]
    return LabelledCategoricalDistribution(lw=proposable_lw, labels=proposable_idx)

class _HilbertProposal:
    # tested, version 170821
    def __init__(self, lw: CategoricalDistribution, ordered: tp.Sequence[int], rank: tp.Sequence[int], max_rss: int, use_lw=True):
        self.lw = lw
        self.ordered = ordered
        self.rank = rank
        self.max_rss = max_rss
        self.use_lw = use_lw

    def __call__(self, i: int) -> LabelledCategoricalDistribution:
        rank_i = self.rank[i]
        N = len(self.lw)
        rankset = _rank_set(rank_i, N, self.max_rss)
        return self._rankset_to_proposal(rankset)

    @cached_function
    def _rankset_to_proposal(self, rankset):
        proposable_idx = self.ordered[rankset[0]:rankset[1]]
        if self.use_lw:
            proposable_lw = self.lw.lw[proposable_idx]
            return LabelledCategoricalDistribution(lw=proposable_lw, labels=proposable_idx, small_support=(self.max_rss < 100))
        else:
            return DiscreteUniform(proposable_idx)

SetOfPoints = tp.Sequence[tp.Sequence[float]]

def sort_filtering_particles(self: tp.Union['HilbertMoveCreator', 'IrreversibleHilbertKernel'], X_tm1, filtering_dist_tm1):
    N = len(filtering_dist_tm1)
    d = X_tm1.shape[1]
    # cov_Xtm1 = np.cov(X_tm1, rowvar=False)  # do not include prior weights
    # cov_Xtm1 = np.atleast_2d(cov_Xtm1)
    # assert cov_Xtm1.shape == (d, d)
    #
    # decorred_Xtm1 = Decorrelator(cov_Xtm1).transform_samples(X_tm1)
    # assert decorred_Xtm1.shape == (N, d)
    decorred_Xtm1 = self.presort_transformer(X_tm1, filtering_dist_tm1)
    decorred_Xtm1 = np.c_[decorred_Xtm1, filtering_dist_tm1.W]
    assert decorred_Xtm1.shape == (N, d + 1)
    ordered_idx = self.sorter(decorred_Xtm1)
    if d == 1:
        ordered_idx = ordered_idx.reshape((N,))
    assert ordered_idx.shape == (N,)
    h_rank = inverse_permutation(ordered_idx)
    return h_rank, ordered_idx

class HilbertMoveCreator(MCMCKernelsCreator):
    # tested
    def __init__(self, tldc: TargetLogDensityCalculator, max_rss: int, presort_transformer: tp.Callable[[SetOfPoints, CategoricalDistribution], SetOfPoints], use_lw=True, puresort=False):
        self.tldc = tldc
        self.max_rss = max_rss
        self.presort_transformer = presort_transformer
        self.use_lw = use_lw
        self.sorter = hilbert.hilbert_sort if not puresort else hilbert_sort_alternative

    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1],
                 smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        # noinspection PyTypeChecker
        h_rank, ordered_idx = sort_filtering_particles(self, X_tm1, filtering_dist_tm1)
        # proposal_dist = partial(_hilbert_proposal, lw=filtering_dist_tm1, ordered=ordered_idx, rank=h_rank,
        #                         max_rss=self.max_rss)
        proposal_dist = _HilbertProposal(lw=filtering_dist_tm1, ordered=ordered_idx, rank=h_rank,max_rss=self.max_rss, use_lw=self.use_lw)
        proposaler = Proposaler(get_proposal_dist=proposal_dist)
        dummy_starting_point = filtering_dist_tm1.rvs()

        res = []
        for x_t in smoothed_X_t:
            target_log_dens = self.tldc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t)
            kernel = MHKernel(log_density=target_log_dens, proposal=proposaler)
            start_kit = MCMCStartKit(kernel=kernel, starting_point=dummy_starting_point)
            res.append(start_kit)

        return res

def _proposal_for_irreversible_hilbert(i: int, ordered: tp.Sequence[int], rank: tp.Sequence[int], sign: bool) -> SingletonDist:
    # tested with 240821
    N = len(ordered)
    if rank[i] % 2 == sign:
        proposal_rank = min(rank[i] + 1, N - 1)
    else:
        proposal_rank = max(rank[i] - 1, 0)
    return SingletonDist(ordered[proposal_rank])

class IrreversibleHilbertKernel(MCMCKernelsCreator):
    # tested using hilbert_mcmc_test.py
    # Version 250821
    # Invariance cannot be doubted
    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1],
                 smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        # noinspection PyTypeChecker
        hrank, ordered_idx = sort_filtering_particles(self=self, X_tm1=X_tm1, filtering_dist_tm1=filtering_dist_tm1)
        proposal1 = Proposaler(partial(_proposal_for_irreversible_hilbert, ordered=ordered_idx, rank=hrank, sign=False))
        proposal2 = Proposaler(partial(_proposal_for_irreversible_hilbert, ordered=ordered_idx, rank=hrank, sign=True))
        dummy_starting_point = filtering_dist_tm1.rvs()

        res = []

        for xt in smoothed_X_t:
            target_log_dens = self.tldc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=xt)
            kernel1 = MHKernel(log_density=target_log_dens, proposal=proposal1)
            kernel2 = MHKernel(log_density=target_log_dens, proposal=proposal2)
            kernel = MHCompositeKernel(kernel1, kernel2)
            res.append(MCMCStartKit(kernel=kernel, starting_point=dummy_starting_point))

        return res

    def __init__(self, tldc: TargetLogDensityCalculator, presort_transformer: tp.Callable[[SetOfPoints, CategoricalDistribution], SetOfPoints]):
        self.tldc = tldc
        self.presort_transformer = presort_transformer
        self.sorter = hilbert.hilbert_sort