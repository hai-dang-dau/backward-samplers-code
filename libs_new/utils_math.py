import math
import typing as tp
from bisect import bisect
from functools import cached_property, partial
from itertools import cycle

import numpy as np
import sys
from particles import resampling as rs
from particles.hilbert import hilbert_array
from particles.kalman import MeanAndCov
from particles.resampling import multinomial_once
from scipy.integrate import solve_ivp
from scipy.stats import random_correlation, geom, norm, multivariate_normal, binom, beta, uniform, bernoulli
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen
from tqdm import tqdm
from libs_new.utils import sum_of_each_row, pickable_make_plot_function, has_len, unpack, qqplot_to_gaussian, \
    plot_histogram_vs_gaussian, quad_form, zip_with_assert, compare_densities
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt

VERY_SMALL = 1e-12

class RejectionSampler(ABC):
    """
    Generate one sample using rejection sampling.

    :param proposal: object with attributes `logpdf` and `rvs`
    :param logM: upper bound of the target_logpdf() - proposal.logpdf() ratio
    """
    # tested
    @property
    @abstractmethod
    def sample(self):
        ...

    @property
    def n_trials(self):
        return self._n_trials

    def __init__(self, proposal, target_logpdf: tp.Callable[[tp.Any], float], logM: float):
        self.proposal = proposal
        self.target_logpdf = target_logpdf
        self.logM = logM
        # Mutable attributes
        self._n_trials = 0
        self._log_accepts = []

    def _one_trial(self):
        x = self.proposal.rvs()
        u = np.random.rand()
        log_accept = self.target_logpdf(x) - self.logM - self.proposal.logpdf(x)
        if log_accept > 0:
            raise ValueError('logM is underestimated.')
        self._log_accepts.append(log_accept)
        self._n_trials += 1
        return x, np.log(u) < log_accept

class basic_rejection_sampler(RejectionSampler):
    # tested
    def __init__(self, proposal, target_logpdf: tp.Callable[[tp.Any], float], logM: float):
        super().__init__(proposal, target_logpdf, logM)
        while True:
            x, accepted = self._one_trial()
            if accepted:
                self._sample = x
                break

    @property
    def sample(self):
        return self._sample

    @property
    def log_mean_acc(self):
        """
        Log mean acceptance rate. The mean acceptance rate here is calculated over only one run and must be weighted by n_trials to produce consistent estimate of acceptance rate in case n_run -> +inf.
        """
        return rs.log_mean_exp(np.array(self._log_accepts))

def log(x):
    """
    Like np.log, but does not raise errors when x = 0
    """
    with np.errstate(divide='ignore'):
        return np.log(x)

def log_manual(x):
    return [math.log(e) if e > 1e-200 else -math.inf for e in x]

class CategoricalDistribution:
    # tested
    """
    Categorical distribution with queued generator.

    :param lw: log-weights, might not be normalized
    """
    def __init__(self, lw):
        self.W = rs.exp_and_normalise(np.array(lw))
        self.lw = log(self.W)
        self.gen = rs.MultinomialQueue(self.W)

    def rvs(self, size: int = None):
        if size is None:
            return self.gen.dequeue(1)[0]
        else:
            try:
                return self.gen.dequeue(size)
            except ValueError:
                return np.array([self.rvs() for _ in range(size)])

    def logpdf(self, x):
        return self.lw[x]

    def __len__(self):
        return len(self.lw)

def _unsafe_exp_and_normalise(lw):
    t = np.exp(lw)
    return t/t.sum()

class SmallSupportCategoricalDistribution:
    # tested
    def __init__(self, lw: tp.Sequence[float], unsafe_exp_and_normalise: bool):
        exp_and_normaliser = _unsafe_exp_and_normalise if unsafe_exp_and_normalise else rs.exp_and_normalise
        self.W = exp_and_normaliser(np.array(lw))
        self.lw = log_manual(self.W)

    def rvs(self, size: int = None):
        if size is None:
            return rs.multinomial_once(W=self.W)
        return rs.multinomial(W=self.W, M=size)

    def logpdf(self, x):
        return self.lw[x]

    def __len__(self):
        return len(self.lw)

    @classmethod
    def fromWeights(cls, W: tp.Sequence[float], unsafe_exp_and_normalise: bool):
        return cls(lw=log(W), unsafe_exp_and_normalise=unsafe_exp_and_normalise)

_TLabel = tp.TypeVar('_TLabel')

class LabelledCategoricalDistribution(tp.Generic[_TLabel]):
    # tested
    def __init__(self, lw: tp.Sequence[float], labels: tp.Sequence[_TLabel], small_support=True, unsafe_exp_and_normalise=True):
        assert len(lw) == len(labels)
        if small_support:
            self._cat_dist = SmallSupportCategoricalDistribution(lw, unsafe_exp_and_normalise=unsafe_exp_and_normalise)
        else:
            self._cat_dist = CategoricalDistribution(lw)
        self.int_to_label = labels
        self.label_to_int = {lab: i for i, lab in enumerate(labels)}

    def rvs(self, size: int = None) -> tp.Union[_TLabel, tp.Sequence[_TLabel]]:
        int_sample = self._cat_dist.rvs(size=size)
        if size is None:
            return self.int_to_label[int_sample]
        else:
            return [self.int_to_label[i] for i in int_sample]

    def logpdf(self, x: _TLabel) -> float:
        return self._cat_dist.logpdf(self.label_to_int[x])

    def __len__(self):
        return len(self._cat_dist)

CatLikeDist = tp.Union[CategoricalDistribution, SmallSupportCategoricalDistribution]

class CoupledCategoricalDist:
    # tested version 080921
    def __init__(self, d1: CatLikeDist, d2: CatLikeDist):
        self.d1 = d1
        self.d2 = d2
        assert len(d1) == len(d2)

        unnormalised_common_mass = np.minimum(d1.W, d2.W)
        self.common_proba = np.sum(unnormalised_common_mass)
        self.common_mass = SmallSupportCategoricalDistribution.fromWeights(W=unnormalised_common_mass, unsafe_exp_and_normalise=True)
        if not np.allclose(d1.W, d2.W):
            self.excess_d1 = SmallSupportCategoricalDistribution.fromWeights(W=d1.W - unnormalised_common_mass, unsafe_exp_and_normalise=True)
            self.excess_d2 = SmallSupportCategoricalDistribution.fromWeights(W=d2.W - unnormalised_common_mass, unsafe_exp_and_normalise=True)

    def rvs(self) -> tp.Tuple[int, int, bool]:
        if np.random.rand() < self.common_proba:
            r = self.common_mass.rvs()
            return r, r, True
        else:
            r1 = self.excess_d1.rvs()
            r2 = self.excess_d2.rvs()
            return r1, r2, False

class CoupledNormalDist:
    # tested using version 090921
    def __init__(self, mu1: float, mu2: float, sigma: float):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma

    def rvs(self) -> tp.Tuple[float, float, bool]:
        x1 = np.random.normal() * self.sigma + self.mu1
        log_proba_common_mass = ((x1 - self.mu1)**2 - (x1 - self.mu2)**2)/(2 * self.sigma**2)
        if np.log(np.random.rand()) < log_proba_common_mass:
            return x1, x1, True
        else:
            return x1, self.mu1 + self.mu2 - x1, False

class DiscreteUniform(tp.Generic[_TLabel]):
    # tested
    def __init__(self, state_set: tp.Sequence[_TLabel]):
        self.state_set = state_set
        self._logpdf = -np.log(len(self.state_set))

    def rvs(self, size: int = None) -> tp.Union[_TLabel, tp.Sequence[_TLabel]]:
        return self.state_set[np.random.randint(low=0, high=len(self.state_set), size=size)]

    # noinspection PyUnusedLocal
    def logpdf(self, x: _TLabel) -> float:
        return self._logpdf

    def __len__(self):
        return len(self.state_set)

class SingletonDist:
    def __init__(self, s):
        self.s = s

    def rvs(self, size: int = None):
        if size is None:
            return self.s
        else:
            print('Normally this shoudnt be called')
            return np.array([self.s] * size)

    def logpdf(self, s):
        if s == self.s:
            return 0
        else:
            print('Normally this shouldnt be called')
            return -np.inf

    def __len__(self):
        return 1

class discrete_rejection_sampler(RejectionSampler):
    # tested
    """
    Sampling for discrete distribution based on a rejection sampler, while switching to direct sampling if the cost of the rejection sampler is deemed to high. The `n_trials` attribute simulates the behavior of a pure rejection sampler.
    """
    def __init__(self, proposal: CategoricalDistribution, target_logpdf: tp.Callable[[int], float], logM: float):
        super().__init__(proposal, target_logpdf, logM)
        accepted = False
        for _ in range(len(proposal)):
            x, accepted = self._one_trial()
            if accepted:
                self._sample = x
                break
        if not accepted:
            self._direct_simulation()

    @cached_property
    def _target_logpdf_full(self):
        return np.array([self.target_logpdf(i) for i in range(len(self.proposal))])

    @cached_property
    def _geometric_dist_parameter(self):
        """
        The parameter of the geometric distribution associated with the number of trials
        """
        logp = rs.log_sum_exp(self._target_logpdf_full - self.logM)
        return np.exp(logp)

    def _direct_simulation(self):
        # FOR DEBUG ONLY
        self._sample = rs.multinomial_once(rs.exp_and_normalise(self._target_logpdf_full))
        # self._sample = CategoricalDistribution(self._target_logpdf_full).rvs()
        self._n_trials = len(self.proposal) + np.random.geometric(self._geometric_dist_parameter)

    @property
    def sample(self):
        return self._sample

class MultivariateNormalDistribution:
    """
    Multivariate normal distribution.

    :param mean: numpy array of length d
    :param cov: numpy array of shape (d,d)
    """
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov
        self.precision = np.linalg.inv(cov)
        self.d = cov.shape[0]
        self.cst = -self.d/2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))

    def logpdf(self, x: np.ndarray) -> tp.Union[float, np.ndarray]:
        # tested
        """
        Calculate the log probability density of the distribution at point(s) `x`.

        :param x: numpy array of shape (n,d). If n = 1, returns a float, otherwise returns an array of shape (n,)
        """
        x = x - self.mean
        z = (x @ self.precision) * x
        try:
            x.shape[1]
        except IndexError:
            return self.cst - 0.5 * sum(z)
        else:
            return self.cst - 0.5 * sum_of_each_row(z)

def random_mean_and_cov(d: int, range_mean: tp.Tuple[float, float] = (-1, 1), range_std: tp.Tuple[float, float] = (1, 2), seed=None) -> tp.Tuple[np.ndarray, np.ndarray]:
    # tested
    """
    Generate a random vector in `R^d` and a random `dxd` matrix to be used as the mean and the covariance matrix of a multivariate gaussian distribution. The correlation matrix is generated so that their eigenvalues are uniformly distributed in the simplex `{x_1 + ... + x_d = 1; x_i >=0}`.

    :param d: dimension
    :param range_mean: range of the uniform distribution on which the mean is drawed
    :param range_std: range of the uniform distribution on which the standard deviation is drawed
    :return: mean and cov
    """
    if seed is not None:
        np.random.seed(seed)
    eigs = np.diff(sorted(np.r_[0, np.random.rand(d - 1), 1])) * d
    corr = random_correlation.rvs(eigs)
    mu = np.random.uniform(range_mean[0], range_mean[1], size=d)
    S = np.diag(np.random.uniform(range_std[0], range_std[1], size=d))
    return mu, S @ corr @ S.T

class MixtureGeometricDistributions:
    # tested
    """
    Object to represent a mixture of geometric distributions (supported on [1,+infty[).

    :param weights: weights of the mixture components, must sum up to 1
    :param parameters: parameters of the geometric distributions, must be positive and of the same length as `weights`
    """
    def __init__(self, weights: tp.Sequence, parameters: tp.Sequence):
        assert len(weights) == len(parameters)
        assert np.allclose(np.sum(weights), 1)
        self.weights = np.array(weights)
        self.parameters = np.array(parameters)
        self.N = len(weights)
        self.cat_dist = CategoricalDistribution(lw=log(weights))

    def rvs(self, size: tp.Union[tuple, int]):
        rsize = np.product(size)
        components = self.cat_dist.rvs(rsize)
        parameters = self.parameters[components]
        res = geom.rvs(p=parameters)
        return res.reshape(size)

    def rvs_mean_of_truncated(self) -> float:
        """
        Generates a sample of distribution 1/N * sum_n min(T_n,N), where T_n are i.i.d. of the mixture distribution
        """
        original = self.rvs(self.N)
        truncated = np.minimum(original, self.N)
        # noinspection PyTypeChecker
        return np.mean(truncated)

class MCMC(ABC):
    # tested
    """
    Unvectorized MCMC algorithm.
    """
    # All attributes are immutable.
    def __init__(self, target_logpdf: tp.Callable[[tp.Any], float], starting_point, niter: int, history: bool, verbose: bool = False):
        self.target_logpdf = target_logpdf
        current_state = self.extend(starting_point)
        all_states = [current_state]
        if verbose:
            iteration_range = tqdm(range(niter))
        else:
            iteration_range = range(niter)
        for _ in iteration_range:
            current_state = self.next_state(current_state)
            if history:
                all_states.append(current_state)
        self._last_state = self.contract(current_state)
        self._all_states = [self.contract(state) for state in all_states] if history else None

    @abstractmethod
    def next_state(self, current_state):
        ...

    # noinspection PyMethodMayBeStatic
    def extend(self, state):
        """
        Sometimes it would be more practical to think of a Markov chain on an extended space rather than on the original space.
        """
        return state

    # noinspection PyMethodMayBeStatic
    def contract(self, state):
        """
        The inverse of `extend`
        """
        return state

    @property
    def last_state(self):
        return self._last_state

    @property
    def all_states(self):
        return self._all_states

    @pickable_make_plot_function
    def diagnostic_plot(self, f: tp.Callable[[tp.Any], float], ax: plt.Axes = None):
        if self.all_states is None:
            raise ValueError('History must have been enabled to use diagnostic plot')
        x = np.arange(len(self.all_states))
        y = [f(state) for state in self.all_states]
        ax.plot(x, y)

MHParticle = tp.NewType('MHParticle', object)
MHState = tp.NamedTuple('MHState', [('particle', MHParticle), ('loglik', float)])
MHProposal = tp.NamedTuple('MHProposal', [('xstar', MHParticle), ('log_att', float)])

class MetropolisHastings(MCMC, ABC):
    # tested
    """
    Unvectorized Metropolis-Hastings algorithm
    """
    def __init__(self, target_logpdf: tp.Callable[[tp.Any], float], starting_point, niter: int, history: bool, verbose: bool = False):
        self._acceptance_rate_tracker = []
        super().__init__(target_logpdf=target_logpdf, starting_point=starting_point, niter=niter, history=history, verbose=verbose)

    def extend(self, state: MHParticle) -> MHState:
        return MHState(particle=state, loglik=self.target_logpdf(state))

    def contract(self, state: MHState) -> MHParticle:
        return state.particle

    def next_state(self, current_state: MHState) -> MHState:
        x = current_state.particle
        ll = current_state.loglik

        proposal = self.proposal(x)

        x_star = proposal.xstar
        ll_star = self.target_logpdf(x_star)
        l_att = proposal.log_att

        log_acc = ll_star - ll - l_att
        if np.log(np.random.rand()) < log_acc:
            self._acceptance_rate_tracker.append(1)
            return MHState(x_star, ll_star)
        else:
            self._acceptance_rate_tracker.append(0)
            return current_state

    @abstractmethod
    def proposal(self, x: MHParticle) -> MHProposal:
        """
        returns xstar and its log-attractiveness (i.e. log q(x*|x) - log q(x|x^*))
        """
        ...

    @property
    def acceptance_rate(self):
        return sum(self._acceptance_rate_tracker)/len(self._acceptance_rate_tracker)

class independent_MH(MetropolisHastings):
    # tested
    """
    Independent Metropolis-Hastings algorithm.

    :param proposal_dist: object with logpdf and rvs method
    """

    def proposal(self, x: MHParticle) -> MHProposal:
        xstar = self.proposal_dist.rvs()
        return MHProposal(xstar=xstar, log_att=self.proposal_dist.logpdf(xstar) - self.proposal_dist.logpdf(x))

    def __init__(self, proposal_dist, target_logpdf: tp.Callable[[tp.Any], float], starting_point, niter: int, history: bool, verbose: bool = False):
        self.proposal_dist = proposal_dist
        super().__init__(target_logpdf, starting_point, niter, history, verbose)

class RWMH(MetropolisHastings):
    # tested
    def __init__(self, proposal_covariance_matrix: np.ndarray, target_logpdf: tp.Callable[[np.ndarray], float], starting_point: np.ndarray, niter: int, history: bool, verbose: bool = False):
        if has_len(starting_point):
            d = max(len(starting_point), 1)
        else:
            d = 1
        if d == 1:
            sd = unpack(proposal_covariance_matrix)**0.5
            self.proposal_dist = norm(loc=0, scale=sd)
            starting_point = unpack(starting_point)
        else:
            self.proposal_dist = multivariate_normal(mean=[0]*d, cov=proposal_covariance_matrix)
        super().__init__(target_logpdf=target_logpdf, starting_point=starting_point, niter=niter, history=history, verbose=verbose)

    def proposal(self, x: np.ndarray) -> MHProposal:
        return MHProposal(xstar=x + self.proposal_dist.rvs(), log_att=0)

class Generator(tp.Protocol):
    def __call__(self, size: int = None) -> np.ndarray:
        ...

def test_finite_expectation(generator: Generator, N_clt: int, N_sample: int = 3000, verbose: bool = True, show: bool = True):
    # tested
    """
    returns a sample that would be close to normal-distributed if the given generator (which has a `size` parameter) generates a distribution with finite expectation.
    """
    sample = []
    for _ in tqdm(range(N_sample), disable=not verbose):
        x = generator(size=N_clt)
        sample.append(np.mean(np.sqrt(x)))
    if show:
        qqplot_to_gaussian(sample, quantile=0.95)
        plot_histogram_vs_gaussian(sample)
    return sample

def integrate_normal_ratio(mean1: np.ndarray, Sigma1: np.ndarray, mean2: np.ndarray, Sigma2: np.ndarray) -> float:
    # tested
    """
    Calculate int f(x)/g(x) dx, where f and g are densities of two multivariate normal densities.
    """
    d = Sigma1.shape[0]
    L1 = np.linalg.inv(Sigma1)
    L2 = np.linalg.inv(Sigma2)
    L = L1 - L2
    if any(np.linalg.eigvals(L) < 1e-12):
        return np.inf
    mu = np.linalg.inv(L) @ (L1 @ mean1 - L2 @ mean2)
    r = 1/2 * (quad_form(L1, mean1) - quad_form(L2, mean2) - quad_form(L, mu))
    return (np.linalg.det(L1) / np.linalg.det(L2) / np.linalg.det(L) * (2*np.pi)**d)**0.5 * np.exp(-r)

def integrate_varianced_normal_ratio(mean1: np.ndarray, Sigma1: np.ndarray, mean2: np.ndarray, Sigma2: np.ndarray) -> float:
    # tested
    """
    Calculate int f(x)/g(x)**2 dx, where f and g are densities of two multivariate normal distributions.
    """
    d = Sigma1.shape[0]
    return ((4*np.pi)**d * np.linalg.det(Sigma2))**.5 * integrate_normal_ratio(mean1, Sigma1, mean2, 0.5*Sigma2)

class SimulatedCosts(tp.NamedTuple):
    pure: float
    hybrid: float

class DRSResult(tp.NamedTuple):
    sample: tp.Any
    costs: SimulatedCosts
    direct_sampler: tp.Union[None, Generator]
    exact_acc_rate: tp.Union[None, float]

def discrete_rejection_sampler_v2(proposal: CategoricalDistribution, log_target_density: tp.Callable[[int], float], log_M: float) -> DRSResult:
    # tested
    """
    :param log_M: upper bound of ``log_target_density - log_proposal``.
    """
    for i in range(len(proposal)):
        x = proposal.rvs()
        u = np.random.rand()
        log_acc = log_target_density(x) - log_M - proposal.logpdf(x)
        if log_acc > VERY_SMALL:
            raise ValueError('log_M is under-estimated')
        elif np.log(u) < log_acc:
            return DRSResult(sample=x, costs=SimulatedCosts(pure=i+1, hybrid=i+1), direct_sampler=None, exact_acc_rate=None)

    log_target = np.array([log_target_density(j) for j in range(len(proposal))])
    sampler = CategoricalDistribution(lw=log_target)
    exact_acc_rate = np.exp(-log_M + rs.log_sum_exp(log_target))
    pure_cost = len(proposal) + np.random.geometric(exact_acc_rate)
    return DRSResult(sample=sampler.rvs(), direct_sampler=sampler.rvs, costs=SimulatedCosts(pure=pure_cost, hybrid=len(proposal)), exact_acc_rate=exact_acc_rate)

class MDRSResult(tp.NamedTuple):
    samples: tp.Sequence
    costs: SimulatedCosts

def multiple_discrete_rejection_sampler(proposal: CategoricalDistribution, log_target_density: tp.Callable[[int], float], log_M: float, N_tilde: int) -> MDRSResult:
    # tested
    direct_sampler: tp.Union[None, Generator] = None
    pure_cost = 0
    hybrid_cost = 0
    samples = []
    acc_rate: tp.Union[None, float] = None

    for i in range(N_tilde):
        if direct_sampler is None:
            sampler = discrete_rejection_sampler_v2(proposal, log_target_density, log_M)
            samples.append(sampler.sample)
            direct_sampler = sampler.direct_sampler
            acc_rate = sampler.exact_acc_rate
            hybrid_cost += sampler.costs.hybrid
            pure_cost += sampler.costs.pure
        else:
            samples.append(direct_sampler())
            hybrid_cost += 1
            assert acc_rate is not None
            pure_cost += np.random.geometric(acc_rate)

    return MDRSResult(samples=samples, costs=SimulatedCosts(pure=pure_cost, hybrid=hybrid_cost))

_Tstate = tp.TypeVar('_Tstate')
_T_extended_state = tp.TypeVar('_T_extended_state')
_T_MCMC_Info = tp.TypeVar('_T_MCMC_Info')

class MCMCStep(tp.NamedTuple):
    extended_state: _T_extended_state
    info: _T_MCMC_Info

class MCMCKernel(ABC):
    @abstractmethod
    def extend(self, x: _Tstate) -> _T_extended_state:
        ...

    @abstractmethod
    def step(self, x: _T_extended_state) -> MCMCStep:
        ...

    @abstractmethod
    def contract(self, x: _T_extended_state) -> _Tstate:
        ...

class MHProposalv2(tp.NamedTuple, tp.Generic[_Tstate]):
    x_star: _Tstate
    log_proposal_attractiveness: float  # log(x_star) - log(x)

class MHExtendedState(tp.NamedTuple):
    state: _Tstate
    logpdf: float

class MHInfo(tp.NamedTuple):
    accepted: bool
    really_moved: bool

ProposalProtocol = tp.Callable[[_Tstate], MHProposalv2]

class MHKernel(MCMCKernel):
    # tested
    def __init__(self, log_density: tp.Callable[[_Tstate], float], proposal: ProposalProtocol):
        self.log_density = log_density
        self.proposal = proposal

    def extend(self, x: _Tstate) -> MHExtendedState:
        return MHExtendedState(state=x, logpdf=self.log_density(x))

    def step(self, x: MHExtendedState) -> MCMCStep:
        x_star, log_proposal_att = self.proposal(x.state)
        log_pdf_xstar = self.log_density(x_star)
        log_acc = log_pdf_xstar - x.logpdf - log_proposal_att
        if np.log(np.random.rand()) < log_acc:
            return MCMCStep(extended_state=MHExtendedState(state=x_star, logpdf=log_pdf_xstar), info=MHInfo(accepted=True, really_moved=(x_star != x.state)))
        else:
            return MCMCStep(extended_state=x, info=MHInfo(accepted=False, really_moved=False))

    def contract(self, x: _T_extended_state) -> _Tstate:
        return x.state

class _MCMCRunner:
    # tested
    def __init__(self, kernel: MCMCKernel, starting_point: _Tstate):
        self.kernel = kernel
        self.current_state = starting_point

    def __iter__(self):
        self.current_extended_state = self.kernel.extend(self.current_state)
        self.current_info = None
        return self

    def __next__(self):
        step = self.kernel.step(self.current_extended_state)
        self.current_extended_state = step.extended_state
        self.current_state = self.kernel.contract(self.current_extended_state)
        self.current_info = step.info
        return self

class DistributionLike(tp.Protocol[_Tstate]):
    def rvs(self) -> _Tstate: ...

    def logpdf(self, x: _Tstate) -> float: ...

class IndepMHKernel(MHKernel):
    # tested
    def __init__(self, log_target: tp.Callable[[_Tstate], float], proposal_dist: DistributionLike):
        super().__init__(log_density=log_target, proposal=partial(self._proposal_func, proposal_dist=proposal_dist))

    @staticmethod
    def _proposal_func(x: _Tstate, proposal_dist: DistributionLike) -> MHProposalv2:
        x_star = proposal_dist.rvs()
        log_att = proposal_dist.logpdf(x_star) - proposal_dist.logpdf(x)
        return MHProposalv2(x_star=x_star, log_proposal_attractiveness=log_att)

class Proposaler:
    # tested
    def __init__(self, get_proposal_dist: tp.Callable[[_Tstate], DistributionLike]):
        self.get_proposal_dist = get_proposal_dist

    def __call__(self, x: _Tstate) -> MHProposalv2:
        proposal_dist_x = self.get_proposal_dist(x)
        xstar = proposal_dist_x.rvs()
        logp_x_to_xstar = proposal_dist_x.logpdf(xstar)
        log_xstar_to_x = self.get_proposal_dist(xstar).logpdf(x)
        return MHProposalv2(x_star=xstar, log_proposal_attractiveness=logp_x_to_xstar - log_xstar_to_x)

# noinspection PyTypeChecker
Observation = tp.TypeVar('Observation', tp.Sequence[float], None)

def get_gaussian_posterior(prior: MeanAndCov, transform: tp.Sequence[tp.Sequence[float]], noise_cov: tp.Sequence[tp.Sequence[float]]) -> tp.Callable[[Observation], MeanAndCov]:
    # tested
    Q0 = prior.cov; m0 = prior.mean
    B = transform; R = noise_cov
    dx = len(Q0); idm = np.identity(dx)
    aux = np.linalg.inv(B @ Q0 @ B.T + R)
    Q1 = Q0 @ (idm - B.T @ aux @ B @ Q0)
    m1_first_part = (idm - Q0 @ B.T @ aux @ B) @ m0
    m1_second_part_prefix = Q0 @ B.T @ aux
    return partial(_get_gaussian_posterior, Q1=Q1, m1_first_part=m1_first_part, m1_second_part_prefix=m1_second_part_prefix)

def _get_gaussian_posterior(obs, Q1, m1_first_part, m1_second_part_prefix):
    m1 = m1_first_part + m1_second_part_prefix @ obs
    return MeanAndCov(mean=m1, cov=Q1)

def expectation_wrt_binomial_dist(f: tp.Callable[[int], float], N: int, p: float) -> float:
    # tested
    dist = binom(n=N, p=p)
    ls = [f(n) * dist.pmf(n) for n in range(0, N+1)]
    return sum(ls)

def hilbert_sort_alternative(x):
    d = 1 if x.ndim == 1 else x.shape[1]
    if d == 1:
        return np.argsort(x, axis=0)
    xs = x/np.max(np.abs(x))
    maxint = np.floor(2 ** (62 / d))
    xint = np.floor(xs * maxint).astype(np.int)
    return np.argsort(hilbert_array(xint))

class MHCompositeKernel:
    """
    Combine multiple Metropolis-Hastings kernels to make a new irreversible kernel
    """
    def __init__(self, *kernels: MHKernel):
        self.kernels = kernels
        assert len(set([k.log_density for k in kernels])) == 1

class _MCMCRunnerComposite:
    # tested, 240821
    def __init__(self, kernel: MHCompositeKernel, starting_point: _Tstate):
        self.kernel = kernel
        self.current_state = starting_point
        self.current_extended_state = self.kernel.kernels[0].extend(starting_point)
        self.current_info = None
        self.kernel_cycler = iter(cycle(self.kernel.kernels))

    def __iter__(self):
        return self

    def __next__(self):
        kernel_now = next(self.kernel_cycler)
        self.current_extended_state, self.current_info = kernel_now.step(self.current_extended_state)
        self.current_state = kernel_now.contract(self.current_extended_state)
        return self

@tp.overload
def MCMCRunner(kernel: MCMCKernel, starting_point: _Tstate) -> _MCMCRunner: ...
@tp.overload
def MCMCRunner(kernel: MHCompositeKernel, starting_point: _Tstate) -> _MCMCRunnerComposite: ...

def MCMCRunner(kernel, starting_point):
    if isinstance(kernel, MCMCKernel):
        return _MCMCRunner(kernel, starting_point)
    elif isinstance(kernel, MHCompositeKernel):
        return _MCMCRunnerComposite(kernel, starting_point)
    else:
        raise TypeError

def estimate_mean(x: tp.Sequence[float]) -> tp.Tuple[float, float]:
    # tested
    mean = np.mean(x)
    sd = np.std(x)
    N = len(x)
    return mean - sd * 1.96/np.sqrt(N), mean + sd * 1.96/np.sqrt(N)

class MarkovJumpProcess:
    # tested using version 110921
    """
    A Markov jump process is represented by a directed graph where each edge is marked by the associated rate. The graph is stored using the neighbour function.

    Terminology: a * neighbour function * sends each state into a * neighbour map *. The latter sends each state into a float.
    """
    def __init__(self, neighbour_func: tp.Callable[[_Tstate], tp.Mapping[_Tstate, float]], x0 : _Tstate, print_all_state: bool = False):
        """
        :param neighbour_func: for each state, return its adjacent edges with the weights. Black hole states return an empty mapping
        """
        self.neighbour_func = neighbour_func
        self.time_evolution = [0]
        self.state_evolution = [x0]
        self.print_all_state = print_all_state

    def step(self):
        adjacent_edges_and_weights = self.neighbour_func(self.state_evolution[-1])
        adjacent_edges_and_weights = self.regularise_map(adjacent_edges_and_weights, self.state_evolution[-1])
        if len(adjacent_edges_and_weights) > 0:
            # noinspection PyTypeChecker
            next_possible_states = [_ for _ in adjacent_edges_and_weights.keys()]
            rates = np.array([_ for _ in adjacent_edges_and_weights.values()])
            sum_rates = sum(rates)
            dt = np.random.exponential(scale=1/sum_rates)
            standardised_rates = rates/sum_rates
            next_state = next_possible_states[rs.multinomial_once(standardised_rates)]
        else:
            dt = np.inf
            next_state = self.state_evolution[-1]
        self.time_evolution.append(self.time_evolution[-1] + dt)
        self.state_evolution.append(next_state)
        if self.print_all_state:
            print(next_state)

    def get_xt(self, t: float) -> _Tstate:
        self._simulate_until(t)
        return self._return_state(time_evolution=self.time_evolution, state_evolution=self.state_evolution, t=t)

    def _simulate_until(self, t: float):
        while not (self.time_evolution[-1] > t):
            self.step()

    @staticmethod
    def _return_state(time_evolution: tp.Sequence[float], state_evolution: tp.Sequence[_Tstate], t: float) -> _Tstate:
        # tested
        if t < 0 or t > time_evolution[-1]:
            raise ValueError
        else:
            return state_evolution[bisect(time_evolution, t) - 1]

    @classmethod
    def fromMatrix(cls, A_matrix: np.ndarray, state_labels: tp.Tuple[_Tstate], initial_val: _Tstate):
        """
        Create a new Markov jump process on a finite space using a transition rate matrix
        :param A_matrix: transition rate matrix (shape (|S|,|S|))
        :param state_labels: names of states (must be of length |S|)
        """
        if len(state_labels) != len(A_matrix):
            raise ValueError
        transition_dict = {}
        for i, s in enumerate(state_labels):
            dict_s = {}
            for j, s_prime in enumerate(state_labels):
                if j != i:
                    dict_s[s_prime] = A_matrix[i, j]
            transition_dict[s] = dict_s
        return cls(neighbour_func=transition_dict.__getitem__, x0=initial_val)

    @staticmethod
    def get_invariant_measure(A_matrix: np.ndarray):
        # tested
        S = len(A_matrix)
        B = np.r_[np.array([[1] * S]), A_matrix.T]
        return np.linalg.solve(B[0:S], [1] + [0] * (S - 1))

    @staticmethod
    def regularise_map(neighbour_map: tp.Mapping[_Tstate, float], state: _Tstate) -> tp.Mapping[_Tstate, float]:
        neighbour_map = neighbour_map.copy()
        if state in neighbour_map:
            del neighbour_map[state]
        zero_keys = []
        # noinspection PyTypeChecker
        for k, v in neighbour_map.items():
            if v == 0:
                zero_keys.append(k)
        for zk in zero_keys:
            del neighbour_map[zk]
        return neighbour_map

_Tsubstate = tp.TypeVar('_Tsubstate')

def random_generator_matrix(n: int) -> np.ndarray:
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = 0
        A[i, i] = -np.sum(A[i])
    return A

def random_sparse_generator_matrix(n: int, sup: int) -> np.ndarray:
    A = np.zeros(shape=(n,n))
    for i in range(n):
        A[i, (i + 1) % n] = np.random.rand() * 2
    for _ in range(sup):
        A[np.random.randint(low=0, high=n), np.random.randint(low=0, high=n)] = np.random.rand()
    for i in range(n):
        A[i, i] = 0
        A[i, i] = -np.sum(A[i])
    return A

def newton_method(f: tp.Callable[[float], float], fprime: tp.Callable[[float], float], ftwoprime: tp.Callable[[float], float], niter: int, x0: float = 0, verbose: bool = False) -> float:
    # tested
    x = x0
    for _ in range(niter):
        a = ftwoprime(x)/2
        b = fprime(x)
        c = f(x)
        if verbose:
            print('x: {}, obj: {}'.format(x, c))
        dx = newton_quad_solver(a=a, b=b, c=c)
        x = x + dx
    if verbose:
        print('=== Completed ===')
    return x

def newton_quad_solver(a, b, c):
    if a == 0:
        if b == 0:
            return 0
        else:
            return -c/b
    else:
        return 1/(2 * a) * (-b + (max(0, b**2 - 4 * a * c)) ** 0.5)

def exp_without_overflow(x):
    return np.exp(np.minimum(x, 100))

def bisection_method(f: tp.Callable[[float], float], niter: int, xlow: float = 0, xhigh: float = 1, verbose: bool = False) -> float:
    # tested
    for i in range(niter):
        f_low = f(xlow)
        f_high = f(xhigh)
        xmid = (xlow + xhigh) / 2
        if f_low * f_high > 0:
            xlow = 2 * xlow - xmid
            xhigh = 2 * xhigh - xmid
        else:
            f_mid = f(xmid)
            if f_low * f_mid > 0:
                xlow = (xmid + xhigh)/2
            else:
                xhigh = (xmid + xlow)/2
        if verbose:
            print('x: {}, func: {}'.format(xmid, f(xmid)))
    return (xlow + xhigh)/2

def conf_int(npos, nobs):
    return beta.ppf(q=[0.025, 0.975], a=npos + 1/2, b=nobs-npos + 1/2)

def Euler_solver(f: tp.Callable, t0: float, t1: float, nstep: int, x0) -> tp.Tuple[tp.Sequence[float], tp.Sequence]:
    # tested
    """
    Solve the ODE x' = f(x) over [t0,t1] using nstep of the Euler method for an unvectorised function f.
    """
    grid = np.linspace(t0, t1, nstep)
    x = [x0]
    for i, t in zip_with_assert(range(1, len(grid)), grid[1:]):
        t_old = grid[i-1]
        x_old = x[-1]
        x.append(f(x_old) * (t - t_old) + x_old)
    return grid, x

def _plot_solution(t, y, label, ax):
    line, = ax.plot(t, y[0], label=label)
    line: plt.Line2D
    color = line.get_color()
    for i in range(len(y)):
        ax.plot(t, y[i], color=color)

class _add_dummy_args:
    def __init__(self, f):
        self.f = f

    def __call__(self, t, x):
        return self.f(x)

@pickable_make_plot_function
def Euler_solver_diag_plot(f: tp.Callable, t0: float, t1: float, nsteps: tp.Sequence[int], x0, nstep_scipy: int, ax: plt.Axes):
    for nstep in nsteps:
        sol = Euler_solver(f, t0, t1, nstep, x0)
        transposed_sol1 = np.array(sol[1]).T
        # noinspection PyTypeChecker
        _plot_solution(sol[0], transposed_sol1, str(nstep), ax)
    if nstep_scipy is not None:
        scipy_sol = solve_ivp(_add_dummy_args(f), (t0, t1), x0, t_eval=np.linspace(t0, t1, nstep_scipy))
        _plot_solution(scipy_sol.t, scipy_sol.y, 'scipy', ax)
    ax.legend()

def coupling_rate_estimate(d1: DistributionLike, d2: DistributionLike, N: int, cint=True) -> tp.Tuple[float, float]:
    # tested 271021
    """
    Estimate the maximal coupling probability between two distributions (i.e. one minus their total variation distance).
    """
    success = 0
    for _ in range(N):
        x = d1.rvs()
        # y = np.random.uniform(0, np.exp(d1.logpdf(x)))
        # if y < np.exp(d2.logpdf(x)):
        #     success += 1
        if np.log(np.random.rand()) < d2.logpdf(x) - d1.logpdf(x):
            success += 1
    return conf_int(success, N) if cint else success/N

class MixtureKernel:
    # tested 151121
    """
    Mixture of several Markov kernels on the same space. Programmatically, a Markov kernel is a function with random outputs. Input functions must therefore have the same signature
    """
    def __init__(self, kernels: tp.List[tp.Callable], weights: tp.Sequence[float]):
        if (len(kernels) != len(weights)) or (sum(weights) != 1):
            raise ValueError
        self.kernels = kernels
        self.weights = weights

    def __call__(self, *args, **kwargs):
        i = multinomial_once(self.weights)
        return self.kernels[i](*args, **kwargs)

def test_multivariate_gaussian(sample: tp.Sequence[tp.Sequence[float]], expected_mu, expected_Sigma, weight_dist: rv_frozen) -> None:
    # tested
    """
    Display a diagnostic graph to verify the distribution of a sample which is expected to have been drawn from a multivariate Gaussian.
    """
    sample = np.array(sample)
    N, d = sample.shape
    w = weight_dist.rvs(size=d)
    sample = (sample @ w).reshape((N,))
    expected_mu = float(np.dot(expected_mu, w))
    expected_Sigma = float(np.dot(expected_Sigma @ w, w))
    compare_densities(sample, norm.rvs(size=N, loc=expected_mu, scale=expected_Sigma**0.5))

def multinomial_sampling(W: np.ndarray) -> tp.Tuple[int, ...]:
    # tested
    """Multinomial sampling for multi-dimensional numpy arrays.
    :param W: a numpy array of weights, summing to 1
    :returns: a tuple of indices indicating the chosen element
    """
    W_raveled = np.ravel(W)
    #chosen_raveled_index = np.random.choice(len(W_raveled), p=W_raveled)
    chosen_raveled_index = rs.multinomial_once(W_raveled)
    return tuple(np.unravel_index(chosen_raveled_index, W.shape))

def TV_distance(x, y):
    """
    Total variation distance between two discrete distributions, represented by their weight arrays
    """
    return 1/2 * np.sum(np.abs(x - y))

def uniform_frozen_distribution(low, high) -> rv_frozen:
    return uniform(loc=low, scale=high-low)

def is_singular(A: np.ndarray, two_d: bool = False) -> bool:
    if two_d:
        if A.shape != (2, 2):
            raise ValueError
        return np.abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) < VERY_SMALL
    return np.linalg.cond(A) >= 1 / sys.float_info.epsilon

class rademacher:
    # tested
    @staticmethod
    def rvs(size=None):
        return 2 * bernoulli.rvs(size=size, p=0.5) - 1

    @staticmethod
    def logpmf(x):
        x = np.array(x)
        res = np.where(np.logical_or(x == 1, x == -1), np.log(1/2), -np.inf)
        if res.shape == ():
            return float(res)
        else:
            return res