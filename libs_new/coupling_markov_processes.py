from abc import abstractmethod
from functools import partial, cached_property

import numpy as np
import typing as tp
from collections import defaultdict
from libs_new.utils import reusable_map, cached_function
from libs_new.utils_math import MarkovJumpProcess

_Tsubstate = tp.TypeVar('_Tsubstate')
_Tstate = tp.TypeVar('_Tstate')

# Tested: wrap_and_check guarantees that the correct marginal distributions are being simulated.

class wrap_and_check:
    """
    Wrapper for the neighbor function of a coupled Markov jump process that verifies whether the marginal distribution is correctly coded.
    """
    def __init__(self, coupled_neighbour_func: tp.Callable[[tp.Tuple[_Tsubstate, _Tsubstate]], tp.Any], single_neighbour_func: tp.Callable[[_Tsubstate], tp.Mapping[_Tsubstate, float]]):
        self.coupled_neighbour_func = coupled_neighbour_func
        self.single_neighbour_func = single_neighbour_func

    def __call__(self, coupled_state: tp.Tuple[_Tsubstate, _Tsubstate]) -> tp.Dict[tp.Tuple[_Tsubstate, _Tsubstate], float]:
        substate1, substate2 = coupled_state
        coupled_neighbour_map: tp.Dict[tp.Tuple[_Tsubstate, _Tsubstate], float] = self.coupled_neighbour_func(coupled_state)
        # Main idea: the coupled neighbor map induces two marginal neighbour maps
        marginal_neighbour_map1: tp.Dict[_Tsubstate, float] = defaultdict(float)
        marginal_neighbour_map2: tp.Dict[_Tsubstate, float] = defaultdict(float)
        for (ss1, ss2), p in coupled_neighbour_map.items():
            assert float(p) >= 0
            marginal_neighbour_map1[ss1] += p
            marginal_neighbour_map2[ss2] += p
        self.equal_map(marginal_neighbour_map1, self.single_neighbour_func(substate1), substate1)
        self.equal_map(marginal_neighbour_map2, self.single_neighbour_func(substate2), substate2)
        return coupled_neighbour_map

    @staticmethod
    def equal_map(map1: tp.Dict[_Tstate, float], map2: tp.Dict[_Tstate, float], source: _Tstate):
        map1 = MarkovJumpProcess.regularise_map(map1, source)
        map2 = MarkovJumpProcess.regularise_map(map2, source)
        assert len(map1) == len(map2)
        for k in map1:
            assert np.allclose(map1[k], map2[k])

class Coupling_A_Matrix:
    # tested 161021
    """
    Represents the A matrix in the problem of coupling Markov processes, as well as related utilities. The definition of the A matrix is given in the following. If the state of the first process has M neighbours and that of the second one has N, the coupled process is described by a vector v of size (M+1)(N+1) - 1. This vector is subjected to a condition of the form A @ v = b.
    """
    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self.i_to_mn, self.mn_to_i = self.bidir_map(M, N)

    @staticmethod
    def bidir_map(M: int, N: int):
        # tested
        i_to_mn = []
        mn_to_i = {}
        i = 0
        for m in range(M + 1):
            for n in range(N + 1):
                i_to_mn.append((m, n))
                mn_to_i[m,n] = i
                i += 1
        i_to_mn = i_to_mn[:-1]
        del mn_to_i[M, N]
        return i_to_mn, mn_to_i

    @cached_property
    def free_vector_idx(self) -> tp.Sequence[int]:
        # tested
        """
        returns a sequence s of integers such that for any given vector b of length M + N, the vector v such that v = b at position defined by s and v = 0 elsewhere satisfies A @ v = b.
        """
        return [self.mn_to_i[m, n] for m, n in
                [(_m, self.N) for _m in range(self.M)] +
                [(self.M, _n) for _n in range(self.N)]
                ]

    @cached_property
    def shape(self):
        return self.M + self.N, (self.M + 1) * (self.N + 1) - 1

    @cached_property
    def non_zero_coordinates(self) -> tp.Sequence[tp.Tuple[int, int]]:
        # tested
        """
        List of the non-zero coordinates of A
        """
        res = []
        i = 0
        for m in range(self.M):
            res.extend([(i, self.mn_to_i[m, _n]) for _n in range(self.N + 1)])
            i += 1
        for n in range(self.N):
            res.extend([(i, self.mn_to_i[_m, n]) for _m in range(self.M + 1)])
            i += 1
        return res

    @cached_property
    def materialise(self) -> np.ndarray:
        A = np.zeros(self.shape)
        for i, j in self.non_zero_coordinates:
            A[i, j] = 1
        return A

    def __hash__(self):
        return hash((self.M, self.N))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.M == self.M and other.N == self.N

    def get_one_solution(self, b):
        res = np.zeros(self.shape[1])
        res[self.free_vector_idx] = b
        return res

class CoupledNeighbourMapCreator:
    """
    Convenient tools to help create coupled neighbour map
    """
    def __init__(self, state1: _Tsubstate, state2: _Tsubstate, single_neighbour_func: tp.Callable[[_Tsubstate], tp.Mapping[_Tsubstate, float]], coupling_A_matrix_creator: tp.Callable[[int, int], Coupling_A_Matrix] = None):
        nmap1 = single_neighbour_func(state1)
        self.nmap1 = MarkovJumpProcess.regularise_map(nmap1, state1)
        nmap2 = single_neighbour_func(state2)
        self.nmap2 = MarkovJumpProcess.regularise_map(nmap2, state2)
        self.state1 = state1
        self.state2 = state2
        self.coupling_A_matrix_creator = coupling_A_matrix_creator if coupling_A_matrix_creator is not None else Coupling_A_Matrix

    @cached_property
    def A(self):
        return self.coupling_A_matrix_creator(M=self.M, N=self.N)

    @cached_property
    def M(self):
        return len(self.nmap1)

    @cached_property
    def N(self):
        return len(self.nmap2)

    @cached_property
    def b(self) -> tp.Sequence[float]:
        return [_ for _ in self.nmap1.values()] + [_ for _ in self.nmap2.values()]

    @cached_property
    def nmap1_extended_keys(self) -> tp.Sequence[_Tsubstate]:
        # noinspection PyTypeChecker
        return [_ for _ in self.nmap1.keys()] + [self.state1]

    @cached_property
    def nmap2_extended_keys(self) -> tp.Sequence[_Tsubstate]:
        # noinspection PyTypeChecker
        return [_ for _ in self.nmap2.keys()] + [self.state2]

    @cached_property
    def i_to_key(self) -> tp.Sequence[tp.Tuple[_Tsubstate, _Tsubstate]]:
        assert isinstance(self.A.i_to_mn, list)
        return [(self.nmap1_extended_keys[m], self.nmap2_extended_keys[n]) for m, n in self.A.i_to_mn]

def successfully_coupled_neighbour_map(unique_state: _Tsubstate, single_neighbour_func: tp.Callable[[_Tsubstate], tp.Mapping[_Tsubstate, float]]) -> tp.Mapping[tp.Tuple[_Tsubstate, _Tsubstate], float]:
    """
    This function is never actually tested, but can be dynamically tested via is_vector_legal and wrap_and_check
    """
    unique_neighbour_map = single_neighbour_func(unique_state)
    unique_neighbour_map = MarkovJumpProcess.regularise_map(unique_neighbour_map, unique_state)
    # noinspection PyTypeChecker
    res = {(s, s): val for s, val in unique_neighbour_map.items()}
    return res

class OptimizeResult(tp.NamedTuple):
    x: tp.Sequence[float]
    status: int

class LossFunction(tp.Protocol):
    def __call__(self, state1, state2, state1_prime, state2_prime) -> float:
        ...

class Ac_Tuple(tp.NamedTuple):
    A: tp.Tuple[tp.Tuple[float, ...], ...]  # converted numpy array (2D matrix)
    c: tp.Tuple[float, ...]  # converted numpy array (vector)

class bx_Tuple(tp.NamedTuple):
    b: tp.Tuple[float, ...]
    x: tp.Tuple[float, ...]

class Optimiser:
    # tested 201021
    def __init__(self, cache_threshold: tp.Optional[float] = None):
        """
        :param cache_threshold: if not None, reuse the result of previous optimisation problems if the ratio between the previous `b` and the current `b` does not exceed `cache_threshold`
        """
        self.cache_threshold = cache_threshold
        self.previous_opts_results: tp.Dict[Ac_Tuple, bx_Tuple] = {}
        self.call_stats = dict(optcalls_skipped=0, optcalls_cached=0, optcalls_performed=0)

    @abstractmethod
    def _call(self, A: Coupling_A_Matrix, b, c, x0) -> OptimizeResult:
        ...

    def __call__(self, A, b, c_gen, x0, c_signature) -> OptimizeResult:
        """
        Solve the linear optimisation problem:
        min dot(c,x)
        s.t. A @ x = b and x >= 0
        using as starting point x0
        """
        c_signature = np.array(c_signature)
        if not self.cache_threshold:
            self.call_stats['optcalls_performed'] += 1
            return self._call(A=A, b=b, c=c_gen, x0=x0)
        try:
            res = OptimizeResult(x=self._use_previous_results(A=A, b=b, c=c_signature), status=0)
            self.call_stats['optcalls_cached'] += 1
            return res
        except (BufferError, KeyError):
            self.call_stats['optcalls_performed'] += 1
            res = self._call(A=A, b=b, c=c_gen, x0=x0)
            if res.status == 0:
                self._record(A=A, b=b, c=c_signature, x=res.x)
            return res

    def _use_previous_results(self, A, b, c) -> tp.Sequence[float]:
        tA, tc = A, self.vector_to_tuple(c)
        old_b, old_x = self.previous_opts_results[tA, tc]
        ratio = old_b/(b + 1e-12)
        diff = max(ratio)/min(ratio)
        if diff < self.cache_threshold:
            return self.free_variable_projection(A=A, old_x=old_x, old_b=old_b, new_b=b)
        else:
            raise BufferError

    def _record(self, A, b, c, x):
        tA, tc = A, self.vector_to_tuple(c)
        self.previous_opts_results[tA, tc] = (b, x)

    @staticmethod
    def vector_to_tuple(v):
        return tuple(v.tolist())

    @staticmethod
    def free_variable_projection(A: Coupling_A_Matrix, old_x: tp.Sequence[float], old_b: tp.Sequence[float], new_b: tp.Sequence[float]) -> tp.Sequence[float]:
        # noinspection PyTypeChecker
        lbd = min(new_b/(old_b + 1e-12))
        z = A.get_one_solution(new_b - lbd * old_b)
        return lbd * old_x + z

class OptimisedCoupling:
    # tested 201021
    def __init__(self, single_neighbour_function: tp.Callable[[_Tsubstate], tp.Dict[_Tsubstate, float]], loss_function: LossFunction, optimiser: Optimiser, loss_vector_signature: tp.Callable[[_Tsubstate, _Tsubstate], tp.Any] = None):
        """
        :param loss_vector_signature: used to replace loss vector comparison. This is because repeatedly calculate the loss function may be costly (e.g. taking O(# neighbours)^2 operations)
        """
        self.single_neighbour_function = single_neighbour_function
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.loss_vector_signature = loss_vector_signature
        self.coupling_A_matrix_creator = cached_function(Coupling_A_Matrix)

    def __call__(self, state_couple: tp.Tuple[_Tsubstate, _Tsubstate]) -> tp.Dict[tp.Tuple[_Tsubstate, _Tsubstate], float]:
        state1, state2 = state_couple
        if state1 == state2:
            self.optimiser.call_stats['optcalls_skipped'] += 1
            return successfully_coupled_neighbour_map(state1, self.single_neighbour_function)
        creator = CoupledNeighbourMapCreator(state1=state1, state2=state2,
                                             single_neighbour_func=self.single_neighbour_function,
                                             coupling_A_matrix_creator=self.coupling_A_matrix_creator)
        if len(creator.nmap1) == len(creator.nmap2) == 0:
            return {}
        A, b = creator.A, creator.b
        assert isinstance(creator.i_to_key, list)
        # c = [loss_function(state1, state2, state1_prime, state2_prime) for (state1_prime, state2_prime) in creator.vpis.int_to_pair]
        c_gen = reusable_map(partial(_loser, self.loss_function, state1, state2), creator.i_to_key)
        if self.loss_vector_signature is not None:
            c_signature = self.loss_vector_signature(state1, state2)
        else:
            c_gen = [_ for _ in c_gen]
            c_signature = c_gen
        x0 = creator.A.get_one_solution(creator.b)
        res = self.optimiser(A=A, b=np.array(b), c_gen=c_gen, x0=x0, c_signature=c_signature)
        if res.status == 0:
            res_vec = res.x
        else:
            print('Optimisation failed with error {}'.format(res.status))
            res_vec = x0
        return {key: val for key, val in zip(creator.i_to_key, res_vec)}

def _loser(loss_function, state1, state2, s12p):
    return loss_function(state1, state2, *s12p)

def KISS_coupled_neighbour_map(state_couple: tp.Tuple[_Tsubstate, _Tsubstate],single_neighbour_function: tp.Callable[[_Tsubstate], tp.Dict[_Tsubstate, float]]) -> tp.Dict[tp.Tuple[_Tsubstate, _Tsubstate], float]:
    # KISS: keep it simple and stupid
    s1, s2 = state_couple
    n1_dict = single_neighbour_function(s1)
    if s1 == s2:
        return {(s, s): v for s, v in n1_dict.items()}
    n2_dict = single_neighbour_function(s2)
    res = {}
    for s1_prime, v in n1_dict.items():
        res[(s1_prime, s2)] = v
    for s2_prime, v in n2_dict.items():
        res[(s1, s2_prime)] = v
    return res