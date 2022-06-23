import typing as tp
import numpy as np
from abc import ABC, abstractmethod

from libs_new.coupling_gaussians import Lindvall_Rogers_coupling
from libs_new.utils import add_dummy_first_argument
from programs.sde.model import SmoothableSDE


class MeetingInfo(tp.NamedTuple):
    arr1: tp.Sequence[float]
    arr2: tp.Sequence[float]
    meet_idx : int

def meeting_info(arr1: tp.Sequence[float], arr2: tp.Sequence[float]) -> MeetingInfo:
    # tested
    diff = np.array(arr1) - np.array(arr2)
    equal = [np.allclose(d, 0) for d in diff]
    sign_changed = [np.sign(diff[max(i-1, 0)]) != np.sign(diff[i]) for i in range(len(diff))]
    meet = np.logical_or(equal, sign_changed)
    meet_idx = list(meet).index(True)
    arr1, arr2 = list(arr1), list(arr2)
    newval = (arr1[meet_idx] + arr2[meet_idx])/2
    arr1[meet_idx] = newval
    arr2[meet_idx] = newval
    return MeetingInfo(arr1=arr1, arr2=arr2, meet_idx=meet_idx)

def reflection_coupling_brownian(a: float, b: float, delta: float, T: float) -> tp.Tuple['DiscretisedTrajectory', 'DiscretisedTrajectory']:
    # tested
    """
    Simulate two reflection-coupled brownian motions
    :param a: starting point for first motion
    :param b: starting point for second motion
    :param delta: discretisation size
    :param T: max simulation time
    """
    res0 = [a]
    res1 = [b]
    for _ in range(int(T/delta)):
        dw = np.random.normal(scale=delta**0.5)
        res0.append(res0[-1] + dw)
        res1.append(res1[-1] - dw)
    # return res0, res1
    return DiscretisedTrajectory(x_array=res0, delta=delta), DiscretisedTrajectory(x_array=res1, delta=delta)

class DiscretisedTrajectory:
    def __init__(self, x_array: tp.Sequence[float], delta: float, t_start: float = 0):
        self._x_array = x_array
        self._t_array = [t_start + k * delta for k in range(len(x_array))]
        self._delta = delta
        self._t_start = t_start

    @property
    def x_array(self):
        return self._x_array

    @property
    def t_array(self):
        return self._t_array

    @property
    def t_start(self):
        return self._t_start

    @property
    def N_discretisation(self):
        return len(self._t_array)

    @x_array.setter
    def x_array(self, value):
        if len(value) != len(self._x_array):
            raise ValueError
        self._x_array = value

    @property
    def delta(self):
        return self._delta

    def cut(self, Tcut: float):
        # tested
        cut_xarray = [x for x, t in zip(self._x_array, self._t_array) if t <= Tcut]
        return self.__class__(x_array=cut_xarray, delta=self._delta, t_start=self._t_start)

    def thin(self, thin_number: int):
        # tested
        return self.__class__(x_array=self._x_array[::thin_number], delta=self._delta * thin_number, t_start=self._t_start)

class Meeting_time_simulator(ABC):
    # tested
    def __init__(self, a: float, b: float, delta: float, n_debug: int, mode: tp.Literal['min', 'max']):
        """
        if mode is `min`, will simulate the two motions until min(meeting time, n_debug)
        """
        self.a = a
        self.b = b
        self.delta = delta
        self.n_debug = n_debug
        self.mode = mode

        self.m_a = [a]
        self.m_b = [b]
        self.meeting_idx: tp.Optional[int] = None
        self._run()

    @staticmethod
    @abstractmethod
    def _next(x: float, y: float, delta: float) -> tp.Tuple[float, float]:
        ...

    @staticmethod
    @abstractmethod
    def _last_met(x: tp.Sequence[float], y: tp.Sequence[float]) -> bool:
        ...

    def _done(self) -> bool:
        met = self.meeting_idx is not None
        long_enough = (len(self.m_a) >= (self.n_debug + 1))
        if self.mode == 'min':
            return met or long_enough
        elif self.mode == 'max':
            return met and long_enough
        else:
            raise ValueError

    def _run(self):
        while not self._done():
            self._run_once()

    def _run_once(self):
        newx, newy = self._next(self.m_a[-1], self.m_b[-1], self.delta)
        self.m_a.append(newx)
        self.m_b.append(newy)
        if (self.meeting_idx is None) and self._last_met(self.m_a, self.m_b):
            self.meeting_idx = len(self.m_a) - 1

def last_met_1d(x: tp.Sequence[float], y: tp.Sequence[float]) -> bool:
    # tested
    assert len(x) >= 2 and len(y) >= 2
    sign = np.sign(x[-1] - y[-1])
    prev_sign = np.sign(x[-2] - y[-2])
    return sign != prev_sign

class SDEMeeting(Meeting_time_simulator, ABC):
    def __init__(self, f: tp.Callable, sigma: tp.Callable, a, b, delta, n_debug, mode: tp.Literal['min', 'max']):
        self.sde_f = add_dummy_first_argument(f)
        self.sde_sigma = add_dummy_first_argument(sigma)
        super().__init__(a=a, b=b, delta=delta, n_debug=n_debug, mode=mode)

class HybridLindvallMeeting(SDEMeeting):
    @staticmethod
    def _last_met(x: tp.Sequence[float], y: tp.Sequence[float]) -> bool:
        return x[-1] is y[-1]

    def _next(self, x: float, y: float, delta: float) -> tp.Tuple[float, float]:
        return SmoothableSDE.coupled_M_single_step(eps=delta, f=self.sde_f, is2d=False, sigma=self.sde_sigma, t=None, x1=x, x2=y)

class PureLindvallMeeting(SDEMeeting):
    @staticmethod
    def _last_met(x: tp.Sequence[float], y: tp.Sequence[float]) -> bool:
        assert len(x) >= 2 and len(y) >= 2
        n, d = np.array(x).shape
        if d == 1:
            return last_met_1d(np.ravel(x[-2:]), np.ravel(y[-2:]))
        # noinspection PyTypeChecker
        return intersection(segment1=[x[-1], x[-2]], segment2=[y[-1], y[-2]]) is not None

    def _next(self, x: float, y: float, delta: float) -> tp.Tuple[float, float]:
        return SmoothableSDE.coupled_M_single_step(eps=delta, f=self.sde_f, is2d=False, sigma=self.sde_sigma, t=None, x1=x, x2=y, func=self.pure_coupling)

    @staticmethod
    def pure_coupling(dist1, dist2, two_d):
        assert not two_d
        return Lindvall_Rogers_coupling(dist1=dist1, dist2=dist2)

class ClosestResult(tp.NamedTuple):
    point1: np.ndarray
    point2: np.ndarray
    param1: float
    param2: float

def closest(line1: tp.Tuple[np.ndarray, np.ndarray], line2: tp.Tuple[np.ndarray, np.ndarray]) -> ClosestResult:
    """
    Find two points lying on two given lines which minimise their distance. Returns the two points as well as their location parameters. (Given two points, any other point on the line forming by them has a location parameter.)
    """
    vec1 = line1[0] - line1[1]
    vec2 = line2[1] - line2[0]
    delta = line2[1] - line1[1]
    gram = np.array([[np.dot(vec1, vec1), np.dot(vec1, vec2)],
                     [np.dot(vec1, vec2), np.dot(vec2, vec2)]])
    expected = np.array([np.dot(delta, vec1), np.dot(delta, vec2)])
    param1, param2 = np.linalg.solve(gram, expected)
    point1 = param1 * line1[0] + (1 - param1) * line1[1]
    point2 = param2 * line2[0] + (1 - param2) * line2[1]
    return ClosestResult(point1=point1, point2=point2, param1=param1, param2=param2)

def intersection(segment1: tp.Tuple[np.ndarray, np.ndarray], segment2: tp.Tuple[np.ndarray, np.ndarray]) -> tp.Optional[np.ndarray]:
    # tested
    pre_int = closest(line1=segment1, line2=segment2)
    if not (0 <= pre_int.param1 <= 1):
        return None
    if not (0 <= pre_int.param2 <= 1):
        return None
    if not np.allclose(pre_int.point1, pre_int.point2):
        return None
    return pre_int.point1

def four_points_cond_number(a, b, c, d):
    # tested
    """
    Check whether four points are on the same plane by calculating the condition number of the difference matrix. If the returned value is high, it's likely that they are.
    """
    if len(a) <= 2:
        raise ValueError
    return np.linalg.cond([b-a, c-a, d-a])