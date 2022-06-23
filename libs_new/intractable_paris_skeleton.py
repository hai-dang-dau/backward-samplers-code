import typing as tp

from libs_new.smoothing_high_level import PaRIS, PaRIS_Output
from libs_new.utils_math import CategoricalDistribution

_Tt = tp.TypeVar('_Tt')
_Ttm1 = tp.TypeVar('_Ttm1')

class iPaRIS_Input(tp.NamedTuple):
    X: tp.Sequence[_Tt]
    W: CategoricalDistribution
    psi: tp.Union[tp.Callable[[_Tt], float], tp.Callable[[_Ttm1, _Tt], float]]
    backward_idx: tp.Sequence[tp.Sequence[int]]

class IntractablePaRIS(PaRIS):
    # tested two ways differently, using version 080921
    def send(self, val: tp.Optional[iPaRIS_Input]) -> tp.Optional[PaRIS_Output]:
        if val is None:
            return None
        valX, valW, valpsi = val.X, val.W, val.psi
        if self.t == 0:
            prodd = self._process0(valW, valX, valpsi)
            return PaRIS_Output(prodd, None)
        prodd = self._process_t(val.backward_idx, valW, valX, valpsi)
        return PaRIS_Output(prodd, None)