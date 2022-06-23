import typing as tp

"""
Idealised Lotka-Volterra model. Mainly used for testing efficiency of couplers.
"""

def replace(s: tp.Tuple[int, ...], idx: int, new: int) -> tp.Tuple[int, ...]:
    s = [_ for _ in s]
    s[idx] = new
    return tuple(s)

def idealised_single_state_dynamic(state: tp.Tuple[int, ...], tau):
    return {replace(state, i, state[i] + sign): tau for i in range(len(state)) for sign in [-1, 1]}

def idealised_double_state_dynamic_1D(s1: int, s2: int, tau: float):
    if s1 == s2:
        return {(s1 + 1, s2 + 1): tau, (s1 - 1, s2 - 1): tau}
    elif abs(s1 - s2) == 1:
        return {(s1, s1): tau, (s2, s2): tau, (2 * s1 - s2, 2 * s2 - s1): tau}
    else:
        return {(s1 - 1, s2 + 1): tau, (s1 + 1, s2 - 1): tau}

def idealised_double_state_dynamic(state_couple: tp.Tuple[tp.Tuple[int, ...], tp.Tuple[int, ...]], tau: float):
    s1, s2 = state_couple
    assert len(s1) == len(s2)
    res = {}
    for i in range(len(s1)):
        onedim_dynamic = idealised_double_state_dynamic_1D(s1[i], s2[i], tau)
        for (s1i_new, s2i_new), rate in onedim_dynamic.items():
            res[replace(s1, i, s1i_new), replace(s2, i, s2i_new)] = rate
    return res