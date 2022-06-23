import numpy as np
import matplotlib.pyplot as plt

from programs.sde.illustrations.supporting_funcs import reflection_coupling_brownian, meeting_info

seed = 8
a = 0
b = 4
Tmax = 5
hidden_delta = 1/1000
surface_delta = 1
Tplot = 5

np.random.seed(seed)
fig, ax = plt.subplots()
ax: plt.Axes
fig: plt.Figure

m1, m2 = reflection_coupling_brownian(a=a, b=b, delta=hidden_delta, T=Tmax)
meet = meeting_info(arr1=m1.x_array, arr2=m2.x_array)
m1.x_array = meet.arr1
m2.x_array = meet.arr2
meet_time = meet.meet_idx * hidden_delta
meet_place = meet.arr1[meet.meet_idx]
ax.scatter(meet_time, meet_place, label=r'True meeting point $\tau$', color='black', s=100)

m1s = m1.cut(meet_time + 1e-6)
m2s = m2.cut(meet_time + 1e-6)
ax.plot(m1s.t_array, m1s.x_array, label='True Lindvall-Rogers coupling', color='grey', alpha=0.8)
ax.plot(m2s.t_array, m2s.x_array, color='grey', alpha=0.8)

m1 = m1.cut(Tplot)
m2 = m2.cut(Tplot)
thin_number = int(surface_delta/hidden_delta)
m1 = m1.thin(thin_number)
m2 = m2.thin(thin_number)
ax.plot(m1.t_array, m1.x_array, label='Discretised Lindvall-Rogers coupling', color='black', linestyle='dashed', marker='x', markersize=10)
ax.plot(m2.t_array, m2.x_array, color='black', linestyle='dashed', marker='x', markersize=10)

ax.axhline(y=(a+b)/2, linestyle='dotted', label='Reflection "plane"', color='black')
ax.legend()

ax.set_xlabel('time')

fig.show()