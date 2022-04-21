


import numpy as np
import matplotlib.pyplot as plt

case_ID = "A"

iters = np.array([285, 264, 355, 474, 500, 304, 435, 327, 313, 496, 500, 382, 386])
emoa_sch_time = 0.153 # seconds.
obj_val = np.array([10763688, 2854991, 2658071, 2393952, 2204654, 3296353, 2764473, 2590231, 2805368, 2390316, 2536165, 2069354, 5527384])

obj_val_percent = obj_val / np.min(obj_val)
iter_percent = iters / np.max(iters)

fig = plt.figure(figsize=(3,2))
plt.plot(range(len(iters)), obj_val_percent, "r^")
# plt.plot(range(len(iters)), iter_percent, "bv")
plt.grid()
plt.draw()
plt.pause(2)
plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-obj_percent.png", bbox_inches='tight', dpi=200)



fig = plt.figure(figsize=(3,2))
# plt.plot(range(len(iters)), obj_val_percent, "r^")
plt.plot(range(len(iters)), iter_percent, "bv")
plt.grid()
plt.draw()
plt.pause(2)
plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-iter_percent.png", bbox_inches='tight', dpi=200)


