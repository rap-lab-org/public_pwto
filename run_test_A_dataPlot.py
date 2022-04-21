


import numpy as np
import matplotlib.pyplot as plt


iters = np.array([285, 313, 264, 355, 474, 500, 435, 304, 327, 496, 382, 500, 386])
emoa_sch_time = 0.153 # seconds.
obj_val = np.array([10763688, 2805368, 2854991, 2658071, 2393952, 2204654, 2764473, 3296353, 2590231, 2390316, 2069354, 2536165, 5527384])

obj_val_percent = obj_val / np.min(obj_val)

fig = plt.figure(figsize=(3,2))
plt.plot(range(len(iters)), obj_val_percent, "r^")
plt.show()

