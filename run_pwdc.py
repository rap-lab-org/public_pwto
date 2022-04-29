"""
"""

import numpy as np
import context
import pwdc

def test_pwdc():
  """
  """
  configs = dict()
  configs["folder"] = "data/random_instance_A/"
  configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
  configs["n"] = 5
  configs["m"] = 2
  configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
  configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
  configs["interval_value"] = 0.1
  configs["npix"] = 100
  configs["emoa_path"] = "../public_emoa/build/run_emoa"
  configs["iters_per_episode"] = 500
  configs["optm_weights"] = [0.01, 5000, 200]
    # w1 = 0.01 # control cost, for the u terms.
    # w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
    # w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
  configs["total_epi"] = 10
  configs["hausdorf_filter_thres"] = 10

  solver = pwdc.PWDC(configs)
  res = solver.Solve()
  print(res)

if __name__ == "__main__":
  test_pwdc()