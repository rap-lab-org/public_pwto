"""
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

import context
import pwdc
import misc

import optm_ddc2
import obstacle as obs

# From config code provided by Husarion, 
# https://github.com/husarion/rosbot_description/blob/master/src/rosbot_navigation/config/trajectory_planner.yaml
# We consider the value provided by their doc for actual robot: https://husarion.com/manuals/rosbot/

class Config():
	"""
	Get configs for different maps
	"""
	def __init__(self,mapname):

		self.configs = None
		if mapname == 'random32C':
			self.configs = self.getConfigRandom32C()

		if self.configs is None:
		    print("[ERROR] PWDC, please input the right mapname")


		# return self.configs

	def getConfigRandom32C(self):
		"""
		self-generated random 32x32 B.
		"""

		folder = "results/res_random32C/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		
		configs["n"] = 5 # state size
		configs["m"] = 2 # control size
		
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0]) # will be override by the test instance !
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0]) # will be override by the test instance !
		
		configs["interval_value"] = dt
		configs["npix"] = 200 # fixed grid size.
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.05 # control cost, for the u terms.
		# w2 = 1000 # obstacle cost, larger = stay more far away from obstacles, due to numerical integration, 
		#           # dt is already considered in this term, should have large scale to be similar as the first term.
		# w3 = 50 # stay close to the initial guess, larger = stay closer to the initial guess. Too large may cause more local minima.

		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		configs["obst_cov_val"] = obs.FIX_COV_VAL
		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs
