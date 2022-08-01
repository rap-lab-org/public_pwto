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
import naive_init_baseline as naive
import kAstar_init_baseline as kAstar


# From config code provided by Husarion, 
# https://github.com/husarion/rosbot_description/blob/master/src/rosbot_navigation/config/trajectory_planner.yaml
# The max acceleration in either x or y direction can reache 1.5 m/s^2.
# Although the max vel in this doc is 0.2m/s
# We use the value provided by their doc for actual robot: https://husarion.com/manuals/rosbot/
# - Maximum translational velocity	1.0 m/s
# - Maximum rotational velocity	420 deg/s (7.33 rad/s)

class Config():
	"""
	Get configs for different maps
	"""
	def __init__(self,mapname):

		self.configs = None

		if mapname == 'random32_1':
			self.configs = self.getConfigRandom32()

		elif mapname == 'random16_1':
			self.configs = self.getConfigRandom16()

		elif mapname == 'random16_simple':
			self.configs = self.getConfigRandom16Simple()

		elif mapname == 'paris_64':
			self.configs = self.getConfigParis64()

		if self.configs is None:
		    print("[ERROR] PWDC, please input the right mapname")


		# return self.configs
	


	def getConfigRandom32(self):

		folder = "results/res_random32_1/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.01, 5000, 200]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = 2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list


		return configs


	def getConfigRandom16(self):

		folder = "results/res_random16_1_diffcov/"
		mapscale = 6.2
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-16-16-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.05, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt

		# max_v = 1/mapscale, here 1 is the actual velocity limit 1.0m/s
		# step dis = max_v * dt
		# npix (length dis) = 1/ step_dis, here 1 is the normalized v

		configs["npix"] = int(mapscale/dt) + 20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.01, 5000, 200]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = 7*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 

		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs




	def getConfigRandom16Simple(self):
		folder = "results/res_random16_simple/"
		mapscale = 6.2
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-16-16-simple.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.92, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = int(mapscale/dt) 

		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.01, 5000, 200]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = 7*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 

		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list


		return configs



	def getConfigParis64(self):
		folder = "results/res_paris64/"
		mapscale = 24.8
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "Paris_64.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = int(mapscale/dt)+40
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.01, 5000, 200]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = 2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 0.5/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 

		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list


		return configs