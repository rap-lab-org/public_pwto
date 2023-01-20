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

		elif mapname == 'random32_2':
			self.configs = self.getConfigRandom32_10()

		elif mapname == "room32_1":
			self.configs = self.getConfigRoom32()

		elif mapname == 'random64_1':
			self.configs = self.getConfigRandom64()

		elif mapname == 'random64_2':
			self.configs = self.getConfigRandom64_10()

		elif mapname == 'random16_1':
			self.configs = self.getConfigRandom16()

		elif mapname == 'random16_simple':
			self.configs = self.getConfigRandom16Simple()

		elif mapname == 'paris_64':
			self.configs = self.getConfigParis64()


		elif mapname == 'random32A':
			self.configs = self.getConfigRandom32A()
		elif mapname == 'random32B':
			self.configs = self.getConfigRandom32B()
		elif mapname == 'random32C':
			self.configs = self.getConfigRandom32C()
		elif mapname == 'random32D':
			self.configs = self.getConfigRandom32D()
		elif mapname == 'random32E':
			self.configs = self.getConfigRandom32E()
		elif mapname == 'random32F':
			self.configs = self.getConfigRandom32F()
		elif mapname == 'random32G':
			self.configs = self.getConfigRandom32G()
		elif mapname == 'random32H':
			self.configs = self.getConfigRandom32H()

		elif mapname == 'random32I':
			self.configs = self.getConfigRandom32I()

		elif mapname == 'random32J':
			self.configs = self.getConfigRandom32J()

		elif mapname == 'random32AA':
			self.configs = self.getConfigRandom32AA()


		elif mapname == 'random32K_1':
			self.configs = self.getConfigRandom32K_1()

		elif mapname == 'random32K_2':
			self.configs = self.getConfigRandom32K_2()

		elif mapname == 'random32K_3':
			self.configs = self.getConfigRandom32K_3()

		elif mapname == 'random32K_4':
			self.configs = self.getConfigRandom32K_4()

		elif mapname == 'random32K_5':
			self.configs = self.getConfigRandom32K_5()

		elif mapname == 'random32K_6':
			self.configs = self.getConfigRandom32K_6()

		elif mapname == 'random32L_1':
			self.configs = self.getConfigRandom32L_1()

		elif mapname == 'random32L_2':
			self.configs = self.getConfigRandom32L_2()

		elif mapname == 'random32L_3':
			self.configs = self.getConfigRandom32L_3()

		elif mapname == 'random32L_4':
			self.configs = self.getConfigRandom32L_4()

		elif mapname == 'random32L_5':
			self.configs = self.getConfigRandom32L_5()

		elif mapname == 'random32L_6':
			self.configs = self.getConfigRandom32L_6()

		elif mapname == 'random32L_7':
			self.configs = self.getConfigRandom32L_7()

		elif mapname == 'random32L_8':
			self.configs = self.getConfigRandom32L_8()
		elif mapname == 'random32L_9':
			self.configs = self.getConfigRandom32L_9()
		elif mapname == 'random32L_10':
			self.configs = self.getConfigRandom32L_10()


		elif mapname == 'random32_trrt':
			self.configs = self.getConfigRandom32TRRT()

		elif mapname == 'random32_trrt_small':
			self.configs = self.getConfigRandom32TRRT_small()

		elif mapname == 'random16_trrt':
			self.configs = self.getConfigRandom16TRRT()


		elif mapname == 'random32A_trrt':
			self.configs = self.getConfigRandom32A_trrt()
		elif mapname == 'random32B_trrt':
			self.configs = self.getConfigRandom32B_trrt()
		elif mapname == 'random32C_trrt':
			self.configs = self.getConfigRandom32C_trrt()

		if self.configs is None:
		    print("[ERROR] PWDC, please input the right mapname")


		# return self.configs
	
	def getConfigRandom32(self):
		"""
		random 32x32, obstacle density 20.
		"""

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


	def getConfigRandom32_10(self):
		"""
		random 32x32, obstacle density 10.
		"""

		folder = "results/res_random32_2/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-32-32-10.map"
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

	def getConfigRoom32(self):
		"""
		random 32x32, obstacle density 20.
		"""

		folder = "results/res_room32_1/"
		mapscale = 20
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "room-32-32-4.map"
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

		folder = "results/res_random16_1/"
		mapscale = 7 #6.2
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



	def getConfigRandom64(self):
		folder = "results/res_random64_1/"
		# mapscale = 24.8
		mapscale = 23
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-64-64-20.map"
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

	def getConfigRandom64_10(self):
		folder = "results/res_random64_2/"
		mapscale = 24.8
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-64-64-10.map"
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

	def getConfigParis64(self):
		folder = "results/res_paris64_1/"
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




	def getConfigRandom32A(self):
		"""
		self-generated random 32x32 A.
		"""

		folder = "results/res_random32A/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		# configs["optm_weights"] = [0.01, 5000, 10] # [0.01, 5000, 200]
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32B(self):
		"""
		self-generated random 32x32 B.
		"""

		folder = "results/res_random32B/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32C(self):
		"""
		self-generated random 32x32 B.
		"""

		folder = "results/res_random32C/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32D(self):
		"""
		self-generated random 32x32 D.
		"""

		folder = "results/res_random32D/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32E(self):
		"""
		self-generated random 32x32 E. The change is smaller cov and fewer obstacle number.
		"""

		folder = "results/res_random32E/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		configs["obst_cov_val"] = 5*1e-4
		# configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


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


	def getConfigRandom32F(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32F/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 150 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		configs["obst_cov_val"] = 5*1e-4
		# configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


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


	def getConfigRandom32G(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32G/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 150 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		configs["obst_cov_val"] = 5*1e-4
		# configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


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



	def getConfigRandom32H(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32H/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		configs["obst_cov_val"] = 5*1e-4
		# configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([2/mapscale, 7.33, 4.4/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32I(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32I/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		configs["obst_cov_val"] = 5*1e-4
		# configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([0.8/mapscale, 7.33, 1.76/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs

	def getConfigRandom32J(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32J/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([0.8/mapscale, 7.33, 1.76/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs





	def getConfigRandom32AA(self):
		"""
		self-generated random 32x32 F. The change is smaller cov and smaller npix.
		"""

		folder = "results/res_random32AA/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([0.6/mapscale, 3.14, 1.32/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32K_1(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_1/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 100]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([0.8/mapscale, 3.14, 1.76/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32K_2(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_2/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 100]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([0.8/mapscale, 3.14, 1.76/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32K_3(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_3/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 100]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 3.14, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32K_4(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_4/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 3.14, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32K_5(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_5/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 3.14, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32K_6(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32K_6/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 3.14, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		# weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs




	def getConfigRandom32L_1(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_1/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 2, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs




	def getConfigRandom32L_2(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_2/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 2, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32L_3(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_3/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 2, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs




	def getConfigRandom32L_4(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_4/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32L_5(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_5/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32L_6(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_6/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32L_7(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_7/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 2]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32L_8(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_8/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32L_9(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_9/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32L_10(self):

		"""
		Use smaller v and w
		"""

		folder = "results/res_random32L_10/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.05, 1000, 50]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8

		# configs["obst_cov_val"] = 1*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4


		configs["vu_bounds"] = np.array([1/mapscale, 1.57, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32TRRT(self):
		"""
		random 32x32, obstacle density 20.
		"""

		folder = "results/res_random32_1_trrt/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.9, 0 ,0, 0])
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
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32TRRT_small(self):
		"""
		random 32x32, obstacle density 20.
		"""

		folder = "results/res_random32_1_trrt/"
		mapscale = 6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.9, 0 ,0, 0])
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
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom16TRRT(self):
		"""
		random 16x16, obstacle density 20.
		"""

		folder = "results/res_random16_trrt/"
		mapscale = 6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		configs["map_grid_path"] = configs["folder"] + "random-16-16-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.9, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		configs["optm_weights"] = [0.01, 5000, 200]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 15
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = 12*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		configs["Astar_weight_list"] = weight_list

		return configs



	def getConfigRandom32A_trrt(self):
		"""
		self-generated random 32x32 A.
		"""

		folder = "results/res_random32A_trrt/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		# configs["optm_weights"] = [0.01, 5000, 10] # [0.01, 5000, 200]
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
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


	def getConfigRandom32B_trrt(self):
		"""
		self-generated random 32x32 B.
		"""

		folder = "results/res_random32B_trrt/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		# configs["optm_weights"] = [0.01, 5000, 10] # [0.01, 5000, 200]
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs


	def getConfigRandom32C_trrt(self):
		"""
		self-generated random 32x32 C trrt.
		"""

		folder = "results/res_random32C_trrt/"
		mapscale = 18.6
		dt = 0.1

		configs = dict()
		configs["folder"] = folder
		# configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
		configs["n"] = 5
		configs["m"] = 2
		configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
		configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
		configs["interval_value"] = dt
		configs["npix"] = 200 #int(mapscale/dt)+20
		configs["emoa_path"] = "../public_emoa/build/run_emoa"
		configs["iters_per_episode"] = 100
		# configs["optm_weights"] = [0.01, 5000, 10] # [0.01, 5000, 200]
		configs["optm_weights"] = [0.05, 1000, 10]
		# w1 = 0.01 # control cost, for the u terms.
		# w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
		# w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
		configs["total_epi"] = 10
		configs["hausdorf_filter_thres"] = 8
		# configs["obst_cov_val"] = 2*1e-4
		configs["obst_cov_val"] = obs.FIX_COV_VAL #2*1e-4
		configs["vu_bounds"] = np.array([1/mapscale, 7.33, 2.2/mapscale, 1]) # v,w,ua,uw
		# Use 0.8 m/s^2 as the max acceleration is reasonable 
		
		weight_list = list()
		weight_list.append([0.5,0.5])
		# weight_list.append([0.3,0.7])
		# weight_list.append([0.2,0.8])
		# weight_list.append([0.9,0.1])
		# weight_list.append([0.7,0.3])
		configs["Astar_weight_list"] = weight_list

		return configs
