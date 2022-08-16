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

import getConfig


##########################################
### Global params BEGIN ###
##########################################

mapname = 'random32C'
ConfigClass = getConfig.Config(mapname)
InstanceIndex = 5 # take values from [0,9]

##########################################
### Global params END ###
##########################################


def PlotPwdc(configs, start_state, goal_state, idx):
  """
  A function to be called by main_plot_pwdc.
  """

  configs["Sinit"] = start_state
  configs["Sgoal"] = goal_state
  res_file_path = configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle"
  
  solver = misc.LoadPickle(res_file_path)
  # configs = solver.cfg
  fig_sz,_ = solver.map_grid.shape
  fig_sz = 4 # fixed figure size, (or adaptive? based on grid size)
  res = solver.sol
  print(" get ", len(solver.emoa_res_dict), " Pareto paths" )
  print(" npix = ", configs["npix"] )

  # traj plot
  fig = plt.figure(figsize=(fig_sz,fig_sz))
  s = configs["Sinit"]
  d = configs["Sgoal"]
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  pf = solver.obs_pf
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(s[0],s[1],"ro")
  plt.plot(d[0],d[1],"r*")

  for k in res:
    px,py = solver._path2xy(res[k].init_path)
    plt.plot(px,py,"b--", alpha=0.6)

    tj = np.array([res[k].getPosX(),res[k].getPosY()])
    # plt.plot(tj[0,:], tj[1,:], "r.", markersize=1.0, alpha=0.6) # random 32x32
    plt.plot(tj[0,:], tj[1,:], "r.", markersize=1.5, alpha=0.6) # random 16x16

    print("k = ", k, "converge episode = ", res[k].epiIdxCvg, " costJ = ", res[k].J, " traj L = ", res[k].l)
  
  plt.xticks([0,1])
  plt.yticks([0,1])
  # plt.axis('off')
  plt.draw()
  plt.pause(1)
  save_path = configs["folder"]+"instance_"+str(idx)+"_solTrajs.png"
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)

  return


def main_plot_pwdc():
  """
  Entry function to plot PWTO results.
  """
  print("--- main_plot_pwdc ---")
  configs = ConfigClass.configs
  print("configs[npix] = ", configs["npix"])
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  # ii = 1
  for ii in range(InstanceIndex,InstanceIndex+1):
    PlotPwdc(configs, start_states[ii,:], goal_states[ii,:], ii)

def RunPwdcOnce(configs, start_state, goal_state, idx, obs_pf = []):
  """
  A procedure called by main_test_pwdc.
  """
  # override the default start and goal with the input ones.
  configs["Sinit"] = start_state
  configs["Sgoal"] = goal_state
  solver = pwdc.PWDC(configs, obs_pf) # pass in obs_pf to avoid repeatitive computation.
  solver.Solve()
  save_file = configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle"
  print("PWDC get", len(solver.sol), " solutions, save to ", save_file)
  misc.SavePickle(solver, save_file)
  return

def main_test_pwdc():
  """
  The entry function where PWTO (In the code, PWDC=PWTO) is runned.
  """
  configs = ConfigClass.configs
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  for ii in range(InstanceIndex,InstanceIndex+1):
    RunPwdcOnce(configs, start_states[ii,:], goal_states[ii,:], ii, obs_pf)
  return

#############


if __name__ == "__main__":

  main_test_pwdc()
  main_plot_pwdc()