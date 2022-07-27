


import pickle
import numpy as np
import matplotlib.pyplot as plt

import context
import pwdc
import misc
import optm_ddc2
import obstacle as obs
import opty.utils 

def costJ(obss, Z, l, w1, w2, n, m):
  """Minimize the sum of the squares of the control torque."""
  X,U,_ = opty.utils.parse_free(Z, n, m, l)
  xy_tj = X[0:2,:].T
  J1 = w1*np.sum(U**2) # control cost
  J2 = w2*obss.arrayCost(xy_tj) # potential field cost
  # there is no cost term for staying close to the path.
  return J1 + J2

def run_naive_init(configs, num_nodes, save_path, max_iter):

  ### generate map and potential field
  map_grid = misc.LoadMapDao(configs["map_grid_path"])
  obsts_all = misc.findObstacles(map_grid)
  grid_size,_ = map_grid.shape
  obsts = obsts_all / grid_size
  obss = obs.ObstSet( obsts )
  pf = obss.potentialField(1,1,configs["npix"])*100
  
  n = configs["n"]
  m = configs["m"]

  initial_guess = misc.linearInitGuess(configs["Sinit"][0:2], configs["Sgoal"][0:2], \
    num_nodes, n, m, configs["interval_value"])
  
  print('initial_guess',initial_guess)

  configs["optm_weights"][2] = 0 # no need to stay close to the initial guess.
  Zsol, info = optm_ddc2.dirCol_ddc2(\
    initial_guess, configs["Sinit"], configs["Sgoal"], \
    configs["optm_weights"], obss, num_nodes, \
    configs["interval_value"], configs["vu_bounds"], max_iter)

  Xsol, Usol, _ = opty.utils.parse_free(Zsol, n, m, num_nodes)

  ### Figure

  fig = plt.figure(figsize=(4,4))
  plt.xticks([0,1])
  plt.yticks([0,1])
  plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b--")
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(configs["Sinit"][0],configs["Sinit"][1],"ro")
  plt.plot(configs["Sgoal"][0],configs["Sgoal"][1],"r*")
  plt.plot(Xsol[0,:],Xsol[1,:],"r.", markersize=2)

  # 2nd, random initial guess
  np.random.seed(0)
  initial_guess = np.random.randn( num_nodes*(n+m) )
  Zsol2, info = optm_ddc2.dirCol_ddc2(\
    initial_guess, configs["Sinit"], configs["Sgoal"], \
    configs["optm_weights"], obss, num_nodes, \
    configs["interval_value"], configs["vu_bounds"], max_iter)
  Xsol2, Usol2, _ = opty.utils.parse_free(Zsol2, n, m, num_nodes)
  plt.plot(Xsol2[0,:],Xsol2[1,:],"y.", markersize=2)

  ### print out cost value
  J1 = costJ(obss, Zsol, num_nodes, configs["optm_weights"][0], configs["optm_weights"][1], n, m)
  J2 = costJ(obss, Zsol2, num_nodes, configs["optm_weights"][0], configs["optm_weights"][1], n, m)
  print("linear init guess, Jcost = ", J1 )
  print("random init guess, Jcost = ", J2 )

  ###
  plt.draw()
  plt.pause(1)
  plt.savefig(save_path, bbox_inches='tight', dpi=200)

  res_dict = dict()
  res_dict["num_nodes"] = num_nodes
  res_dict["lin_sol"] = Zsol
  res_dict["lin_sol_cost"] = J1
  res_dict["rdm_sol"] = Zsol2
  res_dict["rdm_sol_cost"] = J2
  res_dict["max_iter"] = max_iter
  res_dict["pf"] = pf
  misc.SavePickle(res_dict, configs["folder"]+"naive_res.pickle")

  return
