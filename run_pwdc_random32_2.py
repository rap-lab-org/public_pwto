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

folder = "data/test_random32_2/"

def getConfig():

  configs = dict()
  configs["folder"] = folder
  configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
  configs["n"] = 5
  configs["m"] = 2
  configs["Sinit"] = np.array([0.7, 0.1, 0, 0, 0])
  configs["Sgoal"] = np.array([0.5, 0.85, 0 ,0, 0])
  configs["interval_value"] = 0.1
  configs["npix"] = 100
  configs["emoa_path"] = "../public_emoa/build/run_emoa"
  configs["iters_per_episode"] = 100
  configs["optm_weights"] = [0.01, 5000, 200]
    # w1 = 0.01 # control cost, for the u terms.
    # w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
    # w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
  configs["total_epi"] = 10
  configs["hausdorf_filter_thres"] = 10
  configs["obst_cov_val"] = 2*1e-4
  return configs

def test_pwdc():
  """
  """
  configs = getConfig()
  solver = pwdc.PWDC(configs)
  solver.Solve()
  print("PWDC get", len(solver.sol), " solutions.")
  misc.SavePickle(solver, folder+"result.pickle")
  return

def test_pwdc_plot():
  """
  """
  solver = misc.LoadPickle(folder+"result.pickle")
  # solver = loaded[1]
  configs = solver.cfg
  fig_sz,_ = solver.map_grid.shape
  fig_sz = 4
  res = solver.sol
  print(" get ", len(solver.emoa_res_dict), " Pareto paths" )
  for k in res:
    px,py = solver._path2xy(res[k].init_path)
    pwdc.plotTraj(solver.obs_pf, configs, \
                  np.array([px,py]), np.array([res[k].getPosX(),res[k].getPosY()]), \
                  folder+"solTraj_"+str(k)+".png", fig_sz)
    print("k = ", k, "converge episode = ", res[k].epiIdxCvg, " costJ = ", res[k].J, " traj L = ", res[k].l)
  return

def test_and_plot_naive_init():
  configs = getConfig()
  num_nodes = 120
  save_path = folder + "naiveTraj.png"
  max_iter = 1000
  naive.run_naive_init(folder, configs, num_nodes, save_path, max_iter)

# linear init guess, Jcost =  26605788.822472937
# random init guess, Jcost =  35461459.788119234


def plot_compare_iters():
  solver = misc.LoadPickle(folder+"result.pickle")
  naive_res_dict = misc.LoadPickle(folder+"naive_res.pickle")
  res = solver.sol
  plt.figure(figsize=(3,2))
  epiIdxList = list()
  JcostList = list()
  trajLenList = list()
  for k in res:
    epiIdxList.append(res[k].epiIdxCvg)
    JcostList.append(res[k].J)
    trajLenList.append(res[k].l)
  epiIdxList = np.array(epiIdxList)
  JcostList = np.array(JcostList)
  trajLenList = np.array(trajLenList)
  plt.plot(epiIdxList+1, JcostList/np.min(JcostList), "g.")
  # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")
  lin_sol_cost_ratio = naive_res_dict["lin_sol_cost"]/np.min(JcostList)
  plt.plot([1,10],[lin_sol_cost_ratio,lin_sol_cost_ratio],'r--',lw=1)
  rdm_sol_cost_ratio = naive_res_dict["rdm_sol_cost"]/np.min(JcostList)
  plt.plot([1,10],[rdm_sol_cost_ratio,rdm_sol_cost_ratio],'c--',lw=1)
  plt.grid("on")
  plt.draw()
  plt.pause(2)
  plt.savefig(folder+"iter_compare.png", bbox_inches='tight', dpi=200)
  return

def plot_pareto_front():

  solver = misc.LoadPickle(folder+"result.pickle")
  res = solver.emoa_res_dict
  # print(res)
  plt.figure(figsize=(3,3))
  cost_array = np.array( list(res['costs'].values()) )
  
  print("total EMOA* #sol = ", len(cost_array))
  for k in res["costs"]:
    c = res['costs'][k]
    plt.plot(c[0],c[1]/np.min(cost_array),"g.", alpha=0.5)

  print("total PWDC converged #sol = ", len(solver.sol))
  for k in solver.sol:
    c = res['costs'][k]
    plt.plot(c[0],c[1]/np.min(cost_array),"ro", alpha=0.6)
  
  print("total PWDC un-converged #sol = ", len(solver.tjObjDict))
  for k in solver.tjObjDict:
    c = res['costs'][k]
    plt.plot(c[0],c[1]/np.min(cost_array),"bo", alpha=0.6)
  plt.grid("on")
  plt.draw()
  plt.pause(2)
  plt.savefig(folder+"pareto_front.png", bbox_inches='tight', dpi=200)
  return
  
def plot_pareto_paths():
  """
  """
  solver = misc.LoadPickle(folder+"result.pickle")

  plt.figure(figsize=(4,4))
  
  configs = solver.cfg
  pf = solver.obs_pf
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(configs["Sinit"][0],configs["Sinit"][1],"ro")
  plt.plot(configs["Sgoal"][0],configs["Sgoal"][1],"r*")

  res = solver.emoa_res_dict
  npix = solver.cfg["npix"]
  for k in res["paths"]:
    p = res['paths'][k]
    px = list()
    py = list()
    for v in p:
        py.append( (v%npix)*(1/npix) )
        px.append( int(np.floor(v/npix))*(1.0/npix) )
    plt.plot(px,py,"g")

  for k in solver.sol:
    p = res['paths'][k]
    px = list()
    py = list()
    for v in p:
        py.append( (v%npix)*(1/npix) )
        px.append( int(np.floor(v/npix))*(1.0/npix) )
    plt.plot(px,py,"r--")

  for k in solver.tjObjDict:
    p = res['paths'][k]
    px = list()
    py = list()
    for v in p:
        py.append( (v%npix)*(1/npix) )
        px.append( int(np.floor(v/npix))*(1.0/npix) )
    plt.plot(px,py,"b:")
    
  plt.xlim([-0.01,1.01])
  plt.xticks([0,1])
  plt.ylim([-0.01,1.01])
  plt.yticks([0,1])
  plt.draw()
  plt.pause(2)
  plt.savefig(folder+"pareto_paths.png", bbox_inches='tight', dpi=200)
  return


if __name__ == "__main__":

  # test_pwdc()

  # test_pwdc_plot()

  # test_and_plot_naive_init()

  # plot_compare_iters()

  plot_pareto_front()

  # plot_pareto_paths()