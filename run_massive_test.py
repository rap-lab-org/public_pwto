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
import scaleAstar_init_baseline as scaleAstar

import getConfig


##########################################
### Global params BEGIN ###
##########################################

# mapname = 'random32A'
# mapname = 'random32B'
# mapname = 'random32C'
mapname = 'random32D'
ConfigClass = getConfig.Config(mapname)
NInstance = 10

##########################################
### Global params END ###
##########################################


def PlotPwdc(configs, start_state, goal_state, idx):

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


def RunWeightedAstar(configs, start_state, goal_state, idx):
  """
  """
  # override the default start and goal with the input ones.
  configs["Sinit"] = start_state
  configs["Sgoal"] = goal_state
  
  pwdc_fname = configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle"
  solver_pwdc = misc.LoadPickle(pwdc_fname)
  # emoa_res = solver_pwdc.emoa_res_dict

  w1 = 0.5
  w2 = 0.5
  print("weight = ", w1, w2)
  configs["Astar_weight_list"].append([w1,w2])

  astarSol = scaleAstar.AStar(configs, solver_pwdc.obs_pf)
  astarSol.solveScaleAstar()

  res_fname = configs["folder"]+"instance_"+str(idx)+"_wghA_result.pickle"
  misc.SavePickle(astarSol, res_fname)
  return


def RunNaiveInit(configs, start_state, goal_state, idx):
  """
  """
  # override the default start and goal with the input ones.
  configs["Sinit"] = start_state
  configs["Sgoal"] = goal_state

  pwdc_fname = configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle"
  solver_pwdc = misc.LoadPickle(pwdc_fname)
  pwdc_res = solver_pwdc.sol
  trajLenList_pwdc = []
  for k in pwdc_res:
    trajLenList_pwdc.append(pwdc_res[k].l)
  trajLenList_pwdc = np.array(trajLenList_pwdc)

  res_dict = dict()
  if len(trajLenList_pwdc) == 0: # PWDC fails
    res_dict["num_nodes"] = 0
    res_dict["lin_sol"] = []
    res_dict["lin_sol_cost"] = np.nan
    res_dict["rdm_sol"] = []
    res_dict["rdm_sol_cost"] = np.nan
    res_dict["max_iter"] = 0
    # res_dict["pf"] = pf
  else:
    num_nodes = int(np.mean(trajLenList_pwdc))
    max_iter = int(configs["iters_per_episode"]* configs["total_epi"])
    print('num nodes of the navie init',num_nodes, "max_iter = ", max_iter)
    # naive.run_naive_init(folder, configs, num_nodes, save_path, max_iter)
    res_dict = naive.run_naive_init_for_test(configs, num_nodes, max_iter)

  save_path = configs["folder"] + "instance_"+str(idx)+"_dirColInit_result.pickle"
  misc.SavePickle(res_dict, save_path)
  return


def plot_pareto_front():

  configs = ConfigClass.configs
  # TODO: ADD WEIGHTED A STAR 

  solver = misc.LoadPickle(configs["folder"]+"result.pickle")
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
  plt.savefig(configs["folder"]+"pareto_front.png", bbox_inches='tight', dpi=200)
  return


def PlotParetoPaths(configs, start_state, goal_state, idx):
  """
  """

  configs["Sinit"] = start_state
  configs["Sgoal"] = goal_state
  solver = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle")
  plt.figure(figsize=(4,4))
  
  # configs = solver.cfg
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
  save_name = configs["folder"]+"instance_"+str(idx)+"_pareto_paths.png"
  plt.savefig(save_name, bbox_inches='tight', dpi=200)
  return


def main_test_wghA():
  """
    # must run weighted A* after PWDC, since weighted A* needs the #sol in PWDC.
  """
  configs = ConfigClass.configs
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  for ii in range(start_states.shape[0]):
    RunWeightedAstar(configs, start_states[ii,:], goal_states[ii,:], ii)
  return

def main_test_naiveInit():
  configs = ConfigClass.configs
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  for ii in range(start_states.shape[0]):
    RunNaiveInit(configs, start_states[ii,:], goal_states[ii,:], ii)
  return

def main_plot_pwdc():
  print("--- main_plot_pwdc ---")
  configs = ConfigClass.configs
  print("configs[npix] = ", configs["npix"])
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  # ii = 1
  for ii in range(NInstance):
    PlotPwdc(configs, start_states[ii,:], goal_states[ii,:], ii)
    PlotParetoPaths(configs, start_states[ii,:], goal_states[ii,:], ii)


def PlotCvgIters(idx):

  configs = ConfigClass.configs
  pwdc_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle")
  scaleAstar_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_wghA_result.pickle")
  naive_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_dirColInit_result.pickle")

  # kAstar_res_dict = misc.LoadPickle(configs["folder"]+"kAstar_res.pickle")
  pwdc_res = pwdc_res_dict.sol
  plt.figure(figsize=(3,2))
  epiIdxList_pwdc = list()
  JcostList_pwdc = list()
  trajLenList_pwdc = list()
  for k in pwdc_res:
    epiIdxList_pwdc.append(pwdc_res[k].epiIdxCvg)
    JcostList_pwdc.append(pwdc_res[k].J)
    trajLenList_pwdc.append(pwdc_res[k].l)
  epiIdxList_pwdc = np.array(epiIdxList_pwdc)
  JcostList_pwdc = np.array(JcostList_pwdc)
  trajLenList_pwdc = np.array(trajLenList_pwdc)
  if len(epiIdxList_pwdc) > 0:
    plt.plot(epiIdxList_pwdc+1, JcostList_pwdc/np.min(JcostList_pwdc), "ro",alpha=0.5)
  # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")

  scaleAstar_res = scaleAstar_res_dict.sol
  epiIdxList_scaleAstar = list()
  JcostList_scaleAstar = list()
  trajLenList_scaleAstar = list()
  for k in scaleAstar_res:
    epiIdxList_scaleAstar.append(scaleAstar_res[k].epiIdxCvg)
    JcostList_scaleAstar.append(scaleAstar_res[k].J)
    trajLenList_scaleAstar.append(scaleAstar_res[k].l)
  epiIdxList_scaleAstar = np.array(epiIdxList_scaleAstar)
  JcostList_scaleAstar = np.array(JcostList_scaleAstar)
  trajLenList_scaleAstar = np.array(trajLenList_scaleAstar)
  if len(JcostList_pwdc) > 0:
    plt.plot(epiIdxList_scaleAstar + 1, JcostList_scaleAstar/np.min(JcostList_pwdc), "b^",alpha=0.5)
  # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")

  # kAstar_res = kAstar_res_dict.sol
  # epiIdxList_kAstar = list()
  # JcostList_kAstar = list()
  # trajLenList_kAstar = list()
  # for k in kAstar_res:
  #   epiIdxList_kAstar.append(kAstar_res[k].epiIdxCvg)
  #   JcostList_kAstar.append(kAstar_res[k].J)
  #   trajLenList_kAstar.append(kAstar_res[k].l)
  # epiIdxList_kAstar = np.array(epiIdxList_kAstar)
  # JcostList_kAstar = np.array(JcostList_kAstar)
  # trajLenList_kAstar = np.array(trajLenList_kAstar)
  # plt.plot(epiIdxList_kAstar+1, JcostList_kAstar/np.min(JcostList_pwdc), "b.")
  # # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")

  ## DirCol + naive init

  if len(epiIdxList_pwdc) > 0:
    lin_sol_cost_ratio = naive_res_dict["lin_sol_cost"]/np.min(JcostList_pwdc)
    plt.plot([1,10],[lin_sol_cost_ratio,lin_sol_cost_ratio],'k--',lw=2)

    rdm_sol_cost_ratio = naive_res_dict["rdm_sol_cost"]/np.min(JcostList_pwdc)
    plt.plot([1,10],[rdm_sol_cost_ratio,rdm_sol_cost_ratio],'k-',lw=1)

  ### k-best
  # kastar_sol_cost_ratio = kAstar_res_dict["kAstar_sol_cost"]/np.min(JcostList_pwdc)
  # plt.plot([1,10],[kastar_sol_cost_ratio,kastar_sol_cost_ratio],'g--',lw=1)

  plt.grid("on")
  plt.draw()
  plt.pause(2)
  plt.savefig(configs["folder"]+"instance_"+str(idx)+"_iter_compare.png", bbox_inches='tight', dpi=200)
  return

def main_plot_cvg_iters():
  for idx in range(NInstance):
    PlotCvgIters(idx)

def main_compare_cvg_iters():
  """
  """
  configs = ConfigClass.configs
  mat_cvg_data = np.zeros([NInstance,2])
  mat_cost_data = np.zeros([NInstance,4])

  np.set_printoptions(precision=3, suppress=True)
  
  succ_count = 0
  for idx in range(NInstance):
    print("idx = ", idx)
    pwdc_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_pwdc_result.pickle")
    if len(pwdc_res_dict.sol) == 0:
      continue
    else:
      succ_count = succ_count + 1
    scaleAstar_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_wghA_result.pickle")
    naive_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_dirColInit_result.pickle")

    ## pwdc ##
    pwdc_res = pwdc_res_dict.sol
    epiIdxList_pwdc = list()
    JcostList_pwdc = list()
    for k in pwdc_res:
      epiIdxList_pwdc.append(pwdc_res[k].epiIdxCvg)
      JcostList_pwdc.append(pwdc_res[k].J)
    epiIdxList_pwdc = np.array(epiIdxList_pwdc)
    JcostList_pwdc = np.array(JcostList_pwdc)
    mat_cvg_data[idx,0] = np.min(epiIdxList_pwdc)
    mat_cost_data[idx,0] = np.min(JcostList_pwdc)

    ## wA* ##
    scaleAstar_res = scaleAstar_res_dict.sol
    epiIdxList_scaleAstar = list()
    JcostList_scaleAstar = list()
    trajLenList_scaleAstar = list()
    for k in scaleAstar_res:
      epiIdxList_scaleAstar.append(scaleAstar_res[k].epiIdxCvg)
      JcostList_scaleAstar.append(scaleAstar_res[k].J)
      trajLenList_scaleAstar.append(scaleAstar_res[k].l)
    if len(scaleAstar_res) > 0:
      epiIdxList_scaleAstar = np.array(epiIdxList_scaleAstar)
      JcostList_scaleAstar = np.array(JcostList_scaleAstar)
      mat_cvg_data[idx,1] = np.min(epiIdxList_scaleAstar)
      mat_cost_data[idx,1] = np.min(JcostList_scaleAstar)
    else:
      mat_cvg_data[idx,1] = np.inf
      mat_cost_data[idx,1] = np.inf

    ## DirCol + naive init
    naive_res_dict = misc.LoadPickle(configs["folder"]+"instance_"+str(idx)+"_dirColInit_result.pickle")
    mat_cost_data[idx,2] = naive_res_dict["lin_sol_cost"]
    mat_cost_data[idx,3] = naive_res_dict["rdm_sol_cost"]

  print("----mapname = ", mapname, "----")
  print("----mat_cost_data----")
  print(mat_cost_data)
  print("----mat_cost_data, PWDC succeeds in = ", succ_count, " cases ----")
  print("DirCol-wA* > 2*PWDC", np.sum(mat_cost_data[:,1] > 2*mat_cost_data[:,0]), np.sum(mat_cost_data[:,1] > 2*mat_cost_data[:,0])/succ_count)
  print("DirCol-linear > 2*PWDC", np.sum(mat_cost_data[:,2] > 2*mat_cost_data[:,0]), np.sum(mat_cost_data[:,2] > 2*mat_cost_data[:,0])/succ_count)
  print("DirCol-random > 2*PWDC", np.sum(mat_cost_data[:,3] > 2*mat_cost_data[:,0]), np.sum(mat_cost_data[:,3] > 2*mat_cost_data[:,0])/succ_count)
  print("------------------")
  print("DirCol-wA* > 1.5*PWDC", np.sum(mat_cost_data[:,1] > 1.5*mat_cost_data[:,0]), np.sum(mat_cost_data[:,1] > 1.5*mat_cost_data[:,0])/succ_count)
  print("DirCol-linear > 1.5*PWDC", np.sum(mat_cost_data[:,2] > 1.5*mat_cost_data[:,0]), np.sum(mat_cost_data[:,2] > 1.5*mat_cost_data[:,0])/succ_count)
  print("DirCol-random > 1.5*PWDC", np.sum(mat_cost_data[:,3] > 1.5*mat_cost_data[:,0]), np.sum(mat_cost_data[:,3] > 1.5*mat_cost_data[:,0])/succ_count)
  print("------------------")
  print("DirCol-wA* > 1.0*PWDC", np.sum(mat_cost_data[:,1] > 1.0*mat_cost_data[:,0]), np.sum(mat_cost_data[:,1] > 1.0*mat_cost_data[:,0])/succ_count)
  print("DirCol-linear > 1.0*PWDC", np.sum(mat_cost_data[:,2] > 1.0*mat_cost_data[:,0]), np.sum(mat_cost_data[:,2] > 1.0*mat_cost_data[:,0])/succ_count)
  print("DirCol-random > 1.0*PWDC", np.sum(mat_cost_data[:,3] > 1.0*mat_cost_data[:,0]), np.sum(mat_cost_data[:,3] > 1.0*mat_cost_data[:,0])/succ_count)

  print("----mat_cvg_data----")
  print(mat_cvg_data)
    
  print("DirCol-wA* > 2*PWDC", np.sum(mat_cvg_data[:,1] > 2*mat_cvg_data[:,0]), np.sum(mat_cvg_data[:,1] > 2*mat_cvg_data[:,0])/succ_count)
  # print("DirCol-linear > 2*PWDC", np.sum(mat_cvg_data[:,2] > 2*mat_cvg_data[:,0]))
  # print("DirCol-random > 2*PWDC", np.sum(mat_cvg_data[:,3] > 2*mat_cvg_data[:,0]))


  plt.figure(figsize=(3,2))

  count, _ = np.histogram( mat_cost_data[:,1] /mat_cost_data[:,0], bins=[0.5, 1, 1.5, 2.0, np.inf] )
  plt.bar(np.array([0.5, 1, 1.5, 2.0])+0.1, count, width=0.1, facecolor='r', alpha=0.8)

  count, _ = np.histogram( mat_cost_data[:,2] /mat_cost_data[:,0], bins=[0.5, 1, 1.5, 2.0, np.inf] )
  plt.bar(np.array([0.5, 1, 1.5, 2.0])+0.2, count, width=0.1, facecolor='g', alpha=0.8)

  count, _ = np.histogram( mat_cost_data[:,3] /mat_cost_data[:,0], bins=[0.5, 1, 1.5, 2.0, np.inf] )
  plt.bar(np.array([0.5, 1, 1.5, 2.0])+0.3, count, width=0.1, facecolor='b', alpha=0.8)

  plt.draw()
  plt.pause(2)
  plt.savefig(configs["folder"]+"cost_ratio_hist.png", bbox_inches='tight', dpi=200)
  return


#############

def RunPwdcOnce(configs, start_state, goal_state, idx, obs_pf = []):
  """
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
  """
  configs = ConfigClass.configs
  instance = misc.LoadPickle(configs["folder"] + mapname + ".pickle")
  obs_pf = instance["obs_pf"]
  start_states = instance["starts"]
  goal_states = instance["goals"]
  configs["map_grid"] = instance["grids"]
  for ii in range(start_states.shape[0]):
    RunPwdcOnce(configs, start_states[ii,:], goal_states[ii,:], ii, obs_pf)
  return

#############


if __name__ == "__main__":

  main_test_pwdc()
  main_test_wghA()
  main_test_naiveInit()

  main_plot_pwdc()
  main_plot_cvg_iters()
  main_compare_cvg_iters()