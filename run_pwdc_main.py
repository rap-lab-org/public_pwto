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



# mapname = 'random32_1'
# mapname = 'random16_1'
mapname = 'random16_simple'
# mapname = 'paris_64'
ConfigClass = getConfig.Config(mapname)
configs = ConfigClass.configs


def test_pwdc():
  """
  """
  solver = pwdc.PWDC(configs)
  solver.Solve()
  print("PWDC get", len(solver.sol), " solutions.")
  misc.SavePickle(solver, configs["folder"]+"result.pickle")
  return

def test_pwdc_plot():
  """
  """
  solver = misc.LoadPickle(configs["folder"]+"result.pickle")
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
                  configs["folder"]+"solTraj_"+str(k)+".png", fig_sz)
    print("k = ", k, "converge episode = ", res[k].epiIdxCvg, " costJ = ", res[k].J, " traj L = ", res[k].l)
    
    trj_x= np.array([res[k].getPosX()]).flatten()
    trj_y= np.array([res[k].getPosY()]).flatten()
    trj_theta= np.array([res[k].getPosTheta()]).flatten()
    trj_v= np.array([res[k].getVelLinear()]).flatten()
    trj_w= np.array([res[k].getVelRot()]).flatten()
    print("len",len(trj_v))

    trj_av= np.array([res[k].getAccVel()]).flatten()
    trj_aw= np.array([res[k].getAccRot()]).flatten()
    
    file1 = open(configs["folder"]+"solTraj_data_"+str(k)+".txt","w")
    for i in range(len(trj_x)):
      file1.write(str(trj_x[i]) + "  " + str(trj_y[i]) + "   " + str(trj_theta[i])+"   " \
        + str(trj_v[i]) +"   " + str(trj_w[i]) +"   " + str(trj_av[i]) +"   " + str(trj_aw[i]) +'\n')
    file1.close()


  return

def test_and_plot_naive_init():
  # num_nodes = 170
  num_nodes = 220
  save_path = configs["folder"] + "naiveTraj.png"
  max_iter = 1000
  # naive.run_naive_init(folder, configs, num_nodes, save_path, max_iter)
  naive.run_naive_init(configs, num_nodes, save_path, max_iter)

  # linear init guess, Jcost =  16581811.360421443
  # random init guess, Jcost =  25993129.972545788


# def test_and_plot_kAstat_init():
#   configs = ConfigClass.configs


#   astarSol = kAstar.AStar(configs)
#   astarSol.SolvekAstar()
#   misc.SavePickle(astarSol, configs["folder"]+"kAstar_res.pickle")
#   configs = astarSol.cfg
#   fig_sz,_ = astarSol.map_grid.shape
#   fig_sz = 4
#   kAstar_res = astarSol.sol
#   print(" get ", len(astarSol.kAstar_pathlist), " A-STAR paths" )


#   for k in kAstar_res:

#     px,py,_ = astarSol._path2xy(kAstar_res[k].init_path)
#     kAstar.plotTraj(astarSol.obs_pf, configs, \
#                   np.array([px,py]), np.array([kAstar_res[k].getPosX(),kAstar_res[k].getPosY()]), \
#                   configs["folder"]+"kAstar_raj_"+str(k)+".png", fig_sz)
#     print("k = ", k, "converge episode = ", kAstar_res[k].epiIdxCvg, " costJ = ", kAstar_res[k].J, " traj L = ", kAstar_res[k].l)


def test_and_plot_scaleAstat_init():

  astarSol = scaleAstar.AStar(configs)
  astarSol.solveScaleAstar()
  misc.SavePickle(astarSol, configs["folder"]+"scaleAstar_res.pickle")
  configs = astarSol.cfg
  fig_sz,_ = astarSol.map_grid.shape
  fig_sz = 4
  scaleAstar_res = astarSol.sol
  print(" get ", len(astarSol.kAstar_pathlist), " A-STAR paths" )


  for k in scaleAstar_res:

    px,py,_ = astarSol._path2xy(scaleAstar_res[k].init_path)
    scaleAstar.plotTraj(astarSol.obs_pf, configs, \
                  np.array([px,py]), np.array([scaleAstar_res[k].getPosX(),scaleAstar_res[k].getPosY()]), \
                  configs["folder"]+"kAstar_raj_"+str(k)+".png", fig_sz)
    print("k = ", k, "converge episode = ", scaleAstar_res[k].epiIdxCvg, " costJ = ", scaleAstar_res[k].J, " traj L = ", scaleAstar_res[k].l)




def plot_compare_iters():

  configs = ConfigClass.configs

  pwdc_res_dict = misc.LoadPickle(configs["folder"]+"result.pickle")
  naive_res_dict = misc.LoadPickle(configs["folder"]+"naive_res.pickle")
  scaleAstar_res_dict = misc.LoadPickle(configs["folder"]+"scaleAstar_res.pickle")

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
  plt.plot(epiIdxList_pwdc+1, JcostList_pwdc/np.min(JcostList_pwdc), "g.")
  # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")


  scaleAstar_res = scaleAstar__res_dict.sol
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
  plt.plot(epiIdxList_scaleAstar + 1, JcostList_scaleAstar/np.min(JcostList_pwdc), "b.")
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



  lin_sol_cost_ratio = naive_res_dict["lin_sol_cost"]/np.min(JcostList_pwdc)
  plt.plot([1,10],[lin_sol_cost_ratio,lin_sol_cost_ratio],'r--',lw=1)

  rdm_sol_cost_ratio = naive_res_dict["rdm_sol_cost"]/np.min(JcostList_pwdc)
  plt.plot([1,10],[rdm_sol_cost_ratio,rdm_sol_cost_ratio],'c--',lw=1)

  # kastar_sol_cost_ratio = kAstar_res_dict["kAstar_sol_cost"]/np.min(JcostList_pwdc)
  # plt.plot([1,10],[kastar_sol_cost_ratio,kastar_sol_cost_ratio],'g--',lw=1)

  plt.grid("on")
  plt.draw()
  plt.pause(2)
  plt.savefig(configs["folder"]+"iter_compare.png", bbox_inches='tight', dpi=200)
  return



def plot_pareto_front():

  configs = ConfigClass.configs


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

def plot_pareto_paths():
  """
  """


  configs = ConfigClass.configs
  solver = misc.LoadPickle(configs["folder"]+"result.pickle")

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
  plt.savefig(configs["folder"]+"pareto_paths.png", bbox_inches='tight', dpi=200)
  return


if __name__ == "__main__":

  test_pwdc()

  test_pwdc_plot()

  test_and_plot_naive_init()

  test_and_plot_kAstat_init()

  plot_compare_iters()

  plot_pareto_front()

  plot_pareto_paths()