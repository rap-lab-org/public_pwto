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

# mapname = 'random16_1'

# mapname = 'room32_1'

# mapname = 'random32_1'
mapname = 'random32_2'

# mapname = 'random64_1'
# mapname = 'random64_2'
# mapname = 'paris_64'

ConfigClass = getConfig.Config(mapname)
# configs = ConfigClass.configs

NInstance = 10

##########################################
### Global params END ###
##########################################


def GetFreeXY(grids):
  """
  xfree,yfree = GetFreeXY()
  return non-obstacle locations in input grids in form of lists.
  """
  op_x = list()
  op_y = list()
  (nyt, nxt) = grids.shape # nyt = ny total, nxt = nx total
  for ix in range(1,nxt-1):
    for iy in range(1,nyt-1):
      if mapname == 'random32_1' and \
         grids[iy,ix]==0 and grids[iy-1,ix]==0 and grids[iy+1,ix]==0 and \
         grids[iy,ix-1]==0 and grids[iy,ix+1]==0:
        op_x.append(ix)
        op_y.append(iy)
      elif (mapname == 'random32_2' or mapname == "room32_1") and \
         grids[iy,ix]==0 and grids[iy-1,ix]==0 and grids[iy+1,ix]==0 and \
         grids[iy,ix-1]==0 and grids[iy,ix+1]==0 and grids[ix-1,iy-1]==0 and grids[ix+1,iy+1]==0:
        op_x.append(ix)
        op_y.append(iy)
      elif mapname == 'random64_1' and \
         grids[iy,ix]==0 and grids[iy-1,ix]==0 and grids[iy+1,ix]==0 and \
         grids[iy,ix-1]==0 and grids[iy,ix+1]==0:
        op_x.append(ix)
        op_y.append(iy)
      elif mapname == 'random64_2' and \
         grids[iy,ix]==0 and grids[iy-1,ix]==0 and grids[iy+1,ix]==0 and \
         grids[iy,ix-1]==0 and grids[iy,ix+1]==0 and grids[ix-1,iy-1]==0 and grids[ix+1,iy+1]==0:
        op_x.append(ix)
        op_y.append(iy)
      elif mapname == 'random16_1' and \
         grids[iy,ix]==0:
        op_x.append(ix)
        op_y.append(iy)
  return op_x, op_y


def GenStartGoals(configs):
  """
  randomly generate starts and goals within a grid map.
  """
  grids = misc.LoadMapDao(configs["map_grid_path"] )
  grid_size,_ = grids.shape # assume to be a square
  xfre,yfre = GetFreeXY(grids)
  xfree = np.array(xfre)
  yfree = np.array(yfre)
  n = NInstance # number of start-goal pairs
  np.random.seed(0)
  
  # these two numbers help with the random generation of starts and goals. 
  # We want the distance between start and goal to be not too short.
  # The first way we randomly generate may fail to generate 10 instances.
  # So I add these two numbers to help.
  # It's just one way to possibly generate enough instance.
  N = 0 
  N0 = 0 
  if mapname == "random16_1":
    # 16x16
    N = 20 
    N0 = 9
  elif mapname == "random32_1":
    # # 32x32
    N = 50 
    N0 = 1
  elif mapname == "random32_2":
    # # 32x32
    N = 50
    N0 = 3
  elif mapname == "room32_1":
    # # 32x32
    N = 30
    N0 = 0
  elif mapname == "random64_1":
    # # 32x32
    N = 50 
    N0 = 1
  elif mapname == "random64_2":
    # # 32x32
    N = 50
    N0 = 3
  elif mapname == "paris_64":
    # # 64x64
    N = 50 
    N0 = 0


  print("total free cells = ", len(xfree))
  idx_list = np.random.permutation(len(xfree))[0:2*N]
  sx = xfree[idx_list[0:N]]
  sy = yfree[idx_list[0:N]]
  gx = xfree[idx_list[N-N0:2*N-N0]]
  gy = yfree[idx_list[N-N0:2*N-N0]]

  start_states = np.zeros([n,5])
  goal_states = np.zeros([n,5])
  ii = 0
  for jj in range(N):
    start_states[ii,:] = np.array( [ sx[jj]/grid_size,sy[jj]/grid_size, 0, 0, 0 ] )
    goal_states[ii,:] = np.array( [ gx[jj]/grid_size,gy[jj]/grid_size, 0, 0, 0 ] )

    min_manhattan_dist_thres = 100
    if mapname == "random16_1":
      # 16x16
      min_manhattan_dist_thres = 0.65 # for random 16x16
    elif mapname == "random32_1":
      # 32x32
      min_manhattan_dist_thres = 1.0 # for random 32x32
    elif mapname == 'random32_2':
      # 32x32
      min_manhattan_dist_thres = 1.0 # for random 32x32
    elif mapname == "room32_1" :
      # 32x32 room
      min_manhattan_dist_thres = 0.6 # for room 32x32      
    elif mapname == "random64_1":
      # 64x64
      min_manhattan_dist_thres = 1.0 # for random 64x64
    elif mapname == "random64_2":
      # 64x64
      min_manhattan_dist_thres = 1.0 # for random 64x64
    elif mapname == "paris_64":
      # # 64x64
      min_manhattan_dist_thres = 1.0 # for paris 64x64

    if np.sum( np.abs(start_states[ii,:] - goal_states[ii,:]) ) < min_manhattan_dist_thres:
      # don't want the start and goals to be too close to each other, which makes the instances trivial.
      continue
    ii = ii + 1
    if ii >= n:
      break
  return start_states, goal_states


def GenObsPf(configs):
    map_grid = misc.LoadMapDao(configs["map_grid_path"] )
    grid_size,_ = map_grid.shape # assume to be a square
    obsts_all = misc.findObstacles(map_grid)
    obsts = obsts_all / grid_size # scale coordinates into [0,1]x[0,1]
    obss = obs.ObstSet( obsts, configs["obst_cov_val"] )
    obs_pf = obss.potentialField(1, 1, configs["npix"])*100
    return obs_pf

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

  # terrain plot
  fig = plt.figure(figsize=(fig_sz,fig_sz))
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  pf = solver.obs_pf
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.xticks([0,1])
  plt.yticks([0,1])
  plt.draw()
  save_path = configs["folder"]+"instance_"+str(idx)+"_costField.png"
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)

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


  # # plot Pareto paths
  # npix = solver.cfg["npix"]
  # for k in solver.sol:
  #   p = solver.emoa_res_dict[k]
  #   px = list()
  #   py = list()
  #   for v in p:
  #       py.append( (v%npix)*(1/npix) )
  #       px.append( int(np.floor(v/npix))*(1.0/npix) )
  #   plt.plot(px,py,"b--")

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

  # num_pwdc_path = 5 # fixed
  # # num_pwdc_path = len(emoa_res)
  # # print('num_pwdc_path',num_pwdc_path)
  # # co = 1
  # dw = 1/(num_pwdc_path-1)
  # for k in range(num_pwdc_path):
  #   # weight_offset = co*k/num_pwdc_path*0.5
  #   w1 = k*dw
  #   w2 = 1-w1
  #   print("weight = ", w1, w2)
  #   configs["Astar_weight_list"].append([w1,w2])

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

  num_nodes = int(np.mean(trajLenList_pwdc))
  max_iter = int(configs["iters_per_episode"]* configs["total_epi"])
  print('num nodes of the navie init',num_nodes, "max_iter = ", max_iter)

  # naive.run_naive_init(folder, configs, num_nodes, save_path, max_iter)
  res_dict = naive.run_naive_init_for_test(configs, num_nodes, max_iter)

  save_path = configs["folder"] + "instance_"+str(idx)+"_dirColInit_result.pickle"
  misc.SavePickle(res_dict, save_path)
  return



# plot??
#   configs = astarSol.cfg
#   fig_sz,_ = astarSol.map_grid.shape
#   fig_sz = 4
#   scaleAstar_res = astarSol.sol
#   print(" get ", len(astarSol.Astar_pathlist), " A-STAR paths" )


#   for k in scaleAstar_res:

#     px,py,_ = astarSol._path2xy(scaleAstar_res[k].init_path)
#     scaleAstar.plotTraj(astarSol.obs_pf, configs, \
#                   np.array([px,py]), np.array([scaleAstar_res[k].getPosX(),scaleAstar_res[k].getPosY()]), \
#                   configs["folder"]+"scaleAstar_raj_"+str(k)+".png", fig_sz)
#     print("k = ", k, "converge episode = ", scaleAstar_res[k].epiIdxCvg, " costJ = ", scaleAstar_res[k].J, " traj L = ", scaleAstar_res[k].l)




# def test_pwdc_plot():
#   """
#   """
#   configs = ConfigClass.configs

#   solver = misc.LoadPickle(configs["folder"]+"result.pickle")
#   # solver = loaded[1]
#   configs = solver.cfg
#   fig_sz,_ = solver.map_grid.shape
#   fig_sz = 4
#   res = solver.sol
#   print(" get ", len(solver.emoa_res_dict), " Pareto paths" )
#   for k in res:
#     px,py = solver._path2xy(res[k].init_path)
#     pwdc.plotTraj(solver.obs_pf, configs, \
#                   np.array([px,py]), np.array([res[k].getPosX(),res[k].getPosY()]), \
#                   configs["folder"]+"solTraj_"+str(k)+".png", fig_sz)
#     print("k = ", k, "converge episode = ", res[k].epiIdxCvg, " costJ = ", res[k].J, " traj L = ", res[k].l)
    
#     trj_x= np.array([res[k].getPosX()]).flatten()
#     trj_y= np.array([res[k].getPosY()]).flatten()
#     trj_theta= np.array([res[k].getPosTheta()]).flatten()
#     trj_v= np.array([res[k].getVelLinear()]).flatten()
#     trj_w= np.array([res[k].getVelRot()]).flatten()
#     print("len",len(trj_v))

#     trj_av= np.array([res[k].getAccVel()]).flatten()
#     trj_aw= np.array([res[k].getAccRot()]).flatten()
    
#     file1 = open(configs["folder"]+"solTraj_data_"+str(k)+".txt","w")
#     for i in range(len(trj_x)):
#       file1.write(str(trj_x[i]) + "  " + str(trj_y[i]) + "   " + str(trj_theta[i])+"   " \
#         + str(trj_v[i]) +"   " + str(trj_w[i]) +"   " + str(trj_av[i]) +"   " + str(trj_aw[i]) +'\n')
#     file1.close()

#   return

# def test_and_plot_naive_init():

#   configs = ConfigClass.configs

#   solver_pwdc = misc.LoadPickle(configs["folder"]+"result.pickle")
#   pwdc_res = solver_pwdc.sol
#   trajLenList_pwdc = []
#   for k in pwdc_res:
#     trajLenList_pwdc.append(pwdc_res[k].l)
#   trajLenList_pwdc = np.array(trajLenList_pwdc)

#   num_nodes = int(np.mean(trajLenList_pwdc))
#   print('num nodes of the navie init',num_nodes)
#   save_path = configs["folder"] + "naiveTraj.png"
#   max_iter = int(configs["iters_per_episode"]* configs["total_epi"])

#   # naive.run_naive_init(folder, configs, num_nodes, save_path, max_iter)
#   naive.run_naive_init(configs, num_nodes, save_path, max_iter)

#   # linear init guess, Jcost =  16581811.360421443
#   # random init guess, Jcost =  25993129.972545788


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


# def plot_compare_iters():

#   configs = ConfigClass.configs

#   pwdc_res_dict = misc.LoadPickle(configs["folder"]+"result.pickle")
#   naive_res_dict = misc.LoadPickle(configs["folder"]+"naive_res.pickle")
#   scaleAstar_res_dict = misc.LoadPickle(configs["folder"]+"scaleAstar_res.pickle")

#   # kAstar_res_dict = misc.LoadPickle(configs["folder"]+"kAstar_res.pickle")

#   pwdc_res = pwdc_res_dict.sol
#   plt.figure(figsize=(3,2))
#   epiIdxList_pwdc = list()
#   JcostList_pwdc = list()
#   trajLenList_pwdc = list()
#   for k in pwdc_res:
#     epiIdxList_pwdc.append(pwdc_res[k].epiIdxCvg)
#     JcostList_pwdc.append(pwdc_res[k].J)
#     trajLenList_pwdc.append(pwdc_res[k].l)
#   epiIdxList_pwdc = np.array(epiIdxList_pwdc)
#   JcostList_pwdc = np.array(JcostList_pwdc)
#   trajLenList_pwdc = np.array(trajLenList_pwdc)
#   plt.plot(epiIdxList_pwdc+1, JcostList_pwdc/np.min(JcostList_pwdc), "g.",alpha=0.5)
#   # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")


#   scaleAstar_res = scaleAstar_res_dict.sol
#   epiIdxList_scaleAstar = list()
#   JcostList_scaleAstar = list()
#   trajLenList_scaleAstar = list()
#   for k in scaleAstar_res:
#     epiIdxList_scaleAstar.append(scaleAstar_res[k].epiIdxCvg)
#     JcostList_scaleAstar.append(scaleAstar_res[k].J)
#     trajLenList_scaleAstar.append(scaleAstar_res[k].l)
#   epiIdxList_scaleAstar = np.array(epiIdxList_scaleAstar)
#   JcostList_scaleAstar = np.array(JcostList_scaleAstar)
#   trajLenList_scaleAstar = np.array(trajLenList_scaleAstar)
#   plt.plot(epiIdxList_scaleAstar + 1, JcostList_scaleAstar/np.min(JcostList_pwdc), "b^",alpha=0.5)
#   # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")


#   # kAstar_res = kAstar_res_dict.sol
#   # epiIdxList_kAstar = list()
#   # JcostList_kAstar = list()
#   # trajLenList_kAstar = list()
#   # for k in kAstar_res:
#   #   epiIdxList_kAstar.append(kAstar_res[k].epiIdxCvg)
#   #   JcostList_kAstar.append(kAstar_res[k].J)
#   #   trajLenList_kAstar.append(kAstar_res[k].l)
#   # epiIdxList_kAstar = np.array(epiIdxList_kAstar)
#   # JcostList_kAstar = np.array(JcostList_kAstar)
#   # trajLenList_kAstar = np.array(trajLenList_kAstar)
#   # plt.plot(epiIdxList_kAstar+1, JcostList_kAstar/np.min(JcostList_pwdc), "b.")
#   # # plt.plot(epiIdxList+1, trajLenList/np.min(trajLenList), "bo")



#   lin_sol_cost_ratio = naive_res_dict["lin_sol_cost"]/np.min(JcostList_pwdc)
#   plt.plot([1,10],[lin_sol_cost_ratio,lin_sol_cost_ratio],'r--',lw=2)

#   rdm_sol_cost_ratio = naive_res_dict["rdm_sol_cost"]/np.min(JcostList_pwdc)
#   plt.plot([1,10],[rdm_sol_cost_ratio,rdm_sol_cost_ratio],'c--',lw=1)

#   # kastar_sol_cost_ratio = kAstar_res_dict["kAstar_sol_cost"]/np.min(JcostList_pwdc)
#   # plt.plot([1,10],[kastar_sol_cost_ratio,kastar_sol_cost_ratio],'g--',lw=1)

#   plt.grid("on")
#   plt.draw()
#   plt.pause(2)
#   plt.savefig(configs["folder"]+"iter_compare.png", bbox_inches='tight', dpi=200)
#   return

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


def main_test_pwdc():
  """
  """
  print("--- main_test_pwdc ---")
  configs = ConfigClass.configs
  print("----starts and goals start to generate----")
  start_states, goal_states = GenStartGoals(configs)
  print("----starts and goals generated----")
  print(start_states)
  print(goal_states)
  obs_pf = GenObsPf(configs)
  print("----obs_pf generated----")
  for ii in range(start_states.shape[0]):
    RunPwdcOnce(configs, start_states[ii,:], goal_states[ii,:], ii, obs_pf)
  return

def main_test_wghA():
  """
    # must run weighted A* after PWDC, since weighted A* needs the #sol in PWDC.
  """
  configs = ConfigClass.configs
  start_states, goal_states = GenStartGoals(configs)
  for ii in range(start_states.shape[0]):
    RunWeightedAstar(configs, start_states[ii,:], goal_states[ii,:], ii)
  return

def main_test_naiveInit():
  configs = ConfigClass.configs
  start_states, goal_states = GenStartGoals(configs)
  for ii in range(start_states.shape[0]):
    RunNaiveInit(configs, start_states[ii,:], goal_states[ii,:], ii)
  return

def main_plot_pwdc():
  print("--- main_plot_pwdc ---")
  configs = ConfigClass.configs
  start_states, goal_states = GenStartGoals(configs)
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

  print("----mat_cvg_data----")
  print(mat_cvg_data)
    
  print("DirCol-wA* > 2*PWDC", np.sum(mat_cvg_data[:,1] > 2*mat_cvg_data[:,0]), np.sum(mat_cvg_data[:,1] > 2*mat_cvg_data[:,0])/succ_count)
  # print("DirCol-linear > 2*PWDC", np.sum(mat_cvg_data[:,2] > 2*mat_cvg_data[:,0]))
  # print("DirCol-random > 2*PWDC", np.sum(mat_cvg_data[:,3] > 2*mat_cvg_data[:,0]))
  return


if __name__ == "__main__":

  # main_test_pwdc()
  # main_test_wghA()
  # main_test_naiveInit()
  # main_plot_pwdc()
  # main_plot_cvg_iters()
  main_compare_cvg_iters()