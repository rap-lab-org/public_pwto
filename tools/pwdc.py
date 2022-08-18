"""
Pareto Warm-start Direct Collocation (PWDC)
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import context

# import context
import misc
import obstacle as obs
import emoa_py_api as emoa
import optm_ddc2
import opty.utils 

class TrajStruct:
  """
  A data structure for round-robin search.
  """
  def __init__(self):
    self.id = 0
    self.Z = [] # solution, X and U.
    self.J = -1 # overall cost
    self.l = 0 # length of traj
    self.epiCount = 0 # total number of episode
    self.epiIdxCvg = 0 # The index of the episode where the optimization converges.
    self.X = []
    self.U = []
    self.init_path = [] # the path that warm-start this trajectory.
    self.isConverged = False # whether optimization already converges.

  def parseZ(self, n, m):
    """
    return X, U
    """
    return opty.utils.parse_free(self.Z, n, m, self.l)
  
  def getPosX(self):
    """
    l is the number of nodes
    """
    return self.Z[:self.l]

  def getPosY(self):
    """
    l is the number of nodes
    """
    return self.Z[self.l:2*self.l]

  def getPosTheta(self):
    """
    l is the number of nodes
    """
    return self.Z[2*self.l:3*self.l]

  def getVelLinear(self):
    """
    l is the number of nodes
    """
    return self.Z[3*self.l:4*self.l]
  def getVelRot(self):
    """
    l is the number of nodes
    """
    return self.Z[4*self.l:5*self.l]

  def getAccVel(self):
    """
    l is the number of nodes
    """
    return self.Z[5*self.l:6*self.l]

  def getAccRot(self):
    """
    l is the number of nodes
    """
    return self.Z[6*self.l:7*self.l]


class PWDC():
  """
  Pareto Warm-start Direct Collocation.
  """
  def __init__(self, configs, obs_pf0 = []):
    """
    """
    self.cfg = configs
    self.obss = []
    self.obs_pf = obs_pf0
    self.emoa_res_dict = dict()
    # self.open = misc.PrioritySet()
    self.tjObjDict = dict()
    self.sol = dict()
    return

  def costJ(self, Z, l):
    """
    Objective function J as defined in the problem statement. The numerical integrated version.
    """
    X,U,_ = opty.utils.parse_free(Z, self.cfg["n"], self.cfg["m"], l)
    xy_tj = X[0:2,:].T
    J1 = self.cfg["optm_weights"][0]*np.sum(U**2)
    J2 = self.cfg["optm_weights"][1]*self.obss.arrayCost(xy_tj)
    # there is no cost term for staying close to the path.
    return J1 + J2

  def _init(self):
    """
    """
    ## load map and generate obstacle_set.
    if "map_grid_path" in self.cfg:
      if "downsample" in self.cfg:
        self.map_grid = misc.LoadMapDaoDownSample(self.cfg["map_grid_path"], self.cfg["downsample"])
      else:
        self.map_grid = misc.LoadMapDao(self.cfg["map_grid_path"] )
    else:
      self.map_grid = self.cfg["map_grid"]

    grid_size,_ = self.map_grid.shape
    obsts_all = misc.findObstacles(self.map_grid)
    obsts = obsts_all / grid_size # scale coordinates into [0,1]x[0,1]
    self.obss = obs.ObstSet( obsts, self.cfg["obst_cov_val"] )
    if self.obs_pf == []:
      self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])*100
    return
    
  def _graphSearch(self):
    """
    Solve MOPP problem.
    """

    ## generate cost, 2-d now. TODO, more costs ?
    npix = self.cfg["npix"]
    Sinit = self.cfg["Sinit"]
    Sgoal = self.cfg["Sgoal"]
    c1 = np.ones([npix,npix]) # distance
    c2 = self.obs_pf # dist to obstacle

    ## start and goal node.
    # vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
    # vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)
    vo = int(int(Sinit[0]*npix)*npix + int(Sinit[1]*npix))
    vd = int(int(Sgoal[0]*npix)*npix + int(Sgoal[1]*npix))

    print("npix = ", npix)
    print(" vo, vd = ", vo, vd)


    ## run EMOA*
    self.emoa_res_dict = dict()
    self.emoa_res_dict = emoa.runEMOA([c1,c2], self.cfg["folder"], \
      self.cfg["emoa_path"], self.cfg["folder"]+"temp-res.txt", \
      vo, vd, 60)
    misc.lexSortResult(self.emoa_res_dict)

  def _path2xy(self,p):
    """
    """
    npix = self.cfg['npix']
    py = []
    px = []
    idx = 0
    for v in p:
      py.append( (v%npix)*(1/npix) )
      px.append( int(np.floor(v/npix))*(1.0/npix) )
    return px,py
    # not right?

  def _getInitGuess(self, k):
    """
    generate initial guess
    """
    px,py = self._path2xy(self.emoa_res_dict["paths"][k])
    return misc.path2InitialGuess(\
      px, py, len(px), self.cfg["n"], self.cfg["m"], self.cfg["interval_value"])

  def _ifTerminate(self):
    """
    """
    if self.open.size() == 0:
      print("[INFO] PWDC, terminates since open is empty.")
      return True
    return

  # def _initOpenOnJ(self):
  #   """
  #   optimize all initial paths for a few iterations.
  #   """
  #   for k in self.emoa_res_dict["paths"]:
  #     tjObj = TrajStruct()
  #     tjObj.id = k
  #     tjObj.Z = self._getInitGuess(k)
  #     tjObj.l = len(self.emoa_res_dict["paths"][k])
  #     Zsol, info = optm_ddc2.dirCol_ddc2(\
  #       tjObj.Z, self.cfg["Sinit"], self.cfg["Sgoal"], \
  #       self.cfg["optm_weights"] , self.obss, tjObj.l, \
  #       self.cfg["interval_value"], max_iter=1) # just one iter, to init.
  #     tjObj.J = info['obj_val']
  #     if info["status"] == -1:
  #       tjObj.isConverged = False
  #     else:
  #       tjObj.isConverged = True
  #       # TODO, more efforts need to be spent in IPOPT output flags.
  #     self.open.add(tjObj.J, tjObj)
  #   # end for
  #   return

  def _initOpen(self):
    """
    optimize all initial paths for a few iterations.
    """
    # self.open = dict()
    n_pareto_sol = len(self.emoa_res_dict["paths"])
    temp_copy = copy.deepcopy(self.emoa_res_dict["paths"])
    picked_dict = dict()
    picked_dict[0] = self.emoa_res_dict["paths"][0]
    temp_copy.pop(0)
    last_path = picked_dict[0]
    _,nxt = self.map_grid.shape

    for k in temp_copy:
      filtered = False
      for k2 in picked_dict:
        hd = misc.pathHdf(self.cfg["npix"], np.array(picked_dict[k2]), np.array(temp_copy[k]))
        if hd < self.cfg["hausdorf_filter_thres"]:
          filtered = True
          break
      if not filtered:
        picked_dict[k] = temp_copy[k]

    print("[INFO] After Hausdorff Filter, #paths = ", len(picked_dict))

    for k in picked_dict:
      tjObj = TrajStruct()
      tjObj.id = k
      tjObj.init_path = picked_dict[k]
      tjObj.Z = self._getInitGuess(k)
      tjObj.l = len(picked_dict[k])
      Zsol, info = optm_ddc2.dirCol_ddc2(\
        tjObj.Z, self.cfg["Sinit"], self.cfg["Sgoal"], \
        self.cfg["optm_weights"] , self.obss, tjObj.l, \
        self.cfg["interval_value"], self.cfg["vu_bounds"], max_iter=1) # just one iter, to init.
      # tjObj.J = info['obj_val']
      tjObj.J = self.costJ(tjObj.Z, tjObj.l)
      self.tjObjDict[k] = tjObj
    # end for
    return

  def _optmEpisode(self, tjObj, epiIdx):
    """
    One episode of optpimization.
    """
    Z, info = optm_ddc2.dirCol_ddc2(\
        tjObj.Z, self.cfg["Sinit"], self.cfg["Sgoal"], \
        self.cfg["optm_weights"] , self.obss, tjObj.l, \
        self.cfg["interval_value"], self.cfg["vu_bounds"], max_iter=self.cfg['iters_per_episode'])

    tjObj.epiCount += 1 # total number of episode
    tjObj.Z = Z
    # tjObj.J = info['obj_val']
    tjObj.J = self.costJ(tjObj.Z, tjObj.l)
    if info["status"] == -1:
      tjObj.isConverged = False
      # self.open.add(tjObj.J, tjObj)
    else:
      print("tjObj.id = ", tjObj.id, " converges, epiIdx = ", epiIdx)
      tjObj.isConverged = True
      tjObj.epiIdxCvg = epiIdx
      self.sol[tjObj.id] = tjObj
      # self.tjObjDict.pop(tjObj.id) # remove it from candidates.
    return tjObj.isConverged

  def Solve(self):
    """
    Solve process.
    """
    print("[INFO] PWDC, enter _init...")
    self._init()
    print("[INFO] PWDC, enter _graphSearch...")
    self._graphSearch()
    print("[INFO] PWDC, enter _initOpen...")
    self._initOpen()
    # round robin fashion
    for epiIdx in range(self.cfg["total_epi"]):
      print("------ epiIdx = ", epiIdx, " ------")
      tbd = list()
      for k in self.tjObjDict:
        tjObj = self.tjObjDict[k]
        cvg = self._optmEpisode(tjObj, epiIdx)
        if cvg:
          tbd.append(k)
      for k in tbd:
        self.tjObjDict.pop(k)
    return self.sol

def plotTraj(pf, configs, p, tj, save_path, fig_sz):
  """
  """
  fig = plt.figure(figsize=(fig_sz,fig_sz))
  s = configs["Sinit"]
  d = configs["Sgoal"]
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(s[0],s[1],"ro")
  plt.plot(d[0],d[1],"r*")
  plt.plot(p[0,:], p[1,:], "b--")
  plt.plot(tj[0,:], tj[1,:], "r.", markersize=1.5)
  plt.xticks([0,1])
  plt.yticks([0,1])
  # plt.axis('off')
  plt.draw()
  plt.pause(1)
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)

def plotTrajs(pf, configs, tjs, save_path, fig_sz):
  """
  """
  fig = plt.figure(figsize=(fig_sz,fig_sz))
  s = configs["Sinit"]
  d = configs["Sgoal"]
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(s[0],s[1],"ro")
  plt.plot(d[0],d[1],"r*")
  for tj in tjs:
    plt.plot(tj[0,:], tj[1,:], "r.", markersize=1.5)
  plt.xticks([0,1])
  plt.yticks([0,1])
  # plt.axis('off')
  plt.draw()
  plt.pause(1)
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)
