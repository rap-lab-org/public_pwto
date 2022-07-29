

import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import context
import pwdc
import misc
import optm_ddc2
import obstacle as obs
import opty.utils 


import os
import sys
import math
import heapq

from pwdc import TrajStruct

class Astar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, configs):


        self.cfg = configs
        self.obss = []
        self.obs_pf = []

        # self.Env = env.Env()  # class Env

        self.u_set = [(-1, 0), (0, 1),
                        (1, 0),  (0, -1)]  # feasible input set

        # self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

        self.tjObjDict = dict()
        self.sol = dict()
        self.kAstar_pathlist = []

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
      # self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])*100
      self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])/100


      print(self.obs_pf)
      npix = self.cfg["npix"]
      Sinit = self.cfg["Sinit"]
      Sgoal = self.cfg["Sgoal"]
      self.c1 = np.ones([npix,npix]) # distance
      self.c2 = self.obs_pf # dist to obstacle
      # max_c2 = np.nanmax(self.c2)
      # min_c2 = np.nanmin(self.c2)
      # self.c2 = (self.c2 - min_c2)/(max_c2-min_c2)


      ## start and goal node.
      self.s_start = (int(Sinit[0]*npix)-1, int(Sinit[1]*npix)-1)
      self.s_goal  = (int(Sgoal[0]*npix)-1, int(Sgoal[1]*npix)-1)

      print(self.s_start)
      print(self.s_goal)
      return


    def searching_repeated_astar(self):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        weight_list = self.cfg["weight_list"]

        path_list, visited = [], []


        for weight in weight_list:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, weight)
            path_list.append(p_k)
            visited.append(v_k)

        # self.kAstar_pathlist = path_list
        return path_list, visited

    def repeated_searching(self, s_start, s_goal, weight):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            # print('current s',s)

            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n,weight)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors

        """
        s_neighbor = []
        for u in self.u_set:
            s_n  = (s[0] + u[0], s[1] + u[1])
            if s_n[0] < 0 or s_n[0] >= self.cfg['npix'] \
                or s_n[1] < 0 or s_n[1] >= self.cfg['npix']:
                continue
            s_neighbor.append(s_n)
        return s_neighbor

    def cost(self, s_start, s_goal,weight):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        c1_motion = weight[0] * self.c1[s_start]
        c2_motion = weight[1] * self.c2[s_goal]
        cost_motion = c1_motion + c2_motion
        return cost_motion


    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        # goal = self.s_goal  # goal node
        # return math.hypot(goal[0] - s[0], goal[1] - s[1])

        # # use 0 for now

        return 0


    def _path2xy(self,plist):
        """
        """
        npix = self.cfg['npix']
        py = []
        px = []
        idx = 0
        for p in plist:

            px.insert(0, (p[0]+1)*(1.0/npix))
            py.insert(0, (p[1]+1)*(1.0/npix))

        node_num = len(px)

        return px,py,node_num


    def _getInitGuess(self, k):
        """
        generate initial guess
        """
        px,py,node_num = self._path2xy(self.kAstar_pathlist[k])
        return misc.path2InitialGuess(\
          px, py, node_num, self.cfg["n"], self.cfg["m"], self.cfg["interval_value"]),node_num


    def costJ(self, Z, l):
        """Minimize the sum of the squares of the control torque."""
        X,U,_ = opty.utils.parse_free(Z, self.cfg["n"], self.cfg["m"], l)
        xy_tj = X[0:2,:].T
        J1 = self.cfg["optm_weights"][0]*np.sum(U**2)
        J2 = self.cfg["optm_weights"][1]*self.obss.arrayCost(xy_tj)
        # there is no cost term for staying close to the path.
        return J1 + J2


    def _initOpen(self):
        configs = self.cfg

        for k in range(len(self.kAstar_pathlist)):
            tjObj = TrajStruct()
            tjObj.id = k
            tjObj.init_path = self.kAstar_pathlist[k]
            tjObj.Z, tjObj.l = self._getInitGuess(k)
            Zsol, info = optm_ddc2.dirCol_ddc2(\
                                tjObj.Z, configs["Sinit"], configs["Sgoal"], \
                                configs["optm_weights"] , self.obss, tjObj.l, \
                                configs["interval_value"], configs["vu_bounds"], max_iter=1) 
                                # just one iter, to init.
            # tjObj.J = info['obj_val']
            tjObj.J = self.costJ(tjObj.Z, tjObj.l)
            self.tjObjDict[k] = tjObj




    def _optmEpisode(self, tjObj, epiIdx):
        """
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

    def SolvekAstar(self):
        """
        """
        print("[INFO] scalarized AStar, enter _init...")
        self._init()

        print("[INFO] scalarized AStar, enter searching_repeated_astar...")
        self.kAstar_pathlist,_ = self.searching_repeated_astar()

        print("[INFO] scalarized AStar, enter _initOpen...")
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


    # def runopt(self):

    #     cost_kAstar = []
    #     node_num_kAstar = []
    #     Zsol_kAstar = []
    #     max_iter_kAstar = 1000


    #     for k in range(len(self.kAstar_pathlist)):

    #         astar_initial_guess,node_num = self._getInitGuess(k)

    #         Zsol, info = optm_ddc2.dirCol_ddc2(\
    #         astar_initial_guess, configs["Sinit"], configs["Sgoal"], \
    #         configs["optm_weights"], self.obss, node_num, \
    #         configs["interval_value"], configs["vu_bounds"], max_iter=max_iter_kAstar)

    #         Xsol, Usol, _ = opty.utils.parse_free(Zsol, configs["n"], configs["m"], node_num)

    #         ### Figure

    #         fig = plt.figure(figsize=(4,4))
    #         plt.xticks([0,1])
    #         plt.yticks([0,1])
    #         plt.plot(astar_initial_guess[:node_num],astar_initial_guess[node_num:2*node_num],"b--")
    #         xx = np.linspace(0,1,num=configs["npix"])
    #         yy = np.linspace(0,1,num=configs["npix"])
    #         Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
    #         plt.contourf(X, Y, self.obs_pf, levels=np.linspace(np.min(self.obs_pf), np.max(self.obs_pf),500), cmap='gray_r')
    #         plt.plot(configs["Sinit"][0],configs["Sinit"][1],"ro")
    #         plt.plot(configs["Sgoal"][0],configs["Sgoal"][1],"r*")
    #         plt.plot(Xsol[0,:],Xsol[1,:],"r.", markersize=2)

    #         ###
    #         plt.draw()
    #         plt.pause(1)
    #         plt.savefig(configs["folder"]+'kAstar_'+str(k)+'.png' , bbox_inches='tight', dpi=200)


    #         ### print out cost value
    #         J = self.costJ(self.obss, Zsol, node_num, configs["optm_weights"][0], configs["optm_weights"][1], configs["n"], configs["m"])
    #         cost_kAstar.append(J)
    #         print("linear init guess, Jcost = ", J)

    #         node_num_kAstar.append(node_num)
    #         Zsol_kAstar.append(Zsol)

    #     res_dict = dict()
    #     res_dict["num_nodes"] = node_num_kAstar
    #     res_dict["kAstar_sol"] = Zsol_kAstar
    #     res_dict["kAstar_sol_cost"] = cost_kAstar
    #     res_dict["max_iter"] = max_iter_kAstar
    #     res_dict["pf"] = self.obs_pf
    #     misc.SavePickle(res_dict, configs["folder"]+"kAstar_res.pickle")

        # return cost_kAstar



if __name__ == '__main__':
    # main()


    folder = "results/res_random32_1/"
    mapscale = 18.6

    configs = dict()
    configs["folder"] = folder
    configs["map_grid_path"] = configs["folder"] + "random-32-32-20.map"
    configs["n"] = 5
    configs["m"] = 2
    configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
    configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
    configs["interval_value"] = 0.2
    configs["npix"] = 100
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
    configs["vu_bounds"] = np.array([1/mapscale, 5, 0.3, 0.8]) # v,w,ua,uw
    configs["weight_list"] = ([0.01, 1.2], [0.1,0.95],[0.2,0.8],[0.5,0.5],[0.8,0.2],[0.95,0.1],[1.2,0.01])
    # configs["weight_list"] = ([0.5,0.5],[0.8,0.2],[0.95,0.1],[1.2,0.01])

    astar = Astar(configs)

    astar.SolvekAstar()

