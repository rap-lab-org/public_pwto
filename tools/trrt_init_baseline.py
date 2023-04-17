

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

import random
import os
import sys
import math
import heapq

from pwdc import TrajStruct

class transitionRRT:
    """
    """

    class MyCar:
        def __init__(self):
            self.length = 3
            self.width = 4

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

            self.cost = 0.0
            self.parent = None
            self.goals = []

    def __init__(self, configs, obs_pf = []):


        self.cfg = configs
        self.obss = []
        self.obs_pf = obs_pf


        self.tjObjDict = dict()
        self.sol = dict()

        self._initMap()

        # RRT
        self.max_iter = 6000
        self.goal_sample_rate = 50
        self.expand_dis = 1
        self.obstacleList = [] # obstacleList:obstacle Positions [[x,y,size],...]
        self.connect_circle_dist = 15
        self.final_dis_range = 6


    def _initMap(self):
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
            self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])/10


        print(self.obs_pf)
        npix = self.cfg["npix"]
        Sinit = self.cfg["Sinit"]
        Sgoal = self.cfg["Sgoal"]
        self.cost_map = self.obs_pf # dist to obstacle
        max_costmap = np.nanmax(self.cost_map)
        min_costmap = np.nanmin(self.cost_map)
        self.cost_map = (self.cost_map - min_costmap)/(max_costmap-min_costmap)

        map_size = self.cost_map.shape
        print("Shape of the cost map is: ")
        print(self.cost_map.shape)
        self.map_x_bound = [0,map_size[0]]
        self.map_y_bound = [0,map_size[1]]

        self.map_x_span = np.linspace(self.map_x_bound[0], self.map_x_bound[1], map_size[0])
        self.map_y_span = np.linspace(self.map_y_bound[0], self.map_y_bound[1], map_size[1])
        self.map_mesh_grid = np.meshgrid(self.map_x_span, self.map_y_span)
        print("self.map_mesh_grid: ", self.map_mesh_grid)
        ## start and goal node.
        self.s_start = (int(Sinit[0]*npix)-1, int(Sinit[1]*npix)-1)
        self.s_goal  = (int(Sgoal[0]*npix)-1, int(Sgoal[1]*npix)-1)
        

        self.start = self.Node(self.s_start[0], self.s_start[1])
        self.end = self.Node(self.s_goal[0], self.s_goal[1])

        print(self.s_start)
        print(self.s_goal)
        return



    def planning(self, animation=True, search_until_maxiter=False):
        """
        rrt star path planning
        animation: flag for animation on or off
        search_until_maxiter: search until max iteration for path improving or not
        """
        n_fail = 0
        T = 1
        my_car = self.MyCar()
        print(my_car.length)

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print('ite: ',i)
            rnd = self.get_random_point()
            print('-- random node: ---',"[", rnd, "]")
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            print("-- nearest_node --", "[", nearest_node.x,nearest_node.y, "]")

            new_node = self.steer(rnd, nearest_node)
            print("-- new node", "[" , new_node.x, new_node.y, "]")
            # Modified
            new_node.parent = nearest_node

            d, _ = self.calc_distance_and_angle(new_node, nearest_node)
            c_near = self.get_point_cost(nearest_node.x, nearest_node.y)
            c_new = self.get_point_cost(new_node.x, new_node.y)

            # cmax = 0.5 origin
            [trans_test, n_fail, T] = self.transition_test(c_near, c_new, d, cmax=0.7, k=2, t=T, nFail=n_fail)
            print('--trans-test:', trans_test)
            print('--------------------------------------')
            if self.check_collision(new_node, self.obstacleList) and trans_test and \
                    not self.map_vehicle_collision(my_car, new_node.x, new_node.y, threshold=0.6):
                # near_inds = self.find_near_nodes(new_node)
                # new_node = self.choose_parent(new_node, near_inds)
                # if new_node:
                #     self.node_list.append(new_node)
                #     self.rewire(new_node, near_inds)
                # Modified
                if new_node:
                    self.node_list.append(new_node) 
            else:
                n_fail += 1

            if animation and i % 5 == 0:  # draw after every 5 iterations
                self.draw_graph(rnd)

            if not search_until_maxiter and new_node:  # check reaching the goal
                # d, _ = self.calc_dist_to_end(new_node)
                d = self.calc_dist_to_goal(new_node.x,new_node.y)
                # if d <= self.expand_dis:
                if d <= self.final_dis_range:
                    return self.generate_final_course(len(self.node_list) - 1)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None


    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.map_x_bound[0], self.map_x_bound[1]),
                   random.uniform(self.map_y_bound[0], self.map_y_bound[1])]
        else:  # goal point sampling
            rnd = self.s_goal
        return rnd


    def get_nearest_list_index(self, node_list, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def steer(self, rnd, nearest_node):
        new_node = self.Node(rnd[0], rnd[1])
        d, theta = self.calc_distance_and_angle(nearest_node, new_node)
        if d > self.expand_dis:
            new_node.x = nearest_node.x + self.expand_dis * math.cos(theta)
            new_node.y = nearest_node.y + self.expand_dis * math.sin(theta)

        return new_node

    def get_point_cost(self, x, y):
        j = list(self.map_x_span).index(min(self.map_x_span, key=lambda temp: abs(temp - x)))
        i = list(self.map_y_span).index(min(self.map_y_span, key=lambda temp: abs(temp - y)))
        return self.cost_map[j, i] # pay attention I changed the order here

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    def transition_test(self, ci, cj, dij, cmax, k, t, nFail):
        """
        Note: This does not include nFail or auto-tuning of
        temperature. Refer to pg. 640 of "SAMPLING-BASED PATH PLANNING ON CONFIGURATION-SPACE COSTMAPS"
        to incorporate these features into this function

        Note: need to try and determine the cmax
        """
        alpha = 2
        nFail_max = 5

        if cj > cmax:
            return [False, nFail, t]
        if cj < ci:
            # t /= alpha
            nFail = 0
            return [True, nFail, t]
        p = math.exp((-(cj-ci)/dij)/(k*t))
        print('cj-ci',cj-ci)
        print('dij',dij)
        print('ttttttt',t)
        print('ppppppppppppppppp:',p)
        if random.uniform(0, 1) < p:
            t /= alpha
            nFail = 0
            return [True, nFail, t]
        else:
            if nFail > nFail_max:
                t *= alpha
                nFail = 0
            else:
                nFail += 1
            return [False, nFail, t]

    def check_collision(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= size ** 2:
                return False  # collision

        return True  # safe

    def map_vehicle_collision(self, my_vehicle, x, y, threshold):

        X, Y = self.map_mesh_grid
        x_min = x - my_vehicle.length
        x_max = x + my_vehicle.length
        y_min = y - my_vehicle.width
        y_max = y + my_vehicle.width

        for i in range(0, len(self.map_x_span)):
            for j in range(0, len(self.map_y_span)):
                if (x_min <= X[i, j]) and (X[i, j] <= x_max):
                    if (y_min <= Y[i, j]) and (Y[i, j] <= y_max):
                        if self.cost_map[i, j] >= threshold:
                            return True
        return False


    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds


    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            d, theta = self.calc_distance_and_angle(self.node_list[i], new_node)
            if self.check_collision_extend(self.node_list[i], theta, d):
                costs.append(self.node_list[i].cost + d)
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        new_node.cost = min_cost
        min_ind = near_inds[costs.index(min_cost)]
        new_node.parent = self.node_list[min_ind]

        return new_node
        

    def check_collision_extend(self, near_node, theta, d):

        tmp_node = copy.deepcopy(near_node)

        for i in range(int(d / self.expand_dis)):
            tmp_node.x += self.expand_dis * math.cos(theta)
            tmp_node.y += self.expand_dis * math.sin(theta)
            if not self.check_collision(tmp_node, self.obstacleList):
                return False

        return True


    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            d, theta = self.calc_distance_and_angle(near_node, new_node)
            new_cost = new_node.cost + d

            if near_node.cost > new_cost:
                if self.check_collision_extend(near_node, theta, d):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                d, _ = self.calc_distance_and_angle(parent_node, node)
                node.cost = parent_node.cost + d
                self.propagate_cost_to_leaves(node)

    
    # functions to check goal

    def calc_dist_to_end(self, from_node):
        d = []
        dx = []
        dy = []
        print("self.end.goals:",self.end.goals)
        for goal in self.end.goals:
            dx.append(goal[0] - from_node.x)
            dy.append(goal[1] - from_node.y)
            d.append(math.sqrt(dx[len(dx) - 1] ** 2 + dy[len(dy) - 1] ** 2))
        d_min = min(d)
        idx = d.index(d_min)
        theta_min = math.atan2(dy[idx], dx[idx])
        return d_min, theta_min


    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.final_dis_range]

        if not goal_inds:
            closest_idx = min(dist_to_goal_list) 
            goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i ==closest_idx]
            # return None

        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None



    def draw_graph(self, rnd=None):
        plt.clf()
        plt.contourf(self.map_mesh_grid[1], self.map_mesh_grid[0], self.cost_map)
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x],
                         [node.y, node.parent.y],
                         "-y")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.map_x_bound[0], self.map_x_bound[1], self.map_y_bound[0], self.map_y_bound[1]])
        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.01)


# optimization

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


    def _getInitGuess(self):
        """
        generate initial guess
        """
        px,py,node_num = self._path2xy(self.path)
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
        tjObj = TrajStruct()
        tjObj.id = 0
        tjObj.init_path = self.path
        tjObj.Z, tjObj.l = self._getInitGuess()
        Zsol, info = optm_ddc2.dirCol_ddc2(\
                            tjObj.Z, configs["Sinit"], configs["Sgoal"], \
                            configs["optm_weights"] , self.obss, tjObj.l, \
                            configs["interval_value"], configs["vu_bounds"], max_iter=1) 
                            # just one iter, to init.
        # tjObj.J = info['obj_val']
        tjObj.J = self.costJ(tjObj.Z, tjObj.l)
        self.tjObjDict[0] = tjObj

        # for k in range(len(self.Astar_pathlist)):
        #     tjObj = TrajStruct()
        #     tjObj.id = k
        #     tjObj.init_path = self.Astar_pathlist[k]
        #     tjObj.Z, tjObj.l = self._getInitGuess(k)
        #     Zsol, info = optm_ddc2.dirCol_ddc2(\
        #                         tjObj.Z, configs["Sinit"], configs["Sgoal"], \
        #                         configs["optm_weights"] , self.obss, tjObj.l, \
        #                         configs["interval_value"], configs["vu_bounds"], max_iter=1) 
        #                         # just one iter, to init.
        #     # tjObj.J = info['obj_val']
        #     tjObj.J = self.costJ(tjObj.Z, tjObj.l)
        #     self.tjObjDict[k] = tjObj




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

    def solveTRRT(self):
        """
        """
        print("[INFO] translation based RRT, enter _init...")
        
        print("[INFO] translation based RRT, enter searching_repeated_astar...")
        self.path = self.planning()
        if self.path is None:
            print("Cannot find path")
            return self.sol
        else:
            print("found path!! The node number is ",len(self.path))
            self.draw_graph()
            plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            # plt.show()

        print("[INFO] translation based RRT, enter _initOpen...")
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

    #     cost_Astar = []
    #     node_num_Astar = []
    #     Zsol_Astar = []
    #     max_iter_Astar = 1000


    #     for k in range(len(self.Astar_pathlist)):

    #         astar_initial_guess,node_num = self._getInitGuess(k)

    #         Zsol, info = optm_ddc2.dirCol_ddc2(\
    #         astar_initial_guess, configs["Sinit"], configs["Sgoal"], \
    #         configs["optm_weights"], self.obss, node_num, \
    #         configs["interval_value"], configs["vu_bounds"], max_iter=max_iter_Astar)

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
    #         plt.savefig(configs["folder"]+'Astar_'+str(k)+'.png' , bbox_inches='tight', dpi=200)


    #         ### print out cost value
    #         J = self.costJ(self.obss, Zsol, node_num, configs["optm_weights"][0], configs["optm_weights"][1], configs["n"], configs["m"])
    #         cost_Astar.append(J)
    #         print("linear init guess, Jcost = ", J)

    #         node_num_Astar.append(node_num)
    #         Zsol_Astar.append(Zsol)

    #     res_dict = dict()
    #     res_dict["num_nodes"] = node_num_Astar
    #     res_dict["Astar_sol"] = Zsol_Astar
    #     res_dict["Astar_sol_cost"] = cost_Astar
    #     res_dict["max_iter"] = max_iter_Astar
    #     res_dict["pf"] = self.obs_pf
    #     misc.SavePickle(res_dict, configs["folder"]+"Astar_res.pickle")

        # return cost_Astar



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
    configs["Astar_weight_list"] = ([0.01, 1.2], [0.1,0.95],[0.2,0.8],[0.5,0.5],[0.8,0.2],[0.95,0.1],[1.2,0.01])
    # configs["Astar_weight_list"] = ([0.5,0.5],[0.8,0.2],[0.95,0.1],[1.2,0.01])

    astar = Astar(configs)

    astar.solveScaleAstar()

