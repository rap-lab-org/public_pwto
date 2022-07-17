


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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
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
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

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

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    astar = AStar(s_start, s_goal, "euclidean")
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()











def costJ(obss, Z, l, w1, w2, n, m):
  """Minimize the sum of the squares of the control torque."""
  X,U,_ = opty.utils.parse_free(Z, n, m, l)

  xy_tj = X[0:2,:].T
  J1 = w1*np.sum(U**2) # control cost
  J2 = w2*obss.arrayCost(xy_tj) # potential field cost
  # there is no cost term for staying close to the path.
  return J1 + J2

def  kAStar():
  pass


def run_naive_init(configs, num_nodes, save_path, max_iter):

  ### generate map and potential field
  map_grid = misc.LoadMapDao(configs["map_grid_path"])
  obsts_all = misc.findObstacles(map_grid)
  grid_size,_ = map_grid.shape
  obsts = obsts_all / grid_size
  obss = obs.ObstSet( obsts )
  pf = obss.potentialField(1,1,configs["npix"])
  
  n = configs["n"]
  m = configs["m"]

  # initial_guess = misc.linearInitGuess(configs["Sinit"][0:2], configs["Sgoal"][0:2], \
  #   num_nodes, n, m, configs["interval_value"])
  
  # To do: initaial guess from k-AStar

  

  configs["optm_weights"][2] = 0 # no need to stay close to the initial guess.
  Zsol, info = optm_ddc2.dirCol_ddc2(\
    initial_guess, configs["Sinit"], configs["Sgoal"], \
    configs["optm_weights"], obss, num_nodes, \
    configs["interval_value"], max_iter)

  Xsol, Usol, _ = opty.utils.parse_free(Zsol, n, m, num_nodes)

  ### Figure

  fig = plt.figure(figsize=(4,4))
  plt.xticks([0,1])
  plt.yticks([0,1])
  plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b--")
  xx = np.linspace(0,1,num=configs["npix"])
  yy = np.linspace(0,1,num=configs["npix"])
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
  plt.plot(configs["Sinit"][0],configs["Sinit"][1],"ro")
  plt.plot(configs["Sgoal"][0],configs["Sgoal"][1],"r*")
  plt.plot(Xsol[0,:],Xsol[1,:],"r.", markersize=2)

  # 2nd, random initial guess
  np.random.seed(0)
  initial_guess = np.random.randn( num_nodes*(n+m) )
  Zsol2, info = optm_ddc2.dirCol_ddc2(\
    initial_guess, configs["Sinit"], configs["Sgoal"], \
    configs["optm_weights"], obss, num_nodes, \
    configs["interval_value"], max_iter)
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
