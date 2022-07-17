"""
Pareto Warm-start Direct Collocation (PWDC)
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from pywsto import context
import misc
import obstacle as obs
# from public_emoa import emoa_py_api as emoa
import optm_ddc2
import opty.utils 

## TODO, try 8-connected grid.


class loadPlotMap():
  """
  Pareto Warm-start Direct Collocation.
  """
  def __init__(self, configs):
    """
    """
    self.cfg = configs
    self.obss = []
    self.obs_pf = []
    # self.open = misc.PrioritySet()

    self.fig_sz = 4

    ## load map and generate obstacle_set.
    if "map_grid_path" in self.cfg:
      self.downmap, self.map_grid = self.LoadMapDao(self.cfg["map_grid_path"] , self.cfg["downsample"] )
    else:
      self.map_grid = self.cfg["map_grid"]
    print("self.map_grid",self.map_grid)
    print('size of map grid',np.shape(self.map_grid))
    grid_size,_ = self.map_grid.shape
    obsts_all = misc.findObstacles(self.map_grid)
    print("obsts_all",len(obsts_all))


    with open(self.cfg["folder"]+'/64-64.txt','w') as f:
      f.write(str(self.downmap))

    obsts = obsts_all / grid_size # scale coordinates into [0,1]x[0,1]
    self.obss = obs.ObstSet( obsts, self.cfg["obst_cov_val"] )
    print("obsts",len(self.obss.mvn_list))

    # self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])*100
    self.obs_pf = self.obss.potentialField(1,1,self.cfg["npix"])*100



    print("self.obss.shape",self.obss.potentialField(1,1,self.cfg["npix"]).shape)
    # print("obss[0]",self.obss.potentialField(1,1,self.cfg["npix"])[0])
    print("self.obs_pf.shape",self.obs_pf.shape)
    # print("obs_pf[0]",self.obs_pf[0])


  def LoadMapDao(self, map_file, downsample=1):
    grids = np.zeros((2,2))
    with open(map_file,'r') as f:
      lines = f.readlines()
      downmap = ""

      lidx = 0
      nx = 0
      ny = 0
      
      a = lines[1].split(" ")
      nx = int(int(a[1])/downsample)
      b = lines[2].split(" ")
      ny = int(int(b[1])/downsample)
      
      grids = np.zeros((nx,ny))

      for lidx in range(nx):
        downmap_line = ""
        x = int(lidx)*downsample + 4
        linedata = lines[x].split("\n")
        linedata = str(linedata[0])
        print(linedata)
        for cidy in range(ny):
          y = int(cidy)*downsample
          ia = linedata[y]
          downmap_line += ia

          if ia == "." or ia == "G":
            grids[lidx,cidy] = 0
          else:
            grids[lidx,cidy] = 1
        downmap += downmap_line+"\n"

    print(downmap)

    return downmap, grids

  def plotTraj(self):
    """
    """

    pf = self.obs_pf
    fig_sz = self.fig_sz
    configs = self.cfg
    save_path = configs["map_grid_path"]
    save_path = save_path.replace(".map","_terrain_map.png")

    fig = plt.figure(figsize=(fig_sz,fig_sz))
    s = configs["Sinit"]
    d = configs["Sgoal"]
    xx = np.linspace(0,1,num=configs["npix"])
    yy = np.linspace(0,1,num=configs["npix"])
    Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
    plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.axis('off')
    plt.draw()
    plt.pause(1)
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)


if __name__ == '__main__':


  folder = "data/test_result/"

  configs = dict()
  configs["folder"] = folder
  configs["map_grid_path"] = configs["folder"] + "Paris_1_256.map"
  # configs["map_grid_path"] = configs["folder"] + "random-16-16-20.map"
  # configs["map_grid_path"] = configs["folder"] + "random-16-16-simple.map"

  configs["n"] = 5
  configs["m"] = 2
  configs["Sinit"] = np.array([0.1, 0.1, 0, 0, 0])
  configs["Sgoal"] = np.array([0.9, 0.8, 0 ,0, 0])
  configs["interval_value"] = 0.2
  configs["npix"] = 100
  # configs["emoa_path"] = "../public_emoa/build/run_emoa"
  # configs["iters_per_episode"] = 100
  # configs["optm_weights"] = [0.01, 5000, 200]
  #   # w1 = 0.01 # control cost, for the u terms.
  #   # w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
  #   # w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
  # configs["total_epi"] = 10
  # configs["hausdorf_filter_thres"] = 8
  configs["obst_cov_val"] = 7*1e-4
  configs["downsample"] = 4


  map_plotter = loadPlotMap(configs)
  map_plotter.plotTraj()