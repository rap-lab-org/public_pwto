
import numpy as np
import pickle
import matplotlib.pyplot as plt

import context
import misc
import obstacle as obs

def GenerateRandomGrids(ny, nx, obst_density):
  """
  grid(ny,nx), obstacle threshold 0 = no obst, 1 = all obst.
  """
  grids = np.zeros([ny,nx])
  
  np.random.seed()
  for i in range(int(ny*nx*obst_density)):
    x = int(np.random.random()*31.999)
    y = int(np.random.random()*31.999)
    grids[y,x] = 1

  return grids

def PlotGrids(grids):

  fig = plt.figure(figsize=(4,4))
  plt.imshow(grids, cmap='Greys',  interpolation='nearest', origin="lower")
  # plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  sx,sy = grids.shape
  plt.xticks(np.arange(0, sx, 4))
  plt.yticks(np.arange(0, sy, 4))
  # plt.axis("off")
  plt.draw()
  plt.pause(1)
  return

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

      if grids[iy,ix]==0 :
        op_x.append(ix)
        op_y.append(iy)

      # if grids[iy,ix]==0 and \
      #    grids[iy-1,ix]==0 and grids[iy+1,ix]==0 and grids[iy,ix-1]==0 and grids[iy,ix+1]==0 and \
      #    grids[ix-1,iy-1]==0 and grids[ix+1,iy+1]==0 and grids[ix+1,iy-1]==0 and grids[ix-1,iy+1]==0:
      #   op_x.append(ix)
      #   op_y.append(iy)

  return op_x, op_y

def GenStartGoals(grids, n):
  """
  randomly generate starts and goals within a grid map.
  """
  grid_size = grids.shape[0]
  xfre,yfre = GetFreeXY(grids)
  xfree = np.array(xfre)
  yfree = np.array(yfre)
  np.random.seed()
  print("total free cells = ", len(xfree))
  N = int(len(xfree)/2-1)
  idx_list = np.random.permutation(len(xfree))[0:2*N]
  sx = xfree[idx_list[0:N]]
  sy = yfree[idx_list[0:N]]
  gx = xfree[idx_list[N:2*N]]
  gy = yfree[idx_list[N:2*N]]

  start_states = np.zeros([n,5])
  goal_states = np.zeros([n,5])
  ii = 0
  for jj in range(N):
    start_states[ii,:] = np.array( [ sx[jj]/grid_size,sy[jj]/grid_size, 0, 0, 0 ] )
    goal_states[ii,:] = np.array( [ gx[jj]/grid_size,gy[jj]/grid_size, 0, 0, 0 ] )
    min_manhattan_dist_thres = 1.0
    if np.sum( np.abs(start_states[ii,:] - goal_states[ii,:]) ) < min_manhattan_dist_thres:
      # don't want the start and goals to be too close to each other, which makes the instances trivial.
      continue
    ii = ii + 1
    if ii >= n:
      break
  return start_states, goal_states


def GenObsPf(map_grid, npix):
    grid_size,_ = map_grid.shape # assume to be a square
    obsts_all = misc.findObstacles(map_grid)
    obsts = obsts_all / grid_size # scale coordinates into [0,1]x[0,1]
    obss = obs.ObstSet( obsts) #/// 

    obs_pf = obss.potentialField(1, 1, npix)*100
    return obs_pf

#

def main_gen_tests(ts_name):
  folder = "./results/instances/" # folder path containing tests.
  # obs_cov_val = 1e-3
  npix = 200

  # GenerateTestSerie(gridx, gridy, num of robots, num of test cases in a serie, obst_thres):
  gridx = 32
  gridy = 32
  ntest = 10
  if ts_name == "random32A":
    obst_thres = 0.13
  elif ts_name == "random32B":
    obst_thres = 0.10
  elif ts_name == "random32C":
    obst_thres = 0.20
  elif ts_name == "random32E":  # very sparse map
    obst_thres = 0.05
  elif ts_name == "random32F":  # very sparse map
    obst_thres = 0.05
  elif ts_name == "random32G":  # very sparse map
    obst_thres = 0.10
  elif ts_name == "random32H":  # very sparse map
    obst_thres = 0.18
  elif ts_name == "random32I":  # very sparse map
    obst_thres = 0.18
  elif ts_name == "random32J":  # cov=1e-3
    obst_thres = 0.18

  elif ts_name == "random32AA":  # cov=6e-4
    obst_thres = 0.20

  elif ts_name == "random32K_1":  # cov=5e-4
    obst_thres = 0.20
  elif ts_name == "random32K_2":  # cov=5e-4
    obst_thres = 0.15
  elif ts_name == "random32D":
    obst_thres = 0.20
  elif ts_name == "random32E":  # very sparse map
    obst_thres = 0.05
  elif ts_name == "random32F":  # very sparse map
    obst_thres = 0.05
  elif ts_name == "random32G":  # very sparse map
    obst_thres = 0.10
  elif ts_name == "random32H":  # very sparse map
    obst_thres = 0.18
  elif ts_name == "random32I":  # very sparse map
    obst_thres = 0.18
  elif ts_name == "random32J":  # cov=1e-3
    obst_thres = 0.18

  elif ts_name == "random32AA":  # cov=6e-4
    obst_thres = 0.20

  elif ts_name == "random32K_1":  # cov=5e-4
    obst_thres = 0.20
  elif ts_name == "random32K_2":  # cov=5e-4
    obst_thres = 0.15
  elif ts_name == "random32K_3":  # cov=5e-4
    obst_thres = 0.10
  elif ts_name == "random32K_4":  # cov=5e-4
    obst_thres = 0.35
  elif ts_name == "random32K_5":  # cov=5e-4
    obst_thres = 0.30
  elif ts_name == "random32K_6":  # cov=5e-4
    obst_thres = 0.25

  elif ts_name == "random32L_1":  # cov=2e-4
    obst_thres = 0.3
  elif ts_name == "random32L_2":  # cov=2e-4
    obst_thres = 0.25
  elif ts_name == "random32L_3":  # cov=2e-4
    obst_thres = 0.20

  elif ts_name == "random32L_4":  # cov=2e-4
    obst_thres = 0.3
  elif ts_name == "random32L_5":  # cov=2e-4
    obst_thres = 0.25
  elif ts_name == "random32L_6":  # cov=2e-4
    obst_thres = 0.20
  elif ts_name == "random32L_7":  # cov=2e-4
    obst_thres = 0.15


  elif ts_name == "random32L_8":  # cov=2e-4
    obst_thres = 0.1
    
  elif ts_name == "random32L_9":  # cov=2e-4
    obst_thres = 0.15
    
  elif ts_name == "random32L_10":  # cov=2e-4
    obst_thres = 0.2

  elif ts_name == "random32M":  # cov=10e-4
    obst_thres = 0.03
  elif ts_name == "random32N":  # cov=12e-4
    obst_thres = 0.02
  elif ts_name == "random32O":  # cov=20e-4
    obst_thres = 0.015

  grids = GenerateRandomGrids(gridx,gridy,obst_thres)
  starts, goals = GenStartGoals(grids, ntest)

  instances = dict()
  instances["name"] = ts_name
  instances["starts"] = starts
  instances["goals"] = goals
  instances["grids"] = grids

  ### grid plot
  PlotGrids(grids)
  plt.savefig(folder+ts_name+"_grid.png", dpi=200)

  instances["obs_pf"] = GenObsPf(grids, npix)

  ### cost field plot
  fig = plt.figure(figsize=(4,4))
  xx = np.linspace(0,1,num=npix)
  yy = np.linspace(0,1,num=npix)
  X,Y = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next. # no it's not. still x,y
  pf = instances["obs_pf"]
  print("pf.shape = ", pf.shape)
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.xticks([0,1])
  plt.yticks([0,1])
  plt.draw()
  save_path = folder+ts_name+"_costField.png"
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)

  misc.SavePickle(instances, folder + ts_name+".pickle")
  return

if __name__ == "__main__":
  # main_gen_tests("random32A")
  # main_gen_tests("random32B")
  # main_gen_tests("random32C")
  # main_gen_tests("random32D")
  # main_gen_tests("random32E")
  # main_gen_tests("random32F")
  # main_gen_tests("random32G")
  # main_gen_tests("random32H")
  # main_gen_tests("random32I")
  # main_gen_tests("random32J")

  # main_gen_tests("random32AA")
  # main_gen_tests("random32K_1")
  # main_gen_tests("random32K_2")
  # main_gen_tests("random32K_3")
  # main_gen_tests("random32K_4")
  # main_gen_tests("random32K_5")
  # main_gen_tests("random32K_6")
  # main_gen_tests("random32L_1")
  # main_gen_tests("random32L_2")
  # main_gen_tests("random32L_3")
  # main_gen_tests("random32L_4")
  # main_gen_tests("random32L_5")
  # main_gen_tests("random32L_6")
  # main_gen_tests("random32L_7")
  # main_gen_tests("random32L_8")
  # main_gen_tests("random32L_9")
  # main_gen_tests("random32L_10")

  # main_gen_tests("random32M")
  # main_gen_tests("random32N")
  main_gen_tests("random32O")