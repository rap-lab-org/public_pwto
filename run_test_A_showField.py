"""This solves the simple pendulum swing up problem presented here:

http://hmc.csuohio.edu/resources/human-motion-seminar-jan-23-2014

A simple pendulum is controlled by a torque at its joint. The goal is to
swing the pendulum from its rest equilibrium to a target angle by minimizing
the energy used to do so.

"""

from collections import OrderedDict

import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
from opty.utils import building_docs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from opty.utils import parse_free

import context
import obstacle as obs

import emoa_py_api as emoa


def LoadMapDao(map_file):
  grids = np.zeros((2,2))
  with open(map_file,'r') as f:
    lines = f.readlines()
    lidx = 0
    nx = 0
    ny = 0
    for line in lines:
      if lidx == 1:
        a = line.split(" ")
        nx = int(a[1])
      if lidx == 2:
        a = line.split(" ")
        ny = int(a[1])
      if lidx == 4:
        grids = np.zeros((nx,ny))
      if lidx >= 4: # map data begin
        x = lidx - 4
        y = 0
        a = line.split("\n")
        # print(a[0])
        # print(len(str(a[0])))
        for ia in str(a[0]):
          # print(ia)
          if ia == "." or ia == "G":
            grids[x,y] = 0
          else:
            grids[x,y] = 1
          y = y+1
      lidx = lidx + 1
  return grids

def findObstacles(grid):
  """
  """
  out = list()
  nyt,nxt = grid.shape
  for iy in range(nxt):
    for ix in range(nyt):
      if grid[iy,ix] == 1:
        out.append(np.array([iy,ix]))
  return np.array(out)


map_grid = LoadMapDao("runtime_data/random-32-32-20.map")
obsts_all = findObstacles(map_grid)
obsts = obsts_all / 32.0
# print(obsts)

Sinit = np.array([0.1, 0.1, 0, 0, 0])
Sgoal = np.array([0.9, 0.8, 0 ,0, 0])

num_nodes = 100
save_animation = False

# interval_value = duration / (num_nodes - 1)
interval_value = 0.1
duration = (num_nodes-1)*interval_value

obss = obs.ObstSet( obsts )
npix = 100
print("start to compute pf...")
pf = obss.potentialField(1,1,npix)*100
print("pf done...")

## convert to a 100x100 grid
c1 = np.ones([npix,npix]) # distance
c2 = pf # dist to obstacle

vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)

fig = plt.figure(figsize=(5,5))

xx = np.linspace(0,1,num=100)
yy = np.linspace(0,1,num=100)
Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.

plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
plt.plot(Sinit[0],Sinit[1],"ro")
plt.plot(Sgoal[0],Sgoal[1],"r*")


plt.draw()
plt.pause(1)
# print(" select_path_x = ", select_path_x)
# print(" select_path_y = ", select_path_y)

case_ID = "A"
plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-instance.png", bbox_inches='tight', dpi=200)


