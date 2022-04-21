

import numpy as np


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


def path2InitialGuess(px, py, n_nodes, n, m, interval_value):
  """
  p = path find by EMOA*.
  """
  lp = len(px)
  initial_guess = np.ones(n_nodes*(n+m))*0
  for i in range(n_nodes):
    idx = int( np.floor( lp*(i/n_nodes) ) )
    initial_guess[i] = px[idx] # x
    initial_guess[n_nodes+i] = py[idx] # y
  for i in range(1,n_nodes):
    dy = initial_guess[i] - initial_guess[i-1]
    dx = initial_guess[n_nodes+i] - initial_guess[n_nodes+i-1]
    initial_guess[2*n_nodes+i-1] = np.arctan2(dy,dx) # theta
    initial_guess[3*n_nodes+i-1] = np.sqrt(dy**2+dx**2) / interval_value # v
  sidx = n*(n_nodes)
  for i in range(1,n_nodes):
    dtheta = initial_guess[2*n_nodes+i] - initial_guess[2*n_nodes+i-1]
    dtheta = np.arctan2(np.cos(dtheta),np.sin(dtheta)) # round to [-pi,pi]
    initial_guess[4*n_nodes+i-1] = ( dtheta ) / interval_value # w
  for i in range(1,n_nodes):
    dy = initial_guess[i] - initial_guess[i-1]
    dx = initial_guess[n_nodes+i] - initial_guess[n_nodes+i-1]
    initial_guess[sidx+i-1] = ( initial_guess[3*n_nodes+i] - initial_guess[3*n_nodes+i-1] ) / interval_value # ua
    dw = ( initial_guess[4*n_nodes+i] - initial_guess[4*n_nodes+i-1] )
    initial_guess[sidx+n_nodes+i-1] = dw / interval_value # uw
  return initial_guess

def linearInitGuess(pstart, pend, n_nodes, n, m, interval_value):
  """
  """
  initial_guess = np.ones(n_nodes*(n+m))*0
  initial_guess[:n_nodes] = np.linspace(pstart[0],pend[0],n_nodes) # x
  initial_guess[n_nodes:2*n_nodes] = np.linspace(pstart[1],pend[1],n_nodes) # y

  for i in range(1,n_nodes):
    dy = initial_guess[i] - initial_guess[i-1]
    dx = initial_guess[n_nodes+i] - initial_guess[n_nodes+i-1]
    initial_guess[2*n_nodes+i-1] = np.arctan2(dy,dx) # theta
    initial_guess[3*n_nodes+i-1] = np.sqrt(dy**2+dx**2) / interval_value # v
  sidx = n*(n_nodes)
  for i in range(1,n_nodes):
    dtheta = initial_guess[2*n_nodes+i] - initial_guess[2*n_nodes+i-1]
    dtheta = np.arctan2(np.cos(dtheta),np.sin(dtheta)) # round to [-pi,pi]
    initial_guess[4*n_nodes+i-1] = ( dtheta ) / interval_value # w
  for i in range(1,n_nodes):
    dy = initial_guess[i] - initial_guess[i-1]
    dx = initial_guess[n_nodes+i] - initial_guess[n_nodes+i-1]
    initial_guess[sidx+i-1] = ( initial_guess[3*n_nodes+i] - initial_guess[3*n_nodes+i-1] ) / interval_value # ua
    dw = ( initial_guess[4*n_nodes+i] - initial_guess[4*n_nodes+i-1] )
    initial_guess[sidx+n_nodes+i-1] = dw / interval_value # uw

  return initial_guess
