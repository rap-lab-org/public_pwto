

import pickle
import heapq as hpq
import numpy as np
import copy

import emoa_py_api as emoa

def SavePickle(data, file_path):
  pickle_out = open(file_path,"wb")
  pickle.dump(data, pickle_out)
  pickle_out.close()
  return

def LoadPickle(file_path):
  pickle_in = open(file_path,"rb")
  return pickle.load(pickle_in)

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
  return np.array(out)+0.5


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


class PrioritySet(object):
  """
  priority queue, min-heap
  """
  def __init__(self):
    """
    no duplication allowed
    """
    self.heap_ = []
    self.set_ = set()
  def add(self, pri, d):
    """
    will check for duplication and avoid.
    """
    if not d in self.set_:
        hpq.heappush(self.heap_, (pri, d))
        self.set_.add(d)
  def pop(self):
    """
    impl detail: return the first(min) item that is in self.set_
    """
    pri, d = hpq.heappop(self.heap_)
    while d not in self.set_:
      pri, d = hpq.heappop(self.heap_)
    self.set_.remove(d)
    return pri, d
  def size(self):
    return len(self.set_)
  def print(self):
    print(self.heap_)
    print(self.set_)
    return
  def remove(self, d):
    """
    implementation: only remove from self.set_, not remove from self.heap_ list.
    """
    if not d in self.set_:
      return False
    self.set_.remove(d)
    return True

def lexSortResult( res_dict ):
  """
  """
  pq = PrioritySet()
  for k in res_dict["costs"]:
    pq.add(tuple(res_dict["costs"][k]), k)
  k_list = list()
  while pq.size() > 0:
    cvec, k = pq.pop()
    print(" cvec = ", cvec, " k = ", k)
    k_list.append(k)
  new_cdict = dict()
  new_pdict = dict()
  idx = 0
  for k in k_list:
    new_cdict[idx] = res_dict["costs"][k]
    new_pdict[idx] = res_dict["paths"][k]
    idx += 1
  res_dict["paths"] = new_pdict
  res_dict["costs"] = new_cdict
  return 

def normalize(cmat):
  """
  """
  m = np.min(cmat)
  M = np.max(cmat)
  if m == M:
    return cmat
  res_mat = (cmat-m) / (M-m) * 1000 # equivalent to keep 3 digits float number.
  return res_mat

def scalarizeSolveEach(c_list, w, folder, emoa_exe, res_file, vo, vd, tlimit):
  """
  """
  scalarized_mat = np.zeros(c_list[0].shape)
  for i in range(len(c_list)):
    scalarized_mat += w[i]*normalize(c_list[i])
  res_dict = emoa.runEMOA([scalarized_mat], folder, emoa_exe, res_file, vo, vd, tlimit)
  return res_dict

def scalarizeSolve(c_list, n_weight, folder, emoa_exe, res_file, vo, vd, tlimit):
  """
  """
  out_dict = dict()
  w1_list = np.linspace(0,1,n_weight)
  out_dict['paths'] = dict()
  out_dict['costs'] = dict()
  idx = 0
  for w1 in w1_list:
    w2 = 1-w1
    res = scalarizeSolveEach(c_list, np.array([w1,w2]), folder, emoa_exe, res_file, vo, vd, tlimit)
    if len(res['paths']) > 1:
      sys.exit("[ERROR] Scalar map has more than one solution.")
    for k in res['paths']:
      out_dict['paths'][idx] = res['paths'][k] # there should be only one solution.
      # out_dict['costs'][idx] = recoverPathCost(c_list, res['paths'][k]) # there should be only one solution.
    idx += 1
  return out_dict

def pointHdfMinPart(nxt, v, p):
  """
  find the nearest dist between a point and a path.
  """
  vx = v%nxt
  vy = int(v/nxt)
  pathx = p%nxt
  pathy = np.floor(p/nxt)
  d = np.abs(pathx - vx) + np.abs(pathy-vy)
  return np.min(d)

def pathHdf(nxt,p1,p2):
  """
  given two path, return the Hausdorf distance.
  """
  dmax1 = 0
  for v in p1:
    d = pointHdfMinPart(nxt,v,p2)
    if d > dmax1:
      dmax1 = d
  dmax2 = 0
  for v in p2:
    d = pointHdfMinPart(nxt,v,p1)
    if d > dmax2:
      dmax2 = d
  return max(dmax1, dmax2)

def theta2quat(theta):
  """
  return q = [w,x,y,z]
  """
  if hasattr(theta, "__len__"):
    q = np.zeros((len(theta),4))
    q[:,0] = np.cos(theta/2)
    q[:,1] = 0
    q[:,1] = 0
    q[:,3] = np.sin(theta/2)
    return q
  else:
    q = np.zeros(4)
    q[0] = np.cos(theta/2)
    q[1] = 0
    q[2] = 0
    q[3] = np.sin(theta/2)
    return q