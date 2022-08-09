
import numpy as np
from scipy.stats import multivariate_normal
import time
import random

# FIX_COV_VAL = 2*1e-4

FIX_COV_VAL = 1e-3

class ObstSet:
  
  def __init__(self, obs_pos_array, fix_cov_val=FIX_COV_VAL):
    """
    obs_pos_array should be a numpy 2d array.
    obs_pos_array = [[x1,y1], [x2,y2], ..., [xn,yn]] - positions of obstacles.
    """
    self.cov = np.array([[fix_cov_val,0],[0,fix_cov_val]])
    self.mus = obs_pos_array
    self.mvn_list = list()
    self.rand_normheight = []
    self.flat_offset = 0

    random.seed(0) # Richard: fixed with seed 0.

    for k in range(len(obs_pos_array)):

      self.rand_normheight.append(random.random())

      # # up and down
      # self.rand_normheight.append(random.choice((-1, 1))*random.random())


    for k in range(len(obs_pos_array)):

      # self.mvn_list.append( multivariate_normal(obs_pos_array[k],self.cov) )  

      ### use the following two lines for random cov, basic cov 7*1e-4 for 16*16
      rand_cov_co = 1.0 *(self.rand_normheight[k] - 0.5) +1
      self.mvn_list.append( multivariate_normal(obs_pos_array[k],rand_cov_co*self.cov) )

    return


  def pointCostSingle(self,k,p):
    """
    Random normal height
    p=[x,y]
    """
    cost_single = self.mvn_list[k].pdf(p) + self.flat_offset

    # random height magnitude
    # cost_single = self.rand_normheight[k] * self.mvn_list[k].pdf(p) + self.flat_offset

    return cost_single


  def pointCost(self, p):
    """
    p=[x,y]
    """
    cost = 0

    for k in range(len(self.mvn_list)):
      cost += self.pointCostSingle(k, p)
    return cost


  def arrayCost(self, pList):
    """
    pList = [[p1x,p1y],[p2x,p2y],...,[pnx,pny]]
    """
    cost = 0
    for k in range(len(self.mvn_list)):

      cost += np.sum( self.mvn_list[k].pdf(pList) + self.flat_offset)  # original 

      # cost_single = 0
      # for p in pList:
      #   cost_single += self.pointCostSingle(k, p)  
      # cost += cost_single                         # new

    return cost

  def pointGrad(self, p):
    """
    """
    grad = 0
    for k in range(len(self.mvn_list)):
      grad -= self.mvn_list[k].pdf(p) * np.dot(self.cov, (p-self.mus[k]))
      grad -= self.pointCostSingle(k, p) * np.dot(self.cov, (p-self.mus[k]))

    return grad

  def arrayGrad(self, pList):
    """
    """
    grad = np.zeros(pList.shape)
    for k in range(len(self.mvn_list)):
      a = np.dot((pList-self.mus[k]), self.cov)

      b = self.mvn_list[k].pdf(pList)  # original 
      # b = []
      # for p in pList:
      #   b.append(self.pointCostSingle(k, p))  # new

      grad -= np.reshape(b, [len(b),1]) * a

    # for k in range(len(self.mvn_list)):
    #   for idx in range(len(pList)):
    #     grad[idx,:] += self.pointGrad(pList[idx])
    return grad


  def potentialField(self, xmax, ymax, npix):
    """
    workspace has range [0,xmax] x [0,ymax].
    """
    xg, yg = np.meshgrid(
      np.linspace(0, xmax, npix), np.linspace(0, ymax, npix), indexing='ij'
    )
    flatgrid = np.stack([xg.flatten(), yg.flatten()], axis=1)

    p = np.zeros((npix, npix))
    # for k in range(len(self.mus)):
    for ix in range(npix):
      for iy in range(npix):
        x = ix*(xmax/npix)
        y = iy*(ymax/npix)
        p[ix,iy] += self.pointCost(np.array([x,y]))

      # print(" mu = ", self.mus[k])
      # p += multivariate_normal.pdf(flatgrid, mean=self.mus[k], cov=self.cov).reshape((npix, npix))
    # return p / np.sum(p)
    # print(p)
    return p



if __name__ == "__main__":

  obss = ObstSet([[1,0],[2,0],[3,0]])
  # obss = ObstSet([[1,0],[3,0]])

  tstart = time.perf_counter()
  cost = 0
  cost += obss.pointCost([1,3]) 
  cost += obss.pointCost([2,3]) 
  cost += obss.pointCost([3,3]) 
  cost += obss.pointCost([4,3]) 
  cost += obss.pointCost([5,3]) 
  cost += obss.pointCost([6,3]) 
  cost += obss.pointCost([7,3]) 
  cost += obss.pointCost([8,3]) 
  cost += obss.pointCost([9,3]) 
  print(cost)
  print(' time used = ', time.perf_counter() - tstart)

  
#   tstart = time.perf_counter()
#   print( obss.arrayCost([[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[9,3]]) )
#   print(' time used = ', time.perf_counter() - tstart)

#   tstart = time.perf_counter()
#   ptGrad = obss.pointGrad(np.array([1,3]) )
#   print( ptGrad )
#   ptGrad = obss.pointGrad(np.array([2,3]) )
#   print( ptGrad )
#   ptGrad = obss.pointGrad(np.array([3,3]) )
#   print( ptGrad )
#   print(' time used = ', time.perf_counter() - tstart)

#   tstart = time.perf_counter()
#   arrayGrad = obss.arrayGrad(np.array([[1,3],[2,3],[3,3]]) )
#   print( arrayGrad )
#   print(' time used = ', time.perf_counter() - tstart)




# class ObstSet:
  
#   def __init__(self, obs_pos_array, rObst, rSafe, k):
#     """
#     obs_pos_array should be a numpy 2d array.
#     obs_pos_array = [[x1,y1], [x2,y2], ..., [xn,yn]] - positions of obstacles.
#     r = radius
#     k = cost scaling factor
#     """
#     self.obs_pos_array = obs_pos_array
#     self.rObst = rObst
#     self.rSafe = rSafe
#     self.k = k
#     return

#   def _dist(self,p):
#     diff = self.obs_pos_array - p
#     return np.hypot(diff[:,0],diff[:,1])


#   def pointCost(self, p):
#     """
#     p=[x,y]
#     """
#     cost = 0

#     # diff = self.obs_pos_array - p
#     # dist = np.hypot(diff[:,0],diff[:,1])
    
#     dist = self._dist(p)

#     # dist[dist < self.rObst] = self.rObst
#     minDist = dist[dist == np.min(dist)]
#     minDist[minDist < self.rObst] = self.rObst

#     return self.k*np.sum( 1 / minDist[minDist < self.rSafe] )

#   def arrayCost(self, pList):
#     """
#     pList = [[p1x,p1y],[p2x,p2y],...,[pnx,pny]]
#     """
#     cost = 0
#     for k in range(len(pList)):
#       cost += self.pointCost(pList[k])
#     return cost

#   def pointGrad(self, p):
#     """
#     """

#     dist = self._dist(p)

#     dist[dist < self.rObst] = self.rObst
#     selector = dist < self.rSafe
#     near_dist = dist[selector]
#     near_obs = self.obs_pos_array[selector]

#     # select1 = (dist == np.min(dist))
#     # dist[dist < self.rObst] = self.rObst

#     # select2 = dist < self.rSafe
#     # near_dist = dist[select1&select2]
#     # near_obs = self.obs_pos_array[select1&select2]

#     dq = near_obs - p
#     temp = self.k / near_dist
#     grad_array = np.multiply( dq, temp[:,np.newaxis])
#     return np.sum(grad_array, axis=0)

#   def arrayGrad(self, pList):
#     """
#     """
#     grad = np.zeros(pList.shape)
#     for k in range(len(pList)):
#       grad[k,:] = self.pointGrad(pList[k])
#     return grad

#   def potentialField(self, xmax, ymax, npix):
#     """
#     workspace has range [0,xmax] x [0,ymax].
#     """
#     xg, yg = np.meshgrid(
#       np.linspace(0, xmax, npix), np.linspace(0, ymax, npix), indexing='ij'
#     )
#     flatgrid = np.stack([xg.flatten(), yg.flatten()], axis=1)

#     p = np.zeros((npix, npix))
#     # for k in range(len(self.mus)):
#     for ix in range(npix):
#       for iy in range(npix):
#         x = ix*(xmax/npix)
#         y = iy*(ymax/npix)
#         p[ix,iy] += self.pointCost(np.array([x,y]))
#       # print(" mu = ", self.mus[k])
#       # p += multivariate_normal.pdf(flatgrid, mean=self.mus[k], cov=self.cov).reshape((npix, npix))
#     # return p / np.sum(p)
#     print("pf = ", p, "min max = ", np.min(p), np.max(p))
#     print("pf = ", p, "min max median = ", np.min(p), np.max(p), np.median(p))
#     return p


# # ===


# class ObstSet:
  
#   def __init__(self, obs_pos_array, rObst, rSafe, k):
#     """
#     obs_pos_array should be a numpy 2d array.
#     obs_pos_array = [[x1,y1], [x2,y2], ..., [xn,yn]] - positions of obstacles.
#     r = radius
#     k = cost scaling factor
#     """
#     self.obs_pos_array = obs_pos_array
#     self.rObst = rObst
#     self.rSafe = rSafe
#     self.k = k
#     return

#   def _dist(self,p):
#     diff = self.obs_pos_array - p
#     d = np.hypot(diff[:,0],diff[:,1]) # sqrt(x**2,y**2)
#     # d = np.min(np.array([np.abs(diff[:,0]),np.abs(diff[:,1])]), axis=0)
#     d[d < self.rObst] = self.rObst
#     return d

#   def pointCost(self, p):
#     """
#     p=[x,y]
#     """
#     dist = self._dist(p)
#     min_d = np.min(dist)
#     if min_d > self.rSafe:
#       return 0
#     return 1/2 *self.k*(1.0/min_d - 1.0/self.rSafe)**2

#   # def pointCost(self, p):
#   #   """
#   #   p=[x,y]
#   #   """
#   #   cost = 0

#   #   # diff = self.obs_pos_array - p
#   #   # dist = np.hypot(diff[:,0],diff[:,1])
#   #   # # dist[dist < self.rObst] = self.rObst


#   #   # minDist = dist[dist == np.min(dist)]
#   #   # minDist[minDist < self.rObst] = self.rObst

#   #   # return self.k*np.sum( 1 / minDist[minDist < self.rSafe] )

#   #   dist = self._dist(p)
#   #   dist[dist < self.rObst] = self.rObst
#   #   effective_dist = dist[dist < self.rSafe]
#   #   if len(effective_dist) == 0:
#   #     return 0
#   #   return self.k*np.sum(1/effective_dist) / len(effective_dist) # take average


#   def arrayCost(self, pList):
#     """
#     pList = [[p1x,p1y],[p2x,p2y],...,[pnx,pny]]
#     """
#     cost = 0
#     for k in range(len(pList)):
#       cost += self.pointCost(pList[k])
#     return cost

#   def pointGrad(self, p):
#     """
#     """

#     # diff = self.obs_pos_array - p
#     # dist = np.hypot(diff[:,0],diff[:,1])

#     dist = self._dist(p)
#     dmin = np.min(dist)

#     # dist[dist < self.rObst] = self.rObst
#     # selector = dist < self.rSafe
#     # near_dist = dist[selector]
#     # near_obs = self.obs_pos_array[selector]

#     select1 = (dist == dmin)
#     select2 = (dist < self.rSafe)
#     near_dist = dist[select1&select2]
#     near_obs = self.obs_pos_array[select1&select2]

#     if len(near_obs) == 0:
#       return 0
#     elif len(near_obs) == 1:
#       p_ref = near_obs[0]
#       return -self.k*(1.0/dmin - 1.0/self.rSafe) * (1/dmin) * (1/dmin) * (1/dmin) * (p-p_ref)
#     else:
#       grad = np.zeros(2)
#       for p_ref in near_obs:
#         grad += -self.k*(1.0/dmin - 1.0/self.rSafe) * (1/dmin) * (1/dmin) * (1/dmin) * (p-p_ref)
#       return grad

#     # dq = near_obs - p
#     # temp = self.k / near_dist
#     # grad_array = np.multiply( dq, temp[:,np.newaxis])
#     # return np.sum(grad_array, axis=0)

#   def arrayGrad(self, pList):
#     """
#     """
#     grad = np.zeros(pList.shape)
#     for k in range(len(pList)):
#       grad[k,:] = self.pointGrad(pList[k])
#     return grad

#   def potentialField(self, xmax, ymax, npix):
#     """
#     workspace has range [0,xmax] x [0,ymax].
#     """
#     xg, yg = np.meshgrid(
#       np.linspace(0, xmax, npix), np.linspace(0, ymax, npix), indexing='ij'
#     )
#     flatgrid = np.stack([xg.flatten(), yg.flatten()], axis=1)

#     p = np.zeros((npix, npix))
#     # for k in range(len(self.mus)):
#     for ix in range(npix):
#       for iy in range(npix):
#         x = ix*(xmax/npix)
#         y = iy*(ymax/npix)
#         p[ix,iy] += self.pointCost(np.array([x,y]))
#       # print(" mu = ", self.mus[k])
#       # p += multivariate_normal.pdf(flatgrid, mean=self.mus[k], cov=self.cov).reshape((npix, npix))
#     # return p / np.sum(p)
#     print("pf = ", p, "min max = ", np.min(p), np.max(p))
#     print("pf = ", p, "min max median = ", np.min(p), np.max(p), np.median(p))
#     return p


# def map1():
#   """
#   10x10
#   """
#   out = np.zeros((10,10))
#   out[0:7,3] = 1
#   out[3:10,5] = 1
#   out[0:7,7] = 1
#   return out