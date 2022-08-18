
import numpy as np
from scipy.stats import multivariate_normal
import time
import random

FIX_COV_VAL = 2*1e-4

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

    random.seed(0) # fixed with seed 0.

    for k in range(len(obs_pos_array)):

      self.rand_normheight.append(random.random())

    for k in range(len(obs_pos_array)):

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
      cost += np.sum( self.mvn_list[k].pdf(pList) + self.flat_offset) 
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
    compute gradient, for DirCol.
    """
    grad = np.zeros(pList.shape)
    for k in range(len(self.mvn_list)):
      a = np.dot((pList-self.mus[k]), self.cov)
      b = self.mvn_list[k].pdf(pList)  # original 
      grad -= np.reshape(b, [len(b),1]) * a
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

    return p

