
import numpy as np
from scipy.stats import multivariate_normal
import time

class ObstSet:
  
  def __init__(self, obs_pos_array, cov=np.array([[0.001, 0],[0, 0.001]])):
    """
    obs_pos_array should be a numpy 2d array.
    obs_pos_array = [[x1,y1], [x2,y2], ..., [xn,yn]] - positions of obstacles.
    """
    self.cov = cov
    self.mus = obs_pos_array
    self.mvn_list = list()
    for k in range(len(obs_pos_array)):
      self.mvn_list.append( multivariate_normal(obs_pos_array[k],self.cov) )
    return

  def pointCost(self, p):
    """
    p=[x,y]
    """
    cost = 0

    for k in range(len(self.mvn_list)):
      cost += self.mvn_list[k].pdf(p)
    return cost

  def arrayCost(self, pList):
    """
    pList = [[p1x,p1y],[p2x,p2y],...,[pnx,pny]]
    """
    cost = 0
    for k in range(len(self.mvn_list)):
      cost += np.sum( self.mvn_list[k].pdf(pList) )
    return cost

  def pointGrad(self, p):
    """
    """
    grad = 0
    for k in range(len(self.mvn_list)):
      grad -= self.mvn_list[k].pdf(p) * np.dot(self.cov, (p-self.mus[k]))
    return grad

  def arrayGrad(self, pList):
    """
    """
    grad = np.zeros(pList.shape)
    for k in range(len(self.mvn_list)):
      a = np.dot((pList-self.mus[k]), self.cov)
      b = self.mvn_list[k].pdf(pList)
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

  
  tstart = time.perf_counter()
  print( obss.arrayCost([[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[9,3]]) )
  print(' time used = ', time.perf_counter() - tstart)

  tstart = time.perf_counter()
  ptGrad = obss.pointGrad(np.array([1,3]) )
  print( ptGrad )
  ptGrad = obss.pointGrad(np.array([2,3]) )
  print( ptGrad )
  ptGrad = obss.pointGrad(np.array([3,3]) )
  print( ptGrad )
  print(' time used = ', time.perf_counter() - tstart)

  tstart = time.perf_counter()
  arrayGrad = obss.arrayGrad(np.array([[1,3],[2,3],[3,3]]) )
  print( arrayGrad )
  print(' time used = ', time.perf_counter() - tstart)

