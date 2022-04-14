
import numpy as np
import cyipopt
import sys

import context
import dirCol as dc
import obstacle as obs
import dynamics as dyn

import matplotlib.pyplot as plt

def main1():
  
  # self.w = 100
  # self.R = np.identity(2)*0.1
  # self.Q = np.identity(5)*0.1
  # self.Q[2:5,2:5] = np.zeros([3,3])
  # self.Qf = np.identity(5)*1000
  
  np.set_printoptions(precision=3, suppress=True)

  xo = np.array([1, 3, 0, 0, 0])

  N = 11
  Uinit = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]])

  Xinit = np.zeros([N,5])
  Xinit[0,:] = xo
  dt = 0.2
  for k in range(len(Uinit)):
    u = Uinit[k]
    Xinit[k+1,:] = dyn.fd_ddc2(Xinit[k,:],u,dt)
  xf = Xinit[N-1,:]
  xf[2:5] = 0

  Ulb = np.array([-1.5,-1.5])
  Uub = np.array([ 1.5, 1.5])

  obs_pos_array = np.array([[-1,-1]])
  obss = obs.ObstSet( obs_pos_array )

  nlp_prob = dc.NLP_ddc2(xo, xf, N, dt, Ulb, Uub, Xinit, Uinit, obss, w=100.0)

  Zinit = nlp_prob.toZ(nlp_prob.Xinit, nlp_prob.Uinit)

  zlb = nlp_prob.Zlb()
  print("zlb = ", zlb)
  zub = nlp_prob.Zub()
  print("zub = ", zub)

  nlp = cyipopt.Problem(
     n=nlp_prob.lenZ,
     m=nlp_prob.lenC,
     problem_obj=nlp_prob,
     lb=zlb,
     ub=zub,
     cl=np.ones(nlp_prob.lenC)*(-0.01),
     cu=np.ones(nlp_prob.lenC)*(0.01),
  )

  nlp.add_option('mu_strategy', 'adaptive')
  # nlp.add_option('tol', 1e-3)
  nlp.add_option('max_iter', 1000)

  Zsol, info = nlp.solve(Zinit)

  print("================================================")
  print(Zsol)
  print(info)

  print(" constr eval = ", nlp_prob.constraints(Zsol) )
  print(" obj eval = ", nlp_prob.objective(Zsol) )

  # print("Z = ", Zsol)
  # print("grad = ", nlp_prob.gradient(Zsol))


  # fig = plt.figure(figsize=(3,3))
  fig = plt.figure()

  xy_tj = nlp_prob.getTrajXY(Zinit)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"ro--")

  xy_tj = nlp_prob.getTrajXY(Zsol)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"bo--")

  plt.show()

  return


def main2():
  
  np.set_printoptions(precision=3, suppress=True)

  N = 11
  dt = 0.1
  Xinit = np.array([[1,3,0,0,0],
                    [1,3,0,0,0],
                    [2,3,0,0,0],
                    [3,3,0,0,0],
                    [4,3,0,0,0],
                    [5,3,0,0,0],
                    [5,4,0,0,0],
                    [6,5,0,0,0],
                    [6,6,0,0,0],
                    [6,7,0,0,0],
                    [6,8,np.pi/2,0,0]])*0.2
  xo = Xinit[0,:]
  xf = Xinit[N-1,:]
  Uinit = np.zeros([N-1,2])

  print("================================================")
  print("Xinit = ",Xinit)

  Ulb = np.array([-1.5,-1.5])
  Uub = np.array([ 1.5, 1.5])

  obs_pos_array = np.array([[0.7, 1.1],[0.8, 1.2]])
  obss = obs.ObstSet( obs_pos_array )

  nlp_prob = dc.NLP_ddc2(xo, xf, N, dt, Ulb, Uub, Xinit, Uinit, obss, 100.0)

  Zinit = nlp_prob.toZ(nlp_prob.Xinit, nlp_prob.Uinit)

  zlb = nlp_prob.Zlb()
  print("zlb = ", zlb)
  zub = nlp_prob.Zub()
  print("zub = ", zub)

  nlp = cyipopt.Problem(
     n=nlp_prob.lenZ,
     m=nlp_prob.lenC,
     problem_obj=nlp_prob,
     lb=zlb,
     ub=zub,
     cl=np.ones(nlp_prob.lenC)*(-0.01),
     cu=np.ones(nlp_prob.lenC)*(0.01),
  )


  nlp.add_option('max_iter', 1000)

  Zsol, info = nlp.solve(Zinit)

  print("================================================")
  print(Zsol)
  print(info)

  print(" constr eval = ", nlp_prob.constraints(Zsol) )
  print(" obj eval = ", nlp_prob.objective(Zsol) )

  print(" pointCost = ", obss.pointCost(np.array([1.1, 0.7])))
  print(" pointCost = ", obss.pointCost(np.array([0.7, 1.1])))

  fig = plt.figure()

  xx = np.linspace(0,2,num=100)
  yy = np.linspace(0,2,num=100)
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  pf = obss.potentialField(2,2,100)
  print(pf)

  plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')
  plt.plot([1.1],[0.7],'y*')

  xy_tj = nlp_prob.getTrajXY(Zinit)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"ro--")

  xy_tj = nlp_prob.getTrajXY(Zsol)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"bo--")

  plt.show()

  return

def main3():
  
  np.set_printoptions(precision=3, suppress=True)

  N = 11
  dt = 0.1
  Xinit = np.array([[0.2, 0.6, 0, 0, 0],
                    [0.2 ,0.6, 0, 0, 0],
                    [0.4, 0.6, 0, 0, 0],
                    [0.6 ,0.6, 0, 0, 0],
                    [0.8, 0.6, 0, 0, 0],
                    [1.0, 0.6, 0, 0, 0],
                    [1.2, 0.6, 0, 0, 0],
                    [1.2, 0.8, np.pi/2, 0, 0],
                    [1.2, 1.0, np.pi/2, 0, 0],
                    [1.2, 1.2, np.pi/2, 0, 0],
                    [1.2, 1.4, np.pi/2 ,0, 0]])
  Xinit[10,2] = np.pi
  xo = Xinit[0,:]
  xf = Xinit[N-1,:]
  Uinit = np.zeros([N-1,2])

  print("================================================")
  print("Xinit = ",Xinit)

  Ulb = np.array([-1.5,-1.5])
  Uub = np.array([ 1.5, 1.5])

  obs_pos_array = np.array([[0.5,1.2],[0.6,1.2],[0.7, 1.2],[0.8,1.2], [0.7,0.8],[0.8,0.8],[0.9,0.8], [0.6,1.0], [0.7,1.0],[0.8,1.0], [0.8,0.9]])
  obss = obs.ObstSet( obs_pos_array )

  nlp_prob = dc.NLP_ddc2(xo, xf, N, dt, Ulb, Uub, Xinit, Uinit, obss, 100000.0)

  Zinit = nlp_prob.toZ(nlp_prob.Xinit, nlp_prob.Uinit)

  zlb = nlp_prob.Zlb()
  print("zlb = ", zlb)
  zub = nlp_prob.Zub()
  print("zub = ", zub)

  nlp = cyipopt.Problem(
     n=nlp_prob.lenZ,
     m=nlp_prob.lenC,
     problem_obj=nlp_prob,
     lb=zlb,
     ub=zub,
     cl=np.ones(nlp_prob.lenC)*(-0.01),
     cu=np.ones(nlp_prob.lenC)*(0.01),
  )


  nlp.add_option('max_iter', 1000)
  nlp.add_option("hessian_approximation", "limited-memory")

  Zsol, info = nlp.solve(Zinit)

  print("================================================")
  print(Zsol)
  print(info)

  print(" constr eval = ", nlp_prob.constraints(Zsol) )
  print(" obj eval = ", nlp_prob.objective(Zsol) )

  print(" pointCost = ", obss.pointCost(np.array([1.1, 0.7])))
  print(" pointCost = ", obss.pointCost(np.array([0.7, 1.1])))

  fig = plt.figure()

  xx = np.linspace(0,2,num=100)
  yy = np.linspace(0,2,num=100)
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
  pf = obss.potentialField(2,2,100)
  print(pf)

  plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')
  plt.plot([1.1],[0.7],'y*')

  xy_tj = nlp_prob.getTrajXY(Zinit)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"ro--")

  xy_tj = nlp_prob.getTrajXY(Zsol)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"bo--")

  plt.show()

  return

if __name__ == "__main__":
  # main1() 
  # main2() 
  main3() 