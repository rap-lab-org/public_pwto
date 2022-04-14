
import numpy as np
import cyipopt
import sys

import context
import dirCol as dc
import obstacle as obs
import dynamics as dyn

import matplotlib.pyplot as plt

def main1():
  
  # xo = np.array([3, 1, 0, 0, 0])
  # xf = np.array([3, 5, -np.pi, 0, 0])
  
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

  # obs_pos_array = np.array([[1,2],[2,2],[3,2],[3,4],[4,4],[5,4]])
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

  # print("***********")
  # print("diff = ", Zsol-Zinit)
  # print("state 0 = ",  nlp_prob.Zx(Zsol,0) ) 
  # print("state 1 = ",  nlp_prob.Zx(Zsol,1) ) 

  xy_tj = nlp_prob.getTrajXY(Zsol)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"bo--")

  plt.show()


  return

def main2():
  
  # xo = np.array([3, 1, 0, 0, 0])
  # xf = np.array([3, 5, -np.pi, 0, 0])
  
  np.set_printoptions(precision=3, suppress=True)

  # xo = np.array([1, 3, 0, 0, 0])

  # N = 4
  # Uinit = np.array([[1,1],[0,0],[-1,-1]])

  # N = 11
  # Uinit = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]])

  # Xinit = np.zeros([N,5])
  # Xinit[0,:] = xo
  # dt = 0.1
  # for k in range(len(Uinit)):
  #   u = Uinit[k]
  #   Xinit[k+1,:] = dyn.fd_ddc2(Xinit[k,:],u,dt)
  # xf = Xinit[N-1,:]
  # xf[2:5] = 0

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
                    [6,8,np.pi/2,0,0]])
  xo = Xinit[0,:]
  xf = Xinit[N-1,:]
  Uinit = np.zeros([N-1,2])

  print("================================================")
  print("Xinit = ",Xinit)

  Ulb = np.array([-1.5,-1.5])
  Uub = np.array([ 1.5, 1.5])
  
  # Xinit = np.zeros([N,5])
  # for k in range(5):
  #   Xinit[:,k] = np.linspace(xo[k],xf[k],N)
  # Xinit[:,3] = 4.0/100
  # Xinit[:,4] = 0
  # Uinit = np.ones([N-1,2])

  # obs_pos_array = np.array([[1,2],[2,2],[3,2],[3,4],[4,4],[5,4]])
  obs_pos_array = np.array([[1,1],[1,5]])
  obss = obs.ObstSet( obs_pos_array )

  nlp_prob = dc.NLP_ddc2(xo, xf, N, dt, Ulb, Uub, Xinit, Uinit, obss, w=100.0)

  Zinit = nlp_prob.toZ(nlp_prob.Xinit, nlp_prob.Uinit)

  # print(" constr eval = ", nlp_prob.constraints(Zinit) )
  # print(" obj eval = ", nlp_prob.objective(Zinit) )

  # print("Zinit = ", Zinit)
  # print("grad = ", nlp_prob.gradient(Zinit))
  # jac = nlp_prob.jacobian(Zinit)
  # print("jac = ", jac.shape)

  # for k in range(N):
  #   print("Zx[k] = ", nlp_prob.Zx(Zinit,k))
  #   print("Zu[k] = ", nlp_prob.Zu(Zinit,k))

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


  # nlp.add_option('mu_strategy', 'adaptive')
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

  # print("***********")
  # print("diff = ", Zsol-Zinit)
  # print("state 0 = ",  nlp_prob.Zx(Zsol,0) ) 
  # print("state 1 = ",  nlp_prob.Zx(Zsol,1) ) 

  xy_tj = nlp_prob.getTrajXY(Zsol)
  plt.plot(xy_tj[:,0],xy_tj[:,1],"bo--")

  plt.show()


  return

if __name__ == "__main__":
  main1() 