
import numpy as np
import sys

import obstacle as obs
import dynamics as dyn

class NLP_ddc2:
  """
  """
  def __init__(self, xo, xf, N, dt, Ulb, Uub, Xinit, Uinit, obss, w=100.0):
    """
    N = length of trajectory, index range from [0,1,...,N-1]
    obss = obstacle set
    Ulb,Uub, lower and upper bound on control input. Each should be of vector of length 2.
    """
    self.xo = xo
    self.xf = xf
    self.obss = obss
    self.n = 5 # state size, (x,y,theta,v,w)
    self.m = 2 # control size, (v_dot, w_dot)
    self.N = N # number of points in the trajectory. There are N-1 collocation points.
    self.dt = dt
    self.obss = obss
    self.Xinit = Xinit
    self.Uinit = Uinit
    self.Ulb = Ulb
    self.Uub = Uub

    self.w = w
    self.R = np.identity(2)*0.05
    self.Qf = np.identity(5)

    # self.lenZ = self.n+(self.N-1)*(self.m+self.n) # num of decision variables, [x0,u0,x1,u1,x2,...,nN-2,xN-1]
    self.lenZ = (self.N)*(self.m+self.n) # num of decision variables, [x0,u0,x1,u1,x2,...,nN-2,xN-1,uN-1]
    self.lenC = self.n + (self.N-1)*self.n + self.n # num of constraints, x0 is also added as constraints here.

    self.Xm = np.zeros([self.N-1,self.n])
    self.Um = np.zeros([self.N-1,self.m])
    self.fXm = np.zeros([self.N-1,self.n])

    return

  def toZ(self,X,U):
    """
    Given a complete trajectory (X,U) with len(X)=N, len(U) = N-1.
    Return a corresponding array Z = [x0,u0,x1,u1,x2,...,nN-2,xN-1] of length (N-1)*(m+n)
    """
    if len(U) != self.N-1 or len(X) != self.N:
      sys.exit("[ERROR] NLP_ddc2, toZ, input length error!")
    Z = np.zeros( self.lenZ )
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      Z[a : a+self.n] = X[k]
      Z[a+self.n : a+self.n+self.m] = U[k]
    a = (self.N-1)*(self.n+self.m)
    Z[a : a+self.n] = X[self.N-1]
    Z[a+self.n : a+self.n+self.m] = np.zeros(2) # last control is assumed to be zero.
    return Z

  # def setZx(self,k,x):
  #   if k == 0 or k > N-1:
  #     sys.exit("[ERROR] NLP_ddc2, setZx, out of range!")
  #   a = (k-1)*(self.n+self.m)
  #   self.Z[a+self.m:a+self.m+self.n] = x
  #   return

  # def setZu(self,k,u):
  #   if k > N-2:
  #     sys.exit("[ERROR] NLP_ddc2, setZu, out of range!")
  #   a = (k)*(self.n+self.m)
  #   self.Z[a:a+self.m] = u
  #   return

  def Zx(self,Z,k):
    """
    input k range [0,N-1]. range length = N.
    """
    a = (k)*(self.n+self.m)
    if k > self.N-1:
      sys.exit("[ERROR] NLP_ddc2, Zx, out of range!")
    return Z[a : a+self.n]

  def Zu(self,Z,k):
    """
    input k range [0,N-2]. range length = N-1.
    """
    # print("Zu, input k = ",k)
    if k > self.N-1:
      sys.exit("[ERROR] NLP_ddc2, Zu, out of range!")
    # if k == self.N-1:
    #   return np.zeros(self.m) # assume last control input corresponding to xf is zero.
    a = (k)*(self.n+self.m)
    return Z[a+self.n : a+self.n+self.m]

  def getTrajXY(self,Z):
    """
    """
    tj = np.zeros([self.N,2])
    for k in range(self.N):
      x = self.Zx(Z,k)
      tj[k,:] = x[0:2]
    return tj

  def getTrajU(self,Z):
    """
    """
    # print("getTrajU")
    tj = np.zeros([self.N-1,self.m])
    for k in range(self.N-1):
      tj[k,:] = self.Zu(Z,k)
    # print("getTrajU done")
    return tj

  def objective(self,Z):
    """
    return the scalar objective value given decision variable Z.
    """

    # obst cost
    self.xy_tj = self.getTrajXY(Z) # cache the result for gradient usage.
    J1 = self.w*self.obss.arrayCost(self.xy_tj[1:])

    # control cost
    self.u_tj = self.getTrajU(Z) # cache the result for gradient usage.
    self.u_tj_R = np.dot(self.u_tj,self.R) # cache the result for gradient usage.
    J2 = 0.5*np.sum(self.u_tj_R*self.u_tj)

    # final state cost
    dxf = self.Zx(Z,self.N-1) - self.xf
    dx_Qf = np.dot(dxf,self.Qf)
    J3 = 0.5*np.dot(dx_Qf, dxf)

    # print("exit objective")
    # return J1+J2+J3
    return J2+J3

  def gradient(self, Z):
    """
    Returns the gradient of the objective with respect to Z
    """
    grad = np.zeros(self.lenZ)
    m = self.m
    n = self.n

    # # obst cost
    # self.xy_tj = self.getTrajXY(Z) # cache the result for gradient usage.
    # dJ1dXY = self.obss.arrayGrad(self.xy_tj) # include x0
    # # print("dJ1dXY shape = ", dJ1dXY.shape)
    # for k in range(self.N-1):
    #   a = (k)*(n+m)
    #   grad[a:a+2] += dJ1dXY[k]
    
    # control cost
    self.u_tj = self.getTrajU(Z) # cache the result for gradient usage.
    dJ2dU = np.dot(self.R,self.u_tj.T).T
    # print("dJ2dU shape = ", dJ2dU.shape)
    for k in range(self.N-1):
      a = (k)*(n+m)
      # grad[a+n:a+n+2] += dJ2dU[k]
      grad[a+n:a+n+2] += np.dot(self.R,self.u_tj[k])

    # final state cost
    dxf = self.Zx(Z,self.N-1) - self.xf
    dJ3dxf = np.dot(self.Qf, dxf)
    a = (self.N-1)*(n+m)
    grad[a:a+n] += dJ3dxf

    return grad

  # def hessian(self, x, lagrange, obj_factor):
  #   """
  #   return hessian
  #   """
  #   H = np.zeros()
  #   return

  def constraints(self, Z):
    """
    return the constraints
    """
    C = np.zeros(self.lenC)

    m = self.m
    n = self.n

    # evaluate dynamics at x0,x1,,.,xN-1 (totally N points)
    self.fXk = list() # of length N
    # print("constraints")
    for k in range(self.N):
      x = self.Zx(Z,k)
      u = self.Zu(Z,k)
      self.fXk.append( dyn.fc_ddc2(x,u) )
    
    # evalute collocation points
    self.Xm = np.zeros([self.N-1,n])
    self.Um = np.zeros([self.N-1,m])
    self.fXm = np.zeros([self.N-1,n])
    # print("constraints2")
    for k in range(self.N-1):
      self.Um[k] = 0.5*( self.Zu(Z,k) + self.Zu(Z,k+1) )
      self.Xm[k] = 0.5*( self.Zx(Z,k) + self.Zx(Z,k+1) ) + self.dt/8.0*(self.fXk[k] - self.fXk[k+1])
      self.fXm[k] = dyn.fc_ddc2(self.Xm[k], self.Um[k]) # at time t+self.dt/2.0

    # initial state
    C[0:n] = self.Zx(Z,0) - self.xo

    # dynamics constraints
    for k in range(self.N-1):
      a = n + k*n
      C[a:a+n] = ( self.Zx(Z,k) - self.Zx(Z,k+1) ) + self.dt/6.0 * (self.fXk[k] + 4*self.fXm[k] + self.fXk[k+1])

    # final state
    C[n+(self.N-1)*n : n+self.N*self.n] = self.Zx(Z,self.N-1) - self.xf

    return C

  def jacobian(self, Z):
    """
    Returns the Jacobian of the constraints with respect to Z.
    """
    # TODO, replace with sparse matrix

    jac = np.zeros([self.lenC, self.lenZ])

    m = self.m
    n = self.n

    # print("jacobian")

    Ak = list()
    Bk = list()
    for k in range(self.N):
      u = self.Zu(Z,k)
      Ak.append( dyn.fc_ddc2_A(self.Zx(Z,k), u) )
      Bk.append( dyn.fc_ddc2_B(self.Zx(Z,k), u) )

    # print("jacobian2")

    Am = list()
    Bm = list()
    for k in range(self.N-1):
      Am.append( dyn.fc_ddc2_A(self.Xm[k],self.Um[k]) )
      Bm.append( dyn.fc_ddc2_B(self.Xm[k],self.Um[k]) )

    In = np.identity(n)
    Im = np.identity(m)

    # initial state
    jac[0:n, 0:n] += In
    
    for k in range(self.N-1):
      row_idx = n+k*n
      col_idx = k*(m+n)
      # print(" prev, k = ", k, " col_idx = ", col_idx)
      
      dxm_dx1 = 0.5*In + self.dt/8.0 * Ak[k]
      jac[row_idx:row_idx+n, col_idx:col_idx+n] += In + self.dt/6*(Ak[k] + 4*np.dot(Am[k],dxm_dx1) )

      dxm_dx2 = 0.5*In - self.dt/8.0 * Ak[k]
      jac[row_idx:row_idx+n, col_idx+m+n:col_idx+m+n+n] += (-In) + self.dt/6*(Ak[k+1] + 4*np.dot(Am[k],dxm_dx2) )

      dum_du1 = 0.5*Im
      dxm_du1 = self.dt/8.0 * Bk[k]
      jac[row_idx:row_idx+n, col_idx+n:col_idx+n+m] += self.dt/6*(Bk[k] + 4*np.dot(Am[k],dxm_du1) + 4*np.dot(Bm[k],dum_du1) )

      # if k < self.N-2:
      dum_du2 = 0.5*Im
      dxm_du2 = -self.dt/8.0 * Bk[k+1]
      cidx = col_idx+m+n
      jac[row_idx:row_idx+n, col_idx+n+m+n:col_idx+n+m+n+m] += self.dt/6*(Bk[k+1] + 4*np.dot(Am[k],dxm_du2) + 4*np.dot(Bm[k],dum_du2) )

    # final state
    row_idx = n+(self.N-1)*n
    col_idx = (self.N-1)*(m+n)
    jac[row_idx:row_idx+n, col_idx:col_idx+n] += In

    # print("jacobian3")
    return jac


  def Zlb(self):
    """
    """
    lb = np.ones(self.lenZ)*(-1e9)
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      lb[a : a+self.m] = self.Ulb
    return lb


  def Zub(self):
    """
    """
    ub = np.ones(self.lenZ)*(1e9)
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      ub[a : a+self.m] = self.Uub
    return ub

  def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                 d_norm, regularization_size, alpha_du, alpha_pr,
                 ls_trials):
    """Prints information at every Ipopt iteration."""

    msg = "Objective value at iteration #{:d} is - {:g}"

    print(msg.format(iter_count, obj_value))

