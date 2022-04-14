
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
    self.R = np.identity(2)
    self.Qf = np.identity(5)

    self.lenZ = (self.N-1)*(self.m+self.n) # num of decision variables, [u0,x1,u1,x2,...,nN-2,xN-1]
    self.lenC = (self.N-1)*self.n + self.n # num of constraints, x0 is not added as constraints here. x0 is assumed to be the same.

    # self.Z = np.zeros( (self.N-1)*(self.n+self.m) ) # X[0]=xo is given, not part of the decision variables.

    self.Xm = np.zeros([self.N-1,self.n])
    self.Um = np.zeros([self.N-1,self.m])
    self.fXm = np.zeros([self.N-1,self.n])

    return

  def toZ(self,X,U):
    """
    Given a complete trajectory (X,U) with len(X)=N, len(U) = N-1.
    Return a corresponding array Z = [u0,x1,u1,x2,...,nN-2,xN-1] of length (N-1)*(m+n)
    """
    if len(U) != self.N-1 or len(X) != self.N:
      sys.exit("[ERROR] NLP_ddc2, toZ, input length error!")
    Z = np.zeros( self.lenZ )
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      Z[a : a+self.m] = U[k]
      Z[a+self.m : a+self.m+self.n] = X[k+1]
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
    if k == 0:
      return self.xo
    a = (k-1)*(self.n+self.m)
    if k > self.N-1:
      sys.exit("[ERROR] NLP_ddc2, Zx, out of range!")
    return Z[a+self.m : a+self.m+self.n]

  def Zu(self,Z,k):
    """
    input k range [0,N-2]. range length = N-1.
    """
    # print("Zu, input k = ",k)
    if k > self.N-2:
      print("error here.")
      sys.exit("[ERROR] NLP_ddc2, Zu, out of range!")
    a = (k)*(self.n+self.m)
    return Z[a : a+self.m]

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
    return J1+J2+J3

  def gradient(self, Z):
    """
    Returns the gradient of the objective with respect to Z
    """
    grad = np.zeros(self.lenZ)

    # obst cost
    self.xy_tj = self.getTrajXY(Z) # cache the result for gradient usage.
    dJ1dXY = self.obss.arrayGrad(self.xy_tj[1:]) # remove x0
    # print("dJ1dXY shape = ", dJ1dXY.shape)
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      grad[a+self.m:a+self.m+2] += dJ1dXY[k]
    
    # control cost
    self.u_tj = self.getTrajU(Z) # cache the result for gradient usage.
    dJ2dU = np.dot(self.R,self.u_tj.T).T
    # print("dJ2dU shape = ", dJ2dU.shape)
    for k in range(self.N-1):
      a = (k)*(self.n+self.m)
      grad[a:a+2] += dJ2dU[k]

    # final state cost
    dxf = self.Zx(Z,self.N-1) - self.xf
    dJ3dxf = np.dot(self.Qf, dxf.T).T
    a = (self.N-2)*(self.n+self.m)
    grad[a+self.m:a+self.m+self.n] += dJ3dxf

    return grad

  # def hessian(self, x, lagrange, obj_factor):
  #   """
  #   return hessian
  #   """

  #   H = obj_factor*np.array((
  #       (2*x[3], 0, 0, 0),
  #       (x[3],   0, 0, 0),
  #       (x[3],   0, 0, 0),
  #       (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

  #   H += lagrange[0]*np.array((
  #       (0, 0, 0, 0),
  #       (x[2]*x[3], 0, 0, 0),
  #       (x[1]*x[3], x[0]*x[3], 0, 0),
  #       (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

  #   H += lagrange[1]*2*np.eye(4)

  #   row, col = self.hessianstructure()

  #   return H[row, col]

  def constraints(self, Z):
    """
    return the constraints
    """
    C = np.zeros(self.lenC)

    # evaluate dynamics at x0,x1,,.,xN-1 (totally N points)
    self.fXk = list() # of length N
    # print("constraints")
    for k in range(self.N):
      x = self.Zx(Z,k)
      u = []
      if k == self.N-1:
        u = np.zeros(self.m) # the control at xN is set to zero.
      else:
        u = self.Zu(Z,k)
      self.fXk.append( dyn.fc_ddc2(x,u) )
    
    # evalute collocation points
    self.Xm = np.zeros([self.N-1,self.n])
    self.Um = np.zeros([self.N-1,self.m])
    self.fXm = np.zeros([self.N-1,self.n])
    # print("constraints2")
    for k in range(self.N-1):
      self.Um[k] = np.zeros(2) 
      if (k+1) < self.N-1:
        self.Um[k] = 0.5*( self.Zu(Z,k) + self.Zu(Z,k+1) )
      else:
        self.Um[k] = 0.5*( self.Zu(Z,k) + np.zeros(2) )

      self.Xm[k] = 0.5*( self.Zx(Z,k) + self.Zx(Z,k) ) + self.dt/8.0*(self.fXk[k] - self.fXk[k+1])
      self.fXm[k] = dyn.fc_ddc2(self.Xm[k], self.Um[k]) # at time t+self.dt/2.0

    # dynamics constraints
    for k in range(self.N-1):
      a = k*self.n
      C[a:a+self.n] = ( self.Zx(Z,k) - self.Zx(Z,k+1) ) + self.dt/6.0 * (self.fXk[k] + 4*self.fXm[k] + self.fXk[k+1])

    # final state
    C[(self.N-1)*self.n : self.N*self.n] = self.Zx(Z,self.N-1) - self.xf

    return C

  def jacobian(self, Z):
    """
    Returns the Jacobian of the constraints with respect to Z.
    """
    # TODO, replace with sparse matrix

    jac = np.zeros([self.lenC, self.lenZ])
    # print("jacobian")

    Ak = list()
    Bk = list()
    for k in range(self.N):

      u = np.zeros(2)
      if k < self.N-1:
        u = self.Zu(Z,k)
      Ak.append( dyn.fc_ddc2_A(self.Zx(Z,k), u) )
      Bk.append( dyn.fc_ddc2_B(self.Zx(Z,k), u) )

    # print("jacobian2")

    Am = list()
    Bm = list()
    for k in range(self.N-1):
      Am.append( dyn.fc_ddc2_A(self.Xm[k],self.Um[k]) )
      Bm.append( dyn.fc_ddc2_B(self.Xm[k],self.Um[k]) )

    In = np.identity(self.n)
    Im = np.identity(self.m)

    for k in range(self.N-1):
      row_idx = k*self.n
      col_idx = k*(self.m+self.n)
      # print(" prev, k = ", k, " col_idx = ", col_idx)
      
      if k >= 1:
        dxm_dx1 = 0.5*In + self.dt/8.0 * Ak[k]
        jac[row_idx:row_idx+self.n, col_idx-self.n:col_idx] += In + self.dt/6*(Ak[k] + 4*np.dot(Am[k],dxm_dx1) )

      dxm_dx2 = 0.5*In - self.dt/8.0 * Ak[k]
      jac[row_idx:row_idx+self.n, col_idx+self.m:col_idx+self.m+self.n] += (-In) + self.dt/6*(Ak[k+1] + 4*np.dot(Am[k],dxm_dx2) )

      dum_du1 = 0.5*Im
      dxm_du1 = self.dt/8.0 * Bk[k]
      jac[row_idx:row_idx+self.n, col_idx:col_idx+self.m] += self.dt/6*(Bk[k] + 4*np.dot(Am[k],dxm_du1) + 4*np.dot(Bm[k],dum_du1) )

      if k < self.N-2:
        dum_du2 = 0.5*Im
        dxm_du2 = -self.dt/8.0 * Bk[k+1]
        cidx = col_idx+self.m+self.n
        # print(" shape = ", Bk[k+1].shape, " shape = ", np.dot(Am[k],dxm_du2).shape, " shape = ", np.dot(Bm[k],dum_du2).shape )
        # print(" shape,,, = ", jac[row_idx:row_idx+self.n, cidx:cidx+self.m].shape, " cidx = ", cidx)
        jac[row_idx:row_idx+self.n, cidx:cidx+self.m] += self.dt/6*(Bk[k+1] + 4*np.dot(Am[k],dxm_du2) + 4*np.dot(Bm[k],dum_du2) )

    # final state
    row_idx = (self.N-2)*self.n
    col_idx = (self.N-2)*(self.m+self.n)
    jac[row_idx:row_idx+self.n, col_idx+self.m:col_idx+self.m+self.n] += In

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

