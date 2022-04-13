
import numpy as np
import time


def fd_ddc2(x,u,dt):
  """
  fd = discrete dynamics, ddc2 = differential-drive car 2nd-order
  x = [x,y,theta,v,w]
  u = [v_dot, w_dot]
  """
  return np.array( [ x[0] + np.cos(x[2])*x[3]*dt ,\
                      x[1] + np.sin(x[2])*x[3]*dt ,\
                      x[2] + x[4]*dt ,\
                      x[3] + u[0]*dt ,\
                      x[4] + u[1]*dt ] )

def fd_ddc2_A(x,u,dt):
  """
  linearize the discrete dynamics of the 2nd-order differential-drive car to obtain matrix A
  """
  return np.array([[1, 0, -np.sin(x[2])*x[3]*dt, np.cos(x[2])*dt, 0],
                   [0, 1,  np.cos(x[2])*x[3]*dt, np.sin(x[2])*dt, 0],
                   [0, 0, 1, 0, dt],
                   [0, 0, 0, 1,  0],
                   [0, 0, 0, 0,  1]])

def fd_ddc2_B(x,u,dt):
  """
  linearize the discrete dynamics of the 2nd-order differential-drive car to obtain matrix B
  """
  return np.array([[0, 0],
  	               [0, 0],
  	               [0, 0],
  	               [dt,0],
  	               [0,dt]])

if __name__ == "__main__":

  tstart = time.perf_counter()
  x = np.ones(5)*0.1
  u = np.ones(2)*0.1
  dt = 0.1
  xnew = fd_ddc2(x,u,dt)
  print("old x = ",x, " xnew = ", xnew)
  print(' time used = ', time.perf_counter() - tstart)

  x = np.ones(5)*0.2
  u = np.ones(2)*0.1
  dt = 0.1
  xnew = fd_ddc2(x,u,dt)
  print("old x = ",x, " xnew = ", xnew)
  print(' time used = ', time.perf_counter() - tstart)

  x = np.ones(5)*0.3
  u = np.ones(2)*0.1
  dt = 0.1
  xnew = fd_ddc2(x,u,dt)
  print("old x = ",x, " xnew = ", xnew)
  A = fd_ddc2_A(x,u,dt)
  B = fd_ddc2_B(x,u,dt)
  print("linear predict xnew = ", np.dot(A,x)+np.dot(B,u))

  print(' time used = ', time.perf_counter() - tstart)
