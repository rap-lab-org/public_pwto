
import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
from opty.utils import parse_free
import obstacle as obs

# def fc_ddc2(x,u):
#   """
#   fc = continuous dynamics, ddc2 = differential-drive car 2nd-order
#   """
#   return np.array( [ np.cos(x[2])*x[3] ,\
#                      np.sin(x[2])*x[3] ,\
#                      x[4] ,\
#                      u[0] ,\
#                      u[1] ] )
# def fc_ddc2_A(x,u):
#   """
#   continuous, linearize, A
#   """
#   return np.array([[0, 0, -np.sin(x[2])*x[3], np.cos(x[2]), 0],
#                    [0, 0,  np.cos(x[2])*x[3], np.sin(x[2]), 0],
#                    [0, 0, 0, 0,  1],
#                    [0, 0, 0, 0,  0],
#                    [0, 0, 0, 0,  0]])

# def fc_ddc2_B(x,u):
#   """
#   continuous, linearize, B
#   """
#   return np.array([[0, 0],
#                    [0, 0],
#                    [0, 0],
#                    [1, 0],
#                    [0, 1]])

# def fd_ddc2(x,u,dt):
#   """
#   fd = discrete dynamics, ddc2 = differential-drive car 2nd-order
#   x = [x,y,theta,v,w]
#   u = [v_dot, w_dot]
#   """
#   return np.array( [ x[0] + np.cos(x[2])*x[3]*dt ,\
#                      x[1] + np.sin(x[2])*x[3]*dt ,\
#                      x[2] + x[4]*dt ,\
#                      x[3] + u[0]*dt ,\
#                      x[4] + u[1]*dt ] )

# def fd_ddc2_A(x,u,dt):
#   """
#   linearize the discrete dynamics of the 2nd-order differential-drive car to obtain matrix A
#   """
#   return np.array([[1, 0, -np.sin(x[2])*x[3]*dt, np.cos(x[2])*dt, 0],
#                    [0, 1,  np.cos(x[2])*x[3]*dt, np.sin(x[2])*dt, 0],
#                    [0, 0, 1, 0, dt],
#                    [0, 0, 0, 1,  0],
#                    [0, 0, 0, 0,  1]])

# def fd_ddc2_B(x,u,dt):
#   """
#   linearize the discrete dynamics of the 2nd-order differential-drive car to obtain matrix B
#   """
#   return np.array([[0, 0],
#   	               [0, 0],
#   	               [0, 0],
#   	               [dt,0],
#   	               [0,dt]])


def dirCol_ddc2(initial_guess, Sinit, Sgoal, w, obss, num_nodes, interval_value, max_iter):
  """
  obss = ObstacleSet object.
  """
  w1 = w[0] # control cost, for the u terms.
  w2 = w[1] # obstacle cost, larger = stay more far away from obstacles
  w3 = w[2] # stay close to the initial guess, larger = stay closer to the initial guess. 

  duration = (num_nodes-1)*interval_value

  t = sym.symbols('t')
  sx,sy,stheta,sv,sw,ua,uw = sym.symbols('sx, sy, stheta, sv, sw, ua, uw', cls=sym.Function)
  state_symbols = (sx(t),sy(t),stheta(t),sv(t),sw(t))
  # constant_symbols = ()
  specified_symbols = (ua(t), uw(t))

  eom = sym.Matrix([sx(t).diff() - sv(t)*sym.cos(stheta(t)),
                    sy(t).diff() - sv(t)*sym.sin(stheta(t)),
                    stheta(t).diff() - sw(t),
                    sv(t).diff() - ua(t),
                    sw(t).diff() - uw(t)])

  def obj(Z):
    """Minimize the sum of the squares of the control torque."""
    # print(Z.shape)
    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    # print(xy_tj)
    J1 = w1*np.sum(U**2)
    J2 = w2*obss.arrayCost(xy_tj)
    J3 = w3*np.sum( (X[0,:]-initial_guess[:num_nodes])**2 + (X[1,:]-initial_guess[num_nodes:2*num_nodes])**2 )
    # print(" J1 = ", J1, " J2 = ", J2, " J3 = ", J3)
    return J1 + J2 + J3

  def obj_grad(Z):
    grad = np.zeros_like(Z)
    # X,U = parse_free(Z,5,2,num_nodes)
    # grad[2 * num_nodes:] = w1*2.*interval_value * Z[2 * num_nodes:]

    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    obst_grad = obss.arrayGrad(xy_tj)

    grad[5 * num_nodes:] = w1*2*Z[5 * num_nodes:] # u1,u2

    # print(obst_grad)
    grad[0 : num_nodes] = w2*obst_grad[:,0] # x
    grad[num_nodes : 2*num_nodes] = w2*obst_grad[:,1] # y
    # print(grad)
    grad[0: num_nodes] += w3*( X[0,:]-initial_guess[:num_nodes] )
    grad[num_nodes: 2*num_nodes] += w3*( X[1,:]-initial_guess[num_nodes:2*num_nodes] )
    return grad

  # Specify the symbolic instance constraints, i.e. initial and end
  # conditions.
  instance_constraints = (sx(0.0) - Sinit[0],
                          sy(0.0) - Sinit[1],
                          # stheta(0.0) - Sinit[2],
                          sv(0.0) - Sinit[3],
                          sw(0.0) - Sinit[4],
                          sx(duration) - Sgoal[0],
                          sy(duration) - Sgoal[1],
                          # stheta(duration) - Sgoal[2],
                          sv(duration) - Sgoal[3],
                          sw(duration) - Sgoal[4] )

  # Create an optimization problem.

  prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
                 instance_constraints=instance_constraints,
                 bounds={sx(t): (0,1), sy(t): (0,1), sv(t): (0, 0.2), sw(t): (-5, 5), ua(t): (-1, 1), uw(t): (-5, 5)})

  prob.addOption("max_iter",max_iter)

  # Find the optimal solution.
  Zsol, info = prob.solve(initial_guess)

  Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
  Xinit, Uinit, _ = parse_free(initial_guess, 5, 2, num_nodes)

  return Zsol, info

