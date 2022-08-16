
import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
from opty.utils import parse_free
import obstacle as obs

def dirCol_ddc2(initial_guess, Sinit, Sgoal, w, obss, num_nodes, interval_value, vu_bounds,max_iter):
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

  # prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
  #                instance_constraints=instance_constraints,
  #                bounds={sx(t): (0,1), sy(t): (0,1), sv(t): (0, 0.2), sw(t): (-5, 5), ua(t): (-1, 1), uw(t): (-5, 5)})


  v_up = vu_bounds[0]
  w_up = vu_bounds[1]
  ua_up = vu_bounds[2]
  uw_up = vu_bounds[3]

  prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
                 instance_constraints=instance_constraints,
                 bounds={sx(t): (0,1), sy(t): (0,1), sv(t): (0, v_up), sw(t): (-w_up, w_up), ua(t): (-ua_up, ua_up), uw(t): (-uw_up, uw_up)})

  prob.addOption("max_iter",max_iter)

  # Find the optimal solution.
  Zsol, info = prob.solve(initial_guess)

  Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
  Xinit, Uinit, _ = parse_free(initial_guess, 5, 2, num_nodes)

  return Zsol, info

