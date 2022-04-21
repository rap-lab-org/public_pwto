"""This solves the simple pendulum swing up problem presented here:

http://hmc.csuohio.edu/resources/human-motion-seminar-jan-23-2014

A simple pendulum is controlled by a torque at its joint. The goal is to
swing the pendulum from its rest equilibrium to a target angle by minimizing
the energy used to do so.

"""

from collections import OrderedDict

import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
from opty.utils import building_docs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from opty.utils import parse_free

import context
import obstacle as obs

import emoa_py_api as emoa


def LoadMapDao(map_file):
  grids = np.zeros((2,2))
  with open(map_file,'r') as f:
    lines = f.readlines()
    lidx = 0
    nx = 0
    ny = 0
    for line in lines:
      if lidx == 1:
        a = line.split(" ")
        nx = int(a[1])
      if lidx == 2:
        a = line.split(" ")
        ny = int(a[1])
      if lidx == 4:
        grids = np.zeros((nx,ny))
      if lidx >= 4: # map data begin
        x = lidx - 4
        y = 0
        a = line.split("\n")
        # print(a[0])
        # print(len(str(a[0])))
        for ia in str(a[0]):
          # print(ia)
          if ia == "." or ia == "G":
            grids[x,y] = 0
          else:
            grids[x,y] = 1
          y = y+1
      lidx = lidx + 1
  return grids

def findObstacles(grid):
  """
  """
  out = list()
  nyt,nxt = grid.shape
  for iy in range(nxt):
    for ix in range(nyt):
      if grid[iy,ix] == 1:
        out.append(np.array([iy,ix]))
  return np.array(out)


map_grid = LoadMapDao("runtime_data/random-32-32-20.map")
obsts_all = findObstacles(map_grid)
obsts = obsts_all / 32.0
print(obsts)

Sinit = np.array([0.1, 0.1, 0, 0, 0])
Sgoal = np.array([0.8, 0.7, 0 ,0, 0])

# obs_pos_array = np.array([[0.35, 0.55],[0.4, 0.55],[0.45, 0.55],[0.35, 0.4],[0.4, 0.4],[0.45, 0.4]])
# obs_pos_array = np.array([[0.3, 0.55]])
# obss = obs.ObstSet( obs_pos_array )
obss = obs.ObstSet( obsts )
npix = 100
print("start to compute pf...")
pf = obss.potentialField(1,1,npix)*100
print("pf done...")

## convert to a 100x100 grid
c1 = np.ones([npix,npix]) # distance
c2 = pf # dist to obstacle

vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)

res_dict = emoa.runEMOA([c1,c2], "runtime_data/", "../public_emoa/build/run_emoa", "runtime_data/temp-res.txt", vo, vd, 60)
print(res_dict)


fig = plt.figure(figsize=(10,10))

xx = np.linspace(0,1,num=100)
yy = np.linspace(0,1,num=100)
Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')

paths = res_dict['paths']
select_path_x = []
select_path_y = []
for k in paths:
    p = paths[k]
    px = list()
    py = list()
    for v in p:
        py.append( (v%npix)*(1/npix) )
        px.append( int(np.floor(v/npix))*(1/npix) )
    select_path_x = px
    select_path_y = py
    plt.plot(px,py,"b")
plt.draw()
plt.pause(2)


print(" select_path_x = ", select_path_x)

target_angle = np.pi
# duration = 20.0
num_nodes = 150
save_animation = False

# interval_value = duration / (num_nodes - 1)
interval_value = 0.1
duration = (num_nodes-1)*interval_value

# Symbolic equations of motion
# I, m, g, d, t = sym.symbols('I, m, g, d, t')
# theta, omega, T = sym.symbols('theta, omega, T', cls=sym.Function)


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


# state_symbols = (theta(t), omega(t))
# constant_symbols = (I, m, g, d)
# specified_symbols = (T(t),)

# eom = sym.Matrix([theta(t).diff() - omega(t),
#                   I * omega(t).diff() + m * g * d * sym.sin(theta(t)) - T(t)])

# # Specify the known system parameters.
# par_map = OrderedDict()
# par_map[I] = 1.0
# par_map[m] = 1.0
# par_map[g] = 9.81
# par_map[d] = 1.0

# Specify the objective function and it's gradient.

# Sinit = np.array([0.1, 0.2, 0.0, 0, 0])
# Sgoal = np.array([1.2, 1.7, 0.0, 0, 0])

w1 = 0.1
w2 = 10000

def obj(Z):
    """Minimize the sum of the squares of the control torque."""
    # print(Z.shape)
    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    # print(xy_tj)
    J1 = w1*np.sum(U**2)
    J2 = w2*obss.arrayCost(xy_tj)
    # print(" J1 = ", J1, " J2 = ", J2)
    return J1 + J2

def obj_grad(Z):
    grad = np.zeros_like(Z)
    # X,U = parse_free(Z,5,2,num_nodes)
    # grad[2 * num_nodes:] = w1*2.*interval_value * Z[2 * num_nodes:]

    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    obst_grad = obss.arrayGrad(xy_tj)
    # print(obst_grad)
    grad[0 : num_nodes] = w2*obst_grad[:,0] # x
    grad[num_nodes : 2*num_nodes] = w2*obst_grad[:,1] # y
    grad[5 * num_nodes:] = w1*2*Z[5 * num_nodes:] # u1,u2
    # print(grad)
    return grad

# Specify the symbolic instance constraints, i.e. initial and end
# conditions.
instance_constraints = (sx(0.0) - Sinit[0],
                        sy(0.0) - Sinit[1],
                        stheta(0.0) - Sinit[2],
                        sv(0.0) - Sinit[3],
                        sw(0.0) - Sinit[4],
                        sx(duration) - Sgoal[0],
                        sy(duration) - Sgoal[1],
                        stheta(duration) - Sgoal[2],
                        sv(duration) - Sgoal[3],
                        sw(duration) - Sgoal[4] )

# Create an optimization problem.

prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
               instance_constraints=instance_constraints,
               bounds={ua(t): (-0.5, 0.5), uw(t): (-1, 1), sv(t): (0.0, 3), sw(t): (-3, 3)})


# prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
#                instance_constraints=instance_constraints,
#                bounds={ua(t): (-1, 1), uw(t): (-2, 2), sv(t): (0.0, 3), sw(t): (-2, 2)})

# # Create an optimization problem.
# prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
#                known_parameter_map=par_map,
#                instance_constraints=instance_constraints,
#                bounds={T(t): (-2.0, 2.0)})

prob.addOption("max_iter",100)

def path2InitialGuess(px, py, n_nodes):
  """
  p = path find by EMOA*.
  """
  initial_guess = np.ones(prob.num_free)*0
  lp = len(px)
  for i in range(n_nodes):
    idx = int( np.floor( lp*(i/n_nodes) ) )
    # idy = idx + 1
    # if idy >= lp:
      # idy = idx
    initial_guess[i*7] = px[idx]
    initial_guess[i*7+1] = py[idx]
    # initial_guess[i*7+5:i*7+7]
  return initial_guess

initial_guess = path2InitialGuess(select_path_x, select_path_y, num_nodes)

plt.plot(px,py,"g")

# # Use a random positive initial guess.
# np.random.seed(0)
# # initial_guess = np.random.randn(prob.num_free)
# initial_guess = np.ones(prob.num_free)*0

# Find the optimal solution.
Zsol, info = prob.solve(initial_guess)

print(Zsol.shape)
print(info)

Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
Xinit, Uinit, _ = parse_free(initial_guess, 5, 2, num_nodes)

# fig = plt.figure()

# xx = np.linspace(0,1,num=100)
# yy = np.linspace(0,1,num=100)
# Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
# pf = obss.potentialField(1,1,100)

# plt.plot(Xinit[0,:],Xinit[1,:],"b.")
plt.plot(Xsol[0,:],Xsol[1,:],"r.")
plt.show()

# Make some plots
# prob.plot_trajectories(solution)
# prob.plot_constraint_violations(solution)
# prob.plot_objective_value()

# # Display animation
# if not building_docs():
#     time = np.linspace(0.0, duration, num=num_nodes)
#     angle = solution[:num_nodes]

#     fig = plt.figure()
#     ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2),
#                          ylim=(-2, 2))
#     ax.grid()

#     line, = ax.plot([], [], 'o-', lw=2)
#     time_template = 'time = {:0.1f}s'
#     time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

#     def init():
#         line.set_data([], [])
#         time_text.set_text('')
#         return line, time_text

#     def animate(i):
#         x = [0, par_map[d] * np.sin(angle[i])]
#         y = [0, -par_map[d] * np.cos(angle[i])]

#         line.set_data(x, y)
#         time_text.set_text(time_template.format(i * interval_value))
#         return line, time_text

#     ani = animation.FuncAnimation(fig, animate, np.arange(1, len(time)),
#                                   interval=25, blit=True, init_func=init)

#     if save_animation:
#         ani.save('pendulum_swing_up.mp4', writer='ffmpeg',
#                  fps=1 / interval_value)

plt.show()
