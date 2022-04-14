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

target_angle = np.pi
duration = 10.0
num_nodes = 10
save_animation = False

interval_value = duration / (num_nodes - 1)

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

Sinit = np.array([0.2, 0.6, 0, 0, 0])
Sgoal = np.array([1.2, 1.4, np.pi/2 ,0, 0])

obs_pos_array = np.array([[0.7, 1.1],[0.8, 1.2]])
obss = obs.ObstSet( obs_pos_array )

w1 = 10
w2 = 100

def obj(Z):
    """Minimize the sum of the squares of the control torque."""
    # print(Z.shape)
    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    # print(xy_tj)
    return interval_value * np.sum(U**2) * w1 + w2*obss.arrayCost(xy_tj)


def obj_grad(Z):
    grad = np.zeros_like(Z)
    # X,U = parse_free(Z,5,2,num_nodes)
    # grad[2 * num_nodes:] = w1*2.*interval_value * Z[2 * num_nodes:]

    X,U,_ = parse_free(Z,5,2,num_nodes)
    xy_tj = X[0:2,:].T
    obst_grad = obss.arrayGrad(xy_tj)
    print(" obst_grad.shape = ", obst_grad.shape)

    grad[0 : num_nodes] = w2 * obst_grad[:,0]
    grad[num_nodes : 2*num_nodes] = w2 * obst_grad[:,1]
    grad[5 * num_nodes:] = w1 * 2.*interval_value * Z[5 * num_nodes:]
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
               bounds={ua(t): (-0.5, 0.5), uw(t): (-1.0, 1.0), sv(t): (0.0, 1.5)})

# # Create an optimization problem.
# prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
#                known_parameter_map=par_map,
#                instance_constraints=instance_constraints,
#                bounds={T(t): (-2.0, 2.0)})

# Use a random positive initial guess.
np.random.seed(0)
initial_guess = np.random.randn(prob.num_free)

# Find the optimal solution.
Zsol, info = prob.solve(initial_guess)

print(Zsol.shape)
print(info)

Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
Xinit, Uinit, _ = parse_free(initial_guess, 5, 2, num_nodes)


fig = plt.figure()

xx = np.linspace(0,2,num=100)
yy = np.linspace(0,2,num=100)
Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
pf = obss.potentialField(2,2,100)

plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')

plt.plot(Xinit[0,:],Xinit[1,:],"b.")
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
