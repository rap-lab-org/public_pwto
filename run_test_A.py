"""
Author: Zhongqiang (Richard) Ren
Date: 2022-04-21
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

from misc import LoadMapDao, findObstacles, path2InitialGuess, linearInitGuess, lexSortResult


def run_test(select_idx, naive_init):

    ############## Global Params Begin ################
    case_ID = "A"
    Sinit = np.array([0.1, 0.1,  0, 0, 0])
    Sgoal = np.array([0.94, 0.8, 0,0, 0])
    num_nodes = 100
    interval_value = 0.1
    # select_idx = 12 # change this to select diff paths.
    # naive_init = False
    w1 = 0.01 # control cost, for the u terms.
    w2 = 5000 # obstacle cost, larger = stay more far away from obstacles
    w3 = 200 # stay close to the initial guess, larger = stay closer to the initial guess.
    ############## Global Params End ################

    if naive_init:
      w3 = 0

    duration = (num_nodes-1)*interval_value
    map_grid = LoadMapDao("runtime_data/random-32-32-20.map")
    obsts_all = findObstacles(map_grid)
    obsts = obsts_all / 32.0
    obss = obs.ObstSet( obsts )
    npix = 100
    print("[INFO] start to compute pf...")
    pf = obss.potentialField(1,1,npix)*100

    fig = plt.figure(figsize=(5,5))

    xx = np.linspace(0,1,num=100)
    yy = np.linspace(0,1,num=100)
    Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
    print("pf range = ", np.mean(pf), np.median(pf), np.max(pf))
    # plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')
    plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
    plt.plot(Sinit[0],Sinit[1],"ro")
    plt.plot(Sgoal[0],Sgoal[1],"r*")

    plt.draw()
    plt.pause(1)
    plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-pf-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)

    ######### Fig 2.

    if not naive_init:
        ## convert to a 100x100 grid
        c1 = np.ones([npix,npix]) # distance
        c2 = pf # dist to obstacle

        vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
        vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)

        res_dict = emoa.runEMOA([c1,c2], "runtime_data/", "../public_emoa/build/run_emoa", "runtime_data/temp-res.txt", vo, vd, 60)

        lexSortResult(res_dict)
        print(res_dict['costs'])

        fig = plt.figure(figsize=(5,5))

        xx = np.linspace(0,1,num=100)
        yy = np.linspace(0,1,num=100)
        Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
        print("pf range = ", np.mean(pf), np.median(pf), np.max(pf))
        # plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')
        plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
        plt.plot(Sinit[0],Sinit[1],"ro")
        plt.plot(Sgoal[0],Sgoal[1],"r*")

        paths = res_dict['paths']
        select_path_x = []
        select_path_y = []
        idx = 0
        for k in paths:
            p = paths[k]
            px = list()
            py = list()
            for v in p:
                py.append( (v%npix)*(1/npix) )
                px.append( int(np.floor(v/npix))*(1.0/npix) )
            plt.plot(px,py,"g--")
            if idx == select_idx:
              select_path_x = px
              select_path_y = py
            idx += 1
        plt.draw()
        plt.pause(1)
        # print(" select_path_x = ", select_path_x)
        # print(" select_path_y = ", select_path_y)

    initial_guess = []
    if naive_init:
      initial_guess = linearInitGuess(Sinit[0:2], Sgoal[0:2], num_nodes, 5, 2, interval_value)
    else:
      initial_guess = path2InitialGuess(select_path_x, select_path_y, num_nodes, 5, 2, interval_value)
      plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b")
      plt.draw()
      plt.pause(1)
      plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-pareto_paths-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)


    ### Fig 2b, Pareto front

    if not naive_init:
        fig = plt.figure(figsize=(3,3))

        res_dict['costs']
        idx = 0
        for k in res_dict['costs']:
            c = res_dict['costs'][k]
            plt.plot(c[0],c[1],"go")
            if idx == select_idx:
                plt.plot(c[0],c[1],"bs")
            idx += 1

        plt.grid()
        plt.draw()
        plt.pause(1)
        plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-pareto_front-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)


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
                   bounds={sx(t): (0,1), sy(t): (0,1), ua(t): (-1, 1), uw(t): (-5, 5), sv(t): (0, 0.2), sw(t): (-5, 5)})

    prob.addOption("max_iter",500)


    # # Use a random positive initial guess.
    # np.random.seed(0)
    # # initial_guess = np.random.randn(prob.num_free)
    # initial_guess = np.ones(prob.num_free)*0

    # Find the optimal solution.
    Zsol, info = prob.solve(initial_guess)

    # print(Zsol.shape)

    Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
    Xinit, Uinit, _ = parse_free(initial_guess, 5, 2, num_nodes)


    ### Fig 3

    fig = plt.figure(figsize=(5,5))

    xx = np.linspace(0,1,num=100)
    yy = np.linspace(0,1,num=100)
    Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
    print("pf range = ", np.mean(pf), np.median(pf), np.max(pf))
    # plt.contourf(X, Y, -pf, levels=np.linspace(np.min(-pf), np.max(-pf),100), cmap='gray')
    plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
    plt.plot(Sinit[0],Sinit[1],"ro")
    plt.plot(Sgoal[0],Sgoal[1],"r*")

    plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b")

    plt.plot(Xsol[0,:],Xsol[1,:],"r.")

    plt.draw()
    plt.pause(1)
    if naive_init:
        plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-traj-naiveInit.png", bbox_inches='tight', dpi=200)
    else:
        plt.savefig("runtime_data/random-32-32-20-"+str(case_ID)+"-traj-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)

    # for k in info:
    #     print(" - ", k, " = ", info[k])

    if not naive_init:
        print("---------------------SOLUTION RESULT BEGIN------------------------")
        print("[RESULT] EMOA*, search time = ", res_dict['rt_search'])
        print("[RESULT] EMOA*, num solutions = ", res_dict['n_sol'])
        print("[RESULT] IPOPT, traj optm obj val = ", info['obj_val'])
        # print("[RESULT] IPOPT, traj optm obj val = ", info['iterations'])
        print("---------------------SOLUTION RESULT END------------------------")


if __name__ == "__main__":
    # for idx in range(13):
    #     run_test(idx, True)
  
    run_test(12, False)