"""
Author: Zhongqiang (Richard) Ren
Date: 2022-04-21
"""

from collections import OrderedDict

import numpy as np
import sympy as sym
# from opty.direct_collocation import Problem
# from opty.utils import building_docs
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from opty.utils import parse_free

import context
import optm_ddc2
import obstacle as obs
import emoa_py_api as emoa

from misc import LoadMapDao, findObstacles, path2InitialGuess, linearInitGuess, lexSortResult, scalarizeSolve


def plotInstance():

  folder = "data/random_instance_A/"
  
  map_grid = LoadMapDao(folder+"random-32-32-20.map")
  obsts_all = findObstacles(map_grid)
  obsts = obsts_all / 32.0
  # print(obsts)

  Sinit = np.array([0.1, 0.1, 0, 0, 0])
  Sgoal = np.array([0.9, 0.8, 0 ,0, 0])

  num_nodes = 100
  save_animation = False

  # interval_value = duration / (num_nodes - 1)
  interval_value = 0.1
  duration = (num_nodes-1)*interval_value

  obss = obs.ObstSet( obsts, 0.5/32, 2.0/32, 10 )
  npix = 100
  print("start to compute pf...")
  pf = obss.potentialField(1,1,npix)
  print("pf done...")

  ## convert to a 100x100 grid
  c1 = np.ones([npix,npix]) # distance
  c2 = pf # dist to obstacle

  vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
  vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)

  fig = plt.figure(figsize=(5,5))

  xx = np.linspace(0,1,num=npix)
  yy = np.linspace(0,1,num=npix)
  Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.

  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.plot(Sinit[0],Sinit[1],"ro")
  plt.plot(Sgoal[0],Sgoal[1],"r*")


  plt.draw()
  plt.pause(1)
  # print(" select_path_x = ", select_path_x)
  # print(" select_path_y = ", select_path_y)

  case_ID = "A"
  plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-instance.png", bbox_inches='tight', dpi=200)

  plt.close("all")

  return

def run_test(select_idx, mode):
    """
    mode = 0, Warm start using Pareto-optimal solutions.
    mode = 1, naive init guess.
    mode = 2, scalarized map.
    mode = 3, random init guess.
    """

    ##############################################################
    # Set up parameters
    ##############################################################

    case_ID = "A"
    Sinit = np.array([0.1, 0.1,  0, 0, 0])
    Sgoal = np.array([0.94, 0.8, 0,0, 0])
    num_nodes = 100
    interval_value = 0.1
    n_weight = 13 # only useful when mode = 2
    folder = "data/random_instance_A/"
    emoa_path = "../public_emoa/build/run_emoa"
    w1 = 0 # control cost, for the u terms.
    w2 = 1 # obstacle cost, larger = stay more far away from obstacles
    w3 = 0 # stay close to the initial guess, larger = stay closer to the initial guess.

    ##############################################################
    # Compute potential field for obstacles and visualize
    ##############################################################

    if mode == 1 or mode == 3: # naive_init
      w3 = 0

    duration = (num_nodes-1)*interval_value
    map_grid = LoadMapDao(folder+"random-32-32-20.map")
    obsts_all = findObstacles(map_grid)
    obsts = obsts_all / 32.0
    obss = obs.ObstSet( obsts, 0.5/32, 2.0/32, 10 )
    npix = 100
    print("[INFO] start to compute pf...")
    pf = obss.potentialField(1,1,npix)

    if mode == 1: # plot just for this mode is enough.
        fig = plt.figure(figsize=(5,5))
        xx = np.linspace(0,1,num=npix)
        yy = np.linspace(0,1,num=npix)
        Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
        plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
        plt.plot(Sinit[0],Sinit[1],"ro")
        plt.plot(Sgoal[0],Sgoal[1],"r*")

        plt.draw()
        plt.pause(1)
        plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-mode-"+str(mode)+"-pf-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)
    

    ##############################################################
    # Plan paths and show
    ##############################################################
    if mode == 0 or mode == 2:
        ## convert to a 100x100 grid
        c1 = np.ones([npix,npix]) # distance
        c2 = pf # dist to obstacle

        vo = int(Sinit[0]*npix*npix + Sinit[1]*npix)
        vd = int(Sgoal[0]*npix*npix + Sgoal[1]*npix)

        res_dict = dict()
        if mode == 0:
            res_dict = emoa.runEMOA([c1,c2], folder, emoa_path, folder+"temp-res.txt", vo, vd, 60)
            lexSortResult(res_dict)
        elif mode == 2:
            res_dict = scalarizeSolve([c1,c2], n_weight, folder, emoa_path, folder+"temp-res.txt", vo, vd, 60)

        fig = plt.figure(figsize=(5,5))

        xx = np.linspace(0,1,num=npix)
        yy = np.linspace(0,1,num=npix)
        Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
        print("pf range = ", np.mean(pf), np.median(pf), np.max(pf))
        plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
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
    
    # generate initial guess
    initial_guess = []
    if mode == 1:
      initial_guess = linearInitGuess(Sinit[0:2], Sgoal[0:2], num_nodes, 5, 2, interval_value)
    elif mode == 0 or mode == 2:
      initial_guess = path2InitialGuess(select_path_x, select_path_y, num_nodes, 5, 2, interval_value)
      if mode == 0:
        plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b")
      plt.draw()
      plt.pause(1)
      plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-mode-"+str(mode)+"-pareto_paths-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)


    ### Fig 2b, Pareto front

    if mode == 0:
        fig = plt.figure(figsize=(3,3))

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
        plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-mode-"+str(mode)+"-pareto_front-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)

    if mode == 2: # for scalarize, just show the paths, it's enough.
        return
    
    ##############################################################
    # Trajectory optimization
    ##############################################################

    Zsol, info = optm_ddc2.dirCol_ddc2(initial_guess, Sinit, Sgoal, [w1,w2,w3], obss, num_nodes, interval_value, max_iter=500)
    Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)

    ### Fig 3

    fig = plt.figure(figsize=(5,5))
    xx = np.linspace(0,1,num=npix)
    yy = np.linspace(0,1,num=npix)
    Y,X = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next.
    plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),500), cmap='gray_r')
    plt.plot(Sinit[0],Sinit[1],"ro")
    plt.plot(Sgoal[0],Sgoal[1],"r*")
    plt.plot(initial_guess[:num_nodes],initial_guess[num_nodes:2*num_nodes],"b")
    plt.plot(Xsol[0,:],Xsol[1,:],"r.")

    if mode == 1:
        # Use a random positive initial guess.
        np.random.seed(0)
        initial_guess = np.random.randn( num_nodes*(5+2) )
        Zsol, info = optm_ddc2.dirCol_ddc2(initial_guess, Sinit, Sgoal, [w1,w2,w3], obss, num_nodes, interval_value, max_iter=500)
        Xsol, Usol, _ = parse_free(Zsol, 5, 2, num_nodes)
        plt.plot(Xsol[0,:],Xsol[1,:],"y.")

    plt.draw()
    plt.pause(1)
    plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-mode-"+str(mode)+"-traj-k-"+str(select_idx)+".png", bbox_inches='tight', dpi=200)

    if mode == 0:
        print("---------------------SOLUTION RESULT BEGIN------------------------")
        print("[RESULT] EMOA*, search time = ", res_dict['rt_search'])
        print("[RESULT] EMOA*, num solutions = ", res_dict['n_sol'])
        print("[RESULT] IPOPT, traj optm obj val = ", info['obj_val'])
        print("---------------------SOLUTION RESULT END------------------------")

    plt.close("all")

    print("[RESULT] IPOPT, info = ", info)
    print("...", info["status"])

    return


def plotRes():
    folder = "data/random_instance_A/"
    case_ID = "A"

    # these data are currently manually extracted from the data file.

    iters = np.array([285, 264, 355, 474, 500, 304, 435, 327, 313, 496, 500, 382, 386])
    emoa_sch_time = 0.153 # seconds.
    obj_val = np.array([10763688, 2854991, 2658071, 2393952, 2204654, 3296353, 2764473, 2590231, 2805368, 2390316, 2536165, 2069354, 5527384])

    obj_val_percent = obj_val / np.min(obj_val)
    iter_percent = iters / np.max(iters)

    fig = plt.figure(figsize=(3,2))
    plt.plot(range(len(iters)), obj_val_percent, "r^")
    # plt.plot(range(len(iters)), iter_percent, "bv")
    plt.grid()
    plt.draw()
    plt.pause(2)
    plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-obj_percent.png", bbox_inches='tight', dpi=200)



    fig = plt.figure(figsize=(3,2))
    # plt.plot(range(len(iters)), obj_val_percent, "r^")
    plt.plot(range(len(iters)), iter_percent, "bv")
    plt.grid()
    plt.draw()
    plt.pause(2)
    plt.savefig(folder+"random-32-32-20-"+str(case_ID)+"-iter_percent.png", bbox_inches='tight', dpi=200)



if __name__ == "__main__":
  
    ### plot instance
    
    plotInstance()

    ### Run tests
    for k in range(0,13): # this instance has totally 13 cost-unique Pareto-optimal paths.
        run_test(k, mode=0) # change idx from 0 to 12

    run_test(-1, mode=1) 

    run_test(1, mode=2) 

    ### plot results

    plotRes()

