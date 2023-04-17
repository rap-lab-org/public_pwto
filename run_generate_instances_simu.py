
import numpy as np
import pickle
import matplotlib.pyplot as plt

import context
import misc
import obstacle as obs


def main_gen_tests(ts_name):

  res_file_path = "./results/res_random32_simu_refactor/result.pickle"

  results = misc.LoadPickle(res_file_path)
  pf = results.obs_pf

  # obs_cov_val = 1e-3
  npix = 206

  # GenerateTestSerie(gridx, gridy, num of robots, num of test cases in a serie, obst_thres):
  gridx = 32
  gridy = 32

  starts = np.array([0.1, 0.1, 0, 0, 0])
  goals = np.array([0.9, 0.8, 0 ,0, 0])

  instances = dict()
  instances["name"] = ts_name
  instances["starts"] = starts
  instances["goals"] = goals


  instances["obs_pf"] = pf
  print(pf.shape)

  ### cost field plot
  fig = plt.figure(figsize=(4,4))
  xx = np.linspace(0,1,num=npix)
  yy = np.linspace(0,1,num=npix)
  X,Y = np.meshgrid(xx,yy) # this seems to be the correct way... Y first, X next. # no it's not. still x,y
  pf = instances["obs_pf"]
  print("pf.shape = ", pf.shape)
  plt.contourf(X, Y, pf, levels=np.linspace(np.min(pf), np.max(pf),200), cmap='gray_r')
  plt.xticks([0,1])
  plt.yticks([0,1])
  plt.draw()
  save_path = "./results/res_random32_simu_refactor/"+ts_name+"_costField.png"
  plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=200)

  misc.SavePickle(instances, "./results/res_random32_simu_refactor/" + ts_name+".pickle")
  return

if __name__ == "__main__":
  main_gen_tests("random32_simu")