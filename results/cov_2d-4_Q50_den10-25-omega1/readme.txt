

  npix = 200

  # GenerateTestSerie(gridx, gridy, num of robots, num of test cases in a serie, obst_thres):
  gridx = 32
  gridy = 32
  ntest = 10
  if ts_name == "random32A":
    obst_thres = 0.10
  elif ts_name == "random32B":
    obst_thres = 0.15
  elif ts_name == "random32C":
    obst_thres = 0.20
  elif ts_name == "random32D":
    obst_thres = 0.25



Q=50
omega limit = 1.57