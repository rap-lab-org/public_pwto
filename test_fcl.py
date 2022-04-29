
import numpy as np
import fcl

import context
import misc


if __name__ == "__main__":
    
  # v1 = np.array([1.0, 2.0, 3.0])
  # v2 = np.array([2.0, 1.0, 3.0])
  # v3 = np.array([3.0, 2.0, 1.0])
  # x, y, z = 1, 2, 3
  # rad, lz = 1.0, 3.0
  # n = np.array([1.0, 0.0, 0.0])
  # d = 5.0

  # b = fcl.Box(x, y, z)          # Axis-aligned box with given side lengths

  # R = np.array([[1.0, 0.0, 0.0],
  #              [0.0,  1.0, 0.0],
  #              [0.0,  0.0, 1.0]])
  # q = misc.theta2quat(0.0)
  # tf = fcl.Transform(R, T) # Matrix rotation and translation
  # tf = fcl.Transform(q, T) # Quaternion rotation and translation

  # obj = fcl.CollisionObject(b, tf)


  g1 = fcl.Box(1, 1, 0)
  Q1 = misc.theta2quat(np.pi/4)
  T1 = np.array([0.0, 0.0, 0.0])
  tf1 = fcl.Transform(Q1, T1)
  o1 = fcl.CollisionObject(g1, tf1)

  g2 = fcl.Box(1, 1, 0)
  Q2 = misc.theta2quat(np.pi/4)
  T2 = np.array([2.0, 0.0, 0.0])
  tf2 = fcl.Transform(Q2, T2)
  o2 = fcl.CollisionObject(g2, tf2)

  request = fcl.CollisionRequest()
  result = fcl.CollisionResult()

  ret = fcl.collide(o1, o2, request, result)
  print("collide = ", ret)

  request = fcl.DistanceRequest()
  result = fcl.DistanceResult()

  ret = fcl.distance(o1, o2, request, result)
  print("distance = ", ret)