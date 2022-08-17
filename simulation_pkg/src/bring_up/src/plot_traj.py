#!/usr/bin/env python3


import os
import math
import numpy as np
import matplotlib.pyplot as plt

traj_load_path = "../trajectory/traj_data_random32/solTraj_data_13"
traj_save_path = "../trajectory/traj_data_random32/recTraj_terrain_data_13_open"

dt = 0.1
scale = 18.6

wp_x =[]
wp_y =[]
wp_yaw =[]
wp_v=[]
wp_w=[]
file_plan = open(traj_load_path + ".txt", "r")
for eachLine in file_plan:
    x, y, yaw, v, w, av, aw = eachLine.split()
    wp_x.append(float(x)*scale)
    wp_y.append(float(y)*scale)
    wp_yaw.append(float(yaw))   
    wp_v.append(float(v)*scale)
    wp_w.append(float(w))


real_x =[]
real_y =[]
real_yaw =[]
real_v=[]
real_w=[]
file_real = open(traj_save_path + ".txt", "r")
for eachLine in file_real:
    x, y, yaw, v, w = eachLine.split()
    real_x.append(float(x))    
    real_y.append(float(y))
    real_yaw.append(float(yaw))    
    real_v.append(float(v))
    real_w.append(float(w))

step_num = len(wp_x)
# time = 0.1*float(range(step_num))
time = np.arange(0,step_num*dt,dt)

print("len(time)",len(time))
print("step_num",step_num)

fig, axs = plt.subplots(2,3,figsize=(17,8))
# fig.suptitle('Vertically stacked subplots')
axs[0,0].plot(time,wp_x,"--",linewidth=2.5,c=(0,0.7,0))
axs[0,0].plot(time,real_x,"r-")
# axs[0,0].set_title('x')
axs[0,0].set(xlabel='time (s)', ylabel=' x (m)')
# axs[0,0].legend(["Desired", "Actual"])

axs[0,1].plot(time,wp_y,"--",linewidth=2.5,c=(0,0.7,0))
axs[0,1].plot(time,real_y,"r-")
# axs[0,1].set_title('y')
axs[0,1].set(xlabel='time (s)', ylabel=' y (m)')
# axs[0,1].legend(["Desired", "Actual"])


axs[0,2].plot(wp_x,wp_y,"--",linewidth=2.5,c=(0,0.7,0))
axs[0,2].plot(wp_x,real_y,"r-")
# axs[0,2].set_title('x y')
axs[0,2].set(xlabel='x (m)', ylabel=' y (m)')
# axs[0,2].legend(["Desired", "Actual"])


axs[1,0].plot(time,wp_yaw,"--",linewidth=2.5,c=(0,0.7,0))
axs[1,0].plot(time,real_yaw,"r-")
# axs[1,0].set_title('x y')
axs[1,0].set(xlabel='time (s)', ylabel=' yaw (rad)')
# axs[1,0].legend(["Desired", "Actual"])

axs[1,1].plot(time,wp_v,"--",linewidth=2.5,c=(0,0.7,0))
axs[1,1].plot(time,real_v,"r-")
# axs[1,1].set_title('x y')
axs[1,1].set(xlabel='time (s)', ylabel=' v (m/s)')
# axs[1,1].legend(["Desired", "Actual"])

axs[1,2].plot(time,wp_w,"--",linewidth=2.5,c=(0,0.7,0))
axs[1,2].plot(time,real_w,"r-")
# axs[1,2].set_title('x y')
axs[1,2].set(xlabel='time (s)', ylabel=' w (rad/s)')
# axs[1,2].legend(["Desired", "Actual"])

plt.draw()
# plt.show()

plt.pause(2)
plt.savefig(traj_save_path + ".png", bbox_inches='tight', dpi=200)