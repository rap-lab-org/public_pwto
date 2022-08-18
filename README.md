



### Installation

* Download and compile [EMOA\*](https://github.com/wonderren/public_emoa) by following the instructions there. The downloading directory is suggested to be "../public_emoa" with respect to the directory of this readme file. Otherwise, context.py may need to be changed to get this package running (not tested).

* Install [cyipopt](https://cyipopt.readthedocs.io/en/stable/install.html#) by following the section "On Ubuntu 18.04 Using APT Dependencies".

* Install [opty](https://opty.readthedocs.io/en/latest/) by `pip3 install opty`

* Run pip3 install for any missing packages, e.g. sympy.

* File run_example.py provides an entry point to the PWTO implementation.


### Run Simulation

The simulation verificartion of the trajectory feasibility is conducted in ROS with gazebo, based on the [ROSbot](https://husarion.com/manuals/rosbot) from HUSARION. All the packages to run the simulation are in "/simulation_pkg", where the `rosbot_description` is downloaded from the [thrid-party repository](https://github.com/husarion/rosbot_description) provided by HUSARION.

* To run the simulation, first create a separate workspace and copy the "/src" folder in "/simulation_pkg" to your workspace. Then build the packages in the workspace.

* In "/src/bring_up/world/random32_map.world", Change the `$(PATH_TO_WORKSPACE)` in `uri` to the path to your the workspace folder.

* Run the simulation enironment by `roslaunch bring_up rosbot_bringup_random32.launch` to launch Gazebo and Rviz.

* Run the trajectory tracking by `roslaunch bring_up rosbot_traj_tracking.launch`.

