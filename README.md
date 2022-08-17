



### Installation

* Download and compile [EMOA*](https://github.com/wonderren/public_emoa) by following the instructions there. The downloading directory is suggested to be "../public_emoa" with respect to the directory of this readme file. Otherwise, context.py may need to be changed to get this package running (not tested).

* Install [cyipopt](https://cyipopt.readthedocs.io/en/stable/install.html#) by following the section "On Ubuntu 18.04 Using APT Dependencies".

* Install [opty](https://opty.readthedocs.io/en/latest/) by `pip3 install opty`

* Run pip3 install for any missing packages, e.g. sympy.

* File run_example.py provides an entry point to the PWTO implementation.


### Run SImulation

* Create a separate workspace and copy the "/src" folder in "/simulation_pkg" to your workspace.

* Build the packages in the workspace.

* Change the `uri` link in "/src/bring_up/world/random32_map.world" to your local path.

* run the simulation by `roslaunch bring_up rosbot_bringup_random32.launch` to launch Gazebo and Rviz.

* run the trajectory tracking by `roslaunch bring_up rosbot_traj_tracking.launch`.

* To plot the trajectory tracking results, run `python3 plot_traj.py`. Change the 'traj_load_path' and 'traj_save_path' if necessary.