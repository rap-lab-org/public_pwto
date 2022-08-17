#!/usr/bin/env python3

import os
import rospy
import math
import numpy as np
import tf
from geometry_msgs.msg import Quaternion, PoseStamped, TwistStamped, Vector3, Twist
from styx_msgs.msg import Lane, Waypoint
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion


ua_lim = 2.2
uw_lim = 1

class TrajTracking(object):

    def __init__(self):
        rospy.init_node('waypoint_publisher', log_level=rospy.DEBUG)

        rospy.Subscriber('/odom', Odometry, self.callback_pose_vel,queue_size = 1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        self.wp_x = []
        self.wp_y = []
        self.wp_yaw = []
        self.wp_v = []
        self.wp_w = []
        self.current_pose = None
        self.current_vel = None

        self.scale = rospy.get_param('~scale')
        self.dt = rospy.get_param('~time_step')
        self.pubrate = 1/self.dt 

        self.stop_sign = 0

        self.loadWaypoints(rospy.get_param('~traj_load_path'))

        self.file_recordTraj = open(rospy.get_param('~traj_save_path'),"w")
        self.cmd_vel = Twist()


        self.dv_lim = ua_lim * self.dt 
        self.dw_lim = uw_lim * self.dt

        self.publish()


    def callback_pose_vel(self,data):
        self.current_pose =  data.pose.pose
        self.current_vel =  data.twist.twist


    def loadWaypoints(self,path):
        if os.path.isfile(path):
            file = open(path, "r")
            for eachLine in file:
                x, y, yaw, v, w, av, aw = eachLine.split()
                self.wp_x.append(float(x)*self.scale)
                self.wp_y.append(float(y)*self.scale)
                self.wp_yaw.append(float(yaw))
                self.wp_v.append(float(v)*self.scale)
                self.wp_w.append(float(w))
            rospy.loginfo('waypoint loaded')
        else:
            rospy.logerr('%s is not a file', path)

    def stopRobot(self,event):
        print('Stop the robot!')

    def publish(self):
        rate = rospy.Rate(self.pubrate)
        wp_len = len(self.wp_v)
        # rospy.Timer(rospy.Duration(10), rospy.signal_shutdown('**888'))
        k = 0
        while not rospy.is_shutdown():
            if self.current_pose is not None and self.current_vel is not None:
                while k<15:
                    self.cmd_vel.linear.x = 0
                    self.cmd_vel.angular.z = 0
                    print('Robot Ready!')
                    self.cmd_pub.publish(self.cmd_vel)
                    rate.sleep()
                    k+=1
                    v_cmd_prev = 0
                    w_cmd_prev = 0
                if not self.stop_sign:
                    for i in range(wp_len):
                        # current pose, velocity
                        current_x = self.current_pose.position.x
                        current_y = self.current_pose.position.y
                        current_yaw = euler_from_quaternion((self.current_pose.orientation.x, \
                                                            self.current_pose.orientation.y, \
                                                            self.current_pose.orientation.z, \
                                                            self.current_pose.orientation.w))[2]
                        current_v = np.sqrt((self.current_vel.linear.x)**2 + (self.current_vel.linear.y)**2)
                        current_w = self.current_vel.angular.z

                        self.file_recordTraj.write(str(current_x) + "  " + str(current_y) + "   " + str(current_yaw)+"   " \
                                                 + str(current_v) +"   " + str(current_w) +'\n')

                        # Control rule
                        pose_err = np.array([[ math.cos(current_yaw), math.sin(current_yaw), 0], \
                                             [-math.sin(current_yaw), math.cos(current_yaw), 0], \
                                             [0, 0, 1]]) \
                                 @ np.array([[self.wp_x[i] - current_x], \
                                             [self.wp_y[i] - current_y], \
                                             [self.wp_yaw[i]-current_yaw]])
                        print('pose_err',pose_err)

                        K_control = np.array([0.4,5,6]) # 32 by 32

                        v_cmd = self.wp_v[i]*math.cos(pose_err[2]) + K_control[0]*pose_err[0]
                        w_cmd = self.wp_w[i] + self.wp_v[i]*(K_control[1]*pose_err[1] + K_control[2]*math.sin(pose_err[2]))

                        if v_cmd - v_cmd_prev > self.dv_lim:
                            v_cmd = v_cmd_prev + self.dv_lim
                        elif v_cmd - v_cmd_prev < self.dv_lim:
                            v_cmd = v_cmd_prev - self.dv_lim

                        if w_cmd - w_cmd_prev > self.dw_lim:
                            w_cmd = w_cmd_prev + self.dw_lim
                        elif w_cmd - w_cmd_prev < self.dw_lim:
                            w_cmd = w_cmd_prev - self.dw_lim

                        # publish velocity command
                        self.cmd_vel.linear.x = v_cmd
                        self.cmd_vel.angular.z = w_cmd

                        print('Publishing linear vel',str(self.wp_v[i]))
                        print('Publishing angular vel',str(self.wp_w[i]))
                        print('step',i)
                        self.cmd_pub.publish(self.cmd_vel)
                        i+=1
                        v_cmd_prev = v_cmd
                        w_cmd_prev = w_cmd
                        
                        rate.sleep()
                    self.stop_sign = 1
                else:
                    self.file_recordTraj.close
                    self.cmd_vel.linear.x = 0
                    self.cmd_vel.angular.z = 0
                    print('Robot Stop!')
                    self.cmd_pub.publish(self.cmd_vel)


if __name__ == '__main__':
    # WaypointPublisher()
    try:
        TrajTracking()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not publish waypoint node.')
