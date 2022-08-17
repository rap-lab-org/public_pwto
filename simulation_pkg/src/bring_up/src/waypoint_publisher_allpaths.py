#!/usr/bin/env python3

import os
import rospy
import math
import numpy as np
import tf
from geometry_msgs.msg import Quaternion, PoseStamped, TwistStamped, Vector3
from styx_msgs.msg import Lane, Waypoint
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion


init_x = 0
init_y = 0

class WaypointPublisher(object):

    def __init__(self):
        rospy.init_node('waypoint_publisher', log_level=rospy.DEBUG)

        self.path_name = rospy.get_param('~path_name')
        self.pub_path = rospy.Publisher(self.path_name, Path, queue_size=1, latch=True)

        self.wp_x = 0
        self.wp_y = 0
        self.wp_yaw= 0
        self.wp_v = 0
        self.wp_w = 0

        self.base_wp = []
        self.base_path = Path()
        self.base_path.header.frame_id = 'odom'

        self.new_waypoint_publisher(rospy.get_param('~path'))
        rospy.spin()

    def new_waypoint_publisher(self,path):
        if os.path.isfile(path):
            self.load_waypoints(path)
            self.publish()
            rospy.loginfo('waypoint loaded')
        else:
            rospy.logerr('%s is not a file', path)

    def load_waypoints(self,path):
        file = open(path, "r")
        #rate = rospy.Rate(10)
        for eachLine in file:
            x, y, yaw, v, w, av, aw = eachLine.split()
            self.wp_x = float(x)*rospy.get_param('~scale')
            self.wp_y = float(y)*rospy.get_param('~scale')
            self.wp_yaw= float(yaw)
            self.wp_v= float(v)*rospy.get_param('~scale')
            self.wp_w= float(w)

            wp = Waypoint()
            wp.pose.pose.position.x = self.wp_x
            wp.pose.pose.position.y = self.wp_y
            wp_quat = self.quaternion_from_yaw(self.wp_yaw)
            wp.pose.pose.orientation = Quaternion(*wp_quat)
            wp.twist.twist.linear.x = self.wp_v
            wp.twist.twist.angular.z = self.wp_w
            wp.forward = True
            self.base_wp.append(wp)

            path_element = PoseStamped()
            path_element.pose.position.x = wp.pose.pose.position.x + init_x
            path_element.pose.position.y = wp.pose.pose.position.y + init_y
            self.base_path.poses.append(path_element)

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def publish(self):
        self.pub_path.publish(self.base_path)

if __name__ == '__main__':
    try:
        WaypointPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not publish waypoint node.')
