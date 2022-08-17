#! /usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker

rospy.init_node('mesh_marker')

marker_pub = rospy.Publisher("/mesh_marker", Marker, queue_size = 2)

path = rospy.get_param('~mesh_path')

marker = Marker()

marker.header.frame_id = "odom"
marker.header.stamp = rospy.Time.now()
marker.ns = ""

# Shape (mesh resource type - 10)
marker.type = 10
marker.id = 0
marker.action = 0

print('path',path)
# Note: Must set mesh_resource to a valid URL for a model to appear
marker.mesh_resource = path
marker.mesh_use_embedded_materials = True

# Scale
marker.scale.x = 1.0
marker.scale.y = 1.0
marker.scale.z = 1.0

# Color
marker.color.r = 0.2
marker.color.g = 0.2
marker.color.b = 0.2
marker.color.a = 1.0

# Pose
marker.pose.position.x = 0
marker.pose.position.y = 0
marker.pose.position.z = -1
marker.pose.orientation.x = 0.0
marker.pose.orientation.y = 0.0
marker.pose.orientation.z = 0.0
marker.pose.orientation.w = 1.0

while not rospy.is_shutdown():
  marker_pub.publish(marker)
  rospy.rostime.wallsleep(1.0)
