#!/usr/bin/env python3
import rospy
import numpy as np
import nav_msgs.msg as nav
import std_msgs.msg as std
import ackermann_msgs.msg as ackermann
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
import math

# based on example from https://answers.ros.org/question/11135/plotting-a-markerarray-of-spheres-with-rviz/

class MarkerTest(object):

    def __init__(self):
        self._count = 0
        self._max = 0
        self._min = 0
        
        # self._adj_matrix = np.zeros((38, 38))
    
        self.ncp_status_sub = rospy.Subscriber("/ncp_status", Float32MultiArray, self.data_callback, queue_size=1)
        self.ncp_adj_sub = rospy.Subscriber("/ncp_adj", Float32MultiArray, self.adj_callback)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
        self.arrow_pub = rospy.Publisher('/visualization_arrow_array', MarkerArray, queue_size=1)
        print("marker_translate: INIT")

    def pos_by_index(self, index):
        if index < 2:
            pos_x = (index % 2) * 1.5  + 5
            pos_y = -15
        elif index < 2 + 9:
            pos_x = ((index - 2)) * 1.5 + 2
            pos_y = -10 + ((index - 2) % 2) * 1.5
        elif index < 2 + 9 + 12:
            pos_x = ((index - 2 - 9)) * 1.5
            pos_y = -5
        else:
            pos_x = (index % 5) * 1.5
            pos_y = math.floor(index / 5) * 1.5
        return pos_x, pos_y

    def adj_callback(self, data_msg):
        data_arr = np.array(data_msg.data)
        print("adj_callback: data_callback ", data_arr, data_arr.shape)
        sz = int(math.sqrt(data_arr.shape[0]))
        
        self._adj_matrix = data_arr.reshape(sz, sz)
        print("adj_callback: self._adj_matrix ", self._adj_matrix, self._adj_matrix.shape)
        
        k = 0
        self._arrowArray = MarkerArray()
        for i in range(self._adj_matrix.shape[0]):
            for j in range(self._adj_matrix.shape[0]):
                pos_x_i, pos_y_i = self.pos_by_index(i)
                pos_x_j, pos_y_j = self.pos_by_index(j)
                
                if abs(self._adj_matrix[i,j]) > 0.1:
                    arrow = Marker()
                    arrow.id = k
                    arrow.header.frame_id = "laser"
                    arrow.type = arrow.ARROW
                    arrow.action = arrow.ADD
                    arrow.color.a = 1.0
                    arrow.color.r = 0.8
                    arrow.color.g = 0.8
                    arrow.color.b = 0.8
                    arrow.points = [Point(), Point()]
                    arrow.points[0].x = pos_x_i
                    arrow.points[0].y = pos_y_i
                    arrow.points[0].z = 1.0
                    arrow.points[1].x = pos_x_j
                    arrow.points[1].y = pos_y_j
                    arrow.points[1].z = 1.0
                    arrow.pose.orientation.w = 1.0
                    arrow.pose.position.x = 0
                    arrow.pose.position.y = 0
                    arrow.pose.position.z = 0
                    arrow.scale.x = 0.05
                    arrow.scale.y = 0.2
                    arrow.scale.z = 0.5
                    k += 1

                    self._arrowArray.markers.append(arrow)

        self.arrow_pub.publish(self._arrowArray)
    
    
    def data_callback(self, data_msg):
        data_arr = np.array(data_msg.data)
        print("marker_translate: data_callback ", data_arr.shape[0])

        for i in range(data_arr.shape[0]):
            self._max = max(self._max, data_arr[i])
            self._min = min(self._min, data_arr[i])

        self._markerArray = MarkerArray()
        for i in range(data_arr.shape[0]):
            scaled = (data_arr[i] - self._min) / (self._max - self._min)
            pos_x, pos_y = self.pos_by_index(i)

            marker = Marker()
            marker.id = i
            marker.header.frame_id = "laser"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2 + (scaled / 2)
            marker.scale.y = 0.2 + (scaled / 2)
            marker.scale.z = 0.2 + (scaled / 2)
            marker.color.a = 1.0
            marker.color.r = scaled
            marker.color.g = 0.5
            marker.color.b = 1 - scaled
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = pos_x
            marker.pose.position.y = pos_y
            marker.pose.position.z = 1

            self._markerArray.markers.append(marker)

        self.marker_pub.publish(self._markerArray)

def main():
    rospy.init_node('marker_translate')

    mn = MarkerTest()
    rospy.spin()

if __name__ == '__main__':
    main()
