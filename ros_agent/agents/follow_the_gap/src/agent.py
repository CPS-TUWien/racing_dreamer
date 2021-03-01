#!/usr/bin/env python

# -------------------------------------------------------
# Follow the gap controller for f1tenth 
# based on lab solution by Thomas Pintaric
# -------------------------------------------------------
# Author: Thomas Pintaric, Andreas Brandstaetter
# SPDX-License-Identifier: 0BSD
# -------------------------------------------------------

from __future__ import print_function
from scipy.ndimage import filters
import rospy
import argparse
import sys
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# ========================================
# PID
# ========================================

class PID():

    def __init__(self, kp=0, ki=0, kd=0, target_output_value=0):
        self.Kp = kp # proportional gain
        self.Ki = ki # integral gain
        self.Kd = kd # derivative gain
        self.I = 0 # integral term (initialized to zero, accumulated over successive calls)
        self.target_output_value = target_output_value
        self.previous_input_value = np.nan

    def change_tuning_parameters(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.Kp = kp
        if ki is not None:
            self.Ki = ki
            self.I = 0
        if kd is not None:
            self.Kd = kd

    def calculate(self, input_value, dt):
        error = self.target_output_value - input_value # steady-state error
        if dt <= 0:
            raise ValueError('dt>0 required')
        P = self.Kp * error
        self.I += self.Ki * error * dt
        D = 0 if np.isnan(self.previous_input_value) else \
            (self.Kd * (self.previous_input_value - input_value) / dt)
        self.previous_input_value = input_value
        return (P + self.I + D) if (self.Kp + self.Ki + self.Kd > 0) else (input_value)

# ========================================
# AgentNode
# ========================================

class AgentNode:

    def __init__(self, phase, identifier):
        self.phase = phase
        self.identifier = identifier
        scan_topic = "/scan"
        odom_topic = "/odom"
        drive_topic = "/nav"

        print("=== follow the gap node. ===");
        print("phase: {}".format(self.phase));
        print("identifier: {}".format(self.identifier));

        self.max_speed = 7.0   # meters/second
        self.max_decel = 8.26  # meters/second^2
        self.max_stopping_distance = np.square(self.max_speed) / (2.0 * self.max_decel)
        self.lookahead_distance = 2.0 * self.max_stopping_distance
        self.use_lookahead_distance_only_for_visualization = False
        self.vehicle_width = (0.3302 * 1.2) # use 120% of the wheelbase as vehicle width

        self.last_lidar_timestamp = rospy.Time()
        self.dt_threshold = 1.0 / 50.0 # run computation at max. 50 Hz

        self.forward_scan_arc = (np.deg2rad(-90.0), np.deg2rad(+90.0))
        self.heading_computation_arc = np.deg2rad(30.0)
        self.heading_computation_percentile = 100 * (1.0 - (self.heading_computation_arc / \
                                                            (self.forward_scan_arc[1] - self.forward_scan_arc[0])))

        self.minimum_gap_lenth = 0.2 # units: m
        self.median_range_deviation_threshold = 9.0 # outlier test: x[i] > median(x) * median_deviation_threshold

        self.pid_controller = PID()
        self.pid_controller.change_tuning_parameters(kp=1.4, ki=0.0, kd=0.1)
#        self.dyn_reconf_server = DynamicReconfigureServer(ConfigType, self.reconfigure)

        #self.linear_velocity = 0.0 # units: rad/s
        #self.angular_velocity = 0.0 # units: m/s
        self.heading_error = 0.0
        self.last_heading_timestamp = rospy.Time()

        self.steering_angle = 0.0
        self.vehicle_speed = 0.0
        self.max_vehicle_speed = 6.0

        self.max_steering_angle = np.deg2rad(24) # maximum (absolute) steering angle
        self.dt_threshold = 1.0 / 100.0 # run PID at 100 Hz (max.)

        self.last_odom_timestamp = 0
        self.last_drive_timestamp = 0
        self.last_race_info_timestamp = 0
        self.last_lap_count = 0
        self.last_lap_time = 0
        self.last_lap_timestamp = 0

        rospy.Subscriber(scan_topic, LaserScan, self.laserscan_callback, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odometry_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(name=drive_topic, data_class=AckermannDriveStamped, queue_size=1)

    def get_lidar_scan_arc(self, data, angle_start, angle_end):
        if angle_start < data.angle_min or angle_end > data.angle_max or angle_start >= angle_end:
            raise ValueError('requested angles are out-of-range')
        subrange = np.divide(np.subtract((angle_start, angle_end), data.angle_min), \
                            data.angle_increment).astype(np.int)
        angles = np.add(np.multiply(np.arange(subrange[0],subrange[1]+1).astype(np.float), \
                                    data.angle_increment), data.angle_min)
        ranges = np.array(data.ranges[subrange[0]:subrange[1]+1])
        return angles, ranges, subrange

    def laserscan_callback(self, scan_msg: LaserScan):
        has_previous_timestamp = not self.last_lidar_timestamp.is_zero()
        current_timestamp = rospy.Time(secs=scan_msg.header.stamp.secs, nsecs=scan_msg.header.stamp.nsecs)

        if not has_previous_timestamp:
            self.last_lidar_timestamp = current_timestamp
            return

        dt = current_timestamp.to_sec() - self.last_lidar_timestamp.to_sec()

        if dt <= self.dt_threshold:
            return

        self.last_lidar_timestamp = current_timestamp

        angles, ranges, _ = self.get_lidar_scan_arc(scan_msg, \
                                                    self.forward_scan_arc[0], \
                                                    self.forward_scan_arc[1])

        if not self.use_lookahead_distance_only_for_visualization:
            ranges = np.clip(ranges, a_min=0.0, a_max=self.lookahead_distance)

        diff_ranges = np.abs(np.ediff1d(ranges))

        filter_scan_arc = np.deg2rad(10.0) # units: radians
        filter_width = int(filter_scan_arc / scan_msg.angle_increment) # specify filter width in terms of the circular scan arc
        median_filtered_diff_ranges = filters.median_filter(input=diff_ranges.flatten(), size=(filter_width,), mode='nearest')
        max_filtered_diff_ranges = filters.maximum_filter1d(input=diff_ranges.flatten(), size=filter_width)

        mask = np.logical_and(np.equal(diff_ranges, max_filtered_diff_ranges), \
                              np.greater(diff_ranges, np.multiply(median_filtered_diff_ranges, \
                                                                  self.median_range_deviation_threshold)))
        mask = np.logical_and(mask, np.greater(diff_ranges, self.minimum_gap_lenth))

        masked_angles = np.ma.masked_where(np.logical_not(mask), angles[:-1])

        def enumerate_masked_array(masked_array):
            mask = ~masked_array.mask.ravel()
            for i, m in zip(np.ndenumerate(masked_array), mask):
                if m: yield i

        adjusted_ranges = np.copy(ranges)

        for i, theta in enumerate_masked_array(masked_angles):
            short_side = self.vehicle_width
            long_side = np.amin(ranges[i[0]-1:i[0]+2])
            beta = np.arccos((2.0 * np.square(long_side) - np.square(short_side)) / (2.0 * np.square(long_side)))
            gamma = np.array((theta - beta, theta + beta))
            index_range = np.divide(np.subtract((gamma[0], gamma[1]), angles[0]), \
                                    scan_msg.angle_increment).astype(np.int)

            index_range = np.clip(index_range, 0, len(ranges)-1)
            for x in np.nditer(adjusted_ranges[index_range[0]:index_range[1] + 1], op_flags=['readwrite']):
                x[...] = min(x, long_side)

        if self.use_lookahead_distance_only_for_visualization:
            adjusted_ranges = np.clip(adjusted_ranges, a_min=0.0, a_max=self.lookahead_distance)

        # Publish the vehicle's target heading
        heading_msg = Float32()
        indices = np.digitize(adjusted_ranges, [0.0, np.percentile(adjusted_ranges, q=self.heading_computation_percentile), scan_msg.range_max])
        heading_msg.data = np.mean(angles[indices==2])
        heading_distance = np.mean(ranges[indices==2])
        #self.heading_pub.publish(heading_msg)
        #print('Computed heading: {}'.format(heading_msg.data))
        self.publish_drive_from_heading(heading_msg, heading_distance)

    def odometry_callback(self, odom_msg: Odometry):
        if self.last_odom_timestamp != odom_msg.header.stamp.secs:
            self.last_odom_timestamp = odom_msg.header.stamp.secs
            print('Got odom: sec={}, nsec={}, x={}, y={}, v={}'.format(odom_msg.header.stamp.secs, odom_msg.header.stamp.nsecs, odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.twist.twist.linear.x))

    def publish_drive_from_heading(self, data, heading_distance):
        self.heading_error = data.data

        has_previous_timestamp = not self.last_heading_timestamp.is_zero()
        current_timestamp = rospy.Time.now() # NOTE: Float32 message has no header/timestamp field

        if not has_previous_timestamp:
            self.last_heading_timestamp = current_timestamp
            return

        dt = current_timestamp.to_sec() - self.last_heading_timestamp.to_sec()

        if dt <= self.dt_threshold:
            return

        self.last_heading_timestamp = current_timestamp

        # Execute PID control loop
        control_value = self.pid_controller.calculate(self.heading_error, dt=dt)

        self.steering_angle = -control_value

        self.steering_angle = np.clip(self.steering_angle, \
                                      -np.abs(self.max_steering_angle), \
                                      +np.abs(self.max_steering_angle))

        abs_steering_angle = np.abs(self.steering_angle)
        self.vehicle_speed = self.max_vehicle_speed

        speed_limit_angle_thr = np.deg2rad(5)
        if abs_steering_angle > speed_limit_angle_thr:
            self.vehicle_speed = self.max_vehicle_speed - (abs_steering_angle / self.max_steering_angle) * (self.max_vehicle_speed * 0.30)
        if heading_distance < 5:
            self.vehicle_speed = min(self.vehicle_speed, heading_distance / 5 * 4)
        self.vehicle_speed = max(self.vehicle_speed, 1.5) # drive at least with speed 1.5


        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        #drive_msg.header.frame_id = "" # leave blank
        drive_msg.drive.steering_angle = self.steering_angle
        drive_msg.drive.speed = self.vehicle_speed
        self.drive_pub.publish(drive_msg)
        if self.last_drive_timestamp != drive_msg.header.stamp.secs:
            self.last_drive_timestamp = drive_msg.header.stamp.secs
            print('Published drive: angle={}, speed={} based on heading={}, dist={}'.format(self.steering_angle, self.vehicle_speed, data.data, heading_distance))


# /scan: The ego agent's laser scan
# /odom: The ego agent's odometry
# /opp_odom: The opponent agent's odometry
# /opp_scan: The opponent agent's laser scan (only available on the multi_node branch)
# /map: The map of the environment
# /race_info: Information of the environment including both agents' elapsed runtimes, both agents' lap count, and both agents' collsion info. Currently, the race ends after both agents finish two laps, so the elapsed times will stop increasing after both lap counts are > 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', help='race phase (one of "1a", "1b", "2" or "auto")', default="auto")
    parser.add_argument('--identifier', help='car identifiert (one of "ego" or "opp")', default="ego")
    args = parser.parse_args()

    rospy.init_node('follow-the-gap_agent', anonymous=True)
    agent = AgentNode(phase=args.phase, identifier=args.identifier)
    rospy.spin()
