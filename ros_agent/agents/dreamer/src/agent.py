import rospy
import argparse
import numpy as np
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped
import pathlib

from models import RacingAgent


class AgentNode:

    def __init__(self, hardware, agent, checkpoint):
        self.hardware = hardware
        self.agent = agent
        self.checkpoint = checkpoint

        self._laserscan_last_time = rospy.Time(0)
        self._motor = 0
        self._steering = 0

        self._joy_msg = 0;
        self._config_a = 75;
        self._config_b = 60;

        self._first_scan = 1;

        scan_topic = "/scan"
        odom_topic = "/odom"

        print("=== dreamer node. ===");
        if self.hardware == 'car':
            drive_topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
        elif self.hardware == 'other':
            drive_topic = "/nav"
        else:
            raise NotImplementedError("hardware \"{}\" not implemented".format(self.hardware))

        # init agent
        self._dreamer_state = None
#        self._agent = RacingDreamer(pathlib.Path("/cwd/checkpoints/checkpoint_3630808steps_return99.5"))
#        self._agent = RacingDreamer(pathlib.Path("/cwd/checkpoints/treitlstrasse_613380"))
        self._agent = RacingAgent(algorithm=self.agent, checkpoint_path=self.checkpoint)


        # queue_size=1 --> ROS will discard messages if they arrive faster than they are processed by the callback function
        self._scan_sub = rospy.Subscriber(scan_topic, LaserScan, self._laserscan_callback, queue_size=1)
        self._joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_callback, queue_size=1)
        self._drive_pub = rospy.Publisher(name=drive_topic, data_class=AckermannDriveStamped, queue_size=1)
        self._ncp_status_pub = rospy.Publisher(name="/ncp_status", data_class=Float32MultiArray, queue_size=1)
        self._ncp_adj_pub = rospy.Publisher(name="/ncp_adj", data_class=Float32MultiArray, queue_size=1)
        self._scan_pub = rospy.Publisher(name="/scan_noised", data_class=LaserScan, queue_size=1)
        
    def _laserscan_callback(self, scan_msg: LaserScan):
        since_last_laserscan = scan_msg.header.stamp - self._laserscan_last_time
        if since_last_laserscan.to_sec() < (0.08 - 0.001): # limit to approx. 10Hz
            return

        self._laserscan_last_time = scan_msg.header.stamp
        #print('dreamer received LIDAR scan @ rate:', since_last_laserscan.to_sec())
        observation = dict()
        observation['lidar'] = np.flip(np.array(scan_msg.ranges))

        #print("observation = ", observation);
        obs_lidar = observation['lidar'][0:1080,]
        #print("observation['lidar'] = ", obs_lidar)
        #print("observation['lidar'].shape = ", obs_lidar.shape)
        extra_noise_stddev = 0.3 # 0.3m
        extra_noise = np.random.normal(0, extra_noise_stddev, 1080)

        observation['lidar'] = obs_lidar # + extra_noise # this would add extra gaussian noise
        scan_noised = scan_msg
        scan_noised.ranges = np.flip(observation['lidar'])
        self._scan_pub.publish(scan_noised)

        proc_ranges = np.array(obs_lidar)
        proc_ranges = np.clip(proc_ranges,None,4) # clip LIDAR distances at max. 4m
        proc_ranges[3:-3] = 1.0/3*(proc_ranges[3:-3]+proc_ranges[2:-4]+proc_ranges[4:-2])
        aim_max = max(proc_ranges[int(1080/2 - 30):int(1080/2 + 30)])
        forward_max = max(proc_ranges[int(1080/2 - 150):int(1080/2 + 150)])
        #print("min = ", np.min(obs_lidar), " max = ", np.max(obs_lidar), " forward_max = ", forward_max)

        before = rospy.Time.now()
        #print("before action ", before)
        #action = {'motor': 0, 'steering': 0};
        action, self._dreamer_state = self._agent.action(observation, self._dreamer_state)
        after = rospy.Time.now()
        duration = after - before
        #print("after action ", after, " duration ", after - before)
        #print("action = ", action)
        #print("state = ", state);

        if float(action['motor']) < 0.5:
            self._motor = self._motor - float(self._config_b)/1000 #0.05
        else:
            self._motor = self._motor + float(self._config_a)/1000 #0.065
        #self._motor = self._motor + (float(action['motor']) - 0,44) / 8,7)
        #self._motor = self._motor + (float(action['motor']) - (float(self._config_a)/100)) / (float(self._config_b)/10)
        # self._motor = self._motor + (float(action['motor']) - 0.42) / 5
        if self._motor > 5:
            self._motor = 5
        if self._motor < 1.7:
            self._motor = 1.7
        # self._motor = 1.5
        # motor = (float(action['motor']) * 2 + 0.5)

        #steering = 0 - float(action['steering'] * (float(self._config_a) / 10) * 0.42) # working better in hardware
        steering = 0 - float(action['steering'] * 0.6 * 0.42) # working better in hardware
        # steering = 0 - float(action['steering'] * 0.7 * 0.42) # working better in simulation
        # self._steering = float(self._steering) * 2 / 3 + float(steering) * 1 / 3 # lowpass in hardware
        #div = (self._motor * 1.5) + 0.5
        div = 20
        val = self._config_a
        val = 18 - forward_max * 3
        self._steering = float(self._steering) * (div - val) / (div) + float(steering) * (val) / (div) # lowpass in hardware
        # self._steering = float(self._steering) * 1 / 6 + float(steering) * 5 / 6 # lowpass in simulation
        # self._steering = float(steering) # direct feed
        #self._steering = self._steering + float(steering) / self._config_a # integrating
        #if self._steering > 0.42:
        #    self._steering = 0.42
        #if self._steering < -0.42:
        #    self._steering = -0.42

        #if self._joy_msg.axes[6] > 0.5: # add pertubation
        #    self._steering = 0.2

        print("DREAMER({}) rate {:5.2f}Hz | dur {:4.3f}s | a.motor {:5.2f} | a.steer {:5.2f} | m.vel {:4.3f}m/s | m.steer {:7.3f} | a {:2.0f} | b {:2.0f}".format(self.agent, 1000000000.0/since_last_laserscan.to_nsec(), duration.to_sec(), float(action['motor']), float(action['steering']), self._motor, self._steering, self._config_a, self._config_b))

        #self._motor = 1.5 # safety for debug
        drive_msg = self._convert_action(self._steering, self._motor)
        #drive_msg = self._convert_action(steering, self._motor)
        self._drive_pub.publish(drive_msg)

        if self.agent == 'ncp':
            if self._first_scan % 100 == 5:
                adj_matrix = np.array(self._agent._agent._ncp._wirings.adjacency_matrix)
                flatten_adjacency_matrix = adj_matrix.flatten()
                adjacency_matrix = Float32MultiArray()
                adjacency_matrix.data = flatten_adjacency_matrix
                self._ncp_adj_pub.publish(adjacency_matrix)
            self._first_scan = (self._first_scan + 1) % 100
            neuron_states = Float32MultiArray()
            neuron_states.data = self._dreamer_state[1].numpy()[0,:]
            self._ncp_status_pub.publish(neuron_states)

    def _convert_action(self, steering_angle, speed) -> AckermannDriveStamped:
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        #print('dreamer published action: steering_angle = ', steering_angle, "; speed = ", speed)
        return drive_msg

    def joy_callback(self, joy_msg):
        if self._joy_msg == 0:
            self._joy_msg = joy_msg;
            return;

        if joy_msg.axes[6] != self._joy_msg.axes[6] and joy_msg.axes[6] < -0.5: # right
            self._config_a = self._config_a + 1
        if joy_msg.axes[6] != self._joy_msg.axes[6] and joy_msg.axes[6] > 0.5: # left
            self._config_a = self._config_a - 1
        if joy_msg.axes[7] != self._joy_msg.axes[7] and joy_msg.axes[7] < -0.5: # down
            self._config_b = self._config_b - 1
        if joy_msg.axes[7] != self._joy_msg.axes[7] and joy_msg.axes[7] > 0.5: # up
            self._config_b = self._config_b + 1
        if joy_msg.buttons[2] != self._joy_msg.buttons[2] and joy_msg.buttons[2] == 1: # button X
            self._config_a = 0
            self._config_b = 0
        if joy_msg.buttons[3] != self._joy_msg.buttons[3] and joy_msg.buttons[3] == 1: # button Y
            self._config_a = 7
            self._config_b = 7

        self._joy_msg = joy_msg;

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', help='determine if running on hardware car', default="unset")
    parser.add_argument("--agent", choices=['dreamer', 'ncp'], required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help='directory where pickle files are located')
    parser.add_argument("--action_repeat", type=int, default=8, help='number of repeatition of the same action')
    parser.add_argument("--time_limit", type=float, default=100.0, help='max time in seconds')
    args = parser.parse_args()

    rospy.init_node('dreamer_agent', anonymous=True)
    agent = AgentNode(args.hardware, args.agent, args.checkpoint)
    rospy.spin()
