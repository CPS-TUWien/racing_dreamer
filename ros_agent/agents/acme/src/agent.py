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
        self._config_a = 38;
        self._config_b = 69;

        self._first_scan = 1;

        scan_topic = "/scan"
        odom_topic = "/odom"

        print("=== acme node. ===");
        if self.hardware == 'car':
            drive_topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
        elif self.hardware == 'other':
            drive_topic = "/nav"
        else:
            raise NotImplementedError("hardware \"{}\" not implemented".format(self.hardware))

        # init agent
        self._agent_state = None
        self._agent = RacingAgent(algorithm=self.agent, checkpoint_path=self.checkpoint)


        # queue_size=1 --> ROS will discard messages if they arrive faster than they are processed by the callback function
        self._scan_sub = rospy.Subscriber(scan_topic, LaserScan, self._laserscan_callback, queue_size=1)
        self._joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_callback, queue_size=1)
        self._drive_pub = rospy.Publisher(name=drive_topic, data_class=AckermannDriveStamped, queue_size=1)
        self._ncp_status_pub = rospy.Publisher(name="/ncp_status", data_class=Float32MultiArray, queue_size=1)
        self._ncp_adj_pub = rospy.Publisher(name="/ncp_adj", data_class=Float32MultiArray, queue_size=1)
        
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
#        print("min = ", np.min(obs_lidar), " max = ", np.max(obs_lidar))
        observation['lidar'] = obs_lidar

        before = rospy.Time.now()
#        print("before action ", before)
        #action = {'motor': 0, 'steering': 0};
        action, self._agent_state = self._agent.action(observation, self._agent_state)
        after = rospy.Time.now()
        duration = after - before
#        print("after action ", after, " duration ", after - before)

        #print("action = ", action)

        #print("state = ", state);

        self._motor = self._motor + (float(action['motor']) - (float(self._config_a)/100)) / (float(self._config_b)/10)
        # self._motor = self._motor + (float(action['motor']) - 0.42) / 5
        if self._motor > 3:
            self._motor = 3
        if self._motor < 1.5:
            self._motor = 1.5
        # motor = (float(action['motor']) * 2 + 0.5)

        steering = 0 - float(action['steering'] * 0.4 * 0.42) # working better in hardware
        # steering = 0 - float(action['steering'] * 0.7 * 0.42) # working better in simulation
        # self._steering = float(self._steering) * 2 / 3 + float(steering) * 1 / 3 # lowpass in hardware
        div = (self._motor * 1.5) + 0.5
        self._steering = float(self._steering) * (div - 1) / (div) + float(steering) * (1) / (div) # lowpass in hardware
        # self._steering = float(self._steering) * 1 / 6 + float(steering) * 5 / 6 # lowpass in simulation
        # self._steering = float(steering) # direct feed

        print("ACME({}) rate {:5.2f}Hz | dur {:4.3f}s | a.motor {:5.2f} | a.steer {:5.2f} | m.vel {:4.3f}m/s | m.steer {:7.3f} | a {:2.0f} | b {:2.0f}".format(self.agent, 1000000000.0/since_last_laserscan.to_nsec(), duration.to_sec(), float(action['motor']), float(action['steering']), self._motor, self._steering, self._config_a, self._config_b))

        #self._motor = 1.5 # safety for debug
        drive_msg = self._convert_action(self._steering, self._motor)
#        drive_msg = self._convert_action(steering, self._motor)
        self._drive_pub.publish(drive_msg)

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

        #if joy_msg.buttons != self._joy_msg.buttons:
        #    print("safety_brake: joy callback: ", joy_msg)

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
    parser.add_argument("--agent", choices=['mpo', 'd4pg'], required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help='directory where checkpoint files are located')
    args = parser.parse_args()

    rospy.init_node('acme_agent', anonymous=True)
    agent = AgentNode(args.hardware, args.agent, args.checkpoint)
    rospy.spin()
