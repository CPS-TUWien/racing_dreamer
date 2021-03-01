import rospy
import argparse
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class AgentNode:

    def __init__(self, phase, identifier):
        self.phase = phase
        self.identifier = identifier
        scan_topic = "/scan"
        odom_topic = "/odom"
        drive_topic = "/nav"

        print("=== demo_agent node. ===");
        print("phase: {}".format(self.phase)); # this will not be available in the actual race!
        print("identifier: {}".format(self.identifier));

        rospy.Subscriber(scan_topic, LaserScan, self._laserscan_callback, queue_size=1)
        self._drive_pub = rospy.Publisher(name=drive_topic, data_class=AckermannDriveStamped, queue_size=1)

        #self._agent = Agent.load(directory='model')
        #with open("/ws/src/f1tenth_agent_ros/model/model/.gitkeep") as f:
        #    content = f.read()
        #    print("/ws/src/f1tenth_agent_ros/model/model/.gitkeep content: {}".format(content));

    def _laserscan_callback(self, scan_msg: LaserScan):
        print('demo_agent received LIDAR scan.')
        obs = scan_msg.ranges
        #action = self._agent.act(obs)
        action = [0.0, 5.0] # just drive straigth s.t. car will crash into the wall
        drive_msg = self._convert_action(action)
        self._drive_pub.publish(drive_msg)

    def _convert_action(self, action) -> AckermannDriveStamped:
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = action[0]
        drive_msg.drive.speed = action[1]
        return drive_msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', help='race phase', default="default_phase")
    parser.add_argument('--identifier', help='car identifier', default="default_identifier")
    args = parser.parse_args()

    rospy.init_node('demo_agent', anonymous=True)
    agent = AgentNode(phase=args.phase, identifier=args.identifier)
    rospy.spin()
