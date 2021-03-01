# f1tenth_dreamer

## How to use

### run with ROS environment

#### install prerequisites

- install ROS
```
sudo apt-get install ros-noetic-desktop ros-noetic-tf2-geometry-msgs ros-noetic-ackermann-msgs ros-noetic-joy ros-noetic-map-server
```
- install maps and f1tenth simulator (not required on race car)
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/f1tenth/f1tenth_simulator.git
git clone https://github.com/CPS-TUWien/f1tenth_maps.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

#### build this Repo

- Build the docker image (*takes a while*)
```
make build-dreamer
```
or for another demo:
```
make build-follow_the_gap
```

#### run on race car

- requires working setup of `f1tenth_system` (see [https://github.com/f1tenth/f1tenth_system](https://github.com/f1tenth/f1tenth_system))
- start using roslaunch
```
roslaunch f1tenth_dreamer teleop_docker.launch
```

#### run in simulator

- start the f1tenth simulator (simultanously in a separate terminal)
```
source devel/setup.bash
roslaunch f1tenth_dreamer simulator.launch
```
- start the docker with dreamer (or any other) agent (again in the other terminal)
```
make run-dreamer
```
or to use NCP and the specified checkpoint
```
make run-dreamer AGENT_SELECT=ncp AGENT_CHECKPOINT=treitlstrasse_ncp_32_24_12_2
```

### Generate video (without ROS)

* Build the docker image (*takes a while*)
```
make build-no_ros_make_video
```
- Run the agent and render video into current working direktory ...
```
make video-demo
```

## References

- [https://github.com/f1tenth/f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros): F1TENTH gym environment ROS communication bridge
- [https://github.com/CPS-TUWien/arc_agent_ros](https://github.com/CPS-TUWien/arc_agent_ros): Agent-ROS
- [https://github.com/f1tenth/f1tenth_system](https://github.com/f1tenth/f1tenth_system): Drivers and system level code for the F1TENTH vehicles

