# Collection of tracks

This repository is a collection of several tracks to be used in [f1tenth](https://f1tenth.org/)-style races. It can be directly used as [ROS](https://www.ros.org/)-package, where subsequently maps are referred in launch files of other packages.

## Usage

### As ROS-package
* Clone this repository into the `src/`-folder of your ROS workspace.
* Use the maps in launch files, e.g.:

        <arg name="map" default="$(find f1tenth_maps)/maps/f1_aut.yaml"/>
        <node pkg="map_server" name="map_server" type="map_server" args="$(arg map)"/>

### Costmap generator
The costmap generator in folder `costmaps` provides a script to calculate several metrics for the map. E.g.: Distance from start line, distance from obstacles, shortest path, spline optimzied path.

* Use the makefile to generate it for any map of folder `maps`, e.g.:

        make f1_aut

## Sources
* `berlin`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `columbia`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `columbia_simple`: redrawn based on `columbia-track`
* `columbia_small`: redrawn based on `columbia-track`
* `columbia-track`: [f1tenth/tracks](https://f1tenth.org/tracks/columbia-track.png)
* `f1_aut`: based on [Circuit_Red_Bull_Ring](https://de.wikipedia.org/wiki/Datei:Circuit_Red_Bull_Ring.svg)
* `f1_aut_wide`: based on [Circuit_Red_Bull_Ring](https://de.wikipedia.org/wiki/Datei:Circuit_Red_Bull_Ring.svg)
* `f1_esp`: based on [Formula1_Circuit_Catalunya](https://commons.wikimedia.org/wiki/File:Formula1_Circuit_Catalunya.svg)
* `f1_gbr`: based on [Silverstone_circuit](https://commons.wikimedia.org/wiki/File:Silverstone_circuit.svg)
* `f1_mco`: based on [Circuit_Monaco](https://commons.wikimedia.org/wiki/File:Circuit_Monaco.svg)
* `levine_blocked`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `levinelobby`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `levine`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `mtl`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `porto`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `skirk`: [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym/tree/cpp_backend_archive/maps)
* `stata_basement`: [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym/tree/cpp_backend_archive/maps)
* `torino`: [f1tenth/f1tenth_simulator](https://github.com/f1tenth/f1tenth_simulator/tree/master/maps)
* `torino_redraw`: redrawn based on `torino`
* `torino_redraw_small`: redrawn based on `torino`
* `torino_redraw_small_with_obstacles`: redrawn based on `torino`
* `unreal`: [pmusau17/f1tenth_gym_ros](https://github.com/pmusau17/f1tenth_gym_ros/tree/master/maps)
* `vegas`: [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym/tree/cpp_backend_archive/maps)

All other maps were created by us.

## Map files format
For yaml format see: [http://wiki.ros.org/map_server#YAML_format](http://wiki.ros.org/map_server#YAML_format)


