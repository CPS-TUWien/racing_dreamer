# ---------------------------------------------------------------------------------------------------------------------
# Author (original Version): Thomas Pintaric (thomas.pintaric@gmail.com)
# Author (modified Version): Andreas Brandstaetter
# SPDX-License-Identifier: GPL-3.0-or-later
# ---------------------------------------------------------------------------------------------------------------------

import os
import argparse
import yaml
import numpy as np
import scipy.interpolate as si
import cmapy
from skimage import io, morphology, img_as_ubyte
from scipy import ndimage
# import h5py

class CostmapReader:
    def __init__(self,
                 input_npz_path):

        self.data = np.load(input_npz_path)
        
        self.resolution = self.data["arr_0"][11]
        self.grid_origin_x = self.data["arr_0"][2]
        self.grid_origin_y = self.data["arr_0"][3]
        self.center_x = self.data["arr_0"][4]
        self.center_y = self.data["arr_0"][5]
        
# ===================================================================================================================
    def print_verbose(self):
        print("Verbose infos:")
        print("Starting position (world coordinates): [x,y] = [{:.2f}, {:.2f}]".format(
            self.data["arr_0"][0], self.data["arr_0"][1]
        ))
        print("Starting position (grid coordinates): [x,y] = [{:f}, {:f}]".format(
            self.data["arr_0"][2], self.data["arr_0"][3]
        ))
        print("Center position (grid coordinates): [x,y] = [{:.2f}, {:.2f}]".format(
            self.center_x, self.center_y
        ))
        print("Occupied threshold: {:.4f}".format(self.data["arr_0"][6]))
        #print("Min. pixel value*: {:8.4f} (*) before normalization".format(self.data["arr_0"][7]))
        #print("Max. pixel value*: {:8.4f}".format(self.data["arr_0"][8]))
        print("Dimensions: {:f} x {:f}".format(self.data["arr_0"][9], self.data["arr_0"][10]))
        print("Resolution: {:8.4f}".format(self.resolution))

    def get_cost(self, x, y, map_index = 0):	
        print("Get cost (map index {}) for position (world coordinates): [x,y] = [{:.2f}, {:.2f}]".format(map_index, x, y))
        x_pixel = x / self.resolution + self.grid_origin_x
        y_pixel = y / self.resolution + self.grid_origin_y
        
        return self.get_cost_grid_coordinates(x_pixel, y_pixel, map_index)

    def get_cost_grid_coordinates(self, x, y, map_index = 0):	
        map_name = "arr_{}".format(map_index + 1)
        cost_value = float(self.data[map_name][int(y)][int(x)])
        
        print("Get cost (map index {}) for position (grid coordinates): [x,y] = [{:.2f}, {:.2f}], value = {}".format(map_index, x, y, cost_value))
        
        return cost_value

# ======================================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='filename of the costmap self.dataset (.npz)')
    args = parser.parse_args()

    costmap = CostmapReader(input_npz_path=args.input)
    costmap.print_verbose()
    
    # map index
    #   0: drivable_area, 
    #   1: normalized_distance_from_start_line
    #   2: normalized_distance_to_target_smoothed
    #   3: normalized_distances_from_nearest_obstacle
    
    print("showing normalized_distance_from_start_line:")
    costmap.get_cost(0,0,1)
    costmap.get_cost(1,0,1)
    costmap.get_cost(0,1,1)
    costmap.get_cost(1,1,1)
    costmap.get_cost(0,-1,1)
    costmap.get_cost(-1,-1,1)

    print("showing normalized_distances_from_nearest_obstacle:")
    costmap.get_cost(0,0,3)
    costmap.get_cost(1,0,3)
    costmap.get_cost(0,1,3)
    costmap.get_cost(1,1,3)

    print("test:")
    costmap.get_cost(0,2,0)
    costmap.get_cost(0,2,1)
    costmap.get_cost(0,2,2)
    costmap.get_cost(0,2,3)

    costmap.get_cost_grid_coordinates(0,2,0)
    costmap.get_cost_grid_coordinates(0,2,1)
    costmap.get_cost_grid_coordinates(0,2,2)
    costmap.get_cost_grid_coordinates(0,2,3)

