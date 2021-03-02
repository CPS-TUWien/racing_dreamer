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

class CostmapGenerator:
    def __init__(self,
                 starting_position,
                 input_yaml_path,
                 output_path,
                 export_images = False,
                 export_debug = False):

        self.world_starting_position = np.asarray(starting_position)
        self.export_images = export_images
        self.export_debug = export_debug
        self.verbose = True

        with open(input_yaml_path) as file:
            self.map_properties = yaml.safe_load(file)

        self.output_path = output_path
        self.basename, _ = os.path.splitext(os.path.basename(input_yaml_path))
        self.output_filename_root, _ = os.path.splitext(self.output_path)
        self.input_directory = os.path.dirname(input_yaml_path)

        image_filename = os.path.join(self.input_directory, self.map_properties['image'])
        self.image = io.imread(image_filename, as_gray=True).astype(np.float)
        self.image_center = (self.image.shape + np.array(self.map_properties['origin'])[0:2] /
                             self.map_properties['resolution']).astype(np.int)
        self.normalized_image = self.image / np.amax(self.image.flatten()) # normalize image
        self.binary_image = self.normalized_image > (self.map_properties['occupied_thresh'])
        
        # extra for optimized spline at Treitlstrasse_3-U_v3
        self.binary_image[987, 1294] = 0; # first turn
        #self.binary_image[930, 1300] = 0; # second turn at narrow passage

        self.grid_starting_position = np.divide(self.world_starting_position - np.array(self.map_properties['origin'])[0:2],
                                                self.map_properties['resolution'])
        self.grid_starting_position[1] = self.image.shape[1] - self.grid_starting_position[1] - 1
        self.grid_starting_position = self.grid_starting_position.astype(int)
        
        # draw starting position as circle, 1-meter grid and x/y-axis
        self.start_and_scale = np.copy(self.binary_image)
        self.start_and_scale = self.start_and_scale * 0.5 + 0.5
        iter_x = 0	
        while iter_x < self.image.shape[1]:
            iter_y = 0
            while iter_y < self.image.shape[0]:
                if (iter_y - self.grid_starting_position[1]) * self.map_properties['resolution'] % 1 == 0 or \
                   (iter_x - self.grid_starting_position[0]) * self.map_properties['resolution'] % 1 == 0 :
                    self.start_and_scale[iter_y][iter_x] = self.start_and_scale[iter_y][iter_x] * 0.85 + (1 - self.start_and_scale[iter_y][iter_x]) * 0.3
                iter_y += 1
            iter_x += 1
        for [step_x, step_y] in [[0, 1],[0, -1],[1, 0],[-1, 0],[1, 1],[-1, 1],[1, -1],[-1, -1]]:
            self.start_and_scale[self.grid_starting_position[1] + step_y][self.grid_starting_position[0] + step_x] = 0
        meter_in_pixel = int(1 / self.map_properties['resolution'])
        iter_i = 1
        while iter_i < meter_in_pixel:
            self.start_and_scale[self.grid_starting_position[1]][self.grid_starting_position[0] + iter_i] = 0
            self.start_and_scale[self.grid_starting_position[1] - iter_i][self.grid_starting_position[0]] = 0
            iter_i += 1
        self.start_and_scale[self.grid_starting_position[1] - 1][self.grid_starting_position[0] + meter_in_pixel - 2] = 0 # draw arrow
        self.start_and_scale[self.grid_starting_position[1] + 1][self.grid_starting_position[0] + meter_in_pixel - 2] = 0
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel + 2][self.grid_starting_position[0] - 1] = 0 # draw arrow
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel + 2][self.grid_starting_position[0] + 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - 5][self.grid_starting_position[0] + meter_in_pixel] = 0 # draw "x"
        self.start_and_scale[self.grid_starting_position[1] - 5 - 1][self.grid_starting_position[0] + meter_in_pixel - 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - 5 + 1][self.grid_starting_position[0] + meter_in_pixel - 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - 5 + 1][self.grid_starting_position[0] + meter_in_pixel + 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - 5 - 1][self.grid_starting_position[0] + meter_in_pixel + 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel][self.grid_starting_position[0] + 5] = 0 # draw "y"
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel - 1][self.grid_starting_position[0] + 5 - 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel - 1][self.grid_starting_position[0] + 5 + 1] = 0
        self.start_and_scale[self.grid_starting_position[1] - meter_in_pixel + 1][self.grid_starting_position[0] + 5] = 0
        
        if self.basename == "Treitlstrasse_3-U_v3":
            print("map with custom settings: ", self.basename)
            self.erosion_value = 9
            self.d_value = 25
            self.use_blurred_factor = True
            self.sample_every = 10
        elif self.basename == "f1_aut":
            print("map with custom settings: ", self.basename)
            self.erosion_value = 12
            self.d_value = 30
            self.use_blurred_factor = False
            self.sample_every = 10
        elif self.basename == "columbia_small":
            print("map with custom settings: ", self.basename)
            self.erosion_value = 27
            self.d_value = 40
            self.use_blurred_factor = False
            self.sample_every = 30
        else:
            print("no custom settings for map (use default): ", self.basename)
            self.erosion_value = 12
            self.d_value = 25
            self.use_blurred_factor = True
            self.sample_every = 15


        if self.verbose:
            print("Verbose infos:")
            print("Starting position (world coordinates): [x,y] = [{:.2f}, {:.2f}]".format(
                self.world_starting_position[0], self.world_starting_position[1]
            ))
            print("Starting position (grid coordinates): [x,y] = [{:d}, {:d}]".format(
                self.grid_starting_position[0], self.grid_starting_position[1]
            ))
            print("Center position (grid coordinates): [x,y] = [{:.2f}, {:.2f}]".format(
                self.image_center[0], self.image_center[1]
            ))
            print("Occupied threshold: {:.4f}".format(self.map_properties['occupied_thresh']))
            print("Min. pixel value*: {:8.4f} (*) before normalization".format(np.amin(self.image)))
            print("Max. pixel value*: {:8.4f}".format(np.amax(self.image)))
            print("Dimensions: {:d} x {:d}".format(self.image.shape[0], self.image.shape[1]))

    # ===================================================================================================================
    def compute_distance_transform(self, starting_position, erosion_value = 0, forward_direction = True):

        if erosion_value == 0:
            binary_image = np.copy(self.binary_image)
            filename_extra = "full"
        else:
            binary_image = np.copy(self.binary_image)
            binary_image = morphology.binary_erosion(binary_image, selem=morphology.selem.disk(radius=erosion_value, dtype=np.bool))
            filename_extra = "eroded-by-{}".format(erosion_value)

        distances = np.zeros_like(binary_image, np.float)

        mask = np.zeros_like(binary_image, np.bool)
        mask_start = np.zeros_like(binary_image, np.bool)
        mask_target = np.zeros_like(binary_image, np.bool)
        finish_line = np.zeros_like(binary_image, np.bool)
        mask[self.grid_starting_position[1], self.grid_starting_position[0]] = True
        mask_start[self.grid_starting_position[1], self.grid_starting_position[0]+2] = True
        mask_target[self.grid_starting_position[1], self.grid_starting_position[0]-2] = True

        def block_finish_line(row_start, col, row_step, blocked):
            row = row_start
            while True:
                if binary_image[row, col]:
                    binary_image[row, col] = False
                    blocked[row, col] = True
                    row = row + row_step
                else:
                    break

        col_offset = (-1) if forward_direction else (+1)
        block_finish_line(starting_position[1],   starting_position[0]+col_offset, +1, finish_line)
        block_finish_line(starting_position[1]-1, starting_position[0]+col_offset, -1, finish_line)

        print("Computing start area")
        dist = 0.0
        while True:
            dist = dist + 1.0
            dilated_mask_start = morphology.binary_dilation(mask_start, selem=morphology.selem.square(width=3, dtype=np.bool))
            new_pixels = np.logical_and(binary_image, np.logical_xor(mask_start, dilated_mask_start))
            x,y = np.nonzero(new_pixels)
            mask_start = np.logical_or(mask_start, new_pixels)
            if dist == 50:
                mask_start_small = mask_start
            if dist >= 100:
                break
        mask_start = np.logical_xor(mask_start, finish_line)
        mask_start_small = np.logical_xor(mask_start_small, finish_line)

        print("Computing target area")
        dist = 0.0
        while True:
            dist = dist + 1.0
            dilated_mask_target = morphology.binary_dilation(mask_target, selem=morphology.selem.square(width=3, dtype=np.bool))
            new_pixels = np.logical_and(binary_image, np.logical_xor(mask_target, dilated_mask_target))
            x,y = np.nonzero(new_pixels)
            mask_target = np.logical_or(mask_target, new_pixels)
            if dist == 50:
                mask_target_small = mask_target
            if dist >= 100:
                break

        print("Computing distance to finish line from position [{}, {}] ({} direction)...".format(
            starting_position[0], starting_position[1], "forward" if forward_direction else "backward"))

        finish_line_width = np.count_nonzero(finish_line)
        current_distance = 0.0
        while True:
            current_distance = current_distance + 1.0
            dilated_mask = morphology.binary_dilation(mask, selem=morphology.selem.square(width=3, dtype=np.bool))
            new_pixels = np.logical_and(binary_image, np.logical_xor(mask, dilated_mask))
            x,y = np.nonzero(new_pixels)
            if len(x) == 0:
                break
            distances[x,y] = current_distance
            mask = np.logical_or(mask, new_pixels)
            if self.verbose and (np.mod(current_distance, 100) == 0):
                print("distance = {}".format(current_distance))

        #For debugging:
        if self.export_debug:
            io.imsave(self.output_filename_root + '.' + filename_extra + '.finish_line.png', img_as_ubyte(finish_line), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.binary_image.png', img_as_ubyte(binary_image), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.mask.png', img_as_ubyte(mask), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.mask_start.png', img_as_ubyte(mask_start), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.mask_start_small.png', img_as_ubyte(mask_start_small), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.mask_target.png', img_as_ubyte(mask_target), check_contrast=False)
            io.imsave(self.output_filename_root + '.' + filename_extra + '.mask_target_small.png', img_as_ubyte(mask_target_small), check_contrast=False)

        distances[np.nonzero(finish_line)] = (current_distance)
        distances = distances * float(self.map_properties['resolution'])
        normalized_distances = distances / np.amax(distances.flatten())
        drivable_area = np.logical_or(mask, finish_line)
        return drivable_area, distances, normalized_distances, mask_start, mask_start_small, mask_target, mask_target_small

    # ===================================================================================================================
    def compute_distance_transform_smoothed(self, starting_position, erosion_value = 0, forward_direction = True):

        drivable_area, distances, normalized_distances, mask_start, mask_start_small, mask_target, mask_target_small = \
            self.compute_distance_transform(starting_position=self.grid_starting_position, erosion_value=erosion_value, forward_direction=forward_direction)

        distances_blurred = distances
        t_distance = np.amax(distances.flatten())

        # dilate race area using maximum filter, s.t. gaussian filter does not introduce values from outside the race area
        distances_extended = distances_blurred * drivable_area + ndimage.maximum_filter(distances_blurred, size=10) * (1-drivable_area)
        # split sections around finish line, s.t. gaussian filter does not introduce values from the other side of the finish line
        distances_extended_target_blank = np.full_like(self.binary_image, t_distance, np.float) * mask_target + distances_extended * (1-mask_target)
        distances_extended_start_blank = np.full_like(self.binary_image, 0, np.float) * mask_start_small + distances_extended * (1-mask_start_small)

        distances_blurred_target_blank = ndimage.gaussian_filter(distances_extended_target_blank, sigma=3) * drivable_area
        distances_blurred_start_blank = ndimage.gaussian_filter(distances_extended_start_blank, sigma=3) * drivable_area
        distances_blurred = distances_blurred_target_blank * mask_start + distances_blurred_start_blank * (1-mask_start)

        # dilate race area using maximum filter, s.t. gaussian filter does not introduce values from outside the race area
        distances_extended = distances_blurred * drivable_area + ndimage.maximum_filter(distances_blurred, size=10) * (1-drivable_area)
        # split sections arount finish line, s.t. gaussian filter does not introduce values from the other side of the finish line
        distances_extended_target_blank = np.full_like(self.binary_image, t_distance, np.float) * mask_target + distances_extended * (1-mask_target)
        distances_extended_start_blank = np.full_like(self.binary_image, 0, np.float) * mask_start_small + distances_extended * (1-mask_start_small)

        distances_blurred_target_blank = ndimage.gaussian_filter(distances_extended_target_blank, sigma=3) * drivable_area
        distances_blurred_start_blank = ndimage.gaussian_filter(distances_extended_start_blank, sigma=3) * drivable_area
        distances_blurred = distances_blurred_target_blank * mask_start + distances_blurred_start_blank * (1-mask_start)


        distances_extended = distances * drivable_area + np.full_like(self.binary_image, t_distance, np.float) * (1-drivable_area)
        # split sections arount finish line, s.t. gaussian filter does not introduce values from the other side of the finish line
        distances_extended_target_blank = np.full_like(self.binary_image, t_distance, np.float) * mask_target + distances_extended * (1-mask_target)
        distances_extended_start_blank = np.full_like(self.binary_image, 0, np.float) * mask_start_small + distances_extended * (1-mask_start_small)

        distances_blurred_border_target_blank = ndimage.gaussian_filter(distances_extended_target_blank, sigma=5) * drivable_area
        distances_blurred_border_start_blank = ndimage.gaussian_filter(distances_extended_start_blank, sigma=5) * drivable_area
        distances_blurred_border = distances_blurred_border_target_blank * mask_start + distances_blurred_border_start_blank * (1-mask_start)

        #if erosion_value == 0:
        #    distances = distances + distances_blurred * 0.04 + distances_blurred_border * 0.02
        #else:
        #    distances = distances + distances_blurred * 0.08 + distances_blurred_border * 0.06

        if self.use_blurred_factor:
            distances = distances + distances_blurred * 0.08 + distances_blurred_border * 0.06

        distances = distances * float(self.map_properties['resolution'])
        normalized_distances = distances / np.amax(distances.flatten())

        return drivable_area, distances, normalized_distances, mask_start, mask_start_small, mask_target, mask_target_small

    # ===================================================================================================================

    def compute_raceline(self, starting_position, current_drivable_area, full_drivable_area, normalized_target_distances, filename_part):
        race_line = np.zeros_like(self.binary_image, np.float)
        spline_line = np.zeros_like(self.binary_image, np.float)

        race_pos = [starting_position[1], starting_position[0] + 10]; # start ten pixels right of the finish-line

        race_line[race_pos[0], race_pos[1]] = 1;
        cv = [race_pos];

        a_distance = 0.0
        while True:
            a_distance = a_distance + 1.0

            best_val = 100000
            best_pos = race_pos
            better = False

            for [step_x, step_y] in [[0, 1],[0, -1],[1, 0],[-1, 0],[1, 1],[-1, 1],[1, -1],[-1, -1]]:
                candidate_pos = [race_pos[0] + step_x, race_pos[1] + step_y]
                candidate_val = normalized_target_distances[candidate_pos[0], candidate_pos[1]]
                if candidate_val < best_val and current_drivable_area[candidate_pos[0], candidate_pos[1]]:
                    best_val = candidate_val
                    best_pos = candidate_pos
                    better = True
#            if np.mod(a_distance,15) == 0:
            if np.mod(a_distance,self.sample_every) == 0:
                cv = np.append(cv, [race_pos], axis=0)

            #For debugging:
            #if self.export_debug:
            #    print("  best_pos {} {} val {} from current val {}".format(best_pos[0], best_pos[1], best_val, normalized_target_distances[race_pos[0], race_pos[1]]))

            race_pos = best_pos
            if race_line[race_pos[0], race_pos[1]] == 1: # cannot advance, car is stuck
                break

            race_line[race_pos[0], race_pos[1]] = 1;

            if np.mod(a_distance,50) == 0:
                print("a_distance: {}".format(a_distance))
            if a_distance >= 5000:
                break

        #For debugging:
        if self.export_debug:
            race_line = morphology.binary_dilation(race_line, selem=morphology.selem.square(width=2, dtype=np.bool)) # dilate for thicker line
            io.imsave(self.output_filename_root + '.race_line_' + filename_part + '.png', img_as_ubyte(race_line * 255) - self.binary_image.astype(float) * 150)

        #d = 35
        d = self.d_value
        p = scipy_bspline(cv,n=1000,degree=d,periodic=True)
        
        #cv_save = (cv - 1000) * self.map_properties['resolution']
        #cv_save[:,[0, 1]] = cv_save[:,[1, 0]]
        #cv_save = cv_save - 1000 #np.array(self.map_properties['origin'])[0:2]
        #p_save = (p - 1000) * self.map_properties['resolution']
        #p_save[:,[0, 1]] = p_save[:,[1, 0]]
        #p_save[:,1] = 0 - p_save[:,1] # invert y axis
        #np.array(self.map_properties['origin'])[0:2]
        
        #np.savetxt(self.output_filename_root + '.spline_line_cv.csv', cv_save, delimiter=",")
        #np.savetxt(self.output_filename_root + '.spline_line_p.csv', p_save, delimiter=",")

        for [spline_x, spline_y] in p.astype(int):
            spline_line[spline_x, spline_y] = 1;

        #For debugging:
        if self.export_debug:
            spline_line = morphology.binary_dilation(spline_line, selem=morphology.selem.square(width=2, dtype=np.bool)) # dilate for thicker line
            io.imsave(self.output_filename_root + '.spline_line.png', img_as_ubyte(spline_line * 255) - self.binary_image.astype(float) * 150)

        spline_line_blurred = ndimage.gaussian_filter(spline_line.astype(float) * 150, sigma=3) * full_drivable_area;
        for it in range(1,10):
            spline_line_blurred = ndimage.gaussian_filter(spline_line_blurred, sigma=3) * full_drivable_area;

        normalized_spline_line_blurred = spline_line_blurred / np.amax(spline_line_blurred.flatten())
        #For debugging:
        if self.export_debug:
            io.imsave(self.output_filename_root + '.spline_line_' + filename_part + '.colorized.png', cmapy.colorize(img_as_ubyte(normalized_spline_line_blurred), 'plasma', rgb_order=True) - (100*np.repeat(self.binary_image[:, :, np.newaxis], 3, axis=2).astype(float)))

        return normalized_spline_line_blurred


    # ===================================================================================================================

    def run(self):

        drivable_area_eroded, distance_to_target_eroded_smoothed, normalized_distance_to_target_eroded_smoothed, _, _, _, _ = \
            self.compute_distance_transform_smoothed(starting_position=self.grid_starting_position, \
            erosion_value = self.erosion_value, forward_direction = False)

        drivable_area, distance_to_target_smoothed, normalized_distance_to_target_smoothed, _, _, _, _ = \
            self.compute_distance_transform_smoothed(starting_position=self.grid_starting_position, forward_direction = False)

        drivable_area, distance_from_start_line, normalized_distance_from_start_line, _, _, _, _ = \
            self.compute_distance_transform(starting_position=self.grid_starting_position, forward_direction = True)

        self.compute_raceline(self.grid_starting_position, drivable_area, drivable_area, distance_to_target_smoothed, "full")
        normalized_spline_line = self.compute_raceline(self.grid_starting_position, drivable_area_eroded, drivable_area, distance_to_target_eroded_smoothed, "eroded")

        distances_from_nearest_obstacle = ndimage.distance_transform_edt(drivable_area, return_distances=True, return_indices=False)
        distances_from_nearest_obstacle = distances_from_nearest_obstacle.astype(float) * float(self.map_properties['resolution'])
        normalized_distances_from_nearest_obstacle = distances_from_nearest_obstacle / np.amax(distances_from_nearest_obstacle.flatten())

        if self.verbose:
            print("Max. distance from start line: {:8.4f}".format(
                np.amax(distance_from_start_line)))
        #    print("Max. distance to nearest obstacle: {:8.4f}".format(
        #        np.amax(distances_from_nearest_obstacle)))

        def mask_image(image, mask):
            new_image = np.where(mask[..., None], image, 0)
            return (new_image)

        if self.export_images:
            io.imsave(self.output_filename_root + '.start_and_scale.png', img_as_ubyte(self.start_and_scale), check_contrast=False)
            io.imsave(self.output_filename_root + '.drivable_area.png', img_as_ubyte(drivable_area), check_contrast=False)
            io.imsave(self.output_filename_root + '.distance_from_start_line.colorized.png', mask_image(
                          image=cmapy.colorize(img_as_ubyte(normalized_distance_from_start_line), 'plasma', rgb_order=True),
                          mask=drivable_area), check_contrast=False)
            io.imsave(self.output_filename_root + '.distance_to_target_smoothed.colorized.png', mask_image(
                          image=cmapy.colorize(img_as_ubyte(normalized_distance_to_target_smoothed), 'plasma', rgb_order=True),
                          mask=drivable_area), check_contrast=False)
            io.imsave(self.output_filename_root + '.distances_from_nearest_obstacle.colorized.png', mask_image(
                          image=cmapy.colorize(img_as_ubyte(normalized_distances_from_nearest_obstacle), 'plasma', rgb_order=True),
                          mask=drivable_area), check_contrast=False)
            io.imsave(self.output_filename_root + '.spline_line.colorized.png', mask_image(
                          image=cmapy.colorize(img_as_ubyte(normalized_spline_line), 'plasma', rgb_order=True),
                          mask=drivable_area), check_contrast=False)

        np.savez(self.output_path, properties=[    self.world_starting_position[0],
                                        self.world_starting_position[1],
                                        self.grid_starting_position[0], 
                                        self.grid_starting_position[1],
                                        self.image_center[0], 
                                        self.image_center[1],
                                        self.map_properties['occupied_thresh'],
                                        np.amin(self.image),
                                        np.amax(self.image),
                                        self.image.shape[0], 
                                        self.image.shape[1],
                                        self.map_properties['resolution']],
                                 drivable_area=drivable_area, 
                                 norm_distance_from_start=normalized_distance_from_start_line, 
                                 norm_distance_to=normalized_distance_to_target_smoothed,
                                 norm_distance_to_obstacle=normalized_distances_from_nearest_obstacle)

# ======================================================================================================================

# https://stackoverrun.com/de/q/9593488
def scipy_bspline(cv, n=100, degree=3, periodic=False):
       """ Calculate n samples on a bspline

           cv :      Array ov control vertices
           n  :      Number of samples to return
           degree:   Curve degree
           periodic: True - Curve is closed
       """
       cv = np.asarray(cv)
       count = cv.shape[0]

       # Closed curve
       if periodic:
           kv = np.arange(-degree,count+degree+1)
           factor, fraction = divmod(count+degree+1, count)
           cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
           degree = np.clip(degree,1,degree)

       # Opened curve
       else:
           degree = np.clip(degree,1,count-1)
           kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

       # Return samples
       max_param = count - (degree * (1-periodic))
       spl = si.BSpline(kv, cv, degree)
       return spl(np.linspace(0,max_param,n))

# ======================================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', help='starting position (x) in world coordinates', default='0.0')
    parser.add_argument('--start_y', help='starting position (y) in world coordinates', default='0.0')
    parser.add_argument('--input', help='filename of the map description (.yaml)')
    parser.add_argument('--output', help='filename of the output dataset (.npz)')
    parser.add_argument('--export_images', help='export each layer as bitmap', action='store_true')
    parser.add_argument('--export_debug', help='export more layers as bitmap (for debugging purposes)', action='store_true')
    args = parser.parse_args()
    app = CostmapGenerator(starting_position=(float(args.start_x),float(args.start_y)),
                           input_yaml_path=args.input,
                           output_path=args.output,
                           export_images=args.export_images,
                           export_debug=args.export_debug)
    app.run()

