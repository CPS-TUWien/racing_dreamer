#!/bin/python3

##################
import bpy
import glob
import os

def curve_to_mesh(context, curve):
    deg = context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(curve.evaluated_get(deg), depsgraph=deg)

    new_obj = bpy.data.objects.new(curve.name + "_mesh", me)
    context.collection.objects.link(new_obj)

    for o in context.selected_objects:
        o.select_set(False)

    new_obj.matrix_world = curve.matrix_world
    new_obj.select_set(True)
    context.view_layer.objects.active = new_obj

importDir = "../maps_svg" 
print("Importing all SVG from this directory", importDir)

os.chdir(importDir)

# #######
# clean scene

#for obj in bpy.context.scene.objects:
#    if obj.type == 'MESH':
#        obj.select = True
    #else:
    #    obj.select = False
#bpy.ops.object.delete()

for block in bpy.data.curves:
    bpy.data.curves.remove(block)

for block in bpy.data.meshes:
    bpy.data.meshes.remove(block)

for block in bpy.data.materials:
    bpy.data.materials.remove(block)

for block in bpy.data.textures:
    bpy.data.textures.remove(block)

for block in bpy.data.images:
    bpy.data.images.remove(block)

for block in bpy.data.collections:
    bpy.data.collections.remove(block)

# #######
# import SVG

bpy.ops.import_curve.svg(filepath="Treitlstrasse_3-U_v3_wall-polygon.svg")

context = bpy.context
#obj = context.object

# convert curves to meshes
for obj in bpy.context.scene.objects:
    if obj and obj.type == 'CURVE':
        obj.dimensions = (100, 100, 0)
        obj.location = (-50, -50, 0)
        obj.data.extrude=0.5
        #curve_to_mesh(context, obj)
        
# remove all curves
#for block in bpy.data.curves:
#    bpy.data.curves.remove(block)

#directory = os.path.dirname(importDir)
#target_file = os.path.join(directory, "f1_aut_wall-polygon.obj")

target_file = "/home/andreas/ARC/racecar_gym/models/scenes/treitlstrasse_v3/meshes/Walls.obj"
bpy.ops.export_scene.obj(filepath=target_file, \
                        check_existing=True, \
                        axis_forward='X', \
                        axis_up='Z', \
                        filter_glob="*.obj;*.mtl", \
                        use_selection=False, \
                        use_animation=False, \
                        use_mesh_modifiers=True, \
                        use_edges=True, \
                        use_smooth_groups=False, \
                        use_smooth_groups_bitflags=False, \
                        use_normals=True, \
                        use_uvs=True, \
                        use_materials=True, \
                        use_triangles=False, \
                        use_nurbs=False, \
                        use_vertex_groups=False, \
                        use_blen_objects=True, \
                        group_by_object=False, \
                        group_by_material=False, \
                        keep_vertex_order=False, \
                        global_scale=1, \
                        path_mode='AUTO')

print("\ndone.\n\n\n")
