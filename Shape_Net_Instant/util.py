import random
import bpy
from mathutils import Matrix, Vector
import os
import numpy as np
import math
from functools import reduce

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def sample_spherical(n, radius=1.):
    xyz = np.random.normal(size=(n,3))
    xyz = normalize(xyz) * radius
    return xyz


def set_camera_focal_length_in_world_units(camera_data, focal_length):
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera_data.sensor_width
    sensor_height_in_mm = camera_data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camera_data.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    camera_data.lens = focal_length / s_u


# Blender: camera looks in negative z-direction, y points up, x points right.
# Opencv: camera looks in positive z-direction, y points down, x points right.
def cv_cam2world_to_bcam2world(cv_cam2world):
    '''

    :cv_cam2world: numpy array.
    :return:
    '''
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    cam_location = Vector(cv_cam2world[:3, -1].tolist())
    cv_cam2world_rot = Matrix(cv_cam2world[:3, :3].tolist())

    cv_world2cam_rot = cv_cam2world_rot.transposed()
    cv_translation = -1. * cv_world2cam_rot * cam_location

    blender_world2cam_rot = R_bcam2cv * cv_world2cam_rot
    blender_translation = R_bcam2cv * cv_translation

    blender_cam2world_rot = blender_world2cam_rot.transposed()
    blender_cam_location = -1. * blender_cam2world_rot * blender_translation

    blender_matrix_world = Matrix((
        blender_cam2world_rot[0][:] + (blender_cam_location[0],),
        blender_cam2world_rot[1][:] + (blender_cam_location[1],),
        blender_cam2world_rot[2][:] + (blender_cam_location[2],),
        (0, 0, 0, 1)
    ))

    return blender_matrix_world


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_world2cam_from_blender_cam(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2] # Matrix_world returns the cam2world matrix.
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv * R_world2bcam
    T_world2cv = R_bcam2cv * T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0,0,0,1)
    ))
    return RT


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
         (    0  , alpha_v, v_0),
         (    0  , 0,        1 )))
    return K


def cond_mkdir(path):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))

def get_archimedean_spiral(sphere_radius, num_steps=250):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        z = r * math.sin(-theta + math.pi) * math.sin(-i)
        y = r * - math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)

def get_orthogonal_camera_positions(sphere_radius, center=(0, 0, 0)):
    """
    Returns 4 camera positions at 90-degree intervals around the Y axis,
    all at the same elevation (XZ plane), looking at the center.
    """
    positions = []
    for i in range(4):
        angle = i * (math.pi / 2)  # 0, 90, 180, 270 degrees in radians
        x = sphere_radius * math.cos(angle) + center[0]
        y = center[1]
        z = sphere_radius * math.sin(angle) + center[2]
        positions.append((x, y, z))
    return np.array(positions)

def get_uniform_sphere_cameras(num_views, radius=1.5, elevation_range=(0, 30)):
    """
    Generate camera positions uniformly distributed on a sphere
    with controlled elevation for better InstantMesh training.
    """
    positions = []
    for i in range(num_views):
        # Uniform azimuth
        azimuth = 2 * math.pi * i / num_views
        # Random elevation within range
        elevation = math.radians(random.uniform(*elevation_range))
        
        x = radius * math.cos(elevation) * math.cos(azimuth)
        y = radius * math.sin(elevation)
        z = radius * math.cos(elevation) * math.sin(azimuth)
        
        positions.append((x, y, z))
    
    return np.array(positions)

def get_camera_matrices_for_instantmesh(positions, target=(0, 0, 0)):
    """
    Convert camera positions to transformation matrices for InstantMesh
    """
    matrices = []
    for pos in positions:
        cam_matrix = look_at(np.array(pos), np.array(target))
        blender_matrix = cv_cam2world_to_bcam2world(cam_matrix)
        matrices.append(blender_matrix)
    
    return matrices

def validate_camera_for_instantmesh(camera_data, expected_fov=50):
    """
    Validate camera settings match InstantMesh expectations
    """
    actual_fov = math.degrees(camera_data.angle)
    if abs(actual_fov - expected_fov) > 1:
        print("Warning: FOV is {0}°, InstantMesh expects ~{1}°".format(actual_fov, expected_fov))

    
    return True

def get_instantmesh_camera_pattern(num_views=6, radius=1.5):
    """
    Generate camera positions optimized for InstantMesh training
    """
    if num_views == 6:
        # Use orthogonal + additional views for better coverage
        base_positions = get_orthogonal_camera_positions(radius)
        # Add two more views at different elevations
        extra_positions = [
            (0, radius * 0.5, radius * 0.866),  # 30° elevation
            (0, -radius * 0.5, radius * 0.866)  # -30° elevation
        ]
        return np.vstack([base_positions, extra_positions])
    else:
        return get_uniform_sphere_cameras(num_views, radius)
