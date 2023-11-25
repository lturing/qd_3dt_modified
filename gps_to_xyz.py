"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple

import numpy as np
from PIL import Image
import math 
import glob 
import os
from matplotlib import pyplot as plt 
import numpy as np



# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system 
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = {}

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                name = os.path.basename(filename).split('.')[0]
                assert name not in oxts 
                oxts[name] = OxtsData(packet, T_w_imu)
                #oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def load_image(file, mode):
    """Load an image from file."""
    return Image.open(file).convert(mode)


def yield_images(imfiles, mode):
    """Generator to read image files."""
    for file in imfiles:
        yield load_image(file, mode)


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)



def load_calib_rigid(filepath):
    """Read a rigid transform calibration file as a numpy.array."""
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data['R'], data['T'])


def load_calib_cam_to_cam(velo_to_cam_filepath, cam_to_cam_filepath):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates
    T_cam0unrect_velo = load_calib_rigid(velo_to_cam_filepath)
    data['T_cam0_velo_unrect'] = T_cam0unrect_velo

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_filepath)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
    P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Create 4x4 matrices from the rectifying rotation matrices
    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
    R_rect_10 = np.eye(4)
    R_rect_10[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))
    R_rect_20 = np.eye(4)
    R_rect_20[0:3, 0:3] = np.reshape(filedata['R_rect_02'], (3, 3))
    R_rect_30 = np.eye(4)
    R_rect_30[0:3, 0:3] = np.reshape(filedata['R_rect_03'], (3, 3))

    data['R_rect_00'] = R_rect_00
    data['R_rect_10'] = R_rect_10
    data['R_rect_20'] = R_rect_20
    data['R_rect_30'] = R_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T0 = np.eye(4)
    T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data


def pose_inverse(pose):
    R = pose[:3, :3]
    t = pose[:3, 3:4]
    R_new = R.transpose()
    t_new = -np.matmul(R_new, t)

    return np.vstack((np.hstack([R_new, t_new]), [0, 0, 0, 1]))

def load_calib(calib_imu_to_velo_filepath, velo_to_cam_filepath, cam_to_cam_filepath):
    """Load and compute intrinsic and extrinsic calibration parameters."""
    # We'll build the calibration parameters as a dictionary, then
    # convert it to a namedtuple to prevent it from being modified later
    data = {}

    # Load the rigid transformation from IMU to velodyne
    #calib_imu_to_velo_filepath = 'calib_imu_to_velo.txt'
    #velo_to_cam_filepath = 'calib_velo_to_cam.txt'
    #cam_to_cam_filepath = 'calib_cam_to_cam.txt'

    data['T_velo_imu'] = load_calib_rigid(calib_imu_to_velo_filepath)

    # Load the camera intrinsics and extrinsics
    data.update(load_calib_cam_to_cam(velo_to_cam_filepath, cam_to_cam_filepath))

    # Pre-compute the IMU to rectified camera coordinate transforms
    data['T_cam0_imu'] = data['T_cam0_velo'].dot(data['T_velo_imu'])
    data['T_cam1_imu'] = data['T_cam1_velo'].dot(data['T_velo_imu'])
    data['T_cam2_imu'] = data['T_cam2_velo'].dot(data['T_velo_imu'])
    data['T_cam3_imu'] = data['T_cam3_velo'].dot(data['T_velo_imu'])

    # np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    data['T_imu_cam0'] = pose_inverse(data['T_cam0_imu'])
    data['T_imu_cam1'] = pose_inverse(data['T_cam1_imu'])
    data['T_imu_cam2'] = pose_inverse(data['T_cam2_imu'])
    data['T_imu_cam3'] = pose_inverse(data['T_cam3_imu'])

    return data 


if __name__ == '__main__':
    ddir = '/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/oxts/data'
    files = glob.glob(os.path.join(ddir, '*.txt'))
    files.sort()

    oxts = load_oxts_packets_and_poses(files)
    names = list(oxts.keys())
    names.sort()

    x = [oxts[it].T_w_imu[0][3] for it in names]
    y = [oxts[it].T_w_imu[1][3] for it in names]
    z = [oxts[it].T_w_imu[2][3] for it in names]

    print(f"x: {x[:10]}, min(x)={min(x)}, max(x)={max(x)}")
    print(f"y: {y[:10]}, min(y)={min(y)}, max(y)={max(y)}")
    print(f"z: {z[:10]}, min(z)={min(z)}, max(z)={max(z)}")
    
    print(names[0], oxts[names[0]].T_w_imu)

    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z)
    ax.set_aspect('equal')
    plt.show()

    calib_imu_to_velo_filepath =  '/home/spurs/dataset/kitti_raw/2011_10_03/calib_imu_to_velo.txt'
    velo_to_cam_filepath = '/home/spurs/dataset/kitti_raw/2011_10_03/calib_velo_to_cam.txt'
    cam_to_cam_filepath = '/home/spurs/dataset/kitti_raw/2011_10_03/calib_cam_to_cam.txt'

    calib = load_calib(calib_imu_to_velo_filepath, velo_to_cam_filepath, cam_to_cam_filepath)
    poses = [np.matmul(oxts[it].T_w_imu, calib['T_imu_cam2']) for it in names]

    x = [it[0][3] for it in poses]
    y = [it[1][3] for it in poses]
    z = [it[2][3] for it in poses]
    print(f"T_imu_cam2={calib['T_imu_cam2']}")
    
    print(f"x: {x[:10]}, min(x)={min(x)}, max(x)={max(x)}")
    print(f"y: {y[:10]}, min(y)={min(y)}, max(y)={max(y)}")
    print(f"z: {z[:10]}, min(z)={min(z)}, max(z)={max(z)}")

    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z)
    ax.set_aspect('equal')
    plt.show()




