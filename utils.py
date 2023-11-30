import numpy as np
from collections import defaultdict
import torch 
import math 
import cv2

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def track2results(bboxes, labels, ids):
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    ids = ids.cpu().numpy()
    outputs = defaultdict(list)
    for bbox, label, id in zip(bboxes, labels, ids):
        outputs[id] = dict(bbox=bbox, label=label)
    return outputs

def imagetocamera_torch(points, depths, projection):
    """
    points: (N, 2), N points on X-Y image plane
    depths: (N,), N depth values for points
    projection: (3, 4), projection matrix

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)
    corners = torch.cat([points, points.new_ones((points.shape[0], 1))],
                        dim=1).mm(projection[:, 0:3].inverse().t())
    assert torch.all(abs(corners[:, 2] - 1) < 0.01)
    corners_cam = corners * depths.view(-1, 1)

    return corners_cam

def cameratoworld_torch(corners, position, rotation):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert corners.shape[1] == 3, ("Shape ({}) not fit".format(corners.shape))
    corners_global = corners.mm(rotation.t()) + position[None]
    return corners_global

def alpha2yaw_torch(alpha, x_loc, z_loc):
    """
    Get rotation_y by alpha + theta
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    torch_pi = alpha.new_tensor([np.pi])
    rot_y = alpha + torch.atan2(x_loc, z_loc)
    rot_y = (rot_y + torch_pi) % (2 * torch_pi) - torch_pi
    return rot_y


def worldtocamera_torch(corners_global, position, rotation):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, ("Shape ({}) not fit".format(
        corners_global.shape))
    corners = (corners_global - position[None]).mm(rotation)
    return corners


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]


def yaw2alpha_torch(rot_y, x_loc, z_loc):
    """
    Get alpha by rotation_y - theta
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(x_loc, z_loc)
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha

def cameratoimage_torch(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    #points = torch.cat([corners, corners.new_ones((corners.shape[0], 1))], dim=1).mm(projection.t())

    points = corners.mm(projection.t())

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points_img = (points[:, :2] / points[:, 2:3]
                  ) * mask + invalid_value * torch.logical_not(mask)

    return points_img

def quaternion_to_euler(w, x, y, z):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def worldtocamera(corners_global, pose):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, ("Shape ({}) not fit".format(
        corners_global.shape))
    corners = (corners_global - pose.position[np.newaxis]).dot(pose.rotation)
    return corners


def cameratoimage(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    #points = np.hstack([corners, np.ones((corners.shape[0], 1))]).dot(projection.T)
    points = corners.dot(projection.T)

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points = (points[:, :2] / points[:, 2:3]) * mask + invalid_value * (1 -
                                                                        mask)
    return points


def is_before_clip_plane_world(points_world, cam_pose, cam_near_clip=0.15):
    """
    points_world: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates
    cam_near_clip: scalar, the near projection plane

    is_before: (N,), bool, is the point locate before the near clip plane
    """
    return worldtocamera(points_world, cam_pose)[:, 2] > cam_near_clip


def is_before_clip_plane_camera(points_camera, cam_near_clip=0.15):
    """
    points_camera: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    cam_near_clip: scalar, the near projection plane

    is_before: bool, is the point locate before the near clip plane
    """
    return points_camera[:, 2] > cam_near_clip


def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    # get the intersection point of two 3D points and a plane
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2
    return inter_point


def get_3d_bbox_vertex(cam_calib, cam_pose, points3d, cam_near_clip=0.15):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        cam_calib: (3, 4), projection matrix
        cam_pose: a class with position, rotation of the frame
            rotation:  (3, 3), rotation along camera coordinates
            position:  (3), translation of world coordinates
        points3d: (8, 3), box 3D center in camera coordinates
        cam_near_clip: in meter, distance to the near plane
    Output:
        points: numpy array of shape (8, 2) for bbox in image coordinates
    '''
    lineorder = np.array(
        [
            [1, 2, 6, 5],  # front face
            [2, 3, 7, 6],  # left face
            [3, 4, 8, 7],  # back face
            [4, 1, 5, 8],
            [1, 6, 5, 2]
        ],
        dtype=np.int32) - 1  # right

    points = []

    # In camera coordinates
    cam_dir = np.array([0, 0, 1])
    center_pt = cam_dir * cam_near_clip

    for i in range(len(lineorder)):
        for j in range(4):
            p1 = points3d[lineorder[i, j]].copy()
            p2 = points3d[lineorder[i, (j + 1) % 4]].copy()

            before1 = is_before_clip_plane_camera(p1[np.newaxis],
                                                  cam_near_clip)[0]
            before2 = is_before_clip_plane_camera(p2[np.newaxis],
                                                  cam_near_clip)[0]

            inter = get_intersect_point(center_pt, cam_dir, p1, p2)

            if not (before1 or before2):
                # print("Not before 1 or 2")
                continue
            elif before1 and before2:
                # print("Both 1 and 2")
                vp1 = p1
                vp2 = p2
            elif before1 and not before2:
                # print("before 1 not 2")
                vp1 = p1
                vp2 = inter
            elif before2 and not before1:
                # print("before 2 not 1")
                vp1 = inter
                vp2 = p2

            cp1 = cameratoimage(vp1[np.newaxis], cam_calib)[0]
            cp2 = cameratoimage(vp2[np.newaxis], cam_calib)[0]
            points.append((cp1, cp2))
    return points



def construct2dlayout(trks, dims, rots, cam_calib, pose, cam_near_clip=0.15):
    depths = []
    boxs = []
    points = []
    corners_camera = worldtocamera(trks, pose)
    for corners, dim, rot in zip(corners_camera, dims, rots):
        # in camera coordinates
        points3d = computeboxes(rot, dim, corners)
        depths.append(corners[2])
        projpoints = get_3d_bbox_vertex(cam_calib, pose, points3d, cam_near_clip)
        points.append(projpoints)
        if projpoints == []:
            box = np.array([-1000, -1000, -1000, -1000])
            boxs.append(box)
            depths[-1] = -10
            continue
        projpoints = np.vstack(projpoints)[:, :2]
        projpoints = projpoints.reshape(-1, 2)
        minx = projpoints[:, 0].min()
        maxx = projpoints[:, 0].max()
        miny = projpoints[:, 1].min()
        maxy = projpoints[:, 1].max()
        box = np.array([minx, miny, maxx, maxy])
        boxs.append(box)
    return boxs, depths, points


def computeboxes(roty, dim, loc):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        roty: (1,), object orientation, -pi ~ pi
        box_dim: a tuple of (h, w, l)
        loc: (3,), box 3D center
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    roty = roty[0]
    R = np.array([[+np.cos(roty), 0, +np.sin(roty)], [0, 1, 0],
                  [-np.sin(roty), 0, +np.cos(roty)]])
    corners = get_vertex(dim)
    corners = corners.dot(R.T) + loc
    return corners


def cameratoworld(corners, pose):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert corners.shape[1] == 3, ("Shape ({}) not fit".format(corners.shape))
    corners_global = corners.dot(pose.rotation.T) + \
                     pose.position[np.newaxis]
    return corners_global



def get_vertex(box_dim):
    '''Get 3D bbox vertex (used for the upper volume iou calculation)
    Input:
        box_dim: a tuple of (h, w, l)
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    h, w, l = box_dim
    corners = np.array(
        [[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
         [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]])
    return corners.T


def draw_corner_info(frame, x1, y1, info_str, line_color):
    FONT_SCALE = 1.0
    FONT_THICKNESS = 1
    (test_width, text_height), baseline = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - text_height),
                    (x1 + test_width, y1 + baseline), line_color, cv2.FILLED)
    cv2.putText(frame, info_str, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE * 0.5, (0, 0, 0), FONT_THICKNESS,
                cv2.LINE_AA)
    return frame

def draw_3d_bbox(frame,
                points_camera,
                cam_calib,
                cam_pose,
                cam_near_clip: float = 0.15,
                line_color: tuple = (0, 255, 0),
                line_width: int = 2,
                corner_info: str = None):
    projpoints = get_3d_bbox_vertex(cam_calib, cam_pose, points_camera,
                                        cam_near_clip)

    for p1, p2 in projpoints:
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                    line_color, line_width)

    if corner_info is not None:
        is_before = False
        cp1 = cameratoimage(points_camera[0:1], cam_calib)[0]

        if cp1 is not None:
            is_before = tu.is_before_clip_plane_camera(
                points_camera[0:1], cam_near_clip)[0]

        if is_before:
            x1 = int(cp1[0])
            y1 = int(cp1[1])

            frame = draw_corner_info(frame, x1, y1, corner_info, line_color)

    return frame


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def generate_color(tid):
    color = Colors()
    rgb = color(tid)
    rgb = [it / 255.0 for it in rgb]
    return rgb 



def generate_color_v1(tid):
    h = (tid + 33) * 6364136223846793005 + 1442695040888963407
    rgb = [(h & 0xFF)  / 255.0, ((h >> 4) & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0]
    return rgb 