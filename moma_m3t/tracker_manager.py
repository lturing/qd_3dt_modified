import torch
import torch.nn as nn
import numpy as np
from addict import Dict
from pyquaternion import Quaternion
import utils.tracking_utils as tu
from .motion_tracker import MotionTracker
import os
from collections import defaultdict

#################utils####################
def bbox2result(bboxes, labels, num_classes):
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
##########################################

class MotionTrackerManager:

    def __init__(self):
        super(MotionTrackerManager, self).__init__()

        self.tracker = None
        self.writed = []

    def simple_test(self,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    idx=None,
                    det_out=None
                    ):
        # init tracker
        frame_ind = img_meta.get('frame_id', -1)
        is_kitti = 'KITTI' in img_meta['img_info']['file_name']
        use_3d_center = True 

        if self.tracker is None:
            self.tracker = MotionTracker(init_track_id = 0, is_kitti = is_kitti)
        elif img_meta.get('first_frame', False):
            num_tracklets = self.tracker.num_tracklets
            del self.tracker
            self.tracker = MotionTracker(init_track_id = num_tracklets, is_kitti = is_kitti)

        det_depths = det_out['det_depths']
        det_bboxes = det_out['det_bboxes']
        det_labels = det_out['det_labels']
        det_dims = det_out['det_dims']
        det_alphas = det_out['det_alphas']
        det_2dcs = det_out['det_2dcs']

        cat_num = 12 if not is_kitti else 3 #nuscenes: 12 category, kitti, 3 category

        bbox_results = bbox2result(det_bboxes, det_labels, cat_num)
        
        embeds = det_bboxes.new_zeros([det_bboxes.shape[0], 256])

        projection = det_bboxes.new_tensor(img_meta['calib'])
        position = det_bboxes.new_tensor(img_meta['pose']['position'])
        r_camera_to_world = tu.angle2rot(
            np.array(img_meta['pose']['rotation']))
        rotation = det_bboxes.new_tensor(r_camera_to_world)
        cam_rot_quat = Quaternion(matrix=r_camera_to_world)
        quat_det_yaws_world = {'roll_pitch': [], 'yaw_world': []}

        if det_depths is not None and det_2dcs is not None:
            corners = tu.imagetocamera_torch(det_2dcs, det_depths, projection)
            corners_global = tu.cameratoworld_torch(corners, position,
                                                    rotation)
            det_yaws = tu.alpha2yaw_torch(det_alphas, corners[:, 0:1],
                                          corners[:, 2:3])

            for det_yaw in det_yaws:
                yaw_quat = Quaternion(
                    axis=[0, 1, 0], radians=det_yaw.cpu().numpy())
                rotation_world = cam_rot_quat * yaw_quat
                if rotation_world.z < 0:
                    rotation_world *= -1
                roll_world, pitch_world, yaw_world = tu.quaternion_to_euler(
                    rotation_world.w, rotation_world.x, rotation_world.y,
                    rotation_world.z)
                quat_det_yaws_world['roll_pitch'].append(
                    [roll_world, pitch_world])
                quat_det_yaws_world['yaw_world'].append(yaw_world)

            det_yaws_world = rotation.new_tensor(
                np.array(quat_det_yaws_world['yaw_world'])[:, None])
            det_boxes_3d = torch.cat(
                [corners_global, det_yaws_world, det_dims], dim=1)
        else:
            det_boxes_3d = det_bboxes.new_zeros([det_bboxes.shape[0], 7])

        match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
            self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                boxes_3d=det_boxes_3d,
                position=position,
                rotation=rotation,
                cur_frame=frame_ind,)

        if det_depths is not None and det_2dcs is not None:
            match_dims = match_boxes_3ds[:, -3:]
            match_corners_cam = tu.worldtocamera_torch(match_boxes_3ds[:, :3],
                                                       position, rotation)
            
            match_depths = match_corners_cam[:, 2:3]

            match_yaws = []
            for match_order, match_yaw in zip(
                    inds[valids].cpu().numpy(),
                    match_boxes_3ds[:, 3].cpu().numpy()):
                roll_world, pitch_world = quat_det_yaws_world['roll_pitch'][
                    match_order]
                rotation_cam = cam_rot_quat.inverse * Quaternion(
                    tu.euler_to_quaternion(roll_world, pitch_world, match_yaw))
                vtrans = np.dot(rotation_cam.rotation_matrix,
                                np.array([1, 0, 0]))
                match_yaws.append(-np.arctan2(vtrans[2], vtrans[0]))

            match_yaws = rotation.new_tensor(np.array(match_yaws)).unsqueeze(1)
            match_alphas = tu.yaw2alpha_torch(match_yaws,
                                              match_corners_cam[:, 0:1],
                                              match_corners_cam[:, 2:3])
            match_corners_frm = tu.cameratoimage_torch(match_corners_cam,
                                                       projection)
            match_2dcs = match_corners_frm
        else:
            if det_depths is not None:
                match_depths = det_depths[inds][valids]
            else:
                match_depths = None
            if det_2dcs is not None:
                match_2dcs = det_2dcs[inds][valids]
            else:
                match_2dcs = None
            if det_dims is not None:
                match_dims = det_dims[inds][valids]
            else:
                match_dims = None
            if det_alphas is not None:
                match_alphas = det_alphas[inds][valids]
            else:
                match_alphas = None

        # parse tracking results
        track_inds = ids > -1
        track_bboxes = match_bboxes[track_inds]
        track_labels = match_labels[track_inds]
        if match_depths is not None:
            track_depths = match_depths[track_inds]
        else:
            track_depths = None
        if match_dims is not None:
            track_dims = match_dims[track_inds]
        else:
            track_dims = None
        if match_alphas is not None:
            track_alphas = match_alphas[track_inds]
        else:
            track_alphas = None
        if match_2dcs is not None:
            track_2dcs = match_2dcs[track_inds]
        else:
            track_2dcs = None
        track_ids = ids[track_inds]
        track_results = track2results(track_bboxes, track_labels, track_ids)

        outputs = dict(
            bbox_results=bbox_results,
            depth_results=track_depths,
            dim_results=track_dims,
            alpha_results=track_alphas,
            cen_2ds_results=track_2dcs,
            track_results=track_results)
        
        # show or save_txt
        if is_kitti:
            self.save_trk_txt(
                outputs,
                dict(Pedestrian=0, Cyclist=1, Car=2),
                img_meta,
                use_3d_box_center=use_3d_center,
                adjust_center=is_kitti)

        return outputs, use_3d_center


    def save_trk_txt(self,
                     outputs,
                     cfg,
                     img_meta,
                     use_3d_box_center=False,
                     adjust_center=False):
        """
        Only for KITTI dataset
        #Values    Name      Description
        ----------------------------------------------------------------------
        1   frame       Frame within the sequence where the object appearers
        1   track id    Unique tracking id of this object within this sequence
        1   type        Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
        1   truncated   Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries.
                        Truncation 2 indicates an ignored object (in particular
                        in the beginning or end of a track) introduced by manual
                        labeling.
        1   occluded    Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
        1   alpha       Observation angle of object, ranging [-pi..pi]
        4   bbox        2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
        3   dimensions  3D object dimensions: height, width, length (in meters)
        3   location    3D object location x,y,z in camera coordinates (in meters)
        1   rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1   score       Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.

        Args:
            outputs (dict): prediction results
            class_cfg (dict): a dict to convert class.
            img_meta (dict): image meta information.
        """
        self.out = "output/KITTI"
        out_folder = os.path.join(self.out, 'txts')
        os.makedirs(out_folder, exist_ok=True)
        img_info = img_meta['img_info']
        vid_name = os.path.dirname(img_info['file_name']).split('/')[-1]
        txt_file = os.path.join(out_folder, '{}.txt'.format(vid_name))

        # Expand dimension of results
        n_obj_detect = len(outputs['track_results'])
        if outputs.get('depth_results', None) is not None:
            depths = outputs['depth_results'].cpu().numpy().reshape(-1, 1)
        else:
            depths = np.full((n_obj_detect, 1), -1000)
        if outputs.get('dim_results', None) is not None:
            dims = outputs['dim_results'].cpu().numpy().reshape(-1, 3)
        else:
            dims = np.full((n_obj_detect, 3), -1000)
        if outputs.get('alpha_results', None) is not None:
            alphas = outputs['alpha_results'].cpu().numpy().reshape(-1, 1)
        else:
            alphas = np.full((n_obj_detect, 1), -10)

        if outputs.get('cen_2ds_results', None) is not None:
            centers = outputs['cen_2ds_results'].cpu().numpy().reshape(-1, 2)
        else:
            centers = [None] * n_obj_detect

        lines = []
        for (trackId, bbox), depth, dim, alpha, cen in zip(
                outputs['track_results'].items(), depths, dims, alphas,
                centers):
            loc, label = bbox['bbox'], bbox['label']
            if use_3d_box_center and cen is not None:
                box_cen = cen
            else:
                box_cen = np.array([loc[0] + loc[2], loc[1] + loc[3]]) / 2
            if alpha == -10:
                roty = np.full((1, ), -10)
            else:
                roty = tu.alpha2rot_y(alpha,
                                      box_cen[0] - img_info['width'] / 2,
                                      img_info['cali'][0][0])
            if np.all(depths == -1000):
                trans = np.full((3, ), -1000)
            else:
                trans = tu.imagetocamera(box_cen[None], depth,
                                         np.array(img_info['cali'])).flatten()

            if adjust_center:
                # KITTI GT uses the bottom of the car as center (x, 0, z).
                # Prediction uses center of the bbox as center (x, y, z).
                # So we align them to the bottom center as GT does
                trans[1] += dim[0] / 2.0

            cat = ''
            for key in cfg:
                if bbox['label'] == cfg[key]:
                    cat = key.lower()
                    break
            
            if cat == '':
                continue

            # Create lines of results
            line = f"{img_info['index']} {trackId} {cat} {-1} {-1} " \
                   f"{alpha.item():.6f} " \
                   f"{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {loc[3]:.6f} " \
                   f"{dim[0]:.6f} {dim[1]:.6f} {dim[2]:.6f} " \
                   f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} " \
                   f"{roty.item():.6f} {loc[4]:.6f}\n"
            lines.append(line)

        if txt_file in self.writed:
            mode = 'a'
        else:
            mode = 'w'
            self.writed.append(txt_file)
        if len(lines) > 0:
            with open(txt_file, mode) as f:
                f.writelines(lines)
        else:
            with open(txt_file, mode):
                pass
