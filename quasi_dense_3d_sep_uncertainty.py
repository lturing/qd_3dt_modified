import torch
import torch.nn as nn
import numpy as np
from addict import Dict
from pyquaternion import Quaternion
from collections import defaultdict

from utils import bbox2result, track2results, imagetocamera_torch, cameratoworld_torch, alpha2yaw_torch, worldtocamera_torch, euler_to_quaternion, yaw2alpha_torch, cameratoimage_torch, quaternion_to_euler, computeboxes, draw_3d_bbox, generate_color, get2dBox
from module import DLA, DLAUp, RPNHead, SingleRoIExtractor, ConvFCBBoxHead, ConvFCBBox3DRotSepConfidenceHead, MultiPos3DTrackHead
from tracker_3d.embedding_3d_bev_motion_uncertainty_tracker import Embedding3DBEVMotionUncertaintyTracker
import cv2 

from moma_m3t.motion_tracker import MotionTracker

class QuasiDense3DSepUncertainty(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 bbox_3d_roi_extractor=None,
                 bbox_3d_head=None,
                 embed_roi_extractor=None,
                 embed_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QuasiDense3DSepUncertainty, self).__init__()

        backbone.pop('type')
        self.backbone = DLA(**backbone)

        if neck is not None:
            neck.pop('type')
            self.neck = DLAUp(**neck)

        if rpn_head is not None:
            rpn_head.pop('type')
            self.rpn_head = RPNHead(**rpn_head)
            
        if bbox_head is not None:
            bbox_roi_extractor.pop('type')
            bbox_head.pop('type')
            self.bbox_roi_extractor = SingleRoIExtractor(**bbox_roi_extractor)
            self.bbox_head = ConvFCBBoxHead(**bbox_head)

        if bbox_3d_head is not None:
            bbox_3d_head.pop('type')
            self.bbox_3d_head = ConvFCBBox3DRotSepConfidenceHead(**bbox_3d_head)
            
        if embed_head is not None:
            embed_roi_extractor.pop('type')
            embed_head.pop('type')
            self.embed_roi_extractor = SingleRoIExtractor(**embed_roi_extractor)
            self.embed_head = MultiPos3DTrackHead(**embed_head)

        self.test_cfg = test_cfg
        test_cfg['track'].pop('type')
        self.tracker = Embedding3DBEVMotionUncertaintyTracker(**test_cfg['track'])
        self.moma_tracker = MotionTracker(init_track_id = 0)

        #self.init_weights(pretrained=pretrained)

        self.spool = len(self.bbox_roi_extractor.featmap_strides)
        self.espool = len(self.embed_roi_extractor.featmap_strides)
        self.track_history = defaultdict(lambda: [])
        self.tracklet_history = {}
        self.colors = {}
        self.frame_count = 0


    def init_weights(self, pretrained=None):
        super(QuasiDense3DSepUncertainty, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_bbox_3d:
            self.bbox_3d_head.init_weights()
        if self.with_embed:
            self.embed_roi_extractor.init_weights()
            self.embed_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        #if self.with_neck:
        x = self.neck(x)
        return x


    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def bbox2roi(self, bbox_list, stack=True):
        """Convert a list of bboxes to roi format.

        Args:
            bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
                of images.

        Returns:
            Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        """
        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            if bboxes.size(0) > 0:
                img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
                rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
            else:
                rois = bboxes.new_zeros((0, 5))
            rois_list.append(rois)
        if stack:
            rois = torch.cat(rois_list, 0)
        else:
            rois = rois_list
        return rois


    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        
        rois = self.bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)

        cls_score, bbox_pred = \
            self.bbox_head(roi_feats)
        
        depth_pred, depth_uncertainty_pred, dim_pred, alpha_pred, cen_2d_pred = \
            self.bbox_3d_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels, det_depths, det_depths_uncertainty, det_dims, det_alphas, cen_2d_preds = \
            self.bbox_3d_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                depth_pred,
                depth_uncertainty_pred,
                dim_pred,
                alpha_pred,
                cen_2d_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
        return det_bboxes, det_labels, det_depths, det_depths_uncertainty, det_dims, det_alphas, cen_2d_preds


    def simple_test(self,
                    img,
                    img_meta,
                    obj_tracker=None,
                    pure_det=False,
                    proposals=None,
                    rescale=False):

        # init tracker
        frame_ind = img_meta[0].get('frame_id', -1)
        use_3d_center = self.test_cfg.get('use_3d_center', False)

        x = self.extract_feat(img)
        # rpn
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg['rpn']) 
        #proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg['rcnn'])
        # bbox head
        
        det_bboxes, det_labels, det_depths, det_depths_uncertainty, det_dims, det_alphas, det_2dcs = \
            self.simple_test_bboxes(x, img_meta, proposal_list,
                                    self.test_cfg['rcnn'], rescale=rescale)
        
        show_detect = False
        if show_detect:
            img = img_meta[0]['ori_img']

            pred = torch.cat([det_bboxes, det_labels.reshape(det_labels.shape[0], 1)], dim=1)
            tracker_outputs  = obj_tracker.update(pred.cpu(), img)

            #print(f"track_outputs.length={len(tracker_outputs)}, pred.size={pred.shape[0]}")
            
            for output in tracker_outputs:
                bbox, track_id, category_id, score = (
                    output[:4],
                    int(output[4]),
                    output[5],
                    output[6],
                )

                track = self.track_history[track_id]
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                track.append(((x+w)/2, (y+h)/2))  # x, y center point

                if len(track) > 60:  # retain 90 tracks for 90 frames
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) 
                cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=2)

            #print(f"img.shape={img.shape}")
            det_bboxes = det_bboxes.numpy()
            for i in range(det_bboxes.shape[0]):
                x1, y1, x2, y2, s = det_bboxes[i]
                #if s < 0.5: continue 

                x1 = int(x1 + 0.5)
                y1 = int(y1 + 0.5)
                x2 = int(x2 + 0.5)
                y2 = int(y2 + 0.5)
                thickness = 2
                #print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, s={s}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness, lineType=cv2.LINE_AA)  # object box
            
            
            cv2.imshow("Tracking", img)
            cv2.waitKey(1)

            return img 

            #if cv2.waitKey(1) & 0xFF == ord("q"): cv2.destroyAllWindows()

        track_flag = True
        if track_flag:
            bbox_results = bbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
            
            # re-pooling for embeddings
            if det_bboxes.size(0) != 0:
                #bboxes = det_bboxes * img_meta[0]['scale_factor']

                #det_bboxes[:, 0:3:2] *= img_meta[0]['scale_factor']['w']
                #det_bboxes[:, 1:4:2] *= img_meta[0]['scale_factor']['h']
                #det_2dcs[:, 0] *= img_meta[0]['scale_factor']['w']
                #det_2dcs[:, 1] *= img_meta[0]['scale_factor']['h']
                bboxes = det_bboxes 

                embed_rois = self.bbox2roi([bboxes])
                embed_feats = self.embed_roi_extractor(x[:self.spool], embed_rois)
                embeds, emb_depth = self.embed_head(embed_feats)
            else:
                embeds = det_bboxes.new_zeros(
                    [det_bboxes.shape[0], self.embed_head.embed_channels])

            # TODO: use boxes_3d to match KF3d in track
            projection = det_bboxes.new_tensor(img_meta[0]['calib'])
            position = det_bboxes.new_tensor(img_meta[0]['pose']['position'])
            r_camera_to_world = img_meta[0]['pose']['rotation']
            
            rotation = det_bboxes.new_tensor(r_camera_to_world)
            cam_rot_quat = Quaternion(matrix=r_camera_to_world, rtol=1e-04, atol=1e-05)
            quat_det_yaws_world = {'roll_pitch': [], 'yaw_world': []}

            if det_depths is not None and det_2dcs is not None:
                corners = imagetocamera_torch(det_2dcs, det_depths, projection)
                corners_global = cameratoworld_torch(corners, position,
                                                        rotation)
                det_yaws = alpha2yaw_torch(det_alphas, corners[:, 0:1],
                                            corners[:, 2:3])

                for det_yaw in det_yaws:
                    yaw_quat = Quaternion(
                        axis=[0, 1, 0], radians=det_yaw.cpu().numpy())
                    rotation_world = cam_rot_quat * yaw_quat # T_w_obj
                    if rotation_world.z < 0:
                        rotation_world *= -1
                    roll_world, pitch_world, yaw_world = quaternion_to_euler(
                        rotation_world.w, rotation_world.x, rotation_world.y,
                        rotation_world.z)
                    quat_det_yaws_world['roll_pitch'].append(
                        [roll_world, pitch_world])
                    quat_det_yaws_world['yaw_world'].append(yaw_world)

                det_yaws_world = rotation.new_tensor(
                    np.array(quat_det_yaws_world['yaw_world'])[:, None])
                
                det_roll_pitch_world = rotation.new_tensor(
                    np.array(quat_det_yaws_world['roll_pitch']))

                det_boxes_3d = torch.cat(
                    [corners_global, det_yaws_world, det_dims], dim=1)
            else:
                det_boxes_3d = det_bboxes.new_zeros([det_bboxes.shape[0], 7])
                det_roll_pitch_world = det_bboxes.new_zeros([det_bboxes.shape[0], 2])

            if 1:
                match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
                    self.tracker.match(
                        bboxes=det_bboxes,
                        labels=det_labels,
                        boxes_3d=det_boxes_3d,
                        depth_uncertainty=det_depths_uncertainty,
                        position=position,
                        rotation=rotation,
                        embeds=embeds,
                        cur_frame=frame_ind,
                        pure_det=pure_det)
            else:
                match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
                    self.moma_tracker.match(
                        bboxes=det_bboxes,
                        labels=det_labels,
                        boxes_3d=det_boxes_3d,
                        position=position,
                        rotation=rotation,
                        cur_frame=frame_ind,)

            match_yaws = []
            if det_depths is not None and det_2dcs is not None:
                match_dims = match_boxes_3ds[:, -3:]
                match_corners_cam = worldtocamera_torch(match_boxes_3ds[:, :3],
                                                        position, rotation)
                match_depths = match_corners_cam[:, 2:3]

                for match_order, match_yaw in zip(
                        inds[valids].cpu().numpy(),
                        match_boxes_3ds[:, 3].cpu().numpy()):
                    roll_world, pitch_world = quat_det_yaws_world['roll_pitch'][
                        match_order]
                    rotation_cam = cam_rot_quat.inverse * Quaternion(
                        euler_to_quaternion(roll_world, pitch_world, match_yaw))
                    vtrans = np.dot(rotation_cam.rotation_matrix,
                                    np.array([1, 0, 0]))
                    match_yaws.append(-np.arctan2(vtrans[2], vtrans[0]))

                match_yaws = rotation.new_tensor(np.array(match_yaws)).unsqueeze(1)
                match_alphas = yaw2alpha_torch(match_yaws,
                                                match_corners_cam[:, 0:1],
                                                match_corners_cam[:, 2:3])
                
                match_corners_frm = cameratoimage_torch(match_corners_cam,
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
                track_corners_cam = match_corners_cam[track_inds]
            else:
                track_depths = None
            if match_dims is not None:
                track_dims = match_dims[track_inds]
            else:
                track_dims = None
            if match_alphas is not None:
                track_alphas = match_alphas[track_inds]
                match_yaws = match_yaws[track_inds]
            else:
                track_alphas = None
                match_yaws = []
            if match_2dcs is not None:
                track_2dcs = match_2dcs[track_inds]
            else:
                track_2dcs = None
            track_ids = ids[track_inds]

            track_results = track2results(track_bboxes, track_labels, track_ids)
            outputs = dict(
                bbox_results=bbox_results,
                depth_results=track_depths,
                depth_uncertainty_results=det_depths_uncertainty,
                dim_results=track_dims,
                alpha_results=track_alphas,
                cen_2ds_results=track_2dcs,
                track_results=track_results)

            #print(f"det_bboxes.shape={det_bboxes.shape}, track_2dcs.shape={track_2dcs.shape}, track_alphas={track_alphas.shape}, track_bboxes.shape={track_bboxes.shape}")
            #return outputs, use_3d_center

            # tracklet_history
            show_detect = True 
            if show_detect:
                if 0:
                    img = img[0].permute(1, 2, 0)
                    img = img.numpy() * img_meta[0]['std'] + img_meta[0]['mean']
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                img = img_meta[0]['ori_img']

                #det_bboxes[:, 0:3:2] *= img_meta[0]['scale_factor']['w']
                #det_bboxes[:, 1:4:2] *= img_meta[0]['scale_factor']['h']

                pred = torch.cat([track_bboxes, track_labels.reshape(track_bboxes.shape[0], 1)], dim=1)
                tracker_outputs  = obj_tracker.update(pred.cpu(), img)
                for output in tracker_outputs:
                    bbox, track_id, category_id, score = (
                        output[:4],
                        int(output[4]),
                        output[5],
                        output[6],
                    )

                    track = self.track_history[track_id]
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    track.append(((x+w)/2, (y+h)/2))  # x, y center point
                    if len(track) > 60:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) 
                    #cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=2)

                #print(f"img.shape={img.shape}")
                track_bboxes = track_bboxes.numpy()
                for i in range(track_bboxes.shape[0]):
                    x1, y1, x2, y2, s = track_bboxes[i]
                    #if s < 0.5: continue 

                    x1 = int(x1 + 0.5)
                    y1 = int(y1 + 0.5)
                    x2 = int(x2 + 0.5)
                    y2 = int(y2 + 0.5)
                    thickness = 2
                    #print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, s={s}")
                    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness, lineType=cv2.LINE_AA)  # object box

                dims = track_dims.numpy()
                yaws = match_yaws.numpy()
                locs = track_corners_cam.numpy()
                scale_dict = img_meta[0]['scale_factor']
                self.frame_count += 1

                # (track_bboxes, track_labels, track_ids)
                # [cuboid_center(3), yaw, cuboid_scale(3), [x1 y1 w h]], prob track_id # in camera's coordinate
                #print(f"track_bboxes.shape={track_bboxes.shape}, track_labels.shape={track_labels.shape}, track_ids.shape={track_ids.shape}, track_dims.shape={track_dims.shape}, match_yaws.shape={match_yaws.shape}, track_corners_cam.shape={track_corners_cam.shape}")
                # track_bboxes.shape=(12, 5), track_labels.shape=torch.Size([12]), track_ids.shape=torch.Size([12]), track_dims.shape=torch.Size([12, 3]), match_yaws.shape=torch.Size([12, 1]), track_corners_cam.shape=torch.Size([12, 3])
                f = open(img_meta[0]['result'], 'w', encoding='utf-8')
                for i in range(track_dims.shape[0]):
                    roty = [yaws[i][0]]
                    dim = dims[i]
                    loc = locs[i]
                    tid = int(track_ids[i])
                    assert tid > -1
                    #det_bboxes[:, 0:3:2] *= img_meta[0]['scale_factor']['w']
                    #det_bboxes[:, 1:4:2] *= img_meta[0]['scale_factor']['h']
                    #det_2dcs[:, 0] *= img_meta[0]['scale_factor']['w']
                    #det_2dcs[:, 1] *= img_meta[0]['scale_factor']['h']
                    cam_pose = None 
                    p3ds_camera = computeboxes(roty, dim, loc) # (8, 3)
                    
                    box = get2dBox(projection, cam_pose, p3ds_camera, img_meta[0]['img_shape'])
                    if not box:
                        print('invaild box and skip')
                        continue 

                    # position
                    traslation = position.cpu().numpy()
                    rotation = r_camera_to_world # T_w_c
                    p3ds_world = p3ds_camera.dot(rotation.T) + traslation[np.newaxis] # 8 * 3
                    loc_world = loc.reshape(-1, 3).dot(rotation.T) + traslation[np.newaxis]
                    loc_world = loc_world.reshape(-1)

                    if tid not in self.tracklet_history:
                        self.tracklet_history[tid] = []
                        self.colors[tid] = generate_color(tid)
                    
                    self.tracklet_history[tid].append([loc_world, p3ds_world, self.colors[tid], self.frame_count])

                    draw_3d_bbox(img, p3ds_camera, projection, cam_pose, line_color=[int(it * 255) for it in self.colors[tid][::-1]])

                    # # [cuboid_center(3), yaw, cuboid_scale(3), [x1 y1 w h]], prob track_id # in camera's coordinate
                    f.write(' '.join(map(str, [loc[0], loc[1], loc[2], roty[0], dim[0], dim[1], dim[2], box[0], box[1], box[2], box[3], track_bboxes[i][4], tid])) + '\n')
                f.close()
                # draw_3d_bbox
                # get_3d_bbox_vertex

                if False:
                    cv2.imshow("Tracking", img)
                    cv2.waitKey(1)

            return img

