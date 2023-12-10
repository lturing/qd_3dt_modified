import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from moma_m3t.motion_model import MotionTrans
from moma_m3t.motion_tracklet import MotionTracklet



def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious



class MotionTracker(object):

    def __init__(self, init_track_id):
        
        ####Config for nuScenes#####
        self.init_score_thr = 0.05
        self.score_thr = 0.08
        self.match_score_thr = 0.5
        self.memo_tracklet_frames = 10
        self.with_cats = True
        
        model_ckpt_name = '/home/spurs/x/yolov8/qd-3dt/moma_kitti.pth'        
        self.feat_dim = 128
        self.loc_dim = 7
        self.xyz_dim = 3
        self.nfr = 5
        self.xyz_scale = 10

        self.device = torch.device("cpu")
        self.model = MotionTrans(self.feat_dim, self.nfr, self.loc_dim).to(self.device)
        self.model.eval()
        ckpt = torch.load(model_ckpt_name)
        try:
            self.model.load_state_dict(ckpt['state_dict'])
        except (RuntimeError, KeyError) as ke:
            print("Cannot load full model: {}".format(ke))
            exit()
        print(f"=> Successfully loaded checkpoint {model_ckpt_name}")
        del ckpt

        self.num_tracklets = init_track_id
        self.tracklets = dict()
    
    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, boxes_3d, labels, cur_frame):
        tracklet_inds = ids > -1

        # update memo
        for tid, bbox, box_3d, label in zip(
                ids[tracklet_inds], bboxes[tracklet_inds],
                boxes_3d[tracklet_inds], labels[tracklet_inds]):
            tid = int(tid)
            if tid in self.tracklets.keys():
                self.tracklets[tid]['bbox'] = bbox

                self.tracklets[tid]['tracker'].update(
                    box_3d.cpu().numpy(),
                    cur_frame)

                tracker_box = self.tracklets[tid]['tracker'].get_state()[:self.loc_dim]
                pd_box_3d = box_3d.new_tensor(tracker_box)
                self.tracklets[tid]['box_3d'] = pd_box_3d
                self.tracklets[tid]['label'] = label
                self.tracklets[tid]['last_frame'] = cur_frame
            else:
                built_tracker = MotionTracklet(
                    self.device,
                    box_3d.cpu().numpy(),
                    cur_frame
                ) 
                self.tracklets[tid] = dict(
                    bbox=bbox,
                    box_3d=box_3d,
                    tracker=built_tracker,
                    label=label,
                    last_frame=cur_frame,
                    velocity=torch.zeros_like(box_3d),
                    acc_frame=0)

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if cur_frame - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_trackers = []
        memo_labels = []
        for k, v in self.tracklets.items():
            memo_trackers.append(v['tracker'])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return memo_labels, memo_trackers, memo_ids.squeeze(0)

    def match(self,
              bboxes: torch.Tensor,
              labels: torch.Tensor,
              boxes_3d: torch.Tensor,
              position: torch.Tensor,
              rotation: torch.Tensor,
              cur_frame: int):
        
        _, inds = bboxes[:, -1].sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        boxes_3d = boxes_3d[inds]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = 0.3 if bboxes[i, -1] < self.score_thr else 0.7
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        score_valids = bboxes[...,-1] > self.init_score_thr
        cls_valids = labels < 7 #TODO: main category number in nuScenes
        valids = score_valids * valids * cls_valids

        bboxes = bboxes[valids, :]
        labels = labels[valids]
        boxes_3d = boxes_3d[valids]

        motion_score_list = []

        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_labels, memo_trackers, memo_ids = self.memo
            
            motion_score_list = torch.zeros(len(memo_trackers), len(boxes_3d)).to(self.device)
            track_feat_list = torch.zeros(len(memo_trackers), self.nfr+1, self.loc_dim).to(self.device)
            cur_fr_list = torch.zeros(len(memo_trackers), self.nfr+1).to(self.device)

            for ind, memo_tracker in enumerate(memo_trackers):
                
                track_history = memo_tracker.get_matching_history(position)
                fr_idx_history = torch.from_numpy(memo_tracker.get_fr_idx())
                track_feat_list[ind] = track_history
                cur_fr_list[ind] = fr_idx_history
            
            obj_start = position.cpu().numpy()
            cur = boxes_3d[:,:self.loc_dim].clone()
            cur[:,:self.xyz_dim] -= torch.tensor(obj_start).to(self.device)
            cur[..., :self.xyz_dim] /= self.xyz_scale

            fr_diff = cur_frame - cur_fr_list

            score_list = self.model(track_feat_list, cur, fr_diff.unsqueeze(0))
            motion_score_list = torch.sigmoid(score_list)[0]
            scores_iou = motion_score_list.permute(1,0)
                   
            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores_cats = cat_same.float()
            else:
                scores_cats = scores_iou.new_ones(scores_iou.shape)

            scores = scores_iou * scores_cats

            #Hungarian matching
            matched_indices = linear_assignment(-scores.cpu().numpy())
            for idx in range(len(matched_indices[0])):
                i = matched_indices[0][idx]
                memo_ind = matched_indices[1][idx]
                conf = scores[i, memo_ind]
                tid = memo_ids[memo_ind]
                if conf > self.match_score_thr and tid > -1:
                    ids[i] = tid
                    scores[:i, memo_ind] = 0
                    scores[i + 1:, memo_ind] = 0
            del matched_indices

        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long)
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, boxes_3d, labels, cur_frame)

        update_bboxes = bboxes.detach().clone()
        update_labels = labels.detach().clone()
        update_boxes_3d = boxes_3d.detach().clone()
        for tid in ids[ids > -1]:
            update_boxes_3d[ids == tid] = self.tracklets[int(tid)]['box_3d']
        update_ids = ids.detach().clone()

        return update_bboxes, update_labels, update_boxes_3d, update_ids, inds, valids
