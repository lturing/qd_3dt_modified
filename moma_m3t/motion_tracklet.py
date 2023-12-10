import numpy as np
import torch

class MotionTracklet(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, device, bbox3D, cur_frame):
        """
        Initialises a tracker using initial bounding box.
        """
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        self.device = device
        self.loc_dim = 7
        self.pos_dim = 3
        self.id = MotionTracklet.count
        MotionTracklet.count += 1
        self.nfr = 5
        self.init_flag = True
        self.div_scale = 10 #reduce the postion scale
        self.obj_state = bbox3D.reshape((self.loc_dim,)) 
        self.history = np.tile(
            np.zeros_like(bbox3D[:self.loc_dim]), (self.nfr, 1))
        self.ref_history = np.tile(bbox3D[:self.loc_dim], (self.nfr + 1, 1))
        self.cur_frame = np.tile(cur_frame, (self.nfr + 1, 1))
        self.avg_angle = bbox3D[3]
        self.avg_dim = np.array(bbox3D[4:])
        self.prev_obs = bbox3D.copy()
        self.prev_ref = bbox3D[:self.loc_dim].copy()

    @staticmethod
    def fix_alpha(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def update_array(origin_array: np.ndarray,
                     input_array: np.ndarray) -> np.ndarray:
        new_array = origin_array.copy()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    def _update_history(self, bbox3D, cur_frame):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.cur_frame = self.update_array(self.cur_frame, cur_frame)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2])
        
        # align orientation history
        if self.loc_dim > 3:
            self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def _init_history(self, bbox3D, cur_frame):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.cur_frame = self.update_array(self.cur_frame, cur_frame)
        self.history = np.tile([self.ref_history[-1] - self.ref_history[-2]],
                               (self.nfr, 1))
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def update(self, bbox3D, cur_frame):
        """
        Updates the state vector with observed bbox.
        """
        obj = bbox3D[:self.loc_dim]
        
        if self.loc_dim > 3:
            obj[3] = self.fix_alpha(obj[3])

        self.obj_state[:self.loc_dim] = obj
        self.prev_obs = bbox3D

        if self.loc_dim > 3 and np.pi / 2.0 < abs(bbox3D[3] - self.avg_angle) < np.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][3] = self.fix_alpha(
                    self.ref_history[r_indx][3] + np.pi)

        if self.init_flag:
            self._init_history(obj, cur_frame)
            self.init_flag = False
        else:
            self._update_history(obj, cur_frame)

    def get_matching_history(self, position):
        obj_start = position.cpu().numpy()
        obj_obs_past = self.ref_history[...,:self.loc_dim].copy()
        obj_obs_past[:,:self.pos_dim] -= obj_start
        obj_obs_past[..., :self.pos_dim] /= self.div_scale
        obj_obs_past = torch.from_numpy(obj_obs_past).to(self.device).unsqueeze(0)
        return obj_obs_past

    def get_fr_idx(self):
        
        return self.cur_frame.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


TRACKER_MODEL_ZOO = {
    'TransformerTracker': MotionTracklet,
}

def get_tracker(tracker_model_name) -> object:
    tracker_model = TRACKER_MODEL_ZOO.get(tracker_model_name, None)
    if tracker_model is None:
        raise NotImplementedError

    return tracker_model

