import argparse
from collections import defaultdict
import os.path as osp
import sys 
from importlib import import_module
import sys, importlib.util
from quasi_dense_3d_sep_uncertainty import QuasiDense3DSepUncertainty
from tracker.tracker_module import load_tracker_module
import torch
import glob
import os 
import cv2
import numpy as np
from tqdm import tqdm 
from gps_to_xyz import load_oxts_packets_and_poses, load_calib


kitti_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'truck': 4,
    'tram': 5,
    'misc': 6,
    'dontcare': 7
}


def parse_args():
    parser = argparse.ArgumentParser(description='qd3dt test detector')
    parser.add_argument('--config', help='config file path', default='/home/spurs/x/yolov8/qd-3dt/configs/KITTI/quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_mod_anchor_ratio_small_strides_GTA.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='/home/spurs/x/yolov8/qd-3dt/latest_kitti.pth')
    parser.add_argument('--lstm_checkpoint', help='track 3d checkpoint file', default='/home/spurs/x/yolov8/qd-3dt/batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth')
    parser.add_argument('--out', help='output result file')
    if 1:
        parser.add_argument('--data', help='the directory of the dataset', default = '/home/spurs/dataset/kitti_raw/2011_09_29/2011_09_29_drive_0071_sync/image_02/data')
        parser.add_argument('--pose', help='the pose directory', default='/home/spurs/dataset/kitti_raw/2011_09_29/2011_09_29_drive_0071_sync/oxts/data')
        parser.add_argument('--cali', help='the calibration file directory', default='/home/spurs/dataset/kitti_raw/2011_09_29')
    else:
        parser.add_argument('--data', help='the directory of the dataset', default='/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data')
        parser.add_argument('--pose', help='the pose directory', default='/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/oxts/data')
        parser.add_argument('--cali', help='the calibration file directory', default='/home/spurs/dataset/kitti_raw/2011_10_03')
    args = parser.parse_args()

    return args

def loadConfig(filename):
    filename = osp.abspath(osp.expanduser(filename))
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }

    return cfg_dict

def loadConfigStr(ckpt):
    spec = importlib.util.spec_from_loader("cfg", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(ckpt['meta']['config'], module.__dict__)
    return module

def build_model(args):
    ''' 
    cfg = loadConfig(args.config)
    cfg['test_cfg']['use_3d_center'] = True
    cfg['test_cfg']['track.with_bbox_iou'] = True
    cfg['test_cfg']['track.with_deep_feat'] = True
    cfg['test_cfg']['track.with_depth_ordering'] = True
    cfg['test_cfg']['track.with_depth_uncertainty'] = True
    cfg['test_cfg']['track.init_score_thr'] = 0.5
    cfg['test_cfg']['track.nms_class_iou_thr'] = 0.8
    cfg['test_cfg']['track.motion_momentum'] = 0.9
    cfg['test_cfg']['track.track_bbox_iou'] = 'box3d'
    cfg['test_cfg']['track.depth_match_metric'] = 'motion'
    cfg['test_cfg']['track.tracker_model_name'] = 'LSTM3DTracker'
    '''
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    cfg = loadConfigStr(checkpoint)
    cfg.model.pop('type')

    params = {}
    for k in cfg.model:
        params[k] = cfg.model[k]
    params['test_cfg'] = cfg.test_cfg 

    '''
    in checkpoint
    params['test_cfg']['track']={
        'type': 'Embedding3DBEVMotionUncertaintyTracker', 
        'init_score_thr': 0.8, 
        'init_track_id': 0, 
        'obj_score_thr': 0.5, 
        'match_score_thr': 0.5, 
        'memo_tracklet_frames': 10, 
        'memo_backdrop_frames': 1, 
        'memo_momentum': 0.8, 
        'motion_momentum': 0.8, 
        'nms_conf_thr': 0.5, 
        'nms_backdrop_iou_thr': 0.3, 
        'nms_class_iou_thr': 0.7, 
        'with_deep_feat': True, 
        'with_cats': True, 
        'with_bbox_iou': True, 
        'with_depth_ordering': True, 
        'track_bbox_iou': 'box3d', 
        'depth_match_metric': 'motion', 
        'tracker_model_name': 'DummyTracker', 
        'match_metric': 'cycle_softmax', 
        'lstm_ckpt_name': '/home/spurs/x/yolov8/qd-3dt/batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth'
        }
    '''
    
    '''
    in config file
        type='Embedding3DBEVMotionUncertaintyTracker',
        init_score_thr=0.8,
        init_track_id=0,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        motion_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        loc_dim=7,
        with_deep_feat=True,
        with_cats=True,
        with_bbox_iou=True,
        with_depth_ordering=True,
        lstm_name='VeloLSTM',
        lstm_ckpt_name=
        './checkpoints/batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth',
        track_bbox_iou='box3d',
        depth_match_metric='motion',
        tracker_model_name='DummyTracker',
        match_metric='cycle_softmax',
        match_algo='greedy'
    
    '''
    params['test_cfg']['use_3d_center'] = True 
    params['test_cfg']['track']['with_bbox_iou'] = True 
    params['test_cfg']['track']['with_deep_feat'] = True
    params['test_cfg']['track']['with_depth_ordering'] = True
    params['test_cfg']['track']['with_depth_uncertainty'] = True
    params['test_cfg']['track']['init_score_thr'] = 0.5
    params['test_cfg']['track']['nms_class_iou_thr'] = 0.8
    params['test_cfg']['track']['motion_momentum'] = 0.9
    params['test_cfg']['track']['track_bbox_iou'] = 'box3d'
    params['test_cfg']['track']['depth_match_metric'] = 'motion'
    params['test_cfg']['track']['tracker_model_name'] = 'LSTM3DTracker'
    params['test_cfg']['track']['lstm_name'] = 'VeloLSTM'
    params['test_cfg']['track']['lstm_ckpt_name'] = args.lstm_checkpoint
    params['test_cfg']['track']['loc_dim'] = 7
    params['test_cfg']['track']['match_algo'] = 'greedy'


    print(f"params['test_cfg']['track']={params['test_cfg']['track']}")

    #model = QuasiDense3DSepUncertainty(**cfg.model)
    model = QuasiDense3DSepUncertainty(**params)

    ks = list(checkpoint['state_dict'].keys())
    for k in ks: 
        if 'embed_head.convs' in k and 'gn' in k:
            nk = k.replace('gn', 'norm')
            checkpoint['state_dict'][nk] = checkpoint['state_dict'][k]
            #params.add(k)

    for k in model.state_dict():
        assert k in checkpoint['state_dict']
        

    ''' 
    checkpoint['state_dict']['embed_head.convs.0.norm.weight'] = checkpoint['state_dict']['embed_head.convs.0.gn.weight']
    checkpoint['state_dict']['embed_head.convs.0.norm.bias'] = checkpoint['state_dict']['embed_head.convs.0.gn.bias']

    checkpoint['state_dict']['embed_head.convs.1.norm.weight'] = checkpoint['state_dict']['embed_head.convs.1.gn.weight']
    checkpoint['state_dict']['embed_head.convs.1.norm.bias'] = checkpoint['state_dict']['embed_head.convs.1.gn.bias']

    checkpoint['state_dict']['embed_head.convs.2.norm.weight'] = checkpoint['state_dict']['embed_head.convs.2.gn.weight']
    checkpoint['state_dict']['embed_head.convs.2.norm.bias'] = checkpoint['state_dict']['embed_head.convs.2.gn.bias']

    checkpoint['state_dict']['embed_head.convs.3.norm.weight'] = checkpoint['state_dict']['embed_head.convs.3.gn.weight']
    checkpoint['state_dict']['embed_head.convs.3.norm.bias'] = checkpoint['state_dict']['embed_head.convs.3.gn.bias']
    '''

    model.load_state_dict(checkpoint['state_dict'])
    
    model.CLASSES = checkpoint['meta']['CLASSES']

    img_norm = cfg.data['train']['img_norm_cfg']

    model.eval()
    return model, img_norm

def load_data(args):
    imgs = glob.glob(os.path.join(args.data, "*.png"))
    imgs.sort()
    return imgs


def get_img_shape(img_path):
    img = cv2.imread(img_path) # bgr
    return img.shape 


def preprocess(img_path, img_norm, oxts, calib):
    #print(img_path)
    img = cv2.imread(img_path) # bgr
    ori_img = img.copy()
    #print(type(img), img.dtype)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    #print(img.shape) # 376, 1241, 3

    old_height, old_width = img.shape[:2]
    
    if 0:
        if True:
            if img.shape[0] % (2**6) != 0:
                new_height = (img.shape[0] // (2**6) + 1) * (2**6)
            else:
                new_height = img.shape[0] // (2**6) * (2**6)
            
            if img.shape[1] % (2**6) != 0:
                new_width = (img.shape[1] // (2**6) + 1) * (2**6)
            else:
                new_width = img.shape[1] // (2**6) * (2**6)

            #print(f"img.shape={img.shape}, new_height={new_height}, new_width={new_width}")
            img = cv2.resize(img, (new_width, new_height))
        else:
            new_height = img.shape[0] // (2**6) * (2**6)
            new_width = img.shape[1] // (2**6) * (2**6)
            #print(f"img.shape={img.shape}, new_height={new_height}, new_width={new_width}")
            img = img[:new_height, :new_width, :]
            old_height = new_height 
            old_width = new_width
    else:
        new_width, new_height = old_width, old_height

    #print(f"new img.shape={img.shape}")

    #img = np.zeros((1485, 448, 3)).astype(np.float32)
    mean = np.array(img_norm['mean'], dtype=np.float32)
    std = np.array(img_norm['std'], dtype=np.float32)
    
    img = (img-mean) / std
    #print(f"new img.shape={img.shape}")
    
    img = torch.from_numpy(img.copy())
    img = torch.unsqueeze(img, 0)
    #img = img.permute(0, 3, 1, 2)
    img = img.permute(0, 3, 1, 2)

    img_shape = img.shape 
    #print(f"after img.shape={img.shape}")
    name = os.path.basename(img_path).split('.')[0]
    assert name in oxts 
    T_w_imu = oxts[name].T_w_imu 
    T_imu_cam2 = calib['T_imu_cam2']
    T_w_cam2 = np.matmul(T_w_imu, T_imu_cam2)
    
    img_info = [{
                    'img_shape': [img_shape[2], img_shape[3]], 
                    'ori_img' : ori_img, 
                    'scale_factor': {'h': old_height * 1.0 / new_height, 'w': old_width * 1.0 / new_width}, 
                    "mean": mean, 
                    "std": std,
                    "pose": {"rotation": T_w_cam2[:3, :3], "position": T_w_cam2[:3, 3]},
                    #"calib": np.hstack([calib['K_cam2'], np.asarray([0, 0, 1])])
                    "calib": calib['K_cam2']
                    
                }]

    return img, img_info


if __name__ == "__main__":
    args = parse_args()
    
    imgs = load_data(args)
    model, img_norm = build_model(args)
    shape = get_img_shape(imgs[0])

    size = (shape[1], shape[0])

    print(f"video size: {size}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = 'output.mp4'
    fps = 10

    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, size, True)

    calib_imu_to_velo_filepath =  os.path.join(args.cali, 'calib_imu_to_velo.txt')
    velo_to_cam_filepath = os.path.join(args.cali, 'calib_velo_to_cam.txt')
    cam_to_cam_filepath = os.path.join(args.cali, 'calib_cam_to_cam.txt')

    calib = load_calib(calib_imu_to_velo_filepath, velo_to_cam_filepath, cam_to_cam_filepath)
    files = glob.glob(os.path.join(args.pose, '*.txt'))
    files.sort()

    oxts = load_oxts_packets_and_poses(files)

    tracker_type = 'BYTETRACK'
    tracker_config_path = './tracker/config/byte_track.yaml'
    config_path = "./tracker/config/default_config.yaml"
    tracker_module = load_tracker_module(
            config_path=config_path,
            tracker_type=tracker_type,
            tracker_config_path=tracker_config_path,
        )
    
    #print(f"k_cam2={calib['K_cam2']}")
    for i, ipath in enumerate(tqdm(imgs)):
        img, img_info = preprocess(ipath, img_norm, oxts, calib)
        with torch.no_grad():
            img = model.simple_test(img, img_info, obj_tracker=tracker_module)
            videoWriter.write(img)
        
        if 0xFF == ord("q"):
            break 

        if i == 0:
            print(f"img.shape={img_info[0]['img_shape']}, ori_img.shape={img_info[0]['ori_img'].shape[:2]}")
    
    videoWriter.release()
    print('sucess')

