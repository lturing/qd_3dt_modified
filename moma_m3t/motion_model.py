import torch
import torch.nn as nn
import torch.nn.functional as F
from moma_m3t.motion_encoder import TemporalEncoder, StateEncoder, SpatialEncoder

class MotionTrans(nn.Module):

    def __init__(self,
                 feature_dim: int = 128,
                 time_steps: int = 5,
                 loc_dim: int = 7):
        super(MotionTrans, self).__init__()

        self.fc = nn.Linear(loc_dim, feature_dim//2, bias=False)
        self.fc2 = nn.Linear(feature_dim//2, feature_dim, bias=False)

        self.coord_fc1 = nn.Linear(feature_dim, feature_dim//2, bias=False)
        self.coord_fc2 = nn.Linear(feature_dim//2, 1, bias=False)

        self.temporal = TemporalEncoder(historical_steps=time_steps,
                                    embed_dim=feature_dim,
                                    num_heads=4,
                                    dropout=0.1,
                                    num_layers=2)

        self.spatial = SpatialEncoder(feature_dim)
        
        xy_dim = 2
        self.rel_emb = StateEncoder(xy_dim, feature_dim)

        self.tpe = nn.Embedding(100, feature_dim)

    def forward(self, pre_coord_all, cur_coord_all, fr_diff, pre_mask=None, cur_mask=None):
        if pre_coord_all.dim() == 3:
            pre_coord_all = pre_coord_all.unsqueeze(0)
            cur_coord_all = cur_coord_all.unsqueeze(0)
        if type(pre_mask) == type(None):
            pre_mask = torch.ones(pre_coord_all.shape[0], pre_coord_all.shape[1])
            cur_mask = torch.ones(cur_coord_all.shape[0], cur_coord_all.shape[1])
        cur_coord_all = cur_coord_all.unsqueeze(1)
        
        pre_num = pre_coord_all.shape[1]
        cur_num = cur_coord_all.shape[2]
        
        pre_last = pre_coord_all[...,-1,:2].detach().clone().unsqueeze(2)

        pre_trans = pre_last.permute(0,2,1,3)
        relative = pre_trans - pre_last
        pos_emb = self.rel_emb(pre_last).squeeze(2)

        cur = cur_coord_all[...,:2].detach().clone()

        cur = cur - pre_last
        
        cur_coord_all = cur_coord_all.expand(-1,pre_num,-1,-1).clone()

        cur_coord_all[...,:2] = cur

        cur = cur_coord_all.detach()
        
        pre_interval = pre_coord_all[...,1:,:2] - pre_coord_all[...,:-1,:2]
        
        pre = pre_coord_all[...,1:,:].detach().clone()
        pre[...,:2] = pre_interval

        fr_diff = fr_diff[...,:-1]

        pre = pre.detach()
        cur = cur.detach()

        b, np, l, d = pre.shape
        pre = pre.view(-1,l,d)

        pre = self.fc2(F.relu(self.fc(pre)))
        tpe = self.tpe(fr_diff.long()).view(-1,l,self.tpe.weight.shape[-1])
        pre = pre + tpe

        mask = torch.zeros(b*np, l).bool()
        pre = self.temporal(pre.permute(1,0,2), mask).unsqueeze(-2)

        pre = pre.view(b,np,-1)
        cur = self.fc2(F.relu(self.fc(cur)))
        
        pre = self.spatial(pre, pre, pos_emb, pre_mask, pre_mask).unsqueeze(2)
        
        pre_feat = F.normalize(pre, dim = -1)
        pre = pre.expand(-1,-1,cur_num,-1)

        seq = torch.cat([pre, cur], -1)

        coord = self.coord_fc2(F.relu(self.coord_fc1(pre-cur))).squeeze(-1)
        
        cur_feat = F.normalize(cur, dim = -1)

        return coord#, pre_feat, cur_feat
