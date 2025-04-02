# train_experiment_dynamic.py

import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------
# (1) GarmentDataset (동적 포인트)
# --------------------------
class GarmentDataset(Dataset):
    def __init__(self, measurements_dir, dataset_dir):
        import json
        with open(measurements_dir, 'r') as f:
            self.measurements = json.load(f)
        self.dataset_dir = dataset_dir
        self.max_num_points = 0

        for item in self.measurements:
            file_path = os.path.join(self.dataset_dir, item['file_name'])
            mesh = trimesh.load(file_path)
            v_count = len(mesh.vertices)
            if v_count > self.max_num_points:
                self.max_num_points = v_count

        logging.info(f"[GarmentDataset] max_num_points = {self.max_num_points}")

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        data = self.measurements[idx]
        measurement = np.array([
            data['shoulder_width'],
            data['total_length'],
            data['sleeve_length'],
            data['chest_width']
        ], dtype=np.float32)

        cat_map = {'sleeveless': 0, 'tshirts': 1, 'opened-shirts': 2}
        cat_one_hot = np.zeros(3, dtype=np.float32)
        category = data.get('category', 'sleeveless')
        if category in cat_map:
            cat_one_hot[cat_map[category]] = 1

        input_vec = np.concatenate([measurement, cat_one_hot], axis=0)

        obj_path = os.path.join(self.dataset_dir, data['file_name'])
        mesh = trimesh.load(obj_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        normals = np.array(mesh.vertex_normals, dtype=np.float32)
        real_count = vertices.shape[0]

        if real_count < self.max_num_points:
            pad_sz = self.max_num_points - real_count
            v_pad = np.zeros((pad_sz, 3), dtype=np.float32)
            n_pad = np.zeros((pad_sz, 3), dtype=np.float32)
            vertices = np.vstack([vertices, v_pad])
            normals = np.vstack([normals, n_pad])

        inp_t = torch.tensor(input_vec, dtype=torch.float32)
        verts_t = torch.tensor(vertices, dtype=torch.float32)
        norms_t = torch.tensor(normals, dtype=torch.float32)
        count_t = torch.tensor(real_count, dtype=torch.int64)

        return inp_t, verts_t, norms_t, count_t

def collate_fn(batch):
    inps, verts, norms, counts = [], [], [], []
    for b in batch:
        inps.append(b[0])
        verts.append(b[1])
        norms.append(b[2])
        counts.append(b[3])
    inps  = torch.stack(inps, dim=0)
    verts = torch.stack(verts,dim=0)
    norms = torch.stack(norms,dim=0)
    counts= torch.stack(counts,dim=0)
    return inps, verts, norms, counts

# --------------------------
# (2) BigPredictor 정의 (Skip Connection 추가 가능)
# --------------------------
class BigPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, max_points=7078, use_skip=False, activation_type='relu'):
        super().__init__()
        self.max_points = max_points

        act_map = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.2),
            'silu': nn.SiLU()
        }
        act_fn = act_map.get(activation_type)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, max_points * 6)
        )
        self.use_skip = use_skip

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.max_points, 6)
        verts = out[:, :, :3]
        norms = out[:, :, 3:]
        if self.use_skip:
            verts += x[:, None, :3]  # Skip connection 적용 (입력 일부를 더함)
        return verts, norms

# --------------------------
# (3) 외부카운트 모델 적용
# --------------------------
class GarmentGeneratorWithExternalCount(nn.Module):
    def __init__(self, external_count_model, big_hidden=128, max_num_points=5000, use_skip=False, activation_type='relu'):
        super().__init__()
        self.external_count_model = external_count_model
        self.external_count_model.eval()
        for p in self.external_count_model.parameters():
            p.requires_grad = False

        self.max_num_points = max_num_points
        self.big_predictor = BigPredictor(
            input_dim=7, hidden_dim=big_hidden, max_points=max_num_points, use_skip=use_skip, activation_type=activation_type
        )

    def forward(self, x):
        with torch.no_grad():
            raw_count = self.external_count_model(x)
        pred_counts = torch.round(raw_count).long()
        pred_counts = torch.clamp(pred_counts, 1, self.max_num_points)
        verts, norms = self.big_predictor(x)
        return verts, norms, pred_counts

# --------------------------
# (4) Loss 함수
# --------------------------
def dynamic_chamfer_loss(pred_v, pred_n, pred_counts, tgt_v, tgt_n, tgt_counts, mode='chamfer_with_normals'):
    if mode == 'chamfer_only':
        return _dynamic_chamfer_core(pred_v, tgt_v, pred_counts, tgt_counts)
    elif mode == 'chamfer_with_normals':
        return _dynamic_chamfer_core(pred_v, tgt_v, pred_counts, tgt_counts) + _dynamic_normal_core(pred_n, tgt_n, pred_counts, tgt_counts)
    elif mode == 'multiscale':
        scales = [1.0, 0.5]
        return sum(_dynamic_chamfer_core(pred_v, tgt_v, pred_counts, tgt_counts, scale=sc) for sc in scales) / len(scales)
    else:
        return _dynamic_chamfer_core(pred_v, tgt_v, pred_counts, tgt_counts) + _dynamic_normal_core(pred_n, tgt_n, pred_counts, tgt_counts)


# --------------------------
# (5) 내부 함수
# --------------------------
def _dynamic_chamfer_core(pred_v, tgt_v, pred_counts, tgt_counts, scale=1.0):
    B, MP, _= pred_v.shape
    total= 0.
    for i in range(B):
        pc= pred_counts[i].item()
        tc= tgt_counts[i].item()
        pc= max(1, min(pc, MP))
        tc= max(1, min(tc, MP))

        p_v= pred_v[i,:pc]
        t_v= tgt_v[i,:tc]

        if scale<1:
            new_pc= int(pc*scale)
            new_tc= int(tc*scale)
            new_pc= max(1,new_pc)
            new_tc= max(1,new_tc)
            p_v= p_v[:new_pc]
            t_v= t_v[:new_tc]

        dist= torch.cdist(p_v.unsqueeze(0), t_v.unsqueeze(0)) # (1,new_pc,new_tc)
        minA,_= dist.min(dim=2)
        minB,_= dist.min(dim=1)
        c_loss= minA.mean() + minB.mean()
        total+= c_loss
    return total/B

def _dynamic_normal_core(pred_n, tgt_n, pred_counts, tgt_counts):
    B,MP,_= pred_n.shape
    total=0.
    for i in range(B):
        pc= pred_counts[i].item()
        tc= tgt_counts[i].item()
        pc= max(1, min(pc, MP))
        tc= max(1, min(tc, MP))
        p_n= pred_n[i,:pc]
        t_n= tgt_n[i,:tc]
        p_n= F.normalize(p_n, dim=-1)
        t_n= F.normalize(t_n, dim=-1)

        dot= torch.einsum("ik,jk->ij", p_n, t_n)
        normal_match= dot.mean()
        normal_loss= 1- normal_match
        total+= normal_loss
    return total/B
