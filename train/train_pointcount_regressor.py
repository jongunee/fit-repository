# train_pointcount_regressor.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import time
import logging

# 1) Dataset
class PointCountDataset(Dataset):
    """
    - JSON 항목: 
      {
        'file_name': ...,
        'category': 'sleeveless'|'tshirts'|'opened-shirts',
        'shoulder_width': float,
        'total_length': float,
        'sleeve_length': float,
        'chest_width': float,
        'pointcloud_num': int
      }
    - 입력: (category_onehot, shoulder, total, sleeve, chest)
    - 출력: pointcloud_num
    """
    def __init__(self, json_path):
        with open(json_path,'r') as f:
            self.data = json.load(f)
        self.category_map = {'sleeveless':0, 'tshirts':1, 'opened-shirts':2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        cat_str = d['category']
        cat_onehot = np.zeros(3,dtype=np.float32)
        cat_idx = self.category_map.get(cat_str,0)
        cat_onehot[cat_idx] = 1

        x_arr = np.array([
            d['shoulder_width'],
            d['total_length'],
            d['sleeve_length'],
            d['chest_width']
        ], dtype=np.float32)
        # concatenate
        input_vec = np.concatenate([x_arr, cat_onehot], axis=0)  # shape=(7,)

        y_val = d['pointcloud_num']  # int
        # to float
        y_val = float(y_val)

        return torch.tensor(input_vec, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


# 2) MLP Model
class PointCountRegressor(nn.Module):
    """
    입력: (7,)  => (hidden...) => 스칼라( pointcount )
    """
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        # x: (batch,7)
        out = self.net(x)  # shape: (batch, 1)
        return out.squeeze(1)  # (batch,)

# 3) collate_fn (기본)
def collate_fn(batch):
    xs = []
    ys = []
    for (x,y) in batch:
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs, dim=0)
    Y = torch.stack(ys, dim=0)
    return X,Y

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss=0.
    crit = nn.L1Loss()
    for X,Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        pred = model(X)  # (batch,)
        loss = crit(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss/len(loader)
    return avg

def val_one_epoch(model, loader, device):
    model.eval()
    total_loss=0.
    crit = nn.L1Loss()
    with torch.no_grad():
        for X,Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = crit(pred, Y)
            total_loss+= loss.item()
    avg= total_loss/len(loader)
    return avg

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('--json_path',type=str, default='./data/measurements_with_counts.json')
    parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args= parser.parse_args()

    out_dir= f'pointcount_output'
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(out_dir,'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device= {device}")

    dataset= PointCountDataset(args.json_path)
    val_size= int(0.1* len(dataset))
    train_size= len(dataset)- val_size
    train_ds, val_ds= random_split(dataset,[train_size,val_size])
    train_loader= DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader=   DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model= PointCountRegressor(input_dim=7, hidden_dim=args.hidden_dim).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val= 1e12
    best_epoch=0

    for epoch in range(1, args.epochs+1):
        t_loss= train_one_epoch(model, train_loader, optimizer, device)
        v_loss= val_one_epoch(model, val_loader, device)

        logging.info(f"Epoch {epoch}/{args.epochs} => TrainLoss= {t_loss:.4f}, ValLoss= {v_loss:.4f}")
        print(f"Epoch {epoch}/{args.epochs} => TrainLoss= {t_loss:.4f}, ValLoss= {v_loss:.4f}")

        if v_loss< best_val:
            best_val= v_loss
            best_epoch= epoch
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    logging.info(f"Finished => BestVal= {best_val:.4f} at epoch= {best_epoch}")
    print(f"Finished => BestVal= {best_val:.4f} at epoch= {best_epoch}")


if __name__=="__main__":
    main()
