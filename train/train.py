import os
import argparse
import logging
import time
import json
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import trimesh

# -----------------------------------
# "train_experiment_dynamic.py"에서 임포트
# -----------------------------------
from train_experiment_dynamic import (
    GarmentDataset,
    collate_fn,
    GarmentGeneratorWithExternalCount,
    dynamic_chamfer_loss
)


def train_one_epoch_dynamic(model, optimizer, loader, device, loss_mode, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    for (inputs, tgt_verts, tgt_norms, tgt_counts) in loader:
        inputs = inputs.to(device)
        tgt_verts = tgt_verts.to(device)
        tgt_norms = tgt_norms.to(device)
        tgt_counts = tgt_counts.to(device)

        optimizer.zero_grad()
        # forward => (pred_v, pred_n, pred_c)
        pred_v, pred_n, pred_c = model(inputs)

        # Chamfer Loss
        loss = dynamic_chamfer_loss(
            pred_v, pred_n, pred_c,
            tgt_verts, tgt_norms, tgt_counts,
            mode=loss_mode  # 예: 'chamfer_only'
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[Epoch {epoch}/{num_epochs}] Train Loss= {avg_loss:.4f}")
    return avg_loss


def val_one_epoch_dynamic(model, loader, device, loss_mode, epoch, num_epochs):
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for (inputs, tgt_verts, tgt_norms, tgt_counts) in loader:
            inputs = inputs.to(device)
            tgt_verts = tgt_verts.to(device)
            tgt_norms = tgt_norms.to(device)
            tgt_counts = tgt_counts.to(device)

            pred_v, pred_n, pred_c = model(inputs)

            val_loss = dynamic_chamfer_loss(
                pred_v, pred_n, pred_c,
                tgt_verts, tgt_norms, tgt_counts,
                mode=loss_mode
            )
            total_val += val_loss.item()

    avg_val = total_val / len(loader)
    print(f"[Epoch {epoch}/{num_epochs}] Val   Loss= {avg_val:.4f}")
    return avg_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_model_path', type=str, default='./checkpoints/pointcount_prediction_model.pth',
                        help='Trained pointcount regressor model path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for final training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00077904708, help='Learning rate')
    parser.add_argument('--loss_mode', type=str, default='chamfer_only', 
                        choices=['chamfer_only','chamfer_with_normals','multiscale'],
                        help='Loss mode for dynamic chamfer loss')
    parser.add_argument('--big_hidden', type=int, default=64, help='Hidden dimension for BigPredictor')
    parser.add_argument('--use_skip', action='store_true', help='If passed, use skip connection (else False).')

    args = parser.parse_args()

    # 출력 폴더 설정
    current_time = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'final_dynamic_out/{current_time}'
    os.makedirs(output_dir, exist_ok=True)

    # 로그 설정
    logging.basicConfig(
        filename=os.path.join(output_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Final Training] device= {device}")
    logging.info(f"[Final Training] device= {device}")

    # 1) 외부 count 모델 로드
    from train_pointcount_regressor import PointCountRegressor
    external_count_model = PointCountRegressor(input_dim=7, hidden_dim=128).to(device)
    if os.path.isfile(args.count_model_path):
        external_count_model.load_state_dict(torch.load(args.count_model_path, map_location=device))
        external_count_model.eval()
        for p in external_count_model.parameters():
            p.requires_grad = False
        print(f"Loaded external count model => {args.count_model_path}")
        logging.info(f"Loaded external count model => {args.count_model_path}")
    else:
        print(f"Error: not found => {args.count_model_path}")
        logging.info(f"Error: not found => {args.count_model_path}")
        return

    # 2) 데이터셋 준비
    measurements_dir = './data/measurements_with_counts.json'
    dataset_dir = './data/final_datasets'

    # torch.manual_seed(42)
    # np.random.seed(42)

    dataset = GarmentDataset(
        measurements_dir=measurements_dir,
        dataset_dir=dataset_dir
    )

    val_split = 0.1
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 학습 및 검증 데이터 목록 저장
    train_samples = [dataset.measurements[idx] for idx in train_ds.indices]
    val_samples   = [dataset.measurements[idx] for idx in val_ds.indices]

    with open(os.path.join(output_dir, "train_samples.json"), "w") as f:
        json.dump(train_samples, f, indent=4)

    with open(os.path.join(output_dir, "val_samples.json"), "w") as f:
        json.dump(val_samples, f, indent=4)

    logging.info(f"Train samples saved: {len(train_samples)}")
    logging.info(f"Val samples saved: {len(val_samples)}")

    # 3) 모델 생성
    model = GarmentGeneratorWithExternalCount(
        external_count_model=external_count_model,
        big_hidden=args.big_hidden,
        max_num_points=dataset.max_num_points,
        use_skip=args.use_skip,           # 구현에 따라 무시될 수도 있음
        activation_type='silu'            # 최종 실험 결과 상 silu 사용 (필요시 하드코딩)
    ).to(device)

    # 옵티마이저 (Adam 고정, 필요시 AdamW/RMSprop로 변경 가능)
    optimizer = optim.Adam(model.big_predictor.parameters(), lr=args.learning_rate)

    # 4) 학습 루프
    train_losses = []
    val_losses   = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss = train_one_epoch_dynamic(
            model, optimizer, train_loader,
            device, args.loss_mode,
            epoch, args.epochs
        )
        # Validation
        val_loss = val_one_epoch_dynamic(
            model, val_loader,
            device, args.loss_mode,
            epoch, args.epochs
        )

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        # Best 모델 갱신 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(output_dir, 'best_model.pth'))

        # 매 50에폭마다 체크포인트 저장
        if epoch % 50 == 0:
            ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Epoch={epoch}, saved => {ckpt_path}")
            logging.info(f"[Checkpoint] Epoch={epoch}, saved => {ckpt_path}")

        print(f"Epoch [{epoch}/{args.epochs}] => Train: {tr_loss:.4f}, Val: {val_loss:.4f}  (best={best_val_loss:.4f} at ep={best_epoch})")

        
    # 최종 best 모델 저장
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(output_dir, 'best_model_final.pth'))
        print(f"[Final] Best model saved at epoch= {best_epoch}, val_loss= {best_val_loss:.4f}")
        logging.info(f"[Final] Best model saved => epoch={best_epoch}, val={best_val_loss:.4f}")

    # 5) 학습 곡선 그래프 저장
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, args.epochs + 1), val_losses,   label='Val Loss',   color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train / Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"[Plot] Loss curve saved => {loss_plot_path}")
    logging.info(f"[Plot] Loss curve saved => {loss_plot_path}")

    # 6) 예시 메쉬 출력 (마지막 Best 모델 기준)
    with torch.no_grad():
        example_in = torch.tensor([0.3, 0.3, 0.1, 0.5, 0, 1, 0], dtype=torch.float32).unsqueeze(0).to(device)
        pred_v, pred_n, pred_c = model(example_in)
        c = pred_c[0].item()
        c = max(1, min(c, dataset.max_num_points))
        final_v = pred_v[0, :c].cpu().numpy()

        tri = trimesh.Trimesh(vertices=final_v)
        out_obj = os.path.join(output_dir, "example_final.obj")
        tri.export(out_obj)
        print(f"Example mesh saved => {out_obj}")
        logging.info(f"Example mesh saved => {out_obj}")


if __name__ == "__main__":
    main()
