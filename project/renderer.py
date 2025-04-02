import sys
import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import open3d as o3d
import logging
from pathlib import Path

from models.dgcnn import Dgcnn  # Encoder 모델
from utils import (
    get_o3d_mesh_from_tensors,
    get_tensor_pcd_from_o3d,
    random_point_sampling,

)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from train.train_experiment_dynamic import (
    # 동적 Dataset/Model
    GarmentGeneratorWithExternalCount,
    # 필요한 함수들 ...
)

# (만약 external_count_model 필요하다면)
from train.train_pointcount_regressor import PointCountRegressor

# --------------------------
# (A) 로그 설정 함수
# --------------------------
def setup_logging(output_dir, serial_number):
    log_file = os.path.join(output_dir, f"{serial_number}.log")
    # 필요 시 logging 초기화 로직
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# --------------------------
# (B) Alpha Shape
# --------------------------
import open3d as o3d
import numpy as np

def alpha_shape_reconstruction(vertices, alpha=0.03, nb_neighbors=50, std_ratio=2.0, voxel_size=None, 
                               smoothing_iterations=5, target_triangles=8000):
    """
    노이즈 제거, 스무딩, 그리고 리메싱(쿼드릭 감쇠)를 포함한 Open3D Alpha Shape 기반 메시 재구성 함수.
    
    Parameters:
      vertices (np.ndarray): 입력 포인트 클라우드 좌표 배열.
      alpha (float): Alpha 값 (너무 작으면 메시 구멍이 많아질 수 있음).
      nb_neighbors (int): 통계적 이상치 제거 시 고려할 이웃 수.
      std_ratio (float): 통계적 이상치 제거 기준 (표준편차 비율).
      voxel_size (float, optional): 보켈 다운샘플링 크기. None이면 다운샘플링하지 않음.
      smoothing_iterations (int): Taubin smoothing 반복 횟수 (0이면 스무딩하지 않음).
      target_triangles (int): 리메싱 후 목표 삼각형 수 (숫자를 늘리면 더 세밀하지만 계산비용 증가).
    
    Returns:
      v (np.ndarray): 재구성된 메시의 정점 배열.
      f (np.ndarray): 재구성된 메시의 삼각형 면 배열.
    """
    if vertices.size == 0:
        print("[alpha_shape] Empty vertex array.")
        return None, None

    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # 옵션: 보켈 다운샘플링
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    # 통계적 이상치 제거
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pts = np.asarray(pcd.points)
    print(f"[alpha_shape] Filtered pts= {len(filtered_pts)}, alpha= {alpha}")

    # Alpha Shape 기반 메시 생성
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    except Exception as e:
        print("[alpha_shape] Error:", e)
        return None, None

    if mesh.is_empty():
        print("[alpha_shape] mesh empty. Maybe alpha is too small or data is degenerate.")
        return None, None

    # 후처리: 불필요한 삼각형과 정점 제거
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if mesh.is_empty():
        print("[alpha_shape] mesh empty after cleanup.")
        return None, None

    # 옵션: 메시 스무딩 (Taubin smoothing)
    if smoothing_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations)
    
    mesh.compute_vertex_normals()

    # 추가: 쿼드릭 감쇠를 통한 리메싱
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    return v, f



# --------------------------
# (C) .obj 저장 함수
# --------------------------
def save_obj(vertices, faces, file_name):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(file_name)


# --------------------------
# (D) Dynamic 모델 로드 함수
# --------------------------
def load_dynamic_model(
    count_model_path: str,
    dynamic_model_path: str,
    max_num_points: int,
    big_hidden: int = 64,
    use_skip: bool = False,
    activation_type: str = 'silu',
    device=None
):
    """
    1) 외부 카운트 모델(PointCountRegressor) 로드
    2) 동적 GarmentGeneratorWithExternalCount 로드
    3) state_dict 불러오기
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (1) 외부 카운트 모델
    count_model = PointCountRegressor(input_dim=7, hidden_dim=128).to(device)
    count_model.load_state_dict(torch.load(count_model_path, map_location=device))
    count_model.eval()
    for p in count_model.parameters():
        p.requires_grad = False

    # (2) 동적 모델
    model = GarmentGeneratorWithExternalCount(
        external_count_model=count_model,
        big_hidden=big_hidden,
        max_num_points=max_num_points,
        use_skip=use_skip,              # 구현부에 따라 무시될 수도 있음
        activation_type=activation_type # 구현부에 따라 무시될 수도 있음
    ).to(device)

    # (3) state_dict 로드
    ckpt = torch.load(dynamic_model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    return model


# --------------------------
# (E) Dynamic 추론 함수
# --------------------------
def dynamic_predict(model, measurement: np.ndarray, category: str, device=None):
    """
    measurement: shape (4,)  => [shoulder_width, total_length, sleeve_length, chest_width]
    category: 'sleeveless' | 'tshirts' | 'opened-shirts'
    returns (verts_np, norms_np) in numpy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    cat_map = {'sleeveless': 0, 'tshirts': 1, 'opened-shirts': 2}
    cat_onehot = np.zeros(3, dtype=np.float32)
    cat_onehot[cat_map[category]] = 1.0

    input_arr = np.concatenate([measurement, cat_onehot], axis=0)  # shape=(7,)
    print("**********input_arr: ", input_arr)
    inp_t = torch.tensor(input_arr, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_v, pred_n, pred_c = model(inp_t)
        c = pred_c[0].item()
        c = max(1, min(c, pred_v.shape[1]))

        verts_np = pred_v[0, :c].cpu().numpy()
        norms_np = pred_n[0, :c].cpu().numpy()

    return verts_np, norms_np, c


# --------------------------
# (F) Latent code 계산
# --------------------------
def load_encoder(encoder_ckpt_path: str, latent_size: int, device=None):
    """
    Drapenet 쪽 Dgcnn 로더
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Dgcnn(latent_size)
    ckpt = torch.load(encoder_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    encoder = encoder.to(device)
    return encoder


def compute_latent_code(verts_np: np.ndarray, faces_np: np.ndarray, pred_count:int, encoder: nn.Module, device=None):
    """
    1) np -> open3d -> sample -> to device
    2) encoder -> latent code
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step1: np-> open3d PointCloud
    if verts_np.shape[0] == 0:
        print("[compute_latent_code] Empty verts => skip.")
        return None

    v_np, t_np = verts_np, faces_np  # 이미 메쉬 데이터를 가지고 있음
    v = torch.tensor(v_np, dtype=torch.float32)
    t = torch.tensor(t_np, dtype=torch.long)
        
    # Open3D 메쉬 생성
    mesh_o3d = get_o3d_mesh_from_tensors(v, t)

    # 포인트 클라우드 샘플링
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=pred_count)
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]
    # Latent code 생성
    num_points_pcd = pred_count  # 포인트 클라우드 포인트 수
    pcds = pcd.unsqueeze(0).to(device)  # 배치 차원 추가
    pcds = random_point_sampling(pcds, num_points_pcd)
    
    # Step3: Encoder
    with torch.no_grad():
        latent_code = encoder(pcds)
    latent_code = latent_code.cpu()
    return latent_code

# --------------------------
# (G) 최종 generate_mesh
# --------------------------
def generate_mesh(
    serial_number: str,
    measurements: list,
    sizes: list,
    categories: list,
    output_dir: str
):
    """
    1) load dynamic model
    2) for each (measurement, size, category):
       - dynamic_predict => (verts, norms)
       - alpha shape => (v_a, f_a)
       - save_obj => .obj
       - compute latent => store
    3) save latent_codes_list => .pt
    """
    # 경로/로거
    output_dir = os.path.join(output_dir, serial_number)
    render_dir = os.path.join(output_dir, "render_output")
    latent_dir = os.path.join(output_dir, "latent_codes")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    logger = setup_logging(output_dir, serial_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[generate_mesh] Using device= {device}")

    # (1) Load dynamic model
    count_model_path = 'resources/checkpoints/pointcount_prediction_model.pth'
    dynamic_model_path= 'resources/checkpoints/best_model.pth'
    max_num_points = 7078  # 예: dataset.max_num_points
    # hyperparams => big_hidden=64, use_skip=False, activation='silu'
    dynamic_model = load_dynamic_model(
        count_model_path,
        dynamic_model_path,
        max_num_points=max_num_points,
        big_hidden=64,
        use_skip=False,
        activation_type='silu',
        device=device
    )
    logger.info("[generate_mesh] Loaded dynamic model successfully.")

    # (2) Load encoder for latent code
    encoder_ckpt_path= 'resources/checkpoints/top_udf.pt'
    latent_size=32
    encoder = load_encoder(encoder_ckpt_path, latent_size, device=device)
    logger.info("[generate_mesh] Loaded encoder for latent code.")

    obj_files = []
    latent_codes_list = []

    for meas, sz, cat in zip(measurements, sizes, categories):
        logger.info(f"--- Generating for meas={meas}, size={sz}, category={cat} ---")

        # 2-1) Dynamic predict
        verts_np, norms_np, pred_count = dynamic_predict(dynamic_model, np.array(meas), cat, device=device)
        if verts_np.shape[0] == 0:
            logger.warning("[generate_mesh] zero predicted points => skip")
            continue

        # 2-2) alpha shape => .obj
        # alpha=0.04
        v_a, f_a = alpha_shape_reconstruction(verts_np)
        if v_a is None or v_a.shape[0] == 0:
            logger.warning("[generate_mesh] alpha shape empty => skip")
            continue

        out_obj = os.path.join(render_dir, f"{serial_number}_{sz}.obj")
        save_obj(v_a, f_a, out_obj)
        logger.info(f"Saved .obj => {out_obj}")
        obj_files.append(out_obj)

        # 2-3) latent code => encoder
        #   (v_a -> open3d-> sample-> encoder -> latent)
        latent_code = compute_latent_code(v_a, f_a, pred_count, encoder, device=device)
        if latent_code is not None:
            # list에 저장
            latent_codes_list.append({
                'size'      : sz,
                'category'  : cat,
                'measurement': meas,
                'latent_code': latent_code
            })
        else:
            logger.warning("[generate_mesh] no latent => skip")

    # (3) all latent => .pt
    latent_file = os.path.join(latent_dir, f"{serial_number}_latent_codes.pt")
    torch.save(latent_codes_list, latent_file)
    logger.info(f"latent codes saved => {latent_file}")

    return {
        'message': f"Generated files for {serial_number}",
        'obj_files': obj_files,
        'latent_codes_file': latent_file
    }