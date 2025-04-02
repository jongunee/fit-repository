import os
import os.path as osp
import trimesh
import torch
import numpy as np
import open3d as o3d
import pickle
import smplx
from smplx import build_layer
from loguru import logger
from pathlib import Path
from resources.shapy.attributes.attributes.utils.renderer import Renderer
from resources.shapy.attributes.attributes.attributes_betas.build import MODEL_DICT
from resources.shapy.attributes.attributes.utils.config import default_conf
from omegaconf import OmegaConf, DictConfig
from labels import LABELS
from resources.drapenet.smpl_pytorch.body_models import SMPL
from resources.drapenet.utils_drape import draping, load_udf, load_lbs, reconstruct

import sys
# 경로 추가 및 필요한 모듈 불러오기
from resources.smplx.transfer_model.utils.def_transfer import read_deformation_transfer
from resources.smplx.transfer_model.utils.o3d_utils import np_mesh_to_o3d
from resources.smplx.transfer_model.data.build import build_dataloader
from resources.smplx.transfer_model.transfer_model import run_fitting

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class A2SPredictor:
    def __init__(self, ds_gender='female', config='00_a2s', model_gender='neutral',
                 model_type='smplx', height: float = 170.0, weight: float = 65.0,
                 chest: float = 100.0, waist: float = 80.0, hips: float = 90.0, rating=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ds_gender = ds_gender
        self.model_gender = model_gender
        self.model_type = model_type
        self.eval_bs = 1

        if rating is None:
            rating_np = np.empty((0, len(LABELS[ds_gender])))
        else:
            rating_np = np.atleast_2d(rating)

        self.db = {
            'height_gt': np.array([height / 100]),
            'weight_gt': np.array([weight]),
            'chest': np.array([chest / 100]),
            'waist': np.array([waist / 100]),
            'hips': np.array([hips / 100]),
            'rating': rating_np
        }

        # Get project root and set paths
        project_root = Path(__file__).parent.parent.parent
        self.model_path = str(project_root / 'resources' / 'smplx' / 'models')
        self.checkpoint_path = str(project_root / 'resources' / 'shapy' / 'data' / 'trained_models' / 'a2b' / 
                                 f'caesar-{ds_gender}_smplx-{model_gender}-10betas' / 'poynomial' / f'{config}.yaml' / 'last.ckpt')
        self.config = str(project_root / 'resources' / 'shapy' / 'attributes' / 'configs' / 'a2s_variations_polynomial' / f'{config}.yaml')
        self.smpl_model_path = self.model_path
        self.yaml_path = str(project_root / 'resources' / 'smplx2smpl.yaml')
        self.betas = None

    def render(self):
        os.makedirs('./static/output', exist_ok=True)  # static 폴더 내에 output 폴더 생성
        output_dir = './static/output'
        # output_dir = './stored_smpls'
        network_type = 'a2b'
        cfg = OmegaConf.load(self.yaml_path)

        renderer = Renderer(is_registration=False)
        device = self.device

        if not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            sys.exit(3)
        
        smplx_model = smplx.create(self.model_path, gender=self.model_gender, num_betas=10, model_type=self.model_type).to(device)
        loaded_model = MODEL_DICT[network_type].load_from_checkpoint(
            checkpoint_path=self.checkpoint_path)

        test_input, _ = loaded_model.create_input_feature_vec(self.db)
        test_input = loaded_model.to_whw2s(test_input, None) if loaded_model.whw2s_model else test_input
        prediction = loaded_model.a2b.predict(test_input)

        data_folder = cfg.datasets.mesh_folder.data_folder
        smplx_output_paths = []

        for idx, betas in enumerate(prediction):
            body = smplx_model(betas=betas.unsqueeze(0).to(device))
            shaped_vertices = body.vertices.detach().cpu().numpy()[0]
            smplx_output_path = osp.join(data_folder, f'smplx_output_{idx}.obj')
            smplx_mesh = trimesh.Trimesh(shaped_vertices, smplx_model.faces)
            smplx_mesh.export(file_obj=smplx_output_path)
            smplx_output_paths.append(smplx_output_path)
            logger.info(f"SMPLX output saved to {smplx_output_path}")

        model_path = cfg.body_model.folder
        body_model = build_layer(model_path, **cfg.body_model)
        logger.info(body_model)
        body_model = body_model.to(device=device)
    
        # Prepare deformation transfer matrix
        deformation_transfer_path = cfg.deformation_transfer_path
        def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)
        logger.info(def_matrix)

        # Data loader 준비
        data_obj_dict = build_dataloader(cfg)
        logger.info(data_obj_dict)

        dataloader = data_obj_dict['dataloader']

        if not os.listdir(data_folder):
            logger.error(f"Data folder '{data_folder}' is empty. Please check the data folder path and ensure there are data files available.")
            raise ValueError(f"Data folder '{data_folder}' is empty. Cannot proceed with model fitting and rendering.")

        if not any(dataloader):
            logger.error("Dataloader is empty. Please check the data folder path and ensure there are data files available.")
            raise ValueError("Data loader is empty. Cannot proceed with model fitting and rendering.")
        
        for batch in dataloader:
            logger.info(f"Processing batch: {batch}")
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            var_dict = run_fitting(cfg, batch, body_model, def_matrix)
            paths = batch['paths']

            for ii, path in enumerate(paths):
                _, fname = osp.split(path)

                output_path = osp.join(
                    output_dir, f'{osp.splitext(fname)[0]}.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump(var_dict, f)

                output_path = osp.join(
                    output_dir, f'{osp.splitext(fname)[0]}.obj')
                mesh = np_mesh_to_o3d(
                    var_dict['vertices'][ii], var_dict['faces'])
                o3d.io.write_triangle_mesh(output_path, mesh)

        betas = var_dict.get('betas')
        if betas is None:
            raise ValueError("Betas is None after run_fitting")
        logger.info("Rendering process completed successfully.")

        return output_path, var_dict['vertices'], var_dict['faces'], betas.detach().cpu()


    def add_clothing(self, smpl_vertices, smpl_faces, pose, 
                    checkpoints_dir, extra_dir, smpl_model_dir,
                    latent_codes_file, output_folder, device, 
                    betas=None, latent_code_top=None, 
                    bottom_idx=0, resolution=256,
                    fit_name="regular"):
        if betas is None:
            betas = self.betas

        logger.info(f"Betas used in add_clothing: {betas}")
        logger.info(f"Input smpl_vertices in add_clothing: {smpl_vertices}")

        models = load_lbs(checkpoints_dir, device)
        bottom_codes_file = os.path.join(checkpoints_dir, 'bottom_codes.pt')
        coords_encoder, latent_codes_top_list, decoder_top = load_udf(
            checkpoints_dir, latent_codes_file, 'top_udf.pt', device)
        _, latent_codes_bottom, decoder_bottom = load_udf(
            checkpoints_dir, bottom_codes_file, 'bottom_udf.pt', device)

        mesh_top, vertices_top_T, faces_top = reconstruct(
            coords_encoder, decoder_top, latent_code_top,
            udf_max_dist=0.1, resolution=resolution, differentiable=False)
        mesh_bottom, vertices_bottom_T, faces_bottom = reconstruct(
            coords_encoder, decoder_bottom, latent_codes_bottom[[bottom_idx]],
            udf_max_dist=0.1, resolution=resolution, differentiable=False)

        smpl_server = SMPL(model_path=smpl_model_dir, gender='f').to(device)
        tfs_c_inv = torch.FloatTensor(np.load(os.path.join(extra_dir, 'body_info_f.npz'))['tfs_c_inv']).to(device)

        betas = betas.to(device)
        pose = pose.to(device)
        smpl_vertices = smpl_vertices.clone().detach().to(device)
        smpl_faces = smpl_faces.to(device)

        top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh = draping(
            smpl_vertices, smpl_faces, 
            [vertices_top_T.to(device), vertices_bottom_T.to(device)], 
            [faces_top.cpu().numpy(), faces_bottom.cpu().numpy()],
            [latent_code_top.to(device), latent_codes_bottom[[bottom_idx]].to(device)],
            pose, betas, models, smpl_server, tfs_c_inv
        )

        os.makedirs(output_folder, exist_ok=True)

        combined_mesh = trimesh.util.concatenate([body_mesh, top_mesh, bottom_mesh_layer])
        # fit_name이 "regular", "slim" 등이라면 "regular.obj", "slim.obj" 형태로 저장
        top_obj_name = f"{fit_name}_cloth_reconstructed.obj"
        fit_obj_name = f"{fit_name}.obj"
        combined_mesh_path = os.path.join(output_folder, fit_obj_name)
        top_mesh_path = os.path.join(output_folder, top_obj_name)
        combined_mesh.export(combined_mesh_path)
        mesh_top.export(top_mesh_path)

        return combined_mesh_path