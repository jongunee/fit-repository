import math
from flask import Flask, request, render_template, url_for, redirect
import os
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
import shutil
import sys
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "resources" / "smplx"))

from models.a2s_predictor import A2SPredictor
from labels import LABELS
import renderer

app = Flask(__name__)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Use environment variable for CUDA device, default to "0"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE", "0")

# Base directories (configurable through environment variables)
BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(project_root / "static" / "output"))
EXTRA_DIR = os.getenv("EXTRA_DIR", str(project_root / "resources" / "drapenet" / "extra-data"))
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", str(project_root / "resources" / "checkpoints"))
SMPL_MODEL_DIR = os.getenv("SMPL_MODEL_DIR", str(project_root / "resources" / "smplx"))

# Create necessary directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 0) HOME
# ---------------------------
@app.route('/')
def home():
    """
    홈 화면에서 <username>/fit_records/*.json을 찾아서,
    화면에는 'category (fit_type)' 형식으로 표시.
    """
    base_path = Path(BASE_OUTPUT_DIR)
    user_data = []
    if base_path.exists():
        for user_dir in base_path.iterdir():
            if user_dir.is_dir():
                username = user_dir.name
                fit_dir = user_dir / "fit_records"
                fits = []
                if fit_dir.exists():
                    for fit_file in fit_dir.glob("*.json"):
                        fit_name_no_ext = fit_file.stem  # 파일 이름 (확장자 제외)
                        category, fit_type = fit_name_no_ext.split("_", 1)
                        fits.append({
                            "display_name": f"{category} ({fit_type})",
                            "filename_no_ext": fit_name_no_ext
                        })
                user_data.append({
                    "username": username,
                    "fits": fits
                })
    return render_template('home.html', user_data=user_data)

# ---------------------------
# 0-1) View Fit
# ---------------------------
@app.route('/view_fit/<username>/<fit_name>')
def view_fit(username, fit_name):
    """
    home.html에서 fit_name='regular' 라고 클릭 → 여기로 옴
    실제 파일은 'regular.json'
    """
    user_root = Path(BASE_OUTPUT_DIR) / username
    fit_records_dir = user_root / "fit_records"

    # 실제 JSON 파일 경로: 'regular.json'
    fit_json_file = fit_records_dir / f"{fit_name}.json"
    if not fit_json_file.exists():
        return f"No fit record found: {fit_json_file}", 404

    with open(fit_json_file, 'r') as f:
        record_data = json.load(f)

    combined_mesh_path = record_data.get('combined_mesh_path')
    if combined_mesh_path:
        # static/ 이하 경로로 변환
        rel_path = os.path.relpath(combined_mesh_path, 'static')  
        mesh_url = url_for('static', filename=rel_path)
        print(f"Mesh URL: {mesh_url}")  # 디버깅용

        return render_template('view_fit.html', record_data=record_data, mesh_url=mesh_url)
    else:
        return render_template('view_fit.html', record_data=record_data, mesh_url=None)

# ---------------------------
# 1) Generate SMPL
# ---------------------------
@app.route('/generate_smpl', methods=['GET', 'POST'])
def generate_smpl():
    if request.method == 'POST':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gender = "female"
        ai_model_gender = "female"

        additional_info = request.form.get('additional_info')
        linguistic_info = request.form.get('linguistic_info')
        
        if not additional_info and not linguistic_info:
            config = '02a_hw2s'
        elif not additional_info and linguistic_info:
            config = '02b_ahw2s'
        elif additional_info and not linguistic_info:
            config = '05a_hwcwh2s'
        else:
            config = '05b_ahwcwh2s'

        name = request.form.get('name', 'anonymous')

        user_root = os.path.join(BASE_OUTPUT_DIR, name)
        os.makedirs(user_root, exist_ok=True)
        smpl_storage_dir = os.path.join(user_root, "stored_smpls")
        os.makedirs(smpl_storage_dir, exist_ok=True)

        height = float(request.form.get('height', 170))
        weight = float(request.form.get('weight', 65))
        chest  = float(request.form.get('chest', 100))
        waist  = float(request.form.get('waist', 80))
        hips   = float(request.form.get('hips', 90))

        rating_labels = LABELS[gender]
        rating = np.array([float(request.form.get(label, 0)) for label in rating_labels])

        if additional_info:
            chest = float(request.form.get('chest', 100))
            waist = float(request.form.get('waist', 80))
            hips  = float(request.form.get('hips', 90))
        else:
            chest, waist, hips = 0, 0, 0

        predictor = A2SPredictor(
            ds_gender=gender,
            config=config,
            model_gender=ai_model_gender,
            height=height,
            weight=weight,
            chest=chest,
            waist=waist,
            hips=hips,
            rating=rating,
            device=device,
        )
        smpl_obj_path, smpl_vertices, smpl_faces, betas = predictor.render()

        smpl_obj_new = os.path.join(smpl_storage_dir, f'{name}.obj')
        os.rename(smpl_obj_path, smpl_obj_new)

        params = {
            'name': name,
            'gender': gender,
            'config': config,
            'ai_model_gender': ai_model_gender,
            'height': height,
            'weight': weight,
            'chest': chest,
            'waist': waist,
            'hips': hips,
            'rating': rating.tolist(),
            'betas': betas.detach().cpu().numpy().tolist()
        }
        smpl_json_new = os.path.join(smpl_storage_dir, f'{name}.json')
        with open(smpl_json_new, 'w') as f:
            json.dump(params, f)

        return redirect(url_for('generate_smpl'))

    else:
        # 기존 SMPL들
        smpl_models = []
        base_path = Path(BASE_OUTPUT_DIR)
        if base_path.exists():
            for user_dir in base_path.iterdir():
                if user_dir.is_dir():
                    smpl_path = user_dir / "stored_smpls"
                    for obj_file in smpl_path.glob("*.obj"):
                        user_name = user_dir.name
                        smpl_name = obj_file.stem
                        json_path = smpl_path / f"{smpl_name}.json"
                        if json_path.exists():
                            with open(json_path, 'r') as f:
                                params = json.load(f)
                        else:
                            params = {}
                        smpl_models.append({'name': smpl_name, 'params': params, 'user_dir': user_name})
        return render_template('generate_smpl.html', smpl_models=smpl_models, LABELS=LABELS)

@app.route('/view_smpl/<smpl_name>')
def view_smpl(smpl_name):
    """
    smpl_name에 해당하는 SMPL json + OBJ 로드 후 시각화
    예) /view_smpl/alice -> alice.json, alice.obj
    """
    user_root = os.path.join(BASE_OUTPUT_DIR, smpl_name)
    smpl_storage_dir = os.path.join(user_root, "stored_smpls")

    # 1) JSON 파일 위치: stored_smpls/<smpl_name>.json
    smpl_json_path = os.path.join(smpl_storage_dir, f"{smpl_name}.json")
    if not os.path.exists(smpl_json_path):
        return f"SMPL JSON not found for {smpl_name}", 404

    # 2) JSON 로드
    with open(smpl_json_path, 'r', encoding='utf-8') as f:
        smpl_data = json.load(f)

    # 3) OBJ 파일 위치: stored_smpls/<smpl_name>.obj
    smpl_obj_path = os.path.join(smpl_storage_dir, f"{smpl_name}.obj")
    if not os.path.exists(smpl_obj_path):
        # OBJ가 없으면 mesh_url=None으로 넘기고, 템플릿에서 "No 3D mesh found" 표시
        mesh_url = None
    else:
        # Flask 정적 파일 경로로 변환
        # 예: /static/output/alice/stored_smpls/alice.obj
        rel_path = os.path.relpath(smpl_obj_path, 'static')
        mesh_url = url_for('static', filename=rel_path)

    # 4) 템플릿으로 전달
    # smpl_data = JSON 내용 (ex: height, weight, etc.)
    # mesh_url  = OBJ 경로 (None이면 3D 표시 안 함)
    return render_template('view_smpl.html', smpl_data=smpl_data, mesh_url=mesh_url)


# ---------------------------
# 2) Fitting Room
# ---------------------------
@app.route('/fitting_room', methods=['GET', 'POST'])
def fitting_room():
    """
    1단계: 의류 생성 + SMPL 합성 (preview.obj)
    2단계: Fit 최종 저장 (fit_type.json)
    """
    base_path = Path(BASE_OUTPUT_DIR)
    smpl_list = []
    if base_path.exists():
        for user_dir in base_path.iterdir():
            smpl_path = user_dir / "stored_smpls"
            if smpl_path.exists():
                for obj_file in smpl_path.glob("*.obj"):
                    smpl_list.append(obj_file.stem)

    if request.method == 'GET':
        return render_template('fitting_room.html', smpl_list=smpl_list)

    else:
        fit_type = request.form.get('fit_type', None)

        if fit_type:
            # 2단계
            smpl_name     = request.form.get('smpl_name', 'unknown')
            brand         = request.form.get('brand', 'unknown')
            clothing_type = request.form.get('clothing_type', 'tshirts')
            size          = request.form.get('size', 'M')

            shoulder_width = float(request.form.get('shoulder_width', 0))
            total_length   = float(request.form.get('total_length', 0))
            sleeve_length  = float(request.form.get('sleeve_length', 0))
            chest_width    = float(request.form.get('chest_width', 0))

            pose_idx = int(request.form.get('pose', 0))  # 기본값 A = 0

            # fit_type이 "etc: ..."
            if fit_type.startswith("etc:"):
                raw_etc = fit_type.replace("etc:", "").strip()
                safe_etc = raw_etc.replace(" ", "_").replace(":", "")
                fit_type = f"etc_{safe_etc}"

            user_root = os.path.join(BASE_OUTPUT_DIR, smpl_name)
            fit_records_dir = os.path.join(user_root, "fit_records")
            os.makedirs(fit_records_dir, exist_ok=True)

            base_name = f"{clothing_type}_{fit_type}"
            fit_json_path = os.path.join(fit_records_dir, f"{base_name}.json")
            fit_obj_path = os.path.join(fit_records_dir, f"{base_name}.obj")
            preview_obj_path = os.path.join(fit_records_dir, "preview.obj")  # 기존 preview.obj 경로

            # preview.obj -> base_name.obj (복사 또는 이름 변경)
            if os.path.exists(preview_obj_path):
                shutil.copy(preview_obj_path, fit_obj_path)  # 복사
                # os.rename(preview_obj_path, fit_obj_path)  # 이름 변경 (원하면 복사 대신 사용)

            # combined_mesh_path에 OBJ 파일 경로 저장
            combined_mesh_path = fit_obj_path

            record_data = {
                'name': smpl_name,
                'brand': brand,
                'clothing_type': clothing_type,
                'size': size,
                'shoulder_width': shoulder_width,
                'total_length': total_length,
                'sleeve_length': sleeve_length,
                'chest_width': chest_width,
                'fit_type': fit_type,
                'combined_mesh_path': combined_mesh_path,
                'pose_idx': pose_idx
            }
            with open(fit_json_path, 'w') as f:
                json.dump(record_data, f, indent=4)

            return render_template('fitting_room.html', fit_saved=True, record_data=record_data)

        else:
            # 1단계
            smpl_name     = request.form.get('smpl_name', 'unknown')
            brand         = request.form.get('brand', 'unknown')
            clothing_type = request.form.get('clothing_type', 'tshirts')
            size          = request.form.get('size', 'M')

            shoulder_width = float(request.form.get('shoulder_width', 40)) / 100.0
            total_length   = float(request.form.get('total_length', 70))  / 100.0
            sleeve_length  = float(request.form.get('sleeve_length', 20)) / 100.0
            chest_width    = float(request.form.get('chest_width', 50))   / 100.0

            pose_idx = int(request.form.get('pose', 0))  # 기본값 A = 0

            user_root = os.path.join(BASE_OUTPUT_DIR, smpl_name)
            registered_clothes = os.path.join(user_root, "registered_clothes")
            fit_records_dir    = os.path.join(user_root, "fit_records")
            stored_smpls_dir   = os.path.join(user_root, "stored_smpls")
            os.makedirs(registered_clothes, exist_ok=True)
            os.makedirs(fit_records_dir,    exist_ok=True)
            os.makedirs(stored_smpls_dir,   exist_ok=True)

            serial_number = f"{smpl_name}_{brand}"
            measurements = [[shoulder_width, total_length, sleeve_length, chest_width]]
            sizes = [size]
            categories = [clothing_type]

            result = renderer.generate_mesh(
                serial_number=serial_number,
                measurements=measurements,
                sizes=sizes,
                categories=categories,
                output_dir=Path(registered_clothes)
            )
            if not result["obj_files"]:
                return "No mesh generated. Check logs.", 500

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            smpl_obj_path = os.path.join(stored_smpls_dir, f'{smpl_name}.obj')
            if not os.path.exists(smpl_obj_path):
                return f"SMPL '{smpl_name}' not found in {stored_smpls_dir}.", 400

            smpl_json_path = os.path.join(stored_smpls_dir, f'{smpl_name}.json')
            with open(smpl_json_path, 'r') as f:
                smpl_params = json.load(f)

            betas = torch.tensor(smpl_params['betas'], dtype=torch.float32).to(device)
            smpl_mesh = trimesh.load(smpl_obj_path, process=False)
            smpl_vertices = torch.tensor(smpl_mesh.vertices, dtype=torch.float32).to(device)
            smpl_faces    = torch.tensor(smpl_mesh.faces.astype(np.int32), dtype=torch.int32).to(device)

            clothing_dir = os.path.join(registered_clothes, serial_number)
            latent_codes_file = os.path.join(clothing_dir, 'latent_codes', f'{serial_number}_latent_codes.pt')
            latent_codes_list = torch.load(latent_codes_file, map_location=device)
            latent_codes_dict = { item['size']: item['latent_code'] for item in latent_codes_list }
            if size not in latent_codes_dict:
                return f"Size '{size}' not found in latent codes.", 400

            latent_code_top = latent_codes_dict[size]

            predictor = A2SPredictor()
            poses = torch.load(os.path.join(EXTRA_DIR, 'pose-sample.pt'))
            if pose_idx == 0:
                pose = torch.zeros(1, 72).to(device)
            else:
                pose = poses[[pose_idx]].to(device)
                
            betas_tensor = betas.float().to(device)

            # preview.obj
            combined_mesh_path = predictor.add_clothing(
                smpl_vertices, smpl_faces, pose,
                checkpoints_dir=CHECKPOINTS_DIR,
                extra_dir=EXTRA_DIR,
                smpl_model_dir=SMPL_MODEL_DIR,
                latent_codes_file=latent_codes_file,
                output_folder=fit_records_dir,
                device=device,
                betas=betas_tensor, latent_code_top=latent_code_top,
                bottom_idx=15, resolution=256,
                fit_name="preview"
            )

            rel_path = os.path.relpath(combined_mesh_path, 'static')
            combined_mesh_url = url_for('static', filename=rel_path)

            return render_template(
                'fitting_room.html',
                smpl_list=smpl_list,
                smpl_name=smpl_name,
                brand=brand,
                clothing_type=clothing_type,
                size=size,
                shoulder_width=shoulder_width * 100.0,
                total_length=total_length  * 100.0,
                sleeve_length=sleeve_length * 100.0,
                chest_width=chest_width    * 100.0,
                combined_mesh_url=combined_mesh_url,
                combined_mesh_path=combined_mesh_path
            )

if __name__ == '__main__':
    app.run(port=5001, debug=True)
