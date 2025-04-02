import os
import trimesh
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

# 데이터셋 경로
measurements_file = './measurements.json'  # measurements.json 경로
dataset_dir = './final_datasets'  # obj 파일이 있는 폴더 경로
output_dir = './output/distribution_analysis'  # 결과 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)

def load_obj_point_count(file_path):
    """
    .obj 파일을 로드하고 포인트 개수를 반환
    """
    mesh = trimesh.load(file_path, force='mesh')
    return len(mesh.vertices)

def analyze_point_distribution(measurements_file, dataset_dir):
    """
    포인트 개수를 계산하고, 전체 및 카테고리별 분포를 분석
    """
    # measurements.json 로드
    with open(measurements_file, 'r') as f:
        measurements = json.load(f)

    # 카테고리별 포인트 개수 집계
    category_point_counts = defaultdict(list)

    for item in measurements:
        file_name = item['file_name']
        category = item['category']
        obj_path = os.path.join(dataset_dir, file_name)
        if os.path.isfile(obj_path):
            point_count = load_obj_point_count(obj_path)
            category_point_counts[category].append(point_count)

    # 전체 데이터셋의 포인트 개수 분포
    all_point_counts = [count for counts in category_point_counts.values() for count in counts]

    return all_point_counts, category_point_counts

def save_point_distribution(all_point_counts, category_point_counts, output_dir):
    """
    전체 및 카테고리별 포인트 분포를 이미지로 저장
    """
    # 전체 데이터셋 분포
    overall_hist_path = os.path.join(output_dir, 'overall_point_distribution.png')
    plt.figure(figsize=(10, 6))
    plt.hist(all_point_counts, bins=30, alpha=0.7, label='전체')
    plt.xlabel('points')
    plt.ylabel('num')
    plt.title('all categories')
    plt.legend()
    plt.savefig(overall_hist_path)
    plt.close()

    # 카테고리별 분포
    category_hist_paths = {}
    for category, counts in category_point_counts.items():
        category_hist_path = os.path.join(output_dir, f'{category}_point_distribution.png')
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=30, alpha=0.7, label=f'{category}')
        plt.xlabel('points')
        plt.ylabel('num')
        plt.title(f'{category}')
        plt.legend()
        plt.savefig(category_hist_path)
        plt.close()
        category_hist_paths[category] = category_hist_path

    return overall_hist_path, category_hist_paths

def summarize_distribution(all_point_counts, category_point_counts, output_dir):
    """
    포인트 개수 분포의 통계 요약 및 저장
    """
    def get_stats(counts):
        return {
            "mean": np.mean(counts),
            "median": np.median(counts),
            "min": np.min(counts),
            "max": np.max(counts),
            "std": np.std(counts)
        }

    overall_stats = get_stats(all_point_counts)
    category_stats = {category: get_stats(counts) for category, counts in category_point_counts.items()}

    # 저장을 위해 DataFrame 생성
    overall_df = pd.DataFrame([overall_stats], index=['all'])
    category_df = pd.DataFrame(category_stats).T
    category_df.index.name = 'category'

    # CSV 저장
    overall_csv_path = os.path.join(output_dir, 'overall_distribution_stats.csv')
    category_csv_path = os.path.join(output_dir, 'category_distribution_stats.csv')
    overall_df.to_csv(overall_csv_path)
    category_df.to_csv(category_csv_path)

    return overall_csv_path, category_csv_path

# 실행
all_point_counts, category_point_counts = analyze_point_distribution(measurements_file, dataset_dir)
overall_hist_path, category_hist_paths = save_point_distribution(all_point_counts, category_point_counts, output_dir)
overall_csv_path, category_csv_path = summarize_distribution(all_point_counts, category_point_counts, output_dir)

# 결과 요약 출력
print(f"Overall histogram saved at: {overall_hist_path}")
print(f"Category histograms saved at: {category_hist_paths}")
print(f"Overall stats saved at: {overall_csv_path}")
print(f"Category stats saved at: {category_csv_path}")
