import json

import numpy as np
from sklearn.neighbors import BallTree
import math

from tqdm import tqdm


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except json.JSONDecodeError:
        print(f"错误：无法解析 {file_path} 中的 JSON 数据。请检查文件格式是否正确。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return data

def compute_frequency_differences(freq_matrices):
    diff_matrices = []

    for freq in tqdm(freq_matrices, total=len(freq_matrices), desc="compute_frequency_differences"):
        diff_freq = np.zeros_like(freq)
        diff_freq[0] = freq[0]
        for i in range(1, len(freq)):
            diff_freq[i] = freq[i] - freq[i-1]

        diff_matrices.append(diff_freq)

    return diff_matrices

def calculate_poi_category_frequencies(poi_list, num_cate):
    n = len(poi_list)
    if n == 0:
        return []

    # only neighbor in train datasets
    train_size = int(n * 0.8)
    train_poi = poi_list[:train_size]

    train_coords = np.array([[p[0], p[1]] for p in train_poi])  # (train_size, 2)
    train_categories = np.array([p[2] for p in train_poi])  # (train_size,)

    all_coords = np.array([[p[0], p[1]] for p in poi_list])  # (n, 2)


    train_coords_rad = np.radians(train_coords)
    all_coords_rad = np.radians(all_coords)

    EARTH_RADIUS = 6371000

    tree = BallTree(train_coords_rad, metric='haversine')

    thresholds = np.array([50,100,150,200,250,300,350,400,450,500])
    print(len(thresholds))

    max_radius = 500 / EARTH_RADIUS

    indices_list,dist_rad_list = tree.query_radius(
        all_coords_rad,
        r=max_radius,
        return_distance=True
    )


    results = []
    for i in tqdm(range(n), total=len(indices_list), desc="calculate_poi_category_frequencies"):
        dist_rad = dist_rad_list[i]
        indices = indices_list[i]

        if i < train_size:  # 当前POI属于训练集
            self_mask = (indices != i)
            indices = indices[self_mask]
            dist_rad = dist_rad[self_mask]

        dist_m = dist_rad * EARTH_RADIUS

        if len(indices) == 0:
            results.append(np.zeros((len(thresholds), num_cate), dtype=np.int32))
            continue

        neighbor_categories = train_categories[indices]

        mask_matrix = thresholds.reshape(-1, 1) >= dist_m

        one_hot = np.zeros((len(neighbor_categories), num_cate), dtype=np.int32)
        one_hot[np.arange(len(neighbor_categories)), neighbor_categories] = 1

        freq_matrix = mask_matrix.astype(np.int32) @ one_hot
        results.append(freq_matrix)

    circle_result = compute_frequency_differences(results)

    return circle_result


if __name__ == "__main__":
    poi_list = []
    id_dataset = 'lixia'
    source_data = read_json(f'datasets\\{id_dataset}\\poi_data_neighbor_k_15.json')
    n = len(source_data)
    if id_dataset == 'lixia' or id_dataset == 'haidian':
      for data in source_data:
        poi_list.append([data['info']['coordinates'][1], data['info']['coordinates'][0], data['info']['class']])
    else:
      for data in source_data:
        poi_list.append([data['info']['coordinates'][0], data['info']['coordinates'][1], data['info']['class']])
    num_cate = 140
    results = calculate_poi_category_frequencies(poi_list, num_cate)
    results = np.array(results)
    np.save(f'datasets\\{id_dataset}\\neighbor_counts_50_500_small_circle.npy', results)

