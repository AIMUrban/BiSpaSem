import csv
import json
import math
import re
from math import radians, cos, sin, asin, sqrt, ceil

from pandas.core.frame import DataFrame
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

from sklearn.neighbors import BallTree
from tqdm import tqdm
from collections import defaultdict


def swap_tuple_elements(input_list):
    return [(t[1],t[0]) for t in input_list]

def process_source_data(id_dataset):
    """
        For haidian and lixia
    """
    original_data = []
    with open(f'\datasets\\{id_dataset}F_data.csv', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            original_data.append(row)

    category_list = []
    category_LAG_list = []
    category_MID_list = []
    name_list = []
    for data in original_data:
        category_str = data[5]
        category_str = category_str.split('|')
        category_str = category_str[0]
        categories = category_str.split(';')
        if len(categories) < 3:
            print(categories)
        category_list.append(categories[-1])
        category_LAG_list.append(categories[-3])
        category_MID_list.append(categories[-2])
        name_list.append(data[4])


    def process_poi_name(name_list):
        namelist = []
        for i in range(len(name_list)):
            poiname = re.sub('\(.*?\)', '', name_list[i])  # 去掉poiname中的括号以及括号内的内容
            p1 = re.compile(r'[(](.*?)[)]', re.S)
            bracketscontent = re.findall(p1, name_list[i])
            if len(bracketscontent) >= 1 and bracketscontent[0] != "装修中":
                poiname = poiname + "("+bracketscontent[0]+")"  #
            namelist.append(poiname)

        return namelist

    category_list = list(set(category_list))
    category_LAG_list = list(set(category_LAG_list))
    category_MID_list = list(set(category_MID_list))
    name_list = process_poi_name(name_list)
    def convert_coordinate(coordinate_str):
        try:
            # 分割字符串并去除可能的空白字符
            parts = [part.strip() for part in coordinate_str.split('#')]

            # 转换为浮点数
            longitude = float(parts[0])
            latitude = float(parts[1])

            return longitude, latitude

        except Exception as e:
            print(f"转换失败: {e}")
            return None

    long_lan_list = []
    for item in tqdm(original_data, desc="get original lan_lon", total=len(original_data)):
        longitude, latitude = convert_coordinate(item[3])
        long_lan_list.append((longitude, latitude))

    long_lan_list = swap_tuple_elements(long_lan_list)

    def split_list_and_save(data_list, prefix="poi_original_location", encoding="utf-8"):
        if not data_list:
            print("错误：数据列表为空")
            return

        half_length = len(data_list) // 2

        first_half = data_list[:half_length]
        second_half = data_list[half_length:]

        save_to_csv(first_half, type_save_path + f"{prefix}_first_half.csv", encoding)

        save_to_csv(second_half, type_save_path + f"{prefix}_second_half.csv", encoding)

        print(f"成功将列表分割为两部分并保存到 {prefix}_first_half.csv 和 {prefix}_second_half.csv")

    def save_to_csv(data, file_path, encoding="utf-8"):
        try:
            with open(file_path, 'w', newline='', encoding=encoding) as csvfile:
                writer = csv.writer(csvfile)
                for item in data:
                    writer.writerow(item)

            print(f"成功保存数据到 {file_path}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    split_list_and_save(long_lan_list)

    """
        Please convert the coordinates (GCJ-02) in the raw data before executing the subsequent code!!!
    """


    pro_loclist_first_half = []
    with open(f'\datasets\\{id_dataset}\\type_tree\\poi_processed_original_location_first_half.csv',
              encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            pro_loclist_first_half.append(row)

    pro_loclist_second_half = []
    with open(f'\datasets\\{id_dataset}\\type_tree\\poi_processed_original_location_second_half.csv',
              encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            pro_loclist_second_half.append(row)



    pro_loclist_first_half.extend(pro_loclist_second_half)
    long_lan_list = pro_loclist_first_half

    def list_to_dict(lst):
        return {value: index for index, value in enumerate(lst)}

    category_dict = list_to_dict(category_list)
    category_LAG_dict = list_to_dict(category_LAG_list)
    category_MID_dict = list_to_dict(category_MID_list)

    full_poi_data = []
    for i, (op, long_lan) in tqdm(enumerate(zip(original_data, long_lan_list)), total=len(long_lan_list), desc="get_full_poi_data"):
        name = name_list[i]
        category_str = op[5]
        category_str = category_str.split('|')
        category_str = category_str[0]
        categories = category_str.split(';')
        small_category = categories[-1]
        mid_category = categories[-2]
        large_category = categories[-3]
        class_id = category_dict[small_category]
        mid_class_id = category_MID_dict[mid_category]
        large_class_id = category_LAG_dict[large_category]
        coor = (float(long_lan[1]), float(long_lan[0]))
        full_poi_data.append((name, coor, (small_category, class_id, mid_category, mid_class_id, large_category, large_class_id)))

    train_index = int(len(full_poi_data) * 0.8) # only train data
    poi_list_train = full_poi_data[0:train_index]


    train_coords = []
    for _, coords, _ in poi_list_train:
        lat, lon = coords[1], coords[0]
        train_coords.append([math.radians(lat), math.radians(lon)])

    tree = BallTree(np.array(train_coords), metric='haversine')

    result = []
    query_points = []
    for name, coords, small_category in full_poi_data:
        lat, lon = coords[1], coords[0]
        query_points.append([math.radians(lat), math.radians(lon)])

    distances_rad, indices = tree.query(query_points, k=16)  # 多查1个以排除自身

    EARTH_RADIUS = 6371000

    for i, ((name, coordinates, small_category), dists, idxs) in tqdm(
            enumerate(zip(full_poi_data, distances_rad, indices)),
            total=len(full_poi_data),
            desc='Finding neighbors'
    ):
        neighbors = []
        for j, (dist_rad, idx) in enumerate(zip(dists, idxs)):
            if i < train_index and idx == i:
                continue

            if idx >= len(poi_list_train):
                continue

            neighbor_name, neighbor_coords, neighbor_type = poi_list_train[idx]

            distance_m = dist_rad * EARTH_RADIUS

            neighbors.append((distance_m, neighbor_name, neighbor_coords, neighbor_type))

        top_15_neighbors = neighbors[:15]

        name_list = [neighbor[1] for neighbor in top_15_neighbors]
        coordinates_list = [neighbor[2] for neighbor in top_15_neighbors]
        poi_type_list = [neighbor[3] for neighbor in top_15_neighbors]

        poi_dict = {
            "info": {
                "name": name,
                "coordinates": coordinates,
                "class_name": small_category[0],
                "class": small_category[1],
                "mid_class": small_category[2],
                "mid_class_id": small_category[3],
                "large_class": small_category[4],
                "large_class_id": small_category[5]
            },
            "neighbor_info": {
                    "name_list": name_list,
                    "coordinates_list": [{"coordinates":[coordinate[0],coordinate[1]]} for coordinate in coordinates_list],
                    "neighbor_class_name_list": [poi_type[0] for poi_type in poi_type_list],
                    "neighbor_class_list": [poi_type[1] for poi_type in poi_type_list],
                    "neighbor_mid_class_name_list":[poi_type[2] for poi_type in poi_type_list],
                    "neighbor_mid_class_list": [poi_type[3] for poi_type in poi_type_list],
                    "neighbor_large_class_name_list": [poi_type[4] for poi_type in poi_type_list],
                    "neighbor_large_class_list": [poi_type[5] for poi_type in poi_type_list],
                }
            }

        result.append(poi_dict)
    with open(f'\datasets\\{id_dataset}\\poi_data_neighbor_k_15.json', 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

def v1_new_neighbor_add():
    """
            For NY and TYO
    """
    original_data = []
    with open(f'\datasets\\{id_dataset}\\poi_2024_with_cate.csv',
              encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            original_data.append(row)

    category_list = []
    name_list = []
    for data in original_data:
        category_str = data[3]
        category_list.append(category_str)
        name_list.append(data[0])

    category_list = list(set(category_list))
    print(len(category_list))

    long_lan_list = []
    for item in tqdm(original_data, desc="get original lan_lon", total=len(original_data)):
        longitude, latitude = float(item[2]), float(item[1])
        long_lan_list.append([longitude, latitude])

    def list_to_dict(lst):
        """将字符串列表转换为 {元素: 索引} 的字典"""
        return {value: index for index, value in enumerate(lst)}

    category_dict = list_to_dict(category_list)
    category_LAG_dict = list_to_dict(category_list)
    category_MID_dict = list_to_dict(category_list)

    full_poi_data = []
    for i, (op, long_lan) in tqdm(enumerate(zip(original_data, long_lan_list)), total=len(long_lan_list),
                                  desc="get_full_poi_data"):
        name = name_list[i]
        categories = op[3]
        small_category = categories
        mid_category = categories
        large_category = categories
        class_id = category_dict[small_category]
        mid_class_id = category_MID_dict[mid_category]
        large_class_id = category_LAG_dict[large_category]
        coor = (float(long_lan[1]), float(long_lan[0]))
        full_poi_data.append(
            (name, coor, (small_category, class_id, mid_category, mid_class_id, large_category, large_class_id)))

    train_index = int(len(full_poi_data) * 0.8)
    poi_list_train = full_poi_data[0:train_index]

    train_coords = []
    for _, coords, _ in poi_list_train:
        lon, lat = coords[1], coords[0]
        train_coords.append([math.radians(lat), math.radians(lon)])

    tree = BallTree(np.array(train_coords), metric='haversine')

    result = []
    query_points = []
    for name, coords, small_category in full_poi_data:
        lon, lat = coords[1], coords[0]
        query_points.append([math.radians(lat), math.radians(lon)])

    distances_rad, indices = tree.query(query_points, k=16)  # 多查1个以排除自身
    EARTH_RADIUS = 6371000

    for i, ((name, coordinates, small_category), dists, idxs) in tqdm(
            enumerate(zip(full_poi_data, distances_rad, indices)),
            total=len(full_poi_data),
            desc='Finding neighbors'
    ):
        neighbors = []
        for j, (dist_rad, idx) in enumerate(zip(dists, idxs)):
            if i < train_index and idx == i:
                continue
            if idx >= len(poi_list_train):
                continue
            neighbor_name, neighbor_coords, neighbor_type = poi_list_train[idx]
            distance_m = dist_rad * EARTH_RADIUS

            neighbors.append((distance_m, neighbor_name, neighbor_coords, neighbor_type))

        top_15_neighbors = neighbors[:15]
        name_list = [neighbor[1] for neighbor in top_15_neighbors]
        coordinates_list = [neighbor[2] for neighbor in top_15_neighbors]
        poi_type_list = [neighbor[3] for neighbor in top_15_neighbors]

        poi_dict = {
            "info": {
                "name": name,
                "coordinates": coordinates,
                "class_name": small_category[0],
                "class": small_category[1],
                "mid_class": small_category[2],
                "mid_class_id": small_category[3],
                "large_class": small_category[4],
                "large_class_id": small_category[5]
            },
            "neighbor_info": {
                "name_list": name_list,
                "coordinates_list": [{"coordinates": [coordinate[0], coordinate[1]]} for coordinate in
                                     coordinates_list],
                "neighbor_class_name_list": [poi_type[0] for poi_type in poi_type_list],
                "neighbor_class_list": [poi_type[1] for poi_type in poi_type_list],
                "neighbor_mid_class_name_list": [poi_type[2] for poi_type in poi_type_list],
                "neighbor_mid_class_list": [poi_type[3] for poi_type in poi_type_list],
                "neighbor_large_class_name_list": [poi_type[4] for poi_type in poi_type_list],
                "neighbor_large_class_list": [poi_type[5] for poi_type in poi_type_list],
            }
        }

        result.append(poi_dict)
    with open(f'\datasets\\{id_dataset}\\v1_poi_data_neighbor_k_15.json',
              'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    print("处理完成，数据已保存到 v1_poi_data_neighbor_k_15.json 文件中。")
    print()

def get_class_text_neighbor_data():
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
    data_list = read_json(v1_data)
    text_data = []
    if id_dataset == 'haidian' or id_dataset == 'lixia':
        name_prompt = '名称：'
        category_prompt = '类别：'
    elif id_dataset == 'NY':
        name_prompt = 'name:'
        category_prompt = 'category:'
    elif id_dataset == 'TYO':
        name_prompt = '名称:'
        category_prompt = '類別:'
    for data in tqdm(data_list, desc="get_pretrain_data", total=len(data_list)):
        text_data.append({
            "info": {
                "name": data["info"]["name"],
                "coordinates": data["info"]["coordinates"],
                "class": data["info"]["class"]
            },
            "neighbor_info": {
                "name_list": [f'{name_prompt}{name},{category_prompt}{neighbor_class}' for name, neighbor_class in zip(data["neighbor_info"]["name_list"], data["neighbor_info"]["neighbor_class_name_list"])],
                "coordinates_list": data["neighbor_info"]["coordinates_list"]
            }

        })

    with open(f'\datasets\\{id_dataset}\\v1_poi_data with_class_neighbor_k_15.json', 'w',
                  encoding='utf-8') as json_file:
            json.dump(text_data, json_file, ensure_ascii=False, indent=4)
    print("处理完成，数据已保存到 v1_poi_data with_class_neighbor_k_15.json 文件中。")


if __name__ == "__main__":

    id_dataset = 'lixia'
    type_save_path = f'\datasets\\{id_dataset}\\type_tree\\'
    v1_data = f'\datasets\\{id_dataset}\\v1_poi_data_neighbor_k_15.json'
    process_source_data(id_dataset)
