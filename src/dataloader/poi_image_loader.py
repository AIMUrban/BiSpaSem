import numpy as np
import torch
from torch.utils.data import Dataset
import json
class GeoBERTDataset(Dataset):
    def __init__(self, tokenizer, max_token_len, sep_between_neighbors=True):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.sep_between_neighbors = sep_between_neighbors

    def parse_spatial_context(self, poi_name, poi_coordinate, neighbor_name_list, neighbor_coordinate_list,
                              spatial_coordinate_fill):

        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_token_len = self.max_token_len

        poi_name_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(poi_name))
        poi_token_len = len(poi_name_tokens)

        pivot_lng = poi_coordinate[0]
        pivot_lat = poi_coordinate[1]

        # prepare entity mask
        entity_mask_arr = []
        # random number for masking
        rand_entity = np.random.uniform(
            size=len(neighbor_name_list) + 1)


        # 15%
        if rand_entity[0] < 0.15:
            entity_mask_arr.extend([True] * poi_token_len)
        else:
            entity_mask_arr.extend([False] * poi_token_len)

        # process neighbors
        neighbor_token_list = []
        neighbor_lng_list = []
        neighbor_lat_list = []
        token_len_all = poi_token_len + 1
        # add separator between poi and neighbor tokens

        if self.sep_between_neighbors:
            neighbor_lng_list.append(spatial_coordinate_fill)
            neighbor_lat_list.append(spatial_coordinate_fill)
            neighbor_token_list.append(sep_token_id)
            token_len_all = token_len_all + 1  # 用来限制长度

        for neighbor_name, neighbor_coordinate, rnd in zip(neighbor_name_list, neighbor_coordinate_list, rand_entity[1:]):

            if not neighbor_name[0].isalpha():

                continue

            neighbor_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neighbor_name))
            neighbor_token_len = len(neighbor_token)
            if token_len_all + neighbor_token_len + 1 > max_token_len:
                break
            else:
                token_len_all = token_len_all + neighbor_token_len


            neighbor_lng_list.extend([neighbor_coordinate['coordinates'][0]] * neighbor_token_len)
            neighbor_lat_list.extend([neighbor_coordinate['coordinates'][1]] * neighbor_token_len)
            neighbor_token_list.extend(neighbor_token)

            if self.sep_between_neighbors:
                neighbor_lng_list.append(spatial_coordinate_fill)
                neighbor_lat_list.append(spatial_coordinate_fill)
                neighbor_token_list.append(sep_token_id)

                entity_mask_arr.extend([False])
                token_len_all = token_len_all + 1

            if rnd < 0.15:
                # True: mask out, False: Keey original token
                entity_mask_arr.extend([True] * neighbor_token_len)
            else:
                entity_mask_arr.extend([False] * neighbor_token_len)

        pseudo_sentence = poi_name_tokens + neighbor_token_list
        sentence_lng_list = [pivot_lng] * poi_token_len + neighbor_lng_list
        sentence_lat_list = [pivot_lat] * poi_token_len + neighbor_lat_list

        # including cls and sep
        sent_len = len(pseudo_sentence)

        max_token_len_middle = max_token_len - 2

        # padding and truncation
        if sent_len > max_token_len_middle:
            pseudo_sentence = [cls_token_id] + pseudo_sentence[:max_token_len_middle] + [sep_token_id]
            sentence_lat_list = [spatial_coordinate_fill] + sentence_lat_list[:max_token_len_middle] + [spatial_coordinate_fill]
            sentence_lng_list = [spatial_coordinate_fill] + sentence_lng_list[:max_token_len_middle] + [spatial_coordinate_fill]
            attention_mask = [False] + [1] * max_token_len_middle + [False]  # attention
        else:
            pad_len = max_token_len_middle - sent_len
            assert pad_len >= 0

            pseudo_sentence = [cls_token_id] + pseudo_sentence + [sep_token_id] + [pad_token_id] * pad_len
            sentence_lat_list = [spatial_coordinate_fill] + sentence_lat_list + [spatial_coordinate_fill] + [spatial_coordinate_fill] * pad_len
            sentence_lng_list = [spatial_coordinate_fill] + sentence_lng_list + [spatial_coordinate_fill] + [spatial_coordinate_fill] * pad_len
            attention_mask = [False] + [1] * sent_len + [0] * pad_len + [False]



        # mask entity in the pseudo sentence
        # Account for the leading CLS and exclude positions removed by truncation.
        entity_mask_indices = np.where(entity_mask_arr)[0] + 1
        entity_mask_indices = entity_mask_indices[entity_mask_indices <= max_token_len_middle]
        masked_entity_input = [mask_token_id if i in entity_mask_indices else pseudo_sentence[i] for i in
                               range(0, max_token_len)]

        # mask token in the pseudo sentence
        rand_token = np.random.uniform(size=len(pseudo_sentence))
        # do not mask out cls and sep token
        token_mask_arr = (rand_token < 0.15) & (np.array(pseudo_sentence) != cls_token_id) & (
                    np.array(pseudo_sentence) != sep_token_id) & (np.array(pseudo_sentence) != pad_token_id)
        token_mask_indices = np.where(token_mask_arr)[0]

        masked_token_input = [mask_token_id if i in token_mask_indices else pseudo_sentence[i] for i in
                              range(0, max_token_len)]

        # yield masked_token with 50% prob, masked_entity with 50% prob
        if np.random.rand() > 0.5:
            masked_input = torch.tensor(masked_entity_input)
        else:
            masked_input = torch.tensor(masked_token_input)
        sentence_lat_list = np.expand_dims(sentence_lat_list, axis=0)
        sentence_lng_list = np.expand_dims(sentence_lng_list, axis=0)
        train_data = {}
        train_data['poi_name'] = poi_name
        train_data['poi_token_len'] = poi_token_len
        train_data['masked_input'] = masked_input
        train_data['sent_position_ids'] = torch.tensor(np.arange(0, len(pseudo_sentence)))
        train_data['attention_mask'] = torch.tensor(attention_mask)
        train_data['sent_coordinate'] = torch.tensor(np.concatenate((sentence_lng_list, sentence_lat_list), axis=0).transpose(1, 0)).to(torch.float32)
        train_data['pseudo_sentence'] = torch.tensor(pseudo_sentence)

        return train_data

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


def read_json(file_path):
        try:
            # 以只读模式打开 JSON 文件，并指定编码为 UTF-8
            with open(file_path, 'r', encoding='utf-8') as file:
                # 使用 json.load 方法将文件内容解析为 Python 对象
                data = json.load(file)

        except FileNotFoundError:
            print(f"错误：未找到文件 {file_path}。")
        except json.JSONDecodeError:
            print(f"错误：无法解析 {file_path} 中的 JSON 数据。请检查文件格式是否正确。")
        except Exception as e:
            print(f"发生未知错误：{e}")
        return data

class GeoSemPA_Dataset(GeoBERTDataset):
    def __init__(self, poi_data_file_path,
                 poi_source_data_file_path,
                 load_neighbor_counts_multi_scale,
                 load_name_embeddings_path,
                 tokenizer=None,
                 max_token_len=512,
                 spatial_coordinate_fill=0,
                 sep_between_neighbors=True,
                 mode=None,
                 num_neighbor_limit=None,
                 random_remove_neighbor=0.,
                 one_hot_shape=140,
                 type_key_str='class',
                 id_dataset = 'haidian',

                 ):
        self.tokenizer = tokenizer
        self.type_key_str = type_key_str  # key name of the class type in the input data dictionary
        self.max_token_len = max_token_len
        self.spatial_coordinate_fill = spatial_coordinate_fill
        self.sep_between_neighbors = sep_between_neighbors
        self.num_neighbor_limit = num_neighbor_limit

        self.random_remove_neighbor = random_remove_neighbor
        self.read_file(poi_data_file_path,poi_source_data_file_path,mode, one_hot_shape, load_name_embeddings_path, load_neighbor_counts_multi_scale)
        self.id_dataset = id_dataset

        super(GeoSemPA_Dataset, self).__init__(self.tokenizer, max_token_len, sep_between_neighbors)

    def read_file(self, poi_data_file_path, poi_source_data_file_path,mode,one_hot_shape,load_name_embeddings_path,load_neighbor_counts_multi_scale):

        poi_data = read_json(poi_data_file_path)
        poi_source_data = read_json(poi_source_data_file_path)
        first_layer_list = []
        second_layer_list = []
        third_layer_list = []
        labels = []
        coordinates = []
        # 遍历原始列表
        for poi in poi_source_data:
            first_layer_list.append(poi["info"]["large_class"])
            second_layer_list.append(poi["info"]["mid_class"])
            third_layer_list.append(poi["info"]["class_name"])
            labels.append(poi["info"]["class"])
            coordinates.append(poi["info"]["coordinates"])
        name_embeddings = torch.load(load_name_embeddings_path)
        neighbor_counts_multiscale = np.load(load_neighbor_counts_multi_scale)
        first_index = one_hot_shape

        first_index_result = []
        element_index_map = {}
        for element in first_layer_list:
            if element not in element_index_map:
                element_index_map[element] = first_index
                first_index += 1
            first_index_result.append(element_index_map[element])

        second_index_result = []
        element_index_map = {}
        for element in second_layer_list:
            if element not in element_index_map:
                element_index_map[element] = first_index
                first_index += 1
            second_index_result.append(element_index_map[element])

        third_layer_list = []
        for element in poi_data:
            poi_type = element['info'][self.type_key_str]
            third_layer_list.append(poi_type)
        poi_layer_type_list = [[x, y, z] for x, y, z in zip(first_index_result, second_index_result, third_layer_list)]
        layer_count = len(set(first_index_result+second_index_result+third_layer_list))

        max_idx = max(first_index_result+second_index_result+third_layer_list)
        mask_idx = max_idx + 1
        train_index = int(len(poi_data) * 0.8)
        if mode == 'train':
            poi_layer_type_list = poi_layer_type_list[0:train_index]
            first_index_result = first_index_result[0:train_index]
            second_index_result = second_index_result[0:train_index]
            labels = labels[0:train_index]
            poi_data = poi_data[0:train_index]
            coordinates = coordinates[0:train_index]
            name_embeddings = name_embeddings[0:train_index]
            neighbor_counts_multiscale = neighbor_counts_multiscale[0:train_index]
        elif mode == 'test':
            # poi_streetimage_embedding = poi_streetimage_embedding[train_index:]
            poi_layer_type_list = poi_layer_type_list[train_index:]
            first_index_result = first_index_result[train_index:]
            second_index_result = second_index_result[train_index:]
            labels = labels[train_index:]
            poi_data = poi_data[train_index:]
            coordinates = coordinates[train_index:]
            name_embeddings = name_embeddings[train_index:]
            neighbor_counts_multiscale = neighbor_counts_multiscale[train_index:]
        elif mode is None:
            pass
        else:
            raise NotImplementedError

        self.len_data = len(poi_data)
        self.poi_layer_type_list = poi_layer_type_list
        self.first_index_result = first_index_result
        self.second_index_result = second_index_result
        self.labels =labels
        self.poi_data = poi_data
        self.mask_idx = mask_idx
        self.layer_count = layer_count
        self.coordinates = coordinates
        self.name_embeddings = name_embeddings
        self.neighbor_counts_multiscale = neighbor_counts_multiscale

    def load_data(self, index):

        spatial_coordinate_fill = self.spatial_coordinate_fill
        poi_line = self.poi_data[index]
        poi_layer_type = self.poi_layer_type_list[index]
        poi_line_data_dict = poi_line
        poi_name = poi_line_data_dict['info']['name']
        poi_coordinate = poi_line_data_dict['info']['coordinates']

        neighbor_info = poi_line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_coordinate_list = neighbor_info['coordinates_list']

        if self.random_remove_neighbor != 0:
            num_neighbors = len(neighbor_name_list)
            rand_neighbor = np.random.uniform(size=num_neighbors)

            neighbor_keep_arr = (rand_neighbor >= self.random_remove_neighbor)  # select the neighbors to be removed
            neighbor_keep_arr = np.where(neighbor_keep_arr)[0]

            new_neighbor_name_list, new_neighbor_coordinate_list = [], []
            for i in range(0, num_neighbors):
                if i in neighbor_keep_arr:
                    new_neighbor_name_list.append(neighbor_name_list[i])
                    new_neighbor_coordinate_list.append(neighbor_coordinate_list[i])

            neighbor_name_list = new_neighbor_name_list
            neighbor_coordinate_list = new_neighbor_coordinate_list

        if self.num_neighbor_limit is not None:
            neighbor_name_list = neighbor_name_list[0:self.num_neighbor_limit]
            neighbor_coordinate_list = neighbor_coordinate_list[0:self.num_neighbor_limit]

        train_data = self.parse_spatial_context(poi_name, poi_coordinate, neighbor_name_list, neighbor_coordinate_list,
                                                spatial_coordinate_fill)

        poi_type = poi_line_data_dict['info'][self.type_key_str]
        train_data['poi_type'] = torch.tensor(poi_type)
        train_data['first_layer_type'] = torch.tensor(self.first_index_result[index])
        train_data['second_layer_type'] = torch.tensor(self.second_index_result[index])
        train_data['poi_layer_type'] = poi_layer_type
        train_data['coordinates'] = torch.tensor(self.coordinates[index])
        train_data['name_embeddings'] = torch.tensor(self.name_embeddings[index], dtype=torch.float32)
        train_data['neighbor_counts_multiscale'] = torch.tensor(self.neighbor_counts_multiscale[index], dtype=torch.float32)

        return train_data

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)
