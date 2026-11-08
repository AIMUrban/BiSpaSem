import json
from pathlib import Path

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertJapaneseTokenizer
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

id_dataset = 'TYO'
# 1. 加载预训练中文BERT模型和tokenizer
if id_dataset == 'haidian' or id_dataset == 'lixia':
    model_name = str(PROJECT_ROOT / 'model_weight' / 'bert-base-chinese')
elif id_dataset == 'NY':
    model_name = str(PROJECT_ROOT / 'model_weight' / 'bert-base-uncased')
elif id_dataset == 'TYO':
    model_name = str(PROJECT_ROOT / 'model_weight' / 'bert-base-japanese-v3')
if id_dataset == 'TYO':
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. 定义POI名称列表示例（替换为实际数据）
source_data_path = PROJECT_ROOT / 'datasets' / id_dataset / 'v1_poi_data_neighbor_k_15.json'


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


source_data = read_json(source_data_path)
poi_names = []
for item in source_data:
    poi_names.append(item["info"]["name"])

def get_bert_embeddings(texts, batch_size=32):

    model.eval()  # 设置为评估模式
    embeddings = []

    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="使用BERT生成文本嵌入向量"):
        batch_texts = texts[i:i + batch_size]

        # 文本编码
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,  # 设置最大长度
            return_tensors='pt'
        )

        # GPU加速（如果可用）
        if torch.cuda.is_available():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs.items()}
            model.cuda()

        # 获取嵌入
        with torch.no_grad():
            outputs = model(**encoded_inputs)

        # 使用最后一层[CLS]标记作为句子表示
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# 4. 生成所有POI的嵌入向量
poi_vectors = get_bert_embeddings(poi_names)
torch.save(poi_vectors, PROJECT_ROOT / 'datasets' / id_dataset / 'name_embedding.pt')
