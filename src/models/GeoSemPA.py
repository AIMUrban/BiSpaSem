import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import CrossEntropyLoss

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term_len = (d_model + 1) // 2  # 21→11
        div_term = torch.exp(
            torch.arange(0, div_term_len, dtype=torch.float) *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]  # 奇数索引只取前10列

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = torch.nn.Linear(int(hidden_size), 1, bias=False)

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class linearRegression_multi_att(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression_multi_att, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, inputSize, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(inputSize, outputSize, bias=True)

        self.attention = Attention(in_size=inputSize)


    def forward(self, x1, x2):  #
        Features = torch.stack([x1, x2], dim=1)

        Features = self.attention(Features)

        out = self.linear1(Features)
        out = self.act1(out)
        out = self.linear2(out)

        return out

class AnchorPOIPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, poi_len_list):
        # We "pool" the model by simply taking the hidden state corresponding
        bsize = hidden_states.shape[0]

        tensor_list = []
        for i in torch.arange(0, bsize):
            poi_token_full = hidden_states[i, 1:poi_len_list[i] + 1]
            # mag_embedding = hidden_states[i,360].unsqueeze(0)
            # poi_token_full = torch.cat([poi_token_full,mag_embedding],dim=0)

            poi_token_tensor = torch.mean(torch.unsqueeze(poi_token_full, 0), dim=1)
            tensor_list.append(poi_token_tensor)

        batch_poi_tensor = torch.cat(tensor_list, dim=0)

        return batch_poi_tensor

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class NameCategoryAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_fused,
                 dim_head=16,
                 heads=4,
                 ff_mult=1,
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.dim_fused = dim if dim_fused is None else dim_fused

        self.norm = LayerNorm(dim)
        self.norm_y = LayerNorm(self.dim_fused)
        self.norm_out = LayerNorm(dim_fused)

        self.to_q = nn.Sequential(nn.Linear(dim, inner_dim, bias=False))

        self.to_kv = nn.Linear(dim_fused, 2*inner_dim, bias=False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim,bias=False),
            nn.Dropout(p=0.3),
        )

        ff_inner_dim = ff_mult * dim
        self.ff1 = nn.Linear(dim, ff_inner_dim, bias=False)
        self.ff2 = nn.Linear(ff_inner_dim, dim, bias=False)
        self.feedforward = nn.Sequential(
            self.ff1,
            nn.ELU(),
            nn.Dropout(p=0.3),
            self.ff2
        )
    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm_y(y)
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale
        kv = self.to_kv(y)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', kv=2, h=self.heads)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + self.feedforward(x)
        return out


class GeoSemPA(nn.Module):
    def __init__(self, geobert,bert_input_dim, hidden_dim, num_semantic_types,third_layer_class,num_circle):
        super().__init__()
        self.num_layer_classes = third_layer_class
        self.dim = bert_input_dim
        self.geobert =geobert
        self.all_emb_dim =  bert_input_dim*2
        self.projector1 = nn.Sequential(
            nn.Linear(self.all_emb_dim, self.all_emb_dim),
            nn.Dropout(p=0.3),
            nn.ELU(),
            nn.Linear(self.all_emb_dim, hidden_dim)
        )
        self.attention_fusion_layer = linearRegression_multi_att(hidden_dim, hidden_dim)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ELU(),
            nn.Linear(hidden_dim, num_semantic_types)
        )
        self.num_semantic_types = num_semantic_types
        self.poi_pooler = AnchorPOIPooler()
        self.multiscale_cross_attention_fusion = NameCategoryAttentionBlock(dim=bert_input_dim, dim_fused=self.num_layer_classes)
        self.multiscale_positional_encoding = PositionalEncoding(d_model=self.num_layer_classes,max_len=num_circle)
        self.pretrain_cls = nn.Sequential(
            nn.Linear(self.all_emb_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ELU(),
            nn.Linear(hidden_dim, num_semantic_types)
        )

    def forward(self, input_ids, attention_mask, sent_position_ids, sent_coordinate_list, poi_len_list, labels,neighbor_counts_multiscale,name_embeddings,id_dataset):

        if neighbor_counts_multiscale.dim() == 2:
            neighbor_counts_multiscale = neighbor_counts_multiscale.unsqueeze(0)
        neighbor_counts_multiscale = self.multiscale_positional_encoding(
            neighbor_counts_multiscale * math.sqrt(self.num_layer_classes))

        name_embeddings = name_embeddings.unsqueeze(1)
        mix_multiscale_name_out = self.multiscale_cross_attention_fusion(name_embeddings, neighbor_counts_multiscale)
        mix_multiscale_name_out = mix_multiscale_name_out.squeeze()

        bert_output = self.geobert(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                                   sent_coordinate_list=sent_coordinate_list,
                                   poi_len_list=poi_len_list)
        bert_token_embedding = bert_output[1]

        if id_dataset == 'haidian' or id_dataset == 'lixia':
            bert_embedding = self.poi_pooler(bert_token_embedding, poi_len_list)
        elif id_dataset == 'NY' or id_dataset == 'TYO':
            if bert_token_embedding.dim() == 2:
                bert_token_embedding = bert_token_embedding.unsqueeze(0)
            bert_embedding = bert_token_embedding[0:, 0, 0:]
            bert_embedding = bert_embedding.squeeze()
        """
            Training the model with average pooling on POI names on the NY and TYO datasets results in the occurrence of NaN values. 
            This issue may be associated with the BERT models themselves used for these two datasets. We have not yet investigated 
            and identified the exact underlying causes. Therefore, we instead adopt the first token of BERT, [CLS] token, as a replacement.
        """

        if bert_embedding.dim() == 1:
            bert_embedding = bert_embedding.unsqueeze(0)
        if mix_multiscale_name_out.dim() == 1:
            mix_multiscale_name_out = mix_multiscale_name_out.unsqueeze(0)

        temp_out = torch.cat((bert_embedding, mix_multiscale_name_out), dim=1)

        all_embedding = self.projector1(temp_out)
        type_prediction_score = self.cls(all_embedding)

        loss_fct = CrossEntropyLoss()
        typing_loss = loss_fct(type_prediction_score.view(-1, self.num_semantic_types), labels.view(-1))
        total_loss = typing_loss

        return total_loss, type_prediction_score, typing_loss, typing_loss


