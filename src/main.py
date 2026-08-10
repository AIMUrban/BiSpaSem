import random
from models.TCSE_model.TCSE import GeoBERTConfig, GeoBertForSemanticTyping
from models.GeoSemPA import GeoSemPA
from dataloader.poi_image_loader import GeoSemPA_Dataset
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
from transformers import BertTokenizer, BertJapaneseTokenizer
import gc

def training(args):
    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epoch
    max_token_len = args.max_token_len

    if args.id_dataset == 'TYO':
        tokenizer = BertJapaneseTokenizer.from_pretrained(args.bert_model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    config = GeoBERTConfig(num_semantic_types=args.num_classes,vocab_size=args.vocab_size)

    print("Data_path:")
    print(args.load_data_path)

    train_dataset = GeoSemPA_Dataset(
        poi_data_file_path=args.load_data_path,
        poi_source_data_file_path=args.load_source_data_path,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        spatial_coordinate_fill=0,
        sep_between_neighbors=True,
        one_hot_shape=args.num_classes,
        mode='train',
        id_dataset=args.id_dataset,
        load_neighbor_counts_multi_scale=args.load_neighbor_counts_multi_scale,
        load_name_embeddings_path=args.load_name_embeddings_path,
        num_neighbor_limit=args.k
    )
    test_dataset = GeoSemPA_Dataset(
        poi_data_file_path=args.load_data_path,
        poi_source_data_file_path=args.load_source_data_path,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        spatial_coordinate_fill=0,
        sep_between_neighbors=True,
        one_hot_shape=args.num_classes,
        mode='test',
        id_dataset=args.id_dataset,
        load_neighbor_counts_multi_scale=args.load_neighbor_counts_multi_scale,
        load_name_embeddings_path=args.load_name_embeddings_path,
        num_neighbor_limit=args.k
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)

    device = args.device

    bert = GeoBertForSemanticTyping(config)

    print("model_path:")
    print(args.load_model_path)
    bert.load_state_dict(torch.load(args.load_model_path), strict=False)

    loaded_weights = bert.state_dict()
    for name, param in bert.named_parameters():
        if name in loaded_weights:
            if not torch.equal(param, loaded_weights[name]):
                print(f"{name} 的权重没有被加载")


    print(args.num_classes)

    model = GeoSemPA(
            geobert=bert,
            bert_input_dim=768,
            hidden_dim=256,
            num_semantic_types=args.num_classes,
            num_circle=args.num_circle,
            third_layer_class=args.num_classes
    )
    model.to(device)

    for param in model.geobert.parameters():
        param.requires_grad = False

    """
    In our actual training process, due to the risk of overfitting, after completing the training of two mask
    tasks on GeoBERT, we separately conducted pre-training for POI semantic annotation on GeoBERT. We used
    a two-layer neural network with a ELU activation function as the prediction head and trained it for 3
    epochs. After the pre-training was completed, we reloaded the trained GeoBERT into
    the model, froze its parameters, and then performed the complete training and testing of GeoSemPA.
    """

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=5E-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=6, gamma=0.9)

    print('start training...')
    print("save_model_path:")

    loss_list = []


    iter = 0
    epoch_idx = 0

    for epoch in tqdm(range(epochs)):
        epoch_loss_list = []

        print("load_model_path:")
        print(args.load_model_path)
        print("load_data_path:")
        print(args.load_data_path)
        print("save_result_path:")
        print(args.base_result_outdir)
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_coordinate_list = batch['sent_coordinate'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)
            poi_lens = batch['poi_token_len'].to(device)
            labels = batch['poi_type'].to(device)
            name_embeddings = batch['name_embeddings'].to(device)
            neighbor_counts_multiscale = batch['neighbor_counts_multiscale'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                            sent_coordinate_list=sent_coordinate_list, labels=labels,
                            poi_len_list=poi_lens,neighbor_counts_multiscale=neighbor_counts_multiscale,name_embeddings=name_embeddings,
                            id_dataset=args.id_dataset)

            loss = outputs[0]
            typing_loss = outputs[2]
            weighted_typing_loss = outputs[3]
            loss.backward()
            optim.step()
            optim.zero_grad()


            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss': loss.item(), 'typing_loss': typing_loss.item(), 'weighted_typing_loss': weighted_typing_loss.item()})
            loss_list.append(loss.item())

            epoch_loss_list.append(loss.item())

            if iter %2000 == 0:
                plt_batch(args, loss_list)
            iter += 1
        scheduler.step()
        test(test_loader, model, device, epoch, args.base_result_outdir, args.l, args.task,args)

def test(test_loader, model, device, epoch, base_result_outdir,l, task,args):
    model.eval()
    print("save_result_path:")
    print(base_result_outdir)
    if not os.path.exists(base_result_outdir):
        # 目录不存在，进行创建操作
        os.makedirs(base_result_outdir)
    header = ('l', 'epoch', 'Accuracy', 'F1-score', 'MRR',task)
    with open(base_result_outdir + 'result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with torch.no_grad():

        loop = tqdm(test_loader, leave=True)
        pred_list = []
        label_list = []
        weighted_prediction = []
        for batch in loop:
            # geobert
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_coordinate_list = batch['sent_coordinate'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)
            poi_lens = batch['poi_token_len'].to(device)
            labels = batch['poi_type'].to(device)
            name_embeddings = batch['name_embeddings'].to(device)
            neighbor_counts_multiscale = batch['neighbor_counts_multiscale'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                            sent_coordinate_list=sent_coordinate_list, labels=labels,
                            poi_len_list=poi_lens,
                            neighbor_counts_multiscale=neighbor_counts_multiscale, name_embeddings=name_embeddings,id_dataset=args.id_dataset)

            pred_list.extend(outputs[1].cpu().detach().numpy())
            label_list.extend(batch['poi_type'])
        y_testlabel = np.array(label_list)
        predictions_test = np.array(pred_list)
        predictions_test_dim = np.argmax(predictions_test, axis=1)
        accuracy_score_test = accuracy_score(y_testlabel, predictions_test_dim)
        f1_score_test = f1_score(y_testlabel, predictions_test_dim, average="macro")
        mrr_test = compute_mrr(y_testlabel, predictions_test)
        with open(base_result_outdir + 'result.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([l, epoch, accuracy_score_test, f1_score_test, mrr_test])

    return

def compute_mrr(true_labels, machine_preds):
    """Compute the MRR """
    rr_total = 0.0
    for i in range(len(true_labels)):
        ranklist = list(np.argsort(machine_preds[i])[::-1])
        rank = ranklist.index(true_labels[i]) + 1
        rr_total = rr_total + 1.0 / rank
    mrr = rr_total / len(true_labels)
    return mrr
def plt_batch(args, lists):
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    rcParams['figure.figsize'] = (16, 6)
    plt.figure()
    # loss
    plt.plot(lists, 'g', label='train loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    if not os.path.exists(args.base_result_image_outdir):
        # 目录不存在，进行创建操作
        os.makedirs(args.base_result_image_outdir)
    plt.savefig(args.base_result_image_outdir + str(args.l) + ".png")
    plt.close()

class Config(object):
    def __init__(self, id_dataset, gpu_ids="0", k=15,epoch=10):
        self.PATH = os.path.dirname(os.path.realpath(__file__))
        self.PATH = self.PATH.replace('\\', '/')
        self.id_dataset = id_dataset
        self.gpu_ids = gpu_ids  # 指定GPU编号，如"0,1,2"
        if self.id_dataset == "lixia":
            self.max_token_len_dcit = {
                15: 512,
            }
        if self.id_dataset == "haidian":
            self.max_token_len_dcit = {
                15: 512,
            }
        if self.id_dataset == "TYO":
            self.max_token_len_dcit = {
                15: 512,
            }
        if self.id_dataset == "NY":
            self.max_token_len_dcit = {
                15: 512,
            }

        if self.id_dataset == "lixia" or self.id_dataset == "haidian":
            self.bert_model_path = self.PATH[:-3] + "model_weight/bert-base-chinese"
        elif self.id_dataset == "NY":
            self.bert_model_path = self.PATH[:-3] + "model_weight/bert-base-uncased"
        elif self.id_dataset == "TYO":
            self.bert_model_path = self.PATH[:-3] + "model_weight/bert-base-japanese-v3"

        if self.id_dataset == "haidian" or self.id_dataset == "lixia":
            self.vocab_size = 21128
        elif self.id_dataset == 'NY':
            self.vocab_size = 30522
        elif self.id_dataset == 'TYO':
            self.vocab_size = 32768
        self.k = k
        self.epoch = epoch
        self.batch_size = 60
        self.max_token_len = 512
        self.r = 0
        print(self.PATH)
        self.model_save_path = self.PATH[:-3] + "model_save_path/" + self.id_dataset + "/"
        self.load_model_path = self.PATH[:-3] + "model_weight/" + self.id_dataset + "/v1_class_geobert.pth"
        self.load_data_path = self.PATH[:-3] + "datasets/" + self.id_dataset + '/v1_poi_data with_class_neighbor_k_15.json'
        self.load_name_embeddings_path = self.PATH[:-3] + "datasets/" + self.id_dataset + '/name_embedding.pt'
        self.load_neighbor_counts_multi_scale = self.PATH[:-3] + "datasets/" + self.id_dataset + '/neighbor_counts_50_500_small_circle.npy'
        self.data_path = self.PATH[:-3] + "datasets/" + self.id_dataset + '/'
        self.load_source_data_path = self.PATH[:-3] + "datasets/" + self.id_dataset + '/v1_poi_data_neighbor_k_15.json'
        self.base_result_outdir = self.PATH[:-3] + "result/resulttxt/" + self.id_dataset + "/"
        self.base_result_image_outdir = self.PATH[:-3] + "result/image/" + self.id_dataset + "/"
        if self.id_dataset == "lixia":
            self.num_classes = 140
            self.lr = 0.0001
            self.num_semantic_types = self.num_classes
        if self.id_dataset == "haidian":
            self.num_classes = 248
            self.num_semantic_types = self.num_classes
            self.lr = 0.0001
        if self.id_dataset == "TYO":
            self.num_classes = 804
            self.num_semantic_types = self.num_classes
            self.lr = 0.0001
        if self.id_dataset == "NY":
            self.num_classes = 899
            self.num_semantic_types = self.num_classes
            self.lr = 0.0001
        self.num_workers = 0
        self.device = torch.device('cuda:0')
        self.task = 'GeoSemPA'
        self.random_remove_neighbor = 0
        self.random_remove_neighbor_cat = 0
        self.l = 0
        self.num_circle = 10

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed = 42
    seed_torch(seed)
    id_dataset = 'TYO'
    config = Config(id_dataset=id_dataset, k=15,epoch=10)
    training(config)







