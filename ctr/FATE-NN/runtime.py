import pandas as pd
import numpy as np
import os
import argparse
import random
from model import Model
from metric import *
from utils import *
from control import *
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

import time
import logging

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='SARec')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='/home/wuqitian/FTrans/data/large', help='data_dir')
parser.add_argument('--dataset', type=str, default='criteo', help='dataset')
# splice, mushroom, nursery, drive, wine
parser.add_argument('--lr1', type=float, default=1e-4, help='learning_rate for pretrain')
parser.add_argument('--wd1', type=float, default=0., help='weight_decay for pretrain')
parser.add_argument('--lr2', type=float, default=1e-4, help='learning_rate for train')
parser.add_argument('--wd2', type=float, default=0., help='weight_decay for train')
parser.add_argument('--con_reg', type=float, default=0., help='contrastive regularization')
parser.add_argument('--batch_size', type=int, default=100000, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=1, help='epoch_num')
parser.add_argument('--embedding_size', type=int, default=10, help='hidden_size')
parser.add_argument('--hidden_size', type=int, default=400, help='hidden_size')
parser.add_argument('--is_detach', type=bool, default=False, help='whether to detach W')
parser.add_argument('--load_pretrain', type=bool, default=False, help='whether to detach W')
parser.add_argument('--gnn_layer_num', type=int, default=2, help='gnn_layer_num') # 2
parser.add_argument('--graphconv', type=str, default='SAGE', help='graphconv type')
parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio')
parser.add_argument('--field_ratio', type=float, default=0.8, help='train_ratio')
parser.add_argument('--new_field_ratio', type=float, default=0.2, help='train_ratio')
parser.add_argument('--known_num', type=float, default=13, help='known_ratio')
parser.add_argument('--val_ratio', type=float, default=0.1, help='val_ratio')
parser.add_argument('--test_ratio', type=float, default=0.1, help='test_ratio')
parser.add_argument('--dropout', type=float, default=0., help='dropout prob')
parser.add_argument('--dropedge_prob', type=float, default=0.5, help='known_ratio')
parser.add_argument('--training_mode', type=str, default='k-shot-random', help='training_mode')
parser.add_argument('--data_mode', type=str, default='less', help='training_mode')
parser.add_argument('--backbone', type=str, default='LR', help='backbone model')
parser.add_argument('--split_mode', type=str, default='ratio', help='how to split the data')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

fix_seed(args.seed)
device = f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

if args.dataset == 'avazu':
    # 0 ~ 21 categorical
    field_ind = torch.tensor(list(range(0, 22)), dtype=torch.long)
    new_field_ind = torch.tensor(list(range(22, 22)), dtype=torch.long)
elif args.dataset == 'criteo':
    # 0 ~ 12 continuous, 13 ~ 38 categorical
    field_ind = torch.tensor(list(range(0, 12)) + list(range(13, 39)), dtype=torch.long) # 7, 26
    new_field_ind = torch.tensor(list(range(12, 12)) + list(range(39, 39)), dtype=torch.long) # 12, 39

data_dir = os.path.join(args.data_dir, args.dataset)
with open(os.path.join(data_dir, 'data_filter4.pkl'), 'rb') as f:
    x = pickle.load(f)
    y = pickle.load(f)
x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.float32)

train_mask, val_mask, test_mask, field_mask, new_field_mask = data_split(x, train_ratio=args.train_ratio, \
                                                                                       field_ind=field_ind,
                                                                                        test_ratio=args.test_ratio,
                                                                                       val_ratio=args.val_ratio,
                                                                                       new_field_ind=new_field_ind,
                                                                                       mode='ratio')
train_x, train_y = x[train_mask], y[train_mask]
val_x, val_y = x[val_mask], y[val_mask]
test_x, test_y = x[test_mask], y[test_mask]
# train_x, train_y, test_x, test_y = data_split(x, y, ratio=0.5)
print(field_mask.sum(), new_field_mask.sum())
field_num = x.size(1)
field_dims = torch.max(x, dim=0)[0] + 1
print(field_num, field_dims)
print(train_x.size(), val_x.size(), test_x.size())

ce = torch.nn.BCEWithLogitsLoss()
field_nums = [39, 36, 33, 30, 27, 24, 21, 18, 15]
# batch_sizes = [1000000, 500000, 200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
batch_sizes = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]

for field_num in field_nums:

    batch_size = 100000
    train_x_ = train_x[:, :field_num]
    test_x_ = train_x[:, :field_num]
    field_mask_, new_field_mask_ = field_mask[:field_num], new_field_mask[:field_num]
    field_dims_ = field_dims[:field_num]
    tr_times, te_times = [], []
    tr_mems, te_mems = [], []
    model = Model(field_dims=field_dims_, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
              gnn_layer_num=args.gnn_layer_num, graphconv=args.graphconv, dropout_prob=args.dropout,
                 backbone=args.backbone, device=device).to(device)
    max_auc = 0.
    optimizer_f = torch.optim.Adam(model.param_bb, lr=args.lr1, weight_decay=args.wd1)
    optimizer_s = torch.optim.Adam(model.param_sn, lr=args.lr2, weight_decay=args.wd2)
    optimizer_nn = torch.optim.Adam(model.param_nn, lr=1e-3, weight_decay=0.)
    if args.training_mode == 'leave-one-out':
        known_masks = get_known_mask(field_mask_, known_num=args.known_num, mode=args.training_mode)
    for e in range(args.epoch_num):
        if args.training_mode == 'k-fold':
            known_masks = get_known_mask(field_mask_, known_num=args.known_num, mode=args.training_mode)
        if args.training_mode == 'k-shot-random':
            known_masks = get_known_mask(field_mask_, known_num=args.known_num, mode=args.training_mode, K=5)
        train_ind = torch.randperm(train_x_.size(0))
        for i in range(train_x_.size(0) // batch_size + 1):
            train_ind_i = train_ind[i * batch_size: (i + 1) * batch_size]
            train_x_i, train_y_i = train_x_[train_ind_i].to(device), train_y[train_ind_i].to(device)
            model.train()
            optimizer_s.zero_grad()
            optimizer_f.zero_grad()
            t_now = time.time()
            for k in range(len(known_masks)):
                logit, w_recon = model(train_x_i, field_mask_, known_mask=known_masks[k], mode='train', args=args)
                loss_fit = ce(logit, train_y_i)
                loss = loss_fit  # + args.con_reg*w_recon
                tr_mem1 = get_gpu_memory_map()[int(args.gpus)]
                loss.backward()
                tr_mem2 = get_gpu_memory_map()[int(args.gpus)]
                tr_mems.append(max(tr_mem1, tr_mem2))
                optimizer_f.step()
            optimizer_s.step()
            tr_times.append(time.time() - t_now)
            if i >= 20:
                break
        torch.cuda.empty_cache()
        with torch.no_grad():
            test_ind = torch.arange(test_x_.size(0))
            for i in range(test_x_.size(0) // batch_size + 1):
                test_ind_i = test_ind[i * batch_size: (i + 1) * batch_size]
                test_x_i, test_y_i = test_x_[test_ind_i].to(device), test_y[test_ind_i].to(device)
                t_now = time.time()
                pred = model(test_x_i, field_mask_, new_field_mask=new_field_mask_, mode='test')
                te_times.append(time.time() - t_now)
                te_mem = get_gpu_memory_map()[int(args.gpus)]
                te_mems.append(te_mem)
                if i >= 20:
                    break
        torch.cuda.empty_cache()

    tr_time = np.mean(tr_times)
    te_time = np.mean(te_times)
    tr_mem = np.mean(tr_mems)
    te_mem = np.mean(te_mems)
    print(field_dims[:field_num].sum(), tr_time, te_time, tr_mem, te_mem)

# for batch_size in batch_sizes:
#
#     tr_times, te_times = [], []
#     tr_mems, te_mems = [], []
#     model = Model(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
#               gnn_layer_num=args.gnn_layer_num, graphconv=args.graphconv, dropout_prob=args.dropout,
#                  backbone=args.backbone, device=device).to(device)
#     max_auc = 0.
#     optimizer_f = torch.optim.Adam(model.param_bb, lr=args.lr1, weight_decay=args.wd1)
#     optimizer_s = torch.optim.Adam(model.param_sn, lr=args.lr2, weight_decay=args.wd2)
#     optimizer_nn = torch.optim.Adam(model.param_nn, lr=1e-3, weight_decay=0.)
#     if args.training_mode == 'leave-one-out':
#         known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode)
#     for e in range(args.epoch_num):
#         if args.training_mode == 'k-fold':
#             known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode)
#         if args.training_mode == 'k-shot-random':
#             known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode, K=5)
#         train_ind = torch.randperm(train_x.size(0))
#         for i in range(train_x.size(0) // batch_size + 1):
#             train_ind_i = train_ind[i * batch_size: (i + 1) * batch_size]
#             train_x_i, train_y_i = train_x[train_ind_i].to(device), train_y[train_ind_i].to(device)
#             model.train()
#             optimizer_s.zero_grad()
#             optimizer_f.zero_grad()
#             t_now = time.time()
#             for k in range(len(known_masks)):
#                 logit, w_recon = model(train_x_i, field_mask, known_mask=known_masks[k], mode='train', args=args)
#                 loss_fit = ce(logit, train_y_i)
#                 loss = loss_fit  # + args.con_reg*w_recon
#                 tr_mem1 = get_gpu_memory_map()[int(args.gpus)]
#                 loss.backward()
#                 tr_mem2 = get_gpu_memory_map()[int(args.gpus)]
#                 tr_mems.append(max(tr_mem1, tr_mem2))
#                 optimizer_f.step()
#             optimizer_s.step()
#             tr_times.append(time.time() - t_now)
#             if i >= 20:
#                 break
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             test_ind = torch.arange(test_x.size(0))
#             for i in range(test_x.size(0) // batch_size + 1):
#                 test_ind_i = test_ind[i * batch_size: (i + 1) * batch_size]
#                 test_x_i, test_y_i = test_x[test_ind_i].to(device), test_y[test_ind_i].to(device)
#                 t_now = time.time()
#                 pred = model(test_x_i, field_mask, new_field_mask=new_field_mask, mode='test')
#                 te_times.append(time.time() - t_now)
#                 te_mem = get_gpu_memory_map()[int(args.gpus)]
#                 te_mems.append(te_mem)
#                 if i >= 20:
#                     break
#         torch.cuda.empty_cache()
#
#     tr_time = np.mean(tr_times)
#     te_time = np.mean(te_times)
#     tr_mem = np.mean(tr_mems)
#     te_mem = np.mean(te_mems)
#     print(batch_size, tr_time, te_time, tr_mem, te_mem)