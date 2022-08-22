import pandas as pd
import numpy as np
import os
import argparse
import random
from model import Model
from metric import *
from utils import get_known_mask, data_split, create_edge
from control import *
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

import logging

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='FATE-NN')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='../../data/large', help='data_dir')
parser.add_argument('--dataset', type=str, default='criteo', help='dataset')
parser.add_argument('--lr1', type=float, default=1e-4, help='learning_rate for pretrain')
parser.add_argument('--wd1', type=float, default=0., help='weight_decay for pretrain')
parser.add_argument('--lr2', type=float, default=1e-4, help='learning_rate for train')
parser.add_argument('--wd2', type=float, default=0., help='weight_decay for train')
parser.add_argument('--con_reg', type=float, default=0., help='contrastive regularization')
parser.add_argument('--batch_size', type=int, default=100000, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=100, help='epoch_num')
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
if args.split_mode == 'day':
    train_mask, val_mask, test_mask, field_mask, new_field_mask = data_split(x, train_ratio=args.train_ratio, \
                            field_ind=field_ind, val_ratio=args.val_ratio, new_field_ind=new_field_ind, mode='day')
elif args.split_mode == 'ratio':
    train_mask, val_mask, test_mask, field_mask, new_field_mask = data_split(x, train_ratio=args.train_ratio, \
                                                                                   field_ind=field_ind,
                                                                                    test_ratio=args.test_ratio,
                                                                                   val_ratio=args.val_ratio,
                                                                                   new_field_ind=new_field_ind,
                                                                                   mode='ratio')
train_x, train_y = x[train_mask], y[train_mask]
val_x, val_y = x[val_mask], y[val_mask]
test_x, test_y = x[test_mask], y[test_mask]
print("observed/unobserved field num: {}/{}".format(field_mask.sum(), new_field_mask.sum()))
field_dims = torch.max(x, dim=0)[0] + 1
print("field dims: {}".format(field_dims))
field_dims2 = torch.max(train_x, dim=0)[0] + 1
print("train field dims: {}".format(field_dims2))
print("train/val/test instance num: {}/{}/{}".format(train_x.size(), val_x.size(), test_x.size()))

results = []

for _ in range(5):
    model = Model(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
              gnn_layer_num=args.gnn_layer_num, graphconv=args.graphconv, dropout_prob=args.dropout,
                 backbone=args.backbone, device=device).to(device)
    max_auc = 0.
    optimizer_f = torch.optim.Adam(model.param_bb, lr=args.lr1, weight_decay=args.wd1)
    optimizer_s = torch.optim.Adam(model.param_sn, lr=args.lr2, weight_decay=args.wd2)
    optimizer_nn = torch.optim.Adam(model.param_nn, lr=1e-3, weight_decay=0.)
    if args.training_mode == 'leave-one-out':
        known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode)
    for e in range(args.epoch_num):
        if args.training_mode == 'k-fold':
            known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode)
        if args.training_mode == 'k-shot-random':
            known_masks = get_known_mask(field_mask, known_num=args.known_num, mode=args.training_mode, K=5)
        train_ind = torch.randperm(train_x.size(0))
        for i in range(train_x.size(0) // args.batch_size + 1):
            train_ind_i = train_ind[i * args.batch_size: (i + 1) * args.batch_size]
            train_x_i, train_y_i = train_x[train_ind_i].to(device), train_y[train_ind_i].to(device)
            loss_tr, loss_recon = train(model, field_mask, known_masks, train_x_i, train_y_i, optimizer_s, optimizer_f, args, device)
            if i % 10 == 0:
                loss_val, metric_val = validation(model, field_mask, new_field_mask, val_x, val_y, args, device)
                loss_te, metric_te = evaluation(model, field_mask, new_field_mask, test_x, test_y, args, device)
                print('Epoch {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'. \
                      format(e, loss_tr, loss_val, loss_te, loss_recon, metric_val, metric_te))
                if metric_val > max_auc:
                    max_auc = metric_val
                    best_auc_te = metric_te
                    best_logloss = loss_te
                    torch.save(model.state_dict(), './model/lr-criteo-day.pkl')
    print('Test AUC: {:.4f} Test Logloss: {:.4f}'.format(best_auc_te, best_logloss))
    results.append([best_auc_te, best_logloss])