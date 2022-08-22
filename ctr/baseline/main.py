import pandas as pd
import numpy as np
import os
import argparse
import random
from model import DNN, DFM
from utils import data_split
from metric import *
from control import *

import torch
import torch.nn.functional as F

import logging
import pickle

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='../../data/large', help='data_dir')
parser.add_argument('--dataset', type=str, default='avazu', help='dataset')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay') # 1e-2 for criteo deepfm
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=100, help='epoch_num')
parser.add_argument('--embedding_size', type=int, default=10, help='embedding_size') # 64
parser.add_argument('--hidden_size', type=int, default=400, help='hidden_size')
parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio')
parser.add_argument('--observed_ratio', type=float, default=0.8, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.1, help='val_ratio')
parser.add_argument('--test_ratio', type=float, default=0.1, help='test_ratio')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
parser.add_argument('--model', type=str, default='partial', help='feature extrapolation model')
parser.add_argument('--backbone', type=str, default='DeepFM', help='backbone model')
parser.add_argument('--split_mode', type=str, default='day', help='how to split the data')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

fix_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'avazu':
    # 0 ~ 21 categorical
    field_ind = torch.tensor(list(range(0, 22)), dtype=torch.long) # 11
    new_field_ind = torch.tensor(list(range(22, 22)), dtype=torch.long)
elif args.dataset == 'criteo':
    # 0 ~ 12 continuous, 13 ~ 38 categorical
    field_ind = torch.tensor(list(range(0, 12)) + list(range(13, 39)), dtype=torch.long) # 7, 26
    new_field_ind = torch.tensor(list(range(12, 12)) + list(range(39, 39)), dtype=torch.long)

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
    if args.backbone == 'DNN':
        model = DNN(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size, \
                    dropout_prob=args.dropout, backbone=args.backbone, device=device).to(device)
    elif args.backbone in ['DeepFM', 'xDeepFM']:
        model = DFM(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size, \
                    dropout_prob=args.dropout, backbone=args.backbone, device=device).to(device)
    min_auc = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    # model.load_state_dict(torch.load('./model/tmp.pkl'))
    for i in range(args.epoch_num):
        loss_tr = train(model, train_x, train_y, field_mask, new_field_mask, optimizer, args, device)
        loss_val, auc_val = evaluation(model, val_x, val_y, field_mask, new_field_mask, args, device)
        loss_te, auc_te = evaluation(model, test_x, test_y, field_mask, new_field_mask, args, device)
        print(loss_tr, loss_val, loss_te, auc_val, auc_te)
        if auc_val > min_auc:
            min_auc = auc_val
            best_auc_te = auc_te
            best_logloss = loss_te
            torch.save(model.state_dict(), './model/deepfm-avazu-day.pkl')
    print('Test AUC: {:.4f} Test Logloss: {:.4f}'.format(best_auc_te, best_logloss))
    results.append([best_auc_te, best_logloss])