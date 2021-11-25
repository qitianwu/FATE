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

parser = argparse.ArgumentParser(description='SARec')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='/home/wuqitian/FTrans/data/large', help='data_dir')
parser.add_argument('--dataset', type=str, default='criteo', help='dataset')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=300, help='epoch_num')
parser.add_argument('--embedding_size', type=int, default=10, help='embedding_size') # 64
parser.add_argument('--hidden_size', type=int, default=400, help='hidden_size')
parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio')
parser.add_argument('--observed_ratio', type=float, default=0.8, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.1, help='val_ratio')
parser.add_argument('--test_ratio', type=float, default=0.1, help='test_ratio')
parser.add_argument('--dropout', type=float, default=0., help='dropout prob')
parser.add_argument('--model', type=str, default='partial', help='feature extrapolation model')
parser.add_argument('--backbone', type=str, default='DeepFM', help='backbone model')
parser.add_argument('--split_mode', type=str, default='ratio', help='how to split the data')
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
x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.float32)
data_idx = torch.arange(0, x.size(0))
test_mask = torch.zeros(x.size(0), dtype=torch.bool)
field_mask = torch.zeros(x.size(1), dtype=torch.bool)
field_mask[field_ind] = True
new_field_mask = torch.zeros(x.size(1), dtype=torch.bool)
new_field_mask[new_field_ind] = True
day_range = np.array([5337126, 3870752, 3335302, 3363122, 3835892, 3225010, 5287222, 3832608, 4218938, 0], dtype=np.int)
day_range = np.cumsum(day_range, axis=0)
ratio_range = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# train_x, train_y, test_x, test_y = data_split(x, y, ratio=0.5)
print(field_mask.sum(), new_field_mask.sum())
field_num = x.size(1)
field_dims = torch.max(x, dim=0)[0] + 1
print(field_num, field_dims)

results = []

if args.backbone == 'DNN':
    model = DNN(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size, \
                dropout_prob=args.dropout, backbone=args.backbone, device=device).to(device)
    model.load_state_dict(torch.load('./model/dnn-{}-day.pkl'.format(args.dataset)))
elif args.backbone in ['DeepFM', 'xDeepFM']:
    model = DFM(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size, \
                dropout_prob=args.dropout, backbone=args.backbone, device=device).to(device)
    model.load_state_dict(torch.load('./model/deepfm-{}-day.pkl'.format(args.dataset)))
for i in range(0, 8):
    if args.split_mode == 'ratio':
        test_idx = data_idx[int(x.size(0) * ratio_range[i]): int(x.size(0) * ratio_range[i+1])]
    elif args.split_mode == 'day':
        test_idx = data_idx[day_range[i+1]: day_range[i+2]]
    test_mask[test_idx] = True
    test_x, test_y = x[test_mask], y[test_mask]
    print(test_x.size())
    loss_te, auc_te = evaluation(model, test_x, test_y, field_mask, new_field_mask, args, device)
    print('Test Split {}: Logloss {:.4f} AUC {:.4f} '. \
          format(i, loss_te, auc_te))
    results.append(auc_te)
results = np.array(results)
print('Overall Test AUC: {:.4f} + {:.4f}'.format(results.mean(), results.std()))