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

parser = argparse.ArgumentParser(description='SARec')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='/home/wuqitian/FTrans/data/large', help='data_dir')
parser.add_argument('--dataset', type=str, default='criteo', help='dataset')
parser.add_argument('--lr1', type=float, default=1e-4, help='learning_rate for pretrain') # 0.0001
parser.add_argument('--wd1', type=float, default=0., help='weight_decay for pretrain')
parser.add_argument('--lr2', type=float, default=1e-4, help='learning_rate for train')
parser.add_argument('--wd2', type=float, default=0., help='weight_decay for train') # 0.001
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
parser.add_argument('--dropedge_prob', type=float, default=0.5, help='known_ratio') # 0. for avazu, 0.3 for criteo
parser.add_argument('--training_mode', type=str, default='k-shot-random', help='training_mode')
parser.add_argument('--data_mode', type=str, default='less', help='training_mode')
parser.add_argument('--backbone', type=str, default='DeepFM', help='backbone model')
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
data_idx = torch.arange(0, x.size(0))
test_mask = torch.zeros(x.size(0), dtype=torch.bool)
field_mask = torch.zeros(x.size(1), dtype=torch.bool)
field_mask[field_ind] = True
new_field_mask = torch.zeros(x.size(1), dtype=torch.bool)
new_field_mask[new_field_ind] = True
day_range = np.array([5337126, 3870752, 3335302, 3363122, 3835892, 3225010, 5287222, 3832608, 4218938, 0], dtype=np.int)
day_range = np.cumsum(day_range, axis=0)
ratio_range = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(field_mask.sum(), new_field_mask.sum())
field_num = x.size(1)
field_dims = torch.max(x, dim=0)[0] + 1
print(field_num, field_dims)

results = []

model = Model(field_dims=field_dims, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
          gnn_layer_num=args.gnn_layer_num, graphconv=args.graphconv, dropout_prob=args.dropout,
             backbone=args.backbone, device=device).to(device)
model.load_state_dict(torch.load('./model/deepfm-{}-day.pkl'.format(args.dataset)))

for i in range(0, 8):
    if args.split_mode == 'ratio':
        test_idx = data_idx[int(x.size(0) * ratio_range[i]): int(x.size(0) * ratio_range[i+1])]
    elif args.split_mode == 'day':
        test_idx = data_idx[day_range[i+1]: day_range[i+2]]
    test_mask[test_idx] = True
    test_x, test_y = x[test_mask], y[test_mask]
    print(test_x.size())
    loss_te, metric_te = evaluation(model, field_mask, new_field_mask, test_x, test_y, args, device)
    print('Test Split {}: Logloss {:.4f} AUC {:.4f} '. \
                      format(i, loss_te, metric_te))
    results.append(metric_te)
results = np.array(results)
print('Overall Test AUC: {:.4f} + {:.4f}'.format(results.mean(), results.std()))

# logging.basicConfig(level=logging.INFO, filename='../log3/{}.log'.format(args.dataset), format='%(message)s')
# results = np.array(results)
# logging.info("ours {}: {:.4f} + {:.4f}, {}".format(args.feature_ratio, np.mean(results), np.std(results), results))