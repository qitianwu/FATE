import pandas as pd
import numpy as np
import os
import argparse
import random
from model import Model
from metric import *
from utils import *
from control import *

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

import pickle
import logging

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='FATE-NN')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--data_dir', default='../../data/uci', help='data_dir')
parser.add_argument('--dataset', type=str, default='protein', help='dataset')
parser.add_argument('--lr1', type=float, default=1e-2, help='learning_rate for pretrain') # 1e-2
parser.add_argument('--wd1', type=float, default=0., help='weight_decay for pretrain')
parser.add_argument('--lr2', type=float, default=1e-3, help='learning_rate for train') # 1e-3
parser.add_argument('--wd2', type=float, default=0., help='weight_decay for train')
parser.add_argument('--con_reg', type=float, default=0., help='contrastive regularization')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=200, help='epoch_num') # 100
parser.add_argument('--hidden_size', type=int, default=8, help='hidden_size')
parser.add_argument('--is_detach', type=bool, default=False, help='whether to detach W')
parser.add_argument('--load_pretrain', type=bool, default=False, help='whether to detach W')
parser.add_argument('--gnn_layer_num', type=int, default=4, help='gnn_layer_num') # 4
parser.add_argument('--graphconv', type=str, default='GCN', help='graphconv type')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--feature_ratio', type=float, default=0.5, help='train_ratio')
parser.add_argument('--new_feature_ratio', type=float, default=0.5, help='train_ratio')
parser.add_argument('--known_ratio', type=float, default=0.8, help='known_ratio') # 0。8
parser.add_argument('--val_ratio', type=float, default=0.2, help='known_ratio')
parser.add_argument('--dropedge_prob', type=float, default=0.5, help='known_ratio') # 0.5
parser.add_argument('--training_mode', type=str, default='k-fold', help='training_mode')
parser.add_argument('--K', type=int, default=10, help='k shot random')
parser.add_argument('--data_mode', type=str, default='less', help='training_mode')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

# fix_seed(args.seed)
device = f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

multi_class = ['gene', 'drive', 'protein', 'robot', 'calls']
binary_class = ['github']
if args.dataset in multi_class:
    args.multi_class = True
elif args.dataset in binary_class:
    args.multi_class = False
else:
    raise NotImplementedError

data_dir = os.path.join(args.data_dir, args.dataset)
data = np.loadtxt(data_dir + '/data.csv', delimiter=',', dtype=float)
y, x = data[:, 0], data[:, 1:]
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

split_file_path = data_dir + '/split_{}_{}.pkl'.format(args.train_ratio, args.feature_ratio)
if os.path.exists(split_file_path):
    print("Split exists, using the saved split..")
    with open(split_file_path, 'rb') as f:
        train_mask = pickle.load(f)
        val_mask = pickle.load(f)
        test_mask = pickle.load(f)
        feature_mask = pickle.load(f)
        new_feature_mask = pickle.load(f)
else:
    print("Split does not exists, create new split..")
    train_mask, val_mask, test_mask, feature_mask, new_feature_mask = data_split(x, train_ratio=args.train_ratio, \
                                                                                 val_ratio=args.val_ratio,
                                                                                 feature_ratio=args.feature_ratio,
                                                                                 new_feature_ratio=args.new_feature_ratio)
    with open(split_file_path, 'wb') as f:
        pickle.dump(train_mask, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_mask, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_mask, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feature_mask, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(new_feature_mask, f, pickle.HIGHEST_PROTOCOL)

train_x, train_y = x[train_mask], y[train_mask]
val_x, val_y = x[val_mask], y[val_mask]
test_x, test_y = x[test_mask], y[test_mask]
feat_num, class_num = x.size(1), int(y.max()) + 1
print("Feature/Observed/Unobserved/Known Num: {}/{}/{}/{}".format(feat_num, feature_mask.sum(), new_feature_mask.sum(),
                                                                       feature_mask.sum() * args.known_ratio))
print("Train/Val/Test Size: {}/{}/{}:".format(train_x.size(), val_x.size(), test_x.size()))
print("Class Num: {}".format(class_num))

train_x, train_y = train_x.to(device), train_y.to(device)
val_x, val_y = val_x.to(device), val_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)
train_val_mask = train_mask + val_mask

edge_index = create_edge(x, feat_num, feature_mask, train_val_mask, mode=args.data_mode)
results = []

for _ in range(5):
    model = Model(feat_num=feat_num, hidden_size=args.hidden_size, class_num=class_num,
              gnn_layer_num=args.gnn_layer_num, graphconv=args.graphconv, device=device).to(device)
    min_loss = 10.0
    optimizer_f = torch.optim.Adam(model.param_bb, lr=args.lr1, weight_decay=args.wd1)
    optimizer_s = torch.optim.Adam(model.param_sn, lr=args.lr2, weight_decay=args.wd2)
    if args.training_mode == 'leave-one-out':
        known_masks = get_known_mask(feature_mask, known_ratio=args.known_ratio, mode=args.training_mode)
    for i in range(args.epoch_num):
        if args.training_mode == 'k-fold':
            known_masks = get_known_mask(feature_mask, known_ratio=args.known_ratio, mode=args.training_mode)
        if args.training_mode == 'k-shot-random':
            known_masks = get_known_mask(feature_mask, known_ratio=args.known_ratio, mode=args.training_mode, K=args.K)
        loss_tr, loss_recon, metric_tr = train(model, feature_mask, train_mask, known_masks, train_x, train_y, edge_index, optimizer_s, optimizer_f, args, device)
        loss_val, metric_val = validation(model, feature_mask, train_val_mask, val_x, val_y, edge_index, args, device)
        loss_te, metric_te = evaluation(model, feature_mask, train_mask, new_feature_mask, test_x, test_y, edge_index, args, device)
        print('Epoch {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'. \
              format(i, loss_tr, loss_val, loss_te, loss_recon, metric_tr, metric_val, metric_te))
        if loss_val < min_loss:
            min_loss = loss_val
            best_metric_te = metric_te
            # torch.save(model.state_dict(), './model/{}-{}-{}.pkl'.format(args.dataset, args.train_ratio, args.feature_ratio))
    if args.multi_class:
        print('Test Acc: {:.4f}'.format(best_metric_te))
    else:
        print('Test AUC: {:.4f}'.format(best_metric_te))
    results.append(best_metric_te)
print(np.mean(results), np.std(results))

# logging.basicConfig(level=logging.INFO, filename='../log/{}.log'.format(args.dataset), format='%(message)s')
# results = np.array(results)
# logging.info("{} {} {} {}: {:.4f} + {:.4f}, {}".format(args.model, args.train_ratio, args.feature_ratio, args.new_feature_ratio, np.mean(results), np.std(results), results))