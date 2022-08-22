import pandas as pd
import numpy as np
import os
import argparse
import random
from model import LR
from utils import *
from metric import *
from control import *
import torch
import torch.nn.functional as F

import pickle
import logging

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--gpus', default='1', help='gpus')
parser.add_argument('--data_dir', default='../../data/uci', help='data_dir')
parser.add_argument('--dataset', type=str, default='github', help='dataset')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--epoch_num', type=int, default=200, help='epoch_num')
parser.add_argument('--hidden_size', type=int, default=8, help='hidden_size')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--feature_ratio', type=float, default=0.5, help='feature_ratio')
parser.add_argument('--new_feature_ratio', type=float, default=0.5, help='new_feature_ratio')
parser.add_argument('--observed_ratio', type=float, default=0.8, help='observed_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--model', type=str, default='all', help='baseline model')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

#fix_seed(args.seed)
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
print("Feature/Observed/Unobserved: {}/{}/{}".format(feat_num, feature_mask.sum(), new_feature_mask.sum()))
print("Train/Val/Test Size: {}/{}/{}:".format(train_x.size(), val_x.size(), test_x.size()))
print("Class Num: {}".format(class_num))

train_x, train_y = train_x.to(device), train_y.to(device)
val_x, val_y = val_x.to(device), val_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)

results = []
for _ in range(5):
    model = NN(feat_num=feat_num, hidden_size=args.hidden_size, class_num=class_num, device=device).to(device)
    min_loss = 10.0
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    for i in range(args.epoch_num):
        loss_tr, metric_tr = train(model, train_x, train_y, feature_mask, optimizer, args, device)
        loss_val, metric_val = evaluation(model, val_x, val_y, feature_mask, new_feature_mask, args, device)
        loss_te, metric_te = evaluation(model, test_x, test_y, feature_mask, new_feature_mask, args, device)
        print(loss_tr, loss_te, metric_tr, metric_te)
        if loss_val < min_loss:
            min_loss = loss_val
            best_metric_te = metric_te
            # torch.save(model.state_dict(), '../checkpoint/dnn-{}-{}-{}.pkl'.format(args.dataset, args.train_ratio, args.feature_ratio))
    if args.multi_class:
        print('Test Acc: {:.4f}'.format(best_metric_te))
    else:
        print('Test AUC: {:.4f}'.format(best_metric_te))
    results.append(best_metric_te)
print(np.mean(results), np.std(results))

# logging.basicConfig(level=logging.INFO, filename='../log/{}.log'.format(args.dataset), format='%(message)s')
# results = np.array(results)
# logging.info("{} {} {} {}: {:.4f} + {:.4f}, {}".format(args.model, args.train_ratio, args.feature_ratio, args.new_feature_ratio, np.mean(results), np.std(results), results))