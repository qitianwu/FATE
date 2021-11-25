import torch
import os
import math
from metric import *
from utils import *
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

ce = torch.nn.CrossEntropyLoss()

def train(model, train_x, train_y, feature_mask, optimizer, args, device):
    model.train()
    optimizer.zero_grad()
    logit = model(train_x, feature_mask, mode='train')
    loss = ce(logit, train_y)
    loss.backward()
    optimizer.step()
    if args.multi_class:
        pred = torch.argmax(logit, dim=1)
        acc = accuracy_calc(pred, train_y)
        return loss.item(), acc.item()
    else:
        pred = logit[:, 1]
        auc = roc_auc_score(train_y.cpu().tolist(), pred.cpu().tolist())
        return loss.item(), auc

def evaluation(model, test_x, test_y, feature_mask, new_feature_mask, args, device):
    model.eval()
    with torch.no_grad():
        logit = model(test_x, feature_mask, new_feature_mask, mode='test', model=args.model)
        loss = ce(logit, test_y).item()
        if args.multi_class:
            pred = torch.argmax(logit, dim=1)
            acc = accuracy_calc(pred, test_y)
            return loss, acc.item()
        else:
            pred = logit[:, 1]
            auc = roc_auc_score(test_y.cpu().tolist(), pred.cpu().tolist())
            return loss, auc.item()