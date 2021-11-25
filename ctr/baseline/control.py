import torch
import os
import math
from metric import *
from utils import *
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

ce = torch.nn.BCEWithLogitsLoss()

def train(model, train_x, train_y, field_mask, new_field_mask, optimizer, args, device):
    model.train()
    optimizer.zero_grad()
    loss_tr = 0
    train_ind = torch.randperm(train_x.size(0))
    for i in range(train_x.size(0) // args.batch_size + 1):
        train_ind_i = train_ind[i*args.batch_size : (i+1)*args.batch_size]
        train_x_i, train_y_i = train_x[train_ind_i].to(device), train_y[train_ind_i].to(device)
        pred = model(train_x_i, field_mask, new_field_mask, mode='train').view(-1)
        loss = ce(pred, train_y_i)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item() * train_x_i.size(0)
    return loss_tr / train_x.size(0)

def evaluation(model, test_x, test_y, field_mask, new_field_mask, args, device):
    model.eval()
    loss_te, num = 0, 0
    preds, labels = list(), list()
    with torch.no_grad():
        for i in range(test_x.size(0) // args.batch_size + 1):
            test_x_i = test_x[i*args.batch_size : (i+1)*args.batch_size].to(device)
            test_y_i = test_y[i*args.batch_size : (i+1)*args.batch_size].to(device)
            pred = model(test_x_i, field_mask, new_field_mask, mode='test', model=args.model).view(-1)
            loss = ce(pred, test_y_i)
            loss_te += loss.item() * test_x_i.size(0)
            preds.extend(pred.cpu().tolist())
            labels.extend(test_y_i.cpu().tolist())
        auc = roc_auc_score(labels, preds)
        return loss_te / test_x.size(0), auc