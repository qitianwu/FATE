import torch
import os
import math
from metric import *
from utils import *
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

ce = torch.nn.BCEWithLogitsLoss()

def train(model, field_mask, known_masks, train_x_i, train_y_i, optimizer_s, optimizer_f, args, device):
    model.train()
    optimizer_s.zero_grad()
    optimizer_f.zero_grad()
    loss_tr, reg_tr = 0, 0
    for k in range(len(known_masks)):
        logit, w_recon = model(train_x_i, field_mask, known_mask=known_masks[k], mode='train', args=args)
        loss_fit = ce(logit, train_y_i)
        loss = loss_fit
        loss.backward()
        optimizer_f.step()
    optimizer_s.step()
    loss_tr += loss_fit.item() * train_x_i.size(0)
    reg_tr += 0 * train_x_i.size(0)
    return loss_tr / train_x_i.size(0), reg_tr / train_x_i.size(0)

def validation(model, field_mask, new_field_mask, val_x, val_y, args, device):
    loss_val = 0
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        val_ind = torch.arange(val_x.size(0))
        for i in range(val_x.size(0) // args.batch_size + 1):
            val_ind_i = val_ind[i * args.batch_size: (i + 1) * args.batch_size]
            val_x_i, val_y_i = val_x[val_ind_i].to(device), val_y[val_ind_i].to(device)
            pred = model(val_x_i, field_mask, new_field_mask=new_field_mask, mode='test')
            loss = ce(pred, val_y_i)
            loss_val += loss.item() * val_x_i.size(0)
            preds.extend(pred.cpu().tolist())
            labels.extend(val_y_i.cpu().tolist())
        auc = roc_auc_score(labels, preds)

    return loss_val / val_x.size(0), auc

def evaluation(model, field_mask, new_field_mask, test_x, test_y, args, device):
    model.eval()
    loss_te = 0
    preds, labels = [], []
    with torch.no_grad():
        test_ind = torch.arange(test_x.size(0))
        for i in range(test_x.size(0) // args.batch_size + 1):
            test_ind_i = test_ind[i * args.batch_size: (i + 1) * args.batch_size]
            test_x_i, test_y_i = test_x[test_ind_i].to(device), test_y[test_ind_i].to(device)
            pred = model(test_x_i, field_mask, new_field_mask=new_field_mask, mode='test')
            loss = ce(pred, test_y_i)
            loss_te += loss.item() * test_x_i.size(0)
            preds.extend(pred.cpu().tolist())
            labels.extend(test_y_i.cpu().tolist())
        auc = roc_auc_score(labels, preds)
    return loss_te / test_x.size(0), auc

def save_model(model, path, name):
    path = os.path.join(path, name)
    torch.save(model.state_dict(), path)

def load_model(model, path, name):
    path = os.path.join(path, name)
    if name == 'train':
        model_dict = torch.load(path)
        model.load_state_dict(model_dict)
    elif name == 'pre':
        model_dict = torch.load(path)
        model.load_state_dict(model_dict)

        # pretrained_dict = torch.load(path)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
