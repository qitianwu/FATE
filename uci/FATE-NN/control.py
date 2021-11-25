import torch
import os
import math
from metric import *
from utils import *
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

ce = torch.nn.CrossEntropyLoss()

def train(model, feature_mask, train_mask, known_masks, train_x, train_y, edge_index, optimizer_s, optimizer_f, args, device):
    model.train()
    optimizer_s.zero_grad()
    optimizer_f.zero_grad()
    for i in range(len(known_masks)):
        w = model.get_embedding().cpu()
        if args.is_detach:
            w_ = w.detach()
            node_feature = torch.cat([ w_, torch.zeros(train_mask.size(0), args.hidden_size, dtype=w_.dtype) ], dim=0)
        else:
            node_feature = torch.cat([w, torch.zeros(train_mask.size(0), args.hidden_size, dtype=w.dtype)], dim=0)
        data = Data(x=node_feature, edge_index=edge_index).to(device)
        data.train_node_mask = torch.cat([feature_mask, train_mask], dim=0)
        data.train_edge_index, _ = subgraph(data.train_node_mask, data.edge_index, relabel_nodes=True)
        data.train_edge_index = drop_edge(data.train_edge_index, args.dropedge_prob)
        logit, w_recon = model(train_x, feature_mask, data, known_masks[i], mode='train')
        loss_fit = ce(logit, train_y)
        loss = loss_fit + args.con_reg*w_recon
        loss.backward()
        optimizer_f.step()
    optimizer_s.step()
    if args.multi_class:
        pred = torch.argmax(logit, dim=1)
        acc = accuracy_calc(pred, train_y)
        return loss_fit.item(), w_recon.item(), acc.item()
    else:
        pred = logit[:, 1]
        auc = roc_auc_score(train_y.cpu().tolist(), pred.cpu().tolist())
        return loss_fit.item(), w_recon.item(), auc

def validation(model, feature_mask, train_val_mask, val_x, val_y, edge_index, args, device):
    fold_num = math.ceil(1. / (1 - args.known_ratio))
    known_masks = get_known_mask(feature_mask, known_ratio=args.known_ratio, mode='k-fold')
    metrics, losses = [], []
    with torch.no_grad():
        for i in range(fold_num):
            w = model.get_embedding().cpu()
            node_feature = torch.cat([w, torch.zeros(train_val_mask.size(0), args.hidden_size, dtype=w.dtype)], dim=0)
            data = Data(x=node_feature, edge_index=edge_index).to(device)
            data.train_node_mask = torch.cat([feature_mask, train_val_mask], dim=0)
            data.train_edge_index, _ = subgraph(data.train_node_mask, data.edge_index, relabel_nodes=True)
            data.train_edge_index = drop_edge(data.train_edge_index, args.dropedge_prob)
            logit, _ = model(x=val_x, feature_mask=feature_mask, data=data, known_mask=known_masks[i], mode='train')
            loss = ce(logit, val_y)
            losses.append(loss.item())
            if args.multi_class:
                pred = torch.argmax(logit, dim=1)
                acc = accuracy_calc(pred, val_y).item()
                metrics.append(acc)
            else:
                pred = logit[:, 1]
                auc = roc_auc_score(val_y.cpu().tolist(), pred.cpu().tolist())
                metrics.append(auc)
    return sum(losses) / fold_num, sum(metrics) / fold_num

def evaluation(model, feature_mask, train_mask, new_feature_mask, test_x, test_y, edge_index, args, device):
    model.eval()
    with torch.no_grad():
        w = model.get_embedding().cpu()
        node_feature = torch.cat([w, torch.zeros(train_mask.size(0), args.hidden_size, dtype=w.dtype)], dim=0)
        data = Data(x=node_feature, edge_index=edge_index).to(device)
        logit = model(x=test_x, feature_mask=feature_mask, data=data, new_feature_mask=new_feature_mask, mode='test')
        loss = ce(logit, test_y).item()
        if args.multi_class:
            pred = torch.argmax(logit, dim=1)
            acc = accuracy_calc(pred, test_y)
            return loss, acc.item()
        else:
            pred = logit[:, 1]
            auc = roc_auc_score(test_y.cpu().tolist(), pred.cpu().tolist())
            return loss, auc

def save_model(model, path, name):
    path = os.path.join(path, name)
    torch.save(model.state_dict(), path)

def load_model(model, path, name):
    path = os.path.join(path, name)
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
