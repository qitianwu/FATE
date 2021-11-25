import numpy as np
import torch
import pickle
import os
import random
import pandas as pd
import math
from torch_geometric.data import Data

def data_split(x, train_ratio, val_ratio, feature_ratio, new_feature_ratio, split_thre=None):
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    if split_thre is not None:
        train_data_idx = torch.randperm(split_thre)
        val_idx = train_data_idx[:int(split_thre * val_ratio)]
        train_idx = train_data_idx[int(split_thre * val_ratio):]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[split_thre:] = True
    else:
        data_idx = torch.randperm(x.size(0))
        train_idx = data_idx[:int(x.size(0) * train_ratio)]
        val_idx = data_idx[int(x.size(0) * train_ratio): int(x.size(0) * train_ratio) + int(x.size(0) * val_ratio)]
        test_idx = data_idx[int(x.size(0) * train_ratio) + int(x.size(0) * val_ratio):]
        val_mask[val_idx] = True
        train_mask[train_idx] = True
        test_mask[test_idx] = True

    all_feature_idx = torch.randperm(x.size(1))
    feature_mask = torch.zeros(x.size(1), dtype=torch.bool)
    new_feature_mask = torch.zeros(x.size(1), dtype=torch.bool)
    feature_idx = all_feature_idx[:int(x.size(1) * feature_ratio)]
    new_feature_idx = all_feature_idx[-int(x.size(1) * new_feature_ratio): ]
    feature_mask[feature_idx] = True
    new_feature_mask[new_feature_idx] = True
    return train_mask, val_mask, test_mask, feature_mask, new_feature_mask

def feature_ratio_cut(feature_mask, ratio):
    idx = feature_mask.nonzero().view(-1)
    idx = idx[torch.randperm(idx.size(0))]
    idx_ = idx[:int(idx.size(0) * ratio)]
    feature_mask_ = torch.zeros_like(feature_mask)
    feature_mask_[idx_] = True
    return feature_mask_

def get_known_mask(feature_mask, known_ratio, mode, K=None):
    index = feature_mask.nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    known_masks = []
    if mode == 'leave-one-out':
        known_mask = torch.zeros_like(feature_mask, dtype=torch.bool)
        known_idx = index[:int(index.size(0) * known_ratio)]
        known_mask[known_idx] = True
        known_masks.append(known_mask)
    if mode == 'k-fold':
        fold_size = (1 - known_ratio) * index.size(0)
        fold_num = math.ceil(1. / (1 - known_ratio))
        known_masks = []
        for i in range(fold_num):
            known_mask = torch.zeros_like(feature_mask, dtype=torch.bool)
            index_mask = torch.ones_like(index, dtype=torch.bool)
            index_mask[int(i*fold_size):int((i+1)*fold_size)] = False
            index_masked = index[index_mask]
            known_mask[index_masked] = True
            known_masks.append(known_mask)
    if mode == 'k-shot-random':
        for i in range(K):
            index = index[torch.randperm(index.size(0))]
            known_mask = torch.zeros_like(feature_mask, dtype=torch.bool)
            known_idx = index[:int(index.size(0) * known_ratio)]
            known_mask[known_idx] = True
            known_masks.append(known_mask)
    return known_masks

def drop_edge(edge_index, dropout_prob):
    drop_mask = (torch.FloatTensor(edge_index.size(1), 1).uniform_() < dropout_prob).view(-1)
    edge_index_new = edge_index[:, drop_mask]
    return edge_index_new

def randomize_node(node_feature, feature_num):
    data_num = node_feature.size(0) - feature_num
    node_feature[feature_num:] = torch.FloatTensor(data_num, node_feature.size(1)).normal_()
    return node_feature

def create_node(node_num, hidden_size):
    node_feature = torch.zeros(node_num, hidden_size, dtype=torch.float)
    return node_feature

def create_edge(x, feature_num, feature_mask=None, train_val_mask=None, mode='more'):
    if mode == 'less':
        x_ = x.clone()
        _feature_index = (~feature_mask).nonzero().reshape(1, -1)
        _train_val_index = train_val_mask.nonzero().reshape(-1, 1)
        _feature_index.repeat(_train_val_index.size(0), 1).reshape(-1)
        _train_val_index.repeat(1, _feature_index.size(1)).reshape(-1)
        x_[_train_val_index, _feature_index] = 0.
        init_source, init_end = torch.nonzero(x_).transpose(0, 1)
    else:
        init_source, init_end = torch.nonzero(x).transpose(0, 1)
    edge_source = init_source + feature_num
    edge_end = init_end
    edge_source_ = torch.cat([edge_source, edge_end], dim=0)
    edge_end_ = torch.cat([edge_end, edge_source], dim=0)
    edge_index = torch.stack([edge_source_, edge_end_], dim=0) # [2, edge_num]
    return edge_index

def graph_generation(x, hidden_size):
    sample_num, feature_num = x.size()
    node_feature = create_node(sample_num+feature_num, hidden_size)
    edge_index = create_edge(x, feature_num)
    data = Data(x=node_feature, edge_index=edge_index)
    return data






