import numpy as np
import torch
import pickle
import os
import random
import pandas as pd
import math

day_range = np.array([5337126, 3870752, 3335302, 3363122, 3835892, 3225010, 5287222, 3832608, 4218938, 0], dtype=np.int)
day_range = np.cumsum(day_range, axis=0)
def data_split(x, train_ratio, field_ind, new_field_ind, test_ratio=None, val_ratio=None, mode='ratio'):
    data_idx = torch.arange(0, x.size(0))
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    field_mask = torch.zeros(x.size(1), dtype=torch.bool)
    field_mask[field_ind] = True
    new_field_mask = torch.zeros(x.size(1), dtype=torch.bool)
    new_field_mask[new_field_ind] = True
    if mode == 'ratio':
        train_idx = data_idx[:int(x.size(0) * train_ratio)]
        val_idx = data_idx[int(x.size(0) * train_ratio): int(x.size(0) * train_ratio) + int(x.size(0) * val_ratio)]
        test_idx = data_idx[int(x.size(0) * train_ratio) + int(x.size(0) * val_ratio): \
                            int(x.size(0) * train_ratio) + int(x.size(0) * val_ratio) + int(x.size(0) * test_ratio)]
        val_mask[val_idx] = True
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, val_mask, test_mask, field_mask, new_field_mask
    elif mode == 'day':
        train_idx = data_idx[:day_range[0]]
        val_idx = data_idx[day_range[0]: day_range[1]]
        test_idx = data_idx[day_range[1]: day_range[2]]
        val_mask[val_idx] = True
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, val_mask, test_mask, field_mask, new_field_mask

def get_known_mask(field_mask, known_num, mode, K=None):
    index = field_mask.nonzero().view(-1)
    unknown_num = index.size(0) - known_num
    known_masks = []
    if mode == 'leave-one-out':
        known_mask = field_mask.clone()
        index_mask = torch.randint(0, index.size(0), (unknown_num,)).view(-1)
        index_masked = index[index_mask]
        known_mask[index_masked] = False
        known_masks.append(known_mask)
    elif mode == 'k-fold':
        fold_num = math.ceil(index.size(0) / unknown_num)
        index = index[torch.randperm(index.size(0))]
        for i in range(fold_num):
            known_mask = torch.zeros_like(field_mask, dtype=torch.bool)
            index_mask = torch.ones_like(index, dtype=torch.bool)
            index_mask[int(i*unknown_num):int((i+1)*unknown_num)] = False
            index_masked = index[index_mask]
            known_mask[index_masked] = True
            known_masks.append(known_mask)
    elif mode == 'k-shot-random':
        for i in range(K):
            known_mask = field_mask.clone()
            index_mask = torch.randint(0, index.size(0), (unknown_num,)).view(-1)
            index_masked = index[index_mask]
            known_mask[index_masked] = False
            known_masks.append(known_mask)
    return known_masks

def drop_edge(edge_index, dropout_prob, device):
    drop_mask = (torch.FloatTensor(edge_index.size(1), 1).uniform_() > dropout_prob).view(-1).to(device)
    edge_index_new = edge_index[:, drop_mask]
    return edge_index_new

def randomize_node(node_feature, feature_num):
    data_num = node_feature.size(0) - feature_num
    node_feature[feature_num:] = torch.FloatTensor(data_num, node_feature.size(1)).normal_()
    return node_feature

def create_node(node_num, hidden_size):
    node_feature = torch.zeros(node_num, hidden_size, dtype=torch.float)
    return node_feature

def create_edge(x, filed_mask, feature_num, device):
    x_ = x[:, filed_mask]
    edge_source = torch.arange(0, x_.size(0)).to(device) + feature_num
    edge_source = edge_source.unsqueeze(1).repeat(1, x_.size(1)).reshape(-1) # [N*F']
    edge_end = x_.reshape(-1) # [N*F']
    edge_source_ = torch.cat([edge_source, edge_end], dim=0)
    edge_end_ = torch.cat([edge_end, edge_source], dim=0)
    edge_index = torch.stack([edge_source_, edge_end_], dim=0)  # [2, edge_num]
    return edge_index

def graph_generation(x, hidden_size):
    sample_num, feature_num = x.size()
    node_feature = create_node(sample_num+feature_num, hidden_size)
    edge_index = create_edge(x, feature_num)
    data = Data(x=node_feature, edge_index=edge_index)
    return data






