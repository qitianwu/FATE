import numpy as np
import torch
import pickle
import os
import random
import pandas as pd

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