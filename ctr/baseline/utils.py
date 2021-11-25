import numpy as np
import torch
import pickle
import os
import random
import pandas as pd

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