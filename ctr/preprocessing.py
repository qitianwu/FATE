import numpy as np
import torch
import pickle
import os
import random
import pandas as pd
import tqdm
import math

def _build_map(d):
    num = 0
    m = {}
    for k, r in enumerate(d):
        if r in m.keys():
            d[k] = m[r]
        else:
            m[r] = num
            d[k] = num
            num += 1
    return d, m

def _one_hot(d):
    dim = np.max(d) + 1
    d_ = np.zeros((d.shape[0], dim), dtype=int)
    d_[np.arange(d.shape[0]), d] = 1
    return d_

def _one_hot_all(x):
    output = []
    for i in range(x.shape[1]):
        x_i = x[:, i]
        x_i_ = _one_hot(x_i)
        if i <= 0:
            x_ = x_i_
        else:
            x_ = np.concatenate([x_, x_i_], axis=1)
    return x_

def _discretize(a):
    for i in range(a.shape[1]):
        b = a[:, i]
        min, max = np.min(b), np.max(b)
        b = (b-min) / (max - min + 1e-16)
        a[:, i] = np.floor(b * 10)
    return a

def _fillna(df, dtype):
    if dtype in ['float64', 'float32', 'float']:
        mean_val = df.mean()
        return df.fillna(mean_val)
    elif dtype in ['int64', 'int32', 'int']:
        mode_val = df.mode()
        return df.fillna(mode_val)
    else:
        return df.fillna(method='pad')

def _convert_dis(x):
    if x == '':
        return ''
    v = int(x)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)

def data_preprocess(data_dir, dataset, feat_threshold=0):
    data_dir = os.path.join(data_dir, dataset)
    field_times = []
    x, y = [], []
    count = 0
    if dataset == 'avazu':
        with open(data_dir + '/train.txt', "r") as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i].strip().split(',')
            x_i = line[2:]
            for k, x_ik in enumerate(x_i):
                if i <= 1:
                    t_k = {}
                    t_k[x_ik] = 1
                    field_times.append(t_k)
                else:
                    t_k = field_times[k]
                    if x_ik in t_k.keys():
                        t_k[x_ik] += 1
                    else:
                        t_k[x_ik] = 1
                    field_times[k] = t_k
        field_times_min = [min(field_times[k].values()) for k in range(len(field_times))]
        print(field_times_min)
        field_dims, field_maps = [], []
        for k in range(len(field_times)):
            field_dims += [1  if field_times_min[k] < feat_threshold else 0]
            field_maps.append({})
        for i in range(1, len(lines)):
            line = lines[i].strip().split(',')
            x_i, y_i = line[2:], int(line[1])
            x_i_new = []
            for k, x_ik in enumerate(x_i):
                t_k, m_k = field_times[k], field_maps[k]
                if t_k[x_ik] < feat_threshold:
                    x_i_new.append(0)
                else:
                    if x_ik in m_k.keys():
                        x_i_new.append(m_k[x_ik])
                    else:
                        field_dims[k] += 1
                        m_k[x_ik] = field_dims[k]
                        field_maps[k] = m_k
                        x_i_new.append(m_k[x_ik])
            x.append(x_i_new)
            y.append(y_i)
            count += 1
            if count % 100000 == 0:
                print(field_dims)
        for k in range(len(field_times)):
            if field_times_min[k] >= feat_threshold:
                field_dims[k] += 1
        print(x[:5])
        print(y[:5])
        print(field_dims)
        if feat_threshold <= 0:
            with open(data_dir + '/data.pkl', 'wb') as f:
                pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(data_dir + '/data_filter{}.pkl'.format(feat_threshold), 'wb') as f:
                pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

    if dataset == 'criteo':
        with open(data_dir + '/train.txt', "r") as f:
            lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i].split('\t')
            x_i = line[1:]
            for k, x_ik in enumerate(x_i):
                x_ik = x_ik.replace("\n", "")
                if k < 13:
                    x_ik = _convert_dis(x_ik)
                if i <= 0:
                    t_k = {}
                    t_k[x_ik] = 1
                    field_times.append(t_k)
                    continue
                # if x_ik == '':
                #     continue
                t_k = field_times[k]
                if x_ik in t_k.keys():
                    t_k[x_ik] += 1
                else:
                    t_k[x_ik] = 1
                field_times[k] = t_k
            count += 1
            if count % 100000 == 0:
                print(count)
        field_times_min = [min(field_times[k].values()) for k in range(len(field_times))]
        print(field_times_min)

        field_keys_max = []
        for k in range(len(field_times)):
            t_k = field_times[k]
            times_k = list(t_k.values())
            ind = times_k.index(max(times_k))
            keys_k = list(t_k.keys())
            field_keys_max.append(keys_k[ind])
        field_dims, field_maps = [], []
        for k in range(len(field_times)):
            field_dims += [1 if field_times_min[k] < feat_threshold else 0]
            field_maps.append({})
        for i in range(0, len(lines)):
            line = lines[i].split('\t')
            x_i, y_i = line[1:], int(line[0])
            x_i_new = []
            for k, x_ik in enumerate(x_i):
                x_ik = x_ik.replace("\n", "")
                if k < 13:
                    x_ik = _convert_dis(x_ik)
                t_k, m_k = field_times[k], field_maps[k]
                #if x_ik == '':
                #    x_ik = field_keys_max[k]
                if t_k[x_ik] < feat_threshold:
                    x_i_new.append(0)
                else:
                    if x_ik in m_k.keys():
                        x_i_new.append(m_k[x_ik])
                    else:
                        field_dims[k] += 1
                        m_k[x_ik] = field_dims[k]
                        field_maps[k] = m_k
                        x_i_new.append(m_k[x_ik])
            x.append(x_i_new)
            y.append(y_i)
            count += 1
            if count % 100000 == 0:
                print(field_dims)
        for k in range(len(field_times)):
            if field_times_min[k] >= feat_threshold:
                field_dims[k] += 1
        print(field_dims)
        if feat_threshold <= 0:
            with open(data_dir + '/data.pkl', 'wb') as f:
                pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(data_dir + '/data_filter{}.pkl'.format(feat_threshold), 'wb') as f:
                pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

def subset_extract(data_dir, dataset, num=10000):
    data_dir = os.path.join(data_dir, dataset)
    with open(data_dir + '/data.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    x_, y_ = x[:num], y[:num]
    with open(data_dir + '/subset.pkl', 'wb') as f:
        pickle.dump(x_, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_, f, pickle.HIGHEST_PROTOCOL)

def get_class_num(data_dir, dataset):
    data_dir = os.path.join(data_dir, dataset)
    with open(data_dir + '/data_filter4.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    num1, num2 = 0, 0
    for i in range(len(y)):
        num1 += 1 if y[i]==1 else 0
        num2 += 1 if y[i]==0 else 0
    print(num1, num2)

def get_day_range(data_dir, dataset):
    data_dir = os.path.join(data_dir, dataset)
    with open(data_dir + '/data_filter4.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    x = np.array(x, dtype=np.int)
    y = np.linspace(24, 240, 10)
    num = [x[x[:, 0]>int(y[k])].shape[0] for k in range(y.shape[0])]
    num += [0]
    num2 = [num[k]-num[k+1] for k in range(len(num)-1)]
    print(num2)

def get_feature_number(data_dir, dataset):
    data_dir = os.path.join(data_dir, dataset)
    with open(data_dir + '/data_filter4.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    x = np.array(x, dtype=np.int)
    field_num = x.shape[1]
    field_dims = np.max(x, axis=0) + 1
    print(field_num, field_dims, field_dims.sum())

def get_feature_numbers(data_dir, dataset):
    data_dir = os.path.join(data_dir, dataset)
    with open(data_dir + '/data_filter4.pkl', 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
    day_range = np.array([5337126, 3870752, 3335302, 3363122, 3835892, 3225010, 5287222, 3832608, 4218938, 0],
                         dtype=np.int)
    day_range = np.cumsum(day_range, axis=0)
    ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x = np.array(x, dtype=np.int)
    data_idx = np.arange(0, x.shape[0])

    if dataset == 'criteo':
        train_idx = data_idx[:int(x.shape[0] * ratio_range[0])]
        train_x = x[train_idx]
        field_dims_tr = np.max(train_x, axis=0) + 1
        print(field_dims_tr)
        print(field_dims_tr.sum())
        for i in range(0, 9):
            test_idx = data_idx[int(x.shape[0] * ratio_range[i]): int(x.shape[0] * ratio_range[i+1])]
            test_x = x[test_idx]
            field_dims_te = np.max(test_x, axis=0) + 1
            field_dims_te -= field_dims_tr
            print(field_dims_te.sum())
    elif dataset == 'avazu':
        train_idx = data_idx[:day_range[0]]
        train_x = x[train_idx]
        field_dims_tr = np.max(train_x, axis=0) + 1
        print(field_dims_tr)
        print(field_dims_tr.sum())
        for i in range(0, 9):
            print(day_range[i], day_range[i+1])
            test_idx = data_idx[day_range[i]: day_range[i + 1]]
            test_x = x[test_idx]
            field_dims_te = np.max(test_x, axis=0) + 1
            field_dims_te -= field_dims_tr
            print(field_dims_te.sum())

if __name__ == '__main__':
    import argparse
    data_dir = '../../data/ctr'
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('--dataset', default='criteo', help='gpus')
    args = parser.parse_args()
    get_feature_numbers(data_dir, dataset=args.dataset)
