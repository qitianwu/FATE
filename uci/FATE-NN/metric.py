import torch

def accuracy_calc(pred, label):
    return torch.sum(pred == label) / label.shape[0]

def mae_calc(pred, label):
    return torch.sum( torch.abs(pred - label) ) / pred.shape[0]

def rmse_calc(pred, label):
    return torch.sqrt( torch.sum( (pred - label) ** 2 ) / pred.shape[0] )