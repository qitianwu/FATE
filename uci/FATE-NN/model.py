import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv

class GraphModel(nn.Module):
    def __init__(self, in_features, layer_num, graphconv='GCN'):
        super(GraphModel, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(layer_num):
            if graphconv == 'GCN':
                self.convs.append(GCNConv(in_features, in_features))
            elif graphconv == 'SAGE':
                self.convs.append(SAGEConv(in_features, in_features, normalize=True))
            elif graphconv == 'SGC':
                self.convs.append(SGConv(in_features, in_features))
            else:
                raise NotImplementedError

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.out_features)
    #     self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
        output = h
        return output

class Backbone(nn.Module):
    def __init__(self, feat_num, hidden_size, class_num, device):
        super(Backbone, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_num, hidden_size))
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(hidden_size),
        nn.Linear(hidden_size, class_num) )

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, x, feature_mask=None, w_proxy=None):
        if w_proxy is None: # train
            w_ = self.w[feature_mask, :]
            x_ = x[:, feature_mask]
            h = torch.matmul(x_, w_)
        else: # test
            h = torch.matmul(x, w_proxy)
        output = self.nn(h)

        return output

class Model(nn.Module):
    def __init__(self, feat_num, hidden_size, class_num, gnn_layer_num, graphconv, device):
        super(Model, self).__init__()
        self.backbone = Backbone(feat_num, hidden_size, class_num, device)
        self.supernet = GraphModel(hidden_size, layer_num=gnn_layer_num, graphconv=graphconv)
        self.device = device

        self.param_sn = self.supernet.parameters()
        self.param_bb = self.backbone.parameters()
        self.param_nn = self.backbone.nn.parameters()

    def forward(self, x, feature_mask, data=None, known_mask=None, new_feature_mask=None, mode='train'):
        if mode == 'pretrain':
            output = self.backbone(x, feature_mask)
            return output
        elif mode == 'train':
            w_known = self.backbone.w[known_mask, :]
            unknown_mask = feature_mask * ~known_mask
            unknown_in_feature_mask = unknown_mask[feature_mask]
            feature_num = feature_mask.sum()
            ######### mask W
            data.x[:feature_mask.size(0)][unknown_mask] = \
                torch.zeros(unknown_mask.sum(), w_known.size(1), dtype=torch.float).to(self.device)
            w_pred = self.supernet(data.x[data.train_node_mask], data.train_edge_index)[:feature_num]
            w_pred_unknown = w_pred[unknown_in_feature_mask]
            w_ = torch.cat([w_known, w_pred_unknown], dim=0)
            x_known = x[:, known_mask]
            x_unknown = x[:, unknown_mask]
            x_ = torch.cat([x_known, x_unknown], dim=1)
            output = self.backbone(x=x_, w_proxy=w_)

            w_unknown = self.backbone.w[unknown_mask, :].detach()
            sim_mat = torch.matmul(w_unknown, w_pred_unknown.t())
            w_recon = - torch.mean( sim_mat.diagonal(0) - sim_mat.logsumexp(dim=1))
            return output, w_recon

        elif mode == 'test':
            w_pretrain = self.backbone.w[feature_mask, :]
            feature_num = feature_mask.size(0)
            ######### mask W
            data.x[:feature_mask.size(0)][~feature_mask] = \
               torch.zeros((~feature_mask).sum(), w_pretrain.size(1), dtype=torch.float).to(self.device)
            w_pred = self.supernet(data.x, data.edge_index)[:feature_num]
            w_pred_new = w_pred[new_feature_mask]
            w_ = torch.cat([w_pretrain, w_pred_new], dim=0)
            x_old = x[:, feature_mask]
            x_new = x[:, new_feature_mask]
            x_ = torch.cat([x_old, x_new], dim=1)
            output = self.backbone(x=x_, w_proxy=w_)
            return output

    def get_embedding(self):
        return self.backbone.w

    # def load_embedding(self, path):
    #     pretrained_dict = torch.load(path)
    #     model_dict = self.embedding_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.embedding_model.load_state_dict(model_dict)