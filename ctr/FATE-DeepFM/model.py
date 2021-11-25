import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, InstanceNorm
from torch_geometric.data import Data
from utils import create_edge, drop_edge

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

class FactorizationMachine(torch.nn.Module):

    def __init__(self, embedding_size, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x, x_):
        square_of_sum = torch.sum(x, dim=1) * torch.sum(x_, dim=1)
        sum_of_square = torch.sum(x * x_, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class Backbone(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, dropout_prob, backbone, device=None):
        super(Backbone, self).__init__()
        self.b = nn.Parameter(torch.Tensor(field_dims.sum(), 1))
        self.w = nn.Parameter(torch.Tensor(field_dims.sum(), embedding_size))
        if backbone == 'DeepFM':
            self.l = nn.Sequential(nn.Linear(embedding_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, 1))
            self.fm = FactorizationMachine(embedding_size)
        elif backbone == 'xDeepFM':
            pass
        self.bn = nn.InstanceNorm1d(embedding_size)

        self.backbone = backbone

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, x_linear, x_emb, x_emb_):
        h = x_emb_.sum(dim=1)

        if self.backbone == 'DeepFM':
            output = torch.sum(x_linear, dim=1) + self.fm(x_emb, x_emb_) + self.l(h)
        if self.backbone == 'xDeepFM':
            pass
        return output.view(-1)

class Model(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, gnn_layer_num, dropout_prob, backbone, graphconv, device=None):
        super(Model, self).__init__()
        self.backbone = Backbone(field_dims, embedding_size, hidden_size, dropout_prob, backbone, device)
        self.supernet = GraphModel(embedding_size, layer_num=gnn_layer_num, graphconv=graphconv)
        self.device = device

        self.param_sn = self.supernet.parameters()
        self.param_bb = self.backbone.parameters()
        self.param_nn = self.backbone.l.parameters()
        self.b = self.backbone.b
        self.w = self.backbone.w

        field_dims_ = torch.cumsum(field_dims, dim=0)
        self.field_dims_ = torch.zeros_like(field_dims_).to(device)
        self.field_dims_[1:] = field_dims_[:-1]
        self.field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)

    def forward(self, x, field_mask, new_field_mask=None, known_mask=None, mode='train', args=None):
        x = x + self.field_dims_
        x_linear = self.b[x] # [N, F, 1]
        x_emb = self.w[x]  # [N, F, D]
        feature_num = self.field_range[-1]
        if mode == 'train':
            x_emb[:, ~field_mask, :] = 0
            unknown_mask = field_mask * ~known_mask
            w_pred_unknown = x_emb[:, unknown_mask].reshape(-1, unknown_mask.sum()*x_emb.size(2))
            ######### construct G data
            # b = self.b.clone()
            w = self.w.clone()
            for i in (~field_mask).nonzero():
                # b[self.field_range[i]:self.field_range[i+1]] = 0
                w[self.field_range[i]:self.field_range[i + 1]] = 0
            for i in unknown_mask.nonzero():
                # b[self.field_range[i]:self.field_range[i+1]] = 0
                w[self.field_range[i]:self.field_range[i + 1]] = 0
            # w = torch.cat([w1, w2], dim=1)
            node_feature = torch.cat([w, torch.zeros(x.size(0), w.size(1), dtype=w.dtype).to(self.device)], dim=0)
            edge_index = create_edge(x, field_mask, feature_num, self.device)
            if args is not None:
                edge_index = drop_edge(edge_index, args.dropedge_prob, self.device)
            w_pred = self.supernet(x=node_feature, edge_index=edge_index)[:feature_num]
            # w1_pred, w2_pred = w_pred[:, 0].unsqueeze(1), w_pred[:, 1:]
            # x_linear_ = w1_pred[x] # [N, F, 1]
            x_emb_ = w_pred[x] # [N, F, D]
            # x_linear[:, field_mask, :] = x_linear_[:, field_mask, :]
            x_emb_[:, ~field_mask, :] = 0
            output = self.backbone(x_linear, x_emb, x_emb_)

            return output

        elif mode == 'test':
            # b = self.b.clone()
            w = self.w.clone()
            for i in new_field_mask.nonzero():
                # b[self.field_range[i]:self.field_range[i + 1]] = 0
                w[self.field_range[i]:self.field_range[i + 1]] = 0
            # w = torch.cat([w1, w2], dim=1)
            node_feature = torch.cat([w, torch.zeros(x.size(0), w.size(1), dtype=w.dtype).to(self.device)], dim=0)
            edge_index = create_edge(x, torch.ones_like(field_mask).to(self.device), feature_num, self.device)
            w_pred = self.supernet(x=node_feature, edge_index=edge_index)[:feature_num]
            # w1_pred, w2_pred = w_pred[:, 0].unsqueeze(1), w_pred[:, 1:]
            # x_linear_ = w1_pred[x]  # [N, F, 1]
            x_emb_ = w_pred[x]  # [N, F, D]
            output = self.backbone(x_linear, x_emb, x_emb_)

            return output

    def get_embedding(self):
        return self.backbone.w

    # def load_embedding(self, path):
    #     pretrained_dict = torch.load(path)
    #     model_dict = self.embedding_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.embedding_model.load_state_dict(model_dict)