import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv
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

class Backbone(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, dropout_prob, backbone, device=None):
        super(Backbone, self).__init__()
        self.w = nn.Parameter(torch.Tensor(field_dims.sum(), embedding_size))
        if backbone == 'LR':
            self.l = nn.Sequential(nn.Linear(embedding_size, 1))
        elif backbone == 'DNN':
            self.l = nn.Sequential(nn.Linear(embedding_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, 1))
        elif backbone == 'WD':
            self.l1 = nn.Sequential(nn.Linear(embedding_size, 1))
            self.l2 = nn.Sequential(nn.Linear(embedding_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_prob),
                                   nn.Linear(hidden_size, 1))

        self.backbone = backbone

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, h):
        if self.backbone == 'LR':
            output = self.l(h).view(-1)
        if self.backbone == 'DNN':
            output = self.l(h).view(-1)
        elif self.backbone == 'WD':
            output = self.l1(h).view(-1) + self.l2(h).view(-1)
        return output

class Model(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, gnn_layer_num, dropout_prob, backbone, graphconv, device=None):
        super(Model, self).__init__()
        self.backbone = Backbone(field_dims, embedding_size, hidden_size, dropout_prob, backbone, device)
        self.supernet = GraphModel(embedding_size, layer_num=gnn_layer_num, graphconv=graphconv)
        self.device = device

        self.param_sn = self.supernet.parameters()
        self.param_bb = self.backbone.parameters()
        self.param_nn = self.backbone.l.parameters()
        self.w = self.backbone.w

        field_dims_ = torch.cumsum(field_dims, dim=0)
        self.field_dims_ = torch.zeros_like(field_dims_).to(device)
        self.field_dims_[1:] = field_dims_[:-1]
        self.field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)

    def forward(self, x, field_mask, new_field_mask=None, known_mask=None, mode='train', args=None):
        x = x + self.field_dims_
        x_emb = self.w[x]  # [N, F, D]
        feature_num = self.field_range[-1]
        if mode == 'train':
            x_emb[:, ~field_mask, :] = 0
            unknown_mask = field_mask * ~known_mask
            w_pred_unknown = x_emb[:, unknown_mask].reshape(-1, unknown_mask.sum()*x_emb.size(2))
            ######### construct G data
            w = self.w.clone()
            for i in (~field_mask).nonzero():
                w[self.field_range[i]:self.field_range[i+1]] = 0
            for i in unknown_mask.nonzero():
                w[self.field_range[i]:self.field_range[i+1]] = 0
            node_feature = torch.cat([w, torch.zeros(x.size(0), w.size(1), dtype=w.dtype).to(self.device)], dim=0)
            edge_index = create_edge(x, field_mask, feature_num, self.device)
            if args is not None:
                edge_index = drop_edge(edge_index, args.dropedge_prob, self.device)
            w_pred = self.supernet(x=node_feature, edge_index=edge_index)[:feature_num]
            x_emb_ = w_pred[x] # [N, F, D]
            x_emb[:, field_mask, :] = x_emb_[:, field_mask, :]
            h = x_emb.sum(dim=1)  # [N, D]
            output = self.backbone(h)

            # w_unknown = torch.cat(
            #     [self.w[self.field_range[i]:self.field_range[i+1]].detach() for i in unknown_mask.nonzero()],
            #     dim = 0)
            # w_pred_unknown = torch.cat(
            #     [w_pred[self.field_range[i]:self.field_range[i+1]] for i in unknown_mask.nonzero()],
            #     dim = 0)

            # w_unknown = x_emb[:, unknown_mask].reshape(-1, unknown_mask.sum()*x_emb.size(2))

            # w_recon = torch.sum(torch.square(w_unknown - w_pred_unknown))
            # w_recon = - torch.cosine_similarity(w_unknown.view(-1), w_pred_unknown.view(-1), dim=0)
            # w_unknown_ = (w_unknown - torch.mean(w_unknown, dim=1)) / torch.std(w_unknown, dim=1)
            # w_pred_unknown_ = (w_pred_unknown - torch.mean(w_pred_unknown, dim=1)) / torch.std(w_pred_unknown, dim=1)

            # sim_mat = torch.matmul(w_unknown, w_pred_unknown.t()) # [N, N]
            # sim_mat_de = sim_mat.logsumexp(dim=1)
            # w_recon = - torch.mean(
            #     sim_mat.diagonal(0) - sim_mat_de
            # )

            # w_recon = torch.sum(torch.square())
            return output, 0.

        elif mode == 'test':
            w = self.w.clone()
            for i in new_field_mask.nonzero():
                w[self.field_range[i]:self.field_range[i + 1]] = 0
            node_feature = torch.cat([w, torch.zeros(x.size(0), w.size(1), dtype=w.dtype).to(self.device)], dim=0)
            edge_index = create_edge(x, torch.ones_like(field_mask).to(self.device), feature_num, self.device)
            w_pred = self.supernet(x=node_feature, edge_index=edge_index)[:feature_num]
            x_emb_ = w_pred[x]  # [N, F, D]
            h = x_emb_.sum(dim=1)  # [N, D]
            output = self.backbone(h)

            return output

    def get_embedding(self):
        return self.backbone.w

    # def load_embedding(self, path):
    #     pretrained_dict = torch.load(path)
    #     model_dict = self.embedding_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.embedding_model.load_state_dict(model_dict)