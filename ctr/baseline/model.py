import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, dropout_prob, backbone='LR', device=None):
        super(DNN, self).__init__()
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
        field_dims_ = torch.cumsum(field_dims, dim=0)
        self.field_dims_ = torch.zeros_like(field_dims_).to(device)
        self.field_dims_[1:] = field_dims_[:-1]

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, x, field_mask, new_field_mask, mode='train', model=None):
        x = x + self.field_dims_
        if mode == 'train':
            x_emb = self.w[x] # [N, F, D]
            x_emb[:, ~field_mask, :] = 0
            h = x_emb.sum(dim=1)  # [N, D]
        if mode == 'test':
            if model == 'partial':
                x_emb = self.w[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                h = x_emb.sum(dim=1)  # [N, D]
            elif model == 'avg':
                x_emb = self.w[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                field_ind = field_mask.nonzero()
                field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)
                avg_emb = torch.mean(
                    torch.cat([self.w[field_range[field_ind[i]]:field_range[field_ind[i]+1]]  for i in range(field_mask.sum())], dim=0), dim=0
                                     )
                x_emb[:, new_field_mask, :] = avg_emb
                h = x_emb.sum(dim=1)  # [N, D]
            elif model == 'field_avg':
                x_emb = self.w[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                num1, num2 = field_mask.sum(), new_field_mask.sum()
                x_1 = x[:, field_mask].unsqueeze(2).repeat(1, 1, num2) # [N, F1, F2]
                x_2 = x[:, new_field_mask].unsqueeze(1).repeat(1, num1, 1)
                sim = x_1.eq(x_2).sum(dim=0)
                sim /= x_1.size(0) * 2 - sim
                field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)
                avg_emb_ = torch.stack([
                    self.w[field_range[i]:field_range[i+1]] for i in range(field_mask.size(0)).mean(0)
                ], dim=0)
                x_emb[:, new_field_mask, :] = avg_emb_[new_field_mask].unsqueeze(0).repeat(x.size(0), 1)
                h = x_emb.sum(dim=1)  # [N, D]
        if self.backbone == 'LR':
            output = self.l(h)
        if self.backbone == 'DNN':
            output = self.l(h)
        elif self.backbone == 'WD':
            output = self.l1(h).view(-1) + self.l2(h).view(-1)
        return output

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class DFM(nn.Module):
    def __init__(self, field_dims, embedding_size, hidden_size, dropout_prob, backbone='LR', device=None):
        super(DFM, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(field_dims.sum(), 1))
        self.w2 = nn.Parameter(torch.Tensor(field_dims.sum(), embedding_size))
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
            self.fm = FactorizationMachine()
        elif backbone == 'xDeepFM':
            pass

        self.backbone = backbone
        field_dims_ = torch.cumsum(field_dims, dim=0)
        self.field_dims_ = torch.zeros_like(field_dims_).to(device)
        self.field_dims_[1:] = field_dims_[:-1]

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, x, field_mask, new_field_mask, mode='train', model=None):
        x = x + self.field_dims_
        if mode == 'train':
            x_linear = self.w1[x] # [N, F, 1]
            x_linear[:, ~field_mask, :] = 0
            x_emb = self.w2[x] # [N, F, D]
            x_emb[:, ~field_mask, :] = 0
            h = x_emb.sum(dim=1)  # [N, D]
        if mode == 'test':
            if model == 'partial':
                x_linear = self.w1[x]  # [N, F, 1]
                x_linear[:, ~field_mask, :] = 0
                x_emb = self.w2[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                h = x_emb.sum(dim=1)  # [N, D]
            elif model == 'avg':
                x_linear = self.w1[x]  # [N, F, 1]
                x_linear[:, ~field_mask, :] = 0
                x_emb = self.w2[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                field_ind = field_mask.nonzero()
                field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)
                avg_linear = torch.mean(
                    torch.cat([self.w1[field_range[field_ind[i]]:field_range[field_ind[i] + 1]] for i in
                               range(field_mask.sum())], dim=0), dim=0
                )
                x_linear[:, new_field_mask, :] = avg_linear
                avg_emb = torch.mean(
                    torch.cat([self.w2[field_range[field_ind[i]]:field_range[field_ind[i]+1]]  for i in range(field_mask.sum())], dim=0), dim=0
                                     )
                x_emb[:, new_field_mask, :] = avg_emb
                h = x_emb.sum(dim=1)  # [N, D]
            elif model == 'field_avg':
                x_linear = self.w1[x]  # [N, F, 1]
                x_linear[:, ~field_mask, :] = 0
                x_emb = self.w2[x]  # [N, F, D]
                x_emb[:, ~field_mask, :] = 0
                field_range = torch.cat([self.field_dims_, torch.tensor([self.w.size(0)]).to(self.device)], dim=0)
                avg_linear_ = torch.stack([
                    self.w1[field_range[i]:field_range[i + 1]] for i in range(field_mask.size(0)).mean(0)
                ], dim=0)
                x_linear[:, new_field_mask, :] = avg_linear_[new_field_mask].unsqueeze(0).repeat(x.size(0), 1)
                avg_emb_ = torch.stack([
                    self.w2[field_range[i]:field_range[i+1]] for i in range(field_mask.size(0)).mean(0)
                ], dim=0)
                x_emb[:, new_field_mask, :] = avg_emb_[new_field_mask].unsqueeze(0).repeat(x.size(0), 1)
                h = x_emb.sum(dim=1)  # [N, D]
        if self.backbone == 'DeepFM':
            output = torch.sum(x_linear, dim=1) + self.fm(x_emb) + self.l(h)
        if self.backbone == 'xDeepFM':
            pass
        return output

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)
