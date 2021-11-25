import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, feat_num, hidden_size, class_num, device):
        super(NN, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_num, hidden_size))
        self.l = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(hidden_size),
        nn.Linear(hidden_size, class_num)
        )

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def regularization_loss(self):
        return torch.sqrt(torch.sum(self.w ** 2))

    def forward(self, x, feature_mask, new_feature_mask=None, mode='train', model=None):
        if mode == 'train':
            if model == 'all':
                h = torch.matmul(x, self.w)
                output = self.l(h)
            else:
                x_ = x[:, feature_mask]
                w_ = self.w[feature_mask]
                h = torch.matmul(x_, w_)
                output = self.l(h)
        if mode == 'test':
            if model == 'partial':
                x1 = x[:, feature_mask]
                w1 = self.w[feature_mask]
                x2 = x[:, new_feature_mask]
                w2 = self.w[new_feature_mask]
                x_ = torch.cat([x1, x2], dim=1)
                w_ = torch.cat([w1, w2], dim=0)
                h = torch.matmul(x_, w_)
            elif model == 'avg':
                x1 = x[:, feature_mask]
                w1 = self.w[feature_mask]
                w2 = torch.mean(self.w[feature_mask], 0).reshape(1, -1).repeat(new_feature_mask.sum(), 1)
                x2 = x[:, new_feature_mask]
                x_ = torch.cat([x1, x2], dim=1)
                w_ = torch.cat([w1, w2], dim=0)
                h = torch.matmul(x_, w_)
            elif model == 'knn':
                num1, num2 = feature_mask.sum(), new_feature_mask.sum()
                if num2 > 200:
                    slice_size = num2 // 20
                    feature_idx = feature_mask.nonzero().view(-1)
                    new_feature_idx = new_feature_mask.nonzero().view(-1)
                    feature_mask_ = torch.zeros_like(feature_mask).to(self.device)
                    feature_mask_[feature_idx[torch.randint(0, num1, (int(num1 * 0.1),))]] = True
                    for i in range(num2 // slice_size + 1):
                        new_feature_idx_i = new_feature_idx[int(i * slice_size):int((i + 1) * slice_size)]
                        x_1 = x[:, feature_mask_].unsqueeze(2).repeat(1, 1, new_feature_idx_i.size(0))  # [N, f1, f2]
                        new_feature_mask_i = torch.zeros_like(new_feature_mask).to(self.device)
                        new_feature_mask_i[new_feature_idx_i] = True
                        x_2 = x[:, new_feature_mask_i].unsqueeze(1).repeat(1, x_1.size(1), 1) # [N, f1, f2]
                        sim = torch.multiply(x_1, x_2).sum(dim=0).transpose(0, 1) / x_1.size(0)  # [f2, f1]
                        top_idx_i = torch.topk(sim, k=int(num1 * 0.1 * 0.2), dim=1).indices  # [f2, k]
                        if i <= 0:
                            top_idx = top_idx_i
                        else:
                            top_idx = torch.cat([top_idx, top_idx_i], dim=0)
                    x1 = x[:, feature_mask]
                    w1 = self.w[feature_mask]
                    w_n = self.w[feature_mask_]
                    w2 = w_n[top_idx].mean(dim=1)  # [F2, D]
                    x2 = x[:, new_feature_mask]
                else:
                    x_1 = x[:, feature_mask].unsqueeze(2).repeat(1, 1, num2)  # [N, F1, F2]
                    x_2 = x[:, new_feature_mask].unsqueeze(1).repeat(1, num1, 1)
                    sim = torch.multiply(x_1, x_2).sum(dim=0).transpose(0,1) / x_1.size(0) # [F2, F1]
                    top_idx = torch.topk(sim, k=int(num1*0.2), dim=1).indices # [F2, k]
                    x1 = x[:, feature_mask]
                    w1 = self.w[feature_mask]
                    w2 = w1[top_idx].mean(dim=1) # [F2, D]
                    x2 = x[:, new_feature_mask]
                x_ = torch.cat([x1, x2], dim=1)
                w_ = torch.cat([w1, w2], dim=0)
                h = torch.matmul(x_, w_)
            elif model == 'local':
                num1, num2 = feature_mask.sum(), new_feature_mask.sum()
                if num2 > 200:
                    slice_size = num2 // 10
                    feature_idx = feature_mask.nonzero().view(-1)
                    new_feature_idx = new_feature_mask.nonzero().view(-1)
                    for i in range(num2 // slice_size + 1):
                        feature_mask_ = torch.zeros_like(feature_mask).to(self.device)
                        feature_mask_[feature_idx[torch.randint(0, num1, (int(num1 * 0.1),))]] = True
                        new_feature_idx_i = new_feature_idx[int(i * slice_size):int((i + 1) * slice_size)]
                        x_1 = x[:, feature_mask_].unsqueeze(2).repeat(1, 1, new_feature_idx_i.size(0))  # [N, f1, f2]
                        new_feature_mask_i = torch.zeros_like(new_feature_mask).to(self.device)
                        new_feature_mask_i[new_feature_idx_i] = True
                        x_2 = x[:, new_feature_mask_i].unsqueeze(1).repeat(1, x_1.size(1), 1)  # [N, f1, f2]
                        sim = torch.multiply(x_1, x_2).sum(dim=0).bool().float() # [f1, f2]
                        w1_ = self.w[feature_mask_] # [f1, D]
                        norm = torch.matmul(sim.transpose(0, 1), torch.ones(x_1.size(1), self.w.size(1)).to(self.device))  # [f2, D]
                        w2_i = torch.matmul(sim.transpose(0, 1), w1_) / (norm + 1e-16)  # [f2, D]
                        if i <= 0:
                            w2 = w2_i
                        else:
                            w2 = torch.cat([w2, w2_i], dim=0)
                    x1 = x[:, feature_mask]
                    w1 = self.w[feature_mask]
                    x2 = x[:, new_feature_mask]
                else:
                    num1, num2 = feature_mask.sum(), new_feature_mask.sum()
                    x_1 = x[:, feature_mask].unsqueeze(2).repeat(1, 1, num2)  # [N, F1, F2]
                    x_2 = x[:, new_feature_mask].unsqueeze(1).repeat(1, num1, 1)
                    sim = torch.multiply(x_1, x_2).sum(dim=0).bool().float()
                    x1 = x[:, feature_mask]
                    w1 = self.w[feature_mask]
                    norm = torch.matmul(sim.transpose(0, 1), torch.ones(num1, self.w.size(1)).to(self.device))  # [F2, D]
                    w2 = torch.matmul(sim.transpose(0, 1), w1) / (norm + 1e-16) # [F2, D]
                    x2 = x[:, new_feature_mask]
                x_ = torch.cat([x1, x2], dim=1)
                w_ = torch.cat([w1, w2], dim=0)
                h = torch.matmul(x_, w_)
            elif model == 'all':
                h = torch.matmul(x, self.w)
            output = self.l(h)
        return output

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)

    # def load_embedding(self, path):
    #     pretrained_dict = torch.load(path)
    #     model_dict = self.embedding_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.embedding_model.load_state_dict(model_dict)