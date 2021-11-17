import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .layers import TimeBlock

class FirstStep(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout, alpha, concat=True):
        super(FirstStep, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2.0))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a2.data, gain=np.sqrt(2.0))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        batch_size = input.size(0)
        h = torch.bmm(input, self.W.expand(batch_size, self.in_features, self.out_features))
        f_1 = torch.bmm(h, self.a1.expand(batch_size, self.out_features, 1))
        f_2 = torch.bmm(h, self.a2.expand(batch_size, self.out_features, 1))
        e = self.leakyrelu(f_1 + f_2.transpose(2,1))
        # dot_prod = torch.mul(adj, e)
        # print('dot_prod', dot_prod.shape)
        return e, h

class LastStep(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout, alpha, concat=True):
        super(LastStep, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes
        # self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))
        # nn.init.constant_(self.bias, 0.1)
        self.dp = nn.Dropout(self.dropout)
        self.out_features = out_features
        self.downsample = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.downsample.data, gain=np.sqrt(2.0))
        
    def forward(self, attention, h, res, adj):
        batch_size = attention.size(0)
        attention = torch.mul(adj, attention)
        attention = self.dp(attention)
        output = torch.bmm(attention, h)
        # print('res', res.shape, output.shape, self.downsample.shape)
        if res.shape[-1] != output.shape[-1]:
            res = torch.bmm(res, self.downsample.expand(batch_size, -1, -1))
            output = output + res
        else:
            output = output + res
        return F.elu(output)


class TalkingHeadLayer(nn.Module):
    
    def __init__(self, in_features, out_features, num_nodes, dropout, alpha, concat=True, nheads=4, needSE=True):
        super(TalkingHeadLayer, self).__init__()
        self.concat = concat
        self.nheads = nheads
        self.in_features = in_features
        self.needSE = needSE
        self.out_features = out_features
        if needSE:
            self.firststep = [FirstStep(in_features+64, out_features, num_nodes=num_nodes, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        else:
            self.firststep = [FirstStep(in_features, out_features, num_nodes=num_nodes, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        self.firststep = nn.ModuleList(self.firststep)
        self.laststep = [LastStep(in_features, out_features, num_nodes=num_nodes, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        self.laststep = nn.ModuleList(self.laststep)
        self.P_l = nn.Parameter(torch.zeros((nheads, nheads)))
        self.P_w = nn.Parameter(torch.zeros((nheads, nheads)))
        nn.init.xavier_uniform_(self.P_l.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.P_w.data, gain=np.sqrt(2.0))
        

    def forward(self, input, adj, SE):
        batch_size = input.size(0)
        res = input
        if self.needSE:
            input = torch.cat([input, SE.repeat(batch_size, 1, 1)], dim=2)
        e_list = []
        h_list = []
        for i, func in enumerate(self.firststep):
            e, h = func(input, adj)
            e_list.append(e.unsqueeze(-1))
            h_list.append(h)
        # h_output = torch.cat(h_list, dim=3)
        # print('h', h_output.shape)
        f_output = torch.cat(e_list, dim=3)
        f_output = f_output @ self.P_l
        f_output = f_output.permute(3, 0, 1, 2).contiguous()
        attention = F.softmax(f_output, dim=2).permute(1, 2, 3, 0).contiguous()
        attention = attention @ self.P_w
        if self.concat:
            output = torch.cat([func(attention[:,:,:,i], h_list[i], res, adj) for i, func in enumerate(self.laststep)], dim=2)
        else:
            output = sum([func(attention[:,:,:,i], h_list[i], res, adj) for i, func in enumerate(self.laststep)]) / self.nheads
        return output