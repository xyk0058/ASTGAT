import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .layers import TimeBlock
from .gat import MultiheadLayer, GraphAttentionLayer, TalkingHeadLayer

# Functions
def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    gumbel = gumbel.cuda()
    return gumbel
def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax( y / temperature, dim = 1)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
      Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
      Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        #k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y

class NetworkGenerator(nn.Module):

    def __init__(self, num_nodes):
        super(NetworkGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.matrix = nn.Parameter(torch.randn(num_nodes, num_nodes, 2))
        self.tau = 10
        self.weight = nn.Parameter(torch.ones(1, num_nodes, num_nodes))

    def forward(self, adj, batch_size, a):
        adj_mx = self.matrix.view(-1, 2)
        adj_mx = F.softmax(adj_mx, dim=1)
        adj_mx = F.gumbel_softmax(adj_mx, tau=self.tau, dim=1)[:,0]
        # adj_mx = gumbel_softmax(adj_mx, self.tau)[:,0]
        adjs = adj_mx.view(1, self.num_nodes, self.num_nodes)
        adjs = self.weight * adjs
        
        # print('adj shape', adj.shape)
        # adjs = a * adjs.repeat(batch_size, 1, 1) + (1-a) * adj
        return adjs
    
    def drop_temp(self, a=0.9999):
        self.tau = self.tau * a

class MultiNetworkGenerator(nn.Module):

    def __init__(self, num_nodes, nheads=6):
        super(MultiNetworkGenerator, self).__init__()
        self.nheads = nheads
        self.num_nodes = num_nodes
        self.generators = [NetworkGenerator(num_nodes) for _ in range(nheads)]
        self.generators = nn.ModuleList(self.generators)

    def forward(self, adj, batch_size, a):
        # a = 0.9
        adjs = sum([generator(adj, batch_size, a) for i, generator in enumerate(self.generators)]) / self.nheads
        
        # mask = torch.eye(self.num_nodes, device=adjs.device).view(1, self.num_nodes, self.num_nodes)
        # adjs = (1. - mask) * adjs + mask

        adjs = a * adjs.repeat(batch_size, 1, 1) + (1-a) * adj
        return adjs
    
    def drop_temp(self, a=0.9999):
        for i, generator in enumerate(self.generators):
            generator.drop_temp(a)


class SimMetricBlock(nn.Module):
    def __init__(self, num_features, num_nodes, emp=0.1):
        super(SimMetricBlock, self).__init__()
        self.emp = emp
        self.num_nodes = num_nodes
        self.w = nn.Parameter(torch.randn(1, num_nodes, num_features))
    
    def forward(self, x):
        # x (batch, node_num, feature_num)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x * self.w
        # innode (batch, innode, innode, feature_num)
        innode = x.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        outnode = innode.transpose(1, 2)
        cos = torch.cosine_similarity(innode, outnode, dim=3)
        zero_vec = 9e-15*torch.ones_like(cos)
        cos = torch.where(cos > self.emp, cos, zero_vec)
        return cos
        
class MultiSimMetricBlock(nn.Module):
    def __init__(self, num_features, num_nodes, lam=0.5, nheads=4):
        super(MultiSimMetricBlock, self).__init__()
        self.lam = lam
        self.num_nodes = num_nodes
        self.nheads = nheads
        self.SimMetricBlocks = [SimMetricBlock(num_features, num_nodes) for _ in range(nheads)]
        self.SimMetricBlocks = nn.ModuleList(self.SimMetricBlocks)
    
    def forward(self, x, A0):
        adj = sum([block(x) for block in self.SimMetricBlocks]) / self.nheads
        adj = self.lam * A0 + (1-self.lam) * adj
        return adj

class STGATBlock(nn.Module):
    def __init__(self, cuda, in_channels, spatial_channels, out_channels,
                num_nodes, num_timesteps_input, new_nodes, dropout=0.6, alpha=0.2, nheads=4, concat=True):
        super(STGATBlock, self).__init__()
        self.nheads = nheads
        self.concat = concat
        self.cuda = cuda
        self.spatial_channels = spatial_channels
        self.num_timesteps_input = num_timesteps_input
        self.temporal1 = TimeBlock(in_channels=in_channels, cuda=cuda,
                                   out_channels=spatial_channels)
        self.attentions = TalkingHeadLayer(spatial_channels*(num_timesteps_input), spatial_channels, num_nodes=num_nodes,
                            dropout=dropout, alpha=alpha, concat=concat, nheads=nheads)
        # nh = 1
        # if concat:
        #     nh = nheads
        # self.node_edge = EdgeConv(spatial_channels*(num_timesteps_input), spatial_channels*nh)

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(new_nodes, momentum=0.1)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.fill_(0.1)

    def forward(self, X, A_hat, SE):
        residual = X
        t = self.temporal1(X)
        t = t.contiguous().view(t.shape[0], t.shape[1], -1)
        # print('gat_in', t.shape)
        t2 = self.attentions(t, A_hat, SE)
        # gate = torch.sigmoid(self.node_edge(t, A_hat))
        # print('t2, gate', t2.shape, gate.shape)
        t3 = t2 #* gate

        t3 = t3.view(t3.shape[0], t3.shape[1], self.num_timesteps_input, -1)
        if t3.shape[-1] == residual.shape[-1]:
            t3 = t3 + residual[:,:,-t3.shape[2]:,:]
        else:
            t3 = t3
        # if self.training:
        #     self.attentions.drop_temp()
        return self.relu(self.batch_norm(t3))


class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels, nhid_channels, layer=4):
        super(EndConv, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.BatchNorm1d(in_channels, momentum=0.2),
            # nn.ReLU(),
            nn.Conv1d(in_channels, nhid_channels, 1))
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(nhid_channels, momentum=0.2),
            nn.ReLU(),
            nn.Conv1d(nhid_channels, nhid_channels, 1))
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(nhid_channels, momentum=0.2),
            nn.ReLU(),
            nn.Conv1d(nhid_channels, nhid_channels, 1))
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(nhid_channels, momentum=0.2),
            nn.ReLU(),
            nn.Conv1d(nhid_channels, out_channels, 1))
    
    def forward(self, X):
        X = X.permute(0, 2, 1)
        x1 = self.conv1(X)
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        x4 = self.conv4(x3)
        x4 = x4.permute(0, 2, 1)
        return x4


# NEGAT_Gumbel
class NEGAT(nn.Module):
    def __init__(self, cuda, adj, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nheads=6, nhid=48, layers=4, dropout=0.6, alpha=0.2):
        super(NEGAT, self).__init__()
        self.cuda_device = cuda
        self.num_timesteps_input = num_timesteps_input
        self.nheads = nheads
        self.layers = layers
        self.blocks = nn.ModuleList()

        self.network_generator = MultiNetworkGenerator(num_nodes, nheads)
        # self.start_conv = nn.Conv2d(in_channels=num_features,
        #                             out_channels=nhid,
        #                             kernel_size=(1,1))
        self.blocks.append( STGATBlock(cuda, in_channels=num_features, out_channels=nhid, concat=True,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads)
                            )
        self.blocks.append( STGATBlock(cuda, in_channels=int(nhid*(nheads/1)/num_timesteps_input), out_channels=nhid, concat=True,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads)
                            )
        self.blocks.append( STGATBlock(cuda, in_channels=int(nhid*(nheads/1)/num_timesteps_input), out_channels=nhid, concat=False,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads+2)
                            )
        self.end_conv = nn.Sequential(
            weight_norm(nn.Linear(nhid, 512)),
            nn.Tanh(),
            nn.Dropout(),
            weight_norm(nn.Linear(512, 512)),
            nn.Tanh(),
            nn.Dropout(),
            weight_norm(nn.Linear(512, 12)),
        )
        # self.end_conv = EndConv(nhid, 12, 512)

    def forward(self, X, A_hat, SE, a=0.5):
        batch_size = X.shape[0]
        norm_loss = 0
        origion_A_hat = A_hat

        A_hat = self.network_generator(A_hat, batch_size, a)
        if self.training:
            self.network_generator.drop_temp()
        # X = X.permute(0, 3, 1, 2)
        # X = self.start_conv(X)
        # X = X.permute(0, 2, 3, 1)
        out = self.blocks[0](X, A_hat, SE)
        out = self.blocks[1](out, A_hat, SE)
        out = self.blocks[2](out, A_hat, SE)
        
        norm_loss += self.calLoss(out, A_hat)
        
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('out1', out.shape)
        out = self.end_conv(out)
        # print('out2', out.shape)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], 1)

        if self.training:
            return out, norm_loss
        else:
            return out
    
    def calLoss2(self, x, A):
        loss = 0
        if self.training:
            n = A.shape[1]
            x = x.view(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1) @ A @ x
            loss = x[0].trace()
            for batch in range(1, x.shape[0]):
                loss += x[batch].trace()
            loss = loss / (n * n)
            loss = loss / A.shape[0]
        return loss

    def calLoss(self, x, A, a=1, b=1):
        loss = 0
        if self.training:
            n = A.shape[1]
            if self.cuda_device:
                A1 = A @ torch.ones(x.shape[0], n, 1).cuda()
                zero_vec = 9e-15*torch.ones_like(A1)
                A1 = torch.where(A1 > 0, A1, zero_vec)
                FA1 = torch.sum(torch.ones((x.shape[0], 1, n)).cuda() @ torch.log10(A1))
            else:
                FA1 = torch.sum(torch.ones((1, n)) @ torch.log2(A @ torch.ones(n, 1)))
            mmax = torch.max(A)

            FA2 = torch.sum(A * A)
            FA = (-b) * FA1 / n + a * FA2 / (n*n)
            FA = FA / A.shape[0]
            x = x.view(x.shape[0], x.shape[1], -1)

            x = x.permute(0, 2, 1) @ A @ x
            loss = x[0].trace()
            for batch in range(1, x.shape[0]):
                loss += x[batch].trace()
            loss = loss / (n * n)
            loss = loss / A.shape[0]
            # print('NetGumbel', loss, FA1, FA2)
            loss += FA
        return loss

# NEGAT_SimMetric
class NEGAT_SimMetric(nn.Module):
    def __init__(self, cuda, adj, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nheads=6, nhid=48, layers=4, dropout=0.6, alpha=0.2):
        super(NEGAT_SimMetric, self).__init__()
        self.cuda_device = cuda
        self.num_timesteps_output = num_timesteps_output
        self.nheads = nheads
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.adj = adj

        self.MultiSimMetricBlock = MultiSimMetricBlock(nhid, num_nodes)
        self.blocks.append( STGATBlock(cuda, in_channels=num_features, out_channels=nhid, concat=True,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads)
                            )
        self.blocks.append( STGATBlock(cuda, in_channels=int(nhid*(nheads/1)/num_timesteps_input), out_channels=nhid, concat=True,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads)
                            )
        self.blocks.append( STGATBlock(cuda, in_channels=int(nhid*(nheads/1)/num_timesteps_input), out_channels=nhid, concat=False,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=num_timesteps_input, new_nodes=num_nodes, nheads=nheads+2)
                            )

        # self.end_conv = nn.Sequential(
        #     weight_norm(nn.Linear(nhid, 512)),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     weight_norm(nn.Linear(512, 512)),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     weight_norm(nn.Linear(512, 12)),
        # )
        self.end_conv = EndConv(nhid, 12, 512)

    def forward(self, X, A_hat, SE, a=0.4):
        norm_loss = 0
        origion_A_hat = A_hat
        A_hat = self.adj
        out = self.blocks[0](X, A_hat, SE)
        # norm_loss += self.calLoss(out, A_hat)
        out = self.blocks[1](out, A_hat, SE)
        # norm_loss += self.calLoss(out, A_hat)
        out = self.blocks[2](out, A_hat, SE)
        # norm_loss += self.calLoss(out, A_hat)
        if self.training:
            self.adj = self.MultiSimMetricBlock(out, self.adj)
            norm_loss += self.calLoss(out, self.adj)
            self.adj = torch.mean(self.adj, dim=0).detach()
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.end_conv(out)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], 1)
        if self.training:
            return out, norm_loss
        else:
            return out

    def calLoss(self, x, A, a=0.01, b=0.01):
        loss = 0
        if self.training:
            n = A.shape[1]
            if self.cuda_device:
                A1 = A @ torch.ones(x.shape[0], n, 1).cuda()
                zero_vec = 9e-15*torch.ones_like(A1)
                A1 = torch.where(A1 > 0, A1, zero_vec)
                FA1 = torch.sum(torch.ones((x.shape[0], 1, n)).cuda() @ torch.log(A1))
            else:
                FA1 = torch.sum(torch.ones((1, n)) @ torch.log(A @ torch.ones(n, 1)))
            mmax = torch.max(A)

            FA2 = torch.sum(A * A)
            FA = (-b) * FA1 / n + a * FA2 / (n*n)
            FA = FA / A.shape[0]
            x = x.view(x.shape[0], x.shape[1], -1)

            x = x.permute(0, 2, 1) @ A @ x
            loss = x[0].trace()
            for batch in range(1, x.shape[0]):
                loss += x[batch].trace()
            loss = loss / (n * n)
            loss = loss / A.shape[0]
            # print('lossSim', loss, FA1, FA2)
            loss += FA
        return loss

class GatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedFusion, self).__init__()
        self.L1 = nn.Linear(in_channels, out_channels)
        self.L2 = nn.Linear(in_channels, out_channels)

    def forward(self, temporal, spatial):
        gated = torch.sigmoid(self.L1(temporal) + self.L2(spatial))
        out = temporal * gated + (torch.ones_like(gated)-gated) * spatial
        return out

# class NEGAT(nn.Module):
#     def __init__(self, cuda, adj, num_nodes, num_features, num_timesteps_input,
#                  num_timesteps_output, nheads=6, nhid=48, layers=4, dropout=0.6, alpha=0.2):
#         super(NEGAT, self).__init__()
#         self.cuda_device = cuda
#         self.num_timesteps_output = num_timesteps_output
#         self.nheads = nheads
#         self.layers = layers
#         self.blocks = nn.ModuleList()
#         self.adj = adj
#         self.net1 = NEGAT_Gumbel(cuda, adj, num_nodes, num_features, num_timesteps_input, num_timesteps_output)
#         self.net2 = NEGAT_Gumbel(cuda, adj, num_nodes, num_features, num_timesteps_input, num_timesteps_output)
#         self.fusion = GatedFusion(nhid, nhid)
#         self.end_conv = EndConv(nhid, 12, 512)
#         # self.end_conv = nn.Sequential(
#         #     weight_norm(nn.Linear(nhid, 512)),
#         #     nn.ReLU(),
#         #     nn.Dropout(),
#         #     weight_norm(nn.Linear(512, 512)),
#         #     nn.ReLU(),
#         #     nn.Dropout(),
#         #     weight_norm(nn.Linear(512, 12)),
#         # )

#     def forward(self, X, A_hat, SE, a=0.4):
#         loss1 = 0
#         loss2 = 0
#         if self.training:
#             out1, loss1 = self.net1(X, A_hat, SE)
#             out2, loss2 = self.net2(X, A_hat, SE)
#             fusion = self.fusion(out1, out2)
#             out = self.end_conv(fusion)
#             out = out.reshape(out.shape[0], out.shape[1], out.shape[2], 1)
#             norm_loss = loss1 + loss2
#             return out, norm_loss
#         else:
#             out1 = self.net1(X, A_hat, SE)
#             out2 = self.net2(X, A_hat, SE)
#             fusion = self.fusion(out1, out2)
#             out = self.end_conv(fusion)
#             out = out.reshape(out.shape[0], out.shape[1], out.shape[2], 1)
#             return out
