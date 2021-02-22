import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Define a Graph convolutional layer with a learnable adjacency matrix
    """

    def __init__(self, in_features, out_features, bias=True, node_n=57):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    """
    Define a residual block of GCN
    """

    def __init__(self, in_features, p_dropout, bias=True, node_n=57):

        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()

    def forward(self, x):
        y = self.gc1(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_corr(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, num_stage=2, node_n=57):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_corr, self).__init__()
        self.num_stage = num_stage

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature, 1, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()

    def forward(self, x):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = self.gcout(y)

        att = self.gcatt(y)
        att = self.act_fatt(att)

        return out, att


class GCN_class(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=32, p_dropout=0.5, node_n=57, classes=12):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_class, self).__init__()

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bnin = nn.BatchNorm1d(node_n * hidden_feature)
        self.bnout = nn.BatchNorm1d(node_n * input_feature)
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x):

        if len(x.shape) == 3:
            b, n, f = x.shape
        else:
            b = 1
            n, f = x.shape

        y = self.gcin(x)
        if b > 1:
            y = self.bnin(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcout(y)
        if b > 1:
            y = self.bnout(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = y.view(-1, n * f)
        y = self.lin(y)
        y = self.act_flin(y)

        return y