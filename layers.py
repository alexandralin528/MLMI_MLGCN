# simple GCN layer copied from pygcn 
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

## This class describes a simple GCN layer like it was presented in the "Semi supervised cassifictio wh GCNs" paper


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) #HL:input: 10351*10312, weight: 10312*400 --> 10351*400; LL: input: 10351*10312, weight: 10312*400 --> 10351*400
        output = torch.matmul(adj, support) #HL: F_tilde: 39* 10351 support: 10351*400 --> 39*400; LL: E_tilde: 10312*10351 support: 10351*400 --> 10312*400
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'