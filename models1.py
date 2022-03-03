import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


# In this file, we'll define two GCNs : the high layer GCN and the low layer GCN
## This GCN is a two layer GCN 


class Low_Layer(nn.Module):
    def __init__(self, nfeat, nhid_lowlayer, nhid_highlayer , nclass, dropout):
        super(Low_Layer, self).__init__()
        self.fc1 = nn.Linear(nhid_highlayer, nfeat)
        self.gc1 = GraphConvolution(nfeat, nhid_lowlayer)
        self.dropout = dropout
        self.gc2 = GraphConvolution(nhid_lowlayer, nclass)
    
    def forward(self,Y_embedding, X, E_tilde, A_tilde):
        ### Y_embedding is the output comes from the high layer forward function
        Y_new_dimension = self.fc1(Y_embedding)
        X_star = torch.cat((X,Y_new_dimension), dim = 0)
        ### remember that the order of nodes in E_tilde should be the same as the X_star features
        ### (e.g, the first row of both E_tilde and X_star should be adjacency and features of the same node)
        X_embedding = F.relu(self.gc1(X_star, E_tilde))
        X_embedding = F.dropout(X_embedding, self.dropout, training = self.training)
        output = self.gc2(X_embedding, A_tilde)
        ### this X_embedding will be used as input for high layer model and the fc in high layer model will be applied on it
        ### fc in high layer model will have dimension from nhid_lowlayer to nclass (nn.Linear(nhid_lowlayer, nclass)) since the input dimension in high layer model is nclass and you should change the dimension of node features to nclass
        return torch.sigmoid(output), X_embedding
    
    
    
class High_Layer(nn.Module):
    def __init__(self, nfeat, nhid_lowlayer, nhid_highlayer, nclass, dropout):
        super(High_Layer, self).__init__()
        self.fc1 = nn.Linear(nhid_lowlayer, nfeat)
        self.gc1 = GraphConvolution(nfeat, nhid_highlayer)
        self.dropout = dropout
        self.gc2 = GraphConvolution(nhid_highlayer, nclass)
    def forward(self, X_embedding, Y, F_tilde, C_tilde):
        X_new_dimension = self.fc1(X_embedding)
        Y_star = torch.cat((Y, X_new_dimension), dim = 0)
        Y_embedding = F.relu(self.gc1(Y_star,F_tilde))
        Y_embedding = F.dropout(Y_embedding, self.dropout, training = self.training)
        output = self.gc2(Y_embedding, C_tilde)
        return F.log_softmax(output,dim = 1), Y_embedding

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # two GCN layers
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        X_embedding = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.sigmoid(x), X_embedding

class GCN1(nn.Module): 
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, y, adj):
        y = F.leaky_relu(self.gc1(y, adj))
        Y_embedding = y
        #x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc2(y, adj)
        return F.softmax(y, dim = 1), Y_embedding 