import numpy as np
import torch
from dgl.nn.pytorch import GraphConv, GATv2Conv
from torch import nn, topk


class Lupool(nn.Module):
    def __init__(self,in_channels,ratio=0.8):
        super(Lupool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU()
        self.non_linearity = torch.tanh
        self.hid=int(in_channels/2)
        self.score_layer = GATv2Conv(in_channels, self.hid,num_heads=2)
        self.score_layer2 = GATv2Conv(in_channels, self.hid, num_heads=1)
        self.score_sum =GraphConv(self.hid,1, weight=True, bias=True)


    def forward(self, G,x):
        score=self.score_layer(G,x)
        score=self.relu(score)
        g1 = score.reshape(-1, self.in_channels)
        score2=self.score_layer2(G,g1)
        score2 = self.relu(score2)
        g2 = score2.reshape(-1, self.hid)
        socres=self.score_sum(G,g2).squeeze()
        nodesid=G.nodes()
        npnodelist=nodesid.cpu().numpy()
        value, indices = topk(socres, int(len(nodesid)*self.ratio))
        npindices=indices.cpu().numpy()
        diff = np.setdiff1d(npnodelist, npindices)
        g2=g2[indices]*self.non_linearity(socres[indices]).view(-1, 1)
        G.remove_nodes(diff)
        return G,g2
