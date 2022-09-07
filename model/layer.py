import numpy as np
import torch
from dgl.nn.pytorch import GraphConv, GATv2Conv
from torch import nn, topk, tensor


class Lupool(nn.Module):
    def __init__(self,in_channels,ratio=0.5):
        super(Lupool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU()
        self.non_linearity = torch.tanh
        self.hid=int(in_channels/2)
        self.score_layer = GATv2Conv(in_channels, self.hid, num_heads=2)
        self.score_layer2 = GATv2Conv(in_channels, in_channels, num_heads=1)
        self.score_sum =GraphConv(self.in_channels,1, weight=True, bias=True)


    def forward(self, G,x):
        score = self.score_layer(G, x)
        score = self.relu(score)
        g1 = score.reshape(-1, self.in_channels)
        score2 = self.score_layer2(G, g1)
        score2 = self.relu(score2)
        g2 = score2.reshape(-1, self.in_channels)
        socres = self.score_sum(G, g2).squeeze()
        nodesid = G.nodes()
        npnodelist=nodesid.cpu().numpy()
        value, indices = topk(socres, int(len(nodesid)*self.ratio),largest=True)
        npindices=indices.cpu().numpy()
        diff = np.setdiff1d(npnodelist, npindices)
        # x=x[indices]*self.non_linearity(socres[indices]).view(-1, 1)
        G.remove_nodes(diff)
        return G,G.ndata['feat']
class ConvsLayer(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ConvsLayer, self).__init__()
        self.embedding_size = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=128, kernel_size=3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx3 = nn.MaxPool1d(108, stride=1)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.conv1(x)
        features = self.mx1(features)
        features = self.mx2(self.conv2(features))
        features = self.conv3(features)
        features = self.mx3(features)
        features = features.squeeze(2)
        return features

def testpool():
    ab=tensor([0.1013, 0.1238, 0.1412,-0.0563, -0.0690, -0.0561])
    at=tensor([[-0.1480, -0.9819, -0.3364, 0.7912, -0.3263],
            [-0.8013, -0.9083, 0.7973, 0.1458, -0.9156],
            [-0.2334, -0.0142, -0.5493, 0.0673, 0.8185],
            [-0.4075, -0.1097, 0.8193, -0.2352, -0.9273]])
    value, indices = topk(ab, int(len(ab) * 0.8), largest=True)
    print(value)
    print(indices)
if __name__ == '__main__':
    testpool()