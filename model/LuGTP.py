import torch.nn
from dgl.nn.pytorch import GraphConv, AvgPooling, MaxPooling
from torch import nn
import torch.nn.functional as F

from config import dropout_ratio
from model.layer import Lupool, ConvsLayer


class LuGTP(torch.nn.Module):
    def __init__(self, inputfeature=1024, hiddsize=128):
        super(LuGTP, self).__init__()
        self.inputfeature = inputfeature
        self.hiddsize = hiddsize
        self.gconv1 = GraphConv(inputfeature, hiddsize, weight=True, bias=True)
        self.lupool = Lupool(hiddsize)
        self.gconv2 = GraphConv(hiddsize, hiddsize, weight=True, bias=True)
        self.lupool2 = Lupool(hiddsize)
        self.gconv3 = GraphConv(hiddsize,hiddsize, weight=True, bias=True)
        self.lupool3 = Lupool(hiddsize)
        self.catlay512 = nn.Linear(512, 128)
        self.catlay256 = nn.Linear(256, 128)
        self.lin128 = nn.Linear(128, 64)
        self.lin64 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hiddsize, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.dropout = nn.Dropout(dropout_ratio)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.textcnn = ConvsLayer(self.inputfeature)
        self.textflatten = nn.Linear(128, hiddsize)
    def forward(self, gbatch, pad_dmap):
        # 第一次卷积+池化
        g1 = self.gconv1(gbatch, gbatch.ndata['feat'])
        g1 = self.relu(g1)
        gbatch.ndata['feat']=g1
        G, x = self.lupool(gbatch, g1)
        # G=gbatch
        # x=g1
        avx = self.avgpool(G, x)
        maxx = self.maxpool(G, x)
        readout1 = torch.cat([avx, maxx], dim=1)
        # 第二次卷积+池化
        g2 = self.gconv2(G, G.ndata['feat'])
        g2 = self.relu(g2)
        G.ndata['feat'] = g2
        G2, x2 = self.lupool2(G, g2)
        # G2 = G
        # x2 = g2
        avx2 = self.avgpool(G2, x2)
        maxx2 = self.maxpool(G2, x2)
        readout2 = torch.cat([avx2, maxx2], dim=1)
        # 第三次卷积池化
        g3 = self.gconv3(G2, G2.ndata['feat'])
        g3 = self.relu(g3)
        G2.ndata['feat'] = g3
        G3, x3 = self.lupool3(G2, g3)
        # G3 = G2
        # x3 = g3
        avx3 = self.avgpool(G3, x3)
        maxx3 = self.maxpool(G3, x3)
        readout3 = torch.cat([avx3, maxx3], dim=1)
        readout = readout1 + readout2 + readout3
        gnn=self.relu(self.catlay256(readout))
        #textcnn
        seq1 = self.textcnn(pad_dmap)
        seq1 = self.relu(self.textflatten(seq1))
        w1 = torch.sigmoid(self.w1)
        gc1 = torch.add((1 - w1) * gnn, w1 * seq1)
        # add some dense layers
        gc = self.fc1(gc1)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        gc = self.fc2(gc)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        out = self.out(gc)
        out = self.relu(out)
        output = torch.softmax(out, dim=1)

        # x = F.relu(self.catlay256(readout))
        # x = F.dropout(x, p=dropout_ratio, training=self.training)
        # x = F.relu(self.lin128(x))
        # x = F.log_softmax(self.lin64(x), dim=-1)
        return output
