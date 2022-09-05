import torch.nn
from dgl.nn.pytorch import GraphConv, AvgPooling,MaxPooling
from torch import nn

from model.layer import Lupool


class LuGTP(torch.nn.Module):
    def __init__(self,inputfeature=1024,hiddsize=512):
        super(LuGTP, self).__init__()
        self.inputfeature = inputfeature
        self.hiddsize=hiddsize
        self.gconv1 = GraphConv(inputfeature,hiddsize, weight=True, bias=True)
        self.lupool= Lupool(hiddsize)
        self.gconv2 = GraphConv(int(hiddsize/2),int(hiddsize/2), weight=True, bias=True)
        self.lupool2= Lupool(int(hiddsize/2))
        self.gconv3 = GraphConv(int(hiddsize/4),int(hiddsize/4), weight=True, bias=True)
        self.lupool3= Lupool(int(hiddsize/4))
        self.catlay512=GraphConv(512,128, weight=True, bias=True)
        self.catlay256 = GraphConv(256, 128, weight=True, bias=True)
        self.relu = nn.ReLU()
        self.avgpool=AvgPooling()
        self.maxpool = MaxPooling()
    def forward(self,gbatch,pad_dmap):
        #第一次卷积+池化
        g1 = self.gconv1(gbatch, gbatch.ndata['feat'])
        g1=self.relu(g1)
        G,x=self.lupool(gbatch,g1)
        G.ndata['feat']=x
        avx=self.avgpool(G, x)
        maxx=self.maxpool(G, x)
        readout1=torch.cat([avx,maxx],dim=1)
        # 第二次卷积+池化
        g2 = self.gconv2(G, G.ndata['feat'])
        g2 = self.relu(g2)
        G2, x2 = self.lupool2(G, g2)
        G2.ndata['feat'] = x2
        avx2 = self.avgpool(G2, x2)
        maxx2 = self.maxpool(G2, x2)
        readout2 = torch.cat([avx2, maxx2], dim=1)
        #第三次卷积池化
        g3 = self.gconv3(G2, G2.ndata['feat'])
        g3 = self.relu(g3)
        G3, x3 = self.lupool3(G2, g3)
        G3.ndata['feat'] = x3
        avx3 = self.avgpool(G3, x3)
        maxx3 = self.maxpool(G3, x3)
        readout3 = torch.cat([avx3, maxx3], dim=1)
        out=readout1+readout2+readout3
        print(out)
        print(G)
