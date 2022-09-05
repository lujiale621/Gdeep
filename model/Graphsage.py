import dgl
import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class SAGEConvq(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGEConvq, self).__init__()
        self.layer1 = GCNLayer(in_feats, h_feats)
        self.layer2 = GCNLayer(h_feats, num_classes)
        self.out = nn.Linear(num_classes, 2)
        self.relu = nn.ReLU()
    def forward(self, mfgs, x):
        mfg0srcfeat = mfgs[0].srcdata['feat']
        print("上层源节点特征：")
        print(mfg0srcfeat)
        mfg0dstfeat = mfgs[0].dstdata['feat']
        print("上层目标节点特征：")
        print(mfg0dstfeat)
        print("上层源节点个数" + str(mfgs[0].num_src_nodes()), "上层目标节点个数" + str(mfgs[0].num_dst_nodes()))
        print("上层源节点ID" + str(mfgs[0].srcdata[dgl.NID]), "上层目标节点ID" + str(mfgs[0].dstdata[dgl.NID]))
        GCN1=self.layer1(mfgs[0], x)
        x = F.relu(GCN1)
        x = self.layer2(mfgs[1], x)

        x = self.out(x)
        x = self.relu(x)
        output = torch.softmax(x, dim=1)
        return output

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = dgl.function.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats*2, out_feats)

    def forward(self, block, h):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            # Lines that are changed are marked with an arrow: "<---"
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            # block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            out1=block.dstdata['h']
            out2=block.dstdata['h_neigh']
            # return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
            return self.linear(torch.cat(
                [block.dstdata['h'], block.dstdata['h_neigh']], 1))