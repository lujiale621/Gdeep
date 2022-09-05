from torch import nn
from torch.utils.data import DataLoader

from config import *
import dgl
import torch
import torch.nn.functional as F

from datasetpre.dataprocess import productGraph, listclass_to_one_hot, pygnewgraph
from datasetpre.dataset import LuDataset
from model.Graphsage import  SAGEConvq
from model.LuGTP import LuGTP
from model.Sagpool import Net
from utils.assess import calculate_indicators
from utils.datasetut import pygdatasetloader

def collate(samples):
    seqmatrix, label, pssms, dssps ,emd, graph = map(list, zip(*samples))
    labels = []
    for i in label:
        labels.append(i)
    labels = torch.tensor(labels)
    return seqmatrix, labels, pssms, dssps, emd, graph
def train(train_file_name,device=torch.device('cuda')):
    # 批量装载数据集
    train_data = LuDataset(train_file_name=train_file_name)
    train_iter = DataLoader(dataset=train_data, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
    print("加载训练数据集 batch_size=128")
    net = LuGTP().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        for batch_idx, (seqmatrix, label, pssms, dssps, emd, graph) in enumerate(train_iter):
            optimizer.zero_grad()
            predict= net(dgl.batch(graph).to(device),pad_dmap(emd))
def pad_dmap(dmaplist):
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1000, 1024)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors
if __name__ == '__main__':
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train(train_file_name)
