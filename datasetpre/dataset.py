import numpy as np
from torch.utils.data import Dataset
import torch

from datasetpre.dataprocess import productGraph, getpssmMatrix, getdsspMatrix, getseqMatrix_Label


class LuDataset(Dataset):
    def __init__(self, train_neg_file=None, train_pos_file=None, train_file_name=None, window_size=51):
        super(LuDataset, self).__init__()
        labels=[]
        if train_file_name != None:
            # 获取位点短序列one-hot矩阵
            print(train_file_name + ":获取位点短序列one-hot矩阵")
            seqmatrix, label = getseqMatrix_Label(train_file_name=train_file_name, window_size=window_size)
            # 获取蛋白质pssm矩阵
            print(train_file_name + ":获取蛋白质pssm+dssp矩阵")
            pssms = getpssmMatrix(train_file_name)
            dssps = getdsspMatrix(train_file_name)
            print(train_file_name + "：获取蛋白图节点 蛋白cmap矩阵")
            # 获取蛋白图节点 蛋白cmap矩阵

            data = productGraph(train_file_name)
            self.glist = data[0]
            self.emb =  data[1]
            self.seqmatrix, self.label, self.pssms, self.dssps = torch.Tensor(seqmatrix), label, torch.Tensor(pssms), torch.Tensor(dssps)

    def __getitem__(self, item):
        return self.seqmatrix[item], self.label[item], self.pssms[item], self.dssps[item],self.emb[item], self.glist[item]

    def __len__(self):
        return len(self.label)


