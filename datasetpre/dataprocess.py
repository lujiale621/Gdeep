import csv
import os
import zlib

import dgl
import scipy.sparse as spp
import numpy as np
import torch
from dgl import save_graphs, load_graphs
from torch_geometric.data import data, InMemoryDataset, Batch

from config import *
from datasetpre.emdata import Embdict, dictdata, Emdata
from datasetpre.feature_computation.dssp_computation.compute import loaddsspfile
from datasetpre.feature_computation.pssm_computation.compute import load_fasta_and_compute

dictdata = Emdata()
letterDict = {}
letterDict["A"] = 0
letterDict["C"] = 1
letterDict["D"] = 2
letterDict["E"] = 3
letterDict["F"] = 4
letterDict["G"] = 5
letterDict["H"] = 6
letterDict["I"] = 7
letterDict["K"] = 8
letterDict["L"] = 9
letterDict["M"] = 10
letterDict["N"] = 11
letterDict["P"] = 12
letterDict["Q"] = 13
letterDict["R"] = 14
letterDict["S"] = 15
letterDict["T"] = 16
letterDict["V"] = 17
letterDict["W"] = 18
letterDict["Y"] = 19
letterDict["*"] = 20
def checkinfo(sseq, protein, position):
    # global mapdata
    Tag = True
    # 检查序列是否存在
    if sseq == 'none':
        Tag = False

    # 检查蛋白质cmap
    url = cpath + protein + '.cm'
    if os.path.exists(url) == False:
        Tag = False
    # 检查embed是否存在

    mappostion = dictdata.getTag(protein)
    if mappostion == -1:
        Tag = False
    if position > len(sseq):
        Tag = False
    return Tag
def getcmap(url, cmdata):
    with open(url, 'rb') as rf:
        data = rf.read()
        st = zlib.decompress(bytes(data)).decode("utf-8")
        lines = st.strip().split("\n")

    ret = np.zeros((len(lines), len(lines)))

    for x in range(len(lines)):
        # z = 0
        raw_row = lines[x].strip().split(' ')
        for y in range(len(lines)):
            if raw_row[y] == '1':
                ret[x][y] = 1
    return ret
embdict = Embdict()
def getseqMatrix_Label(train_file_name, window_size=51, empty_aa='*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []
    short_seqs = []
    temp_row = []
    half_len = (window_size - 1) / 2

    with open(train_file_name, 'r', encoding='utf-8') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            temp_row = row
            position = int(row[2])
            sseq = row[3]
            protein = row[1]
            # if sseq == 'none':
            #     continue
            # if position>len(sseq):
            #     continue
            if checkinfo(sseq, protein, position) == False:
                continue
            rawseq.append(row[3])
            center = sseq[position - 1]
            all_label.append(int(row[0]))
            prot.append(row[1])
            pos.append(row[2])
            # short seq
            if position - half_len > 0:
                start = position - int(half_len)
                left_seq = sseq[start - 1:position - 1]
            else:
                left_seq = sseq[0:position - 1]

            end = len(sseq)
            if position + half_len < end:
                end = position + half_len
            right_seq = sseq[position:int(end)]

            if len(left_seq) < half_len:
                nb_lack = half_len - len(left_seq)
                left_seq = ''.join([empty_aa for count in range(int(nb_lack))]) + left_seq

            if len(right_seq) < half_len:
                nb_lack = half_len - len(right_seq)
                right_seq = right_seq + ''.join([empty_aa for count in range(int(nb_lack))])
            shortseq = left_seq + center + right_seq
            short_seqs.append(shortseq)
        targetY = listclass_to_one_hot(all_label)
        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo + 1
            samplenumber = samplenumber + 1
    return Matr, targetY

def getdsspMatrix(train_file_name, window_size=51):
    prot = []  # list of protein name
    dssps = []
    half_len = (window_size - 1) / 2

    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            if checkinfo(sseq, protein, position) == False:
                continue
            dssp = loaddsspfile(dssp_fn, protein + '_' + str(position), len(sseq))
            dssps.append((np.transpose(dssp)))
    return dssps

def getpssmMatrix(train_file_name, window_size=51, empty_aa="*"):
    prot = []  # list of protein name
    pssms = []
    half_len = (window_size - 1) / 2
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            if checkinfo(sseq, protein, position) == False:
                continue
            if position < (window_size - 1) / 2:
                start = 0
                l_padding = (window_size - 1) / 2 - position
            else:
                start = position - (window_size - 1) / 2
                l_padding = 0
            if position > len(sseq) - (window_size - 1) / 2:
                end = len(sseq)
                r_padding = (window_size - 1) / 2 - (len(sseq) - position)
            else:
                end = position + (window_size - 1) / 2
                r_padding = 0
            # if position > len(sseq):
            #     continue

            prot.append(protein)
            seq_fn = pssm_root_url + "/" + str(protein) + '.fasta'
            if not os.path.exists(seq_fn):
                fp = open(seq_fn, "w")
                fp.write('>' + str(protein) + '\n')
                fp.write(sseq)
                fp.close()
            out_base_fn = pssm_root_url
            raw_pssm_dir = pssm_fn
            pssm = load_fasta_and_compute(protein, position, raw_pssm_dir, start, end, l_padding, r_padding)
            pssms.append(np.transpose(pssm))

    return pssms
def productGraph(train_file_name):
    labels=[]
    Glist = []
    positionlist=[]
    emblist=[]
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            cmdata = []
            sseq = row[3]
            position = int(row[2])
            protein = row[1]
            label=int(row[0])
            if checkinfo(sseq, protein, position) == False:
                continue
            tag = os.path.exists(graph_path + protein+".g")
            if tag == False:
                url = cpath + protein + '.cm'
                g_embed = embdict.getTag(protein_name=protein)
                g_embed = torch.from_numpy(g_embed)
                cmdata = getcmap(url, cmdata)
                if len(g_embed) != len(cmdata):
                    cmdata = cmdata[:len(g_embed), :len(g_embed)]
                adj = spp.coo_matrix(cmdata)
                G = dgl.from_scipy(adj)
                G.ndata['feat'] = g_embed.float()
                graph_labels = {"glabel": torch.tensor([label])}
                save_graphs(graph_path+protein+".g", [G],graph_labels)
            else:
                G, label = load_graphs(graph_path+protein+".g")
                # labels.append(label)
                g=G[0]
                edgindex=g.edges()
                Glist.append(g)
                positionlist.append(position)
                g_embed=g.ndata['feat']
                nodenum = len(g_embed)
                if nodenum > 1000:
                    textembed = g_embed[:1000]
                elif nodenum < 1000:
                    textembed = np.concatenate((g_embed, np.zeros((1000 - nodenum, 1024))))
                    textembed = torch.from_numpy(textembed)
                emblist.append(textembed)
    return Glist,emblist
def listclass_to_one_hot(list, isnumpy=True):
    list_len = len(list)
    li = []
    for i in range(list_len):
        li.append([list[i]])
    one_hot_list = torch.LongTensor(li)
    # 标签独热编码的形式
    if isnumpy:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1).numpy()
    else:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1)
    return one_hot_list

def pygnewgraph(train_file_name):
    labels = []
    Glist = []
    positionlist = []
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            cmdata = []
            sseq = row[3]
            position = int(row[2])
            protein = row[1]
            label = int(row[0])
            if checkinfo(sseq, protein, position) == False:
                continue
            tag = os.path.exists(pyggraph_path + protein + ".pt")
            if tag == False:
                url = cpath + protein + '.cm'
                g_embed = embdict.getTag(protein_name=protein)
                g_embed = torch.from_numpy(g_embed)
                cmdata = getcmap(url, cmdata)
                if len(g_embed) != len(cmdata):
                    cmdata = cmdata[:len(g_embed), :len(g_embed)]
                tmp_coo = spp.coo_matrix(cmdata)
                values = tmp_coo.data
                indices = np.vstack((tmp_coo.row, tmp_coo.col))
                i = torch.LongTensor(indices)
                v = torch.LongTensor(values)
                edge_index = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
                x=g_embed.float()
                y=torch.tensor(label)
                g = data.Data(edge_index=edge_index, x=x,y=y)
                torch.save([g],pyggraph_path+protein+".pt")
            else:
                g = torch.load(pyggraph_path+protein+".pt")[0]
                Glist.append(g)

        return Glist


















#pyg框架训练
# pygdata=pygnewgraph(train_file_name)
# loader= pygdatasetloader(pygdata)
# model = Net(num_features=1024,nhid=128,pooling_ratio=0.5,dropout_ratio=0.5,num_classes=2).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
# for epoch in range(epochs):
#     model.train()
#     for i, data in enumerate(loader):
#         data = data.to(device)
#         out = model(data)
#         loss = F.nll_loss(out, data.y)
#         print("Training loss:{}".format(loss.item()))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
