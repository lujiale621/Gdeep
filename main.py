from torch import nn, Tensor
from torch.utils.data import DataLoader

from config import *
import dgl
import torch
import torch.nn.functional as F

from datasetpre.dataset import LuDataset

from model.LuGTP import LuGTP
from utils.assess import calculate_indicators, calculate_auc


def collate(samples):
    seqmatrix, label, pssms, dssps, emd, graph = map(list, zip(*samples))
    labels = []
    for i in label:
        labels.append(i)
    labels = torch.tensor(labels)
    return seqmatrix, labels, pssms, dssps, emd, graph

def evaluate_val_accuracy_gpu(epoch, net, data_iter, device=None):
    pred_labs = []
    real_labs = []
    val_l = 0.
    net.eval()
    loss = nn.CrossEntropyLoss()
    for batch_idx, (seqmatrix, label, pssms, dssps, emd, graph) in enumerate(data_iter):
        # data = [d.to(device) for d in data]
        y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
        pred = y_hat.argmax(dim=1)
        l = loss(y_hat, label.to(device))
        list_len = len(pred)
        li = []
        for i in range(list_len):
            li.append([pred[i]])
        one_hot_list = torch.LongTensor(li).to(device)
        lab = torch.zeros(list_len, 2).to(device).scatter_(1, one_hot_list, 1)
        pred_labs.append(lab)
        real_labs.append(label.to(device))
        val_l += l
    assess = calculate_indicators(epoch, pred_labs, real_labs)
    print(
        "Val   Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
            epoch, val_l / len(data_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
            assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))

def evaluate_test_accuracy_gpu(epoch, net, data_iter, device=None):
    pred_labs = []
    real_labs = []
    val_l = 0.
    net.eval()
    roc_poslist = []
    reallab = []
    val_loss = []
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (seqmatrix, label, pssms, dssps, emd, graph) in enumerate(data_iter):
            # data = [d.to(device) for d in data]
            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
            for yh in y_hat:
                roc_poslist.append(Tensor.cpu(yh[1]).numpy())
            real_l = label.argmax(dim=1)
            for rl in real_l:
                reallab.append(Tensor.cpu(rl).numpy())
            pred = y_hat.argmax(dim=1)
            l = loss(y_hat, label.to(device))
            list_len = len(pred)
            li = []
            for i in range(list_len):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            lab = torch.zeros(list_len, 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(lab)
            real_labs.append(label.to(device))
            val_l += l
    # val_loss.append(val_l / len(data_iter))
    assess = calculate_indicators(epoch, pred_labs, real_labs)
    auccal = calculate_auc(roc_poslist, reallab)
    print(
        "Test  Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
            epoch, val_l / len(data_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
            assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))
    print(
        "---------------------------------------------------------------------------------------------------------------------------------------------------")

def train(train_file_name, device=torch.device('cuda')):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    # 批量装载数据集
    print("加载训练数据集:" + train_file_name)
    train_data = LuDataset(train_file_name=train_file_name)
    train_iter = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=collate)
    print("加载验证数据集:"+val_file)
    val_data = LuDataset(train_file_name=val_file)
    val_iter = DataLoader(dataset=val_data, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=collate)

    print("加载测试数据集："+test_file)
    test_data = LuDataset(train_file_name=test_file)
    test_iter = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False, drop_last=True, collate_fn=collate)

    net = LuGTP().to(device)
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
    loss = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        net.train()
        train_l = 0.
        pred_labs = []
        real_labs = []
        train_loss = []
        for batch_idx, (seqmatrix, label, pssms, dssps, emd, graph) in enumerate(train_iter):
            optimizer.zero_grad()
            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
            pred = y_hat.argmax(dim=1)
            li = []
            for i in range(len(pred)):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            pre_lab = torch.zeros(len(pred), 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(pre_lab)
            l = loss(y_hat, label.to(device))
            l.backward()
            train_l += l
            real_labs.append(label.to(device))
            optimizer.step()
        # train_loss.append(train_l / len(train_iter))
        assess = calculate_indicators(epoch, pred_labs, real_labs)
        print(
            "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
                epoch, train_l / len(train_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
                assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))
        evaluate_val_accuracy_gpu(epoch, net, val_iter, device)
        evaluate_test_accuracy_gpu(epoch, net, test_iter, device)

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
