import dgl
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.data.data
from datasetpre.dataprocess import listclass_to_one_hot, productGraph


def pygdatasetloader(data):
    loader = DataLoader(
        data,
        shuffle=True,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128
    )
    return loader
def dglrandomloaders(data):
    # Glist, labels, postions = data[0], data[1], data[2]
    # train_l = 0.
    # for i in range(len(Glist)):
    #     g = Glist[i]
    #     label = labels[i]
    #     label = label['glabel']
    #     label = listclass_to_one_hot(label, False)
    #     postion = postions[i]
    #     # train_nid = torch.tensor(g.nodes()[[postion]])
    #     train_nid = torch.tensor(g.nodes())
    #     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    #     dataloader = dgl.dataloading.NodeDataLoader(
    #         g, train_nid, sampler,
    #         batch_size=128,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=0)
    #     input = Glist[i].ndata['feat'].shape[1]
    #     model = SAGEConvq(input, 512, 128).to(device)
    #     for minibatch in dataloader:
    #         pred_labs = []
    #         real_labs = []
    #         opt = torch.optim.Adam(model.parameters())
    #         input_nodes, output_nodes, mfgs = minibatch
    #         print(minibatch)
    #         print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes),
    #                                                                                       len(input_nodes)))
    #         mfgs = [b.to(torch.device('cuda')) for b in mfgs]
    #         inputs = mfgs[0].srcdata['feat']  # returns a dict
    #
    #         output_labels = label.to(device)
    #         predictions = model(mfgs, inputs)
    #         pred = predictions.argmax(dim=1)
    #         li = []
    #         for i in range(len(pred)):
    #             li.append([pred[i]])
    #         one_hot_list = torch.LongTensor(li).to(device)
    #         pre_lab = torch.zeros(len(pred), 2).to(device).scatter_(1, one_hot_list, 1)
    #         pred_labs.append(pre_lab)
    #         real_labs.append(output_labels.to(device))
    #         loss = F.cross_entropy(predictions, output_labels)
    #         opt.zero_grad()
    #         loss.backward()
    #         train_l += loss
    #         opt.step()
    #         assess = calculate_indicators(i, pred_labs, real_labs)
    #         print(
    #             "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
    #                 i, train_l / len(dataloader), assess.get("accuracy"), assess.get("precision"),
    #                 assess.get("recall"),
    #                 assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))
    return None