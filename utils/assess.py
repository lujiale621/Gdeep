import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from torch import Tensor

def calculate_indicators(epoch, pre_lablist, real_lablist):
    try:
        # pre_lablist是标签预测值 real_lablist标签真实值
        TP = 0  # “预测为正样本，并且实际是正样本预测对了”（真阳性）
        TN = 0  # “预测为负样本，而且实际是负样本预测对了”（真阴性）
        FP = 0  # 意思是“预测为正样本，但是实际是负样本，预测错了”（假阳性）
        FN = 0  # “预测为负样本，但是实际是正样本，预测错了”（假阴性）
        for pre_lab, real_lab in zip(pre_lablist, real_lablist):
            for pl, rl in zip(pre_lab, real_lab):

                # s = torch.tensor([1., 0.]).to(torch.device('cuda'))
                # ps = torch.eq(pl, s)

                if (torch.eq(pl, torch.tensor([0., 1.]).to(torch.device('cuda'))).sum()) / 2 == 1:
                    if (torch.eq(pl, rl).sum()) / 2 == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                elif (torch.eq(pl, torch.tensor([1., 0.]).to(torch.device('cuda'))).sum()) / 2 == 1:
                    if (torch.eq(pl, rl).sum()) / 2 == 1:
                        TN = TN + 1
                    else:
                        FN = FN + 1
        accuracy = (TP + TN) / (TP + TN + FP + FN)  # (所有预测正确的正例和负例，占所有样本的比例)
        precision = TP / (TP + FP+ 1e-6)  # (预测为正例并且确实是正例的部分，占所有预测为正例的比例)
        recall = TP / (TP + FN+ 1e-6)  # (预测为正例并且确实是正例的部分，占所有确实是正类的比例)
        f1 = 2 * (precision * recall) / (precision + recall+ 1e-6)  # (精确率与召回率的调和平均数，计算公式同Dice相似系数)
        sensitivity = TP / (TP + FN+ 1e-6)  # (灵敏度，同召回率)
        specificity = TN / (TN + FP+ 1e-6)  # (预测为负例并且确实是负例的部分，占所有确实为负例的比例)

        # (我们比较关心正样本，要查看有多少负样本被错误的预测为正样本)
        # ROC (Receiver Operating Characteristic) 曲线，又称接受者操作特征曲线。该曲线最早应用于雷达信号检测领域，用于区分信号与噪声，ROC曲线是基于混淆矩阵得出的。横坐标为假正率(FPR)，纵坐标为真正率(TPR)。
        # TPR越高，同时FPR越低（即ROC曲线越陡），那么模型的性能就越好
        par = {"epoch": epoch, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
               "sensitivity": sensitivity,
               "specificity": specificity}
    except Exception as r:
        print('错误跳过%s' % (r))
        par = {"epoch": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "sensitivity": 0, "specificity": 0}
        return par
    return par