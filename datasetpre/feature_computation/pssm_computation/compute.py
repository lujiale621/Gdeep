import os
import pickle

import numpy as np
import time
import sys
import math



# def LoadPSSM():
#     global min_value, max_value
#     fin = open(pssm_fn, "r")

def load_fasta_and_compute(protein, position, raw_pssm_dir, start, end, l_padding=0, r_padding=0):
    pssm = []
    pssm_fn = raw_pssm_dir + "/" + protein + '_' + str(position) + ".pssm"
    pssm = LoadPSSMandPrintFeature(pssm_fn, protein, end - start)
    # pssm=LoadPSSM()
    # fout.write(line_Pid)
    # fout.write(line_Pseq)n

    # fout.write(",".join(map(str,Feature)) + "\n")

    # pssm2 = pssm[:,int(start):int(end)]

    # pssm2 = pssm
    # if l_padding>0:
    #     newRows=np.zeros((20,int(l_padding)))
    #     pssm2=np.c_[newRows,pssm2]
    # if r_padding>0:
    #     newRows=np.zeros((20,int(r_padding)))
    #     pssm2=np.c_[pssm2,newRows]

    return pssm


def extract_lines(pssmFile):
    fin = open(pssmFile)
    pssmLines = []
    if fin == None:
        return
    for i in range(3):
        fin.readline()  # exclude the first three lines
    while True:
        psspLine = fin.readline()
        if psspLine.strip() == '' or psspLine.strip() == None:
            break
        pssmLines.append(psspLine)
    fin.close()
    return pssmLines


def LoadPSSMandPrintFeature(pssm_fn, Pid, line_Pseq):
    global min_value, max_value
    fin = open(pssm_fn, "r")
    pssmLines = extract_lines(pssm_fn)
    # print(pssmLines)KK
    seq_len = len(pssmLines)

    # pssm_np_2D = np.zeros(shape=(20, seq_len)) # 你看这里不是一个初始的0矩阵吗，这里写死他就好了 所以不可以用seqlen 因为你有一些seqlen是小于51的
    pssm_np_2D = np.zeros(shape=(20, 51))  # 标准的 20 * 51 0矩阵
    # 然后计算一个少了多少行 这个less永远都是偶数对吧
    less = 51 - len(pssmLines)
    # 然后应该跳过多少行
    skipLen = int(less / 2)

    # 如果是整个矩阵得话就应该在这里用pssmlines 算min max
    for i in range(len(pssmLines)):
        values_20 = pssmLines[i].split()[2:22]
        for j in range(len(values_20)):
            max_value = max(max_value, float(values_20[j]))
            min_value = min(min_value, float(values_20[j]))

    max_value += 1
    min_value -= 1

    # 然后这个seq_len 也要改， 应该是 51才对
    # for i in range(seq_len):
    for i in range(51):  # 外层51 你不够51，那就应该假设有skiplen行是0
        # fist 69 chars are what we need
        # print(pssmLines[i][9:70])
        if i < skipLen or i > 51 - skipLen - 1:
            # 创建一个0向量
            # values_20 = [0] * 20
            continue
        else:
            values_20 = pssmLines[i - skipLen].split()[2:22]  # 这个原来的意思是拿一行的20个数值，既然在前skiplen行都是0，那只要跳过这个就可以了

            # print(values_20)
        for aa_index in range(20):
            # max_value = max(max_value, float(values_20[aa_index]))
            # min_value = min(min_value, float(values_20[aa_index]))
            pssm_np_2D[aa_index][i] = (float(values_20[aa_index]) - min_value) / (
                        max_value - min_value)  # 他这个就是将概率范围移动到正轴而已 就是求概率分布
            # 他这里算一行更新一次 所以第86行的计算结果每次都可能变
    fin.close()
    return pssm_np_2D


def main():
    seq_fn = './F1D8N4_HUMAN.fasta'
    out_base_fn = './'
    raw_pssm_dir = './'
    load_fasta_and_compute(seq_fn, out_base_fn, raw_pssm_dir)
    print("max_value: ", max_value)
    print("min_value: ", min_value)


max_value = 0.
min_value = 0.
if __name__ == '__main__':
    main()
