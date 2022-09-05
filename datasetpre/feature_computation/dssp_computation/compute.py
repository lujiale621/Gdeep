import numpy as np


def extract_lines(pssmFile):
    fin = open(pssmFile)
    pssmLines = []
    if fin == None:
        return
    while True:
        psspLine = fin.readline()
        if psspLine.strip() == '' or psspLine.strip() == None:
            break
        pssmLines.append(psspLine)
    fin.close()
    return pssmLines


def loaddsspfile(dssp_file_url, protein, seqlen):
    dssp_fn = dssp_file_url + "/" + protein + ".dssp"
    fin = open(dssp_fn, "r")
    dsspLines = extract_lines(dssp_fn)
    dssp_np_2D = np.zeros(shape=(9, 51))  # 标准的 20 * 51 0矩阵
    less = 51 - len(dsspLines)
    # 然后应该跳过多少行
    skipLen = int(less / 2)
    for i in range(51):
        if len(dsspLines) == 0:
            continue
        if i < skipLen or i > 51 - skipLen - 1:
            # 创建一个0向量
            # values_20 = [0] * 20
            continue
        else:
            values_9 = dsspLines[i - skipLen].split()[2:11]
        for aa_index in range(8):
            dssp_np_2D[aa_index][i] = float(values_9[aa_index])
    fin.close()
    return dssp_np_2D
