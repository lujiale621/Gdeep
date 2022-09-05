import os

import numpy
import numpy as np

from config import *
dictdata={}
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            print(f)
            g_embed = np.load(emb_path+str(f))
            protein=str(f).split('.')
            dictdata[protein[0]]=g_embed['data']
    numpy.savez('D:/embdict', **dictdata)

# findAllFile(emb_path)

class Embdict():
    def __init__(self):
        self.g_embed =np.load(embdict_path)

    def getTag(self,protein_name):
        try:
            return self.g_embed[protein_name]
        except:
            return None
class Emdata():
    def __init__(self):
        self.dict = {}
        with open(emb_map, 'r') as f:
            st = f.readline()
            self.dict = eval(st)

    def getTag(self, protein_name):
        return self.dict.get(protein_name, -1)
if __name__ == '__main__':
    findAllFile(emb_path)