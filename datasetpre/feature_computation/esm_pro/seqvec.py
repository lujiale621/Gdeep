# import torch
#
# from pathlib import Path
#
# from datasetpre.feature_computation.esm_pro.elmo import ElmoEmbedder
#
#
# def protein_emd(data):
#     model_dir = Path('D:/data/model')
#     weights = model_dir / 'weights.hdf5'
#     options = model_dir / 'options.json'
#     embedder = ElmoEmbedder(options, weights, cuda_device=-1)  # cuda_device=-1 for CPU
#     seq = data[1]
#     protein_name=data[0]
#     embedding = embedder.embed_sentence(list(seq))
#     residue_embd = torch.tensor(embedding).sum(dim=0)
#     protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)  # Vector with shape [1024]
#     return protein_name,residue_embd,protein_embd
# if __name__ == '__main__':
#     protein_emd(None)