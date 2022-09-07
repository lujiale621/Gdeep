# # from config import protein_faste
# # from transformers import T5EncoderModel, T5Tokenizer
# # import torch
# # import time
#
# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # print("Using {}".format(device))
# # # whether to retrieve embeddings for each residue in a protein
# # # --> Lx1024 matrix per protein with L being the protein's length
# # # as a rule of thumb: 1k proteins require around 1GB RAM/disk
# # per_residue = True
# # per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings
# #
# # # whether to retrieve per-protein embeddings
# # # --> only one 1024-d vector per protein, irrespective of its length
# # per_protein = True
# # per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings
# #
# # # whether to retrieve secondary structure predictions
# # # This can be replaced by your method after being trained on ProtT5 embeddings
# # sec_struct = True
# # sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions
#
#
# class ConvNet(torch.nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # This is only called "elmo_feature_extractor" for historic reason
#         # CNN weights are trained on ProtT5 embeddings
#         self.elmo_feature_extractor = torch.nn.Sequential(
#             torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.25),
#         )
#         n_final_in = 32
#         self.dssp3_classifier = torch.nn.Sequential(
#             torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
#         )
#
#         self.dssp8_classifier = torch.nn.Sequential(
#             torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
#         )
#         self.diso_classifier = torch.nn.Sequential(
#             torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
#         )
#
#     def forward(self, x):
#         # IN: X = (B x L x F); OUT: (B x F x L, 1)
#         x = x.permute(0, 2, 1).unsqueeze(dim=-1)
#         x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
#         d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
#         d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
#         diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
#         return d3_Yhat, d8_Yhat, diso_Yhat
# def load_sec_struct_model():
#   checkpoint_dir="datasetpre/feature_computation/esm_pro/protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
#   state = torch.load( checkpoint_dir )
#   model = ConvNet()
#   model.load_state_dict(state['state_dict'])
#   model = model.eval()
#   model = model.to(device)
#   print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))
#
#   return model
# def get_T5_model():
#     model = T5EncoderModel.from_pretrained("datasetpre/feature_computation/esm_pro/Rostlab/prot_t5_xl_half_uniref50-enc")
#     model = model.to(device) # move model to GPU
#     model = model.eval() # set model to evaluation model
#     tokenizer = T5Tokenizer.from_pretrained('datasetpre/feature_computation/esm_pro/Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
#
#     return model, tokenizer
#
# def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
#                    max_residues=4000, max_seq_len=1000, max_batch=100):
#     if sec_struct:
#         sec_struct_model = load_sec_struct_model()
#
#
#     # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
#     seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
#     start = time.time()
#     batch = list()
#     for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
#         seq = seq
#         seq_len = len(seq)
#         seq = ' '.join(list(seq))
#         batch.append((pdb_id, seq, seq_len))
#
#         # count residues in current batch and add the last sequence length to
#         # avoid that batches with (n_res_batch > max_residues) get processed
#         n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
#         if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
#             pdb_ids, seqs, seq_lens = zip(*batch)
#             batch = list()
#
#             # add_special_tokens adds extra token at the end of each sequence
#             token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
#             input_ids = torch.tensor(token_encoding['input_ids']).to(device)
#             attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
#
#             try:
#                 with torch.no_grad():
#                     # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
#                     embedding_repr = model(input_ids, attention_mask=attention_mask)
#             except RuntimeError:
#                 print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
#                 continue
#
#             if sec_struct:  # in case you want to predict secondary structure from embeddings
#                 d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)
#
#             for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
#                 s_len = seq_lens[batch_idx]
#                 # slice off padding --> batch-size x seq_len x embedding_dim
#                 emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
#                 if sec_struct:  # get classification results
#                     sec_structs = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[
#                         1].detach().cpu().numpy().squeeze()
#                     print("t")
#                 if per_residue:  # store per-residue embeddings (Lx1024)
#                     residue_embs = emb.detach().cpu().numpy().squeeze()
#                     print("t")
#
#                 if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
#                     protein_emb = emb.mean(dim=0)
#                     protein_embs = protein_emb.detach().cpu().numpy().squeeze()
#                     print("t")
#     return sec_structs,residue_embs,protein_embs
#
#
#
# # Load example fasta.
# def getemd(seqs):
#     model, tokenizer = get_T5_model()
# # Compute embeddings and/or secondary structure predictions
#     results = get_embeddings( model, tokenizer, seqs,
#                              per_residue, per_protein, sec_struct)
#     return results
#
