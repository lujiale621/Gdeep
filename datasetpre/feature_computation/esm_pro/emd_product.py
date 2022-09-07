# import torch
# import esm

def getproteinemd(mdata):
    data=[]
    data.append(mdata)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))
    protein_name=[]
    protein_seq = []
    protein_attention=[]
    for (name, seq), attention_contacts in zip(data, results["contacts"]):
        protein_seq.append(seq)
        protein_attention.append(attention_contacts)
        protein_name.append(name)
    leng=len(protein_seq[0])
    return protein_name[0],token_representations[0][1:leng+1],protein_attention[0],protein_seq[0]
if __name__ == '__main__':
    da= ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    dat=getproteinemd(da)
    print(dat)
