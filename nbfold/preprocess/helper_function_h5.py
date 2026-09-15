import h5py
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/pasang/all_experiment/nbfold')
import nbfold

project_path = os.path.abspath(os.path.join(nbfold.__file__, "../.."))
data_path = os.path.join(project_path, "data/")

def embedding():
    # Function to read embeddings from the HDF5 file
    def read_embeddings(file_path):
        emb_dict = {}
        with h5py.File(file_path, 'r') as hf:
            group = hf['embeddings']
            for key in sorted(group.keys(), key=lambda x: int(x)):
                emb_dict[group[key].attrs['sequence_id']] = group[key][:]
        return emb_dict

    # Read per-residue embeddings
    my_emb = read_embeddings(data_path + 'per_residue_embeddings.h5')

    # Extract nanobody-specific data
    with h5py.File(data_path + "antibody.h5", "r") as f:
        a_group_key = list(f.keys())[5]
        h = list(f.keys())[4]

        ds_arr = f[a_group_key][()]  # Main data
        h_arr = f[h][()]  # Heavy chain lengths

        train_data = ds_arr.tolist()
        h_len = h_arr.tolist()

    t_data = []
    t_data_len = {}
    # for i, h in zip(train_data, h_len):
    #     file = i.decode('utf-8')
    #     t_data_len[file[:4]] = h  # Only use heavy chain length
    #     t_data.append(file[:4])

    for i, h in zip(train_data, h_len):
        file = i.decode('utf-8')
        file_base = os.path.splitext(file)[0]  # removes .pdb or .fasta
        t_data_len[file_base] = h  # Use full file name without extension
        t_data.append(file_base)


    # Process embeddings for heavy chains only
    name = {}

    def add_cancat(id, my_emb):
        h = id + ":H"
        # print(my_emb)
        if h in my_emb:
            name[id] = my_emb[h]  # Use only the heavy chain embedding
        else:
            print(f"Warning: Key '{h}' not found in my_emb. Skipping.")

    for i in t_data:
        add_cancat(i, my_emb)

    # Convert numpy arrays to PyTorch tensors
    protein_embeddings = name
    new_embeddings = {
        protein_id: torch.from_numpy(embedding).transpose(1, 0)
        for protein_id, embedding in protein_embeddings.items()
    }

    # DataLoader for batched processing
    class PadSequence:
        def __call__(self, batch):
            key, data = zip(*batch)
            max_len = max(v.shape[1] for v in new_embeddings.values())
            data = [
                torch.nn.functional.pad(t, (0, max_len - t.shape[1], 0, 0), mode='constant', value=0)
                for t in data
            ]
            return key, torch.stack(data)

    class ProteinDataset:
        def __init__(self, new_embeddings):
            self.new_embeddings = new_embeddings

        def __len__(self):
            return len(self.new_embeddings)

        def __getitem__(self, idx):
            key = list(self.new_embeddings.keys())[idx]
            value = self.new_embeddings[key]
            return key, value

    dataset = ProteinDataset(new_embeddings)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PadSequence())

    return data_loader
