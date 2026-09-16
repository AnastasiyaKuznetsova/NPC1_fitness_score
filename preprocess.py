"""
This script merges separate 1xdim embeddings to a single file for reference and mutation sequences
"""

import numpy as np
import os

path = 'evo2_emb_1b_layer_24'
path_to_save = 'DNA_Evo2_1b_layer_24_preprocessed'

files = os.listdir(path)
max_ind = max([int(x.split("_")[-1].split(".npy")[0]) for x in files])
print(max_ind)


muts = []
refs = []

for i in range(max_ind+1):
    ref_matrix = np.load(os.path.join(path, f"ref_seq_emb_DNA_avg_True_{i}.npy"))
    mut_matrix = np.load(os.path.join(path, f"mut_seq_emb_DNA_avg_True_{i}.npy"))

    refs.append(ref_matrix)
    muts.append(mut_matrix)

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

np.save(f"{path_to_save}/refs.npy", np.array(refs))
np.save(f"{path_to_save}/muts.npy", np.array(muts))