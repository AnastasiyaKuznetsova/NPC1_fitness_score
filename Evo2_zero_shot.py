import torch
from transformer_engine.common.recipe import _OverrideLinearPrecision
torch.serialization.add_safe_globals([_OverrideLinearPrecision])

from evo2.models import Evo2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

model = Evo2("evo2_7b")
# Let's do for forward
var_seqs = np.load('output/mut_seq_DNA_forward_20260825.npy')
ref_seqs = np.load('output/ref_seq_DNA_forward_20260825.npy')
df = pd.read_csv('output/df_preprocessed.csv')

print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
ref_scores = model.score_sequences(ref_seqs)

print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
var_scores = model.score_sequences(var_seqs)

delta_scores = np.array(var_scores) - np.array(ref_scores)
print(spearmanr(df['Function Score'], delta_scores))