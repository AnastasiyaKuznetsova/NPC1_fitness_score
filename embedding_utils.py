"""Helpers shared by extract_embeddings_Evo2.py, extract_embeddings_JanusDNA.py,
and extract_embeddings_DNABERT2.py.

DNABERT-2 pools via the tokenizer's `attention_mask` (BPE, so token count != base
count) rather than plain sequence lengths, so its pool()/pool_downstream() stay
local to extract_embeddings_DNABERT2.py — only parse_context_window and
parse_downstream_k are shared with it.
"""

import argparse
import os
import re

import torch


def parse_context_window(path: str) -> str:
    """Extract the '<N>bp' context-window token from a filename written by
    prepare_dataset.py (e.g. ref_seq_DNA_forward_8192bp.npy or
    variant_meta_8192bp.csv), for use in the output folder name."""
    m = re.search(r"(\d+bp)", os.path.basename(path))
    if not m:
        raise ValueError(
            f"Could not find a '<N>bp' context-window token in filename {path!r}. "
            "Expected a name like '..._8192bp.npy' as written by prepare_dataset.py."
        )
    return m.group(1)


def parse_downstream_k(value: str):
    if value == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--downstream-k values must be 'all' or an integer, got {value!r}")


def pool(hidden: torch.Tensor, lengths: list[int], emb_type: str) -> torch.Tensor:
    """Apply pooling to a (B, L, D) hidden state tensor over the full sequence,
    respecting padding via per-sequence lengths."""
    if emb_type == "average":
        mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=hidden.device)
        for b, l in enumerate(lengths):
            mask[b, :l] = True
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
    else:  # last
        last_indices = torch.tensor([l - 1 for l in lengths], device=hidden.device)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]


def pool_downstream(hidden: torch.Tensor, lengths: list[int], starts: list[int], k) -> torch.Tensor:
    """Mean over positions [start, start+k] per sequence (inclusive — k=0 is
    just the position at `starts`; k == "all" means [start, length)). Caller
    derives `starts` (e.g. the allele's own last base for a forward-causal
    stream)."""
    B, L = hidden.shape[0], hidden.shape[1]
    region_start = [min(s, l) for s, l in zip(starts, lengths)]
    region_end = lengths if k == "all" else [min(s + k + 1, l) for s, l in zip(starts, lengths)]
    if any(e <= s for s, e in zip(region_start, region_end)):
        print("  WARNING: empty downstream region for at least one sequence "
              "(edit sits at/past the sequence end) — that row's pooled vector will be all zeros")
    mask = torch.zeros((B, L), dtype=torch.bool, device=hidden.device)
    for b, (s, e) in enumerate(zip(region_start, region_end)):
        mask[b, s:e] = True
    mask_expanded = mask.unsqueeze(-1).float()
    return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
