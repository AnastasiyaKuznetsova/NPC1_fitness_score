"""
Fine-tune Evo2 7B on NPC1 variant fitness scores.

The backbone is frozen except for LoRA adapters on every MLP's linear layers
(l1/l2/l3 of each ParallelGatedMLP, in both attention and hyena/conv blocks).
Attention projections (MHA.Wqkv/out_proj) and the hyena/conv mixer itself
(HyenaCascade, projections, out_filter_dense) are never touched. A regression
head attaches to the hidden state at any --layer of your choosing.

Architecture
------------
  Evo2 7B (frozen, LoRA on MLPs) → mask-aware mean-pool hidden state at
  --layer → MLP regression head → scalar

Training
--------
  - Loss: MAE + optional Spearman rank loss
  - Groups by Protein Annotation to prevent data leakage across folds
  - Saves best checkpoint (head + LoRA adapters) by validation Spearman

Usage
-----
  # Single GPU, LoRA on MLPs (default):
  python finetune_evo2.py --sequences output/ref_seq_DNA_forward.npy \
      --df output/df_preprocessed.csv --out_dir checkpoints/

  # Head-only, no LoRA:
  python finetune_evo2.py --sequences output/ref_seq_DNA_forward.npy \
      --df output/df_preprocessed.csv --out_dir checkpoints/ --lora_r 0

  # Attach the head to a different layer (e.g. an attention block's output):
  python finetune_evo2.py --sequences output/ref_seq_DNA_forward.npy \
      --df output/df_preprocessed.csv --out_dir checkpoints/ \
      --layer blocks.24.inner_mha_cls.out_proj

  # With mutant sequences (delta mode: mut - ref representation):
  python finetune_evo2.py \
      --sequences output/ref_seq_DNA_forward.npy \
      --mut_sequences output/mut_seq_DNA_forward.npy \
      --delta --df output/df_preprocessed.csv --out_dir checkpoints/

Requirements
------------
  pip install evo2 torch scipy scikit-learn
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
from evo2 import Evo2
from vortex.model.layers import ParallelGatedMLP
from transformer_engine.common.recipe import (  # type: ignore[attr-defined]
    _OverrideLinearPrecision, DelayedScaling, Format,
)
torch.serialization.add_safe_globals([_OverrideLinearPrecision, DelayedScaling, Format])

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VariantDataset(Dataset):
    def __init__(self, ref_seqs: list[str], scores: np.ndarray,
                 mut_seqs: list[str] | None = None):
        self.ref_seqs  = ref_seqs
        self.mut_seqs  = mut_seqs
        self.scores    = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.ref_seqs)

    def __getitem__(self, idx):
        item = {"ref": self.ref_seqs[idx], "score": self.scores[idx]}
        if self.mut_seqs is not None:
            item["mut"] = self.mut_seqs[idx]
        return item


def collate_fn(batch):
    refs   = [b["ref"]   for b in batch]
    scores = torch.stack([b["score"] for b in batch])
    muts   = [b["mut"] for b in batch] if "mut" in batch[0] else None
    return refs, muts, scores


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize_batch(tokenizer, seqs: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Character-level tokenize + right-pad to the batch's longest sequence.
    Returns (input_ids, padding_mask) — padding_mask is 1.0 for real tokens, 0.0 for pad."""
    ids_list = [tokenizer.tokenize(s) for s in seqs]
    max_len  = max(len(ids) for ids in ids_list)
    input_ids    = torch.full((len(seqs), max_len), tokenizer.pad_id, dtype=torch.long)
    padding_mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, ids in enumerate(ids_list):
        input_ids[i, :len(ids)]    = torch.tensor(ids, dtype=torch.long)
        padding_mask[i, :len(ids)] = 1.0
    return input_ids.to(device), padding_mask.to(device)


class _EarlyExit(Exception):
    """Raised from the capture hook to abort the rest of the forward pass once
    layer_name's output is in hand — lets --layer point at an early/mid block
    without paying for every block after it."""


def _hidden_at_layer(raw_model: nn.Module, layer_name: str,
                      input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Forward raw_model and capture the (B, L, D) hidden state at layer_name via a hook,
    then abort the remaining blocks (this is a no-op if layer_name is the last block).

    Calls raw_model directly instead of the Evo2 wrapper's __call__/forward, which
    hardcodes torch.no_grad() and would silently block gradients from ever reaching
    the LoRA adapters."""
    captured = {}

    def hook(_, __, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
        raise _EarlyExit()

    handle = raw_model.get_submodule(layer_name).register_forward_hook(hook)
    try:
        raw_model(input_ids, padding_mask=padding_mask)
    except _EarlyExit:
        pass
    finally:
        handle.remove()
    return captured["h"]


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Frozen nn.Linear + a trainable low-rank update: y = W_0 x + (alpha/r) * B(A(x))."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        device, dtype = base.weight.device, base.weight.dtype
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A  = nn.Linear(base.in_features,  r, bias=False, device=device, dtype=dtype)
        self.lora_B  = nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)  # start as a no-op: base output unchanged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def add_lora_to_mlps(model: nn.Module, r: int, alpha: float, dropout: float) -> list[LoRALinear]:
    """Wrap l1/l2/l3 of every ParallelGatedMLP (present in both attention blocks and
    hyena/conv blocks) with LoRA. Attention (MHA.Wqkv/out_proj) and the hyena/conv
    mixer (HyenaCascade filter, projections, out_filter_dense) are left untouched."""
    lora_modules = []
    for module in model.modules():
        if isinstance(module, ParallelGatedMLP):
            for name in ("l1", "l2", "l3"):
                wrapped = LoRALinear(getattr(module, name), r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
                lora_modules.append(wrapped)
    return lora_modules


# ── Regression head ───────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class Evo2Regressor(nn.Module):
    def __init__(self, backbone, layer_name: str, hidden_dim: int,
                 delta: bool = False, dropout: float = 0.1,
                 lora_modules: list[LoRALinear] = ()):
        super().__init__()
        self.raw_model    = backbone.model      # underlying StripedHyena nn.Module
        self.tokenizer    = backbone.tokenizer
        self.layer_name   = layer_name
        self.delta        = delta
        self.lora_modules = list(lora_modules)  # for train()/eval() toggling only —
                                                 # already registered via raw_model
        self.head         = RegressionHead(hidden_dim * (2 if delta else 1), dropout)

    def set_lora_training(self, mode: bool) -> None:
        for m in self.lora_modules:
            m.train(mode)

    def _encode(self, seqs: list[str]) -> torch.Tensor:
        """Tokenize, run the raw backbone, and mask-aware mean-pool the hidden
        state at the target layer."""
        device = next(self.raw_model.parameters()).device
        input_ids, padding_mask = _tokenize_batch(self.tokenizer, seqs, device)
        h = _hidden_at_layer(self.raw_model, self.layer_name, input_ids, padding_mask)
        mask = padding_mask.unsqueeze(-1)                          # (B, L, 1)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)

    def forward(self, ref_seqs: list[str],
                mut_seqs: list[str] | None = None) -> torch.Tensor:
        ref_emb = self._encode(ref_seqs)
        if self.delta and mut_seqs is not None:
            mut_emb = self._encode(mut_seqs)
            emb = mut_emb - ref_emb
        elif mut_seqs is not None:
            mut_emb = self._encode(mut_seqs)
            emb = torch.cat([ref_emb, mut_emb], dim=-1)
        else:
            emb = ref_emb
        return self.head(emb)


# ── Spearman rank loss (differentiable approximation) ─────────────────────────

def spearman_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft rank correlation loss (1 - Spearman). Differentiable via soft-rank."""
    n = pred.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=pred.device)
    pred_rank   = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()
    pred_rank   = pred_rank   - pred_rank.mean()
    target_rank = target_rank - target_rank.mean()
    cos = (pred_rank * target_rank).sum() / (
        pred_rank.norm() * target_rank.norm() + 1e-8
    )
    return 1.0 - cos


# ── Training loop ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> tuple[float, float]:
    model.head.eval()
    model.set_lora_training(False)
    preds, targets = [], []
    with torch.no_grad():
        for ref_seqs, mut_seqs, scores in loader:
            scores = scores.to(device)
            out    = model(ref_seqs, mut_seqs)
            preds.append(out.cpu().numpy())
            targets.append(scores.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    r, _    = spearmanr(preds, targets)
    mae     = float(np.mean(np.abs(preds - targets)))
    return float(r), mae


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    sep = "\t" if args.df.endswith(".tsv") else ","
    df  = pd.read_csv(args.df, sep=sep)
    scores = df["Function Score"].to_numpy(dtype=np.float32)
    groups = df["Protein Annotation"].to_numpy()

    ref_seqs = list(np.load(args.sequences, allow_pickle=True))
    mut_seqs = list(np.load(args.mut_sequences, allow_pickle=True)) \
               if args.mut_sequences else None

    assert len(ref_seqs) == len(scores), \
        f"Sequence count ({len(ref_seqs)}) != score count ({len(scores)})"

    # ── Train / val split (group-aware) ───────────────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
    train_idx, val_idx = next(splitter.split(ref_seqs, scores, groups))

    def subset(lst, idx):
        return [lst[i] for i in idx] if lst is not None else None

    train_ds = VariantDataset(subset(ref_seqs, train_idx), scores[train_idx],
                               subset(mut_seqs, train_idx))
    val_ds   = VariantDataset(subset(ref_seqs, val_idx),   scores[val_idx],
                               subset(mut_seqs, val_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    logging.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Load Evo2 ──────────────────────────────────────────────────────────────
    logging.info(f"Loading Evo2 model: {args.model} ...")
    backbone = Evo2(args.model)
    backbone.model.to(device)

    # Freeze all backbone parameters, then re-open the MLP linears we want to adapt
    for p in backbone.model.parameters():
        p.requires_grad_(False)
    backbone.model.eval()
    logging.info("Backbone frozen.")

    lora_modules = []
    if args.lora_r > 0:
        lora_modules = add_lora_to_mlps(backbone.model, r=args.lora_r,
                                        alpha=args.lora_alpha, dropout=args.lora_dropout)
        logging.info(f"LoRA: wrapped {len(lora_modules)} MLP linear layers "
                     f"(r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    else:
        logging.info("LoRA disabled (--lora_r 0) — backbone stays fully frozen.")

    # Infer hidden dim from a dummy forward pass
    dummy_ids, dummy_mask = _tokenize_batch(backbone.tokenizer, ["ACGT"], device)
    with torch.no_grad():
        hidden_dim = _hidden_at_layer(backbone.model, args.layer, dummy_ids, dummy_mask).shape[-1]
    logging.info(f"Hidden dim at {args.layer}: {hidden_dim}")

    model = Evo2Regressor(backbone, layer_name=args.layer, hidden_dim=hidden_dim,
                          delta=args.delta, dropout=args.dropout, lora_modules=lora_modules)
    model.to(device)

    # Train the head + LoRA adapters only; the rest of the backbone stays frozen
    trainable_params = list(model.head.parameters())
    for m in lora_modules:
        trainable_params += [p for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    mae_loss = nn.L1Loss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_r = -np.inf
    best_ckpt  = out_dir / "best_head.pt"

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.head.train()
        model.set_lora_training(True)
        epoch_loss = 0.0

        for step, (ref_seqs_b, mut_seqs_b, scores_b) in enumerate(train_loader):
            scores_b = scores_b.to(device)
            optimizer.zero_grad()

            preds = model(ref_seqs_b, mut_seqs_b)

            loss = mae_loss(preds, scores_b)
            if args.spearman_weight > 0:
                loss = loss + args.spearman_weight * spearman_loss(preds, scores_b)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        val_r, val_mae = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        logging.info(
            f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"val Spearman={val_r:.4f} | val MAE={val_mae:.4f}"
        )

        if val_r > best_val_r:
            best_val_r = val_r
            torch.save({"epoch": epoch,
                        "head_state": model.head.state_dict(),
                        "lora_state": [m.state_dict() for m in lora_modules],
                        "val_r": val_r, "val_mae": val_mae,
                        "args": vars(args)},
                       best_ckpt)
            logging.info(f"  → New best checkpoint (val Spearman={val_r:.4f})")

    logging.info(f"\nBest val Spearman: {best_val_r:.4f}")
    logging.info(f"Checkpoint saved to: {best_ckpt.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Evo2 regression head on NPC1 fitness scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="CSV/TSV with 'Function Score' and 'Protein Annotation' columns.")
    parser.add_argument("--sequences", required=True,
                        help="ref_seq_DNA_forward.npy — reference sequences.")
    parser.add_argument("--mut_sequences", default=None,
                        help="mut_seq_DNA_forward.npy — mutant sequences (optional).")
    parser.add_argument("--delta", action="store_true",
                        help="Use mut - ref embedding. Requires --mut_sequences.")

    # Model
    parser.add_argument("--model", default="evo2_7b",
                        choices=["evo2_1b_base", "evo2_7b", "evo2_40b"],
                        help="Evo2 model variant.")
    parser.add_argument("--layer", default="blocks.28.mlp.l3",
                        help="Layer name to attach the regression head to — any "
                             "submodule path from model.get_submodule(), e.g. "
                             "'blocks.28.mlp.l3' or 'blocks.24.inner_mha_cls.out_proj'. "
                             "Picking an earlier block skips computing every block "
                             "after it (forward pass exits early once this layer runs).")

    # LoRA — applied to every ParallelGatedMLP's l1/l2/l3 (present in both attention
    # and hyena/conv blocks). Attention projections and the conv/hyena mixer itself
    # are never touched.
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank for backbone MLP linear layers. Set to 0 to "
                             "disable LoRA and train only the regression head.")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA scaling numerator (effective scale = lora_alpha / lora_r).")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout applied before the LoRA low-rank projection.")

    # Training
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-2)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--val_frac",       type=float, default=0.2,
                        help="Fraction of data held out for validation (group-aware split).")
    parser.add_argument("--spearman_weight", type=float, default=0.5,
                        help="Weight of the soft Spearman rank loss (0 = MSE only).")

    # Output
    parser.add_argument("--out_dir", default="checkpoints/",
                        help="Directory to save the best head checkpoint.")

    args = parser.parse_args()

    if args.delta and args.mut_sequences is None:
        parser.error("--delta requires --mut_sequences")

    train(args)


if __name__ == "__main__":
    main()
