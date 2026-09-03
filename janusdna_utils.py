"""JanusDNA model-loading helpers shared by extract_embeddings_JanusDNA.py and
janusdna_zero_shot.py. See extract_embeddings_JanusDNA.py's module docstring
for the full architecture/context-length background these values come from.
"""

import json
import os

import torch

# Metadata that ISN'T part of JanusDNAConfig and so can't be read back out of
# a checkpoint's own config JSON — params_label (activated params) and
# context_length (the model's own pretraining sequence length — NOT uniform
# across presets) are both from Table 9 of the JanusDNA paper
# (arXiv:2505.17257): only 144dim_nomidattn was pretrained long-context
# (131,072 bp); every 32dim/72dim preset was pretrained at just 1,024 bp.
# Keyed to match infer_preset_key()'s "{hidden_size}dim_{midattn|nomidattn}"
# naming, so it can look this up for a config it never built itself.
MODEL_PRESETS = {
    "32dim_midattn":    dict(params_label="~0.42M", context_length=1024),
    "32dim_nomidattn":  dict(params_label="~0.43M", context_length=1024),
    "72dim_midattn":    dict(params_label="~1.98M", context_length=1024),
    "72dim_nomidattn":  dict(params_label="~1.99M", context_length=1024),
    "144dim_nomidattn": dict(params_label="~7.66M", context_length=131072),
}
N_LAYER = 8  # fixed across every released checkpoint — used for CLI --layer validation

# CharacterTokenizer scheme from src/dataloaders/datasets/hg38_char_tokenizer.py
# (characters=['A','C','G','T','N'], ids assigned after 7 reserved specials).
SPECIAL_IDS = {"[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3, "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6}
CHAR_IDS = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
PAD_ID = SPECIAL_IDS["[PAD]"]
UNK_ID = SPECIAL_IDS["[UNK]"]


def tokenize(seq: str) -> list[int]:
    return [CHAR_IDS.get(base, UNK_ID) for base in seq.upper()]


def find_sibling_config_json(checkpoint_path: str) -> str | None:
    """Dataverse ships each checkpoint with a sibling '<stem>_model_config.json'
    (e.g. 144_without_midattn.ckpt -> 144_without_midattn_model_config.json).
    Returns that path if it sits next to checkpoint_path, else None."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    candidate = os.path.join(os.path.dirname(checkpoint_path), f"{stem}_model_config.json")
    return candidate if os.path.isfile(candidate) else None


def infer_preset_key(config) -> str | None:
    """Reverse-maps a loaded JanusDNAConfig back to a MODEL_PRESETS key, by
    matching hidden_size and whether any layer is a real attention layer
    (config.layers_block_type — a JanusDNAConfig property, so this works
    whether config came from a checkpoint's own JSON or a preset). Returns
    None if no known preset matches (e.g. an architecture MODEL_PRESETS
    doesn't cover), since context_length is training metadata that can't be
    recovered from the architecture alone in that case."""
    has_midattn = "attention" in config.layers_block_type
    key = f"{config.hidden_size}dim_{'midattn' if has_midattn else 'nomidattn'}"
    return key if key in MODEL_PRESETS else None


def build_config(vocab_size: int, JanusDNAConfig, config_json: str):
    """Builds the architecture from config_json — the real config Dataverse
    ships next to each checkpoint (see find_sibling_config_json). Requires
    that file (no manual-architecture fallback, since a hand-maintained
    hidden_size/intermediate_size/... guess would just drift from whatever
    Dataverse actually shipped). vocab_size is overridden with the value
    inferred from the checkpoint itself (infer_vocab_size).

    Dataverse's JSON is NOT the flat dump JanusDNAConfig.from_json_file()
    expects — it nests everything under a "config" key alongside a Hydra
    "_target_" marker, e.g. {"config": {"_target_": "...", "hidden_size": 32,
    ...}}. from_json_file() would parse that whole nested dict as a single
    unrecognized `config=` kwarg (JanusDNAConfig has no such parameter),
    which PretrainedConfig silently absorbs via **kwargs — so every real
    field would silently stay at its class default instead of erroring. This
    unwraps that nesting and constructs JanusDNAConfig from the real fields.
    """
    with open(config_json) as f:
        raw = json.load(f)
    fields = dict(raw.get("config", raw))
    fields.pop("_target_", None)
    # A training-script quirk in Dataverse's dump: bidirectional is sometimes
    # the literal string "true," (trailing comma) instead of a bool.
    if isinstance(fields.get("bidirectional"), str):
        fields["bidirectional"] = fields["bidirectional"].strip(", ").lower() == "true"

    config = JanusDNAConfig(**fields)
    config.vocab_size = vocab_size
    # sdpa avoids needing flash-attn / a working torch.compile'd flex_attention
    # just to run mid-stack attention sublayers on midattn checkpoints.
    config._attn_implementation = "sdpa"
    print(f"Loaded architecture from {config_json}: hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}, num_hidden_layers={config.num_hidden_layers}")
    return config


def load_checkpoint(model, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_keys = set(model.state_dict().keys())
    if not (set(state_dict.keys()) & model_keys):
        # Common wrapper prefixes (e.g. Lightning's "model.model." or
        # DNAEmbeddingModelJanusDNA's "model.") — strip until keys line up.
        for prefix in ("model.model.", "model."):
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if set(stripped.keys()) & model_keys:
                state_dict = stripped
                break

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint {checkpoint_path}: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")


def infer_vocab_size(checkpoint_path: str) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    for key, tensor in state_dict.items():
        if key.endswith("embed_tokens.weight"):
            return tensor.shape[0]
    default = len(SPECIAL_IDS) + len(CHAR_IDS)
    print(f"WARNING: could not find embed_tokens.weight in checkpoint to infer vocab_size; "
          f"defaulting to {default}.")
    return default
