# NPC1 Variant Fitness Prediction with Evo2

Predicts functional scores for NPC1 variants using DNA embeddings from the [Evo2](https://github.com/ArcInstitute/evo2) genomic foundation model. Supports missense, nonsense, insertion, deletion, duplication, and splice variants from ClinVar or custom input.

---

## Pipeline overview

```
Variants (xlsx / ClinVar TSV / CLI coords)
        ↓
  prepare_dataset.py       → DNA windows (.npy)
        ↓
  extract_embeddings.py    → Evo2 embeddings (.npy)
        ↓
  regressor.py / run_all.py → trained model + metrics CSV
        ↓
  predict.py               → predictions on new variants
```

---

## Setup

### Local (Mac/Linux)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas biopython requests scikit-learn scipy lightgbm joblib
pip install git+https://github.com/ArcInstitute/evo2.git
```

### HPC (no conda/pip access) — Singularity
Build locally with Docker, convert to Singularity on HPC:
```bash
# Build image locally
docker build -t evo2_env .
docker save evo2_env | gzip > evo2_env.tar.gz

# On HPC
singularity build evo2_env.sif docker-archive://evo2_env.tar.gz
```

---

## Scripts

### 1. `prepare_dataset.py` — generate DNA sequence windows

Fetches the chromosome 18 reference from NCBI and extracts an 8192 bp window centered on each variant (ref and mutant, forward and reverse strand).

```bash
# From ClinVar TSV (2-star NPC1 variants)
python prepare_dataset.py --tsv data/npc1_2star_GRCh38.tsv

# From Excel (original fitness score dataset)
python prepare_dataset.py --xlsx data/NPC1_mut_fitness_scores.xlsx

# Single variant from CLI
python prepare_dataset.py --coords chr18:23535479:T:C

# Use a local FASTA instead of fetching from NCBI
python prepare_dataset.py --tsv data/npc1_2star_GRCh38.tsv --fasta data/chr18.fa
```

**Output** (in `output/`, timestamped):
```
ref_seq_DNA_forward_YYYYMMDD.npy
ref_seq_DNA_reverse_YYYYMMDD.npy
mut_seq_DNA_forward_YYYYMMDD.npy
mut_seq_DNA_reverse_YYYYMMDD.npy
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tsv` | — | ClinVar TSV with `PositionVCF`/`ReferenceAlleleVCF`/`AlternateAlleleVCF` columns |
| `--xlsx` | `data/NPC1_mut_fitness_scores.xlsx` | Excel file (used if no source given) |
| `--coords` | — | One or more `CHR:POS:REF:ALT` strings |
| `--fasta` | — | Local FASTA; fetches from NCBI if omitted |
| `--accession` | `NC_000018.10` | NCBI accession for chr18 |
| `--window` | `8192` | Window size in bp |
| `--out_dir` | `output/` | Output directory |

---

### 2. `extract_embeddings.py` — extract Evo2 hidden-state embeddings

Runs Evo2 forward passes and saves mean-pooled or last-token embeddings per layer.

```bash
# Default: evo2_7b, layer blocks.28.mlp.l3, forward strand
python extract_embeddings.py

# Specify input files explicitly (recommended when using dated outputs)
python extract_embeddings.py \
    --ref-file output/ref_seq_DNA_forward_20260714.npy \
    --mut-file output/mut_seq_DNA_forward_20260714.npy \
    --strand forward

# Run on reverse strand
python extract_embeddings.py --strand reverse

# Use a different model / multiple layers
python extract_embeddings.py --model evo2_40b --layer blocks.28.mlp.l3 blocks.40.mlp.l3
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `evo2_7b` | Model variant: `evo2_1b_base`, `evo2_7b`, `evo2_40b` |
| `--layer` | `blocks.28.mlp.l3` | One or more layer names |
| `--strand` | `forward` | `forward` or `reverse` |
| `--emb-type` | `average` | `average` (mean pool) or `last` (last token) |
| `--batch-size` | `1` | Sequences per forward pass |
| `--ref-file` | auto | Override default input path for ref sequences |
| `--mut-file` | auto | Override default input path for mut sequences |

> **Note:** `evo2_7b` runs on A100 or newer (FP8 disabled automatically). `evo2_1b_base` and `evo2_40b` require H100 (compute capability ≥ 8.9).

---

### 3. `regressor.py` — train and evaluate regressors

Nested group K-fold cross-validation with hyperparameter tuning. Groups are defined by protein annotation or protein region to prevent data leakage.

```bash
# DNA delta embeddings, Ridge, forward strand
python regressor.py \
    --emb_mode dna --emb emb_7B/ \
    --delta --strand forward \
    --models Ridge

# Multiple models, PCA(50), split by protein region
python regressor.py \
    --emb_mode dna --emb emb_7B/ \
    --delta --strand forward \
    --pca 50 \
    --models Ridge SVR ElasticNet \
    --fold_by merged_region

# Both strands concatenated
python regressor.py \
    --emb_mode dna --emb emb_7B/ \
    --delta --strand both \
    --models Ridge
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--emb_mode` | `dna` | `dna` (ref+mut .npy) or `rna` (single .npy) |
| `--emb` | — | Embedding directory (DNA) or .npy file (RNA) |
| `--delta` | off | Use `mut - ref` embeddings; without flag uses `mut \|\| ref` |
| `--strand` | `forward` | `forward`, `reverse`, or `both` |
| `--fold_by` | `Protein Annotation` | Grouping column: `Protein Annotation` or `merged_region` |
| `--pca` | — | PCA components before regression |
| `--models` | `Ridge` | One or more model names |
| `--model_dir` | — | Save best model per run as `<tag>_YYYYMMDD_HHMMSS.joblib` |
| `--layer` | — | Filter embeddings by layer index |
| `--pooling` | — | Filter embeddings by pooling mode (`average`/`last`) |

> When `--fold_by merged_region` is used and only 2 groups exist, a simple 2-fold CV is used instead of nested CV.

**Available models:** `Ridge`, `Lasso`, `ElasticNet`, `KernelRidge`, `SVR`, `PLS`, `GaussianProcess`, `kNN`, `RandomForest`, `DecisionTree`, `Dummy`

---

### 4. `run_all.py` / `run_all.sbatch` — sweep all layer/pooling combinations

Iterates over every layer and pooling mode found in the embedding directory and runs all regressors, saving a single `metrics.csv`.

```bash
# Local
python run_all.py --emb emb_7B/ --out_dir results/ --model_dir saved_models/

# HPC (SLURM)
sbatch run_all.sbatch
```

---

### 5. `predict.py` — run predictions with a saved model

```bash
# With ground-truth labels (reports Spearman / MSE / MAE)
python predict.py \
    --model saved_models/Ridge__delta__forward_20260714_143022.joblib \
    --emb emb_to_test/ \
    --df output/df_preprocessed.csv \
    --strand forward --delta --layer 27 --pooling average \
    --out predictions.csv

# Inference only (no labels)
python predict.py \
    --model saved_models/Ridge__delta__forward_20260714_143022.joblib \
    --emb emb_to_test/ \
    --strand forward --delta
```

---

## Data

| File | Description |
|---|---|
| `data/NPC1_mut_fitness_scores.xlsx` | NPC1 variant fitness scores (training labels) |
| `data/npc1_2star_GRCh38.tsv` | ClinVar 2-star NPC1 variants (GRCh38) |
| `output/df_preprocessed.csv` | Preprocessed dataframe with Function Score and grouping columns |

---

## HPC job submission

```bash
# Submit embedding + regression pipeline
sbatch run_all.sbatch

# Run extract_embeddings inside Singularity container
singularity exec --nv \
    --bind $PWD:/workspace \
    --bind $HOME/.cache/huggingface:/root/.cache/huggingface \
    evo2_env.sif \
    python /workspace/extract_embeddings.py \
        --ref-file /workspace/output/ref_seq_DNA_forward_20260714.npy \
        --mut-file /workspace/output/mut_seq_DNA_forward_20260714.npy
```

> Model weights (~15 GB for 7B) are downloaded on first use to `~/.cache/huggingface`. Mount this directory to avoid re-downloading across jobs.
