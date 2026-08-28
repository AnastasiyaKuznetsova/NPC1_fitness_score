#!/bin/bash

# Runs extract_embeddings.py across middle to late layers, both strands, and both emb-types.
# All layers are extracted in a single forward pass per strand+emb-type combination.

set -uoe pipefail

MODEL="evo2_7b"
SIF="sif/evo2.sif"
LOGDIR="logs_extract_emb_7B_1M_cw"

# Folder written by a single prepare_dataset.py run (output/<YYYYMMDD_HHMMSS>/).
# extract_embeddings_Evo2.py requires --ref-file/--mut-file whose filename
# contains a "<N>bp" token (it no longer has a usable stale default), so this
# must point at a real prepare_dataset.py output directory before submitting.
DATASET_DIR="output/20260828_125426"

mkdir -p "$LOGDIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR '$DATASET_DIR' does not exist. Set it to a" \
         "prepare_dataset.py output folder (output/<timestamp>/) before running." >&2
    exit 1
fi

if [ "$MODEL" == "evo2_7b" ]; then
    LAYERS=(22 23 24 25 26 27 28 29 30 31)
elif [ "$MODEL" == "evo2_40b" ]; then
    LAYERS=(40 41 42 43 44 45 46 47 48 49)
else
    LAYERS=(15 16 17 18 19 20 21 22 23 24)
fi

# Build layer name list: blocks.N.mlp.l3 for each N
LAYER_NAMES=()
for layer in "${LAYERS[@]}"; do
    LAYER_NAMES+=("blocks.${layer}.mlp.l3")
done

# Requesting many layers in one forward pass makes vortex's pipeline-parallel
# sharding hold several full (batch, seq_len, hidden_dim) hook outputs on the
# same one or two GPUs at once (contiguous blocks -> contiguous devices), which
# OOMs at a ~1M bp context regardless of total GPU count. Running fewer layers
# per forward pass trades wall-clock time (one extra forward pass per chunk)
# for a much smaller peak memory footprint per GPU. Lower to 1 if 2 still OOMs.
LAYERS_PER_FORWARD=2

STRANDS=(forward reverse)
EMB_TYPES=(average last)

for strand in "${STRANDS[@]}"; do
    # Locate this strand's ref/mut files by pattern, since the filename also
    # carries the window size (and jitter offsets, if used) — e.g.
    # ref_seq_DNA_forward_8192bp.npy or ref_seq_DNA_forward_8192bp_jitter...npy.
    mapfile -t ref_matches < <(find "$DATASET_DIR" -maxdepth 1 -name "ref_seq_DNA_${strand}_*.npy")
    mapfile -t mut_matches < <(find "$DATASET_DIR" -maxdepth 1 -name "mut_seq_DNA_${strand}_*.npy")

    if [ "${#ref_matches[@]}" -ne 1 ] || [ "${#mut_matches[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one ref_seq_DNA_${strand}_*.npy and one" \
             "mut_seq_DNA_${strand}_*.npy in $DATASET_DIR, found" \
             "${#ref_matches[@]} and ${#mut_matches[@]}." >&2
        exit 1
    fi
    ref_file="${ref_matches[0]}"
    mut_file="${mut_matches[0]}"

    for emb_type in "${EMB_TYPES[@]}"; do
        for ((chunk_start=0; chunk_start<${#LAYER_NAMES[@]}; chunk_start+=LAYERS_PER_FORWARD)); do
            chunk=("${LAYER_NAMES[@]:chunk_start:LAYERS_PER_FORWARD}")
            chunk_tag=$(printf '%s+' "${chunk[@]}")
            chunk_tag=${chunk_tag//./}
            chunk_tag=${chunk_tag%+}
            logfile="${LOGDIR}/${MODEL}_${strand}_${emb_type}_${chunk_tag}.log"
            echo "Running: strand=${strand} emb-type=${emb_type} layers=${chunk[*]}"

            apptainer exec --nv "$SIF" python "extract_embeddings_Evo2.py" \
                --model "$MODEL" \
                --layer "${chunk[@]}" \
                --strand "$strand" \
                --emb-type "$emb_type" \
                --ref-file "$ref_file" \
                --mut-file "$mut_file" \
                > "$logfile" 2>&1

            if [ $? -ne 0 ]; then
                echo "  FAILED (see $logfile)"
            else
                echo "  done"
            fi
        done
    done
done
