#!/bin/bash

# Runs extract_embeddings.py across middle to late layers, both strands, and both emb-types.
# All layers are extracted in a single forward pass per strand+emb-type combination.

set -uoe pipefail

MODEL="evo2_7b"
SIF="sif/evo2.sif"
LOGDIR="logs_extract_emb_7B_1M_cw"

mkdir -p "$LOGDIR"

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

STRANDS=(forward reverse)
EMB_TYPES=(average last)

for strand in "${STRANDS[@]}"; do
    for emb_type in "${EMB_TYPES[@]}"; do
        logfile="${LOGDIR}/${MODEL}_${strand}_${emb_type}.log"
        echo "Running: strand=${strand} emb-type=${emb_type} layers=${LAYER_NAMES[*]}"

        apptainer exec --nv "$SIF" python "extract_embeddings.py" \
            --model "$MODEL" \
            --layer "${LAYER_NAMES[@]}" \
            --strand "$strand" \
            --emb-type "$emb_type" \
            > "$logfile" 2>&1

        if [ $? -ne 0 ]; then
            echo "  FAILED (see $logfile)"
        else
            echo "  done"
        fi
    done
done
