#!/bin/bash

# Runs extract_embeddings.py across middle to late layers, both strands, and both emb-types.

set -uoe pipefail

MODEL="evo2_40b"
SIF="evo2.sif"
LOGDIR="logs_extract_emb_40B"

mkdir -p "$LOGDIR"

# Layer set depends on the model: evo2_7b has 32 blocks, so "last 10 layers"
# means blocks 22-31. evo2_1b_base keeps its original middle-to-late range.
if [ "$MODEL" == "evo2_7b" ]; then
        LAYERS=(22 23 24 25 26 27 28 29 30 31)
elif [ "$MODEL" == "evo2_40b" ]; then
	LAYERS=(40 41 42 43 44 45 46 47 48 49)
else
        LAYERS=(10 12 13 14 15 16 17 18 19 20 21 22 23 24)
fi

STRANDS=(forward reverse)
EMB_TYPES=(average last)

for layer in "${LAYERS[@]}"; do
        for strand in "${STRANDS[@]}"; do
                for emb_type in "${EMB_TYPES[@]}"; do
                        layer_name="blocks.${layer}.mlp.l3"
                        logfile="${LOGDIR}/${MODEL}_layer${layer}_${strand}_${emb_type}.log"

                        echo "Running: layer=${layer_name} strand=${strand} emb-type=${emb_type}"

                        apptainer exec --nv "$SIF" python "extract_embeddings.py" \
                                --model "$MODEL" \
                                --layer "$layer_name" \
                                --seq-type DNA \
                                --strand "$strand" \
                                --emb-type "$emb_type" \
                                > "$logfile" 2>&1

                        if [ $? -ne 0 ]; then
                                echo " FAILED (see $logfile)"
                        else
                                echo " done"
                        fi
                done
        done
done
