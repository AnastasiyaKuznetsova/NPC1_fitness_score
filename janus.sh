#!/bin/bash 

tmux new-session -d -s janus "srun --reservation=balch -p gpu --pty bash \
apptainer exec --nv sif/janusdna.sif python extract_embeddings_JanusDNA.py --janusdna-repo JanusDNA \
--checkpoint Janus_ckpt_json/32_without_midattn.ckpt \
--ref-file output/20260903_120806/ref_seq_DNA_forward_1024bp.npy \
--mut-file output/20260903_120806/mut_seq_DNA_forward_1024bp.npy; exec bash"
