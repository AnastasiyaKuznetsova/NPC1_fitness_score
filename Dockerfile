# Use an NVIDIA CUDA base image with development tools (needed for potential compilations)
FROM nvcr.io/nvidia/pytorch:25.04-py3 as base
RUN apt-get update && apt-get install -y git python3-pip python3-tomli && rm -rf /var/lib/apt/lists/*

# Install flash-attn and evo2
RUN pip install evo2

WORKDIR /workdir