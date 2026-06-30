# Shakespeare Transformer Assignment Design

## Goal

Build a Jupyter notebook submission that implements, trains, and analyzes a 6-layer decoder-only Transformer in PyTorch for Shakespeare-style text generation.

## Deliverable

The primary deliverable is `shakespeare_transformer.ipynb`. It must contain the implementation, training details, visualizations, generated samples, and written analysis required by `requirements.md`.

## Data And Tokenizer

The notebook prepares TinyShakespeare text under `data/`. It first uses an already-downloaded local file if present. The setup script attempts to download a public TinyShakespeare text file as a practical fallback when Kaggle credentials are unavailable.

The tokenizer is the Llama 2 tokenizer from ModelScope model `shakechen/Llama-2-7b-hf`. Only tokenizer files are needed. The code supports loading from `tokenizer/llama-2-7b-hf` and documents the expected download command.

## Model

The model is a 6-layer decoder-only Transformer implemented with PyTorch:

- token embedding
- RoPE positional encoding for attention queries and keys
- explicit LayerNorm
- causal masked multi-head self-attention
- feed-forward network
- language modeling head

The default training configuration is intentionally small enough for coursework iteration: 6 layers, 6 heads, 384 hidden size, 128 context length, and configurable batch size. The notebook can scale settings up or down based on available GPU memory.

## Analysis

The notebook includes layer-by-layer parameter counts and approximate training/inference FLOPs formulas for embeddings, attention projections, attention score/value operations, FFN, normalization, and LM head. It also computes experimental parameter counts from PyTorch modules and compares them against the theoretical table.

## Verification

Core model utilities live in `src/shakespeare_transformer.py` so they can be tested outside the notebook. Tests verify output shapes, causal mask behavior, RoPE shape preservation, parameter counting, and one tiny training loss calculation. The notebook imports the same module to avoid diverging implementations.
