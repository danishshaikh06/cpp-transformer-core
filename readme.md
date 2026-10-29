# LatentForge

**A PyTorch workbench for building and exploring transformer architectures from first principles.**

LatentForge brings together a decoder-only transformer, configurable attention mechanisms, a custom BPE tokenizer, and training utilities. The goal is to understand how language models work by implementing their components, examining architectural choices, and evaluating their behaviour.

Originally developed for C/C++ code generation, the project is expanding into a broader environment for transformer experimentation. The training corpus and evaluation plan are still being selected.

## Features

### Transformer Architecture

- Decoder-only transformer implemented in PyTorch.
- Pre-norm residual blocks and causal self-attention.
- Tied token-embedding and output-projection weights.
- Configurable model dimensions, layer counts, and attention heads.

### Attention and KV Caching

- Multi-Head Attention (**MHA**).
- Grouped-Query Attention (**GQA**).
- Multi-Query Attention (**MQA**) through configurable KV heads.
- Explicit eager attention and PyTorch **SDPA** backends.
- Request-owned KV caching with separate cache entries for each decoder layer.

### Custom Tokenization

- Byte-level **Byte Pair Encoding (BPE)**.
- Vocabulary construction through learned merge rules.
- UTF-8 encoding and decoding.
- Tokenizer serialization and loading.
- Corpus cleaning and token preparation utilities.

### Training Utilities

- AdamW optimization.
- Mixed-precision training on CUDA.
- Gradient accumulation and clipping.
- Model checkpointing.
- Validation loss and perplexity tracking.

## Getting Started

### Requirements

- Python 3.10 or newer.
- Dependencies declared in `pyproject.toml`.
- A CUDA-compatible GPU for GPU execution; the model example below also supports CPU execution.

### Installation

```bash
git clone https://github.com/danishshaikh06/LatentForge.git
cd LatentForge

python -m pip install -e .
```

Using a virtual environment is recommended.

The project is named **LatentForge**, while the internal Python package currently remains `cpp_transformer`.

## Build a Transformer

The following example creates a small model with randomly initialized weights. It requires no dataset or checkpoint.

```python
import torch

from cpp_transformer.models.decoder import build_transformer

model = build_transformer(
    tgt_vocab_size=3000,
    tgt_seq_len=128,
    d_model=128,
    N=2,
    h=4,
    kv_heads=2,
    d_ff=512,
    dropout=0.0,
    attention_backend="sdpa",
    position_encoding="sinusoidal",
).eval()

tokens = torch.randint(0, 3000, (2, 32))

with torch.no_grad():
    logits = model(tokens)

print(logits.shape)
# torch.Size([2, 32, 3000])
```

### Attention Configuration

The relationship between query heads and KV heads determines the attention variant:

| Configuration | Attention variant |
|---|---|
| `kv_heads == h` | Multi-Head Attention |
| `1 < kv_heads < h` | Grouped-Query Attention |
| `kv_heads == 1` | Multi-Query Attention |

The number of query heads must be divisible by the number of KV heads.

Set `attention_backend="eager"` to use the explicit attention implementation instead of SDPA.

## Project Structure

```text
LatentForge/
├── src/cpp_transformer/
│   ├── configs/          # Attention configuration and training profiles
│   ├── core/             # Shared paths and artifact locations
│   ├── models/
│   │   ├── attention/    # Attention and positional-embedding components
│   │   └── decoder.py    # Decoder-only transformer
│   ├── tokenization/     # BPE tokenizer and token preparation
│   └── training/         # Training, fine-tuning, and inference scripts
├── scripts/data/         # Corpus preparation utilities
├── tests/                # Tokenizer and attention test code
├── data/                 # Local datasets and token artifacts
└── pyproject.toml
```

## Development Roadmap

- Complete rotary positional embedding integration.
- Implement **Multi-head Latent Attention (MLA)** with **decoupled RoPE**.
- Align configuration paths and checkpoint loading across training and inference.
- Select and document the training corpus.
- Expand correctness checks and benchmark attention variants.
- Publish reproducible training and evaluation results.

