# From-Scratch C++ Transformer Training Project

This repository contains a from-scratch decoder-only transformer training pipeline
specialized for C/C++ source code, with a primary focus on C++. The corpus builder collects C/C++ files, the data
preparation tools filter C++-oriented instruction examples, and the tokenizer and
training scripts operate on that corpus. It is not a general-purpose language-model
project.

```text
src/cpp_transformer/models/     decoder-only transformer implementation
src/cpp_transformer/tokenization/ custom BPE tokenizer and token preparation tools
src/cpp_transformer/training/   base-training, fine-tuning, and local inference scripts
scripts/data/                   corpus collection, cleaning, and formatting scripts
configs/                        CPU/GPU training profiles and special-token IDs
data/raw/                       source datasets (local; not committed)
data/processed/                 cleaned corpora and token artifacts (local; not committed)
tests/                          fast tokenizer unit tests
third_party/                    extracted C++ repositories used to build the corpus
```

Install dependencies with `pip install -r requirements.txt` and install the package
for local development with `pip install -e .`.

Run tokenizer preparation or training as modules, for example:

```powershell
python -m cpp_transformer.tokenization.train_tokenizer
python -m cpp_transformer.training.train
```

Training settings and special-token IDs are versioned in
[`configs/training.json`](configs/training.json). Generated datasets and checkpoints
are deliberately excluded from version control.
