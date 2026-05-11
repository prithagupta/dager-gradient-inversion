# Synthetic Autoregressive Prototype

This folder is a **standalone, synthetic-only scaffold** for testing the idea:

`evidence -> token graph -> slot inference -> constrained decoding -> reranking`

It is intentionally **not wired into** the current attack pipeline and does not
consume real gradients or model internals. The goal is to let you prototype the
research structure without disturbing the current implementation.

## Files

- `config.py`
- `types.py`
- `evidence_extractor.py`
- `token_graph.py`
- `slot_inference.py`
- `constrained_decoder.py`
- `batch_reranker.py`
- `demo.py`

## What it does

- builds synthetic token evidence from toy sequences
- creates a compatibility graph
- assigns tokens to latent slots
- decodes one sequence per slot under simple constraints
- reranks candidate batches against synthetic evidence consistency

## What it does not do

- no real gradient loading
- no integration with `attack.py`
- no integration with `attack_hybrid.py`
- no real reconstruction from private data

## Quick start

```bash
cd /Users/prithagupta/projects/dager-gradient-inversion
python synthetic_autoregressive_proto/demo.py
```

## Dataset-backed benchmark

This version reuses the repo's current:

- `TextDataset`
- ROUGE loading
- per-sentence and per-input metric evaluation

while still staying synthetic-only.

Example:

```bash
cd /Users/prithagupta/projects/dager-gradient-inversion
python synthetic_autoregressive_proto/benchmark.py \
  --dataset sst2 \
  --split val \
  --batch_size 64 \
  --n_inputs 13 \
  --model_path gpt2 \
  --cache_dir /media/data/shared/hf_cache/gia_cache \
  --use_hf_split \
  --n_slots 64
```

Equivalent top-level attack entrypoint:

```bash
python attack_synthetic_autoregressive_proto.py \
  --dataset sst2 \
  --split val \
  --batch_size 64 \
  --n_inputs 13 \
  --model_path gpt2 \
  --use_hf_split
```

By default, outputs are written under the normal hashed result layout:

```text
results/autoregressivedager/results_<job_hash>/
```

with:

- `sentence_results.csv`
- `input_results.csv`
- `run_summary.json`

The prototype keeps its own arguments here, including autoregressive slot beam
size and joint batch reranking. It does not add defaults to `args_factory.py`,
so the existing DAGER/hybrid hash path is left alone.

The terminal follows the main attack scripts and prints `Hash Value <hash>
Started/Done/Skipped`. The shared log helper writes the run log under `logs/`,
while artifacts are written under `results/autoregressivedager/results_<job_hash>/`.

## Parameter sets to try

### Small and stable

```python
PrototypeConfig(
    n_slots=4,
    top_k_per_position=4,
    graph_edge_threshold=0.20,
    assignment_temperature=0.8,
    decoder_beam_size=3,
    rerank_weight_support=1.0,
    rerank_weight_coherence=0.5,
)
```

### More permissive

```python
PrototypeConfig(
    n_slots=4,
    top_k_per_position=6,
    graph_edge_threshold=0.10,
    assignment_temperature=1.0,
    decoder_beam_size=5,
    rerank_weight_support=1.0,
    rerank_weight_coherence=0.8,
)
```

### Stress test

```python
PrototypeConfig(
    n_slots=8,
    top_k_per_position=8,
    graph_edge_threshold=0.05,
    assignment_temperature=1.2,
    decoder_beam_size=6,
    rerank_weight_support=1.0,
    rerank_weight_coherence=1.0,
)
```

Delete the whole folder if you do not want to keep it.
