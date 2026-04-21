#!/bin/bash

./scripts/dager_gpt2.sh rte 2 --use_hf_split --rank_tol 1e-8
./scripts/hybrid_gpt2.sh rte 1 --use_hf_split --rank_tol 1e-5 --n_steps 10 --coeff_perplexity 0.01

./scripts/dager_gpt2-large.sh sst2 8 --use_hf_split --rank_tol 1e-5
./scripts/hybrid_gpt2-large.sh sst2 8 --use_hf_split --n_steps 50 --rank_tol 1e-5 --coeff_perplexity 0.01


./scripts/dager_gpt2.sh sst2 4 --use_hf_split --rank_tol 1e-8
./scripts/hybrid_gpt2.sh sst2 2 --use_hf_split --rank_tol 1e-5 --n_steps 10 --coeff_perplexity 0.01

./scripts/dager_gpt2.sh cola 4 --use_hf_split --rank_tol 1e-8
./scripts/hybrid_gpt2.sh cola 2 --use_hf_split --rank_tol 1e-5 --n_steps 10 --coeff_perplexity 0.01

./scripts/dager_llama_3.1.sh sst2 1 --n_inputs 100 --use_hf_split --precision full --rank_tol 1e-11 --l1_span_thresh 1e-4 \
  --l2_span_thresh 5e-10 --parallel 50 --n_incorrect 1

./scripts/hybrid_llama_3.1.sh sst2 1 --n_inputs 100 --n_steps 20 --use_hf_split --precision half --rank_tol 1e-11 \
  --l1_span_thresh 1e-4 --l2_span_thresh 5e-10 --parallel 50 --n_incorrect 1

./scripts/dager_llama_3.1.sh sst2 1 --n_inputs 5 --use_hf_split --precision half --rank_tol 1e-5 --parallel 50 --n_incorrect 1
./scripts/hybrid_llama_3.1.sh sst2 1 --n_inputs 5 --n_steps 20 --use_hf_split --precision half --rank_tol 1e-5

./scripts/dager_gemma_2b.sh sst2 4 --n_inputs 50 --use_hf_split --rank_tol 1e-6 --l1_span_thresh 1e-4 \
  --l2_span_thresh 1e-6 --max_len 64

./scripts/hybrid_gemma_2b.sh sst2 4 --n_inputs 50 --use_hf_split --rank_tol 1e-6 --l1_span_thresh 1e-4 \
  --l2_span_thresh 1e-6 --n_steps 30 --lr 0.05 --coeff_perplexity 0.01 --hybrid_project_every 10  --max_len 64

#./scripts/dager_vault_gemma.sh sst2 4 --n_inputs 50 --use_hf_split --rank_tol 1e-6 --l1_span_thresh 1e-4 \
#  --l2_span_thresh 1e-6 --max_len 64
#./scripts/hybrid_vault_gemma.sh sst2 4 --n_inputs 50 --use_hf_split --rank_tol 1e-6 --l1_span_thresh 1e-4 \
#  --l2_span_thresh 1e-6 --n_steps 30 --lr 0.05 --coeff_perplexity 0.01 --hybrid_project_every 10 --max_len 64
