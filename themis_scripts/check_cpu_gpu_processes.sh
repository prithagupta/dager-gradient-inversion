#!/bin/bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  mapfile -t PIDS < <(pgrep -u "$USER" -f 'python .*attack(_hybrid)?\.py|python .*attack_autoregressive_dager\.py' || true)
else
  PIDS=( "$@" )
fi

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No running attack.py / attack_hybrid.py / attack_autoregressive_dager.py processes found for $USER."
  exit 0
fi

echo "===== Process summary ====="
printf "%-9s %-6s %-6s %-6s %-6s %-8s %s\n" "PID" "CPU%" "MEM%" "NLWP" "PSR" "ELAPSED" "COMMAND"
for pid in "${PIDS[@]}"; do
  if [ ! -d "/proc/$pid" ]; then
    continue
  fi
  ps -p "$pid" -o pid=,pcpu=,pmem=,nlwp=,psr=,etime=,args= | awk '{$1=$1; print}'
done

echo ""
echo "===== NVIDIA process view ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
else
  echo "nvidia-smi not found"
fi

echo ""
echo "===== Per-process environment and hottest threads ====="
for pid in "${PIDS[@]}"; do
  if [ ! -d "/proc/$pid" ]; then
    continue
  fi
  echo ""
  echo "--- PID $pid ---"
  echo "Command:"
  tr '\0' ' ' <"/proc/$pid/cmdline"
  echo ""
  echo "Thread count:"
  grep -E '^Threads:' "/proc/$pid/status" || true
  echo "Selected environment:"
  tr '\0' '\n' <"/proc/$pid/environ" | grep -E '^(CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|TOKENIZERS_PARALLELISM|PYTORCH_CUDA_ALLOC_CONF|DAGER_FORCE_CPU_DECOMP)=' || true
  echo "Top process threads:"
  ps -L -p "$pid" -o pid,tid,psr,pcpu,comm --sort=-pcpu | head -12 || true
done
