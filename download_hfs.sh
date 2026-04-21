#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export HF_HOME=/path/to/hf_home
#   bash download_hfs.sh
#
# Everything lives under:
#   $HF_HOME/gia_exp_cache

CACHE_DIR="${HF_HOME}/gia_exp_cache"
mkdir -p "${CACHE_DIR}"

# Keep all Hugging Face internal caches under the same root.
export HF_HUB_CACHE="${CACHE_DIR}/.hf_hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${CACHE_DIR}/.hf_datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/.transformers"
export HF_MODULES_CACHE="${CACHE_DIR}/.hf_modules"
export HF_EVALUATE_CACHE="${CACHE_DIR}/.hf_evaluate"
export HF_METRICS_CACHE="${CACHE_DIR}/.hf_metrics"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_MODULES_CACHE}" "${HF_EVALUATE_CACHE}" "${HF_METRICS_CACHE}"

# Defaults internal to the script.
SKIP_EXISTING="1"
TRY_ONLINE_GLUE="1"

echo "HF_HOME=${HF_HOME}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "HF_EVALUATE_CACHE=${HF_EVALUATE_CACHE}"
echo "HF_METRICS_CACHE=${HF_METRICS_CACHE}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "TRY_ONLINE_GLUE=${TRY_ONLINE_GLUE}"

echo
echo "=============================="
echo "0) Force ONLINE mode for download/materialization"
echo "=============================="

# Clear any inherited offline flags from the parent shell/session.
unset HF_HUB_OFFLINE || true
unset HF_DATASETS_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true
unset HF_EVALUATE_OFFLINE || true

# Explicitly force online behavior for this part of the script.
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_EVALUATE_OFFLINE=0
export HF_UPDATE_DOWNLOAD_COUNTS=0

echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_EVALUATE_OFFLINE=${HF_EVALUATE_OFFLINE}"

download_repo() {
  local repo_id="$1"
  local repo_type="$2"
  local local_dir="$3"

  if [[ "${SKIP_EXISTING}" == "1" && -d "${local_dir}" && -n "$(ls -A "${local_dir}" 2>/dev/null || true)" ]]; then
    echo "[skip] ${local_dir} already exists"
    return 0
  fi

  echo "[download] ${repo_type}: ${repo_id} -> ${local_dir}"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  hf download "${repo_id}" \
    --repo-type "${repo_type}" \
    --local-dir "${local_dir}" \
    --include "*"
}

echo
echo "=============================="
echo "1) Download models"
echo "=============================="

download_repo "openai-community/gpt2-large" "model" "${CACHE_DIR}/openai-community__gpt2-large"
download_repo "openai-community/gpt2" "model" "${CACHE_DIR}/openai-community__gpt2"
download_repo "meta-llama/Meta-Llama-3.1-8B" "model" "${CACHE_DIR}/meta-llama__Meta-Llama-3.1-8B"
download_repo "google/vaultgemma-1b" "model" "${CACHE_DIR}/google__vaultgemma-1b"
download_repo "google/gemma-2b" "model" "${CACHE_DIR}/google__gemma-2b"

echo
echo "=============================="
echo "2) Download dataset snapshots"
echo "=============================="

download_repo "swj0419/WikiMIA" "dataset" "${CACHE_DIR}/swj0419__WikiMIA"
download_repo "glnmario/ECHR" "dataset" "${CACHE_DIR}/glnmario__ECHR_snapshot"
download_repo "stanfordnlp/imdb" "dataset" "${CACHE_DIR}/stanfordnlp__imdb_snapshot"

set +e
download_repo "rotten_tomatoes" "dataset" "${CACHE_DIR}/rotten_tomatoes_snapshot"
RT_STATUS=$?
set -e
if [[ $RT_STATUS -ne 0 ]]; then
  echo "[warn] Could not snapshot rotten_tomatoes via hf download. Continuing."
fi

echo
echo "=============================="
echo "3) Materialize datasets into save_to_disk format"
echo "=============================="

HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TRY_ONLINE_GLUE="${TRY_ONLINE_GLUE}" python - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset, load_from_disk

cache_dir = Path(os.environ["CACHE_DIR"])
datasets_cache = Path(os.environ["HF_DATASETS_CACHE"])
try_online_glue = os.environ.get("TRY_ONLINE_GLUE", "1") == "1"

seq_keys = {
    "cola": "sentence",
    "sst2": "sentence",
    "rte": "sentence1",
    "rotten_tomatoes": "text",
    "glnmario/ECHR": "text",
    "stanfordnlp/imdb": "text",
    "swj0419/WikiMIA": None,
}

def safe_name(name: str) -> str:
    return name.replace("/", "__")

def is_saved_dataset_dir(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        load_from_disk(str(path))
        return True
    except Exception:
        return False

def load_from_snapshot(snapshot_dir: Path):
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    json_files = sorted(snapshot_dir.rglob("*.json")) + sorted(snapshot_dir.rglob("*.jsonl"))
    csv_files = sorted(snapshot_dir.rglob("*.csv"))

    if parquet_files:
        return load_dataset("parquet", data_files=[str(p) for p in parquet_files], cache_dir=str(datasets_cache))
    if json_files:
        return load_dataset("json", data_files=[str(p) for p in json_files], cache_dir=str(datasets_cache))
    if csv_files:
        return load_dataset("csv", data_files=[str(p) for p in csv_files], cache_dir=str(datasets_cache))

    raise FileNotFoundError(f"No parquet/json/jsonl/csv files found in {snapshot_dir}")

def materialize_snapshot_dataset(dataset_name: str, text_key):
    save_dir = cache_dir / safe_name(dataset_name)

    if is_saved_dataset_dir(save_dir):
        print(f"[ok] already materialized: {save_dir}")
        print(load_from_disk(str(save_dir)))
        return

    candidates = [
        cache_dir / f"{safe_name(dataset_name)}_snapshot",
        cache_dir / safe_name(dataset_name),
    ]
    snapshot_dir = next((p for p in candidates if p.exists()), None)

    if snapshot_dir is None:
        print(f"[skip] no local snapshot found for {dataset_name}")
        return

    ds = load_from_snapshot(snapshot_dir)
    print(f"[load] {dataset_name} from local snapshot: {snapshot_dir}")
    print(ds)

    split = next(iter(ds.keys()))
    cols = ds[split].column_names
    print(f"[cols] {split}: {cols}")
    if text_key is not None and text_key not in cols:
        print(f"[warn] expected text key '{text_key}' not found")

    print(f"[save] {save_dir}")
    ds.save_to_disk(str(save_dir))

    print(f"[verify] saved dataset readable: {save_dir}")
    print(load_from_disk(str(save_dir)))

def materialize_glue(task: str, text_key: str):
    save_dir = cache_dir / f"glue__{task}"

    if is_saved_dataset_dir(save_dir):
        print(f"[ok] already materialized: {save_dir}")
        print(load_from_disk(str(save_dir)))
        return

    if not try_online_glue:
        print(f"[skip] glue/{task} not materialized (TRY_ONLINE_GLUE=0)")
        return

    print(f"[online-load] glue/{task}")
    print(f"[cache] HF_DATASETS_CACHE={datasets_cache}")

    ds = load_dataset("glue", task, cache_dir=str(datasets_cache))
    print(ds)

    split = next(iter(ds.keys()))
    cols = ds[split].column_names
    print(f"[cols] {split}: {cols}")
    if text_key not in cols:
        print(f"[warn] expected text key '{text_key}' not found")

    print(f"[save] {save_dir}")
    ds.save_to_disk(str(save_dir))

    print(f"[verify] saved dataset readable: {save_dir}")
    print(load_from_disk(str(save_dir)))

for name, text_key in seq_keys.items():
    print("=" * 80)
    print(f"Processing: {name}")
    try:
        if name in {"cola", "sst2", "rte"}:
            materialize_glue(name, text_key)
        else:
            materialize_snapshot_dataset(name, text_key)
    except Exception as e:
        print(f"[fail] {name}: {e}")

print("=" * 80)
print("Materialization step complete.")
PY

echo
echo "=============================="
echo "4) Download evaluation metric cache"
echo "=============================="

HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_EVALUATE_OFFLINE=0 HF_UPDATE_DOWNLOAD_COUNTS=0 python - <<'PY'
import os
import evaluate

cache_dir = os.environ["CACHE_DIR"]
print(f"[metric] cache_dir={cache_dir}")
metric = evaluate.load("rouge", cache_dir=cache_dir)
print("[ok] loaded rouge metric:", metric)

# Run a tiny compute call so the metric module is fully materialized.
result = metric.compute(predictions=["a test sentence"], references=["a test sentence"])
print("[ok] rouge compute keys:", sorted(result.keys()))
PY

echo
echo "=============================="
echo "5) Test local model loading (OFFLINE)"
echo "=============================="

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

cache_dir = Path(os.environ["CACHE_DIR"])

models = [
    "openai-community__gpt2",
    "openai-community__gpt2-large",
    "google__gemma-2b",
    "google__vaultgemma-1b",
    "meta-llama__Meta-Llama-3.1-8B",
]

for model_dir_name in models:
    model_dir = cache_dir / model_dir_name
    print("=" * 80)
    print(f"Testing model: {model_dir}")

    if not model_dir.exists():
        print("[missing]")
        continue

    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        print("[ok] tokenizer:", tok.__class__.__name__)
    except Exception as e:
        print("[fail] tokenizer:", repr(e))
        continue

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
        print("[ok] model:", model.__class__.__name__)
        del model
    except Exception as e:
        print("[fail] model:", repr(e))
        if "vaultgemma" in repr(e).lower():
            print("[hint] upgrade transformers for vaultgemma support")
        elif "rope_scaling" in repr(e).lower():
            print("[hint] upgrade transformers for Llama 3.1 support")

    del tok
PY

echo
echo "=============================="
echo "6) Test offline loading of saved datasets"
echo "=============================="

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
import os
from pathlib import Path
from datasets import load_from_disk

cache_dir = Path(os.environ["CACHE_DIR"])

dataset_paths = {
    "cola": ("glue__cola", "sentence"),
    "sst2": ("glue__sst2", "sentence"),
    "rte": ("glue__rte", "sentence1"),
    "rotten_tomatoes": ("rotten_tomatoes", "text"),
    "glnmario/ECHR": ("glnmario__ECHR", "text"),
    "stanfordnlp/imdb": ("stanfordnlp__imdb", "text"),
    "swj0419/WikiMIA": ("swj0419__WikiMIA", None),
}

for ds_name, (dir_name, text_key) in dataset_paths.items():
    path = cache_dir / dir_name
    print("=" * 80)
    print(f"Dataset: {ds_name}")
    print(f"Path: {path}")

    if not path.exists():
        print("[missing]")
        continue

    try:
        ds = load_from_disk(str(path))
        print("[ok] loaded offline")
        print(ds)

        split = next(iter(ds.keys()))
        cols = ds[split].column_names
        print(f"[cols] {split}: {cols}")

        if text_key is not None and text_key not in cols:
            print(f"[warn] expected text key '{text_key}' not found")
    except Exception as e:
        print("[fail]", repr(e))
PY

echo
echo "=============================="
echo "7) Test offline loading of evaluation metric"
echo "=============================="

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 HF_UPDATE_DOWNLOAD_COUNTS=0 python - <<'PY'
import os
import evaluate

cache_dir = os.environ["CACHE_DIR"]
try:
    metric = evaluate.load("rouge", cache_dir=str(cache_dir))
    print("[ok] offline rouge metric load succeeded")
    result = metric.compute(predictions=["offline test"], references=["offline test"])
    print("[ok] offline rouge compute keys:", sorted(result.keys()))
except Exception as e:
    print("[fail] offline rouge metric:", repr(e))
PY

echo
echo "Done."
