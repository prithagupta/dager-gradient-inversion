import atexit
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import signal
import torch
import time

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger(__name__)
_LOG_LOCK_HANDLES = []
_LOCK_PATHS_BY_HANDLE = {}
_ACTIVE_LOG_PATHS = set()
_PREVIOUS_SIGNAL_HANDLERS = {}
_SIGNAL_CLEANUP_INSTALLED = False


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_for_hash(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(value[k]) for k in sorted(value)}
    return str(value)


def args_to_dict(args):
    dict_args = {key: _normalize_for_hash(value) for key, value in sorted(vars(args).items())}
    if dict_args.get("preprocess_numbered_markers") is False:
        dict_args.pop("preprocess_numbered_markers", None)
    if dict_args.get("preprocess_boundary_markers") is False:
        dict_args.pop("preprocess_boundary_markers", None)
    if dict_args.get("preprocess_unique_canary_markers") is False:
        dict_args.pop("canary_marker_prefix", None)
    if dict_args.get("preprocess_unique_canary_markers") is False:
        dict_args.pop("preprocess_unique_canary_markers", None)
    return dict_args


def get_hash_value_for_args(args):
    dict_args = args_to_dict(args)
    if "device" in dict_args.keys():
        del dict_args["device"]
    if "device_grad" in dict_args.keys():
        del dict_args["device_grad"]
    if "cache_dir" in dict_args.keys():
        del dict_args["cache_dir"]
    if "neptune" in dict_args.keys():
        del dict_args["neptune"]
    if "neptune_offline" in dict_args.keys():
        del dict_args["neptune_offline"]
    hash_string = json.dumps(dict_args, sort_keys=True, default=str)
    return hashlib.sha1(hash_string.encode()).hexdigest()[:12]


def create_directory_safely(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if path:
        os.makedirs(path, exist_ok=True)


def _log_file_completed(log_path):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return False

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            nonempty_lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return False

    if len(nonempty_lines) < 2:
        return False

    return (nonempty_lines[-1].endswith("Done with all."))


def compact_attack_log(log_path):
    """Trim noisy attack logs while preserving the first completed run record."""
    if not log_path or not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return False

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return False

    compacted = []
    saw_done = False
    changed = False
    for line in lines:
        if saw_done:
            changed = True
            continue
        if " INFO     L1 Position " in line:
            changed = True
            continue
        compacted.append(line)
        if " INFO     Done with all." in line:
            saw_done = True

    if not changed:
        return False

    tmp_path = f"{log_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(compacted)
        os.replace(tmp_path, log_path)
        return True
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def finalize_completed_attack_log(log_path, reason="complete"):
    """Flush handlers and compact a completed attack log immediately."""
    if not log_path:
        return False
    try:
        logging.shutdown()
    except Exception:
        pass
    changed = compact_attack_log(log_path)
    status = "compacted" if changed else "already compact"
    print(f"Log cleanup after confirmed {reason}: {status} | {log_path}")
    return changed


def _acquire_log_lock(log_path):
    if fcntl is None:
        return None, log_path, True

    lock_path = f"{log_path}.lock"
    create_directory_safely(lock_path, is_file_path=True)
    lock_handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _LOG_LOCK_HANDLES.append(lock_handle)
        _LOCK_PATHS_BY_HANDLE[id(lock_handle)] = lock_path
        return lock_handle, log_path, True
    except BlockingIOError:
        lock_handle.close()
        return None, log_path, False


def release_log_lock(lock_handle):
    if lock_handle is None:
        return
    lock_path = _LOCK_PATHS_BY_HANDLE.pop(id(lock_handle), None)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass
        if lock_handle in _LOG_LOCK_HANDLES:
            _LOG_LOCK_HANDLES.remove(lock_handle)
        if lock_path and os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except OSError:
                pass


def release_all_log_locks():
    try:
        logging.shutdown()
    except Exception:
        pass
    for log_path in list(_ACTIVE_LOG_PATHS):
        compact_attack_log(log_path)
    for handle in list(_LOG_LOCK_HANDLES):
        release_log_lock(handle)


atexit.register(release_all_log_locks)


def _cleanup_log_locks_on_signal(signum, frame):
    release_all_log_locks()
    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
    if callable(previous):
        signal.signal(signum, previous)
        return previous(signum, frame)
    if previous == signal.SIG_IGN:
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def install_log_lock_cleanup_handlers():
    global _SIGNAL_CLEANUP_INSTALLED
    if _SIGNAL_CLEANUP_INSTALLED:
        return

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            _PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, _cleanup_log_locks_on_signal)
        except Exception:
            continue

    _SIGNAL_CLEANUP_INSTALLED = True


install_log_lock_cleanup_handlers()


def get_results_dir(attack_name, job_hash):
    results_root = os.environ.get("DAGER_RESULTS_ROOT")
    if not results_root:
        results_root = os.path.join(_repo_root(), "results")
    return os.path.join(results_root, attack_name, f"results_{job_hash}")


def get_results_root():
    results_root = os.environ.get("DAGER_RESULTS_ROOT")
    if not results_root:
        results_root = os.path.join(_repo_root(), "results")
    return results_root


def get_completion_index_path(results_root=None):
    if results_root is None:
        results_root = get_results_root()
    return os.path.join(results_root, "completed_hashes.h5")


def _completion_index_enabled():
    return str(os.environ.get("DAGER_COMPLETION_INDEX", "1")).lower() not in {"0", "false", "no"}


def _completion_key(attack_name, job_hash):
    return f"{attack_name}/{job_hash}"


def _load_completion_index(index_path):
    return set(_load_completion_records(index_path).keys())


def _decode_h5_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _record_from_completion_key(key):
    if "/" in key:
        attack_name, hash_value = key.split("/", 1)
    else:
        attack_name, hash_value = "", key
    return {
        "key": key,
        "attack_name": attack_name,
        "hash_value": hash_value,
        "result_dir": "",
    }


def _load_completion_records(index_path):
    try:
        import h5py
    except Exception:
        return {}

    if not os.path.isfile(index_path) or os.path.getsize(index_path) == 0:
        return {}

    try:
        with h5py.File(index_path, "r") as h5:
            if "completed" not in h5:
                return {}
            group = h5["completed"]
            records = {}
            if "records" in group:
                for raw_record in group["records"][()]:
                    try:
                        record = json.loads(_decode_h5_string(raw_record))
                    except Exception:
                        continue
                    key = record.get("key")
                    if key:
                        records[str(key)] = {
                            "key": str(key),
                            "attack_name": str(record.get("attack_name", "")),
                            "hash_value": str(record.get("hash_value", "")),
                            "result_dir": str(record.get("result_dir", "")),
                        }
            if records:
                return records
            if "keys" in group:
                for raw_key in group["keys"][()]:
                    key = _decode_h5_string(raw_key)
                    records[key] = _record_from_completion_key(key)
                return records
            if "hash_values" in group:
                for raw_hash in group["hash_values"][()]:
                    hash_value = _decode_h5_string(raw_hash)
                    records[hash_value] = {
                        "key": hash_value,
                        "attack_name": "",
                        "hash_value": hash_value,
                        "result_dir": "",
                    }
            return records
    except Exception:
        return {}


def _write_completion_index_records(index_path, records):
    import h5py

    create_directory_safely(index_path, is_file_path=True)
    tmp_path = f"{index_path}.tmp.{os.getpid()}"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    ordered_records = [records[key] for key in sorted(set(records))]
    keys = [record.get("key", "") for record in ordered_records]
    attack_names = [record.get("attack_name", "") for record in ordered_records]
    hash_values = [record.get("hash_value", "") for record in ordered_records]
    result_dirs = [record.get("result_dir", "") for record in ordered_records]
    payload = [json.dumps(record, sort_keys=True) for record in ordered_records]
    with h5py.File(tmp_path, "w") as h5:
        group = h5.create_group("completed")
        group.create_dataset("keys", data=np.array(keys, dtype=object), dtype=string_dtype)
        group.create_dataset("attack_names", data=np.array(attack_names, dtype=object), dtype=string_dtype)
        group.create_dataset("hash_values", data=np.array(hash_values, dtype=object), dtype=string_dtype)
        group.create_dataset("result_dirs", data=np.array(result_dirs, dtype=object), dtype=string_dtype)
        group.create_dataset("records", data=np.array(payload, dtype=object), dtype=string_dtype)
        group.attrs["updated_at_unix"] = float(time.time())
        group.attrs["count"] = int(len(ordered_records))
    os.replace(tmp_path, index_path)


def _write_completion_index(index_path, keys):
    records = {str(key): _record_from_completion_key(str(key)) for key in sorted(set(keys))}
    _write_completion_index_records(index_path, records)


def _completion_index_lock(index_path):
    lock_path = f"{index_path}.lock"
    create_directory_safely(lock_path, is_file_path=True)
    lock_handle = open(lock_path, "a+", encoding="utf-8")
    if fcntl is not None:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle, lock_path


def _release_completion_index_lock(lock_handle, lock_path):
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass
        try:
            if lock_path and os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError:
            pass


def mark_attack_complete_in_index(attack_name, job_hash, results_dir=None):
    if not _completion_index_enabled():
        return False
    index_path = get_completion_index_path()
    key = _completion_key(attack_name, job_hash)
    lock_handle = None
    lock_path = None
    try:
        lock_handle, lock_path = _completion_index_lock(index_path)
        records = _load_completion_records(index_path)
        already_present = key in records
        records[key] = {
            "key": key,
            "attack_name": str(attack_name),
            "hash_value": str(job_hash),
            "result_dir": os.path.abspath(results_dir) if results_dir else "",
        }
        _write_completion_index_records(index_path, records)
        if already_present:
            logger.info("Refreshed completed hash in %s: %s", index_path, key)
            return False
        logger.info("Added completed hash in %s: %s", index_path, key)
        return True
    except Exception as exc:
        logger.warning("Could not update completion index %s for %s: %s", index_path, key, exc)
        return False
    finally:
        if lock_handle is not None:
            _release_completion_index_lock(lock_handle, lock_path)


def is_attack_complete_in_index(attack_name, job_hash):
    results_dir = get_results_dir(attack_name, job_hash)
    if not _completion_index_enabled():
        return False, results_dir
    index_path = get_completion_index_path()
    completion_key = _completion_key(attack_name, job_hash)
    completed_keys = _load_completion_index(index_path)
    return completion_key in completed_keys, results_dir


def get_private_experiment_family(args):
    if getattr(args, "algo", None) == "fedavg":
        return "fedavg"

    finetuned_path = getattr(args, "finetuned_path", None)
    if not finetuned_path:
        return None

    normalized_path = os.path.normpath(str(finetuned_path)).lower()
    path_parts = set(normalized_path.split(os.sep))
    if "dp" in path_parts or "noise_" in normalized_path:
        return "dp"
    return "fine_tune"


def configure_private_experiment_roots(args):
    family = get_private_experiment_family(args)
    if family is None:
        return None

    os.environ.setdefault("DAGER_RESULTS_ROOT", os.path.join(_repo_root(), "results", family))
    os.environ.setdefault("DAGER_LOGS_ROOT", os.path.join(_repo_root(), "logs", family))
    return family


def is_attack_complete(attack_name, job_hash, required_files=None):
    if required_files is None:
        required_files = ("sentence_results.csv", "input_results.csv", "run_summary.json")

    results_dir = get_results_dir(attack_name, job_hash)
    index_path = get_completion_index_path()
    if _completion_index_enabled():
        completion_key = _completion_key(attack_name, job_hash)
        completed_keys = _load_completion_index(index_path)
        if completion_key in completed_keys:
            logger.info("Completion index hit at %s for %s", index_path, completion_key)
            return True, results_dir

    if not os.path.isdir(results_dir):
        return False, results_dir

    for filename in required_files:
        file_path = os.path.join(results_dir, filename)
        if not os.path.isfile(file_path):
            return False, results_dir
        if os.path.getsize(file_path) == 0:
            return False, results_dir
        logger.info(f"Results already exist at {file_path}")

    summary_path = os.path.join(results_dir, "run_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if summary.get("status") == "incomplete":
                return False, results_dir
        except Exception:
            return False, results_dir

    mark_attack_complete_in_index(attack_name, job_hash, results_dir=results_dir)
    return True, results_dir


SENTENCE_META_COLUMNS = {
    "attack", "model", "dataset", "input_index", "sentence_index", "reference", "prediction"
}

INPUT_META_COLUMNS = {
    "attack", "model", "dataset", "input_index", "num_sentences",
    "joined_reference", "joined_prediction", "reconstruction_time_sec"
}


def _coerce_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _records_from_dataframe(df, exclude_columns):
    records = []
    for record in df.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if key in exclude_columns:
                continue
            clean[key] = _coerce_scalar(value)
        records.append(clean)
    return records


def load_partial_attack_state(results_dir):
    state = {
        "sentence_rows": [],
        "input_rows": [],
        "predictions": [],
        "references": [],
        "final_results": [],
        "final_per_input_results": [],
        "input_times": [],
        "completed_inputs": set(),
    }

    sentence_path = os.path.join(results_dir, "sentence_results.csv")
    input_path = os.path.join(results_dir, "input_results.csv")

    if os.path.isfile(sentence_path) and os.path.getsize(sentence_path) > 0:
        sentence_df = pd.read_csv(sentence_path)
        if "input_index" in sentence_df.columns:
            sentence_df = sentence_df.sort_values(["input_index", "sentence_index"], kind="stable")
        state["sentence_rows"] = sentence_df.to_dict(orient="records")
        state["predictions"] = [str(v) for v in sentence_df.get("prediction", pd.Series(dtype=str)).tolist()]
        state["references"] = [str(v) for v in sentence_df.get("reference", pd.Series(dtype=str)).tolist()]
        state["final_results"] = _records_from_dataframe(sentence_df, SENTENCE_META_COLUMNS)

    if os.path.isfile(input_path) and os.path.getsize(input_path) > 0:
        input_df = pd.read_csv(input_path)
        if "input_index" in input_df.columns:
            input_df = input_df.sort_values(["input_index"], kind="stable")
            state["completed_inputs"] = set(int(v) for v in input_df["input_index"].tolist())
        state["input_rows"] = input_df.to_dict(orient="records")
        state["final_per_input_results"] = _records_from_dataframe(input_df, INPUT_META_COLUMNS)
        if "reconstruction_time_sec" in input_df.columns:
            state["input_times"] = [float(v) for v in input_df["reconstruction_time_sec"].tolist() if not pd.isna(v)]

    return state


def write_attack_artifacts(results_dir, sentence_rows, input_rows, summary_results=None, status="complete"):
    os.makedirs(results_dir, exist_ok=True)
    sentence_df = pd.DataFrame(sentence_rows)
    input_df = pd.DataFrame(input_rows)
    for df in (sentence_df, input_df):
        if "run_id" in df.columns:
            df.drop(columns=["run_id"], inplace=True)
    sentence_df.to_csv(os.path.join(results_dir, "sentence_results.csv"), index=False)
    input_df.to_csv(os.path.join(results_dir, "input_results.csv"), index=False)
    if summary_results is None:
        summary_results = {}
    payload = dict(summary_results)
    if not input_df.empty and "reconstruction_time_sec" in input_df.columns:
        timing_df = input_df[["input_index", "reconstruction_time_sec"]].copy()
        timing_df = timing_df.dropna(subset=["reconstruction_time_sec"])
        if not timing_df.empty:
            timing_df["input_index"] = timing_df["input_index"].astype(int)
            timing_df["reconstruction_time_sec"] = timing_df["reconstruction_time_sec"].astype(float)
            payload["Input Timings"] = {
                "completed_inputs": int(len(timing_df)),
                "input_indices": [int(v) for v in timing_df["input_index"].tolist()],
                "reconstruction_time_sec": [float(v) for v in timing_df["reconstruction_time_sec"].tolist()],
                "reconstruction_time_mean": float(timing_df["reconstruction_time_sec"].mean()),
                "reconstruction_time_std": float(timing_df["reconstruction_time_sec"].std(ddof=0)),
            }
    payload["status"] = status
    with open(os.path.join(results_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def setup_random_seed(seed=1234, logger=None):
    torch_threads = os.environ.get("DAGER_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if torch_threads:
        try:
            torch.set_num_threads(max(int(torch_threads), 1))
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not set torch num threads from %s: %s", torch_threads, exc)
    torch_interop_threads = os.environ.get("DAGER_TORCH_INTEROP_THREADS")
    if torch_interop_threads:
        try:
            torch.set_num_interop_threads(max(int(torch_interop_threads), 1))
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not set torch interop threads from %s: %s", torch_interop_threads, exc)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if logger is not None:
        logger.info("Random seed set to %s", seed)
        logger.info("Torch threads: %s", torch.get_num_threads())


def configure_torch_numeric(args, logger=None):
    if getattr(args, "aug_disable_tf32", False):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if logger is not None:
            logger.info("Torch TF32 disabled for augmented numeric reproducibility.")

    if getattr(args, "aug_deterministic_linalg", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
        if logger is not None:
            logger.info("Torch deterministic algorithms enabled with warn_only=True.")


def cleanup_memory():
    """Release Python and accelerator-side cached memory between attack inputs."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def load_rouge_metric(cache_dir=None, logger=None):
    import evaluate
    from datasets import DownloadConfig
    from pathlib import Path

    if cache_dir is None:
        cache_dir = os.path.join(_repo_root(), "models_cache")
    cache_dir = os.path.abspath(str(cache_dir))

    modules_cache = os.environ.get("HF_MODULES_CACHE")
    if modules_cache is None:
        cache_modules_dir = os.path.join(cache_dir, ".hf_modules")
        if os.path.isdir(cache_modules_dir):
            modules_cache = cache_modules_dir
        else:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                modules_cache = os.path.join(hf_home, "gia_cache", ".hf_modules")
            else:
                modules_cache = os.path.expanduser("~/.cache/huggingface/modules")

    offline_requested = any(
        str(os.environ.get(flag, "")).lower() in {"1", "true", "yes"}
        for flag in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_EVALUATE_OFFLINE"]
    )

    rouge_script_candidates = []
    modules_path = Path(modules_cache)
    if modules_path.exists():
        rouge_script_candidates.extend(
            sorted(modules_path.glob("evaluate_modules/metrics/evaluate-metric--rouge/*/rouge.py"))
        )
        rouge_script_candidates.extend(
            sorted(modules_path.glob("datasets_modules/metrics/rouge/*/rouge.py"))
        )

    local_download_config = DownloadConfig(local_files_only=True)

    previous_eval_offline = os.environ.get("HF_EVALUATE_OFFLINE")
    previous_update_counts = os.environ.get("HF_UPDATE_DOWNLOAD_COUNTS")
    os.environ["HF_EVALUATE_OFFLINE"] = "1"
    os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = "0"
    try:
        for rouge_script in rouge_script_candidates:
            try:
                metric = evaluate.load(
                    str(rouge_script),
                    cache_dir=cache_dir,
                    download_config=local_download_config,
                )
                if logger is not None:
                    logger.info("Loaded ROUGE metric from local script: %s", rouge_script)
                return metric
            except Exception as local_script_exc:
                if logger is not None:
                    logger.warning("Failed loading local ROUGE script %s: %r", rouge_script, local_script_exc)

        try:
            metric = evaluate.load("rouge", cache_dir=cache_dir, download_config=local_download_config)
            if logger is not None:
                logger.info("Loaded ROUGE metric from local cache: %s", cache_dir)
            return metric
        except Exception as local_exc:
            if logger is not None:
                logger.warning("Local ROUGE metric cache not available at %s: %r", cache_dir, local_exc)
            if offline_requested:
                raise RuntimeError(
                    f"ROUGE metric is not available in local cache under {cache_dir} / {modules_cache}, "
                    f"and offline mode is enabled. Run download_hfs.sh in an online environment first."
                ) from local_exc
    finally:
        if previous_eval_offline is None:
            os.environ.pop("HF_EVALUATE_OFFLINE", None)
        else:
            os.environ["HF_EVALUATE_OFFLINE"] = previous_eval_offline
        if previous_update_counts is None:
            os.environ.pop("HF_UPDATE_DOWNLOAD_COUNTS", None)
        else:
            os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = previous_update_counts

    if logger is not None:
        logger.info("Falling back to online ROUGE metric download.")
    return evaluate.load("rouge", cache_dir=cache_dir)


def setup_experiment_logging(args, attack_name, level=logging.INFO):
    log_dir = os.environ.get("DAGER_LOGS_ROOT")
    if not log_dir:
        log_dir = os.path.join(_repo_root(), "logs")
    create_directory_safely(log_dir)

    hash_value = get_hash_value_for_args(args)
    primary_log_path = os.path.join(log_dir, f"{attack_name}_{hash_value}.log")
    _, log_path, lock_acquired = _acquire_log_lock(primary_log_path)
    if lock_acquired:
        if compact_attack_log(log_path):
            print(f"Compacted existing log before status check: {log_path}")
        _ACTIVE_LOG_PATHS.add(log_path)
    append_to_completed_log = lock_acquired and _log_file_completed(log_path)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S", )
    print(f"Does the log file exists and the experiment is done {append_to_completed_log}")
    if append_to_completed_log:
        print(
            f"Existing completed log reopened for hash {hash_value}; checking result folder before deciding whether to rerun.")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(level)

    logger = logging.getLogger(attack_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = True
    if lock_acquired:
        file_handler = logging.FileHandler(log_path, mode="a" if append_to_completed_log else "w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root_logger.addHandler(stream_handler)

    for noisy_logger_name in ["matplotlib", "absl", "absl.logging", "urllib3.connectionpool"]:
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.propagate = False

    if not lock_acquired:
        logger.warning(
            "Primary log file %s is locked by another process. Skipping this experiment run and assuming the other run will finish it.",
            primary_log_path,
        )
    elif append_to_completed_log:
        logger.info("=" * 80)
        logger.info(f"Appending to existing log for repeated run of the same argument hash {hash_value}")
    logger.info(f"Log file path: {log_path}")
    logger.info("Argument hash: %s", hash_value)
    logger.info("Arguments:\n%s", json.dumps(args_to_dict(args), indent=2, sort_keys=True, default=str))
    setup_random_seed(getattr(args, "rng_seed", 1234), logger=logger)
    configure_torch_numeric(args, logger=logger)
    return logger, log_path, hash_value, lock_acquired
