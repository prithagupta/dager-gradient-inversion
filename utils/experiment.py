import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import signal
import torch
import atexit

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger(__name__)
_LOG_LOCK_HANDLES = []
_LOCK_PATHS_BY_HANDLE = {}
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
    return os.path.join(_repo_root(), "results", attack_name, f"results_{job_hash}")


def is_attack_complete(attack_name, job_hash, required_files=None):
    if required_files is None:
        required_files = ("sentence_results.csv", "input_results.csv", "run_summary.json")

    results_dir = get_results_dir(attack_name, job_hash)
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
    payload["status"] = status
    with open(os.path.join(results_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def setup_random_seed(seed=1234, logger=None):
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
                modules_cache = os.path.join(hf_home, "gia_exp_cache", ".hf_modules")
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
    log_dir = os.path.join(_repo_root(), "logs")
    create_directory_safely(log_dir)

    hash_value = get_hash_value_for_args(args)
    primary_log_path = os.path.join(log_dir, f"{attack_name}_{hash_value}.log")
    _, log_path, lock_acquired = _acquire_log_lock(primary_log_path)
    append_to_completed_log = lock_acquired and _log_file_completed(log_path)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S", )
    print(f"Does the log file exits and the experiments are done {append_to_completed_log}")

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
