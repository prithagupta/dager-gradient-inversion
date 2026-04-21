import hashlib
import json
import logging
import os
import random

import numpy as np
import torch


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
    return {
        key: _normalize_for_hash(value)
        for key, value in sorted(vars(args).items())
    }


def get_hash_value_for_args(args):
    hash_string = json.dumps(args_to_dict(args), sort_keys=True, default=str)
    return hashlib.sha1(hash_string.encode()).hexdigest()[:12]


def create_directory_safely(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if path:
        os.makedirs(path, exist_ok=True)


def setup_random_seed(seed=1234, logger=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if logger is not None:
        logger.info("Random seed set to %s", seed)
        logger.info("Torch threads: %s", torch.get_num_threads())


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
    log_path = os.path.join(log_dir, f"{attack_name}_{hash_value}.log")

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(attack_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = True

    for noisy_logger_name in ["matplotlib", "absl", "absl.logging", "urllib3.connectionpool"]:
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.propagate = False

    logger.info("Log file path: %s", log_path)
    logger.info("Argument hash: %s", hash_value)
    logger.info("Arguments:\n%s", json.dumps(args_to_dict(args), indent=2, sort_keys=True, default=str))
    setup_random_seed(getattr(args, "rng_seed", 1234), logger=logger)
    return logger, log_path, hash_value
