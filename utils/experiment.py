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
    return hashlib.sha1(hash_string.encode()).hexdigest()


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


def setup_experiment_logging(args, attack_name, level=logging.INFO):
    log_dir = os.path.join(_repo_root(), "logs")
    create_directory_safely(log_dir)

    hash_value = get_hash_value_for_args(args)
    log_path = os.path.join(log_dir, f"{attack_name}_{hash_value}.log")

    logger = logging.getLogger(attack_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    logger.info("Log file path: %s", log_path)
    logger.info("Argument hash: %s", hash_value)
    logger.info("Arguments:\n%s", json.dumps(args_to_dict(args), indent=2, sort_keys=True, default=str))
    setup_random_seed(getattr(args, "rng_seed", 1234), logger=logger)
    return logger, log_path, hash_value
