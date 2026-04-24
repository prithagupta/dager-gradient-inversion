import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from args_factory import get_args
from utils.data import TextDataset
from utils.experiment import cleanup_memory, get_results_dir, is_attack_complete, load_rouge_metric
from utils.experiment import setup_experiment_logging
from utils.filtering_decoder import filter_decoder
from utils.filtering_encoder import filter_encoder
from utils.functional import (
    _rouge_triplet,
    check_if_in_span,
    evaluate_prediction,
    fallback_gpt2_l1_candidates,
    fallback_rope_l1_candidates,
    filter_outliers,
    get_span_dists,
    get_top_B_in_span,
    log_distances,
    print_single_metric_dict,
    print_summary_table,
    remove_padding,
    summarize_metrics,
)
from utils.models import ModelWrapper


def get_augmented_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aug_global_topk", type=int, default=1024)
    parser.add_argument("--aug_global_expand_k", type=int, default=128)
    parser.add_argument("--aug_position_topk", type=int, default=128)
    parser.add_argument("--aug_score_chunk_size", type=int, default=256)
    parser.add_argument("--aug_alpha_subspace", type=float, default=1.0)
    parser.add_argument("--aug_beta_global", type=float, default=0.005)
    parser.add_argument("--aug_gamma_quality", type=float, default=0.0)
    parser.add_argument("--aug_delta_local", type=float, default=0.0005)
    parser.add_argument("--aug_expand_min_candidates", type=int, default=96)
    parser.add_argument("--aug_support_topk", type=int, default=3000)
    parser.add_argument("--aug_support_eps_ratio", type=float, default=1e-6)
    parser.add_argument("--aug_support_max_nonzero_frac", type=float, default=0.2)
    parser.add_argument("--aug_support_score_weight", type=float, default=1.0)
    parser.add_argument("--aug_alignment_score_weight", type=float, default=0.1)
    parser.add_argument(
        "--aug_disable_support_pool",
        action="store_true",
        help="Disable embedding-gradient support candidates and use only the alignment global pool.",
    )
    parser.set_defaults(aug_expand_all_positions=True)
    parser.add_argument(
        "--aug_expand_all_positions",
        dest="aug_expand_all_positions",
        action="store_true",
        help="Augment every decoder position with global candidates before reranking.",
    )
    parser.add_argument(
        "--aug_disable_expand_all_positions",
        dest="aug_expand_all_positions",
        action="store_false",
        help="Only augment low-candidate decoder positions instead of all positions.",
    )
    parser.add_argument(
        "--aug_quality_penalty_short",
        type=float,
        default=0.25,
        help="Penalty for very short alphabetic decoded tokens during reranking.",
    )
    parser.add_argument(
        "--aug_quality_penalty_punct",
        type=float,
        default=0.5,
        help="Penalty for punctuation-only decoded tokens during reranking.",
    )
    aug_args, base_argv = parser.parse_known_args()
    args = get_args(base_argv)
    for key, value in vars(aug_args).items():
        setattr(args, key, value)
    return args


args = get_augmented_args()
logger, log_path, job_hash = setup_experiment_logging(args, "augmented_attack")
logger.info(f"Arguments {args}")
logger.info("\n\n\nCommand: %s\n\n\n", " ".join(sys.argv))

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0


def _all_token_positions(mask, start=0, stop=None):
    if mask.ndim == 1:
        return torch.all(mask[start:stop]).item()
    if mask.ndim == 2:
        return torch.all(mask[:, start:stop]).item()
    raise ValueError(f"Expected 1D or 2D span mask, got shape {tuple(mask.shape)}")


def _derive_global_grad_signal(model_wrapper, true_grads):
    grad = true_grads[model_wrapper.layer_ids[0]]
    if grad is None:
        return None
    grad = grad.detach().float()
    embed_dim = model_wrapper.get_input_embeddings_weight().shape[1]

    if grad.ndim == 2:
        if grad.shape[-1] == embed_dim:
            signal = grad.mean(dim=0)
        elif grad.shape[0] == embed_dim:
            signal = grad.mean(dim=1)
        else:
            flat = grad.reshape(-1)
            if flat.numel() >= embed_dim:
                signal = flat[:embed_dim]
            else:
                signal = F.pad(flat, (0, embed_dim - flat.numel()))
    else:
        flat = grad.reshape(-1)
        if flat.numel() >= embed_dim:
            signal = flat[:embed_dim]
        else:
            signal = F.pad(flat, (0, embed_dim - flat.numel()))

    norm = signal.norm(p=2)
    if norm <= 0:
        return None
    return signal / norm


def _normalize_scores(scores):
    scores = scores.detach().float().cpu()
    finite = torch.isfinite(scores)
    if finite.any():
        finite_scores = scores[finite]
        min_score = finite_scores.min()
        max_score = finite_scores.max()
        denom = max_score - min_score
        if denom > 0:
            scores = (scores - min_score) / denom
        else:
            scores = torch.zeros_like(scores)
    scores[~finite] = 0.0
    return scores


def _find_input_embedding_grad(model_wrapper, true_grads):
    emb_weight = model_wrapper.get_input_embeddings_weight()
    trainable_params = [
        (name, param)
        for name, param in model_wrapper.model.named_parameters()
        if param.requires_grad
    ]
    for idx, (name, param) in enumerate(trainable_params):
        same_object = param is emb_weight
        same_storage = (
            param.shape == emb_weight.shape
            and param.data_ptr() == emb_weight.data_ptr()
        )
        if (same_object or same_storage) and idx < len(true_grads):
            return true_grads[idx], name
    return None, None


def _gold_unique_token_set(model_wrapper, orig_batch):
    if orig_batch is None or "input_ids" not in orig_batch:
        return set()
    ids = orig_batch["input_ids"].detach().reshape(-1).cpu()
    pad_id = model_wrapper.tokenizer.pad_token_id
    if pad_id is not None:
        ids = ids[ids != pad_id]
    return set(int(token_id) for token_id in ids.tolist())


def _log_pool_gold_recall(model_wrapper, orig_batch, pool_ids, label):
    gold_set = _gold_unique_token_set(model_wrapper, orig_batch)
    if not gold_set:
        return
    pool_set = set(int(token_id) for token_id in pool_ids.detach().cpu().tolist())
    hit = len(gold_set & pool_set)
    recall = hit / max(len(gold_set), 1)
    logger.info(
        "%s unique-token recall=%s/%s = %.4f",
        label,
        hit,
        len(gold_set),
        recall,
    )


def _compute_embedding_support_pool(args, model_wrapper, true_grads, orig_batch=None):
    if args.aug_disable_support_pool:
        return torch.empty(0, dtype=torch.long), torch.empty(0), "disabled"

    grad_wte, grad_name = _find_input_embedding_grad(model_wrapper, true_grads)
    if grad_wte is None:
        logger.info("Embedding-gradient support pool unavailable: input embedding gradient not found.")
        return torch.empty(0, dtype=torch.long), torch.empty(0), "missing"

    vocab_size = model_wrapper.get_input_embeddings_weight().shape[0]
    grad_wte = grad_wte.detach().float()
    if grad_wte.ndim != 2 or grad_wte.shape[0] != vocab_size:
        logger.info(
            "Embedding-gradient support pool unavailable: %s grad shape=%s, vocab=%s.",
            grad_name,
            tuple(grad_wte.shape),
            vocab_size,
        )
        return torch.empty(0, dtype=torch.long), torch.empty(vocab_size), "bad_shape"

    row_norm = torch.nan_to_num(grad_wte.norm(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    max_norm = row_norm.max().item()
    if max_norm <= 0:
        logger.info("Embedding-gradient support pool unavailable: all row norms are zero.")
        return torch.empty(0, dtype=torch.long), row_norm.cpu(), "zero"

    eps = max(1e-12, max_norm * args.aug_support_eps_ratio)
    support_ids = torch.where(row_norm > eps)[0]
    nonzero_frac = support_ids.numel() / max(row_norm.numel(), 1)
    mode = "support"

    if support_ids.numel() == 0 or nonzero_frac > args.aug_support_max_nonzero_frac:
        topk = min(args.aug_support_topk, row_norm.numel())
        support_ids = torch.topk(row_norm, k=topk).indices
        mode = "top_norm_dense" if nonzero_frac > args.aug_support_max_nonzero_frac else "top_norm_empty"
    elif support_ids.numel() > args.aug_support_topk:
        support_norms = row_norm[support_ids]
        top_idx = torch.topk(support_norms, k=args.aug_support_topk).indices
        support_ids = support_ids[top_idx]
        mode = "support_topk"

    support_scores = _normalize_scores(row_norm)
    logger.info(
        "Embedding support pool mode=%s | grad=%s | eps=%.3g | nonzero_frac=%.6f | size=%s | max_norm=%.6g",
        mode,
        grad_name,
        eps,
        nonzero_frac,
        support_ids.numel(),
        max_norm,
    )
    _log_pool_gold_recall(model_wrapper, orig_batch, support_ids, "Embedding support pool")
    return support_ids.cpu(), support_scores.cpu(), mode


def _compute_alignment_candidate_pool(args, model_wrapper, true_grads):
    grad_signal = _derive_global_grad_signal(model_wrapper, true_grads)
    if grad_signal is None:
        return torch.empty(0, dtype=torch.long), torch.empty(0)

    emb_matrix = model_wrapper.get_input_embeddings_weight().detach().float()
    emb_norm = F.normalize(emb_matrix, dim=-1)
    grad_signal = grad_signal.to(emb_norm.device)
    scores = emb_norm @ grad_signal

    blocked = [model_wrapper.pad_token, model_wrapper.eos_token, model_wrapper.start_token]
    for token_id in blocked:
        if token_id is not None and 0 <= token_id < scores.numel():
            scores[token_id] = -torch.inf

    topk = min(args.aug_global_topk, scores.numel())
    top_ids = torch.topk(scores, k=topk).indices
    logger.info(
        "Alignment global pool size=%s | best_score=%.6f | worst_score=%.6f",
        top_ids.numel(),
        scores[top_ids[0]].item(),
        scores[top_ids[-1]].item(),
    )
    return top_ids.cpu(), scores.cpu()


def _compute_global_candidate_pool(args, model_wrapper, true_grads, orig_batch=None):
    support_ids, support_scores, support_mode = _compute_embedding_support_pool(
        args,
        model_wrapper,
        true_grads,
        orig_batch=orig_batch,
    )
    alignment_ids, alignment_scores = _compute_alignment_candidate_pool(args, model_wrapper, true_grads)
    if alignment_ids.numel() > 0:
        _log_pool_gold_recall(model_wrapper, orig_batch, alignment_ids, "Alignment global pool")

    combined_ids = _ordered_unique_ids(support_ids, alignment_ids)
    vocab_size = model_wrapper.get_input_embeddings_weight().shape[0]
    combined_scores = torch.zeros(vocab_size)
    if support_scores.numel() == vocab_size:
        combined_scores += args.aug_support_score_weight * support_scores
    if alignment_scores.numel() == vocab_size:
        combined_scores += args.aug_alignment_score_weight * _normalize_scores(alignment_scores)

    logger.info(
        "Augmented global pool combined size=%s | support_size=%s | alignment_size=%s | support_mode=%s",
        combined_ids.numel(),
        support_ids.numel(),
        alignment_ids.numel(),
        support_mode,
    )
    _log_pool_gold_recall(model_wrapper, orig_batch, combined_ids, "Combined global pool")
    return combined_ids.cpu(), combined_scores.cpu()


def _special_token_ids(model_wrapper):
    tokens = {
        model_wrapper.pad_token,
        model_wrapper.eos_token,
        model_wrapper.start_token,
        model_wrapper.tokenizer.pad_token_id,
        model_wrapper.tokenizer.eos_token_id,
        model_wrapper.tokenizer.bos_token_id,
    }
    return {int(token_id) for token_id in tokens if token_id is not None and token_id >= 0}


def _ordered_unique_ids(*id_tensors):
    seen = set()
    ordered = []
    for ids in id_tensors:
        if ids is None or ids.numel() == 0:
            continue
        for token_id in ids.tolist():
            token_id = int(token_id)
            if token_id not in seen:
                seen.add(token_id)
                ordered.append(token_id)
    return torch.tensor(ordered, dtype=torch.long)


def _as_vocab_embeddings(embeds):
    if embeds.ndim == 2:
        return embeds
    if embeds.ndim == 3:
        if embeds.shape[0] == 1:
            return embeds[0]
        if embeds.shape[1] == 1:
            return embeds[:, 0]
    raise ValueError(f"Expected embeddings with vocab dimension, got shape {tuple(embeds.shape)}")


def _span_distances_for_ids(args, R_Q, vocab_embeds, candidate_ids):
    candidate_ids = candidate_ids.detach().reshape(-1).cpu()
    if candidate_ids.numel() == 0:
        return torch.empty(0)

    chunk_size = max(1, int(args.aug_score_chunk_size))
    all_dists = []
    for start in range(0, candidate_ids.numel(), chunk_size):
        ids_chunk = candidate_ids[start : start + chunk_size].to(vocab_embeds.device)
        embeds_chunk = vocab_embeds.index_select(0, ids_chunk)
        dists_chunk = check_if_in_span(R_Q, embeds_chunk, args.dist_norm).reshape(-1).detach().cpu()
        all_dists.append(dists_chunk)
        del embeds_chunk, dists_chunk

    return torch.cat(all_dists, dim=0)


def _token_quality_score(token_id, tokenizer, cache, args):
    if token_id in cache:
        return cache[token_id]

    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    stripped = token_text.strip()
    score = 0.0
    if stripped == "":
        score -= 1.0
    if len(stripped) <= 2 and stripped.isalpha():
        score -= args.aug_quality_penalty_short
    if stripped != "" and all(not ch.isalnum() for ch in stripped):
        score -= args.aug_quality_penalty_punct
    if stripped in {"'", '"', ",", ".", "-", "(", ")", "[", "]"}:
        score -= args.aug_quality_penalty_punct
    cache[token_id] = score
    return score


def _augment_decoder_position_candidates(
    args,
    model_wrapper,
    position,
    embeds,
    local_ids,
    global_ids,
    global_scores,
    quality_cache,
    R_Q,
):
    vocab_embeds = _as_vocab_embeddings(embeds)
    local_ids_cpu = local_ids.detach().reshape(-1).cpu()
    if local_ids_cpu.numel() == 0:
        should_expand = True
    else:
        should_expand = args.aug_expand_all_positions or local_ids_cpu.numel() < args.aug_expand_min_candidates

    position_global_ids = torch.empty(0, dtype=torch.long)
    if should_expand and global_ids.numel() > 0:
        global_dists = _span_distances_for_ids(args, R_Q, vocab_embeds, global_ids)
        expand_k = min(args.aug_global_expand_k, global_ids.numel())
        if expand_k > 0:
            top_idx = torch.topk(-global_dists, k=expand_k).indices
            position_global_ids = global_ids[top_idx].cpu()

    combined_ids = _ordered_unique_ids(local_ids_cpu, position_global_ids)
    if combined_ids.numel() == 0:
        return combined_ids

    combined_dists = _span_distances_for_ids(args, R_Q, vocab_embeds, combined_ids)
    local_set = {int(token_id) for token_id in local_ids_cpu.tolist()}
    special_set = _special_token_ids(model_wrapper)
    scored = []
    for idx, token_id in enumerate(combined_ids.tolist()):
        subspace_score = -float(combined_dists[idx].item())
        global_score = float(global_scores[token_id].item()) if token_id < global_scores.numel() else 0.0
        quality_score = _token_quality_score(token_id, model_wrapper.tokenizer, quality_cache, args)
        local_bonus = args.aug_delta_local if token_id in local_set else 0.0
        total = (
            args.aug_alpha_subspace * subspace_score
            + args.aug_beta_global * global_score
            + args.aug_gamma_quality * quality_score
            + local_bonus
        )
        scored.append((token_id, total, subspace_score, global_score, quality_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    keep = args.aug_position_topk
    if args.max_ids > 0:
        keep = min(keep, args.max_ids)
    keep = min(keep, len(scored))
    protected = [item for item in scored if item[0] in special_set and item[0] in local_set]
    kept = scored[:keep]
    if protected and keep > 0:
        kept_ids = {item[0] for item in kept}
        for item in protected:
            if item[0] not in kept_ids:
                kept[-1] = item
                kept_ids.add(item[0])
                break
    if kept:
        logger.info(
            "Aug L1 Position %s | local=%s | global_added=%s | merged=%s | kept=%s | best_total=%.6f",
            position,
            local_ids_cpu.numel(),
            position_global_ids.numel(),
            len(scored),
            keep,
            kept[0][1],
        )
    return torch.tensor([token_id for token_id, *_ in kept], dtype=torch.long)


def filter_l1(args, model_wrapper, R_Qs, true_grads, max_positions=None, orig_batch=None):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []

    sentence_ends = []
    p = 0

    global_ids = torch.empty(0, dtype=torch.long)
    global_scores = torch.empty(0)
    quality_cache = {}
    if model_wrapper.is_decoder():
        global_ids, global_scores = _compute_global_candidate_pool(
            args,
            model_wrapper,
            true_grads,
            orig_batch=orig_batch,
        )

    while True:
        if max_positions is not None and p >= max_positions:
            logger.info("Stopping L1 at tokenized batch length %s.", max_positions)
            break
        logger.info(f"L1 Position {p}")
        embeds = model_wrapper.get_embeddings(p)
        distance_values = None
        if model_wrapper.is_bert():
            if args.defense_noise is None:
                _, res_ids_new, res_types_new = get_top_B_in_span(
                    R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm
                )
            else:
                raise NotImplementedError
        else:
            if args.defense_noise is None:
                _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
                if model_wrapper.has_rope() and len(res_ids_new) == 0:
                    res_ids_new = fallback_rope_l1_candidates(args, model_wrapper, R_Qs[0], embeds)
                elif model_wrapper.is_decoder() and len(res_ids_new) == 0:
                    res_ids_new = fallback_gpt2_l1_candidates(args, model_wrapper, R_Qs[0], embeds)
            else:
                std_thrs = args.p1_std_thrs if p == 0 else None
                distance_values = get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(
                    distance_values,
                    std_thrs=std_thrs,
                    maxB=max(50 * model_wrapper.args.batch_size, int(0.05 * len(model_wrapper.tokenizer))),
                )
            res_types_new = torch.zeros_like(res_ids_new)

            if model_wrapper.is_decoder() and global_ids.numel() > 0:
                res_ids_new = _augment_decoder_position_candidates(
                    args,
                    model_wrapper,
                    p,
                    embeds,
                    res_ids_new.reshape(-1),
                    global_ids,
                    global_scores,
                    quality_cache,
                    R_Qs[0],
                )
                res_types_new = torch.zeros_like(res_ids_new)

        log_distances(res_ids_new, R_Qs[0], embeds, args.dist_norm, p, dists=distance_values, log=logger)
        res_pos_new = torch.ones_like(res_ids_new) * p

        del embeds

        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > tokenizer.model_max_length or p > args.max_len:
            break
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][end_token_ind]
            sentence_ends.append((p, sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind + 1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind + 1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        p += 1
        if model_wrapper.has_rope():
            break

    return res_pos, res_ids, res_types, sentence_ends


def reconstruct(args, device, sample, metric, model_wrapper: ModelWrapper):
    global total_correct_tokens, total_tokens, total_correct_maxB_tokens

    tokenizer = model_wrapper.tokenizer
    sequences, true_labels = sample
    orig_batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)

    true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape) * args.defense_noise
    prediction, predicted_sentences, predicted_sentences_scores = [], [], []

    with torch.no_grad():
        B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        R_Q2 = R_Qs[1]

        if B is None:
            reference = [
                remove_padding(tokenizer, orig_batch["input_ids"][i], left=(args.pad == "left"))
                for i in range(orig_batch["input_ids"].shape[0])
            ]
            del true_grads, orig_batch
            cleanup_memory()
            return ["" for _ in range(len(reference))], reference

        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range(orig_batch["input_ids"].shape[1]):
            total_true_token_count2 += args.batch_size - (orig_batch["input_ids"][:, i] == model_wrapper.pad_token).sum()
            uniques = torch.unique(orig_batch["input_ids"][:, i])
            total_true_token_count += uniques.numel()
            if model_wrapper.pad_token in uniques.tolist():
                total_true_token_count -= 1

        logger.info(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        max_l1_positions = orig_batch["input_ids"].shape[1] if model_wrapper.is_decoder() else None
        res_pos, res_ids, res_types, sentence_ends = filter_l1(
            args,
            model_wrapper,
            R_Qs,
            true_grads,
            max_positions=max_l1_positions,
            orig_batch=orig_batch,
        )

        if len(res_ids) == 0:
            reference = [
                remove_padding(tokenizer, orig_batch["input_ids"][i], left=(args.pad == "left"))
                for i in range(orig_batch["input_ids"].shape[0])
            ]
            return ["" for _ in reference], reference
        if len(res_ids[0]) < 500:
            logger.info("L1 candidate counts: %s", [len(ids) for ids in res_ids])

        rec_l1, rec_l1_maxB, rec_l2 = [], [], []

        for s in range(orig_batch["input_ids"].shape[0]):
            sentence_in = True
            sentence_in_max_B = True
            orig_sentence = orig_batch["input_ids"][s]
            last_idx = torch.where(orig_batch["input_ids"][s] != tokenizer.pad_token_id)[0][-1].item()
            for pos, token in enumerate(orig_sentence):
                if not model_wrapper.is_bert() and pos == last_idx:
                    break
                if pos >= len(res_ids) and not model_wrapper.has_rope():
                    sentence_in = False
                    break
                if token == model_wrapper.pad_token and args.pad == "right":
                    pos -= 1
                    break
                elif token == model_wrapper.pad_token and args.pad == "left":
                    continue
                if model_wrapper.has_rope():
                    total_correct_tokens += 1 if token in res_ids[0] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[0][: min(args.batch_size, len(res_ids[0]))] else 0
                    total_tokens += 1
                else:
                    total_correct_tokens += 1 if token in res_ids[pos] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[pos][: min(args.batch_size, len(res_ids[pos]))] else 0
                    total_tokens += 1
                if token == model_wrapper.eos_token and args.pad == "right":
                    break

                if model_wrapper.has_rope():
                    if model_wrapper.has_bos() and token == model_wrapper.start_token:
                        continue
                    sentence_in = sentence_in and (token in res_ids[0])
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[0][: min(args.batch_size, len(res_ids[0]))])
                else:
                    sentence_in = sentence_in and (token in res_ids[pos])
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[pos][: min(args.batch_size, len(res_ids[pos]))])
            if model_wrapper.is_bert():
                sentence_in = sentence_in and (pos, orig_batch["token_type_ids"][s][pos]) in sentence_ends
                sentence_in_max_B = sentence_in and (pos, orig_batch["token_type_ids"][s][pos]) in sentence_ends

            rec_l1.append(sentence_in)
            rec_l1_maxB.append(sentence_in_max_B)
            if model_wrapper.has_rope():
                orig_sentence = orig_sentence.reshape(1, -1)
            else:
                orig_sentence = orig_sentence[: pos + 1].reshape(1, -1)
            if model_wrapper.is_bert():
                token_type_ids = orig_batch["token_type_ids"][s][: orig_sentence.shape[1]].reshape(1, -1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, token_type_ids)[0]
            else:
                attention_mask = orig_batch["attention_mask"][s][: orig_sentence.shape[1]].reshape(1, -1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, attention_mask=attention_mask)[0]

            sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
            l2_span_thresh = model_wrapper.effective_l2_span_thresh(args.l2_span_thresh)
            boolsq2 = sizesq2 < l2_span_thresh
            if model_wrapper.has_rope():
                special_tokens = torch.tensor(
                    [model_wrapper.pad_token, model_wrapper.start_token],
                    device=orig_sentence.device,
                    dtype=orig_sentence.dtype,
                )
                boolsq2 = torch.logical_or(boolsq2, torch.isin(orig_sentence, special_tokens))
            logger.info(sizesq2)

            if args.task == "next_token_pred":
                rec_l2.append(_all_token_positions(boolsq2, stop=-1))
            elif model_wrapper.has_rope():
                rec_l2.append(_all_token_positions(boolsq2, start=1))
            else:
                rec_l2.append(torch.all(boolsq2).item())

        logger.info(
            f"Rec L1: {rec_l1}, Rec L1 MaxB: {rec_l1_maxB}, Rec MaxB Token: {total_correct_maxB_tokens / total_tokens}, Rec Token: {total_correct_tokens / total_tokens}, Rec L2: {rec_l2}"
        )

        if model_wrapper.is_decoder():
            max_ids = -1
            for ids in res_ids:
                if len(ids) > args.max_ids:
                    max_ids = args.max_ids
            predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores = filter_decoder(
                args, model_wrapper, R_Qs, res_ids, max_ids=max_ids
            )
            if len(predicted_sentences) < orig_batch["input_ids"].shape[0]:
                predicted_sentences += top_B_incorrect_sentences
                predicted_sentences_scores += top_B_incorrect_scores
        else:
            for l, token_type in sentence_ends:
                if args.l1_filter == "maxB":
                    max_ids = args.batch_size
                elif args.l1_filter == "all":
                    max_ids = -1
                else:
                    raise AssertionError

                if args.l2_filter == "non-overlap":
                    correct_sentences = []
                    approx_sentences = []
                    approx_scores = []
                    for sent, sc in zip(predicted_sentences, predicted_sentences_scores):
                        if sc < model_wrapper.effective_l2_span_thresh(args.l2_span_thresh):
                            correct_sentences.append(sent)
                        else:
                            approx_sentences.append(sent)
                            approx_scores.append(sc)

                    new_predicted_sentences, new_predicted_scores = filter_encoder(
                        args,
                        model_wrapper,
                        R_Q2,
                        l,
                        token_type,
                        res_ids,
                        correct_sentences,
                        approx_sentences,
                        approx_scores,
                        max_ids,
                        args.batch_size,
                    )
                elif args.l2_filter == "overlap":
                    new_predicted_sentences, new_predicted_scores = filter_encoder(
                        args, model_wrapper, R_Q2, l, token_type, res_ids, [], [], [], max_ids, args.batch_size
                    )
                else:
                    raise AssertionError

                predicted_sentences += new_predicted_sentences
                predicted_sentences_scores += new_predicted_scores

    reference = [
        remove_padding(tokenizer, orig_batch["input_ids"][i, : tokenizer.model_max_length], left=(args.pad == "left"))
        for i in range(orig_batch["input_ids"].shape[0])
    ]

    if len(predicted_sentences) == 0:
        logger.info("Decoder produced no candidate reconstructions after augmented L1/L2 filtering; returning empty predictions.")
        return ["" for _ in range(len(reference))], reference

    correct_sentences = []
    approx_sentences = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    max_len = max([len(s) for s in predicted_sentences])
    for sent, sc in zip(predicted_sentences, predicted_sentences_scores):
        if sc < model_wrapper.effective_l2_span_thresh(args.l2_span_thresh):
            correct_sentences.append(sent)
        else:
            approx_sentences.append(sent)
            approx_sentences_ext.append(sent + [-1] * (max_len - len(sent)))
            approx_sentences_lens.append(len(sent))
            approx_scores.append(sc)
    approx_scores = torch.tensor(approx_scores)
    approx_sentences_lens = torch.tensor(approx_sentences_lens)

    if len(approx_sentences) > 0:
        for i in range(len(correct_sentences)):
            sent = correct_sentences[i]
            similar_sentences = (
                (torch.tensor(sent) == torch.tensor(approx_sentences_ext)[:, : len(sent)]).sum(1)
                >= torch.min(approx_sentences_lens, torch.tensor(len(sent))) * args.distinct_thresh
            )
            approx_scores[similar_sentences] = torch.inf

        predicted_sentences = correct_sentences.copy()
        for _ in range(len(correct_sentences), args.batch_size):
            idx = torch.argmin(approx_scores)
            predicted_sentences.append(approx_sentences[idx])
            similar_sentences = (
                (torch.tensor(approx_sentences_ext[idx]) == torch.tensor(approx_sentences_ext)).sum(1)
                >= max_len * args.distinct_thresh
            )
            approx_scores[similar_sentences] = torch.inf

    for s in predicted_sentences:
        prediction.append(tokenizer.decode(s))

    if len(prediction) > len(reference):
        prediction = prediction[: len(reference)]

    if model_wrapper.is_decoder():
        new_prediction = []
        og_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        for i in range(len(reference)):
            sequences = [reference[i]] + prediction
            batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
            best_idx = (batch["input_ids"][1:] == batch["input_ids"][0]).sum(1).argmax()
            new_prediction.append(prediction[best_idx])
        tokenizer.padding_side = og_side
        prediction = new_prediction
    else:
        cost = np.zeros((len(prediction), len(prediction)))
        for i in range(len(prediction)):
            for j in range(len(prediction)):
                fm, _, _ = _rouge_triplet(metric.compute(predictions=[prediction[i]], references=[reference[j]])["rouge1"])
                cost[i, j] = 1.0 - fm
        row_ind, col_ind = linear_sum_assignment(cost)
        ids = list(range(len(prediction)))
        ids.sort(key=lambda i: col_ind[i])
        prediction = [prediction[ids[i]] for i in range(len(prediction))]

    return prediction, reference


def main():
    attack_name = f"augmented_dager_{args.loss}"
    is_complete, results_dir = is_attack_complete(attack_name, job_hash)
    print(f"Hash Value {job_hash} Started")
    if is_complete:
        logger.info("Results already exist for this config at %s; skipping attack.", results_dir)
        logger.info("Done with all.")
        print(f"Hash Value {job_hash} is already done")
        return

    device = torch.device(args.device)
    metric = load_rouge_metric(cache_dir=args.cache_dir, logger=logger)
    dataset = TextDataset(device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir, use_hf_split=args.use_hf_split)
    model_wrapper = ModelWrapper(args)
    wrapper_tokenizer = model_wrapper.tokenizer

    logger.info("\n\nAttacking with augmented DAGER..\n")
    predictions, references = [], []
    final_results = []
    final_per_input_results = []
    input_times = []
    sentence_rows = []
    input_rows = []
    results_dir = get_results_dir(attack_name, job_hash)
    os.makedirs(results_dir, exist_ok=True)
    t_start = time.time()

    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
        t_input_start = time.time()
        sample = dataset[i]
        logger.info("Running input #%d of %d.", i, args.n_inputs)
        logger.info("reference:")
        for seq in sample[0]:
            logger.info("========================")
            logger.info(seq)
        logger.info("========================")

        prediction, reference = reconstruct(args, device, sample, metric, model_wrapper)
        predictions += prediction
        references += reference

        logger.info("Done with input #%d of %d.", i, args.n_inputs)
        curr_metrics = []
        for sent_idx, (ref, pred) in enumerate(zip(reference, prediction)):
            logger.info("========================")
            logger.info("Reference: %s", ref)
            logger.info("Prediction: %s", pred)
            metrics = evaluate_prediction(pred, ref, wrapper_tokenizer, metric)
            curr_metrics.append(metrics)
            sentence_rows.append(
                {
                    "run_id": job_hash,
                    "attack": attack_name,
                    "model": args.model_path,
                    "dataset": args.dataset,
                    "input_index": i,
                    "sentence_index": sent_idx,
                    "reference": ref,
                    "prediction": pred,
                    **metrics,
                }
            )
        logger.info("========================")
        summary = summarize_metrics(curr_metrics)
        final_results.extend(curr_metrics)
        logger.info("[Curr input metrics]:")
        logger.info("%s", print_summary_table(summary))
        logger.info("[Aggregate metrics]:")
        aggregated_results = evaluate_prediction(" ".join(prediction), " ".join(reference), wrapper_tokenizer, metric)

        final_per_input_results.append(aggregated_results)
        logger.info("%s", print_single_metric_dict(aggregated_results))
        input_time_sec = time.time() - t_input_start
        total_time_sec = time.time() - t_start
        input_time = str(datetime.timedelta(seconds=input_time_sec)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=total_time_sec)).split(".")[0]

        logger.info("input #%d time: %s | total time: %s", i, input_time, total_time)
        input_rows.append(
            {
                "run_id": job_hash,
                "attack": attack_name,
                "model": args.model_path,
                "dataset": args.dataset,
                "input_index": i,
                "num_sentences": len(reference),
                "joined_reference": " ".join(reference),
                "joined_prediction": " ".join(prediction),
                "reconstruction_time_sec": input_time_sec,
                **aggregated_results,
            }
        )
        input_times.append(input_time_sec)
        del sample, prediction, reference, curr_metrics, aggregated_results
        cleanup_memory()
        logger.info("")
        logger.info("")

    logger.info("[Aggregate metrics]:")
    total_time_sec = time.time() - t_start
    aggregated_results = evaluate_prediction(" ".join(predictions), " ".join(references), wrapper_tokenizer, metric)
    aggregated_results["experiment_time_mean"] = float(total_time_sec)
    aggregated_results["experiment_time_std"] = float(0)
    logger.info("Overall %s", print_single_metric_dict(aggregated_results))
    summary = summarize_metrics(final_results)
    summary["reconstruction_time_mean"] = float(np.mean(input_times))
    summary["reconstruction_time_std"] = float(np.std(input_times))
    logger.info("Per Sentence%s", print_summary_table(summary))
    summary_per_input = summarize_metrics(final_per_input_results)
    summary_per_input["reconstruction_time_mean"] = float(np.mean(input_times))
    summary_per_input["reconstruction_time_std"] = float(np.std(input_times))
    logger.info("Per Input Results %s", print_summary_table(summary_per_input))
    summary_results = {
        "Overall Results": aggregated_results,
        "Per Sentence Results": summary,
        "Per Input Results": summary_per_input,
        "Arguments": vars(args),
    }
    logger.info("Experiment time %s", total_time_sec)
    pd.DataFrame(sentence_rows).to_csv(os.path.join(results_dir, "sentence_results.csv"), index=False)
    pd.DataFrame(input_rows).to_csv(os.path.join(results_dir, "input_results.csv"), index=False)
    with open(os.path.join(results_dir, "run_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)
    logger.info("Done with all.")
    print(f"Hash Value {job_hash} Done")


if __name__ == "__main__":
    main()
