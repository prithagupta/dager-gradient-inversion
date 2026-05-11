import datetime
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from args_factory import get_args
from utils.data import TextDataset
from utils.experiment import (
    args_to_dict,
    cleanup_memory,
    get_results_dir,
    is_attack_complete,
    load_partial_attack_state,
    load_rouge_metric,
    release_all_log_locks,
    setup_experiment_logging,
    write_attack_artifacts,
)
from utils.filtering_decoder import filter_decoder
from utils.functional import (
    _safe_aggregated_metrics,
    evaluate_prediction,
    extract_canary_metric_means,
    extract_canary_metric_summary,
    fallback_gpt2_l1_candidates,
    fallback_rope_l1_candidates,
    get_span_dists,
    get_top_B_in_span,
    maybe_add_canary_audit_metrics,
    print_single_metric_dict,
    print_summary_table,
    remove_padding,
    summarize_metrics,
)
from utils.models import ModelWrapper

ATTACK_NAME = "autoregressivedager"

args = get_args()
setattr(args, "ar_dager_rerank_batches", int(os.environ.get("AR_DAGER_RERANK_BATCHES", "4")))
setattr(args, "ar_dager_candidates_per_pos", int(os.environ.get("AR_DAGER_CANDIDATES_PER_POS", "96")))
setattr(args, "ar_dager_diversity_penalty", float(os.environ.get("AR_DAGER_DIVERSITY_PENALTY", "0.35")))
setattr(args, "ar_dager_support_weight", float(os.environ.get("AR_DAGER_SUPPORT_WEIGHT", "0.35")))
setattr(args, "ar_dager_position_weight", float(os.environ.get("AR_DAGER_POSITION_WEIGHT", "0.75")))
setattr(args, "ar_dager_quality_weight", float(os.environ.get("AR_DAGER_QUALITY_WEIGHT", "0.25")))
setattr(args, "ar_dager_refine_trials", int(os.environ.get("AR_DAGER_REFINE_TRIALS", "64")))

logger, log_path, job_hash, log_claim_acquired = setup_experiment_logging(args, ATTACK_NAME)
logger.info("Arguments %s", args)
logger.info("\n\n\nCommand: %s\n\n\n", " ".join(sys.argv))


def _log_text(text, max_chars=320):
    text = str(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def _all_token_positions(mask, start=0, stop=None):
    if mask.ndim == 1:
        return torch.all(mask[start:stop]).item()
    if mask.ndim == 2:
        return torch.all(mask[:, start:stop]).item()
    raise ValueError(f"Expected 1D or 2D span mask, got shape {tuple(mask.shape)}")


def filter_l1(args, model_wrapper, R_Qs, max_positions=None):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
    sentence_ends = []
    p = 0

    while True:
        if max_positions is not None and p >= max_positions:
            break
        logger.info("L1 Position %s", p)
        embeds = model_wrapper.get_embeddings(p)
        distance_values = None
        if model_wrapper.is_bert():
            if args.defense_noise is None:
                _, res_ids_new, res_types_new = get_top_B_in_span(
                    R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm
                )
            else:
                raise NotImplementedError("Autoregressive DAGER currently expects undefended DAGER L1 for BERT.")
        else:
            if args.defense_noise is None:
                _, res_ids_new = get_top_B_in_span(
                    R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm
                )
                if model_wrapper.has_rope() and len(res_ids_new) == 0:
                    res_ids_new = fallback_rope_l1_candidates(args, model_wrapper, R_Qs[0], embeds)
                elif model_wrapper.is_decoder() and len(res_ids_new) == 0:
                    res_ids_new = fallback_gpt2_l1_candidates(args, model_wrapper, R_Qs[0], embeds)
            else:
                std_thrs = args.p1_std_thrs if p == 0 else None
                distance_values = get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                from utils.functional import filter_outliers

                res_ids_new = filter_outliers(
                    distance_values,
                    std_thrs=std_thrs,
                    maxB=max(50 * model_wrapper.args.batch_size, int(0.05 * len(model_wrapper.tokenizer))),
                )
            res_types_new = torch.zeros_like(res_ids_new)

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
        res_ids.append(ids)
        res_pos += res_pos_new.tolist()
        p += 1
        if model_wrapper.has_rope():
            break

    return res_pos, res_ids, res_types, sentence_ends


def _find_input_embedding_grad(model_wrapper, true_grads):
    emb_weight = model_wrapper.get_input_embeddings_weight()
    trainable_params = [(name, param) for name, param in model_wrapper.model.named_parameters() if param.requires_grad]
    for idx, (name, param) in enumerate(trainable_params):
        same_object = param is emb_weight
        same_storage = param.shape == emb_weight.shape and param.data_ptr() == emb_weight.data_ptr()
        if (same_object or same_storage) and idx < len(true_grads):
            return true_grads[idx], name
    return None, None


def _find_position_embedding_grad(model_wrapper, true_grads):
    position_weight = None
    if hasattr(model_wrapper.model, "transformer") and hasattr(model_wrapper.model.transformer, "wpe"):
        position_weight = model_wrapper.model.transformer.wpe.weight
    elif hasattr(model_wrapper.model, "bert") and hasattr(model_wrapper.model.bert, "embeddings"):
        position_weight = model_wrapper.model.bert.embeddings.position_embeddings.weight
    if position_weight is None:
        return None, None

    trainable_params = [(name, param) for name, param in model_wrapper.model.named_parameters() if param.requires_grad]
    for idx, (name, param) in enumerate(trainable_params):
        same_object = param is position_weight
        same_storage = param.shape == position_weight.shape and param.data_ptr() == position_weight.data_ptr()
        name_match = name.endswith("wpe.weight") or "position_embeddings.weight" in name
        if (same_object or same_storage or name_match) and idx < len(true_grads):
            return true_grads[idx], name
    return None, None


def build_grad_token_scores(model_wrapper, true_grads, res_ids, max_positions):
    grad_wte, wte_name = _find_input_embedding_grad(model_wrapper, true_grads)
    if grad_wte is None or grad_wte.ndim != 2:
        return {}, [{} for _ in range(min(max_positions, len(res_ids)))], None, None

    grad_wte = grad_wte.detach().float().cpu()
    row_norm = torch.nan_to_num(grad_wte.norm(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    max_norm = row_norm.max().clamp_min(1e-12)
    all_candidate_ids = set()
    for ids in res_ids[:max_positions]:
        all_candidate_ids.update(int(token_id) for token_id in ids)
    support_scores = {
        token_id: float((row_norm[token_id] / max_norm).item())
        for token_id in all_candidate_ids
        if 0 <= token_id < row_norm.numel()
    }

    position_scores = [{} for _ in range(min(max_positions, len(res_ids)))]
    grad_wpe, wpe_name = _find_position_embedding_grad(model_wrapper, true_grads)
    if grad_wpe is None or grad_wpe.ndim != 2:
        return support_scores, position_scores, wte_name, None

    grad_wpe = grad_wpe.detach().float().cpu()
    for pos in range(len(position_scores)):
        if pos >= grad_wpe.shape[0] or len(res_ids[pos]) == 0:
            continue
        candidate_ids = [int(token_id) for token_id in res_ids[pos] if 0 <= int(token_id) < grad_wte.shape[0]]
        if not candidate_ids:
            continue
        pos_grad = grad_wpe[pos]
        if pos_grad.norm(p=2) <= 0:
            continue
        cand_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        cand_vecs = torch.nn.functional.normalize(grad_wte.index_select(0, cand_tensor), dim=-1)
        pos_vec = torch.nn.functional.normalize(pos_grad.reshape(1, -1), dim=-1).reshape(-1)
        scores = torch.nan_to_num(cand_vecs @ pos_vec, nan=0.0, posinf=0.0, neginf=0.0)
        position_scores[pos] = {token_id: float(score) for token_id, score in zip(candidate_ids, scores.tolist())}
    return support_scores, position_scores, wte_name, wpe_name


def grad_match_loss(dummy_grads, true_grads, args):
    loss = None
    n_g = 0
    for dummy_grad, true_grad in zip(dummy_grads, true_grads):
        if dummy_grad is None or true_grad is None:
            continue
        true_grad = true_grad.to(dummy_grad.device)
        if args.grad_loss == "cos":
            curr = 1.0 - (dummy_grad * true_grad).sum() / (
                dummy_grad.reshape(-1).norm(p=2) * true_grad.reshape(-1).norm(p=2) + 1e-9
            )
        elif args.grad_loss == "dlg":
            curr = (dummy_grad - true_grad).square().sum()
        elif args.grad_loss == "tag":
            diff = dummy_grad - true_grad
            curr = diff.square().sum() + args.tag_factor * diff.abs().sum()
        else:
            raise ValueError(f"Unknown grad_loss: {args.grad_loss}")
        loss = curr if loss is None else loss + curr
        n_g += 1
    if loss is None:
        raise RuntimeError("No comparable gradients were produced for autoregressive DAGER reranking.")
    if args.grad_loss == "cos":
        loss = loss / max(n_g, 1)
    return loss


def compute_rec_loss_from_ids(args, model_wrapper, true_grads, ids, attention_mask, true_labels):
    emb_matrix = model_wrapper.get_input_embeddings_weight().detach().to(args.device)
    x_embeds = emb_matrix[ids.to(args.device)].detach().requires_grad_(True)
    dummy_grads = model_wrapper.compute_grads_from_embeds(
        x_embeds,
        true_labels,
        attention_mask=attention_mask,
        create_graph=False,
    )
    loss = grad_match_loss(dummy_grads, true_grads, args).detach()
    del x_embeds, dummy_grads
    return loss


def _creates_repeated_ngram(prefix, token_id, n):
    if len(prefix) < n - 1:
        return False
    gram = tuple(prefix[-(n - 1):] + [int(token_id)])
    for start in range(0, len(prefix) - n + 1):
        if tuple(prefix[start:start + n]) == gram:
            return True
    return False


def _token_quality_score(tokenizer, prefix_tokens, token_id):
    token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    stripped = token_text.strip()
    if stripped == "":
        return -1.0
    if len(prefix_tokens) == 0:
        return 0.15 if stripped[:1].isalnum() else -0.25
    score = 0.0
    if token_text.startswith(" ") and stripped[:1].isalnum():
        score += 0.15
    if stripped[:1].isalnum() and not token_text.startswith(" ") and len(prefix_tokens) > 0:
        score -= 0.15
    if all(not ch.isalnum() for ch in stripped):
        score -= 0.10
    return score


def _candidate_ids_for_pos(args, model_wrapper, res_ids, pos):
    pad_token = int(model_wrapper.pad_token)
    eos_token = int(model_wrapper.eos_token) if model_wrapper.eos_token is not None else pad_token
    if model_wrapper.has_rope():
        ids = list(res_ids[0]) if res_ids else []
    elif pos < len(res_ids):
        ids = list(res_ids[pos])
    else:
        ids = []
    ids = [int(token_id) for token_id in ids if int(token_id) != pad_token]
    if pos == 0 and model_wrapper.has_bos() and model_wrapper.start_token is not None:
        return [int(model_wrapper.start_token)]
    ids = [token_id for token_id in ids if token_id != eos_token]
    cap = max(1, int(args.ar_dager_candidates_per_pos))
    return ids[:cap]


def _allowed_candidate_set(args, model_wrapper, res_ids, pos):
    allowed = set(_candidate_ids_for_pos(args, model_wrapper, res_ids, pos))
    if model_wrapper.has_bos() and model_wrapper.start_token is not None and pos == 0:
        allowed.add(int(model_wrapper.start_token))
    allowed.add(int(model_wrapper.pad_token))
    return allowed


def log_candidate_diagnostics(args, model_wrapper, orig_batch, final_ids, res_ids):
    input_ids = orig_batch["input_ids"].detach().cpu()
    attention_mask = orig_batch["attention_mask"].detach().cpu().bool()
    final_cpu = final_ids.detach().cpu()
    true_total = 0
    true_hits = 0
    generated_total = 0
    generated_violations = 0
    sentence_possible = []
    violation_examples = []

    for row_idx in range(input_ids.shape[0]):
        row_possible = True
        for pos in range(input_ids.shape[1]):
            if not bool(attention_mask[row_idx, pos].item()):
                continue
            allowed = _allowed_candidate_set(args, model_wrapper, res_ids, pos)
            true_token = int(input_ids[row_idx, pos].item())
            pred_token = int(final_cpu[row_idx, pos].item())
            true_total += 1
            generated_total += 1
            true_in = true_token in allowed
            pred_in = pred_token in allowed
            true_hits += int(true_in)
            generated_violations += int(not pred_in)
            row_possible = row_possible and true_in
            if not pred_in and len(violation_examples) < 8:
                violation_examples.append((row_idx, pos, pred_token))
        sentence_possible.append(row_possible)

    logger.info(
        "Autoregressive DAGER candidate diagnostics | true_token_recall=%.4f (%s/%s) | "
        "true_sentence_recall=%.4f (%s/%s) | generated_violation_rate=%.6f (%s/%s) | violation_examples=%s",
        true_hits / max(true_total, 1),
        true_hits,
        true_total,
        sum(sentence_possible) / max(len(sentence_possible), 1),
        sum(sentence_possible),
        len(sentence_possible),
        generated_violations / max(generated_total, 1),
        generated_violations,
        generated_total,
        violation_examples,
    )


def _score_candidate(args, tokenizer, token_id, pos, prefix, support_scores, position_scores, pos_counts, global_counts):
    repeat_penalty = 0.0
    if token_id in prefix[-8:]:
        repeat_penalty += 1.0
    if _creates_repeated_ngram(prefix, token_id, 3) or _creates_repeated_ngram(prefix, token_id, 4):
        repeat_penalty += 2.0
    diversity = pos_counts.get(int(token_id), 0) + 0.15 * max(0, global_counts.get(int(token_id), 0) - 4)
    return (
        args.ar_dager_position_weight * position_scores[pos].get(int(token_id), 0.0)
        + args.ar_dager_support_weight * support_scores.get(int(token_id), 0.0)
        + args.ar_dager_quality_weight * _token_quality_score(tokenizer, prefix, token_id)
        - args.ar_dager_diversity_penalty * diversity
        - 0.45 * repeat_penalty
    )


def decode_autoregressive_batch(args, model_wrapper, res_ids, attention_mask, support_scores, position_scores, variant_idx):
    tokenizer = model_wrapper.tokenizer
    batch_size, seq_len = attention_mask.shape
    ids = torch.full((batch_size, seq_len), int(model_wrapper.pad_token), dtype=torch.long, device=args.device)
    pos_counts = [dict() for _ in range(seq_len)]
    global_counts = {}

    row_order = list(range(batch_size))
    if row_order:
        shift = variant_idx % len(row_order)
        row_order = row_order[shift:] + row_order[:shift]

    prefixes = [[] for _ in range(batch_size)]
    attn_cpu = attention_mask.detach().cpu().bool()
    for pos in range(seq_len):
        candidates = _candidate_ids_for_pos(args, model_wrapper, res_ids, pos)
        if not candidates:
            continue
        for row_idx in row_order:
            if not bool(attn_cpu[row_idx, pos].item()):
                continue
            prefix = prefixes[row_idx]
            ranked = sorted(
                candidates,
                key=lambda token_id: _score_candidate(
                    args,
                    tokenizer,
                    token_id,
                    pos if pos < len(position_scores) else len(position_scores) - 1,
                    prefix,
                    support_scores,
                    position_scores,
                    pos_counts[pos],
                    global_counts,
                ),
                reverse=True,
            )
            offset = 0
            if len(ranked) > 1 and variant_idx > 0:
                offset = (variant_idx + row_idx + pos) % min(len(ranked), max(2, args.batch_size))
            chosen = int(ranked[offset])
            ids[row_idx, pos] = chosen
            prefixes[row_idx].append(chosen)
            pos_counts[pos][chosen] = pos_counts[pos].get(chosen, 0) + 1
            global_counts[chosen] = global_counts.get(chosen, 0) + 1
    return ids


def _ids_from_sentences(args, model_wrapper, sentences, attention_mask, seq_len):
    if not sentences:
        return None
    ids = torch.full(
        (attention_mask.shape[0], seq_len),
        int(model_wrapper.pad_token),
        dtype=torch.long,
        device=args.device,
    )
    for row_idx, sentence in enumerate(sentences[:attention_mask.shape[0]]):
        sent = [int(token_id) for token_id in sentence[:seq_len]]
        if sent:
            ids[row_idx, :len(sent)] = torch.tensor(sent, dtype=torch.long, device=args.device)
    ids[~attention_mask.bool()] = int(model_wrapper.pad_token)
    return ids


def build_dager_decoder_seed(args, model_wrapper, R_Qs, res_ids, orig_batch):
    if not model_wrapper.is_decoder():
        return None
    max_ids = -1
    for pos_ids in res_ids:
        if args.max_ids > 0 and len(pos_ids) > args.max_ids:
            max_ids = args.max_ids
            break
    predicted, scores, top_incorrect, top_incorrect_scores = filter_decoder(
        args,
        model_wrapper,
        R_Qs,
        res_ids,
        max_ids=max_ids,
    )
    if len(predicted) < orig_batch["input_ids"].shape[0]:
        predicted += top_incorrect
        scores += top_incorrect_scores
    if not predicted:
        logger.info("Autoregressive DAGER decoder seed unavailable: filter_decoder returned no candidates.")
        return None
    logger.info(
        "Autoregressive DAGER decoder seed | exact_or_l2=%s | fallback=%s | selected=%s",
        len(scores) - len(top_incorrect_scores),
        len(top_incorrect),
        min(len(predicted), orig_batch["input_ids"].shape[0]),
    )
    return _ids_from_sentences(
        args,
        model_wrapper,
        predicted,
        orig_batch["attention_mask"].to(args.device),
        orig_batch["input_ids"].shape[1],
    )


def rerank_autoregressive_batches(args, model_wrapper, true_grads, res_ids, orig_batch, true_labels, R_Qs=None):
    attention_mask = orig_batch["attention_mask"].to(args.device)
    seq_len = orig_batch["input_ids"].shape[1]
    support_scores, position_scores, wte_name, wpe_name = build_grad_token_scores(
        model_wrapper,
        true_grads,
        res_ids,
        seq_len,
    )
    logger.info(
        "Autoregressive DAGER scoring | WTE=%s | WPE=%s | support=%s | positions=%s",
        wte_name,
        wpe_name,
        len(support_scores),
        sum(1 for scores in position_scores if scores),
    )

    variants = max(1, int(args.ar_dager_rerank_batches))
    best_ids, best_loss, best_variant = None, None, None
    losses = []
    seed_ids = build_dager_decoder_seed(args, model_wrapper, R_Qs, res_ids, orig_batch) if R_Qs is not None else None
    if seed_ids is not None:
        with torch.enable_grad():
            seed_loss = compute_rec_loss_from_ids(
                args,
                model_wrapper,
                true_grads,
                seed_ids,
                attention_mask,
                true_labels,
            ).item()
        losses.append(("dager_seed", seed_loss))
        best_ids = seed_ids.detach().clone()
        best_loss = seed_loss
        best_variant = "dager_seed"

    for variant_idx in range(variants):
        candidate_ids = decode_autoregressive_batch(
            args,
            model_wrapper,
            res_ids,
            attention_mask,
            support_scores,
            position_scores,
            variant_idx,
        )
        with torch.enable_grad():
            loss = compute_rec_loss_from_ids(
                args,
                model_wrapper,
                true_grads,
                candidate_ids,
                attention_mask,
                true_labels,
            ).item()
        losses.append((variant_idx, loss))
        if best_loss is None or loss < best_loss:
            best_ids = candidate_ids.detach().clone()
            best_loss = loss
            best_variant = variant_idx
        cleanup_memory()

    logger.info(
        "Autoregressive full-gradient rerank | variants=%s | best=%s %.6f | losses=%s",
        variants,
        best_variant,
        best_loss,
        [(idx, round(float(loss), 6)) for idx, loss in losses],
    )
    refined_ids, refined_loss = refine_autoregressive_ids(
        args,
        model_wrapper,
        true_grads,
        res_ids,
        best_ids,
        attention_mask,
        true_labels,
        support_scores,
        position_scores,
        best_loss,
    )
    logger.info("Autoregressive DAGER selected loss %.6f -> %.6f after refinement.", best_loss, refined_loss)
    return refined_ids


def refine_autoregressive_ids(args, model_wrapper, true_grads, res_ids, ids, attention_mask, true_labels,
                              support_scores, position_scores, current_loss):
    max_trials = max(0, int(args.ar_dager_refine_trials))
    if max_trials <= 0:
        return ids, current_loss

    tokenizer = model_wrapper.tokenizer
    current = ids.detach().clone()
    current_loss = float(current_loss)
    accepted = 0
    trials = 0
    attention_cpu = attention_mask.detach().cpu().bool()
    current_cpu = current.detach().cpu()
    cells = []

    for pos in range(current.shape[1]):
        candidates = _candidate_ids_for_pos(args, model_wrapper, res_ids, pos)
        if len(candidates) <= 1:
            continue
        pos_counts = {}
        for row_idx in torch.where(attention_cpu[:, pos])[0].tolist():
            token_id = int(current_cpu[row_idx, pos].item())
            pos_counts[token_id] = pos_counts.get(token_id, 0) + 1
        for row_idx in torch.where(attention_cpu[:, pos])[0].tolist():
            current_token = int(current_cpu[row_idx, pos].item())
            prefix = current_cpu[row_idx, :pos].tolist()
            ranked = sorted(
                [token_id for token_id in candidates if int(token_id) != current_token],
                key=lambda token_id: _score_candidate(
                    args,
                    tokenizer,
                    int(token_id),
                    pos if pos < len(position_scores) else len(position_scores) - 1,
                    prefix,
                    support_scores,
                    position_scores,
                    pos_counts,
                    {},
                ),
                reverse=True,
            )
            if ranked:
                duplicated = pos_counts.get(current_token, 0) > 1
                cells.append((0 if duplicated else 1, pos, row_idx, ranked[:2]))

    cells.sort()
    for _, pos, row_idx, ranked in cells:
        if trials >= max_trials:
            break
        for candidate in ranked:
            if trials >= max_trials:
                break
            trial_ids = current.detach().clone()
            trial_ids[row_idx, pos] = int(candidate)
            with torch.enable_grad():
                trial_loss = compute_rec_loss_from_ids(
                    args,
                    model_wrapper,
                    true_grads,
                    trial_ids,
                    attention_mask,
                    true_labels,
                ).item()
            trials += 1
            if trial_loss + 1e-6 < current_loss:
                current = trial_ids
                current_cpu[row_idx, pos] = int(candidate)
                current_loss = float(trial_loss)
                accepted += 1
                break
            if trials % 16 == 0:
                cleanup_memory()

    logger.info(
        "Autoregressive DAGER discrete refinement | trials=%s | accepted=%s | final_loss=%.6f",
        trials,
        accepted,
        current_loss,
    )
    return current, current_loss


def reconstruct(args, sample, metric, model_wrapper):
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
            grad.data = grad.data + torch.randn(grad.shape, device=grad.device) * args.defense_noise

    reference = [
        remove_padding(tokenizer, orig_batch["input_ids"][i], left=(args.pad == "left"))
        for i in range(orig_batch["input_ids"].shape[0])
    ]

    with torch.no_grad():
        rank, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        if rank is None:
            return ["" for _ in reference], reference
        logger.info("Autoregressive DAGER rank: %s", rank)
        _, res_ids, _, _ = filter_l1(args, model_wrapper, R_Qs, max_positions=orig_batch["input_ids"].shape[1])

    if not res_ids:
        return ["" for _ in reference], reference
    logger.info("Autoregressive DAGER candidate counts: %s", [len(ids) for ids in res_ids[:16]])

    final_ids = rerank_autoregressive_batches(args, model_wrapper, true_grads, res_ids, orig_batch, true_labels, R_Qs=R_Qs)
    log_candidate_diagnostics(args, model_wrapper, orig_batch, final_ids, res_ids)
    prediction = [
        remove_padding(tokenizer, final_ids[i], left=(args.pad == "left"))
        for i in range(final_ids.shape[0])
    ]
    return prediction[:len(reference)], reference


def main():
    if args.task != "seq_class":
        raise NotImplementedError("Autoregressive DAGER currently supports --task seq_class.")
    print(f"Hash Value {job_hash} Started")
    is_complete, results_dir = is_attack_complete(ATTACK_NAME, job_hash)
    if not log_claim_acquired:
        logger.info("Skipping hash %s because another job currently owns the primary log file for this run.", job_hash)
        print(f"Hash Value {job_hash} Skipped (locked)")
        return
    if is_complete:
        logger.info("Results already exist for this config at %s; skipping attack.", results_dir)
        logger.info("Done with all.")
        print(f"Hash Value {job_hash} is already done")
        return

    metric = load_rouge_metric(cache_dir=args.cache_dir, logger=logger)
    dataset = TextDataset(
        torch.device(args.device),
        args.dataset,
        args.split,
        args.n_inputs,
        args.batch_size,
        args.cache_dir,
        use_hf_split=args.use_hf_split,
        preprocess_numbered_markers=args.preprocess_numbered_markers,
        preprocess_boundary_markers=args.preprocess_boundary_markers,
        preprocess_unique_canary_markers=args.preprocess_unique_canary_markers,
        canary_marker_prefix=args.canary_marker_prefix,
    )
    effective_n_inputs = len(dataset)
    if effective_n_inputs != args.n_inputs:
        logger.warning(
            "Dataset effective_n_inputs=%s differs from requested n_inputs=%s; iterating over effective dataset length.",
            effective_n_inputs,
            args.n_inputs,
        )
    model_wrapper = ModelWrapper(args)
    tokenizer = model_wrapper.tokenizer

    predictions, references = [], []
    final_results = []
    final_per_input_results = []
    input_times = []
    sentence_rows = []
    input_rows = []
    results_dir = get_results_dir(ATTACK_NAME, job_hash)
    os.makedirs(results_dir, exist_ok=True)
    partial_state = load_partial_attack_state(results_dir)
    if partial_state["completed_inputs"]:
        sentence_rows = partial_state["sentence_rows"]
        input_rows = partial_state["input_rows"]
        predictions = partial_state["predictions"]
        references = partial_state["references"]
        final_results = partial_state["final_results"]
        final_per_input_results = partial_state["final_per_input_results"]
        input_times = partial_state["input_times"]
        logger.info(
            "Resuming from partial results in %s | completed_inputs=%s | last_input=%s",
            results_dir,
            len(partial_state["completed_inputs"]),
            max(partial_state["completed_inputs"]),
        )

    t_start = time.time()
    for i in range(args.start_input, min(effective_n_inputs, args.end_input)):
        if i in partial_state["completed_inputs"]:
            logger.info("Skipping already completed input #%s.", i)
            continue
        t_input_start = time.time()
        sample = dataset[i]
        logger.info("Running input #%s of %s.", i, effective_n_inputs)

        prediction, reference = reconstruct(args, sample, metric, model_wrapper)
        predictions += prediction
        references += reference

        curr_metrics = []
        for sent_idx, (ref, pred) in enumerate(zip(reference, prediction)):
            logger.info("========================")
            logger.info("Reference: %s", _log_text(ref))
            logger.info("Prediction: %s", _log_text(pred))
            metrics = evaluate_prediction(pred, ref, tokenizer, metric)
            metrics = maybe_add_canary_audit_metrics(
                metrics,
                pred,
                ref,
                tokenizer,
                metric,
                enabled=args.preprocess_unique_canary_markers,
                canary_prefix=args.canary_marker_prefix,
            )
            curr_metrics.append(metrics)
            sentence_rows.append({
                "attack": ATTACK_NAME,
                "model": args.model_path,
                "dataset": args.dataset,
                "input_index": i,
                "sentence_index": sent_idx,
                "reference": ref,
                "prediction": pred,
                **metrics,
            })
        logger.info("========================")
        summary = summarize_metrics(curr_metrics)
        input_canary_means = extract_canary_metric_means(summary)
        final_results.extend(curr_metrics)
        logger.info("[Curr input metrics]:")
        logger.info("%s", print_summary_table(summary))
        logger.info("[Aggregate metrics]:")
        aggregated_results = _safe_aggregated_metrics(
            prediction,
            reference,
            tokenizer,
            metric,
            curr_metrics,
            f"input #{i}",
        )
        aggregated_results.update(input_canary_means)
        final_per_input_results.append(aggregated_results)
        logger.info("%s", print_single_metric_dict(aggregated_results))

        input_time_sec = time.time() - t_input_start
        input_times.append(input_time_sec)
        logger.info(
            "input #%s time: %s | total time: %s",
            i,
            str(datetime.timedelta(seconds=input_time_sec)).split(".")[0],
            str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0],
        )
        input_rows.append({
            "attack": ATTACK_NAME,
            "model": args.model_path,
            "dataset": args.dataset,
            "input_index": i,
            "num_sentences": len(reference),
            "joined_reference": " ".join(reference),
            "joined_prediction": " ".join(prediction),
            "reconstruction_time_sec": input_time_sec,
            **aggregated_results,
        })

        partial_summary = {
            "Arguments": args_to_dict(args),
            "Attack": ATTACK_NAME,
            "Job Hash": job_hash,
            "Overall Results": _safe_aggregated_metrics(
                predictions,
                references,
                tokenizer,
                metric,
                final_results,
                f"partial input #{i}",
            ),
            "Per Input Results": summarize_metrics(final_per_input_results),
            "Per Sentence Results": summarize_metrics(final_results),
        }
        write_attack_artifacts(results_dir, sentence_rows, input_rows, partial_summary, status="incomplete")
        cleanup_memory()

    aggregated_results = _safe_aggregated_metrics(
        predictions,
        references,
        tokenizer,
        metric,
        final_per_input_results,
        "full experiment",
    )
    summary = summarize_metrics(final_results)
    aggregated_results.update(extract_canary_metric_means(summary))
    if input_times:
        summary["reconstruction_time_mean"] = float(np.mean(input_times))
        summary["reconstruction_time_std"] = float(np.std(input_times))
    summary_per_input = summarize_metrics(final_per_input_results)
    if input_times:
        summary_per_input["reconstruction_time_mean"] = float(np.mean(input_times))
        summary_per_input["reconstruction_time_std"] = float(np.std(input_times))
    summary_results = {
        "Overall Results": aggregated_results,
        "Per Sentence Results": summary,
        "Per Input Results": summary_per_input,
        "Arguments": args_to_dict(args)}
    canary_summary = extract_canary_metric_summary(summary)
    if canary_summary:
        summary_results["Canary Audit Results"] = canary_summary
    write_attack_artifacts(results_dir, sentence_rows, input_rows, summary_results, status="complete")
    logger.info("Done with all.")
    print(f"Hash Value {job_hash} Done")


if __name__ == "__main__":
    try:
        main()
    finally:
        release_all_log_locks()
