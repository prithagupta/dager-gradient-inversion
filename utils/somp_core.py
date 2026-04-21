import logging
from collections import Counter

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from utils.experiment import cleanup_memory
from utils.functional import check_if_in_span
from utils.functional import get_top_B_in_span
from utils.functional import remove_padding

logger = logging.getLogger(__name__)


def ensure_somp_args(args):
    defaults = {
        "headwise_factorization": True,
        "target_pool": 1600,
        "k_per_head_max": 4096,
        "wte_chunk": 4096,
        "sparse_q": 0.25,
        "frac_active_heads": 0.5,
        "lambda_sub": 0.8,
        "lambda_cons": 0.5,
        "lambda_sparse": 0.5,
        "booster_front_positions": 5,
        "booster_topm": 800,
        "always_include_eos_period": True,
        "pos_topk": 256,
        "beam_width": 16,
        "beam_groups": 4,
        "beam_max_steps": None,
        "diversity_lambda": 0.35,
        "ngram_diversity": 2,
        "ngram_lambda": 0.25,
        "beta_glm": 0.33,
        "cluster_rouge_l": 0.7,
        "length_bonus_gamma": 0.2,
        "max_omp_candidates": 128,
        "somp_add_special_tokens": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _to_cpu_grads(grads):
    return [None if grad is None else grad.detach().to("cpu", copy=True).float() for grad in grads]


def _dot_sum(grads_a, grads_b):
    value = 0.0
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is not None and grad_b is not None:
            value += float((grad_a * grad_b).sum().item())
    return value


def _pair_dot(grads_a, grads_b):
    total = None
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        value = (grad_a * grad_b).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def _solve_least_squares(component_grads, mixed_grads, ridge=1e-6):
    n_components = len(component_grads)
    if n_components == 0:
        return torch.empty(0)

    gram = torch.zeros((n_components, n_components), dtype=torch.float32)
    rhs = torch.zeros((n_components,), dtype=torch.float32)
    for i in range(n_components):
        rhs[i] = _pair_dot(mixed_grads, component_grads[i]).cpu()
        for j in range(i, n_components):
            dot = _pair_dot(component_grads[i], component_grads[j]).cpu()
            gram[i, j] = dot
            gram[j, i] = dot

    gram[range(n_components), range(n_components)] += ridge
    try:
        return torch.linalg.solve(gram, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(gram, rhs).solution


def _rebuild_residual(mixed_grads, component_grads, alpha):
    residual = [None if grad is None else grad.clone() for grad in mixed_grads]
    for component_idx, coeff in enumerate(alpha.tolist()):
        comp = component_grads[component_idx]
        for grad_idx in range(len(residual)):
            if residual[grad_idx] is not None and comp[grad_idx] is not None:
                residual[grad_idx].data.add_(comp[grad_idx].data, alpha=-float(coeff))
    return residual


def _get_label_for_sample(true_labels, sample_idx):
    if isinstance(true_labels, torch.Tensor):
        if true_labels.ndim == 0:
            return true_labels.clone()
        if true_labels.size(0) == 1 and true_labels.ndim > 1:
            return true_labels[0, sample_idx].clone()
        if true_labels.size(0) > sample_idx:
            return true_labels[sample_idx].clone()
        return true_labels.view(-1)[0].clone()
    try:
        return torch.tensor(true_labels[sample_idx])
    except Exception:
        return torch.tensor(0)


def create_candidate_labels(model_wrapper, true_labels, sample_idx, input_tensor):
    device = input_tensor.device
    if model_wrapper.args.task == "seq_class":
        label = _get_label_for_sample(true_labels, sample_idx)
        if label.ndim > 0:
            label = label.view(-1)[0]
        return label.long().view(1).to(device)

    labels = input_tensor.clone().to(device)
    if hasattr(model_wrapper, "pad_token") and model_wrapper.pad_token is not None:
        labels = torch.where(input_tensor == model_wrapper.pad_token, -100, labels)
    return labels


def to_text_list(items, tokenizer):
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif hasattr(item, "tolist"):
            out.append(tokenizer.decode(item.tolist(), skip_special_tokens=True))
        elif isinstance(item, (list, tuple)):
            out.append(tokenizer.decode(list(item), skip_special_tokens=True))
        else:
            out.append(str(item))
    return out


def _rouge_l_value(metric, pred, ref):
    score = metric.compute(predictions=[pred], references=[ref])["rougeL"]
    if hasattr(score, "mid"):
        return float(score.mid.fmeasure)
    return float(score)


def _candidate_quality(ids, tokenizer):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    end_bonus = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.0
    unigram_repeat = 0.0
    if len(ids) >= 2:
        unigram_repeat = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1]) / max(1, len(ids) - 1)
    bigram_repeat = 0.0
    if len(ids) >= 3:
        counts = Counter(zip(ids[:-1], ids[1:]))
        bigram_repeat = sum(count - 1 for count in counts.values() if count > 1) / max(1, len(ids) - 1)
    length_penalty = 0.0 if len(ids) >= 4 else 0.2
    return end_bonus - 1.5 * unigram_repeat - bigram_repeat - length_penalty


@torch.no_grad()
def build_global_candidate_pool(args, model_wrapper, residual_grads, R_Qs, head_R_Qs, eff_len_each):
    if head_R_Qs is None:
        raise RuntimeError("SOMP requires head-wise query subspaces. Use SOMPModelWrapper.")

    tokenizer = model_wrapper.tokenizer
    word_embeddings = model_wrapper._WTE_CPU
    grad_l1 = residual_grads[model_wrapper.layer_ids[0]]
    d_model = getattr(model_wrapper.model.config, "hidden_size", model_wrapper.emb_size)
    n_heads = getattr(model_wrapper.model.config, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(model_wrapper.model.config, "n_head")
    d_head = d_model // n_heads
    grad_l1_query = grad_l1[:, :d_model].to(args.device).float()
    grad_slices = [grad_l1_query[:, i * d_head:(i + 1) * d_head] for i in range(n_heads)]

    head_dists = []
    head_sparse = []
    vocab_size = word_embeddings.size(0)
    chunk_size = int(args.wte_chunk)
    for R_Q_head, grad_slice in zip(head_R_Qs, grad_slices):
        dists = []
        sparse_scores = []
        R_Q_head = R_Q_head.to(args.device)
        grad_slice = grad_slice.to(args.device)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            emb_chunk = word_embeddings[start:end].to(args.device)
            projected = emb_chunk @ grad_slice
            dist = check_if_in_span(R_Q_head, projected, args.dist_norm)
            abs_projected = projected.abs()
            threshold = torch.quantile(abs_projected, q=float(args.sparse_q), dim=1, keepdim=True)
            sparse = (abs_projected <= threshold).float().mean(dim=1)
            dists.append(dist.cpu())
            sparse_scores.append(sparse.cpu())
        head_dists.append(torch.cat(dists).unsqueeze(0))
        head_sparse.append(torch.cat(sparse_scores).unsqueeze(0))

    score_by_head = torch.cat(head_dists, dim=0)
    sparse_by_head = torch.cat(head_sparse, dim=0)
    n_active = max(1, int(float(args.frac_active_heads) * score_by_head.size(0)))
    active_heads = torch.topk(score_by_head.mean(dim=1), k=n_active, largest=False).indices

    subspace_score = score_by_head[active_heads].mean(dim=0)
    consistency_score = score_by_head[active_heads].std(dim=0)
    sparse_score = sparse_by_head[active_heads].mean(dim=0)
    fused_score = (
        float(args.lambda_sub) * subspace_score
        + float(args.lambda_cons) * consistency_score
        - float(args.lambda_sparse) * sparse_score
    )

    k = min(int(args.k_per_head_max), fused_score.numel())
    candidate_pool = torch.topk(fused_score, k=k, largest=False).indices
    if bool(args.always_include_eos_period):
        extras = []
        period_ids = tokenizer.encode(".", add_special_tokens=False)
        extras.extend(period_ids)
        if tokenizer.eos_token_id is not None:
            extras.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            extras.append(tokenizer.pad_token_id)
        if extras:
            candidate_pool = torch.unique(torch.cat([candidate_pool, torch.tensor(extras, dtype=torch.long)]))

    max_eff_len = int(eff_len_each.max().item())
    boost_positions = list(range(min(int(args.booster_front_positions), max_eff_len)))
    if max_eff_len > 0 and max_eff_len - 1 not in boost_positions:
        boost_positions.append(max_eff_len - 1)

    boost_ids = []
    for pos in boost_positions:
        embeddings = model_wrapper.get_embeddings(pos)[0]
        dists = check_if_in_span(R_Qs[0], embeddings, args.dist_norm)
        topm = min(int(args.booster_topm), dists.numel())
        boost_ids.append(torch.topk(dists, k=topm, largest=False).indices.cpu())
    if boost_ids:
        candidate_pool = torch.unique(torch.cat([candidate_pool, *boost_ids]))

    target_pool = min(int(args.target_pool), candidate_pool.numel())
    if candidate_pool.numel() > target_pool:
        candidate_pool = candidate_pool[:target_pool]

    logger.info("SOMP global candidate pool size: %d", candidate_pool.numel())
    return candidate_pool.to(args.device)


@torch.no_grad()
def position_filter_per_sample(args, model_wrapper, R_Qs, candidate_pool_indices, target_length):
    res_ids = []
    res_types = []
    sentence_ends = []
    eos_id = model_wrapper.tokenizer.eos_token_id

    for pos in range(int(target_length)):
        full_emb_pos = model_wrapper.get_embeddings(pos)[0]
        candidate_embeds = full_emb_pos.index_select(0, candidate_pool_indices)
        topk = min(int(args.pos_topk), candidate_embeds.size(0))
        if model_wrapper.is_bert():
            _, top_idx, types = get_top_B_in_span(
                R_Qs[0],
                candidate_embeds,
                topk,
                args.l1_span_thresh,
                args.dist_norm,
            )
        else:
            top_idx, = get_top_B_in_span(
                R_Qs[0],
                candidate_embeds,
                topk,
                args.l1_span_thresh,
                args.dist_norm,
            )
            types = torch.zeros_like(top_idx)

        ids = candidate_pool_indices[top_idx].tolist()
        type_ids = types.tolist()
        if eos_id is not None and eos_id in ids:
            eos_index = ids.index(eos_id)
            sentence_ends.append((pos, type_ids[eos_index]))
            ids = ids[:eos_index]
            type_ids = type_ids[:eos_index]
        if args.max_ids > 0:
            ids = ids[:args.max_ids]
            type_ids = type_ids[:args.max_ids]
        if not ids:
            logger.info("SOMP position %d has no candidates after filtering.", pos)
            break
        res_ids.append([int(token_id) for token_id in ids])
        res_types.append(type_ids)

    return res_ids, res_types, sentence_ends


@torch.no_grad()
def beam_search_decoder(args, model_wrapper, R_Qs, res_ids):
    R_Q2 = R_Qs[1]
    device = args.device
    max_steps = args.beam_max_steps
    res_iter = res_ids[:max_steps] if max_steps is not None else res_ids
    if len(res_iter) == 0:
        return []

    n_groups = max(1, int(args.beam_groups))
    beam_width = max(1, int(args.beam_width))
    per_group = max(1, beam_width // n_groups)
    beams_by_group = [[([], 0.0)] for _ in range(n_groups)]
    lm_embeddings = model_wrapper.get_input_embeddings_weight().detach().to(device)

    for pos, candidates in enumerate(res_iter):
        if not candidates:
            break
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        new_groups = []
        used_tokens = set()
        for group_idx, beams in enumerate(beams_by_group):
            if not beams:
                new_groups.append([])
                continue
            base = torch.tensor([beam[0] for beam in beams], dtype=torch.long, device=device)
            if pos == 0:
                group_candidates = candidate_tensor[group_idx:group_idx + 1] if group_idx < candidate_tensor.numel() else candidate_tensor[:1]
            else:
                group_candidates = candidate_tensor
            base_rep = base.repeat_interleave(group_candidates.numel(), dim=0)
            cand_rep = group_candidates.repeat(len(beams)).view(-1, 1)
            new_batch = torch.cat([base_rep, cand_rep], dim=1)
            attention_mask = torch.ones_like(new_batch)
            hidden = model_wrapper.get_layer_inputs(new_batch, attention_mask=attention_mask)[0]
            last_hidden = hidden[:, -1, :]
            scores = check_if_in_span(R_Q2, last_hidden, args.dist_norm)

            token_vec = lm_embeddings.index_select(0, cand_rep.view(-1))
            lm_score = (last_hidden * token_vec).sum(dim=1)
            lm_score = (lm_score - lm_score.mean()) / (lm_score.std(unbiased=False) + 1e-6)
            scores = scores - float(args.beta_glm) * lm_score

            prev_scores = torch.tensor([beam[1] for beam in beams], device=device).repeat_interleave(group_candidates.numel())
            scores = scores + prev_scores
            if used_tokens:
                used_tensor = torch.tensor(sorted(used_tokens), dtype=torch.long, device=device)
                scores = scores + float(args.diversity_lambda) * torch.isin(cand_rep.view(-1), used_tensor).float()

            top_idx = torch.topk(-scores, k=min(per_group, scores.numel()), largest=True).indices
            chosen = [(new_batch[idx].tolist(), float(scores[idx].item())) for idx in top_idx.tolist()]
            if chosen:
                used_tokens.add(chosen[0][0][-1])
            new_groups.append(chosen)
        beams_by_group = new_groups

    all_beams = [beam for group in beams_by_group for beam in group]
    all_beams.sort(key=lambda item: item[1] / max(1, len(item[0])))
    return [seq for seq, _ in all_beams[:beam_width]]


def _cluster_candidates(candidate_pool, tokenizer, metric, threshold):
    clusters = []
    logger = logging.getLogger('_cluster_candidates')

    for candidate in sorted(candidate_pool, key=len, reverse=True):
        candidate_text = tokenizer.decode(candidate, skip_special_tokens=True)
        found_cluster = False
        for cluster in clusters:
            ref_text = tokenizer.decode(cluster[0], skip_special_tokens=True)
            if _rouge_l_value(metric, candidate_text, ref_text) > threshold:
                cluster.append(candidate)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([candidate])

    representatives = []
    for cluster in clusters:
        representatives.append(max(cluster, key=lambda ids: _candidate_quality(ids, tokenizer)))
    representatives = sorted(representatives, key=lambda ids: _candidate_quality(ids, tokenizer), reverse=True)
    logger.info(f"SOMP sorted clustered candidates: {len(representatives)}", )
    return representatives


def reconstruct_with_omp(args, sample, metric, model_wrapper):
    ensure_somp_args(args)
    tokenizer = model_wrapper.tokenizer
    sequences, true_labels = sample
    batch_size = len(sequences)

    orig_batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
        add_special_tokens=bool(args.somp_add_special_tokens),
    ).to(args.device)

    raw_mixed_grads = model_wrapper.compute_grads(orig_batch, true_labels)
    mixed_grads = _to_cpu_grads(raw_mixed_grads)
    del raw_mixed_grads
    cleanup_memory()
    rank, R_Qs, head_R_Qs = model_wrapper.get_matrices_expansions(mixed_grads, B=None, tol=args.rank_tol)
    logger.info("SOMP rank: %s", rank)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model_wrapper.pad_token
    if "attention_mask" in orig_batch:
        eff_len_each = orig_batch["attention_mask"].sum(dim=1).cpu()
    else:
        eff_len_each = (orig_batch["input_ids"] != pad_id).sum(dim=1).cpu()

    candidate_pool_indices = build_global_candidate_pool(
        args,
        model_wrapper,
        mixed_grads,
        R_Qs,
        head_R_Qs,
        eff_len_each,
    )

    candidate_pool = []
    unique_texts = set()
    for length in sorted({int(length.item()) for length in eff_len_each}):
        res_ids, _, _ = position_filter_per_sample(args, model_wrapper, R_Qs, candidate_pool_indices, length)
        sequences_for_length = beam_search_decoder(args, model_wrapper, R_Qs, res_ids)
        for candidate in sequences_for_length:
            candidate = candidate[:length]
            text = tokenizer.decode(candidate, skip_special_tokens=True)
            if text and text not in unique_texts:
                unique_texts.add(text)
                candidate_pool.append(candidate)

    if not candidate_pool:
        references = [
            remove_padding(tokenizer, ids, left=(args.pad == "left"))
            for ids in orig_batch["input_ids"]
        ]
        del mixed_grads, orig_batch, candidate_pool_indices
        cleanup_memory()
        return ["" for _ in references], references
    logger.info("=== Candidate Pool ===")
    for idx, seq in enumerate(candidate_pool):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        logger.info(f"{idx} len={len(seq)} | {text}")
    candidate_pool = _cluster_candidates(candidate_pool, tokenizer, metric, float(args.cluster_rouge_l))
    logger.info(f"SOMP clustered candidates:", )
    for idx, seq in enumerate(candidate_pool):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        logger.info(f"{idx} len={len(seq)} | {text}")
    candidate_pool = candidate_pool[: int(args.max_omp_candidates)]
    logger.info("SOMP candidate representatives: %d", len(candidate_pool))

    candidate_components = []
    labels_to_try = range(batch_size) if args.task == "seq_class" else [0]
    for candidate in candidate_pool:
        token_tensor = torch.tensor(candidate, dtype=torch.long, device=args.device).unsqueeze(0)
        candidate_batch = BatchEncoding(
            {"input_ids": token_tensor, "attention_mask": torch.ones_like(token_tensor)}
        )
        for label_idx in labels_to_try:
            labels = create_candidate_labels(model_wrapper, true_labels, label_idx, token_tensor)
            grads = model_wrapper.compute_grads(candidate_batch.__class__(candidate_batch), labels)
            cpu_grads = _to_cpu_grads(grads)
            candidate_components.append(
                {
                    "ids": candidate,
                    "label_idx": label_idx,
                    "grads": cpu_grads,
                }
            )
            del grads, cpu_grads, labels
        del candidate_batch, token_tensor
        if len(candidate_components) % 8 == 0:
            cleanup_memory()

    residual = [None if grad is None else grad.clone() for grad in mixed_grads]
    selected = []
    for step in range(batch_size):
        best_score = -float("inf")
        best_idx = -1
        for idx, component in enumerate(candidate_components):
            if idx in selected:
                continue
            numerator = _dot_sum(residual, component["grads"])
            denominator = max(_dot_sum(component["grads"], component["grads"]), 1e-12) ** 0.5
            length_bonus = max(1, len(component["ids"])) ** float(args.length_bonus_gamma)
            score = abs(numerator / denominator) * length_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        active_grads = [candidate_components[idx]["grads"] for idx in selected]
        alpha = _solve_least_squares(active_grads, mixed_grads)
        residual = _rebuild_residual(mixed_grads, active_grads, alpha)
        logger.info(
            "SOMP OMP step %d/%d selected candidate %d score %.6f: %s",
            step + 1,
            batch_size,
            best_idx,
            best_score,
            tokenizer.decode(candidate_components[best_idx]["ids"], skip_special_tokens=True),
        )

    prediction_ids = [candidate_components[idx]["ids"] for idx in selected]
    while len(prediction_ids) < batch_size:
        prediction_ids.append([])

    predictions = to_text_list(prediction_ids[:batch_size], tokenizer)
    references = [
        remove_padding(tokenizer, ids, left=(args.pad == "left"))
        for ids in orig_batch["input_ids"]
    ]
    del candidate_components, residual, mixed_grads, orig_batch
    cleanup_memory()
    return predictions, references
