import json
import logging
import numpy as np
import re
import torch
import torch.nn.functional as F

from constants import DEFAULT_CANARY_MARKER_PREFIX, config

logger = logging.getLogger(__name__)


def _safe_serialize(obj):
    """Recursively convert non-JSON types."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    elif hasattr(obj, "item"):  # torch / numpy scalar
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy / torch array
        return obj.tolist()
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)


def _rouge_triplet(score):
    if hasattr(score, 'mid'):
        score = score.mid
    if hasattr(score, 'fmeasure'):
        return score.fmeasure, score.precision, score.recall
    value = float(score)
    return value, value, value


def print_single_metric_dict(metrics):
    output = "\n===== Single Evaluation =====\n"
    output += f"{'Metric':<25} {'Value':>10}\n"
    output += "-" * 40 + "\n"
    for k, v in metrics.items():
        output += f"{k:<25} {float(v):>10.4f}\n"
    return output


def print_summary_table(summary):
    output = "\n===== Metrics Summary =====\n"
    output += f"{'Metric':<25} {'Mean':>10} {'Std':>10}\n"
    output += "-" * 50 + "\n"

    # group mean/std pairs
    metrics = sorted(set(k.replace("_mean", "").replace("_std", "") for k in summary))

    for m in metrics:
        mean = summary.get(f"{m}_mean", 0.0)
        std = summary.get(f"{m}_std", 0.0)
        output += f"{m:<25} {mean:>10.4f} {std:>10.4f}\n"

    return output


def summarize_metrics(metrics_list):
    summary = {}
    keys = metrics_list[0].keys()

    for k in keys:
        values = np.array([m[k] for m in metrics_list], dtype=float)
        summary[f"{k}_mean"] = float(values.mean())
        summary[f"{k}_std"] = float(values.std())
    return summary


def evaluate_prediction(pred, ref, tokenizer, rouge_metric):
    rouge = rouge_metric.compute(predictions=[pred], references=[ref])

    rouge1_fm, _, _ = _rouge_triplet(rouge["rouge1"])
    rouge2_fm, _, _ = _rouge_triplet(rouge["rouge2"])
    rougeL_fm, _, _ = _rouge_triplet(rouge["rougeL"])

    exact_match = int(pred.strip() == ref.strip())

    pred_ids = tokenizer(pred, add_special_tokens=False)["input_ids"]
    ref_ids = tokenizer(ref, add_special_tokens=False)["input_ids"]

    L = min(len(pred_ids), len(ref_ids))
    token_acc = 0.0 if L == 0 else sum(int(pred_ids[i] == ref_ids[i]) for i in range(L)) / max(len(ref_ids), 1)

    pad_id = tokenizer.pad_token_id
    mask = [ref_ids[i] != pad_id for i in range(len(ref_ids))]
    correct = sum(int(pred_ids[i] == ref_ids[i]) for i in range(L) if mask[i])
    total = sum(mask)
    padded_token_acc = 0.0 if total == 0 else correct / total
    results_dict = {"rouge1_fm": rouge1_fm, "rouge2_fm": rouge2_fm, "rougeL_fm": rougeL_fm,
                    "exact_match": exact_match, "token_acc": token_acc, "padded_token_acc": padded_token_acc,
                    "pred_len": len(pred_ids), "ref_len": len(ref_ids)}
    return results_dict


def _canary_pattern(canary_prefix=DEFAULT_CANARY_MARKER_PREFIX):
    return re.compile(rf"\b{re.escape(canary_prefix)}\d{{6}}\b")


def _extract_canary_markers(text, canary_prefix=DEFAULT_CANARY_MARKER_PREFIX):
    return _canary_pattern(canary_prefix).findall(text)


def _strip_canary_markers(text, canary_prefix=DEFAULT_CANARY_MARKER_PREFIX):
    stripped = _canary_pattern(canary_prefix).sub(' ', text)
    return ' '.join(stripped.split())


def maybe_add_canary_audit_metrics(metrics, pred, ref, tokenizer, rouge_metric, enabled=False,
                                   canary_prefix=DEFAULT_CANARY_MARKER_PREFIX):
    if not enabled:
        return metrics

    ref_markers = _extract_canary_markers(ref, canary_prefix)
    if not ref_markers:
        return metrics

    pred_markers = _extract_canary_markers(pred, canary_prefix)
    expected_marker = ref_markers[0]
    matched_markers = [marker for marker in pred_markers if marker == expected_marker]
    foreign_markers = [marker for marker in pred_markers if marker != expected_marker]

    ref_marker_text = ' '.join(ref_markers)
    pred_marker_text = ' '.join(matched_markers)
    marker_rouge = rouge_metric.compute(predictions=[pred_marker_text], references=[ref_marker_text])

    non_canary_pred = _strip_canary_markers(pred, canary_prefix)
    non_canary_ref = _strip_canary_markers(ref, canary_prefix)
    non_canary_metrics = evaluate_prediction(non_canary_pred, non_canary_ref, tokenizer, rouge_metric)

    audit_metrics = {
        'canary_exact_match': int(pred_markers == ref_markers),
        'canary_recovered_any': int(len(matched_markers) > 0),
        'canary_token_recall': 0.0 if len(ref_markers) == 0 else len(matched_markers) / len(ref_markers),
        'canary_token_precision': 0.0 if len(pred_markers) == 0 else len(matched_markers) / len(pred_markers),
        'canary_rouge1_fm': _rouge_triplet(marker_rouge['rouge1'])[0],
        'canary_rouge2_fm': _rouge_triplet(marker_rouge['rouge2'])[0],
        'canary_rougeL_fm': _rouge_triplet(marker_rouge['rougeL'])[0],
        'foreign_canary_mentions': float(len(foreign_markers)),
        'pred_canary_count': float(len(pred_markers)),
        'ref_canary_count': float(len(ref_markers)),
        'non_canary_rouge1_fm': non_canary_metrics['rouge1_fm'],
        'non_canary_rouge2_fm': non_canary_metrics['rouge2_fm'],
        'non_canary_rougeL_fm': non_canary_metrics['rougeL_fm'],
        'non_canary_exact_match': non_canary_metrics['exact_match'],
        'non_canary_token_acc': non_canary_metrics['token_acc'],
        'non_canary_padded_token_acc': non_canary_metrics['padded_token_acc'],
        'non_canary_pred_len': non_canary_metrics['pred_len'],
        'non_canary_ref_len': non_canary_metrics['ref_len'],
    }
    merged = dict(metrics)
    merged.update(audit_metrics)
    return merged


def extract_canary_metric_means(summary):
    canary_metrics = {}
    for key, value in summary.items():
        if not key.endswith('_mean'):
            continue
        base_key = key[:-5]
        if (base_key.startswith('canary_') or base_key.startswith('non_canary_') or
                base_key in {'foreign_canary_mentions', 'pred_canary_count', 'ref_canary_count'}):
            canary_metrics[base_key] = float(value)
    return canary_metrics


def extract_canary_metric_summary(summary):
    return {
        key: float(value)
        for key, value in summary.items()
        if key.startswith('canary_') or key.startswith('non_canary_') or
        key.startswith('foreign_canary_mentions') or key.startswith('pred_canary_count') or
        key.startswith('ref_canary_count')
    }


def _fallback_metric_means(metrics_list):
    if not metrics_list:
        return {
            "rouge1_fm": float("nan"),
            "rouge2_fm": float("nan"),
            "rougeL_fm": float("nan"),
            "exact_match": float("nan"),
            "token_acc": float("nan"),
            "padded_token_acc": float("nan"),
            "pred_len": float("nan"),
            "ref_len": float("nan"),
        }
    summary = summarize_metrics(metrics_list)
    return {k[:-5]: float(v) for k, v in summary.items() if k.endswith("_mean")}


def _is_oom_like(exc):
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or 'out of memory' in msg or 'oom' in msg


def _clear_eval_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _chunk_pairs_by_char_budget(predictions, references, char_budget):
    chunks = []
    cur_preds = []
    cur_refs = []
    cur_chars = 0
    for pred, ref in zip(predictions, references):
        pair_chars = len(pred) + len(ref) + 2
        if cur_preds and cur_chars + pair_chars > char_budget:
            chunks.append((cur_preds, cur_refs))
            cur_preds = []
            cur_refs = []
            cur_chars = 0
        cur_preds.append(pred)
        cur_refs.append(ref)
        cur_chars += pair_chars
    if cur_preds:
        chunks.append((cur_preds, cur_refs))
    return chunks


def _merge_chunk_metrics(chunk_metrics):
    if not chunk_metrics:
        return _fallback_metric_means([])

    total_ref_len = sum(max(float(m.get('ref_len', 0.0)), 1.0) for m in chunk_metrics)
    merged = {
        'rouge1_fm': 0.0,
        'rouge2_fm': 0.0,
        'rougeL_fm': 0.0,
        'token_acc': 0.0,
        'padded_token_acc': 0.0,
        'pred_len': float(sum(float(m.get('pred_len', 0.0)) for m in chunk_metrics)),
        'ref_len': float(sum(float(m.get('ref_len', 0.0)) for m in chunk_metrics)),
        'exact_match': float(all(int(m.get('exact_match', 0)) == 1 for m in chunk_metrics)),
    }
    for metric_name in ['rouge1_fm', 'rouge2_fm', 'rougeL_fm', 'token_acc', 'padded_token_acc']:
        merged[metric_name] = float(sum(
            float(m.get(metric_name, 0.0)) * max(float(m.get('ref_len', 0.0)), 1.0) for m in chunk_metrics
        ) / max(total_ref_len, 1.0))
    return merged


def _safe_aggregated_metrics(predictions, references, tokenizer, metric, fallback_metrics, scope):
    try:
        return evaluate_prediction(" ".join(predictions), " ".join(references), tokenizer, metric)
    except (RuntimeError, MemoryError) as exc:
        if not _is_oom_like(exc):
            raise
        logger.warning(
            'Aggregate evaluation OOM for %s (%s). Retrying with chunked joined evaluation.',
            scope, exc,
        )
        _clear_eval_cache()

    max_pair_chars = max((len(p) + len(r) + 2) for p, r in zip(predictions, references)) if predictions else 1
    char_budget = max(16000, max_pair_chars)
    total_chars = sum(len(p) + len(r) + 2 for p, r in zip(predictions, references))
    while True:
        try:
            chunks = _chunk_pairs_by_char_budget(predictions, references, char_budget)
            chunk_metrics = [
                evaluate_prediction(" ".join(chunk_preds), " ".join(chunk_refs), tokenizer, metric)
                for chunk_preds, chunk_refs in chunks
            ]
            logger.warning(
                'Aggregate evaluation for %s succeeded with %s chunk(s) at char_budget=%s.',
                scope, len(chunks), char_budget,
            )
            return _merge_chunk_metrics(chunk_metrics)
        except (RuntimeError, MemoryError) as exc:
            if not _is_oom_like(exc):
                raise
            _clear_eval_cache()
            if char_budget <= max_pair_chars:
                break
            next_budget = max(max_pair_chars, char_budget // 2)
            if next_budget == char_budget:
                break
            logger.warning(
                'Chunked aggregate evaluation still OOM for %s at char_budget=%s; retrying with %s.',
                scope, char_budget, next_budget,
            )
            char_budget = next_budget

    logger.warning(
        'Aggregate evaluation failed for %s even after chunking. Falling back to mean of precomputed metrics.',
        scope,
    )
    return _fallback_metric_means(fallback_metrics)


def remove_padding(tokenizer, ids, left=False):
    if left:
        for i in range(ids.shape[0]):
            if ids[i].item() != config['PAD_TOKEN']:
                ids = ids[i:]
                break
    else:
        for i in range(ids.shape[0] - 1, -1, -1):
            if ids[i].item() != config['PAD_TOKEN']:
                ids = ids[:i + 1]
                break
    return tokenizer.decode(ids)


def grad_dist(grads1, grads2, args):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2) + 1e-9)
            elif args.loss == 'dlg':
                ret += (g1 - g2).square().sum()
            elif args.loss == 'tag':
                ret += (g1 - g2).square().sum() + args.tag_factor * torch.abs(g1 - g2).sum()
            else:
                assert False
            n_g += 1
    if args.loss == 'cos':
        ret /= n_g
    return ret


def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric='cos'):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == 'l2':
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == 'cos':
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]


def stable_matrix_rank(grad, tol=None):
    grad = torch.nan_to_num(
        grad.detach().cpu(),
        nan=0.0,
        posinf=1e5,
        neginf=-1e5,
    ).to(torch.float64)
    if grad.ndim != 2 or min(grad.shape) == 0:
        return 0
    singular_values = torch.linalg.svdvals(grad)
    if singular_values.numel() == 0:
        return 0
    if tol is None:
        tol = max(grad.shape) * torch.finfo(grad.dtype).eps * singular_values.max().item()
    return int((singular_values > float(tol)).sum().item())


def _orthonormalize_rows(rows):
    if rows.numel() == 0:
        return rows
    q, _ = torch.linalg.qr(rows.T.contiguous(), mode="reduced")
    return q.T.contiguous()


def get_layer_decomp(grad, B=None, tol=None, upcast=False):
    grad = torch.nan_to_num(
        grad.detach().cpu(),
        nan=0.0,
        posinf=1e5,
        neginf=-1e5,
    )
    work_dtype = torch.float32 if upcast else torch.float64
    grad = grad.to(work_dtype)
    if B is None:
        B = stable_matrix_rank(grad, tol=tol)
    B = min(max(int(B), 1), min(grad.shape))

    try:
        _, _, vh = torch.linalg.svd(grad, full_matrices=False)
        R = vh[:B]
    except RuntimeError:
        # Fallback for rare LAPACK/CUDA SVD failures; this path is approximate.
        _, _, v = torch.svd_lowrank(grad.float(), q=B, niter=10)
        R = v.T

    R = _orthonormalize_rows(R.float())
    if upcast:
        R = R.half()
    return B, R.detach()


def get_perplexity(gpt2, x_embeds, bert_embeddings_weight, gpt2_embeddings_weight, c=0.1):
    gpt2_embeddings_weight = gpt2_embeddings_weight.repeat(x_embeds.shape[0], 1, 1)

    # Get alphas on BERT embeddings --> transfer to GPT-2
    alpha, _ = get_closest_tokens(x_embeds, bert_embeddings_weight)
    # alpha = torch.cdist(x_embeds[:, :-1, :], bert_embeddings_weight, p=2)
    alpha = F.softmax(-alpha / c, dim=2)
    gpt2_embeds = alpha.bmm(gpt2_embeddings_weight)

    # Pass through GPT-2 and get average perplexity
    out_gpt2 = gpt2(inputs_embeds=gpt2_embeds)
    log_probs = out_gpt2.logits.log_softmax(dim=2)
    fuzzy_perplexity = -(log_probs[:, :-1, :] * alpha[:, 1:, :]).sum(dim=2).mean(dim=1).sum()
    return fuzzy_perplexity


def check_if_in_span(R_K_norm, v, norm='l2'):
    work_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
    basis = R_K_norm.to(device=v.device, dtype=work_dtype)
    v_norm = F.normalize(v.to(work_dtype), dim=-1, eps=1e-12)
    basis = F.normalize(basis, dim=-1, eps=1e-12)
    proj = torch.matmul(torch.matmul(v_norm, basis.T), basis)
    out_of_span = proj - v_norm
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)
    else:
        raise ValueError(f"Unknown span distance norm: {norm}")

    return size


def filter_in_span(R_K_norm, v, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)
    bools = size < thresh
    return torch.where(bools)


def get_top_B_in_span(R_K_norm, v, B, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)
    bools = size < thresh
    which = torch.where(bools)
    _, idx = torch.sort(size[which])
    which_new = []
    for w in which:
        which_new.append(w[idx])
    which_new = tuple(which_new)
    return which_new


def log_distances(res_ids_new, R_Q, embeds, dist_norm, p, dists=None, log=None, label="L1"):
    """Log distance statistics for selected token candidates."""
    log = logger if log is None else log

    n_candidates = len(res_ids_new)
    if n_candidates == 0:
        log.info("%s Position %s | candidates=0", label, p)
        return

    if dists is None:
        dists = check_if_in_span(R_Q, embeds, dist_norm).reshape(-1)
    else:
        dists = torch.as_tensor(dists, device=embeds.device).reshape(-1)

    candidate_ids = torch.as_tensor(res_ids_new, dtype=torch.long, device=dists.device).reshape(-1)
    candidate_ids = candidate_ids[(candidate_ids >= 0) & (candidate_ids < dists.numel())]
    if candidate_ids.numel() == 0:
        log.info("%s Position %s | candidates=%s | no valid candidate ids", label, p, n_candidates)
        return

    selected_dists = dists[candidate_ids]
    log.info(
        "%s Position %s | candidates=%s | best_dist=%.6g | worst_dist=%.6g",
        label,
        p,
        n_candidates,
        selected_dists.min().item(),
        selected_dists.max().item(),
    )


def log_candidate_recall_debug(log, args, model_wrapper, orig_batch, res_ids, rec_l1, rec_l1_maxB, rec_l2):
    if not getattr(args, 'debug_candidates', False):
        return

    tokenizer = model_wrapper.tokenizer
    topk = max(0, int(getattr(args, 'debug_decode_topk', 3)))
    input_ids = orig_batch['input_ids']
    attention_mask = orig_batch.get('attention_mask', torch.where(input_ids != model_wrapper.pad_token, 1, 0))
    log.info(
        "Candidate debug recall summary | samples=%s | positions=%s | per_position_candidate_counts=%s",
        input_ids.shape[0],
        len(res_ids),
        [len(ids) for ids in res_ids],
    )

    for sample_idx in range(input_ids.shape[0]):
        active_positions = torch.where(attention_mask[sample_idx] != 0)[0].tolist()
        if not active_positions:
            active_positions = list(range(input_ids.shape[1]))

        missing_l1 = []
        missing_maxb = []
        for pos in active_positions:
            token_id = int(input_ids[sample_idx, pos].item())
            if token_id == model_wrapper.pad_token:
                continue
            if model_wrapper.is_decoder() and pos == active_positions[-1]:
                continue
            candidate_pos = 0 if model_wrapper.has_rope() else pos
            if candidate_pos >= len(res_ids):
                decoded = tokenizer.decode([token_id])
                missing_l1.append(f"{pos}:{token_id}:{decoded!r}")
                missing_maxb.append(f"{pos}:{token_id}:{decoded!r}")
                continue
            pos_ids = res_ids[candidate_pos]
            maxb_ids = pos_ids[:min(args.batch_size, len(pos_ids))]
            if token_id not in pos_ids:
                missing_l1.append(f"{pos}:{token_id}:{tokenizer.decode([token_id])!r}")
            if token_id not in maxb_ids:
                missing_maxb.append(f"{pos}:{token_id}:{tokenizer.decode([token_id])!r}")

        ref_text = remove_padding(tokenizer, input_ids[sample_idx], left=(getattr(args, 'pad', 'right') == 'left'))
        log.info(
            "Candidate debug sample %s | ref=%r | rec_l1=%s | rec_l1_maxB=%s | rec_l2=%s | missing_l1=%s | missing_maxB=%s",
            sample_idx,
            ref_text,
            rec_l1[sample_idx] if sample_idx < len(rec_l1) else None,
            rec_l1_maxB[sample_idx] if sample_idx < len(rec_l1_maxB) else None,
            rec_l2[sample_idx] if sample_idx < len(rec_l2) else None,
            missing_l1[:max(topk * 4, 8)],
            missing_maxb[:max(topk * 4, 8)],
        )


def filter_outliers(d, stage='token', std_thrs=None, maxB=None):
    if std_thrs is None:
        res_ids = torch.tensor(d.argsort()[:maxB])
        bools = torch.zeros_like(d).bool()
        bools[res_ids] = True
    elif maxB is None:
        logger.info(f'Wrong dists: {d.mean()} +- {d.std()}')
        d = (d - d.mean()) / d.std()
        bools = d < -std_thrs
        res_ids = torch.tensor(np.nonzero(bools)[:, 0])
    else:
        bools = torch.zeros_like(d).bool()
        bools[torch.tensor(d.argsort()[:maxB])] = True
        logger.info(f'Wrong dists: {d.mean()} +- {d.std()}')
        d = (d - d.mean()) / d.std()
        bools = bools & (d < -std_thrs)
        res_ids = torch.tensor(np.nonzero(bools)[:, 0])

    if stage == 'token':
        return res_ids
    else:
        return torch.tensor(d).unsqueeze(1), torch.tensor(bools)


def get_span_dists(args, model_wrapper, R_Qs, embeds, p=0, stage='token'):
    dists = []
    if stage == 'token':
        dists.append(check_if_in_span(R_Qs[0], embeds, args.dist_norm).T)
        sentences = torch.arange(embeds.shape[1]).unsqueeze(1).to(model_wrapper.args.device)
        embs = model_wrapper.get_layer_inputs(sentences, layers=args.n_layers - 1)

    else:
        embs = [e.to(model_wrapper.args.device) for e in embeds]

    if p == 0:
        for i in range(model_wrapper.args.n_layers - 1):
            dists.append(check_if_in_span(R_Qs[i + 1], embs[i], args.dist_norm))

    logger.info("dists %s", torch.cat(dists, axis=1).shape)
    span_dists = torch.cat(dists, axis=1).clamp(min=1e-12, max=1.0 - 1e-6)
    d = torch.log(span_dists) - torch.log1p(-span_dists)
    d = d.mean(axis=1).cpu().detach()

    return d


def fallback_decoder_l1_candidates(args, model_wrapper, R_Q, embeds, top_k, reason):
    dists = check_if_in_span(R_Q, embeds, args.dist_norm).reshape(-1)
    blocked_tokens = [model_wrapper.pad_token, model_wrapper.eos_token, model_wrapper.start_token]
    for token_id in blocked_tokens:
        if token_id is not None and 0 <= token_id < dists.numel():
            dists[token_id] = torch.inf

    if args.max_ids > 0:
        top_k = min(top_k, args.max_ids)
    top_k = min(top_k, dists.numel())
    res_ids = torch.topk(dists, k=top_k, largest=False).indices
    logger.info(
        "%s strict L1 returned no candidates; using nearest %s token candidates "
        "(best_dist=%s, worst_dist=%s).",
        reason,
        len(res_ids),
        dists[res_ids[0]].item(),
        dists[res_ids[-1]].item(),
    )
    return res_ids


def fallback_rope_l1_candidates(args, model_wrapper, R_Q, embeds):
    return fallback_decoder_l1_candidates(
        args,
        model_wrapper,
        R_Q,
        embeds,
        top_k=max(args.parallel, 200 * args.batch_size),
        reason="Llama/RoPE",
    )


def fallback_gpt2_l1_candidates(args, model_wrapper, R_Q, embeds):
    # GPT-2 decoding expands over many positions, so keep the fallback bounded.
    bounded_top_k = max(2 * args.batch_size, 64)
    bounded_top_k = min(bounded_top_k, 256)
    return fallback_decoder_l1_candidates(
        args,
        model_wrapper,
        R_Q,
        embeds,
        top_k=bounded_top_k,
        reason="GPT-2",
    )
