import logging

import numpy as np
import torch
import torch.nn.functional as F

from constants import config

logger = logging.getLogger(__name__)


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


def get_layer_decomp(grad, B=None, tol=None, upcast=False):
    grad = torch.nan_to_num(grad, nan=0.0, posinf=1e5, neginf=-1e5)
    grad = grad.detach().float().cpu().numpy()
    if upcast:
        grad = grad.astype(np.float32)
    if B == None:
        if upcast:
            B = np.linalg.matrix_rank(grad.astype(np.float32), tol=tol)
            grad = grad.float()
        else:
            B = np.linalg.matrix_rank(grad, tol=tol)
    U, S, Vh = torch.svd_lowrank(torch.tensor(grad), q=B, niter=10)
    if upcast:
        R = Vh.T.half()
    else:
        R = Vh.T
    return B, torch.Tensor(R).detach()


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
    v /= v.pow(2).sum(-1, keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v)
    out_of_span = proj - v
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)

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
    d = torch.log(torch.cat(dists, axis=1)) - torch.log(1 - torch.cat(dists, axis=1))
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
