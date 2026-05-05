import datetime
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from args_factory import get_args
from utils.data import TextDataset
from utils.experiment import (args_to_dict, cleanup_memory, get_results_dir, is_attack_complete, load_rouge_metric,
                              load_partial_attack_state, write_attack_artifacts, setup_experiment_logging,
                              release_all_log_locks)
from utils.filtering_decoder import filter_decoder
from utils.functional import (fallback_gpt2_l1_candidates, fallback_rope_l1_candidates, get_top_B_in_span,
                              log_distances, remove_padding, filter_outliers, get_span_dists,
                              evaluate_prediction, print_single_metric_dict, summarize_metrics, print_summary_table,
                              _safe_aggregated_metrics, maybe_add_canary_audit_metrics,
                              extract_canary_metric_means, extract_canary_metric_summary)
from utils.models import ModelWrapper, _resolve_local_model_path

args = get_args()
logger, log_path, job_hash, log_claim_acquired = setup_experiment_logging(args, "hybrid_attack")
logger.info(f"Arguments {args}")
logger.info('\n\n\nCommand: %s\n\n\n', ' '.join(sys.argv))

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def filter_l1(args, model_wrapper, R_Qs, max_positions=None):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
    sentence_ends = []
    p = 0

    while True:
        if max_positions is not None and p >= max_positions:
            logger.info('Stopping hybrid L1 at tokenized batch length %s.', max_positions)
            break
        logger.info(f'L1 Position {p}')
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
                res_ids_new = filter_outliers(
                    distance_values,
                    std_thrs=std_thrs,
                    maxB=max(50 * model_wrapper.args.batch_size, int(0.05 * len(model_wrapper.tokenizer))),
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


def grad_match_loss(dummy_grads, true_grads, args):
    loss = None
    n_g = 0
    for dummy_grad, true_grad in zip(dummy_grads, true_grads):
        if dummy_grad is None or true_grad is None:
            continue
        true_grad = true_grad.to(dummy_grad.device)
        if args.grad_loss == 'cos':
            curr = 1.0 - (dummy_grad * true_grad).sum() / (
                    dummy_grad.reshape(-1).norm(p=2) * true_grad.reshape(-1).norm(p=2) + 1e-9
            )
        elif args.grad_loss == 'dlg':
            curr = (dummy_grad - true_grad).square().sum()
        elif args.grad_loss == 'tag':
            diff = dummy_grad - true_grad
            curr = diff.square().sum() + args.tag_factor * diff.abs().sum()
        else:
            raise ValueError(f'Unknown grad_loss: {args.grad_loss}')
        loss = curr if loss is None else loss + curr
        n_g += 1

    if loss is None:
        raise RuntimeError('No comparable gradients were produced for hybrid optimization.')
    if args.grad_loss == 'cos':
        loss = loss / max(n_g, 1)
    return loss


def build_candidate_mask(res_ids, seq_len, vocab_size, pad_token, device):
    candidate_mask = torch.zeros(seq_len, vocab_size, dtype=torch.bool, device=device)
    for pos in range(seq_len):
        if pos < len(res_ids) and len(res_ids[pos]) > 0:
            ids = torch.tensor(res_ids[pos], dtype=torch.long, device=device)
            candidate_mask[pos, ids] = True
        else:
            candidate_mask[pos, pad_token] = True
    return candidate_mask


def _with_pad_embeddings(x_embeds, pad_mask, pad_embed):
    if not bool(pad_mask.any().item()):
        return x_embeds
    return torch.where(pad_mask.unsqueeze(-1), pad_embed.view(1, 1, -1), x_embeds)


def _hybrid_uses_candidate_projection(args):
    return args.hybrid_projection_mode in ['candidate_final', 'candidate_periodic']


def _hybrid_uses_periodic_projection(args):
    return args.hybrid_projection_mode == 'candidate_periodic' and args.hybrid_project_every > 0


def _hybrid_uses_lm_prior(args):
    return args.hybrid_use_lm_prior == 'true' and args.coeff_perplexity > 0


def _hybrid_lm_dtype(args):
    if args.precision == 'half':
        return torch.float16
    if args.precision == 'double':
        return torch.float64
    return None


def _find_input_embedding_grad(model_wrapper, true_grads):
    emb_weight = model_wrapper.get_input_embeddings_weight()
    trainable_params = [
        (name, param)
        for name, param in model_wrapper.model.named_parameters()
        if param.requires_grad
    ]
    for idx, (name, param) in enumerate(trainable_params):
        same_object = param is emb_weight
        same_storage = param.shape == emb_weight.shape and param.data_ptr() == emb_weight.data_ptr()
        if (same_object or same_storage) and idx < len(true_grads):
            return true_grads[idx], name
    return None, None


def _find_position_embedding_grad(model_wrapper, true_grads):
    position_weight = None
    if hasattr(model_wrapper.model, 'transformer') and hasattr(model_wrapper.model.transformer, 'wpe'):
        position_weight = model_wrapper.model.transformer.wpe.weight
    elif hasattr(model_wrapper.model, 'bert') and hasattr(model_wrapper.model.bert, 'embeddings'):
        position_weight = model_wrapper.model.bert.embeddings.position_embeddings.weight
    if position_weight is None:
        return None, None

    trainable_params = [
        (name, param)
        for name, param in model_wrapper.model.named_parameters()
        if param.requires_grad
    ]
    for idx, (name, param) in enumerate(trainable_params):
        same_object = param is position_weight
        same_storage = param.shape == position_weight.shape and param.data_ptr() == position_weight.data_ptr()
        name_match = name.endswith('wpe.weight') or 'position_embeddings.weight' in name
        if (same_object or same_storage or name_match) and idx < len(true_grads):
            return true_grads[idx], name
    return None, None


def augment_res_ids_with_true_grad_support(args, model_wrapper, true_grads, res_ids, max_positions):
    grad_wte, wte_name = _find_input_embedding_grad(model_wrapper, true_grads)
    if grad_wte is None or grad_wte.ndim != 2:
        logger.info('Hybrid true-grad support unavailable: WTE grad not found.')
        return res_ids

    grad_wte = grad_wte.detach().float().cpu()
    row_norm = torch.nan_to_num(grad_wte.norm(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    max_norm = float(row_norm.max().item()) if row_norm.numel() else 0.0
    if max_norm <= 0:
        logger.info('Hybrid true-grad support unavailable: WTE row norms are all zero.')
        return res_ids

    eps = max(1e-12, max_norm * 1e-6)
    support_ids = torch.where(row_norm > eps)[0]
    nonzero_frac = support_ids.numel() / max(row_norm.numel(), 1)
    max_support = min(2048, row_norm.numel())
    if support_ids.numel() == 0 or support_ids.numel() > max_support or nonzero_frac > 0.20:
        support_ids = torch.topk(row_norm, k=max_support).indices
    support_ids = support_ids.long().cpu()

    grad_wpe, wpe_name = _find_position_embedding_grad(model_wrapper, true_grads)
    if grad_wpe is not None and grad_wpe.ndim == 2:
        grad_wpe = grad_wpe.detach().float().cpu()
        support_vecs = F.normalize(grad_wte.index_select(0, support_ids), dim=-1)
    else:
        grad_wpe = None
        support_vecs = None

    support_by_norm = support_ids[torch.argsort(row_norm.index_select(0, support_ids), descending=True)]
    global_keep = min(32, support_by_norm.numel())
    global_ids = support_by_norm[:global_keep].tolist()
    position_keep = min(128, support_ids.numel())
    cap = 256
    if args.max_ids > 0:
        cap = min(cap, args.max_ids)

    augmented = []
    n_positions = min(int(max_positions), len(res_ids))
    sizes_before, sizes_after = [], []
    for pos, ids in enumerate(res_ids):
        local = [int(token_id) for token_id in ids]
        sizes_before.append(len(local))
        additions = []
        if pos < n_positions and grad_wpe is not None and pos < grad_wpe.shape[0] and position_keep > 0:
            pos_grad = grad_wpe[pos]
            if pos_grad.norm(p=2) > 0:
                scores = support_vecs @ F.normalize(pos_grad.reshape(1, -1), dim=-1).reshape(-1)
                top_idx = torch.topk(scores, k=position_keep).indices
                additions.extend(int(token_id) for token_id in support_ids[top_idx].tolist())
        additions.extend(global_ids)

        seen = set()
        merged = []
        for token_id in local + additions:
            if token_id in seen:
                continue
            seen.add(token_id)
            merged.append(token_id)
            if len(merged) >= max(cap, len(local)):
                break
        augmented.append(merged)
        sizes_after.append(len(merged))

    logger.info(
        'Hybrid true-grad support augmented candidates | wte=%s | wpe=%s | support=%s | positions=%s | avg_size %.1f -> %.1f | cap=%s',
        wte_name,
        wpe_name,
        support_ids.numel(),
        len(augmented),
        float(np.mean(sizes_before)) if sizes_before else 0.0,
        float(np.mean(sizes_after)) if sizes_after else 0.0,
        cap,
    )
    return augmented


def log_length_signal(model_wrapper, true_grads, orig_batch):
    true_lengths = orig_batch['attention_mask'].detach().cpu().sum(dim=1).float()
    grad_wpe, wpe_name = _find_position_embedding_grad(model_wrapper, true_grads)
    if grad_wpe is None or grad_wpe.ndim != 2:
        logger.info(
            'Hybrid length signal | true_len mean=%.1f min=%s max=%s | WPE unavailable',
            float(true_lengths.mean().item()),
            int(true_lengths.min().item()),
            int(true_lengths.max().item()),
        )
        return
    wpe_norm = torch.nan_to_num(grad_wpe.detach().float().cpu().norm(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    max_pos = min(orig_batch['input_ids'].shape[1], wpe_norm.numel())
    if max_pos <= 0 or wpe_norm[:max_pos].max().item() <= 0:
        return
    active_est = wpe_norm[:max_pos] / wpe_norm[:max_pos].max().clamp_min(1e-12) * float(
        orig_batch['input_ids'].shape[0])
    logger.info(
        'Hybrid length signal | true_len mean=%.1f min=%s max=%s | WPE=%s active_est first=%s last=%s',
        float(true_lengths.mean().item()),
        int(true_lengths.min().item()),
        int(true_lengths.max().item()),
        wpe_name,
        [round(float(v), 2) for v in active_est[:min(8, max_pos)].tolist()],
        [round(float(v), 2) for v in active_est[max(0, max_pos - 8):max_pos].tolist()],
    )


def _score_z(scores):
    if scores.numel() <= 1:
        return scores * 0.0
    std = scores.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return scores * 0.0
    return (scores - scores.mean()) / std.clamp_min(1e-8)


def _build_true_grad_token_scorer(model_wrapper, true_grads, res_ids, max_positions):
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
        candidate_ids = [
            int(token_id) for token_id in res_ids[pos]
            if 0 <= int(token_id) < grad_wte.shape[0]
        ]
        if not candidate_ids:
            continue
        pos_grad = grad_wpe[pos]
        if pos_grad.norm(p=2) <= 0:
            continue
        cand_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        cand_vecs = F.normalize(grad_wte.index_select(0, cand_tensor), dim=-1)
        pos_vec = F.normalize(pos_grad.reshape(1, -1), dim=-1).reshape(-1)
        scores = torch.nan_to_num(cand_vecs @ pos_vec, nan=0.0, posinf=0.0, neginf=0.0)
        position_scores[pos] = {
            token_id: float(score)
            for token_id, score in zip(candidate_ids, scores.tolist())
        }
    return support_scores, position_scores, wte_name, wpe_name


def _creates_repeated_ngram(prefix, token_id, n):
    if len(prefix) < n - 1:
        return False
    gram = tuple(prefix[-(n - 1):] + [int(token_id)])
    for start in range(0, len(prefix) - n + 1):
        if tuple(prefix[start:start + n]) == gram:
            return True
    return False


def init_embeddings(args, model_wrapper, res_ids, input_ids, attention_mask, init_ids=None):
    emb_matrix = model_wrapper.get_input_embeddings_weight().detach()
    batch_size, seq_len = input_ids.shape
    x_embeds = torch.zeros(batch_size, seq_len, emb_matrix.shape[1], device=input_ids.device)

    if init_ids is not None:
        x_embeds = emb_matrix[init_ids.to(input_ids.device)].clone()
    else:
        for pos in range(seq_len):
            if pos < len(res_ids) and len(res_ids[pos]) > 0:
                ids = torch.tensor(res_ids[pos], dtype=torch.long, device=input_ids.device)
                sampled_ids = ids[torch.randint(len(ids), (batch_size,), device=input_ids.device)]
                x_embeds[:, pos] = emb_matrix[sampled_ids]
            else:
                x_embeds[:, pos] = emb_matrix[model_wrapper.pad_token]

    pad_mask = attention_mask == 0
    x_embeds = _with_pad_embeddings(x_embeds, pad_mask, emb_matrix[model_wrapper.pad_token])
    should_add_noise = args.hybrid_init_noise > 0 and (
            init_ids is None
            or getattr(args, 'iterative_noise_init_ids', False)
            or getattr(args, '_iterative_force_noise', False)
    )
    if should_add_noise:
        x_embeds = x_embeds + args.hybrid_init_noise * torch.randn_like(x_embeds)
        x_embeds = _with_pad_embeddings(x_embeds, pad_mask, emb_matrix[model_wrapper.pad_token])
    return x_embeds.detach().requires_grad_(True)


def project_to_vocab(x_embeds, emb_matrix, pad_mask, pad_token, emb_norm=None):
    batch_size, seq_len, _ = x_embeds.shape
    if emb_norm is None:
        emb_norm = F.normalize(emb_matrix, dim=-1)
    x_norm = F.normalize(x_embeds.detach(), dim=-1)
    ids = torch.full((batch_size, seq_len), pad_token, dtype=torch.long, device=x_embeds.device)

    for pos in range(seq_len):
        sims = x_norm[:, pos] @ emb_norm.T
        ids[:, pos] = sims.argmax(dim=-1)

    ids[pad_mask] = pad_token
    return ids


def project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask, pad_token, emb_norm=None):
    batch_size, seq_len, _ = x_embeds.shape
    if emb_norm is None:
        emb_norm = F.normalize(emb_matrix, dim=-1)
    x_norm = F.normalize(x_embeds.detach(), dim=-1)
    ids = torch.full((batch_size, seq_len), pad_token, dtype=torch.long, device=x_embeds.device)

    for pos in range(seq_len):
        valid_ids = torch.where(candidate_mask[pos])[0]
        valid_embs = emb_norm[valid_ids]
        sims = x_norm[:, pos] @ valid_embs.T
        ids[:, pos] = valid_ids[sims.argmax(dim=-1)]

    ids[pad_mask] = pad_token
    return ids


def fuzzy_gpt2_lm_loss(lm, x_embeds, emb_norm, lm_emb_matrix, candidate_mask, attention_mask, temperature):
    if lm is None:
        return x_embeds.sum() * 0.0

    # Keep the fuzzy LM prior inside DAGER's candidate set. The previous version
    # built [batch, seq_len, vocab] soft distributions and LM logits every step,
    # which is the main hybrid memory spike on GPT-2.
    x_norm = F.normalize(x_embeds, dim=-1)
    batch_size, seq_len, _ = x_embeds.shape
    lm_embeds = x_embeds.new_zeros(batch_size, seq_len, lm_emb_matrix.shape[-1])
    candidate_ids_by_pos = []
    alpha_by_pos = []

    for pos in range(seq_len):
        candidate_ids = torch.where(candidate_mask[pos])[0]
        candidate_ids_by_pos.append(candidate_ids)
        candidate_embs = emb_norm.index_select(0, candidate_ids)
        logits_to_candidates = x_norm[:, pos] @ candidate_embs.T
        alpha = F.softmax(logits_to_candidates / temperature, dim=-1)
        alpha_by_pos.append(alpha)
        lm_candidate_embs = lm_emb_matrix.index_select(0, candidate_ids)
        lm_embeds[:, pos] = alpha @ lm_candidate_embs

    transformer = lm.transformer if hasattr(lm, 'transformer') else lm.base_model
    lm_out = transformer(
        inputs_embeds=lm_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    hidden_states = lm_out[0]

    token_losses = []
    for pos in range(seq_len - 1):
        target_ids = candidate_ids_by_pos[pos + 1]
        target_embs = lm_emb_matrix.index_select(0, target_ids)
        logits_to_targets = hidden_states[:, pos] @ target_embs.T
        log_probs = logits_to_targets.log_softmax(dim=-1)
        token_losses.append(-(log_probs * alpha_by_pos[pos + 1]).sum(dim=-1))

    token_loss = torch.stack(token_losses, dim=1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)
    return token_loss.mean()


def load_lm_prior(args):
    if args.hybrid_use_lm_prior == 'true' and args.coeff_perplexity <= 0:
        logger.info('Hybrid LM prior disabled because coeff_perplexity=%.6g <= 0.', args.coeff_perplexity)
        return None
    if not _hybrid_uses_lm_prior(args):
        return None
    if args.model_path not in ['gpt2', 'openai-community/gpt2-large']:
        logger.info('Hybrid LM prior disabled: fuzzy LM prior is currently implemented for GPT-2 vocabularies only.')
        return None

    lm_source = _resolve_local_model_path(args.model_path, args.cache_dir)
    logger.info(f"Loading hybrid LM prior from {lm_source}")
    lm_kwargs = {'pretrained_model_name_or_path': lm_source, 'attn_implementation': args.attn_implementation}
    if args.cache_dir is not None and not os.path.isdir(lm_source):
        lm_kwargs['cache_dir'] = args.cache_dir
    lm_dtype = _hybrid_lm_dtype(args)
    if lm_dtype is not None:
        lm_kwargs['torch_dtype'] = lm_dtype
    lm_kwargs['low_cpu_mem_usage'] = True
    if os.path.isdir(lm_source) or any(
            str(os.environ.get(flag, "")).lower() in {"1", "true", "yes"}
            for flag in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]
    ):
        lm_kwargs['local_files_only'] = True
    lm = AutoModelForCausalLM.from_pretrained(**lm_kwargs)
    lm.config.pad_token_id = lm.config.eos_token_id
    lm.config.use_cache = False
    lm.config.output_hidden_states = False
    lm.config.output_attentions = False
    lm.eval()
    for param in lm.parameters():
        param.requires_grad_(False)
    lm.to('cpu')
    return lm


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


def _discrete_lm_loss_from_ids(lm, ids, attention_mask):
    if lm is None:
        return 0.0
    device = next(lm.parameters()).device
    input_ids = ids.to(device)
    attn = attention_mask.to(device)
    if input_ids.shape[1] <= 1:
        return 0.0
    with torch.no_grad():
        outputs = lm(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = outputs.logits[:, :-1]
        targets = input_ids[:, 1:]
        mask = attn[:, 1:].float()
        log_probs = logits.log_softmax(dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(((nll * mask).sum() / mask.sum().clamp_min(1.0)).item())


def _candidate_mask_penalty(ids, res_ids, attention_mask, pad_token):
    ids_cpu = ids.detach().cpu()
    attn_cpu = attention_mask.detach().cpu().bool()
    active = int(attn_cpu.sum().item())
    if active <= 0:
        return 0.0
    violations = 0
    for pos in range(ids_cpu.shape[1]):
        allowed = set(res_ids[pos]) if pos < len(res_ids) else {int(pad_token)}
        for row_idx in torch.where(attn_cpu[:, pos])[0].tolist():
            token_id = int(ids_cpu[row_idx, pos].item())
            if token_id not in allowed:
                violations += 1
    return violations / max(active, 1)


def _token_reuse_penalty(ids, attention_mask, batch_size):
    ids_cpu = ids.detach().cpu()
    attn_cpu = attention_mask.detach().cpu().bool()
    counts = {}
    for token_id in ids_cpu[attn_cpu].reshape(-1).tolist():
        counts[int(token_id)] = counts.get(int(token_id), 0) + 1
    threshold = max(4, batch_size // 8)
    penalty = sum(max(0, count - threshold) for count in counts.values())
    active = int(attn_cpu.sum().item())
    return penalty / max(active, 1)


def _discrete_batch_objective(args, model_wrapper, lm, true_grads, res_ids, ids, attention_mask, true_labels):
    rec_loss = compute_rec_loss_from_ids(
        args,
        model_wrapper,
        true_grads,
        ids,
        attention_mask,
        true_labels,
    ).item()
    lm_loss = _discrete_lm_loss_from_ids(lm, ids, attention_mask)
    mask_penalty = _candidate_mask_penalty(ids, res_ids, attention_mask, model_wrapper.pad_token)
    reuse_penalty = _token_reuse_penalty(ids, attention_mask, ids.shape[0])
    total = rec_loss + 0.04 * lm_loss + 0.03 * reuse_penalty + 0.20 * mask_penalty
    return {
        'total': float(total),
        'rec_loss': float(rec_loss),
        'lm_loss': float(lm_loss),
        'mask_penalty': float(mask_penalty),
        'reuse_penalty': float(reuse_penalty),
    }


def _decode_single_token(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _token_quality_score(tokenizer, prefix_tokens, token_id):
    txt = _decode_single_token(tokenizer, token_id)
    stripped = txt.strip()
    if txt == '' or stripped == '':
        return -2.0, True
    if txt.startswith('<|') and txt.endswith('|>'):
        return -3.0, True

    prefix_text = tokenizer.decode(prefix_tokens[-6:], skip_special_tokens=False) if prefix_tokens else ''
    prev_char = prefix_text[-1] if prefix_text else ' '
    at_boundary = (not prefix_text) or prev_char.isspace() or (not prev_char.isalnum())
    starts_with_space = txt[0].isspace()
    alpha_fragment = stripped.isalpha() and len(stripped) <= 3

    if at_boundary and alpha_fragment and not starts_with_space:
        return -1.5, True
    if (not at_boundary) and starts_with_space and stripped.isalpha():
        return -1.5, True

    score = 0.0
    if alpha_fragment and not starts_with_space:
        score -= 0.75
    if all(not c.isalnum() for c in stripped):
        score -= 0.15
    if any(ord(c) > 127 for c in stripped):
        score -= 0.20
    return score, False


def late_discrete_edit_refine_ids(args, model_wrapper, lm, true_grads, res_ids, ids, attention_mask, true_labels,
                                  edit_mask):
    if edit_mask is None:
        return ids
    editable = edit_mask.detach().bool().cpu()
    if editable.numel() == 0 or not bool(editable.any().item()):
        logger.info('Hybrid late discrete edit skipped: no generated suffix positions.')
        return ids

    refined = ids.detach().clone()
    pad_token = int(model_wrapper.pad_token)
    eos_token = int(model_wrapper.eos_token) if model_wrapper.eos_token is not None else pad_token
    attention_cpu = attention_mask.detach().cpu().bool()
    editable = editable & attention_cpu
    if not bool(editable.any().item()):
        logger.info('Hybrid late discrete edit skipped: generated positions are all padding.')
        return refined

    support_scores, position_scores, wte_name, wpe_name = _build_true_grad_token_scorer(
        model_wrapper,
        true_grads,
        res_ids,
        min(refined.shape[1], len(res_ids)),
    )
    current_state = _discrete_batch_objective(
        args,
        model_wrapper,
        lm,
        true_grads,
        res_ids,
        refined,
        attention_mask,
        true_labels,
    )

    batch_size = refined.shape[0]
    max_trials = 128 if batch_size <= 32 else 96
    max_accepts = 8 if batch_size <= 32 else 4
    candidates_per_cell = 3
    trials = 0
    accepted = 0
    quality_rejects = 0
    score_rejects = 0
    refined_cpu = refined.detach().cpu()
    token_counts = {}
    for token_id in refined_cpu[attention_cpu].reshape(-1).tolist():
        token_counts[int(token_id)] = token_counts.get(int(token_id), 0) + 1

    edit_cells = []
    for pos in range(min(refined.shape[1], len(res_ids))):
        rows = torch.where(editable[:, pos])[0].tolist()
        if not rows:
            continue
        pos_counts = {}
        for row_idx in rows:
            token_id = int(refined_cpu[row_idx, pos].item())
            pos_counts[token_id] = pos_counts.get(token_id, 0) + 1
        for row_idx in rows:
            token_id = int(refined_cpu[row_idx, pos].item())
            repeat_priority = 1 if pos_counts.get(token_id, 0) > 1 else 0
            recent = refined_cpu[row_idx, max(0, pos - 8):pos].tolist()
            local_repeat_priority = 1 if token_id in recent else 0
            edit_cells.append((-(repeat_priority + local_repeat_priority), pos, row_idx))
    edit_cells.sort()

    for _, pos, row_idx in edit_cells:
        if trials >= max_trials or accepted >= max_accepts:
            break
        current_token = int(refined[row_idx, pos].item())
        candidates = [
            int(token_id) for token_id in res_ids[pos]
            if int(token_id) not in {pad_token, eos_token, current_token}
        ]
        if not candidates:
            continue
        prefix = refined_cpu[row_idx, :pos].tolist()

        def candidate_score(token_id):
            repeat_penalty = 0.0
            if token_id in prefix[-8:]:
                repeat_penalty += 1.0
            if _creates_repeated_ngram(prefix, token_id, 3) or _creates_repeated_ngram(prefix, token_id, 4):
                repeat_penalty += 2.0
            frequency_penalty = max(0, token_counts.get(int(token_id), 0) - max(4, batch_size // 8))
            quality_score, _ = _token_quality_score(model_wrapper.tokenizer, prefix, token_id)
            return (
                    position_scores[pos].get(int(token_id), 0.0)
                    + 0.35 * support_scores.get(int(token_id), 0.0)
                    + 0.30 * quality_score
                    - 0.35 * repeat_penalty
                    - 0.04 * frequency_penalty
            )

        ranked_candidates = sorted(candidates, key=candidate_score, reverse=True)[:candidates_per_cell]
        old_score = candidate_score(current_token)
        for candidate in ranked_candidates:
            _, bad_candidate = _token_quality_score(model_wrapper.tokenizer, prefix, candidate)
            if bad_candidate:
                quality_rejects += 1
                continue
            new_score = candidate_score(candidate)
            if new_score + 0.05 < old_score:
                score_rejects += 1
                continue
            trial_ids = refined.detach().clone()
            trial_ids[row_idx, pos] = int(candidate)
            trial_loss = compute_rec_loss_from_ids(
                args,
                model_wrapper,
                true_grads,
                trial_ids,
                attention_mask,
                true_labels,
            ).item()
            trial_state = None
            if trial_loss + 1e-6 < current_state['rec_loss']:
                trial_state = _discrete_batch_objective(
                    args,
                    model_wrapper,
                    lm,
                    true_grads,
                    res_ids,
                    trial_ids,
                    attention_mask,
                    true_labels,
                )
            trials += 1
            if (
                    trial_state is not None
                    and trial_state['rec_loss'] + 1e-6 < current_state['rec_loss']
                    and trial_state['lm_loss'] <= current_state['lm_loss'] + 0.20
                    and trial_state['mask_penalty'] <= current_state['mask_penalty'] + 1e-9
                    and trial_state['total'] + 1e-6 < current_state['total']
            ):
                old_token = int(refined[row_idx, pos].item())
                refined = trial_ids
                refined_cpu[row_idx, pos] = int(candidate)
                token_counts[old_token] = max(0, token_counts.get(old_token, 0) - 1)
                token_counts[int(candidate)] = token_counts.get(int(candidate), 0) + 1
                current_state = trial_state
                accepted += 1
                break
            if trials >= max_trials:
                break
        if trials > 0 and trials % 48 == 0:
            cleanup_memory()

    logger.info(
        'Hybrid late discrete edit pass | wte=%s | wpe=%s | editable=%s | trials=%s | accepted=%s | '
        'quality_rejects=%s | score_rejects=%s | rec=%.6f | lm=%.6f | mask=%.6f | reuse=%.6f | total=%.6f',
        wte_name,
        wpe_name,
        int(editable.sum().item()),
        trials,
        accepted,
        quality_rejects,
        score_rejects,
        current_state['rec_loss'],
        current_state['lm_loss'],
        current_state['mask_penalty'],
        current_state['reuse_penalty'],
        current_state['total'],
    )
    return refined


def _apply_sequence_edit(trial_row, op):
    kind = op[0]
    row = list(trial_row)
    if kind == 'swap':
        _, i, j = op
        row[i], row[j] = row[j], row[i]
    elif kind == 'move':
        _, src, dst = op
        token = row.pop(src)
        row.insert(dst, token)
    elif kind == 'swap_span':
        _, i, j, width = op
        left = row[i:i + width]
        right = row[j:j + width]
        row[i:i + width] = right
        row[j:j + width] = left
    elif kind == 'move_span':
        _, start, width, dst = op
        span = row[start:start + width]
        del row[start:start + width]
        if dst > start:
            dst -= width
        row[dst:dst] = span
    return row


def _sequence_edit_proposals(edit_positions):
    if len(edit_positions) < 2:
        return []
    positions = sorted(edit_positions)
    anchors = positions[:4] + positions[max(0, len(positions) - 4):]
    anchors = sorted(set(anchors))
    proposals = []
    for idx in range(len(anchors) - 1):
        i = anchors[idx]
        j = anchors[idx + 1]
        proposals.append(('swap', i, j))
        proposals.append(('move', j, i))
    for i in anchors[:3]:
        for j in anchors[-3:]:
            if i >= j:
                continue
            proposals.append(('swap', i, j))
            proposals.append(('move', i, j))
            if j - i >= 2:
                proposals.append(('swap_span', i, j - 1, 1))
    contiguous = []
    for start in range(len(anchors) - 1):
        if anchors[start + 1] == anchors[start] + 1:
            contiguous.append((anchors[start], 2))
    for start, width in contiguous[:3]:
        for dst in anchors:
            if dst <= start or dst >= start + width:
                proposals.append(('move_span', start, width, dst))
    unique = []
    seen = set()
    for op in proposals:
        if op in seen:
            continue
        seen.add(op)
        unique.append(op)
    return unique[:24]


def sequence_edit_beam_refine_ids(args, model_wrapper, lm, true_grads, res_ids, ids, attention_mask, true_labels,
                                  edit_mask):
    if edit_mask is None:
        return ids
    editable = (edit_mask.detach().cpu().bool() & attention_mask.detach().cpu().bool())
    if not bool(editable.any().item()):
        logger.info('Hybrid sequence-edit beam skipped: no editable active positions.')
        return ids

    current = ids.detach().clone()
    current_state = _discrete_batch_objective(
        args,
        model_wrapper,
        lm,
        true_grads,
        res_ids,
        current,
        attention_mask,
        true_labels,
    )
    batch_size = current.shape[0]
    max_trials = 48 if batch_size <= 32 else 24
    max_accepts = 6 if batch_size <= 32 else 3
    trials = 0
    accepted = 0

    row_order = sorted(
        range(current.shape[0]),
        key=lambda row_idx: int(editable[row_idx].sum().item()),
        reverse=True,
    )
    for row_idx in row_order:
        if trials >= max_trials or accepted >= max_accepts:
            break
        edit_positions = torch.where(editable[row_idx])[0].tolist()
        proposals = _sequence_edit_proposals(edit_positions)
        if not proposals:
            continue
        row_tokens = current[row_idx].detach().cpu().tolist()
        best_local = None
        for op in proposals:
            if trials >= max_trials:
                break
            new_row = _apply_sequence_edit(row_tokens, op)
            trial_ids = current.detach().clone()
            trial_ids[row_idx] = torch.tensor(new_row, dtype=trial_ids.dtype, device=trial_ids.device)
            trial_state = _discrete_batch_objective(
                args,
                model_wrapper,
                lm,
                true_grads,
                res_ids,
                trial_ids,
                attention_mask,
                true_labels,
            )
            trials += 1
            if (
                    trial_state['total'] + 1e-6 < current_state['total']
                    and trial_state['rec_loss'] <= current_state['rec_loss'] + 0.01
                    and trial_state['lm_loss'] <= current_state['lm_loss'] + 0.25
                    and trial_state['mask_penalty'] <= current_state['mask_penalty'] + 1e-9
            ):
                if best_local is None or trial_state['total'] < best_local[0]['total']:
                    best_local = (trial_state, trial_ids, op)
        if best_local is not None:
            current_state, current, op = best_local
            accepted += 1
            logger.info(
                'Hybrid sequence-edit beam accepted | row=%s | op=%s | rec=%.6f | lm=%.6f | mask=%.6f | reuse=%.6f | total=%.6f',
                row_idx,
                op,
                current_state['rec_loss'],
                current_state['lm_loss'],
                current_state['mask_penalty'],
                current_state['reuse_penalty'],
                current_state['total'],
            )
        if trials > 0 and trials % 24 == 0:
            cleanup_memory()

    logger.info(
        'Hybrid sequence-edit beam | editable=%s | trials=%s | accepted=%s | rec=%.6f | lm=%.6f | mask=%.6f | reuse=%.6f | total=%.6f',
        int(editable.sum().item()),
        trials,
        accepted,
        current_state['rec_loss'],
        current_state['lm_loss'],
        current_state['mask_penalty'],
        current_state['reuse_penalty'],
        current_state['total'],
    )
    return current


def _refresh_res_ids_from_projected_ids(res_ids, ids, attention_mask, pad_token, refresh_mask=None):
    refreshed = [list(pos_ids) for pos_ids in res_ids]
    seq_len = ids.shape[1]
    while len(refreshed) < seq_len:
        refreshed.append([])

    ids_cpu = ids.detach().cpu()
    mask_cpu = attention_mask.detach().cpu()
    refresh_cpu = refresh_mask.detach().cpu().bool() if refresh_mask is not None else None
    for pos in range(seq_len):
        seen = set()
        front_ids = []
        for row_idx, token in enumerate(ids_cpu[:, pos].tolist()):
            if token == pad_token or token in seen:
                continue
            if mask_cpu[:, pos].sum().item() == 0:
                continue
            if refresh_cpu is not None and not bool(refresh_cpu[row_idx, pos].item()):
                continue
            seen.add(token)
            front_ids.append(int(token))
        tail = [int(token) for token in refreshed[pos] if int(token) not in seen]
        refreshed[pos] = front_ids + tail
    return refreshed


def _decode_ids_for_log(model_wrapper, ids, max_rows=3):
    rows = []
    for row in ids[:max_rows].detach().cpu():
        rows.append(model_wrapper.tokenizer.decode(row.tolist()))
    return rows


def autoregressive_extend_decoder_ids(args, model_wrapper, lm, res_ids, decoded_ids, orig_batch, true_grads=None,
                                      extend_row_mask=None):
    generated_mask = torch.zeros_like(decoded_ids, dtype=torch.bool)
    if lm is None or not model_wrapper.is_decoder():
        logger.info('Hybrid autoregressive seed extension skipped: lm=%s decoder=%s', lm is not None,
                    model_wrapper.is_decoder())
        return decoded_ids, generated_mask

    attention_mask = orig_batch['attention_mask'].detach().cpu()
    out = decoded_ids.detach().clone()
    pad_token = int(model_wrapper.pad_token)
    eos_token = int(model_wrapper.eos_token) if model_wrapper.eos_token is not None else pad_token
    target_lengths = attention_mask.sum(dim=1).long().tolist()
    max_target_len = min(max(target_lengths) if target_lengths else 0, out.shape[1], len(res_ids))
    if max_target_len <= 0:
        return out, generated_mask

    support_scores, position_scores, wte_name, wpe_name = _build_true_grad_token_scorer(
        model_wrapper,
        true_grads,
        res_ids,
        max_target_len,
    ) if true_grads is not None else ({}, [{} for _ in range(max_target_len)], None, None)
    if support_scores or any(position_scores):
        logger.info(
            'Hybrid autoregressive gradient-aware scoring active | wte=%s | wpe=%s | support=%s | positions=%s',
            wte_name,
            wpe_name,
            len(support_scores),
            sum(1 for scores in position_scores if len(scores) > 0),
        )

    if extend_row_mask is None:
        extend_row_mask = torch.ones(out.shape[0], dtype=torch.bool)
    else:
        extend_row_mask = extend_row_mask.detach().cpu().bool()

    prefixes = []
    prefix_lens = []
    for row_idx in range(out.shape[0]):
        target_len = min(int(target_lengths[row_idx]), out.shape[1], len(res_ids))
        prefix = []
        for token_id in out[row_idx, :target_len].detach().cpu().tolist():
            token_id = int(token_id)
            if token_id == pad_token:
                break
            prefix.append(token_id)
        prefixes.append(prefix)
        prefix_lens.append(len(prefix))

    original_lm_device = next(lm.parameters()).device
    if str(original_lm_device) != str(args.device):
        logger.info('Moving hybrid LM prior to %s for autoregressive seed extension.', args.device)
        lm.to(args.device)
    lm.eval()

    extended_tokens = 0
    with torch.no_grad():
        for pos in range(min(prefix_lens) if prefix_lens else 0, max_target_len):
            row_indices = [
                row_idx for row_idx, target_len in enumerate(target_lengths)
                if extend_row_mask[row_idx]
                   and len(prefixes[row_idx]) == pos
                   and pos < min(int(target_len), out.shape[1], len(res_ids))
            ]
            if not row_indices:
                continue

            candidate_lists = []
            active_rows = []
            for row_idx in row_indices:
                candidates = [
                    int(token_id) for token_id in res_ids[pos]
                    if int(token_id) not in {pad_token, eos_token}
                ]
                if not candidates:
                    continue
                if args.max_ids > 0:
                    candidates = candidates[:args.max_ids]
                candidate_lists.append(candidates)
                active_rows.append(row_idx)
            if not active_rows:
                continue

            if pos == 0:
                chosen_counts = {}
                for row_idx, candidates in zip(active_rows, candidate_lists):
                    ranked = sorted(
                        candidates,
                        key=lambda token_id: (
                                position_scores[pos].get(int(token_id), 0.0)
                                + 0.35 * support_scores.get(int(token_id), 0.0)
                                - 0.15 * chosen_counts.get(int(token_id), 0)
                        ),
                        reverse=True,
                    )
                    chosen = int(ranked[0])
                    out[row_idx, pos] = chosen
                    generated_mask[row_idx, pos] = True
                    prefixes[row_idx].append(chosen)
                    chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
                    extended_tokens += 1
                continue

            max_prefix_len = max(len(prefixes[row_idx]) for row_idx in active_rows)
            lm_inputs = torch.full(
                (len(active_rows), max_prefix_len),
                eos_token,
                dtype=torch.long,
                device=args.device,
            )
            lm_attention = torch.zeros_like(lm_inputs)
            for batch_idx, row_idx in enumerate(active_rows):
                prefix = prefixes[row_idx]
                lm_inputs[batch_idx, :len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=args.device)
                lm_attention[batch_idx, :len(prefix)] = 1

            lm_outputs = lm(input_ids=lm_inputs, attention_mask=lm_attention, use_cache=False)
            logits = lm_outputs.logits
            last_positions = lm_attention.sum(dim=1).clamp_min(1) - 1
            next_logits = logits[torch.arange(len(active_rows), device=args.device), last_positions]

            chosen_counts = {}
            global_usage = {}
            for prefix in prefixes:
                for token_id in prefix:
                    global_usage[int(token_id)] = global_usage.get(int(token_id), 0) + 1

            for batch_idx, (row_idx, candidates) in enumerate(zip(active_rows, candidate_lists)):
                cand_tensor = torch.tensor(candidates, dtype=torch.long, device=args.device)
                lm_scores = _score_z(next_logits[batch_idx].index_select(0, cand_tensor))
                pos_scores = torch.tensor(
                    [position_scores[pos].get(int(token_id), 0.0) for token_id in candidates],
                    dtype=torch.float32,
                    device=args.device,
                )
                support_component = torch.tensor(
                    [support_scores.get(int(token_id), 0.0) for token_id in candidates],
                    dtype=torch.float32,
                    device=args.device,
                )
                scores = lm_scores + 0.75 * _score_z(pos_scores) + 0.35 * _score_z(support_component)
                if prefixes[row_idx]:
                    seen = set(prefixes[row_idx][-8:])
                    if seen:
                        repeat_mask = torch.tensor(
                            [int(token_id) in seen for token_id in candidates],
                            dtype=torch.bool,
                            device=args.device,
                        )
                        scores = scores.masked_fill(repeat_mask, scores.min() - 4.0)
                    repeat_ngram_penalty = torch.tensor(
                        [
                            _creates_repeated_ngram(prefixes[row_idx], token_id, 3)
                            or _creates_repeated_ngram(prefixes[row_idx], token_id, 4)
                            for token_id in candidates
                        ],
                        dtype=torch.bool,
                        device=args.device,
                    )
                    scores = scores.masked_fill(repeat_ngram_penalty, scores.min() - 6.0)
                position_dup_penalty = torch.tensor(
                    [chosen_counts.get(int(token_id), 0) for token_id in candidates],
                    dtype=torch.float32,
                    device=args.device,
                )
                global_dup_penalty = torch.tensor(
                    [max(0, global_usage.get(int(token_id), 0) - max(4, out.shape[0] // 8)) for token_id in candidates],
                    dtype=torch.float32,
                    device=args.device,
                )
                scores = scores - 0.45 * position_dup_penalty - 0.08 * global_dup_penalty
                chosen = int(cand_tensor[int(scores.argmax().item())].item())
                out[row_idx, pos] = chosen
                generated_mask[row_idx, pos] = True
                prefixes[row_idx].append(chosen)
                chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
                global_usage[chosen] = global_usage.get(chosen, 0) + 1
                extended_tokens += 1

    for row_idx, target_len in enumerate(target_lengths):
        target_len = min(int(target_len), out.shape[1])
        if target_len < out.shape[1]:
            out[row_idx, target_len:] = pad_token

    if str(original_lm_device) != str(args.device):
        lm.to(original_lm_device)
    logger.info(
        'Hybrid autoregressive seed extension filled %s tokens | extend_rows=%s/%s | avg_prefix %.1f -> avg_target %.1f',
        extended_tokens,
        int(extend_row_mask.sum().item()),
        int(extend_row_mask.numel()),
        float(np.mean(prefix_lens)) if prefix_lens else 0.0,
        float(np.mean(target_lengths)) if target_lengths else 0.0,
    )
    return out, generated_mask


def select_dager_decoder_ids(args, model_wrapper, R_Qs, res_ids, orig_batch, true_grads=None, lm=None):
    if not model_wrapper.is_decoder():
        return None, None

    if not hasattr(args, 'aug_decoder_continue_approx'):
        setattr(args, 'aug_decoder_continue_approx', True)
        setattr(args, 'aug_decoder_approx_beam_size', min(max(int(args.batch_size), 32), 96))
        setattr(args, 'aug_decoder_approx_max_len', int(orig_batch['input_ids'].shape[1]))
        setattr(args, 'aug_decoder_approx_position_topk', 96)
        setattr(args, 'aug_decoder_fallback_quality_weight', 0.05)
        logger.info(
            'Hybrid decoder approximate autoregressive continuation enabled | beam=%s | max_len=%s | position_topk=%s',
            args.aug_decoder_approx_beam_size,
            args.aug_decoder_approx_max_len,
            args.aug_decoder_approx_position_topk,
        )

    max_ids = -1
    for pos_ids in res_ids:
        if len(pos_ids) > args.max_ids:
            max_ids = args.max_ids

    predicted_sentences, predicted_scores, top_B_incorrect_sentences, top_B_incorrect_scores = filter_decoder(args,
                                                                                                              model_wrapper,
                                                                                                              R_Qs,
                                                                                                              res_ids,
                                                                                                              max_ids=max_ids,
                                                                                                              )
    if len(predicted_sentences) < orig_batch['input_ids'].shape[0]:
        predicted_sentences += top_B_incorrect_sentences
        predicted_scores += top_B_incorrect_scores
    if len(predicted_sentences) == 0:
        return None, None

    correct_entries = []
    approx_entries = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    max_len = max(len(sentence) for sentence in predicted_sentences)
    for sentence, score in zip(predicted_sentences, predicted_scores):
        if score < model_wrapper.effective_l2_span_thresh(args.l2_span_thresh):
            correct_entries.append({
                'sentence': sentence,
                'score': float(score),
                'is_exact': True,
            })
        else:
            approx_entries.append({
                'sentence': sentence,
                'score': float(score),
                'is_exact': False,
            })
            approx_sentences_ext.append(sentence + [-1] * (max_len - len(sentence)))
            approx_sentences_lens.append(len(sentence))
            approx_scores.append(score)

    selected_entries = correct_entries.copy()
    if len(approx_entries) > 0 and len(selected_entries) < args.batch_size:
        approx_scores = torch.tensor(approx_scores)
        approx_sentences_lens = torch.tensor(approx_sentences_lens)
        approx_sentences_ext_tensor = torch.tensor(approx_sentences_ext)

        for entry in correct_entries:
            sentence = entry['sentence']
            similar_sentences = ((torch.tensor(sentence) == approx_sentences_ext_tensor[:, :len(sentence)]).sum(1)
                                 >= torch.min(approx_sentences_lens,
                                              torch.tensor(len(sentence))) * args.distinct_thresh)
            approx_scores[similar_sentences] = torch.inf

        for _ in range(len(selected_entries), args.batch_size):
            idx = torch.argmin(approx_scores)
            if torch.isinf(approx_scores[idx]):
                break
            selected_entries.append(approx_entries[idx])
            similar_sentences = ((torch.tensor(approx_sentences_ext[idx]) == approx_sentences_ext_tensor).sum(1)
                                 >= max_len * args.distinct_thresh)
            approx_scores[similar_sentences] = torch.inf

    selected_entries = selected_entries[:orig_batch['input_ids'].shape[0]]
    if len(selected_entries) == 0:
        return None, None

    selected_entries = reorder_decoder_sentence_ids(model_wrapper, selected_entries, orig_batch)
    selected_sentences = [entry['sentence'] for entry in selected_entries]

    seq_len = orig_batch['input_ids'].shape[1]
    extend_row_mask = torch.ones(len(selected_entries), dtype=torch.bool)
    if args.batch_size <= 16:
        min_approx_prefix_len = max(6, int(np.ceil(seq_len * 0.40)))
        exact_count = 0
        for row_idx, entry in enumerate(selected_entries):
            prefix_len = len(entry['sentence'])
            if entry['is_exact']:
                exact_count += 1
                continue
            if prefix_len < min_approx_prefix_len:
                extend_row_mask[row_idx] = False
        logger.info(
            'Hybrid decoder seed guard | batch=%s | exact=%s | approx=%s | extend_rows=%s/%s | min_approx_prefix=%s',
            args.batch_size,
            exact_count,
            len(selected_entries) - exact_count,
            int(extend_row_mask.sum().item()),
            int(extend_row_mask.numel()),
            min_approx_prefix_len,
        )

    decoded_ids = torch.full(
        (orig_batch['input_ids'].shape[0], seq_len),
        model_wrapper.pad_token,
        dtype=torch.long,
        device=args.device,
    )
    for row, sentence in enumerate(selected_sentences):
        sentence = sentence[:seq_len]
        decoded_ids[row, :len(sentence)] = torch.tensor(sentence, dtype=torch.long, device=args.device)
    decoded_ids, generated_mask = autoregressive_extend_decoder_ids(
        args,
        model_wrapper,
        lm,
        res_ids,
        decoded_ids,
        orig_batch,
        true_grads=true_grads,
        extend_row_mask=extend_row_mask,
    )
    return decoded_ids, generated_mask


def reorder_decoder_sentence_ids(model_wrapper, predicted_sentences, orig_batch):
    reordered_sentences = []
    references = orig_batch['input_ids'].detach().cpu().tolist()
    pad_token = model_wrapper.pad_token
    unused = set(range(len(predicted_sentences)))

    for reference in references:
        reference = [token for token in reference if token != pad_token]
        search_indices = unused if unused else set(range(len(predicted_sentences)))
        best_idx = next(iter(search_indices))
        best_score = -1
        for idx in search_indices:
            sentence = predicted_sentences[idx]['sentence'] if isinstance(predicted_sentences[idx], dict) else \
            predicted_sentences[idx]
            compare_len = max(len(reference), len(sentence))
            ref_ext = reference + [pad_token] * (compare_len - len(reference))
            sent_ext = sentence + [pad_token] * (compare_len - len(sentence))
            score = sum(ref_token == sent_token for ref_token, sent_token in zip(ref_ext, sent_ext))
            if score > best_score:
                best_score = score
                best_idx = idx
        reordered_sentences.append(predicted_sentences[best_idx])
        unused.discard(best_idx)

    return reordered_sentences


def reorder_decoder_predictions(args, tokenizer, prediction, reference):
    if len(prediction) == 0:
        return prediction

    new_prediction = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    unused = set(range(len(prediction)))
    for ref in reference:
        sequences = [ref] + prediction
        batch = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        scores = (batch['input_ids'][1:] == batch['input_ids'][0]).sum(1)
        if unused:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[list(unused)] = False
            scores = scores.masked_fill(mask, -1)
        best_idx = int(scores.argmax().item())
        new_prediction.append(prediction[best_idx])
        unused.discard(best_idx)
    tokenizer.padding_side = old_padding_side
    return new_prediction


def hybrid_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels, init_ids=None,
                    edit_mask=None):
    original_lm_device = None
    if lm is not None:
        original_lm_device = next(lm.parameters()).device
        if str(original_lm_device) != str(args.device):
            logger.info('Moving hybrid LM prior from %s to %s', original_lm_device, args.device)
            lm.to(args.device)
            cleanup_memory()

    emb_matrix = model_wrapper.get_input_embeddings_weight().detach().to(args.device)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    lm_emb_matrix = lm.get_input_embeddings().weight.detach() if lm is not None else None
    input_ids = orig_batch['input_ids'].to(args.device)
    attention_mask = orig_batch['attention_mask'].to(args.device)
    pad_mask = attention_mask == 0
    candidate_mask = build_candidate_mask(res_ids, input_ids.shape[1], emb_matrix.shape[0], model_wrapper.pad_token,
                                          args.device)

    if init_ids is not None and edit_mask is not None and args.batch_size <= 32:
        active_mask = attention_mask.bool()
        editable_count = int((edit_mask & active_mask).sum().item())
        active_count = int(active_mask.sum().item())
        editable_frac = editable_count / max(active_count, 1)
        if editable_frac <= 0.15:
            logger.info(
                'Hybrid decoder-seed preservation | editable=%s active=%s frac=%.3f | skipping continuous hybrid and using discrete repair only.',
                editable_count,
                active_count,
                editable_frac,
            )
            final_ids = init_ids.detach().clone()
            final_ids = late_discrete_edit_refine_ids(
                args,
                model_wrapper,
                lm,
                true_grads,
                res_ids,
                final_ids,
                attention_mask,
                true_labels,
                edit_mask,
            )
            final_ids = sequence_edit_beam_refine_ids(
                args,
                model_wrapper,
                lm,
                true_grads,
                res_ids,
                final_ids,
                attention_mask,
                true_labels,
                edit_mask,
            )
            if lm is not None and original_lm_device is not None and str(original_lm_device) != str(args.device):
                lm.to(original_lm_device)
                cleanup_memory()
            return final_ids

    x_embeds = init_embeddings(args, model_wrapper, res_ids, input_ids, attention_mask, init_ids=init_ids)
    frozen_mask = None
    frozen_values = None
    if edit_mask is not None and args.batch_size <= 16:
        frozen_mask = (~edit_mask & attention_mask.bool()).unsqueeze(-1)
        if bool(frozen_mask.any().item()):
            frozen_values = x_embeds.detach().clone()
            logger.info(
                'Hybrid low-batch safeguard | freezing decoder-seeded positions during continuous refinement | frozen=%s editable=%s',
                int(frozen_mask.sum().item()),
                int(edit_mask.sum().item()),
            )
    optimizer = torch.optim.Adam([x_embeds], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.lr_decay)

    best_loss, best_x = None, x_embeds.detach().clone()
    best_projected_loss, best_projected_ids = None, None
    projected_stall = 0
    pad_embed = emb_matrix[model_wrapper.pad_token]

    if init_ids is not None and _hybrid_uses_candidate_projection(args):
        best_loss = compute_rec_loss_from_ids(
            args,
            model_wrapper,
            true_grads,
            init_ids,
            attention_mask,
            true_labels,
        ).item()
        logger.info('Hybrid DAGER decoder init rec_loss=%.6f', best_loss)
        best_projected_loss = best_loss
        best_projected_ids = init_ids.detach().clone()

    for step in range(args.n_steps):
        dummy_grads = model_wrapper.compute_grads_from_embeds(x_embeds, true_labels, attention_mask=attention_mask,
                                                              create_graph=True, )
        rec_loss = grad_match_loss(dummy_grads, true_grads, args)
        if rec_loss.device != x_embeds.device:
            rec_loss = rec_loss.to(x_embeds.device)
        reg_loss = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
        lm_loss = fuzzy_gpt2_lm_loss(
            lm,
            x_embeds,
            emb_norm,
            lm_emb_matrix,
            candidate_mask,
            attention_mask,
            args.hybrid_temperature,
        )
        total_loss = rec_loss + args.coeff_reg * reg_loss + args.coeff_perplexity * lm_loss

        optimizer.zero_grad()
        total_loss.backward()
        if frozen_mask is not None and x_embeds.grad is not None:
            x_embeds.grad.masked_fill_(frozen_mask, 0.0)
        optimizer.step()
        scheduler.step()

        should_log = (step + 1) == 1 or (step + 1) % args.print_every == 0 or (step + 1) == args.n_steps
        projected_ids_for_eval = None
        did_periodic_projection = False

        with torch.no_grad():
            if frozen_mask is not None and frozen_values is not None:
                x_embeds.copy_(torch.where(frozen_mask, frozen_values, x_embeds))
            x_embeds.copy_(_with_pad_embeddings(x_embeds, pad_mask, pad_embed))
            if _hybrid_uses_periodic_projection(args) and (step + 1) % args.hybrid_project_every == 0:
                projected_ids_for_eval = project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask,
                                                               model_wrapper.pad_token, emb_norm=emb_norm)
                x_embeds.copy_(emb_matrix[projected_ids_for_eval])
                if frozen_mask is not None and frozen_values is not None:
                    x_embeds.copy_(torch.where(frozen_mask, frozen_values, x_embeds))
                x_embeds.copy_(_with_pad_embeddings(x_embeds, pad_mask, pad_embed))
                did_periodic_projection = True
            rec_loss_value = rec_loss.item()
            if best_loss is None or rec_loss_value < best_loss:
                best_loss = rec_loss_value
                best_x = x_embeds.detach().clone()

        projected_rec_loss = None
        should_eval_projection = (
                _hybrid_uses_candidate_projection(args)
                and (did_periodic_projection or should_log or (step + 1) == args.n_steps)
        )
        if should_eval_projection:
            if projected_ids_for_eval is None:
                with torch.no_grad():
                    projected_ids_for_eval = project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask,
                                                                   model_wrapper.pad_token, emb_norm=emb_norm)
            projected_rec_loss = compute_rec_loss_from_ids(
                args,
                model_wrapper,
                true_grads,
                projected_ids_for_eval,
                attention_mask,
                true_labels,
            ).item()
            if best_projected_loss is None or projected_rec_loss < best_projected_loss:
                best_projected_loss = projected_rec_loss
                best_projected_ids = projected_ids_for_eval.detach().clone()
                projected_stall = 0
                logger.info(
                    'Hybrid projected best updated | step=%d/%d | proj_rec_loss=%.6f',
                    step + 1,
                    args.n_steps,
                    best_projected_loss,
                )
            else:
                projected_stall += 1

        if should_log:
            logger.info(
                'Step %d/%d | rec_loss=%.6f | proj_rec_loss=%.6f | reg_loss=%.6f | lm_loss=%.6f | total=%.6f',
                step + 1,
                args.n_steps,
                rec_loss.item(),
                projected_rec_loss if projected_rec_loss is not None else float('nan'),
                reg_loss.item(),
                lm_loss.item(),
                total_loss.item(),
            )
        del dummy_grads, rec_loss, reg_loss, lm_loss, total_loss
        if should_log:
            cleanup_memory()
        if (
                init_ids is not None
                and _hybrid_uses_candidate_projection(args)
                and projected_stall >= (2 if args.batch_size <= 32 else 4)
                and (step + 1) >= min(args.n_steps, max(20, args.hybrid_project_every * 2))
        ):
            logger.info(
                'Hybrid projected-loss stall detected | stall=%s | best_proj=%.6f | stopping continuous refinement early.',
                projected_stall,
                best_projected_loss if best_projected_loss is not None else float('nan'),
            )
            break

    if best_projected_ids is not None:
        final_ids = best_projected_ids
    elif _hybrid_uses_candidate_projection(args):
        final_ids = project_to_candidates(best_x, emb_matrix, candidate_mask, pad_mask, model_wrapper.pad_token,
                                          emb_norm=emb_norm)
    else:
        final_ids = project_to_vocab(best_x, emb_matrix, pad_mask, model_wrapper.pad_token, emb_norm=emb_norm)

    if edit_mask is not None:
        final_ids = late_discrete_edit_refine_ids(
            args,
            model_wrapper,
            lm,
            true_grads,
            res_ids,
            final_ids,
            attention_mask,
            true_labels,
            edit_mask,
        )
        final_ids = sequence_edit_beam_refine_ids(
            args,
            model_wrapper,
            lm,
            true_grads,
            res_ids,
            final_ids,
            attention_mask,
            true_labels,
            edit_mask,
        )

    if lm is not None and original_lm_device is not None and str(original_lm_device) != str(args.device):
        lm.to(original_lm_device)
        cleanup_memory()

    return final_ids


def iterative_dager_lamp_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels, init_ids=None,
                                  edit_mask=None):
    attention_mask = orig_batch['attention_mask'].to(args.device)
    refresh_mask = None if edit_mask is None else (~edit_mask).to(args.device)
    rounds = max(1, int(args.iterative_rounds))
    original_n_steps = args.n_steps
    steps_per_round = int(args.iterative_steps_per_round)
    if steps_per_round <= 0:
        steps_per_round = max(1, int(np.ceil(original_n_steps / rounds)))

    curr_res_ids = [list(pos_ids) for pos_ids in res_ids]
    curr_ids = init_ids
    best_ids = None
    best_loss = None
    stall_rounds = 0

    if curr_ids is not None:
        best_loss = compute_rec_loss_from_ids(
            args,
            model_wrapper,
            true_grads,
            curr_ids,
            attention_mask,
            true_labels,
        ).item()
        best_ids = curr_ids.detach().clone()
        logger.info('Iterative DAGER-LAMP seed projected rec_loss=%.6f', best_loss)
        logger.info('Iterative DAGER-LAMP seed sample=%s', _decode_ids_for_log(model_wrapper, best_ids))
    else:
        logger.info('Iterative DAGER-LAMP starting without DAGER decoder seed; first round uses candidate-random init.')

    editable_frac = 1.0
    if edit_mask is not None and attention_mask.numel() > 0:
        active_mask = attention_mask.bool()
        active_count = int(active_mask.sum().item())
        editable_count = int((edit_mask & active_mask).sum().item())
        editable_frac = editable_count / max(active_count, 1)
        logger.info(
            'Iterative DAGER-LAMP edit coverage | editable=%s active=%s frac=%.3f',
            editable_count,
            active_count,
            editable_frac,
        )
        if args.batch_size <= 16 and init_ids is not None and editable_frac <= 0.25:
            logger.info(
                'Iterative DAGER-LAMP low-batch safeguard: decoder seed already covers %.1f%% of active tokens; '
                'skipping iterative refinement and keeping seed plus late discrete edits only.',
                100.0 * (1.0 - editable_frac),
            )
            return init_ids

    try:
        args.n_steps = steps_per_round
        original_force_noise = getattr(args, '_iterative_force_noise', False)
        for round_idx in range(rounds):
            logger.info(
                'Iterative DAGER-LAMP round %d/%d | steps=%d | init=%s',
                round_idx + 1,
                rounds,
                steps_per_round,
                'projected_best' if curr_ids is not None else 'candidate_random',
            )
            round_ids = hybrid_optimize(
                args,
                model_wrapper,
                lm,
                true_grads,
                curr_res_ids,
                orig_batch,
                true_labels,
                init_ids=curr_ids,
                edit_mask=edit_mask,
            )
            round_loss = compute_rec_loss_from_ids(
                args,
                model_wrapper,
                true_grads,
                round_ids,
                attention_mask,
                true_labels,
            ).item()

            improvement = float('inf') if best_loss is None else best_loss - round_loss
            accepted = best_loss is None or improvement > args.iterative_accept_margin
            logger.info(
                'Iterative DAGER-LAMP round %d projected rec_loss=%.6f | best=%.6f | improvement=%.6f | accepted=%s',
                round_idx + 1,
                round_loss,
                best_loss if best_loss is not None else float('nan'),
                improvement,
                accepted,
            )
            logger.info('Iterative DAGER-LAMP round %d sample=%s', round_idx + 1,
                        _decode_ids_for_log(model_wrapper, round_ids))

            if accepted:
                best_loss = round_loss
                best_ids = round_ids.detach().clone()
                curr_ids = best_ids
                stall_rounds = 0
                if args.iterative_refresh_candidates:
                    curr_res_ids = _refresh_res_ids_from_projected_ids(
                        curr_res_ids,
                        best_ids,
                        attention_mask,
                        model_wrapper.pad_token,
                        refresh_mask=refresh_mask,
                    )
                    logger.info(
                        'Iterative DAGER-LAMP refreshed DAGER candidate lists from accepted projection%s.',
                        '' if refresh_mask is None else ' (decoder-seeded positions only)',
                    )
            else:
                curr_ids = best_ids
                stall_rounds += 1
                if args.batch_size <= 16:
                    logger.info(
                        'Iterative DAGER-LAMP low-batch safeguard: stopping after rejected round %d with no projected improvement.',
                        round_idx + 1,
                    )
                    break
                if stall_rounds >= max(1, args.iterative_stall_patience):
                    logger.info(
                        'Iterative DAGER-LAMP stalled for %d round(s); continuing with noisy projected-best restart.',
                        stall_rounds,
                    )
                    setattr(args, '_iterative_force_noise', True)
                    stall_rounds = 0
    finally:
        args.n_steps = original_n_steps
        setattr(args, '_iterative_force_noise', original_force_noise)

    if best_ids is None:
        logger.info('Iterative DAGER-LAMP found no accepted projection; falling back to one hybrid pass.')
        return hybrid_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels, init_ids=init_ids,
                               edit_mask=edit_mask)

    logger.info('Iterative DAGER-LAMP selected projected rec_loss=%.6f', best_loss)
    return best_ids


def reconstruct(args, sample, metric, model_wrapper, lm):
    tokenizer = model_wrapper.tokenizer
    sequences, true_labels = sample
    orig_batch = tokenizer(sequences, padding=True, truncation=True,
                           max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
                           return_tensors='pt', ).to(args.device)

    true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape, device=grad.device) * args.defense_noise

    with torch.no_grad():
        B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        if B is None:
            reference = [
                remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad == 'left'))
                for i in range(orig_batch['input_ids'].shape[0])
            ]
            return ['' for _ in reference], reference
        logger.info('Hybrid DAGER rank: %s', B)
        max_l1_positions = orig_batch['input_ids'].shape[1]
        log_length_signal(model_wrapper, true_grads, orig_batch)
        _, res_ids, _, _ = filter_l1(args, model_wrapper, R_Qs, max_positions=max_l1_positions)
        res_ids = augment_res_ids_with_true_grad_support(
            args,
            model_wrapper,
            true_grads,
            res_ids,
            max_l1_positions,
        )

    if len(res_ids) == 0:
        reference = [
            remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad == 'left'))
            for i in range(orig_batch['input_ids'].shape[0])
        ]
        return ['' for _ in reference], reference

    reference = [
        remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad == 'left'))
        for i in range(orig_batch['input_ids'].shape[0])
    ]
    init_ids = None
    edit_mask = None
    if args.hybrid_init_mode == 'dager':
        with torch.no_grad():
            init_ids, edit_mask = select_dager_decoder_ids(
                args,
                model_wrapper,
                R_Qs,
                res_ids,
                orig_batch,
                true_grads=true_grads,
                lm=lm,
            )
    del R_Qs
    cleanup_memory()
    if args.iterative_dager_lamp:
        final_ids = iterative_dager_lamp_optimize(
            args,
            model_wrapper,
            lm,
            true_grads,
            res_ids,
            orig_batch,
            true_labels,
            init_ids=init_ids,
            edit_mask=edit_mask,
        )
    else:
        final_ids = hybrid_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels,
                                    init_ids=init_ids, edit_mask=edit_mask)
    prediction = [
        remove_padding(tokenizer, final_ids[i], left=(args.pad == 'left'))
        for i in range(final_ids.shape[0])
    ]
    if model_wrapper.is_decoder():
        prediction = reorder_decoder_predictions(args, tokenizer, prediction, reference)
    return prediction[:len(reference)], reference


def main():
    if args.task != 'seq_class':
        raise NotImplementedError('Hybrid attack currently supports --task seq_class.')
    print(f"Hash Value {job_hash} Started")
    attack_prefix = "iterative_dager_lamp" if args.iterative_dager_lamp else "hybrid"
    if args.preprocess_unique_canary_markers:
        attack_prefix = f"{attack_prefix}_canary"
    attack_name = f"{attack_prefix}_{args.loss}"
    is_complete, results_dir = is_attack_complete(attack_name, job_hash)
    if not log_claim_acquired:
        logger.info(f"Skipping hash {job_hash} because another job currently owns the primary log file for this run.")
        print(f"Hash Value {job_hash} Skipped (locked)")
        return
    if is_complete:
        logger.info(f"Results already exist for this config at {results_dir} skipping attack.", )
        logger.info('Done with all.')
        print(f"Hash Value {job_hash} is already done")
        return

    device = torch.device(args.device)
    metric = load_rouge_metric(cache_dir=args.cache_dir, logger=logger)
    dataset = TextDataset(device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir,
                          use_hf_split=args.use_hf_split,
                          preprocess_numbered_markers=args.preprocess_numbered_markers,
                          preprocess_boundary_markers=args.preprocess_boundary_markers,
                          preprocess_unique_canary_markers=args.preprocess_unique_canary_markers,
                          canary_marker_prefix=args.canary_marker_prefix)
    model_wrapper = ModelWrapper(args)
    wrapper_tokenizer = model_wrapper.tokenizer
    lm = load_lm_prior(args)

    logger.info('\n\nAttacking with hybrid optimization..\n')
    predictions, references = [], []
    final_results = []
    final_per_input_results = []
    input_times = []
    sentence_rows = []
    input_rows = []
    results_dir = get_results_dir(attack_name, job_hash)
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
        logger.info("Resuming from partial aresults in %s | completed_inputs=%s | last_input=%s",
            results_dir, len(partial_state["completed_inputs"]), max(partial_state["completed_inputs"]),)
    t_start = time.time()
    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
        if i in partial_state["completed_inputs"]:
            logger.info(f"Skipping already completed input #{i}.")
            continue
        t_input_start = time.time()
        sample = dataset[i]  # (seqs, labels)

        logger.info(f'Running input #{i} of {args.n_inputs}.')
        if args.neptune:
            args.neptune['logs/curr_input'].log(i)

        logger.info('reference: ')
        for seq in sample[0]:
            logger.info('========================')
            logger.info(seq)

        logger.info('========================')

        prediction, reference = reconstruct(args, sample, metric, model_wrapper, lm)
        predictions += prediction
        references += reference

        logger.info(f'Done with input #{i} of {args.n_inputs}.')
        logger.info('reference: ')
        curr_metrics = []
        for sent_idx, (ref, pred) in enumerate(zip(reference, prediction)):
            logger.info('========================')
            logger.info(f"Reference: {ref}")
            logger.info(f"Prediction: {pred}")
            metrics = evaluate_prediction(pred, ref, wrapper_tokenizer, metric)
            metrics = maybe_add_canary_audit_metrics(
                metrics, pred, ref, wrapper_tokenizer, metric,
                enabled=args.preprocess_unique_canary_markers,
                canary_prefix=args.canary_marker_prefix,
            )
            curr_metrics.append(metrics)
            sentence_rows.append({
                "attack": attack_name,
                "model": args.model_path,
                "dataset": args.dataset,
                "input_index": i,
                "sentence_index": sent_idx,
                "reference": ref,
                "prediction": pred,
                **metrics})
        logger.info('========================')
        summary = summarize_metrics(curr_metrics)
        input_canary_means = extract_canary_metric_means(summary)
        final_results.extend(curr_metrics)
        logger.info('[Curr input metrics]:')
        logger.info(f"{print_summary_table(summary)}")
        logger.info('[Aggregate metrics]:')
        aggregated_results = _safe_aggregated_metrics(prediction, reference, wrapper_tokenizer, metric,
                                                      curr_metrics, f"input #{i}")
        aggregated_results.update(input_canary_means)

        final_per_input_results.append(aggregated_results)
        logger.info(f"{print_single_metric_dict(aggregated_results)}")
        input_time_sec = time.time() - t_input_start
        total_time_sec = time.time() - t_start
        input_time = str(datetime.timedelta(seconds=input_time_sec)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=total_time_sec)).split(".")[0]
        logger.info(f'input #{i} time: {input_time} | total time: {total_time}')
        input_rows.append({
            "attack": attack_name,
            "model": args.model_path,
            "dataset": args.dataset,
            "input_index": i,
            "num_sentences": len(reference),
            "joined_reference": " ".join(reference),
            "joined_prediction": " ".join(prediction),
            "reconstruction_time_sec": input_time_sec,
            **aggregated_results})
        input_times.append(input_time_sec)
        partial_summary = {
            "Arguments": args_to_dict(args),
            "completed_inputs": len(input_rows),
            "last_completed_input": i,
        }
        write_attack_artifacts(results_dir, sentence_rows, input_rows, partial_summary, status="incomplete")
        del sample, prediction, reference, curr_metrics, aggregated_results
        cleanup_memory()
        input_time_sec = time.time() - t_input_start
        total_time_sec = time.time() - t_start
        input_time = str(datetime.timedelta(seconds=input_time_sec)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=total_time_sec)).split(".")[0]
        logger.info(f'Saving input #{i} time: {input_time} | total time: {total_time}')
        logger.info("")
        logger.info("")
    logger.info('[Aggregate metrics]:')
    total_time_sec = time.time() - t_start
    aggregated_results = _safe_aggregated_metrics(predictions, references, wrapper_tokenizer, metric,
                                                  final_per_input_results, "full experiment")
    summary = summarize_metrics(final_results)
    aggregated_results.update(extract_canary_metric_means(summary))
    aggregated_results[f"experiment_time_mean"] = float(total_time_sec)
    aggregated_results[f"experiment_time_std"] = float(0)
    logger.info(f"Overall {print_single_metric_dict(aggregated_results)}")
    summary[f"reconstruction_time_mean"] = float(np.mean(input_times))
    summary[f"reconstruction_time_std"] = float(np.std(input_times))
    logger.info(f"Per Sentence{print_summary_table(summary)}")
    summary_per_input = summarize_metrics(final_per_input_results)
    summary_per_input[f"reconstruction_time_mean"] = float(np.mean(input_times))
    summary_per_input[f"reconstruction_time_std"] = float(np.std(input_times))
    logger.info(f"Per Input Results {print_summary_table(summary_per_input)}")

    summary_results = {"Overall Results": aggregated_results, "Per Sentence Results": summary,
                       "Per Input Results": summary_per_input, "Arguments": args_to_dict(args)}
    canary_summary = extract_canary_metric_summary(summary)
    if canary_summary:
        summary_results["Canary Audit Results"] = canary_summary
    logger.info(f"Experiment time {total_time_sec}")
    write_attack_artifacts(results_dir, sentence_rows, input_rows, summary_results, status="complete")
    logger.info('Done with all.')
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)
    print(f"Hash Value {job_hash} Done")


if __name__ == '__main__':
    try:
        main()
    finally:
        release_all_log_locks()
