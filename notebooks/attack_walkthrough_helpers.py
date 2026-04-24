import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from args_factory import get_args
from utils.experiment import cleanup_memory, load_rouge_metric
from utils.filtering_decoder import filter_decoder
from utils.functional import (
    check_if_in_span,
    fallback_gpt2_l1_candidates,
    fallback_rope_l1_candidates,
    filter_outliers,
    get_span_dists,
    get_top_B_in_span,
    remove_padding,
)
from utils.models import ModelWrapper, _resolve_local_model_path
from utils.somp_core import (
    _cluster_candidates,
    build_global_candidate_pool,
    ensure_somp_args,
    position_filter_per_sample,
    reconstruct_with_omp,
    beam_search_decoder,
)
from utils.somp_models import SOMPModelWrapper


SMALL_SENTENCES = [
    "bad.",
    "a clever little movie with heart.",
]

LONG_SENTENCES = [
    "the film starts with a familiar premise, but it slowly opens into a much stranger and more emotionally precise story about grief, guilt, and the way memory edits the people we love.",
    "although the performances are committed and the direction is often elegant, the screenplay keeps circling the same conflict until the final act feels less like a revelation and more like a graceful surrender.",
]


def default_cache_dir():
    if os.environ.get("CACHE_DIR"):
        return os.environ["CACHE_DIR"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "gia_exp_cache")
    return str(REPO_ROOT / "models_cache" / "gia_exp_cache")


def build_attack_args(
    *,
    dataset="sst2",
    batch_size=2,
    model_path="gpt2",
    split="val",
    n_inputs=1,
    rng_seed=42,
    cache_dir=None,
    extra_args=None,
):
    cache_dir = cache_dir or default_cache_dir()
    argv = [
        "--dataset",
        dataset,
        "--split",
        split,
        "--n_inputs",
        str(n_inputs),
        "--batch_size",
        str(batch_size),
        "--l1_filter",
        "all",
        "--l2_filter",
        "non-overlap",
        "--model_path",
        model_path,
        "--device",
        "auto",
        "--task",
        "seq_class",
        "--cache_dir",
        cache_dir,
        "--rng_seed",
        str(rng_seed),
    ]
    if dataset in {"sst2", "cola", "rte"}:
        argv.append("--use_hf_split")
    if extra_args:
        argv.extend(extra_args)
    return get_args(argv)


def make_labels(batch_size, device, label=0):
    if isinstance(label, (list, tuple)):
        values = list(label)
    else:
        values = [label] * batch_size
    return torch.tensor([values], device=device)


def tokenize_sequences(model_wrapper, sequences, args):
    tokenizer = model_wrapper.tokenizer
    return tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)


def dager_filter_l1_trace(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
    sentence_ends = []
    rows = []
    p = 0

    while True:
        embeds = model_wrapper.get_embeddings(p)
        distance_values = None
        if model_wrapper.is_bert():
            _, res_ids_new, res_types_new = get_top_B_in_span(
                R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm
            )
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

        ids = res_ids_new.tolist()
        token_preview = model_wrapper.tokenizer.convert_ids_to_tokens(ids[: min(12, len(ids))])
        rows.append(
            {
                "position": p,
                "n_candidates": len(ids),
                "preview_ids": ids[: min(12, len(ids))],
                "preview_tokens": token_preview,
            }
        )

        res_pos_new = torch.ones_like(res_ids_new) * p
        del embeds

        res_types += [res_types_new.tolist()]
        if len(ids) == 0 or p > tokenizer.model_max_length or p > args.max_len:
            break
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][end_token_ind]
            sentence_ends.append((p, sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind + 1 :]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind + 1 :]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        p += 1
        if model_wrapper.has_rope():
            break

    return {
        "res_pos": res_pos,
        "res_ids": res_ids,
        "res_types": res_types,
        "sentence_ends": sentence_ends,
        "candidate_counts": pd.DataFrame(rows),
    }


def preview_sentences(tokenizer, sequences):
    rows = []
    for idx, seq in enumerate(sequences):
        rows.append({"index": idx, "token_ids": seq, "decoded": tokenizer.decode(seq)})
    return pd.DataFrame(rows)


def trace_dager_attack(args, sequences, labels=None):
    model_wrapper = ModelWrapper(args)
    tokenizer = model_wrapper.tokenizer
    labels = labels if labels is not None else make_labels(len(sequences), args.device)
    orig_batch = tokenize_sequences(model_wrapper, sequences, args)
    true_grads = model_wrapper.compute_grads(orig_batch, labels)
    rank, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)

    l1 = dager_filter_l1_trace(args, model_wrapper, R_Qs)
    decoder_candidates = pd.DataFrame()
    if model_wrapper.is_decoder() and l1["res_ids"]:
        max_ids = -1
        for ids in l1["res_ids"]:
            if len(ids) > args.max_ids:
                max_ids = args.max_ids
        predicted_sentences, predicted_scores, fallback_sentences, fallback_scores = filter_decoder(
            args, model_wrapper, R_Qs, l1["res_ids"], max_ids=max_ids
        )
        rows = []
        for seq, score in zip(predicted_sentences, predicted_scores):
            rows.append({"kind": "accepted", "score": float(score), "token_ids": seq, "decoded": tokenizer.decode(seq)})
        for seq, score in zip(fallback_sentences, fallback_scores):
            if not seq:
                continue
            rows.append({"kind": "fallback", "score": float(score), "token_ids": seq, "decoded": tokenizer.decode(seq)})
        decoder_candidates = pd.DataFrame(rows).sort_values(by="score", na_position="last").reset_index(drop=True)

    references = [
        remove_padding(tokenizer, orig_batch["input_ids"][i], left=(args.pad == "left"))
        for i in range(orig_batch["input_ids"].shape[0])
    ]
    result = {
        "model_wrapper": model_wrapper,
        "tokenizer": tokenizer,
        "orig_batch": orig_batch,
        "labels": labels,
        "true_grads": true_grads,
        "rank": rank,
        "R_Qs": R_Qs,
        "references": references,
        "reference_table": preview_sentences(tokenizer, references),
        "l1_trace": l1,
        "decoder_candidates": decoder_candidates,
    }
    return result


def grad_match_loss(dummy_grads, true_grads, args):
    loss = None
    n_grads = 0
    for dummy_grad, true_grad in zip(dummy_grads, true_grads):
        if dummy_grad is None or true_grad is None:
            continue
        true_grad = true_grad.to(dummy_grad.device)
        curr = 1.0 - (dummy_grad * true_grad).sum() / (
            dummy_grad.reshape(-1).norm(p=2) * true_grad.reshape(-1).norm(p=2) + 1e-9
        )
        loss = curr if loss is None else loss + curr
        n_grads += 1
    if loss is None:
        raise RuntimeError("No comparable gradients were produced.")
    return loss / max(n_grads, 1)


def build_candidate_mask(res_ids, seq_len, vocab_size, pad_token, device):
    candidate_mask = torch.zeros(seq_len, vocab_size, dtype=torch.bool, device=device)
    for pos in range(seq_len):
        if pos < len(res_ids) and len(res_ids[pos]) > 0:
            ids = torch.tensor(res_ids[pos], dtype=torch.long, device=device)
            candidate_mask[pos, ids] = True
        else:
            candidate_mask[pos, pad_token] = True
    return candidate_mask


def project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask, pad_token, emb_norm=None):
    if emb_norm is None:
        emb_norm = F.normalize(emb_matrix, dim=-1)
    x_norm = F.normalize(x_embeds.detach(), dim=-1)
    ids = torch.full((x_embeds.shape[0], x_embeds.shape[1]), pad_token, dtype=torch.long, device=x_embeds.device)
    for pos in range(x_embeds.shape[1]):
        valid_ids = torch.where(candidate_mask[pos])[0]
        sims = x_norm[:, pos] @ emb_norm[valid_ids].T
        ids[:, pos] = valid_ids[sims.argmax(dim=-1)]
    ids[pad_mask] = pad_token
    return ids


def init_embeddings_from_candidates(args, model_wrapper, res_ids, input_ids, attention_mask, init_ids=None):
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
    if pad_mask.any():
        x_embeds[pad_mask] = emb_matrix[model_wrapper.pad_token]
    if args.hybrid_init_noise > 0 and init_ids is None:
        x_embeds = x_embeds + args.hybrid_init_noise * torch.randn_like(x_embeds)
        if pad_mask.any():
            x_embeds[pad_mask] = emb_matrix[model_wrapper.pad_token]
    return x_embeds.detach().requires_grad_(True)


def fuzzy_gpt2_lm_loss(lm, x_embeds, emb_norm, lm_emb_matrix, candidate_mask, attention_mask, temperature):
    if lm is None:
        return x_embeds.sum() * 0.0

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

    transformer = lm.transformer if hasattr(lm, "transformer") else lm.base_model
    hidden_states = transformer(
        inputs_embeds=lm_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )[0]

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


def load_hybrid_lm_prior(args):
    if args.hybrid_use_lm_prior != "true" or args.coeff_perplexity <= 0:
        return None
    if args.model_path not in ["gpt2", "openai-community/gpt2-large"]:
        return None
    lm_source = _resolve_local_model_path(args.model_path, args.cache_dir)
    lm_kwargs = {
        "pretrained_model_name_or_path": lm_source,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if args.cache_dir is not None and not os.path.isdir(lm_source):
        lm_kwargs["cache_dir"] = args.cache_dir
    if os.path.isdir(lm_source):
        lm_kwargs["local_files_only"] = True
    lm = AutoModelForCausalLM.from_pretrained(**lm_kwargs)
    lm.config.pad_token_id = lm.config.eos_token_id
    lm.config.use_cache = False
    lm.eval()
    for param in lm.parameters():
        param.requires_grad_(False)
    return lm.to(args.device)


def select_dager_decoder_ids(args, model_wrapper, R_Qs, res_ids, orig_batch):
    max_ids = -1
    for pos_ids in res_ids:
        if len(pos_ids) > args.max_ids:
            max_ids = args.max_ids
    predicted_sentences, predicted_scores, top_B_incorrect_sentences, top_B_incorrect_scores = filter_decoder(
        args, model_wrapper, R_Qs, res_ids, max_ids=max_ids
    )
    if len(predicted_sentences) < orig_batch["input_ids"].shape[0]:
        predicted_sentences += top_B_incorrect_sentences
        predicted_scores += top_B_incorrect_scores
    if len(predicted_sentences) == 0:
        return None

    correct_sentences = []
    approx_sentences = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    max_len = max(len(sentence) for sentence in predicted_sentences)
    for sentence, score in zip(predicted_sentences, predicted_scores):
        if score < model_wrapper.effective_l2_span_thresh(args.l2_span_thresh):
            correct_sentences.append(sentence)
        else:
            approx_sentences.append(sentence)
            approx_sentences_ext.append(sentence + [-1] * (max_len - len(sentence)))
            approx_sentences_lens.append(len(sentence))
            approx_scores.append(score)

    selected_sentences = correct_sentences.copy()
    if len(approx_sentences) > 0 and len(selected_sentences) < args.batch_size:
        approx_scores = torch.tensor(approx_scores)
        approx_sentences_lens = torch.tensor(approx_sentences_lens)
        approx_sentences_ext_tensor = torch.tensor(approx_sentences_ext)

        for sentence in correct_sentences:
            similar = (
                (torch.tensor(sentence) == approx_sentences_ext_tensor[:, : len(sentence)]).sum(1)
                >= torch.min(approx_sentences_lens, torch.tensor(len(sentence))) * args.distinct_thresh
            )
            approx_scores[similar] = torch.inf

        for _ in range(len(selected_sentences), args.batch_size):
            idx = torch.argmin(approx_scores)
            if torch.isinf(approx_scores[idx]):
                break
            selected_sentences.append(approx_sentences[idx])
            similar = (
                (torch.tensor(approx_sentences_ext[idx]) == approx_sentences_ext_tensor).sum(1)
                >= max_len * args.distinct_thresh
            )
            approx_scores[similar] = torch.inf

    selected_sentences = selected_sentences[: orig_batch["input_ids"].shape[0]]
    if len(selected_sentences) == 0:
        return None

    seq_len = orig_batch["input_ids"].shape[1]
    decoded_ids = torch.full(
        (orig_batch["input_ids"].shape[0], seq_len),
        model_wrapper.pad_token,
        dtype=torch.long,
        device=args.device,
    )
    for row, sentence in enumerate(selected_sentences):
        sentence = sentence[:seq_len]
        decoded_ids[row, : len(sentence)] = torch.tensor(sentence, dtype=torch.long, device=args.device)
    return decoded_ids


def trace_hybrid_attack(args, sequences, labels=None, traced_steps=8):
    model_wrapper = ModelWrapper(args)
    tokenizer = model_wrapper.tokenizer
    labels = labels if labels is not None else make_labels(len(sequences), args.device)
    orig_batch = tokenize_sequences(model_wrapper, sequences, args)
    true_grads = model_wrapper.compute_grads(orig_batch, labels)
    rank, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
    l1 = dager_filter_l1_trace(args, model_wrapper, R_Qs)

    init_ids = None
    if l1["res_ids"] and args.hybrid_init_mode == "dager" and model_wrapper.is_decoder():
        init_ids = select_dager_decoder_ids(args, model_wrapper, R_Qs, l1["res_ids"], orig_batch)

    lm = load_hybrid_lm_prior(args)
    emb_matrix = model_wrapper.get_input_embeddings_weight().detach().to(args.device)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    lm_emb_matrix = lm.get_input_embeddings().weight.detach() if lm is not None else None
    input_ids = orig_batch["input_ids"].to(args.device)
    attention_mask = orig_batch["attention_mask"].to(args.device)
    pad_mask = attention_mask == 0
    candidate_mask = build_candidate_mask(l1["res_ids"], input_ids.shape[1], emb_matrix.shape[0], model_wrapper.pad_token, args.device)
    x_embeds = init_embeddings_from_candidates(args, model_wrapper, l1["res_ids"], input_ids, attention_mask, init_ids=init_ids)
    optimizer = torch.optim.Adam([x_embeds], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.lr_decay)
    history = []

    for step in range(min(traced_steps, args.n_steps)):
        dummy_grads = model_wrapper.compute_grads_from_embeds(
            x_embeds, labels, attention_mask=attention_mask, create_graph=True
        )
        rec_loss = grad_match_loss(dummy_grads, true_grads, args)
        reg_loss = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
        lm_loss = fuzzy_gpt2_lm_loss(
            lm, x_embeds, emb_norm, lm_emb_matrix, candidate_mask, attention_mask, args.hybrid_temperature
        )
        total_loss = rec_loss + args.coeff_reg * reg_loss + args.coeff_perplexity * lm_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            projected_ids = project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask, model_wrapper.pad_token, emb_norm)
            projected_text = [remove_padding(tokenizer, projected_ids[i], left=(args.pad == "left")) for i in range(projected_ids.shape[0])]

        history.append(
            {
                "step": step + 1,
                "rec_loss": float(rec_loss.item()),
                "reg_loss": float(reg_loss.item()),
                "lm_loss": float(lm_loss.item()),
                "total_loss": float(total_loss.item()),
                "projected_preview": projected_text[0] if projected_text else "",
            }
        )
        del dummy_grads, rec_loss, reg_loss, lm_loss, total_loss
        cleanup_memory()

    if lm is not None:
        del lm
    return {
        "model_wrapper": model_wrapper,
        "tokenizer": tokenizer,
        "rank": rank,
        "l1_trace": l1,
        "init_ids": init_ids,
        "init_table": preview_sentences(tokenizer, init_ids.tolist()) if init_ids is not None else pd.DataFrame(),
        "step_history": pd.DataFrame(history),
        "reference_table": preview_sentences(
            tokenizer,
            [remove_padding(tokenizer, orig_batch["input_ids"][i], left=(args.pad == "left")) for i in range(orig_batch["input_ids"].shape[0])],
        ),
    }


def trace_somp_attack(args, sequences, labels=None):
    args = ensure_somp_args(args)
    metric = load_rouge_metric(cache_dir=args.cache_dir)
    model_wrapper = SOMPModelWrapper(args)
    tokenizer = model_wrapper.tokenizer
    labels = labels if labels is not None else make_labels(len(sequences), args.device)
    orig_batch = tokenize_sequences(model_wrapper, sequences, args)

    raw_mixed_grads = model_wrapper.compute_grads(orig_batch, labels)
    mixed_grads = [None if grad is None else grad.detach().cpu() for grad in raw_mixed_grads]
    del raw_mixed_grads
    cleanup_memory()

    rank, R_Qs, head_R_Qs = model_wrapper.get_matrices_expansions(mixed_grads, B=None, tol=args.rank_tol)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model_wrapper.pad_token
    eff_len_each = (orig_batch["input_ids"] != pad_id).sum(dim=1).cpu()
    candidate_pool_indices = build_global_candidate_pool(args, model_wrapper, mixed_grads, R_Qs, head_R_Qs, eff_len_each)

    length_rows = []
    candidate_pool = []
    unique_texts = set()
    for length in sorted({int(length.item()) for length in eff_len_each}):
        res_ids, _, _ = position_filter_per_sample(args, model_wrapper, R_Qs, candidate_pool_indices, length)
        beam_sequences = beam_search_decoder(args, model_wrapper, R_Qs, res_ids)
        length_rows.append(
            {
                "length": length,
                "positions_with_candidates": len(res_ids),
                "beam_candidates": len(beam_sequences),
            }
        )
        for candidate in beam_sequences:
            candidate = candidate[:length]
            text = tokenizer.decode(candidate, skip_special_tokens=True)
            if text and text not in unique_texts:
                unique_texts.add(text)
                candidate_pool.append(candidate)

    clustered = _cluster_candidates(candidate_pool, tokenizer, metric, float(args.cluster_rouge_l)) if candidate_pool else []
    predictions, references = reconstruct_with_omp(args, (sequences, labels), metric, model_wrapper)
    return {
        "model_wrapper": model_wrapper,
        "tokenizer": tokenizer,
        "rank": rank,
        "candidate_pool_size": int(candidate_pool_indices.numel()),
        "length_table": pd.DataFrame(length_rows),
        "candidate_pool_preview": preview_sentences(tokenizer, candidate_pool[: min(12, len(candidate_pool))]),
        "clustered_preview": preview_sentences(tokenizer, clustered[: min(12, len(clustered))]),
        "predictions": pd.DataFrame({"reference": references, "prediction": predictions}),
    }
