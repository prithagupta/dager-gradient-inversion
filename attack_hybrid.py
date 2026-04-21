import datetime
import json
import os
import sys
import time
import numpy as np
import evaluate
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from args_factory import get_args
from utils.data import TextDataset
from utils.experiment import _repo_root, cleanup_memory
from utils.experiment import setup_experiment_logging
from utils.filtering_decoder import filter_decoder
from utils.functional import (fallback_rope_l1_candidates, get_top_B_in_span, log_distances, remove_padding,
                              filter_outliers, get_span_dists,
                              evaluate_prediction, print_single_metric_dict, summarize_metrics, print_summary_table)
from utils.models import ModelWrapper

args = get_args()
logger, log_path, job_hash = setup_experiment_logging(args, "hybrid_attack")
logger.info(f"Arguments {args}")
logger.info('\n\n\nCommand: %s\n\n\n', ' '.join(sys.argv))

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)




def filter_l1(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
    sentence_ends = []
    p = 0

    while True:
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


def _hybrid_uses_candidate_projection(args):
    return args.hybrid_projection_mode in ['candidate_final', 'candidate_periodic']


def _hybrid_uses_periodic_projection(args):
    return args.hybrid_projection_mode == 'candidate_periodic' and args.hybrid_project_every > 0


def _hybrid_uses_lm_prior(args):
    return args.hybrid_use_lm_prior == 'true' and args.coeff_perplexity > 0


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
    if pad_mask.any():
        x_embeds[pad_mask] = emb_matrix[model_wrapper.pad_token]
    if args.hybrid_init_noise > 0 and init_ids is None:
        x_embeds = x_embeds + args.hybrid_init_noise * torch.randn_like(x_embeds)
        if pad_mask.any():
            x_embeds[pad_mask] = emb_matrix[model_wrapper.pad_token]
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
    if not _hybrid_uses_lm_prior(args):
        return None
    if args.model_path not in ['gpt2', 'openai-community/gpt2-large']:
        logger.info('Hybrid LM prior disabled: fuzzy LM prior is currently implemented for GPT-2 vocabularies only.')
        return None

    lm_kwargs = {'pretrained_model_name_or_path': args.model_path, 'attn_implementation': args.attn_implementation}
    if args.cache_dir is not None:
        lm_kwargs['cache_dir'] = args.cache_dir
    lm = AutoModelForCausalLM.from_pretrained(**lm_kwargs).to(args.device)
    lm.config.pad_token_id = lm.config.eos_token_id
    lm.config.use_cache = False
    lm.config.output_hidden_states = False
    lm.config.output_attentions = False
    lm.eval()
    for param in lm.parameters():
        param.requires_grad_(False)
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


def select_dager_decoder_ids(args, model_wrapper, R_Qs, res_ids, orig_batch):
    if not model_wrapper.is_decoder():
        return None

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
            similar_sentences = ((torch.tensor(sentence) == approx_sentences_ext_tensor[:, :len(sentence)]).sum(1)
                                 >= torch.min(approx_sentences_lens,
                                              torch.tensor(len(sentence))) * args.distinct_thresh)
            approx_scores[similar_sentences] = torch.inf

        for _ in range(len(selected_sentences), args.batch_size):
            idx = torch.argmin(approx_scores)
            if torch.isinf(approx_scores[idx]):
                break
            selected_sentences.append(approx_sentences[idx])
            similar_sentences = ((torch.tensor(approx_sentences_ext[idx]) == approx_sentences_ext_tensor).sum(1)
                                 >= max_len * args.distinct_thresh)
            approx_scores[similar_sentences] = torch.inf

    selected_sentences = selected_sentences[:orig_batch['input_ids'].shape[0]]
    if len(selected_sentences) == 0:
        return None

    selected_sentences = reorder_decoder_sentence_ids(model_wrapper, selected_sentences, orig_batch)

    seq_len = orig_batch['input_ids'].shape[1]
    decoded_ids = torch.full(
        (orig_batch['input_ids'].shape[0], seq_len),
        model_wrapper.pad_token,
        dtype=torch.long,
        device=args.device,
    )
    for row, sentence in enumerate(selected_sentences):
        sentence = sentence[:seq_len]
        decoded_ids[row, :len(sentence)] = torch.tensor(sentence, dtype=torch.long, device=args.device)
    return decoded_ids


def reorder_decoder_sentence_ids(model_wrapper, predicted_sentences, orig_batch):
    reordered_sentences = []
    references = orig_batch['input_ids'].detach().cpu().tolist()
    pad_token = model_wrapper.pad_token

    for reference in references:
        reference = [token for token in reference if token != pad_token]
        best_idx = 0
        best_score = -1
        for idx, sentence in enumerate(predicted_sentences):
            compare_len = max(len(reference), len(sentence))
            ref_ext = reference + [pad_token] * (compare_len - len(reference))
            sent_ext = sentence + [pad_token] * (compare_len - len(sentence))
            score = sum(ref_token == sent_token for ref_token, sent_token in zip(ref_ext, sent_ext))
            if score > best_score:
                best_score = score
                best_idx = idx
        reordered_sentences.append(predicted_sentences[best_idx])

    return reordered_sentences


def reorder_decoder_predictions(args, tokenizer, prediction, reference):
    if len(prediction) == 0:
        return prediction

    new_prediction = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    for ref in reference:
        sequences = [ref] + prediction
        batch = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        best_idx = (batch['input_ids'][1:] == batch['input_ids'][0]).sum(1).argmax()
        new_prediction.append(prediction[best_idx])
    tokenizer.padding_side = old_padding_side
    return new_prediction


def hybrid_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels, init_ids=None):
    emb_matrix = model_wrapper.get_input_embeddings_weight().detach().to(args.device)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    lm_emb_matrix = lm.get_input_embeddings().weight.detach() if lm is not None else None
    input_ids = orig_batch['input_ids'].to(args.device)
    attention_mask = orig_batch['attention_mask'].to(args.device)
    pad_mask = attention_mask == 0
    candidate_mask = build_candidate_mask(res_ids, input_ids.shape[1], emb_matrix.shape[0], model_wrapper.pad_token,
                                          args.device)

    x_embeds = init_embeddings(args, model_wrapper, res_ids, input_ids, attention_mask, init_ids=init_ids)
    optimizer = torch.optim.Adam([x_embeds], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.lr_decay)

    best_loss, best_x = None, x_embeds.detach().clone()
    best_projected_loss, best_projected_ids = None, None
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
        optimizer.step()
        scheduler.step()

        should_log = (step + 1) == 1 or (step + 1) % args.print_every == 0 or (step + 1) == args.n_steps

        with torch.no_grad():
            if pad_mask.any():
                x_embeds[pad_mask] = pad_embed
            if _hybrid_uses_periodic_projection(args) and (step + 1) % args.hybrid_project_every == 0:
                projected_ids = project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask,
                                                      model_wrapper.pad_token, emb_norm=emb_norm)
                x_embeds.copy_(emb_matrix[projected_ids])
                if pad_mask.any():
                    x_embeds[pad_mask] = pad_embed
            rec_loss_value = rec_loss.item()
            if best_loss is None or rec_loss_value < best_loss:
                best_loss = rec_loss_value
                best_x = x_embeds.detach().clone()

        projected_rec_loss = None
        if should_log or (step + 1) == args.n_steps:
            if _hybrid_uses_candidate_projection(args):
                with torch.no_grad():
                    projected_ids = project_to_candidates(x_embeds, emb_matrix, candidate_mask, pad_mask,
                                                          model_wrapper.pad_token, emb_norm=emb_norm)
                projected_rec_loss = compute_rec_loss_from_ids(
                    args,
                    model_wrapper,
                    true_grads,
                    projected_ids,
                    attention_mask,
                    true_labels,
                ).item()
                if best_projected_loss is None or projected_rec_loss < best_projected_loss:
                    best_projected_loss = projected_rec_loss
                    best_projected_ids = projected_ids.detach().clone()

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

    if best_projected_ids is not None:
        return best_projected_ids
    if _hybrid_uses_candidate_projection(args):
        return project_to_candidates(best_x, emb_matrix, candidate_mask, pad_mask, model_wrapper.pad_token,
                                     emb_norm=emb_norm)
    return project_to_vocab(best_x, emb_matrix, pad_mask, model_wrapper.pad_token, emb_norm=emb_norm)


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
        _, res_ids, _, _ = filter_l1(args, model_wrapper, R_Qs)

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
    if args.hybrid_init_mode == 'dager':
        init_ids = select_dager_decoder_ids(args, model_wrapper, R_Qs, res_ids, orig_batch)
    final_ids = hybrid_optimize(args, model_wrapper, lm, true_grads, res_ids, orig_batch, true_labels,
                                init_ids=init_ids)
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
    device = torch.device(args.device)
    metric = evaluate.load('rouge', cache_dir=args.cache_dir)
    dataset = TextDataset(device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir,
                          use_hf_split=args.use_hf_split)
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
    attack_name = f"hybrid_{args.loss}"
    results_dir = os.path.join(_repo_root(), "results", attack_name ,f"results_{job_hash}")
    os.makedirs(results_dir, exist_ok=True)
    t_start = time.time()
    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
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
            curr_metrics.append(metrics)
            sentence_rows.append({
                "run_id": job_hash,
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
        final_results.extend(curr_metrics)
        logger.info('[Curr input metrics]:')
        logger.info(f"{print_summary_table(summary)}")
        logger.info('[Aggregate metrics]:')
        aggregated_results = evaluate_prediction(" ".join(prediction), " ".join(reference), wrapper_tokenizer,
                                                 metric)

        final_per_input_results.append(aggregated_results)
        logger.info(f"{print_single_metric_dict(aggregated_results)}")
        input_time_sec = time.time() - t_input_start
        total_time_sec = time.time() - t_start
        input_time = str(datetime.timedelta(seconds=input_time_sec)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=total_time_sec)).split(".")[0]
        logger.info(f'input #{i} time: {input_time} | total time: {total_time}')
        input_rows.append({
            "run_id": job_hash,
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
        del sample, prediction, reference, curr_metrics, aggregated_results
        cleanup_memory()
        logger.info("")
        logger.info("")
    logger.info('[Aggregate metrics]:')
    total_time = time.time() - t_start
    aggregated_results = evaluate_prediction(" ".join(predictions), " ".join(references), wrapper_tokenizer, metric)
    aggregated_results[f"reconstruction_time_mean"] = float(np.mean(input_times))
    aggregated_results[f"reconstruction_time_std"] = float(np.std(input_times))
    logger.info(f"Overall {print_single_metric_dict(aggregated_results)}")
    summary = summarize_metrics(final_results)
    summary[f"reconstruction_time_mean"] = float(np.mean(input_times))
    summary[f"reconstruction_time_std"] = float(np.std(input_times))
    logger.info(f"Per Sentence{print_summary_table(summary)}")
    summary_per_input = summarize_metrics(final_per_input_results)
    summary_per_input[f"reconstruction_time_mean"] = float(np.mean(input_times))
    summary_per_input[f"reconstruction_time_std"] = float(np.std(input_times))
    logger.info(f"Per Input Results {print_summary_table(summary_per_input)}")

    summary_results = {"Overall Results": aggregated_results, "Per Sentence Results": summary,
                       "Per Input Results": summary_per_input, "Arguments": vars(args)}
    logger.info(f"Experiment time {total_time}")
    pd.DataFrame(sentence_rows).to_csv(os.path.join(results_dir, "sentence_results.csv"), index=False)
    pd.DataFrame(input_rows).to_csv(os.path.join(results_dir, "input_results.csv"), index=False)
    with open(os.path.join(results_dir, "run_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)
    logger.info('Done with all.')
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)


if __name__ == '__main__':
    main()
