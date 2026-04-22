import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from args_factory import get_args
from data_utils import TextDataset
from init import get_init
from nlp_utils import load_gpt2_from_dict
from utilities import compute_grads, get_closest_tokens, get_reconstruction_loss, fix_special_tokens, \
    remove_padding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment import cleanup_memory, get_results_dir, is_attack_complete, load_rouge_metric, \
    setup_experiment_logging
from utils.functional import (
    _rouge_triplet,
    evaluate_prediction,
    print_single_metric_dict,
    print_summary_table,
    summarize_metrics,
)

from transformers.utils import logging

logging.set_verbosity_error()

args = get_args()
logger, log_path, job_hash = setup_experiment_logging(args, "lamp_attack")
logger.info(f"Arguments {args}")

if args.neptune:
    import neptune

    neptune.init(api_token=os.getenv('NEPTUNE_API_KEY'), project_qualified_name=args.neptune)
    neptune.create_experiment(args.neptune_label, params=vars(args))


def disable_extra_model_outputs(model):
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.output_hidden_states = False
        model.config.output_attentions = False


def load_hf_model_eager(model_cls, model_name, **kwargs):
    try:
        return model_cls.from_pretrained(model_name, attn_implementation="eager", **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_name, **kwargs)


def get_loss(args, lm, model, ids, x_embeds, true_labels, true_grads, create_graph=False):
    perplexity = lm(
        input_ids=ids,
        labels=ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    ).loss
    rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=create_graph)
    return perplexity, rec_loss, rec_loss + args.coeff_perplexity * perplexity


def swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads):
    logger.info('Attempt swap')
    best_x_embeds, best_tot_loss = None, None
    changed = None
    for sen_id in range(x_embeds.data.shape[0]):
        for sample_idx in range(200):
            perm_ids = np.arange(x_embeds.shape[1])

            if sample_idx != 0:
                if sample_idx % 4 == 0 and max_len[sen_id] > 2:  # swap two tokens
                    i, j = 1 + np.random.randint(max_len[sen_id] - 2), 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                elif sample_idx % 4 == 1 and max_len[sen_id] > 2:  # move a token to another place
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    j = 1 + np.random.randint(max_len[sen_id] - 1)
                    if i < j:
                        perm_ids = np.concatenate([perm_ids[:i], perm_ids[i + 1:j], perm_ids[i:i + 1], perm_ids[j:]])
                    else:
                        perm_ids = np.concatenate([perm_ids[:j], perm_ids[i:i + 1], perm_ids[j:i], perm_ids[i + 1:]])
                elif sample_idx % 4 == 2:  # move a sequence to another place
                    b = 1 + np.random.randint(max_len[sen_id] - 1)
                    e = 1 + np.random.randint(max_len[sen_id] - 1)
                    if b > e:
                        b, e = e, b
                    p = 1 + np.random.randint(max_len[sen_id] - 1 - (e - b))
                    if p >= b:
                        p += e - b
                    if p < b:
                        perm_ids = np.concatenate([perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]])
                    elif p >= e:
                        perm_ids = np.concatenate([perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]])
                    else:
                        assert False
                elif sample_idx % 4 == 3 and max_len[sen_id] > 2:  # take some prefix and put it at the end
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids = np.concatenate([perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]])

            new_ids = cos_ids.clone()
            new_ids[sen_id] = cos_ids[sen_id, perm_ids]
            new_x_embeds = x_embeds.clone()
            new_x_embeds[sen_id] = x_embeds[sen_id, perm_ids, :]

            _, _, new_tot_loss = get_loss(args, lm, model, new_ids, new_x_embeds, true_labels, true_grads)

            if (best_tot_loss is None) or (new_tot_loss < best_tot_loss):
                best_x_embeds = new_x_embeds
                best_tot_loss = new_tot_loss
                if sample_idx != 0:
                    changed = sample_idx % 4
            del new_ids, new_x_embeds, new_tot_loss
            if sample_idx % 25 == 0:
                cleanup_memory()
        if not (changed is None):
            change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
            logger.info(change)
        x_embeds.data = best_x_embeds


def reconstruct(args, device, sample, metric, tokenizer, lm, model):
    sequences, true_labels = sample
    start_time = time.time()
    lm_tokenizer = tokenizer

    gpt2_embeddings = lm.get_input_embeddings()
    gpt2_embeddings_weight = gpt2_embeddings.weight.unsqueeze(0)

    bert_embeddings = model.get_input_embeddings()
    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)

    orig_batch = tokenizer(sequences, padding=True, truncation=True, max_length=tokenizer.model_max_length,
                           return_tensors='pt').to(device)
    true_embeds = bert_embeddings(orig_batch['input_ids'])
    true_grads = compute_grads(model, true_embeds, true_labels)

    if args.defense_pct_mask is not None:
        for grad in true_grads:
            grad.data = grad.data * (torch.rand(grad.shape).to(device) > args.defense_pct_mask).float()
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape).to(device) * args.defense_noise

    # BERT special tokens (0-999) are never part of the sentence
    unused_tokens = []
    if args.use_embedding:
        for i in range(tokenizer.vocab_size):
            if true_grads[0][i].abs().sum() < 1e-9 and i != tokenizer.pad_token_id:
                unused_tokens += [i]
    elif args.bert_path != 'gpt2':
        unused_tokens += list(range(1, 100))
        unused_tokens += list(range(104, 999))
    unused_tokens = np.array(unused_tokens)

    # If length of sentences is known to attacker keep padding fixed
    pads = None
    if args.know_padding:
        pads = [orig_batch['input_ids'].shape[1]] * orig_batch['input_ids'].shape[0]
        for sen_id in range(orig_batch['input_ids'].shape[0]):
            for i in range(orig_batch['input_ids'].shape[1] - 1, 0, -1):
                if orig_batch['input_ids'][sen_id][i] == tokenizer.pad_token_id:
                    pads[sen_id] = i
                else:
                    break
    logger.info(f'Debug: ids_shape = {orig_batch["input_ids"].shape[1]}, pads = {pads}')
    logger.info(f'Debug: input ids = {orig_batch["input_ids"]}')
    logger.info(f'Debug: ref = {tokenizer.batch_decode(orig_batch["input_ids"])}')

    # Get initial embeddings + set up opt
    x_embeds = get_init(args, model, unused_tokens, true_embeds.shape, true_labels, true_grads, bert_embeddings,
                        bert_embeddings_weight, tokenizer, lm, lm_tokenizer, orig_batch['input_ids'], pads)

    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)
    if args.opt_alg == 'adam':
        opt = optim.Adam([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bfgs':
        opt = optim.LBFGS([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bert-adam':
        opt = torch.optim.AdamW([x_embeds], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    if args.lr_decay_type == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=args.lr_decay)
    elif args.lr_decay_type == 'LambdaLR':
        def lr_lambda(current_step: int):
            return max(0.0, float(args.lr_max_it - current_step) / float(max(1, args.lr_max_it)))

        lr_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    logger.info(f'Nsteps: {args.n_steps}')

    if pads is None:
        max_len = [x_embeds.shape[1]] * x_embeds.shape[0]
    else:
        max_len = pads

    # Main loop
    best_final_error, best_final_x = None, x_embeds.detach().clone()
    for it in range(args.n_steps):
        t_start = time.time()
        if args.time_limit is not None and t_start - start_time > args.time_limit and it % args.swap_every == args.swap_every - 1:
            break

        def closure():
            opt.zero_grad()
            rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=True)
            reg_loss = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
            tot_loss = rec_loss + args.coeff_reg * reg_loss
            tot_loss.backward()
            with torch.no_grad():
                if args.grad_clip is not None:
                    grad_norm = x_embeds.grad.norm()
                    if grad_norm > args.grad_clip:
                        x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))
            return tot_loss

        error = opt.step(closure)
        if best_final_error is None or error <= best_final_error:
            best_final_error = error.item()
            best_final_x.data[:] = x_embeds.data[:]
        del error

        lr_scheduler.step()

        fix_special_tokens(x_embeds, bert_embeddings.weight, pads, is_bert=args.bert_path != 'gpt2',
                           pad_token=tokenizer.pad_token_id)

        _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)

        # Trying swaps
        if args.use_swaps and it >= args.swap_burnin * args.n_steps and it % args.swap_every == 1:
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads)

        steps_done = it + 1
        if steps_done % args.print_every == 0:
            _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)
            x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(
                cos_ids).norm(dim=2, p=2, keepdim=True)
            _, _, tot_loss_proj = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
            perplexity, rec_loss, tot_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels, true_grads)

            step_time = time.time() - t_start

            logger.info('[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f [t=%.2fs]' % (
                steps_done, args.n_steps, tot_loss.item(), perplexity.item(), rec_loss.item(), tot_loss_proj.item(),
                step_time))
            logger.info('prediction: %s' % (tokenizer.batch_decode(cos_ids)))

            tokenizer.batch_decode(cos_ids)
            del x_embeds_proj, tot_loss_proj, perplexity, rec_loss, tot_loss
            cleanup_memory()

    # Swaps in the end for ablation
    if args.use_swaps_at_end:
        swap_at_end_it = int((1 - args.swap_burnin) * args.n_steps // args.swap_every)
        logger.info('Trying %i swaps' % swap_at_end_it)
        for i in range(swap_at_end_it):
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads)

    # Postprocess
    x_embeds.data = best_final_x
    fix_special_tokens(x_embeds, bert_embeddings.weight, pads, is_bert=args.bert_path != 'gpt2',
                       pad_token=tokenizer.pad_token_id)
    m = 5
    d, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight, metric='cos')
    x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(cos_ids).norm(
        dim=2, p=2, keepdim=True)
    _, _, best_tot_loss = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
    best_ids = cos_ids
    best_x_embeds_proj = x_embeds_proj

    prediction, reference = [], []
    for i in range(best_ids.shape[0]):
        prediction += [remove_padding(tokenizer, best_ids[i], is_bert=args.bert_path != 'gpt2')]
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], is_bert=args.bert_path != 'gpt2')]

    # Matching
    cost = np.zeros((x_embeds.shape[0], x_embeds.shape[0]))
    for i in range(x_embeds.shape[0]):
        for j in range(x_embeds.shape[0]):
            fm, _, _ = _rouge_triplet(
                metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1']
            )
            cost[i, j] = 1.0 - fm
    row_ind, col_ind = linear_sum_assignment(cost)

    ids = list(range(x_embeds.shape[0]))
    ids.sort(key=lambda i: col_ind[i])
    new_prediction = []
    for i in range(x_embeds.shape[0]):
        new_prediction += [prediction[ids[i]]]
    prediction = new_prediction
    del true_grads, true_embeds, x_embeds, best_final_x, best_x_embeds_proj, x_embeds_proj
    cleanup_memory()

    return prediction, reference


def main():
    logger.info('\n\n\nCommand: %s\n\n\n', ' '.join(sys.argv))
    attack_name = f"lamp_{args.loss}"
    is_complete, results_dir = is_attack_complete(attack_name, job_hash)
    if is_complete:
        logger.info("Results already exist for this config at %s; skipping attack.", results_dir)
        logger.info('Done with all.')
        print(f"Hash Value {job_hash} is already done")
        return

    device = torch.device(args.device)
    metric = load_rouge_metric(cache_dir=args.cache_dir, logger=logger)
    dataset = TextDataset(
        args.device,
        args.dataset,
        args.split,
        args.n_inputs,
        args.batch_size,
        cache_dir=args.cache_dir,
        use_hf_split=args.use_hf_split,
    )

    if args.bert_path == 'gpt2':
        lm = load_hf_model_eager(AutoModelForCausalLM, 'gpt2', cache_dir=args.cache_dir).to(device)
    else:
        lm = load_gpt2_from_dict("transformer_wikitext-103.pth", device, output_hidden_states=False).to(device)
    lm.eval()
    disable_extra_model_outputs(lm)

    model = load_hf_model_eager(
        AutoModelForSequenceClassification,
        args.bert_path,
        cache_dir=args.cache_dir,
    ).to(device)
    model.eval()
    disable_extra_model_outputs(model)
    if args.bert_path == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True, cache_dir=args.cache_dir)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, cache_dir=args.cache_dir)
    tokenizer.model_max_length = 512
    lm.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info('\n\nAttacking..\n')
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
        sample = dataset[i]  # (seqs, labels)

        logger.info(f'Running input #{i} of {args.n_inputs}.')
        if args.neptune:
            neptune.log_metric('curr_input', i)

        logger.info('reference: ')
        for seq in sample[0]:
            logger.info('========================')
            logger.info(seq)

        logger.info('========================')

        prediction, reference = reconstruct(args, device, sample, metric, tokenizer, lm, model)
        predictions += prediction
        references += reference

        logger.info(f'Done with input #{i} of {args.n_inputs}.')
        logger.info('reconstructed output:')
        curr_metrics = []
        for sent_idx, (ref, pred) in enumerate(zip(reference, prediction)):
            logger.info('========================')
            logger.info("Reference: %s", ref)
            logger.info("Prediction: %s", pred)
            metrics = evaluate_prediction(pred, ref, tokenizer, metric)
            curr_metrics.append(metrics)
            sentence_rows.append({
                "run_id": job_hash,
                "attack": attack_name,
                "model": args.bert_path,
                "dataset": args.dataset,
                "input_index": i,
                "sentence_index": sent_idx,
                "reference": ref,
                "prediction": pred,
                **metrics,
            })
        logger.info('========================')
        final_results.extend(curr_metrics)

        logger.info('[Curr input metrics]:')
        logger.info("%s", print_summary_table(summarize_metrics(curr_metrics)))

        logger.info('[Aggregate metrics]:')
        aggregated_results = evaluate_prediction(" ".join(prediction), " ".join(reference), tokenizer, metric)
        final_per_input_results.append(aggregated_results)
        logger.info("%s", print_single_metric_dict(aggregated_results))

        input_time_sec = time.time() - t_input_start
        total_time_sec = time.time() - t_start
        input_time = str(datetime.timedelta(seconds=input_time_sec)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=total_time_sec)).split(".")[0]
        logger.info(f'input #{i} time: {input_time} | total time: {total_time}\n\n')
        input_times.append(input_time_sec)
        input_rows.append({
            "run_id": job_hash,
            "attack": attack_name,
            "model": args.bert_path,
            "dataset": args.dataset,
            "input_index": i,
            "num_sentences": len(reference),
            "joined_reference": " ".join(reference),
            "joined_prediction": " ".join(prediction),
            "reconstruction_time_sec": input_time_sec,
            **aggregated_results,
        })
        del sample, prediction, reference, curr_metrics, aggregated_results
        cleanup_memory()

    logger.info('[Aggregate metrics]:')
    total_time = time.time() - t_start
    aggregated_results = evaluate_prediction(" ".join(predictions), " ".join(references), tokenizer, metric)
    aggregated_results["experiment_time_mean"] = float(total_time)
    aggregated_results["experiment_time_std"] = float(0)
    summary = summarize_metrics(final_results) if final_results else {}
    if summary:
        summary["reconstruction_time_mean"] = float(np.mean(input_times)) if input_times else 0.0
        summary["reconstruction_time_std"] = float(np.std(input_times)) if input_times else 0.0
    summary_per_input = summarize_metrics(final_per_input_results) if final_per_input_results else {}
    if summary_per_input:
        summary_per_input["reconstruction_time_mean"] = float(np.mean(input_times)) if input_times else 0.0
        summary_per_input["reconstruction_time_std"] = float(np.std(input_times)) if input_times else 0.0
    logger.info("Overall %s", print_single_metric_dict(aggregated_results))
    logger.info("Per Sentence %s", print_summary_table(summary) if summary else "{}")
    logger.info("Per Input Results %s", print_summary_table(summary_per_input) if summary_per_input else "{}")
    summary_results = {
        "Overall Results": aggregated_results,
        "Per Sentence Results": summary,
        "Per Input Results": summary_per_input,
        "Arguments": vars(args),
    }
    logger.info("Experiment time %s", total_time)
    pd.DataFrame(sentence_rows).to_csv(os.path.join(results_dir, "sentence_results.csv"), index=False)
    pd.DataFrame(input_rows).to_csv(os.path.join(results_dir, "input_results.csv"), index=False)
    with open(os.path.join(results_dir, "run_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)
    logger.info("Results directory: %s", results_dir)

    logger.info('Done with all.')
    print(f"Hash Value {job_hash} Done")
    if args.neptune:
        neptune.log_metric('curr_input', args.n_inputs)


if __name__ == '__main__':
    main()
