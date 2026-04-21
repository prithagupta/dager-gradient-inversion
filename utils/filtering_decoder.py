import copy
import itertools
import logging

import numpy as np
import torch
from tqdm import tqdm

from utils.functional import check_if_in_span, get_span_dists, filter_outliers

logger = logging.getLogger(__name__)


def _as_candidate_position_scores(scores, n_candidates):
    if scores.ndim == 2:
        return scores
    if scores.ndim != 1:
        raise ValueError(f'Expected 1D or 2D decoder scores, got shape {tuple(scores.shape)}')
    if scores.numel() == n_candidates:
        return scores.unsqueeze(1)
    if n_candidates == 1:
        return scores.unsqueeze(0)
    if scores.numel() % n_candidates == 0:
        return scores.reshape(n_candidates, -1)
    raise ValueError(f'Cannot reshape decoder scores with shape {tuple(scores.shape)} for {n_candidates} candidates')


def _candidate_sequence_scores(scores, model_wrapper):
    if model_wrapper.has_bos() and scores.shape[1] > 1:
        return scores[:, 1:].mean(dim=1)
    return scores.mean(dim=1)


def filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=-1):
    R_Q2 = R_Qs[1]
    res_ids = copy.deepcopy(res_ids)
    for i in range(len(res_ids)):
        if max_ids >= 0:
            res_ids[i] = res_ids[i][:min(max_ids, len(res_ids[i]))]
    if args.pad == 'right':
        batch = torch.tensor(res_ids[0]).unsqueeze(1)
    elif args.pad == 'left':
        start_ids = res_ids[0].copy()
        if model_wrapper.start_token is not None:
            start_ids = [model_wrapper.start_token]
        batch = torch.tensor(start_ids).unsqueeze(1)

    is_batch_incorrect = torch.zeros_like(batch).squeeze(1)

    scores = check_if_in_span(R_Q2, model_wrapper.get_layer_inputs(batch.to(args.device))[0], args.dist_norm)
    scores = _as_candidate_position_scores(scores, batch.shape[0]).mean(dim=1).to('cpu')

    predicted_sentences = []
    predicted_sentences_scores = []

    top_B_incorrect_sentences = [[] for _ in range(args.batch_size)]
    top_B_incorrect_scores = [torch.inf for _ in range(args.batch_size)]

    i = 1
    while True:
        logger.info(f'Position {i}')

        top_B_incorrect_sentences_len = [[] for _ in range(args.batch_size)]
        top_B_incorrect_scores_len = [torch.inf for _ in range(args.batch_size)]

        if len(batch) == 0 or (not model_wrapper.has_rope() and i >= len(res_ids)):
            break

        if model_wrapper.has_rope():
            ends = torch.Tensor(res_ids[0])
            ends = ends[ends != model_wrapper.pad_token]
        else:
            ends = torch.Tensor(res_ids[i])

        lst = itertools.product(range(batch.shape[0]), range(len(ends)))
        it_lst = iter(lst)
        next_batch = []
        next_scores = []
        is_next_batch_incorrect = []
        ds = []
        is_complete = args.defense_noise is None
        curr_sentence = 0
        progress_bar = tqdm(total=batch.shape[0] * ends.shape[0], disable=True)

        while True:
            els_b = []
            els_ends = []
            for _ in range(max((args.parallel // ends.shape[0]), 1) * ends.shape[0]):
                el = next(it_lst, None)
                if el is None:
                    break
                els_b.append(el[0])
                els_ends.append(el[1])
            els_b = torch.tensor(np.array(els_b))
            els_ends = torch.tensor(np.array(els_ends))
            if els_b.shape[0] == 0 and is_complete:
                break
            elif els_b.shape[0] == 0:
                idxs = np.array(list(itertools.product(range(batch.shape[0]), range(len(ends)))))
                new_batch = torch.cat((torch.tensor(batch[idxs[:, 0]]).long(), \
                                       torch.tensor(ends[idxs[:, 1]]).long().unsqueeze(1)), dim=-1).to(args.device)
                is_new_batch_incorrect = is_batch_incorrect[idxs[:, 0]].to(args.device)
                sizesq2 = torch.cat(ds)
                sizesq2, correct_sentences = filter_outliers(sizesq2, stage='sequence', std_thrs=args.l2_std_thrs,
                                                             maxB=args.batch_size)
                is_complete = True
                logger.info(sizesq2.min())
            else:
                new_batch = torch.cat((batch[els_b], ends[els_ends].unsqueeze(1)), dim=-1).int().to(args.device)
                is_new_batch_incorrect = is_batch_incorrect[els_b].to(args.device)

                if args.defense_noise is None:
                    sizesq2, correct_sentences = filter_decoder_step(args, model_wrapper, R_Qs, new_batch, i)
                else:
                    ds.append(filter_decoder_step(args, model_wrapper, R_Qs, new_batch, i))
                    continue

            if i > 1:
                complete_batches = torch.where(~correct_sentences.reshape(-1, ends.shape[0]).any(dim=1))[0]
                for pred_idx in complete_batches:
                    if not is_batch_incorrect[curr_sentence + pred_idx]:
                        predicted_sentences.append(batch[curr_sentence + pred_idx].cpu().numpy().tolist())
                        predicted_sentences_scores.append(scores[curr_sentence + pred_idx].item())

            next_batch.append(new_batch[correct_sentences].to('cpu'))
            next_scores.append(_candidate_sequence_scores(sizesq2, model_wrapper)[correct_sentences].to('cpu'))
            is_next_batch_incorrect.append(is_new_batch_incorrect[correct_sentences].to('cpu'))

            curr_sentence += len(els_b) // ends.shape[0]

            incorrect_sentences = new_batch[~correct_sentences]
            sizesq2_incorrect = _candidate_sequence_scores(sizesq2[~correct_sentences], model_wrapper)

            if len(incorrect_sentences) == 0:
                continue

            # Draw unique
            scores_best_batch, sentences_best_batch = [], []
            for b_idx in range(args.batch_size):
                idx_best_batch = torch.argmin(sizesq2_incorrect)
                best_score = sizesq2_incorrect[idx_best_batch]
                best_sentence = incorrect_sentences[idx_best_batch]
                sentences_best_batch.append(best_sentence.cpu().numpy().tolist())
                scores_best_batch.append(best_score.item())
                similar_sentences = (best_sentence == incorrect_sentences).sum(1) >= (i + 1) * args.distinct_thresh
                sizesq2_incorrect[similar_sentences] = torch.inf

            for b_idx in range(len(scores_best_batch)):
                if scores_best_batch[b_idx] > top_B_incorrect_scores_len[-1]:
                    break
                predicted_idx = 0
                while scores_best_batch[b_idx] > top_B_incorrect_scores_len[predicted_idx]:
                    predicted_idx += 1
                for rep_idx in range(predicted_idx, args.batch_size):
                    if len(top_B_incorrect_sentences_len[rep_idx]) > 0 and \
                            (torch.tensor(sentences_best_batch[b_idx:b_idx + 1]) == torch.tensor(
                                top_B_incorrect_sentences_len[rep_idx:rep_idx + 1])).sum(1) \
                            >= (i + 1) * args.distinct_thresh:

                        continue
                    else:
                        top_B_incorrect_sentences_len = top_B_incorrect_sentences_len[
                                                            :predicted_idx] + sentences_best_batch[
                                                            b_idx:b_idx + 1] + top_B_incorrect_sentences_len[
                                                            predicted_idx:rep_idx] + top_B_incorrect_sentences_len[
                                                            rep_idx + 1:]
                        top_B_incorrect_scores_len = top_B_incorrect_scores_len[:predicted_idx] + scores_best_batch[
                            b_idx:b_idx + 1] + top_B_incorrect_scores_len[
                                                         predicted_idx:rep_idx] + top_B_incorrect_scores_len[
                                                         rep_idx + 1:]
                        break
            progress_bar.update(new_batch.shape[0])

        batch = torch.cat(next_batch)
        valid_fallbacks = [
            (sent, score)
            for sent, score in zip(top_B_incorrect_sentences_len, top_B_incorrect_scores_len)
            if len(sent) > 0 and np.isfinite(float(score))
        ]
        if len(batch) == 0:
            if valid_fallbacks and (i > 1 or not model_wrapper.has_rope()):
                logger.info(
                    "Decoder L2 accepted no candidates at position %s; keeping %s best approximate candidates.",
                    i,
                    len(valid_fallbacks),
                )
                top_B_incorrect_sentences += [sent for sent, _ in valid_fallbacks]
                top_B_incorrect_scores += [score for _, score in valid_fallbacks]
            elif valid_fallbacks:
                logger.info(
                    "Decoder L2 accepted no candidates at position %s; ignoring short approximate prefixes.",
                    i,
                )
            break
        else:
            is_batch_incorrect = torch.cat(is_next_batch_incorrect)
            scores = torch.cat(next_scores)

        if i != len(res_ids) - 1 and valid_fallbacks:
            fallback_sentences = [sent for sent, _ in valid_fallbacks]
            fallback_scores = [score for _, score in valid_fallbacks]
            batch = torch.cat((batch, torch.tensor(fallback_sentences)))
            scores = torch.cat((scores, torch.tensor(fallback_scores)))
            is_batch_incorrect = torch.cat((is_batch_incorrect, torch.ones(len(fallback_sentences))))

        top_B_incorrect_scores += top_B_incorrect_scores_len
        top_B_incorrect_sentences += top_B_incorrect_sentences_len

        if args.reduce_incorrect > 0:
            final_incorrect_scores = []
            final_incorrect_sentences = []
            sorted_idx = np.argsort(top_B_incorrect_scores)[::-1]
            for j, idx in enumerate(sorted_idx):
                if len(top_B_incorrect_scores) - j <= args.batch_size - len(final_incorrect_scores):
                    final_incorrect_scores.append(top_B_incorrect_scores[idx])
                    final_incorrect_sentences.append(top_B_incorrect_sentences[idx])
                    continue
                proposal_sent = np.array(top_B_incorrect_sentences[idx])
                fail = False
                for accepted_sent in final_incorrect_sentences:
                    if len(accepted_sent) < len(proposal_sent):
                        s1 = np.pad(accepted_sent, (0, len(proposal_sent) - len(accepted_sent)), 'constant',
                                    constant_values=(0, -1))
                        s2 = proposal_sent
                    else:
                        s1 = np.pad(proposal_sent, (0, len(accepted_sent) - len(proposal_sent)), 'constant',
                                    constant_values=(0, -1))
                        s2 = accepted_sent
                    if np.sum(s1 == s2) < len(s1) * args.distinct_thresh:
                        fail = True
                        break
                if not fail:
                    final_incorrect_scores.append(top_B_incorrect_scores[idx])
                    final_incorrect_sentences.append(top_B_incorrect_sentences[idx])
            top_B_incorrect_scores = final_incorrect_scores
            top_B_incorrect_sentences = final_incorrect_sentences
        progress_bar.close()
        i += 1
    # Add remaining sentences
    for i in range(batch.shape[0]):
        predicted_sentences.append(batch[i].cpu().numpy().tolist())
        predicted_sentences_scores.append(scores[i].item())

    clean_top_B_incorrect_sentences = []
    clean_top_B_incorrect_scores = []
    for sent, score in zip(top_B_incorrect_sentences, top_B_incorrect_scores):
        if len(sent) > 0 and np.isfinite(float(score)):
            clean_top_B_incorrect_sentences.append(sent)
            clean_top_B_incorrect_scores.append(score)

    return predicted_sentences, predicted_sentences_scores, clean_top_B_incorrect_sentences, clean_top_B_incorrect_scores


def filter_decoder_step(args, model_wrapper, R_Qs, batch, p):
    if args.defense_noise is None:
        R_Q2 = R_Qs[1]
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layer1 = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask)[0]
        sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
        sizesq2 = _as_candidate_position_scores(sizesq2, batch.shape[0])
        l2_span_thresh = model_wrapper.effective_l2_span_thresh(args.l2_span_thresh)
        boolsq2 = sizesq2 < l2_span_thresh

        if model_wrapper.has_rope():
            special_tokens = torch.tensor(
                [model_wrapper.pad_token, model_wrapper.start_token],
                device=batch.device,
                dtype=batch.dtype,
            )
            boolsq2 = torch.logical_or(boolsq2, torch.isin(batch, special_tokens))
            if p > 1:
                repeats = torch.logical_and(batch[:, -2] == model_wrapper.start_token,
                                            torch.isin(batch[:, -1], batch[:, 1].to(batch.device)))
                correct_sentences = torch.logical_and(boolsq2.all(dim=1), ~repeats.to(args.device))
            else:
                correct_sentences = boolsq2.all(dim=1)
        else:
            correct_sentences = boolsq2.all(dim=1)

        return sizesq2, correct_sentences

    else:
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layers = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask, layers=args.n_layers - 1)
        return get_span_dists(args, model_wrapper, R_Qs, input_layers, stage='sequence')
