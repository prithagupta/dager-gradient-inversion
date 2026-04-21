import datetime
import numpy as np
import torch
import torch.nn.functional as F
from evaluate import load as load_metric
from utils.models import ModelWrapper
from utils.data import TextDataset
from utils.filtering_encoder import filter_encoder
from utils.filtering_decoder import filter_decoder
from utils.functional import get_top_B_in_span, check_if_in_span, remove_padding, filter_outliers, get_span_dists
from args_factory import get_args
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0

import os, psutil, torch, gc


@torch.no_grad()
def _fit_alpha_stream(mix_grads, comp_grads, cos_abs_th: float = 0.0, clamp=(-1.5, 1.5), shrink: float = 1.0):
    dev = None
    for g in mix_grads:
        if g is not None:
            dev = g.device
            break
    if dev is None:
        dev = torch.device("cpu")

    num = torch.zeros((), device=dev)  # m·c
    den = torch.zeros((), device=dev)  # c·c
    nm2 = torch.zeros((), device=dev)  # ||m||^2
    nc2 = torch.zeros((), device=dev)  # ||c||^2

    for gm, gc in zip(mix_grads, comp_grads):
        if gm is None or gc is None:
            continue
        num += (gm * gc).sum()
        den += (gc * gc).sum()
        nm2 += (gm * gm).sum()
        nc2 += (gc * gc).sum()

    # 余弦阈值（绝对值），默认不启用
    if cos_abs_th > 0.0:
        nm = nm2.clamp_min(1e-12).sqrt()
        nc = nc2.clamp_min(1e-12).sqrt()
        cos = num / (nm * nc + 1e-12)
        if torch.isnan(cos) or abs(float(cos)) < cos_abs_th:
            return 0.0

    alpha = float(num / max(float(den), 1e-12))
    if clamp is not None:
        lo, hi = clamp
        alpha = max(lo, min(hi, alpha))
    alpha *= float(shrink)
    return alpha

@torch.no_grad()
def _block_view(grads, model_wrapper, blocks):
    out = [None if g is None else torch.zeros_like(g) for g in grads]
    for tag, a, *rest in blocks:
        if tag == 'l1q':
            lid, start, end = a, rest[0], rest[1]
            if grads[lid] is not None:
                out[lid][:, start:end] = grads[lid][:, start:end]
        elif tag == 'wte':
            wid = getattr(model_wrapper, "wte_grad_id", None)
            if (wid is not None) and (grads[wid] is not None):
                out[wid].copy_(grads[wid])
    return out

@torch.no_grad()
def _dot_norm_cos(a_grads, b_grads):
    dot = 0.0; na2 = 0.0; nb2 = 0.0
    for a, b in zip(a_grads, b_grads):
        if a is None or b is None: continue
        dot += float((a*b).sum().item())
        na2 += float((a*a).sum().item())
        nb2 += float((b*b).sum().item())
    na = (na2 + 1e-12)**0.5; nb = (nb2 + 1e-12)**0.5
    return dot, na, nb, dot/(na*nb + 1e-12), dot/(nb2 + 1e-12)  # 返回 dot, ||a||, ||b||, cos, phi

@torch.no_grad()
def verify_deflation(residual_grads, pred_grads, model_wrapper,
                     prev_residual_norm=None, tol_phi=5e-3, tol_cos=2e-2):
    dot, nr, ng, cos, phi = _dot_norm_cos(residual_grads, pred_grads)

    l1_id = model_wrapper.layer_ids[0]
    d_model = model_wrapper.model.config.hidden_size
    blocks = [('l1q', l1_id, 0, d_model)]

    r_blk = _block_view(residual_grads, model_wrapper, blocks)
    g_blk = _block_view(pred_grads,     model_wrapper, blocks)
    dot_b, nr_b, ng_b, cos_b, phi_b = _dot_norm_cos(r_blk, g_blk)

    alpha_star = _fit_alpha_stream(residual_grads, pred_grads,
                                   cos_abs_th=0.0, clamp=None, shrink=1.0)

    delta = None
    if prev_residual_norm is not None and prev_residual_norm > 0:
        delta = (prev_residual_norm - nr) / prev_residual_norm

    ok = (abs(phi)   <= tol_phi   and abs(cos)   <= tol_cos and
          abs(phi_b) <= tol_phi   and abs(cos_b) <= tol_cos and
          abs(alpha_star) < tol_phi)

    report = {
        "global": {"dot": dot, "norm_r": nr, "norm_g": ng, "cos": cos, "phi": phi},
        "block_l1q": {"dot": dot_b, "norm_r": nr_b, "norm_g": ng_b, "cos": cos_b, "phi": phi_b},
        "alpha_star": alpha_star,
        "residual_drop_ratio": delta,
    }

    return ok, report

def _project_on_blocks(residual, comp, spec):
    for tag, a, *rest in spec:
        if tag == 'l1q':
            lid, start, end = a, rest[0], rest[1]
            r = residual[lid][:, start:end]; c = comp[lid][:, start:end]
            num = float((r*c).sum().item()); den = float((c*c).sum().item()) + 1e-12
            a_coeff = num / den
            r.data.add_(c, alpha=-a_coeff)
        elif tag == 'wte':
            wid = getattr(model_wrapper, "wte_grad_id", None)
            if wid is not None and residual[wid] is not None and comp[wid] is not None:
                r = residual[wid]; c = comp[wid]
                num = float((r*c).sum().item()); den = float((c*c).sum().item()) + 1e-12
                a_coeff = num / den
                r.data.add_(c, alpha=-a_coeff)

@torch.no_grad()
def _dot_over_norm2(res_grads, cand_grads):
    num = 0.0
    den = 0.0
    for r, g in zip(res_grads, cand_grads):
        if r is None or g is None:
            continue
        num += float((r * g).sum().item())
        den += float((g * g).sum().item())
    if den <= 1e-12:
        return float("-inf")
    return num / den

def _pred_labels_like(orig_batch, s, target_len, device):
    model = getattr(_pred_labels_like, "_model", None)
    true_labels = getattr(_pred_labels_like, "_true_labels", None)
    if true_labels is None:
        raise RuntimeError("Set _pred_labels_like._true_labels before using.")
    lbl_s = true_labels[s]
    import torch
    if not isinstance(lbl_s, torch.Tensor):
        lbl_s = torch.tensor(lbl_s)
    lbl_s = lbl_s.to(device)
    if lbl_s.ndim == 0:
        lbl_s = lbl_s.view(1, 1)
    elif lbl_s.ndim == 1:
        lbl_s = lbl_s.unsqueeze(0)
    if lbl_s.size(0) != 1:
        lbl_s = lbl_s[:1, :]
    if lbl_s.size(1) != target_len:
        new_lbl = torch.full((1, target_len), fill_value=-100, dtype=lbl_s.dtype, device=device)
        copy_len = min(target_len, lbl_s.size(1))
        new_lbl[:, :copy_len] = lbl_s[:, :copy_len]
        lbl_s = new_lbl
    return lbl_s

import os, psutil, torch, gc
@torch.no_grad()
def _project_out_component(residual_grads, comp_grads, tol=1e-10, max_iter=2):
    for _ in range(max_iter):
        num = den = 0.0
        for r, c in zip(residual_grads, comp_grads):
            if r is None or c is None: continue
            num += float((r * c).sum().item())
            den += float((c * c).sum().item())
        if den <= 1e-20 or abs(num) <= tol:
            break
        a = num / den
        for i in range(len(residual_grads)):
            r, c = residual_grads[i], comp_grads[i]
            if r is not None and c is not None:
                r.data.add_(c, alpha=-a)


@torch.no_grad()
def _dot_sum(a_grads, b_grads, model_wrapper=None, ignore_final_layer=False) -> float:
    s = 0.0
    last_layer_idx = -1
    if ignore_final_layer and model_wrapper is not None:
        if hasattr(model_wrapper.model, 'classifier'):
            num_params_in_classifier = len(list(model_wrapper.model.classifier.parameters()))
            last_layer_idx = len(a_grads) - num_params_in_classifier
        elif hasattr(model_wrapper.model, 'score'):
            num_params_in_score = len(list(model_wrapper.model.score.parameters()))
            last_layer_idx = len(a_grads) - num_params_in_score
        if last_layer_idx < 0:
            last_layer_idx = len(a_grads) - 2

    for i, (a, b) in enumerate(zip(a_grads, b_grads)):
        if ignore_final_layer and last_layer_idx > 0 and i >= last_layer_idx:
            continue

        if a is not None and b is not None:
            s += float((a * b).sum().item())
    return s


def cuda_ok():
    try:
        return bool(torch.cuda.is_available() and torch.version.cuda)
    except Exception:
        return False

def reset_peak_memory():
    if cuda_ok():
        try: torch.cuda.reset_peak_memory_stats()
        except Exception: pass

def mem_snap(tag=""):
    if cuda_ok():
        try:
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserv = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[GPU]{tag} alloc={alloc:.1f}MB | reserved={reserv:.1f}MB | peak={peak:.1f}MB")
        except Exception as e:
            print(f"[GPU]{tag} <na>: {e}")
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[CPU]{tag} rss={rss:.1f}MB")

def hard_cleanup_locals(locals_dict):
    kill = [
        "R_Q","R_Q2","R_Qs_original","head_R_Qs_split",
        "grad_l1","grad_l1_query","grad_slices_per_head",
        "per_samples","res_ids_s","seqs_s","scores_s",
        "full_emb_pos","candidate_pos_embeds",
        "pred_batch","pred_labels","pred_grads","t","attn",
        "candidate_pool_indices","fuse_min",
    ]
    for k in kill:
        if k in locals_dict:
            try: del locals_dict[k]
            except: pass
    gc.collect()
    if cuda_ok():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass

def _to_cpu_grads(gs):
    return [None if g is None else g.detach().to('cpu', copy=True).float() for g in gs]


@torch.no_grad()
def check_position_recall(res_ids_s, gold_ids, pad_id=None, eos_id=None, max_print=10, tag=""):
    misses = []
    n_pos = min(len(res_ids_s), len(gold_ids))
    for p in range(n_pos):
        g = int(gold_ids[p])
        if pad_id is not None and g == pad_id:
            break
        if eos_id is not None and g == eos_id:
            pass
        cand_set = set(res_ids_s[p])
        if g not in cand_set:
            misses.append((p, g))
    total = len(gold_ids)
    found = total - len(misses)
    print(f"[Recall@Pos{tag}] total={total} | miss={len(misses)} | recall={found / total:.4f}")
    if misses:
        print(f"[Recall@Pos{tag}] first_misses: {misses[:max_print]}")
    return {
        "total": total,
        "miss": len(misses),
        "recall": found / total,
        "miss_list": misses,
    }


def get_mandatory_punct_ids(tokenizer):
    punct_strs = [
        ".", ",", "!", "?", ";", ":", "...", "—", "-", "(", ")", "[", "]", "{", "}",
        "'", "\"", "’", "”", "“", "…", "\n", "\r", "\t", " ", "—", "–"
    ]
    ids = set()
    for s in punct_strs:
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            for t in toks:
                ids.add(int(t))
        except Exception:
            pass
    for s in ["\n\n", "\r\n", "\n \n"]:
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            for t in toks:
                ids.add(int(t))
        except Exception:
            pass
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if eos_id is not None and eos_id != -1:
        ids.add(int(eos_id))
    if pad_id is not None and pad_id != -1:
        ids.add(int(pad_id))
    return ids


@torch.no_grad()
def sparsify_res_ids_per_position(
        args, model_wrapper, res_ids_s, grad_slices_per_head,
        eps=None, tau=None, K_keep=None, keep_head=None, always_topm=None, min_keep=None
):
    if grad_slices_per_head is None or len(res_ids_s) == 0:
        return res_ids_s

    eps = eps if eps is not None else getattr(args, "pre_sparse_eps", 1e-3)
    tau = tau if tau is not None else getattr(args, "pre_sparse_tau", 0.35)
    K_keep = K_keep if K_keep is not None else getattr(args, "pre_sparse_K_keep", 48)
    keep_head = keep_head if keep_head is not None else getattr(args, "pre_sparse_keep_head", -1)
    always_topm = always_topm if always_topm is not None else getattr(args, "pre_sparse_always_topm", 16)
    min_keep = min_keep if min_keep is not None else getattr(args, "pre_sparse_min_keep", 24)

    H = len(grad_slices_per_head)
    heads = range(H) if keep_head == -1 else range(min(keep_head, H))

    res_filtered = []
    for p, ids_p in enumerate(res_ids_s):
        K = len(ids_p)
        ids_t = torch.as_tensor(ids_p, device=model_wrapper.device, dtype=torch.long)
        full_emb_pos = model_wrapper.get_embeddings(p)[0]  # [V, d_model]
        E = full_emb_pos.index_select(0, ids_t)  # [K, d_model]

        scores = []
        for hi in heads:
            G = grad_slices_per_head[hi]  # [d_model, d_head]
            g = E @ G  # [K, d_head]
            thr = torch.quantile(g.abs(), q=getattr(args, "pre_sparse_q", 0.25), dim=0, keepdim=True)  # [1, d_head]
            s_h = (g.abs() <= thr).float().mean(dim=1)
            scores.append(s_h)

        S_all = torch.stack(scores, dim=1)
        k_head = max(1, int(S_all.shape[1] / 5))
        S = torch.topk(S_all, k=k_head, dim=1, largest=True).values.mean(dim=1)
        print(f"[pre-sparse][pos={p}] S.min={S.min().item():.4f}  "
              f"S.mean={S.mean().item():.4f}  S.max={S.max().item():.4f}  K={len(ids_p)}")

        always_topm_eff = max(always_topm, int(0.125 * K))
        min_keep_eff = max(min_keep, int(0.35 * K))
        K_keep_eff = max(K_keep, int(0.5 * K))

        if p < getattr(args, "pre_sparse_head_warmup", 8) or \
                p > (getattr(args, "pre_sparse_tail_warmup", 5) + 1e9):
            res_filtered.append(ids_p)
            continue

        keep = set(range(min(always_topm_eff, K)))
        mandatory_ids = getattr(model_wrapper, "_mandatory_punct_ids", None)
        if mandatory_ids is None:
            mandatory_ids = get_mandatory_punct_ids(model_wrapper.tokenizer)
            setattr(model_wrapper, "_mandatory_punct_ids", mandatory_ids)
        keep |= {i for i, tok in enumerate(ids_p) if tok in mandatory_ids}

        tau_eff = getattr(args, "pre_sparse_tau", 0.6)
        good = torch.where(S >= tau_eff)[0].tolist()
        keep.update(good)

        if len(keep) < min_keep_eff:
            rest = [i for i in range(K) if i not in keep]
            add = torch.topk(S[rest], k=min(min_keep_eff - len(keep), len(rest)), largest=True).indices.tolist()
            keep.update([rest[i] for i in add])

        keep_list = sorted(list(keep))
        if len(keep_list) > K_keep_eff:
            pri = list(range(min(always_topm_eff, K))) + [i for i, t in enumerate(ids_p) if t in mandatory_ids]
            pri = list(dict.fromkeys(pri))
            others = [i for i in keep_list if i not in pri]
            top_others = []
            if len(others) > 0:
                top_others = torch.topk(S[others], k=min(K_keep_eff - len(pri), len(others)),
                                        largest=True).indices.tolist()
                top_others = [others[i] for i in top_others]
            keep_list = (pri + top_others)[:K_keep_eff]

        ids_new = [ids_p[i] for i in keep_list]
        res_filtered.append(ids_new)

        print(f"[pre-sparse] pos={p} K:{len(ids_p)} -> {len(ids_new)} (kept)")

    return res_filtered


def to_text_list(x, tokenizer):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            return []
        if isinstance(x[0], str):
            return list(x)
        if isinstance(x[0], int):
            return [tokenizer.decode(list(x), skip_special_tokens=True)]
        out = []
        for t in x:
            if isinstance(t, str):
                out.append(t)
            elif hasattr(t, "tolist"):  # torch.Tensor / np.ndarray
                out.append(tokenizer.decode(list(t.tolist()), skip_special_tokens=True))
            elif isinstance(t, (list, tuple)):
                out.append(tokenizer.decode(list(t), skip_special_tokens=True))
            else:
                out.append(str(t))
        return out
    if hasattr(x, "tolist"):
        arr = x.tolist()
        if isinstance(arr, list) and (len(arr) > 0) and isinstance(arr[0], int):
            return [tokenizer.decode(arr, skip_special_tokens=True)]
        return [str(arr)]
    return [str(x)]


def _recall_report(ref_ids, pool_ids, res_ids):
    pool = set(pool_ids.tolist() if hasattr(pool_ids, "tolist") else pool_ids)
    miss1, miss2 = [], []
    for p, t in enumerate(ref_ids):
        if t not in pool:
            miss1.append((p, int(t)))
        elif p < len(res_ids) and int(t) not in set(res_ids[p]):
            miss2.append((p, int(t)))
    print(f"[Recall] len={len(ref_ids)} | Stage1-miss={len(miss1)} | Stage2-miss={len(miss2)}")
    if miss1: print("[Recall][S1] first misses:", miss1[:10])
    if miss2: print("[Recall][S2] first misses:", miss2[:10])


import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def beam_search_decoder(
    args, model_wrapper, R_Qs, res_ids,
    forced_start_token=None,
    beam_width=12,
    beam_groups=4,
    diversity_lambda=0.35,
    length_norm="avg",
    forced_start_tokens_per_group=None,
    ngram_diversity=2,
    ngram_lambda=0.25,
    G_lm=None,
    beta_glm=0.33,
):
    R_Q2 = R_Qs[1]
    device = model_wrapper.device

    max_steps = getattr(args, "beam_max_steps", None)
    res_iter = res_ids[:max_steps] if max_steps is not None else res_ids
    if len(res_iter) == 0:
        return [], []

    G = max(1, int(beam_groups))
    W = max(1, int(beam_width))
    per_g = max(1, W // G)
    beams_groups = [[([], 0.0)] for _ in range(G)]

    from tqdm import tqdm
    pbar = tqdm(total=len(res_iter), desc="BeamSearch positions", dynamic_ncols=True)
    if hasattr(model_wrapper, "model"):
        model_wrapper.model.eval()

    def last_ngram(seq, n):
        if n <= 0 or len(seq) < n: return None
        return tuple(seq[-n:])

    for pos_idx, pos_candidates in enumerate(res_iter):
        if any(len(bg) == 0 for bg in beams_groups): break
        if len(pos_candidates) == 0: break

        per_group_first_tokens = [None]*G
        if pos_idx == 0:
            if forced_start_tokens_per_group is not None:
                for gi in range(min(G, len(forced_start_tokens_per_group))):
                    tok = forced_start_tokens_per_group[gi]
                    if tok is not None:
                        per_group_first_tokens[gi] = tok
            elif forced_start_token is not None:
                per_group_first_tokens = [forced_start_token]*G

        taken_tokens_this_step = set()
        taken_ngrams_this_step = set()

        new_groups = []
        for g in range(G):
            beams = beams_groups[g]
            B = len(beams)
            K = len(pos_candidates)

            base = torch.tensor([b[0] for b in beams], device=device, dtype=torch.long)
            cand = torch.tensor(pos_candidates, device=device, dtype=torch.long)

            if pos_idx == 0 and per_group_first_tokens[g] is not None:
                cand = torch.tensor([per_group_first_tokens[g]], device=device, dtype=torch.long)
                K = cand.numel()

            base_rep = base.repeat_interleave(K, dim=0)
            cand_rep = cand.repeat(B).view(-1, 1)
            new_tensor = torch.cat([base_rep, cand_rep], dim=1)

            if model_wrapper.is_decoder():
                attn = torch.ones_like(new_tensor, dtype=torch.long)
                hs = model_wrapper.get_layer_inputs(new_tensor, attention_mask=attn)[0]
            elif model_wrapper.is_bert():
                token_type_ids = torch.zeros_like(new_tensor, dtype=torch.long)
                hs = model_wrapper.get_layer_inputs(new_tensor, token_type_ids=token_type_ids)[0]
            else:
                hs = model_wrapper.get_layer_inputs(new_tensor)[0]
            last = hs[:, -1, :]
            dist = check_if_in_span(R_Q2, last, args.dist_norm)
            if G_lm is not None:
                tok_ids = cand_rep.view(-1)  # [B*K]
                if G_lm.device != last.device:
                    G_lm_dev = G_lm.to(last.device)
                else:
                    G_lm_dev = G_lm

                v_tok = G_lm_dev.index_select(0, tok_ids)      # [B*K, d_model]
                lm_score = (last * v_tok).sum(dim=1)          # [B*K]

                lm_score = (lm_score - lm_score.mean()) / (lm_score.std() + 1e-6)

                dist = dist - float(beta_glm) * lm_score


            prev_sum = torch.tensor([b[1] for b in beams], device=device).repeat_interleave(K)
            sum_now  = prev_sum + dist
            score_now = sum_now/float(pos_idx+1) if length_norm=="avg" else sum_now

            cand_tokens = cand_rep.view(-1)
            if len(taken_tokens_this_step) > 0:
                mask_hit = torch.isin(cand_tokens, torch.tensor(list(taken_tokens_this_step),
                                                                device=device, dtype=torch.long))
                score_now = score_now + (diversity_lambda * mask_hit.float())

            if ngram_diversity > 0:
                if ngram_diversity == 1:
                    ngr_cand = cand_tokens
                    mask_ng = torch.isin(ngr_cand, torch.tensor([ng[-1] for ng in taken_ngrams_this_step if ng],
                                                                device=device, dtype=torch.long)) \
                              if taken_ngrams_this_step else torch.zeros_like(score_now, dtype=torch.bool)
                    score_now = score_now + (ngram_lambda * mask_ng.float())
                else:
                    if (pos_idx) >= (ngram_diversity-1) and len(taken_ngrams_this_step)>0:
                        mask_ng = torch.zeros_like(score_now, dtype=torch.bool)
                        for i in range(new_tensor.size(0)):
                            ng = tuple(new_tensor[i].tolist()[-ngram_diversity:])
                            if ng in taken_ngrams_this_step:
                                mask_ng[i] = True
                        score_now = score_now + (ngram_lambda * mask_ng.float())

            k = min(per_g, score_now.numel())
            top_idx = torch.topk(-score_now, k=k, largest=True).indices
            chosen = [(new_tensor[i].tolist(), float(sum_now[i].item())) for i in top_idx.tolist()]
            new_groups.append(chosen)

            if len(chosen) > 0:
                best_seq = chosen[0][0]
                taken_tokens_this_step.add(best_seq[-1])
                if ngram_diversity > 0:
                    ng = last_ngram(best_seq, ngram_diversity)
                    if ng: taken_ngrams_this_step.add(tuple(ng))

        beams_groups = new_groups

        eos_id = getattr(model_wrapper.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            all_eos = True
            for g in range(G):
                if len(beams_groups[g]) == 0 or beams_groups[g][0][0][-1] != eos_id:
                    all_eos = False; break
            if all_eos:
                pbar.update(1)
                pbar.set_postfix_str(f"pos={pos_idx} | all groups EOS stop")
                break

        pbar.update(1)
        alive = sum(len(bg) for bg in beams_groups)
        pbar.set_postfix_str(f"pos={pos_idx} | total_beams={alive}")

    pbar.close()

    all_beams = []
    for g in range(G):
        all_beams += beams_groups[g]
    if length_norm == "avg":
        scored = [ (seq, (s / max(1, len(seq)))) for (seq, s) in all_beams ]
    else:
        scored = [ (seq, s) for (seq, s) in all_beams ]
    scored.sort(key=lambda x: x[1])
    final_sentences = [seq for (seq, s) in scored[:W]]
    final_scores    = [s   for (seq, s) in scored[:W]]
    return final_sentences, final_scores


@torch.no_grad()
def build_global_candidate_pool(args, model_wrapper, residual_grads,
                                R_Qs_original, head_R_Qs_split,
                                eff_len_each, tokenizer):
    word_embeddings = model_wrapper._WTE_CPU
    grad_l1 = residual_grads[model_wrapper.layer_ids[0]]
    d_model = model_wrapper.model.config.hidden_size
    h = model_wrapper.model.config.num_attention_heads
    d_head = d_model // h
    grad_l1_query = grad_l1[:, :d_model].to(args.device).float()
    grad_slices_per_head = [grad_l1_query[:, i*d_head:(i+1)*d_head] for i in range(h)]

    eff_len = int(eff_len_each.max().item())

    def build_pool_unsupervised(R_Q_heads, grad_slices, wte, args, tokenizer):
        head_dists = []
        head_sparse = []
        device = args.device
        V = wte.size(0)
        chunk = getattr(args, "wte_chunk", 4096)

        for R_Q_i, G_i in zip(R_Q_heads, grad_slices):
            G_i_dev = G_i.to(device, non_blocking=True)
            R_i_dev = R_Q_i.to(device, non_blocking=True)

            dists = []
            spars = []
            for st in range(0, V, chunk):
                ed = min(st + chunk, V)
                w_chunk = wte[st:ed].to(device, non_blocking=True)  # [C, d_model]

                proj = torch.matmul(w_chunk, G_i_dev)  # [C, d_head]
                dist_c = check_if_in_span(R_i_dev, proj, args.dist_norm)  # [C]

                g_abs = proj.abs()  # [C, d_head]
                q = getattr(args, "sparse_q", 0.25)
                thr = torch.quantile(g_abs, q=q, dim=1, keepdim=True)  # [C,1]
                s_c = (g_abs <= thr).float().mean(dim=1)  # [C]

                dists.append(dist_c.to('cpu', copy=True))
                spars.append(s_c.to('cpu', copy=True))

                del w_chunk, proj, dist_c, g_abs, thr, s_c
                if cuda_ok():
                    torch.cuda.empty_cache()

            dist = torch.cat(dists, dim=0)  # [V] CPU
            sps = torch.cat(spars, dim=0)  # [V] CPU
            head_dists.append(dist.unsqueeze(0))
            head_sparse.append(sps.unsqueeze(0))

        scores = torch.cat(head_dists, dim=0)  # [H, V] dist
        Ssp = torch.cat(head_sparse, dim=0)  # [H, V] sparsity

        H = scores.size(0)
        frac = getattr(args, "frac_active_heads", 0.5)
        H_act = max(1, int(frac * H))
        head_energy = scores.mean(dim=1)  # [H]
        H_act_idx = torch.topk(head_energy, k=H_act, largest=False).indices

        s_sub = scores[H_act_idx].mean(dim=0)  # [V]
        s_cons = scores[H_act_idx].std(dim=0)  # [V]

        Ssp_act = Ssp[H_act_idx]  # [H_act, V]
        k_sparse = max(1, H_act // 2)
        sparse_head_score = Ssp_act.mean(dim=1)
        topk_sparse_idx = torch.topk(sparse_head_score, k=k_sparse, largest=True).indices
        s_sparse = Ssp_act[topk_sparse_idx].mean(dim=0)  # [V]

        lambda1 = getattr(args, "lambda_sub", 0.8)
        lambda2 = getattr(args, "lambda_cons", 0.5)
        lambda3 = getattr(args, "lambda_sparse", 0.5)
        s_total = (lambda1 * s_sub + lambda2 * s_cons - lambda3 * s_sparse)  # [V]

        fuse_min = s_total
        fuse_mean = s_sub

        def pool_for_k(k):
            idx_total = torch.topk(fuse_min, k=k, largest=False).indices
            idx_sub = torch.topk(fuse_mean, k=max(1, k // 2), largest=False).indices
            pool = torch.unique(torch.cat([idx_total, idx_sub]))

            if getattr(args, 'always_include_eos_period', True):
                extras = torch.tensor(
                    [13, tokenizer.eos_token_id or 50256],
                    device=pool.device
                )
                pool = torch.unique(torch.cat([pool, extras]))
            return pool

        target = getattr(args, "target_pool", 1600)
        k_lo, k_hi = 64, getattr(args, "k_per_head_max", 4096)
        best = pool_for_k(k_hi)

        while k_lo <= k_hi:
            mid = (k_lo + k_hi) // 2
            pool = pool_for_k(mid)
            if pool.numel() >= target:
                best = pool
                k_hi = mid - 1
            else:
                k_lo = mid + 1

        if best.numel() > target:
            best = best[:target]

        best = best.to(device)
        fuse_min = fuse_min.to(device)

        return best, fuse_min

    candidate_pool_indices, fuse_min = build_pool_unsupervised(
        head_R_Qs_split, grad_slices_per_head, word_embeddings, args, tokenizer
    )

    P_front = min(getattr(args, "booster_front_positions", 5), eff_len)
    pos_list = list(range(P_front))
    if eff_len > 0 and (eff_len - 1) not in pos_list:
        pos_list.append(eff_len - 1)

    boost_ids = set()
    for p in pos_list:
        full_emb_pos = model_wrapper.get_embeddings(p)[0]          # [V, d]
        dists_p = check_if_in_span(R_Qs_original[0], full_emb_pos, args.dist_norm)
        m_each = getattr(args, "booster_topm", 800)
        topm = torch.topk(dists_p, k=min(m_each, dists_p.numel()), largest=False).indices
        boost_ids.update(topm.tolist())

    if boost_ids:
        boost_ids = torch.tensor(sorted(boost_ids), device=args.device)
        union = torch.unique(torch.cat([candidate_pool_indices, boost_ids]))

        mandatory = set(boost_ids.tolist())
        if getattr(args, 'always_include_eos_period', True):
            mandatory.update([13, tokenizer.eos_token_id or 50256])
        mandatory = torch.tensor(sorted(mandatory), device=args.device)

        target = getattr(args, "target_pool", 3500)
        if union.numel() > target:
            mask = torch.ones(union.numel(), dtype=torch.bool, device=args.device)
            mand_set = set(mandatory.tolist())
            for i, t in enumerate(union.tolist()):
                if t in mand_set: mask[i] = False
            residual = union[mask]
            need = max(0, target - mandatory.numel())
            if residual.numel() > 0 and need > 0:
                keep_idx = torch.topk(fuse_min[residual], k=min(need, residual.numel()), largest=False).indices
                candidate_pool_indices = torch.cat([mandatory, residual[keep_idx]])
            else:
                candidate_pool_indices = mandatory[:target]
        else:
            candidate_pool_indices = union

    return candidate_pool_indices  # LongTensor

@torch.no_grad()
def position_filter_per_sample(args, model_wrapper, R_Qs_original,
                               candidate_pool_indices, input_batch_single, recall_ref_ids=None):
    tokenizer = model_wrapper.tokenizer
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if "attention_mask" in input_batch_single:
        Lb = int(input_batch_single["attention_mask"].sum(dim=1)[0].item())
    else:
        ids = input_batch_single["input_ids"]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            Lb = int(ids.size(1))
        else:
            Lb = int((ids != pad_id).sum(dim=1)[0].item())
    Lb = min(Lb, args.max_len, tokenizer.model_max_length)

    res_pos_i, res_ids_i, res_types_i = [], [], []
    sentence_ends_i = []

    for p in range(Lb):
        if candidate_pool_indices.numel() == 0:
            print("[Stage2] Candidate pool empty, stop.")
            break

        full_emb_pos = model_wrapper.get_embeddings(p)[0]                         # [V, d]
        candidate_pos_embeds = full_emb_pos.index_select(0, candidate_pool_indices)  # [V_pool, d]
        del full_emb_pos

        K2 = getattr(args, "pos_topk", 256)
        if model_wrapper.is_bert():
            _, top_idx_in_pool, types_in_pool = get_top_B_in_span(
                R_Qs_original[0], candidate_pos_embeds, K2, args.l1_span_thresh, args.dist_norm
            )
        else:
            top_idx_in_pool, = get_top_B_in_span(
                R_Qs_original[0], candidate_pos_embeds, K2, args.l1_span_thresh, args.dist_norm
            )
            types_in_pool = torch.zeros_like(top_idx_in_pool)

        final_token_ids_for_p = candidate_pool_indices[top_idx_in_pool]
        ids_p   = final_token_ids_for_p.tolist()
        types_p = types_in_pool.tolist()

        # EOS
        if eos_id is not None and eos_id in ids_p:
            end_token_ind = ids_p.index(eos_id)
            sentence_token_type = types_p[end_token_ind]
            sentence_ends_i.append((p, sentence_token_type))
            ids_p   = ids_p[:end_token_ind]
            types_p = types_p[:end_token_ind]

        if args.max_ids > 0:
            ids_p   = ids_p[:args.max_ids]
            types_p = types_p[:args.max_ids]

        if len(ids_p) == 0:
            continue

        res_ids_i.append(ids_p)
        res_types_i.append(types_p)
        res_pos_i += [p] * len(ids_p)
    if recall_ref_ids is not None:
        try:
            _recall_report(recall_ref_ids, candidate_pool_indices, res_ids_i)
        except Exception as e:
            print(f"[Recall] report failed: {e}")
    return (res_pos_i, res_ids_i, res_types_i, sentence_ends_i)


def filter_l1(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []

    sentence_ends = []
    p = 0
    n_tokens = 0

    while True:
        print(f'L1 Position {p}')
        embeds = model_wrapper.get_embeddings(p)
        if model_wrapper.is_bert():
            if args.defense_noise is None:
                _, res_ids_new, res_types_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh,
                                                                  args.dist_norm)
            else:
                raise NotImplementedError
        else:
            if args.defense_noise is None:
                _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh,
                                                   args.dist_norm)
            else:
                std_thrs = args.p1_std_thrs if p == 0 else None
                d = get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(d, std_thrs=std_thrs, maxB=max(50 * model_wrapper.args.batch_size,
                                                                             int(0.05 * len(model_wrapper.tokenizer))))
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
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
        if model_wrapper.has_rope():
            break

    return res_pos, res_ids, res_types, sentence_ends


@torch.no_grad()
def _to_cpu_grads(gs):
    return [None if g is None else g.detach().to('cpu', copy=True).float() for g in gs]


@torch.no_grad()
def _pair_dot(ga, gb):
    tot = None
    for a, b in zip(ga, gb):
        if a is None or b is None: continue
        v = (a * b).sum()
        tot = v if tot is None else tot + v
    if tot is None:
        dev = ga[0].device if ga and ga[0] is not None else 'cpu'
        return torch.tensor(0.0, device=dev)
    return tot


def _get_label_for_sample(true_labels, s):
    if isinstance(true_labels, torch.Tensor):
        if true_labels.ndim == 0: return true_labels.clone()
        if true_labels.size(0) > s: return true_labels[s].clone()
        return true_labels.view(-1)[0].clone()
    try:
        return torch.tensor(true_labels[s])
    except Exception:
        if isinstance(true_labels, (list, tuple)) and len(true_labels) > 0:
            return torch.tensor(true_labels[0])
        return torch.tensor(0)


@torch.no_grad()
def _solve_least_squares(decoded_grads, mix_grads0, ridge: float = 1e-8):
    n = len(decoded_grads)
    if n == 0: return torch.empty(0)
    device = 'cpu'
    for g_list in [mix_grads0] + decoded_grads:
        for g in g_list:
            if g is not None: device = g.device; break
        if device != 'cpu': break

    G = torch.zeros((n, n), device=device, dtype=torch.float32)
    b = torch.zeros((n,), device=device, dtype=torch.float32)

    for i in range(n):
        b[i] = _pair_dot(mix_grads0, decoded_grads[i])
        for j in range(i, n):
            gij = _pair_dot(decoded_grads[i], decoded_grads[j])
            G[i, j] = G[j, i] = gij

    if ridge > 0: G[range(n), range(n)] += ridge
    try:
        alpha = torch.linalg.solve(G, b)
    except RuntimeError:
        alpha = torch.linalg.lstsq(G, b).solution
    return alpha


@torch.no_grad()
def _rebuild_residual_from_m0(mix_grads0, decoded_grads, alpha_vec):
    residual = [g.clone() for g in mix_grads0]
    for i, ai in enumerate(alpha_vec.tolist()):
        ci = decoded_grads[i]
        for k in range(len(residual)):
            if residual[k] is not None and ci[k] is not None:
                residual[k].data.add_(ci[k].data, alpha=-float(ai))
    return residual


def create_correct_labels(model_wrapper, true_labels, sample_idx, input_tensor):
    dev = input_tensor.device
    model_cls = model_wrapper.model.__class__.__name__
    cfg = getattr(model_wrapper.model, "config", None)
    num_labels = getattr(cfg, "num_labels", None)

    is_seqcls = ("ForSequenceClassification" in model_cls) or \
                (isinstance(num_labels, int) and num_labels > 0 and \
                 getattr(cfg, "problem_type", None) in [None, "single_label_classification"])
    is_tokencls = ("ForTokenClassification" in model_cls)
    is_causal_lm = ("ForCausalLM" in model_cls) or hasattr(model_wrapper.model, "lm_head")
    is_masked_lm = ("ForMaskedLM" in model_cls)

    lbl_s = _get_label_for_sample(true_labels, sample_idx)

    if is_seqcls and not (is_causal_lm or is_masked_lm or is_tokencls):
        if isinstance(lbl_s, torch.Tensor) and lbl_s.ndim > 0:
            lbl_s = lbl_s.view(-1)[0]
        return lbl_s.long().view(1).to(dev)
    else:
        if not isinstance(lbl_s, torch.Tensor):
            lbl_s = torch.tensor(lbl_s)
        lbl_s = lbl_s.to(dev)
        if lbl_s.ndim == 0:
            lbl_s = lbl_s.view(1, 1)
        elif lbl_s.ndim == 1:
            lbl_s = lbl_s.unsqueeze(0)
        if lbl_s.size(0) != 1: lbl_s = lbl_s[:1, :]

        L_pred = input_tensor.size(1)
        if lbl_s.size(1) != L_pred:
            new_lbl = torch.full((1, L_pred), fill_value=-100, dtype=lbl_s.dtype, device=dev)
            copy_len = min(L_pred, lbl_s.size(1))
            new_lbl[:, :copy_len] = lbl_s[:, :copy_len]
            lbl_s = new_lbl
        return lbl_s



@torch.no_grad()
def _deflate(mix_grads, comp_grads, alpha):
    if alpha == 0.0:
        return
    for i in range(len(mix_grads)):
        gi = mix_grads[i]
        ci = comp_grads[i]
        if gi is None or ci is None:
            continue
        gi.data -= float(alpha) * ci.data


def reconstruct_with_omp(args, device, sample, metric, model_wrapper: ModelWrapper):
    tokenizer = model_wrapper.tokenizer
    sequences, true_labels = sample
    Bsz = len(sequences)

    orig_batch = tokenizer(sequences, padding=True, truncation=True,
                           max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
                           return_tensors='pt', add_special_tokens=False).to(args.device)
    mix_grads_original = _to_cpu_grads(model_wrapper.compute_grads(orig_batch, true_labels))

    print("--- Stage 1: Discovering diverse candidates via Multi-Length Exploration ---")

    _, R_Qs_original, head_R_Qs_split = model_wrapper.get_matrices_expansions(mix_grads_original, B=None,
                                                                              tol=args.rank_tol)
    if R_Qs_original is None:
        print("Failed to build subspaces from the original gradient.")
        return [""] * Bsz, to_text_list([remove_padding(tokenizer, s) for s in orig_batch['input_ids']], tokenizer)

    eff_len_each = (orig_batch['input_ids'] != (tokenizer.pad_token_id or 0)).sum(dim=1)
    candidate_pool_indices = build_global_candidate_pool(args, model_wrapper, mix_grads_original, R_Qs_original,
                                                         head_R_Qs_split, eff_len_each, tokenizer)
    candidate_pool = []
    unique_texts = set()

    unique_lengths = torch.unique(eff_len_each).tolist()
    print(f"Found unique lengths in batch: {unique_lengths}. Exploring each...")

    for length in unique_lengths:
        print(f"\n--- Exploring for length: {length} ---")
        pseudo_batch = {"input_ids": torch.zeros((1, length), dtype=torch.long),
                        "attention_mask": torch.ones((1, length), dtype=torch.long)}
        _, res_ids_s, _, _ = position_filter_per_sample(args, model_wrapper, R_Qs_original, candidate_pool_indices,
                                                        pseudo_batch)
        res_ids_s = [list(map(int, ids)) for ids in res_ids_s]

        G = 4
        start_pool = res_ids_s[0][:max(G, 1)] if len(res_ids_s) > 0 and len(res_ids_s[0]) > 0 else []
        forced_starts = start_pool + [None] * (G - len(start_pool))

        G_lm = model_wrapper.model.get_input_embeddings().weight.detach()

        seqs_s, _ = beam_search_decoder(
            args, model_wrapper, R_Qs_original, res_ids_s,
            beam_width=16,
            beam_groups=G,
            diversity_lambda=0.35,
            length_norm="avg",
            forced_start_tokens_per_group=forced_starts,  # ★
            ngram_diversity=2,
            ngram_lambda=0.25,
            G_lm=G_lm,
            beta_glm=getattr(args, "beta_glm", 0.33),
        )
        for cand_seq in seqs_s:
            cand_seq = cand_seq[:length]
            if not cand_seq:
                continue

            decoded_text = tokenizer.decode(cand_seq, skip_special_tokens=True)

            if decoded_text and decoded_text not in unique_texts:
                unique_texts.add(decoded_text)
                candidate_pool.append(cand_seq)
                print(f"Added new candidate to pool (total: {len(candidate_pool)}): {decoded_text}")

    print(f"\n--- Candidate discovery finished. Total unique candidates: {len(candidate_pool)} ---")

    print("\n--- Stage 1.5: Clustering candidates and selecting representatives ---")

    candidate_pool.sort(key=len, reverse=True)

    clusters = []

    for cand_seq in candidate_pool:
        found_cluster = False
        for cluster in clusters:
            similarity = metric.compute(
                predictions=[tokenizer.decode(cand_seq, skip_special_tokens=True)],
                references=[tokenizer.decode(cluster[0], skip_special_tokens=True)]
            )['rougeL']
            if similarity > 0.7:
                cluster.append(cand_seq)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([cand_seq])

    representative_pool = []
    for C in clusters:
        best_seq = None
        best_score = -1e9
        for ids in C:
            text = model_wrapper.tokenizer.decode(ids, skip_special_tokens=True)
            eos_bonus = 1.0 if (text.endswith(".") or text.endswith("!") or text.endswith("?")) else 0.0
            rep_uni = 0.0
            if len(ids) >= 2:
                rep_uni = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1]) / max(1, len(ids) - 1)
            rep_bi = 0.0
            if len(ids) >= 3:
                bi = list(zip(ids[:-1], ids[1:]))
                if bi:
                    from collections import Counter
                    c = Counter(bi)
                    rep_bi = sum(v - 1 for v in c.values() if v > 1) / len(bi)
            length_pen = 0.0 if len(ids) >= 4 else 0.2
            score = eos_bonus - (1.5 * rep_uni + 1.0 * rep_bi + length_pen)
            if score > best_score:
                best_score = score
                best_seq = ids
        representative_pool.append(best_seq)
    print(f"Clustering reduced {len(candidate_pool)} candidates to {len(representative_pool)} unique representatives.")

    for i, seq_ids in enumerate(representative_pool):
        text = model_wrapper.tokenizer.decode(seq_ids, skip_special_tokens=True)
        print(f"[Cluster-REP #{i}] {text}")


    candidate_pool = representative_pool

    print("\n--- Stage 2: Orthogonal Matching Pursuit to select the best combination ---")

    print("Pre-computing gradients for all candidates...")
    candidate_grads = []
    for seq_ids in tqdm(candidate_pool, desc="Computing Grads"):
        t = torch.tensor(seq_ids, dtype=torch.long).unsqueeze(0).to(args.device)
        pred_labels = create_correct_labels(model_wrapper, true_labels, 0, t)
        with torch.enable_grad():
            grads = model_wrapper.compute_grads(orig_batch.__class__({"input_ids": t}), pred_labels)
        candidate_grads.append(_to_cpu_grads(grads))

    residual = [g.clone() for g in mix_grads_original]
    selected_indices = []

    for k in range(Bsz):
        print(f"\nOMP Step {k + 1}/{Bsz}")
        best_corr, best_idx = -1, -1
        for i in range(len(candidate_pool)):
            if i in selected_indices: continue
            num = _dot_sum(residual, candidate_grads[i], model_wrapper, ignore_final_layer=False)
            den = (_dot_sum(candidate_grads[i], candidate_grads[i], model_wrapper, ignore_final_layer=False)) ** 0.5
            corr = abs(num / (den + 1e-9))

            gamma = 0.2
            length_bonus = len(candidate_pool[i]) ** gamma
            final_score = corr * length_bonus

            if final_score > best_corr:
                best_corr, best_idx = final_score, i

        if best_idx == -1: break
        selected_indices.append(best_idx)
        print(f"Selected candidate {best_idx}: {tokenizer.decode(candidate_pool[best_idx], skip_special_tokens=True)}")

        active_grads = [candidate_grads[i] for i in selected_indices]
        alpha = _solve_least_squares(active_grads, mix_grads_original, ridge=1e-6)
        residual = _rebuild_residual_from_m0(mix_grads_original, active_grads, alpha)

    prediction_ids = [candidate_pool[i] for i in selected_indices]
    while len(prediction_ids) < Bsz:
        prediction_ids.append([])

    prediction = to_text_list(prediction_ids, tokenizer)
    reference = to_text_list([remove_padding(tokenizer, s) for s in orig_batch['input_ids']], tokenizer)

    return prediction, reference

def print_metrics(args, res, suffix):
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        fm = res[metric] * 100
        print(f'{metric:10} | fm: {fm:.3f}', flush=True)
        if args.neptune:
            args.neptune[f'logs/{metric}-fm_{suffix}'].log(fm)
    sum_12_fm = (res['rouge1'] + res['rouge2']) * 100
    print(f'r1fm+r2fm = {sum_12_fm:.3f}', flush=True)
    if args.neptune:
        args.neptune[f'logs/r1fm+r2fm_{suffix}'].log(sum_12_fm)


def main():

    device = torch.device(args.device)
    metric = load_metric('rouge', cache_dir=args.cache_dir)
    print("Creating TextDataset with:", args.dataset, args.split)
    dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)

    model_wrapper = ModelWrapper(args)

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()


    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
        t_input_start = time.time()
        # sample = dataset[i]  # (seqs, labels)

        # sequences = [
        #     "Les cafés de la ville sont calmes et chaleureux.",
        #     # "Dans le parc au petit matin, les coureurs respirent l'air humide.",
        #     # "Un étudiant est assis dans un coin de la bibliothèque.",
        #     "Le marché du week-end est animé, les marchands crient pour attirer les passants.",
        #     "Il marche au bord de la mer avec son sac à dos.",
        #     "La ville s’illumine de milliers de lumières à la tombée de la nuit."
        #     # "L’odeur chaude des nouilles s’échappe d’un petit restaurant dans la ruelle.",
        #     # "Les gouttes de pluie glissent le long des vitres."
        # ]

        # sequences = [
        #     "城市里的咖啡馆安静又温暖，窗外行人匆匆而过，店内的人却沉浸在自己的小世界里。",
        #     "清晨的公园里，跑步的人呼吸着湿润空气，小鸟在树上叽叽喳喳迎接新的一天。",
        #     "图书馆角落坐着一位学生，翻阅厚重的书籍，记下那些对未来有帮助的知识。",
        #     # "周末的市场很热闹，摊主们大声招呼路人，售卖新鲜蔬果和各种小吃。",
        #     # "他背着旅行包走在海边，海风吹乱头发，脚印被海水一浪一浪冲刷掉。",
        #     # "入夜后的城市亮起无数灯光，车流不断，人们仍在追赶属于自己的目标。",
        #     # "小巷里的面馆飘出热腾腾的香气，让路人肚子咕咕叫，忍不住进去点一碗面。",
        #     "雨滴顺着窗玻璃滑落，屋内的灯光映出温柔影子，让人感觉安心又放松。"
        # ]
        # sequences = [
        #     "Im Café der Stadt herrscht eine ruhige, warme Atmosphäre, während draußen die Menschen eilig vorbeigehen.",
        #     "Im Park am frühen Morgen atmen Jogger die feuchte Luft ein, während Vögel fröhlich den neuen Tag begrüßen.",
        #     "In einer Ecke der Bibliothek sitzt ein Student, blättert in dicken Büchern und notiert wichtiges Wissen für die Zukunft.",
        #     "Die Regentropfen gleiten langsam an der Fensterscheibe hinunter, und das weiche Licht im Raum schafft eine beruhigende Stimmung."
        # ]
        sequences = [
            "Nel caffè della città regna un’atmosfera tranquilla e accogliente, mentre fuori la gente passa in fretta.",
            "Nel parco del mattino presto i corridori respirano l’aria umida e gli uccelli salutano allegramente il nuovo giorno.",
            "In un angolo della biblioteca uno studente sfoglia libri spessi e annota le conoscenze utili per il futuro.",
            "Le gocce di pioggia scendono lentamente sul vetro della finestra, e la luce soffusa crea un ambiente calmo e rilassante."
        ]

        # sequences = [
        #     # "Artificial intelligence has evolved far beyond simple automation, emerging as a creative collaborator capable of generating music, visual art, literature, and even scientific hypotheses. By analyzing massive datasets, AI models can detect hidden patterns and recombine ideas in ways that often surprise human creators. Yet this collaboration between humans and machines raises philosophical questions: if creativity once defined humanity’s uniqueness, what happens when algorithms begin to compose symphonies or paint in styles indistinguishable from humans? Some argue that AI expands creative potential by removing technical barriers, allowing anyone to express ideas effortlessly, while others fear that overreliance on generative tools may dilute originality and homogenize culture. Ultimately, the challenge lies not in resisting AI creativity but in learning to direct it—using algorithms as instruments of inspiration rather than replacements for imagination.",
        #     "Quantum computing represents a paradigm shift that could reshape the boundaries of computational power. By exploiting superposition and entanglement, qubits can encode vast amounts of information in parallel, enabling algorithms like Shor’s and Grover’s to outperform their classical counterparts exponentially. However, practical quantum computing remains elusive due to noise, decoherence, and scalability constraints. Researchers are developing quantum error correction codes, cryogenic hardware, and hybrid quantum-classical architectures to make progress toward fault-tolerant systems. Governments and tech giants are investing billions to achieve quantum advantage, while scientists explore applications in cryptography, materials science, and artificial intelligence. Yet beyond the technical race lies a broader question: how will society secure and regulate such immense power once quantum systems can break today’s encryption or simulate molecules beyond classical reach?",
        #     # "The accelerating pace of climate change threatens not only ecosystems but also human survival. Rising sea levels, prolonged droughts, and extreme weather events are forcing governments and citizens to rethink industrial growth, energy consumption, and urban development. The Paris Agreement established a global framework for reducing carbon emissions, yet implementation remains uneven, as economic priorities often clash with environmental goals. Renewable energy technologies—such as wind, solar, and hydrogen—offer viable alternatives, but large-scale adoption requires infrastructure investment and policy reform. Beyond technology, the climate crisis demands cultural transformation: people must shift from a mindset of exploitation toward one of stewardship, recognizing that sustainability is not a burden but an opportunity to design a fairer, more resilient planet for future generations.",
        #     "The internet began as a decentralized network designed for open information exchange, but over time it has evolved into a highly commercialized and algorithm-driven ecosystem. Platforms powered by machine learning now curate what billions of users see daily, shaping opinions, behaviors, and even elections. While this connectivity has democratized knowledge and empowered marginalized voices, it has also concentrated influence in a handful of corporations that collect unprecedented amounts of personal data. Cybersecurity threats, misinformation, and online polarization have become defining challenges of our age. The future of the internet depends on whether societies can restore transparency, interoperability, and user control. Achieving true digital freedom will require new governance models—balancing innovation with ethics—to ensure that technology remains a tool for empowerment rather than manipulation."
        # ]
        labels = torch.zeros(len(sequences), dtype=torch.long)
        sample = (sequences, labels)
        # print(sample)

        print(f'Running input #{i} of {args.n_inputs}.', flush=True)
        if args.neptune:
            args.neptune['logs/curr_input'].log(i)

        print('reference: ', flush=True)
        for seq in sample[0]:
            print('========================', flush=True)
            print(seq, flush=True)

        print('========================', flush=True)

        prediction, reference = reconstruct_with_omp(args, device, sample, metric, model_wrapper)
        predictions += prediction
        references += reference

        print(f'Done with input #{i} of {args.n_inputs}.', flush=True)
        print('reference: ', flush=True)
        for seq in reference:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        print('predicted: ', flush=True)
        for seq in prediction:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        tok = model_wrapper.tokenizer
        prediction_texts = to_text_list(prediction, tok)
        reference_texts = to_text_list(reference, tok)

        print('[Curr input metrics]:', flush=True)
        res = metric.compute(predictions=prediction_texts, references=reference_texts)
        print_metrics(args, res, suffix='curr')

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}', flush=True)
        print()
        print()

    print('Done with all.', flush=True)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)


if __name__ == '__main__':
    print("Dataset argument:", args.dataset)
    main()
