import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def _auto_device():
    import torch

    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _resolve_device(requested_device):
    import torch

    if requested_device in (None, 'auto'):
        return _auto_device()
    if requested_device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if requested_device == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    if requested_device == 'cpu':
        return 'cpu'

    resolved_device = _auto_device()
    logger.info(
        f"Requested device '{requested_device}' is unavailable. Falling back to '{resolved_device}'.",
    )
    return resolved_device


def _resolve_attn_implementation(requested_implementation, model_path, task):
    if requested_implementation == 'eager':
        return 'eager'
    if requested_implementation == 'sdpa':
        return 'sdpa'

    model_path_lower = model_path.lower()
    if task == 'seq_class' and (
            model_path in ['gpt2', 'openai-community/gpt2-large'] or 'gemma' in model_path_lower
    ):
        return 'eager'
    return 'sdpa'


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='DAGER attack')

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune',
                        default=None)
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')
    parser.add_argument('--label', type=str, default='name of the run', required=False)

    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101)
    parser.add_argument('--dataset',
                        choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', 'stanfordnlp/imdb', 'glnmario/ECHR'],
                        required=True)
    parser.add_argument('--task', choices=['seq_class', 'next_token_pred'], required=True)
    parser.add_argument('--pad', choices=['right', 'left'], default='right')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--use_hf_split', action='store_true',
                        help='Use the official Hugging Face validation split for GLUE datasets. '
                             'By default, DAGER keeps its original train-subset split protocol.')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True)  # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--finetuned_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--device_grad', type=str, default='cpu')
    parser.add_argument('--attn_implementation', type=str, default='auto', choices=['auto', 'sdpa', 'eager'])

    parser.add_argument('--precision', type=str, default='full', choices=['8bit', 'half', 'full', 'double'])
    parser.add_argument('--parallel', type=int, default=100)
    parser.add_argument('--grad_b', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--rank_tol', type=float, default=None)
    parser.add_argument('--rank_cutoff', type=int, default=20)
    parser.add_argument('--l1_span_thresh', type=float, default=1e-5)
    parser.add_argument('--l2_span_thresh', type=float, default=1e-3)
    parser.add_argument('--l1_filter', choices=['maxB', 'all'], required=True)
    parser.add_argument('--l2_filter', choices=['overlap', 'non-overlap'], required=True)
    parser.add_argument('--distinct_thresh', type=float, default=0.7)
    parser.add_argument('--max_ids', type=int, default=-1)
    parser.add_argument('--maxC', type=int, default=10000000)
    parser.add_argument('--reduce_incorrect', type=int, default=0)
    parser.add_argument('--n_incorrect', type=int, default=None)

    # FedAVG
    parser.add_argument('--algo', type=str, default='sgd', choices=['sgd', 'fedavg'])
    parser.add_argument('--avg_epochs', type=int, default=None)
    parser.add_argument('--b_mini', type=int, default=None)
    parser.add_argument('--avg_lr', type=float, default=None)
    parser.add_argument('--dist_norm', type=str, default='l2', choices=['l1', 'l2'])

    # DP
    parser.add_argument('--defense_noise', type=float, default=None)  # add noise to true grads
    parser.add_argument('--max_len', type=int, default=1e10)
    parser.add_argument('--p1_std_thrs', type=float, default=5)
    parser.add_argument('--l2_std_thrs', type=float, default=5)
    parser.add_argument('--dp_l2_filter', type=str, default='maxB', choices=['maxB', 'outliers'])
    parser.add_argument('--defense_pct_mask', type=float, default=None)  # mask some percentage of gradients

    # Dropout
    parser.add_argument('--grad_mode', type=str, default='eval', choices=['eval', 'train'])

    # Rebuttal experiments
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])

    # Hybrid DAGER+LAMP attack
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.89)
    parser.add_argument('--coeff_perplexity', type=float, default=0.02)
    parser.add_argument('--coeff_reg', type=float, default=1.0)
    parser.add_argument('--init_size', type=float, default=3.98)
    parser.add_argument('--grad_loss', type=str, default='cos', choices=['cos', 'dlg', 'tag'])
    parser.add_argument('--tag_factor', type=float, default=0.01)
    parser.add_argument('--hybrid_temperature', type=float, default=0.1)
    parser.add_argument('--hybrid_init_noise', type=float, default=0.01)
    parser.add_argument('--hybrid_project_every', type=int, default=0,
                        help='If >0, snap continuous embeddings to DAGER candidates every N steps. '
                             'Default 0 keeps optimization continuous and only projects for selection.')
    parser.add_argument('--hybrid_init_mode', type=str, default='dager',
                        choices=['dager', 'candidate_random'],
                        help='Hybrid initialization mode. "dager" uses decoder-selected DAGER ids when available; '
                             '"candidate_random" samples initial tokens from the per-position DAGER candidate sets.')
    parser.add_argument('--hybrid_use_lm_prior', type=str, default='true', choices=['true', 'false'],
                        help='Whether to use the hybrid LM prior when available. Defaults to true.')
    parser.add_argument('--hybrid_projection_mode', type=str, default='candidate_final',
                        choices=['candidate_final', 'candidate_periodic', 'none'],
                        help='How to discretize hybrid embeddings. "candidate_final" matches the current default: '
                             'continuous optimization with final projection to DAGER candidates only. '
                             '"candidate_periodic" also snaps to DAGER candidates during optimization based on '
                             '--hybrid_project_every. "none" disables DAGER-candidate projection and decodes by '
                             'nearest full-vocab embeddings.')
    parser.add_argument('--print_every', type=int, default=50)

    # LoRA
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)

    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.device = _resolve_device(args.device)
    args.attn_implementation = _resolve_attn_implementation(
        args.attn_implementation,
        args.model_path,
        args.task,
    )

    if args.n_incorrect is None:
        args.n_incorrect = args.batch_size

    if args.neptune is not None:
        import neptune.new as neptune
        assert ('label' in args)
        nep_par = {'project': f"{args.neptune}", 'source_files': ["*.py"]}
        if args.neptune_offline:
            nep_par['mode'] = 'offline'
            args.neptune_id = 'DAG-0'

        run = neptune.init(**nep_par)
        args_dict = vars(args)
        run[f"parameters"] = args_dict
        args.neptune = run
        if not args.neptune_offline:
            logger.info('waiting...')
            start_wait = time.time()
            args.neptune.wait()
            logger.info("waited: %s", time.time() - start_wait)
            args.neptune_id = args.neptune['sys/id'].fetch()
        logger.info('\n\n\nArgs: %s\n\n\n', " ".join(argv))
    return args
