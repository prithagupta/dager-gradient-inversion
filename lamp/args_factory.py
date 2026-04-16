import argparse
import logging

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


def get_args():
    parser = argparse.ArgumentParser(description='LAMP attack')

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune',
                        default=None)
    parser.add_argument('--neptune-label', type=str, help='name of the run', required=False, default=None)

    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101)
    parser.add_argument('--baseline', action='store_true', help='use baseline defaults + disable all new improvements')
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes', 'glnmario/ECHR'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--use_hf_split', action='store_true',
                        help='Use the official Hugging Face validation split for GLUE datasets. '
                             'By default, LAMP keeps its original train-subset split protocol.')
    parser.add_argument('--loss', choices=['cos', 'dlg', 'tag'], required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True)  # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)
    parser.add_argument('--defense_noise', type=float, default=None)  # add noise to true grads
    parser.add_argument('--defense_pct_mask', type=float, default=None)  # mask some percentage of gradients

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased')

    # Frozen params
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--opt_alg', choices=['adam', 'bfgs', 'bert-adam'], default='adam')
    parser.add_argument('--n_steps', type=int, default=2000)  #
    parser.add_argument('--init_candidates', type=int, default=500)  #
    parser.add_argument('--init', choices=['lm', 'random'], default='random')
    parser.add_argument('--use_swaps', type=bool, default=True)  #
    parser.add_argument('--no-use_swaps', dest='use_swaps', action='store_false')
    parser.add_argument('--use_swaps_at_end', action='store_true')
    parser.add_argument('--swap_burnin', type=float, default=0.1)
    parser.add_argument('--swap_every', type=int, default=75)
    parser.add_argument('--use_embedding', action='store_true')
    parser.add_argument('--know_padding', type=bool, default=True)
    parser.add_argument('--init_size', type=float, default=3.98)  #
    parser.add_argument('--lr_decay_type', type=str, default='StepLR')

    # Tuneable params
    # Ours:         coeff_preplexity, coeff_reg, lr, lr_decay
    # Baselines:    lr, lr_decay, tag_factor
    parser.add_argument('--coeff_perplexity', type=float, default=0.1)  #
    parser.add_argument('--coeff_reg', type=float, default=0.1)  #
    parser.add_argument('--lr', type=float, default=0.01)  # TAG best: 0.1
    parser.add_argument('--lr_decay', type=float, default=0.9)  # TAG best: 0.985
    parser.add_argument('--tag_factor', type=float, default=None)  # TAG best: 1e-3
    parser.add_argument('--grad_clip', type=float, default=None)  # TAG best: 1, ours 0.5, only applicable to BERT_Large
    parser.add_argument('--lr_max_it', type=int, default=None)

    # Debug params
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--time_limit', type=int, default=None)

    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    if not args.neptune is None:
        assert not args.neptune_label is None
        assert len(args.neptune_label) > 0

    # Defaults above are for Ours, use different defaults if running one of the baseline methods
    if args.baseline:
        args.init_candidates = 1
        args.use_swaps = False
        args.init_size = -1
        args.coeff_perplexity = 0.0
        args.coeff_reg = 0.0

    if args.lr_max_it is None:
        args.lr_max_it = args.n_steps
    if args.use_swaps_at_end:
        args.use_swaps = False

    return args
