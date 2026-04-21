import logging

import numpy as np
import torch

from utils.functional import get_layer_decomp
from utils.models import ModelWrapper
from utils.models import _is_gpt2_path

logger = logging.getLogger(__name__)


class SOMPModelWrapper(ModelWrapper):
    """DAGER ModelWrapper extension with SOMP-specific subspace helpers.

    This class intentionally lives outside ``utils.models`` so the existing
    DAGER and hybrid attacks keep their current behavior.
    """

    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        self._WTE_CPU = self.get_input_embeddings_weight().detach().cpu().float()
        self.wte_grad_id = self._find_first_parameter_index(
            [
                "wte.weight",
                "embed_tokens.weight",
                "word_embeddings.weight",
            ]
        )

    def _find_first_parameter_index(self, name_parts):
        for idx, (name, _) in enumerate(self.model.named_parameters()):
            if any(part in name for part in name_parts):
                return idx
        return None

    def _needs_qkv_transpose(self):
        return _is_gpt2_path(self.args.model_path)

    def _query_grad(self, grad):
        if self._needs_qkv_transpose():
            return grad.T
        return grad

    def get_matrices_expansions(self, true_grads, B=None, tol=None):
        """Return global layer subspaces and optional head-wise query subspaces.

        The base DAGER wrapper returns ``(B, R_Qs)``. SOMP needs the same
        subspaces plus per-attention-head query subspaces for the sparse
        candidate-pool stage, so this subclass returns ``(B, R_Qs, head_R_Qs)``.
        """
        if tol is None:
            tol = self.args.rank_tol

        if B is None:
            max_rank = 0
            for layer_id in self.layer_ids[: min(10, len(self.layer_ids))]:
                grad = true_grads[layer_id]
                if grad is None:
                    continue
                grad = self._query_grad(grad)
                grad_np = grad.detach().float().cpu().numpy()
                rank = np.linalg.matrix_rank(grad_np, tol=tol)
                max_rank = max(max_rank, rank)
            B = max_rank

        if self.args.algo == "fedavg":
            B += 60

        if hasattr(self, "emb_size"):
            B = min(B, self.emb_size - self.args.rank_cutoff)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is not None:
            B = min(B, hidden_size - self.args.rank_cutoff)
        B = max(int(B), 1)

        R_Qs = []
        for i in range(self.args.n_layers):
            grad_q = true_grads[self.layer_ids[i]]
            grad_q = self._query_grad(grad_q)
            _, R_Q = get_layer_decomp(grad_q, B=B, tol=tol, upcast=(self.args.precision == "half"))
            R_Qs.append(R_Q.to(self.args.device))

        head_R_Qs = None
        if getattr(self.args, "headwise_factorization", True):
            head_R_Qs = self._get_headwise_query_subspaces(true_grads, B=B, tol=tol)

        return B, R_Qs, head_R_Qs

    def _get_headwise_query_subspaces(self, true_grads, B, tol):
        grad_l1 = true_grads[self.layer_ids[0]]
        grad_l1 = self._query_grad(grad_l1)

        d_model = getattr(self.model.config, "hidden_size", self.emb_size)
        n_heads = getattr(self.model.config, "num_attention_heads", None)
        if n_heads is None:
            n_heads = getattr(self.model.config, "n_head", None)
        if n_heads is None or n_heads <= 0:
            logger.info("SOMP head-wise factorization disabled: cannot infer number of heads.")
            return None

        d_head = d_model // n_heads
        query_grad = grad_l1[:, :d_model]
        head_R_Qs = []
        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = (head_idx + 1) * d_head
            head_grad = query_grad[:, start:end]
            head_rank = min(B, max(1, min(head_grad.shape) - 1))
            _, R_Q_i = get_layer_decomp(
                head_grad,
                B=head_rank,
                tol=tol,
                upcast=(self.args.precision == "half"),
            )
            head_R_Qs.append(R_Q_i.to(self.args.device))
        return head_R_Qs
