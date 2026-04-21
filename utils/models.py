import logging
import os
from types import SimpleNamespace

import numpy as np
import peft
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from constants import config
from utils.functional import get_layer_decomp
from utils.partial_models import add_partial_forward_gpt2, add_partial_forward_bert, add_partial_forward_llama

logger = logging.getLogger(__name__)

GPT2_MODEL_PATHS = ['gpt2', 'openai-community/gpt2', 'openai-community/gpt2-large']
MODEL_PATH_ALIASES = {
    'gemma-2b': 'google/gemma-2b',
    'gemma_2b': 'google/gemma-2b',
    'gemma2b': 'google/gemma-2b',
    'vault_gemma': 'google/vaultgemma-1b',
    'vault-gemma': 'google/vaultgemma-1b',
    'vaultgemma': 'google/vaultgemma-1b',
}
LLAMA_MODEL_PATHS = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Llama-3.1-8B',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3.1-8B',
    'meta-llama/Meta-Llama-3-70B',

]
GEMMA_MODEL_PATHS = [
    'google/gemma-2b',
    'google/gemma-7b',
    'google/gemma-2-2b',
    'google/gemma-2-9b',
    'google/gemma-2-27b',
    'google/gemma-2-2b-it',
    'google/gemma-3-1b-it',
    'google/gemma-3-4b-it',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
]
VAULT_GEMMA_MODEL_PATHS = [
    'google/vaultgemma-1b',
]
LOCAL_MODEL_DIR_ALIASES = {
    'gpt2': ['openai-community__gpt2', 'gpt2'],
    'openai-community/gpt2': ['openai-community__gpt2', 'gpt2'],
    'openai-community/gpt2-large': ['openai-community__gpt2-large'],
    'meta-llama/Meta-Llama-3.1-8B': ['meta-llama__Meta-Llama-3.1-8B'],
    'meta-llama/Meta-Llama-3-8B': ['meta-llama__Meta-Llama-3-8B'],
    'meta-llama/Llama-2-7b-hf': ['meta-llama__Llama-2-7b-hf'],
    'meta-llama/Llama-2-70b-hf': ['meta-llama__Llama-2-70b-hf'],
    'google/gemma-2b': ['google__gemma-2b'],
    'google/gemma-2-2b': ['google__gemma-2-2b'],
    'google/vaultgemma-1b': ['google__vaultgemma-1b'],
}


def _normalize_model_path(model_path):
    return MODEL_PATH_ALIASES.get(model_path.lower(), model_path)


def _is_offline_mode():
    return any(
        str(os.environ.get(flag, "")).lower() in {"1", "true", "yes"}
        for flag in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]
    )


def _sanitize_model_dir_name(model_path):
    return model_path.replace("/", "__")


def _resolve_local_model_path(model_path, cache_dir):
    if not model_path:
        return model_path

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        return model_path

    if cache_dir is None:
        return model_path

    candidate_dir_names = []
    normalized = _normalize_model_path(model_path)
    for key in {model_path, normalized}:
        candidate_dir_names.extend(LOCAL_MODEL_DIR_ALIASES.get(key, []))
        candidate_dir_names.append(_sanitize_model_dir_name(key))
        candidate_dir_names.append(key)

    seen = set()
    for dir_name in candidate_dir_names:
        if dir_name in seen:
            continue
        seen.add(dir_name)
        candidate_path = os.path.join(cache_dir, dir_name)
        if os.path.isdir(candidate_path) and os.path.exists(os.path.join(candidate_path, "config.json")):
            logger.info("Resolved model path %s -> local cache directory %s", model_path, candidate_path)
            return candidate_path

    return model_path


def _is_gpt2_path(model_path):
    return model_path in GPT2_MODEL_PATHS


def _is_llama_path(model_path):
    return model_path in LLAMA_MODEL_PATHS


def _is_gemma_path(model_path):
    model_path = _normalize_model_path(model_path)
    return model_path in GEMMA_MODEL_PATHS or 'gemma' in model_path.lower()


def _is_vault_gemma_path(model_path):
    model_path = _normalize_model_path(model_path)
    return model_path in VAULT_GEMMA_MODEL_PATHS or 'vaultgemma' in model_path.lower()


def _is_supported_model_path(model_path):
    model_path = _normalize_model_path(model_path)
    return model_path in ['bert-base-uncased'] or _is_gpt2_path(model_path) or _is_llama_path(
        model_path) or _is_gemma_path(model_path)


def _maybe_add_legacy_llama_rope_config(model_kwargs, model_path, attn_implementation):
    """Patch Llama 3.1 configs for older Transformers releases.

    Transformers versions before Llama-3.1 support reject the newer
    ``{"rope_type": "llama3", ...}`` shape and only accept ``type``/``factor``.
    For short attack sequences this legacy dynamic approximation is enough to
    load the model without changing the rest of the DAGER code path.
    """
    if 'Llama-3.1' not in model_path:
        return

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        return

    config_lookup_kwargs = {}
    for key in ('cache_dir', 'token', 'local_files_only'):
        if key in model_kwargs:
            config_lookup_kwargs[key] = model_kwargs[key]

    pretrained_name = model_kwargs['pretrained_model_name_or_path']
    config_dict, _ = LlamaConfig.get_config_dict(pretrained_name, **config_lookup_kwargs)
    try:
        LlamaConfig.from_dict(dict(config_dict), attn_implementation=attn_implementation)
        return
    except ValueError as exc:
        if 'rope_scaling' not in str(exc):
            raise

    rope_scaling = config_dict.get('rope_scaling')
    if not isinstance(rope_scaling, dict) or rope_scaling.get('rope_type') != 'llama3':
        raise

    legacy_config = dict(config_dict)
    legacy_config['rope_scaling'] = {
        'type': 'dynamic',
        'factor': float(rope_scaling.get('factor', 1.0)),
    }
    model_kwargs['config'] = LlamaConfig.from_dict(
        legacy_config,
        attn_implementation=attn_implementation,
    )
    logger.info(
        "Patched Llama 3.1 rope_scaling for legacy Transformers: %s -> %s",
        rope_scaling,
        legacy_config['rope_scaling'],
    )


def _ensure_vault_gemma_available(model_path):
    if not _is_vault_gemma_path(model_path):
        return

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING['vaultgemma']
        return
    except KeyError as exc:
        import transformers
        raise RuntimeError(
            "VaultGemma cannot be loaded in this Python environment because Transformers "
            f"{transformers.__version__} does not recognize model_type='vaultgemma'. "
            "Upgrade the environment to a Transformers release with VaultGemma support, "
            "or use google/gemma-2b/google/gemma-2-2b for the current seq_class attack. "
            "The repository-side VaultGemma sequence-classification wrapper will be used "
            "once the architecture is available in Transformers."
        ) from exc


class CausalLMSequenceClassifier(nn.Module):
    """Small sequence-classification head for decoder-only causal-LM models."""

    def __init__(self, causal_lm, num_labels=2):
        super().__init__()
        self.model = causal_lm.model
        self.config = causal_lm.config
        self.num_labels = getattr(self.config, 'num_labels', num_labels)
        self.config.num_labels = self.num_labels
        self.config.problem_type = getattr(self.config, 'problem_type', None)
        self.score = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)
        first_param = next(self.model.parameters())
        self.score.to(device=first_param.device, dtype=first_param.dtype)
        if hasattr(causal_lm, 'lm_head'):
            self.lm_head = causal_lm.lm_head

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.score(hidden_states).float()

        if attention_mask is None:
            sequence_lengths = torch.full(
                (hidden_states.shape[0],),
                hidden_states.shape[1] - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            flipped_mask = torch.flip(attention_mask.long(), dims=[1])
            sequence_lengths = attention_mask.shape[1] - 1 - flipped_mask.argmax(dim=1)
        pooled_logits = logits[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(pooled_logits, labels.view(-1))

        return SimpleNamespace(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModelWrapper():
    def __init__(self, args):
        args.model_path = _normalize_model_path(args.model_path)
        assert _is_supported_model_path(args.model_path), \
            'Model is not yet supported - add it to assertion list and specify implementation details'
        access_token = os.environ.get('HF_TOKEN')
        self.args = args
        self._logged_llama_non_cuda_float32 = False
        self.resolved_model_path = _resolve_local_model_path(args.model_path, args.cache_dir)
        model_kwargs = {'cache_dir': args.cache_dir} if args.cache_dir is not None else {}

        model_source = self.resolved_model_path if args.finetuned_path is None or args.train_method == 'lora' else args.finetuned_path
        model_kwargs['pretrained_model_name_or_path'] = model_source
        model_kwargs['attn_implementation'] = args.attn_implementation
        if access_token is not None:
            model_kwargs['token'] = access_token
        if _is_offline_mode():
            model_kwargs['local_files_only'] = True

        if args.hidden_act is not None and _is_gpt2_path(args.model_path):
            model_kwargs['activation_function'] = args.hidden_act
        elif args.hidden_act is not None and (_is_llama_path(args.model_path) or _is_gemma_path(args.model_path)):
            model_kwargs['hidden_act'] = args.hidden_act

        if args.precision == '8bit':
            model_kwargs['load_in_8bit'] = True
        if args.precision == 'half':
            model_kwargs['torch_dtype'] = torch.float16
        if args.precision == 'double':
            model_kwargs['torch_dtype'] = torch.float64
        if _is_llama_path(args.model_path) or _is_gemma_path(args.model_path) or _is_vault_gemma_path(args.model_path):
            # Large decoder checkpoints can briefly exceed node RAM during shard loading unless we ask
            # Transformers to stream weights in a low-memory path.
            model_kwargs.setdefault('low_cpu_mem_usage', True)
            model_kwargs.setdefault('use_safetensors', True)
        if _is_llama_path(args.model_path):
            _maybe_add_legacy_llama_rope_config(model_kwargs, args.model_path, args.attn_implementation)
        _ensure_vault_gemma_available(args.model_path)
        if args.task == 'seq_class' and _is_vault_gemma_path(args.model_path):
            causal_lm = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.model = CausalLMSequenceClassifier(causal_lm)
        elif args.task == 'seq_class':
            self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        elif args.task == 'next_token_pred':
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        else:
            assert False
        g_cpu = torch.Generator(device=self.model.device)
        g_cpu.manual_seed(0)
        self.model.eval()
        self.model.config.use_cache = False
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        tokenizer_kwargs = {'use_fast': True, 'cache_dir': args.cache_dir}
        if access_token is not None:
            tokenizer_kwargs['token'] = access_token
        if _is_offline_mode():
            tokenizer_kwargs['local_files_only'] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_model_path, **tokenizer_kwargs)
        self.tokenizer.model_max_length = 512

        if args.pad == 'left':
            self.tokenizer.padding_side = "left"

        if _is_gpt2_path(args.model_path):
            self.start_token = None
            self.eos_token = self.model.config.eos_token_id
            self.layer_ids = list(range(4, 137, 12))

            if args.task == 'seq_class':
                self.model.score.weight.data.normal_(mean=0.0, std=1e-3, generator=g_cpu)

            # Set padding token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.pad_token = self.model.config.eos_token_id
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)

            self.emb_size = self.model.config.n_embd
            add_partial_forward_gpt2(self.model.transformer)

        elif args.model_path in ['bert-base-uncased']:

            self.start_token = 101
            self.eos_token = 102
            self.pad_token = 0
            self.layer_ids = list(range(5, 190, 16))

            # Store embeddings
            bert_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
            bert_embeddings_weight_token = self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)

            self.embeddings_weight_nopos = (bert_embeddings_weight_token + bert_embeddings_weight[0][:, None, :])[
                None, :, :, :]
            self.emb_size = self.model.config.hidden_size
            add_partial_forward_bert(self.model.bert)
        elif _is_llama_path(args.model_path):

            self.start_token = self.tokenizer.bos_token_id
            self.eos_token = self.tokenizer.eos_token_id
            if args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf']:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.unk_token})
                self.pad_token = self.tokenizer.unk_token_id
                self.model.config.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.pad_token = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            if args.train_method == 'lora' and args.finetuned_path is not None:
                lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=['q_proj'])
                self.model = peft.LoraModel(self.model, lora_cfg, 'default')
                self.model.load_state_dict(torch.load(args.finetuned_path, map_location=torch.device('cpu')))
                self.model = self.model.model
                self.layer_ids = list(range(0, 64, 2))
            else:
                if args.task == 'seq_class':
                    self.model.score.weight.data.normal_(mean=0.0, std=1e-3)
                # else:
                # self.model.lm_head.weight.data.normal_(mean=0.0, std=1e-6)

                if args.train_method == 'lora':
                    lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=['q_proj'])
                    self.full_model = peft.LoraModel(self.model, lora_cfg, "default")
                    self.model = self.full_model.model
                    self.layer_ids = list(range(1, 64, 2))

                else:
                    self.layer_ids = list(range(1, 281, 9))

            self.emb_size = self.model.config.hidden_size
            self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
            add_partial_forward_llama(self.model.model)

            if self._force_llama_non_cuda_float32(args.device):
                self._prepare_llama_non_cuda_float32()

        elif _is_gemma_path(args.model_path):

            self.start_token = self.tokenizer.bos_token_id
            self.eos_token = self.tokenizer.eos_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.pad_token = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = self.pad_token

            if args.task == 'seq_class' and hasattr(self.model, 'score'):
                self.model.score.weight.data.normal_(mean=0.0, std=1e-3, generator=g_cpu)

            self.layer_ids = self._find_parameter_indices(['self_attn.q_proj.weight', 'q_proj.weight'])
            if len(self.layer_ids) < args.n_layers:
                raise RuntimeError(
                    f'Found only {len(self.layer_ids)} Gemma q_proj gradient tensors, '
                    f'but --n_layers={args.n_layers}. Check the model architecture/name.'
                )

            self.emb_size = self.model.config.hidden_size
            self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
            add_partial_forward_llama(self.model.model)

        self.trainable_parameters = lambda: (param for param in self.model.parameters() if param.requires_grad)
        config['START_TOKEN'] = self.start_token
        config['EOS_TOKEN'] = self.eos_token
        config['PAD_TOKEN'] = self.pad_token
        self.set_model_device(args.device)

    def _force_llama_non_cuda_float32(self, device=None):
        device = self.args.device if device is None else device
        return (
            _is_llama_path(self.args.model_path)
            and self.args.precision == 'half'
            and not str(device).startswith('cuda')
        )

    def _log_llama_non_cuda_float32(self):
        if self._logged_llama_non_cuda_float32:
            return
        logger.info(
            "Requested --precision half for Llama on a non-CUDA device, but PyTorch does not reliably support "
            "the required half-precision loss/backward path there. Using float32 for Llama gradient computations."
        )
        self._logged_llama_non_cuda_float32 = True

    def _prepare_llama_non_cuda_float32(self):
        self._log_llama_non_cuda_float32()
        self.model.float()
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        if hasattr(self.model, "model") and hasattr(self.model.model.config, "_attn_implementation"):
            self.model.model.config._attn_implementation = "eager"

    def compute_grads_fed_avg(self, batch, labels, create_graph=False):
        og_weights = [param.data.clone() for param in self.model.parameters()]

        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.avg_lr)

        n_minib = batch['input_ids'].shape[0] // self.args.b_mini
        logger.info(n_minib)
        for _ in range(self.args.avg_epochs):
            for i in range(n_minib):
                logger.info(batch['input_ids'].shape)
                b_mini = {k: batch[k][i * self.args.b_mini:(i + 1) * self.args.b_mini] for k in batch.keys()}
                y_mini = labels[:, i * self.args.b_mini:(i + 1) * self.args.b_mini]
                logger.info("%s %s", b_mini['input_ids'].shape, y_mini)
                optimizer.zero_grad()
                outs = self.model(**b_mini, labels=y_mini)
                outs.loss.backward()
                optimizer.step()

        grad = [-(param.data.detach() - og_weights[i]) / n_minib / self.args.avg_lr / self.args.avg_epochs for i, param
                in enumerate(self.model.parameters())]
        for i, param in enumerate(self.model.parameters()):
            param.data = og_weights[i]
        self.model.eval()
        return grad

    def compute_grads(self, batch, y_labels, create_graph=False):
        if self.args.grad_mode == 'eval':
            self.model.eval()
        else:
            self.model.train()
        dev = y_labels.device
        if self.args.precision != '8bit':
            batch = batch.to(self.args.device_grad)
            y_labels = y_labels.to(self.args.device_grad)
            self.model.to(self.args.device_grad)
        if self.args.task == 'next_token_pred':
            labels = torch.where(batch['attention_mask'].bool(), batch['input_ids'], -100)
        elif self.args.task == 'seq_class':
            labels = y_labels
        if self.args.grad_b is None:
            if self.args.algo == 'fedavg':
                grad = self.compute_grads_fed_avg(batch, labels, create_graph)
            elif self.is_lower():
                outputs = self.model(**batch)
                logits = outputs.logits.float()
                loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1), labels.squeeze(0))
                for name, param in self.model.named_parameters():
                    param.requires_grad = True
                    logger.info("%s %s %s", name, param.shape, param.requires_grad)
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph,
                                           allow_unused=True)

            else:

                forward_kwargs = {
                    'labels': labels,
                    'output_hidden_states': (self.args.loss == 'mse'),
                    'output_attentions': False,
                }
                if self.is_decoder():
                    forward_kwargs['use_cache'] = False
                outs = self.model(**batch, **forward_kwargs)

                if self.args.loss == 'mse':
                    loss = outs.hidden_states[-1].pow(2).mean()
                elif self.args.loss == 'ce':
                    loss = outs.loss
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph,
                                           allow_unused=True)

        else:

            minib_size = self.args.batch_size // self.args.grad_b
            for i in range(self.args.grad_b):
                mini_batch = {k: batch[k][i * minib_size:(i + 1) * minib_size] for k in batch.keys()}
                if self.is_lower():
                    outputs = self.model(**mini_batch)
                    logits = outputs.logits.float()
                    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1),
                                                        labels.squeeze(0)[i * minib_size:(i + 1) * minib_size])
                    loss.backward()
                else:
                    outs = self.model(**mini_batch, labels=labels[:, i * minib_size:(i + 1) * minib_size])
                    outs.loss.backward()
            grad = tuple([param.grad.detach().cpu() / self.args.grad_b for param in self.model.parameters()])
        self.set_model_device(dev)
        if self.args.precision != '8bit':
            batch = batch.to(dev)
            y_labels = y_labels.to(dev)
        self.model.eval()
        # torch.save(grad, f'./grad_{self.args.algo}1.pt')
        # raise ValueError
        return grad

    def compute_grads_from_embeds(self, x_embeds, y_labels, attention_mask=None, create_graph=False):
        if self.args.task != 'seq_class':
            raise NotImplementedError('Hybrid continuous optimization currently supports seq_class only.')

        if self.args.grad_mode == 'eval':
            self.model.eval()
        else:
            self.model.train()

        dev = y_labels.device
        y_labels = y_labels.to(x_embeds.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(x_embeds.device)

        self.model.to(x_embeds.device)
        if self._force_llama_non_cuda_float32(x_embeds.device):
            self._prepare_llama_non_cuda_float32()
            x_embeds = x_embeds.float()

        if _is_gpt2_path(self.args.model_path):
            transformer_outputs = self.model.transformer(
                inputs_embeds=x_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            hidden_states = transformer_outputs[0]
            logits = self.model.score(hidden_states).float()
            if attention_mask is None:
                sequence_lengths = torch.full(
                    (x_embeds.shape[0],),
                    x_embeds.shape[1] - 1,
                    dtype=torch.long,
                    device=x_embeds.device,
                )
            else:
                flipped_mask = torch.flip(attention_mask.long(), dims=[1])
                sequence_lengths = attention_mask.shape[1] - 1 - flipped_mask.argmax(dim=1)
            pooled_logits = logits[torch.arange(x_embeds.shape[0], device=x_embeds.device), sequence_lengths]
            loss = F.cross_entropy(pooled_logits, y_labels.view(-1))
        else:
            forward_kwargs = {
                'inputs_embeds': x_embeds,
                'attention_mask': attention_mask,
                'labels': y_labels.view(-1),
                'output_attentions': False,
                'output_hidden_states': False,
            }
            if self.is_decoder():
                forward_kwargs['use_cache'] = False
            outs = self.model(**forward_kwargs)
            loss = outs.loss

        grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph, allow_unused=True)
        self.set_model_device(dev)
        self.model.eval()
        return grad

    def set_model_device(self, device):
        if self.args.precision == '8bit':
            return
        if self.has_rope() and device != 'cpu':
            self.model.model.embed_tokens.to(device)
            if hasattr(self.model.model, 'rotary_emb'):
                self.model.model.rotary_emb.to(device)
            for i in range(min(self.args.n_layers, len(self.model.model.layers))):
                self.model.model.layers[i].to(device)
            if hasattr(self.model, 'score'):
                self.model.score.to(device)
            if hasattr(self.model, 'lm_head'):
                self.model.lm_head.to(device)
        else:
            self.model.to(device)

    def get_matrices_expansions(self, true_grads, B=None, tol=None):
        if B is None:
            max_rank = 0
            for i in self.layer_ids[:10]:
                grad = true_grads[i]
                if _is_gpt2_path(self.args.model_path):
                    grad = grad.T
                grad_np = grad.detach().float().cpu().numpy()
                if self.args.precision == 'half':
                    B = np.linalg.matrix_rank(grad_np, tol=tol)
                else:
                    B = np.linalg.matrix_rank(grad_np, tol=tol)
                if max_rank < B:
                    max_rank = B
            B = max_rank
        if self.args.algo == 'fedavg':
            B += 60
        B = min(B, self.emb_size - self.args.rank_cutoff)

        R_Qs = []

        for i in range(self.args.n_layers):
            grad_Q = true_grads[self.layer_ids[i]]
            if _is_gpt2_path(self.args.model_path):
                grad_Q = grad_Q.T
            _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol, upcast=(self.args.precision == 'half'))
            R_Q = R_Q.to(self.args.device)
            R_Qs.append(R_Q)
        return B, R_Qs

    def get_embeddings(self, pos=None):
        if self.args.model_path in ['bert-base-uncased']:
            bert_embeddings_weight_position = self.model.bert.embeddings.position_embeddings.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + bert_embeddings_weight_position[0][
                pos:pos + 1, None, None, :]
            emb = self.model.bert.embeddings.LayerNorm(emb)
            return emb

        elif _is_gpt2_path(self.args.model_path):
            gpt_embeddings_weight_position = self.model.transformer.wpe.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + gpt_embeddings_weight_position[0][
                pos:pos + 1, None, :]
            emb = self.model.transformer.h[0].ln_1(emb)
            return emb
        elif self.has_rope():
            emb = self.embeddings_weight_nopos.to(self.args.device)
            return self.model.model.layers[0].input_layernorm(emb)

    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1):
        if self.args.model_path in ['bert-base-uncased']:
            # if token_type_ids is None:
            #     raise ValueError('Token type must be defined when model is BERT')
            # emb = self.model.bert.embeddings( input_ids=sentences, token_type_ids=token_type_ids )
            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.bert.encoder.layer[i](emb)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(emb[ : , :-1, : ].clone())
            # return layer_inputs
            return self.model.bert.get_hidden_states(input_ids=sentences, token_type_ids=token_type_ids,
                                                     n_layers=layers)

        elif _is_gpt2_path(self.args.model_path):
            return self.model.transformer.get_hidden_states(input_ids=sentences, attention_mask=attention_mask,
                                                            n_layers=layers)

        elif self.has_rope():
            position_ids = torch.arange(sentences.size(1)).unsqueeze(0).repeat(sentences.size(0), 1).to(
                self.args.device)
            # if attention_mask is not None:
            #     first_item_idx = torch.argmax(attention_mask, dim=1).unsqueeze(1)
            #     position_ids = torch.maximum(position_ids - first_item_idx, torch.tensor(0).to(self.args.device))
            #     attention_mask = update_causal_mask(self.model.model, attention_mask, emb).to(self.args.device)

            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.model.layers[i](emb, attention_mask=attention_mask, position_ids=position_ids)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(self.model.model.layers[i+1].input_layernorm(emb))
            # return layer_inputs
            return self.model.model.get_hidden_states(input_ids=sentences, position_ids=position_ids,
                                                      attention_mask=attention_mask, n_layers=layers)

    def is_bert(self):
        return self.args.model_path in ['bert-base-uncased']

    def is_decoder(self):
        return _is_gpt2_path(self.args.model_path) or _is_llama_path(self.args.model_path) or _is_gemma_path(
            self.args.model_path)

    def has_rope(self):
        return _is_llama_path(self.args.model_path) or _is_gemma_path(self.args.model_path)

    def is_gemma_family(self):
        return _is_gemma_path(self.args.model_path)

    def is_vault_gemma(self):
        return _is_vault_gemma_path(self.args.model_path)

    def effective_l2_span_thresh(self, requested_thresh):
        if self.is_vault_gemma():
            return max(requested_thresh, 5e-3)
        if self.is_gemma_family():
            return max(requested_thresh, 1e-5)
        return requested_thresh

    def has_bos(self):
        return self.start_token is not None

    def is_lower(self):
        if self._force_llama_non_cuda_float32():
            return False
        return self.args.precision in ['8bit', 'half']

    def get_input_embeddings_weight(self):
        return self.model.get_input_embeddings().weight

    def _find_parameter_indices(self, name_suffixes):
        indices = []
        for idx, (name, _) in enumerate(self.model.named_parameters()):
            if any(name.endswith(suffix) or suffix in name for suffix in name_suffixes):
                indices.append(idx)
        return indices
