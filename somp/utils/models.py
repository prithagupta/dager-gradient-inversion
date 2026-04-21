import os
import torch
import peft
import numpy as np
from utils.ext import update_causal_mask
from utils.partial_models import add_partial_forward_gpt2, add_partial_forward_bert, add_partial_forward_llama
from constants import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from utils.functional import get_layer_decomp
from transformers import GPT2Config, GPT2ForSequenceClassification
from transformers import GPT2LMHeadModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer, MT5Tokenizer


def _build_mgpt_gpt2_config(repo_id, cache_dir=None):
    import json
    from transformers import GPT2Config
    raw = {}
    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir)
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        pass

    vocab_size = raw.get("vocab_size", None)
    n_embd     = raw.get("n_embd", None)
    n_layer    = raw.get("n_layer", None)
    n_head     = raw.get("n_head", None)
    n_positions= raw.get("n_positions", 1024)

    state = None
    try:
        st_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=cache_dir)
        state = safe_load(st_path, device="cpu")
    except Exception:
        pass
    if state is None:
        try:
            import torch
            pt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
            state = torch.load(pt_path, map_location="cpu")
        except Exception:
            pass

    if state is not None:
        if vocab_size is None or n_embd is None:
            wte = state.get("transformer.wte.weight", None)
            if wte is not None:
                vocab_size, n_embd = wte.shape  # e.g. 100000, 2048
        wpe = state.get("transformer.wpe.weight", None)
        if wpe is not None:
            n_positions = wpe.shape[0]
        if n_layer is None:
            n = 0
            for k in state.keys():
                if k.startswith("transformer.h.") and k.split(".")[2].isdigit():
                    n = max(n, int(k.split(".")[2]) + 1)
            n_layer = n if n > 0 else None
        if n_head is None and n_embd is not None:
            for h in (8, 12, 16, 24, 32, 64):
                if n_embd % h == 0:
                    n_head = h
                    break

    cfg = GPT2Config(
        vocab_size=vocab_size or 50257,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd or 768,
        n_layer=n_layer or 12,
        n_head=n_head or 12,
        activation_function=raw.get("activation_function", "gelu_new"),
        layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
        bos_token_id=raw.get("bos_token_id", None),
        eos_token_id=raw.get("eos_token_id", None),
    )
    return cfg


class ModelWrapper():
    def __init__(self, args):
        assert (args.model_path in ['THUMT/mGPT', 'ai-forever/mGPT', 'bert-base-uncased', 'gpt2', 'openai-community/gpt2-large', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']),\
            'Model is not yet supported - add it to assertion list and specify implementation details'
        access_token = os.environ['HF_TOKEN']
        self.args = args
        self.device = args.device
        model_kwargs = {'cache_dir': args.cache_dir} if args.cache_dir is not None else {}

        model_kwargs['pretrained_model_name_or_path'] = args.model_path if args.finetuned_path is None or args.train_method == 'lora' else args.finetuned_path
        model_kwargs['attn_implementation'] = args.attn_implementation

        if args.hidden_act is not None and args.model_path in ['gpt2', 'openai-community/gpt2-large', 'ai-forever/mGPT', 'THUMT/mGPT']:
            model_kwargs['activation_function'] = args.hidden_act
        elif args.hidden_act is not None and args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            model_kwargs['hidden_act'] = args.hidden_act
            
        if args.precision == '8bit':
            model_kwargs['load_in_8bit'] = True
        if args.precision == 'half':
            model_kwargs['torch_dtype'] = torch.float16
        if args.precision == 'double':
            model_kwargs['torch_dtype'] = torch.float64
        if args.task == 'seq_class':
            if args.model_path in ['ai-forever/mGPT', 'THUMT/mGPT']:
                cfg = _build_mgpt_gpt2_config(args.model_path, args.cache_dir)
                print("mGPT cfg:", cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.n_head)
                base_lm = GPT2LMHeadModel.from_pretrained(
                    args.model_path,
                    config=cfg,
                    cache_dir=args.cache_dir,
                    torch_dtype=model_kwargs.get('torch_dtype', None)
                )
                wte_shape = tuple(base_lm.transformer.wte.weight.shape)
                assert wte_shape == (cfg.vocab_size,
                                     cfg.n_embd), f"WTE {wte_shape} != cfg {(cfg.vocab_size, cfg.n_embd)}"

                cfg.num_labels = getattr(args, "num_labels", 2)
                self.model = GPT2ForSequenceClassification(cfg)
                self.model.transformer.load_state_dict(base_lm.transformer.state_dict(), strict=False)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)


        elif args.task == 'next_token_pred':
            if args.model_path in ['ai-forever/mGPT', 'THUMT/mGPT']:
                cfg = _build_mgpt_gpt2_config(args.model_path, args.cache_dir)
                self.model = GPT2LMHeadModel.from_pretrained(
                    args.model_path,
                    config=cfg,
                    cache_dir=args.cache_dir,
                    torch_dtype=model_kwargs.get('torch_dtype', None)
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        else:
            assert False

        g_cpu = torch.Generator(device=self.model.device)
        g_cpu.manual_seed(0)
        self.model.eval()
        from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
        def _load_mgpt_tokenizer(repo_id, cache_dir=None):
            try:
                tok_json = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=cache_dir)
                tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
                return tok
            except Exception:
                pass
            try:
                vocab = hf_hub_download(repo_id=repo_id, filename="vocab.json", cache_dir=cache_dir)
                merges = hf_hub_download(repo_id=repo_id, filename="merges.txt", cache_dir=cache_dir)
                tok = GPT2Tokenizer(vocab_file=vocab, merges_file=merges)
                return tok
            except Exception as e:
                raise RuntimeError(
                    f"error: {repr(e)}"
                )

        if args.model_path == 'ai-forever/mGPT':
            self.tokenizer = _load_mgpt_tokenizer(args.model_path, args.cache_dir)
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.tokenizer.padding_side = "left"

        elif args.model_path == 'THUMT/mGPT':
            self.tokenizer = MT5Tokenizer.from_pretrained(
                args.model_path,
                token=access_token,
                cache_dir=args.cache_dir
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                use_fast=True,
                token=access_token,
                cache_dir=args.cache_dir
            )

        self.tokenizer.model_max_length = 512
        
        if args.pad == 'left':
            self.tokenizer.padding_side = "left"

        if args.model_path in ['gpt2', 'openai-community/gpt2-large', 'ai-forever/mGPT']:
            self.start_token = None
            self.eos_token = self.model.config.eos_token_id
            self.layer_ids = list(range(4, 137, 12))
            
            if args.task == 'seq_class':
                self.model.score.weight.data.normal_( mean=0.0, std=1e-3, generator=g_cpu )
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.pad_token = self.model.config.eos_token_id
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

            self.model.resize_token_embeddings(len(self.tokenizer))

            self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)
            
            self.emb_size = self.model.config.n_embd
            add_partial_forward_gpt2(self.model.transformer)
        elif args.model_path == 'THUMT/mGPT':
            self.start_token = None
            self.eos_token = self.tokenizer.eos_token_id
            self.pad_token = self.tokenizer.pad_token_id
            self.layer_ids = list(range(4, 137, 12))

            if args.task == 'seq_class':
                self.model.score.weight.data.normal_(mean=0.0, std=1e-3, generator=g_cpu)
            self.model.config.pad_token_id = self.pad_token

            self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)
            self.emb_size = self.model.config.n_embd
            add_partial_forward_gpt2(self.model.transformer)

        elif args.model_path in ['bert-base-uncased']:
          
            self.start_token = 101
            self.eos_token = 102
            self.pad_token = 0
            self.layer_ids = list(range(5,190,16))
            
            # Store embeddings
            bert_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
            bert_embeddings_weight_token = self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)
            
            self.embeddings_weight_nopos = (bert_embeddings_weight_token + bert_embeddings_weight[0][:,None,:])[None,:,:,:]
            self.emb_size = self.model.config.hidden_size
            add_partial_forward_bert(self.model.bert)
        elif args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            
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
                lora_cfg = peft.LoraConfig(r=args.lora_r,target_modules=['q_proj'])
                self.model = peft.LoraModel(self.model, lora_cfg, 'default')
                self.model.load_state_dict(torch.load(args.finetuned_path, map_location=torch.device('cpu')))
                self.model = self.model.model
                self.layer_ids = list(range(0,64,2))
            else:
                if args.task == 'seq_class':
                    self.model.score.weight.data.normal_(mean=0.0, std=1e-3)
                    
                if args.train_method == 'lora':
                    lora_cfg = peft.LoraConfig(r=args.lora_r,target_modules=['q_proj'])
                    self.full_model = peft.LoraModel(self.model, lora_cfg, "default")
                    self.model = self.full_model.model
                    self.layer_ids = list(range(1,64,2))
                    
                else:
                    self.layer_ids = list(range(1,281,9))
            
            self.emb_size = self.model.config.hidden_size
            self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
            add_partial_forward_llama(self.model.model)
        
        self.trainable_parameters = lambda: (param for param in self.model.parameters() if param.requires_grad)
        config['START_TOKEN'] = self.start_token
        config['EOS_TOKEN'] = self.eos_token
        config['PAD_TOKEN'] = self.pad_token
        self.set_model_device(args.device)
        self._WTE_CPU = self.get_word_embeddings().detach().to('cpu', copy=True).float()

    def compute_grads_fed_avg(self, batch, labels, create_graph=False):
        og_weights = [param.data.clone() for param in self.model.parameters()]

        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.avg_lr)

        n_minib = batch['input_ids'].shape[0] // self.args.b_mini
        print(n_minib)
        for _ in range(self.args.avg_epochs):
            for i in range(n_minib):
                print(batch['input_ids'].shape)
                b_mini = {k:batch[k][i*self.args.b_mini:(i+1)*self.args.b_mini] for k in batch.keys()}
                y_mini = labels[:, i*self.args.b_mini:(i+1)*self.args.b_mini]
                print(b_mini['input_ids'].shape, y_mini)
                optimizer.zero_grad()
                outs = self.model(**b_mini, labels=y_mini)
                outs.loss.backward()
                optimizer.step()
           
        grad = [-(param.data.detach() - og_weights[i])/n_minib/self.args.avg_lr/self.args.avg_epochs for i, param in enumerate(self.model.parameters())]
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
                grad=self.compute_grads_fed_avg(batch, labels, create_graph)
            elif self.is_lower():
                outputs = self.model(**batch)
                logits = outputs.logits.float()
                loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1), labels.squeeze(0))
                for name, param in self.model.named_parameters():
                    param.requires_grad=True
                    print(name, param.shape, param.requires_grad)
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph,
                                                retain_graph=False, allow_unused=True)
                
            else:
                
                outs = self.model(**batch, labels=labels, output_hidden_states=True)
                    
                if self.args.loss == 'mse':
                    loss = outs.hidden_states[-1].pow(2).mean()
                elif self.args.loss == 'ce':
                    loss = outs.loss
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph,retain_graph=False, allow_unused=True)

        else:
            
            minib_size = self.args.batch_size // self.args.grad_b
            for i in range(self.args.grad_b):
                mini_batch = {k: batch[k][i*minib_size:(i+1)*minib_size] for k in batch.keys()}
                if self.is_lower():
                    outputs = self.model(**mini_batch)
                    logits = outputs.logits.float()
                    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1), labels.squeeze(0)[i*minib_size:(i+1)*minib_size])
                    loss.backward()
                else:
                    outs = self.model(**mini_batch, labels=labels[:,i*minib_size:(i+1)*minib_size])
                    outs.loss.backward()
            grad = tuple([param.grad.detach().cpu()/self.args.grad_b for param in self.model.parameters()])
        self.set_model_device(dev)
        if self.args.precision != '8bit':
            batch = batch.to(dev)
            y_labels = y_labels.to(dev)
        self.model.eval()
        return grad

    def set_model_device(self, device):
        if self.args.precision == '8bit':
            return
        self.model.to(device)

    def get_matrices_expansions(self, true_grads, B=None, tol=None):
        if tol is None:
            tol = self.args.rank_tol

        if B is None:
            max_rank = 0
            num_layers_to_check = min(10, len(self.layer_ids))
            for i in self.layer_ids[:num_layers_to_check]:
                grad = true_grads[i]
                if self.args.model_path in ['gpt2', 'openai-community/gpt2-large', 'gpt2-large','ai-forever/mGPT', 'THUMT/mGPT']:
                    grad = grad.T
                if self.args.precision == 'half':
                    rank = np.linalg.matrix_rank(grad.float().cpu(), tol=tol)
                else:
                    rank = np.linalg.matrix_rank(grad.cpu(), tol=tol)
                if max_rank < rank:
                    max_rank = rank
            B = max_rank

        if self.args.algo == 'fedavg':
            B += 60

        if hasattr(self, 'emb_size'):
            B = min(B, self.emb_size - self.args.rank_cutoff)
            B = min(B, self.model.config.hidden_size - self.args.rank_cutoff)
        R_Qs_original = []
        for i in range(self.args.n_layers):
            grad_Q = true_grads[self.layer_ids[i]]
            if self.args.model_path in ['gpt2', 'openai-community/gpt2-large', 'gpt2-large', 'ai-forever/mGPT', 'THUMT/mGPT']:
                grad_Q = grad_Q.T
            _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol, upcast=(self.args.precision == 'half'))
            R_Q = R_Q.to(self.args.device)
            R_Qs_original.append(R_Q)
        head_R_Qs_split = None
        if self.args.headwise_factorization:
            h = self.model.config.num_attention_heads

            grad_l1 = true_grads[self.layer_ids[0]]
            if self.args.model_path in ['gpt2', 'openai-community/gpt2-large', 'gpt2-large', 'ai-forever/mGPT', 'THUMT/mGPT']:
                grad_l1 = grad_l1.T
            d_model = self.model.config.hidden_size
            grad_l1_Q = grad_l1[:, :d_model]

            head_R_Qs_split = []
            h = self.model.config.num_attention_heads
            d_head = d_model // h
            for i in range(h):
                start, end = i * d_head, (i + 1) * d_head
                head_grad_slice = grad_l1_Q[:, start:end]
                _, R_Q_i = get_layer_decomp(head_grad_slice, B=B, tol=tol, upcast=(self.args.precision == 'half'))
                head_R_Qs_split.append(R_Q_i.to(self.args.device))

        return B, R_Qs_original, head_R_Qs_split

    def get_word_embeddings(self):
        return self.model.get_input_embeddings().weight

    def apply_positional_encoding(self, embeddings, position):
        position_ids = torch.tensor([position], device=self.device)
        position_embeddings = self.model.transformer.wpe(position_ids)

        return embeddings + position_embeddings

    def get_embeddings(self, pos=None):
        device = self.args.device

        if self.args.model_path in ['bert-base-uncased']:
            bert_pos = (
                self.model.bert.embeddings.position_embeddings.weight
                .to(device)
                .unsqueeze(0)
            )

            emb = (
                    self.embeddings_weight_nopos.to(device)
                    + bert_pos[0][pos:pos + 1, None, None, :]
            )

            ln = self.model.bert.embeddings.LayerNorm.to(device)
            emb = ln(emb)
            return emb

        elif self.args.model_path in [
            'gpt2', 'openai-community/gpt2-large',
            'ai-forever/mGPT', 'THUMT/mGPT'
        ]:
            gpt_pos = (
                self.model.transformer.wpe.weight
                .to(device)
                .unsqueeze(0)
            )

            emb = (
                    self.embeddings_weight_nopos.to(device)
                    + gpt_pos[0][pos:pos + 1, None, :]
            )

            ln = self.model.transformer.h[0].ln_1.to(device)
            emb = ln(emb)
            return emb

        elif self.args.model_path in [
            'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3-70B'
        ]:
            emb = self.embeddings_weight_nopos.to(device)
            ln = self.model.model.layers[0].input_layernorm.to(device)
            return ln(emb)

    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1):
        if self.args.model_path in ['bert-base-uncased']:
            return self.model.bert.get_hidden_states(input_ids=sentences, token_type_ids=token_type_ids, n_layers=layers)
        
        elif self.args.model_path in ['gpt2', 'openai-community/gpt2-large','ai-forever/mGPT', 'THUMT/mGPT']:
            return self.model.transformer.get_hidden_states(input_ids=sentences, attention_mask=attention_mask, n_layers=layers)
        
        elif self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            position_ids = torch.arange(sentences.size(1)).unsqueeze(0).repeat(sentences.size(0), 1).to(self.args.device)
            return self.model.model.get_hidden_states(input_ids=sentences, position_ids=position_ids,attention_mask=attention_mask, n_layers=layers)
        
    def is_bert(self):
        return self.args.model_path in ['bert-base-uncased']
    
    def is_decoder(self):
        return self.args.model_path in ['ai-forever/mGPT', 'THUMT/mGPT', 'gpt2', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf','openai-community/gpt2-large', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']
        
    def has_rope(self):
        return self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']

    def has_bos(self):
        return self.start_token is not None
    def is_lower(self):
        return self.args.precision in ['8bit', 'half']

