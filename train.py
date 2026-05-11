import argparse
import logging
import numpy as np
import os
import peft
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

from datasets import load_dataset, load_metric

device = 'cuda'
logger = logging.getLogger(__name__)


def _default_cache_dir():
    if os.environ.get('DAGER_CACHE_DIR'):
        return os.environ['DAGER_CACHE_DIR']
    if os.environ.get('HF_HOME'):
        return os.path.join(os.environ['HF_HOME'], 'gia_cache')
    return './models_cache'


def _safe_model_name(model_path):
    return model_path.strip('/').replace('/', '__')


def _checkpoint_path(output_dir, dataset, model_path, train_method, n_steps):
    model_name = _safe_model_name(model_path)
    if train_method == 'lora':
        return os.path.join(output_dir, f'{dataset}_{model_name}_{train_method}_steps{n_steps}.pt')
    return os.path.join(output_dir, f'{dataset}_{model_name}_{train_method}_steps{n_steps}')


def save_model(model, tokenizer, save_path, train_method):
    logger.info('SAVING')

    parent = os.path.dirname(save_path) if train_method == 'lora' else save_path
    os.makedirs(parent, exist_ok=True)

    # Save the model
    try:
        if train_method != 'lora':
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), save_path)
        logger.info("Model saved successfully.")
        if train_method != 'lora':
            logger.info(os.listdir(save_path))
        else:
            logger.info(save_path)
    except Exception as e:
        logger.info(f"Error saving model: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--pct_mask', type=float, default=None)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)
    parser.add_argument('--models_cache', type=str, default=None,
                        help='Deprecated alias for --cache_dir; kept for old scripts.')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Clip gradient norm before optional training noise. Use with --noise for clipped/noisy training.')
    parser.add_argument('--rng_seed', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rng_seed)
    args.cache_dir = args.cache_dir or args.models_cache or _default_cache_dir()
    args.output_dir = args.output_dir or os.path.join(args.cache_dir, 'finetuned')
    run_device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if run_device == 'auto':
        run_device = 'cpu'

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels,
                                                               cache_dir=args.cache_dir).to(run_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, cache_dir=args.cache_dir)

    # Configure tokenizer and model
    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Load LoRA model if applicable
    if args.train_method == 'lora':
        lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=['q_proj'])
        model = peft.LoraModel(model, lora_cfg, "default")

    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        metric = load_metric('matthews_correlation')
        train_metric = load_metric('matthews_correlation')
    else:
        metric = load_metric('./train_utils/accuracy.py')
        train_metric = load_metric('./train_utils/accuracy.py')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    elif args.dataset == 'rte':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence1', 'sentence2'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    n_steps = 0
    train_loss = 0

    # Run training loop
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(run_device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])

            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if args.pct_mask is not None:
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    grad_mask = (torch.rand(param.grad.shape).to(run_device) > args.pct_mask).float()
                    param.grad.data = param.grad.data * grad_mask

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.noise is not None:
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    param.grad.data = param.grad.data + torch.randn(param.grad.shape).to(run_device) * args.noise

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            progress_bar.update(1)

            n_steps += 1
            if n_steps % args.save_every == 0:
                save_model(
                    model,
                    tokenizer,
                    _checkpoint_path(args.output_dir, args.dataset, args.model_path, args.train_method, n_steps),
                    args.train_method,
                )
                logger.info("metric train: %s", train_metric.compute())
                logger.info("loss train: %s", train_loss / n_steps)
                train_loss = 0.0

        model.eval()

        for batch in eval_loader:
            batch = {k: v.to(run_device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'{args.dataset}_metric.txt'), 'w') as fou:
            eval_metric = f'metric eval: {metric.compute()}'
            fou.write(eval_metric + '\n')
            logger.info(eval_metric)
    logger.info('END')
    final_path = _checkpoint_path(args.output_dir, args.dataset, args.model_path, args.train_method, n_steps)
    save_model(model, tokenizer, final_path, args.train_method)
    logger.info("Final finetuned model path: %s", final_path)


if __name__ == '__main__':
    main()
