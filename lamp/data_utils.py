import numpy as np
import torch
from datasets import load_dataset


class TextDataset:
    def __init__(self, device, dataset, split, n_inputs, batch_size, cache_dir=None, use_hf_split=False):

        seq_keys = {
            'cola': 'sentence',
            'sst2': 'sentence',
            'rotten_tomatoes': 'text',
            'glnmario/ECHR': 'text'
        }
        seq_key = seq_keys[dataset]

        using_hf_source_split = use_hf_split and dataset in ['cola', 'sst2']

        if dataset in ['cola', 'sst2']:
            source_split = 'validation' if using_hf_source_split else 'train'
            full = load_dataset('glue', dataset, cache_dir=cache_dir)[source_split]
        elif dataset == 'glnmario/ECHR':
            full = load_dataset(
                'csv',
                data_files=['models_cache/datasets--glnmario--ECHR/ECHR_Dataset.csv'],
                cache_dir=cache_dir,
            )['train']
        else:
            full = load_dataset(dataset, cache_dir=cache_dir)['train']

        idxs = list(range(len(full)))
        np.random.shuffle(idxs)
        if dataset == 'cola' and not use_hf_split:
            assert idxs[0] == 2310  # with seed 101

        n_samples = n_inputs * batch_size

        if using_hf_source_split:
            assert n_samples <= len(full)
            idxs = idxs[:n_samples]
        elif split == 'test':
            assert n_samples <= 1000
            idxs = idxs[:n_samples]
        elif split == 'val':
            idxs = idxs[1000:]  # first 1000 saved for testing
            assert len(idxs) >= n_samples

            zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
            zipped = sorted(zipped, key=lambda x: x[1])
            chunk_sz = len(zipped) // n_samples
            idxs = []
            for i in range(n_samples):
                tmp = chunk_sz * i + np.random.randint(0, chunk_sz)
                idxs.append(zipped[tmp][0])
            np.random.shuffle(idxs)

        # Slice
        self.seqs = []
        self.labels = []
        for i in range(n_inputs):
            seqs = []
            for j in range(batch_size):
                seqs.append(full[idxs[i * batch_size + j]][seq_key])
            if dataset != 'glnmario/ECHR':
                labels = torch.tensor([full[idxs[i * batch_size: (i + 1) * batch_size]]['label']], device=device)
            else:
                labels = torch.tensor([full[idxs[i * batch_size: (i + 1) * batch_size]]['binary_judgement']],
                                      device=device)
            self.seqs.append(seqs)
            self.labels.append(labels)
        assert len(self.seqs) == n_inputs
        assert len(self.labels) == n_inputs

    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])
