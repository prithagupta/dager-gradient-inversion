import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


class TextDataset:
    def __init__(self, device, dataset, split, n_inputs, batch_size, cache_dir=None, use_hf_split=False):
        if cache_dir is None:
            cache_dir = "/lustre/guptap69/.cache/huggingface/gia_exp_cache"

        seq_keys = {
            'cola': 'sentence',
            'sst2': 'sentence',
            'rte': 'sentence1',
            'rotten_tomatoes': 'text',
            'glnmario/ECHR': 'text',
            'stanfordnlp/imdb': 'text',
            'swj0419/WikiMIA': 'input',
        }
        seq_key = seq_keys[dataset]

        def _safe_name(name: str) -> str:
            return name.replace("/", "__")

        def _load_saved_dataset(save_dir: str):
            if not os.path.isdir(save_dir):
                return None
            try:
                return load_from_disk(save_dir)
            except Exception:
                return None

        def _load_local_snapshot_dataset(snapshot_dir: str):
            parquet_files = []
            json_files = []
            csv_files = []

            for root, _, files in os.walk(snapshot_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if f.endswith(".parquet"):
                        parquet_files.append(path)
                    elif f.endswith(".json") or f.endswith(".jsonl"):
                        json_files.append(path)
                    elif f.endswith(".csv"):
                        csv_files.append(path)

            parquet_files.sort()
            json_files.sort()
            csv_files.sort()

            if parquet_files:
                return load_dataset("parquet", data_files=parquet_files, cache_dir=cache_dir)
            if json_files:
                return load_dataset("json", data_files=json_files, cache_dir=cache_dir)
            if csv_files:
                return load_dataset("csv", data_files=csv_files, cache_dir=cache_dir)

            raise FileNotFoundError(f"No parquet/json/jsonl/csv files found in {snapshot_dir}")

        def _load_dataset_anywhere(name: str):
            # 1) Preferred: already-materialized save_to_disk datasets under cache_dir.
            saved_dataset_map = {
                'cola': os.path.join(cache_dir, 'glue__cola'),
                'sst2': os.path.join(cache_dir, 'glue__sst2'),
                'rte': os.path.join(cache_dir, 'glue__rte'),
                'rotten_tomatoes': os.path.join(cache_dir, 'rotten_tomatoes'),
                'glnmario/ECHR': os.path.join(cache_dir, 'glnmario__ECHR'),
                'stanfordnlp/imdb': os.path.join(cache_dir, 'stanfordnlp__imdb'),
                'swj0419/WikiMIA': os.path.join(cache_dir, 'swj0419__WikiMIA'),
            }

            saved_dir = saved_dataset_map.get(name, os.path.join(cache_dir, _safe_name(name)))
            ds = _load_saved_dataset(saved_dir)
            if ds is not None:
                return ds

            # 2) Regular HF datasets loading
            try:
                if name in ['cola', 'sst2', 'rte']:
                    return load_dataset("glue", name, cache_dir=cache_dir)
                return load_dataset(name, cache_dir=cache_dir)
            except Exception:
                pass

            # 3) Fallback: raw local snapshot folders under cache_dir
            snapshot_map = {
                'rotten_tomatoes': os.path.join(cache_dir, 'rotten_tomatoes_snapshot'),
                'glnmario/ECHR': os.path.join(cache_dir, 'glnmario__ECHR_snapshot'),
                'stanfordnlp/imdb': os.path.join(cache_dir, 'stanfordnlp__imdb_snapshot'),
                'swj0419/WikiMIA': os.path.join(cache_dir, 'swj0419__WikiMIA'),
            }

            if name in snapshot_map and os.path.isdir(snapshot_map[name]):
                return _load_local_snapshot_dataset(snapshot_map[name])

            raise RuntimeError(
                f"Could not load dataset '{name}' from saved dataset directories, Hugging Face cache/Hub, "
                f"or local snapshot files in {cache_dir}"
            )

        ds = _load_dataset_anywhere(dataset)

        using_hf_source_split = use_hf_split and dataset in ['cola', 'sst2', 'rte']

        if dataset in ['cola', 'sst2', 'rte']:
            source_split = 'validation' if using_hf_source_split else 'train'
            full = ds[source_split]
        else:
            if split in ds:
                full = ds[split]
            elif 'train' in ds:
                full = ds['train']
            else:
                available = list(ds.keys())
                raise ValueError(f"Split '{split}' not found for dataset '{dataset}'. Available splits: {available}")

        idxs = list(range(len(full)))
        np.random.shuffle(idxs)
        # if dataset == 'cola':
        #    import pdb; pdb.set_trace()
        #    assert idxs[0] == 2310 # with seed 101

        n_samples = n_inputs * batch_size

        def _raise_insufficient_samples(selected_split_name: str):
            max_inputs = len(full) // batch_size
            raise ValueError(
                f"Requested n_inputs={n_inputs} with batch_size={batch_size} "
                f"({n_samples} total samples), but split '{selected_split_name}' for dataset '{dataset}' "
                f"contains only {len(full)} samples. "
                f"Use a smaller --n_inputs (max {max_inputs} for this batch size), "
                f"a smaller --batch_size, or disable --use_hf_split if you want the larger "
                f"train-derived DAGER validation protocol for GLUE datasets."
            )

        if using_hf_source_split:
            if n_samples > len(full):
                _raise_insufficient_samples(source_split)
            idxs = idxs[:n_samples]
        elif split == 'test':
            if n_samples > 1000:
                raise ValueError(
                    f"Requested n_inputs={n_inputs} with batch_size={batch_size} "
                    f"({n_samples} total samples), but DAGER split='test' reserves only 1000 samples. "
                    f"Use max n_inputs={1000 // batch_size} for this batch size."
                )
            idxs = idxs[:n_samples]
        elif split == 'val':
            idxs = idxs[1000:]  # first 1000 saved for testing

            final_idxs = []
            while len(final_idxs) < n_samples:
                zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
                zipped = sorted(zipped, key=lambda x: x[1])
                chunk_sz = max(len(zipped) // n_samples, 1)

                l = min(len(zipped), n_samples - len(final_idxs))
                for i in range(l):
                    tmp = chunk_sz * i + np.random.randint(0, chunk_sz)
                    final_idxs.append(zipped[tmp][0])
                np.random.shuffle(idxs)

            np.random.shuffle(final_idxs)
            idxs = final_idxs

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
