# preprocess.py

import os
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_tokenize(
    ds_name: str,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    save_to_disk: bool = False
):
    """
    Prefer loading local pre-tokenized JSONL splits from data/{ds_name}/;
    if they are missing, download from HF and split + tokenize.
    Returns a DatasetDict with 'train','validation','test'.
    """
    data_dir = os.path.join("data", ds_name)
    local_paths = {
        split: os.path.join(data_dir, f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    has_local = all(os.path.exists(path) for path in local_paths.values())

    if ds_name == "goemotions" and not has_local:
        # Download & split
        raw = load_dataset("mrm8488/goemotions")
        if list(raw.keys()) == ["train"]:
            full = raw["train"]
            tmp  = full.train_test_split(test_size=0.2, seed=42)
            vt   = tmp["test"].train_test_split(test_size=0.5, seed=42)
            raw = DatasetDict({
                "train":      tmp["train"],
                "validation": vt["train"],
                "test":       vt["test"]
            })
        # Tokenization + multi-hot happens below
    else:
        # Local JSONL already contains input_ids, attention_mask, labels, intensity
        raw = load_dataset("json", data_files=local_paths)

        # Set formats and return immediately
        raw.set_format(type="torch")
        return raw

    # At this point, raw has splits of plain text + label indices + intensity
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_batch(batch):
        # tokenize text
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # multi-hot encode 'labels'
        if "labels" in batch:
            num_labels = 28 if ds_name == "goemotions" else len(batch["labels"][0])
            multi_hot = []
            for lbls in batch["labels"]:
                vec = np.zeros(num_labels, dtype=int)
                for idx in lbls:
                    vec[idx] = 1
                multi_hot.append(vec.tolist())
            enc["labels"] = multi_hot
        # pass through 'intensity' if present
        if "intensity" in batch:
            enc["intensity"] = batch["intensity"]
        return enc

    # apply preprocessing to each split
    ds = raw.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc=f"Tokenizing {ds_name}"
    )

    ds.set_format(type="torch")
    return ds


if __name__ == "__main__":
    ds = load_and_tokenize("goemotions")
    print(ds.keys())         # should show train, validation, test
    print(ds["train"][0])    # inspect first example
