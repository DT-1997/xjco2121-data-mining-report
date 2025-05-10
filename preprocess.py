# preprocess.py

import os
import json
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_tokenize(
    ds_name: str = "goemotions",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    save_to_disk: bool = False
) -> DatasetDict:
    """
    Load a dataset from local JSONL under data/{ds_name}/, then tokenize texts
    and prepare multi-hot labels for multi-label classification.
    If local files are missing, fall back to the HF repo for GoEmotions.
    """
    data_dir = os.path.join("data", ds_name)
    os.makedirs(data_dir, exist_ok=True)

    # 1) Gather expected file paths for train/validation/test
    files = {
        split: os.path.join(data_dir, f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    have_local = all(os.path.exists(path) for path in files.values())

    # 2) If local JSONL exist, load & return immediately
    #    (they already have input_ids, attention_mask, labels)
    if have_local:
        ds = load_dataset("json", data_files=files)
        ds.set_format(type="torch")
        return ds

    # 3) Otherwise download simplified GoEmotions from HF
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    # ds now has splits: train, validation, test

    # 4) Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 5) Define batch preprocessing: tokenize + multi-hot encode labels
    def preprocess_batch(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        if "labels" in batch:
            num_labels = 28
            mh = []
            for idxs in batch["labels"]:
                vec = [0] * num_labels
                for i in idxs:
                    vec[i] = 1
                mh.append(vec)
            enc["labels"] = mh
        return enc

    # 6) Apply the above to all splits
    ds = ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc=f"Tokenizing {ds_name}"
    )

    # 7) Convert to torch tensors (so downstream Trainer sees Tensors)
    ds.set_format(type="torch")

    return ds


if __name__ == "__main__":
    ds = load_and_tokenize("goemotions", max_length=128, save_to_disk=False)
    print("Available splits:", ds.keys())
    print("Sample:", ds["train"][0])
