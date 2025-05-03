# src/preprocess.py

import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import numpy as np

def load_and_tokenize(
    ds_name: str,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128
):
    """
    Load a dataset from HF or local JSONL/CSV,
    tokenize texts and prepare labels/intensity vectors.
    """
    # 1. load raw dataset
    if ds_name == "goemotions":
        raw = load_dataset("mrm8488/goemotions")
    else:
        # assume local JSONL with fields "text","labels","intensity"
        raw = load_dataset("json", data_files={ 
            'train': f"data/{ds_name}/train.jsonl",
            'validation': f"data/{ds_name}/valid.jsonl",
            'test': f"data/{ds_name}/test.jsonl"
        })
    # 2. init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 3. preprocessing function
    def preprocess_batch(batch):
        enc = tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        # convert list-of-indices to multi-hot
        if "labels" in batch:
            # e.g. labels = [[1,4], [0], ...]
            ml = []
            for lbls in batch["labels"]:
                v = np.zeros(11 if ds_name!="goemotions" else 28, dtype=int)
                for i in lbls:
                    v[i] = 1
                ml.append(v)
            enc["labels"] = ml
        # pass through intensity if exists
        if "intensity" in batch:
            enc["intensity"] = batch["intensity"]
        return enc

    # 4. map and return
    ds = raw.map(preprocess_batch, batched=True, remove_columns=raw["train"].column_names)
    ds.set_format(type="torch")
    return ds

if __name__ == "__main__":
    # example usage
    ds_goe = load_and_tokenize("goemotions")
    print(ds_goe["train"][0])
