# preprocess.py

import os
import json
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_tokenize(
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    save_to_disk: bool = False,
    dataset_version: str = "goemotions"
) -> DatasetDict:
    """
    Load a dataset from local JSONL files under data/{dataset_version}/,
    tokenize the text, and prepare multi-hot labels for multi-label classification.
    If local files are missing, fall back to the Hugging Face Hub.
    Optionally save the tokenized dataset back to disk as JSONL files.

    Args:
        tokenizer_name (str): Pretrained tokenizer name or path.
        max_length (int): Maximum token sequence length.
        save_to_disk (bool): Whether to save tokenized splits to disk.
        dataset_version (str): "goemotions" or "goemotions-augmented".

    Returns:
        DatasetDict: A Hugging Face DatasetDict with train/validation/test splits.
    """
    # Build the directory name by replacing '-' with '_'
    dir_name = dataset_version.replace("-", "_")
    data_dir = os.path.join("data", dir_name)
    os.makedirs(data_dir, exist_ok=True)

    # Define expected file paths for train/validation/test
    files = {
        split: os.path.join(data_dir, f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    have_local = all(os.path.exists(path) for path in files.values())

    # If all local JSONL files exist, load them immediately
    if have_local:
        dataset = load_dataset("json", data_files=files)
        dataset.set_format(type="torch")
        return dataset

    # Load dataset from HF hub based on version
    if dataset_version == "goemotions":
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    elif dataset_version == "goemotions-augmented":
        dataset = load_dataset("jellyshroom/go_emotions_augmented")
    else:
        raise ValueError("Unsupported dataset_version. Choose 'goemotions' or 'goemotions-augmented'.")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define preprocessing function for each batch
    def preprocess_batch(batch):
        # Tokenize the text samples
        encoding = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # If labels exist, convert them to multi-hot vectors
        if "labels" in batch:
            num_labels = 28
            multi_hot = []
            for label_indices in batch["labels"]:
                vector = [0] * num_labels
                for idx in label_indices:
                    vector[idx] = 1
                multi_hot.append(vector)
            encoding["labels"] = multi_hot
        return encoding

    # Apply the preprocessing to all splits, removing original columns
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc=f"Tokenizing {dataset_version}"
    )

    # Set the format to PyTorch tensors
    dataset.set_format(type="torch")

    # Optionally save tokenized data back to disk as JSONL
    if save_to_disk:
        for split in ("train", "validation", "test"):
            with open(files[split], "w", encoding="utf-8") as f:
                for example in dataset[split]:
                    # Convert tensor or numpy arrays to lists for JSON serialization
                    serializable = {
                        key: value.tolist() if hasattr(value, "tolist") else value
                        for key, value in example.items()
                    }
                    json.dump(serializable, f)
                    f.write("\n")

    return dataset


if __name__ == "__main__":
    ds = load_and_tokenize(
        tokenizer_name="bert-base-uncased",
        max_length=128,
        save_to_disk=True,
        dataset_version="goemotions"
    )
    print("Available splits:", ds.keys())
    print("Sample example from train split:", ds["train"][0])
