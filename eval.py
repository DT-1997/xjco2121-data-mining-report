# src/evaluate.py

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from src.model import MultiTaskBert
from sklearn.metrics import f1_score, mean_squared_error, r2_score

def main():
    # load model & tokenizer
    model = MultiTaskBert.from_pretrained("outputs/run1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # load test data (preprocessed)
    ds = load_dataset("json", data_files="data/semeval2018/test.jsonl")["train"]

    # prepare and run
    all_logits, all_ints, all_labels, all_golds = [], [], [], []
    for example in ds:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        all_logits.append(out["logits"].cpu())
        all_ints.append(out["intensity"].cpu())
        all_labels.append(torch.tensor(example["labels"]))
        all_golds.append(torch.tensor(example["intensity"]))

    # compute metrics...
    # print results

if __name__ == "__main__":
    main()
