# train.py

import argparse
import random
import torch
import numpy as np
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, AutoTokenizer
from preprocess import load_and_tokenize
from model import build_model
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed: int = 42):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomTrainer(Trainer):
    """
    Custom Trainer that casts labels to float32 before loss computation
    and logs metrics per epoch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.to(torch.float32)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        return (loss, outputs) if return_outputs else loss

    def log_metrics(self, metrics: dict):
        print(f"Logging metrics: {metrics}")
        self.epoch_metrics.append(metrics)


def compute_metrics(eval_pred):
    """
    Compute precision, recall, and F1 for multi-label classification.
    """
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    # Parse the dataset version argument
    parser = argparse.ArgumentParser(
        description="Train BERT multi-label classifier on GoEmotions variants"
    )
    parser.add_argument(
        "dataset_arg",
        choices=["goemotions", "goemotions_augmented"],
        help="Which dataset to train on: 'goemotions' or 'goemotions_augmented'",
    )
    args = parser.parse_args()

    # Map to actual dataset_version for preprocess
    if args.dataset_arg == "goemotions":
        dataset_version = "goemotions"
        output_tag = "goemotions"
    else:
        dataset_version = "goemotions-augmented"
        output_tag = "goemotions_augmented"

    # Reproducibility
    set_seed(42)

    # 1) Load and preprocess dataset
    ds: DatasetDict = load_and_tokenize(
        tokenizer_name="bert-base-uncased",
        max_length=128,
        save_to_disk=False,
        dataset_version=dataset_version,
    )

    # 2) Build model
    model = build_model(
        model_name_or_path="bert-base-uncased",
        num_labels=28,
    )

    # 3) Set training arguments
    output_dir = f"outputs/full_ft_{output_tag}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=10,
        logging_dir=f"logs/{output_tag}",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # 4) Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        compute_metrics=compute_metrics,
    )

    # 5) Train
    trainer.train()

    # 6) Test evaluation
    test_metrics = trainer.evaluate(eval_dataset=ds.get("test"))
    print("\n=== Final Test Results ===")
    print(f"Macro Precision: {test_metrics['eval_macro_precision']:.4f}")
    print(f"Macro Recall:    {test_metrics['eval_macro_recall']:.4f}")
    print(f"Macro F1:        {test_metrics['eval_macro_f1']:.4f}")

    # 7) Save final model + tokenizer
    save_path = f"{output_dir}/final_model"
    model.save_pretrained(save_path)
    AutoTokenizer.from_pretrained("bert-base-uncased").save_pretrained(save_path)


if __name__ == "__main__":
    main()
