# train.py

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from preprocess import load_and_tokenize
from model import build_model        # PEFT + BertForSequenceClassification setup
from sklearn.metrics import precision_recall_fscore_support


class FloatLabelTrainer(Trainer):
    """
    A custom Trainer that casts integer labels to float32
    before computing the loss, so BCEWithLogitsLoss won't error.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) Extract labels and cast to float32 if present
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.float()

        # 2) Delegate to the parent class (it will handle loss calculation)
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True
        )

        # 3) Return in the form Trainer expects
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Compute macro-averaged precision, recall and F1
    for multi-label classification.
    """
    logits, labels = eval_pred
    # Apply sigmoid + threshold
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
    # 1) Load & preprocess GoEmotions
    ds = load_and_tokenize(
        ds_name="goemotions",
        tokenizer_name="bert-base-uncased",
        max_length=128,
        save_to_disk=False
    )

    # 2) Build our PEFT-wrapped multi-label classifier
    model = build_model(
        model_name_or_path="bert-base-uncased",
        num_labels=28,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # 3) Prepare TrainingArguments
    args = TrainingArguments(
        output_dir="outputs/single_task",
        eval_strategy="epoch",     # evaluate each epoch
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_dir="logs",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,     # keep our 'labels'
        label_names=["labels"],          # explicitly name the label column
    )

    # 4) Instantiate our custom Trainer
    trainer = FloatLabelTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    # 5) Fine-tune
    trainer.train()

    # 6) Evaluate on test set
    test_res = trainer.evaluate(eval_dataset=ds["test"])
    print("\n=== Final Test Results ===")
    print(f"Macro Precision: {test_res['eval_macro_precision']:.4f}")
    print(f"Macro Recall:    {test_res['eval_macro_recall']:.4f}")
    print(f"Macro F1:        {test_res['eval_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
