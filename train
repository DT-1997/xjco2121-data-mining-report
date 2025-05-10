import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from preprocess import load_and_tokenize
from model import build_model
from sklearn.metrics import precision_recall_fscore_support


class CustomTrainer(Trainer):
    """
    Custom Trainer class that ensures labels are cast to float32
    before computing the loss, as BCEWithLogitsLoss requires float type labels.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure the labels are float32
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.to(torch.float32)

        # Call the original Trainer's compute_loss method
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Compute macro-averaged precision, recall, and F1 for multi-label classification.
    """
    logits, labels = eval_pred
    # Apply sigmoid and threshold at 0.5 for multi-label classification
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
    # 1) Load and preprocess GoEmotions dataset
    ds = load_and_tokenize(
        ds_name="goemotions",
        tokenizer_name="bert-base-uncased",
        max_length=128,
        save_to_disk=False
    )

    # 2) Build the model for multi-label classification
    model = build_model(
        model_name_or_path="bert-base-uncased",
        num_labels=28
    )

    # 3) Prepare TrainingArguments with full fine-tuning settings
    args = TrainingArguments(
        output_dir="outputs/full_ft",  # output directory
        eval_strategy="epoch",         # evaluate each epoch
        save_strategy="epoch",         # save each epoch
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=10,
        logging_dir="logs",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # Use F1 score for best model selection
        greater_is_better=True,
        fp16=True,  # mixed precision if supported
        report_to="none",
        remove_unused_columns=False,  # keep the labels column
        label_names=["labels"],  # explicitly name the label column
    )

    # 4) Initialize the custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    # 5) Fine-tune the model
    trainer.train()

    # 6) Evaluate on the test set
    test_res = trainer.evaluate(eval_dataset=ds["test"])
    print("\n=== Final Test Results ===")
    print(f"Macro Precision: {test_res['eval_macro_precision']:.4f}")
    print(f"Macro Recall:    {test_res['eval_macro_recall']:.4f}")
    print(f"Macro F1:        {test_res['eval_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
