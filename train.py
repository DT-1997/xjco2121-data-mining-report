# src/train_mtl.py

from transformers import Trainer, TrainingArguments
from src.model import MultiTaskBert
from src.preprocess import load_and_tokenize

def main():
    # 1. load tokenized dataset
    ds = load_and_tokenize("semeval2018", tokenizer_name="bert-base-uncased")

    # 2. init model
    model = MultiTaskBert(
        model_name_or_path="bert-base-uncased",
        num_labels=11,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # 3. training args
    args = TrainingArguments(
        output_dir="outputs/run1",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=3e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_cls_macro_f1",
    )

    # 4. compute_metrics
    def compute_metrics(pred):
        logits, pred_int, labels_and_int = pred
        # ... same as before ...
        return { ... }

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(ds["test"])

if __name__ == "__main__":
    main()
