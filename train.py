# train.py

import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer
from model import MultiTaskBert
from preprocess import load_and_tokenize
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, r2_score

def compute_metrics(eval_pred):
    """
    Compute multi-label Macro-F1 for classification
    and RMSE/R2 for regression.
    """
    logits, pred_int, labels_and_int = eval_pred

    # labels_and_int is a dict with 'labels' and 'intensity'
    labels = labels_and_int['labels'].numpy()
    intensity = labels_and_int['intensity'].numpy()

    # classification: sigmoid + threshold 0.5
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # regression
    rmse = mean_squared_error(intensity, pred_int)
    r2   = r2_score(intensity, pred_int)

    return {
        'cls_macro_f1': f1,
        'reg_rmse': rmse,
        'reg_r2': r2
    }


def main():
    # 1. Load and tokenize dataset (we use local data/goemotions JSONL)
    ds = load_and_tokenize(
        ds_name='goemotions',
        tokenizer_name='bert-base-uncased',
        max_length=128,
        save_to_disk=False  # we already have the JSONL files
    )

    # 2. Initialize tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskBert(
        model_name_or_path='bert-base-uncased',
        num_labels=28,     # GoEmotions 有 27 emotion + 1 neutral = 28
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # 3. Set up training arguments
    args = TrainingArguments(
        output_dir='outputs/goe_run1',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        num_train_epochs=5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        load_best_model_at_end=True,
        metric_for_best_model='cls_macro_f1',
        fp16=True,  # enable mixed precision if supported
    )

    # 4. Create Trainer and start training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    # 5. Final evaluation on test set
    test_metrics = trainer.evaluate(eval_dataset=ds['test'])
    print("=== Test Metrics ===")
    print(test_metrics)


if __name__ == '__main__':
    main()
