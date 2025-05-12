import torch
import random
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer
from preprocess import load_and_tokenize
from model import build_model
from sklearn.metrics import precision_recall_fscore_support

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA seed (for GPU)
    # Optional: Ensure deterministic behavior (for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Trainer class to ensure labels are cast to float32 before loss computation
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics = []  # To log metrics for each epoch

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")  # Get the labels from inputs
        if labels is not None:
            inputs["labels"] = labels.to(torch.float32)  # Convert labels to float32
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)  # Compute loss using the parent class method
        return (loss, outputs) if return_outputs else loss  # Return loss and outputs (if needed)

    def log_metrics(self, metrics):
        print(f"Logging metrics: {metrics}")  # Debugging log to check if metrics are being logged
        self.epoch_metrics.append(metrics)  # Log metrics for each epoch

# Function to compute evaluation metrics (precision, recall, and F1 score)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()  # Apply sigmoid to logits for multi-label classification
    preds = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0  # Use macro average for multi-label classification
    )
    return {
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # 1) Load and preprocess GoEmotions dataset
    ds = load_and_tokenize(
        ds_name="goemotions",  # Dataset name
        tokenizer_name="bert-base-uncased",  # Pretrained BERT tokenizer
        max_length=128,  # Max sequence length for tokenization
        save_to_disk=False,  # Don't save the tokenized dataset to disk
        dataset_version="goemotions"
    )

    # 2) Build the multi-label classification model
    model = build_model(
        model_name_or_path="bert-base-uncased",  # Pretrained BERT model
        num_labels=28  # Number of classes (labels)
    )

    # 3) Set up training arguments with full fine-tuning settings
    args = TrainingArguments(
        output_dir="outputs/full_ft",  # Output directory
        eval_strategy="epoch",  # Evaluate the model after each epoch
        save_strategy="epoch",  # Save the model after each epoch
        per_device_train_batch_size=16,  # Training batch size per device
        per_device_eval_batch_size=32,  # Evaluation batch size per device
        learning_rate=2e-5,  # Learning rate
        num_train_epochs=10,  # Number of training epochs (set to 1 for testing)
        logging_dir="logs",  # Directory for logging
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="macro_f1",  # Use F1 score for selecting the best model
        greater_is_better=True,  # Higher F1 score is better
        fp16=True,  # Use mixed precision training if supported
        report_to="none",  # No need to report to external services
        remove_unused_columns=False,  # Keep the labels column in the dataset
        label_names=["labels"],  # Explicitly name the label column
    )

    # 4) Initialize the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],  # Training dataset
        eval_dataset=ds["validation"],  # Validation dataset
        compute_metrics=compute_metrics,  # Function to compute evaluation metrics
    )

    # 5) Start training the model
    trainer.train()

    # 6) Evaluate the model on the test set
    test_res = trainer.evaluate(eval_dataset=ds["test"])
    print("\n=== Final Test Results ===")
    print(f"Macro Precision: {test_res['eval_macro_precision']:.4f}")
    print(f"Macro Recall:    {test_res['eval_macro_recall']:.4f}")
    print(f"Macro F1:        {test_res['eval_macro_f1']:.4f}")

    # 7) Save the final model to the output directory
    model.save_pretrained("outputs/full_ft/final_model")  # Save the model weights and configuration
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Load tokenizer
    tokenizer.save_pretrained("outputs/full_ft/final_model")  # Save the tokenizer

if __name__ == "__main__":
    main()
