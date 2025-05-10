# model.py

from transformers import BertConfig, BertModel, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn


class MultiTaskBert(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        num_labels: int = 11,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        # 1. Load BERT configuration and pretrained weights
        config = BertConfig.from_pretrained(model_name_or_path)
        super().__init__(config)
        backbone = BertModel.from_pretrained(model_name_or_path, config=config)

        # 2. Define a classification head for multi-label emotion categories
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        #    and a regression head for emotion intensity scores
        self.regressor = nn.Linear(config.hidden_size, 1)

        # 3. Set up LoRA adapters targeting the query & value projection matrices
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
        )
        #    Inject LoRA into BERT backbone to enable parameter-efficient fine-tuning
        self.bert = get_peft_model(backbone, peft_config)

        # 4. Initialize newly added modules (classification/regression heads + LoRA adapters)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        intensity=None,
    ):
        """
        Accept exactly the arguments Trainer will supply:
        - input_ids, attention_mask, token_type_ids → only these go into BERT
        - labels, intensity                         → used solely for loss computation
        """
        # A) Prepare inputs for BERT encoder, excluding any label fields
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids

        # B) Forward pass through the BERT encoder (LoRA-wrapped)
        outputs = self.bert(**bert_inputs)
        pooled = outputs.pooler_output  # shape: [batch_size, hidden_size]

        # C) Apply task-specific heads
        logits = self.classifier(pooled)          # [batch_size, num_labels]
        pred_intensity = self.regressor(pooled)   # [batch_size, 1]
        pred_intensity = pred_intensity.squeeze(-1)

        # D) Compute combined loss if ground-truth is provided
        loss = None
        if labels is not None and intensity is not None:
            # 1) Multi-label classification loss (BCE with logits)
            cls_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            # 2) Regression loss (MSE)
            reg_loss = nn.MSELoss()(pred_intensity, intensity)
            # 3) Combine them equally
            loss = 0.5 * cls_loss + 0.5 * reg_loss

        # E) Return a dict compatible with Trainer: includes 'loss' key
        return {
            "loss": loss,
            "logits": logits,
            "intensity": pred_intensity,
        }
