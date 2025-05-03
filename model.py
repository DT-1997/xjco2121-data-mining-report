from transformers import BertConfig, BertModel, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn


class MultiTaskBert(PreTrainedModel):
    """
    Multi-task BERT model with LoRA injection.
    Performs multi-label emotion classification and intensity regression.
    """
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
        # Load BERT configuration and backbone
        config = BertConfig.from_pretrained(model_name_or_path)
        super().__init__(config)
        bert = BertModel.from_pretrained(model_name_or_path, config=config)

        # Define classification and regression heads
        hidden_size = config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.regressor  = nn.Linear(hidden_size, 1)

        # Configure and inject LoRA adapters into Q/V projection
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
        )
        self.bert = get_peft_model(bert, lora_config)

        # Initialize model weights
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,    # Tensor of shape [batch_size, num_labels]
        intensity=None, # Tensor of shape [batch_size]
    ):
        # Encode inputs with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.pooler_output  # [batch_size, hidden_size]

        # Compute logits for classification and regression
        logits   = self.classifier(pooled)         # [batch_size, num_labels]
        pred_int = self.regressor(pooled).squeeze(-1)  # [batch_size]

        loss = None
        if labels is not None and intensity is not None:
            # Multi-label classification loss (BCE with logits)
            cls_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            # Regression loss (MSE)
            reg_loss = nn.MSELoss()(pred_int, intensity)
            # Combined loss (balanced weight)
            loss = 0.5 * cls_loss + 0.5 * reg_loss

        return {
            "loss": loss,
            "logits": logits,
            "intensity": pred_int,
        }