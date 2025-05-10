# # model.py

# from transformers import BertConfig, BertForSequenceClassification
# from peft import LoraConfig, get_peft_model, TaskType

# def build_model(
#     model_name_or_path: str = "bert-base-uncased",
#     num_labels: int = 28,
#     lora_r: int    = 8,
#     lora_alpha: int= 32,
#     lora_dropout: float = 0.05,
# ):
#     # 1) Load config and set multi-label
#     config = BertConfig.from_pretrained(model_name_or_path)
#     config.num_labels     = num_labels
#     config.problem_type   = "multi_label_classification"

#     # 2) Load seq-classification model (has its own classifier head)
#     model = BertForSequenceClassification.from_pretrained(
#         model_name_or_path,
#         config=config
#     )

#     # 3) Configure LoRA adapters for query & value projections
#     peft_config = LoraConfig(
#         task_type=TaskType.SEQ_CLS,
#         inference_mode=False,
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         lora_dropout=lora_dropout,
#         target_modules=["query", "value"],
#     )
#     #    Wrap the entire classification model
#     model = get_peft_model(model, peft_config)

#     return model

# model.py

from transformers import BertConfig, BertForSequenceClassification

def build_model(
    model_name_or_path: str = "bert-base-uncased",
    num_labels: int    = 28,
):
    """
    Load a BertForSequenceClassification configured
    for multi-label classification, and return it.
    """

    # 1. Load BERT config and set number of labels + problem type
    config = BertConfig.from_pretrained(model_name_or_path)
    config.num_labels   = num_labels
    config.problem_type = "multi_label_classification"

    # 2. Load the pre-trained model with a fresh classification head
    model = BertForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config
    )

    return model

