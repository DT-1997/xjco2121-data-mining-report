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

