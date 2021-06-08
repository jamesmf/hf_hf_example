from transformers import (
    TFAutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
)
from typing import Tuple


def load_model_artifacts(
    model_name: str, cache_dir: str = ".models"
) -> Tuple[BertTokenizer, BertConfig, BertForSequenceClassification]:
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir, num_labels=4
    )
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return (tokenizer, config, model)