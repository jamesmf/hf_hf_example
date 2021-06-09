from transformers import (
    TFAutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
)
import tensorflow as tf
import numpy as np
from typing import Tuple, Any, Dict
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def load_model_artifacts(
    model_name: str, cache_dir: str = ".models"
) -> Tuple[BertTokenizer, BertConfig, BertForSequenceClassification]:
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir, num_labels=4
    )
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return (tokenizer, config, model)


def evaluate_model(
    dataset: tf.data.Dataset, model: BertForSequenceClassification
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Custom evaluation on the test dataset. Simpler than defining Sparse versions of metrics
    """
    preds, all_labels = [], []
    # softmax in case we use metrics that don't work with logits - for now all do
    activation = tf.keras.layers.Activation("softmax")
    # small batches avoid drastic differences in padding
    for examples, labels in tqdm(dataset):
        preds.append(activation(model.predict(examples)["logits"]))
        all_labels.append(labels)
    preds = np.array(preds).reshape(-1, 4).argmax(-1)
    dense_preds = tf.keras.utils.to_categorical(preds, 4)
    all_labels = np.array(all_labels).reshape(-1, 1)
    dense_labels = tf.keras.utils.to_categorical(all_labels, 4)
    precision, recall, f1, _ = precision_recall_fscore_support(
        dense_labels, dense_preds, average="macro"
    )
    acc = accuracy_score(dense_labels, dense_preds)
    confusion = confusion_matrix(all_labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }, confusion