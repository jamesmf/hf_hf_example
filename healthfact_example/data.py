from typing import Tuple
import datasets
import pandas as pd
import tensorflow as tf
from typing import List
from transformers import BertTokenizer


def load_data(cache_dir: str = ".data") -> List[datasets.Dataset]:
    healthfact_data = datasets.load_dataset("health_fact", cache_dir=cache_dir)
    output_dfs: List[pd.DataFrame] = []

    for split in ("train", "validation", "test"):
        sub = healthfact_data[split].filter(
            lambda example: len(example["claim"]) <= 400
        )
        sub = sub.filter(lambda example: len(example["main_text"]) >= 50)
        sub = sub.filter(lambda example: example["label"] >= 0)
        output_dfs.append(sub.to_pandas()[["main_text", "claim", "label"]])
    return output_dfs


def map_example_to_dict(
    input_ids: tf.Tensor,
    attention_masks: tf.Tensor,
    token_type_ids: tf.Tensor,
    label: tf.Tensor,
):
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, label


def to_tf_dataset(
    df: pd.DataFrame, tokenizer: BertTokenizer, batch_size: int = 4, max_len: int = 128
) -> tf.data.Dataset:
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids, token_types, attention_masks, labels = [], [], [], []

    encoded = tokenizer(list(df["text"].values))
    labels = [[v] for v in df.label.values]
    for n in range(len(df)):
        input_ids.append(encoded["input_ids"][n][:max_len])
        token_types.append(encoded["token_type_ids"][n][:max_len])
        attention_masks.append(encoded["attention_mask"][n][:max_len])
    input_ids = tf.ragged.constant(input_ids, dtype=tf.int32)
    token_types = tf.ragged.constant(token_types, dtype=tf.int32)
    attention_masks = tf.ragged.constant(attention_masks, dtype=tf.int32)
    return (
        tf.data.Dataset.from_tensor_slices(
            (input_ids, token_types, attention_masks, labels)
        )
        .map(map_example_to_dict)
        .shuffle(buffer_size=1000)
        .padded_batch(batch_size)
    )
