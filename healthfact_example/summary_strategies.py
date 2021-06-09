"""
This submodule contains strategies for choosing which piece of the `main_text` should be included in the final model.

The paper originally uses S-BERT to embed each sentences, then chooses the `k` most similar.

Parameterizing the function that chooses the text to use allows us to track performance differences across different strategies.

Each strategy can optionally cache its results.
"""
import pandas as pd
from typing import List
from transformers import AutoTokenizer
import transformers


def extract_main_text_subset(strategy: str, df: pd.DataFrame, **kwargs) -> List[str]:
    if strategy == "first":
        return extract_first(df, kwargs.get("n_tokens", 5000))
    assert False, "Passed an invalid strategy"


def extract_first(
    df: pd.DataFrame,
    n_chars: int,
) -> List[str]:
    return list(
        (df["claim"] + " [SEP] " + df["main_text"].apply(lambda x: x[:n_chars])).values
    )
