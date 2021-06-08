import argparse
import mlflow
import os
from transformers import TFAutoModel, AutoTokenizer, AutoConfig
from .data import load_data, to_tf_dataset
from .models import load_model_artifacts
from .summary_strategies import extract_main_text_subset


def main(args: argparse.Namespace):
    mlflow.log_params(vars(args))
    mlflow.tensorflow.autolog(every_n_iter=1)

    tokenizer, config, model = load_model_artifacts(
        args.model_name, cache_dir=".models"
    )

    train, val, test = load_data(cache_dir=".data")

    train = train.head(100)

    train["text"] = extract_main_text_subset(args.strategy, train, tokenizer=tokenizer)
    val["text"] = extract_main_text_subset(args.strategy, val, tokenizer=tokenizer)
    test["text"] = extract_main_text_subset(args.strategy, test, tokenizer=tokenizer)

    tf_train = to_tf_dataset(
        train, tokenizer, max_len=args.max_len, batch_size=args.batch_size
    )

    model.compile("adam", "sparse_categorical_crossentropy")
    model.fit(tf_train, steps_per_epoch=2, epochs=1)

    return tf_train, train, tokenizer, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="bert-base-uncased",
        help="ID of model in transformers library",
        dest="model_name",
    )
    ap.add_argument(
        "--strategy",
        default="first",
        help="Strategy for choosing which section of `main_text` to provider as evidence",
        dest="strategy",
    )
    ap.add_argument(
        "--maxlen",
        default=128,
        help="Maximum sequence length for model",
        dest="max_len",
        type=int,
    )
    ap.add_argument(
        "--batchsize",
        default=8,
        help="Batch size -- lower means more examples on the GPU, but might restrict longer sequences",
        dest="batch_size",
        type=int,
    )
    ap.add_argument("--lr", default=0.1, type=float, help="learning rate", dest="lr")
    args = ap.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///tracking.db"))
    result = main(args)