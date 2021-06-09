import argparse
import mlflow
import tensorflow as tf
import os
import numpy as np
from transformers import TFTrainer, TFTrainingArguments
from .data import load_data, to_tf_dataset
from .models import evaluate_model, load_model_artifacts
from .summary_strategies import extract_main_text_subset


def main(args: argparse.Namespace):
    mlflow.log_params(vars(args))
    # mlflow.tensorflow.autolog(every_n_iter=1)  # too verbose if you don't need it

    tokenizer, config, model = load_model_artifacts(
        args.model_name, cache_dir=".models"
    )

    # load data
    train, val, test = load_data(cache_dir=".data")

    # create the input we'll pass to the transformer according to our chosen strategy
    train["text"] = extract_main_text_subset(args.strategy, train, tokenizer=tokenizer)
    val["text"] = extract_main_text_subset(args.strategy, val, tokenizer=tokenizer)
    test["text"] = extract_main_text_subset(args.strategy, test, tokenizer=tokenizer)

    # create tf.data.Dataset objects from the datasets
    tf_train = to_tf_dataset(
        train, tokenizer, max_len=args.max_len, batch_size=args.batch_size
    )
    train_steps_per_epoch = int(len(train) / args.batch_size) + 1 * (
        len(train) % args.batch_size != 0
    )
    tf_val = to_tf_dataset(
        val, tokenizer, max_len=args.max_len, batch_size=args.batch_size
    )
    val_steps_per_epoch = int(len(val) / args.batch_size) + 1 * (
        len(val) % args.batch_size != 0
    )
    tf_test = to_tf_dataset(
        test, tokenizer, max_len=args.max_len, batch_size=args.batch_size, repeat=False
    )

    # train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer,
        loss,
        metrics=[
            "accuracy",
        ],
    )
    model.fit(
        tf_train,
        steps_per_epoch=train_steps_per_epoch,
        epochs=10,
        validation_data=tf_val,
        validation_steps=val_steps_per_epoch,
        validation_batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.01,
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    # log the metrics for the run on the test set
    metrics, confusion = evaluate_model(tf_test, model)
    mlflow.log_metrics(metrics)
    artifact_path = mlflow.active_run().to_dictionary()["info"]["artifact_uri"]
    os.makedirs(artifact_path, exist_ok=True)
    np.save(os.path.join(artifact_path, "confusion"), confusion)

    return tf_train, train, tokenizer, model, tf_test, metrics, confusion


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
        default=300,
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
    ap.add_argument("--lr", default=3e-5, type=float, help="learning rate", dest="lr")
    args = ap.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///tracking.db"))
    result = main(args)
