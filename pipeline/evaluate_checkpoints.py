#!/usr/bin/env python3
"""Evaluate MixedNet checkpoints on one fixed, deterministic feature set."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import yaml

from microwakeword import mixednet
from microwakeword.data import MmapFeatureGenerator


def make_flags(config):
    model = config.get("model", {})
    return SimpleNamespace(
        pointwise_filters=model.get("pointwise_filters", "48, 48, 48, 48"),
        residual_connection=model.get("residual_connection", "0,0,0,0"),
        repeat_in_block=model.get("repeat_in_block", "1,1,1,1"),
        mixconv_kernel_sizes=model.get("mixconv_kernel_sizes", "[5], [9], [13], [21]"),
        max_pool=model.get("max_pool", 0),
        first_conv_filters=model.get("first_conv_filters", 32),
        first_conv_kernel_size=model.get("first_conv_kernel_size", 3),
        spatial_attention=model.get("spatial_attention", 0),
        pooled=model.get("pooled", 0),
        stride=config.get("stride", 1),
        embedding_dim=config.get("embedding_dim", 32),
        num_classes=len(config["classes"]),
        dropout_rate=config.get("dropout_rate", 0.1),
    )


def fixed_examples(config, split, samples_per_class, feature_length):
    root = Path(config["feature_root"])
    features = []
    labels = []
    for label, class_name in enumerate(config["classes"]):
        provider = MmapFeatureGenerator(
            str(root / class_name),
            label=label,
            sampling_weight=1.0,
            penalty_weight=1.0,
            truncation_strategy="truncate_start",
            stride=config.get("stride", 1),
            step=config.get("window_step_s", 0.01),
        )
        count = min(samples_per_class, provider.get_mode_size(split))
        if count == 0:
            raise RuntimeError(f"{class_name} has no {split} features under {root}")
        for index, feature in enumerate(
            provider.get_feature_generator(split, feature_length, "truncate_start")
        ):
            if index >= count:
                break
            features.append(feature)
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def metrics(labels, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    classes = probabilities.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    top_k = min(3, classes)
    top = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
    return {
        "samples": int(labels.size),
        "loss": float(np.mean(loss.numpy())),
        "accuracy": float(np.mean(predictions == labels)),
        "top3_accuracy": float(np.mean(np.any(top == labels[:, None], axis=1))),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_run(name, config_path, weights_path, features, labels, batch_size):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = mixednet.model(
        make_flags(config),
        (config.get("feature_length", 49), 40),
        batch_size=batch_size,
    )
    model.load_weights(weights_path)
    probabilities = model.predict(features, batch_size=batch_size, verbose=0)
    result = metrics(labels, probabilities)
    result["config"] = str(config_path)
    result["weights"] = str(weights_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "testing"), default="validation")
    parser.add_argument("--samples-per-class", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "WEIGHTS"),
        required=True,
    )
    args = parser.parse_args()
    first_config = yaml.safe_load(Path(args.run[0][1]).read_text(encoding="utf-8"))
    features, labels = fixed_examples(
        first_config,
        args.split,
        args.samples_per_class,
        first_config.get("feature_length", 49),
    )
    report = {
        "split": args.split,
        "samples_per_class": args.samples_per_class,
        "samples": int(labels.size),
        "feature_root": first_config["feature_root"],
        "runs": {},
    }
    for name, config_path, weights_path in args.run:
        try:
            report["runs"][name] = evaluate_run(
                name, config_path, weights_path, features, labels, args.batch_size
            )
        except Exception as error:
            report["runs"][name] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "config": config_path,
                "weights": weights_path,
            }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
