#!/usr/bin/env python3
"""Pretrain a multiclass MixedNet embedding from per-class RaggedMmap data."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import yaml

from microwakeword.data import MmapFeatureGenerator
from microwakeword import mixednet


class ValidationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, dataset, steps, num_classes):
        super().__init__()
        self.dataset = dataset
        self.steps = steps
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for features, labels in self.dataset.take(self.steps):
            predictions = np.argmax(self.model(features, training=False).numpy(), axis=-1)
            np.add.at(confusion, (labels.numpy().astype(np.int64), predictions), 1)
        true_positive = np.diag(confusion).astype(np.float64)
        precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
        recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        logs["val_macro_precision"] = float(precision.mean())
        logs["val_macro_recall"] = float(recall.mean())
        logs["val_macro_f1"] = float(f1.mean())
        print(
            f" - val_macro_precision: {logs['val_macro_precision']:.4f}"
            f" - val_macro_recall: {logs['val_macro_recall']:.4f}"
            f" - val_macro_f1: {logs['val_macro_f1']:.4f}"
        )


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
        stride=model.get("stride", 1),
        embedding_dim=config.get("embedding_dim", 32),
        num_classes=len(config["classes"]),
        dropout_rate=config.get("dropout_rate", 0.1),
    )


def providers(config):
    root = Path(config["feature_root"])
    missing = [name for name in config["classes"] if not (root / name).exists()]
    if missing:
        raise SystemExit(f"missing class feature directories: {', '.join(missing)}")
    return [
        MmapFeatureGenerator(
            str(root / name),
            label=index,
            sampling_weight=1.0,
            penalty_weight=1.0,
            truncation_strategy="random",
            stride=config.get("stride", 1),
            step=config.get("window_step_s", 0.01),
        )
        for index, name in enumerate(config["classes"])
    ]


def train(config):
    batch_size = config.get("batch_size", 64)
    feature_length = config.get("feature_length", 49)
    providers_by_class = providers(config)
    empty = [name for name, provider in zip(config["classes"], providers_by_class) if provider.get_mode_size("training") == 0]
    if empty:
        raise SystemExit(f"classes without training mmap: {', '.join(empty)}")

    flags = make_flags(config)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = mixednet.model(flags, (feature_length, 40), batch_size=batch_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.get("learning_rate", 1e-3)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ],
            jit_compile=False,
        )

    def train_gen():
        while True:
            for _ in range(batch_size):
                index = int(np.random.randint(len(providers_by_class)))
                yield providers_by_class[index].get_random_spectrogram(
                    "training", feature_length, "random"
                ), index

    def validation_gen():
        for index, provider in enumerate(providers_by_class):
            for feature in provider.get_feature_generator("validation", feature_length, "truncate_start"):
                yield feature, index

    output = Path(config.get("train_dir", "./outputs/base_embedding"))
    output.mkdir(parents=True, exist_ok=True)
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(tf.TensorSpec((feature_length, 40), tf.float32), tf.TensorSpec((), tf.int32)),
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        validation_gen,
        output_signature=(tf.TensorSpec((feature_length, 40), tf.float32), tf.TensorSpec((), tf.int32)),
    ).batch(batch_size, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)

    validation_steps = config.get("validation_steps", 100)
    callbacks = [
        ValidationMetrics(val_ds, validation_steps, len(config["classes"])),
        tf.keras.callbacks.ModelCheckpoint(
            output / "best.weights.h5", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=config.get("steps_per_epoch", 500),
        validation_steps=validation_steps,
        epochs=config.get("epochs", 60),
        callbacks=callbacks,
    )
    model.save_weights(output / "last.weights.h5")
    (output / "classes.txt").write_text("\n".join(config["classes"]) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
