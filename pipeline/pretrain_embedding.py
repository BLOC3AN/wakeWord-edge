#!/usr/bin/env python3
"""Pretrain a multiclass MixedNet embedding from per-class RaggedMmap data."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import yaml

from microwakeword.data import MmapFeatureGenerator
from microwakeword import mixednet


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
    if any(p.get_mode_size("training") == 0 for p in providers_by_class):
        raise SystemExit("every class needs at least one training mmap")

    flags = make_flags(config)
    model = mixednet.model(flags, (feature_length, 40), batch_size=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.get("learning_rate", 1e-3)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    def train_gen():
        while True:
            for _ in range(batch_size):
                index = tf.random.uniform([], maxval=len(providers_by_class), dtype=tf.int32)
                index = int(index.numpy())
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
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        validation_gen,
        output_signature=(tf.TensorSpec((feature_length, 40), tf.float32), tf.TensorSpec((), tf.int32)),
    ).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=config.get("steps_per_epoch", 500),
        validation_steps=config.get("validation_steps", 100),
        epochs=config.get("epochs", 20),
        callbacks=[tf.keras.callbacks.ModelCheckpoint(output / "best.weights.h5", save_best_only=True, save_weights_only=True)],
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
