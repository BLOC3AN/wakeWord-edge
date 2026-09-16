#!/usr/bin/env python3
"""Short Optuna sweep for the MixedNet base-pretraining configuration."""
import argparse
import json
import shutil
from pathlib import Path

import optuna
import tensorflow as tf
import yaml

from pretrain_embedding import train


def objective(trial, base_config, args, strategy):
    config = dict(base_config)
    width = trial.suggest_categorical("pointwise_width", [48, 64])
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.05, 0.25)
    config["embedding_dim"] = trial.suggest_categorical("embedding_dim", [32, 48])
    first_filters = trial.suggest_categorical("first_conv_filters", [32, 48, 64])
    config["model"] = {
        "pointwise_filters": ",".join([str(width)] * 4),
        "first_conv_filters": first_filters,
        "first_conv_kernel_size": 3,
        "residual_connection": "0,0,0,0",
        "repeat_in_block": "1,1,1,1",
        "mixconv_kernel_sizes": "[5], [9], [13], [21]",
        "spatial_attention": 0,
        "pooled": 0,
        "max_pool": 0,
    }
    config["epochs"] = args.epochs
    config["steps_per_epoch"] = args.steps_per_epoch
    config["validation_steps"] = None
    config["validation_samples_per_class"] = args.validation_samples_per_class
    config["seed"] = args.seed + trial.number
    trial_dir = args.output / f"trial-{trial.number:03d}"
    config["train_dir"] = str(trial_dir)
    try:
        result = train(config, trial=trial, strategy=strategy)
    except RuntimeError as error:
        if str(error) == "optuna_pruned":
            shutil.rmtree(trial_dir, ignore_errors=True)
            raise optuna.TrialPruned()
        raise
    finally:
        tf.keras.backend.clear_session()
        shutil.rmtree(trial_dir, ignore_errors=True)
    for key, value in result.items():
        trial.set_user_attr(key, value)
    return result["val_macro_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="base-pretraining YAML")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-samples-per-class", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/optuna_v21"))
    parser.add_argument("--storage", help="Optuna storage URL, e.g. sqlite:///outputs/optuna_v21.db")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    strategy = tf.distribute.MirroredStrategy()
    study = optuna.create_study(
        study_name="mixednet_v21",
        direction="maximize",
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,
            n_warmup_steps=8,
            interval_steps=1,
        ),
    )
    study.optimize(lambda trial: objective(trial, base_config, args, strategy), n_trials=args.trials)
    report = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ],
    }
    (args.output / "study.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best_config = dict(base_config)
    best_config.update(
        learning_rate=study.best_params["learning_rate"],
        batch_size=study.best_params["batch_size"],
        dropout_rate=study.best_params["dropout_rate"],
        embedding_dim=study.best_params["embedding_dim"],
        model={
            "pointwise_filters": ",".join([str(study.best_params["pointwise_width"])] * 4),
            "first_conv_filters": study.best_params["first_conv_filters"],
            "first_conv_kernel_size": 3,
            "residual_connection": "0,0,0,0",
            "repeat_in_block": "1,1,1,1",
            "mixconv_kernel_sizes": "[5], [9], [13], [21]",
        },
        train_dir=str(args.output / "final_best"),
    )
    (args.output / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))


if __name__ == "__main__":
    main()
