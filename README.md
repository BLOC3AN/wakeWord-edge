# wakeWord-edge

MixedNet pipeline for a reusable audio embedding and custom 2–3 syllable wake words on ESP32-S3. Large audio, feature maps, checkpoints and TFLite files stay outside Git.

## Current flow

```text
Speech Commands / custom audio
        ↓
prepare_speech_commands.py
        ↓
RaggedMmap features
        ↓
MixedNet base pretraining (V2 baseline)
        ↓
fixed validation/test evaluation
        ↓
Optuna V2.1 sweep
        ↓
user wakeword head → TFLite Micro / ESP32-S3
```

## Repository

```text
configs/                 small YAML examples
data/                    ignored audio/features
docs/                    dataset, training and deployment notes
models/micro-wake-word/  vendored MixedNet runtime/trainer
pipeline/                preparation, training, evaluation and Optuna
scripts/                 environment, custom training, export and metric checks
outputs/                 ignored checkpoints/reports
```

Piper voices and other datasets are external dependencies; no voice model is bundled.

## Setup

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

## Base pretraining

```bash
cp configs/base_pretrain.example.yaml configs/base_pretrain.yaml
python pipeline/prepare_speech_commands.py \
  data/external/base/downloads/speech_commands_v0.02 \
  data/external/base
CUDA_VISIBLE_DEVICES=2,3 python pipeline/pretrain_embedding.py configs/base_pretrain.yaml
```

The trainer saves `best.weights.h5`, `last.weights.h5` and `classes.txt` under `outputs/`.

## Fixed evaluation

Use the same clean feature tensor for every checkpoint; never select a winner from the test set during tuning.

```bash
python pipeline/evaluate_checkpoints.py \
  --split validation --samples-per-class 350 \
  --run v2 configs/base_pretrain.yaml outputs/base_embedding_v2/best.weights.h5
```

The report includes accuracy, top-3 accuracy, macro precision/recall/F1, loss and a confusion matrix. These are pretraining metrics, not wakeword FAR/FRR.

## Optuna V2.1

```bash
python pipeline/optuna_sweep.py configs/base_pretrain.yaml \
  --trials 12 --epochs 25 \
  --storage sqlite:///outputs/optuna_v21.db
```

Trials optimize validation macro-F1. Median pruning stops weak trials after the warmup; the held-out test split is used only after the final candidates are retrained.

## Custom wakeword

```bash
cp configs/mixednet.example.yaml configs/mixednet.yaml
./scripts/train_mixednet.sh configs/mixednet.yaml
./scripts/export_tflite.sh configs/mixednet.yaml
python scripts/check_metrics.py <streaming_roc.txt> --max-faph 1
```

Keep positive recordings speaker-disjoint from validation/test, add hard negatives and measure false rejects, false accepts/hour, latency and arena RAM on the target board.

## License

Project code is Apache-2.0. Third-party code and datasets retain their own licenses.
