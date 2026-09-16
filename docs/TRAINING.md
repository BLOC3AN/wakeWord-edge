# Training and tuning

## MixedNet baseline

The current V2 baseline uses four MixedNet blocks with 48 pointwise filters, a 32-dimensional embedding and 0.1 dropout. It has about 20,393 parameters (~80 KiB FP32 weights, ~20 KiB INT8 weights).

The trainer uses Adam, ReduceLROnPlateau, early stopping and macro precision/recall/F1. The fixed benchmark is the source of truth; the old per-run validation counters are not comparable across experiments.

## V2.1 Optuna sweep

`pipeline/optuna_sweep.py` searches only a small, useful space:

- learning rate, batch size and dropout;
- embedding dimension 32/48;
- pointwise width 48/64;
- first convolution width 32/48/64.

Median pruning reports validation macro-F1 after each epoch. Start with short trials, then retrain the top candidates for up to 100 epochs with early stopping and at least three seeds. Keep the test split untouched until selection is complete.

```bash
python pipeline/optuna_sweep.py configs/base_pretrain.yaml \
  --trials 12 --epochs 25 --storage sqlite:///outputs/optuna_v21.db
```

A wider model is still small enough for ESP32-S3: 32k parameters are roughly 125 KiB in FP32 or 31 KiB as raw INT8 weights. Measure the final TFLite arena and latency before shipping.

## Custom wakeword metrics

Base pretraining accuracy is only a proxy. For the final wakeword, report recall, false-reject rate, false-accepts/hour, ROC/AUC, threshold and board latency/RAM. A `nan` ambient metric is a failed evaluation, not a passing result.
