# Deployment

The exported model must use the same frontend as training: 16 kHz mono audio, 30 ms window, 10 ms stride and 40 feature bins. Keep streaming state between invokes.

```bash
./scripts/export_tflite.sh configs/mixednet.yaml
```

Only trigger the wake event after the probability threshold is met for the chosen number of consecutive frames. Select that threshold from real ambient recordings, not training accuracy.

Before shipping, measure:

- TFLite file size and tensor arena;
- per-frame latency;
- false accepts/hour and false rejects;
- behavior with VAD and other ESP32-S3 workloads.

The base embedding head is removed or replaced by a tiny user-specific head during personalization.
