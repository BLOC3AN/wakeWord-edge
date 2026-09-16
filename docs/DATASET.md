# Dataset contract

## Audio

- mono PCM WAV, 16 kHz;
- positive class is one wake phrase only;
- split by speaker/file before any augmentation;
- validation and testing stay clean.

The base pretraining dataset uses nine classes: `yes`, `no`, `up`, `down`, `left`, `right`, `stop`, `go` and `speech`. The generated feature bank is ignored by Git.

```text
data/external/base/
├── downloads/speech_commands_v0.02/
├── split/<class>/{training,validation,testing}/
└── features/<class>/{training,validation,testing}/
```

## Preparation

```bash
python pipeline/prepare_speech_commands.py \
  data/external/base/downloads/speech_commands_v0.02 \
  data/external/base
```

For a controlled background experiment, augment training only:

```bash
python pipeline/prepare_speech_commands.py \
  data/external/base/downloads/speech_commands_v0.02 \
  data/external/base_augmented \
  --repeat 2 --augment-train --background-only \
  --background-probability 0.2 \
  --background-min-snr-db 10 --background-max-snr-db 25 \
  --background-dir data/external/base/downloads/speech_commands_v0.02/_background_noise_
```

Use real recordings for the final wakeword. TTS can increase speaker/speed diversity, but it does not replace microphone recordings.

## Fixed evaluation

The current benchmark caps each class to the same deterministic number of clean features: 350/class for validation and 396/class for testing in the available Speech Commands split. Keep this benchmark unchanged while tuning.

Do not use the testing split to choose Optuna parameters.
