# Base embedding pretraining

The base model is not tied to one wake phrase. It learns a small streaming audio embedding that is later adapted with a user-specific head.

## Data sources

- **Speech Commands**: isolated keyword classes for keyword-spotting features. Download outside the repository.
- **Common Voice Vietnamese**: speaker/accent/speech diversity. Keep only the required language subset.
- **Ambient/no-speech**: noise, music, TV, room tone and ordinary speech.

Suggested layout:

```text
data/external/base/
├── speech_commands/
├── common_voice_vi/
└── ambient/
```

Do not commit these folders. Keep dataset URLs, licenses, versions and checksums in a local manifest.

## Training objective

Use MixedNet as the streaming encoder, expose a 32–64 dimensional embedding before the final classifier, and pretrain on many keyword classes plus speech/no-speech negatives. The final classifier is discarded before personalization.

User enrollment then uses 10–20 positive recordings and 30–100 negative speech recordings to train a tiny head or compute a prototype.

## Split rules

Split by speaker before augmentation. Do not put augmented copies of one source recording in different splits. Keep a held-out speaker set and a long ambient validation set for false-accepts/hour.

## Licensing

Check the license and attribution requirements for every downloaded dataset before redistributing a trained model or audio sample.
