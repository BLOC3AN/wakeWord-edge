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

The model builder now supports `embedding_dim` and `num_classes` options. The default remains the original one-output binary wakeword head, so existing wakeword training is unchanged. The multiclass pretraining loop will consume this embedding head once the base dataset manifests are prepared.

After each class has been converted to the same RaggedMmap layout under `data/external/base/features/<class>/`, run:

```bash
python pipeline/pretrain_embedding.py configs/base_pretrain.yaml
```

The script writes only checkpoints and `classes.txt` under the ignored `outputs/` directory.

User enrollment then uses 10–20 positive recordings and 30–100 negative speech recordings to train a tiny head or compute a prototype.

## Split rules

Split by speaker before augmentation. Do not put augmented copies of one source recording in different splits. Keep a held-out speaker set and a long ambient validation set for false-accepts/hour.

## Licensing

Check the license and attribution requirements for every downloaded dataset before redistributing a trained model or audio sample.
