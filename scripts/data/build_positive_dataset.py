"""
Script 08: Augment + split vi_samples cho wake word "buồn"
Đọc tất cả 5 variants: buồn, buồn quá, buồn buồn, buồn ghê, thấy buồn
→ Augment × TARGET_AUG mỗi file → chia train/val/test 80/10/10
→ Lưu vào data/datasets/buon/train/buon/, val/buon/, test/buon/
"""

import os
import random
import pathlib
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ──────────────────────────────────────────────────────────────────
SR = 16000
DURATION_SEC = 1.0
TARGET_AUG = 3          # Số bản augment mỗi file (không tính bản gốc) → tổng 4x
NUM_THREADS = 16
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test = 1 - TRAIN - VAL = 0.10

BASE_DIR    = pathlib.Path(__file__).resolve().parent.parent
VI_DIR      = BASE_DIR / "data/generated/vi_samples"
DATASET_DIR = BASE_DIR / "data/datasets/buon"

# 5 variants đều là positive cho wake word "buồn"
POSITIVE_VARIANTS = [
    "buồn",
    "buồn quá",
    "buồn buồn",
    "buồn ghê",
    "thấy buồn",
]

random.seed(42)
np.random.seed(42)

# ── Helpers ──────────────────────────────────────────────────────────────────
def center_pad_or_crop(audio: np.ndarray, sr: int, sec: float = 1.0) -> np.ndarray:
    target = int(sr * sec)
    if len(audio) >= target:
        start = (len(audio) - target) // 2
        return audio[start : start + target].copy()
    pad_total = target - len(audio)
    return np.pad(audio, (pad_total // 2, pad_total - pad_total // 2), mode="constant").astype(np.float32)


def augment_sample(audio: np.ndarray, sr: int) -> np.ndarray:
    x = audio.copy()

    # Time shift ±100 ms
    shift = random.randint(-int(0.1 * sr), int(0.1 * sr))
    if shift > 0:
        x = np.pad(x, (shift, 0), mode="constant")[:-shift]
    elif shift < 0:
        x = np.pad(x, (0, -shift), mode="constant")[-shift:]

    # Pitch shift ±2 semitones (50% probability)
    if random.random() < 0.5:
        x = librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-2, 2))

    # Speed perturbation 0.9–1.1 (50% probability)
    if random.random() < 0.5:
        rate = random.uniform(0.9, 1.1)
        x = librosa.effects.time_stretch(x, rate=rate)
        x = center_pad_or_crop(x, sr, DURATION_SEC)

    # Add background noise
    if random.random() < 0.8:
        noise_level = random.uniform(0.001, 0.025)
        x = x + np.random.randn(len(x)).astype(np.float32) * noise_level

    # Random gain 0.5–1.2
    x = x * random.uniform(0.5, 1.2)

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def assign_split() -> str:
    r = random.random()
    if r < TRAIN_RATIO:
        return "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Tạo thư mục output
    for split in ["train", "val", "test"]:
        (DATASET_DIR / split / "buon").mkdir(parents=True, exist_ok=True)

    # Thu thập tất cả WAV từ 5 variants
    all_wavs = []
    for variant in POSITIVE_VARIANTS:
        variant_dir = VI_DIR / variant
        if not variant_dir.exists():
            print(f"[WARN] Không tìm thấy: {variant_dir}")
            continue
        wavs = list(variant_dir.glob("*.wav"))
        print(f"  {variant}: {len(wavs)} files")
        all_wavs.extend(wavs)

    print(f"\nTổng cộng: {len(all_wavs)} WAV files từ {len(POSITIVE_VARIANTS)} variants")
    print(f"Sau augment ×{TARGET_AUG + 1}: ~{len(all_wavs) * (TARGET_AUG + 1):,} samples\n")

    def process_one(wav_path: pathlib.Path) -> int:
        try:
            audio, _ = librosa.load(wav_path, sr=SR)
        except Exception as e:
            print(f"[ERROR] Bỏ qua {wav_path.name}: {e}")
            return 0

        audio_1s = center_pad_or_crop(audio, SR, DURATION_SEC)
        split = assign_split()
        out_dir = DATASET_DIR / split / "buon"

        base = f"{wav_path.parent.name.replace(' ', '_')}_{wav_path.stem}"
        sf.write(out_dir / f"{base}_orig.wav", audio_1s, SR, subtype="PCM_16")
        count = 1

        for i in range(TARGET_AUG):
            aug = augment_sample(audio_1s, SR)
            sf.write(out_dir / f"{base}_aug{i}.wav", aug, SR, subtype="PCM_16")
            count += 1

        return count

    total_saved = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_one, p): p for p in all_wavs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
            total_saved += future.result()

    # Thống kê
    print(f"\n✅ Xong! Tổng {total_saved:,} samples đã lưu vào {DATASET_DIR}")
    for split in ["train", "val", "test"]:
        n = len(list((DATASET_DIR / split / "buon").glob("*.wav")))
        print(f"  {split}: {n:,} files")


if __name__ == "__main__":
    main()
