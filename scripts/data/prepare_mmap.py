"""
Script 09: Chuyển data/datasets/buon/ → data/mmap/buon/ (RaggedMmap format)
Dùng microwakeword SpectrogramGeneration — giống script 05 nhưng cho keyword buồn.
"""

import os
import pathlib as _pl
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.clips import Clips
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.spectrograms import SpectrogramGeneration

BASE_DIR    = str(_pl.Path(__file__).resolve().parent.parent)
DATASET_DIR = str(_pl.Path(BASE_DIR) / "data/datasets/buon")
OUTPUT_DIR  = str(_pl.Path(BASE_DIR) / "data/mmap/buon")

# split name trong folder → tên microWakeWord mong đợi
SPLIT_MAP = {
    "train": "training",
    "val":   "validation",
    "test":  "testing",
}


def create_mmap(split_name: str, folder_name: str):
    mww_split = SPLIT_MAP[split_name]
    input_dir = os.path.join(DATASET_DIR, split_name, folder_name)

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"[SKIP] {input_dir} — không tồn tại hoặc rỗng")
        return

    out_dir = os.path.join(OUTPUT_DIR, folder_name, mww_split, f"{folder_name}_mmap")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    if os.path.exists(out_dir):
        print(f"[SKIP] {out_dir} — đã tồn tại, xoá thủ công nếu muốn tạo lại")
        return

    print(f"[INFO] Tạo mmap: {folder_name}/{mww_split} ...")

    clips = Clips(
        input_directory=input_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
    )
    augmenter = Augmentation(augmentation_duration_s=1.0, augmentation_probabilities={})

    # slide_frames=10 cho train (streaming simulation), =1 cho val/test
    slide = 10 if split_name == "train" else 1
    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        slide_frames=slide,
        step_ms=10,
    )

    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=spectrograms.spectrogram_generator(repeat=1),
        batch_size=100,
        verbose=True,
    )
    print(f"[DONE] {out_dir}")


def main():
    for split in ["train", "val", "test"]:
        create_mmap(split, "buon")
    print("\n✅ Tất cả mmap cho 'buon' đã xong!")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
