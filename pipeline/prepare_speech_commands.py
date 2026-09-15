#!/usr/bin/env python3
"""Convert Speech Commands WAVs into per-class microWakeWord RaggedMmaps."""
import argparse
import os
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def read_list(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def link_split(source, target, classes, validation, testing):
    for cls in classes:
        (target / cls / "training").mkdir(parents=True, exist_ok=True)
        (target / cls / "validation").mkdir(parents=True, exist_ok=True)
        (target / cls / "testing").mkdir(parents=True, exist_ok=True)

    for wav in source.glob("*/*.wav"):
        cls = wav.parent.name
        target_cls = cls if cls in classes else "speech" if "speech" in classes else None
        if target_cls is None:
            continue
        rel = str(wav.relative_to(source))
        split = "testing" if rel in testing else "validation" if rel in validation else "training"
        dst = target / target_cls / split / f"{cls}_{wav.name}"
        if not dst.exists():
            dst.symlink_to(wav.resolve())


def make_mmap(input_dir, output_dir, step):
    if not any(input_dir.glob("*.wav")):
        return
    clips = Clips(
        input_directory=str(input_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
    )
    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=Augmentation(augmentation_duration_s=1.0, augmentation_probabilities={}),
        slide_frames=10 if step == "training" else 1,
        step_ms=10,
    )
    out = output_dir / f"{step}_mmap"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=str(out),
        sample_generator=spectrograms.spectrogram_generator(repeat=1),
        batch_size=100,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="extracted speech_commands directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--classes", default="yes,no,up,down,left,right,stop,go")
    args = parser.parse_args()
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    validation = read_list(args.source / "validation_list.txt")
    testing = read_list(args.source / "testing_list.txt")
    split_root = args.output / "split"
    link_split(args.source, split_root, classes, validation, testing)
    for cls in classes:
        for split in ("training", "validation", "testing"):
            make_mmap(split_root / cls / split, args.output / "features" / cls, split)
    print(f"prepared {len(classes)} classes under {args.output / 'features'}")


if __name__ == "__main__":
    main()
