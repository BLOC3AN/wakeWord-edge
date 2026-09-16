#!/usr/bin/env python3
"""Convert Speech Commands WAVs into per-class microWakeWord RaggedMmaps."""
import argparse
import os
from pathlib import Path

import soundfile as sf
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.spectrograms import SpectrogramGeneration


class LocalClips:
    """Read local WAVs without the optional Hugging Face audio decoder."""

    def __init__(self, paths):
        self.paths = sorted(paths)

    def audio_generator(self, repeat=1, **_):
        for _ in range(max(1, repeat)):
            for path in self.paths:
                audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
                if sample_rate != 16000:
                    raise ValueError(f"expected 16 kHz WAV: {path}")
                yield audio


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


def make_mmap(input_dir, output_dir, step, repeat, augmenter):
    if not any(input_dir.glob("*.wav")):
        return
    clips = LocalClips(input_dir.glob("*.wav"))
    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter if step == "training" else None,
        slide_frames=10 if step == "training" else 1,
        step_ms=10,
    )
    out = output_dir / step / f"{step}_mmap"
    if out.exists():
        if any(out.iterdir()):
            return
        out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=str(out),
        sample_generator=spectrograms.spectrogram_generator(
            repeat=repeat if step == "training" else 1
        ),
        batch_size=100,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="extracted speech_commands directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--classes", default="yes,no,up,down,left,right,stop,go,speech")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--background-only", action="store_true")
    parser.add_argument("--background-probability", type=float, default=0.75)
    parser.add_argument("--background-min-snr-db", type=float, default=-5)
    parser.add_argument("--background-max-snr-db", type=float, default=15)
    parser.add_argument("--background-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.background_probability <= 1:
        parser.error("--background-probability must be between 0 and 1")
    if args.background_min_snr_db > args.background_max_snr_db:
        parser.error("--background-min-snr-db must not exceed --background-max-snr-db")
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    validation = read_list(args.source / "validation_list.txt")
    testing = read_list(args.source / "testing_list.txt")
    split_root = args.output / "split"
    link_split(args.source, split_root, classes, validation, testing)
    probabilities = (
        {
            "AddColorNoise": 0.25,
            "AddBackgroundNoise": args.background_probability,
            "Gain": 1.0,
            "GainTransition": 0.25,
            "PitchShift": 0.25,
            "BandStopFilter": 0.10,
        }
        if args.augment_train
        else {}
    )
    if args.background_only:
        probabilities = (
            {"AddBackgroundNoise": args.background_probability}
            if args.augment_train
            else {}
        )
    background_paths = [str(args.background_dir)] if args.background_dir else []
    augmenter = Augmentation(
        augmentation_duration_s=1.0,
        augmentation_probabilities=probabilities,
        background_paths=background_paths,
        background_min_snr_db=args.background_min_snr_db,
        background_max_snr_db=args.background_max_snr_db,
    )
    if args.augment_train:
        print(f"training augmentation probabilities: {probabilities}")
        print(
            "background SNR range: "
            f"{args.background_min_snr_db}..{args.background_max_snr_db} dB"
        )
    for cls in classes:
        for split in ("training", "validation", "testing"):
            make_mmap(
                split_root / cls / split,
                args.output / "features" / cls,
                split,
                args.repeat,
                augmenter,
            )
    print(f"prepared {len(classes)} classes under {args.output / 'features'}")


if __name__ == "__main__":
    main()
