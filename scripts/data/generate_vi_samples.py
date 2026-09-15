"""
07_generate_vi_samples.py
==========================
Generate Vietnamese keyword samples using Piper TTS (vi_VN-vivos-x_low).

Model: vi_VN-vivos-x_low
  - 65 speakers (VIVOSSPK* + VIVOSDEV*)
  - x_low quality, 16kHz
  - Covers both male and female voices

Output structure:
  data/generated/vi_samples/<keyword>/
      <keyword>_spk<id>_<idx>.wav

Usage:
  python scripts/07_generate_vi_samples.py
  python scripts/07_generate_vi_samples.py --keywords "buồn,vui,giận" --samples-per-speaker 3
  python scripts/07_generate_vi_samples.py --list-speakers
"""

import argparse
import os
import subprocess
import pathlib
import json
import sys

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
VOICE_DIR  = BASE_DIR / "models/piper-sample-generator/voices"
VOICE_ONNX = VOICE_DIR / "vi_VN-vivos-x_low.onnx"
OUTPUT_DIR = BASE_DIR / "data/generated/vi_samples"
# Dùng hai_venv_py310 vì có torch CUDA 12.1 hoạt động
# tflite-microcontroller có cuDNN 8 mismatch với torch nên không dùng
VENV_PYTHON = pathlib.Path("/home/ubuntu/data/hai_venv_py310/bin/python3")

# Vietnamese keywords to generate — chỉnh theo nhu cầu
DEFAULT_KEYWORDS = [
    "buồn",
    "vui",
    "giận",
    "dừng",
    "bật",
    "tắt",
    "mở",
    "đóng",
]

# 65 speakers từ VIVOS corpus
ALL_SPEAKERS = list(range(65))  # 0 → 64

# ─────────────────────────────────────────────────────────────────────────────

def list_speakers():
    json_path = VOICE_DIR / "vi_VN-vivos-x_low.onnx.json"
    if not json_path.exists():
        print("Cannot find .onnx.json file")
        return
    with open(json_path) as f:
        config = json.load(f)
    speaker_map = config.get("speaker_id_map", {})
    print(f"Total speakers: {len(speaker_map)}")
    for name, idx in sorted(speaker_map.items(), key=lambda x: x[1]):
        print(f"  [{idx:2d}] {name}")


def generate_for_keyword(keyword: str, speaker_ids: list, samples_per_speaker: int, length_scales: list, gpu_id: str):
    """
    Dùng piper Python API (ONNX) để generate WAV cho từng speaker + speed scale.
    piper_sample_generator chỉ hỗ trợ .pt model (English only), không dùng được với .onnx.
    """
    import wave as wave_module
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError:
        print("[ERROR] piper không được cài. Chạy: pip install piper-tts")
        return 0

    out_dir = OUTPUT_DIR / keyword
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load voice model 1 lần duy nhất (ONNX, CPU — piper không dùng CUDA)
    voice = PiperVoice.load(str(VOICE_ONNX), use_cuda=False)
    sample_rate = voice.config.sample_rate

    total = 0
    for spk_id in speaker_ids:
        for scale in length_scales:
            scale_str = str(scale).replace(".", "")
            existing = list(out_dir.glob(f"{keyword}_spk{spk_id:02d}_s{scale_str}_*.wav"))
            if len(existing) >= samples_per_speaker:
                total += len(existing)
                continue

            for i in range(samples_per_speaker):
                dst = out_dir / f"{keyword}_spk{spk_id:02d}_s{scale_str}_{i:03d}.wav"
                if dst.exists():
                    total += 1
                    continue
                try:
                    syn_cfg = SynthesisConfig(
                        speaker_id=spk_id,
                        length_scale=float(scale),
                    )
                    with wave_module.open(str(dst), "w") as wav_file:
                        voice.synthesize_wav(keyword, wav_file, syn_config=syn_cfg, set_wav_format=True)
                    total += 1
                except Exception as e:
                    print(f"  [WARN] spk={spk_id}, scale={scale}, i={i}: {e}")
                    dst.unlink(missing_ok=True)

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese keyword samples using Piper TTS (vi_VN-vivos-x_low)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated list of Vietnamese keywords to generate"
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default="all",
        help="Comma-separated speaker IDs to use, or 'all' for all 65 speakers"
    )
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=2,
        help="Number of samples per speaker per speed variant"
    )
    parser.add_argument(
        "--length-scales",
        type=str,
        default="0.85,1.0,1.15",
        help="Comma-separated speaking speed scales (< 1.0 = faster, > 1.0 = slower)"
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print all speaker IDs and exit"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="2",
        help="GPU ID(s) to use, e.g. '0' or '0,1' (passed to CUDA_VISIBLE_DEVICES)"
    )
    args = parser.parse_args()

    if args.list_speakers:
        list_speakers()
        return

    # Validate model
    if not VOICE_ONNX.exists():
        print(f"[ERROR] Voice model not found: {VOICE_ONNX}")
        print("  Run: wget -O models/piper-sample-generator/voices/vi_VN-vivos-x_low.onnx \\")
        print("       https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx")
        return

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    length_scales = [float(s.strip()) for s in args.length_scales.split(",")]

    if args.speakers == "all":
        speaker_ids = ALL_SPEAKERS
    else:
        speaker_ids = [int(s.strip()) for s in args.speakers.split(",")]

    print(f"[INFO] Voice model : {VOICE_ONNX.name}")
    print(f"[INFO] Keywords    : {keywords}")
    print(f"[INFO] Speakers    : {len(speaker_ids)} speakers ({speaker_ids[0]}→{speaker_ids[-1]})")
    print(f"[INFO] Speed scales: {length_scales}")
    print(f"[INFO] Samples/spk : {args.samples_per_speaker}")
    print(f"[INFO] GPU         : {args.gpu}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")
    estimated = len(keywords) * len(speaker_ids) * len(length_scales) * args.samples_per_speaker
    print(f"[INFO] Estimated total samples: ~{estimated}")
    print()

    for keyword in keywords:
        print(f"[INFO] Generating: '{keyword}'...")
        total = generate_for_keyword(keyword, speaker_ids, args.samples_per_speaker, length_scales, args.gpu)
        files = list((OUTPUT_DIR / keyword).glob("*.wav"))
        print(f"  → {len(files)} WAV files in data/generated/vi_samples/{keyword}/")

    print()
    print("[INFO] Done! Next steps:")
    print("  1. Review samples: ls data/generated/vi_samples/")
    print("  2. Build dataset: adapt scripts/01_build_dataset.py for Vietnamese keywords")
    print("  3. Or use as additional training data for negative samples")


if __name__ == "__main__":
    main()
