#!/usr/bin/env python3
"""Check microWakeWord ROC output and select a usable cutoff."""
import argparse
import re
from pathlib import Path

ROW = re.compile(r"Cutoff\s+([0-9.]+):\s+frr=([0-9.eE+-]+|nan),\s+faph=([0-9.eE+-]+|nan)")


def read_metrics(path: Path):
    text = path.read_text(encoding="utf-8")
    auc = re.search(r"AUC\s+([0-9.eE+-]+|nan)", text)
    rows = [tuple(map(float, m.groups())) for m in ROW.finditer(text) if "nan" not in m.group(0).lower()]
    if not auc or auc.group(1).lower() == "nan" or not rows:
        raise SystemExit("invalid metrics: AUC/faph is missing or nan")
    return float(auc.group(1)), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roc", type=Path)
    parser.add_argument("--max-faph", type=float, default=1.0)
    args = parser.parse_args()
    auc, rows = read_metrics(args.roc)
    eligible = [row for row in rows if row[2] <= args.max_faph]
    if not eligible:
        raise SystemExit(f"no cutoff meets max-faph={args.max_faph}")
    cutoff, frr, faph = min(eligible, key=lambda row: (row[1], -row[0]))
    print(f"auc={auc:.6f} cutoff={cutoff:.2f} frr={frr:.6f} faph={faph:.6f}")


if __name__ == "__main__":
    # ponytail: stdlib-only parser; add richer reporting only when needed.
    main()
