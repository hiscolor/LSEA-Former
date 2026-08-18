#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from common import MAX_MASK_AREA_RATIO, mask_area_ratio, mask_ratio_valid


def scan_mask_video(mask_mp4: Path):
    cap = cv2.VideoCapture(str(mask_mp4))
    max_ratio = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        max_ratio = max(max_ratio, mask_area_ratio(gray))
    cap.release()
    return max_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--remove-invalid", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    kept = 0
    removed = 0
    for mask_mp4 in tqdm(sorted(root.rglob("mask.mp4")), desc="filter"):
        max_ratio = scan_mask_video(mask_mp4)
        if mask_ratio_valid(max_ratio):
            kept += 1
        else:
            removed += 1
            print(f"[reject] {mask_mp4.parent.name} max={max_ratio:.4f} limit={MAX_MASK_AREA_RATIO:.4f}")
            if args.remove_invalid:
                for path in (mask_mp4.parent / "video.mp4", mask_mp4):
                    if path.exists():
                        path.unlink()
    print(f"kept={kept} removed={removed}")


if __name__ == "__main__":
    main()
