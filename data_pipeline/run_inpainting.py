#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def find_pairs(root: Path):
    pairs = []
    for mask_mp4 in sorted(root.rglob("mask.mp4")):
        video_mp4 = mask_mp4.parent / "video.mp4"
        if video_mp4.exists():
            pairs.append((video_mp4, mask_mp4))
    return pairs


def run_diffueraser(diffueraser_root: Path, video_mp4: Path, mask_mp4: Path, output_dir: Path):
    ensure = output_dir
    ensure.mkdir(parents=True, exist_ok=True)
    infer_script = diffueraser_root / "infer" / "infer.py"
    if infer_script.exists():
        cmd = [
            "python", str(infer_script),
            "--video_path", str(video_mp4),
            "--mask_path", str(mask_mp4),
            "--output_dir", str(output_dir),
        ]
        subprocess.run(cmd, check=True)
        return
    demo_script = diffueraser_root / "demo.py"
    if demo_script.exists():
        cmd = [
            "python", str(demo_script),
            "--input_video", str(video_mp4),
            "--input_mask", str(mask_mp4),
            "--output_path", str(output_dir / "diffueraser_result.mp4"),
        ]
        subprocess.run(cmd, check=True)
        return
    raise FileNotFoundError(f"DiffuEraser entry script not found under {diffueraser_root}")


def write_erase_info(output_dir: Path, video_mp4: Path, mask_mp4: Path, fps: float, total_frames: int):
    info = {
        "video_path": str(video_mp4),
        "mask_path": str(mask_mp4),
        "fps": fps,
        "total_frames": total_frames,
        "output_video": str(output_dir / "diffueraser_result.mp4"),
    }
    with open(output_dir / "erase_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--diffueraser-root", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    mask_root = Path(args.mask_root)
    output_root = Path(args.output_root)
    diffueraser_root = Path(args.diffueraser_root)
    pairs = find_pairs(mask_root)
    if args.max_videos is not None:
        pairs = pairs[: args.max_videos]
    ok = 0
    for video_mp4, mask_mp4 in pairs:
        out_dir = output_root / video_mp4.parent.name
        try:
            run_diffueraser(diffueraser_root, video_mp4, mask_mp4, out_dir)
            write_erase_info(out_dir, video_mp4, mask_mp4, fps=10.0, total_frames=0)
            ok += 1
        except Exception as exc:
            print(f"[skip] {video_mp4.parent.name}: {exc}")
    print(f"done {ok}/{len(pairs)}")


if __name__ == "__main__":
    main()
