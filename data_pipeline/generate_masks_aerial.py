#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from common import (
    MAX_MASK_AREA_RATIO,
    OUT_H,
    OUT_W,
    Sam2Segmentor,
    YOLO_MODEL,
    clip_box,
    ensure_dir,
    mask_area_ratio,
)

CONF_THRES = 0.10
IOU_THRES_NMS = 0.45
IMGSZ = 1280
TARGET_CLASSES = [0, 1, 2, 3, 5, 6, 7]
MIN_TRACK_FRAMES = 5
MIN_MASK_RATIO = 0.15
MAX_MASK_RATIO = 0.60
MIN_BBOX_AREA_RATIO = 0.0005
MAX_BBOX_AREA_RATIO = 0.40
SMALL_TARGET_AREA = 0.01
MEDIUM_TARGET_AREA = 0.05
TARGETS_FOR_SMALL = (4, 10)
TARGETS_FOR_MEDIUM = (2, 6)
TARGETS_FOR_LARGE = (1, 3)


def extract_frames(video_path: Path, output_dir: Path):
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(output_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx, fps


def make_mp4(frames_dir: Path, pattern: str, out_mp4: Path, fps: float, start_number: int = 0):
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-start_number", str(start_number),
        "-i", str(frames_dir / pattern), "-pix_fmt", "yuv420p", "-crf", "18", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def bbox_area_ratio(box, width, height):
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / float(width * height)


def run_tracking(yolo, frames_dir: Path, width: int, height: int, device):
    track_data = defaultdict(list)
    results = yolo.track(
        source=str(frames_dir),
        stream=True,
        tracker="botsort.yaml",
        conf=CONF_THRES,
        iou=IOU_THRES_NMS,
        classes=TARGET_CLASSES,
        imgsz=IMGSZ,
        persist=True,
        device=device,
        verbose=False,
    )
    for frame_idx, result in enumerate(results):
        if result is None or result.boxes is None or result.boxes.shape[0] == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        ids = result.boxes.id
        ids = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1, dtype=int)
        for bbox, conf, cls_id, track_id in zip(boxes, confs, classes, ids):
            if track_id < 0:
                continue
            box = clip_box(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), width, height)
            area_ratio = bbox_area_ratio(box, width, height)
            if area_ratio < MIN_BBOX_AREA_RATIO or area_ratio > MAX_BBOX_AREA_RATIO:
                continue
            track_data[int(track_id)].append((frame_idx, box, int(cls_id), float(conf)))
    return track_data


def compute_avg_target_size(track_data, width, height):
    areas = []
    for detections in track_data.values():
        for _, box, _, _ in detections:
            areas.append(bbox_area_ratio(box, width, height))
    return float(np.mean(areas)) if areas else 0.0


def get_target_count_range(avg_area):
    if avg_area < SMALL_TARGET_AREA:
        return TARGETS_FOR_SMALL
    if avg_area < MEDIUM_TARGET_AREA:
        return TARGETS_FOR_MEDIUM
    return TARGETS_FOR_LARGE


def select_targets_and_interval(track_data, n_frames, fps, width, height, rng):
    valid = {tid: items for tid, items in track_data.items() if len(items) >= MIN_TRACK_FRAMES}
    if not valid and track_data:
        sorted_tracks = sorted(track_data.items(), key=lambda item: len(item[1]), reverse=True)
        valid = {tid: items for tid, items in sorted_tracks[: min(10, len(sorted_tracks))]}
    if not valid:
        return [], 0, 0
    avg_area = compute_avg_target_size(valid, width, height)
    min_targets, max_targets = get_target_count_range(avg_area)
    track_ids = list(valid.keys())
    n_select = rng.randint(min(min_targets, len(track_ids)), min(max_targets, len(track_ids)))
    selected = sorted(track_ids, key=lambda tid: len(valid[tid]), reverse=True)[:n_select]
    all_frames: Set[int] = set()
    for track_id in selected:
        all_frames.update(frame_idx for frame_idx, _, _, _ in valid[track_id])
    if not all_frames:
        return [], 0, 0
    sorted_frames = sorted(all_frames)
    min_span = max(1, int(round(n_frames * MIN_MASK_RATIO)))
    max_span = max(min_span, int(round(n_frames * MAX_MASK_RATIO)))
    span = rng.randint(min_span, min(max_span, len(sorted_frames)))
    start_idx = rng.randint(0, max(0, len(sorted_frames) - span))
    selected_frames = sorted_frames[start_idx:start_idx + span]
    return selected, min(selected_frames), max(selected_frames)


def list_videos(root: Path):
    return sorted(root.rglob("*.mp4"))


def process_video(video_path: Path, output_dir: Path, yolo, segmentor, rng, device):
    name = video_path.stem.replace(" ", "_")
    out_dir = output_dir / name
    frames_dir = out_dir / "frames"
    mask_dir = out_dir / "mask_frames"
    ensure_dir(frames_dir)
    ensure_dir(mask_dir)
    n_frames, fps = extract_frames(video_path, frames_dir)
    if n_frames == 0:
        return None
    track_data = run_tracking(yolo, frames_dir, OUT_W, OUT_H, device)
    if not track_data:
        return None
    selected_tids, start_frame, end_frame = select_targets_and_interval(
        track_data, n_frames, fps, OUT_W, OUT_H, rng
    )
    if not selected_tids:
        return None
    frame_to_boxes: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for track_id in selected_tids:
        for frame_idx, box, _, _ in track_data[track_id]:
            if start_frame <= frame_idx <= end_frame:
                frame_to_boxes[frame_idx].append(box)
    max_ratio = 0.0
    has_mask = False
    for frame_idx in range(n_frames):
        image = cv2.imread(str(frames_dir / f"{frame_idx:06d}.jpg"))
        if image is None:
            continue
        if frame_idx in frame_to_boxes:
            mask = segmentor.segment_union(image, frame_to_boxes[frame_idx])
        else:
            mask = np.zeros((OUT_H, OUT_W), dtype=np.uint8)
        if mask.any():
            has_mask = True
        max_ratio = max(max_ratio, mask_area_ratio(mask))
        cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)
    if not has_mask or max_ratio > MAX_MASK_AREA_RATIO:
        return None
    video_mp4 = out_dir / "video.mp4"
    mask_mp4 = out_dir / "mask.mp4"
    make_mp4(frames_dir, "%06d.jpg", video_mp4, fps)
    make_mp4(mask_dir, "%06d.png", mask_mp4, fps)
    meta = {
        "source_video": str(video_path),
        "selected_track_ids": [int(t) for t in selected_tids],
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "max_mask_ratio": float(max_ratio),
        "video_mp4": str(video_mp4),
        "mask_mp4": str(mask_mp4),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--sam-weights", type=str, default="sam2.1_b.pt")
    parser.add_argument("--yolo-weights", type=str, default=YOLO_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    videos = list_videos(input_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.yolo_weights)
    segmentor = Sam2Segmentor(args.sam_weights)
    rng = random.Random(args.seed)
    ok = 0
    for video_path in tqdm(videos, desc="aerial_masks"):
        meta = process_video(video_path, output_root, yolo, segmentor, rng, device)
        if meta is not None:
            ok += 1
        else:
            print(f"[skip] {video_path.name}")
    print(f"done {ok}/{len(videos)}")


if __name__ == "__main__":
    main()
