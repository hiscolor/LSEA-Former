#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from common import (
    FPS,
    OUT_H_DOTA,
    OUT_W,
    Sam2Segmentor,
    YOLO_MODEL,
    boxes_for_frame,
    collect_tracks,
    count_frames,
    ensure_dir,
    list_dota_videos,
    load_frame,
    load_gt_boxes_per_frame,
    make_mp4,
    mask_area_ratio,
    mask_ratio_valid,
    read_video_json,
    select_seed_tracks,
    stitch_tracks_by_iou,
)


def process_video(video_dir, yolo, segmentor, device, overwrite=False):
    json_meta = read_video_json(video_dir)
    if json_meta is None:
        return False, "missing_json"
    frames_dir = video_dir / "frames"
    n_frames = count_frames(frames_dir)
    if n_frames == 0:
        return False, "empty_frames"
    sample = load_frame(frames_dir, 0)
    height, width = sample.shape[:2]
    gt_noncar, gt_car = load_gt_boxes_per_frame(json_meta, n_frames, width, height)
    per_frame_tracks, track_cls, names = collect_tracks(yolo, frames_dir, n_frames, width, height, device)
    seed_tracks = select_seed_tracks(per_frame_tracks, gt_noncar, gt_car)
    if not seed_tracks:
        return False, "no_seed_tracks"
    keep_tracks = stitch_tracks_by_iou(per_frame_tracks, seed_tracks)
    frame_dir = video_dir / "frame_export"
    mask_dir = video_dir / "mask_frames"
    ensure_dir(frame_dir)
    ensure_dir(mask_dir)
    max_ratio = 0.0
    min_ratio = 1.0
    has_mask = False
    per_frame_noncar = []
    per_frame_car = []
    for frame_idx in range(n_frames):
        image = load_frame(frames_dir, frame_idx)
        if image is None:
            per_frame_noncar.append([])
            per_frame_car.append([])
            continue
        boxes_noncar, boxes_car = boxes_for_frame(
            frame_idx, keep_tracks, per_frame_tracks, gt_noncar, gt_car, track_cls, names
        )
        per_frame_noncar.append(boxes_noncar)
        per_frame_car.append(boxes_car)
        if boxes_noncar:
            mask = segmentor.segment_union(image, boxes_noncar)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        if mask.any():
            has_mask = True
        ratio = mask_area_ratio(mask)
        max_ratio = max(max_ratio, ratio)
        if ratio > 0:
            min_ratio = min(min_ratio, ratio)
        cv2.imwrite(str(frame_dir / f"{frame_idx:06d}.jpg"), image)
        cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)
    if not has_mask:
        for frame_idx in range(n_frames):
            image = load_frame(frames_dir, frame_idx)
            if image is None:
                continue
            boxes = per_frame_noncar[frame_idx] + per_frame_car[frame_idx]
            mask = segmentor.segment_union(image, boxes) if boxes else np.zeros((height, width), dtype=np.uint8)
            ratio = mask_area_ratio(mask)
            max_ratio = max(max_ratio, ratio)
            if ratio > 0:
                min_ratio = min(min_ratio, ratio)
            cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)
    if not mask_ratio_valid(max_ratio, min_ratio if has_mask else min_ratio if min_ratio < 1.0 else 0.0):
        return False, f"mask_ratio_{min_ratio:.4f}_{max_ratio:.4f}"
    video_mp4 = video_dir / "video.mp4"
    mask_mp4 = video_dir / "mask.mp4"
    if overwrite or not video_mp4.exists():
        make_mp4(frame_dir, "%06d.jpg", video_mp4, FPS, OUT_W, OUT_H_DOTA, neighbor=False)
    if overwrite or not mask_mp4.exists():
        make_mp4(mask_dir, "%06d.png", mask_mp4, FPS, OUT_W, OUT_H_DOTA, neighbor=True)
    return True, f"tracks={len(keep_tracks)} ratio={max_ratio:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--sam-weights", type=str, default="sam2.1_b.pt")
    parser.add_argument("--yolo-weights", type=str, default=YOLO_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.input_root)
    videos = list_dota_videos(root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.yolo_weights)
    segmentor = Sam2Segmentor(args.sam_weights)
    ok = 0
    for video_dir in tqdm(videos, desc="dota_masks"):
        success, info = process_video(video_dir, yolo, segmentor, device, overwrite=args.overwrite)
        if success:
            ok += 1
        else:
            print(f"[skip] {video_dir.name}: {info}")
    print(f"done {ok}/{len(videos)}")


if __name__ == "__main__":
    main()
