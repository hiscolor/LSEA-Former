import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

FPS = 10
OUT_W = 1280
OUT_H = 720
OUT_H_DOTA = 960
YOLO_MODEL = "yolov8x.pt"
SAM2_WEIGHTS = "sam2.1_b.pt"
CONF_THRES = 0.06
IOU_THRES_NMS = 0.50
IMGSZ = 1920
YOLO_CLASSES = [0, 1, 2, 3, 5, 6, 7]
CAR_CLASS_NAMES = {"car", "bus", "truck"}
IOU_MATCH_THR = 0.50
LINK_IOU_THR = 0.30
MAX_MASK_AREA_RATIO = 1.0 / 3.0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clip_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def load_frame(frames_dir: Path, idx: int):
    stem = f"{idx:06d}"
    for ext in (".jpg", ".png", ".jpeg"):
        path = frames_dir / f"{stem}{ext}"
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    return None


def count_frames(frames_dir: Path):
    n = 0
    while load_frame(frames_dir, n) is not None:
        n += 1
    return n


def read_video_json(vid_dir: Path):
    preferred = vid_dir / f"{vid_dir.name}.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf-8"))
    for candidate in vid_dir.glob("*.json"):
        return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def list_dota_videos(root: Path):
    videos = []
    for accident_dir in sorted(root.iterdir()):
        if not accident_dir.is_dir():
            continue
        for video_dir in sorted(accident_dir.iterdir()):
            if video_dir.is_dir() and (video_dir / "frames").exists():
                videos.append(video_dir)
    return videos


def is_car_from_obj(obj):
    for key in ("category", "category_name", "cls_name", "label", "name", "type"):
        if key in obj and obj[key] is not None:
            value = str(obj[key]).lower()
            for name in CAR_CLASS_NAMES:
                if name in value:
                    return True
    return False


def load_gt_boxes_per_frame(json_meta, n_frames, width, height):
    gt_noncar = [[] for _ in range(n_frames)]
    gt_car = [[] for _ in range(n_frames)]

    def frame_objects(frame_idx):
        labels = json_meta.get("labels", [])
        if not labels:
            return []
        if isinstance(labels[0], dict) and "frame_id" in labels[0]:
            for item in labels:
                if int(item.get("frame_id", -1)) == frame_idx:
                    return item.get("objects", []) or []
            return []
        if 0 <= frame_idx < len(labels):
            return labels[frame_idx].get("objects", []) or []
        return []

    for frame_idx in range(n_frames):
        for obj in frame_objects(frame_idx):
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue
            box = clip_box(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if is_car_from_obj(obj):
                gt_car[frame_idx].append(box)
            else:
                gt_noncar[frame_idx].append(box)
    return gt_noncar, gt_car


def make_mp4(frames_dir: Path, pattern: str, out_mp4: Path, fps: int, out_w: int, out_h: int, start_number: int = 0, neighbor: bool = False):
    scale_flags = ":flags=neighbor" if neighbor else ""
    vf = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease{scale_flags},"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-start_number", str(start_number),
        "-i", str(frames_dir / pattern), "-vf", vf,
        "-pix_fmt", "yuv420p", "-crf", "18", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


class Sam2Segmentor:
    def __init__(self, weights=SAM2_WEIGHTS):
        self.model = SAM(weights)

    def segment_union(self, image_bgr, boxes):
        height, width = image_bgr.shape[:2]
        if not boxes:
            return np.zeros((height, width), dtype=np.uint8)
        clipped = [clip_box(x1, y1, x2, y2, width, height) for x1, y1, x2, y2 in boxes]
        try:
            result = self.model(
                image_bgr,
                bboxes=[list(map(float, box)) for box in clipped],
                verbose=False,
            )
            if not result or result[0].masks is None or result[0].masks.data is None:
                return self._rect_union(width, height, clipped)
            masks = result[0].masks.data
            if hasattr(masks, "cpu"):
                masks = masks.float().cpu().numpy()
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]
            union = (masks.max(axis=0) > 0.5).astype(np.uint8) * 255
            if union.shape != (height, width):
                union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
            return union
        except Exception:
            return self._rect_union(width, height, clipped)

    @staticmethod
    def _rect_union(width, height, boxes):
        union = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            union[y1:y2 + 1, x1:x2 + 1] = 255
        return union


def collect_tracks(yolo, frames_dir, n_frames, width, height, device):
    per_frame_tracks = [[] for _ in range(n_frames)]
    track_cls = {}
    results = yolo.track(
        source=str(frames_dir),
        stream=True,
        tracker="botsort.yaml",
        conf=CONF_THRES,
        iou=IOU_THRES_NMS,
        classes=YOLO_CLASSES,
        imgsz=IMGSZ,
        persist=True,
        device=device,
        verbose=False,
    )
    names = yolo.model.names if hasattr(yolo, "model") else yolo.names
    for frame_idx, result in enumerate(results):
        if frame_idx >= n_frames or result is None or result.boxes is None or result.boxes.shape[0] == 0:
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
            per_frame_tracks[frame_idx].append((int(track_id), box, int(cls_id), float(conf)))
            track_cls[int(track_id)] = int(cls_id)
    return per_frame_tracks, track_cls, names


def select_seed_tracks(per_frame_tracks, gt_noncar, gt_car, iou_thr=IOU_MATCH_THR):
    seed_tracks = set()
    n_frames = len(per_frame_tracks)
    for frame_idx in range(n_frames):
        gt_boxes = gt_noncar[frame_idx] + gt_car[frame_idx]
        if not gt_boxes or not per_frame_tracks[frame_idx]:
            continue
        for gt_box in gt_boxes:
            best_track = None
            best_iou = 0.0
            for track_id, box, _, _ in per_frame_tracks[frame_idx]:
                overlap = iou_xyxy(box, gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track_id
            if best_track is not None and best_iou >= iou_thr:
                seed_tracks.add(best_track)
    return seed_tracks


def stitch_tracks_by_iou(per_frame_tracks, seed_tracks, link_iou_thr=LINK_IOU_THR):
    neighbors = {}
    for frame_idx in range(len(per_frame_tracks) - 1):
        current = per_frame_tracks[frame_idx]
        nxt = per_frame_tracks[frame_idx + 1]
        for track_id1, box1, _, _ in current:
            for track_id2, box2, _, _ in nxt:
                if iou_xyxy(box1, box2) >= link_iou_thr:
                    neighbors.setdefault(track_id1, set()).add(track_id2)
                    neighbors.setdefault(track_id2, set()).add(track_id1)
    keep = set()
    stack = list(seed_tracks)
    while stack:
        track_id = stack.pop()
        if track_id in keep:
            continue
        keep.add(track_id)
        for neighbor in neighbors.get(track_id, []):
            if neighbor not in keep:
                stack.append(neighbor)
    return keep


def is_car_detection(cls_id, names):
    if cls_id is None:
        return False
    if int(cls_id) in range(len(names)):
        return str(names[int(cls_id)]).lower() in CAR_CLASS_NAMES
    return False


def boxes_for_frame(frame_idx, keep_tracks, per_frame_tracks, gt_noncar, gt_car, track_cls, names):
    boxes_noncar = []
    boxes_car = []
    for track_id, box, cls_id, _ in per_frame_tracks[frame_idx]:
        if track_id not in keep_tracks:
            continue
        if is_car_detection(cls_id, names):
            boxes_car.append(box)
        else:
            boxes_noncar.append(box)
    boxes_noncar.extend(gt_noncar[frame_idx])
    boxes_car.extend(gt_car[frame_idx])
    return boxes_noncar, boxes_car


def mask_area_ratio(mask):
    return float((mask > 127).sum()) / float(mask.size)


def mask_ratio_valid(max_ratio, max_allowed=MAX_MASK_AREA_RATIO):
    return max_ratio <= max_allowed
