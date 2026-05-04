"""Preprocessing: GRID .mpg + .align → grayscale mouth-ROI .npz clips."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PAD = 10
_TARGET_W = 140
_TARGET_H = 46
_TARGET_FRAMES = 75


def load_align(align_path: str | Path) -> str:
    """Parse a GRID .align file, returning words (sil/sp tokens skipped)."""
    words = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] not in ("sil", "sp"):
                words.append(parts[2])
    return " ".join(words)


def extract_mouth(
    video_path: str | Path,
    detector,
    predictor,
    target_frames: int = _TARGET_FRAMES,
) -> np.ndarray:
    """Extract resampled grayscale mouth-ROI frames from a video clip.

    Uses dlib 68-point landmarks (pts 48-67) to locate the mouth, crops with
    padding=10, resizes to (H=46, W=140), and resamples to exactly target_frames.
    Returns uint8 array of shape (target_frames, H, W).
    """
    cap = cv2.VideoCapture(str(video_path))
    raw_frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            raw_frames.append(np.zeros((_TARGET_H, _TARGET_W), dtype=np.uint8))
            continue

        shape = predictor(gray, faces[0])
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        x, y, w, h = cv2.boundingRect(pts)
        x = max(0, x - _PAD)
        y = max(0, y - _PAD)
        w = w + 2 * _PAD
        h = h + 2 * _PAD
        mouth = gray[y : y + h, x : x + w]
        mouth = cv2.resize(mouth, (_TARGET_W, _TARGET_H))
        raw_frames.append(mouth)

    cap.release()

    if not raw_frames:
        return np.zeros((target_frames, _TARGET_H, _TARGET_W), dtype=np.uint8)

    if len(raw_frames) >= target_frames:
        indices = np.linspace(0, len(raw_frames) - 1, target_frames).astype(int)
        raw_frames = [raw_frames[i] for i in indices]
    else:
        while len(raw_frames) < target_frames:
            raw_frames.append(raw_frames[-1])

    return np.array(raw_frames)  # (T, H, W) uint8


def normalize(frames: np.ndarray) -> np.ndarray:
    """Clip-level mean/std normalization. Returns float32 array, same shape."""
    frames = frames.astype(np.float32) / 255.0
    mean = frames.mean()
    std = frames.std() + 1e-6
    return (frames - mean) / std


def process_clip(
    video_path: str | Path,
    align_path: str | Path,
    detector,
    predictor,
    target_frames: int = _TARGET_FRAMES,
) -> tuple[np.ndarray, str]:
    """Process one clip: extract → normalize → load label string."""
    frames = extract_mouth(video_path, detector, predictor, target_frames)
    frames = normalize(frames)
    label = load_align(align_path).lower().strip()
    return frames, label


def process_speaker(
    video_dir: Path,
    align_dir: Path,
    output_dir: Path,
    speaker: str,
    detector,
    predictor,
    target_frames: int = _TARGET_FRAMES,
) -> tuple[int, int]:
    """Process all clips for one speaker, writing speaker/<speaker>_<stem>.npz.

    Returns (n_saved, n_skipped).
    """
    spk_video_dir = video_dir / speaker
    spk_align_dir = align_dir / speaker
    spk_out_dir = output_dir / speaker
    spk_out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for align_path in sorted(spk_align_dir.glob("*.align")):
        stem = align_path.stem
        video_path = spk_video_dir / f"{stem}.mpg"
        if not video_path.exists():
            logger.warning("Missing video %s — skipped", video_path)
            n_skip += 1
            continue

        out_path = spk_out_dir / f"{speaker}_{stem}.npz"
        if out_path.exists():
            n_ok += 1
            continue

        try:
            frames, label = process_clip(
                video_path, align_path, detector, predictor, target_frames
            )
            np.savez_compressed(out_path, frames=frames, label=label)
            n_ok += 1
        except Exception as exc:
            logger.warning("Failed %s: %s — skipped", stem, exc)
            n_skip += 1

    return n_ok, n_skip
