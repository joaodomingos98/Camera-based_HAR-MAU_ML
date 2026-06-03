"""
infer_new_videos.py
-------------------
Run inference with the trained S3D model on new single MKV videos.

Expected video naming convention:
    person91_handwaving.mkv
    person91_handclapping.mkv
    person91_boxing.mkv
    person91_walking.mkv
    person91_running.mkv
    person91_jogging.mkv

Usage:
    python infer_videos.py --videos_dir ./new_videos --model_path ./best_model.pth
    python infer_videos.py --videos_dir ./new_videos --model_path ./best_model.pth --num_clips 5
    python notebooks\infer_videos.py --videos_dir "Data_Collection\AxisRecordings\axis-ACCC8EE68D0A" --model_path "results\S3D\run_20260519_194434\best_model.pth" --num_clips 7
"""

import os
import re
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
#from ultralytics import YOLO
#detector = YOLO("yolov8n.pt")  # tiny model, fast on CPU


from torchvision.models.video import s3d, S3D_Weights


# ── Constants (must match training) ───────────────────────────────────────────
CLASS_NAMES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
CLIP_LEN = 32           # frames per clip — same as training


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Rebuild the S3D head and load saved weights."""
    weights = S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights)

    # Replace classifier head — identical to setup_model() in the notebook
    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model

# Preprocessing video as in KTH
def preprocess_frame_kth_style(frame_bgr, target_size=(160, 120)):
    """Detects person positon and resizes and convert to grayscale-as-RGB, matching KTH appearance."""
    """results = detector(frame_bgr, classes=[0], verbose=False)  # class 0 = person
    boxes = results[0].boxes
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        # Add padding so person isn't right at the edge
        pad = 20
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(frame_bgr.shape[1], x2+pad), min(frame_bgr.shape[0], y2+pad)
        frame_bgr = frame_bgr[y1:y2, x1:x2]
    """
    frame = cv2.resize(frame_bgr, target_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# ── Video → Tensor ────────────────────────────────────────────────────────────
def load_clip(video_path: str, clip_len: int, start_frame: int | None = None) -> torch.Tensor:
    """
    Read `clip_len` consecutive frames from a video file, starting at
    `start_frame`. If start_frame is None the centre clip is used (same
    as eval mode in the notebook's KTHVideoDataset).

    Returns a float32 tensor of shape (C, T, H, W) in [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < clip_len:
        cap.release()
        raise ValueError(
            f"Video too short ({total_frames} frames < clip_len={clip_len}): {video_path}"
        )

    if start_frame is None:
        start_frame = (total_frames - clip_len) // 2   # centre clip

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame_kth_style(frame, target_size=(160, 120))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # (C, H, W)

    cap.release()

    if len(frames) < clip_len:
        raise RuntimeError(
            f"Short read on {video_path}: got {len(frames)}/{clip_len} frames"
        )

    # Shape: (T, C, H, W) → (C, T, H, W), normalised to [0, 1]
    video = torch.stack(frames, dim=0).float() / 255.0          # (T, C, H, W)
    video = video.permute(1, 0, 2, 3)                           # (C, T, H, W)
    return video


def get_random_start_frames(video_path: str, clip_len: int, num_clips: int) -> list[int]:
    """Return `num_clips` distinct random start frames for temporal augmentation."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_start = total_frames - clip_len
    if max_start <= 0:
        return [0]

    starts = random.sample(range(0, max_start + 1), min(num_clips, max_start + 1))
    return starts


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_video(
    model: nn.Module,
    video_path: str,
    transform,
    device: torch.device,
    clip_len: int = CLIP_LEN,
    num_clips: int = 1,
) -> dict:
    """
    Run inference on a single video.

    If num_clips > 1, sample multiple clips from the video and average
    the softmax scores (temporal ensemble) for a more robust prediction.

    Returns a dict with:
        predicted_class  – string label
        confidence       – float in [0, 1]
        scores           – dict mapping each class name to its averaged score
    """
    if num_clips == 1:
        start_frames = [None]  # centre clip
    else:
        start_frames = get_random_start_frames(video_path, clip_len, num_clips)

    all_probs = []

    with torch.no_grad():
        for start in start_frames:
            clip = load_clip(video_path, clip_len, start_frame=start)  # (C, T, H, W)
            clip_t = clip.permute(1, 0, 2, 3)                          # (T, C, H, W)
            clip_transformed = transform(clip_t)                        # (C, T, H, W) — transform permutes internally
            batch = clip_transformed.unsqueeze(0).to(device)           # (1, C, T, H, W)

            # Now batch is beautifully shaped as (1, 3, 32, 224, 224)
            logits = model(batch)                               
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()  
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)     # (num_classes,)
    pred_idx = avg_probs.argmax().item()

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": avg_probs[pred_idx].item(),
        "scores": {cls: avg_probs[i].item() for i, cls in enumerate(CLASS_NAMES)},
    }


# ── Filename parsing ──────────────────────────────────────────────────────────
def parse_true_label(filename: str) -> str | None:
    """
    Extract the action label from filenames like 'person91_boxing.mkv'.
    Returns None if the pattern is not recognised.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # e.g. person91_handwaving  →  handwaving
    match = re.match(r'^person\d+_(.+)$', stem)
    if match:
        label = match.group(1).lower()
        return label if label in CLASS_NAMES else None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(
        description="Run S3D inference on new MKV videos."
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="./new_videos",
        help="Folder containing the new .mkv video files.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./best_model.pth",
        help="Path to the saved best_model.pth checkpoint.",
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=1,
        help=(
            "Number of clips to sample per video for temporal ensemble. "
            "1 = single centre clip (fast); >1 = averaged for better accuracy."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mkv", ".avi", ".mp4"],
        help="Video file extensions to look for.",
    )
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_model(args.model_path, num_classes=len(CLASS_NAMES), device=device)

    # After loading model, check the classifier weights
    w = model.classifier[1].weight.data  # shape: (6, 1024, 1, 1, 1)
    print("Classifier weight std per class:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {w[i].std().item():.6f}")
    # If walking's weights are very different from others → biased head
    # If all stds are near-zero → head never learned anything meaningful


    # Use the same preprocessing as during training
    weights = S3D_Weights.KINETICS400_V1
    transform = weights.transforms()

    # ── Gather videos ────────────────────────────────────────────────────────
    video_files = sorted([
        os.path.join(args.videos_dir, f)
        for f in os.listdir(args.videos_dir)
        if os.path.splitext(f)[1].lower() in args.extensions
    ])

    if not video_files:
        print(f"No video files found in: {args.videos_dir}")
        return

    print(f"Found {len(video_files)} video(s). Running inference...\n")
    print(f"{'File':<40} {'True Label':<15} {'Predicted':<15} {'Confidence':>10}  {'Correct?'}")
    print("-" * 95)

    correct = 0
    total_with_label = 0

    for video_path in video_files:
        filename = os.path.basename(video_path)
        true_label = parse_true_label(filename)

        try:
            result = predict_video(
                model, video_path, transform, device,
                clip_len=CLIP_LEN, num_clips=args.num_clips
            )
        except Exception as exc:
            print(f"{filename:<40} {'?':<15} ERROR: {exc}")
            continue

        pred = result["predicted_class"]
        conf = result["confidence"]

        is_correct = "—"
        if true_label is not None:
            total_with_label += 1
            if pred == true_label:
                correct += 1
                is_correct = "✓"
            else:
                is_correct = "✗"

        print(
            f"{filename:<40} {(true_label or '?'):<15} {pred:<15} "
            f"{conf:>9.1%}  {is_correct}"
        )

        # Print per-class scores at debug level
        scores_str = "  ".join(
            f"{cls}:{score:.2f}" for cls, score in result["scores"].items()
        )
        print(f"  scores → {scores_str}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    if total_with_label > 0:
        accuracy = correct / total_with_label
        print("-" * 95)
        print(
            f"\nAccuracy on labelled videos: {correct}/{total_with_label} "
            f"({accuracy:.1%})"
        )


if __name__ == "__main__":
    main()