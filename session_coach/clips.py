"""Cut short highlight clips from practice video."""

import os
import cv2


def extract_clip(video_path, frame_idx, fps, output_path, padding_sec=1.5):
    """Save a short mp4 around frame_idx."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(0, int(frame_idx - padding_sec * fps))
    end = int(frame_idx + padding_sec * fps)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        idx += 1

    out.release()
    cap.release()
    return os.path.exists(output_path)


def build_highlight_clips(video_path, highlights, output_dir, fps, prefix):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, shot in enumerate(highlights):
        name = f"{prefix}_{i}_{shot.get('shot_type', 'shot')}.mp4"
        out_path = os.path.join(output_dir, name)
        ok = extract_clip(video_path, shot["frame_idx"], fps, out_path)
        if ok:
            paths.append({
                "file": name,
                "shot_type": shot.get("shot_type", "drive"),
                "timestamp": shot.get("timestamp", "0:00"),
                "stability": round(shot.get("stability", 0), 1),
                "score": round(shot.get("clip_score", 0), 1),
            })
    return paths
