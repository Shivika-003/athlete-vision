"""Extract per-frame pose features with MediaPipe."""

import math
import cv2
import mediapipe as mp

from ai_engine.angle_utils import (
    calculate_all_angles,
    detect_dominant_side,
    get_avg_visibility,
    KalmanSmoother,
    MIN_VISIBILITY,
)
from ai_engine.pose_analyzer import classify_shot_type, detect_shot_phase

mp_pose = mp.solutions.pose


def _wrist_velocity(landmarks, prev_landmarks, side):
    if not prev_landmarks:
        return 0.0, 0.0
    if side == "right":
        w_c = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        w_p = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    else:
        w_c = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        w_p = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    vel = math.hypot(w_c.x - w_p.x, w_c.y - w_p.y)
    return vel, vel


def _stability(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_center_x = (l_hip.x + r_hip.x) / 2.0
    drift = abs(nose.x - hip_center_x)
    return max(0, 100 - drift * 300)


def extract_frame_series(video_path, frame_skip=1, max_frames=None):
    """
    Returns list of dicts: frame_idx, angles, phase, shot_type, wrist_vel,
    stability, visibility.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total / fps if fps else 0

    smoother = KalmanSmoother()
    frames = []
    prev_landmarks = None
    prev_phase = "idle"
    prev_wrist_vel = 0.0
    frame_idx = 0
    processed = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, _ = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            if max_frames and processed >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            landmarks = result.pose_landmarks

            if landmarks:
                lm = landmarks.landmark
                vis = get_avg_visibility(lm)
                if vis >= MIN_VISIBILITY:
                    side = detect_dominant_side(lm)
                    raw_angles = calculate_all_angles(lm, side)
                    angles = smoother.update(raw_angles) if raw_angles else None
                    wrist_vel, _ = _wrist_velocity(lm, prev_landmarks, side)
                    wrist_accel = wrist_vel - prev_wrist_vel
                    phase = detect_shot_phase(lm, wrist_vel, wrist_accel, prev_phase)
                    shot_type = classify_shot_type(lm)
                    stability = _stability(lm)

                    frames.append({
                        "frame_idx": frame_idx,
                        "angles": angles or {},
                        "phase": phase,
                        "shot_type": shot_type,
                        "wrist_vel": wrist_vel,
                        "stability": stability,
                        "visibility": vis,
                        "side": side,
                    })
                    prev_landmarks = lm
                    prev_phase = phase
                    prev_wrist_vel = wrist_vel

            frame_idx += frame_skip
            processed += 1

    cap.release()
    return {
        "fps": fps,
        "duration_sec": round(duration_sec, 1),
        "total_frames": total,
        "frames": frames,
    }
