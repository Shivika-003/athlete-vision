"""
Athlete Vision 2.0 — Pose Analyzer (Rewritten)
=================================================
3D pose estimation with confidence filtering, EMA smoothing,
shot phase detection, and structured angle output.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import math
import collections
import json

from ai_engine.angle_utils import (
    calculate_angle_3d, get_landmark_3d, get_landmark_2d,
    get_avg_visibility, detect_dominant_side, calculate_all_angles,
    KalmanSmoother, MIN_VISIBILITY
)
from ai_engine.reference_builder import (
    get_active_reference_player, get_reference_angles
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# LAZY YOLO LOADING: Only load YOLO when actually needed during processing
# This prevents heavy model loading from slowing down app startup
_yolo_model_cache = None
_yolo_loaded = False

def _get_yolo_model():
    """Lazy-load YOLO model only when first needed."""
    global _yolo_model_cache, _yolo_loaded
    if _yolo_loaded:
        return _yolo_model_cache
    _yolo_loaded = True
    try:
        from ultralytics import YOLO
        if os.path.exists('badminton_yolo.pt'):
            _yolo_model_cache = YOLO('badminton_yolo.pt')
            print("[PoseAnalyzer] Custom YOLO model loaded (lazy).")
        else:
            print("[PoseAnalyzer] No custom YOLO model found, skipping shuttlecock tracking.")
            _yolo_model_cache = None
    except Exception as e:
        print(f"[PoseAnalyzer] YOLO unavailable: {e}")
        _yolo_model_cache = None
    return _yolo_model_cache


# =====================================================================
# SHOT PHASE DETECTION
# =====================================================================

def detect_shot_phase(landmarks, wrist_vel, wrist_accel, prev_phase):
    """Detect the current shot phase based on body position and motion.
    
    Phases:
        'idle'         - No shot in progress
        'preparation'  - Arm drawing back, loading
        'swing'        - Arm moving upward/forward, accelerating
        'contact'      - Peak velocity / deceleration spike = shuttle hit
        'follow_through' - Arm decelerating after contact
    
    Args:
        landmarks: MediaPipe pose landmarks
        wrist_vel: Current wrist velocity (normalized)
        wrist_accel: Current wrist acceleration (velocity change)
        prev_phase: Previous frame's phase
        
    Returns:
        Current phase string
    """
    side = detect_dominant_side(landmarks)
    
    if side == 'right':
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    else:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    
    wrist_above_shoulder = wrist.y < shoulder.y
    wrist_above_head = wrist.y < landmarks[mp_pose.PoseLandmark.NOSE.value].y
    elbow_above_shoulder = elbow.y < shoulder.y
    hands_at_rest = wrist.y > hip.y * 0.9
    
    # Velocity thresholds
    MOTION_THRESHOLD = 0.02
    SWING_THRESHOLD = 0.04
    CONTACT_DECEL_THRESHOLD = -0.015
    
    # Phase state machine
    if wrist_vel < MOTION_THRESHOLD and hands_at_rest:
        return 'idle'
    
    if prev_phase == 'idle' and wrist_vel > MOTION_THRESHOLD and not wrist_above_shoulder:
        return 'preparation'
    
    if prev_phase in ('idle', 'preparation') and wrist_above_shoulder and wrist_vel > SWING_THRESHOLD:
        return 'swing'
    
    if prev_phase == 'swing' and wrist_accel < CONTACT_DECEL_THRESHOLD and wrist_vel > MOTION_THRESHOLD:
        return 'contact'
    
    if prev_phase == 'contact':
        return 'follow_through'
    
    if prev_phase == 'follow_through' and (wrist_vel < MOTION_THRESHOLD or hands_at_rest):
        return 'idle'
    
    # Maintain current phase if no transition triggered
    if prev_phase == 'follow_through':
        return 'follow_through'
    if prev_phase == 'preparation' and wrist_vel > MOTION_THRESHOLD:
        return 'preparation' if not wrist_above_shoulder else 'swing'
    
    return prev_phase if prev_phase else 'idle'


# =====================================================================
# SHOT TYPE CLASSIFICATION (Improved with angles)
# =====================================================================

def classify_shot_type(landmarks):
    """Classify badminton shot type based on body posture and limb positions.
    
    Uses relative positions of wrists, elbows, shoulders, and hips plus
    joint angles to determine the shot type.
    
    Returns: 'smash', 'clear', 'drive', 'drop', or 'net'
    """
    side = detect_dominant_side(landmarks)
    
    if side == 'right':
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    else:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    wrist_above_head = wrist.y < nose.y
    wrist_above_shoulder = wrist.y < shoulder.y
    elbow_above_shoulder = elbow.y < shoulder.y
    wrist_below_hip = wrist.y > hip.y
    wrist_near_hip = abs(wrist.y - hip.y) < 0.1
    
    # Calculate elbow angle for additional signal
    elbow_pt = get_landmark_3d(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value if side == 'right' else mp_pose.PoseLandmark.LEFT_ELBOW.value, 0.3)
    shoulder_pt = get_landmark_3d(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value if side == 'right' else mp_pose.PoseLandmark.LEFT_SHOULDER.value, 0.3)
    wrist_pt = get_landmark_3d(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value if side == 'right' else mp_pose.PoseLandmark.LEFT_WRIST.value, 0.3)
    
    elbow_angle = 0
    if elbow_pt and shoulder_pt and wrist_pt:
        elbow_angle = calculate_angle_3d(shoulder_pt, elbow_pt, wrist_pt)
    
    # Knee bend depth
    knee_bend = abs(hip.y - knee.y)
    
    if wrist_above_head and elbow_above_shoulder:
        if knee_bend < 0.18 or elbow_angle > 155:
            return 'smash'
        else:
            return 'clear'
    elif wrist_above_shoulder and not wrist_above_head:
        wrist_shoulder_gap = shoulder.y - wrist.y
        if wrist_shoulder_gap < 0.08:
            return 'drop'
        else:
            return 'drive'
    elif wrist_below_hip or wrist_near_hip:
        return 'net'
    else:
        return 'drive'


# =====================================================================
# STABILITY & BALANCE ANALYSIS
# =====================================================================

def calculate_stability(landmarks):
    """
    Measures player's balance and stability.
    High score = Head is centered over hips (stable).
    Low score = Head is leaning too far (unbalanced).
    """
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    hip_center_x = (l_hip.x + r_hip.x) / 2.0
    
    # Calculate horizontal drift of the head relative to the hips
    drift = abs(nose.x - hip_center_x)
    
    # Scale: 0 drift = 100%, 0.33 drift = 0%
    stability_score = max(0, 100 - (drift * 300))
    return stability_score



# =====================================================================
# MAIN VIDEO PROCESSING PIPELINE
# =====================================================================

def process_video(input_path, output_filename, output_dir="processed"):
    """Process a user's badminton video with 3D pose analysis.
    
    Pipeline:
        1. Extract frames with OpenCV
        2. Run MediaPipe Pose (model_complexity=1 for speed)
        3. Filter by landmark confidence
        4. Apply EMA smoothing to angles
        5. Detect shot phases and classify shot type
        6. Find best contact frame
        7. Calculate all joint angles
        8. Generate annotated output images/videos
    
    Args:
        input_path: Path to uploaded video file
        output_filename: Base name for output files
        output_dir: Directory for processed outputs
        
    Returns:
        dict with all analysis results including angles and comparison data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(output_filename)[0]
    
    # ─── PASS 1: Full frame analysis ───
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize for processing speed but maintain clarity (640p)
    max_w = 640
    if original_width > max_w:
        scale = max_w / original_width
        width = int(original_width * scale)
        height = int(original_height * scale)
    else:
        width = original_width
        height = original_height
    
    # TURBO SKIP: Aggressively skip frames to cut processing time
    skip = 2  # Always skip at least every other frame
    if total_frames > 300:    # > 10 seconds
        skip = 5
    elif total_frames > 150:  # > 5 seconds
        skip = 4
    elif total_frames > 90:   # > 3 seconds
        skip = 3

    
    print(f"[PoseAnalyzer] Processing {total_frames} frames at {width}x{height}, skip={skip}")
    
    smoother = KalmanSmoother()
    frame_data = []
    prev_wrist_vel = 0
    prev_phase = 'idle'
    
    # Player lock: track the CLOSEST (largest) person and stick with them
    locked_player_x = None  # Center-x of the locked player (0.0 to 1.0)
    locked_player_size = 0  # Height of the locked player's bounding box
    PLAYER_LOCK_TOLERANCE = 0.25  # Max horizontal drift allowed
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0               # Optimized for SPEED (0 = fastest, 1 = standard)
    ) as pose:
        frame_idx = 0
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed
            if frame_idx % skip != 0:
                frame_idx += 1
                continue
            
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # ─── Player identification: size + position lock ───
                # Calculate bounding box of detected skeleton
                xs = [lm.x for lm in landmarks if lm.visibility > 0.3]
                ys = [lm.y for lm in landmarks if lm.visibility > 0.3]
                if xs and ys:
                    bbox_h = max(ys) - min(ys)
                    center_x = (min(xs) + max(xs)) / 2.0
                    
                    if locked_player_x is None:
                        # First detection: lock onto the LARGEST person (closest to camera)
                        max_y = max(ys)
                        # Require the player to be relatively large (>15% of screen) OR near the bottom (max_y > 0.6)
                        if bbox_h > 0.15 or (bbox_h > 0.05 and max_y > 0.6):
                            locked_player_x = center_x
                            locked_player_size = bbox_h
                            print(f"[PoseAnalyzer] Locked onto player at x={center_x:.2f}, max_y={max_y:.2f}, size={bbox_h:.2f}")
                        else:
                            frame_idx += 1
                            continue
                    else:
                        max_y = max(ys)
                        # Reject if this detection looks like the opponent (far away, small, top of screen)
                        # and is far from our locked horizontal position.
                        if bbox_h < 0.15 and max_y < 0.5 and abs(center_x - locked_player_x) > 0.2:
                            # This is almost certainly the opponent, skip
                            frame_idx += 1
                            continue
                        # Update locked position with slight smoothing (allow the lock to follow the player)
                        locked_player_x = locked_player_x * 0.8 + center_x * 0.2
                
                # Check overall visibility
                critical_joints = [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                ]
                avg_vis = get_avg_visibility(landmarks, critical_joints)
                
                if avg_vis < MIN_VISIBILITY:
                    frame_idx += 1
                    prev_landmarks = landmarks
                    continue
                
                # Determine dominant side
                side = detect_dominant_side(landmarks)
                
                # Calculate raw angles
                raw_angles = calculate_all_angles(landmarks, side)
                
                # Apply EMA smoothing
                smoothed_angles = smoother.smooth(raw_angles) if raw_angles else None
                
                # ─── Calculate wrist metrics FIRST (used for YOLO skipping) ───
                wrist_vel = 0
                wrist_pos = None
                if side == 'right':
                    wrist_pos = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                else:
                    wrist_pos = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                if prev_landmarks:
                    r_w_c = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_w_p = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_v = math.hypot(r_w_c.x - r_w_p.x, r_w_c.y - r_w_p.y)
                    
                    l_w_c = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_w_p = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_v = math.hypot(l_w_c.x - l_w_p.x, l_w_c.y - l_w_p.y)
                    
                    wrist_vel = max(r_v, l_v)
                
                wrist_accel = wrist_vel - prev_wrist_vel
                
                # TURBO: YOLO disabled for speed — velocity-based contact detection is used instead
                # (YOLO on laptop CPU takes 0.5-2s per frame, which is too slow for 10-15s target)
                shuttlecock_pos = None
                shuttle_dist = None

                # Detect shot phase
                phase = detect_shot_phase(landmarks, wrist_vel, wrist_accel, prev_phase)
                
                # Classify shot type
                shot_type = classify_shot_type(landmarks)
                
                frame_data.append({
                    'frame_idx': frame_idx,
                    'angles': smoothed_angles,
                    'raw_angles': raw_angles,
                    'wrist_vel': wrist_vel,
                    'wrist_accel': wrist_accel,
                    'phase': phase,
                    'shot_type': shot_type,
                    'side': side,
                    'visibility': avg_vis,
                    'landmarks': landmarks,
                    'pose_landmarks': results.pose_landmarks,
                    # RESOLVED: Removed frame.copy() to save GIGABYTES of RAM
                    'shuttle_pos': shuttlecock_pos,
                    'shuttle_dist': shuttle_dist,
                    'stability': calculate_stability(landmarks)
                })
                
                prev_phase = phase
                prev_wrist_vel = wrist_vel
                prev_landmarks = landmarks
            
            frame_idx += 1
    
    cap.release()
    print(f"[PoseAnalyzer] Analyzed {len(frame_data)} frames out of {total_frames}")
    
    # ─── Handle no-motion case ───
    if not frame_data:
        return _empty_result()
    
    # ─── Find best contact frame using YOLO SHUTTLE TRACKING ───
    # We look for the frame with the absolute MINIMUM distance between shuttlecock and wrist
    shuttle_hits = [f for f in frame_data if f.get('shuttle_dist') is not None]
    
    yolo_contact = None
    if shuttle_hits:
        # Sort by distance: closest is most likely contact
        yolo_contact = min(shuttle_hits, key=lambda x: x['shuttle_dist'])
        print(f"[PoseAnalyzer] YOLO detected shuttlecock contact at frame {yolo_contact['frame_idx']} (dist: {yolo_contact['shuttle_dist']:.4f})")
    
    # Existing fallback logic
    contact_frames = [f for f in frame_data if f['phase'] == 'contact']
    
    if not contact_frames:
        # Fallback: use highest wrist velocity frames as "contact"
        sorted_by_vel = sorted(frame_data, key=lambda x: x['wrist_vel'], reverse=True)
        contact_frames = sorted_by_vel[:max(1, len(sorted_by_vel) // 10)]
    
    # Determine the dominant shot type from all contact frames
    shot_types = [f.get('shot_type', 'drive') for f in contact_frames]
    dominant_shot = max(set(shot_types), key=shot_types.count) if shot_types else 'drive'
    
    # Load reference angles for comparison
    ref_player = get_active_reference_player()
    ref_angles = None
    if ref_player:
        ref_angles = get_reference_angles(ref_player.id, dominant_shot, 'contact')
        if not ref_angles:
            ref_angles = get_reference_angles(ref_player.id, 'drive', 'contact')
    
    print(f"[PoseAnalyzer] Shot type: {dominant_shot}, Reference available: {ref_angles is not None}")
    
    # Score ALL frames by similarity to pro reference
    for f in frame_data:
        if f['angles'] and ref_angles:
            # Score = how CLOSE this frame is to the pro's form
            # Lower total difference = higher score
            total_diff = 0
            joint_count = 0
            for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
                user_val = f['angles'].get(joint)
                ref_val = ref_angles.get(joint)
                if user_val is not None and ref_val is not None:
                    total_diff += abs(user_val - ref_val)
                    joint_count += 1
            
            if joint_count > 0:
                avg_diff = total_diff / joint_count
                # Convert to a 0-100 similarity score (0° diff = 100, 90°+ diff = 0)
                f['quality_score'] = max(0, 100 - (avg_diff * 100 / 90))
            else:
                f['quality_score'] = 0
        elif f['angles']:
            # No reference available: fallback to extension-based scoring
            elbow = f['angles'].get('elbow') or 0
            shoulder = f['angles'].get('shoulder') or 0
            knee = f['angles'].get('knee') or 0
            wrist = f['angles'].get('wrist') or 0
            vel = f.get('wrist_vel', 0)
            f['quality_score'] = (
                (elbow / 180.0) * 35 + (shoulder / 180.0) * 25 +
                (knee / 180.0) * 15 + (wrist / 180.0) * 10 +
                min(vel * 200, 15)
            )
        else:
            f['quality_score'] = 0
    
    # ─── Multi-Shot Compilation Logic ───
    # We want ALL good shots in the best video, and ALL bad shots in the worst video.
    # First, we must group consecutive 'contact' frames into distinct shots.
    distinct_shots = []
    if contact_frames:
        contact_frames.sort(key=lambda x: x['frame_idx'])
        current_cluster = [contact_frames[0]]
        for f in contact_frames[1:]:
            # If frames are within 1 second of each other, they are the same shot
            if f['frame_idx'] - current_cluster[-1]['frame_idx'] < fps:
                current_cluster.append(f)
            else:
                best_in_cluster = max(current_cluster, key=lambda x: x.get('quality_score', 0))
                distinct_shots.append(best_in_cluster)
                current_cluster = [f]
        if current_cluster:
            best_in_cluster = max(current_cluster, key=lambda x: x.get('quality_score', 0))
            distinct_shots.append(best_in_cluster)
            
    if distinct_shots:
        distinct_shots.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        mid_point = max(1, len(distinct_shots) // 2)
        if len(distinct_shots) == 1:
            best_frames = distinct_shots
            worst_frames = distinct_shots
        else:
            best_frames = distinct_shots[:mid_point]
            worst_frames = distinct_shots[mid_point:]
    else:
        # Fallback if no contact frames
        sorted_frames = sorted(frame_data, key=lambda x: x.get('quality_score', 0), reverse=True)
        best_frames = [sorted_frames[0]] if sorted_frames else []
        worst_frames = [sorted_frames[-1]] if sorted_frames else []

    # Get single best and worst for the static PDF snapshots
    best_contact = best_frames[0] if best_frames else {'frame_idx': 0, 'quality_score': 0}
    worst_frame = worst_frames[0] if worst_frames else {'frame_idx': 0, 'quality_score': 0}

    # Sort them chronologically for video generation
    best_frames.sort(key=lambda x: x['frame_idx'])
    worst_frames.sort(key=lambda x: x['frame_idx'])

    # ─── Generate output images (Re-reading from file to save RAM) ───
    best_img_name = f"best_{base_name}.jpg"
    worst_img_name = f"worst_{base_name}.jpg"
    
    # Snapshot of the absolute best and absolute worst
    _save_annotated_snapshot(input_path, os.path.join(output_dir, best_img_name), 
                             best_contact['frame_idx'], width, height, best_contact)
    _save_annotated_snapshot(input_path, os.path.join(output_dir, worst_img_name), 
                             worst_frame['frame_idx'], width, height, worst_frame)
    
    snapshot_filename = f"{best_img_name}|{int(best_contact.get('quality_score', 0))}|{worst_img_name}|{int(worst_frame.get('quality_score', 0))}"
    
    # ─── Generate video clips (Re-reading from file to save RAM) ───
    best_video_name = f"best_{base_name}.mp4"
    worst_video_name = f"worst_{base_name}.mp4"
    
    best_chapters = _generate_clip_safe(input_path, best_frames, best_video_name, output_dir, 
                        fps, width, height, frame_data, skip, ref_angles)
    worst_chapters = _generate_clip_safe(input_path, worst_frames, worst_video_name, output_dir,
                        fps, width, height, frame_data, skip, ref_angles)
                        
    # Save chapters to JSON
    best_chapters_file = f"best_chapters_{base_name}.json"
    worst_chapters_file = f"worst_chapters_{base_name}.json"
    with open(os.path.join(output_dir, best_chapters_file), 'w') as f:
        json.dump(best_chapters, f)
    with open(os.path.join(output_dir, worst_chapters_file), 'w') as f:
        json.dump(worst_chapters, f)
        
    processed_video_filename = f"{best_video_name}|{worst_video_name}"

    
    # ─── Collect phase angles for comparison ───
    phase_angles = {}
    for phase_name in ['preparation', 'swing', 'contact', 'follow_through']:
        phase_frames = [f for f in frame_data if f['phase'] == phase_name and f['angles']]
        if phase_frames:
            # Average angles across all frames in this phase
            avg = {}
            for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
                vals = [f['angles'][joint] for f in phase_frames if f['angles'].get(joint) is not None]
                avg[joint] = round(sum(vals) / len(vals), 1) if vals else None
            phase_angles[phase_name] = avg
    
    # Use contact angles as primary, fallback to best available
    contact_angles = best_contact.get('angles', {}) or {}
    shot_type = best_contact.get('shot_type', 'drive')
    
    # ─── Calculate overall scores (backward compatible) ───
    all_scores = [f.get('quality_score', 0) for f in frame_data if f.get('quality_score', 0) > 0]
    final_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    arm_score = min(100, max(20, contact_angles.get('elbow', 0) / 1.8)) if contact_angles.get('elbow') else 50
    knee_score = min(100, max(20, contact_angles.get('knee', 0) / 1.8)) if contact_angles.get('knee') else 50
    hip_score = min(100, max(20, contact_angles.get('shoulder', 0) / 1.8)) if contact_angles.get('shoulder') else 50
    
    # Normalize final score to 0-100
    # When reference comparison is available, scores are already 0-100
    # When no reference, scores can exceed 100 due to extension-based scoring
    if not ref_angles:
        final_score = min(100, max(0, final_score / 1.5))
    else:
        final_score = min(100, max(0, final_score))
    
    # ─── Generate feedback text (backward compatible format) ───
    worst_angles = worst_frame.get('angles', {}) or {}
    feedback_text = _generate_basic_feedback(contact_angles, worst_angles, shot_type)
    
    # ─── Timestamp ───
    best_time_sec = int(best_contact['frame_idx'] / fps)
    best_mins = best_time_sec // 60
    best_secs = best_time_sec % 60
    best_timestamp = f"{best_mins:02d}:{best_secs:02d}"

    worst_time_sec = int(worst_frame['frame_idx'] / fps)
    mins = worst_time_sec // 60
    secs = worst_time_sec % 60
    worst_timestamp = f"{mins:02d}:{secs:02d}"
    
    combined_timestamps = f"{best_timestamp}|{worst_timestamp}"
    
    # Check for low FPS warning
    if fps < 55:
        feedback_text += " | ⚠️ Low Accuracy Warning: Your video is under 60fps. Fast racket swings and wrist angles may suffer from motion blur. For professional accuracy, film in 60fps."
        
    # ─── Calculate Stability & Speed ───
    stability_scores = [f.get('stability', 0) for f in frame_data if f.get('phase') in ('swing', 'contact')]
    avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 70
    max_wrist_vel = max([f.get('wrist_vel', 0) for f in frame_data if f.get('phase') in ('swing', 'contact')] + [0])
    
    best_duration = round(sum(c['duration'] for c in best_chapters), 1)
    worst_duration = round(sum(c['duration'] for c in worst_chapters), 1)
    
    return {
        # Backward compatible fields
        "final_score": round(final_score, 1),
        "arm_score": round(arm_score, 1),
        "knee_score": round(knee_score, 1),
        "hip_score": round(hip_score, 1),
        "snapshot_filename": snapshot_filename,
        "processed_video_filename": processed_video_filename,
        "feedback_text": feedback_text,
        "worst_timestamp": combined_timestamps,
        "stability_score": round(avg_stability, 1),
        "max_wrist_vel": max_wrist_vel,
        
        # Athlete Vision 2.0 fields
        "shot_type": shot_type,
        "contact_angles": contact_angles,
        "phase_angles": phase_angles,
        "total_frames_analyzed": len(frame_data),
        "contact_frame_idx": best_contact['frame_idx'],
        "dominant_side": best_contact.get('side', 'right'),
        
        # Automated Processing Reel Metadata
        "best_chapters_file": best_chapters_file,
        "worst_chapters_file": worst_chapters_file,
        "best_shot_count": len(best_frames),
        "worst_shot_count": len(worst_frames),
        "best_duration_sec": best_duration,
        "worst_duration_sec": worst_duration
    }


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _empty_result():
    """Return empty result dict when no motion is detected."""
    return {
        "final_score": 0, "arm_score": 0, "knee_score": 0, "hip_score": 0,
        "snapshot_filename": None, "processed_video_filename": None,
        "feedback_text": "No significant motion or swings detected in the video.",
        "worst_timestamp": "00:00",
        "shot_type": None, "contact_angles": {},
        "phase_angles": {}, "total_frames_analyzed": 0,
        "contact_frame_idx": 0, "dominant_side": "right",
        "best_chapters_file": None, "worst_chapters_file": None,
        "best_shot_count": 0, "worst_shot_count": 0,
        "best_duration_sec": 0.0, "worst_duration_sec": 0.0
    }


def _save_annotated_snapshot(input_path, output_path, frame_idx, width, height, fdata):
    """Memory-safe snapshot: Open file, seek to frame, annotate, and save."""
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (width, height))
        if fdata.get('pose_landmarks'):
            # Removed pink skeleton line, using only blue bounding box
            xs = [lm.x for lm in fdata['pose_landmarks'].landmark if lm.visibility > 0.3]
            ys = [lm.y for lm in fdata['pose_landmarks'].landmark if lm.visibility > 0.3]
            if xs and ys:
                min_x, max_x = int(min(xs) * width), int(max(xs) * width)
                min_y, max_y = int(min(ys) * height), int(max(ys) * height)
                padding = 20
                cv2.rectangle(frame, (max(0, min_x - padding), max(0, min_y - padding)),
                              (min(width, max_x + padding), min(height, max_y + padding)),
                              (255, 0, 0), 2)  # Blue Box
                cv2.putText(frame, "USER", (max(0, min_x - padding), max(0, min_y - padding - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
        cv2.imwrite(output_path, frame)
    cap.release()


def _generate_clip_safe(input_path, target_frames, output_name, output_dir,
                       fps, width, height, frame_data, skip_val, ref_angles=None):
    """Memory-safe multi-clip generator: Re-reads only short windows around each target frame."""
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    clip_fps = fps # Normal playback speed for smooth viewing
    writer = cv2.VideoWriter(
        os.path.join(output_dir, output_name), fourcc,
        clip_fps, (width, height)
    )
    
    cap = cv2.VideoCapture(input_path)
    
    last_bbox = None # Box hold algorithm state
    trail_points = collections.deque(maxlen=20) # Kinetic Swing Trail

    chapters = []
    current_time_sec = 0.0
    
    for idx, target_frame in enumerate(target_frames):
        target_idx = target_frame['frame_idx']
        window = int(fps * 0.75)  # 0.75 seconds around contact (1.5 seconds per shot)
        start_f = max(0, target_idx - window)
        end_f = target_idx + window
        
        fdata_map = {f['frame_idx']: f for f in frame_data if start_f <= f['frame_idx'] <= end_f}
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        curr_idx = start_f
        frames_written = 0
        while curr_idx <= end_f:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (width, height))
            fdata = fdata_map.get(curr_idx)
            
            if fdata and fdata.get('pose_landmarks'):
                # Update box hold
                xs = [lm.x for lm in fdata['pose_landmarks'].landmark if lm.visibility > 0.3]
                ys = [lm.y for lm in fdata['pose_landmarks'].landmark if lm.visibility > 0.3]
                
                # Track wrist for Kinetic Trail
                wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
                wrist_lm = fdata['pose_landmarks'].landmark[wrist_idx]
                if wrist_lm.visibility > 0.3:
                    trail_points.append((int(wrist_lm.x * width), int(wrist_lm.y * height)))
                
                if xs and ys:
                    min_x, max_x = int(min(xs) * width), int(max(xs) * width)
                    min_y, max_y = int(min(ys) * height), int(max(ys) * height)
                    padding = 20
                    last_bbox = (max(0, min_x - padding), max(0, min_y - padding), 
                                 min(width, max_x + padding), min(height, max_y + padding))
            
            # Draw the box from the "box hold" state to ensure smoothness
            if last_bbox:
                cv2.rectangle(frame, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, "USER", (last_bbox[0], last_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            # Flash the "CONTACT POINT" text
            if abs(curr_idx - target_idx) <= max(1, int(fps * 0.15)):
                cv2.putText(frame, "CONTACT POINT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            writer.write(frame)
            curr_idx += 1
            frames_written += 1
            
        duration_sec = frames_written / clip_fps
        chapters.append({
            "shot_index": idx + 1,
            "start_time": round(current_time_sec, 2),
            "end_time": round(current_time_sec + duration_sec, 2),
            "duration": round(duration_sec, 2)
        })
        current_time_sec += duration_sec
            
    cap.release()
    writer.release()
    return chapters


def _generate_basic_feedback(contact_angles, worst_angles, shot_type):
    """Generate backward-compatible feedback text."""
    shot_label = shot_type.capitalize() if shot_type else 'Shot'
    
    if not contact_angles:
        return "❌ Issue: No clear shot detected|🎯 Fix: Ensure you are fully visible in the video|💡 Why: The AI needs to see your full body to analyze your technique."
    
    # Find the joint with the worst angle (lowest value often means less extension)
    issues = []
    if contact_angles.get('elbow') and contact_angles['elbow'] < 140:
        issues.append(('arm', f"Elbow not fully extended during {shot_label} ({contact_angles['elbow']:.0f}°)"))
    if contact_angles.get('shoulder') and contact_angles['shoulder'] < 130:
        issues.append(('shoulder', f"Shoulder rotation restricted during {shot_label} ({contact_angles['shoulder']:.0f}°)"))
    if contact_angles.get('knee') and contact_angles['knee'] < 110:
        issues.append(('knee', f"Insufficient knee bend during {shot_label} ({contact_angles['knee']:.0f}°)"))
    
    if not issues:
        return f"❌ Issue: None|🎯 Fix: Keep up the great form!|💡 Why: Your biomechanics look solid for your {shot_label}."
    
    # Pick biggest issue
    issue_type, issue_desc = issues[0]
    
    fixes = {
        'arm': f"Extend your arm fully upwards at the point of contact for your {shot_label}",
        'shoulder': f"Reach higher and rotate your torso into the {shot_label}",
        'knee': f"Bend your knees more dynamically for an explosive {shot_label}",
    }
    
    whys = {
        'arm': f"Full extension maximizes racket reach and power transfer during a {shot_label}.",
        'shoulder': f"Proper shoulder engagement increases downward angle and {shot_label} speed.",
        'knee': f"Knee bend generates core torque for a more explosive {shot_label}.",
    }
    
    fix = fixes.get(issue_type, f"Focus on smooth kinetic chain from legs through arm for your {shot_label}")
    why = whys.get(issue_type, "Coordinated movement improves both power and consistency.")
    
    return f"❌ Issue: {issue_desc}|🎯 Fix: {fix}|💡 Why: {why}"
