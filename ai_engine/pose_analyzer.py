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

from ai_engine.angle_utils import (
    calculate_angle_3d, get_landmark_3d, get_landmark_2d,
    get_avg_visibility, detect_dominant_side, calculate_all_angles,
    EMASmoother, MIN_VISIBILITY
)
from ai_engine.reference_builder import (
    get_active_reference_player, get_reference_angles
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Try loading YOLO for player detection/cropping (optional)
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
except Exception:
    yolo_model = None


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
    
    # Resize for processing speed
    max_w = 480
    if original_width > max_w:
        scale = max_w / original_width
        width = int(original_width * scale)
        height = int(original_height * scale)
    else:
        width = original_width
        height = original_height
    
    # Frame skipping: process every Nth frame for long videos
    # Frame skipping: ONLY skip frames for long videos so we don't miss the exact contact point
    skip = 1
    if total_frames > 240:    # > 8 seconds
        skip = 3
    elif total_frames > 120:  # > 4 seconds
        skip = 2
    
    print(f"[PoseAnalyzer] Processing {total_frames} frames at {width}x{height}, skip={skip}")
    
    smoother = EMASmoother(alpha=0.35)
    frame_data = []
    prev_wrist_vel = 0
    prev_phase = 'idle'
    
    # Player lock: track the CLOSEST (largest) person and stick with them
    locked_player_x = None  # Center-x of the locked player (0.0 to 1.0)
    locked_player_size = 0  # Height of the locked player's bounding box
    PLAYER_LOCK_TOLERANCE = 0.25  # Max horizontal drift allowed
    
    with mp_pose.Pose(
        min_detection_confidence=0.6,    # Raised slightly for higher strictness
        min_tracking_confidence=0.6,
        model_complexity=1               # Restored to 1 for high accuracy tracking!
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
                        # First detection: lock onto the LARGEST person
                        if bbox_h > 0.15:  # Must take up at least 15% of frame height
                            locked_player_x = center_x
                            locked_player_size = bbox_h
                            print(f"[PoseAnalyzer] Locked onto player at x={center_x:.2f}, size={bbox_h:.2f}")
                        else:
                            frame_idx += 1
                            continue
                    else:
                        # Subsequent frames: reject if this is a different person
                        if abs(center_x - locked_player_x) > PLAYER_LOCK_TOLERANCE:
                            # This is probably the opponent, skip
                            frame_idx += 1
                            continue
                        # Update locked position with slight smoothing
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
                
                # Calculate wrist velocity
                wrist_vel = 0
                if prev_landmarks:
                    r_w_c = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_w_p = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_v = math.hypot(r_w_c.x - r_w_p.x, r_w_c.y - r_w_p.y)
                    
                    l_w_c = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_w_p = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_v = math.hypot(l_w_c.x - l_w_p.x, l_w_c.y - l_w_p.y)
                    
                    wrist_vel = max(r_v, l_v)
                
                wrist_accel = wrist_vel - prev_wrist_vel
                
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
                    'image': frame.copy(),
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
    
    # ─── Find best contact frame by comparing with PRO REFERENCE ───
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
    
    # Best contact frame = MOST SIMILAR to pro reference
    best_contact = max(contact_frames, key=lambda x: x.get('quality_score', 0))
    
    # ─── Find improvement frame = MOST DIFFERENT from pro ───
    # Skip first 2 seconds of video (usually setup/walking, not real play)
    skip_frames = int(fps * 2)
    active_frames = [f for f in frame_data 
                     if f['phase'] in ('swing', 'contact', 'follow_through', 'preparation')
                     and f['frame_idx'] > skip_frames]
    if not active_frames:
        active_frames = [f for f in frame_data if f['frame_idx'] > skip_frames]
    if not active_frames:
        active_frames = frame_data
    
    # Enforce minimum temporal separation (1.5 seconds apart from best)
    min_frame_gap = int(fps * 1.5)
    distant_frames = [f for f in active_frames 
                      if abs(f['frame_idx'] - best_contact['frame_idx']) > min_frame_gap]
    
    if not distant_frames:
        min_frame_gap = int(fps * 0.5)
        distant_frames = [f for f in active_frames 
                          if abs(f['frame_idx'] - best_contact['frame_idx']) > min_frame_gap]
    
    candidate_frames = distant_frames if distant_frames else active_frames
    
    # Worst frame = LOWEST quality score (most different from pro)
    worst_frame = min(candidate_frames, key=lambda x: x.get('quality_score', 0))
    
    # ─── Generate output images ───
    best_img_name = f"best_{base_name}.jpg"
    worst_img_name = f"worst_{base_name}.jpg"
    
    # Draw pose on best frame
    best_annotated = _annotate_frame(best_contact, width, height)
    worst_annotated = _annotate_frame(worst_frame, width, height)
    
    cv2.imwrite(os.path.join(output_dir, best_img_name), best_annotated)
    cv2.imwrite(os.path.join(output_dir, worst_img_name), worst_annotated)
    
    snapshot_filename = f"{best_img_name}|{int(best_contact.get('quality_score', 0))}|{worst_img_name}|{int(worst_frame.get('quality_score', 0))}"
    
    # ─── Generate video clips ───
    best_video_name = f"best_{base_name}.mp4"
    worst_video_name = f"worst_{base_name}.mp4"
    
    _generate_clip(frame_data, best_contact, best_video_name, output_dir, 
                   fps, width, height)
    _generate_clip(frame_data, worst_frame, worst_video_name, output_dir,
                   fps, width, height)
    
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
    
    arm_score = min(100, max(20, contact_angles.get('elbow', 0) / 1.8 * 100)) if contact_angles.get('elbow') else 50
    knee_score = min(100, max(20, contact_angles.get('knee', 0) / 1.8 * 100)) if contact_angles.get('knee') else 50
    hip_score = min(100, max(20, contact_angles.get('shoulder', 0) / 1.8 * 100)) if contact_angles.get('shoulder') else 50
    
    # Normalize final score to 0-100
    final_score = min(100, max(0, final_score / 1.5))
    
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
        
        # Athlete Vision 2.0 fields
        "shot_type": shot_type,
        "contact_angles": contact_angles,
        "phase_angles": phase_angles,
        "total_frames_analyzed": len(frame_data),
        "contact_frame_idx": best_contact['frame_idx'],
        "dominant_side": best_contact.get('side', 'right'),
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
    }


def _annotate_frame(frame_data, width, height):
    """Draw pose skeleton in pink."""
    image = frame_data['image'].copy()
    pose_landmarks = frame_data.get('pose_landmarks')
    
    if pose_landmarks:
        import mediapipe as mp
        mp_drawing_inst = mp.solutions.drawing_utils
        
        # Pink skeleton lines in BGR: (180, 105, 255)
        mp_drawing_inst.draw_landmarks(
            image, pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing_inst.DrawingSpec(color=(180, 105, 255), thickness=3, circle_radius=3),
            mp_drawing_inst.DrawingSpec(color=(180, 105, 255), thickness=3)
        )
    
    return image


def _generate_clip(all_frame_data, target_frame, output_name, output_dir,
                   fps, width, height):
    """Generate a short video clip using pre-analyzed frame data.
    
    Uses the already-stored frames from the first pass so we always
    show the SAME player that was analyzed (no re-running MediaPipe).
    """
    target_idx = target_frame['frame_idx']
    
    # Find nearby frames in the stored data (within ~1 second)
    frame_window = int(fps * 1)
    clip_frames = [f for f in all_frame_data 
                   if abs(f['frame_idx'] - target_idx) <= frame_window]
    
    # Sort by frame index for correct playback order
    clip_frames.sort(key=lambda x: x['frame_idx'])
    
    if not clip_frames:
        clip_frames = [target_frame]
    
    # Determine actual frame dimensions from stored images
    h, w = clip_frames[0]['image'].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    clip_fps = max(fps * 0.5, 10)  # Slow motion for review
    writer = cv2.VideoWriter(
        os.path.join(output_dir, output_name), fourcc,
        clip_fps, (w, h)
    )
    
    for f in clip_frames:
        annotated = _annotate_frame(f, w, h)
        writer.write(annotated)
    
    writer.release()


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
