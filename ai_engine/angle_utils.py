"""
Athlete Vision 2.0 — Shared Angle Utilities
=============================================
3D angle calculation, confidence filtering, temporal smoothing (EMA),
and comparison metrics (MAE, MSE, similarity score).
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Minimum landmark visibility to consider a joint "reliable"
MIN_VISIBILITY = 0.5

# Joint angle definitions: (pointA, vertex, pointC) for each angle
# Angle is measured at the VERTEX joint
JOINT_ANGLE_DEFINITIONS = {
    'shoulder': (
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value
    ),
    'elbow': (
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value
    ),
    'wrist': (
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_INDEX.value
    ),
    'knee': (
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value
    ),
    'ankle': (
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    ),
}

# Left-side equivalents for when left hand is dominant
JOINT_ANGLE_DEFINITIONS_LEFT = {
    'shoulder': (
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value
    ),
    'elbow': (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value
    ),
    'wrist': (
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_INDEX.value
    ),
    'knee': (
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value
    ),
    'ankle': (
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    ),
}


# =====================================================================
# 3D ANGLE CALCULATION
# =====================================================================

def calculate_angle_3d(a, b, c):
    """Calculate the angle at joint B using 3D coordinates.
    
    Uses the dot product method on 3D vectors for camera-independent
    angle measurement. Works with both 2D [x,y] and 3D [x,y,z] inputs.
    
    Args:
        a: Point A coordinates [x, y, z] or [x, y]
        b: Point B (vertex) coordinates
        c: Point C coordinates
        
    Returns:
        Angle at B in degrees (0-180)
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    
    ba = a - b  # Vector from B to A
    bc = c - b  # Vector from B to C
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0  # Points are overlapping
    
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)  # Prevent NaN from floating point
    angle = np.degrees(np.arccos(cosine))
    
    return float(angle)


# =====================================================================
# LANDMARK EXTRACTION WITH CONFIDENCE FILTERING
# =====================================================================

def get_landmark_3d(landmarks, landmark_id, min_visibility=MIN_VISIBILITY):
    """Extract 3D coordinates from a landmark if it meets visibility threshold.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        landmark_id: Integer ID of the landmark
        min_visibility: Minimum visibility score (0.0-1.0)
        
    Returns:
        [x, y, z] array if visible enough, None otherwise
    """
    lm = landmarks[landmark_id]
    if lm.visibility < min_visibility:
        return None
    return [lm.x, lm.y, lm.z]


def get_landmark_2d(landmarks, landmark_id, width, height, min_visibility=MIN_VISIBILITY):
    """Extract 2D pixel coordinates from a landmark if visible.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        landmark_id: Integer ID of the landmark
        width: Frame width in pixels
        height: Frame height in pixels
        min_visibility: Minimum visibility score
        
    Returns:
        (x_px, y_px) tuple if visible enough, None otherwise
    """
    lm = landmarks[landmark_id]
    if lm.visibility < min_visibility:
        return None
    return (int(lm.x * width), int(lm.y * height))


def get_avg_visibility(landmarks, joint_ids):
    """Get average visibility score for a set of joints.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        joint_ids: List of landmark IDs to check
        
    Returns:
        Average visibility score (0.0-1.0)
    """
    vis_scores = [landmarks[jid].visibility for jid in joint_ids]
    return sum(vis_scores) / len(vis_scores) if vis_scores else 0.0


# =====================================================================
# DETERMINE DOMINANT HAND
# =====================================================================

def detect_dominant_side(landmarks):
    """Determine which hand is the playing hand based on wrist position.
    
    The hand that is higher (smaller y) during a shot is the dominant/playing hand.
    
    Returns:
        'right' or 'left'
    """
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    return 'right' if r_wrist.y < l_wrist.y else 'left'


# =====================================================================
# CALCULATE ALL 5 JOINT ANGLES
# =====================================================================

def calculate_all_angles(landmarks, side='right'):
    """Calculate all 5 joint angles using 3D coordinates.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        side: 'right' or 'left' for dominant hand
        
    Returns:
        dict with angle values, or None if critical landmarks are missing.
        Example: {'shoulder': 145.2, 'elbow': 162.3, 'wrist': 155.1, 'knee': 130.5, 'ankle': 95.2}
    """
    definitions = JOINT_ANGLE_DEFINITIONS if side == 'right' else JOINT_ANGLE_DEFINITIONS_LEFT
    
    angles = {}
    all_joint_ids = set()
    for joint_name, (a_id, b_id, c_id) in definitions.items():
        all_joint_ids.update([a_id, b_id, c_id])
    
    # Check that critical landmarks are visible
    for jid in all_joint_ids:
        pt = get_landmark_3d(landmarks, jid)
        if pt is None:
            return None  # Key landmark not visible enough
    
    for joint_name, (a_id, b_id, c_id) in definitions.items():
        a = get_landmark_3d(landmarks, a_id, min_visibility=0.3)
        b = get_landmark_3d(landmarks, b_id, min_visibility=0.3)
        c = get_landmark_3d(landmarks, c_id, min_visibility=0.3)
        
        if a is None or b is None or c is None:
            angles[joint_name] = None
        else:
            angles[joint_name] = round(calculate_angle_3d(a, b, c), 1)
    
    return angles


# =====================================================================
# EXPONENTIAL MOVING AVERAGE (EMA) SMOOTHER
# =====================================================================

class EMASmoother:
    """Smooths joint angles across frames to reduce MediaPipe jitter.
    
    Uses Exponential Moving Average: smoothed = alpha * raw + (1 - alpha) * prev
    Lower alpha = smoother (more lag), higher alpha = more responsive (less smooth)
    
    Usage:
        smoother = EMASmoother(alpha=0.3)
        smoothed_angles = smoother.smooth(raw_angles_dict)
    """
    
    def __init__(self, alpha=0.35):
        """Initialize smoother.
        
        Args:
            alpha: Smoothing factor (0.0-1.0). 0.35 is a good balance
                   between responsiveness and noise reduction.
        """
        self.alpha = alpha
        self.prev = {}
    
    def smooth(self, angles_dict):
        """Apply EMA smoothing to a dictionary of angle values.
        
        Args:
            angles_dict: {'shoulder': 145.2, 'elbow': 162.3, ...}
            
        Returns:
            Smoothed angles dictionary
        """
        if angles_dict is None:
            return None
            
        smoothed = {}
        for joint_name, raw_value in angles_dict.items():
            if raw_value is None:
                smoothed[joint_name] = self.prev.get(joint_name)
                continue
                
            if joint_name not in self.prev:
                smoothed[joint_name] = raw_value
            else:
                smoothed[joint_name] = round(
                    self.alpha * raw_value + (1 - self.alpha) * self.prev[joint_name], 1
                )
            self.prev[joint_name] = smoothed[joint_name]
        
        return smoothed
    
    def reset(self):
        """Reset smoother state for a new video."""
        self.prev = {}


# =====================================================================
# COMPARISON METRICS
# =====================================================================

def compute_mae(user_angles, ref_angles):
    """Compute Mean Absolute Error between user and reference angles.
    
    Args:
        user_angles: dict {'shoulder': val, 'elbow': val, ...}
        ref_angles: dict {'shoulder': val, 'elbow': val, ...}
        
    Returns:
        MAE value (lower is better)
    """
    diffs = []
    for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
        u = user_angles.get(joint)
        r = ref_angles.get(joint)
        if u is not None and r is not None:
            diffs.append(abs(u - r))
    
    return round(sum(diffs) / len(diffs), 1) if diffs else 0.0


def compute_mse(user_angles, ref_angles):
    """Compute Mean Squared Error between user and reference angles.
    
    Penalizes large deviations more heavily than MAE.
    
    Args:
        user_angles: dict {'shoulder': val, 'elbow': val, ...}
        ref_angles: dict {'shoulder': val, 'elbow': val, ...}
        
    Returns:
        MSE value (lower is better)
    """
    diffs_sq = []
    for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
        u = user_angles.get(joint)
        r = ref_angles.get(joint)
        if u is not None and r is not None:
            diffs_sq.append((u - r) ** 2)
    
    return round(sum(diffs_sq) / len(diffs_sq), 1) if diffs_sq else 0.0


def compute_similarity(user_angles, ref_angles, max_deviation=45.0):
    """Compute similarity percentage between user and reference angles.
    
    Uses normalized MAE to produce a 0-100% score.
    A deviation of 0 degrees = 100% similarity.
    A deviation of max_deviation degrees or more = 0% similarity.
    
    Args:
        user_angles: dict {'shoulder': val, 'elbow': val, ...}
        ref_angles: dict {'shoulder': val, 'elbow': val, ...}
        max_deviation: Maximum expected deviation in degrees
        
    Returns:
        Similarity percentage (0-100, higher is better)
    """
    mae = compute_mae(user_angles, ref_angles)
    similarity = max(0.0, 100.0 * (1.0 - mae / max_deviation))
    return round(similarity, 1)


def compute_per_joint_similarity(user_angles, ref_angles, max_deviation=45.0):
    """Compute similarity for each individual joint.
    
    Returns:
        dict {'shoulder': 85.2, 'elbow': 92.1, ...} with similarity %
    """
    result = {}
    for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
        u = user_angles.get(joint)
        r = ref_angles.get(joint)
        if u is not None and r is not None:
            diff = abs(u - r)
            sim = max(0.0, 100.0 * (1.0 - diff / max_deviation))
            result[joint] = round(sim, 1)
        else:
            result[joint] = None
    return result


def rank_weaknesses(user_angles, ref_angles):
    """Rank joints by deviation from reference (biggest weakness first).
    
    Returns:
        List of (joint_name, user_angle, ref_angle, deviation) tuples,
        sorted by deviation descending.
    """
    deviations = []
    for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
        u = user_angles.get(joint)
        r = ref_angles.get(joint)
        if u is not None and r is not None:
            dev = abs(u - r)
            deviations.append((joint, u, r, round(dev, 1)))
    
    deviations.sort(key=lambda x: x[3], reverse=True)
    return deviations
