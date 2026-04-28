"""
Athlete Vision 3.0 — Pose Gating Module
=========================================
Uses YOLOv8-Pose keypoints to distinguish active players from
seated spectators, coaches, and umpires.

Only passes through detections whose skeleton shows an upright,
athletic stance or active movement.
"""

import numpy as np


# COCO Pose keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


class PoseGate:
    """Filters person detections by analyzing their skeletal keypoints
    to determine if the person is in an athletic/playing stance.

    Rejects detections that show sitting postures, which indicates
    spectators, coaches, or officials.
    """

    def __init__(self, min_stance_score=0.45):
        """
        Args:
            min_stance_score: Minimum score (0-1) to pass the pose gate.
                              Higher = stricter filtering.
        """
        self.min_stance_score = min_stance_score

    def is_athletic(self, keypoints):
        """Check if a person's keypoints indicate an athletic (playing) stance.

        Args:
            keypoints: numpy array of shape (17, 3) — [x, y, confidence]
                       COCO format from YOLOv8-Pose.

        Returns:
            (bool, float) — (passes_gate, stance_score)
        """
        score = self.get_stance_score(keypoints)
        return score >= self.min_stance_score, score

    def get_stance_score(self, keypoints):
        """Calculate an athletic stance score from 0 (sitting) to 1 (playing).

        Criteria:
        1. Upright torso: shoulder-hip vertical alignment
        2. Standing: hip-to-ankle distance vs torso length
        3. Active arms: at least one arm raised or extended
        4. Knee angle: not sitting (knees bent < 60°)

        Args:
            keypoints: numpy array of shape (17, 3) — [x, y, confidence]

        Returns:
            Float score in [0, 1].
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0

        kp = np.array(keypoints)

        # Minimum confidence threshold for keypoints
        CONF_THRESH = 0.3

        # ─── 1. Torso Verticality (0 to 0.3 points) ───
        torso_score = 0.0
        l_shoulder = kp[LEFT_SHOULDER]
        r_shoulder = kp[RIGHT_SHOULDER]
        l_hip = kp[LEFT_HIP]
        r_hip = kp[RIGHT_HIP]

        if (l_shoulder[2] > CONF_THRESH and r_shoulder[2] > CONF_THRESH and
                l_hip[2] > CONF_THRESH and r_hip[2] > CONF_THRESH):
            shoulder_mid = (l_shoulder[:2] + r_shoulder[:2]) / 2
            hip_mid = (l_hip[:2] + r_hip[:2]) / 2

            torso_vec = hip_mid - shoulder_mid
            torso_len = np.linalg.norm(torso_vec)

            if torso_len > 5:
                # Angle from vertical (0° = perfectly upright)
                vertical_angle = abs(np.arctan2(torso_vec[0], torso_vec[1]))
                # < 30° from vertical = good
                if vertical_angle < np.radians(30):
                    torso_score = 0.3
                elif vertical_angle < np.radians(50):
                    torso_score = 0.15

        # ─── 2. Standing Height (0 to 0.35 points) ───
        standing_score = 0.0
        # Check hip-to-ankle vertical distance
        ankles_visible = False
        ankle_y = 0
        hip_y = 0

        if l_hip[2] > CONF_THRESH and r_hip[2] > CONF_THRESH:
            hip_y = (l_hip[1] + r_hip[1]) / 2

            l_ankle = kp[LEFT_ANKLE]
            r_ankle = kp[RIGHT_ANKLE]

            if l_ankle[2] > CONF_THRESH or r_ankle[2] > CONF_THRESH:
                ankles_visible = True
                if l_ankle[2] > CONF_THRESH and r_ankle[2] > CONF_THRESH:
                    ankle_y = (l_ankle[1] + r_ankle[1]) / 2
                elif l_ankle[2] > CONF_THRESH:
                    ankle_y = l_ankle[1]
                else:
                    ankle_y = r_ankle[1]

                # Hip-to-ankle distance should be significant (standing)
                shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2 if (
                    l_shoulder[2] > CONF_THRESH and r_shoulder[2] > CONF_THRESH
                ) else hip_y - 50

                total_height = ankle_y - shoulder_mid_y
                leg_height = ankle_y - hip_y

                if total_height > 10:
                    leg_ratio = leg_height / total_height
                    # Standing: legs are ~50% of total height
                    # Sitting: legs are compressed, < 30%
                    if leg_ratio > 0.30:
                        standing_score = 0.35
                    elif leg_ratio > 0.20:
                        standing_score = 0.15
            else:
                # Ankles not visible — could be cropped at bottom (player close to camera)
                # Give partial credit if other signs are good
                standing_score = 0.15

        # ─── 3. Active Arms (0 to 0.20 points) ───
        arm_score = 0.0
        l_wrist = kp[LEFT_WRIST]
        r_wrist = kp[RIGHT_WRIST]

        arm_raised = False
        if l_wrist[2] > CONF_THRESH and l_shoulder[2] > CONF_THRESH:
            if l_wrist[1] < l_shoulder[1]:  # Wrist above shoulder
                arm_raised = True
        if r_wrist[2] > CONF_THRESH and r_shoulder[2] > CONF_THRESH:
            if r_wrist[1] < r_shoulder[1]:
                arm_raised = True

        # Also check arm extension (elbows extended)
        arm_extended = False
        l_elbow = kp[LEFT_ELBOW]
        r_elbow = kp[RIGHT_ELBOW]

        if (r_wrist[2] > CONF_THRESH and r_elbow[2] > CONF_THRESH and
                r_shoulder[2] > CONF_THRESH):
            arm_span = np.linalg.norm(r_wrist[:2] - r_shoulder[:2])
            upper_arm = np.linalg.norm(r_elbow[:2] - r_shoulder[:2])
            if upper_arm > 5 and arm_span > upper_arm * 1.3:
                arm_extended = True

        if (l_wrist[2] > CONF_THRESH and l_elbow[2] > CONF_THRESH and
                l_shoulder[2] > CONF_THRESH):
            arm_span = np.linalg.norm(l_wrist[:2] - l_shoulder[:2])
            upper_arm = np.linalg.norm(l_elbow[:2] - l_shoulder[:2])
            if upper_arm > 5 and arm_span > upper_arm * 1.3:
                arm_extended = True

        if arm_raised:
            arm_score = 0.20
        elif arm_extended:
            arm_score = 0.10

        # ─── 4. Knee Bend — HARD REJECTION for seated poses ───
        knee_score = 0.0
        l_knee = kp[LEFT_KNEE]
        r_knee = kp[RIGHT_KNEE]

        def knee_angle(hip, knee, ankle):
            """Calculate angle at the knee joint."""
            if hip[2] < CONF_THRESH or knee[2] < CONF_THRESH or ankle[2] < CONF_THRESH:
                return None
            v1 = hip[:2] - knee[:2]
            v2 = ankle[:2] - knee[:2]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        l_angle = knee_angle(l_hip, l_knee, kp[LEFT_ANKLE])
        r_angle = knee_angle(r_hip, r_knee, kp[RIGHT_ANKLE])

        angles = [a for a in [l_angle, r_angle] if a is not None]
        if angles:
            avg_knee_angle = sum(angles) / len(angles)
            # HARD REJECT: both knees bent < 100° = SITTING (umpire chair)
            if avg_knee_angle < 100:
                return 0.0   # Instant fail — this person is seated
            knee_score = 0.15

        # ─── 5. Sitting Detector: knees at hip level = chair ───
        if (l_knee[2] > CONF_THRESH and r_knee[2] > CONF_THRESH and
                l_hip[2] > CONF_THRESH and r_hip[2] > CONF_THRESH):
            knee_mid_y = (l_knee[1] + r_knee[1]) / 2
            hip_mid_y = (l_hip[1] + r_hip[1]) / 2
            # If knees are at nearly the same height as hips → sitting
            if abs(knee_mid_y - hip_mid_y) < 15:
                return 0.0   # Seated on chair

        total = torso_score + standing_score + arm_score + knee_score
        return min(1.0, total)

    def filter_detections(self, detections, keypoints_list):
        """Filter a list of detections, keeping only athletic stances.

        Args:
            detections:     List of (x1, y1, x2, y2, conf) bounding boxes.
            keypoints_list: List of keypoint arrays, one per detection.
                            Each is shape (17, 3) from YOLOv8-Pose.

        Returns:
            List of (detection, keypoints, stance_score) tuples that pass the gate.
        """
        passed = []
        for det, kps in zip(detections, keypoints_list):
            is_ok, score = self.is_athletic(kps)
            if is_ok:
                passed.append((det, kps, score))
        return passed
