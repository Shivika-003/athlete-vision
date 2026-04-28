"""
Athlete Vision 3.0 — Deterministic Shot Classifier
====================================================
Rule-based temporal shot classification using a 30-frame sliding window
of player wrist keypoints and shuttlecock positions.

Replaces all random.choice() logic with deterministic analysis of:
- Wrist trajectory (height, velocity, direction)
- Shuttlecock impact position
- Body center for forehand/backhand
- Trajectory angle for direction
"""

import collections
import numpy as np


# COCO Pose keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_WRIST = 9
RIGHT_WRIST = 10
NOSE = 0


class ShotClassifier:
    """Analyzes the last 30 frames of player movement and shuttlecock
    trajectory to classify shots deterministically.

    Each player gets their own instance to track independent windows.
    """

    def __init__(self, window_size=30):
        """
        Args:
            window_size: Number of frames to keep in the sliding window.
        """
        self.window_size = window_size
        self.wrist_history = collections.deque(maxlen=window_size)
        self.shuttle_history = collections.deque(maxlen=window_size)
        self.body_center_history = collections.deque(maxlen=window_size)
        self.body_lines_history = collections.deque(maxlen=window_size)

        # Cached results (updated every analysis cycle)
        self.current_shot = 'Neutral'
        self.current_handle = 'Forehand'
        self.current_pressure = 'In Control'
        self.current_direction = 'Straight'
        self.is_swinging = False

    def update(self, keypoints, shuttle_pos, frame_h, frame_w):
        """Feed a new frame's data into the sliding window.

        Args:
            keypoints:   Shape (17, 3) keypoints from YOLO-Pose, or None.
            shuttle_pos: (x, y) shuttlecock position, or None.
            frame_h:     Frame height for normalization.
            frame_w:     Frame width for normalization.
        """
        wrist_data = None
        body_center = None

        if keypoints is not None and len(keypoints) >= 17:
            kp = np.array(keypoints)
            CONF = 0.3

            # Get both wrists and pick the dominant (higher) one
            lw = kp[LEFT_WRIST]
            rw = kp[RIGHT_WRIST]

            if rw[2] > CONF and lw[2] > CONF:
                # Use the one that's higher (smaller y = higher in frame)
                if rw[1] < lw[1]:
                    wrist_data = {'x': rw[0] / frame_w, 'y': rw[1] / frame_h, 'side': 'right'}
                else:
                    wrist_data = {'x': lw[0] / frame_w, 'y': lw[1] / frame_h, 'side': 'left'}
            elif rw[2] > CONF:
                wrist_data = {'x': rw[0] / frame_w, 'y': rw[1] / frame_h, 'side': 'right'}
            elif lw[2] > CONF:
                wrist_data = {'x': lw[0] / frame_w, 'y': lw[1] / frame_h, 'side': 'left'}

            # Shoulder midpoint (for handle classification)
            ls = kp[LEFT_SHOULDER]
            rs = kp[RIGHT_SHOULDER]
            if ls[2] > CONF and rs[2] > CONF:
                body_center = {
                    'x': (ls[0] + rs[0]) / (2 * frame_w),
                    'y': (ls[1] + rs[1]) / (2 * frame_h)
                }
            
            # Calculate dynamic body lines (Nose=0, Shoulders=5,6, Hips=11,12)
            nose_y = kp[0][1] / frame_h
            shoulder_y = (kp[5][1] + kp[6][1]) / 2 / frame_h
            waist_y = (kp[11][1] + kp[12][1]) / 2 / frame_h
            
            self.body_lines_history.append({
                'head': nose_y,
                'shoulder': shoulder_y,
                'waist': waist_y
            })
        else:
            self.body_lines_history.append(None)

        self.wrist_history.append(wrist_data)
        self.shuttle_history.append(
            {'x': shuttle_pos[0] / frame_w, 'y': shuttle_pos[1] / frame_h} if shuttle_pos else None
        )
        self.body_center_history.append(body_center)

    def classify(self):
        """Analyze the current window and update all classifications.

        Returns:
            dict with 'shot', 'handle', 'pressure', 'direction'.
        """
        self._classify_shot_type()
        self._classify_handle()
        self._classify_pressure()
        self._classify_direction()

        return {
            'Shot': self.current_shot,
            'Handle': self.current_handle,
            'Pressure': self.current_pressure,
            'Direction': self.current_direction,
            'Is_Swinging': self.is_swinging
        }

    def _classify_shot_type(self):
        self.is_swinging = False
        wrists = [w for w in self.wrist_history if w is not None]
        if len(wrists) < 3:
            return

        recent_w = wrists[-min(10, len(wrists)):]
        ys = [w['y'] for w in recent_w]
        
        # Calculate wrist speed
        dy = ys[-1] - ys[0]  # Positive = wrist moving down the screen (towards waist/floor)
        wrist_velocity = abs(dy) / len(ys)
        
        current_y = ys[-1]
        
        recent_lines = [b for b in self.body_lines_history if b is not None]
        if not recent_lines:
            return
            
        lines = recent_lines[-1]
            
        # Use the player's actual dynamic body keypoints
        # (Add slight offsets because wrist center might be slightly below actual head top)
        HEAD_LINE = lines['head'] + 0.02
        SHOULDER_LINE = lines['shoulder']
        WAIST_LINE = lines['waist']

        if current_y < HEAD_LINE:
            # High shot
            if wrist_velocity > 0.015 and dy > 0: # Swing down fast
                self.current_shot = 'Smash'
                self.is_swinging = True
            else:
                self.current_shot = 'Clear'
                if wrist_velocity > 0.01: self.is_swinging = True
        elif current_y < SHOULDER_LINE:
            if wrist_velocity < 0.008:
                self.current_shot = 'Drop'
            else:
                self.current_shot = 'Drive'
                if wrist_velocity > 0.01: self.is_swinging = True
        elif current_y > WAIST_LINE:
            if dy < -0.01:
                self.current_shot = 'Lift'
                self.is_swinging = True
            else:
                self.current_shot = 'Net'
                if wrist_velocity > 0.01: self.is_swinging = True
        else:
            self.current_shot = 'Drive'
            if wrist_velocity > 0.01: self.is_swinging = True

    def _classify_handle(self):
        recent_wrists = [w for w in self.wrist_history if w is not None]
        recent_bodies = [b for b in self.body_center_history if b is not None]

        if not recent_wrists or not recent_bodies:
            return

        wrist_x = recent_wrists[-1]['x']
        shoulder_mid_x = recent_bodies[-1]['x']
        shoulder_mid_y = recent_bodies[-1]['y']
        dominant_side = recent_wrists[-1].get('side', 'right')

        # Player at the bottom of the screen (facing up/away)
        is_bottom_player = shoulder_mid_y > 0.5

        if dominant_side == 'right':
            if is_bottom_player:
                self.current_handle = 'Forehand' if wrist_x > shoulder_mid_x else 'Backhand'
            else:
                self.current_handle = 'Forehand' if wrist_x < shoulder_mid_x else 'Backhand'
        else:
            if is_bottom_player:
                self.current_handle = 'Forehand' if wrist_x < shoulder_mid_x else 'Backhand'
            else:
                self.current_handle = 'Forehand' if wrist_x > shoulder_mid_x else 'Backhand'

    def _classify_pressure(self):
        """Determine pressure state from player position and shot context.

        Offensive: Player in strong position, hitting power shots
        Defending: Player stretched, hitting defensive shots
        In Control: Player centered, balanced
        Under Pressure: Player forced into awkward position
        """
        if self.current_shot in ('Smash', 'Clear'):
            self.current_pressure = 'Offensive'
        elif self.current_shot in ('Lift', 'Net'):
            # Check body position — if stretched, under pressure
            recent_wrists = [w for w in self.wrist_history if w is not None]
            recent_bodies = [b for b in self.body_center_history if b is not None]

            if recent_wrists and recent_bodies:
                wrist = recent_wrists[-1]
                body = recent_bodies[-1]
                stretch = abs(wrist['x'] - body['x'])
                if stretch > 0.15:
                    self.current_pressure = 'Under Pressure'
                else:
                    self.current_pressure = 'Defending'
            else:
                self.current_pressure = 'Defending'
        elif self.current_shot == 'Drive':
            self.current_pressure = 'In Control'
        elif self.current_shot == 'Drop':
            self.current_pressure = 'Offensive'
        else:
            self.current_pressure = 'In Control'

    def _classify_direction(self):
        """Determine shot direction from shuttlecock trajectory.

        Tracks shuttle position over last 10 frames after contact.
        < 20° from center = Straight
        > 20° = Cross-court (Left or Right)
        """
        shuttles = [s for s in self.shuttle_history if s is not None]
        if len(shuttles) < 3:
            self.current_direction = 'Straight'
            return

        recent = shuttles[-min(10, len(shuttles)):]
        start_x = recent[0]['x']
        end_x = recent[-1]['x']
        start_y = recent[0]['y']
        end_y = recent[-1]['y']

        dx = end_x - start_x
        dy = end_y - start_y

        if abs(dy) < 0.01:
            self.current_direction = 'Straight'
            return

        # Angle of trajectory from vertical
        angle_deg = abs(np.degrees(np.arctan2(dx, dy)))

        if angle_deg < 20:
            self.current_direction = 'Straight'
        elif dx > 0:
            self.current_direction = 'Cross/Right'
        else:
            self.current_direction = 'Cross/Left'
