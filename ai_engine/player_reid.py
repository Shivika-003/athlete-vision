"""
Athlete Vision 3.0 — Player Re-Identification Module
======================================================
Lightweight color-histogram-based Re-ID for maintaining Player_1 / Player_2
identity across frames. Uses HSV histograms of the upper torso region
and cv2.compareHist for matching.

When players cross paths at the net, Re-ID score is prioritized over
spatial proximity to prevent ID swapping.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class PlayerReID:
    """Maintains visual identity for exactly 2 badminton players using
    upper-torso HSV color histograms.
    """

    H_BINS = 30
    S_BINS = 32
    V_BINS = 16

    def __init__(self, ema_alpha=0.10, min_confidence=0.35):
        self.ema_alpha = ema_alpha
        self.min_confidence = min_confidence
        self.players = {}

    def extract_features(self, frame, bbox):
        """Extract HSV color histogram from the upper 60% of a bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < 20 or box_w < 10:
            return None

        upper_y2 = y1 + int(box_h * 0.60)
        inset_x = int(box_w * 0.15)
        crop = frame[y1:upper_y2, x1 + inset_x:x2 - inset_x]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.H_BINS, self.S_BINS, self.V_BINS],
            [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def register_player(self, player_id, frame, bbox):
        """Store the initial color features for a player."""
        hist = self.extract_features(frame, bbox)
        if hist is not None:
            self.players[player_id] = {
                'hist': hist,
                'bbox': bbox[:4],
                'registered': True
            }

    def compare(self, hist_a, hist_b):
        """Compute similarity between two histograms using correlation."""
        if hist_a is None or hist_b is None:
            return 0.0
        corr = cv2.compareHist(
            hist_a.astype(np.float32),
            hist_b.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        return max(0.0, float(corr))

    def match_players(self, frame, detections):
        """Match detected persons to registered player IDs using Hungarian algorithm.
        
        Returns dict mapping player_id -> detection_index.
        """
        if not self.players or not detections:
            return {}

        player_ids = sorted(self.players.keys())
        n_players = len(player_ids)
        n_dets = len(detections)
        if n_dets == 0:
            return {}

        cost_matrix = np.ones((n_players, n_dets), dtype=np.float64)
        for i, pid in enumerate(player_ids):
            stored_hist = self.players[pid]['hist']
            for j, det in enumerate(detections):
                det_hist = self.extract_features(frame, det)
                similarity = self.compare(stored_hist, det_hist)
                cost_matrix[i, j] = 1.0 - similarity

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        assignments = {}
        for row, col in zip(row_indices, col_indices):
            pid = player_ids[row]
            similarity = 1.0 - cost_matrix[row, col]
            if similarity >= self.min_confidence:
                assignments[pid] = col
        return assignments

    def update_features(self, player_id, frame, bbox):
        """Update stored features with EMA blending."""
        if player_id not in self.players:
            return
        new_hist = self.extract_features(frame, bbox)
        if new_hist is None:
            return
        old_hist = self.players[player_id]['hist']
        self.players[player_id]['hist'] = (
            (1 - self.ema_alpha) * old_hist + self.ema_alpha * new_hist
        )
        self.players[player_id]['bbox'] = bbox[:4]

    def both_registered(self):
        """Check if both Player 0 and Player 1 are registered."""
        return 0 in self.players and 1 in self.players

    def get_confidence(self, player_id, frame, bbox):
        """Get Re-ID confidence for a specific detection against a player."""
        if player_id not in self.players:
            return 0.0
        det_hist = self.extract_features(frame, bbox)
        return self.compare(self.players[player_id]['hist'], det_hist)
