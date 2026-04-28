"""
Athlete Vision 3.0 — Court Mask Module
========================================
4-point perspective transform to define the playable court area.
Filters out any person detection whose "feet" (bottom of bounding box)
fall outside the court polygon.
"""

import cv2
import numpy as np


class CourtMask:
    """Defines a badminton court polygon from 4 corner points and provides
    spatial filtering to reject detections outside the court.

    The 4 corners should be provided in order:
        [top-left, top-right, bottom-right, bottom-left]
    as seen from the camera's perspective (not bird's-eye view).

    Usage:
        mask = CourtMask(corners=[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
                         frame_shape=(height, width))
        if mask.is_on_court(bbox):
            # keep this detection
    """

    def __init__(self, corner_points, frame_shape, margin_pct=0.12):
        """
        Args:
            corner_points: List of 4 (x, y) tuples — court corners in pixel coords.
                           Order: [top-left, top-right, bottom-right, bottom-left].
            frame_shape:   (height, width) of the video frame.
            margin_pct:    Extra margin around the court polygon (0.12 = 12%).
                           Accounts for players standing just outside court lines.
        """
        self.frame_h, self.frame_w = frame_shape[:2]
        self.raw_corners = np.array(corner_points, dtype=np.float32)

        # Expand polygon outward by margin to catch players near sidelines
        self.corners = self._expand_polygon(self.raw_corners, margin_pct)
        self.polygon = self.corners.astype(np.int32)

        # Pre-compute binary mask for fast lookup
        self._mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillConvexPoly(self._mask, self.polygon, 255)

        # Perspective transform: court corners → standard rectangle
        # Standard badminton court aspect ratio ≈ 13.4m × 6.1m ≈ 2.2:1
        rect_w, rect_h = 440, 200
        dst = np.array([
            [0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]
        ], dtype=np.float32)
        self.perspective_matrix = cv2.getPerspectiveTransform(self.raw_corners, dst)
        self.rect_size = (rect_w, rect_h)

    def is_on_court(self, bbox):
        """Check if a player's feet are inside the court polygon.

        Uses the bottom-center point of the bounding box as the "feet" position.

        Args:
            bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.

        Returns:
            True if the feet point is inside the expanded court polygon.
        """
        x1, y1, x2, y2 = bbox[:4]
        # Feet = bottom-center of bounding box
        feet_x = int((x1 + x2) / 2)
        feet_y = int(y2)

        # Clamp to frame boundaries
        feet_x = max(0, min(feet_x, self.frame_w - 1))
        feet_y = max(0, min(feet_y, self.frame_h - 1))

        return self._mask[feet_y, feet_x] > 0

    def get_bird_eye_view(self, frame):
        """Warp the court region to a top-down (bird's-eye) rectangle.

        Useful for trajectory analysis and minimap rendering.

        Args:
            frame: BGR frame from the video.

        Returns:
            Warped court image (rect_w × rect_h).
        """
        return cv2.warpPerspective(frame, self.perspective_matrix, self.rect_size)

    def draw_overlay(self, frame, color=(0, 255, 0), alpha=0.15):
        """Draw a semi-transparent court overlay on the frame.

        Args:
            frame:  BGR frame (modified in-place).
            color:  BGR color for the overlay.
            alpha:  Transparency (0 = invisible, 1 = opaque).

        Returns:
            Frame with overlay drawn.
        """
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, self.polygon, color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=2)
        return frame

    def point_to_court_coords(self, x, y):
        """Convert a pixel point to normalized court coordinates (0-1, 0-1).

        Uses the perspective transform to map the point onto the court rectangle.

        Args:
            x, y: Pixel coordinates in the video frame.

        Returns:
            (court_x, court_y) normalized to [0, 1] range, or None if outside.
        """
        if not self.is_on_court((x - 1, y - 1, x + 1, y + 1)):
            return None

        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.perspective_matrix)
        cx = transformed[0][0][0] / self.rect_size[0]
        cy = transformed[0][0][1] / self.rect_size[1]
        return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)))

    # ─── Internal Helpers ───

    def _expand_polygon(self, corners, margin_pct):
        """Expand polygon outward from its centroid by margin_pct.

        This ensures players standing just outside court lines are still captured.
        """
        centroid = corners.mean(axis=0)
        expanded = []
        for pt in corners:
            direction = pt - centroid
            expanded_pt = pt + direction * margin_pct
            # Clamp to frame boundaries
            expanded_pt[0] = max(0, min(expanded_pt[0], self.frame_w - 1))
            expanded_pt[1] = max(0, min(expanded_pt[1], self.frame_h - 1))
            expanded.append(expanded_pt)
        return np.array(expanded, dtype=np.float32)


def auto_detect_court(frame):
    """Attempt to auto-detect badminton court corners using line detection.

    Uses HSV color filtering (court lines are typically white on green)
    combined with Hough Line Transform.

    Args:
        frame: BGR frame from the video.

    Returns:
        List of 4 (x, y) corner points if detected, or None.
    """
    h, w = frame.shape[:2]

    # Convert to HSV and filter for white/bright lines on green court
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green court mask (filter court surface)
    green_lower = np.array([30, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # White line mask (bright areas)
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Lines should be ON the green court
    court_lines = cv2.bitwise_and(white_mask, green_mask,
                                   dst=cv2.dilate(green_mask, None, iterations=5))

    # Edge detection on the white lines
    edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=int(w * 0.15), maxLineGap=30)

    if lines is None or len(lines) < 4:
        return None

    # Find the outermost intersection points to form the court rectangle
    # This is a simplified approach — group lines into horizontal and vertical
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 30 or angle > 150:
            h_lines.append(line[0])
        elif 60 < angle < 120:
            v_lines.append(line[0])

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # Sort horizontal lines by y-position (top to bottom)
    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    top_line = h_lines[0]
    bottom_line = h_lines[-1]

    # Sort vertical lines by x-position (left to right)
    v_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    left_line = v_lines[0]
    right_line = v_lines[-1]

    # Find intersection points
    def line_intersection(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (int(ix), int(iy))

    corners = [
        line_intersection(top_line, left_line),     # top-left
        line_intersection(top_line, right_line),    # top-right
        line_intersection(bottom_line, right_line), # bottom-right
        line_intersection(bottom_line, left_line),  # bottom-left
    ]

    # Validate all intersections found and within frame
    for c in corners:
        if c is None:
            return None
        if c[0] < 0 or c[0] >= w or c[1] < 0 or c[1] >= h:
            return None

    return corners
