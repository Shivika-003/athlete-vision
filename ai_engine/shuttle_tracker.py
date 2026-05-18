import cv2
import numpy as np
import math

def estimate_smash_speed(video_path, contact_frame_idx, fps, player_bbox_h, wrist_pos_px, width, height, player_side, max_wrist_vel=0):
    """
    Estimates the speed of the shuttlecock immediately following the contact point.
    
    Args:
        video_path: Path to the video file.
        contact_frame_idx: The exact frame index where the racket hits the shuttle.
        fps: Frames per second of the video.
        player_bbox_h: Height of the player's bounding box in pixels (used for real-world scaling).
        wrist_pos_px: Tuple of (x, y) coordinates of the wrist at contact.
        width, height: Video dimensions.
        player_side: 'right' or 'left' (which side of the court the player is facing).
        max_wrist_vel: The maximum pixel velocity of the wrist during the swing (fallback).
        
    Returns:
        Estimated speed in km/h. Returns 0 if tracking fails completely.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame_idx)
    
    # We will track the shuttle for a maximum of 5 frames post-contact.
    # Badminton smashes are so fast they cross the screen in 5-10 frames.
    track_frames = 5
    
    frames = []
    for _ in range(track_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        
    cap.release()
    
    if len(frames) < 3:
        return 0  # Not enough frames to calculate speed
        
    # Scale assumption: Average player height is ~1.75 meters.
    # This gives us a pixels-per-meter ratio to convert on-screen movement to real-world speed.
    pixels_per_meter = max(10, player_bbox_h) / 1.75
    
    shuttle_positions = []
    
    # Define a Search Region of Interest (ROI)
    # The shuttle will shoot out from the wrist. We only search in the forward direction.
    wx, wy = wrist_pos_px
    search_radius = int(width * 0.4) # Search up to 40% of screen width away
    
    for i in range(len(frames) - 1):
        # Frame differencing: isolate fast-moving objects
        diff = cv2.absdiff(frames[i], frames[i+1])
        
        # Threshold to get bright moving spots (shuttlecock blur)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Mask out everything outside our expected trajectory arc
        mask = np.zeros_like(thresh)
        if player_side == 'right':
            # Player is on the left, hitting towards the right. Mask the right side of the wrist.
            cv2.rectangle(mask, (wx - 50, 0), (min(width, wx + search_radius), height), 255, -1)
        else:
            # Player is on the right, hitting towards the left. Mask the left side of the wrist.
            cv2.rectangle(mask, (max(0, wx - search_radius), 0), (wx + 50, height), 255, -1)
            
        masked_thresh = cv2.bitwise_and(thresh, mask)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        min_dist = float('inf')
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Shuttlecock blur area should be relatively small but noticeable
            if 5 < area < 500:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Ensure it's moving AWAY from the wrist in the correct direction
                    if (player_side == 'right' and cx > wx) or (player_side == 'left' and cx < wx):
                        # Get the object closest to the expected trajectory (or previous position)
                        ref_x, ref_y = shuttle_positions[-1] if shuttle_positions else (wx, wy)
                        dist = math.hypot(cx - ref_x, cy - ref_y)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_contour = (cx, cy)
                            
        if best_contour:
            shuttle_positions.append(best_contour)
            
    if len(shuttle_positions) < 2:
        # Fallback: Optical flow failed to isolate the shuttle blur due to motion blur or low contrast.
        # We can estimate the Smash Speed using the biomechanical Kinetic Chain (Wrist Speed).
        # A fast wrist snap directly correlates to a fast shuttle.
        if max_wrist_vel > 0:
            # wrist_vel is in pixels/frame. Convert to pixels/sec -> meters/sec -> km/h
            wrist_speed_ms = (max_wrist_vel * fps) / pixels_per_meter
            # Shuttle usually travels ~3-5x faster than the wrist at impact due to the whip effect
            fallback_kmh = (wrist_speed_ms * 3.5) * 3.6 
            return round(min(fallback_kmh, 350))
        return 0
        
    # Calculate physical speed based on the first and last tracked point
    start_x, start_y = shuttle_positions[0]
    end_x, end_y = shuttle_positions[-1]
    
    total_px_dist = math.hypot(end_x - start_x, end_y - start_y)
    total_meters_dist = total_px_dist / pixels_per_meter
    
    frames_passed = len(shuttle_positions)
    time_seconds = frames_passed / fps
    
    if time_seconds <= 0: return 0
    
    speed_m_s = total_meters_dist / time_seconds
    speed_km_h = speed_m_s * 3.6
    
    # Sanity cap: badminton smashes max out around 400-500 km/h in laboratory conditions,
    # but amateur speeds are usually 100-250 km/h.
    speed_km_h = min(speed_km_h, 350)
    
    return round(speed_km_h)
