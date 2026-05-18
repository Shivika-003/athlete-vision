"""
Athlete Vision 6.3 — Accurate Analysis Tracker
================================================
  1. 0.5x Time Expansion  — Output at half input FPS for slow-motion
  2. Analysis Hysteresis   — 8-frame hold + 3-frame confirm (tuned)
  3. UI Smoothing          — Alpha-0.6 panel, monospaced font, WMA-anchored
  4. Zero-Skip Guardrail   — Linear interpolation + constant Neon Green 2px
  5. Accurate Analysis     — Only classify on fresh YOLO keypoints
  + All prior systems (CSRT, Kalman, NO_SKIP, Lookahead) retained.
"""

import cv2, numpy as np, collections, os, time, math
from ultralytics import YOLO
from ai_engine.shot_classifier import ShotClassifier
from ai_engine.player_reid import PlayerReID
from ai_engine.pose_gate import PoseGate

print("[MatchAnalyzer] Loading YOLO models globally into memory...")
GLOBAL_POSE_MODEL = YOLO('yolov8n-pose.pt')
print("[MatchAnalyzer] Models loaded successfully.")

# UI Constants
NEON_GREEN = (50, 255, 50)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
RED = (60, 60, 255)
YELLOW = (0, 220, 255)

class SpeedTracker:
    def __init__(self, fps, skip, window=15):
        self.history = collections.deque(maxlen=window)
        self.last_pos = None
        self.fps = fps
        self.skip = skip
        self.time_per_update = skip / fps

    def update(self, cx, cy, box_height):
        if box_height <= 0:
            pixels_per_meter = 100
        else:
            pixels_per_meter = box_height / 1.75
            
        if self.last_pos is not None:
            dist_px = math.hypot(cx - self.last_pos[0], cy - self.last_pos[1])
            dist_m = dist_px / pixels_per_meter
            speed_ms = dist_m / self.time_per_update
            speed_kmh = speed_ms * 3.6
            self.history.append(speed_kmh)
        self.last_pos = (cx, cy)

    def get_speed(self):
        if len(self.history) < 3: return '0.0 km/h', 0.0
        avg = sum(self.history) / len(self.history)
        return f"{avg:.1f} km/h", avg

class BoxKalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def init(self, box):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        self.kalman.statePre = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized: return None
        pred = self.kalman.predict()
        return (float(pred[0]), float(pred[1]), float(pred[0]+pred[2]), float(pred[1]+pred[3]))

    def correct(self, box):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        meas = np.array([[np.float32(x)], [np.float32(y)], [np.float32(w)], [np.float32(h)]])
        self.kalman.correct(meas)
        
class LookaheadBuffer:
    def __init__(self, window=4):
        self.history = collections.deque(maxlen=window)
        
    def push(self, box):
        self.history.append(box)
        
    def get_smoothed(self):
        if not self.history: return None
        avg_box = np.mean(self.history, axis=0)
        return (float(avg_box[0]), float(avg_box[1]), float(avg_box[2]), float(avg_box[3]))

def compute_depth_score(box, H, W):
    # Score based on distance from bottom of screen (closer = lower depth score)
    y_center = (box[1] + box[3]) / 2.0
    return H - y_center

def _barea(box): return (box[2]-box[0]) * (box[3]-box[1])

def lerp_box(box_a, box_b, t):
    if box_a is None or box_b is None: return box_b
    return tuple(int(box_a[j]*(1-t) + box_b[j]*t) for j in range(4))

def unsharp_mask(img, sigma=1.0, strength=0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

def compute_stance(keypoints):
    """Analyze player stance from keypoints: Stable / Leaning / Lunging."""
    if keypoints is None or len(keypoints) < 17:
        return 'N/A'
    kp = np.array(keypoints); C = 0.3
    nose = kp[0]; lhip, rhip = kp[11], kp[12]
    lsho, rsho = kp[5], kp[6]
    if nose[2] < C or lhip[2] < C or rhip[2] < C or lsho[2] < C or rsho[2] < C:
        return 'N/A'
    mid_hip_x = (lhip[0] + rhip[0]) / 2.0
    shoulder_w = abs(lsho[0] - rsho[0]) + 1
    offset = abs(nose[0] - mid_hip_x) / shoulder_w
    
    lknee, rknee = kp[13], kp[14]
    knee_spread = 0
    if lknee[2] > C and rknee[2] > C:
        knee_spread = abs(lknee[0] - rknee[0]) / shoulder_w
        
    if knee_spread > 1.8 or offset > 0.7:
        return 'Lunging'
    elif offset > 0.35:
        return 'Leaning'
    return 'Stable'

def draw_data_panel(frame, box, shot, grip, smashes, speed_label, stance, position, W, H):
    PW, PH = 120, 130
    px1 = 20
    py1 = 20
    px2, py2 = px1 + PW, py1 + PH

    # Sleek translucent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (10, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Accent top border
    cv2.rectangle(frame, (px1, py1), (px2, py1 + 4), NEON_GREEN, -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

    # Crisp, elegant font
    fn = cv2.FONT_HERSHEY_SIMPLEX
    fs, lh = 0.35, 18
    lx, vx, y = px1 + 8, px1 + 60, py1 + 20

    def _row(label, value, val_color=WHITE):
        nonlocal y
        # Draw label
        cv2.putText(frame, label, (lx, y), fn, fs, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Right-align the value
        (vw, vh), _ = cv2.getTextSize(str(value), fn, fs, 1)
        val_x = px2 - 10 - vw
        cv2.putText(frame, str(value), (val_x, y), fn, fs, val_color, 1, cv2.LINE_AA)
        y += lh

    _row("Shot", str(shot), NEON_GREEN if shot != '---' else GREY)
    _row("Grip", str(grip), WHITE if grip != '---' else GREY)
    
    # Divider line
    cv2.line(frame, (px1 + 10, y - 6), (px2 - 10, y - 6), (80, 80, 80), 1)
    y += 4
    
    _row("Smashes", str(smashes), (200, 200, 255))
    
    # Parse speed value to determine color (km/h)
    speed_val = 0.0
    try:
        speed_val = float(speed_label.split(' ')[0])
    except:
        pass
    ic = RED if speed_val > 12.0 else (YELLOW if speed_val > 6.0 else NEON_GREEN)
    _row("Speed", str(speed_label), ic)
    
    sc = RED if stance == 'Lunging' else (YELLOW if stance == 'Leaning' else (NEON_GREEN if stance == 'Stable' else GREY))
    _row("Balance", str(stance), sc)
    
    _row("Position", str(position), WHITE if position != 'N/A' else GREY)

def draw_box_noskip(frame, box, ghost_frames, shot, grip, smashes, speed_label, stance, position, W, H):
    a,b,c,d = map(int, box)
    
    # Glowing neon green box
    cv2.rectangle(frame, (a,b), (c,d), NEON_GREEN, 2)
    
    lbl = "TRACKING LOCKED" if ghost_frames == 0 else ("REACQUIRING..." if ghost_frames <= 10 else "HOLD")
    lbl_color = NEON_GREEN if ghost_frames == 0 else YELLOW
    
    cv2.putText(frame, lbl, (a, max(0, b-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, lbl_color, 1, cv2.LINE_AA)
    draw_data_panel(frame, box, shot, grip, smashes, speed_label, stance, position, W, H)

def process_match_video(input_path, output_filename, output_dir="processed", player1_name="Player 1", player2_name="Player 2"):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    model = GLOBAL_POSE_MODEL

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure dimensions are perfectly even for H.264 web playback compatibility
    W = W if W % 2 == 0 else W - 1
    H = H if H % 2 == 0 else H - 1

    INFER_MAX = 320
    if W > INFER_MAX:
        sc = INFER_MAX / W; iW, iH = int(W*sc), int(H*sc)
    else:
        sc = 1.0; iW, iH = W, H
    SKIP = 4

    OUT_W, OUT_H = W, H
    sx_out, sy_out = 1.0, 1.0
    print(f"[Tracker] {total_frames}f {W}x{H} -> {OUT_W}x{OUT_H} (1x), infer {iW}x{iH}, skip={SKIP}")

    output_fps = fps / 2.0
    os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "video_bitrate;5000000"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, output_fps, (OUT_W, OUT_H))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, output_fps, (OUT_W, OUT_H))

    reid = PlayerReID(ema_alpha=0.03, min_confidence=0.40)
    box_kalman = BoxKalmanFilter()
    lookahead = LookaheadBuffer(window=4)
    cls_fg = ShotClassifier(window_size=30)

    # ── Action-Lock State ──
    was_swinging = False
    best_swing_shot = '---'
    best_swing_grip = '---'
    locked_shot = '---'
    locked_grip = '---'
    action_hold_timer = 0
    ACTION_HOLD_FRAMES = 20  # Hold on screen for 0.5s at 15fps to improve responsiveness

    locked = False
    fg_box = None; fg_kps = None
    ghost_frames = 0
    INTERP_MAX = 10
    last_good_box = None

    speed_tracker = SpeedTracker(fps=fps, skip=SKIP, window=5)
    smashes_count = 0
    last_speed_label = '0.0 km/h'; last_speed_val = 0.0
    display_stance = 'N/A'
    display_position = 'Mid Court'
    fresh_kps = False
    smooth_box = None
    BOX_PADDING = 8

    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        yolo_detected = False
        if fi % SKIP == 0:
            small = cv2.resize(frame, (iW, iH)) if sc < 1.0 else frame
            results = model(small, verbose=False, imgsz=iW)
            sx, sy = W/iW, H/iH

            raw_dets, raw_kps = [], []
            for r in results:
                if r.boxes is None or r.keypoints is None: continue
                for i, box in enumerate(r.boxes):
                    if int(box.cls[0]) != 0: continue
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.25: continue
                    b = box.xyxy[0].cpu().numpy()
                    b = [max(0,min(W,b[0]*sx)), max(0,min(H,b[1]*sy)),
                         max(0,min(W,b[2]*sx)), max(0,min(H,b[3]*sy))]
                    w_box, h_box = b[2]-b[0], b[3]-b[1]
                    if w_box > h_box*1.5 or h_box < 75: continue
                    k = r.keypoints[i].data[0].cpu().numpy().copy()
                    k[:,0] *= sx; k[:,1] *= sy
                    raw_dets.append((*b, conf)); raw_kps.append(k)

            scored = []
            for det, kps in zip(raw_dets, raw_kps):
                scored.append((det, kps, compute_depth_score(det, H, W), _barea(det[:4])))
            scored.sort(key=lambda x: x[2])

            if not locked:
                best_initial_det, best_initial_kps, max_y = None, None, -1
                for det, kps, depth, area in scored:
                    cx = (det[0]+det[2])/2
                    # Prevent locking onto umpires/benches on the extreme edges
                    if cx < W * 0.12 or cx > W * 0.88:
                        continue
                    
                    if det[3] > max_y:
                        max_y = det[3]
                        best_initial_det = det
                        best_initial_kps = kps
                        
                # Only lock if the lowest person is clearly in the bottom half (feet > 0.65)
                if best_initial_det is not None and max_y > H * 0.65:
                    fg_box = best_initial_det[:4]; fg_kps = best_initial_kps; fresh_kps = True
                    reid.register_player(0, frame, fg_box)
                    locked = True; ghost_frames = 0
                    box_kalman.init(fg_box)
                    last_good_box = fg_box
                    yolo_detected = True
            else:
                best_det, best_kps, best_score = None, None, -1
                for det, kps, depth, area in scored:
                    cy = (det[1]+det[3])/2
                    if cy < H*0.45: continue  # Reject anyone whose center is in the top half (opponent)
                    
                    # Distance gate: Prevent snapping to far-court player
                    if last_good_box is not None:
                        cx = (det[0]+det[2])/2
                        lcx, lcy = (last_good_box[0]+last_good_box[2])/2, (last_good_box[1]+last_good_box[3])/2
                        if math.hypot(cx - lcx, cy - lcy) > H * 0.20:
                            continue

                    sim = reid.get_confidence(0, frame, det[:4])
                    # Heavily weight vertical position (det[3]) so it always prefers the bottom player
                    score = sim*0.4 + (area/(H*W))*0.2 + (det[3]/H)*0.4
                    if score > best_score:
                        best_score = score; best_det = det; best_kps = kps

                if best_det is not None:
                    fg_box = best_det[:4]; fg_kps = best_kps; fresh_kps = True
                    reid.update_features(0, frame, fg_box)
                    yolo_detected = True; ghost_frames = 0
                    last_good_box = fg_box
                    box_kalman.correct(fg_box)

        if locked and not yolo_detected:
            ghost_frames += 1
            kalman_box = box_kalman.predict()
            if kalman_box is not None and ghost_frames <= INTERP_MAX:
                if last_good_box:
                    t = min(1.0, ghost_frames / max(1, INTERP_MAX))
                    fg_box = lerp_box(last_good_box, kalman_box, t)
                else:
                    fg_box = kalman_box
            
            # If tracker has been stuck with no YOLO detections for too long, break the lock
            if ghost_frames > 24:
                locked = False
                fg_box = None
                ghost_frames = 0
                last_good_box = None
                action_hold_timer = 0
                cls_fg = ShotClassifier(window_size=30)

        # ── ACTION-LOCK LOGIC ──
        if locked and fg_kps is not None and fresh_kps and ghost_frames == 0:
            kp_center_y = float(np.mean(fg_kps[:, 1]))
            if kp_center_y >= H * 0.45:
                cls_fg.update(fg_kps, H, W)
                result = cls_fg.classify()
    
                if result:
                    raw_shot = result.get('Shot', '---')
                    raw_grip = result.get('Handle', '---')
                    is_swinging = result.get('Is_Swinging', False)
    
                    if is_swinging:
                        # Priority hierarchy: Smash > Clear > Drive > Lift > Drop > Net
                        priorities = {'Smash': 6, 'Clear': 5, 'Drive': 4, 'Lift': 3, 'Drop': 2, 'Net': 1, 'Ready': 0, 'Neutral': 0, '---': 0}
                        curr_prio = priorities.get(raw_shot, 0)
                        best_prio = priorities.get(best_swing_shot, 0)
                        
                        if curr_prio > best_prio:
                            best_swing_shot = raw_shot
                            best_swing_grip = raw_grip
    
                    # Detect follow-through (swing just ended)
                    if was_swinging and not is_swinging:
                        # Lock it into the display!
                        locked_shot = best_swing_shot if best_swing_shot != '---' else raw_shot
                        locked_grip = best_swing_grip if best_swing_grip != '---' else raw_grip
                        action_hold_timer = ACTION_HOLD_FRAMES
                        
                        if locked_shot == 'Smash':
                            smashes_count += 1
                            
                        # Reset for next swing
                        best_swing_shot = '---'
                        best_swing_grip = '---'
    
                    was_swinging = is_swinging
                    
                # Update Stance and Position
                raw_stance = compute_stance(fg_kps)
                if raw_stance != 'N/A':
                    display_stance = raw_stance
                    
                cy = (fg_box[1] + fg_box[3]) / 2
                if cy > H * 0.75:
                    display_position = 'Back Court'
                elif cy < H * 0.55:
                    display_position = 'Front Court'
                else:
                    display_position = 'Mid Court'
                    
                fresh_kps = False

        # Evaluate display state based on lock timer
        if action_hold_timer > 0:
            display_shot = locked_shot
            display_grip = locked_grip
            action_hold_timer -= 1
        else:
            display_shot = '---'
            display_grip = '---'

        if locked and fg_box:
            box_height = fg_box[3] - fg_box[1]
            speed_tracker.update((fg_box[0]+fg_box[2])/2, (fg_box[1]+fg_box[3])/2, box_height)
            last_speed_label, last_speed_val = speed_tracker.get_speed()

        if locked and fg_box and len(fg_box) == 4:
            bw_raw = fg_box[2] - fg_box[0]
            bh_raw = fg_box[3] - fg_box[1]
            tight = [
                fg_box[0] + bw_raw * 0.05,
                fg_box[1] + bh_raw * 0.05,
                fg_box[2] - bw_raw * 0.05,
                fg_box[3] - bh_raw * 0.05,
            ]
            tw = tight[2] - tight[0]
            th = tight[3] - tight[1]
            if th > 0 and tw / th > 0.6:
                excess = tw - 0.6 * th
                tight[0] += excess / 2
                tight[2] -= excess / 2
            padded = (
                max(0, int(tight[0]) - BOX_PADDING),
                max(0, int(tight[1]) - BOX_PADDING),
                min(W, int(tight[2]) + BOX_PADDING),
                min(H, int(tight[3]) + BOX_PADDING),
            )
            lookahead.push(padded)
            smooth_box = lookahead.get_smoothed()

        frame_hd = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

        if locked and smooth_box:
            hd_box = (
                int(smooth_box[0] * sx_out),
                int(smooth_box[1] * sy_out),
                int(smooth_box[2] * sx_out),
                int(smooth_box[3] * sy_out),
            )
            
            # Crop the player for the Zoom Box BEFORE we draw UI overlays on frame_hd
            a, b, c, d = map(int, hd_box)
            a_cl, b_cl, c_cl, d_cl = max(0, a), max(0, b), min(OUT_W, c), min(OUT_H, d)
            zoom_crop = None
            ZW, ZH = 110, 165
            if c_cl > a_cl and d_cl > b_cl:
                crop = frame_hd[b_cl:d_cl, a_cl:c_cl].copy()
                zoom_crop = cv2.resize(crop, (ZW, ZH))

            draw_box_noskip(frame_hd, hd_box, ghost_frames,
                            display_shot, display_grip,
                            smashes_count, last_speed_label, display_stance, display_position, OUT_W, OUT_H)
                            
            # Render the Zoom Box on the top right
            if zoom_crop is not None:
                zx1, zy1 = OUT_W - ZW - 10, 10
                zx2, zy2 = zx1 + ZW, zy1 + ZH
                
                # Dark backing for borders
                cv2.rectangle(frame_hd, (zx1-2, zy1-2), (zx2+2, zy2+2), (15, 20, 25), -1)
                # Neon border
                cv2.rectangle(frame_hd, (zx1-1, zy1-1), (zx2+1, zy2+1), NEON_GREEN, 1)
                
                # Paste the zoom crop
                frame_hd[zy1:zy2, zx1:zx2] = zoom_crop
                
                # Top Label
                cv2.rectangle(frame_hd, (zx1-1, zy1-18), (zx1 + 75, zy1-1), NEON_GREEN, -1)
                cv2.putText(frame_hd, "PLAYER", (zx1 + 4, zy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

        writer.write(frame_hd)
        fi += 1

    cap.release(); writer.release()
    elapsed = time.time() - start_time
    print(f"[Tracker] Done! {fi}f in {elapsed:.1f}s")

    return {
        "processed_video_filename": output_filename,
        "status": "success",
        "match_analysis": True,
        "processing_time_sec": round(elapsed, 1)
    }
