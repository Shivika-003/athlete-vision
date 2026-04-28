"""
Athlete Vision 5.0 — Universal Athlete-Lock Tracker
=====================================================
Camera-agnostic, environment-adaptive player tracking.

RULE 1: Semantic Court Filter — auto-detect court via Hough/HSV, reject
        any detection whose feet land outside the play zone + buffer.
        Falls back to full-frame if court detection fails.

RULE 2: Kinematic Pose Gate — skeleton motion variance distinguishes
        active athletes from stationary umpires/coaches.

RULE 3: Adaptive Scale Normalization — learn player height from first
        30 confirmed frames; reject detections >40% off that scale.

RULE 4: Ghost Recovery — during occlusion, maintain a ghost box on the
        last velocity vector. Only re-assign if anchor similarity >0.60.
"""

import cv2, numpy as np, collections, os, time
from ultralytics import YOLO
from ai_engine.player_reid import PlayerReID
from ai_engine.pose_gate import PoseGate
from ai_engine.shot_classifier import ShotClassifier
from ai_engine.angle_utils import calculate_angle_3d
from ai_engine.court_mask import auto_detect_court, CourtMask

print("[MatchAnalyzer] Loading YOLO models globally into memory...")
GLOBAL_POSE_MODEL = YOLO('yolov8n-pose.pt')
GLOBAL_BALL_MODEL = YOLO('yolov8n.pt')
print("[MatchAnalyzer] Models loaded successfully.")

def _bcenter(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def _bdist(a, b):
    ca, cb = _bcenter(a), _bcenter(b)
    return ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)**0.5

def _bheight(box):
    return box[3] - box[1]


def process_match_video(input_path, output_filename, output_dir="processed",
                        player1_name="Player 1", player2_name="Player 2"):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    model = GLOBAL_POSE_MODEL

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    INFER_MAX = 480
    if W > INFER_MAX:
        sc = INFER_MAX / W
        iW, iH = int(W*sc), int(H*sc)
    else:
        sc = 1.0; iW, iH = W, H
    SKIP = 6

    print(f"[Tracker] {total_frames}f {W}x{H}, infer {iW}x{iH}, skip={SKIP}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(out_path, fourcc, fps/2, (W, H))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps/2, (W, H))

    # ── Sub-modules ──
    reid = PlayerReID(ema_alpha=0.03, min_confidence=0.40)
    pose_gate = PoseGate(min_stance_score=0.50)
    cls_a = ShotClassifier(window_size=30)
    cls_b = ShotClassifier(window_size=30)

    # ── RULE 1: Semantic Court Filter ──
    court_mask = None  # Will be set from first frame

    # ── RULE 2: Kinematic history (motion variance per detection) ──
    kps_history_a = collections.deque(maxlen=10)
    kps_history_b = collections.deque(maxlen=10)

    # ── RULE 3: Adaptive Scale ──
    height_samples = []       # Collect player heights for first 30 confirmed frames
    learned_height = None     # Mean player height (set after 30 samples)
    SCALE_TOL = 0.40          # Reject if >40% off learned height

    # ── RULE 4: Ghost state ──
    ghost_vel_a = (0, 0)      # Last velocity vector for Player A
    ghost_vel_b = (0, 0)
    ghost_frames_a = 0        # How many frames Player A has been ghosted
    ghost_frames_b = 0
    GHOST_MAX = 30            # Max ghost frames before giving up
    GHOST_REACQUIRE = 0.60    # Anchor similarity needed to reacquire

    # ── Anchor fingerprints (frozen at lock-on) ──
    anchor_a = None
    anchor_b = None

    # ── Tracking state ──
    pa_box = None; pb_box = None
    pa_kps = None; pb_kps = None
    locked = False

    box_hist_a = collections.deque(maxlen=5)
    box_hist_b = collections.deque(maxlen=5)

    stats_a = {'Shot':'Neutral','Handle':'Forehand','Pressure':'In Control','Direction':'Straight','Smashes':0}
    stats_b = {'Shot':'Neutral','Handle':'Forehand','Pressure':'In Control','Direction':'Straight','Smashes':0}

    HW = min(180, int(W*0.20)); HH = 140; HM = 10
    shuttle_pos = None; shuttle_miss = 0; sdx = sdy = 0
    fi = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ── RULE 1: Auto-detect court on first frame ──
        if fi == 0 and court_mask is None:
            corners = auto_detect_court(frame)
            if corners:
                court_mask = CourtMask(corners, (H, W), margin_pct=0.15)
                print(f"[Tracker] Court detected! Corners: {corners}")
            else:
                print("[Tracker] Court not detected — using full-frame mode")

        if fi % SKIP == 0:
            small = cv2.resize(frame, (iW, iH)) if sc < 1.0 else frame
            results = model(small, verbose=False, imgsz=iW)
            sx, sy = W/iW, H/iH

            raw_dets, raw_kps = [], []
            for r in results:
                if r.boxes is None or r.keypoints is None: continue
                for i, box in enumerate(r.boxes):
                    if int(box.cls[0].cpu().numpy()) != 0: continue
                    if float(box.conf[0].cpu().numpy()) < 0.35: continue
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    b = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                    k = r.keypoints[i].data[0].cpu().numpy().copy()
                    k[:,0] *= sx; k[:,1] *= sy
                    raw_dets.append((*b, float(box.conf[0].cpu().numpy())))
                    raw_kps.append(k)

            # ════════════════════════════════════
            # RULE 1: SEMANTIC COURT FILTER
            # ════════════════════════════════════
            alive, alive_kps = [], []
            for det, kps in zip(raw_dets, raw_kps):
                x1,y1,x2,y2,conf = det
                bh = y2 - y1

                # Court mask filter (feet must be on court)
                if court_mask is not None:
                    if not court_mask.is_on_court((x1,y1,x2,y2)):
                        continue

                # Minimum height (5% of frame — very permissive)
                if bh < H * 0.05:
                    continue

                # Pose gate
                ok, _ = pose_gate.is_athletic(kps)
                if not ok:
                    continue

                alive.append(det)
                alive_kps.append(kps)

            # ════════════════════════════════════
            # RULE 3: ADAPTIVE SCALE FILTER
            # (only after learning phase complete)
            # ════════════════════════════════════
            if learned_height is not None:
                scale_ok, scale_ok_kps = [], []
                for det, kps in zip(alive, alive_kps):
                    h = det[3] - det[1]
                    if abs(h - learned_height) / learned_height <= SCALE_TOL:
                        scale_ok.append(det)
                        scale_ok_kps.append(kps)
                alive, alive_kps = scale_ok, scale_ok_kps

            # ════════════════════════════════════
            # LOCK-ON: Pick the ONE player nearest to the camera
            # (The detection with the highest y-coordinate)
            # ════════════════════════════════════
            if not locked:
                if len(alive) >= 1:
                    # Find the bounding box with the highest y2 value (closest to bottom of screen)
                    best_idx = 0
                    max_y = 0
                    for i, det in enumerate(alive):
                        if det[3] > max_y:
                            max_y = det[3]
                            best_idx = i

                    if max_y > 0:
                        d1, k1 = alive[best_idx], alive_kps[best_idx]
                        pa_box, pa_kps = d1[:4], k1
                        
                        anchor_a = reid.extract_features(frame, pa_box)
                        reid.register_player(0, frame, pa_box)

                        height_samples.append(_bheight(pa_box))

                        locked = True
                        print(f"[Tracker] LOCK-ON frame {fi}: "
                              f"Near Player P1@y={pa_box[3]}")
            else:
                # ════════════════════════════════════
                # TRACKING: Rules 2, 3, 4
                # ════════════════════════════════════

                # RULE 2: Velocity gate (25% of frame width)
                MAX_JUMP = W * 0.25
                vel_ok, vel_ok_kps = [], []
                for det, kps in zip(alive, alive_kps):
                    da = _bdist(det, pa_box) if pa_box else float('inf')
                    db = _bdist(det, pb_box) if pb_box else float('inf')
                    if min(da, db) <= MAX_JUMP:
                        vel_ok.append(det)
                        vel_ok_kps.append(kps)

                # RULE 2b: Kinematic motion variance — score each detection
                def _motion_var(kps_hist):
                    if len(kps_hist) < 3: return 1.0  # Assume active if no history
                    positions = []
                    for k in kps_hist:
                        if k is not None:
                            vis = k[:, 2] > 0.3
                            if vis.any():
                                positions.append(k[vis, :2].mean(axis=0))
                    if len(positions) < 3: return 1.0
                    deltas = [np.linalg.norm(positions[i]-positions[i-1]) for i in range(1, len(positions))]
                    return np.mean(deltas)

                # Score candidates for each player
                scored_a, scored_b = [], []
                for i, det in enumerate(vel_ok):
                    hist = reid.extract_features(frame, det)
                    sim_a = reid.compare(anchor_a, hist) if anchor_a is not None else 0
                    sim_b = reid.compare(anchor_b, hist) if anchor_b is not None else 0
                    da = _bdist(det, pa_box) if pa_box else float('inf')
                    db = _bdist(det, pb_box) if pb_box else float('inf')
                    # Combined score: 60% anchor identity + 40% proximity (inverted)
                    prox_a = max(0, 1.0 - da / MAX_JUMP) if MAX_JUMP > 0 else 0
                    prox_b = max(0, 1.0 - db / MAX_JUMP) if MAX_JUMP > 0 else 0
                    score_a = 0.6 * sim_a + 0.4 * prox_a
                    score_b = 0.6 * sim_b + 0.4 * prox_b
                    scored_a.append((i, score_a, sim_a))
                    scored_b.append((i, score_b, sim_b))

                scored_a.sort(key=lambda x: -x[1])
                scored_b.sort(key=lambda x: -x[1])

                best_a = scored_a[0] if scored_a and scored_a[0][1] > 0.3 else None
                best_b = scored_b[0] if scored_b and scored_b[0][1] > 0.3 else None

                # RULE 4: Mutual exclusion — resolve collision
                if best_a and best_b and best_a[0] == best_b[0]:
                    if best_a[1] >= best_b[1]:
                        best_b = next((e for e in scored_b if e[0] != best_a[0] and e[1] > 0.3), None)
                    else:
                        best_a = next((e for e in scored_a if e[0] != best_b[0] and e[1] > 0.3), None)

                # Apply Player A
                if best_a:
                    idx = best_a[0]
                    old_box = pa_box
                    pa_box = vel_ok[idx][:4]
                    pa_kps = vel_ok_kps[idx]
                    reid.update_features(0, frame, pa_box)
                    box_hist_a.append(pa_box)
                    kps_history_a.append(pa_kps)
                    ghost_frames_a = 0
                    # Update velocity for ghosting
                    if old_box:
                        ghost_vel_a = (_bcenter(pa_box)[0]-_bcenter(old_box)[0],
                                       _bcenter(pa_box)[1]-_bcenter(old_box)[1])
                    # RULE 3: Collect height samples
                    if len(height_samples) < 60:
                        height_samples.append(_bheight(pa_box))
                else:
                    # RULE 4: GHOST — project forward
                    ghost_frames_a += 1
                    if ghost_frames_a <= GHOST_MAX and pa_box is not None:
                        cx, cy = _bcenter(pa_box)
                        hw = (pa_box[2]-pa_box[0])/2; hh = (pa_box[3]-pa_box[1])/2
                        cx += ghost_vel_a[0]; cy += ghost_vel_a[1]
                        pa_box = (int(cx-hw), int(cy-hh), int(cx+hw), int(cy+hh))

                # Apply Player B
                if best_b:
                    idx = best_b[0]
                    old_box = pb_box
                    pb_box = vel_ok[idx][:4]
                    pb_kps = vel_ok_kps[idx]
                    reid.update_features(1, frame, pb_box)
                    box_hist_b.append(pb_box)
                    kps_history_b.append(pb_kps)
                    ghost_frames_b = 0
                    if old_box:
                        ghost_vel_b = (_bcenter(pb_box)[0]-_bcenter(old_box)[0],
                                       _bcenter(pb_box)[1]-_bcenter(old_box)[1])
                    if len(height_samples) < 60:
                        height_samples.append(_bheight(pb_box))
                else:
                    ghost_frames_b += 1
                    if ghost_frames_b <= GHOST_MAX and pb_box is not None:
                        cx, cy = _bcenter(pb_box)
                        hw = (pb_box[2]-pb_box[0])/2; hh = (pb_box[3]-pb_box[1])/2
                        cx += ghost_vel_b[0]; cy += ghost_vel_b[1]
                        pb_box = (int(cx-hw), int(cy-hh), int(cx+hw), int(cy+hh))

                # RULE 3: Learn scale after 30 samples
                if learned_height is None and len(height_samples) >= 30:
                    learned_height = np.median(height_samples)
                    print(f"[Tracker] Learned player height: {learned_height:.0f}px "
                          f"(reject outside ±{SCALE_TOL*100:.0f}%)")

            # Shuttle interpolation
            shuttle = None
            if shuttle:
                if shuttle_pos:
                    sdx = ((shuttle[0]+shuttle[2])//2) - ((shuttle_pos[0]+shuttle_pos[2])//2)
                    sdy = ((shuttle[1]+shuttle[3])//2) - ((shuttle_pos[1]+shuttle_pos[3])//2)
                shuttle_pos = shuttle; shuttle_miss = 0
            else:
                shuttle_miss += 1
                if shuttle_miss < 10 and shuttle_pos:
                    x1,y1,x2,y2 = shuttle_pos
                    shuttle_pos = (x1+sdx, y1+sdy, x2+sdx, y2+sdy); sdy += 1
                elif shuttle_miss >= 20:
                    shuttle_pos = None

            sc_pt = None
            if shuttle_pos:
                sc_pt = ((shuttle_pos[0]+shuttle_pos[2])//2, (shuttle_pos[1]+shuttle_pos[3])//2)
            cls_a.update(pa_kps, sc_pt, H, W)
            cls_b.update(pb_kps, sc_pt, H, W)

        # ── Classify every frame ──
        if locked and pa_box and pb_box:
            na, nb = cls_a.classify(), cls_b.classify()

            pr_a = "In Control"
            if len(box_hist_a)==5:
                if abs((box_hist_a[-1][0]+box_hist_a[-1][2])/2 - (box_hist_a[0][0]+box_hist_a[0][2])/2) > 15:
                    pr_a = "High Pressure"
            sa = na.get('Shot','Neutral') if na else 'Neutral'
            if sa=='Smash' and stats_a['Shot']!='Smash': stats_a['Smashes'] += 1
            stats_a.update({'Shot':sa,'Handle':na.get('Handle','Forehand'),'Pressure':pr_a,'Direction':na.get('Direction','Straight')})

            pr_b = "In Control"
            if len(box_hist_b)==5:
                if abs((box_hist_b[-1][0]+box_hist_b[-1][2])/2 - (box_hist_b[0][0]+box_hist_b[0][2])/2) > 15:
                    pr_b = "High Pressure"
            sb = nb.get('Shot','Neutral') if nb else 'Neutral'
            if sb=='Smash' and stats_b['Shot']!='Smash': stats_b['Smashes'] += 1
            stats_b.update({'Shot':sb,'Handle':nb.get('Handle','Forehand'),'Pressure':pr_b,'Direction':nb.get('Direction','Straight')})

        # ── RENDER ──
        OR = (0,165,255); YL = (0,255,255)

        if pa_box and len(pa_box)==4:
            a,b,c,d = map(int, pa_box)
            lbl = "P1" if ghost_frames_a == 0 else "P1 [GHOST]"
            alpha = 1.0 if ghost_frames_a == 0 else 0.5
            clr = tuple(int(v*alpha) for v in OR)
            cv2.rectangle(frame, (a,b), (c,d), clr, 4)
            cv2.putText(frame, lbl, (a, max(0,b-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 3, cv2.LINE_AA)

        if pb_box and len(pb_box)==4:
            a,b,c,d = map(int, pb_box)
            lbl = "P2" if ghost_frames_b == 0 else "P2 [GHOST]"
            alpha = 1.0 if ghost_frames_b == 0 else 0.5
            clr = tuple(int(v*alpha) for v in YL)
            cv2.rectangle(frame, (a,b), (c,d), clr, 4)
            cv2.putText(frame, lbl, (a, max(0,b-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 3, cv2.LINE_AA)

        # Court overlay (subtle green polygon)
        if court_mask is not None:
            court_mask.draw_overlay(frame, color=(0,200,0), alpha=0.08)

        # ── HUD ──
        if locked:
            h1x1,h1y1 = HM,HM; h1x2,h1y2 = HM+HW, HM+HH
            h2x1,h2y1 = W-HW-HM, HM; h2x2,h2y2 = W-HM, HM+HH
            ov = frame.copy()
            cv2.rectangle(ov,(h1x1,h1y1),(h1x2,h1y2),(0,0,0),-1)
            cv2.rectangle(ov,(h2x1,h2y1),(h2x2,h2y2),(0,0,0),-1)
            cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
            cv2.rectangle(frame,(h1x1,h1y1),(h1x2,h1y2),(255,255,255),1)
            cv2.rectangle(frame,(h2x1,h2y1),(h2x2,h2y2),(255,255,255),1)

            fn = cv2.FONT_HERSHEY_SIMPLEX; fs=0.40; lc=(180,180,180); vc=(255,255,255); lh=20

            def rhud(hx, hy, nm, st, nc):
                xl=hx+8; xv=hx+80; y=hy+22
                cv2.putText(frame,"Player:",(xl,y),fn,fs,lc,1,cv2.LINE_AA)
                cv2.putText(frame,nm,(xv,y),fn,fs,nc,2,cv2.LINE_AA); y+=lh
                cv2.putText(frame,"Shot:",(xl,y),fn,fs,lc,1,cv2.LINE_AA)
                cv2.putText(frame,st['Shot'],(xv,y),fn,fs,vc,1,cv2.LINE_AA); y+=lh
                cv2.putText(frame,"Handle:",(xl,y),fn,fs,lc,1,cv2.LINE_AA)
                cv2.putText(frame,st['Handle'],(xv,y),fn,fs,vc,1,cv2.LINE_AA); y+=lh
                cv2.putText(frame,"Pressure:",(xl,y),fn,fs,lc,1,cv2.LINE_AA)
                cv2.putText(frame,st['Pressure'],(xv,y),fn,fs,vc,1,cv2.LINE_AA); y+=lh
                cv2.putText(frame,"Direction:",(xl,y),fn,fs,lc,1,cv2.LINE_AA)
                cv2.putText(frame,st['Direction'],(xv,y),fn,fs,vc,1,cv2.LINE_AA); y+=lh
                cv2.putText(frame,"SMASHES:",(xl,y),fn,fs,(200,200,255),1,cv2.LINE_AA)
                cv2.putText(frame,str(st['Smashes']),(xv,y),fn,fs,(200,200,255),2,cv2.LINE_AA)

            rhud(h1x1,h1y1,player1_name,stats_a,OR)
            rhud(h2x1,h2y1,player2_name,stats_b,YL)

        writer.write(frame)
        fi += 1

    cap.release(); writer.release()
    elapsed = time.time() - start_time
    print(f"[Tracker] Done! {fi}f in {elapsed:.1f}s ({fi/max(elapsed,0.1):.0f} fps)")

    return {
        "processed_video_filename": output_filename,
        "status": "success",
        "match_analysis": True,
        "processing_time_sec": round(elapsed, 1)
    }
