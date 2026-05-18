"""
Athlete Vision — Shot Classifier (Fixed)
"""
import collections
import numpy as np

LEFT_SHOULDER=5; RIGHT_SHOULDER=6; LEFT_HIP=11
RIGHT_HIP=12; LEFT_WRIST=9; RIGHT_WRIST=10; NOSE=0

class ShotClassifier:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.wrist_history = collections.deque(maxlen=window_size)
        self.body_center_history = collections.deque(maxlen=window_size)
        self.body_lines_history = collections.deque(maxlen=window_size)
        self.current_shot = 'Neutral'
        self.current_handle = 'Forehand'
        self.is_swinging = False
        self._cooldown = 0
        self.COOLDOWN_FRAMES = 12
        self._swing_peak_y = None
        self._swing_frames = 0
        self._descending = False

    def reset_swing_state(self):
        self._swing_frames = 0
        self._swing_peak_y = None
        self._descending = False
        self.current_shot = 'Neutral'
        self.is_swinging = False
        self._cooldown = 0

    def update(self, keypoints, frame_h, frame_w):
        wrist_data = None
        body_center = None
        if keypoints is not None and len(keypoints) >= 17:
            kp = np.array(keypoints)
            CONF = 0.3
            lw, rw = kp[LEFT_WRIST], kp[RIGHT_WRIST]
            if rw[2]>CONF and lw[2]>CONF:
                wrist_data = {'x':rw[0]/frame_w,'y':rw[1]/frame_h,'side':'right'} \
                    if rw[1]<lw[1] else \
                    {'x':lw[0]/frame_w,'y':lw[1]/frame_h,'side':'left'}
            elif rw[2]>CONF:
                wrist_data={'x':rw[0]/frame_w,'y':rw[1]/frame_h,'side':'right'}
            elif lw[2]>CONF:
                wrist_data={'x':lw[0]/frame_w,'y':lw[1]/frame_h,'side':'left'}
            ls,rs=kp[LEFT_SHOULDER],kp[RIGHT_SHOULDER]
            if ls[2]>CONF and rs[2]>CONF:
                body_center={'x':(ls[0]+rs[0])/(2*frame_w),'y':(ls[1]+rs[1])/(2*frame_h)}
            nose_y=kp[0][1]/frame_h
            shoulder_y=(kp[5][1]+kp[6][1])/2/frame_h
            waist_y=(kp[11][1]+kp[12][1])/2/frame_h
            self.body_lines_history.append({'head':nose_y,'shoulder':shoulder_y,'waist':waist_y})
        else:
            self.body_lines_history.append(None)
        self.wrist_history.append(wrist_data)
        self.body_center_history.append(body_center)
        if self._cooldown>0:
            self._cooldown-=1

    def classify(self):
        self._classify_shot_type()
        self._classify_handle()
        return {'Shot':self.current_shot,'Handle':self.current_handle,'Is_Swinging':self.is_swinging}

    def _classify_shot_type(self):
        self.is_swinging=False
        if self._cooldown>0:
            self.current_shot='Neutral'
            return
        wrists=[w for w in self.wrist_history if w is not None]
        if len(wrists)<4:
            return
        lines_list=[b for b in self.body_lines_history if b is not None]
        if not lines_list:
            return
        lines=lines_list[-1]
        shoulder_y=lines['shoulder']; waist_y=lines['waist']; nose_y=lines['head']
        torso_height=abs(waist_y-shoulder_y)
        if torso_height<0.01: torso_height=0.01
        recent_ys=[w['y'] for w in wrists[-10:]]
        total_travel=max(recent_ys)-min(recent_ys)
        if total_travel<0.04:
            self.current_shot='Neutral'
            return

        # Use highest body points over recent window to fairly compare against peak_y
        recent_lines = [l for l in lines_list[-10:] if l is not None]
        peak_shoulder_y = min([l['shoulder'] for l in recent_lines])
        peak_waist_y = min([l['waist'] for l in recent_lines])
        peak_nose_y = min([l['head'] for l in recent_lines])
        
        head_unit = abs(peak_shoulder_y - peak_nose_y)
        HEAD_TOP = peak_nose_y - (head_unit * 0.5)
        
        # Calculate isolated full swing velocity (cancel out lunging drop)
        raw_dy3 = wrists[-1]['y']-wrists[-3]['y'] if len(wrists)>=3 else (wrists[-1]['y']-wrists[-2]['y'] if len(wrists)>=2 else 0)
        body_dy3 = lines_list[-1]['shoulder'] - lines_list[-3]['shoulder'] if len(lines_list)>=3 else 0
        dy3 = raw_dy3 - body_dy3
        
        norm_dy=dy3/torso_height
        FAST_THRESH=0.15; SWING_THRESH=0.06
        
        if norm_dy < -SWING_THRESH:
            self.is_swinging = True
            return

        if norm_dy > SWING_THRESH:
            peak_y = min([w['y'] for w in wrists[-6:]]) if len(wrists)>=6 else min([w['y'] for w in wrists])
            if peak_y<HEAD_TOP:
                shot='Smash' if norm_dy>FAST_THRESH else ('Clear' if norm_dy>SWING_THRESH else 'Neutral')
            elif peak_y<peak_shoulder_y:
                shot='Drive' if norm_dy>FAST_THRESH else ('Drop' if norm_dy>SWING_THRESH else 'Neutral')
            elif peak_y<peak_waist_y:
                shot='Drop' if norm_dy>SWING_THRESH else 'Neutral'
            else:
                shot='Net'
        else:
            if len(wrists)>=2 and wrists[-1]['y']<wrists[-2]['y'] and raw_dy3 < -0.05:
                shot='Lift'
            else:
                shot='Neutral'

        if shot!='Neutral':
            self.current_shot=shot
            self.is_swinging=True
            self._cooldown=self.COOLDOWN_FRAMES

    def _classify_handle(self):
        recent_wrists=[w for w in self.wrist_history if w is not None]
        recent_bodies=[b for b in self.body_center_history if b is not None]
        if not recent_wrists or not recent_bodies: return
        wrist_x=recent_wrists[-1]['x']
        shoulder_mid_x=recent_bodies[-1]['x']
        shoulder_mid_y=recent_bodies[-1]['y']
        dominant_side=recent_wrists[-1].get('side','right')
        is_bottom=shoulder_mid_y>0.5
        if dominant_side=='right':
            self.current_handle=('Forehand' if wrist_x>shoulder_mid_x else 'Backhand') \
                if is_bottom else ('Forehand' if wrist_x<shoulder_mid_x else 'Backhand')
        else:
            self.current_handle=('Forehand' if wrist_x<shoulder_mid_x else 'Backhand') \
                if is_bottom else ('Forehand' if wrist_x>shoulder_mid_x else 'Backhand')
