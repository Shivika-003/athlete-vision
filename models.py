from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Profile extra fields
    sport = db.Column(db.String(100), nullable=True)
    bio = db.Column(db.String(500), nullable=True)
    play_style = db.Column(db.String(50), nullable=True)
    current_level = db.Column(db.String(50), nullable=True)
    racket_brand = db.Column(db.String(100), nullable=True)
    training_goal = db.Column(db.String(200), nullable=True)
    
    # Password Reset OTP
    otp = db.Column(db.String(10), nullable=True)
    
    # Relationship to videos
    videos = db.relationship('VideoRecord', backref='uploader', lazy=True)

class VideoRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) 
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Overall Metrics
    performance_score = db.Column(db.Float, nullable=True)
    arm_score = db.Column(db.Float, nullable=True)
    knee_score = db.Column(db.Float, nullable=True)
    hip_score = db.Column(db.Float, nullable=True)
    
    # Output assets
    processed_video_path = db.Column(db.String(255), nullable=True)
    snapshot_path = db.Column(db.String(255), nullable=True)
    worst_timestamp = db.Column(db.String(20), nullable=True)
    
    # Textual Advice
    feedback_text = db.Column(db.String(500), nullable=True)
    
    status = db.Column(db.String(20), default='processing')

    # ─── Athlete Vision 2.0: Reference Comparison Fields ───
    shot_type = db.Column(db.String(50), nullable=True)
    similarity_score = db.Column(db.Float, nullable=True)
    
    # User's joint angles at best contact frame
    shoulder_angle = db.Column(db.Float, nullable=True)
    elbow_angle = db.Column(db.Float, nullable=True)
    wrist_angle = db.Column(db.Float, nullable=True)
    knee_angle = db.Column(db.Float, nullable=True)
    ankle_angle = db.Column(db.Float, nullable=True)
    
    # Full comparison data (JSON string)
    comparison_details = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<VideoRecord {self.filename} - Score: {self.performance_score}>'


# ─── Athlete Vision 2.0: Reference Player Tables ───

class ReferencePlayer(db.Model):
    """Stores reference player profiles (e.g., An Se-young)."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    nationality = db.Column(db.String(100), nullable=True)
    sport = db.Column(db.String(100), default='Badminton')
    hand = db.Column(db.String(10), default='right')
    achievements = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    shots = db.relationship('ReferenceShotData', backref='player', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ReferencePlayer {self.name}>'


class ReferenceShotData(db.Model):
    """Stores biomechanical angle data for reference player shots.
    
    Each row represents one shot type + phase combination.
    e.g., An Se-young's smash at contact point has specific ideal angles.
    """
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('reference_player.id'), nullable=False)
    
    shot_type = db.Column(db.String(50), nullable=False)    # smash, clear, drive, drop, net
    phase = db.Column(db.String(50), nullable=False)         # preparation, swing, contact, follow_through
    
    # Joint angles (degrees)
    shoulder_angle = db.Column(db.Float, nullable=False)
    elbow_angle = db.Column(db.Float, nullable=False)
    wrist_angle = db.Column(db.Float, nullable=False)
    knee_angle = db.Column(db.Float, nullable=False)
    ankle_angle = db.Column(db.Float, nullable=False)
    
    # Metadata
    video_source = db.Column(db.String(255), nullable=True)
    frame_number = db.Column(db.Integer, nullable=True)
    visibility_avg = db.Column(db.Float, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    def to_angles_dict(self):
        """Convert to the standard angles dictionary format."""
        return {
            'shoulder': self.shoulder_angle,
            'elbow': self.elbow_angle,
            'wrist': self.wrist_angle,
            'knee': self.knee_angle,
            'ankle': self.ankle_angle,
        }
    
    def __repr__(self):
        return f'<RefShot {self.shot_type}/{self.phase} S:{self.shoulder_angle} E:{self.elbow_angle}>'
