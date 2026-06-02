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
    
    # Subscription Tier ('Free' or 'Subscriber')
    subscription_tier = db.Column(db.String(50), default='Free', nullable=True)
    
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

    # Added features
    smash_speed_kmh = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<VideoRecord {self.filename} - Score: {self.performance_score}>'



