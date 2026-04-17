import os
import random
import json
import threading
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from models import db, VideoRecord, User, ReferencePlayer, ReferenceShotData
from ai_engine.pose_analyzer import process_video
from ai_engine.comparison_engine import compare_user_with_reference, serialize_comparison, deserialize_comparison
from ai_engine.feedback_generator import generate_feedback, format_feedback_for_display
from ai_engine.reference_builder import seed_reference_from_json, get_active_reference_player, get_reference_summary

app = Flask(__name__)

# ─── Configuration ───
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-athlete-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['REFERENCE_FOLDER'] = 'reference_videos'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///athlete_vision.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Email Config
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your_email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your_app_password')

db.init_app(app)
mail = Mail(app)

# Setup Login Manager
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Initialize DB & Seed Reference Data ───
with app.app_context():
    db.create_all()
    
    # Create required directories
    for folder in ['uploads', 'processed', 'reference_videos']:
        os.makedirs(folder, exist_ok=True)
    
    # Auto-seed An Se-young reference data on first launch
    seed_reference_from_json("reference_data/an_seyoung.json")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════
# AUTHENTICATION ROUTES (unchanged)
# ═══════════════════════════════════════════════════════

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_by_email = User.query.filter_by(email=email).first()
        user_by_username = User.query.filter_by(username=username).first()
        
        if user_by_email or user_by_username:
            flash('An account with that Email or Username already exists. Please login or choose another.')
            return redirect(url_for('register'))
            
        new_user = User(
            username=username, email=email, 
            password_hash=generate_password_hash(password, method='scrypt')
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.')
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if not user or not check_password_hash(user.password_hash, request.form.get('password')):
            flash('Invalid login details.')
            return redirect(url_for('login'))
        login_user(user, remember=request.form.get('remember'))
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            otp = str(random.randint(100000, 999999))
            user.otp = otp
            db.session.commit()
            flash(f'An OTP has been sent to your email! (SIMULATION: Your OTP is {otp})')
            return redirect(url_for('reset_password', email=email))
        else:
            flash('Email not found.')
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    if request.method == 'POST':
        email = request.form.get('email')
        otp = request.form.get('otp')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and user.otp == otp:
            user.password_hash = generate_password_hash(password, method='scrypt')
            user.otp = None
            db.session.commit()
            flash('Password successfully reset! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP or Email.')
    return render_template('reset_password.html', email=email)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.sport = request.form.get('sport')
        current_user.play_style = request.form.get('play_style')
        current_user.current_level = request.form.get('current_level')
        current_user.racket_brand = request.form.get('racket_brand')
        current_user.training_goal = request.form.get('training_goal')
        current_user.bio = request.form.get('bio')
        db.session.commit()
        flash('Profile updated!')
        return redirect(url_for('profile'))
    return render_template('profile.html', user=current_user)


# ═══════════════════════════════════════════════════════
# BACKGROUND VIDEO PROCESSING WORKER (2.0 Enhanced)
# ═══════════════════════════════════════════════════════

def worker_process_video(record_id, app_instance, upload_path, processed_filename, processed_folder):
    """Background thread that runs the full 2.0 pipeline:
    1. Pose analysis with 3D angles
    2. Reference comparison
    3. Feedback generation
    """
    with app_instance.app_context():
        try:
            # Step 1: Analyze the video
            results = process_video(upload_path, processed_filename, processed_folder)
            
            # Step 2: Compare with reference player
            comparison = compare_user_with_reference(results)
            
            # Step 3: Generate coaching feedback
            feedback_data = generate_feedback(comparison)
            feedback_text = format_feedback_for_display(feedback_data)
            
            # Step 4: Save everything to the database
            record = VideoRecord.query.get(record_id)
            if record:
                # Backward compatible fields
                record.performance_score = results['final_score']
                record.arm_score = results['arm_score']
                record.knee_score = results['knee_score']
                record.hip_score = results['hip_score']
                record.snapshot_path = results['snapshot_filename']
                record.processed_video_path = results.get('processed_video_filename')
                record.feedback_text = feedback_text
                record.worst_timestamp = results['worst_timestamp']
                
                # 2.0 reference comparison fields
                record.shot_type = results.get('shot_type')
                record.similarity_score = comparison.get('similarity_score', 0)
                
                contact_angles = results.get('contact_angles', {})
                record.shoulder_angle = contact_angles.get('shoulder')
                record.elbow_angle = contact_angles.get('elbow')
                record.wrist_angle = contact_angles.get('wrist')
                record.knee_angle = contact_angles.get('knee')
                record.ankle_angle = contact_angles.get('ankle')
                
                # Store full comparison as JSON
                comparison['feedback'] = feedback_data
                record.comparison_details = serialize_comparison(comparison)
                
                record.status = 'completed'
                db.session.commit()
                
        except Exception as e:
            print(f"[Worker Error] {e}")
            import traceback
            traceback.print_exc()
            record = VideoRecord.query.get(record_id)
            if record:
                record.status = 'failed'
                db.session.commit()


# ═══════════════════════════════════════════════════════
# CORE APP ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    records = VideoRecord.query.filter_by(user_id=current_user.id).order_by(VideoRecord.upload_date.desc()).all()
    
    # Calculate Daily Averages for chart
    from collections import defaultdict
    daily_scores = defaultdict(list)
    for r in records:
        if r.status == 'completed' and r.performance_score:
            day_str = r.upload_date.strftime('%Y-%m-%d')
            daily_scores[day_str].append(r.performance_score)
            
    chart_labels = []
    chart_data = []
    for day in sorted(daily_scores.keys()): 
        import datetime
        formatted_day = datetime.datetime.strptime(day, '%Y-%m-%d').strftime('%b %d')
        chart_labels.append(formatted_day)
        avg = sum(daily_scores[day]) / len(daily_scores[day])
        chart_data.append(round(avg, 1))
    
    # Calculate average similarity score
    completed = [r for r in records if r.status == 'completed' and r.similarity_score]
    avg_similarity = round(sum(r.similarity_score for r in completed) / len(completed), 1) if completed else 0
    
    # Get reference player info
    ref_player = get_active_reference_player()
    
    # Training streak
    import datetime
    dates = sorted(set(r.upload_date.date() for r in records if r.status == 'completed'))
    streak = 0
    today = datetime.date.today()
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] >= today - datetime.timedelta(days=streak + 1):
            streak += 1
        else:
            break
        
    return render_template('dashboard.html', 
        records=records, 
        chart_labels=chart_labels, 
        chart_data=chart_data,
        ref_player=ref_player,
        avg_similarity=avg_similarity,
        streak=streak,
        total_sessions=len(completed)
    )

@app.route('/upload', methods=['POST'])
@login_required
def upload_video():
    if 'videoFile' not in request.files:
        flash('No file selected.')
        return redirect(url_for('dashboard'))
    
    file = request.files['videoFile']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('dashboard'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload MP4, AVI, MOV, MKV, or WebM.')
        return redirect(url_for('dashboard'))
    
    import uuid
    original_filename = secure_filename(file.filename)
    name, ext = os.path.splitext(original_filename)
    filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    new_record = VideoRecord(
        user_id=current_user.id,
        filename=filename, 
        status='processing'
    )
    db.session.add(new_record)
    db.session.commit()
    
    processed_filename = f"processed_{filename}"
    thread = threading.Thread(target=worker_process_video, args=(
        new_record.id, app, upload_path, processed_filename, app.config['PROCESSED_FOLDER']
    ))
    thread.start()
    
    flash('Your video has been uploaded and is being analyzed by the AI!')
    return redirect(url_for('dashboard'))
        
@app.route('/analysis/<int:record_id>')
@login_required
def analysis(record_id):
    record = VideoRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    # Deserialize comparison data
    comparison = deserialize_comparison(record.comparison_details)
    ref_player = get_active_reference_player()
    
    # Historical data for PDF Report
    records = VideoRecord.query.filter_by(user_id=current_user.id, status='completed').order_by(VideoRecord.upload_date.desc()).all()
    
    # Training streak
    import datetime
    from collections import defaultdict
    dates = sorted(set(r.upload_date.date() for r in records))
    streak = 0
    today = datetime.date.today()
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] >= today - datetime.timedelta(days=streak + 1):
            streak += 1
        else:
            break
            
    # Chart data
    daily_scores = defaultdict(list)
    for r in records:
        if r.performance_score:
            day_str = r.upload_date.strftime('%Y-%m-%d')
            daily_scores[day_str].append(r.performance_score)
            
    chart_labels = []
    chart_data = []
    for day in sorted(daily_scores.keys()): 
        formatted_day = datetime.datetime.strptime(day, '%Y-%m-%d').strftime('%b %d')
        chart_labels.append(formatted_day)
        avg = sum(daily_scores[day]) / len(daily_scores[day])
        chart_data.append(round(avg, 1))
        
    avg_similarity = round(sum(r.similarity_score for r in records if r.similarity_score) / len(records), 1) if len(records) > 0 else 0
    
    return render_template('analysis.html', 
        record=record, 
        comparison=comparison,
        ref_player=ref_player,
        streak=streak,
        chart_labels=chart_labels,
        chart_data=chart_data,
        avg_similarity=avg_similarity,
        user_records=records
    )

@app.route('/compare')
@login_required
def compare():
    v1_id = request.args.get('v1', type=int)
    v2_id = request.args.get('v2', type=int)
    if not v1_id or not v2_id: return redirect(url_for('dashboard'))
        
    record1 = VideoRecord.query.get_or_404(v1_id)
    record2 = VideoRecord.query.get_or_404(v2_id)
    if record1.user_id != current_user.id or record2.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    return render_template('compare.html', r1=record1, r2=record2)


# ═══════════════════════════════════════════════════════
# REFERENCE PLAYER ROUTES (New in 2.0)
# ═══════════════════════════════════════════════════════

@app.route('/reference')
@login_required
def reference():
    """Display reference player info and angle data."""
    ref_player = get_active_reference_player()
    ref_data = {}
    if ref_player:
        ref_data = get_reference_summary(ref_player.id)
    
    return render_template('reference.html', ref_player=ref_player, ref_data=ref_data)

@app.route('/reference/upload', methods=['POST'])
@login_required
def upload_reference_video():
    """Upload a new reference video for processing."""
    if 'refVideoFile' not in request.files:
        flash('No file selected.')
        return redirect(url_for('reference'))
    
    file = request.files['refVideoFile']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file. Please upload a video file.')
        return redirect(url_for('reference'))
    
    ref_player = get_active_reference_player()
    if not ref_player:
        flash('No reference player configured.')
        return redirect(url_for('reference'))
    
    import uuid
    original_filename = secure_filename(file.filename)
    name, ext = os.path.splitext(original_filename)
    filename = f"ref_{name}_{uuid.uuid4().hex[:8]}{ext}"
    ref_path = os.path.join(app.config['REFERENCE_FOLDER'], filename)
    file.save(ref_path)
    
    shot_type = request.form.get('shot_type', 'drive')
    
    # Process in background
    from ai_engine.reference_builder import process_reference_video
    try:
        count = process_reference_video(ref_path, ref_player.id, shot_type)
        flash(f'Reference video processed! {count} shot data entries added for {shot_type}.')
    except Exception as e:
        flash(f'Error processing reference video: {str(e)}')
    
    return redirect(url_for('reference'))


# ═══════════════════════════════════════════════════════
# FILE SERVING ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/processed/<filename>')
def serve_processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete_video/<int:record_id>', methods=['POST'])
@login_required
def delete_video(record_id):
    record = VideoRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    try:
        if record.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], record.filename)
            if os.path.exists(file_path): os.remove(file_path)
        if record.snapshot_path:
            # Clean up snapshot files (may contain pipe-separated names)
            for part in record.snapshot_path.split('|'):
                if part and not part.isdigit():
                    snap_path = os.path.join(app.config['PROCESSED_FOLDER'], part)
                    if os.path.exists(snap_path): os.remove(snap_path)
        if record.processed_video_path:
            for part in record.processed_video_path.split('|'):
                if part:
                    vid_path = os.path.join(app.config['PROCESSED_FOLDER'], part)
                    if os.path.exists(vid_path): os.remove(vid_path)
    except Exception:
        pass 
        
    db.session.delete(record)
    db.session.commit()
    return redirect(url_for('dashboard'))


# ═══════════════════════════════════════════════════════
# API ENDPOINTS (for AJAX calls)
# ═══════════════════════════════════════════════════════

@app.route('/api/reference_angles/<shot_type>')
@login_required
def api_reference_angles(shot_type):
    """Return reference angles as JSON for chart rendering."""
    ref_player = get_active_reference_player()
    if not ref_player:
        return json.dumps({'error': 'No reference player'}), 404
    
    from ai_engine.reference_builder import get_all_reference_angles
    angles = get_all_reference_angles(ref_player.id, shot_type)
    return json.dumps(angles)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
